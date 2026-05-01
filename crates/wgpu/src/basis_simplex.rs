//! WGPU path for [`reed_cpu::basis_simplex::SimplexBasis`]. On `f32` with a runtime, `Interp`,
//! `Grad`, `Div` (`ncomp == dim`), and vector `Curl` in 2D/3D reuse the same WGSL kernels as
//! `WgpuBasis` after repacking the simplex gradient matrix. `EvalMode::Weight` tiles quadrature
//! weights on the GPU like `WgpuBasis`.

use std::{any::TypeId, sync::Arc};

use num_traits::NumCast;
use reed_core::{
    enums::{ElemTopology, EvalMode},
    error::ReedResult,
    scalar::Scalar,
    BasisTrait, ReedError,
};
use reed_cpu::basis_simplex::SimplexBasis;
use wgpu::util::DeviceExt;

use crate::{
    basis::{
        basis_post_words, dispatch_basis_weight_tile_f32, gpu_prep_then_grad_transpose,
        map_readback_f32,
    },
    runtime::GpuRuntime,
};

/// `SimplexBasis` stores `grad[(qpt * num_dof + dof) * dim + d]`. WGSL `basis_grad_*` kernels expect
/// the same dense layout as tensor `LagrangeBasis`: row `(qpt * dim + d)` has length `num_dof`.
fn repack_simplex_grad_for_wgpu(
    grad: &[f32],
    num_qpoints: usize,
    num_dof: usize,
    dim: usize,
) -> ReedResult<Vec<f32>> {
    let n = num_qpoints * dim * num_dof;
    if grad.len() != num_qpoints * num_dof * dim {
        return Err(ReedError::Basis(format!(
            "simplex grad length {} != {}*{}*{}",
            grad.len(),
            num_qpoints,
            num_dof,
            dim
        )));
    }
    let mut out = vec![0.0_f32; n];
    for qpt in 0..num_qpoints {
        for dof in 0..num_dof {
            for d in 0..dim {
                let src = (qpt * num_dof + dof) * dim + d;
                let dst = (qpt * dim + d) * num_dof + dof;
                out[dst] = grad[src];
            }
        }
    }
    Ok(out)
}

pub struct WgpuSimplexBasis<T: Scalar> {
    cpu_fallback: SimplexBasis<T>,
    runtime: Option<Arc<GpuRuntime>>,
    interp_matrix_f32: Option<Vec<f32>>,
    grad_matrix_f32: Option<Vec<f32>>,
}

impl<T: Scalar> WgpuSimplexBasis<T> {
    pub fn new(
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
        runtime: Option<Arc<GpuRuntime>>,
    ) -> ReedResult<Self> {
        let cpu_fallback = SimplexBasis::<T>::new(topo, poly, ncomp, q)?;
        let (interp_matrix_f32, grad_matrix_f32) =
            if TypeId::of::<T>() == TypeId::of::<f32>() && runtime.is_some() {
                let interp = cpu_fallback
                    .interp_matrix()
                    .iter()
                    .map(|x| NumCast::from(*x))
                    .collect::<Option<Vec<f32>>>()
                    .ok_or_else(|| ReedError::Basis("simplex interp f32 cast failed".into()))?;
                let grad_raw = cpu_fallback
                    .grad_matrix()
                    .iter()
                    .map(|x| NumCast::from(*x))
                    .collect::<Option<Vec<f32>>>()
                    .ok_or_else(|| ReedError::Basis("simplex grad f32 cast failed".into()))?;
                let grad = repack_simplex_grad_for_wgpu(
                    &grad_raw,
                    cpu_fallback.num_qpoints(),
                    cpu_fallback.num_dof(),
                    cpu_fallback.dim(),
                )?;
                (Some(interp), Some(grad))
            } else {
                (None, None)
            };
        Ok(Self {
            cpu_fallback,
            runtime,
            interp_matrix_f32,
            grad_matrix_f32,
        })
    }

    fn supports_f32_gpu() -> bool {
        TypeId::of::<T>() == TypeId::of::<f32>()
    }

    fn try_apply_interp_gpu(
        &self,
        num_elem: usize,
        transpose: bool,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<bool> {
        if !Self::supports_f32_gpu() {
            return Ok(false);
        }
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        let Some(interp) = &self.interp_matrix_f32 else {
            return Ok(false);
        };

        let num_dof = self.cpu_fallback.num_dof();
        let num_qpoints = self.cpu_fallback.num_qpoints();
        let ncomp = self.cpu_fallback.num_comp();
        let in_size = if transpose {
            num_elem * num_qpoints * ncomp
        } else {
            num_elem * num_dof * ncomp
        };
        let out_size = if transpose {
            num_elem * num_dof * ncomp
        } else {
            num_elem * num_qpoints * ncomp
        };
        if u.len() != in_size || v.len() != out_size {
            return Err(ReedError::Basis(format!(
                "simplex interp apply size mismatch: input {}, expected {}; output {}, expected {}",
                u.len(),
                in_size,
                v.len(),
                out_size
            )));
        }

        let Some(u_f32) = u
            .iter()
            .map(|x| NumCast::from(*x))
            .collect::<Option<Vec<f32>>>()
        else {
            return Ok(false);
        };
        let mut v_f32 = vec![0.0_f32; out_size];

        let mat_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-interp-mat"),
                contents: bytemuck::cast_slice(interp),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-interp-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-interp-v"),
                contents: bytemuck::cast_slice(&v_f32),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params: [u32; 8] = [
            num_elem as u32,
            num_dof as u32,
            num_qpoints as u32,
            ncomp as u32,
            out_size as u32,
            0,
            0,
            0,
        ];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-interp-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-simplex-interp-bind"),
                layout: runtime.basis_interp_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: mat_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buffer.as_entire_binding(),
                    },
                ],
            });

        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-simplex-interp-readback"),
            size: (out_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-simplex-interp-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-simplex-interp-pass"),
                timestamp_writes: None,
            });
            if transpose {
                pass.set_pipeline(runtime.basis_interp_transpose_pipeline());
            } else {
                pass.set_pipeline(runtime.basis_interp_pipeline());
            }
            pass.set_bind_group(0, &bind, &[]);
            let groups = (out_size as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &v_buffer,
            0,
            &readback,
            0,
            (out_size * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        map_readback_f32(&runtime.device, &readback, &mut v_f32)?;
        for (dst, src) in v.iter_mut().zip(v_f32.iter()) {
            *dst = NumCast::from(*src)
                .ok_or_else(|| ReedError::Basis("simplex interp f32->T readback failed".into()))?;
        }
        Ok(true)
    }

    fn try_apply_grad_gpu(
        &self,
        num_elem: usize,
        transpose: bool,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<bool> {
        if !Self::supports_f32_gpu() {
            return Ok(false);
        }
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        let Some(grad) = &self.grad_matrix_f32 else {
            return Ok(false);
        };

        let dim = self.cpu_fallback.dim();
        let num_dof = self.cpu_fallback.num_dof();
        let num_qpoints = self.cpu_fallback.num_qpoints();
        let ncomp = self.cpu_fallback.num_comp();
        let in_size = if transpose {
            num_elem * num_qpoints * dim * ncomp
        } else {
            num_elem * num_dof * ncomp
        };
        let out_size = if transpose {
            num_elem * num_dof * ncomp
        } else {
            num_elem * num_qpoints * dim * ncomp
        };
        if u.len() != in_size || v.len() != out_size {
            return Err(ReedError::Basis(format!(
                "simplex grad apply size mismatch: input {}, expected {}; output {}, expected {}",
                u.len(),
                in_size,
                v.len(),
                out_size
            )));
        }

        let Some(u_f32) = u
            .iter()
            .map(|x| NumCast::from(*x))
            .collect::<Option<Vec<f32>>>()
        else {
            return Ok(false);
        };
        let mut v_f32 = vec![0.0_f32; out_size];

        let mat_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-grad-mat"),
                contents: bytemuck::cast_slice(grad),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-grad-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-grad-v"),
                contents: bytemuck::cast_slice(&v_f32),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params: [u32; 8] = [
            num_elem as u32,
            num_dof as u32,
            num_qpoints as u32,
            ncomp as u32,
            out_size as u32,
            dim as u32,
            0,
            0,
        ];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-grad-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-simplex-grad-bind"),
                layout: runtime.basis_interp_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: mat_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buffer.as_entire_binding(),
                    },
                ],
            });

        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-simplex-grad-readback"),
            size: (out_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-simplex-grad-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-simplex-grad-pass"),
                timestamp_writes: None,
            });
            if transpose {
                pass.set_pipeline(runtime.basis_grad_transpose_pipeline());
            } else {
                pass.set_pipeline(runtime.basis_grad_pipeline());
            }
            pass.set_bind_group(0, &bind, &[]);
            let groups = (out_size as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &v_buffer,
            0,
            &readback,
            0,
            (out_size * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        map_readback_f32(&runtime.device, &readback, &mut v_f32)?;
        for (dst, src) in v.iter_mut().zip(v_f32.iter()) {
            *dst = NumCast::from(*src)
                .ok_or_else(|| ReedError::Basis("simplex grad f32->T readback failed".into()))?;
        }
        Ok(true)
    }

    fn try_apply_weight_gpu(
        &self,
        num_elem: usize,
        transpose: bool,
        _u: &[T],
        v: &mut [T],
    ) -> ReedResult<bool> {
        if transpose {
            return Ok(false);
        }
        if !Self::supports_f32_gpu() {
            return Ok(false);
        }
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        let num_qpoints = self.cpu_fallback.num_qpoints();
        let out_size = num_elem * num_qpoints;
        if v.len() != out_size {
            return Err(ReedError::Basis(format!(
                "simplex weight: output length {} != {}",
                v.len(),
                out_size
            )));
        }
        let Some(weights_f32) = self
            .cpu_fallback
            .q_weights()
            .iter()
            .map(|x| NumCast::from(*x))
            .collect::<Option<Vec<f32>>>()
        else {
            return Ok(false);
        };
        let mut v_f32 = vec![0.0_f32; out_size];
        dispatch_basis_weight_tile_f32(runtime, &weights_f32, num_elem, num_qpoints, &mut v_f32)?;
        for (dst, src) in v.iter_mut().zip(v_f32.iter()) {
            *dst = NumCast::from(*src)
                .ok_or_else(|| ReedError::Basis("simplex weight f32->T readback failed".into()))?;
        }
        Ok(true)
    }

    fn try_apply_div_gpu(
        &self,
        num_elem: usize,
        transpose: bool,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<bool> {
        if !Self::supports_f32_gpu() {
            return Ok(false);
        }
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        let Some(grad) = &self.grad_matrix_f32 else {
            return Ok(false);
        };
        let dim = self.cpu_fallback.dim();
        let ncomp = self.cpu_fallback.num_comp();
        if ncomp != dim {
            return Ok(false);
        }
        let num_dof = self.cpu_fallback.num_dof();
        let num_qpoints = self.cpu_fallback.num_qpoints();
        let qcomp = ncomp * dim;

        if transpose {
            let in_size = num_elem * num_qpoints;
            let out_size = num_elem * num_dof * ncomp;
            if u.len() != in_size || v.len() != out_size {
                return Err(ReedError::Basis(format!(
                    "simplex div transpose: input {}, expected {}; output {}, expected {}",
                    u.len(),
                    in_size,
                    v.len(),
                    out_size
                )));
            }
            let Some(u_f32) = u
                .iter()
                .map(|x| NumCast::from(*x))
                .collect::<Option<Vec<f32>>>()
            else {
                return Ok(false);
            };
            let w_buffer = runtime
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("wgpu-simplex-div-t-w"),
                    contents: bytemuck::cast_slice(&u_f32),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let v_f32 = gpu_prep_then_grad_transpose(
                runtime,
                grad,
                1,
                &w_buffer,
                num_elem,
                num_dof,
                num_qpoints,
                ncomp,
                dim,
                qcomp,
            )?;
            for (dst, src) in v.iter_mut().zip(v_f32.iter()) {
                *dst = NumCast::from(*src)
                    .ok_or_else(|| ReedError::Basis("simplex div transpose readback".into()))?;
            }
            return Ok(true);
        }

        let in_sz = num_elem * num_dof * ncomp;
        let div_len = num_elem * num_qpoints;
        if u.len() != in_sz || v.len() != div_len {
            return Err(ReedError::Basis(format!(
                "simplex div forward: input {}, expected {}; output {}, expected {}",
                u.len(),
                in_sz,
                v.len(),
                div_len
            )));
        }
        let Some(u_f32) = u
            .iter()
            .map(|x| NumCast::from(*x))
            .collect::<Option<Vec<f32>>>()
        else {
            return Ok(false);
        };
        let grad_len = num_elem * num_qpoints * qcomp;
        let grad_out_sz = grad_len;

        let mat_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-div-f-mat"),
                contents: bytemuck::cast_slice(grad),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-div-f-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let grad_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-simplex-div-f-grad"),
            size: (grad_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grad_params: [u32; 8] = [
            num_elem as u32,
            num_dof as u32,
            num_qpoints as u32,
            ncomp as u32,
            grad_out_sz as u32,
            dim as u32,
            0,
            0,
        ];
        let gp_buf = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-div-f-gp"),
                contents: bytemuck::cast_slice(&grad_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let grad_bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-simplex-div-f-gb"),
                layout: runtime.basis_interp_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: mat_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grad_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: gp_buf.as_entire_binding(),
                    },
                ],
            });

        let div_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-simplex-div-f-out"),
            size: (div_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let post_p = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-div-f-post"),
                contents: bytemuck::cast_slice(&basis_post_words(
                    0,
                    num_elem,
                    num_qpoints,
                    dim,
                    ncomp,
                    qcomp,
                    div_len,
                )),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let post_bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-simplex-div-f-pb"),
                layout: runtime.basis_post_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: grad_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: div_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: post_p.as_entire_binding(),
                    },
                ],
            });

        let mut v_f32 = vec![0.0_f32; div_len];
        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-simplex-div-f-rb"),
            size: (div_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-simplex-div-f-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-simplex-div-f-g"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.basis_grad_pipeline());
            pass.set_bind_group(0, &grad_bind, &[]);
            pass.dispatch_workgroups((grad_out_sz as u32).div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-simplex-div-f-post"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.basis_post_pipeline());
            pass.set_bind_group(0, &post_bind, &[]);
            pass.dispatch_workgroups((div_len as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &div_buffer,
            0,
            &readback,
            0,
            (div_len * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        map_readback_f32(&runtime.device, &readback, &mut v_f32)?;
        for (dst, src) in v.iter_mut().zip(v_f32.iter()) {
            *dst = NumCast::from(*src)
                .ok_or_else(|| ReedError::Basis("simplex div forward readback".into()))?;
        }
        Ok(true)
    }

    fn try_apply_curl_gpu(
        &self,
        num_elem: usize,
        transpose: bool,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<bool> {
        let dim = self.cpu_fallback.dim();
        let ncomp = self.cpu_fallback.num_comp();
        match (dim, ncomp) {
            (2, 2) => self.try_apply_curl2d_gpu(num_elem, transpose, u, v),
            (3, 3) => self.try_apply_curl3d_gpu(num_elem, transpose, u, v),
            _ => Ok(false),
        }
    }

    fn try_apply_curl2d_gpu(
        &self,
        num_elem: usize,
        transpose: bool,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<bool> {
        if !Self::supports_f32_gpu() {
            return Ok(false);
        }
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        let Some(grad) = &self.grad_matrix_f32 else {
            return Ok(false);
        };
        let dim = 2usize;
        let ncomp = 2usize;
        let qcomp = 4usize;
        let num_dof = self.cpu_fallback.num_dof();
        let num_qpoints = self.cpu_fallback.num_qpoints();

        if transpose {
            let in_size = num_elem * num_qpoints;
            let out_size = num_elem * num_dof * ncomp;
            if u.len() != in_size || v.len() != out_size {
                return Err(ReedError::Basis(format!(
                    "simplex curl2d transpose: input {} expected {}; output {} expected {}",
                    u.len(),
                    in_size,
                    v.len(),
                    out_size
                )));
            }
            let Some(u_f32) = u
                .iter()
                .map(|x| NumCast::from(*x))
                .collect::<Option<Vec<f32>>>()
            else {
                return Ok(false);
            };
            let w_buffer = runtime
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("wgpu-simplex-curl2d-t-w"),
                    contents: bytemuck::cast_slice(&u_f32),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let v_f32 = gpu_prep_then_grad_transpose(
                runtime,
                grad,
                3,
                &w_buffer,
                num_elem,
                num_dof,
                num_qpoints,
                ncomp,
                dim,
                qcomp,
            )?;
            for (dst, src) in v.iter_mut().zip(v_f32.iter()) {
                *dst = NumCast::from(*src)
                    .ok_or_else(|| ReedError::Basis("simplex curl2d transpose readback".into()))?;
            }
            return Ok(true);
        }

        let in_sz = num_elem * num_dof * ncomp;
        let curl_len = num_elem * num_qpoints;
        if u.len() != in_sz || v.len() != curl_len {
            return Err(ReedError::Basis(format!(
                "simplex curl2d forward: size {} / {}",
                u.len(),
                v.len()
            )));
        }
        let Some(u_f32) = u
            .iter()
            .map(|x| NumCast::from(*x))
            .collect::<Option<Vec<f32>>>()
        else {
            return Ok(false);
        };
        let grad_len = num_elem * num_qpoints * qcomp;
        let grad_out_sz = grad_len;

        let mat_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-curl2d-f-mat"),
                contents: bytemuck::cast_slice(grad),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-curl2d-f-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let grad_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-simplex-curl2d-f-grad"),
            size: (grad_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gp_buf = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-curl2d-f-gp"),
                contents: bytemuck::cast_slice(&[
                    num_elem as u32,
                    num_dof as u32,
                    num_qpoints as u32,
                    ncomp as u32,
                    grad_out_sz as u32,
                    dim as u32,
                    0,
                    0,
                ]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let grad_bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-simplex-curl2d-f-gb"),
                layout: runtime.basis_interp_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: mat_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grad_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: gp_buf.as_entire_binding(),
                    },
                ],
            });
        let curl_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-simplex-curl2d-f-out"),
            size: (curl_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let post_p = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-curl2d-f-post"),
                contents: bytemuck::cast_slice(&basis_post_words(
                    2,
                    num_elem,
                    num_qpoints,
                    dim,
                    ncomp,
                    qcomp,
                    curl_len,
                )),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let post_bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-simplex-curl2d-f-pb"),
                layout: runtime.basis_post_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: grad_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: curl_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: post_p.as_entire_binding(),
                    },
                ],
            });
        let mut v_f32 = vec![0.0_f32; curl_len];
        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-simplex-curl2d-f-rb"),
            size: (curl_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-simplex-curl2d-f-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-simplex-curl2d-f-g"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.basis_grad_pipeline());
            pass.set_bind_group(0, &grad_bind, &[]);
            pass.dispatch_workgroups((grad_out_sz as u32).div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-simplex-curl2d-f-post"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.basis_post_pipeline());
            pass.set_bind_group(0, &post_bind, &[]);
            pass.dispatch_workgroups((curl_len as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &curl_buffer,
            0,
            &readback,
            0,
            (curl_len * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));
        map_readback_f32(&runtime.device, &readback, &mut v_f32)?;
        for (dst, src) in v.iter_mut().zip(v_f32.iter()) {
            *dst = NumCast::from(*src)
                .ok_or_else(|| ReedError::Basis("simplex curl2d forward readback".into()))?;
        }
        Ok(true)
    }

    fn try_apply_curl3d_gpu(
        &self,
        num_elem: usize,
        transpose: bool,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<bool> {
        if !Self::supports_f32_gpu() {
            return Ok(false);
        }
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        let Some(grad) = &self.grad_matrix_f32 else {
            return Ok(false);
        };
        let dim = 3usize;
        let ncomp = 3usize;
        let qcomp = 9usize;
        let num_dof = self.cpu_fallback.num_dof();
        let num_qpoints = self.cpu_fallback.num_qpoints();

        if transpose {
            let in_size = num_elem * num_qpoints * 3;
            let out_size = num_elem * num_dof * ncomp;
            if u.len() != in_size || v.len() != out_size {
                return Err(ReedError::Basis(format!(
                    "simplex curl3d transpose: input {} expected {}; output {} expected {}",
                    u.len(),
                    in_size,
                    v.len(),
                    out_size
                )));
            }
            let Some(u_f32) = u
                .iter()
                .map(|x| NumCast::from(*x))
                .collect::<Option<Vec<f32>>>()
            else {
                return Ok(false);
            };
            let w_buffer = runtime
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("wgpu-simplex-curl3d-t-w"),
                    contents: bytemuck::cast_slice(&u_f32),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let v_f32 = gpu_prep_then_grad_transpose(
                runtime,
                grad,
                5,
                &w_buffer,
                num_elem,
                num_dof,
                num_qpoints,
                ncomp,
                dim,
                qcomp,
            )?;
            for (dst, src) in v.iter_mut().zip(v_f32.iter()) {
                *dst = NumCast::from(*src)
                    .ok_or_else(|| ReedError::Basis("simplex curl3d transpose readback".into()))?;
            }
            return Ok(true);
        }

        let in_sz = num_elem * num_dof * ncomp;
        let curl_len = num_elem * num_qpoints * 3;
        if u.len() != in_sz || v.len() != curl_len {
            return Err(ReedError::Basis(format!(
                "simplex curl3d forward: size {} / {}",
                u.len(),
                v.len()
            )));
        }
        let Some(u_f32) = u
            .iter()
            .map(|x| NumCast::from(*x))
            .collect::<Option<Vec<f32>>>()
        else {
            return Ok(false);
        };
        let grad_len = num_elem * num_qpoints * qcomp;
        let grad_out_sz = grad_len;

        let mat_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-curl3d-f-mat"),
                contents: bytemuck::cast_slice(grad),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-curl3d-f-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let grad_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-simplex-curl3d-f-grad"),
            size: (grad_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gp_buf = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-curl3d-f-gp"),
                contents: bytemuck::cast_slice(&[
                    num_elem as u32,
                    num_dof as u32,
                    num_qpoints as u32,
                    ncomp as u32,
                    grad_out_sz as u32,
                    dim as u32,
                    0,
                    0,
                ]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let grad_bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-simplex-curl3d-f-gb"),
                layout: runtime.basis_interp_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: mat_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grad_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: gp_buf.as_entire_binding(),
                    },
                ],
            });
        let curl_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-simplex-curl3d-f-out"),
            size: (curl_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let post_p = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-simplex-curl3d-f-post"),
                contents: bytemuck::cast_slice(&basis_post_words(
                    4,
                    num_elem,
                    num_qpoints,
                    dim,
                    ncomp,
                    qcomp,
                    curl_len,
                )),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let post_bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-simplex-curl3d-f-pb"),
                layout: runtime.basis_post_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: grad_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: curl_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: post_p.as_entire_binding(),
                    },
                ],
            });
        let mut v_f32 = vec![0.0_f32; curl_len];
        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-simplex-curl3d-f-rb"),
            size: (curl_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-simplex-curl3d-f-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-simplex-curl3d-f-g"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.basis_grad_pipeline());
            pass.set_bind_group(0, &grad_bind, &[]);
            pass.dispatch_workgroups((grad_out_sz as u32).div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-simplex-curl3d-f-post"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.basis_post_pipeline());
            pass.set_bind_group(0, &post_bind, &[]);
            pass.dispatch_workgroups((curl_len as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &curl_buffer,
            0,
            &readback,
            0,
            (curl_len * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));
        map_readback_f32(&runtime.device, &readback, &mut v_f32)?;
        for (dst, src) in v.iter_mut().zip(v_f32.iter()) {
            *dst = NumCast::from(*src)
                .ok_or_else(|| ReedError::Basis("simplex curl3d forward readback".into()))?;
        }
        Ok(true)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: Scalar> BasisTrait<T> for WgpuSimplexBasis<T> {
    fn dim(&self) -> usize {
        self.cpu_fallback.dim()
    }

    fn num_dof(&self) -> usize {
        self.cpu_fallback.num_dof()
    }

    fn num_qpoints(&self) -> usize {
        self.cpu_fallback.num_qpoints()
    }

    fn num_comp(&self) -> usize {
        self.cpu_fallback.num_comp()
    }

    fn apply(
        &self,
        num_elem: usize,
        transpose: bool,
        eval_mode: EvalMode,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<()> {
        if matches!(eval_mode, EvalMode::Interp)
            && self.try_apply_interp_gpu(num_elem, transpose, u, v)?
        {
            return Ok(());
        }
        if matches!(eval_mode, EvalMode::Grad)
            && self.try_apply_grad_gpu(num_elem, transpose, u, v)?
        {
            return Ok(());
        }
        if matches!(eval_mode, EvalMode::Div)
            && self.try_apply_div_gpu(num_elem, transpose, u, v)?
        {
            return Ok(());
        }
        if matches!(eval_mode, EvalMode::Curl)
            && self.try_apply_curl_gpu(num_elem, transpose, u, v)?
        {
            return Ok(());
        }
        if matches!(eval_mode, EvalMode::Weight)
            && self.try_apply_weight_gpu(num_elem, transpose, u, v)?
        {
            return Ok(());
        }
        self.cpu_fallback
            .apply(num_elem, transpose, eval_mode, u, v)
    }

    fn q_weights(&self) -> &[T] {
        self.cpu_fallback.q_weights()
    }

    fn q_ref(&self) -> &[T] {
        self.cpu_fallback.q_ref()
    }
}
