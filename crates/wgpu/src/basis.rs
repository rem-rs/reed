use std::{any::TypeId, sync::Arc};

use num_traits::NumCast;
use reed_core::{
    enums::{EvalMode, QuadMode},
    error::ReedResult,
    scalar::Scalar,
    BasisTrait, ReedError,
};
use reed_cpu::basis_lagrange::LagrangeBasis;
use wgpu::util::DeviceExt;

use crate::runtime::GpuRuntime;

/// Second-stage `basis_post_main` uniforms (see `runtime.rs` WGSL).
pub(crate) fn basis_post_words(
    mode: u32,
    num_elem: usize,
    num_qpoints: usize,
    dim: usize,
    ncomp: usize,
    qcomp: usize,
    out_size: usize,
) -> [u32; 8] {
    [
        mode,
        num_elem as u32,
        num_qpoints as u32,
        dim as u32,
        ncomp as u32,
        qcomp as u32,
        out_size as u32,
        0,
    ]
}

/// `basis_post` prep (modes 1 / 3 / 5) then dense `Gradᵀ` on the packed quadrature buffer.
pub(crate) fn gpu_prep_then_grad_transpose(
    runtime: &GpuRuntime,
    grad_mat: &[f32],
    prep_mode: u32,
    prep_in: &wgpu::Buffer,
    num_elem: usize,
    num_dof: usize,
    num_qpoints: usize,
    ncomp: usize,
    dim: usize,
    qcomp: usize,
) -> ReedResult<Vec<f32>> {
    let grad_len = num_elem * num_qpoints * qcomp;
    let out_dof = num_elem * num_dof * ncomp;
    let prep_threads = num_elem * num_qpoints;

    let grad_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-prep-grad-q"),
        size: (grad_len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let post_p = runtime
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu-prep-post-u"),
            contents: bytemuck::cast_slice(&basis_post_words(
                prep_mode,
                num_elem,
                num_qpoints,
                dim,
                ncomp,
                qcomp,
                prep_threads,
            )),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let post_bind = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-prep-post-bind"),
            layout: runtime.basis_post_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prep_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: post_p.as_entire_binding(),
                },
            ],
        });

    let mat_buffer = runtime
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu-prep-grad-t-mat"),
            contents: bytemuck::cast_slice(grad_mat),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let v_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-prep-grad-t-v"),
        size: (out_dof * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let gp_buf = runtime
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu-prep-grad-t-params"),
            contents: bytemuck::cast_slice(&[
                num_elem as u32,
                num_dof as u32,
                num_qpoints as u32,
                ncomp as u32,
                out_dof as u32,
                dim as u32,
                0,
                0,
            ]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let grad_bind = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-prep-grad-t-bind"),
            layout: runtime.basis_interp_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gp_buf.as_entire_binding(),
                },
            ],
        });

    let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-prep-grad-t-rb"),
        size: (out_dof * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut v_f32 = vec![0.0_f32; out_dof];
    let mut encoder = runtime
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu-prep-grad-t-enc"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu-prep-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(runtime.basis_post_pipeline());
        pass.set_bind_group(0, &post_bind, &[]);
        pass.dispatch_workgroups((prep_threads as u32).div_ceil(64), 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu-grad-t-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(runtime.basis_grad_transpose_pipeline());
        pass.set_bind_group(0, &grad_bind, &[]);
        pass.dispatch_workgroups((out_dof as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &v_buffer,
        0,
        &readback,
        0,
        (out_dof * std::mem::size_of::<f32>()) as u64,
    );
    runtime.queue.submit(Some(encoder.finish()));
    map_readback_f32(&runtime.device, &readback, &mut v_f32)?;
    Ok(v_f32)
}

/// `v_out[e * num_qpoints + q] = weights[q]` (matches CPU `EvalMode::Weight`).
pub(crate) fn dispatch_basis_weight_tile_f32(
    runtime: &GpuRuntime,
    weights: &[f32],
    num_elem: usize,
    num_qpoints: usize,
    v_out: &mut [f32],
) -> ReedResult<()> {
    let out_size = num_elem * num_qpoints;
    if weights.len() != num_qpoints {
        return Err(ReedError::Basis(format!(
            "weight tile: weights length {} != num_qpoints {}",
            weights.len(),
            num_qpoints
        )));
    }
    if v_out.len() != out_size {
        return Err(ReedError::Basis(format!(
            "weight tile: output length {} != num_elem * num_qpoints ({})",
            v_out.len(),
            out_size
        )));
    }

    let w_buffer = runtime
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu-basis-weight-w"),
            contents: bytemuck::cast_slice(weights),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let v_init = vec![0.0_f32; out_size];
    let v_buffer = runtime
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu-basis-weight-v"),
            contents: bytemuck::cast_slice(&v_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
    let params: [u32; 4] = [num_qpoints as u32, out_size as u32, 0, 0];
    let p_buffer = runtime
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu-basis-weight-p"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bind = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-basis-weight-bind"),
            layout: runtime.basis_weight_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p_buffer.as_entire_binding(),
                },
            ],
        });

    let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-basis-weight-rb"),
        size: (out_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = runtime
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu-basis-weight-enc"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu-basis-weight-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(runtime.basis_weight_pipeline());
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((out_size as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &v_buffer,
        0,
        &readback,
        0,
        (out_size * std::mem::size_of::<f32>()) as u64,
    );
    runtime.queue.submit(Some(encoder.finish()));
    map_readback_f32(&runtime.device, &readback, v_out)?;
    Ok(())
}

pub struct WgpuBasis<T: Scalar> {
    cpu_fallback: LagrangeBasis<T>,
    runtime: Option<Arc<GpuRuntime>>,
    interp_matrix_f32: Option<Vec<f32>>,
    /// Dense `(num_qpoints * dim) × num_dof` grad operator per scalar component (same for all comps).
    grad_matrix_f32: Option<Vec<f32>>,
}

impl<T: Scalar> WgpuBasis<T> {
    pub fn new(
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: QuadMode,
        runtime: Option<Arc<GpuRuntime>>,
    ) -> ReedResult<Self> {
        let cpu_fallback = LagrangeBasis::<T>::new(dim, ncomp, p, q, qmode)?;
        let interp_matrix_f32 = if TypeId::of::<T>() == TypeId::of::<f32>() && runtime.is_some() {
            Some(build_interp_matrix_f32(dim, p, q, qmode)?)
        } else {
            None
        };
        let grad_matrix_f32 = if TypeId::of::<T>() == TypeId::of::<f32>() && runtime.is_some() {
            Some(build_grad_matrix_f32(dim, p, q, qmode)?)
        } else {
            None
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
                "interp apply size mismatch: input {}, expected {}; output {}, expected {}",
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
                label: Some("wgpu-basis-interp-mat"),
                contents: bytemuck::cast_slice(interp),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-basis-interp-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-basis-interp-v"),
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
            0, // dim unused by interp kernels
            0,
            0,
        ];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-basis-interp-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-basis-interp-bind"),
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
            label: Some("wgpu-basis-interp-readback"),
            size: (out_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-basis-interp-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-basis-interp-pass"),
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
            *dst = NumCast::from(*src).ok_or_else(|| {
                ReedError::Basis("f32->T conversion failed during readback".into())
            })?;
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
                "grad apply size mismatch: input {}, expected {}; output {}, expected {}",
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
                label: Some("wgpu-basis-grad-mat"),
                contents: bytemuck::cast_slice(grad),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-basis-grad-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-basis-grad-v"),
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
                label: Some("wgpu-basis-grad-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-basis-grad-bind"),
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
            label: Some("wgpu-basis-grad-readback"),
            size: (out_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-basis-grad-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-basis-grad-pass"),
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
            *dst = NumCast::from(*src).ok_or_else(|| {
                ReedError::Basis("f32->T conversion failed during readback".into())
            })?;
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
                "weight apply: output length {} != num_elem * num_qpoints ({})",
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
                .ok_or_else(|| ReedError::Basis("weight f32->T readback failed".into()))?;
        }
        Ok(true)
    }

    /// `EvalMode::Div` for `ncomp == dim`: Grad on GPU, then trace (`basis_post` mode 0) or prep + Gradᵀ (modes 1 + gradᵀ).
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
                    "div transpose apply size mismatch: input {}, expected {}; output {}, expected {}",
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
                    label: Some("wgpu-div-t-w"),
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
                *dst = NumCast::from(*src).ok_or_else(|| {
                    ReedError::Basis(
                        "f32->T conversion failed during div transpose readback".into(),
                    )
                })?;
            }
            return Ok(true);
        }

        // Forward: Grad u → grad, then trace → div
        let in_sz = num_elem * num_dof * ncomp;
        let div_len = num_elem * num_qpoints;
        if u.len() != in_sz || v.len() != div_len {
            return Err(ReedError::Basis(format!(
                "div forward apply size mismatch: input {}, expected {}; output {}, expected {}",
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
                label: Some("wgpu-div-f-grad-mat"),
                contents: bytemuck::cast_slice(grad),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-div-f-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let grad_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-div-f-grad"),
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
                label: Some("wgpu-div-f-grad-params"),
                contents: bytemuck::cast_slice(&grad_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let grad_bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-div-f-grad-bind"),
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
            label: Some("wgpu-div-f-out"),
            size: (div_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let post_p = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-div-f-post-params"),
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
                label: Some("wgpu-div-f-post"),
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
            label: Some("wgpu-div-f-readback"),
            size: (div_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-div-f-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-div-f-grad-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.basis_grad_pipeline());
            pass.set_bind_group(0, &grad_bind, &[]);
            let groups = (grad_out_sz as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-div-f-post-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.basis_post_pipeline());
            pass.set_bind_group(0, &post_bind, &[]);
            let groups = (div_len as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
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
            *dst = NumCast::from(*src).ok_or_else(|| {
                ReedError::Basis("f32->T conversion failed during div forward readback".into())
            })?;
        }
        Ok(true)
    }

    /// `EvalMode::Curl` for `(dim,ncomp) = (2,2)` or `(3,3)` (same H1 vector conventions as CPU `LagrangeBasis`).
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
                    "curl 2d transpose gpu: size mismatch input {} expected {}; output {} expected {}",
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
                    label: Some("wgpu-curl2d-t-w"),
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
                *dst = NumCast::from(*src).ok_or_else(|| {
                    ReedError::Basis(
                        "f32->T conversion failed during curl2d transpose readback".into(),
                    )
                })?;
            }
            return Ok(true);
        }

        let in_sz = num_elem * num_dof * ncomp;
        let curl_len = num_elem * num_qpoints;
        if u.len() != in_sz || v.len() != curl_len {
            return Err(ReedError::Basis(format!(
                "curl 2d forward gpu: size mismatch {} / {}",
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
                label: Some("wgpu-curl2d-f-mat"),
                contents: bytemuck::cast_slice(grad),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-curl2d-f-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let grad_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-curl2d-f-grad"),
            size: (grad_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gp_buf = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-curl2d-f-gp"),
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
                label: Some("wgpu-curl2d-f-gb"),
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
            label: Some("wgpu-curl2d-f-out"),
            size: (curl_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let post_p = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-curl2d-f-post"),
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
                label: Some("wgpu-curl2d-f-pb"),
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
            label: Some("wgpu-curl2d-f-rb"),
            size: (curl_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-curl2d-f-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-curl2d-f-g"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.basis_grad_pipeline());
            pass.set_bind_group(0, &grad_bind, &[]);
            pass.dispatch_workgroups((grad_out_sz as u32).div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-curl2d-f-post"),
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
                .ok_or_else(|| ReedError::Basis("f32->T curl2d forward readback".into()))?;
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
                    "curl 3d transpose gpu: size mismatch input {} expected {}; output {} expected {}",
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
                    label: Some("wgpu-curl3d-t-w"),
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
                *dst = NumCast::from(*src).ok_or_else(|| {
                    ReedError::Basis(
                        "f32->T conversion failed during curl3d transpose readback".into(),
                    )
                })?;
            }
            return Ok(true);
        }

        let in_sz = num_elem * num_dof * ncomp;
        let curl_len = num_elem * num_qpoints * 3;
        if u.len() != in_sz || v.len() != curl_len {
            return Err(ReedError::Basis(format!(
                "curl 3d forward gpu: size mismatch {} / {}",
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
                label: Some("wgpu-curl3d-f-mat"),
                contents: bytemuck::cast_slice(grad),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-curl3d-f-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let grad_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-curl3d-f-grad"),
            size: (grad_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gp_buf = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-curl3d-f-gp"),
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
                label: Some("wgpu-curl3d-f-gb"),
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
            label: Some("wgpu-curl3d-f-out"),
            size: (curl_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let post_p = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-curl3d-f-post"),
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
                label: Some("wgpu-curl3d-f-pb"),
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
            label: Some("wgpu-curl3d-f-rb"),
            size: (curl_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-curl3d-f-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-curl3d-f-g"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.basis_grad_pipeline());
            pass.set_bind_group(0, &grad_bind, &[]);
            pass.dispatch_workgroups((grad_out_sz as u32).div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-curl3d-f-post"),
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
                .ok_or_else(|| ReedError::Basis("f32->T curl3d forward readback".into()))?;
        }
        Ok(true)
    }
}

/// On WASM, wgpu::Device (inside GpuRuntime) is not Send+Sync, so the
/// BasisTrait impl is restricted to non-WASM targets only.
#[cfg(not(target_arch = "wasm32"))]
impl<T: Scalar> BasisTrait<T> for WgpuBasis<T> {
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
        // Scalar `Weight` transpose matches `Interp` transpose (CPU `LagrangeBasis`); reuse the
        // f32 interpᵀ GPU kernel when available.
        if matches!(eval_mode, EvalMode::Weight)
            && transpose
            && self.cpu_fallback.num_comp() == 1
            && self.try_apply_interp_gpu(num_elem, true, u, v)?
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

fn build_interp_matrix_f32(
    dim: usize,
    p: usize,
    q: usize,
    qmode: QuadMode,
) -> ReedResult<Vec<f32>> {
    let probe = LagrangeBasis::<f32>::new(dim, 1, p, q, qmode)?;
    let num_dof = probe.num_dof();
    let num_qpoints = probe.num_qpoints();

    let mut interp = vec![0.0_f32; num_qpoints * num_dof];
    for dof in 0..num_dof {
        let mut u = vec![0.0_f32; num_dof];
        u[dof] = 1.0;
        let mut v = vec![0.0_f32; num_qpoints];
        probe.apply(1, false, EvalMode::Interp, &u, &mut v)?;
        for qpt in 0..num_qpoints {
            interp[qpt * num_dof + dof] = v[qpt];
        }
    }
    Ok(interp)
}

fn build_grad_matrix_f32(dim: usize, p: usize, q: usize, qmode: QuadMode) -> ReedResult<Vec<f32>> {
    let probe = LagrangeBasis::<f32>::new(dim, 1, p, q, qmode)?;
    let num_dof = probe.num_dof();
    let num_qpoints = probe.num_qpoints();
    let rows = num_qpoints * dim;

    let mut mat = vec![0.0_f32; rows * num_dof];
    for dof in 0..num_dof {
        let mut u = vec![0.0_f32; num_dof];
        u[dof] = 1.0;
        let mut v = vec![0.0_f32; rows];
        probe.apply(1, false, EvalMode::Grad, &u, &mut v)?;
        for r in 0..rows {
            mat[r * num_dof + dof] = v[r];
        }
    }
    Ok(mat)
}

pub(crate) fn map_readback_f32(
    device: &wgpu::Device,
    readback: &wgpu::Buffer,
    out: &mut [f32],
) -> ReedResult<()> {
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    let map_result = rx
        .recv()
        .map_err(|e| ReedError::Basis(format!("map recv error: {e}")))?;
    map_result.map_err(|e| ReedError::Basis(format!("map error: {e:?}")))?;

    let data = slice.get_mapped_range();
    let mapped: &[f32] = bytemuck::cast_slice(&data);
    if mapped.len() != out.len() {
        return Err(ReedError::Basis("basis readback length mismatch".into()));
    }
    out.copy_from_slice(mapped);
    drop(data);
    readback.unmap();
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod wgpu_basis_tests {
    use std::sync::Arc;

    use reed_core::{BasisTrait, EvalMode, QuadMode};
    use reed_cpu::basis_lagrange::LagrangeBasis;

    use super::WgpuBasis;
    use crate::runtime::GpuRuntime;

    fn gpu_runtime_or_skip() -> Option<GpuRuntime> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))?;
        GpuRuntime::new(&adapter)
    }

    #[test]
    fn wgpu_weight_transpose_matches_interp_transpose_f32() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let b = WgpuBasis::<f32>::new(1, 1, 2, 2, QuadMode::Gauss, Some(Arc::new(rt))).unwrap();
        assert_eq!(b.num_comp(), 1);
        let ne = 2usize;
        let u: Vec<f32> = (0..ne * b.num_qpoints())
            .map(|i| 0.1 * (i + 1) as f32)
            .collect();
        let mut v_w = vec![0.0_f32; ne * b.num_dof() * b.num_comp()];
        let mut v_i = vec![0.0_f32; ne * b.num_dof() * b.num_comp()];
        b.apply(ne, true, EvalMode::Weight, &u, &mut v_w).unwrap();
        b.apply(ne, true, EvalMode::Interp, &u, &mut v_i).unwrap();
        for i in 0..v_w.len() {
            assert!(
                (v_w[i] - v_i[i]).abs() < 1e-5,
                "i={i} w={} i={}",
                v_w[i],
                v_i[i]
            );
        }
    }

    #[test]
    fn wgpu_weight_transpose_matches_cpu_lagrange_reference() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuBasis::<f32>::new(1, 1, 2, 2, QuadMode::Gauss, Some(Arc::new(rt))).unwrap();
        let cpu = LagrangeBasis::<f32>::new(1, 1, 2, 2, QuadMode::Gauss).unwrap();
        let ne = 2usize;
        let u: Vec<f32> = (0..ne * gpu.num_qpoints())
            .map(|i| 0.07 * (i as i32 - 3) as f32)
            .collect();
        let mut v_gpu = vec![0.0_f32; ne * gpu.num_dof()];
        let mut v_cpu = vec![0.0_f32; ne * cpu.num_dof()];
        gpu.apply(ne, true, EvalMode::Weight, &u, &mut v_gpu)
            .unwrap();
        cpu.apply(ne, true, EvalMode::Weight, &u, &mut v_cpu)
            .unwrap();
        for i in 0..v_gpu.len() {
            assert!(
                (v_gpu[i] - v_cpu[i]).abs() < 1e-5,
                "i={i} gpu={} cpu={}",
                v_gpu[i],
                v_cpu[i]
            );
        }
    }
}
