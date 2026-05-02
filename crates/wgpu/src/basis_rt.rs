//! WGPU wrapper for Raviart-Thomas H(div) basis.
//!
//! Stores pre-built interp and divergence matrices as `f32` and dispatches
//! matrix-vector products through GPU compute shaders when a runtime is
//! available. Falls back to the CPU basis for `f64` or WASM targets.
//!
//! ## GPU path
//!
//! RT basis evaluation uses dense matrix-vector products (typically
//! 4--20 DOFs per element). The interp and div matrices are uploaded once
//! to GPU buffers during construction, and simple compute shaders replace
//! the element-wise CPU loops at evaluation time.

use std::{
    any::TypeId,
    sync::Arc,
};

use num_traits::NumCast;
use reed_core::{
    enums::{ElemTopology, EvalMode},
    error::ReedResult,
    scalar::Scalar,
    BasisTrait, ReedError,
};
use reed_cpu::basis_rt::RaviartThomasBasis;
use wgpu::util::DeviceExt;

use crate::{basis::map_readback_f32, runtime::GpuRuntime};

/// WGPU-wrapped Raviart-Thomas H(div) basis.
///
/// Created by [`crate::WgpuBackend::create_basis_hdiv_raviart_thomas`].
pub struct WgpuRaviartThomasBasis<T: Scalar> {
    cpu: RaviartThomasBasis<T>,
    runtime: Option<Arc<GpuRuntime>>,
    /// Dense interp matrix `[(qpt*ndof+dof)*dim + d]` in `f32`, cloned from the CPU basis
    /// when `T = f32` and a runtime is available.
    interp_f32: Option<Vec<f32>>,
    /// Dense divergence matrix `[qpt*ndof + dof]` (scalar), in `f32`.
    div_f32: Option<Vec<f32>>,
}

impl<T: Scalar> WgpuRaviartThomasBasis<T> {
    /// Create a new WGPU-wrapped Raviart-Thomas basis.
    pub fn new(
        topology: ElemTopology,
        p: usize,
        q: usize,
        runtime: Option<Arc<GpuRuntime>>,
    ) -> ReedResult<Self> {
        let cpu = RaviartThomasBasis::<T>::new(topology, p, q)?;

        let interp_f32 = if TypeId::of::<T>() == TypeId::of::<f32>() && runtime.is_some() {
            let src = cpu.interp_data();
            let ptr = src.as_ptr() as *const f32;
            Some(unsafe { std::slice::from_raw_parts(ptr, src.len()) }.to_vec())
        } else {
            None
        };

        let div_f32 = if TypeId::of::<T>() == TypeId::of::<f32>() && runtime.is_some() {
            let src = cpu.div_data();
            let ptr = src.as_ptr() as *const f32;
            Some(unsafe { std::slice::from_raw_parts(ptr, src.len()) }.to_vec())
        } else {
            None
        };

        Ok(Self {
            cpu,
            runtime,
            interp_f32,
            div_f32,
        })
    }

    fn supports_f32_gpu(&self) -> bool {
        TypeId::of::<T>() == TypeId::of::<f32>()
    }

    // ── GPU interp dispatch ─────────────────────────────────────────────

    fn try_apply_interp_gpu(
        &self,
        num_elem: usize,
        transpose: bool,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<bool> {
        if !self.supports_f32_gpu() {
            return Ok(false);
        }
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        let Some(interp) = &self.interp_f32 else {
            return Ok(false);
        };

        let nd = self.cpu.num_dof();
        let nq = self.cpu.num_qpoints();
        let dim = self.cpu.dim();
        let ncomp = dim;

        let (in_size, out_size) = if transpose {
            (num_elem * nq * dim, num_elem * nd * ncomp)
        } else {
            (num_elem * nd * ncomp, num_elem * nq * dim)
        };
        if u.len() != in_size || v.len() != out_size {
            return Err(ReedError::Basis(format!(
                "rt interp gpu size mismatch: input {} (expected {}), output {} (expected {})",
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
                label: Some("wgpu-rt-interp-mat"),
                contents: bytemuck::cast_slice(interp),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-rt-interp-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-rt-interp-v"),
                contents: bytemuck::cast_slice(&v_f32),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params: [u32; 8] = [
            num_elem as u32,
            nd as u32,
            nq as u32,
            ncomp as u32,
            out_size as u32,
            dim as u32,
            0,
            0,
        ];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-rt-interp-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-rt-interp-bind"),
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
            label: Some("wgpu-rt-interp-readback"),
            size: (out_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-rt-interp-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-rt-interp-pass"),
                timestamp_writes: None,
            });
            if transpose {
                pass.set_pipeline(runtime.basis_vector_interp_transpose_pipeline());
            } else {
                pass.set_pipeline(runtime.basis_vector_interp_pipeline());
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
                ReedError::Basis("f32->T rt interp readback failed".into())
            })?;
        }
        Ok(true)
    }

    // ── GPU div dispatch ────────────────────────────────────────────────

    fn try_apply_div_gpu(
        &self,
        num_elem: usize,
        transpose: bool,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<bool> {
        if !self.supports_f32_gpu() {
            return Ok(false);
        }
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        let Some(div) = &self.div_f32 else {
            return Ok(false);
        };

        let nd = self.cpu.num_dof();
        let nq = self.cpu.num_qpoints();
        let dim = self.cpu.dim();
        let ncomp = dim;

        let (in_size, out_size) = if transpose {
            (num_elem * nq, num_elem * nd * ncomp)
        } else {
            (num_elem * nd * ncomp, num_elem * nq)
        };
        if u.len() != in_size || v.len() != out_size {
            return Err(ReedError::Basis(format!(
                "rt div gpu size mismatch: input {} (expected {}), output {} (expected {})",
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
                label: Some("wgpu-rt-div-mat"),
                contents: bytemuck::cast_slice(div),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-rt-div-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-rt-div-v"),
                contents: bytemuck::cast_slice(&v_f32),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params: [u32; 8] = [
            num_elem as u32,
            nd as u32,
            nq as u32,
            ncomp as u32,
            out_size as u32,
            dim as u32,
            0,
            0,
        ];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-rt-div-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-rt-div-bind"),
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
            label: Some("wgpu-rt-div-readback"),
            size: (out_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-rt-div-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-rt-div-pass"),
                timestamp_writes: None,
            });
            if transpose {
                pass.set_pipeline(runtime.basis_scalar_transpose_pipeline());
            } else {
                pass.set_pipeline(runtime.basis_scalar_forward_pipeline());
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
                ReedError::Basis("f32->T rt div readback failed".into())
            })?;
        }
        Ok(true)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: Scalar> BasisTrait<T> for WgpuRaviartThomasBasis<T> {
    fn dim(&self) -> usize {
        self.cpu.dim()
    }

    fn num_dof(&self) -> usize {
        self.cpu.num_dof()
    }

    fn num_qpoints(&self) -> usize {
        self.cpu.num_qpoints()
    }

    fn num_comp(&self) -> usize {
        self.cpu.num_comp()
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
        if matches!(eval_mode, EvalMode::HDiv)
            && self.try_apply_div_gpu(num_elem, transpose, u, v)?
        {
            return Ok(());
        }
        // Weight: delegate to CPU
        self.cpu.apply(num_elem, transpose, eval_mode, u, v)
    }

    fn q_weights(&self) -> &[T] {
        self.cpu.q_weights()
    }

    fn q_ref(&self) -> &[T] {
        self.cpu.q_ref()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::sync::Arc;

    use reed_core::{BasisTrait, EvalMode};
    use reed_cpu::basis_rt::RaviartThomasBasis;

    use super::WgpuRaviartThomasBasis;
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

    /// Verify WGPU-wrapped RT matches CPU-only RT for all EvalModes.
    fn check_agreement(
        gpu_basis: &WgpuRaviartThomasBasis<f32>,
        cpu_basis: &RaviartThomasBasis<f32>,
        eval_mode: EvalMode,
        nelem: usize,
    ) {
        let nd = gpu_basis.num_dof();
        let dim = gpu_basis.dim();
        let nq = gpu_basis.num_qpoints();

        let input_cases: Vec<(usize, Vec<f32>)> = match eval_mode {
            EvalMode::Interp => {
                vec![
                    (
                        nelem * nd * dim,
                        (0..nelem * nd * dim)
                            .map(|i| 0.1 * (i + 1) as f32)
                            .collect(),
                    ),
                    (
                        nelem * nq * dim,
                        (0..nelem * nq * dim)
                            .map(|i| 0.1 * (i + 1) as f32)
                            .collect(),
                    ),
                ]
            }
            EvalMode::HDiv => {
                vec![
                    (
                        nelem * nd * dim,
                        (0..nelem * nd * dim)
                            .map(|i| 0.1 * (i as i32 - 2) as f32)
                            .collect(),
                    ),
                    (
                        nelem * nq,
                        (0..nelem * nq)
                            .map(|i| 0.1 * (i as i32 - 1) as f32)
                            .collect(),
                    ),
                ]
            }
            EvalMode::Weight => {
                vec![(
                    nelem * nq,
                    (0..nelem * nq).map(|i| 0.5 * i as f32).collect(),
                )]
            }
            _ => return,
        };

        for (forward, (_in_size, ref u_data)) in input_cases.iter().enumerate() {
            for transpose in [false, true] {
                // Skip mismatched direction/size combos
                if transpose != (forward == 1) {
                    continue;
                }
                // Weight transpose delegates to Interp transpose in CPU impl
                if eval_mode == EvalMode::Weight && forward == 1 && transpose {
                    continue;
                }

                let out_size = match (eval_mode, transpose) {
                    (EvalMode::Interp, false) => nelem * nq * dim,
                    (EvalMode::Interp, true) => nelem * nd * dim,
                    (EvalMode::HDiv, false) => nelem * nq,
                    (EvalMode::HDiv, true) => nelem * nd * dim,
                    (EvalMode::Weight, false) => nelem * nq,
                    _ => continue,
                };

                let mut v_gpu = vec![0.0_f32; out_size];
                let mut v_cpu = vec![0.0_f32; out_size];

                gpu_basis
                    .apply(nelem, transpose, eval_mode, u_data, &mut v_gpu)
                    .unwrap();
                cpu_basis
                    .apply(nelem, transpose, eval_mode, u_data, &mut v_cpu)
                    .unwrap();

                for i in 0..out_size {
                    assert!(
                        (v_gpu[i] - v_cpu[i]).abs() < 1e-4,
                        "mode={:?} transpose={} forward={} i={}: gpu={} cpu={}",
                        eval_mode,
                        transpose,
                        forward,
                        i,
                        v_gpu[i],
                        v_cpu[i]
                    );
                }
            }
        }
    }

    #[test]
    fn wgpu_rt_matches_cpu_tri_rt0() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuRaviartThomasBasis::<f32>::new(
            reed_core::enums::ElemTopology::Triangle,
            0,
            3,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu =
            RaviartThomasBasis::<f32>::new(reed_core::enums::ElemTopology::Triangle, 0, 3).unwrap();

        for &mode in &[EvalMode::Interp, EvalMode::HDiv, EvalMode::Weight] {
            check_agreement(&gpu, &cpu, mode, 4);
        }
    }

    #[test]
    fn wgpu_rt_matches_cpu_tet_rt0() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuRaviartThomasBasis::<f32>::new(
            reed_core::enums::ElemTopology::Tet,
            0,
            4,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu =
            RaviartThomasBasis::<f32>::new(reed_core::enums::ElemTopology::Tet, 0, 4).unwrap();

        for &mode in &[EvalMode::Interp, EvalMode::HDiv, EvalMode::Weight] {
            check_agreement(&gpu, &cpu, mode, 4);
        }
    }

    #[test]
    fn wgpu_rt_matches_cpu_tri_rt1() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuRaviartThomasBasis::<f32>::new(
            reed_core::enums::ElemTopology::Triangle,
            1,
            6,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu =
            RaviartThomasBasis::<f32>::new(reed_core::enums::ElemTopology::Triangle, 1, 6).unwrap();

        for &mode in &[EvalMode::Interp, EvalMode::HDiv, EvalMode::Weight] {
            check_agreement(&gpu, &cpu, mode, 2);
        }
    }

    #[test]
    fn wgpu_rt_matches_cpu_tet_rt1() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuRaviartThomasBasis::<f32>::new(
            reed_core::enums::ElemTopology::Tet,
            1,
            5,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu =
            RaviartThomasBasis::<f32>::new(reed_core::enums::ElemTopology::Tet, 1, 5).unwrap();

        for &mode in &[EvalMode::Interp, EvalMode::HDiv, EvalMode::Weight] {
            check_agreement(&gpu, &cpu, mode, 3);
        }
    }

    /// Verify GPU interp and div paths individually match CPU.
    #[test]
    fn wgpu_rt_gpu_interp_matches_cpu_tri_rt0() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuRaviartThomasBasis::<f32>::new(
            reed_core::enums::ElemTopology::Triangle,
            0,
            3,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu =
            RaviartThomasBasis::<f32>::new(reed_core::enums::ElemTopology::Triangle, 0, 3).unwrap();

        check_agreement(&gpu, &cpu, EvalMode::Interp, 4);
    }

    #[test]
    fn wgpu_rt_gpu_div_matches_cpu_tri_rt0() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuRaviartThomasBasis::<f32>::new(
            reed_core::enums::ElemTopology::Triangle,
            0,
            3,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu =
            RaviartThomasBasis::<f32>::new(reed_core::enums::ElemTopology::Triangle, 0, 3).unwrap();

        check_agreement(&gpu, &cpu, EvalMode::HDiv, 4);
    }
}
