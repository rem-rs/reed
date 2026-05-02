//! WGPU wrapper for Nédélec H(curl) edge-element basis.
//!
//! Stores pre-built interp and curl matrices as `f32` and dispatches
//! matrix-vector products through GPU compute shaders when a runtime is
//! available. Falls back to the CPU basis for `f64` or WASM targets.
//!
//! ## GPU path
//!
//! Nédélec basis evaluation uses dense matrix-vector products (typically
//! 3--20 DOFs per element). The interp and curl matrices are uploaded once
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
use reed_cpu::basis_nedelec::NedelecBasis;
use wgpu::util::DeviceExt;

use crate::{basis::map_readback_f32, runtime::GpuRuntime};

/// WGPU-wrapped Nédélec H(curl) basis.
///
/// Created by [`crate::WgpuBackend::create_basis_hcurl_nedelec`].
pub struct WgpuNedelecBasis<T: Scalar> {
    cpu: NedelecBasis<T>,
    runtime: Option<Arc<GpuRuntime>>,
    /// Dense interp matrix `[(qpt*ndof+dof)*dim + d]` in `f32`, cloned from the CPU basis
    /// when `T = f32` and a runtime is available.
    interp_f32: Option<Vec<f32>>,
    /// Dense curl matrix. 2D: `[qpt*ndof + dof]`; 3D: `[(qpt*ndof+dof)*3 + d]`.
    curl_f32: Option<Vec<f32>>,
}

impl<T: Scalar> WgpuNedelecBasis<T> {
    /// Create a new WGPU-wrapped Nédélec basis.
    pub fn new(
        topology: ElemTopology,
        p: usize,
        q: usize,
        runtime: Option<Arc<GpuRuntime>>,
    ) -> ReedResult<Self> {
        let cpu = NedelecBasis::<T>::new(topology, p, q)?;

        let interp_f32 = if TypeId::of::<T>() == TypeId::of::<f32>() && runtime.is_some() {
            let src = cpu.interp_data();
            let ptr = src.as_ptr() as *const f32;
            Some(unsafe { std::slice::from_raw_parts(ptr, src.len()) }.to_vec())
        } else {
            None
        };

        let curl_f32 = if TypeId::of::<T>() == TypeId::of::<f32>() && runtime.is_some() {
            let src = cpu.curl_data();
            let ptr = src.as_ptr() as *const f32;
            Some(unsafe { std::slice::from_raw_parts(ptr, src.len()) }.to_vec())
        } else {
            None
        };

        Ok(Self {
            cpu,
            runtime,
            interp_f32,
            curl_f32,
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
        let ncomp = dim; // vector-valued basis: num_comp = dim

        let (in_size, out_size) = if transpose {
            (num_elem * nq * dim, num_elem * nd * ncomp)
        } else {
            (num_elem * nd * ncomp, num_elem * nq * dim)
        };
        if u.len() != in_size || v.len() != out_size {
            return Err(ReedError::Basis(format!(
                "nedelec interp gpu size mismatch: input {} (expected {}), output {} (expected {})",
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
                label: Some("wgpu-ned-interp-mat"),
                contents: bytemuck::cast_slice(interp),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-ned-interp-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-ned-interp-v"),
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
                label: Some("wgpu-ned-interp-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-ned-interp-bind"),
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
            label: Some("wgpu-ned-interp-readback"),
            size: (out_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-ned-interp-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-ned-interp-pass"),
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
                ReedError::Basis("f32->T nedelec interp readback failed".into())
            })?;
        }
        Ok(true)
    }

    // ── GPU curl 2D / curl 3D dispatch ──────────────────────────────────

    fn try_apply_curl_gpu(
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
        let Some(curl) = &self.curl_f32 else {
            return Ok(false);
        };

        let nd = self.cpu.num_dof();
        let nq = self.cpu.num_qpoints();
        let dim = self.cpu.dim();
        let ncomp = dim;

        let (in_size, out_size, is_curl3d) = match dim {
            2 => {
                if transpose {
                    (num_elem * nq, num_elem * nd * ncomp, false)
                } else {
                    (num_elem * nd * ncomp, num_elem * nq, false)
                }
            }
            3 => {
                if transpose {
                    (num_elem * nq * 3, num_elem * nd * ncomp, true)
                } else {
                    (num_elem * nd * ncomp, num_elem * nq * 3, true)
                }
            }
            _ => return Ok(false),
        };
        if u.len() != in_size || v.len() != out_size {
            return Err(ReedError::Basis(format!(
                "nedelec curl gpu size mismatch: input {} (expected {}), output {} (expected {})",
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
                label: Some("wgpu-ned-curl-mat"),
                contents: bytemuck::cast_slice(curl),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-ned-curl-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-ned-curl-v"),
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
                label: Some("wgpu-ned-curl-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-ned-curl-bind"),
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
            label: Some("wgpu-ned-curl-readback"),
            size: (out_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-ned-curl-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-ned-curl-pass"),
                timestamp_writes: None,
            });
            if is_curl3d {
                if transpose {
                    pass.set_pipeline(runtime.basis_curl3d_transpose_pipeline());
                } else {
                    pass.set_pipeline(runtime.basis_curl3d_pipeline());
                }
            } else {
                if transpose {
                    pass.set_pipeline(runtime.basis_scalar_transpose_pipeline());
                } else {
                    pass.set_pipeline(runtime.basis_scalar_forward_pipeline());
                }
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
                ReedError::Basis("f32->T nedelec curl readback failed".into())
            })?;
        }
        Ok(true)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: Scalar> BasisTrait<T> for WgpuNedelecBasis<T> {
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
        if matches!(eval_mode, EvalMode::HCurl)
            && self.try_apply_curl_gpu(num_elem, transpose, u, v)?
        {
            return Ok(());
        }
        // Weight: delegate to CPU (quadrature weights broadcast, not matrix-based)
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
    use reed_cpu::basis_nedelec::NedelecBasis;

    use super::WgpuNedelecBasis;
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

    /// Verify WGPU-wrapped Nédélec matches CPU-only Nédélec for all EvalModes.
    fn check_agreement(
        gpu_basis: &WgpuNedelecBasis<f32>,
        cpu_basis: &NedelecBasis<f32>,
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
            EvalMode::HCurl => {
                if dim == 2 {
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
                } else {
                    vec![
                        (
                            nelem * nd * dim,
                            (0..nelem * nd * dim)
                                .map(|i| 0.1 * (i as i32 - 3) as f32)
                                .collect(),
                        ),
                        (
                            nelem * nq * 3,
                            (0..nelem * nq * 3)
                                .map(|i| 0.1 * (i as i32 - 2) as f32)
                                .collect(),
                        ),
                    ]
                }
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
                // Weight transpose delegates to Interp transpose in CPU impl
                if eval_mode == EvalMode::Weight && forward == 1 && transpose {
                    continue;
                }
                // Weight with transpose=true has input size nq, but the CPU
                // Weight transpose uses the interp transpose path with different
                // semantics. Skip these non-matching cases.
                if eval_mode == EvalMode::Weight && transpose && forward == 0 {
                    continue;
                }
                // Non-matching combinations (wrong direction for the input size)
                if transpose != (forward == 1) {
                    continue;
                }

                let out_size = match (eval_mode, transpose) {
                    (EvalMode::Interp, false) => nelem * nq * dim,
                    (EvalMode::Interp, true) => nelem * nd * dim,
                    (EvalMode::HCurl, false) if dim == 2 => nelem * nq,
                    (EvalMode::HCurl, true) if dim == 2 => nelem * nd * dim,
                    (EvalMode::HCurl, false) => nelem * nq * 3,
                    (EvalMode::HCurl, true) => nelem * nd * dim,
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
    fn wgpu_nedelec_matches_cpu_tri_p1() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuNedelecBasis::<f32>::new(
            reed_core::enums::ElemTopology::Triangle,
            1,
            3,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu = NedelecBasis::<f32>::new(reed_core::enums::ElemTopology::Triangle, 1, 3).unwrap();

        for &mode in &[EvalMode::Interp, EvalMode::HCurl, EvalMode::Weight] {
            check_agreement(&gpu, &cpu, mode, 4);
        }
    }

    #[test]
    fn wgpu_nedelec_matches_cpu_tet_p1() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuNedelecBasis::<f32>::new(
            reed_core::enums::ElemTopology::Tet,
            1,
            4,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu = NedelecBasis::<f32>::new(reed_core::enums::ElemTopology::Tet, 1, 4).unwrap();

        for &mode in &[EvalMode::Interp, EvalMode::HCurl, EvalMode::Weight] {
            check_agreement(&gpu, &cpu, mode, 4);
        }
    }

    #[test]
    fn wgpu_nedelec_matches_cpu_tri_p2() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuNedelecBasis::<f32>::new(
            reed_core::enums::ElemTopology::Triangle,
            2,
            6,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu = NedelecBasis::<f32>::new(reed_core::enums::ElemTopology::Triangle, 2, 6).unwrap();

        for &mode in &[EvalMode::Interp, EvalMode::HCurl, EvalMode::Weight] {
            check_agreement(&gpu, &cpu, mode, 2);
        }
    }

    #[test]
    fn wgpu_nedelec_matches_cpu_tet_p2() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuNedelecBasis::<f32>::new(
            reed_core::enums::ElemTopology::Tet,
            2,
            5,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu = NedelecBasis::<f32>::new(reed_core::enums::ElemTopology::Tet, 2, 5).unwrap();

        for &mode in &[EvalMode::Interp, EvalMode::HCurl, EvalMode::Weight] {
            check_agreement(&gpu, &cpu, mode, 3);
        }
    }

    /// Verify GPU interp and curl paths individually match CPU.
    #[test]
    fn wgpu_nedelec_gpu_interp_matches_cpu_tri_p1() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuNedelecBasis::<f32>::new(
            reed_core::enums::ElemTopology::Triangle,
            1,
            3,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu = NedelecBasis::<f32>::new(reed_core::enums::ElemTopology::Triangle, 1, 3).unwrap();

        check_agreement(&gpu, &cpu, EvalMode::Interp, 4);
    }

    #[test]
    fn wgpu_nedelec_gpu_curl_matches_cpu_tri_p1() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuNedelecBasis::<f32>::new(
            reed_core::enums::ElemTopology::Triangle,
            1,
            3,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu = NedelecBasis::<f32>::new(reed_core::enums::ElemTopology::Triangle, 1, 3).unwrap();

        check_agreement(&gpu, &cpu, EvalMode::HCurl, 4);
    }

    #[test]
    fn wgpu_nedelec_gpu_interp_matches_cpu_tet_p1() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuNedelecBasis::<f32>::new(
            reed_core::enums::ElemTopology::Tet,
            1,
            4,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu = NedelecBasis::<f32>::new(reed_core::enums::ElemTopology::Tet, 1, 4).unwrap();

        check_agreement(&gpu, &cpu, EvalMode::Interp, 4);
    }

    #[test]
    fn wgpu_nedelec_gpu_curl_matches_cpu_tet_p1() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let gpu = WgpuNedelecBasis::<f32>::new(
            reed_core::enums::ElemTopology::Tet,
            1,
            4,
            Some(Arc::new(rt)),
        )
        .unwrap();
        let cpu = NedelecBasis::<f32>::new(reed_core::enums::ElemTopology::Tet, 1, 4).unwrap();

        check_agreement(&gpu, &cpu, EvalMode::HCurl, 4);
    }
}
