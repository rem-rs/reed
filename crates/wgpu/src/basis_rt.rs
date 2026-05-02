//! WGPU wrapper for Raviart-Thomas H(div) basis.
//!
//! Currently delegates all evaluation to the CPU basis. The wrapped basis
//! stores a [`GpuRuntime`] handle for future GPU-accelerated matrix-vector
//! product kernels.
//!
//! ## Future GPU path
//!
//! RT basis evaluation uses dense matrix-vector products (typically
//! 4--20 DOFs per element). When GPU kernels are added, the interp and
//! div matrices (both `f32`) will be uploaded once to GPU buffers, and
//! simple compute shaders will replace the element-wise CPU loops.

use std::sync::Arc;

use reed_core::{
    enums::{ElemTopology, EvalMode},
    error::ReedResult,
    scalar::Scalar,
    BasisTrait,
};
use reed_cpu::basis_rt::RaviartThomasBasis;

use crate::runtime::GpuRuntime;

/// WGPU-wrapped Raviart-Thomas H(div) basis.
///
/// Created by [`crate::WgpuBackend::create_basis_hdiv_raviart_thomas`].
pub struct WgpuRaviartThomasBasis<T: Scalar> {
    cpu: RaviartThomasBasis<T>,
    #[allow(dead_code)]
    runtime: Option<Arc<GpuRuntime>>,
}

impl<T: Scalar> WgpuRaviartThomasBasis<T> {
    /// Create a new WGPU-wrapped Raviart-Thomas basis.
    ///
    /// The `runtime` parameter is stored for future GPU-accelerated evaluation.
    pub fn new(
        topology: ElemTopology,
        p: usize,
        q: usize,
        runtime: Option<Arc<GpuRuntime>>,
    ) -> ReedResult<Self> {
        let cpu = RaviartThomasBasis::<T>::new(topology, p, q)?;
        Ok(Self { cpu, runtime })
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
                        (v_gpu[i] - v_cpu[i]).abs() < 1e-5,
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
}
