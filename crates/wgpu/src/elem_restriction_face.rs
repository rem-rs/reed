use std::sync::Arc;

use reed_core::{
    enums::TransposeMode,
    error::ReedResult,
    scalar::Scalar,
    types::CeedInt,
    ElemRestrictionTrait,
};
use reed_cpu::elem_restriction_face::CpuFaceElemRestriction;
use wgpu::util::DeviceExt;

use crate::runtime::GpuRuntime;

/// GPU face element restriction: maps boundary faces to their parent elements.
///
/// In v1, the GPU path delegates to the CPU fallback for the actual restriction
/// work (gather/scatter), similar to how [`WgpuNedelecBasis`] works. The struct
/// encapsulates the [`GpuRuntime`] and GPU buffers for future kernel dispatch.
///
/// From [`ElemRestrictionTrait`]'s perspective, each "element" is a boundary face,
/// so [`num_elements`](ElemRestrictionTrait::num_elements) returns `num_faces` and
/// [`num_dof_per_elem`](ElemRestrictionTrait::num_dof_per_elem) returns `num_dof_per_face`.
pub struct WgpuFaceElemRestriction<T: Scalar> {
    cpu_fallback: CpuFaceElemRestriction<T>,
    #[allow(dead_code)]
    runtime: Option<Arc<GpuRuntime>>,
    /// f32 GPU buffers for face_offsets, elem_offsets, face_to_elem_local
    #[allow(dead_code)]
    face_offsets_gpu: Option<wgpu::Buffer>,
    #[allow(dead_code)]
    elem_offsets_gpu: Option<wgpu::Buffer>,
    #[allow(dead_code)]
    face_to_elem_local_gpu: Option<wgpu::Buffer>,
}

impl<T: Scalar> WgpuFaceElemRestriction<T> {
    pub fn new(
        num_faces: usize,
        num_dof_per_face: usize,
        num_dof_per_elem: usize,
        ncomp: usize,
        num_global_dof: usize,
        face_to_elem: Vec<(usize, usize)>,
        face_offsets: &[CeedInt],
        elem_offsets: &[CeedInt],
        face_to_elem_local: Vec<usize>,
        runtime: Option<Arc<GpuRuntime>>,
    ) -> ReedResult<Self> {
        // v1: GPU buffers are pre-allocated but not yet used for kernel dispatch.
        // The buffers will be consumed by future face-restriction WGSL pipelines.
        // Build GPU buffers from the original data before it is moved into the CPU fallback.
        let (face_offsets_gpu, elem_offsets_gpu, face_to_elem_local_gpu) =
            if let Some(rt) = &runtime {
                // Upload face_offsets as f32 (CeedInt -> f32 for WGSL compat)
                let fo_f32: Vec<f32> = face_offsets.iter().map(|&x| x as f32).collect();
                let fo_buf = Some(rt.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("wgpu-face-restriction-face-offsets"),
                        contents: bytemuck::cast_slice(&fo_f32),
                        usage: wgpu::BufferUsages::STORAGE,
                    },
                ));

                let eo_f32: Vec<f32> = elem_offsets.iter().map(|&x| x as f32).collect();
                let eo_buf = Some(rt.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("wgpu-face-restriction-elem-offsets"),
                        contents: bytemuck::cast_slice(&eo_f32),
                        usage: wgpu::BufferUsages::STORAGE,
                    },
                ));

                let ftel_f32: Vec<f32> =
                    face_to_elem_local.iter().map(|&x| x as f32).collect();
                let ftel_buf = Some(rt.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("wgpu-face-restriction-face-to-elem-local"),
                        contents: bytemuck::cast_slice(&ftel_f32),
                        usage: wgpu::BufferUsages::STORAGE,
                    },
                ));

                (fo_buf, eo_buf, ftel_buf)
            } else {
                (None, None, None)
            };

        let cpu_fallback = CpuFaceElemRestriction::<T>::new(
            num_faces,
            num_dof_per_face,
            num_dof_per_elem,
            ncomp,
            num_global_dof,
            face_to_elem,
            face_offsets.to_vec(),
            elem_offsets.to_vec(),
            face_to_elem_local,
        )?;

        Ok(Self {
            cpu_fallback,
            runtime,
            face_offsets_gpu,
            elem_offsets_gpu,
            face_to_elem_local_gpu,
        })
    }
}

impl<T: Scalar> ElemRestrictionTrait<T> for WgpuFaceElemRestriction<T> {
    fn num_elements(&self) -> usize {
        self.cpu_fallback.num_elements()
    }

    fn num_dof_per_elem(&self) -> usize {
        self.cpu_fallback.num_dof_per_elem()
    }

    fn num_global_dof(&self) -> usize {
        self.cpu_fallback.num_global_dof()
    }

    fn num_comp(&self) -> usize {
        self.cpu_fallback.num_comp()
    }

    fn apply(&self, t_mode: TransposeMode, u: &[T], v: &mut [T]) -> ReedResult<()> {
        // v1: delegate to CPU fallback for the actual restriction work.
        // GPU dispatch will be added in a future version.
        self.cpu_fallback.apply(t_mode, u, v)
    }

    fn boxed_clone(&self) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Ok(Box::new(WgpuFaceElemRestriction {
            cpu_fallback: self.cpu_fallback.clone(),
            runtime: self.runtime.clone(),
            face_offsets_gpu: None,
            elem_offsets_gpu: None,
            face_to_elem_local_gpu: None,
        }))
    }
}

impl<T: Scalar> Clone for WgpuFaceElemRestriction<T> {
    fn clone(&self) -> Self {
        Self {
            cpu_fallback: self.cpu_fallback.clone(),
            runtime: self.runtime.clone(),
            face_offsets_gpu: None,
            elem_offsets_gpu: None,
            face_to_elem_local_gpu: None,
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use reed_core::ElemRestrictionTrait;

    fn gpu_runtime_or_skip() -> Option<Arc<GpuRuntime>> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))?;
        GpuRuntime::new(&adapter).map(Arc::new)
    }

    fn make_face_restriction(
        runtime: Option<Arc<GpuRuntime>>,
    ) -> WgpuFaceElemRestriction<f64> {
        let num_faces = 2;
        let num_dof_per_face = 2;
        let num_dof_per_elem = 4;
        let ncomp = 1;
        // Use indices within range: face 0 DOFs -> global 0,2; face 1 DOFs -> global 1,3
        let num_global_dof = 4;

        let face_to_elem = vec![(0, 0), (0, 1)];
        let face_offsets: Vec<CeedInt> = vec![0, 2, 1, 3];
        let elem_offsets: Vec<CeedInt> = vec![0, 1, 2, 3];
        let face_to_elem_local = vec![0, 2, 1, 3];

        WgpuFaceElemRestriction::new(
            num_faces,
            num_dof_per_face,
            num_dof_per_elem,
            ncomp,
            num_global_dof,
            face_to_elem,
            &face_offsets,
            &elem_offsets,
            face_to_elem_local,
            runtime,
        )
        .unwrap()
    }

    #[test]
    fn test_num_elements_returns_num_faces() {
        let r = make_face_restriction(None);
        assert_eq!(r.num_elements(), 2);
        assert_eq!(r.num_dof_per_elem(), 2);
        assert_eq!(r.num_global_dof(), 4);
        assert_eq!(r.num_comp(), 1);
    }

    #[test]
    fn test_face_restriction_gather_scatter() {
        let r = make_face_restriction(None);
        let u = vec![10.0_f64, 20.0, 30.0, 40.0];
        let mut v = vec![0.0_f64; 4];
        r.apply(TransposeMode::NoTranspose, &u, &mut v).unwrap();
        // face 0: global[0]=10, global[2]=30 -> [10, 30]
        // face 1: global[1]=20, global[3]=40 -> [20, 40]
        assert_eq!(v, vec![10.0, 30.0, 20.0, 40.0]);
    }

    #[test]
    fn test_face_restriction_gpu_roundtrip_matches_cpu() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };

        // Simple roundtrip: faces have identity mapping to global DOFs
        let num_faces = 2;
        let num_dof_per_face = 2;
        let num_dof_per_elem = 4;
        let ncomp = 1;
        let num_global_dof = 4;

        let face_to_elem = vec![(0, 0), (0, 1)];
        let face_offsets: Vec<CeedInt> = vec![0, 2, 1, 3];
        let elem_offsets: Vec<CeedInt> = vec![0, 1, 2, 3];
        let face_to_elem_local = vec![0, 2, 1, 3];

        let r_wgpu = WgpuFaceElemRestriction::<f64>::new(
            num_faces, num_dof_per_face, num_dof_per_elem, ncomp, num_global_dof,
            face_to_elem.clone(),
            &face_offsets,
            &elem_offsets,
            face_to_elem_local.clone(),
            Some(rt),
        )
        .unwrap();

        let r_cpu = CpuFaceElemRestriction::<f64>::new(
            num_faces, num_dof_per_face, num_dof_per_elem, ncomp, num_global_dof,
            face_to_elem,
            face_offsets.to_vec(),
            elem_offsets.to_vec(),
            face_to_elem_local,
        )
        .unwrap();

        // Gather
        let u = vec![10.0, 20.0, 30.0, 40.0];
        let mut v_wgpu = vec![0.0; 4];
        let mut v_cpu = vec![0.0; 4];
        r_wgpu.apply(TransposeMode::NoTranspose, &u, &mut v_wgpu).unwrap();
        r_cpu.apply(TransposeMode::NoTranspose, &u, &mut v_cpu).unwrap();
        assert_eq!(v_wgpu, v_cpu);

        // Scatter
        let mut gathered_wgpu = vec![0.0; 4];
        let mut gathered_cpu = vec![0.0; 4];
        r_wgpu.apply(TransposeMode::Transpose, &v_wgpu, &mut gathered_wgpu).unwrap();
        r_cpu.apply(TransposeMode::Transpose, &v_cpu, &mut gathered_cpu).unwrap();
        assert_eq!(gathered_wgpu, gathered_cpu);
        assert_eq!(gathered_wgpu, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_boxed_clone_works() {
        let r = make_face_restriction(None);
        let clone = r.boxed_clone().unwrap();
        assert_eq!(clone.num_elements(), r.num_elements());
        assert_eq!(clone.num_dof_per_elem(), r.num_dof_per_elem());
        assert_eq!(clone.num_global_dof(), r.num_global_dof());
        assert_eq!(clone.num_comp(), r.num_comp());

        // Clone retains correct behavior
        let u = vec![0.0_f64; 4];
        let mut v = vec![0.0_f64; 4];
        clone.apply(TransposeMode::NoTranspose, &u, &mut v).unwrap();
    }

    #[test]
    fn test_constructor_validation() {
        // Wrong face_offsets length
        let result = WgpuFaceElemRestriction::<f64>::new(
            2, 2, 4, 1, 5,
            vec![(0, 0), (0, 1)],
            &[0, 1, 2], // wrong: should be length 4
            &[0, 1, 2, 3],
            vec![0, 2, 1, 3],
            None,
        );
        assert!(result.is_err());

        // Wrong face_to_elem_local length
        let result = WgpuFaceElemRestriction::<f64>::new(
            2, 2, 4, 1, 5,
            vec![(0, 0), (0, 1)],
            &[0, 1, 2, 3],
            &[0, 1, 2, 3],
            vec![0, 2, 1], // wrong: should be length 4
            None,
        );
        assert!(result.is_err());
    }
}
