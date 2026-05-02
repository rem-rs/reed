mod basis;
mod basis_simplex;
mod elem_restriction;
pub mod qfunction_device;
mod runtime;
mod vector;

pub use qfunction_device::{
    IdentityF32Wgpu, IdentityScalarF32Wgpu, Mass1DBuildF32Wgpu, Mass2DBuildF32Wgpu,
    Mass3DBuildF32Wgpu, MassApplyF32Wgpu, MassApplyInterpTimesWeightF32Wgpu, Poisson1DApplyF32Wgpu,
    Poisson1DBuildF32Wgpu, Poisson2DApplyF32Wgpu, Poisson2DBuildF32Wgpu, Poisson3DApplyF32Wgpu,
    Poisson3DBuildF32Wgpu, QFunctionPrototypeScaleF32, ScaleF32Wgpu, Vec2DotF32Wgpu,
    Vec3DotF32Wgpu, Vector2MassApplyF32Wgpu, Vector2Poisson1DApplyF32Wgpu,
    Vector2Poisson2DApplyF32Wgpu, Vector3MassApplyF32Wgpu, Vector3Poisson1DApplyF32Wgpu,
    Vector3Poisson2DApplyF32Wgpu, Vector3Poisson3DApplyF32Wgpu,
};
use reed_core::{
    enums::*,
    error::{ReedError, ReedResult},
    qfunction::QFunctionTrait,
    scalar::Scalar,
    types::CeedInt,
    BasisTrait, ElemRestrictionTrait, VectorTrait,
};
pub use runtime::GpuRuntime;
use std::sync::Arc;

/// # Safety
/// Caller must ensure `T` is `f32` (`TypeId::of::<T>() == TypeId::of::<f32>()`).
unsafe fn coerce_qfunction_f32_box<T: Scalar>(
    q: Box<dyn QFunctionTrait<f32>>,
) -> Box<dyn QFunctionTrait<T>> {
    std::mem::transmute(q)
}

/// Backend factory trait (implemented by each backend).
#[cfg(not(target_arch = "wasm32"))]
pub trait Backend<T: Scalar>: Send + Sync {
    fn resource_name(&self) -> &str;

    fn create_vector(&self, size: usize) -> ReedResult<Box<dyn VectorTrait<T>>>;

    fn create_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[CeedInt],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>>;

    fn create_strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [CeedInt; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>>;

    fn create_face_elem_restriction(
        &self,
        _num_faces: usize,
        _num_dof_per_face: usize,
        _num_dof_per_elem: usize,
        _ncomp: usize,
        _num_global_dof: usize,
        _face_to_elem: Vec<(usize, usize)>,
        _face_offsets: &[CeedInt],
        _elem_offsets: &[CeedInt],
        _face_to_elem_local: Vec<usize>,
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Err(ReedError::BackendNotSupported(
            "create_face_elem_restriction is not implemented for this backend".into(),
        ))
    }

    fn create_basis_tensor_h1_lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;

    fn create_basis_h1_simplex(
        &self,
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;

    fn create_basis_hcurl_nedelec(
        &self,
        topology: ElemTopology,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;

    fn create_basis_hdiv_raviart_thomas(
        &self,
        topology: ElemTopology,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;
}

/// WASM variant — wgpu::Device is not Send+Sync in browser.
#[cfg(target_arch = "wasm32")]
pub trait Backend<T: Scalar> {
    fn resource_name(&self) -> &str;

    fn create_vector(&self, size: usize) -> ReedResult<Box<dyn VectorTrait<T>>>;

    fn create_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[CeedInt],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>>;

    fn create_strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [CeedInt; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>>;

    fn create_face_elem_restriction(
        &self,
        _num_faces: usize,
        _num_dof_per_face: usize,
        _num_dof_per_elem: usize,
        _ncomp: usize,
        _num_global_dof: usize,
        _face_to_elem: Vec<(usize, usize)>,
        _face_offsets: &[CeedInt],
        _elem_offsets: &[CeedInt],
        _face_to_elem_local: Vec<usize>,
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Err(ReedError::BackendNotSupported(
            "create_face_elem_restriction is not implemented for this backend".into(),
        ))
    }

    fn create_basis_tensor_h1_lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;

    fn create_basis_h1_simplex(
        &self,
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;

    fn create_basis_hcurl_nedelec(
        &self,
        topology: ElemTopology,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;

    fn create_basis_hdiv_raviart_thomas(
        &self,
        topology: ElemTopology,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;
}

pub struct WgpuBackend<T: Scalar> {
    gpu_available: bool,
    adapter_name: Option<String>,
    runtime: Option<Arc<GpuRuntime>>,
    /// CPU fallback for basis creation on WASM (where GPU basis is unavailable).
    #[allow(dead_code)]
    cpu_backend: reed_cpu::CpuBackend<T>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> Default for WgpuBackend<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> Clone for WgpuBackend<T> {
    fn clone(&self) -> Self {
        Self {
            gpu_available: self.gpu_available,
            adapter_name: self.adapter_name.clone(),
            runtime: self.runtime.clone(),
            cpu_backend: reed_cpu::CpuBackend::new(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar> WgpuBackend<T> {
    /// Synchronous init — uses pollster internally (native only).
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }));

        let (gpu_available, adapter_name, runtime) = if let Some(adapter) = adapter {
            let info = adapter.get_info();
            let rt = GpuRuntime::new(&adapter).map(GpuRuntime::shared);
            (
                rt.is_some(),
                Some(format!("{} ({:?})", info.name, info.backend)),
                rt,
            )
        } else {
            (false, None, None)
        };

        Self {
            gpu_available,
            adapter_name,
            runtime,
            cpu_backend: reed_cpu::CpuBackend::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Async init for WASM (no pollster — await the WebGPU futures).
    pub async fn new_async() -> Self {
        let instance = wgpu::Instance::default();
        let runtime =
            GpuRuntime::new_async(&instance, wgpu::PowerPreference::HighPerformance, false)
                .await
                .map(Arc::new);
        let (gpu_available, adapter_name) = if runtime.is_some() {
            (true, Some("WebGPU (WGSL compute)".to_string()))
        } else {
            (false, None)
        };
        Self {
            gpu_available,
            adapter_name,
            runtime,
            cpu_backend: reed_cpu::CpuBackend::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Build from an already-initialized GpuRuntime.
    pub fn from_runtime(runtime: Arc<GpuRuntime>, adapter_name: Option<String>) -> Self {
        Self {
            gpu_available: true,
            adapter_name,
            runtime: Some(runtime),
            cpu_backend: reed_cpu::CpuBackend::new(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    pub fn adapter_name(&self) -> Option<&str> {
        self.adapter_name.as_deref()
    }

    /// Shared GPU runtime when [`Self::is_gpu_available`] is true (cloneable `Arc` for device Q-functions).
    pub fn gpu_runtime(&self) -> Option<Arc<GpuRuntime>> {
        self.runtime.clone()
    }

    pub(crate) fn try_device_q_function_dispatch(
        &self,
        name: &str,
    ) -> Option<ReedResult<Box<dyn reed_core::QFunctionTrait<T>>>> {
        use std::any::TypeId;
        if TypeId::of::<T>() != TypeId::of::<f32>() {
            return None;
        }
        let rt = self.gpu_runtime()?;
        crate::qfunction_device::try_create_device_q_function_f32(name, rt).map(|r| {
            r.map(|q| unsafe {
                debug_assert_eq!(TypeId::of::<T>(), TypeId::of::<f32>());
                coerce_qfunction_f32_box(q)
            })
        })
    }
}

// Also implement reed_core::Backend so it satisfies reed::Backend (= reed_core::Backend).
// This impl is only available on non-WASM where wgpu::Device is Send+Sync.
#[cfg(not(target_arch = "wasm32"))]
impl<T: Scalar> reed_core::Backend<T> for WgpuBackend<T> {
    fn resource_name(&self) -> &str {
        <Self as Backend<T>>::resource_name(self)
    }

    fn create_vector(
        &self,
        size: usize,
    ) -> reed_core::ReedResult<Box<dyn reed_core::VectorTrait<T>>> {
        Backend::<T>::create_vector(self, size)
    }

    fn create_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[CeedInt],
    ) -> reed_core::ReedResult<Box<dyn reed_core::ElemRestrictionTrait<T>>> {
        Backend::<T>::create_elem_restriction(
            self, nelem, elemsize, ncomp, compstride, lsize, offsets,
        )
    }

    fn create_strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [CeedInt; 3],
    ) -> reed_core::ReedResult<Box<dyn reed_core::ElemRestrictionTrait<T>>> {
        Backend::<T>::create_strided_elem_restriction(self, nelem, elemsize, ncomp, lsize, strides)
    }

    fn create_face_elem_restriction(
        &self,
        num_faces: usize,
        num_dof_per_face: usize,
        num_dof_per_elem: usize,
        ncomp: usize,
        num_global_dof: usize,
        face_to_elem: Vec<(usize, usize)>,
        face_offsets: &[CeedInt],
        elem_offsets: &[CeedInt],
        face_to_elem_local: Vec<usize>,
    ) -> reed_core::ReedResult<Box<dyn reed_core::ElemRestrictionTrait<T>>> {
        Backend::<T>::create_face_elem_restriction(
            self,
            num_faces,
            num_dof_per_face,
            num_dof_per_elem,
            ncomp,
            num_global_dof,
            face_to_elem,
            face_offsets,
            elem_offsets,
            face_to_elem_local,
        )
    }

    fn create_basis_tensor_h1_lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: reed_core::enums::QuadMode,
    ) -> reed_core::ReedResult<Box<dyn reed_core::BasisTrait<T>>> {
        Backend::<T>::create_basis_tensor_h1_lagrange(self, dim, ncomp, p, q, qmode)
    }

    fn create_basis_h1_simplex(
        &self,
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
    ) -> reed_core::ReedResult<Box<dyn reed_core::BasisTrait<T>>> {
        Backend::<T>::create_basis_h1_simplex(self, topo, poly, ncomp, q)
    }

    fn create_basis_hcurl_nedelec(
        &self,
        topology: ElemTopology,
        p: usize,
        q: usize,
        qmode: reed_core::enums::QuadMode,
    ) -> reed_core::ReedResult<Box<dyn reed_core::BasisTrait<T>>> {
        Backend::<T>::create_basis_hcurl_nedelec(self, topology, p, q, qmode)
    }

    fn create_basis_hdiv_raviart_thomas(
        &self,
        topology: ElemTopology,
        p: usize,
        q: usize,
        qmode: reed_core::enums::QuadMode,
    ) -> reed_core::ReedResult<Box<dyn reed_core::BasisTrait<T>>> {
        Backend::<T>::create_basis_hdiv_raviart_thomas(self, topology, p, q, qmode)
    }

    fn try_device_q_function_by_name(
        &self,
        name: &str,
    ) -> Option<reed_core::ReedResult<Box<dyn reed_core::QFunctionTrait<T>>>> {
        self.try_device_q_function_dispatch(name)
    }
}

/// Non-WASM impl: WgpuBackend implements reed_wgpu::Backend with Send+Sync bounds.
#[cfg(not(target_arch = "wasm32"))]
impl<T: Scalar> Backend<T> for WgpuBackend<T> {
    fn resource_name(&self) -> &str {
        "/gpu/wgpu"
    }

    fn create_vector(&self, size: usize) -> ReedResult<Box<dyn VectorTrait<T>>> {
        Ok(Box::new(crate::vector::WgpuVector::<T>::new(
            size,
            self.runtime.clone(),
        )))
    }

    fn create_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[CeedInt],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Ok(Box::new(
            crate::elem_restriction::WgpuElemRestriction::<T>::new_offset(
                nelem,
                elemsize,
                ncomp,
                compstride,
                lsize,
                offsets,
                self.runtime.clone(),
            )?,
        ))
    }

    fn create_strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [CeedInt; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Ok(Box::new(
            crate::elem_restriction::WgpuElemRestriction::<T>::new_strided(
                nelem,
                elemsize,
                ncomp,
                lsize,
                strides,
                self.runtime.clone(),
            )?,
        ))
    }

    fn create_basis_tensor_h1_lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        Ok(Box::new(crate::basis::WgpuBasis::<T>::new(
            dim,
            ncomp,
            p,
            q,
            qmode,
            self.runtime.clone(),
        )?))
    }

    fn create_basis_h1_simplex(
        &self,
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        Ok(Box::new(crate::basis_simplex::WgpuSimplexBasis::<T>::new(
            topo,
            poly,
            ncomp,
            q,
            self.runtime.clone(),
        )?))
    }

    fn create_basis_hcurl_nedelec(
        &self,
        _topology: ElemTopology,
        _p: usize,
        _q: usize,
        _qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        Err(ReedError::BackendNotSupported(
            "Nedelec basis not yet implemented on WGPU".into(),
        ))
    }

    fn create_basis_hdiv_raviart_thomas(
        &self,
        _topology: ElemTopology,
        _p: usize,
        _q: usize,
        _qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        Err(ReedError::BackendNotSupported(
            "RT basis not yet implemented on WGPU".into(),
        ))
    }
}

/// WASM-only impl: on WASM, basis creation falls back to CPU since GPU basis is unavailable.
#[cfg(target_arch = "wasm32")]
impl<T: Scalar> reed_core::Backend<T> for WgpuBackend<T> {
    fn resource_name(&self) -> &str {
        "/gpu/wgpu"
    }

    fn create_vector(&self, size: usize) -> ReedResult<Box<dyn VectorTrait<T>>> {
        // On WASM, fall back to CPU vector (data stays on CPU, no GPU transfer needed for now)
        Ok(Box::new(crate::vector::WgpuVector::<T>::new(
            size,
            self.runtime.clone(),
        )))
    }

    fn create_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[CeedInt],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Ok(Box::new(
            crate::elem_restriction::WgpuElemRestriction::<T>::new_offset(
                nelem,
                elemsize,
                ncomp,
                compstride,
                lsize,
                offsets,
                self.runtime.clone(),
            )?,
        ))
    }

    fn create_strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [CeedInt; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Ok(Box::new(
            crate::elem_restriction::WgpuElemRestriction::<T>::new_strided(
                nelem,
                elemsize,
                ncomp,
                lsize,
                strides,
                self.runtime.clone(),
            )?,
        ))
    }

    fn create_face_elem_restriction(
        &self,
        _num_faces: usize,
        _num_dof_per_face: usize,
        _num_dof_per_elem: usize,
        _ncomp: usize,
        _num_global_dof: usize,
        _face_to_elem: Vec<(usize, usize)>,
        _face_offsets: &[CeedInt],
        _elem_offsets: &[CeedInt],
        _face_to_elem_local: Vec<usize>,
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Err(ReedError::BackendNotSupported(
            "create_face_elem_restriction is not implemented for this backend".into(),
        ))
    }

    fn create_basis_tensor_h1_lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        // On WASM, WgpuBasis doesn't implement BasisTrait (GPU runtime is not Send+Sync).
        // Fall back to CPU LagrangeBasis for basis evaluation.
        reed_core::Backend::create_basis_tensor_h1_lagrange(
            &self.cpu_backend,
            dim,
            ncomp,
            p,
            q,
            qmode,
        )
    }

    fn create_basis_h1_simplex(
        &self,
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        reed_core::Backend::create_basis_h1_simplex(&self.cpu_backend, topo, poly, ncomp, q)
    }

    fn create_basis_hcurl_nedelec(
        &self,
        _topology: ElemTopology,
        _p: usize,
        _q: usize,
        _qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        Err(ReedError::BackendNotSupported(
            "Nedelec basis not yet implemented on WGPU".into(),
        ))
    }

    fn create_basis_hdiv_raviart_thomas(
        &self,
        _topology: ElemTopology,
        _p: usize,
        _q: usize,
        _qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        Err(ReedError::BackendNotSupported(
            "RT basis not yet implemented on WGPU".into(),
        ))
    }

    fn try_device_q_function_by_name(
        &self,
        name: &str,
    ) -> Option<ReedResult<Box<dyn reed_core::QFunctionTrait<T>>>> {
        self.try_device_q_function_dispatch(name)
    }
}

pub fn wgpu_available() -> bool {
    let backend = WgpuBackend::<f64>::new();
    backend.is_gpu_available()
}

pub fn wgpu_adapter_name() -> Option<String> {
    let backend = WgpuBackend::<f64>::new();
    backend.adapter_name().map(ToOwned::to_owned)
}

pub fn require_wgpu() -> ReedResult<()> {
    if wgpu_available() {
        Ok(())
    } else {
        Err(ReedError::BackendNotSupported(
            "no suitable wgpu adapter found".into(),
        ))
    }
}
