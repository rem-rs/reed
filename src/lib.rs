//! Root `reed` crate: [`Reed`] context, backend resource strings, and re-exports.
//!
//! # Resource strings (`Reed::init`, `parse_backend_request`)
//!
//! Path-like IDs describe **where execution happens** (host CPU, GPU, future ISA
//! tiers such as AVX-512). Discrete PDE operators live here; algebraic solvers are
//! outside this crate.
//!
//! Recommended host CPU entry points: `/cpu/self`, `/cpu/self/ref`.
//! GPU: `/gpu/wgpu` (optional feature). **`/gpu/cuda`** and **`/gpu/hip`** are
//! reserved resource IDs (parsed + reported); execution is not implemented yet.
//!
//! Reed is designed as a **discrete backend** (operators, bases, restrictions)
//! for higher-level orchestration; resource strings describe execution hardware,
//! not any particular consumer.

pub use reed_core::{
    csr_sparsity_from_offset_lnodes, csr_sparsity_from_offset_restriction, Backend, BasisTrait,
    CeedInt, CeedMatrix, CeedMatrixStorage, ClosureQFunction, CsrMatrix, CsrPattern,
    ElemRestrictionTrait, ElemTopology, EvalMode, OperatorAssembleKind, OperatorTrait,
    OperatorTransposeRequest, QFunctionCategory, QFunctionClosure, QFunctionContext,
    QFunctionContextField, QFunctionContextFieldKind, QFunctionField, QFunctionTrait, QuadMode,
    ReedError, ReedResult, Scalar, TransposeMode, VectorTrait,
};
pub use reed_cpu::{
    q_function_by_name, CompositeOperator, CompositeOperatorBorrowed, CpuBackend,
    CpuFdmDenseInverseOperator, CpuFdmJacobiInverseOperator, CpuFdmTensorInverseOperator,
    CpuOperator, FdmOperatorKind, FieldVector, FDM_DENSE_MAX_N, NedelecBasis, OperatorBuilder,
    QFUNCTION_INTERIOR_GALLERY_NAMES, QFUNCTION_LIBCEED_MAIN_GALLERY_NAMES, RaviartThomasBasis,
};
#[cfg(feature = "wgpu-backend")]
pub use reed_wgpu::{GpuRuntime, WgpuBackend};

use std::sync::Arc;

pub struct Reed<T: Scalar> {
    inner: reed_core::Reed<T>,
}

/// Backend request surface exposed by reed (execution device / API only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReedBackendRequest {
    /// Default host CPU execution (`/cpu/self`, etc.).
    CpuHost,
    GpuWgpu,
    /// Reserved CUDA API route (`/gpu/cuda`); not implemented (placeholder).
    GpuCuda,
    /// Reserved HIP/ROCm API route (`/gpu/hip`); not implemented (placeholder).
    GpuHip,
}

/// Compile-time/runtime capability snapshot for reed backend routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReedBackendCapabilities {
    pub gpu_wgpu: bool,
    pub wasm_target: bool,
}

impl ReedBackendCapabilities {
    pub fn detect() -> Self {
        Self {
            gpu_wgpu: cfg!(feature = "wgpu-backend"),
            wasm_target: cfg!(target_arch = "wasm32"),
        }
    }
}

/// Deterministic backend-selection result used for diagnostics/integration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReedBackendSelectionReport {
    pub requested: Option<ReedBackendRequest>,
    pub effective_resource: String,
    pub capabilities: ReedBackendCapabilities,
    pub note: String,
}

impl<T: Scalar> Reed<T> {
    /// Return capability snapshot used by reed backend selection.
    pub fn backend_capabilities() -> ReedBackendCapabilities {
        ReedBackendCapabilities::detect()
    }

    /// Resolve requested backend into a deterministic reed runtime resource.
    pub fn backend_selection_report(
        requested: Option<ReedBackendRequest>,
    ) -> ReedBackendSelectionReport {
        let caps = ReedBackendCapabilities::detect();
        resolve_backend_request(requested, caps)
    }

    /// Initialize reed with a backend request and return selection diagnostics.
    pub fn init_with_backend(
        requested: ReedBackendRequest,
    ) -> ReedResult<(Self, ReedBackendSelectionReport)> {
        let report = Self::backend_selection_report(Some(requested));
        let reed = Self::init(&report.effective_resource)?;
        Ok((reed, report))
    }

    /// Parse a backend/resource string into a [`ReedBackendRequest`].
    ///
    /// Returns `None` when the resource is unknown.
    pub fn parse_backend_request(resource: &str) -> Option<ReedBackendRequest> {
        match resource {
            "/cpu/self" | "/cpu/self/ref" => Some(ReedBackendRequest::CpuHost),
            "/gpu/wgpu" | "/gpu/wgpu/ref" => Some(ReedBackendRequest::GpuWgpu),
            "/gpu/cuda" | "/gpu/cuda/ref" => Some(ReedBackendRequest::GpuCuda),
            "/gpu/hip" | "/gpu/hip/ref" => Some(ReedBackendRequest::GpuHip),
            _ => None,
        }
    }

    /// Initialize reed from backend/resource strings (see module docs).
    ///
    /// Examples: `/cpu/self`, `/gpu/wgpu`. `/gpu/cuda` and `/gpu/hip` parse but
    /// [`init`](Self::init) returns an error until those backends exist.
    pub fn init_with_backend_resource(
        resource: &str,
    ) -> ReedResult<(Self, ReedBackendSelectionReport)> {
        let requested = Self::parse_backend_request(resource)
            .ok_or_else(|| ReedError::BackendNotSupported(resource.into()))?;
        Self::init_with_backend(requested)
    }

    pub fn init(resource: &str) -> ReedResult<Self> {
        if matches!(resource, "/cpu/self" | "/cpu/self/ref") {
            return Ok(Self {
                inner: reed_core::Reed::from_backend(Arc::new(CpuBackend::<T>::new())),
            });
        }

        #[cfg(feature = "wgpu-backend")]
        if matches!(resource, "/gpu/wgpu" | "/gpu/wgpu/ref") {
            return Ok(Self {
                inner: reed_core::Reed::from_backend(Arc::new(WgpuBackend::<T>::new())),
            });
        }

        #[cfg(not(feature = "wgpu-backend"))]
        if matches!(resource, "/gpu/wgpu" | "/gpu/wgpu/ref") {
            return Err(ReedError::BackendNotSupported(
                "wgpu backend is disabled; build with feature 'wgpu-backend'".into(),
            ));
        }

        if matches!(resource, "/gpu/cuda" | "/gpu/cuda/ref") {
            return Err(ReedError::BackendNotSupported(
                "backend /gpu/cuda is not implemented yet (reserved placeholder)".into(),
            ));
        }
        if matches!(resource, "/gpu/hip" | "/gpu/hip/ref") {
            return Err(ReedError::BackendNotSupported(
                "backend /gpu/hip is not implemented yet (reserved placeholder)".into(),
            ));
        }

        Err(ReedError::BackendNotSupported(resource.into()))
    }

    /// Build a Reed context from a pre-configured backend.
    pub fn from_backend(backend: Arc<dyn Backend<T>>) -> Self {
        Self {
            inner: reed_core::Reed::from_backend(backend),
        }
    }

    pub fn resource(&self) -> String {
        self.inner.resource()
    }

    pub fn vector(&self, n: usize) -> ReedResult<Box<dyn VectorTrait<T>>> {
        self.inner.vector(n)
    }

    pub fn vector_from_slice(&self, data: &[T]) -> ReedResult<Box<dyn VectorTrait<T>>> {
        self.inner.vector_from_slice(data)
    }

    pub fn elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[CeedInt],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        self.inner
            .elem_restriction(nelem, elemsize, ncomp, compstride, lsize, offsets)
    }

    pub fn strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [CeedInt; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        self.inner
            .strided_elem_restriction(nelem, elemsize, ncomp, lsize, strides)
    }

    /// See [`reed_core::Reed::elem_restriction_at_points`].
    pub fn elem_restriction_at_points(
        &self,
        nelem: usize,
        npoints_per_elem: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[CeedInt],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        self.inner.elem_restriction_at_points(
            nelem,
            npoints_per_elem,
            ncomp,
            compstride,
            lsize,
            offsets,
        )
    }

    /// See [`reed_core::Reed::elem_restriction_ceed_int_offsets`].
    pub fn elem_restriction_ceed_int_offsets(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i64],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        self.inner
            .elem_restriction_ceed_int_offsets(nelem, elemsize, ncomp, compstride, lsize, offsets)
    }

    /// See [`reed_core::Reed::elem_restriction_at_points_ceed_int_offsets`].
    pub fn elem_restriction_at_points_ceed_int_offsets(
        &self,
        nelem: usize,
        npoints_per_elem: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i64],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        self.inner.elem_restriction_at_points_ceed_int_offsets(
            nelem,
            npoints_per_elem,
            ncomp,
            compstride,
            lsize,
            offsets,
        )
    }

    /// See [`reed_core::Reed::strided_elem_restriction_ceed_int_strides`].
    pub fn strided_elem_restriction_ceed_int_strides(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i64; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        self.inner
            .strided_elem_restriction_ceed_int_strides(nelem, elemsize, ncomp, lsize, strides)
    }

    pub fn basis_tensor_h1_lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        self.inner.basis_tensor_h1_lagrange(dim, ncomp, p, q, qmode)
    }

    pub fn basis_h1_simplex(
        &self,
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        self.inner.basis_h1_simplex(topo, poly, ncomp, q)
    }

    pub fn operator_builder<'a>(&'a self) -> OperatorBuilder<'a, T> {
        OperatorBuilder::new()
    }

    /// Build a composite operator `y = sum_i A_i x` (libCEED `CeedCompositeOperator`-style additive apply).
    pub fn composite_operator(
        &self,
        ops: Vec<Box<dyn OperatorTrait<T>>>,
    ) -> ReedResult<CompositeOperator<T>> {
        let _ = self;
        CompositeOperator::new(ops)
    }

    /// Same as [`Self::composite_operator`], but composes `&dyn` sub-operators (e.g. two `CpuOperator`
    /// values that borrow the same mesh in one scope). Matches libCEED-style composition with a shared context.
    pub fn composite_operator_refs<'a>(
        &'a self,
        ops: &[&'a dyn OperatorTrait<T>],
    ) -> ReedResult<CompositeOperatorBorrowed<'a, T>> {
        let _ = self;
        CompositeOperatorBorrowed::new(ops.to_vec())
    }

    /// User-defined **interior** QFunction — host `apply` path matches [`Self::q_function_exterior`];
    /// [`QFunctionTrait::q_function_category`] is [`QFunctionCategory::Interior`] (libCEED interior migration marker).
    pub fn q_function_interior(
        &self,
        vector_length: usize,
        inputs: Vec<QFunctionField>,
        outputs: Vec<QFunctionField>,
        context_byte_len: usize,
        closure: Box<QFunctionClosure<T>>,
    ) -> ReedResult<Box<dyn QFunctionTrait<T>>> {
        let _ = self;
        if vector_length == 0 {
            return Err(ReedError::InvalidArgument(
                "qfunction vector_length must be greater than zero".into(),
            ));
        }
        Ok(Box::new(ClosureQFunction::new(
            inputs,
            outputs,
            context_byte_len,
            closure,
        )))
    }

    /// User-defined **exterior** (boundary) QFunction — same host evaluation as [`Self::q_function_interior`],
    /// but reports [`QFunctionCategory::Exterior`] (libCEED exterior / active-side migration marker).
    pub fn q_function_exterior(
        &self,
        vector_length: usize,
        inputs: Vec<QFunctionField>,
        outputs: Vec<QFunctionField>,
        context_byte_len: usize,
        closure: Box<QFunctionClosure<T>>,
    ) -> ReedResult<Box<dyn QFunctionTrait<T>>> {
        let _ = self;
        if vector_length == 0 {
            return Err(ReedError::InvalidArgument(
                "qfunction vector_length must be greater than zero".into(),
            ));
        }
        Ok(Box::new(ClosureQFunction::new_with_category(
            inputs,
            outputs,
            context_byte_len,
            QFunctionCategory::Exterior,
            closure,
        )))
    }

    /// Gallery QFunction by libCEED-compatible name (see `reed_cpu::gallery` and `design_mapping.md`).
    ///
    /// On `/gpu/wgpu` with scalar type `f32`, supported names are resolved to WGSL device kernels
    /// when a GPU adapter is available; other names and scalar types use the host CPU gallery.
    pub fn q_function_by_name(&self, name: &str) -> ReedResult<Box<dyn QFunctionTrait<T>>> {
        let dev = {
            let g = self.inner.backend().lock().unwrap();
            (**g).try_device_q_function_by_name(name)
        };
        if let Some(res) = dev {
            return res;
        }
        q_function_by_name(name)
    }
}

fn resolve_backend_request(
    requested: Option<ReedBackendRequest>,
    caps: ReedBackendCapabilities,
) -> ReedBackendSelectionReport {
    match requested {
        None | Some(ReedBackendRequest::CpuHost) => ReedBackendSelectionReport {
            requested,
            effective_resource: "/cpu/self".to_string(),
            capabilities: caps,
            note: "No GPU route requested; using default host CPU backend (/cpu/self).".to_string(),
        },
        Some(ReedBackendRequest::GpuWgpu) => {
            if caps.wasm_target {
                ReedBackendSelectionReport {
                    requested,
                    effective_resource: "/cpu/self".to_string(),
                    capabilities: caps,
                    note: "Requested gpu/wgpu on wasm32 target; using deterministic fallback to host CPU (/cpu/self).".to_string(),
                }
            } else if caps.gpu_wgpu {
                ReedBackendSelectionReport {
                    requested,
                    effective_resource: "/gpu/wgpu".to_string(),
                    capabilities: caps,
                    note: "Requested gpu/wgpu and feature is enabled; using reed wgpu backend."
                        .to_string(),
                }
            } else {
                ReedBackendSelectionReport {
                    requested,
                    effective_resource: "/cpu/self".to_string(),
                    capabilities: caps,
                    note: "Requested gpu/wgpu but feature wgpu-backend is disabled; using deterministic fallback to host CPU (/cpu/self).".to_string(),
                }
            }
        }
        Some(ReedBackendRequest::GpuCuda) => ReedBackendSelectionReport {
            requested,
            effective_resource: "/gpu/cuda".to_string(),
            capabilities: caps,
            note: "Reserved /gpu/cuda backend; not implemented (placeholder).".to_string(),
        },
        Some(ReedBackendRequest::GpuHip) => ReedBackendSelectionReport {
            requested,
            effective_resource: "/gpu/hip".to_string(),
            capabilities: caps,
            note: "Reserved /gpu/hip backend; not implemented (placeholder).".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_report_defaults_to_host_cpu() {
        let rep = Reed::<f64>::backend_selection_report(None);
        assert_eq!(rep.effective_resource, "/cpu/self");
    }

    #[test]
    fn backend_report_gpu_without_feature_falls_back() {
        let rep = Reed::<f64>::backend_selection_report(Some(ReedBackendRequest::GpuWgpu));
        if !rep.capabilities.gpu_wgpu || rep.capabilities.wasm_target {
            assert_eq!(rep.effective_resource, "/cpu/self");
            assert!(rep.note.contains("fallback"));
        }
    }

    #[test]
    fn parse_backend_request_supports_canonical_paths() {
        assert_eq!(
            Reed::<f64>::parse_backend_request("/cpu/self"),
            Some(ReedBackendRequest::CpuHost)
        );
        assert_eq!(
            Reed::<f64>::parse_backend_request("/gpu/wgpu"),
            Some(ReedBackendRequest::GpuWgpu)
        );
        assert_eq!(
            Reed::<f64>::parse_backend_request("/gpu/cuda"),
            Some(ReedBackendRequest::GpuCuda)
        );
        assert_eq!(
            Reed::<f64>::parse_backend_request("/gpu/hip"),
            Some(ReedBackendRequest::GpuHip)
        );
        assert_eq!(
            Reed::<f64>::parse_backend_request("/unknown/resource"),
            None
        );
    }

    #[test]
    fn backend_report_cuda_hip_placeholders_keep_effective_paths() {
        let cuda = Reed::<f64>::backend_selection_report(Some(ReedBackendRequest::GpuCuda));
        assert_eq!(cuda.effective_resource, "/gpu/cuda");
        assert!(cuda.note.contains("placeholder"));

        let hip = Reed::<f64>::backend_selection_report(Some(ReedBackendRequest::GpuHip));
        assert_eq!(hip.effective_resource, "/gpu/hip");
        assert!(hip.note.contains("placeholder"));
    }

    #[test]
    fn init_cuda_hip_placeholders_error() {
        assert!(Reed::<f64>::init("/gpu/cuda").is_err());
        assert!(Reed::<f64>::init("/gpu/hip").is_err());
    }
}
