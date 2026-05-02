use crate::{
    basis::BasisTrait,
    elem_restriction::ElemRestrictionTrait,
    enums::*,
    error::{ReedError, ReedResult},
    qfunction::QFunctionTrait,
    scalar::Scalar,
    types::CeedInt,
    vector::VectorTrait,
};
use std::sync::{Arc, Mutex};

/// Convert libCEED `CeedInt` / `int64_t` offset buffers from C interop into `i32` for Reed.
fn ceed_int_offsets_to_i32(offsets: &[i64]) -> ReedResult<Vec<i32>> {
    offsets
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            i32::try_from(v).map_err(|_| {
                ReedError::InvalidArgument(format!(
                    "offset[{i}]={v} does not fit in i32 (Reed restriction index type)"
                ))
            })
        })
        .collect()
}

fn ceed_int_strides_to_i32(strides: [i64; 3]) -> ReedResult<[i32; 3]> {
    let mut out = [0i32; 3];
    for (i, &v) in strides.iter().enumerate() {
        out[i] = i32::try_from(v).map_err(|_| {
            ReedError::InvalidArgument(format!(
                "strides[{i}]={v} does not fit in i32 (Reed restriction stride type)"
            ))
        })?;
    }
    Ok(out)
}

/// Backend factory trait (implemented by each backend).
///
/// On WASM targets, the `Send + Sync` bounds are omitted because wgpu::Device
/// is not thread-safe in the browser's single-threaded environment.
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
        num_faces: usize,
        num_dof_per_face: usize,
        num_dof_per_elem: usize,
        ncomp: usize,
        num_global_dof: usize,
        face_to_elem: Vec<(usize, usize)>,
        face_offsets: &[CeedInt],
        elem_offsets: &[CeedInt],
        face_to_elem_local: Vec<usize>,
    ) -> ReedResult<Box<dyn crate::elem_restriction::ElemRestrictionTrait<T>>> {
        let _ = (
            num_faces,
            num_dof_per_face,
            num_dof_per_elem,
            ncomp,
            num_global_dof,
            face_to_elem,
            face_offsets,
            elem_offsets,
            face_to_elem_local,
        );
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

    /// Create an H1 Lagrange basis on a simplex reference element.
    ///
    /// # Parameters
    /// * `topo`  — `ElemTopology::Line`, `Triangle`, or `Tet` (CPU simplex basis).
    /// * `poly`  — polynomial order (1 = P1, 2 = P2).
    /// * `ncomp` — number of field components.
    /// * `q`     — number of quadrature points (see `SimplexBasis` docs for
    ///             valid values per topology).
    fn create_basis_h1_simplex(
        &self,
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
    ) -> ReedResult<Box<dyn BasisTrait<T>>>;

    fn create_basis_hcurl_nedelec(
        &self,
        _topology: ElemTopology,
        _p: usize,
        _q: usize,
        _qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        Err(ReedError::BackendNotSupported(
            "create_basis_hcurl_nedelec is not implemented for this backend".into(),
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
            "create_basis_hdiv_raviart_thomas is not implemented for this backend".into(),
        ))
    }

    /// Optional device-side gallery QFunction (e.g. WGSL on wgpu for `f32`).
    ///
    /// When this returns `None`, callers should fall back to the host CPU gallery lookup.
    fn try_device_q_function_by_name(
        &self,
        _name: &str,
    ) -> Option<ReedResult<Box<dyn QFunctionTrait<T>>>> {
        None
    }
}

/// On WASM, wgpu::Device is not Send+Sync so neither is the Backend trait.
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
        num_faces: usize,
        num_dof_per_face: usize,
        num_dof_per_elem: usize,
        ncomp: usize,
        num_global_dof: usize,
        face_to_elem: Vec<(usize, usize)>,
        face_offsets: &[CeedInt],
        elem_offsets: &[CeedInt],
        face_to_elem_local: Vec<usize>,
    ) -> ReedResult<Box<dyn crate::elem_restriction::ElemRestrictionTrait<T>>> {
        let _ = (
            num_faces,
            num_dof_per_face,
            num_dof_per_elem,
            ncomp,
            num_global_dof,
            face_to_elem,
            face_offsets,
            elem_offsets,
            face_to_elem_local,
        );
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
        _topology: ElemTopology,
        _p: usize,
        _q: usize,
        _qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        Err(ReedError::BackendNotSupported(
            "create_basis_hcurl_nedelec is not implemented for this backend".into(),
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
            "create_basis_hdiv_raviart_thomas is not implemented for this backend".into(),
        ))
    }

    fn try_device_q_function_by_name(
        &self,
        _name: &str,
    ) -> Option<ReedResult<Box<dyn QFunctionTrait<T>>>> {
        None
    }
}

/// Top-level Reed library context.
pub struct Reed<T: Scalar> {
    backend: Arc<Mutex<Arc<dyn Backend<T>>>>,
}

impl<T: Scalar> Reed<T> {
    /// Create from an existing backend (mainly for tests and internal usage).
    pub fn from_backend(backend: Arc<dyn Backend<T>>) -> Self {
        Self {
            backend: Arc::new(Mutex::new(backend)),
        }
    }

    pub fn resource(&self) -> String {
        (**self.backend.lock().unwrap()).resource_name().to_owned()
    }

    // -- Vector factory --

    pub fn vector(&self, n: usize) -> ReedResult<Box<dyn VectorTrait<T>>> {
        (**self.backend.lock().unwrap()).create_vector(n)
    }

    pub fn vector_from_slice(&self, data: &[T]) -> ReedResult<Box<dyn VectorTrait<T>>> {
        let mut v = (**self.backend.lock().unwrap()).create_vector(data.len())?;
        v.copy_from_slice(data)?;
        Ok(v)
    }

    // -- ElemRestriction factory --

    pub fn elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[CeedInt],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        (**self.backend.lock().unwrap())
            .create_elem_restriction(nelem, elemsize, ncomp, compstride, lsize, offsets)
    }

    pub fn strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [CeedInt; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        (**self.backend.lock().unwrap())
            .create_strided_elem_restriction(nelem, elemsize, ncomp, lsize, strides)
    }

    /// Restriction with `elemsize = npoints_per_elem` (dofs indexed per quadrature point per element).
    ///
    /// Same implementation as [`Self::elem_restriction`]; aligns with libCEED
    /// `CeedElemRestrictionCreateAtPoints` naming. `offsets.len()` must be `nelem * npoints_per_elem`.
    pub fn elem_restriction_at_points(
        &self,
        nelem: usize,
        npoints_per_elem: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[CeedInt],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        let expected = nelem.checked_mul(npoints_per_elem).ok_or_else(|| {
            ReedError::InvalidArgument("elem_restriction_at_points: size overflow".into())
        })?;
        if offsets.len() != expected {
            return Err(ReedError::InvalidArgument(format!(
                "elem_restriction_at_points: offsets.len() {} != nelem * npoints_per_elem ({})",
                offsets.len(),
                expected
            )));
        }
        self.elem_restriction(nelem, npoints_per_elem, ncomp, compstride, lsize, offsets)
    }

    /// Like [`Self::elem_restriction`], but accepts `i64` offsets (typical when bridging from
    /// libCEED `CeedInt` arrays stored as `int64_t` / `i64` in generated bindings).
    pub fn elem_restriction_ceed_int_offsets(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i64],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        let v = ceed_int_offsets_to_i32(offsets)?;
        self.elem_restriction(nelem, elemsize, ncomp, compstride, lsize, &v)
    }

    /// Like [`Self::elem_restriction_at_points`], but accepts `i64` offsets.
    pub fn elem_restriction_at_points_ceed_int_offsets(
        &self,
        nelem: usize,
        npoints_per_elem: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i64],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        let v = ceed_int_offsets_to_i32(offsets)?;
        self.elem_restriction_at_points(nelem, npoints_per_elem, ncomp, compstride, lsize, &v)
    }

    /// Like [`Self::strided_elem_restriction`], but accepts `i64` strides (libCEED `CeedInt[3]`).
    pub fn strided_elem_restriction_ceed_int_strides(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i64; 3],
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        let s = ceed_int_strides_to_i32(strides)?;
        self.strided_elem_restriction(nelem, elemsize, ncomp, lsize, s)
    }

    // -- Face ElemRestriction factory --

    /// Create a face element restriction that maps boundary faces to their parent elements.
    ///
    /// Each "element" from the restriction's perspective is a boundary face.
    /// [`ElemRestrictionTrait::num_elements`] returns `num_faces` and
    /// [`ElemRestrictionTrait::num_dof_per_elem`] returns `num_dof_per_face`.
    pub fn face_elem_restriction(
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
    ) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        (**self.backend.lock().unwrap()).create_face_elem_restriction(
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

    // -- Basis factory --

    pub fn basis_tensor_h1_lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        (**self.backend.lock().unwrap()).create_basis_tensor_h1_lagrange(dim, ncomp, p, q, qmode)
    }

    /// Create an H1 Lagrange basis on a simplex reference element (segment, triangle, or tet).
    ///
    /// See [`Backend::create_basis_h1_simplex`] for parameter details.
    pub fn basis_h1_simplex(
        &self,
        topo: ElemTopology,
        poly: usize,
        ncomp: usize,
        q: usize,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        (**self.backend.lock().unwrap()).create_basis_h1_simplex(topo, poly, ncomp, q)
    }

    pub fn basis_hcurl_nedelec(
        &self,
        topo: ElemTopology,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        (**self.backend.lock().unwrap()).create_basis_hcurl_nedelec(topo, p, q, qmode)
    }

    pub fn basis_hdiv_raviart_thomas(
        &self,
        topo: ElemTopology,
        p: usize,
        q: usize,
        qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        (**self.backend.lock().unwrap()).create_basis_hdiv_raviart_thomas(topo, p, q, qmode)
    }

    /// Get the backend handle.
    pub fn backend(&self) -> &Arc<Mutex<Arc<dyn Backend<T>>>> {
        &self.backend
    }
}

/// Initialize a Reed context from a resource string.
///
/// Supported resources:
/// - "/cpu/self" or "/cpu/self/ref" -> CPU backend
pub fn init<T: Scalar>(resource: &str) -> ReedResult<Reed<T>> {
    let _resource = resource;
    // Backend registration is done in the reed-cpu crate.
    // This function only provides the lookup path.
    Err(ReedError::BackendNotSupported(format!(
        "No backend registered for resource '{}'. \
         Use Reed::from_backend() or enable the appropriate backend crate.",
        resource
    )))
}
