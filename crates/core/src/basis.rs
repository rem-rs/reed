use crate::{enums::EvalMode, error::ReedResult, scalar::Scalar};

/// Reference-element basis function trait.
///
/// On WASM targets, the `Send + Sync` bounds are omitted because the wgpu
/// Device stored in WgpuBasis is not thread-safe on WASM.
#[cfg(not(target_arch = "wasm32"))]
pub trait BasisTrait<T: Scalar>: Send + Sync {
    /// Topological dimension.
    fn dim(&self) -> usize;

    /// Number of dofs per element (interpolation nodes).
    fn num_dof(&self) -> usize;

    /// Number of quadrature points per element.
    fn num_qpoints(&self) -> usize;

    /// Number of components (1 for scalar fields).
    fn num_comp(&self) -> usize;

    /// Apply basis operator.
    ///
    /// - `transpose = false`: forward, u_local -> v_qpt
    /// - `transpose = true` : transpose, v_qpt -> u_local
    /// - `eval_mode`: requested evaluation type
    /// - `num_elem`: number of elements processed in batch
    fn apply(
        &self,
        num_elem: usize,
        transpose: bool,
        eval_mode: EvalMode,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<()>;

    /// Quadrature weights (length = num_qpoints()).
    fn q_weights(&self) -> &[T];

    /// Quadrature point coordinates (reference element, row-major [nqpts x dim]).
    fn q_ref(&self) -> &[T];

    /// Tensor-product FDM: if this basis supports fast diagonalization, return
    /// `(interp_1d, grad_1d, weights_1d, p, q)` where each slice is 1D data.
    /// Default `None`.
    fn tensor_fdm_1d_data(&self) -> Option<(&[T], &[T], &[T], usize, usize)> {
        None
    }

    /// Optional face quadrature weights for the given local face number.
    /// Default `None` (no per-face quadrature).
    fn face_q_weights(&self, _local_face: usize) -> Option<Vec<T>> {
        None
    }

    /// Optional face quadrature point coordinates (row-major, face-local
    /// dimension) for the given local face number.  Default `None`.
    fn face_q_ref(&self, _local_face: usize) -> Option<Vec<T>> {
        None
    }
}

/// WASM variant without Send+Sync bounds.
#[cfg(target_arch = "wasm32")]
pub trait BasisTrait<T: Scalar> {
    fn dim(&self) -> usize;
    fn num_dof(&self) -> usize;
    fn num_qpoints(&self) -> usize;
    fn num_comp(&self) -> usize;
    fn apply(
        &self,
        num_elem: usize,
        transpose: bool,
        eval_mode: EvalMode,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<()>;
    fn q_weights(&self) -> &[T];
    fn q_ref(&self) -> &[T];

    /// Tensor-product FDM: if this basis supports fast diagonalization, return
    /// `(interp_1d, grad_1d, weights_1d, p, q)` where each slice is 1D data.
    /// Default `None`.
    fn tensor_fdm_1d_data(&self) -> Option<(&[T], &[T], &[T], usize, usize)> {
        None
    }

    /// Optional face quadrature weights for the given local face number.
    /// Default `None` (no per-face quadrature).
    fn face_q_weights(&self, _local_face: usize) -> Option<Vec<T>> {
        None
    }

    /// Optional face quadrature point coordinates (row-major, face-local
    /// dimension) for the given local face number.  Default `None`.
    fn face_q_ref(&self, _local_face: usize) -> Option<Vec<T>> {
        None
    }
}
