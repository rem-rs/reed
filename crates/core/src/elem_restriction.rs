use crate::{
    csr::CsrPattern,
    enums::TransposeMode,
    error::{ReedError, ReedResult},
    scalar::Scalar,
};

/// Degree-of-freedom restriction operator trait.
#[cfg(not(target_arch = "wasm32"))]
pub trait ElemRestrictionTrait<T: Scalar>: Send + Sync {
    fn num_elements(&self) -> usize;
    fn num_dof_per_elem(&self) -> usize;
    fn num_global_dof(&self) -> usize;
    fn num_comp(&self) -> usize;
    fn apply(&self, t_mode: TransposeMode, u: &[T], v: &mut [T]) -> ReedResult<()>;
    fn local_size(&self) -> usize {
        self.num_elements() * self.num_dof_per_elem() * self.num_comp()
    }

    /// Assembled-operator sparsity from **offset** element connectivity (`ncomp == 1` v1); default `Err`.
    fn assembled_csr_pattern(&self) -> ReedResult<CsrPattern> {
        Err(ReedError::ElemRestriction(
            "assembled_csr_pattern is not implemented for this restriction type".into(),
        ))
    }

    /// Clone this restriction into a boxed trait object.
    fn boxed_clone(&self) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Err(ReedError::ElemRestriction(
            "boxed_clone is not implemented for this restriction type".into(),
        ))
    }
}

#[cfg(target_arch = "wasm32")]
pub trait ElemRestrictionTrait<T: Scalar> {
    fn num_elements(&self) -> usize;
    fn num_dof_per_elem(&self) -> usize;
    fn num_global_dof(&self) -> usize;
    fn num_comp(&self) -> usize;
    fn apply(&self, t_mode: TransposeMode, u: &[T], v: &mut [T]) -> ReedResult<()>;
    fn local_size(&self) -> usize {
        self.num_elements() * self.num_dof_per_elem() * self.num_comp()
    }

    fn assembled_csr_pattern(&self) -> ReedResult<CsrPattern> {
        Err(ReedError::ElemRestriction(
            "assembled_csr_pattern is not implemented for this restriction type".into(),
        ))
    }

    /// Clone this restriction into a boxed trait object.
    fn boxed_clone(&self) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Err(ReedError::ElemRestriction(
            "boxed_clone is not implemented for this restriction type".into(),
        ))
    }
}
