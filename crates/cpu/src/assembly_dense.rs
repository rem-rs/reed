//! Dense **global** linear-operator assembly helpers for [`crate::operator::CpuOperator`].
//!
//! libCEED exposes `CeedOperatorLinearAssembleSymbolic` / `LinearAssemble` / `LinearAssembleAdd`
//! into a matrix object.
//! Reed stores an optional dense `n × n` buffer on the operator (`O(n²)` memory) as a migration
//! stepping stone. Callers may drop it with **[`crate::operator::CpuOperator::clear_dense_linear_assembly`]**
//! when the matrix is no longer needed. The numeric entries are **columns of the forward Jacobian** `A e_j` when
//! `apply` is linear in the active unknown (e.g. mass with fixed `qdata`); for nonlinear kernels
//! this is **not** a guaranteed global Jacobian unless documented otherwise.

use reed_core::{scalar::Scalar, ReedError, ReedResult};

#[derive(Debug)]
pub(crate) struct DenseLinearAssemblySlot<T: Scalar> {
    pub(crate) n: usize,
    pub(crate) a: Vec<T>,
    pub(crate) symbolic_done: bool,
    pub(crate) numeric_done: bool,
}

impl<T: Scalar> DenseLinearAssemblySlot<T> {
    pub(crate) fn new_symbolic(n: usize) -> ReedResult<Self> {
        let len = n.checked_mul(n).ok_or_else(|| {
            ReedError::Operator(
                "linear_assemble_symbolic: active global DOF count overflow in n*n".into(),
            )
        })?;
        Ok(Self {
            n,
            a: vec![T::ZERO; len],
            symbolic_done: true,
            numeric_done: false,
        })
    }
}
