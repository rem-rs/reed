//! Sum of sub-operators, analogous to libCEED’s `CeedCompositeOperator` additive apply.
//!
//! Each sub-operator must map the same global input/output vector space (same `VectorTrait::len()`).
//! If any sub-operator sets [`OperatorTrait::requires_field_named_buffers`], use
//! [`OperatorTrait::apply_field_buffers`] / [`OperatorTrait::apply_field_buffers_with_transpose`]
//! on the composite (single-buffer `apply` is then unavailable).
//! [`OperatorTrait::apply_with_transpose`] with [`OperatorTransposeRequest::Adjoint`] applies each
//! sub-operator’s adjoint and **sums** the results (same global dual space as the forward sum).
//!
//! **Assemble hooks:** [`OperatorTrait::operator_supports_assemble`] is the **conjunction** of each
//! sub-operator’s probe. [`OperatorTrait::linear_assemble_symbolic`] / [`OperatorTrait::linear_assemble`] /
//! [`OperatorTrait::linear_assemble_add`] / [`OperatorTrait::linear_assemble_csr_matrix`] /
//! [`OperatorTrait::linear_assemble_csr_matrix_add`] are **not** supported on composites (no shared matrix sink);
//! [`OperatorTrait::linear_assemble_add_diagonal`] **is** supported and **sums** sub-operator contributions.
//! Assemble each sub-operator separately for dense / CSR matrix sinks.
//! `OperatorAssembleKind::FdmElementInverse` and **`LinearCsrNumeric`** are **`false`** on composites
//! (even if each sub-operator would support CSR assembly), and [`OperatorTrait::operator_create_fdm_element_inverse`]
//! always returns an error — additive composites have no single assembled Jacobian to invert in Reed.
//!
//! ## `Box<dyn>` vs borrowed composites
//!
//! [`CompositeOperator`] owns `Box<dyn OperatorTrait<T>>` (implicitly `'static`), matching
//! libCEED-style composition when sub-operators own their data.
//!
//! [`CompositeOperatorBorrowed`] holds `&dyn OperatorTrait<T>` so you can sum [`CpuOperator`]
//! values that borrow mesh objects (`restriction`, `basis`) in the same scope—closer to
//! composing `CeedOperator` handles that share a `Ceed` context.

use reed_core::{
    csr::{CsrMatrix, CsrPattern},
    error::{ReedError, ReedResult},
    matrix::CeedMatrix,
    operator::{OperatorAssembleKind, OperatorTrait, OperatorTransposeRequest},
    scalar::Scalar,
    vector::VectorTrait,
};

use crate::vector::CpuVector;

fn validate_composite_suboperators<T: Scalar>(ops: &[&dyn OperatorTrait<T>]) -> ReedResult<()> {
    merge_vector_len_hints_strict(ops.iter().map(|o| o.global_vector_len_hint()))?;
    Ok(())
}

fn merge_vector_len_hints_strict(hints: impl Iterator<Item = Option<usize>>) -> ReedResult<()> {
    let mut expected: Option<usize> = None;
    for h in hints {
        if let Some(n) = h {
            match expected {
                None => expected = Some(n),
                Some(e) if e != n => {
                    return Err(ReedError::Operator(format!(
                        "composite sub-operators disagree on global vector length: {} vs {}",
                        e, n
                    )));
                }
                Some(_) => {}
            }
        }
    }
    Ok(())
}

fn merged_vector_len_hint(hints: impl Iterator<Item = Option<usize>>) -> Option<usize> {
    let mut out: Option<usize> = None;
    for n_opt in hints {
        if let Some(n) = n_opt {
            match out {
                None => out = Some(n),
                Some(prev) if prev != n => return None,
                Some(_) => {}
            }
        }
    }
    out
}

/// `y = sum_i A_i x` for [`OperatorTrait::apply`]; [`OperatorTrait::apply_add`] accumulates all sub-operators.
pub struct CompositeOperator<T: Scalar> {
    ops: Vec<Box<dyn OperatorTrait<T>>>,
}

impl<T: Scalar> CompositeOperator<T> {
    pub fn new(ops: Vec<Box<dyn OperatorTrait<T>>>) -> ReedResult<Self> {
        if ops.is_empty() {
            return Err(ReedError::Operator(
                "CompositeOperator requires at least one sub-operator".into(),
            ));
        }
        let refs: Vec<&dyn OperatorTrait<T>> = ops.iter().map(|b| b.as_ref()).collect();
        validate_composite_suboperators(&refs)?;
        Ok(Self { ops })
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl<T: Scalar> OperatorTrait<T> for CompositeOperator<T> {
    fn global_vector_len_hint(&self) -> Option<usize> {
        merged_vector_len_hint(self.ops.iter().map(|o| o.global_vector_len_hint()))
    }

    fn check_ready(&self) -> ReedResult<()> {
        for op in &self.ops {
            op.check_ready()?;
        }
        Ok(())
    }

    fn apply(&self, input: &dyn VectorTrait<T>, output: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        if self.requires_field_named_buffers() {
            return Err(ReedError::Operator(
                "CompositeOperator: at least one sub-operator requires field-named buffers; use apply_field_buffers"
                    .into(),
            ));
        }
        output.set_value(T::ZERO)?;
        for op in &self.ops {
            op.apply_add(input, output)?;
        }
        Ok(())
    }

    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        if self.requires_field_named_buffers() {
            return Err(ReedError::Operator(
                "CompositeOperator: at least one sub-operator requires field-named buffers; use apply_add_field_buffers"
                    .into(),
            ));
        }
        for op in &self.ops {
            op.apply_add(input, output)?;
        }
        Ok(())
    }

    fn apply_field_buffers<'io>(
        &self,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        for (_, out) in outputs.iter_mut() {
            out.set_value(T::ZERO)?;
        }
        let outputs_ptr: *mut [(&'io str, &'io mut dyn VectorTrait<T>)] = outputs;
        for op in &self.ops {
            // SAFETY: calls are strictly sequential; no `out_short` escapes this loop body.
            let out_short = unsafe { &mut *outputs_ptr };
            op.apply_add_field_buffers(inputs, out_short)?;
        }
        Ok(())
    }

    fn apply_add_field_buffers<'io>(
        &self,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        let outputs_ptr: *mut [(&'io str, &'io mut dyn VectorTrait<T>)] = outputs;
        for op in &self.ops {
            // SAFETY: calls are strictly sequential; no `out_short` escapes this loop body.
            let out_short = unsafe { &mut *outputs_ptr };
            op.apply_add_field_buffers(inputs, out_short)?;
        }
        Ok(())
    }

    fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        let n = assembled.len();
        assembled.set_value(T::ZERO)?;
        let mut tmp = CpuVector::new(n);
        for op in &self.ops {
            tmp.set_value(T::ZERO)?;
            op.linear_assemble_diagonal(&mut tmp)?;
            for i in 0..n {
                assembled.as_mut_slice()[i] = assembled.as_slice()[i] + tmp.as_slice()[i];
            }
        }
        Ok(())
    }

    fn linear_assemble_add_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        for op in &self.ops {
            op.linear_assemble_add_diagonal(assembled)?;
        }
        Ok(())
    }

    fn apply_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply(input, output),
            OperatorTransposeRequest::Adjoint => {
                if self.requires_field_named_buffers() {
                    return Err(ReedError::Operator(
                        "CompositeOperator: at least one sub-operator requires field-named buffers; use apply_field_buffers_with_transpose(Adjoint)"
                            .into(),
                    ));
                }
                output.set_value(T::ZERO)?;
                for op in &self.ops {
                    op.apply_add_with_transpose(OperatorTransposeRequest::Adjoint, input, output)?;
                }
                Ok(())
            }
        }
    }

    fn apply_add_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_add(input, output),
            OperatorTransposeRequest::Adjoint => {
                if self.requires_field_named_buffers() {
                    return Err(ReedError::Operator(
                        "CompositeOperator: at least one sub-operator requires field-named buffers; use apply_add_field_buffers_with_transpose(Adjoint)"
                            .into(),
                    ));
                }
                for op in &self.ops {
                    op.apply_add_with_transpose(OperatorTransposeRequest::Adjoint, input, output)?;
                }
                Ok(())
            }
        }
    }

    fn apply_field_buffers_with_transpose<'io>(
        &self,
        request: OperatorTransposeRequest,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_field_buffers(inputs, outputs),
            OperatorTransposeRequest::Adjoint => {
                for (_, out) in outputs.iter_mut() {
                    out.set_value(T::ZERO)?;
                }
                let outputs_ptr: *mut [(&'io str, &'io mut dyn VectorTrait<T>)] = outputs;
                for op in &self.ops {
                    // SAFETY: calls are strictly sequential; no `out_short` escapes this loop body.
                    let out_short = unsafe { &mut *outputs_ptr };
                    op.apply_add_field_buffers_with_transpose(
                        OperatorTransposeRequest::Adjoint,
                        inputs,
                        out_short,
                    )?;
                }
                Ok(())
            }
        }
    }

    fn apply_add_field_buffers_with_transpose<'io>(
        &self,
        request: OperatorTransposeRequest,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_add_field_buffers(inputs, outputs),
            OperatorTransposeRequest::Adjoint => {
                let outputs_ptr: *mut [(&'io str, &'io mut dyn VectorTrait<T>)] = outputs;
                for op in &self.ops {
                    // SAFETY: calls are strictly sequential; no `out_short` escapes this loop body.
                    let out_short = unsafe { &mut *outputs_ptr };
                    op.apply_add_field_buffers_with_transpose(
                        OperatorTransposeRequest::Adjoint,
                        inputs,
                        out_short,
                    )?;
                }
                Ok(())
            }
        }
    }

    fn requires_field_named_buffers(&self) -> bool {
        self.ops.iter().any(|o| o.requires_field_named_buffers())
    }

    fn operator_supports_assemble(&self, kind: OperatorAssembleKind) -> bool {
        match kind {
            OperatorAssembleKind::FdmElementInverse | OperatorAssembleKind::LinearCsrNumeric => {
                false
            }
            _ => self.ops.iter().all(|o| o.operator_supports_assemble(kind)),
        }
    }

    fn linear_assemble_symbolic(&self) -> ReedResult<()> {
        Err(ReedError::Operator(
            "CompositeOperator: linear_assemble_symbolic is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn linear_assemble(&self) -> ReedResult<()> {
        Err(ReedError::Operator(
            "CompositeOperator: linear_assemble is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn linear_assemble_add(&self) -> ReedResult<()> {
        Err(ReedError::Operator(
            "CompositeOperator: linear_assemble_add is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn linear_assemble_csr_matrix(&self, _pattern: &CsrPattern) -> ReedResult<CsrMatrix<T>> {
        Err(ReedError::Operator(
            "CompositeOperator: linear_assemble_csr_matrix is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn linear_assemble_csr_matrix_add(&self, _matrix: &mut CsrMatrix<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "CompositeOperator: linear_assemble_csr_matrix_add is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn linear_assemble_ceed_matrix(&self, _matrix: &mut CeedMatrix<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "CompositeOperator: linear_assemble_ceed_matrix is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn linear_assemble_add_ceed_matrix(&self, _matrix: &mut CeedMatrix<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "CompositeOperator: linear_assemble_add_ceed_matrix is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn operator_create_fdm_element_inverse(&self) -> ReedResult<Box<dyn OperatorTrait<T>>> {
        Err(ReedError::Operator(
            "CompositeOperator: operator_create_fdm_element_inverse is not supported (use a single-element operator)"
                .into(),
        ))
    }

    fn operator_create_fdm_element_inverse_jacobi(&self) -> ReedResult<Box<dyn OperatorTrait<T>>> {
        Err(ReedError::Operator(
            "CompositeOperator: operator_create_fdm_element_inverse_jacobi is not supported (use a single-element operator)"
                .into(),
        ))
    }
}

/// Same semantics as [`CompositeOperator`], but stores `&dyn` sub-operators so [`CpuOperator`]
/// with non-`'static` lifetimes can be summed in one stack frame (libCEED-like shared-context composition).
pub struct CompositeOperatorBorrowed<'a, T: Scalar> {
    ops: Vec<&'a dyn OperatorTrait<T>>,
}

impl<'a, T: Scalar> CompositeOperatorBorrowed<'a, T> {
    pub fn new(ops: Vec<&'a dyn OperatorTrait<T>>) -> ReedResult<Self> {
        if ops.is_empty() {
            return Err(ReedError::Operator(
                "CompositeOperatorBorrowed requires at least one sub-operator".into(),
            ));
        }
        validate_composite_suboperators(&ops)?;
        Ok(Self { ops })
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl<'a, T: Scalar> OperatorTrait<T> for CompositeOperatorBorrowed<'a, T> {
    fn global_vector_len_hint(&self) -> Option<usize> {
        merged_vector_len_hint(self.ops.iter().map(|o| o.global_vector_len_hint()))
    }

    fn check_ready(&self) -> ReedResult<()> {
        for op in &self.ops {
            op.check_ready()?;
        }
        Ok(())
    }

    fn apply(&self, input: &dyn VectorTrait<T>, output: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        if self.requires_field_named_buffers() {
            return Err(ReedError::Operator(
                "CompositeOperatorBorrowed: at least one sub-operator requires field-named buffers; use apply_field_buffers"
                    .into(),
            ));
        }
        output.set_value(T::ZERO)?;
        for op in &self.ops {
            op.apply_add(input, output)?;
        }
        Ok(())
    }

    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        if self.requires_field_named_buffers() {
            return Err(ReedError::Operator(
                "CompositeOperatorBorrowed: at least one sub-operator requires field-named buffers; use apply_add_field_buffers"
                    .into(),
            ));
        }
        for op in &self.ops {
            op.apply_add(input, output)?;
        }
        Ok(())
    }

    fn apply_field_buffers<'io>(
        &self,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        for (_, out) in outputs.iter_mut() {
            out.set_value(T::ZERO)?;
        }
        let outputs_ptr: *mut [(&'io str, &'io mut dyn VectorTrait<T>)] = outputs;
        for op in &self.ops {
            // SAFETY: calls are strictly sequential; no `out_short` escapes this loop body.
            let out_short = unsafe { &mut *outputs_ptr };
            op.apply_add_field_buffers(inputs, out_short)?;
        }
        Ok(())
    }

    fn apply_add_field_buffers<'io>(
        &self,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        let outputs_ptr: *mut [(&'io str, &'io mut dyn VectorTrait<T>)] = outputs;
        for op in &self.ops {
            // SAFETY: calls are strictly sequential; no `out_short` escapes this loop body.
            let out_short = unsafe { &mut *outputs_ptr };
            op.apply_add_field_buffers(inputs, out_short)?;
        }
        Ok(())
    }

    fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        let n = assembled.len();
        assembled.set_value(T::ZERO)?;
        let mut tmp = CpuVector::new(n);
        for op in &self.ops {
            tmp.set_value(T::ZERO)?;
            op.linear_assemble_diagonal(&mut tmp)?;
            for i in 0..n {
                assembled.as_mut_slice()[i] = assembled.as_slice()[i] + tmp.as_slice()[i];
            }
        }
        Ok(())
    }

    fn linear_assemble_add_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        for op in &self.ops {
            op.linear_assemble_add_diagonal(assembled)?;
        }
        Ok(())
    }

    fn apply_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply(input, output),
            OperatorTransposeRequest::Adjoint => {
                if self.requires_field_named_buffers() {
                    return Err(ReedError::Operator(
                        "CompositeOperatorBorrowed: at least one sub-operator requires field-named buffers; use apply_field_buffers_with_transpose(Adjoint)"
                            .into(),
                    ));
                }
                output.set_value(T::ZERO)?;
                for op in &self.ops {
                    op.apply_add_with_transpose(OperatorTransposeRequest::Adjoint, input, output)?;
                }
                Ok(())
            }
        }
    }

    fn apply_add_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_add(input, output),
            OperatorTransposeRequest::Adjoint => {
                if self.requires_field_named_buffers() {
                    return Err(ReedError::Operator(
                        "CompositeOperatorBorrowed: at least one sub-operator requires field-named buffers; use apply_add_field_buffers_with_transpose(Adjoint)"
                            .into(),
                    ));
                }
                for op in &self.ops {
                    op.apply_add_with_transpose(OperatorTransposeRequest::Adjoint, input, output)?;
                }
                Ok(())
            }
        }
    }

    fn apply_field_buffers_with_transpose<'io>(
        &self,
        request: OperatorTransposeRequest,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_field_buffers(inputs, outputs),
            OperatorTransposeRequest::Adjoint => {
                for (_, out) in outputs.iter_mut() {
                    out.set_value(T::ZERO)?;
                }
                let outputs_ptr: *mut [(&'io str, &'io mut dyn VectorTrait<T>)] = outputs;
                for op in &self.ops {
                    // SAFETY: calls are strictly sequential; no `out_short` escapes this loop body.
                    let out_short = unsafe { &mut *outputs_ptr };
                    op.apply_add_field_buffers_with_transpose(
                        OperatorTransposeRequest::Adjoint,
                        inputs,
                        out_short,
                    )?;
                }
                Ok(())
            }
        }
    }

    fn apply_add_field_buffers_with_transpose<'io>(
        &self,
        request: OperatorTransposeRequest,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_add_field_buffers(inputs, outputs),
            OperatorTransposeRequest::Adjoint => {
                let outputs_ptr: *mut [(&'io str, &'io mut dyn VectorTrait<T>)] = outputs;
                for op in &self.ops {
                    // SAFETY: calls are strictly sequential; no `out_short` escapes this loop body.
                    let out_short = unsafe { &mut *outputs_ptr };
                    op.apply_add_field_buffers_with_transpose(
                        OperatorTransposeRequest::Adjoint,
                        inputs,
                        out_short,
                    )?;
                }
                Ok(())
            }
        }
    }

    fn requires_field_named_buffers(&self) -> bool {
        self.ops.iter().any(|o| o.requires_field_named_buffers())
    }

    fn operator_supports_assemble(&self, kind: OperatorAssembleKind) -> bool {
        match kind {
            OperatorAssembleKind::FdmElementInverse | OperatorAssembleKind::LinearCsrNumeric => {
                false
            }
            _ => self.ops.iter().all(|o| o.operator_supports_assemble(kind)),
        }
    }

    fn linear_assemble_symbolic(&self) -> ReedResult<()> {
        Err(ReedError::Operator(
            "CompositeOperatorBorrowed: linear_assemble_symbolic is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn linear_assemble(&self) -> ReedResult<()> {
        Err(ReedError::Operator(
            "CompositeOperatorBorrowed: linear_assemble is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn linear_assemble_add(&self) -> ReedResult<()> {
        Err(ReedError::Operator(
            "CompositeOperatorBorrowed: linear_assemble_add is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn linear_assemble_csr_matrix(&self, _pattern: &CsrPattern) -> ReedResult<CsrMatrix<T>> {
        Err(ReedError::Operator(
            "CompositeOperatorBorrowed: linear_assemble_csr_matrix is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn linear_assemble_csr_matrix_add(&self, _matrix: &mut CsrMatrix<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "CompositeOperatorBorrowed: linear_assemble_csr_matrix_add is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn linear_assemble_ceed_matrix(&self, _matrix: &mut CeedMatrix<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "CompositeOperatorBorrowed: linear_assemble_ceed_matrix is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn linear_assemble_add_ceed_matrix(&self, _matrix: &mut CeedMatrix<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "CompositeOperatorBorrowed: linear_assemble_add_ceed_matrix is not supported (call on each sub-operator)"
                .into(),
        ))
    }

    fn operator_create_fdm_element_inverse(&self) -> ReedResult<Box<dyn OperatorTrait<T>>> {
        Err(ReedError::Operator(
            "CompositeOperatorBorrowed: operator_create_fdm_element_inverse is not supported (use a single-element operator)"
                .into(),
        ))
    }

    fn operator_create_fdm_element_inverse_jacobi(&self) -> ReedResult<Box<dyn OperatorTrait<T>>> {
        Err(ReedError::Operator(
            "CompositeOperatorBorrowed: operator_create_fdm_element_inverse_jacobi is not supported (use a single-element operator)"
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_lagrange::LagrangeBasis;
    use crate::elem_restriction::CpuElemRestriction;
    use crate::gallery::MassApplyInterpTimesWeight;
    use crate::operator::{FieldVector, OperatorBuilder};
    use crate::vector::CpuVector;
    use reed_core::csr::{CsrMatrix, CsrPattern};
    use reed_core::enums::QuadMode;
    use reed_core::operator::{OperatorAssembleKind, OperatorTransposeRequest};
    use reed_core::qfunction::QFunctionTrait;

    fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// `y += s x`; diagonal of `s I` is `s` on each component.
    struct ScaleOp {
        n: usize,
        scale: f64,
    }

    impl OperatorTrait<f64> for ScaleOp {
        fn global_vector_len_hint(&self) -> Option<usize> {
            Some(self.n)
        }

        fn apply(
            &self,
            input: &dyn VectorTrait<f64>,
            output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            output.set_value(0.0)?;
            self.apply_add(input, output)
        }

        fn apply_add(
            &self,
            input: &dyn VectorTrait<f64>,
            output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            for i in 0..self.n {
                output.as_mut_slice()[i] += self.scale * input.as_slice()[i];
            }
            Ok(())
        }

        fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<f64>) -> ReedResult<()> {
            assembled.set_value(0.0)?;
            for i in 0..self.n {
                assembled.as_mut_slice()[i] = self.scale;
            }
            Ok(())
        }

        fn linear_assemble_add_diagonal(
            &self,
            assembled: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            if assembled.len() != self.n {
                return Err(ReedError::Operator(format!(
                    "linear_assemble_add_diagonal: length {} != {}",
                    assembled.len(),
                    self.n
                )));
            }
            for i in 0..self.n {
                assembled.as_mut_slice()[i] += self.scale;
            }
            Ok(())
        }

        fn apply_with_transpose(
            &self,
            request: OperatorTransposeRequest,
            input: &dyn VectorTrait<f64>,
            output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            match request {
                OperatorTransposeRequest::Forward => self.apply(input, output),
                OperatorTransposeRequest::Adjoint => self.apply(input, output),
            }
        }

        fn apply_add_with_transpose(
            &self,
            request: OperatorTransposeRequest,
            input: &dyn VectorTrait<f64>,
            output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            match request {
                OperatorTransposeRequest::Forward => self.apply_add(input, output),
                OperatorTransposeRequest::Adjoint => self.apply_add(input, output),
            }
        }
    }

    /// Like [`ScaleOp`] but reports **no** supported assemble kinds (diagonal assembly still callable for tests).
    struct NoAssembleKindsOp {
        n: usize,
        scale: f64,
    }

    impl OperatorTrait<f64> for NoAssembleKindsOp {
        fn global_vector_len_hint(&self) -> Option<usize> {
            Some(self.n)
        }

        fn operator_supports_assemble(&self, _kind: OperatorAssembleKind) -> bool {
            false
        }

        fn apply(
            &self,
            input: &dyn VectorTrait<f64>,
            output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            output.set_value(0.0)?;
            self.apply_add(input, output)
        }

        fn apply_add(
            &self,
            input: &dyn VectorTrait<f64>,
            output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            for i in 0..self.n {
                output.as_mut_slice()[i] += self.scale * input.as_slice()[i];
            }
            Ok(())
        }

        fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<f64>) -> ReedResult<()> {
            assembled.set_value(0.0)?;
            for i in 0..self.n {
                assembled.as_mut_slice()[i] = self.scale;
            }
            Ok(())
        }

        fn linear_assemble_add_diagonal(
            &self,
            assembled: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            if assembled.len() != self.n {
                return Err(ReedError::Operator(format!(
                    "linear_assemble_add_diagonal: length {} != {}",
                    assembled.len(),
                    self.n
                )));
            }
            for i in 0..self.n {
                assembled.as_mut_slice()[i] += self.scale;
            }
            Ok(())
        }
    }

    /// Same apply dimension as [`ScaleOp`] but reports a different hint (for validation tests).
    struct MismatchedHintOp {
        n: usize,
        hint: usize,
        scale: f64,
    }

    impl OperatorTrait<f64> for MismatchedHintOp {
        fn global_vector_len_hint(&self) -> Option<usize> {
            Some(self.hint)
        }

        fn apply(
            &self,
            input: &dyn VectorTrait<f64>,
            output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            output.set_value(0.0)?;
            self.apply_add(input, output)
        }

        fn apply_add(
            &self,
            input: &dyn VectorTrait<f64>,
            output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            for i in 0..self.n {
                output.as_mut_slice()[i] += self.scale * input.as_slice()[i];
            }
            Ok(())
        }

        fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<f64>) -> ReedResult<()> {
            assembled.set_value(0.0)?;
            for i in 0..self.n {
                assembled.as_mut_slice()[i] = self.scale;
            }
            Ok(())
        }

        fn linear_assemble_add_diagonal(
            &self,
            assembled: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            if assembled.len() != self.n {
                return Err(ReedError::Operator(format!(
                    "linear_assemble_add_diagonal: length {} != {}",
                    assembled.len(),
                    self.n
                )));
            }
            for i in 0..self.n {
                assembled.as_mut_slice()[i] += self.scale;
            }
            Ok(())
        }
    }

    #[test]
    fn composite_operator_supports_assemble_is_conjunction() {
        let ok = CompositeOperator::new(vec![
            Box::new(ScaleOp { n: 2, scale: 1.0 }) as Box<dyn OperatorTrait<f64>>,
            Box::new(ScaleOp { n: 2, scale: 2.0 }),
        ])
        .unwrap();
        assert!(ok.operator_supports_assemble(OperatorAssembleKind::Diagonal));
        assert!(!ok.operator_supports_assemble(OperatorAssembleKind::LinearSymbolic));
        assert!(!ok.operator_supports_assemble(OperatorAssembleKind::LinearCsrNumeric));
        assert!(!ok.operator_supports_assemble(OperatorAssembleKind::FdmElementInverse));

        let bad = CompositeOperator::new(vec![
            Box::new(ScaleOp { n: 2, scale: 1.0 }) as Box<dyn OperatorTrait<f64>>,
            Box::new(NoAssembleKindsOp { n: 2, scale: 1.0 }),
        ])
        .unwrap();
        assert!(!bad.operator_supports_assemble(OperatorAssembleKind::Diagonal));
    }

    #[test]
    fn composite_operator_create_fdm_element_inverse_errors() {
        let c = CompositeOperator::new(vec![
            Box::new(ScaleOp { n: 2, scale: 1.0 }) as Box<dyn OperatorTrait<f64>>
        ])
        .unwrap();
        let msg = match c.operator_create_fdm_element_inverse() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected FDM inverse on composite to fail"),
        };
        assert!(
            msg.contains("CompositeOperator") && msg.contains("not supported"),
            "{msg}"
        );
    }

    #[test]
    fn composite_operator_create_fdm_element_inverse_jacobi_errors() {
        let c = CompositeOperator::new(vec![
            Box::new(ScaleOp { n: 2, scale: 1.0 }) as Box<dyn OperatorTrait<f64>>
        ])
        .unwrap();
        let msg = match c.operator_create_fdm_element_inverse_jacobi() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected Jacobi FDM inverse on composite to fail"),
        };
        assert!(
            msg.contains("CompositeOperator")
                && msg.contains("operator_create_fdm_element_inverse_jacobi"),
            "{msg}"
        );
    }

    #[test]
    fn composite_linear_assemble_add_errors() {
        let c = CompositeOperator::new(vec![
            Box::new(ScaleOp { n: 2, scale: 1.0 }) as Box<dyn OperatorTrait<f64>>
        ])
        .unwrap();
        let msg = c.linear_assemble_add().unwrap_err().to_string();
        assert!(
            msg.contains("CompositeOperator") && msg.contains("linear_assemble_add"),
            "{msg}"
        );
    }

    #[test]
    fn composite_linear_assemble_csr_matrix_add_errors() {
        let c = CompositeOperator::new(vec![
            Box::new(ScaleOp { n: 2, scale: 1.0 }) as Box<dyn OperatorTrait<f64>>
        ])
        .unwrap();
        let pat = CsrPattern {
            nrows: 2,
            ncols: 2,
            row_ptr: vec![0, 1, 2],
            col_ind: vec![0, 1],
        };
        let mut m = CsrMatrix {
            pattern: pat,
            values: vec![0.0_f64, 0.0_f64],
        };
        let msg = c
            .linear_assemble_csr_matrix_add(&mut m)
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("CompositeOperator") && msg.contains("linear_assemble_csr_matrix_add"),
            "{msg}"
        );
    }

    #[test]
    fn composite_linear_assemble_ceed_matrix_errors() {
        let c = CompositeOperator::new(vec![
            Box::new(ScaleOp { n: 2, scale: 1.0 }) as Box<dyn OperatorTrait<f64>>
        ])
        .unwrap();
        let mut dense = CeedMatrix::<f64>::dense_col_major_symbolic(2, 2).unwrap();
        let msg = c
            .linear_assemble_ceed_matrix(&mut dense)
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("CompositeOperator") && msg.contains("linear_assemble_ceed_matrix"),
            "{msg}"
        );
    }

    #[test]
    fn composite_borrowed_linear_assemble_add_errors() {
        let a = ScaleOp { n: 2, scale: 1.0 };
        let b = ScaleOp { n: 2, scale: 2.0 };
        let c = CompositeOperatorBorrowed::new(vec![
            &a as &dyn OperatorTrait<f64>,
            &b as &dyn OperatorTrait<f64>,
        ])
        .unwrap();
        let msg = c.linear_assemble_add().unwrap_err().to_string();
        assert!(
            msg.contains("CompositeOperatorBorrowed") && msg.contains("linear_assemble_add"),
            "{msg}"
        );
    }

    #[test]
    fn composite_borrowed_linear_assemble_csr_matrix_add_errors() {
        let a = ScaleOp { n: 2, scale: 1.0 };
        let b = ScaleOp { n: 2, scale: 2.0 };
        let c = CompositeOperatorBorrowed::new(vec![
            &a as &dyn OperatorTrait<f64>,
            &b as &dyn OperatorTrait<f64>,
        ])
        .unwrap();
        let pat = CsrPattern {
            nrows: 2,
            ncols: 2,
            row_ptr: vec![0, 1, 2],
            col_ind: vec![0, 1],
        };
        let mut m = CsrMatrix {
            pattern: pat,
            values: vec![0.0_f64, 0.0_f64],
        };
        let msg = c
            .linear_assemble_csr_matrix_add(&mut m)
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("CompositeOperatorBorrowed")
                && msg.contains("linear_assemble_csr_matrix_add"),
            "{msg}"
        );
    }

    #[test]
    fn composite_borrowed_linear_assemble_add_ceed_matrix_errors() {
        let a = ScaleOp { n: 2, scale: 1.0 };
        let b = ScaleOp { n: 2, scale: 2.0 };
        let c = CompositeOperatorBorrowed::new(vec![
            &a as &dyn OperatorTrait<f64>,
            &b as &dyn OperatorTrait<f64>,
        ])
        .unwrap();
        let mut dense = CeedMatrix::<f64>::dense_col_major_symbolic(2, 2).unwrap();
        let msg = c
            .linear_assemble_add_ceed_matrix(&mut dense)
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("CompositeOperatorBorrowed")
                && msg.contains("linear_assemble_add_ceed_matrix"),
            "{msg}"
        );
    }

    #[test]
    fn composite_borrowed_create_fdm_element_inverse_jacobi_errors() {
        let a = ScaleOp { n: 2, scale: 1.0 };
        let b = ScaleOp { n: 2, scale: 2.0 };
        let c = CompositeOperatorBorrowed::new(vec![
            &a as &dyn OperatorTrait<f64>,
            &b as &dyn OperatorTrait<f64>,
        ])
        .unwrap();
        let msg = match c.operator_create_fdm_element_inverse_jacobi() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected Jacobi FDM inverse on borrowed composite to fail"),
        };
        assert!(
            msg.contains("CompositeOperatorBorrowed")
                && msg.contains("operator_create_fdm_element_inverse_jacobi"),
            "{msg}"
        );
    }

    #[test]
    fn composite_apply_transpose_adjoint_sums_suboperators() {
        let c = CompositeOperator::new(vec![
            Box::new(ScaleOp { n: 2, scale: 0.5 }) as Box<dyn OperatorTrait<f64>>,
            Box::new(ScaleOp { n: 2, scale: 1.5 }),
        ])
        .unwrap();
        let x = CpuVector::from_vec(vec![1.0, -2.0]);
        let mut y = CpuVector::new(2);
        c.apply_with_transpose(OperatorTransposeRequest::Adjoint, &x, &mut y)
            .unwrap();
        assert!((y.as_slice()[0] - 2.0).abs() < 1e-14);
        assert!((y.as_slice()[1] + 4.0).abs() < 1e-14);
    }

    #[test]
    fn composite_new_rejects_empty() {
        assert!(matches!(
            CompositeOperator::<f64>::new(vec![]),
            Err(ReedError::Operator(_))
        ));
    }

    /// Stub operator that only supports per-field buffers.
    struct NamedBuffersOnlyOp {
        scale: f64,
    }

    impl OperatorTrait<f64> for NamedBuffersOnlyOp {
        fn requires_field_named_buffers(&self) -> bool {
            true
        }

        fn global_vector_len_hint(&self) -> Option<usize> {
            Some(3)
        }

        fn apply(
            &self,
            _input: &dyn VectorTrait<f64>,
            _output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            Err(ReedError::Operator(
                "NamedBuffersOnlyOp: use apply_field_buffers".into(),
            ))
        }

        fn apply_add(
            &self,
            _input: &dyn VectorTrait<f64>,
            _output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            Err(ReedError::Operator(
                "NamedBuffersOnlyOp: use apply_field_buffers".into(),
            ))
        }

        fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<f64>) -> ReedResult<()> {
            assembled.set_value(0.0)?;
            Ok(())
        }

        fn linear_assemble_add_diagonal(
            &self,
            _assembled: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            Ok(())
        }

        fn apply_field_buffers<'io>(
            &self,
            inputs: &'io [(&'io str, &'io dyn VectorTrait<f64>)],
            outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<f64>)],
        ) -> ReedResult<()> {
            let (_, x) = inputs
                .iter()
                .find(|(name, _)| *name == "x")
                .ok_or_else(|| {
                    ReedError::Operator("NamedBuffersOnlyOp: missing input 'x'".into())
                })?;
            let (_, y) = outputs
                .iter_mut()
                .find(|(name, _)| *name == "x")
                .ok_or_else(|| {
                    ReedError::Operator("NamedBuffersOnlyOp: missing output 'x'".into())
                })?;
            if x.len() != y.len() {
                return Err(ReedError::Operator(
                    "NamedBuffersOnlyOp: input/output length mismatch".into(),
                ));
            }
            y.set_value(0.0)?;
            for i in 0..x.len() {
                y.as_mut_slice()[i] += self.scale * x.as_slice()[i];
            }
            Ok(())
        }

        fn apply_add_field_buffers<'io>(
            &self,
            inputs: &'io [(&'io str, &'io dyn VectorTrait<f64>)],
            outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<f64>)],
        ) -> ReedResult<()> {
            let (_, x) = inputs
                .iter()
                .find(|(name, _)| *name == "x")
                .ok_or_else(|| {
                    ReedError::Operator("NamedBuffersOnlyOp: missing input 'x'".into())
                })?;
            let (_, y) = outputs
                .iter_mut()
                .find(|(name, _)| *name == "x")
                .ok_or_else(|| {
                    ReedError::Operator("NamedBuffersOnlyOp: missing output 'x'".into())
                })?;
            if x.len() != y.len() {
                return Err(ReedError::Operator(
                    "NamedBuffersOnlyOp: input/output length mismatch".into(),
                ));
            }
            for i in 0..x.len() {
                y.as_mut_slice()[i] += self.scale * x.as_slice()[i];
            }
            Ok(())
        }

        fn apply_field_buffers_with_transpose<'io>(
            &self,
            request: OperatorTransposeRequest,
            inputs: &'io [(&'io str, &'io dyn VectorTrait<f64>)],
            outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<f64>)],
        ) -> ReedResult<()> {
            match request {
                OperatorTransposeRequest::Forward => self.apply_field_buffers(inputs, outputs),
                OperatorTransposeRequest::Adjoint => self.apply_field_buffers(inputs, outputs),
            }
        }

        fn apply_add_field_buffers_with_transpose<'io>(
            &self,
            request: OperatorTransposeRequest,
            inputs: &'io [(&'io str, &'io dyn VectorTrait<f64>)],
            outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<f64>)],
        ) -> ReedResult<()> {
            match request {
                OperatorTransposeRequest::Forward => self.apply_add_field_buffers(inputs, outputs),
                OperatorTransposeRequest::Adjoint => self.apply_add_field_buffers(inputs, outputs),
            }
        }
    }

    #[test]
    fn composite_new_accepts_suboperator_named_buffers() {
        let comp = CompositeOperator::new(vec![
            Box::new(ScaleOp { n: 3, scale: 1.0 }) as Box<dyn OperatorTrait<f64>>,
            Box::new(NamedBuffersOnlyOp { scale: 2.0 }),
        ])
        .unwrap();
        assert!(comp.requires_field_named_buffers());

        let a = ScaleOp { n: 3, scale: 1.0 };
        let b = NamedBuffersOnlyOp { scale: 2.0 };
        let comp_b = CompositeOperatorBorrowed::new(vec![
            &a as &dyn OperatorTrait<f64>,
            &b as &dyn OperatorTrait<f64>,
        ])
        .unwrap();
        assert!(comp_b.requires_field_named_buffers());
    }

    #[test]
    fn composite_named_field_buffers_forward_and_adjoint_sum_suboperators() {
        let comp = CompositeOperator::new(vec![
            Box::new(NamedBuffersOnlyOp { scale: 0.5 }) as Box<dyn OperatorTrait<f64>>,
            Box::new(NamedBuffersOnlyOp { scale: 1.5 }),
        ])
        .unwrap();

        let x = CpuVector::from_vec(vec![1.0, -2.0, 3.0]);
        let mut y = CpuVector::new(3);
        let in_named = [("x", &x as &dyn VectorTrait<f64>)];
        let mut out_named = [("x", &mut y as &mut dyn VectorTrait<f64>)];

        comp.apply_field_buffers(&in_named, &mut out_named).unwrap();
        assert_eq!(y.as_slice(), &[2.0, -4.0, 6.0]);

        let mut dx = CpuVector::new(3);
        let in_adj = [("x", &x as &dyn VectorTrait<f64>)];
        let mut out_adj = [("x", &mut dx as &mut dyn VectorTrait<f64>)];
        comp.apply_field_buffers_with_transpose(
            OperatorTransposeRequest::Adjoint,
            &in_adj,
            &mut out_adj,
        )
        .unwrap();
        assert_eq!(dx.as_slice(), &[2.0, -4.0, 6.0]);
    }

    #[test]
    fn composite_borrowed_mass_interp_times_weight_named_adjoint_inner_product() -> ReedResult<()> {
        let nelem = 2usize;
        let p = 2usize;
        let q = 2usize;
        let ndofs = nelem + 1;
        let ind = vec![0i32, 1, 1, 2];
        let r = CpuElemRestriction::<f64>::new_offset(nelem, p, 1, 1, ndofs, &ind)?;
        let b = LagrangeBasis::new(1, 1, p, q, QuadMode::Gauss)?;
        let passive_dummy = CpuVector::from_vec(vec![0.0_f64]);

        let op_a = OperatorBuilder::new()
            .qfunction(
                Box::new(MassApplyInterpTimesWeight::default()) as Box<dyn QFunctionTrait<f64>>
            )
            .field("u", Some(&r), Some(&b), FieldVector::Active)
            .field("w", None, Some(&b), FieldVector::Passive(&passive_dummy))
            .field("v", Some(&r), Some(&b), FieldVector::Active)
            .build()?;
        let op_b = OperatorBuilder::new()
            .qfunction(
                Box::new(MassApplyInterpTimesWeight::default()) as Box<dyn QFunctionTrait<f64>>
            )
            .field("u", Some(&r), Some(&b), FieldVector::Active)
            .field("w", None, Some(&b), FieldVector::Passive(&passive_dummy))
            .field("v", Some(&r), Some(&b), FieldVector::Active)
            .build()?;
        let comp = CompositeOperatorBorrowed::new(vec![
            &op_a as &dyn OperatorTrait<f64>,
            &op_b as &dyn OperatorTrait<f64>,
        ])?;

        let u = CpuVector::from_vec(vec![1.0, 0.5, -0.25]);
        let w = CpuVector::from_vec(vec![0.3, -0.7, 0.05]);

        let mut v = CpuVector::new(ndofs);
        v.set_value(0.0)?;
        let ins = [("u", &u as &dyn VectorTrait<f64>)];
        let mut outs = [("v", &mut v as &mut dyn VectorTrait<f64>)];
        comp.apply_field_buffers(&ins, &mut outs)?;

        let mut du = CpuVector::new(ndofs);
        du.set_value(0.0)?;
        let range_in = [("v", &w as &dyn VectorTrait<f64>)];
        let mut domain_out = [("u", &mut du as &mut dyn VectorTrait<f64>)];
        comp.apply_field_buffers_with_transpose(
            OperatorTransposeRequest::Adjoint,
            &range_in,
            &mut domain_out,
        )?;

        let lhs = dot_f64(v.as_slice(), w.as_slice());
        let rhs = dot_f64(u.as_slice(), du.as_slice());
        assert!(
            (lhs - rhs).abs() < 1e-9_f64.max(1e-9 * lhs.abs()),
            "composite borrowed inner product identity (interp × qp-weight passive): lhs={lhs} rhs={rhs}"
        );
        Ok(())
    }

    #[test]
    fn composite_new_rejects_mismatched_hints() {
        let a = MismatchedHintOp {
            n: 3,
            hint: 3,
            scale: 1.0,
        };
        let b = MismatchedHintOp {
            n: 3,
            hint: 4,
            scale: 1.0,
        };
        assert!(matches!(
            CompositeOperator::new(vec![
                Box::new(a) as Box<dyn OperatorTrait<f64>>,
                Box::new(b) as Box<dyn OperatorTrait<f64>>,
            ]),
            Err(ReedError::Operator(_))
        ));
    }

    struct FailCheckOp;

    impl OperatorTrait<f64> for FailCheckOp {
        fn global_vector_len_hint(&self) -> Option<usize> {
            Some(3)
        }

        fn check_ready(&self) -> ReedResult<()> {
            Err(ReedError::Operator("check_ready failure".into()))
        }

        fn apply(
            &self,
            _input: &dyn VectorTrait<f64>,
            _output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            Ok(())
        }

        fn apply_add(
            &self,
            _input: &dyn VectorTrait<f64>,
            _output: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            Ok(())
        }

        fn linear_assemble_diagonal(
            &self,
            _assembled: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            Ok(())
        }

        fn linear_assemble_add_diagonal(
            &self,
            _assembled: &mut dyn VectorTrait<f64>,
        ) -> ReedResult<()> {
            Ok(())
        }
    }

    #[test]
    fn composite_check_ready_propagates_sub_error() {
        let comp = CompositeOperator::new(vec![
            Box::new(ScaleOp { n: 3, scale: 1.0 }) as Box<dyn OperatorTrait<f64>>,
            Box::new(FailCheckOp),
        ])
        .unwrap();
        assert!(matches!(comp.check_ready(), Err(ReedError::Operator(_))));
    }

    #[test]
    fn composite_apply_sums_suboperators() {
        let n = 3;
        let ops: Vec<Box<dyn OperatorTrait<f64>>> = vec![
            Box::new(ScaleOp { n, scale: 2.0 }),
            Box::new(ScaleOp { n, scale: 3.0 }),
        ];
        let comp = CompositeOperator::new(ops).unwrap();
        let x = CpuVector::from_vec(vec![1.0, 2.0, 3.0]);
        let mut y = CpuVector::new(n);
        comp.apply(&x, &mut y).unwrap();
        assert_eq!(y.as_slice(), &[5.0, 10.0, 15.0]);
    }

    #[test]
    fn composite_borrowed_apply_sums_suboperators() {
        let n = 2usize;
        let a = ScaleOp { n, scale: 1.0 };
        let b = ScaleOp { n, scale: 2.0 };
        let comp = CompositeOperatorBorrowed::new(vec![
            &a as &dyn OperatorTrait<f64>,
            &b as &dyn OperatorTrait<f64>,
        ])
        .unwrap();
        let x = CpuVector::from_vec(vec![1.0, 1.0]);
        let mut y = CpuVector::new(n);
        comp.apply(&x, &mut y).unwrap();
        assert_eq!(y.as_slice(), &[3.0, 3.0]);
    }

    #[test]
    fn composite_diagonal_sums_suboperators() {
        let n = 2;
        let ops: Vec<Box<dyn OperatorTrait<f64>>> = vec![
            Box::new(ScaleOp { n, scale: 1.0 }),
            Box::new(ScaleOp { n, scale: 4.0 }),
        ];
        let comp = CompositeOperator::new(ops).unwrap();
        let mut d = CpuVector::new(n);
        comp.linear_assemble_diagonal(&mut d).unwrap();
        assert_eq!(d.as_slice(), &[5.0, 5.0]);
    }

    #[test]
    fn composite_linear_assemble_add_diagonal_accumulates() {
        let n = 2;
        let comp = CompositeOperator::new(vec![
            Box::new(ScaleOp { n, scale: 2.0 }) as Box<dyn OperatorTrait<f64>>,
            Box::new(ScaleOp { n, scale: 3.0 }),
        ])
        .unwrap();
        let mut v = CpuVector::from_vec(vec![10.0, 20.0]);
        comp.linear_assemble_add_diagonal(&mut v).unwrap();
        assert!((v.as_slice()[0] - 15.0).abs() < 1e-14);
        assert!((v.as_slice()[1] - 25.0).abs() < 1e-14);
    }
}
