//! Discrete weak-form operator surface (libCEED `CeedOperator` migration target).
//!
//! **Implemented here (trait):** `apply` / `apply_add`, `apply_with_transpose` / `apply_add_with_transpose`
//! (`Forward` delegates; `Adjoint` default `Err`), `apply_field_buffers` / `apply_add_field_buffers`,
//! `apply_field_buffers_with_transpose` / `apply_add_field_buffers_with_transpose` (default stubs
//! for multi-field apply), `operator_label`, `linear_assemble_diagonal` / `linear_assemble_add_diagonal`, optional `global_vector_len_hint`
//! for composite validation, `check_ready` (`CeedOperatorCheckReady`), and **libCEED-shaped stubs**
//! [`operator_supports_assemble`](OperatorTrait::operator_supports_assemble),
//! [`linear_assemble_symbolic`](OperatorTrait::linear_assemble_symbolic),
//! [`linear_assemble`](OperatorTrait::linear_assemble),
//! [`linear_assemble_add`](OperatorTrait::linear_assemble_add),
//! [`linear_assemble_csr_matrix`](OperatorTrait::linear_assemble_csr_matrix),
//! [`linear_assemble_csr_matrix_add`](OperatorTrait::linear_assemble_csr_matrix_add),
//! [`linear_assemble_ceed_matrix`](OperatorTrait::linear_assemble_ceed_matrix),
//! [`linear_assemble_add_ceed_matrix`](OperatorTrait::linear_assemble_add_ceed_matrix),
//! [`operator_create_fdm_element_inverse`](OperatorTrait::operator_create_fdm_element_inverse),
//! [`operator_create_fdm_element_inverse_jacobi`](OperatorTrait::operator_create_fdm_element_inverse_jacobi),
//! [`linear_assemble_add_diagonal`](OperatorTrait::linear_assemble_add_diagonal)
//! (defaults return `Err`; dense **`linear_assemble*` / `linear_assemble_add`** / **CSR set & add** /
//! **`linear_assemble_add_diagonal`** and **small-`n` dense inverse FDM hook** on
//! `reed_cpu::CpuOperator` when documented).
//! **CPU-only helpers (not on this trait):** the concrete **`reed_cpu::CpuOperator`** type also exposes
//! `assembled_linear_matrix_col_major`, `dense_linear_assembly_n`, `dense_linear_assembly_numeric_ready`, and
//! `clear_dense_linear_assembly` for the optional dense `Mutex` slot (see `reed_cpu` / `design_mapping.md` §4.5).
//! **CPU assembly:** `reed_cpu::CpuOperator` via `reed_cpu::OperatorBuilder` (re-exported from the `reed` crate).
//!
//! **Single global active input and single global active output** per [`OperatorTrait::apply`]
//! call (same pattern as many libCEED volume examples). When multiple active fields participate
//! on the input and/or output side, use [`OperatorTrait::apply_field_buffers`] (implemented on
//! `CpuOperator`; default on other types returns `ReedError::Operator`) when
//! [`OperatorTrait::requires_field_named_buffers`] is `true`. Asymmetric
//! build operators use different input/output restriction sizes; use
//! `CpuOperator::active_input_global_len` / `active_output_global_len` on the concrete type (see
//! `design_mapping.md` §4.5.1).

use crate::{
    csr::{CsrMatrix, CsrPattern},
    error::{ReedError, ReedResult},
    matrix::CeedMatrix,
    scalar::Scalar,
    vector::VectorTrait,
};

/// Which operator direction to apply (libCEED `CeedTransposeMode` / `CeedOperatorApply*`).
///
/// - [`Self::Forward`] corresponds to `CEED_NOTRANSPOSE` (standard `y = A x`).
/// - [`Self::Adjoint`] corresponds to `CEED_TRANSPOSE` (adjoint / transposed apply). The default
///   [`OperatorTrait`] methods return [`ReedError::Operator`]; `reed_cpu::CpuOperator` overrides
///   when the qfunction supports [`crate::qfunction::QFunctionTrait::apply_operator_transpose`] and
///   the operator meets the CPU v1 constraints (single-buffer or named field maps; scalar `Weight`
///   fields use the same qp↔nodal layout as `Interp` on the CPU Lagrange/simplex basis).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperatorTransposeRequest {
    Forward,
    Adjoint,
}

/// libCEED-style **optional operator assembly** hooks (`CeedOperatorLinearAssemble*`,
/// `CeedOperatorCreateFDMElementInverse`, …). Used with [`OperatorTrait::operator_supports_assemble`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperatorAssembleKind {
    /// [`OperatorTrait::linear_assemble_diagonal`] / [`OperatorTrait::linear_assemble_add_diagonal`]
    /// (implemented on `reed_cpu::CpuOperator` when applicable; same probe for **set** vs **add** diagonal).
    Diagonal,
    /// `CeedOperatorLinearAssembleSymbolic` — on **`reed_cpu::CpuOperator`**, allocates a dense `n×n`
    /// buffer (`n = active_global_dof_len`); not CSR. Composites return `Err`.
    LinearSymbolic,
    /// `CeedOperatorLinearAssemble` / **`CeedOperatorLinearAssembleAdd`** — on **`CpuOperator`**, dense
    /// **set** ([`OperatorTrait::linear_assemble`]) and **add** ([`OperatorTrait::linear_assemble_add`])
    /// share the same capability probe (columns `A e_j`). Composites return `Err` for both.
    LinearNumeric,
    /// Sparse numeric assembly: **[`OperatorTrait::linear_assemble_csr_matrix`]** /
    /// **[`OperatorTrait::linear_assemble_csr_matrix_add`]** with caller-supplied structure (libCEED
    /// `LinearAssemble` / `LinearAssembleAdd` into CSR; pattern from
    /// **[`crate::elem_restriction::ElemRestrictionTrait::assembled_csr_pattern`]**, not from the operator).
    /// **`CpuOperator`**: same **`active_global_dof_len`** predicate as [`Self::LinearNumeric`]. **Composites**:
    /// probe **`false`** (no shared CSR sink; methods return **`Err`**).
    LinearCsrNumeric,
    /// `CeedOperatorCreateFDMElementInverse` — on **`reed_cpu::CpuOperator`**, supported when
    /// `active_global_dof_len ≤ reed_cpu::FDM_DENSE_MAX_N` via **assembled dense inverse** (not
    /// libCEED’s tensor FDM). Composites: probe is **`false`**; creation always **`Err`**.
    FdmElementInverse,
}

/// Full weak-form operator trait.
///
/// Combines ElemRestriction + Basis + QFunction to realize:
///   v = A(u) = Eᵀ Bᵀ D B E u
#[cfg(not(target_arch = "wasm32"))]
pub trait OperatorTrait<T: Scalar>: Send + Sync {
    fn apply(&self, input: &dyn VectorTrait<T>, output: &mut dyn VectorTrait<T>) -> ReedResult<()>;
    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()>;
    fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()>;

    /// libCEED `CeedOperatorLinearAssembleAddDiagonal`: **add** this operator’s Jacobian diagonal
    /// into `assembled` (**no** zero-fill). Default `Err`; implemented on **`reed_cpu::CpuOperator`**
    /// and **`reed_cpu::CompositeOperator*`** when [`Self::linear_assemble_diagonal`] is supported.
    fn linear_assemble_add_diagonal(&self, _assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_add_diagonal is not implemented (libCEED CeedOperatorLinearAssembleAddDiagonal)"
                .into(),
        ))
    }

    /// When known (e.g. assembled `CpuOperator` from `OperatorBuilder`),
    /// composite builders may require all sub-operators to report the same length for
    /// active global input/output vectors.
    fn global_vector_len_hint(&self) -> Option<usize> {
        None
    }

    /// `true` when this operator cannot use [`Self::apply`] / [`Self::apply_add`] with one global
    /// input buffer and one global output buffer (e.g. multiple active fields on the CPU path).
    /// Additive `CompositeOperator` (in `reed_cpu`) rejects sub-operators that return `true` here.
    fn requires_field_named_buffers(&self) -> bool {
        false
    }

    /// Optional human-readable label (libCEED `CeedOperatorSetName` / logging).
    fn operator_label(&self) -> Option<&str> {
        None
    }

    /// Apply with transpose mode (libCEED `CeedOperatorApply` with `CeedTransposeMode`).
    ///
    /// Default: [`OperatorTransposeRequest::Forward`] delegates to [`Self::apply`];
    /// [`OperatorTransposeRequest::Adjoint`] returns [`ReedError::Operator`] unless overridden
    /// (e.g. `reed_cpu::CpuOperator` when the qfunction supports operator transpose).
    fn apply_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply(input, output),
            OperatorTransposeRequest::Adjoint => Err(ReedError::Operator(
                "OperatorTrait::apply_with_transpose(Adjoint) is not implemented for this operator (libCEED CeedOperatorApplyTranspose)".into(),
            )),
        }
    }

    /// Same as [`Self::apply_with_transpose`] but uses [`Self::apply_add`] on the forward path.
    fn apply_add_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_add(input, output),
            OperatorTransposeRequest::Adjoint => Err(ReedError::Operator(
                "OperatorTrait::apply_add_with_transpose(Adjoint) is not implemented for this operator (libCEED CeedOperatorApplyTranspose)".into(),
            )),
        }
    }

    /// libCEED-style **multi-vector** apply: one global handle per active field (matched by field
    /// name). Default returns [`ReedError::Operator`]; `reed_cpu::CpuOperator` assembled from
    /// `reed_cpu::OperatorBuilder` provides the implementation.
    fn apply_field_buffers<'io>(
        &self,
        _inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        _outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::apply_field_buffers is not implemented for this operator type (only reed_cpu::CpuOperator from OperatorBuilder)".into(),
        ))
    }

    /// Same as [`Self::apply_field_buffers`] but accumulates into each output field.
    fn apply_add_field_buffers<'io>(
        &self,
        _inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        _outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::apply_add_field_buffers is not implemented for this operator type (only reed_cpu::CpuOperator from OperatorBuilder)".into(),
        ))
    }

    /// Multi-vector apply with transpose mode (libCEED `CeedOperatorApply` + field maps).
    ///
    /// - [`OperatorTransposeRequest::Forward`] delegates to [`Self::apply_field_buffers`].
    /// - [`OperatorTransposeRequest::Adjoint`]: `inputs` holds one vector per **active output field**
    ///   (cotangent on the forward range); `outputs` receives one buffer per **active input field**
    ///   (cotangent on the forward domain). **Passive / `None` input slots** do not appear in
    ///   `outputs` (no domain cotangent is accumulated for them). Default returns [`ReedError::Operator`].
    fn apply_field_buffers_with_transpose<'io>(
        &self,
        request: OperatorTransposeRequest,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_field_buffers(inputs, outputs),
            OperatorTransposeRequest::Adjoint => Err(ReedError::Operator(
                "OperatorTrait::apply_field_buffers_with_transpose(Adjoint) is not implemented for this operator"
                    .into(),
            )),
        }
    }

    /// Same as [`Self::apply_field_buffers_with_transpose`] but uses [`Self::apply_add_field_buffers`]
    /// on the forward path and accumulates on the adjoint path.
    fn apply_add_field_buffers_with_transpose<'io>(
        &self,
        request: OperatorTransposeRequest,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_add_field_buffers(inputs, outputs),
            OperatorTransposeRequest::Adjoint => Err(ReedError::Operator(
                "OperatorTrait::apply_add_field_buffers_with_transpose(Adjoint) is not implemented for this operator"
                    .into(),
            )),
        }
    }

    /// libCEED `CeedOperatorCheckReady`: verify fields, restrictions, passive vector sizes, and
    /// quadrature consistency before [`Self::apply`]. Default is a no-op for custom operators.
    fn check_ready(&self) -> ReedResult<()> {
        Ok(())
    }

    /// Capability probe for optional libCEED-style assembly paths.
    fn operator_supports_assemble(&self, kind: OperatorAssembleKind) -> bool {
        matches!(kind, OperatorAssembleKind::Diagonal)
    }

    /// libCEED `CeedOperatorLinearAssembleSymbolic`. Default `Err`; implemented on **`reed_cpu::CpuOperator`**
    /// (dense buffer) when applicable. On **`CpuOperator`**, each successful call **allocates or replaces**
    /// the dense `n×n` slot (any prior buffer is dropped); **`numeric_done`** is reset until the next
    /// [`Self::linear_assemble`] / [`Self::linear_assemble_add`].
    fn linear_assemble_symbolic(&self) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_symbolic is not implemented (libCEED CeedOperatorLinearAssembleSymbolic)"
                .into(),
        ))
    }

    /// libCEED `CeedOperatorLinearAssemble`. Default `Err`; implemented on **`reed_cpu::CpuOperator`** (dense fill).
    fn linear_assemble(&self) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble is not implemented (libCEED CeedOperatorLinearAssemble)".into(),
        ))
    }

    /// libCEED `CeedOperatorLinearAssembleAdd`: **add** Jacobian columns into the dense buffer from
    /// [`Self::linear_assemble_symbolic`] (**no** column overwrite). Default `Err`; implemented on
    /// **`reed_cpu::CpuOperator`** when applicable. Composites return `Err`.
    fn linear_assemble_add(&self) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_add is not implemented (libCEED CeedOperatorLinearAssembleAdd)".into(),
        ))
    }

    /// Assemble the linearized operator into **CSR** for a caller-supplied **pattern** (libCEED
    /// `CeedOperatorLinearAssemble` into a sparse matrix with prior symbolic structure). Default
    /// `Err`; implemented on **`reed_cpu::CpuOperator`** when `active_global_dof_len` matches the pattern.
    fn linear_assemble_csr_matrix(&self, _pattern: &CsrPattern) -> ReedResult<CsrMatrix<T>> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_csr_matrix is not implemented (use reed_cpu::CpuOperator or supply a CsrPattern from ElemRestrictionTrait::assembled_csr_pattern)"
                .into(),
        ))
    }

    /// libCEED `CeedOperatorLinearAssembleAdd` into an existing **[`CsrMatrix`]**: **add** numeric
    /// entries for each pattern `(row,col)` (**no** zero-fill). Default `Err`; implemented on
    /// **`reed_cpu::CpuOperator`** when dimensions match. Composites return `Err`.
    fn linear_assemble_csr_matrix_add(&self, _matrix: &mut CsrMatrix<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_csr_matrix_add is not implemented (libCEED CeedOperatorLinearAssembleAdd into CSR)"
                .into(),
        ))
    }

    /// Assemble into a libCEED-shaped matrix handle (**set** semantics): numeric entries are written
    /// into `matrix` according to its storage (`DenseColMajor` or `Csr`), replacing prior numeric values.
    /// Default `Err`; implemented on **`reed_cpu::CpuOperator`**.
    fn linear_assemble_ceed_matrix(&self, _matrix: &mut CeedMatrix<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_ceed_matrix is not implemented (libCEED-style matrix handle set assembly)"
                .into(),
        ))
    }

    /// Assemble into a libCEED-shaped matrix handle (**add** semantics): numeric entries are
    /// accumulated into `matrix` (`+=`) without clearing. Default `Err`; implemented on
    /// **`reed_cpu::CpuOperator`**.
    fn linear_assemble_add_ceed_matrix(&self, _matrix: &mut CeedMatrix<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_add_ceed_matrix is not implemented (libCEED-style matrix handle add assembly)"
                .into(),
        ))
    }

    /// libCEED `CeedOperatorCreateFDMElementInverse`. Default `Err`; implemented on
    /// **`reed_cpu::CpuOperator`** as a small-`n` dense inverse (see `reed_cpu::fdm_inverse` module).
    fn operator_create_fdm_element_inverse(&self) -> ReedResult<Box<dyn OperatorTrait<T>>> {
        Err(ReedError::Operator(
            "OperatorTrait::operator_create_fdm_element_inverse is not implemented (libCEED CeedOperatorCreateFDMElementInverse)"
                .into(),
        ))
    }

    /// Structured diagonal-only inverse path (Jacobi), useful as a lightweight fallback when a
    /// full dense inverse is undesirable. Default `Err`; implemented on **`reed_cpu::CpuOperator`**.
    fn operator_create_fdm_element_inverse_jacobi(&self) -> ReedResult<Box<dyn OperatorTrait<T>>> {
        Err(ReedError::Operator(
            "OperatorTrait::operator_create_fdm_element_inverse_jacobi is not implemented".into(),
        ))
    }
}

#[cfg(target_arch = "wasm32")]
pub trait OperatorTrait<T: Scalar> {
    fn apply(&self, input: &dyn VectorTrait<T>, output: &mut dyn VectorTrait<T>) -> ReedResult<()>;
    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()>;
    fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()>;

    fn linear_assemble_add_diagonal(&self, _assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_add_diagonal is not implemented (wasm32 stub)".into(),
        ))
    }

    fn global_vector_len_hint(&self) -> Option<usize> {
        None
    }

    fn requires_field_named_buffers(&self) -> bool {
        false
    }

    fn operator_label(&self) -> Option<&str> {
        None
    }

    /// Default: [`OperatorTransposeRequest::Forward`] delegates to [`Self::apply`];
    /// [`OperatorTransposeRequest::Adjoint`] returns [`ReedError::Operator`] unless overridden.
    fn apply_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply(input, output),
            OperatorTransposeRequest::Adjoint => Err(ReedError::Operator(
                "OperatorTrait::apply_with_transpose(Adjoint) is not implemented for this operator (libCEED CeedOperatorApplyTranspose)".into(),
            )),
        }
    }

    /// Same as [`Self::apply_with_transpose`] but uses [`Self::apply_add`] on the forward path.
    fn apply_add_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_add(input, output),
            OperatorTransposeRequest::Adjoint => Err(ReedError::Operator(
                "OperatorTrait::apply_add_with_transpose(Adjoint) is not implemented for this operator (libCEED CeedOperatorApplyTranspose)".into(),
            )),
        }
    }

    fn apply_field_buffers<'io>(
        &self,
        _inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        _outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::apply_field_buffers is not implemented for this operator type (only reed_cpu::CpuOperator from OperatorBuilder)".into(),
        ))
    }

    fn apply_add_field_buffers<'io>(
        &self,
        _inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        _outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::apply_add_field_buffers is not implemented for this operator type (only reed_cpu::CpuOperator from OperatorBuilder)".into(),
        ))
    }

    fn apply_field_buffers_with_transpose<'io>(
        &self,
        request: OperatorTransposeRequest,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_field_buffers(inputs, outputs),
            OperatorTransposeRequest::Adjoint => Err(ReedError::Operator(
                "OperatorTrait::apply_field_buffers_with_transpose(Adjoint) is not implemented for this operator"
                    .into(),
            )),
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
            OperatorTransposeRequest::Adjoint => Err(ReedError::Operator(
                "OperatorTrait::apply_add_field_buffers_with_transpose(Adjoint) is not implemented for this operator"
                    .into(),
            )),
        }
    }

    fn check_ready(&self) -> ReedResult<()> {
        Ok(())
    }

    fn operator_supports_assemble(&self, kind: OperatorAssembleKind) -> bool {
        matches!(kind, OperatorAssembleKind::Diagonal)
    }

    /// See non-wasm [`OperatorTrait::linear_assemble_symbolic`] (includes **`CpuOperator`** “replace slot” semantics on native); default `Err` here.
    fn linear_assemble_symbolic(&self) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_symbolic is not implemented (libCEED CeedOperatorLinearAssembleSymbolic)"
                .into(),
        ))
    }

    /// See non-wasm [`OperatorTrait::linear_assemble`]; default `Err` here.
    fn linear_assemble(&self) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble is not implemented (libCEED CeedOperatorLinearAssemble)".into(),
        ))
    }

    fn linear_assemble_add(&self) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_add is not implemented (wasm32 stub)".into(),
        ))
    }

    fn linear_assemble_csr_matrix(&self, _pattern: &CsrPattern) -> ReedResult<CsrMatrix<T>> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_csr_matrix is not implemented (wasm32 stub)".into(),
        ))
    }

    fn linear_assemble_csr_matrix_add(&self, _matrix: &mut CsrMatrix<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_csr_matrix_add is not implemented (wasm32 stub)".into(),
        ))
    }

    fn linear_assemble_ceed_matrix(&self, _matrix: &mut CeedMatrix<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_ceed_matrix is not implemented (wasm32 stub)".into(),
        ))
    }

    fn linear_assemble_add_ceed_matrix(&self, _matrix: &mut CeedMatrix<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "OperatorTrait::linear_assemble_add_ceed_matrix is not implemented (wasm32 stub)"
                .into(),
        ))
    }

    fn operator_create_fdm_element_inverse(&self) -> ReedResult<Box<dyn OperatorTrait<T>>> {
        Err(ReedError::Operator(
            "OperatorTrait::operator_create_fdm_element_inverse is not implemented (libCEED CeedOperatorCreateFDMElementInverse)"
                .into(),
        ))
    }

    fn operator_create_fdm_element_inverse_jacobi(&self) -> ReedResult<Box<dyn OperatorTrait<T>>> {
        Err(ReedError::Operator(
            "OperatorTrait::operator_create_fdm_element_inverse_jacobi is not implemented (wasm32 stub)"
                .into(),
        ))
    }
}
