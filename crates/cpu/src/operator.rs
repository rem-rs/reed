use crate::assembly_dense::DenseLinearAssemblySlot;
use reed_core::{
    basis::BasisTrait,
    csr::{CsrMatrix, CsrPattern},
    elem_restriction::ElemRestrictionTrait,
    enums::{EvalMode, TransposeMode},
    error::ReedResult,
    matrix::{CeedMatrix, CeedMatrixStorage},
    operator::{OperatorAssembleKind, OperatorTrait, OperatorTransposeRequest},
    qfunction::QFunctionTrait,
    scalar::Scalar,
    vector::VectorTrait,
    QFunctionContext, ReedError,
};
use std::collections::HashSet;
use std::sync::Mutex;

/// Active global data for qfunction input fields: single buffer (legacy `apply`) or one handle per field name.
#[derive(Clone, Copy)]
enum ActiveInputSource<'b, T: Scalar> {
    Single(&'b dyn VectorTrait<T>),
    Named(&'b [(&'b str, &'b dyn VectorTrait<T>)]),
}

/// Active global data for qfunction output fields: single buffer or named handles (libCEED multi-vector apply).
enum ActiveOutputSink<'io, T: Scalar> {
    Single(&'io mut dyn VectorTrait<T>),
    Named(&'io mut [(&'io str, &'io mut dyn VectorTrait<T>)]),
}

pub enum FieldVector<'a, T: Scalar> {
    Active,
    Passive(&'a dyn VectorTrait<T>),
    None,
}

pub struct OperatorField<'a, T: Scalar> {
    name: String,
    restriction: Option<&'a dyn ElemRestrictionTrait<T>>,
    basis: Option<&'a dyn BasisTrait<T>>,
    vector: FieldVector<'a, T>,
}

#[derive(Clone, Copy)]
struct InputPlan {
    field_index: usize,
    eval_mode: EvalMode,
}

#[derive(Clone, Copy)]
struct OutputPlan {
    field_index: usize,
    eval_mode: EvalMode,
}

pub struct OperatorBuilder<'a, T: Scalar> {
    qfunction: Option<Box<dyn QFunctionTrait<T>>>,
    qfunction_context: Option<QFunctionContext>,
    /// Optional label for logging (libCEED `CeedOperatorSetName`).
    op_label: Option<String>,
    fields: Vec<OperatorField<'a, T>>,
}

impl<'a, T: Scalar> Default for OperatorBuilder<'a, T> {
    fn default() -> Self {
        Self {
            qfunction: None,
            qfunction_context: None,
            op_label: None,
            fields: Vec::new(),
        }
    }
}

impl<'a, T: Scalar> OperatorBuilder<'a, T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn qfunction(mut self, qfunction: Box<dyn QFunctionTrait<T>>) -> Self {
        self.qfunction = Some(qfunction);
        self
    }

    /// User [`QFunctionContext`] buffer; byte length must match
    /// [`QFunctionTrait::context_byte_len`] of the configured qfunction (often zero).
    pub fn qfunction_context(mut self, ctx: QFunctionContext) -> Self {
        self.qfunction_context = Some(ctx);
        self
    }

    /// Human-readable operator name for logging (libCEED `CeedOperatorSetName`).
    pub fn operator_label(mut self, label: impl Into<String>) -> Self {
        self.op_label = Some(label.into());
        self
    }

    pub fn field(
        mut self,
        name: impl Into<String>,
        restriction: Option<&'a dyn ElemRestrictionTrait<T>>,
        basis: Option<&'a dyn BasisTrait<T>>,
        vector: FieldVector<'a, T>,
    ) -> Self {
        self.fields.push(OperatorField {
            name: name.into(),
            restriction,
            basis,
            vector,
        });
        self
    }

    pub fn build(self) -> ReedResult<CpuOperator<'a, T>> {
        let qfunction = self
            .qfunction
            .ok_or_else(|| ReedError::Operator("operator builder requires a qfunction".into()))?;
        let ctx_need = qfunction.context_byte_len();
        let qfunction_context = match (self.qfunction_context, ctx_need) {
            (Some(c), need) if c.byte_len() != need => {
                return Err(ReedError::Operator(format!(
                    "QFunctionContext length {} does not match qfunction.context_byte_len() {}",
                    c.byte_len(),
                    need
                )));
            }
            (Some(c), _) => c,
            (None, 0) => QFunctionContext::new(0),
            (None, need) => {
                return Err(ReedError::Operator(format!(
                    "qfunction requires {} byte(s) of QFunctionContext; call .qfunction_context(...)",
                    need
                )));
            }
        };
        let input_plans = qfunction
            .inputs()
            .iter()
            .map(|descriptor| {
                Ok(InputPlan {
                    field_index: CpuOperator::field_index_by_name(&self.fields, &descriptor.name)?,
                    eval_mode: descriptor.eval_mode,
                })
            })
            .collect::<ReedResult<Vec<_>>>()?;
        let output_plans = qfunction
            .outputs()
            .iter()
            .map(|descriptor| {
                Ok(OutputPlan {
                    field_index: CpuOperator::field_index_by_name(&self.fields, &descriptor.name)?,
                    eval_mode: descriptor.eval_mode,
                })
            })
            .collect::<ReedResult<Vec<_>>>()?;
        let num_qfunction_inputs = input_plans.len();
        let num_qfunction_outputs = output_plans.len();
        let num_elem = self
            .fields
            .iter()
            .find_map(|field| {
                field
                    .restriction
                    .map(|restriction| restriction.num_elements())
            })
            .ok_or_else(|| {
                ReedError::Operator(
                    "operator builder requires at least one restricted field".into(),
                )
            })?;
        let num_qpoints = self
            .fields
            .iter()
            .find_map(|field| field.basis.map(|basis| basis.num_qpoints()))
            .or_else(|| {
                self.fields.iter().find_map(|field| {
                    field
                        .restriction
                        .map(|restriction| restriction.num_dof_per_elem())
                })
            })
            .ok_or_else(|| {
                ReedError::Operator(
                    "operator builder requires at least one basis or restriction".into(),
                )
            })?;
        Ok(CpuOperator {
            qfunction,
            qfunction_context,
            op_label: self.op_label,
            fields: self.fields,
            input_plans,
            output_plans,
            num_elem,
            num_qpoints,
            num_qfunction_inputs,
            num_qfunction_outputs,
            dense_linear_assembly: Mutex::new(None),
            last_forward_q_inputs: Mutex::new(None),
        })
    }
}

pub struct CpuOperator<'a, T: Scalar> {
    qfunction: Box<dyn QFunctionTrait<T>>,
    qfunction_context: QFunctionContext,
    op_label: Option<String>,
    fields: Vec<OperatorField<'a, T>>,
    input_plans: Vec<InputPlan>,
    output_plans: Vec<OutputPlan>,
    num_elem: usize,
    num_qpoints: usize,
    num_qfunction_inputs: usize,
    num_qfunction_outputs: usize,
    /// Optional dense global matrix from [`OperatorTrait::linear_assemble_symbolic`] /
    /// [`OperatorTrait::linear_assemble`] / [`OperatorTrait::linear_assemble_add`] (libCEED-shaped migration hook).
    /// Clear with [`Self::clear_dense_linear_assembly`]. `Mutex` keeps [`CpuOperator`] `Sync` for [`OperatorTrait`].
    dense_linear_assembly: Mutex<Option<DenseLinearAssemblySlot<T>>>,
    /// Cached quadrature-point inputs from the latest forward pass (`execute_inner`) for optional
    /// nonlinear/forward-state-aware adjoint kernels.
    last_forward_q_inputs: Mutex<Option<Vec<Vec<T>>>>,
}

impl<'a, T: Scalar> CpuOperator<'a, T> {
    /// Number of mesh elements (from restrictions on this operator).
    #[inline]
    pub fn num_elements(&self) -> usize {
        self.num_elem
    }

    /// Quadrature points per element (from the first basis, or restriction `elemsize` fallback in builder).
    #[inline]
    pub fn num_quadrature_points_per_elem(&self) -> usize {
        self.num_qpoints
    }

    /// Byte length of the qfunction context buffer.
    #[inline]
    pub fn qfunction_context_byte_len(&self) -> usize {
        self.qfunction_context.byte_len()
    }

    fn ensure_io_lengths(
        &self,
        input: &dyn VectorTrait<T>,
        output: &dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        if let Some(e) = self.active_input_global_len()? {
            if input.len() != e {
                return Err(ReedError::Operator(format!(
                    "operator apply: input length {} != active input global DOF count {}",
                    input.len(),
                    e
                )));
            }
        }
        if let Some(e) = self.active_output_global_len()? {
            if output.len() != e {
                return Err(ReedError::Operator(format!(
                    "operator apply: output length {} != active output global DOF count {}",
                    output.len(),
                    e
                )));
            }
        }
        Ok(())
    }

    /// Merge [`FieldVector::Active`] + restriction global sizes for qfunction **input** fields only.
    pub fn active_input_global_len(&self) -> ReedResult<Option<usize>> {
        Self::merge_active_global_for_field_indices(
            &self.fields,
            self.input_plans.iter().map(|p| p.field_index),
        )
    }

    /// Merge active + restriction global sizes for qfunction **output** fields only.
    pub fn active_output_global_len(&self) -> ReedResult<Option<usize>> {
        Self::merge_active_global_for_field_indices(
            &self.fields,
            self.output_plans.iter().map(|p| p.field_index),
        )
    }

    fn merge_active_global_for_field_indices(
        fields: &[OperatorField<'a, T>],
        field_indices: impl Iterator<Item = usize>,
    ) -> ReedResult<Option<usize>> {
        let mut sizes = HashSet::new();
        for idx in field_indices {
            let field = &fields[idx];
            if matches!(field.vector, FieldVector::Active) {
                if let Some(r) = field.restriction {
                    sizes.insert(r.num_global_dof());
                }
            }
        }
        if sizes.is_empty() {
            return Ok(None);
        }
        if sizes.len() > 1 {
            return Ok(None);
        }
        Ok(sizes.into_iter().next())
    }

    /// Distinct field indices that appear as [`FieldVector::Active`] on qfunction **inputs**.
    pub fn distinct_active_input_field_indices(&self) -> Vec<usize> {
        let mut s = HashSet::new();
        for p in &self.input_plans {
            if matches!(self.fields[p.field_index].vector, FieldVector::Active) {
                s.insert(p.field_index);
            }
        }
        let mut v: Vec<usize> = s.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// Distinct field indices that appear as [`FieldVector::Active`] on qfunction **outputs**.
    pub fn distinct_active_output_field_indices(&self) -> Vec<usize> {
        let mut s = HashSet::new();
        for p in &self.output_plans {
            if matches!(self.fields[p.field_index].vector, FieldVector::Active) {
                s.insert(p.field_index);
            }
        }
        let mut v: Vec<usize> = s.into_iter().collect();
        v.sort_unstable();
        v
    }

    fn multi_distinct_active_io_fields(&self) -> bool {
        self.distinct_active_input_field_indices().len() > 1
            || self.distinct_active_output_field_indices().len() > 1
    }

    /// True when this operator cannot use single-buffer [`OperatorTrait::apply`] and requires
    /// [`OperatorTrait::apply_field_buffers`] (multiple active fields on the input side and/or output side).
    pub fn requires_field_named_buffers(&self) -> bool {
        self.multi_distinct_active_io_fields()
    }

    fn lookup_named_read<'b>(
        m: &[(&'b str, &'b dyn VectorTrait<T>)],
        name: &str,
    ) -> ReedResult<&'b dyn VectorTrait<T>> {
        let mut found: Option<&'b dyn VectorTrait<T>> = None;
        for (k, v) in m {
            if *k == name {
                if found.is_some() {
                    return Err(ReedError::Operator(format!(
                        "duplicate field name {:?} in input vector map",
                        name
                    )));
                }
                found = Some(*v);
            }
        }
        found.ok_or_else(|| {
            ReedError::Operator(format!("missing vector for active input field {:?}", name))
        })
    }

    fn lookup_named_write_slot(
        m: &[(&str, &mut dyn VectorTrait<T>)],
        name: &str,
    ) -> ReedResult<usize> {
        let mut found: Option<usize> = None;
        for (i, (k, _)) in m.iter().enumerate() {
            if *k == name {
                if found.is_some() {
                    return Err(ReedError::Operator(format!(
                        "duplicate field name {:?} in output vector map",
                        name
                    )));
                }
                found = Some(i);
            }
        }
        found.ok_or_else(|| {
            ReedError::Operator(format!("missing vector for active output field {:?}", name))
        })
    }

    fn assert_unique_field_keys(names: &[&str], label: &str) -> ReedResult<()> {
        let mut s = HashSet::new();
        for k in names {
            if !s.insert(*k) {
                return Err(ReedError::Operator(format!(
                    "duplicate field name {:?} in {}",
                    k, label
                )));
            }
        }
        Ok(())
    }

    fn validate_named_field_buffers(
        &self,
        inputs: &[(&str, &dyn VectorTrait<T>)],
        outputs: &[(&str, &mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        let in_keys: Vec<&str> = inputs.iter().map(|(k, _)| *k).collect();
        Self::assert_unique_field_keys(&in_keys, "apply_field_buffers inputs")?;
        let out_keys: Vec<&str> = outputs.iter().map(|(k, _)| *k).collect();
        Self::assert_unique_field_keys(&out_keys, "apply_field_buffers outputs")?;

        for &idx in &self.distinct_active_input_field_indices() {
            let f = &self.fields[idx];
            let v = Self::lookup_named_read(inputs, f.name.as_str())?;
            if let Some(r) = f.restriction {
                let need = r.num_global_dof();
                if v.len() != need {
                    return Err(ReedError::Operator(format!(
                        "apply_field_buffers: input field '{}' length {} != restriction global DOF {}",
                        f.name,
                        v.len(),
                        need
                    )));
                }
            }
        }
        for &idx in &self.distinct_active_output_field_indices() {
            let f = &self.fields[idx];
            let (_, v) = outputs
                .iter()
                .find(|(name, _)| *name == f.name.as_str())
                .ok_or_else(|| {
                    ReedError::Operator(format!(
                        "apply_field_buffers: missing output for active field {:?}",
                        f.name
                    ))
                })?;
            if let Some(r) = f.restriction {
                let need = r.num_global_dof();
                if v.len() != need {
                    return Err(ReedError::Operator(format!(
                        "apply_field_buffers: output field '{}' length {} != restriction global DOF {}",
                        f.name,
                        v.len(),
                        need
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_adjoint_named_field_buffers(
        &self,
        range_inputs: &[(&str, &dyn VectorTrait<T>)],
        domain_outputs: &[(&str, &mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        let in_keys: Vec<&str> = range_inputs.iter().map(|(k, _)| *k).collect();
        Self::assert_unique_field_keys(
            &in_keys,
            "apply_field_buffers_with_transpose(Adjoint) range (output cotangent) inputs",
        )?;
        let out_keys: Vec<&str> = domain_outputs.iter().map(|(k, _)| *k).collect();
        Self::assert_unique_field_keys(
            &out_keys,
            "apply_field_buffers_with_transpose(Adjoint) domain (input cotangent) outputs",
        )?;

        for &idx in &self.distinct_active_output_field_indices() {
            let f = &self.fields[idx];
            let v = Self::lookup_named_read(range_inputs, f.name.as_str())?;
            if let Some(r) = f.restriction {
                let need = r.num_global_dof();
                if v.len() != need {
                    return Err(ReedError::Operator(format!(
                        "adjoint apply_field_buffers: range cotangent field '{}' length {} != restriction global DOF {}",
                        f.name,
                        v.len(),
                        need
                    )));
                }
            }
        }
        for &idx in &self.distinct_active_input_field_indices() {
            let f = &self.fields[idx];
            let (_, v) = domain_outputs
                .iter()
                .find(|(name, _)| *name == f.name.as_str())
                .ok_or_else(|| {
                    ReedError::Operator(format!(
                        "adjoint apply_field_buffers: missing domain cotangent buffer for active input field {:?}",
                        f.name
                    ))
                })?;
            if let Some(r) = f.restriction {
                let need = r.num_global_dof();
                if v.len() != need {
                    return Err(ReedError::Operator(format!(
                        "adjoint apply_field_buffers: domain cotangent field '{}' length {} != restriction global DOF {}",
                        f.name,
                        v.len(),
                        need
                    )));
                }
            }
        }
        Ok(())
    }

    /// Global vector length when active inputs and active outputs use the **same** restriction size
    /// (typical `MassApply` / Poisson apply). For asymmetric build operators (`Mass1DBuild`, …), use
    /// [`Self::active_input_global_len`] and [`Self::active_output_global_len`] separately.
    pub fn active_global_dof_len(&self) -> ReedResult<usize> {
        if self.multi_distinct_active_io_fields() {
            return Err(ReedError::Operator(
                "active_global_dof_len is undefined for multi-field active operators; use apply_field_buffers (diagonal assembly requires a single global vector space)".into(),
            ));
        }
        match (self.active_input_global_len()?, self.active_output_global_len()?) {
            (Some(a), Some(b)) if a == b => Ok(a),
            (Some(a), None) | (None, Some(a)) => Ok(a),
            (None, None) => Err(ReedError::Operator(
                "could not infer global active vector length (need an active field with restriction on inputs or outputs)"
                    .into(),
            )),
            (Some(a), Some(b)) => Err(ReedError::Operator(format!(
                "active input global DOF {} != output {}; use active_input_global_len / active_output_global_len for asymmetric operators",
                a, b
            ))),
        }
    }

    /// Dense `n × n` matrix from the last successful [`OperatorTrait::linear_assemble`], **column-major**
    /// (`(row, col)` → `a[row + col * n]`). Meaningful only when forward [`OperatorTrait::apply`] is **linear**
    /// in the active unknown for fixed passive data (see `assembly_dense`).
    pub fn assembled_linear_matrix_col_major(&self) -> Option<(usize, Vec<T>)> {
        let g = self.dense_linear_assembly.lock().ok()?;
        let s = g.as_ref()?;
        if !s.numeric_done {
            return None;
        }
        Some((s.n, s.a.clone()))
    }

    /// Returns **`Some(n)`** when a dense assembly slot exists (after [`OperatorTrait::linear_assemble_symbolic`]),
    /// including before [`OperatorTrait::linear_assemble`] / [`OperatorTrait::linear_assemble_add`] finish.
    /// Returns [`None`] after [`Self::clear_dense_linear_assembly`] or if symbolic was never called.
    ///
    /// See also [`Self::dense_linear_assembly_numeric_ready`] and [`Self::assembled_linear_matrix_col_major`].
    pub fn dense_linear_assembly_n(&self) -> Option<usize> {
        let g = self.dense_linear_assembly.lock().ok()?;
        g.as_ref().map(|s| s.n)
    }

    /// `true` iff the dense slot exists **and** the last numeric dense pass completed (same condition as
    /// [`Self::assembled_linear_matrix_col_major`] returning [`Some`]).
    pub fn dense_linear_assembly_numeric_ready(&self) -> bool {
        let Ok(g) = self.dense_linear_assembly.lock() else {
            return false;
        };
        matches!(g.as_ref(), Some(s) if s.numeric_done)
    }

    /// Drop the **dense** global linear-assembly buffer (`O(n²)` host memory) held after
    /// [`OperatorTrait::linear_assemble_symbolic`] / [`OperatorTrait::linear_assemble`] /
    /// [`OperatorTrait::linear_assemble_add`].
    ///
    /// **[`OperatorTrait::apply`]** and **[`Self::linear_assemble_csr_matrix`]** / **[`Self::linear_assemble_csr_matrix_add`]**
    /// are unaffected (CSR paths do not use this slot).
    ///
    /// After clearing, [`Self::assembled_linear_matrix_col_major`] returns [`None`] until a new
    /// **`linear_assemble_symbolic`** + numeric dense pass. **Idempotent** if no buffer was allocated.
    /// [`Self::dense_linear_assembly_n`] becomes [`None`] and [`Self::dense_linear_assembly_numeric_ready`] becomes `false`.
    ///
    /// [`OperatorTrait::operator_create_fdm_element_inverse`] builds its inverse from a locally assembled
    /// canonical Jacobian at construction time; clearing later does not change an already-built
    /// [`crate::fdm_inverse::CpuFdmDenseInverseOperator`].
    pub fn clear_dense_linear_assembly(&self) -> ReedResult<()> {
        let mut g = self.dense_linear_assembly.lock().map_err(|_| {
            ReedError::Operator("clear_dense_linear_assembly: assembly mutex poisoned".into())
        })?;
        *g = None;
        Ok(())
    }

    /// Structured fallback inverse assembled from the operator diagonal only (Jacobi).
    ///
    /// This is a lightweight alternative when a full dense inverse is undesirable and can be used
    /// as a preconditioner-like approximation. Unlike `operator_create_fdm_element_inverse`, this
    /// does not build or invert a dense `n x n` matrix.
    pub fn operator_create_fdm_element_inverse_jacobi(
        &self,
    ) -> ReedResult<Box<dyn OperatorTrait<T>>> {
        self.check_ready()?;
        let n = self.active_global_dof_len()?;
        let mut diag = crate::vector::CpuVector::new(n);
        self.linear_assemble_diagonal(&mut diag)?;
        let mut inv_diag = vec![T::ZERO; n];
        for (i, &d) in diag.as_slice().iter().enumerate() {
            if d == T::ZERO {
                return Err(ReedError::Operator(format!(
                    "operator_create_fdm_element_inverse_jacobi: zero diagonal entry at row {}",
                    i
                )));
            }
            inv_diag[i] = T::ONE / d;
        }
        Ok(Box::new(
            crate::fdm_inverse::CpuFdmJacobiInverseOperator::new(inv_diag),
        ))
    }

    /// Quick check whether tensor-FDM inverse is available for this operator
    /// configuration, without performing the full construction.
    fn can_tensor_fdm(&self) -> bool {
        match self.input_plans.first() {
            Some(p) => {
                let field = &self.fields[p.field_index];
                field
                    .basis
                    .and_then(|b| {
                        if field.restriction.is_none() {
                            None
                        } else {
                            b.tensor_fdm_1d_data()
                        }
                    })
                    .is_some()
            }
            None => false,
        }
    }

    /// Attempt to create a tensor-FDM inverse when the first active field uses a
    /// tensor-product basis (e.g. LagrangeBasis) that provides 1D FDM data.
    fn try_create_fdm_tensor_inverse(&self) -> ReedResult<Option<Box<dyn OperatorTrait<T>>>> {
        use crate::fdm_tensor::{CpuFdmTensorInverseOperator, FdmOperatorKind};

        let field_idx = match self.input_plans.first() {
            Some(p) => p.field_index,
            None => return Ok(None),
        };
        let field = &self.fields[field_idx];
        let basis = match field.basis {
            Some(b) => b,
            None => return Ok(None),
        };
        let restriction = match field.restriction {
            Some(r) => r,
            None => return Ok(None),
        };

        let (interp_1d, grad_1d, weights_1d, p, q) = match basis.tensor_fdm_1d_data() {
            Some(d) => d,
            None => return Ok(None),
        };

        let dim = basis.dim();
        let nelem = self.num_elem;

        // Heuristic: check QFunction inputs for gradient-like field names -> Stiffness, else Mass.
        let op_kind = if self
            .qfunction
            .inputs()
            .iter()
            .any(|f| f.name.contains("du") || f.name.contains("grad"))
        {
            FdmOperatorKind::Stiffness
        } else {
            FdmOperatorKind::Mass
        };

        let restriction_box = restriction.boxed_clone()?;

        let inv = CpuFdmTensorInverseOperator::new(
            interp_1d,
            grad_1d,
            weights_1d,
            p,
            q,
            dim,
            nelem,
            op_kind,
            restriction_box,
        )?;
        Ok(Some(Box::new(inv)))
    }

    /// Assemble into a libCEED-shaped matrix handle (set semantics).
    pub fn linear_assemble_ceed_matrix(&self, matrix: &mut CeedMatrix<T>) -> ReedResult<()> {
        match matrix.storage_mut() {
            CeedMatrixStorage::DenseColMajor {
                nrows,
                ncols,
                values,
            } => {
                self.linear_assemble_symbolic()?;
                self.linear_assemble()?;
                let (n, a) = self.assembled_linear_matrix_col_major().ok_or_else(|| {
                    ReedError::Operator(
                        "linear_assemble_ceed_matrix: dense assembly slot is not numeric-ready"
                            .into(),
                    )
                })?;
                if *nrows != n || *ncols != n || values.len() != a.len() {
                    return Err(ReedError::Operator(format!(
                        "linear_assemble_ceed_matrix: dense handle shape {}x{} (len {}) != operator {}x{} (len {})",
                        *nrows,
                        *ncols,
                        values.len(),
                        n,
                        n,
                        a.len()
                    )));
                }
                values.copy_from_slice(&a);
                matrix.mark_numeric_done(true);
                Ok(())
            }
            CeedMatrixStorage::Csr(m) => {
                let out = self.linear_assemble_csr_matrix(&m.pattern)?;
                m.values.copy_from_slice(&out.values);
                matrix.mark_numeric_done(true);
                Ok(())
            }
        }
    }

    /// Assemble into a libCEED-shaped matrix handle (add semantics).
    pub fn linear_assemble_add_ceed_matrix(&self, matrix: &mut CeedMatrix<T>) -> ReedResult<()> {
        match matrix.storage_mut() {
            CeedMatrixStorage::DenseColMajor {
                nrows,
                ncols,
                values,
            } => {
                self.linear_assemble_symbolic()?;
                self.linear_assemble()?;
                let (n, a) = self.assembled_linear_matrix_col_major().ok_or_else(|| {
                    ReedError::Operator(
                        "linear_assemble_add_ceed_matrix: dense assembly slot is not numeric-ready"
                            .into(),
                    )
                })?;
                if *nrows != n || *ncols != n || values.len() != a.len() {
                    return Err(ReedError::Operator(format!(
                        "linear_assemble_add_ceed_matrix: dense handle shape {}x{} (len {}) != operator {}x{} (len {})",
                        *nrows,
                        *ncols,
                        values.len(),
                        n,
                        n,
                        a.len()
                    )));
                }
                for (dst, src) in values.iter_mut().zip(a.iter()) {
                    *dst += *src;
                }
                matrix.mark_numeric_done(true);
                Ok(())
            }
            CeedMatrixStorage::Csr(m) => {
                self.linear_assemble_csr_matrix_add(m)?;
                matrix.mark_numeric_done(true);
                Ok(())
            }
        }
    }

    /// Assemble `A` into **CSR** using the given **pattern** (typically from
    /// [`ElemRestrictionTrait::assembled_csr_pattern`] on the active trial/test restriction).
    /// Requires `active_global_dof_len == pattern.nrows == pattern.ncols`. Uses `n` forward `apply`s
    /// (same semantics as dense column assembly for entries present in the pattern).
    pub fn linear_assemble_csr_matrix(&self, pattern: &CsrPattern) -> ReedResult<CsrMatrix<T>> {
        let n = self.active_global_dof_len()?;
        if pattern.nrows != n || pattern.ncols != n {
            return Err(ReedError::Operator(format!(
                "linear_assemble_csr_matrix: pattern is {}×{} but active global DOF is {}",
                pattern.nrows, pattern.ncols, n
            )));
        }
        let mut values = vec![T::ZERO; pattern.nnz()];
        for j in 0..n {
            let mut input = vec![T::ZERO; n];
            input[j] = T::ONE;
            let x = crate::vector::CpuVector::from_vec(input);
            let mut y = crate::vector::CpuVector::new(n);
            self.apply(&x, &mut y)?;
            let ys = y.as_slice();
            for i in 0..n {
                if let Some(p) = pattern.index_of(i, j) {
                    values[p] = ys[i];
                }
            }
        }
        Ok(CsrMatrix {
            pattern: pattern.clone(),
            values,
        })
    }

    /// **Add** Jacobian columns into an existing **[`CsrMatrix`]** (libCEED `CeedOperatorLinearAssembleAdd`).
    /// Pattern dimensions must match [`Self::active_global_dof_len`]; `matrix.values.len()` must equal
    /// `matrix.pattern.nnz()`. Existing entries are **accumulated** (`+=`).
    pub fn linear_assemble_csr_matrix_add(&self, matrix: &mut CsrMatrix<T>) -> ReedResult<()> {
        let n = self.active_global_dof_len()?;
        let pattern = &matrix.pattern;
        if pattern.nrows != n || pattern.ncols != n {
            return Err(ReedError::Operator(format!(
                "linear_assemble_csr_matrix_add: pattern is {}×{} but active global DOF is {}",
                pattern.nrows, pattern.ncols, n
            )));
        }
        let nnz = pattern.nnz();
        if matrix.values.len() != nnz {
            return Err(ReedError::Operator(format!(
                "linear_assemble_csr_matrix_add: values.len {} != pattern.nnz {}",
                matrix.values.len(),
                nnz
            )));
        }
        for j in 0..n {
            let mut input = vec![T::ZERO; n];
            input[j] = T::ONE;
            let x = crate::vector::CpuVector::from_vec(input);
            let mut y = crate::vector::CpuVector::new(n);
            self.apply(&x, &mut y)?;
            let ys = y.as_slice();
            for i in 0..n {
                if let Some(p) = pattern.index_of(i, j) {
                    matrix.values[p] += ys[i];
                }
            }
        }
        Ok(())
    }

    /// [`ElemRestrictionTrait::assembled_csr_pattern`] on **`restriction`**, then
    /// [`Self::linear_assemble_csr_matrix`]. **`restriction`** must describe the same global active
    /// DOF layout as this operator (same `active_global_dof_len` / mesh connectivity as the active
    /// trial–test restriction used in [`OperatorBuilder::field`]).
    pub fn linear_assemble_csr_from_elem_restriction(
        &self,
        restriction: &dyn ElemRestrictionTrait<T>,
    ) -> ReedResult<CsrMatrix<T>> {
        let pat = restriction.assembled_csr_pattern()?;
        self.linear_assemble_csr_matrix(&pat)
    }

    fn field_index_by_name(fields: &[OperatorField<'a, T>], name: &str) -> ReedResult<usize> {
        fields
            .iter()
            .position(|field| field.name == name)
            .ok_or_else(|| ReedError::Operator(format!("field '{}' not found", name)))
    }

    fn qpoint_component_count(
        field: &OperatorField<'a, T>,
        eval_mode: EvalMode,
    ) -> ReedResult<usize> {
        match eval_mode {
            EvalMode::None => {
                if let Some(restriction) = field.restriction {
                    Ok(restriction.num_comp())
                } else {
                    Err(ReedError::Operator(format!(
                        "field '{}' without basis requires a restriction to infer component count",
                        field.name
                    )))
                }
            }
            EvalMode::Weight => Ok(1),
            EvalMode::Interp => field.basis.map(|basis| basis.num_comp()).ok_or_else(|| {
                ReedError::Operator(format!("field '{}' requires basis", field.name))
            }),
            EvalMode::Grad => field
                .basis
                .map(|basis| basis.num_comp() * basis.dim())
                .ok_or_else(|| {
                    ReedError::Operator(format!("field '{}' requires basis", field.name))
                }),
            EvalMode::Div => {
                let basis = field.basis.ok_or_else(|| {
                    ReedError::Operator(format!("field '{}' requires basis for Div", field.name))
                })?;
                if basis.num_comp() != basis.dim() {
                    return Err(ReedError::Operator(format!(
                        "field '{}': EvalMode::Div requires basis.num_comp() == basis.dim() (vector field), got comp {} dim {}",
                        field.name,
                        basis.num_comp(),
                        basis.dim()
                    )));
                }
                Ok(1)
            }
            EvalMode::Curl => {
                let basis = field.basis.ok_or_else(|| {
                    ReedError::Operator(format!("field '{}' requires basis for Curl", field.name))
                })?;
                match (basis.dim(), basis.num_comp()) {
                    (2, 2) => Ok(1),
                    (3, 3) => Ok(3),
                    _ => Err(ReedError::Operator(format!(
                        "field '{}': EvalMode::Curl requires (dim, ncomp) = (2, 2) or (3, 3), got dim {} comp {}",
                        field.name,
                        basis.dim(),
                        basis.num_comp()
                    ))),
                }
            }
            EvalMode::HCurl => {
                let basis = field.basis.ok_or_else(|| {
                    ReedError::Operator(format!("field '{}' requires basis for HCurl", field.name))
                })?;
                match basis.dim() {
                    2 => Ok(1),
                    3 => Ok(3),
                    d => Err(ReedError::Operator(format!(
                        "field '{}': HCurl requires dim=2 or 3, got {}", field.name, d
                    ))),
                }
            }
            EvalMode::HDiv => Ok(1),
        }
    }

    fn ensure_adjoint_io_lengths(
        &self,
        range_cotangent: &dyn VectorTrait<T>,
        domain_cotangent: &dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        let out_len = self.active_output_global_len()?.ok_or_else(|| {
            ReedError::Operator(
                "operator adjoint: could not infer active output global length (restriction merge)"
                    .into(),
            )
        })?;
        if range_cotangent.len() != out_len {
            return Err(ReedError::Operator(format!(
                "operator adjoint: input (range cotangent) length {} != active output global DOF {}",
                range_cotangent.len(),
                out_len
            )));
        }
        let in_len = self.active_input_global_len()?.ok_or_else(|| {
            ReedError::Operator(
                "operator adjoint: could not infer active input global length (restriction merge)"
                    .into(),
            )
        })?;
        if domain_cotangent.len() != in_len {
            return Err(ReedError::Operator(format!(
                "operator adjoint: output (domain cotangent) length {} != active input global DOF {}",
                domain_cotangent.len(),
                in_len
            )));
        }
        Ok(())
    }

    fn pull_range_cotangent_to_qp(
        &self,
        field: &OperatorField<'a, T>,
        eval_mode: EvalMode,
        range_global: &[T],
        local_buffer: &mut Vec<T>,
        q_buffer: &mut Vec<T>,
    ) -> ReedResult<()> {
        if !matches!(field.vector, FieldVector::Active) {
            return Err(ReedError::Operator(format!(
                "operator adjoint: output field '{}' must be active",
                field.name
            )));
        }

        let local = if let Some(restriction) = field.restriction {
            local_buffer.resize(restriction.local_size(), T::ZERO);
            restriction.apply(TransposeMode::NoTranspose, range_global, local_buffer)?;
            local_buffer.as_slice()
        } else {
            if let Some(basis) = field.basis {
                let qcomp = Self::qpoint_component_count(field, eval_mode)?;
                let need = self.num_elem * basis.num_qpoints() * qcomp;
                if range_global.len() != need {
                    return Err(ReedError::Operator(format!(
                        "operator adjoint: unrestricted output field '{}': global length {} != expected qp length {}",
                        field.name,
                        range_global.len(),
                        need
                    )));
                }
            } else if range_global.len() != self.num_elem * self.num_qpoints {
                return Err(ReedError::Operator(format!(
                    "operator adjoint: output field '{}': global length {} != element qp length {}",
                    field.name,
                    range_global.len(),
                    self.num_elem * self.num_qpoints
                )));
            }
            range_global
        };

        if let Some(basis) = field.basis {
            let qcomp = Self::qpoint_component_count(field, eval_mode)?;
            q_buffer.resize(self.num_elem * basis.num_qpoints() * qcomp, T::ZERO);
            let basis_eval = if matches!(eval_mode, EvalMode::Weight) {
                if basis.num_comp() != 1 {
                    return Err(ReedError::Operator(format!(
                        "operator adjoint: field '{}' EvalMode::Weight pullback requires basis.num_comp() == 1",
                        field.name
                    )));
                }
                EvalMode::Interp
            } else {
                eval_mode
            };
            basis.apply(self.num_elem, false, basis_eval, local, q_buffer)?;
        } else {
            q_buffer.clear();
            q_buffer.extend_from_slice(local);
        }
        Ok(())
    }

    fn scatter_domain_cotangent_qp_to_global(
        &self,
        field: &OperatorField<'a, T>,
        eval_mode: EvalMode,
        q_in_cot: &[T],
        local_buffer: &mut Vec<T>,
        domain_global: &mut [T],
    ) -> ReedResult<()> {
        match field.vector {
            FieldVector::Active => {
                let local = if let Some(basis) = field.basis {
                    local_buffer
                        .resize(self.num_elem * basis.num_dof() * basis.num_comp(), T::ZERO);
                    if matches!(eval_mode, EvalMode::Weight) && basis.num_comp() != 1 {
                        return Err(ReedError::Operator(format!(
                            "operator adjoint: field '{}' EvalMode::Weight push requires basis.num_comp() == 1",
                            field.name
                        )));
                    }
                    basis.apply(self.num_elem, true, eval_mode, q_in_cot, local_buffer)?;
                    local_buffer.as_slice()
                } else {
                    q_in_cot
                };

                if let Some(restriction) = field.restriction {
                    restriction.apply(TransposeMode::Transpose, local, domain_global)
                } else {
                    if domain_global.len() != local.len() {
                        return Err(ReedError::Operator(format!(
                            "operator adjoint: domain length {} != local length {} for field '{}'",
                            domain_global.len(),
                            local.len(),
                            field.name
                        )));
                    }
                    #[cfg(feature = "parallel")]
                    {
                        use rayon::prelude::*;
                        domain_global
                            .par_iter_mut()
                            .zip(local.par_iter())
                            .for_each(|(dst, src)| *dst += *src);
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        for (dst, src) in domain_global.iter_mut().zip(local.iter()) {
                            *dst += *src;
                        }
                    }
                    Ok(())
                }
            }
            FieldVector::Passive(_) | FieldVector::None => Ok(()),
        }
    }

    fn prepare_input_into(
        &self,
        field: &OperatorField<'a, T>,
        eval_mode: EvalMode,
        active_input: ActiveInputSource<'_, T>,
        local_buffer: &mut Vec<T>,
        q_buffer: &mut Vec<T>,
    ) -> ReedResult<()> {
        if matches!(eval_mode, EvalMode::Weight) {
            let basis = field.basis.ok_or_else(|| {
                ReedError::Operator(format!("field '{}' requires basis for Weight", field.name))
            })?;
            q_buffer.resize(self.num_elem * basis.num_qpoints(), T::ZERO);
            basis.apply(self.num_elem, false, EvalMode::Weight, &[], q_buffer)?;
            return Ok(());
        }

        let source = match field.vector {
            FieldVector::Active => match active_input {
                ActiveInputSource::Single(v) => v.as_slice(),
                ActiveInputSource::Named(m) => {
                    Self::lookup_named_read(m, field.name.as_str())?.as_slice()
                }
            },
            FieldVector::Passive(vector) => vector.as_slice(),
            FieldVector::None => {
                return Err(ReedError::Operator(format!(
                    "field '{}' has no vector source",
                    field.name
                )));
            }
        };

        let local = if let Some(restriction) = field.restriction {
            local_buffer.resize(restriction.local_size(), T::ZERO);
            restriction.apply(TransposeMode::NoTranspose, source, local_buffer)?;
            local_buffer.as_slice()
        } else {
            source
        };

        if let Some(basis) = field.basis {
            let qcomp = Self::qpoint_component_count(field, eval_mode)?;
            q_buffer.resize(self.num_elem * basis.num_qpoints() * qcomp, T::ZERO);
            basis.apply(self.num_elem, false, eval_mode, local, q_buffer)?;
        } else {
            q_buffer.clear();
            q_buffer.extend_from_slice(local);
        }
        Ok(())
    }

    fn scatter_output_to_slice(
        &self,
        field: &OperatorField<'a, T>,
        eval_mode: EvalMode,
        q_output: &[T],
        local_buffer: &mut Vec<T>,
        active_output: &mut [T],
    ) -> ReedResult<()> {
        let local = if let Some(basis) = field.basis {
            local_buffer.resize(self.num_elem * basis.num_dof() * basis.num_comp(), T::ZERO);
            basis.apply(self.num_elem, true, eval_mode, q_output, local_buffer)?;
            local_buffer.as_slice()
        } else {
            q_output
        };

        match field.vector {
            FieldVector::Active => {
                if let Some(restriction) = field.restriction {
                    restriction.apply(TransposeMode::Transpose, &local, active_output)
                } else {
                    if active_output.len() != local.len() {
                        return Err(ReedError::Operator(format!(
                            "output length {} != local length {} for field '{}'",
                            active_output.len(),
                            local.len(),
                            field.name
                        )));
                    }
                    #[cfg(feature = "parallel")]
                    {
                        use rayon::prelude::*;
                        active_output
                            .par_iter_mut()
                            .zip(local.par_iter())
                            .for_each(|(dst, src)| *dst += *src);
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        for (dst, src) in active_output.iter_mut().zip(local.iter()) {
                            *dst += *src;
                        }
                    }
                    Ok(())
                }
            }
            FieldVector::Passive(_) | FieldVector::None => Err(ReedError::Operator(format!(
                "output field '{}' must be active",
                field.name
            ))),
        }
    }

    fn execute_adjoint_inner<'io>(
        &self,
        range_cotangent: ActiveInputSource<'_, T>,
        domain_cotangent: &mut ActiveOutputSink<'io, T>,
        add: bool,
    ) -> ReedResult<()> {
        if !self.qfunction.supports_operator_transpose() {
            return Err(ReedError::Operator(
                "operator adjoint: qfunction does not implement apply_operator_transpose".into(),
            ));
        }

        match range_cotangent {
            ActiveInputSource::Single(rc) => match domain_cotangent {
                ActiveOutputSink::Single(dc) => self.ensure_adjoint_io_lengths(rc, &**dc)?,
                ActiveOutputSink::Named(_) => {
                    return Err(ReedError::Operator(
                        "operator adjoint: mixed single-buffer (range) and named-buffer (domain) is not supported"
                            .into(),
                    ));
                }
            },
            ActiveInputSource::Named(ri) => match domain_cotangent {
                ActiveOutputSink::Named(m) => {
                    self.validate_adjoint_named_field_buffers(ri, &*m)?;
                }
                ActiveOutputSink::Single(_) => {
                    return Err(ReedError::Operator(
                        "operator adjoint: mixed named-buffer (range) and single-buffer (domain) is not supported"
                            .into(),
                    ));
                }
            },
        }

        if !add {
            match domain_cotangent {
                ActiveOutputSink::Single(v) => v.set_value(T::ZERO)?,
                ActiveOutputSink::Named(m) => {
                    let zero_names: HashSet<&str> = self
                        .distinct_active_input_field_indices()
                        .iter()
                        .map(|&i| self.fields[i].name.as_str())
                        .collect();
                    for (k, v) in m.iter_mut() {
                        if zero_names.contains(*k) {
                            v.set_value(T::ZERO)?;
                        }
                    }
                }
            }
        }

        let mut q_out_cot: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();
        let mut q_in_cot: Vec<Vec<T>> =
            (0..self.num_qfunction_inputs).map(|_| Vec::new()).collect();
        let mut q_passive_fwd: Vec<Vec<T>> =
            (0..self.num_qfunction_inputs).map(|_| Vec::new()).collect();
        let mut input_locals: Vec<Vec<T>> =
            (0..self.num_qfunction_inputs).map(|_| Vec::new()).collect();
        let mut output_locals: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();

        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            match &field.vector {
                FieldVector::Passive(_) => {
                    self.prepare_input_into(
                        field,
                        plan.eval_mode,
                        range_cotangent,
                        &mut input_locals[slot],
                        &mut q_passive_fwd[slot],
                    )?;
                }
                FieldVector::Active => {
                    q_passive_fwd[slot].clear();
                }
                FieldVector::None => {
                    return Err(ReedError::Operator(format!(
                        "operator adjoint: input field '{}' has no vector source",
                        field.name
                    )));
                }
            }
        }

        for (slot, plan) in self.output_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            let range_sl = match range_cotangent {
                ActiveInputSource::Single(v) => v.as_slice(),
                ActiveInputSource::Named(m) => {
                    Self::lookup_named_read(m, field.name.as_str())?.as_slice()
                }
            };
            self.pull_range_cotangent_to_qp(
                field,
                plan.eval_mode,
                range_sl,
                &mut output_locals[slot],
                &mut q_out_cot[slot],
            )?;
        }

        let input_descriptors = self.qfunction.inputs();
        for slot in 0..self.num_qfunction_inputs {
            let len = self.num_elem * self.num_qpoints * input_descriptors[slot].num_comp;
            q_in_cot[slot].resize(len, T::ZERO);
            if matches!(
                self.fields[self.input_plans[slot].field_index].vector,
                FieldVector::Passive(_)
            ) {
                q_in_cot[slot].copy_from_slice(&q_passive_fwd[slot]);
            }
        }

        let out_cot_refs: Vec<&[T]> = q_out_cot.iter().map(Vec::as_slice).collect();
        let mut in_cot_mut: Vec<&mut [T]> = q_in_cot.iter_mut().map(|v| v.as_mut_slice()).collect();
        let primal_q_inputs_owned = self
            .last_forward_q_inputs
            .lock()
            .ok()
            .and_then(|g| g.clone());
        let primal_q_inputs: Vec<&[T]> = primal_q_inputs_owned
            .as_ref()
            .map(|v| v.iter().map(Vec::as_slice).collect())
            .unwrap_or_default();
        self.qfunction.apply_operator_transpose_with_primal(
            self.qfunction_context.as_bytes(),
            self.num_elem * self.num_qpoints,
            &primal_q_inputs,
            &out_cot_refs,
            &mut in_cot_mut,
        )?;

        match domain_cotangent {
            ActiveOutputSink::Single(v) => {
                let dom_sl = v.as_mut_slice();
                for (slot, plan) in self.input_plans.iter().enumerate() {
                    let field = &self.fields[plan.field_index];
                    if !matches!(field.vector, FieldVector::Active) {
                        continue;
                    }
                    self.scatter_domain_cotangent_qp_to_global(
                        field,
                        plan.eval_mode,
                        &q_in_cot[slot],
                        &mut input_locals[slot],
                        dom_sl,
                    )?;
                }
            }
            ActiveOutputSink::Named(m) => {
                for (slot, plan) in self.input_plans.iter().enumerate() {
                    let field = &self.fields[plan.field_index];
                    if !matches!(field.vector, FieldVector::Active) {
                        continue;
                    }
                    let j = Self::lookup_named_write_slot(m, field.name.as_str())?;
                    let out_v = &mut *m[j].1;
                    self.scatter_domain_cotangent_qp_to_global(
                        field,
                        plan.eval_mode,
                        &q_in_cot[slot],
                        &mut input_locals[slot],
                        out_v.as_mut_slice(),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn execute_inner<'io>(
        &self,
        input: ActiveInputSource<'_, T>,
        output: &mut ActiveOutputSink<'io, T>,
        add: bool,
    ) -> ReedResult<()> {
        match input {
            ActiveInputSource::Single(i) => match output {
                ActiveOutputSink::Single(o) => self.ensure_io_lengths(i, &**o)?,
                _ => {
                    return Err(ReedError::Operator(
                        "internal error: mixed single-buffer and named-buffer active IO".into(),
                    ));
                }
            },
            ActiveInputSource::Named(_) => match output {
                ActiveOutputSink::Named(_) => {}
                _ => {
                    return Err(ReedError::Operator(
                        "internal error: mixed single-buffer and named-buffer active IO".into(),
                    ));
                }
            },
        }

        if !add {
            match output {
                ActiveOutputSink::Single(v) => v.set_value(T::ZERO)?,
                ActiveOutputSink::Named(m) => {
                    let zero_names: HashSet<&str> = self
                        .distinct_active_output_field_indices()
                        .iter()
                        .map(|&i| self.fields[i].name.as_str())
                        .collect();
                    for (k, v) in m.iter_mut() {
                        if zero_names.contains(*k) {
                            v.set_value(T::ZERO)?;
                        }
                    }
                }
            }
        }

        // Allocate workspace buffers on each call to avoid Mutex overhead
        // The allocation cost is negligible compared to the compute work
        let mut q_inputs: Vec<Vec<T>> =
            (0..self.num_qfunction_inputs).map(|_| Vec::new()).collect();
        let mut q_outputs: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();
        let mut input_locals: Vec<Vec<T>> =
            (0..self.num_qfunction_inputs).map(|_| Vec::new()).collect();
        let mut output_locals: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();

        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            self.prepare_input_into(
                field,
                plan.eval_mode,
                input,
                &mut input_locals[slot],
                &mut q_inputs[slot],
            )?;
        }
        if let Ok(mut cache) = self.last_forward_q_inputs.lock() {
            *cache = Some(q_inputs.clone());
        }

        for (slot, descriptor) in self.qfunction.outputs().iter().enumerate() {
            q_outputs[slot].resize(
                self.num_elem * self.num_qpoints * descriptor.num_comp,
                T::ZERO,
            );
        }

        let input_slices = q_inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut output_slices = q_outputs
            .iter_mut()
            .map(Vec::as_mut_slice)
            .collect::<Vec<_>>();
        self.qfunction.apply(
            self.qfunction_context.as_bytes(),
            self.num_elem * self.num_qpoints,
            &input_slices,
            &mut output_slices,
        )?;

        match output {
            ActiveOutputSink::Single(v) => {
                let out_sl = v.as_mut_slice();
                for (slot, plan) in self.output_plans.iter().enumerate() {
                    let field = &self.fields[plan.field_index];
                    self.scatter_output_to_slice(
                        field,
                        plan.eval_mode,
                        &q_outputs[slot],
                        &mut output_locals[slot],
                        out_sl,
                    )?;
                }
            }
            ActiveOutputSink::Named(m) => {
                for (slot, plan) in self.output_plans.iter().enumerate() {
                    let field = &self.fields[plan.field_index];
                    let j = Self::lookup_named_write_slot(m, field.name.as_str())?;
                    let out_v = &mut *m[j].1;
                    self.scatter_output_to_slice(
                        field,
                        plan.eval_mode,
                        &q_outputs[slot],
                        &mut output_locals[slot],
                        out_v.as_mut_slice(),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn execute(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
        add: bool,
    ) -> ReedResult<()> {
        if self.multi_distinct_active_io_fields() {
            return Err(ReedError::Operator(
                "multi-field active operator: use OperatorTrait::apply_field_buffers / apply_add_field_buffers (CpuOperator; single-buffer apply is not supported)".into(),
            ));
        }
        let mut sink = ActiveOutputSink::Single(output);
        self.execute_inner(ActiveInputSource::Single(input), &mut sink, add)
    }

    fn apply_field_buffers_impl<'io>(
        &self,
        inputs: &[(&str, &dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
        add: bool,
    ) -> ReedResult<()> {
        self.validate_named_field_buffers(inputs, &*outputs)?;
        let mut sink = ActiveOutputSink::Named(outputs);
        self.execute_inner(ActiveInputSource::Named(inputs), &mut sink, add)
    }

    fn execute_adjoint(
        &self,
        range_cotangent: &dyn VectorTrait<T>,
        domain_cotangent: &mut dyn VectorTrait<T>,
        add: bool,
    ) -> ReedResult<()> {
        if self.requires_field_named_buffers() {
            return Err(ReedError::Operator(
                "operator adjoint: this operator uses multiple active fields; use OperatorTrait::apply_field_buffers_with_transpose / apply_add_field_buffers_with_transpose (Adjoint) with one buffer per active input/output field name"
                    .into(),
            ));
        }
        let mut sink = ActiveOutputSink::Single(domain_cotangent);
        self.execute_adjoint_inner(ActiveInputSource::Single(range_cotangent), &mut sink, add)
    }

    fn execute_adjoint_field_buffers_impl<'io>(
        &self,
        range: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        domain: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
        add: bool,
    ) -> ReedResult<()> {
        let mut sink = ActiveOutputSink::Named(domain);
        self.execute_adjoint_inner(ActiveInputSource::Named(range), &mut sink, add)
    }
}

impl<'a, T: Scalar> OperatorTrait<T> for CpuOperator<'a, T> {
    fn global_vector_len_hint(&self) -> Option<usize> {
        self.active_global_dof_len().ok()
    }

    fn requires_field_named_buffers(&self) -> bool {
        self.multi_distinct_active_io_fields()
    }

    fn operator_supports_assemble(&self, kind: OperatorAssembleKind) -> bool {
        match kind {
            OperatorAssembleKind::FdmElementInverse => {
                let dense_ok = self
                    .active_global_dof_len()
                    .map(|n| n <= crate::fdm_inverse::FDM_DENSE_MAX_N)
                    .unwrap_or(false);
                dense_ok || self.can_tensor_fdm()
            }
            OperatorAssembleKind::Diagonal
            | OperatorAssembleKind::LinearSymbolic
            | OperatorAssembleKind::LinearNumeric
            | OperatorAssembleKind::LinearCsrNumeric => self.active_global_dof_len().is_ok(),
            _ => false,
        }
    }

    /// **Replaces** any existing dense assembly slot with a freshly zeroed `n×n` buffer and clears
    /// **`numeric_done`** until the next [`OperatorTrait::linear_assemble`] / [`OperatorTrait::linear_assemble_add`].
    fn linear_assemble_symbolic(&self) -> ReedResult<()> {
        self.check_ready()?;
        let n = self.active_global_dof_len()?;
        let slot = DenseLinearAssemblySlot::new_symbolic(n)?;
        *self.dense_linear_assembly.lock().map_err(|_| {
            ReedError::Operator("linear_assemble_symbolic: assembly mutex poisoned".into())
        })? = Some(slot);
        Ok(())
    }

    fn linear_assemble(&self) -> ReedResult<()> {
        let n = self.active_global_dof_len()?;
        {
            let g = self.dense_linear_assembly.lock().map_err(|_| {
                ReedError::Operator("linear_assemble: assembly mutex poisoned".into())
            })?;
            let slot = g.as_ref().ok_or_else(|| {
                ReedError::Operator("linear_assemble: call linear_assemble_symbolic first".into())
            })?;
            if !slot.symbolic_done {
                return Err(ReedError::Operator(
                    "linear_assemble: internal assembly state missing symbolic phase".into(),
                ));
            }
            if slot.n != n {
                return Err(ReedError::Operator(format!(
                    "linear_assemble: symbolic size {} != current active global DOF {}",
                    slot.n, n
                )));
            }
        }
        for j in 0..n {
            let mut input = vec![T::ZERO; n];
            input[j] = T::ONE;
            let x = crate::vector::CpuVector::from_vec(input);
            let mut y = crate::vector::CpuVector::new(n);
            self.apply(&x, &mut y)?;
            let mut g = self.dense_linear_assembly.lock().map_err(|_| {
                ReedError::Operator("linear_assemble: assembly mutex poisoned".into())
            })?;
            let slot = g.as_mut().ok_or_else(|| {
                ReedError::Operator(
                    "linear_assemble: assembly buffer disappeared during fill".into(),
                )
            })?;
            for i in 0..n {
                slot.a[i + j * n] = y.as_slice()[i];
            }
        }
        let mut g = self
            .dense_linear_assembly
            .lock()
            .map_err(|_| ReedError::Operator("linear_assemble: assembly mutex poisoned".into()))?;
        let slot = g.as_mut().ok_or_else(|| {
            ReedError::Operator("linear_assemble: assembly buffer disappeared after fill".into())
        })?;
        slot.numeric_done = true;
        Ok(())
    }

    fn linear_assemble_add(&self) -> ReedResult<()> {
        let n = self.active_global_dof_len()?;
        {
            let g = self.dense_linear_assembly.lock().map_err(|_| {
                ReedError::Operator("linear_assemble_add: assembly mutex poisoned".into())
            })?;
            let slot = g.as_ref().ok_or_else(|| {
                ReedError::Operator(
                    "linear_assemble_add: call linear_assemble_symbolic first".into(),
                )
            })?;
            if !slot.symbolic_done {
                return Err(ReedError::Operator(
                    "linear_assemble_add: internal assembly state missing symbolic phase".into(),
                ));
            }
            if slot.n != n {
                return Err(ReedError::Operator(format!(
                    "linear_assemble_add: symbolic size {} != current active global DOF {}",
                    slot.n, n
                )));
            }
        }
        for j in 0..n {
            let mut input = vec![T::ZERO; n];
            input[j] = T::ONE;
            let x = crate::vector::CpuVector::from_vec(input);
            let mut y = crate::vector::CpuVector::new(n);
            self.apply(&x, &mut y)?;
            let mut g = self.dense_linear_assembly.lock().map_err(|_| {
                ReedError::Operator("linear_assemble_add: assembly mutex poisoned".into())
            })?;
            let slot = g.as_mut().ok_or_else(|| {
                ReedError::Operator(
                    "linear_assemble_add: assembly buffer disappeared during fill".into(),
                )
            })?;
            for i in 0..n {
                slot.a[i + j * n] += y.as_slice()[i];
            }
        }
        let mut g = self.dense_linear_assembly.lock().map_err(|_| {
            ReedError::Operator("linear_assemble_add: assembly mutex poisoned".into())
        })?;
        let slot = g.as_mut().ok_or_else(|| {
            ReedError::Operator(
                "linear_assemble_add: assembly buffer disappeared after fill".into(),
            )
        })?;
        slot.numeric_done = true;
        Ok(())
    }

    fn linear_assemble_csr_matrix(&self, pattern: &CsrPattern) -> ReedResult<CsrMatrix<T>> {
        CpuOperator::linear_assemble_csr_matrix(self, pattern)
    }

    fn linear_assemble_csr_matrix_add(&self, matrix: &mut CsrMatrix<T>) -> ReedResult<()> {
        CpuOperator::linear_assemble_csr_matrix_add(self, matrix)
    }

    fn linear_assemble_ceed_matrix(&self, matrix: &mut CeedMatrix<T>) -> ReedResult<()> {
        CpuOperator::linear_assemble_ceed_matrix(self, matrix)
    }

    fn linear_assemble_add_ceed_matrix(&self, matrix: &mut CeedMatrix<T>) -> ReedResult<()> {
        CpuOperator::linear_assemble_add_ceed_matrix(self, matrix)
    }

    fn operator_create_fdm_element_inverse(&self) -> ReedResult<Box<dyn OperatorTrait<T>>> {
        self.check_ready()?;
        let n = self.active_global_dof_len()?;

        // Try tensor FDM path when the basis supports it (any n).
        if let Some(tensor_inv) = self.try_create_fdm_tensor_inverse()? {
            return Ok(tensor_inv);
        }

        // Fallback: dense inversion for small n.
        if n > crate::fdm_inverse::FDM_DENSE_MAX_N {
            return Err(ReedError::Operator(format!(
                "operator_create_fdm_element_inverse: global DOF {} exceeds dense limit {} and tensor FDM not available for this operator configuration",
                n, crate::fdm_inverse::FDM_DENSE_MAX_N
            )));
        }
        let len = n.checked_mul(n).ok_or_else(|| {
            ReedError::Operator("operator_create_fdm_element_inverse: n*n overflow".into())
        })?;
        let mut a_vec = vec![T::ZERO; len];
        for j in 0..n {
            let mut input = vec![T::ZERO; n];
            input[j] = T::ONE;
            let x = crate::vector::CpuVector::from_vec(input);
            let mut y = crate::vector::CpuVector::new(n);
            self.apply(&x, &mut y)?;
            for i in 0..n {
                a_vec[i + j * n] = y.as_slice()[i];
            }
        }
        let inv = crate::fdm_inverse::invert_dense_col_major(&a_vec, n)?;
        Ok(Box::new(
            crate::fdm_inverse::CpuFdmDenseInverseOperator::new(n, inv),
        ))
    }

    fn operator_create_fdm_element_inverse_jacobi(&self) -> ReedResult<Box<dyn OperatorTrait<T>>> {
        CpuOperator::operator_create_fdm_element_inverse_jacobi(self)
    }

    fn operator_label(&self) -> Option<&str> {
        self.op_label.as_deref()
    }

    fn check_ready(&self) -> ReedResult<()> {
        for field in &self.fields {
            if let FieldVector::Passive(v) = &field.vector {
                if let Some(r) = field.restriction {
                    let need = r.num_global_dof();
                    if v.len() != need {
                        return Err(ReedError::Operator(format!(
                            "check_ready: passive field '{}' vector length {} != restriction global DOF {}",
                            field.name,
                            v.len(),
                            need
                        )));
                    }
                }
            }
            if let Some(r) = field.restriction {
                if r.num_elements() != self.num_elem {
                    return Err(ReedError::Operator(format!(
                        "check_ready: field '{}' restriction num_elements {} != operator num_elements {}",
                        field.name,
                        r.num_elements(),
                        self.num_elem
                    )));
                }
            }
            if let Some(b) = field.basis {
                if b.num_qpoints() != self.num_qpoints {
                    return Err(ReedError::Operator(format!(
                        "check_ready: field '{}' basis num_qpoints {} != operator num_qpoints {}",
                        field.name,
                        b.num_qpoints(),
                        self.num_qpoints
                    )));
                }
            }
        }
        for p in &self.input_plans {
            let field = &self.fields[p.field_index];
            let _ = Self::qpoint_component_count(field, p.eval_mode)?;
        }
        for p in &self.output_plans {
            let field = &self.fields[p.field_index];
            let _ = Self::qpoint_component_count(field, p.eval_mode)?;
        }
        Ok(())
    }

    fn apply_field_buffers<'io>(
        &self,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        self.apply_field_buffers_impl(inputs, outputs, false)
    }

    fn apply_add_field_buffers<'io>(
        &self,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        self.apply_field_buffers_impl(inputs, outputs, true)
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
                self.execute_adjoint_field_buffers_impl(inputs, outputs, false)
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
                self.execute_adjoint_field_buffers_impl(inputs, outputs, true)
            }
        }
    }

    fn apply(&self, input: &dyn VectorTrait<T>, output: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        self.execute(input, output, false)
    }

    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        self.execute(input, output, true)
    }

    fn apply_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply(input, output),
            OperatorTransposeRequest::Adjoint => self.execute_adjoint(input, output, false),
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
            OperatorTransposeRequest::Adjoint => self.execute_adjoint(input, output, true),
        }
    }

    fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        let n = self.active_global_dof_len()?;
        if assembled.len() != n {
            return Err(ReedError::Operator(format!(
                "linear_assemble_diagonal: assembled length {} != active global DOF count {}",
                assembled.len(),
                n
            )));
        }
        assembled.set_value(T::ZERO)?;
        for i in 0..n {
            let mut input = vec![T::ZERO; n];
            input[i] = T::ONE;
            let x = crate::vector::CpuVector::from_vec(input);
            let mut y = crate::vector::CpuVector::new(n);
            self.apply(&x, &mut y)?;
            assembled.as_mut_slice()[i] = y.as_slice()[i];
        }
        Ok(())
    }

    fn linear_assemble_add_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        let n = self.active_global_dof_len()?;
        if assembled.len() != n {
            return Err(ReedError::Operator(format!(
                "linear_assemble_add_diagonal: assembled length {} != active global DOF count {}",
                assembled.len(),
                n
            )));
        }
        for i in 0..n {
            let mut input = vec![T::ZERO; n];
            input[i] = T::ONE;
            let x = crate::vector::CpuVector::from_vec(input);
            let mut y = crate::vector::CpuVector::new(n);
            self.apply(&x, &mut y)?;
            assembled.as_mut_slice()[i] += y.as_slice()[i];
        }
        Ok(())
    }
}

/// Inherent aliases for [`OperatorTrait::apply_field_buffers`] on [`CpuOperator`] (optional import).
impl<'a, T: Scalar> CpuOperator<'a, T> {
    /// Apply with one global vector per active field (names match [`OperatorBuilder::field`]).
    pub fn apply_field_buffers<'io>(
        &self,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        OperatorTrait::apply_field_buffers(self, inputs, outputs)
    }

    /// Same as [`Self::apply_field_buffers`] but accumulates into output vectors.
    pub fn apply_add_field_buffers<'io>(
        &self,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        OperatorTrait::apply_add_field_buffers(self, inputs, outputs)
    }

    pub fn apply_field_buffers_with_transpose<'io>(
        &self,
        request: OperatorTransposeRequest,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        OperatorTrait::apply_field_buffers_with_transpose(self, request, inputs, outputs)
    }

    pub fn apply_add_field_buffers_with_transpose<'io>(
        &self,
        request: OperatorTransposeRequest,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        OperatorTrait::apply_add_field_buffers_with_transpose(self, request, inputs, outputs)
    }
}

#[cfg(test)]
mod adjoint_field_buffer_tests {
    use super::*;
    use crate::basis_lagrange::LagrangeBasis;
    use crate::elem_restriction::CpuElemRestriction;
    use crate::gallery::MassApplyInterpTimesWeight;
    use crate::vector::CpuVector;
    use reed_core::enums::{EvalMode, QuadMode};
    use reed_core::operator::OperatorTrait;
    use reed_core::qfunction::{QFunctionField, QFunctionTrait};
    use reed_core::scalar::Scalar;

    /// `v = u + aux` at quadrature (all `Interp`); discrete adjoint splits output cotangent onto both inputs.
    #[derive(Clone)]
    struct SumTwoInterpQf {
        inputs: Vec<QFunctionField>,
        outputs: Vec<QFunctionField>,
    }

    impl SumTwoInterpQf {
        fn new() -> Self {
            Self {
                inputs: vec![
                    QFunctionField {
                        name: "u".into(),
                        num_comp: 1,
                        eval_mode: EvalMode::Interp,
                    },
                    QFunctionField {
                        name: "aux".into(),
                        num_comp: 1,
                        eval_mode: EvalMode::Interp,
                    },
                ],
                outputs: vec![QFunctionField {
                    name: "v".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Interp,
                }],
            }
        }
    }

    impl<T: Scalar> QFunctionTrait<T> for SumTwoInterpQf {
        fn inputs(&self) -> &[QFunctionField] {
            &self.inputs
        }

        fn outputs(&self) -> &[QFunctionField] {
            &self.outputs
        }

        fn apply(
            &self,
            _ctx: &[u8],
            q: usize,
            inputs: &[&[T]],
            outputs: &mut [&mut [T]],
        ) -> ReedResult<()> {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(ReedError::QFunction(
                    "SumTwoInterpQf: expected 2 inputs and 1 output".into(),
                ));
            }
            let u = inputs[0];
            let aux = inputs[1];
            let v = &mut outputs[0];
            if u.len() != q || aux.len() != q || v.len() != q {
                return Err(ReedError::QFunction(
                    "SumTwoInterpQf: length mismatch".into(),
                ));
            }
            for i in 0..q {
                v[i] = u[i] + aux[i];
            }
            Ok(())
        }

        fn supports_operator_transpose(&self) -> bool {
            true
        }

        fn apply_operator_transpose(
            &self,
            _ctx: &[u8],
            q: usize,
            output_cotangents: &[&[T]],
            input_cotangents: &mut [&mut [T]],
        ) -> ReedResult<()> {
            if output_cotangents.len() != 1 || input_cotangents.len() != 2 {
                return Err(ReedError::QFunction(
                    "SumTwoInterpQf transpose: expected 1 output cotangent and 2 input buffers"
                        .into(),
                ));
            }
            let dv = output_cotangents[0];
            if dv.len() != q {
                return Err(ReedError::QFunction(
                    "SumTwoInterpQf transpose: length mismatch".into(),
                ));
            }
            let (du_buf, daux_buf) = input_cotangents.split_at_mut(1);
            let du = &mut du_buf[0];
            let daux = &mut daux_buf[0];
            if du.len() != q || daux.len() != q {
                return Err(ReedError::QFunction(
                    "SumTwoInterpQf transpose: length mismatch".into(),
                ));
            }
            for i in 0..q {
                du[i] += dv[i];
                daux[i] += dv[i];
            }
            Ok(())
        }
    }

    fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn named_field_buffers_adjoint_inner_product_identity() -> ReedResult<()> {
        let nelem = 2usize;
        let p = 2usize;
        let q = 2usize;
        let ndofs = nelem + 1;
        let ind = vec![0i32, 1, 1, 2];
        let r = CpuElemRestriction::<f64>::new_offset(nelem, p, 1, 1, ndofs, &ind)?;
        let b = LagrangeBasis::new(1, 1, p, q, QuadMode::Gauss)?;

        let op = OperatorBuilder::new()
            .qfunction(Box::new(SumTwoInterpQf::new()) as Box<dyn QFunctionTrait<f64>>)
            .field("u", Some(&r), Some(&b), FieldVector::Active)
            .field("aux", Some(&r), Some(&b), FieldVector::Active)
            .field("v", Some(&r), Some(&b), FieldVector::Active)
            .build()?;

        let u = CpuVector::from_vec(vec![1.0, 0.5, -0.25]);
        let aux = CpuVector::from_vec(vec![2.0, -1.0, 0.0]);
        let mut v = CpuVector::new(ndofs);
        v.set_value(0.0)?;
        let ins = [
            ("u", &u as &dyn VectorTrait<f64>),
            ("aux", &aux as &dyn VectorTrait<f64>),
        ];
        let mut outs = [("v", &mut v as &mut dyn VectorTrait<f64>)];
        OperatorTrait::apply_field_buffers(&op, &ins, &mut outs)?;

        let dv = CpuVector::from_vec(vec![0.3, -0.7, 0.05]);
        let mut du = CpuVector::new(ndofs);
        let mut daux = CpuVector::new(ndofs);
        du.set_value(0.0)?;
        daux.set_value(0.0)?;
        let range_in = [("v", &dv as &dyn VectorTrait<f64>)];
        let mut domain_out = [
            ("u", &mut du as &mut dyn VectorTrait<f64>),
            ("aux", &mut daux as &mut dyn VectorTrait<f64>),
        ];
        OperatorTrait::apply_field_buffers_with_transpose(
            &op,
            OperatorTransposeRequest::Adjoint,
            &range_in,
            &mut domain_out,
        )?;

        let lhs = dot_f64(v.as_slice(), dv.as_slice());
        let rhs = dot_f64(u.as_slice(), du.as_slice()) + dot_f64(aux.as_slice(), daux.as_slice());
        assert!(
            (lhs - rhs).abs() < 1e-9_f64.max(1e-9 * lhs.abs()),
            "inner product identity: lhs={lhs} rhs={rhs}"
        );
        Ok(())
    }

    /// `MassApplyInterpTimesWeight` with passive **`EvalMode::Weight`** slot: `<M u, w> = <u, M^* w>``.
    #[test]
    fn mass_interp_times_passive_weight_adjoint_inner_product_identity() -> ReedResult<()> {
        let nelem = 2usize;
        let p = 2usize;
        let q = 2usize;
        let ndofs = nelem + 1;
        let ind = vec![0i32, 1, 1, 2];
        let r = CpuElemRestriction::<f64>::new_offset(nelem, p, 1, 1, ndofs, &ind)?;
        let b = LagrangeBasis::new(1, 1, p, q, QuadMode::Gauss)?;
        let passive_dummy = CpuVector::from_vec(vec![0.0_f64]);

        let op = OperatorBuilder::new()
            .qfunction(
                Box::new(MassApplyInterpTimesWeight::default()) as Box<dyn QFunctionTrait<f64>>
            )
            .field("u", Some(&r), Some(&b), FieldVector::Active)
            .field("w", None, Some(&b), FieldVector::Passive(&passive_dummy))
            .field("v", Some(&r), Some(&b), FieldVector::Active)
            .build()?;

        let u = CpuVector::from_vec(vec![1.0, 0.5, -0.25]);
        let w = CpuVector::from_vec(vec![0.3, -0.7, 0.05]);

        let mut mu = CpuVector::new(ndofs);
        mu.set_value(0.0)?;
        OperatorTrait::apply(&op, &u, &mut mu)?;

        let mut du = CpuVector::new(ndofs);
        du.set_value(0.0)?;
        OperatorTrait::apply_with_transpose(&op, OperatorTransposeRequest::Adjoint, &w, &mut du)?;

        let lhs = dot_f64(mu.as_slice(), w.as_slice());
        let rhs = dot_f64(u.as_slice(), du.as_slice());
        assert!(
            (lhs - rhs).abs() < 1e-9_f64.max(1e-9 * lhs.abs()),
            "inner product identity (interp × qp-weight passive): lhs={lhs} rhs={rhs}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod clear_dense_linear_assembly_tests {
    use super::*;
    use crate::basis_lagrange::LagrangeBasis;
    use crate::elem_restriction::CpuElemRestriction;
    use crate::gallery::{Identity, MassApply};
    use crate::vector::CpuVector;
    use reed_core::enums::QuadMode;
    use reed_core::matrix::{CeedMatrix, CeedMatrixStorage};
    use reed_core::operator::OperatorTrait;
    use reed_core::qfunction::QFunctionTrait;

    #[test]
    fn clear_dense_linear_assembly_idempotent_without_slot() -> ReedResult<()> {
        let nelem = 1usize;
        let p = 2usize;
        let q = 1usize;
        let ndofs = 2usize;
        let ind = vec![0i32, 1];
        let r = CpuElemRestriction::<f64>::new_offset(nelem, p, 1, 1, ndofs, &ind)?;
        let b = LagrangeBasis::new(1, 1, p, q, QuadMode::Gauss)?;
        let op = OperatorBuilder::new()
            .qfunction(Box::new(Identity::default()) as Box<dyn QFunctionTrait<f64>>)
            .field("input", Some(&r), Some(&b), FieldVector::Active)
            .field("output", Some(&r), Some(&b), FieldVector::Active)
            .build()?;
        assert_eq!(op.dense_linear_assembly_n(), None);
        assert!(!op.dense_linear_assembly_numeric_ready());
        op.clear_dense_linear_assembly()?;
        op.clear_dense_linear_assembly()?;
        assert!(op.assembled_linear_matrix_col_major().is_none());
        Ok(())
    }

    #[test]
    fn dense_linear_assembly_probes_track_symbolic_numeric_and_clear() -> ReedResult<()> {
        let nelem = 1usize;
        let p = 2usize;
        let q = 1usize;
        let ndofs = 2usize;
        let ind = vec![0i32, 1];
        let r = CpuElemRestriction::<f64>::new_offset(nelem, p, 1, 1, ndofs, &ind)?;
        let b = LagrangeBasis::new(1, 1, p, q, QuadMode::Gauss)?;
        let op = OperatorBuilder::new()
            .qfunction(Box::new(Identity::default()) as Box<dyn QFunctionTrait<f64>>)
            .field("input", Some(&r), Some(&b), FieldVector::Active)
            .field("output", Some(&r), Some(&b), FieldVector::Active)
            .build()?;
        assert_eq!(op.dense_linear_assembly_n(), None);
        assert!(!op.dense_linear_assembly_numeric_ready());

        OperatorTrait::linear_assemble_symbolic(&op)?;
        assert_eq!(op.dense_linear_assembly_n(), Some(2));
        assert!(!op.dense_linear_assembly_numeric_ready());
        assert!(op.assembled_linear_matrix_col_major().is_none());

        OperatorTrait::linear_assemble(&op)?;
        assert_eq!(op.dense_linear_assembly_n(), Some(2));
        assert!(op.dense_linear_assembly_numeric_ready());
        assert!(op.assembled_linear_matrix_col_major().is_some());

        op.clear_dense_linear_assembly()?;
        assert_eq!(op.dense_linear_assembly_n(), None);
        assert!(!op.dense_linear_assembly_numeric_ready());
        assert!(op.assembled_linear_matrix_col_major().is_none());
        Ok(())
    }

    #[test]
    fn second_linear_assemble_symbolic_resets_numeric_state() -> ReedResult<()> {
        let nelem = 1usize;
        let p = 2usize;
        let q = 1usize;
        let ndofs = 2usize;
        let ind = vec![0i32, 1];
        let r = CpuElemRestriction::<f64>::new_offset(nelem, p, 1, 1, ndofs, &ind)?;
        let b = LagrangeBasis::new(1, 1, p, q, QuadMode::Gauss)?;
        let op = OperatorBuilder::new()
            .qfunction(Box::new(Identity::default()) as Box<dyn QFunctionTrait<f64>>)
            .field("input", Some(&r), Some(&b), FieldVector::Active)
            .field("output", Some(&r), Some(&b), FieldVector::Active)
            .build()?;

        OperatorTrait::linear_assemble_symbolic(&op)?;
        OperatorTrait::linear_assemble(&op)?;
        assert!(op.dense_linear_assembly_numeric_ready());
        let (n, a1) = op.assembled_linear_matrix_col_major().unwrap();

        OperatorTrait::linear_assemble_symbolic(&op)?;
        assert_eq!(op.dense_linear_assembly_n(), Some(n));
        assert!(!op.dense_linear_assembly_numeric_ready());
        assert!(op.assembled_linear_matrix_col_major().is_none());

        OperatorTrait::linear_assemble(&op)?;
        assert!(op.dense_linear_assembly_numeric_ready());
        let (_, a2) = op.assembled_linear_matrix_col_major().unwrap();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (a1[i + j * n] - a2[i + j * n]).abs() < 1e-11,
                    "second symbolic+assemble should match first A at ({i},{j})"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn fdm_creation_does_not_mutate_dense_slot() -> ReedResult<()> {
        let nelem = 1usize;
        let p = 2usize;
        let q = 2usize;
        let ndofs = 2usize;
        let ind = vec![0i32, 1];
        let r = CpuElemRestriction::<f64>::new_offset(nelem, p, 1, 1, ndofs, &ind)?;
        let r_q = CpuElemRestriction::<f64>::new_strided(
            nelem,
            q,
            1,
            nelem * q,
            [1, q as i32, q as i32],
        )?;
        let b = LagrangeBasis::new(1, 1, p, q, QuadMode::Gauss)?;
        let mut qdata = CpuVector::new(nelem * q);
        qdata.set_value(1.0)?;
        let op = OperatorBuilder::new()
            .qfunction(Box::new(MassApply::default()) as Box<dyn QFunctionTrait<f64>>)
            .field("u", Some(&r), Some(&b), FieldVector::Active)
            .field("qdata", Some(&r_q), None, FieldVector::Passive(&qdata))
            .field("v", Some(&r), Some(&b), FieldVector::Active)
            .build()?;

        OperatorTrait::linear_assemble_symbolic(&op)?;
        OperatorTrait::linear_assemble(&op)?;
        let (n, a) = op.assembled_linear_matrix_col_major().unwrap();
        OperatorTrait::linear_assemble_add(&op)?;
        let (_, a_twice) = op.assembled_linear_matrix_col_major().unwrap();
        for i in 0..n {
            for j in 0..n {
                assert!((a_twice[i + j * n] - 2.0 * a[i + j * n]).abs() < 1e-11);
            }
        }

        let _inv = OperatorTrait::operator_create_fdm_element_inverse(&op)?;
        let (_, after) = op.assembled_linear_matrix_col_major().unwrap();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (after[i + j * n] - a_twice[i + j * n]).abs() < 1e-11,
                    "fdm creation should not mutate dense slot at ({i},{j})"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn ceed_matrix_handle_dense_and_csr_set_add_paths() -> ReedResult<()> {
        let nelem = 1usize;
        let p = 2usize;
        let q = 1usize;
        let ndofs = 2usize;
        let ind = vec![0i32, 1];
        let r = CpuElemRestriction::<f64>::new_offset(nelem, p, 1, 1, ndofs, &ind)?;
        let b = LagrangeBasis::new(1, 1, p, q, QuadMode::Gauss)?;
        let op = OperatorBuilder::new()
            .qfunction(Box::new(Identity::default()) as Box<dyn QFunctionTrait<f64>>)
            .field("input", Some(&r), Some(&b), FieldVector::Active)
            .field("output", Some(&r), Some(&b), FieldVector::Active)
            .build()?;

        let mut dense = CeedMatrix::<f64>::dense_col_major_symbolic(ndofs, ndofs)?;
        op.linear_assemble_ceed_matrix(&mut dense)?;
        let dense_once = match dense.storage() {
            CeedMatrixStorage::DenseColMajor { values, .. } => values.clone(),
            _ => panic!("expected dense matrix handle"),
        };
        op.linear_assemble_add_ceed_matrix(&mut dense)?;
        match dense.storage() {
            CeedMatrixStorage::DenseColMajor { values, .. } => {
                for i in 0..values.len() {
                    assert!((values[i] - 2.0 * dense_once[i]).abs() < 1e-12);
                }
            }
            _ => panic!("expected dense matrix handle"),
        }

        let pat = r.assembled_csr_pattern()?;
        let mut csr = CeedMatrix::<f64>::csr_symbolic(pat);
        op.linear_assemble_ceed_matrix(&mut csr)?;
        let csr_once = match csr.storage() {
            CeedMatrixStorage::Csr(m) => m.values.clone(),
            _ => panic!("expected csr matrix handle"),
        };
        op.linear_assemble_add_ceed_matrix(&mut csr)?;
        match csr.storage() {
            CeedMatrixStorage::Csr(m) => {
                for (i, &v) in m.values.iter().enumerate() {
                    assert!((v - 2.0 * csr_once[i]).abs() < 1e-12);
                }
            }
            _ => panic!("expected csr matrix handle"),
        }
        Ok(())
    }
}
