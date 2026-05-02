//! GPU end-to-end operator pipeline: restriction -> basis -> QFunction -> basis^T -> restriction^T.
//!
//! `WgpuOperator<T>` mirrors the `CpuOperator` pattern but executes restriction and basis stages
//! on GPU via [`GpuRuntime`] WGSL compute pipelines. In v1, the QFunction still runs on CPU
//! (host-side data after GPU readback), and the transpose stages are also GPU-accelerated.
//!
//! ## Architecture
//!
//! ```text
//! WgpuOperator::apply(x_global, y_global):
//!   1. For each input field: restriction gather (GPU) -> element-local buffer
//!   2. For each input field: basis apply (GPU) -> q-point buffer
//!   3. QFunction dispatch at q-points (CPU for v1)
//!   4. For each output field: basis^T apply (GPU) -> element-local buffer
//!   5. For each output field: restriction scatter (GPU) -> accumulate to y_global
//! ```
//!
//! ## v1 scope
//!
//! - Forward apply with single active input/output vectors
//! - QFunction runs on CPU with host data (GPU readback between stages)
//! - No multi-field named-buffer apply in v1
//! - No adjoint/transpose in v1

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use reed_core::{
    basis::BasisTrait,
    elem_restriction::ElemRestrictionTrait,
    enums::{EvalMode, TransposeMode},
    error::{ReedError, ReedResult},
    operator::{OperatorTrait, OperatorTransposeRequest},
    qfunction::QFunctionTrait,
    scalar::Scalar,
    vector::VectorTrait,
    QFunctionContext,
};

use crate::runtime::GpuRuntime;

// ---------------------------------------------------------------------------
// Plan types (mirrors CpuOperator)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// WgpuFieldVector -- owned variant of FieldVector
// ---------------------------------------------------------------------------

/// Owned field-vector role for [`WgpuOperatorField`].
///
/// - [`Active`](WgpuFieldVector::Active): the vector is supplied at `apply()` time
///   (single global buffer on the input side, accumulated into on the output side).
/// - [`Passive`](WgpuFieldVector::Passive): a pre-set owned vector (e.g. `qdata`).
/// - [`None`](WgpuFieldVector::None): no associated vector (e.g. weight-only fields).
pub enum WgpuFieldVector<T: Scalar> {
    Active,
    Passive(Box<dyn VectorTrait<T>>),
    None,
}

// ---------------------------------------------------------------------------
// WgpuOperatorField
// ---------------------------------------------------------------------------

/// One field in a [`WgpuOperator`]: a named slot with optional restriction, basis,
/// and passive vector. Active fields receive their vector from the `apply()` arguments.
pub struct WgpuOperatorField<T: Scalar> {
    name: String,
    restriction: Option<Box<dyn ElemRestrictionTrait<T>>>,
    basis: Option<Box<dyn BasisTrait<T>>>,
    vector: WgpuFieldVector<T>,
}

// ---------------------------------------------------------------------------
// WgpuOperator
// ---------------------------------------------------------------------------

/// GPU end-to-end FE operator: restriction -> basis -> QFunction -> basis^T -> restriction^T.
///
/// Owns its restriction and basis objects (unlike [`CpuOperator`](reed_cpu::operator::CpuOperator)
/// which borrows them). The restriction and basis stages execute on GPU via
/// [`GpuRuntime`] WGSL pipelines when the inner [`WgpuElemRestriction`](crate::elem_restriction::WgpuElemRestriction)
/// / [`WgpuBasis`](crate::basis::WgpuBasis) have a runtime available.
///
/// # v1 limitations
///
/// - QFunction runs on CPU (data is read back from GPU after restriction/basis).
/// - Single active input/output vector only (no named-buffer multi-field apply).
/// - No adjoint/transpose support.
pub struct WgpuOperator<T: Scalar> {
    runtime: Arc<GpuRuntime>,
    num_elem: usize,
    num_qpoints: usize,
    fields: Vec<WgpuOperatorField<T>>,
    qfunction: Box<dyn QFunctionTrait<T>>,
    qfunction_context: QFunctionContext,
    input_plans: Vec<InputPlan>,
    output_plans: Vec<OutputPlan>,
    num_qfunction_inputs: usize,
    num_qfunction_outputs: usize,
    op_label: Option<String>,
    /// Cached forward quadrature inputs for [`QFunctionTrait::apply_operator_transpose_with_primal`].
    last_forward_q_inputs: Mutex<Option<Vec<Vec<T>>>>,
}

impl<T: Scalar> WgpuOperator<T> {
    /// Number of mesh elements governed by this operator.
    #[inline]
    pub fn num_elements(&self) -> usize {
        self.num_elem
    }

    /// Quadrature points per element.
    #[inline]
    pub fn num_quadrature_points_per_elem(&self) -> usize {
        self.num_qpoints
    }

    /// Shared GPU runtime.
    #[inline]
    pub fn runtime(&self) -> &Arc<GpuRuntime> {
        &self.runtime
    }

    /// Look up a field index by its name.
    fn field_index_by_name(fields: &[WgpuOperatorField<T>], name: &str) -> ReedResult<usize> {
        fields
            .iter()
            .position(|f| f.name == name)
            .ok_or_else(|| ReedError::Operator(format!("field {:?} not found in operator fields", name)))
    }

    /// Number of scalar components per quadrature point for a given field and eval mode.
    /// Mirrors [`CpuOperator::qpoint_component_count`](reed_cpu::operator::CpuOperator).
    fn qpoint_component_count(
        field: &WgpuOperatorField<T>,
        eval_mode: EvalMode,
    ) -> ReedResult<usize> {
        match eval_mode {
            EvalMode::None => {
                if let Some(restriction) = &field.restriction {
                    Ok(restriction.num_comp())
                } else {
                    Err(ReedError::Operator(format!(
                        "field '{}' without basis requires a restriction to infer component count",
                        field.name
                    )))
                }
            }
            EvalMode::Weight => Ok(1),
            EvalMode::Interp => field.basis.as_ref().map(|b| b.num_comp()).ok_or_else(|| {
                ReedError::Operator(format!("field '{}' requires basis for Interp", field.name))
            }),
            EvalMode::Grad => field
                .basis
                .as_ref()
                .map(|b| b.num_comp() * b.dim())
                .ok_or_else(|| {
                    ReedError::Operator(format!("field '{}' requires basis for Grad", field.name))
                }),
            EvalMode::Div => {
                let basis = field.basis.as_ref().ok_or_else(|| {
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
                let basis = field.basis.as_ref().ok_or_else(|| {
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
                let basis = field.basis.as_ref().ok_or_else(|| {
                    ReedError::Operator(format!("field '{}' requires basis for HCurl", field.name))
                })?;
                match basis.dim() {
                    2 => Ok(1),
                    3 => Ok(3),
                    d => Err(ReedError::Operator(format!(
                        "field '{}': HCurl requires dim=2 or 3, got {}",
                        field.name, d
                    ))),
                }
            }
            EvalMode::HDiv => Ok(1),
        }
    }

    // ------------------------------------------------------------------
    // Active field queries (mirrors CpuOperator)
    // ------------------------------------------------------------------

    /// Distinct field indices that appear as [`WgpuFieldVector::Active`] on qfunction **inputs**.
    fn distinct_active_input_field_indices(&self) -> Vec<usize> {
        let mut s = HashSet::new();
        for p in &self.input_plans {
            if matches!(self.fields[p.field_index].vector, WgpuFieldVector::Active) {
                s.insert(p.field_index);
            }
        }
        let mut v: Vec<usize> = s.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// Distinct field indices that appear as [`WgpuFieldVector::Active`] on qfunction **outputs**.
    fn distinct_active_output_field_indices(&self) -> Vec<usize> {
        let mut s = HashSet::new();
        for p in &self.output_plans {
            if matches!(self.fields[p.field_index].vector, WgpuFieldVector::Active) {
                s.insert(p.field_index);
            }
        }
        let mut v: Vec<usize> = s.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// True when the operator has multiple distinct active input or output fields.
    fn multi_distinct_active_io_fields(&self) -> bool {
        self.distinct_active_input_field_indices().len() > 1
            || self.distinct_active_output_field_indices().len() > 1
    }

    /// True when this operator cannot use single-buffer [`OperatorTrait::apply`] and requires
    /// [`OperatorTrait::apply_field_buffers`] (multiple active fields on input and/or output side).
    pub fn requires_field_named_buffers(&self) -> bool {
        self.multi_distinct_active_io_fields()
    }

    // ------------------------------------------------------------------
    // Named-buffer helpers (mirrors CpuOperator)
    // ------------------------------------------------------------------

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

        // Validate active input fields
        for &idx in &self.distinct_active_input_field_indices() {
            let f = &self.fields[idx];
            let v = Self::lookup_named_read(inputs, f.name.as_str())?;
            if let Some(r) = &f.restriction {
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

        // Validate active output fields
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
            if let Some(r) = &f.restriction {
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
            if let Some(r) = f.restriction.as_ref() {
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
            if let Some(r) = f.restriction.as_ref() {
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

    /// Core forward apply: restriction gather -> basis apply -> QFunction -> basis^T -> restriction scatter.
    fn apply_forward(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
        add: bool,
    ) -> ReedResult<()> {
        // Zero the output if not accumulating
        if !add {
            output.set_value(T::ZERO)?;
        }

        // Per-call workspace allocations (avoids Mutex overhead; allocation cost
        // is negligible relative to the GPU dispatch + readback work).
        let mut q_inputs: Vec<Vec<T>> = (0..self.num_qfunction_inputs)
            .map(|_| Vec::new())
            .collect();
        let mut q_outputs: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();
        let mut input_locals: Vec<Vec<T>> = (0..self.num_qfunction_inputs)
            .map(|_| Vec::new())
            .collect();
        let mut output_locals: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();

        // Step 1-2: For each input field, restriction gather + basis apply -> q_inputs
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

        // Cache forward q_inputs for adjoint (apply_operator_transpose_with_primal)
        if let Ok(mut cache) = self.last_forward_q_inputs.lock() {
            *cache = Some(q_inputs.clone());
        }

        // Step 3: Resize output q-point buffers and call QFunction
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

        // Step 4-5: For each output field, basis^T apply + restriction scatter -> output
        let out_sl = output.as_mut_slice();
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

        Ok(())
    }

    /// Multi-field forward apply: one buffer per active field (libCEED `CeedOperatorApply` with
    /// field maps). Delegates to the same restriction -> basis -> QFunction -> basis^T -> restriction^T
    /// pipeline but resolves active slices from the named maps.
    fn apply_forward_field_buffers(
        &self,
        inputs: &[(&str, &dyn VectorTrait<T>)],
        outputs: &mut [(&str, &mut dyn VectorTrait<T>)],
        add: bool,
    ) -> ReedResult<()> {
        self.validate_named_field_buffers(inputs, &*outputs)?;

        // Zero active output fields if not accumulating
        if !add {
            let zero_names: HashSet<&str> = self
                .distinct_active_output_field_indices()
                .iter()
                .map(|&i| self.fields[i].name.as_str())
                .collect();
            for (k, v) in outputs.iter_mut() {
                if zero_names.contains(*k) {
                    v.set_value(T::ZERO)?;
                }
            }
        }

        let mut q_inputs: Vec<Vec<T>> = (0..self.num_qfunction_inputs)
            .map(|_| Vec::new())
            .collect();
        let mut q_outputs: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();
        let mut input_locals: Vec<Vec<T>> = (0..self.num_qfunction_inputs)
            .map(|_| Vec::new())
            .collect();
        let mut output_locals: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();

        // Step 1-2: For each input field, restriction gather + basis apply -> q_inputs
        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            let source = Self::lookup_named_read(inputs, field.name.as_str())?;
            self.prepare_input_from_slice(
                field,
                plan.eval_mode,
                source.as_slice(),
                &mut input_locals[slot],
                &mut q_inputs[slot],
            )?;
        }

        // Cache forward q_inputs for adjoint
        if let Ok(mut cache) = self.last_forward_q_inputs.lock() {
            *cache = Some(q_inputs.clone());
        }

        // Step 3: QFunction dispatch
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

        // Step 4-5: For each output field, basis^T + restriction scatter
        for (slot, plan) in self.output_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            let j = Self::lookup_named_write_slot(&*outputs, field.name.as_str())?;
            let out_v = &mut *outputs[j].1;
            self.scatter_output_to_slice(
                field,
                plan.eval_mode,
                &q_outputs[slot],
                &mut output_locals[slot],
                out_v.as_mut_slice(),
            )?;
        }

        Ok(())
    }

    /// Core adjoint apply: range cotangent pull -> QFunction^T -> domain cotangent push.
    ///
    /// Follows [`CpuOperator::execute_adjoint_inner`] exactly:
    /// 1. Passive input fields: evaluate forward (for qdata, etc.)
    /// 2. Output fields: pull range cotangent to q-point (restriction gather + basis forward)
    /// 3. QFunction `apply_operator_transpose` or `apply_operator_transpose_with_primal`
    /// 4. Active input fields: push domain cotangent from q-point (basis transpose + restriction scatter)
    fn apply_adjoint(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
        add: bool,
    ) -> ReedResult<()> {
        if self.multi_distinct_active_io_fields() {
            return Err(ReedError::Operator(
                "operator adjoint: this operator uses multiple active fields; use OperatorTrait::apply_field_buffers_with_transpose / apply_add_field_buffers_with_transpose (Adjoint) with one buffer per active input/output field name".into(),
            ));
        }

        if !self.qfunction.supports_operator_transpose() {
            return Err(ReedError::Operator(
                "operator adjoint: qfunction does not implement apply_operator_transpose".into(),
            ));
        }

        // Zero domain cotangent if not accumulating
        if !add {
            output.set_value(T::ZERO)?;
        }

        // Workspace buffers
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

        // Step 1: Evaluate passive input fields forward (to get passive qdata)
        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            match &field.vector {
                WgpuFieldVector::Passive(_) => {
                    self.prepare_input_into(
                        field,
                        plan.eval_mode,
                        input,
                        &mut input_locals[slot],
                        &mut q_passive_fwd[slot],
                    )?;
                }
                WgpuFieldVector::Active => {
                    q_passive_fwd[slot].clear();
                }
                WgpuFieldVector::None => {
                    return Err(ReedError::Operator(format!(
                        "operator adjoint: input field '{}' has no vector source",
                        field.name
                    )));
                }
            }
        }

        // Step 2: Pull range cotangent from global to q-point for each output field
        let range_sl = input.as_slice();
        for (slot, plan) in self.output_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            self.pull_range_cotangent_to_qp(
                field,
                plan.eval_mode,
                range_sl,
                &mut output_locals[slot],
                &mut q_out_cot[slot],
            )?;
        }

        // Step 3: Resize input cotangent buffers and call QFunction transpose
        let input_descriptors = self.qfunction.inputs();
        for slot in 0..self.num_qfunction_inputs {
            let len = self.num_elem * self.num_qpoints * input_descriptors[slot].num_comp;
            q_in_cot[slot].resize(len, T::ZERO);
            if matches!(
                self.fields[self.input_plans[slot].field_index].vector,
                WgpuFieldVector::Passive(_)
            ) {
                q_in_cot[slot].copy_from_slice(&q_passive_fwd[slot]);
            }
        }

        let out_cot_refs: Vec<&[T]> = q_out_cot.iter().map(Vec::as_slice).collect();
        let mut in_cot_mut: Vec<&mut [T]> =
            q_in_cot.iter_mut().map(|v| v.as_mut_slice()).collect();
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

        // Step 4: Push domain cotangent from q-point to global for each active input field
        let dom_sl = output.as_mut_slice();
        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            if !matches!(field.vector, WgpuFieldVector::Active) {
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

        Ok(())
    }

    /// Multi-field adjoint apply: range cotangent named buffers -> QFunction^T -> domain cotangent
    /// named buffers.
    fn apply_adjoint_field_buffers(
        &self,
        range_inputs: &[(&str, &dyn VectorTrait<T>)],
        domain_outputs: &mut [(&str, &mut dyn VectorTrait<T>)],
        add: bool,
    ) -> ReedResult<()> {
        if !self.qfunction.supports_operator_transpose() {
            return Err(ReedError::Operator(
                "operator adjoint: qfunction does not implement apply_operator_transpose".into(),
            ));
        }

        self.validate_adjoint_named_field_buffers(range_inputs, &*domain_outputs)?;

        // Zero domain cotangent if not accumulating
        if !add {
            let zero_names: HashSet<&str> = self
                .distinct_active_input_field_indices()
                .iter()
                .map(|&i| self.fields[i].name.as_str())
                .collect();
            for (k, v) in domain_outputs.iter_mut() {
                if zero_names.contains(*k) {
                    v.set_value(T::ZERO)?;
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

        // Step 1: Evaluate passive input fields forward
        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            match &field.vector {
                WgpuFieldVector::Passive(_) => {
                    let source = Self::lookup_named_read(range_inputs, field.name.as_str())?;
                    self.prepare_input_from_slice(
                        field,
                        plan.eval_mode,
                        source.as_slice(),
                        &mut input_locals[slot],
                        &mut q_passive_fwd[slot],
                    )?;
                }
                WgpuFieldVector::Active => {
                    q_passive_fwd[slot].clear();
                }
                WgpuFieldVector::None => {
                    return Err(ReedError::Operator(format!(
                        "operator adjoint: input field '{}' has no vector source",
                        field.name
                    )));
                }
            }
        }

        // Step 2: Pull range cotangent from global to q-point for each output field
        for (slot, plan) in self.output_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            let range_sl =
                Self::lookup_named_read(range_inputs, field.name.as_str())?.as_slice();
            self.pull_range_cotangent_to_qp(
                field,
                plan.eval_mode,
                range_sl,
                &mut output_locals[slot],
                &mut q_out_cot[slot],
            )?;
        }

        // Step 3: QFunction transpose
        let input_descriptors = self.qfunction.inputs();
        for slot in 0..self.num_qfunction_inputs {
            let len = self.num_elem * self.num_qpoints * input_descriptors[slot].num_comp;
            q_in_cot[slot].resize(len, T::ZERO);
            if matches!(
                self.fields[self.input_plans[slot].field_index].vector,
                WgpuFieldVector::Passive(_)
            ) {
                q_in_cot[slot].copy_from_slice(&q_passive_fwd[slot]);
            }
        }

        let out_cot_refs: Vec<&[T]> = q_out_cot.iter().map(Vec::as_slice).collect();
        let mut in_cot_mut: Vec<&mut [T]> =
            q_in_cot.iter_mut().map(|v| v.as_mut_slice()).collect();
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

        // Step 4: Push domain cotangent from q-point to global for each active input field
        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            if !matches!(field.vector, WgpuFieldVector::Active) {
                continue;
            }
            let j = Self::lookup_named_write_slot(domain_outputs, field.name.as_str())?;
            let out_v = &mut *domain_outputs[j].1;
            self.scatter_domain_cotangent_qp_to_global(
                field,
                plan.eval_mode,
                &q_in_cot[slot],
                &mut input_locals[slot],
                out_v.as_mut_slice(),
            )?;
        }

        Ok(())
    }

    /// Restriction gather + basis apply for one input field.
    ///
    /// GPU acceleration kicks in when the field's restriction is a
    /// [`WgpuElemRestriction`](crate::elem_restriction::WgpuElemRestriction) and/or the
    /// basis is a [`WgpuBasis`](crate::basis::WgpuBasis) with an active runtime.
    fn prepare_input_into(
        &self,
        field: &WgpuOperatorField<T>,
        eval_mode: EvalMode,
        active_input: &dyn VectorTrait<T>,
        local_buffer: &mut Vec<T>,
        q_buffer: &mut Vec<T>,
    ) -> ReedResult<()> {
        // Resolve active source slice from the vector
        let source: &[T] = active_input.as_slice();
        self.prepare_input_from_slice(field, eval_mode, source, local_buffer, q_buffer)
    }

    /// Restriction gather + basis apply for one input field, using a pre-resolved source slice.
    ///
    /// This is the workhorse behind [`Self::prepare_input_into`]; it accepts a raw `&[T]`
    /// so that named-buffer multi-field paths can resolve the active slice once and pass it in.
    fn prepare_input_from_slice(
        &self,
        field: &WgpuOperatorField<T>,
        eval_mode: EvalMode,
        active_source: &[T],
        local_buffer: &mut Vec<T>,
        q_buffer: &mut Vec<T>,
    ) -> ReedResult<()> {
        // Weight mode: no vector source needed; basis computes quadrature weights directly
        if matches!(eval_mode, EvalMode::Weight) {
            let basis = field.basis.as_ref().ok_or_else(|| {
                ReedError::Operator(format!(
                    "field '{}' requires a basis for Weight eval mode",
                    field.name
                ))
            })?;
            q_buffer.resize(self.num_elem * basis.num_qpoints(), T::ZERO);
            basis.apply(self.num_elem, false, EvalMode::Weight, &[], q_buffer)?;
            return Ok(());
        }

        // Resolve the source vector
        let source: &[T] = match &field.vector {
            WgpuFieldVector::Active => active_source,
            WgpuFieldVector::Passive(v) => v.as_slice(),
            WgpuFieldVector::None => {
                return Err(ReedError::Operator(format!(
                    "field '{}' has no vector source (set Active or Passive)",
                    field.name
                )));
            }
        };

        // Restriction gather (GPU-accelerated if WgpuElemRestriction with runtime)
        let local = if let Some(restriction) = &field.restriction {
            local_buffer.resize(restriction.local_size(), T::ZERO);
            restriction.apply(TransposeMode::NoTranspose, source, local_buffer)?;
            local_buffer.as_slice()
        } else {
            source
        };

        // Basis apply (GPU-accelerated if WgpuBasis with runtime)
        if let Some(basis) = &field.basis {
            let qcomp = Self::qpoint_component_count(field, eval_mode)?;
            q_buffer.resize(self.num_elem * basis.num_qpoints() * qcomp, T::ZERO);
            basis.apply(self.num_elem, false, eval_mode, local, q_buffer)?;
        } else {
            // No basis: pass local data directly to q-point buffer
            q_buffer.clear();
            q_buffer.extend_from_slice(local);
        }

        Ok(())
    }

    /// Basis^T apply + restriction scatter for one output field.
    ///
    /// GPU acceleration kicks in when the field's basis/restriction are WGPU-backed.
    fn scatter_output_to_slice(
        &self,
        field: &WgpuOperatorField<T>,
        eval_mode: EvalMode,
        q_output: &[T],
        local_buffer: &mut Vec<T>,
        active_output: &mut [T],
    ) -> ReedResult<()> {
        // Basis transpose (GPU-accelerated if WgpuBasis with runtime)
        let local = if let Some(basis) = &field.basis {
            local_buffer.resize(
                self.num_elem * basis.num_dof() * basis.num_comp(),
                T::ZERO,
            );
            basis.apply(self.num_elem, true, eval_mode, q_output, local_buffer)?;
            local_buffer.as_slice()
        } else {
            q_output
        };

        // Restriction scatter (GPU-accelerated if WgpuElemRestriction with runtime)
        match &field.vector {
            WgpuFieldVector::Active => {
                if let Some(restriction) = &field.restriction {
                    restriction.apply(TransposeMode::Transpose, local, active_output)
                } else {
                    // No restriction: direct accumulate into output
                    if active_output.len() != local.len() {
                        return Err(ReedError::Operator(format!(
                            "output length {} != local length {} for field '{}'",
                            active_output.len(),
                            local.len(),
                            field.name
                        )));
                    }
                    for (dst, src) in active_output.iter_mut().zip(local.iter()) {
                        *dst += *src;
                    }
                    Ok(())
                }
            }
            WgpuFieldVector::Passive(_) | WgpuFieldVector::None => Err(ReedError::Operator(
                format!("output field '{}' must be Active", field.name),
            )),
        }
    }

    // ------------------------------------------------------------------
    // Adjoint helpers (mirrors CpuOperator)
    // ------------------------------------------------------------------

    /// Pull the range cotangent (global output field) down to quadrature points.
    ///
    /// - restriction gather (NoTranspose) -> element-local
    /// - basis forward -> q-point buffer
    fn pull_range_cotangent_to_qp(
        &self,
        field: &WgpuOperatorField<T>,
        eval_mode: EvalMode,
        range_global: &[T],
        local_buffer: &mut Vec<T>,
        q_buffer: &mut Vec<T>,
    ) -> ReedResult<()> {
        if !matches!(field.vector, WgpuFieldVector::Active) {
            return Err(ReedError::Operator(format!(
                "operator adjoint: output field '{}' must be active",
                field.name
            )));
        }

        let local = if let Some(restriction) = &field.restriction {
            local_buffer.resize(restriction.local_size(), T::ZERO);
            restriction.apply(TransposeMode::NoTranspose, range_global, local_buffer)?;
            local_buffer.as_slice()
        } else {
            if let Some(basis) = &field.basis {
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

        if let Some(basis) = &field.basis {
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

    /// Scatter domain cotangent from quadrature points up to global (element-local -> global).
    ///
    /// - basis transpose -> element-local
    /// - restriction scatter (Transpose) -> global accumulation
    fn scatter_domain_cotangent_qp_to_global(
        &self,
        field: &WgpuOperatorField<T>,
        eval_mode: EvalMode,
        q_in_cot: &[T],
        local_buffer: &mut Vec<T>,
        domain_global: &mut [T],
    ) -> ReedResult<()> {
        match &field.vector {
            WgpuFieldVector::Active => {
                let local = if let Some(basis) = &field.basis {
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

                if let Some(restriction) = &field.restriction {
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
                    for (dst, src) in domain_global.iter_mut().zip(local.iter()) {
                        *dst += *src;
                    }
                    Ok(())
                }
            }
            WgpuFieldVector::Passive(_) | WgpuFieldVector::None => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// OperatorTrait impl
// ---------------------------------------------------------------------------

impl<T: Scalar> OperatorTrait<T> for WgpuOperator<T> {
    fn apply(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        if self.multi_distinct_active_io_fields() {
            return Err(ReedError::Operator(
                "multi-field active operator: use OperatorTrait::apply_field_buffers / apply_add_field_buffers (WgpuOperator; single-buffer apply is not supported)".into(),
            ));
        }
        self.apply_forward(input, output, false)
    }

    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        if self.multi_distinct_active_io_fields() {
            return Err(ReedError::Operator(
                "multi-field active operator: use OperatorTrait::apply_field_buffers / apply_add_field_buffers (WgpuOperator; single-buffer apply is not supported)".into(),
            ));
        }
        self.apply_forward(input, output, true)
    }

    fn operator_label(&self) -> Option<&str> {
        self.op_label.as_deref()
    }

    fn global_vector_len_hint(&self) -> Option<usize> {
        // Infer from the first active field with a restriction on the input or output side
        let input_len = self.fields.iter().find_map(|f| {
            if matches!(f.vector, WgpuFieldVector::Active) {
                f.restriction.as_ref().map(|r| r.num_global_dof())
            } else {
                None
            }
        });
        input_len
    }

    fn requires_field_named_buffers(&self) -> bool {
        self.multi_distinct_active_io_fields()
    }

    fn linear_assemble_diagonal(&self, _assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "WgpuOperator::linear_assemble_diagonal is not implemented on the GPU path".into(),
        ))
    }

    /// Forward delegates to [`Self::apply`]; Adjoint runs restriction-gather → basis-forward →
    /// QFunction^T → basis-transpose → restriction-scatter.
    fn apply_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply(input, output),
            OperatorTransposeRequest::Adjoint => self.apply_adjoint(input, output, false),
        }
    }

    /// Same as [`Self::apply_with_transpose`] but accumulates on both forward and adjoint paths.
    fn apply_add_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_add(input, output),
            OperatorTransposeRequest::Adjoint => self.apply_adjoint(input, output, true),
        }
    }

    fn apply_field_buffers<'io>(
        &self,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        self.apply_forward_field_buffers(inputs, outputs, false)
    }

    fn apply_add_field_buffers<'io>(
        &self,
        inputs: &'io [(&'io str, &'io dyn VectorTrait<T>)],
        outputs: &'io mut [(&'io str, &'io mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        self.apply_forward_field_buffers(inputs, outputs, true)
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
                self.apply_adjoint_field_buffers(inputs, outputs, false)
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
                self.apply_adjoint_field_buffers(inputs, outputs, true)
            }
        }
    }

    fn check_ready(&self) -> ReedResult<()> {
        if self.fields.is_empty() {
            return Err(ReedError::Operator(
                "WgpuOperator has no fields".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WgpuOperatorBuilder
// ---------------------------------------------------------------------------

/// Builder for [`WgpuOperator`], following the [`OperatorBuilder`](reed_cpu::operator::OperatorBuilder)
/// pattern but requiring owned WGPU backend objects.
///
/// # Example
///
/// ```ignore
/// use reed_wgpu::{WgpuOperatorBuilder, WgpuFieldVector};
///
/// let op = WgpuOperatorBuilder::new()
///     .runtime(runtime.clone())
///     .qfunction(qf)
///     .field("u", Some(Box::new(restr)), Some(Box::new(basis)), WgpuFieldVector::Active)
///     .field("v", Some(Box::new(restr_out)), Some(Box::new(basis_out)), WgpuFieldVector::Active)
///     .build()?;
/// ```
pub struct WgpuOperatorBuilder<T: Scalar> {
    runtime: Option<Arc<GpuRuntime>>,
    qfunction: Option<Box<dyn QFunctionTrait<T>>>,
    qfunction_context: Option<QFunctionContext>,
    op_label: Option<String>,
    fields: Vec<WgpuOperatorField<T>>,
    /// Optional gallery name for device QFunction auto-detection.
    /// When set, the builder tries to resolve a GPU-resident QFunction from
    /// [`GpuRuntime`] and uses it instead of the user-provided one.
    device_qfunction_name: Option<String>,
}

impl<T: Scalar> Default for WgpuOperatorBuilder<T> {
    fn default() -> Self {
        Self {
            runtime: None,
            qfunction: None,
            qfunction_context: None,
            op_label: None,
            fields: Vec::new(),
            device_qfunction_name: None,
        }
    }
}

impl<T: Scalar> WgpuOperatorBuilder<T> {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the shared GPU runtime (required).
    pub fn runtime(mut self, runtime: Arc<GpuRuntime>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Set the QFunction that will be applied at quadrature points.
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

    /// Human-readable operator name for logging.
    pub fn operator_label(mut self, label: impl Into<String>) -> Self {
        self.op_label = Some(label.into());
        self
    }

    /// Optional gallery name for device QFunction auto-detection.
    ///
    /// When set, the builder attempts to create a GPU-resident QFunction via
    /// [`crate::qfunction_device::try_create_device_q_function_f32`] and uses it in place
    /// of the user-provided boxed QFunction. If the device QFunction cannot be created,
    /// the build fails with an appropriate error.
    ///
    /// Supported names include `"Identity"`, `"MassApply"`, `"Poisson1DApply"`, etc.
    pub fn device_qfunction_name(mut self, name: impl Into<String>) -> Self {
        self.device_qfunction_name = Some(name.into());
        self
    }

    /// Add a named field with optional restriction, basis, and vector role.
    ///
    /// - `restriction`: [`WgpuElemRestriction`](crate::elem_restriction::WgpuElemRestriction) for
    ///   global-element gather/scatter (GPU-accelerated with runtime).
    /// - `basis`: [`WgpuBasis`](crate::basis::WgpuBasis) for element-qpoint mapping
    ///   (GPU-accelerated with runtime).
    /// - `vector`: [`WgpuFieldVector::Active`] for apply-time supplied vectors,
    ///   [`WgpuFieldVector::Passive`] for pre-set data (e.g. `qdata`).
    pub fn field(
        mut self,
        name: impl Into<String>,
        restriction: Option<Box<dyn ElemRestrictionTrait<T>>>,
        basis: Option<Box<dyn BasisTrait<T>>>,
        vector: WgpuFieldVector<T>,
    ) -> Self {
        self.fields.push(WgpuOperatorField {
            name: name.into(),
            restriction,
            basis,
            vector,
        });
        self
    }

    /// Consume the builder and produce a [`WgpuOperator`].
    ///
    /// Validates:
    /// - A GPU runtime is set
    /// - A QFunction is set
    /// - QFunctionContext length matches the qfunction requirement
    /// - All qfunction input/output field names exist in the field list
    /// - At least one field has a restriction (for `num_elem`)
    /// - At least one field has a basis or restriction (for `num_qpoints`)
    ///
    /// If [`Self::device_qfunction_name`] was set, the builder attempts to resolve a device
    /// QFunction via [`GpuRuntime`] and uses it as the operator's QFunction, replacing the
    /// user-provided boxed one.
    pub fn build(self) -> ReedResult<WgpuOperator<T>> {
        let runtime = self
            .runtime
            .clone()
            .ok_or_else(|| ReedError::Operator("WgpuOperatorBuilder requires a GpuRuntime".into()))?;

        let user_qfunction = self
            .qfunction
            .ok_or_else(|| ReedError::Operator("WgpuOperatorBuilder requires a qfunction".into()))?;

        // Auto-detect device QFunction if a gallery name was provided
        let qfunction: Box<dyn QFunctionTrait<T>> = if let Some(ref name) = self.device_qfunction_name
        {
            // Try to create a device QFunction via the runtime's gallery lookup.
            match crate::qfunction_device::try_create_device_q_function_f32(
                name,
                runtime.clone(),
            ) {
                Some(Ok(device_qf)) => {
                    // SAFETY: device QFunctions are f32-only; T must be f32 for this path.
                    debug_assert_eq!(
                        std::any::TypeId::of::<T>(),
                        std::any::TypeId::of::<f32>()
                    );
                    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
                        return Err(ReedError::Operator(format!(
                            "device QFunction '{}' is f32-only; operator scalar type must be f32",
                            name
                        )));
                    }
                    unsafe { crate::coerce_qfunction_f32_box(device_qf) }
                }
                Some(Err(e)) => {
                    return Err(ReedError::Operator(format!(
                        "failed to create device QFunction '{}': {}",
                        name, e
                    )));
                }
                None => {
                    return Err(ReedError::Operator(format!(
                        "no device QFunction available for gallery name '{}'",
                        name
                    )));
                }
            }
        } else {
            user_qfunction
        };

        // Validate QFunctionContext length
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

        // Build input/output plans from qfunction descriptors
        let input_plans = qfunction
            .inputs()
            .iter()
            .map(|descriptor| {
                Ok(InputPlan {
                    field_index: WgpuOperator::<T>::field_index_by_name(
                        &self.fields,
                        &descriptor.name,
                    )?,
                    eval_mode: descriptor.eval_mode,
                })
            })
            .collect::<ReedResult<Vec<_>>>()?;

        let output_plans = qfunction
            .outputs()
            .iter()
            .map(|descriptor| {
                Ok(OutputPlan {
                    field_index: WgpuOperator::<T>::field_index_by_name(
                        &self.fields,
                        &descriptor.name,
                    )?,
                    eval_mode: descriptor.eval_mode,
                })
            })
            .collect::<ReedResult<Vec<_>>>()?;

        let num_qfunction_inputs = input_plans.len();
        let num_qfunction_outputs = output_plans.len();

        // Infer num_elem from first field with a restriction
        let num_elem = self
            .fields
            .iter()
            .find_map(|f| f.restriction.as_ref().map(|r| r.num_elements()))
            .ok_or_else(|| {
                ReedError::Operator(
                    "WgpuOperatorBuilder requires at least one field with a restriction".into(),
                )
            })?;

        // Infer num_qpoints from first field with a basis, fallback to restriction elemsize
        let num_qpoints = self
            .fields
            .iter()
            .find_map(|f| f.basis.as_ref().map(|b| b.num_qpoints()))
            .or_else(|| {
                self.fields.iter().find_map(|f| {
                    f.restriction
                        .as_ref()
                        .map(|r| r.num_dof_per_elem())
                })
            })
            .ok_or_else(|| {
                ReedError::Operator(
                    "WgpuOperatorBuilder requires at least one field with a basis or restriction"
                        .into(),
                )
            })?;

        Ok(WgpuOperator {
            runtime,
            num_elem,
            num_qpoints,
            fields: self.fields,
            qfunction,
            qfunction_context,
            input_plans,
            output_plans,
            num_qfunction_inputs,
            num_qfunction_outputs,
            op_label: self.op_label,
            last_forward_q_inputs: Mutex::new(None),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use reed_core::QuadMode;
    use reed_cpu::gallery::{Identity, MassApply};
    use crate::elem_restriction::WgpuElemRestriction;
    use crate::basis::WgpuBasis;
    use reed_cpu::vector::CpuVector;

    fn gpu_runtime_or_skip() -> Option<Arc<GpuRuntime>> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))?;
        GpuRuntime::new(&adapter).map(Arc::new)
    }

    /// Smoke test: build a minimal WgpuOperator and verify it doesn't panic.
    #[test]
    fn build_minimal_operator() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };

        let nelem = 2usize;
        let p = 2usize;
        let q = 3usize;
        let num_dof = p; // 1D Lagrange: num_dof = p^dim = p^1 = p
        let ncomp = 1;

        // Create restriction and basis (WGPU-backed)
        let offsets: Vec<i32> = (0..nelem * num_dof).map(|i| i as i32).collect();
        let restr = WgpuElemRestriction::<f32>::new_offset(
            nelem,
            num_dof,
            ncomp,
            1,
            nelem * num_dof,
            &offsets,
            Some(rt.clone()),
        )
        .unwrap();

        let basis = WgpuBasis::<f32>::new(
            1, // dim
            ncomp,
            p,
            q,
            QuadMode::Gauss,
            Some(rt.clone()),
        )
        .unwrap();

        // Identity QFunction: input -> output, 1 component each
        let qf = Identity::with_components(1);

        let restr2 = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, nelem * num_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis2 = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();

        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(qf))
            .field(
                "input",
                Some(Box::new(restr)),
                Some(Box::new(basis)),
                WgpuFieldVector::Active,
            )
            .field(
                "output",
                Some(Box::new(restr2)),
                Some(Box::new(basis2)),
                WgpuFieldVector::Active,
            )
            .build();

        assert!(op.is_ok(), "build failed: {:?}", op.err());
    }

    /// Integration test: WgpuOperator apply produces same result as CpuOperator apply
    /// for a simple identity operator.
    #[test]
    fn identity_operator_matches_cpu() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };

        let nelem = 2usize;
        let p = 2usize;
        let q = 3usize;
        let num_dof = p; // 1D Lagrange: num_dof = p^dim = p^1 = p
        let ncomp = 1;
        let global_dof = nelem * num_dof;

        // Build WGPU restriction and basis
        let offsets: Vec<i32> = (0..nelem * num_dof).map(|i| i as i32).collect();
        let restr_wgpu = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        )
        .unwrap();
        let basis_wgpu = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();
        let restr_wgpu2 = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        )
        .unwrap();
        let basis_wgpu2 = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();

        // Build CPU restriction and basis
        let restr_cpu = reed_cpu::elem_restriction::CpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets,
        )
        .unwrap();
        let basis_cpu = reed_cpu::basis_lagrange::LagrangeBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss)
            .unwrap();

        // Identity QFunction: expects fields named "input" and "output"
        let qf_wgpu = Identity::with_components(1);
        let qf_cpu = Identity::with_components(1);

        // Build operators with correct field names
        let op_wgpu = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(qf_wgpu))
            .field("input", Some(Box::new(restr_wgpu)), Some(Box::new(basis_wgpu)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr_wgpu2)), Some(Box::new(basis_wgpu2)), WgpuFieldVector::Active)
            .operator_label("identity-wgpu")
            .build()
            .unwrap();

        let op_cpu = reed_cpu::operator::OperatorBuilder::new()
            .qfunction(Box::new(qf_cpu))
            .field("input", Some(&restr_cpu), Some(&basis_cpu), reed_cpu::operator::FieldVector::Active)
            .field("output", Some(&restr_cpu), Some(&basis_cpu), reed_cpu::operator::FieldVector::Active)
            .operator_label("identity-cpu")
            .build()
            .unwrap();

        // Apply both operators with the same input
        let input = CpuVector::from_vec((0..global_dof).map(|i| 0.1 * (i + 1) as f32).collect());
        let mut output_wgpu = CpuVector::new(global_dof);
        let mut output_cpu = CpuVector::new(global_dof);

        op_wgpu.apply(&input, &mut output_wgpu).unwrap();
        op_cpu.apply(&input, &mut output_cpu).unwrap();

        // Compare results
        let wgpu_data = output_wgpu.as_slice();
        let cpu_data = output_cpu.as_slice();
        for i in 0..global_dof {
            let diff = (wgpu_data[i] - cpu_data[i]).abs();
            assert!(
                diff < 1e-4,
                "mismatch at index {}: wgpu={} cpu={} diff={}",
                i, wgpu_data[i], cpu_data[i], diff
            );
        }
    }

    /// Test apply_add: accumulate into a non-zero output.
    #[test]
    fn apply_add_accumulates() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };

        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p; // 1D Lagrange: num_dof = p^dim = p^1 = p
        let ncomp = 1;
        let global_dof = nelem * num_dof;

        let offsets: Vec<i32> = (0..nelem * num_dof).map(|i| i as i32).collect();
        let restr = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        )
        .unwrap();
        let restr2 = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        )
        .unwrap();
        let basis =
            WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let basis2 =
            WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let qf = Identity::with_components(1);

        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(qf))
            .field("input", Some(Box::new(restr)), Some(Box::new(basis)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr2)), Some(Box::new(basis2)), WgpuFieldVector::Active)
            .build()
            .unwrap();

        let input = CpuVector::from_vec((0..global_dof).map(|i| (i + 1) as f32).collect());

        // First apply: output gets the identity result
        let mut output = CpuVector::new(global_dof);
        op.apply(&input, &mut output).unwrap();
        let first_result: Vec<f32> = output.as_slice().to_vec();

        // apply_add: should double the result
        op.apply_add(&input, &mut output).unwrap();

        for i in 0..global_dof {
            let expected = 2.0 * first_result[i];
            let got = output.as_slice()[i];
            assert!(
                (got - expected).abs() < 1e-4,
                "apply_add mismatch at {}: got {} expected {}",
                i, got, expected
            );
        }
    }

    /// Test with a mass operator (passive qdata field).
    #[test]
    fn mass_operator_with_passive_qdata() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };

        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p; // 1D Lagrange: num_dof = p^dim = p^1 = p
        let ncomp = 1;
        let global_dof = nelem * num_dof;

        let offsets: Vec<i32> = (0..nelem * num_dof).map(|i| i as i32).collect();
        let restr_u = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        )
        .unwrap();
        let restr_v = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        )
        .unwrap();
        let basis_u =
            WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let basis_v =
            WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();

        // MassApply: inputs=["u", "qdata"], outputs=["v"]
        let qdata: Vec<f32> = (0..nelem * q).map(|i| 0.5 + 0.1 * (i as f32)).collect();
        let qdata_vec = CpuVector::from_vec(qdata);
        let qf = MassApply::default();

        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(qf))
            .field("u", Some(Box::new(restr_u)), Some(Box::new(basis_u)), WgpuFieldVector::Active)
            .field("qdata", None, None, WgpuFieldVector::Passive(Box::new(qdata_vec)))
            .field("v", Some(Box::new(restr_v)), Some(Box::new(basis_v)), WgpuFieldVector::Active)
            .build()
            .unwrap();

        let input = CpuVector::from_vec((0..global_dof).map(|i| (i + 1) as f32).collect());
        let mut output = CpuVector::new(global_dof);

        // Should not panic
        op.apply(&input, &mut output).unwrap();

        // Output should be non-zero
        let out_slice = output.as_slice();
        let has_nonzero = out_slice.iter().any(|&v| v.abs() > 1e-8);
        assert!(has_nonzero, "mass operator output is all zeros");
    }

    // -----------------------------------------------------------------
    // Adjoint tests
    // -----------------------------------------------------------------

    /// Adjoint inner product identity:
    ///   <A u, dv> == <u, A^* dv>
    /// using a simple identity QFunction ("input" -> "output").
    #[test]
    fn adjoint_identity_inner_product() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };

        let nelem = 3usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;

        let offsets: Vec<i32> = (0..global_dof).map(|i| i as i32).collect();
        let restr = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr2 = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();
        let basis2 = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();

        let qf = Identity::with_components(1);

        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(qf))
            .field("input", Some(Box::new(restr)), Some(Box::new(basis)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr2)), Some(Box::new(basis2)), WgpuFieldVector::Active)
            .operator_label("adjoint-identity")
            .build()
            .unwrap();

        // Forward: y = A x
        let x = CpuVector::from_vec((0..global_dof).map(|i| 0.1 * (i + 1) as f32).collect());
        let mut y = CpuVector::new(global_dof);
        op.apply(&x, &mut y).unwrap();

        // Adjoint: dx = A^* dy
        let dy = CpuVector::from_vec((0..global_dof).map(|i| 0.05 * ((global_dof - i) as f32)).collect());
        let mut dx = CpuVector::new(global_dof);
        op.apply_with_transpose(
            OperatorTransposeRequest::Adjoint,
            &dy,
            &mut dx,
        ).unwrap();

        // Inner product identity: dot(y, dy) == dot(x, dx)
        let dot_fwd = y.as_slice().iter().zip(dy.as_slice()).map(|(a, b)| a * b).sum::<f32>();
        let dot_adj = x.as_slice().iter().zip(dx.as_slice()).map(|(a, b)| a * b).sum::<f32>();
        assert!(
            (dot_fwd - dot_adj).abs() < 1e-4,
            "adjoint inner product identity failed: dot_fwd={} dot_adj={} diff={}",
            dot_fwd, dot_adj, (dot_fwd - dot_adj).abs()
        );
    }

    /// adjoint matches CPU operator adjoint: same input/output sizes, same QFunction.
    #[test]
    fn adjoint_matches_cpu() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };

        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;

        // WGPU components
        let offsets: Vec<i32> = (0..global_dof).map(|i| i as i32).collect();
        let restr_wgpu = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr_wgpu2 = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis_wgpu = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();
        let basis_wgpu2 = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();

        // CPU components
        let restr_cpu = reed_cpu::elem_restriction::CpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets,
        ).unwrap();
        let basis_cpu = reed_cpu::basis_lagrange::LagrangeBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss)
            .unwrap();

        let qf_wgpu = Identity::with_components(1);
        let qf_cpu = Identity::with_components(1);

        let op_wgpu = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(qf_wgpu))
            .field("input", Some(Box::new(restr_wgpu)), Some(Box::new(basis_wgpu)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr_wgpu2)), Some(Box::new(basis_wgpu2)), WgpuFieldVector::Active)
            .build()
            .unwrap();

        let op_cpu = reed_cpu::operator::OperatorBuilder::new()
            .qfunction(Box::new(qf_cpu))
            .field("input", Some(&restr_cpu), Some(&basis_cpu), reed_cpu::operator::FieldVector::Active)
            .field("output", Some(&restr_cpu), Some(&basis_cpu), reed_cpu::operator::FieldVector::Active)
            .build()
            .unwrap();

        // Forward on both (warm-up: caches forward q-inputs)
        let x = CpuVector::from_vec((0..global_dof).map(|i| 0.1 * (i + 1) as f32).collect());
        let mut y_wgpu = CpuVector::new(global_dof);
        let mut y_cpu = CpuVector::new(global_dof);
        op_wgpu.apply(&x, &mut y_wgpu).unwrap();
        op_cpu.apply(&x, &mut y_cpu).unwrap();

        // Adjoint on both: compare results
        let dy = CpuVector::from_vec((0..global_dof).map(|i| 0.05 * (i + 1) as f32).collect());
        let mut dx_wgpu = CpuVector::new(global_dof);
        let mut dx_cpu = CpuVector::new(global_dof);
        op_wgpu.apply_with_transpose(
            OperatorTransposeRequest::Adjoint,
            &dy,
            &mut dx_wgpu,
        ).unwrap();
        op_cpu.apply_with_transpose(
            OperatorTransposeRequest::Adjoint,
            &dy,
            &mut dx_cpu,
        ).unwrap();

        for i in 0..global_dof {
            let diff = (dx_wgpu.as_slice()[i] - dx_cpu.as_slice()[i]).abs();
            assert!(
                diff < 1e-4,
                "adjoint mismatch at index {}: wgpu={} cpu={} diff={}",
                i, dx_wgpu.as_slice()[i], dx_cpu.as_slice()[i], diff
            );
        }
    }

    /// adjoint with `add`: apply_add_with_transpose(Adjoint) accumulates into domain cotangent.
    #[test]
    fn adjoint_add_accumulates() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };

        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;

        let offsets: Vec<i32> = (0..global_dof).map(|i| i as i32).collect();
        let restr = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr2 = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();
        let basis2 = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();

        let qf = Identity::with_components(1);
        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(qf))
            .field("input", Some(Box::new(restr)), Some(Box::new(basis)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr2)), Some(Box::new(basis2)), WgpuFieldVector::Active)
            .build()
            .unwrap();

        // Forward (needed to cache q_inputs)
        let x = CpuVector::from_vec((0..global_dof).map(|i| (i + 1) as f32).collect());
        let mut y = CpuVector::new(global_dof);
        op.apply(&x, &mut y).unwrap();

        // First adjoint
        let dy = CpuVector::from_vec((0..global_dof).map(|i| (i + 1) as f32).collect());
        let mut dx = CpuVector::new(global_dof);
        op.apply_with_transpose(
            OperatorTransposeRequest::Adjoint,
            &dy,
            &mut dx,
        ).unwrap();
        let first_result: Vec<f32> = dx.as_slice().to_vec();

        // Second adjoint with add: should double
        op.apply_add_with_transpose(
            OperatorTransposeRequest::Adjoint,
            &dy,
            &mut dx,
        ).unwrap();

        for i in 0..global_dof {
            let expected = 2.0 * first_result[i];
            let got = dx.as_slice()[i];
            assert!(
                (got - expected).abs() < 1e-4,
                "adjoint add accumulate mismatch at {}: got {} expected {}",
                i, got, expected
            );
        }
    }

    // -----------------------------------------------------------------
    // Multi-field tests
    // -----------------------------------------------------------------

    /// Multi-field forward: apply_field_buffers with named buffers.
    #[test]
    fn multi_field_forward_matches_single_buffer() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };

        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;

        let offsets: Vec<i32> = (0..global_dof).map(|i| i as i32).collect();
        let restr = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr2 = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();
        let basis2 = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();

        let qf = Identity::with_components(1);
        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(qf))
            .field("input", Some(Box::new(restr)), Some(Box::new(basis)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr2)), Some(Box::new(basis2)), WgpuFieldVector::Active)
            .build()
            .unwrap();

        // Single-buffer apply
        let x = CpuVector::from_vec((0..global_dof).map(|i| 0.1 * (i + 1) as f32).collect());
        let mut y_single = CpuVector::new(global_dof);
        op.apply(&x, &mut y_single).unwrap();

        // Multi-field apply with same buffers
        let mut y_multi: CpuVector<f32> = CpuVector::new(global_dof);
        let ins = [("input", &x as &dyn VectorTrait<f32>)];
        let mut outs = [("output", &mut y_multi as &mut dyn VectorTrait<f32>)];
        op.apply_field_buffers(&ins, &mut outs).unwrap();

        // Results should be identical
        let single_data = y_single.as_slice();
        let multi_data = y_multi.as_slice();
        for i in 0..global_dof {
            let diff = (single_data[i] - multi_data[i]).abs();
            assert!(
                diff < 1e-6,
                "multi-field mismatch at index {}: single={} multi={}",
                i, single_data[i], multi_data[i]
            );
        }
    }

    /// Multi-field adjoint: apply_field_buffers_with_transpose(Adjoint) inner product identity.
    #[test]
    fn multi_field_adjoint_inner_product() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };

        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;

        let offsets: Vec<i32> = (0..global_dof).map(|i| i as i32).collect();
        let restr = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr2 = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();
        let basis2 = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();

        let qf = Identity::with_components(1);
        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(qf))
            .field("input", Some(Box::new(restr)), Some(Box::new(basis)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr2)), Some(Box::new(basis2)), WgpuFieldVector::Active)
            .build()
            .unwrap();

        // Forward via field_buffers
        let x = CpuVector::from_vec((0..global_dof).map(|i| 0.1 * (i + 1) as f32).collect());
        let mut y = CpuVector::new(global_dof);
        let ins = [("input", &x as &dyn VectorTrait<f32>)];
        let mut outs = [("output", &mut y as &mut dyn VectorTrait<f32>)];
        op.apply_field_buffers(&ins, &mut outs).unwrap();

        // Adjoint via field_buffers_with_transpose
        let dy = CpuVector::from_vec((0..global_dof).map(|i| 0.05 * (i + 1) as f32).collect());
        let mut dx = CpuVector::new(global_dof);
        let range_in = [("output", &dy as &dyn VectorTrait<f32>)];
        let mut domain_out = [("input", &mut dx as &mut dyn VectorTrait<f32>)];
        op.apply_field_buffers_with_transpose(
            OperatorTransposeRequest::Adjoint,
            &range_in,
            &mut domain_out,
        ).unwrap();

        // Inner product identity
        let dot_fwd = y.as_slice().iter().zip(dy.as_slice()).map(|(a, b)| a * b).sum::<f32>();
        let dot_adj = x.as_slice().iter().zip(dx.as_slice()).map(|(a, b)| a * b).sum::<f32>();
        assert!(
            (dot_fwd - dot_adj).abs() < 1e-4,
            "multi-field adjoint inner product identity failed: dot_fwd={} dot_adj={}",
            dot_fwd, dot_adj
        );
    }

    // -----------------------------------------------------------------
    // Device QFunction tests
    // -----------------------------------------------------------------

    /// Build an operator with a device QFunction via `device_qfunction_name` and verify
    /// forward apply produces the same result as the CPU path.
    #[test]
    fn device_qfunction_identity_matches_cpu() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };

        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;

        let offsets: Vec<i32> = (0..global_dof).map(|i| i as i32).collect();
        let restr_wgpu = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr_wgpu2 = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis_wgpu = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();
        let basis_wgpu2 = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();

        // CPU operator using CPU Identity
        let restr_cpu = reed_cpu::elem_restriction::CpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets,
        ).unwrap();
        let basis_cpu = reed_cpu::basis_lagrange::LagrangeBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss)
            .unwrap();
        let qf_cpu = Identity::with_components(1);
        let op_cpu = reed_cpu::operator::OperatorBuilder::new()
            .qfunction(Box::new(qf_cpu))
            .field("input", Some(&restr_cpu), Some(&basis_cpu), reed_cpu::operator::FieldVector::Active)
            .field("output", Some(&restr_cpu), Some(&basis_cpu), reed_cpu::operator::FieldVector::Active)
            .build()
            .unwrap();

        // WGPU operator with device QFunction auto-detection.
        // Pass a dummy QFunction (will be replaced by device version).
        let dummy_qf = Identity::with_components(1);
        let op_wgpu = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(dummy_qf))
            .device_qfunction_name("Identity")
            .field("input", Some(Box::new(restr_wgpu)), Some(Box::new(basis_wgpu)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr_wgpu2)), Some(Box::new(basis_wgpu2)), WgpuFieldVector::Active)
            .build()
            .unwrap();

        // Forward on both
        let x = CpuVector::from_vec((0..global_dof).map(|i| 0.1 * (i + 1) as f32).collect());
        let mut y_wgpu = CpuVector::new(global_dof);
        let mut y_cpu = CpuVector::new(global_dof);
        op_wgpu.apply(&x, &mut y_wgpu).unwrap();
        op_cpu.apply(&x, &mut y_cpu).unwrap();

        for i in 0..global_dof {
            let diff = (y_wgpu.as_slice()[i] - y_cpu.as_slice()[i]).abs();
            assert!(
                diff < 1e-4,
                "device QF mismatch at index {}: wgpu={} cpu={}",
                i, y_wgpu.as_slice()[i], y_cpu.as_slice()[i]
            );
        }
    }

    /// Device QFunction with adjoint: inner product identity.
    #[test]
    fn device_qfunction_adjoint_inner_product() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };

        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;

        let offsets: Vec<i32> = (0..global_dof).map(|i| i as i32).collect();
        let restr = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr2 = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();
        let basis2 = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone()))
            .unwrap();

        let dummy_qf = Identity::with_components(1);
        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(dummy_qf))
            .device_qfunction_name("Identity")
            .field("input", Some(Box::new(restr)), Some(Box::new(basis)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr2)), Some(Box::new(basis2)), WgpuFieldVector::Active)
            .build()
            .unwrap();

        // Forward
        let x = CpuVector::from_vec((0..global_dof).map(|i| 0.1 * (i + 1) as f32).collect());
        let mut y = CpuVector::new(global_dof);
        op.apply(&x, &mut y).unwrap();

        // Adjoint
        let dy = CpuVector::from_vec((0..global_dof).map(|i| 0.05 * (i + 1) as f32).collect());
        let mut dx = CpuVector::new(global_dof);
        op.apply_with_transpose(
            OperatorTransposeRequest::Adjoint,
            &dy,
            &mut dx,
        ).unwrap();

        // Inner product identity
        let dot_fwd = y.as_slice().iter().zip(dy.as_slice()).map(|(a, b)| a * b).sum::<f32>();
        let dot_adj = x.as_slice().iter().zip(dx.as_slice()).map(|(a, b)| a * b).sum::<f32>();
        assert!(
            (dot_fwd - dot_adj).abs() < 1e-4,
            "device QF adjoint identity failed: dot_fwd={} dot_adj={}",
            dot_fwd, dot_adj
        );
    }
}
