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
//!   3. QFunction dispatch at q-points (CPU for v1, device QFunction when available)
//!   4. For each output field: basis^T apply (GPU) -> element-local buffer
//!   5. For each output field: restriction scatter (GPU) -> accumulate to y_global
//! ```
//!
//! ## Adjoint
//!
//! ```text
//! WgpuOperator::apply_with_transpose(Adjoint, range_cot, domain_cot):
//!   1. Evaluate passive input fields forward (to get primal q-point inputs for transpose kernel)
//!   2. For each output field: restriction gather (GPU) -> basis forward (GPU) -> q_out_cot
//!   3. QFunction.apply_operator_transpose_with_primal at q-points -> q_in_cot
//!   4. For each active input field: basis^T (GPU) -> restriction scatter (GPU) -> domain_cot
//! ```
//!
//! ## Multi-field
//!
//! `apply_field_buffers` / `apply_add_field_buffers` dispatch per-field named buffers
//! following the same restriction/basis/QFunction pipeline.
//!
//! ## Device QFunction auto-integration
//!
//! When a gallery QFunction reports a [`gallery_name`](reed_core::QFunctionTrait::gallery_name),
//! the builder tries to fetch a device counterpart via [`GpuRuntime`].
//! If available (f32 only), the device QFunction replaces the CPU QFunction for the pointwise
//! evaluation step.

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
/// [`GpuRuntime`] WGSL pipelines when the inner restriction / basis have a runtime available.
///
/// Supports adjoint, multi-field named-buffer apply, and device QFunction auto-integration.
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
    /// Snapshot of forward q-point inputs for adjoint kernels that need primal data.
    last_forward_q_inputs: Mutex<Option<Vec<Vec<T>>>>,
    /// Optional device-side QFunction (f32 WGSL dispatch); replaces the host QFunction
    /// for the pointwise evaluation step when set.
    device_qfunction: Option<Box<dyn QFunctionTrait<T>>>,
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
    // Multi-field helper methods
    // ------------------------------------------------------------------

    /// Distinct field indices that appear as [`WgpuFieldVector::Active`] on qfunction **inputs**.
    pub fn distinct_active_input_field_indices(&self) -> Vec<usize> {
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
    pub fn distinct_active_output_field_indices(&self) -> Vec<usize> {
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

    fn multi_distinct_active_io_fields(&self) -> bool {
        self.distinct_active_input_field_indices().len() > 1
            || self.distinct_active_output_field_indices().len() > 1
    }

    /// True when this operator cannot use single-buffer [`OperatorTrait::apply`] and requires
    /// [`OperatorTrait::apply_field_buffers`].
    pub fn requires_field_named_buffers(&self) -> bool {
        self.multi_distinct_active_io_fields()
    }

    fn merge_active_global_for_field_indices(
        fields: &[WgpuOperatorField<T>],
        field_indices: impl Iterator<Item = usize>,
    ) -> ReedResult<Option<usize>> {
        let mut sizes = HashSet::new();
        for idx in field_indices {
            let field = &fields[idx];
            if matches!(field.vector, WgpuFieldVector::Active) {
                if let Some(r) = &field.restriction {
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

    /// Merge active + restriction global sizes for qfunction **input** fields only.
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
        range: &[(&str, &dyn VectorTrait<T>)],
        domain: &[(&str, &mut dyn VectorTrait<T>)],
    ) -> ReedResult<()> {
        let in_keys: Vec<&str> = range.iter().map(|(k, _)| *k).collect();
        Self::assert_unique_field_keys(&in_keys, "adjoint range cotangent inputs")?;
        let out_keys: Vec<&str> = domain.iter().map(|(k, _)| *k).collect();
        Self::assert_unique_field_keys(&out_keys, "adjoint domain cotangent outputs")?;

        for &idx in &self.distinct_active_output_field_indices() {
            let f = &self.fields[idx];
            let v = Self::lookup_named_read(range, f.name.as_str())?;
            if let Some(r) = &f.restriction {
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
            let (_, v) = domain
                .iter()
                .find(|(name, _)| *name == f.name.as_str())
                .ok_or_else(|| {
                    ReedError::Operator(format!(
                        "adjoint apply_field_buffers: missing domain cotangent buffer for active input field {:?}",
                        f.name
                    ))
                })?;
            if let Some(r) = &f.restriction {
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

    // ------------------------------------------------------------------
    // Forward apply pipeline
    // ------------------------------------------------------------------

    /// Resolve the QFunction to use for the pointwise evaluation step.
    fn active_qfunction(&self) -> &dyn QFunctionTrait<T> {
        self.device_qfunction
            .as_deref()
            .unwrap_or_else(|| self.qfunction.as_ref())
    }

    /// Core forward apply: restriction gather -> basis apply -> QFunction -> basis^T -> restriction scatter.
    fn apply_forward(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
        add: bool,
    ) -> ReedResult<()> {
        if self.multi_distinct_active_io_fields() {
            return Err(ReedError::Operator(
                "multi-field active operator: use OperatorTrait::apply_field_buffers / apply_add_field_buffers (WgpuOperator; single-buffer apply is not supported)".into(),
            ));
        }
        if !add {
            output.set_value(T::ZERO)?;
        }

        let (_q_inputs, q_outputs) = self.run_forward_qfunction_stages(input)?;

        let out_sl = output.as_mut_slice();
        let mut output_locals: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();
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

    /// Restriction + basis -> q-point input buffers, then QFunction -> q-point output buffers.
    fn run_forward_qfunction_stages(
        &self,
        active_input: &dyn VectorTrait<T>,
    ) -> ReedResult<(Vec<Vec<T>>, Vec<Vec<T>>)> {
        let mut q_inputs: Vec<Vec<T>> = (0..self.num_qfunction_inputs)
            .map(|_| Vec::new())
            .collect();
        let mut q_outputs: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();
        let mut input_locals: Vec<Vec<T>> = (0..self.num_qfunction_inputs)
            .map(|_| Vec::new())
            .collect();

        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            self.prepare_input_into_single(
                field,
                plan.eval_mode,
                active_input,
                &mut input_locals[slot],
                &mut q_inputs[slot],
            )?;
        }

        if let Ok(mut cache) = self.last_forward_q_inputs.lock() {
            *cache = Some(q_inputs.clone());
        }

        let qf = self.active_qfunction();
        for (slot, descriptor) in qf.outputs().iter().enumerate() {
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
        qf.apply(
            self.qfunction_context.as_bytes(),
            self.num_elem * self.num_qpoints,
            &input_slices,
            &mut output_slices,
        )?;

        Ok((q_inputs, q_outputs))
    }

    /// Same as `run_forward_qfunction_stages` but with named input vectors.
    fn run_forward_qfunction_stages_named(
        &self,
        active_inputs: &[(&str, &dyn VectorTrait<T>)],
    ) -> ReedResult<(Vec<Vec<T>>, Vec<Vec<T>>)> {
        let mut q_inputs: Vec<Vec<T>> = (0..self.num_qfunction_inputs)
            .map(|_| Vec::new())
            .collect();
        let mut q_outputs: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();
        let mut input_locals: Vec<Vec<T>> = (0..self.num_qfunction_inputs)
            .map(|_| Vec::new())
            .collect();

        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            self.prepare_input_into_named(
                field,
                plan.eval_mode,
                active_inputs,
                &mut input_locals[slot],
                &mut q_inputs[slot],
            )?;
        }

        if let Ok(mut cache) = self.last_forward_q_inputs.lock() {
            *cache = Some(q_inputs.clone());
        }

        let qf = self.active_qfunction();
        for (slot, descriptor) in qf.outputs().iter().enumerate() {
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
        qf.apply(
            self.qfunction_context.as_bytes(),
            self.num_elem * self.num_qpoints,
            &input_slices,
            &mut output_slices,
        )?;

        Ok((q_inputs, q_outputs))
    }

    /// Restriction gather + basis apply for one input field (single-buffer active input).
    fn prepare_input_into_single(
        &self,
        field: &WgpuOperatorField<T>,
        eval_mode: EvalMode,
        active_input: &dyn VectorTrait<T>,
        local_buffer: &mut Vec<T>,
        q_buffer: &mut Vec<T>,
    ) -> ReedResult<()> {
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

        let source: &[T] = match &field.vector {
            WgpuFieldVector::Active => active_input.as_slice(),
            WgpuFieldVector::Passive(v) => v.as_slice(),
            WgpuFieldVector::None => {
                return Err(ReedError::Operator(format!(
                    "field '{}' has no vector source (set Active or Passive)",
                    field.name
                )));
            }
        };

        self.gather_and_basis_apply(field, eval_mode, source, local_buffer, q_buffer)
    }

    /// Restriction gather + basis apply for one input field (named-buffer active input).
    fn prepare_input_into_named(
        &self,
        field: &WgpuOperatorField<T>,
        eval_mode: EvalMode,
        active_inputs: &[(&str, &dyn VectorTrait<T>)],
        local_buffer: &mut Vec<T>,
        q_buffer: &mut Vec<T>,
    ) -> ReedResult<()> {
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

        let source: &[T] = match &field.vector {
            WgpuFieldVector::Active => {
                Self::lookup_named_read(active_inputs, field.name.as_str())?.as_slice()
            }
            WgpuFieldVector::Passive(v) => v.as_slice(),
            WgpuFieldVector::None => {
                return Err(ReedError::Operator(format!(
                    "field '{}' has no vector source (set Active or Passive)",
                    field.name
                )));
            }
        };

        self.gather_and_basis_apply(field, eval_mode, source, local_buffer, q_buffer)
    }

    /// Common restriction gather -> basis apply for a single field.
    fn gather_and_basis_apply(
        &self,
        field: &WgpuOperatorField<T>,
        eval_mode: EvalMode,
        source: &[T],
        local_buffer: &mut Vec<T>,
        q_buffer: &mut Vec<T>,
    ) -> ReedResult<()> {
        let local = if let Some(restriction) = &field.restriction {
            local_buffer.resize(restriction.local_size(), T::ZERO);
            restriction.apply(TransposeMode::NoTranspose, source, local_buffer)?;
            local_buffer.as_slice()
        } else {
            source
        };

        if let Some(basis) = &field.basis {
            let qcomp = Self::qpoint_component_count(field, eval_mode)?;
            q_buffer.resize(self.num_elem * basis.num_qpoints() * qcomp, T::ZERO);
            basis.apply(self.num_elem, false, eval_mode, local, q_buffer)?;
        } else {
            q_buffer.clear();
            q_buffer.extend_from_slice(local);
        }

        Ok(())
    }

    /// Basis^T apply + restriction scatter for one output field.
    fn scatter_output_to_slice(
        &self,
        field: &WgpuOperatorField<T>,
        eval_mode: EvalMode,
        q_output: &[T],
        local_buffer: &mut Vec<T>,
        active_output: &mut [T],
    ) -> ReedResult<()> {
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

        match &field.vector {
            WgpuFieldVector::Active => {
                if let Some(restriction) = &field.restriction {
                    restriction.apply(TransposeMode::Transpose, local, active_output)
                } else {
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
    // Multi-field forward apply
    // ------------------------------------------------------------------

    fn apply_field_buffers_impl(
        &self,
        inputs: &[(&str, &dyn VectorTrait<T>)],
        outputs: &mut [(&str, &mut dyn VectorTrait<T>)],
        add: bool,
    ) -> ReedResult<()> {
        self.validate_named_field_buffers(inputs, &*outputs)?;

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

        let (_q_inputs, q_outputs) = self.run_forward_qfunction_stages_named(inputs)?;

        let mut output_locals: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();
        for (slot, plan) in self.output_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            let j = Self::lookup_named_write_slot(outputs, field.name.as_str())?;
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

    // ------------------------------------------------------------------
    // Adjoint path
    // ------------------------------------------------------------------

    fn ensure_adjoint_io_lengths(
        &self,
        range_cotangent: &dyn VectorTrait<T>,
        domain_cotangent: &dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        let out_len = self.active_output_global_len()?.ok_or_else(|| {
            ReedError::Operator(
                "operator adjoint: could not infer active output global length".into(),
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
                "operator adjoint: could not infer active input global length".into(),
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

    /// Pull range cotangent to quadrature points: restriction(NoTranspose) -> basis(forward).
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

    /// Scatter domain cotangent from q-points to global: basis^T -> restriction(Transpose).
    fn scatter_domain_cotangent_qp_to_global(
        &self,
        field: &WgpuOperatorField<T>,
        eval_mode: EvalMode,
        q_in_cot: &[T],
        local_buffer: &mut Vec<T>,
        domain_global: &mut [T],
    ) -> ReedResult<()> {
        match field.vector {
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

    /// Core adjoint apply (single-buffer path).
    fn execute_adjoint(
        &self,
        range_cotangent: &dyn VectorTrait<T>,
        domain_cotangent: &mut dyn VectorTrait<T>,
        add: bool,
    ) -> ReedResult<()> {
        if self.requires_field_named_buffers() {
            return Err(ReedError::Operator(
                "operator adjoint: this operator uses multiple active fields; use apply_field_buffers_with_transpose (Adjoint) with named buffers"
                    .into(),
            ));
        }
        self.ensure_adjoint_io_lengths(range_cotangent, &*domain_cotangent)?;

        // Check that the active qfunction supports transpose
        if !self.active_qfunction().supports_operator_transpose() {
            return Err(ReedError::Operator(
                "operator adjoint: qfunction does not implement apply_operator_transpose".into(),
            ));
        }

        if !add {
            domain_cotangent.set_value(T::ZERO)?;
        }

        // 1. Evaluate passive input fields forward
        let mut q_passive_fwd: Vec<Vec<T>> =
            (0..self.num_qfunction_inputs).map(|_| Vec::new()).collect();
        let mut input_locals: Vec<Vec<T>> =
            (0..self.num_qfunction_inputs).map(|_| Vec::new()).collect();
        let dummy_vec = reed_cpu::vector::CpuVector::new(0);
        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            match &field.vector {
                WgpuFieldVector::Passive(_) => {
                    self.prepare_input_into_single(
                        field,
                        plan.eval_mode,
                        &dummy_vec,
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

        // 2. Pull range cotangent to q-points for each output field
        let mut q_out_cot: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();
        let mut output_locals: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();
        for (slot, plan) in self.output_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            self.pull_range_cotangent_to_qp(
                field,
                plan.eval_mode,
                range_cotangent.as_slice(),
                &mut output_locals[slot],
                &mut q_out_cot[slot],
            )?;
        }

        // 3. Prepare input cotangent buffers and call QFunction transpose
        let input_descriptors = self.active_qfunction().inputs();
        let mut q_in_cot: Vec<Vec<T>> =
            (0..self.num_qfunction_inputs).map(|_| Vec::new()).collect();
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

        let qf = self.active_qfunction();
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
        qf.apply_operator_transpose_with_primal(
            self.qfunction_context.as_bytes(),
            self.num_elem * self.num_qpoints,
            &primal_q_inputs,
            &out_cot_refs,
            &mut in_cot_mut,
        )?;

        // 4. Scatter domain cotangent back
        let dom_sl = domain_cotangent.as_mut_slice();
        let mut input_locals_ct: Vec<Vec<T>> = (0..self.num_qfunction_inputs)
            .map(|_| Vec::new())
            .collect();
        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            if !matches!(field.vector, WgpuFieldVector::Active) {
                continue;
            }
            self.scatter_domain_cotangent_qp_to_global(
                field,
                plan.eval_mode,
                &q_in_cot[slot],
                &mut input_locals_ct[slot],
                dom_sl,
            )?;
        }

        Ok(())
    }

    /// Named-buffer adjoint path.
    fn execute_adjoint_field_buffers_impl(
        &self,
        range: &[(&str, &dyn VectorTrait<T>)],
        domain: &mut [(&str, &mut dyn VectorTrait<T>)],
        add: bool,
    ) -> ReedResult<()> {
        self.validate_adjoint_named_field_buffers(range, &*domain)?;

        if !self.active_qfunction().supports_operator_transpose() {
            return Err(ReedError::Operator(
                "operator adjoint: qfunction does not implement apply_operator_transpose".into(),
            ));
        }

        // Zero domain cotangent fields if not accumulating
        if !add {
            let zero_names: HashSet<&str> = self
                .distinct_active_input_field_indices()
                .iter()
                .map(|&i| self.fields[i].name.as_str())
                .collect();
            for (k, v) in domain.iter_mut() {
                if zero_names.contains(*k) {
                    v.set_value(T::ZERO)?;
                }
            }
        }

        // 1. Evaluate passive input fields forward
        let mut q_passive_fwd: Vec<Vec<T>> =
            (0..self.num_qfunction_inputs).map(|_| Vec::new()).collect();
        let mut input_locals: Vec<Vec<T>> =
            (0..self.num_qfunction_inputs).map(|_| Vec::new()).collect();
        let dummy_vec = reed_cpu::vector::CpuVector::new(0);
        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            match &field.vector {
                WgpuFieldVector::Passive(_) => {
                    self.prepare_input_into_single(
                        field,
                        plan.eval_mode,
                        &dummy_vec,
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

        // 2. Pull range cotangent to q-points for each output field
        let mut q_out_cot: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();
        let mut output_locals: Vec<Vec<T>> = (0..self.num_qfunction_outputs)
            .map(|_| Vec::new())
            .collect();
        for (slot, plan) in self.output_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            let range_sl =
                Self::lookup_named_read(range, field.name.as_str())?.as_slice();
            self.pull_range_cotangent_to_qp(
                field,
                plan.eval_mode,
                range_sl,
                &mut output_locals[slot],
                &mut q_out_cot[slot],
            )?;
        }

        // 3. Prepare input cotangent buffers and call QFunction transpose
        let input_descriptors = self.active_qfunction().inputs();
        let mut q_in_cot: Vec<Vec<T>> =
            (0..self.num_qfunction_inputs).map(|_| Vec::new()).collect();
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

        let qf = self.active_qfunction();
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
        qf.apply_operator_transpose_with_primal(
            self.qfunction_context.as_bytes(),
            self.num_elem * self.num_qpoints,
            &primal_q_inputs,
            &out_cot_refs,
            &mut in_cot_mut,
        )?;

        // 4. Scatter domain cotangent back - per-field output
        let mut input_locals_ct: Vec<Vec<T>> = (0..self.num_qfunction_inputs)
            .map(|_| Vec::new())
            .collect();
        for (slot, plan) in self.input_plans.iter().enumerate() {
            let field = &self.fields[plan.field_index];
            if !matches!(field.vector, WgpuFieldVector::Active) {
                continue;
            }
            let j = Self::lookup_named_write_slot(domain, field.name.as_str())?;
            let out_v = &mut *domain[j].1;
            self.scatter_domain_cotangent_qp_to_global(
                field,
                plan.eval_mode,
                &q_in_cot[slot],
                &mut input_locals_ct[slot],
                out_v.as_mut_slice(),
            )?;
        }

        Ok(())
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
        self.apply_forward(input, output, false)
    }

    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        self.apply_forward(input, output, true)
    }

    fn operator_label(&self) -> Option<&str> {
        self.op_label.as_deref()
    }

    fn requires_field_named_buffers(&self) -> bool {
        self.multi_distinct_active_io_fields()
    }

    fn global_vector_len_hint(&self) -> Option<usize> {
        self.active_input_global_len().ok().flatten()
    }

    fn linear_assemble_diagonal(&self, _assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "WgpuOperator::linear_assemble_diagonal is not implemented on the GPU path".into(),
        ))
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

/// Builder for [`WgpuOperator`].
///
/// # Device QFunction auto-integration
///
/// When the QFunction reports a [`gallery_name`](reed_core::QFunctionTrait::gallery_name),
/// the builder automatically tries to create a device-side counterpart.
/// This is used during the pointwise evaluation step, avoiding a host round-trip.
pub struct WgpuOperatorBuilder<T: Scalar> {
    runtime: Option<Arc<GpuRuntime>>,
    qfunction: Option<Box<dyn QFunctionTrait<T>>>,
    qfunction_context: Option<QFunctionContext>,
    op_label: Option<String>,
    fields: Vec<WgpuOperatorField<T>>,
    force_cpu_qfunction: bool,
}

impl<T: Scalar> Default for WgpuOperatorBuilder<T> {
    fn default() -> Self {
        Self {
            runtime: None,
            qfunction: None,
            qfunction_context: None,
            op_label: None,
            fields: Vec::new(),
            force_cpu_qfunction: false,
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

    /// User [`QFunctionContext`] buffer.
    pub fn qfunction_context(mut self, ctx: QFunctionContext) -> Self {
        self.qfunction_context = Some(ctx);
        self
    }

    /// Human-readable operator name for logging.
    pub fn operator_label(mut self, label: impl Into<String>) -> Self {
        self.op_label = Some(label.into());
        self
    }

    /// Disable device QFunction auto-detection; always use CPU QFunction.
    pub fn force_cpu_qfunction(mut self) -> Self {
        self.force_cpu_qfunction = true;
        self
    }

    /// Add a named field with optional restriction, basis, and vector role.
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

    /// Try to create a device QFunction from the gallery name.
    fn try_device_qfunction(
        qfunction: &dyn QFunctionTrait<T>,
        runtime: &Arc<GpuRuntime>,
    ) -> Option<Box<dyn QFunctionTrait<T>>> {
        let name = qfunction.gallery_name()?;
        if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
            return None;
        }
        let result = crate::qfunction_device::try_create_device_q_function_f32(name, runtime.clone())?;
        match result {
            Ok(device_qf) => Some(unsafe { crate::coerce_qfunction_f32_box(device_qf) }),
            Err(_) => None,
        }
    }

    /// Consume the builder and produce a [`WgpuOperator`].
    pub fn build(self) -> ReedResult<WgpuOperator<T>> {
        let runtime = self
            .runtime
            .ok_or_else(|| ReedError::Operator("WgpuOperatorBuilder requires a GpuRuntime".into()))?;

        let qfunction = self
            .qfunction
            .ok_or_else(|| ReedError::Operator("WgpuOperatorBuilder requires a qfunction".into()))?;

        // Auto-detect device QFunction
        let device_qfunction = if !self.force_cpu_qfunction {
            Self::try_device_qfunction(qfunction.as_ref(), &runtime)
        } else {
            None
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

        let num_elem = self
            .fields
            .iter()
            .find_map(|f| f.restriction.as_ref().map(|r| r.num_elements()))
            .ok_or_else(|| {
                ReedError::Operator(
                    "WgpuOperatorBuilder requires at least one field with a restriction".into(),
                )
            })?;

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
            device_qfunction,
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

    #[test]
    fn build_minimal_operator() {
        let Some(rt) = gpu_runtime_or_skip() else { return; };
        let nelem = 2usize;
        let p = 2usize;
        let q = 3usize;
        let num_dof = p;
        let ncomp = 1;
        let offsets: Vec<i32> = (0..nelem * num_dof).map(|i| i as i32).collect();
        let restr = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, nelem * num_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let restr2 = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, nelem * num_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis2 = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let qf = Identity::with_components(1);
        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(qf))
            .field("input", Some(Box::new(restr)), Some(Box::new(basis)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr2)), Some(Box::new(basis2)), WgpuFieldVector::Active)
            .build();
        assert!(op.is_ok(), "build failed: {:?}", op.err());
    }

    #[test]
    fn identity_operator_matches_cpu() {
        let Some(rt) = gpu_runtime_or_skip() else { return; };
        let nelem = 2usize;
        let p = 2usize;
        let q = 3usize;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;
        let offsets: Vec<i32> = (0..nelem * num_dof).map(|i| i as i32).collect();
        let restr_w = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis_w = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let restr_w2 = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis_w2 = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let restr_c = reed_cpu::elem_restriction::CpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets,
        ).unwrap();
        let basis_c = reed_cpu::basis_lagrange::LagrangeBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss).unwrap();
        let op_w = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(Identity::with_components(1)))
            .field("input", Some(Box::new(restr_w)), Some(Box::new(basis_w)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr_w2)), Some(Box::new(basis_w2)), WgpuFieldVector::Active)
            .build().unwrap();
        let op_c = reed_cpu::operator::OperatorBuilder::new()
            .qfunction(Box::new(Identity::with_components(1)))
            .field("input", Some(&restr_c), Some(&basis_c), reed_cpu::operator::FieldVector::Active)
            .field("output", Some(&restr_c), Some(&basis_c), reed_cpu::operator::FieldVector::Active)
            .build().unwrap();
        let input = CpuVector::from_vec((0..global_dof).map(|i| 0.1 * (i + 1) as f32).collect());
        let mut out_w = CpuVector::new(global_dof);
        let mut out_c = CpuVector::new(global_dof);
        op_w.apply(&input, &mut out_w).unwrap();
        op_c.apply(&input, &mut out_c).unwrap();
        for i in 0..global_dof {
            let diff = (out_w.as_slice()[i] - out_c.as_slice()[i]).abs();
            assert!(diff < 1e-4, "mismatch at {i}: wgpu={} cpu={}", out_w.as_slice()[i], out_c.as_slice()[i]);
        }
    }

    #[test]
    fn apply_add_accumulates() {
        let Some(rt) = gpu_runtime_or_skip() else { return; };
        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;
        let offsets: Vec<i32> = (0..nelem * num_dof).map(|i| i as i32).collect();
        let restr = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr2 = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let basis2 = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(Identity::with_components(1)))
            .field("input", Some(Box::new(restr)), Some(Box::new(basis)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr2)), Some(Box::new(basis2)), WgpuFieldVector::Active)
            .build().unwrap();
        let input = CpuVector::from_vec((0..global_dof).map(|i| (i + 1) as f32).collect());
        let mut output = CpuVector::new(global_dof);
        op.apply(&input, &mut output).unwrap();
        let first: Vec<f32> = output.as_slice().to_vec();
        op.apply_add(&input, &mut output).unwrap();
        for i in 0..global_dof {
            assert!((output.as_slice()[i] - 2.0 * first[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn identity_operator_adjoint() {
        let Some(rt) = gpu_runtime_or_skip() else { return; };
        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;
        let offsets: Vec<i32> = (0..nelem * num_dof).map(|i| i as i32).collect();
        let restr_in = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr_out = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis_in = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let basis_out = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(Identity::with_components(1)))
            .field("input", Some(Box::new(restr_in)), Some(Box::new(basis_in)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr_out)), Some(Box::new(basis_out)), WgpuFieldVector::Active)
            .build().unwrap();
        let x = CpuVector::from_vec((0..global_dof).map(|i| (i + 1) as f32).collect());
        let mut y = CpuVector::new(global_dof);
        op.apply(&x, &mut y).unwrap();
        let mut adj_y = CpuVector::new(global_dof);
        op.apply_with_transpose(OperatorTransposeRequest::Adjoint, &y, &mut adj_y).unwrap();
        let adj_sl = adj_y.as_slice();
        let has_nonzero = adj_sl.iter().any(|&v| v.abs() > 1e-8);
        assert!(has_nonzero, "adjoint output is all zeros");
        let mut inner_fwd = 0.0f32;
        let y_sl = y.as_slice();
        for i in 0..global_dof { inner_fwd += y_sl[i] * y_sl[i]; }
        let mut inner_adj = 0.0f32;
        let x_sl = x.as_slice();
        for i in 0..global_dof { inner_adj += adj_sl[i] * x_sl[i]; }
        let rel_diff = (inner_fwd - inner_adj).abs() / inner_fwd.max(1e-12);
        assert!(rel_diff < 1e-3, "inner product identity fails: fwd={} adj={}", inner_fwd, inner_adj);
    }

    #[test]
    fn identity_operator_adjoint_accumulates() {
        let Some(rt) = gpu_runtime_or_skip() else { return; };
        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;
        let offsets: Vec<i32> = (0..nelem * num_dof).map(|i| i as i32).collect();
        let restr_in = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr_out = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis_in = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let basis_out = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(Identity::with_components(1)))
            .field("input", Some(Box::new(restr_in)), Some(Box::new(basis_in)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr_out)), Some(Box::new(basis_out)), WgpuFieldVector::Active)
            .build().unwrap();
        let x = CpuVector::from_vec((0..global_dof).map(|i| (i + 1) as f32).collect());
        let mut y = CpuVector::new(global_dof);
        op.apply(&x, &mut y).unwrap();
        let mut adj_y = CpuVector::new(global_dof);
        op.apply_with_transpose(OperatorTransposeRequest::Adjoint, &y, &mut adj_y).unwrap();
        let first_adj: Vec<f32> = adj_y.as_slice().to_vec();
        op.apply_add_with_transpose(OperatorTransposeRequest::Adjoint, &y, &mut adj_y).unwrap();
        for i in 0..global_dof {
            assert!((adj_y.as_slice()[i] - 2.0 * first_adj[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn mass_operator_with_passive_qdata() {
        let Some(rt) = gpu_runtime_or_skip() else { return; };
        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;
        let offsets: Vec<i32> = (0..nelem * num_dof).map(|i| i as i32).collect();
        let restr_u = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr_v = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis_u = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let basis_v = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let qdata: Vec<f32> = (0..nelem * q).map(|i| 0.5 + 0.1 * (i as f32)).collect();
        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(MassApply::default()))
            .field("u", Some(Box::new(restr_u)), Some(Box::new(basis_u)), WgpuFieldVector::Active)
            .field("qdata", None, None, WgpuFieldVector::Passive(Box::new(CpuVector::from_vec(qdata))))
            .field("v", Some(Box::new(restr_v)), Some(Box::new(basis_v)), WgpuFieldVector::Active)
            .build().unwrap();
        let input = CpuVector::from_vec((0..global_dof).map(|i| (i + 1) as f32).collect());
        let mut output = CpuVector::new(global_dof);
        op.apply(&input, &mut output).unwrap();
        assert!(output.as_slice().iter().any(|&v| v.abs() > 1e-8), "mass operator output is all zeros");
    }

    #[test]
    fn named_field_buffers_identity() {
        let Some(rt) = gpu_runtime_or_skip() else { return; };
        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;
        let offsets: Vec<i32> = (0..nelem * num_dof).map(|i| i as i32).collect();
        let restr_in = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr_out = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis_in = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let basis_out = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(Identity::with_components(1)))
            .field("input", Some(Box::new(restr_in)), Some(Box::new(basis_in)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr_out)), Some(Box::new(basis_out)), WgpuFieldVector::Active)
            .build().unwrap();
        let x = CpuVector::from_vec((0..global_dof).map(|i| (i + 1) as f32).collect());
        let mut y = CpuVector::new(global_dof);
        let inputs: &[(&str, &dyn VectorTrait<f32>)] = &[("input", &x)];
        let mut outputs: Vec<(&str, &mut dyn VectorTrait<f32>)> = vec![("output", &mut y)];
        op.apply_field_buffers(inputs, &mut outputs).unwrap();
        assert!(y.as_slice().iter().any(|&v| v.abs() > 1e-8), "named field buffers output is all zeros");
    }

    #[test]
    fn named_field_buffers_adjoint() {
        let Some(rt) = gpu_runtime_or_skip() else { return; };
        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;
        let offsets: Vec<i32> = (0..nelem * num_dof).map(|i| i as i32).collect();
        let restr_in = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr_out = WgpuElemRestriction::<f32>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis_in = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let basis_out = WgpuBasis::<f32>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(Identity::with_components(1)))
            .field("input", Some(Box::new(restr_in)), Some(Box::new(basis_in)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr_out)), Some(Box::new(basis_out)), WgpuFieldVector::Active)
            .build().unwrap();
        let x = CpuVector::from_vec((0..global_dof).map(|i| (i + 1) as f32).collect());
        let mut y = CpuVector::new(global_dof);
        let inputs: &[(&str, &dyn VectorTrait<f32>)] = &[("input", &x)];
        let mut outputs: Vec<(&str, &mut dyn VectorTrait<f32>)> = vec![("output", &mut y)];
        op.apply_field_buffers(inputs, &mut outputs).unwrap();
        let mut adj_x = CpuVector::new(global_dof);
        let adj_inputs: &[(&str, &dyn VectorTrait<f32>)] = &[("output", &y)];
        let mut adj_outputs: Vec<(&str, &mut dyn VectorTrait<f32>)> = vec![("input", &mut adj_x)];
        op.apply_field_buffers_with_transpose(
            OperatorTransposeRequest::Adjoint, adj_inputs, &mut adj_outputs,
        ).unwrap();
        assert!(adj_x.as_slice().iter().any(|&v| v.abs() > 1e-8), "named adjoint output is all zeros");
    }

    #[test]
    fn device_qfunction_auto_detect_skips_f64() {
        let Some(rt) = gpu_runtime_or_skip() else { return; };
        let nelem = 2usize;
        let p = 2;
        let q = 3;
        let num_dof = p;
        let ncomp = 1;
        let global_dof = nelem * num_dof;
        let offsets: Vec<i32> = (0..nelem * num_dof).map(|i| i as i32).collect();
        let restr = WgpuElemRestriction::<f64>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let restr2 = WgpuElemRestriction::<f64>::new_offset(
            nelem, num_dof, ncomp, 1, global_dof, &offsets, Some(rt.clone()),
        ).unwrap();
        let basis = WgpuBasis::<f64>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let basis2 = WgpuBasis::<f64>::new(1, ncomp, p, q, QuadMode::Gauss, Some(rt.clone())).unwrap();
        let op = WgpuOperatorBuilder::new()
            .runtime(rt.clone())
            .qfunction(Box::new(Identity::with_components(1)))
            .field("input", Some(Box::new(restr)), Some(Box::new(basis)), WgpuFieldVector::Active)
            .field("output", Some(Box::new(restr2)), Some(Box::new(basis2)), WgpuFieldVector::Active)
            .build().unwrap();
        let input = CpuVector::from_vec((0..global_dof).map(|i| 0.1 * (i + 1) as f64).collect());
        let mut output = CpuVector::new(global_dof);
        op.apply(&input, &mut output).unwrap();
        assert!(output.as_slice().iter().any(|&v| v.abs() > 1e-8), "f64 operator output is all zeros");
    }
}
