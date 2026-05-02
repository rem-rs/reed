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

use std::sync::Arc;

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
    // Forward apply pipeline
    // ------------------------------------------------------------------

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

        // Step 3: Resize output q-point buffers and call QFunction (CPU for v1)
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
            WgpuFieldVector::Active => active_input.as_slice(),
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

    /// v1: diagonal assembly not implemented on GPU path.
    fn linear_assemble_diagonal(&self, _assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        Err(ReedError::Operator(
            "WgpuOperator::linear_assemble_diagonal is not implemented on the GPU path".into(),
        ))
    }

    /// v1: adjoint not implemented; Forward delegates to [`Self::apply`].
    fn apply_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply(input, output),
            OperatorTransposeRequest::Adjoint => Err(ReedError::Operator(
                "WgpuOperator::apply_with_transpose(Adjoint) is not implemented in v1".into(),
            )),
        }
    }

    /// v1: adjoint not implemented; Forward delegates to [`Self::apply_add`].
    fn apply_add_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_add(input, output),
            OperatorTransposeRequest::Adjoint => Err(ReedError::Operator(
                "WgpuOperator::apply_add_with_transpose(Adjoint) is not implemented in v1".into(),
            )),
        }
    }

    fn check_ready(&self) -> ReedResult<()> {
        // v1: basic validation
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
}

impl<T: Scalar> Default for WgpuOperatorBuilder<T> {
    fn default() -> Self {
        Self {
            runtime: None,
            qfunction: None,
            qfunction_context: None,
            op_label: None,
            fields: Vec::new(),
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
    pub fn build(self) -> ReedResult<WgpuOperator<T>> {
        let runtime = self
            .runtime
            .ok_or_else(|| ReedError::Operator("WgpuOperatorBuilder requires a GpuRuntime".into()))?;

        let qfunction = self
            .qfunction
            .ok_or_else(|| ReedError::Operator("WgpuOperatorBuilder requires a qfunction".into()))?;

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
}
