use crate::{enums::EvalMode, error::ReedResult, scalar::Scalar, ReedError};

/// libCEED-style **volume vs boundary** classification for quadrature functions.
///
/// - [`Interior`](Self::Interior) corresponds to libCEED **interior** gallery / `CeedQFunctionCreateInteriorByName`.
/// - [`Exterior`](Self::Exterior) corresponds to **exterior** (boundary) QFunctions (`CeedQFunctionCreateActiveByName`
///   and related paths in libCEED). Reed still evaluates them on host slices the same way; this flag is for
///   migration, diagnostics, and future operator assembly rules.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum QFunctionCategory {
    #[default]
    Interior,
    Exterior,
}

/// QFunction field descriptor.
#[derive(Debug, Clone)]
pub struct QFunctionField {
    pub name: String,
    pub num_comp: usize,
    pub eval_mode: EvalMode,
}

/// Quadrature-point pointwise operator trait.
///
/// `ctx` is always `context_byte_len()` bytes (often empty). This mirrors libCEED's
/// `CeedQFunctionContext` passed into user kernels.
#[cfg(not(target_arch = "wasm32"))]
pub trait QFunctionTrait<T: Scalar>: Send + Sync {
    /// Byte length of `ctx` passed to [`Self::apply`]. Zero means `ctx` is empty.
    fn context_byte_len(&self) -> usize {
        0
    }

    fn inputs(&self) -> &[QFunctionField];
    fn outputs(&self) -> &[QFunctionField];
    fn apply(
        &self,
        ctx: &[u8],
        q: usize,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> ReedResult<()>;

    /// Whether this qfunction participates in [`Self::apply_operator_transpose`] for operator
    /// adjoint (`OperatorTransposeRequest::Adjoint`) on `reed_cpu::CpuOperator`.
    fn supports_operator_transpose(&self) -> bool {
        false
    }

    /// Cotangent pushback at quadrature points (libCEED `CeedQFunctionApply` transpose mode).
    fn apply_operator_transpose(
        &self,
        ctx: &[u8],
        q: usize,
        output_cotangents: &[&[T]],
        input_cotangents: &mut [&mut [T]],
    ) -> ReedResult<()> {
        let _ = (ctx, q, output_cotangents, input_cotangents);
        Err(ReedError::QFunction(
            "apply_operator_transpose is not implemented for this qfunction".into(),
        ))
    }

    /// Optional adjoint API with access to forward quadrature inputs (`primal_inputs`) from the
    /// most recent forward pass, when available.
    fn apply_operator_transpose_with_primal(
        &self,
        ctx: &[u8],
        q: usize,
        primal_inputs: &[&[T]],
        output_cotangents: &[&[T]],
        input_cotangents: &mut [&mut [T]],
    ) -> ReedResult<()> {
        let _ = primal_inputs;
        self.apply_operator_transpose(ctx, q, output_cotangents, input_cotangents)
    }

    /// libCEED interior vs exterior (boundary) classification. Gallery kernels default to
    /// [`QFunctionCategory::Interior`]; use [`ClosureQFunction::new_with_category`] for user exterior kernels.
    fn q_function_category(&self) -> QFunctionCategory {
        QFunctionCategory::Interior
    }

    /// Gallery name for this QFunction, e.g. `"MassApply"`, `"Identity"`.
    /// Returns `None` for user-defined or non-gallery QFunctions.
    /// Used by GPU backends to auto-detect device-side QFunction counterparts.
    fn gallery_name(&self) -> Option<&str> {
        None
    }
}

#[cfg(target_arch = "wasm32")]
pub trait QFunctionTrait<T: Scalar> {
    fn context_byte_len(&self) -> usize {
        0
    }

    fn inputs(&self) -> &[QFunctionField];
    fn outputs(&self) -> &[QFunctionField];
    fn apply(
        &self,
        ctx: &[u8],
        q: usize,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> ReedResult<()>;

    fn supports_operator_transpose(&self) -> bool {
        false
    }

    fn apply_operator_transpose(
        &self,
        ctx: &[u8],
        q: usize,
        output_cotangents: &[&[T]],
        input_cotangents: &mut [&mut [T]],
    ) -> ReedResult<()> {
        let _ = (ctx, q, output_cotangents, input_cotangents);
        Err(ReedError::QFunction(
            "apply_operator_transpose is not implemented for this qfunction".into(),
        ))
    }

    fn apply_operator_transpose_with_primal(
        &self,
        ctx: &[u8],
        q: usize,
        primal_inputs: &[&[T]],
        output_cotangents: &[&[T]],
        input_cotangents: &mut [&mut [T]],
    ) -> ReedResult<()> {
        let _ = primal_inputs;
        self.apply_operator_transpose(ctx, q, output_cotangents, input_cotangents)
    }

    fn q_function_category(&self) -> QFunctionCategory {
        QFunctionCategory::Interior
    }

    /// Gallery name for this QFunction, e.g. `"MassApply"`, `"Identity"`.
    /// Returns `None` for user-defined or non-gallery QFunctions.
    /// Used by GPU backends to auto-detect device-side QFunction counterparts.
    fn gallery_name(&self) -> Option<&str> {
        None
    }
}

/// User closure type alias (`ctx` is the qfunction context byte slice).
#[cfg(not(target_arch = "wasm32"))]
pub type QFunctionClosure<T> =
    dyn Fn(&[u8], usize, &[&[T]], &mut [&mut [T]]) -> ReedResult<()> + Send + Sync;

#[cfg(target_arch = "wasm32")]
pub type QFunctionClosure<T> = dyn Fn(&[u8], usize, &[&[T]], &mut [&mut [T]]) -> ReedResult<()>;

/// Closure-based QFunction implementation.
pub struct ClosureQFunction<T: Scalar> {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
    context_byte_len: usize,
    category: QFunctionCategory,
    closure: Box<QFunctionClosure<T>>,
}

impl<T: Scalar> ClosureQFunction<T> {
    pub fn new(
        inputs: Vec<QFunctionField>,
        outputs: Vec<QFunctionField>,
        context_byte_len: usize,
        closure: Box<QFunctionClosure<T>>,
    ) -> Self {
        Self::new_with_category(
            inputs,
            outputs,
            context_byte_len,
            QFunctionCategory::Interior,
            closure,
        )
    }

    /// Same as [`Self::new`] but sets [`QFunctionCategory`] (e.g. [`QFunctionCategory::Exterior`] for boundary kernels).
    pub fn new_with_category(
        inputs: Vec<QFunctionField>,
        outputs: Vec<QFunctionField>,
        context_byte_len: usize,
        category: QFunctionCategory,
        closure: Box<QFunctionClosure<T>>,
    ) -> Self {
        Self {
            inputs,
            outputs,
            context_byte_len,
            category,
            closure,
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for ClosureQFunction<T> {
    fn context_byte_len(&self) -> usize {
        self.context_byte_len
    }

    fn inputs(&self) -> &[QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        ctx: &[u8],
        q: usize,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> ReedResult<()> {
        (self.closure)(ctx, q, inputs, outputs)
    }

    fn q_function_category(&self) -> QFunctionCategory {
        self.category
    }
}
