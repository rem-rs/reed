//! Exterior (boundary) gallery QFunctions.
//!
//! These correspond to libCEED exterior gallery kernels registered via
//! `CeedQFunctionCreateActiveByName` and related boundary paths.

use super::helpers::scale_alpha_from_libceed_context;
use reed_core::{
    enums::EvalMode,
    error::ReedResult,
    qfunction::{QFunctionCategory, QFunctionField, QFunctionTrait},
    scalar::Scalar,
    ReedError,
};

// --- NeumannApply ---------------------------------------------------------

/// `"NeumannApply"` — identity (pass-through) on the boundary.
///
/// Passes the test function evaluated at face quadrature points straight through.
/// Boundary integration is handled by the operator.
#[derive(Clone)]
pub struct NeumannApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for NeumannApply {
    fn default() -> Self {
        Self {
            inputs: vec![QFunctionField {
                name: "v".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
            outputs: vec![QFunctionField {
                name: "v".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for NeumannApply {
    fn context_byte_len(&self) -> usize {
        0
    }

    fn inputs(&self) -> &[QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[QFunctionField] {
        &self.outputs
    }

    fn q_function_category(&self) -> QFunctionCategory {
        QFunctionCategory::Exterior
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> ReedResult<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "NeumannApply expects 1 input and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let v = &mut outputs[0];
        if u.len() != q || v.len() != q {
            return Err(ReedError::QFunction(
                "NeumannApply: buffer length mismatch".into(),
            ));
        }
        v.copy_from_slice(u);
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
        if output_cotangents.len() != 1 || input_cotangents.len() != 1 {
            return Err(ReedError::QFunction(
                "NeumannApply transpose expects 1 output cotangent and 1 input cotangent buffer"
                    .into(),
            ));
        }
        let dv = output_cotangents[0];
        let du = &mut input_cotangents[0];
        if dv.len() != q || du.len() != q {
            return Err(ReedError::QFunction(
                "NeumannApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            du[i] += dv[i];
        }
        Ok(())
    }
}

// --- RobinApply -----------------------------------------------------------

/// `"RobinApply"` — multiply boundary value by `alpha` from context.
///
/// Reads a `f64` LE `alpha` parameter from the 8-byte context, cast to `T`,
/// and scales each quadrature point value: `v[q] = u[q] * alpha`.
#[derive(Clone)]
pub struct RobinApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for RobinApply {
    fn default() -> Self {
        Self {
            inputs: vec![QFunctionField {
                name: "u".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
            outputs: vec![QFunctionField {
                name: "v".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for RobinApply {
    fn context_byte_len(&self) -> usize {
        8
    }

    fn inputs(&self) -> &[QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[QFunctionField] {
        &self.outputs
    }

    fn q_function_category(&self) -> QFunctionCategory {
        QFunctionCategory::Exterior
    }

    fn apply(
        &self,
        ctx: &[u8],
        q: usize,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> ReedResult<()> {
        let alpha = scale_alpha_from_libceed_context::<T>(ctx)?;
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "RobinApply expects 1 input and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let v = &mut outputs[0];
        if u.len() != q || v.len() != q {
            return Err(ReedError::QFunction(
                "RobinApply: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            v[i] = alpha * u[i];
        }
        Ok(())
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        ctx: &[u8],
        q: usize,
        output_cotangents: &[&[T]],
        input_cotangents: &mut [&mut [T]],
    ) -> ReedResult<()> {
        let alpha = scale_alpha_from_libceed_context::<T>(ctx)?;
        if output_cotangents.len() != 1 || input_cotangents.len() != 1 {
            return Err(ReedError::QFunction(
                "RobinApply transpose expects 1 output cotangent and 1 input cotangent buffer"
                    .into(),
            ));
        }
        let dv = output_cotangents[0];
        let du = &mut input_cotangents[0];
        if dv.len() != q || du.len() != q {
            return Err(ReedError::QFunction(
                "RobinApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            du[i] += alpha * dv[i];
        }
        Ok(())
    }
}

// --- MassBoundaryApply ----------------------------------------------------

/// `"MassBoundaryApply"` — boundary mass integrand: `u[q]` at quadrature points.
///
/// Provides the solution field on the boundary. Quadrature weights and
/// test-function scaling are handled by the operator.
#[derive(Clone)]
pub struct MassBoundaryApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for MassBoundaryApply {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "u".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Interp,
                },
                QFunctionField {
                    name: "v".into(),
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

impl<T: Scalar> QFunctionTrait<T> for MassBoundaryApply {
    fn context_byte_len(&self) -> usize {
        0
    }

    fn inputs(&self) -> &[QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[QFunctionField] {
        &self.outputs
    }

    fn q_function_category(&self) -> QFunctionCategory {
        QFunctionCategory::Exterior
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
                "MassBoundaryApply expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let v = &mut outputs[0];
        if u.len() != q || v.len() != q {
            return Err(ReedError::QFunction(
                "MassBoundaryApply: buffer length mismatch".into(),
            ));
        }
        v.copy_from_slice(u);
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
                "MassBoundaryApply transpose expects 1 output cotangent and 2 input buffers".into(),
            ));
        }
        let dv = output_cotangents[0];
        if dv.len() != q {
            return Err(ReedError::QFunction(
                "MassBoundaryApply transpose: buffer length mismatch".into(),
            ));
        }
        // Forward inputs are [u, v]; transpose adds cotangent to u, zeros to v.
        let (du_slice, _dv_slice) = input_cotangents.split_at_mut(1);
        let du = &mut du_slice[0];
        if du.len() != q {
            return Err(ReedError::QFunction(
                "MassBoundaryApply transpose: input cotangent buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            du[i] += dv[i];
        }
        Ok(())
    }
}

// --- DiffusionBoundaryApply -----------------------------------------------

/// `"DiffusionBoundaryApply"` — boundary diffusion flux: `du[q]` at quadrature points.
///
/// Provides the directional derivative (gradient component) on the boundary.
/// The normal projection and integration weights are handled by the operator.
/// Designed for tensor-product bases with GLL quadrature on the face.
#[derive(Clone)]
pub struct DiffusionBoundaryApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for DiffusionBoundaryApply {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "du".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Grad,
                },
                QFunctionField {
                    name: "v".into(),
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

impl<T: Scalar> QFunctionTrait<T> for DiffusionBoundaryApply {
    fn context_byte_len(&self) -> usize {
        0
    }

    fn inputs(&self) -> &[QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[QFunctionField] {
        &self.outputs
    }

    fn q_function_category(&self) -> QFunctionCategory {
        QFunctionCategory::Exterior
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
                "DiffusionBoundaryApply expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let v = &mut outputs[0];
        if du.len() != q || v.len() != q {
            return Err(ReedError::QFunction(
                "DiffusionBoundaryApply: buffer length mismatch".into(),
            ));
        }
        v.copy_from_slice(du);
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
                "DiffusionBoundaryApply transpose expects 1 output cotangent and 2 input buffers"
                    .into(),
            ));
        }
        let dv = output_cotangents[0];
        if dv.len() != q {
            return Err(ReedError::QFunction(
                "DiffusionBoundaryApply transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_slice, _dv_slice) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_slice[0];
        if ddu.len() != q {
            return Err(ReedError::QFunction(
                "DiffusionBoundaryApply transpose: input cotangent buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            ddu[i] += dv[i];
        }
        Ok(())
    }
}

// --- ScaleBoundaryApply ----------------------------------------------------

/// `"ScaleBoundaryApply"` — multiply boundary values by a scalar from context.
///
/// Reads an `f64` LE scale factor from the 8-byte context and applies it
/// to each quadrature point: `v[q] = u[q] * scale`.
#[derive(Clone)]
pub struct ScaleBoundaryApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for ScaleBoundaryApply {
    fn default() -> Self {
        Self {
            inputs: vec![QFunctionField {
                name: "u".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
            outputs: vec![QFunctionField {
                name: "v".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for ScaleBoundaryApply {
    fn context_byte_len(&self) -> usize {
        8
    }

    fn inputs(&self) -> &[QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[QFunctionField] {
        &self.outputs
    }

    fn q_function_category(&self) -> QFunctionCategory {
        QFunctionCategory::Exterior
    }

    fn apply(
        &self,
        ctx: &[u8],
        q: usize,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> ReedResult<()> {
        let scale = scale_alpha_from_libceed_context::<T>(ctx)?;
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "ScaleBoundaryApply expects 1 input and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let v = &mut outputs[0];
        if u.len() != q || v.len() != q {
            return Err(ReedError::QFunction(
                "ScaleBoundaryApply: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            v[i] = scale * u[i];
        }
        Ok(())
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        ctx: &[u8],
        q: usize,
        output_cotangents: &[&[T]],
        input_cotangents: &mut [&mut [T]],
    ) -> ReedResult<()> {
        let scale = scale_alpha_from_libceed_context::<T>(ctx)?;
        if output_cotangents.len() != 1 || input_cotangents.len() != 1 {
            return Err(ReedError::QFunction(
                "ScaleBoundaryApply transpose expects 1 output cotangent and 1 input cotangent buffer"
                    .into(),
            ));
        }
        let dv = output_cotangents[0];
        let du = &mut input_cotangents[0];
        if dv.len() != q || du.len() != q {
            return Err(ReedError::QFunction(
                "ScaleBoundaryApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            du[i] += scale * dv[i];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reed_core::qfunction::QFunctionTrait;

    #[test]
    fn neumann_apply_identity() {
        let qf = NeumannApply::default();
        assert_eq!(
            <NeumannApply as QFunctionTrait<f64>>::q_function_category(&qf),
            QFunctionCategory::Exterior
        );
        assert_eq!(
            <NeumannApply as QFunctionTrait<f64>>::context_byte_len(&qf),
            0
        );
        let u = vec![1.0, 2.0, 3.0];
        let mut v = vec![0.0; 3];
        <NeumannApply as QFunctionTrait<f64>>::apply(
            &qf, &[], 3, &[u.as_slice()], &mut [&mut v],
        )
        .unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn robin_apply_scale() {
        let qf = RobinApply::default();
        assert_eq!(
            <RobinApply as QFunctionTrait<f64>>::q_function_category(&qf),
            QFunctionCategory::Exterior
        );
        assert_eq!(
            <RobinApply as QFunctionTrait<f64>>::context_byte_len(&qf),
            8
        );
        let mut ctx = [0u8; 8];
        ctx.copy_from_slice(&2.5f64.to_le_bytes());
        let u = vec![1.0, 2.0, 3.0];
        let mut v = vec![0.0; 3];
        <RobinApply as QFunctionTrait<f64>>::apply(
            &qf, &ctx, 3, &[u.as_slice()], &mut [&mut v],
        )
        .unwrap();
        assert_eq!(v, vec![2.5, 5.0, 7.5]);
    }

    #[test]
    fn neumann_apply_transpose() {
        let qf = NeumannApply::default();
        let mut du = vec![0.0f64; 3];
        let dv = vec![1.0f64, 2.0, 3.0];
        <NeumannApply as QFunctionTrait<f64>>::apply_operator_transpose(
            &qf, &[], 3, &[dv.as_slice()], &mut [&mut du],
        )
        .unwrap();
        assert_eq!(du, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn robin_apply_transpose() {
        let qf = RobinApply::default();
        let mut ctx = [0u8; 8];
        ctx.copy_from_slice(&2.0f64.to_le_bytes());
        let mut du = vec![0.0f64; 3];
        let dv = vec![1.0f64, 2.0, 3.0];
        <RobinApply as QFunctionTrait<f64>>::apply_operator_transpose(
            &qf, &ctx, 3, &[dv.as_slice()], &mut [&mut du],
        )
        .unwrap();
        assert_eq!(du, vec![2.0, 4.0, 6.0]);
    }

    // --- MassBoundaryApply tests ---------------------------------------------

    #[test]
    fn mass_boundary_apply_category() {
        let qf = MassBoundaryApply::default();
        assert_eq!(
            <MassBoundaryApply as QFunctionTrait<f64>>::q_function_category(&qf),
            QFunctionCategory::Exterior
        );
        assert_eq!(
            <MassBoundaryApply as QFunctionTrait<f64>>::context_byte_len(&qf),
            0
        );
    }

    #[test]
    fn mass_boundary_apply_passthrough() {
        let qf = MassBoundaryApply::default();
        let u = vec![1.0, 2.0, 3.0];
        let v = vec![0.0; 3]; // test function (unused in apply body)
        let mut out = vec![0.0f64; 3];
        <MassBoundaryApply as QFunctionTrait<f64>>::apply(
            &qf,
            &[],
            3,
            &[u.as_slice(), v.as_slice()],
            &mut [&mut out],
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn mass_boundary_apply_transpose() {
        let qf = MassBoundaryApply::default();
        let mut du = vec![0.0f64; 3];
        let mut dv = vec![0.0f64; 3];
        let dout = vec![1.0f64, 2.0, 3.0];
        <MassBoundaryApply as QFunctionTrait<f64>>::apply_operator_transpose(
            &qf,
            &[],
            3,
            &[dout.as_slice()],
            &mut [&mut du, &mut dv],
        )
        .unwrap();
        assert_eq!(du, vec![1.0, 2.0, 3.0]);
        assert_eq!(dv, vec![0.0, 0.0, 0.0]);
    }

    // --- DiffusionBoundaryApply tests ---------------------------------------

    #[test]
    fn diffusion_boundary_apply_category() {
        let qf = DiffusionBoundaryApply::default();
        assert_eq!(
            <DiffusionBoundaryApply as QFunctionTrait<f64>>::q_function_category(&qf),
            QFunctionCategory::Exterior
        );
        assert_eq!(
            <DiffusionBoundaryApply as QFunctionTrait<f64>>::context_byte_len(&qf),
            0
        );
    }

    #[test]
    fn diffusion_boundary_apply_passthrough() {
        let qf = DiffusionBoundaryApply::default();
        let du = vec![4.0, 5.0, 6.0];
        let v = vec![0.0; 3]; // test function (unused in apply body)
        let mut out = vec![0.0f64; 3];
        <DiffusionBoundaryApply as QFunctionTrait<f64>>::apply(
            &qf,
            &[],
            3,
            &[du.as_slice(), v.as_slice()],
            &mut [&mut out],
        )
        .unwrap();
        assert_eq!(out, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn diffusion_boundary_apply_transpose() {
        let qf = DiffusionBoundaryApply::default();
        let mut ddu = vec![0.0f64; 3];
        let mut dv = vec![0.0f64; 3];
        let dout = vec![1.0f64, 2.0, 3.0];
        <DiffusionBoundaryApply as QFunctionTrait<f64>>::apply_operator_transpose(
            &qf,
            &[],
            3,
            &[dout.as_slice()],
            &mut [&mut ddu, &mut dv],
        )
        .unwrap();
        assert_eq!(ddu, vec![1.0, 2.0, 3.0]);
        assert_eq!(dv, vec![0.0, 0.0, 0.0]);
    }

    // --- ScaleBoundaryApply tests -------------------------------------------

    #[test]
    fn scale_boundary_apply_category() {
        let qf = ScaleBoundaryApply::default();
        assert_eq!(
            <ScaleBoundaryApply as QFunctionTrait<f64>>::q_function_category(&qf),
            QFunctionCategory::Exterior
        );
        assert_eq!(
            <ScaleBoundaryApply as QFunctionTrait<f64>>::context_byte_len(&qf),
            8
        );
    }

    #[test]
    fn scale_boundary_apply_scale() {
        let qf = ScaleBoundaryApply::default();
        let mut ctx = [0u8; 8];
        ctx.copy_from_slice(&3.0f64.to_le_bytes());
        let u = vec![1.0, 2.0, 3.0];
        let mut v = vec![0.0f64; 3];
        <ScaleBoundaryApply as QFunctionTrait<f64>>::apply(
            &qf,
            &ctx,
            3,
            &[u.as_slice()],
            &mut [&mut v],
        )
        .unwrap();
        assert_eq!(v, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn scale_boundary_apply_transpose() {
        let qf = ScaleBoundaryApply::default();
        let mut ctx = [0u8; 8];
        ctx.copy_from_slice(&3.0f64.to_le_bytes());
        let mut du = vec![0.0f64; 3];
        let dv = vec![1.0f64, 2.0, 3.0];
        <ScaleBoundaryApply as QFunctionTrait<f64>>::apply_operator_transpose(
            &qf,
            &ctx,
            3,
            &[dv.as_slice()],
            &mut [&mut du],
        )
        .unwrap();
        assert_eq!(du, vec![3.0, 6.0, 9.0]);
    }
}
