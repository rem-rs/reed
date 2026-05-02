use reed_core::{
    enums::EvalMode,
    error::ReedResult,
    qfunction::{QFunctionField, QFunctionTrait},
    scalar::Scalar,
    ReedError,
};

pub struct Mass1DBuild {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for Mass1DBuild {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "dx".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Grad,
                },
                QFunctionField {
                    name: "weights".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Weight,
                },
            ],
            outputs: vec![QFunctionField {
                name: "qdata".into(),
                num_comp: 1,
                eval_mode: EvalMode::None,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for Mass1DBuild {
    fn gallery_name(&self) -> Option<&str> {
        Some("Mass1DBuild")
    }

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
                "Mass1DBuild expects 2 inputs and 1 output".into(),
            ));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        for i in 0..q {
            qdata[i] = dx[i].abs() * weights[i];
        }
        Ok(())
    }
}

pub struct MassApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

pub struct Mass2DBuild {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

pub struct Mass3DBuild {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for Mass2DBuild {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "dx".into(),
                    num_comp: 4,
                    eval_mode: EvalMode::Grad,
                },
                QFunctionField {
                    name: "weights".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Weight,
                },
            ],
            outputs: vec![QFunctionField {
                name: "qdata".into(),
                num_comp: 1,
                eval_mode: EvalMode::None,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for Mass2DBuild {
    fn gallery_name(&self) -> Option<&str> {
        Some("Mass2DBuild")
    }

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
                "Mass2DBuild expects 2 inputs and 1 output".into(),
            ));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        for i in 0..q {
            let g00 = dx[i * 4];
            let g01 = dx[i * 4 + 1];
            let g10 = dx[i * 4 + 2];
            let g11 = dx[i * 4 + 3];
            let det_j = g00 * g11 - g01 * g10;
            qdata[i] = det_j.abs() * weights[i];
        }
        Ok(())
    }
}

impl Default for Mass3DBuild {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "dx".into(),
                    num_comp: 9,
                    eval_mode: EvalMode::Grad,
                },
                QFunctionField {
                    name: "weights".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Weight,
                },
            ],
            outputs: vec![QFunctionField {
                name: "qdata".into(),
                num_comp: 1,
                eval_mode: EvalMode::None,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for Mass3DBuild {
    fn gallery_name(&self) -> Option<&str> {
        Some("Mass3DBuild")
    }

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
                "Mass3DBuild expects 2 inputs and 1 output".into(),
            ));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        for i in 0..q {
            let j00 = dx[i * 9];
            let j01 = dx[i * 9 + 1];
            let j02 = dx[i * 9 + 2];
            let j10 = dx[i * 9 + 3];
            let j11 = dx[i * 9 + 4];
            let j12 = dx[i * 9 + 5];
            let j20 = dx[i * 9 + 6];
            let j21 = dx[i * 9 + 7];
            let j22 = dx[i * 9 + 8];

            let det_j = j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20)
                + j02 * (j10 * j21 - j11 * j20);
            qdata[i] = det_j.abs() * weights[i];
        }
        Ok(())
    }
}

impl Default for MassApply {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "u".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Interp,
                },
                QFunctionField {
                    name: "qdata".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::None,
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

impl<T: Scalar> QFunctionTrait<T> for MassApply {
    fn gallery_name(&self) -> Option<&str> {
        Some("MassApply")
    }

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
                "MassApply expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let qdata = inputs[1];
        let v = &mut outputs[0];
        for i in 0..q {
            v[i] = u[i] * qdata[i];
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
                "MassApply transpose expects 1 output cotangent and 2 input cotangent buffers"
                    .into(),
            ));
        }
        let dv = output_cotangents[0];
        if dv.len() != q {
            return Err(ReedError::QFunction(
                "MassApply transpose: buffer length mismatch".into(),
            ));
        }
        let (du_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let du = &mut du_buf[0];
        let qdata: &[T] = &qdata_fwd[0];
        if du.len() != q || qdata.len() != q {
            return Err(ReedError::QFunction(
                "MassApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            du[i] += dv[i] * qdata[i];
        }
        Ok(())
    }
}

/// `"MassApplyInterpTimesWeight"` — same quadrature kernel as [`MassApply`], but the second input is
/// declared as [`EvalMode::Weight`] so operator assembly exercises the basis **Weight** slot (forward
/// values are still the quadrature weights from the basis; discrete transpose matches [`MassApply`]).
/// Alias: **`MassApplyInterpTimesWeightAtPoints`** (same implementation; libCEED-style AtPoints naming).
pub struct MassApplyInterpTimesWeight {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for MassApplyInterpTimesWeight {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "u".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Interp,
                },
                QFunctionField {
                    name: "w".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Weight,
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

impl<T: Scalar> QFunctionTrait<T> for MassApplyInterpTimesWeight {
    fn gallery_name(&self) -> Option<&str> {
        Some("MassApplyInterpTimesWeight")
    }

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
                "MassApplyInterpTimesWeight expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let w = inputs[1];
        let v = &mut outputs[0];
        for i in 0..q {
            v[i] = u[i] * w[i];
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
                "MassApplyInterpTimesWeight transpose expects 1 output cotangent and 2 input cotangent buffers"
                    .into(),
            ));
        }
        let dv = output_cotangents[0];
        if dv.len() != q {
            return Err(ReedError::QFunction(
                "MassApplyInterpTimesWeight transpose: buffer length mismatch".into(),
            ));
        }
        let (du_buf, w_fwd) = input_cotangents.split_at_mut(1);
        let du = &mut du_buf[0];
        let w: &[T] = &w_fwd[0];
        if du.len() != q || w.len() != q {
            return Err(ReedError::QFunction(
                "MassApplyInterpTimesWeight transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            du[i] += dv[i] * w[i];
        }
        Ok(())
    }
}
