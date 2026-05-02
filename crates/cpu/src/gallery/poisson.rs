use super::helpers::singular_jacobian_tol;
use reed_core::{
    enums::EvalMode,
    error::ReedResult,
    qfunction::{QFunctionField, QFunctionTrait},
    scalar::Scalar,
    ReedError,
};

/// `"Poisson1DBuild"` — geometric factor for the 1D Poisson stiffness (`qdata[i] = weights[i] / dx[i]`),
/// matching libCEED `ceed-poisson1dbuild.h` (`J` is the 1×1 Jacobian from `CEED_EVAL_GRAD`).
pub struct Poisson1DBuild {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for Poisson1DBuild {
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

impl<T: Scalar> QFunctionTrait<T> for Poisson1DBuild {
    fn gallery_name(&self) -> Option<&str> {
        Some("Poisson1DBuild")
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
                "Poisson1DBuild expects 2 inputs and 1 output".into(),
            ));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        if dx.len() != q || weights.len() != q || qdata.len() != q {
            return Err(ReedError::QFunction(
                "Poisson1DBuild: buffer length mismatch".into(),
            ));
        }
        let tol: T = singular_jacobian_tol();
        for i in 0..q {
            let j = dx[i];
            if j.abs() < tol {
                return Err(ReedError::QFunction(
                    "Poisson1DBuild encountered near-singular Jacobian".into(),
                ));
            }
            qdata[i] = weights[i] / j;
        }
        Ok(())
    }
}

pub struct Poisson1DApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

pub struct Poisson2DBuild {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

pub struct Poisson3DBuild {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for Poisson2DBuild {
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
                num_comp: 4,
                eval_mode: EvalMode::None,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for Poisson2DBuild {
    fn gallery_name(&self) -> Option<&str> {
        Some("Poisson2DBuild")
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
                "Poisson2DBuild expects 2 inputs and 1 output".into(),
            ));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        let tol: T = singular_jacobian_tol();
        for i in 0..q {
            let j00 = dx[i * 4];
            let j01 = dx[i * 4 + 1];
            let j10 = dx[i * 4 + 2];
            let j11 = dx[i * 4 + 3];
            let det_j = j00 * j11 - j01 * j10;
            if det_j.abs() < tol {
                return Err(ReedError::QFunction(
                    "Poisson2DBuild encountered near-singular Jacobian".into(),
                ));
            }
            let inv00 = j11 / det_j;
            let inv01 = -j01 / det_j;
            let inv10 = -j10 / det_j;
            let inv11 = j00 / det_j;

            // G = |detJ| * J^{-1} * J^{-T}
            let scale = det_j.abs() * weights[i];
            let g00 = scale * (inv00 * inv00 + inv01 * inv01);
            let g01 = scale * (inv00 * inv10 + inv01 * inv11);
            let g10 = scale * (inv10 * inv00 + inv11 * inv01);
            let g11 = scale * (inv10 * inv10 + inv11 * inv11);

            qdata[i * 4] = g00;
            qdata[i * 4 + 1] = g01;
            qdata[i * 4 + 2] = g10;
            qdata[i * 4 + 3] = g11;
        }
        Ok(())
    }
}

pub struct Poisson2DApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

pub struct Poisson3DApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for Poisson2DApply {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "du".into(),
                    num_comp: 2,
                    eval_mode: EvalMode::Grad,
                },
                QFunctionField {
                    name: "qdata".into(),
                    num_comp: 4,
                    eval_mode: EvalMode::None,
                },
            ],
            outputs: vec![QFunctionField {
                name: "dv".into(),
                num_comp: 2,
                eval_mode: EvalMode::Grad,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for Poisson2DApply {
    fn gallery_name(&self) -> Option<&str> {
        Some("Poisson2DApply")
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
                "Poisson2DApply expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        for i in 0..q {
            let du0 = du[i * 2];
            let du1 = du[i * 2 + 1];
            let g00 = qdata[i * 4];
            let g01 = qdata[i * 4 + 1];
            let g10 = qdata[i * 4 + 2];
            let g11 = qdata[i * 4 + 3];
            dv[i * 2] = g00 * du0 + g01 * du1;
            dv[i * 2 + 1] = g10 * du0 + g11 * du1;
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
                "Poisson2DApply transpose expects 1 output cotangent and 2 input buffers".into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q * 2 {
            return Err(ReedError::QFunction(
                "Poisson2DApply transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[T] = &qdata_fwd[0];
        if ddu.len() != q * 2 || qdata.len() != q * 4 {
            return Err(ReedError::QFunction(
                "Poisson2DApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            let ddv0 = ddv[i * 2];
            let ddv1 = ddv[i * 2 + 1];
            let g00 = qdata[i * 4];
            let g01 = qdata[i * 4 + 1];
            let g10 = qdata[i * 4 + 2];
            let g11 = qdata[i * 4 + 3];
            ddu[i * 2] += g00 * ddv0 + g10 * ddv1;
            ddu[i * 2 + 1] += g01 * ddv0 + g11 * ddv1;
        }
        Ok(())
    }
}

impl Default for Poisson3DBuild {
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
                num_comp: 9,
                eval_mode: EvalMode::None,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for Poisson3DBuild {
    fn gallery_name(&self) -> Option<&str> {
        Some("Poisson3DBuild")
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
                "Poisson3DBuild expects 2 inputs and 1 output".into(),
            ));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        let tol: T = singular_jacobian_tol();
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

            let c00 = j11 * j22 - j12 * j21;
            let c01 = -(j10 * j22 - j12 * j20);
            let c02 = j10 * j21 - j11 * j20;
            let c10 = -(j01 * j22 - j02 * j21);
            let c11 = j00 * j22 - j02 * j20;
            let c12 = -(j00 * j21 - j01 * j20);
            let c20 = j01 * j12 - j02 * j11;
            let c21 = -(j00 * j12 - j02 * j10);
            let c22 = j00 * j11 - j01 * j10;

            let det_j = j00 * c00 + j01 * c01 + j02 * c02;
            if det_j.abs() < tol {
                return Err(ReedError::QFunction(
                    "Poisson3DBuild encountered near-singular Jacobian".into(),
                ));
            }

            let inv00 = c00 / det_j;
            let inv01 = c10 / det_j;
            let inv02 = c20 / det_j;
            let inv10 = c01 / det_j;
            let inv11 = c11 / det_j;
            let inv12 = c21 / det_j;
            let inv20 = c02 / det_j;
            let inv21 = c12 / det_j;
            let inv22 = c22 / det_j;

            let s = det_j.abs() * weights[i];
            qdata[i * 9] = s * (inv00 * inv00 + inv01 * inv01 + inv02 * inv02);
            qdata[i * 9 + 1] = s * (inv00 * inv10 + inv01 * inv11 + inv02 * inv12);
            qdata[i * 9 + 2] = s * (inv00 * inv20 + inv01 * inv21 + inv02 * inv22);
            qdata[i * 9 + 3] = s * (inv10 * inv00 + inv11 * inv01 + inv12 * inv02);
            qdata[i * 9 + 4] = s * (inv10 * inv10 + inv11 * inv11 + inv12 * inv12);
            qdata[i * 9 + 5] = s * (inv10 * inv20 + inv11 * inv21 + inv12 * inv22);
            qdata[i * 9 + 6] = s * (inv20 * inv00 + inv21 * inv01 + inv22 * inv02);
            qdata[i * 9 + 7] = s * (inv20 * inv10 + inv21 * inv11 + inv22 * inv12);
            qdata[i * 9 + 8] = s * (inv20 * inv20 + inv21 * inv21 + inv22 * inv22);
        }
        Ok(())
    }
}

impl Default for Poisson3DApply {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "du".into(),
                    num_comp: 3,
                    eval_mode: EvalMode::Grad,
                },
                QFunctionField {
                    name: "qdata".into(),
                    num_comp: 9,
                    eval_mode: EvalMode::None,
                },
            ],
            outputs: vec![QFunctionField {
                name: "dv".into(),
                num_comp: 3,
                eval_mode: EvalMode::Grad,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for Poisson3DApply {
    fn gallery_name(&self) -> Option<&str> {
        Some("Poisson3DApply")
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
                "Poisson3DApply expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        for i in 0..q {
            let du0 = du[i * 3];
            let du1 = du[i * 3 + 1];
            let du2 = du[i * 3 + 2];
            let g00 = qdata[i * 9];
            let g01 = qdata[i * 9 + 1];
            let g02 = qdata[i * 9 + 2];
            let g10 = qdata[i * 9 + 3];
            let g11 = qdata[i * 9 + 4];
            let g12 = qdata[i * 9 + 5];
            let g20 = qdata[i * 9 + 6];
            let g21 = qdata[i * 9 + 7];
            let g22 = qdata[i * 9 + 8];

            dv[i * 3] = g00 * du0 + g01 * du1 + g02 * du2;
            dv[i * 3 + 1] = g10 * du0 + g11 * du1 + g12 * du2;
            dv[i * 3 + 2] = g20 * du0 + g21 * du1 + g22 * du2;
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
                "Poisson3DApply transpose expects 1 output cotangent and 2 input buffers".into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q * 3 {
            return Err(ReedError::QFunction(
                "Poisson3DApply transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[T] = &qdata_fwd[0];
        if ddu.len() != q * 3 || qdata.len() != q * 9 {
            return Err(ReedError::QFunction(
                "Poisson3DApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            let ddv0 = ddv[i * 3];
            let ddv1 = ddv[i * 3 + 1];
            let ddv2 = ddv[i * 3 + 2];
            let g00 = qdata[i * 9];
            let g01 = qdata[i * 9 + 1];
            let g02 = qdata[i * 9 + 2];
            let g10 = qdata[i * 9 + 3];
            let g11 = qdata[i * 9 + 4];
            let g12 = qdata[i * 9 + 5];
            let g20 = qdata[i * 9 + 6];
            let g21 = qdata[i * 9 + 7];
            let g22 = qdata[i * 9 + 8];
            ddu[i * 3] += g00 * ddv0 + g10 * ddv1 + g20 * ddv2;
            ddu[i * 3 + 1] += g01 * ddv0 + g11 * ddv1 + g21 * ddv2;
            ddu[i * 3 + 2] += g02 * ddv0 + g12 * ddv1 + g22 * ddv2;
        }
        Ok(())
    }
}

impl Default for Poisson1DApply {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "du".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::Grad,
                },
                QFunctionField {
                    name: "qdata".into(),
                    num_comp: 1,
                    eval_mode: EvalMode::None,
                },
            ],
            outputs: vec![QFunctionField {
                name: "dv".into(),
                num_comp: 1,
                eval_mode: EvalMode::Grad,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for Poisson1DApply {
    fn gallery_name(&self) -> Option<&str> {
        Some("Poisson1DApply")
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
                "Poisson1DApply expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        for i in 0..q {
            dv[i] = du[i] * qdata[i];
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
                "Poisson1DApply transpose expects 1 output cotangent and 2 input cotangent buffers"
                    .into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q {
            return Err(ReedError::QFunction(
                "Poisson1DApply transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[T] = &qdata_fwd[0];
        if ddu.len() != q || qdata.len() != q {
            return Err(ReedError::QFunction(
                "Poisson1DApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            ddu[i] += ddv[i] * qdata[i];
        }
        Ok(())
    }
}

#[cfg(test)]
mod poisson1d_build_tests {
    use super::*;
    use reed_core::qfunction::QFunctionTrait;

    #[test]
    fn poisson1d_build_matches_libceed() {
        let b = Poisson1DBuild::default();
        let dx = vec![2.0, 2.0];
        let w = vec![0.5, 0.5];
        let mut qdata = vec![0.0; 2];
        b.apply(&[], 2, &[dx.as_slice(), w.as_slice()], &mut [&mut qdata])
            .unwrap();
        assert_eq!(qdata, vec![0.25, 0.25]);
    }

    #[test]
    fn poisson1d_build_f32() {
        let b = Poisson1DBuild::default();
        let dx = vec![2.0f32, 2.0];
        let w = vec![0.5f32, 0.5];
        let mut qdata = vec![0.0f32; 2];
        b.apply(&[], 2, &[dx.as_slice(), w.as_slice()], &mut [&mut qdata])
            .unwrap();
        assert!((qdata[0] - 0.25).abs() < 1e-5);
        assert!((qdata[1] - 0.25).abs() < 1e-5);
    }
}
