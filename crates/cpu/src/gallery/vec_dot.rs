use reed_core::{
    enums::EvalMode,
    error::ReedResult,
    qfunction::{QFunctionField, QFunctionTrait},
    scalar::Scalar,
    ReedError,
};

/// Dot product of two vector fields at quadrature points (2D).
pub struct Vec2Dot {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Vec2Dot {
    pub fn new() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "u".into(),
                    num_comp: 2,
                    eval_mode: EvalMode::Interp,
                },
                QFunctionField {
                    name: "v".into(),
                    num_comp: 2,
                    eval_mode: EvalMode::Interp,
                },
            ],
            outputs: vec![QFunctionField {
                name: "w".into(),
                num_comp: 1,
                eval_mode: EvalMode::None,
            }],
        }
    }
}

impl Default for Vec2Dot {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> QFunctionTrait<T> for Vec2Dot {
    fn gallery_name(&self) -> Option<&str> {
        Some("Vec2Dot")
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
                "Vec2Dot expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let v = inputs[1];
        let w = &mut outputs[0];
        for i in 0..q {
            w[i] = u[i * 2] * v[i * 2] + u[i * 2 + 1] * v[i * 2 + 1];
        }
        Ok(())
    }
}

/// Dot product of two vector fields at quadrature points (3D).
pub struct Vec3Dot {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Vec3Dot {
    pub fn new() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "u".into(),
                    num_comp: 3,
                    eval_mode: EvalMode::Interp,
                },
                QFunctionField {
                    name: "v".into(),
                    num_comp: 3,
                    eval_mode: EvalMode::Interp,
                },
            ],
            outputs: vec![QFunctionField {
                name: "w".into(),
                num_comp: 1,
                eval_mode: EvalMode::None,
            }],
        }
    }
}

impl Default for Vec3Dot {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> QFunctionTrait<T> for Vec3Dot {
    fn gallery_name(&self) -> Option<&str> {
        Some("Vec3Dot")
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
                "Vec3Dot expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let v = inputs[1];
        let w = &mut outputs[0];
        for i in 0..q {
            w[i] = u[i * 3] * v[i * 3] + u[i * 3 + 1] * v[i * 3 + 1] + u[i * 3 + 2] * v[i * 3 + 2];
        }
        Ok(())
    }
}
