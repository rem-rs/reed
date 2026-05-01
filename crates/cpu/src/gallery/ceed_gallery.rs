//! libCEED-compatible gallery names and semantics (see libCEED `gallery/ceed-gallery-list.h`).
//!
//! Reed field layouts follow the same interleaved quadrature indexing as other gallery QFs.

use super::helpers::scale_alpha_from_libceed_context;
use reed_core::{
    enums::EvalMode,
    error::ReedResult,
    qfunction::{QFunctionField, QFunctionTrait},
    scalar::Scalar,
    ReedError,
};

// --- Identity -------------------------------------------------------------

/// `"Identity"` — copy input quadrature values to output (same `num_comp` each side).
#[derive(Clone)]
pub struct Identity {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Identity {
    pub fn with_components(ncomp: usize) -> Self {
        Self {
            inputs: vec![QFunctionField {
                name: "input".into(),
                num_comp: ncomp,
                eval_mode: EvalMode::Interp,
            }],
            outputs: vec![QFunctionField {
                name: "output".into(),
                num_comp: ncomp,
                eval_mode: EvalMode::Interp,
            }],
        }
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self::with_components(1)
    }
}

impl<T: Scalar> QFunctionTrait<T> for Identity {
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
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Identity expects 1 input and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let v = &mut outputs[0];
        if u.len() != v.len() {
            return Err(ReedError::QFunction(
                "Identity: input/output length mismatch".into(),
            ));
        }
        let n = q * self.inputs[0].num_comp;
        if u.len() != n {
            return Err(ReedError::QFunction(
                "Identity: unexpected buffer length".into(),
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
                "Identity transpose expects 1 output cotangent and 1 input cotangent buffer".into(),
            ));
        }
        let dv = output_cotangents[0];
        let du = &mut input_cotangents[0];
        let n = q * self.inputs[0].num_comp;
        if dv.len() != n || du.len() != n {
            return Err(ReedError::QFunction(
                "Identity transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..n {
            du[i] += dv[i];
        }
        Ok(())
    }
}

/// `"Identity to scalar"` — keep the first component per quadrature point (`out[i] = in[i * ncomp]`).
#[derive(Clone)]
pub struct IdentityScalar {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl IdentityScalar {
    pub fn with_input_components(ncomp: usize) -> Self {
        Self {
            inputs: vec![QFunctionField {
                name: "input".into(),
                num_comp: ncomp,
                eval_mode: EvalMode::Interp,
            }],
            outputs: vec![QFunctionField {
                name: "output".into(),
                num_comp: 1,
                eval_mode: EvalMode::Interp,
            }],
        }
    }
}

impl Default for IdentityScalar {
    fn default() -> Self {
        Self::with_input_components(3)
    }
}

impl<T: Scalar> QFunctionTrait<T> for IdentityScalar {
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
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "IdentityScalar expects 1 input and 1 output".into(),
            ));
        }
        let ncomp = self.inputs[0].num_comp;
        let u = inputs[0];
        let v = &mut outputs[0];
        if u.len() != q * ncomp || v.len() != q {
            return Err(ReedError::QFunction(
                "IdentityScalar: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            v[i] = u[i * ncomp];
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
        if output_cotangents.len() != 1 || input_cotangents.len() != 1 {
            return Err(ReedError::QFunction(
                "IdentityScalar transpose expects 1 output cotangent and 1 input buffer".into(),
            ));
        }
        let ncomp = self.inputs[0].num_comp;
        let dv = output_cotangents[0];
        let du = &mut input_cotangents[0];
        if dv.len() != q || du.len() != q * ncomp {
            return Err(ReedError::QFunction(
                "IdentityScalar transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            du[i * ncomp] += dv[i];
        }
        Ok(())
    }
}

// --- Scale ----------------------------------------------------------------

/// `"Scale"` — multiply every input value by `alpha` from context (`f64` LE, 8 bytes; cast to `T`).
#[derive(Clone)]
pub struct Scale {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Scale {
    pub fn with_components(ncomp: usize) -> Self {
        Self {
            inputs: vec![QFunctionField {
                name: "input".into(),
                num_comp: ncomp,
                eval_mode: EvalMode::Interp,
            }],
            outputs: vec![QFunctionField {
                name: "output".into(),
                num_comp: ncomp,
                eval_mode: EvalMode::Interp,
            }],
        }
    }
}

impl Default for Scale {
    fn default() -> Self {
        Self::with_components(1)
    }
}

impl<T: Scalar> QFunctionTrait<T> for Scale {
    fn context_byte_len(&self) -> usize {
        8
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
        let alpha = scale_alpha_from_libceed_context::<T>(ctx)?;
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Scale expects 1 input and 1 output".into(),
            ));
        }
        let ncomp = self.inputs[0].num_comp;
        let u = inputs[0];
        let v = &mut outputs[0];
        if u.len() != q * ncomp || v.len() != q * ncomp {
            return Err(ReedError::QFunction("Scale: buffer length mismatch".into()));
        }
        for i in 0..u.len() {
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
                "Scale transpose expects 1 output cotangent and 1 input cotangent buffer".into(),
            ));
        }
        let dv = output_cotangents[0];
        let du = &mut input_cotangents[0];
        let ncomp = self.inputs[0].num_comp;
        if dv.len() != q * ncomp || du.len() != q * ncomp {
            return Err(ReedError::QFunction(
                "Scale transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..du.len() {
            du[i] += alpha * dv[i];
        }
        Ok(())
    }
}

/// `"Scale (scalar)"` — same kernel as [`Scale`]; separate gallery name for libCEED parity.
#[derive(Clone)]
pub struct ScaleScalar {
    inner: Scale,
}

impl Default for ScaleScalar {
    fn default() -> Self {
        Self {
            inner: Scale::default(),
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for ScaleScalar {
    fn context_byte_len(&self) -> usize {
        <Scale as QFunctionTrait<T>>::context_byte_len(&self.inner)
    }

    fn inputs(&self) -> &[QFunctionField] {
        <Scale as QFunctionTrait<T>>::inputs(&self.inner)
    }

    fn outputs(&self) -> &[QFunctionField] {
        <Scale as QFunctionTrait<T>>::outputs(&self.inner)
    }

    fn apply(
        &self,
        ctx: &[u8],
        q: usize,
        inputs: &[&[T]],
        outputs: &mut [&mut [T]],
    ) -> ReedResult<()> {
        <Scale as QFunctionTrait<T>>::apply(&self.inner, ctx, q, inputs, outputs)
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
        <Scale as QFunctionTrait<T>>::apply_operator_transpose(
            &self.inner,
            ctx,
            q,
            output_cotangents,
            input_cotangents,
        )
    }
}

// --- Vector2 mass / Poisson (Reed extension; 2D vector fields) ------------

/// `"Vector2MassApply"` — `v[k] = qdata * u[k]` with two vector components per quadrature point.
#[derive(Clone, Default)]
pub struct Vector2MassApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Vector2MassApply {
    pub fn new() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "u".into(),
                    num_comp: 2,
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
                num_comp: 2,
                eval_mode: EvalMode::Interp,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for Vector2MassApply {
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
                "Vector2MassApply expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let qdata = inputs[1];
        let v = &mut outputs[0];
        if qdata.len() != q || u.len() != q * 2 || v.len() != q * 2 {
            return Err(ReedError::QFunction(
                "Vector2MassApply: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            let s = qdata[i];
            v[i * 2] = s * u[i * 2];
            v[i * 2 + 1] = s * u[i * 2 + 1];
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
                "Vector2MassApply transpose expects 1 output cotangent and 2 input buffers".into(),
            ));
        }
        let dv = output_cotangents[0];
        if dv.len() != q * 2 {
            return Err(ReedError::QFunction(
                "Vector2MassApply transpose: buffer length mismatch".into(),
            ));
        }
        let (du_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let du = &mut du_buf[0];
        let qdata: &[T] = &qdata_fwd[0];
        if du.len() != q * 2 || qdata.len() != q {
            return Err(ReedError::QFunction(
                "Vector2MassApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            let s = qdata[i];
            du[i * 2] += s * dv[i * 2];
            du[i * 2 + 1] += s * dv[i * 2 + 1];
        }
        Ok(())
    }
}

/// `"Vector2Poisson1DApply"` — two independent scalar 1D Poisson gradient applies (same `qdata` as `Poisson1DApply`).
#[derive(Clone, Default)]
pub struct Vector2Poisson1DApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Vector2Poisson1DApply {
    pub fn new() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "du".into(),
                    num_comp: 2,
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
                num_comp: 2,
                eval_mode: EvalMode::Grad,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for Vector2Poisson1DApply {
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
                "Vector2Poisson1DApply expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        if qdata.len() != q || du.len() != q * 2 || dv.len() != q * 2 {
            return Err(ReedError::QFunction(
                "Vector2Poisson1DApply: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            let g = qdata[i];
            dv[i * 2] = g * du[i * 2];
            dv[i * 2 + 1] = g * du[i * 2 + 1];
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
                "Vector2Poisson1DApply transpose expects 1 output cotangent and 2 input buffers"
                    .into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q * 2 {
            return Err(ReedError::QFunction(
                "Vector2Poisson1DApply transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[T] = &qdata_fwd[0];
        if ddu.len() != q * 2 || qdata.len() != q {
            return Err(ReedError::QFunction(
                "Vector2Poisson1DApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            let g = qdata[i];
            ddu[i * 2] += g * ddv[i * 2];
            ddu[i * 2 + 1] += g * ddv[i * 2 + 1];
        }
        Ok(())
    }
}

/// `"Vector2Poisson2DApply"` — two stacked 2D Poisson gradient applies (`qdata` layout matches `Poisson2DApply`).
#[derive(Clone)]
pub struct Vector2Poisson2DApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Vector2Poisson2DApply {
    pub fn new() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "du".into(),
                    num_comp: 4,
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
                num_comp: 4,
                eval_mode: EvalMode::Grad,
            }],
        }
    }
}

impl Default for Vector2Poisson2DApply {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> QFunctionTrait<T> for Vector2Poisson2DApply {
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
                "Vector2Poisson2DApply expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        if qdata.len() != q * 4 || du.len() != q * 4 || dv.len() != q * 4 {
            return Err(ReedError::QFunction(
                "Vector2Poisson2DApply: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            for c in 0..2 {
                let base = c * 2;
                let du0 = du[i * 4 + base];
                let du1 = du[i * 4 + base + 1];
                let g00 = qdata[i * 4];
                let g01 = qdata[i * 4 + 1];
                let g10 = qdata[i * 4 + 2];
                let g11 = qdata[i * 4 + 3];
                dv[i * 4 + base] = g00 * du0 + g01 * du1;
                dv[i * 4 + base + 1] = g10 * du0 + g11 * du1;
            }
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
                "Vector2Poisson2DApply transpose expects 1 output cotangent and 2 input buffers"
                    .into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q * 4 {
            return Err(ReedError::QFunction(
                "Vector2Poisson2DApply transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[T] = &qdata_fwd[0];
        if ddu.len() != q * 4 || qdata.len() != q * 4 {
            return Err(ReedError::QFunction(
                "Vector2Poisson2DApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            for c in 0..2 {
                let base = c * 2;
                let ddv0 = ddv[i * 4 + base];
                let ddv1 = ddv[i * 4 + base + 1];
                let g00 = qdata[i * 4];
                let g01 = qdata[i * 4 + 1];
                let g10 = qdata[i * 4 + 2];
                let g11 = qdata[i * 4 + 3];
                ddu[i * 4 + base] += g00 * ddv0 + g10 * ddv1;
                ddu[i * 4 + base + 1] += g01 * ddv0 + g11 * ddv1;
            }
        }
        Ok(())
    }
}

// --- Vector3 mass / Poisson -----------------------------------------------

/// `"Vector3MassApply"` — `v[k] = qdata * u[k]` with three vector components at each quadrature point.
#[derive(Clone, Default)]
pub struct Vector3MassApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Vector3MassApply {
    pub fn new() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "u".into(),
                    num_comp: 3,
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
                num_comp: 3,
                eval_mode: EvalMode::Interp,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for Vector3MassApply {
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
                "Vector3MassApply expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let qdata = inputs[1];
        let v = &mut outputs[0];
        if qdata.len() != q || u.len() != q * 3 || v.len() != q * 3 {
            return Err(ReedError::QFunction(
                "Vector3MassApply: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            let s = qdata[i];
            v[i * 3] = s * u[i * 3];
            v[i * 3 + 1] = s * u[i * 3 + 1];
            v[i * 3 + 2] = s * u[i * 3 + 2];
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
                "Vector3MassApply transpose expects 1 output cotangent and 2 input buffers".into(),
            ));
        }
        let dv = output_cotangents[0];
        if dv.len() != q * 3 {
            return Err(ReedError::QFunction(
                "Vector3MassApply transpose: buffer length mismatch".into(),
            ));
        }
        let (du_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let du = &mut du_buf[0];
        let qdata: &[T] = &qdata_fwd[0];
        if du.len() != q * 3 || qdata.len() != q {
            return Err(ReedError::QFunction(
                "Vector3MassApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            let s = qdata[i];
            du[i * 3] += s * dv[i * 3];
            du[i * 3 + 1] += s * dv[i * 3 + 1];
            du[i * 3 + 2] += s * dv[i * 3 + 2];
        }
        Ok(())
    }
}

/// `"Vector3Poisson1DApply"` — three independent scalar 1D Laplacian applies (same `qdata` as Poisson1D).
#[derive(Clone, Default)]
pub struct Vector3Poisson1DApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Vector3Poisson1DApply {
    pub fn new() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "du".into(),
                    num_comp: 3,
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
                num_comp: 3,
                eval_mode: EvalMode::Grad,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for Vector3Poisson1DApply {
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
                "Vector3Poisson1DApply expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        if qdata.len() != q || du.len() != q * 3 || dv.len() != q * 3 {
            return Err(ReedError::QFunction(
                "Vector3Poisson1DApply: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            let g = qdata[i];
            dv[i * 3] = g * du[i * 3];
            dv[i * 3 + 1] = g * du[i * 3 + 1];
            dv[i * 3 + 2] = g * du[i * 3 + 2];
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
                "Vector3Poisson1DApply transpose expects 1 output cotangent and 2 input buffers"
                    .into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q * 3 {
            return Err(ReedError::QFunction(
                "Vector3Poisson1DApply transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[T] = &qdata_fwd[0];
        if ddu.len() != q * 3 || qdata.len() != q {
            return Err(ReedError::QFunction(
                "Vector3Poisson1DApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            let g = qdata[i];
            ddu[i * 3] += g * ddv[i * 3];
            ddu[i * 3 + 1] += g * ddv[i * 3 + 1];
            ddu[i * 3 + 2] += g * ddv[i * 3 + 2];
        }
        Ok(())
    }
}

/// `"Vector3Poisson2DApply"` — three stacked 2D Poisson gradient applies.
///
/// `qdata` uses the same **4** stiffness entries per point as `Poisson2DApply` / `Poisson2DBuild`.
/// libCEED registers 3 symmetric components; Reed keeps 4 to match the existing scalar Poisson pipeline.
#[derive(Clone)]
pub struct Vector3Poisson2DApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Vector3Poisson2DApply {
    pub fn new() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "du".into(),
                    num_comp: 6,
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
                num_comp: 6,
                eval_mode: EvalMode::Grad,
            }],
        }
    }
}

impl Default for Vector3Poisson2DApply {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> QFunctionTrait<T> for Vector3Poisson2DApply {
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
                "Vector3Poisson2DApply expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        if qdata.len() != q * 4 || du.len() != q * 6 || dv.len() != q * 6 {
            return Err(ReedError::QFunction(
                "Vector3Poisson2DApply: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            for c in 0..3 {
                let base = c * 2;
                let du0 = du[i * 6 + base];
                let du1 = du[i * 6 + base + 1];
                let g00 = qdata[i * 4];
                let g01 = qdata[i * 4 + 1];
                let g10 = qdata[i * 4 + 2];
                let g11 = qdata[i * 4 + 3];
                dv[i * 6 + base] = g00 * du0 + g01 * du1;
                dv[i * 6 + base + 1] = g10 * du0 + g11 * du1;
            }
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
                "Vector3Poisson2DApply transpose expects 1 output cotangent and 2 input buffers"
                    .into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q * 6 {
            return Err(ReedError::QFunction(
                "Vector3Poisson2DApply transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[T] = &qdata_fwd[0];
        if ddu.len() != q * 6 || qdata.len() != q * 4 {
            return Err(ReedError::QFunction(
                "Vector3Poisson2DApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            for c in 0..3 {
                let base = c * 2;
                let ddv0 = ddv[i * 6 + base];
                let ddv1 = ddv[i * 6 + base + 1];
                let g00 = qdata[i * 4];
                let g01 = qdata[i * 4 + 1];
                let g10 = qdata[i * 4 + 2];
                let g11 = qdata[i * 4 + 3];
                ddu[i * 6 + base] += g00 * ddv0 + g10 * ddv1;
                ddu[i * 6 + base + 1] += g01 * ddv0 + g11 * ddv1;
            }
        }
        Ok(())
    }
}

/// `"Vector3Poisson3DApply"` — three stacked 3D Poisson gradient applies (`qdata` 9 components, same as `Poisson3DApply`).
#[derive(Clone, Default)]
pub struct Vector3Poisson3DApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Vector3Poisson3DApply {
    pub fn new() -> Self {
        Self {
            inputs: vec![
                QFunctionField {
                    name: "du".into(),
                    num_comp: 9,
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
                num_comp: 9,
                eval_mode: EvalMode::Grad,
            }],
        }
    }
}

impl<T: Scalar> QFunctionTrait<T> for Vector3Poisson3DApply {
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
                "Vector3Poisson3DApply expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        if qdata.len() != q * 9 || du.len() != q * 9 || dv.len() != q * 9 {
            return Err(ReedError::QFunction(
                "Vector3Poisson3DApply: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            for c in 0..3 {
                let base = c * 3;
                let du0 = du[i * 9 + base];
                let du1 = du[i * 9 + base + 1];
                let du2 = du[i * 9 + base + 2];
                let g00 = qdata[i * 9];
                let g01 = qdata[i * 9 + 1];
                let g02 = qdata[i * 9 + 2];
                let g10 = qdata[i * 9 + 3];
                let g11 = qdata[i * 9 + 4];
                let g12 = qdata[i * 9 + 5];
                let g20 = qdata[i * 9 + 6];
                let g21 = qdata[i * 9 + 7];
                let g22 = qdata[i * 9 + 8];
                dv[i * 9 + base] = g00 * du0 + g01 * du1 + g02 * du2;
                dv[i * 9 + base + 1] = g10 * du0 + g11 * du1 + g12 * du2;
                dv[i * 9 + base + 2] = g20 * du0 + g21 * du1 + g22 * du2;
            }
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
                "Vector3Poisson3DApply transpose expects 1 output cotangent and 2 input buffers"
                    .into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q * 9 {
            return Err(ReedError::QFunction(
                "Vector3Poisson3DApply transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[T] = &qdata_fwd[0];
        if ddu.len() != q * 9 || qdata.len() != q * 9 {
            return Err(ReedError::QFunction(
                "Vector3Poisson3DApply transpose: buffer length mismatch".into(),
            ));
        }
        for i in 0..q {
            for c in 0..3 {
                let base = c * 3;
                let ddv0 = ddv[i * 9 + base];
                let ddv1 = ddv[i * 9 + base + 1];
                let ddv2 = ddv[i * 9 + base + 2];
                let g00 = qdata[i * 9];
                let g01 = qdata[i * 9 + 1];
                let g02 = qdata[i * 9 + 2];
                let g10 = qdata[i * 9 + 3];
                let g11 = qdata[i * 9 + 4];
                let g12 = qdata[i * 9 + 5];
                let g20 = qdata[i * 9 + 6];
                let g21 = qdata[i * 9 + 7];
                let g22 = qdata[i * 9 + 8];
                ddu[i * 9 + base] += g00 * ddv0 + g10 * ddv1 + g20 * ddv2;
                ddu[i * 9 + base + 1] += g01 * ddv0 + g11 * ddv1 + g21 * ddv2;
                ddu[i * 9 + base + 2] += g02 * ddv0 + g12 * ddv1 + g22 * ddv2;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reed_core::qfunction::QFunctionTrait;

    #[test]
    fn identity_copies() {
        let id = Identity::default();
        let u = vec![1.0, 2.0, 3.0];
        let mut v = vec![0.0; 3];
        id.apply(&[], 3, &[u.as_slice()], &mut [&mut v]).unwrap();
        assert_eq!(v, u);
    }

    #[test]
    fn identity_scalar_first_component() {
        let id = IdentityScalar::default();
        // Two quadrature points, 3 components: v[i] = u[i*3].
        let u = vec![1.0, 10.0, 11.0, 2.0, 20.0, 21.0];
        let mut v = vec![0.0; 2];
        id.apply(&[], 2, &[u.as_slice()], &mut [&mut v]).unwrap();
        assert_eq!(v, vec![1.0, 2.0]);
    }

    #[test]
    fn scale_uses_context() {
        let sc = Scale::default();
        let mut ctx = [0u8; 8];
        ctx.copy_from_slice(&2.5f64.to_le_bytes());
        let u = vec![1.0, 2.0];
        let mut v = vec![0.0; 2];
        sc.apply(&ctx, 2, &[u.as_slice()], &mut [&mut v]).unwrap();
        assert_eq!(v, vec![2.5, 5.0]);
    }

    #[test]
    fn vector2_mass_apply() {
        let m = Vector2MassApply::new();
        let u = vec![1.0, 2.0, 1.0, 1.0];
        let qdata = vec![2.0, 3.0];
        let mut v = vec![0.0; 4];
        m.apply(&[], 2, &[&u, &qdata], &mut [&mut v]).unwrap();
        assert_eq!(v, vec![2.0, 4.0, 3.0, 3.0]);
    }

    #[test]
    fn vector2_poisson2d_apply() {
        let m = Vector2Poisson2DApply::new();
        // q=1: identity qdata I, du = [1,0, 0,1] -> dv same
        let du = vec![1.0_f64, 0.0, 0.0, 1.0];
        let qdata = vec![1.0, 0.0, 0.0, 1.0];
        let mut dv = vec![0.0_f64; 4];
        m.apply(&[], 1, &[&du, &qdata], &mut [&mut dv]).unwrap();
        assert!((dv[0] - 1.0).abs() < 1e-14);
        assert!((dv[1]).abs() < 1e-14);
        assert!((dv[2]).abs() < 1e-14);
        assert!((dv[3] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn vector3_mass_apply() {
        let m = Vector3MassApply::new();
        let u = vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0];
        let qdata = vec![2.0, 3.0];
        let mut v = vec![0.0; 6];
        m.apply(&[], 2, &[&u, &qdata], &mut [&mut v]).unwrap();
        assert_eq!(v, vec![2.0, 4.0, 6.0, 3.0, 3.0, 3.0]);
    }
}
