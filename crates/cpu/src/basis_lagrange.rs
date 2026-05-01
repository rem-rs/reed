use reed_core::{basis::BasisTrait, enums::EvalMode, error::ReedResult, scalar::Scalar, ReedError};
#[cfg(target_arch = "x86_64")]
use std::any::TypeId;

#[cfg(feature = "parallel")]
const PAR_MIN_ELEMS_PER_TASK: usize = 128;

pub struct LagrangeBasis<T: Scalar> {
    dim: usize,
    ncomp: usize,
    p: usize,
    q: usize,
    num_dof: usize,
    num_qpoints: usize,
    q_ref: Vec<T>,
    weights: Vec<T>,
    weights_1d: Vec<T>,
    interp: Vec<T>,
    grad: Vec<T>,
}

impl<T: Scalar> LagrangeBasis<T> {
    pub fn new(
        dim: usize,
        ncomp: usize,
        p: usize,
        q: usize,
        qmode: reed_core::QuadMode,
    ) -> ReedResult<Self> {
        if !(1..=3).contains(&dim) {
            return Err(ReedError::Basis(format!(
                "current CPU basis supports dim in 1..=3, got {}",
                dim
            )));
        }
        if p < 2 {
            return Err(ReedError::Basis(format!("p must be >= 2, got {}", p)));
        }
        if q < 1 {
            return Err(ReedError::Basis(format!("q must be >= 1, got {}", q)));
        }

        let nodes = gauss_lobatto_nodes(p)?;
        let (q_ref_f64, weights_f64) = match qmode {
            reed_core::QuadMode::Gauss => gauss_quadrature(q)?,
            reed_core::QuadMode::GaussLobatto => gauss_lobatto_quadrature(q)?,
        };
        let num_dof = p.pow(dim as u32);
        let num_qpoints = q.pow(dim as u32);
        let weights_1d: Vec<T> = weights_f64
            .iter()
            .map(|&x| to_scalar::<T>(x))
            .collect::<ReedResult<Vec<T>>>()?;
        let q_ref_tensor = build_tensor_qref::<T>(&q_ref_f64, dim)?;
        let weights_tensor = build_tensor_weights::<T>(&weights_f64, dim)?;

        let interp = build_interp::<T>(&nodes, &q_ref_f64)?;
        let grad = build_grad::<T>(&nodes, &q_ref_f64)?;

        Ok(Self {
            dim,
            ncomp,
            p,
            q,
            num_dof,
            num_qpoints,
            q_ref: q_ref_tensor,
            weights: weights_tensor,
            weights_1d,
            interp,
            grad,
        })
    }
}

impl<T: Scalar> BasisTrait<T> for LagrangeBasis<T> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn num_dof(&self) -> usize {
        self.num_dof
    }

    fn num_qpoints(&self) -> usize {
        self.num_qpoints
    }

    fn num_comp(&self) -> usize {
        self.ncomp
    }

    fn apply(
        &self,
        num_elem: usize,
        transpose: bool,
        eval_mode: EvalMode,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<()> {
        match eval_mode {
            EvalMode::Interp => {
                let in_size = if transpose {
                    num_elem * self.num_qpoints * self.ncomp
                } else {
                    num_elem * self.num_dof * self.ncomp
                };
                let out_size = if transpose {
                    num_elem * self.num_dof * self.ncomp
                } else {
                    num_elem * self.num_qpoints * self.ncomp
                };
                if u.len() != in_size || v.len() != out_size {
                    return Err(ReedError::Basis(format!(
                        "interp apply size mismatch: input {}, expected {}; output {}, expected {}",
                        u.len(),
                        in_size,
                        v.len(),
                        out_size
                    )));
                }
                let in_stride = in_size / num_elem;
                let out_stride = out_size / num_elem;
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    u.par_chunks(in_stride)
                        .zip(v.par_chunks_mut(out_stride))
                        .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                        .for_each(|(u_elem, v_elem)| {
                            self.apply_interp_elem(transpose, u_elem, v_elem)
                        });
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for (u_elem, v_elem) in u.chunks(in_stride).zip(v.chunks_mut(out_stride)) {
                        self.apply_interp_elem(transpose, u_elem, v_elem);
                    }
                }
            }
            EvalMode::Grad => {
                let qcomp = self.ncomp * self.dim;
                let in_size = if transpose {
                    num_elem * self.num_qpoints * qcomp
                } else {
                    num_elem * self.num_dof * self.ncomp
                };
                let out_size = if transpose {
                    num_elem * self.num_dof * self.ncomp
                } else {
                    num_elem * self.num_qpoints * qcomp
                };
                if u.len() != in_size || v.len() != out_size {
                    return Err(ReedError::Basis(format!(
                        "grad apply size mismatch: input {}, expected {}; output {}, expected {}",
                        u.len(),
                        in_size,
                        v.len(),
                        out_size
                    )));
                }
                let in_stride = in_size / num_elem;
                let out_stride = out_size / num_elem;
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    u.par_chunks(in_stride)
                        .zip(v.par_chunks_mut(out_stride))
                        .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                        .for_each(|(u_elem, v_elem)| {
                            self.apply_grad_elem(transpose, u_elem, v_elem)
                        });
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for (u_elem, v_elem) in u.chunks(in_stride).zip(v.chunks_mut(out_stride)) {
                        self.apply_grad_elem(transpose, u_elem, v_elem);
                    }
                }
            }
            EvalMode::Weight => {
                if transpose {
                    // Quadrature → nodal layout matches scalar `Interp` transpose (libCEED-style
                    // `CEED_EVAL_WEIGHT` adjoint uses the same reference operator as `Interp`).
                    if self.ncomp != 1 {
                        return Err(ReedError::Basis(
                            "EvalMode::Weight transpose requires basis.num_comp() == 1".into(),
                        ));
                    }
                    return self.apply(num_elem, true, EvalMode::Interp, u, v);
                }
                if v.len() != num_elem * self.num_qpoints {
                    return Err(ReedError::Basis(format!(
                        "weight output length {} != expected {}",
                        v.len(),
                        num_elem * self.num_qpoints
                    )));
                }
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    v.par_chunks_mut(self.num_qpoints)
                        .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                        .for_each(|v_elem| v_elem.copy_from_slice(&self.weights));
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for v_elem in v.chunks_mut(self.num_qpoints) {
                        v_elem.copy_from_slice(&self.weights);
                    }
                }
            }
            EvalMode::Div => {
                if self.ncomp != self.dim {
                    return Err(ReedError::Basis(
                        "EvalMode::Div requires ncomp == dim (vector field with one component per spatial axis)"
                            .into(),
                    ));
                }
                let qcomp = self.ncomp * self.dim;
                if transpose {
                    // Adjoint of forward Div = H ∘ Grad: expand scalar w[q] onto diagonal entries of
                    // the per-qpoint gradient stencil, then apply Grad^T (libCEED-consistent).
                    let in_size = num_elem * self.num_qpoints;
                    let out_size = num_elem * self.num_dof * self.ncomp;
                    if u.len() != in_size || v.len() != out_size {
                        return Err(ReedError::Basis(format!(
                            "div transpose apply size mismatch: input {}, expected {}; output {}, expected {}",
                            u.len(),
                            in_size,
                            v.len(),
                            out_size
                        )));
                    }
                    let mut grad_buf = vec![T::ZERO; num_elem * self.num_qpoints * qcomp];
                    for e in 0..num_elem {
                        for iq in 0..self.num_qpoints {
                            let w = u[e * self.num_qpoints + iq];
                            let base = (e * self.num_qpoints + iq) * qcomp;
                            for d in 0..self.dim {
                                grad_buf[base + d * self.dim + d] = w;
                            }
                        }
                    }
                    self.apply(num_elem, true, EvalMode::Grad, &grad_buf, v)?;
                } else {
                    let in_size = num_elem * self.num_dof * self.ncomp;
                    let out_size = num_elem * self.num_qpoints;
                    if u.len() != in_size || v.len() != out_size {
                        return Err(ReedError::Basis(format!(
                            "div apply size mismatch: input {}, expected {}; output {}, expected {}",
                            u.len(),
                            in_size,
                            v.len(),
                            out_size
                        )));
                    }
                    let grad_len = num_elem * self.num_qpoints * qcomp;
                    let mut grad_buf = vec![T::ZERO; grad_len];
                    self.apply(num_elem, false, EvalMode::Grad, u, &mut grad_buf)?;
                    for e in 0..num_elem {
                        for iq in 0..self.num_qpoints {
                            let idx = e * self.num_qpoints + iq;
                            let g_base = idx * qcomp;
                            let mut s = T::ZERO;
                            for d in 0..self.dim {
                                s += grad_buf[g_base + d * self.dim + d];
                            }
                            v[idx] = s;
                        }
                    }
                }
            }
            EvalMode::Curl => {
                // 2D: scalar curl (∂uy/∂x − ∂ux/∂y); 3D: vector curl in Cartesian coordinates.
                // Layout matches Grad: `qcomp = ncomp * dim`, index `comp * dim + dir`.
                let qcomp = self.ncomp * self.dim;
                match (self.dim, self.ncomp) {
                    (2, 2) => {
                        if transpose {
                            let in_size = num_elem * self.num_qpoints;
                            let out_size = num_elem * self.num_dof * self.ncomp;
                            if u.len() != in_size || v.len() != out_size {
                                return Err(ReedError::Basis(format!(
                                    "curl transpose apply size mismatch: input {}, expected {}; output {}, expected {}",
                                    u.len(),
                                    in_size,
                                    v.len(),
                                    out_size
                                )));
                            }
                            let mut grad_buf = vec![T::ZERO; num_elem * self.num_qpoints * qcomp];
                            for e in 0..num_elem {
                                for iq in 0..self.num_qpoints {
                                    let w = u[e * self.num_qpoints + iq];
                                    let base = (e * self.num_qpoints + iq) * qcomp;
                                    grad_buf[base + 1] -= w; // −∂/∂uy term
                                    grad_buf[base + 2] += w; // +∂/∂ux term (adjoint of curl_z)
                                }
                            }
                            self.apply(num_elem, true, EvalMode::Grad, &grad_buf, v)?;
                        } else {
                            let in_size = num_elem * self.num_dof * self.ncomp;
                            let out_size = num_elem * self.num_qpoints;
                            if u.len() != in_size || v.len() != out_size {
                                return Err(ReedError::Basis(format!(
                                    "curl apply size mismatch: input {}, expected {}; output {}, expected {}",
                                    u.len(),
                                    in_size,
                                    v.len(),
                                    out_size
                                )));
                            }
                            let mut grad_buf = vec![T::ZERO; num_elem * self.num_qpoints * qcomp];
                            self.apply(num_elem, false, EvalMode::Grad, u, &mut grad_buf)?;
                            for e in 0..num_elem {
                                for iq in 0..self.num_qpoints {
                                    let idx = e * self.num_qpoints + iq;
                                    let g_base = idx * qcomp;
                                    v[idx] = grad_buf[g_base + 2] - grad_buf[g_base + 1];
                                }
                            }
                        }
                    }
                    (3, 3) => {
                        if transpose {
                            let in_size = num_elem * self.num_qpoints * 3;
                            let out_size = num_elem * self.num_dof * self.ncomp;
                            if u.len() != in_size || v.len() != out_size {
                                return Err(ReedError::Basis(format!(
                                    "curl transpose apply size mismatch: input {}, expected {}; output {}, expected {}",
                                    u.len(),
                                    in_size,
                                    v.len(),
                                    out_size
                                )));
                            }
                            let mut grad_buf = vec![T::ZERO; num_elem * self.num_qpoints * qcomp];
                            for e in 0..num_elem {
                                for iq in 0..self.num_qpoints {
                                    let qidx = e * self.num_qpoints + iq;
                                    let w0 = u[qidx * 3];
                                    let w1 = u[qidx * 3 + 1];
                                    let w2 = u[qidx * 3 + 2];
                                    let base = qidx * qcomp;
                                    grad_buf[base + 7] += w0;
                                    grad_buf[base + 5] -= w0;
                                    grad_buf[base + 2] += w1;
                                    grad_buf[base + 6] -= w1;
                                    grad_buf[base + 3] += w2;
                                    grad_buf[base + 1] -= w2;
                                }
                            }
                            self.apply(num_elem, true, EvalMode::Grad, &grad_buf, v)?;
                        } else {
                            let in_size = num_elem * self.num_dof * self.ncomp;
                            let out_size = num_elem * self.num_qpoints * 3;
                            if u.len() != in_size || v.len() != out_size {
                                return Err(ReedError::Basis(format!(
                                    "curl apply size mismatch: input {}, expected {}; output {}, expected {}",
                                    u.len(),
                                    in_size,
                                    v.len(),
                                    out_size
                                )));
                            }
                            let mut grad_buf = vec![T::ZERO; num_elem * self.num_qpoints * qcomp];
                            self.apply(num_elem, false, EvalMode::Grad, u, &mut grad_buf)?;
                            for e in 0..num_elem {
                                for iq in 0..self.num_qpoints {
                                    let qidx = e * self.num_qpoints + iq;
                                    let g_base = qidx * qcomp;
                                    let g = &grad_buf[g_base..g_base + qcomp];
                                    v[qidx * 3] = g[7] - g[5];
                                    v[qidx * 3 + 1] = g[2] - g[6];
                                    v[qidx * 3 + 2] = g[3] - g[1];
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(ReedError::Basis(
                            "EvalMode::Curl requires (dim, ncomp) = (2, 2) or (3, 3)".into(),
                        ));
                    }
                }
            }
            other => {
                return Err(ReedError::Basis(format!(
                    "eval mode {:?} not implemented in CPU basis",
                    other
                )));
            }
        }
        Ok(())
    }

    fn q_weights(&self) -> &[T] {
        &self.weights
    }

    fn q_ref(&self) -> &[T] {
        &self.q_ref
    }

    fn tensor_fdm_1d_data(&self) -> Option<(&[T], &[T], &[T], usize, usize)> {
        Some((&self.interp, &self.grad, &self.weights_1d, self.p, self.q))
    }
}

impl<T: Scalar> LagrangeBasis<T> {
    fn apply_interp_elem(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        match self.dim {
            1 => self.apply_interp_elem_1d(transpose, u_elem, v_elem),
            2 => self.apply_interp_elem_2d(transpose, u_elem, v_elem),
            3 => self.apply_interp_elem_3d(transpose, u_elem, v_elem),
            _ => unreachable!(),
        }
    }

    fn apply_grad_elem(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        match self.dim {
            1 => self.apply_grad_elem_1d(transpose, u_elem, v_elem),
            2 => self.apply_grad_elem_2d(transpose, u_elem, v_elem),
            3 => self.apply_grad_elem_3d(transpose, u_elem, v_elem),
            _ => unreachable!(),
        }
    }

    fn apply_interp_elem_1d(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        for comp in 0..self.ncomp {
            let (u_offset, u_stride, v_offset, v_stride) = if transpose {
                (comp, self.ncomp, comp * self.p, 1)
            } else {
                (comp * self.p, 1, comp, self.ncomp)
            };
            tensor_contract_strided(
                &self.interp,
                &u_elem[u_offset..],
                u_stride,
                &mut v_elem[v_offset..],
                v_stride,
                self.q,
                self.p,
                transpose,
            );
        }
    }

    fn apply_interp_elem_2d(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        let qp = self.q * self.p;
        let qq = self.q * self.q;
        let pp = self.p * self.p;
        // Small buffer: q * p elements (typically 24-80 elements for p=4-8, q=6-10)
        let mut tmp = vec![T::ZERO; qp];

        for comp in 0..self.ncomp {
            if transpose {
                tmp.fill(T::ZERO);
                for qx in 0..self.q {
                    tensor_contract_strided(
                        &self.interp,
                        &u_elem[qx * self.ncomp + comp..],
                        self.q * self.ncomp,
                        &mut tmp[qx..],
                        self.q,
                        self.q,
                        self.p,
                        true,
                    );
                }
                for py in 0..self.p {
                    tensor_contract(
                        &self.interp,
                        &tmp[py * self.q..(py + 1) * self.q],
                        &mut v_elem[comp * pp + py * self.p..comp * pp + (py + 1) * self.p],
                        self.q,
                        self.p,
                        true,
                    );
                }
            } else {
                let u_comp = &u_elem[comp * pp..(comp + 1) * pp];
                for py in 0..self.p {
                    tensor_contract(
                        &self.interp,
                        &u_comp[py * self.p..(py + 1) * self.p],
                        &mut tmp[py * self.q..(py + 1) * self.q],
                        self.q,
                        self.p,
                        false,
                    );
                }
                for qx in 0..self.q {
                    tensor_contract_strided(
                        &self.interp,
                        &tmp[qx..],
                        self.q,
                        &mut v_elem[qx * self.ncomp + comp..],
                        self.q * self.ncomp,
                        self.q,
                        self.p,
                        false,
                    );
                }
            }
        }

        let _ = qq;
    }

    fn apply_interp_elem_3d(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        let p2 = self.p * self.p;
        let q2 = self.q * self.q;
        let ppp = p2 * self.p;
        let qqq = q2 * self.q;
        // Small buffers: q * p^2 (typically 96-512) and q^2 * p (typically 144-800)
        let mut tmp_x = vec![T::ZERO; self.q * p2];
        let mut tmp_xy = vec![T::ZERO; q2 * self.p];

        for comp in 0..self.ncomp {
            if transpose {
                tmp_xy.fill(T::ZERO);
                for pz in 0..self.p {
                    for py in 0..self.p {
                        for qx in 0..self.q {
                            let mut sum = T::ZERO;
                            for qz in 0..self.q {
                                for qy in 0..self.q {
                                    let qpt = (qz * q2) + (qy * self.q) + qx;
                                    sum += self.interp[qz * self.p + pz]
                                        * self.interp[qy * self.p + py]
                                        * u_elem[qpt * self.ncomp + comp];
                                }
                            }
                            tmp_xy[(pz * self.p + py) * self.q + qx] = sum;
                        }
                    }
                }
                for pz in 0..self.p {
                    for py in 0..self.p {
                        let row = (pz * self.p + py) * self.q;
                        let dst = comp * ppp + pz * p2 + py * self.p;
                        tensor_contract(
                            &self.interp,
                            &tmp_xy[row..row + self.q],
                            &mut v_elem[dst..dst + self.p],
                            self.q,
                            self.p,
                            true,
                        );
                    }
                }
            } else {
                let u_comp = &u_elem[comp * ppp..(comp + 1) * ppp];
                for pz in 0..self.p {
                    for py in 0..self.p {
                        let src = pz * p2 + py * self.p;
                        let dst = (pz * self.p + py) * self.q;
                        tensor_contract(
                            &self.interp,
                            &u_comp[src..src + self.p],
                            &mut tmp_x[dst..dst + self.q],
                            self.q,
                            self.p,
                            false,
                        );
                    }
                }
                for pz in 0..self.p {
                    for qy in 0..self.q {
                        for qx in 0..self.q {
                            let mut sum = T::ZERO;
                            for py in 0..self.p {
                                sum += self.interp[qy * self.p + py]
                                    * tmp_x[(pz * self.p + py) * self.q + qx];
                            }
                            tmp_xy[(pz * q2) + (qy * self.q) + qx] = sum;
                        }
                    }
                }
                for qz in 0..self.q {
                    for qy in 0..self.q {
                        for qx in 0..self.q {
                            let mut sum = T::ZERO;
                            for pz in 0..self.p {
                                sum += self.interp[qz * self.p + pz]
                                    * tmp_xy[(pz * q2) + (qy * self.q) + qx];
                            }
                            v_elem[((qz * q2) + (qy * self.q) + qx) * self.ncomp + comp] = sum;
                        }
                    }
                }
            }
        }

        let _ = qqq;
    }

    fn apply_grad_elem_1d(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        for comp in 0..self.ncomp {
            let (u_offset, u_stride, v_offset, v_stride) = if transpose {
                (comp, self.ncomp, comp * self.p, 1)
            } else {
                (comp * self.p, 1, comp, self.ncomp)
            };
            tensor_contract_strided(
                &self.grad,
                &u_elem[u_offset..],
                u_stride,
                &mut v_elem[v_offset..],
                v_stride,
                self.q,
                self.p,
                transpose,
            );
        }
    }

    fn apply_grad_elem_2d(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        let qcomp = self.ncomp * 2;
        let pp = self.p * self.p;
        // Small buffers: q * p (typically 24-80) and p^2 (typically 16-64)
        let mut tmp_interp_x = vec![T::ZERO; self.q * self.p];
        let mut tmp_grad_x = vec![T::ZERO; self.q * self.p];
        let mut accum_x = vec![T::ZERO; pp];
        let mut accum_y = vec![T::ZERO; pp];

        for comp in 0..self.ncomp {
            if transpose {
                accum_x.fill(T::ZERO);
                accum_y.fill(T::ZERO);

                for qx in 0..self.q {
                    tensor_contract_strided(
                        &self.interp,
                        &u_elem[qx * qcomp + comp * 2..],
                        self.q * qcomp,
                        &mut tmp_grad_x[qx..],
                        self.q,
                        self.q,
                        self.p,
                        true,
                    );
                    tensor_contract_strided(
                        &self.grad,
                        &u_elem[qx * qcomp + comp * 2 + 1..],
                        self.q * qcomp,
                        &mut tmp_interp_x[qx..],
                        self.q,
                        self.q,
                        self.p,
                        true,
                    );
                }

                for py in 0..self.p {
                    tensor_contract(
                        &self.grad,
                        &tmp_grad_x[py * self.q..(py + 1) * self.q],
                        &mut accum_x[py * self.p..(py + 1) * self.p],
                        self.q,
                        self.p,
                        true,
                    );
                    tensor_contract(
                        &self.interp,
                        &tmp_interp_x[py * self.q..(py + 1) * self.q],
                        &mut accum_y[py * self.p..(py + 1) * self.p],
                        self.q,
                        self.p,
                        true,
                    );
                    for px in 0..self.p {
                        let dst = comp * pp + py * self.p + px;
                        v_elem[dst] = accum_x[py * self.p + px] + accum_y[py * self.p + px];
                    }
                }
            } else {
                let u_comp = &u_elem[comp * pp..(comp + 1) * pp];
                for py in 0..self.p {
                    let row = &u_comp[py * self.p..(py + 1) * self.p];
                    tensor_contract(
                        &self.interp,
                        row,
                        &mut tmp_interp_x[py * self.q..(py + 1) * self.q],
                        self.q,
                        self.p,
                        false,
                    );
                    tensor_contract(
                        &self.grad,
                        row,
                        &mut tmp_grad_x[py * self.q..(py + 1) * self.q],
                        self.q,
                        self.p,
                        false,
                    );
                }

                for qx in 0..self.q {
                    tensor_contract_strided(
                        &self.interp,
                        &tmp_grad_x[qx..],
                        self.q,
                        &mut v_elem[qx * qcomp + comp * 2..],
                        self.q * qcomp,
                        self.q,
                        self.p,
                        false,
                    );
                    tensor_contract_strided(
                        &self.grad,
                        &tmp_interp_x[qx..],
                        self.q,
                        &mut v_elem[qx * qcomp + comp * 2 + 1..],
                        self.q * qcomp,
                        self.q,
                        self.p,
                        false,
                    );
                }
            }
        }
    }

    fn apply_grad_elem_3d(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        let qcomp = self.ncomp * 3;
        let p2 = self.p * self.p;
        let q2 = self.q * self.q;
        let ppp = p2 * self.p;
        // Small buffers: q * p^2 (typically 96-512), q^2 * p (typically 144-800), p^3 (typically 64-512)
        let mut tmp_x = vec![T::ZERO; self.q * p2];
        let mut tmp_y = vec![T::ZERO; q2 * self.p];
        let mut accum = vec![T::ZERO; ppp];

        for comp in 0..self.ncomp {
            if transpose {
                accum.fill(T::ZERO);

                for direction in 0..3 {
                    let bx = if direction == 0 {
                        &self.grad
                    } else {
                        &self.interp
                    };
                    for pz in 0..self.p {
                        for qy in 0..self.q {
                            for qx in 0..self.q {
                                let mut sum = T::ZERO;
                                for qz in 0..self.q {
                                    let base = ((qz * q2) + (qy * self.q) + qx) * qcomp
                                        + comp * 3
                                        + direction;
                                    let bz = if direction == 2 {
                                        self.grad[qz * self.p + pz]
                                    } else {
                                        self.interp[qz * self.p + pz]
                                    };
                                    sum += bz * u_elem[base];
                                }
                                tmp_y[(pz * q2) + (qy * self.q) + qx] = sum;
                            }
                        }
                    }

                    for pz in 0..self.p {
                        for py in 0..self.p {
                            for qx in 0..self.q {
                                let mut sum = T::ZERO;
                                for qy in 0..self.q {
                                    let by = if direction == 1 {
                                        self.grad[qy * self.p + py]
                                    } else {
                                        self.interp[qy * self.p + py]
                                    };
                                    sum += by * tmp_y[(pz * q2) + (qy * self.q) + qx];
                                }
                                tmp_x[(pz * self.p + py) * self.q + qx] = sum;
                            }
                        }
                    }

                    for pz in 0..self.p {
                        for py in 0..self.p {
                            let src = (pz * self.p + py) * self.q;
                            let dst = pz * p2 + py * self.p;
                            tensor_contract_accumulate(
                                bx,
                                &tmp_x[src..src + self.q],
                                &mut accum[dst..dst + self.p],
                                self.q,
                                self.p,
                                true,
                            );
                        }
                    }
                }

                let dst = &mut v_elem[comp * ppp..(comp + 1) * ppp];
                dst.copy_from_slice(&accum);
            } else {
                let u_comp = &u_elem[comp * ppp..(comp + 1) * ppp];
                for direction in 0..3 {
                    let bx = if direction == 0 {
                        &self.grad
                    } else {
                        &self.interp
                    };
                    let by = if direction == 1 {
                        &self.grad
                    } else {
                        &self.interp
                    };
                    let bz = if direction == 2 {
                        &self.grad
                    } else {
                        &self.interp
                    };
                    for pz in 0..self.p {
                        for py in 0..self.p {
                            let src = pz * p2 + py * self.p;
                            let dst = (pz * self.p + py) * self.q;
                            tensor_contract(
                                bx,
                                &u_comp[src..src + self.p],
                                &mut tmp_x[dst..dst + self.q],
                                self.q,
                                self.p,
                                false,
                            );
                        }
                    }

                    for pz in 0..self.p {
                        for qx in 0..self.q {
                            tensor_contract_strided(
                                by,
                                &tmp_x[pz * self.p * self.q + qx..],
                                self.q,
                                &mut tmp_y[pz * q2 + qx..],
                                self.q,
                                self.q,
                                self.p,
                                false,
                            );
                        }
                    }

                    for qy in 0..self.q {
                        for qx in 0..self.q {
                            tensor_contract_strided(
                                bz,
                                &tmp_y[qy * self.q + qx..],
                                q2,
                                &mut v_elem[(qy * self.q + qx) * qcomp + comp * 3 + direction..],
                                q2 * qcomp,
                                self.q,
                                self.p,
                                false,
                            );
                        }
                    }
                }
            }
        }
    }
}

pub fn tensor_contract_strided<T: Scalar>(
    b: &[T],
    u: &[T],
    u_stride: usize,
    v: &mut [T],
    v_stride: usize,
    q: usize,
    p: usize,
    transpose: bool,
) {
    // Try f32 SIMD first (8-wide vectors, potentially faster than f64)
    if try_tensor_contract_simd_f32(b, u, u_stride, v, v_stride, q, p, transpose) {
        return;
    }
    // Fall back to f64 SIMD (4-wide vectors)
    if try_tensor_contract_simd_f64(b, u, u_stride, v, v_stride, q, p, transpose) {
        return;
    }

    tensor_contract_strided_scalar(b, u, u_stride, v, v_stride, q, p, transpose);
}

fn tensor_contract_strided_scalar<T: Scalar>(
    b: &[T],
    u: &[T],
    u_stride: usize,
    v: &mut [T],
    v_stride: usize,
    q: usize,
    p: usize,
    transpose: bool,
) {
    if transpose {
        for pi in 0..p {
            let mut sum = T::ZERO;
            for qi in 0..q {
                sum += b[qi * p + pi] * u[qi * u_stride];
            }
            v[pi * v_stride] = sum;
        }
    } else {
        for qi in 0..q {
            let mut sum = T::ZERO;
            for pi in 0..p {
                sum += b[qi * p + pi] * u[pi * u_stride];
            }
            v[qi * v_stride] = sum;
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn try_tensor_contract_simd_f32<T: Scalar>(
    b: &[T],
    u: &[T],
    u_stride: usize,
    v: &mut [T],
    v_stride: usize,
    q: usize,
    p: usize,
    transpose: bool,
) -> bool {
    if TypeId::of::<T>() != TypeId::of::<f32>() || u_stride != 1 || v_stride != 1 || p < 8 {
        return false;
    }

    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
        return false;
    }

    unsafe {
        let b_f32 = std::slice::from_raw_parts(b.as_ptr().cast::<f32>(), b.len());
        let u_f32 = std::slice::from_raw_parts(u.as_ptr().cast::<f32>(), u.len());
        let v_f32 = std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<f32>(), v.len());
        tensor_contract_f32_avx2(b_f32, u_f32, v_f32, q, p, transpose);
    }
    true
}

#[cfg(not(target_arch = "x86_64"))]
fn try_tensor_contract_simd_f32<T: Scalar>(
    _b: &[T],
    _u: &[T],
    _u_stride: usize,
    _v: &mut [T],
    _v_stride: usize,
    _q: usize,
    _p: usize,
    _transpose: bool,
) -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn try_tensor_contract_simd_f64<T: Scalar>(
    b: &[T],
    u: &[T],
    u_stride: usize,
    v: &mut [T],
    v_stride: usize,
    q: usize,
    p: usize,
    transpose: bool,
) -> bool {
    if TypeId::of::<T>() != TypeId::of::<f64>() || u_stride != 1 || v_stride != 1 || p < 4 {
        return false;
    }

    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
        return false;
    }

    unsafe {
        let b_f64 = std::slice::from_raw_parts(b.as_ptr().cast::<f64>(), b.len());
        let u_f64 = std::slice::from_raw_parts(u.as_ptr().cast::<f64>(), u.len());
        let v_f64 = std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<f64>(), v.len());
        tensor_contract_f64_avx2(b_f64, u_f64, v_f64, q, p, transpose);
    }
    true
}

#[cfg(not(target_arch = "x86_64"))]
fn try_tensor_contract_simd_f64<T: Scalar>(
    _b: &[T],
    _u: &[T],
    _u_stride: usize,
    _v: &mut [T],
    _v_stride: usize,
    _q: usize,
    _p: usize,
    _transpose: bool,
) -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn try_tensor_contract_accumulate_simd_f32<T: Scalar>(
    b: &[T],
    u: &[T],
    u_stride: usize,
    v: &mut [T],
    v_stride: usize,
    q: usize,
    p: usize,
    transpose: bool,
) -> bool {
    if TypeId::of::<T>() != TypeId::of::<f32>() || u_stride != 1 || v_stride != 1 || p < 8 {
        return false;
    }

    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
        return false;
    }

    unsafe {
        let b_f32 = std::slice::from_raw_parts(b.as_ptr().cast::<f32>(), b.len());
        let u_f32 = std::slice::from_raw_parts(u.as_ptr().cast::<f32>(), u.len());
        let v_f32 = std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<f32>(), v.len());
        tensor_contract_f32_avx2_accumulate(b_f32, u_f32, v_f32, q, p, transpose);
    }
    true
}

#[cfg(not(target_arch = "x86_64"))]
fn try_tensor_contract_accumulate_simd_f32<T: Scalar>(
    _b: &[T],
    _u: &[T],
    _u_stride: usize,
    _v: &mut [T],
    _v_stride: usize,
    _q: usize,
    _p: usize,
    _transpose: bool,
) -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn try_tensor_contract_accumulate_simd_f64<T: Scalar>(
    b: &[T],
    u: &[T],
    u_stride: usize,
    v: &mut [T],
    v_stride: usize,
    q: usize,
    p: usize,
    transpose: bool,
) -> bool {
    if TypeId::of::<T>() != TypeId::of::<f64>() || u_stride != 1 || v_stride != 1 || p < 4 {
        return false;
    }

    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
        return false;
    }

    unsafe {
        let b_f64 = std::slice::from_raw_parts(b.as_ptr().cast::<f64>(), b.len());
        let u_f64 = std::slice::from_raw_parts(u.as_ptr().cast::<f64>(), u.len());
        let v_f64 = std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<f64>(), v.len());
        tensor_contract_f64_avx2_accumulate(b_f64, u_f64, v_f64, q, p, transpose);
    }
    true
}

#[cfg(not(target_arch = "x86_64"))]
fn try_tensor_contract_accumulate_simd_f64<T: Scalar>(
    _b: &[T],
    _u: &[T],
    _u_stride: usize,
    _v: &mut [T],
    _v_stride: usize,
    _q: usize,
    _p: usize,
    _transpose: bool,
) -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn tensor_contract_f32_avx2(
    b: &[f32],
    u: &[f32],
    v: &mut [f32],
    q: usize,
    p: usize,
    transpose: bool,
) {
    use std::arch::x86_64::{
        __m256, _mm256_broadcast_ss, _mm256_castps256_ps128, _mm256_extractf128_ps,
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm_add_ps,
        _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    };

    #[inline]
    unsafe fn hsum_ps(sum: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(sum);
        let hi = _mm256_extractf128_ps(sum, 1);
        let pair = _mm_add_ps(lo, hi);
        let shuffled = _mm_movehl_ps(pair, pair);
        let sum = _mm_add_ps(pair, shuffled);
        let shuffled2 = _mm_shuffle_ps(sum, sum, 1);
        _mm_cvtss_f32(_mm_add_ps(sum, shuffled2))
    }

    if transpose {
        let mut pi = 0;
        while pi + 8 <= p {
            let mut acc = _mm256_setzero_ps();
            for qi in 0..q {
                let coeff = _mm256_broadcast_ss(&*u.as_ptr().add(qi));
                let row = _mm256_loadu_ps(b.as_ptr().add(qi * p + pi));
                acc = _mm256_fmadd_ps(coeff, row, acc);
            }
            _mm256_storeu_ps(v.as_mut_ptr().add(pi), acc);
            pi += 8;
        }
        while pi < p {
            let mut sum = 0.0_f32;
            for qi in 0..q {
                sum += b[qi * p + pi] * u[qi];
            }
            v[pi] = sum;
            pi += 1;
        }
    } else {
        for qi in 0..q {
            let row = b.as_ptr().add(qi * p);
            let mut acc = _mm256_setzero_ps();
            let mut pi = 0;
            while pi + 8 <= p {
                let row_v = _mm256_loadu_ps(row.add(pi));
                let u_v = _mm256_loadu_ps(u.as_ptr().add(pi));
                acc = _mm256_fmadd_ps(row_v, u_v, acc);
                pi += 8;
            }
            let mut sum = hsum_ps(acc);
            while pi < p {
                sum += *row.add(pi) * u[pi];
                pi += 1;
            }
            v[qi] = sum;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn tensor_contract_f32_avx2_accumulate(
    b: &[f32],
    u: &[f32],
    v: &mut [f32],
    q: usize,
    p: usize,
    transpose: bool,
) {
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_broadcast_ss, _mm256_castps256_ps128, _mm256_extractf128_ps,
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm_add_ps,
        _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    };

    #[inline]
    unsafe fn hsum_ps(sum: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(sum);
        let hi = _mm256_extractf128_ps(sum, 1);
        let pair = _mm_add_ps(lo, hi);
        let shuffled = _mm_movehl_ps(pair, pair);
        let sum = _mm_add_ps(pair, shuffled);
        let shuffled2 = _mm_shuffle_ps(sum, sum, 1);
        _mm_cvtss_f32(_mm_add_ps(sum, shuffled2))
    }

    if transpose {
        let mut pi = 0;
        while pi + 8 <= p {
            let mut acc = _mm256_setzero_ps();
            for qi in 0..q {
                let coeff = _mm256_broadcast_ss(&*u.as_ptr().add(qi));
                let row = _mm256_loadu_ps(b.as_ptr().add(qi * p + pi));
                acc = _mm256_fmadd_ps(coeff, row, acc);
            }
            let cur = _mm256_loadu_ps(v.as_ptr().add(pi));
            _mm256_storeu_ps(v.as_mut_ptr().add(pi), _mm256_add_ps(cur, acc));
            pi += 8;
        }
        while pi < p {
            let mut sum = 0.0_f32;
            for qi in 0..q {
                sum += b[qi * p + pi] * u[qi];
            }
            v[pi] += sum;
            pi += 1;
        }
    } else {
        for qi in 0..q {
            let row = b.as_ptr().add(qi * p);
            let mut acc = _mm256_setzero_ps();
            let mut pi = 0;
            while pi + 8 <= p {
                let row_v = _mm256_loadu_ps(row.add(pi));
                let u_v = _mm256_loadu_ps(u.as_ptr().add(pi));
                acc = _mm256_fmadd_ps(row_v, u_v, acc);
                pi += 8;
            }
            let mut sum = hsum_ps(acc);
            while pi < p {
                sum += *row.add(pi) * u[pi];
                pi += 1;
            }
            v[qi] += sum;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn tensor_contract_f64_avx2(
    b: &[f64],
    u: &[f64],
    v: &mut [f64],
    q: usize,
    p: usize,
    transpose: bool,
) {
    use std::arch::x86_64::{
        __m128d, __m256d, _mm256_broadcast_sd, _mm256_castpd256_pd128, _mm256_extractf128_pd,
        _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm_add_pd,
        _mm_cvtsd_f64, _mm_unpackhi_pd,
    };

    #[inline]
    unsafe fn hsum_pd(sum: __m256d) -> f64 {
        let lo = _mm256_castpd256_pd128(sum);
        let hi = _mm256_extractf128_pd(sum, 1);
        let pair = _mm_add_pd(lo, hi);
        let swapped: __m128d = _mm_unpackhi_pd(pair, pair);
        _mm_cvtsd_f64(_mm_add_pd(pair, swapped))
    }

    if transpose {
        let mut pi = 0;
        while pi + 4 <= p {
            let mut acc = _mm256_setzero_pd();
            for qi in 0..q {
                let coeff = _mm256_broadcast_sd(&u[qi]);
                let row = _mm256_loadu_pd(b.as_ptr().add(qi * p + pi));
                acc = _mm256_fmadd_pd(coeff, row, acc);
            }
            _mm256_storeu_pd(v.as_mut_ptr().add(pi), acc);
            pi += 4;
        }
        while pi < p {
            let mut sum = 0.0;
            for qi in 0..q {
                sum += b[qi * p + pi] * u[qi];
            }
            v[pi] = sum;
            pi += 1;
        }
    } else {
        for qi in 0..q {
            let row = b.as_ptr().add(qi * p);
            let mut acc = _mm256_setzero_pd();
            let mut pi = 0;
            while pi + 4 <= p {
                let row_v = _mm256_loadu_pd(row.add(pi));
                let u_v = _mm256_loadu_pd(u.as_ptr().add(pi));
                acc = _mm256_fmadd_pd(row_v, u_v, acc);
                pi += 4;
            }
            let mut sum = hsum_pd(acc);
            while pi < p {
                sum += *row.add(pi) * u[pi];
                pi += 1;
            }
            v[qi] = sum;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn tensor_contract_f64_avx2_accumulate(
    b: &[f64],
    u: &[f64],
    v: &mut [f64],
    q: usize,
    p: usize,
    transpose: bool,
) {
    use std::arch::x86_64::{
        __m128d, __m256d, _mm256_add_pd, _mm256_broadcast_sd, _mm256_castpd256_pd128,
        _mm256_extractf128_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd,
        _mm256_storeu_pd, _mm_add_pd, _mm_cvtsd_f64, _mm_unpackhi_pd,
    };

    #[inline]
    unsafe fn hsum_pd(sum: __m256d) -> f64 {
        let lo = _mm256_castpd256_pd128(sum);
        let hi = _mm256_extractf128_pd(sum, 1);
        let pair = _mm_add_pd(lo, hi);
        let swapped: __m128d = _mm_unpackhi_pd(pair, pair);
        _mm_cvtsd_f64(_mm_add_pd(pair, swapped))
    }

    if transpose {
        let mut pi = 0;
        while pi + 4 <= p {
            let mut acc = _mm256_setzero_pd();
            for qi in 0..q {
                let coeff = _mm256_broadcast_sd(&u[qi]);
                let row = _mm256_loadu_pd(b.as_ptr().add(qi * p + pi));
                acc = _mm256_fmadd_pd(coeff, row, acc);
            }
            let cur = _mm256_loadu_pd(v.as_ptr().add(pi));
            _mm256_storeu_pd(v.as_mut_ptr().add(pi), _mm256_add_pd(cur, acc));
            pi += 4;
        }
        while pi < p {
            let mut sum = 0.0;
            for qi in 0..q {
                sum += b[qi * p + pi] * u[qi];
            }
            v[pi] += sum;
            pi += 1;
        }
    } else {
        for qi in 0..q {
            let row = b.as_ptr().add(qi * p);
            let mut acc = _mm256_setzero_pd();
            let mut pi = 0;
            while pi + 4 <= p {
                let row_v = _mm256_loadu_pd(row.add(pi));
                let u_v = _mm256_loadu_pd(u.as_ptr().add(pi));
                acc = _mm256_fmadd_pd(row_v, u_v, acc);
                pi += 4;
            }
            let mut sum = hsum_pd(acc);
            while pi < p {
                sum += *row.add(pi) * u[pi];
                pi += 1;
            }
            v[qi] += sum;
        }
    }
}

pub fn tensor_contract<T: Scalar>(
    b: &[T],
    u: &[T],
    v: &mut [T],
    q: usize,
    p: usize,
    transpose: bool,
) {
    tensor_contract_strided(b, u, 1, v, 1, q, p, transpose);
}

pub fn tensor_contract_accumulate_strided<T: Scalar>(
    b: &[T],
    u: &[T],
    u_stride: usize,
    v: &mut [T],
    v_stride: usize,
    q: usize,
    p: usize,
    transpose: bool,
) {
    // Try f32 SIMD first (8-wide vectors)
    if try_tensor_contract_accumulate_simd_f32(b, u, u_stride, v, v_stride, q, p, transpose) {
        return;
    }
    // Fall back to f64 SIMD (4-wide vectors)
    if try_tensor_contract_accumulate_simd_f64(b, u, u_stride, v, v_stride, q, p, transpose) {
        return;
    }

    if transpose {
        for pi in 0..p {
            let mut sum = T::ZERO;
            for qi in 0..q {
                sum += b[qi * p + pi] * u[qi * u_stride];
            }
            v[pi * v_stride] += sum;
        }
    } else {
        for qi in 0..q {
            let mut sum = T::ZERO;
            for pi in 0..p {
                sum += b[qi * p + pi] * u[pi * u_stride];
            }
            v[qi * v_stride] += sum;
        }
    }
}

pub fn tensor_contract_accumulate<T: Scalar>(
    b: &[T],
    u: &[T],
    v: &mut [T],
    q: usize,
    p: usize,
    transpose: bool,
) {
    tensor_contract_accumulate_strided(b, u, 1, v, 1, q, p, transpose);
}

fn to_scalar<T: Scalar>(value: f64) -> ReedResult<T> {
    T::from(value).ok_or_else(|| ReedError::Basis(format!("failed to convert {} to scalar", value)))
}

fn legendre(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    let mut pnm1 = 1.0;
    let mut pn = x;
    for k in 2..=n {
        let kf = k as f64;
        let pk = ((2.0 * kf - 1.0) * x * pn - (kf - 1.0) * pnm1) / kf;
        pnm1 = pn;
        pn = pk;
    }
    let dp = (n as f64) * (x * pn - pnm1) / (x * x - 1.0);
    (pn, dp)
}

pub fn gauss_quadrature(n: usize) -> ReedResult<(Vec<f64>, Vec<f64>)> {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let m = n.div_ceil(2);
    for i in 0..m {
        let nf = n as f64;
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        for _ in 0..100 {
            let (pn, dpn) = legendre(n, x);
            let dx = -pn / dpn;
            x += dx;
            if dx.abs() < 1.0e-14 {
                break;
            }
        }
        let (_, dpn) = legendre(n, x);
        let w = 2.0 / ((1.0 - x * x) * dpn * dpn);
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    Ok((nodes, weights))
}

pub fn gauss_lobatto_nodes(n: usize) -> ReedResult<Vec<f64>> {
    if n < 2 {
        return Err(ReedError::Basis(format!(
            "gauss_lobatto_nodes requires n>=2, got {}",
            n
        )));
    }
    if n == 2 {
        return Ok(vec![-1.0, 1.0]);
    }

    let mut nodes = vec![-1.0; n];
    nodes[n - 1] = 1.0;
    for (i, node) in nodes.iter_mut().enumerate().take(n - 1).skip(1) {
        let mut x = -(std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos();
        for _ in 0..100 {
            let (pnm1, dpnm1) = legendre(n - 1, x);
            let ddpnm1 = (2.0 * x * dpnm1 - (n as f64 - 1.0) * n as f64 * pnm1) / (1.0 - x * x);
            let dx = -dpnm1 / ddpnm1;
            x += dx;
            if dx.abs() < 1.0e-14 {
                break;
            }
        }
        *node = x;
    }
    Ok(nodes)
}

pub fn gauss_lobatto_quadrature(n: usize) -> ReedResult<(Vec<f64>, Vec<f64>)> {
    let nodes = gauss_lobatto_nodes(n)?;
    let mut weights = vec![0.0; n];
    let nn = n as f64;
    for i in 0..n {
        let (pnm1, _) = legendre(n - 1, nodes[i]);
        weights[i] = 2.0 / (nn * (nn - 1.0) * pnm1 * pnm1);
    }
    Ok((nodes, weights))
}

fn barycentric_weights(nodes: &[f64]) -> Vec<f64> {
    let mut weights = vec![1.0; nodes.len()];
    for j in 0..nodes.len() {
        let mut w = 1.0;
        for (k, &xk) in nodes.iter().enumerate() {
            if j != k {
                w *= nodes[j] - xk;
            }
        }
        weights[j] = 1.0 / w;
    }
    weights
}

fn build_interp<T: Scalar>(nodes: &[f64], qref: &[f64]) -> ReedResult<Vec<T>> {
    let bary = barycentric_weights(nodes);
    let mut interp = Vec::with_capacity(qref.len() * nodes.len());
    for &x in qref {
        let mut exact = None;
        for (j, &node) in nodes.iter().enumerate() {
            if (x - node).abs() < 1.0e-14 {
                exact = Some(j);
                break;
            }
        }
        if let Some(index) = exact {
            for j in 0..nodes.len() {
                interp.push(to_scalar::<T>(if j == index { 1.0 } else { 0.0 })?);
            }
            continue;
        }
        let denom: f64 = nodes
            .iter()
            .enumerate()
            .map(|(j, &node)| bary[j] / (x - node))
            .sum();
        for (j, &node) in nodes.iter().enumerate() {
            interp.push(to_scalar::<T>((bary[j] / (x - node)) / denom)?);
        }
    }
    Ok(interp)
}

fn build_grad<T: Scalar>(nodes: &[f64], qref: &[f64]) -> ReedResult<Vec<T>> {
    let bary = barycentric_weights(nodes);
    let interp = build_interp::<T>(nodes, qref)?;
    let interp_f64 = interp
        .iter()
        .map(|value| value.to_f64().unwrap())
        .collect::<Vec<_>>();
    let mut grad = Vec::with_capacity(qref.len() * nodes.len());
    for (qi, &x) in qref.iter().enumerate() {
        let exact = nodes.iter().position(|&node| (x - node).abs() < 1.0e-14);
        if let Some(j_exact) = exact {
            for i in 0..nodes.len() {
                if i == j_exact {
                    let mut sum = 0.0;
                    for m in 0..nodes.len() {
                        if m != i {
                            sum += 1.0 / (nodes[i] - nodes[m]);
                        }
                    }
                    grad.push(to_scalar::<T>(sum)?);
                } else {
                    grad.push(to_scalar::<T>(
                        bary[i] / (bary[j_exact] * (nodes[j_exact] - nodes[i])),
                    )?);
                }
            }
            continue;
        }
        let s1: f64 = nodes
            .iter()
            .enumerate()
            .map(|(j, &node)| bary[j] / (x - node))
            .sum();
        let s2: f64 = nodes
            .iter()
            .enumerate()
            .map(|(j, &node)| bary[j] / ((x - node) * (x - node)))
            .sum();
        for i in 0..nodes.len() {
            let li = interp_f64[qi * nodes.len() + i];
            let value = li * (s2 / s1 - 1.0 / (x - nodes[i]));
            grad.push(to_scalar::<T>(value)?);
        }
    }
    Ok(grad)
}

fn build_tensor_qref<T: Scalar>(qref_1d: &[f64], dim: usize) -> ReedResult<Vec<T>> {
    let mut q_ref = Vec::with_capacity(qref_1d.len().pow(dim as u32) * dim);
    match dim {
        1 => {
            for &x in qref_1d {
                q_ref.push(to_scalar::<T>(x)?);
            }
        }
        2 => {
            for &y in qref_1d {
                for &x in qref_1d {
                    q_ref.push(to_scalar::<T>(x)?);
                    q_ref.push(to_scalar::<T>(y)?);
                }
            }
        }
        3 => {
            for &z in qref_1d {
                for &y in qref_1d {
                    for &x in qref_1d {
                        q_ref.push(to_scalar::<T>(x)?);
                        q_ref.push(to_scalar::<T>(y)?);
                        q_ref.push(to_scalar::<T>(z)?);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    Ok(q_ref)
}

fn build_tensor_weights<T: Scalar>(weights_1d: &[f64], dim: usize) -> ReedResult<Vec<T>> {
    let mut weights = Vec::with_capacity(weights_1d.len().pow(dim as u32));
    match dim {
        1 => {
            for &w in weights_1d {
                weights.push(to_scalar::<T>(w)?);
            }
        }
        2 => {
            for &wy in weights_1d {
                for &wx in weights_1d {
                    weights.push(to_scalar::<T>(wx * wy)?);
                }
            }
        }
        3 => {
            for &wz in weights_1d {
                for &wy in weights_1d {
                    for &wx in weights_1d {
                        weights.push(to_scalar::<T>(wx * wy * wz)?);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_weights_sum_to_two() {
        let (_, weights) = gauss_quadrature(3).unwrap();
        assert!((weights.iter().sum::<f64>() - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_interp_of_constant() {
        let basis = LagrangeBasis::<f64>::new(1, 1, 3, 4, reed_core::QuadMode::Gauss).unwrap();
        let u = vec![2.0; 3];
        let mut v = vec![0.0; 4];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        for value in v {
            assert!((value - 2.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn test_2d_weights_sum_to_four() {
        let basis = LagrangeBasis::<f64>::new(2, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        assert!((basis.q_weights().iter().sum::<f64>() - 4.0).abs() < 1.0e-12);
        assert_eq!(basis.num_dof(), 4);
        assert_eq!(basis.num_qpoints(), 4);
    }

    #[test]
    fn test_2d_interp_of_constant() {
        let basis = LagrangeBasis::<f64>::new(2, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        let u = vec![3.5; 4];
        let mut v = vec![0.0; 4];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        for value in v {
            assert!((value - 3.5).abs() < 1.0e-12);
        }
    }

    #[test]
    fn test_2d_grad_of_linear_function() {
        let basis = LagrangeBasis::<f64>::new(2, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        let nodes = [-1.0_f64, 1.0];
        let mut u = Vec::new();
        for &y in &nodes {
            for &x in &nodes {
                u.push(2.0 * x - 3.0 * y + 1.0);
            }
        }
        let mut grad = vec![0.0; 4 * 2];
        basis
            .apply(1, false, EvalMode::Grad, &u, &mut grad)
            .unwrap();
        for qpt in 0..4 {
            assert!((grad[qpt * 2] - 2.0).abs() < 1.0e-12);
            assert!((grad[qpt * 2 + 1] + 3.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn test_3d_weights_sum_to_eight() {
        let basis = LagrangeBasis::<f64>::new(3, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        assert!((basis.q_weights().iter().sum::<f64>() - 8.0).abs() < 1.0e-12);
        assert_eq!(basis.num_dof(), 8);
        assert_eq!(basis.num_qpoints(), 8);
    }

    #[test]
    fn test_3d_grad_of_linear_function() {
        let basis = LagrangeBasis::<f64>::new(3, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        let nodes = [-1.0_f64, 1.0];
        let mut u = Vec::new();
        for &z in &nodes {
            for &y in &nodes {
                for &x in &nodes {
                    u.push(2.0 * x - 3.0 * y + 4.0 * z + 1.0);
                }
            }
        }
        let mut grad = vec![0.0; 8 * 3];
        basis
            .apply(1, false, EvalMode::Grad, &u, &mut grad)
            .unwrap();
        for qpt in 0..8 {
            assert!((grad[qpt * 3] - 2.0).abs() < 1.0e-12);
            assert!((grad[qpt * 3 + 1] + 3.0).abs() < 1.0e-12);
            assert!((grad[qpt * 3 + 2] - 4.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn test_3d_interp_transpose_matches_naive() {
        let basis = LagrangeBasis::<f64>::new(3, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        let u = vec![0.5, -1.0, 2.0, 0.25, 1.5, -0.75, 0.1, 3.0];
        let mut v = vec![0.0; 8];
        basis.apply(1, true, EvalMode::Interp, &u, &mut v).unwrap();

        let mut expected = vec![0.0; 8];
        for pz in 0..2 {
            for py in 0..2 {
                for px in 0..2 {
                    let mut sum = 0.0;
                    for qz in 0..2 {
                        for qy in 0..2 {
                            for qx in 0..2 {
                                let qpt = (qz * 4) + (qy * 2) + qx;
                                sum += basis.interp[qx * 2 + px]
                                    * basis.interp[qy * 2 + py]
                                    * basis.interp[qz * 2 + pz]
                                    * u[qpt];
                            }
                        }
                    }
                    expected[pz * 4 + py * 2 + px] = sum;
                }
            }
        }

        for (got, want) in v.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-12);
        }
    }

    #[test]
    fn test_3d_grad_transpose_matches_naive() {
        let basis = LagrangeBasis::<f64>::new(3, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        let u = vec![
            0.5, -1.0, 2.0, 0.25, 1.5, -0.75, 0.1, 3.0, -2.0, 1.25, -0.5, 0.75, -1.0, 0.2, 1.8,
            2.2, -1.3, 0.4, 0.6, 0.7, -0.8, -0.9, 1.1, 2.4,
        ];
        let mut v = vec![0.0; 8];
        basis.apply(1, true, EvalMode::Grad, &u, &mut v).unwrap();

        let mut expected = vec![0.0; 8];
        for pz in 0..2 {
            for py in 0..2 {
                for px in 0..2 {
                    let mut sum = 0.0;
                    for qz in 0..2 {
                        for qy in 0..2 {
                            for qx in 0..2 {
                                let qpt = (qz * 4) + (qy * 2) + qx;
                                let ux = u[qpt * 3];
                                let uy = u[qpt * 3 + 1];
                                let uz = u[qpt * 3 + 2];
                                sum += basis.grad[qx * 2 + px]
                                    * basis.interp[qy * 2 + py]
                                    * basis.interp[qz * 2 + pz]
                                    * ux;
                                sum += basis.interp[qx * 2 + px]
                                    * basis.grad[qy * 2 + py]
                                    * basis.interp[qz * 2 + pz]
                                    * uy;
                                sum += basis.interp[qx * 2 + px]
                                    * basis.interp[qy * 2 + py]
                                    * basis.grad[qz * 2 + pz]
                                    * uz;
                            }
                        }
                    }
                    expected[pz * 4 + py * 2 + px] = sum;
                }
            }
        }

        for (got, want) in v.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-12);
        }
    }

    /// Discrete inner-product identity: ⟨D u, w⟩ = ⟨u, Dᵀ w⟩ for forward/transpose Div.
    #[test]
    fn test_div_forward_transpose_adjoint_identity() {
        let basis = LagrangeBasis::<f64>::new(2, 2, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        let num_elem = 1usize;
        let nd = num_elem * basis.num_dof() * basis.num_comp();
        let nq = num_elem * basis.num_qpoints();
        let u: Vec<f64> = (0..nd).map(|i| (i as f64) * 0.25 - 1.0).collect();
        let w: Vec<f64> = (0..nq).map(|i| (i as f64) * 0.1 + 0.3).collect();

        let mut div_u = vec![0.0_f64; nq];
        basis
            .apply(num_elem, false, EvalMode::Div, &u, &mut div_u)
            .unwrap();

        let mut dt_w = vec![0.0_f64; nd];
        basis
            .apply(num_elem, true, EvalMode::Div, &w, &mut dt_w)
            .unwrap();

        let lhs: f64 = u.iter().zip(dt_w.iter()).map(|(a, b)| a * b).sum();
        let rhs: f64 = div_u.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-9 * (1.0 + lhs.abs()));
    }

    #[test]
    fn test_curl_2d_forward_transpose_adjoint_identity() {
        let basis = LagrangeBasis::<f64>::new(2, 2, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        let num_elem = 1usize;
        let nd = num_elem * basis.num_dof() * basis.num_comp();
        let nq = num_elem * basis.num_qpoints();
        let u: Vec<f64> = (0..nd).map(|i| (i as f64) * 0.11 - 0.4).collect();
        let w: Vec<f64> = (0..nq).map(|i| (i as f64) * 0.07 + 0.2).collect();

        let mut curl_u = vec![0.0_f64; nq];
        basis
            .apply(num_elem, false, EvalMode::Curl, &u, &mut curl_u)
            .unwrap();

        let mut ct_w = vec![0.0_f64; nd];
        basis
            .apply(num_elem, true, EvalMode::Curl, &w, &mut ct_w)
            .unwrap();

        let lhs: f64 = u.iter().zip(ct_w.iter()).map(|(a, b)| a * b).sum();
        let rhs: f64 = curl_u.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-9 * (1.0 + lhs.abs()));
    }

    #[test]
    fn test_curl_3d_forward_transpose_adjoint_identity() {
        let basis = LagrangeBasis::<f64>::new(3, 3, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        let num_elem = 1usize;
        let nd = num_elem * basis.num_dof() * basis.num_comp();
        let nq = num_elem * basis.num_qpoints();
        let u: Vec<f64> = (0..nd).map(|i| (i as f64) * 0.09 - 0.5).collect();
        let w: Vec<f64> = (0..nq * 3).map(|i| (i as f64) * 0.03 + 0.1).collect();

        let mut curl_u = vec![0.0_f64; nq * 3];
        basis
            .apply(num_elem, false, EvalMode::Curl, &u, &mut curl_u)
            .unwrap();

        let mut ct_w = vec![0.0_f64; nd];
        basis
            .apply(num_elem, true, EvalMode::Curl, &w, &mut ct_w)
            .unwrap();

        let lhs: f64 = u.iter().zip(ct_w.iter()).map(|(a, b)| a * b).sum();
        let rhs: f64 = curl_u.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-8 * (1.0 + lhs.abs()));
    }

    #[test]
    fn test_tensor_contract_strided_matches_dense_reference() {
        let basis: [f64; 12] = [
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64, 7.0_f64, 8.0_f64, 9.0_f64,
            10.0_f64, 11.0_f64, 12.0_f64,
        ];
        let u: [f64; 8] = [
            2.0_f64, -99.0_f64, -1.0_f64, -99.0_f64, 0.5_f64, -99.0_f64, 3.0_f64, -99.0_f64,
        ];
        let mut v: [f64; 6] = [0.0_f64; 6];

        tensor_contract_strided(&basis, &u, 2, &mut v, 2, 3, 4, false);

        assert!((v[0] - 13.5_f64).abs() < 1.0e-12);
        assert!((v[2] - 31.5_f64).abs() < 1.0e-12);
        assert!((v[4] - 49.5_f64).abs() < 1.0e-12);

        let u_t: [f64; 6] = [1.0_f64, -99.0_f64, -2.0_f64, -99.0_f64, 0.5_f64, -99.0_f64];
        let mut v_t: [f64; 8] = [0.0_f64; 8];

        tensor_contract_strided(&basis, &u_t, 2, &mut v_t, 2, 3, 4, true);

        assert!((v_t[0] + 4.5_f64).abs() < 1.0e-12);
        assert!((v_t[2] + 5.0_f64).abs() < 1.0e-12);
        assert!((v_t[4] + 5.5_f64).abs() < 1.0e-12);
        assert!((v_t[6] + 6.0_f64).abs() < 1.0e-12);
    }

    #[test]
    fn weight_transpose_matches_interp_transpose_scalar() {
        let b = LagrangeBasis::<f64>::new(1, 1, 2, 2, reed_core::QuadMode::Gauss).unwrap();
        assert_eq!(b.num_comp(), 1);
        let ne = 2usize;
        let u: Vec<f64> = (0..ne * b.num_qpoints())
            .map(|i| 0.1 * (i + 1) as f64)
            .collect();
        let mut v_w = vec![0.0_f64; ne * b.num_dof() * b.num_comp()];
        let mut v_i = vec![0.0_f64; ne * b.num_dof() * b.num_comp()];
        b.apply(ne, true, EvalMode::Weight, &u, &mut v_w).unwrap();
        b.apply(ne, true, EvalMode::Interp, &u, &mut v_i).unwrap();
        assert_eq!(v_w.len(), v_i.len());
        for i in 0..v_w.len() {
            assert!(
                (v_w[i] - v_i[i]).abs() < 1e-14,
                "i={i} w={} i={}",
                v_w[i],
                v_i[i]
            );
        }
    }
}
