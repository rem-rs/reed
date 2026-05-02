//! Simplex basis functions for segments, triangles, and tetrahedra.
//!
//! Implements [`BasisTrait`] for H1-conforming Lagrange bases on simplex
//! reference elements:
//!
//! | Type | Topology | DOFs | Exact up to polynomial degree |
//! |------|----------|------|-------------------------------|
//! | P1 segment | Line | 2 | linear |
//! | P2 segment | Line | 3 | quadratic |
//! | P3 segment | Line | 4 | cubic |
//! | P1 triangle | Tri3 | 3 | linear |
//! | P2 triangle | Tri6 | 6 | quadratic |
//! | P3 triangle | Tri10 | 10 | cubic |
//! | P1 tet | Tet4 | 4 | linear |
//! | P2 tet | Tet10 | 10 | quadratic |
//! | P3 tet | Tet20 | 20 | cubic |
//!
//! ## Reference elements
//!
//! **Segment** — reference interval [0,1] (1D simplex).
//!
//! **Triangle** — vertices (0,0), (1,0), (0,1).
//!
//! **Tetrahedron** — vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1).
//!
//! ## Quadrature rules
//!
//! Pass the desired number of quadrature points via the `q` constructor argument.
//!
//! | `q` (triangle) | degree exact | `q` (tet) | degree exact |
//! |----------------|--------------|-----------|--------------|
//! | 1 | 1 | 1 | 1 |
//! | 3 | 2 | 4 | 2 |
//! | 4 | 3 | 5 | 3 |
//! | 6 | 4 | — | — |
//! | 7 | 5 | — | — |
//!
//! **P3 triangle** needs at least `q = 4` (degree-3 exact) for typical variational integrals; `q = 6`
//! integrates degree 4 (e.g. `∇P3 · ∇P3`). **P3 tet** uses the same `q` table; for degree-5 exactness
//! on tets use `q = 5` (Keast) or higher-order rules as added in the future.
//!
//! ## Memory layout (matches [`LagrangeBasis`](super::basis_lagrange::LagrangeBasis))
//!
//! * `interp`  — row-major `[nqpts × num_dof]`
//! * `grad`    — row-major `[nqpts × num_dof × dim]`,
//!               stored as `[qpt][dof][d]` ↔ index `(qpt*num_dof + dof)*dim + d`
//!
//! **Element buffers passed to `apply`:**
//!
//! * Forward interp : `u=[ncomp × num_dof]`, `v=[nqpts × ncomp]`
//! * Forward grad   : `u=[ncomp × num_dof]`, `v=[nqpts × ncomp × dim]`
//!   (`qcomp = ncomp*dim`, layout `v[qpt*qcomp + comp*dim + d]`)
//! * Weight         : `v=[nqpts]` (per element, repeated `num_elem` times)

#[path = "simplex_p3_data.rs"]
mod simplex_p3_data;

use reed_core::{
    basis::BasisTrait,
    enums::{ElemTopology, EvalMode},
    error::{ReedError, ReedResult},
    scalar::Scalar,
};
use simplex_p3_data::{TET_P3_COEFF, TET_P3_EXP, TRI_P3_COEFF};

use super::basis_lagrange::gauss_quadrature;

#[cfg(feature = "parallel")]
const PAR_MIN_ELEMS_PER_TASK: usize = 128;

// ── SimplexBasis ──────────────────────────────────────────────────────────────

/// H1 Lagrange basis on segment, triangle, or tetrahedron reference elements.
pub struct SimplexBasis<T: Scalar> {
    #[allow(dead_code)]
    topo: ElemTopology,
    dim: usize,
    ncomp: usize,
    num_dof: usize,
    num_qpoints: usize,
    /// Quadrature point coordinates, row-major `[nqpts × dim]`.
    q_ref: Vec<T>,
    /// Quadrature weights, length `nqpts`.
    weights: Vec<T>,
    /// Interpolation matrix, row-major `[nqpts × num_dof]`.
    interp: Vec<T>,
    /// Gradient tensor, layout `[nqpts × num_dof × dim]`.
    grad: Vec<T>,
}

impl<T: Scalar> SimplexBasis<T> {
    /// Construct a simplex basis.
    ///
    /// # Parameters
    /// * `topo`  — `ElemTopology::Line`, `Triangle`, or `Tet` (other simplex topologies are not implemented here).
    /// * `poly`  — polynomial order (1 = P1, 2 = P2, 3 = P3).
    /// * `ncomp` — number of field components (1 for scalar problems).
    /// * `q`     — number of quadrature points (see module-level table for valid
    ///             values per topology).
    ///
    /// # Errors
    /// Returns `ReedError::Basis` for unsupported topology/poly/q combinations.
    pub fn new(topo: ElemTopology, poly: usize, ncomp: usize, q: usize) -> ReedResult<Self> {
        let (dim, num_dof) = match (topo, poly) {
            (ElemTopology::Line, 1) => (1, 2),
            (ElemTopology::Line, 2) => (1, 3),
            (ElemTopology::Line, 3) => (1, 4),
            (ElemTopology::Triangle, 1) => (2, 3),
            (ElemTopology::Triangle, 2) => (2, 6),
            (ElemTopology::Triangle, 3) => (2, 10),
            (ElemTopology::Tet, 1) => (3, 4),
            (ElemTopology::Tet, 2) => (3, 10),
            (ElemTopology::Tet, 3) => (3, 20),
            _ => {
                if matches!(topo, ElemTopology::Pyramid | ElemTopology::Prism) {
                    return Err(ReedError::Basis(format!(
                        "SimplexBasis: {:?} not implemented (requires collapsed-coordinate (Pyramid) or tensor×simplex (Prism/Wedge) transforms)",
                        topo
                    )));
                }
                return Err(ReedError::Basis(format!(
                    "SimplexBasis: unsupported (topology={:?}, poly={})",
                    topo, poly
                )));
            }
        };

        // Quadrature rule ---------------------------------------------------
        let (q_ref_f64, weights_f64) = match topo {
            ElemTopology::Line => line_quadrature(q)?,
            ElemTopology::Triangle => tri_quadrature(q)?,
            ElemTopology::Tet => tet_quadrature(q)?,
            _ => {
                return Err(ReedError::Basis(format!(
                    "SimplexBasis: unsupported topology {:?}",
                    topo
                )))
            }
        };
        let num_qpoints = q_ref_f64.len() / dim;

        // Convert to target scalar type.
        let q_ref: Vec<T> = q_ref_f64
            .iter()
            .map(|&v| to_t::<T>(v))
            .collect::<ReedResult<_>>()?;
        let weights: Vec<T> = weights_f64
            .iter()
            .map(|&v| to_t::<T>(v))
            .collect::<ReedResult<_>>()?;

        // Interpolation & gradient tables -----------------------------------
        let qpts: Vec<[f64; 3]> = (0..num_qpoints)
            .map(|qi| {
                let mut pt = [0.0f64; 3];
                for d in 0..dim {
                    pt[d] = q_ref_f64[qi * dim + d];
                }
                pt
            })
            .collect();

        let mut interp = vec![0.0f64; num_qpoints * num_dof];
        let mut grad = vec![0.0f64; num_qpoints * num_dof * dim];

        for (qi, pt) in qpts.iter().enumerate() {
            let (phi, dphi) = match (topo, poly) {
                (ElemTopology::Line, 1) => line_p1_basis(pt[0]),
                (ElemTopology::Line, 2) => line_p2_basis(pt[0]),
                (ElemTopology::Line, 3) => line_p3_basis(pt[0]),
                (ElemTopology::Triangle, 1) => tri_p1_basis(pt[0], pt[1]),
                (ElemTopology::Triangle, 2) => tri_p2_basis(pt[0], pt[1]),
                (ElemTopology::Triangle, 3) => tri_p3_basis(pt[0], pt[1]),
                (ElemTopology::Tet, 1) => tet_p1_basis(pt[0], pt[1], pt[2]),
                (ElemTopology::Tet, 2) => tet_p2_basis(pt[0], pt[1], pt[2]),
                (ElemTopology::Tet, 3) => tet_p3_basis(pt[0], pt[1], pt[2]),
                _ => unreachable!(),
            };
            for dof in 0..num_dof {
                interp[qi * num_dof + dof] = phi[dof];
                for d in 0..dim {
                    grad[(qi * num_dof + dof) * dim + d] = dphi[dof * dim + d];
                }
            }
        }

        let interp_t: Vec<T> = interp
            .iter()
            .map(|&v| to_t::<T>(v))
            .collect::<ReedResult<_>>()?;
        let grad_t: Vec<T> = grad
            .iter()
            .map(|&v| to_t::<T>(v))
            .collect::<ReedResult<_>>()?;

        Ok(Self {
            topo,
            dim,
            ncomp,
            num_dof,
            num_qpoints,
            q_ref,
            weights,
            interp: interp_t,
            grad: grad_t,
        })
    }

    /// Row-major interpolation operator `[num_qpoints × num_dof]` (same packing as `LagrangeBasis`).
    pub fn interp_matrix(&self) -> &[T] {
        &self.interp
    }

    /// Row-major gradient tensor `(qpt * num_dof + dof) * dim + d` (same packing as `LagrangeBasis`).
    pub fn grad_matrix(&self) -> &[T] {
        &self.grad
    }

    // ── accessor helpers ───────────────────────────────────────────────────

    #[inline]
    fn interp_val(&self, qpt: usize, dof: usize) -> T {
        self.interp[qpt * self.num_dof + dof]
    }

    /// `grad[(qpt * num_dof + dof) * dim + d]`
    #[inline]
    fn grad_val(&self, qpt: usize, dof: usize, d: usize) -> T {
        self.grad[(qpt * self.num_dof + dof) * self.dim + d]
    }

    // ── element-level apply ────────────────────────────────────────────────

    fn apply_interp_elem(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        for comp in 0..self.ncomp {
            if transpose {
                // v: [ncomp × num_dof],  u: [nqpts × ncomp]
                for dof in 0..self.num_dof {
                    let mut sum = T::ZERO;
                    for qpt in 0..self.num_qpoints {
                        sum += self.interp_val(qpt, dof) * u_elem[qpt * self.ncomp + comp];
                    }
                    v_elem[comp * self.num_dof + dof] += sum;
                }
            } else {
                // u: [ncomp × num_dof],  v: [nqpts × ncomp]
                for qpt in 0..self.num_qpoints {
                    let mut sum = T::ZERO;
                    for dof in 0..self.num_dof {
                        sum += self.interp_val(qpt, dof) * u_elem[comp * self.num_dof + dof];
                    }
                    v_elem[qpt * self.ncomp + comp] = sum;
                }
            }
        }
    }

    fn apply_grad_elem(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        let qcomp = self.ncomp * self.dim;
        for comp in 0..self.ncomp {
            if transpose {
                // u: [nqpts × ncomp × dim],  v: [ncomp × num_dof]
                for dof in 0..self.num_dof {
                    let mut sum = T::ZERO;
                    for qpt in 0..self.num_qpoints {
                        for d in 0..self.dim {
                            sum += self.grad_val(qpt, dof, d)
                                * u_elem[qpt * qcomp + comp * self.dim + d];
                        }
                    }
                    v_elem[comp * self.num_dof + dof] += sum;
                }
            } else {
                // u: [ncomp × num_dof],  v: [nqpts × ncomp × dim]
                for qpt in 0..self.num_qpoints {
                    for d in 0..self.dim {
                        let mut sum = T::ZERO;
                        for dof in 0..self.num_dof {
                            sum += self.grad_val(qpt, dof, d) * u_elem[comp * self.num_dof + dof];
                        }
                        v_elem[qpt * qcomp + comp * self.dim + d] = sum;
                    }
                }
            }
        }
    }
}

// ── BasisTrait impl ───────────────────────────────────────────────────────────

impl<T: Scalar> BasisTrait<T> for SimplexBasis<T> {
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
                let in_stride = if transpose {
                    self.num_qpoints * self.ncomp
                } else {
                    self.num_dof * self.ncomp
                };
                let out_stride = if transpose {
                    self.num_dof * self.ncomp
                } else {
                    self.num_qpoints * self.ncomp
                };
                check_sizes(u, in_stride * num_elem, v, out_stride * num_elem, "interp")?;
                // zero output for accumulation in transpose path
                if transpose {
                    v.fill(T::ZERO);
                }
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
                let in_stride = if transpose {
                    self.num_qpoints * qcomp
                } else {
                    self.num_dof * self.ncomp
                };
                let out_stride = if transpose {
                    self.num_dof * self.ncomp
                } else {
                    self.num_qpoints * qcomp
                };
                check_sizes(u, in_stride * num_elem, v, out_stride * num_elem, "grad")?;
                if transpose {
                    v.fill(T::ZERO);
                }
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
                        "EvalMode::Div requires ncomp == dim for SimplexBasis".into(),
                    ));
                }
                let qcomp = self.ncomp * self.dim;
                if transpose {
                    let in_size = num_elem * self.num_qpoints;
                    let out_size = num_elem * self.num_dof * self.ncomp;
                    check_sizes(u, in_size, v, out_size, "div transpose")?;
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
                    check_sizes(u, in_size, v, out_size, "div")?;
                    let mut grad_buf = vec![T::ZERO; num_elem * self.num_qpoints * qcomp];
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
                let qcomp = self.ncomp * self.dim;
                match (self.dim, self.ncomp) {
                    (2, 2) => {
                        if transpose {
                            let in_size = num_elem * self.num_qpoints;
                            let out_size = num_elem * self.num_dof * self.ncomp;
                            check_sizes(u, in_size, v, out_size, "curl transpose")?;
                            let mut grad_buf = vec![T::ZERO; num_elem * self.num_qpoints * qcomp];
                            for e in 0..num_elem {
                                for iq in 0..self.num_qpoints {
                                    let w = u[e * self.num_qpoints + iq];
                                    let base = (e * self.num_qpoints + iq) * qcomp;
                                    grad_buf[base + 1] -= w;
                                    grad_buf[base + 2] += w;
                                }
                            }
                            self.apply(num_elem, true, EvalMode::Grad, &grad_buf, v)?;
                        } else {
                            let in_size = num_elem * self.num_dof * self.ncomp;
                            let out_size = num_elem * self.num_qpoints;
                            check_sizes(u, in_size, v, out_size, "curl")?;
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
                            check_sizes(u, in_size, v, out_size, "curl transpose")?;
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
                            check_sizes(u, in_size, v, out_size, "curl")?;
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
                            "EvalMode::Curl requires (dim, ncomp) = (2, 2) or (3, 3) for SimplexBasis"
                                .into(),
                        ));
                    }
                }
            }
            other => {
                return Err(ReedError::Basis(format!(
                    "SimplexBasis: eval mode {:?} not implemented",
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

    fn face_q_weights(&self, local_face: usize) -> Option<Vec<T>> {
        crate::face_quadrature::face_quadrature_simplex(self, local_face)
            .map(|(_, w)| w)
            .ok()
    }

    fn face_q_ref(&self, local_face: usize) -> Option<Vec<T>> {
        crate::face_quadrature::face_quadrature_simplex(self, local_face)
            .map(|(qr, _)| qr)
            .ok()
    }
}

// ── shape functions ───────────────────────────────────────────────────────────

/// P1 line on [0,1]: nodes at 0 and 1.
fn line_p1_basis(t: f64) -> (Vec<f64>, Vec<f64>) {
    let phi = vec![1.0 - t, t];
    let dphi = vec![-1.0, 1.0];
    (phi, dphi)
}

/// P2 line on [0,1]: nodes at 0, ½, 1.
fn line_p2_basis(t: f64) -> (Vec<f64>, Vec<f64>) {
    let phi = vec![
        2.0 * (t - 0.5) * (t - 1.0),
        -4.0 * t * (t - 1.0),
        2.0 * t * (t - 0.5),
    ];
    let dphi = vec![2.0 * (2.0 * t - 1.5), -8.0 * t + 4.0, 4.0 * t - 1.0];
    (phi, dphi)
}

/// P3 line on [0,1]: uniform nodes 0, ⅓, ⅔, 1.
fn line_p3_basis(t: f64) -> (Vec<f64>, Vec<f64>) {
    let x = [0.0_f64, 1.0 / 3.0, 2.0 / 3.0, 1.0];
    let n = x.len();
    let mut phi = vec![0.0; n];
    let mut dphi = vec![0.0; n];
    for i in 0..n {
        let mut l = 1.0;
        for j in 0..n {
            if i == j {
                continue;
            }
            l *= (t - x[j]) / (x[i] - x[j]);
        }
        phi[i] = l;
        let mut s = 0.0;
        for j in 0..n {
            if i == j {
                continue;
            }
            s += 1.0 / (t - x[j]);
        }
        dphi[i] = l * s;
    }
    (phi, dphi)
}

/// P1 triangle basis: φ and ∇φ at (x,y).
///
/// Returns `(phi[3], dphi[3*2])` where `dphi[i*2+d]` = ∂φᵢ/∂xd.
/// Gradients are constant (independent of x,y).
fn tri_p1_basis(x: f64, y: f64) -> (Vec<f64>, Vec<f64>) {
    let _ = (x, y); // gradients are constant
                    // φ₀ = 1-x-y,  φ₁ = x,  φ₂ = y
    let phi = vec![1.0 - x - y, x, y];
    // ∇φ₀ = (-1,-1),  ∇φ₁ = (1,0),  ∇φ₂ = (0,1)
    let dphi = vec![
        -1.0, -1.0, // dof 0
        1.0, 0.0, // dof 1
        0.0, 1.0, // dof 2
    ];
    (phi, dphi)
}

/// P2 triangle basis: φ and ∇φ at (x,y).
///
/// Node ordering (standard serendipity):
/// ```text
///  2
///  |\\
///  5  4
///  |    \\
///  0--3--1
/// ```
/// DOF 0=(0,0), 1=(1,0), 2=(0,1), 3=(½,0), 4=(½,½), 5=(0,½).
fn tri_p2_basis(x: f64, y: f64) -> (Vec<f64>, Vec<f64>) {
    let l0 = 1.0 - x - y;
    let l1 = x;
    let l2 = y;
    let phi = vec![
        l0 * (2.0 * l0 - 1.0), // 0: vertex (0,0)
        l1 * (2.0 * l1 - 1.0), // 1: vertex (1,0)
        l2 * (2.0 * l2 - 1.0), // 2: vertex (0,1)
        4.0 * l0 * l1,         // 3: midpoint (½,0)
        4.0 * l1 * l2,         // 4: midpoint (½,½)
        4.0 * l0 * l2,         // 5: midpoint (0,½)
    ];
    // ∂φ/∂x (d=0), ∂φ/∂y (d=1)
    // dl0/dx=-1, dl0/dy=-1; dl1/dx=1, dl1/dy=0; dl2/dx=0, dl2/dy=1
    let dphi = vec![
        // dof 0: φ = l0*(2l0-1)  → ∂/∂x = (-1)*(4l0-1), ∂/∂y = (-1)*(4l0-1)
        -(4.0 * l0 - 1.0),
        -(4.0 * l0 - 1.0),
        // dof 1: φ = l1*(2l1-1)  → ∂/∂x = 4l1-1, ∂/∂y = 0
        4.0 * l1 - 1.0,
        0.0,
        // dof 2: φ = l2*(2l2-1)  → ∂/∂x = 0, ∂/∂y = 4l2-1
        0.0,
        4.0 * l2 - 1.0,
        // dof 3: φ = 4*l0*l1     → ∂/∂x = 4*(l0-l1), ∂/∂y = -4*l1
        4.0 * (l0 - l1),
        -4.0 * l1,
        // dof 4: φ = 4*l1*l2     → ∂/∂x = 4*l2, ∂/∂y = 4*l1
        4.0 * l2,
        4.0 * l1,
        // dof 5: φ = 4*l0*l2     → ∂/∂x = -4*l2, ∂/∂y = 4*(l0-l2)
        -4.0 * l2,
        4.0 * (l0 - l2),
    ];
    (phi, dphi)
}

/// P3 triangle nodal Lagrange basis (10 lattice nodes, degree-3 polynomials in `x`,`y`).
fn tri_p3_basis(x: f64, y: f64) -> (Vec<f64>, Vec<f64>) {
    let (m, mx, my) = tri_p3_mono_val_grad(x, y);
    let mut phi = vec![0.0_f64; 10];
    let mut dphi = vec![0.0_f64; 20];
    for k in 0..10 {
        let mut pk = 0.0_f64;
        let mut gx = 0.0_f64;
        let mut gy = 0.0_f64;
        for j in 0..10 {
            let c = TRI_P3_COEFF[j][k];
            pk += c * m[j];
            gx += c * mx[j];
            gy += c * my[j];
        }
        phi[k] = pk;
        dphi[k * 2] = gx;
        dphi[k * 2 + 1] = gy;
    }
    (phi, dphi)
}

fn tri_p3_mono_val_grad(x: f64, y: f64) -> ([f64; 10], [f64; 10], [f64; 10]) {
    let x2 = x * x;
    let y2 = y * y;
    let m = [1.0, x, y, x2, x * y, y2, x2 * x, x2 * y, x * y2, y2 * y];
    let mx = [
        0.0,
        1.0,
        0.0,
        2.0 * x,
        y,
        0.0,
        3.0 * x2,
        2.0 * x * y,
        y2,
        0.0,
    ];
    let my = [
        0.0,
        0.0,
        1.0,
        0.0,
        x,
        2.0 * y,
        0.0,
        x2,
        2.0 * x * y,
        3.0 * y2,
    ];
    (m, mx, my)
}

/// P1 tet basis: φ and ∇φ at (x,y,z).
///
/// DOF 0=(0,0,0), 1=(1,0,0), 2=(0,1,0), 3=(0,0,1).
fn tet_p1_basis(x: f64, y: f64, z: f64) -> (Vec<f64>, Vec<f64>) {
    let phi = vec![1.0 - x - y - z, x, y, z];
    // ∇φ₀=(-1,-1,-1), ∇φ₁=(1,0,0), ∇φ₂=(0,1,0), ∇φ₃=(0,0,1)
    let dphi = vec![
        -1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let _ = (x, y, z);
    (phi, dphi)
}

/// P2 tet basis: φ and ∇φ at (x,y,z).
///
/// 10 DOFs: 4 vertices + 6 edge midpoints.
///
/// Vertex ordering: V0=(0,0,0), V1=(1,0,0), V2=(0,1,0), V3=(0,0,1).
/// Edge midpoints:  E4=(½,0,0), E5=(½,½,0), E6=(0,½,0),
///                  E7=(0,0,½), E8=(½,0,½), E9=(0,½,½)...
///
/// Wait, standard ordering for Tet10 edges:
/// E4 = midpoint V0-V1 = (½,0,0)
/// E5 = midpoint V1-V2 = (½,½,0)
/// E6 = midpoint V0-V2 = (0,½,0)
/// E7 = midpoint V0-V3 = (0,0,½)
/// E8 = midpoint V1-V3 = (½,0,½)
/// E9 = midpoint V2-V3 = (0,½,½)
fn tet_p2_basis(x: f64, y: f64, z: f64) -> (Vec<f64>, Vec<f64>) {
    let l0 = 1.0 - x - y - z;
    let l1 = x;
    let l2 = y;
    let l3 = z;
    let phi = vec![
        l0 * (2.0 * l0 - 1.0), // 0: V0
        l1 * (2.0 * l1 - 1.0), // 1: V1
        l2 * (2.0 * l2 - 1.0), // 2: V2
        l3 * (2.0 * l3 - 1.0), // 3: V3
        4.0 * l0 * l1,         // 4: E4 midpoint V0-V1
        4.0 * l1 * l2,         // 5: E5 midpoint V1-V2
        4.0 * l0 * l2,         // 6: E6 midpoint V0-V2
        4.0 * l0 * l3,         // 7: E7 midpoint V0-V3
        4.0 * l1 * l3,         // 8: E8 midpoint V1-V3
        4.0 * l2 * l3,         // 9: E9 midpoint V2-V3
    ];
    // dl0=(−1,−1,−1), dl1=(1,0,0), dl2=(0,1,0), dl3=(0,0,1)
    let dphi = vec![
        // dof 0: ∂/∂x = -(4l0-1), ∂/∂y = -(4l0-1), ∂/∂z = -(4l0-1)
        -(4.0 * l0 - 1.0),
        -(4.0 * l0 - 1.0),
        -(4.0 * l0 - 1.0),
        // dof 1: ∂/∂x = 4l1-1, ∂/∂y = 0, ∂/∂z = 0
        4.0 * l1 - 1.0,
        0.0,
        0.0,
        // dof 2: ∂/∂x = 0, ∂/∂y = 4l2-1, ∂/∂z = 0
        0.0,
        4.0 * l2 - 1.0,
        0.0,
        // dof 3: ∂/∂x = 0, ∂/∂y = 0, ∂/∂z = 4l3-1
        0.0,
        0.0,
        4.0 * l3 - 1.0,
        // dof 4: φ=4l0l1 → ∂/∂x=4(l0-l1), ∂/∂y=-4l1, ∂/∂z=-4l1
        4.0 * (l0 - l1),
        -4.0 * l1,
        -4.0 * l1,
        // dof 5: φ=4l1l2 → ∂/∂x=4l2, ∂/∂y=4l1, ∂/∂z=0
        4.0 * l2,
        4.0 * l1,
        0.0,
        // dof 6: φ=4l0l2 → ∂/∂x=-4l2, ∂/∂y=4(l0-l2), ∂/∂z=-4l2
        -4.0 * l2,
        4.0 * (l0 - l2),
        -4.0 * l2,
        // dof 7: φ=4l0l3 → ∂/∂x=-4l3, ∂/∂y=-4l3, ∂/∂z=4(l0-l3)
        -4.0 * l3,
        -4.0 * l3,
        4.0 * (l0 - l3),
        // dof 8: φ=4l1l3 → ∂/∂x=4l3, ∂/∂y=0, ∂/∂z=4l1
        4.0 * l3,
        0.0,
        4.0 * l1,
        // dof 9: φ=4l2l3 → ∂/∂x=0, ∂/∂y=4l3, ∂/∂z=4l2
        0.0,
        4.0 * l3,
        4.0 * l2,
    ];
    (phi, dphi)
}

/// P3 tet nodal Lagrange basis (20 lattice nodes).
fn tet_p3_basis(x: f64, y: f64, z: f64) -> (Vec<f64>, Vec<f64>) {
    let (m, mx, my, mz) = tet_p3_mono_val_grad(x, y, z);
    let mut phi = vec![0.0_f64; 20];
    let mut dphi = vec![0.0_f64; 60];
    for k in 0..20 {
        let mut pk = 0.0_f64;
        let mut gx = 0.0_f64;
        let mut gy = 0.0_f64;
        let mut gz = 0.0_f64;
        for j in 0..20 {
            let c = TET_P3_COEFF[j][k];
            pk += c * m[j];
            gx += c * mx[j];
            gy += c * my[j];
            gz += c * mz[j];
        }
        phi[k] = pk;
        dphi[k * 3] = gx;
        dphi[k * 3 + 1] = gy;
        dphi[k * 3 + 2] = gz;
    }
    (phi, dphi)
}

fn tet_p3_mono_val_grad(x: f64, y: f64, z: f64) -> ([f64; 20], [f64; 20], [f64; 20], [f64; 20]) {
    let mut m = [0.0_f64; 20];
    let mut mx = [0.0_f64; 20];
    let mut my = [0.0_f64; 20];
    let mut mz = [0.0_f64; 20];
    for (j, &(i, iy, iz)) in TET_P3_EXP.iter().enumerate() {
        let i = i as i32;
        let iy = iy as i32;
        let iz = iz as i32;
        let xv = x.powi(i);
        let yv = y.powi(iy);
        let zv = z.powi(iz);
        m[j] = xv * yv * zv;
        mx[j] = if i > 0 {
            (i as f64) * x.powi(i - 1) * y.powi(iy) * z.powi(iz)
        } else {
            0.0
        };
        my[j] = if iy > 0 {
            (iy as f64) * x.powi(i) * y.powi(iy - 1) * z.powi(iz)
        } else {
            0.0
        };
        mz[j] = if iz > 0 {
            (iz as f64) * x.powi(i) * y.powi(iy) * z.powi(iz - 1)
        } else {
            0.0
        };
    }
    (m, mx, my, mz)
}

// ── quadrature rules ─────────────────────────────────────────────────────────

/// Gauss quadrature on reference segment [0,1] (`q` points; maps 1D Legendre Gauss from [-1,1]).
fn line_quadrature(q: usize) -> ReedResult<(Vec<f64>, Vec<f64>)> {
    if q < 1 {
        return Err(ReedError::Basis(format!(
            "line quadrature: unsupported q={q} (need >= 1)"
        )));
    }
    let (xi, wi) = gauss_quadrature(q)?;
    let mut pts = Vec::with_capacity(q);
    let mut wts = Vec::with_capacity(q);
    for i in 0..q {
        let t = 0.5 * (xi[i] + 1.0);
        let w = 0.5 * wi[i];
        pts.push(t);
        wts.push(w);
    }
    Ok((pts, wts))
}

/// Triangle Gauss quadrature rules (reference triangle area = 1/2).
///
/// Returns `(ref_coords, weights)` where `ref_coords` is row-major `[q×2]`.
///
/// | q | degree exact |
/// |---|--------------|
/// | 1 | 1 |
/// | 3 | 2 |
/// | 4 | 3 |
/// | 6 | 4 |
/// | 7 | 5 |
pub(crate) fn tri_quadrature(q: usize) -> ReedResult<(Vec<f64>, Vec<f64>)> {
    match q {
        1 => {
            // Centroid rule (degree 1 exact)
            let pts = vec![1.0 / 3.0, 1.0 / 3.0];
            let wts = vec![0.5];
            Ok((pts, wts))
        }
        3 => {
            // Degree 2 exact (Dunavant / midpoint rule)
            let a = 1.0 / 6.0;
            let b = 2.0 / 3.0;
            let pts = vec![a, a, b, a, a, b];
            let wts = vec![1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0];
            Ok((pts, wts))
        }
        4 => {
            // Degree 3 exact (Dunavant 4-point, one negative weight)
            let pts = vec![1.0 / 3.0, 1.0 / 3.0, 0.2, 0.2, 0.6, 0.2, 0.2, 0.6];
            let wts = vec![-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0];
            Ok((pts, wts))
        }
        6 => {
            // Degree 4 exact (Dunavant 6-point)
            // Group 1 (a1 symmetry, 3 points)
            let a1 = 0.445948490915965_f64;
            let b1 = 0.108103018168070_f64;
            let w1 = 0.111690794839005_f64;
            // Group 2 (a2 symmetry, 3 points)
            let a2 = 0.091576213509771_f64;
            let b2 = 0.816847572980459_f64;
            let w2 = 0.054975871827661_f64;
            let pts = vec![a1, a1, b1, a1, a1, b1, a2, a2, b2, a2, a2, b2];
            let wts = vec![w1, w1, w1, w2, w2, w2];
            Ok((pts, wts))
        }
        7 => {
            // Degree 5 exact (Dunavant 7-point).
            // Weights are for reference triangle with area = 0.5.
            // Closed-form via sqrt(15):
            //   w_center = 9/40 * (1/2) — reference area factor
            //   a_inner  = (6 - sqrt(15)) / 21,  w_inner = (155 - sqrt(15)) / 1200
            //   a_outer  = (6 + sqrt(15)) / 21,  w_outer = (155 + sqrt(15)) / 1200
            // (Dunavant 1985; weights sum to 1 on unit-area triangle, halved here.)
            let sq15 = 15.0_f64.sqrt();
            let w1 = 9.0 / 80.0; // 9/40 * 1/2
            let a_inner = (6.0 - sq15) / 21.0;
            let b_inner = 1.0 - 2.0 * a_inner;
            let w_inner = (155.0 - sq15) / 2400.0;
            let a_outer = (6.0 + sq15) / 21.0;
            let b_outer = 1.0 - 2.0 * a_outer;
            let w_outer = (155.0 + sq15) / 2400.0;
            let pts = vec![
                1.0 / 3.0,
                1.0 / 3.0,
                a_outer,
                a_outer,
                b_outer,
                a_outer,
                a_outer,
                b_outer,
                a_inner,
                a_inner,
                b_inner,
                a_inner,
                a_inner,
                b_inner,
            ];
            let wts = vec![w1, w_outer, w_outer, w_outer, w_inner, w_inner, w_inner];
            Ok((pts, wts))
        }
        _ => Err(ReedError::Basis(format!(
            "triangle quadrature: unsupported q={q} (valid: 1,3,4,6,7)"
        ))),
    }
}

/// Tetrahedron Gauss quadrature rules (reference tet volume = 1/6).
///
/// Returns `(ref_coords, weights)` where `ref_coords` is row-major `[q×3]`.
///
/// | q | degree exact |
/// |---|--------------|
/// | 1 | 1 |
/// | 4 | 2 |
/// | 5 | 3 |
pub(crate) fn tet_quadrature(q: usize) -> ReedResult<(Vec<f64>, Vec<f64>)> {
    match q {
        1 => {
            let pts = vec![0.25, 0.25, 0.25];
            let wts = vec![1.0 / 6.0];
            Ok((pts, wts))
        }
        4 => {
            // Degree 2 exact (symmetric 4-point rule)
            // a = (5 - sqrt(5)) / 20,  b = (5 + 3*sqrt(5)) / 20
            let sq5 = 5.0_f64.sqrt();
            let a = (5.0 - sq5) / 20.0; // ≈ 0.1381966
            let b = (5.0 + 3.0 * sq5) / 20.0; // ≈ 0.5854102
            let w = 1.0 / 24.0;
            let pts = vec![a, a, a, b, a, a, a, b, a, a, a, b];
            let wts = vec![w, w, w, w];
            Ok((pts, wts))
        }
        5 => {
            // Degree 3 exact (Keast 5-point rule with negative weight)
            // Point at centroid with negative weight
            let pts = vec![
                0.25,
                0.25,
                0.25,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                0.5,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                0.5,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                0.5,
            ];
            let wts = vec![
                -4.0 / 30.0,
                9.0 / 120.0,
                9.0 / 120.0,
                9.0 / 120.0,
                9.0 / 120.0,
            ];
            Ok((pts, wts))
        }
        _ => Err(ReedError::Basis(format!(
            "tet quadrature: unsupported q={q} (valid: 1,4,5)"
        ))),
    }
}

// ── utilities ─────────────────────────────────────────────────────────────────

fn to_t<T: Scalar>(v: f64) -> ReedResult<T> {
    T::from(v)
        .ok_or_else(|| ReedError::Basis(format!("SimplexBasis: failed to convert {v} to scalar")))
}

fn check_sizes<T>(
    u: &[T],
    u_expected: usize,
    v: &[T],
    v_expected: usize,
    mode: &str,
) -> ReedResult<()> {
    if u.len() != u_expected || v.len() != v_expected {
        return Err(ReedError::Basis(format!(
            "SimplexBasis {mode} size mismatch: \
             input {} (expected {}), output {} (expected {})",
            u.len(),
            u_expected,
            v.len(),
            v_expected
        )));
    }
    Ok(())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    // ── partition-of-unity ────────────────────────────────────────────────

    #[test]
    fn tri_p1_partition_of_unity() {
        for &(x, y) in &[(0.1, 0.2), (0.5, 0.3), (1.0 / 3.0, 1.0 / 3.0)] {
            let (phi, _) = tri_p1_basis(x, y);
            let sum: f64 = phi.iter().sum();
            assert!(
                (sum - 1.0).abs() < TOL,
                "PoU failed at ({x},{y}): sum={sum}"
            );
        }
    }

    #[test]
    fn tri_p2_partition_of_unity() {
        for &(x, y) in &[(0.1, 0.2), (0.5, 0.25), (0.25, 0.5)] {
            let (phi, _) = tri_p2_basis(x, y);
            let sum: f64 = phi.iter().sum();
            assert!(
                (sum - 1.0).abs() < TOL,
                "PoU failed at ({x},{y}): sum={sum}"
            );
        }
    }

    #[test]
    fn tet_p1_partition_of_unity() {
        for &(x, y, z) in &[(0.1, 0.2, 0.3), (0.25, 0.25, 0.25)] {
            let (phi, _) = tet_p1_basis(x, y, z);
            let sum: f64 = phi.iter().sum();
            assert!(
                (sum - 1.0).abs() < TOL,
                "PoU failed at ({x},{y},{z}): sum={sum}"
            );
        }
    }

    #[test]
    fn tet_p2_partition_of_unity() {
        for &(x, y, z) in &[(0.1, 0.2, 0.1), (0.2, 0.2, 0.2)] {
            let (phi, _) = tet_p2_basis(x, y, z);
            let sum: f64 = phi.iter().sum();
            assert!(
                (sum - 1.0).abs() < TOL,
                "PoU failed at ({x},{y},{z}): sum={sum}"
            );
        }
    }

    #[test]
    fn tri_p3_partition_of_unity() {
        for &(x, y) in &[(0.12, 0.21), (0.4, 0.35), (0.25, 0.25)] {
            let (phi, _) = tri_p3_basis(x, y);
            let sum: f64 = phi.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-11,
                "PoU failed at ({x},{y}): sum={sum}"
            );
        }
    }

    #[test]
    fn tet_p3_partition_of_unity() {
        for &(x, y, z) in &[(0.1, 0.15, 0.05), (0.2, 0.15, 0.1)] {
            let (phi, _) = tet_p3_basis(x, y, z);
            let sum: f64 = phi.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "PoU failed at ({x},{y},{z}): sum={sum}"
            );
        }
    }

    // ── gradient consistency via finite differences ───────────────────────

    // Centered finite difference — O(h²) error so 1e-8 tolerance is safe with h=1e-5.
    fn fd_check_tri(poly: usize, x: f64, y: f64, h: f64) {
        let basis_fn: fn(f64, f64) -> (Vec<f64>, Vec<f64>) = match poly {
            1 => tri_p1_basis,
            2 => tri_p2_basis,
            3 => tri_p3_basis,
            _ => panic!("fd_check_tri: poly {poly}"),
        };
        let (_, dphi) = basis_fn(x, y);
        let (phi_px, _) = basis_fn(x + h, y);
        let (phi_mx, _) = basis_fn(x - h, y);
        let (phi_py, _) = basis_fn(x, y + h);
        let (phi_my, _) = basis_fn(x, y - h);
        let num_dof = dphi.len() / 2;
        for i in 0..num_dof {
            let fd_x = (phi_px[i] - phi_mx[i]) / (2.0 * h);
            let fd_y = (phi_py[i] - phi_my[i]) / (2.0 * h);
            let an_x = dphi[i * 2];
            let an_y = dphi[i * 2 + 1];
            assert!(
                (fd_x - an_x).abs() < 1e-8,
                "tri P{poly} dof {i} ∂/∂x: FD={fd_x:.10}, analytic={an_x:.10}"
            );
            assert!(
                (fd_y - an_y).abs() < 1e-8,
                "tri P{poly} dof {i} ∂/∂y: FD={fd_y:.10}, analytic={an_y:.10}"
            );
        }
    }

    fn fd_check_line(poly: usize, t: f64, h: f64) {
        let basis_fn: fn(f64) -> (Vec<f64>, Vec<f64>) = match poly {
            1 => line_p1_basis,
            2 => line_p2_basis,
            3 => line_p3_basis,
            _ => panic!("fd_check_line: poly {poly}"),
        };
        let (_, dphi) = basis_fn(t);
        let (phi_p, _) = basis_fn(t + h);
        let (phi_m, _) = basis_fn(t - h);
        let num_dof = dphi.len();
        for i in 0..num_dof {
            let fd = (phi_p[i] - phi_m[i]) / (2.0 * h);
            let an = dphi[i];
            assert!(
                (fd - an).abs() < 1e-7,
                "line P{poly} dof {i}: FD={fd:.10}, analytic={an:.10}"
            );
        }
    }

    #[test]
    fn line_p1_gradient_fd() {
        fd_check_line(1, 0.31, 1e-6);
    }

    #[test]
    fn line_p2_gradient_fd() {
        fd_check_line(2, 0.27, 1e-6);
    }

    #[test]
    fn line_p3_gradient_fd() {
        fd_check_line(3, 0.26, 1e-6);
    }

    #[test]
    fn tri_p1_gradient_fd() {
        fd_check_tri(1, 0.2, 0.3, 1e-6);
    }
    #[test]
    fn tri_p2_gradient_fd() {
        fd_check_tri(2, 0.2, 0.3, 1e-6);
    }
    #[test]
    fn tri_p3_gradient_fd() {
        fd_check_tri(3, 0.22, 0.31, 1e-6);
    }

    fn fd_check_tet(poly: usize, x: f64, y: f64, z: f64, h: f64) {
        let basis_fn: fn(f64, f64, f64) -> (Vec<f64>, Vec<f64>) = match poly {
            1 => tet_p1_basis,
            2 => tet_p2_basis,
            3 => tet_p3_basis,
            _ => panic!("fd_check_tet: poly {poly}"),
        };
        let (_, dphi) = basis_fn(x, y, z);
        let (phi_px, _) = basis_fn(x + h, y, z);
        let (phi_mx, _) = basis_fn(x - h, y, z);
        let (phi_py, _) = basis_fn(x, y + h, z);
        let (phi_my, _) = basis_fn(x, y - h, z);
        let (phi_pz, _) = basis_fn(x, y, z + h);
        let (phi_mz, _) = basis_fn(x, y, z - h);
        let num_dof = dphi.len() / 3;
        for i in 0..num_dof {
            let fd_x = (phi_px[i] - phi_mx[i]) / (2.0 * h);
            let fd_y = (phi_py[i] - phi_my[i]) / (2.0 * h);
            let fd_z = (phi_pz[i] - phi_mz[i]) / (2.0 * h);
            let an_x = dphi[i * 3];
            let an_y = dphi[i * 3 + 1];
            let an_z = dphi[i * 3 + 2];
            assert!((fd_x - an_x).abs() < 1e-7, "tet P{poly} dof {i} ∂x");
            assert!((fd_y - an_y).abs() < 1e-7, "tet P{poly} dof {i} ∂y");
            assert!((fd_z - an_z).abs() < 1e-7, "tet P{poly} dof {i} ∂z");
        }
    }

    #[test]
    fn tet_p1_gradient_fd() {
        fd_check_tet(1, 0.12, 0.1, 0.08, 1e-6);
    }

    #[test]
    fn tet_p2_gradient_fd() {
        fd_check_tet(2, 0.15, 0.11, 0.07, 1e-6);
    }

    #[test]
    fn tet_p3_gradient_fd() {
        fd_check_tet(3, 0.15, 0.12, 0.08, 1e-6);
    }

    // ── quadrature weight sums ────────────────────────────────────────────

    #[test]
    fn line_quad_weight_sums() {
        for q in [1usize, 2, 3, 4, 5] {
            let (_, wts) = line_quadrature(q).unwrap();
            let sum: f64 = wts.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-13,
                "line q={q}: weights sum to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn tri_quad_weight_sums() {
        for q in [1usize, 3, 4, 6, 7] {
            let (_, wts) = tri_quadrature(q).unwrap();
            let sum: f64 = wts.iter().sum();
            assert!(
                (sum - 0.5).abs() < 1e-13,
                "tri q={q}: weights sum to {sum}, expected 0.5"
            );
        }
    }

    #[test]
    fn tet_quad_weight_sums() {
        for q in [1usize, 4, 5] {
            let (_, wts) = tet_quadrature(q).unwrap();
            let sum: f64 = wts.iter().sum();
            assert!(
                (sum - 1.0 / 6.0).abs() < 1e-13,
                "tet q={q}: weights sum to {sum}, expected 1/6"
            );
        }
    }

    // ── interp exactly reproduces linear fields ───────────────────────────

    #[test]
    fn tri_p1_basis_apply_interp() {
        // u = x at the 3 P1 nodes: u[0]=(0,0)→0, u[1]=(1,0)→1, u[2]=(0,1)→0
        // At q=3 Gauss points, interpolated value should equal x-coordinate.
        let basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, 1, 1, 3).unwrap();
        let u = vec![0.0_f64, 1.0, 0.0]; // ncomp=1 × num_dof=3
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        let (q_ref, _) = tri_quadrature(3).unwrap();
        for (qi, vv) in v.iter().enumerate() {
            let expected = q_ref[qi * 2]; // x-coordinate of qpt
            assert!(
                (*vv - expected).abs() < TOL,
                "qpt {qi}: got {vv}, expected {expected}"
            );
        }
    }

    #[test]
    fn tet_p1_basis_apply_interp() {
        // u = y at P1 tet nodes: (0,0,0)→0, (1,0,0)→0, (0,1,0)→1, (0,0,1)→0
        let basis = SimplexBasis::<f64>::new(ElemTopology::Tet, 1, 1, 4).unwrap();
        let u = vec![0.0_f64, 0.0, 1.0, 0.0]; // u=y at each vertex
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        let (q_ref, _) = tet_quadrature(4).unwrap();
        for (qi, vv) in v.iter().enumerate() {
            let expected = q_ref[qi * 3 + 1]; // y-coordinate of qpt
            assert!(
                (*vv - expected).abs() < TOL,
                "qpt {qi}: got {vv}, expected {expected}"
            );
        }
    }

    #[test]
    fn weight_transpose_matches_interp_transpose_tri_p1_scalar() {
        let b = SimplexBasis::<f64>::new(ElemTopology::Triangle, 1, 1, 3).unwrap();
        assert_eq!(b.num_comp(), 1);
        let ne = 2usize;
        let u: Vec<f64> = (0..ne * b.num_qpoints())
            .map(|i| 0.1 * (i + 1) as f64)
            .collect();
        let mut v_w = vec![0.0_f64; ne * b.num_dof() * b.num_comp()];
        let mut v_i = vec![0.0_f64; ne * b.num_dof() * b.num_comp()];
        b.apply(ne, true, EvalMode::Weight, &u, &mut v_w).unwrap();
        b.apply(ne, true, EvalMode::Interp, &u, &mut v_i).unwrap();
        for i in 0..v_w.len() {
            assert!(
                (v_w[i] - v_i[i]).abs() < TOL,
                "i={i} w={} i={}",
                v_w[i],
                v_i[i]
            );
        }
    }

    // ── P2/P3 nodal identity & exact interpolation ──────────────────────

    fn tri_p2_lattice_nodes() -> [(f64, f64); 6] {
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.0, 0.5),
        ]
    }

    /// P3 triangle lattice (matches `simplex_p3_data` Vandermonde node order).
    fn tri_p3_lattice_nodes() -> [(f64, f64); 10] {
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0 / 3.0, 0.0),
            (2.0 / 3.0, 0.0),
            (2.0 / 3.0, 1.0 / 3.0),
            (1.0 / 3.0, 2.0 / 3.0),
            (0.0, 1.0 / 3.0),
            (0.0, 2.0 / 3.0),
            (1.0 / 3.0, 1.0 / 3.0),
        ]
    }

    fn tet_p3_lattice_nodes() -> Vec<(f64, f64, f64)> {
        let mut out = Vec::with_capacity(20);
        for d0 in 0..4 {
            for d1 in 0..(4 - d0) {
                for d2 in 0..(4 - d0 - d1) {
                    let d3 = 3 - d0 - d1 - d2;
                    out.push((d1 as f64 / 3.0, d2 as f64 / 3.0, d3 as f64 / 3.0));
                }
            }
        }
        assert_eq!(out.len(), 20);
        out
    }

    #[test]
    fn tri_p2_nodal_kronecker() {
        const EPS: f64 = 1e-12;
        let nodes = tri_p2_lattice_nodes();
        for (i, &(xi, yi)) in nodes.iter().enumerate() {
            let (phi, _) = tri_p2_basis(xi, yi);
            for (j, &v) in phi.iter().enumerate() {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v - e).abs() < EPS,
                    "tri P2 node {i}, dof {j}: got {v}, expected {e}"
                );
            }
        }
    }

    #[test]
    fn tri_p3_nodal_kronecker() {
        const EPS: f64 = 5e-11;
        let nodes = tri_p3_lattice_nodes();
        for (i, &(xi, yi)) in nodes.iter().enumerate() {
            let (phi, _) = tri_p3_basis(xi, yi);
            for (j, &v) in phi.iter().enumerate() {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v - e).abs() < EPS,
                    "tri P3 node {i}, dof {j}: got {v}, expected {e}"
                );
            }
        }
    }

    #[test]
    fn tet_p3_nodal_kronecker() {
        const EPS: f64 = 5e-10;
        let nodes = tet_p3_lattice_nodes();
        for (i, &(xi, yi, zi)) in nodes.iter().enumerate() {
            let (phi, _) = tet_p3_basis(xi, yi, zi);
            for (j, &v) in phi.iter().enumerate() {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v - e).abs() < EPS,
                    "tet P3 node {i}, dof {j}: got {v}, expected {e}"
                );
            }
        }
    }

    #[test]
    fn tri_p3_basis_apply_interp_cubic_exact() {
        let nodes = tri_p3_lattice_nodes();
        fn f(x: f64, y: f64) -> f64 {
            x * x * y + 0.25 * y * y * y
        }
        let u: Vec<f64> = nodes.iter().map(|&(x, y)| f(x, y)).collect();
        let basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, 3, 1, 7).unwrap();
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        let q_ref = basis.q_ref();
        for iq in 0..basis.num_qpoints() {
            let x = q_ref[iq * 2];
            let y = q_ref[iq * 2 + 1];
            let expected = f(x, y);
            assert!(
                (v[iq] - expected).abs() < 2e-11,
                "qpt {iq}: got {}, expected {}",
                v[iq],
                expected
            );
        }
    }

    #[test]
    fn tet_p3_basis_apply_interp_cubic_exact() {
        let nodes = tet_p3_lattice_nodes();
        fn g(x: f64, y: f64, z: f64) -> f64 {
            x * y * z + x * x - 0.37 * y + 1.2 * z * z
        }
        let u: Vec<f64> = nodes.iter().map(|&(x, y, z)| g(x, y, z)).collect();
        let basis = SimplexBasis::<f64>::new(ElemTopology::Tet, 3, 1, 5).unwrap();
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        let q_ref = basis.q_ref();
        for iq in 0..basis.num_qpoints() {
            let x = q_ref[iq * 3];
            let y = q_ref[iq * 3 + 1];
            let z = q_ref[iq * 3 + 2];
            let expected = g(x, y, z);
            assert!(
                (v[iq] - expected).abs() < 3e-10,
                "qpt {iq}: got {}, expected {}",
                v[iq],
                expected
            );
        }
    }

    #[test]
    fn tri_p3_basis_apply_grad_cubic_matches_analytic() {
        let nodes = tri_p3_lattice_nodes();
        fn f(x: f64, y: f64) -> f64 {
            x * x * y
        }
        fn grad_f(x: f64, y: f64) -> (f64, f64) {
            (2.0 * x * y, x * x)
        }
        let u: Vec<f64> = nodes.iter().map(|&(x, y)| f(x, y)).collect();
        let basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, 3, 1, 6).unwrap();
        let mut v = vec![0.0_f64; basis.num_qpoints() * 2];
        basis.apply(1, false, EvalMode::Grad, &u, &mut v).unwrap();
        let q_ref = basis.q_ref();
        for iq in 0..basis.num_qpoints() {
            let x = q_ref[iq * 2];
            let y = q_ref[iq * 2 + 1];
            let (gx, gy) = grad_f(x, y);
            assert!((v[iq * 2] - gx).abs() < 2e-10, "qpt {iq} ∂x");
            assert!((v[iq * 2 + 1] - gy).abs() < 2e-10, "qpt {iq} ∂y");
        }
    }

    // ── weight mode ───────────────────────────────────────────────────────

    #[test]
    fn tri_p1_weight_sums_to_area() {
        let basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, 1, 1, 3).unwrap();
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis
            .apply(1, false, EvalMode::Weight, &[], &mut v)
            .unwrap();
        let sum: f64 = v.iter().sum();
        assert!(
            (sum - 0.5).abs() < TOL,
            "weight sum={sum}, expected 0.5 (triangle area)"
        );
    }

    #[test]
    fn tet_p1_weight_sums_to_volume() {
        let basis = SimplexBasis::<f64>::new(ElemTopology::Tet, 1, 1, 4).unwrap();
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis
            .apply(1, false, EvalMode::Weight, &[], &mut v)
            .unwrap();
        let sum: f64 = v.iter().sum();
        assert!(
            (sum - 1.0 / 6.0).abs() < TOL,
            "weight sum={sum}, expected 1/6 (tet volume)"
        );
    }

    #[test]
    fn tri_p1_div_curl_adjoint_identities() {
        let basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, 1, 2, 3).unwrap();
        assert_eq!(basis.dim(), 2);
        assert_eq!(basis.num_comp(), 2);
        let nd = basis.num_dof() * basis.num_comp();
        let nq = basis.num_qpoints();
        let u: Vec<f64> = (0..nd).map(|i| 0.13 * i as f64 - 0.2).collect();
        let w_div: Vec<f64> = (0..nq).map(|i| 0.05 * i as f64 + 0.4).collect();
        let w_curl: Vec<f64> = (0..nq).map(|i| 0.11 * i as f64 - 0.1).collect();

        let mut div_u = vec![0.0_f64; nq];
        basis
            .apply(1, false, EvalMode::Div, &u, &mut div_u)
            .unwrap();
        let mut dt_w = vec![0.0_f64; nd];
        basis
            .apply(1, true, EvalMode::Div, &w_div, &mut dt_w)
            .unwrap();
        let lhs: f64 = u.iter().zip(dt_w.iter()).map(|(a, b)| a * b).sum();
        let rhs: f64 = div_u.iter().zip(w_div.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-10 * (1.0 + lhs.abs()));

        let mut curl_u = vec![0.0_f64; nq];
        basis
            .apply(1, false, EvalMode::Curl, &u, &mut curl_u)
            .unwrap();
        dt_w.fill(0.0);
        basis
            .apply(1, true, EvalMode::Curl, &w_curl, &mut dt_w)
            .unwrap();
        let lhs2: f64 = u.iter().zip(dt_w.iter()).map(|(a, b)| a * b).sum();
        let rhs2: f64 = curl_u.iter().zip(w_curl.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs2 - rhs2).abs() < 1e-10 * (1.0 + lhs2.abs()));
    }

    #[test]
    fn tet_p1_div_curl_adjoint_identities() {
        let basis = SimplexBasis::<f64>::new(ElemTopology::Tet, 1, 3, 4).unwrap();
        assert_eq!(basis.dim(), 3);
        let nd = basis.num_dof() * basis.num_comp();
        let nq = basis.num_qpoints();
        let u: Vec<f64> = (0..nd).map(|i| 0.07 * i as f64 - 0.15).collect();
        let w_div: Vec<f64> = (0..nq).map(|i| 0.03 * i as f64 + 0.5).collect();
        let w_curl: Vec<f64> = (0..nq * 3).map(|i| 0.02 * i as f64 + 0.12).collect();

        let mut div_u = vec![0.0_f64; nq];
        basis
            .apply(1, false, EvalMode::Div, &u, &mut div_u)
            .unwrap();
        let mut dt_w = vec![0.0_f64; nd];
        basis
            .apply(1, true, EvalMode::Div, &w_div, &mut dt_w)
            .unwrap();
        let lhs: f64 = u.iter().zip(dt_w.iter()).map(|(a, b)| a * b).sum();
        let rhs: f64 = div_u.iter().zip(w_div.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-10 * (1.0 + lhs.abs()));

        let mut curl_u = vec![0.0_f64; nq * 3];
        basis
            .apply(1, false, EvalMode::Curl, &u, &mut curl_u)
            .unwrap();
        dt_w.fill(0.0);
        basis
            .apply(1, true, EvalMode::Curl, &w_curl, &mut dt_w)
            .unwrap();
        let lhs2: f64 = u.iter().zip(dt_w.iter()).map(|(a, b)| a * b).sum();
        let rhs2: f64 = curl_u.iter().zip(w_curl.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs2 - rhs2).abs() < 1e-9 * (1.0 + lhs2.abs()));
    }
}
