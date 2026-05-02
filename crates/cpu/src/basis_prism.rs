//! Prism (wedge) basis via tensor product of Triangle × Line.
//!
//! A prism reference element is the Cartesian product of:
//! - A reference Triangle in the (r,s) plane with vertices (0,0), (1,0), (0,1)
//! - A reference Line in the t direction with interval [-1, 1]
//!
//! The basis functions are products of triangle Lagrange basis functions
//! (in r,s) and 1D Lagrange basis functions (in t). Quadrature is the tensor
//! product of triangle quadrature and 1D Gauss quadrature.
//!
//! ## Memory layout
//!
//! * `interp` — row-major `[nqpts × num_dof]`
//! * `grad`   — row-major `[nqpts × num_dof × dim]`,
//!              index: `(qpt * num_dof + dof) * dim + d`

use reed_core::{
    basis::BasisTrait,
    enums::EvalMode,
    error::{ReedError, ReedResult},
    scalar::Scalar,
};

use super::basis_lagrange::{build_grad, build_interp, gauss_quadrature, gauss_lobatto_nodes, to_scalar};
use super::basis_simplex::SimplexBasis;
use reed_core::enums::ElemTopology;

/// H1 Lagrange basis on a prism (wedge) reference element, formed as the
/// tensor product of a Triangle basis in (r,s) and a 1D Lagrange basis in t.
pub struct PrismBasis<T: Scalar> {
    dim: usize,
    ncomp: usize,
    num_dof: usize,
    num_qpoints: usize,
    /// Quadrature point coordinates, row-major `[nqpts × 3]`: (r, s, t).
    q_ref: Vec<T>,
    /// Quadrature weights, length `nqpts`.
    weights: Vec<T>,
    /// Interpolation matrix, row-major `[nqpts × num_dof]`.
    interp: Vec<T>,
    /// Gradient tensor, layout `[nqpts × num_dof × 3]`.
    grad: Vec<T>,
}

impl<T: Scalar> PrismBasis<T> {
    /// Construct a prism basis.
    ///
    /// # Parameters
    /// * `tri_poly` — polynomial degree for the triangle in (r,s) (1 = P1, 2 = P2, 3 = P3).
    /// * `line_p`  — polynomial degree for the line in t (>= 2 uses Gauss-Lobatto-Lagrange;
    ///                p=1 falls back to linear Lagrange on [-1,1]).
    /// * `tri_q`   — number of quadrature points for the triangle (see `tri_quadrature`).
    /// * `line_q`  — number of 1D Gauss quadrature points for the t direction.
    /// * `ncomp`   — number of field components.
    ///
    /// # Errors
    /// Returns `ReedError::Basis` for unsupported degree/q combinations.
    pub fn new(
        tri_poly: usize,
        line_p: usize,
        tri_q: usize,
        line_q: usize,
        ncomp: usize,
    ) -> ReedResult<Self> {
        // --- Build internal triangle basis -----------------------------------
        let tri_basis = SimplexBasis::<T>::new(ElemTopology::Triangle, tri_poly, 1, tri_q)?;
        let tri_ndof = tri_basis.num_dof();
        let tri_nqpts = tri_basis.num_qpoints();
        let tri_interp = tri_basis.interp_matrix().to_vec();
        let tri_grad = tri_basis.grad_matrix().to_vec(); // [nqpts × ndof × 2]
        let tri_q_ref = tri_basis.q_ref().to_vec(); // [nqpts × 2]
        let tri_weights = tri_basis.q_weights().to_vec();

        // --- Build 1D line basis in t ----------------------------------------
        // Use Gauss-Lobatto nodes for Lagrange interpolation on [-1,1].
        let line_nodes = if line_p >= 2 {
            gauss_lobatto_nodes(line_p)?
        } else {
            // p=1: linear, two nodes at endpoints
            vec![-1.0_f64, 1.0]
        };
        let line_ndof = line_nodes.len();

        let (line_q_ref_f64, line_weights_f64) = gauss_quadrature(line_q)?;
        let line_nqpts = line_q_ref_f64.len();

        // Build 1D interpolation and gradient matrices.
        let line_interp: Vec<T> = build_interp::<T>(&line_nodes, &line_q_ref_f64)?;
        let line_grad: Vec<T> = build_grad::<T>(&line_nodes, &line_q_ref_f64)?;

        // --- Tensor-product sizes --------------------------------------------
        let dim = 3;
        let num_dof = tri_ndof * line_ndof;
        let num_qpoints = tri_nqpts * line_nqpts;

        // --- Tensor-product quadrature reference points and weights ----------
        // Order: tri-major, line-minor to match interp/grad layout
        // (qpt = it * line_nqpts + il)
        let mut q_ref = Vec::with_capacity(num_qpoints * dim);
        let mut weights = Vec::with_capacity(num_qpoints);

        for it in 0..tri_nqpts {
            for il in 0..line_nqpts {
                let r = tri_q_ref[it * 2];
                let s = tri_q_ref[it * 2 + 1];
                let t = to_scalar::<T>(line_q_ref_f64[il])?;
                let wl = to_scalar::<T>(line_weights_f64[il])?;
                q_ref.push(r);
                q_ref.push(s);
                q_ref.push(t);
                weights.push(tri_weights[it] * wl);
            }
        }

        // --- Tensor-product (Kronecker) interpolation matrix -----------------
        let mut interp = vec![T::ZERO; num_qpoints * num_dof];
        for il in 0..line_nqpts {
            let li_off = il * line_ndof;
            for it in 0..tri_nqpts {
                let ti_off = it * tri_ndof;
                let qpt = it * line_nqpts + il;
                for dt in 0..tri_ndof {
                    let tri_val = tri_interp[ti_off + dt];
                    for dl in 0..line_ndof {
                        let line_val = line_interp[li_off + dl];
                        let dof_idx = dt * line_ndof + dl;
                        interp[qpt * num_dof + dof_idx] = tri_val * line_val;
                    }
                }
            }
        }

        // --- Tensor-product gradient -----------------------------------------
        // grad[(qpt * num_dof + dof) * 3 + d]
        // d=0: ∂/∂r = ∂tri/∂r * line_interp
        // d=1: ∂/∂s = ∂tri/∂s * line_interp
        // d=2: ∂/∂t = tri_interp * ∂line/∂t
        let mut grad = vec![T::ZERO; num_qpoints * num_dof * dim];

        for il in 0..line_nqpts {
            let li_off = il * line_ndof;
            for it in 0..tri_nqpts {
                let ti_off = it * tri_ndof;
                let qpt = it * line_nqpts + il;
                for dt in 0..tri_ndof {
                    let tri_val = tri_interp[ti_off + dt];
                    // tri_grad is [(qt * tri_ndof + dt) * 2 + d]
                    let tri_gr = tri_grad[(it * tri_ndof + dt) * 2];
                    let tri_gs = tri_grad[(it * tri_ndof + dt) * 2 + 1];
                    for dl in 0..line_ndof {
                        let line_val = line_interp[li_off + dl];
                        let line_dt = line_grad[li_off + dl];
                        let dof_idx = dt * line_ndof + dl;
                        let base = (qpt * num_dof + dof_idx) * dim;
                        grad[base] = tri_gr * line_val;       // ∂/∂r
                        grad[base + 1] = tri_gs * line_val;   // ∂/∂s
                        grad[base + 2] = tri_val * line_dt;   // ∂/∂t
                    }
                }
            }
        }

        Ok(Self {
            dim,
            ncomp,
            num_dof,
            num_qpoints,
            q_ref,
            weights,
            interp,
            grad,
        })
    }

    // ── accessor helpers ───────────────────────────────────────────────────

    #[inline]
    fn interp_val(&self, qpt: usize, dof: usize) -> T {
        self.interp[qpt * self.num_dof + dof]
    }

    #[inline]
    fn grad_val(&self, qpt: usize, dof: usize, d: usize) -> T {
        self.grad[(qpt * self.num_dof + dof) * self.dim + d]
    }

    // ── element-level apply ────────────────────────────────────────────────

    fn apply_interp_elem(&self, transpose: bool, u_elem: &[T], v_elem: &mut [T]) {
        for comp in 0..self.ncomp {
            if transpose {
                for dof in 0..self.num_dof {
                    let mut sum = T::ZERO;
                    for qpt in 0..self.num_qpoints {
                        sum += self.interp_val(qpt, dof) * u_elem[qpt * self.ncomp + comp];
                    }
                    v_elem[comp * self.num_dof + dof] += sum;
                }
            } else {
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
                for qpt in 0..self.num_qpoints {
                    for d in 0..self.dim {
                        let mut sum = T::ZERO;
                        for dof in 0..self.num_dof {
                            sum += self.grad_val(qpt, dof, d)
                                * u_elem[comp * self.num_dof + dof];
                        }
                        v_elem[qpt * qcomp + comp * self.dim + d] = sum;
                    }
                }
            }
        }
    }
}

// ── BasisTrait impl ───────────────────────────────────────────────────────

impl<T: Scalar> BasisTrait<T> for PrismBasis<T> {
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
                if u.len() != in_stride * num_elem || v.len() != out_stride * num_elem {
                    return Err(ReedError::Basis(format!(
                        "PrismBasis interp size mismatch: input {}, expected {}; output {}, expected {}",
                        u.len(),
                        in_stride * num_elem,
                        v.len(),
                        out_stride * num_elem
                    )));
                }
                if transpose {
                    v.fill(T::ZERO);
                }
                for (u_elem, v_elem) in u.chunks(in_stride).zip(v.chunks_mut(out_stride)) {
                    self.apply_interp_elem(transpose, u_elem, v_elem);
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
                if u.len() != in_stride * num_elem || v.len() != out_stride * num_elem {
                    return Err(ReedError::Basis(format!(
                        "PrismBasis grad size mismatch: input {}, expected {}; output {}, expected {}",
                        u.len(),
                        in_stride * num_elem,
                        v.len(),
                        out_stride * num_elem
                    )));
                }
                if transpose {
                    v.fill(T::ZERO);
                }
                for (u_elem, v_elem) in u.chunks(in_stride).zip(v.chunks_mut(out_stride)) {
                    self.apply_grad_elem(transpose, u_elem, v_elem);
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
                        "PrismBasis weight output length {} != expected {}",
                        v.len(),
                        num_elem * self.num_qpoints
                    )));
                }
                for v_elem in v.chunks_mut(self.num_qpoints) {
                    v_elem.copy_from_slice(&self.weights);
                }
            }
            EvalMode::Div => {
                if self.ncomp != self.dim {
                    return Err(ReedError::Basis(
                        "EvalMode::Div requires ncomp == dim for PrismBasis".into(),
                    ));
                }
                let qcomp = self.ncomp * self.dim;
                if transpose {
                    let in_size = num_elem * self.num_qpoints;
                    let out_size = num_elem * self.num_dof * self.ncomp;
                    if u.len() != in_size || v.len() != out_size {
                        return Err(ReedError::Basis(format!(
                            "PrismBasis div transpose size mismatch: input {}, expected {}; output {}, expected {}",
                            u.len(), in_size, v.len(), out_size
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
                            "PrismBasis div size mismatch: input {}, expected {}; output {}, expected {}",
                            u.len(), in_size, v.len(), out_size
                        )));
                    }
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
                    (3, 3) => {
                        if transpose {
                            let in_size = num_elem * self.num_qpoints * 3;
                            let out_size = num_elem * self.num_dof * self.ncomp;
                            if u.len() != in_size || v.len() != out_size {
                                return Err(ReedError::Basis(format!(
                                    "PrismBasis curl transpose size mismatch"
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
                                    "PrismBasis curl size mismatch"
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
                            "EvalMode::Curl requires (dim, ncomp) = (3, 3) for PrismBasis".into(),
                        ));
                    }
                }
            }
            other => {
                return Err(ReedError::Basis(format!(
                    "PrismBasis: eval mode {:?} not implemented",
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
}

// ── tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    // ── partition-of-unity ────────────────────────────────────────────────

    #[test]
    fn prism_p1_partition_of_unity() {
        // P1 in triangle, P1 on line -> 3 * 2 = 6 DOFs
        let basis = PrismBasis::<f64>::new(1, 1, 3, 2, 1).unwrap();
        assert_eq!(basis.num_dof(), 6);
        // Sum of interpolation values should be 1 at every qpt.
        for qpt in 0..basis.num_qpoints() {
            let mut sum = 0.0_f64;
            for dof in 0..basis.num_dof() {
                sum += basis.interp_val(qpt, dof);
            }
            assert!((sum - 1.0).abs() < TOL, "PoU failed at qpt {qpt}: sum={sum}");
        }
    }

    #[test]
    fn prism_p2_partition_of_unity() {
        // P2 triangle (6 DOFs) × P2 line (3 nodes via p=3 quadratic) = 18 DOFs
        let basis = PrismBasis::<f64>::new(2, 3, 6, 3, 1).unwrap();
        assert_eq!(basis.num_dof(), 18);
        for qpt in 0..basis.num_qpoints() {
            let mut sum = 0.0_f64;
            for dof in 0..basis.num_dof() {
                sum += basis.interp_val(qpt, dof);
            }
            assert!((sum - 1.0).abs() < TOL, "PoU failed at qpt {qpt}: sum={sum}");
        }
    }

    // ── DOF counts ────────────────────────────────────────────────────────

    #[test]
    fn prism_dof_counts() {
        // P1 tri (3 DOFs) × P1 line (2 nodes via p=2 linear) = 6
        let b = PrismBasis::<f64>::new(1, 2, 3, 2, 1).unwrap();
        assert_eq!(b.num_dof(), 6);
        assert_eq!(b.dim(), 3);

        // P2 tri (6 DOFs) × P2 line (3 nodes via p=3 quadratic) = 18
        let b = PrismBasis::<f64>::new(2, 3, 6, 3, 1).unwrap();
        assert_eq!(b.num_dof(), 18);

        // P1 tri (3 DOFs) × P2 line (3 nodes via p=3 quadratic) = 9
        let b = PrismBasis::<f64>::new(1, 3, 3, 3, 1).unwrap();
        assert_eq!(b.num_dof(), 9);
    }

    // ── quadrature weight sums ────────────────────────────────────────────

    #[test]
    fn prism_weight_sums_to_volume() {
        // Reference prism volume = area(triangle) * length(line) = 0.5 * 2 = 1.0
        for (tri_p, line_p, tri_q, line_q) in [
            (1usize, 1usize, 3usize, 2usize),
            (2, 2, 6, 3),
            (1, 2, 4, 4),
        ] {
            let basis = PrismBasis::<f64>::new(tri_p, line_p, tri_q, line_q, 1).unwrap();
            let mut v = vec![0.0_f64; basis.num_qpoints()];
            basis.apply(1, false, EvalMode::Weight, &[], &mut v).unwrap();
            let sum: f64 = v.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "Prism (tri_p={tri_p}, line_p={line_p}, tri_q={tri_q}, line_q={line_q}): \
                 weight sum={sum}, expected 1.0"
            );
        }
    }

    // ── constant interpolation ─────────────────────────────────────────────

    #[test]
    fn prism_p1_interp_constant() {
        let basis = PrismBasis::<f64>::new(1, 1, 3, 2, 1).unwrap();
        // u = 2.5 at all DOFs
        let c = 2.5_f64;
        let u = vec![c; basis.num_dof()];
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        for (i, &val) in v.iter().enumerate() {
            assert!((val - c).abs() < TOL, "qpt {i}: got {val}, expected {c}");
        }
    }

    // ── linear interpolation ───────────────────────────────────────────────

    #[test]
    fn prism_p1_interp_linear_r() {
        // u = r at DOF nodes. r varies from 0 to 1 on the triangle.
        // Triangle nodes: (0,0) -> r=0, (1,0) -> r=1, (0,1) -> r=0
        let basis = PrismBasis::<f64>::new(1, 1, 3, 2, 1).unwrap();
        let tri_ndof = 3;
        let line_ndof = 2;
        let num_dof = tri_ndof * line_ndof;
        let mut u = vec![0.0_f64; num_dof];
        // Triangle node r-values at vertices: 0, 1, 0
        let tri_r = [0.0_f64, 1.0, 0.0];
        for dt in 0..tri_ndof {
            for dl in 0..line_ndof {
                u[dt * line_ndof + dl] = tri_r[dt];
            }
        }
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        for (iq, &val) in v.iter().enumerate() {
            let r = basis.q_ref[iq * 3];
            assert!((val - r).abs() < TOL, "qpt {iq}: got {val}, expected {r}");
        }
    }

    #[test]
    fn prism_p1_interp_linear_t() {
        // u = t at DOF nodes. t is independent of triangle node.
        // Line nodes: t = -1, 1
        let basis = PrismBasis::<f64>::new(1, 1, 3, 2, 1).unwrap();
        let tri_ndof = 3;
        let line_ndof = 2;
        let num_dof = tri_ndof * line_ndof;
        let mut u = vec![0.0_f64; num_dof];
        let line_t = [-1.0_f64, 1.0];
        for dt in 0..tri_ndof {
            for dl in 0..line_ndof {
                u[dt * line_ndof + dl] = line_t[dl];
            }
        }
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        for (iq, &val) in v.iter().enumerate() {
            let t = basis.q_ref[iq * 3 + 2];
            assert!((val - t).abs() < TOL, "qpt {iq}: got {val}, expected {t}");
        }
    }

    // ── gradient of linear field ───────────────────────────────────────────

    #[test]
    fn prism_p1_grad_r() {
        // u = r: ∂/∂r = 1, ∂/∂s = 0, ∂/∂t = 0
        let basis = PrismBasis::<f64>::new(1, 1, 3, 2, 1).unwrap();
        let tri_ndof = 3;
        let line_ndof = 2;
        let num_dof = tri_ndof * line_ndof;
        let mut u = vec![0.0_f64; num_dof];
        let tri_r = [0.0_f64, 1.0, 0.0];
        for dt in 0..tri_ndof {
            for dl in 0..line_ndof {
                u[dt * line_ndof + dl] = tri_r[dt];
            }
        }
        let mut grad_out = vec![0.0_f64; basis.num_qpoints() * 3];
        basis.apply(1, false, EvalMode::Grad, &u, &mut grad_out).unwrap();
        for iq in 0..basis.num_qpoints() {
            let gr = grad_out[iq * 3];
            let gs = grad_out[iq * 3 + 1];
            let gt = grad_out[iq * 3 + 2];
            assert!((gr - 1.0).abs() < TOL, "qpt {iq} ∂r: got {gr}, expected 1");
            assert!(gs.abs() < TOL, "qpt {iq} ∂s: got {gs}, expected 0");
            assert!(gt.abs() < TOL, "qpt {iq} ∂t: got {gt}, expected 0");
        }
    }

    #[test]
    fn prism_p1_grad_s() {
        // u = s: ∂/∂r = 0, ∂/∂s = 1, ∂/∂t = 0
        let basis = PrismBasis::<f64>::new(1, 1, 3, 2, 1).unwrap();
        let tri_ndof = 3;
        let line_ndof = 2;
        let num_dof = tri_ndof * line_ndof;
        let mut u = vec![0.0_f64; num_dof];
        let tri_s = [0.0_f64, 0.0, 1.0];
        for dt in 0..tri_ndof {
            for dl in 0..line_ndof {
                u[dt * line_ndof + dl] = tri_s[dt];
            }
        }
        let mut grad_out = vec![0.0_f64; basis.num_qpoints() * 3];
        basis.apply(1, false, EvalMode::Grad, &u, &mut grad_out).unwrap();
        for iq in 0..basis.num_qpoints() {
            let gr = grad_out[iq * 3];
            let gs = grad_out[iq * 3 + 1];
            let gt = grad_out[iq * 3 + 2];
            assert!(gr.abs() < TOL, "qpt {iq} ∂r: got {gr}, expected 0");
            assert!((gs - 1.0).abs() < TOL, "qpt {iq} ∂s: got {gs}, expected 1");
            assert!(gt.abs() < TOL, "qpt {iq} ∂t: got {gt}, expected 0");
        }
    }

    #[test]
    fn prism_p1_grad_t() {
        // u = t: ∂/∂r = 0, ∂/∂s = 0, ∂/∂t = 1
        let basis = PrismBasis::<f64>::new(1, 1, 3, 2, 1).unwrap();
        let tri_ndof = 3;
        let line_ndof = 2;
        let num_dof = tri_ndof * line_ndof;
        let mut u = vec![0.0_f64; num_dof];
        let line_t = [-1.0_f64, 1.0];
        for dt in 0..tri_ndof {
            for dl in 0..line_ndof {
                u[dt * line_ndof + dl] = line_t[dl];
            }
        }
        let mut grad_out = vec![0.0_f64; basis.num_qpoints() * 3];
        basis.apply(1, false, EvalMode::Grad, &u, &mut grad_out).unwrap();
        for iq in 0..basis.num_qpoints() {
            let gr = grad_out[iq * 3];
            let gs = grad_out[iq * 3 + 1];
            let gt = grad_out[iq * 3 + 2];
            assert!(gr.abs() < TOL, "qpt {iq} ∂r: got {gr}, expected 0");
            assert!(gs.abs() < TOL, "qpt {iq} ∂s: got {gs}, expected 0");
            assert!((gt - 1.0).abs() < TOL, "qpt {iq} ∂t: got {gt}, expected 1");
        }
    }

    // ── linear field in 3D ─────────────────────────────────────────────────

    #[test]
    fn prism_p1_interp_linear_3d() {
        // u = 2r + 3s - 0.5t + 1.7
        let basis = PrismBasis::<f64>::new(1, 1, 3, 2, 1).unwrap();
        let tri_ndof = 3;
        let line_ndof = 2;
        let num_dof = tri_ndof * line_ndof;
        let mut u = vec![0.0_f64; num_dof];
        let tri_r = [0.0_f64, 1.0, 0.0];
        let tri_s = [0.0_f64, 0.0, 1.0];
        let line_t = [-1.0_f64, 1.0];
        for dt in 0..tri_ndof {
            for dl in 0..line_ndof {
                let r_val = tri_r[dt];
                let s_val = tri_s[dt];
                let t_val = line_t[dl];
                u[dt * line_ndof + dl] = 2.0 * r_val + 3.0 * s_val - 0.5 * t_val + 1.7;
            }
        }
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        for (iq, &val) in v.iter().enumerate() {
            let r = basis.q_ref[iq * 3];
            let s = basis.q_ref[iq * 3 + 1];
            let t = basis.q_ref[iq * 3 + 2];
            let expected = 2.0 * r + 3.0 * s - 0.5 * t + 1.7;
            assert!((val - expected).abs() < TOL, "qpt {iq}: got {val}, expected {expected}");
        }
    }

    #[test]
    fn prism_p1_grad_linear_3d() {
        // u = 2r + 3s - 0.5t + 1.7 -> grad = (2, 3, -0.5)
        let basis = PrismBasis::<f64>::new(1, 1, 3, 2, 1).unwrap();
        let tri_ndof = 3;
        let line_ndof = 2;
        let num_dof = tri_ndof * line_ndof;
        let mut u = vec![0.0_f64; num_dof];
        let tri_r = [0.0_f64, 1.0, 0.0];
        let tri_s = [0.0_f64, 0.0, 1.0];
        let line_t = [-1.0_f64, 1.0];
        for dt in 0..tri_ndof {
            for dl in 0..line_ndof {
                let r_val = tri_r[dt];
                let s_val = tri_s[dt];
                let t_val = line_t[dl];
                u[dt * line_ndof + dl] = 2.0 * r_val + 3.0 * s_val - 0.5 * t_val + 1.7;
            }
        }
        let mut grad_out = vec![0.0_f64; basis.num_qpoints() * 3];
        basis.apply(1, false, EvalMode::Grad, &u, &mut grad_out).unwrap();
        for iq in 0..basis.num_qpoints() {
            let gr = grad_out[iq * 3];
            let gs = grad_out[iq * 3 + 1];
            let gt = grad_out[iq * 3 + 2];
            assert!((gr - 2.0).abs() < TOL, "qpt {iq} ∂r: got {gr}, expected 2");
            assert!((gs - 3.0).abs() < TOL, "qpt {iq} ∂s: got {gs}, expected 3");
            assert!((gt + 0.5).abs() < TOL, "qpt {iq} ∂t: got {gt}, expected -0.5");
        }
    }

    // ── adjoint identities ─────────────────────────────────────────────────

    #[test]
    fn prism_div_adjoint_identity() {
        let basis = PrismBasis::<f64>::new(1, 1, 3, 2, 3).unwrap();
        assert_eq!(basis.dim(), 3);
        assert_eq!(basis.num_comp(), 3);
        let nd = basis.num_dof() * basis.num_comp();
        let nq = basis.num_qpoints();
        let u: Vec<f64> = (0..nd).map(|i| 0.07 * i as f64 - 0.15).collect();
        let w_div: Vec<f64> = (0..nq).map(|i| 0.03 * i as f64 + 0.5).collect();

        let mut div_u = vec![0.0_f64; nq];
        basis.apply(1, false, EvalMode::Div, &u, &mut div_u).unwrap();
        let mut dt_w = vec![0.0_f64; nd];
        basis.apply(1, true, EvalMode::Div, &w_div, &mut dt_w).unwrap();
        let lhs: f64 = u.iter().zip(dt_w.iter()).map(|(a, b)| a * b).sum();
        let rhs: f64 = div_u.iter().zip(w_div.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-10 * (1.0 + lhs.abs()));
    }

    #[test]
    fn prism_curl_adjoint_identity() {
        let basis = PrismBasis::<f64>::new(1, 2, 3, 3, 3).unwrap();
        assert_eq!(basis.dim(), 3);
        let nd = basis.num_dof() * basis.num_comp();
        let nq = basis.num_qpoints();
        let u: Vec<f64> = (0..nd).map(|i| 0.05 * i as f64 - 0.2).collect();
        let w_curl: Vec<f64> = (0..nq * 3).map(|i| 0.02 * i as f64 + 0.12).collect();

        let mut curl_u = vec![0.0_f64; nq * 3];
        basis.apply(1, false, EvalMode::Curl, &u, &mut curl_u).unwrap();
        let mut ct_w = vec![0.0_f64; nd];
        basis.apply(1, true, EvalMode::Curl, &w_curl, &mut ct_w).unwrap();
        let lhs: f64 = u.iter().zip(ct_w.iter()).map(|(a, b)| a * b).sum();
        let rhs: f64 = curl_u.iter().zip(w_curl.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-9 * (1.0 + lhs.abs()));
    }
}
