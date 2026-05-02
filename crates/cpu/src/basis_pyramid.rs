//! Pyramid basis via collapsed-coordinate transform from a hex.
//!
//! A pyramid with 5 vertices (4 on the base, 1 at the apex) is handled by
//! mapping from a reference hex [-1,1]^3 via:
//!
//! ```text
//! r = ξ * (1 - ζ) / 2
//! s = η * (1 - ζ) / 2
//! t = ζ
//! ```
//!
//! At the apex (ζ = 1), all points collapse to (r,s) = (0,0), t = 1.
//!
//! The Jacobian determinant is ((1-ζ)/2)^2, which vanishes at the apex.
//! Gauss quadrature (which avoids the apex points at ζ=1) is used for
//! integration.
//!
//! The interpolation basis is a standard tensor-product Lagrange basis on
//! the hex, but the gradient is transformed via the inverse Jacobian:
//!
//! ```text
//! ∇_pyramid f = J^{-T} · ∇_hex f
//! ```
//!
//! ## Memory layout
//!
//! * `interp` — row-major `[nqpts × num_dof]` (same as hex interp)
//! * `grad`   — row-major `[nqpts × num_dof × 3]`,
//!              index: `(qpt * num_dof + dof) * 3 + d`

use reed_core::{
    basis::BasisTrait,
    enums::EvalMode,
    error::{ReedError, ReedResult},
    scalar::Scalar,
    QuadMode,
};

use super::basis_lagrange::LagrangeBasis;

/// H1 Lagrange basis on a pyramid reference element, implemented via
/// collapsed-coordinate mapping from a reference hex [-1,1]^3.
pub struct PyramidBasis<T: Scalar> {
    dim: usize,
    ncomp: usize,
    num_dof: usize,
    num_qpoints: usize,
    /// Quadrature point coordinates in pyramid (r,s,t) space,
    /// row-major `[nqpts × 3]`.
    q_ref: Vec<T>,
    /// Quadrature weights (hex weights × |det J|), length `nqpts`.
    weights: Vec<T>,
    /// Interpolation matrix, row-major `[nqpts × num_dof]`.
    interp: Vec<T>,
    /// Gradient tensor in pyramid space, layout `[nqpts × num_dof × 3]`.
    grad: Vec<T>,
}

impl<T: Scalar> PyramidBasis<T> {
    /// Construct a pyramid basis via collapsed-coordinate transform.
    ///
    /// # Parameters
    /// * `p`     — polynomial degree in each direction of the parent hex
    ///             (>= 2 for Gauss-Lobatto-Lagrange nodes).
    /// * `q`     — number of 1D quadrature points (per direction on the hex).
    /// * `qmode` — quadrature mode (`Gauss` or `GaussLobatto`).
    /// * `ncomp` — number of field components.
    ///
    /// # Errors
    /// Returns `ReedError::Basis` for unsupported parameter combinations.
    ///
    /// # Note
    /// The Jacobian is singular at the apex (ζ=1). Gauss quadrature rules
    /// avoid qpts at ζ=1, so integration remains well-conditioned.  Using
    /// `GaussLobatto` quadrature (which includes the apex point) will
    /// produce a NaN Jacobian and an error.
    pub fn new(p: usize, q: usize, qmode: QuadMode, ncomp: usize) -> ReedResult<Self> {
        // Build internal hex Lagrange basis to get the 1D interp/grad data.
        let hex_basis = LagrangeBasis::<T>::new(3, 1, p, q, qmode)?;
        let num_dof = hex_basis.num_dof(); // p^3
        let num_qpoints = hex_basis.num_qpoints(); // q^3
        let hex_q_ref = hex_basis.q_ref(); // [nqpts × 3] in lexicographic (ξ,η,ζ)
        let hex_weights = hex_basis.q_weights(); // tensor-product 3D weights
        let interp_1d = hex_basis.interp_matrix(); // [q × p] 1D interp
        let grad_1d = hex_basis.grad_matrix(); // [q × p] 1D gradient

        let dim = 3;
        let p2 = p * p;
        let q2 = q * q;

        // --- Map quadrature points and compute Jacobian data ---------------
        let mut q_ref = Vec::with_capacity(num_qpoints * dim);
        let mut weights = Vec::with_capacity(num_qpoints);
        // J^{-T} stored as 3×3 row-major per qpt: [3*3 = 9] entries each
        let mut jac_inv_t = vec![T::ZERO; num_qpoints * 9];

        for iq in 0..num_qpoints {
            let xi: f64 = hex_q_ref[iq * 3].to_f64().unwrap();
            let eta: f64 = hex_q_ref[iq * 3 + 1].to_f64().unwrap();
            let zeta: f64 = hex_q_ref[iq * 3 + 2].to_f64().unwrap();
            let w: f64 = hex_weights[iq].to_f64().unwrap();

            // Pyramid coordinates via collapsed-coordinate mapping:
            //   r = ξ * (1-ζ) / 2
            //   s = η * (1-ζ) / 2
            //   t = ζ
            // Base (ζ=-1): r ∈ [-1,1], s ∈ [-1,1] square
            // Apex (ζ=1): r = s = 0 (collapsed)
            let one_minus_zeta = 1.0 - zeta;
            let half_omz = 0.5 * one_minus_zeta;

            let r = xi * half_omz;
            let s = eta * half_omz;
            let t = zeta;

            // Jacobian determinant: ((1-ζ)/2)^2
            let det_j = half_omz * half_omz;

            if det_j <= 0.0 {
                return Err(ReedError::Basis(format!(
                    "PyramidBasis: singular Jacobian at qpt {iq} (ξ={xi}, η={eta}, ζ={zeta}). \
                     Use Gauss quadrature to avoid apex point ζ=1."
                )));
            }

            // Inverse transpose Jacobian J^{-T}:
            // [  2/(1-ζ)     0        0    ]
            // [     0     2/(1-ζ)     0    ]
            // [ ξ/(1-ζ)   η/(1-ζ)     1    ]
            let jit_00 = 2.0 / one_minus_zeta;
            let jit_11 = jit_00;
            let jit_20 = xi / one_minus_zeta;
            let jit_21 = eta / one_minus_zeta;

            q_ref.push(to_scalar(r)?);
            q_ref.push(to_scalar(s)?);
            q_ref.push(to_scalar(t)?);
            weights.push(to_scalar(w * det_j)?);

            let jbase = iq * 9;
            jac_inv_t[jbase] = to_scalar(jit_00)?;
            jac_inv_t[jbase + 1] = T::ZERO;
            jac_inv_t[jbase + 2] = T::ZERO;
            jac_inv_t[jbase + 3] = T::ZERO;
            jac_inv_t[jbase + 4] = to_scalar(jit_11)?;
            jac_inv_t[jbase + 5] = T::ZERO;
            jac_inv_t[jbase + 6] = to_scalar(jit_20)?;
            jac_inv_t[jbase + 7] = to_scalar(jit_21)?;
            jac_inv_t[jbase + 8] = T::ONE;
        }

        // --- Build full 3D interpolation matrix (tensor product of 1D) -----
        // DOF: pz*p^2 + py*p + px, QPT: qz*q^2 + qy*q + qx
        let mut interp = vec![T::ZERO; num_qpoints * num_dof];
        for pz in 0..p {
            for py in 0..p {
                for px in 0..p {
                    let dof = pz * p2 + py * p + px;
                    for qz in 0..q {
                        let iz = qz * p + pz;
                        for qy in 0..q {
                            let iy = qy * p + py;
                            for qx in 0..q {
                                let ix = qx * p + px;
                                let qpt = qz * q2 + qy * q + qx;
                                interp[qpt * num_dof + dof] =
                                    interp_1d[ix] * interp_1d[iy] * interp_1d[iz];
                            }
                        }
                    }
                }
            }
        }

        // --- Build full 3D hex gradient and transform to pyramid gradient --
        // hex_grad component 0 (∂/∂ξ): grad_1d[x] * interp_1d[y]  * interp_1d[z]
        // hex_grad component 1 (∂/∂η): interp_1d[x] * grad_1d[y]  * interp_1d[z]
        // hex_grad component 2 (∂/∂ζ): interp_1d[x] * interp_1d[y] * grad_1d[z]
        // Then: g_pyr = J^{-T} · g_hex
        let mut grad = vec![T::ZERO; num_qpoints * num_dof * dim];
        for pz in 0..p {
            for py in 0..p {
                for px in 0..p {
                    let dof = pz * p2 + py * p + px;
                    for qz in 0..q {
                        let iz = qz * p + pz;
                        let iq_z = qz;
                        for qy in 0..q {
                            let iy = qy * p + py;
                            let iq_y = qy;

                            for qx in 0..q {
                                let ix = qx * p + px;
                                let iq_x = qx;
                                let qpt = iq_z * q2 + iq_y * q + iq_x;

                                // Hex gradient components
                                let g_xi = grad_1d[ix] * interp_1d[iy] * interp_1d[iz];
                                let g_eta = interp_1d[ix] * grad_1d[iy] * interp_1d[iz];
                                let g_zeta = interp_1d[ix] * interp_1d[iy] * grad_1d[iz];

                                // Compute J^{-T} at this qpt
                                let full_qpt = qpt;
                                let jb = full_qpt * 9;
                                let j00 = jac_inv_t[jb];
                                let j11 = jac_inv_t[jb + 4];
                                let j20 = jac_inv_t[jb + 6];
                                let j21 = jac_inv_t[jb + 7];

                                let base = (qpt * num_dof + dof) * dim;
                                grad[base] = j00 * g_xi;
                                grad[base + 1] = j11 * g_eta;
                                grad[base + 2] = j20 * g_xi + j21 * g_eta + g_zeta;
                            }
                        }
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

impl<T: Scalar> BasisTrait<T> for PyramidBasis<T> {
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
                        "PyramidBasis interp size mismatch: input {}, expected {}; output {}, expected {}",
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
                        "PyramidBasis grad size mismatch: input {}, expected {}; output {}, expected {}",
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
                        "PyramidBasis weight output length {} != expected {}",
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
                        "EvalMode::Div requires ncomp == dim for PyramidBasis".into(),
                    ));
                }
                let qcomp = self.ncomp * self.dim;
                if transpose {
                    let in_size = num_elem * self.num_qpoints;
                    let out_size = num_elem * self.num_dof * self.ncomp;
                    if u.len() != in_size || v.len() != out_size {
                        return Err(ReedError::Basis(format!(
                            "PyramidBasis div transpose size mismatch"
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
                            "PyramidBasis div size mismatch"
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
                                    "PyramidBasis curl transpose size mismatch"
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
                                    "PyramidBasis curl size mismatch"
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
                            "EvalMode::Curl requires (dim, ncomp) = (3, 3) for PyramidBasis".into(),
                        ));
                    }
                }
            }
            other => {
                return Err(ReedError::Basis(format!(
                    "PyramidBasis: eval mode {:?} not implemented",
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

// ── helper ─────────────────────────────────────────────────────────────────

fn to_scalar<T: Scalar>(value: f64) -> ReedResult<T> {
    T::from(value).ok_or_else(|| ReedError::Basis(format!("failed to convert {value} to scalar")))
}

// ── tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-11;

    // ── DOF counts ────────────────────────────────────────────────────────

    #[test]
    fn pyramid_dof_counts() {
        let b = PyramidBasis::<f64>::new(2, 2, QuadMode::Gauss, 1).unwrap();
        assert_eq!(b.dim(), 3);
        // p=2 -> 2^3 = 8 DOFs on the parent hex
        assert_eq!(b.num_dof(), 8);
        // q=2 -> 2^3 = 8 qpts
        assert_eq!(b.num_qpoints(), 8);
    }

    // ── quadrature weight sums ────────────────────────────────────────────

    #[test]
    fn pyramid_weight_sums_to_volume() {
        // Reference pyramid: base [-1,1]^2 at t=-1, apex (0,0,1).
        // Volume = (1/3) * base_area * height = (1/3) * 4 * 2 = 8/3
        let expected = 8.0 / 3.0;
        for q in [2usize, 3, 4] {
            let basis = PyramidBasis::<f64>::new(2, q, QuadMode::Gauss, 1).unwrap();
            let mut v = vec![0.0_f64; basis.num_qpoints()];
            basis.apply(1, false, EvalMode::Weight, &[], &mut v).unwrap();
            let sum: f64 = v.iter().sum();
            assert!(
                (sum - expected).abs() < 1e-10,
                "Pyramid q={q}: weight sum={sum}, expected {expected}"
            );
        }
    }

    // ── apex singularity check ────────────────────────────────────────────

    #[test]
    fn pyramid_gauss_qref_no_apex() {
        // Gauss quadrature points should all have ζ < 1 (no apex qpts)
        let basis = PyramidBasis::<f64>::new(2, 3, QuadMode::Gauss, 1).unwrap();
        for iq in 0..basis.num_qpoints() {
            let t = basis.q_ref[iq * 3 + 2];
            assert!(
                t < 1.0 - 1e-14,
                "Gauss qpt {iq} at t={t} is at or near apex (should be avoided)"
            );
        }
    }

    #[test]
    fn pyramid_lobatto_rejected() {
        // GaussLobatto includes ζ=1 (apex) -> singular Jacobian -> should error
        let result = PyramidBasis::<f64>::new(2, 3, QuadMode::GaussLobatto, 1);
        assert!(
            result.is_err(),
            "Pyramid with Gauss-Lobatto quadrature should be rejected"
        );
    }

    // ── constant interpolation ─────────────────────────────────────────────

    #[test]
    fn pyramid_interp_constant() {
        let basis = PyramidBasis::<f64>::new(2, 2, QuadMode::Gauss, 1).unwrap();
        let c = 3.5_f64;
        let u = vec![c; basis.num_dof()];
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        for (i, &val) in v.iter().enumerate() {
            assert!((val - c).abs() < TOL, "qpt {i}: got {val}, expected {c}");
        }
    }

    // ── linear interpolation ───────────────────────────────────────────────

    #[test]
    fn pyramid_interp_linear_r() {
        // On the hex, u = ξ. In pyramid space, r = (1+ξ)(1-ζ)/2 - 1.
        // At a given pyramid qpt, the interpolated value should be linear in r
        // when u varies only with ξ.
        let basis = PyramidBasis::<f64>::new(2, 2, QuadMode::Gauss, 1).unwrap();
        // p=2: hex nodes are [-1, 1] at each axis
        let mut u = vec![0.0_f64; basis.num_dof()];
        // Hex DOFs use lexicographic ordering: z major, y middle, x minor.
        // p=2: 2 nodes per axis -> 8 DOFs.
        // DOF (px, py, pz): px={-1,1}, py={-1,1}, pz={-1,1}
        for pz in 0..2 {
            for py in 0..2 {
                for px in 0..2 {
                    let xi_val = if px == 0 { -1.0_f64 } else { 1.0 };
                    let dof = pz * 4 + py * 2 + px;
                    u[dof] = xi_val;
                }
            }
        }
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        // At each qpt, the hex qpt has ξ coordinate. The value should equal ξ.
        let hex_basis = LagrangeBasis::<f64>::new(3, 1, 2, 2, QuadMode::Gauss).unwrap();
        let mut hex_v = vec![0.0_f64; hex_basis.num_qpoints()];
        hex_basis.apply(1, false, EvalMode::Interp, &u, &mut hex_v).unwrap();
        for (iq, &val) in v.iter().enumerate() {
            let expected = hex_v[iq]; // should be the ξ coordinate
            assert!(
                (val - expected).abs() < TOL,
                "qpt {iq}: got {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn pyramid_interp_linear_t() {
        // u = ζ on hex nodes. On pyramid, t = ζ.
        let basis = PyramidBasis::<f64>::new(2, 2, QuadMode::Gauss, 1).unwrap();
        let mut u = vec![0.0_f64; basis.num_dof()];
        for pz in 0..2 {
            for py in 0..2 {
                for px in 0..2 {
                    let zeta_val = if pz == 0 { -1.0_f64 } else { 1.0 };
                    let dof = pz * 4 + py * 2 + px;
                    u[dof] = zeta_val;
                }
            }
        }
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        let hex_basis = LagrangeBasis::<f64>::new(3, 1, 2, 2, QuadMode::Gauss).unwrap();
        let mut hex_v = vec![0.0_f64; hex_basis.num_qpoints()];
        hex_basis.apply(1, false, EvalMode::Interp, &u, &mut hex_v).unwrap();
        for (iq, &val) in v.iter().enumerate() {
            let expected = hex_v[iq]; // t = ζ
            assert!(
                (val - expected).abs() < TOL,
                "qpt {iq}: got {val}, expected {expected}"
            );
        }
    }

    // ── gradient transformation ────────────────────────────────────────────

    #[test]
    fn pyramid_grad_xi() {
        // f = ξ (linear in hex coordinates).
        // In hex space: ∇_hex f = (1, 0, 0).
        // In pyramid space: ∇_pyr f = J^{-T} * (1, 0, 0)^T
        //   gr = 2/(1-ζ), gs = 0, gt = ξ/(1-ζ)
        let basis = PyramidBasis::<f64>::new(2, 2, QuadMode::Gauss, 1).unwrap();
        let mut u = vec![0.0_f64; basis.num_dof()];
        let hex_basis = LagrangeBasis::<f64>::new(3, 1, 2, 2, QuadMode::Gauss).unwrap();
        for pz in 0..2 {
            for py in 0..2 {
                for px in 0..2 {
                    let xi = if px == 0 { -1.0_f64 } else { 1.0 };
                    let dof = pz * 4 + py * 2 + px;
                    u[dof] = xi;
                }
            }
        }
        let mut grad_out = vec![0.0_f64; basis.num_qpoints() * 3];
        basis.apply(1, false, EvalMode::Grad, &u, &mut grad_out).unwrap();
        for iq in 0..basis.num_qpoints() {
            let hex_xi = hex_basis.q_ref()[iq * 3];
            let hex_zeta = hex_basis.q_ref()[iq * 3 + 2];
            let expected_gr = 2.0 / (1.0 - hex_zeta);
            let expected_gt = hex_xi / (1.0 - hex_zeta);
            let gr = grad_out[iq * 3];
            let gs = grad_out[iq * 3 + 1];
            let gt = grad_out[iq * 3 + 2];
            assert!((gr - expected_gr).abs() < 1e-10,
                "qpt {iq} ∂r: got {gr}, expected {expected_gr}");
            assert!(gs.abs() < 1e-10,
                "qpt {iq} ∂s: got {gs}, expected 0");
            assert!((gt - expected_gt).abs() < 1e-10,
                "qpt {iq} ∂t: got {gt}, expected {expected_gt}");
        }
    }

    #[test]
    fn pyramid_grad_t() {
        // f = t (in pyramid space, t = ζ).
        // ∂f/∂r = 0, ∂f/∂s = 0, ∂f/∂t = 1 in pyramid space.
        let basis = PyramidBasis::<f64>::new(2, 2, QuadMode::Gauss, 1).unwrap();
        let mut u = vec![0.0_f64; basis.num_dof()];
        for pz in 0..2 {
            for py in 0..2 {
                for px in 0..2 {
                    let zeta = if pz == 0 { -1.0_f64 } else { 1.0 };
                    let dof = pz * 4 + py * 2 + px;
                    u[dof] = zeta; // t = ζ
                }
            }
        }
        let mut grad_out = vec![0.0_f64; basis.num_qpoints() * 3];
        basis.apply(1, false, EvalMode::Grad, &u, &mut grad_out).unwrap();
        for iq in 0..basis.num_qpoints() {
            let gr = grad_out[iq * 3];
            let gs = grad_out[iq * 3 + 1];
            let gt = grad_out[iq * 3 + 2];
            assert!(gr.abs() < 1e-10, "qpt {iq} ∂r: got {gr}, expected 0");
            assert!(gs.abs() < 1e-10, "qpt {iq} ∂s: got {gs}, expected 0");
            assert!((gt - 1.0).abs() < 1e-10, "qpt {iq} ∂t: got {gt}, expected 1");
        }
    }

    // ── linear interpolation in pyramid space ─────────────────────────────

    #[test]
    fn pyramid_interp_xi_at_qpts() {
        // Interpolate f = ξ (linear in hex coordinates) and verify
        // against the hex ξ coordinate at each qpt.
        let basis = PyramidBasis::<f64>::new(2, 3, QuadMode::Gauss, 1).unwrap();
        let p = 2; // 2 nodes per direction
        let mut u = vec![0.0_f64; basis.num_dof()];
        for pz in 0..p {
            for py in 0..p {
                for px in 0..p {
                    let xi = if px == 0 { -1.0_f64 } else { 1.0 };
                    let dof = pz * p * p + py * p + px;
                    u[dof] = xi;
                }
            }
        }
        // Get hex basis to access hex q_ref for ground truth
        let hex_basis = LagrangeBasis::<f64>::new(3, 1, 2, 3, QuadMode::Gauss).unwrap();
        let mut v = vec![0.0_f64; basis.num_qpoints()];
        basis.apply(1, false, EvalMode::Interp, &u, &mut v).unwrap();
        for (iq, &val) in v.iter().enumerate() {
            let xi = hex_basis.q_ref()[iq * 3];
            assert!((val - xi).abs() < 1e-10, "qpt {iq}: got {val}, expected xi={xi}");
        }
    }

    // ── adjoint identities ─────────────────────────────────────────────────

    #[test]
    fn pyramid_div_adjoint_identity() {
        let basis = PyramidBasis::<f64>::new(2, 2, QuadMode::Gauss, 3).unwrap();
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
    fn pyramid_curl_adjoint_identity() {
        let basis = PyramidBasis::<f64>::new(2, 2, QuadMode::Gauss, 3).unwrap();
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

    // ── r-coordinate mapping consistency ───────────────────────────────────

    #[test]
    fn pyramid_qref_mapping_consistency() {
        let basis = PyramidBasis::<f64>::new(2, 3, QuadMode::Gauss, 1).unwrap();
        // Verify that q_ref points lie within the pyramid:
        // base: -1 <= r <= 1, -1 <= s <= 1 at t=-1
        // apex: r=0, s=0 at t=1
        // Mapping: r = ξ(1-ζ)/2, s = η(1-ζ)/2, t = ζ
        // So |r| <= (1-ζ)/2 = (1-t)/2
        for iq in 0..basis.num_qpoints() {
            let r = basis.q_ref[iq * 3];
            let s = basis.q_ref[iq * 3 + 1];
            let t = basis.q_ref[iq * 3 + 2];
            assert!((-1.0..=1.0).contains(&r), "qpt {iq}: r={r} out of [-1,1]");
            assert!((-1.0..=1.0).contains(&s), "qpt {iq}: s={s} out of [-1,1]");
            assert!((-1.0..=1.0).contains(&t), "qpt {iq}: t={t} out of [-1,1]");
            // r and s should shrink toward zero as t -> 1
            let max_rs = (1.0 - t) / 2.0; // max |r|,|s| at this t level
            assert!(
                r.abs() <= max_rs + 1e-14,
                "qpt {iq}: |r|={} > max_rs={max_rs} at t={t}",
                r.abs()
            );
            assert!(
                s.abs() <= max_rs + 1e-14,
                "qpt {iq}: |s|={} > max_rs={max_rs} at t={t}",
                s.abs()
            );
        }
    }
}
