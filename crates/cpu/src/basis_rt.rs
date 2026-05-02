//! Raviart-Thomas H(div) basis functions for triangles and tetrahedra.
//!
//! Implements [`BasisTrait`] for the lowest-order Raviart-Thomas (RT0/R1) basis on
//! simplex reference elements:
//!
//! | Type | Topology | DOFs | Polynomial space |
//! |------|----------|------|------------------|
//! | P0 triangle | Tri3 | 3 | RT0 (edge) |
//! | P1 triangle | Tri3 | 8 | RT1 (edge + face) |
//! | P2 triangle | Tri3 | 15 | RT2 (edge + face) |
//! | P0 tet | Tet4 | 4 | RT0 (face) |
//! | P1 tet | Tet4 | 20 | RT1 (face + interior) |
//!
//! ## Reference elements
//!
//! **Triangle** — vertices (0,0), (1,0), (0,1).  Area |T| = 1/2.
//!
//! **Tetrahedron** — vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1).  Volume |T| = 1/6.
//!
//! ## Basis functions
//!
//! ### RT0 (order 0)
//!
//! RT0 basis functions on a simplex K are of the form
//! ψ_i = (x − x_i) / (d · |K|)
//! where x_i is vertex i and d is the spatial dimension.
//!
//! Each ψ_i has unit normal flux through the face/edge opposite vertex i and
//! zero flux through all other faces/edges.
//!
//! ### RT1 (order 1) — Triangle and Tet
//!
//! Hierarchical basis.
//!
//! **Triangle** — 8 DOFs: 2 per edge (6) + 2 interior (2).
//!
//! **Edge DOFs** (0–5):
//! - ψ_k^{(0)} = x − x_k  (RT0, DOFs 0–2, constant normal flux on edge k)
//! - ψ_k^{(1)} = (λ_i − λ_j) · ψ_k^{(0)}  (linear moment, DOFs 3–5)
//!   where i, j are the vertices adjacent to edge k (the ones NOT opposite).
//!
//! **Interior DOFs** (6–7): rot90 of Nédélec face functions (zero normal flux on edges)
//! - ψ_int^{(1)} = rot90(λ_0 λ_1 ∇λ_2) = (−λ_0 λ_1, 0)
//! - ψ_int^{(2)} = rot90(λ_0 λ_2 ∇λ_1) = (0, λ_0 λ_2)
//!
//! ### RT2 (order 2) — Triangle
//!
//! Hierarchical extension of RT1 with 15 DOFs.
//!
//! **Edge DOFs** (0–8): 3 per edge
//! - DOF 0–2: RT0 ψ_k^{(0)} = x − x_k (unchanged)
//! - DOF 3–5: RT1 ψ_k^{(1)} = (λ_i − λ_j) ψ_k^{(0)} (unchanged)
//! - DOF 8–10: RT2 ψ_k^{(2)} = (λ_i − λ_j)^2 ψ_k^{(0)}
//!
//! **Interior DOFs** (6–7 RT1, 11–14 RT2):
//! - DOF 6–7: rot90 of N2 face functions (unchanged)
//! - DOF 11–14: rot90 of N3 face functions (higher-order bubbles)
//!
//! RT1 is a direct subspace: DOFs 0–7 match the RT1 basis exactly.
//!
//! **Tetrahedron** — 20 DOFs: 3 per face (12) + 8 interior (8).
//!
//! **Face DOFs** (0–11):
//! - ψ_i = 2(x − x_i)  (RT0, DOFs 0–3, constant normal flux on face i)
//! - λ_j · ψ_i  (linear moment, DOFs 4–11, 2 per face × 4 faces)
//!
//! **Interior DOFs** (12–19): curl of Nédélec face functions (divergence-free)
//! - For each of the 8 Nédélec face functions ψ = λ_a λ_b ∇λ_c,
//!   ψ_int = curl(ψ) = ∇(λ_a λ_b) × ∇λ_c
//!   These have div = 0 and zero normal flux on all faces.
//!
//! ## Memory layout
//!
//! * `interp` — row-major `[nqpts × num_dof × dim]`,
//!   stored as `(qpt*num_dof + dof)*dim + d`
//! * `div_matrix` — `[nqpts × num_dof]` (scalar divergence at each q-pt)

use reed_core::{
    basis::BasisTrait,
    enums::{ElemTopology, EvalMode},
    error::{ReedError, ReedResult},
    scalar::Scalar,
};

use super::basis_simplex::{tet_quadrature, tri_quadrature};

#[cfg(feature = "parallel")]
const PAR_MIN_ELEMS_PER_TASK: usize = 128;

/// H(div) Raviart-Thomas (RT0/RT1) basis on triangles and tetrahedra.
pub struct RaviartThomasBasis<T: Scalar> {
    dim: usize,
    num_dof: usize,
    num_qpoints: usize,
    /// Quadrature weights, length `num_qpoints`.
    weights: Vec<T>,
    /// Quadrature point coordinates, row-major `[num_qpoints × dim]`.
    q_ref: Vec<T>,
    /// Interpolation matrix, row-major `[num_qpoints × num_dof × dim]`.
    interp: Vec<T>,
    /// Divergence matrix, `[num_qpoints × num_dof]` (scalar divergence).
    div_matrix: Vec<T>,
}

impl<T: Scalar> RaviartThomasBasis<T> {
    /// Construct a Raviart-Thomas H(div) basis.
    ///
    /// # Parameters
    /// * `topo` — `ElemTopology::Triangle` or `Tet`.
    /// * `p`    — polynomial order. Triangle: 0, 1, or 2; Tet: 0 or 1.
    /// * `q`    — number of quadrature points (see `tri_quadrature` / `tet_quadrature` for
    ///            valid values).
    ///
    /// # Errors
    /// Returns `ReedError::Basis` for unsupported topology/p/q combinations.
    pub fn new(topo: ElemTopology, p: usize, q: usize) -> ReedResult<Self> {
        let (dim, num_dof) = match topo {
            ElemTopology::Triangle => match p {
                0 => (2, 3),
                1 => (2, 8),
                2 => (2, 15),
                _ => {
                    return Err(ReedError::Basis(format!(
                        "RaviartThomasBasis: p={p} on Triangle not supported; use p=0, 1, or 2"
                    )));
                }
            },
            ElemTopology::Tet => match p {
                0 => (3, 4),
                1 => (3, 20),
                _ => {
                    return Err(ReedError::Basis(format!(
                        "RaviartThomasBasis: p={p} on Tet not supported; use p=0 (RT0) or p=1 (RT1)"
                    )));
                }
            },
            _ => {
                if matches!(topo, ElemTopology::Pyramid | ElemTopology::Prism) {
                    return Err(ReedError::Basis(format!(
                        "RaviartThomasBasis: {:?} not implemented (requires collapsed-coordinate or tensor×simplex transforms; available: Triangle, Tet)",
                        topo
                    )));
                }
                return Err(ReedError::Basis(format!(
                    "RaviartThomasBasis: unsupported topology {:?} (need Triangle or Tet)",
                    topo
                )))
            }
        };

        // Quadrature rule
        let (q_ref_f64, weights_f64) = match topo {
            ElemTopology::Triangle => tri_quadrature(q)?,
            ElemTopology::Tet => tet_quadrature(q)?,
            _ => unreachable!(),
        };
        let num_qpoints = q_ref_f64.len() / dim;

        let order = p; // captured for use in the evaluation loop below

        // Convert to target scalar type
        let q_ref: Vec<T> = q_ref_f64
            .iter()
            .map(|&v| to_t::<T>(v))
            .collect::<ReedResult<_>>()?;
        let weights: Vec<T> = weights_f64
            .iter()
            .map(|&v| to_t::<T>(v))
            .collect::<ReedResult<_>>()?;

        // Pack quadrature point coordinates as [x,y] or [x,y,z] for evaluation
        let qpts: Vec<[f64; 3]> = (0..num_qpoints)
            .map(|qi| {
                let mut pt = [0.0f64; 3];
                for d in 0..dim {
                    pt[d] = q_ref_f64[qi * dim + d];
                }
                pt
            })
            .collect();

        // Build interp and divergence matrices
        let mut interp = vec![0.0f64; num_qpoints * num_dof * dim];
        let mut div_matrix = vec![0.0f64; num_qpoints * num_dof];

        for (qi, pt) in qpts.iter().enumerate() {
            match dim {
                2 => {
                    let (phi, div) = match order {
                        2 => tri_rt2(pt[0], pt[1]),
                        1 => tri_rt1(pt[0], pt[1]),
                        _ => tri_rt0(pt[0], pt[1]),
                    };
                    for dof in 0..num_dof {
                        for d in 0..dim {
                            interp[(qi * num_dof + dof) * dim + d] =
                                phi[dof * dim + d];
                        }
                        div_matrix[qi * num_dof + dof] = div[dof];
                    }
                }
                3 => {
                    let (phi, div) = if order == 1 {
                        tet_rt1(pt[0], pt[1], pt[2])
                    } else {
                        let (p, d) = tet_rt0(pt[0], pt[1], pt[2]);
                        (p, d)
                    };
                    for dof in 0..num_dof {
                        for d in 0..dim {
                            interp[(qi * num_dof + dof) * dim + d] =
                                phi[dof * dim + d];
                        }
                        div_matrix[qi * num_dof + dof] = div[dof];
                    }
                }
                _ => unreachable!(),
            }
        }

        let interp_t: Vec<T> = interp
            .iter()
            .map(|&v| to_t::<T>(v))
            .collect::<ReedResult<_>>()?;
        let div_t: Vec<T> = div_matrix
            .iter()
            .map(|&v| to_t::<T>(v))
            .collect::<ReedResult<_>>()?;

        Ok(Self {
            dim,
            num_dof,
            num_qpoints,
            weights,
            q_ref,
            interp: interp_t,
            div_matrix: div_t,
        })
    }

    // ── public data accessors ──────────────────────────────────────────────

    /// Raw interp matrix: `[(qpt * num_dof + dof) * dim + d]`,
    /// length `num_qpoints * num_dof * dim`.
    #[inline]
    pub fn interp_data(&self) -> &[T] {
        &self.interp
    }

    /// Raw divergence matrix: `[qpt * num_dof + dof]` (scalar), length `nq × ndof`.
    #[inline]
    pub fn div_data(&self) -> &[T] {
        &self.div_matrix
    }

    // ── accessor helpers ───────────────────────────────────────────────────

    /// `interp[(qpt * num_dof + dof) * dim + d]`
    #[inline]
    fn interp_val(&self, qpt: usize, dof: usize, d: usize) -> T {
        self.interp[(qpt * self.num_dof + dof) * self.dim + d]
    }

    /// `div_matrix[qpt * num_dof + dof]` (scalar divergence).
    #[inline]
    fn div_val(&self, qpt: usize, dof: usize) -> T {
        self.div_matrix[qpt * self.num_dof + dof]
    }

    // ── element-level apply helpers ────────────────────────────────────────

    /// Forward interp: scalar DOFs → vector field at qpts.
    /// u_elem: `[num_dof * dim]` — each DOF has `dim` entries (redundant scalar).
    ///          Read `u_elem[dof * self.dim]` as the scalar DOF value.
    /// v_elem: `[num_qpoints * dim]` — vector values at quadrature points.
    fn apply_interp_forward(&self, u_elem: &[T], v_elem: &mut [T]) {
        for qpt in 0..self.num_qpoints {
            for d in 0..self.dim {
                let mut sum = T::ZERO;
                for dof in 0..self.num_dof {
                    sum += self.interp_val(qpt, dof, d) * u_elem[dof * self.dim];
                }
                v_elem[qpt * self.dim + d] = sum;
            }
        }
    }

    /// Transpose interp: vector field at qpts → scalar DOFs.
    /// u_elem: `[num_qpoints * dim]` — vector values at quadrature points.
    /// v_elem: `[num_dof * dim]` — accumulator; writes scalar result to
    ///          first component of each DOF (`v_elem[dof * self.dim]`).
    fn apply_interp_transpose(&self, u_elem: &[T], v_elem: &mut [T]) {
        for dof in 0..self.num_dof {
            let mut sum = T::ZERO;
            for qpt in 0..self.num_qpoints {
                for d in 0..self.dim {
                    sum += self.interp_val(qpt, dof, d) * u_elem[qpt * self.dim + d];
                }
            }
            v_elem[dof * self.dim] += sum;
        }
    }

    /// Forward HDiv: scalar DOFs → scalar divergence at qpts.
    /// u_elem: `[num_dof * dim]`, read `u_elem[dof * self.dim]` as scalar.
    /// v_elem: `[num_qpoints]` — scalar divergence values.
    fn apply_hdiv_forward(&self, u_elem: &[T], v_elem: &mut [T]) {
        for qpt in 0..self.num_qpoints {
            let mut sum = T::ZERO;
            for dof in 0..self.num_dof {
                sum += self.div_val(qpt, dof) * u_elem[dof * self.dim];
            }
            v_elem[qpt] = sum;
        }
    }

    /// Transpose HDiv: scalar divergence at qpts → scalar DOFs.
    /// u_elem: `[num_qpoints]` — scalar divergence values.
    /// v_elem: `[num_dof * dim]` — accumulator; writes to `v_elem[dof * self.dim]`.
    fn apply_hdiv_transpose(&self, u_elem: &[T], v_elem: &mut [T]) {
        for dof in 0..self.num_dof {
            let mut sum = T::ZERO;
            for qpt in 0..self.num_qpoints {
                sum += self.div_val(qpt, dof) * u_elem[qpt];
            }
            v_elem[dof * self.dim] += sum;
        }
    }
}

// ── BasisTrait impl ───────────────────────────────────────────────────────────

impl<T: Scalar> BasisTrait<T> for RaviartThomasBasis<T> {
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
        self.dim // vector-valued basis
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
                    self.num_qpoints * self.dim
                } else {
                    self.num_dof * self.dim
                };
                let out_stride = if transpose {
                    self.num_dof * self.dim
                } else {
                    self.num_qpoints * self.dim
                };
                check_sizes(u, in_stride * num_elem, v, out_stride * num_elem, "interp")?;
                if transpose {
                    v.fill(T::ZERO);
                }
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    if transpose {
                        u.par_chunks(in_stride)
                            .zip(v.par_chunks_mut(out_stride))
                            .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                            .for_each(|(u_elem, v_elem)| {
                                self.apply_interp_transpose(u_elem, v_elem)
                            });
                    } else {
                        u.par_chunks(in_stride)
                            .zip(v.par_chunks_mut(out_stride))
                            .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                            .for_each(|(u_elem, v_elem)| {
                                self.apply_interp_forward(u_elem, v_elem)
                            });
                    }
                }
                #[cfg(not(feature = "parallel"))]
                {
                    if transpose {
                        for (u_elem, v_elem) in u.chunks(in_stride).zip(v.chunks_mut(out_stride)) {
                            self.apply_interp_transpose(u_elem, v_elem);
                        }
                    } else {
                        for (u_elem, v_elem) in u.chunks(in_stride).zip(v.chunks_mut(out_stride)) {
                            self.apply_interp_forward(u_elem, v_elem);
                        }
                    }
                }
            }
            EvalMode::HDiv => {
                // HDiv is always scalar-valued (divergence of a vector field is a scalar)
                let in_stride = if transpose {
                    self.num_qpoints
                } else {
                    self.num_dof * self.dim
                };
                let out_stride = if transpose {
                    self.num_dof * self.dim
                } else {
                    self.num_qpoints
                };
                check_sizes(
                    u,
                    in_stride * num_elem,
                    v,
                    out_stride * num_elem,
                    "hdiv",
                )?;
                if transpose {
                    v.fill(T::ZERO);
                }
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    if transpose {
                        u.par_chunks(in_stride)
                            .zip(v.par_chunks_mut(out_stride))
                            .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                            .for_each(|(u_elem, v_elem)| {
                                self.apply_hdiv_transpose(u_elem, v_elem)
                            });
                    } else {
                        u.par_chunks(in_stride)
                            .zip(v.par_chunks_mut(out_stride))
                            .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                            .for_each(|(u_elem, v_elem)| {
                                self.apply_hdiv_forward(u_elem, v_elem)
                            });
                    }
                }
                #[cfg(not(feature = "parallel"))]
                {
                    if transpose {
                        for (u_elem, v_elem) in
                            u.chunks(in_stride).zip(v.chunks_mut(out_stride))
                        {
                            self.apply_hdiv_transpose(u_elem, v_elem);
                        }
                    } else {
                        for (u_elem, v_elem) in
                            u.chunks(in_stride).zip(v.chunks_mut(out_stride))
                        {
                            self.apply_hdiv_forward(u_elem, v_elem);
                        }
                    }
                }
            }
            EvalMode::Weight => {
                if transpose {
                    // Weight transpose delegates to interp transpose
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
            other => {
                return Err(ReedError::Basis(format!(
                    "RaviartThomasBasis: eval mode {:?} not implemented",
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

// ── shape functions ───────────────────────────────────────────────────────────

/// RT0 basis functions on the reference triangle.
///
/// Basis: ψ_i = (x − x_i) / (d · |T|) with d=2, |T|=1/2 → ψ_i = x − x_i.
///
/// Vertices: x₀=(0,0), x₁=(1,0), x₂=(0,1).
/// - ψ₀ = (x, y)
/// - ψ₁ = (x−1, y)
/// - ψ₂ = (x, y−1)
///
/// Divergence is constant: ∇·ψ_i = 1/|T| = 2 for all i.
///
/// Returns `(phi[num_dof * 2], div[num_dof])`.
fn tri_rt0(x: f64, y: f64) -> (Vec<f64>, Vec<f64>) {
    let verts: [[f64; 2]; 3] = [
        [0.0, 0.0], // x₀
        [1.0, 0.0], // x₁
        [0.0, 1.0], // x₂
    ];
    let num_dof = 3;
    let dim = 2;
    let div_const = 2.0; // 1/|T| = 2

    let mut phi = vec![0.0f64; num_dof * dim];
    let mut div = vec![0.0f64; num_dof];

    for dof in 0..num_dof {
        for d in 0..dim {
            // ψ_i = x − x_i  (since factor = 1)
            phi[dof * dim + d] = [x, y][d] - verts[dof][d];
        }
        div[dof] = div_const;
    }

    (phi, div)
}

/// RT1 basis functions on the reference triangle (8 DOFs).
///
/// Hierarchical construction extending RT0 with linear edge moments and
/// divergence-free interior bubble functions.
///
/// **Edge DOFs** (0–5):
/// - DOF 0–2: RT0 edge basis  ψ_k^{(0)} = x − x_k  (constant normal flux)
/// - DOF 3–5: linear edge moments  ψ_k^{(1)} = (λ_i − λ_j) · ψ_k^{(0)}
///   where i, j are the vertices of edge k.
///
///   Edge 0 (opposite v0, between v1 and v2): ψ_0^{(1)} = (x−y)·(x, y)
///   Edge 1 (opposite v1, between v0 and v2): ψ_1^{(1)} = (2y+x−1)·(x−1, y)
///   Edge 2 (opposite v2, between v0 and v1): ψ_2^{(1)} = (1−2x−y)·(x, y−1)
///
/// **Interior DOFs** (6–7):
/// - DOF 6: ψ_int^{(1)} = rot90(λ_0 λ_1 ∇λ_2) = (−(1−x−y)x, 0)
/// - DOF 7: ψ_int^{(2)} = rot90(λ_0 λ_2 ∇λ_1) = (0, (1−x−y)y)
///
/// Returns `(phi[num_dof * 2], div[num_dof])`.
fn tri_rt1(x: f64, y: f64) -> (Vec<f64>, Vec<f64>) {
    let verts: [[f64; 2]; 3] = [
        [0.0, 0.0], // x₀
        [1.0, 0.0], // x₁
        [0.0, 1.0], // x₂
    ];
    let num_dof = 8;
    let dim = 2;

    let lam = [1.0 - x - y, x, y]; // λ₀, λ₁, λ₂

    let mut phi = vec![0.0f64; num_dof * dim];
    let mut div = vec![0.0f64; num_dof];

    // ── RT0 edge functions (DOFs 0–2) ───────────────────────────────────
    // ψ_k = x − x_k; div = 2
    for dof in 0..3 {
        for d in 0..dim {
            phi[dof * dim + d] = [x, y][d] - verts[dof][d];
        }
        div[dof] = 2.0;
    }

    // ── Linear edge moments (DOFs 3–5) ──────────────────────────────────
    // ψ_k^{(1)} = (λ_i − λ_j) · ψ_k^{(0)} where edge k connects vertices i,j
    // Edge 0: connects v1=(1,0) to v2=(0,1), i=1, j=2
    //   factor = λ_1 − λ_2 = x − y
    {
        let dof = 3;
        let f = lam[1] - lam[2]; // x - y
        for d in 0..dim {
            let rt0_val = [x, y][d] - verts[0][d]; // ψ_0^{(0)} = (x, y)
            phi[dof * dim + d] = f * rt0_val;
        }
        // div = 3(x − y)
        div[dof] = 3.0 * (x - y);
    }
    // Edge 1: connects v0=(0,0) to v2=(0,1), i=0, j=2
    //   factor = λ_0 − λ_2 = (1-x-y) − y = 1 − x − 2y
    //   Wait, that gives 1-x-2y. But I computed 2y+x-1 earlier. Let me verify:
    //   ψ_1^{(0)} = (x-1, y)
    //   f = λ_2 - λ_0 = y - (1-x-y) = 2y + x - 1
    //   Actually I had ψ_1^{(1)} = (2y+x-1)*(x-1, y)
    //   So the factor is λ_2 - λ_0 = x + 2y - 1
    {
        let dof = 4;
        let f = lam[2] - lam[0]; // y - (1-x-y) = x + 2y - 1
        for d in 0..dim {
            let rt0_val = [x, y][d] - verts[1][d]; // ψ_1^{(0)} = (x-1, y)
            phi[dof * dim + d] = f * rt0_val;
        }
        // div = 3(x + 2y − 1)
        div[dof] = 3.0 * (x + 2.0 * y - 1.0);
    }
    // Edge 2: connects v0=(0,0) to v1=(1,0), i=0, j=1
    //   factor = λ_0 − λ_1 = (1-x-y) − x = 1 − 2x − y
    {
        let dof = 5;
        let f = lam[0] - lam[1]; // 1 - x - y - x = 1 - 2x - y
        for d in 0..dim {
            let rt0_val = [x, y][d] - verts[2][d]; // ψ_2^{(0)} = (x, y-1)
            phi[dof * dim + d] = f * rt0_val;
        }
        // div = 3(1 − 2x − y)
        div[dof] = 3.0 * (1.0 - 2.0 * x - y);
    }

    // ── Interior bubble functions (DOFs 6–7) ────────────────────────────
    // DOF 6: ψ_int^{(1)} = rot90(λ_0 λ_1 ∇λ_2) = (−(1−x−y)x, 0)
    {
        let dof = 6;
        phi[dof * dim] = -(1.0 - x - y) * x; // x-component
        phi[dof * dim + 1] = 0.0; // y-component
        // div = ∂/∂x(−(1−x−y)x) = −(1−x−y) + x = 2x + y − 1
        div[dof] = 2.0 * x + y - 1.0;
    }
    // DOF 7: ψ_int^{(2)} = rot90(λ_0 λ_2 ∇λ_1) = (0, (1−x−y)y)
    {
        let dof = 7;
        phi[dof * dim] = 0.0; // x-component
        phi[dof * dim + 1] = (1.0 - x - y) * y; // y-component
        // div = ∂/∂y((1−x−y)y) = (1−x−y) − y = 1 − x − 2y
        div[dof] = 1.0 - x - 2.0 * y;
    }

    (phi, div)
}

/// RT2 basis functions on the reference triangle (15 DOFs).
///
/// Hierarchical construction extending RT1 with quadratic edge moments and
/// higher-order interior bubble functions.
///
/// **DOF layout** (RT1-subspace preserving):
/// - DOF 0–2: RT0 edge basis  ψ_k^{(0)} = x − x_k
/// - DOF 3–5: RT1 linear edge moments  ψ_k^{(1)} = (λ_i − λ_j) ψ_k^{(0)}
/// - DOF 6–7: RT1 interior = rot90(N2 face)
/// - DOF 8–10: RT2 quadratic edge moments  ψ_k^{(2)} = (λ_i − λ_j)^2 ψ_k^{(0)}
/// - DOF 11–14: RT2 interior = rot90(N3 face)
///
/// ## Divergence formulas
///
/// RT2 edge: div((λ_i−λ_j)^2 ψ_k) = 4(λ_i−λ_j)^2  (since div(ψ_k)=2, ∇f·ψ_k=f)
///
/// RT2 interior: div = −curl(N3 face DOF) in 2D
///
/// Returns `(phi[num_dof * 2], div[num_dof])`.
fn tri_rt2(x: f64, y: f64) -> (Vec<f64>, Vec<f64>) {
    let verts: [[f64; 2]; 3] = [
        [0.0, 0.0], // x₀
        [1.0, 0.0], // x₁
        [0.0, 1.0], // x₂
    ];
    let num_dof = 15;
    let dim = 2;

    let lam = [1.0 - x - y, x, y]; // λ₀, λ₁, λ₂

    let mut phi = vec![0.0f64; num_dof * dim];
    let mut div = vec![0.0f64; num_dof];

    // ── RT0 edge functions (DOFs 0–2) ───────────────────────────────────
    for dof in 0..3 {
        for d in 0..dim {
            phi[dof * dim + d] = [x, y][d] - verts[dof][d];
        }
        div[dof] = 2.0;
    }

    // ── RT1 linear edge moments (DOFs 3–5) ──────────────────────────────
    // Edge 0 (opposite v0): DOF 3, factor = λ_1 − λ_2 = x − y
    {
        let dof = 3;
        let f = lam[1] - lam[2];
        for d in 0..dim {
            let rt0_val = [x, y][d] - verts[0][d];
            phi[dof * dim + d] = f * rt0_val;
        }
        div[dof] = 3.0 * (x - y);
    }
    // Edge 1 (opposite v1): DOF 4, factor = λ_2 − λ_0 = x + 2y − 1
    {
        let dof = 4;
        let f = lam[2] - lam[0];
        for d in 0..dim {
            let rt0_val = [x, y][d] - verts[1][d];
            phi[dof * dim + d] = f * rt0_val;
        }
        div[dof] = 3.0 * (x + 2.0 * y - 1.0);
    }
    // Edge 2 (opposite v2): DOF 5, factor = λ_0 − λ_1 = 1 − 2x − y
    {
        let dof = 5;
        let f = lam[0] - lam[1];
        for d in 0..dim {
            let rt0_val = [x, y][d] - verts[2][d];
            phi[dof * dim + d] = f * rt0_val;
        }
        div[dof] = 3.0 * (1.0 - 2.0 * x - y);
    }

    // ── RT1 interior bubble functions (DOFs 6–7) ─────────────────────────
    // DOF 6: rot90(λ_0 λ_1 ∇λ_2) = (−(1−x−y)x, 0)
    {
        let dof = 6;
        phi[dof * dim] = -(1.0 - x - y) * x;
        phi[dof * dim + 1] = 0.0;
        div[dof] = 2.0 * x + y - 1.0;
    }
    // DOF 7: rot90(λ_0 λ_2 ∇λ_1) = (0, (1−x−y)y)
    {
        let dof = 7;
        phi[dof * dim] = 0.0;
        phi[dof * dim + 1] = (1.0 - x - y) * y;
        div[dof] = 1.0 - x - 2.0 * y;
    }

    // ── RT2 quadratic edge moments (DOFs 8–10) ──────────────────────────
    // ψ_k^{(2)} = (λ_i − λ_j)^2 · ψ_k^{(0)}
    // Edge 0 (opposite v0): (λ_1 − λ_2)^2 · (x, y)
    {
        let dof = 8;
        let f = lam[1] - lam[2];
        let f2 = f * f;
        for d in 0..dim {
            let rt0_val = [x, y][d] - verts[0][d];
            phi[dof * dim + d] = f2 * rt0_val;
        }
        div[dof] = 4.0 * f2;
    }
    // Edge 1 (opposite v1): (λ_0 − λ_2)^2 · (x−1, y)
    {
        let dof = 9;
        let f = lam[0] - lam[2];
        let f2 = f * f;
        for d in 0..dim {
            let rt0_val = [x, y][d] - verts[1][d];
            phi[dof * dim + d] = f2 * rt0_val;
        }
        div[dof] = 4.0 * f2;
    }
    // Edge 2 (opposite v2): (λ_0 − λ_1)^2 · (x, y−1)
    {
        let dof = 10;
        let f = lam[0] - lam[1];
        let f2 = f * f;
        for d in 0..dim {
            let rt0_val = [x, y][d] - verts[2][d];
            phi[dof * dim + d] = f2 * rt0_val;
        }
        div[dof] = 4.0 * f2;
    }

    // ── RT2 interior bubble functions (DOFs 11–14) ──────────────────────
    // rot90 of N3 face functions; div = −curl(N3 face DOF)
    //
    // DOF 11: rot90(λ_0 λ_1 λ_2 ∇λ_0)
    // N11 = (−λ_0λ_1λ_2, −λ_0λ_1λ_2)
    // rot90 = (λ_0λ_1λ_2, −λ_0λ_1λ_2)
    {
        let dof = 11;
        let b = lam[0] * lam[1] * lam[2];
        phi[dof * dim] = b;
        phi[dof * dim + 1] = -b;
        // div = −curl(N11) = λ_0(λ_2 − λ_1)
        div[dof] = lam[0] * (lam[2] - lam[1]);
    }
    // DOF 12: rot90(λ_0 λ_1 λ_2 ∇λ_1)
    // N12 = (λ_0λ_1λ_2, 0)
    // rot90 = (0, λ_0λ_1λ_2)
    {
        let dof = 12;
        let b = lam[0] * lam[1] * lam[2];
        phi[dof * dim] = 0.0;
        phi[dof * dim + 1] = b;
        // div = −curl(N12) = λ_0 λ_1 − λ_1 λ_2
        div[dof] = lam[0] * lam[1] - lam[1] * lam[2];
    }
    // DOF 13: rot90(λ_0^2 λ_1 ∇λ_2)
    // N13 = (0, λ_0^2 λ_1)
    // rot90 = (−λ_0^2 λ_1, 0)
    {
        let dof = 13;
        let v = lam[0] * lam[0] * lam[1];
        phi[dof * dim] = -v;
        phi[dof * dim + 1] = 0.0;
        // div = −curl(N13) = 2λ_0 λ_1 − λ_0^2
        div[dof] = 2.0 * lam[0] * lam[1] - lam[0] * lam[0];
    }
    // DOF 14: rot90(λ_0 λ_1^2 ∇λ_2)
    // N14 = (0, λ_0 λ_1^2)
    // rot90 = (−λ_0 λ_1^2, 0)
    {
        let dof = 14;
        let v = lam[0] * lam[1] * lam[1];
        phi[dof * dim] = -v;
        phi[dof * dim + 1] = 0.0;
        // div = −curl(N14) = λ_1^2 − 2λ_0 λ_1
        div[dof] = lam[1] * lam[1] - 2.0 * lam[0] * lam[1];
    }

    (phi, div)
}

/// RT0 basis functions on the reference tetrahedron.
///
/// Basis: ψ_i = (x − x_i) / (d · |T|) with d=3, |T|=1/6 → ψ_i = 2(x − x_i).
///
/// Vertices: x₀=(0,0,0), x₁=(1,0,0), x₂=(0,1,0), x₃=(0,0,1).
/// - ψ₀ = 2(x, y, z)
/// - ψ₁ = 2(x−1, y, z)
/// - ψ₂ = 2(x, y−1, z)
/// - ψ₃ = 2(x, y, z−1)
///
/// Divergence is constant: ∇·ψ_i = 1/|T| = 6 for all i.
///
/// Returns `(phi[num_dof * 3], div[num_dof])`.
fn tet_rt0(x: f64, y: f64, z: f64) -> (Vec<f64>, Vec<f64>) {
    let verts: [[f64; 3]; 4] = [
        [0.0, 0.0, 0.0], // x₀
        [1.0, 0.0, 0.0], // x₁
        [0.0, 1.0, 0.0], // x₂
        [0.0, 0.0, 1.0], // x₃
    ];
    let num_dof = 4;
    let dim = 3;
    let factor = 2.0; // 1/(d·|T|) = 1/(3·1/6) = 2
    let div_const = 6.0; // 1/|T| = 6

    let mut phi = vec![0.0f64; num_dof * dim];
    let mut div = vec![0.0f64; num_dof];

    for dof in 0..num_dof {
        for d in 0..dim {
            // ψ_i = 2(x − x_i)
            phi[dof * dim + d] = factor * ([x, y, z][d] - verts[dof][d]);
        }
        div[dof] = div_const;
    }

    (phi, div)
}

/// RT1 basis functions on the reference tetrahedron (20 DOFs).
///
/// Hierarchical construction: RT0 + linear face moments + interior bubbles.
///
/// **RT0 face DOFs** (0–3):
/// ψ_i = 2(x − x_i) for each vertex i; div = 6 (constant).
///
/// **Linear face moments** (4–11):
/// For face opposite vertex i, two linear moments λ_j ψ_i and λ_k ψ_i.
/// - DOF 4: λ_1 ψ_0, DOF 5: λ_2 ψ_0  (face opposite v0)
/// - DOF 6: λ_0 ψ_1, DOF 7: λ_2 ψ_1  (face opposite v1)
/// - DOF 8: λ_0 ψ_2, DOF 9: λ_1 ψ_2  (face opposite v2)
/// - DOF 10: λ_0 ψ_3, DOF 11: λ_1 ψ_3 (face opposite v3)
///
/// **Interior DOFs** (12–19):
/// curl of Nédélec face functions (divergence-free).
/// - DOF 12: curl(λ_1 λ_2 ∇λ_3) = (λ_1, −λ_2, 0)
/// - DOF 13: curl(λ_1 λ_3 ∇λ_2) = (−λ_1, 0, λ_3)
/// - DOF 14: curl(λ_0 λ_2 ∇λ_3) = (λ_0−λ_2, λ_2, 0)
/// - DOF 15: curl(λ_0 λ_3 ∇λ_2) = (λ_3−λ_0, 0, −λ_3)
/// - DOF 16: curl(λ_0 λ_1 ∇λ_3) = (−λ_1, λ_1−λ_0, 0)
/// - DOF 17: curl(λ_0 λ_3 ∇λ_1) = (0, λ_0−λ_3, λ_3)
/// - DOF 18: curl(λ_0 λ_1 ∇λ_2) = (λ_1, 0, λ_0−λ_1)
/// - DOF 19: curl(λ_0 λ_2 ∇λ_1) = (0, −λ_2, λ_2−λ_0)
///
/// Returns `(phi[num_dof * 3], div[num_dof])`.
fn tet_rt1(x: f64, y: f64, z: f64) -> (Vec<f64>, Vec<f64>) {
    let verts: [[f64; 3]; 4] = [
        [0.0, 0.0, 0.0], // x₀
        [1.0, 0.0, 0.0], // x₁
        [0.0, 1.0, 0.0], // x₂
        [0.0, 0.0, 1.0], // x₃
    ];
    let num_dof = 20;
    let dim = 3;
    let factor = 2.0; // 1/(d·|T|) = 1/(3·1/6) = 2
    let div_const = 6.0; // 1/|T| = 6

    let lam = [1.0 - x - y - z, x, y, z]; // λ₀, λ₁, λ₂, λ₃

    let mut phi = vec![0.0f64; num_dof * dim];
    let mut div = vec![0.0f64; num_dof];

    // ── RT0 face functions (DOFs 0–3) ────────────────────────────────────
    // ψ_i = 2(x − x_i); div = 6
    for dof in 0..4 {
        for d in 0..dim {
            phi[dof * dim + d] = factor * ([x, y, z][d] - verts[dof][d]);
        }
        div[dof] = div_const;
    }

    // ── Linear face moments (DOFs 4–11) ──────────────────────────────────
    // div(λ_j ψ_i) = ∇λ_j · ψ_i + λ_j ∇·ψ_i

    // Face opposite v0: use λ_1, λ_2
    // DOF 4: λ_1 ψ_0
    {
        let dof = 4;
        let rt0 = |d: usize| -> f64 { factor * ([x, y, z][d] - verts[0][d]) };
        for d in 0..dim {
            phi[dof * dim + d] = lam[1] * rt0(d);
        }
        // div = ∇λ_1 · ψ_0 + λ_1 · 6 = (1,0,0)·(2x,2y,2z) + 6x = 2x + 6x = 8x
        div[dof] = 8.0 * x;
    }
    // DOF 5: λ_2 ψ_0
    {
        let dof = 5;
        let rt0 = |d: usize| -> f64 { factor * ([x, y, z][d] - verts[0][d]) };
        for d in 0..dim {
            phi[dof * dim + d] = lam[2] * rt0(d);
        }
        // div = ∇λ_2 · ψ_0 + λ_2 · 6 = (0,1,0)·(2x,2y,2z) + 6y = 2y + 6y = 8y
        div[dof] = 8.0 * y;
    }

    // Face opposite v1: use λ_0, λ_2
    // DOF 6: λ_0 ψ_1
    {
        let dof = 6;
        let rt0 = |d: usize| -> f64 { factor * ([x, y, z][d] - verts[1][d]) };
        for d in 0..dim {
            phi[dof * dim + d] = lam[0] * rt0(d);
        }
        // div = ∇λ_0 · ψ_1 + λ_0 · 6 = (-1,-1,-1)·(2x-2,2y,2z) + 6(1-x-y-z)
        //     = -2x+2-2y-2z + 6-6x-6y-6z = 8-8x-8y-8z
        div[dof] = 8.0 * (1.0 - x - y - z);
    }
    // DOF 7: λ_2 ψ_1
    {
        let dof = 7;
        let rt0 = |d: usize| -> f64 { factor * ([x, y, z][d] - verts[1][d]) };
        for d in 0..dim {
            phi[dof * dim + d] = lam[2] * rt0(d);
        }
        // div = ∇λ_2 · ψ_1 + λ_2 · 6 = (0,1,0)·(2x-2,2y,2z) + 6y = 2y + 6y = 8y
        div[dof] = 8.0 * y;
    }

    // Face opposite v2: use λ_0, λ_1
    // DOF 8: λ_0 ψ_2
    {
        let dof = 8;
        let rt0 = |d: usize| -> f64 { factor * ([x, y, z][d] - verts[2][d]) };
        for d in 0..dim {
            phi[dof * dim + d] = lam[0] * rt0(d);
        }
        // div = ∇λ_0 · ψ_2 + λ_0 · 6 = (-1,-1,-1)·(2x,2y-2,2z) + 6(1-x-y-z)
        //     = -2x-2y+2-2z + 6-6x-6y-6z = 8-8x-8y-8z
        div[dof] = 8.0 * (1.0 - x - y - z);
    }
    // DOF 9: λ_1 ψ_2
    {
        let dof = 9;
        let rt0 = |d: usize| -> f64 { factor * ([x, y, z][d] - verts[2][d]) };
        for d in 0..dim {
            phi[dof * dim + d] = lam[1] * rt0(d);
        }
        // div = ∇λ_1 · ψ_2 + λ_1 · 6 = (1,0,0)·(2x,2y-2,2z) + 6x = 2x + 6x = 8x
        div[dof] = 8.0 * x;
    }

    // Face opposite v3: use λ_0, λ_1
    // DOF 10: λ_0 ψ_3
    {
        let dof = 10;
        let rt0 = |d: usize| -> f64 { factor * ([x, y, z][d] - verts[3][d]) };
        for d in 0..dim {
            phi[dof * dim + d] = lam[0] * rt0(d);
        }
        // div = ∇λ_0 · ψ_3 + λ_0 · 6 = (-1,-1,-1)·(2x,2y,2z-2) + 6(1-x-y-z)
        //     = -2x-2y-2z+2 + 6-6x-6y-6z = 8-8x-8y-8z
        div[dof] = 8.0 * (1.0 - x - y - z);
    }
    // DOF 11: λ_1 ψ_3
    {
        let dof = 11;
        let rt0 = |d: usize| -> f64 { factor * ([x, y, z][d] - verts[3][d]) };
        for d in 0..dim {
            phi[dof * dim + d] = lam[1] * rt0(d);
        }
        // div = ∇λ_1 · ψ_3 + λ_1 · 6 = (1,0,0)·(2x,2y,2z-2) + 6x = 2x + 6x = 8x
        div[dof] = 8.0 * x;
    }

    // ── Interior bubble functions (DOFs 12–19) ────────────────────────────
    // curl of Nédélec face functions: curl(λ_a λ_b ∇λ_c) = (λ_a ∇λ_b + λ_b ∇λ_a) × ∇λ_c
    // All have div = 0.

    // DOF 12: curl(λ_1 λ_2 ∇λ_3) = (λ_1, −λ_2, 0)
    {
        let dof = 12;
        phi[dof * dim] = lam[1];
        phi[dof * dim + 1] = -lam[2];
        phi[dof * dim + 2] = 0.0;
        div[dof] = 0.0;
    }
    // DOF 13: curl(λ_1 λ_3 ∇λ_2) = (−λ_1, 0, λ_3)
    {
        let dof = 13;
        phi[dof * dim] = -lam[1];
        phi[dof * dim + 1] = 0.0;
        phi[dof * dim + 2] = lam[3];
        div[dof] = 0.0;
    }
    // DOF 14: curl(λ_0 λ_2 ∇λ_3) = (λ_0−λ_2, λ_2, 0)
    {
        let dof = 14;
        phi[dof * dim] = lam[0] - lam[2];
        phi[dof * dim + 1] = lam[2];
        phi[dof * dim + 2] = 0.0;
        div[dof] = 0.0;
    }
    // DOF 15: curl(λ_0 λ_3 ∇λ_2) = (λ_3−λ_0, 0, −λ_3)
    {
        let dof = 15;
        phi[dof * dim] = lam[3] - lam[0];
        phi[dof * dim + 1] = 0.0;
        phi[dof * dim + 2] = -lam[3];
        div[dof] = 0.0;
    }
    // DOF 16: curl(λ_0 λ_1 ∇λ_3) = (−λ_1, λ_1−λ_0, 0)
    {
        let dof = 16;
        phi[dof * dim] = -lam[1];
        phi[dof * dim + 1] = lam[1] - lam[0];
        phi[dof * dim + 2] = 0.0;
        div[dof] = 0.0;
    }
    // DOF 17: curl(λ_0 λ_3 ∇λ_1) = (0, λ_0−λ_3, λ_3)
    {
        let dof = 17;
        phi[dof * dim] = 0.0;
        phi[dof * dim + 1] = lam[0] - lam[3];
        phi[dof * dim + 2] = lam[3];
        div[dof] = 0.0;
    }
    // DOF 18: curl(λ_0 λ_1 ∇λ_2) = (λ_1, 0, λ_0−λ_1)
    {
        let dof = 18;
        phi[dof * dim] = lam[1];
        phi[dof * dim + 1] = 0.0;
        phi[dof * dim + 2] = lam[0] - lam[1];
        div[dof] = 0.0;
    }
    // DOF 19: curl(λ_0 λ_2 ∇λ_1) = (0, −λ_2, λ_2−λ_0)
    {
        let dof = 19;
        phi[dof * dim] = 0.0;
        phi[dof * dim + 1] = -lam[2];
        phi[dof * dim + 2] = lam[2] - lam[0];
        div[dof] = 0.0;
    }

    (phi, div)
}

// ── utilities ─────────────────────────────────────────────────────────────────

fn to_t<T: Scalar>(v: f64) -> ReedResult<T> {
    T::from(v).ok_or_else(|| {
        ReedError::Basis(format!(
            "RaviartThomasBasis: failed to convert {v} to scalar"
        ))
    })
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
            "RaviartThomasBasis {mode} size mismatch: \
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

    // ── shape function tests ───────────────────────────────────────────────

    #[test]
    fn tri_rt0_div_is_constant_2() {
        // At any point, all divergence values should be 2.
        for &(x, y) in &[(0.1, 0.2), (0.5, 0.3), (1.0 / 3.0, 1.0 / 3.0)] {
            let (_phi, div) = tri_rt0(x, y);
            for dof in 0..3 {
                assert!(
                    (div[dof] - 2.0).abs() < TOL,
                    "tri div[dof={}] = {} at ({},{})",
                    dof,
                    div[dof],
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn tet_rt0_div_is_constant_6() {
        // At any point, all divergence values should be 6.
        for &(x, y, z) in &[
            (0.1, 0.2, 0.3),
            (0.5, 0.1, 0.1),
            (0.25, 0.25, 0.25),
        ] {
            let (_phi, div) = tet_rt0(x, y, z);
            for dof in 0..4 {
                assert!(
                    (div[dof] - 6.0).abs() < TOL,
                    "tet div[dof={}] = {} at ({},{},{})",
                    dof,
                    div[dof],
                    x,
                    y,
                    z
                );
            }
        }
    }

    #[test]
    fn tri_rt0_basis_values() {
        // Verify specific basis values at the centroid (1/3, 1/3).
        let x = 1.0 / 3.0;
        let y = 1.0 / 3.0;
        let (phi, _div) = tri_rt0(x, y);
        // ψ₀ = (x, y) = (1/3, 1/3)
        assert!((phi[0] - 1.0 / 3.0).abs() < TOL);
        assert!((phi[1] - 1.0 / 3.0).abs() < TOL);
        // ψ₁ = (x-1, y) = (-2/3, 1/3)
        assert!((phi[2] - (-2.0 / 3.0)).abs() < TOL);
        assert!((phi[3] - 1.0 / 3.0).abs() < TOL);
        // ψ₂ = (x, y-1) = (1/3, -2/3)
        assert!((phi[4] - 1.0 / 3.0).abs() < TOL);
        assert!((phi[5] - (-2.0 / 3.0)).abs() < TOL);
    }

    #[test]
    fn tet_rt0_basis_values() {
        // Verify specific basis values at the centroid (1/4, 1/4, 1/4).
        let x = 0.25;
        let y = 0.25;
        let z = 0.25;
        let (phi, _div) = tet_rt0(x, y, z);
        // ψ₀ = 2(x, y, z) = (0.5, 0.5, 0.5)
        assert!((phi[0] - 0.5).abs() < TOL);
        assert!((phi[1] - 0.5).abs() < TOL);
        assert!((phi[2] - 0.5).abs() < TOL);
        // ψ₁ = 2(x-1, y, z) = (-1.5, 0.5, 0.5)
        assert!((phi[3] - (-1.5)).abs() < TOL);
        assert!((phi[4] - 0.5).abs() < TOL);
        assert!((phi[5] - 0.5).abs() < TOL);
        // ψ₂ = 2(x, y-1, z) = (0.5, -1.5, 0.5)
        assert!((phi[6] - 0.5).abs() < TOL);
        assert!((phi[7] - (-1.5)).abs() < TOL);
        assert!((phi[8] - 0.5).abs() < TOL);
        // ψ₃ = 2(x, y, z-1) = (0.5, 0.5, -1.5)
        assert!((phi[9] - 0.5).abs() < TOL);
        assert!((phi[10] - 0.5).abs() < TOL);
        assert!((phi[11] - (-1.5)).abs() < TOL);
    }

    // ── basis construction tests ───────────────────────────────────────────

    #[test]
    fn construct_tri_rt0() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 0, 3).unwrap();
        assert_eq!(basis.dim(), 2);
        assert_eq!(basis.num_dof(), 3);
        assert_eq!(basis.num_qpoints(), 3);
        assert_eq!(basis.num_comp(), 2);
        assert_eq!(basis.q_weights().len(), 3);
        assert_eq!(basis.q_ref().len(), 6); // 3 qpts × 2
    }

    #[test]
    fn construct_tet_rt0() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 0, 4).unwrap();
        assert_eq!(basis.dim(), 3);
        assert_eq!(basis.num_dof(), 4);
        assert_eq!(basis.num_qpoints(), 4);
        assert_eq!(basis.num_comp(), 3);
        assert_eq!(basis.q_weights().len(), 4);
        assert_eq!(basis.q_ref().len(), 12); // 4 qpts × 3
    }

    #[test]
    fn reject_invalid_p() {
        // Triangle: p=0, 1, 2 are OK; higher p rejected
        assert!(RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 0, 3).is_ok());
        assert!(RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 1, 3).is_ok());
        assert!(RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 2, 3).is_ok());
        assert!(RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 3, 3).is_err());
        // Tet: p=0 (RT0) and p=1 (RT1) are OK; higher p rejected
        assert!(RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 0, 4).is_ok());
        assert!(RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 1, 4).is_ok());
        assert!(RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 2, 4).is_err());
    }

    #[test]
    fn reject_unsupported_topo() {
        assert!(RaviartThomasBasis::<f64>::new(ElemTopology::Line, 0, 2).is_err());
    }

    // ── apply: Interp mode ─────────────────────────────────────────────────

    #[test]
    fn tri_rt0_interp_forward_size() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 0, 3).unwrap();
        let nelem = 2;
        let ndof = 3;
        let nqpts = 3;
        let dim = 2;
        let u = vec![0.0f64; nelem * ndof * dim];
        let mut v = vec![0.0f64; nelem * nqpts * dim];
        basis
            .apply(nelem, false, EvalMode::Interp, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tri_rt0_interp_transpose_size() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 0, 3).unwrap();
        let nelem = 2;
        let ndof = 3;
        let nqpts = 3;
        let dim = 2;
        let u = vec![0.0f64; nelem * nqpts * dim];
        let mut v = vec![0.0f64; nelem * ndof * dim];
        basis
            .apply(nelem, true, EvalMode::Interp, &u, &mut v)
            .unwrap();
    }

    // ── apply: HDiv mode ──────────────────────────────────────────────────

    #[test]
    fn tri_rt0_hdiv_forward_size() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 0, 3).unwrap();
        let nelem = 2;
        let ndof = 3;
        let nqpts = 3;
        let dim = 2;
        let u = vec![0.0f64; nelem * ndof * dim];
        let mut v = vec![0.0f64; nelem * nqpts]; // scalar divergence
        basis
            .apply(nelem, false, EvalMode::HDiv, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tri_rt0_hdiv_transpose_size() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 0, 3).unwrap();
        let nelem = 2;
        let ndof = 3;
        let nqpts = 3;
        let dim = 2;
        let u = vec![0.0f64; nelem * nqpts]; // scalar divergence
        let mut v = vec![0.0f64; nelem * ndof * dim];
        basis
            .apply(nelem, true, EvalMode::HDiv, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tet_rt0_hdiv_forward_size() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 0, 4).unwrap();
        let nelem = 2;
        let ndof = 4;
        let nqpts = 4;
        let dim = 3;
        let u = vec![0.0f64; nelem * ndof * dim];
        let mut v = vec![0.0f64; nelem * nqpts]; // scalar divergence in 3D too
        basis
            .apply(nelem, false, EvalMode::HDiv, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tet_rt0_hdiv_transpose_size() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 0, 4).unwrap();
        let nelem = 2;
        let ndof = 4;
        let nqpts = 4;
        let dim = 3;
        let u = vec![0.0f64; nelem * nqpts]; // scalar divergence
        let mut v = vec![0.0f64; nelem * ndof * dim];
        basis
            .apply(nelem, true, EvalMode::HDiv, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tri_rt0_rejects_hcurl_mode() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 0, 3).unwrap();
        let u = vec![0.0f64; 3 * 2];
        let mut v = vec![0.0f64; 3 * 2];
        assert!(basis.apply(1, false, EvalMode::HCurl, &u, &mut v).is_err());
    }

    #[test]
    fn tri_rt0_rejects_grad_mode() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 0, 3).unwrap();
        let u = vec![0.0f64; 3 * 2];
        let mut v = vec![0.0f64; 3 * 2];
        assert!(basis.apply(1, false, EvalMode::Grad, &u, &mut v).is_err());
    }

    // ── apply: Weight mode ─────────────────────────────────────────────────

    #[test]
    fn tri_rt0_weight_mode() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 0, 3).unwrap();
        let nelem = 2;
        let mut v = vec![0.0f64; nelem * 3];
        basis
            .apply(nelem, false, EvalMode::Weight, &[], &mut v)
            .unwrap();
        // each element gets the same quadrature weights
        assert!((v[0] - v[3]).abs() < TOL);
    }

    // ── divergence correctness tests ───────────────────────────────────────

    #[test]
    fn tri_rt0_hdiv_forward_gives_div2() {
        // With all DOFs set to 1.0, divergence at each qpt should be sum(div_i) = 3*2 = 6.
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 0, 3).unwrap();
        let nelem = 1;
        let ndof = 3;
        let nqpts = 3;
        let dim = 2;
        let mut u = vec![0.0f64; nelem * ndof * dim];
        for dof in 0..ndof {
            u[dof * dim] = 1.0;
        }
        let mut v = vec![0.0f64; nelem * nqpts];
        basis
            .apply(nelem, false, EvalMode::HDiv, &u, &mut v)
            .unwrap();
        for qpt in 0..nqpts {
            assert!((v[qpt] - 6.0).abs() < TOL, "qpt {}: div = {}", qpt, v[qpt]);
        }
    }

    #[test]
    fn tet_rt0_hdiv_forward_gives_div6() {
        // With all DOFs set to 1.0, divergence at each qpt should be sum(div_i) = 4*6 = 24.
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 0, 4).unwrap();
        let nelem = 1;
        let ndof = 4;
        let nqpts = 4;
        let dim = 3;
        let mut u = vec![0.0f64; nelem * ndof * dim];
        for dof in 0..ndof {
            u[dof * dim] = 1.0;
        }
        let mut v = vec![0.0f64; nelem * nqpts];
        basis
            .apply(nelem, false, EvalMode::HDiv, &u, &mut v)
            .unwrap();
        for qpt in 0..nqpts {
            assert!(
                (v[qpt] - 24.0).abs() < TOL,
                "qpt {}: div = {}",
                qpt,
                v[qpt]
            );
        }
    }

    // ── transpose consistency: forward+transpose = nonzero projection ──────

    #[test]
    fn tri_rt0_interp_transpose_consistency() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 0, 3).unwrap();
        let nelem = 1;
        let ndof = 3;
        let nqpts = 3;
        let dim = 2;

        let mut u_dof = vec![0.0f64; nelem * ndof * dim];
        for dof in 0..ndof {
            u_dof[dof * dim] = (dof + 1) as f64;
        }

        let mut v_qpts = vec![0.0f64; nelem * nqpts * dim];
        basis
            .apply(nelem, false, EvalMode::Interp, &u_dof, &mut v_qpts)
            .unwrap();

        let mut u_dof_back = vec![0.0f64; nelem * ndof * dim];
        basis
            .apply(nelem, true, EvalMode::Interp, &v_qpts, &mut u_dof_back)
            .unwrap();

        for dof in 0..ndof {
            let val = u_dof_back[dof * dim];
            assert!(
                val.abs() > TOL,
                "transpose consistency: dof {dof} is zero"
            );
        }
    }

    #[test]
    fn tet_rt0_interp_transpose_consistency() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 0, 4).unwrap();
        let nelem = 1;
        let ndof = 4;
        let nqpts = 4;
        let dim = 3;

        let mut u_dof = vec![0.0f64; nelem * ndof * dim];
        for dof in 0..ndof {
            u_dof[dof * dim] = (dof + 1) as f64;
        }

        let mut v_qpts = vec![0.0f64; nelem * nqpts * dim];
        basis
            .apply(nelem, false, EvalMode::Interp, &u_dof, &mut v_qpts)
            .unwrap();

        let mut u_dof_back = vec![0.0f64; nelem * ndof * dim];
        basis
            .apply(nelem, true, EvalMode::Interp, &v_qpts, &mut u_dof_back)
            .unwrap();

        for dof in 0..ndof {
            let val = u_dof_back[dof * dim];
            assert!(val.abs() > TOL, "transpose consistency: dof {dof} is zero");
        }
    }

    #[test]
    fn tri_rt0_hdiv_transpose_consistency() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 0, 3).unwrap();
        let nelem = 1;
        let ndof = 3;
        let nqpts = 3;
        let dim = 2;

        let mut u_dof = vec![0.0f64; nelem * ndof * dim];
        for dof in 0..ndof {
            u_dof[dof * dim] = (dof + 1) as f64;
        }

        // forward HDiv
        let mut v_div = vec![0.0f64; nelem * nqpts];
        basis
            .apply(nelem, false, EvalMode::HDiv, &u_dof, &mut v_div)
            .unwrap();

        // transpose HDiv
        let mut u_dof_back = vec![0.0f64; nelem * ndof * dim];
        basis
            .apply(nelem, true, EvalMode::HDiv, &v_div, &mut u_dof_back)
            .unwrap();

        for dof in 0..ndof {
            let val = u_dof_back[dof * dim];
            assert!(
                val.abs() > TOL,
                "hdiv transpose consistency: dof {dof} is zero"
            );
        }
    }

    // ── RT1 triangle tests ────────────────────────────────────────────────

    #[test]
    fn construct_tri_rt1() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 1, 6).unwrap();
        assert_eq!(basis.dim(), 2);
        assert_eq!(basis.num_dof(), 8);
        assert_eq!(basis.num_qpoints(), 6);
        assert_eq!(basis.num_comp(), 2);
    }

    #[test]
    fn tri_rt1_rt0_subspace() {
        // The RT0 basis should be exactly the first 3 DOFs of the RT1 basis.
        let basis_rt0 = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 0, 6).unwrap();
        let basis_rt1 = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 1, 6).unwrap();

        assert_eq!(basis_rt0.num_qpoints(), basis_rt1.num_qpoints());

        for qpt in 0..basis_rt0.num_qpoints() {
            for dof in 0..3 {
                for d in 0..2 {
                    let v0 = basis_rt0.interp[(qpt * 3 + dof) * 2 + d];
                    let v1 = basis_rt1.interp[(qpt * 8 + dof) * 2 + d];
                    assert!(
                        (v0 - v1).abs() < TOL,
                        "RT0/RT1 mismatch at qpt={qpt} dof={dof} d={d}: {v0} vs {v1}"
                    );
                }
            }
            for dof in 0..3 {
                let d0 = basis_rt0.div_matrix[qpt * 3 + dof];
                let d1 = basis_rt1.div_matrix[qpt * 8 + dof];
                assert!(
                    (d0 - d1).abs() < TOL,
                    "RT0/RT1 div mismatch at qpt={qpt} dof={dof}: {d0} vs {d1}"
                );
            }
        }
    }

    #[test]
    fn tri_rt1_interior_zero_on_boundary() {
        // Interior bubble functions (DOFs 6 and 7) should vanish at all vertices.
        for &(x, y) in &[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)] {
            let (phi, _div) = tri_rt1(x, y);
            for dof in 6..8 {
                let fx = phi[dof * 2];
                let fy = phi[dof * 2 + 1];
                assert!(
                    fx.abs() < TOL && fy.abs() < TOL,
                    "Interior DOF {dof} non-zero at vertex ({x},{y}): ({fx},{fy})"
                );
            }
        }
    }

    #[test]
    fn tri_rt1_div_varies() {
        // RT1 divergences should not be constant (unlike RT0 where div=2 for all).
        let (_phi1, div1) = tri_rt1(0.1, 0.2);
        let (_phi2, div2) = tri_rt1(0.7, 0.1);

        // RT0 DOFs 0-2 should still have constant div=2
        for dof in 0..3 {
            assert!(
                (div1[dof] - div2[dof]).abs() < TOL && (div1[dof] - 2.0).abs() < TOL,
                "RT0 DOF {dof}: div should be constant 2, got {} and {}",
                div1[dof],
                div2[dof]
            );
        }

        // RT1 DOFs 3-7 should have spatially varying divergence
        let mut any_varied = false;
        for dof in 3..8 {
            if (div1[dof] - div2[dof]).abs() > TOL {
                any_varied = true;
                break;
            }
        }
        assert!(any_varied, "RT1 higher-order DOF divergences should vary with position");

        // Directly verify analytic divergence formulas at a test point
        let (_phi, div) = tri_rt1(0.3, 0.4);

        // DOF 3 (edge 0 linear): div = 3(x-y) = 3*(-0.1) = -0.3
        assert!((div[3] - 3.0 * (0.3 - 0.4)).abs() < TOL);

        // DOF 4 (edge 1 linear): div = 3(x + 2y - 1) = 3*(0.3 + 0.8 - 1) = 3*0.1 = 0.3
        assert!((div[4] - 3.0 * (0.3 + 0.8 - 1.0)).abs() < TOL);

        // DOF 5 (edge 2 linear): div = 3(1 - 2x - y) = 3*(1 - 0.6 - 0.4) = 0
        assert!((div[5] - 3.0 * (1.0 - 0.6 - 0.4)).abs() < TOL);

        // DOF 6 (interior 1): div = 2x + y - 1 = 0.6 + 0.4 - 1 = 0
        assert!((div[6] - (2.0 * 0.3 + 0.4 - 1.0)).abs() < TOL);

        // DOF 7 (interior 2): div = 1 - x - 2y = 1 - 0.3 - 0.8 = -0.1
        assert!((div[7] - (1.0 - 0.3 - 0.8)).abs() < TOL);
    }

    #[test]
    fn tri_rt1_interp_forward_size() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 1, 6).unwrap();
        let nelem = 2;
        let ndof = 8;
        let nqpts = 6;
        let dim = 2;
        let u = vec![0.0f64; nelem * ndof * dim];
        let mut v = vec![0.0f64; nelem * nqpts * dim];
        basis
            .apply(nelem, false, EvalMode::Interp, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tri_rt1_hdiv_forward_size() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 1, 6).unwrap();
        let nelem = 2;
        let ndof = 8;
        let nqpts = 6;
        let dim = 2;
        let u = vec![0.0f64; nelem * ndof * dim];
        let mut v = vec![0.0f64; nelem * nqpts];
        basis
            .apply(nelem, false, EvalMode::HDiv, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tri_rt1_interp_transpose_consistency() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 1, 6).unwrap();
        let nelem = 1;
        let ndof = 8;
        let nqpts = 6;
        let dim = 2;

        let mut u_dof = vec![0.0f64; nelem * ndof * dim];
        for dof in 0..ndof {
            u_dof[dof * dim] = (dof + 1) as f64;
        }

        let mut v_qpts = vec![0.0f64; nelem * nqpts * dim];
        basis
            .apply(nelem, false, EvalMode::Interp, &u_dof, &mut v_qpts)
            .unwrap();

        let mut u_dof_back = vec![0.0f64; nelem * ndof * dim];
        basis
            .apply(nelem, true, EvalMode::Interp, &v_qpts, &mut u_dof_back)
            .unwrap();

        for dof in 0..ndof {
            let val = u_dof_back[dof * dim];
            assert!(
                val.abs() > TOL,
                "RT1 transpose consistency: dof {dof} is zero"
            );
        }
    }

    // ── RT1 tet tests ────────────────────────────────────────────────────

    #[test]
    fn construct_tet_rt1() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 1, 4).unwrap();
        assert_eq!(basis.dim(), 3);
        assert_eq!(basis.num_dof(), 20);
        assert_eq!(basis.num_qpoints(), 4);
        assert_eq!(basis.num_comp(), 3);
    }

    #[test]
    fn tet_rt1_rt0_subspace() {
        // The RT0 basis should be exactly the first 4 DOFs of the RT1 basis.
        let basis_rt0 = RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 0, 4).unwrap();
        let basis_rt1 = RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 1, 4).unwrap();

        assert_eq!(basis_rt0.num_qpoints(), basis_rt1.num_qpoints());

        for qpt in 0..basis_rt0.num_qpoints() {
            for dof in 0..4 {
                for d in 0..3 {
                    let v0 = basis_rt0.interp[(qpt * 4 + dof) * 3 + d];
                    let v1 = basis_rt1.interp[(qpt * 20 + dof) * 3 + d];
                    assert!(
                        (v0 - v1).abs() < TOL,
                        "RT0/RT1 mismatch at qpt={qpt} dof={dof} d={d}: {v0} vs {v1}"
                    );
                }
            }
            for dof in 0..4 {
                let d0 = basis_rt0.div_matrix[qpt * 4 + dof];
                let d1 = basis_rt1.div_matrix[qpt * 20 + dof];
                assert!(
                    (d0 - d1).abs() < TOL,
                    "RT0/RT1 div mismatch at qpt={qpt} dof={dof}: {d0} vs {d1}"
                );
            }
        }
    }

    #[test]
    fn tet_rt1_interior_divergence_free() {
        // Interior bubble functions (DOFs 12–19) should have zero divergence
        // at any point since they are curls of Nédélec face functions.
        for &(x, y, z) in &[
            (0.1, 0.2, 0.3),
            (0.25, 0.25, 0.25),
            (0.5, 0.1, 0.1),
        ] {
            let (_phi, div) = tet_rt1(x, y, z);
            for dof in 12..20 {
                assert!(
                    div[dof].abs() < TOL,
                    "Interior DOF {dof} div={} at ({x},{y},{z}), expected 0",
                    div[dof]
                );
            }
        }
    }

    #[test]
    fn tet_rt1_interior_from_nedelec_curl() {
        // The RT1 interior functions are curl of Nedelec face functions.
        // Verify this relationship analytically: for DOF 12 =
        // curl(λ_1 λ_2 ∇λ_3), check that the vector matches the formula.
        let (phi, _div) = tet_rt1(0.3, 0.2, 0.1);
        // DOF 12: (λ_1, −λ_2, 0) = (0.3, -0.2, 0)
        assert!((phi[12 * 3] - 0.3).abs() < TOL);
        assert!((phi[12 * 3 + 1] - (-0.2)).abs() < TOL);
        assert!((phi[12 * 3 + 2] - 0.0).abs() < TOL);
        // DOF 13: (−λ_1, 0, λ_3) = (-0.3, 0, 0.1)
        assert!((phi[13 * 3] - (-0.3)).abs() < TOL);
        assert!((phi[13 * 3 + 1] - 0.0).abs() < TOL);
        assert!((phi[13 * 3 + 2] - 0.1).abs() < TOL);
        // DOF 18: (λ_1, 0, λ_0−λ_1) = (0.3, 0, 0.4-0.3) = (0.3, 0, 0.1)
        assert!((phi[18 * 3] - 0.3).abs() < TOL);
        assert!((phi[18 * 3 + 1] - 0.0).abs() < TOL);
        assert!((phi[18 * 3 + 2] - 0.1).abs() < TOL);
    }

    #[test]
    fn tet_rt1_div_varies() {
        // RT0 DOFs 0-3 should have constant div=6.
        // RT1 DOFs 4-11 should have spatially varying divergence.
        // Interior DOFs 12-19 should have div=0.
        let (_phi1, div1) = tet_rt1(0.1, 0.2, 0.3);
        let (_phi2, div2) = tet_rt1(0.4, 0.1, 0.2);

        for dof in 0..4 {
            assert!(
                (div1[dof] - 6.0).abs() < TOL && (div2[dof] - 6.0).abs() < TOL,
                "RT0 DOF {dof}: div should be constant 6, got {} and {}",
                div1[dof],
                div2[dof]
            );
        }

        let mut any_varied = false;
        for dof in 4..12 {
            if (div1[dof] - div2[dof]).abs() > TOL {
                any_varied = true;
                break;
            }
        }
        assert!(any_varied, "Face moment divergences should vary with position");

        // Verify analytic divergence for linear moments at a test point
        let (_phi, div) = tet_rt1(0.3, 0.2, 0.1);

        // DOF 4: λ_1 ψ_0 → div = 8x = 2.4
        assert!((div[4] - 8.0 * 0.3).abs() < TOL);
        // DOF 5: λ_2 ψ_0 → div = 8y = 1.6
        assert!((div[5] - 8.0 * 0.2).abs() < TOL);
        // DOF 6: λ_0 ψ_1 → div = 8(1-x-y-z) = 8*0.4 = 3.2
        assert!((div[6] - 8.0 * (1.0 - 0.3 - 0.2 - 0.1)).abs() < TOL);
        // DOF 7: λ_2 ψ_1 → div = 8y = 1.6
        assert!((div[7] - 8.0 * 0.2).abs() < TOL);
        // DOF 8: λ_0 ψ_2 → div = 8(1-x-y-z) = 3.2
        assert!((div[8] - 8.0 * (1.0 - 0.3 - 0.2 - 0.1)).abs() < TOL);
        // DOF 9: λ_1 ψ_2 → div = 8x = 2.4
        assert!((div[9] - 8.0 * 0.3).abs() < TOL);
        // DOF 10: λ_0 ψ_3 → div = 8(1-x-y-z) = 3.2
        assert!((div[10] - 8.0 * (1.0 - 0.3 - 0.2 - 0.1)).abs() < TOL);
        // DOF 11: λ_1 ψ_3 → div = 8x = 2.4
        assert!((div[11] - 8.0 * 0.3).abs() < TOL);
    }

    #[test]
    fn tet_rt1_hdiv_forward_rt0_subspace_constant() {
        // With all RT0 DOFs set to 1.0 and all higher DOFs set to 0.0,
        // divergence should be 4*6 = 24 at each qpt.
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 1, 4).unwrap();
        let nelem = 1;
        let ndof = 20;
        let nqpts = 4;
        let dim = 3;
        let mut u = vec![0.0f64; nelem * ndof * dim];
        for dof in 0..4 {
            u[dof * dim] = 1.0;
        }
        let mut v = vec![0.0f64; nelem * nqpts];
        basis
            .apply(nelem, false, EvalMode::HDiv, &u, &mut v)
            .unwrap();
        for qpt in 0..nqpts {
            assert!(
                (v[qpt] - 24.0).abs() < TOL,
                "qpt {}: div = {}",
                qpt,
                v[qpt]
            );
        }
    }

    #[test]
    fn tet_rt1_interp_forward_size() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 1, 4).unwrap();
        let nelem = 2;
        let ndof = 20;
        let nqpts = 4;
        let dim = 3;
        let u = vec![0.0f64; nelem * ndof * dim];
        let mut v = vec![0.0f64; nelem * nqpts * dim];
        basis
            .apply(nelem, false, EvalMode::Interp, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tet_rt1_hdiv_forward_size() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 1, 4).unwrap();
        let nelem = 2;
        let ndof = 20;
        let nqpts = 4;
        let dim = 3;
        let u = vec![0.0f64; nelem * ndof * dim];
        let mut v = vec![0.0f64; nelem * nqpts];
        basis
            .apply(nelem, false, EvalMode::HDiv, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tet_rt1_interp_transpose_consistency() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Tet, 1, 4).unwrap();
        let nelem = 1;
        let ndof = 20;
        let nqpts = 4;
        let dim = 3;

        let mut u_dof = vec![0.0f64; nelem * ndof * dim];
        for dof in 0..ndof {
            u_dof[dof * dim] = (dof + 1) as f64;
        }

        let mut v_qpts = vec![0.0f64; nelem * nqpts * dim];
        basis
            .apply(nelem, false, EvalMode::Interp, &u_dof, &mut v_qpts)
            .unwrap();

        let mut u_dof_back = vec![0.0f64; nelem * ndof * dim];
        basis
            .apply(nelem, true, EvalMode::Interp, &v_qpts, &mut u_dof_back)
            .unwrap();

        for dof in 0..ndof {
            let val = u_dof_back[dof * dim];
            assert!(
                val.abs() > TOL,
                "Tet RT1 transpose consistency: dof {dof} is zero"
            );
        }
    }

    // ── RT2 triangle tests ────────────────────────────────────────────────

    #[test]
    fn construct_tri_rt2() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 2, 6).unwrap();
        assert_eq!(basis.dim(), 2);
        assert_eq!(basis.num_dof(), 15);
        assert_eq!(basis.num_qpoints(), 6);
        assert_eq!(basis.num_comp(), 2);
    }

    #[test]
    fn tri_rt2_rt1_subspace() {
        // The RT1 basis should be exactly the first 8 DOFs of the RT2 basis.
        let basis_rt1 = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 1, 6).unwrap();
        let basis_rt2 = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 2, 6).unwrap();

        assert_eq!(basis_rt1.num_qpoints(), basis_rt2.num_qpoints());

        for qpt in 0..basis_rt1.num_qpoints() {
            for dof in 0..8 {
                for d in 0..2 {
                    let v1 = basis_rt1.interp[(qpt * 8 + dof) * 2 + d];
                    let v2 = basis_rt2.interp[(qpt * 15 + dof) * 2 + d];
                    assert!(
                        (v1 - v2).abs() < TOL,
                        "RT1/RT2 mismatch at qpt={qpt} dof={dof} d={d}: {v1} vs {v2}"
                    );
                }
            }
            for dof in 0..8 {
                let d1 = basis_rt1.div_matrix[qpt * 8 + dof];
                let d2 = basis_rt2.div_matrix[qpt * 15 + dof];
                assert!(
                    (d1 - d2).abs() < TOL,
                    "RT1/RT2 div mismatch at qpt={qpt} dof={dof}: {d1} vs {d2}"
                );
            }
        }
    }

    #[test]
    fn tri_rt2_interior_zero_at_vertices() {
        // RT2 interior bubble functions (DOFs 11–14) should vanish at all vertices.
        for &(x, y) in &[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)] {
            let (phi, _div) = tri_rt2(x, y);
            for dof in 11..15 {
                let fx = phi[dof * 2];
                let fy = phi[dof * 2 + 1];
                assert!(
                    fx.abs() < TOL && fy.abs() < TOL,
                    "RT2 interior DOF {dof} non-zero at vertex ({x},{y}): ({fx},{fy})"
                );
            }
        }
    }

    #[test]
    fn tri_rt2_edge_quadratic_zero_at_midpoint() {
        // RT2 quadratic edge moments (DOFs 8–10) should vanish at edge midpoints
        // where λ_i = λ_j (since (λ_i − λ_j)^2 = 0).
        // Edge mapping:
        //   DOF 8 = edge opposite v0 (connects v1,v2): midpoint (0.5, 0.5)
        //   DOF 9 = edge opposite v1 (connects v0,v2): midpoint (0, 0.5)
        //   DOF 10 = edge opposite v2 (connects v0,v1): midpoint (0.5, 0)
        for &(x, y, dof_bubble) in &[
            (0.5, 0.5, 8),  // edge 0 midpoint
            (0.0, 0.5, 9),  // edge 1 midpoint
            (0.5, 0.0, 10), // edge 2 midpoint
        ] {
            let (phi, _div) = tri_rt2(x, y);
            let fx = phi[dof_bubble * 2];
            let fy = phi[dof_bubble * 2 + 1];
            assert!(
                fx.abs() < TOL && fy.abs() < TOL,
                "RT2 edge quadratic DOF {dof_bubble} non-zero at midpoint ({x},{y}): ({fx},{fy})"
            );
        }
    }

    #[test]
    fn tri_rt2_div_formulas() {
        // Verify analytic divergence formulas at a test point.
        let x = 0.3;
        let y = 0.4;
        let lam0 = 1.0 - x - y; // 0.3
        let lam1 = x; // 0.3
        let lam2 = y; // 0.4

        let (_phi, div) = tri_rt2(x, y);

        // RT0 DOFs 0-2: div = 2
        for dof in 0..3 {
            assert!((div[dof] - 2.0).abs() < TOL,
                "RT0 DOF {dof}: div should be 2, got {}", div[dof]);
        }

        // DOF 3 (edge 0 linear): div = 3(λ_1 − λ_2) = 3*(-0.1) = -0.3
        assert!((div[3] - 3.0 * (lam1 - lam2)).abs() < TOL);
        // DOF 4 (edge 1 linear): div = 3(λ_2 − λ_0) = 3*(0.1) = 0.3
        assert!((div[4] - 3.0 * (lam2 - lam0)).abs() < TOL);
        // DOF 5 (edge 2 linear): div = 3(λ_0 − λ_1) = 3*(0) = 0
        assert!((div[5] - 3.0 * (lam0 - lam1)).abs() < TOL);

        // DOF 6 (interior 1): div = 2x + y − 1 = 0.6 + 0.4 - 1 = 0
        assert!((div[6] - (2.0 * x + y - 1.0)).abs() < TOL);
        // DOF 7 (interior 2): div = 1 − x − 2y = 1 - 0.3 - 0.8 = -0.1
        assert!((div[7] - (1.0 - x - 2.0 * y)).abs() < TOL);

        // DOF 8 (edge 0 quadratic): div = 4(λ_1−λ_2)^2 = 4*0.01 = 0.04
        assert!((div[8] - 4.0 * (lam1 - lam2) * (lam1 - lam2)).abs() < TOL);
        // DOF 9 (edge 1 quadratic): div = 4(λ_0−λ_2)^2 = 4*0.01 = 0.04
        assert!((div[9] - 4.0 * (lam0 - lam2) * (lam0 - lam2)).abs() < TOL);
        // DOF 10 (edge 2 quadratic): div = 4(λ_0−λ_1)^2 = 4*0 = 0
        assert!((div[10] - 4.0 * (lam0 - lam1) * (lam0 - lam1)).abs() < TOL);

        // DOF 11: div = λ_0(λ_2 − λ_1) = 0.3*(0.4-0.3) = 0.03
        assert!((div[11] - lam0 * (lam2 - lam1)).abs() < TOL);
        // DOF 12: div = λ_0 λ_1 − λ_1 λ_2 = 0.09 - 0.12 = -0.03
        assert!((div[12] - (lam0 * lam1 - lam1 * lam2)).abs() < TOL);
        // DOF 13: div = 2λ_0 λ_1 − λ_0^2 = 2*0.09 - 0.09 = 0.09
        assert!((div[13] - (2.0 * lam0 * lam1 - lam0 * lam0)).abs() < TOL);
        // DOF 14: div = λ_1^2 − 2λ_0 λ_1 = 0.09 - 0.18 = -0.09
        assert!((div[14] - (lam1 * lam1 - 2.0 * lam0 * lam1)).abs() < TOL);
    }

    #[test]
    fn tri_rt2_interp_forward_size() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 2, 6).unwrap();
        let nelem = 2;
        let ndof = 15;
        let nqpts = 6;
        let dim = 2;
        let u = vec![0.0f64; nelem * ndof * dim];
        let mut v = vec![0.0f64; nelem * nqpts * dim];
        basis
            .apply(nelem, false, EvalMode::Interp, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tri_rt2_hdiv_forward_size() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 2, 6).unwrap();
        let nelem = 2;
        let ndof = 15;
        let nqpts = 6;
        let dim = 2;
        let u = vec![0.0f64; nelem * ndof * dim];
        let mut v = vec![0.0f64; nelem * nqpts];
        basis
            .apply(nelem, false, EvalMode::HDiv, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tri_rt2_interp_transpose_consistency() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 2, 6).unwrap();
        let nelem = 1;
        let ndof = 15;
        let nqpts = 6;
        let dim = 2;

        let mut u_dof = vec![0.0f64; nelem * ndof * dim];
        for dof in 0..ndof {
            u_dof[dof * dim] = (dof + 1) as f64;
        }

        let mut v_qpts = vec![0.0f64; nelem * nqpts * dim];
        basis
            .apply(nelem, false, EvalMode::Interp, &u_dof, &mut v_qpts)
            .unwrap();

        let mut u_dof_back = vec![0.0f64; nelem * ndof * dim];
        basis
            .apply(nelem, true, EvalMode::Interp, &v_qpts, &mut u_dof_back)
            .unwrap();

        for dof in 0..ndof {
            let val = u_dof_back[dof * dim];
            assert!(
                val.abs() > TOL,
                "RT2 transpose consistency: dof {dof} is zero"
            );
        }
    }

    #[test]
    fn tri_rt2_hdiv_transpose_consistency() {
        let basis = RaviartThomasBasis::<f64>::new(ElemTopology::Triangle, 2, 6).unwrap();
        let nelem = 1;
        let ndof = 15;
        let nqpts = 6;
        let dim = 2;

        let mut u_dof = vec![0.0f64; nelem * ndof * dim];
        for dof in 0..ndof {
            u_dof[dof * dim] = (dof + 1) as f64;
        }

        let mut v_div = vec![0.0f64; nelem * nqpts];
        basis
            .apply(nelem, false, EvalMode::HDiv, &u_dof, &mut v_div)
            .unwrap();

        let mut u_dof_back = vec![0.0f64; nelem * ndof * dim];
        basis
            .apply(nelem, true, EvalMode::HDiv, &v_div, &mut u_dof_back)
            .unwrap();

        for dof in 0..ndof {
            let val = u_dof_back[dof * dim];
            assert!(
                val.abs() > TOL,
                "RT2 HDiv transpose consistency: dof {dof} is zero"
            );
        }
    }
}
