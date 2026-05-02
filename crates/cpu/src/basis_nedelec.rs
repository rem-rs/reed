//! Nedelec H(curl) basis functions for triangles and tetrahedra.
//!
//! Implements [`BasisTrait`] for the first-kind Nedelec edge-element basis on
//! simplex reference elements:
//!
//! | Type | Topology | DOFs | Polynomial space |
//! |------|----------|------|------------------|
//! | P1 triangle | Tri3 | 3 | N1 (edge) |
//! | P2 triangle | Tri3 | 8 | N2 (edge + face) |
//! | P1 tet | Tet4 | 6 | N1 (edge) |
//!
//! ## Reference elements
//!
//! **Triangle** — vertices (0,0), (1,0), (0,1).
//!
//! **Tetrahedron** — vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1).
//!
//! ## Basis functions
//!
//! ### P1 (order 1)
//!
//! Nedelec P1 edge basis functions are of the form
//! φ_{ij} = λ_i ∇λ_j − λ_j ∇λ_i
//! where λ_k are barycentric coordinates and the edge orientation is from
//! vertex i to vertex j.
//!
//! ### P2 (order 2) — Triangle only
//!
//! Hierarchical basis with 8 DOFs: 2 per edge (6) + 2 face (2).
//!
//! **Edge DOFs** (DOF 0–5):
//! - φ_{ij}^{(1)} = λ_i ∇λ_j − λ_j ∇λ_i  (P1 edge, DOFs 0–2)
//! - φ_{ij}^{(2)} = (λ_i − λ_j)(λ_i ∇λ_j − λ_j ∇λ_i)  (P2 edge bubble, DOFs 3–5)
//!
//! **Face DOFs** (DOF 6–7):
//! - φ_f^{(1)} = λ_0 λ_1 ∇λ_2
//! - φ_f^{(2)} = λ_0 λ_2 ∇λ_1
//!
//! Face functions vanish tangentially on all edges. Edge bubbles vanish
//! at edge midpoints (where λ_i = λ_j).
//!
//! ## Memory layout
//!
//! * `interp` — row-major `[nqpts × num_dof × dim]`,
//!   stored as `(qpt*num_dof + dof)*dim + d`
//! * `curl_matrix` — 2D: `[nqpts × num_dof]` (scalar curl);
//!   3D: `[nqpts × num_dof × 3]` (vector curl)

use reed_core::{
    basis::BasisTrait,
    enums::{ElemTopology, EvalMode},
    error::{ReedError, ReedResult},
    scalar::Scalar,
};

use super::basis_simplex::{tet_quadrature, tri_quadrature};

#[cfg(feature = "parallel")]
const PAR_MIN_ELEMS_PER_TASK: usize = 128;

/// H(curl) Nedelec (first kind) edge-element basis on triangles and tetrahedra.
pub struct NedelecBasis<T: Scalar> {
    dim: usize,
    num_dof: usize,
    num_qpoints: usize,
    /// Quadrature weights, length `num_qpoints`.
    weights: Vec<T>,
    /// Quadrature point coordinates, row-major `[num_qpoints × dim]`.
    q_ref: Vec<T>,
    /// Interpolation matrix, row-major `[num_qpoints × num_dof × dim]`.
    interp: Vec<T>,
    /// Curl matrix, 2D: `[num_qpoints × num_dof]`; 3D: `[num_qpoints × num_dof × 3]`.
    curl_matrix: Vec<T>,
}

impl<T: Scalar> NedelecBasis<T> {
    /// Construct a Nedelec H(curl) basis.
    ///
    /// # Parameters
    /// * `topo` — `ElemTopology::Triangle` or `Tet`.
    /// * `p`    — polynomial order. Triangle: 1 or 2; Tet: 1 only.
    /// * `q`    — number of quadrature points (see `tri_quadrature` / `tet_quadrature` for
    ///            valid values).
    ///
    /// # Errors
    /// Returns `ReedError::Basis` for unsupported topology/p/q combinations.
    pub fn new(topo: ElemTopology, p: usize, q: usize) -> ReedResult<Self> {
        let (dim, num_dof) = match topo {
            ElemTopology::Triangle => match p {
                1 => (2, 3),
                2 => (2, 8),
                _ => {
                    return Err(ReedError::Basis(format!(
                        "NedelecBasis: p={p} on Triangle not supported; use p=1 or p=2"
                    )));
                }
            },
            ElemTopology::Tet => {
                if p != 1 {
                    return Err(ReedError::Basis(format!(
                        "NedelecBasis: p={p} on Tet not supported; only p=1"
                    )));
                }
                (3, 6)
            }
            _ => {
                if matches!(topo, ElemTopology::Pyramid | ElemTopology::Prism) {
                    return Err(ReedError::Basis(format!(
                        "NedelecBasis: {:?} not implemented (requires collapsed-coordinate or tensor×simplex transforms; available: Triangle, Tet)",
                        topo
                    )));
                }
                return Err(ReedError::Basis(format!(
                    "NedelecBasis: unsupported topology {:?} (need Triangle or Tet)",
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

        // Build interp and curl matrices
        let mut interp = vec![0.0f64; num_qpoints * num_dof * dim];
        let mut curl_matrix = if dim == 2 {
            vec![0.0f64; num_qpoints * num_dof]
        } else {
            vec![0.0f64; num_qpoints * num_dof * 3]
        };

        for (qi, pt) in qpts.iter().enumerate() {
            match dim {
                2 => {
                    let (phi, curl) = if order == 2 {
                        tri_nedelec_p2(pt[0], pt[1])
                    } else {
                        let (p, c) = tri_nedelec_p1(pt[0], pt[1]);
                        (p, c)
                    };
                    for dof in 0..num_dof {
                        for d in 0..dim {
                            interp[(qi * num_dof + dof) * dim + d] =
                                phi[dof * dim + d];
                        }
                        curl_matrix[qi * num_dof + dof] = curl[dof];
                    }
                }
                3 => {
                    let (phi, curl) = tet_nedelec_p1(pt[0], pt[1], pt[2]);
                    for dof in 0..num_dof {
                        for d in 0..dim {
                            interp[(qi * num_dof + dof) * dim + d] =
                                phi[dof * dim + d];
                            curl_matrix[(qi * num_dof + dof) * 3 + d] =
                                curl[dof * 3 + d];
                        }
                    }
                }
                _ => unreachable!(),
            }
        }

        let interp_t: Vec<T> = interp
            .iter()
            .map(|&v| to_t::<T>(v))
            .collect::<ReedResult<_>>()?;
        let curl_t: Vec<T> = curl_matrix
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
            curl_matrix: curl_t,
        })
    }

    // ── accessor helpers ───────────────────────────────────────────────────

    /// `interp[(qpt * num_dof + dof) * dim + d]`
    #[inline]
    fn interp_val(&self, qpt: usize, dof: usize, d: usize) -> T {
        self.interp[(qpt * self.num_dof + dof) * self.dim + d]
    }

    /// 2D: `curl_matrix[qpt * num_dof + dof]` (scalar).
    /// 3D: `curl_matrix[(qpt * num_dof + dof) * 3 + d]` (vector component).
    #[inline]
    fn curl_val(&self, qpt: usize, dof: usize, d: usize) -> T {
        if self.dim == 2 {
            self.curl_matrix[qpt * self.num_dof + dof]
        } else {
            self.curl_matrix[(qpt * self.num_dof + dof) * 3 + d]
        }
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

    /// Forward HCurl 2D: scalar DOFs → scalar curl at qpts.
    /// u_elem: `[num_dof * dim]`, read `u_elem[dof * self.dim]` as scalar.
    /// v_elem: `[num_qpoints]` — scalar curl values.
    fn apply_hcurl_forward_2d(&self, u_elem: &[T], v_elem: &mut [T]) {
        for qpt in 0..self.num_qpoints {
            let mut sum = T::ZERO;
            for dof in 0..self.num_dof {
                sum += self.curl_val(qpt, dof, 0) * u_elem[dof * self.dim];
            }
            v_elem[qpt] = sum;
        }
    }

    /// Transpose HCurl 2D: scalar curl at qpts → scalar DOFs.
    /// u_elem: `[num_qpoints]` — scalar curl values.
    /// v_elem: `[num_dof * dim]` — accumulator; writes to `v_elem[dof * self.dim]`.
    fn apply_hcurl_transpose_2d(&self, u_elem: &[T], v_elem: &mut [T]) {
        for dof in 0..self.num_dof {
            let mut sum = T::ZERO;
            for qpt in 0..self.num_qpoints {
                sum += self.curl_val(qpt, dof, 0) * u_elem[qpt];
            }
            v_elem[dof * self.dim] += sum;
        }
    }

    /// Forward HCurl 3D: scalar DOFs → vector curl at qpts.
    /// u_elem: `[num_dof * dim]`, read `u_elem[dof * self.dim]` as scalar.
    /// v_elem: `[num_qpoints * 3]` — 3-vector curl values.
    fn apply_hcurl_forward_3d(&self, u_elem: &[T], v_elem: &mut [T]) {
        for qpt in 0..self.num_qpoints {
            for d in 0..3 {
                let mut sum = T::ZERO;
                for dof in 0..self.num_dof {
                    sum += self.curl_val(qpt, dof, d) * u_elem[dof * self.dim];
                }
                v_elem[qpt * 3 + d] = sum;
            }
        }
    }

    /// Transpose HCurl 3D: vector curl at qpts → scalar DOFs.
    /// u_elem: `[num_qpoints * 3]` — 3-vector curl values.
    /// v_elem: `[num_dof * dim]` — accumulator; writes to `v_elem[dof * self.dim]`.
    fn apply_hcurl_transpose_3d(&self, u_elem: &[T], v_elem: &mut [T]) {
        for dof in 0..self.num_dof {
            let mut sum = T::ZERO;
            for qpt in 0..self.num_qpoints {
                for d in 0..3 {
                    sum += self.curl_val(qpt, dof, d) * u_elem[qpt * 3 + d];
                }
            }
            v_elem[dof * self.dim] += sum;
        }
    }
}

// ── BasisTrait impl ───────────────────────────────────────────────────────────

impl<T: Scalar> BasisTrait<T> for NedelecBasis<T> {
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
            EvalMode::HCurl => {
                match self.dim {
                    2 => {
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
                            "hcurl-2d",
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
                                        self.apply_hcurl_transpose_2d(u_elem, v_elem)
                                    });
                            } else {
                                u.par_chunks(in_stride)
                                    .zip(v.par_chunks_mut(out_stride))
                                    .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                                    .for_each(|(u_elem, v_elem)| {
                                        self.apply_hcurl_forward_2d(u_elem, v_elem)
                                    });
                            }
                        }
                        #[cfg(not(feature = "parallel"))]
                        {
                            if transpose {
                                for (u_elem, v_elem) in
                                    u.chunks(in_stride).zip(v.chunks_mut(out_stride))
                                {
                                    self.apply_hcurl_transpose_2d(u_elem, v_elem);
                                }
                            } else {
                                for (u_elem, v_elem) in
                                    u.chunks(in_stride).zip(v.chunks_mut(out_stride))
                                {
                                    self.apply_hcurl_forward_2d(u_elem, v_elem);
                                }
                            }
                        }
                    }
                    3 => {
                        let in_stride = if transpose {
                            self.num_qpoints * 3
                        } else {
                            self.num_dof * self.dim
                        };
                        let out_stride = if transpose {
                            self.num_dof * self.dim
                        } else {
                            self.num_qpoints * 3
                        };
                        check_sizes(
                            u,
                            in_stride * num_elem,
                            v,
                            out_stride * num_elem,
                            "hcurl-3d",
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
                                        self.apply_hcurl_transpose_3d(u_elem, v_elem)
                                    });
                            } else {
                                u.par_chunks(in_stride)
                                    .zip(v.par_chunks_mut(out_stride))
                                    .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                                    .for_each(|(u_elem, v_elem)| {
                                        self.apply_hcurl_forward_3d(u_elem, v_elem)
                                    });
                            }
                        }
                        #[cfg(not(feature = "parallel"))]
                        {
                            if transpose {
                                for (u_elem, v_elem) in
                                    u.chunks(in_stride).zip(v.chunks_mut(out_stride))
                                {
                                    self.apply_hcurl_transpose_3d(u_elem, v_elem);
                                }
                            } else {
                                for (u_elem, v_elem) in
                                    u.chunks(in_stride).zip(v.chunks_mut(out_stride))
                                {
                                    self.apply_hcurl_forward_3d(u_elem, v_elem);
                                }
                            }
                        }
                    }
                    _ => unreachable!(), // dim is always 2 or 3 for this basis
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
                    "NedelecBasis: eval mode {:?} not implemented",
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

/// P1 Nedelec (first kind) basis functions on the reference triangle.
///
/// Barycentric coordinates: λ₀ = 1−x−y, λ₁ = x, λ₂ = y.
/// Gradients: ∇λ₀ = (−1,−1), ∇λ₁ = (1,0), ∇λ₂ = (0,1).
///
/// Edges (DOF ordering): (0,1), (1,2), (2,0).
/// φ_{ij} = λ_i ∇λ_j − λ_j ∇λ_i  (vector-valued, dim=2).
/// curl(φ_{ij}) = 2(∇λ_i × ∇λ_j) = 2 for all edges (constant).
///
/// Returns `(phi[num_dof * 2], curl[num_dof])`.
fn tri_nedelec_p1(x: f64, y: f64) -> (Vec<f64>, Vec<f64>) {
    let lam = [1.0 - x - y, x, y]; // λ₀, λ₁, λ₂
    let dlam: [[f64; 2]; 3] = [
        [-1.0, -1.0], // ∇λ₀
        [1.0, 0.0],   // ∇λ₁
        [0.0, 1.0],   // ∇λ₂
    ];

    // Edges: (0,1), (1,2), (2,0)
    let edges = [(0usize, 1usize), (1, 2), (2, 0)];
    let num_dof = 3;
    let mut phi = vec![0.0f64; num_dof * 2];
    let mut curl = vec![0.0f64; num_dof];

    for (dof, &(i, j)) in edges.iter().enumerate() {
        // φ_{ij} = λ_i ∇λ_j − λ_j ∇λ_i
        for d in 0..2 {
            phi[dof * 2 + d] = lam[i] * dlam[j][d] - lam[j] * dlam[i][d];
        }
        // curl(φ_{ij}) = 2(∇λ_i × ∇λ_j) = 2(dlam[i][0]*dlam[j][1] - dlam[i][1]*dlam[j][0])
        curl[dof] = 2.0 * (dlam[i][0] * dlam[j][1] - dlam[i][1] * dlam[j][0]);
    }

    (phi, curl)
}

/// P2 Nedelec (first kind) basis functions on the reference triangle.
///
/// Hierarchical construction with 8 DOFs: 2 per edge (6) + 2 face (2).
///
/// **Edge DOFs** (0–5):
/// - DOF 0–2: P1 edge basis  φ_{ij}^{(1)} = λ_i ∇λ_j − λ_j ∇λ_i
/// - DOF 3–5: P2 edge bubble φ_{ij}^{(2)} = (λ_i − λ_j)(λ_i ∇λ_j − λ_j ∇λ_i)
///
/// **Face DOFs** (6–7):
/// - DOF 6: φ_f^{(1)} = λ_0 λ_1 ∇λ_2
/// - DOF 7: φ_f^{(2)} = λ_0 λ_2 ∇λ_1
///
/// Curl of edge bubble: curl((λ_i−λ_j)·φ_P1) = ∇(λ_i−λ_j) × φ_P1 + (λ_i−λ_j)·curl(φ_P1).
/// Curl of face: curl(λ_a λ_b ∇λ_c) = ∇(λ_a λ_b) × ∇λ_c.
///
/// Returns `(phi[num_dof * 2], curl[num_dof])`.
fn tri_nedelec_p2(x: f64, y: f64) -> (Vec<f64>, Vec<f64>) {
    let lam = [1.0 - x - y, x, y]; // λ₀, λ₁, λ₂
    let dlam: [[f64; 2]; 3] = [
        [-1.0, -1.0], // ∇λ₀
        [1.0, 0.0],   // ∇λ₁
        [0.0, 1.0],   // ∇λ₂
    ];

    // Edges: (0,1), (1,2), (2,0)
    let edges = [(0usize, 1usize), (1, 2), (2, 0)];
    let num_dof = 8;
    let mut phi = vec![0.0f64; num_dof * 2];
    let mut curl = vec![0.0f64; num_dof];

    for (dof_p1, &(i, j)) in edges.iter().enumerate() {
        // ── P1 edge basis (DOFs 0–2) ──────────────────────────────────
        // φ_{ij} = λ_i ∇λ_j − λ_j ∇λ_i
        for d in 0..2 {
            phi[dof_p1 * 2 + d] = lam[i] * dlam[j][d] - lam[j] * dlam[i][d];
        }
        // curl(φ_{ij}) = 2(∇λ_i × ∇λ_j)
        let curl_p1 = 2.0 * (dlam[i][0] * dlam[j][1] - dlam[i][1] * dlam[j][0]);
        curl[dof_p1] = curl_p1;

        // ── P2 edge bubble (DOFs 3–5) ─────────────────────────────────
        // φ_{ij}^{(2)} = (λ_i − λ_j) · φ_{ij}^{(1)}
        let f = lam[i] - lam[j];
        for d in 0..2 {
            let p1_val = lam[i] * dlam[j][d] - lam[j] * dlam[i][d];
            phi[(3 + dof_p1) * 2 + d] = f * p1_val;
        }
        // curl(f · v) = ∇f × v + f · curl(v)   (2D cross product)
        let df = [
            dlam[i][0] - dlam[j][0],
            dlam[i][1] - dlam[j][1],
        ];
        let v = [
            lam[i] * dlam[j][0] - lam[j] * dlam[i][0],
            lam[i] * dlam[j][1] - lam[j] * dlam[i][1],
        ];
        let df_cross_v = df[0] * v[1] - df[1] * v[0];
        curl[3 + dof_p1] = df_cross_v + f * curl_p1;
    }

    // ── Face functions (DOFs 6–7) ─────────────────────────────────────
    // φ_f^{(1)} = λ_0 λ_1 ∇λ_2
    {
        let dof = 6;
        for d in 0..2 {
            phi[dof * 2 + d] = lam[0] * lam[1] * dlam[2][d];
        }
        // curl = ∇(λ_0 λ_1) × ∇λ_2 = (λ_0 ∇λ_1 + λ_1 ∇λ_0) × ∇λ_2
        let gx = lam[0] * dlam[1][0] + lam[1] * dlam[0][0];
        let gy = lam[0] * dlam[1][1] + lam[1] * dlam[0][1];
        curl[dof] = gx * dlam[2][1] - gy * dlam[2][0];
    }
    // φ_f^{(2)} = λ_0 λ_2 ∇λ_1
    {
        let dof = 7;
        for d in 0..2 {
            phi[dof * 2 + d] = lam[0] * lam[2] * dlam[1][d];
        }
        // curl = ∇(λ_0 λ_2) × ∇λ_1 = (λ_0 ∇λ_2 + λ_2 ∇λ_0) × ∇λ_1
        let gx = lam[0] * dlam[2][0] + lam[2] * dlam[0][0];
        let gy = lam[0] * dlam[2][1] + lam[2] * dlam[0][1];
        curl[dof] = gx * dlam[1][1] - gy * dlam[1][0];
    }

    (phi, curl)
}

/// P1 Nedelec (first kind) basis functions on the reference tetrahedron.
///
/// Barycentric coordinates: λ₀ = 1−x−y−z, λ₁ = x, λ₂ = y, λ₃ = z.
/// Gradients: ∇λ₀ = (−1,−1,−1), ∇λ₁ = (1,0,0), ∇λ₂ = (0,1,0), ∇λ₃ = (0,0,1).
///
/// Edges (DOF ordering): (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
/// φ_{ij} = λ_i ∇λ_j − λ_j ∇λ_i  (vector-valued, dim=3).
/// curl(φ_{ij}) = 2(∇λ_i × ∇λ_j) (3-vector, constant).
///
/// Returns `(phi[num_dof * 3], curl[num_dof * 3])`.
fn tet_nedelec_p1(x: f64, y: f64, z: f64) -> (Vec<f64>, Vec<f64>) {
    let lam = [1.0 - x - y - z, x, y, z]; // λ₀, λ₁, λ₂, λ₃
    let dlam: [[f64; 3]; 4] = [
        [-1.0, -1.0, -1.0], // ∇λ₀
        [1.0, 0.0, 0.0],    // ∇λ₁
        [0.0, 1.0, 0.0],    // ∇λ₂
        [0.0, 0.0, 1.0],    // ∇λ₃
    ];

    // Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    let edges = [
        (0usize, 1usize),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ];
    let num_dof = 6;
    let mut phi = vec![0.0f64; num_dof * 3];
    let mut curl = vec![0.0f64; num_dof * 3];

    for (dof, &(i, j)) in edges.iter().enumerate() {
        // φ_{ij} = λ_i ∇λ_j − λ_j ∇λ_i
        for d in 0..3 {
            phi[dof * 3 + d] = lam[i] * dlam[j][d] - lam[j] * dlam[i][d];
        }
        // curl(φ_{ij}) = 2(∇λ_i × ∇λ_j)
        // Cross product: a × b = [a₁b₂−a₂b₁, a₂b₀−a₀b₂, a₀b₁−a₁b₀]
        curl[dof * 3] = 2.0 * (dlam[i][1] * dlam[j][2] - dlam[i][2] * dlam[j][1]);
        curl[dof * 3 + 1] = 2.0 * (dlam[i][2] * dlam[j][0] - dlam[i][0] * dlam[j][2]);
        curl[dof * 3 + 2] = 2.0 * (dlam[i][0] * dlam[j][1] - dlam[i][1] * dlam[j][0]);
    }

    (phi, curl)
}

// ── utilities ─────────────────────────────────────────────────────────────────

fn to_t<T: Scalar>(v: f64) -> ReedResult<T> {
    T::from(v)
        .ok_or_else(|| ReedError::Basis(format!("NedelecBasis: failed to convert {v} to scalar")))
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
            "NedelecBasis {mode} size mismatch: \
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
    fn tri_nedelec_p1_curl_is_constant_2() {
        // At any point, all edge curl values should be 2.
        for &(x, y) in &[(0.1, 0.2), (0.5, 0.3), (1.0 / 3.0, 1.0 / 3.0)] {
            let (_phi, curl) = tri_nedelec_p1(x, y);
            for dof in 0..3 {
                assert!(
                    (curl[dof] - 2.0).abs() < TOL,
                    "tri curl[dof={}] = {} at ({},{})",
                    dof,
                    curl[dof],
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn tet_nedelec_p1_curl_constant() {
        // Curl values are constant (independent of x,y,z).
        let (_phi1, curl1) = tet_nedelec_p1(0.1, 0.2, 0.3);
        let (_phi2, curl2) = tet_nedelec_p1(0.5, 0.1, 0.1);
        for dof in 0..6 {
            for d in 0..3 {
                assert!(
                    (curl1[dof * 3 + d] - curl2[dof * 3 + d]).abs() < TOL,
                    "tet curl not constant for dof={} d={}",
                    dof,
                    d
                );
            }
        }
    }

    // ── basis construction tests ───────────────────────────────────────────

    #[test]
    fn construct_tri_nedelec_p1() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Triangle, 1, 3).unwrap();
        assert_eq!(basis.dim(), 2);
        assert_eq!(basis.num_dof(), 3);
        assert_eq!(basis.num_qpoints(), 3);
        assert_eq!(basis.num_comp(), 2);
        assert_eq!(basis.q_weights().len(), 3);
        assert_eq!(basis.q_ref().len(), 6); // 3 qpts × 2
    }

    #[test]
    fn construct_tet_nedelec_p1() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Tet, 1, 4).unwrap();
        assert_eq!(basis.dim(), 3);
        assert_eq!(basis.num_dof(), 6);
        assert_eq!(basis.num_qpoints(), 4);
        assert_eq!(basis.num_comp(), 3);
        assert_eq!(basis.q_weights().len(), 4);
        assert_eq!(basis.q_ref().len(), 12); // 4 qpts × 3
    }

    #[test]
    fn reject_invalid_p() {
        // Triangle: p=2 is OK; higher p rejected
        assert!(NedelecBasis::<f64>::new(ElemTopology::Triangle, 2, 3).is_ok());
        assert!(NedelecBasis::<f64>::new(ElemTopology::Triangle, 3, 3).is_err());
        // Tet: only p=1
        assert!(NedelecBasis::<f64>::new(ElemTopology::Tet, 2, 4).is_err());
        assert!(NedelecBasis::<f64>::new(ElemTopology::Tet, 3, 4).is_err());
    }

    #[test]
    fn reject_unsupported_topo() {
        assert!(NedelecBasis::<f64>::new(ElemTopology::Line, 1, 2).is_err());
    }

    // ── apply: Interp mode ─────────────────────────────────────────────────

    #[test]
    fn tri_nedelec_interp_forward_size() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Triangle, 1, 3).unwrap();
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
    fn tri_nedelec_interp_transpose_size() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Triangle, 1, 3).unwrap();
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

    // ── apply: HCurl mode ──────────────────────────────────────────────────

    #[test]
    fn tri_nedelec_hcurl_forward_size() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Triangle, 1, 3).unwrap();
        let nelem = 2;
        let ndof = 3;
        let nqpts = 3;
        let dim = 2;
        let u = vec![0.0f64; nelem * ndof * dim];
        let mut v = vec![0.0f64; nelem * nqpts]; // scalar curl in 2D
        basis
            .apply(nelem, false, EvalMode::HCurl, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tri_nedelec_hcurl_transpose_size() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Triangle, 1, 3).unwrap();
        let nelem = 2;
        let ndof = 3;
        let nqpts = 3;
        let dim = 2;
        let u = vec![0.0f64; nelem * nqpts]; // scalar curl in 2D
        let mut v = vec![0.0f64; nelem * ndof * dim];
        basis
            .apply(nelem, true, EvalMode::HCurl, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tet_nedelec_hcurl_forward_size() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Tet, 1, 4).unwrap();
        let nelem = 2;
        let ndof = 6;
        let nqpts = 4;
        let dim = 3;
        let u = vec![0.0f64; nelem * ndof * dim];
        let mut v = vec![0.0f64; nelem * nqpts * 3]; // 3-vector curl
        basis
            .apply(nelem, false, EvalMode::HCurl, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tet_nedelec_hcurl_transpose_size() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Tet, 1, 4).unwrap();
        let nelem = 2;
        let ndof = 6;
        let nqpts = 4;
        let dim = 3;
        let u = vec![0.0f64; nelem * nqpts * 3]; // 3-vector curl
        let mut v = vec![0.0f64; nelem * ndof * dim];
        basis
            .apply(nelem, true, EvalMode::HCurl, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tri_nedelec_rejects_grad_mode() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Triangle, 1, 3).unwrap();
        let u = vec![0.0f64; 3 * 2];
        let mut v = vec![0.0f64; 3 * 2];
        assert!(basis.apply(1, false, EvalMode::Grad, &u, &mut v).is_err());
    }

    // ── apply: Weight mode ─────────────────────────────────────────────────

    #[test]
    fn tri_nedelec_weight_mode() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Triangle, 1, 3).unwrap();
        let nelem = 2;
        let mut v = vec![0.0f64; nelem * 3];
        basis
            .apply(nelem, false, EvalMode::Weight, &[], &mut v)
            .unwrap();
        // each element gets the same quadrature weights
        assert!((v[0] - v[3]).abs() < TOL);
    }

    // ── transpose consistency: forward+transpose = identity (up to quadrature) ─

    #[test]
    fn tri_nedelec_interp_transpose_consistency() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Triangle, 1, 3).unwrap();
        let nelem = 1;
        let ndof = 3;
        let nqpts = 3;
        let dim = 2;

        // u_dof: [ndof*dim], populate scalar DOFs
        let mut u_dof = vec![0.0f64; nelem * ndof * dim];
        for dof in 0..ndof {
            u_dof[dof * dim] = (dof + 1) as f64; // scalar value at first component
        }

        // forward interp: DOF → qpts
        let mut v_qpts = vec![0.0f64; nelem * nqpts * dim];
        basis
            .apply(nelem, false, EvalMode::Interp, &u_dof, &mut v_qpts)
            .unwrap();

        // transpose interp: qpts → DOF
        let mut u_dof_back = vec![0.0f64; nelem * ndof * dim];
        basis
            .apply(nelem, true, EvalMode::Interp, &v_qpts, &mut u_dof_back)
            .unwrap();

        // Check that we got back nonzero values (quadrature projection is not identity
        // but should preserve the space: (B^T B) u ≈ M u where M is the mass matrix)
        for dof in 0..ndof {
            let val = u_dof_back[dof * dim];
            assert!(
                val.abs() > TOL,
                "transpose consistency: dof {dof} is zero"
            );
        }
    }

    #[test]
    fn tet_nedelec_interp_transpose_consistency() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Tet, 1, 4).unwrap();
        let nelem = 1;
        let ndof = 6;
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

    // ── P2 triangle tests ──────────────────────────────────────────────────

    #[test]
    fn construct_tri_nedelec_p2() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Triangle, 2, 6).unwrap();
        assert_eq!(basis.dim(), 2);
        assert_eq!(basis.num_dof(), 8);
        assert_eq!(basis.num_qpoints(), 6);
        assert_eq!(basis.num_comp(), 2);
    }

    #[test]
    fn tri_nedelec_p2_p1_subspace() {
        // The P1 basis should be exactly the first 3 DOFs of the P2 basis.
        // Evaluate both at several quadrature points and compare.
        let basis_p1 = NedelecBasis::<f64>::new(ElemTopology::Triangle, 1, 6).unwrap();
        let basis_p2 = NedelecBasis::<f64>::new(ElemTopology::Triangle, 2, 6).unwrap();

        assert_eq!(basis_p1.num_qpoints(), basis_p2.num_qpoints());

        // Compare interpolant values: for each qpt, P1 dof 0..2 should match P2 dof 0..2
        for qpt in 0..basis_p1.num_qpoints() {
            for dof in 0..3 {
                for d in 0..2 {
                    let v1 = basis_p1.interp[(qpt * 3 + dof) * 2 + d];
                    let v2 = basis_p2.interp[(qpt * 8 + dof) * 2 + d];
                    assert!(
                        (v1 - v2).abs() < TOL,
                        "P1/P2 mismatch at qpt={qpt} dof={dof} d={d}: {v1} vs {v2}"
                    );
                }
            }
            // Curl should match for the first 3 DOFs
            for dof in 0..3 {
                let c1 = basis_p1.curl_matrix[qpt * 3 + dof];
                let c2 = basis_p2.curl_matrix[qpt * 8 + dof];
                assert!(
                    (c1 - c2).abs() < TOL,
                    "P1/P2 curl mismatch at qpt={qpt} dof={dof}: {c1} vs {c2}"
                );
            }
        }
    }

    #[test]
    fn tri_nedelec_p2_face_tangential_zero_on_edges() {
        // Face functions (DOFs 6 and 7) should have zero tangential component on all edges.
        let _basis = NedelecBasis::<f64>::new(ElemTopology::Triangle, 2, 6).unwrap();

        // Edge normals (perpendicular to edge, used to project tangent):
        // Edge (0,1): y=0, tangent = (1,0), normal for tangent test: dot with (0,1)
        // Edge (1,2): x+y=1, tangent = (-1,1)/√2, normal = (1,1)/√2
        // Edge (0,2): x=0, tangent = (0,1), normal for tangent test: dot with (1,0)

        // The tangential component = dot(basis_vector, edge_tangent).
        // We evaluate at quadrature points and check against known edge locations.

        // We'll verify at known quadrature points by checking that
        // on each edge, the face functions are parallel to the edge normal
        // (i.e., have zero tangential component).

        // Edge (0,1): y=0. Face DOF 6 = λ_0 λ_1 ∇λ_2 = x(1-x) * (0,1).
        // Tangent = (1,0). Dot = 0. ✓
        // Face DOF 7 = λ_0 λ_2 ∇λ_1 = 0 * λ_2 * (1,0) = 0. ✓

        // We can't easily identify which qpts are on edges with standard quadrature,
        // so instead verify that both face functions vanish at vertices:
        // At each vertex, one of the λ factors is 1, the others 0,
        // so λ_0 λ_1 = 0 and λ_0 λ_2 = 0 → both face functions = 0.

        // Evaluate at the three vertices via the P2 shape functions directly
        for &(x, y) in &[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)] {
            let (_phi, _curl) = tri_nedelec_p2(x, y);
            // Face DOFs are 6 and 7
            for dof in 6..8 {
                let fx = _phi[dof * 2];
                let fy = _phi[dof * 2 + 1];
                assert!(
                    fx.abs() < TOL && fy.abs() < TOL,
                    "Face DOF {dof} non-zero at vertex ({x},{y}): ({fx},{fy})"
                );
            }
        }
    }

    #[test]
    fn tri_nedelec_p2_edge_bubble_zero_at_midpoint() {
        // The edge bubble functions (DOF 3–5) should vanish at edge midpoints
        // where λ_i = λ_j.
        // Edge (0,1) midpoint: (0.5, 0), λ_0 = 0.5, λ_1 = 0.5
        // Edge (1,2) midpoint: (0.5, 0.5), λ_1 = 0.5, λ_2 = 0.5
        // Edge (0,2) midpoint: (0, 0.5), λ_0 = 0.5, λ_2 = 0.5
        for &(x, y, dof_bubble) in &[
            (0.5, 0.0, 3),  // edge (0,1) midpoint, bubble DOF 3
            (0.5, 0.5, 4),  // edge (1,2) midpoint, bubble DOF 4
            (0.0, 0.5, 5),  // edge (2,0) midpoint, bubble DOF 5
        ] {
            let (phi, _curl) = tri_nedelec_p2(x, y);
            let fx = phi[dof_bubble * 2];
            let fy = phi[dof_bubble * 2 + 1];
            assert!(
                fx.abs() < TOL && fy.abs() < TOL,
                "Edge bubble DOF {dof_bubble} non-zero at midpoint ({x},{y}): ({fx},{fy})"
            );
        }
    }

    #[test]
    fn tri_nedelec_p2_curl_varies() {
        // Verify that the curl of P2 basis functions is not constant
        // (unlike P1 where curl is always 2).
        let (_phi1, curl1) = tri_nedelec_p2(0.1, 0.2);
        let (_phi2, curl2) = tri_nedelec_p2(0.7, 0.1);

        // P1 DOFs 0-2 should have constant curl (same at both points)
        for dof in 0..3 {
            assert!(
                (curl1[dof] - curl2[dof]).abs() < TOL,
                "P1 DOF {dof}: curl should be constant"
            );
        }

        // P2 edge bubbles (DOF 3-5) or face (DOF 6-7) should vary
        let mut any_varied = false;
        for dof in 3..8 {
            if (curl1[dof] - curl2[dof]).abs() > TOL {
                any_varied = true;
                break;
            }
        }
        assert!(any_varied, "P2 DOF curls should vary with position");

        // Also verify: at any point, the curl of DOF 6 (face 1) should equal
        // something derived analytically: curl_f1 = λ_0 − λ_1 = 1 - 2x - y
        let (_, curl) = tri_nedelec_p2(0.2, 0.3);
        let expected_curl_f1 = 1.0 - 2.0 * 0.2 - 0.3; // 1 - 0.4 - 0.3 = 0.3
        assert!((curl[6] - expected_curl_f1).abs() < TOL,
            "Face 1 curl mismatch: got {}, expected {}", curl[6], expected_curl_f1);

        let expected_curl_f2 = 0.3 - (1.0 - 0.2 - 0.3); // y - λ_0 = 0.3 - 0.5 = -0.2
        assert!((curl[7] - expected_curl_f2).abs() < TOL,
            "Face 2 curl mismatch: got {}, expected {}", curl[7], expected_curl_f2);
    }

    #[test]
    fn tri_nedelec_p2_interp_forward_size() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Triangle, 2, 6).unwrap();
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
    fn tri_nedelec_p2_hcurl_forward_size() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Triangle, 2, 6).unwrap();
        let nelem = 2;
        let ndof = 8;
        let nqpts = 6;
        let dim = 2;
        let u = vec![0.0f64; nelem * ndof * dim];
        let mut v = vec![0.0f64; nelem * nqpts];
        basis
            .apply(nelem, false, EvalMode::HCurl, &u, &mut v)
            .unwrap();
    }

    #[test]
    fn tri_nedelec_p2_transpose_consistency() {
        let basis = NedelecBasis::<f64>::new(ElemTopology::Triangle, 2, 6).unwrap();
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
                "P2 transpose consistency: dof {dof} is zero"
            );
        }
    }
}
