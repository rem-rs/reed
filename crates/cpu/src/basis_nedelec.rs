//! Nedelec H(curl) basis functions for triangles and tetrahedra.
//!
//! Implements [`BasisTrait`] for the first-kind Nedelec edge-element basis on
//! simplex reference elements:
//!
//! | Type | Topology | DOFs | Polynomial space |
//! |------|----------|------|------------------|
//! | P1 triangle | Tri3 | 3 | N1 (edge) |
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
//! Nedelec P1 edge basis functions are of the form
//! φ_{ij} = λ_i ∇λ_j − λ_j ∇λ_i
//! where λ_k are barycentric coordinates and the edge orientation is from
//! vertex i to vertex j.
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
    /// * `p`    — polynomial order (must be 1 for P1 Nedelec).
    /// * `q`    — number of quadrature points (see `tri_quadrature` / `tet_quadrature` for
    ///            valid values).
    ///
    /// # Errors
    /// Returns `ReedError::Basis` for unsupported topology/p/q combinations.
    pub fn new(topo: ElemTopology, p: usize, q: usize) -> ReedResult<Self> {
        if p != 1 {
            return Err(ReedError::Basis(format!(
                "NedelecBasis: p={p} not supported; only p=1"
            )));
        }

        let (dim, num_dof) = match topo {
            ElemTopology::Triangle => (2, 3),
            ElemTopology::Tet => (3, 6),
            _ => {
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
                    let (phi, curl) = tri_nedelec_p1(pt[0], pt[1]);
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
    fn reject_non_p1() {
        assert!(
            NedelecBasis::<f64>::new(ElemTopology::Triangle, 2, 3).is_err()
        );
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
}
