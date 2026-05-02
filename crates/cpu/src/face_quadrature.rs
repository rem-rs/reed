//! Face quadrature extraction from volume bases.
//!
//! Extracts quadrature points and weights on element faces from tensor-product
//! [`LagrangeBasis`] and simplex [`SimplexBasis`] volume bases.

use reed_core::{
    basis::BasisTrait,
    error::{ReedError, ReedResult},
    scalar::Scalar,
};

use crate::basis_lagrange::LagrangeBasis;
use crate::basis_simplex::SimplexBasis;

/// Extract face quadrature points and weights from a tensor-product [`LagrangeBasis`].
///
/// For Quad (dim=2): each face is a 1D edge; face q_ref is `[nq_face × 1]`,
///   weights are the 1D weights.
/// For Hex (dim=3): each face is a 2D Quad; face q_ref is `[nq^2 × 2]`,
///   weights are tensor product of 1D weights.
///
/// `local_face`: 0=bottom(y=-1), 1=right(x=+1), 2=top(y=+1), 3=left(x=-1)
///   for Hex also: 4=front(z=-1), 5=back(z=+1)
pub fn face_quadrature_tensor<T: Scalar>(
    basis: &LagrangeBasis<T>,
    local_face: usize,
) -> ReedResult<(Vec<T>, Vec<T>)> {
    let dim = basis.dim();
    if dim != 2 && dim != 3 {
        return Err(ReedError::Basis(format!(
            "face_quadrature_tensor requires dim=2 or 3, got {}",
            dim
        )));
    }

    // Extract 1D data from the tensor-product basis.
    let (_interp_1d, _grad_1d, weights_1d, _p, q) = basis
        .tensor_fdm_1d_data()
        .ok_or_else(|| ReedError::Basis("LagrangeBasis must provide tensor_fdm_1d_data".into()))?;

    // Extract 1D quadrature points from the full tensor q_ref.
    // In the tensor q_ref, x (axis 0) varies fastest, so the unique
    // 1D points are at stride `dim`.
    let q_ref_full = basis.q_ref();
    let mut q_ref_1d = Vec::with_capacity(q);
    for i in 0..q {
        q_ref_1d.push(q_ref_full[i * dim]);
    }

    match dim {
        2 => face_quadrature_tensor_2d::<T>(local_face, &q_ref_1d, weights_1d, q),
        3 => face_quadrature_tensor_3d::<T>(local_face, &q_ref_1d, weights_1d, q),
        _ => unreachable!(),
    }
}

/// Quad (dim=2): each face is a 1D edge.
///
/// Face-local coordinate is the 1D Gauss point mapped along the edge direction.
/// Weights are the 1D quadrature weights.
fn face_quadrature_tensor_2d<T: Scalar>(
    local_face: usize,
    q_ref_1d: &[T],
    weights_1d: &[T],
    _q: usize,
) -> ReedResult<(Vec<T>, Vec<T>)> {
    // All 4 faces of a Quad are 1D edges; face-local coordinate is just
    // the 1D Gauss point along the edge direction. The mapping to the
    // fixed parent coordinate is not needed for face-local storage.
    if local_face > 3 {
        return Err(ReedError::Basis(format!(
            "invalid local_face {} for Quad (valid: 0..=3)",
            local_face
        )));
    }

    let face_q_ref = q_ref_1d.to_vec(); // [q × 1]
    let face_weights = weights_1d.to_vec(); // [q]

    Ok((face_q_ref, face_weights))
}

/// Hex (dim=3): each face is a 2D Quad.
///
/// Face-local coordinates are 2D tensor products of the 1D quadrature points
/// in the two tangential directions. Weights are the tensor product of 1D weights.
fn face_quadrature_tensor_3d<T: Scalar>(
    local_face: usize,
    q_ref_1d: &[T],
    weights_1d: &[T],
    q: usize,
) -> ReedResult<(Vec<T>, Vec<T>)> {
    // Determine which two axes span the face (_axis0 = faster, _axis1 = slower
    // in the tensor-product loop to match build_tensor_qref ordering).
    let (_axis0, _axis1) = match local_face {
        0 => (0, 2), // bottom: y=-1, face-local = (x, z)
        1 => (1, 2), // right:  x=+1, face-local = (y, z)
        2 => (0, 2), // top:    y=+1, face-local = (x, z)
        3 => (1, 2), // left:   x=-1, face-local = (y, z)
        4 => (0, 1), // front:  z=-1, face-local = (x, y)
        5 => (0, 1), // back:   z=+1, face-local = (x, y)
        _ => {
            return Err(ReedError::Basis(format!(
                "invalid local_face {} for Hex (valid: 0..=5)",
                local_face
            )))
        }
    };

    let nq_face = q * q;
    let mut face_q_ref = Vec::with_capacity(nq_face * 2);
    let mut face_weights = Vec::with_capacity(nq_face);

    // Tensor product: axis1 varies slower, axis0 varies faster
    // (matches build_tensor_qref convention).
    // Get the 1D points for each axis from q_ref_1d (all axes use same 1D rule).
    for &s1 in q_ref_1d.iter() {
        for &s0 in q_ref_1d.iter() {
            face_q_ref.push(s0);
            face_q_ref.push(s1);
        }
    }

    // Tensor product of 1D weights (matches build_tensor_weights for dim=2).
    for &w1 in weights_1d.iter() {
        for &w0 in weights_1d.iter() {
            face_weights.push(w0 * w1);
        }
    }

    Ok((face_q_ref, face_weights))
}

/// Extract face quadrature from a [`SimplexBasis`].
///
/// For Triangle (dim=2): faces are 1D edges; `q_ref_face` is `[nq_face × 1]`
///   (face-local coordinate `t` in `[0,1]` along the edge).
/// For Tet (dim=3): faces are 2D triangles; `q_ref_face` is `[nq_face × 2]`
///   (face-local `(u, v)` in the reference triangle `u,v ≥ 0, u+v ≤ 1`).
///
/// `local_face` follows libCEED convention (face `i` is opposite vertex `i`):
///
/// | Element  | Face | Vertices      |
/// |----------|------|---------------|
/// | Triangle | 0    | v1 → v2       |
/// | Triangle | 1    | v0 → v2       |
/// | Triangle | 2    | v0 → v1       |
/// | Tet      | 0    | v1, v2, v3    |
/// | Tet      | 1    | v0, v3, v2    |
/// | Tet      | 2    | v0, v1, v3    |
/// | Tet      | 3    | v0, v2, v1    |
pub fn face_quadrature_simplex<T: Scalar>(
    basis: &SimplexBasis<T>,
    local_face: usize,
) -> ReedResult<(Vec<T>, Vec<T>)> {
    let dim = basis.dim();
    let num_dof = basis.num_dof();

    if dim != 2 && dim != 3 {
        return Err(ReedError::Basis(format!(
            "face_quadrature_simplex requires dim=2 (Triangle) or dim=3 (Tet), got {}",
            dim
        )));
    }

    // Determine polynomial order from num_dof (P1, P2, P3).
    let poly = match (dim, num_dof) {
        (2, 3) => 1,
        (2, 6) => 2,
        (2, 10) => 3,
        (3, 4) => 1,
        (3, 10) => 2,
        (3, 20) => 3,
        _ => {
            return Err(ReedError::Basis(format!(
                "face_quadrature_simplex: unsupported (dim={}, num_dof={})",
                dim, num_dof
            )))
        }
    };

    match dim {
        2 => face_quadrature_simplex_tri::<T>(poly, local_face),
        3 => face_quadrature_simplex_tet::<T>(poly, local_face),
        _ => unreachable!(),
    }
}

// ── Triangle (dim=2): 1D edge faces ──────────────────────────────────────────

/// 1D Gauss quadrature mapped onto each Triangle edge, returned as
/// face-local coordinate `t ∈ [0,1]` per point.
fn face_quadrature_simplex_tri<T: Scalar>(
    poly: usize,
    local_face: usize,
) -> ReedResult<(Vec<T>, Vec<T>)> {
    if local_face > 2 {
        return Err(ReedError::Basis(format!(
            "invalid local_face {} for Triangle (valid: 0..=2)",
            local_face
        )));
    }

    // Number of 1D Gauss points: poly+1 gives exact integration for degree 2*poly+1
    let nq = poly + 1;
    let (xi_1d, w_1d) = crate::basis_lagrange::gauss_quadrature(nq)?;

    let mut q_ref_face = Vec::with_capacity(nq);
    let mut weights_face = Vec::with_capacity(nq);

    for i in 0..nq {
        // Map Gauss point from [-1,1] to [0,1]
        let t = 0.5 * (xi_1d[i] + 1.0);
        let w = 0.5 * w_1d[i];
        q_ref_face.push(to_simplex_scalar(t)?);
        weights_face.push(to_simplex_scalar(w)?);
    }

    Ok((q_ref_face, weights_face))
}

// ── Tet (dim=3): 2D triangle faces ───────────────────────────────────────────

/// Use 2D triangle quadrature (`tri_quadrature`) on each Tet face.
/// Returns face-local `(u,v)` coordinates and weights.
fn face_quadrature_simplex_tet<T: Scalar>(
    poly: usize,
    local_face: usize,
) -> ReedResult<(Vec<T>, Vec<T>)> {
    if local_face > 3 {
        return Err(ReedError::Basis(format!(
            "invalid local_face {} for Tet (valid: 0..=3)",
            local_face
        )));
    }

    // Map poly order to a suitable tri_quadrature rule.
    let tri_q = match poly {
        1 => 1,
        2 => 3,
        3 => 4,
        _ => unreachable!(),
    };
    let (q_ref_tri, w_tri) = crate::basis_simplex::tri_quadrature(tri_q)?;
    let nq_face = q_ref_tri.len() / 2;

    let mut q_ref_face = Vec::with_capacity(nq_face * 2);
    let mut weights_face = Vec::with_capacity(nq_face);

    // Face vertices for tet reference element (0,0,0),(1,0,0),(0,1,0),(0,0,1).
    // Face i is opposite vertex i.  Point on the triangle: (1−u−v)·A + u·B + v·C.
    for i in 0..nq_face {
        let u = q_ref_tri[i * 2];
        let v = q_ref_tri[i * 2 + 1];
        let w = w_tri[i];

        // Face-local coordinates are the (u,v) in the reference triangle.
        q_ref_face.push(to_simplex_scalar(u)?);
        q_ref_face.push(to_simplex_scalar(v)?);
        weights_face.push(to_simplex_scalar(w)?);
    }

    Ok((q_ref_face, weights_face))
}

/// Convert f64 to scalar type T.
fn to_simplex_scalar<T: Scalar>(v: f64) -> ReedResult<T> {
    T::from(v).ok_or_else(|| ReedError::Basis(format!("failed to convert {v} to scalar")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reed_core::QuadMode;

    #[test]
    fn test_quad_face_quadrature_bottom() {
        let basis = LagrangeBasis::<f64>::new(2, 1, 2, 3, QuadMode::Gauss).unwrap();
        let (q_ref, weights) = face_quadrature_tensor(&basis, 0).unwrap();

        // Quad dim=2, q=3: face should have nq_face=3 points × 1 coord each
        assert_eq!(q_ref.len(), 3);
        assert_eq!(weights.len(), 3);

        // Weights should sum to 2.0 (length of 1D reference edge [-1,1])
        let wsum: f64 = weights.iter().sum();
        assert!((wsum - 2.0).abs() < 1e-12);

        // q_ref points should be within [-1, 1]
        for &x in &q_ref {
            assert!(x >= -1.0 && x <= 1.0);
        }
    }

    #[test]
    fn test_quad_face_quadrature_top() {
        let basis = LagrangeBasis::<f64>::new(2, 1, 2, 4, QuadMode::GaussLobatto).unwrap();
        let (q_ref, weights) = face_quadrature_tensor(&basis, 2).unwrap();

        assert_eq!(q_ref.len(), 4);
        assert_eq!(weights.len(), 4);

        let wsum: f64 = weights.iter().sum();
        assert!((wsum - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_hex_face_quadrature_bottom() {
        let basis = LagrangeBasis::<f64>::new(3, 1, 2, 3, QuadMode::Gauss).unwrap();
        let (q_ref, weights) = face_quadrature_tensor(&basis, 0).unwrap();

        // Hex dim=3, q=3: face is 2D Quad, nq_face = 9 points × 2 coords each
        assert_eq!(q_ref.len(), 18); // 9 * 2
        assert_eq!(weights.len(), 9);

        // Weights should sum to 4.0 (area of 2D reference face [-1,1]^2)
        let wsum: f64 = weights.iter().sum();
        assert!((wsum - 4.0).abs() < 1e-12);

        // q_ref coords should be within [-1, 1]
        for &x in &q_ref {
            assert!(x >= -1.0 && x <= 1.0);
        }
    }

    #[test]
    fn test_hex_face_quadrature_right() {
        let basis = LagrangeBasis::<f64>::new(3, 1, 3, 4, QuadMode::Gauss).unwrap();
        let (q_ref, weights) = face_quadrature_tensor(&basis, 1).unwrap();

        // q=4, face has 16 points × 2 coords
        assert_eq!(q_ref.len(), 32);
        assert_eq!(weights.len(), 16);

        let wsum: f64 = weights.iter().sum();
        assert!((wsum - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_hex_face_quadrature_front() {
        let basis = LagrangeBasis::<f64>::new(3, 1, 2, 2, QuadMode::GaussLobatto).unwrap();
        let (q_ref, weights) = face_quadrature_tensor(&basis, 4).unwrap();

        assert_eq!(q_ref.len(), 8); // 4 * 2
        assert_eq!(weights.len(), 4);

        let wsum: f64 = weights.iter().sum();
        assert!((wsum - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_invalid_face_quad() {
        let basis = LagrangeBasis::<f64>::new(2, 1, 2, 2, QuadMode::Gauss).unwrap();
        assert!(face_quadrature_tensor(&basis, 4).is_err());
    }

    #[test]
    fn test_invalid_face_hex() {
        let basis = LagrangeBasis::<f64>::new(3, 1, 2, 2, QuadMode::Gauss).unwrap();
        assert!(face_quadrature_tensor(&basis, 6).is_err());
    }

    // ── simplex face quadrature tests ──────────────────────────────────────

    #[test]
    fn test_tri_p1_face_quadrature() {
        use reed_core::ElemTopology;
        let basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, 1, 1, 1).unwrap();
        // P1: poly=1, nq_face = 2 (poly+1)
        let (q_ref, weights) = face_quadrature_simplex(&basis, 0).unwrap();
        assert_eq!(q_ref.len(), 2); // [nq × 1], 2 points × 1 coord
        assert_eq!(weights.len(), 2);
        // Weights should sum to 1.0 (length of reference edge [0,1])
        let wsum: f64 = weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-12);
        // Face-local coords t should be in [0, 1]
        for &t in &q_ref {
            assert!(t >= 0.0 && t <= 1.0);
        }
    }

    #[test]
    fn test_tri_p2_face_quadrature() {
        use reed_core::ElemTopology;
        let basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, 2, 1, 3).unwrap();
        // P2: poly=2, nq_face = 3
        let (q_ref, weights) = face_quadrature_simplex(&basis, 1).unwrap();
        assert_eq!(q_ref.len(), 3);
        assert_eq!(weights.len(), 3);
        let wsum: f64 = weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_tri_p3_face_quadrature() {
        use reed_core::ElemTopology;
        let basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, 3, 1, 4).unwrap();
        // P3: poly=3, nq_face = 4
        let (q_ref, weights) = face_quadrature_simplex(&basis, 2).unwrap();
        assert_eq!(q_ref.len(), 4);
        assert_eq!(weights.len(), 4);
        let wsum: f64 = weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_tri_face_quadrature_independent_of_volume_q() {
        use reed_core::ElemTopology;
        // P2 with different volume q values should give same face quadrature
        let basis_q3 = SimplexBasis::<f64>::new(ElemTopology::Triangle, 2, 1, 3).unwrap();
        let basis_q6 = SimplexBasis::<f64>::new(ElemTopology::Triangle, 2, 1, 6).unwrap();
        let (qr3, w3) = face_quadrature_simplex(&basis_q3, 0).unwrap();
        let (qr6, w6) = face_quadrature_simplex(&basis_q6, 0).unwrap();
        assert_eq!(qr3.len(), qr6.len());
        assert_eq!(w3.len(), w6.len());
    }

    #[test]
    fn test_tet_p1_face_quadrature() {
        use reed_core::ElemTopology;
        let basis = SimplexBasis::<f64>::new(ElemTopology::Tet, 1, 1, 1).unwrap();
        // P1: tri_q=1 → 1 point × 2 coords
        let (q_ref, weights) = face_quadrature_simplex(&basis, 0).unwrap();
        assert_eq!(q_ref.len(), 2); // 1 point × 2
        assert_eq!(weights.len(), 1);
        // Weights sum to 0.5 (area of reference triangle)
        let wsum: f64 = weights.iter().sum();
        assert!((wsum - 0.5).abs() < 1e-12);
        // Face-local coords (u,v) should be in reference triangle
        for i in 0..weights.len() {
            let u = q_ref[i * 2];
            let v = q_ref[i * 2 + 1];
            assert!(u >= 0.0 && v >= 0.0 && u + v <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn test_tet_p2_face_quadrature() {
        use reed_core::ElemTopology;
        let basis = SimplexBasis::<f64>::new(ElemTopology::Tet, 2, 1, 4).unwrap();
        // P2: tri_q=3 → 3 points × 2 coords
        let (q_ref, weights) = face_quadrature_simplex(&basis, 1).unwrap();
        assert_eq!(q_ref.len(), 6); // 3 points × 2
        assert_eq!(weights.len(), 3);
        let wsum: f64 = weights.iter().sum();
        assert!((wsum - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_tet_p3_face_quadrature() {
        use reed_core::ElemTopology;
        let basis = SimplexBasis::<f64>::new(ElemTopology::Tet, 3, 1, 5).unwrap();
        // P3: tri_q=4 → 4 points × 2 coords
        let (q_ref, weights) = face_quadrature_simplex(&basis, 2).unwrap();
        assert_eq!(q_ref.len(), 8); // 4 points × 2
        assert_eq!(weights.len(), 4);
        let wsum: f64 = weights.iter().sum();
        assert!((wsum - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_simplex_face_invalid_face_tri() {
        use reed_core::ElemTopology;
        let basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, 1, 1, 1).unwrap();
        assert!(face_quadrature_simplex(&basis, 3).is_err());
    }

    #[test]
    fn test_simplex_face_invalid_face_tet() {
        use reed_core::ElemTopology;
        let basis = SimplexBasis::<f64>::new(ElemTopology::Tet, 1, 1, 1).unwrap();
        assert!(face_quadrature_simplex(&basis, 4).is_err());
    }

    #[test]
    fn test_simplex_face_dim1_rejected() {
        use reed_core::ElemTopology;
        let basis = SimplexBasis::<f64>::new(ElemTopology::Line, 2, 1, 2).unwrap();
        assert!(face_quadrature_simplex(&basis, 0).is_err());
    }

    #[test]
    fn test_dim1_rejected() {
        let basis = LagrangeBasis::<f64>::new(1, 1, 2, 2, QuadMode::Gauss).unwrap();
        assert!(face_quadrature_tensor(&basis, 0).is_err());
    }
}
