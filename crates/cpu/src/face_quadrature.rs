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
/// For Triangle (dim=2): faces are 1D edges
/// For Tet (dim=3): faces are 2D triangles
///
/// `local_face` follows libCEED convention.
///
/// **v1 scope**: not yet implemented.
pub fn face_quadrature_simplex<T: Scalar>(
    _basis: &SimplexBasis<T>,
    _local_face: usize,
) -> ReedResult<(Vec<T>, Vec<T>)> {
    Err(ReedError::Basis(
        "face_quadrature_simplex not yet implemented".into(),
    ))
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

    #[test]
    fn test_simplex_not_implemented() {
        use reed_core::ElemTopology;
        let basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, 1, 1, 1).unwrap();
        let result = face_quadrature_simplex(&basis, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dim1_rejected() {
        let basis = LagrangeBasis::<f64>::new(1, 1, 2, 2, QuadMode::Gauss).unwrap();
        assert!(face_quadrature_tensor(&basis, 0).is_err());
    }
}
