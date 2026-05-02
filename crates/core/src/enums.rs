/// Memory location type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemType {
    Host,
    Device,
}

/// Basis evaluation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalMode {
    None,
    /// Interpolation B·u.
    Interp,
    /// Gradient ∇B·u.
    Grad,
    /// Divergence ∇·B·u (vector fields, `ncomp == dim`; on `LagrangeBasis` / `SimplexBasis`, this is the sum of Cartesian component partial derivatives).
    Div,
    /// Curl ∇×B·u (2D: `ncomp=2` outputs a scalar; 3D: `ncomp=3` outputs 3 components; layout matches `Grad`).
    Curl,
    /// Quadrature weights w_q.
    Weight,
    /// Curl of H(curl) basis (Nédélec): ∇×φ. 2D→scalar, 3D→3-vector at each q-pt.
    HCurl,
    /// Divergence of H(div) basis (Raviart-Thomas): ∇·ψ. Scalar at each q-pt.
    HDiv,
}

/// Quadrature point distribution type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadMode {
    Gauss,
    GaussLobatto,
}

/// Transpose mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransposeMode {
    NoTranspose,
    Transpose,
}

/// Vector norm type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// L1 norm.
    One,
    /// L2 norm.
    Two,
    /// L-infinity norm.
    Max,
}

/// Element topology type (reference element names aligned with libCEED).
///
/// **CPU `SimplexBasis` (`Reed::basis_h1_simplex`)**: H1 Lagrange is implemented on `Line` / `Triangle` / `Tet`.
/// **Tensor-product `LagrangeBasis` (`Reed::basis_tensor_h1_lagrange`)**: `Quad` (`dim=2`), `Hex` (`dim=3`), etc.
/// **`Pyramid` / `Prism`**: enum placeholders for libCEED mesh-type alignment. Neither topology is implemented
/// in any basis type yet. Pyramid requires collapsed-coordinate transforms (mapping a hex to a 5-vertex pyramid);
/// Prism (wedge) requires a tensor×simplex product basis. Basis constructors return `ReedError::Basis` with
/// a specific message for these topologies.
///
/// **H(curl) / H(div)**: Nédélec (`basis_hcurl_nedelec`) and Raviart-Thomas (`basis_hdiv_raviart_thomas`)
/// elements are available on `Triangle` / `Tet` (v1: P1/RT0 only; P2/P3 and RT1/RT2 are planned).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemTopology {
    Line,
    Triangle,
    Quad,
    Tet,
    Pyramid,
    Prism,
    Hex,
}
