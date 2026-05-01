# libCEED Alignment: Tensor FDM, H(div)/H(curl) Basis, Exterior Path

Date: 2026-05-01 | Status: Design Approved

Three independent CPU-side features to close the highest-priority alignment gaps
identified in `libceed_alignment_assessment.md`.

---

## Feature 1: Native Tensor-Product FDM

### Motivation

Current `CpuFdmDenseInverseOperator` assembles the full global Jacobian and inverts
it via Gauss-Jordan (O(n³) build, O(n²) apply). This is limited to `n ≤
FDM_DENSE_MAX_N = 256`. libCEED's native FDM exploits the tensor-product structure
of Lagrange bases on Quad/Hex elements to achieve O(p^{d+1}) apply via 1D
eigendecompositions, scaling to arbitrary mesh sizes.

### Design

New file `crates/cpu/src/fdm_tensor.rs` with a new `OperatorTrait` impl that
performs element-wise fast diagonalization.

**New types:**

```rust
struct Fdm1dEigenData<T> {
    eigenvectors: Vec<T>,   // V: p×p, column-major, M-orthonormal
    mass_evals: Vec<T>,     // λ^M_k (standard eigenvalues of 1D mass matrix)
    stiff_evals: Vec<T>,    // λ^K_k (generalized: K·x = λ·M·x)
}

enum FdmOperatorKind { Mass, Stiffness }

pub struct CpuFdmTensorInverseOperator<T: Scalar> {
    fdm_1d: Fdm1dEigenData<T>,
    dim: usize,              // 1, 2, or 3
    p: usize,
    ncomp: usize,
    num_elem: usize,
    num_local_dof: usize,    // p^dim
    op_kind: FdmOperatorKind,
    restriction: Box<dyn ElemRestrictionTrait<T>>,
    basis: Box<dyn BasisTrait<T>>,
}
```

**Construction flow** (in `CpuOperator::operator_create_fdm_element_inverse`):

1. `n ≤ FDM_DENSE_MAX_N` → existing dense inverse path (unchanged)
2. `n > FDM_DENSE_MAX_N` → new tensor FDM path:
   - Extract 1D interp and grad matrices from `LagrangeBasis`
   - Build `M_1d = B_1d^T · W · B_1d` (1D mass matrix)
   - Build `K_1d = G_1d^T · W · G_1d` (1D stiffness matrix)
   - Solve standard EV for M: `M·v = λ·v` → eigenvectors V, mass_evals Λ_M
   - Project K onto eigenbasis: `λ^K_k = v_k^T · K_1d · v_k` → stiff_evals Λ_K
   - Return `CpuFdmTensorInverseOperator`

**Apply flow** (e.g., 2D mass):

```
for each element e:
  1. Gather: u_local = E_e · u_global
  2. Transform: û = V^T · reshape(u_local, p×p) · V      (O(p³))
  3. Scale: v̂[i,j] = û[i,j] / (λ^M_i · λ^M_j)           (O(p²))
  4. Inverse transform: v_local = V · v̂ · V^T            (O(p³))
  5. Scatter: v_global += E_e^T · v_local
```

For stiffness in 2D: denominator = `λ^K_i + λ^K_j`. In 3D: `λ^K_i + λ^K_j + λ^K_k`.

### Files Changed

| File | Change |
|------|--------|
| `crates/cpu/src/fdm_tensor.rs` | **New**: `Fdm1dEigenData`, `CpuFdmTensorInverseOperator`, helpers |
| `crates/cpu/src/operator.rs` | Route to tensor path when `n > FDM_DENSE_MAX_N` and basis is tensor-product |
| `crates/cpu/src/lib.rs` | Re-export `CpuFdmTensorInverseOperator` |
| `tests/integration.rs` | Numerical comparison: tensor FDM vs dense inverse on small n; large-n smoke test |

### Tests

- Small-n comparison: tensor FDM vs existing dense inverse (< 1e-12)
- Identity: `A * FDM_inv(b) ≈ b` for random b
- Large-n: 2D P4 on 25×25 grid (n=625, verifies tensor path handles what dense can't)
- 1D/2D/3D coverage for both mass and stiffness

---

## Feature 2: H(div)/H(curl) Independent Basis Types

### Motivation

Current `EvalMode::Div` and `Curl` are differential operators applied to H1 vector
fields (Grad + trace/cross-difference). libCEED provides independent
H(curl)-conforming Nédélec elements and H(div)-conforming Raviart-Thomas elements
with distinct DOF layouts (edge DOFs, face DOFs) and basis function spaces.

### Design

**New EvalMode variants** in `crates/core/src/enums.rs`:

```rust
pub enum EvalMode {
    None, Interp, Grad, Div, Curl, Weight,  // existing
    HCurl,  // [new] curl of H(curl) basis: ∇×φ
    HDiv,   // [new] divergence of H(div) basis: ∇·ψ
}
```

Semantic distinction:

| | `Curl` (existing, H1) | `HCurl` (new, Nédélec) |
|---|---|---|
| Basis | LagrangeBasis/SimplexBasis | NedelecBasis |
| Input | H1 nodal DOFs | Edge DOFs |
| Algorithm | Grad + cross-difference | Direct Nédélec curl matrix |
| libCEED | `CEED_EVAL_CURL` on H1 | `CEED_EVAL_CURL` on H(curl) |

| | `Div` (existing, H1) | `HDiv` (new, RT) |
|---|---|---|
| Basis | LagrangeBasis/SimplexBasis | RaviartThomasBasis |
| Input | H1 nodal DOFs | Face/edge DOFs |
| Algorithm | Grad + trace | Direct RT divergence matrix |
| libCEED | `CEED_EVAL_DIV` on H1 | `CEED_EVAL_DIV` on H(div) |

**New basis types:**

`NedelecBasis<T>` in `crates/cpu/src/basis_nedelec.rs`:
```rust
pub struct NedelecBasis<T: Scalar> {
    dim: usize,              // 2 or 3
    p: usize,                // v1: p=1 only
    topology: ElemTopology,  // Triangle or Tet
    num_dof: usize,          // Tri P1: 3, Tet P1: 6
    num_qpoints: usize,
    interp: Vec<T>,          // [nqpts × dim] × num_dof
    curl_matrix: Vec<T>,     // [nqpts × qcomp] × num_dof
    weights: Vec<T>,
    q_ref: Vec<T>,
}
```

`RaviartThomasBasis<T>` in `crates/cpu/src/basis_rt.rs`:
```rust
pub struct RaviartThomasBasis<T: Scalar> {
    dim: usize,              // 2 or 3
    p: usize,                // v1: RT0 (p=0) only
    topology: ElemTopology,  // Triangle or Tet
    num_dof: usize,          // Tri RT0: 3, Tet RT0: 4
    num_qpoints: usize,
    interp: Vec<T>,          // [nqpts × dim] × num_dof
    div_matrix: Vec<T>,      // nqpts × num_dof
    weights: Vec<T>,
    q_ref: Vec<T>,
}
```

**BasisTrait impl dispatch:**

| EvalMode | NedelecBasis | RaviartThomasBasis |
|----------|-------------|-------------------|
| `Interp` | Nédélec interpolation matrix | RT interpolation matrix |
| `HCurl` | Nédélec curl matrix | Err |
| `HDiv` | Err | RT divergence matrix |
| All others | Err | Err |

`num_comp()` returns `dim` for both (vector-valued basis functions).

**v1 coverage:**

| Basis | Topology | Order | DOFs |
|-------|----------|-------|------|
| Nédélec | Triangle | P1 | 3 |
| Nédélec | Tet | P1 | 6 |
| RT | Triangle | RT0 | 3 |
| RT | Tet | RT0 | 4 |

**New Backend trait methods:**

```rust
fn create_basis_hcurl_nedelec(&self, topology, p, q, qmode)
    -> ReedResult<Box<dyn BasisTrait<T>>>;    // default: Err

fn create_basis_hdiv_raviart_thomas(&self, topology, p, q, qmode)
    -> ReedResult<Box<dyn BasisTrait<T>>>;    // default: Err
```

WGPU backend returns `Err` for v1 (CPU only).

### Files Changed

| File | Change |
|------|--------|
| `crates/core/src/enums.rs` | Add `HCurl`, `HDiv` variants |
| `crates/core/src/reed.rs` | Add two new `Backend` factory methods (default `Err`) |
| `crates/cpu/src/basis_nedelec.rs` | **New**: `NedelecBasis<T>` |
| `crates/cpu/src/basis_rt.rs` | **New**: `RaviartThomasBasis<T>` |
| `crates/cpu/src/lib.rs` | Implement new factory methods on `CpuBackend`; re-exports |
| `crates/wgpu/src/lib.rs` | New factory methods return `Err` |
| `src/lib.rs` | `Reed<T>` convenience methods `basis_hcurl_nedelec` / `basis_hdiv_rt` |
| `tests/integration.rs` | New tests |

### Tests

- DOF count verification for each topology/order
- Constant field: Nédélec `HCurl` returns zero; RT `HDiv` returns zero
- Interpolation projection identity on linear fields
- Cross-comparison: H1 `Curl` vs Nédélec `HCurl` on same mesh (physical agreement, different spaces)

---

## Feature 3: Exterior (Boundary Face) Path

### Motivation

Current `QFunctionCategory::Exterior` is metadata only — the execution path is
identical to interior (volume element loop). libCEED's exterior path requires
face-to-element restriction, face quadrature, exterior gallery kernels, and
boundary-face iteration in the operator.

### Design

**Core principle:** No new trait or operator type. `CpuFaceElemRestriction`
implements `ElemRestrictionTrait` where `num_elements()` returns the number of
boundary faces. The existing `CpuOperator` branches on
`qfunction.q_function_category()` to use face iteration.

**`CpuFaceElemRestriction<T>`** in `crates/cpu/src/elem_restriction_face.rs`:

```rust
pub struct CpuFaceElemRestriction<T: Scalar> {
    num_faces: usize,
    num_dof_per_face: usize,
    num_dof_per_elem: usize,
    num_comp: usize,
    num_global_dof: usize,
    face_to_elem: Vec<(usize, usize)>,  // (elem_id, local_face_number)
    face_offsets: Vec<CeedInt>,         // face L-vector
    elem_offsets: Vec<CeedInt>,         // parent element L-vector
}
```

Implements `ElemRestrictionTrait<T>`:
- `num_elements()` → `num_faces`
- `apply(NoTranspose)`: gather global → face-local
- `apply(Transpose)`: scatter face-local → element-local → global (additive)
- `assembled_csr_pattern()` → `Err`

**Face quadrature helpers** in `crates/cpu/src/face_quadrature.rs`:

```rust
pub fn face_quadrature_from_tensor_basis<T: Scalar>(
    basis: &LagrangeBasis<T>, face_number: usize,
) -> ReedResult<(Vec<T>, Vec<T>)>;  // (q_ref_face, weights_face)

pub fn face_quadrature_from_simplex_basis<T: Scalar>(
    basis: &SimplexBasis<T>, face_number: usize,
) -> ReedResult<(Vec<T>, Vec<T>)>;
```

v1 approach: face quadrature weights passed via `QFunctionContext` (no
`BasisTrait` modification). Future: optional `face_q_weights`/`face_q_ref`
methods on `BasisTrait`.

**Exterior gallery** in `crates/cpu/src/gallery/boundary.rs` (new):

```rust
pub struct NeumannApply;   // ∫_∂Ω g_N · v ds
pub struct RobinApply;     // ∫_∂Ω α u · v ds
```

Name registry in `crates/cpu/src/lib.rs`:
```rust
pub static QFUNCTION_EXTERIOR_GALLERY_NAMES: &[&str] = &[
    "NeumannApply",
    "RobinApply",
];
```

**Operator face iteration** in `CpuOperator::execute_inner`:

```
if qfunction.category() == Exterior:
  for each face f in 0..num_faces:
    (elem_id, local_face) = face_to_elem[f]
    restrict global → element-local (via elem restriction)
    filter element-local → face-local DOFs
    evaluate basis at face quadrature points
    call qfunction.apply() at face q-points
    scatter face-local → element-local → global (additive)
else:
  existing volume element loop
```

### Files Changed

| File | Change |
|------|--------|
| `crates/cpu/src/elem_restriction_face.rs` | **New**: `CpuFaceElemRestriction<T>` |
| `crates/cpu/src/face_quadrature.rs` | **New**: face quadrature extraction helpers |
| `crates/cpu/src/gallery/boundary.rs` | **New**: `NeumannApply`, `RobinApply` |
| `crates/cpu/src/lib.rs` | Exterior gallery names, name resolution, re-exports |
| `crates/cpu/src/operator.rs` | Exterior face loop branch |
| `crates/core/src/reed.rs` | `Backend` trait: `create_face_elem_restriction` (default `Err`) |
| `src/lib.rs` | `Reed<T>`: `face_elem_restriction`, `q_function_by_name_exterior` |
| `tests/integration.rs` | New tests |

### Tests

- Face restriction gather/scatter round-trip
- Face quadrature extraction vs 1D reference rule
- Neumann BC on 2D Poisson: interior + exterior operator solution vs known reference
- Exterior gallery name resolution and category reporting
- Mixed interior/exterior composite operator

---

## Implementation Order

1. **Tensor FDM** — smallest scope, existing infrastructure is closest
2. **H(div)/H(curl) basis** — independent types, no operator changes needed
3. **Exterior path** — most architectural, depends on operator iteration changes

---

## Revision History

| Date | Description |
|------|-------------|
| 2026-05-01 | Initial design: three features approved (FDM: both paths; Basis: new EvalMode; Exterior: full libCEED alignment) |
