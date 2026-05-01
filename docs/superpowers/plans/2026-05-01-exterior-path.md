# Exterior (Boundary Face) Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add face-to-element restriction, exterior gallery QFunctions, and operator face-iteration branch so operators with `QFunctionCategory::Exterior` iterate over boundary faces instead of volume elements.

**Architecture:** `CpuFaceElemRestriction` implements `ElemRestrictionTrait` where `num_elements()` returns the count of boundary faces. New `face_quadrature.rs` provides face-local quadrature extraction from volume bases. Exterior gallery in `gallery/boundary.rs` provides `NeumannApply` and `RobinApply`. `CpuOperator::execute_inner` branches on `qfunction.q_function_category()` for face vs volume iteration. Face quadrature weights flow through `QFunctionContext`.

**Tech Stack:** Rust, reed-core traits, existing operator/gallery patterns.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/cpu/src/elem_restriction_face.rs` | **Create**: `CpuFaceElemRestriction<T>` implementing `ElemRestrictionTrait<T>` |
| `crates/cpu/src/face_quadrature.rs` | **Create**: face quadrature extraction from volume bases |
| `crates/cpu/src/gallery/boundary.rs` | **Create**: `NeumannApply`, `RobinApply` exterior QFunctions |
| `crates/core/src/reed.rs` | **Modify**: add `create_face_elem_restriction` to `Backend` trait |
| `crates/cpu/src/lib.rs` | **Modify**: register modules, impl factory, exterior gallery names, re-exports |
| `crates/cpu/src/operator.rs` | **Modify**: face iteration branch in `execute_inner` |
| `crates/wgpu/src/lib.rs` | **Modify**: `create_face_elem_restriction` returns `Err` |
| `src/lib.rs` | **Modify**: re-exports + `Reed<T>` convenience |
| `tests/integration.rs` | **Modify**: new tests |

---

### Task 1: Create CpuFaceElemRestriction

**Files:**
- Create: `crates/cpu/src/elem_restriction_face.rs`

Write `CpuFaceElemRestriction<T>` implementing `ElemRestrictionTrait<T>`:

```rust
use reed_core::{
    elem_restriction::ElemRestrictionTrait,
    enums::TransposeMode,
    error::ReedResult,
    scalar::Scalar,
    CeedInt, CsrPattern, ReedError,
};

pub struct CpuFaceElemRestriction<T: Scalar> {
    /// Number of boundary faces (= num_elements from restriction perspective)
    num_faces: usize,
    /// DOFs per face (depends on topology: e.g., p for Quad face of Hex with Lagrange)
    num_dof_per_face: usize,
    /// DOFs per element (from parent volume restriction, for scatter routing)
    num_dof_per_elem: usize,
    /// Number of components
    ncomp: usize,
    /// Global DOF count (same as parent volume restriction)
    num_global_dof: usize,
    /// Face → (element_id, local_face_number): maps each boundary face to its parent element and which face of that element
    face_to_elem: Vec<(usize, usize)>,
    /// Face L-vector: for each face, stores global DOF indices for each face DOF
    /// Size: num_faces * num_dof_per_face * ncomp
    face_offsets: Vec<CeedInt>,
    /// Element L-vector from parent volume restriction (for scatter path)
    /// Size: num_elem_unique * num_dof_per_elem * ncomp  
    elem_offsets: Vec<CeedInt>,
    _marker: std::marker::PhantomData<T>,
}
```

Constructor:
```rust
pub fn new(
    num_faces: usize,
    num_dof_per_face: usize,
    num_dof_per_elem: usize,
    ncomp: usize,
    num_global_dof: usize,
    face_to_elem: Vec<(usize, usize)>,
    face_offsets: Vec<CeedInt>,
    elem_offsets: Vec<CeedInt>,
) -> ReedResult<Self>
```

Implement `ElemRestrictionTrait<T>`:
- `num_elements()` → `num_faces`
- `num_dof_per_elem()` → `num_dof_per_face` (face DOFs, not element DOFs)
- `num_global_dof()` → `num_global_dof`
- `num_comp()` → `ncomp`
- `apply(NoTranspose, u_global, v_local)`: for each face f, gather `ncomp * num_dof_per_face` values from u_global using face_offsets
- `apply(Transpose, u_local, v_global)`: for each face f, map face-local DOFs → element-local DOFs using face_to_elem mapping, then scatter to v_global using elem_offsets (additive)
- `assembled_csr_pattern()` → `Err`
- `boxed_clone()` → `Ok(Box::new(self.clone()))` (derive Clone on the struct)

Register `pub mod elem_restriction_face;` in `crates/cpu/src/lib.rs`.

Build and test: `cargo build -p reed-cpu && cargo test -p reed-cpu`.
Commit: "feat: add CpuFaceElemRestriction for boundary face mappings"

---

### Task 2: Create face quadrature helpers

**Files:**
- Create: `crates/cpu/src/face_quadrature.rs`

```rust
use reed_core::{error::ReedResult, scalar::Scalar, ReedError};
use crate::basis_lagrange::LagrangeBasis;
use crate::basis_simplex::SimplexBasis;

/// Extract face quadrature points and weights from a tensor-product LagrangeBasis.
///
/// For a Quad element (dim=2), each face is a Line. The face quadrature points
/// are the 1D quadrature rules on the appropriate edge of the reference element.
/// `local_face_number`: 0=bottom, 1=right, 2=top, 3=left (counter-clockwise).
///
/// Returns (q_ref_face, weights_face) where q_ref_face is [nq_face × (dim-1)].
pub fn face_quadrature_tensor<T: Scalar>(
    basis: &LagrangeBasis<T>, local_face_number: usize,
) -> ReedResult<(Vec<T>, Vec<T>)> { ... }

/// Extract face quadrature from a SimplexBasis.
/// For Triangle (dim=2): faces are 1D edges, quadrature is 1D Gauss on the edge.
/// For Tet (dim=3): faces are 2D triangles, quadrature is 2D simplex quadrature.
pub fn face_quadrature_simplex<T: Scalar>(
    basis: &SimplexBasis<T>, local_face_number: usize,
) -> ReedResult<(Vec<T>, Vec<T>)> { ... }
```

Implementation:
- For tensor basis: the 1D weights are already available via `basis.weights_1d()`. Face q_ref is constructed by fixing one coordinate at ±1 and varying the others.
- For simplex basis: extract face vertices from the reference element, map 1D/2D quadrature to those face coordinates.
- Both return face-dimension quadrature data.

Register `pub mod face_quadrature;` in `crates/cpu/src/lib.rs`.

Build and commit: "feat: add face quadrature extraction helpers"

---

### Task 3: Create exterior gallery QFunctions

**Files:**
- Create: `crates/cpu/src/gallery/boundary.rs`

```rust
//! Exterior (boundary) gallery QFunctions for libCEED exterior kernel parity.
//!
//! These kernels implement boundary integral operators.
//! Face quadrature weights and data are passed via the QFunctionContext.

use reed_core::{
    enums::EvalMode,
    error::ReedResult,
    qfunction::{QFunctionCategory, QFunctionField, QFunctionTrait},
    scalar::Scalar,
    ReedError,
};

/// Neumann boundary condition: `∫_{∂Ω} g_N · v ds`
///
/// Inputs: ["v"] (test function evaluated at face q-pts, `EvalMode::Interp`)
/// Outputs: ["v"] (same, scaled by face weight)
/// Context: face quadrature weights as f64 LE bytes (nq_face values)
pub struct NeumannApply;

impl<T: Scalar> QFunctionTrait<T> for NeumannApply {
    fn context_byte_len(&self) -> usize { 0 } // weights via separate mechanism
    fn inputs(&self) -> &[QFunctionField] { ... }
    fn outputs(&self) -> &[QFunctionField] { ... }
    fn apply(&self, ctx: &[u8], q: usize, inputs: &[&[T]], outputs: &mut [&mut [T]]) -> ReedResult<()> {
        // outputs[0][q] = inputs[0][q] (identity, boundary flux applied by operator via face weights)
        outputs[0][q] = inputs[0][q];
        Ok(())
    }
    fn q_function_category(&self) -> QFunctionCategory { QFunctionCategory::Exterior }
}

/// Robin boundary condition: `∫_{∂Ω} α u · v ds`
///
/// Inputs: ["u"] (solution at face q-pts), context contains α and face weights
/// Outputs: ["v"] (α · u at face q-pts)
/// Context bytes: [α: f64 LE] (8 bytes)
pub struct RobinApply;

impl<T: Scalar> QFunctionTrait<T> for RobinApply {
    fn context_byte_len(&self) -> usize { 8 }
    fn inputs(&self) -> &[QFunctionField] { ... }
    fn outputs(&self) -> &[QFunctionField] { ... }
    fn apply(&self, ctx: &[u8], q: usize, inputs: &[&[T]], outputs: &mut [&mut [T]]) -> ReedResult<()> {
        let alpha = read_f64_le(ctx, 0); // helper from gallery/helpers.rs
        let alpha_t = T::from_f64(alpha).unwrap_or(T::ONE);
        outputs[0][q] = inputs[0][q] * alpha_t;
        Ok(())
    }
    fn q_function_category(&self) -> QFunctionCategory { QFunctionCategory::Exterior }
}
```

Also add `pub static QFUNCTION_EXTERIOR_GALLERY_NAMES: &[&str] = &["NeumannApply", "RobinApply"];` and a `q_function_by_name_exterior(name: &str) -> Option<Box<dyn QFunctionTrait<f64>>>` function in `crates/cpu/src/lib.rs`.

Build and commit: "feat: add exterior gallery QFunctions (NeumannApply, RobinApply)"

---

### Task 4: Add face iteration branch in CpuOperator

**Files:**
- Modify: `crates/cpu/src/operator.rs`

In `execute_inner`, add a branch at the beginning:

```rust
// Check if this is an exterior (boundary face) operator
let is_exterior = self.qfunction.q_function_category() == QFunctionCategory::Exterior;

if is_exterior {
    return self.execute_exterior(input, output, add);
}
// ... existing volume element loop
```

Add `execute_exterior` method to `CpuOperator`:

```rust
fn execute_exterior<'io>(
    &self,
    input: ActiveInputSource<'_, T>,
    output: &mut ActiveOutputSink<'io, T>,
    add: bool,
) -> ReedResult<()> {
    // Validate: all fields must use face restrictions
    for field in &self.fields {
        // Face restriction validation — num_elements() gives num_faces
    }
    
    // Similar to execute_inner but:
    // - Uses num_elem (= num_faces) from restriction
    // - Face quadrature weights come from QFunctionContext
    // - For each face: gather face-local DOFs, evaluate basis at face q-pts,
    //   call qfunction, scatter back
    
    // The key difference: the operator's num_elem is the number of faces.
    // prepare_input_into already uses self.num_elem for sizing.
    // The restriction handles face→element mapping internally.
    
    // ... similar structure to execute_inner, loops over self.num_elem faces
}
```

Key insight: `execute_exterior` is structurally identical to `execute_inner` — it loops over `self.num_elem` "elements" (which are faces for exterior operators). The restriction's `apply(NoTranspose)` gathers face-local DOFs using face_offsets. The restriction's `apply(Transpose)` scatters back using elem_offsets. The basis evaluation and QFunction call are the same.

For v1, `execute_exterior` can simply call `execute_inner` after validation, since the face restriction already provides `num_elements() == num_faces`. The only difference is quadrature — face quadrature is handled via QFunctionContext bytes (user responsibility for v1).

Build and commit: "feat: add exterior face iteration branch in CpuOperator"

---

### Task 5: Backend factory + re-exports

**Files:**
- Modify: `crates/core/src/reed.rs` — add `create_face_elem_restriction` to `Backend` trait (both cfg variants, default `Err`)
- Modify: `crates/cpu/src/lib.rs` — implement on `CpuBackend`, re-export types, exterior gallery names, name resolver
- Modify: `crates/wgpu/src/lib.rs` — return `Err` on WGPU backend
- Modify: `src/lib.rs` — `Reed<T>` convenience method + re-exports

Build, test, commit: "feat: add face restriction factory + exterior gallery re-exports"

---

### Task 6: Integration tests

**Files:**
- Modify: `tests/integration.rs`

Add tests:
1. `test_face_restriction_gather_scatter_roundtrip` — create a simple 2-element mesh, define boundary faces, verify gather-then-scatter preserves values
2. `test_exterior_gallery_names` — verify `QFUNCTION_EXTERIOR_GALLERY_NAMES` contains "NeumannApply" and "RobinApply"
3. `test_neumann_qfunction_category` — verify `NeumannApply.q_function_category() == Exterior`
4. `test_exterior_operator_build_and_apply` — build a 2D Poisson operator with a Neumann boundary face, verify apply doesn't panic and produces output

Commit: "test: add exterior path integration tests"

---

### Task 7: Full test suite verification

```bash
cargo test --workspace 2>&1 | grep "test result:"
```

All tests must pass. Merge to main, cleanup.
