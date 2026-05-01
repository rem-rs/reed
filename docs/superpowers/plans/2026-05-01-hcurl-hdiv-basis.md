# H(div)/H(curl) Independent Basis Types — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Nédélec H(curl) and Raviart-Thomas H(div) basis types with new `EvalMode::HCurl`/`HDiv` variants, CPU-only, covering Triangle/Tet P1 Nédélec and Triangle/Tet RT0.

**Architecture:** Two new `BasisTrait` impls (`NedelecBasis`, `RaviartThomasBasis`), new `EvalMode` variants with operator qcomp routing, new `Backend` factory methods (default `Err`). `num_comp()` returns `dim` (vector-valued bases); DOF input uses redundant per-component replication consistent with existing operator sizing.

**Tech Stack:** Rust, reed-core traits, barycentric coordinates on simplex reference elements.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/core/src/enums.rs` | **Modify**: add `HCurl`, `HDiv` to `EvalMode` |
| `crates/core/src/reed.rs` | **Modify**: add 2 new `Backend` factory methods + `Reed<T>` convenience methods (both cfg variants) |
| `crates/cpu/src/basis_nedelec.rs` | **Create**: `NedelecBasis<T>` — Tri P1 + Tet P1 |
| `crates/cpu/src/basis_rt.rs` | **Create**: `RaviartThomasBasis<T>` — Tri RT0 + Tet RT0 |
| `crates/cpu/src/operator.rs` | **Modify**: add `HCurl`/`HDiv` arms in `qcomp_size_for` |
| `crates/cpu/src/lib.rs` | **Modify**: register modules, impl factory methods, re-export |
| `crates/wgpu/src/lib.rs` | **Modify**: new factory methods return `Err` |
| `src/lib.rs` | **Modify**: convenience methods on `Reed<T>` |
| `tests/integration.rs` | **Modify**: new tests |

---

### Task 1: Add HCurl and HDiv to EvalMode

**Files:**
- Modify: `crates/core/src/enums.rs`

- [ ] **Step 1: Add variants**

In the `EvalMode` enum (after `Weight`), add:

```rust
    /// Curl of H(curl) basis (Nédélec): ∇×φ. 2D→scalar, 3D→3-vector at each q-pt.
    HCurl,
    /// Divergence of H(div) basis (Raviart-Thomas): ∇·ψ. Scalar at each q-pt.
    HDiv,
```

- [ ] **Step 2: Fix exhaustive matches across workspace**

Run `cargo build --workspace 2>&1 | grep "pattern.*not covered"`. For each match error, add:
```rust
EvalMode::HCurl | EvalMode::HDiv => Err(ReedError::Basis(
    "HCurl/HDiv not supported on this basis type".into(),
))
```

In `crates/cpu/src/basis_lagrange.rs`, `crates/cpu/src/basis_simplex.rs`, `crates/wgpu/src/basis.rs`, `crates/wgpu/src/basis_simplex.rs`.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: add HCurl and HDiv to EvalMode, fix exhaustive matches"
```

---

### Task 2: Add Backend factory methods + Reed convenience methods

**Files:**
- Modify: `crates/core/src/reed.rs`

- [ ] **Step 1: Add trait methods (both cfg variants)**

After `create_basis_h1_simplex`, add to **both** the non-wasm32 and wasm32 `Backend` trait definitions:

```rust
    fn create_basis_hcurl_nedelec(
        &self, _topology: crate::enums::ElemTopology, _p: usize, _q: usize,
        _qmode: crate::enums::QuadMode,
    ) -> ReedResult<Box<dyn crate::basis::BasisTrait<T>>> {
        Err(ReedError::BackendNotSupported(
            "create_basis_hcurl_nedelec is not implemented for this backend".into(),
        ))
    }

    fn create_basis_hdiv_raviart_thomas(
        &self, _topology: crate::enums::ElemTopology, _p: usize, _q: usize,
        _qmode: crate::enums::QuadMode,
    ) -> ReedResult<Box<dyn crate::basis::BasisTrait<T>>> {
        Err(ReedError::BackendNotSupported(
            "create_basis_hdiv_raviart_thomas is not implemented for this backend".into(),
        ))
    }
```

- [ ] **Step 2: Add Reed<T> convenience methods**

In the `impl<T: Scalar> Reed<T>` block (after `basis_h1_simplex`):

```rust
    pub fn basis_hcurl_nedelec(&self, topo: crate::enums::ElemTopology, p: usize, q: usize,
        qmode: crate::enums::QuadMode,
    ) -> ReedResult<Box<dyn crate::basis::BasisTrait<T>>> {
        (**self.backend.lock().unwrap()).create_basis_hcurl_nedelec(topo, p, q, qmode)
    }

    pub fn basis_hdiv_raviart_thomas(&self, topo: crate::enums::ElemTopology, p: usize, q: usize,
        qmode: crate::enums::QuadMode,
    ) -> ReedResult<Box<dyn crate::basis::BasisTrait<T>>> {
        (**self.backend.lock().unwrap()).create_basis_hdiv_raviart_thomas(topo, p, q, qmode)
    }
```

- [ ] **Step 3: Build and commit**

```bash
cargo build -p reed-core && git add crates/core/src/reed.rs && git commit -m "feat: add H(curl)/H(div) factory methods to Backend trait and Reed"
```

---

### Task 3: Add HCurl/HDiv arms in operator qcomp_size_for

**Files:**
- Modify: `crates/cpu/src/operator.rs`

- [ ] **Step 1: Add match arms**

In `qcomp_size_for`, add after the `EvalMode::Curl` arm:

```rust
            EvalMode::HCurl => {
                let basis = field.basis.ok_or_else(|| {
                    ReedError::Operator(format!("field '{}' requires basis for HCurl", field.name))
                })?;
                match basis.dim() {
                    2 => Ok(1),
                    3 => Ok(3),
                    d => Err(ReedError::Operator(format!(
                        "field '{}': HCurl requires dim=2 or 3, got {}", field.name, d
                    ))),
                }
            }
            EvalMode::HDiv => Ok(1),
```

- [ ] **Step 2: Build and commit**

```bash
cargo build -p reed-cpu && git add crates/cpu/src/operator.rs && git commit -m "feat: add HCurl/HDiv qcomp routing in CpuOperator"
```

---

### Task 4: Create NedelecBasis

**Files:**
- Create: `crates/cpu/src/basis_nedelec.rs`

Write the complete file with:

- `NedelecBasis<T>` struct: dim, num_dof, num_qpoints, weights, q_ref, interp (nqpts × num_dof × dim), curl_matrix (nqpts × num_dof for 2D, nqpts × num_dof × 3 for 3D)
- `new(topo, p, q)` constructor: validates p=1, topo=Triangle/Tet, builds interp and curl matrices at quad points
- `nedelec_p1_tri_basis(x, y)`: returns (interp_vals[6], curl_vals[3]) using φ_{ij} = λ_i∇λ_j − λ_j∇λ_i
- `nedelec_p1_tet_basis(x, y, z)`: returns (interp_vals[18], curl_vals[18]) for 6 edges in 3D
- `BasisTrait<T>` impl: dim()=dim, num_dof()=edges, num_qpoints()=q, num_comp()=dim
- apply(): Interp → vector-valued interpolation; HCurl → curl evaluation; all other modes → Err

**Key formulas:**
- Triangle barycentrics: λ₀=1-x-y, λ₁=x, λ₂=y. Gradients: [-1,-1], [1,0], [0,1]
- Tet barycentrics: λ₀=1-x-y-z, λ₁=x, λ₂=y, λ₃=z
- φ_{ij} = λ_i∇λ_j − λ_j∇λ_i (vector-valued)
- curl(φ_{ij}) = 2(∇λ_i × ∇λ_j) (constant; scalar in 2D, vector in 3D)

**Buffer sizing:**
- Forward Interp: u = num_elem * num_dof * dim, v = num_elem * num_qpoints * dim
- Forward HCurl (2D): u = num_elem * num_dof * dim, v = num_elem * num_qpoints * 1
- Forward HCurl (3D): u = num_elem * num_dof * dim, v = num_elem * num_qpoints * 3
- For input u, each DOF has dim entries; the basis reads the first component (u[dof * dim + 0]) as the DOF scalar value

After writing, run `cargo build -p reed-cpu && cargo test -p reed-cpu` and commit.

---

### Task 5: Create RaviartThomasBasis

**Files:**
- Create: `crates/cpu/src/basis_rt.rs`

Write the complete file with:

- `RaviartThomasBasis<T>` struct: dim, num_dof, num_qpoints, weights, q_ref, interp (nqpts × num_dof × dim), div_matrix (nqpts × num_dof)
- `new(topo, p, q)`: validates p=0 (RT0), topo=Triangle/Tet
- `rt0_tri_basis(x, y)`: RT0 on triangle (3 DOFs = edge normals). ψ_i = a_i (x - x_i) where x_i is the opposite vertex
- `rt0_tet_basis(x, y, z)`: RT0 on tet (4 DOFs = face normals). ψ_i = a_i (x - x_i)
- `BasisTrait<T>` impl: dim()=dim, num_dof()=3(Tri)/4(Tet), num_qpoints()=q, num_comp()=dim
- apply(): Interp → vector-valued; HDiv → scalar divergence; all others → Err

**Key formulas (RT0 Triangle):**
- ψ_{edge_i} = (x - x_i) / (2|T|) where x_i is vertex i (the vertex opposite edge i), |T| is triangle area = 1/2 for reference triangle
- Actually, simpler: on reference triangle (0,0),(1,0),(0,1): ψ₀ = (x, y-1), ψ₁ = (x-1, y), ψ₂ = (x, y)
- Normalization: each ψ_i has flux 1 through edge i, 0 through other edges
- ∇·ψ = constant on element (2/|T| for all 3 basis functions on reference triangle)

**Key formulas (RT0 Tet):**
- ψ_{face_i} = (x - x_i) / (3|T|) where x_i is vertex i opposite face i, |T| = 1/6 for reference tet

After writing, run `cargo build -p reed-cpu && cargo test -p reed-cpu` and commit.

---

### Task 6: Implement Backend methods on CpuBackend + WgpuBackend

**Files:**
- Modify: `crates/cpu/src/lib.rs`
- Modify: `crates/wgpu/src/lib.rs`

- [ ] **Step 1: CpuBackend impl**

In `crates/cpu/src/lib.rs`, add to `impl<T: Scalar> Backend<T> for CpuBackend<T>`:

```rust
    fn create_basis_hcurl_nedelec(
        &self, topology: ElemTopology, p: usize, q: usize, _qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        Ok(Box::new(NedelecBasis::new(topology, p, q)?))
    }

    fn create_basis_hdiv_raviart_thomas(
        &self, topology: ElemTopology, p: usize, q: usize, _qmode: QuadMode,
    ) -> ReedResult<Box<dyn BasisTrait<T>>> {
        Ok(Box::new(RaviartThomasBasis::new(topology, p, q)?))
    }
```

Also add module declarations (`pub mod basis_nedelec; pub mod basis_rt;`) and re-exports.

- [ ] **Step 2: WgpuBackend impl**

In `crates/wgpu/src/lib.rs`, add to both non-wasm32 and wasm32 `Backend` impls:

```rust
    fn create_basis_hcurl_nedelec(&self, ...) -> ... {
        Err(ReedError::BackendNotSupported("Nedelec basis not yet on WGPU".into()))
    }
    fn create_basis_hdiv_raviart_thomas(&self, ...) -> ... {
        Err(ReedError::BackendNotSupported("RT basis not yet on WGPU".into()))
    }
```

- [ ] **Step 3: Top-level re-exports**

In `src/lib.rs`, add `NedelecBasis` and `RaviartThomasBasis` to the `pub use reed_cpu::{}` block. Add `HCurl` and `HDiv` to the `EvalMode` re-exports.

- [ ] **Step 4: Build and commit**

```bash
cargo build -p reed && cargo test -p reed-cpu && git add -A && git commit -m "feat: implement H(curl)/H(div) factory methods on CPU/WGPU backends"
```

---

### Task 7: Integration tests

**Files:**
- Modify: `tests/integration.rs`

Add these tests:

1. **`test_nedelec_tri_p1_constant_curl`**: For constant input DOFs, verify HCurl is constant (equal to 2*value per element for a specific orientation). Verify Interp produces expected vector values at vertices.

2. **`test_nedelec_tet_p1_dof_count`**: Verify dim=3, num_dof=6, num_qpoints as requested.

3. **`test_rt_tri_rt0_constant_div`**: For constant input DOFs, verify HDiv output is constant per element.

4. **`test_rt_tet_rt0_dof_count`**: Verify dim=3, num_dof=4.

5. **`test_nedelec_interp_values_at_quadrature_points`**: Manual check: at a specific q-pt, compute φ_{ij} values and compare with known formula.

6. **`test_backend_factory_creates_basis`**: Verify `reed.basis_hcurl_nedelec(Triangle, 1, 3, Gauss)` and `reed.basis_hdiv_raviart_thomas(Triangle, 0, 3, Gauss)` succeed. Verify unsupported topologies fail.

Run `cargo test -p reed -- <test_names>` then `cargo test --workspace` to verify no regressions. Commit.

---

### Task 8: Full test suite verification

```bash
cargo test --workspace 2>&1 | grep "test result:"
```

All tests must pass. If any pre-existing test fails due to new EvalMode variants, fix the match arm and re-commit.
