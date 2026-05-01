# Tensor-Product FDM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement native tensor-product FDM for `CpuOperator`, providing `A^{-1}` apply via 1D eigendecomposition on Quad/Hex Lagrange bases.

**Architecture:** `BasisTrait` gains optional `tensor_fdm_1d_data()`. `ElemRestrictionTrait` gains `boxed_clone()`. New `fdm_tensor.rs` module with `CpuFdmTensorInverseOperator` implementing `OperatorTrait`. `CpuOperator::operator_create_fdm_element_inverse` routes to tensor path when basis provides 1D FDM data.

**Tech Stack:** Rust, reed-core traits, reed-cpu basis_lagrange, Jacobi EV solver (no new deps).

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/core/src/basis.rs` | **Modify**: add `tensor_fdm_1d_data()` to `BasisTrait` |
| `crates/core/src/elem_restriction.rs` | **Modify**: add `boxed_clone()` to `ElemRestrictionTrait` |
| `crates/cpu/src/basis_lagrange.rs` | **Modify**: impl `tensor_fdm_1d_data()` |
| `crates/cpu/src/elem_restriction.rs` | **Modify**: impl `boxed_clone()` on `CpuElemRestriction` |
| `crates/cpu/src/fdm_tensor.rs` | **Create**: `CpuFdmTensorInverseOperator<T>`, 1D matrix builders, Jacobi EV solver |
| `crates/cpu/src/operator.rs` | **Modify**: route `operator_create_fdm_element_inverse` to tensor path |
| `crates/cpu/src/lib.rs` | **Modify**: register `fdm_tensor` module, re-export |
| `src/lib.rs` | **Modify**: re-export `CpuFdmTensorInverseOperator` |
| `tests/integration.rs` | **Modify**: new tensor FDM tests |

---

### Task 1: Add tensor_fdm_1d_data() to BasisTrait

**Files:**
- Modify: `crates/core/src/basis.rs`

- [ ] **Step 1: Add the optional method**

In both cfg variants of `BasisTrait` (non-wasm32 after `fn q_ref` on line 40, wasm32 after `fn q_ref` on line 59), add:

```rust
    /// Tensor-product FDM: if this basis supports fast diagonalization, return
    /// `(interp_1d, grad_1d, weights_1d, p, q)` where each slice is 1D data.
    /// Default `None`.
    fn tensor_fdm_1d_data(&self) -> Option<(&[T], &[T], &[T], usize, usize)> {
        None
    }
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p reed-core 2>&1 | tail -5`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/basis.rs
git commit -m "feat: add tensor_fdm_1d_data() optional method to BasisTrait"
```

---

### Task 2: Add boxed_clone() to ElemRestrictionTrait

**Files:**
- Modify: `crates/core/src/elem_restriction.rs`

- [ ] **Step 1: Add the method**

In both cfg variants of `ElemRestrictionTrait`, add after `assembled_csr_pattern()`:

```rust
    /// Clone this restriction into a boxed trait object.
    /// Implementors that support cloning should override this.
    fn boxed_clone(&self) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Err(ReedError::ElemRestriction(
            "boxed_clone is not implemented for this restriction type".into(),
        ))
    }
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p reed-core 2>&1 | tail -5`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/elem_restriction.rs
git commit -m "feat: add boxed_clone() to ElemRestrictionTrait"
```

---

### Task 3: Implement tensor_fdm_1d_data() on LagrangeBasis

**Files:**
- Modify: `crates/cpu/src/basis_lagrange.rs`

- [ ] **Step 1: Add the impl**

In the `impl<T: Scalar> BasisTrait<T> for LagrangeBasis<T>` block (after `fn q_ref` around line 398):

```rust
    fn tensor_fdm_1d_data(&self) -> Option<(&[T], &[T], &[T], usize, usize)> {
        Some((&self.interp, &self.grad, &self.weights[..self.q], self.p, self.q))
    }
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p reed-cpu 2>&1 | tail -5`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/cpu/src/basis_lagrange.rs
git commit -m "feat: implement tensor_fdm_1d_data() on LagrangeBasis"
```

---

### Task 4: Implement boxed_clone() on CpuElemRestriction

**Files:**
- Modify: `crates/cpu/src/elem_restriction.rs`

- [ ] **Step 1: Read CpuElemRestriction struct to understand fields**

Run: `grep -n 'pub struct CpuElemRestriction\|struct CpuElemRestriction\|offsets\|strides\|compstride\|ncomp\|nelem\|elemsize\|ng' crates/cpu/src/elem_restriction.rs | head -20`

- [ ] **Step 2: Add the impl**

In `crates/cpu/src/elem_restriction.rs`, add to the `impl<T: Scalar> ElemRestrictionTrait<T> for CpuElemRestriction<T>` block:

```rust
    fn boxed_clone(&self) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Ok(Box::new(self.clone()))
    }
```

Important: `CpuElemRestriction<T>` must implement `Clone`. Check if it already derives `Clone`; if not, add `#[derive(Clone)]` or a manual impl. If the struct contains `Vec<CeedInt>` fields (which it likely does), derive should work.

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p reed-cpu 2>&1 | tail -5`
Expected: Compiles.

- [ ] **Step 4: Commit**

```bash
git add crates/cpu/src/elem_restriction.rs
git commit -m "feat: implement boxed_clone() on CpuElemRestriction"
```

---

### Task 5: Create fdm_tensor.rs — complete module

**Files:**
- Create: `crates/cpu/src/fdm_tensor.rs`

- [ ] **Step 1: Write the complete module file**

Write `crates/cpu/src/fdm_tensor.rs`:

```rust
//! Tensor-product Fast Diagonalization Method (FDM) element inverse.
//!
//! libCEED-aligned `CeedOperatorCreateFDMElementInverse`: for tensor-product H1 Lagrange
//! bases on Quad/Hex elements, the local Jacobian is separable. A 1D eigen-decomposition
//! diagonalizes the element operator, yielding O(p^{d+1}) apply instead of O(p^{3d}).
//!
//! [`CpuFdmTensorInverseOperator`] implements [`OperatorTrait`] and is created by
//! [`CpuOperator::operator_create_fdm_element_inverse`](crate::operator::CpuOperator::operator_create_fdm_element_inverse).

use reed_core::{
    basis::BasisTrait,
    elem_restriction::ElemRestrictionTrait,
    enums::TransposeMode,
    error::{ReedError, ReedResult},
    operator::{OperatorAssembleKind, OperatorTrait, OperatorTransposeRequest},
    scalar::Scalar,
    vector::VectorTrait,
};

// ── 1D eigen-data ──────────────────────────────────────────────────

struct Fdm1dEigenData<T: Scalar> {
    eigenvectors: Vec<T>,  // p×p column-major, M-orthonormal
    mass_evals: Vec<T>,    // λ^M_k, sorted ascending
    stiff_evals: Vec<T>,   // λ^K_k (Rayleigh quotient)
}

/// Which operator the FDM inverse targets.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FdmOperatorKind {
    Mass,
    Stiffness,
}

// ── 1D matrix builders ─────────────────────────────────────────────

fn build_mass_1d<T: Scalar>(b: &[T], w: &[T], p: usize, q: usize) -> Vec<T> {
    let mut m = vec![T::ZERO; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut s = T::ZERO;
            for qi in 0..q { s += w[qi] * b[qi * p + i] * b[qi * p + j]; }
            m[i + j * p] = s;
        }
    }
    m
}

fn build_stiffness_1d<T: Scalar>(g: &[T], w: &[T], p: usize, q: usize) -> Vec<T> {
    let mut k = vec![T::ZERO; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut s = T::ZERO;
            for qi in 0..q { s += w[qi] * g[qi * p + i] * g[qi * p + j]; }
            k[i + j * p] = s;
        }
    }
    k
}

// ── Jacobi eigenvalue solver ───────────────────────────────────────

fn jacobi_eigen_symmetric<T: Scalar>(
    a: &[T], p: usize, max_sweeps: usize,
) -> ReedResult<(Vec<T>, Vec<T>)> {
    let mut v = vec![T::ZERO; p * p];
    for i in 0..p { v[i + i * p] = T::ONE; }
    let mut w = a.to_vec();
    let mut d: Vec<T> = (0..p).map(|i| w[i + i * p]).collect();

    for _sweep in 0..max_sweeps {
        let mut converged = true;
        for i in 0..p {
            for j in (i + 1)..p {
                let a_ij = w[i + j * p];
                let tol = T::epsilon()
                    * d[i].abs().max(d[j].abs()).max(T::ONE)
                    * T::from_f64(p as f64).unwrap();
                if a_ij.abs() <= tol { continue; }
                converged = false;

                let two = T::from_f64(2.0).unwrap();
                let tau = (d[j] - d[i]) / (two * a_ij);
                let t = if tau >= T::ZERO {
                    T::ONE / (tau + (T::ONE + tau * tau).sqrt())
                } else {
                    -T::ONE / (-tau + (T::ONE + tau * tau).sqrt())
                };
                let c = T::ONE / (T::ONE + t * t).sqrt();
                let s = t * c;

                d[i] = d[i] - t * a_ij;
                d[j] = d[j] + t * a_ij;
                w[i + j * p] = T::ZERO;
                w[j + i * p] = T::ZERO;

                for k in 0..p {
                    if k == i || k == j { continue; }
                    let idx_ik = if k < i { k + i * p } else { i + k * p };
                    let idx_jk = if k < j { k + j * p } else { j + k * p };
                    let aik = w[idx_ik]; let ajk = w[idx_jk];
                    w[idx_ik] = c * aik - s * ajk;
                    w[idx_jk] = s * aik + c * ajk;
                }
                for k in 0..p {
                    let vki = v[k + i * p]; let vkj = v[k + j * p];
                    v[k + i * p] = c * vki - s * vkj;
                    v[k + j * p] = s * vki + c * vkj;
                }
            }
        }
        if converged { break; }
    }

    let mut evals: Vec<T> = (0..p).map(|i| w[i + i * p]).collect();
    let mut perm: Vec<usize> = (0..p).collect();
    perm.sort_by(|&a, &b| evals[a].partial_cmp(&evals[b]).unwrap_or(std::cmp::Ordering::Equal));
    let evals_sorted: Vec<T> = perm.iter().map(|&i| evals[i]).collect();
    let mut vs = vec![T::ZERO; p * p];
    for (nc, &oc) in perm.iter().enumerate() {
        for row in 0..p { vs[row + nc * p] = v[row + oc * p]; }
    }
    Ok((vs, evals_sorted))
}

fn build_fdm_1d_data<T: Scalar>(
    interp_1d: &[T], grad_1d: &[T], weights_1d: &[T], p: usize, q: usize,
) -> ReedResult<Fdm1dEigenData<T>> {
    let mass_1d = build_mass_1d(interp_1d, weights_1d, p, q);
    let stiff_1d = build_stiffness_1d(grad_1d, weights_1d, p, q);
    let (eigenvectors, mass_evals) = jacobi_eigen_symmetric(&mass_1d, p, 50)?;

    let mut stiff_evals = Vec::with_capacity(p);
    let mut temp = vec![T::ZERO; p];
    for k in 0..p {
        for i in 0..p {
            let mut s = T::ZERO;
            for j in 0..p { s += stiff_1d[i + j * p] * eigenvectors[j + k * p]; }
            temp[i] = s;
        }
        let mut lambda = T::ZERO;
        for i in 0..p { lambda += eigenvectors[i + k * p] * temp[i]; }
        stiff_evals.push(lambda);
    }
    Ok(Fdm1dEigenData { eigenvectors, mass_evals, stiff_evals })
}

// ── CpuFdmTensorInverseOperator ────────────────────────────────────

/// Tensor-product FDM element inverse operator.
///
/// Computes `y = A^{-1} x` using per-element fast diagonalization.
/// Works with tensor-product H1 Lagrange bases (Quad/Hex elements).
pub struct CpuFdmTensorInverseOperator<T: Scalar> {
    fdm_1d: Fdm1dEigenData<T>,
    dim: usize,
    p: usize,
    num_elem: usize,
    global_dof: usize,
    op_kind: FdmOperatorKind,
    restriction: Box<dyn ElemRestrictionTrait<T>>,
}

impl<T: Scalar> CpuFdmTensorInverseOperator<T> {
    pub fn new(
        interp_1d: &[T], grad_1d: &[T], weights_1d: &[T],
        p: usize, q: usize, dim: usize, num_elem: usize,
        op_kind: FdmOperatorKind,
        restriction: Box<dyn ElemRestrictionTrait<T>>,
    ) -> ReedResult<Self> {
        let fdm_1d = build_fdm_1d_data(interp_1d, grad_1d, weights_1d, p, q)?;
        let global_dof = restriction.num_global_dof();
        Ok(Self { fdm_1d, dim, p, num_elem, global_dof, op_kind, restriction })
    }

    fn lambda_slice(&self) -> &[T] {
        match self.op_kind {
            FdmOperatorKind::Mass => &self.fdm_1d.mass_evals,
            FdmOperatorKind::Stiffness => &self.fdm_1d.stiff_evals,
        }
    }

    fn apply_element_1d(&self, u: &[T], v: &mut [T]) {
        let p = self.p;
        let v_mat = &self.fdm_1d.eigenvectors;
        let lambda = self.lambda_slice();
        let mut u_hat = vec![T::ZERO; p];
        for k in 0..p {
            let mut s = T::ZERO;
            for i in 0..p { s += v_mat[i + k * p] * u[i]; }
            u_hat[k] = s;
        }
        v.fill(T::ZERO);
        for k in 0..p {
            let scaled = u_hat[k] / lambda[k];
            if scaled != T::ZERO {
                for i in 0..p { v[i] += v_mat[i + k * p] * scaled; }
            }
        }
    }

    fn apply_element_2d(&self, u: &[T], v: &mut [T]) {
        let p = self.p;
        let v_mat = &self.fdm_1d.eigenvectors;
        let lambda = self.lambda_slice();
        let p2 = p * p;

        let mut tmp = vec![T::ZERO; p2];
        for i in 0..p {
            for k in 0..p {
                let mut s = T::ZERO;
                for j in 0..p { s += u[i * p + j] * v_mat[j + k * p]; }
                tmp[i * p + k] = s;
            }
        }
        v.fill(T::ZERO);
        let is_mass = self.op_kind == FdmOperatorKind::Mass;
        for l in 0..p {
            for k in 0..p {
                let mut u_hat = T::ZERO;
                for i in 0..p { u_hat += v_mat[i + l * p] * tmp[i * p + k]; }
                let denom = if is_mass { lambda[l] * lambda[k] } else { lambda[l] + lambda[k] };
                let scaled = u_hat / denom;
                if scaled == T::ZERO { continue; }
                for i in 0..p {
                    let vi = v_mat[i + l * p] * scaled;
                    if vi == T::ZERO { continue; }
                    for j in 0..p { v[i * p + j] += vi * v_mat[j + k * p]; }
                }
            }
        }
    }

    fn apply_element_3d(&self, u: &[T], v: &mut [T]) {
        let p = self.p;
        let v_mat = &self.fdm_1d.eigenvectors;
        let lambda = self.lambda_slice();
        let p2 = p * p;
        let p3 = p2 * p;

        let mut t1 = vec![T::ZERO; p3];
        for i in 0..p { for j in 0..p {
            let base = (i * p + j) * p;
            for l in 0..p {
                let mut s = T::ZERO;
                for k in 0..p { s += u[base + k] * v_mat[k + l * p]; }
                t1[base + l] = s;
            }
        }}
        let mut t2 = vec![T::ZERO; p3];
        for i in 0..p { for m in 0..p { for l in 0..p {
            let mut s = T::ZERO;
            for j in 0..p { s += v_mat[j + m * p] * t1[(i * p + j) * p + l]; }
            t2[(i * p + m) * p + l] = s;
        }}}
        v.fill(T::ZERO);
        let is_mass = self.op_kind == FdmOperatorKind::Mass;
        for n in 0..p { for m in 0..p { for l in 0..p {
            let mut u_hat = T::ZERO;
            for i in 0..p { u_hat += v_mat[i + n * p] * t2[(i * p + m) * p + l]; }
            let denom = if is_mass {
                lambda[n] * lambda[m] * lambda[l]
            } else {
                lambda[n] + lambda[m] + lambda[l]
            };
            let scaled = u_hat / denom;
            if scaled == T::ZERO { continue; }
            for i in 0..p {
                let vin = v_mat[i + n * p] * scaled;
                if vin == T::ZERO { continue; }
                for j in 0..p {
                    let vij = vin * v_mat[j + m * p];
                    if vij == T::ZERO { continue; }
                    for k in 0..p { v[(i * p + j) * p + k] += vij * v_mat[k + l * p]; }
                }
            }
        }}}
    }

    fn apply_element(&self, u_local: &[T], v_local: &mut [T]) -> ReedResult<()> {
        match self.dim {
            1 => { self.apply_element_1d(u_local, v_local); Ok(()) }
            2 => { self.apply_element_2d(u_local, v_local); Ok(()) }
            3 => { self.apply_element_3d(u_local, v_local); Ok(()) }
            _ => Err(ReedError::Operator(format!("FDM: unsupported dim {}", self.dim))),
        }
    }
}

// ── OperatorTrait impl ─────────────────────────────────────────────

impl<T: Scalar> OperatorTrait<T> for CpuFdmTensorInverseOperator<T> {
    fn global_vector_len_hint(&self) -> Option<usize> { Some(self.global_dof) }

    fn operator_supports_assemble(&self, kind: OperatorAssembleKind) -> bool {
        matches!(kind, OperatorAssembleKind::Diagonal)
    }

    fn apply(&self, x: &dyn VectorTrait<T>, y: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        self.apply_impl(x.as_slice(), y.as_mut_slice(), false)
    }

    fn apply_add(&self, x: &dyn VectorTrait<T>, y: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        self.apply_impl(x.as_slice(), y.as_mut_slice(), true)
    }

    fn apply_with_transpose(
        &self, req: OperatorTransposeRequest,
        x: &dyn VectorTrait<T>, y: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match req {
            OperatorTransposeRequest::Forward => self.apply(x, y),
            OperatorTransposeRequest::Adjoint => self.apply(x, y), // symmetric
        }
    }

    fn apply_add_with_transpose(
        &self, req: OperatorTransposeRequest,
        x: &dyn VectorTrait<T>, y: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match req {
            OperatorTransposeRequest::Forward => self.apply_add(x, y),
            OperatorTransposeRequest::Adjoint => self.apply_add(x, y),
        }
    }

    fn linear_assemble_diagonal(&self, a: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        if a.len() != self.global_dof {
            return Err(ReedError::Operator(format!("diag: len {} != {}", a.len(), self.global_dof)));
        }
        let s = a.as_mut_slice();
        let mut x = vec![T::ZERO; self.global_dof];
        let mut y = vec![T::ZERO; self.global_dof];
        for j in 0..self.global_dof {
            x[j] = T::ONE;
            self.apply_impl(&x, &mut y, false);
            s[j] = y[j];
            x[j] = T::ZERO; y[j] = T::ZERO;
        }
        Ok(())
    }

    fn linear_assemble_add_diagonal(&self, a: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        if a.len() != self.global_dof {
            return Err(ReedError::Operator(format!("diag_add: len {} != {}", a.len(), self.global_dof)));
        }
        let s = a.as_mut_slice();
        let mut x = vec![T::ZERO; self.global_dof];
        let mut y = vec![T::ZERO; self.global_dof];
        for j in 0..self.global_dof {
            x[j] = T::ONE;
            self.apply_impl(&x, &mut y, false);
            s[j] += y[j];
            x[j] = T::ZERO; y[j] = T::ZERO;
        }
        Ok(())
    }
}

impl<T: Scalar> CpuFdmTensorInverseOperator<T> {
    fn apply_impl(&self, xg: &[T], yg: &mut [T], add: bool) -> ReedResult<()> {
        if xg.len() != self.global_dof || yg.len() != self.global_dof {
            return Err(ReedError::Operator(format!(
                "FDM apply: expected len {}, got x={} y={}", self.global_dof, xg.len(), yg.len()
            )));
        }
        let edof = self.restriction.num_dof_per_elem() * self.restriction.num_comp();
        let nelem = self.restriction.num_elements();
        let lsize = nelem * edof;
        let mut ul = vec![T::ZERO; lsize];
        let mut vl = vec![T::ZERO; lsize];

        self.restriction.apply(TransposeMode::NoTranspose, xg, &mut ul)?;
        for e in 0..nelem {
            self.apply_element(
                &ul[e * edof..(e + 1) * edof],
                &mut vl[e * edof..(e + 1) * edof],
            )?;
        }
        if !add { yg.fill(T::ZERO); }
        self.restriction.apply(TransposeMode::Transpose, &vl, yg)?;
        Ok(())
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p reed-cpu 2>&1 | tail -20`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/cpu/src/fdm_tensor.rs
git commit -m "feat: add CpuFdmTensorInverseOperator with full OperatorTrait impl"
```

---

### Task 6: Wire tensor FDM into CpuOperator

**Files:**
- Modify: `crates/cpu/src/operator.rs`
- Modify: `crates/cpu/src/lib.rs`

- [ ] **Step 1: Register fdm_tensor module**

In `crates/cpu/src/lib.rs`, add after `mod fdm_inverse;`:

```rust
mod fdm_tensor;
```

- [ ] **Step 2: Modify operator_create_fdm_element_inverse**

In `crates/cpu/src/operator.rs`, replace the method body (lines 1592–1624) with:

```rust
    fn operator_create_fdm_element_inverse(&self) -> ReedResult<Box<dyn OperatorTrait<T>>> {
        self.check_ready()?;
        let n = self.active_global_dof_len()?;

        // Try tensor FDM path when the basis supports it.
        if let Some(tensor_inv) = self.try_create_fdm_tensor_inverse()? {
            return Ok(tensor_inv);
        }

        // Fallback: dense inversion for small n.
        if n > crate::fdm_inverse::FDM_DENSE_MAX_N {
            return Err(ReedError::Operator(format!(
                "operator_create_fdm_element_inverse: global DOF {} exceeds dense limit {} and tensor FDM not available",
                n, crate::fdm_inverse::FDM_DENSE_MAX_N
            )));
        }
        let len = n.checked_mul(n).ok_or_else(|| {
            ReedError::Operator("operator_create_fdm_element_inverse: n*n overflow".into())
        })?;
        let mut a_vec = vec![T::ZERO; len];
        for j in 0..n {
            let mut input = vec![T::ZERO; n];
            input[j] = T::ONE;
            let x = crate::vector::CpuVector::from_vec(input);
            let mut y = crate::vector::CpuVector::new(n);
            self.apply(&x, &mut y)?;
            for i in 0..n { a_vec[i + j * n] = y.as_slice()[i]; }
        }
        let inv = crate::fdm_inverse::invert_dense_col_major(&a_vec, n)?;
        Ok(Box::new(crate::fdm_inverse::CpuFdmDenseInverseOperator::new(n, inv)))
    }

    fn try_create_fdm_tensor_inverse(&self) -> ReedResult<Option<Box<dyn OperatorTrait<T>>>> {
        use crate::fdm_tensor::{CpuFdmTensorInverseOperator, FdmOperatorKind};

        let field_idx = match self.input_plans.first() {
            Some(p) => p.field_index,
            None => return Ok(None),
        };
        let field = &self.fields[field_idx];
        let basis = match field.basis {
            Some(b) => b,
            None => return Ok(None),
        };
        let restriction = match field.restriction {
            Some(r) => r,
            None => return Ok(None),
        };

        let (interp_1d, grad_1d, weights_1d, p, q) = match basis.tensor_fdm_1d_data() {
            Some(d) => d,
            None => return Ok(None),
        };

        let dim = basis.dim();
        let nelem = self.num_elem;

        // Heuristic: check QFunction inputs for "du"/"grad" → Stiffness, else Mass.
        let op_kind = if self.qfunction.inputs().iter().any(|f| {
            f.name.contains("du") || f.name.contains("grad")
        }) {
            FdmOperatorKind::Stiffness
        } else {
            FdmOperatorKind::Mass
        };

        let restriction_box = restriction.boxed_clone()?;

        let inv = CpuFdmTensorInverseOperator::new(
            interp_1d, grad_1d, weights_1d, p, q, dim, nelem,
            op_kind, restriction_box,
        )?;
        Ok(Some(Box::new(inv)))
    }
```

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p reed-cpu 2>&1 | tail -20`
Expected: Compiles.

- [ ] **Step 4: Commit**

```bash
git add crates/cpu/src/operator.rs crates/cpu/src/lib.rs
git commit -m "feat: wire tensor FDM into CpuOperator::operator_create_fdm_element_inverse"
```

---

### Task 7: Re-export from top-level crate

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 1: Add re-export**

Find the existing `CpuFdmDenseInverseOperator` re-export block and add alongside it:

```rust
    pub use reed_cpu::fdm_tensor::CpuFdmTensorInverseOperator;
```

If the `fdm_tensor` module is `pub mod`, this works. If not, also export it from `reed-cpu`'s `lib.rs`:

```rust
pub mod fdm_tensor;
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p reed 2>&1 | tail -5`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add src/lib.rs crates/cpu/src/lib.rs
git commit -m "feat: re-export CpuFdmTensorInverseOperator"
```

---

### Task 8: Integration tests

**Files:**
- Modify: `tests/integration.rs`

- [ ] **Step 1: Add tensor FDM tests**

Append to `tests/integration.rs`:

```rust
#[test]
fn test_tensor_fdm_mass_2d_small_vs_dense_inverse() {
    // Build a small 2D mass operator with a single Quad element.
    // Compare tensor FDM inverse vs dense inverse.
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let p: usize = 3;
    let q: usize = 4;
    let nelem: usize = 1;
    let ndof = p * p;
    let ng = ndof * nelem;

    let basis = reed.basis_tensor_h1_lagrange(2, 1, p, q, QuadMode::GaussLobatto).unwrap();
    let offsets: Vec<CeedInt> = (0..ndof as CeedInt).collect();
    let restr = reed.elem_restriction(nelem, ndof, 1, ng, &offsets).unwrap();
    let qf = q_function_by_name("MassApply").unwrap();

    let op = reed.operator_builder()
        .qfunction(qf)
        .field("u", Some(&*restr), Some(&*basis), FieldVector::Active)
        .field("v", Some(&*restr), Some(&*basis), FieldVector::Active)
        .build()
        .unwrap();

    // Dense inverse (reference)
    let dense_inv = op.operator_create_fdm_element_inverse().unwrap(); // n=9 ≤ 256 → dense path

    // Force tensor path by checking n > 0 (the operator will prefer tensor when basis supports it)
    // Actually, with n=9 ≤ 256, the current code goes to dense first. We test tensor directly.
    // Build a large operator to trigger tensor path.
    // ... or just test via the direct tensor constructor for now.
    // For v1: test via the operator's internal path on a larger problem.

    // Here we just verify the small dense path still works:
    let mut x = reed.vector(ndof).unwrap();
    let mut y = reed.vector(ndof).unwrap();
    x.set_value(1.0).unwrap();
    dense_inv.apply(&*x, &mut *y).unwrap();
    let y_slice = y.as_slice().to_vec();
    assert!(y_slice.iter().any(|&v| v.abs() > 0.0), "FDM inverse should produce non-zero output");
}
```

Wait, this test design doesn't work as-is because the operator with n=9 will prefer dense. Let me design proper tests.

- [ ] **Step 1 (revised): Add tensor FDM tests**

Append to `tests/integration.rs`:

```rust
#[test]
fn test_tensor_fdm_mass_1d_vs_apply_identity() {
    // 1D mass operator: A(u) = M·u. Verify FDM(A)(b) ≈ M^{-1}·b
    // i.e., M * FDM_inv(b) ≈ b
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let p: usize = 4;
    let q: usize = 5;
    let nelem: usize = 2;
    let ndof = p;
    let ng = nelem * (p - 1) + 1; // H1 continuity
    let ng_fdm = nelem * ndof;     // for simplicity, use discontinuous mesh

    let basis = reed.basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::GaussLobatto).unwrap();
    let offsets: Vec<CeedInt> = (0..(nelem * ndof) as CeedInt).collect();
    let restr = reed.elem_restriction(nelem, ndof, 1, nelem * ndof, &offsets).unwrap();
    let qf = q_function_by_name("MassApply").unwrap();

    let op = reed.operator_builder()
        .qfunction(qf)
        .field("u", Some(&*restr), Some(&*basis), FieldVector::Active)
        .field("v", Some(&*restr), Some(&*basis), FieldVector::Active)
        .build()
        .unwrap();

    // With ng = nelem * ndof = 8 ≤ 256, this goes to dense path.
    // For tensor path test: build an operator with many elements to exceed FDM_DENSE_MAX_N.
    // Use p=4, nelem=100, discontinuous → ng=400 > 256 → tensor path.
    let p2: usize = 4;
    let q2: usize = 5;
    let nelem2: usize = 100;
    let ndof2 = p2;
    let ng2 = nelem2 * ndof2;

    let basis2 = reed.basis_tensor_h1_lagrange(1, 1, p2, q2, QuadMode::GaussLobatto).unwrap();
    let offsets2: Vec<CeedInt> = (0..ng2 as CeedInt).collect();
    let restr2 = reed.elem_restriction(nelem2, ndof2, 1, ng2, &offsets2).unwrap();
    let qf2 = q_function_by_name("MassApply").unwrap();

    let op2 = reed.operator_builder()
        .qfunction(qf2)
        .field("u", Some(&*restr2), Some(&*basis2), FieldVector::Active)
        .field("v", Some(&*restr2), Some(&*basis2), FieldVector::Active)
        .build()
        .unwrap();

    assert!(op2.operator_supports_assemble(OperatorAssembleKind::FdmElementInverse));

    // Create tensor FDM inverse
    let fdm_inv = op2.operator_create_fdm_element_inverse().unwrap();

    // Verify M * FDM_inv(e_j) ≈ e_j for a few basis vectors
    for j in [0, ng2 / 2, ng2 - 1].iter() {
        let mut ej = reed.vector(ng2).unwrap();
        ej.set_value(0.0).unwrap();
        ej.as_mut_slice()[*j] = 1.0;

        let mut fdm_ej = reed.vector(ng2).unwrap();
        fdm_inv.apply(&*ej, &mut *fdm_ej).unwrap();

        let mut m_fdm_ej = reed.vector(ng2).unwrap();
        op2.apply(&*fdm_ej, &mut *m_fdm_ej).unwrap();

        let err = (m_fdm_ej.as_slice()[*j] - 1.0).abs();
        assert!(err < 0.1, "M * FDM_inv(e_{j})[{j}] = {}, expected ≈ 1.0, err={err}",
            m_fdm_ej.as_slice()[*j]);
    }
}

#[test]
fn test_tensor_fdm_stiffness_2d_large_n_does_not_panic() {
    // Verify tensor FDM is created without error for n > 256.
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let p: usize = 4;
    let q: usize = 5;
    let nelem: usize = 50;
    let ndof = p * p;
    let ng = nelem * ndof;

    let basis = reed.basis_tensor_h1_lagrange(2, 1, p, q, QuadMode::GaussLobatto).unwrap();
    let offsets: Vec<CeedInt> = (0..ng as CeedInt).collect();
    let restr = reed.elem_restriction(nelem, ndof, 1, ng, &offsets).unwrap();
    let qf = q_function_by_name("Poisson2DApply").unwrap();

    let op = reed.operator_builder()
        .qfunction(qf)
        .field("du", Some(&*restr), Some(&*basis), FieldVector::Active)
        .field("v", Some(&*restr), Some(&*basis), FieldVector::Active)
        .build()
        .unwrap();

    // ng = 50 * 16 = 800 > 256 → should go to tensor FDM path.
    assert!(op.operator_supports_assemble(OperatorAssembleKind::FdmElementInverse));
    let fdm_inv = op.operator_create_fdm_element_inverse().unwrap();

    // Apply to a random vector — should not panic.
    let mut x = reed.vector(ng).unwrap();
    let mut y = reed.vector(ng).unwrap();
    x.set_value(1.0).unwrap();
    fdm_inv.apply(&*x, &mut *y).unwrap();
    assert!(y.as_slice().iter().any(|v| v.abs() > 0.0));
}

#[test]
fn test_tensor_fdm_stiffness_1d_non_tensor_basis_returns_none() {
    // SimplexBasis does NOT support tensor FDM — verify fallback to dense.
    let reed = Reed::<f64>::init("/cpu/self").unwrap();
    let basis = reed.basis_h1_simplex(ElemTopology::Line, 1, 2, QuadMode::Gauss).unwrap();
    let ndof = 2; let nelem = 1; let ng = nelem * ndof;
    let offsets: Vec<CeedInt> = (0..ng as CeedInt).collect();
    let restr = reed.elem_restriction(nelem, ndof, 1, ng, &offsets).unwrap();
    let qf = q_function_by_name("Poisson1DApply").unwrap();
    let op = reed.operator_builder()
        .qfunction(qf)
        .field("du", Some(&*restr), Some(&*basis), FieldVector::Active)
        .field("v", Some(&*restr), Some(&*basis), FieldVector::Active)
        .build()
        .unwrap();

    // n=2 ≤ 256 → goes to dense path (simplex doesn't support tensor_fdm_1d_data).
    let fdm_inv = op.operator_create_fdm_element_inverse().unwrap();
    let mut x = reed.vector(ng).unwrap();
    let mut y = reed.vector(ng).unwrap();
    x.set_value(1.0).unwrap();
    fdm_inv.apply(&*x, &mut *y).unwrap();
    assert!(y.as_slice().iter().any(|v| v.abs() > 0.0));
}
```

- [ ] **Step 2: Run the new tests**

Run: `cargo test -p reed -- test_tensor_fdm 2>&1 | tail -20`
Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/integration.rs
git commit -m "test: add tensor FDM integration tests"
```

---

### Task 9: Run full test suite and verify no regressions

- [ ] **Step 1: Run all tests**

Run: `cargo test --workspace 2>&1 | tail -30`
Expected: All existing tests pass. No regressions.

- [ ] **Step 2: Commit any fixes if needed**

```bash
git add -A
git commit -m "chore: fix any regressions from tensor FDM changes"
```
(If no fixes needed, skip this step.)

