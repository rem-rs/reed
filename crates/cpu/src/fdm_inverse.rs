//! libCEED-shaped **`CeedOperatorCreateFDMElementInverse`** on the CPU path.
//!
//! libCEED builds a **tensor fast diagonalization** approximate inverse when the operator uses
//! tensor H1 bases. Reed does not yet implement that factorization; instead, when the assembled
//! **global** Jacobian is small (`active_global_dof_len ≤` [`FDM_DENSE_MAX_N`]), we
//! [`CpuOperator::linear_assemble_symbolic`] / [`CpuOperator::linear_assemble`], invert the dense
//! matrix by Gauss–Jordan elimination, and return [`CpuFdmDenseInverseOperator`] whose `apply`
//! multiplies by \(A^{-1}\). This matches the **action** of an exact inverse on that subspace and is
//! sufficient for tests and small coarse operators.

use num_traits::cast::NumCast;
use reed_core::{
    error::{ReedError, ReedResult},
    operator::{OperatorAssembleKind, OperatorTrait, OperatorTransposeRequest},
    scalar::Scalar,
    vector::VectorTrait,
};

/// Maximum `n` for [`CpuOperator::operator_create_fdm_element_inverse`](crate::operator::CpuOperator::operator_create_fdm_element_inverse)
/// (dense assembly + inversion cost `O(n³)` memory/time).
pub const FDM_DENSE_MAX_N: usize = 256;

/// Lightweight structured fallback inverse: diagonal (Jacobi) inverse assembled from
/// `linear_assemble_diagonal`, then applied as pointwise scaling.
pub struct CpuFdmJacobiInverseOperator<T: Scalar> {
    n: usize,
    inv_diag: Vec<T>,
}

impl<T: Scalar> CpuFdmJacobiInverseOperator<T> {
    pub fn new(inv_diag: Vec<T>) -> Self {
        let n = inv_diag.len();
        Self { n, inv_diag }
    }
}

/// Invert `n × n` matrix stored **column-major**: `a[row + col * n] = A[row, col]`.
pub fn invert_dense_col_major<T: Scalar>(a: &[T], n: usize) -> ReedResult<Vec<T>> {
    if a.len() != n * n {
        return Err(ReedError::Operator(format!(
            "invert_dense_col_major: expected {} entries for n={}, got {}",
            n * n,
            n,
            a.len()
        )));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let two_n = n
        .checked_mul(2)
        .ok_or_else(|| ReedError::Operator("invert_dense_col_major: n*2 overflow".into()))?;
    let len = n.checked_mul(two_n).ok_or_else(|| {
        ReedError::Operator("invert_dense_col_major: augmented matrix size overflow".into())
    })?;
    let mut work = vec![T::ZERO; len];
    for i in 0..n {
        for j in 0..n {
            work[i * two_n + j] = a[i + j * n];
        }
        work[i * two_n + n + i] = T::ONE;
    }
    let scale: T = NumCast::from((n as f64).max(64.0)).unwrap_or(T::ONE);
    let tol = T::epsilon() * scale;
    for k in 0..n {
        let mut piv = k;
        let mut best = work[k * two_n + k].abs();
        for r in (k + 1)..n {
            let v = work[r * two_n + k].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best <= tol {
            return Err(ReedError::Operator(
                "operator_create_fdm_element_inverse: assembled matrix is singular or ill-conditioned (dense inversion failed)"
                    .into(),
            ));
        }
        if piv != k {
            for c in 0..two_n {
                work.swap(k * two_n + c, piv * two_n + c);
            }
        }
        let pivot = work[k * two_n + k];
        let inv_pivot = T::ONE / pivot;
        for c in 0..two_n {
            work[k * two_n + c] = work[k * two_n + c] * inv_pivot;
        }
        for r in 0..n {
            if r == k {
                continue;
            }
            let f = work[r * two_n + k];
            if f != T::ZERO {
                for c in 0..two_n {
                    let v = work[k * two_n + c];
                    work[r * two_n + c] = work[r * two_n + c] - f * v;
                }
            }
        }
    }
    let mut inv = vec![T::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i + j * n] = work[i * two_n + n + j];
        }
    }
    Ok(inv)
}

/// Operator `y = A^{-1} x` with `A` from a parent [`crate::operator::CpuOperator`] at construction time.
pub struct CpuFdmDenseInverseOperator<T: Scalar> {
    n: usize,
    inv_col_major: Vec<T>,
}

impl<T: Scalar> CpuFdmDenseInverseOperator<T> {
    pub fn new(n: usize, inv_col_major: Vec<T>) -> Self {
        Self { n, inv_col_major }
    }

    fn matvec(&self, transpose: bool, x: &[T], y: &mut [T], add: bool) -> ReedResult<()> {
        if x.len() != self.n || y.len() != self.n {
            return Err(ReedError::Operator(format!(
                "CpuFdmDenseInverseOperator: expected length {}, got x={}, y={}",
                self.n,
                x.len(),
                y.len()
            )));
        }
        for i in 0..self.n {
            let mut acc = T::ZERO;
            for j in 0..self.n {
                let m_ij = if transpose {
                    self.inv_col_major[j + i * self.n]
                } else {
                    self.inv_col_major[i + j * self.n]
                };
                acc += m_ij * x[j];
            }
            if add {
                y[i] += acc;
            } else {
                y[i] = acc;
            }
        }
        Ok(())
    }
}

impl<T: Scalar> OperatorTrait<T> for CpuFdmDenseInverseOperator<T> {
    fn global_vector_len_hint(&self) -> Option<usize> {
        Some(self.n)
    }

    fn operator_supports_assemble(&self, kind: OperatorAssembleKind) -> bool {
        matches!(kind, OperatorAssembleKind::Diagonal)
    }

    fn apply(&self, input: &dyn VectorTrait<T>, output: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        self.matvec(false, input.as_slice(), output.as_mut_slice(), false)
    }

    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        self.matvec(false, input.as_slice(), output.as_mut_slice(), true)
    }

    fn apply_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply(input, output),
            OperatorTransposeRequest::Adjoint => {
                self.matvec(true, input.as_slice(), output.as_mut_slice(), false)
            }
        }
    }

    fn apply_add_with_transpose(
        &self,
        request: OperatorTransposeRequest,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match request {
            OperatorTransposeRequest::Forward => self.apply_add(input, output),
            OperatorTransposeRequest::Adjoint => {
                self.matvec(true, input.as_slice(), output.as_mut_slice(), true)
            }
        }
    }

    fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        if assembled.len() != self.n {
            return Err(ReedError::Operator(format!(
                "linear_assemble_diagonal: assembled length {} != {}",
                assembled.len(),
                self.n
            )));
        }
        let s = assembled.as_mut_slice();
        for i in 0..self.n {
            s[i] = self.inv_col_major[i + i * self.n];
        }
        Ok(())
    }

    fn linear_assemble_add_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        if assembled.len() != self.n {
            return Err(ReedError::Operator(format!(
                "linear_assemble_add_diagonal: assembled length {} != {}",
                assembled.len(),
                self.n
            )));
        }
        let s = assembled.as_mut_slice();
        for i in 0..self.n {
            s[i] += self.inv_col_major[i + i * self.n];
        }
        Ok(())
    }
}

impl<T: Scalar> OperatorTrait<T> for CpuFdmJacobiInverseOperator<T> {
    fn global_vector_len_hint(&self) -> Option<usize> {
        Some(self.n)
    }

    fn operator_supports_assemble(&self, kind: OperatorAssembleKind) -> bool {
        matches!(kind, OperatorAssembleKind::Diagonal)
    }

    fn apply(&self, input: &dyn VectorTrait<T>, output: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        if input.len() != self.n || output.len() != self.n {
            return Err(ReedError::Operator(format!(
                "CpuFdmJacobiInverseOperator: expected length {}, got x={}, y={}",
                self.n,
                input.len(),
                output.len()
            )));
        }
        for i in 0..self.n {
            output.as_mut_slice()[i] = self.inv_diag[i] * input.as_slice()[i];
        }
        Ok(())
    }

    fn apply_add(
        &self,
        input: &dyn VectorTrait<T>,
        output: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        if input.len() != self.n || output.len() != self.n {
            return Err(ReedError::Operator(format!(
                "CpuFdmJacobiInverseOperator: expected length {}, got x={}, y={}",
                self.n,
                input.len(),
                output.len()
            )));
        }
        for i in 0..self.n {
            output.as_mut_slice()[i] += self.inv_diag[i] * input.as_slice()[i];
        }
        Ok(())
    }

    fn linear_assemble_diagonal(&self, assembled: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        if assembled.len() != self.n {
            return Err(ReedError::Operator(format!(
                "linear_assemble_diagonal: assembled length {} != {}",
                assembled.len(),
                self.n
            )));
        }
        assembled.copy_from_slice(&self.inv_diag)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_identity() {
        let n = 3usize;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            a[i + i * n] = 1.0;
        }
        let inv = invert_dense_col_major(&a, n).unwrap();
        for i in 0..n {
            for j in 0..n {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i + j * n] - e).abs() < 1e-12, "({i},{j})");
            }
        }
    }

    #[test]
    fn invert_2x2_matches_formula() {
        let a = vec![2.0f64, 1.0, 1.0, 3.0]; // column0 [2,1], column1 [1,3] -> A = [[2,1],[1,3]]
        let inv = invert_dense_col_major(&a, 2).unwrap();
        let det = 2.0 * 3.0 - 1.0 * 1.0;
        assert!((inv[0 + 0 * 2] - 3.0 / det).abs() < 1e-12);
        assert!((inv[1 + 0 * 2] - (-1.0) / det).abs() < 1e-12);
        assert!((inv[0 + 1 * 2] - (-1.0) / det).abs() < 1e-12);
        assert!((inv[1 + 1 * 2] - 2.0 / det).abs() < 1e-12);
    }
}
