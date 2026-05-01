//! libCEED-shaped matrix handle semantics for assembled operators.

use crate::{
    csr::{CsrMatrix, CsrPattern},
    error::{ReedError, ReedResult},
    scalar::Scalar,
};

/// Backing storage for a Reed matrix handle.
#[derive(Clone, Debug, PartialEq)]
pub enum CeedMatrixStorage<T: Scalar> {
    DenseColMajor {
        nrows: usize,
        ncols: usize,
        values: Vec<T>,
    },
    Csr(CsrMatrix<T>),
}

/// Matrix handle with explicit symbolic/numeric readiness flags, similar to libCEED assembly flow.
#[derive(Clone, Debug, PartialEq)]
pub struct CeedMatrix<T: Scalar> {
    storage: CeedMatrixStorage<T>,
    symbolic_done: bool,
    numeric_done: bool,
}

impl<T: Scalar> CeedMatrix<T> {
    pub fn dense_col_major_symbolic(nrows: usize, ncols: usize) -> ReedResult<Self> {
        let len = nrows
            .checked_mul(ncols)
            .ok_or_else(|| ReedError::InvalidArgument("dense matrix size overflow".into()))?;
        Ok(Self {
            storage: CeedMatrixStorage::DenseColMajor {
                nrows,
                ncols,
                values: vec![T::ZERO; len],
            },
            symbolic_done: true,
            numeric_done: false,
        })
    }

    pub fn csr_symbolic(pattern: CsrPattern) -> Self {
        let nnz = pattern.nnz();
        Self {
            storage: CeedMatrixStorage::Csr(CsrMatrix {
                pattern,
                values: vec![T::ZERO; nnz],
            }),
            symbolic_done: true,
            numeric_done: false,
        }
    }

    pub fn symbolic_done(&self) -> bool {
        self.symbolic_done
    }

    pub fn numeric_done(&self) -> bool {
        self.numeric_done
    }

    pub fn storage(&self) -> &CeedMatrixStorage<T> {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut CeedMatrixStorage<T> {
        &mut self.storage
    }

    pub fn mark_numeric_done(&mut self, done: bool) {
        self.numeric_done = done;
    }

    pub fn clear_numeric_values(&mut self) {
        match &mut self.storage {
            CeedMatrixStorage::DenseColMajor { values, .. } => values.fill(T::ZERO),
            CeedMatrixStorage::Csr(m) => m.values.fill(T::ZERO),
        }
        self.numeric_done = false;
    }

    pub fn add_dense_col_major(&mut self, nrows: usize, ncols: usize, a: &[T]) -> ReedResult<()> {
        match &mut self.storage {
            CeedMatrixStorage::DenseColMajor {
                nrows: mr,
                ncols: mc,
                values,
            } => {
                if *mr != nrows || *mc != ncols {
                    return Err(ReedError::InvalidArgument(format!(
                        "dense shape mismatch: matrix is {}x{}, add is {}x{}",
                        *mr, *mc, nrows, ncols
                    )));
                }
                if a.len() != values.len() {
                    return Err(ReedError::InvalidArgument(format!(
                        "dense value length mismatch: matrix {}, add {}",
                        values.len(),
                        a.len()
                    )));
                }
                for (dst, src) in values.iter_mut().zip(a.iter()) {
                    *dst += *src;
                }
                self.numeric_done = true;
                Ok(())
            }
            CeedMatrixStorage::Csr(_) => Err(ReedError::InvalidArgument(
                "add_dense_col_major on CSR matrix".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_matrix_handle_accumulates() {
        let mut m = CeedMatrix::<f64>::dense_col_major_symbolic(2, 2).unwrap();
        m.add_dense_col_major(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        m.add_dense_col_major(2, 2, &[0.5, 1.0, 1.5, 2.0]).unwrap();
        match m.storage() {
            CeedMatrixStorage::DenseColMajor { values, .. } => {
                assert_eq!(values, &vec![1.5, 3.0, 4.5, 6.0]);
            }
            _ => panic!("expected dense matrix"),
        }
        assert!(m.numeric_done());
    }
}
