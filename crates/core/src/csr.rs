//! Compressed sparse row (CSR) pattern and values — migration stepping stone toward libCEED
//! `CeedOperatorLinearAssembleSymbolic` / `LinearAssemble` with a `CeedMatrix`-like layout in Rust.

use crate::error::{ReedError, ReedResult};
use crate::scalar::Scalar;

/// CSR sparsity pattern (`row_ptr` length `nrows + 1`, `col_ind` length `nnz`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsrPattern {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptr: Vec<usize>,
    pub col_ind: Vec<usize>,
}

impl CsrPattern {
    #[inline]
    pub fn nnz(&self) -> usize {
        self.col_ind.len()
    }

    /// CSR storage index for `A[row, col]`, if present. Columns within each row are sorted ascending.
    pub fn index_of(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.nrows || col >= self.ncols {
            return None;
        }
        let lo = self.row_ptr[row];
        let hi = self.row_ptr[row + 1];
        self.col_ind[lo..hi]
            .binary_search(&col)
            .ok()
            .map(|k| lo + k)
    }
}

/// CSR matrix = [`CsrPattern`] + numeric `values` (`len == nnz`).
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix<T: Scalar> {
    pub pattern: CsrPattern,
    pub values: Vec<T>,
}

impl<T: Scalar> CsrMatrix<T> {
    pub fn get(&self, row: usize, col: usize) -> Option<T> {
        let i = self.pattern.index_of(row, col)?;
        Some(self.values[i])
    }

    /// `y = A x` with `A` in CSR; overwrites `y`. Shapes: `x.len() == ncols`, `y.len() == nrows`.
    pub fn mul_vec(&self, x: &[T], y: &mut [T]) -> ReedResult<()> {
        if x.len() != self.pattern.ncols || y.len() != self.pattern.nrows {
            return Err(ReedError::InvalidArgument(format!(
                "CsrMatrix::mul_vec: x len {} (need {}), y len {} (need {})",
                x.len(),
                self.pattern.ncols,
                y.len(),
                self.pattern.nrows
            )));
        }
        y.fill(T::ZERO);
        self.mul_vec_add(x, y)
    }

    /// `y += A x` (accumulating GEMV).
    pub fn mul_vec_add(&self, x: &[T], y: &mut [T]) -> ReedResult<()> {
        if x.len() != self.pattern.ncols || y.len() != self.pattern.nrows {
            return Err(ReedError::InvalidArgument(format!(
                "CsrMatrix::mul_vec_add: x len {} (need {}), y len {} (need {})",
                x.len(),
                self.pattern.ncols,
                y.len(),
                self.pattern.nrows
            )));
        }
        for row in 0..self.pattern.nrows {
            let r0 = self.pattern.row_ptr[row];
            let r1 = self.pattern.row_ptr[row + 1];
            let mut acc = T::ZERO;
            for k in r0..r1 {
                let col = self.pattern.col_ind[k];
                acc += self.values[k] * x[col];
            }
            y[row] += acc;
        }
        Ok(())
    }
}

/// Build the **FEM-style sparsity pattern** on `0..lsize`: global DOFs `i` and `j` may be coupled iff
/// they both appear on the **same mesh element** (same `elem` index), using the same layout as
/// [`crate::elem_restriction::ElemRestrictionTrait`] **offset** gather: for each `(local, comp)`,
/// global index is `offsets[elem * elemsize + local] as usize + comp * compstride`.
///
/// `offsets.len()` must equal `nelem * elemsize`. `ncomp >= 1`.
pub fn csr_sparsity_from_offset_restriction(
    nelem: usize,
    elemsize: usize,
    ncomp: usize,
    compstride: usize,
    lsize: usize,
    offsets: &[i32],
) -> ReedResult<CsrPattern> {
    if ncomp == 0 || elemsize == 0 {
        return Err(ReedError::InvalidArgument(
            "csr_sparsity_from_offset_restriction: ncomp and elemsize must be positive".into(),
        ));
    }
    if offsets.len() != nelem.saturating_mul(elemsize) {
        return Err(ReedError::InvalidArgument(format!(
            "csr_sparsity_from_offset_restriction: offsets.len {} != nelem*elemsize {}",
            offsets.len(),
            nelem.saturating_mul(elemsize)
        )));
    }
    let mut coo: Vec<(usize, usize)> = Vec::new();
    for elem in 0..nelem {
        let sl = &offsets[elem * elemsize..(elem + 1) * elemsize];
        let mut indices = Vec::with_capacity(ncomp.saturating_mul(elemsize));
        for comp in 0..ncomp {
            let comp_base = comp.checked_mul(compstride).ok_or_else(|| {
                ReedError::InvalidArgument("csr_sparsity: comp * compstride overflow".into())
            })?;
            for &base in sl {
                if base < 0 {
                    return Err(ReedError::InvalidArgument(format!(
                        "csr_sparsity_from_offset_restriction: negative offset {base} at element {elem}"
                    )));
                }
                let g = (base as usize).checked_add(comp_base).ok_or_else(|| {
                    ReedError::InvalidArgument(
                        "csr_sparsity_from_offset_restriction: global index overflow".into(),
                    )
                })?;
                if g >= lsize {
                    return Err(ReedError::InvalidArgument(format!(
                        "csr_sparsity_from_offset_restriction: global index {g} >= lsize {lsize}"
                    )));
                }
                indices.push(g);
            }
        }
        for &gi in &indices {
            for &gj in &indices {
                coo.push((gi, gj));
            }
        }
    }
    coo.sort_unstable();
    coo.dedup();

    let nrows = lsize;
    let ncols = lsize;
    let mut row_ptr = vec![0usize; nrows + 1];
    for &(r, _) in &coo {
        row_ptr[r + 1] += 1;
    }
    for i in 0..nrows {
        row_ptr[i + 1] += row_ptr[i];
    }
    let nnz = coo.len();
    debug_assert_eq!(row_ptr[nrows], nnz);
    let mut col_ind = vec![0usize; nnz];
    let mut head: Vec<usize> = (0..nrows).map(|i| row_ptr[i]).collect();
    for &(r, c) in &coo {
        let k = head[r];
        col_ind[k] = c;
        head[r] += 1;
    }
    Ok(CsrPattern {
        nrows,
        ncols,
        row_ptr,
        col_ind,
    })
}

/// Same as [`csr_sparsity_from_offset_restriction`] with `ncomp = 1`, `compstride = 1` (scalar L-node indices only).
pub fn csr_sparsity_from_offset_lnodes(
    nelem: usize,
    elemsize: usize,
    lsize: usize,
    offsets: &[i32],
) -> ReedResult<CsrPattern> {
    csr_sparsity_from_offset_restriction(nelem, elemsize, 1, 1, lsize, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_sparsity_line_two_elements_seven_nonzeros() {
        let p = csr_sparsity_from_offset_lnodes(2, 2, 3, &[0, 1, 1, 2]).unwrap();
        assert_eq!(p.nnz(), 7);
        assert_eq!(p.index_of(0, 2), None);
        assert!(p.index_of(1, 2).is_some());
    }

    #[test]
    fn csr_sparsity_ncomp2_matches_element_closure() {
        let p = csr_sparsity_from_offset_restriction(1, 2, 2, 3, 10, &[0, 1]).unwrap();
        assert_eq!(p.nrows, 10);
        assert_eq!(p.nnz(), 4 * 4);
        assert!(p.index_of(0, 3).is_some());
        assert!(p.index_of(3, 0).is_some());
    }

    #[test]
    fn csr_matvec_diagonal_2x2() {
        let pat = CsrPattern {
            nrows: 2,
            ncols: 2,
            row_ptr: vec![0, 1, 2],
            col_ind: vec![0, 1],
        };
        let m = CsrMatrix {
            pattern: pat,
            values: vec![2.0_f64, 3.0_f64],
        };
        let x = vec![1.0_f64, 4.0_f64];
        let mut y = vec![0.0_f64; 2];
        m.mul_vec(&x, &mut y).unwrap();
        assert!((y[0] - 2.0).abs() < 1e-14);
        assert!((y[1] - 12.0).abs() < 1e-14);
    }
}
