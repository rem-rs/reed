//! Tensor-product Fast Diagonalization Method (FDM) element inverse.
//!
//! libCEED-aligned `CeedOperatorCreateFDMElementInverse`: for tensor-product H1 Lagrange
//! bases on Quad/Hex elements, the local Jacobian is separable. A 1D eigen-decomposition
//! diagonalizes the element operator, yielding O(p^{d+1}) apply instead of O(p^{3d}).
//!
//! [`CpuFdmTensorInverseOperator`] implements [`OperatorTrait`] and is created by
//! [`CpuOperator::operator_create_fdm_element_inverse`](crate::operator::CpuOperator::operator_create_fdm_element_inverse).

use num_traits::NumCast;
use reed_core::{
    elem_restriction::ElemRestrictionTrait,
    enums::TransposeMode,
    error::{ReedError, ReedResult},
    operator::{OperatorAssembleKind, OperatorTrait, OperatorTransposeRequest},
    scalar::Scalar,
    vector::VectorTrait,
};

// ── 1D eigen-data ──────────────────────────────────────────────────

struct Fdm1dEigenData<T: Scalar> {
    eigenvectors: Vec<T>, // p×p column-major, M-orthonormal
    mass_evals: Vec<T>,   // λ^M_k, sorted ascending
    stiff_evals: Vec<T>,  // λ^K_k (Rayleigh quotient)
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
            for qi in 0..q {
                s += w[qi] * b[qi * p + i] * b[qi * p + j];
            }
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
            for qi in 0..q {
                s += w[qi] * g[qi * p + i] * g[qi * p + j];
            }
            k[i + j * p] = s;
        }
    }
    k
}

// ── Jacobi eigenvalue solver ───────────────────────────────────────

fn jacobi_eigen_symmetric<T: Scalar>(
    a: &[T],
    p: usize,
    max_sweeps: usize,
) -> ReedResult<(Vec<T>, Vec<T>)> {
    let mut v = vec![T::ZERO; p * p];
    for i in 0..p {
        v[i + i * p] = T::ONE;
    }
    let mut w = a.to_vec();
    let mut d: Vec<T> = (0..p).map(|i| w[i + i * p]).collect();

    for _sweep in 0..max_sweeps {
        let mut converged = true;
        for i in 0..p {
            for j in (i + 1)..p {
                let a_ij = w[i + j * p];
                let p_t: T = NumCast::from(p as f64).unwrap_or(T::ONE);
                let tol = T::epsilon()
                    * d[i].abs().max(d[j].abs()).max(T::ONE)
                    * p_t;
                if a_ij.abs() <= tol {
                    continue;
                }
                converged = false;

                let two: T = NumCast::from(2.0_f64).unwrap_or(T::ONE);
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
                    if k == i || k == j {
                        continue;
                    }
                    let idx_ik = if k < i { k + i * p } else { i + k * p };
                    let idx_jk = if k < j { k + j * p } else { j + k * p };
                    let aik = w[idx_ik];
                    let ajk = w[idx_jk];
                    w[idx_ik] = c * aik - s * ajk;
                    w[idx_jk] = s * aik + c * ajk;
                }
                for k in 0..p {
                    let vki = v[k + i * p];
                    let vkj = v[k + j * p];
                    v[k + i * p] = c * vki - s * vkj;
                    v[k + j * p] = s * vki + c * vkj;
                }
            }
        }
        if converged {
            break;
        }
    }

    let evals: Vec<T> = (0..p).map(|i| w[i + i * p]).collect();
    let mut perm: Vec<usize> = (0..p).collect();
    perm.sort_by(|&a, &b| {
        evals[a]
            .partial_cmp(&evals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let evals_sorted: Vec<T> = perm.iter().map(|&i| evals[i]).collect();
    let mut vs = vec![T::ZERO; p * p];
    for (nc, &oc) in perm.iter().enumerate() {
        for row in 0..p {
            vs[row + nc * p] = v[row + oc * p];
        }
    }
    Ok((vs, evals_sorted))
}

fn build_fdm_1d_data<T: Scalar>(
    interp_1d: &[T],
    grad_1d: &[T],
    weights_1d: &[T],
    p: usize,
    q: usize,
) -> ReedResult<Fdm1dEigenData<T>> {
    let mass_1d = build_mass_1d(interp_1d, weights_1d, p, q);
    let stiff_1d = build_stiffness_1d(grad_1d, weights_1d, p, q);
    let (eigenvectors, mass_evals) = jacobi_eigen_symmetric(&mass_1d, p, 50)?;

    let mut stiff_evals = Vec::with_capacity(p);
    let mut temp = vec![T::ZERO; p];
    for k in 0..p {
        for i in 0..p {
            let mut s = T::ZERO;
            for j in 0..p {
                s += stiff_1d[i + j * p] * eigenvectors[j + k * p];
            }
            temp[i] = s;
        }
        let mut lambda = T::ZERO;
        for i in 0..p {
            lambda += eigenvectors[i + k * p] * temp[i];
        }
        stiff_evals.push(lambda);
    }
    Ok(Fdm1dEigenData {
        eigenvectors,
        mass_evals,
        stiff_evals,
    })
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
    #[allow(dead_code)]
    num_elem: usize,
    global_dof: usize,
    op_kind: FdmOperatorKind,
    restriction: Box<dyn ElemRestrictionTrait<T>>,
}

impl<T: Scalar> CpuFdmTensorInverseOperator<T> {
    pub fn new(
        interp_1d: &[T],
        grad_1d: &[T],
        weights_1d: &[T],
        p: usize,
        q: usize,
        dim: usize,
        num_elem: usize,
        op_kind: FdmOperatorKind,
        restriction: Box<dyn ElemRestrictionTrait<T>>,
    ) -> ReedResult<Self> {
        let fdm_1d = build_fdm_1d_data(interp_1d, grad_1d, weights_1d, p, q)?;
        let global_dof = restriction.num_global_dof();
        Ok(Self {
            fdm_1d,
            dim,
            p,
            num_elem,
            global_dof,
            op_kind,
            restriction,
        })
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
            for i in 0..p {
                s += v_mat[i + k * p] * u[i];
            }
            u_hat[k] = s;
        }
        v.fill(T::ZERO);
        for k in 0..p {
            let scaled = u_hat[k] / lambda[k];
            if scaled != T::ZERO {
                for i in 0..p {
                    v[i] += v_mat[i + k * p] * scaled;
                }
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
                for j in 0..p {
                    s += u[i * p + j] * v_mat[j + k * p];
                }
                tmp[i * p + k] = s;
            }
        }
        v.fill(T::ZERO);
        let is_mass = self.op_kind == FdmOperatorKind::Mass;
        for l in 0..p {
            for k in 0..p {
                let mut u_hat = T::ZERO;
                for i in 0..p {
                    u_hat += v_mat[i + l * p] * tmp[i * p + k];
                }
                let denom = if is_mass {
                    lambda[l] * lambda[k]
                } else {
                    lambda[l] + lambda[k]
                };
                let scaled = u_hat / denom;
                if scaled == T::ZERO {
                    continue;
                }
                for i in 0..p {
                    let vi = v_mat[i + l * p] * scaled;
                    if vi == T::ZERO {
                        continue;
                    }
                    for j in 0..p {
                        v[i * p + j] += vi * v_mat[j + k * p];
                    }
                }
            }
        }
    }

    fn apply_element_3d(&self, u: &[T], v: &mut [T]) {
        let p = self.p;
        let v_mat = &self.fdm_1d.eigenvectors;
        let lambda = self.lambda_slice();
        let p3 = p * p * p;

        let mut t1 = vec![T::ZERO; p3];
        for i in 0..p {
            for j in 0..p {
                let base = (i * p + j) * p;
                for l in 0..p {
                    let mut s = T::ZERO;
                    for k in 0..p {
                        s += u[base + k] * v_mat[k + l * p];
                    }
                    t1[base + l] = s;
                }
            }
        }
        let mut t2 = vec![T::ZERO; p3];
        for i in 0..p {
            for m in 0..p {
                for l in 0..p {
                    let mut s = T::ZERO;
                    for j in 0..p {
                        s += v_mat[j + m * p] * t1[(i * p + j) * p + l];
                    }
                    t2[(i * p + m) * p + l] = s;
                }
            }
        }
        v.fill(T::ZERO);
        let is_mass = self.op_kind == FdmOperatorKind::Mass;
        for n in 0..p {
            for m in 0..p {
                for l in 0..p {
                    let mut u_hat = T::ZERO;
                    for i in 0..p {
                        u_hat += v_mat[i + n * p] * t2[(i * p + m) * p + l];
                    }
                    let denom = if is_mass {
                        lambda[n] * lambda[m] * lambda[l]
                    } else {
                        lambda[n] + lambda[m] + lambda[l]
                    };
                    let scaled = u_hat / denom;
                    if scaled == T::ZERO {
                        continue;
                    }
                    for i in 0..p {
                        let vin = v_mat[i + n * p] * scaled;
                        if vin == T::ZERO {
                            continue;
                        }
                        for j in 0..p {
                            let vij = vin * v_mat[j + m * p];
                            if vij == T::ZERO {
                                continue;
                            }
                            for k in 0..p {
                                v[(i * p + j) * p + k] += vij * v_mat[k + l * p];
                            }
                        }
                    }
                }
            }
        }
    }

    fn apply_element(&self, u_local: &[T], v_local: &mut [T]) -> ReedResult<()> {
        match self.dim {
            1 => {
                self.apply_element_1d(u_local, v_local);
                Ok(())
            }
            2 => {
                self.apply_element_2d(u_local, v_local);
                Ok(())
            }
            3 => {
                self.apply_element_3d(u_local, v_local);
                Ok(())
            }
            _ => Err(ReedError::Operator(format!(
                "FDM: unsupported dim {}",
                self.dim
            ))),
        }
    }
}

// ── OperatorTrait impl ─────────────────────────────────────────────

impl<T: Scalar> OperatorTrait<T> for CpuFdmTensorInverseOperator<T> {
    fn global_vector_len_hint(&self) -> Option<usize> {
        Some(self.global_dof)
    }

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
        &self,
        req: OperatorTransposeRequest,
        x: &dyn VectorTrait<T>,
        y: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match req {
            OperatorTransposeRequest::Forward => self.apply(x, y),
            OperatorTransposeRequest::Adjoint => self.apply(x, y), // symmetric
        }
    }

    fn apply_add_with_transpose(
        &self,
        req: OperatorTransposeRequest,
        x: &dyn VectorTrait<T>,
        y: &mut dyn VectorTrait<T>,
    ) -> ReedResult<()> {
        match req {
            OperatorTransposeRequest::Forward => self.apply_add(x, y),
            OperatorTransposeRequest::Adjoint => self.apply_add(x, y),
        }
    }

    fn linear_assemble_diagonal(&self, a: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        if a.len() != self.global_dof {
            return Err(ReedError::Operator(format!(
                "diag: len {} != {}",
                a.len(),
                self.global_dof
            )));
        }
        let s = a.as_mut_slice();
        let mut x = vec![T::ZERO; self.global_dof];
        let mut y = vec![T::ZERO; self.global_dof];
        for j in 0..self.global_dof {
            x[j] = T::ONE;
            self.apply_impl(&x, &mut y, false)?;
            s[j] = y[j];
            x[j] = T::ZERO;
            y[j] = T::ZERO;
        }
        Ok(())
    }

    fn linear_assemble_add_diagonal(&self, a: &mut dyn VectorTrait<T>) -> ReedResult<()> {
        if a.len() != self.global_dof {
            return Err(ReedError::Operator(format!(
                "diag_add: len {} != {}",
                a.len(),
                self.global_dof
            )));
        }
        let s = a.as_mut_slice();
        let mut x = vec![T::ZERO; self.global_dof];
        let mut y = vec![T::ZERO; self.global_dof];
        for j in 0..self.global_dof {
            x[j] = T::ONE;
            self.apply_impl(&x, &mut y, false)?;
            s[j] += y[j];
            x[j] = T::ZERO;
            y[j] = T::ZERO;
        }
        Ok(())
    }
}

impl<T: Scalar> CpuFdmTensorInverseOperator<T> {
    fn apply_impl(&self, xg: &[T], yg: &mut [T], add: bool) -> ReedResult<()> {
        if xg.len() != self.global_dof || yg.len() != self.global_dof {
            return Err(ReedError::Operator(format!(
                "FDM apply: expected len {}, got x={} y={}",
                self.global_dof,
                xg.len(),
                yg.len()
            )));
        }
        let edof = self.restriction.num_dof_per_elem() * self.restriction.num_comp();
        let nelem = self.restriction.num_elements();
        let lsize = nelem * edof;
        let mut ul = vec![T::ZERO; lsize];
        let mut vl = vec![T::ZERO; lsize];

        self.restriction
            .apply(TransposeMode::NoTranspose, xg, &mut ul)?;
        for e in 0..nelem {
            self.apply_element(
                &ul[e * edof..(e + 1) * edof],
                &mut vl[e * edof..(e + 1) * edof],
            )?;
        }
        if !add {
            yg.fill(T::ZERO);
        }
        self.restriction
            .apply(TransposeMode::Transpose, &vl, yg)?;
        Ok(())
    }
}
