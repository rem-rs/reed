use reed_core::{
    csr::csr_sparsity_from_offset_restriction, elem_restriction::ElemRestrictionTrait,
    enums::TransposeMode, error::ReedResult, scalar::Scalar, CsrPattern, ReedError,
};

#[cfg(feature = "parallel")]
const PAR_MIN_ELEMS_PER_TASK: usize = 128;

#[derive(Clone)]
enum RestrictionLayout {
    Offset {
        offsets: Vec<i32>,
        compstride: usize,
    },
    Strided {
        strides: [i32; 3],
    },
}

#[derive(Clone)]
pub struct CpuElemRestriction<T: Scalar> {
    nelem: usize,
    elemsize: usize,
    ncomp: usize,
    lsize: usize,
    layout: RestrictionLayout,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> CpuElemRestriction<T> {
    pub fn new_offset(
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i32],
    ) -> ReedResult<Self> {
        if offsets.len() != nelem * elemsize {
            return Err(ReedError::InvalidArgument(format!(
                "offsets length {} != nelem*elemsize {}",
                offsets.len(),
                nelem * elemsize
            )));
        }
        Ok(Self {
            nelem,
            elemsize,
            ncomp,
            lsize,
            layout: RestrictionLayout::Offset {
                offsets: offsets.to_vec(),
                compstride,
            },
            _marker: std::marker::PhantomData,
        })
    }

    pub fn new_strided(
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
    ) -> ReedResult<Self> {
        Ok(Self {
            nelem,
            elemsize,
            ncomp,
            lsize,
            layout: RestrictionLayout::Strided { strides },
            _marker: std::marker::PhantomData,
        })
    }

    fn local_index(&self, elem: usize, comp: usize, local: usize) -> usize {
        ((elem * self.ncomp + comp) * self.elemsize) + local
    }

    #[cfg(not(feature = "parallel"))]
    fn transpose_offset_serial(
        &self,
        offsets: &[i32],
        compstride: usize,
        u: &[T],
        v: &mut [T],
    ) -> ReedResult<()> {
        for elem in 0..self.nelem {
            let elem_offsets = &offsets[elem * self.elemsize..(elem + 1) * self.elemsize];
            for comp in 0..self.ncomp {
                let comp_base = comp * compstride;
                for (local, &base) in elem_offsets.iter().enumerate() {
                    if base < 0 {
                        return Err(ReedError::ElemRestriction(format!(
                            "negative offset {} at element {}, local {}",
                            base, elem, local
                        )));
                    }
                    let g = base as usize + comp_base;
                    if g >= self.lsize {
                        return Err(ReedError::ElemRestriction(format!(
                            "global index {} out of bounds for lsize {}",
                            g, self.lsize
                        )));
                    }
                    let l = self.local_index(elem, comp, local);
                    v[g] += u[l];
                }
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    fn transpose_strided_serial(&self, strides: [i32; 3], u: &[T], v: &mut [T]) -> ReedResult<()> {
        for elem in 0..self.nelem {
            for comp in 0..self.ncomp {
                for local in 0..self.elemsize {
                    let index = local as i32 * strides[0]
                        + comp as i32 * strides[1]
                        + elem as i32 * strides[2];
                    if index < 0 {
                        return Err(ReedError::ElemRestriction(format!(
                            "negative strided index {} at element {}, comp {}, local {}",
                            index, elem, comp, local
                        )));
                    }
                    let g = index as usize;
                    if g >= self.lsize {
                        return Err(ReedError::ElemRestriction(format!(
                            "global index {} out of bounds for lsize {}",
                            g, self.lsize
                        )));
                    }
                    let l = self.local_index(elem, comp, local);
                    v[g] += u[l];
                }
            }
        }
        Ok(())
    }
}

impl<T: Scalar> ElemRestrictionTrait<T> for CpuElemRestriction<T> {
    fn num_elements(&self) -> usize {
        self.nelem
    }

    fn num_dof_per_elem(&self) -> usize {
        self.elemsize
    }

    fn num_global_dof(&self) -> usize {
        self.lsize
    }

    fn num_comp(&self) -> usize {
        self.ncomp
    }

    fn apply(&self, t_mode: TransposeMode, u: &[T], v: &mut [T]) -> ReedResult<()> {
        let local_size = self.local_size();
        match t_mode {
            TransposeMode::NoTranspose => {
                if u.len() != self.lsize {
                    return Err(ReedError::ElemRestriction(format!(
                        "input length {} != global size {}",
                        u.len(),
                        self.lsize
                    )));
                }
                if v.len() != local_size {
                    return Err(ReedError::ElemRestriction(format!(
                        "output length {} != local size {}",
                        v.len(),
                        local_size
                    )));
                }
                match &self.layout {
                    RestrictionLayout::Offset {
                        offsets,
                        compstride,
                    } => {
                        #[cfg(feature = "parallel")]
                        {
                            use rayon::prelude::*;
                            let elem_chunk = self.ncomp * self.elemsize;
                            v.par_chunks_mut(elem_chunk).enumerate().try_for_each(
                                |(elem, v_elem)| -> ReedResult<()> {
                                    let elem_offsets =
                                        &offsets[elem * self.elemsize..(elem + 1) * self.elemsize];
                                    for comp in 0..self.ncomp {
                                        let comp_base = comp * *compstride;
                                        let v_comp = &mut v_elem
                                            [comp * self.elemsize..(comp + 1) * self.elemsize];
                                        for (local, dst) in v_comp.iter_mut().enumerate() {
                                            let base = elem_offsets[local];
                                            if base < 0 {
                                                return Err(ReedError::ElemRestriction(format!(
                                                    "negative offset {} at element {}, local {}",
                                                    base, elem, local
                                                )));
                                            }
                                            let g = base as usize + comp_base;
                                            if g >= self.lsize {
                                                return Err(ReedError::ElemRestriction(format!(
                                                    "global index {} out of bounds for lsize {}",
                                                    g, self.lsize
                                                )));
                                            }
                                            *dst = u[g];
                                        }
                                    }
                                    Ok(())
                                },
                            )?;
                        }
                        #[cfg(not(feature = "parallel"))]
                        {
                            for elem in 0..self.nelem {
                                let elem_offsets =
                                    &offsets[elem * self.elemsize..(elem + 1) * self.elemsize];
                                for comp in 0..self.ncomp {
                                    let comp_base = comp * *compstride;
                                    for (local, &base) in elem_offsets.iter().enumerate() {
                                        if base < 0 {
                                            return Err(ReedError::ElemRestriction(format!(
                                                "negative offset {} at element {}, local {}",
                                                base, elem, local
                                            )));
                                        }
                                        let g = base as usize + comp_base;
                                        if g >= self.lsize {
                                            return Err(ReedError::ElemRestriction(format!(
                                                "global index {} out of bounds for lsize {}",
                                                g, self.lsize
                                            )));
                                        }
                                        let l = self.local_index(elem, comp, local);
                                        v[l] = u[g];
                                    }
                                }
                            }
                        }
                    }
                    RestrictionLayout::Strided { strides } => {
                        #[cfg(feature = "parallel")]
                        {
                            use rayon::prelude::*;
                            let elem_chunk = self.ncomp * self.elemsize;
                            v.par_chunks_mut(elem_chunk).enumerate().try_for_each(
                                |(elem, v_elem)| -> ReedResult<()> {
                                    for comp in 0..self.ncomp {
                                        let v_comp =
                                            &mut v_elem[comp * self.elemsize..(comp + 1) * self.elemsize];
                                        for (local, dst) in v_comp.iter_mut().enumerate() {
                                            let index = local as i32 * strides[0]
                                                + comp as i32 * strides[1]
                                                + elem as i32 * strides[2];
                                            if index < 0 {
                                                return Err(ReedError::ElemRestriction(format!(
                                                    "negative strided index {} at element {}, comp {}, local {}",
                                                    index, elem, comp, local
                                                )));
                                            }
                                            let g = index as usize;
                                            if g >= self.lsize {
                                                return Err(ReedError::ElemRestriction(format!(
                                                    "global index {} out of bounds for lsize {}",
                                                    g, self.lsize
                                                )));
                                            }
                                            *dst = u[g];
                                        }
                                    }
                                    Ok(())
                                },
                            )?;
                        }
                        #[cfg(not(feature = "parallel"))]
                        {
                            for elem in 0..self.nelem {
                                for comp in 0..self.ncomp {
                                    for local in 0..self.elemsize {
                                        let index = local as i32 * strides[0]
                                            + comp as i32 * strides[1]
                                            + elem as i32 * strides[2];
                                        if index < 0 {
                                            return Err(ReedError::ElemRestriction(format!(
                                                "negative strided index {} at element {}, comp {}, local {}",
                                                index, elem, comp, local
                                            )));
                                        }
                                        let g = index as usize;
                                        if g >= self.lsize {
                                            return Err(ReedError::ElemRestriction(format!(
                                                "global index {} out of bounds for lsize {}",
                                                g, self.lsize
                                            )));
                                        }
                                        let l = self.local_index(elem, comp, local);
                                        v[l] = u[g];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            TransposeMode::Transpose => {
                if u.len() != local_size {
                    return Err(ReedError::ElemRestriction(format!(
                        "input length {} != local size {}",
                        u.len(),
                        local_size
                    )));
                }
                if v.len() != self.lsize {
                    return Err(ReedError::ElemRestriction(format!(
                        "output length {} != global size {}",
                        v.len(),
                        self.lsize
                    )));
                }
                match &self.layout {
                    RestrictionLayout::Offset {
                        offsets,
                        compstride,
                    } => {
                        #[cfg(feature = "parallel")]
                        {
                            use rayon::prelude::*;
                            let accum = (0..self.nelem)
                                .into_par_iter()
                                .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                                .try_fold(
                                    || vec![T::ZERO; self.lsize],
                                    |mut partial, elem| -> ReedResult<Vec<T>> {
                                        let elem_offsets = &offsets
                                            [elem * self.elemsize..(elem + 1) * self.elemsize];
                                        for comp in 0..self.ncomp {
                                            let comp_base = comp * *compstride;
                                            for (local, &base) in elem_offsets.iter().enumerate() {
                                                if base < 0 {
                                                    return Err(ReedError::ElemRestriction(format!(
                                                        "negative offset {} at element {}, local {}",
                                                        base, elem, local
                                                    )));
                                                }
                                                let g = base as usize + comp_base;
                                                if g >= self.lsize {
                                                    return Err(ReedError::ElemRestriction(format!(
                                                        "global index {} out of bounds for lsize {}",
                                                        g, self.lsize
                                                    )));
                                                }
                                                let l = self.local_index(elem, comp, local);
                                                partial[g] += u[l];
                                            }
                                        }
                                        Ok(partial)
                                    },
                                )
                                .try_reduce(
                                    || vec![T::ZERO; self.lsize],
                                    |mut left, right| -> ReedResult<Vec<T>> {
                                        for (dst, src) in left.iter_mut().zip(right.into_iter()) {
                                            *dst += src;
                                        }
                                        Ok(left)
                                    },
                                )?;
                            for (dst, src) in v.iter_mut().zip(accum.into_iter()) {
                                *dst += src;
                            }
                        }
                        #[cfg(not(feature = "parallel"))]
                        {
                            self.transpose_offset_serial(offsets, *compstride, u, v)?;
                        }
                    }
                    RestrictionLayout::Strided { strides } => {
                        #[cfg(feature = "parallel")]
                        {
                            use rayon::prelude::*;
                            let accum = (0..self.nelem)
                                .into_par_iter()
                                .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                                .try_fold(
                                    || vec![T::ZERO; self.lsize],
                                    |mut partial, elem| -> ReedResult<Vec<T>> {
                                        for comp in 0..self.ncomp {
                                            for local in 0..self.elemsize {
                                                let index = local as i32 * strides[0]
                                                    + comp as i32 * strides[1]
                                                    + elem as i32 * strides[2];
                                                if index < 0 {
                                                    return Err(ReedError::ElemRestriction(format!(
                                                        "negative strided index {} at element {}, comp {}, local {}",
                                                        index, elem, comp, local
                                                    )));
                                                }
                                                let g = index as usize;
                                                if g >= self.lsize {
                                                    return Err(ReedError::ElemRestriction(format!(
                                                        "global index {} out of bounds for lsize {}",
                                                        g, self.lsize
                                                    )));
                                                }
                                                let l = self.local_index(elem, comp, local);
                                                partial[g] += u[l];
                                            }
                                        }
                                        Ok(partial)
                                    },
                                )
                                .try_reduce(
                                    || vec![T::ZERO; self.lsize],
                                    |mut left, right| -> ReedResult<Vec<T>> {
                                        for (dst, src) in left.iter_mut().zip(right.into_iter()) {
                                            *dst += src;
                                        }
                                        Ok(left)
                                    },
                                )?;
                            for (dst, src) in v.iter_mut().zip(accum.into_iter()) {
                                *dst += src;
                            }
                        }
                        #[cfg(not(feature = "parallel"))]
                        {
                            self.transpose_strided_serial(*strides, u, v)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn assembled_csr_pattern(&self) -> ReedResult<CsrPattern> {
        match &self.layout {
            RestrictionLayout::Offset {
                offsets,
                compstride,
            } => csr_sparsity_from_offset_restriction(
                self.nelem,
                self.elemsize,
                self.ncomp,
                *compstride,
                self.lsize,
                offsets,
            ),
            RestrictionLayout::Strided { .. } => Err(ReedError::ElemRestriction(
                "assembled_csr_pattern: strided layout has no explicit per-element L-node indices"
                    .into(),
            )),
        }
    }

    fn boxed_clone(&self) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Ok(Box::new(self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reed_core::enums::TransposeMode;

    #[test]
    fn assembled_csr_pattern_line_two_p1_elements() {
        let r = CpuElemRestriction::<f64>::new_offset(2, 2, 1, 1, 3, &[0, 1, 1, 2]).unwrap();
        let p = r.assembled_csr_pattern().unwrap();
        assert_eq!(p.nrows, 3);
        assert_eq!(p.nnz(), 7);
    }

    #[test]
    fn assembled_csr_pattern_ncomp2_one_segment() {
        let r = CpuElemRestriction::<f64>::new_offset(1, 2, 2, 3, 10, &[0, 1]).unwrap();
        let p = r.assembled_csr_pattern().unwrap();
        assert_eq!(p.nnz(), 16);
    }

    #[test]
    fn test_offset_restriction_roundtrip() {
        let r = CpuElemRestriction::<f64>::new_offset(2, 2, 1, 1, 3, &[0, 1, 1, 2]).unwrap();
        let global = vec![10.0, 20.0, 30.0];
        let mut local = vec![0.0; 4];
        r.apply(TransposeMode::NoTranspose, &global, &mut local)
            .unwrap();
        assert_eq!(local, vec![10.0, 20.0, 20.0, 30.0]);

        let mut gathered = vec![0.0; 3];
        r.apply(TransposeMode::Transpose, &local, &mut gathered)
            .unwrap();
        assert_eq!(gathered, vec![10.0, 40.0, 30.0]);
    }
}
