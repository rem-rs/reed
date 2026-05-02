use reed_core::{
    elem_restriction::ElemRestrictionTrait,
    enums::TransposeMode,
    error::ReedResult,
    scalar::Scalar,
    CeedInt, ReedError,
};

#[cfg(feature = "parallel")]
const PAR_MIN_ELEMS_PER_TASK: usize = 128;

/// Face element restriction: maps boundary faces to their parent elements.
///
/// From [`ElemRestrictionTrait`]'s perspective, each "element" is a boundary face,
/// so [`num_elements`](ElemRestrictionTrait::num_elements) returns `num_faces` and
/// [`num_dof_per_elem`](ElemRestrictionTrait::num_dof_per_elem) returns `num_dof_per_face`.
///
/// Gathering (NoTranspose) reads global DOFs via `face_offsets` into the face-local
/// L-vector. Scattering (Transpose) routes each face-local DOF through
/// `face_to_elem_local` to its position within the parent element's L-vector, then
/// looks up the global DOF via `elem_offsets`.
#[derive(Clone)]
pub struct CpuFaceElemRestriction<T: Scalar> {
    num_faces: usize,
    num_dof_per_face: usize,
    num_dof_per_elem: usize,
    ncomp: usize,
    num_global_dof: usize,
    face_to_elem: Vec<(usize, usize)>,
    face_offsets: Vec<CeedInt>,
    elem_offsets: Vec<CeedInt>,
    /// Precomputed: face_to_elem_local[f * face_dof_stride + k] = position in element L-vector
    /// where f is the face index and k is the face-local DOF position (0..num_dof_per_face).
    face_to_elem_local: Vec<usize>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> CpuFaceElemRestriction<T> {
    pub fn new(
        num_faces: usize,
        num_dof_per_face: usize,
        num_dof_per_elem: usize,
        ncomp: usize,
        num_global_dof: usize,
        face_to_elem: Vec<(usize, usize)>,
        face_offsets: Vec<CeedInt>,
        elem_offsets: Vec<CeedInt>,
        face_to_elem_local: Vec<usize>,
    ) -> ReedResult<Self> {
        let face_offsets_expected = num_faces * num_dof_per_face * ncomp;
        if face_offsets.len() != face_offsets_expected {
            return Err(ReedError::InvalidArgument(format!(
                "face_offsets length {} != num_faces * num_dof_per_face * ncomp = {}",
                face_offsets.len(),
                face_offsets_expected
            )));
        }
        let face_to_elem_local_expected = num_faces * num_dof_per_face;
        if face_to_elem_local.len() != face_to_elem_local_expected {
            return Err(ReedError::InvalidArgument(format!(
                "face_to_elem_local length {} != num_faces * num_dof_per_face = {}",
                face_to_elem_local.len(),
                face_to_elem_local_expected
            )));
        }
        if face_to_elem.len() != num_faces {
            return Err(ReedError::InvalidArgument(format!(
                "face_to_elem length {} != num_faces {}",
                face_to_elem.len(),
                num_faces
            )));
        }
        let num_parent_elems = elem_offsets.len() / (ncomp * num_dof_per_elem);
        if ncomp * num_dof_per_elem == 0
            || elem_offsets.len() != num_parent_elems * ncomp * num_dof_per_elem
        {
            return Err(ReedError::InvalidArgument(format!(
                "elem_offsets length {} must be a positive multiple of ncomp * num_dof_per_elem = {}",
                elem_offsets.len(),
                ncomp * num_dof_per_elem
            )));
        }
        // Validate face_to_elem indices are in range
        for (face, &(elem_id, _)) in face_to_elem.iter().enumerate() {
            if elem_id >= num_parent_elems {
                return Err(ReedError::InvalidArgument(format!(
                    "face_to_elem[{}] elem_id {} out of range (num_parent_elems = {})",
                    face, elem_id, num_parent_elems
                )));
            }
        }
        // Validate face_to_elem_local indices are in range
        for (idx, &elocal) in face_to_elem_local.iter().enumerate() {
            if elocal >= num_dof_per_elem {
                return Err(ReedError::InvalidArgument(format!(
                    "face_to_elem_local[{}] = {} out of range (num_dof_per_elem = {})",
                    idx, elocal, num_dof_per_elem
                )));
            }
        }
        Ok(Self {
            num_faces,
            num_dof_per_face,
            num_dof_per_elem,
            ncomp,
            num_global_dof,
            face_to_elem,
            face_offsets,
            elem_offsets,
            face_to_elem_local,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T: Scalar> ElemRestrictionTrait<T> for CpuFaceElemRestriction<T> {
    fn num_elements(&self) -> usize {
        self.num_faces
    }

    fn num_dof_per_elem(&self) -> usize {
        self.num_dof_per_face
    }

    fn num_global_dof(&self) -> usize {
        self.num_global_dof
    }

    fn num_comp(&self) -> usize {
        self.ncomp
    }

    fn apply(&self, t_mode: TransposeMode, u: &[T], v: &mut [T]) -> ReedResult<()> {
        let local_size = self.local_size();
        let face_chunk = self.ncomp * self.num_dof_per_face;

        match t_mode {
            TransposeMode::NoTranspose => {
                if u.len() != self.num_global_dof {
                    return Err(ReedError::ElemRestriction(format!(
                        "input length {} != global size {}",
                        u.len(),
                        self.num_global_dof
                    )));
                }
                if v.len() != local_size {
                    return Err(ReedError::ElemRestriction(format!(
                        "output length {} != local size {}",
                        v.len(),
                        local_size
                    )));
                }

                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    v.par_chunks_mut(face_chunk).enumerate().try_for_each(
                        |(face, v_face)| -> ReedResult<()> {
                            let face_base = face * face_chunk;
                            for k in 0..face_chunk {
                                let g = self.face_offsets[face_base + k];
                                if g < 0 {
                                    return Err(ReedError::ElemRestriction(format!(
                                        "negative face offset {} at face {}, local {}",
                                        g, face, k
                                    )));
                                }
                                let g = g as usize;
                                if g >= self.num_global_dof {
                                    return Err(ReedError::ElemRestriction(format!(
                                        "global index {} out of bounds for lsize {}",
                                        g, self.num_global_dof
                                    )));
                                }
                                v_face[k] = u[g];
                            }
                            Ok(())
                        },
                    )?;
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for face in 0..self.num_faces {
                        let face_base = face * face_chunk;
                        for k in 0..face_chunk {
                            let g = self.face_offsets[face_base + k];
                            if g < 0 {
                                return Err(ReedError::ElemRestriction(format!(
                                    "negative face offset {} at face {}, local {}",
                                    g, face, k
                                )));
                            }
                            let g = g as usize;
                            if g >= self.num_global_dof {
                                return Err(ReedError::ElemRestriction(format!(
                                    "global index {} out of bounds for lsize {}",
                                    g, self.num_global_dof
                                )));
                            }
                            let l = face_base + k;
                            v[l] = u[g];
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
                if v.len() != self.num_global_dof {
                    return Err(ReedError::ElemRestriction(format!(
                        "output length {} != global size {}",
                        v.len(),
                        self.num_global_dof
                    )));
                }

                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    let accum = (0..self.num_faces)
                        .into_par_iter()
                        .with_min_len(PAR_MIN_ELEMS_PER_TASK)
                        .try_fold(
                            || vec![T::ZERO; self.num_global_dof],
                            |mut partial, face| -> ReedResult<Vec<T>> {
                                let (elem_id, _) = self.face_to_elem[face];
                                let face_base = face * face_chunk;
                                for comp in 0..self.ncomp {
                                    for k in 0..self.num_dof_per_face {
                                        let elocal = self.face_to_elem_local
                                            [face * self.num_dof_per_face + k];
                                        let g = self.elem_offsets
                                            [elem_id * self.ncomp * self.num_dof_per_elem
                                                + comp * self.num_dof_per_elem
                                                + elocal];
                                        if g < 0 {
                                            return Err(ReedError::ElemRestriction(format!(
                                                "negative elem offset {} at face {}, elem {}, comp {}, local {}",
                                                g, face, elem_id, comp, k
                                            )));
                                        }
                                        let g = g as usize;
                                        if g >= self.num_global_dof {
                                            return Err(ReedError::ElemRestriction(format!(
                                                "global index {} out of bounds for lsize {}",
                                                g, self.num_global_dof
                                            )));
                                        }
                                        let l = face_base + comp * self.num_dof_per_face + k;
                                        partial[g] += u[l];
                                    }
                                }
                                Ok(partial)
                            },
                        )
                        .try_reduce(
                            || vec![T::ZERO; self.num_global_dof],
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
                    for face in 0..self.num_faces {
                        let (elem_id, _) = self.face_to_elem[face];
                        let face_base = face * face_chunk;
                        for comp in 0..self.ncomp {
                            for k in 0..self.num_dof_per_face {
                                let elocal =
                                    self.face_to_elem_local[face * self.num_dof_per_face + k];
                                let g = self.elem_offsets[elem_id * self.ncomp * self.num_dof_per_elem
                                    + comp * self.num_dof_per_elem
                                    + elocal];
                                if g < 0 {
                                    return Err(ReedError::ElemRestriction(format!(
                                        "negative elem offset {} at face {}, elem {}, comp {}, local {}",
                                        g, face, elem_id, comp, k
                                    )));
                                }
                                let g = g as usize;
                                if g >= self.num_global_dof {
                                    return Err(ReedError::ElemRestriction(format!(
                                        "global index {} out of bounds for lsize {}",
                                        g, self.num_global_dof
                                    )));
                                }
                                let l = face_base + comp * self.num_dof_per_face + k;
                                v[g] += u[l];
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn boxed_clone(&self) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>> {
        Ok(Box::new(self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two faces, each with 2 DOFs, belonging to one 4-DOF element, 1 component.
    ///
    /// Element e0 has DOFs at global indices [10, 20, 30, 40].
    /// Face f0 maps to e0, local face 0, with local positions 0,2 (global 10,30).
    /// Face f1 maps to e0, local face 1, with local positions 1,3 (global 20,40).
    fn simple_two_face_restriction() -> CpuFaceElemRestriction<f64> {
        let num_faces = 2;
        let num_dof_per_face = 2;
        let num_dof_per_elem = 4;
        let ncomp = 1;
        let num_global_dof = 5;

        let face_to_elem = vec![(0, 0), (0, 1)];
        // face_offsets: face 0 DOFs -> global 10,30; face 1 DOFs -> global 20,40
        let face_offsets: Vec<CeedInt> = vec![10, 30, 20, 40];
        // elem_offsets: element 0 DOFs -> global 10,20,30,40
        let elem_offsets: Vec<CeedInt> = vec![10, 20, 30, 40];
        // face_to_elem_local: face 0 local positions -> elem positions 0,2;
        // face 1 local positions -> elem positions 1,3
        let face_to_elem_local = vec![0, 2, 1, 3];

        CpuFaceElemRestriction::new(
            num_faces,
            num_dof_per_face,
            num_dof_per_elem,
            ncomp,
            num_global_dof,
            face_to_elem,
            face_offsets,
            elem_offsets,
            face_to_elem_local,
        )
        .unwrap()
    }

    #[test]
    fn test_num_elements_returns_num_faces() {
        let r = simple_two_face_restriction();
        assert_eq!(r.num_elements(), 2);
        assert_eq!(r.num_dof_per_elem(), 2);
        assert_eq!(r.num_global_dof(), 5);
        assert_eq!(r.num_comp(), 1);
    }

    #[test]
    fn test_face_restriction_gather_scatter_roundtrip() {
        let num_faces = 2;
        let num_dof_per_face = 2;
        let num_dof_per_elem = 4;
        let ncomp = 1;
        let num_global_dof = 4;

        let face_to_elem = vec![(0, 0), (0, 1)];
        let face_offsets: Vec<CeedInt> = vec![0, 2, 1, 3];
        let elem_offsets: Vec<CeedInt> = vec![0, 1, 2, 3];
        let face_to_elem_local = vec![0, 2, 1, 3];

        let r = CpuFaceElemRestriction::<f64>::new(
            num_faces,
            num_dof_per_face,
            num_dof_per_elem,
            ncomp,
            num_global_dof,
            face_to_elem,
            face_offsets,
            elem_offsets,
            face_to_elem_local,
        )
        .unwrap();

        // Gather: global -> face-local
        let u = vec![10.0, 20.0, 30.0, 40.0];
        let mut v = vec![0.0; 4];
        r.apply(TransposeMode::NoTranspose, &u, &mut v)
            .unwrap();
        // face 0: global[0]=10, global[2]=30 -> [10, 30]
        // face 1: global[1]=20, global[3]=40 -> [20, 40]
        assert_eq!(v, vec![10.0, 30.0, 20.0, 40.0]);

        // Scatter: face-local -> global (additive)
        let mut gathered = vec![0.0; 4];
        r.apply(TransposeMode::Transpose, &v, &mut gathered)
            .unwrap();
        // With face_to_elem_local mapping face 0 loc 0->elem loc 0, face 0 loc 1->elem loc 2
        // elem_offsets gives global indices directly
        assert_eq!(gathered, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_face_restriction_ncomp2() {
        let num_faces = 1;
        let num_dof_per_face = 2;
        let num_dof_per_elem = 4;
        let ncomp = 2;
        let num_global_dof = 8;

        // One face belonging to element 0
        let face_to_elem = vec![(0, 0)];
        // face_offsets: length = 1 * 2 * 2 = 4
        let face_offsets: Vec<CeedInt> = vec![0, 2, 4, 6];
        // elem_offsets: length = 1 * 2 * 4 = 8
        let elem_offsets: Vec<CeedInt> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        // face_to_elem_local: length = 1 * 2 = 2
        let face_to_elem_local = vec![0, 3];

        let r = CpuFaceElemRestriction::<f64>::new(
            num_faces,
            num_dof_per_face,
            num_dof_per_elem,
            ncomp,
            num_global_dof,
            face_to_elem,
            face_offsets,
            elem_offsets,
            face_to_elem_local,
        )
        .unwrap();

        assert_eq!(r.num_elements(), 1);
        assert_eq!(r.num_dof_per_elem(), 2);
        assert_eq!(r.num_comp(), 2);
        assert_eq!(r.local_size(), 4);

        // Gather
        let u: Vec<f64> = (0..8).map(|i| (i * 10) as f64).collect();
        // u = [0, 10, 20, 30, 40, 50, 60, 70]
        let mut v = vec![0.0; 4];
        r.apply(TransposeMode::NoTranspose, &u, &mut v).unwrap();
        // face_offsets = [0, 2, 4, 6]
        // v[0] = u[0] = 0, v[1] = u[2] = 20, v[2] = u[4] = 40, v[3] = u[6] = 60
        assert_eq!(v, vec![0.0, 20.0, 40.0, 60.0]);

        // Scatter
        let mut gathered = vec![0.0; 8];
        r.apply(TransposeMode::Transpose, &v, &mut gathered)
            .unwrap();
        // face 0, elem 0
        // comp 0, k=0: elocal=face_to_elem_local[0+0]=0, g=elem_offsets[0+0*4+0]=0, gathered[0]+=v[0]=0
        // comp 0, k=1: elocal=face_to_elem_local[0+1]=3, g=elem_offsets[0+0*4+3]=3, gathered[3]+=v[1]=20
        // comp 1, k=0: elocal=face_to_elem_local[0+0]=0, g=elem_offsets[0+1*4+0]=4, gathered[4]+=v[2]=40
        // comp 1, k=1: elocal=face_to_elem_local[0+1]=3, g=elem_offsets[0+1*4+3]=7, gathered[7]+=v[3]=60
        assert_eq!(gathered, vec![0.0, 0.0, 0.0, 20.0, 40.0, 0.0, 0.0, 60.0]);
    }

    #[test]
    fn test_boxed_clone_works() {
        let r = simple_two_face_restriction();
        let clone = r.boxed_clone().unwrap();
        assert_eq!(clone.num_elements(), r.num_elements());
        assert_eq!(clone.num_dof_per_elem(), r.num_dof_per_elem());
        assert_eq!(clone.num_global_dof(), r.num_global_dof());
        assert_eq!(clone.num_comp(), r.num_comp());
    }

    #[test]
    fn test_assembled_csr_pattern_returns_err() {
        let r = simple_two_face_restriction();
        assert!(r.assembled_csr_pattern().is_err());
    }

    #[test]
    fn test_constructor_validation() {
        // Wrong face_offsets length
        let result = CpuFaceElemRestriction::<f64>::new(
            2, 2, 4, 1, 5,
            vec![(0, 0), (0, 1)],
            vec![0, 1, 2], // wrong: should be length 4
            vec![0, 1, 2, 3],
            vec![0, 2, 1, 3],
        );
        assert!(result.is_err());

        // Wrong face_to_elem_local length
        let result = CpuFaceElemRestriction::<f64>::new(
            2, 2, 4, 1, 5,
            vec![(0, 0), (0, 1)],
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3],
            vec![0, 2, 1], // wrong: should be length 4
        );
        assert!(result.is_err());

        // Wrong face_to_elem length
        let result = CpuFaceElemRestriction::<f64>::new(
            2, 2, 4, 1, 5,
            vec![(0, 0)], // wrong: should be length 2
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3],
            vec![0, 2, 1, 3],
        );
        assert!(result.is_err());
    }
}
