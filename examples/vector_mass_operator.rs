//! Vector-valued mass operator example aligned with libCEED vector gallery style.
//!
//! Uses `Mass2DBuild` + `Vector2MassApply` on the CPU backend.

use reed::{FieldVector, OperatorTrait, QuadMode, Reed};

fn build_offsets_2d(nelem_1d: usize, p: usize, ndofs_1d: usize) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(nelem_1d * nelem_1d * p * p);
    for ey in 0..nelem_1d {
        for ex in 0..nelem_1d {
            let sy = ey * (p - 1);
            let sx = ex * (p - 1);
            for jy in 0..p {
                for jx in 0..p {
                    let gi = (sy + jy) * ndofs_1d + (sx + jx);
                    offsets.push(gi as i32);
                }
            }
        }
    }
    offsets
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reed = Reed::<f64>::init("/cpu/self")?;
    let nelem_1d = 2usize;
    let p = 2usize;
    let q = 3usize;
    let nelem = nelem_1d * nelem_1d;
    let ndofs_1d = nelem_1d * (p - 1) + 1;
    let ndofs = ndofs_1d * ndofs_1d;
    let qpts_per_elem = q * q;

    let offsets = build_offsets_2d(nelem_1d, p, ndofs_1d);

    let mut x_comp0 = vec![0.0_f64; ndofs];
    let mut x_comp1 = vec![0.0_f64; ndofs];
    for iy in 0..ndofs_1d {
        for ix in 0..ndofs_1d {
            let i = iy * ndofs_1d + ix;
            x_comp0[i] = -1.0 + 2.0 * ix as f64 / (ndofs_1d - 1) as f64;
            x_comp1[i] = -1.0 + 2.0 * iy as f64 / (ndofs_1d - 1) as f64;
        }
    }
    let mut x_data = Vec::with_capacity(2 * ndofs);
    x_data.extend_from_slice(&x_comp0);
    x_data.extend_from_slice(&x_comp1);
    let x_coord = reed.vector_from_slice(&x_data)?;

    let r_x = reed.elem_restriction(nelem, p * p, 2, ndofs, 2 * ndofs, &offsets)?;
    let r_u = reed.elem_restriction(nelem, p * p, 2, ndofs, 2 * ndofs, &offsets)?;
    let r_q = reed.strided_elem_restriction(
        nelem,
        qpts_per_elem,
        1,
        nelem * qpts_per_elem,
        [1, qpts_per_elem as i32, qpts_per_elem as i32],
    )?;
    let b_x = reed.basis_tensor_h1_lagrange(2, 2, p, q, QuadMode::Gauss)?;
    let b_u = reed.basis_tensor_h1_lagrange(2, 2, p, q, QuadMode::Gauss)?;

    let mut qdata = reed.vector(nelem * qpts_per_elem)?;
    qdata.set_value(0.0)?;
    let op_build = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Mass2DBuild")?)
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()?;
    op_build.apply(&*x_coord, &mut *qdata)?;

    let mut u_data = vec![0.0_f64; 2 * ndofs];
    for i in 0..ndofs {
        u_data[i] = 1.0;
        u_data[ndofs + i] = 2.0;
    }
    let u = reed.vector_from_slice(&u_data)?;
    let mut v = reed.vector(2 * ndofs)?;
    v.set_value(0.0)?;
    let op_mass = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Vector2MassApply")?)
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()?;
    op_mass.apply(&*u, &mut *v)?;

    let ys = v.as_slice();
    let (y0, y1) = ys.split_at(ndofs);
    let norm0: f64 = y0.iter().map(|x| x.abs()).sum();
    let norm1: f64 = y1.iter().map(|x| x.abs()).sum();
    let ratio = norm1 / norm0;

    println!("vector_mass_operator (2D, Vector2MassApply)");
    println!("nelem_1d={nelem_1d}, p={p}, q={q}");
    println!(
        "|y0|_1={:.12e}, |y1|_1={:.12e}, ratio={:.6}",
        norm0, norm1, ratio
    );
    if (ratio - 2.0).abs() > 1e-10 {
        return Err(format!("unexpected component ratio: got {}, expected 2", ratio).into());
    }
    Ok(())
}
