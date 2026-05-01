//! libCEED-style Poisson example using AtPoints naming on CPU backend.
//!
//! This demonstrates `elem_restriction_at_points` with `Poisson2DBuild` and
//! `Poisson2DApplyAtPoints`, aligned with Reed's AtPoints migration aliases.

use reed::{FieldVector, OperatorTrait, QuadMode, Reed};
use std::env;

fn parse_arg(args: &[String], key: &str, default: usize) -> usize {
    args.windows(2)
        .find_map(|w| (w[0] == key).then(|| w[1].parse::<usize>().ok()).flatten())
        .unwrap_or(default)
}

fn build_offsets_2d(nelem_1d: usize, p: usize, ndofs_1d: usize) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(nelem_1d * nelem_1d * p * p);
    for ey in 0..nelem_1d {
        for ex in 0..nelem_1d {
            let sy = ey * (p - 1);
            let sx = ex * (p - 1);
            for jy in 0..p {
                for jx in 0..p {
                    offsets.push(((sy + jy) * ndofs_1d + (sx + jx)) as i32);
                }
            }
        }
    }
    offsets
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("Usage: cargo run --example poisson_at_points -- [--nelem N] [--p P] [--q Q]");
        println!("  --nelem N   elements per dimension (default 4)");
        println!("  --p P       element polynomial order (default 2)");
        println!("  --q Q       quadrature points per dimension (default p+2)");
        return Ok(());
    }

    let nelem_1d = parse_arg(&args, "--nelem", 4);
    let p = parse_arg(&args, "--p", 2);
    let q = parse_arg(&args, "--q", p + 2);
    if nelem_1d < 1 {
        return Err("--nelem must be >= 1".into());
    }
    if p < 2 {
        return Err("--p must be >= 2".into());
    }
    if q < 1 {
        return Err("--q must be >= 1".into());
    }

    let reed = Reed::<f64>::init("/cpu/self")?;
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

    let r_x = reed.elem_restriction_at_points(nelem, p * p, 2, ndofs, 2 * ndofs, &offsets)?;
    let r_u = reed.elem_restriction_at_points(nelem, p * p, 1, 1, ndofs, &offsets)?;
    let r_q = reed.strided_elem_restriction(
        nelem,
        qpts_per_elem,
        4,
        nelem * qpts_per_elem * 4,
        [1, qpts_per_elem as i32, (qpts_per_elem * 4) as i32],
    )?;
    let b_x = reed.basis_tensor_h1_lagrange(2, 2, p, q, QuadMode::Gauss)?;
    let b_u = reed.basis_tensor_h1_lagrange(2, 1, p, q, QuadMode::Gauss)?;

    let mut qdata = reed.vector(nelem * qpts_per_elem * 4)?;
    qdata.set_value(0.0)?;
    let op_build = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Poisson2DBuild")?)
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()?;
    op_build.apply(&*x_coord, &mut *qdata)?;

    let mut u_vals = vec![0.0_f64; ndofs];
    for i in 0..ndofs {
        u_vals[i] = x_comp0[i] + x_comp1[i];
    }
    let u = reed.vector_from_slice(&u_vals)?;
    let mut v = reed.vector(ndofs)?;
    v.set_value(0.0)?;
    let op_apply = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Poisson2DApplyAtPoints")?)
        .field("du", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("dv", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()?;
    op_apply.apply(&*u, &mut *v)?;

    let norm1: f64 = v.as_slice().iter().map(|x| x.abs()).sum();
    println!("poisson_at_points (2D)");
    println!("nelem_1d={nelem_1d}, p={p}, q={q}");
    println!("output norm1 = {:.12e}", norm1);
    if !(norm1.is_finite() && norm1 > 0.0) {
        return Err("unexpected non-positive or non-finite norm".into());
    }
    Ok(())
}
