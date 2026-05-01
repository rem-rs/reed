//! libCEED ex1-volume style example using `elem_restriction_at_points` naming.
//!
//! This follows the same mass-build + mass-apply pipeline as `ex1_volume`, but uses
//! `Reed::elem_restriction_at_points` to mirror libCEED AtPoints migration surfaces.

use reed::{FieldVector, OperatorTrait, QuadMode, Reed};
use std::env;

fn parse_arg(args: &[String], key: &str, default: usize) -> usize {
    args.windows(2)
        .find_map(|w| (w[0] == key).then(|| w[1].parse::<usize>().ok()).flatten())
        .unwrap_or(default)
}

fn build_offsets_1d(nelem_1d: usize, p: usize) -> Vec<i32> {
    let mut offsets = Vec::with_capacity(nelem_1d * p);
    for e in 0..nelem_1d {
        let start = e * (p - 1);
        for j in 0..p {
            offsets.push((start + j) as i32);
        }
    }
    offsets
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("Usage: cargo run --example ex1_volume_at_points -- [--nelem N] [--p P] [--q Q]");
        println!("  --nelem N   elements (default 8)");
        println!("  --p P       element polynomial order (default 2)");
        println!("  --q Q       quadrature points (default p+2)");
        return Ok(());
    }

    let nelem = parse_arg(&args, "--nelem", 8);
    let p = parse_arg(&args, "--p", 2);
    let q = parse_arg(&args, "--q", p + 2);
    if nelem < 1 {
        return Err("--nelem must be >= 1".into());
    }
    if p < 2 {
        return Err("--p must be >= 2".into());
    }
    if q < 1 {
        return Err("--q must be >= 1".into());
    }

    let reed = Reed::<f64>::init("/cpu/self")?;
    let ndofs = nelem * (p - 1) + 1;
    let offsets_x = build_offsets_1d(nelem, 2);
    let offsets_u = build_offsets_1d(nelem, p);
    let node_coords: Vec<f64> = (0..ndofs)
        .map(|i| -1.0 + 2.0 * i as f64 / (ndofs - 1) as f64)
        .collect();
    let x_coord = reed.vector_from_slice(&node_coords)?;

    let r_x = reed.elem_restriction_at_points(nelem, 2, 1, 1, ndofs, &offsets_x)?;
    let r_u = reed.elem_restriction_at_points(nelem, p, 1, 1, ndofs, &offsets_u)?;
    let r_q = reed.strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])?;
    let b_x = reed.basis_tensor_h1_lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    let b_u = reed.basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)?;

    let mut qdata = reed.vector(nelem * q)?;
    qdata.set_value(0.0)?;
    let op_build = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Mass1DBuild")?)
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()?;
    op_build.apply(&*x_coord, &mut *qdata)?;

    let u = reed.vector_from_slice(&vec![1.0_f64; ndofs])?;
    let mut v = reed.vector(ndofs)?;
    v.set_value(0.0)?;
    let op_mass = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApplyAtPoints")?)
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()?;
    op_mass.apply(&*u, &mut *v)?;

    let mut values = vec![0.0_f64; ndofs];
    v.copy_to_slice(&mut values)?;
    let computed = values.iter().sum::<f64>();
    let exact = 2.0_f64;
    let error = (computed - exact).abs();

    println!("ex1_volume_at_points (1D)");
    println!("nelem={nelem}, p={p}, q={q}");
    println!("Exact value    : {:.12}", exact);
    println!("Computed value : {:.12}", computed);
    println!("Error          : {:.12e}", error);
    if error > 2.0e3 * f64::EPSILON {
        return Err(format!("error too large: {:.3e}", error).into());
    }
    Ok(())
}
