//! Minimal mass pipeline: `Mass1DBuild` then `MassApply`, with libCEED-style `check_ready` and
//! explicit active input/output sizes (asymmetric build vs square apply). See `design_mapping.md` §4.5.1.

use reed::{
    CeedMatrix, CeedMatrixStorage, CpuOperator, FieldVector, OperatorAssembleKind, OperatorTrait,
    QuadMode, Reed,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reed = Reed::<f64>::init("/cpu/self")?;
    let nelem = 2usize;
    let p = 2usize;
    let q = 2usize;
    let ndofs = nelem + 1;

    let node_coords = vec![-1.0, 0.0, 1.0];
    let x_coord = reed.vector_from_slice(&node_coords)?;
    let mut qdata = reed.vector(nelem * q)?;
    qdata.set_value(0.0)?;

    let ind_x = vec![0, 1, 1, 2];
    let ind_u = ind_x.clone();

    let r_x = reed.elem_restriction(nelem, 2, 1, 1, ndofs, &ind_x)?;
    let r_u = reed.elem_restriction(nelem, p, 1, 1, ndofs, &ind_u)?;
    let r_q = reed.strided_elem_restriction(nelem, q, 1, nelem * q, [1, q as i32, q as i32])?;

    let b_x = reed.basis_tensor_h1_lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    let b_u = reed.basis_tensor_h1_lagrange(1, 1, p, q, QuadMode::Gauss)?;

    let build: CpuOperator<'_, f64> = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("Mass1DBuild")?)
        .field("dx", Some(&*r_x), Some(&*b_x), FieldVector::Active)
        .field("weights", None, Some(&*b_x), FieldVector::None)
        .field("qdata", Some(&*r_q), None, FieldVector::Active)
        .build()?;
    build.check_ready()?;
    println!(
        "Mass1DBuild active sizes: input_global={:?} output_global={:?}",
        build.active_input_global_len()?,
        build.active_output_global_len()?
    );
    build.apply(&*x_coord, &mut *qdata)?;

    let op_mass: CpuOperator<'_, f64> = reed
        .operator_builder()
        .qfunction(reed.q_function_by_name("MassApply")?)
        .field("u", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .field("qdata", Some(&*r_q), None, FieldVector::Passive(&*qdata))
        .field("v", Some(&*r_u), Some(&*b_u), FieldVector::Active)
        .build()?;
    op_mass.check_ready()?;
    println!(
        "MassApply active_global_dof_len={:?}",
        op_mass.active_global_dof_len()
    );
    println!(
        "MassApply supports: Diagonal={} Dense={} CSR={} FDM={}",
        op_mass.operator_supports_assemble(OperatorAssembleKind::Diagonal),
        op_mass.operator_supports_assemble(OperatorAssembleKind::LinearNumeric),
        op_mass.operator_supports_assemble(OperatorAssembleKind::LinearCsrNumeric),
        op_mass.operator_supports_assemble(OperatorAssembleKind::FdmElementInverse),
    );

    let mut dense = CeedMatrix::<f64>::dense_col_major_symbolic(ndofs, ndofs)?;
    OperatorTrait::linear_assemble_ceed_matrix(&op_mass, &mut dense)?;
    let dense_once = match dense.storage() {
        CeedMatrixStorage::DenseColMajor { values, .. } => values.clone(),
        _ => unreachable!(),
    };
    OperatorTrait::linear_assemble_add_ceed_matrix(&op_mass, &mut dense)?;
    if let CeedMatrixStorage::DenseColMajor { values, .. } = dense.storage() {
        println!(
            "Dense handle A(0,0)={} after add={}",
            dense_once[0], values[0]
        );
    }

    let u = reed.vector_from_slice(&vec![1.0_f64; ndofs])?;
    let mut v = reed.vector(ndofs)?;
    v.set_value(0.0)?;
    op_mass.apply(&*u, &mut *v)?;

    let mut values = vec![0.0; ndofs];
    v.copy_to_slice(&mut values)?;
    println!("mass operator output: {values:?}");

    let pat = r_u.assembled_csr_pattern()?;
    let mut csr = CeedMatrix::<f64>::csr_symbolic(pat);
    OperatorTrait::linear_assemble_ceed_matrix(&op_mass, &mut csr)?;
    let csr_once = match csr.storage() {
        CeedMatrixStorage::Csr(m) => m.values.clone(),
        _ => unreachable!(),
    };
    OperatorTrait::linear_assemble_add_ceed_matrix(&op_mass, &mut csr)?;
    if let CeedMatrixStorage::Csr(m) = csr.storage() {
        println!("CSR handle A(0)={} after add={}", csr_once[0], m.values[0]);
    }

    Ok(())
}
