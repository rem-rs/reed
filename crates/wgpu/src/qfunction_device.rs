//! Device-side QFunction design (Reed / WGSL / WGPU).
//!
//! libCEED models a QFunction as a pointwise kernel over quadrature points with packed I/O and
//! optional context bytes. Reed preserves that on CPU via [`reed_core::QFunctionTrait`]. This
//! module documents the **GPU path** and contains gallery-compatible `f32` compute kernels.
//! Supported names are also returned from [`reed_core::Backend::try_device_q_function_by_name`]
//! on the wgpu backend (surfaced by the workspace `reed` crate’s `Reed::q_function_by_name`).
//! Restriction/basis staging in [`reed_cpu::operator::CpuOperator`] stays on the host; the
//! QFunction `apply` step runs WGSL then readbacks.
//!
//! ## 1. Packed I/O (align with `CpuOperator` staging)
//!
//! Each QFunction input/output slot is a **dense 1D buffer** in quadrature order, element-major:
//! index `e * nqp * C + q * C + c` where `C` is that slot’s `num_comp`. This matches the `Vec<T>`
//! layout passed to [`reed_core::QFunctionTrait::apply`] today. Passive fields (e.g. `qdata`) use
//! the same packing as **read-only** storage bindings.
//!
//! ## 2. `QFunctionContext`
//!
//! Host-owned bytes are copied each dispatch into either a **small uniform** block (typical
//! constants, few scalars) or a **read-only storage** buffer for larger tables. After the host
//! updates context, enqueue a `queue.write_buffer` (or mapped upload) before the operator’s
//! compute pass. [`reed_core::ClosureQFunction`] remains CPU-only unless a compiled-kernel
//! registration path is added later.
//!
//! ## 3. WGSL sources (incremental)
//!
//! - **Gallery**: hand-written (or `include!` template) WGSL per named kernel; parity-tested
//!   against CPU gallery.
//! - **Bind layout**: one group can hold uniform ctx + several `storage, read` inputs + several
//!   `storage, read_write` outputs; larger operators may split passes if binding limits bite.
//!
//! ## 4. Operator integration
//!
//! Restriction/basis still produce host `Vec` quadrature data in [`reed_cpu::operator::CpuOperator`].
//! A device QFunction **uploads those slices → SSBO**, runs compute, then **readbacks** into the
//! output `Vec` before restriction transpose / basis transpose on the host (or future GPU scatter).
//! For [`QFunctionTrait::apply_operator_transpose`], scalar mass / 1D Poisson apply reuse the
//! `mass_apply_qp_transpose` WGSL path. Vector2/3 MassApply and Vector2/3 Poisson1DApply broadcast
//! per-point `qdata` and reuse the same kernel (`GpuRuntime::mass_apply_qp_transpose_broadcast_scalar_qdata_f32_host`).
//! Poisson2D/3D and stacked vector Poisson2D transposes use dedicated WGSL accumulate kernels on
//! [`GpuRuntime`] (same four-slot bind layout as the forward applies).
//! [`IdentityF32Wgpu`] / [`ScaleF32Wgpu`] transpose uses the unary-layout accumulate shaders
//! (`qf_identity_transpose_accumulate_f32` / `qf_scale_transpose_accumulate_f32`).
//! [`IdentityScalarF32Wgpu`] uses `qf_identity_scalar_gather_f32` / `qf_identity_scalar_transpose_accumulate_f32`
//! on the same unary bind layout (packed input `q * ncomp`, packed output or cotangent slot length `q`).
//!
//! ## 5. Bring-up types
//!
//! - [`QFunctionPrototypeScaleF32`] — `out[i] = scale * in[i]` (no trait).
//! - [`MassApplyF32Wgpu`] — gallery-compatible scalar [`MassApply`](reed_cpu::MassApply); implements
//!   [`QFunctionTrait`] for `f32` and can be passed to [`reed_cpu::OperatorBuilder::qfunction`].
//! - [`MassApplyInterpTimesWeightF32Wgpu`] — [`MassApplyInterpTimesWeight`](reed_cpu::MassApplyInterpTimesWeight)
//!   (same multiply / transpose as mass apply).
//! - [`Mass1DBuildF32Wgpu`] / [`Mass2DBuildF32Wgpu`] / [`Mass3DBuildF32Wgpu`] — Jacobian-weight `qdata`
//!   gallery builds (`qf_mass1d_build_f32` / `2d` / `3d`).
//! - [`Poisson1DBuildF32Wgpu`] / [`Poisson2DBuildF32Wgpu`] / [`Poisson3DBuildF32Wgpu`] — Poisson stiffness
//!   `qdata` gallery builds; singular Jacobians are rejected on the host before GPU (`qf_poisson*_build_f32`).
//! - [`Poisson1DApplyF32Wgpu`] — gallery [`Poisson1DApply`](reed_cpu::Poisson1DApply) (`dv = du *
//!   qdata`); uses the same [`GpuRuntime`] pointwise multiply pipeline as [`MassApplyF32Wgpu`].
//! - [`Poisson2DApplyF32Wgpu`] — gallery [`Poisson2DApply`](reed_cpu::Poisson2DApply); 2×2 block per
//!   point via [`GpuRuntime::qfunction_poisson2d_apply_pipeline`] (same four-slot bind layout).
//! - [`Poisson3DApplyF32Wgpu`] — gallery [`Poisson3DApply`](reed_cpu::Poisson3DApply); 3×3 block per
//!   point via [`GpuRuntime::qfunction_poisson3d_apply_pipeline`].
//! - [`IdentityF32Wgpu`] / [`IdentityScalarF32Wgpu`] / [`ScaleF32Wgpu`] — gallery [`Identity`](reed_cpu::Identity),
//!   [`IdentityScalar`](reed_cpu::IdentityScalar), and [`Scale`](reed_cpu::Scale) on `f32` via
//!   [`GpuRuntime::qfunction_unary_layout`] (and identity-scalar gather / transpose entry points).
//! - [`Vector2MassApplyF32Wgpu`] — gallery [`Vector2MassApply`](reed_cpu::Vector2MassApply) (`u`,`v`
//!   have `2` components per quadrature point); uses [`GpuRuntime::qfunction_vector2_mass_apply_pipeline`]
//!   with the same bind layout as scalar pointwise multiply.
//! - [`Vector3MassApplyF32Wgpu`] — gallery [`Vector3MassApply`](reed_cpu::Vector3MassApply); uses
//!   [`GpuRuntime::qfunction_vector3_mass_apply_pipeline`] with the same bind layout.
//! - [`Vector2Poisson1DApplyF32Wgpu`] / [`Vector3Poisson1DApplyF32Wgpu`] — gallery
//!   [`Vector2Poisson1DApply`](reed_cpu::Vector2Poisson1DApply) /
//!   [`Vector3Poisson1DApply`](reed_cpu::Vector3Poisson1DApply); numerically the same per-point
//!   scaling as vector mass (reuse `vector2_mass_apply_f32` / `vector3_mass_apply_f32`).
//! - [`Vector2Poisson2DApplyF32Wgpu`] / [`Vector3Poisson2DApplyF32Wgpu`] — gallery
//!   [`Vector2Poisson2DApply`](reed_cpu::Vector2Poisson2DApply) /
//!   [`Vector3Poisson2DApply`](reed_cpu::Vector3Poisson2DApply); shared 2×2 stiffness per point
//!   (`vector2_poisson2d_apply_f32` / `vector3_poisson2d_apply_f32`).
//! - [`Vector3Poisson3DApplyF32Wgpu`] — gallery [`Vector3Poisson3DApply`](reed_cpu::Vector3Poisson3DApply);
//!   shared 3×3 `qdata` on three stacked 3-gradients (`vector3_poisson3d_apply_f32` / transpose).
//! - [`Vec2DotF32Wgpu`] / [`Vec3DotF32Wgpu`] — gallery [`Vec2Dot`](reed_cpu::Vec2Dot) /
//!   [`Vec3Dot`](reed_cpu::Vec3Dot); per-point dot (`qf_vec2_dot_f32` / `qf_vec3_dot_f32`).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use reed_core::{qfunction::QFunctionTrait, QFunctionContext, ReedError, ReedResult};
use wgpu::util::DeviceExt;

use crate::runtime::GpuRuntime;

/// Transpose for Vector2/3 MassApply and Vector2/3 Poisson1DApply (`components` ∈ {2, 3}).
fn transpose_qp_broadcast_components_f32(
    runtime: &GpuRuntime,
    q: usize,
    components: usize,
    _ctx: &[u8],
    output_cotangents: &[&[f32]],
    input_cotangents: &mut [&mut [f32]],
) -> ReedResult<()> {
    if output_cotangents.len() != 1 || input_cotangents.len() != 2 {
        return Err(ReedError::QFunction(
            "transpose expects 1 output cotangent and 2 input cotangent buffers".into(),
        ));
    }
    let dv = output_cotangents[0];
    if dv.len() != q * components {
        return Err(ReedError::QFunction(
            "transpose: output cotangent length mismatch".into(),
        ));
    }
    let (du_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
    let du = &mut du_buf[0];
    let qdata: &[f32] = &qdata_fwd[0];
    if du.len() != q * components || qdata.len() != q {
        return Err(ReedError::QFunction(
            "transpose: input cotangent / qdata length mismatch".into(),
        ));
    }
    runtime.mass_apply_qp_transpose_broadcast_scalar_qdata_f32_host(dv, qdata, du, components)
}

const WGSL_SCALE_PROTO: &str = r#"
struct QfProtoParams {
    num_q: u32,
    _pad0: u32,
    scale: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> qp: QfProtoParams;
@group(0) @binding(1) var<storage, read> q_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_out: array<f32>;

@compute @workgroup_size(256)
fn qf_scale_proto(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    q_out[i] = qp.scale * q_in[i];
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QfProtoParamsHost {
    num_q: u32,
    _pad0: u32,
    scale: f32,
    _pad1: f32,
}

/// One-input one-output `f32` QFunction prototype: pointwise multiply by `scale`.
///
/// Intended for bring-up only; real gallery kernels will share the same buffer/bind conventions
/// documented in this module’s crate-level docs.
pub struct QFunctionPrototypeScaleF32 {
    runtime: Arc<GpuRuntime>,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl QFunctionPrototypeScaleF32 {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let device = &runtime.device;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reed-qf-proto-scale"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-proto-scale"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WGSL_SCALE_PROTO)),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reed-qf-proto-scale-pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reed-qf-proto-scale-pipe"),
            layout: Some(&pipeline_layout),
            module: &sm,
            entry_point: "qf_scale_proto",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Ok(Self {
            runtime,
            layout,
            pipeline,
        })
    }

    /// Writes `out[i] = scale * in[i]` for `i in 0..num_q`.
    ///
    /// `input` / `output` slices must have length at least `num_q`.
    pub fn apply(
        &self,
        num_q: usize,
        scale: f32,
        input: &[f32],
        output: &mut [f32],
    ) -> ReedResult<()> {
        if num_q == 0 {
            return Ok(());
        }
        if input.len() < num_q || output.len() < num_q {
            return Err(ReedError::QFunction(format!(
                "apply: need len >= num_q ({num_q}), got in={} out={}",
                input.len(),
                output.len()
            )));
        }
        let device = &self.runtime.device;
        let queue = &self.runtime.queue;
        let bytes = (num_q * std::mem::size_of::<f32>()) as u64;

        let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reed-qf-proto-in"),
            contents: bytemuck::cast_slice(&input[..num_q]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-qf-proto-out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = QfProtoParamsHost {
            num_q: num_q as u32,
            _pad0: 0,
            scale,
            _pad1: 0.0,
        };
        let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reed-qf-proto-uni"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reed-qf-proto-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reed-qf-proto-enc"),
        });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reed-qf-proto-pass"),
                ..Default::default()
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind, &[]);
            let wg = 256u32;
            let groups = ((num_q as u32) + wg - 1) / wg;
            cpass.dispatch_workgroups(groups, 1, 1);
        }

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-qf-proto-rb"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, bytes);
        queue.submit(std::iter::once(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
            .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
        {
            let data = slice.get_mapped_range();
            let mapped: &[f32] = bytemuck::cast_slice(&data);
            output[..num_q].copy_from_slice(&mapped[..num_q]);
        }
        readback.unmap();
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QfPointwiseMulParamsHost {
    num_q: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// `out[i] = in0[i] * in1[i]` using [`GpuRuntime::qfunction_pointwise_mul_pipeline`].
pub(crate) fn dispatch_qf_pointwise_mul_f32(
    runtime: &GpuRuntime,
    q: usize,
    in0: &[f32],
    in1: &[f32],
    out: &mut [f32],
) -> ReedResult<()> {
    if q == 0 {
        return Ok(());
    }
    if in0.len() < q || in1.len() < q || out.len() < q {
        return Err(ReedError::QFunction(format!(
            "pointwise_mul: need len >= q ({q}), got in0={} in1={} out={}",
            in0.len(),
            in1.len(),
            out.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (q * std::mem::size_of::<f32>()) as u64;

    let in0_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-pw-in0"),
        contents: bytemuck::cast_slice(&in0[..q]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let in1_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-pw-in1"),
        contents: bytemuck::cast_slice(&in1[..q]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-pw-out"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-pw-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-pw-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: in0_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: in1_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-pw-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-pw-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_pointwise_mul_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((q as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-pw-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        out[..q].copy_from_slice(&mapped[..q]);
    }
    readback.unmap();
    Ok(())
}

/// `w[i] = u[2*i]*v[2*i] + u[2*i+1]*v[2*i+1]` using [`GpuRuntime::qfunction_vec2_dot_pipeline`].
pub(crate) fn dispatch_qf_vec2_dot_f32(
    runtime: &GpuRuntime,
    q: usize,
    u: &[f32],
    v: &[f32],
    w: &mut [f32],
) -> ReedResult<()> {
    if q == 0 {
        return Ok(());
    }
    let need = q * 2;
    if u.len() < need || v.len() < need || w.len() < q {
        return Err(ReedError::QFunction(format!(
            "vec2_dot: need u,v>={need} w>={q}, got u={} v={} w={}",
            u.len(),
            v.len(),
            w.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let w_bytes = (q * std::mem::size_of::<f32>()) as u64;

    let u_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2d-u"),
        contents: bytemuck::cast_slice(&u[..need]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2d-v"),
        contents: bytemuck::cast_slice(&v[..need]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let w_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v2d-w"),
        size: w_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2d-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v2d-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: w_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v2d-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v2d-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vec2_dot_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((q as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v2d-rb"),
        size: w_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&w_buf, 0, &readback, 0, w_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        w[..q].copy_from_slice(&mapped[..q]);
    }
    readback.unmap();
    Ok(())
}

/// `w[i] = u[3*i]*v[3*i] + …` using [`GpuRuntime::qfunction_vec3_dot_pipeline`].
pub(crate) fn dispatch_qf_vec3_dot_f32(
    runtime: &GpuRuntime,
    q: usize,
    u: &[f32],
    v: &[f32],
    w: &mut [f32],
) -> ReedResult<()> {
    if q == 0 {
        return Ok(());
    }
    let need = q * 3;
    if u.len() < need || v.len() < need || w.len() < q {
        return Err(ReedError::QFunction(format!(
            "vec3_dot: need u,v>={need} w>={q}, got u={} v={} w={}",
            u.len(),
            v.len(),
            w.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let w_bytes = (q * std::mem::size_of::<f32>()) as u64;

    let u_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3d-u"),
        contents: bytemuck::cast_slice(&u[..need]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3d-v"),
        contents: bytemuck::cast_slice(&v[..need]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let w_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3d-w"),
        size: w_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3d-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v3d-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: w_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v3d-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v3d-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vec3_dot_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((q as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3d-rb"),
        size: w_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&w_buf, 0, &readback, 0, w_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        w[..q].copy_from_slice(&mapped[..q]);
    }
    readback.unmap();
    Ok(())
}

fn dispatch_qf_mass_build_f32_impl(
    runtime: &GpuRuntime,
    pipeline: &wgpu::ComputePipeline,
    q: usize,
    dx_word_len: usize,
    dx: &[f32],
    weights: &[f32],
    qdata: &mut [f32],
) -> ReedResult<()> {
    if q == 0 {
        return Ok(());
    }
    if dx.len() < dx_word_len || weights.len() < q || qdata.len() < q {
        return Err(ReedError::QFunction(format!(
            "mass_build: need dx>={dx_word_len} w>={q} qdata>={q}, got dx={} w={} qdata={}",
            dx.len(),
            weights.len(),
            qdata.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let w_bytes = (q * std::mem::size_of::<f32>()) as u64;
    let out_bytes = w_bytes;

    let dx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-mbld-dx"),
        contents: bytemuck::cast_slice(&dx[..dx_word_len]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let w_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-mbld-w"),
        contents: bytemuck::cast_slice(&weights[..q]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-mbld-out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-mbld-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-mbld-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dx_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: q_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-mbld-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-mbld-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((q as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-mbld-rb"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&q_buf, 0, &readback, 0, out_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        qdata[..q].copy_from_slice(&mapped[..q]);
    }
    readback.unmap();
    Ok(())
}

/// [`reed_cpu::Mass1DBuild`]: `qdata[i] = |dx[i]| * weights[i]`.
pub(crate) fn dispatch_qf_mass1d_build_f32(
    runtime: &GpuRuntime,
    q: usize,
    dx: &[f32],
    weights: &[f32],
    qdata: &mut [f32],
) -> ReedResult<()> {
    dispatch_qf_mass_build_f32_impl(
        runtime,
        runtime.qfunction_mass1d_build_pipeline(),
        q,
        q,
        dx,
        weights,
        qdata,
    )
}

/// [`reed_cpu::Mass2DBuild`]: `qdata[i] = |det J| * weights[i]` from 2×2 Jacobian row-major per point.
pub(crate) fn dispatch_qf_mass2d_build_f32(
    runtime: &GpuRuntime,
    q: usize,
    dx: &[f32],
    weights: &[f32],
    qdata: &mut [f32],
) -> ReedResult<()> {
    let dx_word_len = q.saturating_mul(4);
    dispatch_qf_mass_build_f32_impl(
        runtime,
        runtime.qfunction_mass2d_build_pipeline(),
        q,
        dx_word_len,
        dx,
        weights,
        qdata,
    )
}

/// [`reed_cpu::Mass3DBuild`]: `qdata[i] = |det J| * weights[i]` from 3×3 Jacobian row-major per point.
pub(crate) fn dispatch_qf_mass3d_build_f32(
    runtime: &GpuRuntime,
    q: usize,
    dx: &[f32],
    weights: &[f32],
    qdata: &mut [f32],
) -> ReedResult<()> {
    let dx_word_len = q.saturating_mul(9);
    dispatch_qf_mass_build_f32_impl(
        runtime,
        runtime.qfunction_mass3d_build_pipeline(),
        q,
        dx_word_len,
        dx,
        weights,
        qdata,
    )
}

/// GPU `f32` implementation of the scalar gallery [`MassApply`](reed_cpu::MassApply): `v = u * qdata`.
///
/// I/O layout matches [`QFunctionTrait::apply`] / `CpuOperator` staging (one scalar per quadrature
/// point, contiguous). Host upload and readback occur inside [`QFunctionTrait::apply`].
pub struct MassApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl MassApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::MassApply::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for MassApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "MassApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let qdata = inputs[1];
        let v = &mut outputs[0];
        dispatch_qf_pointwise_mul_f32(&self.runtime, q, u, qdata, v)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        _ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if output_cotangents.len() != 1 || input_cotangents.len() != 2 {
            return Err(ReedError::QFunction(
                "MassApplyF32Wgpu transpose expects 1 output cotangent and 2 input cotangent buffers"
                    .into(),
            ));
        }
        let dv = output_cotangents[0];
        if dv.len() != q {
            return Err(ReedError::QFunction(
                "MassApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        let (du_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let du = &mut du_buf[0];
        let qdata: &[f32] = &qdata_fwd[0];
        if du.len() != q || qdata.len() != q {
            return Err(ReedError::QFunction(
                "MassApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        self.runtime
            .mass_apply_qp_transpose_accumulate_f32_host(dv, qdata, du)
    }
}

/// GPU `f32` gallery [`MassApplyInterpTimesWeight`](reed_cpu::MassApplyInterpTimesWeight): same kernel as [`MassApplyF32Wgpu`] with weight-slot field names.
pub struct MassApplyInterpTimesWeightF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl MassApplyInterpTimesWeightF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::MassApplyInterpTimesWeight::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for MassApplyInterpTimesWeightF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "MassApplyInterpTimesWeightF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let w = inputs[1];
        let v = &mut outputs[0];
        dispatch_qf_pointwise_mul_f32(&self.runtime, q, u, w, v)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        _ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if output_cotangents.len() != 1 || input_cotangents.len() != 2 {
            return Err(ReedError::QFunction(
                "MassApplyInterpTimesWeightF32Wgpu transpose expects 1 output cotangent and 2 input buffers"
                    .into(),
            ));
        }
        let dv = output_cotangents[0];
        if dv.len() != q {
            return Err(ReedError::QFunction(
                "MassApplyInterpTimesWeightF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        let (du_buf, w_fwd) = input_cotangents.split_at_mut(1);
        let du = &mut du_buf[0];
        let w: &[f32] = &w_fwd[0];
        if du.len() != q || w.len() != q {
            return Err(ReedError::QFunction(
                "MassApplyInterpTimesWeightF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        self.runtime
            .mass_apply_qp_transpose_accumulate_f32_host(dv, w, du)
    }
}

/// GPU `f32` gallery [`Mass1DBuild`](reed_cpu::Mass1DBuild).
pub struct Mass1DBuildF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Mass1DBuildF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Mass1DBuild::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Mass1DBuildF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Mass1DBuildF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        if dx.len() != q || weights.len() != q || qdata.len() != q {
            return Err(ReedError::QFunction(
                "Mass1DBuildF32Wgpu: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_mass1d_build_f32(&self.runtime, q, dx, weights, qdata)
    }
}

/// GPU `f32` gallery [`Mass2DBuild`](reed_cpu::Mass2DBuild).
pub struct Mass2DBuildF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Mass2DBuildF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Mass2DBuild::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Mass2DBuildF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Mass2DBuildF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        if dx.len() != q * 4 || weights.len() != q || qdata.len() != q {
            return Err(ReedError::QFunction(
                "Mass2DBuildF32Wgpu: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_mass2d_build_f32(&self.runtime, q, dx, weights, qdata)
    }
}

/// GPU `f32` gallery [`Mass3DBuild`](reed_cpu::Mass3DBuild).
pub struct Mass3DBuildF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Mass3DBuildF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Mass3DBuild::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Mass3DBuildF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Mass3DBuildF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        if dx.len() != q * 9 || weights.len() != q || qdata.len() != q {
            return Err(ReedError::QFunction(
                "Mass3DBuildF32Wgpu: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_mass3d_build_f32(&self.runtime, q, dx, weights, qdata)
    }
}

/// Matches [`reed_cpu::gallery::helpers::singular_jacobian_tol`] for `f32` (`1e-12` from `f64`).
const POISSON_JACOBIAN_TOL_F32: f32 = 1e-12_f32;

fn poisson1d_build_preflight(q: usize, dx: &[f32]) -> ReedResult<()> {
    for i in 0..q {
        if dx[i].abs() < POISSON_JACOBIAN_TOL_F32 {
            return Err(ReedError::QFunction(
                "Poisson1DBuild encountered near-singular Jacobian".into(),
            ));
        }
    }
    Ok(())
}

fn poisson2d_build_preflight(q: usize, dx: &[f32]) -> ReedResult<()> {
    for i in 0..q {
        let b = i * 4;
        let det_j = dx[b] * dx[b + 3] - dx[b + 1] * dx[b + 2];
        if det_j.abs() < POISSON_JACOBIAN_TOL_F32 {
            return Err(ReedError::QFunction(
                "Poisson2DBuild encountered near-singular Jacobian".into(),
            ));
        }
    }
    Ok(())
}

fn poisson3d_build_preflight(q: usize, dx: &[f32]) -> ReedResult<()> {
    for i in 0..q {
        let b = i * 9;
        let j00 = dx[b];
        let j01 = dx[b + 1];
        let j02 = dx[b + 2];
        let j10 = dx[b + 3];
        let j11 = dx[b + 4];
        let j12 = dx[b + 5];
        let j20 = dx[b + 6];
        let j21 = dx[b + 7];
        let j22 = dx[b + 8];
        let c00 = j11 * j22 - j12 * j21;
        let c01 = -(j10 * j22 - j12 * j20);
        let c02 = j10 * j21 - j11 * j20;
        let det_j = j00 * c00 + j01 * c01 + j02 * c02;
        if det_j.abs() < POISSON_JACOBIAN_TOL_F32 {
            return Err(ReedError::QFunction(
                "Poisson3DBuild encountered near-singular Jacobian".into(),
            ));
        }
    }
    Ok(())
}

fn dispatch_qf_poisson_build_f32_impl(
    runtime: &GpuRuntime,
    pipeline: &wgpu::ComputePipeline,
    q: usize,
    dx_word_len: usize,
    out_word_len: usize,
    dx: &[f32],
    weights: &[f32],
    qdata: &mut [f32],
) -> ReedResult<()> {
    if q == 0 {
        return Ok(());
    }
    if dx.len() < dx_word_len || weights.len() < q || qdata.len() < out_word_len {
        return Err(ReedError::QFunction(format!(
            "poisson_build: need dx>={dx_word_len} w>={q} qdata>={out_word_len}, got dx={} w={} qdata={}",
            dx.len(),
            weights.len(),
            qdata.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let out_bytes = (out_word_len * std::mem::size_of::<f32>()) as u64;

    let dx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-pbld-dx"),
        contents: bytemuck::cast_slice(&dx[..dx_word_len]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let w_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-pbld-w"),
        contents: bytemuck::cast_slice(&weights[..q]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-pbld-out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-pbld-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-pbld-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dx_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: q_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-pbld-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-pbld-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((q as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-pbld-rb"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&q_buf, 0, &readback, 0, out_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        qdata[..out_word_len].copy_from_slice(&mapped[..out_word_len]);
    }
    readback.unmap();
    Ok(())
}

pub(crate) fn dispatch_qf_poisson1d_build_f32(
    runtime: &GpuRuntime,
    q: usize,
    dx: &[f32],
    weights: &[f32],
    qdata: &mut [f32],
) -> ReedResult<()> {
    poisson1d_build_preflight(q, dx)?;
    dispatch_qf_poisson_build_f32_impl(
        runtime,
        runtime.qfunction_poisson1d_build_pipeline(),
        q,
        q,
        q,
        dx,
        weights,
        qdata,
    )
}

pub(crate) fn dispatch_qf_poisson2d_build_f32(
    runtime: &GpuRuntime,
    q: usize,
    dx: &[f32],
    weights: &[f32],
    qdata: &mut [f32],
) -> ReedResult<()> {
    poisson2d_build_preflight(q, dx)?;
    let dx_word_len = q.saturating_mul(4);
    let out_word_len = dx_word_len;
    dispatch_qf_poisson_build_f32_impl(
        runtime,
        runtime.qfunction_poisson2d_build_pipeline(),
        q,
        dx_word_len,
        out_word_len,
        dx,
        weights,
        qdata,
    )
}

pub(crate) fn dispatch_qf_poisson3d_build_f32(
    runtime: &GpuRuntime,
    q: usize,
    dx: &[f32],
    weights: &[f32],
    qdata: &mut [f32],
) -> ReedResult<()> {
    poisson3d_build_preflight(q, dx)?;
    let dx_word_len = q.saturating_mul(9);
    let out_word_len = dx_word_len;
    dispatch_qf_poisson_build_f32_impl(
        runtime,
        runtime.qfunction_poisson3d_build_pipeline(),
        q,
        dx_word_len,
        out_word_len,
        dx,
        weights,
        qdata,
    )
}

/// GPU `f32` gallery [`Poisson1DBuild`](reed_cpu::Poisson1DBuild).
pub struct Poisson1DBuildF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Poisson1DBuildF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Poisson1DBuild::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Poisson1DBuildF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Poisson1DBuildF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        if dx.len() != q || weights.len() != q || qdata.len() != q {
            return Err(ReedError::QFunction(
                "Poisson1DBuildF32Wgpu: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_poisson1d_build_f32(&self.runtime, q, dx, weights, qdata)
    }
}

/// GPU `f32` gallery [`Poisson2DBuild`](reed_cpu::Poisson2DBuild).
pub struct Poisson2DBuildF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Poisson2DBuildF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Poisson2DBuild::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Poisson2DBuildF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Poisson2DBuildF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        if dx.len() != q * 4 || weights.len() != q || qdata.len() != q * 4 {
            return Err(ReedError::QFunction(
                "Poisson2DBuildF32Wgpu: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_poisson2d_build_f32(&self.runtime, q, dx, weights, qdata)
    }
}

/// GPU `f32` gallery [`Poisson3DBuild`](reed_cpu::Poisson3DBuild).
pub struct Poisson3DBuildF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Poisson3DBuildF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Poisson3DBuild::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Poisson3DBuildF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Poisson3DBuildF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        if dx.len() != q * 9 || weights.len() != q || qdata.len() != q * 9 {
            return Err(ReedError::QFunction(
                "Poisson3DBuildF32Wgpu: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_poisson3d_build_f32(&self.runtime, q, dx, weights, qdata)
    }
}

/// GPU `f32` gallery [`Vec2Dot`](reed_cpu::Vec2Dot): scalar `w[i] = dot(u[i], v[i])` per point.
pub struct Vec2DotF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vec2DotF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vec2Dot::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vec2DotF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vec2DotF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let v = inputs[1];
        let w = &mut outputs[0];
        if u.len() != q * 2 || v.len() != q * 2 || w.len() != q {
            return Err(ReedError::QFunction(
                "Vec2DotF32Wgpu: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_vec2_dot_f32(&self.runtime, q, u, v, w)
    }
}

/// GPU `f32` gallery [`Vec3Dot`](reed_cpu::Vec3Dot): scalar `w[i] = dot(u[i], v[i])` per point.
pub struct Vec3DotF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vec3DotF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vec3Dot::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vec3DotF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vec3DotF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let v = inputs[1];
        let w = &mut outputs[0];
        if u.len() != q * 3 || v.len() != q * 3 || w.len() != q {
            return Err(ReedError::QFunction(
                "Vec3DotF32Wgpu: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_vec3_dot_f32(&self.runtime, q, u, v, w)
    }
}

/// GPU `f32` gallery [`Poisson1DApply`](reed_cpu::Poisson1DApply): `dv[i] = du[i] * qdata[i]`.
///
/// For scalar 1D Poisson apply this matches [`MassApplyF32Wgpu`] numerically; this type carries
/// Poisson field names (`du`, `dv`, …) for [`OperatorBuilder`](reed_cpu::OperatorBuilder) while
/// reusing the shared [`GpuRuntime`] pointwise multiply pipeline.
pub struct Poisson1DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Poisson1DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Poisson1DApply::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Poisson1DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Poisson1DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_pointwise_mul_f32(&self.runtime, q, du, qdata, dv)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        _ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if output_cotangents.len() != 1 || input_cotangents.len() != 2 {
            return Err(ReedError::QFunction(
                "Poisson1DApplyF32Wgpu transpose expects 1 output cotangent and 2 input cotangent buffers"
                    .into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q {
            return Err(ReedError::QFunction(
                "Poisson1DApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[f32] = &qdata_fwd[0];
        if ddu.len() != q || qdata.len() != q {
            return Err(ReedError::QFunction(
                "Poisson1DApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        self.runtime
            .mass_apply_qp_transpose_accumulate_f32_host(ddv, qdata, ddu)
    }
}

/// 2D Poisson apply: for each quadrature index `i`, `dv[2*i..]` = `qdata[4*i..4*i+4] * du[2*i..]`.
pub(crate) fn dispatch_qf_poisson2d_apply_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    du: &[f32],
    qdata: &[f32],
    dv: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n_du = num_q.saturating_mul(2);
    let n_qd = num_q.saturating_mul(4);
    if du.len() < n_du || qdata.len() < n_qd || dv.len() < n_du {
        return Err(ReedError::QFunction(format!(
            "poisson2d_apply: need du>={n_du}, qdata>={n_qd}, dv>={n_du}; got du={} qdata={} dv={}",
            du.len(),
            qdata.len(),
            dv.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let du_bytes = (n_du * std::mem::size_of::<f32>()) as u64;

    let du_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p2d-du"),
        contents: bytemuck::cast_slice(&du[..n_du]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p2d-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n_qd]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let dv_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-p2d-dv"),
        size: du_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p2d-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-p2d-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: du_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dv_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-p2d-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-p2d-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_poisson2d_apply_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-p2d-rb"),
        size: du_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&dv_buf, 0, &readback, 0, du_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        dv[..n_du].copy_from_slice(&mapped[..n_du]);
    }
    readback.unmap();
    Ok(())
}

/// Cotangent `ddu += G^T ddv` at each quadrature point ([`reed_cpu::Poisson2DApply::apply_operator_transpose`]).
pub(crate) fn dispatch_qf_poisson2d_transpose_accumulate_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    ddv: &[f32],
    qdata: &[f32],
    ddu: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n_dv = num_q.saturating_mul(2);
    let n_qd = num_q.saturating_mul(4);
    if ddv.len() < n_dv || qdata.len() < n_qd || ddu.len() < n_dv {
        return Err(ReedError::QFunction(format!(
            "poisson2d_transpose: need ddv>={n_dv}, qdata>={n_qd}, ddu>={n_dv}; got ddv={} qdata={} ddu={}",
            ddv.len(),
            qdata.len(),
            ddu.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let out_bytes = (n_dv * std::mem::size_of::<f32>()) as u64;

    let ddv_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p2d-tr-ddv"),
        contents: bytemuck::cast_slice(&ddv[..n_dv]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p2d-tr-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n_qd]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let ddu_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p2d-tr-ddu"),
        contents: bytemuck::cast_slice(&ddu[..n_dv]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p2d-tr-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-p2d-tr-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ddv_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ddu_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-p2d-tr-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-p2d-tr-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_poisson2d_transpose_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-p2d-tr-rb"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&ddu_buf, 0, &readback, 0, out_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        ddu[..n_dv].copy_from_slice(&mapped[..n_dv]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` gallery [`Poisson2DApply`](reed_cpu::Poisson2DApply): stiffness `qdata` (4×`q`) times
/// gradient `du` (2×`q`) into `dv` (2×`q`).
pub struct Poisson2DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Poisson2DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Poisson2DApply::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Poisson2DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Poisson2DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_poisson2d_apply_f32(&self.runtime, q, du, qdata, dv)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        _ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if output_cotangents.len() != 1 || input_cotangents.len() != 2 {
            return Err(ReedError::QFunction(
                "Poisson2DApplyF32Wgpu transpose expects 1 output cotangent and 2 input buffers"
                    .into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q * 2 {
            return Err(ReedError::QFunction(
                "Poisson2DApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[f32] = &qdata_fwd[0];
        if ddu.len() != q * 2 || qdata.len() != q * 4 {
            return Err(ReedError::QFunction(
                "Poisson2DApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_poisson2d_transpose_accumulate_f32(&self.runtime, q, ddv, qdata, ddu)
    }
}

/// 3D Poisson apply: for each quadrature index `i`, `dv[3*i..]` = `qdata[9*i..]` (row-major 3×3) × `du[3*i..]`.
pub(crate) fn dispatch_qf_poisson3d_apply_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    du: &[f32],
    qdata: &[f32],
    dv: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n_du = num_q.saturating_mul(3);
    let n_qd = num_q.saturating_mul(9);
    if du.len() < n_du || qdata.len() < n_qd || dv.len() < n_du {
        return Err(ReedError::QFunction(format!(
            "poisson3d_apply: need du>={n_du}, qdata>={n_qd}, dv>={n_du}; got du={} qdata={} dv={}",
            du.len(),
            qdata.len(),
            dv.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let du_bytes = (n_du * std::mem::size_of::<f32>()) as u64;

    let du_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p3d-du"),
        contents: bytemuck::cast_slice(&du[..n_du]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p3d-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n_qd]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let dv_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-p3d-dv"),
        size: du_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p3d-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-p3d-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: du_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dv_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-p3d-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-p3d-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_poisson3d_apply_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-p3d-rb"),
        size: du_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&dv_buf, 0, &readback, 0, du_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        dv[..n_du].copy_from_slice(&mapped[..n_du]);
    }
    readback.unmap();
    Ok(())
}

pub(crate) fn dispatch_qf_poisson3d_transpose_accumulate_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    ddv: &[f32],
    qdata: &[f32],
    ddu: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n_dv = num_q.saturating_mul(3);
    let n_qd = num_q.saturating_mul(9);
    if ddv.len() < n_dv || qdata.len() < n_qd || ddu.len() < n_dv {
        return Err(ReedError::QFunction(format!(
            "poisson3d_transpose: need ddv>={n_dv}, qdata>={n_qd}, ddu>={n_dv}; got ddv={} qdata={} ddu={}",
            ddv.len(),
            qdata.len(),
            ddu.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let out_bytes = (n_dv * std::mem::size_of::<f32>()) as u64;

    let ddv_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p3d-tr-ddv"),
        contents: bytemuck::cast_slice(&ddv[..n_dv]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p3d-tr-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n_qd]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let ddu_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p3d-tr-ddu"),
        contents: bytemuck::cast_slice(&ddu[..n_dv]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-p3d-tr-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-p3d-tr-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ddv_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ddu_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-p3d-tr-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-p3d-tr-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_poisson3d_transpose_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-p3d-tr-rb"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&ddu_buf, 0, &readback, 0, out_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        ddu[..n_dv].copy_from_slice(&mapped[..n_dv]);
    }
    readback.unmap();
    Ok(())
}

/// [`Vector3Poisson3DApply`](reed_cpu::Vector3Poisson3DApply): `du`/`dv`/`qdata` each `9 * num_q` (`q` = number of quadrature points).
pub(crate) fn dispatch_qf_vector3_poisson3d_apply_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    du: &[f32],
    qdata: &[f32],
    dv: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n = num_q.saturating_mul(9);
    if du.len() < n || qdata.len() < n || dv.len() < n {
        return Err(ReedError::QFunction(format!(
            "vector3_poisson3d_apply: need du>={n}, qdata>={n}, dv>={n}; got du={} qdata={} dv={}",
            du.len(),
            qdata.len(),
            dv.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n * std::mem::size_of::<f32>()) as u64;

    let du_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p3-du"),
        contents: bytemuck::cast_slice(&du[..n]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p3-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let dv_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3p3-dv"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p3-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v3p3-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: du_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dv_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v3p3-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v3p3-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vector3_poisson3d_apply_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3p3-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&dv_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        dv[..n].copy_from_slice(&mapped[..n]);
    }
    readback.unmap();
    Ok(())
}

pub(crate) fn dispatch_qf_vector3_poisson3d_transpose_accumulate_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    ddv: &[f32],
    qdata: &[f32],
    ddu: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n = num_q.saturating_mul(9);
    if ddv.len() < n || qdata.len() < n || ddu.len() < n {
        return Err(ReedError::QFunction(format!(
            "vector3_poisson3d_transpose: need ddv>={n}, qdata>={n}, ddu>={n}; got ddv={} qdata={} ddu={}",
            ddv.len(),
            qdata.len(),
            ddu.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n * std::mem::size_of::<f32>()) as u64;

    let ddv_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p3-tr-ddv"),
        contents: bytemuck::cast_slice(&ddv[..n]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p3-tr-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let ddu_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p3-tr-ddu"),
        contents: bytemuck::cast_slice(&ddu[..n]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p3-tr-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v3p3-tr-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ddv_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ddu_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v3p3-tr-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v3p3-tr-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vector3_poisson3d_transpose_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3p3-tr-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&ddu_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        ddu[..n].copy_from_slice(&mapped[..n]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` gallery [`Poisson3DApply`](reed_cpu::Poisson3DApply).
pub struct Poisson3DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Poisson3DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Poisson3DApply::default();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Poisson3DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Poisson3DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_poisson3d_apply_f32(&self.runtime, q, du, qdata, dv)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        _ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if output_cotangents.len() != 1 || input_cotangents.len() != 2 {
            return Err(ReedError::QFunction(
                "Poisson3DApplyF32Wgpu transpose expects 1 output cotangent and 2 input buffers"
                    .into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q * 3 {
            return Err(ReedError::QFunction(
                "Poisson3DApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[f32] = &qdata_fwd[0];
        if ddu.len() != q * 3 || qdata.len() != q * 9 {
            return Err(ReedError::QFunction(
                "Poisson3DApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_poisson3d_transpose_accumulate_f32(&self.runtime, q, ddv, qdata, ddu)
    }
}

/// GPU `f32` gallery [`Vector3Poisson3DApply`](reed_cpu::Vector3Poisson3DApply).
pub struct Vector3Poisson3DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vector3Poisson3DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vector3Poisson3DApply::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vector3Poisson3DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vector3Poisson3DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_vector3_poisson3d_apply_f32(&self.runtime, q, du, qdata, dv)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        _ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if output_cotangents.len() != 1 || input_cotangents.len() != 2 {
            return Err(ReedError::QFunction(
                "Vector3Poisson3DApplyF32Wgpu transpose expects 1 output cotangent and 2 input buffers"
                    .into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q * 9 {
            return Err(ReedError::QFunction(
                "Vector3Poisson3DApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[f32] = &qdata_fwd[0];
        if ddu.len() != q * 9 || qdata.len() != q * 9 {
            return Err(ReedError::QFunction(
                "Vector3Poisson3DApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_vector3_poisson3d_transpose_accumulate_f32(&self.runtime, q, ddv, qdata, ddu)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QfUnaryWordCountHost {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QfIdentityScalarUniformHost {
    num_q: u32,
    ncomp: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QfScaleF32UniformHost {
    n: u32,
    _pad0: u32,
    alpha: f32,
    _pad1: f32,
}

/// Flat `out[..n] = in[..n]` using [`GpuRuntime::qfunction_identity_copy_pipeline`].
pub(crate) fn dispatch_qf_identity_copy_f32(
    runtime: &GpuRuntime,
    n_words: usize,
    input: &[f32],
    output: &mut [f32],
) -> ReedResult<()> {
    if n_words == 0 {
        return Ok(());
    }
    if input.len() < n_words || output.len() < n_words {
        return Err(ReedError::QFunction(format!(
            "identity_copy: need len >= n ({n_words}), got in={} out={}",
            input.len(),
            output.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n_words * std::mem::size_of::<f32>()) as u64;

    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-id-in"),
        contents: bytemuck::cast_slice(&input[..n_words]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-id-out"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfUnaryWordCountHost {
        n: n_words as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-id-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_unary_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-id-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-id-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-id-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_identity_copy_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((n_words as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-id-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        output[..n_words].copy_from_slice(&mapped[..n_words]);
    }
    readback.unmap();
    Ok(())
}

/// Cotangent `du += dv` for [`reed_cpu::Identity::apply_operator_transpose`] (flat `n = q * num_comp` words).
pub(crate) fn dispatch_qf_identity_transpose_accumulate_f32(
    runtime: &GpuRuntime,
    n_words: usize,
    dv: &[f32],
    du: &mut [f32],
) -> ReedResult<()> {
    if n_words == 0 {
        return Ok(());
    }
    if dv.len() < n_words || du.len() < n_words {
        return Err(ReedError::QFunction(format!(
            "identity_transpose_accumulate: need len >= n ({n_words}), got dv={} du={}",
            dv.len(),
            du.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n_words * std::mem::size_of::<f32>()) as u64;

    let dv_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-id-tr-dv"),
        contents: bytemuck::cast_slice(&dv[..n_words]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let du_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-id-tr-du"),
        contents: bytemuck::cast_slice(&du[..n_words]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let params = QfUnaryWordCountHost {
        n: n_words as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-id-tr-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_unary_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-id-tr-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dv_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: du_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-id-tr-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-id-tr-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_identity_transpose_accumulate_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((n_words as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-id-tr-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&du_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        du[..n_words].copy_from_slice(&mapped[..n_words]);
    }
    readback.unmap();
    Ok(())
}

/// Forward [`reed_cpu::IdentityScalar`]: `out[i] = in[i * ncomp]` for `i in 0..num_q`.
pub(crate) fn dispatch_qf_identity_scalar_gather_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    ncomp: usize,
    input: &[f32],
    output: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    if ncomp == 0 {
        return Err(ReedError::QFunction(
            "identity_scalar_gather: ncomp must be > 0".into(),
        ));
    }
    let in_words = num_q.saturating_mul(ncomp);
    if input.len() < in_words || output.len() < num_q {
        return Err(ReedError::QFunction(format!(
            "identity_scalar_gather: need in>={in_words} out>={num_q}, got in={} out={}",
            input.len(),
            output.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let out_bytes = (num_q * std::mem::size_of::<f32>()) as u64;

    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-ids-in"),
        contents: bytemuck::cast_slice(&input[..in_words]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-ids-out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfIdentityScalarUniformHost {
        num_q: num_q as u32,
        ncomp: ncomp as u32,
        _p0: 0,
        _p1: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-ids-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_unary_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-ids-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-ids-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-ids-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_identity_scalar_gather_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((num_q as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-ids-rb"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        output[..num_q].copy_from_slice(&mapped[..num_q]);
    }
    readback.unmap();
    Ok(())
}

/// Cotangent `du[i * ncomp] += dv[i]` for [`reed_cpu::IdentityScalar::apply_operator_transpose`].
pub(crate) fn dispatch_qf_identity_scalar_transpose_accumulate_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    ncomp: usize,
    dv: &[f32],
    du: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    if ncomp == 0 {
        return Err(ReedError::QFunction(
            "identity_scalar_transpose_accumulate: ncomp must be > 0".into(),
        ));
    }
    let du_words = num_q.saturating_mul(ncomp);
    if dv.len() < num_q || du.len() < du_words {
        return Err(ReedError::QFunction(format!(
            "identity_scalar_transpose_accumulate: need dv>={num_q} du>={du_words}, got dv={} du={}",
            dv.len(),
            du.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let du_bytes = (du_words * std::mem::size_of::<f32>()) as u64;

    let dv_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-ids-tr-dv"),
        contents: bytemuck::cast_slice(&dv[..num_q]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let du_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-ids-tr-du"),
        contents: bytemuck::cast_slice(&du[..du_words]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let params = QfIdentityScalarUniformHost {
        num_q: num_q as u32,
        ncomp: ncomp as u32,
        _p0: 0,
        _p1: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-ids-tr-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_unary_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-ids-tr-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dv_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: du_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-ids-tr-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-ids-tr-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_identity_scalar_transpose_accumulate_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((num_q as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-ids-tr-rb"),
        size: du_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&du_buf, 0, &readback, 0, du_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        du[..du_words].copy_from_slice(&mapped[..du_words]);
    }
    readback.unmap();
    Ok(())
}

/// Flat `out[i] = alpha * in[i]` using [`GpuRuntime::qfunction_scale_f32_pipeline`].
pub(crate) fn dispatch_qf_scale_f32(
    runtime: &GpuRuntime,
    n_words: usize,
    alpha: f32,
    input: &[f32],
    output: &mut [f32],
) -> ReedResult<()> {
    if n_words == 0 {
        return Ok(());
    }
    if input.len() < n_words || output.len() < n_words {
        return Err(ReedError::QFunction(format!(
            "scale_f32: need len >= n ({n_words}), got in={} out={}",
            input.len(),
            output.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n_words * std::mem::size_of::<f32>()) as u64;

    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-sc-in"),
        contents: bytemuck::cast_slice(&input[..n_words]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-sc-out"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfScaleF32UniformHost {
        n: n_words as u32,
        _pad0: 0,
        alpha,
        _pad1: 0.0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-sc-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_unary_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-sc-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-sc-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-sc-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_scale_f32_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((n_words as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-sc-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        output[..n_words].copy_from_slice(&mapped[..n_words]);
    }
    readback.unmap();
    Ok(())
}

/// Cotangent `du += alpha * dv` for [`reed_cpu::Scale::apply_operator_transpose`].
pub(crate) fn dispatch_qf_scale_transpose_accumulate_f32(
    runtime: &GpuRuntime,
    n_words: usize,
    alpha: f32,
    dv: &[f32],
    du: &mut [f32],
) -> ReedResult<()> {
    if n_words == 0 {
        return Ok(());
    }
    if dv.len() < n_words || du.len() < n_words {
        return Err(ReedError::QFunction(format!(
            "scale_transpose_accumulate: need len >= n ({n_words}), got dv={} du={}",
            dv.len(),
            du.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n_words * std::mem::size_of::<f32>()) as u64;

    let dv_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-sc-tr-dv"),
        contents: bytemuck::cast_slice(&dv[..n_words]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let du_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-sc-tr-du"),
        contents: bytemuck::cast_slice(&du[..n_words]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let params = QfScaleF32UniformHost {
        n: n_words as u32,
        _pad0: 0,
        alpha,
        _pad1: 0.0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-sc-tr-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_unary_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-sc-tr-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dv_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: du_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-sc-tr-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-sc-tr-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_scale_transpose_accumulate_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let groups = ((n_words as u32) + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-sc-tr-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&du_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        du[..n_words].copy_from_slice(&mapped[..n_words]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` gallery [`Identity`](reed_cpu::Identity): copy packed quadrature values.
pub struct IdentityF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl IdentityF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        Self::with_components(runtime, 1)
    }

    pub fn with_components(runtime: Arc<GpuRuntime>, ncomp: usize) -> ReedResult<Self> {
        let template = reed_cpu::Identity::with_components(ncomp);
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for IdentityF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "IdentityF32Wgpu expects 1 input and 1 output".into(),
            ));
        }
        let ncomp = self.inputs[0].num_comp;
        let n_words = q.saturating_mul(ncomp);
        let u = inputs[0];
        let v = &mut outputs[0];
        dispatch_qf_identity_copy_f32(&self.runtime, n_words, u, v)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        _ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if output_cotangents.len() != 1 || input_cotangents.len() != 1 {
            return Err(ReedError::QFunction(
                "IdentityF32Wgpu transpose expects 1 output cotangent and 1 input cotangent buffer"
                    .into(),
            ));
        }
        let ncomp = self.inputs[0].num_comp;
        let n = q.saturating_mul(ncomp);
        let dv = output_cotangents[0];
        let du = &mut input_cotangents[0];
        if dv.len() != n || du.len() != n {
            return Err(ReedError::QFunction(
                "IdentityF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_identity_transpose_accumulate_f32(&self.runtime, n, dv, du)
    }
}

/// GPU `f32` gallery [`IdentityScalar`](reed_cpu::IdentityScalar): first input component per quadrature point.
pub struct IdentityScalarF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl IdentityScalarF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        Self::with_input_components(runtime, 3)
    }

    pub fn with_input_components(runtime: Arc<GpuRuntime>, ncomp: usize) -> ReedResult<Self> {
        if ncomp == 0 {
            return Err(ReedError::QFunction(
                "IdentityScalarF32Wgpu: ncomp must be > 0".into(),
            ));
        }
        let template = reed_cpu::IdentityScalar::with_input_components(ncomp);
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for IdentityScalarF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "IdentityScalarF32Wgpu expects 1 input and 1 output".into(),
            ));
        }
        let ncomp = self.inputs[0].num_comp;
        let u = inputs[0];
        let v = &mut outputs[0];
        if u.len() != q * ncomp || v.len() != q {
            return Err(ReedError::QFunction(
                "IdentityScalarF32Wgpu: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_identity_scalar_gather_f32(&self.runtime, q, ncomp, u, v)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        _ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if output_cotangents.len() != 1 || input_cotangents.len() != 1 {
            return Err(ReedError::QFunction(
                "IdentityScalarF32Wgpu transpose expects 1 output cotangent and 1 input buffer"
                    .into(),
            ));
        }
        let ncomp = self.inputs[0].num_comp;
        let dv = output_cotangents[0];
        let du = &mut input_cotangents[0];
        if dv.len() != q || du.len() != q * ncomp {
            return Err(ReedError::QFunction(
                "IdentityScalarF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_identity_scalar_transpose_accumulate_f32(&self.runtime, q, ncomp, dv, du)
    }
}

/// GPU `f32` gallery [`Scale`](reed_cpu::Scale): multiply by `alpha` from 8-byte `f64` LE context.
pub struct ScaleF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl ScaleF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        Self::with_components(runtime, 1)
    }

    pub fn with_components(runtime: Arc<GpuRuntime>, ncomp: usize) -> ReedResult<Self> {
        let template = reed_cpu::Scale::with_components(ncomp);
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for ScaleF32Wgpu {
    fn context_byte_len(&self) -> usize {
        8
    }

    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "ScaleF32Wgpu expects 1 input and 1 output".into(),
            ));
        }
        let alpha64 = QFunctionContext::read_f64_le_bytes(ctx, 0)?;
        let alpha = alpha64 as f32;
        let ncomp = self.inputs[0].num_comp;
        let n_words = q.saturating_mul(ncomp);
        let u = inputs[0];
        let v = &mut outputs[0];
        dispatch_qf_scale_f32(&self.runtime, n_words, alpha, u, v)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if output_cotangents.len() != 1 || input_cotangents.len() != 1 {
            return Err(ReedError::QFunction(
                "ScaleF32Wgpu transpose expects 1 output cotangent and 1 input cotangent buffer"
                    .into(),
            ));
        }
        let alpha64 = QFunctionContext::read_f64_le_bytes(ctx, 0)?;
        let alpha = alpha64 as f32;
        let ncomp = self.inputs[0].num_comp;
        let n = q.saturating_mul(ncomp);
        let dv = output_cotangents[0];
        let du = &mut input_cotangents[0];
        if dv.len() != n || du.len() != n {
            return Err(ReedError::QFunction(
                "ScaleF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_scale_transpose_accumulate_f32(&self.runtime, n, alpha, dv, du)
    }
}

/// `v[flat] = qdata[flat / 2] * u[flat]` for `flat in 0..2*num_qp` (gallery [`Vector2MassApply`](reed_cpu::Vector2MassApply)).
pub(crate) fn dispatch_qf_vector2_mass_apply_f32(
    runtime: &GpuRuntime,
    num_qp: usize,
    u: &[f32],
    qdata: &[f32],
    v: &mut [f32],
) -> ReedResult<()> {
    if num_qp == 0 {
        return Ok(());
    }
    let n_uv = num_qp.saturating_mul(2);
    if u.len() < n_uv || qdata.len() < num_qp || v.len() < n_uv {
        return Err(ReedError::QFunction(format!(
            "vector2_mass_apply: need u>={n_uv}, qdata>={num_qp}, v>={n_uv}; got u={} qdata={} v={}",
            u.len(),
            qdata.len(),
            v.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let u_bytes = (n_uv * std::mem::size_of::<f32>()) as u64;

    let u_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2m-u"),
        contents: bytemuck::cast_slice(&u[..n_uv]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2m-qdata"),
        contents: bytemuck::cast_slice(&qdata[..num_qp]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let v_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v2m-v"),
        size: u_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_qp as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2m-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v2m-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v2m-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v2m-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vector2_mass_apply_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = n_uv as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v2m-rb"),
        size: u_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&v_buf, 0, &readback, 0, u_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        v[..n_uv].copy_from_slice(&mapped[..n_uv]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` gallery [`Vector2MassApply`](reed_cpu::Vector2MassApply).
pub struct Vector2MassApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vector2MassApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vector2MassApply::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vector2MassApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vector2MassApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let qdata = inputs[1];
        let v = &mut outputs[0];
        dispatch_qf_vector2_mass_apply_f32(&self.runtime, q, u, qdata, v)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        transpose_qp_broadcast_components_f32(
            &self.runtime,
            q,
            2,
            ctx,
            output_cotangents,
            input_cotangents,
        )
    }
}

/// `v[flat] = qdata[flat / 3] * u[flat]` for `flat in 0..3*num_qp` (gallery [`Vector3MassApply`](reed_cpu::Vector3MassApply)).
pub(crate) fn dispatch_qf_vector3_mass_apply_f32(
    runtime: &GpuRuntime,
    num_qp: usize,
    u: &[f32],
    qdata: &[f32],
    v: &mut [f32],
) -> ReedResult<()> {
    if num_qp == 0 {
        return Ok(());
    }
    let n_uv = num_qp.saturating_mul(3);
    if u.len() < n_uv || qdata.len() < num_qp || v.len() < n_uv {
        return Err(ReedError::QFunction(format!(
            "vector3_mass_apply: need u>={n_uv}, qdata>={num_qp}, v>={n_uv}; got u={} qdata={} v={}",
            u.len(),
            qdata.len(),
            v.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let u_bytes = (n_uv * std::mem::size_of::<f32>()) as u64;

    let u_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3m-u"),
        contents: bytemuck::cast_slice(&u[..n_uv]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3m-qdata"),
        contents: bytemuck::cast_slice(&qdata[..num_qp]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let v_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3m-v"),
        size: u_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_qp as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3m-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v3m-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v3m-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v3m-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vector3_mass_apply_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = n_uv as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3m-rb"),
        size: u_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&v_buf, 0, &readback, 0, u_bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        v[..n_uv].copy_from_slice(&mapped[..n_uv]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` gallery [`Vector3MassApply`](reed_cpu::Vector3MassApply).
pub struct Vector3MassApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vector3MassApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vector3MassApply::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vector3MassApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vector3MassApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let u = inputs[0];
        let qdata = inputs[1];
        let v = &mut outputs[0];
        dispatch_qf_vector3_mass_apply_f32(&self.runtime, q, u, qdata, v)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        transpose_qp_broadcast_components_f32(
            &self.runtime,
            q,
            3,
            ctx,
            output_cotangents,
            input_cotangents,
        )
    }
}

/// GPU `f32` gallery [`Vector2Poisson1DApply`](reed_cpu::Vector2Poisson1DApply) — same kernel as [`Vector2MassApplyF32Wgpu`].
pub struct Vector2Poisson1DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vector2Poisson1DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vector2Poisson1DApply::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vector2Poisson1DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vector2Poisson1DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_vector2_mass_apply_f32(&self.runtime, q, du, qdata, dv)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        transpose_qp_broadcast_components_f32(
            &self.runtime,
            q,
            2,
            ctx,
            output_cotangents,
            input_cotangents,
        )
    }
}

/// GPU `f32` gallery [`Vector3Poisson1DApply`](reed_cpu::Vector3Poisson1DApply) — same kernel as [`Vector3MassApplyF32Wgpu`].
pub struct Vector3Poisson1DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vector3Poisson1DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vector3Poisson1DApply::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vector3Poisson1DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vector3Poisson1DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_vector3_mass_apply_f32(&self.runtime, q, du, qdata, dv)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        transpose_qp_broadcast_components_f32(
            &self.runtime,
            q,
            3,
            ctx,
            output_cotangents,
            input_cotangents,
        )
    }
}

/// [`Vector2Poisson2DApply`](reed_cpu::Vector2Poisson2DApply): `du`/`dv` length `4 * num_q`, `qdata` length `4 * num_q`.
pub(crate) fn dispatch_qf_vector2_poisson2d_apply_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    du: &[f32],
    qdata: &[f32],
    dv: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n = num_q.saturating_mul(4);
    if du.len() < n || qdata.len() < n || dv.len() < n {
        return Err(ReedError::QFunction(format!(
            "vector2_poisson2d_apply: need du>={n}, qdata>={n}, dv>={n}; got du={} qdata={} dv={}",
            du.len(),
            qdata.len(),
            dv.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n * std::mem::size_of::<f32>()) as u64;

    let du_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2p2-du"),
        contents: bytemuck::cast_slice(&du[..n]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2p2-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let dv_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v2p2-dv"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2p2-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v2p2-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: du_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dv_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v2p2-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v2p2-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vector2_poisson2d_apply_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v2p2-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&dv_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        dv[..n].copy_from_slice(&mapped[..n]);
    }
    readback.unmap();
    Ok(())
}

/// [`Vector3Poisson2DApply`](reed_cpu::Vector3Poisson2DApply): `du`/`dv` length `6 * num_q`, `qdata` length `4 * num_q`.
pub(crate) fn dispatch_qf_vector3_poisson2d_apply_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    du: &[f32],
    qdata: &[f32],
    dv: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n_du = num_q.saturating_mul(6);
    let n_qd = num_q.saturating_mul(4);
    if du.len() < n_du || qdata.len() < n_qd || dv.len() < n_du {
        return Err(ReedError::QFunction(format!(
            "vector3_poisson2d_apply: need du>={n_du}, qdata>={n_qd}, dv>={n_du}; got du={} qdata={} dv={}",
            du.len(),
            qdata.len(),
            dv.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n_du * std::mem::size_of::<f32>()) as u64;

    let du_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p2-du"),
        contents: bytemuck::cast_slice(&du[..n_du]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p2-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n_qd]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let dv_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3p2-dv"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p2-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v3p2-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: du_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dv_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v3p2-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v3p2-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vector3_poisson2d_apply_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3p2-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&dv_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        dv[..n_du].copy_from_slice(&mapped[..n_du]);
    }
    readback.unmap();
    Ok(())
}

pub(crate) fn dispatch_qf_vector2_poisson2d_transpose_accumulate_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    ddv: &[f32],
    qdata: &[f32],
    ddu: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n = num_q.saturating_mul(4);
    if ddv.len() < n || qdata.len() < n || ddu.len() < n {
        return Err(ReedError::QFunction(format!(
            "vector2_poisson2d_transpose: need ddv>={n}, qdata>={n}, ddu>={n}; got ddv={} qdata={} ddu={}",
            ddv.len(),
            qdata.len(),
            ddu.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n * std::mem::size_of::<f32>()) as u64;

    let ddv_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2p2-tr-ddv"),
        contents: bytemuck::cast_slice(&ddv[..n]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2p2-tr-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let ddu_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2p2-tr-ddu"),
        contents: bytemuck::cast_slice(&ddu[..n]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v2p2-tr-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v2p2-tr-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ddv_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ddu_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v2p2-tr-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v2p2-tr-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vector2_poisson2d_transpose_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v2p2-tr-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&ddu_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        ddu[..n].copy_from_slice(&mapped[..n]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` gallery [`Vector2Poisson2DApply`](reed_cpu::Vector2Poisson2DApply).
pub struct Vector2Poisson2DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vector2Poisson2DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vector2Poisson2DApply::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vector2Poisson2DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vector2Poisson2DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_vector2_poisson2d_apply_f32(&self.runtime, q, du, qdata, dv)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        _ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if output_cotangents.len() != 1 || input_cotangents.len() != 2 {
            return Err(ReedError::QFunction(
                "Vector2Poisson2DApplyF32Wgpu transpose expects 1 output cotangent and 2 input buffers"
                    .into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q * 4 {
            return Err(ReedError::QFunction(
                "Vector2Poisson2DApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[f32] = &qdata_fwd[0];
        if ddu.len() != q * 4 || qdata.len() != q * 4 {
            return Err(ReedError::QFunction(
                "Vector2Poisson2DApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_vector2_poisson2d_transpose_accumulate_f32(&self.runtime, q, ddv, qdata, ddu)
    }
}

pub(crate) fn dispatch_qf_vector3_poisson2d_transpose_accumulate_f32(
    runtime: &GpuRuntime,
    num_q: usize,
    ddv: &[f32],
    qdata: &[f32],
    ddu: &mut [f32],
) -> ReedResult<()> {
    if num_q == 0 {
        return Ok(());
    }
    let n_dv = num_q.saturating_mul(6);
    let n_qd = num_q.saturating_mul(4);
    if ddv.len() < n_dv || qdata.len() < n_qd || ddu.len() < n_dv {
        return Err(ReedError::QFunction(format!(
            "vector3_poisson2d_transpose: need ddv>={n_dv}, qdata>={n_qd}, ddu>={n_dv}; got ddv={} qdata={} ddu={}",
            ddv.len(),
            qdata.len(),
            ddu.len()
        )));
    }

    let device = &runtime.device;
    let queue = &runtime.queue;
    let bytes = (n_dv * std::mem::size_of::<f32>()) as u64;

    let ddv_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p2-tr-ddv"),
        contents: bytemuck::cast_slice(&ddv[..n_dv]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p2-tr-qdata"),
        contents: bytemuck::cast_slice(&qdata[..n_qd]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let ddu_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p2-tr-ddu"),
        contents: bytemuck::cast_slice(&ddu[..n_dv]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let params = QfPointwiseMulParamsHost {
        num_q: num_q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reed-qf-v3p2-tr-uni"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = runtime.qfunction_pointwise_mul_layout();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reed-qf-v3p2-tr-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uni.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ddv_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: q_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ddu_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reed-qf-v3p2-tr-enc"),
    });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reed-qf-v3p2-tr-pass"),
            ..Default::default()
        });
        cpass.set_pipeline(runtime.qfunction_vector3_poisson2d_transpose_pipeline());
        cpass.set_bind_group(0, &bind, &[]);
        let wg = 256u32;
        let n_dispatch = num_q as u32;
        let groups = (n_dispatch + wg - 1) / wg;
        cpass.dispatch_workgroups(groups, 1, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reed-qf-v3p2-tr-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&ddu_buf, 0, &readback, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map recv: {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map: {e:?}")))?;
    {
        let data = slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        ddu[..n_dv].copy_from_slice(&mapped[..n_dv]);
    }
    readback.unmap();
    Ok(())
}

/// GPU `f32` gallery [`Vector3Poisson2DApply`](reed_cpu::Vector3Poisson2DApply).
pub struct Vector3Poisson2DApplyF32Wgpu {
    runtime: Arc<GpuRuntime>,
    inputs: Vec<reed_core::QFunctionField>,
    outputs: Vec<reed_core::QFunctionField>,
}

impl Vector3Poisson2DApplyF32Wgpu {
    pub fn new(runtime: Arc<GpuRuntime>) -> ReedResult<Self> {
        let template = reed_cpu::Vector3Poisson2DApply::new();
        let inputs = QFunctionTrait::<f32>::inputs(&template).to_vec();
        let outputs = QFunctionTrait::<f32>::outputs(&template).to_vec();
        Ok(Self {
            runtime,
            inputs,
            outputs,
        })
    }
}

impl QFunctionTrait<f32> for Vector3Poisson2DApplyF32Wgpu {
    fn inputs(&self) -> &[reed_core::QFunctionField] {
        &self.inputs
    }

    fn outputs(&self) -> &[reed_core::QFunctionField] {
        &self.outputs
    }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "Vector3Poisson2DApplyF32Wgpu expects 2 inputs and 1 output".into(),
            ));
        }
        let du = inputs[0];
        let qdata = inputs[1];
        let dv = &mut outputs[0];
        dispatch_qf_vector3_poisson2d_apply_f32(&self.runtime, q, du, qdata, dv)
    }

    fn supports_operator_transpose(&self) -> bool {
        true
    }

    fn apply_operator_transpose(
        &self,
        _ctx: &[u8],
        q: usize,
        output_cotangents: &[&[f32]],
        input_cotangents: &mut [&mut [f32]],
    ) -> ReedResult<()> {
        if output_cotangents.len() != 1 || input_cotangents.len() != 2 {
            return Err(ReedError::QFunction(
                "Vector3Poisson2DApplyF32Wgpu transpose expects 1 output cotangent and 2 input buffers"
                    .into(),
            ));
        }
        let ddv = output_cotangents[0];
        if ddv.len() != q * 6 {
            return Err(ReedError::QFunction(
                "Vector3Poisson2DApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        let (ddu_buf, qdata_fwd) = input_cotangents.split_at_mut(1);
        let ddu = &mut ddu_buf[0];
        let qdata: &[f32] = &qdata_fwd[0];
        if ddu.len() != q * 6 || qdata.len() != q * 4 {
            return Err(ReedError::QFunction(
                "Vector3Poisson2DApplyF32Wgpu transpose: buffer length mismatch".into(),
            ));
        }
        dispatch_qf_vector3_poisson2d_transpose_accumulate_f32(&self.runtime, q, ddv, qdata, ddu)
    }
}

/// When `Some`, `name` has an `f32` WGSL gallery implementation; `None` means use the CPU gallery.
///
/// For every entry in [`reed_cpu::QFUNCTION_INTERIOR_GALLERY_NAMES`], this function should return
/// `Some` so `Reed` wgpu paths can avoid the CPU gallery for those kernels (see unit test
/// `interior_gallery_names_have_device_f32_qfunction`).
pub fn try_create_device_q_function_f32(
    name: &str,
    runtime: Arc<GpuRuntime>,
) -> Option<ReedResult<Box<dyn QFunctionTrait<f32>>>> {
    match name {
        "MassApply" | "MassApplyAtPoints" => Some(
            MassApplyF32Wgpu::new(runtime).map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "MassApplyInterpTimesWeight" | "MassApplyInterpTimesWeightAtPoints" => Some(
            MassApplyInterpTimesWeightF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Mass1DBuild" => Some(
            Mass1DBuildF32Wgpu::new(runtime).map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Mass2DBuild" => Some(
            Mass2DBuildF32Wgpu::new(runtime).map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Mass3DBuild" => Some(
            Mass3DBuildF32Wgpu::new(runtime).map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Poisson1DBuild" => Some(
            Poisson1DBuildF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Poisson2DBuild" => Some(
            Poisson2DBuildF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Poisson3DBuild" => Some(
            Poisson3DBuildF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Poisson1DApply" => Some(
            Poisson1DApplyF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Poisson2DApply" | "Poisson2DApplyAtPoints" => Some(
            Poisson2DApplyF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Poisson3DApply" => Some(
            Poisson3DApplyF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Vector2MassApply" => Some(
            Vector2MassApplyF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Vector3MassApply" => Some(
            Vector3MassApplyF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Vector2Poisson1DApply" => Some(
            Vector2Poisson1DApplyF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Vector3Poisson1DApply" => Some(
            Vector3Poisson1DApplyF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Vector2Poisson2DApply" => Some(
            Vector2Poisson2DApplyF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Vector3Poisson2DApply" => Some(
            Vector3Poisson2DApplyF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Vector3Poisson3DApply" => Some(
            Vector3Poisson3DApplyF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Vec2Dot" => {
            Some(Vec2DotF32Wgpu::new(runtime).map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>))
        }
        "Vec3Dot" => {
            Some(Vec3DotF32Wgpu::new(runtime).map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>))
        }
        "Identity" | "IdentityAtPoints" => {
            Some(IdentityF32Wgpu::new(runtime).map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>))
        }
        "Identity to scalar" | "IdentityScalar" => Some(
            IdentityScalarF32Wgpu::new(runtime)
                .map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>),
        ),
        "Scale" | "Scale (scalar)" | "ScaleScalar" | "ScaleAtPoints" => {
            Some(ScaleF32Wgpu::new(runtime).map(|q| Box::new(q) as Box<dyn QFunctionTrait<f32>>))
        }
        _ => None,
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use reed_core::QFunctionContext;

    #[test]
    fn interior_gallery_names_have_device_f32_qfunction() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        for &name in reed_cpu::QFUNCTION_INTERIOR_GALLERY_NAMES {
            match try_create_device_q_function_f32(name, rt.clone()) {
                None => panic!(
                    "interior gallery name {name:?} should have an f32 device QFunction mapping (keep in sync with reed_cpu::QFUNCTION_INTERIOR_GALLERY_NAMES)"
                ),
                Some(Err(e)) => panic!("interior gallery name {name:?} device ctor failed: {e:?}"),
                Some(Ok(_)) => {}
            }
        }
    }

    #[test]
    fn prototype_scale_matches_cpu() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = GpuRuntime::new(&adapter).expect("device");
        let proto = QFunctionPrototypeScaleF32::new(Arc::new(rt)).unwrap();
        let n = 100usize;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut out_gpu = vec![0.0_f32; n];
        proto.apply(n, 2.5, &input, &mut out_gpu).unwrap();
        for i in 0..n {
            assert!(
                (out_gpu[i] - 2.5 * input[i]).abs() < 1.0e-4,
                "i={i} got {} want {}",
                out_gpu[i],
                2.5 * input[i]
            );
        }
    }

    #[test]
    fn mass_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = MassApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::MassApply::default();
        let q = 64usize;
        let u: Vec<f32> = (0..q).map(|i| i as f32 * 0.03).collect();
        let qd: Vec<f32> = (0..q).map(|i| 1.0 + i as f32 * 0.01).collect();
        let mut out_gpu = vec![0.0_f32; q];
        let mut out_cpu = vec![0.0_f32; q];
        gpu.apply(&[], q, &[u.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[u.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn mass_apply_interp_times_weight_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = MassApplyInterpTimesWeightF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::MassApplyInterpTimesWeight::default();
        let q = 37usize;
        let u: Vec<f32> = (0..q).map(|i| (i as f32) * 0.11 - 0.5).collect();
        let w: Vec<f32> = (0..q).map(|i| 0.5 + (i as f32) * 0.02).collect();
        let mut v_gpu = vec![0.0_f32; q];
        let mut v_cpu = vec![0.0_f32; q];
        gpu.apply(&[], q, &[u.as_slice(), w.as_slice()], &mut [&mut v_gpu])
            .unwrap();
        cpu.apply(&[], q, &[u.as_slice(), w.as_slice()], &mut [&mut v_cpu])
            .unwrap();
        assert_eq!(v_gpu, v_cpu);

        let dv: Vec<f32> = (0..q).map(|i| (i as f32) * 0.07).collect();
        let mut du_gpu = vec![0.15_f32; q];
        let mut du_cpu = du_gpu.clone();
        let mut w_slot_gpu = w.clone();
        let mut w_slot_cpu = w.clone();
        gpu.apply_operator_transpose(
            &[],
            q,
            &[dv.as_slice()],
            &mut [&mut du_gpu, &mut w_slot_gpu],
        )
        .unwrap();
        cpu.apply_operator_transpose(
            &[],
            q,
            &[dv.as_slice()],
            &mut [&mut du_cpu, &mut w_slot_cpu],
        )
        .unwrap();
        for i in 0..q {
            assert!(
                (du_gpu[i] - du_cpu[i]).abs() < 1.0e-4,
                "transpose i={i} gpu={} cpu={}",
                du_gpu[i],
                du_cpu[i]
            );
        }
    }

    #[test]
    fn mass1d_build_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Mass1DBuildF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Mass1DBuild::default();
        let q = 31usize;
        let dx: Vec<f32> = (0..q).map(|i| (i as f32) * 0.04 - 0.6).collect();
        let weights: Vec<f32> = (0..q).map(|i| 0.2 + (i as f32) * 0.03).collect();
        let mut q_gpu = vec![0.0_f32; q];
        let mut q_cpu = vec![0.0_f32; q];
        gpu.apply(
            &[],
            q,
            &[dx.as_slice(), weights.as_slice()],
            &mut [&mut q_gpu],
        )
        .unwrap();
        cpu.apply(
            &[],
            q,
            &[dx.as_slice(), weights.as_slice()],
            &mut [&mut q_cpu],
        )
        .unwrap();
        for i in 0..q {
            assert!(
                (q_gpu[i] - q_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                q_gpu[i],
                q_cpu[i]
            );
        }
    }

    #[test]
    fn mass2d_build_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Mass2DBuildF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Mass2DBuild::default();
        let q = 17usize;
        let dx: Vec<f32> = (0..q * 4)
            .map(|i| 0.5 + (i as f32) * 0.02 - (i as f32 % 3.0) * 0.01)
            .collect();
        let weights: Vec<f32> = (0..q).map(|i| 0.15 + (i as f32) * 0.05).collect();
        let mut q_gpu = vec![0.0_f32; q];
        let mut q_cpu = vec![0.0_f32; q];
        gpu.apply(
            &[],
            q,
            &[dx.as_slice(), weights.as_slice()],
            &mut [&mut q_gpu],
        )
        .unwrap();
        cpu.apply(
            &[],
            q,
            &[dx.as_slice(), weights.as_slice()],
            &mut [&mut q_cpu],
        )
        .unwrap();
        for i in 0..q {
            assert!(
                (q_gpu[i] - q_cpu[i]).abs() < 2.0e-5,
                "i={i} gpu={} cpu={}",
                q_gpu[i],
                q_cpu[i]
            );
        }
    }

    #[test]
    fn mass3d_build_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Mass3DBuildF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Mass3DBuild::default();
        let q = 11usize;
        // Well-spaced rows so det is not tiny.
        let mut dx = vec![0.0_f32; q * 9];
        for i in 0..q {
            let b = i * 9;
            dx[b] = 1.0 + i as f32 * 0.01;
            dx[b + 4] = 1.1 + i as f32 * 0.012;
            dx[b + 8] = 0.9 + i as f32 * 0.008;
            dx[b + 1] = 0.05;
            dx[b + 2] = -0.03;
            dx[b + 3] = -0.04;
            dx[b + 5] = 0.06;
            dx[b + 6] = 0.02;
            dx[b + 7] = -0.05;
        }
        let weights: Vec<f32> = (0..q).map(|i| 0.25 + (i as f32) * 0.02).collect();
        let mut q_gpu = vec![0.0_f32; q];
        let mut q_cpu = vec![0.0_f32; q];
        gpu.apply(
            &[],
            q,
            &[dx.as_slice(), weights.as_slice()],
            &mut [&mut q_gpu],
        )
        .unwrap();
        cpu.apply(
            &[],
            q,
            &[dx.as_slice(), weights.as_slice()],
            &mut [&mut q_cpu],
        )
        .unwrap();
        for i in 0..q {
            assert!(
                (q_gpu[i] - q_cpu[i]).abs() < 2.0e-4,
                "i={i} gpu={} cpu={}",
                q_gpu[i],
                q_cpu[i]
            );
        }
    }

    #[test]
    fn poisson1d_build_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Poisson1DBuildF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Poisson1DBuild::default();
        let q = 29usize;
        let dx: Vec<f32> = (0..q).map(|i| 0.4 + (i as f32) * 0.03).collect();
        let weights: Vec<f32> = (0..q).map(|i| 0.7 + (i as f32) * 0.01).collect();
        let mut q_gpu = vec![0.0_f32; q];
        let mut q_cpu = vec![0.0_f32; q];
        gpu.apply(
            &[],
            q,
            &[dx.as_slice(), weights.as_slice()],
            &mut [&mut q_gpu],
        )
        .unwrap();
        cpu.apply(
            &[],
            q,
            &[dx.as_slice(), weights.as_slice()],
            &mut [&mut q_cpu],
        )
        .unwrap();
        for i in 0..q {
            assert!(
                (q_gpu[i] - q_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                q_gpu[i],
                q_cpu[i]
            );
        }
    }

    #[test]
    fn poisson1d_build_wgpu_rejects_near_singular_dx() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Poisson1DBuildF32Wgpu::new(rt).unwrap();
        let q = 3usize;
        let dx = vec![1.0_f32, 1e-30_f32, 1.0_f32];
        let weights = vec![1.0_f32; q];
        let mut qo = vec![0.0_f32; q];
        let err = gpu
            .apply(&[], q, &[dx.as_slice(), weights.as_slice()], &mut [&mut qo])
            .unwrap_err();
        assert!(
            err.to_string().contains("near-singular"),
            "unexpected err: {err:?}"
        );
    }

    #[test]
    fn poisson2d_build_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Poisson2DBuildF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Poisson2DBuild::default();
        let q = 14usize;
        let dx: Vec<f32> = (0..q * 4)
            .map(|i| 0.3 + (i as f32) * 0.017 - (i as f32 % 5.0) * 0.01)
            .collect();
        let weights: Vec<f32> = (0..q).map(|i| 0.22 + (i as f32) * 0.04).collect();
        let mut q_gpu = vec![0.0_f32; q * 4];
        let mut q_cpu = vec![0.0_f32; q * 4];
        gpu.apply(
            &[],
            q,
            &[dx.as_slice(), weights.as_slice()],
            &mut [&mut q_gpu],
        )
        .unwrap();
        cpu.apply(
            &[],
            q,
            &[dx.as_slice(), weights.as_slice()],
            &mut [&mut q_cpu],
        )
        .unwrap();
        for i in 0..q * 4 {
            let tol = 5.0e-3_f32 * q_cpu[i].abs().max(1.0);
            assert!(
                (q_gpu[i] - q_cpu[i]).abs() <= tol,
                "i={i} gpu={} cpu={} tol={tol}",
                q_gpu[i],
                q_cpu[i],
            );
        }
    }

    #[test]
    fn poisson3d_build_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Poisson3DBuildF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Poisson3DBuild::default();
        let q = 7usize;
        let mut dx = vec![0.0_f32; q * 9];
        for i in 0..q {
            let b = i * 9;
            dx[b] = 1.05 + i as f32 * 0.02;
            dx[b + 4] = 1.12 + i as f32 * 0.015;
            dx[b + 8] = 0.95 + i as f32 * 0.018;
            dx[b + 1] = 0.06;
            dx[b + 2] = -0.04;
            dx[b + 3] = -0.05;
            dx[b + 5] = 0.07;
            dx[b + 6] = 0.03;
            dx[b + 7] = -0.06;
        }
        let weights: Vec<f32> = (0..q).map(|i| 0.3 + (i as f32) * 0.025).collect();
        let mut q_gpu = vec![0.0_f32; q * 9];
        let mut q_cpu = vec![0.0_f32; q * 9];
        gpu.apply(
            &[],
            q,
            &[dx.as_slice(), weights.as_slice()],
            &mut [&mut q_gpu],
        )
        .unwrap();
        cpu.apply(
            &[],
            q,
            &[dx.as_slice(), weights.as_slice()],
            &mut [&mut q_cpu],
        )
        .unwrap();
        for i in 0..q * 9 {
            assert!(
                (q_gpu[i] - q_cpu[i]).abs() < 5.0e-4,
                "i={i} gpu={} cpu={}",
                q_gpu[i],
                q_cpu[i]
            );
        }
    }

    #[test]
    fn poisson1d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Poisson1DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Poisson1DApply::default();
        let q = 64usize;
        let du: Vec<f32> = (0..q).map(|i| i as f32 * 0.03).collect();
        let qd: Vec<f32> = (0..q).map(|i| 1.0 + i as f32 * 0.01).collect();
        let mut out_gpu = vec![0.0_f32; q];
        let mut out_cpu = vec![0.0_f32; q];
        gpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn poisson1d_apply_transpose_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Poisson1DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Poisson1DApply::default();
        let q = 41usize;
        let ddv: Vec<f32> = (0..q).map(|i| (i as f32) * 0.04 - 0.2).collect();
        let qd: Vec<f32> = (0..q).map(|i| 0.8 + (i as f32) * 0.015).collect();
        let mut ddu_gpu = vec![0.11_f32; q];
        let mut ddu_cpu = ddu_gpu.clone();
        let mut qd_gpu = qd.clone();
        let mut qd_cpu = qd.clone();
        gpu.apply_operator_transpose(&[], q, &[ddv.as_slice()], &mut [&mut ddu_gpu, &mut qd_gpu])
            .unwrap();
        cpu.apply_operator_transpose(&[], q, &[ddv.as_slice()], &mut [&mut ddu_cpu, &mut qd_cpu])
            .unwrap();
        for i in 0..q {
            assert!(
                (ddu_gpu[i] - ddu_cpu[i]).abs() < 1.0e-4,
                "i={i} gpu={} cpu={}",
                ddu_gpu[i],
                ddu_cpu[i]
            );
        }
    }

    #[test]
    fn poisson2d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Poisson2DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Poisson2DApply::default();
        let q = 40usize;
        let du: Vec<f32> = (0..2 * q).map(|i| (i as f32) * 0.01 - 0.1).collect();
        let qd: Vec<f32> = (0..4 * q).map(|i| 0.25 + (i as f32) * 0.007).collect();
        let mut out_gpu = vec![0.0_f32; 2 * q];
        let mut out_cpu = vec![0.0_f32; 2 * q];
        gpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..2 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 2.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn poisson3d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Poisson3DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Poisson3DApply::default();
        let q = 32usize;
        let du: Vec<f32> = (0..3 * q).map(|i| (i as f32) * 0.02 - 0.3).collect();
        let qd: Vec<f32> = (0..9 * q).map(|i| 0.1 + (i as f32) * 0.005).collect();
        let mut out_gpu = vec![0.0_f32; 3 * q];
        let mut out_cpu = vec![0.0_f32; 3 * q];
        gpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..3 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 3.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn poisson2d_transpose_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Poisson2DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Poisson2DApply::default();
        let q = 37usize;
        let ddv: Vec<f32> = (0..2 * q).map(|i| (i as f32) * 0.02 - 0.35).collect();
        let mut qd_gpu: Vec<f32> = (0..4 * q).map(|i| 0.2 + (i as f32) * 0.011).collect();
        let mut qd_cpu = qd_gpu.clone();
        let mut ddu_gpu: Vec<f32> = (0..2 * q).map(|i| 0.05 * (i as f32)).collect();
        let mut ddu_cpu = ddu_gpu.clone();
        gpu.apply_operator_transpose(&[], q, &[ddv.as_slice()], &mut [&mut ddu_gpu, &mut qd_gpu])
            .unwrap();
        cpu.apply_operator_transpose(&[], q, &[ddv.as_slice()], &mut [&mut ddu_cpu, &mut qd_cpu])
            .unwrap();
        for i in 0..2 * q {
            assert!(
                (ddu_gpu[i] - ddu_cpu[i]).abs() < 3.0e-5,
                "i={i} gpu={} cpu={}",
                ddu_gpu[i],
                ddu_cpu[i]
            );
        }
        assert_eq!(qd_gpu, qd_cpu);
    }

    #[test]
    fn poisson3d_transpose_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Poisson3DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Poisson3DApply::default();
        let q = 29usize;
        let ddv: Vec<f32> = (0..3 * q).map(|i| (i as f32) * 0.03 - 0.4).collect();
        let mut qd_gpu: Vec<f32> = (0..9 * q).map(|i| 0.12 + (i as f32) * 0.004).collect();
        let mut qd_cpu = qd_gpu.clone();
        let mut ddu_gpu: Vec<f32> = (0..3 * q).map(|i| -0.02 * (i as isize as f32)).collect();
        let mut ddu_cpu = ddu_gpu.clone();
        gpu.apply_operator_transpose(&[], q, &[ddv.as_slice()], &mut [&mut ddu_gpu, &mut qd_gpu])
            .unwrap();
        cpu.apply_operator_transpose(&[], q, &[ddv.as_slice()], &mut [&mut ddu_cpu, &mut qd_cpu])
            .unwrap();
        for i in 0..3 * q {
            assert!(
                (ddu_gpu[i] - ddu_cpu[i]).abs() < 4.0e-5,
                "i={i} gpu={} cpu={}",
                ddu_gpu[i],
                ddu_cpu[i]
            );
        }
        assert_eq!(qd_gpu, qd_cpu);
    }

    #[test]
    fn vector2_poisson2d_transpose_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector2Poisson2DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector2Poisson2DApply::new();
        let q = 31usize;
        let ddv: Vec<f32> = (0..4 * q).map(|i| (i as f32) * 0.017 - 0.2).collect();
        let mut qd_gpu: Vec<f32> = (0..4 * q).map(|i| 0.18 + (i as f32) * 0.009).collect();
        let mut qd_cpu = qd_gpu.clone();
        let mut ddu_gpu: Vec<f32> = (0..4 * q).map(|i| 0.03 * (i as f32)).collect();
        let mut ddu_cpu = ddu_gpu.clone();
        gpu.apply_operator_transpose(&[], q, &[ddv.as_slice()], &mut [&mut ddu_gpu, &mut qd_gpu])
            .unwrap();
        cpu.apply_operator_transpose(&[], q, &[ddv.as_slice()], &mut [&mut ddu_cpu, &mut qd_cpu])
            .unwrap();
        for i in 0..4 * q {
            assert!(
                (ddu_gpu[i] - ddu_cpu[i]).abs() < 4.0e-5,
                "i={i} gpu={} cpu={}",
                ddu_gpu[i],
                ddu_cpu[i]
            );
        }
        assert_eq!(qd_gpu, qd_cpu);
    }

    #[test]
    fn vector3_poisson2d_transpose_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector3Poisson2DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector3Poisson2DApply::new();
        let q = 28usize;
        let ddv: Vec<f32> = (0..6 * q).map(|i| (i as f32) * 0.014 - 0.25).collect();
        let mut qd_gpu: Vec<f32> = (0..4 * q).map(|i| 0.22 + (i as f32) * 0.008).collect();
        let mut qd_cpu = qd_gpu.clone();
        let mut ddu_gpu: Vec<f32> = (0..6 * q).map(|i| -0.01 * (i as f32)).collect();
        let mut ddu_cpu = ddu_gpu.clone();
        gpu.apply_operator_transpose(&[], q, &[ddv.as_slice()], &mut [&mut ddu_gpu, &mut qd_gpu])
            .unwrap();
        cpu.apply_operator_transpose(&[], q, &[ddv.as_slice()], &mut [&mut ddu_cpu, &mut qd_cpu])
            .unwrap();
        for i in 0..6 * q {
            assert!(
                (ddu_gpu[i] - ddu_cpu[i]).abs() < 4.0e-5,
                "i={i} gpu={} cpu={}",
                ddu_gpu[i],
                ddu_cpu[i]
            );
        }
        assert_eq!(qd_gpu, qd_cpu);
    }

    #[test]
    fn identity_f32_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = IdentityF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Identity::default();
        let q = 32usize;
        let ncomp = 1usize;
        let n = q * ncomp;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.07).collect();
        let mut out_gpu = vec![0.0_f32; n];
        let mut out_cpu = vec![0.0_f32; n];
        gpu.apply(&[], q, &[input.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[input.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        assert_eq!(out_gpu, out_cpu);
    }

    #[test]
    fn identity_transpose_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = IdentityF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Identity::default();
        let q = 47usize;
        let n = q;
        let dv: Vec<f32> = (0..n).map(|i| (i as f32) * 0.031 - 0.4).collect();
        let mut du_gpu: Vec<f32> = (0..n).map(|i| 0.12 + (i as f32) * 0.02).collect();
        let mut du_cpu = du_gpu.clone();
        gpu.apply_operator_transpose(&[], q, &[dv.as_slice()], &mut [&mut du_gpu])
            .unwrap();
        cpu.apply_operator_transpose(&[], q, &[dv.as_slice()], &mut [&mut du_cpu])
            .unwrap();
        for i in 0..n {
            assert!(
                (du_gpu[i] - du_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                du_gpu[i],
                du_cpu[i]
            );
        }
    }

    #[test]
    fn identity_transpose_ncomp3_wgpu_matches_cpu() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = IdentityF32Wgpu::with_components(rt, 3).unwrap();
        let cpu = reed_cpu::Identity::with_components(3);
        let q = 11usize;
        let n = q * 3;
        let dv: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 0.7).collect();
        let mut du_gpu: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.01).collect();
        let mut du_cpu = du_gpu.clone();
        gpu.apply_operator_transpose(&[], q, &[dv.as_slice()], &mut [&mut du_gpu])
            .unwrap();
        cpu.apply_operator_transpose(&[], q, &[dv.as_slice()], &mut [&mut du_cpu])
            .unwrap();
        assert_eq!(du_gpu, du_cpu);
    }

    #[test]
    fn identity_f32_wgpu_ncomp3_matches_cpu() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = IdentityF32Wgpu::with_components(rt, 3).unwrap();
        let cpu = reed_cpu::Identity::with_components(3);
        let q = 8usize;
        let n = q * 3;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.11).collect();
        let mut out_gpu = vec![0.0_f32; n];
        let mut out_cpu = vec![0.0_f32; n];
        gpu.apply(&[], q, &[input.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[input.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        assert_eq!(out_gpu, out_cpu);
    }

    #[test]
    fn identity_scalar_f32_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = IdentityScalarF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::IdentityScalar::default();
        let q = 19usize;
        let ncomp = 3usize;
        let n_in = q * ncomp;
        let input: Vec<f32> = (0..n_in).map(|i| (i as f32) * 0.13 - 0.5).collect();
        let mut out_gpu = vec![0.0_f32; q];
        let mut out_cpu = vec![0.0_f32; q];
        gpu.apply(&[], q, &[input.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[input.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        assert_eq!(out_gpu, out_cpu);
    }

    #[test]
    fn identity_scalar_transpose_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = IdentityScalarF32Wgpu::with_input_components(rt, 4).unwrap();
        let cpu = reed_cpu::IdentityScalar::with_input_components(4);
        let q = 23usize;
        let ncomp = 4usize;
        let dv: Vec<f32> = (0..q).map(|i| (i as f32) * 0.07 - 0.2).collect();
        let mut du_gpu: Vec<f32> = (0..q * ncomp).map(|i| 0.05 + (i as f32) * 0.03).collect();
        let mut du_cpu = du_gpu.clone();
        gpu.apply_operator_transpose(&[], q, &[dv.as_slice()], &mut [&mut du_gpu])
            .unwrap();
        cpu.apply_operator_transpose(&[], q, &[dv.as_slice()], &mut [&mut du_cpu])
            .unwrap();
        assert_eq!(du_gpu, du_cpu);
    }

    #[test]
    fn try_create_identity_to_scalar_returns_scalar_type() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let qf = try_create_device_q_function_f32("Identity to scalar", rt).unwrap();
        let qf = qf.unwrap();
        assert_eq!(qf.inputs()[0].num_comp, 3);
        assert_eq!(qf.outputs()[0].num_comp, 1);
    }

    #[test]
    fn vec2_dot_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vec2DotF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vec2Dot::new();
        let q = 41usize;
        let u: Vec<f32> = (0..q * 2).map(|i| (i as f32) * 0.09 - 1.1).collect();
        let v: Vec<f32> = (0..q * 2).map(|i| (i as f32) * -0.04 + 0.3).collect();
        let mut w_gpu = vec![0.0_f32; q];
        let mut w_cpu = vec![0.0_f32; q];
        gpu.apply(&[], q, &[u.as_slice(), v.as_slice()], &mut [&mut w_gpu])
            .unwrap();
        cpu.apply(&[], q, &[u.as_slice(), v.as_slice()], &mut [&mut w_cpu])
            .unwrap();
        for i in 0..q {
            assert!(
                (w_gpu[i] - w_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                w_gpu[i],
                w_cpu[i]
            );
        }
    }

    #[test]
    fn vec3_dot_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vec3DotF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vec3Dot::new();
        let q = 29usize;
        let u: Vec<f32> = (0..q * 3).map(|i| (i as f32) * 0.06).collect();
        let v: Vec<f32> = (0..q * 3).map(|i| (i as f32 + 1.0).recip()).collect();
        let mut w_gpu = vec![0.0_f32; q];
        let mut w_cpu = vec![0.0_f32; q];
        gpu.apply(&[], q, &[u.as_slice(), v.as_slice()], &mut [&mut w_gpu])
            .unwrap();
        cpu.apply(&[], q, &[u.as_slice(), v.as_slice()], &mut [&mut w_cpu])
            .unwrap();
        for i in 0..q {
            assert!(
                (w_gpu[i] - w_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                w_gpu[i],
                w_cpu[i]
            );
        }
    }

    #[test]
    fn scale_f32_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = ScaleF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Scale::default();
        let mut ctx = [0u8; 8];
        QFunctionContext::write_f64_le_bytes(&mut ctx, 0, -2.25).unwrap();
        let q = 40usize;
        let input: Vec<f32> = (0..q).map(|i| i as f32 * 0.05).collect();
        let mut out_gpu = vec![0.0_f32; q];
        let mut out_cpu = vec![0.0_f32; q];
        gpu.apply(&ctx, q, &[input.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&ctx, q, &[input.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 2.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn scale_transpose_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = ScaleF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Scale::default();
        let mut ctx = [0u8; 8];
        QFunctionContext::write_f64_le_bytes(&mut ctx, 0, 1.375).unwrap();
        let q = 43usize;
        let dv: Vec<f32> = (0..q).map(|i| (i as f32) * 0.04 - 0.5).collect();
        let mut du_gpu: Vec<f32> = (0..q).map(|i| 0.33 + (i as f32) * 0.015).collect();
        let mut du_cpu = du_gpu.clone();
        gpu.apply_operator_transpose(&ctx, q, &[dv.as_slice()], &mut [&mut du_gpu])
            .unwrap();
        cpu.apply_operator_transpose(&ctx, q, &[dv.as_slice()], &mut [&mut du_cpu])
            .unwrap();
        for i in 0..q {
            assert!(
                (du_gpu[i] - du_cpu[i]).abs() < 2.5e-5,
                "i={i} gpu={} cpu={}",
                du_gpu[i],
                du_cpu[i]
            );
        }
    }

    #[test]
    fn vector2_mass_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector2MassApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector2MassApply::new();
        let q = 48usize;
        let u: Vec<f32> = (0..2 * q).map(|i| i as f32 * 0.02).collect();
        let qd: Vec<f32> = (0..q).map(|i| 0.5 + i as f32 * 0.03).collect();
        let mut out_gpu = vec![0.0_f32; 2 * q];
        let mut out_cpu = vec![0.0_f32; 2 * q];
        gpu.apply(&[], q, &[u.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[u.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..2 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn vector3_mass_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector3MassApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector3MassApply::new();
        let q = 48usize;
        let u: Vec<f32> = (0..3 * q).map(|i| i as f32 * 0.02).collect();
        let qd: Vec<f32> = (0..q).map(|i| 0.5 + i as f32 * 0.03).collect();
        let mut out_gpu = vec![0.0_f32; 3 * q];
        let mut out_cpu = vec![0.0_f32; 3 * q];
        gpu.apply(&[], q, &[u.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[u.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..3 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn vector2_poisson1d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector2Poisson1DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector2Poisson1DApply::new();
        let q = 48usize;
        let du: Vec<f32> = (0..2 * q).map(|i| i as f32 * 0.02).collect();
        let qd: Vec<f32> = (0..q).map(|i| 0.5 + i as f32 * 0.03).collect();
        let mut out_gpu = vec![0.0_f32; 2 * q];
        let mut out_cpu = vec![0.0_f32; 2 * q];
        gpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..2 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn vector3_poisson1d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector3Poisson1DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector3Poisson1DApply::new();
        let q = 48usize;
        let du: Vec<f32> = (0..3 * q).map(|i| i as f32 * 0.02).collect();
        let qd: Vec<f32> = (0..q).map(|i| 0.5 + i as f32 * 0.03).collect();
        let mut out_gpu = vec![0.0_f32; 3 * q];
        let mut out_cpu = vec![0.0_f32; 3 * q];
        gpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..3 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 1.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn vector2_poisson2d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector2Poisson2DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector2Poisson2DApply::new();
        let q = 36usize;
        let du: Vec<f32> = (0..4 * q).map(|i| (i as f32) * 0.015 - 0.2).collect();
        let qd: Vec<f32> = (0..4 * q).map(|i| 0.2 + (i as f32) * 0.006).collect();
        let mut out_gpu = vec![0.0_f32; 4 * q];
        let mut out_cpu = vec![0.0_f32; 4 * q];
        gpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..4 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 2.5e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn vector3_poisson2d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector3Poisson2DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector3Poisson2DApply::new();
        let q = 36usize;
        let du: Vec<f32> = (0..6 * q).map(|i| (i as f32) * 0.012 - 0.15).collect();
        let qd: Vec<f32> = (0..4 * q).map(|i| 0.18 + (i as f32) * 0.005).collect();
        let mut out_gpu = vec![0.0_f32; 6 * q];
        let mut out_cpu = vec![0.0_f32; 6 * q];
        gpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..6 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 2.5e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn vector3_poisson3d_apply_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector3Poisson3DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector3Poisson3DApply::new();
        let q = 27usize;
        let du: Vec<f32> = (0..9 * q).map(|i| (i as f32) * 0.011 - 0.2).collect();
        let qd: Vec<f32> = (0..9 * q).map(|i| 0.14 + (i as f32) * 0.004).collect();
        let mut out_gpu = vec![0.0_f32; 9 * q];
        let mut out_cpu = vec![0.0_f32; 9 * q];
        gpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_gpu])
            .unwrap();
        cpu.apply(&[], q, &[du.as_slice(), qd.as_slice()], &mut [&mut out_cpu])
            .unwrap();
        for i in 0..9 * q {
            assert!(
                (out_gpu[i] - out_cpu[i]).abs() < 4.0e-5,
                "i={i} gpu={} cpu={}",
                out_gpu[i],
                out_cpu[i]
            );
        }
    }

    #[test]
    fn vector3_poisson3d_transpose_wgpu_matches_cpu_gallery() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("adapter");
        let rt = Arc::new(GpuRuntime::new(&adapter).expect("device"));
        let gpu = Vector3Poisson3DApplyF32Wgpu::new(rt).unwrap();
        let cpu = reed_cpu::Vector3Poisson3DApply::new();
        let q = 26usize;
        let ddv: Vec<f32> = (0..9 * q).map(|i| (i as f32) * 0.013 - 0.22).collect();
        let mut qd_gpu: Vec<f32> = (0..9 * q).map(|i| 0.16 + (i as f32) * 0.0035).collect();
        let mut qd_cpu = qd_gpu.clone();
        let mut ddu_gpu: Vec<f32> = (0..9 * q).map(|i| 0.02 * (i as isize as f32)).collect();
        let mut ddu_cpu = ddu_gpu.clone();
        gpu.apply_operator_transpose(&[], q, &[ddv.as_slice()], &mut [&mut ddu_gpu, &mut qd_gpu])
            .unwrap();
        cpu.apply_operator_transpose(&[], q, &[ddv.as_slice()], &mut [&mut ddu_cpu, &mut qd_cpu])
            .unwrap();
        for i in 0..9 * q {
            assert!(
                (ddu_gpu[i] - ddu_cpu[i]).abs() < 5.0e-5,
                "i={i} gpu={} cpu={}",
                ddu_gpu[i],
                ddu_cpu[i]
            );
        }
        assert_eq!(qd_gpu, qd_cpu);
    }
}
