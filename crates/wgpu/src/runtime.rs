use reed_core::{QFunctionContext, ReedError, ReedResult};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Uniform block for [`GpuRuntime::dispatch_mass_apply_qp_f32`] / transpose dispatch (`n` quadrature scalars).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MassApplyQpParamsGpu {
    n: u32,
    _pad: [u32; 3],
}

fn map_readback_f32_result(
    device: &wgpu::Device,
    readback: &wgpu::Buffer,
    out: &mut [f32],
) -> ReedResult<()> {
    let byte_len = out
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| ReedError::QFunction("map_readback_f32: length overflow".into()))?;
    let slice = readback.slice(..byte_len as u64);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReedError::QFunction(format!("map_readback_f32: recv {e}")))?
        .map_err(|e| ReedError::QFunction(format!("map_readback_f32: map {e:?}")))?;
    let data = slice.get_mapped_range();
    if data.len() != byte_len {
        return Err(ReedError::QFunction(
            "map_readback_f32: mapped range size mismatch".into(),
        ));
    }
    for (o, chunk) in out.iter_mut().zip(data.chunks_exact(4)) {
        *o = f32::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| ReedError::QFunction("map_readback_f32: chunk".into()))?,
        );
    }
    drop(data);
    readback.unmap();
    Ok(())
}

pub struct GpuRuntime {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    set_layout: wgpu::BindGroupLayout,
    set_pipeline: wgpu::ComputePipeline,
    scale_layout: wgpu::BindGroupLayout,
    scale_pipeline: wgpu::ComputePipeline,
    axpy_layout: wgpu::BindGroupLayout,
    axpy_pipeline: wgpu::ComputePipeline,
    restriction_layout: wgpu::BindGroupLayout,
    restriction_pipeline: wgpu::ComputePipeline,
    restriction_scatter_pipeline: wgpu::ComputePipeline,
    restriction_strided_layout: wgpu::BindGroupLayout,
    restriction_strided_pipeline: wgpu::ComputePipeline,
    restriction_strided_scatter_pipeline: wgpu::ComputePipeline,
    restriction_gather_f64_pipeline: wgpu::ComputePipeline,
    restriction_strided_gather_f64_pipeline: wgpu::ComputePipeline,
    basis_interp_layout: wgpu::BindGroupLayout,
    basis_interp_pipeline: wgpu::ComputePipeline,
    basis_interp_transpose_pipeline: wgpu::ComputePipeline,
    basis_grad_pipeline: wgpu::ComputePipeline,
    basis_grad_transpose_pipeline: wgpu::ComputePipeline,
    basis_post_layout: wgpu::BindGroupLayout,
    basis_post_pipeline: wgpu::ComputePipeline,
    basis_weight_layout: wgpu::BindGroupLayout,
    basis_weight_pipeline: wgpu::ComputePipeline,
    /// Vector-valued basis interp (Nédélec / Raviart-Thomas). Reuses [`basis_interp_layout`].
    basis_vector_interp_pipeline: wgpu::ComputePipeline,
    basis_vector_interp_transpose_pipeline: wgpu::ComputePipeline,
    /// Scalar forward/transpose (Nédélec curl 2D, Raviart-Thomas div). Reuses [`basis_interp_layout`].
    basis_scalar_forward_pipeline: wgpu::ComputePipeline,
    basis_scalar_transpose_pipeline: wgpu::ComputePipeline,
    /// Nédélec curl 3D forward/transpose. Reuses [`basis_interp_layout`].
    basis_curl3d_pipeline: wgpu::ComputePipeline,
    basis_curl3d_transpose_pipeline: wgpu::ComputePipeline,
    /// Gallery-style scalar `out[i] = in0[i] * in1[i]` at quadrature points (Mass apply, Poisson1D apply, …).
    qfunction_pointwise_mul_layout: wgpu::BindGroupLayout,
    qfunction_pointwise_mul_pipeline: wgpu::ComputePipeline,
    /// [`reed_cpu::Vec2Dot`] / [`reed_cpu::Vec3Dot`]: dot product per quadrature point (same 4-slot layout).
    qfunction_vec2_dot_pipeline: wgpu::ComputePipeline,
    qfunction_vec3_dot_pipeline: wgpu::ComputePipeline,
    /// [`reed_cpu::Mass1DBuild`] / [`reed_cpu::Mass2DBuild`] / [`reed_cpu::Mass3DBuild`]: quadrature `qdata` from `dx` + weights (same 4-slot layout).
    qfunction_mass1d_build_pipeline: wgpu::ComputePipeline,
    qfunction_mass2d_build_pipeline: wgpu::ComputePipeline,
    qfunction_mass3d_build_pipeline: wgpu::ComputePipeline,
    /// [`reed_cpu::Poisson1DBuild`] / [`reed_cpu::Poisson2DBuild`] / [`reed_cpu::Poisson3DBuild`] (same 4-slot layout).
    qfunction_poisson1d_build_pipeline: wgpu::ComputePipeline,
    qfunction_poisson2d_build_pipeline: wgpu::ComputePipeline,
    qfunction_poisson3d_build_pipeline: wgpu::ComputePipeline,
    /// [`reed_cpu::Vector2MassApply`] / [`reed_cpu::Vector2Poisson1DApply`]: `v[2*i+c] = qdata[i] * u[2*i+c]` (same bind layout as pointwise mul).
    qfunction_vector2_mass_apply_pipeline: wgpu::ComputePipeline,
    /// [`reed_cpu::Vector3MassApply`] / [`reed_cpu::Vector3Poisson1DApply`]: `v[3*i+c] = qdata[i] * u[3*i+c]` (same bind layout).
    qfunction_vector3_mass_apply_pipeline: wgpu::ComputePipeline,
    /// [`reed_cpu::Poisson2DApply`]: 2×2 `qdata` times 2-vector `du` per quadrature point (same 4-slot layout).
    qfunction_poisson2d_apply_pipeline: wgpu::ComputePipeline,
    /// [`reed_cpu::Poisson3DApply`]: 3×3 `qdata` times 3-vector `du` per quadrature point (same 4-slot layout).
    qfunction_poisson3d_apply_pipeline: wgpu::ComputePipeline,
    /// [`reed_cpu::Vector2Poisson2DApply`]: same 2×2 `qdata` applied to two stacked 2-gradients per point.
    qfunction_vector2_poisson2d_apply_pipeline: wgpu::ComputePipeline,
    /// [`reed_cpu::Vector3Poisson2DApply`]: same 2×2 `qdata` applied to three stacked 2-gradients per point.
    qfunction_vector3_poisson2d_apply_pipeline: wgpu::ComputePipeline,
    /// Cotangent accumulate for [`reed_cpu::Poisson2DApply::apply_operator_transpose`].
    qfunction_poisson2d_transpose_pipeline: wgpu::ComputePipeline,
    /// Cotangent accumulate for [`reed_cpu::Poisson3DApply::apply_operator_transpose`].
    qfunction_poisson3d_transpose_pipeline: wgpu::ComputePipeline,
    qfunction_vector2_poisson2d_transpose_pipeline: wgpu::ComputePipeline,
    qfunction_vector3_poisson2d_transpose_pipeline: wgpu::ComputePipeline,
    /// [`reed_cpu::Vector3Poisson3DApply`]: shared 3×3 `qdata` on three stacked 3-gradients (`du`/`dv` length `9 * num_q`).
    qfunction_vector3_poisson3d_apply_pipeline: wgpu::ComputePipeline,
    qfunction_vector3_poisson3d_transpose_pipeline: wgpu::ComputePipeline,
    /// Uniform + input SSBO + output SSBO: [`Identity`](reed_cpu::Identity) copy and [`Scale`](reed_cpu::Scale) `f32` path.
    qfunction_unary_layout: wgpu::BindGroupLayout,
    qfunction_identity_copy_pipeline: wgpu::ComputePipeline,
    qfunction_identity_transpose_accumulate_pipeline: wgpu::ComputePipeline,
    /// [`reed_cpu::IdentityScalar`]: `out[i] = in[i * ncomp]` (same unary bind layout as identity copy).
    qfunction_identity_scalar_gather_pipeline: wgpu::ComputePipeline,
    qfunction_identity_scalar_transpose_accumulate_pipeline: wgpu::ComputePipeline,
    qfunction_scale_f32_pipeline: wgpu::ComputePipeline,
    qfunction_scale_transpose_accumulate_pipeline: wgpu::ComputePipeline,
    mass_apply_qp_layout: wgpu::BindGroupLayout,
    mass_apply_qp_pipeline: wgpu::ComputePipeline,
    mass_apply_qp_transpose_pipeline: wgpu::ComputePipeline,
}

impl GpuRuntime {
    /// Synchronous init (native platforms — uses pollster internally).
    pub fn new(adapter: &wgpu::Adapter) -> Option<Self> {
        let req = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("reed-wgpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ));
        let (device, queue) = req.ok()?;
        Some(Self::from_device_queue(device, queue))
    }

    /// Async init for WASM (no pollster — await the WebGPU futures).
    pub async fn new_async(
        instance: &wgpu::Instance,
        power_pref: wgpu::PowerPreference,
        force_fallback: bool,
    ) -> Option<Self> {
        let Some(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: power_pref,
                force_fallback_adapter: force_fallback,
                compatible_surface: None,
            })
            .await
        else {
            return None;
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("reed-wgpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                },
                None,
            )
            .await
            .ok()?;

        Some(Self::from_device_queue(device, queue))
    }

    /// Build from an already-instantiated device + queue.
    pub fn from_device_queue(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let set_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reed-set-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let scale_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reed-scale-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let axpy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reed-axpy-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let restriction_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reed-restriction-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let restriction_strided_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reed-restriction-strided-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let basis_interp_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reed-basis-interp-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let shader_main = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-kernels-main"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(KERNELS_WGSL)),
        });
        let shader_scatter = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-restriction-scatter"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(RESTRICTION_SCATTER_WGSL)),
        });
        let shader_restriction_f64_offset =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("reed-restriction-f64-offset"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    RESTRICTION_OFFSET_GATHER_F64_BITS_WGSL,
                )),
            });
        let shader_restriction_f64_strided =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("reed-restriction-f64-strided"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    RESTRICTION_STRIDED_GATHER_F64_BITS_WGSL,
                )),
            });
        let set_pipeline =
            create_pipeline_with_module(&device, &set_layout, &shader_main, "set_main");
        let scale_pipeline =
            create_pipeline_with_module(&device, &scale_layout, &shader_main, "scale_main");
        let axpy_pipeline =
            create_pipeline_with_module(&device, &axpy_layout, &shader_main, "axpy_main");
        let restriction_pipeline = create_pipeline_with_module(
            &device,
            &restriction_layout,
            &shader_main,
            "restriction_gather_main",
        );
        let restriction_scatter_pipeline = create_pipeline_with_module(
            &device,
            &restriction_layout,
            &shader_scatter,
            "restriction_scatter_main",
        );
        let restriction_strided_pipeline = create_pipeline_with_module(
            &device,
            &restriction_strided_layout,
            &shader_main,
            "restriction_strided_gather_main",
        );
        let restriction_strided_scatter_pipeline = create_pipeline_with_module(
            &device,
            &restriction_strided_layout,
            &shader_main,
            "restriction_strided_scatter_main",
        );
        let restriction_gather_f64_pipeline = create_pipeline_with_module(
            &device,
            &restriction_layout,
            &shader_restriction_f64_offset,
            "restriction_gather_f64_bits_main",
        );
        let restriction_strided_gather_f64_pipeline = create_pipeline_with_module(
            &device,
            &restriction_strided_layout,
            &shader_restriction_f64_strided,
            "restriction_strided_gather_f64_bits_main",
        );
        let basis_interp_pipeline = create_pipeline_with_module(
            &device,
            &basis_interp_layout,
            &shader_main,
            "basis_interp_main",
        );
        let basis_interp_transpose_pipeline = create_pipeline_with_module(
            &device,
            &basis_interp_layout,
            &shader_main,
            "basis_interp_transpose_main",
        );
        let basis_grad_pipeline = create_pipeline_with_module(
            &device,
            &basis_interp_layout,
            &shader_main,
            "basis_grad_main",
        );
        let basis_grad_transpose_pipeline = create_pipeline_with_module(
            &device,
            &basis_interp_layout,
            &shader_main,
            "basis_grad_transpose_main",
        );

        let basis_post_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reed-basis-post-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let basis_post_pipeline = create_pipeline_with_module(
            &device,
            &basis_post_layout,
            &shader_main,
            "basis_post_main",
        );

        let basis_weight_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reed-basis-weight-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let shader_weight = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-basis-weight"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BASIS_WEIGHT_WGSL)),
        });
        let basis_weight_pipeline = create_pipeline_with_module(
            &device,
            &basis_weight_layout,
            &shader_weight,
            "basis_weight_tile_main",
        );

        let basis_vector_interp_pipeline = create_pipeline_with_module(
            &device,
            &basis_interp_layout,
            &shader_main,
            "basis_vector_interp_main",
        );
        let basis_vector_interp_transpose_pipeline = create_pipeline_with_module(
            &device,
            &basis_interp_layout,
            &shader_main,
            "basis_vector_interp_transpose_main",
        );
        let basis_scalar_forward_pipeline = create_pipeline_with_module(
            &device,
            &basis_interp_layout,
            &shader_main,
            "basis_scalar_forward_main",
        );
        let basis_scalar_transpose_pipeline = create_pipeline_with_module(
            &device,
            &basis_interp_layout,
            &shader_main,
            "basis_scalar_transpose_main",
        );
        let basis_curl3d_pipeline = create_pipeline_with_module(
            &device,
            &basis_interp_layout,
            &shader_main,
            "basis_curl3d_main",
        );
        let basis_curl3d_transpose_pipeline = create_pipeline_with_module(
            &device,
            &basis_interp_layout,
            &shader_main,
            "basis_curl3d_transpose_main",
        );

        let qfunction_pointwise_mul_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reed-qf-pointwise-mul-f32"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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
        let shader_qf_pointwise = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-pointwise-mul-f32"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                QFUNCTION_POINTWISE_MUL_WGSL,
            )),
        });
        let qfunction_pointwise_mul_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_pointwise,
            "qf_pointwise_mul_f32",
        );
        let shader_qf_vec_dot = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-vec-dot-f32"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(QFUNCTION_VEC_DOT_WGSL)),
        });
        let qfunction_vec2_dot_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_vec_dot,
            "qf_vec2_dot_f32",
        );
        let qfunction_vec3_dot_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_vec_dot,
            "qf_vec3_dot_f32",
        );
        let shader_qf_mass_build = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-mass-build-f32"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(QFUNCTION_MASS_BUILD_WGSL)),
        });
        let qfunction_mass1d_build_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_mass_build,
            "qf_mass1d_build_f32",
        );
        let qfunction_mass2d_build_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_mass_build,
            "qf_mass2d_build_f32",
        );
        let qfunction_mass3d_build_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_mass_build,
            "qf_mass3d_build_f32",
        );
        let shader_qf_poisson_build = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-poisson-build-f32"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                QFUNCTION_POISSON_BUILD_WGSL,
            )),
        });
        let qfunction_poisson1d_build_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_poisson_build,
            "qf_poisson1d_build_f32",
        );
        let qfunction_poisson2d_build_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_poisson_build,
            "qf_poisson2d_build_f32",
        );
        let qfunction_poisson3d_build_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_poisson_build,
            "qf_poisson3d_build_f32",
        );
        let shader_qf_vector2_mass = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-vector2-mass-apply-f32"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                QFUNCTION_VECTOR2_MASS_APPLY_WGSL,
            )),
        });
        let qfunction_vector2_mass_apply_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_vector2_mass,
            "vector2_mass_apply_f32",
        );
        let shader_qf_vector3_mass = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-vector3-mass-apply-f32"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                QFUNCTION_VECTOR3_MASS_APPLY_WGSL,
            )),
        });
        let qfunction_vector3_mass_apply_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_vector3_mass,
            "vector3_mass_apply_f32",
        );
        let shader_qf_poisson2d = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-poisson2d-apply-f32"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                QFUNCTION_POISSON2D_APPLY_WGSL,
            )),
        });
        let qfunction_poisson2d_apply_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_poisson2d,
            "poisson2d_apply_f32",
        );
        let shader_qf_poisson3d = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-poisson3d-apply-f32"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                QFUNCTION_POISSON3D_APPLY_WGSL,
            )),
        });
        let qfunction_poisson3d_apply_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_poisson3d,
            "poisson3d_apply_f32",
        );
        let shader_qf_v2p2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-vector2-poisson2d-apply-f32"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                QFUNCTION_VECTOR2_POISSON2D_APPLY_WGSL,
            )),
        });
        let qfunction_vector2_poisson2d_apply_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_v2p2,
            "vector2_poisson2d_apply_f32",
        );
        let shader_qf_v3p2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-vector3-poisson2d-apply-f32"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                QFUNCTION_VECTOR3_POISSON2D_APPLY_WGSL,
            )),
        });
        let qfunction_vector3_poisson2d_apply_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_v3p2,
            "vector3_poisson2d_apply_f32",
        );
        let qfunction_poisson2d_transpose_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_poisson2d,
            "poisson2d_transpose_accumulate_f32",
        );
        let qfunction_poisson3d_transpose_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_poisson3d,
            "poisson3d_transpose_accumulate_f32",
        );
        let qfunction_vector2_poisson2d_transpose_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_v2p2,
            "vector2_poisson2d_transpose_accumulate_f32",
        );
        let qfunction_vector3_poisson2d_transpose_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_v3p2,
            "vector3_poisson2d_transpose_accumulate_f32",
        );
        let qfunction_vector3_poisson3d_apply_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_poisson3d,
            "vector3_poisson3d_apply_f32",
        );
        let qfunction_vector3_poisson3d_transpose_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_pointwise_mul_layout,
            &shader_qf_poisson3d,
            "vector3_poisson3d_transpose_accumulate_f32",
        );

        let qfunction_unary_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reed-qf-unary-f32"),
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
        let shader_qf_identity = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-identity-copy-f32"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                QFUNCTION_IDENTITY_COPY_WGSL,
            )),
        });
        let shader_qf_scale = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-scale-f32"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(QFUNCTION_SCALE_F32_WGSL)),
        });
        let qfunction_identity_copy_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_unary_layout,
            &shader_qf_identity,
            "qf_identity_copy_f32",
        );
        let qfunction_scale_f32_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_unary_layout,
            &shader_qf_scale,
            "qf_scale_f32",
        );
        let qfunction_identity_transpose_accumulate_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_unary_layout,
            &shader_qf_identity,
            "qf_identity_transpose_accumulate_f32",
        );
        let shader_qf_identity_scalar = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reed-qf-identity-scalar-f32"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                QFUNCTION_IDENTITY_SCALAR_WGSL,
            )),
        });
        let qfunction_identity_scalar_gather_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_unary_layout,
            &shader_qf_identity_scalar,
            "qf_identity_scalar_gather_f32",
        );
        let qfunction_identity_scalar_transpose_accumulate_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_unary_layout,
            &shader_qf_identity_scalar,
            "qf_identity_scalar_transpose_accumulate_f32",
        );
        let qfunction_scale_transpose_accumulate_pipeline = create_pipeline_with_module(
            &device,
            &qfunction_unary_layout,
            &shader_qf_scale,
            "qf_scale_transpose_accumulate_f32",
        );

        let mass_apply_qp_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reed-mass-apply-qp-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let mass_apply_qp_pipeline = create_pipeline_with_module(
            &device,
            &mass_apply_qp_layout,
            &shader_main,
            "mass_apply_qp_main",
        );
        let mass_apply_qp_transpose_pipeline = create_pipeline_with_module(
            &device,
            &mass_apply_qp_layout,
            &shader_main,
            "mass_apply_qp_transpose_main",
        );

        Self {
            device,
            queue,
            set_layout,
            set_pipeline,
            scale_layout,
            scale_pipeline,
            axpy_layout,
            axpy_pipeline,
            restriction_layout,
            restriction_pipeline,
            restriction_scatter_pipeline,
            restriction_strided_layout,
            restriction_strided_pipeline,
            restriction_strided_scatter_pipeline,
            restriction_gather_f64_pipeline,
            restriction_strided_gather_f64_pipeline,
            basis_interp_layout,
            basis_interp_pipeline,
            basis_interp_transpose_pipeline,
            basis_grad_pipeline,
            basis_grad_transpose_pipeline,
            basis_post_layout,
            basis_post_pipeline,
            basis_weight_layout,
            basis_weight_pipeline,
            basis_vector_interp_pipeline,
            basis_vector_interp_transpose_pipeline,
            basis_scalar_forward_pipeline,
            basis_scalar_transpose_pipeline,
            basis_curl3d_pipeline,
            basis_curl3d_transpose_pipeline,
            qfunction_pointwise_mul_layout,
            qfunction_pointwise_mul_pipeline,
            qfunction_vec2_dot_pipeline,
            qfunction_vec3_dot_pipeline,
            qfunction_mass1d_build_pipeline,
            qfunction_mass2d_build_pipeline,
            qfunction_mass3d_build_pipeline,
            qfunction_poisson1d_build_pipeline,
            qfunction_poisson2d_build_pipeline,
            qfunction_poisson3d_build_pipeline,
            qfunction_vector2_mass_apply_pipeline,
            qfunction_vector3_mass_apply_pipeline,
            qfunction_poisson2d_apply_pipeline,
            qfunction_poisson3d_apply_pipeline,
            qfunction_vector2_poisson2d_apply_pipeline,
            qfunction_vector3_poisson2d_apply_pipeline,
            qfunction_poisson2d_transpose_pipeline,
            qfunction_poisson3d_transpose_pipeline,
            qfunction_vector2_poisson2d_transpose_pipeline,
            qfunction_vector3_poisson2d_transpose_pipeline,
            qfunction_vector3_poisson3d_apply_pipeline,
            qfunction_vector3_poisson3d_transpose_pipeline,
            qfunction_unary_layout,
            qfunction_identity_copy_pipeline,
            qfunction_identity_transpose_accumulate_pipeline,
            qfunction_identity_scalar_gather_pipeline,
            qfunction_identity_scalar_transpose_accumulate_pipeline,
            qfunction_scale_f32_pipeline,
            qfunction_scale_transpose_accumulate_pipeline,
            mass_apply_qp_layout,
            mass_apply_qp_pipeline,
            mass_apply_qp_transpose_pipeline,
        }
    }

    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    pub fn set_layout(&self) -> &wgpu::BindGroupLayout {
        &self.set_layout
    }

    pub fn set_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.set_pipeline
    }

    pub fn scale_layout(&self) -> &wgpu::BindGroupLayout {
        &self.scale_layout
    }

    pub fn scale_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.scale_pipeline
    }

    pub fn axpy_layout(&self) -> &wgpu::BindGroupLayout {
        &self.axpy_layout
    }

    pub fn axpy_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.axpy_pipeline
    }

    pub fn restriction_layout(&self) -> &wgpu::BindGroupLayout {
        &self.restriction_layout
    }

    pub fn restriction_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.restriction_pipeline
    }

    pub fn restriction_scatter_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.restriction_scatter_pipeline
    }

    pub fn restriction_strided_layout(&self) -> &wgpu::BindGroupLayout {
        &self.restriction_strided_layout
    }

    pub fn restriction_strided_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.restriction_strided_pipeline
    }

    pub fn restriction_strided_scatter_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.restriction_strided_scatter_pipeline
    }

    pub fn restriction_gather_f64_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.restriction_gather_f64_pipeline
    }

    pub fn restriction_strided_gather_f64_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.restriction_strided_gather_f64_pipeline
    }

    pub fn basis_interp_layout(&self) -> &wgpu::BindGroupLayout {
        &self.basis_interp_layout
    }

    pub fn basis_interp_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.basis_interp_pipeline
    }

    pub fn basis_interp_transpose_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.basis_interp_transpose_pipeline
    }

    pub fn basis_grad_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.basis_grad_pipeline
    }

    pub fn basis_grad_transpose_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.basis_grad_transpose_pipeline
    }

    pub fn basis_post_layout(&self) -> &wgpu::BindGroupLayout {
        &self.basis_post_layout
    }

    pub fn basis_post_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.basis_post_pipeline
    }

    pub fn basis_weight_layout(&self) -> &wgpu::BindGroupLayout {
        &self.basis_weight_layout
    }

    pub fn basis_weight_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.basis_weight_pipeline
    }

    pub fn basis_vector_interp_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.basis_vector_interp_pipeline
    }

    pub fn basis_vector_interp_transpose_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.basis_vector_interp_transpose_pipeline
    }

    pub fn basis_scalar_forward_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.basis_scalar_forward_pipeline
    }

    pub fn basis_scalar_transpose_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.basis_scalar_transpose_pipeline
    }

    pub fn basis_curl3d_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.basis_curl3d_pipeline
    }

    pub fn basis_curl3d_transpose_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.basis_curl3d_transpose_pipeline
    }

    pub fn qfunction_pointwise_mul_layout(&self) -> &wgpu::BindGroupLayout {
        &self.qfunction_pointwise_mul_layout
    }

    pub fn qfunction_pointwise_mul_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_pointwise_mul_pipeline
    }

    pub fn qfunction_vec2_dot_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_vec2_dot_pipeline
    }

    pub fn qfunction_vec3_dot_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_vec3_dot_pipeline
    }

    pub fn qfunction_mass1d_build_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_mass1d_build_pipeline
    }

    pub fn qfunction_mass2d_build_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_mass2d_build_pipeline
    }

    pub fn qfunction_mass3d_build_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_mass3d_build_pipeline
    }

    pub fn qfunction_poisson1d_build_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_poisson1d_build_pipeline
    }

    pub fn qfunction_poisson2d_build_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_poisson2d_build_pipeline
    }

    pub fn qfunction_poisson3d_build_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_poisson3d_build_pipeline
    }

    pub fn qfunction_vector2_mass_apply_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_vector2_mass_apply_pipeline
    }

    pub fn qfunction_vector3_mass_apply_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_vector3_mass_apply_pipeline
    }

    pub fn qfunction_poisson2d_apply_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_poisson2d_apply_pipeline
    }

    pub fn qfunction_poisson3d_apply_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_poisson3d_apply_pipeline
    }

    pub fn qfunction_vector2_poisson2d_apply_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_vector2_poisson2d_apply_pipeline
    }

    pub fn qfunction_vector3_poisson2d_apply_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_vector3_poisson2d_apply_pipeline
    }

    pub fn qfunction_poisson2d_transpose_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_poisson2d_transpose_pipeline
    }

    pub fn qfunction_poisson3d_transpose_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_poisson3d_transpose_pipeline
    }

    pub fn qfunction_vector2_poisson2d_transpose_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_vector2_poisson2d_transpose_pipeline
    }

    pub fn qfunction_vector3_poisson2d_transpose_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_vector3_poisson2d_transpose_pipeline
    }

    pub fn qfunction_vector3_poisson3d_apply_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_vector3_poisson3d_apply_pipeline
    }

    pub fn qfunction_vector3_poisson3d_transpose_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_vector3_poisson3d_transpose_pipeline
    }

    pub fn qfunction_unary_layout(&self) -> &wgpu::BindGroupLayout {
        &self.qfunction_unary_layout
    }

    pub fn qfunction_identity_copy_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_identity_copy_pipeline
    }

    pub fn qfunction_scale_f32_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_scale_f32_pipeline
    }

    pub fn qfunction_identity_transpose_accumulate_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_identity_transpose_accumulate_pipeline
    }

    pub fn qfunction_identity_scalar_gather_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_identity_scalar_gather_pipeline
    }

    pub fn qfunction_identity_scalar_transpose_accumulate_pipeline(
        &self,
    ) -> &wgpu::ComputePipeline {
        &self.qfunction_identity_scalar_transpose_accumulate_pipeline
    }

    pub fn qfunction_scale_transpose_accumulate_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.qfunction_scale_transpose_accumulate_pipeline
    }

    pub fn mass_apply_qp_layout(&self) -> &wgpu::BindGroupLayout {
        &self.mass_apply_qp_layout
    }

    pub fn mass_apply_qp_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.mass_apply_qp_pipeline
    }

    pub fn mass_apply_qp_transpose_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.mass_apply_qp_transpose_pipeline
    }

    /// Dispatch gallery **MassApply** forward at quadrature: `v[i] = u[i] * qdata[i]` for `i ∈ [0, n)`.
    ///
    /// Buffers must be at least `n * sizeof(f32)` bytes and usable as **storage** bindings
    /// (`STORAGE` + `COPY_DST` for uploads is typical). **`v` is overwritten** (not accumulated).
    pub fn dispatch_mass_apply_qp_f32(
        &self,
        u: &wgpu::Buffer,
        qdata: &wgpu::Buffer,
        v: &wgpu::Buffer,
        n: u32,
    ) -> ReedResult<()> {
        Self::dispatch_mass_apply_qp_inner(self, &self.mass_apply_qp_pipeline, u, qdata, v, n)
    }

    /// Dispatch **MassApply** transpose cotangent at quadrature:
    /// `du[i] += dv[i] * qdata[i]` (matches CPU [`reed_cpu::gallery::MassApply::apply_operator_transpose`]).
    ///
    /// `du` must be a read/write storage buffer; initialize to zero if the cotangent slot starts empty.
    pub fn dispatch_mass_apply_qp_transpose_accumulate_f32(
        &self,
        dv: &wgpu::Buffer,
        qdata: &wgpu::Buffer,
        du: &wgpu::Buffer,
        n: u32,
    ) -> ReedResult<()> {
        Self::dispatch_mass_apply_qp_inner(
            self,
            &self.mass_apply_qp_transpose_pipeline,
            dv,
            qdata,
            du,
            n,
        )
    }

    fn dispatch_mass_apply_qp_inner(
        rt: &GpuRuntime,
        pipeline: &wgpu::ComputePipeline,
        ro0: &wgpu::Buffer,
        ro1: &wgpu::Buffer,
        rw: &wgpu::Buffer,
        n: u32,
    ) -> ReedResult<()> {
        if n == 0 {
            return Ok(());
        }
        let need = (n as u64)
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| ReedError::QFunction("mass_apply_qp: size overflow".into()))?;
        for (label, buf) in [("binding0", ro0), ("binding1", ro1), ("binding2", rw)] {
            if buf.size() < need {
                return Err(ReedError::QFunction(format!(
                    "mass_apply_qp: {label} buffer size {} < {need} bytes for n={n}",
                    buf.size()
                )));
            }
        }
        let p = MassApplyQpParamsGpu { n, _pad: [0; 3] };
        let p_buffer = rt
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reed-mass-apply-qp-params"),
                contents: bytemuck::bytes_of(&p),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind = rt.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reed-mass-apply-qp-bind"),
            layout: &rt.mass_apply_qp_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ro0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ro1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rw.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = rt
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reed-mass-apply-qp-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reed-mass-apply-qp-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
        }
        rt.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Host convenience for gallery [`MassApply`](reed_cpu::gallery::MassApply) at quadrature:
    /// uploads `u` and `qdata`, runs [`Self::dispatch_mass_apply_qp_f32`], readbacks into **`v`**
    /// (fully overwritten). All slices must have equal length.
    pub fn mass_apply_qp_f32_host(
        &self,
        u: &[f32],
        qdata: &[f32],
        v: &mut [f32],
    ) -> ReedResult<()> {
        let n = u.len();
        if qdata.len() != n || v.len() != n {
            return Err(ReedError::QFunction(format!(
                "mass_apply_qp_f32_host: length mismatch u={} qdata={} v={}",
                u.len(),
                qdata.len(),
                v.len()
            )));
        }
        if n == 0 {
            return Ok(());
        }
        let n32 = n as u32;
        let byte_len = (n * std::mem::size_of::<f32>()) as u64;
        let u_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-ma-host-u"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let q_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-ma-host-q"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let v_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-ma-host-v"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&u_buf, 0, bytemuck::cast_slice(u));
        self.queue
            .write_buffer(&q_buf, 0, bytemuck::cast_slice(qdata));
        self.dispatch_mass_apply_qp_f32(&u_buf, &q_buf, &v_buf, n32)?;
        self.device.poll(wgpu::Maintain::Wait);

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-ma-host-readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reed-ma-host-copy"),
            });
        encoder.copy_buffer_to_buffer(&v_buf, 0, &readback, 0, byte_len);
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        map_readback_f32_result(&self.device, &readback, v)
    }

    /// Host convenience for **MassApply** transpose at quadrature: uploads `dv`, `qdata`, and
    /// current `du`, runs [`Self::dispatch_mass_apply_qp_transpose_accumulate_f32`], readbacks into **`du`**
    /// (in-place accumulation on device matches CPU gallery). All slices must have equal length.
    pub fn mass_apply_qp_transpose_accumulate_f32_host(
        &self,
        dv: &[f32],
        qdata: &[f32],
        du: &mut [f32],
    ) -> ReedResult<()> {
        let n = dv.len();
        if qdata.len() != n || du.len() != n {
            return Err(ReedError::QFunction(format!(
                "mass_apply_qp_transpose_accumulate_f32_host: length mismatch dv={} qdata={} du={}",
                dv.len(),
                qdata.len(),
                du.len()
            )));
        }
        if n == 0 {
            return Ok(());
        }
        let n32 = n as u32;
        let byte_len = (n * std::mem::size_of::<f32>()) as u64;
        let dv_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-mat-host-dv"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let q_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-mat-host-q"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let du_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-mat-host-du"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&dv_buf, 0, bytemuck::cast_slice(dv));
        self.queue
            .write_buffer(&q_buf, 0, bytemuck::cast_slice(qdata));
        self.queue
            .write_buffer(&du_buf, 0, bytemuck::cast_slice(du));
        self.dispatch_mass_apply_qp_transpose_accumulate_f32(&dv_buf, &q_buf, &du_buf, n32)?;
        self.device.poll(wgpu::Maintain::Wait);

        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-mat-host-readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reed-mat-host-copy"),
            });
        encoder.copy_buffer_to_buffer(&du_buf, 0, &readback, 0, byte_len);
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        map_readback_f32_result(&self.device, &readback, du)
    }

    /// Transpose for gallery kernels with **`components` scalar slots per quadrature point** sharing one
    /// scalar `qdata[i]`: `du[i * components + c] += dv[i * components + c] * qdata[i]`
    /// ([`reed_cpu::Vector2MassApply`], [`reed_cpu::Vector3MassApply`], and the Poisson1D vector applies
    /// that reuse the same multiply pattern).
    ///
    /// Implemented by expanding `qdata` to length `num_qp * components` and calling
    /// [`Self::mass_apply_qp_transpose_accumulate_f32_host`] (same WGSL kernel as scalar MassApply transpose).
    pub fn mass_apply_qp_transpose_broadcast_scalar_qdata_f32_host(
        &self,
        dv: &[f32],
        qdata: &[f32],
        du: &mut [f32],
        components: usize,
    ) -> ReedResult<()> {
        if components == 0 {
            return Err(ReedError::QFunction(
                "mass_apply_qp_transpose_broadcast_scalar_qdata_f32_host: components must be > 0"
                    .into(),
            ));
        }
        let num_qp = qdata.len();
        let n = num_qp
            .checked_mul(components)
            .ok_or_else(|| ReedError::QFunction("broadcast transpose: length overflow".into()))?;
        if dv.len() < n || du.len() < n {
            return Err(ReedError::QFunction(format!(
                "mass_apply_qp_transpose_broadcast_scalar_qdata_f32_host: need dv>={n}, du>={n} for num_qp={num_qp} components={components}; got dv={} du={}",
                dv.len(),
                du.len()
            )));
        }
        if num_qp == 0 {
            return Ok(());
        }
        let mut q_exp = vec![0.0_f32; n];
        for i in 0..num_qp {
            let s = qdata[i];
            let base = i * components;
            for c in 0..components {
                q_exp[base + c] = s;
            }
        }
        self.mass_apply_qp_transpose_accumulate_f32_host(&dv[..n], &q_exp, &mut du[..n])
    }

    /// When [`QFunctionContext::host_needs_device_upload`] is true, copies host context bytes into
    /// `buffer` at `buffer_offset` using [`wgpu::Queue::write_buffer`], then calls
    /// [`QFunctionContext::mark_host_synced_to_device`]. No-op if already clean or `byte_len() == 0`.
    ///
    /// `buffer` must be usable as a `write_buffer` destination (typically `COPY_DST` and, if bound as
    /// a uniform, sized per the adapter’s uniform alignment rules).
    pub fn sync_qfunction_context_to_buffer(
        &self,
        buffer: &wgpu::Buffer,
        buffer_offset: wgpu::BufferAddress,
        ctx: &QFunctionContext,
    ) -> ReedResult<()> {
        if !ctx.host_needs_device_upload() {
            return Ok(());
        }
        let bytes = ctx.as_bytes();
        let len = bytes.len() as u64;
        if len == 0 {
            ctx.mark_host_synced_to_device();
            return Ok(());
        }
        let end = buffer_offset.checked_add(len).ok_or_else(|| {
            ReedError::QFunction("sync_qfunction_context_to_buffer: size overflow".into())
        })?;
        if end > buffer.size() {
            return Err(ReedError::QFunction(format!(
                "sync_qfunction_context_to_buffer: need {} bytes from offset {}, buffer size {}",
                len,
                buffer_offset,
                buffer.size()
            )));
        }
        self.queue.write_buffer(buffer, buffer_offset, bytes);
        ctx.mark_host_synced_to_device();
        Ok(())
    }

    /// Upload context bytes regardless of dirty state, then mark host clean. Use for first bind or
    /// when the GPU buffer was recreated.
    pub fn write_qfunction_context_to_buffer(
        &self,
        buffer: &wgpu::Buffer,
        buffer_offset: wgpu::BufferAddress,
        ctx: &QFunctionContext,
    ) -> ReedResult<()> {
        let bytes = ctx.as_bytes();
        let len = bytes.len() as u64;
        if len == 0 {
            ctx.mark_host_synced_to_device();
            return Ok(());
        }
        let end = buffer_offset.checked_add(len).ok_or_else(|| {
            ReedError::QFunction("write_qfunction_context_to_buffer: size overflow".into())
        })?;
        if end > buffer.size() {
            return Err(ReedError::QFunction(format!(
                "write_qfunction_context_to_buffer: need {} bytes from offset {}, buffer size {}",
                len,
                buffer_offset,
                buffer.size()
            )));
        }
        self.queue.write_buffer(buffer, buffer_offset, bytes);
        ctx.mark_host_synced_to_device();
        Ok(())
    }
}

fn create_pipeline_with_module(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("reed-vector-pipeline-layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("reed-vector-pipeline"),
        layout: Some(&pipeline_layout),
        module,
        entry_point,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    })
}

const KERNELS_WGSL: &str = r#"
struct Params {
    alpha: f32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
};

struct RestrictionParams {
    nelem: u32,
    elemsize: u32,
    ncomp: u32,
    compstride: u32,
    local_size: u32,
    global_size: u32,
    _pad1: u32,
    _pad2: u32,
};

struct BasisInterpParams {
    num_elem: u32,
    num_dof: u32,
    num_qpoints: u32,
    ncomp: u32,
    output_size: u32,
    dim: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<uniform> p: Params;

@compute @workgroup_size(64)
fn set_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) {
        y[i] = p.alpha;
    }
}

@compute @workgroup_size(64)
fn scale_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) {
        y[i] = y[i] * p.alpha;
    }
}

@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<uniform> p2: Params;

@compute @workgroup_size(64)
fn axpy_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p2.n) {
        y[i] = p2.alpha * x[i] + y[i];
    }
}

@group(0) @binding(0) var<storage, read> rg_u: array<f32>;
@group(0) @binding(1) var<storage, read> rg_offsets: array<i32>;
@group(0) @binding(2) var<storage, read_write> rg_v: array<f32>;
@group(0) @binding(3) var<uniform> rg_p: RestrictionParams;

@compute @workgroup_size(64)
fn restriction_gather_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= rg_p.local_size) {
        return;
    }

    let per_elem = rg_p.ncomp * rg_p.elemsize;
    let elem = idx / per_elem;
    let rem = idx % per_elem;
    let comp = rem / rg_p.elemsize;
    let local = rem % rg_p.elemsize;

    let offset_idx = elem * rg_p.elemsize + local;
    let base = rg_offsets[offset_idx];
    if (base < 0) {
        rg_v[idx] = 0.0;
        return;
    }
    let g = u32(base) + comp * rg_p.compstride;
    rg_v[idx] = rg_u[g];
}

@group(0) @binding(0) var<storage, read> bi_mat: array<f32>;
@group(0) @binding(1) var<storage, read> bi_u: array<f32>;
@group(0) @binding(2) var<storage, read_write> bi_v: array<f32>;
@group(0) @binding(3) var<uniform> bi_p: BasisInterpParams;

@compute @workgroup_size(64)
fn basis_interp_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bi_p.output_size) {
        return;
    }

    let per_elem_out = bi_p.ncomp * bi_p.num_qpoints;
    let elem = idx / per_elem_out;
    let rem = idx % per_elem_out;
    let comp = rem / bi_p.num_qpoints;
    let qpt = rem % bi_p.num_qpoints;

    var sum = 0.0;
    let u_elem_base = (elem * bi_p.ncomp + comp) * bi_p.num_dof;
    let mat_row_base = qpt * bi_p.num_dof;
    for (var dof = 0u; dof < bi_p.num_dof; dof = dof + 1u) {
        sum = sum + bi_mat[mat_row_base + dof] * bi_u[u_elem_base + dof];
    }
    bi_v[idx] = sum;
}

@compute @workgroup_size(64)
fn basis_interp_transpose_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bi_p.output_size) {
        return;
    }

    let per_elem_out = bi_p.ncomp * bi_p.num_dof;
    let elem = idx / per_elem_out;
    let rem = idx % per_elem_out;
    let comp = rem / bi_p.num_dof;
    let dof = rem % bi_p.num_dof;

    var sum = 0.0;
    let u_elem_base = (elem * bi_p.ncomp + comp) * bi_p.num_qpoints;
    for (var qpt = 0u; qpt < bi_p.num_qpoints; qpt = qpt + 1u) {
        sum = sum + bi_mat[qpt * bi_p.num_dof + dof] * bi_u[u_elem_base + qpt];
    }
    bi_v[idx] = sum;
}

@compute @workgroup_size(64)
fn basis_grad_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bi_p.output_size) {
        return;
    }

    let dim = bi_p.dim;
    let nq = bi_p.num_qpoints;
    let per_elem_out = bi_p.ncomp * nq * dim;
    let elem = idx / per_elem_out;
    let rem = idx % per_elem_out;
    let comp = rem / (nq * dim);
    let rem2 = rem % (nq * dim);
    let qpt = rem2 / dim;
    let d_dir = rem2 % dim;

    var sum = 0.0;
    let u_elem_base = (elem * bi_p.ncomp + comp) * bi_p.num_dof;
    let mat_row = (qpt * dim + d_dir) * bi_p.num_dof;
    for (var dof = 0u; dof < bi_p.num_dof; dof = dof + 1u) {
        sum = sum + bi_mat[mat_row + dof] * bi_u[u_elem_base + dof];
    }
    // Interleaved quadrature layout per element: `iq * qcomp + comp * dim + dir` (matches CPU `LagrangeBasis`).
    let qcomp = bi_p.ncomp * dim;
    let out_pos = elem * (nq * qcomp) + qpt * qcomp + comp * dim + d_dir;
    bi_v[out_pos] = sum;
}

@compute @workgroup_size(64)
fn basis_grad_transpose_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bi_p.output_size) {
        return;
    }

    let dim = bi_p.dim;
    let nq = bi_p.num_qpoints;
    let per_elem_out = bi_p.ncomp * bi_p.num_dof;
    let elem = idx / per_elem_out;
    let rem = idx % per_elem_out;
    let comp = rem / bi_p.num_dof;
    let dof = rem % bi_p.num_dof;

    var sum = 0.0;
    let qcomp = bi_p.ncomp * dim;
    for (var iq = 0u; iq < nq; iq = iq + 1u) {
        for (var dd = 0u; dd < dim; dd = dd + 1u) {
            let row = iq * dim + dd;
            let u_idx = elem * (nq * qcomp) + iq * qcomp + comp * dim + dd;
            sum = sum + bi_mat[row * bi_p.num_dof + dof] * bi_u[u_idx];
        }
    }
    bi_v[idx] = sum;
}

struct BasisPostParams {
    mode: u32,
    num_elem: u32,
    num_qpoints: u32,
    dim: u32,
    ncomp: u32,
    qcomp: u32,
    out_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> bp_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> bp_out: array<f32>;
@group(0) @binding(2) var<uniform> bp_p: BasisPostParams;

@compute @workgroup_size(64)
fn basis_post_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bp_p.out_size) {
        return;
    }
    let nq = bp_p.num_qpoints;
    let dim = bp_p.dim;
    let qcomp = bp_p.qcomp;

    if (bp_p.mode == 0u) {
        let e = idx / nq;
        let iq = idx % nq;
        let g_base = (e * nq + iq) * qcomp;
        var s = 0.0;
        for (var d = 0u; d < dim; d = d + 1u) {
            s = s + bp_in[g_base + d * dim + d];
        }
        bp_out[idx] = s;
        return;
    }

    if (bp_p.mode == 1u) {
        let w = bp_in[idx];
        let e = idx / nq;
        let iq = idx % nq;
        let g_base = (e * nq + iq) * qcomp;
        for (var j = 0u; j < qcomp; j = j + 1u) {
            bp_out[g_base + j] = 0.0;
        }
        for (var d = 0u; d < dim; d = d + 1u) {
            bp_out[g_base + d * dim + d] = w;
        }
        return;
    }

    if (bp_p.mode == 2u) {
        let e = idx / nq;
        let iq = idx % nq;
        let g_base = (e * nq + iq) * qcomp;
        bp_out[idx] = bp_in[g_base + 2u] - bp_in[g_base + 1u];
        return;
    }

    if (bp_p.mode == 3u) {
        let w = bp_in[idx];
        let e = idx / nq;
        let iq = idx % nq;
        let g_base = (e * nq + iq) * qcomp;
        for (var j = 0u; j < qcomp; j = j + 1u) {
            bp_out[g_base + j] = 0.0;
        }
        bp_out[g_base + 1u] = -w;
        bp_out[g_base + 2u] = w;
        return;
    }

    if (bp_p.mode == 4u) {
        let nqpt = bp_p.num_qpoints;
        let per = idx / 3u;
        let comp = idx % 3u;
        let e = per / nqpt;
        let iq = per % nqpt;
        let g_base = (e * nqpt + iq) * qcomp;
        if (comp == 0u) {
            bp_out[idx] = bp_in[g_base + 7u] - bp_in[g_base + 5u];
        } else if (comp == 1u) {
            bp_out[idx] = bp_in[g_base + 2u] - bp_in[g_base + 6u];
        } else {
            bp_out[idx] = bp_in[g_base + 3u] - bp_in[g_base + 1u];
        }
        return;
    }

    if (bp_p.mode == 5u) {
        let w0 = bp_in[idx * 3u];
        let w1 = bp_in[idx * 3u + 1u];
        let w2 = bp_in[idx * 3u + 2u];
        let e = idx / nq;
        let iq = idx % nq;
        let g_base = (e * nq + iq) * qcomp;
        for (var j = 0u; j < qcomp; j = j + 1u) {
            bp_out[g_base + j] = 0.0;
        }
        bp_out[g_base + 7u] = w0;
        bp_out[g_base + 5u] = -w0;
        bp_out[g_base + 2u] = w1;
        bp_out[g_base + 6u] = -w1;
        bp_out[g_base + 3u] = w2;
        bp_out[g_base + 1u] = -w2;
        return;
    }
}

/// Gallery [`MassApply`](reed_cpu::gallery::MassApply) at quadrature points.
/// - `mass_apply_qp_main`: `rw[i] = ro0[i] * ro1[i]` (forward: `ro0=u`, `ro1=qdata`, `rw=v`).
/// - `mass_apply_qp_transpose_main`: `rw[i] += ro0[i] * ro1[i]` (transpose: `ro0=dv`, `ro1=qdata`, `rw=du`).
struct MassApplyQpParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> ma_ro0: array<f32>;
@group(0) @binding(1) var<storage, read> ma_ro1: array<f32>;
@group(0) @binding(2) var<storage, read_write> ma_rw: array<f32>;
@group(0) @binding(3) var<uniform> ma_p: MassApplyQpParams;

@compute @workgroup_size(64)
fn mass_apply_qp_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < ma_p.n) {
        ma_rw[i] = ma_ro0[i] * ma_ro1[i];
    }
}

@compute @workgroup_size(64)
fn mass_apply_qp_transpose_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < ma_p.n) {
        ma_rw[i] = ma_rw[i] + ma_ro0[i] * ma_ro1[i];
    }
}

struct StridedRestrictionParams {
    nelem: u32,
    elemsize: u32,
    ncomp: u32,
    _pad0: u32,
    s0: i32,
    s1: i32,
    s2: i32,
    _pad1: u32,
    local_size: u32,
    global_size: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> st_u: array<f32>;
@group(0) @binding(1) var<storage, read_write> st_v: array<f32>;
@group(0) @binding(2) var<uniform> st_p: StridedRestrictionParams;

@compute @workgroup_size(64)
fn restriction_strided_gather_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= st_p.local_size) {
        return;
    }
    let per_elem = st_p.ncomp * st_p.elemsize;
    let elem = idx / per_elem;
    let rem = idx % per_elem;
    let comp = rem / st_p.elemsize;
    let local = rem % st_p.elemsize;

    let g = i32(local) * st_p.s0 + i32(comp) * st_p.s1 + i32(elem) * st_p.s2;
    if (g < 0) {
        st_v[idx] = 0.0;
        return;
    }
    let gu = u32(g);
    if (gu >= st_p.global_size) {
        st_v[idx] = 0.0;
        return;
    }
    st_v[idx] = st_u[gu];
}

@compute @workgroup_size(1)
fn restriction_strided_scatter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    for (var idx = 0u; idx < st_p.local_size; idx = idx + 1u) {
        let per_elem = st_p.ncomp * st_p.elemsize;
        let elem = idx / per_elem;
        let rem = idx % per_elem;
        let comp = rem / st_p.elemsize;
        let local = rem % st_p.elemsize;

        let g = i32(local) * st_p.s0 + i32(comp) * st_p.s1 + i32(elem) * st_p.s2;
        if (g < 0) {
            continue;
        }
        let gu = u32(g);
        if (gu >= st_p.global_size) {
            continue;
        }
        let val = st_u[idx];
        st_v[gu] = st_v[gu] + val;
    }
}

// ── Vector-valued basis interp (Nédélec / Raviart-Thomas) ──────────────
// Reuses bi_mat, bi_u, bi_v, bi_p (BasisInterpParams).
// Input u: [elem * ncomp * num_dof], u[dof*ncomp] is the scalar DOF value.
// Matrix: [(qpt * num_dof + dof) * dim + d], size nq * nd * dim.
// Output v: [elem * nq * dim], vector at quadrature points.

@compute @workgroup_size(64)
fn basis_vector_interp_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bi_p.output_size) {
        return;
    }

    let dim = bi_p.dim;
    let nq = bi_p.num_qpoints;
    let nd = bi_p.num_dof;
    let ncomp = bi_p.ncomp;

    let per_elem_out = nq * dim;
    let elem = idx / per_elem_out;
    let rem = idx % per_elem_out;
    let qpt = rem / dim;
    let d = rem % dim;

    var sum = 0.0;
    let u_elem_base = elem * nd * ncomp;
    let mat_plane_base = qpt * nd;
    for (var dof = 0u; dof < nd; dof = dof + 1u) {
        sum = sum + bi_mat[(mat_plane_base + dof) * dim + d] * bi_u[u_elem_base + dof * ncomp];
    }
    bi_v[idx] = sum;
}

@compute @workgroup_size(64)
fn basis_vector_interp_transpose_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bi_p.output_size) {
        return;
    }

    let dim = bi_p.dim;
    let nq = bi_p.num_qpoints;
    let nd = bi_p.num_dof;
    let ncomp = bi_p.ncomp;

    let per_elem_out = nd * ncomp;
    let elem = idx / per_elem_out;
    let rem = idx % per_elem_out;
    let dof = rem / ncomp;
    let out_local = dof * ncomp;

    var sum = 0.0;
    let u_elem_base = elem * nq * dim;
    for (var qpt = 0u; qpt < nq; qpt = qpt + 1u) {
        let mat_plane_base = qpt * nd;
        for (var d = 0u; d < dim; d = d + 1u) {
            sum = sum + bi_mat[(mat_plane_base + dof) * dim + d] * bi_u[u_elem_base + qpt * dim + d];
        }
    }
    bi_v[elem * per_elem_out + out_local] += sum;
}

// ── Scalar forward/transpose (curl 2D, div) ────────────────────────────
// Matrix: [qpt * num_dof + dof], size nq * nd.
// Input u: [elem * nd * ncomp], u[dof*ncomp] is the scalar DOF value.
// Output v: [elem * nq], scalar per quadrature point.

@compute @workgroup_size(64)
fn basis_scalar_forward_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bi_p.output_size) {
        return;
    }

    let nq = bi_p.num_qpoints;
    let nd = bi_p.num_dof;
    let ncomp = bi_p.ncomp;

    let elem = idx / nq;
    let qpt = idx % nq;

    var sum = 0.0;
    let u_elem_base = elem * nd * ncomp;
    let mat_row = qpt * nd;
    for (var dof = 0u; dof < nd; dof = dof + 1u) {
        sum = sum + bi_mat[mat_row + dof] * bi_u[u_elem_base + dof * ncomp];
    }
    bi_v[idx] = sum;
}

@compute @workgroup_size(64)
fn basis_scalar_transpose_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bi_p.output_size) {
        return;
    }

    let nq = bi_p.num_qpoints;
    let nd = bi_p.num_dof;
    let ncomp = bi_p.ncomp;

    let per_elem_out = nd * ncomp;
    let elem = idx / per_elem_out;
    let rem = idx % per_elem_out;
    let dof = rem / ncomp;
    let out_local = dof * ncomp;

    var sum = 0.0;
    let u_elem_base = elem * nq;
    for (var qpt = 0u; qpt < nq; qpt = qpt + 1u) {
        sum = sum + bi_mat[qpt * nd + dof] * bi_u[u_elem_base + qpt];
    }
    bi_v[elem * per_elem_out + out_local] += sum;
}

// ── Curl 3D forward/transpose ──────────────────────────────────────────
// Matrix: [(qpt * num_dof + dof) * 3 + d], size nq * nd * 3.
// Input u: [elem * nd * ncomp], u[dof*ncomp] is the scalar DOF value.
// Output v: [elem * nq * 3], vector curl per quadrature point.

@compute @workgroup_size(64)
fn basis_curl3d_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bi_p.output_size) {
        return;
    }

    let nq = bi_p.num_qpoints;
    let nd = bi_p.num_dof;
    let ncomp = bi_p.ncomp;

    let per_elem_out = nq * 3u;
    let elem = idx / per_elem_out;
    let rem = idx % per_elem_out;
    let qpt = rem / 3u;
    let d = rem % 3u;

    var sum = 0.0;
    let u_elem_base = elem * nd * ncomp;
    let mat_plane_base = qpt * nd;
    for (var dof = 0u; dof < nd; dof = dof + 1u) {
        sum = sum + bi_mat[(mat_plane_base + dof) * 3u + d] * bi_u[u_elem_base + dof * ncomp];
    }
    bi_v[idx] = sum;
}

@compute @workgroup_size(64)
fn basis_curl3d_transpose_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bi_p.output_size) {
        return;
    }

    let nq = bi_p.num_qpoints;
    let nd = bi_p.num_dof;
    let ncomp = bi_p.ncomp;

    let per_elem_out = nd * ncomp;
    let elem = idx / per_elem_out;
    let rem = idx % per_elem_out;
    let dof = rem / ncomp;
    let out_local = dof * ncomp;

    var sum = 0.0;
    let u_elem_base = elem * nq * 3u;
    for (var qpt = 0u; qpt < nq; qpt = qpt + 1u) {
        for (var d = 0u; d < 3u; d = d + 1u) {
            sum = sum + bi_mat[(qpt * nd + dof) * 3u + d] * bi_u[u_elem_base + qpt * 3u + d];
        }
    }
    bi_v[elem * per_elem_out + out_local] += sum;
}
"#;

/// Scalar gallery QFunction apply: `out[i] = in0[i] * in1[i]` (e.g. `MassApply`, `Poisson1DApply`).
const QFUNCTION_POINTWISE_MUL_WGSL: &str = r#"
struct QfPointwiseMulParams {
    num_q: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> qfp: QfPointwiseMulParams;
@group(0) @binding(1) var<storage, read> qf_in0: array<f32>;
@group(0) @binding(2) var<storage, read> qf_in1: array<f32>;
@group(0) @binding(3) var<storage, read_write> qf_out: array<f32>;

@compute @workgroup_size(256)
fn qf_pointwise_mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qfp.num_q) {
        return;
    }
    qf_out[i] = qf_in0[i] * qf_in1[i];
}
"#;

/// [`reed_cpu::Vec2Dot`] / [`reed_cpu::Vec3Dot`] on `f32`: `w[i] = dot(u[i], v[i])` with packed `u`/`v`.
const QFUNCTION_VEC_DOT_WGSL: &str = r#"
struct QfVecDotParams {
    num_q: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> qp: QfVecDotParams;
@group(0) @binding(1) var<storage, read> qu: array<f32>;
@group(0) @binding(2) var<storage, read> qv: array<f32>;
@group(0) @binding(3) var<storage, read_write> qw: array<f32>;

@compute @workgroup_size(256)
fn qf_vec2_dot_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let b = i * 2u;
    qw[i] = qu[b] * qv[b] + qu[b + 1u] * qv[b + 1u];
}

@compute @workgroup_size(256)
fn qf_vec3_dot_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let b = i * 3u;
    qw[i] = qu[b] * qv[b] + qu[b + 1u] * qv[b + 1u] + qu[b + 2u] * qv[b + 2u];
}
"#;

/// [`reed_cpu::Mass1DBuild`] / [`reed_cpu::Mass2DBuild`] / [`reed_cpu::Mass3DBuild`] on `f32`.
const QFUNCTION_MASS_BUILD_WGSL: &str = r#"
struct QfMassBuildParams {
    num_q: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> qp: QfMassBuildParams;
@group(0) @binding(1) var<storage, read> qdx: array<f32>;
@group(0) @binding(2) var<storage, read> qw: array<f32>;
@group(0) @binding(3) var<storage, read_write> qqdata: array<f32>;

@compute @workgroup_size(256)
fn qf_mass1d_build_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    qqdata[i] = abs(qdx[i]) * qw[i];
}

@compute @workgroup_size(256)
fn qf_mass2d_build_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let b = i * 4u;
    let g00 = qdx[b];
    let g01 = qdx[b + 1u];
    let g10 = qdx[b + 2u];
    let g11 = qdx[b + 3u];
    let det_j = g00 * g11 - g01 * g10;
    qqdata[i] = abs(det_j) * qw[i];
}

@compute @workgroup_size(256)
fn qf_mass3d_build_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let b = i * 9u;
    let j00 = qdx[b];
    let j01 = qdx[b + 1u];
    let j02 = qdx[b + 2u];
    let j10 = qdx[b + 3u];
    let j11 = qdx[b + 4u];
    let j12 = qdx[b + 5u];
    let j20 = qdx[b + 6u];
    let j21 = qdx[b + 7u];
    let j22 = qdx[b + 8u];
    let det_j = j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20) + j02 * (j10 * j21 - j11 * j20);
    qqdata[i] = abs(det_j) * qw[i];
}
"#;

/// [`reed_cpu::Poisson1DBuild`] / [`reed_cpu::Poisson2DBuild`] / [`reed_cpu::Poisson3DBuild`] on `f32`.
/// Near-singular Jacobians are rejected on the **host** before dispatch (same threshold as CPU gallery).
const QFUNCTION_POISSON_BUILD_WGSL: &str = r#"
struct QfPoissonBuildParams {
    num_q: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> qp: QfPoissonBuildParams;
@group(0) @binding(1) var<storage, read> qdx: array<f32>;
@group(0) @binding(2) var<storage, read> qw: array<f32>;
@group(0) @binding(3) var<storage, read_write> qqdata: array<f32>;

@compute @workgroup_size(256)
fn qf_poisson1d_build_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    qqdata[i] = qw[i] / qdx[i];
}

@compute @workgroup_size(256)
fn qf_poisson2d_build_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let b = i * 4u;
    let j00 = qdx[b];
    let j01 = qdx[b + 1u];
    let j10 = qdx[b + 2u];
    let j11 = qdx[b + 3u];
    let det_j = j00 * j11 - j01 * j10;
    let inv00 = j11 / det_j;
    let inv01 = -j01 / det_j;
    let inv10 = -j10 / det_j;
    let inv11 = j00 / det_j;
    let scale = abs(det_j) * qw[i];
    let g00 = scale * (inv00 * inv00 + inv01 * inv01);
    let g01 = scale * (inv00 * inv10 + inv01 * inv11);
    let g10 = scale * (inv10 * inv00 + inv11 * inv01);
    let g11 = scale * (inv10 * inv10 + inv11 * inv11);
    qqdata[b] = g00;
    qqdata[b + 1u] = g01;
    qqdata[b + 2u] = g10;
    qqdata[b + 3u] = g11;
}

@compute @workgroup_size(256)
fn qf_poisson3d_build_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let b = i * 9u;
    let j00 = qdx[b];
    let j01 = qdx[b + 1u];
    let j02 = qdx[b + 2u];
    let j10 = qdx[b + 3u];
    let j11 = qdx[b + 4u];
    let j12 = qdx[b + 5u];
    let j20 = qdx[b + 6u];
    let j21 = qdx[b + 7u];
    let j22 = qdx[b + 8u];

    let c00 = j11 * j22 - j12 * j21;
    let c01 = -(j10 * j22 - j12 * j20);
    let c02 = j10 * j21 - j11 * j20;
    let c10 = -(j01 * j22 - j02 * j21);
    let c11 = j00 * j22 - j02 * j20;
    let c12 = -(j00 * j21 - j01 * j20);
    let c20 = j01 * j12 - j02 * j11;
    let c21 = -(j00 * j12 - j02 * j10);
    let c22 = j00 * j11 - j01 * j10;

    let det_j = j00 * c00 + j01 * c01 + j02 * c02;

    let inv00 = c00 / det_j;
    let inv01 = c10 / det_j;
    let inv02 = c20 / det_j;
    let inv10 = c01 / det_j;
    let inv11 = c11 / det_j;
    let inv12 = c21 / det_j;
    let inv20 = c02 / det_j;
    let inv21 = c12 / det_j;
    let inv22 = c22 / det_j;

    let s = abs(det_j) * qw[i];
    qqdata[b] = s * (inv00 * inv00 + inv01 * inv01 + inv02 * inv02);
    qqdata[b + 1u] = s * (inv00 * inv10 + inv01 * inv11 + inv02 * inv12);
    qqdata[b + 2u] = s * (inv00 * inv20 + inv01 * inv21 + inv02 * inv22);
    qqdata[b + 3u] = s * (inv10 * inv00 + inv11 * inv01 + inv12 * inv02);
    qqdata[b + 4u] = s * (inv10 * inv10 + inv11 * inv11 + inv12 * inv12);
    qqdata[b + 5u] = s * (inv10 * inv20 + inv11 * inv21 + inv12 * inv22);
    qqdata[b + 6u] = s * (inv20 * inv00 + inv21 * inv01 + inv22 * inv02);
    qqdata[b + 7u] = s * (inv20 * inv10 + inv21 * inv11 + inv22 * inv12);
    qqdata[b + 8u] = s * (inv20 * inv20 + inv21 * inv21 + inv22 * inv22);
}
"#;

/// [`reed_cpu::Vector2MassApply`] on `f32`: two components per quadrature point, one scalar `qdata` per point.
const QFUNCTION_VECTOR2_MASS_APPLY_WGSL: &str = r#"
struct QfVec2MassParams {
    num_qp: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> qp: QfVec2MassParams;
@group(0) @binding(1) var<storage, read> qu: array<f32>;
@group(0) @binding(2) var<storage, read> qqdata: array<f32>;
@group(0) @binding(3) var<storage, read_write> qv: array<f32>;

@compute @workgroup_size(256)
fn vector2_mass_apply_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let n = qp.num_qp * 2u;
    if (flat >= n) {
        return;
    }
    let iq = flat / 2u;
    qv[flat] = qqdata[iq] * qu[flat];
}
"#;

/// [`reed_cpu::Vector3MassApply`] on `f32`: three components per quadrature point, one scalar `qdata` per point.
const QFUNCTION_VECTOR3_MASS_APPLY_WGSL: &str = r#"
struct QfVec3MassParams {
    num_qp: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> qp: QfVec3MassParams;
@group(0) @binding(1) var<storage, read> qu: array<f32>;
@group(0) @binding(2) var<storage, read> qqdata: array<f32>;
@group(0) @binding(3) var<storage, read_write> qv: array<f32>;

@compute @workgroup_size(256)
fn vector3_mass_apply_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let n = qp.num_qp * 3u;
    if (flat >= n) {
        return;
    }
    let iq = flat / 3u;
    qv[flat] = qqdata[iq] * qu[flat];
}
"#;

/// [`reed_cpu::Poisson2DApply`] on `f32`: `dv[2*i+0] = g00*du0+g01*du1`, `dv[2*i+1] = g10*du0+g11*du1` with `qdata[4*i+0..4]` storing the 2×2 stiffness block.
const QFUNCTION_POISSON2D_APPLY_WGSL: &str = r#"
struct QfPoisson2DParams {
    num_q: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> qp: QfPoisson2DParams;
@group(0) @binding(1) var<storage, read> qdu: array<f32>;
@group(0) @binding(2) var<storage, read> qqdata: array<f32>;
@group(0) @binding(3) var<storage, read_write> qdv: array<f32>;

@compute @workgroup_size(256)
fn poisson2d_apply_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let du0 = qdu[i * 2u];
    let du1 = qdu[i * 2u + 1u];
    let b = i * 4u;
    let g00 = qqdata[b + 0u];
    let g01 = qqdata[b + 1u];
    let g10 = qqdata[b + 2u];
    let g11 = qqdata[b + 3u];
    qdv[i * 2u] = g00 * du0 + g01 * du1;
    qdv[i * 2u + 1u] = g10 * du0 + g11 * du1;
}

// qdu = ddv (read), qdv = ddu (read_write accumulate); matches CPU Poisson2DApply transpose.
@compute @workgroup_size(256)
fn poisson2d_transpose_accumulate_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let ddv0 = qdu[i * 2u];
    let ddv1 = qdu[i * 2u + 1u];
    let b = i * 4u;
    let g00 = qqdata[b + 0u];
    let g01 = qqdata[b + 1u];
    let g10 = qqdata[b + 2u];
    let g11 = qqdata[b + 3u];
    let o0 = i * 2u;
    qdv[o0] = qdv[o0] + g00 * ddv0 + g10 * ddv1;
    qdv[o0 + 1u] = qdv[o0 + 1u] + g01 * ddv0 + g11 * ddv1;
}
"#;

/// [`reed_cpu::Poisson3DApply`] on `f32`: row-major 3×3 `qdata[9*i+..]` times `du[3*i+..]` into `dv[3*i+..]`.
const QFUNCTION_POISSON3D_APPLY_WGSL: &str = r#"
struct QfPoisson3DParams {
    num_q: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> qp: QfPoisson3DParams;
@group(0) @binding(1) var<storage, read> qdu: array<f32>;
@group(0) @binding(2) var<storage, read> qqdata: array<f32>;
@group(0) @binding(3) var<storage, read_write> qdv: array<f32>;

@compute @workgroup_size(256)
fn poisson3d_apply_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let du0 = qdu[i * 3u];
    let du1 = qdu[i * 3u + 1u];
    let du2 = qdu[i * 3u + 2u];
    let b = i * 9u;
    let g00 = qqdata[b + 0u];
    let g01 = qqdata[b + 1u];
    let g02 = qqdata[b + 2u];
    let g10 = qqdata[b + 3u];
    let g11 = qqdata[b + 4u];
    let g12 = qqdata[b + 5u];
    let g20 = qqdata[b + 6u];
    let g21 = qqdata[b + 7u];
    let g22 = qqdata[b + 8u];
    qdv[i * 3u] = g00 * du0 + g01 * du1 + g02 * du2;
    qdv[i * 3u + 1u] = g10 * du0 + g11 * du1 + g12 * du2;
    qdv[i * 3u + 2u] = g20 * du0 + g21 * du1 + g22 * du2;
}

@compute @workgroup_size(256)
fn poisson3d_transpose_accumulate_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let ddv0 = qdu[i * 3u];
    let ddv1 = qdu[i * 3u + 1u];
    let ddv2 = qdu[i * 3u + 2u];
    let b = i * 9u;
    let g00 = qqdata[b + 0u];
    let g01 = qqdata[b + 1u];
    let g02 = qqdata[b + 2u];
    let g10 = qqdata[b + 3u];
    let g11 = qqdata[b + 4u];
    let g12 = qqdata[b + 5u];
    let g20 = qqdata[b + 6u];
    let g21 = qqdata[b + 7u];
    let g22 = qqdata[b + 8u];
    let o0 = i * 3u;
    qdv[o0] = qdv[o0] + g00 * ddv0 + g10 * ddv1 + g20 * ddv2;
    qdv[o0 + 1u] = qdv[o0 + 1u] + g01 * ddv0 + g11 * ddv1 + g21 * ddv2;
    qdv[o0 + 2u] = qdv[o0 + 2u] + g02 * ddv0 + g12 * ddv1 + g22 * ddv2;
}

@compute @workgroup_size(256)
fn vector3_poisson3d_apply_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let b = i * 9u;
    let g00 = qqdata[b + 0u];
    let g01 = qqdata[b + 1u];
    let g02 = qqdata[b + 2u];
    let g10 = qqdata[b + 3u];
    let g11 = qqdata[b + 4u];
    let g12 = qqdata[b + 5u];
    let g20 = qqdata[b + 6u];
    let g21 = qqdata[b + 7u];
    let g22 = qqdata[b + 8u];
    for (var c = 0u; c < 3u; c = c + 1u) {
        let base = c * 3u;
        let du0 = qdu[b + base];
        let du1 = qdu[b + base + 1u];
        let du2 = qdu[b + base + 2u];
        qdv[b + base] = g00 * du0 + g01 * du1 + g02 * du2;
        qdv[b + base + 1u] = g10 * du0 + g11 * du1 + g12 * du2;
        qdv[b + base + 2u] = g20 * du0 + g21 * du1 + g22 * du2;
    }
}

@compute @workgroup_size(256)
fn vector3_poisson3d_transpose_accumulate_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let b = i * 9u;
    let g00 = qqdata[b + 0u];
    let g01 = qqdata[b + 1u];
    let g02 = qqdata[b + 2u];
    let g10 = qqdata[b + 3u];
    let g11 = qqdata[b + 4u];
    let g12 = qqdata[b + 5u];
    let g20 = qqdata[b + 6u];
    let g21 = qqdata[b + 7u];
    let g22 = qqdata[b + 8u];
    for (var c = 0u; c < 3u; c = c + 1u) {
        let base = c * 3u;
        let ddv0 = qdu[b + base];
        let ddv1 = qdu[b + base + 1u];
        let ddv2 = qdu[b + base + 2u];
        qdv[b + base] = qdv[b + base] + g00 * ddv0 + g10 * ddv1 + g20 * ddv2;
        qdv[b + base + 1u] = qdv[b + base + 1u] + g01 * ddv0 + g11 * ddv1 + g21 * ddv2;
        qdv[b + base + 2u] = qdv[b + base + 2u] + g02 * ddv0 + g12 * ddv1 + g22 * ddv2;
    }
}
"#;

/// [`reed_cpu::Vector2Poisson2DApply`] on `f32`: shared 2×2 `qdata[i*4..]` times each of two 2-gradients packed in `du[i*4..]`.
const QFUNCTION_VECTOR2_POISSON2D_APPLY_WGSL: &str = r#"
struct QfV2P2Params {
    num_q: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> qp: QfV2P2Params;
@group(0) @binding(1) var<storage, read> qdu: array<f32>;
@group(0) @binding(2) var<storage, read> qqdata: array<f32>;
@group(0) @binding(3) var<storage, read_write> qdv: array<f32>;

@compute @workgroup_size(256)
fn vector2_poisson2d_apply_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let qb = i * 4u;
    let g00 = qqdata[qb + 0u];
    let g01 = qqdata[qb + 1u];
    let g10 = qqdata[qb + 2u];
    let g11 = qqdata[qb + 3u];
    let du0a = qdu[qb + 0u];
    let du1a = qdu[qb + 1u];
    let du0b = qdu[qb + 2u];
    let du1b = qdu[qb + 3u];
    qdv[qb + 0u] = g00 * du0a + g01 * du1a;
    qdv[qb + 1u] = g10 * du0a + g11 * du1a;
    qdv[qb + 2u] = g00 * du0b + g01 * du1b;
    qdv[qb + 3u] = g10 * du0b + g11 * du1b;
}

@compute @workgroup_size(256)
fn vector2_poisson2d_transpose_accumulate_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let qb = i * 4u;
    let g00 = qqdata[qb + 0u];
    let g01 = qqdata[qb + 1u];
    let g10 = qqdata[qb + 2u];
    let g11 = qqdata[qb + 3u];
    for (var c = 0u; c < 2u; c = c + 1u) {
        let base = c * 2u;
        let ddv0 = qdu[qb + base];
        let ddv1 = qdu[qb + base + 1u];
        qdv[qb + base] = qdv[qb + base] + g00 * ddv0 + g10 * ddv1;
        qdv[qb + base + 1u] = qdv[qb + base + 1u] + g01 * ddv0 + g11 * ddv1;
    }
}
"#;

/// [`reed_cpu::Vector3Poisson2DApply`] on `f32`: shared 2×2 `qdata` times three 2-gradients in `du[i*6..]`.
const QFUNCTION_VECTOR3_POISSON2D_APPLY_WGSL: &str = r#"
struct QfV3P2Params {
    num_q: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> qp: QfV3P2Params;
@group(0) @binding(1) var<storage, read> qdu: array<f32>;
@group(0) @binding(2) var<storage, read> qqdata: array<f32>;
@group(0) @binding(3) var<storage, read_write> qdv: array<f32>;

@compute @workgroup_size(256)
fn vector3_poisson2d_apply_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let qb = i * 4u;
    let g00 = qqdata[qb + 0u];
    let g01 = qqdata[qb + 1u];
    let g10 = qqdata[qb + 2u];
    let g11 = qqdata[qb + 3u];
    let b6 = i * 6u;
    for (var c = 0u; c < 3u; c = c + 1u) {
        let base = c * 2u;
        let du0 = qdu[b6 + base];
        let du1 = qdu[b6 + base + 1u];
        qdv[b6 + base] = g00 * du0 + g01 * du1;
        qdv[b6 + base + 1u] = g10 * du0 + g11 * du1;
    }
}

@compute @workgroup_size(256)
fn vector3_poisson2d_transpose_accumulate_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let qb = i * 4u;
    let g00 = qqdata[qb + 0u];
    let g01 = qqdata[qb + 1u];
    let g10 = qqdata[qb + 2u];
    let g11 = qqdata[qb + 3u];
    let b6 = i * 6u;
    for (var c = 0u; c < 3u; c = c + 1u) {
        let base = c * 2u;
        let ddv0 = qdu[b6 + base];
        let ddv1 = qdu[b6 + base + 1u];
        qdv[b6 + base] = qdv[b6 + base] + g00 * ddv0 + g10 * ddv1;
        qdv[b6 + base + 1u] = qdv[b6 + base + 1u] + g01 * ddv0 + g11 * ddv1;
    }
}
"#;

/// [`reed_cpu::Identity`] on `f32`: `out[i] = in[i]` for `i in 0..n` (flat `q * num_comp` words).
const QFUNCTION_IDENTITY_COPY_WGSL: &str = r#"
struct QfUnaryWordParams {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> qp: QfUnaryWordParams;
@group(0) @binding(1) var<storage, read> qin: array<f32>;
@group(0) @binding(2) var<storage, read_write> qout: array<f32>;

@compute @workgroup_size(256)
fn qf_identity_copy_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.n) {
        return;
    }
    qout[i] = qin[i];
}

@compute @workgroup_size(256)
fn qf_identity_transpose_accumulate_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.n) {
        return;
    }
    qout[i] = qout[i] + qin[i];
}
"#;

/// [`reed_cpu::IdentityScalar`] on `f32`: forward `out[i] = in[i * ncomp]`; transpose `du[i * ncomp] += dv[i]`.
const QFUNCTION_IDENTITY_SCALAR_WGSL: &str = r#"
struct QfIdentityScalarParams {
    num_q: u32,
    ncomp: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> qp: QfIdentityScalarParams;
@group(0) @binding(1) var<storage, read> qin: array<f32>;
@group(0) @binding(2) var<storage, read_write> qout: array<f32>;

@compute @workgroup_size(256)
fn qf_identity_scalar_gather_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    qout[i] = qin[i * qp.ncomp];
}

@compute @workgroup_size(256)
fn qf_identity_scalar_transpose_accumulate_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.num_q) {
        return;
    }
    let base = i * qp.ncomp;
    qout[base] = qout[base] + qin[i];
}
"#;

/// [`reed_cpu::Scale`] on `f32`: `out[i] = alpha * in[i]`; `alpha` from uniform (host reads libCEED 8-byte `f64` context).
const QFUNCTION_SCALE_F32_WGSL: &str = r#"
struct QfScaleF32Params {
    n: u32,
    _pad0: u32,
    alpha: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> qp: QfScaleF32Params;
@group(0) @binding(1) var<storage, read> qin: array<f32>;
@group(0) @binding(2) var<storage, read_write> qout: array<f32>;

@compute @workgroup_size(256)
fn qf_scale_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.n) {
        return;
    }
    qout[i] = qp.alpha * qin[i];
}

@compute @workgroup_size(256)
fn qf_scale_transpose_accumulate_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= qp.n) {
        return;
    }
    qout[i] = qout[i] + qp.alpha * qin[i];
}
"#;

/// Transpose (scatter): `v[g] += u[l]` for offset or strided layout. Single-thread loop (workgroup size 1) so
/// we avoid `atomicCompareExchange` (not available on all Metal targets) while matching CPU `+=`.

/// Tile reference quadrature weights across elements: `v[e*nq + q] = weights[q]`.
const BASIS_WEIGHT_WGSL: &str = r#"
struct BasisWeightParams {
    num_qpoints: u32,
    out_size: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> bw_weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> bw_v: array<f32>;
@group(0) @binding(2) var<uniform> bw_p: BasisWeightParams;

@compute @workgroup_size(64)
fn basis_weight_tile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bw_p.out_size) {
        return;
    }
    let q = i % bw_p.num_qpoints;
    bw_v[i] = bw_weights[q];
}
"#;

/// Transpose (scatter): `v[g] += u[l]` for offset layout. Single-thread loop (workgroup size 1) so
/// we avoid `atomicCompareExchange` (not available on all Metal targets) while matching CPU `+=`.
const RESTRICTION_SCATTER_WGSL: &str = r#"
struct RestrictionParams {
    nelem: u32,
    elemsize: u32,
    ncomp: u32,
    compstride: u32,
    local_size: u32,
    global_size: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> rs_u: array<f32>;
@group(0) @binding(1) var<storage, read> rs_offsets: array<i32>;
@group(0) @binding(2) var<storage, read_write> rs_v: array<f32>;
@group(0) @binding(3) var<uniform> rs_p: RestrictionParams;

@compute @workgroup_size(1)
fn restriction_scatter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    for (var idx = 0u; idx < rs_p.local_size; idx = idx + 1u) {
        let per_elem = rs_p.ncomp * rs_p.elemsize;
        let elem = idx / per_elem;
        let rem = idx % per_elem;
        let comp = rem / rs_p.elemsize;
        let local = rem % rs_p.elemsize;

        let offset_idx = elem * rs_p.elemsize + local;
        let base = rs_offsets[offset_idx];
        if (base < 0) {
            continue;
        }
        let g = u32(base) + comp * rs_p.compstride;
        if (g >= rs_p.global_size) {
            continue;
        }

        let val = rs_u[idx];
        rs_v[g] = rs_v[g] + val;
    }
}
"#;

/// `f64` gather (`NoTranspose`) via `u32` pairs (IEEE-754 bits). Matches CPU `f64` without fp64 shader ops.
const RESTRICTION_OFFSET_GATHER_F64_BITS_WGSL: &str = r#"
struct RestrictionParams {
    nelem: u32,
    elemsize: u32,
    ncomp: u32,
    compstride: u32,
    local_size: u32,
    global_size: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> rgf_u: array<u32>;
@group(0) @binding(1) var<storage, read> rgf_offsets: array<i32>;
@group(0) @binding(2) var<storage, read_write> rgf_v: array<u32>;
@group(0) @binding(3) var<uniform> rgf_p: RestrictionParams;

@compute @workgroup_size(64)
fn restriction_gather_f64_bits_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= rgf_p.local_size) {
        return;
    }

    let per_elem = rgf_p.ncomp * rgf_p.elemsize;
    let elem = idx / per_elem;
    let rem = idx % per_elem;
    let comp = rem / rgf_p.elemsize;
    let local = rem % rgf_p.elemsize;

    let offset_idx = elem * rgf_p.elemsize + local;
    let base = rgf_offsets[offset_idx];
    let idx2 = idx * 2u;
    if (base < 0) {
        rgf_v[idx2] = 0u;
        rgf_v[idx2 + 1u] = 0u;
        return;
    }
    let g = u32(base) + comp * rgf_p.compstride;
    if (g >= rgf_p.global_size) {
        rgf_v[idx2] = 0u;
        rgf_v[idx2 + 1u] = 0u;
        return;
    }
    let g2 = g * 2u;
    rgf_v[idx2] = rgf_u[g2];
    rgf_v[idx2 + 1u] = rgf_u[g2 + 1u];
}
"#;

const RESTRICTION_STRIDED_GATHER_F64_BITS_WGSL: &str = r#"
struct StridedRestrictionParams {
    nelem: u32,
    elemsize: u32,
    ncomp: u32,
    _pad0: u32,
    s0: i32,
    s1: i32,
    s2: i32,
    _pad1: u32,
    local_size: u32,
    global_size: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> stf_u: array<u32>;
@group(0) @binding(1) var<storage, read_write> stf_v: array<u32>;
@group(0) @binding(2) var<uniform> stf_p: StridedRestrictionParams;

@compute @workgroup_size(64)
fn restriction_strided_gather_f64_bits_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= stf_p.local_size) {
        return;
    }
    let per_elem = stf_p.ncomp * stf_p.elemsize;
    let elem = idx / per_elem;
    let rem = idx % per_elem;
    let comp = rem / stf_p.elemsize;
    let local = rem % stf_p.elemsize;

    let g = i32(local) * stf_p.s0 + i32(comp) * stf_p.s1 + i32(elem) * stf_p.s2;
    let idx2 = idx * 2u;
    if (g < 0) {
        stf_v[idx2] = 0u;
        stf_v[idx2 + 1u] = 0u;
        return;
    }
    let gu = u32(g);
    if (gu >= stf_p.global_size) {
        stf_v[idx2] = 0u;
        stf_v[idx2 + 1u] = 0u;
        return;
    }
    let g2 = gu * 2u;
    stf_v[idx2] = stf_u[g2];
    stf_v[idx2 + 1u] = stf_u[g2 + 1u];
}
"#;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod qfunction_context_sync_tests {
    use super::GpuRuntime;
    use reed_core::QFunctionContext;

    fn gpu_runtime_or_skip() -> Option<GpuRuntime> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))?;
        GpuRuntime::new(&adapter)
    }

    #[test]
    fn sync_qfunction_context_writes_and_clears_dirty() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let mut ctx = QFunctionContext::new(8);
        ctx.write_f64_le(0, std::f64::consts::PI).unwrap();
        assert!(ctx.host_needs_device_upload());

        let gpu_buf = rt.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-test-qfn-ctx"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        rt.sync_qfunction_context_to_buffer(&gpu_buf, 0, &ctx)
            .unwrap();
        assert!(!ctx.host_needs_device_upload());

        let readback = rt.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-test-qfn-ctx-readback"),
            size: 64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = rt
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reed-test-qfn-ctx-copy"),
            });
        encoder.copy_buffer_to_buffer(&gpu_buf, 0, &readback, 0, 8);
        rt.queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..8);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        rt.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let got = f64::from_le_bytes(data[..8].try_into().unwrap());
        drop(data);
        readback.unmap();
        assert!((got - std::f64::consts::PI).abs() < 1e-14);

        rt.sync_qfunction_context_to_buffer(&gpu_buf, 0, &ctx)
            .unwrap();
    }

    #[test]
    fn write_qfunction_context_force_uploads_even_when_clean() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let mut ctx = QFunctionContext::new(4);
        ctx.write_i32_le(0, 0x01020304).unwrap();
        assert!(ctx.host_needs_device_upload());
        let gpu_buf = rt.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-test-qfn-ctx-2"),
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        rt.sync_qfunction_context_to_buffer(&gpu_buf, 0, &ctx)
            .unwrap();
        assert!(!ctx.host_needs_device_upload());

        ctx.write_i32_le(0, 0x11223344).unwrap();
        assert!(ctx.host_needs_device_upload());
        ctx.mark_host_synced_to_device();
        assert!(!ctx.host_needs_device_upload());

        rt.write_qfunction_context_to_buffer(&gpu_buf, 0, &ctx)
            .unwrap();
        assert!(!ctx.host_needs_device_upload());
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod mass_apply_qp_tests {
    use super::{map_readback_f32_result, GpuRuntime};
    use wgpu::util::DeviceExt;

    fn gpu_runtime_or_skip() -> Option<GpuRuntime> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))?;
        GpuRuntime::new(&adapter)
    }

    #[test]
    fn mass_apply_qp_forward_matches_reference() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let n: u32 = 127;
        let u_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let q_host: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.03).collect();
        let byte_len = (n as usize) * 4;

        let u_buf = rt
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reed-test-ma-u"),
                contents: bytemuck::cast_slice(&u_host),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let q_buf = rt
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reed-test-ma-q"),
                contents: bytemuck::cast_slice(&q_host),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let v_buf = rt.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-test-ma-v"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        rt.dispatch_mass_apply_qp_f32(&u_buf, &q_buf, &v_buf, n)
            .unwrap();

        let readback = rt.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-test-ma-readback"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = rt
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reed-test-ma-copy"),
            });
        encoder.copy_buffer_to_buffer(&v_buf, 0, &readback, 0, byte_len as u64);
        rt.queue.submit(Some(encoder.finish()));
        rt.device.poll(wgpu::Maintain::Wait);

        let mut got = vec![0.0_f32; n as usize];
        map_readback_f32_result(&rt.device, &readback, &mut got).unwrap();
        for i in 0..(n as usize) {
            let exp = u_host[i] * q_host[i];
            assert!(
                (got[i] - exp).abs() < 1.0e-4,
                "i={i} got {} exp {}",
                got[i],
                exp
            );
        }
    }

    #[test]
    fn mass_apply_qp_transpose_accumulates_like_cpu_gallery() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let n: u32 = 64;
        let dv_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let q_host: Vec<f32> = (0..n).map(|i| 2.0 - (i as f32) * 0.02).collect();
        let byte_len = (n as usize) * 4;

        let dv_buf = rt
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reed-test-mat-dv"),
                contents: bytemuck::cast_slice(&dv_host),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let q_buf = rt
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reed-test-mat-q"),
                contents: bytemuck::cast_slice(&q_host),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let du_seed: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25).collect();
        let du_buf = rt
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reed-test-mat-du"),
                contents: bytemuck::cast_slice(&du_seed),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        rt.dispatch_mass_apply_qp_transpose_accumulate_f32(&dv_buf, &q_buf, &du_buf, n)
            .unwrap();

        let readback = rt.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reed-test-mat-readback"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = rt
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reed-test-mat-copy"),
            });
        encoder.copy_buffer_to_buffer(&du_buf, 0, &readback, 0, byte_len as u64);
        rt.queue.submit(Some(encoder.finish()));
        rt.device.poll(wgpu::Maintain::Wait);

        let mut got = vec![0.0_f32; n as usize];
        map_readback_f32_result(&rt.device, &readback, &mut got).unwrap();
        for i in 0..(n as usize) {
            let exp = du_seed[i] + dv_host[i] * q_host[i];
            assert!(
                (got[i] - exp).abs() < 1.0e-3,
                "i={i} got {} exp {}",
                got[i],
                exp
            );
        }
    }

    #[test]
    fn mass_apply_qp_f32_host_matches_reference() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let n = 301usize;
        let u: Vec<f32> = (0..n).map(|i| (i as f32) * 0.013).collect();
        let q: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.007).collect();
        let mut v = vec![0.0_f32; n];
        rt.mass_apply_qp_f32_host(&u, &q, &mut v).unwrap();
        for i in 0..n {
            let exp = u[i] * q[i];
            assert!((v[i] - exp).abs() < 2.0e-3, "i={i}");
        }
    }

    #[test]
    fn mass_apply_qp_transpose_f32_host_matches_reference() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        let n = 88usize;
        let dv: Vec<f32> = (0..n).map(|i| (i as f32) * 0.11 + 0.3).collect();
        let qd: Vec<f32> = (0..n).map(|i| 1.1 - (i as f32) * 0.004).collect();
        let mut du: Vec<f32> = (0..n).map(|i| (i as f32) * 0.07).collect();
        let du_before = du.clone();
        rt.mass_apply_qp_transpose_accumulate_f32_host(&dv, &qd, &mut du)
            .unwrap();
        for i in 0..n {
            let exp = du_before[i] + dv[i] * qd[i];
            assert!((du[i] - exp).abs() < 2.0e-3, "i={i}");
        }
    }

    #[test]
    fn mass_apply_qp_transpose_broadcast_matches_cpu_vector2_mass() {
        let Some(rt) = gpu_runtime_or_skip() else {
            return;
        };
        use reed_core::QFunctionTrait;

        let num_qp = 41usize;
        let components = 2usize;
        let dv: Vec<f32> = (0..num_qp * components)
            .map(|i| (i as f32) * 0.04 - 0.3)
            .collect();
        let qdata: Vec<f32> = (0..num_qp).map(|i| 0.65 + (i as f32) * 0.015).collect();
        let du_seed: Vec<f32> = (0..num_qp * components)
            .map(|i| (i as f32) * 0.09)
            .collect();

        let mut du_gpu = du_seed.clone();
        rt.mass_apply_qp_transpose_broadcast_scalar_qdata_f32_host(
            &dv,
            &qdata,
            &mut du_gpu,
            components,
        )
        .unwrap();

        let mut du_cpu = du_seed.clone();
        let mut q_mut = qdata.clone();
        let gallery = reed_cpu::Vector2MassApply::new();
        gallery
            .apply_operator_transpose(&[], num_qp, &[&dv], &mut [&mut du_cpu[..], &mut q_mut[..]])
            .unwrap();

        for i in 0..du_gpu.len() {
            assert!(
                (du_gpu[i] - du_cpu[i]).abs() < 2.0e-3,
                "i={i} gpu={} cpu={}",
                du_gpu[i],
                du_cpu[i]
            );
        }
    }
}
