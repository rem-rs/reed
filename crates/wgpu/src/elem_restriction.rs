use std::{any::TypeId, sync::Arc};

use bytemuck::{Pod, Zeroable};
use num_traits::NumCast;
use reed_core::{
    enums::TransposeMode, error::ReedResult, scalar::Scalar, ElemRestrictionTrait, ReedError,
};
use reed_cpu::elem_restriction::CpuElemRestriction;
use wgpu::util::DeviceExt;

use crate::runtime::GpuRuntime;

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

/// Must match `StridedRestrictionParams` in `runtime.rs` WGSL (`KERNELS_WGSL`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct StridedRestrictionParamsGpu {
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
}

pub struct WgpuElemRestriction<T: Scalar> {
    nelem: usize,
    elemsize: usize,
    ncomp: usize,
    lsize: usize,
    layout: RestrictionLayout,
    runtime: Option<Arc<GpuRuntime>>,
    cpu_fallback: CpuElemRestriction<T>,
}

impl<T: Scalar> WgpuElemRestriction<T> {
    pub fn new_offset(
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        offsets: &[i32],
        runtime: Option<Arc<GpuRuntime>>,
    ) -> ReedResult<Self> {
        let cpu_fallback = CpuElemRestriction::<T>::new_offset(
            nelem, elemsize, ncomp, compstride, lsize, offsets,
        )?;
        Ok(Self {
            nelem,
            elemsize,
            ncomp,
            lsize,
            layout: RestrictionLayout::Offset {
                offsets: offsets.to_vec(),
                compstride,
            },
            runtime,
            cpu_fallback,
        })
    }

    pub fn new_strided(
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
        runtime: Option<Arc<GpuRuntime>>,
    ) -> ReedResult<Self> {
        let cpu_fallback =
            CpuElemRestriction::<T>::new_strided(nelem, elemsize, ncomp, lsize, strides)?;
        Ok(Self {
            nelem,
            elemsize,
            ncomp,
            lsize,
            layout: RestrictionLayout::Strided { strides },
            runtime,
            cpu_fallback,
        })
    }

    fn supports_f32_gpu() -> bool {
        TypeId::of::<T>() == TypeId::of::<f32>()
    }

    fn as_f64_slice(data: &[T]) -> Option<&[f64]> {
        if TypeId::of::<T>() != TypeId::of::<f64>() {
            return None;
        }
        // SAFETY: T is f64
        Some(unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), data.len()) })
    }

    fn as_f64_slice_mut(data: &mut [T]) -> Option<&mut [f64]> {
        if TypeId::of::<T>() != TypeId::of::<f64>() {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), data.len()) })
    }

    fn local_size(&self) -> usize {
        self.nelem * self.elemsize * self.ncomp
    }

    fn try_apply_no_transpose_gpu_f64(
        runtime: &Arc<GpuRuntime>,
        layout: &RestrictionLayout,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        u: &[f64],
        v: &mut [f64],
    ) -> ReedResult<bool> {
        if u.len() != lsize {
            return Err(ReedError::ElemRestriction(format!(
                "input length {} != global size {}",
                u.len(),
                lsize
            )));
        }
        let local_size = nelem * elemsize * ncomp;
        if v.len() != local_size {
            return Err(ReedError::ElemRestriction(format!(
                "output length {} != local size {}",
                v.len(),
                local_size
            )));
        }

        match layout {
            RestrictionLayout::Offset {
                offsets,
                compstride,
            } => Self::try_apply_no_transpose_gpu_f64_offset(
                runtime,
                nelem,
                elemsize,
                ncomp,
                lsize,
                local_size,
                u,
                v,
                offsets,
                *compstride,
            ),
            RestrictionLayout::Strided { strides } => Self::try_apply_no_transpose_gpu_f64_strided(
                runtime, nelem, elemsize, ncomp, lsize, local_size, u, v, *strides,
            ),
        }
    }

    fn try_apply_no_transpose_gpu_f64_offset(
        runtime: &Arc<GpuRuntime>,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        local_size: usize,
        u: &[f64],
        v: &mut [f64],
        offsets: &[i32],
        compstride: usize,
    ) -> ReedResult<bool> {
        let mut v_tmp = vec![0.0_f64; local_size];

        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-f64-u"),
                contents: bytemuck::cast_slice(u),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let offsets_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-f64-off"),
                contents: bytemuck::cast_slice(offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-f64-v"),
                contents: bytemuck::cast_slice(&v_tmp),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params: [u32; 8] = [
            nelem as u32,
            elemsize as u32,
            ncomp as u32,
            compstride as u32,
            local_size as u32,
            lsize as u32,
            0,
            0,
        ];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-f64-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-restriction-f64-bind"),
                layout: runtime.restriction_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buffer.as_entire_binding(),
                    },
                ],
            });

        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-restriction-f64-readback"),
            size: (local_size * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-restriction-f64-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-restriction-f64-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.restriction_gather_f64_pipeline());
            pass.set_bind_group(0, &bind, &[]);
            let groups = (local_size as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &v_buffer,
            0,
            &readback,
            0,
            (local_size * std::mem::size_of::<f64>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        map_readback_f64(&runtime.device, &readback, &mut v_tmp)?;
        v.copy_from_slice(&v_tmp);
        Ok(true)
    }

    fn try_apply_no_transpose_gpu_f64_strided(
        runtime: &Arc<GpuRuntime>,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        local_size: usize,
        u: &[f64],
        v: &mut [f64],
        strides: [i32; 3],
    ) -> ReedResult<bool> {
        let mut v_tmp = vec![0.0_f64; local_size];

        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-f64-s-u"),
                contents: bytemuck::cast_slice(u),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-f64-s-v"),
                contents: bytemuck::cast_slice(&v_tmp),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params = StridedRestrictionParamsGpu {
            nelem: nelem as u32,
            elemsize: elemsize as u32,
            ncomp: ncomp as u32,
            _pad0: 0,
            s0: strides[0],
            s1: strides[1],
            s2: strides[2],
            _pad1: 0,
            local_size: local_size as u32,
            global_size: lsize as u32,
            _pad2: 0,
            _pad3: 0,
        };
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-f64-s-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-restriction-f64-s-bind"),
                layout: runtime.restriction_strided_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: v_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p_buffer.as_entire_binding(),
                    },
                ],
            });

        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-restriction-f64-s-readback"),
            size: (local_size * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-restriction-f64-s-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-restriction-f64-s-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.restriction_strided_gather_f64_pipeline());
            pass.set_bind_group(0, &bind, &[]);
            let groups = (local_size as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &v_buffer,
            0,
            &readback,
            0,
            (local_size * std::mem::size_of::<f64>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        map_readback_f64(&runtime.device, &readback, &mut v_tmp)?;
        v.copy_from_slice(&v_tmp);
        Ok(true)
    }

    fn try_apply_no_transpose_gpu(&self, u: &[T], v: &mut [T]) -> ReedResult<bool> {
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let Some(u_f64) = Self::as_f64_slice(u) else {
                return Ok(false);
            };
            let Some(v_f64) = Self::as_f64_slice_mut(v) else {
                return Ok(false);
            };
            return Self::try_apply_no_transpose_gpu_f64(
                runtime,
                &self.layout,
                self.nelem,
                self.elemsize,
                self.ncomp,
                self.lsize,
                u_f64,
                v_f64,
            );
        }
        if !Self::supports_f32_gpu() {
            return Ok(false);
        }

        if let RestrictionLayout::Strided { strides } = &self.layout {
            return self.try_apply_no_transpose_gpu_strided(u, v, *strides);
        }

        let (offsets, compstride) = match &self.layout {
            RestrictionLayout::Offset {
                offsets,
                compstride,
            } => (offsets, *compstride),
            RestrictionLayout::Strided { .. } => unreachable!(),
        };

        if u.len() != self.lsize {
            return Err(ReedError::ElemRestriction(format!(
                "input length {} != global size {}",
                u.len(),
                self.lsize
            )));
        }

        let local_size = self.local_size();
        if v.len() != local_size {
            return Err(ReedError::ElemRestriction(format!(
                "output length {} != local size {}",
                v.len(),
                local_size
            )));
        }

        let Some(u_f32) = u
            .iter()
            .map(|x| NumCast::from(*x))
            .collect::<Option<Vec<f32>>>()
        else {
            return Ok(false);
        };
        let mut v_f32 = vec![0.0_f32; local_size];

        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let offsets_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-offsets"),
                contents: bytemuck::cast_slice(offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-v"),
                contents: bytemuck::cast_slice(&v_f32),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params: [u32; 8] = [
            self.nelem as u32,
            self.elemsize as u32,
            self.ncomp as u32,
            compstride as u32,
            local_size as u32,
            self.lsize as u32,
            0,
            0,
        ];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-restriction-bind"),
                layout: runtime.restriction_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buffer.as_entire_binding(),
                    },
                ],
            });

        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-restriction-readback"),
            size: (local_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-restriction-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-restriction-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.restriction_pipeline());
            pass.set_bind_group(0, &bind, &[]);
            let groups = (local_size as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &v_buffer,
            0,
            &readback,
            0,
            (local_size * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        map_readback_f32(&runtime.device, &readback, &mut v_f32)?;

        for (dst, src) in v.iter_mut().zip(v_f32.iter()) {
            *dst = NumCast::from(*src).ok_or_else(|| {
                ReedError::ElemRestriction("f32->T conversion failed during readback".into())
            })?;
        }
        Ok(true)
    }

    fn try_apply_no_transpose_gpu_strided(
        &self,
        u: &[T],
        v: &mut [T],
        strides: [i32; 3],
    ) -> ReedResult<bool> {
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        if !Self::supports_f32_gpu() {
            return Ok(false);
        }

        if u.len() != self.lsize {
            return Err(ReedError::ElemRestriction(format!(
                "input length {} != global size {}",
                u.len(),
                self.lsize
            )));
        }

        let local_size = self.local_size();
        if v.len() != local_size {
            return Err(ReedError::ElemRestriction(format!(
                "output length {} != local size {}",
                v.len(),
                local_size
            )));
        }

        let Some(u_f32) = u
            .iter()
            .map(|x| NumCast::from(*x))
            .collect::<Option<Vec<f32>>>()
        else {
            return Ok(false);
        };
        let mut v_f32 = vec![0.0_f32; local_size];

        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-strided-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-strided-v"),
                contents: bytemuck::cast_slice(&v_f32),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params = StridedRestrictionParamsGpu {
            nelem: self.nelem as u32,
            elemsize: self.elemsize as u32,
            ncomp: self.ncomp as u32,
            _pad0: 0,
            s0: strides[0],
            s1: strides[1],
            s2: strides[2],
            _pad1: 0,
            local_size: local_size as u32,
            global_size: self.lsize as u32,
            _pad2: 0,
            _pad3: 0,
        };
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-strided-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-restriction-strided-bind"),
                layout: runtime.restriction_strided_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: v_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p_buffer.as_entire_binding(),
                    },
                ],
            });

        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-restriction-strided-readback"),
            size: (local_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-restriction-strided-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-restriction-strided-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.restriction_strided_pipeline());
            pass.set_bind_group(0, &bind, &[]);
            let groups = (local_size as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &v_buffer,
            0,
            &readback,
            0,
            (local_size * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        map_readback_f32(&runtime.device, &readback, &mut v_f32)?;

        for (dst, src) in v.iter_mut().zip(v_f32.iter()) {
            *dst = NumCast::from(*src).ok_or_else(|| {
                ReedError::ElemRestriction("f32->T conversion failed during readback".into())
            })?;
        }
        Ok(true)
    }

    fn try_apply_transpose_gpu(&self, u: &[T], v: &mut [T]) -> ReedResult<bool> {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // `Transpose` needs `f64` add; WGSL has no portable `f64` ops — use CPU fallback.
            return Ok(false);
        }
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        if !Self::supports_f32_gpu() {
            return Ok(false);
        }

        if let RestrictionLayout::Strided { strides } = &self.layout {
            return self.try_apply_transpose_gpu_strided(u, v, *strides);
        }

        let (offsets, compstride) = match &self.layout {
            RestrictionLayout::Offset {
                offsets,
                compstride,
            } => (offsets, *compstride),
            RestrictionLayout::Strided { .. } => unreachable!(),
        };

        let local_size = self.local_size();
        if u.len() != local_size {
            return Err(ReedError::ElemRestriction(format!(
                "transpose input length {} != local size {}",
                u.len(),
                local_size
            )));
        }
        if v.len() != self.lsize {
            return Err(ReedError::ElemRestriction(format!(
                "transpose output length {} != global size {}",
                v.len(),
                self.lsize
            )));
        }

        let Some(u_f32) = u
            .iter()
            .map(|x| NumCast::from(*x))
            .collect::<Option<Vec<f32>>>()
        else {
            return Ok(false);
        };

        let mut v_f32_host: Vec<f32> = Vec::with_capacity(self.lsize);
        for x in v.iter() {
            let f: f32 = NumCast::from(*x).ok_or_else(|| {
                ReedError::ElemRestriction("transpose: expected f32-compatible values".into())
            })?;
            v_f32_host.push(f);
        }

        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-t-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let offsets_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-t-off"),
                contents: bytemuck::cast_slice(offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-t-v"),
                contents: bytemuck::cast_slice(&v_f32_host),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params: [u32; 8] = [
            self.nelem as u32,
            self.elemsize as u32,
            self.ncomp as u32,
            compstride as u32,
            local_size as u32,
            self.lsize as u32,
            0,
            0,
        ];
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-t-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-restriction-t-bind"),
                layout: runtime.restriction_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p_buffer.as_entire_binding(),
                    },
                ],
            });

        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-restriction-t-readback"),
            size: (self.lsize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-restriction-t-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-restriction-t-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.restriction_scatter_pipeline());
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &v_buffer,
            0,
            &readback,
            0,
            (self.lsize * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        let mut v_out = vec![0.0_f32; self.lsize];
        map_readback_f32(&runtime.device, &readback, &mut v_out)?;
        for (dst, src) in v.iter_mut().zip(v_out.iter()) {
            *dst = NumCast::from(*src).ok_or_else(|| {
                ReedError::ElemRestriction(
                    "f32->T conversion failed after transpose readback".into(),
                )
            })?;
        }
        Ok(true)
    }

    fn try_apply_transpose_gpu_strided(
        &self,
        u: &[T],
        v: &mut [T],
        strides: [i32; 3],
    ) -> ReedResult<bool> {
        let Some(runtime) = &self.runtime else {
            return Ok(false);
        };
        if !Self::supports_f32_gpu() {
            return Ok(false);
        }

        let local_size = self.local_size();
        if u.len() != local_size {
            return Err(ReedError::ElemRestriction(format!(
                "transpose input length {} != local size {}",
                u.len(),
                local_size
            )));
        }
        if v.len() != self.lsize {
            return Err(ReedError::ElemRestriction(format!(
                "transpose output length {} != global size {}",
                v.len(),
                self.lsize
            )));
        }

        let Some(u_f32) = u
            .iter()
            .map(|x| NumCast::from(*x))
            .collect::<Option<Vec<f32>>>()
        else {
            return Ok(false);
        };

        let mut v_f32_host: Vec<f32> = Vec::with_capacity(self.lsize);
        for x in v.iter() {
            let f: f32 = NumCast::from(*x).ok_or_else(|| {
                ReedError::ElemRestriction("transpose: expected f32-compatible values".into())
            })?;
            v_f32_host.push(f);
        }

        let u_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-strided-t-u"),
                contents: bytemuck::cast_slice(&u_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let v_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-strided-t-v"),
                contents: bytemuck::cast_slice(&v_f32_host),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params = StridedRestrictionParamsGpu {
            nelem: self.nelem as u32,
            elemsize: self.elemsize as u32,
            ncomp: self.ncomp as u32,
            _pad0: 0,
            s0: strides[0],
            s1: strides[1],
            s2: strides[2],
            _pad1: 0,
            local_size: local_size as u32,
            global_size: self.lsize as u32,
            _pad2: 0,
            _pad3: 0,
        };
        let p_buffer = runtime
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wgpu-restriction-strided-t-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wgpu-restriction-strided-t-bind"),
                layout: runtime.restriction_strided_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: v_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p_buffer.as_entire_binding(),
                    },
                ],
            });

        let readback = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-restriction-strided-t-readback"),
            size: (self.lsize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wgpu-restriction-strided-t-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-restriction-strided-t-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(runtime.restriction_strided_scatter_pipeline());
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &v_buffer,
            0,
            &readback,
            0,
            (self.lsize * std::mem::size_of::<f32>()) as u64,
        );
        runtime.queue.submit(Some(encoder.finish()));

        let mut v_out = vec![0.0_f32; self.lsize];
        map_readback_f32(&runtime.device, &readback, &mut v_out)?;
        for (dst, src) in v.iter_mut().zip(v_out.iter()) {
            *dst = NumCast::from(*src).ok_or_else(|| {
                ReedError::ElemRestriction(
                    "f32->T conversion failed after transpose readback".into(),
                )
            })?;
        }
        Ok(true)
    }
}

impl<T: Scalar> ElemRestrictionTrait<T> for WgpuElemRestriction<T> {
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
        if matches!(t_mode, TransposeMode::NoTranspose) && self.try_apply_no_transpose_gpu(u, v)? {
            return Ok(());
        }
        if matches!(t_mode, TransposeMode::Transpose) && self.try_apply_transpose_gpu(u, v)? {
            return Ok(());
        }
        self.cpu_fallback.apply(t_mode, u, v)
    }
}

fn map_readback_f64(
    device: &wgpu::Device,
    readback: &wgpu::Buffer,
    out: &mut [f64],
) -> ReedResult<()> {
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    let map_result = rx
        .recv()
        .map_err(|e| ReedError::ElemRestriction(format!("map recv error: {e}")))?;
    map_result.map_err(|e| ReedError::ElemRestriction(format!("map error: {e:?}")))?;

    let data = slice.get_mapped_range();
    if data.len() != out.len() * 8 {
        return Err(ReedError::ElemRestriction(
            "restriction f64 readback length mismatch".into(),
        ));
    }
    for (o, chunk) in out.iter_mut().zip(data.chunks_exact(8)) {
        let arr: [u8; 8] = chunk
            .try_into()
            .map_err(|_| ReedError::ElemRestriction("f64 readback chunk size".into()))?;
        *o = f64::from_le_bytes(arr);
    }
    drop(data);
    readback.unmap();
    Ok(())
}

fn map_readback_f32(
    device: &wgpu::Device,
    readback: &wgpu::Buffer,
    out: &mut [f32],
) -> ReedResult<()> {
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    let map_result = rx
        .recv()
        .map_err(|e| ReedError::ElemRestriction(format!("map recv error: {e}")))?;
    map_result.map_err(|e| ReedError::ElemRestriction(format!("map error: {e:?}")))?;

    let data = slice.get_mapped_range();
    let mapped: &[f32] = bytemuck::cast_slice(&data);
    if mapped.len() != out.len() {
        return Err(ReedError::ElemRestriction(
            "restriction readback length mismatch".into(),
        ));
    }
    out.copy_from_slice(mapped);
    drop(data);
    readback.unmap();
    Ok(())
}
