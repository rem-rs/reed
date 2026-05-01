//! User data passed into [`crate::qfunction::QFunctionTrait::apply`], analogous to
//! libCEED's `CeedQFunctionContext` (opaque bytes sized per qfunction).
//!
//! Reed supports:
//! * **Opaque buffers** with little-endian typed helpers (libCEED-style offsets).
//! * **Optional field layout** (name, offset, kind) for validation and named access,
//!   mirroring `CeedQFunctionContextRegister*` metadata.
//! * **Host/device dirty tracking** for backends that mirror context to GPU: any host
//!   write sets “needs upload”; backends clear after upload. Pure CPU execution ignores this.

use crate::error::{ReedError, ReedResult};
use std::sync::atomic::{AtomicU8, Ordering};

const SYNC_CLEAN: u8 = 0;
const SYNC_HOST_DIRTY: u8 = 1;

/// Scalar / integer kinds for registered context fields (little-endian on host).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QFunctionContextFieldKind {
    F64,
    F32,
    I32,
}

impl QFunctionContextFieldKind {
    #[inline]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F64 => 8,
            Self::F32 | Self::I32 => 4,
        }
    }
}

/// One registered field (libCEED `CeedQFunctionContextRegister*` parity at metadata level).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QFunctionContextField {
    pub name: String,
    pub offset: usize,
    pub kind: QFunctionContextFieldKind,
}

fn validate_field_layout(fields: &[QFunctionContextField], byte_len: usize) -> ReedResult<()> {
    if fields.is_empty() {
        return Err(ReedError::QFunction(
            "QFunctionContext field layout: empty field list".into(),
        ));
    }
    let mut seen = std::collections::HashSet::new();
    let mut intervals: Vec<(usize, usize)> = Vec::with_capacity(fields.len());
    for f in fields {
        if !seen.insert(f.name.as_str()) {
            return Err(ReedError::QFunction(format!(
                "QFunctionContext field layout: duplicate field name {:?}",
                f.name
            )));
        }
        let sz = f.kind.size_bytes();
        let end = f.offset.checked_add(sz).ok_or_else(|| {
            ReedError::QFunction(format!(
                "QFunctionContext field {:?}: offset+size overflow",
                f.name
            ))
        })?;
        if end > byte_len {
            return Err(ReedError::QFunction(format!(
                "QFunctionContext field {:?}: offset {} + size {} exceeds buffer length {}",
                f.name, f.offset, sz, byte_len
            )));
        }
        intervals.push((f.offset, end));
    }
    intervals.sort_by_key(|x| x.0);
    for w in intervals.windows(2) {
        if w[0].1 > w[1].0 {
            return Err(ReedError::QFunction(
                "QFunctionContext field layout: overlapping field byte ranges".into(),
            ));
        }
    }
    Ok(())
}

/// Byte buffer holding per-operator user state for qfunctions (coefficients, flags, etc.).
#[derive(Debug)]
pub struct QFunctionContext {
    data: Vec<u8>,
    /// When `Some`, describes registered fields and enables [`Self::read_field_f64`] etc.
    fields: Option<Vec<QFunctionContextField>>,
    /// `SYNC_CLEAN` or `SYNC_HOST_DIRTY` — host bytes changed since last device upload.
    host_dirty_for_device: AtomicU8,
}

impl Clone for QFunctionContext {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            fields: self.fields.clone(),
            host_dirty_for_device: AtomicU8::new(
                self.host_dirty_for_device.load(Ordering::Relaxed),
            ),
        }
    }
}

impl QFunctionContext {
    /// Allocate a zero-filled buffer of `byte_len` bytes (no registered field layout).
    pub fn new(byte_len: usize) -> Self {
        Self {
            data: vec![0u8; byte_len],
            fields: None,
            host_dirty_for_device: AtomicU8::new(SYNC_CLEAN),
        }
    }

    /// Layout with named fields; buffer length is the maximum end offset of all fields.
    pub fn from_field_layout(fields: Vec<QFunctionContextField>) -> ReedResult<Self> {
        let byte_len = fields
            .iter()
            .map(|f| f.offset + f.kind.size_bytes())
            .max()
            .unwrap_or(0);
        validate_field_layout(&fields, byte_len)?;
        Ok(Self {
            data: vec![0u8; byte_len],
            fields: Some(fields),
            host_dirty_for_device: AtomicU8::new(SYNC_CLEAN),
        })
    }

    /// Fixed-size buffer with an explicit layout (byte length must cover every field).
    pub fn from_field_layout_with_len(
        byte_len: usize,
        fields: Vec<QFunctionContextField>,
    ) -> ReedResult<Self> {
        validate_field_layout(&fields, byte_len)?;
        Ok(Self {
            data: vec![0u8; byte_len],
            fields: Some(fields),
            host_dirty_for_device: AtomicU8::new(SYNC_CLEAN),
        })
    }

    /// Take ownership of raw bytes (e.g. from deserialization). Marks host data dirty for device backends.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self {
            data,
            fields: None,
            host_dirty_for_device: AtomicU8::new(SYNC_HOST_DIRTY),
        }
    }

    /// Registered fields, if this context was built with [`Self::from_field_layout`].
    #[inline]
    pub fn registered_fields(&self) -> Option<&[QFunctionContextField]> {
        self.fields.as_deref()
    }

    /// libCEED gallery `Scale` / `Scale (scalar)` context: one `f64` named `"alpha"` at offset 0.
    pub fn gallery_scale_fields() -> Vec<QFunctionContextField> {
        vec![QFunctionContextField {
            name: "alpha".into(),
            offset: 0,
            kind: QFunctionContextFieldKind::F64,
        }]
    }

    /// True if a GPU backend should re-upload [`Self::as_bytes`] before launching kernels.
    #[inline]
    pub fn host_needs_device_upload(&self) -> bool {
        self.host_dirty_for_device.load(Ordering::Relaxed) == SYNC_HOST_DIRTY
    }

    /// Mark every host byte as potentially stale on device (e.g. after reading context back from GPU).
    #[inline]
    pub fn mark_device_dirty(&self) {
        self.host_dirty_for_device
            .store(SYNC_HOST_DIRTY, Ordering::Relaxed);
    }

    /// Call after uploading host bytes to device memory (WGSL uniform/storage).
    #[inline]
    pub fn mark_host_synced_to_device(&self) {
        self.host_dirty_for_device
            .store(SYNC_CLEAN, Ordering::Relaxed);
    }

    /// Call after mutating the buffer without going through typed `write_*` helpers.
    #[inline]
    pub fn note_host_modified(&self) {
        self.host_dirty_for_device
            .store(SYNC_HOST_DIRTY, Ordering::Relaxed);
    }

    #[inline]
    fn mark_dirty_on_write(&self) {
        self.note_host_modified();
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    #[inline]
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        self.mark_dirty_on_write();
        &mut self.data
    }

    /// Read `f64` from any byte slice (e.g. `ctx` in [`crate::qfunction::QFunctionTrait::apply`]).
    pub fn read_f64_le_bytes(data: &[u8], offset: usize) -> ReedResult<f64> {
        if offset + 8 > data.len() {
            return Err(ReedError::QFunction(format!(
                "read_f64_le_bytes: offset {} + 8 exceeds buffer length {}",
                offset,
                data.len()
            )));
        }
        let b: [u8; 8] = data[offset..offset + 8]
            .try_into()
            .map_err(|_| ReedError::QFunction("read_f64_le_bytes: slice length".into()))?;
        Ok(f64::from_le_bytes(b))
    }

    /// Read `f64` at `offset` (little-endian).
    pub fn read_f64_le(&self, offset: usize) -> ReedResult<f64> {
        Self::read_f64_le_bytes(&self.data, offset)
    }

    /// Write `f64` into any mutable byte slice.
    pub fn write_f64_le_bytes(buf: &mut [u8], offset: usize, v: f64) -> ReedResult<()> {
        if offset + 8 > buf.len() {
            return Err(ReedError::QFunction(format!(
                "write_f64_le_bytes: offset {} + 8 exceeds buffer length {}",
                offset,
                buf.len()
            )));
        }
        buf[offset..offset + 8].copy_from_slice(&v.to_le_bytes());
        Ok(())
    }

    /// Write `f64` at `offset` (little-endian).
    pub fn write_f64_le(&mut self, offset: usize, v: f64) -> ReedResult<()> {
        self.mark_dirty_on_write();
        Self::write_f64_le_bytes(&mut self.data, offset, v)
    }

    /// Read `f32` from any byte slice.
    pub fn read_f32_le_bytes(data: &[u8], offset: usize) -> ReedResult<f32> {
        if offset + 4 > data.len() {
            return Err(ReedError::QFunction(format!(
                "read_f32_le_bytes: offset {} + 4 exceeds buffer length {}",
                offset,
                data.len()
            )));
        }
        let b: [u8; 4] = data[offset..offset + 4]
            .try_into()
            .map_err(|_| ReedError::QFunction("read_f32_le_bytes: slice length".into()))?;
        Ok(f32::from_le_bytes(b))
    }

    /// Read `f32` at `offset` (little-endian).
    pub fn read_f32_le(&self, offset: usize) -> ReedResult<f32> {
        Self::read_f32_le_bytes(&self.data, offset)
    }

    /// Write `f32` into any mutable byte slice.
    pub fn write_f32_le_bytes(buf: &mut [u8], offset: usize, v: f32) -> ReedResult<()> {
        if offset + 4 > buf.len() {
            return Err(ReedError::QFunction(format!(
                "write_f32_le_bytes: offset {} + 4 exceeds buffer length {}",
                offset,
                buf.len()
            )));
        }
        buf[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
        Ok(())
    }

    /// Write `f32` at `offset` (little-endian).
    pub fn write_f32_le(&mut self, offset: usize, v: f32) -> ReedResult<()> {
        self.mark_dirty_on_write();
        Self::write_f32_le_bytes(&mut self.data, offset, v)
    }

    /// Read `i32` from any byte slice.
    pub fn read_i32_le_bytes(data: &[u8], offset: usize) -> ReedResult<i32> {
        if offset + 4 > data.len() {
            return Err(ReedError::QFunction(format!(
                "read_i32_le_bytes: offset {} + 4 exceeds buffer length {}",
                offset,
                data.len()
            )));
        }
        let b: [u8; 4] = data[offset..offset + 4]
            .try_into()
            .map_err(|_| ReedError::QFunction("read_i32_le_bytes: slice length".into()))?;
        Ok(i32::from_le_bytes(b))
    }

    /// Read `i32` at `offset` (little-endian).
    pub fn read_i32_le(&self, offset: usize) -> ReedResult<i32> {
        Self::read_i32_le_bytes(&self.data, offset)
    }

    /// Write `i32` into any mutable byte slice.
    pub fn write_i32_le_bytes(buf: &mut [u8], offset: usize, v: i32) -> ReedResult<()> {
        if offset + 4 > buf.len() {
            return Err(ReedError::QFunction(format!(
                "write_i32_le_bytes: offset {} + 4 exceeds buffer length {}",
                offset,
                buf.len()
            )));
        }
        buf[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
        Ok(())
    }

    /// Write `i32` at `offset` (little-endian).
    pub fn write_i32_le(&mut self, offset: usize, v: i32) -> ReedResult<()> {
        self.mark_dirty_on_write();
        Self::write_i32_le_bytes(&mut self.data, offset, v)
    }

    /// Read a registered `f64` field by name (requires [`Self::from_field_layout`]).
    pub fn read_field_f64(&self, name: &str) -> ReedResult<f64> {
        let f = self.field_by_name(name)?;
        if f.kind != QFunctionContextFieldKind::F64 {
            return Err(ReedError::QFunction(format!(
                "read_field_f64: field {:?} is {:?}, not F64",
                name, f.kind
            )));
        }
        Self::read_f64_le_bytes(&self.data, f.offset)
    }

    /// Read a registered `f32` field by name.
    pub fn read_field_f32(&self, name: &str) -> ReedResult<f32> {
        let f = self.field_by_name(name)?;
        if f.kind != QFunctionContextFieldKind::F32 {
            return Err(ReedError::QFunction(format!(
                "read_field_f32: field {:?} is {:?}, not F32",
                name, f.kind
            )));
        }
        Self::read_f32_le_bytes(&self.data, f.offset)
    }

    /// Read a registered `i32` field by name.
    pub fn read_field_i32(&self, name: &str) -> ReedResult<i32> {
        let f = self.field_by_name(name)?;
        if f.kind != QFunctionContextFieldKind::I32 {
            return Err(ReedError::QFunction(format!(
                "read_field_i32: field {:?} is {:?}, not I32",
                name, f.kind
            )));
        }
        Self::read_i32_le_bytes(&self.data, f.offset)
    }

    pub fn write_field_f64(&mut self, name: &str, v: f64) -> ReedResult<()> {
        let offset = {
            let f = self.field_by_name(name)?;
            if f.kind != QFunctionContextFieldKind::F64 {
                return Err(ReedError::QFunction(format!(
                    "write_field_f64: field {:?} is {:?}, not F64",
                    name, f.kind
                )));
            }
            f.offset
        };
        self.write_f64_le(offset, v)
    }

    pub fn write_field_f32(&mut self, name: &str, v: f32) -> ReedResult<()> {
        let offset = {
            let f = self.field_by_name(name)?;
            if f.kind != QFunctionContextFieldKind::F32 {
                return Err(ReedError::QFunction(format!(
                    "write_field_f32: field {:?} is {:?}, not F32",
                    name, f.kind
                )));
            }
            f.offset
        };
        self.write_f32_le(offset, v)
    }

    pub fn write_field_i32(&mut self, name: &str, v: i32) -> ReedResult<()> {
        let offset = {
            let f = self.field_by_name(name)?;
            if f.kind != QFunctionContextFieldKind::I32 {
                return Err(ReedError::QFunction(format!(
                    "write_field_i32: field {:?} is {:?}, not I32",
                    name, f.kind
                )));
            }
            f.offset
        };
        self.write_i32_le(offset, v)
    }

    fn field_by_name(&self, name: &str) -> ReedResult<&QFunctionContextField> {
        let fields = self.fields.as_ref().ok_or_else(|| {
            ReedError::QFunction(
                "named field access requires QFunctionContext::from_field_layout".into(),
            )
        })?;
        fields.iter().find(|f| f.name == name).ok_or_else(|| {
            ReedError::QFunction(format!("unknown QFunctionContext field {:?}", name))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_roundtrip() {
        let mut c = QFunctionContext::new(16);
        assert!(!c.host_needs_device_upload());
        c.write_f64_le(0, 3.25).unwrap();
        c.write_f64_le(8, -1.0).unwrap();
        assert!((c.read_f64_le(0).unwrap() - 3.25).abs() < 1e-15);
        assert!((c.read_f64_le(8).unwrap() + 1.0).abs() < 1e-15);
        assert!(c.host_needs_device_upload());
        c.mark_host_synced_to_device();
        assert!(!c.host_needs_device_upload());
    }

    #[test]
    fn f32_i32_roundtrip() {
        let mut c = QFunctionContext::new(12);
        c.write_f32_le(0, 1.25).unwrap();
        c.write_i32_le(4, -7).unwrap();
        c.write_i32_le(8, 0x1234_5678_u32 as i32).unwrap();
        assert!((c.read_f32_le(0).unwrap() - 1.25).abs() < 1e-6);
        assert_eq!(c.read_i32_le(4).unwrap(), -7);
        assert_eq!(c.read_i32_le(8).unwrap(), 0x1234_5678_u32 as i32);
    }

    #[test]
    fn read_write_bytes_helpers_match() {
        let mut raw = [0u8; 16];
        QFunctionContext::write_f64_le_bytes(&mut raw, 0, -1.5).unwrap();
        QFunctionContext::write_f32_le_bytes(&mut raw, 8, 2.0).unwrap();
        assert!((QFunctionContext::read_f64_le_bytes(&raw, 0).unwrap() + 1.5).abs() < 1e-15);
        assert!((QFunctionContext::read_f32_le_bytes(&raw, 8).unwrap() - 2.0).abs() < 1e-6);
        QFunctionContext::write_i32_le_bytes(&mut raw, 12, 99).unwrap();
        assert_eq!(QFunctionContext::read_i32_le_bytes(&raw, 12).unwrap(), 99);
    }

    #[test]
    fn field_layout_named_access() {
        let mut c = QFunctionContext::from_field_layout(vec![
            QFunctionContextField {
                name: "alpha".into(),
                offset: 0,
                kind: QFunctionContextFieldKind::F64,
            },
            QFunctionContextField {
                name: "flag".into(),
                offset: 8,
                kind: QFunctionContextFieldKind::I32,
            },
        ])
        .unwrap();
        c.write_field_f64("alpha", 2.5).unwrap();
        c.write_field_i32("flag", 42).unwrap();
        assert!((c.read_field_f64("alpha").unwrap() - 2.5).abs() < 1e-15);
        assert_eq!(c.read_field_i32("flag").unwrap(), 42);
        assert!(c.host_needs_device_upload());
        c.mark_host_synced_to_device();
        assert!(!c.host_needs_device_upload());
    }

    #[test]
    fn field_layout_rejects_overlap() {
        let err = QFunctionContext::from_field_layout(vec![
            QFunctionContextField {
                name: "a".into(),
                offset: 0,
                kind: QFunctionContextFieldKind::F64,
            },
            QFunctionContextField {
                name: "b".into(),
                offset: 4,
                kind: QFunctionContextFieldKind::F64,
            },
        ])
        .unwrap_err();
        assert!(matches!(err, ReedError::QFunction(_)));
    }

    #[test]
    fn from_bytes_marks_dirty() {
        let c = QFunctionContext::from_bytes(vec![1, 2, 3]);
        assert!(c.host_needs_device_upload());
    }
}
