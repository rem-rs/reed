//! Core integer aliases for libCEED-style interop.
//!
//! `CeedInt` is the canonical index/integer type exposed by Reed APIs that mirror libCEED
//! indexing surfaces. Internally, sizes still use `CeedSize` (`usize`) for Rust allocation and
//! slice indexing.

/// libCEED-style integer/index type used by public Reed APIs.
pub type CeedInt = i32;

/// Canonical size type used for allocation lengths and Rust indexing.
pub type CeedSize = usize;
