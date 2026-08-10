//! The seven cuSOLVER entry points cudarc does not declare.
//!
//! **Everything else goes through `cudarc::cusolver::sys`.** This module is not
//! a wrapper layer around cudarc; it exists only because cudarc's bindgen input
//! (`src/cusolver/sys/wrapper.h`) is exactly:
//!
//! ```text
//! cusolverDn.h   cusolverSp.h   cusolverMg.h   cusolverRf.h
//! ```
//!
//! `cusolverSp_LOWLEVEL_PREVIEW.h` is absent, and that header is where the
//! *reusable* sparse Cholesky lives — `cusolverSpCreateCsrcholInfo`,
//! `Xcsrcholanalysis`, `Dcsrcholbufferinfo`, `Dcsrcholfactor`,
//! `Dcsrcholzeropivot`, `Dcsrcholsolve`, `DestroyCsrcholInfo`. The symbols are
//! present in the shipped `libcusolver`; only the declarations are missing.
//!
//! The fill-reducing reorderings (`Xcsrsymrcm`, `Xcsrsymamd`, `Xcsrmetisnd`) and
//! the permutation helpers (`Xcsrperm`) live in `cusolverSp.h`, so cudarc *does*
//! declare them and [`super::context`] calls those through cudarc.
//!
//! # Why this is worth the seven declarations
//!
//! `cusolverSpDcsrlsvchol` — the one cudarc offers — redoes reordering and
//! symbolic analysis on every call, while `H = JᵀJ` keeps a fixed sparsity
//! pattern for the entire optimization, including across Levenberg-Marquardt's
//! `λ` retries, which only touch the diagonal. The CPU solver already exploits
//! this by caching its `SymbolicLlt`. Measured on M3500, dropping the repeated
//! analysis took the GPU solve from 1108 ms to 436 ms; without it the GPU loses
//! to the CPU on every dataset tested.
//!
//! # Why `dlopen` rather than `extern "C"`
//!
//! A plain `extern "C"` block would need `libcusolver` at link time, which would
//! break the property the rest of this module maintains: the crate builds on
//! machines with no CUDA installation and fails gracefully at runtime instead.
//! cudarc resolves its own symbols the same way. The symbols are looked up once
//! and memoized; a missing library or symbol becomes a [`LinAlgError`], never a
//! panic.
//!
//! # How this module goes away
//!
//! Add `cusolverSp_LOWLEVEL_PREVIEW.h` to cudarc's `wrapper.h` upstream. The
//! generated bindings would then cover all seven, and [`super::context`] could
//! switch to `cudarc::cusolver::sys` with no other change.

#![allow(non_camel_case_types)]

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use cudarc::cusolver::sys::{cusolverSpHandle_t, cusolverStatus_t, cusparseMatDescr_t};

use crate::error::ErrorLogging;
use crate::linalg::{LinAlgError, LinAlgResult};

/// Opaque cuSOLVER handle holding the symbolic analysis and the numeric factor.
///
/// Declared as an uninhabited type so the pointer cannot be dereferenced.
pub(crate) enum csrcholInfo {}

/// `csrcholInfo_t` from `cusolverSp_LOWLEVEL_PREVIEW.h`.
pub(crate) type csrcholInfo_t = *mut csrcholInfo;

type FnCreateInfo = unsafe extern "C" fn(*mut csrcholInfo_t) -> cusolverStatus_t;
type FnDestroyInfo = unsafe extern "C" fn(csrcholInfo_t) -> cusolverStatus_t;

type FnAnalysis = unsafe extern "C" fn(
    handle: cusolverSpHandle_t,
    n: c_int,
    nnz: c_int,
    descr: cusparseMatDescr_t,
    csr_row_ptr: *const c_int,
    csr_col_ind: *const c_int,
    info: csrcholInfo_t,
) -> cusolverStatus_t;

type FnBufferInfo = unsafe extern "C" fn(
    handle: cusolverSpHandle_t,
    n: c_int,
    nnz: c_int,
    descr: cusparseMatDescr_t,
    csr_val: *const f64,
    csr_row_ptr: *const c_int,
    csr_col_ind: *const c_int,
    info: csrcholInfo_t,
    internal_bytes: *mut usize,
    workspace_bytes: *mut usize,
) -> cusolverStatus_t;

type FnFactor = unsafe extern "C" fn(
    handle: cusolverSpHandle_t,
    n: c_int,
    nnz: c_int,
    descr: cusparseMatDescr_t,
    csr_val: *const f64,
    csr_row_ptr: *const c_int,
    csr_col_ind: *const c_int,
    info: csrcholInfo_t,
    buffer: *mut c_void,
) -> cusolverStatus_t;

type FnZeroPivot = unsafe extern "C" fn(
    handle: cusolverSpHandle_t,
    info: csrcholInfo_t,
    tol: f64,
    position: *mut c_int,
) -> cusolverStatus_t;

type FnSolve = unsafe extern "C" fn(
    handle: cusolverSpHandle_t,
    n: c_int,
    b: *const f64,
    x: *mut f64,
    info: csrcholInfo_t,
    buffer: *mut c_void,
) -> cusolverStatus_t;

/// The subset of `libcusolver` cudarc leaves undeclared, resolved once.
///
/// Deliberately minimal: everything cudarc *does* declare is called through
/// `cudarc::cusolver::sys` instead. Only the seven `cusolverSp_LOWLEVEL_PREVIEW.h`
/// entry points are duplicated here, because there is no other way to reach them.
pub(crate) struct LowLevelApi {
    pub(crate) create_info: FnCreateInfo,
    pub(crate) destroy_info: FnDestroyInfo,
    pub(crate) analysis: FnAnalysis,
    pub(crate) buffer_info: FnBufferInfo,
    pub(crate) factor: FnFactor,
    pub(crate) zero_pivot: FnZeroPivot,
    pub(crate) solve: FnSolve,
    /// Keeps the library mapped for as long as the function pointers are used.
    _library: libloading::Library,
}

// SAFETY: these are plain C function pointers into a library that stays mapped
// for the process lifetime, and the cuSOLVER routines behind them are documented
// thread-safe (the handle, not the library, carries the mutable state).
unsafe impl Send for LowLevelApi {}
unsafe impl Sync for LowLevelApi {}

/// Names to try, in order. The soname is `.so.12` for both CUDA 12 and CUDA 13.
#[cfg(target_os = "linux")]
const LIBRARY_CANDIDATES: &[&str] = &["libcusolver.so", "libcusolver.so.12", "libcusolver.so.11"];

#[cfg(target_os = "windows")]
const LIBRARY_CANDIDATES: &[&str] = &["cusolver64_12.dll", "cusolver64_11.dll", "cusolver.dll"];

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
const LIBRARY_CANDIDATES: &[&str] = &["libcusolver.dylib", "libcusolver.so"];

/// The resolved low-level API, or the reason it is unavailable.
///
/// Memoized: the `dlopen` and the seven `dlsym`s happen once per process.
pub(crate) fn api() -> LinAlgResult<&'static LowLevelApi> {
    static API: OnceLock<Result<LowLevelApi, String>> = OnceLock::new();
    match API.get_or_init(load) {
        Ok(api) => Ok(api),
        Err(reason) => Err(LinAlgError::InvalidState(format!(
            "cuSOLVER low-level sparse Cholesky unavailable: {reason}"
        ))
        .log()),
    }
}

fn load() -> Result<LowLevelApi, String> {
    // SAFETY: loading a shared library runs its initializers. libcusolver is a
    // vendor library that the process may already have mapped via cudarc; this
    // is the same operation cudarc performs internally.
    let library = LIBRARY_CANDIDATES
        .iter()
        .find_map(|&name| unsafe { libloading::Library::new(name) }.ok())
        .ok_or_else(|| {
            format!(
                "none of {LIBRARY_CANDIDATES:?} could be loaded; is the CUDA toolkit on the \
                 library search path?"
            )
        })?;

    // SAFETY: each symbol is bound to the signature transcribed from
    // `cusolverSp_LOWLEVEL_PREVIEW.h` for this CUDA version. A wrong signature
    // would be undefined behaviour, so these are checked against the installed
    // headers rather than reconstructed from memory.
    unsafe {
        let symbol = |name: &[u8]| -> Result<*const (), String> {
            library
                .get::<*const ()>(name)
                .map(|s| *s)
                .map_err(|e| format!("{}: {e}", String::from_utf8_lossy(name)))
        };
        // `get::<F>` where F is a function pointer type yields the address; going
        // through a raw pointer and transmuting keeps every lookup uniform.
        macro_rules! bind {
            ($name:literal, $ty:ty) => {{
                let address = symbol(concat!($name, "\0").as_bytes())?;
                std::mem::transmute::<*const (), $ty>(address)
            }};
        }

        Ok(LowLevelApi {
            create_info: bind!("cusolverSpCreateCsrcholInfo", FnCreateInfo),
            destroy_info: bind!("cusolverSpDestroyCsrcholInfo", FnDestroyInfo),
            analysis: bind!("cusolverSpXcsrcholAnalysis", FnAnalysis),
            buffer_info: bind!("cusolverSpDcsrcholBufferInfo", FnBufferInfo),
            factor: bind!("cusolverSpDcsrcholFactor", FnFactor),
            zero_pivot: bind!("cusolverSpDcsrcholZeroPivot", FnZeroPivot),
            solve: bind!("cusolverSpDcsrcholSolve", FnSolve),
            _library: library,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolution must never panic, and on a machine with CUDA installed every
    /// symbol must be present — a partial bind would mean the transcribed names
    /// have drifted from the shipped library.
    #[test]
    fn symbol_resolution_is_all_or_nothing() {
        match api() {
            Ok(_) => {}
            Err(e) => {
                let message = e.to_string();
                assert!(
                    message.contains("could not") || message.contains("could be loaded"),
                    "failure should name the missing library, not a partial bind: {message}"
                );
            }
        }
    }
}
