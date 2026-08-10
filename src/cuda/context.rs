//! The CUDA context and the safe wrappers around every cuSOLVER call.
//!
//! This is the only file in the crate that contains `unsafe`. Each FFI call is
//! wrapped in a method that takes typed, checked arguments and returns
//! [`LinAlgResult`], so the solvers in [`crate::linalg::sparse`] read as ordinary
//! safe Rust.
//!
//! # What `unsafe` is doing here, and why it cannot be removed
//!
//! cudarc's `cusolver::safe` module wraps handle creation only — `sp_create`,
//! `sp_set_stream`. There is no safe wrapper for any `csrlsv*` or `csrchol*`
//! routine, and cudarc ships no `cusparse/safe.rs` at all (`sys` and `result`
//! only). Calling a C library is inherently unsafe; the most that can be done is
//! to confine it, give every call a checked wrapper, and let RAII types own the
//! handles so no raw pointer outlives its owner. That is what this module does.

use std::ffi::c_int;
use std::sync::{Arc, OnceLock};

use cudarc::cusolver::{SpHandle, sys as solver_sys};
use cudarc::cusparse::sys as sparse_sys;
use cudarc::driver::{CudaContext as DriverContext, CudaStream};

use crate::cuda::buffers::{CsrStructure, DeviceSystem, to_err};
use crate::cuda::ffi::{self, csrcholInfo_t};
use crate::error::ErrorLogging;
use crate::linalg::{LinAlgError, LinAlgResult};

/// Default tolerance for cuSOLVER's zero-pivot test.
pub(crate) const DEFAULT_SINGULARITY_TOL: f64 = 1e-12;

/// Fill-reducing reordering applied before factorization.
///
/// Reordering trades analysis time for fill-in, and therefore for factorization
/// time. Which one wins is structure-dependent: nested dissection dominates on
/// the grid- and sphere-like graphs 3D pose graphs produce, minimum degree is
/// competitive on chain-like ones.
///
/// Re-exported as `apex_solver::linalg::Reordering`; it lives here because the
/// variants map one-to-one onto cuSOLVER entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reordering {
    /// No reordering. Cheapest analysis, worst fill-in.
    None,
    /// Reverse Cuthill-McKee.
    SymRcm,
    /// Approximate minimum degree.
    SymAmd,
    /// Nested dissection (METIS). Best fill-in for the block-structured Hessians
    /// that pose graphs and bundle adjustment produce, which is why it is the
    /// default.
    #[default]
    Metis,
}

impl Reordering {
    /// The `reorder` argument expected by `cusolverSp*csrlsv*`.
    pub(crate) fn to_arg(self) -> c_int {
        match self {
            Reordering::None => 0,
            Reordering::SymRcm => 1,
            Reordering::SymAmd => 2,
            Reordering::Metis => 3,
        }
    }
}

/// Is a usable CUDA device present?
///
/// Returns `false` rather than erroring when the driver is missing, so callers
/// (and tests) can degrade gracefully on machines without a GPU.
///
/// # Why this catches a panic
///
/// cudarc resolves `libcuda` lazily and **panics** — it does not return `Err` —
/// when the shared library cannot be `dlopen`ed. A plain
/// `CudaContext::new(0).is_ok()` therefore aborts the process on any machine
/// without an NVIDIA driver, which is exactly the case this probe exists to
/// detect. The result is memoized so the (noisy) load attempt happens once.
pub fn is_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        // The load failure panics with a long message about library search
        // paths. Silence it for the duration of the probe — a missing GPU is a
        // supported configuration here, not an error worth printing.
        //
        // Note the panic hook is process-global. This runs exactly once, before
        // any GPU work, so the window in which another thread's panic message
        // could be suppressed is negligible.
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let available = std::panic::catch_unwind(|| DriverContext::new(0).is_ok()).unwrap_or(false);
        std::panic::set_hook(previous_hook);
        available
    })
}

/// Owns the `CUSPARSE_MATRIX_TYPE_GENERAL` / `CUSPARSE_INDEX_BASE_ZERO`
/// descriptor that every `csr*` routine takes.
///
/// RAII: created once, destroyed on drop, never handed out as a raw pointer
/// except to an FFI call in this module.
struct MatDescr {
    raw: sparse_sys::cusparseMatDescr_t,
}

impl MatDescr {
    fn new() -> LinAlgResult<Self> {
        let mut raw: sparse_sys::cusparseMatDescr_t = std::ptr::null_mut();

        // SAFETY: `raw` is a valid out-pointer to a null-initialized handle.
        check_sparse(unsafe { sparse_sys::cusparseCreateMatDescr(&mut raw) }, "cusparseCreateMatDescr")?;
        let descr = Self { raw };

        // SAFETY: `raw` was just created successfully and is non-null; `descr`
        // now owns it, so an error below still releases it on drop.
        check_sparse(
            unsafe {
                sparse_sys::cusparseSetMatType(
                    descr.raw,
                    sparse_sys::cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_GENERAL,
                )
            },
            "cusparseSetMatType",
        )?;

        // SAFETY: as above.
        check_sparse(
            unsafe {
                sparse_sys::cusparseSetMatIndexBase(
                    descr.raw,
                    sparse_sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                )
            },
            "cusparseSetMatIndexBase",
        )?;

        Ok(descr)
    }

    /// The descriptor in the type cuSOLVER expects.
    ///
    /// cudarc generates `cusparseMatDescr_t` twice — once in `cusparse::sys` and
    /// once in `cusolver::sys` — as two distinct opaque structs. Both are `*mut`
    /// handles to the *same* underlying C object; only the cuSPARSE side declares
    /// create/set/destroy, and only the cuSOLVER side is accepted by `csr*`. The
    /// cast bridges the two bindgen views.
    fn as_solver(&self) -> solver_sys::cusparseMatDescr_t {
        self.raw.cast()
    }
}

impl Drop for MatDescr {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `raw` came from `cusparseCreateMatDescr` in `new` and is
            // non-null here; this is its only destruction site, and `Drop` runs
            // exactly once.
            unsafe { sparse_sys::cusparseDestroyMatDescr(self.raw) };
        }
    }
}

/// Owns cuSOLVER's symbolic analysis and numeric factor for one pattern.
///
/// RAII over `csrcholInfo_t`: `cusolverSpDestroyCsrcholInfo` runs exactly once,
/// on drop, so an early return anywhere in the analysis cannot leak it.
pub(crate) struct CholeskyInfo {
    raw: csrcholInfo_t,
}

impl CholeskyInfo {
    fn new() -> LinAlgResult<Self> {
        let api = ffi::api()?;
        let mut raw: csrcholInfo_t = std::ptr::null_mut();
        // SAFETY: out-pointer to a null-initialized handle; ownership moves to
        // the returned value, whose `Drop` is the sole destruction site.
        check_status(unsafe { (api.create_info)(&mut raw) }, "cusolverSpCreateCsrcholInfo")?;
        Ok(Self { raw })
    }
}

impl Drop for CholeskyInfo {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        if let Ok(api) = ffi::api() {
            // SAFETY: `raw` came from `create_info` and is destroyed exactly
            // once, here.
            unsafe { (api.destroy_info)(self.raw) };
        }
        self.raw = std::ptr::null_mut();
    }
}

impl std::fmt::Debug for CholeskyInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CholeskyInfo").finish_non_exhaustive()
    }
}

/// Bytes cuSOLVER needs for one factorization, as reported by `BufferInfo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CholeskyWorkspace {
    /// cuSOLVER's own internal data — allocated by cuSOLVER, reported for
    /// accounting only.
    pub(crate) internal_bytes: usize,
    /// Scratch the caller must allocate and pass to `factor`/`solve`.
    pub(crate) workspace_bytes: usize,
}

/// The CUDA device, cuSOLVER sparse handle, and matrix descriptor.
///
/// Created once per solver and held for its lifetime — handle creation involves
/// device initialization and must not happen per iteration.
///
/// Deliberately **not** `Send`/`Sync`. Earlier revisions asserted both with
/// `unsafe impl`; nothing requires them, since `LinearSolver` carries no such
/// bound and a solver is only ever used through `&mut dyn LinearSolver<_>`. An
/// unchecked auto-trait promise is the most dangerous kind of `unsafe`, so it is
/// simply absent.
pub struct CudaContext {
    /// Kept alive for the lifetime of the handle: the stream borrows from it.
    _device: Arc<DriverContext>,
    handle: SpHandle,
    descr: MatDescr,
}

impl CudaContext {
    /// Initialize CUDA device `ordinal` and create the solver handle.
    ///
    /// Returns [`LinAlgError::InvalidState`] rather than panicking when no CUDA
    /// driver is present — see [`is_available`] for why that guard is needed.
    pub fn new(ordinal: usize) -> LinAlgResult<Self> {
        if !is_available() {
            return Err(LinAlgError::InvalidState(
                "no usable CUDA device found: the NVIDIA driver library could not be loaded. \
                 Build without the `cuda` feature to use the CPU solvers."
                    .to_string(),
            )
            .log());
        }

        let device = DriverContext::new(ordinal).map_err(|e| {
            LinAlgError::InvalidState(format!(
                "CUDA device {ordinal} unavailable (is an NVIDIA driver installed?)"
            ))
            .log_with_source(e)
        })?;

        let handle = SpHandle::new(device.default_stream()).map_err(|e| {
            LinAlgError::InvalidState("failed to create cuSOLVER sparse handle".to_string())
                .log_with_source(e)
        })?;

        Ok(Self {
            _device: device,
            handle,
            descr: MatDescr::new()?,
        })
    }

    /// The stream the cuSOLVER handle is bound to.
    ///
    /// Every device buffer handed to a solve must be allocated on and copied
    /// through *this* stream, otherwise the upload is not ordered before the
    /// solve.
    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        self.handle.stream()
    }

    /// Block until every operation queued on the stream has completed.
    pub(crate) fn synchronize(&self) -> LinAlgResult<()> {
        self.stream().synchronize().map_err(to_err("stream synchronize"))
    }

    fn handle(&self) -> solver_sys::cusolverSpHandle_t {
        self.handle.cu()
    }

    // ------------------------------------------------------------------
    // One-shot path: analyze + factor + solve in a single call
    // ------------------------------------------------------------------

    /// `cusolverSpDcsrlsvchol` — reorder, analyze, factorize and solve.
    ///
    /// Redoes the symbolic work on every call; see [`crate::cuda`] for why the
    /// reusable path exists.
    pub(crate) fn csrlsvchol(
        &self,
        system: &mut DeviceSystem,
        n: c_int,
        nnz: c_int,
        reorder: Reordering,
    ) -> LinAlgResult<()> {
        let mut singularity: c_int = -1;
        let status = system.with_ptrs(self.stream(), n, nnz, |p| {
            // SAFETY: every matrix/vector pointer is a device allocation owned by
            // `system` and kept alive by the stream-ordering guards held across
            // this call; `row_ptr` has n+1 entries and `col_idx`/`vals` have nnz.
            // `singularity` is the host out-parameter this API expects.
            unsafe {
                solver_sys::cusolverSpDcsrlsvchol(
                    self.handle(),
                    p.n,
                    p.nnz,
                    self.descr.as_solver(),
                    p.vals,
                    p.row_ptr,
                    p.col_idx,
                    p.b,
                    DEFAULT_SINGULARITY_TOL,
                    reorder.to_arg(),
                    p.x,
                    &mut singularity,
                )
            }
        });
        check_status(status, "cusolverSpDcsrlsvchol")?;
        check_singularity(singularity, "cusolverSpDcsrlsvchol")
    }

    /// `cusolverSpDcsrlsvqr` — the same, via QR.
    pub(crate) fn csrlsvqr(
        &self,
        system: &mut DeviceSystem,
        n: c_int,
        nnz: c_int,
        reorder: Reordering,
    ) -> LinAlgResult<()> {
        let mut singularity: c_int = -1;
        let status = system.with_ptrs(self.stream(), n, nnz, |p| {
            // SAFETY: identical contract to `csrlsvchol` above.
            unsafe {
                solver_sys::cusolverSpDcsrlsvqr(
                    self.handle(),
                    p.n,
                    p.nnz,
                    self.descr.as_solver(),
                    p.vals,
                    p.row_ptr,
                    p.col_idx,
                    p.b,
                    DEFAULT_SINGULARITY_TOL,
                    reorder.to_arg(),
                    p.x,
                    &mut singularity,
                )
            }
        });
        check_status(status, "cusolverSpDcsrlsvqr")?;
        check_singularity(singularity, "cusolverSpDcsrlsvqr")
    }

    // ------------------------------------------------------------------
    // Host-side reordering (declared by cudarc, from cusolverSp.h)
    // ------------------------------------------------------------------

    /// Compute a fill-reducing permutation `p`, where `p[i] = j` means old row
    /// `j` becomes new row `i`.
    ///
    /// Returns the identity for [`Reordering::None`] without calling cuSOLVER.
    pub(crate) fn reorder(
        &self,
        csr: &CsrStructure,
        kind: Reordering,
    ) -> LinAlgResult<Vec<c_int>> {
        let (n, nnz) = (csr.n_i32()?, csr.nnz_i32()?);
        let mut permutation: Vec<c_int> = (0..n).collect();
        if kind == Reordering::None {
            return Ok(permutation);
        }

        // SAFETY: `row_ptr` has n+1 entries and `col_idx` has nnz, both read-only
        // here; `permutation` has n entries and is the documented out-parameter.
        // The METIS options pointer is null, selecting cuSOLVER's defaults.
        let (status, what) = unsafe {
            match kind {
                Reordering::SymRcm => (
                    solver_sys::cusolverSpXcsrsymrcmHost(
                        self.handle(),
                        n,
                        nnz,
                        self.descr.as_solver(),
                        csr.row_ptr().as_ptr(),
                        csr.col_idx().as_ptr(),
                        permutation.as_mut_ptr(),
                    ),
                    "cusolverSpXcsrsymrcmHost",
                ),
                Reordering::SymAmd => (
                    solver_sys::cusolverSpXcsrsymamdHost(
                        self.handle(),
                        n,
                        nnz,
                        self.descr.as_solver(),
                        csr.row_ptr().as_ptr(),
                        csr.col_idx().as_ptr(),
                        permutation.as_mut_ptr(),
                    ),
                    "cusolverSpXcsrsymamdHost",
                ),
                // `Reordering::None` returned above.
                _ => (
                    solver_sys::cusolverSpXcsrmetisndHost(
                        self.handle(),
                        n,
                        nnz,
                        self.descr.as_solver(),
                        csr.row_ptr().as_ptr(),
                        csr.col_idx().as_ptr(),
                        std::ptr::null(),
                        permutation.as_mut_ptr(),
                    ),
                    "cusolverSpXcsrmetisndHost",
                ),
            }
        };
        check_status(status, what)?;
        Ok(permutation)
    }

    /// Apply a symmetric permutation to `csr` **in place**, returning the value
    /// map: `permuted_values[i] = original_values[map[i]]`.
    ///
    /// The map is what lets later iterations re-apply the same permutation to new
    /// values without redoing any symbolic work.
    pub(crate) fn permute(
        &self,
        csr: &mut CsrStructure,
        permutation: &[c_int],
    ) -> LinAlgResult<Vec<c_int>> {
        let (n, nnz) = (csr.n_i32()?, csr.nnz_i32()?);
        let mut map: Vec<c_int> = (0..nnz).collect();

        let mut buffer_bytes: usize = 0;
        {
            let (row_ptr, col_idx) = csr.pattern_mut();
            // SAFETY: `p` and `q` are the same symmetric permutation of length n,
            // as required for a symmetric reordering; the pattern arrays are
            // correctly sized and read-only in this query.
            let status = unsafe {
                solver_sys::cusolverSpXcsrperm_bufferSizeHost(
                    self.handle(),
                    n,
                    n,
                    nnz,
                    self.descr.as_solver(),
                    row_ptr.as_ptr(),
                    col_idx.as_ptr(),
                    permutation.as_ptr(),
                    permutation.as_ptr(),
                    &mut buffer_bytes,
                )
            };
            check_status(status, "cusolverSpXcsrperm_bufferSizeHost")?;
        }

        let mut buffer = vec![0u8; buffer_bytes.max(1)];
        let (row_ptr, col_idx) = csr.pattern_mut();
        // SAFETY: `row_ptr`/`col_idx` are mutable and correctly sized — this call
        // rewrites the pattern in place; `map` has nnz entries; `buffer` is at
        // least `buffer_bytes` long.
        let status = unsafe {
            solver_sys::cusolverSpXcsrpermHost(
                self.handle(),
                n,
                n,
                nnz,
                self.descr.as_solver(),
                row_ptr.as_mut_ptr(),
                col_idx.as_mut_ptr(),
                permutation.as_ptr(),
                permutation.as_ptr(),
                map.as_mut_ptr(),
                buffer.as_mut_ptr().cast(),
            )
        };
        check_status(status, "cusolverSpXcsrpermHost")?;
        Ok(map)
    }

    // ------------------------------------------------------------------
    // Reusable path: analyze once, factor and solve per iteration
    // ------------------------------------------------------------------

    /// `cusolverSpXcsrcholAnalysis` — symbolic analysis of the uploaded pattern.
    pub(crate) fn chol_analyze(
        &self,
        system: &mut DeviceSystem,
        n: c_int,
        nnz: c_int,
    ) -> LinAlgResult<CholeskyInfo> {
        let api = ffi::api()?;
        let info = CholeskyInfo::new()?;
        let status = system.with_ptrs(self.stream(), n, nnz, |p| {
            // SAFETY: `row_ptr` and `col_idx` are device allocations of n+1 and
            // nnz `i32`s, kept alive by the guards for the duration of the call;
            // `info` was just created and is non-null.
            unsafe {
                (api.analysis)(
                    self.handle(),
                    p.n,
                    p.nnz,
                    self.descr.as_solver(),
                    p.row_ptr,
                    p.col_idx,
                    info.raw,
                )
            }
        });
        check_status(status, "cusolverSpXcsrcholAnalysis")?;
        Ok(info)
    }

    /// `cusolverSpDcsrcholBufferInfo` — how much scratch the factorization needs.
    ///
    /// Depends on the analysis, so it must follow [`chol_analyze`](Self::chol_analyze).
    pub(crate) fn chol_buffer_size(
        &self,
        system: &mut DeviceSystem,
        info: &CholeskyInfo,
        n: c_int,
        nnz: c_int,
    ) -> LinAlgResult<CholeskyWorkspace> {
        let api = ffi::api()?;
        let (mut internal_bytes, mut workspace_bytes) = (0usize, 0usize);
        let status = system.with_ptrs(self.stream(), n, nnz, |p| {
            // SAFETY: device pointers as above, plus two host out-parameters.
            // Only the structure is read here, so the values may still be zero.
            unsafe {
                (api.buffer_info)(
                    self.handle(),
                    p.n,
                    p.nnz,
                    self.descr.as_solver(),
                    p.vals,
                    p.row_ptr,
                    p.col_idx,
                    info.raw,
                    &mut internal_bytes,
                    &mut workspace_bytes,
                )
            }
        });
        check_status(status, "cusolverSpDcsrcholBufferInfo")?;
        Ok(CholeskyWorkspace {
            internal_bytes,
            workspace_bytes,
        })
    }

    /// `cusolverSpDcsrcholFactor` — the numeric factorization.
    pub(crate) fn chol_factor(
        &self,
        system: &mut DeviceSystem,
        info: &CholeskyInfo,
        n: c_int,
        nnz: c_int,
    ) -> LinAlgResult<()> {
        let api = ffi::api()?;
        let status = system.with_ptrs(self.stream(), n, nnz, |p| {
            // SAFETY: all four pointers are device allocations sized nnz, n+1,
            // nnz and the workspace `chol_buffer_size` reported, held alive by
            // the guards. `info` carries the analysis of exactly this pattern.
            unsafe {
                (api.factor)(
                    self.handle(),
                    p.n,
                    p.nnz,
                    self.descr.as_solver(),
                    p.vals,
                    p.row_ptr,
                    p.col_idx,
                    info.raw,
                    p.workspace,
                )
            }
        });
        check_status(status, "cusolverSpDcsrcholFactor")
    }

    /// `cusolverSpDcsrcholZeroPivot` — the permuted row of the first pivot below
    /// `tol`, or `-1` when there is none.
    pub(crate) fn chol_zero_pivot(&self, info: &CholeskyInfo, tol: f64) -> LinAlgResult<c_int> {
        let api = ffi::api()?;
        let mut position: c_int = -1;
        // SAFETY: `info` holds the factorization just computed; `position` is a
        // host out-parameter.
        let status = unsafe { (api.zero_pivot)(self.handle(), info.raw, tol, &mut position) };
        check_status(status, "cusolverSpDcsrcholZeroPivot")?;
        Ok(position)
    }

    /// `cusolverSpDcsrcholSolve` — the forward and back substitutions.
    pub(crate) fn chol_solve(
        &self,
        system: &mut DeviceSystem,
        info: &CholeskyInfo,
        n: c_int,
        nnz: c_int,
    ) -> LinAlgResult<()> {
        let api = ffi::api()?;
        let status = system.with_ptrs(self.stream(), n, nnz, |p| {
            // SAFETY: `b` and `x` are device buffers of exactly n doubles and the
            // workspace is the one sized by `chol_buffer_size`; all are kept alive
            // by the guards across the call.
            unsafe { (api.solve)(self.handle(), p.n, p.b, p.x, info.raw, p.workspace) }
        });
        check_status(status, "cusolverSpDcsrcholSolve")
    }
}

impl std::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaContext").finish_non_exhaustive()
    }
}

/// Translate a `cusolverStatus_t` into a `LinAlgError`.
pub(crate) fn check_status(status: solver_sys::cusolverStatus_t, what: &str) -> LinAlgResult<()> {
    use solver_sys::cusolverStatus_t as S;
    match status {
        S::CUSOLVER_STATUS_SUCCESS => Ok(()),
        S::CUSOLVER_STATUS_ALLOC_FAILED => Err(LinAlgError::InvalidState(format!(
            "{what}: GPU allocation failed (out of device memory)"
        ))
        .log()),
        S::CUSOLVER_STATUS_ZERO_PIVOT => {
            Err(LinAlgError::SingularMatrix(format!("{what}: zero pivot encountered")).log())
        }
        S::CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED => Err(LinAlgError::InvalidInput(format!(
            "{what}: matrix type not supported by cuSOLVER"
        ))
        .log()),
        other => Err(LinAlgError::FactorizationFailed(format!(
            "{what}: cuSOLVER status {other:?}"
        ))
        .log()),
    }
}

/// Translate a `cusparseStatus_t` into a `LinAlgError`.
fn check_sparse(status: sparse_sys::cusparseStatus_t, what: &str) -> LinAlgResult<()> {
    if status == sparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        return Ok(());
    }
    Err(LinAlgError::InvalidState(format!("{what} failed: {status:?}")).log())
}

/// Turn cuSOLVER's `singularity` out-parameter into an error.
///
/// `csrlsvchol`/`csrlsvqr` return `-1` when the matrix is non-singular, or the
/// index of the first zero-pivot row otherwise.
pub(crate) fn check_singularity(singularity: c_int, what: &str) -> LinAlgResult<()> {
    if singularity >= 0 {
        return Err(LinAlgError::SingularMatrix(format!(
            "{what}: matrix is singular or not positive definite at row {singularity}. \
             For a pose graph this usually means unconstrained gauge freedom — add a \
             PriorFactor to anchor it."
        ))
        .log());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `is_available()` must never panic — it is the guard every GPU test uses,
    /// and it runs on CI machines with no GPU.
    #[test]
    fn is_available_does_not_panic_without_a_device() {
        let _ = is_available();
    }

    #[test]
    fn reordering_maps_to_cusolver_arguments() {
        assert_eq!(Reordering::None.to_arg(), 0);
        assert_eq!(Reordering::SymRcm.to_arg(), 1);
        assert_eq!(Reordering::SymAmd.to_arg(), 2);
        assert_eq!(Reordering::Metis.to_arg(), 3);
        assert_eq!(Reordering::default(), Reordering::Metis);
    }

    #[test]
    fn negative_singularity_is_success() {
        assert!(check_singularity(-1, "test").is_ok());
    }

    #[test]
    fn non_negative_singularity_reports_the_row() {
        let err = match check_singularity(42, "test") {
            Ok(()) => panic!("singularity 42 must be reported as an error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("42"), "error should name the row: {err}");
        assert!(
            err.contains("PriorFactor"),
            "error should be actionable: {err}"
        );
    }
}
