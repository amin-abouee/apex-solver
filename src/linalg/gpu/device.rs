//! CUDA context, cuSOLVER handle, and host↔device staging for the GPU solvers.
//!
//! This module owns everything that is expensive to create (CUDA context,
//! cuSOLVER sparse handle, cuSPARSE matrix descriptor) so it can be built once
//! per solver instance and reused across every optimizer iteration.

use std::ffi::c_int;
use std::sync::{Arc, OnceLock};

use cudarc::cusolver::{SpHandle, sys as solver_sys};
use cudarc::cusparse::sys as sparse_sys;
use cudarc::driver::CudaContext;
use faer::sparse::SparseColMat;

use crate::error::ErrorLogging;
use crate::linalg::{LinAlgError, LinAlgResult};

/// Fill-reducing reordering applied before factorization.
///
/// Reordering trades analysis time for fill-in (and therefore factorization
/// time). Because cudarc does not expose cuSOLVER's reusable low-level analysis
/// API, this cost is paid on *every* solve — see the module docs of
/// [`crate::linalg::gpu`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reordering {
    /// No reordering. Cheapest analysis, worst fill-in.
    None,
    /// Reverse Cuthill-McKee.
    SymRcm,
    /// Approximate minimum degree.
    SymAmd,
    /// Nested dissection (METIS). Best fill-in for the block-structured Hessians
    /// that bundle adjustment produces, which is why it is the default.
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
/// when the shared library cannot be `dlopen`ed (`cudarc/src/lib.rs:200`). A
/// plain `CudaContext::new(0).is_ok()` therefore aborts the process on any
/// machine without an NVIDIA driver, which is exactly the case this probe exists
/// to detect. The result is memoized so the (noisy) load attempt happens once.
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
        let available = std::panic::catch_unwind(|| CudaContext::new(0).is_ok()).unwrap_or(false);
        std::panic::set_hook(previous_hook);
        available
    })
}

/// Owns the CUDA context, cuSOLVER sparse handle, and cuSPARSE matrix descriptor.
///
/// Created once per solver and held for its lifetime — handle creation involves
/// device initialization and must not happen per iteration.
pub struct GpuContext {
    /// Kept alive for the lifetime of the handle: the stream borrows from it.
    _context: Arc<CudaContext>,
    handle: SpHandle,
    descr: sparse_sys::cusparseMatDescr_t,
}

// SAFETY: `SpHandle` is documented thread-safe by NVIDIA and is already
// `Send + Sync` in cudarc. `descr` is an opaque immutable descriptor after
// construction — it is only read by cuSOLVER during a solve, never mutated.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
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

        let context = CudaContext::new(ordinal).map_err(|e| {
            LinAlgError::InvalidState(format!(
                "CUDA device {ordinal} unavailable (is an NVIDIA driver installed?)"
            ))
            .log_with_source(e)
        })?;

        let handle = SpHandle::new(context.default_stream()).map_err(|e| {
            LinAlgError::InvalidState("failed to create cuSOLVER sparse handle".to_string())
                .log_with_source(e)
        })?;

        let descr = create_general_zero_based_descr()?;

        Ok(Self {
            _context: context,
            handle,
            descr,
        })
    }

    pub(crate) fn handle(&self) -> solver_sys::cusolverSpHandle_t {
        self.handle.cu()
    }

    /// The matrix descriptor, in the type cuSOLVER expects.
    ///
    /// cudarc generates `cusparseMatDescr_t` twice — once in `cusparse::sys` and
    /// once in `cusolver::sys` — as two distinct opaque structs. Both are
    /// `*mut` handles to the *same* underlying C object; only the cuSPARSE side
    /// declares the create/set/destroy functions, and only the cuSOLVER side is
    /// accepted by `csrlsv*`. The cast bridges the two bindgen views.
    pub(crate) fn descr(&self) -> solver_sys::cusparseMatDescr_t {
        self.descr.cast()
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        if !self.descr.is_null() {
            // SAFETY: `descr` was produced by `cusparseCreateMatDescr` in
            // `new()` and is non-null here; this is its only destruction site,
            // and `Drop` runs exactly once.
            unsafe {
                sparse_sys::cusparseDestroyMatDescr(self.descr);
            }
        }
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext").finish_non_exhaustive()
    }
}

/// Build the `CUSPARSE_MATRIX_TYPE_GENERAL` / `CUSPARSE_INDEX_BASE_ZERO`
/// descriptor that `csrlsvchol`/`csrlsvqr` expect.
fn create_general_zero_based_descr() -> LinAlgResult<sparse_sys::cusparseMatDescr_t> {
    let mut descr: sparse_sys::cusparseMatDescr_t = std::ptr::null_mut();

    // SAFETY: `descr` is a valid out-pointer to a null-initialized handle.
    let status = unsafe { sparse_sys::cusparseCreateMatDescr(&mut descr) };
    if status != sparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        return Err(LinAlgError::InvalidState(format!(
            "cusparseCreateMatDescr failed: {status:?}"
        ))
        .log());
    }

    // SAFETY: `descr` was just created successfully and is non-null.
    let status = unsafe {
        sparse_sys::cusparseSetMatType(
            descr,
            sparse_sys::cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_GENERAL,
        )
    };
    if status != sparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        // SAFETY: `descr` is live; released before propagating the error.
        unsafe { sparse_sys::cusparseDestroyMatDescr(descr) };
        return Err(
            LinAlgError::InvalidState(format!("cusparseSetMatType failed: {status:?}")).log(),
        );
    }

    // SAFETY: as above.
    let status = unsafe {
        sparse_sys::cusparseSetMatIndexBase(
            descr,
            sparse_sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
        )
    };
    if status != sparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        // SAFETY: `descr` is live; released before propagating the error.
        unsafe { sparse_sys::cusparseDestroyMatDescr(descr) };
        return Err(LinAlgError::InvalidState(format!(
            "cusparseSetMatIndexBase failed: {status:?}"
        ))
        .log());
    }

    Ok(descr)
}

/// The CSR structure of `H`, converted to `i32` once and reused.
///
/// `H = JᵀJ` has a constant sparsity pattern across optimizer iterations (the
/// CPU solvers rely on the same invariant to cache their symbolic
/// factorizations), so the `usize → i32` narrowing is done on the first solve
/// only and the arrays are reused thereafter.
///
/// Note `H` is **symmetric**, so faer's CSC storage is already a valid CSR
/// description of the same matrix — no transpose or repacking is needed.
#[derive(Debug, Default)]
pub(crate) struct CsrStructure {
    row_ptr: Vec<c_int>,
    col_idx: Vec<c_int>,
    n: usize,
    nnz: usize,
}

impl CsrStructure {
    /// Rebuild only when the pattern actually changed.
    pub(crate) fn sync(&mut self, hessian: &SparseColMat<usize, f64>) -> LinAlgResult<()> {
        let n = hessian.ncols();
        let symbolic = hessian.symbolic();
        let nnz = symbolic.row_idx().len();

        if self.n == n && self.nnz == nnz && !self.row_ptr.is_empty() {
            return Ok(());
        }

        let to_i32 = |v: usize, what: &str| -> LinAlgResult<c_int> {
            c_int::try_from(v).map_err(|e| {
                LinAlgError::InvalidInput(format!(
                    "{what} ({v}) exceeds i32::MAX; cuSOLVER's sparse API is 32-bit indexed"
                ))
                .log_with_source(e)
            })
        };

        let mut row_ptr = Vec::with_capacity(n + 1);
        for col in 0..=n {
            row_ptr.push(to_i32(symbolic.col_ptr()[col], "CSR row pointer")?);
        }

        let mut col_idx = Vec::with_capacity(nnz);
        for &idx in symbolic.row_idx() {
            col_idx.push(to_i32(idx, "CSR column index")?);
        }

        self.row_ptr = row_ptr;
        self.col_idx = col_idx;
        self.n = n;
        self.nnz = nnz;
        Ok(())
    }

    pub(crate) fn row_ptr(&self) -> &[c_int] {
        &self.row_ptr
    }

    pub(crate) fn col_idx(&self) -> &[c_int] {
        &self.col_idx
    }

    pub(crate) fn n(&self) -> usize {
        self.n
    }

    pub(crate) fn nnz(&self) -> usize {
        self.nnz
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

/// Turn cuSOLVER's `singularity` out-parameter into an error.
///
/// `csrlsvchol`/`csrlsvqr` return `-1` when the matrix is non-singular, or the
/// index of the first zero-pivot row otherwise. This is a strictly better
/// diagnostic than the CPU path, which only reports that factorization failed.
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

/// Default tolerance for cuSOLVER's singularity test.
pub(crate) const DEFAULT_SINGULARITY_TOL: f64 = 1e-12;

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

    /// The CSR structure must convert faer's `usize` indices and be reused when
    /// the pattern is unchanged.
    #[test]
    fn csr_structure_converts_and_caches() -> Result<(), Box<dyn std::error::Error>> {
        use faer::sparse::Triplet;

        // 3x3 symmetric tridiagonal.
        let triplets = vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 0, -1.0),
            Triplet::new(0, 1, -1.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 1, -1.0),
            Triplet::new(1, 2, -1.0),
            Triplet::new(2, 2, 2.0),
        ];
        let h = SparseColMat::try_new_from_triplets(3, 3, &triplets)?;

        let mut csr = CsrStructure::default();
        csr.sync(&h)?;

        assert_eq!(csr.n(), 3);
        assert_eq!(csr.nnz(), 7);
        assert_eq!(csr.row_ptr().len(), 4);
        assert_eq!(csr.col_idx().len(), 7);
        assert_eq!(csr.row_ptr()[0], 0);
        assert_eq!(csr.row_ptr()[3], 7);

        // A second sync with the same pattern must be a no-op, not a rebuild.
        let before = csr.row_ptr().as_ptr();
        csr.sync(&h)?;
        assert_eq!(
            before,
            csr.row_ptr().as_ptr(),
            "unchanged pattern must reuse the cached arrays"
        );
        Ok(())
    }
}
