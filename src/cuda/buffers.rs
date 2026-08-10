//! Host↔device staging for a sparse linear system.
//!
//! Splits into two halves that change on different schedules:
//!
//! - [`CsrStructure`] — the `i32` CSR pattern, on the host. `H = JᵀJ` keeps a
//!   constant sparsity pattern for a whole optimization (the CPU solvers rely on
//!   the same invariant to cache their symbolic factorizations), so the
//!   `usize → i32` narrowing happens once.
//! - [`DeviceSystem`] — the device-resident copies. Structure arrays are uploaded
//!   once per pattern; values and the right-hand side are overwritten every solve.
//!
//! Nothing here is `unsafe`: cudarc's `device_ptr` is a safe accessor. The raw
//! pointers it yields are only *used* in [`super::context`], which is where the
//! FFI calls live.

use std::ffi::{c_int, c_void};

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use faer::sparse::SparseColMat;

use crate::cuda::profile::{CudaProfile, DeviceMemory, HostTimer};
use crate::error::ErrorLogging;
use crate::linalg::{LinAlgError, LinAlgResult};

/// Turn a cudarc driver failure into a `LinAlgError` naming the operation.
pub(crate) fn to_err(what: &'static str) -> impl FnOnce(cudarc::driver::DriverError) -> LinAlgError {
    move |e| LinAlgError::InvalidState(format!("CUDA {what} failed")).log_with_source(e)
}

/// The CSR structure of `H`, narrowed to `i32` once and reused.
///
/// `H` is **symmetric**, so faer's CSC storage is already a valid CSR description
/// of the same matrix — no transpose or repacking is needed.
#[derive(Debug, Default, Clone)]
pub(crate) struct CsrStructure {
    row_ptr: Vec<c_int>,
    col_idx: Vec<c_int>,
    n: usize,
    nnz: usize,
}

impl CsrStructure {
    /// Rebuild only when the pattern actually changed.
    ///
    /// Returns `true` when the arrays were rebuilt, which is the signal that
    /// device buffers and any cached analysis are stale.
    pub(crate) fn sync(
        &mut self,
        hessian: &SparseColMat<usize, f64>,
        profile: &mut CudaProfile,
    ) -> LinAlgResult<bool> {
        let n = hessian.ncols();
        let symbolic = hessian.symbolic();
        let nnz = symbolic.row_idx().len();

        if self.n == n && self.nnz == nnz && !self.row_ptr.is_empty() {
            return Ok(false);
        }

        let _timer = HostTimer::start(&mut profile.pattern_conversion);

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
        Ok(true)
    }

    /// Force the next [`sync`](Self::sync) to rebuild from the Hessian.
    ///
    /// Needed because the reusable Cholesky path permutes these arrays in place:
    /// once permuted, they are no longer the Hessian's own pattern, so anything
    /// that must start from the original ordering — recomputing a permutation,
    /// say — has to discard them first.
    pub(crate) fn invalidate(&mut self) {
        self.row_ptr.clear();
        self.col_idx.clear();
        self.n = 0;
        self.nnz = 0;
    }

    pub(crate) fn row_ptr(&self) -> &[c_int] {
        &self.row_ptr
    }

    pub(crate) fn col_idx(&self) -> &[c_int] {
        &self.col_idx
    }

    /// Mutable views, for the in-place rewrite `cusolverSpXcsrpermHost` performs.
    pub(crate) fn pattern_mut(&mut self) -> (&mut [c_int], &mut [c_int]) {
        (&mut self.row_ptr, &mut self.col_idx)
    }

    pub(crate) fn n(&self) -> usize {
        self.n
    }

    pub(crate) fn nnz(&self) -> usize {
        self.nnz
    }

    pub(crate) fn n_i32(&self) -> LinAlgResult<c_int> {
        to_i32(self.n, "dimension")
    }

    pub(crate) fn nnz_i32(&self) -> LinAlgResult<c_int> {
        to_i32(self.nnz, "nnz")
    }
}

fn to_i32(value: usize, what: &str) -> LinAlgResult<c_int> {
    c_int::try_from(value).map_err(|e| {
        LinAlgError::InvalidInput(format!(
            "{what} ({value}) exceeds i32::MAX; cuSOLVER's sparse API is 32-bit indexed"
        ))
        .log_with_source(e)
    })
}

/// Raw device pointers plus dimensions, in the types cuSOLVER expects.
///
/// Produced by [`DeviceSystem::with_ptrs`] and valid only inside that closure,
/// where the stream-ordering guards are still alive.
pub(crate) struct SystemPtrs {
    pub(crate) n: c_int,
    pub(crate) nnz: c_int,
    pub(crate) vals: *const f64,
    pub(crate) row_ptr: *const c_int,
    pub(crate) col_idx: *const c_int,
    pub(crate) b: *const f64,
    pub(crate) x: *mut f64,
    pub(crate) workspace: *mut c_void,
}

/// Device-resident copies of everything a sparse solve reads and writes.
///
/// `cusolverSpDcsrlsvchol`/`csrlsvqr` and the low-level `Dcsrchol*` family — the
/// unsuffixed, non-`Host` entry points — require `csrVal`, `csrRowPtr`,
/// `csrColInd`, `b` and `x` in **device** memory; only the `singularity` and
/// `position` out-parameters stay host `int*`. (The `...Host` variants accept
/// host pointers and run the factorization on the CPU, which would defeat the
/// purpose.)
pub(crate) struct DeviceSystem {
    row_ptr: CudaSlice<c_int>,
    col_idx: CudaSlice<c_int>,
    vals: CudaSlice<f64>,
    b: CudaSlice<f64>,
    x: CudaSlice<f64>,
    /// Scratch for the low-level factorization; one byte until sized.
    workspace: CudaSlice<u8>,
    n: usize,
    nnz: usize,
    workspace_bytes: usize,
    /// Staging for `−g`, so a solve allocates nothing in steady state.
    rhs_host: Vec<f64>,
    /// Staging for the downloaded solution.
    solution_host: Vec<f64>,
}

impl std::fmt::Debug for DeviceSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceSystem")
            .field("n", &self.n)
            .field("nnz", &self.nnz)
            .field("workspace_bytes", &self.workspace_bytes)
            .finish_non_exhaustive()
    }
}

impl DeviceSystem {
    /// Allocate for `csr` and upload its structure.
    pub(crate) fn new(stream: &Arc<CudaStream>, csr: &CsrStructure) -> LinAlgResult<Self> {
        let (n, nnz) = (csr.n(), csr.nnz());
        Ok(Self {
            row_ptr: stream
                .clone_htod(csr.row_ptr())
                .map_err(to_err("row-pointer upload"))?,
            col_idx: stream
                .clone_htod(csr.col_idx())
                .map_err(to_err("column-index upload"))?,
            vals: stream
                .alloc_zeros::<f64>(nnz)
                .map_err(to_err("value allocation"))?,
            b: stream
                .alloc_zeros::<f64>(n)
                .map_err(to_err("right-hand-side allocation"))?,
            x: stream
                .alloc_zeros::<f64>(n)
                .map_err(to_err("solution allocation"))?,
            workspace: stream
                .alloc_zeros::<u8>(1)
                .map_err(to_err("workspace allocation"))?,
            n,
            nnz,
            workspace_bytes: 0,
            rhs_host: vec![0.0; n],
            solution_host: vec![0.0; n],
        })
    }

    /// Does this allocation still match `csr`?
    pub(crate) fn matches(&self, csr: &CsrStructure) -> bool {
        self.n == csr.n() && self.nnz == csr.nnz()
    }

    /// Grow the factorization scratch to `bytes`, as reported by cuSOLVER.
    pub(crate) fn size_workspace(&mut self, stream: &Arc<CudaStream>, bytes: usize) -> LinAlgResult<()> {
        if bytes <= self.workspace_bytes {
            return Ok(());
        }
        self.workspace = stream
            .alloc_zeros::<u8>(bytes.max(1))
            .map_err(to_err("workspace allocation"))?;
        self.workspace_bytes = bytes;
        Ok(())
    }

    /// Copy matrix values to the device.
    pub(crate) fn upload_values(
        &mut self,
        stream: &Arc<CudaStream>,
        values: &[f64],
    ) -> LinAlgResult<()> {
        if values.len() != self.nnz {
            return Err(LinAlgError::InvalidState(format!(
                "value count ({}) does not match the allocated pattern ({})",
                values.len(),
                self.nnz
            ))
            .log());
        }
        stream
            .memcpy_htod(values, &mut self.vals)
            .map_err(to_err("value upload"))
    }

    /// Copy the right-hand side to the device.
    ///
    /// `rhs` is filled by `write` rather than passed in, so the caller can build
    /// `−g` (optionally permuted) directly into the reusable staging buffer.
    pub(crate) fn upload_rhs(
        &mut self,
        stream: &Arc<CudaStream>,
        write: impl FnOnce(&mut [f64]),
    ) -> LinAlgResult<()> {
        write(&mut self.rhs_host);
        stream
            .memcpy_htod(&self.rhs_host, &mut self.b)
            .map_err(to_err("right-hand-side upload"))
    }

    /// Copy the solution back. Read it with [`solution`](Self::solution) after
    /// the stream has been synchronized.
    pub(crate) fn download_solution(&mut self, stream: &Arc<CudaStream>) -> LinAlgResult<()> {
        stream
            .memcpy_dtoh(&self.x, &mut self.solution_host)
            .map_err(to_err("solution download"))
    }

    pub(crate) fn solution(&self) -> &[f64] {
        &self.solution_host
    }

    /// Bytes resident on the device for this system.
    pub(crate) fn memory(&self) -> DeviceMemory {
        DeviceMemory {
            structure: (self.n + 1 + self.nnz) * size_of::<c_int>(),
            values: self.nnz * size_of::<f64>(),
            vectors: 2 * self.n * size_of::<f64>(),
            workspace: self.workspace_bytes,
            internal: 0,
        }
    }

    /// Run `f` with raw device pointers to every buffer.
    ///
    /// The stream-ordering guards returned by cudarc are bound for the duration
    /// of the closure, so the pointers stay valid throughout — which is exactly
    /// the invariant the `SAFETY` comments in [`super::context`] rely on. The
    /// borrows are of distinct fields, so the mutable `x` and `workspace`
    /// borrows do not conflict with the shared ones.
    pub(crate) fn with_ptrs<R>(
        &mut self,
        stream: &Arc<CudaStream>,
        n: c_int,
        nnz: c_int,
        f: impl FnOnce(SystemPtrs) -> R,
    ) -> R {
        let (vals, _g_vals) = self.vals.device_ptr(stream);
        let (row_ptr, _g_row) = self.row_ptr.device_ptr(stream);
        let (col_idx, _g_col) = self.col_idx.device_ptr(stream);
        let (b, _g_b) = self.b.device_ptr(stream);
        let (x, _g_x) = self.x.device_ptr_mut(stream);
        let (workspace, _g_ws) = self.workspace.device_ptr_mut(stream);
        f(SystemPtrs {
            n,
            nnz,
            vals: vals as *const f64,
            row_ptr: row_ptr as *const c_int,
            col_idx: col_idx as *const c_int,
            b: b as *const f64,
            x: x as *mut f64,
            workspace: workspace as *mut c_void,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::Triplet;

    /// The CSR structure must convert faer's `usize` indices and be reused when
    /// the pattern is unchanged.
    #[test]
    fn csr_structure_converts_and_caches() -> Result<(), Box<dyn std::error::Error>> {
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

        let mut profile = CudaProfile::default();
        let mut csr = CsrStructure::default();
        assert!(csr.sync(&h, &mut profile)?, "first sync must build");

        assert_eq!(csr.n(), 3);
        assert_eq!(csr.nnz(), 7);
        assert_eq!(csr.row_ptr().len(), 4);
        assert_eq!(csr.col_idx().len(), 7);
        assert_eq!(csr.row_ptr()[0], 0);
        assert_eq!(csr.row_ptr()[3], 7);
        assert_eq!(profile.pattern_conversion.calls, 1);

        // A second sync with the same pattern must be a no-op, not a rebuild.
        let before = csr.row_ptr().as_ptr();
        assert!(!csr.sync(&h, &mut profile)?, "unchanged pattern must not rebuild");
        assert_eq!(
            before,
            csr.row_ptr().as_ptr(),
            "unchanged pattern must reuse the cached arrays"
        );
        assert_eq!(
            profile.pattern_conversion.calls, 1,
            "a no-op sync must not be timed as work"
        );
        Ok(())
    }

    #[test]
    fn dimension_accessors_narrow_to_i32() -> Result<(), Box<dyn std::error::Error>> {
        let h = SparseColMat::try_new_from_triplets(2, 2, &[Triplet::new(0, 0, 1.0)])?;
        let mut profile = CudaProfile::default();
        let mut csr = CsrStructure::default();
        csr.sync(&h, &mut profile)?;
        assert_eq!(csr.n_i32()?, 2);
        assert_eq!(csr.nnz_i32()?, 1);
        Ok(())
    }
}
