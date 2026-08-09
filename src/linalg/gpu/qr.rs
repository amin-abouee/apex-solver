//! GPU sparse QR via `cusolverSpDcsrlsvqr`.
//!
//! Like the CPU [`SparseQRSolver`](crate::linalg::sparse::SparseQRSolver), this
//! factorizes the **normal-equation matrix** `H = JᵀJ`, not `J` itself — so it
//! has the same conditioning as the Cholesky path, and is a drop-in alternative
//! rather than a numerically superior QR-on-`J`. It is useful because
//! `csrlsvqr` does not require positive definiteness, so it tolerates matrices
//! where Cholesky fails.

use std::ffi::c_int;
use std::ops::Mul;

use cudarc::cusolver::sys as solver_sys;
use faer::Mat;
use faer::sparse::{SparseColMat, Triplet};

use crate::error::ErrorLogging;
use crate::linalg::gpu::device::{
    CsrStructure, DEFAULT_SINGULARITY_TOL, GpuContext, Reordering, check_singularity, check_status,
};
use crate::linalg::{LinAlgError, LinAlgResult, LinearSolver, SparseMode};

/// Sparse QR solver running on an NVIDIA GPU.
///
/// Solves `H · dx = −g` with `H = JᵀJ` (optionally `+ λI`) via
/// `cusolverSpDcsrlsvqr`.
#[derive(Debug)]
pub struct GpuSparseQRSolver {
    context: GpuContext,
    csr: CsrStructure,
    reordering: Reordering,
    /// Undamped `H = JᵀJ` from the last solve.
    hessian: Option<SparseColMat<usize, f64>>,
    /// Positive `g = Jᵀr` from the last solve.
    gradient: Option<Mat<f64>>,
    rhs_scratch: Vec<f64>,
    solution_scratch: Vec<f64>,
    /// Cached `H⁻¹`, invalidated on every solve.
    covariance_matrix: Option<Mat<f64>>,
}

impl GpuSparseQRSolver {
    /// Create a solver on CUDA device 0.
    pub fn new() -> LinAlgResult<Self> {
        Self::with_device(0)
    }

    /// Create a solver on a specific CUDA device.
    pub fn with_device(ordinal: usize) -> LinAlgResult<Self> {
        Ok(Self {
            context: GpuContext::new(ordinal)?,
            csr: CsrStructure::default(),
            reordering: Reordering::default(),
            hessian: None,
            gradient: None,
            rhs_scratch: Vec::new(),
            solution_scratch: Vec::new(),
            covariance_matrix: None,
        })
    }

    /// Choose the fill-reducing reordering. See [`Reordering`].
    pub fn with_reordering(mut self, reordering: Reordering) -> Self {
        self.reordering = reordering;
        self
    }

    pub fn hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.hessian.as_ref()
    }

    pub fn gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }

    fn solve_on_device(
        &mut self,
        hessian: &SparseColMat<usize, f64>,
        gradient: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        self.csr.sync(hessian)?;
        let n = self.csr.n();

        // Invalidate any cached covariance: it belongs to the previous solve.
        self.covariance_matrix = None;

        // cuSOLVER solves A·x = b; we want H·dx = −g.
        self.rhs_scratch.clear();
        self.rhs_scratch.extend((0..n).map(|i| -gradient[(i, 0)]));
        self.solution_scratch.clear();
        self.solution_scratch.resize(n, 0.0);

        let values = hessian.val();
        if values.len() != self.csr.nnz() {
            return Err(LinAlgError::InvalidState(format!(
                "Hessian value count ({}) does not match cached CSR structure ({})",
                values.len(),
                self.csr.nnz()
            ))
            .log());
        }

        let n_i32 = c_int::try_from(n).map_err(|e| {
            LinAlgError::InvalidInput(format!("dimension {n} exceeds i32::MAX")).log_with_source(e)
        })?;
        let nnz_i32 = c_int::try_from(self.csr.nnz()).map_err(|e| {
            LinAlgError::InvalidInput(format!("nnz {} exceeds i32::MAX", self.csr.nnz()))
                .log_with_source(e)
        })?;

        let mut singularity: c_int = -1;

        // SAFETY: every pointer is derived from a live slice that outlives the
        // call. `row_ptr` has n+1 entries and `col_idx`/`values` have nnz entries
        // (checked above); `rhs_scratch` and `solution_scratch` are exactly n
        // long. `handle` and `descr` are valid for the lifetime of
        // `self.context`. cuSOLVER performs the host↔device transfer itself.
        let status = unsafe {
            solver_sys::cusolverSpDcsrlsvqr(
                self.context.handle(),
                n_i32,
                nnz_i32,
                self.context.descr(),
                values.as_ptr(),
                self.csr.row_ptr().as_ptr(),
                self.csr.col_idx().as_ptr(),
                self.rhs_scratch.as_ptr(),
                DEFAULT_SINGULARITY_TOL,
                self.reordering.to_arg(),
                self.solution_scratch.as_mut_ptr(),
                &mut singularity,
            )
        };

        check_status(status, "cusolverSpDcsrlsvqr")?;
        check_singularity(singularity, "cusolverSpDcsrlsvqr")?;

        let solution = &self.solution_scratch;
        Ok(Mat::from_fn(n, 1, |i, _| solution[i]))
    }
}

fn add_damping(
    hessian: &SparseColMat<usize, f64>,
    lambda: f64,
) -> LinAlgResult<SparseColMat<usize, f64>> {
    let n = hessian.ncols();
    let lambda_triplets: Vec<Triplet<usize, usize, f64>> =
        (0..n).map(|i| Triplet::new(i, i, lambda)).collect();
    let lambda_i = SparseColMat::try_new_from_triplets(n, n, &lambda_triplets).map_err(|e| {
        LinAlgError::SparseMatrixCreation("Failed to create lambda*I matrix".to_string())
            .log_with_source(e)
    })?;
    Ok(hessian + lambda_i)
}

fn normal_matrix(jacobians: &SparseColMat<usize, f64>) -> LinAlgResult<SparseColMat<usize, f64>> {
    let jt = jacobians.as_ref().transpose();
    Ok(jt
        .to_col_major()
        .map_err(|e| {
            LinAlgError::MatrixConversion(
                "Failed to convert transposed Jacobian to column-major format".to_string(),
            )
            .log_with_source(e)
        })?
        .mul(jacobians.as_ref()))
}

impl LinearSolver<SparseMode> for GpuSparseQRSolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        let hessian = normal_matrix(jacobians)?;
        let gradient = jacobians.as_ref().transpose().mul(residuals);

        let dx = self.solve_on_device(&hessian, &gradient)?;

        self.hessian = Some(hessian);
        self.gradient = Some(gradient);
        Ok(dx)
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>> {
        let hessian = normal_matrix(jacobians)?;
        let gradient = jacobians.as_ref().transpose().mul(residuals);

        let augmented = add_damping(&hessian, lambda)?;
        let dx = self.solve_on_device(&augmented, &gradient)?;

        // Cache the UNDAMPED Hessian, matching every other solver.
        self.hessian = Some(hessian);
        self.gradient = Some(gradient);
        Ok(dx)
    }

    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.hessian.as_ref()
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }

    /// Covariance as `H⁻¹` from the **undamped** Hessian.
    ///
    /// Implemented explicitly rather than inheriting the trait's `None` default,
    /// so enabling covariance on the GPU path does not silently produce nothing.
    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>> {
        if self.covariance_matrix.is_none() {
            let hessian = self.hessian.as_ref()?;
            self.covariance_matrix = crate::linalg::gpu::cholesky::invert_undamped_hessian(hessian);
        }
        self.covariance_matrix.as_ref()
    }

    fn get_covariance_matrix(&self) -> Option<&Mat<f64>> {
        self.covariance_matrix.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_without_a_device_errors_cleanly() {
        if crate::linalg::gpu::is_available() {
            return;
        }
        match GpuSparseQRSolver::new() {
            Ok(_) => panic!("expected construction to fail without a CUDA device"),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("CUDA") || msg.contains("cuSOLVER"),
                    "error should mention CUDA: {msg}"
                );
            }
        }
    }
}
