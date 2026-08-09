//! GPU sparse Cholesky via `cusolverSpDcsrlsvchol`.
//!
//! Mirrors [`crate::linalg::sparse::cholesky::SparseCholeskySolver`] exactly in
//! its caching contract, so the optimizers cannot tell the two apart:
//! `get_hessian` returns the **undamped** `JᵀJ` and `get_gradient` the
//! **positive** `Jᵀr`, even from the augmented solve.

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

/// Sparse Cholesky solver running on an NVIDIA GPU.
///
/// Solves `H · dx = −g` with `H = JᵀJ` (optionally `+ λI`), using
/// `cusolverSpDcsrlsvchol`. `H` must be symmetric positive definite; a
/// rank-deficient system produces [`LinAlgError::SingularMatrix`] naming the
/// offending row.
#[derive(Debug)]
pub struct GpuSparseCholeskySolver {
    context: GpuContext,
    csr: CsrStructure,
    reordering: Reordering,
    /// Undamped `H = JᵀJ` from the last solve.
    hessian: Option<SparseColMat<usize, f64>>,
    /// Positive `g = Jᵀr` from the last solve.
    gradient: Option<Mat<f64>>,
    /// Reused staging buffer for `−g`, so the augmented path allocates nothing
    /// in steady state.
    rhs_scratch: Vec<f64>,
    /// Reused staging buffer for the solution.
    solution_scratch: Vec<f64>,
    /// Cached `H⁻¹`. Invalidated on every solve so a reused solver can never
    /// return a covariance belonging to a previous problem.
    covariance_matrix: Option<Mat<f64>>,
}

impl GpuSparseCholeskySolver {
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

    /// Upload `H` and `−g`, run the factor-and-solve, download `dx`.
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

        // SAFETY: every pointer below is derived from a live slice that outlives
        // the call. `row_ptr` has n+1 entries and `col_idx`/`values` have nnz
        // entries (checked above); `rhs_scratch` and `solution_scratch` are both
        // exactly n long. `handle` and `descr` are valid for the lifetime of
        // `self.context`. cuSOLVER performs the host↔device transfer itself.
        let status = unsafe {
            solver_sys::cusolverSpDcsrlsvchol(
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

        check_status(status, "cusolverSpDcsrlsvchol")?;
        check_singularity(singularity, "cusolverSpDcsrlsvchol")?;

        let solution = &self.solution_scratch;
        Ok(Mat::from_fn(n, 1, |i, _| solution[i]))
    }
}

/// Build `H + λI` without disturbing the caller's undamped `H`.
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

/// `H = JᵀJ`, matching the CPU solver's construction exactly.
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

impl LinearSolver<SparseMode> for GpuSparseCholeskySolver {
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

        // Cache the UNDAMPED Hessian — Dog Leg needs the true quadratic model,
        // and covariance estimation must never see lambda.
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
    /// Inverting the undamped `H` — rather than reusing a damped factorization —
    /// keeps Levenberg-Marquardt's `λ` out of the reported uncertainty.
    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>> {
        if self.covariance_matrix.is_none() {
            let hessian = self.hessian.as_ref()?;
            self.covariance_matrix = invert_undamped_hessian(hessian);
        }
        self.covariance_matrix.as_ref()
    }

    fn get_covariance_matrix(&self) -> Option<&Mat<f64>> {
        self.covariance_matrix.as_ref()
    }
}

/// Invert `H` on the CPU via a fresh Cholesky of the undamped matrix.
///
/// Deliberately does not reuse any factorization produced during a solve: the
/// augmented solve factorizes `H + λI`, and covariance must reflect `H` alone.
pub(crate) fn invert_undamped_hessian(hessian: &SparseColMat<usize, f64>) -> Option<Mat<f64>> {
    use faer::Side;
    use faer::linalg::solvers::Solve;
    use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};

    let n = hessian.ncols();
    let symbolic = SymbolicLlt::try_new(hessian.symbolic(), Side::Lower).ok()?;
    let llt = Llt::try_new_with_symbolic(symbolic, hessian.as_ref(), Side::Lower).ok()?;
    Some(llt.solve(Mat::<f64>::identity(n, n)))
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn tridiagonal(n: usize) -> Result<SparseColMat<usize, f64>, faer::sparse::CreationError> {
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push(Triplet::new(i, i, 2.0));
            if i + 1 < n {
                triplets.push(Triplet::new(i + 1, i, -1.0));
                triplets.push(Triplet::new(i, i + 1, -1.0));
            }
        }
        SparseColMat::try_new_from_triplets(n, n, &triplets)
    }

    /// Damping must not disturb the undamped matrix, and must land exactly on
    /// the diagonal.
    #[test]
    fn add_damping_adds_lambda_to_the_diagonal_only() -> TestResult {
        let h = tridiagonal(4)?;
        let damped = add_damping(&h, 0.5)?;

        for i in 0..4 {
            let original = h.get(i, i).copied().unwrap_or(0.0);
            let with_lambda = damped.get(i, i).copied().unwrap_or(0.0);
            assert!(
                (with_lambda - (original + 0.5)).abs() < 1e-15,
                "diagonal {i}: {original} + 0.5 != {with_lambda}"
            );
        }
        // Off-diagonals untouched.
        for i in 0..3 {
            let original = h.get(i + 1, i).copied().unwrap_or(0.0);
            let after = damped.get(i + 1, i).copied().unwrap_or(0.0);
            assert!((original - after).abs() < 1e-15);
        }
        // And the original is unchanged.
        assert!((h.get(0, 0).copied().unwrap_or(0.0) - 2.0).abs() < 1e-15);
        Ok(())
    }

    /// `H = JᵀJ` must match a hand-computed product, since everything downstream
    /// depends on this being identical to the CPU solver's construction.
    #[test]
    fn normal_matrix_matches_hand_computed_jtj() -> TestResult {
        // J = [[1, 0], [1, 1], [0, 2]]  =>  JᵀJ = [[2, 1], [1, 5]]
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 1, 2.0),
        ];
        let j = SparseColMat::try_new_from_triplets(3, 2, &triplets)?;
        let h = normal_matrix(&j)?;

        assert_eq!(h.nrows(), 2);
        assert_eq!(h.ncols(), 2);
        for (r, c, expected) in [(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 5.0)] {
            let actual = h.get(r, c).copied().unwrap_or(0.0);
            assert!(
                (actual - expected).abs() < 1e-12,
                "H[{r},{c}] = {actual}, expected {expected}"
            );
        }
        Ok(())
    }

    /// Constructing a solver on a machine with no GPU must return a typed error,
    /// not panic — this is what keeps CI green on GPU-less runners.
    #[test]
    fn construction_without_a_device_errors_cleanly() {
        if crate::linalg::gpu::is_available() {
            return; // On a real GPU box this is expected to succeed.
        }
        match GpuSparseCholeskySolver::new() {
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
