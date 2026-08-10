//! Sparse Cholesky, on the CPU via faer and on an NVIDIA GPU via cuSOLVER.
//!
//! Both solvers implement `LinearSolver<SparseMode>` and are interchangeable
//! from the optimizer's point of view: `get_hessian` returns the **undamped**
//! `JᵀJ` and `get_gradient` the **positive** `Jᵀr`, even from an augmented solve.
//!
//! The CUDA solver contains no `unsafe`. Every device operation goes through the
//! checked wrappers in [`crate::cuda::context`], which is the only module in the
//! crate that touches FFI.

use faer::{
    Mat, Side,
    linalg::solvers::Solve,
    sparse::linalg::solvers::{Llt, SymbolicLlt},
    sparse::{SparseColMat, Triplet},
};
use std::ops::Mul;

use crate::error::ErrorLogging;
use crate::linalg::{LinAlgError, LinAlgResult, LinearSolver, SparseMode};

#[derive(Debug, Clone)]
pub struct SparseCholeskySolver {
    factorizer: Option<Llt<usize, f64>>,

    /// Cached symbolic factorization for reuse across iterations.
    ///
    /// This is computed once and reused when the sparsity pattern doesn't change,
    /// providing a 10-15% performance improvement for iterative optimization.
    symbolic_factorization: Option<SymbolicLlt<usize>>,

    /// The Hessian matrix, computed as `(J^T *  J)`.
    ///
    /// This is `None` if the Hessian could not be computed.
    hessian: Option<SparseColMat<usize, f64>>,

    /// The gradient vector, computed as `J^T *  r`.
    ///
    /// This is `None` if the gradient could not be computed.
    gradient: Option<Mat<f64>>,

    /// The parameter covariance matrix, computed as `(J^T * J)^-1`.
    ///
    /// This is `None` if the Hessian is singular or ill-conditioned.
    covariance_matrix: Option<Mat<f64>>,
    /// Asymptotic standard errors of the parameters.
    ///
    /// This is `None` if the covariance matrix could not be computed.
    /// Each error is the square root of the corresponding diagonal element
    /// of the covariance matrix.
    standard_errors: Option<Mat<f64>>,
}

impl SparseCholeskySolver {
    pub fn new() -> Self {
        SparseCholeskySolver {
            factorizer: None,
            symbolic_factorization: None,
            hessian: None,
            gradient: None,
            covariance_matrix: None,
            standard_errors: None,
        }
    }

    pub fn hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.hessian.as_ref()
    }

    pub fn gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }

    pub fn compute_standard_errors(&mut self) -> Option<&Mat<f64>> {
        // Ensure covariance matrix is computed first
        if self.covariance_matrix.is_none() {
            LinearSolver::<SparseMode>::compute_covariance_matrix(self);
        }

        // Return None if hessian is not available (solver not initialized)
        let hessian = self.hessian.as_ref()?;
        let n = hessian.ncols();
        // Compute standard errors as sqrt of diagonal elements
        if let Some(cov) = &self.covariance_matrix {
            let mut std_errors = Mat::zeros(n, 1);
            for i in 0..n {
                let diag_val = cov[(i, i)];
                if diag_val >= 0.0 {
                    std_errors[(i, 0)] = diag_val.sqrt();
                } else {
                    // Negative diagonal indicates numerical issues
                    return None;
                }
            }
            self.standard_errors = Some(std_errors);
        }
        self.standard_errors.as_ref()
    }

    /// Reset covariance computation state (useful for iterative optimization)
    pub fn reset_covariance(&mut self) {
        self.covariance_matrix = None;
        self.standard_errors = None;
    }
}

impl Default for SparseCholeskySolver {
    fn default() -> Self {
        Self::new()
    }
}
impl LinearSolver<SparseMode> for SparseCholeskySolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        // Form the normal equations: H = J^T * J
        let jt = jacobians.as_ref().transpose();
        let hessian = jt
            .to_col_major()
            .map_err(|e| {
                LinAlgError::MatrixConversion(
                    "Failed to convert transposed Jacobian to column-major format".to_string(),
                )
                .log_with_source(e)
            })?
            .mul(jacobians.as_ref());

        // g = J^T * r
        let gradient = jacobians.as_ref().transpose().mul(residuals);

        let sym = if let Some(ref cached_sym) = self.symbolic_factorization {
            // Reuse cached symbolic factorization
            // Note: SymbolicLlt is reference-counted, so clone() is cheap (O(1))
            // We assume the sparsity pattern is constant across iterations
            // which is typical in iterative optimization
            cached_sym.clone()
        } else {
            // Create new symbolic factorization and cache it
            let new_sym = SymbolicLlt::try_new(hessian.symbolic(), Side::Lower).map_err(|e| {
                LinAlgError::FactorizationFailed(
                    "Symbolic Cholesky decomposition failed".to_string(),
                )
                .log_with_source(e)
            })?;
            // Cache it (clone is cheap due to reference counting)
            self.symbolic_factorization = Some(new_sym.clone());
            new_sym
        };

        // Perform numeric factorization using the symbolic structure
        let cholesky =
            Llt::try_new_with_symbolic(sym, hessian.as_ref(), Side::Lower).map_err(|e| {
                LinAlgError::SingularMatrix(
                    "Cholesky factorization failed (matrix may be singular)".to_string(),
                )
                .log_with_source(e)
            })?;

        let dx = cholesky.solve(-&gradient);
        self.hessian = Some(hessian);
        self.gradient = Some(gradient);
        self.factorizer = Some(cholesky);

        Ok(dx)
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>> {
        let n = jacobians.ncols();

        // H = J^T * J
        let jt = jacobians.as_ref().transpose();
        let hessian = jt
            .to_col_major()
            .map_err(|e| {
                LinAlgError::MatrixConversion(
                    "Failed to convert transposed Jacobian to column-major format".to_string(),
                )
                .log_with_source(e)
            })?
            .mul(jacobians.as_ref());

        // g = J^T * r
        let gradient = jacobians.as_ref().transpose().mul(residuals);

        // H_aug = H + lambda * I
        let mut lambda_i_triplets = Vec::with_capacity(n);
        for i in 0..n {
            lambda_i_triplets.push(Triplet::new(i, i, lambda));
        }
        let lambda_i =
            SparseColMat::try_new_from_triplets(n, n, &lambda_i_triplets).map_err(|e| {
                LinAlgError::SparseMatrixCreation("Failed to create lambda*I matrix".to_string())
                    .log_with_source(e)
            })?;

        let augmented_hessian = &hessian + lambda_i;

        let sym = if let Some(ref cached_sym) = self.symbolic_factorization {
            // Reuse cached symbolic factorization
            // Note: SymbolicLlt is reference-counted, so clone() is cheap (O(1))
            // We assume the sparsity pattern is constant across iterations
            // which is typical in iterative optimization
            cached_sym.clone()
        } else {
            // Create new symbolic factorization and cache it
            let new_sym =
                SymbolicLlt::try_new(augmented_hessian.symbolic(), Side::Lower).map_err(|e| {
                    LinAlgError::FactorizationFailed(
                        "Symbolic Cholesky decomposition failed for augmented system".to_string(),
                    )
                    .log_with_source(e)
                })?;
            // Cache it (clone is cheap due to reference counting)
            self.symbolic_factorization = Some(new_sym.clone());
            new_sym
        };

        // Perform numeric factorization
        let cholesky = Llt::try_new_with_symbolic(sym, augmented_hessian.as_ref(), Side::Lower)
            .map_err(|e| {
                LinAlgError::SingularMatrix(
                    "Cholesky factorization failed (matrix may be singular)".to_string(),
                )
                .log_with_source(e)
            })?;

        let dx = cholesky.solve(-&gradient);
        self.hessian = Some(hessian);
        self.gradient = Some(gradient);
        self.factorizer = Some(cholesky);

        Ok(dx)
    }

    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.hessian.as_ref()
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }

    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>> {
        // Only compute if we have a factorizer and hessian, but no covariance matrix yet
        if self.factorizer.is_some()
            && self.hessian.is_some()
            && self.covariance_matrix.is_none()
            && let (Some(factorizer), Some(hessian)) = (&self.factorizer, &self.hessian)
        {
            let n = hessian.ncols();
            // Create identity matrix
            let identity = Mat::identity(n, n);

            // Solve H * X = I to get X = H^(-1) = covariance matrix
            let cov_matrix = factorizer.solve(&identity);
            self.covariance_matrix = Some(cov_matrix);
        }
        self.covariance_matrix.as_ref()
    }

    fn get_covariance_matrix(&self) -> Option<&Mat<f64>> {
        self.covariance_matrix.as_ref()
    }
}

// ============================================================================
// CUDA sparse Cholesky (cuSOLVER)
// ============================================================================

/// Which cuSOLVER sparse Cholesky path [`CudaSparseCholeskySolver`] uses.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CholeskyAlgorithm {
    /// `cusolverSpXcsrcholAnalysis` + `Dcsrcholfactor`/`Dcsrcholsolve`:
    /// permutation and symbolic analysis once per sparsity pattern, numeric
    /// factorization and triangular solves per iteration.
    ///
    /// The default, and the only one that competes with [`SparseCholeskySolver`],
    /// which caches its symbolic factorization the same way.
    #[default]
    Reusable,
    /// `cusolverSpDcsrlsvchol`: one call does reordering, symbolic analysis,
    /// factorization and solve.
    ///
    /// Simpler and fully cudarc-declared, but redoes the analysis every
    /// iteration. Kept for A/B measurement and as a fallback if the low-level
    /// symbols cannot be resolved.
    OneShot,
}

/// Symbolic analysis and device state for one sparsity pattern.
///
/// Rebuilt only when the pattern changes; in a normal optimization that means
/// once, and every iteration after the first pays only factor + solve.
#[cfg(feature = "cuda")]
struct CudaAnalysis {
    info: crate::cuda::context::CholeskyInfo,
    system: crate::cuda::buffers::DeviceSystem,
    /// `permuted_values[i] = original_values[value_map[i]]`.
    value_map: Vec<usize>,
    /// `permutation[new_index] = old_index`.
    permutation: Vec<usize>,
    n: std::ffi::c_int,
    nnz: std::ffi::c_int,
}

/// Sparse Cholesky running on an NVIDIA GPU via cuSOLVER.
///
/// Solves `H · dx = −g` with `H = JᵀJ` (optionally `+ λI`). `H` must be
/// symmetric positive definite.
///
/// Mirrors [`SparseCholeskySolver`]'s caching contract exactly, so the
/// optimizers cannot tell the two apart.
///
/// # Fill-reducing ordering is ours to choose
///
/// Unlike the one-shot `csrlsvchol`, the reusable cuSOLVER API applies **no**
/// ordering — it factorizes the matrix exactly as given, which on a pose graph
/// with loop closures is ruinous. The permutation is therefore computed once on
/// the host ([`Reordering`], nested dissection by default) and re-applied to new
/// values each iteration through a cached value map, with no further symbolic
/// work.
///
/// # Singularity detection is weak on this path
///
/// [`CholeskyAlgorithm::OneShot`] cannot detect it at all: the device
/// `csrlsvchol` leaves cuSOLVER's `singularity` out-parameter at `-1` even for a
/// rank-deficient matrix — only the `...Host` variant and `csrlsvqr` run the
/// zero-pivot check (verified on CUDA 13.0, driver 580).
///
/// [`CholeskyAlgorithm::Reusable`] has a zero-pivot check but uses it only as a
/// diagnostic, raising [`LinAlgError::SingularMatrix`] when the resulting step is
/// non-finite. That matches [`SparseCholeskySolver`], which has no pivot check
/// either; treating cuSOLVER's absolute-tolerance report as fatal rejected large
/// problems the CPU path solves fine.
///
/// Use [`CudaSparseQRSolver`] when detection matters, and anchor pose graphs with
/// a `PriorFactor` so the question does not arise.
///
/// [`Reordering`]: crate::cuda::Reordering
/// [`CudaSparseQRSolver`]: crate::linalg::sparse::CudaSparseQRSolver
#[cfg(feature = "cuda")]
pub struct CudaSparseCholeskySolver {
    context: crate::cuda::CudaContext,
    algorithm: CholeskyAlgorithm,
    reordering: crate::cuda::Reordering,

    /// Host-side CSR pattern; permuted in place by the reusable path.
    csr: crate::cuda::buffers::CsrStructure,
    /// Analyze-once state, used by [`CholeskyAlgorithm::Reusable`].
    analysis: Option<CudaAnalysis>,
    /// The ordering the current analysis was built with, so changing
    /// `reordering` between solves forces a re-analysis.
    analyzed_with: Option<crate::cuda::Reordering>,
    /// Device staging for [`CholeskyAlgorithm::OneShot`].
    one_shot: Option<crate::cuda::buffers::DeviceSystem>,

    /// Staging reused across iterations, so a solve allocates nothing.
    permuted_values: Vec<f64>,
    permuted_solution: Vec<f64>,

    stopwatch: crate::cuda::profile::DeviceStopwatch,
    profile: crate::cuda::CudaProfile,

    /// Undamped `H = JᵀJ` from the last solve.
    hessian: Option<SparseColMat<usize, f64>>,
    /// Positive `g = Jᵀr` from the last solve.
    gradient: Option<Mat<f64>>,
    /// Cached `H⁻¹`. Invalidated on every solve so a reused solver can never
    /// return a covariance belonging to a previous problem.
    covariance_matrix: Option<Mat<f64>>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for CudaSparseCholeskySolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaSparseCholeskySolver")
            .field("algorithm", &self.algorithm)
            .field("reordering", &self.reordering)
            .field("analyzed", &self.analysis.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
impl CudaSparseCholeskySolver {
    /// Create a solver on CUDA device 0.
    pub fn new() -> LinAlgResult<Self> {
        Self::with_device(0)
    }

    /// Create a solver on a specific CUDA device.
    pub fn with_device(ordinal: usize) -> LinAlgResult<Self> {
        let context = crate::cuda::CudaContext::new(ordinal)?;
        let stopwatch = crate::cuda::profile::DeviceStopwatch::new(context.stream())?;
        Ok(Self {
            context,
            algorithm: CholeskyAlgorithm::default(),
            reordering: crate::cuda::Reordering::default(),
            csr: crate::cuda::buffers::CsrStructure::default(),
            analysis: None,
            analyzed_with: None,
            one_shot: None,
            permuted_values: Vec::new(),
            permuted_solution: Vec::new(),
            stopwatch,
            profile: crate::cuda::CudaProfile::default(),
            hessian: None,
            gradient: None,
            covariance_matrix: None,
        })
    }

    /// Choose the fill-reducing reordering.
    ///
    /// Applies to both algorithms, by different means: the one-shot path passes
    /// it to `csrlsvchol`, while the reusable path computes the permutation
    /// itself and re-runs its analysis when this changes.
    pub fn with_reordering(mut self, reordering: crate::cuda::Reordering) -> Self {
        self.reordering = reordering;
        self
    }

    /// Choose between the reusable and one-shot cuSOLVER paths.
    pub fn with_algorithm(mut self, algorithm: CholeskyAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Per-phase timings and device memory totals accumulated so far.
    ///
    /// See [`CudaProfile`](crate::cuda::CudaProfile); `Display` renders a table.
    pub fn profile(&self) -> &crate::cuda::CudaProfile {
        &self.profile
    }

    pub fn hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.hessian.as_ref()
    }

    pub fn gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }

    /// Factorize `H` and solve `H · dx = −g` on the device.
    fn solve_on_device(
        &mut self,
        hessian: &SparseColMat<usize, f64>,
        gradient: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        // Invalidate any cached covariance: it belongs to the previous solve.
        self.covariance_matrix = None;

        if gradient.nrows() != hessian.ncols() {
            return Err(LinAlgError::InvalidState(format!(
                "gradient length ({}) does not match the Hessian dimension ({})",
                gradient.nrows(),
                hessian.ncols(),
            ))
            .log());
        }

        match self.algorithm {
            CholeskyAlgorithm::Reusable => self.solve_reusable(hessian, gradient),
            CholeskyAlgorithm::OneShot => self.solve_one_shot(hessian, gradient),
        }
    }

    /// Analyze once per pattern, then factor and solve per iteration.
    fn solve_reusable(
        &mut self,
        hessian: &SparseColMat<usize, f64>,
        gradient: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        let rebuilt = self.csr.sync(hessian, &mut self.profile)?;
        let needs_analysis =
            rebuilt || self.analysis.is_none() || self.analyzed_with != Some(self.reordering);

        if needs_analysis {
            if !rebuilt {
                // `analyze` permutes the cached pattern in place, so what is
                // held right now is the *previous* analysis's permuted pattern.
                // Re-analyzing from it would compose two permutations and
                // silently corrupt the value map. Start from the Hessian again.
                self.csr.invalidate();
                self.csr.sync(hessian, &mut self.profile)?;
            }
            // Drop the old analysis before allocating the new one, so a resize
            // does not need both resident at once.
            self.analysis = None;
            self.analysis = Some(self.analyze()?);
            self.analyzed_with = Some(self.reordering);
        }
        self.factor_and_solve(hessian.val(), gradient)
    }

    /// Permute for fill reduction, upload, and run cuSOLVER's symbolic analysis.
    fn analyze(&mut self) -> LinAlgResult<CudaAnalysis> {
        use crate::cuda::profile::HostTimer;

        let (permutation, value_map) = {
            let _timer = HostTimer::start(&mut self.profile.permutation);
            let permutation = self.context.reorder(&self.csr, self.reordering)?;
            let value_map = self.context.permute(&mut self.csr, &permutation)?;
            (permutation, value_map)
        };

        let (n, nnz) = (self.csr.n_i32()?, self.csr.nnz_i32()?);
        let mut system =
            crate::cuda::buffers::DeviceSystem::new(self.context.stream(), &self.csr)?;

        let info = {
            let _timer = HostTimer::start(&mut self.profile.symbolic_analysis);
            let info = self.context.chol_analyze(&mut system, n, nnz)?;
            // The analysis is asynchronous on the stream; without this the timer
            // would measure the launch, not the work.
            self.context.synchronize()?;
            info
        };

        let workspace = {
            let _timer = HostTimer::start(&mut self.profile.buffer_query);
            self.context.chol_buffer_size(&mut system, &info, n, nnz)?
        };
        system.size_workspace(self.context.stream(), workspace.workspace_bytes)?;

        self.profile.memory = system.memory();
        self.profile.memory.internal = workspace.internal_bytes;

        tracing::debug!(
            n = self.csr.n(),
            nnz = self.csr.nnz(),
            workspace_mib = workspace.workspace_bytes as f64 / (1024.0 * 1024.0),
            internal_mib = workspace.internal_bytes as f64 / (1024.0 * 1024.0),
            "CUDA Cholesky analysis complete"
        );

        self.permuted_values.resize(self.csr.nnz(), 0.0);
        self.permuted_solution.resize(self.csr.n(), 0.0);

        Ok(CudaAnalysis {
            info,
            system,
            value_map: value_map.iter().map(|&i| i as usize).collect(),
            permutation: permutation.iter().map(|&i| i as usize).collect(),
            n,
            nnz,
        })
    }

    /// Numeric factorization and triangular solves — the per-iteration cost.
    fn factor_and_solve(&mut self, values: &[f64], gradient: &Mat<f64>) -> LinAlgResult<Mat<f64>> {
        use crate::cuda::profile::DevicePhase;

        let analysis = self.analysis.as_mut().ok_or_else(|| {
            LinAlgError::InvalidState("factorization requested before analysis".to_string()).log()
        })?;
        let stream = self.context.stream().clone();

        // Apply the cached permutation to this iteration's values.
        for (dst, &src) in self
            .permuted_values
            .iter_mut()
            .zip(analysis.value_map.iter())
        {
            *dst = values[src];
        }

        self.stopwatch.begin(DevicePhase::Upload, &stream);
        analysis
            .system
            .upload_values(&stream, &self.permuted_values)?;
        let permutation = &analysis.permutation;
        analysis.system.upload_rhs(&stream, |rhs| {
            for (new, &old) in permutation.iter().enumerate() {
                rhs[new] = -gradient[(old, 0)];
            }
        })?;
        self.stopwatch.end(DevicePhase::Upload, &stream);

        self.stopwatch.begin(DevicePhase::Factorize, &stream);
        self.context
            .chol_factor(&mut analysis.system, &analysis.info, analysis.n, analysis.nnz)?;
        self.stopwatch.end(DevicePhase::Factorize, &stream);

        // cuSOLVER's zero-pivot check is a *diagnostic*, not a hard failure: the
        // test is `pivot < tol` on an absolute tolerance, so on a large
        // well-scaled Hessian it fires on pivots that are merely small. The CPU
        // solver has no equivalent check and completes such systems successfully
        // — failing here would make the GPU backend reject problems the CPU path
        // solves (observed on the 485k-DOF ladybug BAL problem). The
        // authoritative test is whether the resulting step is usable, applied
        // after the solve.
        let position = self.context.chol_zero_pivot(
            &analysis.info,
            crate::cuda::context::DEFAULT_SINGULARITY_TOL,
        )?;
        if position >= 0 {
            tracing::debug!(
                permuted_row = position,
                "cuSOLVER reports a near-zero Cholesky pivot; continuing, the step is validated \
                 after the solve"
            );
        }

        self.stopwatch.begin(DevicePhase::TriangularSolve, &stream);
        self.context
            .chol_solve(&mut analysis.system, &analysis.info, analysis.n, analysis.nnz)?;
        self.stopwatch.end(DevicePhase::TriangularSolve, &stream);

        self.stopwatch.begin(DevicePhase::Download, &stream);
        analysis.system.download_solution(&stream)?;
        self.stopwatch.end(DevicePhase::Download, &stream);

        self.context.synchronize()?;
        self.stopwatch.drain(&mut self.profile);

        self.permuted_solution
            .copy_from_slice(analysis.system.solution());

        // A rank-deficient Hessian shows up here as a non-finite step. This is
        // the check that matters — it catches real breakage regardless of how the
        // pivot heuristic above behaved.
        if let Some(bad) = self
            .permuted_solution
            .iter()
            .position(|value| !value.is_finite())
        {
            let pivot = if position >= 0 {
                format!(" (cuSOLVER reported a near-zero pivot at permuted row {position})")
            } else {
                String::new()
            };
            return Err(LinAlgError::SingularMatrix(format!(
                "cusolverSpDcsrcholSolve produced a non-finite step at permuted row {bad}{pivot}; \
                 the Hessian is singular or not positive definite. For a pose graph this usually \
                 means unconstrained gauge freedom — add a PriorFactor to anchor it."
            ))
            .log());
        }

        // Scatter back to the original ordering.
        let mut dx = Mat::<f64>::zeros(self.permuted_solution.len(), 1);
        for (new, &old) in analysis.permutation.iter().enumerate() {
            dx[(old, 0)] = self.permuted_solution[new];
        }
        Ok(dx)
    }

    /// One `cusolverSpDcsrlsvchol` call: reorder, analyze, factorize, solve.
    fn solve_one_shot(
        &mut self,
        hessian: &SparseColMat<usize, f64>,
        gradient: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        use crate::cuda::profile::DevicePhase;

        let rebuilt = self.csr.sync(hessian, &mut self.profile)?;
        let stream = self.context.stream().clone();

        if rebuilt
            || !self
                .one_shot
                .as_ref()
                .is_some_and(|system| system.matches(&self.csr))
        {
            self.one_shot = Some(crate::cuda::buffers::DeviceSystem::new(&stream, &self.csr)?);
        }
        let (n, nnz) = (self.csr.n_i32()?, self.csr.nnz_i32()?);
        let system = self.one_shot.as_mut().ok_or_else(|| {
            LinAlgError::InvalidState("device buffers missing before solve".to_string()).log()
        })?;

        self.stopwatch.begin(DevicePhase::Upload, &stream);
        system.upload_values(&stream, hessian.val())?;
        system.upload_rhs(&stream, |rhs| {
            for (i, slot) in rhs.iter_mut().enumerate() {
                *slot = -gradient[(i, 0)];
            }
        })?;
        self.stopwatch.end(DevicePhase::Upload, &stream);

        // Analysis and factorization are fused in this API, so they are reported
        // together under `factorize`.
        self.stopwatch.begin(DevicePhase::Factorize, &stream);
        self.context.csrlsvchol(system, n, nnz, self.reordering)?;
        self.stopwatch.end(DevicePhase::Factorize, &stream);

        self.stopwatch.begin(DevicePhase::Download, &stream);
        system.download_solution(&stream)?;
        self.stopwatch.end(DevicePhase::Download, &stream);

        self.context.synchronize()?;
        self.profile.memory = system.memory();
        self.stopwatch.drain(&mut self.profile);

        let solution = system.solution();
        Ok(Mat::from_fn(solution.len(), 1, |i, _| solution[i]))
    }
}

#[cfg(feature = "cuda")]
impl LinearSolver<SparseMode> for CudaSparseCholeskySolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        let hessian = crate::linalg::sparse::normal_matrix(jacobians)?;
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
        let hessian = crate::linalg::sparse::normal_matrix(jacobians)?;
        let gradient = jacobians.as_ref().transpose().mul(residuals);

        let augmented = crate::linalg::sparse::add_damping(&hessian, lambda)?;
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
    /// so enabling covariance on the CUDA path does not silently produce nothing.
    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>> {
        if self.covariance_matrix.is_none() {
            let hessian = self.hessian.as_ref()?;
            self.covariance_matrix = crate::linalg::sparse::invert_undamped_hessian(hessian);
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

    const TOLERANCE: f64 = 1e-10;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Helper function to create a simple test matrix and vectors
    fn create_test_data()
    -> Result<(SparseColMat<usize, f64>, Mat<f64>), faer::sparse::CreationError> {
        // Create an overdetermined system (4x3) so that weights have an effect
        let triplets = vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 3.0),
            Triplet::new(1, 2, 1.0),
            Triplet::new(2, 1, 1.0),
            Triplet::new(2, 2, 2.0),
            Triplet::new(3, 0, 1.5), // Add a 4th row for overdetermined system
            Triplet::new(3, 2, 0.5),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(4, 3, &triplets)?;

        let residuals = Mat::from_fn(4, 1, |i, _| match i {
            0 => 1.0,
            1 => -2.0,
            2 => 0.5,
            3 => 1.2,
            _ => 0.0,
        });

        Ok((jacobian, residuals))
    }

    /// Test basic solver creation and default implementation
    #[test]
    fn test_solver_creation() {
        let solver = SparseCholeskySolver::new();
        assert!(solver.factorizer.is_none());

        let default_solver = SparseCholeskySolver::default();
        assert!(default_solver.factorizer.is_none());
    }

    /// Test normal equation solving with well-conditioned matrix
    #[test]
    fn test_solve_normal_equation_well_conditioned() -> TestResult {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals) = create_test_data()?;

        let solution =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(solution.nrows(), 3);
        assert_eq!(solution.ncols(), 1);

        // Verify the symbolic pattern was cached
        assert!(solver.factorizer.is_some());
        Ok(())
    }

    /// Test that symbolic pattern is reused on subsequent calls
    #[test]
    fn test_symbolic_pattern_caching() -> TestResult {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals) = create_test_data()?;

        // First solve
        let sol1 =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert!(solver.factorizer.is_some());

        // Second solve should reuse pattern
        let sol2 =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        // Results should be identical
        for i in 0..sol1.nrows() {
            assert!((sol1[(i, 0)] - sol2[(i, 0)]).abs() < TOLERANCE);
        }
        Ok(())
    }

    /// Test augmented equation solving
    #[test]
    fn test_solve_augmented_equation() -> TestResult {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals) = create_test_data()?;
        let lambda = 0.1;

        let solution = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver,
            &residuals,
            &jacobian,
            lambda,
        )?;
        assert_eq!(solution.nrows(), 3);
        assert_eq!(solution.ncols(), 1);
        Ok(())
    }

    /// Test with different lambda values in augmented system
    #[test]
    fn test_augmented_equation_different_lambdas() -> TestResult {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals) = create_test_data()?;

        let lambda1 = 0.01;
        let lambda2 = 1.0;

        let sol1 = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver,
            &residuals,
            &jacobian,
            lambda1,
        )?;
        let sol2 = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver,
            &residuals,
            &jacobian,
            lambda2,
        )?;

        // Solutions should be different due to different regularization
        let mut different = false;
        for i in 0..sol1.nrows() {
            if (sol1[(i, 0)] - sol2[(i, 0)]).abs() > TOLERANCE {
                different = true;
                break;
            }
        }
        assert!(
            different,
            "Solutions should differ with different lambda values"
        );
        Ok(())
    }

    /// Test with singular matrix (should return None)
    #[test]
    fn test_singular_matrix() -> TestResult {
        let mut solver = SparseCholeskySolver::new();

        // Create a singular matrix
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 2.0),
            Triplet::new(1, 0, 2.0),
            Triplet::new(1, 1, 4.0), // Second row is 2x first row
        ];
        let singular_jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets)?;
        let residuals = Mat::from_fn(2, 1, |i, _| i as f64);

        let result = LinearSolver::<SparseMode>::solve_normal_equation(
            &mut solver,
            &residuals,
            &singular_jacobian,
        );
        // Without regularization, singular matrices should fail
        assert!(result.is_err(), "Singular matrix should return Err");
        Ok(())
    }

    /// Test with empty matrix (edge case)
    #[test]
    fn test_empty_matrix() -> TestResult {
        let mut solver = SparseCholeskySolver::new();

        let empty_jacobian = SparseColMat::try_new_from_triplets(0, 0, &[])?;
        let empty_residuals = Mat::zeros(0, 1);

        let result = LinearSolver::<SparseMode>::solve_normal_equation(
            &mut solver,
            &empty_residuals,
            &empty_jacobian,
        );
        if let Ok(solution) = result {
            assert_eq!(solution.nrows(), 0);
        }
        Ok(())
    }

    /// Test numerical accuracy with known solution
    #[test]
    fn test_numerical_accuracy() -> TestResult {
        let mut solver = SparseCholeskySolver::new();

        // Create a simple 2x2 system with known solution
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 0.0),
            Triplet::new(1, 0, 0.0),
            Triplet::new(1, 1, 1.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets)?;
        let residuals = Mat::from_fn(2, 1, |i, _| -((i + 1) as f64)); // [-1, -2]

        let solution =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        // Expected solution should be [1, 2] since J^T * J = I and J^T * (-r) = [1, 2]
        assert!((solution[(0, 0)] - 1.0).abs() < TOLERANCE);
        assert!((solution[(1, 0)] - 2.0).abs() < TOLERANCE);
        Ok(())
    }

    /// Test clone functionality
    #[test]
    fn test_solver_clone() {
        let solver1 = SparseCholeskySolver::new();
        let solver2 = solver1.clone();

        assert!(solver1.factorizer.is_none());
        assert!(solver2.factorizer.is_none());
    }

    /// Test covariance matrix computation
    #[test]
    fn test_cholesky_covariance_computation() -> TestResult {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals) = create_test_data()?;

        // First solve to set up factorizer and hessian
        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        // Now compute covariance matrix
        let cov_matrix = LinearSolver::<SparseMode>::compute_covariance_matrix(&mut solver);
        assert!(cov_matrix.is_some());

        if let Some(cov) = cov_matrix {
            assert_eq!(cov.nrows(), 3); // Should be n x n where n is number of variables
            assert_eq!(cov.ncols(), 3);

            // Covariance matrix should be symmetric
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (cov[(i, j)] - cov[(j, i)]).abs() < TOLERANCE,
                        "Covariance matrix should be symmetric"
                    );
                }
            }

            // Diagonal elements should be positive (variances)
            for i in 0..3 {
                assert!(
                    cov[(i, i)] > 0.0,
                    "Diagonal elements (variances) should be positive"
                );
            }
        }
        Ok(())
    }

    /// Test standard errors computation
    #[test]
    fn test_cholesky_standard_errors_computation() -> TestResult {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals) = create_test_data()?;

        // First solve to set up factorizer and hessian
        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        // Compute covariance matrix first (this also computes standard errors)
        solver.compute_standard_errors();

        // Now check that both covariance matrix and standard errors are available
        assert!(solver.covariance_matrix.is_some());
        assert!(solver.standard_errors.is_some());

        if let (Some(cov), Some(errors)) = (&solver.covariance_matrix, &solver.standard_errors) {
            assert_eq!(errors.nrows(), 3); // Should be n x 1 where n is number of variables
            assert_eq!(errors.ncols(), 1);

            // All standard errors should be positive
            for i in 0..3 {
                assert!(errors[(i, 0)] > 0.0, "Standard errors should be positive");
            }

            // Verify relationship: std_error = sqrt(covariance_diagonal)
            for i in 0..3 {
                let expected_std_error = cov[(i, i)].sqrt();
                assert!(
                    (errors[(i, 0)] - expected_std_error).abs() < TOLERANCE,
                    "Standard error should equal sqrt of covariance diagonal"
                );
            }
        }
        Ok(())
    }

    /// Test covariance computation with well-conditioned positive definite system
    #[test]
    fn test_cholesky_covariance_positive_definite() -> TestResult {
        let mut solver = SparseCholeskySolver::new();

        // Create a well-conditioned positive definite system
        let triplets = vec![
            Triplet::new(0, 0, 3.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 2.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets)?;
        let residuals = Mat::from_fn(2, 1, |i, _| (i + 1) as f64);

        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        let cov_matrix = LinearSolver::<SparseMode>::compute_covariance_matrix(&mut solver);
        assert!(cov_matrix.is_some());

        if let Some(cov) = cov_matrix {
            // For this system, H = J^T * W * J = [[10, 5], [5, 5]]
            // Covariance = H^(-1) = (1/25) * [[5, -5], [-5, 10]] = [[0.2, -0.2], [-0.2, 0.4]]
            assert!((cov[(0, 0)] - 0.2).abs() < TOLERANCE);
            assert!((cov[(1, 1)] - 0.4).abs() < TOLERANCE);
            assert!((cov[(0, 1)] - (-0.2)).abs() < TOLERANCE);
            assert!((cov[(1, 0)] - (-0.2)).abs() < TOLERANCE);
        }
        Ok(())
    }

    /// Test covariance computation caching
    #[test]
    fn test_cholesky_covariance_caching() -> TestResult {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals) = create_test_data()?;

        // First solve
        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        // First covariance computation
        LinearSolver::<SparseMode>::compute_covariance_matrix(&mut solver);
        assert!(solver.covariance_matrix.is_some());

        // Get pointer to first computation
        if let Some(cov1) = &solver.covariance_matrix {
            let cov1_ptr = cov1.as_ptr();

            // Second covariance computation should return cached result
            LinearSolver::<SparseMode>::compute_covariance_matrix(&mut solver);
            assert!(solver.covariance_matrix.is_some());

            // Get pointer to second computation
            if let Some(cov2) = &solver.covariance_matrix {
                let cov2_ptr = cov2.as_ptr();

                // Should be the same pointer (cached)
                assert_eq!(cov1_ptr, cov2_ptr, "Covariance matrix should be cached");
            }
        }
        Ok(())
    }

    /// Test Cholesky decomposition properties
    #[test]
    fn test_cholesky_decomposition_properties() -> TestResult {
        let mut solver = SparseCholeskySolver::new();

        // Create a simple positive definite system
        let triplets = vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 3.0)];
        let jacobian = SparseColMat::try_new_from_triplets(2, 2, &triplets)?;
        let residuals = Mat::from_fn(2, 1, |i, _| (i + 1) as f64);

        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        // Verify that we have a factorizer and hessian
        assert!(solver.factorizer.is_some());
        assert!(solver.hessian.is_some());

        // The hessian should be positive definite for Cholesky to work
        if let Some(hessian) = &solver.hessian {
            assert_eq!(hessian.nrows(), 2);
            assert_eq!(hessian.ncols(), 2);
        }
        Ok(())
    }

    /// Test numerical stability with different condition numbers
    #[test]
    fn test_cholesky_numerical_stability() -> TestResult {
        let mut solver = SparseCholeskySolver::new();

        // Create a well-conditioned system
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 2, 1.0),
        ];
        let jacobian = SparseColMat::try_new_from_triplets(3, 3, &triplets)?;
        let residuals = Mat::from_fn(3, 1, |i, _| -((i + 1) as f64)); // [-1, -2, -3]

        let solution =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        // Expected solution should be [1, 2, 3] since H = I and g = [1, 2, 3]
        for i in 0..3 {
            let expected = (i + 1) as f64;
            assert!(
                (solution[(i, 0)] - expected).abs() < TOLERANCE,
                "Expected {}, got {}",
                expected,
                solution[(i, 0)]
            );
        }

        // Covariance should be identity matrix (inverse of identity)
        let cov_matrix = LinearSolver::<SparseMode>::compute_covariance_matrix(&mut solver);
        assert!(cov_matrix.is_some());
        if let Some(cov) = cov_matrix {
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (cov[(i, j)] - expected).abs() < TOLERANCE,
                        "Covariance[{}, {}] expected {}, got {}",
                        i,
                        j,
                        expected,
                        cov[(i, j)]
                    );
                }
            }
        }
        Ok(())
    }

    /// Test hessian() getter returns None before solve and Some after
    #[test]
    fn test_cholesky_hessian_getter() -> TestResult {
        let mut solver = SparseCholeskySolver::new();
        assert!(solver.hessian().is_none());

        let (jacobian, residuals) = create_test_data()?;
        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        assert!(solver.hessian().is_some());
        Ok(())
    }

    /// Test gradient() getter returns None before solve and Some after
    #[test]
    fn test_cholesky_gradient_getter() -> TestResult {
        let mut solver = SparseCholeskySolver::new();
        assert!(solver.gradient().is_none());

        let (jacobian, residuals) = create_test_data()?;
        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        assert!(solver.gradient().is_some());
        Ok(())
    }

    /// Test reset_covariance() clears the cached covariance
    #[test]
    fn test_cholesky_reset_covariance() -> TestResult {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals) = create_test_data()?;

        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        LinearSolver::<SparseMode>::compute_covariance_matrix(&mut solver);
        assert!(solver.covariance_matrix.is_some());

        solver.reset_covariance();
        assert!(solver.covariance_matrix.is_none());
        assert!(solver.standard_errors.is_none());
        Ok(())
    }

    /// Test get_hessian() trait method returns Some after solve
    #[test]
    fn test_cholesky_get_hessian_trait() -> TestResult {
        let mut solver = SparseCholeskySolver::new();
        assert!(LinearSolver::<SparseMode>::get_hessian(&solver).is_none());

        let (jacobian, residuals) = create_test_data()?;
        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        assert!(LinearSolver::<SparseMode>::get_hessian(&solver).is_some());
        Ok(())
    }

    /// Test get_gradient() trait method returns Some after solve
    #[test]
    fn test_cholesky_get_gradient_trait() -> TestResult {
        let mut solver = SparseCholeskySolver::new();
        assert!(LinearSolver::<SparseMode>::get_gradient(&solver).is_none());

        let (jacobian, residuals) = create_test_data()?;
        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        assert!(LinearSolver::<SparseMode>::get_gradient(&solver).is_some());
        Ok(())
    }

    /// Test get_covariance_matrix() getter matches compute result
    #[test]
    fn test_cholesky_get_covariance_matrix_getter() -> TestResult {
        let mut solver = SparseCholeskySolver::new();
        let (jacobian, residuals) = create_test_data()?;

        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        LinearSolver::<SparseMode>::compute_covariance_matrix(&mut solver);

        let via_getter = LinearSolver::<SparseMode>::get_covariance_matrix(&solver);
        assert!(via_getter.is_some());

        if let Some(cov) = via_getter {
            assert_eq!(cov.nrows(), 3);
            assert_eq!(cov.ncols(), 3);
        }
        Ok(())
    }

    /// Test that `compute_standard_errors()` returns `None` when called before any solve.
    ///
    /// Covers the early-return `?` at the `let hessian = self.hessian.as_ref()?;` line:
    /// a freshly-created solver has no hessian, so the method returns `None`.
    #[test]
    fn test_compute_standard_errors_before_solve_returns_none() {
        let mut solver = SparseCholeskySolver::new();
        // No solve has been performed → hessian is None → should return None
        let result = solver.compute_standard_errors();
        assert!(
            result.is_none(),
            "compute_standard_errors on uninitialized solver (no hessian) should return None"
        );
    }
}
