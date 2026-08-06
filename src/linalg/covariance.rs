//! Parameter covariance estimation.
//!
//! Covariance is a property of the **problem at a point**, not a byproduct of the
//! solver's last linear system. This module re-linearizes the problem at a given
//! set of variable values and inverts a clean Gauss-Newton Hessian:
//!
//! ```text
//! H = JᵀJ
//! Σ = H⁻¹
//! ```
//!
//! Crucially, neither Levenberg-Marquardt damping `λ` nor Jacobi column scaling
//! `D` appears anywhere in that computation. Both are internal devices that make
//! the *step* well-behaved; neither carries information about the estimate's
//! uncertainty, and a user flipping [`with_jacobi_scaling`] or `damping` must see
//! identical covariances.
//!
//! [`with_jacobi_scaling`]: crate::optimizer::levenberg_marquardt::LevenbergMarquardtConfig::with_jacobi_scaling
//!
//! # Robust loss functions
//!
//! When a residual block carries a loss function, `J` here is the Triggs-corrected
//! Jacobian produced during assembly, so `H = J̃ᵀJ̃` is the Gauss-Newton
//! approximation of the Hessian of the **robust** objective. This matches Ceres
//! Solver, and gives the Laplace approximation of the posterior under the robust
//! cost. It is deliberately *not* the Huber/White sandwich estimator `A⁻¹BA⁻¹`
//! used for robust standard errors in the statistics literature.
//!
//! # Units
//!
//! `H⁻¹` is a covariance only if the residuals are already whitened by their
//! measurement information matrix (the usual convention in SLAM and bundle
//! adjustment). For unweighted least squares, set
//! [`CovarianceOptions::apply_variance_scaling`] to estimate the noise scale from
//! the fit residuals.
//!
//! # Example
//!
//! ```no_run
//! use apex_solver::linalg::covariance::{Covariance, CovarianceOptions};
//! # use apex_solver::core::problem::Problem;
//! # fn example(problem: &Problem, result_parameters: &slotmap::SlotMap<apex_solver::core::VarKey, Box<dyn apex_solver::core::variable::ManifoldVariable>>) -> Result<(), Box<dyn std::error::Error>> {
//! let covariance = Covariance::compute(
//!     CovarianceOptions::default(),
//!     problem,
//!     result_parameters,
//! )?;
//! # Ok(())
//! # }
//! ```

use faer::{
    Mat, MatRef, Side,
    linalg::solvers::Solve,
    sparse::linalg::solvers::{Llt, SymbolicLlt},
};
use slotmap::{SecondaryMap, SlotMap};
use std::ops::Mul;
use thiserror::Error;
use tracing::warn;

use crate::core::{VarKey, problem::Problem, variable::ManifoldVariable};
use crate::linearizer::LinearizerError;

// ============================================================================
// Options
// ============================================================================

/// How to invert the Gauss-Newton Hessian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CovarianceAlgorithm {
    /// Sparse Cholesky factorization. Fast and memory-efficient, but requires
    /// `H` to be positive definite — i.e. the problem must be fully constrained.
    ///
    /// A rank-deficient `H` (unfixed gauge freedom, or a variable no residual
    /// observes) produces [`CovarianceError::RankDeficient`].
    #[default]
    SparseCholesky,

    /// Dense SVD, yielding the Moore-Penrose pseudo-inverse.
    ///
    /// Handles rank-deficient `H` — the pseudo-inverse reports zero variance
    /// along the null space, which is the correct answer for an intentionally
    /// gauge-free problem. Costs `O(n³)` time and `O(n²)` memory, so this is only
    /// viable for small problems.
    DenseSvd,
}

/// Configuration for [`Covariance::compute`].
#[derive(Debug, Clone)]
pub struct CovarianceOptions {
    /// Factorization strategy. See [`CovarianceAlgorithm`].
    pub algorithm: CovarianceAlgorithm,

    /// Multiply the result by the estimated noise variance
    /// `σ̂² = 2·cost / (m − n)`, where `m` is the total residual dimension and
    /// `n` the total tangent-space dimension.
    ///
    /// Leave this `false` (the default) when residuals are already whitened by
    /// their measurement information matrix — then `H⁻¹` is directly the
    /// covariance. Set it `true` for unweighted least squares, where the noise
    /// scale has to be estimated from the fit.
    pub apply_variance_scaling: bool,

    /// Singular values below `σ_max · min_reciprocal_condition_number` are treated
    /// as zero by [`CovarianceAlgorithm::DenseSvd`]. Ignored by `SparseCholesky`.
    pub min_reciprocal_condition_number: f64,
}

impl Default for CovarianceOptions {
    fn default() -> Self {
        Self {
            algorithm: CovarianceAlgorithm::default(),
            apply_variance_scaling: false,
            min_reciprocal_condition_number: 1e-14,
        }
    }
}

impl CovarianceOptions {
    /// Create options with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select the factorization strategy.
    pub fn with_algorithm(mut self, algorithm: CovarianceAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Enable or disable estimating the noise scale from the fit residuals.
    pub fn with_variance_scaling(mut self, apply_variance_scaling: bool) -> Self {
        self.apply_variance_scaling = apply_variance_scaling;
        self
    }

    /// Set the singular-value cutoff used by [`CovarianceAlgorithm::DenseSvd`].
    pub fn with_min_reciprocal_condition_number(mut self, value: f64) -> Self {
        self.min_reciprocal_condition_number = value;
        self
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Why covariance estimation failed.
///
/// Covariance never fails silently: a caller that asked for it either gets a
/// matrix or a reason.
#[derive(Debug, Error)]
pub enum CovarianceError {
    /// `H = JᵀJ` is singular or not positive definite.
    #[error(
        "Hessian is singular or not positive definite ({context}). This usually means \
         unconstrained gauge freedom (e.g. a pose graph with no prior anchoring the first \
         pose) or a variable that no residual observes. Add a PriorFactor to fix the gauge, \
         or use CovarianceAlgorithm::DenseSvd to obtain the pseudo-inverse."
    )]
    RankDeficient {
        /// What the chosen algorithm was able to determine about the deficiency.
        context: String,
    },

    /// The problem could not be linearized at the given variables.
    #[error("Failed to linearize the problem for covariance estimation: {0}")]
    Linearization(#[from] LinearizerError),

    /// Covariance is undefined without variables or residuals.
    #[error("Covariance requires at least one variable and one residual block")]
    EmptyProblem,

    /// A matrix operation during Hessian assembly failed.
    #[error("Failed to assemble the Hessian for covariance estimation: {0}")]
    Assembly(String),

    /// The dense SVD itself failed to converge.
    #[error("SVD failed during covariance estimation: {0}")]
    SvdFailed(String),
}

/// Result type for covariance estimation.
pub type CovarianceResult<T> = Result<T, CovarianceError>;

// ============================================================================
// Covariance
// ============================================================================

/// Parameter covariance for a problem evaluated at a set of variable values.
///
/// Blocks are in **tangent space**: a variable with `dof() == d` has a `d × d`
/// covariance block, regardless of its ambient parameterization. For an `SE3`
/// pose this is `6 × 6`, even though the variable stores 7 numbers.
#[derive(Debug, Clone)]
pub struct Covariance {
    matrix: Mat<f64>,
    index_map: SecondaryMap<VarKey, usize>,
    dofs: SecondaryMap<VarKey, usize>,
}

impl Covariance {
    /// Compute the covariance by re-linearizing `problem` at `variables`.
    ///
    /// `variables` is normally `SolverResult::parameters`, i.e. the solution the
    /// optimizer returned.
    pub fn compute(
        options: CovarianceOptions,
        problem: &Problem,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    ) -> CovarianceResult<Self> {
        if variables.is_empty() || problem.num_residual_blocks() == 0 {
            return Err(CovarianceError::EmptyProblem);
        }

        let (index_map, _sorted_vars, total_dof) =
            crate::optimizer::build_variable_index_map(variables);

        if variables
            .values()
            .any(|v| !v.get_fixed_indices().is_empty())
        {
            warn!(
                "Covariance requested for a problem with fixed variable indices. Fixed degrees \
                 of freedom are currently included in H, so the reported blocks are marginals \
                 rather than conditionals and over-state uncertainty on the free parameters. \
                 See cov_issues/07-fixed-variables-and-covariance.md"
            );
        }

        // Re-linearize at the given point. Note this deliberately ignores any
        // `use_jacobi_scaling` setting: the covariance must not depend on the
        // preconditioning the optimizer happened to use.
        let symbolic = crate::linearizer::cpu::sparse::build_symbolic_structure(
            problem, variables, &index_map, total_dof,
        )?;
        let (residuals, jacobian) = crate::linearizer::cpu::sparse::assemble_sparse(
            problem, variables, &index_map, &symbolic,
        )?;

        // H = JᵀJ. No damping term, no column scaling.
        let hessian = jacobian
            .as_ref()
            .transpose()
            .to_col_major()
            .map_err(|e| {
                CovarianceError::Assembly(format!(
                    "failed to convert transposed Jacobian to column-major: {e:?}"
                ))
            })?
            .mul(jacobian.as_ref());

        let mut matrix = match options.algorithm {
            CovarianceAlgorithm::SparseCholesky => invert_via_cholesky(&hessian, total_dof)?,
            CovarianceAlgorithm::DenseSvd => {
                invert_via_svd(&hessian, total_dof, options.min_reciprocal_condition_number)?
            }
        };

        if options.apply_variance_scaling {
            if let Some(sigma_squared) = estimate_variance(problem, &residuals, total_dof) {
                matrix *= faer::Scale(sigma_squared);
            }
        }

        let dofs = variables.iter().map(|(k, v)| (k, v.dof())).collect();

        Ok(Self {
            matrix,
            index_map,
            dofs,
        })
    }

    /// Marginal covariance of one variable: a `dof × dof` block in tangent space.
    pub fn block(&self, key: VarKey) -> Option<MatRef<'_, f64>> {
        self.block_pair(key, key)
    }

    /// Cross-covariance between two variables: a `dof_a × dof_b` block.
    pub fn block_pair(&self, a: VarKey, b: VarKey) -> Option<MatRef<'_, f64>> {
        let (&row, &col) = (self.index_map.get(a)?, self.index_map.get(b)?);
        let (rows, cols) = (*self.dofs.get(a)?, *self.dofs.get(b)?);
        Some(self.matrix.as_ref().submatrix(row, col, rows, cols))
    }

    /// Every variable's marginal block, keyed by variable.
    pub fn per_variable(&self) -> SecondaryMap<VarKey, Mat<f64>> {
        self.index_map
            .keys()
            .filter_map(|key| Some((key, self.block(key)?.to_owned())))
            .collect()
    }

    /// The full covariance matrix, `total_dof × total_dof`.
    pub fn full(&self) -> MatRef<'_, f64> {
        self.matrix.as_ref()
    }

    /// Column offset of each variable within [`full`](Self::full).
    pub fn index_map(&self) -> &SecondaryMap<VarKey, usize> {
        &self.index_map
    }
}

// ============================================================================
// Inversion strategies
// ============================================================================

/// Invert `H` via a fresh sparse Cholesky factorization.
///
/// The factorization is built here rather than reused from the optimizer, which
/// is the entire point: the optimizer's cached factorizer is of `H + λI`.
fn invert_via_cholesky(
    hessian: &faer::sparse::SparseColMat<usize, f64>,
    n: usize,
) -> CovarianceResult<Mat<f64>> {
    let symbolic = SymbolicLlt::try_new(hessian.symbolic(), Side::Lower).map_err(|e| {
        CovarianceError::RankDeficient {
            context: format!("symbolic Cholesky analysis failed: {e:?}"),
        }
    })?;
    let llt = Llt::try_new_with_symbolic(symbolic, hessian.as_ref(), Side::Lower).map_err(|e| {
        CovarianceError::RankDeficient {
            context: format!("numeric Cholesky factorization failed: {e:?}"),
        }
    })?;

    Ok(llt.solve(Mat::<f64>::identity(n, n)))
}

/// Invert `H` via a dense SVD, yielding the Moore-Penrose pseudo-inverse.
///
/// Singular values below `σ_max · rcond` are treated as exactly zero, so the
/// result reports zero variance along the null space of `H` — the correct answer
/// for a problem with intentional gauge freedom.
fn invert_via_svd(
    hessian: &faer::sparse::SparseColMat<usize, f64>,
    n: usize,
    rcond: f64,
) -> CovarianceResult<Mat<f64>> {
    let dense = Mat::from_fn(n, n, |i, j| hessian.get(i, j).copied().unwrap_or(0.0));

    let svd = dense
        .as_ref()
        .svd()
        .map_err(|e| CovarianceError::SvdFailed(format!("{e:?}")))?;

    let singular = svd.S();
    let sigma_max = (0..n).map(|i| singular[i]).fold(0.0_f64, f64::max);
    if sigma_max <= 0.0 {
        return Err(CovarianceError::RankDeficient {
            context: "Hessian is entirely zero (rank 0)".to_string(),
        });
    }

    let threshold = sigma_max * rcond;
    let inverse_singular: Vec<f64> = (0..n)
        .map(|i| {
            let s = singular[i];
            if s > threshold { 1.0 / s } else { 0.0 }
        })
        .collect();
    let rank = inverse_singular.iter().filter(|&&x| x != 0.0).count();

    if rank < n {
        warn!(
            "Hessian is rank deficient: numeric rank {rank} of {n}. Reporting the \
             Moore-Penrose pseudo-inverse, which assigns zero variance along the {} \
             null-space direction(s). This usually indicates unconstrained gauge freedom.",
            n - rank
        );
    }

    // pinv(H) = V · diag(1/σ, thresholded) · Uᵀ
    let (u, v) = (svd.U(), svd.V());
    let mut scaled_v = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            scaled_v[(i, j)] = v[(i, j)] * inverse_singular[j];
        }
    }

    Ok(scaled_v * u.transpose())
}

/// Estimate the noise variance `σ̂² = 2·cost / (m − n)` from the fit residuals.
///
/// Returns `None` (with a warning) when there are no residual degrees of freedom
/// left to estimate a scale from.
fn estimate_variance(problem: &Problem, residuals: &Mat<f64>, n: usize) -> Option<f64> {
    let m = problem.total_residual_dimension();
    if m <= n {
        warn!(
            "Cannot estimate noise variance: {m} residuals for {n} parameters leaves no \
             residual degrees of freedom. Returning the unscaled H^-1."
        );
        return None;
    }
    let cost = 0.5 * residuals.squared_norm_l2();
    Some(2.0 * cost / (m - n) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ManifoldType;
    use crate::factors::Factor;
    use crate::linalg::JacobianMode;
    use faer::prelude::ReborrowMut;
    use nalgebra::dvector;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// `r = a·x − b`, so `J = [a]` and `H = a²` for a single scalar variable.
    struct ScaledFactor {
        a: f64,
        b: f64,
    }

    impl Factor for ScaledFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = self.a * params[0][0] - self.b;
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = self.a;
            }
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 1)
        }
    }

    /// `r = x1 − x2`, an SE2-free relative constraint on two scalars.
    /// Two of these leave the pair's common offset unobserved — a 1-D gauge freedom.
    struct DifferenceFactor;

    impl Factor for DifferenceFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = params[0][0] - params[1][0];
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = 1.0;
                *jac.rb_mut().get_mut(0, 1) = -1.0;
            }
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 2)
        }
    }

    /// Two independent scalars observed with different precisions:
    /// `H = diag(4, 100)`, so `Σ = diag(0.25, 0.01)`.
    fn two_variable_problem() -> (Problem, VarKey, VarKey) {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let x = problem.add_variable(ManifoldType::RN, dvector![0.0]);
        let y = problem.add_variable(ManifoldType::RN, dvector![0.0]);
        problem.add_residual_block(&[x], Box::new(ScaledFactor { a: 2.0, b: 1.0 }), None);
        problem.add_residual_block(&[y], Box::new(ScaledFactor { a: 10.0, b: 1.0 }), None);
        (problem, x, y)
    }

    #[test]
    fn computes_analytic_inverse_hessian() -> TestResult {
        let (problem, x, y) = two_variable_problem();
        let cov = Covariance::compute(
            CovarianceOptions::default(),
            &problem,
            &problem.variables.clone(),
        )?;

        let sigma_x = cov.block(x).ok_or("missing block for x")?;
        let sigma_y = cov.block(y).ok_or("missing block for y")?;

        assert!(
            (sigma_x[(0, 0)] - 0.25).abs() < 1e-12,
            "expected 1/a² = 0.25, got {}",
            sigma_x[(0, 0)]
        );
        assert!(
            (sigma_y[(0, 0)] - 0.01).abs() < 1e-12,
            "expected 1/a² = 0.01, got {}",
            sigma_y[(0, 0)]
        );
        Ok(())
    }

    #[test]
    fn independent_variables_have_zero_cross_covariance() -> TestResult {
        let (problem, x, y) = two_variable_problem();
        let cov = Covariance::compute(
            CovarianceOptions::default(),
            &problem,
            &problem.variables.clone(),
        )?;

        let cross = cov.block_pair(x, y).ok_or("missing cross block")?;
        assert!(cross[(0, 0)].abs() < 1e-12, "got {}", cross[(0, 0)]);
        Ok(())
    }

    #[test]
    fn per_variable_matches_individual_blocks() -> TestResult {
        let (problem, x, y) = two_variable_problem();
        let cov = Covariance::compute(
            CovarianceOptions::default(),
            &problem,
            &problem.variables.clone(),
        )?;

        let per_var = cov.per_variable();
        assert_eq!(per_var.len(), 2);
        for key in [x, y] {
            let expected = cov.block(key).ok_or("missing block")?;
            let actual = per_var.get(key).ok_or("missing per-variable entry")?;
            assert!((actual[(0, 0)] - expected[(0, 0)]).abs() < 1e-15);
        }
        Ok(())
    }

    /// A gauge-free problem must fail loudly under Cholesky rather than silently
    /// returning whatever damping happened to make invertible. See issue #38.
    #[test]
    fn rank_deficient_problem_reports_error_under_cholesky() {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let x = problem.add_variable(ManifoldType::RN, dvector![0.0]);
        let y = problem.add_variable(ManifoldType::RN, dvector![0.0]);
        problem.add_residual_block(&[x, y], Box::new(DifferenceFactor), None);

        let variables = problem.variables.clone();
        match Covariance::compute(CovarianceOptions::default(), &problem, &variables) {
            Ok(_) => panic!("rank-deficient H must not silently succeed"),
            Err(err) => {
                assert!(
                    matches!(err, CovarianceError::RankDeficient { .. }),
                    "expected RankDeficient, got {err:?}"
                );
                let message = err.to_string();
                assert!(
                    message.contains("PriorFactor") && message.contains("DenseSvd"),
                    "error must tell the user how to proceed, got: {message}"
                );
            }
        }
    }

    /// The same gauge-free problem succeeds under `DenseSvd`, which reports the
    /// pseudo-inverse: zero variance along the unobserved common-offset direction.
    #[test]
    fn rank_deficient_problem_succeeds_under_dense_svd() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let x = problem.add_variable(ManifoldType::RN, dvector![0.0]);
        let y = problem.add_variable(ManifoldType::RN, dvector![0.0]);
        problem.add_residual_block(&[x, y], Box::new(DifferenceFactor), None);

        let variables = problem.variables.clone();
        let cov = Covariance::compute(
            CovarianceOptions::new().with_algorithm(CovarianceAlgorithm::DenseSvd),
            &problem,
            &variables,
        )?;

        // H = [[1, -1], [-1, 1]] has eigenvalues {2, 0}. Its pseudo-inverse is
        // [[0.25, -0.25], [-0.25, 0.25]].
        let full = cov.full();
        for (i, j, expected) in [(0, 0, 0.25), (0, 1, -0.25), (1, 0, -0.25), (1, 1, 0.25)] {
            assert!(
                (full[(i, j)] - expected).abs() < 1e-10,
                "pinv[{i},{j}] = {} expected {expected}",
                full[(i, j)]
            );
        }

        // The null-space direction (1, 1) must carry zero variance.
        let ones = Mat::from_fn(2, 1, |_, _| 1.0);
        let quadratic = (ones.transpose() * full * &ones)[(0, 0)];
        assert!(quadratic.abs() < 1e-10, "got {quadratic}");
        Ok(())
    }

    #[test]
    fn empty_problem_is_rejected() {
        let problem = Problem::new(JacobianMode::Sparse);
        let variables = problem.variables.clone();
        match Covariance::compute(CovarianceOptions::default(), &problem, &variables) {
            Ok(_) => panic!("empty problem must be rejected"),
            Err(err) => assert!(matches!(err, CovarianceError::EmptyProblem), "got {err:?}"),
        }
    }

    /// With `m == n` there are no residual degrees of freedom, so variance
    /// scaling is skipped rather than dividing by zero.
    #[test]
    fn variance_scaling_skipped_without_residual_dof() -> TestResult {
        let (problem, x, _) = two_variable_problem();
        let variables = problem.variables.clone();

        let unscaled = Covariance::compute(CovarianceOptions::default(), &problem, &variables)?;
        let scaled = Covariance::compute(
            CovarianceOptions::new().with_variance_scaling(true),
            &problem,
            &variables,
        )?;

        let a = unscaled.block(x).ok_or("missing block")?[(0, 0)];
        let b = scaled.block(x).ok_or("missing block")?[(0, 0)];
        assert!((a - b).abs() < 1e-15, "{a} vs {b}");
        Ok(())
    }

    /// With `m > n`, `Σ` is scaled by `σ̂² = 2·cost / (m − n)`.
    #[test]
    fn variance_scaling_applies_estimated_noise() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let x = problem.add_variable(ManifoldType::RN, dvector![0.0]);
        // Two inconsistent observations of the same scalar: H = 2, and at x = 0
        // the residuals are (-1, -3), so cost = 0.5·10 = 5 and σ̂² = 2·5/(2−1) = 10.
        problem.add_residual_block(&[x], Box::new(ScaledFactor { a: 1.0, b: 1.0 }), None);
        problem.add_residual_block(&[x], Box::new(ScaledFactor { a: 1.0, b: 3.0 }), None);

        let variables = problem.variables.clone();
        let cov = Covariance::compute(
            CovarianceOptions::new().with_variance_scaling(true),
            &problem,
            &variables,
        )?;

        let value = cov.block(x).ok_or("missing block")?[(0, 0)];
        let expected = 10.0 * 0.5; // σ̂² · H⁻¹
        assert!(
            (value - expected).abs() < 1e-12,
            "expected {expected}, got {value}"
        );
        Ok(())
    }
}
