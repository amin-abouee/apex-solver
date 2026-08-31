//! Levenberg-Marquardt algorithm implementation.
//!
//! The Levenberg-Marquardt (LM) method is a robust and widely-used algorithm for solving
//! nonlinear least squares problems of the form:
//!
//! ```text
//! min f(x) = ½||r(x)||² = ½Σᵢ rᵢ(x)²
//! ```
//!
//! where `r: ℝⁿ → ℝᵐ` is the residual vector function.
//!
//! # Algorithm Overview
//!
//! The Levenberg-Marquardt method solves the damped normal equations at each iteration:
//!
//! ```text
//! (J^T·J + λI)·h = -J^T·r
//! ```
//!
//! where:
//! - `J` is the Jacobian matrix (m × n)
//! - `r` is the residual vector (m × 1)
//! - `h` is the step vector (n × 1)
//! - `λ` is the adaptive damping parameter (scalar)
//! - `I` is the identity matrix (or diagonal scaling matrix)
//!
//! ## Damping Parameter Strategy
//!
//! The damping parameter λ adapts based on step quality:
//!
//! - **λ → 0** (small damping): Behaves like Gauss-Newton with fast quadratic convergence
//! - **λ → ∞** (large damping): Behaves like gradient descent with guaranteed descent direction
//!
//! This interpolation between Newton and gradient descent provides excellent robustness
//! while maintaining fast convergence near the solution.
//!
//! ## Step Acceptance and Damping Update
//!
//! The algorithm evaluates each proposed step using the gain ratio:
//!
//! ```text
//! ρ = (actual reduction) / (predicted reduction)
//!   = [f(xₖ) - f(xₖ + h)] / [f(xₖ) - L(h)]
//! ```
//!
//! where `L(h) = f(xₖ) + h^T·g + ½h^T·H·h` is the local quadratic model.
//!
//! **Step acceptance:**
//! - If `ρ > 0`: Accept step (cost decreased), decrease λ to trust the model more
//! - If `ρ ≤ 0`: Reject step (cost increased), increase λ to be more conservative
//!
//! **Damping update** (Nielsen's formula):
//! ```text
//! λₖ₊₁ = λₖ · max(1/3, 1 - (2ρ - 1)³)
//! ```
//!
//! This provides smooth, data-driven adaptation of the damping parameter.
//!
//! ## Convergence Properties
//!
//! - **Global convergence**: Guaranteed to find a stationary point from any starting guess
//! - **Local quadratic convergence**: Near the solution, behaves like Gauss-Newton
//! - **Robust to poor initialization**: Adaptive damping prevents divergence
//! - **Handles ill-conditioning**: Large λ stabilizes nearly singular Hessian
//!
//! ## When to Use
//!
//! Levenberg-Marquardt is the best general-purpose choice when:
//! - Initial parameter guess may be far from the optimum
//! - Problem conditioning is unknown
//! - Robustness is prioritized over raw speed
//! - You want reliable convergence across diverse problem types
//!
//! For problems with specific structure, consider:
//! - [`GaussNewton`](crate::optimizer::GaussNewton) if well-conditioned with good initialization
//! - [`DogLeg`](crate::optimizer::DogLeg) for explicit trust region control
//!
//! # Implementation Features
//!
//! - **Sparse matrix support**: Efficient handling of large-scale problems via `faer` sparse library
//! - **Adaptive damping**: Nielsen's formula for smooth parameter adaptation
//! - **Robust linear solvers**: Cholesky (fast) or QR (stable) factorization
//! - **Jacobi scaling**: Optional diagonal preconditioning for mixed-scale problems
//! - **Covariance computation**: Optional uncertainty quantification after convergence
//! - **Manifold operations**: Native support for optimization on Lie groups (SE2, SE3, SO2, SO3)
//! - **Comprehensive diagnostics**: Detailed summaries of convergence and performance
//!
//! # Mathematical Background
//!
//! The augmented Hessian `J^T·J + λI` combines two beneficial properties:
//!
//! 1. **Positive definiteness**: Always solvable even when `J^T·J` is singular
//! 2. **Regularization**: Prevents taking steps in poorly-determined directions
//!
//! The trust region interpretation: λ controls an implicit spherical trust region where
//! larger λ restricts step size, ensuring the linear model remains valid.
//!
//! # Examples
//!
//! ## Basic usage
//!
//! ```no_run
//! use apex_solver::LevenbergMarquardt;
//! use apex_solver::core::problem::Problem;
//! use apex_solver::JacobianMode;
//!
//! # type TestResult = Result<(), Box<dyn std::error::Error>>;
//! # fn main() -> TestResult {
//! let mut problem = Problem::new(JacobianMode::Sparse);
//! // ... add residual blocks (factors) to problem ...
//!
//! let mut solver = LevenbergMarquardt::new();
//! let result = solver.optimize(&mut problem)?;
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced configuration
//!
//! ```no_run
//! use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardtConfig, LevenbergMarquardt};
//! use apex_solver::linalg::LinearSolverType;
//!
//! # fn main() {
//! let config = LevenbergMarquardtConfig::new()
//!     .with_max_iterations(100)
//!     .with_cost_tolerance(1e-6)
//!     .with_damping(1e-3)  // Initial damping
//!     .with_damping_bounds(1e-12, 1e12)  // Min/max damping
//!     .with_jacobi_scaling(true);  // Improve conditioning
//!
//! let mut solver = LevenbergMarquardt::with_config(config);
//! # }
//! ```
//!
//! # References
//!
//! - Levenberg, K. (1944). "A Method for the Solution of Certain Non-Linear Problems in Least Squares". *Quarterly of Applied Mathematics*.
//! - Marquardt, D. W. (1963). "An Algorithm for Least-Squares Estimation of Nonlinear Parameters". *Journal of the Society for Industrial and Applied Mathematics*.
//! - Madsen, K., Nielsen, H. B., & Tingleff, O. (2004). *Methods for Non-Linear Least Squares Problems* (2nd ed.). Chapter 3.
//! - Nocedal, J. & Wright, S. (2006). *Numerical Optimization* (2nd ed.). Springer. Chapter 10.
//! - Nielsen, H. B. (1999). "Damping Parameter in Marquardt's Method". Technical Report IMM-REP-1999-05.

use crate::core::problem::Problem;
use crate::error;
use crate::error::ErrorLogging;
use crate::linalg::{
    CovarianceOptions, Damping, DenseCholeskySolver, DenseMode, DenseQRSolver, JacobianMode,
    LinearSolver, LinearSolverType, SchurPreconditioner, SchurVariant, SparseCholeskySolver,
    SparseMode, SparseQRSolver, SparseSchurComplementSolver, StructureAware,
};
use crate::optimizer::{
    AssemblyBackend, ConvergenceParams, InitializedState, IterationStats, OptObserverVec,
    OptimizerError, apply_negative_parameter_step, apply_parameter_step,
};
use faer::Mat;
use std::time::{Duration, Instant};
use tracing::debug;

/// Policy for adapting the damping parameter λ between iterations.
///
/// Both policies use the same acceptance test —
/// `ρ > `[`min_relative_decrease`](LevenbergMarquardtConfig::min_relative_decrease) —
/// and differ only in how λ moves afterwards. They read disjoint sets of
/// configuration fields, so the doc for each variant is also the list of knobs
/// that have any effect under it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum DampingUpdate {
    /// Nielsen's rule (default).
    ///
    /// On an accepted step `λ ← λ · max(1/3, 1 − (2ρ − 1)³)` and ν resets to
    /// [`damping_nu`](LevenbergMarquardtConfig::damping_nu); on a rejected step
    /// `λ ← λ·ν` and `ν ← 2ν`, so consecutive failures escalate geometrically.
    /// The cubic makes λ shrink smoothly with step quality rather than by a
    /// fixed factor, which is why it is the default and what Ceres uses.
    ///
    /// Reads `damping_nu`. Ignores `damping_increase_factor`,
    /// `damping_decrease_factor`, `min_step_quality`, `good_step_quality`.
    #[default]
    Nielsen,
    /// Marquardt's classic three-band rule.
    ///
    /// `λ ← λ · damping_decrease_factor` when `ρ ≥ good_step_quality`,
    /// `λ ← λ · damping_increase_factor` when `ρ ≤ min_step_quality` or the step
    /// was rejected, and λ is left alone in between. Deterministic and easy to
    /// reason about, at the cost of reacting to step quality in coarse steps.
    ///
    /// Reads `damping_increase_factor`, `damping_decrease_factor`,
    /// `min_step_quality`, `good_step_quality`. Ignores `damping_nu`.
    ///
    /// Note that at the default `min_step_quality` of 0.0 the third branch is
    /// unreachable, because a step with `ρ ≤ 0` is rejected before it is
    /// consulted. Raise it above `min_relative_decrease` (0.25 is the textbook
    /// value) to make the middle band do anything.
    Marquardt,
}

/// Configuration parameters for the Levenberg-Marquardt optimizer.
///
/// Controls the adaptive damping strategy, convergence criteria, and numerical stability
/// enhancements for the Levenberg-Marquardt algorithm.
///
/// # Builder Pattern
///
/// All configuration options can be set using the builder pattern:
///
/// ```
/// use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
///
/// let config = LevenbergMarquardtConfig::new()
///     .with_max_iterations(100)
///     .with_damping(1e-3)
///     .with_damping_bounds(1e-12, 1e12)
///     .with_jacobi_scaling(true);
/// ```
///
/// # Damping Parameter Behavior
///
/// The damping parameter λ controls the trade-off between Gauss-Newton and gradient descent:
///
/// - **Initial damping** (`damping`): Starting value (default: 1e-4)
/// - **Damping bounds** (`damping_min`, `damping_max`): Valid range (default: 1e-12 to 1e12)
/// - **Adaptation**: Automatically adjusted based on step quality using Nielsen's formula
///
/// # Convergence Criteria
///
/// The optimizer terminates when ANY of the following conditions is met:
///
/// - **Cost tolerance**: `|cost_k - cost_{k-1}| < cost_tolerance`
/// - **Parameter tolerance**: `||step|| < parameter_tolerance`
/// - **Gradient tolerance**: `||J^T·r|| < gradient_tolerance`
/// - **Maximum iterations**: `iteration >= max_iterations`
/// - **Timeout**: `elapsed_time >= timeout`
///
/// # See Also
///
/// - [`LevenbergMarquardt`] - The solver that uses this configuration
/// - [`GaussNewtonConfig`](crate::optimizer::gauss_newton::GaussNewtonConfig) - Undamped variant
/// - [`DogLegConfig`](crate::optimizer::dog_leg::DogLegConfig) - Trust region alternative
#[derive(Clone)]
pub struct LevenbergMarquardtConfig {
    /// Type of linear solver for the linear systems
    pub linear_solver_type: LinearSolverType,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance for cost function
    pub cost_tolerance: f64,
    /// Convergence tolerance for parameter updates
    pub parameter_tolerance: f64,
    /// Convergence tolerance for gradient norm
    pub gradient_tolerance: f64,
    /// Timeout duration
    pub timeout: Option<Duration>,
    /// Initial damping parameter
    pub damping: f64,
    /// Minimum damping parameter
    pub damping_min: f64,
    /// Maximum damping parameter
    pub damping_max: f64,
    /// Factor λ is multiplied by when a step is rejected, or when
    /// `ρ <= min_step_quality`.
    ///
    /// Read only under [`DampingUpdate::Marquardt`]. Default: 10.0
    pub damping_increase_factor: f64,
    /// Factor λ is multiplied by when `ρ >= good_step_quality`.
    ///
    /// Read only under [`DampingUpdate::Marquardt`]. Default: 0.3
    pub damping_decrease_factor: f64,
    /// Nielsen's ν: the initial and reset value of the rejection escalation
    /// factor.
    ///
    /// Read only under [`DampingUpdate::Nielsen`]. Default: 2.0
    pub damping_nu: f64,
    /// Stop after this many consecutive rejected steps.
    pub max_consecutive_rejected_steps: usize,
    /// Step quality at or below which λ is increased even though the step was
    /// accepted.
    ///
    /// Read only under [`DampingUpdate::Marquardt`]. At the default of 0.0 this
    /// band is unreachable — every step with `ρ <= 0` has already been rejected
    /// by [`min_relative_decrease`](Self::min_relative_decrease). Default: 0.0
    pub min_step_quality: f64,
    /// Step quality at or above which λ is decreased.
    ///
    /// Read only under [`DampingUpdate::Marquardt`]. Default: 0.75
    pub good_step_quality: f64,
    /// Lower clamp on the Marquardt damping diagonal (Ceres' `min_lm_diagonal`).
    ///
    /// The augmented system is `(JᵀJ + λ·D)·dx = −Jᵀr` with
    /// `D_jj = clamp(JᵀJ_jj, min_diagonal, max_diagonal)`. Bounding `D` away
    /// from zero guarantees that even a direction with negligible curvature is
    /// damped. Setting `min_diagonal == max_diagonal == 1.0` gives `D = I`, i.e.
    /// classic uniform `λI` damping.
    ///
    /// Default: 1e-6 (Ceres-compatible). See [`Damping`].
    pub min_diagonal: f64,
    /// Upper clamp on the Marquardt damping diagonal (Ceres' `max_lm_diagonal`).
    ///
    /// Keeps one very stiff column from dominating the damped system.
    ///
    /// Default: 1e32 (Ceres-compatible). See [`Damping`].
    pub max_diagonal: f64,
    /// How λ is adapted between iterations.
    ///
    /// Default: [`DampingUpdate::Nielsen`]
    pub damping_update: DampingUpdate,
    /// Minimum objective function cutoff (optional early termination)
    ///
    /// If set, optimization terminates when cost falls below this threshold.
    /// Useful for early stopping when a "good enough" solution is acceptable.
    ///
    /// Default: None (disabled)
    pub min_cost_threshold: Option<f64>,
    /// Maximum condition number for Jacobian matrix (optional check)
    ///
    /// If set, the optimizer checks if condition_number(J^T*J) exceeds this
    /// threshold and terminates with IllConditionedJacobian status.
    /// Note: Computing condition number is expensive, so this is disabled by default.
    ///
    /// Default: None (disabled)
    pub max_condition_number: Option<f64>,
    /// Minimum relative cost decrease for step acceptance
    ///
    /// Used in computing step quality (rho = actual_reduction / predicted_reduction).
    /// Steps with rho < min_relative_decrease are rejected. Matches Ceres Solver's
    /// min_relative_decrease parameter.
    ///
    /// Default: 1e-3 (Ceres-compatible)
    pub min_relative_decrease: f64,
    /// Use Jacobi column scaling (preconditioning)
    ///
    /// When enabled, normalizes Jacobian columns by their L2 norm before solving.
    /// This can improve convergence for problems with mixed parameter scales
    /// (e.g., positions in meters + angles in radians) but adds ~5-10% overhead.
    ///
    /// Default: false (to avoid performance overhead and faster convergence)
    pub use_jacobi_scaling: bool,
    /// Compute per-variable covariance matrices (uncertainty estimation)
    ///
    /// When enabled, computes covariance by re-linearizing the problem at the
    /// solution and inverting the Gauss-Newton Hessian `H = JᵀJ` after
    /// convergence. The full covariance matrix is extracted into per-variable
    /// blocks stored in both Variable structs and SolverResult.
    ///
    /// Scaled (σ̂²·H⁻¹) versus unscaled (H⁻¹) covariance, the factorization
    /// algorithm, and the pseudo-inverse cutoff are configured through
    /// [`covariance_options`](Self::covariance_options).
    ///
    /// Default: false (to avoid performance overhead)
    pub compute_covariances: bool,
    /// Options for covariance estimation when `compute_covariances` is enabled.
    ///
    /// Defaults to unscaled `H⁻¹` via sparse Cholesky — the correct choice when
    /// residuals are whitened by their measurement information matrix. Set
    /// `apply_variance_scaling` for unweighted least squares, where the noise
    /// scale `σ̂² = 2·cost/(m−n)` must be estimated from the fit. See
    /// [`CovarianceOptions`](crate::linalg::covariance::CovarianceOptions).
    pub covariance_options: CovarianceOptions,
    /// Schur complement solver variant (for bundle adjustment problems)
    ///
    /// When using LinearSolverType::SparseSchurComplement, this determines which
    /// variant of the Schur complement method to use:
    /// - Sparse: Direct sparse Cholesky factorization (most accurate, moderate speed)
    /// - Iterative: Preconditioned Conjugate Gradients (memory efficient, good for large problems)
    /// - PowerSeries: Power series approximation (fastest, less accurate)
    ///
    /// Default: Sparse
    pub schur_variant: SchurVariant,
    /// Schur complement preconditioner type
    ///
    /// Determines the preconditioning strategy for iterative Schur methods:
    /// - Diagonal: Simple diagonal preconditioner (fast, less effective)
    /// - BlockDiagonal: Block-diagonal preconditioner (balanced)
    /// - IncompleteCholesky: Incomplete Cholesky factorization (slower, more effective)
    ///
    /// Default: Diagonal
    pub schur_preconditioner: SchurPreconditioner,
    // Note: Visualization is now handled via the observer pattern.
    // Use `solver.add_observer(RerunObserver::new(true)?)` to enable visualization.
    // This provides cleaner separation of concerns and allows multiple observers.
}

impl Default for LevenbergMarquardtConfig {
    fn default() -> Self {
        Self {
            linear_solver_type: LinearSolverType::default(),
            // Ceres Solver default: 50 (changed from 100 for compatibility)
            max_iterations: 50,
            // Ceres Solver default: 1e-6 (changed from 1e-8 for compatibility)
            cost_tolerance: 1e-6,
            // Ceres Solver default: 1e-8 (unchanged)
            parameter_tolerance: 1e-8,
            // Ceres Solver default: 1e-10 (changed from 1e-8 for compatibility)
            // Note: Typically should be 1e-4 * cost_tolerance per Ceres docs
            gradient_tolerance: 1e-10,
            timeout: None,
            // Ceres-equivalent: its default `initial_trust_region_radius` of 1e4
            // corresponds to λ = 1/radius = 1e-4. The previous 1e-3 was hand-tuned
            // against uniform λI damping, where λ alone had to absorb the problem
            // scale; with the Marquardt diagonal the scale lives in D instead.
            damping: 1e-4,
            damping_min: 1e-12,
            damping_max: 1e12,
            damping_increase_factor: 10.0,
            damping_decrease_factor: 0.3,
            damping_nu: 2.0,
            max_consecutive_rejected_steps: 5,
            min_step_quality: 0.0,
            good_step_quality: 0.75,
            min_diagonal: 1e-6,
            max_diagonal: 1e32,
            damping_update: DampingUpdate::default(),
            // New Ceres-compatible parameters
            min_cost_threshold: None,
            max_condition_number: None,
            min_relative_decrease: 1e-3,
            // Existing parameters
            // Jacobi scaling disabled by default for Schur solvers (incompatible with block structure)
            // Enable manually for Cholesky/QR solvers on mixed-scale problems
            use_jacobi_scaling: false,
            compute_covariances: false,
            covariance_options: CovarianceOptions::default(),
            // Schur complement parameters
            schur_variant: SchurVariant::default(),
            schur_preconditioner: SchurPreconditioner::default(),
        }
    }
}

impl LevenbergMarquardtConfig {
    /// Create a new Levenberg-Marquardt configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the linear solver type
    pub fn with_linear_solver_type(mut self, linear_solver_type: LinearSolverType) -> Self {
        self.linear_solver_type = linear_solver_type;
        self
    }

    /// Set the maximum number of iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the cost tolerance
    pub fn with_cost_tolerance(mut self, cost_tolerance: f64) -> Self {
        self.cost_tolerance = cost_tolerance;
        self
    }

    /// Set the parameter tolerance
    pub fn with_parameter_tolerance(mut self, parameter_tolerance: f64) -> Self {
        self.parameter_tolerance = parameter_tolerance;
        self
    }

    /// Set the gradient tolerance
    pub fn with_gradient_tolerance(mut self, gradient_tolerance: f64) -> Self {
        self.gradient_tolerance = gradient_tolerance;
        self
    }

    /// Set the timeout duration
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the initial damping parameter.
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    /// Set the damping parameter bounds.
    pub fn with_damping_bounds(mut self, min: f64, max: f64) -> Self {
        self.damping_min = min;
        self.damping_max = max;
        self
    }

    /// Set the damping adjustment factors used by [`DampingUpdate::Marquardt`].
    ///
    /// These have no effect under the default [`DampingUpdate::Nielsen`] policy,
    /// which derives both directions from ρ; pair this with
    /// [`with_damping_update`](Self::with_damping_update).
    pub fn with_damping_factors(mut self, increase: f64, decrease: f64) -> Self {
        self.damping_increase_factor = increase;
        self.damping_decrease_factor = decrease;
        self
    }

    /// Select the policy that adapts λ between iterations.
    pub fn with_damping_update(mut self, damping_update: DampingUpdate) -> Self {
        self.damping_update = damping_update;
        self
    }

    /// Set the clamp range for the Marquardt damping diagonal.
    ///
    /// The augmented system becomes `(JᵀJ + λ·D)·dx = −Jᵀr` with
    /// `D_jj = clamp(JᵀJ_jj, min, max)`. Pass `(1.0, 1.0)` for classic uniform
    /// `λI` damping.
    pub fn with_diagonal_bounds(mut self, min: f64, max: f64) -> Self {
        self.min_diagonal = min;
        self.max_diagonal = max;
        self
    }

    /// Set the step-quality band used by [`DampingUpdate::Marquardt`].
    ///
    /// λ is decreased at or above `good_quality` and increased at or below
    /// `min_quality`; in between it is left alone. No effect under
    /// [`DampingUpdate::Nielsen`].
    pub fn with_step_quality(mut self, min_quality: f64, good_quality: f64) -> Self {
        self.min_step_quality = min_quality;
        self.good_step_quality = good_quality;
        self
    }

    /// Set the trust region parameters.
    ///
    /// `radius` is ignored: this Levenberg-Marquardt implementation controls step
    /// size through `damping` (Nielsen's update rule), not through a trust-region
    /// radius, and the radius was never read by the algorithm. See
    /// <https://github.com/amin-abouee/apex-solver/issues/40>.
    #[deprecated(
        since = "1.5.0",
        note = "the `radius` argument is ignored — Levenberg-Marquardt here is damping-controlled. \
                Use `with_damping`/`with_damping_bounds` to control step size, and \
                `with_step_quality` for the quality thresholds."
    )]
    pub fn with_trust_region(self, _radius: f64, min_quality: f64, good_quality: f64) -> Self {
        self.with_step_quality(min_quality, good_quality)
    }

    /// Set minimum objective function cutoff for early termination.
    ///
    /// When set, optimization terminates with MinCostThresholdReached status
    /// if the cost falls below this threshold. Useful for early stopping when
    /// a "good enough" solution is acceptable.
    pub fn with_min_cost_threshold(mut self, min_cost: f64) -> Self {
        self.min_cost_threshold = Some(min_cost);
        self
    }

    /// No-op, retained for backward compatibility.
    ///
    /// This value was never read by the algorithm. Unlike Ceres, where
    /// `min_trust_region_radius` terminates the solve when the radius collapses,
    /// this implementation is damping-controlled and has no radius to compare
    /// against. See <https://github.com/amin-abouee/apex-solver/issues/40>.
    #[deprecated(
        since = "1.5.0",
        note = "no-op: Levenberg-Marquardt here is damping-controlled and never read this value. \
                Use `with_damping_bounds` to bound the damping instead."
    )]
    pub fn with_min_trust_region_radius(self, _min_radius: f64) -> Self {
        self
    }

    /// Set maximum condition number for Jacobian matrix.
    ///
    /// If set, the optimizer checks if condition_number(J^T*J) exceeds this
    /// threshold and terminates with IllConditionedJacobian status.
    /// Note: Computing condition number is expensive, disabled by default.
    pub fn with_max_condition_number(mut self, max_cond: f64) -> Self {
        self.max_condition_number = Some(max_cond);
        self
    }

    /// Set minimum relative cost decrease for step acceptance.
    ///
    /// Steps with rho = (actual_reduction / predicted_reduction) below this
    /// threshold are rejected. Default: 1e-3 (Ceres-compatible)
    pub fn with_min_relative_decrease(mut self, min_decrease: f64) -> Self {
        self.min_relative_decrease = min_decrease;
        self
    }

    /// Enable or disable Jacobi column scaling (preconditioning).
    ///
    /// When enabled, normalizes Jacobian columns by their L2 norm before solving.
    /// Can improve convergence for mixed-scale problems but adds ~5-10% overhead.
    pub fn with_jacobi_scaling(mut self, use_jacobi_scaling: bool) -> Self {
        self.use_jacobi_scaling = use_jacobi_scaling;
        self
    }

    /// Enable or disable covariance computation (uncertainty estimation).
    ///
    /// When enabled, computes the full covariance matrix by inverting the Hessian
    /// after convergence, then extracts per-variable covariance blocks.
    pub fn with_compute_covariances(mut self, compute_covariances: bool) -> Self {
        self.compute_covariances = compute_covariances;
        self
    }

    /// Set the options used for covariance estimation.
    ///
    /// Controls scaled (`σ̂²·H⁻¹`) versus unscaled (`H⁻¹`) covariance, the
    /// factorization algorithm (sparse Cholesky or dense SVD pseudo-inverse),
    /// and the singular-value cutoff. Only takes effect when
    /// `compute_covariances` is enabled.
    pub fn with_covariance_options(mut self, options: CovarianceOptions) -> Self {
        self.covariance_options = options;
        self
    }

    /// Set Schur complement solver variant
    pub fn with_schur_variant(mut self, variant: SchurVariant) -> Self {
        self.schur_variant = variant;
        self
    }

    /// Set Schur complement preconditioner
    pub fn with_schur_preconditioner(mut self, preconditioner: SchurPreconditioner) -> Self {
        self.schur_preconditioner = preconditioner;
        self
    }

    /// Configuration optimized for bundle adjustment problems.
    ///
    /// This preset uses settings tuned for large-scale bundle adjustment:
    /// - **Schur complement solver** with iterative PCG (memory efficient)
    /// - **Schur-Jacobi preconditioner** (Ceres-style, best PCG convergence)
    /// - **Moderate initial damping** (1e-3) - not too aggressive
    /// - **200 max iterations** (BA often needs more iterations for full convergence)
    /// - **Very tight tolerances** matching Ceres Solver for accurate reconstruction
    ///
    /// This configuration matches Ceres Solver's recommended BA settings and
    /// should achieve similar convergence quality.
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
    ///
    /// let config = LevenbergMarquardtConfig::for_bundle_adjustment();
    /// ```
    pub fn for_bundle_adjustment() -> Self {
        Self::default()
            .with_linear_solver_type(LinearSolverType::SparseSchurComplement)
            .with_schur_variant(SchurVariant::Iterative)
            .with_schur_preconditioner(SchurPreconditioner::SchurJacobi)
            .with_damping(1e-3) // Moderate initial damping (Ceres default)
            .with_max_iterations(20) // Reduced for early stop when RMSE < 1px
            // Match Ceres tolerances for faster convergence
            .with_cost_tolerance(1e-6) // Ceres function_tolerance (was 1e-12)
            .with_parameter_tolerance(1e-8) // Ceres parameter_tolerance (was 1e-14)
            .with_gradient_tolerance(1e-10) // Relaxed (was 1e-16)
    }

    /// Enable real-time visualization (graphical debugging).
    ///
    /// When enabled, optimization progress is logged to a Rerun viewer with:
    /// - Time series plots of cost, gradient norm, damping, step quality
    /// - Sparse Hessian matrix visualization as heat map
    /// - Gradient vector visualization
    /// - Real-time manifold state updates (for SE2/SE3 problems)
    ///
    /// **Note:** Requires the `visualization` feature to be enabled in `Cargo.toml`.
    /// Use `verbose` for terminal logging.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable visualization
    // Note: with_visualization() method has been removed.
    // Use the observer pattern instead:
    //   let mut solver = LevenbergMarquardt::with_config(config);
    //   solver.add_observer(RerunObserver::new(true)?);
    // This provides cleaner separation and allows multiple observers.
    ///   Print configuration parameters (verbose mode only)
    pub fn print_configuration(&self) {
        debug!(
            "Configuration:\n  Solver:        Levenberg-Marquardt\n  Linear solver: {:?}\n  Convergence Criteria:\n  Max iterations:      {}\n  Cost tolerance:      {:.2e}\n  Parameter tolerance: {:.2e}\n  Gradient tolerance:  {:.2e}\n  Timeout:             {:?}\n  Damping Parameters:\n  Initial damping:     {:.2e}\n  Damping range:       [{:.2e}, {:.2e}]\n  Increase factor:     {:.2}\n  Decrease factor:     {:.2}\n  Step Quality:\n  Min step quality:    {:.2}\n  Good step quality:   {:.2}\n  Numerical Settings:\n  Jacobi scaling:      {}\n  Compute covariances: {}",
            self.linear_solver_type,
            self.max_iterations,
            self.cost_tolerance,
            self.parameter_tolerance,
            self.gradient_tolerance,
            self.timeout,
            self.damping,
            self.damping_min,
            self.damping_max,
            self.damping_increase_factor,
            self.damping_decrease_factor,
            self.min_step_quality,
            self.good_step_quality,
            if self.use_jacobi_scaling {
                "enabled"
            } else {
                "disabled"
            },
            if self.compute_covariances {
                "enabled"
            } else {
                "disabled"
            }
        );
    }
}

/// Result from step computation
struct StepResult {
    step: Mat<f64>,
    gradient_norm: f64,
    predicted_reduction: f64,
}

/// Result from step evaluation
struct StepEvaluation {
    accepted: bool,
    cost_reduction: f64,
    rho: f64,
}

/// Levenberg-Marquardt solver for nonlinear least squares optimization.
///
/// Implements the damped Gauss-Newton method with adaptive damping parameter λ that
/// interpolates between Gauss-Newton and gradient descent based on step quality.
///
/// # Algorithm
///
/// At each iteration k:
/// 1. Compute residual `r(xₖ)` and Jacobian `J(xₖ)`
/// 2. Solve augmented system: `(J^T·J + λI)·h = -J^T·r`
/// 3. Evaluate step quality: `ρ = (actual reduction) / (predicted reduction)`
/// 4. If `ρ > 0`: Accept step and update `xₖ₊₁ = xₖ ⊕ h`, decrease λ
/// 5. If `ρ ≤ 0`: Reject step (keep `xₖ₊₁ = xₖ`), increase λ
/// 6. Check convergence criteria
///
/// The damping parameter λ is updated using Nielsen's smooth formula:
/// `λₖ₊₁ = λₖ · max(1/3, 1 - (2ρ - 1)³)` for accepted steps,
/// or `λₖ₊₁ = λₖ · ν` (with increasing ν) for rejected steps.
///
/// # Examples
///
/// ```no_run
/// use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardtConfig, LevenbergMarquardt};
/// use apex_solver::linalg::LinearSolverType;
///
/// # fn main() {
/// let config = LevenbergMarquardtConfig::new()
///     .with_max_iterations(100)
///     .with_damping(1e-3)
///     .with_damping_bounds(1e-12, 1e12)
///     .with_jacobi_scaling(true);
///
/// let mut solver = LevenbergMarquardt::with_config(config);
/// # }
/// ```
///
/// # See Also
///
/// - [`LevenbergMarquardtConfig`] - Configuration options
/// - [`GaussNewton`](crate::optimizer::GaussNewton) - Undamped variant (faster but less robust)
/// - [`DogLeg`](crate::optimizer::DogLeg) - Alternative trust region method
pub struct LevenbergMarquardt {
    config: LevenbergMarquardtConfig,
    jacobi_scaling: Option<Vec<f64>>,
    observers: OptObserverVec,
    /// Run state: the live damping λ, seeded from `config.damping` at the start
    /// of every solve.
    ///
    /// λ and ν are iteration state, not configuration. Keeping them here rather
    /// than mutating `config` in place means a second `optimize()` call on the
    /// same solver starts from the configured λ instead of inheriting whatever
    /// the previous run happened to end on.
    damping: f64,
    /// Run state: Nielsen's ν, seeded from `config.damping_nu`.
    damping_nu: f64,
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        Self::new()
    }
}

impl LevenbergMarquardt {
    /// Create a new Levenberg-Marquardt solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(LevenbergMarquardtConfig::default())
    }

    /// Create a new Levenberg-Marquardt solver with the given configuration.
    pub fn with_config(config: LevenbergMarquardtConfig) -> Self {
        Self {
            damping: config.damping,
            damping_nu: config.damping_nu,
            config,
            jacobi_scaling: None,
            observers: OptObserverVec::new(),
        }
    }

    /// Add an observer to monitor optimization progress.
    ///
    /// Observers are notified at each iteration with the current variable values.
    /// This enables real-time visualization, logging, metrics collection, etc.
    ///
    /// # Arguments
    ///
    /// * `observer` - Any type implementing `OptObserver`
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use apex_solver::{LevenbergMarquardt, LevenbergMarquardtConfig};
    /// # use apex_solver::core::problem::Problem;
    /// # use std::collections::HashMap;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut solver = LevenbergMarquardt::new();
    ///
    /// #[cfg(feature = "visualization")]
    /// {
    ///     use apex_solver::observers::RerunObserver;
    ///     let rerun_observer = RerunObserver::new(true)?;
    ///     solver.add_observer(rerun_observer);
    /// }
    ///
    /// // ... optimize ...
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_observer(&mut self, observer: impl crate::optimizer::OptObserver + 'static) {
        self.observers.add(observer);
    }

    /// Adapt λ to the observed step quality `rho`.
    ///
    /// `accepted` is decided by the caller against
    /// [`min_relative_decrease`](LevenbergMarquardtConfig::min_relative_decrease)
    /// and passed in, so the acceptance threshold and the damping policy stay
    /// independent — Ceres' `TrustRegionMinimizer` separates them the same way.
    ///
    /// Which fields are read depends on
    /// [`damping_update`](LevenbergMarquardtConfig::damping_update); see
    /// [`DampingUpdate`].
    fn update_damping(&mut self, rho: f64, accepted: bool) {
        match self.config.damping_update {
            DampingUpdate::Nielsen => {
                if accepted {
                    // λ ← λ · max(1/3, 1 − (2ρ − 1)³), and ν resets.
                    // Reference: Introduction to Optimization and Data Fitting,
                    // Algorithm 6.18.
                    let coff = 2.0 * rho - 1.0;
                    self.damping *= (1.0_f64 / 3.0).max(1.0 - coff * coff * coff);
                    self.damping_nu = self.config.damping_nu;
                } else {
                    self.damping *= self.damping_nu;
                    self.damping_nu *= 2.0;
                }
            }
            DampingUpdate::Marquardt => {
                if !accepted {
                    // A rejected step must always tighten the damping, whatever
                    // ρ was: leaving λ unchanged would recompute the identical
                    // step next iteration and stall.
                    self.damping *= self.config.damping_increase_factor;
                } else if rho >= self.config.good_step_quality {
                    self.damping *= self.config.damping_decrease_factor;
                } else if rho <= self.config.min_step_quality {
                    self.damping *= self.config.damping_increase_factor;
                }
            }
        }
        self.damping = self
            .damping
            .clamp(self.config.damping_min, self.config.damping_max);
    }

    /// Compute optimization step by solving the augmented system (generic over assembly mode).
    fn compute_step_generic<M: AssemblyBackend>(
        &self,
        residuals: &Mat<f64>,
        scaled_jacobian: &M::Jacobian,
        linear_solver: &mut dyn LinearSolver<M>,
    ) -> Result<StepResult, OptimizerError> {
        // Solve the augmented equation (J̃ᵀJ̃ + λ·D)·dx̃ = −J̃ᵀr.
        let damping = Damping::new(
            self.damping,
            self.config.min_diagonal,
            self.config.max_diagonal,
        )?;
        let scaled_step = linear_solver
            .solve_augmented_equation(residuals, scaled_jacobian, &damping)
            .map_err(|e| OptimizerError::LinearSolveFailed(e.to_string()).log_with_source(e))?;

        // Get the cached gradient (Jᵀr) and un-damped Hessian (JᵀJ) from the solver
        let gradient = linear_solver.get_gradient().ok_or_else(|| {
            OptimizerError::NumericalInstability("Gradient not available".into()).log()
        })?;
        let gradient_norm = gradient.norm_l2();
        let hessian = linear_solver.get_hessian().ok_or_else(|| {
            OptimizerError::NumericalInstability("Hessian not available".into()).log()
        })?;

        // Compute the predicted reduction BEFORE un-scaling the step: the
        // solver's cached gradient and Hessian are the *scaled* ones, and all
        // three vectors have to live in the same space. The predicted reduction
        // is a value of the quadratic model and is invariant under that change
        // of variables, so it is equally the predicted reduction of the
        // un-scaled step below.
        let predicted_reduction =
            crate::optimizer::compute_predicted_reduction::<M>(&scaled_step, gradient, hessian);

        // Apply inverse Jacobi scaling to get final step (if enabled)
        let step = if self.config.use_jacobi_scaling {
            let scaling = self
                .jacobi_scaling
                .as_ref()
                .ok_or_else(|| OptimizerError::JacobiScalingNotInitialized.log())?;
            M::apply_inverse_scaling(&scaled_step, scaling)
        } else {
            scaled_step
        };

        Ok(StepResult {
            step,
            gradient_norm,
            predicted_reduction,
        })
    }

    /// Evaluate and apply step, handling acceptance/rejection based on step quality
    fn evaluate_and_apply_step(
        &mut self,
        step_result: &StepResult,
        state: &mut InitializedState,
        problem: &Problem,
    ) -> error::ApexSolverResult<StepEvaluation> {
        // Apply parameter updates using manifold operations
        let _step_norm = apply_parameter_step(
            &mut state.variables,
            step_result.step.as_ref(),
            &state.sorted_vars,
        );

        // Compute new cost (residual only, no Jacobian needed for step evaluation)
        let (_new_residual, new_cost) = problem.compute_residual_and_cost_sparse_with_workspace(
            &state.variables,
            &mut state.workspace,
        )?;

        // Compute step quality
        let rho = crate::optimizer::compute_step_quality(
            state.current_cost,
            new_cost,
            step_result.predicted_reduction,
        );

        // Accept on step quality, then adapt λ. Ceres' TrustRegionMinimizer keeps
        // these two decisions separate for the same reason: the acceptance
        // threshold is a property of the problem, the damping update a property
        // of the chosen policy.
        let accepted = rho > self.config.min_relative_decrease;
        self.update_damping(rho, accepted);

        let cost_reduction = if accepted {
            // Accept the step - parameters already updated
            let reduction = state.current_cost - new_cost;
            state.current_cost = new_cost;
            reduction
        } else {
            // Reject the step - revert parameter changes
            apply_negative_parameter_step(
                &mut state.variables,
                step_result.step.as_ref(),
                &state.sorted_vars,
            );
            0.0
        };

        Ok(StepEvaluation {
            accepted,
            cost_reduction,
            rho,
        })
    }

    /// Run optimization using the specified assembly mode and linear solver.
    ///
    /// This is the core generic optimization loop. The public `optimize()` method
    /// dispatches to this based on `LinearSolverType`.
    fn optimize_with_mode<M: AssemblyBackend>(
        &mut self,
        problem: &mut Problem,
        linear_solver: &mut dyn LinearSolver<M>,
    ) -> crate::optimizer::OptimizeResult {
        let start_time = Instant::now();
        let mut iteration = 0;
        let mut cost_evaluations = 1;
        let mut jacobian_evaluations = 0;
        let mut successful_steps = 0;
        let mut unsuccessful_steps = 0;
        let mut consecutive_rejected = 0;

        // Initialize optimization state
        let mut state = crate::optimizer::initialize_optimization_state(problem)?;

        // Seed the run state from the configuration. λ and ν evolve during the
        // solve; resetting them here keeps `optimize()` idempotent when the same
        // solver instance is reused.
        self.damping = self.config.damping;
        self.damping_nu = self.config.damping_nu;
        self.jacobi_scaling = None;

        // Initialize summary tracking variables
        let mut max_gradient_norm: f64 = 0.0;
        let mut max_parameter_update_norm: f64 = 0.0;
        let mut total_cost_reduction = 0.0;
        let mut final_gradient_norm;
        let mut final_parameter_update_norm;

        // Initialize iteration statistics tracking
        let mut iteration_stats = Vec::with_capacity(self.config.max_iterations);
        let mut previous_cost = state.current_cost;

        // Print configuration and header if debug level is enabled
        if tracing::enabled!(tracing::Level::DEBUG) {
            self.config.print_configuration();
            IterationStats::print_header();
        }

        // Main optimization loop
        loop {
            let iter_start = Instant::now();

            // Shared preamble: assemble, conditioning check, Jacobi scaling
            let (residuals, scaled_jacobian) = match crate::optimizer::iteration_preamble::<M>(
                problem,
                &mut state,
                &mut self.jacobi_scaling,
                self.config.use_jacobi_scaling,
                iteration,
                self.config.max_condition_number,
                &mut jacobian_evaluations,
            )? {
                crate::optimizer::IterationPreamble::Proceed {
                    residuals,
                    scaled_jacobian,
                } => (residuals, scaled_jacobian),
                crate::optimizer::IterationPreamble::EarlyExit(status) => {
                    let elapsed = start_time.elapsed();
                    self.observers.notify_complete(&state.variables, iteration);
                    return Ok(crate::optimizer::build_solver_result(
                        status,
                        iteration,
                        state,
                        elapsed,
                        0.0,
                        0.0,
                        cost_evaluations,
                        jacobian_evaluations,
                        None,
                    ));
                }
            };

            // Compute optimization step
            let step_result =
                self.compute_step_generic::<M>(&residuals, &scaled_jacobian, linear_solver)?;

            // Update tracking variables
            max_gradient_norm = max_gradient_norm.max(step_result.gradient_norm);
            final_gradient_norm = step_result.gradient_norm;
            let step_norm = step_result.step.norm_l2();
            max_parameter_update_norm = max_parameter_update_norm.max(step_norm);
            final_parameter_update_norm = step_norm;

            // Evaluate and apply step (handles accept/reject)
            let step_eval = self.evaluate_and_apply_step(&step_result, &mut state, problem)?;
            cost_evaluations += 1;

            // Update counters based on acceptance
            if step_eval.accepted {
                successful_steps += 1;
                consecutive_rejected = 0;
                total_cost_reduction += step_eval.cost_reduction;
            } else {
                unsuccessful_steps += 1;
                consecutive_rejected += 1;
            }

            // OPTIMIZATION: Only collect iteration statistics if debug level is enabled
            if tracing::enabled!(tracing::Level::DEBUG) {
                let iter_elapsed_ms = iter_start.elapsed().as_secs_f64() * 1000.0;
                let total_elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

                let stats = IterationStats {
                    iteration,
                    cost: state.current_cost,
                    cost_change: previous_cost - state.current_cost,
                    gradient_norm: step_result.gradient_norm,
                    step_norm,
                    tr_ratio: step_eval.rho,
                    tr_radius: self.damping,
                    ls_iter: 0,
                    iter_time_ms: iter_elapsed_ms,
                    total_time_ms: total_elapsed_ms,
                    accepted: step_eval.accepted,
                };

                iteration_stats.push(stats.clone());
                stats.print_line();
            }

            previous_cost = state.current_cost;

            // Notify all observers with current state
            crate::optimizer::notify_observers_generic::<M>(
                &mut self.observers,
                &state.variables,
                iteration,
                state.current_cost,
                step_result.gradient_norm,
                Some(self.damping),
                step_norm,
                Some(step_eval.rho),
                linear_solver,
            );

            // Check convergence
            let elapsed = start_time.elapsed();
            let parameter_norm = crate::optimizer::compute_parameter_norm(&state.variables);
            let new_cost = state.current_cost;
            let cost_before_step = if step_eval.accepted {
                state.current_cost + step_eval.cost_reduction
            } else {
                state.current_cost
            };

            let convergence_status = crate::optimizer::check_convergence(&ConvergenceParams {
                iteration,
                current_cost: cost_before_step,
                new_cost,
                parameter_norm,
                parameter_update_norm: step_norm,
                gradient_norm: step_result.gradient_norm,
                elapsed,
                step_accepted: step_eval.accepted,
                max_iterations: self.config.max_iterations,
                gradient_tolerance: self.config.gradient_tolerance,
                parameter_tolerance: self.config.parameter_tolerance,
                cost_tolerance: self.config.cost_tolerance,
                min_cost_threshold: self.config.min_cost_threshold,
                timeout: self.config.timeout,
                trust_region_radius: None,
                min_trust_region_radius: None,
            })
            .or_else(|| {
                // `check_convergence` returns early on a rejected step, so a solver that
                // rejects every trial step would otherwise run out the full iteration
                // budget without the cost ever changing.
                //
                // Both conditions are required. A run of rejections alone is normal — LM
                // raises damping and the next step succeeds. Only once damping has also
                // saturated at `damping_max` can it no longer shrink the step further, so
                // the state is provably stuck and the remaining iterations are wasted.
                let damping_saturated = self.damping >= self.config.damping_max;
                let stalled = consecutive_rejected >= self.config.max_consecutive_rejected_steps
                    && damping_saturated;
                stalled.then_some(crate::optimizer::OptimizationStatus::StalledNoProgress)
            });

            if let Some(status) = convergence_status {
                if tracing::enabled!(tracing::Level::DEBUG) {
                    let summary = crate::optimizer::create_optimizer_summary(
                        "Levenberg-Marquardt",
                        state.initial_cost,
                        state.current_cost,
                        iteration + 1,
                        Some(successful_steps),
                        Some(unsuccessful_steps),
                        max_gradient_norm,
                        final_gradient_norm,
                        max_parameter_update_norm,
                        final_parameter_update_norm,
                        total_cost_reduction,
                        elapsed,
                        iteration_stats.clone(),
                        status.clone(),
                        Some(self.damping),
                        None,
                        Some(step_eval.rho),
                    );
                    debug!("{}", summary);
                }

                // Compute covariances if enabled
                let covariances = if self.config.compute_covariances {
                    problem.compute_and_set_covariances(
                        &mut state.variables,
                        self.config.covariance_options,
                    )
                } else {
                    None
                };

                // Notify observers that optimization is complete
                self.observers
                    .notify_complete(&state.variables, iteration + 1);

                return Ok(crate::optimizer::build_solver_result(
                    status,
                    iteration + 1,
                    state,
                    elapsed,
                    final_gradient_norm,
                    final_parameter_update_norm,
                    cost_evaluations,
                    jacobian_evaluations,
                    covariances,
                ));
            }

            iteration += 1;
        }
    }

    /// Run optimization, dispatching based on `problem.jacobian_mode`.
    ///
    /// - `JacobianMode::Dense` → `DenseCholesky` or `DenseQR`
    /// - `JacobianMode::Sparse` → `SparseCholesky`, `SparseQR` or
    ///   `SparseSchurComplement`
    ///
    /// A solver that does not match the problem's Jacobian mode is rejected
    /// with [`OptimizerError::InvalidParameters`] rather than silently replaced,
    /// matching Gauss-Newton and Dog Leg. Previously a mismatch ran Cholesky in
    /// the problem's own mode, so the solver actually used could differ from the
    /// configured one with no signal.
    pub fn optimize(&mut self, problem: &mut Problem) -> crate::optimizer::OptimizeResult {
        match problem.jacobian_mode {
            JacobianMode::Dense => match self.config.linear_solver_type {
                LinearSolverType::DenseQR => {
                    let mut solver = DenseQRSolver::new();
                    self.optimize_with_mode::<DenseMode>(problem, &mut solver)
                }
                LinearSolverType::DenseCholesky => {
                    let mut solver = DenseCholeskySolver::new();
                    self.optimize_with_mode::<DenseMode>(problem, &mut solver)
                }
                other => Err(OptimizerError::InvalidParameters(format!(
                    "Levenberg-Marquardt in dense Jacobian mode supports DenseCholesky and \
                     DenseQR only; requested {other}"
                ))
                .into()),
            },
            JacobianMode::Sparse => match self.config.linear_solver_type {
                LinearSolverType::SparseQR => {
                    let mut solver = SparseQRSolver::new();
                    self.optimize_with_mode::<SparseMode>(problem, &mut solver)
                }
                LinearSolverType::SparseSchurComplement => {
                    let state = crate::optimizer::initialize_optimization_state(problem)?;
                    let mut solver = SparseSchurComplementSolver::new()
                        .with_variant(self.config.schur_variant)
                        .with_preconditioner(self.config.schur_preconditioner);
                    solver
                        .initialize_structure(
                            &state.variables,
                            &state.variable_index_map,
                            &problem.schur_landmark_keys,
                        )
                        .map_err(|e| {
                            OptimizerError::LinearSolveFailed(format!(
                                "Failed to initialize Schur solver: {}",
                                e
                            ))
                            .log()
                        })?;
                    self.optimize_with_mode::<SparseMode>(problem, &mut solver)
                }
                LinearSolverType::SparseCholesky => {
                    let mut solver = SparseCholeskySolver::new();
                    self.optimize_with_mode::<SparseMode>(problem, &mut solver)
                }
                other => Err(OptimizerError::InvalidParameters(format!(
                    "Levenberg-Marquardt in sparse Jacobian mode supports SparseCholesky, \
                     SparseQR and SparseSchurComplement only; requested {other}"
                ))
                .into()),
            },
        }
    }
}
impl crate::optimizer::Optimizer for LevenbergMarquardt {
    fn optimize(&mut self, problem: &mut Problem) -> crate::optimizer::OptimizeResult {
        self.optimize(problem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ManifoldType;
    use crate::core::VarKey;
    use crate::core::variable::ManifoldVariable;
    use crate::factors::Factor;
    use crate::optimizer::OptimizationStatus;
    use faer::prelude::ReborrowMut;
    use nalgebra::dvector;
    use slotmap::SlotMap;

    type TestResult = Result<(), Box<dyn std::error::Error>>;
    /// Custom Rosenbrock Factor 1: r1 = 10(x2 - x1²)
    /// Demonstrates extensibility - custom factors can be defined outside of factors.rs
    #[derive(Debug, Clone)]
    struct RosenbrockFactor1;

    impl Factor for RosenbrockFactor1 {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            let x1 = params[0][0];
            let x2 = params[1][0];
            residual[0] = 10.0 * (x2 - x1 * x1);
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = -20.0 * x1;
                *jac.rb_mut().get_mut(0, 1) = 10.0;
            }
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 2)
        }
    }

    /// Custom Rosenbrock Factor 2: r2 = 1 - x1
    /// Demonstrates extensibility - custom factors can be defined outside of factors.rs
    #[derive(Debug, Clone)]
    struct RosenbrockFactor2;

    impl Factor for RosenbrockFactor2 {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = 1.0 - params[0][0];
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = -1.0;
            }
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 1)
        }
    }

    // -------------------------------------------------------------------------
    // Ceres-compatibility config fields: behaviour, not builder round-trips
    // -------------------------------------------------------------------------

    /// `min_relative_decrease` gates step acceptance.
    ///
    /// Set just below 1.0, essentially no step qualifies — ρ reaches 1 only when
    /// the quadratic model is exact — so the solver must reject its way to a
    /// stall instead of converging. This is the check that the field is read at
    /// all: before it was wired up, both configurations produced identical runs.
    #[test]
    fn min_relative_decrease_gates_acceptance() -> TestResult {
        let mut permissive_problem = rosenbrock_problem();
        let permissive = LevenbergMarquardt::with_config(
            LevenbergMarquardtConfig::new()
                .with_max_iterations(100)
                .with_min_relative_decrease(1e-3),
        )
        .optimize(&mut permissive_problem)?;

        let mut strict_problem = rosenbrock_problem();
        let strict = LevenbergMarquardt::with_config(
            LevenbergMarquardtConfig::new()
                .with_max_iterations(100)
                .with_min_relative_decrease(0.999_999),
        )
        .optimize(&mut strict_problem)?;

        assert!(
            strict.final_cost > permissive.final_cost,
            "a near-1.0 acceptance threshold should reject nearly every step and \
             leave the cost higher: strict {:.3e} vs permissive {:.3e}",
            strict.final_cost,
            permissive.final_cost
        );
        Ok(())
    }

    /// `min_diagonal` / `max_diagonal` change the damped system.
    ///
    /// Clamping both to 1.0 turns `λ·D` into `λI`; that must produce a
    /// different iterate sequence from the Marquardt default on a problem whose
    /// columns have unequal norms, which Rosenbrock's do.
    #[test]
    fn diagonal_bounds_select_between_marquardt_and_identity_damping() -> TestResult {
        let mut marquardt_problem = rosenbrock_problem();
        let marquardt = LevenbergMarquardt::with_config(
            LevenbergMarquardtConfig::new().with_max_iterations(100),
        )
        .optimize(&mut marquardt_problem)?;

        let mut identity_problem = rosenbrock_problem();
        let identity = LevenbergMarquardt::with_config(
            LevenbergMarquardtConfig::new()
                .with_max_iterations(100)
                .with_diagonal_bounds(1.0, 1.0),
        )
        .optimize(&mut identity_problem)?;

        assert_ne!(
            marquardt.iterations, identity.iterations,
            "λ·D and λI should not produce identical iterate counts on a \
             problem with unequal column norms — the bounds are being ignored"
        );
        // Both still solve Rosenbrock; the point is that they differ, not that
        // one is better on this particular problem.
        assert!(
            marquardt.final_cost < 1e-8,
            "marquardt cost {:.3e}",
            marquardt.final_cost
        );
        assert!(
            identity.final_cost < 1e-8,
            "identity cost {:.3e}",
            identity.final_cost
        );
        Ok(())
    }

    /// `DampingUpdate::Marquardt` follows the configured factors exactly.
    ///
    /// Nielsen derives both directions from ρ, so the two policies must not
    /// produce the same λ trajectory from the same inputs.
    #[test]
    fn marquardt_policy_uses_the_configured_damping_factors() {
        let config = LevenbergMarquardtConfig::new()
            .with_damping(1.0)
            .with_damping_bounds(1e-15, 1e15)
            .with_damping_update(DampingUpdate::Marquardt)
            .with_damping_factors(10.0, 0.3)
            .with_step_quality(0.0, 0.75);

        // Accepted, high quality → λ *= decrease_factor
        let mut solver = LevenbergMarquardt::with_config(config.clone());
        solver.update_damping(0.9, true);
        assert!(
            (solver.damping - 0.3).abs() < 1e-12,
            "got λ = {}",
            solver.damping
        );

        // Rejected → λ *= increase_factor, regardless of ρ
        let mut solver = LevenbergMarquardt::with_config(config.clone());
        solver.update_damping(0.9, false);
        assert!(
            (solver.damping - 10.0).abs() < 1e-12,
            "got λ = {}",
            solver.damping
        );

        // Accepted but mediocre (between min and good) → λ unchanged
        let mut solver = LevenbergMarquardt::with_config(config.clone());
        solver.update_damping(0.5, true);
        assert!(
            (solver.damping - 1.0).abs() < 1e-12,
            "got λ = {}",
            solver.damping
        );
    }

    /// Nielsen is the default and is unaffected by the Marquardt factors.
    #[test]
    fn nielsen_policy_ignores_the_marquardt_factors() {
        let base = LevenbergMarquardtConfig::new()
            .with_damping(1.0)
            .with_damping_bounds(1e-15, 1e15);

        let mut default_solver = LevenbergMarquardt::with_config(base.clone());
        default_solver.update_damping(0.9, true);

        let mut retuned_solver =
            LevenbergMarquardt::with_config(base.with_damping_factors(1e6, 1e-6));
        retuned_solver.update_damping(0.9, true);

        assert!(
            (default_solver.damping - retuned_solver.damping).abs() < 1e-15,
            "Nielsen must ignore damping_increase/decrease_factor, but λ differed: \
             {} vs {}",
            default_solver.damping,
            retuned_solver.damping
        );
    }

    /// λ and ν are run state, so a second `optimize()` reproduces the first.
    ///
    /// Before they were separated from the config, the second call inherited the
    /// λ the first run ended on and silently produced a different answer.
    #[test]
    fn repeated_optimize_calls_are_reproducible() -> TestResult {
        let mut solver = LevenbergMarquardt::with_config(
            LevenbergMarquardtConfig::new().with_max_iterations(100),
        );

        let mut first_problem = rosenbrock_problem();
        let first = solver.optimize(&mut first_problem)?;

        let mut second_problem = rosenbrock_problem();
        let second = solver.optimize(&mut second_problem)?;

        assert_eq!(
            first.iterations, second.iterations,
            "reusing a solver changed the iteration count: {} then {}",
            first.iterations, second.iterations
        );
        assert!(
            (first.final_cost - second.final_cost).abs() < 1e-15,
            "reusing a solver changed the final cost: {:.17e} then {:.17e}",
            first.final_cost,
            second.final_cost
        );
        Ok(())
    }

    /// `max_condition_number` terminates on a variable no residual constrains.
    ///
    /// The unconstrained variable contributes an all-zero Jacobian column, so
    /// the condition-number lower bound is infinite and the check fires.
    #[test]
    fn max_condition_number_detects_an_unconstrained_variable() -> TestResult {
        let mut problem = rosenbrock_problem();
        // Nothing references this variable, so its column of J is empty.
        let _orphan = problem.add_variable(ManifoldType::RN, dvector![0.0]);

        let result = LevenbergMarquardt::with_config(
            LevenbergMarquardtConfig::new()
                .with_max_iterations(10)
                .with_max_condition_number(1e12),
        )
        .optimize(&mut problem)?;

        assert_eq!(
            result.status,
            OptimizationStatus::IllConditionedJacobian,
            "an unconstrained variable should trip the conditioning check"
        );
        Ok(())
    }

    /// Without the check, the same unconstrained variable surfaces as an opaque
    /// linear-algebra failure from deep inside the solver.
    ///
    /// This is what `max_condition_number` buys: the diagnosis moves from
    /// "JᵀJ has structurally empty diagonal entries" — which names an internal
    /// data structure, not the user's mistake — to a typed status naming an
    /// ill-conditioned Jacobian.
    #[test]
    fn without_the_check_an_unconstrained_variable_is_an_opaque_error() {
        let mut problem = rosenbrock_problem();
        let _orphan = problem.add_variable(ManifoldType::RN, dvector![0.0]);

        let outcome = LevenbergMarquardt::with_config(
            LevenbergMarquardtConfig::new().with_max_iterations(10),
        )
        .optimize(&mut problem);

        assert!(
            outcome.is_err(),
            "expected the un-checked path to fail somewhere in the linear solver"
        );
    }

    /// The check stays out of the way on a well-conditioned problem.
    #[test]
    fn max_condition_number_does_not_fire_on_a_healthy_problem() -> TestResult {
        let mut healthy_problem = rosenbrock_problem();
        let healthy = LevenbergMarquardt::with_config(
            LevenbergMarquardtConfig::new()
                .with_max_iterations(100)
                .with_max_condition_number(1e12),
        )
        .optimize(&mut healthy_problem)?;
        assert_ne!(
            healthy.status,
            OptimizationStatus::IllConditionedJacobian,
            "a well-conditioned problem must not trip the check"
        );
        Ok(())
    }

    #[test]
    fn test_rosenbrock_optimization() -> TestResult {
        // Rosenbrock function test:
        // Minimize: r1² + r2² where
        //   r1 = 10(x2 - x1²)
        //   r2 = 1 - x1
        // Starting point: [-1.2, 1.0]
        // Expected minimum: [1.0, 1.0]

        let mut problem = Problem::new(JacobianMode::Sparse);
        let x1 = problem.add_variable(ManifoldType::RN, dvector![-1.2]);
        let x2 = problem.add_variable(ManifoldType::RN, dvector![1.0]);

        // Add custom factors (demonstrates extensibility!)
        problem.add_residual_block(&[x1, x2], Box::new(RosenbrockFactor1), None);
        problem.add_residual_block(&[x1], Box::new(RosenbrockFactor2), None);

        // Configure Levenberg-Marquardt optimizer
        let config = LevenbergMarquardtConfig::new()
            .with_max_iterations(100)
            .with_cost_tolerance(1e-8)
            .with_parameter_tolerance(1e-8)
            .with_gradient_tolerance(1e-10);

        let mut solver = LevenbergMarquardt::with_config(config);
        let result = solver.optimize(&mut problem)?;

        // Extract final values
        let x1_final = result.parameters[x1].as_param_slice()[0];
        let x2_final = result.parameters[x2].as_param_slice()[0];

        // Verify convergence to [1.0, 1.0]
        assert!(
            matches!(
                result.status,
                OptimizationStatus::Converged
                    | OptimizationStatus::CostToleranceReached
                    | OptimizationStatus::ParameterToleranceReached
                    | OptimizationStatus::GradientToleranceReached
            ),
            "Optimization should converge"
        );
        assert!(
            (x1_final - 1.0).abs() < 1e-4,
            "x1 should converge to 1.0, got {}",
            x1_final
        );
        assert!(
            (x2_final - 1.0).abs() < 1e-4,
            "x2 should converge to 1.0, got {}",
            x2_final
        );
        assert!(
            result.final_cost < 1e-6,
            "Final cost should be near zero, got {}",
            result.final_cost
        );
        Ok(())
    }

    /// Trivial factor: r = x - target, J = [[1.0]]
    struct LinearFactor {
        target: f64,
    }

    impl Factor for LinearFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = params[0][0] - self.target;
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = 1.0;
            }
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 1)
        }
    }

    fn rosenbrock_problem() -> Problem {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let x1 = problem.add_variable(ManifoldType::RN, dvector![-1.2]);
        let x2 = problem.add_variable(ManifoldType::RN, dvector![1.0]);
        problem.add_residual_block(&[x1, x2], Box::new(RosenbrockFactor1), None);
        problem.add_residual_block(&[x1], Box::new(RosenbrockFactor2), None);
        problem
    }

    fn linear_problem(start: f64) -> Problem {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let x = problem.add_variable(ManifoldType::RN, dvector![start]);
        problem.add_residual_block(&[x], Box::new(LinearFactor { target: 0.0 }), None);
        problem
    }

    // -------------------------------------------------------------------------
    // Predicted reduction under Jacobi scaling (issue #43)
    // -------------------------------------------------------------------------

    /// Two-variable factor with deliberately mismatched column scales:
    /// `r = [a·x1 - 1, x2 - 1]`, so the Jacobian columns have norms `a` and `1`.
    struct IllScaledFactor {
        a: f64,
    }

    impl Factor for IllScaledFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            let x1 = params[0][0];
            let x2 = params[1][0];
            residual[0] = self.a * x1 - 1.0;
            residual[1] = x2 - 1.0;
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = self.a;
                *jac.rb_mut().get_mut(0, 1) = 0.0;
                *jac.rb_mut().get_mut(1, 0) = 0.0;
                *jac.rb_mut().get_mut(1, 1) = 1.0;
            }
        }
        fn residual_dim(&self) -> usize {
            2
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (2, 2)
        }
    }

    /// The predicted reduction must equal the reduction of the quadratic model
    /// `m(dx) = ½‖r + J·dx‖²` measured with the **un-scaled** `J` and the
    /// **un-scaled** step that the optimizer actually applies.
    ///
    /// This is the invariant issue #43 reports as broken: with Jacobi scaling on,
    /// the old code fed an un-scaled step to a formula that had already been
    /// substituted with the scaled normal equations, mixing the two spaces.
    ///
    /// <https://github.com/amin-abouee/apex-solver/issues/43>
    fn assert_predicted_reduction_matches_model(use_jacobi_scaling: bool) -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let x1 = problem.add_variable(ManifoldType::RN, dvector![0.0]);
        let x2 = problem.add_variable(ManifoldType::RN, dvector![0.0]);
        problem.add_residual_block(&[x1, x2], Box::new(IllScaledFactor { a: 100.0 }), None);

        let config = LevenbergMarquardtConfig::new()
            .with_damping(1e-1)
            .with_jacobi_scaling(use_jacobi_scaling);
        let mut solver = LevenbergMarquardt::with_config(config);

        let mut state = crate::optimizer::initialize_optimization_state(&mut problem)?;
        let (residuals, jacobian) = SparseMode::assemble(
            &problem,
            &state.variables,
            &state.variable_index_map,
            state.symbolic_structure.as_ref(),
            state.total_dof,
            &mut state.workspace,
        )?;

        let solver_jacobian = if use_jacobi_scaling {
            crate::optimizer::process_jacobian_generic::<SparseMode>(
                &jacobian,
                &mut solver.jacobi_scaling,
                0,
            )?
        } else {
            jacobian.clone()
        };

        let mut linear_solver = SparseCholeskySolver::new();
        let step_result = solver.compute_step_generic::<SparseMode>(
            &residuals,
            &solver_jacobian,
            &mut linear_solver,
        )?;

        // Reduction of the quadratic model, computed independently from the
        // un-scaled Jacobian and the step the optimizer will actually apply.
        let predicted_residual = &residuals + &jacobian * &step_result.step;
        let expected =
            0.5 * residuals.squared_norm_l2() - 0.5 * predicted_residual.squared_norm_l2();

        assert!(
            (step_result.predicted_reduction - expected).abs() < 1e-9 * expected.abs().max(1.0),
            "predicted_reduction {} disagrees with the quadratic model {} \
             (use_jacobi_scaling = {})",
            step_result.predicted_reduction,
            expected,
            use_jacobi_scaling,
        );
        Ok(())
    }

    #[test]
    fn test_lm_predicted_reduction_matches_model_without_scaling() -> TestResult {
        assert_predicted_reduction_matches_model(false)
    }

    #[test]
    fn test_lm_predicted_reduction_matches_model_with_scaling() -> TestResult {
        assert_predicted_reduction_matches_model(true)
    }

    // -------------------------------------------------------------------------
    // Config builder tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_lm_config_default() {
        let cfg = LevenbergMarquardtConfig::default();
        assert_eq!(cfg.max_iterations, 50);
        assert!((cfg.cost_tolerance - 1e-6).abs() < 1e-15);
        assert!((cfg.damping - 1e-4).abs() < 1e-15);
        assert!(!cfg.use_jacobi_scaling);
        assert!(!cfg.compute_covariances);
        // Ceres' min_lm_diagonal / max_lm_diagonal.
        assert!((cfg.min_diagonal - 1e-6).abs() < 1e-15);
        assert!((cfg.max_diagonal - 1e32).abs() < 1e17);
        assert!((cfg.min_relative_decrease - 1e-3).abs() < 1e-15);
        assert_eq!(cfg.damping_update, DampingUpdate::Nielsen);
    }

    #[test]
    fn test_lm_config_builders() {
        let cfg = LevenbergMarquardtConfig::new()
            .with_max_iterations(42)
            .with_cost_tolerance(1e-4)
            .with_parameter_tolerance(1e-5)
            .with_gradient_tolerance(1e-6)
            .with_damping(1e-2)
            .with_damping_bounds(1e-15, 1e15)
            .with_damping_factors(8.0, 0.2)
            .with_min_cost_threshold(1e-12)
            .with_jacobi_scaling(true)
            .with_compute_covariances(true)
            .with_linear_solver_type(LinearSolverType::SparseQR);
        assert_eq!(cfg.max_iterations, 42);
        assert!((cfg.cost_tolerance - 1e-4).abs() < 1e-20);
        assert!((cfg.parameter_tolerance - 1e-5).abs() < 1e-20);
        assert!((cfg.gradient_tolerance - 1e-6).abs() < 1e-20);
        assert!((cfg.damping - 1e-2).abs() < 1e-15);
        assert!((cfg.damping_min - 1e-15).abs() < 1e-25);
        assert!((cfg.damping_max - 1e15).abs() < 1.0);
        assert!((cfg.damping_increase_factor - 8.0).abs() < 1e-12);
        assert!((cfg.damping_decrease_factor - 0.2).abs() < 1e-12);
        assert!(cfg.min_cost_threshold.is_some());
        assert!(cfg.use_jacobi_scaling);
        assert!(cfg.compute_covariances);
        assert!(matches!(cfg.linear_solver_type, LinearSolverType::SparseQR));
    }

    #[test]
    fn test_lm_for_bundle_adjustment() {
        let cfg = LevenbergMarquardtConfig::for_bundle_adjustment();
        assert!(matches!(
            cfg.linear_solver_type,
            LinearSolverType::SparseSchurComplement
        ));
        assert_eq!(cfg.max_iterations, 20);
    }

    #[test]
    fn test_lm_print_configuration_no_panic() {
        LevenbergMarquardtConfig::default().print_configuration();
    }

    #[test]
    fn test_lm_default_equals_new() {
        let a = LevenbergMarquardt::new();
        let b = LevenbergMarquardt::default();
        // Both should solve the same problem identically (smoke check)
        drop(a);
        drop(b);
    }

    #[test]
    fn test_lm_with_config_method() {
        let cfg = LevenbergMarquardtConfig::new().with_max_iterations(7);
        let solver = LevenbergMarquardt::with_config(cfg);
        drop(solver);
    }

    // -------------------------------------------------------------------------
    // Convergence termination paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_lm_max_iterations_termination() -> TestResult {
        let mut problem = rosenbrock_problem();
        let cfg = LevenbergMarquardtConfig::new().with_max_iterations(2);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&mut problem)?;
        assert_eq!(result.status, OptimizationStatus::MaxIterationsReached);
        assert!(result.iterations <= 3, "iterations={}", result.iterations);
        Ok(())
    }

    #[test]
    fn test_lm_gradient_tolerance_convergence() -> TestResult {
        let mut problem = linear_problem(1.0);
        // Very loose gradient tolerance → triggers after first accepted step
        let cfg = LevenbergMarquardtConfig::new()
            .with_gradient_tolerance(1e3)
            .with_cost_tolerance(1e-20)
            .with_parameter_tolerance(1e-20);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&mut problem)?;
        assert_eq!(result.status, OptimizationStatus::GradientToleranceReached);
        Ok(())
    }

    #[test]
    fn test_lm_min_cost_threshold() -> TestResult {
        let mut problem = rosenbrock_problem();
        // Set threshold very high so even initial cost triggers it
        let cfg = LevenbergMarquardtConfig::new()
            .with_min_cost_threshold(1e10)
            .with_cost_tolerance(1e-20)
            .with_gradient_tolerance(1e-20)
            .with_parameter_tolerance(1e-20);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&mut problem)?;
        assert_eq!(result.status, OptimizationStatus::MinCostThresholdReached);
        Ok(())
    }

    #[test]
    fn test_lm_qr_solver() -> TestResult {
        let mut problem = rosenbrock_problem();
        let cfg = LevenbergMarquardtConfig::new()
            .with_linear_solver_type(LinearSolverType::SparseQR)
            .with_max_iterations(100);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&mut problem)?;
        assert!(result.final_cost < 1e-6);
        Ok(())
    }

    #[test]
    fn test_lm_jacobi_scaling_enabled() -> TestResult {
        let mut problem = rosenbrock_problem();
        let cfg = LevenbergMarquardtConfig::new()
            .with_jacobi_scaling(true)
            .with_max_iterations(100);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&mut problem)?;
        assert!(result.final_cost < 1e-6);
        Ok(())
    }

    #[test]
    fn test_lm_result_initial_cost_greater_than_final() -> TestResult {
        let mut problem = rosenbrock_problem();
        let mut solver = LevenbergMarquardt::new();
        let result = solver.optimize(&mut problem)?;
        assert!(
            result.initial_cost > result.final_cost,
            "initial={} final={}",
            result.initial_cost,
            result.final_cost
        );
        Ok(())
    }

    #[test]
    fn test_lm_convergence_info_populated() -> TestResult {
        let mut problem = rosenbrock_problem();
        let mut solver = LevenbergMarquardt::new();
        let result = solver.optimize(&mut problem)?;
        assert!(result.convergence_info.is_some());
        Ok(())
    }

    #[test]
    fn test_lm_iterations_positive() -> TestResult {
        let mut problem = rosenbrock_problem();
        let mut solver = LevenbergMarquardt::new();
        let result = solver.optimize(&mut problem)?;
        assert!(result.iterations > 0);
        Ok(())
    }

    #[test]
    fn test_lm_timeout_config() {
        let cfg = LevenbergMarquardtConfig::new().with_timeout(Duration::from_secs(30));
        assert!(cfg.timeout.is_some());
    }

    #[test]
    fn test_lm_config_schur_variant_and_preconditioner() {
        use crate::linalg::{SchurPreconditioner, SchurVariant};
        let cfg = LevenbergMarquardtConfig::new()
            .with_schur_variant(SchurVariant::Iterative)
            .with_schur_preconditioner(SchurPreconditioner::BlockDiagonal);
        assert!(matches!(cfg.schur_variant, SchurVariant::Iterative));
        assert!(matches!(
            cfg.schur_preconditioner,
            SchurPreconditioner::BlockDiagonal
        ));
    }

    // -------------------------------------------------------------------------
    // Dense Jacobian mode dispatch
    // -------------------------------------------------------------------------

    /// Exercises the `JacobianMode::Dense + _ => DenseCholeskySolver` arm of `optimize()`.
    /// All existing tests use `JacobianMode::Sparse`, so this branch was previously uncovered.
    #[test]
    fn test_lm_dense_cholesky_solver() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Dense);
        let x1 = problem.add_variable(ManifoldType::RN, dvector![-1.2]);
        let x2 = problem.add_variable(ManifoldType::RN, dvector![1.0]);
        problem.add_residual_block(&[x1, x2], Box::new(RosenbrockFactor1), None);
        problem.add_residual_block(&[x1], Box::new(RosenbrockFactor2), None);

        // The solver must match the problem's mode; a mismatch is now an error
        // rather than a silent substitution, so ask for DenseCholesky by name.
        let cfg = LevenbergMarquardtConfig::new()
            .with_linear_solver_type(LinearSolverType::DenseCholesky)
            .with_max_iterations(100);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&mut problem)?;
        assert!(
            result.final_cost < 1e-6,
            "Dense Cholesky mode should converge Rosenbrock, got cost={}",
            result.final_cost
        );
        Ok(())
    }

    /// Exercises the `JacobianMode::Dense + DenseQR` arm of `optimize()`.
    #[test]
    fn test_lm_dense_qr_solver() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Dense);
        let x1 = problem.add_variable(ManifoldType::RN, dvector![-1.2]);
        let x2 = problem.add_variable(ManifoldType::RN, dvector![1.0]);
        problem.add_residual_block(&[x1, x2], Box::new(RosenbrockFactor1), None);
        problem.add_residual_block(&[x1], Box::new(RosenbrockFactor2), None);

        let cfg = LevenbergMarquardtConfig::new()
            .with_linear_solver_type(LinearSolverType::DenseQR)
            .with_max_iterations(100);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&mut problem)?;
        assert!(
            result.final_cost < 1e-6,
            "Dense QR mode should converge Rosenbrock, got cost={}",
            result.final_cost
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Covariance computation
    // -------------------------------------------------------------------------

    /// Exercises the `if self.config.compute_covariances { ... }` block at convergence.
    /// This block was completely unreachable in prior tests.
    #[test]
    fn test_lm_compute_covariances_enabled() -> TestResult {
        let mut problem = rosenbrock_problem();
        let cfg = LevenbergMarquardtConfig::new()
            .with_max_iterations(100)
            .with_compute_covariances(true);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&mut problem)?;
        assert!(
            result.covariances.is_some(),
            "compute_covariances=true should populate result.covariances"
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // update_damping() direct unit tests
    // -------------------------------------------------------------------------

    /// Nielsen: an accepted step decreases λ and resets ν.
    #[test]
    fn test_update_damping_accepted_step() {
        let cfg = LevenbergMarquardtConfig::new()
            .with_damping(1e-2)
            .with_damping_bounds(1e-15, 1e15);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let initial_damping = solver.damping;

        solver.update_damping(0.8, true);

        assert!(
            solver.damping < initial_damping,
            "accepted step should decrease damping: {} < {}",
            solver.damping,
            initial_damping
        );
        assert!(
            (solver.damping_nu - 2.0).abs() < 1e-15,
            "damping_nu should be reset to 2.0 after accepted step, got {}",
            solver.damping_nu
        );
    }

    /// Nielsen: a rejected step increases λ and doubles ν.
    #[test]
    fn test_update_damping_rejected_step() {
        let cfg = LevenbergMarquardtConfig::new()
            .with_damping(1e-2)
            .with_damping_bounds(1e-15, 1e15);
        let initial_nu = cfg.damping_nu; // default 2.0
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let initial_damping = solver.damping;

        solver.update_damping(-0.5, false);

        assert!(
            solver.damping > initial_damping,
            "rejected step should increase damping: {} > {}",
            solver.damping,
            initial_damping
        );
        assert!(
            (solver.damping_nu - initial_nu * 2.0).abs() < 1e-15,
            "damping_nu should double on rejected step: expected {}, got {}",
            initial_nu * 2.0,
            solver.damping_nu
        );
    }

    // -------------------------------------------------------------------------
    // Untested config builder methods
    // -------------------------------------------------------------------------

    /// Verifies `with_max_condition_number` and `with_min_relative_decrease` builder methods.
    #[test]
    fn test_lm_config_condition_number_and_relative_decrease() -> TestResult {
        let cfg = LevenbergMarquardtConfig::new()
            .with_max_condition_number(1e8)
            .with_min_relative_decrease(1e-4);
        let max_cond = cfg
            .max_condition_number
            .ok_or("max_condition_number should be Some")?;
        assert!((max_cond - 1e8).abs() < 1.0);
        assert!((cfg.min_relative_decrease - 1e-4).abs() < 1e-20);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Observer integration
    // -------------------------------------------------------------------------

    /// Verifies that `add_observer` registers an observer and `notify_complete` is called
    /// exactly once after optimization finishes.
    #[test]
    fn test_lm_add_observer_called_on_completion() -> TestResult {
        use crate::optimizer::OptObserver;
        use std::sync::{Arc, Mutex};

        struct CountObserver {
            complete_calls: Arc<Mutex<usize>>,
        }

        impl OptObserver for CountObserver {
            fn on_step(
                &self,
                _values: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
                _iteration: usize,
            ) {
            }

            fn on_optimization_complete(
                &self,
                _values: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
                _iterations: usize,
            ) {
                if let Ok(mut guard) = self.complete_calls.lock() {
                    *guard += 1;
                }
            }
        }

        let call_count = Arc::new(Mutex::new(0usize));
        let observer = CountObserver {
            complete_calls: Arc::clone(&call_count),
        };

        let mut problem = rosenbrock_problem();
        let mut solver = LevenbergMarquardt::new();
        solver.add_observer(observer);
        let _ = solver.optimize(&mut problem)?;

        let count = *call_count
            .lock()
            .map_err(|e| format!("mutex poisoned: {e}"))?;
        assert_eq!(
            count, 1,
            "on_optimization_complete should be called exactly once"
        );
        Ok(())
    }

    /// A solver that does not match the problem's Jacobian mode must be
    /// rejected, not silently swapped for Cholesky.
    ///
    /// Regression for the `_ =>` arms that made LM the only optimizer to
    /// coerce a mismatch in silence.
    #[test]
    fn test_lm_rejects_solver_mode_mismatch() {
        fn rosenbrock(mode: JacobianMode) -> Problem {
            let mut problem = Problem::new(mode);
            let x1 = problem.add_variable(ManifoldType::RN, dvector![-1.2]);
            let x2 = problem.add_variable(ManifoldType::RN, dvector![1.0]);
            problem.add_residual_block(&[x1, x2], Box::new(RosenbrockFactor1), None);
            problem.add_residual_block(&[x1], Box::new(RosenbrockFactor2), None);
            problem
        }

        // Dense problem, sparse solvers requested.
        for solver_type in [
            LinearSolverType::SparseCholesky,
            LinearSolverType::SparseQR,
            LinearSolverType::SparseSchurComplement,
        ] {
            let mut problem = rosenbrock(JacobianMode::Dense);
            let config = LevenbergMarquardtConfig::new().with_linear_solver_type(solver_type);
            let Err(err) = LevenbergMarquardt::with_config(config).optimize(&mut problem) else {
                panic!("dense problem must reject sparse solver {solver_type}");
            };
            assert!(
                err.to_string().contains("dense Jacobian mode"),
                "unexpected error for {solver_type}: {err}"
            );
        }

        // Sparse problem, dense solvers requested.
        for solver_type in [
            LinearSolverType::DenseCholesky,
            LinearSolverType::DenseQR,
        ] {
            let mut problem = rosenbrock(JacobianMode::Sparse);
            let config = LevenbergMarquardtConfig::new().with_linear_solver_type(solver_type);
            let Err(err) = LevenbergMarquardt::with_config(config).optimize(&mut problem) else {
                panic!("sparse problem must reject dense solver {solver_type}");
            };
            assert!(
                err.to_string().contains("sparse Jacobian mode"),
                "unexpected error for {solver_type}: {err}"
            );
        }
    }

    /// Every matching mode/solver pair must still run.
    #[test]
    fn test_lm_accepts_matching_solver_modes() -> TestResult {
        for (mode, solver_type) in [
            (JacobianMode::Dense, LinearSolverType::DenseCholesky),
            (JacobianMode::Dense, LinearSolverType::DenseQR),
            (JacobianMode::Sparse, LinearSolverType::SparseCholesky),
            (JacobianMode::Sparse, LinearSolverType::SparseQR),
        ] {
            let mut problem = Problem::new(mode);
            let x1 = problem.add_variable(ManifoldType::RN, dvector![-1.2]);
            let x2 = problem.add_variable(ManifoldType::RN, dvector![1.0]);
            problem.add_residual_block(&[x1, x2], Box::new(RosenbrockFactor1), None);
            problem.add_residual_block(&[x1], Box::new(RosenbrockFactor2), None);

            let config = LevenbergMarquardtConfig::new()
                .with_max_iterations(50)
                .with_linear_solver_type(solver_type);
            LevenbergMarquardt::with_config(config)
                .optimize(&mut problem)
                .map_err(|e| format!("{mode:?} + {solver_type} should be accepted: {e}"))?;
        }
        Ok(())
    }
}
