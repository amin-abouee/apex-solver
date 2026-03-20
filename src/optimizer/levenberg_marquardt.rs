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
//! - [`GaussNewton`](crate::GaussNewton) if well-conditioned with good initialization
//! - [`DogLeg`](crate::DogLeg) for explicit trust region control
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
//! use std::collections::HashMap;
//!
//! # fn main() -> TestResult {
//! let mut problem = Problem::new(JacobianMode::Sparse);
//! // ... add residual blocks (factors) to problem ...
//!
//! let initial_values = HashMap::new();
//! // ... initialize parameters ...
//!
//! let mut solver = LevenbergMarquardt::new();
//! let result = solver.optimize(&problem, &initial_values)?;
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

use crate::core::problem::{Problem, VariableEnum};
use crate::error;
use crate::linalg::{
    DenseCholeskySolver, DenseMode, DenseQRSolver, JacobianMode, LinearSolver, LinearSolverType,
    SchurPreconditioner, SchurVariant, SparseCholeskySolver, SparseMode, SparseQRSolver,
    SparseSchurComplementSolver, StructureAware,
};
use crate::optimizer::{
    AssemblyBackend, ConvergenceParams, InitializedState, IterationStats, OptObserverVec,
    OptimizerError, SolverResult, apply_negative_parameter_step, apply_parameter_step,
    compute_cost,
};
use apex_manifolds::ManifoldType;

use faer::Mat;
use nalgebra::DVector;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::debug;

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
/// - [`GaussNewtonConfig`](crate::GaussNewtonConfig) - Undamped variant
/// - [`DogLegConfig`](crate::DogLegConfig) - Trust region alternative
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
    /// Damping increase factor (when step rejected)
    pub damping_increase_factor: f64,
    /// Damping decrease factor (when step accepted)
    pub damping_decrease_factor: f64,
    /// Damping nu parameter
    pub damping_nu: f64,
    /// Trust region radius
    pub trust_region_radius: f64,
    /// Minimum step quality for acceptance
    pub min_step_quality: f64,
    /// Good step quality threshold
    pub good_step_quality: f64,
    /// Minimum diagonal value for regularization
    pub min_diagonal: f64,
    /// Maximum diagonal value for regularization
    pub max_diagonal: f64,
    /// Minimum objective function cutoff (optional early termination)
    ///
    /// If set, optimization terminates when cost falls below this threshold.
    /// Useful for early stopping when a "good enough" solution is acceptable.
    ///
    /// Default: None (disabled)
    pub min_cost_threshold: Option<f64>,
    /// Minimum trust region radius before termination
    ///
    /// When the trust region radius falls below this value, the optimizer
    /// terminates as it indicates the search has converged or the problem
    /// is ill-conditioned. Matches Ceres Solver's min_trust_region_radius.
    ///
    /// Default: 1e-32 (Ceres-compatible)
    pub min_trust_region_radius: f64,
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
    /// When enabled, computes covariance by inverting the Hessian matrix after
    /// convergence. The full covariance matrix is extracted into per-variable
    /// blocks stored in both Variable structs and SolverResult.
    ///
    /// Default: false (to avoid performance overhead)
    pub compute_covariances: bool,
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
            damping: 1e-3, // Increased from 1e-4 for better initial convergence on BA
            damping_min: 1e-12,
            damping_max: 1e12,
            damping_increase_factor: 10.0,
            damping_decrease_factor: 0.3,
            damping_nu: 2.0,
            trust_region_radius: 1e4,
            min_step_quality: 0.0,
            good_step_quality: 0.75,
            min_diagonal: 1e-6,
            max_diagonal: 1e32,
            // New Ceres-compatible parameters
            min_cost_threshold: None,
            min_trust_region_radius: 1e-32,
            max_condition_number: None,
            min_relative_decrease: 1e-3,
            // Existing parameters
            // Jacobi scaling disabled by default for Schur solvers (incompatible with block structure)
            // Enable manually for Cholesky/QR solvers on mixed-scale problems
            use_jacobi_scaling: false,
            compute_covariances: false,
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

    /// Set the damping adjustment factors.
    pub fn with_damping_factors(mut self, increase: f64, decrease: f64) -> Self {
        self.damping_increase_factor = increase;
        self.damping_decrease_factor = decrease;
        self
    }

    /// Set the trust region parameters.
    pub fn with_trust_region(mut self, radius: f64, min_quality: f64, good_quality: f64) -> Self {
        self.trust_region_radius = radius;
        self.min_step_quality = min_quality;
        self.good_step_quality = good_quality;
        self
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

    /// Set minimum trust region radius before termination.
    ///
    /// When the trust region radius falls below this value, optimization
    /// terminates with TrustRegionRadiusTooSmall status.
    /// Default: 1e-32 (Ceres-compatible)
    pub fn with_min_trust_region_radius(mut self, min_radius: f64) -> Self {
        self.min_trust_region_radius = min_radius;
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
            "Configuration:\n  Solver:        Levenberg-Marquardt\n  Linear solver: {:?}\n  Convergence Criteria:\n  Max iterations:      {}\n  Cost tolerance:      {:.2e}\n  Parameter tolerance: {:.2e}\n  Gradient tolerance:  {:.2e}\n  Timeout:             {:?}\n  Damping Parameters:\n  Initial damping:     {:.2e}\n  Damping range:       [{:.2e}, {:.2e}]\n  Increase factor:     {:.2}\n  Decrease factor:     {:.2}\n  Trust Region:\n  Initial radius:      {:.2e}\n  Min step quality:    {:.2}\n  Good step quality:   {:.2}\n  Numerical Settings:\n  Jacobi scaling:      {}\n  Compute covariances: {}",
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
            self.trust_region_radius,
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
/// - [`GaussNewton`](crate::GaussNewton) - Undamped variant (faster but less robust)
/// - [`DogLeg`](crate::DogLeg) - Alternative trust region method
pub struct LevenbergMarquardt {
    config: LevenbergMarquardtConfig,
    jacobi_scaling: Option<Vec<f64>>,
    observers: OptObserverVec,
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

    /// Update damping parameter based on step quality using trust region approach
    /// Reference: Introduction to Optimization and Data Fitting
    /// Algorithm 6.18
    fn update_damping(&mut self, rho: f64) -> bool {
        if rho > 0.0 {
            // Step accepted - decrease damping
            let coff = 2.0 * rho - 1.0;
            self.config.damping *= (1.0_f64 / 3.0).max(1.0 - coff * coff * coff);
            self.config.damping = self.config.damping.max(self.config.damping_min);
            self.config.damping_nu = 2.0;
            true
        } else {
            // Step rejected - increase damping
            self.config.damping *= self.config.damping_nu;
            self.config.damping_nu *= 2.0;
            self.config.damping = self.config.damping.min(self.config.damping_max);
            false
        }
    }

    /// Compute predicted cost reduction from linear model
    /// Standard LM formula: 0.5 * step^T * (damping * step - gradient)
    fn compute_predicted_reduction(&self, step: &Mat<f64>, gradient: &Mat<f64>) -> f64 {
        // Standard Levenberg-Marquardt predicted reduction formula
        // predicted_reduction = -step^T * gradient - 0.5 * step^T * H * step
        //                     = 0.5 * step^T * (damping * step - gradient)
        let diff = self.config.damping * step - gradient;
        (0.5 * step.transpose() * &diff)[(0, 0)]
    }

    /// Compute optimization step by solving the augmented system (generic over assembly mode).
    fn compute_step_generic<M: AssemblyBackend>(
        &self,
        residuals: &Mat<f64>,
        scaled_jacobian: &M::Jacobian,
        linear_solver: &mut dyn LinearSolver<M>,
    ) -> Result<StepResult, OptimizerError> {
        // Solve augmented equation: (J_scaled^T * J_scaled + λI) * dx_scaled = -J_scaled^T * r
        let residuals_owned = residuals.as_ref().to_owned();
        let scaled_step = linear_solver
            .solve_augmented_equation(&residuals_owned, scaled_jacobian, self.config.damping)
            .map_err(|e| OptimizerError::LinearSolveFailed(e.to_string()).log_with_source(e))?;

        // Get cached gradient from the solver
        let gradient = linear_solver.get_gradient().ok_or_else(|| {
            OptimizerError::NumericalInstability("Gradient not available".into()).log()
        })?;
        let gradient_norm = gradient.norm_l2();

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

        // Compute predicted reduction using scaled values
        let predicted_reduction = self.compute_predicted_reduction(&step, gradient);

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
        let new_residual = problem.compute_residual_sparse(&state.variables)?;
        let new_cost = compute_cost(&new_residual);

        // Compute step quality
        let rho = crate::optimizer::compute_step_quality(
            state.current_cost,
            new_cost,
            step_result.predicted_reduction,
        );

        // Update damping and decide whether to accept step
        let accepted = self.update_damping(rho);

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
        problem: &Problem,
        initial_params: &HashMap<String, (ManifoldType, DVector<f64>)>,
        linear_solver: &mut dyn LinearSolver<M>,
    ) -> Result<SolverResult<HashMap<String, VariableEnum>>, error::ApexSolverError> {
        let start_time = Instant::now();
        let mut iteration = 0;
        let mut cost_evaluations = 1;
        let mut jacobian_evaluations = 0;
        let mut successful_steps = 0;
        let mut unsuccessful_steps = 0;

        // Initialize optimization state
        let mut state = crate::optimizer::initialize_optimization_state(problem, initial_params)?;

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

            // Evaluate residuals and Jacobian using the assembly mode
            let (residuals, jacobian) = M::assemble(
                problem,
                &state.variables,
                &state.variable_index_map,
                state.symbolic_structure.as_ref(),
                state.total_dof,
            )?;
            jacobian_evaluations += 1;

            // Process Jacobian (apply scaling if enabled)
            let scaled_jacobian = if self.config.use_jacobi_scaling {
                crate::optimizer::process_jacobian_generic::<M>(
                    &jacobian,
                    &mut self.jacobi_scaling,
                    iteration,
                )?
            } else {
                jacobian
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
                total_cost_reduction += step_eval.cost_reduction;
            } else {
                unsuccessful_steps += 1;
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
                    tr_radius: self.config.damping,
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
                Some(self.config.damping),
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

            if let Some(status) = crate::optimizer::check_convergence(&ConvergenceParams {
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
                trust_region_radius: Some(self.config.trust_region_radius),
                min_trust_region_radius: Some(self.config.min_trust_region_radius),
            }) {
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
                        Some(self.config.damping),
                        None,
                        Some(step_eval.rho),
                    );
                    debug!("{}", summary);
                }

                // Compute covariances if enabled
                let covariances = if self.config.compute_covariances {
                    problem.compute_and_set_covariances_generic::<M>(
                        linear_solver,
                        &mut state.variables,
                        &state.variable_index_map,
                    )
                } else {
                    None
                };

                // Notify observers that optimization is complete
                let final_parameters: HashMap<String, VariableEnum> = state
                    .variables
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                self.observers
                    .notify_complete(&final_parameters, iteration + 1);

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
    /// - `JacobianMode::Dense` → always uses `DenseCholeskySolver`
    /// - `JacobianMode::Sparse` → uses the solver selected by `config.linear_solver_type`
    pub fn optimize(
        &mut self,
        problem: &Problem,
        initial_params: &HashMap<String, (ManifoldType, DVector<f64>)>,
    ) -> Result<SolverResult<HashMap<String, VariableEnum>>, error::ApexSolverError> {
        match problem.jacobian_mode {
            JacobianMode::Dense => match self.config.linear_solver_type {
                LinearSolverType::DenseQR => {
                    let mut solver = DenseQRSolver::new();
                    self.optimize_with_mode::<DenseMode>(problem, initial_params, &mut solver)
                }
                _ => {
                    let mut solver = DenseCholeskySolver::new();
                    self.optimize_with_mode::<DenseMode>(problem, initial_params, &mut solver)
                }
            },
            JacobianMode::Sparse => match self.config.linear_solver_type {
                LinearSolverType::SparseQR => {
                    let mut solver = SparseQRSolver::new();
                    self.optimize_with_mode::<SparseMode>(problem, initial_params, &mut solver)
                }
                LinearSolverType::SparseSchurComplement => {
                    // Schur complement needs variable structure before the first solve.
                    // Initialize state once to get variables, then create and prepare the solver.
                    // optimize_with_mode will re-initialize state internally, which is acceptable
                    // since Schur structure initialization is cheap.
                    let state =
                        crate::optimizer::initialize_optimization_state(problem, initial_params)?;
                    let mut solver = SparseSchurComplementSolver::new()
                        .with_variant(self.config.schur_variant)
                        .with_preconditioner(self.config.schur_preconditioner);
                    solver
                        .initialize_structure(&state.variables, &state.variable_index_map)
                        .map_err(|e| {
                            OptimizerError::LinearSolveFailed(format!(
                                "Failed to initialize Schur solver: {}",
                                e
                            ))
                            .log()
                        })?;
                    self.optimize_with_mode::<SparseMode>(problem, initial_params, &mut solver)
                }
                _ => {
                    // SparseCholesky (default) or DenseCholesky with sparse mode → SparseCholeskySolver
                    let mut solver = SparseCholeskySolver::new();
                    self.optimize_with_mode::<SparseMode>(problem, initial_params, &mut solver)
                }
            },
        }
    }
}
impl crate::optimizer::Optimizer for LevenbergMarquardt {
    fn optimize(
        &mut self,
        problem: &Problem,
        initial_params: &HashMap<String, (ManifoldType, DVector<f64>)>,
    ) -> Result<SolverResult<HashMap<String, VariableEnum>>, crate::error::ApexSolverError> {
        self.optimize(problem, initial_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::Factor;
    use crate::optimizer::OptimizationStatus;
    use nalgebra::{DMatrix, dvector};

    type TestResult = Result<(), Box<dyn std::error::Error>>;
    /// Custom Rosenbrock Factor 1: r1 = 10(x2 - x1²)
    /// Demonstrates extensibility - custom factors can be defined outside of factors.rs
    #[derive(Debug, Clone)]
    struct RosenbrockFactor1;

    impl Factor for RosenbrockFactor1 {
        fn linearize(
            &self,
            params: &[DVector<f64>],
            compute_jacobian: bool,
        ) -> (DVector<f64>, Option<DMatrix<f64>>) {
            let x1 = params[0][0];
            let x2 = params[1][0];

            // Residual: r1 = 10(x2 - x1²)
            let residual = dvector![10.0 * (x2 - x1 * x1)];

            let jacobian = if compute_jacobian {
                // Jacobian: ∂r1/∂x1 = -20*x1, ∂r1/∂x2 = 10
                let mut jacobian = DMatrix::zeros(1, 2);
                jacobian[(0, 0)] = -20.0 * x1;
                jacobian[(0, 1)] = 10.0;

                Some(jacobian)
            } else {
                None
            };

            (residual, jacobian)
        }

        fn get_dimension(&self) -> usize {
            1
        }
    }

    /// Custom Rosenbrock Factor 2: r2 = 1 - x1
    /// Demonstrates extensibility - custom factors can be defined outside of factors.rs
    #[derive(Debug, Clone)]
    struct RosenbrockFactor2;

    impl Factor for RosenbrockFactor2 {
        fn linearize(
            &self,
            params: &[DVector<f64>],
            compute_jacobian: bool,
        ) -> (DVector<f64>, Option<DMatrix<f64>>) {
            let x1 = params[0][0];

            // Residual: r2 = 1 - x1
            let residual = dvector![1.0 - x1];

            let jacobian = if compute_jacobian {
                // Jacobian: ∂r2/∂x1 = -1
                Some(DMatrix::from_element(1, 1, -1.0))
            } else {
                None
            };

            (residual, jacobian)
        }

        fn get_dimension(&self) -> usize {
            1
        }
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
        let mut initial_values = HashMap::new();

        // Add variables using Rn manifold (Euclidean space)
        initial_values.insert("x1".to_string(), (ManifoldType::RN, dvector![-1.2]));
        initial_values.insert("x2".to_string(), (ManifoldType::RN, dvector![1.0]));

        // Add custom factors (demonstrates extensibility!)
        problem.add_residual_block(&["x1", "x2"], Box::new(RosenbrockFactor1), None);
        problem.add_residual_block(&["x1"], Box::new(RosenbrockFactor2), None);

        // Configure Levenberg-Marquardt optimizer
        let config = LevenbergMarquardtConfig::new()
            .with_max_iterations(100)
            .with_cost_tolerance(1e-8)
            .with_parameter_tolerance(1e-8)
            .with_gradient_tolerance(1e-10);

        let mut solver = LevenbergMarquardt::with_config(config);
        let result = solver.optimize(&problem, &initial_values)?;

        // Extract final values
        let x1_final = result
            .parameters
            .get("x1")
            .ok_or("x1 not found")?
            .to_vector()[0];
        let x2_final = result
            .parameters
            .get("x2")
            .ok_or("x2 not found")?
            .to_vector()[0];

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
            params: &[nalgebra::DVector<f64>],
            compute_jacobian: bool,
        ) -> (nalgebra::DVector<f64>, Option<nalgebra::DMatrix<f64>>) {
            let residual = dvector![params[0][0] - self.target];
            let jacobian = if compute_jacobian {
                Some(nalgebra::DMatrix::from_element(1, 1, 1.0))
            } else {
                None
            };
            (residual, jacobian)
        }

        fn get_dimension(&self) -> usize {
            1
        }
    }

    fn rosenbrock_problem() -> (
        Problem,
        HashMap<String, (apex_manifolds::ManifoldType, nalgebra::DVector<f64>)>,
    ) {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let mut initial_values = HashMap::new();
        initial_values.insert("x1".to_string(), (ManifoldType::RN, dvector![-1.2]));
        initial_values.insert("x2".to_string(), (ManifoldType::RN, dvector![1.0]));
        problem.add_residual_block(&["x1", "x2"], Box::new(RosenbrockFactor1), None);
        problem.add_residual_block(&["x1"], Box::new(RosenbrockFactor2), None);
        (problem, initial_values)
    }

    fn linear_problem(
        start: f64,
    ) -> (
        Problem,
        HashMap<String, (apex_manifolds::ManifoldType, nalgebra::DVector<f64>)>,
    ) {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let mut initial_values = HashMap::new();
        initial_values.insert("x".to_string(), (ManifoldType::RN, dvector![start]));
        problem.add_residual_block(&["x"], Box::new(LinearFactor { target: 0.0 }), None);
        (problem, initial_values)
    }

    // -------------------------------------------------------------------------
    // Config builder tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_lm_config_default() {
        let cfg = LevenbergMarquardtConfig::default();
        assert_eq!(cfg.max_iterations, 50);
        assert!((cfg.cost_tolerance - 1e-6).abs() < 1e-15);
        assert!((cfg.damping - 1e-3).abs() < 1e-15);
        assert!(!cfg.use_jacobi_scaling);
        assert!(!cfg.compute_covariances);
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
            .with_trust_region(500.0, 0.1, 0.8)
            .with_min_cost_threshold(1e-12)
            .with_min_trust_region_radius(1e-35)
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
        assert!((cfg.trust_region_radius - 500.0).abs() < 1e-10);
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
        let (problem, initial_values) = rosenbrock_problem();
        let cfg = LevenbergMarquardtConfig::new().with_max_iterations(2);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&problem, &initial_values)?;
        assert_eq!(result.status, OptimizationStatus::MaxIterationsReached);
        assert!(result.iterations <= 3, "iterations={}", result.iterations);
        Ok(())
    }

    #[test]
    fn test_lm_gradient_tolerance_convergence() -> TestResult {
        let (problem, initial_values) = linear_problem(1.0);
        // Very loose gradient tolerance → triggers after first accepted step
        let cfg = LevenbergMarquardtConfig::new()
            .with_gradient_tolerance(1e3)
            .with_cost_tolerance(1e-20)
            .with_parameter_tolerance(1e-20);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&problem, &initial_values)?;
        assert_eq!(result.status, OptimizationStatus::GradientToleranceReached);
        Ok(())
    }

    #[test]
    fn test_lm_min_cost_threshold() -> TestResult {
        let (problem, initial_values) = rosenbrock_problem();
        // Set threshold very high so even initial cost triggers it
        let cfg = LevenbergMarquardtConfig::new()
            .with_min_cost_threshold(1e10)
            .with_cost_tolerance(1e-20)
            .with_gradient_tolerance(1e-20)
            .with_parameter_tolerance(1e-20);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&problem, &initial_values)?;
        assert_eq!(result.status, OptimizationStatus::MinCostThresholdReached);
        Ok(())
    }

    #[test]
    fn test_lm_qr_solver() -> TestResult {
        let (problem, initial_values) = rosenbrock_problem();
        let cfg = LevenbergMarquardtConfig::new()
            .with_linear_solver_type(LinearSolverType::SparseQR)
            .with_max_iterations(100);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&problem, &initial_values)?;
        assert!(result.final_cost < 1e-6);
        Ok(())
    }

    #[test]
    fn test_lm_jacobi_scaling_enabled() -> TestResult {
        let (problem, initial_values) = rosenbrock_problem();
        let cfg = LevenbergMarquardtConfig::new()
            .with_jacobi_scaling(true)
            .with_max_iterations(100);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&problem, &initial_values)?;
        assert!(result.final_cost < 1e-6);
        Ok(())
    }

    #[test]
    fn test_lm_result_initial_cost_greater_than_final() -> TestResult {
        let (problem, initial_values) = rosenbrock_problem();
        let mut solver = LevenbergMarquardt::new();
        let result = solver.optimize(&problem, &initial_values)?;
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
        let (problem, initial_values) = rosenbrock_problem();
        let mut solver = LevenbergMarquardt::new();
        let result = solver.optimize(&problem, &initial_values)?;
        assert!(result.convergence_info.is_some());
        Ok(())
    }

    #[test]
    fn test_lm_iterations_positive() -> TestResult {
        let (problem, initial_values) = rosenbrock_problem();
        let mut solver = LevenbergMarquardt::new();
        let result = solver.optimize(&problem, &initial_values)?;
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
        let mut initial_values = HashMap::new();
        initial_values.insert("x1".to_string(), (ManifoldType::RN, dvector![-1.2]));
        initial_values.insert("x2".to_string(), (ManifoldType::RN, dvector![1.0]));
        problem.add_residual_block(&["x1", "x2"], Box::new(RosenbrockFactor1), None);
        problem.add_residual_block(&["x1"], Box::new(RosenbrockFactor2), None);

        // Default linear solver type (SparseCholesky) with Dense mode → DenseCholeskySolver
        let cfg = LevenbergMarquardtConfig::new().with_max_iterations(100);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&problem, &initial_values)?;
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
        let mut initial_values = HashMap::new();
        initial_values.insert("x1".to_string(), (ManifoldType::RN, dvector![-1.2]));
        initial_values.insert("x2".to_string(), (ManifoldType::RN, dvector![1.0]));
        problem.add_residual_block(&["x1", "x2"], Box::new(RosenbrockFactor1), None);
        problem.add_residual_block(&["x1"], Box::new(RosenbrockFactor2), None);

        let cfg = LevenbergMarquardtConfig::new()
            .with_linear_solver_type(LinearSolverType::DenseQR)
            .with_max_iterations(100);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&problem, &initial_values)?;
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
        let (problem, initial_values) = rosenbrock_problem();
        let cfg = LevenbergMarquardtConfig::new()
            .with_max_iterations(100)
            .with_compute_covariances(true);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let result = solver.optimize(&problem, &initial_values)?;
        // result.covariances is Option<HashMap<String, Mat<f64>>>
        assert!(
            result.covariances.is_some(),
            "compute_covariances=true should populate result.covariances"
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // update_damping() direct unit tests
    // -------------------------------------------------------------------------

    /// `update_damping(rho > 0)` should accept the step, decrease damping, and reset nu.
    #[test]
    fn test_update_damping_accepted_step() {
        let cfg = LevenbergMarquardtConfig::new()
            .with_damping(1e-2)
            .with_damping_bounds(1e-15, 1e15);
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let initial_damping = solver.config.damping;

        // rho = 0.8 > 0 → accepted branch
        let accepted = solver.update_damping(0.8);

        assert!(accepted, "rho > 0 should return true (step accepted)");
        assert!(
            solver.config.damping < initial_damping,
            "accepted step should decrease damping: {} < {}",
            solver.config.damping,
            initial_damping
        );
        // damping_nu should be reset to 2.0 on acceptance
        assert!(
            (solver.config.damping_nu - 2.0).abs() < 1e-15,
            "damping_nu should be reset to 2.0 after accepted step, got {}",
            solver.config.damping_nu
        );
    }

    /// `update_damping(rho <= 0)` should reject the step, increase damping, and double nu.
    #[test]
    fn test_update_damping_rejected_step() {
        let cfg = LevenbergMarquardtConfig::new()
            .with_damping(1e-2)
            .with_damping_bounds(1e-15, 1e15);
        let initial_nu = cfg.damping_nu; // default 2.0
        let mut solver = LevenbergMarquardt::with_config(cfg);
        let initial_damping = solver.config.damping;

        // rho = -0.5 <= 0 → rejected branch
        let rejected = solver.update_damping(-0.5);

        assert!(!rejected, "rho <= 0 should return false (step rejected)");
        assert!(
            solver.config.damping > initial_damping,
            "rejected step should increase damping: {} > {}",
            solver.config.damping,
            initial_damping
        );
        // damping_nu doubles on rejection
        assert!(
            (solver.config.damping_nu - initial_nu * 2.0).abs() < 1e-15,
            "damping_nu should double on rejected step: expected {}, got {}",
            initial_nu * 2.0,
            solver.config.damping_nu
        );
    }

    // -------------------------------------------------------------------------
    // Untested config builder methods
    // -------------------------------------------------------------------------

    /// Verifies `with_max_condition_number` and `with_min_relative_decrease` builder methods.
    #[test]
    fn test_lm_config_condition_number_and_relative_decrease() {
        let cfg = LevenbergMarquardtConfig::new()
            .with_max_condition_number(1e8)
            .with_min_relative_decrease(1e-4);
        assert!(cfg.max_condition_number.is_some());
        assert!((cfg.max_condition_number.unwrap() - 1e8).abs() < 1.0);
        assert!((cfg.min_relative_decrease - 1e-4).abs() < 1e-20);
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
            fn on_step(&self, _values: &HashMap<String, VariableEnum>, _iteration: usize) {}

            fn on_optimization_complete(
                &self,
                _values: &HashMap<String, VariableEnum>,
                _iterations: usize,
            ) {
                *self.complete_calls.lock().unwrap() += 1;
            }
        }

        let call_count = Arc::new(Mutex::new(0usize));
        let observer = CountObserver {
            complete_calls: Arc::clone(&call_count),
        };

        let (problem, initial_values) = rosenbrock_problem();
        let mut solver = LevenbergMarquardt::new();
        solver.add_observer(observer);
        let _ = solver.optimize(&problem, &initial_values)?;

        assert_eq!(
            *call_count.lock().unwrap(),
            1,
            "on_optimization_complete should be called exactly once"
        );
        Ok(())
    }
}
