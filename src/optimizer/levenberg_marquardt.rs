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
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut problem = Problem::new();
//! // ... add residual blocks (factors) to problem ...
//!
//! let initial_values = HashMap::new();
//! // ... initialize parameters ...
//!
//! let mut solver = LevenbergMarquardt::new();
//! let result = solver.optimize(&problem, &initial_values)?;
//!
//! println!("Status: {:?}", result.status);
//! println!("Final cost: {:.6e}", result.final_cost);
//! println!("Iterations: {}", result.iterations);
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

use crate::core::problem::{Problem, SymbolicStructure, VariableEnum};
use crate::error;
use crate::linalg::{LinearSolverType, SparseCholeskySolver, SparseLinearSolver, SparseQRSolver};
use crate::manifold::ManifoldType;
use crate::optimizer::{
    ConvergenceInfo, OptimizationStatus, Solver, SolverResult, apply_negative_parameter_step,
    apply_parameter_step, compute_cost,
};

#[cfg(feature = "visualization")]
use crate::optimizer::OptimizationVisualizer;
use faer::{
    Mat,
    sparse::{SparseColMat, Triplet},
};
use nalgebra::DVector;
use std::collections::HashMap;
use std::{
    fmt,
    fmt::{Display, Formatter},
    time::{Duration, Instant},
};
use tracing::{debug, info};

/// Summary statistics for the Levenberg-Marquardt optimization process.
#[derive(Debug, Clone)]
pub struct LevenbergMarquardtSummary {
    /// Initial cost value
    pub initial_cost: f64,
    /// Final cost value
    pub final_cost: f64,
    /// Ratio of actual to predicted reduction in cost
    pub rho: f64,
    /// Total number of iterations performed
    pub iterations: usize,
    /// Number of successful steps (cost decreased)
    pub successful_steps: usize,
    /// Number of unsuccessful steps (cost increased, damping increased)
    pub unsuccessful_steps: usize,
    /// Final damping parameter value
    pub final_damping: f64,
    /// Average cost reduction per iteration
    pub average_cost_reduction: f64,
    /// Maximum gradient norm encountered
    pub max_gradient_norm: f64,
    /// Final gradient norm
    pub final_gradient_norm: f64,
    /// Maximum parameter update norm
    pub max_parameter_update_norm: f64,
    /// Final parameter update norm
    pub final_parameter_update_norm: f64,
    /// Total time elapsed
    pub total_time: Duration,
    /// Average time per iteration
    pub average_time_per_iteration: Duration,
    /// Detailed per-iteration statistics history
    pub iteration_history: Vec<IterationStats>,
    /// Convergence status
    pub convergence_status: OptimizationStatus,
}

impl Display for LevenbergMarquardtSummary {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Determine if converged
        let converged = matches!(
            self.convergence_status,
            OptimizationStatus::Converged
                | OptimizationStatus::CostToleranceReached
                | OptimizationStatus::GradientToleranceReached
                | OptimizationStatus::ParameterToleranceReached
        );

        writeln!(f, "=== Levenberg-Marquardt Final Result ===")?;

        // Title with convergence status
        if converged {
            writeln!(f, "CONVERGED ({:?})", self.convergence_status)?;
        } else {
            writeln!(f, "DIVERGED ({:?})", self.convergence_status)?;
        }

        writeln!(f)?;
        writeln!(f, "Cost:")?;
        writeln!(f, "  Initial:   {:.6e}", self.initial_cost)?;
        writeln!(f, "  Final:     {:.6e}", self.final_cost)?;
        writeln!(
            f,
            "  Reduction: {:.6e} ({:.2}%)",
            self.initial_cost - self.final_cost,
            100.0 * (self.initial_cost - self.final_cost) / self.initial_cost.max(1e-12)
        )?;
        writeln!(f)?;
        writeln!(f, "Iterations:")?;
        writeln!(f, "  Total:              {}", self.iterations)?;
        writeln!(
            f,
            "  Successful steps:   {} ({:.1}%)",
            self.successful_steps,
            100.0 * self.successful_steps as f64 / self.iterations.max(1) as f64
        )?;
        writeln!(
            f,
            "  Unsuccessful steps: {} ({:.1}%)",
            self.unsuccessful_steps,
            100.0 * self.unsuccessful_steps as f64 / self.iterations.max(1) as f64
        )?;
        writeln!(f)?;
        writeln!(f, "Gradient:")?;
        writeln!(f, "  Max norm:   {:.2e}", self.max_gradient_norm)?;
        writeln!(f, "  Final norm: {:.2e}", self.final_gradient_norm)?;
        writeln!(f)?;
        writeln!(f, "Parameter Update:")?;
        writeln!(f, "  Max norm:   {:.2e}", self.max_parameter_update_norm)?;
        writeln!(f, "  Final norm: {:.2e}", self.final_parameter_update_norm)?;
        writeln!(f)?;
        writeln!(f, "Performance:")?;
        writeln!(
            f,
            "  Total time:             {:.2}ms",
            self.total_time.as_secs_f64() * 1000.0
        )?;
        writeln!(
            f,
            "  Average per iteration:  {:.2}ms",
            self.average_time_per_iteration.as_secs_f64() * 1000.0
        )?;

        Ok(())
    }
}

/// Per-iteration statistics for detailed logging (Ceres-style output).
///
/// Captures all relevant metrics for each optimization iteration, enabling
/// detailed analysis and debugging of the optimization process.
#[derive(Debug, Clone)]
pub struct IterationStats {
    /// Iteration number (0-indexed)
    pub iteration: usize,
    /// Cost function value at this iteration
    pub cost: f64,
    /// Change in cost from previous iteration
    pub cost_change: f64,
    /// L2 norm of the gradient (||J^T·r||)
    pub gradient_norm: f64,
    /// L2 norm of the parameter update step (||Δx||)
    pub step_norm: f64,
    /// Trust region ratio (ρ = actual_reduction / predicted_reduction)
    pub tr_ratio: f64,
    /// Trust region radius (damping parameter λ)
    pub tr_radius: f64,
    /// Linear solver iterations (0 for direct solvers like Cholesky)
    pub ls_iter: usize,
    /// Time taken for this iteration in milliseconds
    pub iter_time_ms: f64,
    /// Total elapsed time since optimization started in milliseconds
    pub total_time_ms: f64,
    /// Whether the step was accepted (true) or rejected (false)
    pub accepted: bool,
}

impl IterationStats {
    /// Print table header in Ceres-style format
    pub fn print_header() {
        debug!(
            "{:>4}  {:>13}  {:>13}  {:>13}  {:>13}  {:>11}  {:>11}  {:>7}  {:>11}  {:>13}  {:>6}",
            "iter",
            "cost",
            "cost_change",
            "|gradient|",
            "|step|",
            "tr_ratio",
            "tr_radius",
            "ls_iter",
            "iter_time",
            "total_time",
            "status"
        );
    }

    /// Print single iteration line in Ceres-style format with scientific notation
    pub fn print_line(&self) {
        let status = if self.iteration == 0 {
            "-"
        } else if self.accepted {
            "✓"
        } else {
            "✗"
        };

        debug!(
            "{:>4}  {:>13.6e}  {:>13.2e}  {:>13.2e}  {:>13.2e}  {:>11.2e}  {:>11.2e}  {:>7}  {:>9.2}ms  {:>11.2}ms  {:>6}",
            self.iteration,
            self.cost,
            self.cost_change,
            self.gradient_norm,
            self.step_norm,
            self.tr_ratio,
            self.tr_radius,
            self.ls_iter,
            self.iter_time_ms,
            self.total_time_ms,
            status
        );
    }
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
    /// Enable real-time Rerun visualization (graphical debugging)
    ///
    /// When enabled, logs optimization progress to Rerun viewer including:
    /// - Time series plots (cost, gradient norm, damping, step quality)
    /// - Sparse Hessian heat map visualization
    /// - Gradient vector visualization
    /// - Manifold state updates (for SE2/SE3 problems)
    ///
    /// **Note:** Requires the `visualization` feature to be enabled in `Cargo.toml`.
    /// Use `verbose` for terminal output; this is for graphical visualization.
    ///
    /// Default: false
    #[cfg(feature = "visualization")]
    pub enable_visualization: bool,
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
            damping: 1e-4,
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
            use_jacobi_scaling: false,
            compute_covariances: false,
            #[cfg(feature = "visualization")]
            enable_visualization: false,
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
    #[cfg(feature = "visualization")]
    pub fn with_visualization(mut self, enable: bool) -> Self {
        self.enable_visualization = enable;
        self
    }

    /// Print configuration parameters (verbose mode only)
    pub fn print_configuration(&self) {
        info!("Configuration:");
        info!("  Solver:        Levenberg-Marquardt");
        info!("  Linear solver: {:?}", self.linear_solver_type);
        info!("  Convergence Criteria:");
        info!("  Max iterations:      {}", self.max_iterations);
        info!("  Cost tolerance:      {:.2e}", self.cost_tolerance);
        info!("  Parameter tolerance: {:.2e}", self.parameter_tolerance);
        info!("  Gradient tolerance:  {:.2e}", self.gradient_tolerance);
        info!("  Timeout:             {:?}", self.timeout);
        info!("  Damping Parameters:");
        info!("  Initial damping:     {:.2e}", self.damping);
        info!(
            "  Damping range:       [{:.2e}, {:.2e}]",
            self.damping_min, self.damping_max
        );
        info!("  Increase factor:     {:.2}", self.damping_increase_factor);
        info!("  Decrease factor:     {:.2}", self.damping_decrease_factor);
        info!("  Trust Region:");
        info!("  Initial radius:      {:.2e}", self.trust_region_radius);
        info!("  Min step quality:    {:.2}", self.min_step_quality);
        info!("  Good step quality:   {:.2}", self.good_step_quality);
        info!("  Numerical Settings:");
        info!(
            "  Jacobi scaling:      {}",
            if self.use_jacobi_scaling {
                "enabled"
            } else {
                "disabled"
            }
        );
        info!(
            "  Compute covariances: {}",
            if self.compute_covariances {
                "enabled"
            } else {
                "disabled"
            }
        );
    }
}

/// State for optimization iteration
struct LinearizerResult {
    variables: HashMap<String, VariableEnum>,
    variable_index_map: HashMap<String, usize>,
    sorted_vars: Vec<String>,
    symbolic_structure: SymbolicStructure,
    current_cost: f64,
    initial_cost: f64,
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
    jacobi_scaling: Option<SparseColMat<usize, f64>>,
    #[cfg(feature = "visualization")]
    visualizer: Option<OptimizationVisualizer>,
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
        // Create visualizer if enabled (zero overhead when disabled)
        #[cfg(feature = "visualization")]
        let visualizer = if config.enable_visualization {
            match OptimizationVisualizer::new(true) {
                Ok(vis) => Some(vis),
                Err(e) => {
                    eprintln!("[WARNING] Failed to create visualizer: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Self {
            config,
            jacobi_scaling: None,
            #[cfg(feature = "visualization")]
            visualizer,
        }
    }

    /// Create the appropriate linear solver based on configuration
    fn create_linear_solver(&self) -> Box<dyn SparseLinearSolver> {
        match self.config.linear_solver_type {
            LinearSolverType::SparseCholesky => Box::new(SparseCholeskySolver::new()),
            LinearSolverType::SparseQR => Box::new(SparseQRSolver::new()),
        }
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

    /// Compute step quality ratio (actual vs predicted reduction)
    /// Reference: Introduction to Optimization and Data Fitting
    /// Reference: Damping parameter in marquardt's method
    /// Formula 2.2
    fn compute_step_quality(
        &self,
        current_cost: f64,
        new_cost: f64,
        predicted_reduction: f64,
    ) -> f64 {
        let actual_reduction = current_cost - new_cost;
        if predicted_reduction.abs() < 1e-15 {
            if actual_reduction > 0.0 { 1.0 } else { 0.0 }
        } else {
            actual_reduction / predicted_reduction
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

    /// Check convergence criteria
    /// Check convergence using comprehensive termination criteria.
    ///
    /// Implements 9 termination criteria following Ceres Solver standards:
    ///
    /// 1. **Gradient Norm (First-Order Optimality)**: ||g||∞ ≤ gradient_tolerance
    /// 2. **Parameter Change Tolerance**: ||h|| ≤ parameter_tolerance * (||x|| + parameter_tolerance)
    /// 3. **Function Value Change Tolerance**: |ΔF| < cost_tolerance * F
    /// 4. **Objective Function Cutoff**: F_new < min_cost_threshold (optional)
    /// 5. **Trust Region Radius**: radius < min_trust_region_radius
    /// 6. **Singular/Ill-Conditioned Jacobian**: Detected during linear solve
    /// 7. **Invalid Numerical Values**: NaN or Inf in cost or parameters
    /// 8. **Maximum Iterations**: iteration >= max_iterations
    /// 9. **Timeout**: elapsed >= timeout
    ///
    /// # Arguments
    ///
    /// * `iteration` - Current iteration number
    /// * `current_cost` - Cost before applying the step
    /// * `new_cost` - Cost after applying the step
    /// * `parameter_norm` - L2 norm of current parameter vector ||x||
    /// * `parameter_update_norm` - L2 norm of parameter update step ||h||
    /// * `gradient_norm` - Infinity norm of gradient ||g||∞
    /// * `trust_region_radius` - Current trust region radius
    /// * `elapsed` - Elapsed time since optimization start
    /// * `step_accepted` - Whether the current step was accepted
    ///
    /// # Returns
    ///
    /// `Some(OptimizationStatus)` if any termination criterion is satisfied, `None` otherwise.
    #[allow(clippy::too_many_arguments)]
    fn check_convergence(
        &self,
        iteration: usize,
        current_cost: f64,
        new_cost: f64,
        parameter_norm: f64,
        parameter_update_norm: f64,
        gradient_norm: f64,
        trust_region_radius: f64,
        elapsed: Duration,
        step_accepted: bool,
    ) -> Option<OptimizationStatus> {
        // ========================================================================
        // CRITICAL SAFETY CHECKS (perform first, before convergence checks)
        // ========================================================================

        // CRITERION 7: Invalid Numerical Values (NaN/Inf)
        // Always check for numerical instability first
        if !new_cost.is_finite() || !parameter_update_norm.is_finite() || !gradient_norm.is_finite()
        {
            return Some(OptimizationStatus::InvalidNumericalValues);
        }

        // CRITERION 9: Timeout
        // Check wall-clock time limit
        if let Some(timeout) = self.config.timeout
            && elapsed >= timeout
        {
            return Some(OptimizationStatus::Timeout);
        }

        // CRITERION 8: Maximum Iterations
        // Check iteration count limit
        if iteration >= self.config.max_iterations {
            return Some(OptimizationStatus::MaxIterationsReached);
        }

        // ========================================================================
        // CONVERGENCE CRITERIA (only check after successful steps)
        // ========================================================================

        // Only check convergence criteria after accepted steps
        // (rejected steps don't indicate convergence)
        if !step_accepted {
            return None;
        }

        // CRITERION 1: Gradient Norm (First-Order Optimality)
        // Check if gradient infinity norm is below threshold
        // This indicates we're at a critical point (local minimum, saddle, or maximum)
        if gradient_norm < self.config.gradient_tolerance {
            return Some(OptimizationStatus::GradientToleranceReached);
        }

        // Only check parameter and cost criteria after first iteration
        if iteration > 0 {
            // CRITERION 2: Parameter Change Tolerance (xtol)
            // Ceres formula: ||h|| ≤ ε_param * (||x|| + ε_param)
            // This is a relative measure that scales with parameter magnitude
            let relative_step_tolerance = self.config.parameter_tolerance
                * (parameter_norm + self.config.parameter_tolerance);

            if parameter_update_norm <= relative_step_tolerance {
                return Some(OptimizationStatus::ParameterToleranceReached);
            }

            // CRITERION 3: Function Value Change Tolerance (ftol)
            // Ceres formula: |ΔF| < ε_cost * F
            // Check relative cost change (not absolute)
            let cost_change = (current_cost - new_cost).abs();
            let relative_cost_change = cost_change / current_cost.max(1e-10); // Avoid division by zero

            if relative_cost_change < self.config.cost_tolerance {
                return Some(OptimizationStatus::CostToleranceReached);
            }
        }

        // CRITERION 4: Objective Function Cutoff (optional early stopping)
        // Useful for "good enough" solutions
        if let Some(min_cost) = self.config.min_cost_threshold
            && new_cost < min_cost
        {
            return Some(OptimizationStatus::MinCostThresholdReached);
        }

        // CRITERION 5: Trust Region Radius
        // If trust region has collapsed, optimization has converged or problem is ill-conditioned
        if trust_region_radius < self.config.min_trust_region_radius {
            return Some(OptimizationStatus::TrustRegionRadiusTooSmall);
        }

        // CRITERION 6: Singular/Ill-Conditioned Jacobian
        // Note: This is typically detected during the linear solve and handled there
        // The max_condition_number check would be expensive to compute here
        // If linear solve fails, it returns an error that's converted to NumericalFailure

        // No termination criterion satisfied
        None
    }

    /// Compute total parameter vector norm ||x||.
    ///
    /// Computes the L2 norm of all parameter vectors concatenated together.
    /// This is used in the relative parameter tolerance check.
    ///
    /// # Arguments
    ///
    /// * `variables` - Map of variable names to their current values
    ///
    /// # Returns
    ///
    /// The L2 norm of the concatenated parameter vector
    fn compute_parameter_norm(variables: &HashMap<String, VariableEnum>) -> f64 {
        variables
            .values()
            .map(|v| {
                let vec = v.to_vector();
                vec.norm_squared()
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Create Jacobi scaling matrix from Jacobian
    fn create_jacobi_scaling(
        &self,
        jacobian: &SparseColMat<usize, f64>,
    ) -> SparseColMat<usize, f64> {
        let cols = jacobian.ncols();
        let jacobi_scaling_vec: Vec<Triplet<usize, usize, f64>> = (0..cols)
            .map(|c| {
                // Compute column norm: sqrt(sum(J_col^2))
                let col_norm_squared: f64 = jacobian
                    .triplet_iter()
                    .filter(|t| t.col == c)
                    .map(|t| t.val * t.val)
                    .sum();
                let col_norm = col_norm_squared.sqrt();
                // Scaling factor: 1.0 / (1.0 + col_norm)
                let scaling = 1.0 / (1.0 + col_norm);
                Triplet::new(c, c, scaling)
            })
            .collect();

        SparseColMat::try_new_from_triplets(cols, cols, &jacobi_scaling_vec)
            .expect("Failed to create Jacobi scaling matrix")
    }

    /// Initialize optimization state from problem and initial parameters
    fn initialize_optimization_state(
        &self,
        problem: &Problem,
        initial_params: &HashMap<String, (ManifoldType, DVector<f64>)>,
    ) -> Result<LinearizerResult, error::ApexError> {
        // Initialize variables from initial values
        let variables = problem.initialize_variables(initial_params);

        // Create column mapping for variables
        let mut variable_index_map = HashMap::new();
        let mut col_offset = 0;
        let mut sorted_vars: Vec<String> = variables.keys().cloned().collect();
        sorted_vars.sort();

        for var_name in &sorted_vars {
            variable_index_map.insert(var_name.clone(), col_offset);
            col_offset += variables[var_name].get_size();
        }

        // Build symbolic structure for sparse operations
        let symbolic_structure =
            problem.build_symbolic_structure(&variables, &variable_index_map, col_offset)?;

        // Initial cost evaluation (residual only, no Jacobian needed)
        let residual = problem.compute_residual_sparse(&variables)?;
        let current_cost = compute_cost(&residual);
        let initial_cost = current_cost;

        Ok(LinearizerResult {
            variables,
            variable_index_map,
            sorted_vars,
            symbolic_structure,
            current_cost,
            initial_cost,
        })
    }

    /// Process Jacobian by creating and applying Jacobi scaling if enabled
    fn process_jacobian(
        &mut self,
        jacobian: &SparseColMat<usize, f64>,
        iteration: usize,
    ) -> SparseColMat<usize, f64> {
        // Create Jacobi scaling on first iteration if enabled
        if iteration == 0 {
            let scaling = self.create_jacobi_scaling(jacobian);
            self.jacobi_scaling = Some(scaling);
        }
        jacobian * self.jacobi_scaling.as_ref().unwrap()
    }

    /// Compute optimization step by solving the augmented system
    fn compute_levenberg_marquardt_step(
        &self,
        residuals: &Mat<f64>,
        scaled_jacobian: &SparseColMat<usize, f64>,
        linear_solver: &mut Box<dyn SparseLinearSolver>,
    ) -> Option<StepResult> {
        // Solve augmented equation: (J_scaled^T * J_scaled + λI) * dx_scaled = -J_scaled^T * r
        let residuals_owned = residuals.as_ref().to_owned();
        let scaled_step = linear_solver
            .solve_augmented_equation(&residuals_owned, scaled_jacobian, self.config.damping)
            .ok()?;

        // Get cached gradient and Hessian from the solver
        let gradient = linear_solver.get_gradient()?;
        let _hessian = linear_solver.get_hessian()?;
        let gradient_norm = gradient.norm_l2();

        // Apply inverse Jacobi scaling to get final step (if enabled)
        let step = if self.config.use_jacobi_scaling {
            &scaled_step * self.jacobi_scaling.as_ref().unwrap()
        } else {
            scaled_step
        };

        // Compute predicted reduction using scaled values
        let predicted_reduction = self.compute_predicted_reduction(&step, gradient);

        Some(StepResult {
            step,
            gradient_norm,
            predicted_reduction,
        })
    }

    /// Evaluate and apply step, handling acceptance/rejection based on step quality
    fn evaluate_and_apply_step(
        &mut self,
        step_result: &StepResult,
        state: &mut LinearizerResult,
        problem: &Problem,
    ) -> error::ApexResult<StepEvaluation> {
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
        let rho = self.compute_step_quality(
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

    /// Create optimization summary
    #[allow(clippy::too_many_arguments)]
    fn create_summary(
        &self,
        initial_cost: f64,
        final_cost: f64,
        rho: f64,
        iterations: usize,
        successful_steps: usize,
        unsuccessful_steps: usize,
        max_gradient_norm: f64,
        final_gradient_norm: f64,
        max_parameter_update_norm: f64,
        final_parameter_update_norm: f64,
        total_cost_reduction: f64,
        total_time: Duration,
        iteration_history: Vec<IterationStats>,
        convergence_status: OptimizationStatus,
    ) -> LevenbergMarquardtSummary {
        LevenbergMarquardtSummary {
            initial_cost,
            final_cost,
            rho,
            iterations,
            successful_steps,
            unsuccessful_steps,
            final_damping: self.config.damping,
            average_cost_reduction: if iterations > 0 {
                total_cost_reduction / iterations as f64
            } else {
                0.0
            },
            max_gradient_norm,
            final_gradient_norm,
            max_parameter_update_norm,
            final_parameter_update_norm,
            total_time,
            average_time_per_iteration: if iterations > 0 {
                total_time / iterations as u32
            } else {
                Duration::from_secs(0)
            },
            iteration_history,
            convergence_status,
        }
    }

    pub fn optimize(
        &mut self,
        problem: &Problem,
        initial_params: &HashMap<String, (ManifoldType, DVector<f64>)>,
    ) -> Result<SolverResult<HashMap<String, VariableEnum>>, error::ApexError> {
        let start_time = Instant::now();
        let mut iteration = 0;
        let mut cost_evaluations = 1; // Initial cost evaluation
        let mut jacobian_evaluations = 0;
        let mut successful_steps = 0;
        let mut unsuccessful_steps = 0;

        // Initialize optimization state
        let mut state = self.initialize_optimization_state(problem, initial_params)?;

        // Create linear solver
        let mut linear_solver = self.create_linear_solver();

        // Initialize summary tracking variables
        let mut max_gradient_norm: f64 = 0.0;
        let mut max_parameter_update_norm: f64 = 0.0;
        let mut total_cost_reduction = 0.0;
        let mut final_gradient_norm;
        let mut final_parameter_update_norm;

        // Initialize iteration statistics tracking
        let mut iteration_stats = Vec::with_capacity(self.config.max_iterations);
        let mut previous_cost = state.current_cost;

        // Print configuration and header if info level is enabled
        if tracing::enabled!(tracing::Level::INFO) {
            self.config.print_configuration();
        }
        if tracing::enabled!(tracing::Level::DEBUG) {
            IterationStats::print_header();
        }

        // Main optimization loop
        loop {
            let iter_start = Instant::now();
            // Evaluate residuals and Jacobian
            let (residuals, jacobian) = problem.compute_residual_and_jacobian_sparse(
                &state.variables,
                &state.variable_index_map,
                &state.symbolic_structure,
            )?;
            jacobian_evaluations += 1;

            // Process Jacobian (apply scaling if enabled)
            let scaled_jacobian = if self.config.use_jacobi_scaling {
                self.process_jacobian(&jacobian, iteration)
            } else {
                jacobian
            };

            // Compute optimization step
            let step_result = match self.compute_levenberg_marquardt_step(
                &residuals,
                &scaled_jacobian,
                &mut linear_solver,
            ) {
                Some(result) => result,
                None => {
                    return Err(error::ApexError::Solver(
                        "Linear solver failed to solve augmented system".to_string(),
                    ));
                }
            };

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
            // This eliminates ~2-5ms overhead per iteration for non-debug optimization
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
                    ls_iter: 0, // Direct solver (Cholesky) has no iterations
                    iter_time_ms: iter_elapsed_ms,
                    total_time_ms: total_elapsed_ms,
                    accepted: step_eval.accepted,
                };

                iteration_stats.push(stats.clone());
                stats.print_line();
            }

            previous_cost = state.current_cost;

            // Rerun visualization
            #[cfg(feature = "visualization")]
            if let Some(ref vis) = self.visualizer {
                if let Err(e) = vis.log_scalars(
                    iteration,
                    state.current_cost,
                    step_result.gradient_norm,
                    self.config.damping,
                    step_norm,
                    Some(step_eval.rho),
                ) {
                    eprintln!("[WARNING] Failed to log scalars: {}", e);
                }

                // Log expensive visualizations (Hessian, gradient, manifolds)
                if let Err(e) = vis.log_hessian(linear_solver.get_hessian(), iteration) {
                    eprintln!("[WARNING] Failed to log Hessian: {}", e);
                }
                if let Err(e) = vis.log_gradient(linear_solver.get_gradient(), iteration) {
                    eprintln!("[WARNING] Failed to log gradient: {}", e);
                }
                if let Err(e) = vis.log_manifolds(&state.variables, iteration) {
                    eprintln!("[WARNING] Failed to log manifolds: {}", e);
                }
            }

            // Check convergence
            let elapsed = start_time.elapsed();

            // Compute parameter norm for relative parameter tolerance check
            let parameter_norm = Self::compute_parameter_norm(&state.variables);

            // Compute new cost for convergence check (state may already have new cost if step accepted)
            let new_cost = if step_eval.accepted {
                state.current_cost
            } else {
                // Use cost before step application
                state.current_cost
            };

            // Cost before this step (need to add back reduction if step was accepted)
            let cost_before_step = if step_eval.accepted {
                state.current_cost + step_eval.cost_reduction
            } else {
                state.current_cost
            };

            if let Some(status) = self.check_convergence(
                iteration,
                cost_before_step,
                new_cost,
                parameter_norm,
                step_norm,
                step_result.gradient_norm,
                self.config.trust_region_radius,
                elapsed,
                step_eval.accepted,
            ) {
                let summary = self.create_summary(
                    state.initial_cost,
                    state.current_cost,
                    step_eval.rho,
                    iteration + 1,
                    successful_steps,
                    unsuccessful_steps,
                    max_gradient_norm,
                    final_gradient_norm,
                    max_parameter_update_norm,
                    final_parameter_update_norm,
                    total_cost_reduction,
                    elapsed,
                    iteration_stats.clone(),
                    status.clone(),
                );

                // Print summary only if info level is enabled
                if tracing::enabled!(tracing::Level::INFO) {
                    info!("\n{}", summary);
                }

                // Log convergence to Rerun
                #[cfg(feature = "visualization")]
                if let Some(ref vis) = self.visualizer {
                    let _ = vis.log_convergence(&format!("Converged: {}", status));
                }

                // Compute covariances if enabled
                let covariances = if self.config.compute_covariances {
                    problem.compute_and_set_covariances(
                        &mut linear_solver,
                        &mut state.variables,
                        &state.variable_index_map,
                    )
                } else {
                    None
                };

                return Ok(SolverResult {
                    status,
                    iterations: iteration + 1,
                    initial_cost: state.initial_cost,
                    final_cost: state.current_cost,
                    parameters: state.variables.into_iter().collect(),
                    elapsed_time: elapsed,
                    convergence_info: Some(ConvergenceInfo {
                        final_gradient_norm,
                        final_parameter_update_norm,
                        cost_evaluations,
                        jacobian_evaluations,
                    }),
                    covariances,
                });
            }

            // Note: Max iterations and timeout checks are now handled inside check_convergence()

            iteration += 1;
        }
    }
}
// Implement Solver trait
impl Solver for LevenbergMarquardt {
    type Config = LevenbergMarquardtConfig;
    type Error = error::ApexError;

    fn new() -> Self {
        Self::default()
    }

    fn optimize(
        &mut self,
        problem: &Problem,
        initial_params: &HashMap<String, (ManifoldType, DVector<f64>)>,
    ) -> Result<SolverResult<HashMap<String, VariableEnum>>, Self::Error> {
        self.optimize(problem, initial_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::Factor;
    use nalgebra::{DMatrix, dvector};
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
    fn test_rosenbrock_optimization() {
        // Rosenbrock function test:
        // Minimize: r1² + r2² where
        //   r1 = 10(x2 - x1²)
        //   r2 = 1 - x1
        // Starting point: [-1.2, 1.0]
        // Expected minimum: [1.0, 1.0]

        let mut problem = Problem::new();
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
        let result = solver.optimize(&problem, &initial_values).unwrap();

        // Extract final values
        let x1_final = result.parameters.get("x1").unwrap().to_vector()[0];
        let x2_final = result.parameters.get("x2").unwrap().to_vector()[0];

        println!("Rosenbrock optimization result:");
        println!("  Status: {:?}", result.status);
        println!("  Initial cost: {:.6e}", result.initial_cost);
        println!("  Final cost: {:.6e}", result.final_cost);
        println!("  Iterations: {}", result.iterations);
        println!("  x1: {:.6} (expected 1.0)", x1_final);
        println!("  x2: {:.6} (expected 1.0)", x2_final);

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
    }
}
