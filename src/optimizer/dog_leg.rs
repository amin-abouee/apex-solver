//! Dog Leg trust region optimization algorithm implementation.
//!
//! The Dog Leg method is a robust trust region algorithm for solving nonlinear least squares problems:
//!
//! ```text
//! min f(x) = ½||r(x)||² = ½Σᵢ rᵢ(x)²
//! ```
//!
//! where `r: ℝⁿ → ℝᵐ` is the residual vector function.
//!
//! # Algorithm Overview
//!
//! Powell's Dog Leg method constructs a piecewise linear path within a spherical trust region
//! of radius Δ, connecting three key points:
//!
//! 1. **Origin** (current position)
//! 2. **Cauchy Point** p_c = -α·g (optimal steepest descent step)
//! 3. **Gauss-Newton Point** h_gn (full Newton step solving J^T·J·h = -J^T·r)
//!
//! The "dog leg" path travels from the origin to the Cauchy point, then from the Cauchy point
//! toward the Gauss-Newton point, stopping at the trust region boundary.
//!
//! ## Step Selection Strategy
//!
//! Given trust region radius Δ, the algorithm selects a step h based on three cases:
//!
//! **Case 1: GN step inside trust region** (`||h_gn|| ≤ Δ`)
//! ```text
//! h = h_gn  (full Gauss-Newton step)
//! ```
//!
//! **Case 2: Even Cauchy point outside trust region** (`||p_c|| ≥ Δ`)
//! ```text
//! h = (Δ / ||g||) · (-g)  (scaled steepest descent to boundary)
//! ```
//!
//! **Case 3: Dog leg interpolation** (`||p_c|| < Δ < ||h_gn||`)
//! ```text
//! h(β) = p_c + β·(h_gn - p_c),  where β ∈ [0,1] satisfies ||h(β)|| = Δ
//! ```
//!
//! ## Cauchy Point Computation
//!
//! The Cauchy point is the optimal step along the steepest descent direction -g:
//!
//! ```text
//! α = (g^T·g) / (g^T·H·g)  where H = J^T·J
//! p_c = -α·g
//! ```
//!
//! This minimizes the quadratic model along the gradient direction.
//!
//! ## Trust Region Management
//!
//! The trust region radius Δ adapts based on the gain ratio:
//!
//! ```text
//! ρ = (actual reduction) / (predicted reduction)
//! ```
//!
//! **Good step** (`ρ > 0.75`): Increase radius `Δ ← max(Δ, 3·||h||)`
//! **Poor step** (`ρ < 0.25`): Decrease radius `Δ ← Δ/2`
//! **Moderate step**: Keep radius unchanged
//!
//! # Advanced Features (Ceres Solver Enhancements)
//!
//! This implementation includes several improvements from Google's Ceres Solver:
//!
//! ## 1. Adaptive μ Regularization
//!
//! Instead of solving `J^T·J·h = -J^T·r` directly, we solve:
//!
//! ```text
//! (J^T·J + μI)·h = -J^T·r
//! ```
//!
//! where μ adapts to handle ill-conditioned Hessians:
//! - **Increases** (×10) when linear solve fails
//! - **Decreases** (÷5) when steps are accepted
//! - **Bounded** between `min_mu` (1e-8) and `max_mu` (1.0)
//!
//! Default `initial_mu = 1e-4` provides good numerical stability.
//!
//! ## 2. Numerically Robust Beta Computation
//!
//! When computing β for dog leg interpolation, solves: `||p_c + β·v||² = Δ²`
//!
//! Uses two formulas to avoid catastrophic cancellation:
//! ```text
//! If b ≤ 0:  β = (-b + √(b²-ac)) / a    (standard formula)
//! If b > 0:  β = -c / (b + √(b²-ac))    (alternative, avoids cancellation)
//! ```
//!
//! ## 3. Step Reuse Mechanism
//!
//! When a step is rejected, caches the Gauss-Newton step, Cauchy point, and gradient.
//! On the next iteration (with smaller Δ), reuses these cached values instead of
//! recomputing them. This avoids expensive linear solves when trust region shrinks.
//!
//! **Safety limits:**
//! - Maximum 5 consecutive reuses before forcing fresh computation
//! - Cache invalidated when steps are accepted (parameters have moved)
//!
//! ## 4. Jacobi Scaling (Diagonal Preconditioning)
//!
//! Optionally applies column scaling to J before forming J^T·J:
//! ```text
//! S_ii = 1 / (1 + ||J_i||)  where J_i is column i
//! ```
//!
//! This creates an **elliptical trust region** instead of spherical, improving
//! convergence for problems with mixed parameter scales.
//!
//! # Mathematical Background
//!
//! ## Why Dog Leg Works
//!
//! The dog leg path is a cheap approximation to the optimal trust region step
//! (which would require solving a constrained optimization problem). It:
//!
//! 1. Exploits the fact that optimal steps often lie near the 2D subspace
//!    spanned by the gradient and Gauss-Newton directions
//! 2. Provides global convergence guarantees (always finds descent direction)
//! 3. Achieves local quadratic convergence (like Gauss-Newton near solution)
//! 4. Requires only one linear solve per iteration (same as Gauss-Newton)
//!
//! ## Convergence Properties
//!
//! - **Global convergence**: Guaranteed descent at each iteration
//! - **Local quadratic convergence**: Reduces to Gauss-Newton near solution
//! - **Robustness**: Handles ill-conditioning via trust region + μ regularization
//! - **Efficiency**: Comparable cost to Gauss-Newton with better reliability
//!
//! # When to Use
//!
//! Dog Leg is an excellent choice when:
//! - You want explicit control over step size (via trust region radius)
//! - The problem may be ill-conditioned
//! - You need guaranteed descent at each iteration
//! - Initial guess may be poor but you want reliable convergence
//!
//! Compared to alternatives:
//! - **vs Gauss-Newton**: More robust but similar computational cost
//! - **vs Levenberg-Marquardt**: Explicit trust region vs implicit damping
//! - Both Dog Leg and LM are excellent general-purpose choices
//!
//! # Examples
//!
//! ## Basic usage
//!
//! ```no_run
//! use apex_solver::optimizer::DogLeg;
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
//! let mut solver = DogLeg::new();
//! let result = solver.optimize(&problem, &initial_values)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced configuration with Ceres enhancements
//!
//! ```no_run
//! use apex_solver::optimizer::dog_leg::{DogLegConfig, DogLeg};
//! use apex_solver::linalg::LinearSolverType;
//!
//! # fn main() {
//! let config = DogLegConfig::new()
//!     .with_max_iterations(100)
//!     .with_trust_region_radius(1e4)  // Large initial radius
//!     .with_trust_region_bounds(1e-3, 1e6)  // Min/max radius
//!     .with_mu_params(1e-4, 1e-8, 1.0, 10.0)  // Conservative regularization
//!     .with_jacobi_scaling(true)  // Enable elliptical trust regions
//!     .with_step_reuse(true);  // Enable Ceres-style caching
//!
//! let mut solver = DogLeg::with_config(config);
//! # }
//! ```
//!
//! # References
//!
//! - Powell, M. J. D. (1970). "A Hybrid Method for Nonlinear Equations". *Numerical Methods for Nonlinear Algebraic Equations*. Gordon and Breach.
//! - Nocedal, J. & Wright, S. (2006). *Numerical Optimization* (2nd ed.). Springer. Chapter 4 (Trust Region Methods).
//! - Madsen, K., Nielsen, H. B., & Tingleff, O. (2004). *Methods for Non-Linear Least Squares Problems* (2nd ed.). Chapter 6.
//! - Conn, A. R., Gould, N. I. M., & Toint, P. L. (2000). *Trust-Region Methods*. SIAM.
//! - Ceres Solver: http://ceres-solver.org/ - Google's C++ nonlinear least squares library

use crate::{core::problem, error, linalg, manifold, optimizer};

// Note: Visualization support via observer pattern will be added in a future update
// For now, use LevenbergMarquardt optimizer for visualization support

use faer::sparse;
use std::{collections, fmt, time};
use tracing::debug;

/// Summary statistics for the Dog Leg optimization process.
#[derive(Debug, Clone)]
pub struct DogLegSummary {
    /// Initial cost value
    pub initial_cost: f64,
    /// Final cost value
    pub final_cost: f64,
    /// Total number of iterations performed
    pub iterations: usize,
    /// Number of successful steps (cost decreased)
    pub successful_steps: usize,
    /// Number of unsuccessful steps (cost increased, step rejected)
    pub unsuccessful_steps: usize,
    /// Final trust region radius
    pub final_trust_region_radius: f64,
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
    pub total_time: time::Duration,
    /// Average time per iteration
    pub average_time_per_iteration: time::Duration,
    /// Detailed per-iteration statistics history
    pub iteration_history: Vec<IterationStats>,
    /// Convergence status
    pub convergence_status: optimizer::OptimizationStatus,
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
    /// Trust region radius (Δ)
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

impl fmt::Display for DogLegSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Determine if converged
        let converged = matches!(
            self.convergence_status,
            optimizer::OptimizationStatus::Converged
                | optimizer::OptimizationStatus::CostToleranceReached
                | optimizer::OptimizationStatus::GradientToleranceReached
                | optimizer::OptimizationStatus::ParameterToleranceReached
        );

        writeln!(f, "Dog-Leg Final Result")?;

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
        writeln!(f, "Trust Region:")?;
        writeln!(f, "  Final radius: {:.6e}", self.final_trust_region_radius)?;
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

/// Configuration parameters for the Dog Leg trust region optimizer.
///
/// Controls trust region management, convergence criteria, adaptive regularization,
/// and Ceres Solver enhancements for the Dog Leg algorithm.
///
/// # Builder Pattern
///
/// All configuration options can be set using the builder pattern:
///
/// ```
/// use apex_solver::optimizer::dog_leg::DogLegConfig;
///
/// let config = DogLegConfig::new()
///     .with_max_iterations(100)
///     .with_trust_region_radius(1e4)
///     .with_mu_params(1e-4, 1e-8, 1.0, 10.0)
///     .with_jacobi_scaling(true)
///     .with_step_reuse(true);
/// ```
///
/// # Trust Region Behavior
///
/// The trust region radius Δ controls the maximum allowed step size:
///
/// - **Initial radius** (`trust_region_radius`): Starting value (default: 1e4)
/// - **Bounds** (`trust_region_min`, `trust_region_max`): Valid range (default: 1e-3 to 1e6)
/// - **Adaptation**: Increases for good steps, decreases for poor steps
///
/// # Adaptive μ Regularization (Ceres Enhancement)
///
/// Controls the regularization parameter in `(J^T·J + μI)·h = -J^T·r`:
///
/// - `initial_mu`: Starting value (default: 1e-4 for numerical stability)
/// - `min_mu`, `max_mu`: Bounds (default: 1e-8 to 1.0)
/// - `mu_increase_factor`: Multiplier when solve fails (default: 10.0)
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
/// - [`DogLeg`] - The solver that uses this configuration
/// - [`LevenbergMarquardtConfig`](crate::optimizer::LevenbergMarquardtConfig) - Alternative damping approach
/// - [`GaussNewtonConfig`](crate::optimizer::GaussNewtonConfig) - Undamped variant
#[derive(Clone)]
pub struct DogLegConfig {
    /// Type of linear solver for the linear systems
    pub linear_solver_type: linalg::LinearSolverType,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance for cost function
    pub cost_tolerance: f64,
    /// Convergence tolerance for parameter updates
    pub parameter_tolerance: f64,
    /// Convergence tolerance for gradient norm
    pub gradient_tolerance: f64,
    /// Timeout duration
    pub timeout: Option<time::Duration>,
    /// Initial trust region radius
    pub trust_region_radius: f64,
    /// Minimum trust region radius
    pub trust_region_min: f64,
    /// Maximum trust region radius
    pub trust_region_max: f64,
    /// Trust region increase factor (for good steps, rho > 0.75)
    pub trust_region_increase_factor: f64,
    /// Trust region decrease factor (for poor steps, rho < 0.25)
    pub trust_region_decrease_factor: f64,
    /// Minimum step quality for acceptance (typically 0.0)
    pub min_step_quality: f64,
    /// Good step quality threshold (typically 0.75)
    pub good_step_quality: f64,
    /// Poor step quality threshold (typically 0.25)
    pub poor_step_quality: f64,
    /// Use Jacobi column scaling (preconditioning)
    pub use_jacobi_scaling: bool,

    // Ceres-style adaptive mu regularization parameters
    /// Initial mu regularization parameter for Gauss-Newton step
    pub initial_mu: f64,
    /// Minimum mu regularization parameter
    pub min_mu: f64,
    /// Maximum mu regularization parameter
    pub max_mu: f64,
    /// Factor to increase mu when linear solver fails
    pub mu_increase_factor: f64,

    // Ceres-style step reuse optimization
    /// Enable step reuse after rejection (Ceres-style efficiency optimization)
    pub enable_step_reuse: bool,

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

    /// Compute per-variable covariance matrices (uncertainty estimation)
    ///
    /// When enabled, computes covariance by inverting the Hessian matrix after
    /// convergence. The full covariance matrix is extracted into per-variable
    /// blocks stored in both Variable structs and optimizer::SolverResult.
    ///
    /// Default: false (to avoid performance overhead)
    pub compute_covariances: bool,

    /// Enable real-time visualization (graphical debugging).
    ///
    /// When enabled, optimization progress is logged to a Rerun viewer.
    /// **Note:** Requires the `visualization` feature to be enabled in `Cargo.toml`.
    ///
    /// Default: false
    #[cfg(feature = "visualization")]
    pub enable_visualization: bool,
}

impl Default for DogLegConfig {
    fn default() -> Self {
        Self {
            linear_solver_type: linalg::LinearSolverType::default(),
            // Ceres Solver default: 50 (changed from 100 for compatibility)
            max_iterations: 50,
            // Ceres Solver default: 1e-6 (changed from 1e-8 for compatibility)
            cost_tolerance: 1e-6,
            // Ceres Solver default: 1e-8 (unchanged)
            parameter_tolerance: 1e-8,
            // Ceres Solver default: 1e-10 (changed from 1e-8 for compatibility)
            gradient_tolerance: 1e-10,
            timeout: None,
            // Ceres-style: larger initial radius for better global convergence
            trust_region_radius: 1e4,
            trust_region_min: 1e-12,
            trust_region_max: 1e12,
            // Ceres uses adaptive increase (max(radius, 3*step_norm)),
            // but we keep factor for simpler config
            trust_region_increase_factor: 3.0,
            trust_region_decrease_factor: 0.5,
            min_step_quality: 0.0,
            good_step_quality: 0.75,
            poor_step_quality: 0.25,
            // Ceres-style: Enable diagonal scaling by default for elliptical trust region
            use_jacobi_scaling: true,

            // Ceres-style adaptive mu regularization defaults
            // Start with more conservative regularization to avoid singular Hessian
            initial_mu: 1e-4,
            min_mu: 1e-8,
            max_mu: 1.0,
            mu_increase_factor: 10.0,

            // Ceres-style step reuse optimization
            enable_step_reuse: true,

            // New Ceres-compatible termination parameters
            min_cost_threshold: None,
            max_condition_number: None,
            min_relative_decrease: 1e-3,

            compute_covariances: false,
            #[cfg(feature = "visualization")]
            enable_visualization: false,
        }
    }
}

impl DogLegConfig {
    /// Create a new Dog Leg configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the linear solver type
    pub fn with_linear_solver_type(mut self, linear_solver_type: linalg::LinearSolverType) -> Self {
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
    pub fn with_timeout(mut self, timeout: time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the initial trust region radius
    pub fn with_trust_region_radius(mut self, radius: f64) -> Self {
        self.trust_region_radius = radius;
        self
    }

    /// Set the trust region radius bounds
    pub fn with_trust_region_bounds(mut self, min: f64, max: f64) -> Self {
        self.trust_region_min = min;
        self.trust_region_max = max;
        self
    }

    /// Set the trust region adjustment factors
    pub fn with_trust_region_factors(mut self, increase: f64, decrease: f64) -> Self {
        self.trust_region_increase_factor = increase;
        self.trust_region_decrease_factor = decrease;
        self
    }

    /// Set the trust region quality thresholds
    pub fn with_step_quality_thresholds(
        mut self,
        min_quality: f64,
        poor_quality: f64,
        good_quality: f64,
    ) -> Self {
        self.min_step_quality = min_quality;
        self.poor_step_quality = poor_quality;
        self.good_step_quality = good_quality;
        self
    }

    /// Enable or disable Jacobi column scaling (preconditioning)
    pub fn with_jacobi_scaling(mut self, use_jacobi_scaling: bool) -> Self {
        self.use_jacobi_scaling = use_jacobi_scaling;
        self
    }

    /// Set adaptive mu regularization parameters (Ceres-style)
    pub fn with_mu_params(
        mut self,
        initial_mu: f64,
        min_mu: f64,
        max_mu: f64,
        increase_factor: f64,
    ) -> Self {
        self.initial_mu = initial_mu;
        self.min_mu = min_mu;
        self.max_mu = max_mu;
        self.mu_increase_factor = increase_factor;
        self
    }

    /// Enable or disable step reuse optimization (Ceres-style)
    pub fn with_step_reuse(mut self, enable_step_reuse: bool) -> Self {
        self.enable_step_reuse = enable_step_reuse;
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

    /// Enable or disable covariance computation (uncertainty estimation).
    ///
    /// When enabled, computes the full covariance matrix by inverting the Hessian
    /// after convergence, then extracts per-variable covariance blocks.
    pub fn with_compute_covariances(mut self, compute_covariances: bool) -> Self {
        self.compute_covariances = compute_covariances;
        self
    }

    /// Enable real-time visualization.
    ///
    /// **Note:** Requires the `visualization` feature to be enabled in `Cargo.toml`.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable visualization
    #[cfg(feature = "visualization")]
    pub fn with_visualization(mut self, enable: bool) -> Self {
        self.enable_visualization = enable;
        self
    }

    /// Print configuration parameters (info level logging)
    pub fn print_configuration(&self) {
        debug!(
            "Configuration:\n  Solver:        Dog-Leg\n  Linear solver: {:?}\n  Loss function: N/A\n\nConvergence Criteria:\n  Max iterations:      {}\n  Cost tolerance:      {:.2e}\n  Parameter tolerance: {:.2e}\n  Gradient tolerance:  {:.2e}\n  Timeout:             {:?}\n\nTrust Region:\n  Initial radius:      {:.2e}\n  Radius range:        [{:.2e}, {:.2e}]\n  Min step quality:    {:.2}\n  Good step quality:   {:.2}\n  Poor step quality:   {:.2}\n\nRegularization:\n  Initial mu:          {:.2e}\n  Mu range:            [{:.2e}, {:.2e}]\n  Mu increase factor:  {:.2}\n\nNumerical Settings:\n  Jacobi scaling:      {}\n  Step reuse:          {}\n  Compute covariances: {}",
            self.linear_solver_type,
            self.max_iterations,
            self.cost_tolerance,
            self.parameter_tolerance,
            self.gradient_tolerance,
            self.timeout,
            self.trust_region_radius,
            self.trust_region_min,
            self.trust_region_max,
            self.min_step_quality,
            self.good_step_quality,
            self.poor_step_quality,
            self.initial_mu,
            self.min_mu,
            self.max_mu,
            self.mu_increase_factor,
            if self.use_jacobi_scaling {
                "enabled"
            } else {
                "disabled"
            },
            if self.enable_step_reuse {
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

/// State for optimization iteration
struct OptimizationState {
    variables: collections::HashMap<String, problem::VariableEnum>,
    variable_index_map: collections::HashMap<String, usize>,
    sorted_vars: Vec<String>,
    symbolic_structure: problem::SymbolicStructure,
    current_cost: f64,
    initial_cost: f64,
}

/// Result from step computation
struct StepResult {
    step: faer::Mat<f64>,
    gradient_norm: f64,
    predicted_reduction: f64,
}

/// Type of step taken
#[derive(Debug, Clone, Copy)]
enum StepType {
    /// Full Gauss-Newton step
    GaussNewton,
    /// Scaled steepest descent (Cauchy point)
    SteepestDescent,
    /// Dog leg interpolation
    DogLeg,
}

impl fmt::Display for StepType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepType::GaussNewton => write!(f, "GN"),
            StepType::SteepestDescent => write!(f, "SD"),
            StepType::DogLeg => write!(f, "DL"),
        }
    }
}

/// Result from step evaluation
struct StepEvaluation {
    accepted: bool,
    cost_reduction: f64,
    rho: f64,
}

/// Dog Leg trust region solver for nonlinear least squares optimization.
///
/// Implements Powell's Dog Leg algorithm with Ceres Solver enhancements including
/// adaptive μ regularization, numerically robust beta computation, and step reuse caching.
///
/// # Algorithm
///
/// At each iteration k:
/// 1. Compute residual `r(xₖ)` and Jacobian `J(xₖ)`
/// 2. Solve for Gauss-Newton step: `(J^T·J + μI)·h_gn = -J^T·r`
/// 3. Compute steepest descent direction: `-g` where `g = J^T·r`
/// 4. Compute Cauchy point: `p_c = -α·g` (optimal along steepest descent)
/// 5. Construct dog leg step based on trust region radius Δ:
///    - If `||h_gn|| ≤ Δ`: Take full GN step
///    - Else if `||p_c|| ≥ Δ`: Take scaled SD to boundary
///    - Else: Interpolate `h = p_c + β·(h_gn - p_c)` where `||h|| = Δ`
/// 6. Evaluate gain ratio: `ρ = (actual reduction) / (predicted reduction)`
/// 7. Update trust region radius based on ρ
/// 8. Accept/reject step and update parameters
///
/// # Ceres Solver Enhancements
///
/// This implementation includes four major improvements from Google's Ceres Solver:
///
/// **1. Adaptive μ Regularization:** Dynamically adjusts regularization parameter
/// to handle ill-conditioned Hessians (increases on failure, decreases on success).
///
/// **2. Numerically Robust Beta:** Uses two formulas for computing dog leg
/// interpolation parameter β to avoid catastrophic cancellation.
///
/// **3. Step Reuse Mechanism:** Caches GN step, Cauchy point, and gradient when
/// steps are rejected. Limited to 5 consecutive reuses to prevent staleness.
///
/// **4. Jacobi Scaling:** Optional diagonal preconditioning creates elliptical
/// trust regions for better handling of mixed-scale problems.
///
/// # Examples
///
/// ```no_run
/// use apex_solver::optimizer::DogLeg;
/// use apex_solver::core::problem::Problem;
/// use std::collections::HashMap;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut problem = Problem::new();
/// // ... add factors to problem ...
///
/// let initial_values = HashMap::new();
/// // ... initialize parameters ...
///
/// let mut solver = DogLeg::new();
/// let result = solver.optimize(&problem, &initial_values)?;
/// # Ok(())
/// # }
/// ```
///
/// # See Also
///
/// - [`DogLegConfig`] - Configuration options
/// - [`LevenbergMarquardt`](crate::optimizer::LevenbergMarquardt) - Alternative adaptive damping
/// - [`GaussNewton`](crate::optimizer::GaussNewton) - Undamped variant
pub struct DogLeg {
    config: DogLegConfig,
    jacobi_scaling: Option<sparse::SparseColMat<usize, f64>>,
    observers: optimizer::OptObserverVec,

    // Adaptive mu regularization (Ceres-style)
    mu: f64,
    min_mu: f64,
    max_mu: f64,
    mu_increase_factor: f64,

    // Step reuse mechanism (Ceres-style efficiency optimization)
    reuse_step_on_rejection: bool,
    cached_gn_step: Option<faer::Mat<f64>>,
    cached_cauchy_point: Option<faer::Mat<f64>>,
    cached_gradient: Option<faer::Mat<f64>>,
    cached_alpha: Option<f64>,
    cache_reuse_count: usize, // Track consecutive reuses to prevent staleness
}

impl Default for DogLeg {
    fn default() -> Self {
        Self::new()
    }
}

impl DogLeg {
    /// Create a new Dog Leg solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(DogLegConfig::default())
    }

    /// Create a new Dog Leg solver with the given configuration.
    pub fn with_config(config: DogLegConfig) -> Self {
        Self {
            // Initialize adaptive mu from config
            mu: config.initial_mu,
            min_mu: config.min_mu,
            max_mu: config.max_mu,
            mu_increase_factor: config.mu_increase_factor,

            // Initialize step reuse mechanism (disabled initially, enabled after first rejection)
            reuse_step_on_rejection: false,
            cached_gn_step: None,
            cached_cauchy_point: None,
            cached_gradient: None,
            cached_alpha: None,
            cache_reuse_count: 0,

            config,
            jacobi_scaling: None,
            observers: optimizer::OptObserverVec::new(),
        }
    }

    /// Add an observer to the solver.
    ///
    /// Observers are notified at each iteration with the current variable values.
    /// This enables real-time visualization, logging, metrics collection, etc.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use apex_solver::optimizer::DogLeg;
    /// # use apex_solver::optimizer::OptObserver;
    /// # use std::collections::HashMap;
    /// # use apex_solver::core::problem::VariableEnum;
    ///
    /// # struct MyObserver;
    /// # impl OptObserver for MyObserver {
    /// #     fn on_step(&self, _: &HashMap<String, VariableEnum>, _: usize) {}
    /// # }
    /// let mut solver = DogLeg::new();
    /// solver.add_observer(MyObserver);
    /// ```
    pub fn add_observer(&mut self, observer: impl optimizer::OptObserver + 'static) {
        self.observers.add(observer);
    }

    /// Create the appropriate linear solver based on configuration
    fn create_linear_solver(&self) -> Box<dyn linalg::SparseLinearSolver> {
        match self.config.linear_solver_type {
            linalg::LinearSolverType::SparseCholesky => {
                Box::new(linalg::SparseCholeskySolver::new())
            }
            linalg::LinearSolverType::SparseQR => Box::new(linalg::SparseQRSolver::new()),
            linalg::LinearSolverType::SparseSchurComplement => {
                // Schur complement solver requires special handling - fallback to Cholesky
                Box::new(linalg::SparseCholeskySolver::new())
            }
        }
    }

    /// Compute the Cauchy point (steepest descent step)
    /// Returns the optimal step along the negative gradient direction
    /// Compute Cauchy point and optimal step length for steepest descent
    ///
    /// Returns (alpha, cauchy_point) where:
    /// - alpha: optimal step length α = ||g||² / (g^T H g)
    /// - cauchy_point: p_c = -α * g (the Cauchy point)
    ///
    /// This is the optimal point along the steepest descent direction within
    /// the quadratic approximation of the objective function.
    fn compute_cauchy_point_and_alpha(
        &self,
        gradient: &faer::Mat<f64>,
        hessian: &sparse::SparseColMat<usize, f64>,
    ) -> (f64, faer::Mat<f64>) {
        // Optimal step size along steepest descent: α = (g^T*g) / (g^T*H*g)
        let g_norm_sq_mat = gradient.transpose() * gradient;
        let g_norm_sq = g_norm_sq_mat[(0, 0)];

        let h_g = hessian * gradient;
        let g_h_g_mat = gradient.transpose() * &h_g;
        let g_h_g = g_h_g_mat[(0, 0)];

        // Avoid division by zero
        let alpha = if g_h_g.abs() > 1e-15 {
            g_norm_sq / g_h_g
        } else {
            1.0
        };

        // Compute Cauchy point: p_c = -α * gradient
        let mut cauchy_point = faer::Mat::zeros(gradient.nrows(), 1);
        for i in 0..gradient.nrows() {
            cauchy_point[(i, 0)] = -alpha * gradient[(i, 0)];
        }

        (alpha, cauchy_point)
    }

    /// Compute the dog leg step using Powell's Dog Leg method
    ///
    /// The dog leg path consists of two segments:
    /// 1. From origin to Cauchy point (optimal along steepest descent)
    /// 2. From Cauchy point to Gauss-Newton step
    ///
    /// Arguments:
    /// - steepest_descent_dir: -gradient (steepest descent direction, not scaled)
    /// - cauchy_point: p_c = α * (-gradient), the optimal steepest descent step
    /// - h_gn: Gauss-Newton step
    /// - delta: Trust region radius
    ///
    /// Returns: (step, step_type)
    fn compute_dog_leg_step(
        &self,
        steepest_descent_dir: &faer::Mat<f64>,
        cauchy_point: &faer::Mat<f64>,
        h_gn: &faer::Mat<f64>,
        delta: f64,
    ) -> (faer::Mat<f64>, StepType) {
        let gn_norm = h_gn.norm_l2();
        let cauchy_norm = cauchy_point.norm_l2();
        let sd_norm = steepest_descent_dir.norm_l2();

        // Case 1: Full Gauss-Newton step fits in trust region
        if gn_norm <= delta {
            return (h_gn.clone(), StepType::GaussNewton);
        }

        // Case 2: Even Cauchy point is outside trust region
        // Scale steepest descent direction to boundary: (delta / ||δ_sd||) * δ_sd
        if cauchy_norm >= delta {
            let mut scaled_sd = faer::Mat::zeros(steepest_descent_dir.nrows(), 1);
            let scale = delta / sd_norm;
            for i in 0..steepest_descent_dir.nrows() {
                scaled_sd[(i, 0)] = steepest_descent_dir[(i, 0)] * scale;
            }
            return (scaled_sd, StepType::SteepestDescent);
        }

        // Case 3: Dog leg interpolation between Cauchy point and GN step
        // Use Ceres-style numerically robust formula
        //
        // Following Ceres solver implementation for numerical stability:
        // Compute intersection of trust region boundary with line from Cauchy point to GN step
        //
        // Let v = δ_gn - p_c
        // Solve: ||p_c + β*v||² = delta²
        // This gives: a*β² + 2*b*β + c = 0
        // where:
        //   a = v^T·v = ||v||²
        //   b = p_c^T·v
        //   c = p_c^T·p_c - delta² = ||p_c||² - delta²

        let mut v = faer::Mat::zeros(cauchy_point.nrows(), 1);
        for i in 0..cauchy_point.nrows() {
            v[(i, 0)] = h_gn[(i, 0)] - cauchy_point[(i, 0)];
        }

        // Compute coefficients
        let v_squared_norm = v.transpose() * &v;
        let a = v_squared_norm[(0, 0)];

        let pc_dot_v = cauchy_point.transpose() * &v;
        let b = pc_dot_v[(0, 0)];

        let c = cauchy_norm * cauchy_norm - delta * delta;

        // Ceres-style numerically robust beta computation
        // Uses two different formulas based on sign of b to avoid catastrophic cancellation
        let d_squared = b * b - a * c;

        let beta = if d_squared < 0.0 {
            // Should not happen geometrically, but handle gracefully
            1.0
        } else if a.abs() < 1e-15 {
            // Degenerate case: v is nearly zero
            1.0
        } else {
            let d = d_squared.sqrt();

            // Ceres formula: choose formula based on sign of b to avoid cancellation
            // If b <= 0: beta = (-b + d) / a  (standard formula, no cancellation)
            // If b > 0:  beta = -c / (b + d)  (alternative formula, avoids cancellation)
            if b <= 0.0 { (-b + d) / a } else { -c / (b + d) }
        };

        // Clamp beta to [0, 1] for safety
        let beta = beta.clamp(0.0, 1.0);

        // Compute dog leg step: p_dl = p_c + β*(δ_gn - p_c)
        let mut dog_leg = faer::Mat::zeros(cauchy_point.nrows(), 1);
        for i in 0..cauchy_point.nrows() {
            dog_leg[(i, 0)] = cauchy_point[(i, 0)] + beta * v[(i, 0)];
        }

        (dog_leg, StepType::DogLeg)
    }

    /// Update trust region radius based on step quality (Ceres-style)
    fn update_trust_region(&mut self, rho: f64, step_norm: f64) -> bool {
        if rho > self.config.good_step_quality {
            // Good step, increase trust region (Ceres-style: max(radius, 3*step_norm))
            let new_radius = self.config.trust_region_radius.max(3.0 * step_norm);
            self.config.trust_region_radius = new_radius.min(self.config.trust_region_max);

            // Decrease mu on successful step (Ceres-style adaptive regularization)
            self.mu = (self.mu / (0.5 * self.mu_increase_factor)).max(self.min_mu);

            // Clear reuse flag and invalidate cache on acceptance (parameters have moved)
            self.reuse_step_on_rejection = false;
            self.cached_gn_step = None;
            self.cached_cauchy_point = None;
            self.cached_gradient = None;
            self.cached_alpha = None;
            self.cache_reuse_count = 0;

            true
        } else if rho < self.config.poor_step_quality {
            // Poor step, decrease trust region
            self.config.trust_region_radius = (self.config.trust_region_radius
                * self.config.trust_region_decrease_factor)
                .max(self.config.trust_region_min);

            // Enable step reuse flag for next iteration (Ceres-style)
            self.reuse_step_on_rejection = self.config.enable_step_reuse;

            false
        } else {
            // Moderate step, keep trust region unchanged
            // Clear reuse flag and invalidate cache on acceptance (parameters have moved)
            self.reuse_step_on_rejection = false;
            self.cached_gn_step = None;
            self.cached_cauchy_point = None;
            self.cached_gradient = None;
            self.cached_alpha = None;
            self.cache_reuse_count = 0;

            true
        }
    }

    /// Compute step quality ratio (actual vs predicted reduction)
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
    fn compute_predicted_reduction(
        &self,
        step: &faer::Mat<f64>,
        gradient: &faer::Mat<f64>,
        hessian: &sparse::SparseColMat<usize, f64>,
    ) -> f64 {
        // Dog Leg predicted reduction: -step^T * gradient - 0.5 * step^T * H * step
        let linear_term = step.transpose() * gradient;
        let hessian_step = hessian * step;
        let quadratic_term = step.transpose() * &hessian_step;

        -linear_term[(0, 0)] - 0.5 * quadratic_term[(0, 0)]
    }

    /// Check convergence criteria
    /// Check convergence using comprehensive termination criteria.
    ///
    /// Implements 9 termination criteria following Ceres Solver standards for Dog Leg:
    ///
    /// 1. **Gradient Norm (First-Order Optimality)**: ||g||∞ ≤ gradient_tolerance
    /// 2. **Parameter Change Tolerance**: ||h|| ≤ parameter_tolerance * (||x|| + parameter_tolerance)
    /// 3. **Function Value Change Tolerance**: |ΔF| < cost_tolerance * F
    /// 4. **Objective Function Cutoff**: F_new < min_cost_threshold (optional)
    /// 5. **Trust Region Radius**: radius < trust_region_min
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
        elapsed: time::Duration,
        step_accepted: bool,
    ) -> Option<optimizer::OptimizationStatus> {
        // ========================================================================
        // CRITICAL SAFETY CHECKS (perform first, before convergence checks)
        // ========================================================================

        // CRITERION 7: Invalid Numerical Values (NaN/Inf)
        // Always check for numerical instability first
        if !new_cost.is_finite() || !parameter_update_norm.is_finite() || !gradient_norm.is_finite()
        {
            return Some(optimizer::OptimizationStatus::InvalidNumericalValues);
        }

        // CRITERION 9: Timeout
        // Check wall-clock time limit
        if let Some(timeout) = self.config.timeout
            && elapsed >= timeout
        {
            return Some(optimizer::OptimizationStatus::Timeout);
        }

        // CRITERION 8: Maximum Iterations
        // Check iteration count limit
        if iteration >= self.config.max_iterations {
            return Some(optimizer::OptimizationStatus::MaxIterationsReached);
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
            return Some(optimizer::OptimizationStatus::GradientToleranceReached);
        }

        // Only check parameter and cost criteria after first iteration
        if iteration > 0 {
            // CRITERION 2: Parameter Change Tolerance (xtol)
            // Ceres formula: ||h|| ≤ ε_param * (||x|| + ε_param)
            // This is a relative measure that scales with parameter magnitude
            let relative_step_tolerance = self.config.parameter_tolerance
                * (parameter_norm + self.config.parameter_tolerance);

            if parameter_update_norm <= relative_step_tolerance {
                return Some(optimizer::OptimizationStatus::ParameterToleranceReached);
            }

            // CRITERION 3: Function Value Change Tolerance (ftol)
            // Ceres formula: |ΔF| < ε_cost * F
            // Check relative cost change (not absolute)
            let cost_change = (current_cost - new_cost).abs();
            let relative_cost_change = cost_change / current_cost.max(1e-10); // Avoid division by zero

            if relative_cost_change < self.config.cost_tolerance {
                return Some(optimizer::OptimizationStatus::CostToleranceReached);
            }
        }

        // CRITERION 4: Objective Function Cutoff (optional early stopping)
        // Useful for "good enough" solutions
        if let Some(min_cost) = self.config.min_cost_threshold
            && new_cost < min_cost
        {
            return Some(optimizer::OptimizationStatus::MinCostThresholdReached);
        }

        // CRITERION 5: Trust Region Radius
        // If trust region has collapsed, optimization has converged or problem is ill-conditioned
        if trust_region_radius < self.config.trust_region_min {
            return Some(optimizer::OptimizationStatus::TrustRegionRadiusTooSmall);
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
    fn compute_parameter_norm(
        variables: &collections::HashMap<String, problem::VariableEnum>,
    ) -> f64 {
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
        jacobian: &sparse::SparseColMat<usize, f64>,
    ) -> Result<sparse::SparseColMat<usize, f64>, optimizer::OptimizerError> {
        use faer::sparse::Triplet;

        let cols = jacobian.ncols();
        let jacobi_scaling_vec: Vec<Triplet<usize, usize, f64>> = (0..cols)
            .map(|c| {
                let col_norm_squared: f64 = jacobian
                    .triplet_iter()
                    .filter(|t| t.col == c)
                    .map(|t| t.val * t.val)
                    .sum();
                let col_norm = col_norm_squared.sqrt();
                let scaling = 1.0 / (1.0 + col_norm);
                Triplet::new(c, c, scaling)
            })
            .collect();

        sparse::SparseColMat::try_new_from_triplets(cols, cols, &jacobi_scaling_vec).map_err(|e| {
            optimizer::OptimizerError::JacobiScalingCreation(e.to_string()).log_with_source(e)
        })
    }

    /// Initialize optimization state
    fn initialize_optimization_state(
        &self,
        problem: &problem::Problem,
        initial_params: &collections::HashMap<
            String,
            (manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<OptimizationState, error::ApexSolverError> {
        let variables = problem.initialize_variables(initial_params);

        let mut variable_index_map = collections::HashMap::new();
        let mut col_offset = 0;
        let mut sorted_vars: Vec<String> = variables.keys().cloned().collect();
        sorted_vars.sort();

        for var_name in &sorted_vars {
            variable_index_map.insert(var_name.clone(), col_offset);
            col_offset += variables[var_name].get_size();
        }

        let symbolic_structure =
            problem.build_symbolic_structure(&variables, &variable_index_map, col_offset)?;

        // Initial cost evaluation (residual only, no Jacobian needed)
        let residual = problem.compute_residual_sparse(&variables)?;

        let residual_norm = residual.norm_l2();
        let current_cost = residual_norm * residual_norm;
        let initial_cost = current_cost;

        Ok(OptimizationState {
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
        jacobian: &sparse::SparseColMat<usize, f64>,
        iteration: usize,
    ) -> Result<sparse::SparseColMat<usize, f64>, optimizer::OptimizerError> {
        // Create Jacobi scaling on first iteration if enabled
        if iteration == 0 {
            let scaling = self.create_jacobi_scaling(jacobian)?;
            self.jacobi_scaling = Some(scaling);
        }
        let scaling = self
            .jacobi_scaling
            .as_ref()
            .ok_or_else(|| optimizer::OptimizerError::JacobiScalingNotInitialized.log())?;
        Ok(jacobian * scaling)
    }

    /// Compute dog leg optimization step
    fn compute_optimization_step(
        &mut self,
        residuals: &faer::Mat<f64>,
        scaled_jacobian: &sparse::SparseColMat<usize, f64>,
        linear_solver: &mut Box<dyn linalg::SparseLinearSolver>,
    ) -> Option<StepResult> {
        // Check if we can reuse cached step (Ceres-style optimization)
        // Safety limit: prevent excessive reuse that could lead to stale gradient/Hessian
        const MAX_CACHE_REUSE: usize = 5;

        if self.reuse_step_on_rejection
            && self.config.enable_step_reuse
            && self.cache_reuse_count < MAX_CACHE_REUSE
            && let (Some(cached_gn), Some(cached_cauchy), Some(cached_grad), Some(_cached_a)) = (
                &self.cached_gn_step,
                &self.cached_cauchy_point,
                &self.cached_gradient,
                &self.cached_alpha,
            )
        {
            // Increment reuse counter
            self.cache_reuse_count += 1;

            let gradient_norm = cached_grad.norm_l2();
            let mut steepest_descent = faer::Mat::zeros(cached_grad.nrows(), 1);
            for i in 0..cached_grad.nrows() {
                steepest_descent[(i, 0)] = -cached_grad[(i, 0)];
            }

            let (scaled_step, _step_type) = self.compute_dog_leg_step(
                &steepest_descent,
                cached_cauchy,
                cached_gn,
                self.config.trust_region_radius,
            );

            let step = if self.config.use_jacobi_scaling {
                let scaling = self.jacobi_scaling.as_ref()?;
                scaling * &scaled_step
            } else {
                scaled_step.clone()
            };

            let hessian = linear_solver.get_hessian()?;
            let predicted_reduction =
                self.compute_predicted_reduction(&scaled_step, cached_grad, hessian);

            return Some(StepResult {
                step,
                gradient_norm,
                predicted_reduction,
            });
        }

        // Not reusing, compute fresh step
        // 1. Solve for Gauss-Newton step with adaptive mu regularization (Ceres-style)
        let residuals_owned = residuals.as_ref().to_owned();
        let mut scaled_gn_step = None;
        let mut mu_attempts = 0;

        // Try to solve with current mu, increasing if necessary
        while mu_attempts < 10 && self.mu <= self.max_mu {
            let damping = self.mu;

            if let Ok(step) =
                linear_solver.solve_augmented_equation(&residuals_owned, scaled_jacobian, damping)
            {
                scaled_gn_step = Some(step);
                break;
            }

            // Increase mu (Ceres-style)
            self.mu = (self.mu * self.mu_increase_factor).min(self.max_mu);
            mu_attempts += 1;
        }

        let scaled_gn_step = scaled_gn_step?;

        // 2. Get gradient and Hessian (cached by solve_augmented_equation)
        let gradient = linear_solver.get_gradient()?;
        let hessian = linear_solver.get_hessian()?;
        let gradient_norm = gradient.norm_l2();

        // 3. Compute steepest descent direction: δ_sd = -gradient
        let mut steepest_descent = faer::Mat::zeros(gradient.nrows(), 1);
        for i in 0..gradient.nrows() {
            steepest_descent[(i, 0)] = -gradient[(i, 0)];
        }

        // 4. Compute Cauchy point and optimal step length
        let (alpha, cauchy_point) = self.compute_cauchy_point_and_alpha(gradient, hessian);

        // 5. Compute dog leg step based on trust region radius
        let (scaled_step, _step_type) = self.compute_dog_leg_step(
            &steepest_descent,
            &cauchy_point,
            &scaled_gn_step,
            self.config.trust_region_radius,
        );

        // 6. Apply inverse Jacobi scaling if enabled
        let step = if self.config.use_jacobi_scaling {
            let scaling = self.jacobi_scaling.as_ref()?;
            scaling * &scaled_step
        } else {
            scaled_step.clone()
        };

        // 7. Compute predicted reduction
        let predicted_reduction = self.compute_predicted_reduction(&scaled_step, gradient, hessian);

        // 8. Cache step components for potential reuse (Ceres-style)
        self.cached_gn_step = Some(scaled_gn_step.clone());
        self.cached_cauchy_point = Some(cauchy_point.clone());
        self.cached_gradient = Some(gradient.clone());
        self.cached_alpha = Some(alpha);

        Some(StepResult {
            step,
            gradient_norm,
            predicted_reduction,
        })
    }

    /// Evaluate and apply step
    fn evaluate_and_apply_step(
        &mut self,
        step_result: &StepResult,
        state: &mut OptimizationState,
        problem: &problem::Problem,
    ) -> error::ApexSolverResult<StepEvaluation> {
        // Apply parameter updates
        let step_norm = optimizer::apply_parameter_step(
            &mut state.variables,
            step_result.step.as_ref(),
            &state.sorted_vars,
        );

        // Compute new cost (residual only, no Jacobian needed for step evaluation)
        let new_residual = problem.compute_residual_sparse(&state.variables)?;
        let new_residual_norm = new_residual.norm_l2();
        let new_cost = new_residual_norm * new_residual_norm;

        // Compute step quality
        let rho = self.compute_step_quality(
            state.current_cost,
            new_cost,
            step_result.predicted_reduction,
        );

        // Update trust region and decide acceptance
        // Filter out numerical noise with small threshold
        let accepted = rho > 1e-4;
        let _trust_region_updated = self.update_trust_region(rho, step_norm);

        let cost_reduction = if accepted {
            let reduction = state.current_cost - new_cost;
            state.current_cost = new_cost;
            reduction
        } else {
            // Reject step - revert changes
            optimizer::apply_negative_parameter_step(
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
        iterations: usize,
        successful_steps: usize,
        unsuccessful_steps: usize,
        max_gradient_norm: f64,
        final_gradient_norm: f64,
        max_parameter_update_norm: f64,
        final_parameter_update_norm: f64,
        total_cost_reduction: f64,
        total_time: time::Duration,
        iteration_history: Vec<IterationStats>,
        convergence_status: optimizer::OptimizationStatus,
    ) -> DogLegSummary {
        DogLegSummary {
            initial_cost,
            final_cost,
            iterations,
            successful_steps,
            unsuccessful_steps,
            final_trust_region_radius: self.config.trust_region_radius,
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
                time::Duration::from_secs(0)
            },
            iteration_history,
            convergence_status,
        }
    }

    /// Minimize the optimization problem using Dog Leg algorithm
    pub fn optimize(
        &mut self,
        problem: &problem::Problem,
        initial_params: &collections::HashMap<
            String,
            (manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<
        optimizer::SolverResult<collections::HashMap<String, problem::VariableEnum>>,
        error::ApexSolverError,
    > {
        let start_time = time::Instant::now();
        let mut iteration = 0;
        let mut cost_evaluations = 1;
        let mut jacobian_evaluations = 0;
        let mut successful_steps = 0;
        let mut unsuccessful_steps = 0;

        let mut state = self.initialize_optimization_state(problem, initial_params)?;
        let mut linear_solver = self.create_linear_solver();

        let mut max_gradient_norm: f64 = 0.0;
        let mut max_parameter_update_norm: f64 = 0.0;
        let mut total_cost_reduction = 0.0;
        let mut final_gradient_norm;
        let mut final_parameter_update_norm;

        // Initialize iteration statistics tracking
        let mut iteration_stats = Vec::with_capacity(self.config.max_iterations);
        let mut previous_cost = state.current_cost;

        // Print configuration and header if info/debug level is enabled
        if tracing::enabled!(tracing::Level::DEBUG) {
            self.config.print_configuration();
            IterationStats::print_header();
        }

        loop {
            let iter_start = time::Instant::now();
            // Evaluate residuals and Jacobian
            let (residuals, jacobian) = problem.compute_residual_and_jacobian_sparse(
                &state.variables,
                &state.variable_index_map,
                &state.symbolic_structure,
            )?;
            jacobian_evaluations += 1;

            // Process Jacobian (apply scaling if enabled)
            let scaled_jacobian = if self.config.use_jacobi_scaling {
                self.process_jacobian(&jacobian, iteration)?
            } else {
                jacobian
            };

            // Compute dog leg step
            let step_result = match self.compute_optimization_step(
                &residuals,
                &scaled_jacobian,
                &mut linear_solver,
            ) {
                Some(result) => result,
                None => {
                    return Err(optimizer::OptimizerError::LinearSolveFailed(
                        "Linear solver failed to solve system".to_string(),
                    )
                    .into());
                }
            };

            // Update tracking
            max_gradient_norm = max_gradient_norm.max(step_result.gradient_norm);
            final_gradient_norm = step_result.gradient_norm;
            let step_norm = step_result.step.norm_l2();
            max_parameter_update_norm = max_parameter_update_norm.max(step_norm);
            final_parameter_update_norm = step_norm;

            // Evaluate and apply step
            let step_eval = self.evaluate_and_apply_step(&step_result, &mut state, problem)?;
            cost_evaluations += 1;

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
                    tr_radius: self.config.trust_region_radius,
                    ls_iter: 0, // Direct solver (Cholesky) has no iterations
                    iter_time_ms: iter_elapsed_ms,
                    total_time_ms: total_elapsed_ms,
                    accepted: step_eval.accepted,
                };

                iteration_stats.push(stats.clone());
                stats.print_line();
            }

            previous_cost = state.current_cost;

            // Notify all observers with current state
            // First set metrics data, then notify observers
            self.observers.set_iteration_metrics(
                state.current_cost,
                step_result.gradient_norm,
                Some(self.config.trust_region_radius), // Dog Leg uses trust region radius
                step_norm,
                Some(step_eval.rho),
            );

            // Set matrix data if available and there are observers
            if !self.observers.is_empty()
                && let (Some(hessian), Some(gradient)) =
                    (linear_solver.get_hessian(), linear_solver.get_gradient())
            {
                self.observers
                    .set_matrix_data(Some(hessian.clone()), Some(gradient.clone()));
            }

            // Notify observers with current variable values and iteration number
            self.observers.notify(&state.variables, iteration);

            // Check convergence
            let elapsed = start_time.elapsed();

            // Compute parameter norm for relative parameter tolerance check
            let parameter_norm = Self::compute_parameter_norm(&state.variables);

            // Compute costs for convergence check
            let new_cost = state.current_cost;

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

                // Print summary only if debug level is enabled
                if tracing::enabled!(tracing::Level::DEBUG) {
                    debug!("{}", summary);
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

                return Ok(optimizer::SolverResult {
                    status,
                    iterations: iteration + 1,
                    initial_cost: state.initial_cost,
                    final_cost: state.current_cost,
                    parameters: state.variables.into_iter().collect(),
                    elapsed_time: elapsed,
                    convergence_info: Some(optimizer::ConvergenceInfo {
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

impl optimizer::Solver for DogLeg {
    type Config = DogLegConfig;
    type Error = error::ApexSolverError;

    fn new() -> Self {
        Self::default()
    }

    fn optimize(
        &mut self,
        problem: &problem::Problem,
        initial_params: &collections::HashMap<
            String,
            (manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<
        optimizer::SolverResult<collections::HashMap<String, problem::VariableEnum>>,
        Self::Error,
    > {
        self.optimize(problem, initial_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{factors, manifold};
    use nalgebra;

    /// Custom Rosenbrock Factor 1: r1 = 10(x2 - x1²)
    /// Demonstrates extensibility - custom factors can be defined outside of factors.rs
    #[derive(Debug, Clone)]
    struct RosenbrockFactor1;

    impl factors::Factor for RosenbrockFactor1 {
        fn linearize(
            &self,
            params: &[nalgebra::DVector<f64>],
            compute_jacobian: bool,
        ) -> (nalgebra::DVector<f64>, Option<nalgebra::DMatrix<f64>>) {
            let x1 = params[0][0];
            let x2 = params[1][0];

            // Residual: r1 = 10(x2 - x1²)
            let residual = nalgebra::dvector![10.0 * (x2 - x1 * x1)];

            // Jacobian: ∂r1/∂x1 = -20*x1, ∂r1/∂x2 = 10
            let jacobian = if compute_jacobian {
                let mut jac = nalgebra::DMatrix::zeros(1, 2);
                jac[(0, 0)] = -20.0 * x1;
                jac[(0, 1)] = 10.0;
                Some(jac)
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

    impl factors::Factor for RosenbrockFactor2 {
        fn linearize(
            &self,
            params: &[nalgebra::DVector<f64>],
            compute_jacobian: bool,
        ) -> (nalgebra::DVector<f64>, Option<nalgebra::DMatrix<f64>>) {
            let x1 = params[0][0];

            // Residual: r2 = 1 - x1
            let residual = nalgebra::dvector![1.0 - x1];

            // Jacobian: ∂r2/∂x1 = -1
            let jacobian = if compute_jacobian {
                Some(nalgebra::DMatrix::from_element(1, 1, -1.0))
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
    fn test_rosenbrock_optimization() -> Result<(), Box<dyn std::error::Error>> {
        // Rosenbrock function test:
        // Minimize: r1² + r2² where
        //   r1 = 10(x2 - x1²)
        //   r2 = 1 - x1
        // Starting point: [-1.2, 1.0]
        // Expected minimum: [1.0, 1.0]

        let mut problem = problem::Problem::new();
        let mut initial_values = collections::HashMap::new();

        // Add variables using Rn manifold (Euclidean space)
        initial_values.insert(
            "x1".to_string(),
            (manifold::ManifoldType::RN, nalgebra::dvector![-1.2]),
        );
        initial_values.insert(
            "x2".to_string(),
            (manifold::ManifoldType::RN, nalgebra::dvector![1.0]),
        );

        // Add custom factors (demonstrates extensibility!)
        problem.add_residual_block(&["x1", "x2"], Box::new(RosenbrockFactor1), None);
        problem.add_residual_block(&["x1"], Box::new(RosenbrockFactor2), None);

        // Configure Dog Leg optimizer with appropriate trust region
        let config = DogLegConfig::new()
            .with_max_iterations(100)
            .with_cost_tolerance(1e-8)
            .with_parameter_tolerance(1e-8)
            .with_gradient_tolerance(1e-10)
            .with_trust_region_radius(10.0); // Start with larger trust region

        let mut solver = DogLeg::with_config(config);
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
                optimizer::OptimizationStatus::Converged
                    | optimizer::OptimizationStatus::CostToleranceReached
                    | optimizer::OptimizationStatus::ParameterToleranceReached
                    | optimizer::OptimizationStatus::GradientToleranceReached
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
}
