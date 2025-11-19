//! Gauss-Newton optimization algorithm implementation.
//!
//! The Gauss-Newton method is a fundamental iterative algorithm for solving nonlinear least squares problems
//! of the form:
//!
//! ```text
//! min f(x) = ½||r(x)||² = ½Σᵢ rᵢ(x)²
//! ```
//!
//! where `r: ℝⁿ → ℝᵐ` is the residual vector function.
//!
//! # Algorithm Overview
//!
//! The Gauss-Newton method solves the normal equations at each iteration:
//!
//! ```text
//! J^T·J·h = -J^T·r
//! ```
//!
//! where:
//! - `J` is the Jacobian matrix (m × n) of partial derivatives ∂rᵢ/∂xⱼ
//! - `r` is the residual vector (m × 1)
//! - `h` is the step vector (n × 1)
//!
//! The approximated Hessian `H ≈ J^T·J` replaces the true Hessian `∇²f = J^T·J + Σᵢ rᵢ·∇²rᵢ`,
//! which works well when residuals are small or nearly linear.
//!
//! ## Convergence Properties
//!
//! - **Quadratic convergence** near the solution when the Gauss-Newton approximation is valid
//! - **May diverge** if the initial guess is far from the optimum or the problem is ill-conditioned
//! - **No step size control** - always takes the full Newton step without damping
//!
//! ## When to Use
//!
//! Gauss-Newton is most effective when:
//! - The problem is well-conditioned with `J^T·J` having good numerical properties
//! - The initial parameter guess is close to the solution
//! - Fast convergence is prioritized over robustness
//! - Residuals at the solution are expected to be small
//!
//! For ill-conditioned problems or poor initial guesses, consider:
//! - [`LevenbergMarquardt`](crate::optimizer::LevenbergMarquardt) for adaptive damping
//! - [`DogLeg`](crate::optimizer::DogLeg) for trust region control
//!
//! # Implementation Features
//!
//! - **Sparse matrix support**: Efficient handling of large-scale problems via `faer` sparse library
//! - **Robust linear solvers**: Choice between Cholesky (fast) and QR (stable) factorizations
//! - **Jacobi scaling**: Optional diagonal preconditioning to improve conditioning
//! - **Manifold operations**: Support for optimization on Lie groups (SE2, SE3, SO2, SO3)
//! - **Comprehensive diagnostics**: Detailed convergence and performance summaries
//!
//! # Mathematical Background
//!
//! At each iteration k, the algorithm:
//!
//! 1. **Linearizes** the problem around current estimate xₖ: `r(xₖ + h) ≈ r(xₖ) + J(xₖ)·h`
//! 2. **Solves** the normal equations for step h: `J^T·J·h = -J^T·r`
//! 3. **Updates** parameters: `xₖ₊₁ = xₖ ⊕ h` (using manifold plus operation)
//! 4. **Checks** convergence criteria (cost, gradient, parameter change)
//!
//! The method terminates when cost change, gradient norm, or parameter update fall below
//! specified tolerances, or when maximum iterations are reached.
//!
//! # Examples
//!
//! ## Basic usage
//!
//! ```no_run
//! use apex_solver::optimizer::GaussNewton;
//! use apex_solver::core::problem::Problem;
//! use std::collections::HashMap;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create optimization problem
//! let mut problem = Problem::new();
//! // ... add residual blocks (factors) to problem ...
//!
//! // Set up initial parameter values
//! let initial_values = HashMap::new();
//! // ... initialize parameters ...
//!
//! // Create solver with default configuration
//! let mut solver = GaussNewton::new();
//!
//! // Run optimization
//! let result = solver.optimize(&problem, &initial_values)?;
//!
//! println!("Converged: {:?}", result.status);
//! println!("Final cost: {:.6e}", result.final_cost);
//! println!("Iterations: {}", result.iterations);
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced configuration
//!
//! ```no_run
//! use apex_solver::optimizer::gauss_newton::{GaussNewtonConfig, GaussNewton};
//! use apex_solver::linalg::LinearSolverType;
//!
//! # fn main() {
//! let config = GaussNewtonConfig::new()
//!     .with_max_iterations(100)
//!     .with_cost_tolerance(1e-8)
//!     .with_parameter_tolerance(1e-8)
//!     .with_gradient_tolerance(1e-10)
//!     .with_linear_solver_type(LinearSolverType::SparseQR)  // More stable
//!     .with_jacobi_scaling(true);  // Improve conditioning
//!
//! let mut solver = GaussNewton::with_config(config);
//! # }
//! ```
//!
//! # References
//!
//! - Nocedal, J. & Wright, S. (2006). *Numerical Optimization* (2nd ed.). Springer. Chapter 10.
//! - Madsen, K., Nielsen, H. B., & Tingleff, O. (2004). *Methods for Non-Linear Least Squares Problems* (2nd ed.).
//! - Björck, Å. (1996). *Numerical Methods for Least Squares Problems*. SIAM.

use crate::{core::problem, error, linalg, manifold, optimizer};

use faer::sparse;
use std::{collections, fmt, time};
use tracing::debug;

/// Summary statistics for the Gauss-Newton optimization process.
#[derive(Debug, Clone)]
pub struct GaussNewtonSummary {
    /// Initial cost value
    pub initial_cost: f64,
    /// Final cost value
    pub final_cost: f64,
    /// Total number of iterations performed
    pub iterations: usize,
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
    /// Trust region ratio (not used in Gauss-Newton, always 0.0)
    pub tr_ratio: f64,
    /// Trust region radius (not used in Gauss-Newton, always 0.0)
    pub tr_radius: f64,
    /// Linear solver iterations (0 for direct solvers like Cholesky)
    pub ls_iter: usize,
    /// Time taken for this iteration in milliseconds
    pub iter_time_ms: f64,
    /// Total elapsed time since optimization started in milliseconds
    pub total_time_ms: f64,
    /// Whether the step was accepted (always true for Gauss-Newton)
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

impl fmt::Display for GaussNewtonSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Determine if converged
        let converged = matches!(
            self.convergence_status,
            optimizer::OptimizationStatus::Converged
                | optimizer::OptimizationStatus::CostToleranceReached
                | optimizer::OptimizationStatus::GradientToleranceReached
                | optimizer::OptimizationStatus::ParameterToleranceReached
        );

        writeln!(f, "Gauss-Newton Final Result")?;

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
        writeln!(f, "  Total: {}", self.iterations)?;
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

/// Configuration parameters for the Gauss-Newton optimizer.
///
/// Controls the behavior of the Gauss-Newton algorithm including convergence criteria,
/// linear solver selection, and numerical stability enhancements.
///
/// # Builder Pattern
///
/// All configuration options can be set using the builder pattern:
///
/// ```
/// use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
/// use apex_solver::linalg::LinearSolverType;
///
/// let config = GaussNewtonConfig::new()
///     .with_max_iterations(50)
///     .with_cost_tolerance(1e-6)
///     .with_linear_solver_type(LinearSolverType::SparseQR);
/// ```
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
/// - [`GaussNewton`] - The solver that uses this configuration
/// - [`LevenbergMarquardtConfig`](crate::optimizer::LevenbergMarquardtConfig) - For adaptive damping
/// - [`DogLegConfig`](crate::optimizer::DogLegConfig) - For trust region methods
#[derive(Clone)]
pub struct GaussNewtonConfig {
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
    /// Use Jacobi column scaling (preconditioning)
    ///
    /// When enabled, normalizes Jacobian columns by their L2 norm before solving.
    /// This can improve convergence for problems with mixed parameter scales
    /// (e.g., positions in meters + angles in radians) but adds ~5-10% overhead.
    ///
    /// Default: false (Gauss-Newton is typically used on well-conditioned problems)
    pub use_jacobi_scaling: bool,
    /// Small regularization to ensure J^T·J is positive definite
    ///
    /// Pure Gauss-Newton (λ=0) can fail when J^T·J is singular or near-singular.
    /// Adding a tiny diagonal regularization (e.g., 1e-10) ensures numerical stability
    /// while maintaining the fast convergence of Gauss-Newton.
    ///
    /// Default: 1e-10 (very small, practically identical to pure Gauss-Newton)
    pub min_diagonal: f64,

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

    /// Compute per-variable covariance matrices (uncertainty estimation)
    ///
    /// When enabled, computes covariance by inverting the Hessian matrix after
    /// convergence. The full covariance matrix is extracted into per-variable
    /// blocks stored in both Variable structs and optimier::SolverResult.
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

impl Default for GaussNewtonConfig {
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
            use_jacobi_scaling: false,
            min_diagonal: 1e-10,
            // New Ceres-compatible termination parameters
            min_cost_threshold: None,
            max_condition_number: None,
            compute_covariances: false,
            #[cfg(feature = "visualization")]
            enable_visualization: false,
        }
    }
}

impl GaussNewtonConfig {
    /// Create a new Gauss-Newton configuration with default values.
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

    /// Enable or disable Jacobi column scaling (preconditioning).
    ///
    /// When enabled, normalizes Jacobian columns by their L2 norm before solving.
    /// Can improve convergence for mixed-scale problems but adds ~5-10% overhead.
    pub fn with_jacobi_scaling(mut self, use_jacobi_scaling: bool) -> Self {
        self.use_jacobi_scaling = use_jacobi_scaling;
        self
    }

    /// Set the minimum diagonal regularization for numerical stability.
    ///
    /// A small value (e.g., 1e-10) ensures J^T·J is positive definite while
    /// maintaining the fast convergence of pure Gauss-Newton.
    pub fn with_min_diagonal(mut self, min_diagonal: f64) -> Self {
        self.min_diagonal = min_diagonal;
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
            "\nConfiguration:\n  Solver:        Gauss-Newton\n  Linear solver: {:?}\n  Convergence Criteria:\n  Max iterations:      {}\n  Cost tolerance:      {:.2e}\n  Parameter tolerance: {:.2e}\n  Gradient tolerance:  {:.2e}\n  Timeout:             {:?}\n  Numerical Settings:\n  Jacobi scaling:      {}\n  Compute covariances: {}",
            self.linear_solver_type,
            self.max_iterations,
            self.cost_tolerance,
            self.parameter_tolerance,
            self.gradient_tolerance,
            self.timeout,
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

/// State for optimization iteration
struct LinearizerResult {
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
}

/// Result from cost evaluation
struct CostEvaluation {
    new_cost: f64,
    cost_reduction: f64,
}

/// Gauss-Newton solver for nonlinear least squares optimization.
///
/// Implements the classical Gauss-Newton algorithm which solves `J^T·J·h = -J^T·r` at each
/// iteration to find the step `h`. This provides fast quadratic convergence near the solution
/// but may diverge for poor initial guesses or ill-conditioned problems.
///
/// # Algorithm
///
/// At each iteration k:
/// 1. Compute residual `r(xₖ)` and Jacobian `J(xₖ)`
/// 2. Form normal equations: `(J^T·J)·h = -J^T·r`
/// 3. Solve for step `h` using Cholesky or QR factorization
/// 4. Update parameters: `xₖ₊₁ = xₖ ⊕ h` (manifold plus operation)
/// 5. Check convergence criteria
///
/// # Examples
///
/// ```no_run
/// use apex_solver::optimizer::GaussNewton;
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
/// let mut solver = GaussNewton::new();
/// let result = solver.optimize(&problem, &initial_values)?;
///
/// println!("Status: {:?}", result.status);
/// println!("Final cost: {}", result.final_cost);
/// # Ok(())
/// # }
/// ```
///
/// # See Also
///
/// - [`GaussNewtonConfig`] - Configuration options
/// - [`LevenbergMarquardt`](crate::optimizer::LevenbergMarquardt) - For adaptive damping
/// - [`DogLeg`](crate::optimizer::DogLeg) - For trust region control
pub struct GaussNewton {
    config: GaussNewtonConfig,
    jacobi_scaling: Option<sparse::SparseColMat<usize, f64>>,
    observers: optimizer::OptObserverVec,
}

impl Default for GaussNewton {
    fn default() -> Self {
        Self::new()
    }
}

impl GaussNewton {
    /// Create a new Gauss-Newton solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(GaussNewtonConfig::default())
    }

    /// Create a new Gauss-Newton solver with the given configuration.
    pub fn with_config(config: GaussNewtonConfig) -> Self {
        Self {
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
    /// use apex_solver::optimizer::GaussNewton;
    /// # use apex_solver::optimizer::OptObserver;
    /// # use std::collections::HashMap;
    /// # use apex_solver::core::problem::VariableEnum;
    ///
    /// # struct MyObserver;
    /// # impl OptObserver for MyObserver {
    /// #     fn on_step(&self, _: &HashMap<String, VariableEnum>, _: usize) {}
    /// # }
    /// let mut solver = GaussNewton::new();
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
        }
    }

    /// Check convergence criteria
    /// Check convergence using comprehensive termination criteria.
    ///
    /// Implements 8 termination criteria following Ceres Solver standards for Gauss-Newton:
    ///
    /// 1. **Gradient Norm (First-Order Optimality)**: ||g||∞ ≤ gradient_tolerance
    /// 2. **Parameter Change Tolerance**: ||h|| ≤ parameter_tolerance * (||x|| + parameter_tolerance)
    /// 3. **Function Value Change Tolerance**: |ΔF| < cost_tolerance * F
    /// 4. **Objective Function Cutoff**: F_new < min_cost_threshold (optional)
    /// 5. **Singular/Ill-Conditioned Jacobian**: Detected during linear solve
    /// 6. **Invalid Numerical Values**: NaN or Inf in cost or parameters
    /// 7. **Maximum Iterations**: iteration >= max_iterations
    /// 8. **Timeout**: elapsed >= timeout
    ///
    /// **Note**: Trust region radius termination is NOT applicable to Gauss-Newton
    /// as it is a line search method, not a trust region method.
    ///
    /// # Arguments
    ///
    /// * `iteration` - Current iteration number
    /// * `current_cost` - Cost before applying the step
    /// * `new_cost` - Cost after applying the step
    /// * `parameter_norm` - L2 norm of current parameter vector ||x||
    /// * `parameter_update_norm` - L2 norm of parameter update step ||h||
    /// * `gradient_norm` - Infinity norm of gradient ||g||∞
    /// * `elapsed` - Elapsed time since optimization start
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
        elapsed: time::Duration,
    ) -> Option<optimizer::OptimizationStatus> {
        // ========================================================================
        // CRITICAL SAFETY CHECKS (perform first, before convergence checks)
        // ========================================================================

        // CRITERION 6: Invalid Numerical Values (NaN/Inf)
        // Always check for numerical instability first
        if !new_cost.is_finite() || !parameter_update_norm.is_finite() || !gradient_norm.is_finite()
        {
            return Some(optimizer::OptimizationStatus::InvalidNumericalValues);
        }

        // CRITERION 8: Timeout
        // Check wall-clock time limit
        if let Some(timeout) = self.config.timeout
            && elapsed >= timeout
        {
            return Some(optimizer::OptimizationStatus::Timeout);
        }

        // CRITERION 7: Maximum Iterations
        // Check iteration count limit
        if iteration >= self.config.max_iterations {
            return Some(optimizer::OptimizationStatus::MaxIterationsReached);
        }

        // ========================================================================
        // CONVERGENCE CRITERIA
        // ========================================================================
        // Note: Gauss-Newton always accepts the computed step (no step acceptance check)

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

        // CRITERION 5: Singular/Ill-Conditioned Jacobian
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

        sparse::SparseColMat::try_new_from_triplets(cols, cols, &jacobi_scaling_vec).map_err(|e| {
            optimizer::OptimizerError::JacobiScalingCreation(e.to_string()).log_with_source(e)
        })
    }

    /// Initialize optimization state from problem and initial parameters
    fn initialize_optimization_state(
        &self,
        problem: &problem::Problem,
        initial_params: &collections::HashMap<
            String,
            (manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<LinearizerResult, error::ApexError> {
        // Initialize variables from initial values
        let variables = problem.initialize_variables(initial_params);

        // Create column mapping for variables
        let mut variable_index_map = collections::HashMap::new();
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
        let current_cost = optimizer::compute_cost(&residual);
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

    /// Compute Gauss-Newton step by solving the normal equations
    fn compute_gauss_newton_step(
        &self,
        residuals: &faer::Mat<f64>,
        scaled_jacobian: &sparse::SparseColMat<usize, f64>,
        linear_solver: &mut Box<dyn linalg::SparseLinearSolver>,
    ) -> Option<StepResult> {
        // Solve the Gauss-Newton equation: J^T·J·Δx = -J^T·r
        // Use min_diagonal for numerical stability (tiny regularization)
        let residuals_owned = residuals.as_ref().to_owned();
        let scaled_step = linear_solver
            .solve_normal_equation(&residuals_owned, scaled_jacobian)
            .ok()?;

        // Get gradient from the solver (J^T * r)
        let gradient = linear_solver.get_gradient()?;
        // Compute gradient norm for convergence check
        let gradient_norm = gradient.norm_l2();

        // Apply inverse Jacobi scaling to get final step (if enabled)
        let step = if self.config.use_jacobi_scaling {
            let scaling = self.jacobi_scaling.as_ref()?;
            &scaled_step * scaling
        } else {
            scaled_step
        };

        Some(StepResult {
            step,
            gradient_norm,
        })
    }

    /// Apply step to parameters and evaluate new cost
    fn apply_step_and_evaluate_cost(
        &self,
        step_result: &StepResult,
        state: &mut LinearizerResult,
        problem: &problem::Problem,
    ) -> error::ApexResult<CostEvaluation> {
        // Apply parameter updates using manifold operations
        let _step_norm = optimizer::apply_parameter_step(
            &mut state.variables,
            step_result.step.as_ref(),
            &state.sorted_vars,
        );

        // Compute new cost (residual only, no Jacobian needed for step evaluation)
        let new_residual = problem.compute_residual_sparse(&state.variables)?;
        let new_cost = optimizer::compute_cost(&new_residual);

        // Compute cost reduction
        let cost_reduction = state.current_cost - new_cost;

        // Update current cost
        state.current_cost = new_cost;

        Ok(CostEvaluation {
            new_cost,
            cost_reduction,
        })
    }

    /// Create optimization summary
    #[allow(clippy::too_many_arguments)]
    fn create_summary(
        &self,
        initial_cost: f64,
        final_cost: f64,
        iterations: usize,
        max_gradient_norm: f64,
        final_gradient_norm: f64,
        max_parameter_update_norm: f64,
        final_parameter_update_norm: f64,
        total_cost_reduction: f64,
        total_time: time::Duration,
        iteration_history: Vec<IterationStats>,
        convergence_status: optimizer::OptimizationStatus,
    ) -> GaussNewtonSummary {
        GaussNewtonSummary {
            initial_cost,
            final_cost,
            iterations,
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

    pub fn optimize(
        &mut self,
        problem: &problem::Problem,
        initial_params: &collections::HashMap<
            String,
            (manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<
        optimizer::SolverResult<collections::HashMap<String, problem::VariableEnum>>,
        error::ApexError,
    > {
        let start_time = time::Instant::now();
        let mut iteration = 0;
        let mut cost_evaluations = 1; // Initial cost evaluation
        let mut jacobian_evaluations = 0;

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

        // Print configuration and header if debug level is enabled
        if tracing::enabled!(tracing::Level::DEBUG) {
            self.config.print_configuration();
            IterationStats::print_header();
        }

        // Main optimization loop
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

            // Compute Gauss-Newton step
            let step_result = match self.compute_gauss_newton_step(
                &residuals,
                &scaled_jacobian,
                &mut linear_solver,
            ) {
                Some(result) => result,
                None => {
                    return Err(error::ApexError::Solver(
                        "Linear solver failed to solve Gauss-Newton system".to_string(),
                    ));
                }
            };

            // Update tracking variables
            max_gradient_norm = max_gradient_norm.max(step_result.gradient_norm);
            final_gradient_norm = step_result.gradient_norm;
            let step_norm = step_result.step.norm_l2();
            max_parameter_update_norm = max_parameter_update_norm.max(step_norm);
            final_parameter_update_norm = step_norm;

            // Capture cost before applying step (for convergence check)
            let cost_before_step = state.current_cost;

            // Apply step and evaluate new cost
            let cost_eval = self.apply_step_and_evaluate_cost(&step_result, &mut state, problem)?;
            cost_evaluations += 1;
            total_cost_reduction += cost_eval.cost_reduction;

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
                    tr_ratio: 0.0,  // Not used in Gauss-Newton
                    tr_radius: 0.0, // Not used in Gauss-Newton
                    ls_iter: 0,     // Direct solver (Cholesky) has no iterations
                    iter_time_ms: iter_elapsed_ms,
                    total_time_ms: total_elapsed_ms,
                    accepted: true, // Gauss-Newton always accepts steps
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
                None, // Gauss-Newton doesn't use damping
                step_norm,
                None, // Gauss-Newton doesn't use step quality
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

            // Compute parameter norm for convergence check
            let parameter_norm = Self::compute_parameter_norm(&state.variables);

            // Check convergence using comprehensive termination criteria
            let elapsed = start_time.elapsed();
            if let Some(status) = self.check_convergence(
                iteration,
                cost_before_step,
                cost_eval.new_cost,
                parameter_norm,
                step_norm,
                step_result.gradient_norm,
                elapsed,
            ) {
                let summary = self.create_summary(
                    state.initial_cost,
                    state.current_cost,
                    iteration + 1,
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

            iteration += 1;
        }
    }
}

impl optimizer::Solver for GaussNewton {
    type Config = GaussNewtonConfig;
    type Error = error::ApexError;

    fn new() -> Self {
        Self::default()
    }

    fn optimize(
        &mut self,
        problem: &problem::Problem,
        initial_params: &std::collections::HashMap<
            String,
            (manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<
        optimizer::SolverResult<std::collections::HashMap<String, problem::VariableEnum>>,
        Self::Error,
    > {
        self.optimize(problem, initial_params)
    }
}

#[cfg(test)]
mod tests {
    use crate::{core::problem, factors, manifold, optimizer};
    use nalgebra::dvector;
    use std::collections;

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
            (manifold::ManifoldType::RN, dvector![-1.2]),
        );
        initial_values.insert(
            "x2".to_string(),
            (manifold::ManifoldType::RN, dvector![1.0]),
        );

        // Add custom factors (demonstrates extensibility!)
        problem.add_residual_block(&["x1", "x2"], Box::new(RosenbrockFactor1), None);
        problem.add_residual_block(&["x1"], Box::new(RosenbrockFactor2), None);

        // Configure Gauss-Newton optimizer
        let config = optimizer::gauss_newton::GaussNewtonConfig::new()
            .with_max_iterations(100)
            .with_cost_tolerance(1e-8)
            .with_parameter_tolerance(1e-8)
            .with_gradient_tolerance(1e-10);

        let mut solver = optimizer::GaussNewton::with_config(config);
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

        println!("Rosenbrock optimization result (Gauss-Newton):");
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
