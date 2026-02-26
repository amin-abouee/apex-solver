//! Optimization solvers for nonlinear least squares problems.
//!
//! This module provides various optimization algorithms specifically designed
//! for nonlinear least squares problems commonly found in computer vision:
//! - Levenberg-Marquardt algorithm
//! - Gauss-Newton algorithm
//! - Dog Leg algorithm

use crate::core::problem::{Problem, SymbolicStructure, VariableEnum};
use crate::error;
use crate::linalg::{self, SparseCholeskySolver, SparseLinearSolver, SparseQRSolver};
use apex_manifolds::ManifoldType;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, MatRef};
use nalgebra::DVector;
use std::collections::HashMap;
use std::time::{self, Duration};
use std::{
    fmt,
    fmt::{Display, Formatter},
};
use thiserror::Error;
use tracing::{debug, error};

pub mod dog_leg;
pub mod gauss_newton;
pub mod levenberg_marquardt;

pub use dog_leg::DogLeg;
pub use gauss_newton::GaussNewton;
pub use levenberg_marquardt::LevenbergMarquardt;

// Re-export observer types from the observers module
pub use crate::observers::{OptObserver, OptObserverVec};

/// Type of optimization solver algorithm to use
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerType {
    /// Levenberg-Marquardt algorithm (robust, adaptive damping)
    #[default]
    LevenbergMarquardt,
    /// Gauss-Newton algorithm (fast convergence, may be unstable)
    GaussNewton,
    /// Dog Leg algorithm (trust region method)
    DogLeg,
}

impl Display for OptimizerType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerType::LevenbergMarquardt => write!(f, "Levenberg-Marquardt"),
            OptimizerType::GaussNewton => write!(f, "Gauss-Newton"),
            OptimizerType::DogLeg => write!(f, "Dog Leg"),
        }
    }
}

/// Optimizer-specific error types for apex-solver
#[derive(Debug, Clone, Error)]
pub enum OptimizerError {
    /// Linear system solve failed during optimization
    #[error("Linear system solve failed: {0}")]
    LinearSolveFailed(String),

    /// Maximum iterations reached without achieving convergence
    #[error("Maximum iterations ({max_iters}) reached without convergence")]
    MaxIterationsReached { max_iters: usize },

    /// Trust region radius became too small
    #[error("Trust region radius became too small: {radius:.6e} < {min_radius:.6e}")]
    TrustRegionFailure { radius: f64, min_radius: f64 },

    /// Damping parameter became too large (LM-specific)
    #[error("Damping parameter became too large: {damping:.6e} > {max_damping:.6e}")]
    DampingFailure { damping: f64, max_damping: f64 },

    /// Cost increased unexpectedly when it should decrease
    #[error("Cost increased unexpectedly: {old_cost:.6e} -> {new_cost:.6e}")]
    CostIncrease { old_cost: f64, new_cost: f64 },

    /// Jacobian computation failed
    #[error("Jacobian computation failed: {0}")]
    JacobianFailed(String),

    /// Invalid optimization parameters provided
    #[error("Invalid optimization parameters: {0}")]
    InvalidParameters(String),

    /// Numerical instability detected (NaN, Inf in cost, gradient, or parameters)
    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),

    /// Linear algebra operation failed
    #[error("Linear algebra error: {0}")]
    LinAlg(#[from] linalg::LinAlgError),

    /// Problem has no variables to optimize
    #[error("Problem has no variables to optimize")]
    EmptyProblem,

    /// Problem has no residual blocks
    #[error("Problem has no residual blocks")]
    NoResidualBlocks,

    /// Jacobi scaling matrix creation failed
    #[error("Failed to create Jacobi scaling matrix: {0}")]
    JacobiScalingCreation(String),

    /// Jacobi scaling not initialized when expected
    #[error("Jacobi scaling not initialized")]
    JacobiScalingNotInitialized,

    /// Unknown or unsupported linear solver type
    #[error("Unknown linear solver type: {0}")]
    UnknownLinearSolver(String),
}

impl OptimizerError {
    /// Log the error with tracing::error and return self for chaining
    ///
    /// This method allows for a consistent error logging pattern throughout
    /// the optimizer module, ensuring all errors are properly recorded.
    ///
    /// # Example
    /// ```
    /// # use apex_solver::optimizer::OptimizerError;
    /// # fn operation() -> Result<(), OptimizerError> { Ok(()) }
    /// # fn example() -> Result<(), OptimizerError> {
    /// operation()
    ///     .map_err(|e| e.log())?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn log(self) -> Self {
        error!("{}", self);
        self
    }

    /// Log the error with the original source error from a third-party library
    ///
    /// This method logs both the OptimizerError and the underlying error
    /// from external libraries. This provides full debugging context when
    /// errors occur in third-party code.
    ///
    /// # Arguments
    /// * `source_error` - The original error from the third-party library (must implement Debug)
    ///
    /// # Example
    /// ```
    /// # use apex_solver::optimizer::OptimizerError;
    /// # fn sparse_matrix_op() -> Result<(), std::io::Error> { Ok(()) }
    /// # fn example() -> Result<(), OptimizerError> {
    /// sparse_matrix_op()
    ///     .map_err(|e| {
    ///         OptimizerError::JacobiScalingCreation(e.to_string())
    ///             .log_with_source(e)
    ///     })?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn log_with_source<E: std::fmt::Debug>(self, source_error: E) -> Self {
        error!("{} | Source: {:?}", self, source_error);
        self
    }
}

/// Result type for optimizer operations
pub type OptimizerResult<T> = Result<T, OptimizerError>;

// State information during iterative optimization.
// #[derive(Debug, Clone)]
// pub struct IterativeState {
//     /// Current iteration number
//     pub iteration: usize,
//     /// Current cost value
//     pub cost: f64,
//     /// Current gradient norm
//     pub gradient_norm: f64,
//     /// Current parameter update norm
//     pub parameter_update_norm: f64,
//     /// Time elapsed since start
//     pub elapsed_time: Duration,
// }

/// Detailed convergence information.
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Final gradient norm
    pub final_gradient_norm: f64,
    /// Final parameter update norm
    pub final_parameter_update_norm: f64,
    /// Cost function evaluation count
    pub cost_evaluations: usize,
    /// Jacobian evaluation count
    pub jacobian_evaluations: usize,
}

impl Display for ConvergenceInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Final gradient norm: {:.2e}, Final parameter update norm: {:.2e}, Cost evaluations: {}, Jacobian evaluations: {}",
            self.final_gradient_norm,
            self.final_parameter_update_norm,
            self.cost_evaluations,
            self.jacobian_evaluations
        )
    }
}

/// Status of an optimization process
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationStatus {
    /// Optimization converged successfully
    Converged,
    /// Maximum number of iterations reached
    MaxIterationsReached,
    /// Cost function tolerance reached
    CostToleranceReached,
    /// Parameter tolerance reached
    ParameterToleranceReached,
    /// Gradient tolerance reached
    GradientToleranceReached,
    /// Optimization failed due to numerical issues
    NumericalFailure,
    /// User requested termination
    UserTerminated,
    /// Timeout reached
    Timeout,
    /// Trust region radius fell below minimum threshold
    TrustRegionRadiusTooSmall,
    /// Objective function fell below user-specified cutoff
    MinCostThresholdReached,
    /// Jacobian matrix is singular or ill-conditioned
    IllConditionedJacobian,
    /// NaN or Inf detected in cost or parameters
    InvalidNumericalValues,
    /// Other failure
    Failed(String),
}

impl Display for OptimizationStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            OptimizationStatus::Converged => write!(f, "Converged"),
            OptimizationStatus::MaxIterationsReached => write!(f, "Maximum iterations reached"),
            OptimizationStatus::CostToleranceReached => write!(f, "Cost tolerance reached"),
            OptimizationStatus::ParameterToleranceReached => {
                write!(f, "Parameter tolerance reached")
            }
            OptimizationStatus::GradientToleranceReached => write!(f, "Gradient tolerance reached"),
            OptimizationStatus::NumericalFailure => write!(f, "Numerical failure"),
            OptimizationStatus::UserTerminated => write!(f, "User terminated"),
            OptimizationStatus::Timeout => write!(f, "Timeout"),
            OptimizationStatus::TrustRegionRadiusTooSmall => {
                write!(f, "Trust region radius too small")
            }
            OptimizationStatus::MinCostThresholdReached => {
                write!(f, "Minimum cost threshold reached")
            }
            OptimizationStatus::IllConditionedJacobian => {
                write!(f, "Ill-conditioned Jacobian matrix")
            }
            OptimizationStatus::InvalidNumericalValues => {
                write!(f, "Invalid numerical values (NaN/Inf) detected")
            }
            OptimizationStatus::Failed(msg) => write!(f, "Failed: {msg}"),
        }
    }
}

/// Result of a solver execution.
#[derive(Clone)]
pub struct SolverResult<T> {
    /// Final parameters
    pub parameters: T,
    /// Final optimization status
    pub status: OptimizationStatus,
    /// Initial cost value
    pub initial_cost: f64,
    /// Final cost value
    pub final_cost: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Total time elapsed
    pub elapsed_time: time::Duration,
    /// Convergence statistics
    pub convergence_info: Option<ConvergenceInfo>,
    /// Per-variable covariance matrices (uncertainty estimation)
    ///
    /// This is `None` if covariance computation was not enabled in the solver configuration.
    /// When present, it contains a mapping from variable names to their covariance matrices
    /// in tangent space. For example, for SE3 variables this would be 6×6 matrices.
    ///
    /// Enable covariance computation by setting `compute_covariances: true` in the optimizer config.
    pub covariances: Option<HashMap<String, Mat<f64>>>,
}

/// Core trait for optimization solvers.
pub trait Solver {
    /// Configuration type for this solver
    type Config;
    /// Error type
    type Error;

    /// Create a new solver with the given configuration
    fn new() -> Self;

    /// Optimize the problem to minimize the cost function
    fn optimize(
        &mut self,
        problem: &Problem,
        initial_params: &HashMap<String, (ManifoldType, DVector<f64>)>,
    ) -> Result<SolverResult<HashMap<String, VariableEnum>>, Self::Error>;
}

/// Apply parameter update step to all variables.
///
/// This is a common operation used by all optimizers (Levenberg-Marquardt, Gauss-Newton, Dog Leg).
/// It applies a tangent space perturbation to each variable using the proper manifold plus operation.
///
/// # Arguments
/// * `variables` - Mutable map of variables to update
/// * `step` - Full step vector in tangent space (faer matrix view)
/// * `variable_order` - Ordered list of variable names (defines indexing into step vector)
///
/// # Returns
/// * Step norm (L2 norm) for convergence checking
///
/// # Implementation Notes
/// The step vector contains concatenated tangent vectors for all variables in the order
/// specified by `variable_order`. Each variable's tangent space dimension determines
/// how many elements it occupies in the step vector.
///
pub fn apply_parameter_step(
    variables: &mut HashMap<String, VariableEnum>,
    step: MatRef<f64>,
    variable_order: &[String],
) -> f64 {
    let mut step_offset = 0;

    for var_name in variable_order {
        if let Some(var) = variables.get_mut(var_name) {
            let var_size = var.get_size();
            let var_step = step.subrows(step_offset, var_size);

            // Delegate to VariableEnum's apply_tangent_step method
            // This handles all manifold types (SE2, SE3, SO2, SO3, Rn)
            var.apply_tangent_step(var_step);

            step_offset += var_size;
        }
    }

    // Compute and return step norm for convergence checking
    step.norm_l2()
}

/// Apply negative parameter step to rollback variables.
///
/// This is used when an optimization step is rejected (e.g., in trust region methods).
/// It applies the negative of a tangent space perturbation to revert the previous update.
///
/// # Arguments
/// * `variables` - Mutable map of variables to revert
/// * `step` - Full step vector in tangent space (faer matrix view) to negate
/// * `variable_order` - Ordered list of variable names (defines indexing into step vector)
///
pub fn apply_negative_parameter_step(
    variables: &mut HashMap<String, VariableEnum>,
    step: MatRef<f64>,
    variable_order: &[String],
) {
    // Create a negated version of the step vector
    let mut negative_step = Mat::zeros(step.nrows(), 1);
    for i in 0..step.nrows() {
        negative_step[(i, 0)] = -step[(i, 0)];
    }

    // Apply the negative step using the standard apply_parameter_step function
    apply_parameter_step(variables, negative_step.as_ref(), variable_order);
}

pub fn compute_cost(residual: &Mat<f64>) -> f64 {
    let cost = residual.norm_l2();
    0.5 * cost * cost
}

// ============================================================================
// Shared optimizer utilities
// ============================================================================
// The following types and functions are shared across all three optimizer
// implementations (Levenberg-Marquardt, Gauss-Newton, Dog Leg) to eliminate
// code duplication.

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
    /// Trust region radius (damping parameter λ for LM, Δ for Dog Leg)
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

/// Result of optimization state initialization, shared by all optimizers.
pub struct InitializedState {
    pub variables: HashMap<String, VariableEnum>,
    pub variable_index_map: HashMap<String, usize>,
    pub sorted_vars: Vec<String>,
    pub symbolic_structure: SymbolicStructure,
    pub current_cost: f64,
    pub initial_cost: f64,
}

/// Compute total parameter vector norm ||x|| across all variables.
pub fn compute_parameter_norm(variables: &HashMap<String, VariableEnum>) -> f64 {
    variables
        .values()
        .map(|v| {
            let vec = v.to_vector();
            vec.norm_squared()
        })
        .sum::<f64>()
        .sqrt()
}

/// Create Jacobi scaling diagonal matrix from Jacobian column norms.
///
/// The scaling factor for each column is `1 / (1 + ||col||)`, which normalizes
/// the columns to improve conditioning of the linear system.
pub fn create_jacobi_scaling(
    jacobian: &SparseColMat<usize, f64>,
) -> Result<SparseColMat<usize, f64>, OptimizerError> {
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

    SparseColMat::try_new_from_triplets(cols, cols, &jacobi_scaling_vec)
        .map_err(|e| OptimizerError::JacobiScalingCreation(e.to_string()).log_with_source(e))
}

/// Process Jacobian by applying Jacobi scaling (created on first iteration).
///
/// On `iteration == 0`, creates the scaling matrix and stores it. On subsequent
/// iterations, reuses the cached scaling.
pub fn process_jacobian(
    jacobian: &SparseColMat<usize, f64>,
    jacobi_scaling: &mut Option<SparseColMat<usize, f64>>,
    iteration: usize,
) -> Result<SparseColMat<usize, f64>, OptimizerError> {
    if iteration == 0 {
        let scaling = create_jacobi_scaling(jacobian)?;
        *jacobi_scaling = Some(scaling);
    }
    let scaling = jacobi_scaling
        .as_ref()
        .ok_or_else(|| OptimizerError::JacobiScalingNotInitialized.log())?;
    Ok(jacobian * scaling)
}

/// Initialize optimization state from problem and initial parameters.
///
/// This is the common initialization sequence used by all optimizers:
/// 1. Create variables from initial values
/// 2. Build variable-to-column index mapping
/// 3. Build symbolic sparsity structure for Jacobian
/// 4. Compute initial cost
pub fn initialize_optimization_state(
    problem: &Problem,
    initial_params: &HashMap<String, (ManifoldType, DVector<f64>)>,
) -> Result<InitializedState, error::ApexSolverError> {
    let variables = problem.initialize_variables(initial_params);

    let mut variable_index_map = HashMap::new();
    let mut col_offset = 0;
    let mut sorted_vars: Vec<String> = variables.keys().cloned().collect();
    sorted_vars.sort();

    for var_name in &sorted_vars {
        variable_index_map.insert(var_name.clone(), col_offset);
        col_offset += variables[var_name].get_size();
    }

    let symbolic_structure =
        problem.build_symbolic_structure(&variables, &variable_index_map, col_offset)?;

    let residual = problem.compute_residual_sparse(&variables)?;
    let current_cost = compute_cost(&residual);
    let initial_cost = current_cost;

    Ok(InitializedState {
        variables,
        variable_index_map,
        sorted_vars,
        symbolic_structure,
        current_cost,
        initial_cost,
    })
}

/// Parameters for convergence checking, shared across optimizers.
pub struct ConvergenceParams {
    pub iteration: usize,
    pub current_cost: f64,
    pub new_cost: f64,
    pub parameter_norm: f64,
    pub parameter_update_norm: f64,
    pub gradient_norm: f64,
    pub elapsed: Duration,
    pub step_accepted: bool,
    // Config values
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    pub parameter_tolerance: f64,
    pub cost_tolerance: f64,
    pub min_cost_threshold: Option<f64>,
    pub timeout: Option<Duration>,
    /// Trust region radius (LM damping or DogLeg radius). None for GN.
    pub trust_region_radius: Option<f64>,
    /// Minimum trust region radius threshold. None for GN.
    pub min_trust_region_radius: Option<f64>,
}

/// Check convergence criteria common to all optimizers.
///
/// Returns `Some(status)` if a termination criterion is met, `None` otherwise.
pub fn check_convergence(params: &ConvergenceParams) -> Option<OptimizationStatus> {
    // CRITICAL SAFETY CHECKS (perform first)

    // Invalid Numerical Values (NaN/Inf)
    if !params.new_cost.is_finite()
        || !params.parameter_update_norm.is_finite()
        || !params.gradient_norm.is_finite()
    {
        return Some(OptimizationStatus::InvalidNumericalValues);
    }

    // Timeout
    if let Some(timeout) = params.timeout {
        if params.elapsed >= timeout {
            return Some(OptimizationStatus::Timeout);
        }
    }

    // Maximum Iterations
    if params.iteration >= params.max_iterations {
        return Some(OptimizationStatus::MaxIterationsReached);
    }

    // CONVERGENCE CRITERIA (only check after accepted steps)
    if !params.step_accepted {
        return None;
    }

    // Gradient Norm (First-Order Optimality)
    if params.gradient_norm < params.gradient_tolerance {
        return Some(OptimizationStatus::GradientToleranceReached);
    }

    // Parameter and cost criteria (only after first iteration)
    if params.iteration > 0 {
        // Parameter Change Tolerance (xtol)
        let relative_step_tolerance =
            params.parameter_tolerance * (params.parameter_norm + params.parameter_tolerance);
        if params.parameter_update_norm <= relative_step_tolerance {
            return Some(OptimizationStatus::ParameterToleranceReached);
        }

        // Function Value Change Tolerance (ftol)
        let cost_change = (params.current_cost - params.new_cost).abs();
        let relative_cost_change = cost_change / params.current_cost.max(1e-10);
        if relative_cost_change < params.cost_tolerance {
            return Some(OptimizationStatus::CostToleranceReached);
        }
    }

    // Objective Function Cutoff (optional early stopping)
    if let Some(min_cost) = params.min_cost_threshold {
        if params.new_cost < min_cost {
            return Some(OptimizationStatus::MinCostThresholdReached);
        }
    }

    // Trust Region Radius (LM and DogLeg only)
    if let (Some(radius), Some(min_radius)) =
        (params.trust_region_radius, params.min_trust_region_radius)
    {
        if radius < min_radius {
            return Some(OptimizationStatus::TrustRegionRadiusTooSmall);
        }
    }

    None
}

/// Compute step quality ratio (actual vs predicted reduction).
///
/// Used by Levenberg-Marquardt and Dog Leg optimizers to evaluate
/// whether a proposed step improved the objective function as predicted
/// by the local quadratic model.
///
/// Returns `ρ = actual_reduction / predicted_reduction`, handling
/// near-zero predicted reduction gracefully.
pub fn compute_step_quality(current_cost: f64, new_cost: f64, predicted_reduction: f64) -> f64 {
    let actual_reduction = current_cost - new_cost;
    if predicted_reduction.abs() < 1e-15 {
        if actual_reduction > 0.0 { 1.0 } else { 0.0 }
    } else {
        actual_reduction / predicted_reduction
    }
}

/// Create the appropriate linear solver based on configuration.
///
/// Used by Gauss-Newton and Dog Leg optimizers. Levenberg-Marquardt has its own
/// solver creation logic due to special Schur complement adapter requirements.
pub fn create_linear_solver(solver_type: &linalg::LinearSolverType) -> Box<dyn SparseLinearSolver> {
    match solver_type {
        linalg::LinearSolverType::SparseCholesky => Box::new(SparseCholeskySolver::new()),
        linalg::LinearSolverType::SparseQR => Box::new(SparseQRSolver::new()),
        linalg::LinearSolverType::SparseSchurComplement => {
            // Schur complement solver requires special handling - fallback to Cholesky
            Box::new(SparseCholeskySolver::new())
        }
    }
}

/// Notify observers with current optimization state.
///
/// This is the common observer notification pattern used by all three optimizers.
#[allow(clippy::too_many_arguments)]
pub fn notify_observers(
    observers: &mut OptObserverVec,
    variables: &HashMap<String, VariableEnum>,
    iteration: usize,
    cost: f64,
    gradient_norm: f64,
    damping: Option<f64>,
    step_norm: f64,
    step_quality: Option<f64>,
    linear_solver: &dyn SparseLinearSolver,
) {
    observers.set_iteration_metrics(cost, gradient_norm, damping, step_norm, step_quality);

    if !observers.is_empty() {
        if let (Some(hessian), Some(gradient)) =
            (linear_solver.get_hessian(), linear_solver.get_gradient())
        {
            observers.set_matrix_data(Some(hessian.clone()), Some(gradient.clone()));
        }
    }

    observers.notify(variables, iteration);
}

/// Build a SolverResult from common optimization loop outputs.
///
/// All three optimizers construct SolverResult identically at convergence.
#[allow(clippy::too_many_arguments)]
pub fn build_solver_result(
    status: OptimizationStatus,
    iterations: usize,
    state: InitializedState,
    elapsed: Duration,
    final_gradient_norm: f64,
    final_parameter_update_norm: f64,
    cost_evaluations: usize,
    jacobian_evaluations: usize,
    covariances: Option<HashMap<String, Mat<f64>>>,
) -> SolverResult<HashMap<String, VariableEnum>> {
    SolverResult {
        status,
        iterations,
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
    }
}

/// Unified summary statistics for all optimizer types.
///
/// Replaces the separate `LevenbergMarquardtSummary`, `GaussNewtonSummary`,
/// and `DogLegSummary` structs with a single type that handles algorithm-specific
/// fields via `Option`.
#[derive(Debug, Clone)]
pub struct OptimizerSummary {
    /// Name of the optimizer algorithm
    pub optimizer_name: &'static str,
    /// Initial cost value
    pub initial_cost: f64,
    /// Final cost value
    pub final_cost: f64,
    /// Total number of iterations performed
    pub iterations: usize,
    /// Number of successful steps (None for GN which always accepts)
    pub successful_steps: Option<usize>,
    /// Number of unsuccessful steps (None for GN which always accepts)
    pub unsuccessful_steps: Option<usize>,
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
    /// Final damping parameter (LM only)
    pub final_damping: Option<f64>,
    /// Final trust region radius (DL only)
    pub final_trust_region_radius: Option<f64>,
    /// Step quality ratio (LM only)
    pub rho: Option<f64>,
}

impl Display for OptimizerSummary {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let converged = matches!(
            self.convergence_status,
            OptimizationStatus::Converged
                | OptimizationStatus::CostToleranceReached
                | OptimizationStatus::GradientToleranceReached
                | OptimizationStatus::ParameterToleranceReached
        );

        writeln!(f, "{} Final Result", self.optimizer_name)?;

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
        if let (Some(successful), Some(unsuccessful)) =
            (self.successful_steps, self.unsuccessful_steps)
        {
            writeln!(
                f,
                "  Successful steps:   {} ({:.1}%)",
                successful,
                100.0 * successful as f64 / self.iterations.max(1) as f64
            )?;
            writeln!(
                f,
                "  Unsuccessful steps: {} ({:.1}%)",
                unsuccessful,
                100.0 * unsuccessful as f64 / self.iterations.max(1) as f64
            )?;
        }
        if let Some(radius) = self.final_trust_region_radius {
            writeln!(f)?;
            writeln!(f, "Trust Region:")?;
            writeln!(f, "  Final radius: {:.6e}", radius)?;
        }
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

/// Create an OptimizerSummary from common optimization loop outputs.
#[allow(clippy::too_many_arguments)]
pub fn create_optimizer_summary(
    optimizer_name: &'static str,
    initial_cost: f64,
    final_cost: f64,
    iterations: usize,
    successful_steps: Option<usize>,
    unsuccessful_steps: Option<usize>,
    max_gradient_norm: f64,
    final_gradient_norm: f64,
    max_parameter_update_norm: f64,
    final_parameter_update_norm: f64,
    total_cost_reduction: f64,
    total_time: Duration,
    iteration_history: Vec<IterationStats>,
    convergence_status: OptimizationStatus,
    final_damping: Option<f64>,
    final_trust_region_radius: Option<f64>,
    rho: Option<f64>,
) -> OptimizerSummary {
    OptimizerSummary {
        optimizer_name,
        initial_cost,
        final_cost,
        iterations,
        successful_steps,
        unsuccessful_steps,
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
        final_damping,
        final_trust_region_radius,
        rho,
    }
}
