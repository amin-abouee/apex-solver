//! Optimization solvers for nonlinear least squares problems.
//!
//! This module provides various optimization algorithms specifically designed
//! for nonlinear least squares problems commonly found in computer vision:
//! - Levenberg-Marquardt algorithm
//! - Gauss-Newton algorithm
//! - Dog Leg algorithm

use crate::core::problem::{Problem, VariableEnum};
use crate::error;
use crate::linalg::{
    self, JacobianMode, LinearSolver, SparseCholeskySolver, SparseMode, SparseQRSolver,
};
use crate::linearizer::SymbolicStructure;
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

// Re-export AssemblyBackend so optimizer sub-modules can import it from optimizer::
pub use crate::linearizer::AssemblyBackend;

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

/// Unified optimizer interface. Object-safe — `Box<dyn Optimizer>` is valid.
///
/// All three optimizers ([`LevenbergMarquardt`](crate::optimizer::levenberg_marquardt::LevenbergMarquardt),
/// [`GaussNewton`](crate::optimizer::gauss_newton::GaussNewton),
/// [`DogLeg`](crate::optimizer::dog_leg::DogLeg)) implement this trait.
/// Each optimizer also provides inherent `new()`, `with_config()`, and `optimize()` methods
/// for direct (non-polymorphic) usage.
pub trait Optimizer {
    /// Optimize the problem to minimize the cost function.
    fn optimize(
        &mut self,
        problem: &Problem,
        initial_params: &HashMap<String, (ManifoldType, DVector<f64>)>,
    ) -> Result<SolverResult<HashMap<String, VariableEnum>>, crate::error::ApexSolverError>;
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
    pub symbolic_structure: Option<SymbolicStructure>,
    pub total_dof: usize,
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
/// 3. Build symbolic sparsity structure for Jacobian (sparse mode only)
/// 4. Compute initial cost
///
/// The assembly mode is determined by `problem.jacobian_mode`.
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

    let total_dof = col_offset;

    let symbolic_structure = match problem.jacobian_mode {
        JacobianMode::Sparse => Some(crate::linearizer::cpu::sparse::build_symbolic_structure(
            problem,
            &variables,
            &variable_index_map,
            total_dof,
        )?),
        JacobianMode::Dense => None,
    };

    let residual = problem.compute_residual_sparse(&variables)?;
    let current_cost = compute_cost(&residual);
    let initial_cost = current_cost;

    Ok(InitializedState {
        variables,
        variable_index_map,
        sorted_vars,
        symbolic_structure,
        total_dof,
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
pub fn create_linear_solver(
    solver_type: &linalg::LinearSolverType,
) -> Box<dyn LinearSolver<SparseMode>> {
    match solver_type {
        linalg::LinearSolverType::SparseCholesky => Box::new(SparseCholeskySolver::new()),
        linalg::LinearSolverType::SparseQR => Box::new(SparseQRSolver::new()),
        _ => {
            // SparseSchurComplement requires special handling; DenseCholesky/DenseQR are
            // dispatched via the dense path in each optimizer — all fall back to Cholesky here.
            Box::new(SparseCholeskySolver::new())
        }
    }
}

/// Notify observers with current optimization state (sparse path).
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
    linear_solver: &dyn LinearSolver<SparseMode>,
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

/// Notify observers with current optimization state (generic path).
///
/// For dense mode, the Hessian is converted to sparse for observer compatibility.
/// This is acceptable since observers are for visualization/debugging, not the hot path.
#[allow(clippy::too_many_arguments)]
pub fn notify_observers_generic<M: AssemblyBackend>(
    observers: &mut OptObserverVec,
    variables: &HashMap<String, VariableEnum>,
    iteration: usize,
    cost: f64,
    gradient_norm: f64,
    damping: Option<f64>,
    step_norm: f64,
    step_quality: Option<f64>,
    _linear_solver: &dyn LinearSolver<M>,
) {
    observers.set_iteration_metrics(cost, gradient_norm, damping, step_norm, step_quality);
    // Skip matrix data for generic path since observers expect sparse Hessian.
    // Observer matrix visualization is optional and only used for debugging.
    observers.notify(variables, iteration);
}

/// Process Jacobian with Jacobi scaling (generic over assembly mode).
///
/// On `iteration == 0`, computes the scaling factors and stores them.
/// On subsequent iterations, reuses the cached scaling.
pub fn process_jacobian_generic<M: AssemblyBackend>(
    jacobian: &M::Jacobian,
    jacobi_scaling: &mut Option<Vec<f64>>,
    iteration: usize,
) -> Result<M::Jacobian, OptimizerError> {
    if iteration == 0 {
        let norms = M::compute_column_norms(jacobian);
        let scaling: Vec<f64> = norms.iter().map(|n| 1.0 / (1.0 + n)).collect();
        *jacobi_scaling = Some(scaling);
    }
    let scaling = jacobi_scaling
        .as_ref()
        .ok_or_else(|| OptimizerError::JacobiScalingNotInitialized.log())?;
    Ok(M::apply_column_scaling(jacobian, scaling))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::problem::VariableEnum;
    use crate::core::variable::Variable;
    use crate::factors::Factor;
    use crate::linalg::JacobianMode;
    use apex_manifolds::ManifoldType;
    use apex_manifolds::rn::Rn;
    use faer::Mat;
    use faer::sparse::{SparseColMat, Triplet};
    use nalgebra::{DMatrix, DVector, dvector};
    use std::collections::HashMap;
    use std::time::Duration;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    // -------------------------------------------------------------------------
    // compute_cost
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_cost_known_value() {
        // ||[1, 2]||² * 0.5 = (1 + 4) * 0.5 = 2.5
        let r = Mat::from_fn(2, 1, |i, _| (i + 1) as f64);
        let cost = compute_cost(&r);
        assert!((cost - 2.5).abs() < 1e-12, "expected 2.5, got {cost}");
    }

    #[test]
    fn test_compute_cost_zero_residual() {
        let r = Mat::zeros(3, 1);
        assert_eq!(compute_cost(&r), 0.0);
    }

    // -------------------------------------------------------------------------
    // compute_step_quality
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_step_quality_normal() {
        // actual = 1.0-0.0 = 1.0, predicted = 2.0 → rho = 0.5
        let rho = compute_step_quality(1.0, 0.0, 2.0);
        assert!((rho - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_compute_step_quality_zero_predicted_positive_actual() {
        // predicted ≈ 0, actual > 0 → 1.0
        let rho = compute_step_quality(2.0, 1.0, 0.0);
        assert_eq!(rho, 1.0);
    }

    #[test]
    fn test_compute_step_quality_zero_predicted_nonpositive_actual() {
        // predicted ≈ 0, actual ≤ 0 → 0.0
        let rho = compute_step_quality(1.0, 2.0, 0.0); // actual = -1.0
        assert_eq!(rho, 0.0);
    }

    #[test]
    fn test_compute_step_quality_negative_reduction() {
        // cost increased: actual = 1.0 - 2.0 = -1.0, predicted = 1.0 → -1.0
        let rho = compute_step_quality(1.0, 2.0, 1.0);
        assert!((rho - (-1.0)).abs() < 1e-12);
    }

    // -------------------------------------------------------------------------
    // create_jacobi_scaling
    // -------------------------------------------------------------------------

    fn make_identity_jacobian(n: usize) -> SparseColMat<usize, f64> {
        let triplets: Vec<Triplet<usize, usize, f64>> =
            (0..n).map(|i| Triplet::new(i, i, 1.0)).collect();
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap_or_else(|_| {
            let empty: Vec<Triplet<usize, usize, f64>> = vec![];
            SparseColMat::try_new_from_triplets(0, 0, &empty)
                .unwrap_or_else(|_| panic!("failed to create empty matrix"))
        })
    }

    #[test]
    fn test_create_jacobi_scaling_identity_jacobian() -> TestResult {
        // For identity Jacobian each column has norm 1.0 → scaling = 1/(1+1) = 0.5
        let jac = make_identity_jacobian(3);
        let scaling = create_jacobi_scaling(&jac)?;
        for i in 0..3 {
            let val = scaling.get(i, i).copied().unwrap_or(0.0);
            assert!(
                (val - 0.5).abs() < 1e-12,
                "col {i}: expected 0.5, got {val}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_create_jacobi_scaling_zero_column() -> TestResult {
        // A zero column has norm 0 → scaling = 1/(1+0) = 1.0
        let triplets = vec![Triplet::new(0_usize, 0_usize, 1.0_f64)];
        let jac = SparseColMat::try_new_from_triplets(2, 2, &triplets)?;
        let scaling = create_jacobi_scaling(&jac)?;
        // col 0: norm=1 → 0.5; col 1: norm=0 → 1.0
        let s0 = scaling.get(0, 0).copied().unwrap_or(0.0);
        let s1 = scaling.get(1, 1).copied().unwrap_or(0.0);
        assert!((s0 - 0.5).abs() < 1e-12);
        assert!((s1 - 1.0).abs() < 1e-12);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // process_jacobian
    // -------------------------------------------------------------------------

    #[test]
    fn test_process_jacobian_creates_at_iter0() -> TestResult {
        let jac = make_identity_jacobian(2);
        let mut cache: Option<SparseColMat<usize, f64>> = None;
        let scaled = process_jacobian(&jac, &mut cache, 0)?;
        assert!(cache.is_some(), "scaling should be cached after iter 0");
        // Each diagonal entry should be scaled by 0.5
        let s = scaled.get(0, 0).copied().unwrap_or(0.0);
        assert!((s - 0.5).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_process_jacobian_reuses_at_iter1() -> TestResult {
        let jac = make_identity_jacobian(2);
        let mut cache: Option<SparseColMat<usize, f64>> = None;
        // build cache at iter 0
        process_jacobian(&jac, &mut cache, 0)?;
        // now use cached at iter 1
        let scaled = process_jacobian(&jac, &mut cache, 1)?;
        let s = scaled.get(0, 0).copied().unwrap_or(0.0);
        assert!((s - 0.5).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_process_jacobian_error_at_iter1_without_init() {
        let jac = make_identity_jacobian(2);
        let mut cache: Option<SparseColMat<usize, f64>> = None;
        // skip iter 0 — should error
        let result = process_jacobian(&jac, &mut cache, 1);
        assert!(result.is_err(), "should fail without prior iter=0 call");
    }

    // -------------------------------------------------------------------------
    // compute_parameter_norm
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_parameter_norm_two_variables() {
        let mut vars: HashMap<String, VariableEnum> = HashMap::new();
        vars.insert(
            "a".into(),
            VariableEnum::Rn(Variable::new(Rn::new(dvector![3.0]))),
        );
        vars.insert(
            "b".into(),
            VariableEnum::Rn(Variable::new(Rn::new(dvector![4.0]))),
        );
        let norm = compute_parameter_norm(&vars);
        // sqrt(3² + 4²) = 5.0
        assert!((norm - 5.0).abs() < 1e-12, "expected 5.0, got {norm}");
    }

    #[test]
    fn test_compute_parameter_norm_empty() {
        let vars: HashMap<String, VariableEnum> = HashMap::new();
        assert_eq!(compute_parameter_norm(&vars), 0.0);
    }

    // -------------------------------------------------------------------------
    // apply_parameter_step / apply_negative_parameter_step
    // -------------------------------------------------------------------------

    #[test]
    fn test_apply_parameter_step_advances_variable() {
        let mut vars: HashMap<String, VariableEnum> = HashMap::new();
        vars.insert(
            "x".into(),
            VariableEnum::Rn(Variable::new(Rn::new(dvector![0.0]))),
        );
        let order = vec!["x".to_string()];
        let step = Mat::from_fn(1, 1, |_, _| 3.0);
        let norm = apply_parameter_step(&mut vars, step.as_ref(), &order);
        assert!((norm - 3.0).abs() < 1e-12);
        let val = vars["x"].to_vector()[0];
        assert!((val - 3.0).abs() < 1e-12, "expected 3.0, got {val}");
    }

    #[test]
    fn test_apply_negative_parameter_step_reverts() {
        let mut vars: HashMap<String, VariableEnum> = HashMap::new();
        vars.insert(
            "x".into(),
            VariableEnum::Rn(Variable::new(Rn::new(dvector![5.0]))),
        );
        let order = vec!["x".to_string()];
        let step = Mat::from_fn(1, 1, |_, _| 2.0);
        // apply +2 first
        apply_parameter_step(&mut vars, step.as_ref(), &order);
        assert!((vars["x"].to_vector()[0] - 7.0).abs() < 1e-12);
        // then revert with -2
        apply_negative_parameter_step(&mut vars, step.as_ref(), &order);
        assert!((vars["x"].to_vector()[0] - 5.0).abs() < 1e-12);
    }

    // -------------------------------------------------------------------------
    // check_convergence — all branches
    // -------------------------------------------------------------------------

    fn base_params() -> ConvergenceParams {
        ConvergenceParams {
            iteration: 1,
            current_cost: 1.0,
            new_cost: 0.9,
            parameter_norm: 1.0,
            parameter_update_norm: 1e-3,
            gradient_norm: 1e-3,
            elapsed: Duration::from_millis(10),
            step_accepted: true,
            max_iterations: 100,
            gradient_tolerance: 1e-10,
            parameter_tolerance: 1e-10,
            cost_tolerance: 1e-10,
            min_cost_threshold: None,
            timeout: None,
            trust_region_radius: None,
            min_trust_region_radius: None,
        }
    }

    #[test]
    fn test_check_convergence_no_trigger() {
        assert!(check_convergence(&base_params()).is_none());
    }

    #[test]
    fn test_check_convergence_nan_cost() {
        let mut p = base_params();
        p.new_cost = f64::NAN;
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::InvalidNumericalValues)
        );
    }

    #[test]
    fn test_check_convergence_inf_gradient() {
        let mut p = base_params();
        p.gradient_norm = f64::INFINITY;
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::InvalidNumericalValues)
        );
    }

    #[test]
    fn test_check_convergence_timeout() {
        let mut p = base_params();
        p.timeout = Some(Duration::from_millis(5));
        p.elapsed = Duration::from_millis(10);
        assert_eq!(check_convergence(&p), Some(OptimizationStatus::Timeout));
    }

    #[test]
    fn test_check_convergence_max_iterations() {
        let mut p = base_params();
        p.iteration = 100;
        p.max_iterations = 100;
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::MaxIterationsReached)
        );
    }

    #[test]
    fn test_check_convergence_gradient_tolerance() {
        let mut p = base_params();
        p.step_accepted = true;
        p.gradient_norm = 1e-12;
        p.gradient_tolerance = 1e-10;
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::GradientToleranceReached)
        );
    }

    #[test]
    fn test_check_convergence_parameter_tolerance() {
        let mut p = base_params();
        p.step_accepted = true;
        p.gradient_norm = 1.0; // above tolerance
        p.parameter_update_norm = 1e-20;
        p.parameter_tolerance = 1e-8;
        p.parameter_norm = 1.0;
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::ParameterToleranceReached)
        );
    }

    #[test]
    fn test_check_convergence_cost_tolerance() {
        let mut p = base_params();
        p.step_accepted = true;
        p.gradient_norm = 1.0; // above
        p.parameter_update_norm = 1.0; // above
        p.current_cost = 1.0;
        p.new_cost = 1.0 - 1e-15; // nearly no change
        p.cost_tolerance = 1e-10;
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::CostToleranceReached)
        );
    }

    #[test]
    fn test_check_convergence_min_cost_threshold() {
        let mut p = base_params();
        p.step_accepted = true;
        p.gradient_norm = 1.0;
        p.parameter_update_norm = 1.0;
        p.current_cost = 1.0;
        p.new_cost = 1.0 - 0.5; // big change — cost tol not triggered
        p.cost_tolerance = 1e-10;
        p.min_cost_threshold = Some(1.0); // new_cost=0.5 < 1.0 → trigger
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::MinCostThresholdReached)
        );
    }

    #[test]
    fn test_check_convergence_trust_region_too_small() {
        let mut p = base_params();
        p.step_accepted = true;
        p.gradient_norm = 1.0;
        p.parameter_update_norm = 1.0;
        p.current_cost = 1.0;
        p.new_cost = 0.5;
        p.cost_tolerance = 1e-10;
        p.trust_region_radius = Some(1e-40);
        p.min_trust_region_radius = Some(1e-32);
        assert_eq!(
            check_convergence(&p),
            Some(OptimizationStatus::TrustRegionRadiusTooSmall)
        );
    }

    #[test]
    fn test_check_convergence_step_not_accepted_skips_criteria() {
        // With step_accepted=false, gradient/parameter/cost tol should NOT fire
        let mut p = base_params();
        p.step_accepted = false;
        p.gradient_norm = 0.0; // would trigger gradient tol if accepted
        p.parameter_update_norm = 0.0;
        p.new_cost = 0.0;
        assert!(check_convergence(&p).is_none());
    }

    // -------------------------------------------------------------------------
    // create_linear_solver
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_linear_solver_cholesky() {
        let solver = create_linear_solver(&crate::linalg::LinearSolverType::SparseCholesky);
        // just verify it's constructable and callable without panic
        let _ = solver.get_hessian();
    }

    #[test]
    fn test_create_linear_solver_qr() {
        let solver = create_linear_solver(&crate::linalg::LinearSolverType::SparseQR);
        let _ = solver.get_hessian();
    }

    #[test]
    fn test_create_linear_solver_fallback_for_schur() {
        // SparseSchurComplement is special; falls back to Cholesky in create_linear_solver
        let solver = create_linear_solver(&crate::linalg::LinearSolverType::SparseSchurComplement);
        let _ = solver.get_hessian();
    }

    // -------------------------------------------------------------------------
    // Display impls
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimizer_type_display() {
        assert_eq!(
            format!("{}", OptimizerType::LevenbergMarquardt),
            "Levenberg-Marquardt"
        );
        assert_eq!(format!("{}", OptimizerType::GaussNewton), "Gauss-Newton");
        assert_eq!(format!("{}", OptimizerType::DogLeg), "Dog Leg");
    }

    #[test]
    fn test_optimization_status_display() {
        assert_eq!(format!("{}", OptimizationStatus::Converged), "Converged");
        assert_eq!(
            format!("{}", OptimizationStatus::MaxIterationsReached),
            "Maximum iterations reached"
        );
        assert_eq!(format!("{}", OptimizationStatus::Timeout), "Timeout");
        assert_eq!(
            format!("{}", OptimizationStatus::InvalidNumericalValues),
            "Invalid numerical values (NaN/Inf) detected"
        );
        assert!(format!("{}", OptimizationStatus::Failed("oops".into())).contains("oops"));
    }

    #[test]
    fn test_optimizer_error_variants() {
        let e1 = OptimizerError::TrustRegionFailure {
            radius: 1e-40,
            min_radius: 1e-32,
        };
        assert!(e1.to_string().contains("Trust region radius"));

        let e2 = OptimizerError::DampingFailure {
            damping: 1e13,
            max_damping: 1e12,
        };
        assert!(e2.to_string().contains("Damping parameter"));

        let e3 = OptimizerError::CostIncrease {
            old_cost: 1.0,
            new_cost: 2.0,
        };
        assert!(e3.to_string().contains("Cost increased"));

        let e4 = OptimizerError::LinearSolveFailed("singular".into());
        assert!(e4.to_string().contains("singular"));

        let e5 = OptimizerError::EmptyProblem;
        assert!(e5.to_string().contains("no variables"));

        let e6 = OptimizerError::NoResidualBlocks;
        assert!(e6.to_string().contains("residual blocks"));
    }

    // -------------------------------------------------------------------------
    // IterationStats print (smoke tests — no panic)
    // -------------------------------------------------------------------------

    #[test]
    fn test_iteration_stats_print_header_no_panic() {
        IterationStats::print_header();
    }

    #[test]
    fn test_iteration_stats_print_line_no_panic() {
        let stats = IterationStats {
            iteration: 1,
            cost: 1.5,
            cost_change: -0.5,
            gradient_norm: 1e-3,
            step_norm: 1e-4,
            tr_ratio: 0.8,
            tr_radius: 1e3,
            ls_iter: 0,
            iter_time_ms: 2.5,
            total_time_ms: 10.0,
            accepted: true,
        };
        stats.print_line();
    }

    // -------------------------------------------------------------------------
    // build_solver_result
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_solver_result_fields() -> TestResult {
        let mut variables: HashMap<String, VariableEnum> = HashMap::new();
        variables.insert(
            "x".into(),
            VariableEnum::Rn(Variable::new(Rn::new(dvector![3.0]))),
        );
        let state = InitializedState {
            variables,
            variable_index_map: HashMap::new(),
            sorted_vars: vec!["x".to_string()],
            symbolic_structure: None,
            total_dof: 1,
            current_cost: 0.1,
            initial_cost: 5.0,
        };
        let result = build_solver_result(
            OptimizationStatus::CostToleranceReached,
            10,
            state,
            Duration::from_millis(50),
            1e-5,
            1e-6,
            15,
            10,
            None,
        );
        assert_eq!(result.status, OptimizationStatus::CostToleranceReached);
        assert_eq!(result.iterations, 10);
        assert!((result.initial_cost - 5.0).abs() < 1e-12);
        assert!((result.final_cost - 0.1).abs() < 1e-12);
        let ci = result
            .convergence_info
            .as_ref()
            .ok_or("convergence_info is None")?;
        assert_eq!(ci.cost_evaluations, 15);
        assert_eq!(ci.jacobian_evaluations, 10);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // create_optimizer_summary
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_optimizer_summary_averages() {
        let summary = create_optimizer_summary(
            "TestOptimizer",
            10.0,
            1.0,
            4,
            Some(3),
            Some(1),
            2.0,
            0.1,
            3.0,
            0.05,
            9.0, // total_cost_reduction
            Duration::from_millis(400),
            vec![],
            OptimizationStatus::CostToleranceReached,
            Some(1e-3),
            None,
            Some(0.8),
        );
        // average_cost_reduction = 9.0 / 4 = 2.25
        assert!((summary.average_cost_reduction - 2.25).abs() < 1e-10);
        // average_time_per_iteration = 400ms / 4 = 100ms
        assert_eq!(
            summary.average_time_per_iteration,
            Duration::from_millis(100)
        );
        assert_eq!(summary.optimizer_name, "TestOptimizer");
        assert_eq!(summary.successful_steps, Some(3));
    }

    #[test]
    fn test_create_optimizer_summary_zero_iterations() {
        let summary = create_optimizer_summary(
            "Test",
            1.0,
            1.0,
            0, // zero iterations
            None,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            Duration::from_secs(0),
            vec![],
            OptimizationStatus::MaxIterationsReached,
            None,
            None,
            None,
        );
        assert_eq!(summary.average_cost_reduction, 0.0);
        assert_eq!(summary.average_time_per_iteration, Duration::from_secs(0));
    }

    // -------------------------------------------------------------------------
    // Simple Factor for integration tests in mod.rs
    // -------------------------------------------------------------------------

    /// Linear factor: r = x - target, J = [[1.0]]
    struct LinearFactor {
        target: f64,
    }

    impl Factor for LinearFactor {
        fn linearize(
            &self,
            params: &[DVector<f64>],
            compute_jacobian: bool,
        ) -> (DVector<f64>, Option<DMatrix<f64>>) {
            let residual = dvector![params[0][0] - self.target];
            let jacobian = if compute_jacobian {
                Some(DMatrix::from_element(1, 1, 1.0))
            } else {
                None
            };
            (residual, jacobian)
        }

        fn get_dimension(&self) -> usize {
            1
        }
    }

    // -------------------------------------------------------------------------
    // initialize_optimization_state (smoke test)
    // -------------------------------------------------------------------------

    #[test]
    fn test_initialize_optimization_state() -> TestResult {
        use crate::core::problem::Problem;

        let mut problem = Problem::new(JacobianMode::Sparse);
        let mut initial_values: HashMap<String, (ManifoldType, DVector<f64>)> = HashMap::new();
        initial_values.insert("x".to_string(), (ManifoldType::RN, dvector![5.0]));
        problem.add_residual_block(&["x"], Box::new(LinearFactor { target: 0.0 }), None);

        let state = initialize_optimization_state(&problem, &initial_values)?;
        assert_eq!(state.total_dof, 1);
        assert!(state.initial_cost > 0.0);
        assert!(state.sorted_vars.contains(&"x".to_string()));
        Ok(())
    }
}
