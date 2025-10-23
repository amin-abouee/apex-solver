//! Optimization solvers for nonlinear least squares problems.
//!
//! This module provides various optimization algorithms specifically designed
//! for nonlinear least squares problems commonly found in computer vision:
//! - Levenberg-Marquardt algorithm
//! - Gauss-Newton algorithm
//! - Dog Leg algorithm

use crate::core::problem;
use crate::linalg;
use crate::manifold;
use nalgebra;
use std::collections;
use std::fmt;
use std::time;
use thiserror;

pub mod dog_leg;
pub mod gauss_newton;
pub mod levenberg_marquardt;
pub mod visualization;

pub use dog_leg::DogLeg;
pub use gauss_newton::GaussNewton;
pub use levenberg_marquardt::LevenbergMarquardt;
pub use visualization::OptimizationVisualizer;

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

impl fmt::Display for OptimizerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerType::LevenbergMarquardt => write!(f, "Levenberg-Marquardt"),
            OptimizerType::GaussNewton => write!(f, "Gauss-Newton"),
            OptimizerType::DogLeg => write!(f, "Dog Leg"),
        }
    }
}

/// Optimizer-specific error types for apex-solver
#[derive(Debug, Clone, thiserror::Error)]
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

impl fmt::Display for ConvergenceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    /// Other failure
    Failed(String),
}

impl fmt::Display for OptimizationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    pub covariances: Option<std::collections::HashMap<String, faer::Mat<f64>>>,
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
        problem: &problem::Problem,
        initial_params: &collections::HashMap<
            String,
            (manifold::ManifoldType, nalgebra::DVector<f64>),
        >,
    ) -> Result<SolverResult<collections::HashMap<String, problem::VariableEnum>>, Self::Error>;
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
    variables: &mut collections::HashMap<String, problem::VariableEnum>,
    step: faer::MatRef<f64>,
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
    variables: &mut collections::HashMap<String, problem::VariableEnum>,
    step: faer::MatRef<f64>,
    variable_order: &[String],
) {
    use faer::Mat;

    // Create a negated version of the step vector
    let mut negative_step = Mat::zeros(step.nrows(), 1);
    for i in 0..step.nrows() {
        negative_step[(i, 0)] = -step[(i, 0)];
    }

    // Apply the negative step using the standard apply_parameter_step function
    apply_parameter_step(variables, negative_step.as_ref(), variable_order);
}

pub fn compute_cost(residual: &faer::Mat<f64>) -> f64 {
    let cost = residual.norm_l2();
    0.5 * cost * cost
}
