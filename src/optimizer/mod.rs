//! Optimization solvers for nonlinear least squares problems.
//!
//! This module provides various optimization algorithms specifically designed
//! for nonlinear least squares problems commonly found in computer vision:
//! - Levenberg-Marquardt algorithm
//! - Gauss-Newton algorithm
//! - Dog Leg algorithm

use crate::core::problem::{Problem, VariableEnum};
use crate::manifold::ManifoldType;
use nalgebra::DVector;
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

pub mod dog_leg;
pub mod gauss_newton;
pub mod levenberg_marquardt;

pub use dog_leg::DogLeg;
pub use gauss_newton::GaussNewton;
pub use levenberg_marquardt::LevenbergMarquardt;

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

// #[derive(Clone)]
// pub struct OptimizerConfig {
//     /// Type of optimizer algorithm to use
//     pub optimizer_type: OptimizerType,
//     /// Type of linear solver for the linear systems
//     pub linear_solver_type: LinearSolverType,
//     /// Maximum number of iterations
//     pub max_iterations: usize,
//     /// Convergence tolerance for cost function
//     pub cost_tolerance: f64,
//     /// Convergence tolerance for parameter updates
//     pub parameter_tolerance: f64,
//     /// Convergence tolerance for gradient norm
//     pub gradient_tolerance: f64,
//     /// Timeout duration
//     pub timeout: Option<Duration>,
//     /// Enable detailed logging
//     pub verbose: bool,
// }

// impl Default for OptimizerConfig {
//     fn default() -> Self {
//         Self {
//             optimizer_type: OptimizerType::default(),
//             linear_solver_type: LinearSolverType::default(),
//             max_iterations: 100,
//             cost_tolerance: 1e-8,
//             parameter_tolerance: 1e-8,
//             gradient_tolerance: 1e-8,
//             timeout: None,
//             verbose: false,
//         }
//     }
// }

// impl OptimizerConfig {
//     /// Create a new solver configuration with default values
//     pub fn new() -> Self {
//         Self::default()
//     }

//     /// Set the optimizer algorithm type
//     pub fn with_optimizer_type(mut self, optimizer_type: OptimizerType) -> Self {
//         self.optimizer_type = optimizer_type;
//         self
//     }

//     /// Set the linear solver type
//     pub fn with_linear_solver_type(mut self, linear_solver_type: LinearSolverType) -> Self {
//         self.linear_solver_type = linear_solver_type;
//         self
//     }

//     /// Set the maximum number of iterations
//     pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
//         self.max_iterations = max_iterations;
//         self
//     }

//     /// Set the cost tolerance
//     pub fn with_cost_tolerance(mut self, cost_tolerance: f64) -> Self {
//         self.cost_tolerance = cost_tolerance;
//         self
//     }

//     /// Set the parameter tolerance
//     pub fn with_parameter_tolerance(mut self, parameter_tolerance: f64) -> Self {
//         self.parameter_tolerance = parameter_tolerance;
//         self
//     }

//     /// Set the gradient tolerance
//     pub fn with_gradient_tolerance(mut self, gradient_tolerance: f64) -> Self {
//         self.gradient_tolerance = gradient_tolerance;
//         self
//     }

//     /// Set the timeout duration
//     pub fn with_timeout(mut self, timeout: Duration) -> Self {
//         self.timeout = Some(timeout);
//         self
//     }

//     /// Enable or disable verbose logging
//     pub fn with_verbose(mut self, verbose: bool) -> Self {
//         self.verbose = verbose;
//         self
//     }
// }

// impl fmt::Display for OptimizerConfig {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(
//             f,
//             "OptimizerConfig {{ optimizer_type: {:?}, linear_solver_type: {:?}, max_iterations: {}, cost_tolerance: {}, parameter_tolerance: {}, gradient_tolerance: {}, timeout: {:?}, verbose: {} }}",
//             self.optimizer_type,
//             self.linear_solver_type,
//             self.max_iterations,
//             self.cost_tolerance,
//             self.parameter_tolerance,
//             self.gradient_tolerance,
//             self.timeout,
//             self.verbose
//         )
//     }
// }

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
    pub init_cost: f64,
    /// Final cost value
    pub final_cost: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Total time elapsed
    pub elapsed_time: Duration,
    /// Convergence statistics
    pub convergence_info: Option<ConvergenceInfo>,
}

/// Core trait for optimization solvers.
pub trait Solver {
    /// Configuration type for this solver
    type Config;
    /// Error type
    type Error;

    /// Create a new solver with the given configuration
    fn new() -> Self;

    /// Minimize the optimization problem
    fn minimize(
        &mut self,
        problem: &Problem,
        initial_params: &HashMap<String, (ManifoldType, DVector<f64>)>,
    ) -> Result<SolverResult<HashMap<String, VariableEnum>>, Self::Error>;
}

// Legacy optimization function (commented out - use Solver trait instead)
// pub fn solve_problem(
//     problem: &Problem,
//     initial_params: &HashMap<String, (ManifoldType, na::DVector<f64>)>,
//     config: OptimizerConfig,
// ) -> Result<SolverResult<HashMap<String, VariableEnum>>, crate::core::ApexError> {
//     // Use the actual solver implementations based on config via Solver trait
//     match config.optimizer_type {
//         OptimizerType::LevenbergMarquardt => {
//             let mut solver = LevenbergMarquardt::with_config(config);
//             solver.solve_problem(problem, initial_params)
//         }
//         OptimizerType::GaussNewton => {
//             let _solver = GaussNewton::new();
//             // TODO: Implement solve_problem for GaussNewton
//             Err(crate::core::ApexError::Solver(
//                 "GaussNewton not implemented yet".to_string(),
//             ))
//         }
//         OptimizerType::DogLeg => {
//             let _solver = DogLeg::new();
//             // TODO: Implement solve_problem for DogLeg
//             Err(crate::core::ApexError::Solver(
//                 "DogLeg not implemented yet".to_string(),
//             ))
//         }
//     }
// }
