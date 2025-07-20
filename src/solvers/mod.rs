//! Optimization solvers for nonlinear least squares problems.
//!
//! This module provides various optimization algorithms specifically designed
//! for nonlinear least squares problems commonly found in computer vision:
//! - Levenberg-Marquardt algorithm
//! - Gauss-Newton algorithm
//! - Dog Leg algorithm
//! - Trust region methods

use crate::core::OptimizationStatus;
use crate::linalg::LinearSolverType;
use std::time::Duration;

/// Type of optimization solver algorithm to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverType {
    /// Gauss-Newton algorithm (fast convergence, may be unstable)
    GaussNewton,
    /// Levenberg-Marquardt algorithm (robust, adaptive damping)
    LevenbergMarquardt,
    /// Dog Leg algorithm (trust region method)
    DogLeg,
}

impl Default for SolverType {
    fn default() -> Self {
        SolverType::LevenbergMarquardt
    }
}

/// Configuration parameters for optimization solvers.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Type of solver algorithm to use
    pub solver_type: SolverType,
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
    /// Enable detailed logging
    pub verbose: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            solver_type: SolverType::default(),
            linear_solver_type: LinearSolverType::default(),
            max_iterations: 100,
            cost_tolerance: 1e-8,
            parameter_tolerance: 1e-8,
            gradient_tolerance: 1e-8,
            timeout: None,
            verbose: false,
        }
    }
}

impl SolverConfig {
    /// Create a new solver configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the solver algorithm type
    pub fn with_solver_type(mut self, solver_type: SolverType) -> Self {
        self.solver_type = solver_type;
        self
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

    /// Enable or disable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// State information during iterative optimization.
#[derive(Debug, Clone)]
pub struct IterativeState {
    /// Current iteration number
    pub iteration: usize,
    /// Current cost value
    pub cost: f64,
    /// Current gradient norm
    pub gradient_norm: f64,
    /// Current parameter update norm
    pub parameter_update_norm: f64,
    /// Time elapsed since start
    pub elapsed_time: Duration,
}

/// Result of a solver execution.
#[derive(Debug, Clone)]
pub struct SolverResult<T> {
    /// Final parameters
    pub parameters: T,
    /// Final optimization status
    pub status: OptimizationStatus,
    /// Final cost value
    pub final_cost: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Total time elapsed
    pub elapsed_time: Duration,
    /// Convergence statistics
    pub convergence_info: ConvergenceInfo,
}

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

/// Core trait for optimization solvers.
pub trait Solver<P> {
    /// Configuration type for this solver
    type Config;
    /// Error type
    type Error;

    /// Create a new solver with the given configuration
    fn new(config: Self::Config) -> Self;

    /// Solve the optimization problem
    fn solve<T>(&mut self, problem: &T, initial_params: P) -> Result<SolverResult<P>, Self::Error>
    where
        T: crate::core::Optimizable<Parameters = P>;
}

/// Enum wrapper for different solver types to enable dynamic dispatch
pub enum AnySolver {
    GaussNewton(GaussNewton),
    LevenbergMarquardt(LevenbergMarquardt),
    DogLeg(DogLeg),
}

impl AnySolver {
    /// Create a new solver based on the configuration
    pub fn new(config: SolverConfig) -> Self {
        match config.solver_type {
            SolverType::GaussNewton => AnySolver::GaussNewton(GaussNewton::with_config(config)),
            SolverType::LevenbergMarquardt => AnySolver::LevenbergMarquardt(LevenbergMarquardt::with_config(config)),
            SolverType::DogLeg => AnySolver::DogLeg(DogLeg::with_config(config)),
        }
    }

    /// Solve the optimization problem
    pub fn solve<T, P>(&mut self, problem: &T, initial_params: P) -> Result<SolverResult<P>, crate::core::ApexError>
    where
        T: crate::core::Optimizable<Parameters = P>,
        P: Clone,
    {
        match self {
            AnySolver::GaussNewton(solver) => solver.solve(problem, initial_params),
            AnySolver::LevenbergMarquardt(solver) => solver.solve(problem, initial_params),
            AnySolver::DogLeg(solver) => solver.solve(problem, initial_params),
        }
    }
}

// Submodules for specific solver implementations
pub mod gauss_newton;
pub mod levenberg_marquardt;
pub mod dog_leg;
pub mod trust_region;

#[cfg(test)]
mod tests;

pub use gauss_newton::GaussNewton;
pub use levenberg_marquardt::LevenbergMarquardt;
pub use dog_leg::DogLeg;
pub use trust_region::TrustRegion;

/// Factory for creating solvers based on configuration
pub struct SolverFactory;

impl SolverFactory {
    /// Create a solver based on the configuration
    pub fn create_solver(config: SolverConfig) -> AnySolver {
        AnySolver::new(config)
    }
}
