//! Optimization solvers for nonlinear least squares problems.
//!
//! This module provides various optimization algorithms specifically designed
//! for nonlinear least squares problems commonly found in computer vision:
//! - Levenberg-Marquardt algorithm
//! - Gauss-Newton algorithm
//! - Trust region methods
//! - Gradient descent variants

use crate::core::OptimizationStatus;
use std::time::Duration;

/// Configuration parameters for optimization solvers.
#[derive(Debug, Clone)]
pub struct SolverConfig {
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
            max_iterations: 100,
            cost_tolerance: 1e-8,
            parameter_tolerance: 1e-8,
            gradient_tolerance: 1e-8,
            timeout: None,
            verbose: false,
        }
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

// Submodules for specific solver implementations
pub mod gauss_newton;
pub mod levenberg_marquardt;
pub mod trust_region;

pub use gauss_newton::GaussNewton;
pub use levenberg_marquardt::LevenbergMarquardt;
pub use trust_region::TrustRegion;
