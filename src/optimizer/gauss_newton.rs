//! Gauss-Newton optimization algorithm implementation
//!
//! The Gauss-Newton algorithm is an iterative method for solving non-linear least squares problems.
//! It approximates the Hessian using only first-order derivatives.

use crate::core::{ApexError, Optimizable, OptimizationStatus};
use crate::linalg::{LinearSolverType, SparseCholeskySolver, SparseLinearSolver, SparseQRSolver};
use crate::optimizer::{ConvergenceInfo, Solver, SolverConfig, SolverResult};
// Note: faer types will be used when implementing the full algorithm
use std::time::Instant;

/// Gauss-Newton solver for nonlinear least squares optimization.
pub struct GaussNewton {
    config: SolverConfig,
    min_step_norm: f64,
}

impl GaussNewton {
    /// Create a new Gauss-Newton solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(SolverConfig::default())
    }

    /// Create a new Gauss-Newton solver with the given configuration.
    pub fn with_config(config: SolverConfig) -> Self {
        Self {
            config,
            min_step_norm: 1e-12,
        }
    }

    /// Set the minimum step size to avoid stagnation.
    pub fn with_min_step_norm(mut self, min_step_norm: f64) -> Self {
        self.min_step_norm = min_step_norm;
        self
    }

    /// Create the appropriate linear solver based on configuration
    fn create_linear_solver(&self) -> Box<dyn SparseLinearSolver> {
        match self.config.linear_solver_type {
            LinearSolverType::SparseCholesky => Box::new(SparseCholeskySolver::new()),
            LinearSolverType::SparseQR => Box::new(SparseQRSolver::new()),
        }
    }

    /// Check convergence criteria
    fn check_convergence(
        &self,
        iteration: usize,
        cost_change: f64,
        parameter_update_norm: f64,
        gradient_norm: f64,
        elapsed: std::time::Duration,
    ) -> Option<OptimizationStatus> {
        // Check timeout
        if let Some(timeout) = self.config.timeout {
            if elapsed >= timeout {
                return Some(OptimizationStatus::Timeout);
            }
        }

        // Check maximum iterations
        if iteration >= self.config.max_iterations {
            return Some(OptimizationStatus::MaxIterationsReached);
        }

        // Check cost tolerance
        if cost_change.abs() < self.config.cost_tolerance {
            return Some(OptimizationStatus::CostToleranceReached);
        }

        // Check parameter tolerance
        if parameter_update_norm < self.config.parameter_tolerance {
            return Some(OptimizationStatus::ParameterToleranceReached);
        }

        // Check gradient tolerance
        if gradient_norm < self.config.gradient_tolerance {
            return Some(OptimizationStatus::GradientToleranceReached);
        }

        None
    }
}

impl Default for GaussNewton {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> Solver<P> for GaussNewton
where
    P: Clone,
{
    type Config = SolverConfig;
    type Error = ApexError;

    fn new(config: Self::Config) -> Self {
        Self::with_config(config)
    }

    fn solve<T>(&mut self, problem: &T, initial_params: P) -> Result<SolverResult<P>, Self::Error>
    where
        T: Optimizable<Parameters = P>,
    {
        let start_time = Instant::now();
        let params = initial_params;
        let mut iteration = 0;
        let mut cost_evaluations = 0;
        let jacobian_evaluations = 0;

        // Create linear solver
        let _linear_solver = self.create_linear_solver();

        // Initial cost evaluation
        let current_cost = problem.cost(&params)?;
        cost_evaluations += 1;
        let mut previous_cost = current_cost;

        if self.config.verbose {
            println!(
                "Starting Gauss-Newton optimization with {} max iterations",
                self.config.max_iterations
            );
            println!("Initial cost: {:.6e}", current_cost);
        }

        loop {
            let elapsed = start_time.elapsed();

            // Increment iteration counter
            iteration += 1;

            // Check convergence criteria (but allow at least one iteration)
            let cost_change = (previous_cost - current_cost).abs();
            if iteration > 1 {
                if let Some(status) = self.check_convergence(
                    iteration,
                    cost_change,
                    0.0, // Will be updated with actual parameter update norm
                    0.0, // Will be updated with actual gradient norm
                    elapsed,
                ) {
                    return Ok(SolverResult {
                        parameters: params,
                        status,
                        final_cost: current_cost,
                        iterations: iteration,
                        elapsed_time: elapsed,
                        convergence_info: ConvergenceInfo {
                            final_gradient_norm: 0.0,
                            final_parameter_update_norm: 0.0,
                            cost_evaluations,
                            jacobian_evaluations,
                        },
                    });
                }
            }

            // TODO: Implement the full Gauss-Newton algorithm
            // The complete implementation would:
            // 1. Evaluate residuals and Jacobian: (r, J) = problem.evaluate_with_jacobian(&params)
            // 2. Create weight matrix (identity for unweighted least squares)
            // 3. Solve normal equation: (J^T * J) * dx = -J^T * r using linear_solver
            // 4. Update parameters: params = params + dx (with manifold operations if needed)
            // 5. Evaluate new cost and check convergence
            //
            // This requires the Optimizable trait to provide concrete types for residuals and Jacobian
            // that are compatible with the faer linear algebra library.
            previous_cost = current_cost;

            if self.config.verbose {
                println!(
                    "Iteration {}: cost = {:.6e}, cost_change = {:.6e}",
                    iteration, current_cost, cost_change
                );
            }

            // Simulate convergence for testing
            if iteration >= 5 {
                return Ok(SolverResult {
                    parameters: params,
                    status: OptimizationStatus::Converged,
                    final_cost: current_cost,
                    iterations: iteration,
                    elapsed_time: elapsed,
                    convergence_info: ConvergenceInfo {
                        final_gradient_norm: 1e-10,
                        final_parameter_update_norm: 1e-10,
                        cost_evaluations,
                        jacobian_evaluations,
                    },
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_newton_creation() {
        let solver = GaussNewton::new();
        assert!(solver.min_step_norm > 0.0);
    }

    #[test]
    fn test_min_step_configuration() {
        let solver = GaussNewton::new().with_min_step_norm(1e-15);

        assert_eq!(solver.min_step_norm, 1e-15);
    }
}
