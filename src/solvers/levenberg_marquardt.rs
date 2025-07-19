//! Levenberg-Marquardt algorithm implementation.
//!
//! The Levenberg-Marquardt algorithm is a popular optimization method for
//! nonlinear least squares problems. It interpolates between the Gauss-Newton
//! algorithm and gradient descent.

use crate::core::OptimizationStatus;
use crate::solvers::{ConvergenceInfo, Solver, SolverConfig, SolverResult};
use std::time::Instant;

/// Levenberg-Marquardt solver for nonlinear least squares optimization.
pub struct LevenbergMarquardt {
    config: SolverConfig,
    damping: f64,
    damping_min: f64,
    damping_max: f64,
    damping_increase_factor: f64,
    damping_decrease_factor: f64,
}

impl LevenbergMarquardt {
    /// Create a new Levenberg-Marquardt solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(SolverConfig::default())
    }

    /// Create a new Levenberg-Marquardt solver with the given configuration.
    pub fn with_config(config: SolverConfig) -> Self {
        Self {
            config,
            damping: 1e-3,
            damping_min: 1e-12,
            damping_max: 1e12,
            damping_increase_factor: 10.0,
            damping_decrease_factor: 0.1,
        }
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
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> Solver<P> for LevenbergMarquardt
where
    P: Clone,
{
    type Config = SolverConfig;
    type Error = crate::core::ApexError;

    fn new(config: Self::Config) -> Self {
        Self::with_config(config)
    }

    fn solve<T>(&mut self, problem: &T, initial_params: P) -> Result<SolverResult<P>, Self::Error>
    where
        T: crate::core::Optimizable<Parameters = P>,
    {
        let start_time = Instant::now();
        let params = initial_params;
        let mut iteration = 0;
        let cost_evaluations = 0;
        let jacobian_evaluations = 0;

        // Initial cost evaluation
        let _residual = problem.evaluate(&params)?;

        let current_cost = 0.0; // TODO: Compute actual cost from residual
        let _damping = self.damping;

        loop {
            // Check termination criteria
            let elapsed = start_time.elapsed();
            if iteration >= self.config.max_iterations {
                return Ok(SolverResult {
                    parameters: params,
                    status: OptimizationStatus::MaxIterationsReached,
                    final_cost: current_cost,
                    iterations: iteration,
                    elapsed_time: elapsed,
                    convergence_info: ConvergenceInfo {
                        final_gradient_norm: 0.0, // TODO: compute actual gradient norm
                        final_parameter_update_norm: 0.0,
                        cost_evaluations,
                        jacobian_evaluations,
                    },
                });
            }

            if let Some(timeout) = self.config.timeout {
                if elapsed >= timeout {
                    return Ok(SolverResult {
                        parameters: params,
                        status: OptimizationStatus::Timeout,
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

            // Check cost tolerance
            if current_cost < self.config.cost_tolerance {
                return Ok(SolverResult {
                    parameters: params,
                    status: OptimizationStatus::Converged,
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

            // TODO: Implement actual Levenberg-Marquardt step
            // For now, this is a placeholder that prevents infinite loops
            iteration += 1;

            // Placeholder convergence
            if iteration > 10 {
                return Ok(SolverResult {
                    parameters: params,
                    status: OptimizationStatus::Converged,
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenberg_marquardt_creation() {
        let solver = LevenbergMarquardt::new();
        assert!(solver.damping > 0.0);
    }

    #[test]
    fn test_damping_configuration() {
        let solver = LevenbergMarquardt::new()
            .with_damping(1e-6)
            .with_damping_bounds(1e-15, 1e15);

        assert_eq!(solver.damping, 1e-6);
        assert_eq!(solver.damping_min, 1e-15);
        assert_eq!(solver.damping_max, 1e15);
    }
}
