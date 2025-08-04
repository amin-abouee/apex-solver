//! Demonstration of the apex-solver optimization system
//!
//! This example shows how to use the different solver types and configurations
//! available in the apex-solver library.

use apex_solver::{
    core::{ApexError, Optimizable},
    linalg::LinearSolverType,
    optimizer::{AnySolver, OptimizerConfig, OptimizerType},
};
use std::time::Duration;

/// A simple quadratic optimization problem: minimize f(x) = (x - target)^2
struct QuadraticProblem {
    target: f64,
}

impl QuadraticProblem {
    fn new(target: f64) -> Self {
        Self { target }
    }
}

impl Optimizable for QuadraticProblem {
    type Parameters = f64;
    type Residuals = f64;
    type Jacobian = f64;

    fn weights(&self) -> faer::Mat<f64> {
        faer::Mat::from_fn(1, 1, |_, _| 1.0)
    }

    fn evaluate(&self, parameters: &Self::Parameters) -> Result<Self::Residuals, ApexError> {
        Ok(parameters - self.target)
    }

    fn evaluate_with_jacobian(
        &self,
        parameters: &Self::Parameters,
    ) -> Result<(Self::Residuals, Self::Jacobian), ApexError> {
        Ok((parameters - self.target, 1.0))
    }

    fn parameter_count(&self) -> usize {
        1
    }

    fn residual_count(&self) -> usize {
        1
    }

    fn cost(&self, parameters: &Self::Parameters) -> Result<f64, ApexError> {
        let residual = self.evaluate(parameters)?;
        Ok(0.5 * residual * residual)
    }
}

/// Demonstrate Gauss-Newton solver
fn demo_gauss_newton() -> Result<(), ApexError> {
    println!("=== Gauss-Newton Solver Demo ===");

    let problem = QuadraticProblem::new(5.0);
    let initial_params = 0.0;

    let config = OptimizerConfig::new()
        .with_optimizer_type(OptimizerType::GaussNewton)
        .with_linear_solver_type(LinearSolverType::SparseQR)
        .with_max_iterations(50)
        .with_verbose(true);

    let mut solver = AnySolver::new(config);
    let result = solver.solve(&problem, initial_params)?;

    println!("Result: {:?}", result.status);
    println!("Final parameters: {}", result.parameters);
    println!("Final cost: {:.6e}", result.final_cost);
    println!("Iterations: {}", result.iterations);
    println!("Elapsed time: {:?}", result.elapsed_time);
    println!();

    Ok(())
}

/// Demonstrate Levenberg-Marquardt solver
fn demo_levenberg_marquardt() -> Result<(), ApexError> {
    println!("=== Levenberg-Marquardt Solver Demo ===");

    let problem = QuadraticProblem::new(3.14);
    let initial_params = 10.0;

    let config = OptimizerConfig::new()
        .with_optimizer_type(OptimizerType::LevenbergMarquardt)
        .with_linear_solver_type(LinearSolverType::SparseCholesky)
        .with_max_iterations(100)
        .with_cost_tolerance(1e-10)
        .with_verbose(true);

    let mut solver = AnySolver::new(config);
    let result = solver.solve(&problem, initial_params)?;

    println!("Result: {:?}", result.status);
    println!("Final parameters: {}", result.parameters);
    println!("Final cost: {:.6e}", result.final_cost);
    println!("Iterations: {}", result.iterations);
    println!("Elapsed time: {:?}", result.elapsed_time);
    println!();

    Ok(())
}

/// Demonstrate Dog Leg solver
fn demo_dog_leg() -> Result<(), ApexError> {
    println!("=== Dog Leg Solver Demo ===");

    let problem = QuadraticProblem::new(-2.5);
    let initial_params = 8.0;

    let config = OptimizerConfig::new()
        .with_optimizer_type(OptimizerType::DogLeg)
        .with_linear_solver_type(LinearSolverType::SparseQR)
        .with_max_iterations(75)
        .with_parameter_tolerance(1e-12)
        .with_timeout(Duration::from_secs(10))
        .with_verbose(true);

    let mut solver = AnySolver::new(config);
    let result = solver.solve(&problem, initial_params)?;

    println!("Result: {:?}", result.status);
    println!("Final parameters: {}", result.parameters);
    println!("Final cost: {:.6e}", result.final_cost);
    println!("Iterations: {}", result.iterations);
    println!("Elapsed time: {:?}", result.elapsed_time);
    println!();

    Ok(())
}

/// Compare different solver configurations
fn compare_solvers() -> Result<(), ApexError> {
    println!("=== Solver Comparison ===");

    let problem = QuadraticProblem::new(1.0);
    let initial_params = 100.0;

    let optimizer_types = [
        OptimizerType::GaussNewton,
        OptimizerType::LevenbergMarquardt,
        OptimizerType::DogLeg,
    ];

    let linear_solver_types = [LinearSolverType::SparseCholesky, LinearSolverType::SparseQR];

    for &optimizer_type in &optimizer_types {
        for &linear_solver_type in &linear_solver_types {
            let config = OptimizerConfig::new()
                .with_optimizer_type(optimizer_type)
                .with_linear_solver_type(linear_solver_type)
                .with_max_iterations(20);

            let mut solver = AnySolver::new(config);
            let result = solver.solve(&problem, initial_params)?;

            println!(
                "{:?} + {:?}: {} iterations, {:.6e} cost, {:?}",
                optimizer_type,
                linear_solver_type,
                result.iterations,
                result.final_cost,
                result.status
            );
        }
    }

    Ok(())
}

fn main() -> Result<(), ApexError> {
    println!("Apex Solver Demonstration\n");

    demo_gauss_newton()?;
    demo_levenberg_marquardt()?;
    demo_dog_leg()?;
    compare_solvers()?;

    println!("All demos completed successfully!");

    Ok(())
}
