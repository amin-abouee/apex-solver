//! M3500 G2O optimization example
//!
//! This example demonstrates how to load a G2O file and optimize it using the apex-solver
//! optimization framework. It loads the M3500.g2o dataset and performs pose graph optimization.

use apex_solver::{
    core::Optimizable,
    g2o_optimizer::G2oProblem,
    io::load_graph,
    linalg::LinearSolverType,
    solvers::{SolverConfig, SolverType},
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();

    println!("=== M3500 G2O Optimization Example ===\n");

    // Load the M3500.g2o file
    println!("Loading M3500.g2o dataset...");
    let start_time = Instant::now();
    let graph = load_graph("data/M3500.g2o")?;
    let load_time = start_time.elapsed();
    println!("✅ Loaded in {:?}", load_time);

    // Create optimization problem
    println!("\nCreating optimization problem...");
    let start_time = Instant::now();
    let problem = G2oProblem::from_graph(graph)?;
    let setup_time = start_time.elapsed();
    println!("✅ Problem created in {:?}", setup_time);

    // Display problem statistics
    println!("\n{}", problem.statistics());

    // Configure solver
    let config = SolverConfig::new()
        .with_solver_type(SolverType::LevenbergMarquardt)
        .with_linear_solver_type(LinearSolverType::SparseCholesky)
        .with_max_iterations(50)
        .with_cost_tolerance(1e-6)
        .with_parameter_tolerance(1e-6)
        .with_verbose(true);

    println!("Solver Configuration:");
    println!("  Algorithm: {:?}", config.solver_type);
    println!("  Linear Solver: {:?}", config.linear_solver_type);
    println!("  Max Iterations: {}", config.max_iterations);
    println!("  Cost Tolerance: {:.2e}", config.cost_tolerance);
    println!("  Parameter Tolerance: {:.2e}", config.parameter_tolerance);

    // Compute initial cost
    println!("\nComputing initial cost...");
    let initial_cost = problem.cost(&problem.initial_parameters)?;
    println!("Initial cost: {:.6e}", initial_cost);

    // Solve the optimization problem
    println!("\n🚀 Starting optimization...");
    let start_time = Instant::now();
    let result = problem.solve(config)?;
    let solve_time = start_time.elapsed();

    // Display results
    println!("\n=== Optimization Results ===");
    println!("Status: {:?}", result.status);
    println!("Final cost: {:.6e}", result.final_cost);
    println!("Initial cost: {:.6e}", initial_cost);
    println!(
        "Cost reduction: {:.6e} ({:.2}%)",
        initial_cost - result.final_cost,
        (initial_cost - result.final_cost) / initial_cost * 100.0
    );
    println!("Iterations: {}", result.iterations);
    println!("Solve time: {:?}", solve_time);
    println!("Total time: {:?}", load_time + setup_time + solve_time);

    // Display convergence information
    println!("\n=== Convergence Information ===");
    println!(
        "Final gradient norm: {:.6e}",
        result.convergence_info.final_gradient_norm
    );
    println!(
        "Final parameter update norm: {:.6e}",
        result.convergence_info.final_parameter_update_norm
    );
    println!(
        "Cost evaluations: {}",
        result.convergence_info.cost_evaluations
    );
    println!(
        "Jacobian evaluations: {}",
        result.convergence_info.jacobian_evaluations
    );

    // Analyze parameter changes
    println!("\n=== Parameter Analysis ===");
    let mut total_parameter_change = 0.0f64;
    let mut max_parameter_change = 0.0f64;
    let mut changed_variables = 0;

    for (var_name, final_params) in &result.parameters {
        if let Some(initial_params) = problem.initial_parameters.get(var_name) {
            let diff = final_params - initial_params;
            let change_norm = diff.norm();
            total_parameter_change += change_norm;
            max_parameter_change = max_parameter_change.max(change_norm);
            if change_norm > 1e-6 {
                changed_variables += 1;
            }
        }
    }

    println!(
        "Variables changed: {}/{}",
        changed_variables,
        result.parameters.len()
    );
    println!(
        "Average parameter change: {:.6e}",
        total_parameter_change / result.parameters.len() as f64
    );
    println!("Maximum parameter change: {:.6e}", max_parameter_change);

    // Show some example parameter changes
    println!("\n=== Sample Parameter Changes ===");
    let mut count = 0;
    for (var_name, final_params) in &result.parameters {
        if count >= 5 {
            break;
        } // Show only first 5
        if let Some(initial_params) = problem.initial_parameters.get(var_name) {
            let diff = final_params - initial_params;
            let change_norm = diff.norm();
            if change_norm > 1e-6 {
                println!("{}: change norm = {:.6e}", var_name, change_norm);
                count += 1;
            }
        }
    }

    println!("\n✅ Optimization completed successfully!");

    Ok(())
}
