use std::collections::HashMap;
use std::time::Instant;

use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BetweenFactorSE2, BetweenFactorSE3, PriorFactor};
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::dog_leg::DogLegConfig;
use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
use apex_solver::optimizer::{DogLeg, GaussNewton, LevenbergMarquardt, OptimizationStatus};
use clap::Parser;
use nalgebra::dvector;

#[derive(Parser)]
#[command(name = "compare_optimizers")]
#[command(about = "Compare LM, GN, and DL optimizers on real G2O datasets")]
struct Args {
    /// Maximum number of optimization iterations
    #[arg(short, long, default_value = "100")]
    max_iterations: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Cost tolerance for convergence
    #[arg(long, default_value = "1e-3")]
    cost_tolerance: f64,

    /// Parameter tolerance for convergence
    #[arg(long, default_value = "1e-3")]
    parameter_tolerance: f64,
}

#[derive(Clone)]
struct BenchmarkResult {
    dataset: String,
    manifold: String,
    optimizer: String,
    vertices: usize,
    edges: usize,
    initial_cost: f64,
    final_cost: f64,
    improvement: f64,
    iterations: usize,
    time_ms: u128,
    status: String,
}

fn print_summary_table(results: &[BenchmarkResult]) {
    println!("\n{}", "=".repeat(150));
    println!("=== OPTIMIZER COMPARISON SUMMARY ===\n");

    println!(
        "{:<12} | {:<8} | {:<10} | {:<8} | {:<6} | {:<12} | {:<12} | {:<11} | {:<5} | {:<9} | {:<10}",
        "Dataset",
        "Manifold",
        "Optimizer",
        "Vertices",
        "Edges",
        "Init Cost",
        "Final Cost",
        "Improvement",
        "Iters",
        "Time(ms)",
        "Status"
    );
    println!("{}", "-".repeat(150));

    for result in results {
        println!(
            "{:<12} | {:<8} | {:<10} | {:<8} | {:<6} | {:<12.6e} | {:<12.6e} | {:>10.2}% | {:<5} | {:<9} | {:<10}",
            result.dataset,
            result.manifold,
            result.optimizer,
            result.vertices,
            result.edges,
            result.initial_cost,
            result.final_cost,
            result.improvement,
            result.iterations,
            result.time_ms,
            result.status
        );
    }

    println!("{}", "-".repeat(150));
}

fn run_optimizer_se3(
    problem: &Problem,
    initial_values: &HashMap<String, (ManifoldType, nalgebra::DVector<f64>)>,
    optimizer_name: &str,
    max_iterations: usize,
    cost_tolerance: f64,
    parameter_tolerance: f64,
    verbose: bool,
) -> Result<(f64, usize, OptimizationStatus, u128), Box<dyn std::error::Error>> {
    let start = Instant::now();

    let result = match optimizer_name {
        "LM" => {
            let config = LevenbergMarquardtConfig::new()
                .with_max_iterations(max_iterations)
                .with_cost_tolerance(cost_tolerance)
                .with_parameter_tolerance(parameter_tolerance)
                .with_verbose(verbose);
            let mut solver = LevenbergMarquardt::with_config(config);
            solver.optimize(problem, initial_values)?
        }
        "GN" => {
            let config = GaussNewtonConfig::new()
                .with_max_iterations(max_iterations)
                .with_cost_tolerance(cost_tolerance)
                .with_parameter_tolerance(parameter_tolerance)
                .with_verbose(verbose);
            let mut solver = GaussNewton::with_config(config);
            solver.optimize(problem, initial_values)?
        }
        "DL" => {
            let config = DogLegConfig::new()
                .with_max_iterations(max_iterations)
                .with_cost_tolerance(cost_tolerance)
                .with_parameter_tolerance(parameter_tolerance)
                .with_verbose(verbose);
            let mut solver = DogLeg::with_config(config);
            solver.optimize(problem, initial_values)?
        }
        _ => unreachable!(),
    };

    let elapsed = start.elapsed();

    Ok((
        result.final_cost,
        result.iterations,
        result.status,
        elapsed.as_millis(),
    ))
}

fn test_se3_dataset(dataset_name: &str, args: &Args, all_results: &mut Vec<BenchmarkResult>) {
    println!("\n{}", "=".repeat(80));
    println!("=== TESTING {} (SE3) ===", dataset_name.to_uppercase());

    let file_path = format!("data/{}.g2o", dataset_name);
    let graph = match G2oLoader::load(&file_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to load {}: {}", file_path, e);
            return;
        }
    };

    let num_vertices = graph.vertices_se3.len();
    let num_edges = graph.edges_se3.len();

    println!("Loaded: {} vertices, {} edges", num_vertices, num_edges);

    // Create initial values
    let mut initial_values = HashMap::new();
    let mut vertex_ids: Vec<_> = graph.vertices_se3.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se3.get(&id) {
            let var_name = format!("x{}", id);
            let quat = vertex.pose.rotation_quaternion();
            let trans = vertex.pose.translation();
            let se3_data = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];
            initial_values.insert(var_name, (ManifoldType::SE3, se3_data));
        }
    }

    // Create problem
    let mut problem = Problem::new();

    // Add prior factor on first vertex to handle gauge freedom
    // This prevents rank-deficient Hessian and improves convergence for all optimizers
    if let Some(&first_id) = vertex_ids.first() {
        if let Some(first_vertex) = graph.vertices_se3.get(&first_id) {
            let var_name = format!("x{}", first_id);
            let quat = first_vertex.pose.rotation_quaternion();
            let trans = first_vertex.pose.translation();
            let prior_value = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];

            let prior_factor = PriorFactor { data: prior_value };
            // Use HuberLoss with scale=1.0 (allows slight movement if needed)
            let huber_loss = HuberLoss::new(1.0).expect("Failed to create HuberLoss");
            problem.add_residual_block(
                &[&var_name],
                Box::new(prior_factor),
                Some(Box::new(huber_loss)),
            );
        }
    }

    for edge in &graph.edges_se3 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);
        let factor = BetweenFactorSE3::new(edge.measurement.clone());
        problem.add_residual_block(&[&id0, &id1], Box::new(factor), None);
    }

    // Compute initial cost
    let variables = problem.initialize_variables(&initial_values);
    let mut variable_name_to_col_idx_dict = HashMap::new();
    let mut col_offset = 0;
    let mut sorted_vars: Vec<_> = variables.keys().cloned().collect();
    sorted_vars.sort();
    for var_name in &sorted_vars {
        variable_name_to_col_idx_dict.insert(var_name.clone(), col_offset);
        col_offset += variables[var_name].get_size();
    }

    let symbolic_structure = problem
        .build_symbolic_structure(&variables, &variable_name_to_col_idx_dict, col_offset)
        .expect("Failed to build symbolic structure");

    let (residual, _) = problem
        .compute_residual_and_jacobian_sparse(
            &variables,
            &variable_name_to_col_idx_dict,
            &symbolic_structure,
        )
        .expect("Failed to compute residual");

    // Compute initial cost using faer's norm
    let init_cost = residual.as_ref().squared_norm_l2();

    println!("Initial cost: {:.6e}", init_cost);

    // Test all optimizers
    for opt_name in &["LM", "GN", "DL"] {
        println!("\n--- Testing {} ---", opt_name);

        match run_optimizer_se3(
            &problem,
            &initial_values,
            opt_name,
            args.max_iterations,
            args.cost_tolerance,
            args.parameter_tolerance,
            args.verbose,
        ) {
            Ok((final_cost, iterations, status, time_ms)) => {
                let improvement = ((init_cost - final_cost) / init_cost) * 100.0;
                let status_str = match status {
                    OptimizationStatus::Converged => "CONVERGED",
                    OptimizationStatus::CostToleranceReached => "CONVERGED",
                    OptimizationStatus::ParameterToleranceReached => "CONVERGED",
                    OptimizationStatus::GradientToleranceReached => "CONVERGED",
                    _ => "NOT_CONVERGED",
                };

                println!("Final cost: {:.6e}", final_cost);
                println!("Iterations: {}", iterations);
                println!("Time: {}ms", time_ms);
                println!("Status: {}", status_str);

                all_results.push(BenchmarkResult {
                    dataset: dataset_name.to_string(),
                    manifold: "SE3".to_string(),
                    optimizer: opt_name.to_string(),
                    vertices: num_vertices,
                    edges: num_edges,
                    initial_cost: init_cost,
                    final_cost,
                    improvement,
                    iterations,
                    time_ms,
                    status: status_str.to_string(),
                });
            }
            Err(e) => {
                eprintln!("{} failed: {}", opt_name, e);
            }
        }
    }
}

fn test_se2_dataset(dataset_name: &str, args: &Args, all_results: &mut Vec<BenchmarkResult>) {
    println!("\n{}", "=".repeat(80));
    println!("=== TESTING {} (SE2) ===", dataset_name.to_uppercase());

    let file_path = format!("data/{}.g2o", dataset_name);
    let graph = match G2oLoader::load(&file_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to load {}: {}", file_path, e);
            return;
        }
    };

    let num_vertices = graph.vertices_se2.len();
    let num_edges = graph.edges_se2.len();

    println!("Loaded: {} vertices, {} edges", num_vertices, num_edges);

    // Create initial values
    let mut initial_values = HashMap::new();
    let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se2.get(&id) {
            let var_name = format!("x{}", id);
            let se2_data = dvector![vertex.pose.x(), vertex.pose.y(), vertex.pose.angle()];
            initial_values.insert(var_name, (ManifoldType::SE2, se2_data));
        }
    }

    // Create problem
    let mut problem = Problem::new();

    // Add prior factor on first vertex to handle gauge freedom
    // This prevents rank-deficient Hessian and improves convergence for all optimizers
    if let Some(&first_id) = vertex_ids.first() {
        if let Some(first_vertex) = graph.vertices_se2.get(&first_id) {
            let var_name = format!("x{}", first_id);
            let trans = first_vertex.pose.translation();
            let angle = first_vertex.pose.rotation_angle();
            let prior_value = dvector![trans.x, trans.y, angle];

            let prior_factor = PriorFactor { data: prior_value };
            // Use HuberLoss with scale=1.0 (allows slight movement if needed)
            let huber_loss = HuberLoss::new(1.0).expect("Failed to create HuberLoss");
            problem.add_residual_block(
                &[&var_name],
                Box::new(prior_factor),
                Some(Box::new(huber_loss)),
            );
        }
    }

    for edge in &graph.edges_se2 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);
        let factor = BetweenFactorSE2::from_se2(edge.measurement.clone());
        problem.add_residual_block(&[&id0, &id1], Box::new(factor), None);
    }

    // Compute initial cost
    let variables = problem.initialize_variables(&initial_values);
    let mut variable_name_to_col_idx_dict = HashMap::new();
    let mut col_offset = 0;
    let mut sorted_vars: Vec<_> = variables.keys().cloned().collect();
    sorted_vars.sort();
    for var_name in &sorted_vars {
        variable_name_to_col_idx_dict.insert(var_name.clone(), col_offset);
        col_offset += variables[var_name].get_size();
    }

    let symbolic_structure = problem
        .build_symbolic_structure(&variables, &variable_name_to_col_idx_dict, col_offset)
        .expect("Failed to build symbolic structure");

    let (residual, _) = problem
        .compute_residual_and_jacobian_sparse(
            &variables,
            &variable_name_to_col_idx_dict,
            &symbolic_structure,
        )
        .expect("Failed to compute residual");

    // Compute initial cost using faer's norm
    let init_cost = residual.as_ref().squared_norm_l2();

    println!("Initial cost: {:.6e}", init_cost);

    // Test all optimizers
    for opt_name in &["LM", "GN", "DL"] {
        println!("\n--- Testing {} ---", opt_name);

        match run_optimizer_se3(
            &problem,
            &initial_values,
            opt_name,
            args.max_iterations,
            args.cost_tolerance,
            args.parameter_tolerance,
            args.verbose,
        ) {
            Ok((final_cost, iterations, status, time_ms)) => {
                let improvement = ((init_cost - final_cost) / init_cost) * 100.0;
                let status_str = match status {
                    OptimizationStatus::Converged => "CONVERGED",
                    OptimizationStatus::CostToleranceReached => "CONVERGED",
                    OptimizationStatus::ParameterToleranceReached => "CONVERGED",
                    OptimizationStatus::GradientToleranceReached => "CONVERGED",
                    _ => "NOT_CONVERGED",
                };

                println!("Final cost: {:.6e}", final_cost);
                println!("Iterations: {}", iterations);
                println!("Time: {}ms", time_ms);
                println!("Status: {}", status_str);

                all_results.push(BenchmarkResult {
                    dataset: dataset_name.to_string(),
                    manifold: "SE2".to_string(),
                    optimizer: opt_name.to_string(),
                    vertices: num_vertices,
                    edges: num_edges,
                    initial_cost: init_cost,
                    final_cost,
                    improvement,
                    iterations,
                    time_ms,
                    status: status_str.to_string(),
                });
            }
            Err(e) => {
                eprintln!("{} failed: {}", opt_name, e);
            }
        }
    }
}

fn main() {
    let args = Args::parse();

    println!("=== APEX-SOLVER OPTIMIZER COMPARISON ===");
    println!("Comparing LM, GN, and DL optimizers on real datasets\n");

    let mut all_results = Vec::new();

    // Test SE3 datasets
    test_se3_dataset("rim", &args, &mut all_results);
    test_se3_dataset("cubicle", &args, &mut all_results);

    // Test SE2 datasets
    test_se2_dataset("intel", &args, &mut all_results);
    test_se2_dataset("mit", &args, &mut all_results);

    // Print summary
    print_summary_table(&all_results);

    println!("\nComparison complete!");
}
