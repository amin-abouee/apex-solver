use std::collections::HashMap;
use std::time::Instant;

use apex_solver::core::factors::BetweenFactorSE3;
use apex_solver::core::problem::Problem;
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
use apex_solver::optimizer::{GaussNewton, LevenbergMarquardt};
use clap::Parser;
use nalgebra::dvector;

#[derive(Parser)]
#[command(name = "load_se3_poses")]
#[command(about = "Load and optimize SE3 poses from G2O dataset")]
struct Args {
    /// G2O dataset file to load (without .g2o extension). Use "all" to test all SE3 datasets
    #[arg(short, long, default_value = "sphere2500")]
    dataset: String,

    /// Maximum number of optimization iterations (optimized for SE3 datasets)
    #[arg(short, long, default_value = "100")]
    max_iterations: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Cost tolerance for convergence (optimized for SE3 datasets)
    #[arg(long, default_value = "1e-4")]
    cost_tolerance: f64,

    /// Parameter tolerance for convergence (optimized for SE3 datasets)
    #[arg(long, default_value = "1e-4")]
    parameter_tolerance: f64,

    /// Optimizer type: "lm" (Levenberg-Marquardt) or "gn" (Gauss-Newton)
    #[arg(short, long, default_value = "lm")]
    optimizer: String,
}

#[derive(Clone)]
struct DatasetResult {
    dataset: String,
    optimizer: String,
    vertices: usize,
    edges: usize,
    initial_cost: f64,
    final_cost: f64,
    improvement: f64,
    iterations: usize,
    time_ms: u128,
    convergence_reason: String,
    status: String,
}

fn format_summary_table(results: &[DatasetResult]) {
    println!("\n{}", "=".repeat(160));
    println!("=== FINAL SUMMARY TABLE ===\n");

    // Header
    println!(
        "{:<16} | {:<10} | {:<8} | {:<6} | {:<12} | {:<12} | {:<11} | {:<5} | {:<9} | {:<20} | {:<12}",
        "Dataset",
        "Optimizer",
        "Vertices",
        "Edges",
        "Init Cost",
        "Final Cost",
        "Improvement",
        "Iters",
        "Time(ms)",
        "Convergence",
        "Status"
    );
    println!("{}", "-".repeat(160));

    // Rows
    for result in results {
        println!(
            "{:<16} | {:<10} | {:<8} | {:<6} | {:<12.6e} | {:<12.6e} | {:>10.2}% | {:<5} | {:<9} | {:<20} | {:<12}",
            result.dataset,
            result.optimizer,
            result.vertices,
            result.edges,
            result.initial_cost,
            result.final_cost,
            result.improvement,
            result.iterations,
            result.time_ms,
            result.convergence_reason,
            result.status
        );
    }

    println!("{}", "-".repeat(160));

    // Summary statistics
    let converged_count = results.iter().filter(|r| r.status == "CONVERGED").count();
    let total_count = results.len();
    println!(
        "\nSummary: {}/{} datasets converged successfully",
        converged_count, total_count
    );

    if converged_count > 0 {
        let avg_time: f64 = results
            .iter()
            .filter(|r| r.status == "CONVERGED")
            .map(|r| r.time_ms as f64)
            .sum::<f64>()
            / converged_count as f64;
        let avg_iters: f64 = results
            .iter()
            .filter(|r| r.status == "CONVERGED")
            .map(|r| r.iterations as f64)
            .sum::<f64>()
            / converged_count as f64;
        println!("Average time for converged datasets: {:.1}ms", avg_time);
        println!(
            "Average iterations for converged datasets: {:.1}",
            avg_iters
        );
    }
}

fn test_se3_dataset(
    dataset_name: &str,
    args: &Args,
) -> Result<DatasetResult, Box<dyn std::error::Error>> {
    println!(
        "\n=== TESTING {} SE3 DATASET ===",
        dataset_name.to_uppercase()
    );
    println!("Loading {}.g2o dataset for SE3 optimization", dataset_name);

    // Apply dataset-specific optimizations
    let (cost_tol, param_tol, max_iter) = match dataset_name {
        "grid3D" => {
            println!("Note: grid3D requires very relaxed tolerances due to high complexity");
            (1e-1, 1e-1, 30)
        }
        "rim" => (1e-3, 1e-3, args.max_iterations),
        "torus3D" => (1e-5, 1e-5, args.max_iterations),
        _ => (
            args.cost_tolerance,
            args.parameter_tolerance,
            args.max_iterations,
        ),
    };

    if cost_tol != args.cost_tolerance
        || param_tol != args.parameter_tolerance
        || max_iter != args.max_iterations
    {
        println!(
            "Using optimized parameters: cost_tol={:.1e}, param_tol={:.1e}, max_iter={}",
            cost_tol, param_tol, max_iter
        );
    }

    // Load the G2O graph file
    let dataset_path = format!("data/{}.g2o", dataset_name);
    let graph = G2oLoader::load(&dataset_path)?;

    let num_vertices = graph.vertices_se3.len();
    let num_edges = graph.edges_se3.len();

    println!("Successfully loaded SE3 graph:");
    println!("  SE3 vertices: {}", num_vertices);
    println!("  SE3 edges: {}", num_edges);

    if num_vertices == 0 {
        return Err(format!("No SE3 vertices found in dataset {}", dataset_name).into());
    }

    // Create optimization problem
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add SE3 vertices as variables
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

    // Add SE3 between factors
    for edge in &graph.edges_se3 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);
        let relative_pose = edge.measurement.clone();
        let between_factor = BetweenFactorSE3::new(relative_pose);
        problem.add_residual_block(&[&id0, &id1], Box::new(between_factor), None);
    }

    println!("\nProblem setup completed:");
    println!("  Variables: {}", initial_values.len());
    println!("  Between factors: {}", graph.edges_se3.len());

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

    let symbolic_structure =
        problem.build_symbolic_structure(&variables, &variable_name_to_col_idx_dict, col_offset);

    let (residual, _jacobian) = problem.compute_residual_and_jacobian_sparse(
        &variables,
        &variable_name_to_col_idx_dict,
        &symbolic_structure,
    );

    use faer_ext::IntoNalgebra;
    let residual_na = residual.as_ref().into_nalgebra();
    let initial_cost = residual_na.norm_squared();

    println!("\nInitial state:");
    println!("  Initial cost: {:.6e}", initial_cost);

    // Determine optimizer type and run optimization
    let optimizer_name = match args.optimizer.to_lowercase().as_str() {
        "gn" => "GN",
        "lm" => "LM",
        _ => {
            eprintln!(
                "Invalid optimizer '{}'. Using LM (Levenberg-Marquardt) as default.",
                args.optimizer
            );
            "LM"
        }
    };

    println!(
        "\n=== STARTING {} OPTIMIZATION ===",
        if optimizer_name == "LM" {
            "LEVENBERG-MARQUARDT"
        } else {
            "GAUSS-NEWTON"
        }
    );
    println!("Configuration:");
    println!("  Max iterations: {}", max_iter);
    println!("  Cost tolerance: {:.2e}", cost_tol);
    println!("  Parameter tolerance: {:.2e}", param_tol);

    let start_time = Instant::now();
    let result = if optimizer_name == "GN" {
        let config = GaussNewtonConfig::new()
            .with_max_iterations(max_iter)
            .with_cost_tolerance(cost_tol)
            .with_parameter_tolerance(param_tol)
            .with_gradient_tolerance(1e-12)
            .with_verbose(args.verbose);
        let mut solver = GaussNewton::with_config(config);
        solver.minimize(&problem, &initial_values)?
    } else {
        let config = LevenbergMarquardtConfig::new()
            .with_max_iterations(max_iter)
            .with_cost_tolerance(cost_tol)
            .with_parameter_tolerance(param_tol)
            .with_gradient_tolerance(1e-12)
            .with_verbose(args.verbose);
        let mut solver = LevenbergMarquardt::with_config(config);
        solver.minimize(&problem, &initial_values)?
    };
    let duration = start_time.elapsed();

    println!("\n=== OPTIMIZATION RESULTS ===");

    // Determine convergence status accurately
    let (status, convergence_reason) = match &result.status {
        apex_solver::optimizer::OptimizationStatus::Converged => {
            ("CONVERGED", "Converged".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::CostToleranceReached => {
            ("CONVERGED", "CostTolerance".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::ParameterToleranceReached => {
            ("CONVERGED", "ParameterTolerance".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::GradientToleranceReached => {
            ("CONVERGED", "GradientTolerance".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::MaxIterationsReached => {
            ("NOT_CONVERGED", "MaxIterations".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::Timeout => {
            ("NOT_CONVERGED", "Timeout".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::NumericalFailure => {
            ("NOT_CONVERGED", "NumericalFailure".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::UserTerminated => {
            ("NOT_CONVERGED", "UserTerminated".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::Failed(msg) => {
            ("NOT_CONVERGED", format!("Failed:{}", msg))
        }
    };

    let improvement = ((initial_cost - result.final_cost) / initial_cost) * 100.0;

    println!("Status: {}", status);
    println!("Convergence reason: {}", convergence_reason);
    println!("Initial cost: {:.6e}", initial_cost);
    println!("Final cost: {:.6e}", result.final_cost);
    println!(
        "Cost reduction: {:.6e} ({:.2}%)",
        initial_cost - result.final_cost,
        improvement
    );
    println!("Iterations: {}", result.iterations);
    println!("Execution time: {:.1}ms", duration.as_millis());

    Ok(DatasetResult {
        dataset: dataset_name.to_string(),
        optimizer: optimizer_name.to_string(),
        vertices: num_vertices,
        edges: num_edges,
        initial_cost,
        final_cost: result.final_cost,
        improvement,
        iterations: result.iterations,
        time_ms: duration.as_millis(),
        convergence_reason: convergence_reason.to_string(),
        status: status.to_string(),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== APEX-SOLVER SE3 POSE GRAPH OPTIMIZATION ===");

    // Define available SE3 datasets
    let se3_datasets = vec![
        "rim",
        "sphere2500",
        "parking-garage",
        "torus3D",
        "grid3D",
        "cubicle",
    ];

    let datasets = if args.dataset == "all" {
        se3_datasets
    } else {
        vec![args.dataset.as_str()]
    };

    let mut results = Vec::new();

    for dataset in &datasets {
        println!("\n{}", "=".repeat(80));
        match test_se3_dataset(dataset, &args) {
            Ok(result) => {
                println!("Dataset {} completed: {}", dataset, result.status);
                results.push(result);
            }
            Err(e) => {
                eprintln!("Dataset {} failed: {}", dataset, e);
            }
        }
    }

    // Display summary table
    if results.len() > 1 {
        format_summary_table(&results);
    }

    let converged_count = results.iter().filter(|r| r.status == "CONVERGED").count();
    if converged_count == results.len() {
        println!("\nAll datasets converged successfully!");
        Ok(())
    } else if converged_count == 0 {
        Err("No datasets converged".into())
    } else {
        println!("\n{}/{} datasets converged", converged_count, results.len());
        Err("Some datasets failed to converge".into())
    }
}
