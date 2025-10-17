use std::collections::HashMap;
use std::time::Instant;

use apex_solver::core::factors::BetweenFactorSE2;
use apex_solver::core::problem::Problem;
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
use apex_solver::optimizer::{GaussNewton, LevenbergMarquardt};
use clap::Parser;
use nalgebra::dvector;

#[derive(Parser)]
#[command(name = "load_se2_poses")]
#[command(about = "Load and optimize SE2 poses from G2O dataset")]
struct Args {
    /// G2O dataset file to load (without .g2o extension). Use "all" to test all SE2 datasets
    #[arg(short, long, default_value = "M3500")]
    dataset: String,

    /// Maximum number of optimization iterations
    #[arg(short, long, default_value = "150")]
    max_iterations: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

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

fn test_dataset(
    dataset_name: &str,
    args: &Args,
) -> Result<DatasetResult, Box<dyn std::error::Error>> {
    println!("\n=== TESTING {} DATASET ===", dataset_name.to_uppercase());
    println!("Loading {}.g2o dataset for SE2 optimization", dataset_name);

    // Load the G2O graph file
    let dataset_path = format!("data/{}.g2o", dataset_name);
    let graph = G2oLoader::load(&dataset_path)?;

    let num_vertices = graph.vertices_se2.len();
    let num_edges = graph.edges_se2.len();

    println!("Successfully loaded SE2 graph:");
    println!("  SE2 vertices: {}", num_vertices);
    println!("  SE2 edges: {}", num_edges);

    // Check if we have any vertices
    if num_vertices == 0 {
        return Err(format!("No SE2 vertices found in dataset {}", dataset_name).into());
    }

    // Create optimization problem
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add SE2 vertices as variables
    let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se2.get(&id) {
            let var_name = format!("x{}", id);
            let se2_data = dvector![vertex.x(), vertex.y(), vertex.theta()];
            initial_values.insert(var_name, (ManifoldType::SE2, se2_data));
        }
    }

    // Add SE2 between factors
    for edge in &graph.edges_se2 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);

        let dx = edge.measurement.translation().x;
        let dy = edge.measurement.translation().y;
        let dtheta = edge.measurement.angle();

        let between_factor = BetweenFactorSE2::new(dx, dy, dtheta);
        problem.add_residual_block(&[&id0, &id1], Box::new(between_factor), None);
    }

    // Fix the first pose to remove gauge freedom (all 3 DOF: x, y, theta)
    let first_var_name = format!("x{}", vertex_ids[0]);
    problem.fix_variable(&first_var_name, 0);
    problem.fix_variable(&first_var_name, 1);
    problem.fix_variable(&first_var_name, 2);

    println!("\nProblem setup completed:");
    println!("  Variables: {}", initial_values.len());
    println!("  Between factors: {}", graph.edges_se2.len());
    println!("  Fixed first pose: {} (all 3 DOF)", first_var_name);

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

    let (residual, _jacobian) = problem
        .compute_residual_and_jacobian_sparse(
            &variables,
            &variable_name_to_col_idx_dict,
            &symbolic_structure,
        )
        .expect("Failed to compute residual and jacobian");

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

    let start_time = Instant::now();
    let result = if optimizer_name == "GN" {
        let config = GaussNewtonConfig::new()
            .with_max_iterations(args.max_iterations)
            .with_cost_tolerance(1e-4)
            .with_parameter_tolerance(1e-4)
            .with_gradient_tolerance(1e-10)
            .with_verbose(args.verbose);
        let mut solver = GaussNewton::with_config(config);
        solver.minimize(&problem, &initial_values)?
    } else {
        let config = LevenbergMarquardtConfig::new()
            .with_max_iterations(args.max_iterations)
            .with_cost_tolerance(1e-4)
            .with_parameter_tolerance(1e-4)
            .with_gradient_tolerance(1e-10)
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

    println!("=== APEX-SOLVER SE2 POSE GRAPH OPTIMIZATION ===");

    // Determine which datasets to test
    let datasets = if args.dataset == "all" {
        vec![
            "city10000",
            "M3500",
            "M3500a",
            "M3500b",
            "M3500c",
            "intel",
            "mit",
            "manhattanOlson3500",
            "ring",
            "ringCity",
        ]
    } else {
        vec![args.dataset.as_str()]
    };

    let mut results = Vec::new();

    for dataset in &datasets {
        println!("\n{}", "=".repeat(80));
        match test_dataset(dataset, &args) {
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
