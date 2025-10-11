use std::collections::HashMap;
use std::time::Instant;

use apex_solver::core::factors::BetweenFactorSE2;
use apex_solver::core::problem::Problem;
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::LevenbergMarquardt;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
use clap::Parser;
use faer::col;

#[derive(Parser)]
#[command(name = "load_se2_poses")]
#[command(about = "Load and optimize SE2 poses from G2O dataset")]
struct Args {
    /// G2O dataset file to load (without .g2o extension). Use "all" to test all SE2 datasets
    #[arg(short, long, default_value = "M3500")]
    dataset: String,

    /// Maximum number of optimization iterations
    #[arg(short, long, default_value = "50")]
    max_iterations: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Compare with tiny-solver performance expectations
    #[arg(short, long)]
    compare: bool,
}

fn test_dataset(dataset_name: &str, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TESTING {} DATASET ===", dataset_name.to_uppercase());
    println!("Loading {}.g2o dataset for SE2 optimization", dataset_name);

    // Load the G2O graph file
    let dataset_path = format!("data/{}.g2o", dataset_name);
    let graph = G2oLoader::load(&dataset_path)?;

    println!("Successfully loaded SE2 graph:");
    println!("  SE2 vertices: {}", graph.vertices_se2.len());
    println!("  SE2 edges: {}", graph.edges_se2.len());

    // Display first few vertices
    println!("\nFirst 10 SE2 vertices:");
    let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
    vertex_ids.sort();

    for &id in vertex_ids.iter().take(10) {
        if let Some(vertex) = graph.vertices_se2.get(&id) {
            println!(
                "  x{}: theta={:.6}, x={:.6}, y={:.6}",
                id,
                vertex.theta(),
                vertex.x(),
                vertex.y()
            );
        }
    }

    // Display first few edges
    println!("\nFirst 10 SE2 edges (relative transformations):");
    for edge in graph.edges_se2.iter().take(10) {
        println!(
            "  Edge {}->{}: dx={:.6}, dy={:.6}, dtheta={:.6}",
            edge.from,
            edge.to,
            edge.measurement.translation()[0],
            edge.measurement.translation()[1],
            edge.measurement.angle()
        );
    }

    // Create optimization problem
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add SE2 vertices as variables
    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se2.get(&id) {
            let var_name = format!("x{}", id);
            // Format: [theta, x, y] - MATCHES TINY-SOLVER ORDER
            let se2_data = col![vertex.theta(), vertex.x(), vertex.y()];
            initial_values.insert(var_name, (ManifoldType::SE2, se2_data));
        }
    }

    // Add SE2 between factors
    for edge in &graph.edges_se2 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);

        let dx = edge.measurement.translation()[0];
        let dy = edge.measurement.translation()[1];
        let dtheta = edge.measurement.angle();

        let between_factor = BetweenFactorSE2::new(dx, dy, dtheta);

        problem.add_residual_block(
            &[&id0, &id1],
            Box::new(between_factor),
            None, // No loss function
        );
    }

    println!("\nProblem setup completed:");
    println!("  Variables: {}", initial_values.len());
    println!("  Between factors: {}", graph.edges_se2.len());

    // Initialize problem variables
    let variables = problem.initialize_variables(&initial_values);
    println!("  Variable dimensions:");
    for (name, var) in variables.iter().take(3) {
        println!("    {}: {} dimensions", name, var.get_size());
    }

    // Compute initial cost for comparison
    let variables = problem.initialize_variables(&initial_values);

    // Create variable mapping
    let mut variable_name_to_col_idx_dict = std::collections::HashMap::new();
    let mut col_offset = 0;
    let mut sorted_vars: Vec<_> = variables.keys().cloned().collect();
    sorted_vars.sort();
    for var_name in &sorted_vars {
        variable_name_to_col_idx_dict.insert(var_name.clone(), col_offset);
        col_offset += variables[var_name].get_size();
    }

    // Build symbolic structure
    let symbolic_structure =
        problem.build_symbolic_structure(&variables, &variable_name_to_col_idx_dict, col_offset);

    // Compute residual and jacobian
    let (residual, _jacobian) = problem.compute_residual_and_jacobian_sparse(
        &variables,
        &variable_name_to_col_idx_dict,
        &symbolic_structure,
    );

    use faer_ext::IntoNalgebra;
    let residual_na = residual.as_ref().into_nalgebra();
    let initial_cost = residual_na.norm_squared();

    println!("\n=== COMPUTING INITIAL STATE ===");
    println!("Initial residual vector:");
    println!("  Length: {}", residual_na.len());
    println!("  Norm²: {:.12e}", initial_cost);
    println!("  Norm: {:.12e}", residual_na.norm());

    println!("Initial residuals (first 10):");
    for i in 0..std::cmp::min(10, residual_na.len()) {
        println!("  residual[{}] = {:.8}", i, residual_na[i]);
    }

    // Display graph statistics
    println!("\nGraph statistics:");
    let total_poses = vertex_ids.len();
    let total_constraints = graph.edges_se2.len();
    let dof_before = total_poses * 3; // Each SE2 pose has 3 DOF
    let dof_after = dof_before - 3; // Fix one pose (3 DOF removed)
    let constraint_dim = total_constraints * 3; // Each between factor has 3 constraints

    println!("  Total poses: {}", total_poses);
    println!("  Total constraints: {}", total_constraints);
    println!("  DOF before fixing: {}", dof_before);
    println!("  DOF after fixing first pose: {}", dof_after);
    println!("  Total constraint dimensions: {}", constraint_dim);

    if constraint_dim >= dof_after {
        println!("  System is over-constrained (good for optimization)");
    } else {
        println!("  System is under-constrained (may need more constraints)");
    }

    println!("\n=== STARTING LEVENBERG-MARQUARDT OPTIMIZATION ===");
    println!("Configuration:");
    println!("  Optimizer: Levenberg-Marquardt");
    println!("  Max iterations: {}", args.max_iterations);
    println!("  Cost tolerance: 1e-6");
    println!("  Parameter tolerance: 1e-6");
    println!("  Gradient tolerance: 1e-6");

    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(args.max_iterations)
        .with_cost_tolerance(1e-6)
        .with_parameter_tolerance(1e-6)
        .with_gradient_tolerance(1e-6)
        .with_verbose(args.verbose);

    let start_time = Instant::now();
    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.minimize(&problem, &initial_values)?;
    let duration = start_time.elapsed();

    println!("\n=== APEX-SOLVER SE2 GN OPTIMIZATION RESULTS ===");

    match result.status {
        apex_solver::optimizer::OptimizationStatus::Converged
        | apex_solver::optimizer::OptimizationStatus::CostToleranceReached
        | apex_solver::optimizer::OptimizationStatus::ParameterToleranceReached
        | apex_solver::optimizer::OptimizationStatus::GradientToleranceReached => {
            println!("Status: SUCCESS");
            println!("Initial cost: {:.12e}", initial_cost);
            println!("Final cost: {:.12e}", result.final_cost);
            println!(
                "Cost reduction: {:.12e} ({:.2}%)",
                initial_cost - result.final_cost,
                ((initial_cost - result.final_cost) / initial_cost) * 100.0
            );
            println!("Iterations: {}", result.iterations);
            println!("Execution time: {:.1}ms", duration.as_millis());
            if result.iterations > 0 {
                println!(
                    "Time per iteration: ~{:.1}ms",
                    duration.as_millis() as f64 / result.iterations as f64
                );
            }

            println!("\nFinal optimization completed successfully.");

            println!("\n=== BENCHMARK SUMMARY ===");
            println!("Dataset: {}.g2o", dataset_name);
            println!("Solver: APEX Levenberg-Marquardt with Analytical Jacobians");
            println!("Result: ✅ CONVERGED");
            println!(
                "Performance: {:.1}ms total, {:.6e} final cost",
                duration.as_millis(),
                result.final_cost
            );

            // Tiny-solver performance comparison (if requested)
            if args.compare {
                println!("\n=== COMPARISON WITH TINY-SOLVER ===");
                match dataset_name {
                    "M3500" => {
                        println!("Expected (tiny-solver): ~1.095e4 final cost, ~30s time");
                        println!(
                            "APEX Result: {:.3e} final cost, {:.1}s time",
                            result.final_cost,
                            duration.as_secs_f64()
                        );
                        if result.final_cost < 1.2e4 {
                            println!("✅ Cost comparable to tiny-solver");
                        } else {
                            println!("⚠️ Cost higher than expected");
                        }
                    }
                    _ => {
                        println!("No tiny-solver reference available for {}", dataset_name);
                    }
                }
            }

            Ok(())
        }
        _ => {
            println!("Status: FAILED");
            println!("Error: {:?}", result.status);
            println!("Initial cost: {:.12e}", initial_cost);
            println!("Iterations: {}", result.iterations);
            println!("Execution time: {:.1}ms", duration.as_millis());

            println!("\n=== BENCHMARK SUMMARY ===");
            println!("Dataset: {}.g2o", dataset_name);
            println!("Solver: APEX Levenberg-Marquardt");
            println!("Result: ❌ FAILED");
            println!("Error: Optimization did not converge");

            Err(format!("Dataset {} failed to converge", dataset_name).into())
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== APEX-SOLVER SE2 ANALYTICAL JACOBIAN BENCHMARK ===");

    // Determine which datasets to test
    let datasets = if args.dataset == "all" {
        vec!["M3500", "intel", "mit", "manhattanOlson3500", "rim", "ring"]
    } else {
        vec![args.dataset.as_str()]
    };

    let mut successful_datasets = Vec::new();
    let mut failed_datasets = Vec::new();

    for dataset in &datasets {
        println!("\n{}", "=".repeat(60));
        match test_dataset(dataset, &args) {
            Ok(()) => {
                successful_datasets.push(*dataset);
                println!("✅ {} completed successfully", dataset);
            }
            Err(e) => {
                failed_datasets.push(*dataset);
                println!("❌ {} failed: {}", dataset, e);
            }
        }
    }

    // Final summary
    println!("\n{}", "=".repeat(60));
    println!("=== FINAL SUMMARY ===");
    println!("Total datasets tested: {}", datasets.len());
    println!(
        "Successful: {} {:?}",
        successful_datasets.len(),
        successful_datasets
    );
    println!("Failed: {} {:?}", failed_datasets.len(), failed_datasets);

    if successful_datasets.len() == datasets.len() {
        println!("🎉 ALL DATASETS CONVERGED SUCCESSFULLY!");
        Ok(())
    } else {
        Err("Some datasets failed to converge".into())
    }
}
