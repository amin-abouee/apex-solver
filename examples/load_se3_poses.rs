use std::collections::HashMap;
use std::time::Instant;

use apex_solver::core::factors::BetweenFactorSE3;
use apex_solver::core::problem::Problem;
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::{OptimizerConfig, OptimizerType, solve_problem};
use nalgebra as na;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== APEX-SOLVER SPHERE2500 LM BENCHMARK ===");
    println!("Loading sphere2500.g2o dataset for Levenberg-Marquardt optimization comparison");

    // Load the G2O graph file
    let graph = G2oLoader::load("data/sphere2500.g2o")?;

    println!("Successfully loaded SE3 graph:");
    println!("  SE3 vertices: {}", graph.vertices_se3.len());
    println!("  SE3 edges: {}", graph.edges_se3.len());

    // Display first few vertices
    println!("\nFirst 5 SE3 vertices:");
    let mut vertex_ids: Vec<_> = graph.vertices_se3.keys().cloned().collect();
    vertex_ids.sort();

    for &id in vertex_ids.iter().take(5) {
        if let Some(vertex) = graph.vertices_se3.get(&id) {
            let quat = vertex.pose.rotation_quaternion();
            let trans = vertex.pose.translation();
            println!(
                "  x{}: quat=[{:.6},{:.6},{:.6},{:.6}] trans=[{:.6},{:.6},{:.6}]",
                id, quat.i, quat.j, quat.k, quat.w, trans.x, trans.y, trans.z
            );
        }
    }

    // Display first few edges
    println!("\nFirst 5 SE3 edges (relative transformations):");
    for edge in graph.edges_se3.iter().take(5) {
        let quat = edge.measurement.rotation_quaternion();
        let trans = edge.measurement.translation();
        println!(
            "  Edge {}->{}: quat=[{:.6},{:.6},{:.6},{:.6}] trans=[{:.6},{:.6},{:.6}]",
            edge.from, edge.to, quat.i, quat.j, quat.k, quat.w, trans.x, trans.y, trans.z
        );
    }

    // Create optimization problem
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add SE3 vertices as variables
    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se3.get(&id) {
            let var_name = format!("x{}", id);
            // Extract quaternion and translation from SE3
            let quat = vertex.pose.rotation_quaternion();
            let trans = vertex.pose.translation();
            // Format: [qx, qy, qz, qw, tx, ty, tz]
            let se3_data = na::dvector![quat.i, quat.j, quat.k, quat.w, trans.x, trans.y, trans.z];
            initial_values.insert(var_name, (ManifoldType::SE3, se3_data));
        }
    }

    // Add SE3 between factors
    for edge in &graph.edges_se3 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);

        let relative_pose = edge.measurement.clone();
        let between_factor = BetweenFactorSE3::new(relative_pose);

        problem.add_residual_block(
            &[&id0, &id1],
            Box::new(between_factor),
            None, // No loss function
        );
    }

    println!("\nProblem setup completed:");
    println!("  Variables: {}", initial_values.len());
    println!("  Between factors: {}", graph.edges_se3.len());

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
        problem.build_symbolic_structure(&variables, &variable_name_to_col_idx_dict);

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

    println!("\n=== STARTING LEVENBERG-MARQUARDT OPTIMIZATION ===");
    println!("Configuration:");
    println!("  Optimizer: Levenberg-Marquardt");
    println!("  Max iterations: 50");
    println!("  Cost tolerance: 1e-12");
    println!("  Parameter tolerance: 1e-12");
    println!("  Gradient tolerance: 1e-12");

    let config = OptimizerConfig {
        optimizer_type: OptimizerType::LevenbergMarquardt,
        max_iterations: 50,
        cost_tolerance: 1e-12,
        parameter_tolerance: 1e-12,
        gradient_tolerance: 1e-12,
        verbose: false, // Set to false for cleaner comparison output
        ..Default::default()
    };

    let start_time = Instant::now();
    let result = solve_problem(&problem, &initial_values, config)?;
    let duration = start_time.elapsed();

    println!("\n=== APEX-SOLVER SPHERE2500 LM OPTIMIZATION RESULTS ===");

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

            // Note: Final residual computation omitted for simplicity in this benchmark
            println!("\nFinal optimization completed successfully.");

            println!("\n=== BENCHMARK SUMMARY ===");
            println!("Dataset: sphere2500.g2o");
            println!("Solver: APEX Levenberg-Marquardt");
            println!("Result: ✅ CONVERGED");
            println!(
                "Performance: {:.1}ms total, {:.6e} final cost",
                duration.as_millis(),
                result.final_cost
            );
        }
        _ => {
            println!("Status: FAILED");
            println!("Error: {:?}", result.status);
            println!("Initial cost: {:.12e}", initial_cost);
            println!("Iterations: {}", result.iterations);
            println!("Execution time: {:.1}ms", duration.as_millis());

            println!("\n=== BENCHMARK SUMMARY ===");
            println!("Dataset: sphere2500.g2o");
            println!("Solver: APEX Levenberg-Marquardt");
            println!("Result: ❌ FAILED");
            println!("Error: Optimization did not converge");
        }
    }

    Ok(())
}
