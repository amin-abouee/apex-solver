use apex_solver::core::problem::Problem;
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <dataset_name>", args[0]);
        eprintln!("Available datasets: sphere2500, parking-garage, rim, grid3D, torus3D, cubicle");
        std::process::exit(1);
    }

    let dataset = &args[1];
    let file_path = format!("data/{}.g2o", dataset);

    println!("==============================================");
    println!("  Jacobi Scaling Benchmark");
    println!("==============================================");
    println!("Dataset: {}", dataset);
    println!("File: {}", file_path);
    println!();

    // Load graph
    let graph = match G2oLoader::load(&file_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading graph: {}", e);
            std::process::exit(1);
        }
    };

    println!("Loaded graph:");
    println!("  SE3 Vertices: {}", graph.vertices_se3.len());
    println!("  SE3 Edges: {}", graph.edges_se3.len());
    println!();

    // Collect vertices and edges
    let mut vertex_ids: Vec<usize> = graph.vertices_se3.keys().cloned().collect();
    vertex_ids.sort();

    // Build problem (shared for both runs)
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add vertices to initial values
    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se3.get(&id) {
            let var_name = format!("x{}", id);
            let quat = vertex.pose.rotation_quaternion();
            let trans = vertex.pose.translation();
            let se3_data =
                nalgebra::dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];
            initial_values.insert(var_name, (ManifoldType::SE3, se3_data));
        }
    }

    // Add edges to problem
    use apex_solver::core::factors::BetweenFactorSE3;
    use apex_solver::manifold::se3::SE3;
    use nalgebra::dvector;

    for edge in &graph.edges_se3 {
        let from_name = format!("x{}", edge.from);
        let to_name = format!("x{}", edge.to);

        // Create SE3 measurement
        let quat = edge.measurement.rotation_quaternion();
        let trans = edge.measurement.translation();
        let se3_data = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];
        let measurement = SE3::from(se3_data);

        let factor = BetweenFactorSE3::new(measurement);
        problem.add_residual_block(&[&from_name, &to_name], Box::new(factor), None);
    }

    println!("Problem setup complete:");
    println!("  Variables: {}", initial_values.len());
    println!("  Residual blocks: {}", graph.edges_se3.len());
    println!();

    // Configuration parameters (same for both)
    let max_iterations = 100;
    let cost_tol = 1e-3;
    let param_tol = 1e-3;

    println!("==============================================");
    println!("  Run 1: WITH Jacobi Scaling (default)");
    println!("==============================================");

    let config_with_scaling = LevenbergMarquardtConfig::new()
        .with_max_iterations(max_iterations)
        .with_cost_tolerance(cost_tol)
        .with_parameter_tolerance(param_tol)
        .with_jacobi_scaling(true) // Enable scaling
        .with_verbose(false);

    let mut solver_with = LevenbergMarquardt::with_config(config_with_scaling).with_damping(1e-3);

    let start_with = Instant::now();
    let result_with = match solver_with.optimize(&problem, &initial_values) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Optimization failed: {}", e);
            std::process::exit(1);
        }
    };
    let elapsed_with = start_with.elapsed();

    println!("Status: {:?}", result_with.status);
    println!("Initial cost: {:.6e}", result_with.initial_cost);
    println!("Final cost: {:.6e}", result_with.final_cost);
    println!("Iterations: {}", result_with.iterations);
    println!("Time: {:.3}s", elapsed_with.as_secs_f64());
    if let Some(conv) = &result_with.convergence_info {
        println!("Final gradient norm: {:.6e}", conv.final_gradient_norm);
        println!(
            "Final parameter update: {:.6e}",
            conv.final_parameter_update_norm
        );
    }
    println!();

    println!("==============================================");
    println!("  Run 2: WITHOUT Jacobi Scaling");
    println!("==============================================");

    let config_without_scaling = LevenbergMarquardtConfig::new()
        .with_max_iterations(max_iterations)
        .with_cost_tolerance(cost_tol)
        .with_parameter_tolerance(param_tol)
        .with_jacobi_scaling(false) // Disable scaling
        .with_verbose(false);

    let mut solver_without =
        LevenbergMarquardt::with_config(config_without_scaling).with_damping(1e-3);

    let start_without = Instant::now();
    let result_without = match solver_without.optimize(&problem, &initial_values) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Optimization failed: {}", e);
            std::process::exit(1);
        }
    };
    let elapsed_without = start_without.elapsed();

    println!("Status: {:?}", result_without.status);
    println!("Initial cost: {:.6e}", result_without.initial_cost);
    println!("Final cost: {:.6e}", result_without.final_cost);
    println!("Iterations: {}", result_without.iterations);
    println!("Time: {:.3}s", elapsed_without.as_secs_f64());
    if let Some(conv) = &result_without.convergence_info {
        println!("Final gradient norm: {:.6e}", conv.final_gradient_norm);
        println!(
            "Final parameter update: {:.6e}",
            conv.final_parameter_update_norm
        );
    }
    println!();

    println!("==============================================");
    println!("  Comparison Summary");
    println!("==============================================");
    println!("                           WITH Scaling  |  WITHOUT Scaling  |  Difference");
    println!("-----------------------------------------------------------------------------");
    println!(
        "Time:                      {:.3}s         |  {:.3}s           |  {:+.3}s ({:+.1}%)",
        elapsed_with.as_secs_f64(),
        elapsed_without.as_secs_f64(),
        elapsed_without.as_secs_f64() - elapsed_with.as_secs_f64(),
        ((elapsed_without.as_secs_f64() - elapsed_with.as_secs_f64()) / elapsed_with.as_secs_f64())
            * 100.0
    );
    println!(
        "Iterations:                {}              |  {}                |  {:+}",
        result_with.iterations,
        result_without.iterations,
        result_without.iterations as i32 - result_with.iterations as i32
    );
    println!(
        "Final cost:                {:.6e}    |  {:.6e}      |  {:.6e}",
        result_with.final_cost,
        result_without.final_cost,
        result_without.final_cost - result_with.final_cost
    );
    println!(
        "Time per iteration:        {:.3}ms        |  {:.3}ms          |  {:+.3}ms ({:+.1}%)",
        elapsed_with.as_secs_f64() * 1000.0 / result_with.iterations as f64,
        elapsed_without.as_secs_f64() * 1000.0 / result_without.iterations as f64,
        (elapsed_without.as_secs_f64() * 1000.0 / result_without.iterations as f64)
            - (elapsed_with.as_secs_f64() * 1000.0 / result_with.iterations as f64),
        (((elapsed_without.as_secs_f64() / result_without.iterations as f64)
            - (elapsed_with.as_secs_f64() / result_with.iterations as f64))
            / (elapsed_with.as_secs_f64() / result_with.iterations as f64))
            * 100.0
    );
    println!("==============================================");
    println!();

    // Verdict
    if elapsed_without.as_secs_f64() < elapsed_with.as_secs_f64() {
        let speedup = (elapsed_with.as_secs_f64() - elapsed_without.as_secs_f64())
            / elapsed_with.as_secs_f64()
            * 100.0;
        println!("✓ WITHOUT scaling is FASTER by {:.1}%", speedup);
    } else {
        let slowdown = (elapsed_without.as_secs_f64() - elapsed_with.as_secs_f64())
            / elapsed_with.as_secs_f64()
            * 100.0;
        println!("✓ WITH scaling is FASTER by {:.1}%", -slowdown);
    }

    // Check convergence quality
    let cost_diff = (result_without.final_cost - result_with.final_cost).abs();
    let cost_ratio = cost_diff / result_with.final_cost.min(result_without.final_cost);
    if cost_ratio < 0.01 {
        println!("✓ Both methods converge to nearly identical solutions (< 1% difference)");
    } else {
        println!(
            "⚠ Solutions differ by {:.2}% - convergence quality may differ",
            cost_ratio * 100.0
        );
    }
}
