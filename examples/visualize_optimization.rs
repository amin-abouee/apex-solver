//! Visualize optimization progress in real-time using Rerun.
//!
//! This example demonstrates the graphical debugging capabilities of apex-solver
//! by loading a SLAM dataset and visualizing the optimization process with Rerun.
//!
//! # Features
//!
//! - Real-time time series plots (cost, gradient norm, damping, step quality)
//! - Sparse Hessian matrix visualization as heat map
//! - Gradient vector visualization
//! - Pose updates animated through optimization (for SE2/SE3 problems)
//!
//! # Usage
//!
//! ```bash
//! # Visualize optimization on sphere2500 dataset
//! cargo run --example visualize_optimization
//!
//! # Use a different dataset
//! cargo run --example visualize_optimization -- --dataset parking-garage
//!
//! # Adjust visualization frequency (log every N iterations)
//! cargo run --example visualize_optimization -- --log-frequency 10
//! ```
//!
//! The Rerun viewer will open automatically showing optimization progress.

use apex_solver::core::problem::Problem;
use apex_solver::factors::BetweenFactorSE3;
use apex_solver::io::{G2oLoader, Graph, GraphLoader};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::LevenbergMarquardt;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
use clap::Parser;
use nalgebra::dvector;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Dataset file to load (e.g., "sphere2500", "parking-garage", "rim")
    #[arg(short, long, default_value = "sphere2500")]
    dataset: String,

    /// Maximum number of optimization iterations
    #[arg(short, long, default_value = "100")]
    max_iterations: usize,

    /// Enable verbose terminal output
    #[arg(short, long)]
    verbose: bool,

    /// Cost tolerance for convergence
    #[arg(long, default_value = "1e-4")]
    cost_tolerance: f64,

    /// Optional path to save optimized graph
    #[arg(long)]
    save_output: Option<PathBuf>,

    /// Save visualization to file instead of spawning viewer
    /// The file can be viewed later with: rerun <filename>
    #[arg(long, default_value = "optimization.rrd")]
    save_visualization: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Construct dataset path
    let dataset_path = format!("data/{}.g2o", args.dataset);
    println!("Loading dataset: {}", dataset_path);

    // Load graph from G2O file
    let graph = G2oLoader::load(&dataset_path)?;

    println!("\n=== Dataset Statistics ===");
    println!("SE3 vertices: {}", graph.vertices_se3.len());
    println!("SE3 edges:    {}", graph.edges_se3.len());
    println!("SE2 vertices: {}", graph.vertices_se2.len());
    println!("SE2 edges:    {}", graph.edges_se2.len());

    if graph.edges_se3.is_empty() && graph.edges_se2.is_empty() {
        eprintln!("Error: No edges found in dataset");
        return Ok(());
    }

    // Choose appropriate workflow based on graph content
    if !graph.edges_se3.is_empty() {
        println!("\n=== Running SE3 Optimization with Rerun Visualization ===");
        optimize_se3_graph(&graph, &args)?;
    } else if !graph.edges_se2.is_empty() {
        println!("\n=== Running SE2 Optimization with Rerun Visualization ===");
        println!("Note: SE2 visualization is currently limited to 2D point plots");
        optimize_se2_graph(&graph, &args)?;
    }

    // Keep program running briefly so Rerun viewer can display data
    println!("\n✓ Optimization complete!");
    println!("Keeping connection open for 5 seconds...");
    std::thread::sleep(std::time::Duration::from_secs(5));

    println!("If you saved to optimization.rrd, view it with: rerun optimization.rrd");

    Ok(())
}

/// Optimize SE3 pose graph with visualization
fn optimize_se3_graph(graph: &Graph, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    // Print Rerun connection instructions
    if args.save_visualization.is_some() {
        println!("\n=== Rerun Visualization ===");
        println!(
            "Saving to file: {}",
            args.save_visualization.as_ref().unwrap()
        );
        println!(
            "View it later with: rerun {}",
            args.save_visualization.as_ref().unwrap()
        );
    } else {
        println!("\n=== Rerun Visualization ===");
        println!("Starting Rerun server on port 9876");
        println!("To view live data, run in another terminal:");
        println!("  rerun");
        println!("Or connect to: rerun+http://127.0.0.1:9876/proxy\n");
    }

    // Create optimization problem
    let mut problem = Problem::new();

    // Add all SE3 edges as between factors
    for edge in &graph.edges_se3 {
        let from_key = format!("x{}", edge.from);
        let to_key = format!("x{}", edge.to);

        let factor = BetweenFactorSE3::new(edge.measurement.clone());

        let var_keys: Vec<&str> = vec![from_key.as_str(), to_key.as_str()];
        problem.add_residual_block(&var_keys, Box::new(factor), None);
    }

    // Prepare initial parameters
    let mut initial_params = HashMap::new();
    for (&id, vertex) in &graph.vertices_se3 {
        let var_name = format!("x{}", id);
        let quat = vertex.rotation();
        let trans = vertex.translation();
        // Quaternion order must be [w, x, y, z] (scalar first)
        let pose_vec = dvector![
            trans.x,
            trans.y,
            trans.z,
            quat.as_ref().w,
            quat.as_ref().i,
            quat.as_ref().j,
            quat.as_ref().k,
        ];
        initial_params.insert(var_name, (ManifoldType::SE3, pose_vec));
    }

    // Configure optimizer with Rerun visualization
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(args.max_iterations)
        .with_cost_tolerance(args.cost_tolerance)
        .with_parameter_tolerance(1e-4)
        .with_gradient_tolerance(1e-8)
        .with_verbose(args.verbose)
        .with_visualization(true); // Enable graphical visualization!

    println!("\n=== Optimizer Configuration ===");
    println!("Max iterations:     {}", config.max_iterations);
    println!("Cost tolerance:     {:.2e}", config.cost_tolerance);
    println!("Rerun logging:      enabled");
    println!("Verbose output:     {}", config.verbose);
    println!("\n=== Visualization Features ===");
    println!("✓ Separate time series plots for: cost, gradient_norm, damping, step_quality");
    println!("✓ Hessian matrix: downsampled to 100×100 heat map");
    println!("✓ Gradient vector: downsampled to 100-element bar");
    println!("✓ 3D pose visualization (latest iteration only)");

    // Create and run optimizer
    let mut solver = LevenbergMarquardt::with_config(config);

    println!("\n=== Starting Optimization ===");
    println!("The Rerun viewer should open automatically.");

    let result = solver.optimize(&problem, &initial_params)?;

    // Give Rerun time to flush data
    println!("\nWaiting for Rerun to flush visualization data...");
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Print summary
    println!("\n=== Optimization Results ===");
    println!("Status:          {}", result.status);
    println!("Initial cost:    {:.6e}", result.initial_cost);
    println!("Final cost:      {:.6e}", result.final_cost);
    println!(
        "Improvement:     {:.2}%",
        100.0 * (result.initial_cost - result.final_cost) / result.initial_cost.max(1e-12)
    );
    println!("Iterations:      {}", result.iterations);
    println!("Elapsed time:    {:?}", result.elapsed_time);

    if let Some(conv_info) = &result.convergence_info {
        println!("\nConvergence Info:");
        println!(
            "  Gradient norm:      {:.6e}",
            conv_info.final_gradient_norm
        );
        println!(
            "  Parameter update:   {:.6e}",
            conv_info.final_parameter_update_norm
        );
        println!("  Cost evaluations:   {}", conv_info.cost_evaluations);
        println!("  Jacobian evals:     {}", conv_info.jacobian_evaluations);
    }

    // Save optimized graph if requested
    if let Some(output_path) = &args.save_output {
        println!("\n=== Saving Optimized Graph ===");
        let optimized_graph = Graph::from_optimized_variables(&result.parameters, graph);
        G2oLoader::write(&optimized_graph, output_path)?;
        println!("Saved to: {}", output_path.display());
    }

    Ok(())
}

/// Optimize SE2 pose graph with visualization
fn optimize_se2_graph(graph: &Graph, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use apex_solver::factors::BetweenFactorSE2;

    // Create optimization problem
    let mut problem = Problem::new();

    // Add all SE2 edges as between factors
    for edge in &graph.edges_se2 {
        let from_key = format!("x{}", edge.from);
        let to_key = format!("x{}", edge.to);

        let factor = BetweenFactorSE2::new(
            edge.measurement.x(),
            edge.measurement.y(),
            edge.measurement.angle(),
        );

        let var_keys: Vec<&str> = vec![from_key.as_str(), to_key.as_str()];
        problem.add_residual_block(&var_keys, Box::new(factor), None);
    }

    // Prepare initial parameters
    let mut initial_params = HashMap::new();
    for (&id, vertex) in &graph.vertices_se2 {
        let var_name = format!("x{}", id);
        let pose_vec = dvector![vertex.x(), vertex.y(), vertex.theta()];
        initial_params.insert(var_name, (ManifoldType::SE2, pose_vec));
    }

    // Configure optimizer with Rerun visualization
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(args.max_iterations)
        .with_cost_tolerance(args.cost_tolerance)
        .with_parameter_tolerance(1e-4)
        .with_gradient_tolerance(1e-8)
        .with_verbose(args.verbose)
        .with_visualization(true);

    println!("\n=== Optimizer Configuration ===");
    println!("Max iterations:     {}", config.max_iterations);
    println!("Cost tolerance:     {:.2e}", config.cost_tolerance);
    println!("Rerun logging:      enabled");

    // Create and run optimizer
    let mut solver = LevenbergMarquardt::with_config(config);

    println!("\n=== Starting Optimization ===");

    let result = solver.optimize(&problem, &initial_params)?;

    // Print summary
    println!("\n=== Optimization Results ===");
    println!("Status:          {}", result.status);
    println!("Initial cost:    {:.6e}", result.initial_cost);
    println!("Final cost:      {:.6e}", result.final_cost);
    println!(
        "Improvement:     {:.2}%",
        100.0 * (result.initial_cost - result.final_cost) / result.initial_cost.max(1e-12)
    );
    println!("Iterations:      {}", result.iterations);

    // Save if requested
    if let Some(output_path) = &args.save_output {
        let optimized_graph = Graph::from_optimized_variables(&result.parameters, graph);
        G2oLoader::write(&optimized_graph, output_path)?;
        println!("\nSaved to: {}", output_path.display());
    }

    Ok(())
}
