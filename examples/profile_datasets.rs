//! Profiling example for apex-solver dataset optimization
//!
//! This example is designed specifically for profiling with tools like samply or cargo-flamegraph.
//! It runs optimization on SE3 datasets with minimal I/O overhead to focus on computational bottlenecks.
//!
//! Usage with samply (recommended for macOS):
//! ```bash
//! cargo build --profile profiling --example profile_datasets
//! samply record ./target/profiling/examples/profile_datasets sphere2500
//! ```
//!
//! Usage with cargo-flamegraph:
//! ```bash
//! cargo flamegraph --profile profiling --example profile_datasets -- sphere2500
//! ```

use apex_solver::{
    core::problem::Problem,
    factors::BetweenFactorSE3,
    io::{G2oLoader, GraphLoader},
    manifold::ManifoldType,
    optimizer::LevenbergMarquardt,
    optimizer::levenberg_marquardt::LevenbergMarquardtConfig,
};
use nalgebra::dvector;
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use tracing::{Level, info, warn};

fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        warn!("Usage: {} <dataset>", args[0]);
        warn!("Available datasets: rim, sphere2500, parking-garage, torus3D, grid3D, cubicle");
        std::process::exit(1);
    }

    // Initialize tracing with default info level
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(Level::INFO.into())
                .from_env_lossy(),
        )
        .init();

    let dataset = &args[1];

    // Load dataset
    info!("Loading dataset: {}", dataset);
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
    let file_path = data_dir.join(format!("{}.g2o", dataset));

    if !file_path.exists() {
        warn!("Error: Dataset file not found: {:?}", file_path);
        std::process::exit(1);
    }

    // Load graph from file
    let graph = match G2oLoader::load(&file_path) {
        Ok(g) => g,
        Err(e) => {
            warn!("Error loading graph: {}", e);
            std::process::exit(1);
        }
    };

    info!("Loaded graph:");
    info!("  SE3 Vertices: {}", graph.vertices_se3.len());
    info!("  SE3 Edges: {}", graph.edges_se3.len());

    // Build problem
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Get sorted vertex IDs
    let mut vertex_ids: Vec<_> = graph.vertices_se3.keys().copied().collect();
    vertex_ids.sort();

    // Add SE3 vertices as variables
    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se3.get(&id) {
            let var_name = format!("x{}", id);
            let quat = vertex.pose.rotation_quaternion();
            let trans = vertex.pose.translation();
            // Format: [tx, ty, tz, qw, qx, qy, qz]
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

        problem.add_residual_block(
            &[&id0, &id1],
            Box::new(between_factor),
            None, // No loss function
        );
    }

    info!("Problem setup complete:");
    info!("  Variables: {}", initial_values.len());
    info!("  Residual blocks: {}", graph.edges_se3.len());

    // Configure optimizer based on dataset
    let (max_iters, cost_tol, param_tol) = match dataset.as_str() {
        "rim" => (100, 1e-3, 1e-3),
        "sphere2500" => (100, 1e-4, 1e-4),
        "parking-garage" => (100, 1e-4, 1e-4),
        "torus3D" => (100, 1e-5, 1e-5),
        "grid3D" => (30, 1e-1, 1e-1),
        "cubicle" => (100, 1e-4, 1e-4),
        _ => (100, 1e-4, 1e-4),
    };

    // Create optimizer
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(max_iters)
        .with_cost_tolerance(cost_tol)
        .with_parameter_tolerance(param_tol)
        .with_damping(1e-3);

    let mut optimizer = LevenbergMarquardt::with_config(config);

    info!("Starting Optimization (Profiling Mode)");
    info!("Max iterations: {}", max_iters);
    info!("Cost tolerance: {:.2e}", cost_tol);
    info!("Parameter tolerance: {:.2e}", param_tol);

    // Run optimization - THIS IS THE HOT PATH TO PROFILE
    let start = std::time::Instant::now();
    let result = optimizer.optimize(&problem, &initial_values);
    let elapsed = start.elapsed();

    // Print results
    info!("Optimization Complete");
    match result {
        Ok(solver_result) => {
            info!("Status: {:?}", solver_result.status);
            info!("Initial cost: {:.6e}", solver_result.initial_cost);
            info!("Final cost: {:.6e}", solver_result.final_cost);
            info!("Iterations: {}", solver_result.iterations);
            info!("Time: {:.3}s", elapsed.as_secs_f64());

            if let Some(convergence) = solver_result.convergence_info {
                info!(
                    "Final gradient norm: {:.6e}",
                    convergence.final_gradient_norm
                );
                info!(
                    "Final parameter update: {:.6e}",
                    convergence.final_parameter_update_norm
                );
            }
        }
        Err(e) => {
            warn!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
