//! Bundle Adjustment Binary
//!
//! Loads BAL (Bundle Adjustment in the Large) files and optimizes them
//! using Levenberg-Marquardt optimization with Schur complement.
//!
//! # Usage
//! ```bash
//! cargo run --release --bin bundle_adjustment -- path/to/problem.txt
//!
//! # With limited points for quick testing:
//! cargo run --release --bin bundle_adjustment -- problem.txt -n 1000
//! ```
//!
//! This binary uses the same setup as benches/bundle_adjustment_comparison.rs
//! for accurate profiling and benchmarking.

use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::BALCameraFactor;
use apex_solver::init_logger;
use apex_solver::io::{BalDataset, BalLoader};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use clap::Parser;
use nalgebra::DVector;
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

/// Bundle adjustment optimization for BAL datasets
#[derive(Parser)]
#[command(name = "bundle_adjustment")]
#[command(about = "Bundle adjustment optimization for BAL datasets")]
struct Args {
    /// BAL file path (required, positional)
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// Limit number of points (for testing)
    #[arg(short = 'n', long)]
    num_points: Option<usize>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Initialize logger
    init_logger();

    info!("APEX-SOLVER BUNDLE ADJUSTMENT");
    info!("");

    // Validate file exists
    if !args.file.exists() {
        return Err(format!("File not found: {}", args.file.display()).into());
    }

    // Load BAL dataset
    info!("Loading BAL dataset: {}", args.file.display());
    let start_load = Instant::now();
    let dataset = BalLoader::load(args.file.to_string_lossy().as_ref())?;
    let load_time = start_load.elapsed();

    let num_points_to_use = args.num_points.unwrap_or(dataset.points.len());
    let num_points_to_use = num_points_to_use.min(dataset.points.len());

    info!("Dataset statistics:");
    info!("  Cameras: {}", dataset.cameras.len());
    info!("  Total points: {}", dataset.points.len());
    info!("  Points to use: {}", num_points_to_use);
    info!("  Observations: {}", dataset.observations.len());
    info!("  Load time: {:?}", load_time);
    info!("");

    // Run bundle adjustment (matching benches/bundle_adjustment_comparison.rs exactly)
    run_bundle_adjustment(&dataset, num_points_to_use, args.verbose)
}

/// Run bundle adjustment (matches benches/bundle_adjustment_comparison.rs exactly)
///
/// This uses:
/// - RN manifold with 9-param BALCameraFactor (matching Ceres BAL format)
/// - LevenbergMarquardtConfig::for_bundle_adjustment() preset
/// - Huber loss with scale=1.0
/// - Fixes first camera (all 9 DOF)
fn run_bundle_adjustment(
    dataset: &BalDataset,
    num_points: usize,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add cameras as RN (9-param) variables matching BAL format
    info!(
        "Adding {} cameras as RN(9) variables...",
        dataset.cameras.len()
    );
    for (i, cam) in dataset.cameras.iter().enumerate() {
        let var_name = format!("cam_{:04}", i);
        let camera_vec = DVector::from_vec(vec![
            cam.rotation.x,
            cam.rotation.y,
            cam.rotation.z,
            cam.translation.x,
            cam.translation.y,
            cam.translation.z,
            cam.focal_length,
            cam.k1,
            cam.k2,
        ]);
        initial_values.insert(var_name, (ManifoldType::RN, camera_vec));
    }

    // Add landmarks as R3 variables
    info!("Adding {} landmarks as RN(3) variables...", num_points);
    for j in 0..num_points {
        let point = &dataset.points[j];
        let var_name = format!("pt_{:05}", j);
        let point_vec =
            DVector::from_vec(vec![point.position.x, point.position.y, point.position.z]);
        initial_values.insert(var_name, (ManifoldType::RN, point_vec));
    }

    // Count observations for points we're using
    let obs_count: usize = dataset
        .observations
        .iter()
        .filter(|obs| obs.point_index < num_points)
        .count();

    // Add projection factors using BALCameraFactor (9-param camera)
    info!(
        "Adding {} projection factors (9-param BALCameraFactor)...",
        obs_count
    );
    for obs in &dataset.observations {
        if obs.point_index >= num_points {
            continue;
        }

        let factor = BALCameraFactor::new(obs.x, obs.y);
        let cam_name = format!("cam_{:04}", obs.camera_index);
        let pt_name = format!("pt_{:05}", obs.point_index);

        // Huber loss with scale=1.0 (matching benchmark)
        let loss = Box::new(HuberLoss::new(1.0).expect("Failed to create Huber loss"));
        problem.add_residual_block(&[&cam_name, &pt_name], Box::new(factor), Some(loss));
    }

    // Fix first camera (gauge freedom) - all 9 DOF
    info!("Fixing first camera (all 9 DOF) for gauge freedom...");
    for dof in 0..9 {
        problem.fix_variable("cam_0000", dof);
    }

    // Configure solver using BA-optimized preset (matches Ceres settings)
    let config = LevenbergMarquardtConfig::for_bundle_adjustment();

    info!("");
    info!("Solver configuration (for_bundle_adjustment preset):");
    info!("  Linear solver: {:?}", config.linear_solver_type);
    info!("  Schur variant: {:?}", config.schur_variant);
    info!("  Preconditioner: {:?}", config.schur_preconditioner);
    info!("  Initial damping: {:e}", config.damping);
    info!("  Max iterations: {}", config.max_iterations);
    info!("  Cost tolerance: {:e}", config.cost_tolerance);
    info!("  Parameter tolerance: {:e}", config.parameter_tolerance);

    let mut solver = LevenbergMarquardt::with_config(config);

    // Optimize (timing excludes setup)
    info!("");
    info!("Starting optimization...");
    let start = Instant::now();
    let result = solver.optimize(&problem, &initial_values)?;
    let elapsed = start.elapsed();

    info!("");
    info!("Optimization completed!");
    info!("Status: {:?}", result.status);
    info!("Iterations: {}", result.iterations);
    info!("Time: {:.2} seconds", elapsed.as_secs_f64());

    // Compute RMSE from solver costs
    let num_obs = obs_count as f64;
    let initial_mse = result.initial_cost / num_obs;
    let initial_rmse = initial_mse.sqrt();
    let final_mse = result.final_cost / num_obs;
    let final_rmse = final_mse.sqrt();

    info!("");
    info!("Metrics:");
    info!("  Initial cost: {:.6e}", result.initial_cost);
    info!("  Final cost: {:.6e}", result.final_cost);
    info!("  Initial RMSE: {:.3} pixels", initial_rmse);
    info!("  Final RMSE: {:.3} pixels", final_rmse);
    info!(
        "  Improvement: {:.2}%",
        (result.initial_cost - result.final_cost) / result.initial_cost * 100.0
    );

    if verbose {
        info!("");
        info!("Detailed timing breakdown:");
        info!("  Total time: {:.2}s", elapsed.as_secs_f64());
        info!(
            "  Per-iteration: {:.2}s",
            elapsed.as_secs_f64() / result.iterations as f64
        );
    }

    Ok(())
}
