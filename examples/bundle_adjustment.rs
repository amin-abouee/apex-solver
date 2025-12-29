//! Bundle Adjustment Example with Pinhole Camera Model
//!
//! This example demonstrates bundle adjustment optimization using:
//! - Simple pinhole camera model (no distortion)
//! - Optimizing 3D point positions with fixed camera parameters
//! - Comparison of different linear solvers (SparseCholesky vs Schur variants)
//!
//! The example loads BAL (Bundle Adjustment in the Large) dataset files and optimizes
//! 3D landmark positions to minimize reprojection error.

use apex_solver::core::problem::Problem;
use apex_solver::factors::BundleAdjustmentFactor;
use apex_solver::io::{BalDataset, BalLoader};
use apex_solver::linalg::{LinearSolverType, SchurPreconditioner, SchurVariant};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use clap::Parser;
use nalgebra::{DVector, Vector2, Vector3};
use std::collections::HashMap;
use std::error::Error;
use std::time::{Duration, Instant};
use tracing::{info, warn};

// Removed axis_angle_to_se3 - using RN(6) directly for compatibility with BundleAdjustmentFactor

#[derive(Parser, Debug)]
#[command(name = "bundle_adjustment")]
#[command(about = "Bundle adjustment optimization with pinhole camera model")]
struct Args {
    /// Dataset size to load (21, 49, or 89)
    #[arg(short, long, default_value = "21")]
    dataset: u32,

    /// Linear solver: cholesky, schur_sparse, schur_iterative, schur_power_series
    #[arg(short, long, default_value = "cholesky")]
    solver: String,

    /// Maximum number of iterations
    #[arg(short, long, default_value = "20")]
    max_iterations: usize,

    /// Maximum number of points to use (for faster testing)
    #[arg(short, long)]
    num_points: Option<usize>,

    /// Compare all solvers
    #[arg(short, long)]
    compare_all: bool,
}

struct OptimizationMetrics {
    solver_name: String,
    initial_cost: f64,
    final_cost: f64,
    iterations: usize,
    total_time: Duration,
    success: bool,
}

impl OptimizationMetrics {
    fn cost_reduction_percent(&self) -> f64 {
        if self.initial_cost > 0.0 {
            (self.initial_cost - self.final_cost) / self.initial_cost * 100.0
        } else {
            0.0
        }
    }

    fn time_per_iteration(&self) -> Duration {
        if self.iterations > 0 {
            self.total_time / self.iterations as u32
        } else {
            Duration::ZERO
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize tracing subscriber for logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();

    // Construct BAL dataset path
    let path = match args.dataset {
        21 => "data/bundle_adjustment/problem-21-11315-pre.txt",
        49 => "data/bundle_adjustment/problem-49-7776-pre.txt",
        89 => "data/bundle_adjustment/problem-89-110973-pre.txt",
        _ => {
            warn!("Invalid dataset size. Choose 21, 49, or 89.");
            std::process::exit(1);
        }
    };

    info!("Bundle Adjustment with Pinhole Camera Model");
    info!("");

    // Load BAL dataset
    info!("Loading BAL dataset: {}", path);
    let start_load = Instant::now();
    let dataset = BalLoader::load(path)?;
    let load_time = start_load.elapsed();

    let num_points_to_use = args.num_points.unwrap_or(dataset.points.len());
    let num_points_to_use = num_points_to_use.min(dataset.points.len());

    info!("Cameras: {}", dataset.cameras.len());
    info!("Total points: {}", dataset.points.len());
    info!("Points to use: {}", num_points_to_use);
    info!("Observations: {}", dataset.observations.len());
    info!("Load time: {:?}", load_time);
    info!("");

    if args.compare_all {
        info!("Comparing all linear solvers");
        info!("");

        let mut all_metrics = Vec::new();

        // Test SparseCholesky
        info!("Solver: SPARSE CHOLESKY");
        match run_bundle_adjustment("cholesky", &dataset, num_points_to_use, args.max_iterations) {
            Ok(metrics) => {
                print_metrics(&metrics);
                all_metrics.push(metrics);
            }
            Err(e) => {
                warn!("ERROR: {}", e);
            }
        }
        info!("");

        // Test Schur Complement (Sparse)
        info!("Solver: SCHUR COMPLEMENT (SPARSE)");
        match run_bundle_adjustment(
            "schur_sparse",
            &dataset,
            num_points_to_use,
            args.max_iterations,
        ) {
            Ok(metrics) => {
                print_metrics(&metrics);
                all_metrics.push(metrics);
            }
            Err(e) => {
                warn!("ERROR: {}", e);
            }
        }
        info!("");

        // Test Schur Complement (Iterative/PCG)
        info!("Solver: SCHUR COMPLEMENT (ITERATIVE PCG)");
        match run_bundle_adjustment(
            "schur_iterative",
            &dataset,
            num_points_to_use,
            args.max_iterations,
        ) {
            Ok(metrics) => {
                print_metrics(&metrics);
                all_metrics.push(metrics);
            }
            Err(e) => {
                warn!("ERROR: {}", e);
            }
        }
        info!("");

        // Test Schur Complement (Power Series)
        info!("Solver: SCHUR COMPLEMENT (POWER SERIES)");
        match run_bundle_adjustment(
            "schur_power_series",
            &dataset,
            num_points_to_use,
            args.max_iterations,
        ) {
            Ok(metrics) => {
                print_metrics(&metrics);
                all_metrics.push(metrics);
            }
            Err(e) => {
                warn!("ERROR: {}", e);
            }
        }
        info!("");

        // Print comparison table
        if !all_metrics.is_empty() {
            print_comparison_table(&all_metrics);
        }
    } else {
        // Run with selected solver
        info!("Solver: {}", args.solver.to_uppercase());
        let metrics = run_bundle_adjustment(
            &args.solver,
            &dataset,
            num_points_to_use,
            args.max_iterations,
        )?;
        print_metrics(&metrics);
    }

    info!("Complete!");

    Ok(())
}

fn setup_bal_problem(
    dataset: &BalDataset,
    num_points: usize,
) -> (Problem, HashMap<String, (ManifoldType, DVector<f64>)>) {
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    let num_points = num_points.min(dataset.points.len());

    // Add camera poses as variables (RN with 6 DOF: axis-angle rotation + translation)
    // Use zero-padded names for proper lexicographic sorting
    for i in 0..dataset.cameras.len() {
        let cam = &dataset.cameras[i];
        let pose_vec = DVector::from_vec(vec![
            cam.rotation[0],    // rx (axis-angle)
            cam.rotation[1],    // ry
            cam.rotation[2],    // rz
            cam.translation[0], // tx
            cam.translation[1], // ty
            cam.translation[2], // tz
        ]);
        initial_values.insert(format!("cam_{:04}", i), (ManifoldType::RN, pose_vec));
    }

    // First pass: collect which points actually have observations
    let mut observed_points = std::collections::HashSet::new();
    for obs in &dataset.observations {
        if obs.point_index < num_points {
            observed_points.insert(obs.point_index);
        }
    }

    // Add only observed 3D world points as variables (RN with 3 DOF)
    // Use zero-padded names for proper lexicographic sorting
    for &j in &observed_points {
        let point = &dataset.points[j];
        let point_vec = DVector::from_vec(vec![
            point.position[0],
            point.position[1],
            point.position[2],
        ]);
        initial_values.insert(format!("pt_{:05}", j), (ManifoldType::RN, point_vec));
    }

    // Add ProjectionFactors (one per observation)
    // Each factor connects [camera_i, point_j]
    let mut factor_count = 0;
    for obs in &dataset.observations {
        if obs.point_index >= num_points {
            continue;
        }

        let cam = &dataset.cameras[obs.camera_index];

        // Create camera intrinsics vector [fx, fy, cx, cy]
        let camera_intrinsics = DVector::from_vec(vec![
            cam.focal_length, // fx
            cam.focal_length, // fy (same as fx for BAL)
            0.0,              // cx (principal point at origin)
            0.0,              // cy
        ]);

        // Create BundleAdjustmentFactor for this observation
        // Connects one camera pose to one 3D landmark
        let factor = Box::new(BundleAdjustmentFactor::new(
            Vector2::new(obs.x, obs.y), // observed pixel coordinates
            camera_intrinsics,
        ));

        let camera_var = format!("cam_{:04}", obs.camera_index);
        let point_var = format!("pt_{:05}", obs.point_index);

        problem.add_residual_block(&[&camera_var, &point_var], factor, None);
        factor_count += 1;
    }

    // Fix first camera to remove gauge freedom (prevents singular matrix)
    for dof_idx in 0..6 {
        problem.fix_variable("cam_0000", dof_idx);
    }

    info!("Problem setup:");
    info!("Cameras: {}", dataset.cameras.len());
    info!("3D points: {} (observed)", observed_points.len());
    info!("Observations (factors): {}", factor_count);
    info!("Fixed cameras: 1 (cam_0000 - gauge fixing)");
    info!(
        "Total parameters: {}",
        dataset.cameras.len() * 6 + observed_points.len() * 3
    );
    info!("");

    (problem, initial_values)
}

fn run_bundle_adjustment(
    solver_name: &str,
    dataset: &BalDataset,
    num_points: usize,
    max_iterations: usize,
) -> Result<OptimizationMetrics, Box<dyn Error>> {
    // Setup problem
    let (problem, initial_values) = setup_bal_problem(dataset, num_points);

    // Determine linear solver type and Schur variant
    let (linear_solver_type, schur_variant) = match solver_name {
        "cholesky" => (LinearSolverType::SparseCholesky, SchurVariant::Sparse),
        "schur_sparse" => (
            LinearSolverType::SparseSchurComplement,
            SchurVariant::Sparse,
        ),
        "schur_iterative" => (
            LinearSolverType::SparseSchurComplement,
            SchurVariant::Iterative,
        ),
        "schur_power_series" => (
            LinearSolverType::SparseSchurComplement,
            SchurVariant::PowerSeries,
        ),
        _ => (LinearSolverType::SparseCholesky, SchurVariant::Sparse),
    };

    // Create LM configuration
    let mut config = LevenbergMarquardtConfig::new()
        .with_linear_solver_type(linear_solver_type)
        .with_max_iterations(max_iterations)
        .with_cost_tolerance(1e-6)
        .with_parameter_tolerance(1e-8)
        .with_damping(1e-3);

    // For Schur complement solvers, configure the variant and preconditioner
    if solver_name.starts_with("schur") {
        config = config
            .with_schur_variant(schur_variant)
            .with_schur_preconditioner(SchurPreconditioner::BlockDiagonal);
    }

    info!("Starting optimization");
    info!("Linear solver: {}", linear_solver_type);
    if solver_name.starts_with("schur") {
        info!("Schur variant: {:?}", schur_variant);
        info!(
            "Cameras/Landmarks: {} cameras (SE3, 6 DOF), {} points (3 DOF)",
            dataset.cameras.len(),
            num_points
        );
    }
    info!("Max iterations: {}", max_iterations);
    info!("Cost tolerance: {:.0e}", config.cost_tolerance);
    info!("Parameter tolerance: {:.0e}", config.parameter_tolerance);
    info!("");

    // Run optimization with timing
    let start = Instant::now();
    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&problem, &initial_values)?;
    let elapsed = start.elapsed();

    Ok(OptimizationMetrics {
        solver_name: solver_name.to_string(),
        initial_cost: result.initial_cost,
        final_cost: result.final_cost,
        iterations: result.iterations,
        total_time: elapsed,
        success: matches!(result.status, OptimizationStatus::Converged),
    })
}

fn print_metrics(metrics: &OptimizationMetrics) {
    info!("Optimization Results:");
    info!("Initial cost: {:.6e}", metrics.initial_cost);
    info!("Final cost: {:.6e}", metrics.final_cost);
    info!("Cost reduction: {:.2}%", metrics.cost_reduction_percent());
    info!("Iterations: {}", metrics.iterations);
    info!("Total time: {:?}", metrics.total_time);
    info!("Time/iteration: {:?}", metrics.time_per_iteration());
    info!(
        "Status: {}",
        if metrics.success {
            "Converged"
        } else {
            "Failed"
        }
    );
}

fn print_comparison_table(all_metrics: &[OptimizationMetrics]) {
    info!("Solver Comparison - Accuracy and Performance");
    info!("");

    // Header
    info!(
        "{:<25} {:>15} {:>15} {:>12} {:>12} {:>10}",
        "Solver", "Initial Cost", "Final Cost", "Reduction%", "Time(s)", "Iters"
    );

    // Data rows
    for metrics in all_metrics {
        let reduction = metrics.cost_reduction_percent();
        let time_s = metrics.total_time.as_secs_f64();

        info!(
            "{:<25} {:>15.6e} {:>15.6e} {:>11.2}% {:>11.3}s {:>10}",
            metrics.solver_name,
            metrics.initial_cost,
            metrics.final_cost,
            reduction,
            time_s,
            metrics.iterations
        );
    }
    info!("");

    // Find best solver by each metric (with safe comparisons)
    if let Some(fastest) = all_metrics.iter().min_by_key(|m| m.total_time) {
        info!(
            "Fastest: {} ({:.3}s)",
            fastest.solver_name,
            fastest.total_time.as_secs_f64()
        );
    }

    if let Some(best_cost) = all_metrics.iter().filter(|m| m.success).min_by(|a, b| {
        a.final_cost
            .partial_cmp(&b.final_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        info!(
            "Best Final Cost: {} ({:.6e})",
            best_cost.solver_name, best_cost.final_cost
        );
    }

    if let Some(best_reduction) = all_metrics.iter().filter(|m| m.success).max_by(|a, b| {
        a.cost_reduction_percent()
            .partial_cmp(&b.cost_reduction_percent())
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        info!(
            "Best Reduction: {} ({:.2}%)",
            best_reduction.solver_name,
            best_reduction.cost_reduction_percent()
        );
    }
}
