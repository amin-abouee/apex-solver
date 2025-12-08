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
use nalgebra::{DVector, Vector2};
use std::collections::HashMap;
use std::error::Error;
use std::time::{Duration, Instant};

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
    let args = Args::parse();

    // Construct BAL dataset path
    let path = match args.dataset {
        21 => "data/bundle_adjustment/problem-21-11315-pre.txt",
        49 => "data/bundle_adjustment/problem-49-7776-pre.txt",
        89 => "data/bundle_adjustment/problem-89-110973-pre.txt",
        _ => {
            eprintln!("Invalid dataset size. Choose 21, 49, or 89.");
            std::process::exit(1);
        }
    };

    println!("═══════════════════════════════════════════════════════════");
    println!("  Bundle Adjustment with Pinhole Camera Model");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Load BAL dataset
    println!("Loading BAL dataset: {}", path);
    let start_load = Instant::now();
    let dataset = BalLoader::load(path)?;
    let load_time = start_load.elapsed();

    let num_points_to_use = args.num_points.unwrap_or(dataset.points.len());
    let num_points_to_use = num_points_to_use.min(dataset.points.len());

    println!("  Cameras:         {}", dataset.cameras.len());
    println!("  Total points:    {}", dataset.points.len());
    println!("  Points to use:   {}", num_points_to_use);
    println!("  Observations:    {}", dataset.observations.len());
    println!("  Load time:       {:?}", load_time);
    println!();

    if args.compare_all {
        println!("Comparing all linear solvers...");
        println!();

        let mut all_metrics = Vec::new();

        // Test SparseCholesky
        println!("─────────────────────────────────────────────────────────");
        println!("  Solver: SPARSE CHOLESKY");
        println!("─────────────────────────────────────────────────────────");
        match run_bundle_adjustment("cholesky", &dataset, num_points_to_use, args.max_iterations) {
            Ok(metrics) => {
                print_metrics(&metrics);
                all_metrics.push(metrics);
            }
            Err(e) => {
                eprintln!("  ERROR: {}", e);
            }
        }
        println!();

        // Test Schur Complement (Sparse)
        println!("─────────────────────────────────────────────────────────");
        println!("  Solver: SCHUR COMPLEMENT (SPARSE)");
        println!("─────────────────────────────────────────────────────────");
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
                eprintln!("  ERROR: {}", e);
            }
        }
        println!();

        // Test Schur Complement (Iterative/PCG)
        println!("─────────────────────────────────────────────────────────");
        println!("  Solver: SCHUR COMPLEMENT (ITERATIVE PCG)");
        println!("─────────────────────────────────────────────────────────");
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
                eprintln!("  ERROR: {}", e);
            }
        }
        println!();

        // Test Schur Complement (Power Series)
        println!("─────────────────────────────────────────────────────────");
        println!("  Solver: SCHUR COMPLEMENT (POWER SERIES)");
        println!("─────────────────────────────────────────────────────────");
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
                eprintln!("  ERROR: {}", e);
            }
        }
        println!();

        // Print comparison table
        if !all_metrics.is_empty() {
            print_comparison_table(&all_metrics);
        }
    } else {
        // Run with selected solver
        println!("─────────────────────────────────────────────────────────");
        println!("  Solver: {}", args.solver.to_uppercase());
        println!("─────────────────────────────────────────────────────────");
        let metrics = run_bundle_adjustment(
            &args.solver,
            &dataset,
            num_points_to_use,
            args.max_iterations,
        )?;
        print_metrics(&metrics);
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  Complete!");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}

fn setup_bal_problem(
    dataset: &BalDataset,
    num_points: usize,
) -> (Problem, HashMap<String, (ManifoldType, DVector<f64>)>) {
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    let num_points = num_points.min(dataset.points.len());

    // Add camera poses as variables (RN with 6 DOF: axis-angle + translation)
    // Use zero-padded names for proper lexicographic sorting
    for i in 0..dataset.cameras.len() {
        let cam = &dataset.cameras[i];
        let pose = DVector::from_vec(vec![
            cam.rotation[0],    // rx
            cam.rotation[1],    // ry
            cam.rotation[2],    // rz
            cam.translation[0], // tx
            cam.translation[1], // ty
            cam.translation[2], // tz
        ]);
        initial_values.insert(format!("cam_{:04}", i), (ManifoldType::RN, pose));
    }

    // Add 3D world points as variables (RN with 3 DOF)
    // Use zero-padded names for proper lexicographic sorting
    for j in 0..num_points {
        let point = &dataset.points[j];
        let point_vec = DVector::from_vec(vec![
            point.position[0],
            point.position[1],
            point.position[2],
        ]);
        initial_values.insert(format!("pt_{:05}", j), (ManifoldType::RN, point_vec));
    }

    // Add BundleAdjustmentFactors (one per observation)
    // Each factor connects [camera_i, point_j]
    let mut factor_count = 0;
    for obs in &dataset.observations {
        if obs.point_index >= num_points {
            continue;
        }

        let cam = &dataset.cameras[obs.camera_index];

        // Camera intrinsics (fixed during optimization)
        let camera_intrinsics = DVector::from_vec(vec![
            cam.focal_length,
            cam.focal_length,
            0.0, // cx (assume principal point at origin)
            0.0, // cy
        ]);

        let factor = Box::new(BundleAdjustmentFactor::new(
            Vector2::new(obs.x, obs.y),
            camera_intrinsics,
        ));

        let camera_var = format!("cam_{:04}", obs.camera_index);
        let point_var = format!("pt_{:05}", obs.point_index);

        problem.add_residual_block(&[&camera_var, &point_var], factor, None);
        factor_count += 1;
    }

    println!("Problem setup:");
    println!("  Cameras:                {}", dataset.cameras.len());
    println!("  3D points:              {}", num_points);
    println!("  Observations (factors): {}", factor_count);
    println!(
        "  Total parameters:       {}",
        dataset.cameras.len() * 6 + num_points * 3
    );
    println!();

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

    println!("Starting optimization...");
    println!("  Linear solver:       {}", linear_solver_type);
    if solver_name.starts_with("schur") {
        println!("  Schur variant:       {:?}", schur_variant);
        println!(
            "  Cameras/Landmarks:   {} cameras (6 DOF), {} points (3 DOF)",
            dataset.cameras.len(),
            num_points
        );
    }
    println!("  Max iterations:      {}", max_iterations);
    println!("  Cost tolerance:      {:.0e}", config.cost_tolerance);
    println!("  Parameter tolerance: {:.0e}", config.parameter_tolerance);
    println!();

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
    println!("Optimization Results:");
    println!("  Initial cost:       {:.6e}", metrics.initial_cost);
    println!("  Final cost:         {:.6e}", metrics.final_cost);
    println!(
        "  Cost reduction:     {:.2}%",
        metrics.cost_reduction_percent()
    );
    println!("  Iterations:         {}", metrics.iterations);
    println!("  Total time:         {:?}", metrics.total_time);
    println!("  Time/iteration:     {:?}", metrics.time_per_iteration());
    println!(
        "  Status:             {}",
        if metrics.success {
            "Converged ✓"
        } else {
            "Failed ✗"
        }
    );
}

fn print_comparison_table(all_metrics: &[OptimizationMetrics]) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Solver Comparison");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!(
        "{:<20} {:>15} {:>15} {:>10} {:>12}",
        "Solver", "Init Cost", "Final Cost", "Iters", "Time"
    );
    println!(
        "{:-<20} {:-<15} {:-<15} {:-<10} {:-<12}",
        "", "", "", "", ""
    );

    for metrics in all_metrics {
        println!(
            "{:<20} {:>15.6e} {:>15.6e} {:>10} {:>12.3}s",
            metrics.solver_name,
            metrics.initial_cost,
            metrics.final_cost,
            metrics.iterations,
            metrics.total_time.as_secs_f64()
        );
    }
    println!();
}
