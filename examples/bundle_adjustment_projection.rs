//! Bundle Adjustment Example with BAL Pinhole Camera Model (using ProjectionFactor)
//!
//! This example demonstrates bundle adjustment optimization using:
//! - BAL pinhole camera model with radial distortion
//! - Optimizing both camera poses (SE3) and 3D point positions
//! - Using generic ProjectionFactor (Camera-to-World convention)
//!
//! The example loads BAL (Bundle Adjustment in the Large) dataset files and optimizes
//! camera poses and 3D landmark positions to minimize reprojection error.

use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{
    ProjectionFactor,
    camera::{BALPinholeCamera, BundleAdjustment},
};
use apex_solver::io::{BalDataset, BalLoader};
use apex_solver::linalg::{LinearSolverType, SchurPreconditioner, SchurVariant};
use apex_solver::manifold::LieGroup;
use apex_solver::manifold::ManifoldType;
use apex_solver::manifold::se3::SE3;
use apex_solver::manifold::so3::SO3;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use clap::Parser;
use nalgebra::{DVector, Matrix2xX, Vector3};
use std::collections::HashMap;
use std::error::Error;
use std::time::{Duration, Instant};
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "bundle_adjustment_projection")]
#[command(about = "Bundle adjustment with ProjectionFactor")]
struct Args {
    /// Dataset size to load (21, 49, or 89)
    #[arg(short, long, default_value = "21")]
    dataset: u32,

    /// Linear solver: cholesky, schur_sparse, schur_iterative, schur_power_series
    #[arg(short, long, default_value = "cholesky")]
    solver: String,

    /// Maximum number of iterations
    #[arg(short, long, default_value = "100")]
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
    num_observations: usize,
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

    fn rms_reprojection_error(&self) -> f64 {
        // Cost = sum of squared residuals, RMS = sqrt(cost / num_observations)
        // But residual dimension is 2 per observation (u, v)
        if self.num_observations > 0 {
            (self.final_cost / self.num_observations as f64).sqrt()
        } else {
            0.0
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

    info!("Bundle Adjustment with BAL Pinhole Camera Model (ProjectionFactor)");
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

struct ProblemSetup {
    problem: Problem,
    initial_values: HashMap<String, (ManifoldType, DVector<f64>)>,
    num_observations: usize,
}

fn setup_bal_problem(
    dataset: &BalDataset,
    num_points: usize,
) -> Result<ProblemSetup, Box<dyn Error>> {
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    let num_points = num_points.min(dataset.points.len());

    // Add camera poses as SE3 variables
    // BAL convention (from https://grail.cs.washington.edu/projects/bal/):
    // The rotation R and translation t stored in BAL files represent a WORLD-TO-CAMERA
    // transformation: P_cam = R * X_world + t
    //
    // ProjectionFactor uses world-to-camera convention (matching ReprojectionFactor):
    // It computes P_cam = pose.act(X_world) = R * X_world + t
    //
    // So we store the BAL pose directly as world-to-camera (no inversion needed).
    for i in 0..dataset.cameras.len() {
        let cam = &dataset.cameras[i];

        let axis_angle = Vector3::new(cam.rotation[0], cam.rotation[1], cam.rotation[2]);
        let t_bal = Vector3::new(cam.translation[0], cam.translation[1], cam.translation[2]);
        let r_bal = SO3::from_scaled_axis(axis_angle);

        // BAL gives World-to-Camera (T_wc): P_cam = R * X_world + t
        // Store directly as world-to-camera (no inversion!)
        let se3_wc = SE3::from_translation_so3(t_bal, r_bal);

        // Store as SE3 variable
        let pose_vec: DVector<f64> = se3_wc.into();
        initial_values.insert(format!("cam_{:04}", i), (ManifoldType::SE3, pose_vec));
    }

    // Add landmarks as RN variables (optimize both poses and landmarks)
    for j in 0..num_points {
        let point = &dataset.points[j];
        let point_vec = DVector::from_vec(vec![
            point.position[0],
            point.position[1],
            point.position[2],
        ]);
        initial_values.insert(format!("pt_{:05}", j), (ManifoldType::RN, point_vec));
    }

    // Debug: Compute initial reprojection error with correct convention
    let mut total_error_sq = 0.0;
    let mut valid_count = 0;
    let mut max_error = 0.0_f64;
    let mut errors = Vec::new();

    // Also test via SE3 to verify the factor will work
    let mut se3_valid_count = 0;
    let mut se3_total_error_sq = 0.0;

    for obs in &dataset.observations {
        if obs.point_index >= num_points {
            continue;
        }
        let cam = &dataset.cameras[obs.camera_index];
        let pt = &dataset.points[obs.point_index];

        let axis_angle = Vector3::new(cam.rotation[0], cam.rotation[1], cam.rotation[2]);
        let t_bal = Vector3::new(cam.translation[0], cam.translation[1], cam.translation[2]);
        let r_bal = SO3::from_scaled_axis(axis_angle);
        let p_world = Vector3::new(pt.position[0], pt.position[1], pt.position[2]);

        // BAL convention (official): p_cam = R * p_world + t (world-to-camera)
        let p_cam = r_bal.rotation_matrix() * p_world + t_bal;

        // Test via SE3 (how the factor will compute it)
        let se3 = SE3::from_translation_so3(t_bal, r_bal);
        let p_cam_se3 = se3.act(&p_world, None, None);

        // Check if point is behind camera (BAL uses -Z forward)
        if p_cam.z >= 0.0 {
            continue;
        }

        // Project with BAL pinhole model (negative Z forward)
        let inv_neg_z = -1.0 / p_cam.z;
        let x_n = p_cam.x * inv_neg_z;
        let y_n = p_cam.y * inv_neg_z;

        // Radial distortion
        let r_sq = x_n * x_n + y_n * y_n;
        let distortion = 1.0 + cam.k1 * r_sq + cam.k2 * r_sq * r_sq;
        let x_d = x_n * distortion;
        let y_d = y_n * distortion;

        // Pixel coordinates
        let u = cam.focal_length * x_d;
        let v = cam.focal_length * y_d;

        let error_x = u - obs.x;
        let error_y = v - obs.y;
        let error = (error_x * error_x + error_y * error_y).sqrt();

        total_error_sq += error_x * error_x + error_y * error_y;
        max_error = max_error.max(error);
        valid_count += 1;
        if errors.len() < 10 {
            errors.push(format!("{:.1}", error));
        }

        // Check SE3 version
        if p_cam_se3.z < 0.0 {
            let inv_neg_z_se3 = -1.0 / p_cam_se3.z;
            let x_n_se3 = p_cam_se3.x * inv_neg_z_se3;
            let y_n_se3 = p_cam_se3.y * inv_neg_z_se3;
            let r_sq_se3 = x_n_se3 * x_n_se3 + y_n_se3 * y_n_se3;
            let distortion_se3 = 1.0 + cam.k1 * r_sq_se3 + cam.k2 * r_sq_se3 * r_sq_se3;
            let u_se3 = cam.focal_length * x_n_se3 * distortion_se3;
            let v_se3 = cam.focal_length * y_n_se3 * distortion_se3;
            let err_x_se3 = u_se3 - obs.x;
            let err_y_se3 = v_se3 - obs.y;
            se3_total_error_sq += err_x_se3 * err_x_se3 + err_y_se3 * err_y_se3;
            se3_valid_count += 1;
        }
    }

    let rms_error = if valid_count > 0 {
        (total_error_sq / valid_count as f64).sqrt()
    } else {
        0.0
    };
    let se3_rms = if se3_valid_count > 0 {
        (se3_total_error_sq / se3_valid_count as f64).sqrt()
    } else {
        0.0
    };
    info!(
        "Initial RMS error (direct): {:.3} pixels (valid: {}/{})",
        rms_error,
        valid_count,
        num_points * dataset.cameras.len()
    );
    info!(
        "Initial RMS error (via SE3): {:.3} pixels (valid: {})",
        se3_rms, se3_valid_count
    );
    if valid_count > 0 {
        info!("First 10 errors: {:?}, max: {:.1}", errors, max_error);
    }

    // Count observations per point to check for under-constrained points
    let mut point_obs_count: HashMap<usize, usize> = HashMap::new();
    for obs in &dataset.observations {
        if obs.point_index < num_points {
            *point_obs_count.entry(obs.point_index).or_insert(0) += 1;
        }
    }

    // Only include points with at least 2 observations (needed for triangulation)
    let valid_points: std::collections::HashSet<usize> = point_obs_count
        .iter()
        .filter(|&(_, count)| *count >= 2)
        .map(|(&idx, _)| idx)
        .collect();

    info!(
        "Points with >= 2 observations: {}/{}",
        valid_points.len(),
        num_points
    );

    // Add projection factors (one per observation)
    // Each factor connects one camera pose (SE3) to one 3D landmark (R3)
    let mut total_obs = 0;
    for obs in &dataset.observations {
        if obs.point_index >= num_points || !valid_points.contains(&obs.point_index) {
            continue;
        }

        let cam = &dataset.cameras[obs.camera_index];

        // Create BAL camera with distortion
        let camera = BALPinholeCamera::new(
            cam.focal_length,
            cam.focal_length, // fy = fx for BAL
            0.0,              // cx = 0 (principal point at origin)
            0.0,              // cy = 0
            cam.k1,
            cam.k2,
        );

        // Create projection factor for this observation
        // ProjectionFactor expects Matrix2xX of observations (2 rows, N columns)
        let observations = Matrix2xX::from_column_slice(&[obs.x, obs.y]);

        // Use generic ProjectionFactor with BundleAdjustment configuration
        let factor =
            ProjectionFactor::<BALPinholeCamera, BundleAdjustment>::new(observations, camera);

        let camera_name = format!("cam_{:04}", obs.camera_index);
        let point_name = format!("pt_{:05}", obs.point_index);

        // Use Huber loss to handle outliers robustly
        let huber_loss = HuberLoss::new(1.0)?; // Scale of 1 pixel
        problem.add_residual_block(
            &[&camera_name, &point_name],
            Box::new(factor),
            Some(Box::new(huber_loss)),
        );
        total_obs += 1;
    }

    // Only add variables for valid points
    // (Clear old point values and re-add only valid ones)
    let old_values = initial_values.clone();
    initial_values.clear();
    for (name, value) in old_values {
        if name.starts_with("cam_") {
            initial_values.insert(name, value);
        } else if let Some(stripped) = name.strip_prefix("pt_") {
            let idx: usize = stripped.parse().unwrap_or(usize::MAX);
            if valid_points.contains(&idx) {
                initial_values.insert(name, value);
            }
        }
    }

    // Fix first camera to remove gauge freedom (prevents singular matrix)
    for dof_idx in 0..6 {
        problem.fix_variable("cam_0000", dof_idx);
    }

    info!("Problem setup:");
    info!("Cameras: {}", dataset.cameras.len());
    info!("3D points: {}", num_points);
    info!("Observations (total): {}", total_obs);
    info!("Reprojection factors: {} (one per observation)", total_obs);
    info!("Fixed cameras: 1 (cam_0000 - gauge fixing)");
    info!("Optimization mode: Full bundle adjustment (poses + landmarks)");
    info!(
        "Total parameters: {} (cameras: {}, landmarks: {})",
        dataset.cameras.len() * 6 + num_points * 3,
        dataset.cameras.len() * 6,
        num_points * 3
    );
    info!("");

    Ok(ProblemSetup {
        problem,
        initial_values,
        num_observations: total_obs,
    })
}

fn run_bundle_adjustment(
    solver_name: &str,
    dataset: &BalDataset,
    num_points: usize,
    max_iterations: usize,
) -> Result<OptimizationMetrics, Box<dyn Error>> {
    // Setup problem
    let setup = setup_bal_problem(dataset, num_points)?;
    let problem = setup.problem;
    let initial_values = setup.initial_values;
    let num_observations = setup.num_observations;

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
        .with_cost_tolerance(1e-12)
        .with_parameter_tolerance(1e-14)
        .with_damping(1e-3); // Standard damping

    // For Schur complement solvers, configure the variant and preconditioner
    if solver_name.starts_with("schur") {
        config = config
            .with_schur_variant(schur_variant)
            .with_schur_preconditioner(SchurPreconditioner::BlockDiagonal);
    }

    info!("Starting optimization");
    info!("Linear solver: {:?}", linear_solver_type);
    if solver_name.starts_with("schur") {
        info!("Schur variant: {:?}", schur_variant);
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

    // Print convergence details
    info!("Convergence details:");
    info!("  Status: {:?}", result.status);
    if let Some(ref conv_info) = result.convergence_info {
        info!(
            "  Final gradient norm: {:.6e}",
            conv_info.final_gradient_norm
        );
        info!(
            "  Final parameter update norm: {:.6e}",
            conv_info.final_parameter_update_norm
        );
        info!("  Cost evaluations: {}", conv_info.cost_evaluations);
        info!("  Jacobian evaluations: {}", conv_info.jacobian_evaluations);
    }
    info!("");

    Ok(OptimizationMetrics {
        solver_name: solver_name.to_string(),
        initial_cost: result.initial_cost,
        final_cost: result.final_cost,
        iterations: result.iterations,
        total_time: elapsed,
        success: matches!(result.status, OptimizationStatus::Converged),
        num_observations,
    })
}

fn print_metrics(metrics: &OptimizationMetrics) {
    info!("Optimization Results:");
    info!("Initial cost: {:.6e}", metrics.initial_cost);
    info!("Final cost: {:.6e}", metrics.final_cost);
    info!("Cost reduction: {:.2}%", metrics.cost_reduction_percent());
    info!(
        "Iterations: {} / {}",
        metrics.iterations,
        if metrics.success { "converged" } else { "max" }
    );
    info!("Total time: {:?}", metrics.total_time);
    info!("Time/iteration: {:?}", metrics.time_per_iteration());
    info!(
        "Status: {}",
        if metrics.success {
            "Converged"
        } else {
            "Failed (max iterations reached)"
        }
    );
    info!("");

    // Analysis
    info!("=== Convergence Analysis ===");
    info!("Number of observations: {}", metrics.num_observations);
    let rms = metrics.rms_reprojection_error();
    info!("RMS reprojection error: {:.3} pixels", rms);
    info!(
        "Cost per observation: {:.3}",
        metrics.final_cost / metrics.num_observations as f64
    );
    if rms < 1.0 {
        info!("✓ Excellent: RMS error < 1 pixel");
    } else if rms < 2.0 {
        info!("✓ Good: RMS error < 2 pixels");
    } else if rms < 5.0 {
        info!("~ Acceptable: RMS error < 5 pixels");
    } else {
        info!("✗ Poor: RMS error > 5 pixels (needs more iterations or better initialization)");
    }
}

fn print_comparison_table(all_metrics: &[OptimizationMetrics]) {
    info!("=== Solver Comparison ===");
    info!("");
    info!(
        "{:<20} {:>12} {:>12} {:>10} {:>12} {:>10}",
        "Solver", "Init Cost", "Final Cost", "Reduction", "Time", "RMS (px)"
    );
    info!("{}", "-".repeat(80));

    for m in all_metrics {
        info!(
            "{:<20} {:>12.3e} {:>12.3e} {:>9.2}% {:>12.2?} {:>10.3}",
            m.solver_name,
            m.initial_cost,
            m.final_cost,
            m.cost_reduction_percent(),
            m.total_time,
            m.rms_reprojection_error()
        );
    }
    info!("");
}
