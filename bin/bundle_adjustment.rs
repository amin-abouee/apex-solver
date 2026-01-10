//! Bundle Adjustment Binary
//!
//! Loads BAL (Bundle Adjustment in the Large) files and optimizes them
//! using configurable solvers with Levenberg-Marquardt optimization.
//!
//! # Usage
//! ```bash
//! # Basic usage (default: schur solver)
//! cargo run --bin bundle_adjustment -- data/bundle_adjustment/problem-21-11315-pre.txt
//!
//! # Compare all solvers
//! cargo run --bin bundle_adjustment -- --solver all data/bundle_adjustment/problem-21-11315-pre.txt
//!
//! # Use specific solver with custom params
//! cargo run --bin bundle_adjustment -- \
//!     --solver schur-iterative \
//!     --max-iterations 200 \
//!     --loss-function cauchy \
//!     data/bundle_adjustment/problem-89-110973-pre.txt
//!
//! # Limit points for faster testing
//! cargo run --bin bundle_adjustment -- -n 1000 data/bundle_adjustment/problem-89-110973-pre.txt
//! ```

use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use apex_solver::core::loss_functions::{CauchyLoss, HuberLoss, L2Loss, LossFunction};
use apex_solver::core::problem::Problem;
use apex_solver::factors::ProjectionFactor;
use apex_solver::factors::camera::{BALPinholeCamera, BundleAdjustment};
use apex_solver::init_logger;
use apex_solver::io::{BalDataset, BalLoader};
use apex_solver::linalg::{LinearSolverType, SchurPreconditioner, SchurVariant};
use apex_solver::manifold::ManifoldType;
use apex_solver::manifold::se3::SE3;
use apex_solver::manifold::so3::SO3;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use clap::{Parser, ValueEnum};
use nalgebra::{DVector, Matrix2xX, Vector3};
use tracing::{debug, error, info, warn};

/// Bundle adjustment optimization for BAL datasets
#[derive(Parser)]
#[command(name = "bundle_adjustment")]
#[command(about = "Bundle adjustment optimization for BAL datasets")]
struct Args {
    /// BAL file path (required, positional)
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// Solver: cholesky, schur, schur-iterative, schur-power-series, all
    #[arg(short, long, value_enum, default_value = "schur-iterative")]
    solver: SolverChoice,

    /// Maximum iterations
    #[arg(short = 'i', long, default_value = "300")]
    max_iterations: usize,

    /// Cost tolerance
    #[arg(long, default_value = "1e-6")]
    cost_tolerance: f64,

    /// Parameter tolerance
    #[arg(long, default_value = "1e-8")]
    parameter_tolerance: f64,

    /// Loss function: l2, huber, cauchy
    #[arg(long, default_value = "huber")]
    loss_function: String,

    /// Loss scale parameter
    #[arg(long, default_value = "1.0")]
    loss_scale: f64,

    /// Limit number of points (for testing)
    #[arg(short = 'n', long)]
    num_points: Option<usize>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Enable real-time Rerun visualization
    /// (Requires the `visualization` feature to be enabled)
    #[arg(long)]
    #[cfg(feature = "visualization")]
    with_visualizer: bool,

    /// Show only camera poses (hide landmarks)
    #[arg(long)]
    #[cfg(feature = "visualization")]
    vis_cameras_only: bool,

    /// Show only 3D landmarks (hide cameras)
    #[arg(long)]
    #[cfg(feature = "visualization")]
    vis_landmarks_only: bool,

    /// Hide camera frustums from visualization
    #[arg(long)]
    #[cfg(feature = "visualization")]
    vis_hide_cameras: bool,

    /// Hide 3D landmark points from visualization
    #[arg(long)]
    #[cfg(feature = "visualization")]
    vis_hide_landmarks: bool,

    /// Hide time series plots (cost, gradient, etc.)
    #[arg(long)]
    #[cfg(feature = "visualization")]
    vis_hide_plots: bool,

    /// Hide matrix visualizations (Hessian, gradient)
    #[arg(long)]
    #[cfg(feature = "visualization")]
    vis_hide_matrices: bool,

    /// Camera frustum field of view in radians
    #[arg(long, default_value = "0.5")]
    #[cfg(feature = "visualization")]
    vis_camera_fov: f32,

    /// Camera frustum aspect ratio
    #[arg(long, default_value = "1.0")]
    #[cfg(feature = "visualization")]
    vis_camera_aspect: f32,

    /// Camera frustum scale factor
    #[arg(long, default_value = "1.0")]
    #[cfg(feature = "visualization")]
    vis_camera_scale: f32,

    /// 3D landmark point size/radius
    #[arg(long, default_value = "0.02")]
    #[cfg(feature = "visualization")]
    vis_point_size: f32,

    /// Save visualization to file instead of spawning viewer
    #[arg(long)]
    #[cfg(feature = "visualization")]
    vis_save_path: Option<String>,

    /// Show optimization progress at each iteration (default: show only initial and final)
    #[arg(long)]
    #[cfg(feature = "visualization")]
    vis_iterative: bool,
}

#[derive(Clone, Copy, ValueEnum, Debug, PartialEq, Eq)]
enum SolverChoice {
    Cholesky,
    Schur,
    SchurIterative,
    SchurPowerSeries,
    All,
}

impl SolverChoice {
    fn display_name(&self) -> &'static str {
        match self {
            SolverChoice::Cholesky => "Cholesky",
            SolverChoice::Schur => "Schur (Sparse)",
            SolverChoice::SchurIterative => "Schur (PCG)",
            SolverChoice::SchurPowerSeries => "Schur (PowerSeries)",
            SolverChoice::All => "All",
        }
    }

    fn all_solvers() -> Vec<SolverChoice> {
        vec![
            SolverChoice::Cholesky,
            SolverChoice::Schur,
            SolverChoice::SchurIterative,
            SolverChoice::SchurPowerSeries,
        ]
    }
}

/// Result from a single optimization run
#[allow(dead_code)]
struct OptimizationResult {
    solver_name: String,
    num_cameras: usize,
    num_points: usize,
    num_observations: usize,
    initial_cost: f64,
    final_cost: f64,
    iterations: usize,
    time: Duration,
    status: String,
    convergence_reason: String,
}

impl OptimizationResult {
    fn cost_reduction_percent(&self) -> f64 {
        if self.initial_cost > 0.0 {
            (self.initial_cost - self.final_cost) / self.initial_cost * 100.0
        } else {
            0.0
        }
    }

    fn rms_error(&self) -> f64 {
        // Cost = sum of squared residuals (u^2 + v^2 per observation)
        // RMS = sqrt(cost / num_observations)
        if self.num_observations > 0 {
            (self.final_cost / self.num_observations as f64).sqrt()
        } else {
            0.0
        }
    }
}

/// Problem setup containing problem, initial values, and metadata
#[allow(dead_code)]
struct ProblemSetup {
    problem: Problem,
    initial_values: HashMap<String, (ManifoldType, DVector<f64>)>,
    num_cameras: usize,
    num_points: usize,
    num_observations: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Initialize logger
    init_logger();

    info!("APEX-SOLVER BUNDLE ADJUSTMENT");
    info!("");

    // Validate file exists
    if !args.file.exists() {
        error!("File not found: {}", args.file.display());
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

    // Determine which solvers to run
    let solvers = if args.solver == SolverChoice::All {
        SolverChoice::all_solvers()
    } else {
        vec![args.solver]
    };

    let mut results = Vec::new();

    for solver in &solvers {
        info!("Running solver: {}", solver.display_name());
        info!("{}", "-".repeat(60));

        match run_bundle_adjustment(&dataset, num_points_to_use, *solver, &args) {
            Ok(result) => {
                print_single_result(&result);
                results.push(result);
            }
            Err(e) => {
                error!("Solver {} failed: {}", solver.display_name(), e);
                // Add a failed result for the table
                results.push(OptimizationResult {
                    solver_name: solver.display_name().to_string(),
                    num_cameras: dataset.cameras.len(),
                    num_points: num_points_to_use,
                    num_observations: 0,
                    initial_cost: 0.0,
                    final_cost: 0.0,
                    iterations: 0,
                    time: Duration::ZERO,
                    status: "FAILED".to_string(),
                    convergence_reason: e.to_string(),
                });
            }
        }
        info!("");
    }

    // Print comparison table if multiple solvers were run
    if results.len() > 1 {
        print_comparison_table(&results);
    }

    // Determine exit status
    let converged_count = results.iter().filter(|r| r.status == "CONVERGED").count();
    let total_count = results.len();

    if converged_count == total_count {
        info!("All {} solver(s) converged successfully", total_count);
        Ok(())
    } else if converged_count == 0 {
        error!("No solvers converged");
        Err("No solvers converged".into())
    } else {
        warn!("{}/{} solvers converged", converged_count, total_count);
        Ok(())
    }
}

/// Setup the bundle adjustment problem from BAL dataset
fn setup_bal_problem(
    dataset: &BalDataset,
    num_points: usize,
) -> Result<ProblemSetup, Box<dyn Error>> {
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    let num_points = num_points.min(dataset.points.len());

    // Add camera poses as SE3 variables
    // BAL convention: World-to-Camera transformation (P_cam = R * X_world + t)
    for (i, cam) in dataset.cameras.iter().enumerate() {
        let axis_angle = Vector3::new(cam.rotation[0], cam.rotation[1], cam.rotation[2]);
        let t_bal = Vector3::new(cam.translation[0], cam.translation[1], cam.translation[2]);
        let r_bal = SO3::from_scaled_axis(axis_angle);

        // Store directly as world-to-camera (no inversion needed)
        let se3_wc = SE3::from_translation_so3(t_bal, r_bal);
        let pose_vec: DVector<f64> = se3_wc.into();
        initial_values.insert(format!("cam_{:04}", i), (ManifoldType::SE3, pose_vec));

        debug!("Camera {} initialized", i);
    }

    // Add landmarks as RN variables
    for j in 0..num_points {
        let point = &dataset.points[j];
        let point_vec = DVector::from_vec(vec![
            point.position[0],
            point.position[1],
            point.position[2],
        ]);
        initial_values.insert(format!("pt_{:05}", j), (ManifoldType::RN, point_vec));
    }

    // Count observations per point (need >= 2 for triangulation)
    let mut point_obs_count: HashMap<usize, usize> = HashMap::new();
    for obs in &dataset.observations {
        if obs.point_index < num_points {
            *point_obs_count.entry(obs.point_index).or_insert(0) += 1;
        }
    }

    let valid_points: std::collections::HashSet<usize> = point_obs_count
        .iter()
        .filter(|&(_, count)| *count >= 2)
        .map(|(&idx, _)| idx)
        .collect();

    let skipped_points = num_points - valid_points.len();
    if skipped_points > 0 {
        warn!("Skipping {} points with < 2 observations", skipped_points);
    }

    // Add projection factors (one per observation)
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

        // Create projection factor
        let observations = Matrix2xX::from_column_slice(&[obs.x, obs.y]);
        let factor =
            ProjectionFactor::<BALPinholeCamera, BundleAdjustment>::new(observations, camera);

        let camera_name = format!("cam_{:04}", obs.camera_index);
        let point_name = format!("pt_{:05}", obs.point_index);

        // Use Huber loss to handle outliers (added later in run_bundle_adjustment)
        problem.add_residual_block(&[&camera_name, &point_name], Box::new(factor), None);
        total_obs += 1;
    }

    // Remove invalid point variables
    initial_values.retain(|name, _| {
        if name.starts_with("cam_") {
            true
        } else if let Some(stripped) = name.strip_prefix("pt_") {
            if let Ok(idx) = stripped.parse::<usize>() {
                valid_points.contains(&idx)
            } else {
                false
            }
        } else {
            false
        }
    });

    // Fix first camera for gauge freedom
    for dof_idx in 0..6 {
        problem.fix_variable("cam_0000", dof_idx);
    }

    info!("Problem setup complete:");
    info!("  Cameras: {}", dataset.cameras.len());
    info!("  Total points in file: {}", dataset.points.len());
    info!("  Points to optimize: {}", num_points);
    info!("  Valid points (>= 2 observations): {}", valid_points.len());
    info!("  Filtered points: {}", num_points - valid_points.len());
    info!("  Total observations: {}", total_obs);
    info!("  Fixed: cam_0000 (gauge fixing)");

    Ok(ProblemSetup {
        problem,
        initial_values,
        num_cameras: dataset.cameras.len(),
        num_points: valid_points.len(),
        num_observations: total_obs,
    })
}

/// Create loss function based on user choice
fn create_loss_function(
    loss_name: &str,
    scale: f64,
) -> Result<Box<dyn LossFunction + Send>, Box<dyn Error>> {
    let loss: Box<dyn LossFunction + Send> = match loss_name.to_lowercase().as_str() {
        "l2" => Box::new(L2Loss),
        "huber" => Box::new(HuberLoss::new(scale)?),
        "cauchy" => Box::new(CauchyLoss::new(scale)?),
        _ => {
            return Err(format!(
                "Unknown loss function: {}. Valid options: l2, huber, cauchy",
                loss_name
            )
            .into());
        }
    };
    Ok(loss)
}

/// Create solver configuration for a given solver choice
fn create_solver_config(solver: SolverChoice, args: &Args) -> LevenbergMarquardtConfig {
    let (linear_solver_type, schur_variant) = match solver {
        SolverChoice::Cholesky => (LinearSolverType::SparseCholesky, SchurVariant::Sparse),
        SolverChoice::Schur => (
            LinearSolverType::SparseSchurComplement,
            SchurVariant::Sparse,
        ),
        SolverChoice::SchurIterative => (
            LinearSolverType::SparseSchurComplement,
            SchurVariant::Iterative,
        ),
        SolverChoice::SchurPowerSeries => (
            LinearSolverType::SparseSchurComplement,
            SchurVariant::PowerSeries,
        ),
        SolverChoice::All => unreachable!("All should be expanded before calling this function"),
    };

    let mut config = LevenbergMarquardtConfig::new()
        .with_linear_solver_type(linear_solver_type)
        .with_max_iterations(args.max_iterations)
        .with_cost_tolerance(args.cost_tolerance)
        .with_parameter_tolerance(args.parameter_tolerance)
        .with_damping(1e-3);

    // Configure Schur-specific settings
    if matches!(
        solver,
        SolverChoice::Schur | SolverChoice::SchurIterative | SolverChoice::SchurPowerSeries
    ) {
        config = config
            .with_schur_variant(schur_variant)
            .with_schur_preconditioner(SchurPreconditioner::BlockDiagonal);
    }

    config
}

/// Build visualization configuration from CLI arguments
#[cfg(feature = "visualization")]
fn build_visualization_config(args: &Args) -> apex_solver::observers::VisualizationConfig {
    use apex_solver::observers::{VisualizationConfig, VisualizationMode};

    // Determine visualization mode based on CLI flag
    let mode = if args.vis_iterative {
        VisualizationMode::Iterative
    } else {
        VisualizationMode::InitialAndFinal
    };

    let mut config = VisualizationConfig::for_bundle_adjustment().with_visualization_mode(mode);

    // Handle mutually exclusive presets
    if args.vis_cameras_only {
        config = config.with_show_landmarks(false);
    } else if args.vis_landmarks_only {
        config = config.with_show_cameras(false);
    }

    // Individual hide flags
    if args.vis_hide_cameras {
        config = config.with_show_cameras(false);
    }
    if args.vis_hide_landmarks {
        config = config.with_show_landmarks(false);
    }
    if args.vis_hide_plots {
        config = config.with_show_plots(false);
    }
    if args.vis_hide_matrices {
        config = config.with_show_matrices(false);
    }

    // Appearance settings
    config
        .with_camera_fov(args.vis_camera_fov)
        .with_camera_aspect_ratio(args.vis_camera_aspect)
        .with_camera_frustum_scale(args.vis_camera_scale)
        .with_landmark_point_size(args.vis_point_size)
}

/// Run bundle adjustment with a specific solver
fn run_bundle_adjustment(
    dataset: &BalDataset,
    num_points: usize,
    solver: SolverChoice,
    args: &Args,
) -> Result<OptimizationResult, Box<dyn Error>> {
    // Setup problem (fresh for each solver to avoid state contamination)
    let setup = setup_bal_problem(dataset, num_points)?;

    // For loss function, we need to recreate the problem with loss
    let mut problem = Problem::new();
    let initial_values = setup.initial_values;

    // Re-add variables by fixing first camera
    for dof_idx in 0..6 {
        problem.fix_variable("cam_0000", dof_idx);
    }

    // Count valid points
    let mut point_obs_count: HashMap<usize, usize> = HashMap::new();
    for obs in &dataset.observations {
        if obs.point_index < num_points {
            *point_obs_count.entry(obs.point_index).or_insert(0) += 1;
        }
    }

    let valid_points: std::collections::HashSet<usize> = point_obs_count
        .iter()
        .filter(|&(_, count)| *count >= 2)
        .map(|(&idx, _)| idx)
        .collect();

    // Add projection factors with loss function
    let mut total_obs = 0;
    for obs in &dataset.observations {
        if obs.point_index >= num_points || !valid_points.contains(&obs.point_index) {
            continue;
        }

        let cam = &dataset.cameras[obs.camera_index];
        let camera =
            BALPinholeCamera::new(cam.focal_length, cam.focal_length, 0.0, 0.0, cam.k1, cam.k2);

        let observations = Matrix2xX::from_column_slice(&[obs.x, obs.y]);
        let factor =
            ProjectionFactor::<BALPinholeCamera, BundleAdjustment>::new(observations, camera);

        let camera_name = format!("cam_{:04}", obs.camera_index);
        let point_name = format!("pt_{:05}", obs.point_index);

        let loss = create_loss_function(&args.loss_function, args.loss_scale)?;
        problem.add_residual_block(&[&camera_name, &point_name], Box::new(factor), Some(loss));
        total_obs += 1;
    }

    // Create solver config
    let config = create_solver_config(solver, args);

    if args.verbose {
        info!("Solver configuration:");
        info!("  Linear solver: {:?}", config.linear_solver_type);
        if matches!(
            solver,
            SolverChoice::Schur | SolverChoice::SchurIterative | SolverChoice::SchurPowerSeries
        ) {
            info!("  Schur variant: {:?}", config.schur_variant);
        }
        info!("  Max iterations: {}", args.max_iterations);
        info!("  Cost tolerance: {:.0e}", args.cost_tolerance);
        info!("  Parameter tolerance: {:.0e}", args.parameter_tolerance);
        info!(
            "  Loss function: {} (scale={})",
            args.loss_function, args.loss_scale
        );
    }

    // Run optimization
    let start = Instant::now();
    let mut lm_solver = LevenbergMarquardt::with_config(config);

    // Add Rerun visualization if enabled
    #[cfg(feature = "visualization")]
    if args.with_visualizer {
        use apex_solver::observers::RerunObserver;

        let vis_config = build_visualization_config(args);
        let save_path = args.vis_save_path.as_deref();

        match RerunObserver::with_config(true, save_path, vis_config.clone()) {
            Ok(observer) => {
                // Log initial state before optimization
                if let Err(e) = observer.log_initial_ba_state(&initial_values) {
                    warn!("Failed to log initial BA state: {}", e);
                }
                lm_solver.add_observer(observer);
                info!("Rerun visualization enabled (Bundle Adjustment)");
                info!("  - Camera poses inverted (T_wc -> T_cw) for display");
                if vis_config.show_cameras {
                    info!(
                        "  - Camera frustums: FOV={:.2} rad, scale={:.2}",
                        vis_config.camera_fov, vis_config.camera_frustum_scale
                    );
                }
                if vis_config.show_landmarks {
                    info!(
                        "  - 3D landmarks: point size={:.3}",
                        vis_config.landmark_point_size
                    );
                }
                if !vis_config.show_cameras {
                    info!("  - Cameras hidden");
                }
                if !vis_config.show_landmarks {
                    info!("  - Landmarks hidden");
                }
            }
            Err(e) => {
                warn!("Failed to create Rerun observer: {}", e);
            }
        }
    }

    let result = lm_solver.optimize(&problem, &initial_values)?;
    let elapsed = start.elapsed();

    // Determine status and convergence reason
    let (status, convergence_reason) = match &result.status {
        OptimizationStatus::Converged => ("CONVERGED", "Converged".to_string()),
        OptimizationStatus::CostToleranceReached => ("CONVERGED", "CostTolerance".to_string()),
        OptimizationStatus::ParameterToleranceReached => {
            ("CONVERGED", "ParameterTolerance".to_string())
        }
        OptimizationStatus::GradientToleranceReached => {
            ("CONVERGED", "GradientTolerance".to_string())
        }
        OptimizationStatus::TrustRegionRadiusTooSmall => {
            ("CONVERGED", "TrustRegionSmall".to_string())
        }
        OptimizationStatus::MinCostThresholdReached => ("CONVERGED", "MinCostReached".to_string()),
        OptimizationStatus::MaxIterationsReached => ("NOT_CONVERGED", "MaxIterations".to_string()),
        OptimizationStatus::Timeout => ("NOT_CONVERGED", "Timeout".to_string()),
        OptimizationStatus::NumericalFailure => ("FAILED", "NumericalFailure".to_string()),
        OptimizationStatus::IllConditionedJacobian => ("FAILED", "IllConditioned".to_string()),
        OptimizationStatus::InvalidNumericalValues => ("FAILED", "InvalidValues".to_string()),
        OptimizationStatus::UserTerminated => ("NOT_CONVERGED", "UserTerminated".to_string()),
        OptimizationStatus::Failed(msg) => ("FAILED", format!("Failed: {}", msg)),
    };

    Ok(OptimizationResult {
        solver_name: solver.display_name().to_string(),
        num_cameras: setup.num_cameras,
        num_points: setup.num_points,
        num_observations: total_obs,
        initial_cost: result.initial_cost,
        final_cost: result.final_cost,
        iterations: result.iterations,
        time: elapsed,
        status: status.to_string(),
        convergence_reason,
    })
}

/// Print results for a single solver run
fn print_single_result(result: &OptimizationResult) {
    info!("Results:");
    info!("  Initial cost: {:.6e}", result.initial_cost);
    info!("  Final cost: {:.6e}", result.final_cost);
    info!("  Cost reduction: {:.2}%", result.cost_reduction_percent());
    info!("  RMS reprojection error: {:.3} pixels", result.rms_error());
    info!("  Iterations: {}", result.iterations);
    info!("  Time: {:.2?}", result.time);
    info!(
        "  Status: {} ({})",
        result.status, result.convergence_reason
    );
}

/// Print comparison table for multiple solvers
fn print_comparison_table(results: &[OptimizationResult]) {
    info!("");
    info!("=== SOLVER COMPARISON ===");
    info!("");

    // Header
    info!(
        "{:<20} | {:>12} | {:>12} | {:>9} | {:>10} | {:>8} | {:>10} | {:<12}",
        "Solver", "Init Cost", "Final Cost", "Reduction", "RMS (px)", "Iters", "Time", "Status"
    );
    info!("{}", "-".repeat(110));

    // Rows
    for r in results {
        let time_str = format!("{:.2?}", r.time);
        info!(
            "{:<20} | {:>12.4e} | {:>12.4e} | {:>8.2}% | {:>10.3} | {:>8} | {:>10} | {:<12}",
            r.solver_name,
            r.initial_cost,
            r.final_cost,
            r.cost_reduction_percent(),
            r.rms_error(),
            r.iterations,
            time_str,
            r.status
        );
    }
    info!("{}", "-".repeat(110));

    // Summary
    let converged: Vec<_> = results.iter().filter(|r| r.status == "CONVERGED").collect();
    info!("{}/{} solvers converged", converged.len(), results.len());

    if !converged.is_empty() {
        // Best final cost
        if let Some(best_cost) = converged.iter().min_by(|a, b| {
            a.final_cost
                .partial_cmp(&b.final_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            info!(
                "Best final cost: {} ({:.4e})",
                best_cost.solver_name, best_cost.final_cost
            );
        }

        // Fastest
        if let Some(fastest) = converged.iter().min_by(|a, b| a.time.cmp(&b.time)) {
            info!("Fastest: {} ({:.2?})", fastest.solver_name, fastest.time);
        }
    }
}
