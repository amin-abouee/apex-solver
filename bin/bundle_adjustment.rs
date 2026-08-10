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
//!
//! # With specific solver variant:
//! cargo run --release --bin bundle_adjustment -- problem.txt --solver implicit
//!
//! # With specific optimization type:
//! cargo run --release --bin bundle_adjustment -- problem.txt --type bundle-adjustment
//!
//! # With visualization (requires visualization feature):
//! cargo run --release --features visualization --bin bundle_adjustment -- path/to/problem.txt --with-visualizer
//!
//! # With limited points and visualization:
//! cargo run --release --features visualization --bin bundle_adjustment -- problem.txt -n 1000 --with-visualizer
//! ```
//!
//! # Camera Parameterization
//!
//! Uses ProjectionFactor with SE3 poses and BALPinholeCameraStrict camera model.

use apex_solver::JacobianMode;
use apex_solver::apex_camera_models::{BALPinholeCameraStrict, DistortionModel, PinholeParams};
use apex_solver::apex_io::{BalDataset, BalLoader};
use apex_solver::apex_manifolds::se3::SE3;
use apex_solver::apex_manifolds::so3::SO3;
use apex_solver::apex_manifolds::{LieGroup, ManifoldType};
use apex_solver::core::VarKey;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::ProjectionFactor;
use apex_solver::init_logger;
use apex_solver::linalg::{
    LinearSolver, LinearSolverType, SchurVariant, SparseCholeskySolver, SparseMode, TimedSolver,
};
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use clap::{Parser, ValueEnum};
use nalgebra::{DVector, Matrix2xX, Vector2, Vector3};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

/// Linear solver to drive the optimization with.
///
/// The two Schur variants exploit the camera/landmark block structure and are
/// what you want in production. The direct variants factorize the *full* normal
/// equations instead — far more expensive, but the only way to compare the CPU
/// and GPU sparse Cholesky implementations on identical work.
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
enum SolverArg {
    /// Explicit Schur: direct sparse Cholesky factorization
    Explicit,
    /// Implicit Schur: iterative PCG solver (default, most efficient)
    #[default]
    Implicit,
    /// Direct sparse Cholesky on the full normal equations (CPU)
    SparseCholesky,
    /// Direct sparse Cholesky on the full normal equations (GPU, cuSOLVER)
    GpuSparseCholesky,
    /// Direct sparse QR on the full normal equations (GPU, cuSOLVER)
    GpuSparseQr,
}

impl SolverArg {
    /// The Schur variant to configure; irrelevant for the direct solvers, which
    /// never enter the Schur path.
    fn schur_variant(self) -> SchurVariant {
        match self {
            SolverArg::Explicit => SchurVariant::Sparse,
            _ => SchurVariant::Iterative,
        }
    }

    /// The direct linear solver this variant selects, or `None` for the Schur
    /// variants.
    fn linear_solver_type(self) -> Option<LinearSolverType> {
        match self {
            SolverArg::SparseCholesky => Some(LinearSolverType::SparseCholesky),
            SolverArg::GpuSparseCholesky => Some(LinearSolverType::GpuSparseCholesky),
            SolverArg::GpuSparseQr => Some(LinearSolverType::GpuSparseQR),
            SolverArg::Explicit | SolverArg::Implicit => None,
        }
    }
}

/// Optimization type (which parameters to optimize)
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum OptimizationType {
    /// Bundle Adjustment: optimize pose + landmarks (intrinsics fixed)
    BundleAdjustment,
    /// Self-Calibration: optimize pose + landmarks + intrinsics (default)
    #[default]
    SelfCalibration,
    /// Only Pose: optimize pose (landmarks and intrinsics fixed)
    OnlyPose,
    /// Only Landmarks: optimize landmarks (pose and intrinsics fixed)
    OnlyLandmarks,
    /// Only Intrinsics: optimize intrinsics (pose and landmarks fixed)
    OnlyIntrinsics,
}

/// Bundle adjustment optimization for BAL datasets
#[derive(Parser)]
#[command(name = "bundle_adjustment")]
#[command(about = "Bundle adjustment optimization for BAL datasets")]
struct Args {
    /// BAL file path (required, positional).
    /// If the file does not exist, provide --cameras and --points to auto-download it
    /// from the dataset registry (the dataset name is inferred from the parent directory).
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// Number of cameras in the problem (used for auto-download if FILE is missing)
    #[arg(long)]
    cameras: Option<u32>,

    /// Number of points in the problem (used for auto-download if FILE is missing)
    #[arg(long)]
    points: Option<u32>,

    /// Limit number of points (for testing)
    #[arg(short = 'n', long)]
    num_points: Option<usize>,

    /// Solver variant for Schur complement
    #[arg(short = 's', long, value_enum, default_value = "implicit")]
    solver: SolverArg,

    /// Optimization type
    #[arg(short = 't', long, value_enum, default_value = "self-calibration")]
    optimization_type: OptimizationType,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Enable real-time Rerun visualization
    /// (Requires the `visualization` feature to be enabled)
    #[arg(long)]
    #[cfg(feature = "visualization")]
    with_visualizer: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Initialize logger
    init_logger();

    info!("APEX-SOLVER BUNDLE ADJUSTMENT");
    info!("");

    // Resolve file: download on demand if missing and --cameras/--points are provided
    let file_path = if args.file.exists() {
        args.file.clone()
    } else if let (Some(cameras), Some(points)) = (args.cameras, args.points) {
        // Infer dataset name from the parent directory (e.g. data/bundle_adjustment/ladybug/...)
        let dataset_name = args
            .file
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .ok_or_else(|| {
                format!(
                    "Cannot infer dataset name from path '{}'. \
                     Ensure the path has the form data/bundle_adjustment/<name>/problem-…txt",
                    args.file.display()
                )
            })?;
        info!(
            "File not found — downloading {dataset_name}/problem-{cameras}-{points} from registry …"
        );
        apex_solver::apex_io::ensure_ba_dataset(dataset_name, cameras, points)
            .map_err(|e| format!("Auto-download failed: {e}"))?
    } else {
        return Err(format!(
            "File not found: {}. Provide --cameras and --points to download it automatically.",
            args.file.display()
        )
        .into());
    };

    // Load BAL dataset
    info!("Loading BAL dataset: {}", file_path.display());
    let start_load = Instant::now();
    let dataset = BalLoader::load(file_path.to_string_lossy().as_ref())?;
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

    #[cfg(feature = "visualization")]
    let with_visualizer = args.with_visualizer;
    #[cfg(not(feature = "visualization"))]
    let with_visualizer = false;

    // Run bundle adjustment
    run_bundle_adjustment(
        &dataset,
        num_points_to_use,
        args.solver,
        args.optimization_type,
        args.verbose,
        with_visualizer,
    )
}

/// Convert axis-angle rotation to SO3
fn axis_angle_to_so3(axis_angle: &Vector3<f64>) -> SO3 {
    let angle = axis_angle.norm();
    if angle < 1e-10 {
        SO3::identity()
    } else {
        let axis = axis_angle / angle;
        SO3::from_axis_angle(&axis, angle)
    }
}

/// One-line summary of how much of a run went into the linear solve.
fn solve_summary<S: LinearSolver<SparseMode>>(timed: &TimedSolver<S>) -> String {
    format!(
        "{:.2} s over {} solves ({:.1} ms each)",
        timed.solve_time().as_secs_f64(),
        timed.solve_count(),
        timed.mean_solve_time().as_secs_f64() * 1e3,
    )
}

/// A directly-constructed sparse linear solver, if the CLI selected one.
type DirectSolver = Option<Box<dyn LinearSolver<SparseMode>>>;

/// Build the direct linear solver `solver_arg` selects, or `None` for the Schur
/// variants, which the optimizer constructs itself.
fn direct_solver(solver_arg: SolverArg) -> Result<DirectSolver, Box<dyn Error>> {
    Ok(match solver_arg {
        SolverArg::SparseCholesky => Some(Box::new(SparseCholeskySolver::new())),
        #[cfg(feature = "cuda")]
        SolverArg::GpuSparseCholesky => Some(Box::new(
            apex_solver::linalg::CudaSparseCholeskySolver::new()?,
        )),
        #[cfg(feature = "cuda")]
        SolverArg::GpuSparseQr => {
            Some(Box::new(apex_solver::linalg::CudaSparseQRSolver::new()?))
        }
        #[cfg(not(feature = "cuda"))]
        SolverArg::GpuSparseCholesky | SolverArg::GpuSparseQr => {
            return Err("GPU solvers require building with `--features cuda`".into());
        }
        SolverArg::Explicit | SolverArg::Implicit => None,
    })
}

/// Run bundle adjustment with specified solver and optimization type
#[cfg_attr(not(feature = "visualization"), allow(unused_variables))]
fn run_bundle_adjustment(
    dataset: &BalDataset,
    num_points: usize,
    solver_arg: SolverArg,
    opt_type: OptimizationType,
    verbose: bool,
    with_visualizer: bool,
) -> Result<(), Box<dyn Error>> {
    let solver_variant = solver_arg.schur_variant();
    use apex_solver::factors::{
        BundleAdjustment, OnlyIntrinsics, OnlyLandmarks, OnlyPose, SelfCalibration,
    };

    let mut problem = Problem::new(JacobianMode::Sparse);

    // Track VarKeys by camera/point index
    let mut pose_keys: Vec<VarKey> = Vec::with_capacity(dataset.cameras.len());
    let mut intr_keys: Vec<VarKey> = Vec::with_capacity(dataset.cameras.len());
    let mut pt_keys: Vec<VarKey> = Vec::with_capacity(num_points);

    // Add cameras as SE3 poses + intrinsic variables
    info!(
        "Adding {} cameras as SE3 poses + intrinsics...",
        dataset.cameras.len()
    );
    for cam in &dataset.cameras {
        // Convert axis-angle to SE3
        let axis_angle = Vector3::new(cam.rotation.x, cam.rotation.y, cam.rotation.z);
        let translation = Vector3::new(cam.translation.x, cam.translation.y, cam.translation.z);
        let so3 = axis_angle_to_so3(&axis_angle);
        let pose = SE3::from_translation_so3(translation, so3);

        // Add SE3 pose variable (6 DOF)
        let pose_key = problem.add_variable(
            ManifoldType::SE3,
            DVector::from_column_slice(pose.as_param_slice()),
        );
        pose_keys.push(pose_key);

        // Add intrinsics: [focal, k1, k2] (3 DOF)
        let intrinsics_vec = DVector::from_vec(vec![cam.focal_length, cam.k1, cam.k2]);
        let intr_key = problem.add_variable(ManifoldType::RN, intrinsics_vec);
        intr_keys.push(intr_key);
    }

    // Add landmarks as RN(3)
    info!("Adding {} landmarks as RN(3) variables...", num_points);
    for j in 0..num_points {
        let point = &dataset.points[j];
        let point_vec =
            DVector::from_vec(vec![point.position.x, point.position.y, point.position.z]);
        let pt_key = problem.add_variable(ManifoldType::RN, point_vec);
        problem.mark_as_schur_landmark(pt_key);
        pt_keys.push(pt_key);
    }

    // Count valid observations
    let valid_obs: Vec<_> = dataset
        .observations
        .iter()
        .filter(|obs| obs.point_index < num_points)
        .collect();

    // Add projection factors based on optimization type
    info!(
        "Adding {} projection factors (optimization: {:?})...",
        valid_obs.len(),
        opt_type
    );

    match opt_type {
        OptimizationType::SelfCalibration => {
            add_factors::<SelfCalibration>(
                &mut problem,
                dataset,
                &valid_obs,
                &pose_keys,
                &pt_keys,
                &intr_keys,
                true,
            )?;
        }
        OptimizationType::BundleAdjustment => {
            add_factors::<BundleAdjustment>(
                &mut problem,
                dataset,
                &valid_obs,
                &pose_keys,
                &pt_keys,
                &intr_keys,
                false,
            )?;
        }
        OptimizationType::OnlyPose => {
            add_factors::<OnlyPose>(
                &mut problem,
                dataset,
                &valid_obs,
                &pose_keys,
                &pt_keys,
                &intr_keys,
                false,
            )?;
        }
        OptimizationType::OnlyLandmarks => {
            add_factors::<OnlyLandmarks>(
                &mut problem,
                dataset,
                &valid_obs,
                &pose_keys,
                &pt_keys,
                &intr_keys,
                false,
            )?;
        }
        OptimizationType::OnlyIntrinsics => {
            add_factors::<OnlyIntrinsics>(
                &mut problem,
                dataset,
                &valid_obs,
                &pose_keys,
                &pt_keys,
                &intr_keys,
                true,
            )?;
        }
    }

    // Fix first camera pose (gauge freedom) - all 6 DOF
    info!("Fixing first camera pose (all 6 DOF) for gauge freedom...");
    for dof in 0..6 {
        problem.fix_variable(pose_keys[0], dof);
    }

    // Configure solver
    let mut config = LevenbergMarquardtConfig::for_bundle_adjustment();
    config.schur_variant = solver_variant;
    if let Some(linear_solver_type) = solver_arg.linear_solver_type() {
        config.linear_solver_type = linear_solver_type;
    }

    info!("");
    info!("Solver configuration:");
    info!("  Solver variant: {:?}", solver_variant);
    info!("  Optimization type: {:?}", opt_type);
    info!("  Linear solver: {:?}", config.linear_solver_type);
    info!("  Preconditioner: {:?}", config.schur_preconditioner);

    let mut solver = LevenbergMarquardt::with_config(config);

    #[cfg(feature = "visualization")]
    if with_visualizer {
        use apex_solver::observers::RerunObserver;
        use apex_solver::observers::visualization::VisualizationConfig;
        let config = VisualizationConfig::for_bundle_adjustment()
            .with_camera_frustum_scale(0.1)
            .with_initial_landmark_color([255, 255, 255])
            .with_optimized_landmark_color([255, 255, 255]);
        match RerunObserver::with_config(true, None, config) {
            Ok(observer) => {
                solver.add_observer(observer);
                info!("Rerun visualization enabled for bundle adjustment");
            }
            Err(e) => tracing::warn!("Failed to create Rerun observer: {}", e),
        }
    }

    // Print diagnostic info
    let num_cameras = dataset.cameras.len();
    let num_factors = valid_obs.len();
    let pose_dof = num_cameras * 6;
    let intrinsic_dof = num_cameras * 3;
    let landmark_dof = num_points * 3;
    let total_dof = pose_dof + intrinsic_dof + landmark_dof;

    info!("");
    info!("Diagnostics:");
    info!("  Cameras: {}", num_cameras);
    info!("  Number of factors (observations): {}", num_factors);
    info!("  Pose DOF: {} (6 per camera)", pose_dof);
    info!("  Intrinsic DOF: {} (3 per camera)", intrinsic_dof);
    info!("  Landmark DOF: {}", landmark_dof);
    info!("  Total DOF: {}", total_dof);
    info!(
        "  DOF per observation: {:.2}",
        total_dof as f64 / num_factors as f64
    );

    // Optimize
    info!("");
    info!("Starting optimization...");
    let start = Instant::now();
    // The direct solvers are routed through `optimize_with_mode` wrapped in a
    // TimedSolver, so the factorization cost can be separated from the Jacobian
    // assembly that every backend pays identically. The Schur path keeps the
    // ordinary entry point, which builds its own structured solver.
    let (result, solve_time) = match solver_arg {
        #[cfg(feature = "cuda")]
        SolverArg::GpuSparseCholesky => {
            let cuda = apex_solver::linalg::CudaSparseCholeskySolver::new()?;
            let mut timed = TimedSolver::new(cuda);
            let result = solver.optimize_with_mode::<SparseMode>(&mut problem, &mut timed)?;
            let summary = solve_summary(&timed);
            if verbose {
                info!("CUDA phase breakdown:\n{}", timed.inner().profile());
            }
            (result, Some(summary))
        }
        _ => match direct_solver(solver_arg)? {
            Some(inner) => {
                let mut timed = TimedSolver::new(inner);
                let result = solver.optimize_with_mode::<SparseMode>(&mut problem, &mut timed)?;
                let summary = solve_summary(&timed);
                (result, Some(summary))
            }
            None => (solver.optimize(&mut problem)?, None),
        },
    };
    let elapsed = start.elapsed();

    info!("");
    info!("Optimization completed!");
    info!("Status: {:?}", result.status);
    info!("Iterations: {}", result.iterations);
    info!("Time: {:.2} seconds", elapsed.as_secs_f64());
    if let Some(solve_time) = solve_time {
        info!("Linear solve: {}", solve_time);
    }

    let num_obs = valid_obs.len() as f64;
    let initial_rmse = (result.initial_cost / num_obs).sqrt();
    let final_rmse = (result.final_cost / num_obs).sqrt();

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
        info!(
            "  Per-iteration: {:.2}s",
            elapsed.as_secs_f64() / result.iterations as f64
        );
    }

    Ok(())
}

/// Helper function to add factors with a specific optimization configuration
#[allow(clippy::too_many_arguments)]
fn add_factors<OP>(
    problem: &mut Problem,
    dataset: &BalDataset,
    valid_obs: &[&apex_solver::apex_io::BalObservation],
    pose_keys: &[VarKey],
    pt_keys: &[VarKey],
    intr_keys: &[VarKey],
    include_intrinsics: bool,
) -> Result<(), Box<dyn Error>>
where
    OP: apex_solver::factors::projection_factor::OptimizationConfig + 'static,
{
    for obs in valid_obs {
        let cam = &dataset.cameras[obs.camera_index];
        // Focal length is already normalized by BAL loader
        let camera = BALPinholeCameraStrict::new(
            PinholeParams {
                fx: cam.focal_length,
                fy: cam.focal_length,
                cx: 0.0,
                cy: 0.0,
            },
            DistortionModel::Radial {
                k1: cam.k1,
                k2: cam.k2,
            },
        )?;

        // Single observation per factor
        let observations = Matrix2xX::from_columns(&[Vector2::new(obs.x, obs.y)]);
        let factor: ProjectionFactor<BALPinholeCameraStrict, OP> =
            ProjectionFactor::new(observations, camera);

        let pose_key = pose_keys[obs.camera_index];
        let pt_key = pt_keys[obs.point_index];
        let intr_key = intr_keys[obs.camera_index];

        let loss = match HuberLoss::new(1.0) {
            Ok(l) => Box::new(l),
            Err(_) => continue,
        };

        if include_intrinsics {
            problem.add_residual_block(&[pose_key, pt_key, intr_key], Box::new(factor), Some(loss));
        } else {
            problem.add_residual_block(&[pose_key, pt_key], Box::new(factor), Some(loss));
        }
    }
    Ok(())
}
