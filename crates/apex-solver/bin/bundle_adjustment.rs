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
//! ```
//!
//! # Camera Parameterization
//!
//! Uses ProjectionFactor with SE3 poses and BALPinholeCameraStrict camera model.

use apex_solver::apex_camera_models::{BALPinholeCameraStrict, DistortionModel, PinholeParams};
use apex_solver::apex_io::{BalDataset, BalLoader};
use apex_solver::apex_manifolds::ManifoldType;
use apex_solver::apex_manifolds::se3::SE3;
use apex_solver::apex_manifolds::so3::SO3;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::ProjectionFactor;
use apex_solver::init_logger;
use apex_solver::linalg::SchurVariant;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use clap::{Parser, ValueEnum};
use nalgebra::{DVector, Matrix2xX, Vector2, Vector3};
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

/// Solver variant for Schur complement
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum SolverArg {
    /// Explicit Schur: direct sparse Cholesky factorization
    Explicit,
    /// Implicit Schur: iterative PCG solver (default, most efficient)
    #[default]
    Implicit,
}

impl From<SolverArg> for SchurVariant {
    fn from(arg: SolverArg) -> Self {
        match arg {
            SolverArg::Explicit => SchurVariant::Sparse,
            SolverArg::Implicit => SchurVariant::Iterative,
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
    /// BAL file path (required, positional)
    #[arg(value_name = "FILE")]
    file: PathBuf,

    /// Limit number of points (for testing)
    #[arg(short = 'n', long)]
    num_points: Option<usize>,

    /// Solver variant for Schur complement
    #[arg(short = 's', long, value_enum, default_value = "implicit")]
    solver: SolverArg,

    /// Optimization type
    #[arg(short = 't', long, value_enum, default_value = "self-calibration")]
    r#type: OptimizationType,

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

    // Run bundle adjustment
    run_bundle_adjustment(
        &dataset,
        num_points_to_use,
        args.solver.into(),
        args.r#type,
        args.verbose,
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

/// Run bundle adjustment with specified solver and optimization type
fn run_bundle_adjustment(
    dataset: &BalDataset,
    num_points: usize,
    solver_variant: SchurVariant,
    opt_type: OptimizationType,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    use apex_solver::factors::{
        BundleAdjustment, OnlyIntrinsics, OnlyLandmarks, OnlyPose, SelfCalibration,
    };

    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add cameras as SE3 poses + intrinsic variables
    info!(
        "Adding {} cameras as SE3 poses + intrinsics...",
        dataset.cameras.len()
    );
    for (i, cam) in dataset.cameras.iter().enumerate() {
        // Convert axis-angle to SE3
        let axis_angle = Vector3::new(cam.rotation.x, cam.rotation.y, cam.rotation.z);
        let translation = Vector3::new(cam.translation.x, cam.translation.y, cam.translation.z);
        let so3 = axis_angle_to_so3(&axis_angle);
        let pose = SE3::from_translation_so3(translation, so3);

        // Add SE3 pose variable (6 DOF)
        let pose_name = format!("pose_{:04}", i);
        initial_values.insert(pose_name, (ManifoldType::SE3, DVector::from(pose)));

        // Add intrinsics: [focal, k1, k2] (3 DOF)
        let intrinsics_name = format!("intr_{:04}", i);
        let intrinsics_vec = DVector::from_vec(vec![cam.focal_length, cam.k1, cam.k2]);
        initial_values.insert(intrinsics_name, (ManifoldType::RN, intrinsics_vec));
    }

    // Add landmarks as RN(3)
    info!("Adding {} landmarks as RN(3) variables...", num_points);
    for j in 0..num_points {
        let point = &dataset.points[j];
        let var_name = format!("pt_{:05}", j);
        let point_vec =
            DVector::from_vec(vec![point.position.x, point.position.y, point.position.z]);
        initial_values.insert(var_name, (ManifoldType::RN, point_vec));
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

    // We need to use a macro or dynamic dispatch for different optimization types
    // For now, use match with type-specific factor creation
    // include_intrinsics = true when the optimization type has INTRINSIC = true
    match opt_type {
        OptimizationType::SelfCalibration => {
            add_factors::<SelfCalibration>(&mut problem, dataset, &valid_obs, true)?;
        }
        OptimizationType::BundleAdjustment => {
            add_factors::<BundleAdjustment>(&mut problem, dataset, &valid_obs, false)?;
        }
        OptimizationType::OnlyPose => {
            add_factors::<OnlyPose>(&mut problem, dataset, &valid_obs, false)?;
        }
        OptimizationType::OnlyLandmarks => {
            add_factors::<OnlyLandmarks>(&mut problem, dataset, &valid_obs, false)?;
        }
        OptimizationType::OnlyIntrinsics => {
            add_factors::<OnlyIntrinsics>(&mut problem, dataset, &valid_obs, true)?;
        }
    }

    // Fix first camera pose (gauge freedom) - all 6 DOF
    info!("Fixing first camera pose (all 6 DOF) for gauge freedom...");
    for dof in 0..6 {
        problem.fix_variable("pose_0000", dof);
    }

    // Configure solver
    let mut config = LevenbergMarquardtConfig::for_bundle_adjustment();
    config.schur_variant = solver_variant;

    info!("");
    info!("Solver configuration:");
    info!("  Solver variant: {:?}", solver_variant);
    info!("  Optimization type: {:?}", opt_type);
    info!("  Linear solver: {:?}", config.linear_solver_type);
    info!("  Preconditioner: {:?}", config.schur_preconditioner);

    let mut solver = LevenbergMarquardt::with_config(config);

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
    let result = solver.optimize(&problem, &initial_values)?;
    let elapsed = start.elapsed();

    info!("");
    info!("Optimization completed!");
    info!("Status: {:?}", result.status);
    info!("Iterations: {}", result.iterations);
    info!("Time: {:.2} seconds", elapsed.as_secs_f64());

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
fn add_factors<OP>(
    problem: &mut Problem,
    dataset: &BalDataset,
    valid_obs: &[&apex_solver::apex_io::BalObservation],
    include_intrinsics: bool,
) -> Result<(), Box<dyn Error>>
where
    OP: apex_solver::factors::projection_factor::OptimizationConfig + 'static,
{
    for obs in valid_obs {
        let cam = &dataset.cameras[obs.camera_index];
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

        let pose_name = format!("pose_{:04}", obs.camera_index);
        let pt_name = format!("pt_{:05}", obs.point_index);
        let intr_name = format!("intr_{:04}", obs.camera_index);

        let loss = match HuberLoss::new(1.0) {
            Ok(l) => Box::new(l),
            Err(_) => continue,
        };

        if include_intrinsics {
            problem.add_residual_block(
                &[&pose_name, &pt_name, &intr_name],
                Box::new(factor),
                Some(loss),
            );
        } else {
            problem.add_residual_block(&[&pose_name, &pt_name], Box::new(factor), Some(loss));
        }
    }
    Ok(())
}
