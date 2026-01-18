//! Bundle Adjustment Benchmark Comparison
//!
//! Compares Apex Solver against Ceres, GTSAM, and g2o on BAL datasets.
//!
//! This benchmark tests convergence and performance on large-scale bundle adjustment
//! problems to diagnose issues with Apex Solver's implementation.
//!
//! Metrics:
//! - Initial/Final MSE (Mean Squared Error in pixels²)
//! - Initial/Final RMSE (Root Mean Squared Error in pixels)
//! - Runtime in seconds (optimization only, excludes parsing)
//! - Number of iterations
//! - Convergence status

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tracing::{error, info, warn};

// apex-solver imports
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::ProjectionFactor;
use apex_solver::factors::camera::{BALPinholeCameraStrict, BundleAdjustment};
use apex_solver::init_logger;
use apex_solver::io::BalLoader;
// Note: LinearSolverType, SchurPreconditioner, SchurVariant are now set via for_bundle_adjustment()
#[allow(unused_imports)]
use apex_solver::linalg::{LinearSolverType, SchurPreconditioner, SchurVariant};
use apex_solver::manifold::ManifoldType;
use apex_solver::manifold::se3::SE3;
use apex_solver::manifold::so3::SO3;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DVector, Matrix2xX, Vector2, Vector3};

// CSV output
use csv::{Reader, Writer};
use serde::{Deserialize, Serialize};

/// Bundle Adjustment Benchmark Result
#[derive(Debug, Clone, Serialize)]
struct BABenchmarkResult {
    dataset: String,
    solver: String,
    language: String,
    num_cameras: usize,
    num_points: usize,
    num_observations: usize,
    initial_mse: String,
    final_mse: String,
    initial_rmse: String,
    final_rmse: String,
    time_ms: String,
    iterations: String,
    status: String,
}

impl BABenchmarkResult {
    #[allow(clippy::too_many_arguments)]
    fn success(
        dataset: &str,
        solver: &str,
        language: &str,
        num_cameras: usize,
        num_points: usize,
        num_observations: usize,
        initial_mse: f64,
        final_mse: f64,
        initial_rmse: f64,
        final_rmse: f64,
        time_ms: f64,
        iterations: usize,
        status: &str,
    ) -> Self {
        Self {
            dataset: dataset.to_string(),
            solver: solver.to_string(),
            language: language.to_string(),
            num_cameras,
            num_points,
            num_observations,
            initial_mse: format!("{:.6e}", initial_mse),
            final_mse: format!("{:.6e}", final_mse),
            initial_rmse: format!("{:.6}", initial_rmse),
            final_rmse: format!("{:.6}", final_rmse),
            time_ms: format!("{:.2}", time_ms),
            iterations: iterations.to_string(),
            status: status.to_string(),
        }
    }

    fn failed(dataset: &str, solver: &str, language: &str, error: &str) -> Self {
        Self {
            dataset: dataset.to_string(),
            solver: solver.to_string(),
            language: language.to_string(),
            num_cameras: 0,
            num_points: 0,
            num_observations: 0,
            initial_mse: "-".to_string(),
            final_mse: "-".to_string(),
            initial_rmse: "-".to_string(),
            final_rmse: "-".to_string(),
            time_ms: "-".to_string(),
            iterations: format!("error: {}", error),
            status: "FAILED".to_string(),
        }
    }
}

/// Check if Apex Solver converged
fn is_converged(status: &OptimizationStatus) -> bool {
    matches!(
        status,
        OptimizationStatus::Converged
            | OptimizationStatus::CostToleranceReached
            | OptimizationStatus::GradientToleranceReached
            | OptimizationStatus::ParameterToleranceReached
    )
}

fn apex_solver_ba(dataset_path: &str) -> BABenchmarkResult {
    info!("\n=== Apex Solver Benchmark ===");

    // Load dataset
    info!("Loading BAL dataset from {}", dataset_path);
    let dataset = match BalLoader::load(dataset_path) {
        Ok(d) => d,
        Err(e) => {
            error!("Failed to load BAL dataset: {}", e);
            return BABenchmarkResult::failed(
                "problem-1723-156502-pre",
                "apex-solver",
                "Rust",
                &e.to_string(),
            );
        }
    };

    info!(
        "Dataset: {} cameras, {} points, {} observations",
        dataset.cameras.len(),
        dataset.points.len(),
        dataset.observations.len()
    );

    // Setup problem
    info!("Building optimization problem...");
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Helper function to convert axis-angle to SO3
    fn axis_angle_to_so3(axis_angle: &Vector3<f64>) -> SO3 {
        let angle = axis_angle.norm();
        if angle < 1e-10 {
            SO3::identity()
        } else {
            let axis = axis_angle / angle;
            SO3::from_axis_angle(&axis, angle)
        }
    }

    // Add cameras as SE3 poses + intrinsic variables
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

    // Add landmarks as R3 variables
    for (j, point) in dataset.points.iter().enumerate() {
        let var_name = format!("pt_{:05}", j);
        let point_vec =
            DVector::from_vec(vec![point.position.x, point.position.y, point.position.z]);
        initial_values.insert(var_name, (ManifoldType::RN, point_vec));
    }

    // Add projection factors using ProjectionFactor with SE3 + BALPinholeCameraStrict
    info!(
        "Adding {} projection factors (SE3 + BALPinholeCameraStrict)...",
        dataset.observations.len()
    );
    for obs in &dataset.observations {
        let cam = &dataset.cameras[obs.camera_index];
        let camera = BALPinholeCameraStrict::new(cam.focal_length, cam.k1, cam.k2);

        // Single observation per factor
        let observations = Matrix2xX::from_columns(&[Vector2::new(obs.x, obs.y)]);
        let factor: ProjectionFactor<BALPinholeCameraStrict, BundleAdjustment> =
            ProjectionFactor::new(observations, camera);

        let pose_name = format!("pose_{:04}", obs.camera_index);
        let pt_name = format!("pt_{:05}", obs.point_index);

        // Use Huber loss (matching C++ implementations)
        let loss = match HuberLoss::new(1.0) {
            Ok(l) => Box::new(l),
            Err(_) => continue,
        };
        problem.add_residual_block(&[&pose_name, &pt_name], Box::new(factor), Some(loss));
    }

    // Fix first camera pose (gauge freedom) - all 6 DOF
    info!("Fixing first camera pose for gauge freedom...");
    for dof in 0..6 {
        problem.fix_variable("pose_0000", dof);
    }
    // Also fix first camera intrinsics
    for dof in 0..3 {
        problem.fix_variable("intr_0000", dof);
    }

    // Configure solver using BA-optimized preset (matches Ceres settings)
    info!("Configuring solver...");
    let config = LevenbergMarquardtConfig::for_bundle_adjustment();

    info!("Solver configuration (BA-optimized preset):");
    info!("  Linear solver: {:?}", config.linear_solver_type);
    info!("  Schur variant: {:?}", config.schur_variant);
    info!("  Preconditioner: {:?}", config.schur_preconditioner);
    info!("  Initial damping: {:e}", config.damping);
    info!("  Max iterations: {}", config.max_iterations);
    info!("  Cost tolerance: {:e}", config.cost_tolerance);
    info!("  Parameter tolerance: {:e}", config.parameter_tolerance);

    let mut solver = LevenbergMarquardt::with_config(config);

    // Optimize (timing excludes setup)
    info!("\nStarting optimization...");
    let start = Instant::now();
    let result = match solver.optimize(&problem, &initial_values) {
        Ok(r) => r,
        Err(e) => {
            error!("Optimization failed: {}", e);
            return BABenchmarkResult::failed(
                "problem-1723-156502-pre",
                "apex-solver",
                "Rust",
                &e.to_string(),
            );
        }
    };
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    info!("\nOptimization completed!");
    info!("Status: {:?}", result.status);
    info!("Iterations: {}", result.iterations);
    info!("Time: {:.2} seconds", elapsed_ms / 1000.0);

    // Compute initial and final RMSE from solver costs
    // Cost = sum of squared residuals, RMSE = sqrt(cost / num_observations)
    let num_obs = dataset.observations.len() as f64;
    let initial_mse = result.initial_cost / num_obs;
    let initial_rmse = initial_mse.sqrt();
    let final_mse = result.final_cost / num_obs;
    let final_rmse = final_mse.sqrt();

    info!("\nMetrics:");
    info!("  Initial cost: {:.6e}", result.initial_cost);
    info!("  Final cost: {:.6e}", result.final_cost);
    info!("  Initial RMSE: {:.3} pixels", initial_rmse);
    info!("  Final RMSE: {:.3} pixels", final_rmse);

    // Extract optimized points (cameras not extracted since format changed)
    let mut final_points: Vec<Vector3<f64>> = dataset.points.iter().map(|p| p.position).collect();

    // Update from optimized values
    for (var_name, var_enum) in &result.parameters {
        if let Some(id_str) = var_name.strip_prefix("pt_")
            && let Ok(id) = id_str.parse::<usize>()
            && id < final_points.len()
        {
            let val = var_enum.to_vector();
            if val.len() >= 3 {
                final_points[id] = Vector3::new(val[0], val[1], val[2]);
            }
        }
    }

    // Build observations tuple for raw RMSE (currently unused)
    let _obs_tuples: Vec<(usize, usize, f64, f64)> = dataset
        .observations
        .iter()
        .map(|o| (o.camera_index, o.point_index, o.x, o.y))
        .collect();

    // Compute raw RMSE (matching Ceres calculation - no loss function)
    let _initial_cameras: Vec<DVector<f64>> = dataset
        .cameras
        .iter()
        .map(|cam| {
            DVector::from_vec(vec![
                cam.rotation.x,
                cam.rotation.y,
                cam.rotation.z,
                cam.translation.x,
                cam.translation.y,
                cam.translation.z,
                cam.focal_length,
                cam.k1,
                cam.k2,
            ])
        })
        .collect();
    let _initial_points: Vec<Vector3<f64>> = dataset.points.iter().map(|p| p.position).collect();

    // PERF: Skip raw RMSE computation to avoid 678K extra factor evaluations
    // let raw_initial_rmse =
    //     compute_raw_rmse_from_cameras(&initial_cameras, &initial_points, &obs_tuples);
    // let raw_final_rmse = compute_raw_rmse_from_cameras(&final_cameras, &final_points, &obs_tuples);
    //
    // info!("\nRaw RMSE (matching Ceres, no loss):");
    // info!("  Initial raw RMSE: {:.3} pixels", raw_initial_rmse);
    // info!("  Final raw RMSE: {:.3} pixels", raw_final_rmse);

    let improvement_pct = ((initial_mse - final_mse) / initial_mse) * 100.0;
    let converged = is_converged(&result.status);

    info!("  Improvement: {:.2}%", improvement_pct);
    info!("  Converged: {}", converged);

    BABenchmarkResult::success(
        "problem-1723-156502-pre",
        "apex-solver",
        "Rust",
        dataset.cameras.len(),
        dataset.points.len(),
        dataset.observations.len(),
        initial_mse,
        final_mse,
        initial_rmse,
        final_rmse,
        elapsed_ms,
        result.iterations,
        if converged {
            "CONVERGED"
        } else {
            "NOT_CONVERGED"
        },
    )
}

/// C++ BA benchmark result from CSV
#[derive(Debug, Deserialize)]
struct CppBAResult {
    dataset: String,
    solver: String,
    language: String,
    num_cameras: usize,
    num_points: usize,
    num_observations: usize,
    initial_mse: f64,
    final_mse: f64,
    initial_rmse: f64,
    final_rmse: f64,
    time_ms: f64,
    iterations: usize,
    status: String,
}

/// Build C++ benchmarks if not already built
fn build_cpp_benchmarks() -> Result<PathBuf, String> {
    // Use CARGO_MANIFEST_DIR to get absolute path to project root
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let bench_dir = Path::new(&manifest_dir).join("benches/cpp_comparison");
    let build_dir = bench_dir.join("build");

    // Check if executables already exist
    let ceres_exe = build_dir.join("ceres_ba_benchmark");
    let gtsam_exe = build_dir.join("gtsam_ba_benchmark");
    let g2o_exe = build_dir.join("g2o_ba_benchmark");

    if ceres_exe.exists() && gtsam_exe.exists() && g2o_exe.exists() {
        info!("C++ BA benchmarks already built");
        return Ok(build_dir);
    }

    info!("Building C++ BA benchmarks...");

    // Create build directory if needed
    std::fs::create_dir_all(&build_dir)
        .map_err(|e| format!("Failed to create build dir: {}", e))?;

    // Run CMake configure
    let cmake_output = Command::new("cmake")
        .args(["..", "-DCMAKE_BUILD_TYPE=Release"])
        .current_dir(&build_dir)
        .output()
        .map_err(|e| format!("Failed to run cmake: {}", e))?;

    if !cmake_output.status.success() {
        return Err(format!(
            "CMake configure failed: {}",
            String::from_utf8_lossy(&cmake_output.stderr)
        ));
    }

    // Run CMake build
    let build_output = Command::new("cmake")
        .args(["--build", ".", "--config", "Release", "-j"])
        .current_dir(&build_dir)
        .output()
        .map_err(|e| format!("Failed to run cmake build: {}", e))?;

    if !build_output.status.success() {
        return Err(format!(
            "CMake build failed: {}",
            String::from_utf8_lossy(&build_output.stderr)
        ));
    }

    info!("C++ BA benchmarks built successfully");
    Ok(build_dir)
}

/// Run a C++ benchmark executable and return path to CSV output
fn run_cpp_benchmark(
    exe_name: &str,
    build_dir: &Path,
    dataset_path: &str,
) -> Result<PathBuf, String> {
    let exe_path = build_dir.join(exe_name);

    if !exe_path.exists() {
        return Err(format!("Executable not found: {:?}", exe_path));
    }

    info!("Running {} ...", exe_name);

    let output = Command::new(&exe_path)
        .arg(dataset_path)
        .current_dir(build_dir)
        .output()
        .map_err(|e| format!("Failed to run {}: {}", exe_name, e))?;

    if !output.status.success() {
        return Err(format!(
            "{} failed: {}",
            exe_name,
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Print stdout for user visibility
    if !output.stdout.is_empty() {
        info!("{}", String::from_utf8_lossy(&output.stdout));
    }

    // Determine CSV output filename
    let csv_name = format!("{}_results.csv", exe_name);
    let csv_path = build_dir.join(&csv_name);

    if !csv_path.exists() {
        return Err(format!("CSV output not found: {:?}", csv_path));
    }

    Ok(csv_path)
}

/// Parse C++ benchmark CSV results
fn parse_cpp_ba_results(csv_path: &Path) -> Result<Vec<BABenchmarkResult>, String> {
    let mut reader =
        Reader::from_path(csv_path).map_err(|e| format!("Failed to read CSV: {}", e))?;

    let mut results = Vec::new();

    for record in reader.deserialize() {
        let cpp_result: CppBAResult =
            record.map_err(|e| format!("Failed to parse CSV record: {}", e))?;

        let result = BABenchmarkResult::success(
            &cpp_result.dataset,
            &cpp_result.solver,
            &cpp_result.language,
            cpp_result.num_cameras,
            cpp_result.num_points,
            cpp_result.num_observations,
            cpp_result.initial_mse,
            cpp_result.final_mse,
            cpp_result.initial_rmse,
            cpp_result.final_rmse,
            cpp_result.time_ms,
            cpp_result.iterations,
            &cpp_result.status,
        );

        results.push(result);
    }

    Ok(results)
}

/// Run all C++ benchmarks
fn run_cpp_ba_benchmarks(dataset_path: &str) -> Vec<BABenchmarkResult> {
    let mut all_results = Vec::new();

    // Try to build C++ benchmarks
    let build_dir = match build_cpp_benchmarks() {
        Ok(dir) => dir,
        Err(e) => {
            warn!("C++ benchmarks unavailable: {}", e);
            warn!("Continuing with Rust-only benchmark...\n");
            return all_results;
        }
    };

    // Convert to absolute path
    let abs_dataset_path = std::fs::canonicalize(dataset_path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| dataset_path.to_string());

    // List of C++ benchmark executables to run
    let cpp_benchmarks = vec![
        "ceres_ba_benchmark",
        "gtsam_ba_benchmark",
        "g2o_ba_benchmark",
    ];

    for exe_name in cpp_benchmarks {
        match run_cpp_benchmark(exe_name, &build_dir, &abs_dataset_path) {
            Ok(csv_path) => match parse_cpp_ba_results(&csv_path) {
                Ok(results) => {
                    info!("{} completed: {} results", exe_name, results.len());
                    all_results.extend(results);
                }
                Err(e) => {
                    warn!("Failed to parse {} results: {}", exe_name, e);
                }
            },
            Err(e) => {
                warn!("Failed to run {}: {}", exe_name, e);
            }
        }
    }

    all_results
}

/// Save benchmark results to CSV
fn save_csv_results(
    results: &[BABenchmarkResult],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = Writer::from_path(path)?;
    for result in results {
        writer.serialize(result)?;
    }
    writer.flush()?;
    Ok(())
}

/// Print comparison table
fn print_comparison_table(results: &[BABenchmarkResult]) {
    info!("\n{}", "=".repeat(160));
    info!("BUNDLE ADJUSTMENT COMPARISON");
    info!("{}", "=".repeat(160));
    info!(
        "{:<25} {:<15} {:<8} {:<10} {:<10} {:<10} {:<12} {:<12} {:<8} {:<12} {:<10}",
        "Dataset",
        "Solver",
        "Language",
        "Cameras",
        "Points",
        "Obs",
        "Init RMSE",
        "Final RMSE",
        "Iters",
        "Time (s)",
        "Status"
    );
    info!("{}", "-".repeat(160));

    for result in results {
        let time_s = result
            .time_ms
            .parse::<f64>()
            .map(|t| format!("{:.2}", t / 1000.0))
            .unwrap_or_else(|_| result.time_ms.clone());

        info!(
            "{:<25} {:<15} {:<8} {:<10} {:<10} {:<10} {:<12} {:<12} {:<8} {:<12} {:<10}",
            result.dataset,
            result.solver,
            result.language,
            result.num_cameras,
            result.num_points,
            result.num_observations,
            result.initial_rmse,
            result.final_rmse,
            result.iterations,
            time_s,
            result.status
        );
    }

    info!("{}", "=".repeat(160));
}

fn main() {
    // Initialize logger
    init_logger();

    info!("BUNDLE ADJUSTMENT BENCHMARK");
    info!("Testing dataset: problem-1723-156502-pre.txt");
    info!("Running each solver once (C++ benchmarks are already averaged)...\n");

    let dataset_path = "data/bundle_adjustment/problem-1723-156502-pre.txt";
    let mut all_results = Vec::new();

    // Phase 1: Apex Solver (Rust)
    info!("========================================");
    info!("PHASE 1: Apex Solver (Rust)");
    info!("========================================");

    let apex_result = apex_solver_ba(dataset_path);
    all_results.push(apex_result);

    // Phase 2: C++ Solvers
    info!("\n========================================");
    info!("PHASE 2: C++ Solvers (Ceres, GTSAM, g2o)");
    info!("========================================");

    let cpp_results = run_cpp_ba_benchmarks(dataset_path);
    all_results.extend(cpp_results);

    // Save and display results
    let csv_path = "ba_benchmark_results.csv";
    if let Err(e) = save_csv_results(&all_results, csv_path) {
        warn!("Warning: Failed to save CSV results: {}", e);
    } else {
        info!("\nResults written to {}", csv_path);
    }

    print_comparison_table(&all_results);

    info!("\nBenchmark completed!");
}
