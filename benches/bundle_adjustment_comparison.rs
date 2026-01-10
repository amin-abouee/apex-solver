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
use tracing::{info, warn, error};

// apex-solver imports
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::camera::BALPinholeCamera;
use apex_solver::factors::ProjectionFactor;
use apex_solver::BundleAdjustment;
use apex_solver::init_logger;
use apex_solver::io::BalLoader;
use apex_solver::linalg::{LinearSolverType, SchurPreconditioner, SchurVariant};
use apex_solver::manifold::{LieGroup, ManifoldType};
use apex_solver::manifold::se3::SE3;
use apex_solver::manifold::so3::SO3;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DVector, Matrix2xX, Vector3};

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

/// Run Apex Solver bundle adjustment benchmark
fn apex_solver_ba(dataset_path: &str) -> BABenchmarkResult {
    info!("\n=== Apex Solver Benchmark ===");

    // Load dataset
    info!("Loading BAL dataset from {}", dataset_path);
    let mut dataset = match BalLoader::load(dataset_path) {
        Ok(d) => d,
        Err(e) => {
            error!("Failed to load BAL dataset: {}", e);
            return BABenchmarkResult::failed("problem-1723-156502-pre", "apex-solver", "Rust", &e.to_string());
        }
    };

    info!("Dataset: {} cameras, {} points, {} observations",
          dataset.cameras.len(), dataset.points.len(), dataset.observations.len());

    // Setup problem
    info!("Building optimization problem...");
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add camera poses as SE3 variables
    // BAL convention: World-to-Camera transformation (P_cam = R * X_world + t)
    for (i, cam) in dataset.cameras.iter().enumerate() {
        let var_name = format!("cam_{:04}", i);

        let r_wc = SO3::from_scaled_axis(cam.rotation);
        let se3_wc = SE3::from_translation_so3(cam.translation, r_wc);

        // Store directly as world-to-camera (ProjectionFactor expects this)
        let pose_vec: DVector<f64> = se3_wc.into();

        initial_values.insert(var_name, (ManifoldType::SE3, pose_vec));
    }

    // Add landmarks as R3 variables
    for (j, point) in dataset.points.iter().enumerate() {
        let var_name = format!("pt_{:05}", j);
        let point_vec = DVector::from_vec(vec![
            point.position.x, point.position.y, point.position.z
        ]);
        initial_values.insert(var_name, (ManifoldType::RN, point_vec));
    }

    // Add projection factors
    info!("Adding {} projection factors...", dataset.observations.len());
    for obs in &dataset.observations {
        let cam = &dataset.cameras[obs.camera_index];

        // Create BAL pinhole camera model
        let camera_model = BALPinholeCamera::new(
            cam.focal_length,
            cam.focal_length,
            0.0,  // u0 = 0 (principal point at origin)
            0.0,  // v0 = 0
            cam.k1,
            cam.k2,
        );

        // Observation as 2x1 matrix
        let observations_mat = Matrix2xX::from_column_slice(&[obs.x, obs.y]);

        // Create projection factor (optimizes pose + landmark)
        let factor = ProjectionFactor::<BALPinholeCamera, BundleAdjustment>::new(
            observations_mat,
            camera_model,
        );

        let cam_name = format!("cam_{:04}", obs.camera_index);
        let pt_name = format!("pt_{:05}", obs.point_index);

        // Use Huber loss (matching C++ implementations)
        let loss = Box::new(HuberLoss::new(1.0).expect("Failed to create Huber loss"));
        problem.add_residual_block(&[&cam_name, &pt_name], Box::new(factor), Some(loss));
    }

    // Fix first camera (gauge freedom)
    info!("Fixing first camera for gauge freedom...");
    for dof in 0..6 {
        problem.fix_variable("cam_0000", dof);
    }

    // Configure solver (schur-iterative as specified)
    info!("Configuring solver...");
    let config = LevenbergMarquardtConfig::new()
        .with_linear_solver_type(LinearSolverType::SparseSchurComplement)
        .with_schur_variant(SchurVariant::Iterative)
        .with_schur_preconditioner(SchurPreconditioner::BlockDiagonal)
        .with_max_iterations(100)
        .with_cost_tolerance(1e-12)
        .with_parameter_tolerance(1e-14);

    info!("Solver configuration:");
    info!("  Linear solver: Schur Complement (Iterative)");
    info!("  Preconditioner: Block Diagonal (Jacobi)");
    info!("  Max iterations: 100");
    info!("  Cost tolerance: 1e-12");
    info!("  Parameter tolerance: 1e-14");

    let mut solver = LevenbergMarquardt::with_config(config);

    // Optimize (timing excludes setup)
    info!("\nStarting optimization...");
    let start = Instant::now();
    let result = match solver.optimize(&problem, &initial_values) {
        Ok(r) => r,
        Err(e) => {
            error!("Optimization failed: {}", e);
            return BABenchmarkResult::failed("problem-1723-156502-pre", "apex-solver", "Rust", &e.to_string());
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

    // Update dataset with optimized values (for potential visualization)
    for (var_name, var_enum) in &result.parameters {
        if let Some(id_str) = var_name.strip_prefix("cam_") {
            if let Ok(id) = id_str.parse::<usize>() {
                if id < dataset.cameras.len() {
                    let val = var_enum.to_vector();
                    // Extract SE3: [tx, ty, tz, qw, qx, qy, qz]
                    let trans = Vector3::new(val[0], val[1], val[2]);
                    let quat = nalgebra::UnitQuaternion::from_quaternion(
                        nalgebra::Quaternion::new(val[3], val[4], val[5], val[6])
                    );
                    let se3 = SE3::from_translation_quaternion(trans, quat.into_inner());

                    // Convert back to axis-angle for camera storage
                    let axis_angle_dv: DVector<f64> = se3.rotation_so3().log(None).into();
                    dataset.cameras[id].rotation = Vector3::new(axis_angle_dv[0], axis_angle_dv[1], axis_angle_dv[2]);
                    dataset.cameras[id].translation = se3.translation();
                }
            }
        } else if let Some(id_str) = var_name.strip_prefix("pt_") {
            if let Ok(id) = id_str.parse::<usize>() {
                if id < dataset.points.len() {
                    let val = var_enum.to_vector();
                    dataset.points[id].position = Vector3::new(val[0], val[1], val[2]);
                }
            }
        }
    }

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
        if converged { "CONVERGED" } else { "NOT_CONVERGED" },
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
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
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
fn run_cpp_benchmark(exe_name: &str, build_dir: &Path, dataset_path: &str) -> Result<PathBuf, String> {
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
    let cpp_benchmarks = vec!["ceres_ba_benchmark", "gtsam_ba_benchmark", "g2o_ba_benchmark"];

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
        "Dataset", "Solver", "Language", "Cameras", "Points", "Obs",
        "Init RMSE", "Final RMSE", "Iters", "Time (s)", "Status"
    );
    info!("{}", "-".repeat(160));

    for result in results {
        let time_s = result.time_ms.parse::<f64>()
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
