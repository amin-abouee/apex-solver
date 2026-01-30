//! Bundle Adjustment Benchmark Comparison
//!
//! Compares Apex Solver (Iterative Schur) against Ceres, GTSAM, and g2o
//! on 4 standard BAL datasets: Ladybug, Trafalgar, Dubrovnik, Venice.
//!
//! This benchmark tests convergence and performance on large-scale bundle adjustment
//! problems across multiple solvers.
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench bundle_adjustment_comparison
//! ```
//!
//! ## Datasets Tested
//!
//! - **Ladybug**: 89 cameras, 110,973 landmarks, 562,976 observations
//! - **Trafalgar**: 257 cameras, 65,132 landmarks, 225,911 observations
//! - **Dubrovnik**: 356 cameras, 226,730 landmarks, 1,255,268 observations
//! - **Venice**: 1778 cameras, 993,923 landmarks, 5,001,946 observations
//!
//! ## Solvers Compared
//!
//! - **Apex (Iterative Schur)**: PCG with Schur-Jacobi preconditioner, SelfCalibration mode
//! - **Ceres**: Google's sparse nonlinear least squares solver
//! - **GTSAM**: Georgia Tech Smoothing and Mapping
//! - **g2o**: General Graph Optimization
//!
//! ## Metrics
//! - Initial/Final MSE (Mean Squared Error in pixels²)
//! - Initial/Final RMSE (Root Mean Squared Error in pixels)
//! - Runtime in seconds (optimization only, excludes parsing)
//! - Number of iterations
//! - Convergence status

use criterion::{Criterion, criterion_group, criterion_main};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

/// Timeout duration for each solver (10 minutes)
const SOLVER_TIMEOUT: Duration = Duration::from_secs(600);

// apex-solver imports
use apex_camera_models::{BALPinholeCameraStrict, SelfCalibration};
use apex_io::BalLoader;
use apex_manifolds::se3::SE3;
use apex_manifolds::so3::SO3;
use apex_solver::ManifoldType;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::ProjectionFactor;
use apex_solver::init_logger;
use apex_solver::linalg::{LinearSolverType, SchurPreconditioner, SchurVariant};
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DVector, Matrix2xX, Vector2, Vector3};

// CSV output
use csv::{Reader, Writer};
use serde::{Deserialize, Serialize};

/// Dataset configuration
#[derive(Debug, Clone)]
struct DatasetConfig {
    name: String,
    path: String,
}

/// Get all datasets to benchmark
fn get_datasets() -> Vec<DatasetConfig> {
    vec![
        DatasetConfig {
            name: "Ladybug".to_string(),
            path: "../../data/bundle_adjustment/Ladybug/problem-1723-156502-pre.txt".to_string(),
        },
        DatasetConfig {
            name: "Trafalgar".to_string(),
            path: "../../data/bundle_adjustment/Trafalgar/problem-257-65132-pre.txt".to_string(),
        },
        DatasetConfig {
            name: "Dubrovnik".to_string(),
            path: "../../data/bundle_adjustment/Dubrovnik/problem-356-226730-pre.txt".to_string(),
        },
        DatasetConfig {
            name: "Venice".to_string(),
            path: "../../data/bundle_adjustment/Venice/problem-1778-993923-pre.txt".to_string(),
        },
    ]
}

/// Bundle Adjustment Benchmark Result
#[derive(Debug, Clone, Serialize)]
struct BABenchmarkResult {
    dataset: String,
    solver: String,
    language: String,
    num_cameras: usize,
    num_points: usize,
    num_observations: usize,
    initial_rmse: String,
    final_rmse: String,
    time_seconds: String,
    iterations: String,
    status: String,
}

impl BABenchmarkResult {
    /// Create a successful benchmark result.
    ///
    /// # Design Note
    /// This constructor accepts individual benchmark metrics for clear parameter naming in benchmark code.
    /// The large parameter count reflects the comprehensive nature of bundle adjustment benchmarking.
    #[allow(clippy::too_many_arguments)]
    fn success(
        dataset_name: &str,
        solver: &str,
        language: &str,
        num_cameras: usize,
        num_points: usize,
        num_observations: usize,
        initial_rmse: f64,
        final_rmse: f64,
        time_seconds: f64,
        iterations: usize,
        status: &str,
    ) -> Self {
        Self {
            dataset: dataset_name.to_string(),
            solver: solver.to_string(),
            language: language.to_string(),
            num_cameras,
            num_points,
            num_observations,
            initial_rmse: format!("{:.6}", initial_rmse),
            final_rmse: format!("{:.6}", final_rmse),
            time_seconds: format!("{:.2}", time_seconds),
            iterations: iterations.to_string(),
            status: status.to_string(),
        }
    }

    fn failed(dataset_name: &str, solver: &str, language: &str, error: &str) -> Self {
        Self {
            dataset: dataset_name.to_string(),
            solver: solver.to_string(),
            language: language.to_string(),
            num_cameras: 0,
            num_points: 0,
            num_observations: 0,
            initial_rmse: "-".to_string(),
            final_rmse: "-".to_string(),
            time_seconds: "-".to_string(),
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

/// Run Apex Solver bundle adjustment with SelfCalibration + Iterative Schur
fn apex_solver_ba(dataset_name: &str, dataset_path: &str) -> BABenchmarkResult {
    info!("Apex Solver Benchmark ({})", dataset_name);

    // Run solver in separate thread with timeout
    let dataset_name_owned = dataset_name.to_string();
    let dataset_path_owned = dataset_path.to_string();

    let handle =
        thread::spawn(move || apex_solver_ba_impl(&dataset_name_owned, &dataset_path_owned));

    // Wait for completion with timeout
    let start = Instant::now();
    loop {
        if start.elapsed() >= SOLVER_TIMEOUT {
            let timeout_mins = SOLVER_TIMEOUT.as_secs() / 60;
            error!(
                "Apex solver TIMEOUT EXCEEDED ({} minutes) for {}",
                timeout_mins, dataset_name
            );
            return BABenchmarkResult::failed(
                dataset_name,
                "Apex-Iterative",
                "Rust",
                &format!("TIMEOUT ({} minutes)", timeout_mins),
            );
        }

        // Check if thread completed
        if handle.is_finished() {
            return handle.join().unwrap_or_else(|_| {
                BABenchmarkResult::failed(dataset_name, "Apex-Iterative", "Rust", "Thread panicked")
            });
        }

        // Sleep briefly to avoid busy-waiting
        thread::sleep(Duration::from_millis(100));
    }
}

/// Implementation of Apex Solver BA (runs in separate thread)
fn apex_solver_ba_impl(dataset_name: &str, dataset_path: &str) -> BABenchmarkResult {
    // Load dataset
    info!("Loading BAL dataset from {}", dataset_path);
    let dataset = match BalLoader::load(dataset_path) {
        Ok(d) => d,
        Err(e) => {
            error!("Failed to load BAL dataset: {}", e);
            return BABenchmarkResult::failed(
                dataset_name,
                "Apex-Iterative",
                "Rust",
                &e.to_string(),
            );
        }
    };

    info!(
        "Dataset: {}: Cameras: {}, Landmarks: {}, Observations: {}",
        dataset_name,
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

    // Add cameras as SE3 poses
    for (i, cam) in dataset.cameras.iter().enumerate() {
        // Convert axis-angle to SE3
        let axis_angle = Vector3::new(cam.rotation.x, cam.rotation.y, cam.rotation.z);
        let translation = Vector3::new(cam.translation.x, cam.translation.y, cam.translation.z);
        let so3 = axis_angle_to_so3(&axis_angle);
        let pose = SE3::from_translation_so3(translation, so3);

        // Add SE3 pose variable (6 DOF)
        let pose_name = format!("pose_{:04}", i);
        initial_values.insert(pose_name, (ManifoldType::SE3, DVector::from(pose)));

        // Add intrinsics: [focal, k1, k2] (3 DOF) for SelfCalibration mode
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
    // SelfCalibration mode: optimize pose + landmarks + intrinsics
    info!(
        "Adding {} projection factors (SelfCalibration mode)...",
        dataset.observations.len()
    );
    for obs in &dataset.observations {
        let cam = &dataset.cameras[obs.camera_index];
        let camera = BALPinholeCameraStrict::new(cam.focal_length, cam.k1, cam.k2);

        // Single observation per factor
        let observations = Matrix2xX::from_columns(&[Vector2::new(obs.x, obs.y)]);
        let factor: ProjectionFactor<BALPinholeCameraStrict, SelfCalibration> =
            ProjectionFactor::new(observations, camera);

        let pose_name = format!("pose_{:04}", obs.camera_index);
        let intr_name = format!("intr_{:04}", obs.camera_index);
        let pt_name = format!("pt_{:05}", obs.point_index);

        // Use Huber loss (matching C++ implementations)
        let loss = match HuberLoss::new(1.0) {
            Ok(l) => Box::new(l),
            Err(_) => continue,
        };
        problem.add_residual_block(
            &[&pose_name, &pt_name, &intr_name],
            Box::new(factor),
            Some(loss),
        );
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

    // Configure solver with Iterative Schur + Schur-Jacobi preconditioner
    info!("Configuring solver...");
    let config = LevenbergMarquardtConfig::new()
        .with_linear_solver_type(LinearSolverType::SparseSchurComplement)
        .with_max_iterations(50)
        .with_cost_tolerance(1e-6)
        .with_parameter_tolerance(1e-8)
        .with_damping(1e-3)
        .with_schur_variant(SchurVariant::Iterative)
        .with_schur_preconditioner(SchurPreconditioner::SchurJacobi);

    info!("Solver configuration:");
    info!("  Mode: SelfCalibration (pose + landmarks + intrinsics)");
    info!("  Linear solver: {:?}", config.linear_solver_type);
    info!("  Schur variant: {:?}", config.schur_variant);
    info!("  Preconditioner: {:?}", config.schur_preconditioner);
    info!("  Initial damping: {:e}", config.damping);
    info!("  Max iterations: {}", config.max_iterations);
    info!("  Cost tolerance: {:e}", config.cost_tolerance);
    info!("  Parameter tolerance: {:e}", config.parameter_tolerance);

    let mut solver = LevenbergMarquardt::with_config(config);

    // Optimize (timing excludes setup)
    info!("Starting optimization...");
    let start = Instant::now();
    let result = match solver.optimize(&problem, &initial_values) {
        Ok(r) => r,
        Err(e) => {
            error!("Optimization failed: {}", e);
            return BABenchmarkResult::failed(
                dataset_name,
                "Apex-Iterative",
                "Rust",
                &e.to_string(),
            );
        }
    };
    let elapsed_seconds = start.elapsed().as_secs_f64();

    info!("Optimization completed!");
    info!("Status: {:?}", result.status);
    info!("Iterations: {}", result.iterations);
    info!("Time: {:.2} seconds", elapsed_seconds);

    // Compute initial and final RMSE from solver costs
    // Cost = sum of squared residuals, RMSE = sqrt(cost / num_observations)
    let num_obs = dataset.observations.len() as f64;
    let initial_mse = result.initial_cost / num_obs;
    let initial_rmse = initial_mse.sqrt();
    let final_mse = result.final_cost / num_obs;
    let final_rmse = final_mse.sqrt();

    info!("Metrics:");
    info!("  Initial cost: {:.6e}", result.initial_cost);
    info!("  Final cost: {:.6e}", result.final_cost);
    info!("  Initial RMSE: {:.3} pixels", initial_rmse);
    info!("  Final RMSE: {:.3} pixels", final_rmse);

    let improvement_pct = ((initial_mse - final_mse) / initial_mse) * 100.0;
    let converged = is_converged(&result.status);

    info!("  Improvement: {:.2}%", improvement_pct);
    info!("  Converged: {}", converged);

    BABenchmarkResult::success(
        dataset_name,
        "Apex-Iterative",
        "Rust",
        dataset.cameras.len(),
        dataset.points.len(),
        dataset.observations.len(),
        initial_rmse,
        final_rmse,
        elapsed_seconds,
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
#[allow(dead_code)] // Fields needed for CSV deserialization
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

    // Spawn process (non-blocking)
    let mut child = Command::new(&exe_path)
        .arg(dataset_path)
        .current_dir(build_dir)
        .spawn()
        .map_err(|e| format!("Failed to spawn {}: {}", exe_name, e))?;

    // Monitor process with timeout
    let start = Instant::now();
    loop {
        // Check if timeout exceeded
        if start.elapsed() >= SOLVER_TIMEOUT {
            let timeout_mins = SOLVER_TIMEOUT.as_secs() / 60;
            error!(
                "{} TIMEOUT EXCEEDED ({} minutes), killing process",
                exe_name, timeout_mins
            );
            let _ = child.kill();
            let _ = child.wait(); // Clean up zombie process
            return Err(format!("TIMEOUT ({} minutes)", timeout_mins));
        }

        // Check if process completed
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return Err(format!(
                        "{} failed with exit code: {:?}",
                        exe_name,
                        status.code()
                    ));
                }
                break; // Process completed successfully
            }
            Ok(None) => {
                // Still running, sleep briefly
                thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                return Err(format!("Error waiting for {}: {}", exe_name, e));
            }
        }
    }

    // Process completed, read output
    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to get output from {}: {}", exe_name, e))?;

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
fn parse_cpp_ba_results(
    csv_path: &Path,
    dataset_name: &str,
) -> Result<Vec<BABenchmarkResult>, String> {
    let mut reader =
        Reader::from_path(csv_path).map_err(|e| format!("Failed to read CSV: {}", e))?;

    let mut results = Vec::new();

    for record in reader.deserialize() {
        let cpp_result: CppBAResult =
            record.map_err(|e| format!("Failed to parse CSV record: {}", e))?;

        // Use the passed dataset_name instead of extracting from CSV
        let result = BABenchmarkResult::success(
            dataset_name,
            &cpp_result.solver,
            &cpp_result.language,
            cpp_result.num_cameras,
            cpp_result.num_points,
            cpp_result.num_observations,
            cpp_result.initial_rmse,
            cpp_result.final_rmse,
            cpp_result.time_ms / 1000.0, // Convert ms to seconds
            cpp_result.iterations,
            &cpp_result.status,
        );

        results.push(result);
    }

    Ok(results)
}

/// Run all C++ benchmarks for a given dataset
fn run_cpp_ba_benchmarks(dataset_name: &str, dataset_path: &str) -> Vec<BABenchmarkResult> {
    let mut all_results = Vec::new();

    // Try to build C++ benchmarks
    let build_dir = match build_cpp_benchmarks() {
        Ok(dir) => dir,
        Err(e) => {
            warn!("C++ benchmarks unavailable for {}: {}", dataset_name, e);
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
            Ok(csv_path) => match parse_cpp_ba_results(&csv_path, dataset_name) {
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
                // Create timeout result if error contains "TIMEOUT"
                if e.contains("TIMEOUT") {
                    // Extract solver name from exe_name (e.g., "ceres_ba_benchmark" -> "Ceres")
                    let solver_name = exe_name.replace("_ba_benchmark", "");
                    let solver_name = solver_name
                        .split('_')
                        .map(|s| {
                            let mut c = s.chars();
                            match c.next() {
                                None => String::new(),
                                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("");

                    let timeout_result =
                        BABenchmarkResult::failed(dataset_name, &solver_name, "C++", &e);
                    all_results.push(timeout_result);
                }
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

/// Print comparison table grouped by dataset
fn print_comparison_table(results: &[BABenchmarkResult]) {
    info!("{}", "=".repeat(150));
    info!("BUNDLE ADJUSTMENT COMPARISON RESULTS");
    info!("{}", "=".repeat(150));

    // Group results by dataset
    let mut results_by_dataset: HashMap<String, Vec<&BABenchmarkResult>> = HashMap::new();
    for result in results {
        results_by_dataset
            .entry(result.dataset.clone())
            .or_default()
            .push(result);
    }

    // Sort dataset names
    let mut dataset_names: Vec<String> = results_by_dataset.keys().cloned().collect();
    dataset_names.sort();

    for dataset_name in dataset_names {
        let dataset_results = &results_by_dataset[&dataset_name];

        if let Some(first_result) = dataset_results.first() {
            // Print dataset info on one line (use first non-failed result for counts)
            let info_result = dataset_results
                .iter()
                .find(|r| r.num_cameras > 0)
                .unwrap_or(first_result);

            info!(
                "Dataset: {}: Cameras: {}, Landmarks: {}, Observations: {}",
                dataset_name,
                info_result.num_cameras,
                info_result.num_points,
                info_result.num_observations
            );
            info!("{}", "-".repeat(150));
            info!(
                "{:<20} {:<10} {:<15} {:<15} {:<15} {:<10} {:<12}",
                "Solver", "Language", "Initial RMSE", "Final RMSE", "Time (s)", "Iters", "Status"
            );
            info!("{}", "-".repeat(150));

            for result in dataset_results {
                info!(
                    "{:<20} {:<10} {:<15} {:<15} {:<15} {:<10} {:<12}",
                    result.solver,
                    result.language,
                    result.initial_rmse,
                    result.final_rmse,
                    result.time_seconds,
                    result.iterations,
                    result.status
                );
            }

            // Add empty line between datasets
            info!("");
        }
    }

    info!("{}", "=".repeat(150));
}

/// Run the full benchmark comparison
fn run_benchmark_comparison() {
    // Initialize logger only once (avoid panic on multiple calls)
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_logger();
    });

    info!("BUNDLE ADJUSTMENT BENCHMARK COMPARISON");
    info!("Testing 4 datasets: Ladybug, Trafalgar, Dubrovnik, Venice");
    info!("Apex Solver: SelfCalibration mode + Iterative Schur + Schur-Jacobi preconditioner\n");

    let datasets = get_datasets();
    let mut all_results = Vec::new();

    // Run benchmarks for each dataset
    for dataset in &datasets {
        info!("{}", "=".repeat(150));
        info!("DATASET: {}", dataset.name);
        info!("PATH: {}", dataset.path);
        info!("{}", "=".repeat(150));

        // Verify dataset file exists
        if !Path::new(&dataset.path).exists() {
            error!("Dataset file not found: {}", dataset.path);
            error!("Skipping {}...", dataset.name);
            continue;
        }

        // Phase 1: Apex Solver (Rust)
        info!("Phase 1: Apex Solver");
        let apex_result = apex_solver_ba(&dataset.name, &dataset.path);
        all_results.push(apex_result);

        // Phase 2: C++ Solvers
        info!("Phase 2: C++ Solvers (Ceres, GTSAM, g2o)");
        let cpp_results = run_cpp_ba_benchmarks(&dataset.name, &dataset.path);
        all_results.extend(cpp_results);
    }

    // Save results to CSV in output/ folder
    let output_dir = "output";
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        warn!("Warning: Failed to create output directory: {}", e);
    }

    let output_path = format!("{}/ba_comparison_results.csv", output_dir);
    if let Err(e) = save_csv_results(&all_results, &output_path) {
        warn!("Warning: Failed to save CSV results: {}", e);
    } else {
        info!("Results written to {}", output_path);
    }

    // Print comparison table
    print_comparison_table(&all_results);

    info!("\nBenchmark completed!");
}

/// Criterion benchmark function
fn criterion_benchmark(_c: &mut Criterion) {
    // This is a comparison benchmark, not a performance benchmark
    // Run once directly instead of using Criterion's timing infrastructure
    run_benchmark_comparison();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
