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
//! Set `APEX_BENCH_RUST_ONLY=1` to skip the C++ solvers (Ceres, GTSAM, g2o) and run
//! the apex-solver rows only.
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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

/// Timeout duration for each solver (10 minutes)
const SOLVER_TIMEOUT: Duration = Duration::from_secs(600);

// apex-solver imports
use apex_camera_models::{BALPinholeCameraStrict, DistortionModel, PinholeParams};
use apex_io::{BalDataset, BalLoader, utils::DatasetRegistry};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::so3::SO3;
use apex_solver::ManifoldType;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::SelfCalibration;
use apex_solver::factors::visual::ProjectionFactor;
use apex_solver::init_logger;
use apex_solver::linalg::{ExplicitSchurVariant, JacobianMode, LinearSolverType};
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
    let registry =
        DatasetRegistry::load().unwrap_or_else(|e| panic!("failed to load dataset registry: {e}"));
    // (registry_key, display_name, cameras, points)
    let specs = [
        ("ladybug", "Ladybug", 1723u32, 156502u32),
        ("trafalgar", "Trafalgar", 257u32, 65132u32),
        ("dubrovnik", "Dubrovnik", 356u32, 226730u32),
        ("venice", "Venice", 1778u32, 993923u32),
    ];
    specs
        .iter()
        .filter_map(|(key, display, cameras, points)| {
            registry
                .ba_path(key, *cameras, *points)
                .map(|p| DatasetConfig {
                    name: display.to_string(),
                    path: p.to_string_lossy().into_owned(),
                })
        })
        .collect()
}

/// The single dataset the dense-mode row runs: Ladybug, truncated to
/// [`dense_bench_cameras`] cameras by [`build_ba_problem`].
///
/// Named apart from `Ladybug` so its row can never be read as comparable to
/// the full-dataset table in `doc/performance.md`.
fn dense_bench_datasets() -> Vec<DatasetConfig> {
    get_datasets()
        .into_iter()
        .filter(|d| d.name == "Ladybug")
        .map(|d| DatasetConfig {
            name: format!("Ladybug-mini-{}cam", dense_bench_cameras()),
            path: d.path,
        })
        .collect()
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
            | OptimizationStatus::StalledNoProgress
            | OptimizationStatus::ParameterToleranceReached
    )
}

/// Cameras kept from Ladybug for the dense-mode row.
///
/// `ExplicitDenseSchur` forms a *dense* `JᵀJ`, so its problem has to stay in
/// dense mode's target range; the full BAL datasets are ~485k columns and
/// cannot be run this way at all. Cost grows as roughly `rows · cols²`, and
/// Ladybug's landmark count grows sublinearly in cameras, so measured dense
/// runtime is 11 s at 2 cameras, 36 s at 4, and would pass a minute by 8 —
/// 4 buys a representative row without spending the benchmark's whole budget
/// on it. Override with `APEX_BENCH_DENSE_CAMERAS`.
const DENSE_BENCH_CAMERAS: usize = 4;

/// One apex-solver benchmark row: how the problem is built and which linear
/// solver solves it.
///
/// Every field except `solver` is fixed by `APEX_BENCH_SCHUR`; `solver` is the
/// CSV label, so the dense row can carry its sparse reference alongside it.
#[derive(Debug, Clone, Copy)]
struct ApexRun {
    solver: &'static str,
    jacobian_mode: JacobianMode,
    /// Keep only observations made by the first N cameras; `None` uses the
    /// whole dataset.
    max_cameras: Option<usize>,
    linear_solver_type: LinearSolverType,
    /// Ignored unless `linear_solver_type` is `ExplicitSparseSchur`.
    schur_variant: ExplicitSchurVariant,
}

impl ApexRun {
    /// The `for_bundle_adjustment` default: `ExplicitSparseSchur` / `Sparse`
    /// over the full dataset, which is what `doc/performance.md` measured.
    fn default_sparse() -> Self {
        Self {
            solver: "Apex-Solver",
            jacobian_mode: JacobianMode::Sparse,
            max_cameras: None,
            linear_solver_type: LinearSolverType::ExplicitSparseSchur,
            schur_variant: ExplicitSchurVariant::Sparse,
        }
    }
}

/// Number of leading cameras the dense row keeps, from
/// `APEX_BENCH_DENSE_CAMERAS` or [`DENSE_BENCH_CAMERAS`].
fn dense_bench_cameras() -> usize {
    std::env::var("APEX_BENCH_DENSE_CAMERAS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DENSE_BENCH_CAMERAS)
}

/// True when `APEX_BENCH_SCHUR` selects the dense Schur solver, which runs a
/// truncated problem instead of the four BAL datasets.
fn is_dense_schur_bench() -> bool {
    std::env::var("APEX_BENCH_SCHUR").is_ok_and(|v| v == "explicit-dense")
}

/// The apex-solver rows `APEX_BENCH_SCHUR` asks for.
///
/// Unset keeps `for_bundle_adjustment`'s default. `explicit-dense` yields two
/// rows on the same truncated problem — the dense solver plus the sparse
/// explicit solver as its accuracy reference — because a dense row has no
/// counterpart in the full-dataset table to be compared against.
fn apex_runs() -> Vec<ApexRun> {
    let Ok(v) = std::env::var("APEX_BENCH_SCHUR") else {
        return vec![ApexRun::default_sparse()];
    };

    let runs = match v.as_str() {
        "sparse" => vec![ApexRun::default_sparse()],
        "iterative" => vec![ApexRun {
            linear_solver_type: LinearSolverType::ImplicitSparseSchur,
            ..ApexRun::default_sparse()
        }],
        "explicit-iterative" => vec![ApexRun {
            schur_variant: ExplicitSchurVariant::Iterative,
            ..ApexRun::default_sparse()
        }],
        "chunked" => vec![ApexRun {
            schur_variant: ExplicitSchurVariant::Chunked,
            ..ApexRun::default_sparse()
        }],
        "explicit-dense" => {
            let max_cameras = Some(dense_bench_cameras());
            vec![
                ApexRun {
                    solver: "Apex-ExplicitDenseSchur",
                    jacobian_mode: JacobianMode::Dense,
                    max_cameras,
                    linear_solver_type: LinearSolverType::ExplicitDenseSchur,
                    ..ApexRun::default_sparse()
                },
                ApexRun {
                    solver: "Apex-ExplicitSparseSchur",
                    max_cameras,
                    ..ApexRun::default_sparse()
                },
            ]
        }
        other => panic!("APEX_BENCH_SCHUR: unknown variant {other}"),
    };

    for run in &runs {
        info!(
            "APEX_BENCH_SCHUR={v} -> {}: {:?} / {:?} / {:?}",
            run.solver, run.jacobian_mode, run.linear_solver_type, run.schur_variant
        );
    }
    runs
}

/// Run Apex Solver bundle adjustment with SelfCalibration + the Schur solver
/// `run` selects.
fn apex_solver_ba(dataset_name: &str, dataset_path: &str, run: ApexRun) -> BABenchmarkResult {
    info!("Running {} ...", run.solver);

    // Run solver in separate thread with timeout
    let dataset_name_owned = dataset_name.to_string();
    let dataset_path_owned = dataset_path.to_string();

    let handle =
        thread::spawn(move || apex_solver_ba_impl(&dataset_name_owned, &dataset_path_owned, run));

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
                run.solver,
                "Rust",
                &format!("TIMEOUT ({} minutes)", timeout_mins),
            );
        }

        // Check if thread completed
        if handle.is_finished() {
            return handle.join().unwrap_or_else(|_| {
                BABenchmarkResult::failed(dataset_name, run.solver, "Rust", "Thread panicked")
            });
        }

        // Sleep briefly to avoid busy-waiting
        thread::sleep(Duration::from_millis(100));
    }
}

/// Convert a BAL axis-angle rotation to `SO3`.
fn axis_angle_to_so3(axis_angle: &Vector3<f64>) -> SO3 {
    let angle = axis_angle.norm();
    if angle < 1e-10 {
        SO3::identity()
    } else {
        let axis = axis_angle / angle;
        SO3::from_axis_angle(&axis, angle)
    }
}

/// Problem sizes for one benchmark row.
///
/// These are the counts actually added to the problem, not the file's — a
/// truncated row keeps fewer, and RMSE normalizes by observations, so using
/// the file's totals would silently rescale the reported error.
#[derive(Debug, Clone, Copy)]
struct ProblemSize {
    cameras: usize,
    points: usize,
    observations: usize,
}

/// Build the SelfCalibration BA problem: an SE3 pose and 3-parameter
/// intrinsics per camera, an eliminated `Rn` landmark per point, one
/// Huber(1 px) projection factor per observation, camera 0 fixed for gauge
/// freedom.
///
/// `run.max_cameras` keeps only the observations made by the first N cameras
/// and the points those observations reference — the truncation
/// `tests/schur_ba_agreement.rs` already uses — so a dense-mode row can run on
/// a subset whose Hessian actually fits densely. `None` keeps the whole
/// dataset, which is what every sparse row uses.
fn build_ba_problem(dataset: &BalDataset, run: ApexRun) -> Result<(Problem, ProblemSize), String> {
    let num_cameras = run
        .max_cameras
        .unwrap_or(dataset.cameras.len())
        .min(dataset.cameras.len());
    let observations: Vec<_> = dataset
        .observations
        .iter()
        .filter(|o| o.camera_index < num_cameras)
        .collect();

    let mut problem = Problem::new(run.jacobian_mode);

    // Add cameras as SE3 poses plus a 3-parameter intrinsics block each
    let mut pose_keys: Vec<apex_solver::core::VarKey> = Vec::with_capacity(num_cameras);
    let mut intr_keys: Vec<apex_solver::core::VarKey> = Vec::with_capacity(num_cameras);
    for cam in dataset.cameras.iter().take(num_cameras) {
        let axis_angle = Vector3::new(cam.rotation.x, cam.rotation.y, cam.rotation.z);
        let translation = Vector3::new(cam.translation.x, cam.translation.y, cam.translation.z);
        let pose = SE3::from_translation_so3(translation, axis_angle_to_so3(&axis_angle));

        pose_keys.push(problem.add_variable(
            ManifoldType::SE3,
            DVector::from_column_slice(pose.as_param_slice()),
        ));
        intr_keys.push(problem.add_variable(
            ManifoldType::RN,
            DVector::from_vec(vec![cam.focal_length, cam.k1, cam.k2]),
        ));
    }

    // Only points the kept observations reference become variables: an
    // unobserved landmark contributes no residual and would leave its
    // `H_ee` block singular for every Schur solver to trip over.
    let mut point_indices: Vec<usize> = observations.iter().map(|o| o.point_index).collect();
    point_indices.sort_unstable();
    point_indices.dedup();

    let mut pt_keys: HashMap<usize, apex_solver::core::VarKey> =
        HashMap::with_capacity(point_indices.len());
    for index in point_indices {
        let position = &dataset.points[index].position;
        let pt_key = problem.add_variable(
            ManifoldType::RN,
            DVector::from_vec(vec![position.x, position.y, position.z]),
        );
        problem.mark_for_elimination(pt_key);
        pt_keys.insert(index, pt_key);
    }

    // Add projection factors using ProjectionFactor with SE3 + BALPinholeCameraStrict
    // SelfCalibration mode: optimize pose + landmarks + intrinsics
    for obs in &observations {
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
        )
        .map_err(|e| format!("Invalid camera parameters: {e}"))?;

        let measurements = Matrix2xX::from_columns(&[Vector2::new(obs.x, obs.y)]);
        let factor: ProjectionFactor<BALPinholeCameraStrict, SelfCalibration> =
            ProjectionFactor::new(measurements, camera);

        let pt_key = *pt_keys
            .get(&obs.point_index)
            .ok_or_else(|| format!("no variable for point {}", obs.point_index))?;

        // Use Huber loss (matching C++ implementations)
        let loss = HuberLoss::new(1.0).map_err(|e| format!("Invalid Huber loss: {e}"))?;
        problem.add_residual_block(
            &[
                pose_keys[obs.camera_index],
                pt_key,
                intr_keys[obs.camera_index],
            ],
            Box::new(factor),
            Some(Box::new(loss)),
        );
    }

    // Fix first camera pose (gauge freedom) - all 6 DOF
    let first_pose = *pose_keys.first().ok_or("dataset has no cameras")?;
    for dof in 0..6 {
        problem.fix_variable(first_pose, dof);
    }

    let size = ProblemSize {
        cameras: num_cameras,
        points: pt_keys.len(),
        observations: observations.len(),
    };
    Ok((problem, size))
}

/// Implementation of Apex Solver BA (runs in separate thread)
fn apex_solver_ba_impl(dataset_name: &str, dataset_path: &str, run: ApexRun) -> BABenchmarkResult {
    // Load dataset
    let dataset = match BalLoader::load(dataset_path) {
        Ok(d) => d,
        Err(e) => {
            error!("Failed to load BAL dataset: {}", e);
            return BABenchmarkResult::failed(dataset_name, run.solver, "Rust", &e.to_string());
        }
    };

    let (mut problem, size) = match build_ba_problem(&dataset, run) {
        Ok(built) => built,
        Err(e) => {
            error!("Failed to build BA problem: {}", e);
            return BABenchmarkResult::failed(dataset_name, run.solver, "Rust", &e);
        }
    };
    info!(
        "{}: {} cameras, {} landmarks, {} observations ({:?} Jacobian)",
        dataset_name, size.cameras, size.points, size.observations, run.jacobian_mode
    );

    // Use the same tuned config as bin/bundle_adjustment.rs for consistent
    // results; `run` overrides only which Schur solver it uses.
    let mut config = LevenbergMarquardtConfig::for_bundle_adjustment()
        .with_linear_solver_type(run.linear_solver_type)
        .with_schur_variant(run.schur_variant);

    // `APEX_BENCH_MAX_ITERS` raises the optimizer's iteration cap. A solver
    // whose steps are cheaper but less exact trades iteration count for time,
    // so comparing two solvers at a fixed cap can hide that trade entirely —
    // this makes the cap a variable rather than a hidden constant.
    if let Ok(v) = std::env::var("APEX_BENCH_MAX_ITERS")
        && let Ok(n) = v.parse::<usize>()
    {
        config = config.with_max_iterations(n);
        info!("APEX_BENCH_MAX_ITERS={n}");
    }

    // `APEX_BENCH_ETA` sweeps the PCG forcing sequence. It trades linear-solve
    // accuracy for speed, and where that trade lands is dataset-dependent, so
    // it has to be measurable rather than assumed.
    if let Ok(v) = std::env::var("APEX_BENCH_ETA")
        && let Ok(eta) = v.parse::<f64>()
    {
        config = config.with_schur_cg_q_tolerance(eta);
        info!("APEX_BENCH_ETA={eta}");
    }

    let mut solver = LevenbergMarquardt::with_config(config);

    // Optimize (timing excludes setup)
    let start = Instant::now();
    let result = match solver.optimize(&mut problem) {
        Ok(r) => r,
        Err(e) => {
            error!("Optimization failed: {}", e);
            return BABenchmarkResult::failed(dataset_name, run.solver, "Rust", &e.to_string());
        }
    };
    let elapsed_seconds = start.elapsed().as_secs_f64();

    // Compute initial and final RMSE from solver costs.
    // Solver cost = 0.5 * sum ||r_i||², so MSE = mean ||r_i||² = 2 * cost / n.
    let num_obs = size.observations as f64;
    let initial_mse = 2.0 * result.initial_cost / num_obs;
    let initial_rmse = initial_mse.sqrt();
    let final_mse = 2.0 * result.final_cost / num_obs;
    let final_rmse = final_mse.sqrt();

    let converged = is_converged(&result.status);

    BABenchmarkResult::success(
        dataset_name,
        run.solver,
        "Rust",
        size.cameras,
        size.points,
        size.observations,
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
        return Ok(build_dir);
    }

    info!("Building C++ BA benchmarks ...");

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

    Ok(build_dir)
}

/// Map `APEX_BENCH_SCHUR`'s value to the matching Ceres `linear_solver_type`
/// string `ceres_ba_benchmark` understands, so the Rust and C++ sides always
/// compare the same conceptual solver:
///
/// - `"sparse"`/`"chunked"` (both build the same `S`) -> `sparse_schur`
///   (Ceres's `SPARSE_SCHUR`, apex's `ExplicitSparseSchur`)
/// - `"iterative"`/`"explicit-iterative"` (both PCG-based) -> `iterative_schur`
///   (Ceres's `ITERATIVE_SCHUR`+`SCHUR_JACOBI`, apex's `ImplicitSparseSchur`/
///   `ExplicitSparseSchur` with `ExplicitSchurVariant::Iterative`)
/// - unset -> `None`, so the C++ binary keeps its own default
///   (`iterative_schur`, unchanged from before this mapping existed)
fn ceres_linear_solver_for_apex_bench_schur() -> Option<&'static str> {
    match std::env::var("APEX_BENCH_SCHUR").ok()?.as_str() {
        "sparse" | "chunked" => Some("sparse_schur"),
        "iterative" | "explicit-iterative" => Some("iterative_schur"),
        "explicit-dense" => Some("dense_schur"),
        _ => None,
    }
}

/// Run a C++ benchmark executable and return path to CSV output.
///
/// `extra_env` is forwarded as an additional environment variable for the
/// spawned process — used to pass `CERES_LINEAR_SOLVER` to
/// `ceres_ba_benchmark` so it runs the solver `APEX_BENCH_SCHUR` selected on
/// the Rust side.
fn run_cpp_benchmark(
    exe_name: &str,
    build_dir: &Path,
    dataset_path: &str,
    extra_env: Option<(&str, &str)>,
) -> Result<PathBuf, String> {
    let exe_path = build_dir.join(exe_name);

    if !exe_path.exists() {
        return Err(format!("Executable not found: {:?}", exe_path));
    }

    info!("Running {} ...", exe_name);

    // Spawn process (non-blocking). The benchmark itself is silent; only genuine
    // failures reach stderr, which is inherited so they stay visible.
    let mut command = Command::new(&exe_path);
    command
        .arg(dataset_path)
        .current_dir(build_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if let Some((key, value)) = extra_env {
        command.env(key, value);
    }
    let mut child = command
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
            warn!("Continuing with Rust-only benchmark...");
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
    let ceres_solver = ceres_linear_solver_for_apex_bench_schur();

    for exe_name in cpp_benchmarks {
        let extra_env = if exe_name == "ceres_ba_benchmark" {
            ceres_solver.map(|s| ("CERES_LINEAR_SOLVER", s))
        } else {
            None
        };
        match run_cpp_benchmark(exe_name, &build_dir, &abs_dataset_path, extra_env) {
            Ok(csv_path) => match parse_cpp_ba_results(&csv_path, dataset_name) {
                Ok(results) => {
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
    init_logger();

    info!("BUNDLE ADJUSTMENT BENCHMARK COMPARISON");
    info!("Testing 4 datasets: Ladybug, Trafalgar, Dubrovnik, Venice");

    let runs = apex_runs();
    let datasets = if is_dense_schur_bench() {
        // A dense Hessian over a full BAL dataset is ~485k x 485k at the
        // smallest, so the dense solver runs the truncated Ladybug row only.
        dense_bench_datasets()
    } else {
        get_datasets()
    };
    let mut all_results = Vec::new();

    // Run benchmarks for each dataset
    for dataset in &datasets {
        info!("DATASET: {}", dataset.name);

        // Verify dataset file exists
        if !Path::new(&dataset.path).exists() {
            warn!("Dataset file not found, skipping: {}", dataset.path);
            continue;
        }

        // Phase 1: Apex Solver (Rust)
        for run in &runs {
            all_results.push(apex_solver_ba(&dataset.name, &dataset.path, *run));
        }

        // Phase 2: C++ Solvers (skipped when APEX_BENCH_RUST_ONLY is set)
        if std::env::var_os("APEX_BENCH_RUST_ONLY").is_some() {
            info!("APEX_BENCH_RUST_ONLY set: skipping C++ benchmarks");
        } else {
            let cpp_results = run_cpp_ba_benchmarks(&dataset.name, &dataset.path);
            all_results.extend(cpp_results);
        }
    }

    // Save results to CSV in output/ folder
    let output_dir = "output";
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        warn!("Failed to create output directory: {}", e);
    }

    let output_path = format!("{}/ba_comparison_results.csv", output_dir);
    if let Err(e) = save_csv_results(&all_results, &output_path) {
        warn!("Failed to save CSV results: {}", e);
    }

    // Print comparison table
    print_comparison_table(&all_results);

    info!("Results written to {}", output_path);
}

fn main() {
    run_benchmark_comparison();
}
