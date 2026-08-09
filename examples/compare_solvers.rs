use std::collections::HashMap;
use std::time::Instant;
use tracing::{info, warn};

use apex_solver::apex_io::{G2oLoader, Graph, GraphLoader};
use apex_solver::apex_manifolds::ManifoldType;
use apex_solver::core::VarKey;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BetweenFactor, PriorFactor};
use apex_solver::init_logger;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
use apex_solver::optimizer::{LevenbergMarquardt, OptimizationStatus};
use apex_solver::{JacobianMode, LinearSolverType};
use clap::Parser;
use nalgebra::dvector;

#[derive(Parser)]
#[command(about = "Compare all 4 linear solvers (Sparse/Dense x Cholesky/QR) on any G2O dataset")]
struct Args {
    /// Path to a G2O dataset file (SE2 or SE3)
    path: String,

    /// Maximum LM iterations
    #[arg(short, long, default_value = "100")]
    max_iterations: usize,

    /// Cost convergence tolerance
    #[arg(long, default_value = "1e-6")]
    cost_tolerance: f64,
}

struct RunResult {
    solver_name: &'static str,
    init_chi2: f64,
    final_chi2: f64,
    improvement_pct: f64,
    iterations: usize,
    time_ms: u128,
    status: &'static str,
}

fn run_solver(
    problem: &mut Problem,
    solver_type: LinearSolverType,
    solver_name: &'static str,
    args: &Args,
) -> Option<RunResult> {
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(args.max_iterations)
        .with_cost_tolerance(args.cost_tolerance)
        .with_parameter_tolerance(args.cost_tolerance)
        .with_linear_solver_type(solver_type);
    let mut solver = LevenbergMarquardt::with_config(config);

    let start = Instant::now();
    let result = match solver.optimize(problem) {
        Ok(r) => r,
        Err(e) => {
            warn!("{} failed: {}", solver_name, e);
            return None;
        }
    };
    let time_ms = start.elapsed().as_millis();

    let init_chi2 = result.initial_cost * 2.0;
    let final_chi2 = result.final_cost * 2.0;
    let improvement_pct = (result.initial_cost - result.final_cost) / result.initial_cost * 100.0;
    let status = match result.status {
        OptimizationStatus::Converged
        | OptimizationStatus::CostToleranceReached
        | OptimizationStatus::ParameterToleranceReached
        | OptimizationStatus::GradientToleranceReached => "CONVERGED",
        _ => "NOT CONVERGED",
    };

    Some(RunResult {
        solver_name,
        init_chi2,
        final_chi2,
        improvement_pct,
        iterations: result.iterations,
        time_ms,
        status,
    })
}

fn print_table(results: &[RunResult]) {
    let w = 110;
    info!("{}", "-".repeat(w));
    info!(
        "{:<18} | {:>12} | {:>12} | {:>11} | {:>5} | {:>8} | {:<12}",
        "Solver", "Init chi2", "Final chi2", "Improvement", "Iters", "Time(ms)", "Status"
    );
    info!("{}", "-".repeat(w));
    for r in results {
        info!(
            "{:<18} | {:>12.4e} | {:>12.4e} | {:>10.2}% | {:>5} | {:>8} | {:<12}",
            r.solver_name,
            r.init_chi2,
            r.final_chi2,
            r.improvement_pct,
            r.iterations,
            r.time_ms,
            r.status
        );
    }
    info!("{}", "-".repeat(w));
}

fn build_se3_problem(
    graph: &Graph,
    mode: JacobianMode,
) -> Result<Problem, Box<dyn std::error::Error>> {
    let mut problem = Problem::new(mode);
    let mut var_keys: HashMap<usize, VarKey> = HashMap::new();

    let mut vertex_ids: Vec<_> = graph.vertices_se3.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(v) = graph.vertices_se3.get(&id) {
            let q = v.pose.rotation_quaternion();
            let t = v.pose.translation();
            let key = problem.add_variable(
                ManifoldType::SE3,
                dvector![t.x, t.y, t.z, q.w, q.i, q.j, q.k],
            );
            var_keys.insert(id, key);
        }
    }

    if let Some(&first_id) = vertex_ids.first()
        && let Some(v) = graph.vertices_se3.get(&first_id)
    {
        let q = v.pose.rotation_quaternion();
        let t = v.pose.translation();
        let prior = PriorFactor {
            data: dvector![t.x, t.y, t.z, q.w, q.i, q.j, q.k],
        };
        let loss = HuberLoss::new(1.0)?;
        let first_key = var_keys[&first_id];
        problem.add_residual_block(&[first_key], Box::new(prior), Some(Box::new(loss)));
    }

    for edge in &graph.edges_se3 {
        if let (Some(&k0), Some(&k1)) = (var_keys.get(&edge.from), var_keys.get(&edge.to)) {
            problem.add_residual_block(
                &[k0, k1],
                Box::new(BetweenFactor::new(edge.measurement.clone())),
                None,
            );
        }
    }

    Ok(problem)
}

fn build_se2_problem(
    graph: &Graph,
    mode: JacobianMode,
) -> Result<Problem, Box<dyn std::error::Error>> {
    let mut problem = Problem::new(mode);
    let mut var_keys: HashMap<usize, VarKey> = HashMap::new();

    let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(v) = graph.vertices_se2.get(&id) {
            let key = problem.add_variable(
                ManifoldType::SE2,
                dvector![v.pose.x(), v.pose.y(), v.pose.angle()],
            );
            var_keys.insert(id, key);
        }
    }

    if let Some(&first_id) = vertex_ids.first()
        && let Some(v) = graph.vertices_se2.get(&first_id)
    {
        let prior = PriorFactor {
            data: dvector![v.pose.x(), v.pose.y(), v.pose.angle()],
        };
        let loss = HuberLoss::new(1.0)?;
        let first_key = var_keys[&first_id];
        problem.add_residual_block(&[first_key], Box::new(prior), Some(Box::new(loss)));
    }

    for edge in &graph.edges_se2 {
        if let (Some(&k0), Some(&k1)) = (var_keys.get(&edge.from), var_keys.get(&edge.to)) {
            problem.add_residual_block(
                &[k0, k1],
                Box::new(BetweenFactor::new(edge.measurement.clone())),
                None,
            );
        }
    }

    Ok(problem)
}

fn main() {
    let args = Args::parse();
    init_logger();

    let graph = match G2oLoader::load(&args.path) {
        Ok(g) => g,
        Err(e) => {
            warn!("Failed to load '{}': {}", args.path, e);
            std::process::exit(1);
        }
    };

    let is_se3 = !graph.vertices_se3.is_empty();
    let (manifold_label, vertices, edges) = if is_se3 {
        ("SE3", graph.vertices_se3.len(), graph.edges_se3.len())
    } else {
        ("SE2", graph.vertices_se2.len(), graph.edges_se2.len())
    };

    let dataset_name = std::path::Path::new(&args.path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&args.path);

    info!("APEX-SOLVER - LINEAR SOLVER COMPARISON");
    info!("Dataset : {} ({})", dataset_name, manifold_label);
    info!("Problem : {} vertices, {} edges", vertices, edges);
    info!(
        "Config  : max_iter={}, cost_tol={:.0e}",
        args.max_iterations, args.cost_tolerance
    );
    info!("");

    if vertices > 500 {
        warn!(
            "Large problem ({} vertices) - dense solvers may be slow",
            vertices
        );
    }

    const SOLVERS: &[(&str, LinearSolverType, bool)] = &[
        ("Sparse Cholesky", LinearSolverType::SparseCholesky, false),
        ("Sparse QR", LinearSolverType::SparseQR, false),
        ("Dense Cholesky", LinearSolverType::DenseCholesky, true),
        ("Dense QR", LinearSolverType::DenseQR, true),
        // GPU solvers are only listed when built with `--features cuda` AND a
        // CUDA device is actually present, so this example stays runnable
        // everywhere. Selecting them without a device is an error, never a
        // silent CPU fallback.
        #[cfg(feature = "cuda")]
        (
            "GPU Sparse Cholesky",
            LinearSolverType::GpuSparseCholesky,
            false,
        ),
        #[cfg(feature = "cuda")]
        ("GPU Sparse QR", LinearSolverType::GpuSparseQR, false),
    ];

    let mut results = Vec::new();
    for &(name, solver_type, use_dense) in SOLVERS {
        // A GPU solver with no device is a hard error by design (so benchmarks
        // can't silently measure the CPU). Skip it here rather than aborting the
        // whole comparison.
        #[cfg(feature = "cuda")]
        if matches!(
            solver_type,
            LinearSolverType::GpuSparseCholesky | LinearSolverType::GpuSparseQR
        ) && !apex_solver::linalg::gpu::is_available()
        {
            info!("Skipping {} - no CUDA device available", name);
            continue;
        }

        info!("Running {}...", name);
        let mode = if use_dense {
            JacobianMode::Dense
        } else {
            JacobianMode::Sparse
        };
        let mut problem = if is_se3 {
            match build_se3_problem(&graph, mode) {
                Ok(p) => p,
                Err(e) => {
                    warn!("Failed to build problem for {}: {}", name, e);
                    continue;
                }
            }
        } else {
            match build_se2_problem(&graph, mode) {
                Ok(p) => p,
                Err(e) => {
                    warn!("Failed to build problem for {}: {}", name, e);
                    continue;
                }
            }
        };
        if let Some(r) = run_solver(&mut problem, solver_type, name, &args) {
            results.push(r);
        }
    }

    info!("");
    print_table(&results);
    info!("");
    info!("Done.");
}
