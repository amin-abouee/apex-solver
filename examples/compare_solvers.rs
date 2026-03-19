use std::collections::HashMap;
use std::time::Instant;
use tracing::{info, warn};

use apex_solver::apex_io::{G2oLoader, GraphLoader};
use apex_solver::apex_manifolds::ManifoldType;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BetweenFactor, PriorFactor};
use apex_solver::init_logger;
use apex_solver::linearizer::cpu::sparse::build_symbolic_structure;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
use apex_solver::optimizer::{LevenbergMarquardt, OptimizationStatus};
use apex_solver::{JacobianMode, LinearSolverType};
use clap::Parser;
use nalgebra::dvector;

type InitialValues = HashMap<String, (ManifoldType, nalgebra::DVector<f64>)>;
type BuildResult = Result<(Problem, InitialValues), Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(about = "Compare all 4 linear solvers (Sparse/Dense × Cholesky/QR) on any G2O dataset")]
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

/// Evaluate the sparse Jacobian once to obtain NNZ and density.
/// Returns (nrows, ncols, nnz, density_pct).
fn jacobian_stats(problem: &Problem, initial_values: &InitialValues) -> (usize, usize, usize, f64) {
    let variables = problem.initialize_variables(initial_values);

    let mut col_idx = HashMap::new();
    let mut offset = 0usize;
    let mut sorted: Vec<_> = variables.keys().cloned().collect();
    sorted.sort();
    for name in &sorted {
        col_idx.insert(name.clone(), offset);
        offset += variables[name].get_size();
    }

    let Ok(sym) = build_symbolic_structure(problem, &variables, &col_idx, offset) else {
        return (0, 0, 0, 0.0);
    };

    let Ok((_, jacobian)) =
        problem.compute_residual_and_jacobian_sparse(&variables, &col_idx, &sym)
    else {
        return (0, 0, 0, 0.0);
    };

    let nrows = jacobian.nrows();
    let ncols = jacobian.ncols();
    let nnz = jacobian.compute_nnz();
    let density = nnz as f64 / (nrows * ncols) as f64 * 100.0;
    (nrows, ncols, nnz, density)
}

fn run_solver(
    problem: &Problem,
    initial_values: &InitialValues,
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
    let result = match solver.optimize(problem, initial_values) {
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
    info!("{}", "─".repeat(w));
    info!(
        "{:<18} | {:>12} | {:>12} | {:>11} | {:>5} | {:>8} | {:<12}",
        "Solver", "Init χ²", "Final χ²", "Improvement", "Iters", "Time(ms)", "Status"
    );
    info!("{}", "─".repeat(w));
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
    info!("{}", "─".repeat(w));
}

fn build_se3_problem(graph: &apex_solver::apex_io::Graph, mode: JacobianMode) -> BuildResult {
    let mut initial_values = InitialValues::new();
    let mut vertex_ids: Vec<_> = graph.vertices_se3.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(v) = graph.vertices_se3.get(&id) {
            let q = v.pose.rotation_quaternion();
            let t = v.pose.translation();
            initial_values.insert(
                format!("x{id}"),
                (
                    ManifoldType::SE3,
                    dvector![t.x, t.y, t.z, q.w, q.i, q.j, q.k],
                ),
            );
        }
    }

    let mut problem = Problem::new(mode);

    if let Some(&first_id) = vertex_ids.first()
        && let Some(v) = graph.vertices_se3.get(&first_id)
    {
        let q = v.pose.rotation_quaternion();
        let t = v.pose.translation();
        let prior = PriorFactor {
            data: dvector![t.x, t.y, t.z, q.w, q.i, q.j, q.k],
        };
        let loss = HuberLoss::new(1.0)?;
        problem.add_residual_block(
            &[&format!("x{first_id}")],
            Box::new(prior),
            Some(Box::new(loss)),
        );
    }

    for edge in &graph.edges_se3 {
        problem.add_residual_block(
            &[&format!("x{}", edge.from), &format!("x{}", edge.to)],
            Box::new(BetweenFactor::new(edge.measurement.clone())),
            None,
        );
    }

    Ok((problem, initial_values))
}

fn build_se2_problem(graph: &apex_solver::apex_io::Graph, mode: JacobianMode) -> BuildResult {
    let mut initial_values = InitialValues::new();
    let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(v) = graph.vertices_se2.get(&id) {
            initial_values.insert(
                format!("x{id}"),
                (
                    ManifoldType::SE2,
                    dvector![v.pose.x(), v.pose.y(), v.pose.angle()],
                ),
            );
        }
    }

    let mut problem = Problem::new(mode);

    if let Some(&first_id) = vertex_ids.first()
        && let Some(v) = graph.vertices_se2.get(&first_id)
    {
        let prior = PriorFactor {
            data: dvector![v.pose.x(), v.pose.y(), v.pose.angle()],
        };
        let loss = HuberLoss::new(1.0)?;
        problem.add_residual_block(
            &[&format!("x{first_id}")],
            Box::new(prior),
            Some(Box::new(loss)),
        );
    }

    for edge in &graph.edges_se2 {
        problem.add_residual_block(
            &[&format!("x{}", edge.from), &format!("x{}", edge.to)],
            Box::new(BetweenFactor::new(edge.measurement.clone())),
            None,
        );
    }

    Ok((problem, initial_values))
}

fn main() {
    let args = Args::parse();
    init_logger();

    let graph = match G2oLoader::load(&args.path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to load '{}': {}", args.path, e);
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

    info!("APEX-SOLVER — LINEAR SOLVER COMPARISON");
    info!("Dataset : {} ({})", dataset_name, manifold_label);
    info!("Problem : {} vertices, {} edges", vertices, edges);
    info!(
        "Config  : max_iter={}, cost_tol={:.0e}",
        args.max_iterations, args.cost_tolerance
    );

    // Build sparse and dense problems
    let build_sparse = if is_se3 {
        build_se3_problem(&graph, JacobianMode::Sparse)
    } else {
        build_se2_problem(&graph, JacobianMode::Sparse)
    };
    let (sparse_problem, sparse_init) = match build_sparse {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("Failed to build sparse problem: {e}");
            std::process::exit(1);
        }
    };
    let build_dense = if is_se3 {
        build_se3_problem(&graph, JacobianMode::Dense)
    } else {
        build_se2_problem(&graph, JacobianMode::Dense)
    };
    let (dense_problem, dense_init) = match build_dense {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("Failed to build dense problem: {e}");
            std::process::exit(1);
        }
    };

    // Compute and print Jacobian density (from sparse problem)
    let (jac_rows, jac_cols, nnz, density_pct) = jacobian_stats(&sparse_problem, &sparse_init);
    info!(
        "Jacobian: {}×{} | NNZ: {} | Density: {:.4}%",
        jac_rows, jac_cols, nnz, density_pct
    );

    if vertices > 500 {
        warn!(
            "Large problem ({} vertices) — dense solvers may be slow",
            vertices
        );
    }

    info!("");

    const SOLVERS: &[(&str, LinearSolverType, bool)] = &[
        ("Sparse Cholesky", LinearSolverType::SparseCholesky, false),
        ("Sparse QR", LinearSolverType::SparseQR, false),
        ("Dense Cholesky", LinearSolverType::DenseCholesky, true),
        ("Dense QR", LinearSolverType::DenseQR, true),
    ];

    let mut results = Vec::new();
    for &(name, solver_type, use_dense) in SOLVERS {
        info!("Running {}…", name);
        let (problem, init) = if use_dense {
            (&dense_problem, &dense_init)
        } else {
            (&sparse_problem, &sparse_init)
        };
        if let Some(r) = run_solver(problem, init, solver_type, name, &args) {
            results.push(r);
        }
    }

    info!("");
    print_table(&results);
    info!("");
    info!("Done.");
}
