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
use apex_solver::linalg::{
    LinearSolver, SparseCholeskySolver, SparseMode, SparseQRSolver, TimedSolver,
};
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

    /// Comma-separated subset of solvers to run, e.g.
    /// `sparse-cholesky,gpu-sparse-cholesky`. Defaults to every solver that is
    /// sensible for the problem size.
    #[arg(long, value_delimiter = ',')]
    solvers: Option<Vec<String>>,

    /// Run the dense solvers even on large problems. They allocate an
    /// n x n Hessian, which is ~7 GB at 30k DOF (city10000), so they are skipped
    /// automatically above `--dense-limit` DOF.
    #[arg(long)]
    force_dense: bool,

    /// DOF above which the dense solvers are skipped unless `--force-dense`.
    #[arg(long, default_value = "2000")]
    dense_limit: usize,

    /// Print the CUDA per-phase breakdown (permutation, symbolic analysis,
    /// upload, factorization, triangular solve, download) and device memory
    /// totals after each GPU run. Requires `--features cuda`.
    #[arg(long)]
    gpu_profile: bool,
}

struct RunResult {
    solver_name: &'static str,
    init_chi2: f64,
    final_chi2: f64,
    improvement_pct: f64,
    iterations: usize,
    time_ms: u128,
    /// Time inside the linear solver only — the part a GPU backend can change.
    solve_ms: u128,
    /// Number of factorize-and-solve calls (> iterations when LM rejects steps).
    solves: usize,
    status: &'static str,
}

/// What a timed run produced, independent of which backend ran it.
struct TimedRun {
    outcome: apex_solver::optimizer::OptimizeResult,
    solve_ms: u128,
    solves: usize,
}

/// Run the optimization with `inner` wrapped in a [`TimedSolver`], then hand the
/// solver back so backend-specific results (a CUDA profile) can be read off it.
fn run_timed<S: LinearSolver<SparseMode>>(
    solver: &mut LevenbergMarquardt,
    problem: &mut Problem,
    inner: S,
) -> (TimedRun, S) {
    let mut timed = TimedSolver::new(inner);
    let outcome = solver.optimize_with_mode::<SparseMode>(problem, &mut timed);
    let run = TimedRun {
        outcome,
        solve_ms: timed.solve_time().as_millis(),
        solves: timed.solve_count(),
    };
    (run, timed.into_inner())
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
    // Sparse solvers go through `optimize_with_mode` so the linear solve can be
    // timed in isolation; the dense ones use the standard entry point. The CUDA
    // arms keep their concrete type so `--gpu-profile` can read the per-phase
    // breakdown off the solver afterwards.
    let run = match solver_type {
        LinearSolverType::SparseCholesky => {
            run_timed(&mut solver, problem, SparseCholeskySolver::new()).0
        }
        LinearSolverType::SparseQR => run_timed(&mut solver, problem, SparseQRSolver::new()).0,
        #[cfg(feature = "cuda")]
        LinearSolverType::GpuSparseCholesky => {
            let cuda = match apex_solver::linalg::CudaSparseCholeskySolver::new() {
                Ok(cuda) => cuda,
                Err(e) => {
                    warn!("GPU Cholesky unavailable: {e}");
                    return None;
                }
            };
            let (run, cuda) = run_timed(&mut solver, problem, cuda);
            if args.gpu_profile {
                info!("{} phase breakdown:\n{}", solver_name, cuda.profile());
            }
            run
        }
        #[cfg(feature = "cuda")]
        LinearSolverType::GpuSparseQR => {
            let cuda = match apex_solver::linalg::CudaSparseQRSolver::new() {
                Ok(cuda) => cuda,
                Err(e) => {
                    warn!("GPU QR unavailable: {e}");
                    return None;
                }
            };
            let (run, cuda) = run_timed(&mut solver, problem, cuda);
            if args.gpu_profile {
                info!("{} phase breakdown:\n{}", solver_name, cuda.profile());
            }
            run
        }
        _ => TimedRun {
            outcome: solver.optimize(problem),
            solve_ms: 0,
            solves: 0,
        },
    };
    let TimedRun {
        outcome,
        solve_ms,
        solves,
    } = run;
    let result = match outcome {
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
        solve_ms,
        solves,
        status,
    })
}

fn print_table(results: &[RunResult]) {
    let w = 132;
    info!("{}", "-".repeat(w));
    info!(
        "{:<18} | {:>12} | {:>12} | {:>11} | {:>5} | {:>8} | {:>9} | {:>6} | {:<12}",
        "Solver",
        "Init chi2",
        "Final chi2",
        "Improvement",
        "Iters",
        "Time(ms)",
        "Solve(ms)",
        "Solves",
        "Status"
    );
    info!("{}", "-".repeat(w));
    for r in results {
        // Dense runs are not instrumented, so report "-" rather than a zero that
        // would read as "the solve was free".
        let (solve, solves) = if r.solves == 0 {
            ("-".to_string(), "-".to_string())
        } else {
            (r.solve_ms.to_string(), r.solves.to_string())
        };
        info!(
            "{:<18} | {:>12.4e} | {:>12.4e} | {:>10.2}% | {:>5} | {:>8} | {:>9} | {:>6} | {:<12}",
            r.solver_name,
            r.init_chi2,
            r.final_chi2,
            r.improvement_pct,
            r.iterations,
            r.time_ms,
            solve,
            solves,
            r.status
        );
    }
    info!("{}", "-".repeat(w));

    // The headline comparison, when both are present.
    let cpu = results.iter().find(|r| r.solver_name == "Sparse Cholesky");
    let gpu = results.iter().find(|r| r.solver_name == "GPU Sparse Cholesky");
    if let (Some(cpu), Some(gpu)) = (cpu, gpu) {
        let ratio = |c: u128, g: u128| if g == 0 { f64::NAN } else { c as f64 / g as f64 };
        info!(
            "Sparse Cholesky CPU vs GPU: solve {:.2}x, end-to-end {:.2}x (>1 means GPU is faster)",
            ratio(cpu.solve_ms, gpu.solve_ms),
            ratio(cpu.time_ms, gpu.time_ms),
        );
    }
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

    // DOF, not vertices: this is what sizes the dense Hessian.
    let total_dof = if is_se3 { vertices * 6 } else { vertices * 3 };
    let skip_dense = total_dof > args.dense_limit && !args.force_dense;
    if skip_dense {
        warn!(
            "Skipping dense solvers: {} DOF exceeds --dense-limit {} \
             (a dense Hessian would be ~{:.1} GB). Pass --force-dense to run them anyway.",
            total_dof,
            args.dense_limit,
            (total_dof * total_dof * 8) as f64 / 1e9,
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

    // `--solvers` matches on a kebab-case form of the display name, so
    // `gpu-sparse-cholesky` selects "GPU Sparse Cholesky".
    let selected = args
        .solvers
        .as_ref()
        .map(|names| names.iter().map(|n| n.to_lowercase()).collect::<Vec<_>>());

    let mut results = Vec::new();
    for &(name, solver_type, use_dense) in SOLVERS {
        if let Some(selected) = &selected {
            let slug = name.to_lowercase().replace(' ', "-");
            if !selected.contains(&slug) {
                continue;
            }
        } else if use_dense && skip_dense {
            continue;
        }

        // A GPU solver with no device is a hard error by design (so benchmarks
        // can't silently measure the CPU). Skip it here rather than aborting the
        // whole comparison.
        #[cfg(feature = "cuda")]
        if matches!(
            solver_type,
            LinearSolverType::GpuSparseCholesky | LinearSolverType::GpuSparseQR
        ) && !apex_solver::cuda::is_available()
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
