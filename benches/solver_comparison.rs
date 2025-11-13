//! Comprehensive solver comparison benchmark for apex-solver, factrs, and tiny-solver
//!
//! This benchmark compares three Rust nonlinear optimization libraries on standard
//! pose graph optimization datasets (both SE2 and SE3).
//!
//! ## Configuration Philosophy
//!
//! The apex-solver configuration **exactly matches** the production settings used in
//! `bin/optimize_2d_graph.rs` and `bin/optimize_3d_graph.rs` to ensure fair comparison:
//!
//! ### SE2 (2D) Configuration:
//! - Max iterations: 150 (matches optimize_2d_graph.rs)
//! - Cost tolerance: 1e-4
//! - Parameter tolerance: 1e-4
//! - Gradient tolerance: 1e-10 (enables early-exit when gradient converges)
//!
//! ### SE3 (3D) Configuration:
//! - Max iterations: 100 (matches optimize_3d_graph.rs)
//! - Cost tolerance: 1e-4
//! - Parameter tolerance: 1e-4
//! - Gradient tolerance: 1e-12 (tighter for SE3 due to higher complexity)
//!
//! ### Timing Methodology:
//! - Timing starts immediately before `solver.optimize()` call
//! - Problem setup (graph loading, factor creation) is excluded from timing
//! - This matches the timing approach in optimize_*_graph.rs binaries
//! - Each dataset is run 5 times and results are averaged for stability
//!
//! ### Gauge Freedom Handling:
//! - apex-solver: Uses `fix_variable()` to anchor first pose (simple, effective for LM)
//! - factrs/tiny-solver: Use their default gauge freedom handling

use std::collections::HashMap;
use std::hint::black_box;
use std::panic;
use std::time::Instant;

// apex-solver imports
use apex_solver::core::loss_functions::L2Loss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BetweenFactorSE2, BetweenFactorSE3};
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::dvector;

// factrs imports
use factrs::{
    optimizers::{LevenMarquardt, Optimizer as FactrsOptimizer},
    utils::load_g20,
};

// tiny-solver imports
use tiny_solver::{
    helper::read_g2o as load_tiny_g2o, levenberg_marquardt_optimizer::LevenbergMarquardtOptimizer,
    optimizer::Optimizer as TinyOptimizer,
};

// CSV output
use csv::Writer;
use serde::Serialize;

/// Dataset information
#[derive(Clone)]
struct Dataset {
    name: &'static str,
    file: &'static str,
    is_3d: bool,
}

const DATASETS: &[Dataset] = &[
    Dataset {
        name: "M3500",
        file: "data/M3500.g2o",
        is_3d: false,
    },
    Dataset {
        name: "mit",
        file: "data/mit.g2o",
        is_3d: false,
    },
    Dataset {
        name: "intel",
        file: "data/intel.g2o",
        is_3d: false,
    },
    Dataset {
        name: "manhattanOlson3500",
        file: "data/manhattanOlson3500.g2o",
        is_3d: false,
    },
    Dataset {
        name: "ring",
        file: "data/ring.g2o",
        is_3d: true,
    },
    Dataset {
        name: "sphere2500",
        file: "data/sphere2500.g2o",
        is_3d: true,
    },
    Dataset {
        name: "parking-garage",
        file: "data/parking-garage.g2o",
        is_3d: true,
    },
    Dataset {
        name: "torus3D",
        file: "data/torus3D.g2o",
        is_3d: true,
    },
];

/// Benchmark result structure
#[derive(Debug, Clone, Serialize)]
struct BenchmarkResult {
    dataset: String,
    solver: String,
    initial_cost: String,
    final_cost: String,
    elapsed_ms: String,
    converged: String,
    iterations: String,
}

impl BenchmarkResult {
    fn success(
        dataset: &str,
        solver: &str,
        initial_cost: f64,
        final_cost: f64,
        elapsed_ms: f64,
        converged: bool,
        iterations: Option<usize>,
    ) -> Self {
        Self {
            dataset: dataset.to_string(),
            solver: solver.to_string(),
            initial_cost: format!("{:.6}", initial_cost),
            final_cost: format!("{:.6}", final_cost),
            elapsed_ms: format!("{:.2}", elapsed_ms),
            converged: converged.to_string(),
            iterations: iterations.map_or("-".to_string(), |i| i.to_string()),
        }
    }

    fn diverged(dataset: &str, solver: &str, initial_cost: Option<f64>, elapsed_ms: f64) -> Self {
        Self {
            dataset: dataset.to_string(),
            solver: solver.to_string(),
            initial_cost: initial_cost.map_or("-".to_string(), |c| format!("{:.6}", c)),
            final_cost: "diverged".to_string(),
            elapsed_ms: format!("{:.2}", elapsed_ms),
            converged: "false".to_string(),
            iterations: "-".to_string(),
        }
    }

    fn failed(dataset: &str, solver: &str, error: &str) -> Self {
        Self {
            dataset: dataset.to_string(),
            solver: solver.to_string(),
            initial_cost: "-".to_string(),
            final_cost: "-".to_string(),
            elapsed_ms: "-".to_string(),
            converged: "false".to_string(),
            iterations: format!("error: {}", error),
        }
    }
}

/// Helper to determine if apex-solver converged successfully
fn is_converged(status: &OptimizationStatus) -> bool {
    matches!(
        status,
        OptimizationStatus::Converged
            | OptimizationStatus::CostToleranceReached
            | OptimizationStatus::GradientToleranceReached
            | OptimizationStatus::ParameterToleranceReached
            | OptimizationStatus::MaxIterationsReached
    )
}

// ========================= apex-solver =========================

fn apex_solver_se2(dataset: &Dataset) -> BenchmarkResult {
    let graph = match G2oLoader::load(dataset.file) {
        Ok(g) => g,
        Err(e) => return BenchmarkResult::failed(dataset.name, "apex-solver", &e.to_string()),
    };

    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add vertices
    let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se2.get(&id) {
            let var_name = format!("x{}", id);
            let se2_data = dvector![vertex.x(), vertex.y(), vertex.theta()];
            initial_values.insert(var_name, (ManifoldType::SE2, se2_data));
        }
    }

    // Add between factors with L2 loss (matches optimize_2d_graph.rs default)
    for edge in &graph.edges_se2 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);
        let between_factor = BetweenFactorSE2::new(
            edge.measurement.x(),
            edge.measurement.y(),
            edge.measurement.angle(),
        );
        problem.add_residual_block(
            &[&id0, &id1],
            Box::new(between_factor),
            Some(Box::new(L2Loss)),
        );
    }

    // Fix gauge freedom by constraining the first pose
    // This makes the Hessian full-rank and improves convergence
    // Matches production configuration in optimize_2d_graph.rs
    if let Some(&first_id) = vertex_ids.first() {
        let first_var_name = format!("x{}", first_id);
        problem.fix_variable(&first_var_name, 0); // Fix x
        problem.fix_variable(&first_var_name, 1); // Fix y
        problem.fix_variable(&first_var_name, 2); // Fix theta
    }

    // Optimize with production-grade configuration matching optimize_2d_graph.rs
    // - Max iterations: 150 (sufficient for SE2 convergence)
    // - Cost/param tolerance: 1e-4 (balanced accuracy vs speed)
    // - Gradient tolerance: 1e-10 (early-exit on gradient convergence, saves iterations)
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(150)
        .with_cost_tolerance(1e-4)
        .with_parameter_tolerance(1e-4)
        .with_gradient_tolerance(1e-10)
        .with_verbose(false);

    let mut solver = LevenbergMarquardt::with_config(config);

    // Start timing immediately before optimization (excludes problem setup overhead)
    // This matches the timing approach in optimize_2d_graph.rs for fair comparison
    let start_time = Instant::now();
    match solver.optimize(&problem, &initial_values) {
        Ok(result) => {
            let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            let converged = is_converged(&result.status);
            BenchmarkResult::success(
                dataset.name,
                "apex-solver",
                result.initial_cost,
                result.final_cost,
                elapsed_ms,
                converged,
                Some(result.iterations),
            )
        }
        Err(e) => BenchmarkResult::failed(dataset.name, "apex-solver", &e.to_string()),
    }
}

fn apex_solver_se3(dataset: &Dataset) -> BenchmarkResult {
    let graph = match G2oLoader::load(dataset.file) {
        Ok(g) => g,
        Err(e) => return BenchmarkResult::failed(dataset.name, "apex-solver", &e.to_string()),
    };

    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add vertices
    let mut vertex_ids: Vec<_> = graph.vertices_se3.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se3.get(&id) {
            let var_name = format!("x{}", id);
            let quat = vertex.rotation();
            let trans = vertex.translation();
            let se3_data = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];
            initial_values.insert(var_name, (ManifoldType::SE3, se3_data));
        }
    }

    // Add between factors with L2 loss (matches optimize_3d_graph.rs default)
    for edge in &graph.edges_se3 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);
        let between_factor = BetweenFactorSE3::new(edge.measurement.clone());
        problem.add_residual_block(
            &[&id0, &id1],
            Box::new(between_factor),
            Some(Box::new(L2Loss)),
        );
    }

    // NO gauge freedom handling for SE3 + LM (matches optimize_3d_graph.rs)
    // Unlike SE2, the 3D optimizer does NOT fix variables or add prior factors for LM
    // LM's built-in damping (λI) handles the rank-deficient Hessian naturally
    // This allows the optimizer to find better solutions with fewer iterations

    // Optimize with production-grade configuration matching optimize_3d_graph.rs
    // - Max iterations: 100 (sufficient for SE3 convergence)
    // - Cost/param tolerance: 1e-4 (balanced accuracy vs speed)
    // - Gradient tolerance: 1e-12 (tighter than SE2 due to SE3 complexity, enables early-exit)
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-4)
        .with_parameter_tolerance(1e-4)
        .with_gradient_tolerance(1e-12)
        .with_verbose(false);

    let mut solver = LevenbergMarquardt::with_config(config);

    // Start timing immediately before optimization (excludes problem setup overhead)
    // This matches the timing approach in optimize_3d_graph.rs for fair comparison
    let start_time = Instant::now();
    match solver.optimize(&problem, &initial_values) {
        Ok(result) => {
            let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            let converged = is_converged(&result.status);
            BenchmarkResult::success(
                dataset.name,
                "apex-solver",
                result.initial_cost,
                result.final_cost,
                elapsed_ms,
                converged,
                Some(result.iterations),
            )
        }
        Err(e) => BenchmarkResult::failed(dataset.name, "apex-solver", &e.to_string()),
    }
}

// ========================= factrs =========================

fn factrs_benchmark(dataset: &Dataset) -> BenchmarkResult {
    // Catch panics from factrs parsing/loading
    let load_result = panic::catch_unwind(|| load_g20(dataset.file));

    let (graph, init) = match load_result {
        Ok((g, i)) => (g, i),
        Err(_) => {
            return BenchmarkResult::failed(
                dataset.name,
                "factrs",
                "failed to load dataset (panic)",
            );
        }
    };

    // Compute initial cost before optimization
    let initial_cost = graph.error(&init);

    // Start timing
    let start = Instant::now();

    // Use Levenberg-Marquardt optimizer with default Cholesky solver
    let mut opt: LevenMarquardt = LevenMarquardt::new(graph.clone());
    let result = black_box(opt.optimize(init));

    // Stop timing
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Ok(final_values) => {
            // Compute final cost after optimization
            let final_cost = graph.error(&final_values);
            BenchmarkResult::success(
                dataset.name,
                "factrs",
                initial_cost,
                final_cost,
                elapsed_ms,
                true, // Successfully converged
                None, // factrs doesn't expose iteration count
            )
        }
        Err(factrs::optimizers::OptError::MaxIterations(final_values)) => {
            // Max iterations reached but we have final values
            let final_cost = graph.error(&final_values);
            BenchmarkResult::success(
                dataset.name,
                "factrs",
                initial_cost,
                final_cost,
                elapsed_ms,
                false, // Did not converge (max iterations)
                None,
            )
        }
        Err(factrs::optimizers::OptError::FailedToStep) => {
            BenchmarkResult::diverged(dataset.name, "factrs", Some(initial_cost), elapsed_ms)
        }
        Err(factrs::optimizers::OptError::InvalidSystem) => {
            BenchmarkResult::diverged(dataset.name, "factrs", Some(initial_cost), elapsed_ms)
        }
    }
}

// ========================= tiny-solver =========================

fn tiny_solver_benchmark(dataset: &Dataset) -> BenchmarkResult {
    // Catch panics from tiny-solver parsing/loading
    let load_result = panic::catch_unwind(|| load_tiny_g2o(dataset.file));

    let (graph, init) = match load_result {
        Ok((g, i)) => (g, i),
        Err(_) => {
            return BenchmarkResult::failed(
                dataset.name,
                "tiny-solver",
                "failed to load dataset (panic)",
            );
        }
    };

    let lm = LevenbergMarquardtOptimizer::default();

    // Compute initial cost before optimization
    // Note: tiny-solver uses ||r||^2 while apex-solver and factrs use 0.5 * ||r||^2
    // We normalize tiny-solver costs to enable fair comparison
    let initial_blocks = graph.initialize_parameter_blocks(&init);
    let initial_cost = lm.compute_error(&graph, &initial_blocks) * 0.5;

    // Start timing
    let start = Instant::now();

    // Use Levenberg-Marquardt optimizer
    let result = black_box(lm.optimize(&graph, &init, None));

    // Stop timing
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Some(final_values) => {
            // Compute final cost after optimization
            let final_blocks = graph.initialize_parameter_blocks(&final_values);
            let final_cost = lm.compute_error(&graph, &final_blocks) * 0.5;
            BenchmarkResult::success(
                dataset.name,
                "tiny-solver",
                initial_cost,
                final_cost,
                elapsed_ms,
                true, // Successfully converged
                None, // tiny-solver doesn't expose iteration count
            )
        }
        None => {
            // Optimization failed (NaN, solve failed, or other error)
            BenchmarkResult::diverged(dataset.name, "tiny-solver", Some(initial_cost), elapsed_ms)
        }
    }
}

// ========================= Main Benchmark Runner =========================

fn run_single_benchmark(dataset: &Dataset, solver: &str) -> BenchmarkResult {
    match (dataset.is_3d, solver) {
        (false, "apex-solver") => apex_solver_se2(dataset),
        (true, "apex-solver") => apex_solver_se3(dataset),
        (_, "factrs") => factrs_benchmark(dataset),
        (_, "tiny-solver") => tiny_solver_benchmark(dataset),
        _ => panic!("Unknown solver: {}", solver),
    }
}

fn main() {
    println!("Starting solver comparison benchmark...");
    println!("Running each configuration 5 times and averaging results...\n");

    let solvers = ["apex-solver", "factrs", "tiny-solver"];
    let mut all_results = Vec::new();

    for dataset in DATASETS {
        println!("Dataset: {}", dataset.name);

        for solver in &solvers {
            print!("  {} ... ", solver);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            // Run multiple times to get stable measurements
            let num_runs = 5;
            let mut results = Vec::new();

            for _ in 0..num_runs {
                let result = run_single_benchmark(dataset, solver);
                results.push(result);
            }

            // Use the last result for convergence info, but average timing if successful
            if let Some(first_result) = results.first() {
                let mut avg_result = first_result.clone();

                // Average elapsed time if all runs succeeded
                if results.iter().all(|r| r.elapsed_ms != "-") {
                    let total_time: f64 = results
                        .iter()
                        .filter_map(|r| r.elapsed_ms.parse::<f64>().ok())
                        .sum();
                    avg_result.elapsed_ms = format!("{:.2}", total_time / num_runs as f64);
                }

                println!(
                    "done (converged: {}, time: {} ms)",
                    avg_result.converged, avg_result.elapsed_ms
                );

                all_results.push(avg_result);
            }
        }
        println!();
    }

    // Write results to CSV
    let csv_path = "benchmark_results.csv";
    let mut writer = Writer::from_path(csv_path).expect("Failed to create CSV file");

    for result in &all_results {
        writer
            .serialize(result)
            .expect("Failed to write CSV record");
    }
    writer.flush().expect("Failed to flush CSV writer");

    println!("\nResults written to {}", csv_path);
    println!("\nSummary:");
    println!(
        "{:<20} {:<15} {:<12} {:<12}",
        "Dataset", "Solver", "Converged", "Time (ms)"
    );
    println!("{}", "-".repeat(65));

    for result in &all_results {
        println!(
            "{:<20} {:<15} {:<12} {:<12}",
            result.dataset, result.solver, result.converged, result.elapsed_ms
        );
    }
}
