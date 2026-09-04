use std::collections::HashMap;
use std::time::Instant;
use tracing::{info, warn};

use apex_solver::JacobianMode;
use apex_solver::apex_io::{G2oLoader, GraphLoader, ODOMETRY_DATA_DIR_2D, ODOMETRY_DATA_DIR_3D};
use apex_solver::apex_manifolds::ManifoldType;
use apex_solver::core::VarKey;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::pose::{BetweenFactor, PriorFactor};
use apex_solver::init_logger;
use apex_solver::optimizer::dog_leg::DogLegConfig;
use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
use apex_solver::optimizer::{DogLeg, GaussNewton, LevenbergMarquardt, OptimizationStatus};
use clap::Parser;
use nalgebra::dvector;

#[derive(Parser)]
#[command(name = "compare_optimizers")]
#[command(about = "Compare LM, GN, and DL optimizers on real G2O datasets")]
struct Args {
    /// Maximum number of optimization iterations
    #[arg(short, long, default_value = "100")]
    max_iterations: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Cost tolerance for convergence
    #[arg(long, default_value = "1e-3")]
    cost_tolerance: f64,

    /// Parameter tolerance for convergence
    #[arg(long, default_value = "1e-3")]
    parameter_tolerance: f64,
}

#[derive(Clone)]
struct BenchmarkResult {
    dataset: String,
    manifold: String,
    optimizer: String,
    vertices: usize,
    edges: usize,
    initial_cost: f64,
    final_cost: f64,
    improvement: f64,
    iterations: usize,
    time_ms: u128,
    status: String,
}

fn print_summary_table(results: &[BenchmarkResult]) {
    info!("OPTIMIZER COMPARISON SUMMARY");

    info!(
        "{:<12} | {:<8} | {:<10} | {:<8} | {:<6} | {:<12} | {:<12} | {:<11} | {:<5} | {:<9} | {:<10}",
        "Dataset",
        "Manifold",
        "Optimizer",
        "Vertices",
        "Edges",
        "Init Cost",
        "Final Cost",
        "Improvement",
        "Iters",
        "Time(ms)",
        "Status"
    );
    info!("{}", "-".repeat(150));

    for result in results {
        info!(
            "{:<12} | {:<8} | {:<10} | {:<8} | {:<6} | {:<12.6e} | {:<12.6e} | {:>10.2}% | {:<5} | {:<9} | {:<10}",
            result.dataset,
            result.manifold,
            result.optimizer,
            result.vertices,
            result.edges,
            result.initial_cost,
            result.final_cost,
            result.improvement,
            result.iterations,
            result.time_ms,
            result.status
        );
    }

    info!("{}", "-".repeat(150));
}

/// Build an SE3 problem from a loaded graph.
/// Returns `(Problem, num_vertices, num_edges)`.
fn build_se3_problem(
    graph: &apex_solver::apex_io::Graph,
) -> Result<(Problem, HashMap<usize, VarKey>), Box<dyn std::error::Error>> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut var_keys: HashMap<usize, VarKey> = HashMap::new();

    let mut vertex_ids: Vec<_> = graph.vertices_se3.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se3.get(&id) {
            let quat = vertex.pose.rotation_quaternion();
            let trans = vertex.pose.translation();
            let se3_data = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];
            let key = problem.add_variable(ManifoldType::SE3, se3_data);
            var_keys.insert(id, key);
        }
    }

    if let Some(&first_id) = vertex_ids.first()
        && let Some(first_vertex) = graph.vertices_se3.get(&first_id)
    {
        let prior_factor = PriorFactor::new(first_vertex.pose.clone());
        let huber_loss = HuberLoss::new(1.0)?;
        let first_key = var_keys[&first_id];
        problem.add_residual_block(
            &[first_key],
            Box::new(prior_factor),
            Some(Box::new(huber_loss)),
        );
    }

    for edge in &graph.edges_se3 {
        let factor = BetweenFactor::new(edge.measurement.clone());
        if let (Some(&k0), Some(&k1)) = (var_keys.get(&edge.from), var_keys.get(&edge.to)) {
            problem.add_residual_block(&[k0, k1], Box::new(factor), None);
        }
    }

    Ok((problem, var_keys))
}

/// Build an SE2 problem from a loaded graph.
fn build_se2_problem(
    graph: &apex_solver::apex_io::Graph,
) -> Result<(Problem, HashMap<usize, VarKey>), Box<dyn std::error::Error>> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut var_keys: HashMap<usize, VarKey> = HashMap::new();

    let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se2.get(&id) {
            let se2_data = dvector![vertex.pose.x(), vertex.pose.y(), vertex.pose.angle()];
            let key = problem.add_variable(ManifoldType::SE2, se2_data);
            var_keys.insert(id, key);
        }
    }

    if let Some(&first_id) = vertex_ids.first()
        && let Some(first_vertex) = graph.vertices_se2.get(&first_id)
    {
        let prior_factor = PriorFactor::new(first_vertex.pose.clone());
        let huber_loss = HuberLoss::new(1.0)?;
        let first_key = var_keys[&first_id];
        problem.add_residual_block(
            &[first_key],
            Box::new(prior_factor),
            Some(Box::new(huber_loss)),
        );
    }

    for edge in &graph.edges_se2 {
        let factor = BetweenFactor::new(edge.measurement.clone());
        if let (Some(&k0), Some(&k1)) = (var_keys.get(&edge.from), var_keys.get(&edge.to)) {
            problem.add_residual_block(&[k0, k1], Box::new(factor), None);
        }
    }

    Ok((problem, var_keys))
}

fn test_se3_dataset(
    dataset_name: &str,
    args: &Args,
    all_results: &mut Vec<BenchmarkResult>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("TESTING {} (SE3)", dataset_name.to_uppercase());

    let file_path = format!("{}/{}.g2o", ODOMETRY_DATA_DIR_3D, dataset_name);
    let graph = match G2oLoader::load(&file_path) {
        Ok(g) => g,
        Err(e) => {
            warn!("Failed to load {}: {}", file_path, e);
            return Ok(());
        }
    };

    let num_vertices = graph.vertices_se3.len();
    let num_edges = graph.edges_se3.len();
    info!("Loaded: {} vertices, {} edges", num_vertices, num_edges);

    // Run each optimizer on a fresh problem (Problem can't be cloned since Factor isn't Clone)
    for opt_name in &["LM", "GN", "DL"] {
        info!("--- Testing {} ---", opt_name);

        let (mut problem, _) = build_se3_problem(&graph)?;
        let start = Instant::now();

        let result = match *opt_name {
            "LM" => {
                let config = LevenbergMarquardtConfig::new()
                    .with_max_iterations(args.max_iterations)
                    .with_cost_tolerance(args.cost_tolerance)
                    .with_parameter_tolerance(args.parameter_tolerance);
                let mut solver = LevenbergMarquardt::with_config(config);
                solver.optimize(&mut problem)?
            }
            "GN" => {
                let config = GaussNewtonConfig::new()
                    .with_max_iterations(args.max_iterations)
                    .with_cost_tolerance(args.cost_tolerance)
                    .with_parameter_tolerance(args.parameter_tolerance);
                let mut solver = GaussNewton::with_config(config);
                solver.optimize(&mut problem)?
            }
            "DL" => {
                let config = DogLegConfig::new()
                    .with_max_iterations(args.max_iterations)
                    .with_cost_tolerance(args.cost_tolerance)
                    .with_parameter_tolerance(args.parameter_tolerance);
                let mut solver = DogLeg::with_config(config);
                solver.optimize(&mut problem)?
            }
            _ => unreachable!(),
        };

        let time_ms = start.elapsed().as_millis();
        let improvement = if result.initial_cost > 0.0 {
            ((result.initial_cost - result.final_cost) / result.initial_cost) * 100.0
        } else {
            0.0
        };
        let status_str = match result.status {
            OptimizationStatus::Converged
            | OptimizationStatus::CostToleranceReached
            | OptimizationStatus::ParameterToleranceReached
            | OptimizationStatus::GradientToleranceReached => "CONVERGED",
            _ => "NOT_CONVERGED",
        };

        info!("Initial cost: {:.6e}", result.initial_cost);
        info!("Final cost: {:.6e}", result.final_cost);
        info!("Iterations: {}", result.iterations);
        info!("Time: {}ms", time_ms);
        info!("Status: {}\n", status_str);

        all_results.push(BenchmarkResult {
            dataset: dataset_name.to_string(),
            manifold: "SE3".to_string(),
            optimizer: opt_name.to_string(),
            vertices: num_vertices,
            edges: num_edges,
            initial_cost: result.initial_cost,
            final_cost: result.final_cost,
            improvement,
            iterations: result.iterations,
            time_ms,
            status: status_str.to_string(),
        });
    }
    Ok(())
}

fn test_se2_dataset(
    dataset_name: &str,
    args: &Args,
    all_results: &mut Vec<BenchmarkResult>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("TESTING {} (SE2)", dataset_name.to_uppercase());

    let file_path = format!("{}/{}.g2o", ODOMETRY_DATA_DIR_2D, dataset_name);
    let graph = match G2oLoader::load(&file_path) {
        Ok(g) => g,
        Err(e) => {
            warn!("Failed to load {}: {}", file_path, e);
            return Ok(());
        }
    };

    let num_vertices = graph.vertices_se2.len();
    let num_edges = graph.edges_se2.len();
    info!("Loaded: {} vertices, {} edges", num_vertices, num_edges);

    for opt_name in &["LM", "GN", "DL"] {
        info!("--- Testing {} ---", opt_name);

        let (mut problem, _) = build_se2_problem(&graph)?;
        let start = Instant::now();

        let result = match *opt_name {
            "LM" => {
                let config = LevenbergMarquardtConfig::new()
                    .with_max_iterations(args.max_iterations)
                    .with_cost_tolerance(args.cost_tolerance)
                    .with_parameter_tolerance(args.parameter_tolerance);
                let mut solver = LevenbergMarquardt::with_config(config);
                solver.optimize(&mut problem)?
            }
            "GN" => {
                let config = GaussNewtonConfig::new()
                    .with_max_iterations(args.max_iterations)
                    .with_cost_tolerance(args.cost_tolerance)
                    .with_parameter_tolerance(args.parameter_tolerance);
                let mut solver = GaussNewton::with_config(config);
                solver.optimize(&mut problem)?
            }
            "DL" => {
                let config = DogLegConfig::new()
                    .with_max_iterations(args.max_iterations)
                    .with_cost_tolerance(args.cost_tolerance)
                    .with_parameter_tolerance(args.parameter_tolerance);
                let mut solver = DogLeg::with_config(config);
                solver.optimize(&mut problem)?
            }
            _ => unreachable!(),
        };

        let time_ms = start.elapsed().as_millis();
        let improvement = if result.initial_cost > 0.0 {
            ((result.initial_cost - result.final_cost) / result.initial_cost) * 100.0
        } else {
            0.0
        };
        let status_str = match result.status {
            OptimizationStatus::Converged
            | OptimizationStatus::CostToleranceReached
            | OptimizationStatus::ParameterToleranceReached
            | OptimizationStatus::GradientToleranceReached => "CONVERGED",
            _ => "NOT_CONVERGED",
        };

        info!("Initial cost: {:.6e}", result.initial_cost);
        info!("Final cost: {:.6e}", result.final_cost);
        info!("Iterations: {}", result.iterations);
        info!("Time: {}ms", time_ms);
        info!("Status: {}\n", status_str);

        all_results.push(BenchmarkResult {
            dataset: dataset_name.to_string(),
            manifold: "SE2".to_string(),
            optimizer: opt_name.to_string(),
            vertices: num_vertices,
            edges: num_edges,
            initial_cost: result.initial_cost,
            final_cost: result.final_cost,
            improvement,
            iterations: result.iterations,
            time_ms,
            status: status_str.to_string(),
        });
    }
    Ok(())
}

fn main() {
    let args = Args::parse();
    init_logger();

    info!("APEX-SOLVER OPTIMIZER COMPARISON");
    info!("Comparing LM, GN, and DL optimizers on real datasets");

    let mut all_results = Vec::new();

    if let Err(e) = test_se3_dataset("parking-garage", &args, &mut all_results) {
        warn!("Failed to test parking-garage dataset: {}", e);
    }
    if let Err(e) = test_se3_dataset("sphere2500", &args, &mut all_results) {
        warn!("Failed to test sphere2500 dataset: {}", e);
    }
    if let Err(e) = test_se2_dataset("intel", &args, &mut all_results) {
        warn!("Failed to test intel dataset: {}", e);
    }
    if let Err(e) = test_se2_dataset("mit", &args, &mut all_results) {
        warn!("Failed to test mit dataset: {}", e);
    }

    print_summary_table(&all_results);
    info!("Comparison complete!");
}
