use std::collections::HashMap;
use std::time::Instant;
use tracing::{error, info, warn};

use apex_solver::JacobianMode;
use apex_solver::apex_io::{
    G2oLoader, Graph, GraphLoader, ODOMETRY_DATA_DIR_2D, ODOMETRY_DATA_DIR_3D,
};
use apex_solver::apex_manifolds::ManifoldType;
use apex_solver::core::VarKey;
use apex_solver::core::loss_functions::*;
use apex_solver::core::problem::Problem;
use apex_solver::factors::BetweenFactor;
use apex_solver::init_logger;
use apex_solver::optimizer::dog_leg::DogLegConfig;
use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
use apex_solver::optimizer::{DogLeg, GaussNewton, LevenbergMarquardt, OptimizationStatus};
use clap::Parser;
use nalgebra::dvector;

#[derive(Parser)]
#[command(name = "loss_function_comparison")]
#[command(about = "Compare robust loss functions with multiple optimizers on pose graph datasets")]
struct Args {
    /// Maximum number of optimization iterations
    #[arg(short, long, default_value = "50")]
    max_iterations: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Cost tolerance for convergence
    #[arg(long, default_value = "1e-4")]
    cost_tolerance: f64,

    /// Parameter tolerance for convergence
    #[arg(long, default_value = "1e-4")]
    parameter_tolerance: f64,

    /// Output CSV file path (optional)
    #[arg(short, long)]
    output: Option<String>,
}

#[derive(Clone)]
struct BenchmarkResult {
    dataset: String,
    manifold: String,
    optimizer: String,
    loss_function: String,
    scale_param: f64,
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
    info!("{}", "=".repeat(170));
    info!("=== ROBUST LOSS FUNCTION BENCHMARK RESULTS ===");

    info!(
        "{:<12} | {:<4} | {:<10} | {:<18} | {:<5} | {:<4} | {:<5} | {:<12} | {:<12} | {:<10} | {:<5} | {:<8} | {:<10}",
        "Dataset",
        "Man",
        "Optimizer",
        "Loss Function",
        "Scale",
        "Verts",
        "Edges",
        "Init Cost",
        "Final Cost",
        "Improv %",
        "Iters",
        "Time(ms)",
        "Status"
    );
    info!("{}", "-".repeat(170));

    for result in results {
        info!(
            "{:<12} | {:<4} | {:<10} | {:<18} | {:<5.2} | {:<4} | {:<5} | {:<12.6e} | {:<12.6e} | {:>9.2}% | {:<5} | {:<8} | {:<10}",
            result.dataset,
            result.manifold,
            result.optimizer,
            result.loss_function,
            result.scale_param,
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

    info!("{}", "-".repeat(170));
}

fn write_csv(
    results: &[BenchmarkResult],
    filepath: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(filepath)?;
    writeln!(
        file,
        "dataset,manifold,optimizer,loss_function,scale_param,vertices,edges,initial_cost,final_cost,improvement,iterations,time_ms,status"
    )?;
    for result in results {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            result.dataset,
            result.manifold,
            result.optimizer,
            result.loss_function,
            result.scale_param,
            result.vertices,
            result.edges,
            result.initial_cost,
            result.final_cost,
            result.improvement,
            result.iterations,
            result.time_ms,
            result.status
        )?;
    }
    info!("Results written to: {}", filepath);
    Ok(())
}

fn make_loss(
    name: &str,
    scale: f64,
) -> Result<Option<Box<dyn LossFunction + Send + Sync>>, Box<dyn std::error::Error>> {
    let loss: Box<dyn LossFunction + Send + Sync> = match name {
        "L2" => Box::new(L2Loss),
        "L1" => Box::new(L1Loss),
        "Huber" => Box::new(HuberLoss::new(scale)?),
        "Cauchy" => Box::new(CauchyLoss::new(scale)?),
        "Fair" => Box::new(FairLoss::new(scale)?),
        "Welsch" => Box::new(WelschLoss::new(scale)?),
        "Tukey" => Box::new(TukeyBiweightLoss::new(scale)?),
        "GemanMcClure" => Box::new(GemanMcClureLoss::new(scale)?),
        "Andrews" => Box::new(AndrewsWaveLoss::new(scale)?),
        "Ramsay" => Box::new(RamsayEaLoss::new(scale)?),
        "TrimmedMean" => Box::new(TrimmedMeanLoss::new(scale)?),
        "Lp(1.5)" => Box::new(LpNormLoss::new(scale)?),
        "Barron(a=0)" => Box::new(BarronGeneralLoss::new(0.0, scale)?),
        "Barron(a=1)" => Box::new(BarronGeneralLoss::new(1.0, scale)?),
        "Barron(a=-2)" => Box::new(BarronGeneralLoss::new(-2.0, scale)?),
        "TDist(v=5)" => Box::new(TDistributionLoss::new(scale)?),
        "AdaptiveBarron" => Box::new(AdaptiveBarronLoss::new(0.0, scale)?),
        _ => return Ok(None),
    };
    Ok(Some(loss))
}

fn build_se3_problem(
    graph: &Graph,
    loss_name: &str,
    scale: f64,
) -> Result<(Problem, HashMap<usize, VarKey>), Box<dyn std::error::Error>> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut var_keys: HashMap<usize, VarKey> = HashMap::new();

    let mut vertex_ids: Vec<_> = graph.vertices_se3.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se3.get(&id) {
            let quat = vertex.pose.rotation_quaternion();
            let trans = vertex.pose.translation();
            let key = problem.add_variable(
                ManifoldType::SE3,
                dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k],
            );
            var_keys.insert(id, key);
        }
    }

    for edge in &graph.edges_se3 {
        if let (Some(&k0), Some(&k1)) = (var_keys.get(&edge.from), var_keys.get(&edge.to)) {
            let loss = make_loss(loss_name, scale)?;
            problem.add_residual_block(
                &[k0, k1],
                Box::new(BetweenFactor::new(edge.measurement.clone())),
                loss,
            );
        }
    }

    Ok((problem, var_keys))
}

fn build_se2_problem(
    graph: &Graph,
    loss_name: &str,
    scale: f64,
) -> Result<(Problem, HashMap<usize, VarKey>), Box<dyn std::error::Error>> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut var_keys: HashMap<usize, VarKey> = HashMap::new();

    let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se2.get(&id) {
            let trans = vertex.pose.translation();
            let angle = vertex.pose.rotation_angle();
            let key = problem.add_variable(ManifoldType::SE2, dvector![trans.x, trans.y, angle]);
            var_keys.insert(id, key);
        }
    }

    for edge in &graph.edges_se2 {
        if let (Some(&k0), Some(&k1)) = (var_keys.get(&edge.from), var_keys.get(&edge.to)) {
            let loss = make_loss(loss_name, scale)?;
            problem.add_residual_block(
                &[k0, k1],
                Box::new(BetweenFactor::new(edge.measurement.clone())),
                loss,
            );
        }
    }

    Ok((problem, var_keys))
}

#[allow(clippy::type_complexity)]
fn run_optimization(
    problem: &mut Problem,
    optimizer_name: &str,
    max_iterations: usize,
    cost_tolerance: f64,
    parameter_tolerance: f64,
) -> Result<(f64, f64, usize, OptimizationStatus, u128), Box<dyn std::error::Error>> {
    let start = Instant::now();

    let result = match optimizer_name {
        "LM" => {
            let config = LevenbergMarquardtConfig::new()
                .with_max_iterations(max_iterations)
                .with_cost_tolerance(cost_tolerance)
                .with_parameter_tolerance(parameter_tolerance);
            let mut solver = LevenbergMarquardt::with_config(config);
            solver.optimize(problem)?
        }
        "GN" => {
            let config = GaussNewtonConfig::new()
                .with_max_iterations(max_iterations)
                .with_cost_tolerance(cost_tolerance)
                .with_parameter_tolerance(parameter_tolerance);
            let mut solver = GaussNewton::with_config(config);
            solver.optimize(problem)?
        }
        "DL" => {
            let config = DogLegConfig::new()
                .with_max_iterations(max_iterations)
                .with_cost_tolerance(cost_tolerance)
                .with_parameter_tolerance(parameter_tolerance);
            let mut solver = DogLeg::with_config(config);
            solver.optimize(problem)?
        }
        _ => unreachable!(),
    };

    let elapsed = start.elapsed().as_millis();
    Ok((
        result.initial_cost,
        result.final_cost,
        result.iterations,
        result.status,
        elapsed,
    ))
}

const LOSS_CONFIGS: &[(&str, f64)] = &[
    ("L2", 1.0),
    ("L1", 1.0),
    ("Huber", 1.345),
    ("Cauchy", 2.3849),
    ("Fair", 1.3999),
    ("Welsch", 2.9846),
    ("Tukey", 4.6851),
    ("GemanMcClure", 1.0),
    ("Andrews", 1.339),
    ("Ramsay", 0.3),
    ("TrimmedMean", 2.0),
    ("Lp(1.5)", 1.5),
    ("Barron(a=0)", 1.0),
    ("Barron(a=1)", 1.0),
    ("Barron(a=-2)", 1.0),
    ("TDist(v=5)", 5.0),
    ("AdaptiveBarron", 1.0),
];

fn benchmark_dataset_se3(
    graph_path: &str,
    dataset_name: &str,
    args: &Args,
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    info!("Loading SE3 dataset: {}", dataset_name);
    let graph = G2oLoader::load(graph_path)?;
    let num_vertices = graph.vertices_se3.len();
    let num_edges = graph.edges_se3.len();
    info!("Loaded {} vertices, {} edges", num_vertices, num_edges);

    let optimizers = ["LM", "GN", "DL"];
    let mut results = Vec::new();

    for optimizer_name in &optimizers {
        info!("--- Testing Optimizer: {} ---", optimizer_name);

        for &(loss_name, scale) in LOSS_CONFIGS {
            info!("  Testing {} (scale={:.4})...", loss_name, scale);

            let (mut problem, _) = build_se3_problem(&graph, loss_name, scale)?;

            match run_optimization(
                &mut problem,
                optimizer_name,
                args.max_iterations,
                args.cost_tolerance,
                args.parameter_tolerance,
            ) {
                Ok((initial_cost, final_cost, iterations, status, time_ms)) => {
                    let improvement = if initial_cost > 0.0 {
                        ((initial_cost - final_cost) / initial_cost) * 100.0
                    } else {
                        0.0
                    };
                    let status_str = match status {
                        OptimizationStatus::Converged => "CONVERGED",
                        OptimizationStatus::MaxIterationsReached => "MAX_ITERS",
                        _ => "OTHER",
                    };

                    if args.verbose {
                        info!(
                            "    Init: {:.4e}, Final: {:.4e}, Improv: {:.2}%, Iters: {}, Time: {}ms [{}]",
                            initial_cost, final_cost, improvement, iterations, time_ms, status_str
                        );
                    }

                    results.push(BenchmarkResult {
                        dataset: dataset_name.to_string(),
                        manifold: "SE3".to_string(),
                        optimizer: optimizer_name.to_string(),
                        loss_function: loss_name.to_string(),
                        scale_param: scale,
                        vertices: num_vertices,
                        edges: num_edges,
                        initial_cost,
                        final_cost,
                        improvement,
                        iterations,
                        time_ms,
                        status: status_str.to_string(),
                    });
                }
                Err(e) => {
                    error!("    {}", e);
                }
            }
        }
    }

    Ok(results)
}

fn benchmark_dataset_se2(
    graph_path: &str,
    dataset_name: &str,
    args: &Args,
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    info!("Loading SE2 dataset: {}", dataset_name);
    let graph = G2oLoader::load(graph_path)?;
    let num_vertices = graph.vertices_se2.len();
    let num_edges = graph.edges_se2.len();
    info!("Loaded {} vertices, {} edges", num_vertices, num_edges);

    let optimizers = ["LM", "GN", "DL"];
    let mut results = Vec::new();

    for optimizer_name in &optimizers {
        info!("--- Testing Optimizer: {} ---", optimizer_name);

        for &(loss_name, scale) in LOSS_CONFIGS {
            info!("  Testing {} (scale={:.4})...", loss_name, scale);

            let (mut problem, _) = build_se2_problem(&graph, loss_name, scale)?;

            match run_optimization(
                &mut problem,
                optimizer_name,
                args.max_iterations,
                args.cost_tolerance,
                args.parameter_tolerance,
            ) {
                Ok((initial_cost, final_cost, iterations, status, time_ms)) => {
                    let improvement = if initial_cost > 0.0 {
                        ((initial_cost - final_cost) / initial_cost) * 100.0
                    } else {
                        0.0
                    };
                    let status_str = match status {
                        OptimizationStatus::Converged => "CONVERGED",
                        OptimizationStatus::MaxIterationsReached => "MAX_ITERS",
                        _ => "OTHER",
                    };

                    if args.verbose {
                        info!(
                            "    Init: {:.4e}, Final: {:.4e}, Improv: {:.2}%, Iters: {}, Time: {}ms [{}]",
                            initial_cost, final_cost, improvement, iterations, time_ms, status_str
                        );
                    }

                    results.push(BenchmarkResult {
                        dataset: dataset_name.to_string(),
                        manifold: "SE2".to_string(),
                        optimizer: optimizer_name.to_string(),
                        loss_function: loss_name.to_string(),
                        scale_param: scale,
                        vertices: num_vertices,
                        edges: num_edges,
                        initial_cost,
                        final_cost,
                        improvement,
                        iterations,
                        time_ms,
                        status: status_str.to_string(),
                    });
                }
                Err(e) => {
                    error!("    {}", e);
                }
            }
        }
    }

    Ok(results)
}

fn print_analysis(results: &[BenchmarkResult]) {
    info!("{}", "=".repeat(80));
    info!("=== ANALYSIS AND RECOMMENDATIONS ===");
    info!("{}", "=".repeat(80));

    let mut datasets_vec: Vec<String> = results.iter().map(|r| r.dataset.clone()).collect();
    datasets_vec.sort();
    datasets_vec.dedup();

    for dataset in &datasets_vec {
        info!("Dataset: {}", dataset);

        let converged: Vec<&BenchmarkResult> = results
            .iter()
            .filter(|r| r.dataset == *dataset && r.status == "CONVERGED")
            .collect();

        if converged.is_empty() {
            info!("  No converged results");
            continue;
        }

        if let Some(best) = converged
            .iter()
            .min_by(|a, b| a.final_cost.total_cmp(&b.final_cost))
        {
            info!(
                "  Best Overall: {} + {} (cost: {:.4e}, {:.1}% improv, {} iters, {}ms)",
                best.optimizer,
                best.loss_function,
                best.final_cost,
                best.improvement,
                best.iterations,
                best.time_ms
            );
        }

        for opt in &["LM", "GN", "DL"] {
            let opt_results: Vec<_> = converged.iter().filter(|r| r.optimizer == *opt).collect();
            if let Some(best) = opt_results
                .iter()
                .min_by(|a, b| a.final_cost.total_cmp(&b.final_cost))
            {
                info!(
                    "  Best {}: {} (cost: {:.4e}, {} iters)",
                    opt, best.loss_function, best.final_cost, best.iterations
                );
            }
        }
    }

    info!("{}", "=".repeat(80));
    info!("RECOMMENDED DEFAULTS");

    let mut loss_stats: HashMap<String, (usize, usize, f64)> = HashMap::new();
    for result in results {
        let entry = loss_stats
            .entry(result.loss_function.clone())
            .or_insert((0, 0, 0.0));
        entry.0 += 1;
        if result.status == "CONVERGED" {
            entry.1 += 1;
            entry.2 += result.improvement;
        }
    }

    info!(
        "{:<18} | {:>12} | {:>12} | {:>15}",
        "Loss Function", "Conv Rate", "Avg Improv", "Recommendation"
    );
    info!("{}", "-".repeat(65));

    let mut loss_vec: Vec<_> = loss_stats.iter().collect();
    loss_vec.sort_by(|a, b| {
        let rate_a = a.1.1 as f64 / a.1.0 as f64;
        let rate_b = b.1.1 as f64 / b.1.0 as f64;
        rate_b.total_cmp(&rate_a)
    });

    for (loss, (total, converged, sum_improv)) in loss_vec {
        let conv_rate = (*converged as f64 / *total as f64) * 100.0;
        let avg_improv = if *converged > 0 {
            sum_improv / *converged as f64
        } else {
            0.0
        };
        let recommendation = if conv_rate >= 95.0 && avg_improv >= 95.0 {
            "Excellent"
        } else if conv_rate >= 80.0 && avg_improv >= 90.0 {
            "Good"
        } else if conv_rate >= 70.0 {
            "Fair"
        } else {
            "Poor"
        };
        info!(
            "{:<18} | {:>11.0}% | {:>11.1}% | {:>15}",
            loss, conv_rate, avg_improv, recommendation
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    init_logger();

    info!("ROBUST LOSS FUNCTION COMPARISON BENCHMARK");

    let mut all_results = Vec::new();

    for (path, name) in &[
        (
            format!("{}/sphere2500.g2o", ODOMETRY_DATA_DIR_3D),
            "sphere2500",
        ),
        (
            format!("{}/parking-garage.g2o", ODOMETRY_DATA_DIR_3D),
            "parking-garage",
        ),
    ] {
        if std::path::Path::new(path.as_str()).exists() {
            match benchmark_dataset_se3(path.as_str(), name, &args) {
                Ok(mut results) => all_results.append(&mut results),
                Err(e) => warn!("Failed to benchmark {}: {}", name, e),
            }
        } else {
            warn!("Skipping {} (file not found)", name);
        }
    }

    for (path, name) in &[
        (format!("{}/intel.g2o", ODOMETRY_DATA_DIR_2D), "intel"),
        (format!("{}/mit.g2o", ODOMETRY_DATA_DIR_2D), "mit"),
        (format!("{}/ring.g2o", ODOMETRY_DATA_DIR_2D), "ring"),
    ] {
        if std::path::Path::new(path.as_str()).exists() {
            match benchmark_dataset_se2(path.as_str(), name, &args) {
                Ok(mut results) => all_results.append(&mut results),
                Err(e) => warn!("Failed to benchmark {}: {}", name, e),
            }
        } else {
            info!("Skipping {} (file not found)", name);
        }
    }

    if !all_results.is_empty() {
        print_summary_table(&all_results);
        if let Some(output_path) = &args.output {
            write_csv(&all_results, output_path)?;
        }
        print_analysis(&all_results);
    } else {
        info!("No results to display");
    }

    Ok(())
}
