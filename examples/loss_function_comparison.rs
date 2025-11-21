use std::collections::HashMap;
use std::time::Instant;
use tracing::{error, info, warn};

use apex_solver::core::loss_functions::*;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BetweenFactorSE2, BetweenFactorSE3};
use apex_solver::init_logger;
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
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
    info!("\n{}", "=".repeat(170));
    info!("=== ROBUST LOSS FUNCTION BENCHMARK RESULTS ===\n");

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

    info!("\n✓ Results written to: {}", filepath);
    Ok(())
}

fn run_optimization(
    problem: &Problem,
    initial_values: &HashMap<String, (ManifoldType, nalgebra::DVector<f64>)>,
    optimizer_name: &str,
    max_iterations: usize,
    cost_tolerance: f64,
    parameter_tolerance: f64,
    _verbose: bool,
) -> Result<(f64, usize, OptimizationStatus, u128), Box<dyn std::error::Error>> {
    let start = Instant::now();

    let result = match optimizer_name {
        "LM" => {
            let config = LevenbergMarquardtConfig::new()
                .with_max_iterations(max_iterations)
                .with_cost_tolerance(cost_tolerance)
                .with_parameter_tolerance(parameter_tolerance);
            let mut solver = LevenbergMarquardt::with_config(config);
            solver.optimize(problem, initial_values)?
        }
        "GN" => {
            let config = GaussNewtonConfig::new()
                .with_max_iterations(max_iterations)
                .with_cost_tolerance(cost_tolerance)
                .with_parameter_tolerance(parameter_tolerance);
            let mut solver = GaussNewton::with_config(config);
            solver.optimize(problem, initial_values)?
        }
        "DL" => {
            let config = DogLegConfig::new()
                .with_max_iterations(max_iterations)
                .with_cost_tolerance(cost_tolerance)
                .with_parameter_tolerance(parameter_tolerance);
            let mut solver = DogLeg::with_config(config);
            solver.optimize(problem, initial_values)?
        }
        _ => unreachable!(),
    };

    let elapsed = start.elapsed().as_millis();

    Ok((result.final_cost, result.iterations, result.status, elapsed))
}

fn benchmark_dataset_se3(
    graph_path: &str,
    dataset_name: &str,
    args: &Args,
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    info!("Loading SE3 dataset: {}", dataset_name);

    let graph = G2oLoader::load(graph_path)?;
    let num_vertices = graph.vertices_se3.len();
    let num_edges = graph.edges_se3.len();

    info!("✓ Loaded {} vertices, {} edges", num_vertices, num_edges);

    // Create initial values from graph vertices
    let mut initial_values = HashMap::new();
    let mut vertex_ids: Vec<_> = graph.vertices_se3.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se3.get(&id) {
            let var_name = format!("x{}", id);
            let quat = vertex.pose.rotation_quaternion();
            let trans = vertex.pose.translation();
            let se3_data = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];
            initial_values.insert(var_name, (ManifoldType::SE3, se3_data));
        }
    }

    // Define ALL 14 loss functions to test
    let loss_configs: Vec<(&str, f64, Box<dyn LossFunction + Send>)> = vec![
        ("L2", 1.0, Box::new(L2Loss)),
        ("L1", 1.0, Box::new(L1Loss)),
        ("Huber", 1.345, Box::new(HuberLoss::new(1.345)?)),
        ("Cauchy", 2.3849, Box::new(CauchyLoss::new(2.3849)?)),
        ("Fair", 1.3999, Box::new(FairLoss::new(1.3999)?)),
        ("Welsch", 2.9846, Box::new(WelschLoss::new(2.9846)?)),
        ("Tukey", 4.6851, Box::new(TukeyBiweightLoss::new(4.6851)?)),
        ("GemanMcClure", 1.0, Box::new(GemanMcClureLoss::new(1.0)?)),
        ("Andrews", 1.339, Box::new(AndrewsWaveLoss::new(1.339)?)),
        ("Ramsay", 0.3, Box::new(RamsayEaLoss::new(0.3)?)),
        ("TrimmedMean", 2.0, Box::new(TrimmedMeanLoss::new(2.0)?)),
        ("Lp(1.5)", 1.5, Box::new(LpNormLoss::new(1.5)?)),
        (
            "Barron(α=0)",
            1.0,
            Box::new(BarronGeneralLoss::new(0.0, 1.0)?),
        ),
        (
            "Barron(α=1)",
            1.0,
            Box::new(BarronGeneralLoss::new(1.0, 1.0)?),
        ),
        (
            "Barron(α=-2)",
            1.0,
            Box::new(BarronGeneralLoss::new(-2.0, 1.0)?),
        ),
        (
            "TDistribution(ν=5)",
            5.0,
            Box::new(TDistributionLoss::new(5.0)?),
        ),
        (
            "AdaptiveBarron",
            1.0,
            Box::new(AdaptiveBarronLoss::new(0.0, 1.0)?),
        ),
    ];

    let optimizers = vec!["LM", "GN", "DL"];
    let mut results = Vec::new();

    // Test each (optimizer, loss_function) combination
    for optimizer_name in &optimizers {
        info!("--- Testing Optimizer: {} ---", optimizer_name);

        for &(loss_name, scale, _) in &loss_configs {
            info!("  Testing {} (scale={:.4})...", loss_name, scale);

            let scale_value = scale;
            // Build problem with this loss function
            let mut problem = Problem::new();

            for edge in &graph.edges_se3 {
                let id0 = format!("x{}", edge.from);
                let id1 = format!("x{}", edge.to);
                let factor = BetweenFactorSE3::new(edge.measurement.clone());

                // Clone the loss function for this edge
                let loss_clone: Option<Box<dyn LossFunction + Send>> = match loss_name {
                    "L2" => Some(Box::new(L2Loss)),
                    "L1" => Some(Box::new(L1Loss)),
                    "Huber" => Some(Box::new(HuberLoss::new(scale_value)?)),
                    "Cauchy" => Some(Box::new(CauchyLoss::new(scale_value)?)),
                    "Fair" => Some(Box::new(FairLoss::new(scale_value)?)),
                    "Welsch" => Some(Box::new(WelschLoss::new(scale_value)?)),
                    "Tukey" => Some(Box::new(TukeyBiweightLoss::new(scale_value)?)),
                    "GemanMcClure" => Some(Box::new(GemanMcClureLoss::new(scale_value)?)),
                    "Andrews" => Some(Box::new(AndrewsWaveLoss::new(scale_value)?)),
                    "Ramsay" => Some(Box::new(RamsayEaLoss::new(scale_value)?)),
                    "TrimmedMean" => Some(Box::new(TrimmedMeanLoss::new(scale_value)?)),
                    "Lp(1.5)" => Some(Box::new(LpNormLoss::new(scale_value)?)),
                    "Barron(α=0)" => {
                        Some(Box::new(BarronGeneralLoss::new(0.0, scale_value)?))
                    }
                    "Barron(α=1)" => {
                        Some(Box::new(BarronGeneralLoss::new(1.0, scale_value)?))
                    }
                    "Barron(α=-2)" => {
                        Some(Box::new(BarronGeneralLoss::new(-2.0, scale_value)?))
                    }
                    "TDistribution(ν=5)" => {
                        Some(Box::new(TDistributionLoss::new(scale_value)?))
                    }
                    "AdaptiveBarron" => {
                        Some(Box::new(AdaptiveBarronLoss::new(0.0, scale_value)?))
                    }
                    _ => None,
                };

                problem.add_residual_block(&[&id0, &id1], Box::new(factor), loss_clone);
            }

            // Compute initial cost
            let variables = problem.initialize_variables(&initial_values);
            let mut variable_name_to_col_idx_dict = HashMap::new();
            let mut col_offset = 0;
            let mut sorted_vars: Vec<_> = variables.keys().cloned().collect();
            sorted_vars.sort();

            for var_name in &sorted_vars {
                variable_name_to_col_idx_dict.insert(var_name.clone(), col_offset);
                col_offset += variables[var_name].get_size();
            }

            let symbolic_structure = problem.build_symbolic_structure(
                &variables,
                &variable_name_to_col_idx_dict,
                col_offset,
            )?;

            let (residual, _) = problem.compute_residual_and_jacobian_sparse(
                &variables,
                &variable_name_to_col_idx_dict,
                &symbolic_structure,
            )?;

            let initial_cost = residual.as_ref().squared_norm_l2();

            // Run optimization
            match run_optimization(
                &problem,
                &initial_values,
                optimizer_name,
                args.max_iterations,
                args.cost_tolerance,
                args.parameter_tolerance,
                args.verbose,
            ) {
                Ok((final_cost, iterations, status, time_ms)) => {
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

                    info!(
                        "    Init: {:.4e}, Final: {:.4e}, Improv: {:.2}%, Iters: {}, Time: {}ms [{}]",
                        initial_cost, final_cost, improvement, iterations, time_ms, status_str
                    );

                    results.push(BenchmarkResult {
                        dataset: dataset_name.to_string(),
                        manifold: "SE3".to_string(),
                        optimizer: optimizer_name.to_string(),
                        loss_function: loss_name.to_string(),
                        scale_param: scale_value,
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

    info!("✓ Loaded {} vertices, {} edges", num_vertices, num_edges);

    // Create initial values
    let mut initial_values = HashMap::new();
    let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se2.get(&id) {
            let var_name = format!("x{}", id);
            let trans = vertex.pose.translation();
            let angle = vertex.pose.rotation_angle();
            let se2_data = dvector![trans.x, trans.y, angle];
            initial_values.insert(var_name, (ManifoldType::SE2, se2_data));
        }
    }

    // Define ALL 14 loss functions to test
    let loss_configs: Vec<(&str, f64, Box<dyn LossFunction + Send>)> = vec![
        ("L2", 1.0, Box::new(L2Loss)),
        ("L1", 1.0, Box::new(L1Loss)),
        ("Huber", 1.345, Box::new(HuberLoss::new(1.345)?)),
        ("Cauchy", 2.3849, Box::new(CauchyLoss::new(2.3849)?)),
        ("Fair", 1.3999, Box::new(FairLoss::new(1.3999)?)),
        ("Welsch", 2.9846, Box::new(WelschLoss::new(2.9846)?)),
        ("Tukey", 4.6851, Box::new(TukeyBiweightLoss::new(4.6851)?)),
        ("GemanMcClure", 1.0, Box::new(GemanMcClureLoss::new(1.0)?)),
        ("Andrews", 1.339, Box::new(AndrewsWaveLoss::new(1.339)?)),
        ("Ramsay", 0.3, Box::new(RamsayEaLoss::new(0.3)?)),
        ("TrimmedMean", 2.0, Box::new(TrimmedMeanLoss::new(2.0)?)),
        ("Lp(1.5)", 1.5, Box::new(LpNormLoss::new(1.5)?)),
        (
            "Barron(α=0)",
            1.0,
            Box::new(BarronGeneralLoss::new(0.0, 1.0)?),
        ),
        (
            "Barron(α=1)",
            1.0,
            Box::new(BarronGeneralLoss::new(1.0, 1.0)?),
        ),
        (
            "Barron(α=-2)",
            1.0,
            Box::new(BarronGeneralLoss::new(-2.0, 1.0)?),
        ),
        (
            "TDistribution(ν=5)",
            5.0,
            Box::new(TDistributionLoss::new(5.0)?),
        ),
        (
            "AdaptiveBarron",
            1.0,
            Box::new(AdaptiveBarronLoss::new(0.0, 1.0)?),
        ),
    ];

    let optimizers = vec!["LM", "GN", "DL"];
    let mut results = Vec::new();

    for optimizer_name in &optimizers {
        info!("--- Testing Optimizer: {} ---", optimizer_name);

        for &(loss_name, scale, _) in &loss_configs {
            info!("  Testing {} (scale={:.4})...", loss_name, scale);

            let scale_value = scale;
            let mut problem = Problem::new();

            for edge in &graph.edges_se2 {
                let id0 = format!("x{}", edge.from);
                let id1 = format!("x{}", edge.to);
                let trans = edge.measurement.translation();
                let angle = edge.measurement.rotation_angle();
                let factor = BetweenFactorSE2::new(trans.x, trans.y, angle);

                let loss_clone: Option<Box<dyn LossFunction + Send>> = match loss_name {
                    "L2" => Some(Box::new(L2Loss)),
                    "L1" => Some(Box::new(L1Loss)),
                    "Huber" => Some(Box::new(HuberLoss::new(scale_value)?)),
                    "Cauchy" => Some(Box::new(CauchyLoss::new(scale_value)?)),
                    "Fair" => Some(Box::new(FairLoss::new(scale_value)?)),
                    "Welsch" => Some(Box::new(WelschLoss::new(scale_value)?)),
                    "Tukey" => Some(Box::new(TukeyBiweightLoss::new(scale_value)?)),
                    "GemanMcClure" => Some(Box::new(GemanMcClureLoss::new(scale_value)?)),
                    "Andrews" => Some(Box::new(AndrewsWaveLoss::new(scale_value)?)),
                    "Ramsay" => Some(Box::new(RamsayEaLoss::new(scale_value)?)),
                    "TrimmedMean" => Some(Box::new(TrimmedMeanLoss::new(scale_value)?)),
                    "Lp(1.5)" => Some(Box::new(LpNormLoss::new(scale_value)?)),
                    "Barron(α=0)" => {
                        Some(Box::new(BarronGeneralLoss::new(0.0, scale_value)?))
                    }
                    "Barron(α=1)" => {
                        Some(Box::new(BarronGeneralLoss::new(1.0, scale_value)?))
                    }
                    "Barron(α=-2)" => {
                        Some(Box::new(BarronGeneralLoss::new(-2.0, scale_value)?))
                    }
                    "TDistribution(ν=5)" => {
                        Some(Box::new(TDistributionLoss::new(scale_value)?))
                    }
                    "AdaptiveBarron" => {
                        Some(Box::new(AdaptiveBarronLoss::new(0.0, scale_value)?))
                    }
                    _ => None,
                };

                problem.add_residual_block(&[&id0, &id1], Box::new(factor), loss_clone);
            }

            // Compute initial cost
            let variables = problem.initialize_variables(&initial_values);
            let mut variable_name_to_col_idx_dict = HashMap::new();
            let mut col_offset = 0;
            let mut sorted_vars: Vec<_> = variables.keys().cloned().collect();
            sorted_vars.sort();

            for var_name in &sorted_vars {
                variable_name_to_col_idx_dict.insert(var_name.clone(), col_offset);
                col_offset += variables[var_name].get_size();
            }

            let symbolic_structure = problem.build_symbolic_structure(
                &variables,
                &variable_name_to_col_idx_dict,
                col_offset,
            )?;

            let (residual, _) = problem.compute_residual_and_jacobian_sparse(
                &variables,
                &variable_name_to_col_idx_dict,
                &symbolic_structure,
            )?;

            let initial_cost = residual.as_ref().squared_norm_l2();

            match run_optimization(
                &problem,
                &initial_values,
                optimizer_name,
                args.max_iterations,
                args.cost_tolerance,
                args.parameter_tolerance,
                args.verbose,
            ) {
                Ok((final_cost, iterations, status, time_ms)) => {
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

                    info!(
                        "    Init: {:.4e}, Final: {:.4e}, Improv: {:.2}%, Iters: {}, Time: {}ms [{}]",
                        initial_cost, final_cost, improvement, iterations, time_ms, status_str
                    );

                    results.push(BenchmarkResult {
                        dataset: dataset_name.to_string(),
                        manifold: "SE2".to_string(),
                        optimizer: optimizer_name.to_string(),
                        loss_function: loss_name.to_string(),
                        scale_param: scale_value,
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
                    error!("    ✗{}", e);
                }
            }
        }
    }

    Ok(results)
}

fn print_analysis(results: &[BenchmarkResult]) {
    info!("\n{}", "=".repeat(80));
    info!("=== ANALYSIS AND RECOMMENDATIONS ===");
    info!("{}", "=".repeat(80));

    // Group by dataset
    let datasets: std::collections::HashSet<String> =
        results.iter().map(|r| r.dataset.clone()).collect();
    let mut datasets_vec: Vec<String> = datasets.into_iter().collect();
    datasets_vec.sort();

    for dataset in &datasets_vec {
        info!("\n📊 Dataset: {}", dataset);

        // Find best converged result for this dataset
        let converged: Vec<&BenchmarkResult> = results
            .iter()
            .filter(|r| r.dataset == *dataset && r.status == "CONVERGED")
            .collect();

        if converged.is_empty() {
            info!("  ⚠ No converged results");
            continue;
        }

        // Best by final cost
        if let Some(best) = converged
            .iter()
            .min_by(|a, b| a.final_cost.total_cmp(&b.final_cost))
        {
            info!(
                "  ✓ Best Overall: {} + {} (cost: {:.4e}, {:.1}% improv, {} iters, {}ms)",
                best.optimizer,
                best.loss_function,
                best.final_cost,
                best.improvement,
                best.iterations,
                best.time_ms
            );
        }

        // Best per optimizer
        for opt in &["LM", "GN", "DL"] {
            let opt_results: Vec<&&BenchmarkResult> =
                converged.iter().filter(|r| r.optimizer == *opt).collect();

            if let Some(best) = opt_results
                .iter()
                .min_by(|a, b| a.final_cost.total_cmp(&b.final_cost))
            {
                info!(
                    "  ✓ Best {}: {} (cost: {:.4e}, {} iters)",
                    opt, best.loss_function, best.final_cost, best.iterations
                );
            }
        }

        // Convergence rate per loss function
        info!("\n  Convergence Rates:");
        let loss_funcs: std::collections::HashSet<String> =
            converged.iter().map(|r| r.loss_function.clone()).collect();

        for loss in loss_funcs {
            let total = results
                .iter()
                .filter(|r| r.dataset == *dataset && r.loss_function == loss)
                .count();
            let conv = results
                .iter()
                .filter(|r| {
                    r.dataset == *dataset && r.loss_function == loss && r.status == "CONVERGED"
                })
                .count();
            let rate = (conv as f64 / total as f64) * 100.0;
            info!("    {:<18}: {:>3}/{} ({:.0}%)", loss, conv, total, rate);
        }
    }

    // Overall recommendation
    info!("\n{}", "=".repeat(80));
    info!("🎯 RECOMMENDED DEFAULTS");
    info!("{}", "=".repeat(80));

    let _converged_all: Vec<&BenchmarkResult> =
        results.iter().filter(|r| r.status == "CONVERGED").collect();

    // Count convergence by loss function across all datasets
    let mut loss_stats: HashMap<String, (usize, usize, f64)> = HashMap::new();

    for result in results {
        let entry = loss_stats
            .entry(result.loss_function.clone())
            .or_insert((0, 0, 0.0));
        entry.0 += 1; // total runs
        if result.status == "CONVERGED" {
            entry.1 += 1; // converged runs
            entry.2 += result.improvement; // sum of improvements
        }
    }

    info!("\nLoss Function Performance Summary:");
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
            "★ Excellent"
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

    info!("   Default Recommendation: Huber (scale=1.345)");
    info!("   Rationale: Convex, reliable convergence, good robustness");
    info!("   For Heavy Outliers: Cauchy or Welsch");
    info!("   Rationale: Stronger outlier suppression, still reliable");
    info!("   For Clean Data: L2");
    info!("   Rationale: Fastest, optimal for Gaussian noise\n");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize logger with INFO level
    init_logger();

    info!("╔═══════════════════════════════════════════════════════════════╗");
    info!("║     ROBUST LOSS FUNCTION COMPARISON BENCHMARK                 ║");
    info!("╚═══════════════════════════════════════════════════════════════╝");

    let mut all_results = Vec::new();

    // Benchmark SE3 datasets (all available 3D pose graphs)
    let se3_datasets = vec![
        ("data/sphere2500.g2o", "sphere2500"),
        ("data/parking-garage.g2o", "parking-garage"),
        ("data/torus3D.g2o", "torus3D"),
    ];

    for (path, name) in &se3_datasets {
        if std::path::Path::new(path).exists() {
            match benchmark_dataset_se3(path, name, &args) {
                Ok(mut results) => all_results.append(&mut results),
                Err(e) => warn!("Failed to benchmark {}: {}", name, e),
            }
        } else {
            warn!("⚠ Skipping {} (file not found)", name);
        }
    }

    // Benchmark SE2 datasets (all available 2D pose graphs)
    let se2_datasets = vec![
        ("data/intel.g2o", "intel"),
        ("data/mit.g2o", "mit"),
        ("data/M3500.g2o", "M3500"),
        ("data/manhattanOlson3500.g2o", "manhattan"),
        ("data/city10000.g2o", "city10000"),
        ("data/ring.g2o", "ring"),
        ("data/ringCity.g2o", "ringCity"),
    ];

    for (path, name) in &se2_datasets {
        if std::path::Path::new(path).exists() {
            match benchmark_dataset_se2(path, name, &args) {
                Ok(mut results) => all_results.append(&mut results),
                Err(e) => warn!("Failed to benchmark {}: {}", name, e),
            }
        } else {
            info!("⚠ Skipping {} (file not found)", name);
        }
    }

    // Print results
    if !all_results.is_empty() {
        print_summary_table(&all_results);

        if let Some(output_path) = &args.output {
            write_csv(&all_results, output_path)?;
        }

        print_analysis(&all_results);
    } else {
        info!("\n⚠ No results to display");
    }

    Ok(())
}
