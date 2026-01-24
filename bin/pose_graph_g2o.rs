use std::collections::HashMap;
use std::time::Instant;

use apex_solver::core::loss_functions::*;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BetweenFactor, PriorFactor};
use apex_solver::init_logger;
use apex_solver::io::{G2oLoader, Graph, GraphLoader};
use apex_solver::manifold::LieGroup;
use apex_solver::manifold::ManifoldType;
use apex_solver::manifold::se2::SE2;
use apex_solver::manifold::se3::SE3;
use apex_solver::optimizer::dog_leg::DogLegConfig;
use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
use apex_solver::optimizer::{DogLeg, GaussNewton, LevenbergMarquardt, OptimizationStatus};
use clap::Parser;
use nalgebra::dvector;
use tracing::{error, info, warn};

#[derive(Parser)]
#[command(name = "pose_graph_g2o")]
#[command(about = "Optimize 2D and 3D pose graphs from G2O datasets")]
struct Args {
    /// G2O dataset file to load (without .g2o extension). Use "all" to test all datasets
    #[arg(short, long, default_value = "all")]
    dataset: String,

    /// Maximum number of optimization iterations
    #[arg(short, long, default_value = "100")]
    max_iterations: usize,

    /// Cost tolerance for convergence
    #[arg(long, default_value = "1e-4")]
    cost_tolerance: f64,

    /// Parameter tolerance for convergence
    #[arg(long, default_value = "1e-4")]
    parameter_tolerance: f64,

    /// Optimizer type: "lm" (Levenberg-Marquardt), "gn" (Gauss-Newton), "dl" (Dog Leg), or "all"
    #[arg(short, long, default_value = "lm")]
    optimizer: String,

    /// Optional path to save optimized graph (e.g., output/optimized.g2o)
    #[arg(long)]
    save_output: Option<std::path::PathBuf>,

    /// Enable real-time Rerun visualization
    /// (Requires the `visualization` feature to be enabled)
    #[arg(long)]
    #[cfg(feature = "visualization")]
    with_visualizer: bool,

    /// Robust loss function to use: "l2", "l1", "huber", "cauchy", "fair", "welsch", "tukey", "geman", "andrews", "ramsay", "trimmed", "lp", "barron0", "barron1", "barron-2", "t-distribution", "adaptive-barron"
    #[arg(long, default_value = "l2")]
    loss_function: String,

    /// Scale parameter for the loss function (default: 1.345 for Huber)
    #[arg(long)]
    loss_scale: Option<f64>,

    /// Enable detailed profiling output with timing breakdown
    #[arg(long)]
    profile: bool,
}

// ============================================================================
// UNIFIED COST COMPUTATION
// ============================================================================
// These functions compute cost directly from G2O graph data, providing both
// information-weighted (chi2) and unweighted cost metrics.
//
// Chi2 formula: cost = sum_i r_i^T * Omega_i * r_i
// Unweighted formula: cost = 0.5 * sum_i ||r_i||²
// ============================================================================

/// Dual cost metrics for pose graph optimization
#[derive(Debug, Clone, Copy)]
struct CostMetrics {
    /// Chi-squared cost: sum of r^T * Omega * r (information-weighted)
    chi2_cost: f64,
    /// Unweighted cost: 0.5 * sum ||r||^2
    /// Note: This field is computed for consistency with odometry_pose_benchmark.rs
    /// but not currently used since initial_cost comes from the Problem's residual
    #[allow(dead_code)]
    unweighted_cost: f64,
}

/// Compute both SE2 cost metrics from G2O graph data
/// - Chi-squared: sum of r^T * Omega * r (information-weighted)
/// - Unweighted: 0.5 * sum ||r||^2
fn compute_se2_cost_metrics(graph: &Graph) -> CostMetrics {
    let mut chi2_cost = 0.0;
    let mut unweighted_cost = 0.0;

    for edge in &graph.edges_se2 {
        let from_idx = edge.from;
        let to_idx = edge.to;

        if let (Some(v_from), Some(v_to)) = (
            graph.vertices_se2.get(&from_idx),
            graph.vertices_se2.get(&to_idx),
        ) {
            let pose_i = v_from.pose.clone();
            let pose_j = v_to.pose.clone();

            // T_i^-1 * T_j
            let actual_relative = pose_i.inverse(None).compose(&pose_j, None, None);

            // T_ij^-1 * actual_relative
            let error = edge
                .measurement
                .inverse(None)
                .compose(&actual_relative, None, None);

            let residual_tangent = error.log(None);
            let residual_vec: nalgebra::DVector<f64> = residual_tangent.into();

            // Chi-squared: r^T * Omega * r (information-weighted)
            let weighted_sq = &residual_vec.transpose() * edge.information * &residual_vec;
            chi2_cost += weighted_sq[(0, 0)];

            // Unweighted: 0.5 * ||r||^2
            unweighted_cost += 0.5 * residual_vec.norm_squared();
        }
    }

    CostMetrics {
        chi2_cost,
        unweighted_cost,
    }
}

/// Compute both SE3 cost metrics from G2O graph data
/// - Chi-squared: sum of r^T * Omega * r (information-weighted)
/// - Unweighted: 0.5 * sum ||r||^2
fn compute_se3_cost_metrics(graph: &Graph) -> CostMetrics {
    let mut chi2_cost = 0.0;
    let mut unweighted_cost = 0.0;

    for edge in &graph.edges_se3 {
        let from_idx = edge.from;
        let to_idx = edge.to;

        if let (Some(v_from), Some(v_to)) = (
            graph.vertices_se3.get(&from_idx),
            graph.vertices_se3.get(&to_idx),
        ) {
            let pose_i = v_from.pose.clone();
            let pose_j = v_to.pose.clone();

            // T_i^-1 * T_j
            let actual_relative = pose_i.inverse(None).compose(&pose_j, None, None);

            // T_ij^-1 * actual_relative
            let error = edge
                .measurement
                .inverse(None)
                .compose(&actual_relative, None, None);

            let residual_tangent = error.log(None);
            let residual_vec: nalgebra::DVector<f64> = residual_tangent.into();

            // Chi-squared: r^T * Omega * r (information-weighted)
            let weighted_sq = &residual_vec.transpose() * edge.information * &residual_vec;
            chi2_cost += weighted_sq[(0, 0)];

            // Unweighted: 0.5 * ||r||^2
            unweighted_cost += 0.5 * residual_vec.norm_squared();
        }
    }

    CostMetrics {
        chi2_cost,
        unweighted_cost,
    }
}

#[derive(Clone)]
struct DatasetResult {
    dataset: String,
    manifold: String,
    optimizer: String,
    vertices: usize,
    edges: usize,
    // Chi2 metrics (information-weighted)
    initial_chi2: f64,
    final_chi2: f64,
    chi2_improvement: f64,
    // Unweighted cost metrics
    initial_cost: f64,
    final_cost: f64,
    improvement: f64,
    iterations: usize,
    time_ms: u128,
    status: String,
}

fn format_summary_table(results: &[DatasetResult]) {
    info!("Final summary table:");

    info!(
        "{:<16} | {:<7} | {:<4} | {:<8} | {:<6} | {:<12} | {:<12} | {:<9} | {:<12} | {:<12} | {:<9} | {:<5} | {:<9} | {:<12}",
        "Dataset",
        "Manifold",
        "Opt",
        "Vertices",
        "Edges",
        "Init Chi2",
        "Final Chi2",
        "Chi2 Imp%",
        "Init Cost",
        "Final Cost",
        "Cost Imp%",
        "Iters",
        "Time(ms)",
        "Status"
    );
    info!("{}", "-".repeat(180));

    for r in results {
        info!(
            "{:<16} | {:<7} | {:<4} | {:<8} | {:<6} | {:<12.4e} | {:<12.4e} | {:>8.2}% | {:<12.4e} | {:<12.4e} | {:>8.2}% | {:<5} | {:<9} | {:<12}",
            r.dataset,
            r.manifold,
            r.optimizer,
            r.vertices,
            r.edges,
            r.initial_chi2,
            r.final_chi2,
            r.chi2_improvement * 100.0,
            r.initial_cost,
            r.final_cost,
            r.improvement * 100.0,
            r.iterations,
            r.time_ms,
            r.status
        );
    }

    info!("{}", "-".repeat(180));

    let converged_count = results.iter().filter(|r| r.status == "CONVERGED").count();
    let total_count = results.len();
    info!(
        "Summary: {}/{} datasets converged successfully",
        converged_count, total_count
    );

    if converged_count > 0 {
        let avg_time: f64 = results
            .iter()
            .filter(|r| r.status == "CONVERGED")
            .map(|r| r.time_ms as f64)
            .sum::<f64>()
            / converged_count as f64;
        let avg_iters: f64 = results
            .iter()
            .filter(|r| r.status == "CONVERGED")
            .map(|r| r.iterations as f64)
            .sum::<f64>()
            / converged_count as f64;
        info!("Average time for converged datasets: {:.1}ms", avg_time);
        info!(
            "Average iterations for converged datasets: {:.1}",
            avg_iters
        );
    }
}

fn create_loss_function(
    loss_name: &str,
    scale: Option<f64>,
) -> Result<Option<Box<dyn LossFunction + Send>>, Box<dyn std::error::Error>> {
    let loss_lower = loss_name.to_lowercase();

    let default_scale = match loss_lower.as_str() {
        "l2" | "l1" => {
            return Ok(match loss_lower.as_str() {
                "l2" => Some(Box::new(L2Loss)),
                "l1" => Some(Box::new(L1Loss)),
                _ => None,
            });
        }
        "huber" => 1.345,
        "cauchy" => 2.3849,
        "fair" => 1.3999,
        "welsch" => 2.9846,
        "tukey" => 4.6851,
        "geman" | "gemanmcclure" => 1.0,
        "andrews" => 1.339,
        "ramsay" => 0.3,
        "trimmed" | "trimmedmean" => 2.0,
        "lp" => 1.5,
        "barron0" | "barron1" | "barron-2" => 1.0,
        "t-distribution" | "tdistribution" => 5.0,
        "adaptive-barron" | "adaptivebarron" => 1.0,
        _ => {
            return Err(format!("Unknown loss function: {}. Valid options: l2, l1, huber, cauchy, fair, welsch, tukey, geman, andrews, ramsay, trimmed, lp, barron0, barron1, barron-2, t-distribution, adaptive-barron", loss_name).into());
        }
    };

    let scale_param = scale.unwrap_or(default_scale);

    let loss: Box<dyn LossFunction + Send> = match loss_lower.as_str() {
        "huber" => Box::new(HuberLoss::new(scale_param)?),
        "cauchy" => Box::new(CauchyLoss::new(scale_param)?),
        "fair" => Box::new(FairLoss::new(scale_param)?),
        "welsch" => Box::new(WelschLoss::new(scale_param)?),
        "tukey" => Box::new(TukeyBiweightLoss::new(scale_param)?),
        "geman" | "gemanmcclure" => Box::new(GemanMcClureLoss::new(scale_param)?),
        "andrews" => Box::new(AndrewsWaveLoss::new(scale_param)?),
        "ramsay" => Box::new(RamsayEaLoss::new(scale_param)?),
        "trimmed" | "trimmedmean" => Box::new(TrimmedMeanLoss::new(scale_param)?),
        "lp" => Box::new(LpNormLoss::new(scale_param)?),
        "barron0" => Box::new(BarronGeneralLoss::new(0.0, scale_param)?),
        "barron1" => Box::new(BarronGeneralLoss::new(1.0, scale_param)?),
        "barron-2" => Box::new(BarronGeneralLoss::new(-2.0, scale_param)?),
        "t-distribution" | "tdistribution" => Box::new(TDistributionLoss::new(scale_param)?),
        "adaptive-barron" | "adaptivebarron" => {
            Box::new(AdaptiveBarronLoss::new(0.0, scale_param)?)
        }
        _ => unreachable!(),
    };

    Ok(Some(loss))
}

fn test_se2_dataset(
    dataset_name: &str,
    args: &Args,
) -> Result<DatasetResult, apex_solver::error::ApexSolverError> {
    info!(
        "Testing {} dataset by loading {}.g2o for SE2 optimization",
        dataset_name.to_uppercase(),
        dataset_name
    );

    // Apply dataset-specific optimizations
    let (cost_tol, param_tol, max_iter) = match dataset_name {
        "manhattanOlson3500" => (1e-3, 1e-3, args.max_iterations),
        _ => (
            args.cost_tolerance,
            args.parameter_tolerance,
            args.max_iterations,
        ),
    };

    let load_start = Instant::now();
    let dataset_path = format!("data/odometry/{}.g2o", dataset_name);
    let mut graph = G2oLoader::load(&dataset_path)?;
    let load_time = load_start.elapsed();

    if args.profile {
        info!(
            "[PROFILE] Graph load time: {:.2}ms",
            load_time.as_secs_f64() * 1000.0
        );
    }

    let num_vertices = graph.vertices_se2.len();
    let num_edges = graph.edges_se2.len();

    info!(
        "Graph Statistics: Vertices: {}, Edges: {}",
        num_vertices, num_edges
    );

    // Compute initial chi2 metrics from graph
    let initial_metrics = compute_se2_cost_metrics(&graph);

    if num_vertices == 0 {
        return Err(apex_solver::io::IoError::UnsupportedFormat(format!(
            "No SE2 vertices found in dataset {}",
            dataset_name
        ))
        .log()
        .into());
    }

    let setup_start = Instant::now();
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se2.get(&id) {
            let var_name = format!("x{}", id);
            let se2_data = dvector![vertex.x(), vertex.y(), vertex.theta()];
            initial_values.insert(var_name, (ManifoldType::SE2, se2_data));
        }
    }

    let optimizer_type = args.optimizer.to_lowercase();
    let needs_prior = optimizer_type == "gn"
        || optimizer_type == "gauss-newton"
        || optimizer_type == "dl"
        || optimizer_type == "dogleg"
        || optimizer_type == "dog-leg";

    if needs_prior
        && let Some(&first_id) = vertex_ids.first()
        && let Some(first_vertex) = graph.vertices_se2.get(&first_id)
    {
        let var_name = format!("x{}", first_id);
        let trans = first_vertex.pose.translation();
        let angle = first_vertex.pose.rotation_angle();
        let prior_value = dvector![trans.x, trans.y, angle];

        let prior_factor = PriorFactor {
            data: prior_value.clone(),
        };
        let huber_loss = HuberLoss::new(1.0)?;
        problem.add_residual_block(
            &[&var_name],
            Box::new(prior_factor),
            Some(Box::new(huber_loss)),
        );
    } else if optimizer_type == "lm" || optimizer_type == "levenberg-marquardt" {
        let first_var_name = format!("x{}", vertex_ids[0]);
        problem.fix_variable(&first_var_name, 0);
        problem.fix_variable(&first_var_name, 1);
        problem.fix_variable(&first_var_name, 2);
    }

    let loss_fn = create_loss_function(&args.loss_function, args.loss_scale).map_err(|e| {
        apex_solver::error::ApexSolverError::from(
            apex_solver::core::CoreError::InvalidInput(e.to_string()).log(),
        )
    })?;

    for edge in &graph.edges_se2 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);

        let relative_pose = edge.measurement.clone();
        let between_factor = BetweenFactor::new(relative_pose);

        let edge_loss = if loss_fn.is_some() {
            create_loss_function(&args.loss_function, args.loss_scale).map_err(|e| {
                apex_solver::error::ApexSolverError::from(
                    apex_solver::core::CoreError::InvalidInput(e.to_string()).log(),
                )
            })?
        } else {
            None
        };

        problem.add_residual_block(&[&id0, &id1], Box::new(between_factor), edge_loss);
    }

    let setup_time = setup_start.elapsed();

    if args.profile {
        info!(
            "[PROFILE] Problem setup time: {:.2}ms",
            setup_time.as_secs_f64() * 1000.0
        );
    }

    info!(
        "Problem Structure: Variables: {}, Prior factors: {}, Between factors: {}",
        initial_values.len(),
        if needs_prior { "1" } else { "0" },
        graph.edges_se2.len()
    );

    let init_cost_start = Instant::now();
    let variables = problem.initialize_variables(&initial_values);
    let mut variable_name_to_col_idx_dict = HashMap::new();
    let mut col_offset = 0;
    let mut sorted_vars: Vec<_> = variables.keys().cloned().collect();
    sorted_vars.sort();
    for var_name in &sorted_vars {
        variable_name_to_col_idx_dict.insert(var_name.clone(), col_offset);
        col_offset += variables[var_name].get_size();
    }

    let symbolic_structure = problem
        .build_symbolic_structure(&variables, &variable_name_to_col_idx_dict, col_offset)
        .map_err(|e| {
            apex_solver::core::CoreError::SymbolicStructure(format!(
                "Failed to build symbolic structure for dataset {}",
                dataset_name
            ))
            .log_with_source(e)
        })?;

    let (residual, _jacobian) = problem
        .compute_residual_and_jacobian_sparse(
            &variables,
            &variable_name_to_col_idx_dict,
            &symbolic_structure,
        )
        .map_err(|e| {
            apex_solver::core::CoreError::FactorLinearization(format!(
                "Failed to compute residual and jacobian for dataset {}",
                dataset_name
            ))
            .log_with_source(e)
        })?;

    let initial_cost = residual.as_ref().squared_norm_l2();
    let init_cost_time = init_cost_start.elapsed();

    if args.profile {
        info!(
            "[PROFILE] Initial cost computation: {:.2}ms",
            init_cost_time.as_secs_f64() * 1000.0
        );
    }

    let optimizer_name = match args.optimizer.to_lowercase().as_str() {
        "gn" => "GN",
        "lm" => "LM",
        "dl" => "DL",
        _ => {
            warn!(
                "Invalid optimizer '{}'. Using LM (Levenberg-Marquardt) as default.",
                args.optimizer
            );
            "LM"
        }
    };

    let opt_start = Instant::now();
    let result = match optimizer_name {
        "GN" => {
            let config = GaussNewtonConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-10);

            let mut solver = GaussNewton::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
        "DL" => {
            let config = DogLegConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-10);

            let mut solver = DogLeg::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
        _ => {
            let config = LevenbergMarquardtConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-10);

            let mut solver = LevenbergMarquardt::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
    };

    let optimization_time = opt_start.elapsed();

    if args.profile {
        info!(
            "[PROFILE] Optimization time: {:.2}ms",
            optimization_time.as_secs_f64() * 1000.0
        );
        info!("[PROFILE] Total iterations: {}", result.iterations);
        info!(
            "[PROFILE] Time per iteration: {:.2}ms",
            optimization_time.as_secs_f64() * 1000.0 / result.iterations as f64
        );
        let total = load_time + setup_time + init_cost_time + optimization_time;
        info!(
            "Profile summary: Load: {:.2}ms, Setup: {:.2}ms, Init Cost: {:.2}ms, Optimize: {:.2}ms ({} iters, {:.2}ms/iter), TOTAL: {:.2}ms",
            load_time.as_secs_f64() * 1000.0,
            setup_time.as_secs_f64() * 1000.0,
            init_cost_time.as_secs_f64() * 1000.0,
            optimization_time.as_secs_f64() * 1000.0,
            result.iterations,
            optimization_time.as_secs_f64() * 1000.0 / result.iterations as f64,
            total.as_secs_f64() * 1000.0
        );
    }

    // Update graph vertices with optimized values for chi2 computation
    for (var_name, var_enum) in &result.parameters {
        if let Some(id_str) = var_name.strip_prefix("x")
            && let Ok(id) = id_str.parse::<usize>()
            && let Some(vertex) = graph.vertices_se2.get_mut(&id)
        {
            let val = var_enum.to_vector();
            vertex.pose = SE2::from_xy_angle(val[0], val[1], val[2]);
        }
    }

    // Compute final chi2 metrics from updated graph
    let final_metrics = compute_se2_cost_metrics(&graph);

    let final_cost = result.final_cost;
    let improvement = (initial_cost - final_cost) / initial_cost;
    let chi2_improvement = if initial_metrics.chi2_cost > 0.0 {
        (initial_metrics.chi2_cost - final_metrics.chi2_cost) / initial_metrics.chi2_cost
    } else {
        0.0
    };
    let iterations = result.iterations;

    info!("Optimization completed:");
    info!("  Status: {:?}", result.status);
    info!("  Iterations: {}", iterations);
    info!(
        "  Initial chi2: {:.6e}, Final chi2: {:.6e}, Chi2 reduction: {:.2}%",
        initial_metrics.chi2_cost,
        final_metrics.chi2_cost,
        chi2_improvement * 100.0
    );
    info!(
        "  Initial cost: {:.6e}, Final cost: {:.6e}, Cost reduction: {:.2}%",
        initial_cost,
        final_cost,
        improvement * 100.0
    );
    info!(
        "  Optimization time: {:.2}ms",
        optimization_time.as_secs_f64() * 1000.0
    );

    // Save optimized graph if requested
    if let Some(output_base) = &args.save_output {
        info!("Saving optimized graph...");

        // Determine output path - if it's a directory, auto-generate filename
        let output_path = if output_base.is_dir() || output_base.to_string_lossy().ends_with('/') {
            output_base.join(format!("{}_optimized.g2o", dataset_name))
        } else {
            output_base.clone()
        };

        // Create output directory if it doesn't exist
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                apex_solver::io::IoError::log_with_source(
                    apex_solver::io::IoError::FileCreationFailed {
                        path: parent.display().to_string(),
                        reason: "Failed to create directory".to_string(),
                    },
                    e,
                )
            })?;
        }

        // Reconstruct graph from optimized variables
        use apex_solver::io::Graph;
        let optimized_graph = Graph::from_optimized_variables(&result.parameters, &graph);

        // Write to file (default: G2O format)
        use apex_solver::io::GraphLoader;
        G2oLoader::write(&optimized_graph, &output_path)?;

        info!("Saved optimized graph to: {}", output_path.display());
    }

    let status = match result.status {
        OptimizationStatus::Converged
        | OptimizationStatus::CostToleranceReached
        | OptimizationStatus::GradientToleranceReached
        | OptimizationStatus::ParameterToleranceReached => "CONVERGED".to_string(),
        OptimizationStatus::MaxIterationsReached => "ITER_LIMIT".to_string(),
        OptimizationStatus::NumericalFailure => "NUM_FAILURE".to_string(),
        _ => "UNKNOWN".to_string(),
    };

    Ok(DatasetResult {
        dataset: dataset_name.to_string(),
        manifold: "SE2".to_string(),
        optimizer: optimizer_name.to_string(),
        vertices: num_vertices,
        edges: num_edges,
        initial_chi2: initial_metrics.chi2_cost,
        final_chi2: final_metrics.chi2_cost,
        chi2_improvement,
        initial_cost,
        final_cost,
        improvement,
        iterations,
        time_ms: optimization_time.as_millis(),
        status,
    })
}

fn test_se3_dataset(
    dataset_name: &str,
    args: &Args,
) -> Result<DatasetResult, apex_solver::error::ApexSolverError> {
    info!(
        "Testing {} SE3 dataset by loading {}.g2o for optimization",
        dataset_name.to_uppercase(),
        dataset_name
    );

    let (cost_tol, param_tol, max_iter) = match dataset_name {
        "grid3D" => {
            info!("Note: grid3D requires very relaxed tolerances due to high complexity");
            (1e-1, 1e-1, 30)
        }
        "rim" => (1e-3, 1e-3, args.max_iterations),
        "torus3D" => (1e-5, 1e-5, args.max_iterations),
        _ => (
            args.cost_tolerance,
            args.parameter_tolerance,
            args.max_iterations,
        ),
    };

    let dataset_path = format!("data/odometry/{}.g2o", dataset_name);
    let mut graph = G2oLoader::load(&dataset_path)?;

    let num_vertices = graph.vertices_se3.len();
    let num_edges = graph.edges_se3.len();

    info!(
        "Graph Statistics: Vertices: {}, Edges: {}",
        num_vertices, num_edges
    );

    // Compute initial chi2 metrics from graph
    let initial_metrics = compute_se3_cost_metrics(&graph);

    if num_vertices == 0 {
        return Err(apex_solver::io::IoError::UnsupportedFormat(format!(
            "No SE3 vertices found in dataset {}",
            dataset_name
        ))
        .log()
        .into());
    }

    let mut problem = Problem::new();
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

    let optimizer_type = args.optimizer.to_lowercase();
    let needs_prior = optimizer_type == "gn"
        || optimizer_type == "gauss-newton"
        || optimizer_type == "dl"
        || optimizer_type == "dogleg"
        || optimizer_type == "dog-leg";

    if needs_prior
        && let Some(&first_id) = vertex_ids.first()
        && let Some(first_vertex) = graph.vertices_se3.get(&first_id)
    {
        let var_name = format!("x{}", first_id);
        let quat = first_vertex.pose.rotation_quaternion();
        let trans = first_vertex.pose.translation();
        let prior_value = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];

        let prior_factor = PriorFactor {
            data: prior_value.clone(),
        };
        let huber_loss = HuberLoss::new(1.0)?;
        problem.add_residual_block(
            &[&var_name],
            Box::new(prior_factor),
            Some(Box::new(huber_loss)),
        );
    } else if optimizer_type == "lm" || optimizer_type == "levenberg-marquardt" {
        let first_var_name = format!("x{}", vertex_ids[0]);
        problem.fix_variable(&first_var_name, 0);
        problem.fix_variable(&first_var_name, 1);
        problem.fix_variable(&first_var_name, 2);
        problem.fix_variable(&first_var_name, 3);
        problem.fix_variable(&first_var_name, 4);
        problem.fix_variable(&first_var_name, 5);
    }

    let loss_fn = create_loss_function(&args.loss_function, args.loss_scale).map_err(|e| {
        apex_solver::error::ApexSolverError::from(
            apex_solver::core::CoreError::InvalidInput(e.to_string()).log(),
        )
    })?;

    for edge in &graph.edges_se3 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);

        let relative_pose = edge.measurement.clone();
        let between_factor = BetweenFactor::new(relative_pose);

        let edge_loss = if loss_fn.is_some() {
            create_loss_function(&args.loss_function, args.loss_scale).map_err(|e| {
                apex_solver::error::ApexSolverError::from(
                    apex_solver::core::CoreError::InvalidInput(e.to_string()).log(),
                )
            })?
        } else {
            None
        };

        problem.add_residual_block(&[&id0, &id1], Box::new(between_factor), edge_loss);
    }

    info!(
        "Problem Structure: Variables: {}, Prior factors: {}, Between factors: {}",
        initial_values.len(),
        if needs_prior { "1" } else { "0" },
        graph.edges_se3.len()
    );

    let init_cost_start = Instant::now();
    let variables = problem.initialize_variables(&initial_values);
    let mut variable_name_to_col_idx_dict = HashMap::new();
    let mut col_offset = 0;
    let mut sorted_vars: Vec<_> = variables.keys().cloned().collect();
    sorted_vars.sort();
    for var_name in &sorted_vars {
        variable_name_to_col_idx_dict.insert(var_name.clone(), col_offset);
        col_offset += variables[var_name].get_size();
    }

    let symbolic_structure = problem
        .build_symbolic_structure(&variables, &variable_name_to_col_idx_dict, col_offset)
        .map_err(|e| {
            apex_solver::core::CoreError::SymbolicStructure(format!(
                "Failed to build symbolic structure for dataset {}",
                dataset_name
            ))
            .log_with_source(e)
        })?;

    let (residual, _jacobian) = problem
        .compute_residual_and_jacobian_sparse(
            &variables,
            &variable_name_to_col_idx_dict,
            &symbolic_structure,
        )
        .map_err(|e| {
            apex_solver::core::CoreError::FactorLinearization(format!(
                "Failed to compute residual and jacobian for dataset {}",
                dataset_name
            ))
            .log_with_source(e)
        })?;

    let initial_cost = residual.as_ref().squared_norm_l2();
    let init_cost_time = init_cost_start.elapsed();
    info!(
        "Initial cost computation: {:.2}ms",
        init_cost_time.as_secs_f64() * 1000.0
    );

    let optimizer_name = match args.optimizer.to_lowercase().as_str() {
        "gn" => "GN",
        "lm" => "LM",
        "dl" => "DL",
        _ => {
            warn!(
                "Invalid optimizer '{}'. Using LM (Levenberg-Marquardt) as default.",
                args.optimizer
            );
            "LM"
        }
    };

    let opt_start = Instant::now();
    let result = match optimizer_name {
        "GN" => {
            let config = GaussNewtonConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-10);

            let mut solver = GaussNewton::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
        "DL" => {
            let config = DogLegConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-10);

            let mut solver = DogLeg::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
        _ => {
            let config = LevenbergMarquardtConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-10);

            let mut solver = LevenbergMarquardt::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
    };

    let optimization_time = opt_start.elapsed();

    // Update graph vertices with optimized values for chi2 computation
    for (var_name, var_enum) in &result.parameters {
        if let Some(id_str) = var_name.strip_prefix("x")
            && let Ok(id) = id_str.parse::<usize>()
            && let Some(vertex) = graph.vertices_se3.get_mut(&id)
        {
            use nalgebra::{Quaternion, Vector3};
            let val = var_enum.to_vector();
            let translation = Vector3::new(val[0], val[1], val[2]);
            let rotation = Quaternion::new(val[3], val[4], val[5], val[6]);
            vertex.pose = SE3::from_translation_quaternion(translation, rotation);
        }
    }

    // Compute final chi2 metrics from updated graph
    let final_metrics = compute_se3_cost_metrics(&graph);

    let final_cost = result.final_cost;
    let improvement = (initial_cost - final_cost) / initial_cost;
    let chi2_improvement = if initial_metrics.chi2_cost > 0.0 {
        (initial_metrics.chi2_cost - final_metrics.chi2_cost) / initial_metrics.chi2_cost
    } else {
        0.0
    };
    let iterations = result.iterations;

    info!("Optimization completed:");
    info!("  Status: {:?}", result.status);
    info!("  Iterations: {}", iterations);
    info!(
        "  Initial chi2: {:.6e}, Final chi2: {:.6e}, Chi2 reduction: {:.2}%",
        initial_metrics.chi2_cost,
        final_metrics.chi2_cost,
        chi2_improvement * 100.0
    );
    info!(
        "  Initial cost: {:.6e}, Final cost: {:.6e}, Cost reduction: {:.2}%",
        initial_cost,
        final_cost,
        improvement * 100.0
    );
    info!(
        "  Optimization time: {:.2}ms",
        optimization_time.as_secs_f64() * 1000.0
    );

    // Save optimized graph if requested
    if let Some(output_base) = &args.save_output {
        info!("Saving optimized graph...");

        // Determine output path - if it's a directory, auto-generate filename
        let output_path = if output_base.is_dir() || output_base.to_string_lossy().ends_with('/') {
            output_base.join(format!("{}_optimized.g2o", dataset_name))
        } else {
            output_base.clone()
        };

        // Create output directory if it doesn't exist
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                apex_solver::io::IoError::log_with_source(
                    apex_solver::io::IoError::FileCreationFailed {
                        path: parent.display().to_string(),
                        reason: "Failed to create directory".to_string(),
                    },
                    e,
                )
            })?;
        }

        // Reconstruct graph from optimized variables
        use apex_solver::io::Graph;
        let optimized_graph = Graph::from_optimized_variables(&result.parameters, &graph);

        // Write to file (default: G2O format)
        use apex_solver::io::GraphLoader;
        G2oLoader::write(&optimized_graph, &output_path)?;

        info!("Saved optimized graph to: {}", output_path.display());
    }

    let status = match result.status {
        OptimizationStatus::Converged
        | OptimizationStatus::CostToleranceReached
        | OptimizationStatus::GradientToleranceReached
        | OptimizationStatus::ParameterToleranceReached => "CONVERGED".to_string(),
        OptimizationStatus::MaxIterationsReached => "ITER_LIMIT".to_string(),
        OptimizationStatus::NumericalFailure => "NUM_FAILURE".to_string(),
        _ => "UNKNOWN".to_string(),
    };

    Ok(DatasetResult {
        dataset: dataset_name.to_string(),
        manifold: "SE3".to_string(),
        optimizer: optimizer_name.to_string(),
        vertices: num_vertices,
        edges: num_edges,
        initial_chi2: initial_metrics.chi2_cost,
        final_chi2: final_metrics.chi2_cost,
        chi2_improvement,
        initial_cost,
        final_cost,
        improvement,
        iterations,
        time_ms: optimization_time.as_millis(),
        status,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg_attr(not(feature = "visualization"), allow(unused_mut))]
    let mut args = Args::parse();

    init_logger();

    info!("APEX-SOLVER POSE GRAPH OPTIMIZATION (2D + 3D)");
    info!("");

    let se2_datasets = vec!["M3500", "intel", "mit", "ring"];
    let se3_datasets = vec!["sphere2500", "parking-garage", "torus3D", "cubicle"];

    let (se2_datasets_to_run, se3_datasets_to_run) = if args.dataset == "all" {
        (se2_datasets, se3_datasets)
    } else {
        let mut se2_run = Vec::new();
        let mut se3_run = Vec::new();
        if se2_datasets.contains(&args.dataset.as_str()) {
            se2_run.push(args.dataset.as_str());
        }
        if se3_datasets.contains(&args.dataset.as_str()) {
            se3_run.push(args.dataset.as_str());
        }
        if se2_run.is_empty() && se3_run.is_empty() {
            warn!("Unknown dataset: {}", args.dataset);
            warn!("Using default: running all datasets");
            (se2_datasets, se3_datasets)
        } else {
            (se2_run, se3_run)
        }
    };

    #[cfg(feature = "visualization")]
    if args.with_visualizer && (se2_datasets_to_run.len() > 1 || se3_datasets_to_run.len() > 1) {
        warn!("Visualization is not supported when running multiple datasets (--dataset all).");
        warn!("Disabling visualization. To use visualization, specify a single dataset.");
        warn!("Example: --dataset M3500 --with-visualizer");
        args.with_visualizer = false;
    }

    let mut all_results = Vec::new();

    if !se2_datasets_to_run.is_empty() {
        info!("2D POSE GRAPH OPTIMIZATION (SE2)");
        info!("");

        for dataset in &se2_datasets_to_run {
            match test_se2_dataset(dataset, &args) {
                Ok(result) => {
                    info!("Dataset {} completed: {}", dataset, result.status);
                    all_results.push(result);
                }
                Err(e) => {
                    error!("Dataset {} failed", dataset);
                    error!("Error: {}", e);
                    error!("Full error chain:\n{}", e.chain());
                }
            }
            info!("");
        }
    }

    if !se3_datasets_to_run.is_empty() {
        info!("3D POSE GRAPH OPTIMIZATION (SE3)");
        info!("");

        for dataset in &se3_datasets_to_run {
            match test_se3_dataset(dataset, &args) {
                Ok(result) => {
                    info!("Dataset {} completed: {}", dataset, result.status);
                    all_results.push(result);
                }
                Err(e) => {
                    error!("Dataset {} failed", dataset);
                    error!("Error: {}", e);
                    error!("Full error chain:\n{}", e.chain());
                }
            }
            info!("");
        }
    }

    if all_results.len() > 1 {
        format_summary_table(&all_results);
    }

    let converged_count = all_results
        .iter()
        .filter(|r| r.status == "CONVERGED")
        .count();
    if converged_count == all_results.len() {
        info!("All datasets converged successfully");
        Ok(())
    } else if converged_count == 0 {
        Err("No datasets converged".into())
    } else {
        info!(
            "{}/{} datasets converged",
            converged_count,
            all_results.len()
        );
        Err("Some datasets failed to converge".into())
    }
}
