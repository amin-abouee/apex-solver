use std::collections::HashMap;
use std::time::Instant;

use apex_solver::core::loss_functions::*;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BetweenFactorSE3, PriorFactor};
use apex_solver::init_logger;
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::dog_leg::DogLegConfig;
use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
use apex_solver::optimizer::{DogLeg, GaussNewton, LevenbergMarquardt};
use clap::Parser;
use nalgebra::dvector;
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "optimize_3d_graph")]
#[command(about = "Optimize 3D pose graphs from G2O datasets")]
struct Args {
    /// G2O dataset file to load (without .g2o extension). Use "all" to test all SE3 datasets
    #[arg(short, long, default_value = "sphere2500")]
    dataset: String,

    /// Maximum number of optimization iterations (optimized for SE3 datasets)
    #[arg(short, long, default_value = "100")]
    max_iterations: usize,

    /// Cost tolerance for convergence (optimized for SE3 datasets)
    #[arg(long, default_value = "1e-4")]
    cost_tolerance: f64,

    /// Parameter tolerance for convergence (optimized for SE3 datasets)
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
}

#[derive(Clone)]
struct DatasetResult {
    dataset: String,
    optimizer: String,
    vertices: usize,
    edges: usize,
    initial_cost: f64,
    final_cost: f64,
    improvement: f64,
    iterations: usize,
    time_ms: u128,
    convergence_reason: String,
    status: String,
}

fn format_summary_table(results: &[DatasetResult]) {
    info!("Final summary table:");

    // Header
    info!(
        "{:<16} | {:<10} | {:<8} | {:<6} | {:<12} | {:<12} | {:<11} | {:<5} | {:<9} | {:<20} | {:<12}",
        "Dataset",
        "Optimizer",
        "Vertices",
        "Edges",
        "Init Cost",
        "Final Cost",
        "Improvement",
        "Iters",
        "Time(ms)",
        "Convergence",
        "Status"
    );
    info!("{}", "-".repeat(160));

    // Rows
    for result in results {
        info!(
            "{:<16} | {:<10} | {:<8} | {:<6} | {:<12.6e} | {:<12.6e} | {:>10.2}% | {:<5} | {:<9} | {:<20} | {:<12}",
            result.dataset,
            result.optimizer,
            result.vertices,
            result.edges,
            result.initial_cost,
            result.final_cost,
            result.improvement,
            result.iterations,
            result.time_ms,
            result.convergence_reason,
            result.status
        );
    }

    info!("{}", "-".repeat(160));

    // Summary statistics
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

/// Create a loss function based on the user's choice
fn create_loss_function(
    loss_name: &str,
    scale: Option<f64>,
) -> Result<Option<Box<dyn LossFunction + Send>>, Box<dyn std::error::Error>> {
    let loss_lower = loss_name.to_lowercase();

    // Determine default scale if not provided
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

fn test_se3_dataset(
    dataset_name: &str,
    args: &Args,
) -> Result<DatasetResult, Box<dyn std::error::Error>> {
    info!(
        "Testing {} SE3 dataset by loading {}.g2o for optimization",
        dataset_name.to_uppercase(),
        dataset_name
    );

    // Apply dataset-specific optimizations
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

    // Load the G2O graph file
    let dataset_path = format!("data/{}.g2o", dataset_name);
    let graph = G2oLoader::load(&dataset_path)?;

    let num_vertices = graph.vertices_se3.len();
    let num_edges = graph.edges_se3.len();

    info!(
        "Graph Statistics: Vertices: {}, Edges: {}",
        num_vertices, num_edges
    );

    if num_vertices == 0 {
        return Err(format!("No SE3 vertices found in dataset {}", dataset_name).into());
    }

    // Create optimization problem
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add SE3 vertices as variables
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

    // Add prior factor for GN and Dog Leg optimizers
    // - GN: Needs explicit prior to make Hessian full-rank (no built-in damping)
    // - Dog Leg: Requires valid GN step computation, which needs regularization
    // - LM: Has built-in λI damping, can use fix_variable() instead
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
        // Use HuberLoss with scale=1.0 (same as tiny-solver-rs)
        // This allows the first vertex to move slightly if graph structure demands it
        let huber_loss = HuberLoss::new(1.0).expect("Failed to create HuberLoss");
        problem.add_residual_block(
            &[&var_name],
            Box::new(prior_factor),
            Some(Box::new(huber_loss)),
        );
    }

    // Create loss function for between factors
    let loss_fn = create_loss_function(&args.loss_function, args.loss_scale)?;

    // Add SE3 between factors
    for edge in &graph.edges_se3 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);
        let relative_pose = edge.measurement.clone();
        let between_factor = BetweenFactorSE3::new(relative_pose);

        // Clone the loss function for each edge
        let edge_loss = if loss_fn.is_some() {
            Some(create_loss_function(&args.loss_function, args.loss_scale)?.unwrap())
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

    let symbolic_structure = problem
        .build_symbolic_structure(&variables, &variable_name_to_col_idx_dict, col_offset)
        .expect("Failed to build symbolic structure");

    let (residual, _jacobian) = problem
        .compute_residual_and_jacobian_sparse(
            &variables,
            &variable_name_to_col_idx_dict,
            &symbolic_structure,
        )
        .expect("Failed to compute residual and jacobian");

    // Compute initial cost using faer's norm
    let initial_cost = residual.as_ref().squared_norm_l2();

    // Determine optimizer type and run optimization
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

    let start_time = Instant::now();
    let result = match optimizer_name {
        "GN" => {
            let config = GaussNewtonConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-12);
            #[cfg(feature = "visualization")]
            let config = config.with_visualization(args.with_visualizer);
            let mut solver = GaussNewton::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
        "DL" => {
            let config = DogLegConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-12);
            #[cfg(feature = "visualization")]
            let config = config.with_visualization(args.with_visualizer);
            let mut solver = DogLeg::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
        _ => {
            let config = LevenbergMarquardtConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-12);
            #[cfg(feature = "visualization")]
            let config = config.with_visualization(args.with_visualizer);
            let mut solver = LevenbergMarquardt::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
    };
    let duration = start_time.elapsed();

    // Determine convergence status accurately
    let (status, convergence_reason) = match &result.status {
        apex_solver::optimizer::OptimizationStatus::Converged => {
            ("CONVERGED", "Converged".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::CostToleranceReached => {
            ("CONVERGED", "CostTolerance".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::ParameterToleranceReached => {
            ("CONVERGED", "ParameterTolerance".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::GradientToleranceReached => {
            ("CONVERGED", "GradientTolerance".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::TrustRegionRadiusTooSmall => {
            ("CONVERGED", "TrustRegionRadiusTooSmall".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::MinCostThresholdReached => {
            ("CONVERGED", "MinCostThresholdReached".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::MaxIterationsReached => {
            ("NOT_CONVERGED", "MaxIterations".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::Timeout => {
            ("NOT_CONVERGED", "Timeout".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::NumericalFailure => {
            ("NOT_CONVERGED", "NumericalFailure".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::IllConditionedJacobian => {
            ("NOT_CONVERGED", "IllConditionedJacobian".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::InvalidNumericalValues => {
            ("NOT_CONVERGED", "InvalidNumericalValues".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::UserTerminated => {
            ("NOT_CONVERGED", "UserTerminated".to_string())
        }
        apex_solver::optimizer::OptimizationStatus::Failed(msg) => {
            ("NOT_CONVERGED", format!("Failed:{}", msg))
        }
    };

    let improvement = ((initial_cost - result.final_cost) / initial_cost) * 100.0;

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
            std::fs::create_dir_all(parent)?;
        }

        // Reconstruct graph from optimized variables
        use apex_solver::io::Graph;
        let optimized_graph = Graph::from_optimized_variables(&result.parameters, &graph);

        // Write to file (default: G2O format)
        use apex_solver::io::GraphLoader;
        G2oLoader::write(&optimized_graph, &output_path)?;
    }

    Ok(DatasetResult {
        dataset: dataset_name.to_string(),
        optimizer: optimizer_name.to_string(),
        vertices: num_vertices,
        edges: num_edges,
        initial_cost,
        final_cost: result.final_cost,
        improvement,
        iterations: result.iterations,
        time_ms: duration.as_millis(),
        convergence_reason: convergence_reason.to_string(),
        status: status.to_string(),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg_attr(not(feature = "visualization"), allow(unused_mut))]
    let mut args = Args::parse();

    // Initialize logger with INFO level
    init_logger();

    info!("APEX-SOLVER 3D POSE GRAPH OPTIMIZATION\n");

    // Define available SE3 datasets
    let se3_datasets = vec!["sphere2500", "parking-garage", "torus3D"];

    let datasets = if args.dataset == "all" {
        se3_datasets
    } else {
        vec![args.dataset.as_str()]
    };

    // Check if visualization is requested with multiple datasets
    #[cfg(feature = "visualization")]
    if args.with_visualizer && datasets.len() > 1 {
        warn!("Visualization is not supported when running multiple datasets (--dataset all).");
        warn!("Disabling visualization. To use visualization, specify a single dataset.");
        warn!("Example: --dataset sphere2500 --with-visualizer");
        args.with_visualizer = false;
    }

    let mut results = Vec::new();

    for dataset in &datasets {
        match test_se3_dataset(dataset, &args) {
            Ok(result) => {
                info!("Dataset {} completed: {}", dataset, result.status);
                results.push(result);
            }
            Err(e) => {
                warn!("Dataset {} failed: {}", dataset, e);
            }
        }
    }

    // Display summary table
    if results.len() > 1 {
        format_summary_table(&results);
    }

    let converged_count = results.iter().filter(|r| r.status == "CONVERGED").count();
    if converged_count == results.len() {
        info!("All datasets converged successfully");
        Ok(())
    } else if converged_count == 0 {
        Err("No datasets converged".into())
    } else {
        info!("{}/{} datasets converged", converged_count, results.len());
        Err("Some datasets failed to converge".into())
    }
}
