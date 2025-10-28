use std::collections::HashMap;
use std::time::Instant;

use apex_solver::core::factors::{BetweenFactorSE3, PriorFactor};
use apex_solver::core::loss_functions::*;
use apex_solver::core::problem::Problem;
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::dog_leg::DogLegConfig;
use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;
use apex_solver::optimizer::{DogLeg, GaussNewton, LevenbergMarquardt};
use clap::Parser;
use nalgebra::dvector;

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

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

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
    #[arg(long)]
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
    println!("\n{}", "=".repeat(160));
    println!("=== FINAL SUMMARY TABLE ===\n");

    // Header
    println!(
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
    println!("{}", "-".repeat(160));

    // Rows
    for result in results {
        println!(
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

    println!("{}", "-".repeat(160));

    // Summary statistics
    let converged_count = results.iter().filter(|r| r.status == "CONVERGED").count();
    let total_count = results.len();
    println!(
        "\nSummary: {}/{} datasets converged successfully",
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
        println!("Average time for converged datasets: {:.1}ms", avg_time);
        println!(
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
    println!(
        "\n=== TESTING {} SE3 DATASET ===",
        dataset_name.to_uppercase()
    );
    println!("Loading {}.g2o dataset for SE3 optimization", dataset_name);

    // Apply dataset-specific optimizations
    let (cost_tol, param_tol, max_iter) = match dataset_name {
        "grid3D" => {
            println!("Note: grid3D requires very relaxed tolerances due to high complexity");
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

    if cost_tol != args.cost_tolerance
        || param_tol != args.parameter_tolerance
        || max_iter != args.max_iterations
    {
        println!(
            "Using optimized parameters: cost_tol={:.1e}, param_tol={:.1e}, max_iter={}",
            cost_tol, param_tol, max_iter
        );
    }

    // Load the G2O graph file
    let dataset_path = format!("data/{}.g2o", dataset_name);
    let graph = G2oLoader::load(&dataset_path)?;

    let num_vertices = graph.vertices_se3.len();
    let num_edges = graph.edges_se3.len();

    println!("Successfully loaded SE3 graph:");
    println!("  SE3 vertices: {}", num_vertices);
    println!("  SE3 edges: {}", num_edges);

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

        println!(
            "Added soft prior factor (HuberLoss) on vertex {} to remove gauge freedom for {}",
            first_id,
            optimizer_type.to_uppercase()
        );
        println!("  Prior value: {:?}", prior_value.as_slice());
    } else {
        println!(
            "Skipping prior factor for {} optimizer (has built-in regularization)",
            optimizer_type.to_uppercase()
        );
    }

    // Create loss function for between factors
    let loss_fn = create_loss_function(&args.loss_function, args.loss_scale)?;
    let loss_name = args.loss_function.to_uppercase();
    let loss_scale = if loss_fn.is_some() {
        args.loss_scale
            .unwrap_or_else(|| match args.loss_function.to_lowercase().as_str() {
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
                _ => 1.0,
            })
    } else {
        1.0
    };

    println!(
        "Using loss function: {} (scale: {:.4})",
        loss_name, loss_scale
    );

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

    println!("\nProblem setup completed:");
    println!("  Variables: {}", initial_values.len());
    println!(
        "  Prior factors: {} (vertex {})",
        if needs_prior { "1" } else { "0" },
        vertex_ids.first().unwrap_or(&0)
    );
    println!("  Between factors: {}", graph.edges_se3.len());

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

    println!("\nInitial state:");
    println!("  Initial cost: {:.6e}", initial_cost);

    // Determine optimizer type and run optimization
    let optimizer_name = match args.optimizer.to_lowercase().as_str() {
        "gn" => "GN",
        "lm" => "LM",
        "dl" => "DL",
        _ => {
            eprintln!(
                "Invalid optimizer '{}'. Using LM (Levenberg-Marquardt) as default.",
                args.optimizer
            );
            "LM"
        }
    };

    println!(
        "\n=== STARTING {} OPTIMIZATION ===",
        match optimizer_name {
            "LM" => "LEVENBERG-MARQUARDT",
            "GN" => "GAUSS-NEWTON",
            "DL" => "DOG LEG",
            _ => "LEVENBERG-MARQUARDT",
        }
    );
    println!("Configuration:");
    println!("  Max iterations: {}", max_iter);
    println!("  Cost tolerance: {:.2e}", cost_tol);
    println!("  Parameter tolerance: {:.2e}", param_tol);

    let start_time = Instant::now();
    let result = match optimizer_name {
        "GN" => {
            let config = GaussNewtonConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-12)
                .with_verbose(args.verbose)
                .with_visualization(args.with_visualizer);
            let mut solver = GaussNewton::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
        "DL" => {
            let config = DogLegConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-12)
                .with_verbose(args.verbose)
                .with_visualization(args.with_visualizer);
            let mut solver = DogLeg::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
        _ => {
            let config = LevenbergMarquardtConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-12)
                .with_verbose(args.verbose)
                .with_visualization(args.with_visualizer);
            let mut solver = LevenbergMarquardt::with_config(config);
            solver.optimize(&problem, &initial_values)?
        }
    };
    let duration = start_time.elapsed();

    println!("\n=== OPTIMIZATION RESULTS ===");

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

    println!("Status: {}", status);
    println!("Convergence reason: {}", convergence_reason);
    println!("Initial cost: {:.6e}", initial_cost);
    println!("Final cost: {:.6e}", result.final_cost);
    println!(
        "Cost reduction: {:.6e} ({:.2}%)",
        initial_cost - result.final_cost,
        improvement
    );
    println!("Iterations: {}", result.iterations);
    println!("Execution time: {:.1}ms", duration.as_millis());

    // Print first vertex (x0) to verify if it moved during optimization
    if let Some(&first_id) = vertex_ids.first() {
        let var_name = format!("x{}", first_id);
        if let Some(final_var) = result.parameters.get(&var_name) {
            let final_vector = final_var.to_vector();
            println!("\nFirst vertex (x0) after optimization:");
            println!("  Final value: {:?}", final_vector.as_slice());
            if let Some((_, initial_vec)) = initial_values.get(&var_name) {
                let diff: f64 = (final_vector - initial_vec).norm();
                println!("  Change from initial: {:.6e}", diff);
            }
        }
    }

    // Save optimized graph if requested
    if let Some(output_base) = &args.save_output {
        println!("\nSaving optimized graph...");

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

        println!("✓ Saved optimized graph to: {}", output_path.display());
        println!("  Status: {} ({})", status, convergence_reason);
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
    let mut args = Args::parse();

    println!("=== APEX-SOLVER 3D POSE GRAPH OPTIMIZATION ===");

    // Define available SE3 datasets
    let se3_datasets = vec![
        "rim",
        "sphere2500",
        "parking-garage",
        "torus3D",
        "grid3D",
        "cubicle",
    ];

    let datasets = if args.dataset == "all" {
        se3_datasets
    } else {
        vec![args.dataset.as_str()]
    };

    // Check if visualization is requested with multiple datasets
    if args.with_visualizer && datasets.len() > 1 {
        eprintln!(
            "\n[WARNING] Visualization is not supported when running multiple datasets (--dataset all)."
        );
        eprintln!(
            "[WARNING] Disabling visualization. To use visualization, specify a single dataset."
        );
        eprintln!("[WARNING] Example: --dataset sphere2500 --with-visualizer\n");
        args.with_visualizer = false;
    }

    let mut results = Vec::new();

    for dataset in &datasets {
        println!("\n{}", "=".repeat(80));
        match test_se3_dataset(dataset, &args) {
            Ok(result) => {
                println!("Dataset {} completed: {}", dataset, result.status);
                results.push(result);
            }
            Err(e) => {
                eprintln!("Dataset {} failed: {}", dataset, e);
            }
        }
    }

    // Display summary table
    if results.len() > 1 {
        format_summary_table(&results);
    }

    let converged_count = results.iter().filter(|r| r.status == "CONVERGED").count();
    if converged_count == results.len() {
        println!("\nAll datasets converged successfully!");
        Ok(())
    } else if converged_count == 0 {
        Err("No datasets converged".into())
    } else {
        println!("\n{}/{} datasets converged", converged_count, results.len());
        Err("Some datasets failed to converge".into())
    }
}
