use std::collections::HashMap;
use std::time::Instant;

use apex_solver::ErrorLogging;
use apex_solver::JacobianMode;
use apex_solver::NoiseModel;
use apex_solver::apex_io::{
    G2oLoader, Graph, GraphLoader, ODOMETRY_DATA_DIR_2D, ODOMETRY_DATA_DIR_3D,
};
use apex_solver::apex_manifolds::se2::SE2;
use apex_solver::apex_manifolds::se3::SE3;
use apex_solver::apex_manifolds::{LieGroup, ManifoldType, Tangent};
use apex_solver::core::VarKey;
use apex_solver::core::loss_functions::*;
use apex_solver::core::noise::{InformationRepair, RepairStrategy, RepairSummary};
use apex_solver::core::problem::Problem;
use apex_solver::factors::pose::{BetweenFactor, PriorFactor};
use apex_solver::init_logger;
use apex_solver::optimizer::dog_leg::DogLegConfig;
use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
use apex_solver::optimizer::levenberg_marquardt::{DampingUpdate, LevenbergMarquardtConfig};
use apex_solver::optimizer::{
    DogLeg, GaussNewton, LevenbergMarquardt, OptimizationStatus, initialize_optimization_state,
};
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

    /// Robust loss function to use: "l2", "l1", "huber", "cauchy", "fair", "welsch", "tukey", "geman", "dcs", "andrews", "ramsay", "trimmed", "lp", "barron0", "barron1", "barron-2", "t-distribution", "adaptive-barron"
    ///
    /// Keep this list in sync with `loss_canonical_names()` in
    /// `core::loss_functions` (clap attributes cannot call it, so sync is by
    /// convention — adding a kernel means updating both places).
    #[arg(long, default_value = "l2")]
    loss_function: String,

    /// Scale parameter for the loss function (default: 1.345 for Huber)
    #[arg(long)]
    loss_scale: Option<f64>,

    /// Enable detailed profiling output with timing breakdown
    #[arg(long)]
    profile: bool,

    /// Disable the built-in per-dataset tolerance/iteration overrides
    /// (manhattanOlson3500, grid3D — which caps iterations at 30 —, rim,
    /// torus3D) and use the CLI values verbatim for every dataset.
    #[arg(long)]
    no_dataset_overrides: bool,

    /// Ignore the per-edge information matrices parsed from the G2O file and
    /// optimize the unweighted objective (legacy behaviour). By default the
    /// Ω matrices whiten every edge so the optimized objective equals the
    /// reported χ².
    #[arg(long)]
    no_noise: bool,

    /// Initial LM damping λ (config default: 1e-4; GTSAM's odometry default is 1e-5)
    #[arg(long)]
    damping: Option<f64>,

    /// Nielsen ν — the rejected-step multiplier (config default: 2.0)
    #[arg(long)]
    damping_nu: Option<f64>,

    /// Use Marquardt's three-band rule instead of Nielsen's cubic rule, with
    /// these (increase, decrease) factors — e.g. `--damping-marquardt 10,0.3`
    #[arg(long, num_args = 2, value_delimiter = ',')]
    damping_marquardt: Option<Vec<f64>>,

    /// Bounds for the Marquardt diagonal D_jj. `(1,1)` gives scalar λ·I damping
    /// (GTSAM's default); omit for the config default (λ·diag(H) with clamping)
    #[arg(long, num_args = 2, value_delimiter = ',')]
    diagonal_bounds: Option<Vec<f64>>,

    /// Upper bound for λ before the solve is declared a damping failure
    /// (config default: 1e12; GTSAM uses 1e5)
    #[arg(long)]
    damping_max: Option<f64>,

    /// Repair strategy for indefinite per-edge information matrices:
    /// "clamp" (PSD projection — default) or "unit-weight" (identity weighting
    /// for affected edges; trades χ² for unweighted cost)
    #[arg(long, default_value = "clamp")]
    indefinite_repair: String,

    /// Print per-iteration cost/damping/step-quality lines (debug-level solver log)
    #[arg(long)]
    verbose_iters: bool,
}

// ============================================================================
// LM TUNING + NOISE REPAIR
// ============================================================================

/// Apply the CLI's LM damping/termination tuning to a base config.
///
/// Every knob maps onto an existing `LevenbergMarquardtConfig` field — this is
/// parameter tuning only, no algorithm change.
fn apply_lm_tuning(config: LevenbergMarquardtConfig, args: &Args) -> LevenbergMarquardtConfig {
    let mut config = config;
    if let Some(d) = args.damping {
        config = config.with_damping(d);
    }
    if let Some(nu) = args.damping_nu {
        config.damping_nu = nu;
    }
    if let Some(max) = args.damping_max {
        let min = config.damping_min;
        config = config.with_damping_bounds(min, max);
    }
    if let Some(bounds) = &args.diagonal_bounds {
        config = config.with_diagonal_bounds(bounds[0], bounds[1]);
    }
    if let Some(factors) = &args.damping_marquardt {
        config = config
            .with_damping_update(DampingUpdate::Marquardt)
            .with_damping_factors(factors[0], factors[1]);
    }
    config
}

/// Build the per-edge noise model honouring `--indefinite-repair`.
///
/// Returns the model and what the repair did, so the caller can aggregate one
/// summary line per dataset instead of one `warn!` per edge. "clamp" (default)
/// keeps the PSD-projection behaviour of `from_information`; "unit-weight"
/// replaces any edge whose Ω needed a material repair with identity weight,
/// trading χ² for unweighted cost.
fn edge_noise_model(
    info: nalgebra::DMatrix<f64>,
    strategy: RepairStrategy,
) -> Result<(NoiseModel, InformationRepair), apex_solver::error::ApexSolverError> {
    strategy.build(info).map_err(|e| {
        apex_solver::error::ApexSolverError::from(
            apex_solver::core::CoreError::InvalidInput(e.to_string()).log(),
        )
    })
}

/// Parse `--indefinite-repair` once, before the edge loop.
fn repair_strategy(args: &Args) -> Result<RepairStrategy, apex_solver::error::ApexSolverError> {
    RepairStrategy::from_name(&args.indefinite_repair).map_err(|e| {
        apex_solver::error::ApexSolverError::from(
            apex_solver::core::CoreError::InvalidInput(e.to_string()).log(),
        )
    })
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
    /// Unweighted cost: sum of ||r||^2 (no information matrix)
    unweighted_cost: f64,
}

/// Compute SE2 cost metrics from G2O graph data
/// - Chi-squared: sum of r^T * Omega * r (information-weighted)
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

            let residual_tangent = LieGroup::log(&error, None);
            let residual_vec = nalgebra::DVector::from_column_slice(residual_tangent.as_slice());

            // Chi-squared: r^T * Omega * r (information-weighted)
            let weighted_sq = &residual_vec.transpose() * edge.information * &residual_vec;
            chi2_cost += weighted_sq[(0, 0)];
            unweighted_cost += residual_vec.norm_squared();
        }
    }

    CostMetrics {
        chi2_cost,
        unweighted_cost,
    }
}

/// Compute SE3 cost metrics from G2O graph data
/// Returns chi-squared: sum of r^T * Omega * r (information-weighted)
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

            let residual_tangent = LieGroup::log(&error, None);
            let residual_vec = nalgebra::DVector::from_column_slice(residual_tangent.as_slice());

            // Chi-squared: r^T * Omega * r (information-weighted)
            let weighted_sq = &residual_vec.transpose() * edge.information * &residual_vec;
            chi2_cost += weighted_sq[(0, 0)];
            unweighted_cost += residual_vec.norm_squared();
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
) -> Result<Option<Box<dyn LossFunction + Send + Sync>>, Box<dyn std::error::Error>> {
    // Single table lives in the library (`loss_from_name`); this wrapper only
    // adapts the error type so help text and accepted names cannot drift.
    Ok(Some(
        apex_solver::core::loss_functions::loss_from_name(loss_name, scale).map_err(|e| {
            Box::<dyn std::error::Error>::from(format!(
                "{e}. Valid options: {}",
                apex_solver::core::loss_functions::loss_canonical_names().join(", ")
            ))
        })?,
    ))
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

    // Apply dataset-specific optimizations (disable with --no-dataset-overrides)
    let (cost_tol, param_tol, max_iter) = if args.no_dataset_overrides {
        (
            args.cost_tolerance,
            args.parameter_tolerance,
            args.max_iterations,
        )
    } else {
        match dataset_name {
            "manhattanOlson3500" => (1e-3, 1e-3, args.max_iterations),
            _ => (
                args.cost_tolerance,
                args.parameter_tolerance,
                args.max_iterations,
            ),
        }
    };
    info!(
        "Effective SE2 settings: cost_tol={cost_tol:e} param_tol={param_tol:e} max_iter={max_iter}"
    );

    let load_start = Instant::now();
    let dataset_path = format!("{}/{}.g2o", ODOMETRY_DATA_DIR_2D, dataset_name);
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
        return Err(apex_io::IoError::UnsupportedFormat(format!(
            "No SE2 vertices found in dataset {}",
            dataset_name
        ))
        .log()
        .into());
    }

    let setup_start = Instant::now();
    let mut problem = Problem::new(JacobianMode::Sparse);
    // Map from vertex ID to VarKey (for building residual blocks and updating graph)
    let mut var_key_map: HashMap<usize, VarKey> = HashMap::new();

    let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se2.get(&id) {
            let se2_data = dvector![vertex.x(), vertex.y(), vertex.theta()];
            let key = problem.add_variable(ManifoldType::SE2, se2_data);
            var_key_map.insert(id, key);
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
        let prior_factor = PriorFactor::new(first_vertex.pose.clone());
        let huber_loss = HuberLoss::new(1.0)?;
        let first_key = var_key_map[&first_id];
        problem.add_residual_block(
            &[first_key],
            Box::new(prior_factor),
            Some(Box::new(huber_loss)),
        );
    } else if optimizer_type == "lm" || optimizer_type == "levenberg-marquardt" {
        let first_key = var_key_map[&vertex_ids[0]];
        problem.fix_variable(first_key, 0);
        problem.fix_variable(first_key, 1);
        problem.fix_variable(first_key, 2);
    }

    let loss_fn = create_loss_function(&args.loss_function, args.loss_scale).map_err(|e| {
        apex_solver::error::ApexSolverError::from(
            apex_solver::core::CoreError::InvalidInput(e.to_string()).log(),
        )
    })?;

    let strategy = repair_strategy(args)?;
    let mut repair_summary = RepairSummary::default();

    for edge in &graph.edges_se2 {
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
        let edge_noise = if args.no_noise {
            NoiseModel::null()
        } else {
            let (model, repair) = edge_noise_model(
                nalgebra::DMatrix::from_column_slice(3, 3, edge.information.as_slice()),
                strategy,
            )?;
            repair_summary.record(strategy, &repair);
            model
        };

        if let (Some(&k0), Some(&k1)) = (var_key_map.get(&edge.from), var_key_map.get(&edge.to)) {
            problem.add_residual_block_with_noise(
                &[k0, k1],
                Box::new(between_factor),
                edge_loss,
                edge_noise,
            );
        }
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
        var_key_map.len(),
        if needs_prior { "1" } else { "0" },
        graph.edges_se2.len()
    );
    if !args.no_noise && !repair_summary.is_clean() {
        info!(
            "Information repair ({} edges, {} materially repaired, {} unit-weighted): \
             clamped directions carry no information; unit-weighted edges optimize \
             unweighted cost, not χ²",
            repair_summary.edges, repair_summary.materially_repaired, repair_summary.unit_weighted
        );
    }

    let init_cost_start = Instant::now();
    let init_state = initialize_optimization_state(&mut problem).map_err(|e| {
        apex_solver::core::CoreError::SymbolicStructure(format!(
            "Failed to initialize optimization state for dataset {}",
            dataset_name
        ))
        .log_with_source(e)
    })?;
    let initial_cost = init_state.initial_cost;
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

    /// Helper macro to optionally attach a Rerun observer to a solver.
    macro_rules! attach_visualizer {
        ($solver:expr, $args:expr) => {
            #[cfg(feature = "visualization")]
            if $args.with_visualizer {
                use apex_solver::observers::RerunObserver;
                match RerunObserver::new(true) {
                    Ok(observer) => {
                        $solver.add_observer(observer);
                        info!("Rerun visualization enabled");
                    }
                    Err(e) => warn!("Failed to create Rerun observer: {}", e),
                }
            }
        };
    }

    let opt_start = Instant::now();
    let result = match optimizer_name {
        "GN" => {
            let config = GaussNewtonConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-10);

            let mut solver = GaussNewton::with_config(config);
            attach_visualizer!(solver, args);
            solver.optimize(&mut problem)?
        }
        "DL" => {
            let config = DogLegConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-10);

            let mut solver = DogLeg::with_config(config);
            attach_visualizer!(solver, args);
            solver.optimize(&mut problem)?
        }
        _ => {
            let config = apply_lm_tuning(
                LevenbergMarquardtConfig::new()
                    .with_max_iterations(max_iter)
                    .with_cost_tolerance(cost_tol)
                    .with_parameter_tolerance(param_tol)
                    .with_gradient_tolerance(1e-10),
                args,
            );

            let mut solver = LevenbergMarquardt::with_config(config);
            attach_visualizer!(solver, args);
            solver.optimize(&mut problem)?
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
    for (&id, &key) in &var_key_map {
        if let Some(vertex) = graph.vertices_se2.get_mut(&id) {
            let val = result.parameters[key].to_dvector();
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
        "  Unweighted: {:.6e} -> {:.6e}",
        initial_metrics.unweighted_cost, final_metrics.unweighted_cost
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
                apex_io::IoError::log_with_source(
                    apex_io::IoError::FileCreationFailed {
                        path: parent.display().to_string(),
                        reason: "Failed to create directory".to_string(),
                    },
                    e,
                )
            })?;
        }

        // Write updated graph to file (graph already updated in-place above)
        use apex_solver::apex_io::GraphLoader;
        G2oLoader::write(&graph, &output_path)?;

        info!("Saved optimized graph to: {}", output_path.display());
    }

    let status = match result.status {
        OptimizationStatus::Converged
        | OptimizationStatus::CostToleranceReached
        | OptimizationStatus::GradientToleranceReached
        | OptimizationStatus::StalledNoProgress
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

    let (cost_tol, param_tol, max_iter) = if args.no_dataset_overrides {
        (
            args.cost_tolerance,
            args.parameter_tolerance,
            args.max_iterations,
        )
    } else {
        match dataset_name {
            "grid3D" => {
                info!(
                    "Note: grid3D requires very relaxed tolerances due to high complexity \
                     (hard 30-iteration cap; disable with --no-dataset-overrides)"
                );
                (1e-1, 1e-1, 30)
            }
            "rim" => (1e-3, 1e-3, args.max_iterations),
            "torus3D" => (1e-5, 1e-5, args.max_iterations),
            _ => (
                args.cost_tolerance,
                args.parameter_tolerance,
                args.max_iterations,
            ),
        }
    };
    info!(
        "Effective SE3 settings: cost_tol={cost_tol:e} param_tol={param_tol:e} max_iter={max_iter}"
    );

    let dataset_path = format!("{}/{}.g2o", ODOMETRY_DATA_DIR_3D, dataset_name);
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
        return Err(apex_io::IoError::UnsupportedFormat(format!(
            "No SE3 vertices found in dataset {}",
            dataset_name
        ))
        .log()
        .into());
    }

    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut var_key_map: HashMap<usize, VarKey> = HashMap::new();

    let mut vertex_ids: Vec<_> = graph.vertices_se3.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se3.get(&id) {
            let quat = vertex.pose.rotation_quaternion();
            let trans = vertex.pose.translation();
            let se3_data = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];
            let key = problem.add_variable(ManifoldType::SE3, se3_data);
            var_key_map.insert(id, key);
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
        let prior_factor = PriorFactor::new(first_vertex.pose.clone());
        let huber_loss = HuberLoss::new(1.0)?;
        let first_key = var_key_map[&first_id];
        problem.add_residual_block(
            &[first_key],
            Box::new(prior_factor),
            Some(Box::new(huber_loss)),
        );
    } else if optimizer_type == "lm" || optimizer_type == "levenberg-marquardt" {
        let first_key = var_key_map[&vertex_ids[0]];
        for dof in 0..6 {
            problem.fix_variable(first_key, dof);
        }
    }

    let loss_fn = create_loss_function(&args.loss_function, args.loss_scale).map_err(|e| {
        apex_solver::error::ApexSolverError::from(
            apex_solver::core::CoreError::InvalidInput(e.to_string()).log(),
        )
    })?;

    let strategy = repair_strategy(args)?;
    let mut repair_summary = RepairSummary::default();

    for edge in &graph.edges_se3 {
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
        let edge_noise = if args.no_noise {
            NoiseModel::null()
        } else {
            let (model, repair) = edge_noise_model(
                nalgebra::DMatrix::from_column_slice(6, 6, edge.information.as_slice()),
                strategy,
            )?;
            repair_summary.record(strategy, &repair);
            model
        };

        if let (Some(&k0), Some(&k1)) = (var_key_map.get(&edge.from), var_key_map.get(&edge.to)) {
            problem.add_residual_block_with_noise(
                &[k0, k1],
                Box::new(between_factor),
                edge_loss,
                edge_noise,
            );
        }
    }

    info!(
        "Problem Structure: Variables: {}, Prior factors: {}, Between factors: {}",
        var_key_map.len(),
        if needs_prior { "1" } else { "0" },
        graph.edges_se3.len()
    );
    if !args.no_noise && !repair_summary.is_clean() {
        info!(
            "Information repair ({} edges, {} materially repaired, {} unit-weighted): \
             clamped directions carry no information; unit-weighted edges optimize \
             unweighted cost, not χ²",
            repair_summary.edges, repair_summary.materially_repaired, repair_summary.unit_weighted
        );
    }

    let init_cost_start = Instant::now();
    let init_state = initialize_optimization_state(&mut problem).map_err(|e| {
        apex_solver::core::CoreError::SymbolicStructure(format!(
            "Failed to initialize optimization state for dataset {}",
            dataset_name
        ))
        .log_with_source(e)
    })?;
    let initial_cost = init_state.initial_cost;
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

    /// Helper macro to optionally attach a Rerun observer to a solver.
    macro_rules! attach_visualizer {
        ($solver:expr, $args:expr) => {
            #[cfg(feature = "visualization")]
            if $args.with_visualizer {
                use apex_solver::observers::RerunObserver;
                match RerunObserver::new(true) {
                    Ok(observer) => {
                        $solver.add_observer(observer);
                        info!("Rerun visualization enabled");
                    }
                    Err(e) => warn!("Failed to create Rerun observer: {}", e),
                }
            }
        };
    }

    let opt_start = Instant::now();
    let result = match optimizer_name {
        "GN" => {
            let config = GaussNewtonConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-10);

            let mut solver = GaussNewton::with_config(config);
            attach_visualizer!(solver, args);
            solver.optimize(&mut problem)?
        }
        "DL" => {
            let config = DogLegConfig::new()
                .with_max_iterations(max_iter)
                .with_cost_tolerance(cost_tol)
                .with_parameter_tolerance(param_tol)
                .with_gradient_tolerance(1e-10);

            let mut solver = DogLeg::with_config(config);
            attach_visualizer!(solver, args);
            solver.optimize(&mut problem)?
        }
        _ => {
            let config = apply_lm_tuning(
                LevenbergMarquardtConfig::new()
                    .with_max_iterations(max_iter)
                    .with_cost_tolerance(cost_tol)
                    .with_parameter_tolerance(param_tol)
                    .with_gradient_tolerance(1e-10),
                args,
            );

            let mut solver = LevenbergMarquardt::with_config(config);
            attach_visualizer!(solver, args);
            solver.optimize(&mut problem)?
        }
    };

    let optimization_time = opt_start.elapsed();

    // Update graph vertices with optimized values for chi2 computation
    for (&id, &key) in &var_key_map {
        if let Some(vertex) = graph.vertices_se3.get_mut(&id) {
            use nalgebra::{Quaternion, Vector3};
            let val = result.parameters[key].to_dvector();
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
        "  Unweighted: {:.6e} -> {:.6e}",
        initial_metrics.unweighted_cost, final_metrics.unweighted_cost
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
                apex_io::IoError::log_with_source(
                    apex_io::IoError::FileCreationFailed {
                        path: parent.display().to_string(),
                        reason: "Failed to create directory".to_string(),
                    },
                    e,
                )
            })?;
        }

        // Write updated graph to file (graph already updated in-place above)
        use apex_solver::apex_io::GraphLoader;
        G2oLoader::write(&graph, &output_path)?;

        info!("Saved optimized graph to: {}", output_path.display());
    }

    let status = match result.status {
        OptimizationStatus::Converged
        | OptimizationStatus::CostToleranceReached
        | OptimizationStatus::GradientToleranceReached
        | OptimizationStatus::StalledNoProgress
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

    if args.verbose_iters {
        apex_solver::init_logger_with_directives(
            tracing::Level::INFO,
            "info,apex_solver::optimizer=debug",
        );
    } else {
        init_logger();
    }

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
            // Any dataset file on disk is runnable standalone (grid3D, rim, …).
            let se2_path = format!("{}/{}.g2o", ODOMETRY_DATA_DIR_2D, args.dataset);
            let se3_path = format!("{}/{}.g2o", ODOMETRY_DATA_DIR_3D, args.dataset);
            if std::path::Path::new(&se2_path).exists() {
                se2_run.push(args.dataset.as_str());
                (se2_run, se3_run)
            } else if std::path::Path::new(&se3_path).exists() {
                se3_run.push(args.dataset.as_str());
                (se2_run, se3_run)
            } else {
                warn!("Unknown dataset: {}", args.dataset);
                warn!("Using default: running all datasets");
                (se2_datasets, se3_datasets)
            }
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
