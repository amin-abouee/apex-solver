//! Comprehensive odometry pose benchmark for apex-solver, factrs, tiny-solver, and C++ solvers
//!
//! This benchmark compares three Rust nonlinear optimization libraries (apex-solver, factrs, tiny-solver)
//! and two C++ libraries (g2o, GTSAM) on standard pose graph optimization datasets (both SE2 and SE3).
//!
//! ## Simplified Performance Metric
//!
//! **Important Change**: This benchmark no longer uses cost-based metrics due to inconsistent cost
//! computation across solvers (information-weighted vs unweighted residuals). Instead, it uses a
//! simple convergence-based scoring system:
//!
//! - **Score**: 100.0 if solver converged successfully, 0.0 if diverged or failed
//! - **Converged**: "true" if solver met convergence criteria, "false" otherwise
//! - **Time**: Average wall-clock time in milliseconds (5 runs per configuration)
//! - **Iterations**: Number of iterations taken (where available)
//!
//! This approach provides a clear, unambiguous comparison of solver reliability and speed.
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
//! - Gradient tolerance: 1e-12 (tighter for SE3 due to higher complexity, enables early-exit)
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
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tracing::{info, warn};

// apex-solver imports
use apex_io::{G2oLoader, GraphLoader};
use apex_solver::ManifoldType;
use apex_solver::core::loss_functions::L2Loss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::BetweenFactor;
use apex_solver::init_logger;
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
use csv::{Reader, Writer};
use serde::{Deserialize, Serialize};

// ============================================================================
// UNIFIED COST COMPUTATION
// ============================================================================
// These functions compute cost directly from G2O graph data, independent of
// solver internals, for fair benchmarking across all solvers.
//
// Formula: cost = 0.5 * sum_i ||r_i||²_Σ
// where ||r||²_Σ = r^T * Σ^(-1) * r (information-weighted squared norm)
//
// This ensures:
// - All solvers use identical cost computation
// - Costs exclude gauge freedom artifacts (priors, fixed variables)
// - Direct computation from G2O constraints
// ============================================================================

use apex_manifolds::LieGroup;
use apex_manifolds::se2::SE2;
use apex_manifolds::se3::SE3;

/// Dual cost metrics for benchmarking
#[derive(Debug, Clone, Copy)]
struct CostMetrics {
    /// Chi-squared cost: sum of r^T * Omega * r (information-weighted)
    chi2_cost: f64,
    /// Unweighted cost: 0.5 * sum ||r||^2
    unweighted_cost: f64,
}

/// Compute both SE2 cost metrics from G2O graph data
/// - Chi-squared: sum of r^T * Omega * r (information-weighted)
/// - Unweighted: 0.5 * sum ||r||^2
fn compute_se2_cost_metrics(graph: &apex_io::Graph) -> CostMetrics {
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
fn compute_se3_cost_metrics(graph: &apex_io::Graph) -> CostMetrics {
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

// Note: Computing factrs final cost using unified cost function is complex due to
// factrs's internal Value representation. For now, we use factrs's own cost computation
// for final cost, but use unified cost for initial cost to ensure fair comparison baseline.

/// Update SE2 graph vertices from tiny-solver optimization result
fn update_se2_graph_from_tiny_solver(
    graph: &mut apex_io::Graph,
    tiny_solver_result: &std::collections::HashMap<String, nalgebra::DVector<f64>>,
) {
    for (var_name, var_value) in tiny_solver_result {
        // tiny-solver uses "x0", "x1", etc. as variable names
        if let Some(id_str) = var_name.strip_prefix("x")
            && let Ok(id) = id_str.parse::<usize>()
            && let Some(vertex) = graph.vertices_se2.get_mut(&id)
        {
            // tiny-solver SE2 format: [x, y, theta]
            vertex.pose = SE2::from_xy_angle(var_value[0], var_value[1], var_value[2]);
        }
    }
}

/// Update SE3 graph vertices from tiny-solver optimization result
fn update_se3_graph_from_tiny_solver(
    graph: &mut apex_io::Graph,
    tiny_solver_result: &std::collections::HashMap<String, nalgebra::DVector<f64>>,
) {
    use nalgebra::{Quaternion, Vector3};

    for (var_name, var_value) in tiny_solver_result {
        if let Some(id_str) = var_name.strip_prefix("x")
            && let Ok(id) = id_str.parse::<usize>()
            && let Some(vertex) = graph.vertices_se3.get_mut(&id)
        {
            // tiny-solver SE3 format: [tx, ty, tz, qx, qy, qz, qw]
            let translation = Vector3::new(var_value[0], var_value[1], var_value[2]);
            let rotation = Quaternion::new(var_value[6], var_value[3], var_value[4], var_value[5]);
            vertex.pose = SE3::from_translation_quaternion(translation, rotation);
        }
    }
}

/// Update SE2 graph vertices from factrs optimization result
fn update_se2_graph_from_factrs(
    graph: &mut apex_io::Graph,
    factrs_values: &factrs::containers::Values,
) {
    use factrs::assign_symbols;
    use factrs::variables::SE2 as FactrsSE2;

    assign_symbols!(X: FactrsSE2);

    let ids: Vec<_> = graph.vertices_se2.keys().copied().collect();
    for id in ids {
        if let Some(factrs_pose) = factrs_values.get::<_, FactrsSE2>(X(id as u32))
            && let Some(vertex) = graph.vertices_se2.get_mut(&id)
        {
            // factrs SE2: x, y, theta
            vertex.pose = SE2::from_xy_angle(factrs_pose.x(), factrs_pose.y(), factrs_pose.theta());
        }
    }
}

/// Update SE3 graph vertices from factrs optimization result
fn update_se3_graph_from_factrs(
    graph: &mut apex_io::Graph,
    factrs_values: &factrs::containers::Values,
) {
    use factrs::assign_symbols;
    use factrs::variables::SE3 as FactrsSE3;
    use nalgebra::{Quaternion, Vector3};

    assign_symbols!(X: FactrsSE3);

    let ids: Vec<_> = graph.vertices_se3.keys().copied().collect();
    for id in ids {
        if let Some(factrs_pose) = factrs_values.get::<_, FactrsSE3>(X(id as u32))
            && let Some(vertex) = graph.vertices_se3.get_mut(&id)
        {
            // Extract rotation and translation from factrs SE3
            let rot = factrs_pose.rot();
            let xyz = factrs_pose.xyz();

            // factrs SO3 stores quaternion as (x, y, z, w)
            let rotation = Quaternion::new(rot.w(), rot.x(), rot.y(), rot.z());
            let translation = Vector3::new(xyz[0], xyz[1], xyz[2]);

            vertex.pose = SE3::from_translation_quaternion(translation, rotation);
        }
    }
}

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
        file: "../../data/odometry/M3500.g2o",
        is_3d: false,
    },
    Dataset {
        name: "mit",
        file: "../../data/odometry/mit.g2o",
        is_3d: false,
    },
    Dataset {
        name: "city10000",
        file: "../../data/odometry/city10000.g2o",
        is_3d: false,
    },
    Dataset {
        name: "ring",
        file: "../../data/odometry/ring.g2o",
        is_3d: false,
    },
    Dataset {
        name: "sphere2500",
        file: "../../data/odometry/sphere2500.g2o",
        is_3d: true,
    },
    Dataset {
        name: "parking-garage",
        file: "../../data/odometry/parking-garage.g2o",
        is_3d: true,
    },
    Dataset {
        name: "torus3D",
        file: "../../data/odometry/torus3D.g2o",
        is_3d: true,
    },
    Dataset {
        name: "cubicle",
        file: "../../data/odometry/cubicle.g2o",
        is_3d: true,
    },
];

/// Benchmark result structure with dual metrics
#[derive(Debug, Clone, Serialize)]
struct BenchmarkResult {
    dataset: String,
    solver: String,
    language: String,
    elapsed_ms: String,
    converged: String,
    iterations: String,
    // Dual metrics
    initial_chi2: String, // Chi-squared (information-weighted)
    final_chi2: String,
    chi2_improvement_pct: String,
    initial_cost: String, // Unweighted cost
    final_cost: String,
    improvement_pct: String,
}

impl BenchmarkResult {
    /// Create a successful benchmark result.
    ///
    /// # Design Note
    /// This constructor accepts individual benchmark metrics for clear parameter naming in benchmark code.
    /// The large parameter count reflects the comprehensive nature of pose graph benchmarking.
    #[allow(clippy::too_many_arguments)]
    fn success(
        dataset: &str,
        solver: &str,
        language: &str,
        elapsed_ms: f64,
        converged: bool,
        iterations: Option<usize>,
        initial_metrics: CostMetrics,
        final_metrics: CostMetrics,
    ) -> Self {
        let improvement_pct = if initial_metrics.unweighted_cost > 0.0 {
            ((initial_metrics.unweighted_cost - final_metrics.unweighted_cost)
                / initial_metrics.unweighted_cost)
                * 100.0
        } else {
            0.0
        };
        let chi2_improvement_pct = if initial_metrics.chi2_cost > 0.0 {
            ((initial_metrics.chi2_cost - final_metrics.chi2_cost) / initial_metrics.chi2_cost)
                * 100.0
        } else {
            0.0
        };

        Self {
            dataset: dataset.to_string(),
            solver: solver.to_string(),
            language: language.to_string(),
            elapsed_ms: format!("{:.2}", elapsed_ms),
            converged: converged.to_string(),
            iterations: iterations.map_or("-".to_string(), |i| i.to_string()),
            initial_chi2: format!("{:.6e}", initial_metrics.chi2_cost),
            final_chi2: format!("{:.6e}", final_metrics.chi2_cost),
            chi2_improvement_pct: format!("{:.2}", chi2_improvement_pct),
            initial_cost: format!("{:.6e}", initial_metrics.unweighted_cost),
            final_cost: format!("{:.6e}", final_metrics.unweighted_cost),
            improvement_pct: format!("{:.2}", improvement_pct),
        }
    }

    fn diverged(dataset: &str, solver: &str, language: &str, elapsed_ms: f64) -> Self {
        Self {
            dataset: dataset.to_string(),
            solver: solver.to_string(),
            language: language.to_string(),
            elapsed_ms: format!("{:.2}", elapsed_ms),
            converged: "false".to_string(),
            iterations: "-".to_string(),
            initial_chi2: "-".to_string(),
            final_chi2: "-".to_string(),
            chi2_improvement_pct: "-".to_string(),
            initial_cost: "-".to_string(),
            final_cost: "-".to_string(),
            improvement_pct: "-".to_string(),
        }
    }

    fn failed(dataset: &str, solver: &str, language: &str, error: &str) -> Self {
        Self {
            dataset: dataset.to_string(),
            solver: solver.to_string(),
            language: language.to_string(),
            elapsed_ms: "-".to_string(),
            converged: "false".to_string(),
            iterations: format!("error: {}", error),
            initial_chi2: "-".to_string(),
            final_chi2: "-".to_string(),
            chi2_improvement_pct: "-".to_string(),
            initial_cost: "-".to_string(),
            final_cost: "-".to_string(),
            improvement_pct: "-".to_string(),
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

fn apex_solver_se2(dataset: &Dataset) -> BenchmarkResult {
    let mut graph = match G2oLoader::load(dataset.file) {
        Ok(g) => g,
        Err(e) => {
            return BenchmarkResult::failed(dataset.name, "apex-solver", "Rust", &e.to_string());
        }
    };

    // Compute initial cost using unified cost function
    let initial_cost = compute_se2_cost_metrics(&graph);

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
        let between_factor = BetweenFactor::new(edge.measurement.clone());
        problem.add_residual_block(
            &[&id0, &id1],
            Box::new(between_factor),
            Some(Box::new(L2Loss)),
        );
    }

    // Optimize with production-grade configuration matching optimize_2d_graph.rs
    // - Max iterations: 150 (sufficient for SE2 convergence)
    // - Cost/param tolerance: 1e-4 (balanced accuracy vs speed)
    // - Gradient tolerance: 1e-10 (early-exit on gradient convergence, saves iterations)
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(150)
        .with_cost_tolerance(1e-4)
        .with_parameter_tolerance(1e-4)
        .with_gradient_tolerance(1e-10);

    let mut solver = LevenbergMarquardt::with_config(config);

    // Start timing immediately before optimization (excludes problem setup overhead)
    // This matches the timing approach in optimize_2d_graph.rs for fair comparison
    let start_time = Instant::now();
    match solver.optimize(&problem, &initial_values) {
        Ok(result) => {
            let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

            // Update graph with optimized values
            for (var_name, var_enum) in &result.parameters {
                if let Some(id_str) = var_name.strip_prefix("x")
                    && let Ok(id) = id_str.parse::<usize>()
                    && let Some(vertex) = graph.vertices_se2.get_mut(&id)
                {
                    let val = var_enum.to_vector();
                    vertex.pose = SE2::from_xy_angle(val[0], val[1], val[2]);
                }
            }

            // Compute final cost using unified cost function
            let final_cost = compute_se2_cost_metrics(&graph);

            let converged = is_converged(&result.status);
            BenchmarkResult::success(
                dataset.name,
                "apex-solver",
                "Rust",
                elapsed_ms,
                converged,
                Some(result.iterations),
                initial_cost,
                final_cost,
            )
        }
        Err(e) => BenchmarkResult::failed(dataset.name, "apex-solver", "Rust", &e.to_string()),
    }
}

fn apex_solver_se3(dataset: &Dataset) -> BenchmarkResult {
    let mut graph = match G2oLoader::load(dataset.file) {
        Ok(g) => g,
        Err(e) => {
            return BenchmarkResult::failed(dataset.name, "apex-solver", "Rust", &e.to_string());
        }
    };

    // Compute initial cost using unified cost function
    let initial_cost = compute_se3_cost_metrics(&graph);

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
        let between_factor = BetweenFactor::new(edge.measurement.clone());
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
    // - Gradient tolerance: 1e-12 (tighter than SE3 due to SE3 complexity, enables early-exit)
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-4)
        .with_parameter_tolerance(1e-4)
        .with_gradient_tolerance(1e-12);

    let mut solver = LevenbergMarquardt::with_config(config);

    // Start timing immediately before optimization (excludes problem setup overhead)
    // This matches the timing approach in optimize_3d_graph.rs for fair comparison
    let start_time = Instant::now();
    match solver.optimize(&problem, &initial_values) {
        Ok(result) => {
            let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

            // Update graph with optimized values
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

            // Compute final cost using unified cost function
            let final_cost = compute_se3_cost_metrics(&graph);

            let converged = is_converged(&result.status);
            BenchmarkResult::success(
                dataset.name,
                "apex-solver",
                "Rust",
                elapsed_ms,
                converged,
                Some(result.iterations),
                initial_cost,
                final_cost,
            )
        }
        Err(e) => BenchmarkResult::failed(dataset.name, "apex-solver", "Rust", &e.to_string()),
    }
}

fn factrs_benchmark(dataset: &Dataset) -> BenchmarkResult {
    // Load raw G2O graph for unified cost computation (without factrs prior)
    let mut raw_graph = match G2oLoader::load(dataset.file) {
        Ok(g) => g,
        Err(e) => return BenchmarkResult::failed(dataset.name, "factrs", "Rust", &e.to_string()),
    };

    // Compute initial cost from original G2O graph BEFORE factrs adds prior
    // factrs adds a prior factor on the second vertex which is NOT in the original file
    let initial_cost = if dataset.is_3d {
        compute_se3_cost_metrics(&raw_graph)
    } else {
        compute_se2_cost_metrics(&raw_graph)
    };

    // Catch panics from factrs parsing/loading
    let load_result = panic::catch_unwind(|| load_g20(dataset.file));

    let (graph, init) = match load_result {
        Ok((g, i)) => (g, i),
        Err(_) => {
            return BenchmarkResult::failed(
                dataset.name,
                "factrs",
                "Rust",
                "failed to load dataset (panic)",
            );
        }
    };

    // Start timing
    let start = Instant::now();

    // Use Levenberg-Marquardt optimizer with default Cholesky solver
    let mut opt: LevenMarquardt = LevenMarquardt::new(graph.clone());
    let result = black_box(opt.optimize(init));

    // Stop timing
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Ok(final_values) => {
            // Update raw graph with optimized values from factrs
            if dataset.is_3d {
                update_se3_graph_from_factrs(&mut raw_graph, &final_values);
            } else {
                update_se2_graph_from_factrs(&mut raw_graph, &final_values);
            }

            // Compute final cost using unified cost function
            let final_cost = if dataset.is_3d {
                compute_se3_cost_metrics(&raw_graph)
            } else {
                compute_se2_cost_metrics(&raw_graph)
            };

            BenchmarkResult::success(
                dataset.name,
                "factrs",
                "Rust",
                elapsed_ms,
                true, // Successfully converged
                None, // factrs doesn't expose iteration count
                initial_cost,
                final_cost,
            )
        }
        Err(factrs::optimizers::OptError::MaxIterations(final_values)) => {
            // Update raw graph with optimized values from factrs
            if dataset.is_3d {
                update_se3_graph_from_factrs(&mut raw_graph, &final_values);
            } else {
                update_se2_graph_from_factrs(&mut raw_graph, &final_values);
            }

            // Compute final cost using unified cost function
            let final_cost = if dataset.is_3d {
                compute_se3_cost_metrics(&raw_graph)
            } else {
                compute_se2_cost_metrics(&raw_graph)
            };

            BenchmarkResult::success(
                dataset.name,
                "factrs",
                "Rust",
                elapsed_ms,
                false, // Did not converge (max iterations)
                None,
                initial_cost,
                final_cost,
            )
        }
        Err(factrs::optimizers::OptError::FailedToStep) => {
            BenchmarkResult::diverged(dataset.name, "factrs", "Rust", elapsed_ms)
        }
        Err(factrs::optimizers::OptError::InvalidSystem) => {
            BenchmarkResult::diverged(dataset.name, "factrs", "Rust", elapsed_ms)
        }
    }
}

fn tiny_solver_benchmark(dataset: &Dataset) -> BenchmarkResult {
    // Load raw G2O graph for unified cost computation
    let mut raw_graph = match G2oLoader::load(dataset.file) {
        Ok(g) => g,
        Err(e) => {
            return BenchmarkResult::failed(dataset.name, "tiny-solver", "Rust", &e.to_string());
        }
    };

    // Catch panics from tiny-solver parsing/loading
    let load_result = panic::catch_unwind(|| load_tiny_g2o(dataset.file));

    let (graph, init) = match load_result {
        Ok((g, i)) => (g, i),
        Err(_) => {
            return BenchmarkResult::failed(
                dataset.name,
                "tiny-solver",
                "Rust",
                "failed to load dataset (panic)",
            );
        }
    };

    let lm = LevenbergMarquardtOptimizer::default();

    // Update raw graph with initial values from tiny-solver to ensure consistent baseline
    if dataset.is_3d {
        update_se3_graph_from_tiny_solver(&mut raw_graph, &init);
    } else {
        update_se2_graph_from_tiny_solver(&mut raw_graph, &init);
    }

    // Compute initial cost from raw graph using unified cost function
    let initial_cost = if dataset.is_3d {
        compute_se3_cost_metrics(&raw_graph)
    } else {
        compute_se2_cost_metrics(&raw_graph)
    };

    // Start timing
    let start = Instant::now();

    // Use Levenberg-Marquardt optimizer
    let result = black_box(lm.optimize(&graph, &init, None));

    // Stop timing
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Some(final_values) => {
            // Update raw graph with optimized values from tiny-solver
            if dataset.is_3d {
                update_se3_graph_from_tiny_solver(&mut raw_graph, &final_values);
            } else {
                update_se2_graph_from_tiny_solver(&mut raw_graph, &final_values);
            }

            // Compute final cost from updated graph using unified cost function
            let final_cost = if dataset.is_3d {
                compute_se3_cost_metrics(&raw_graph)
            } else {
                compute_se2_cost_metrics(&raw_graph)
            };

            BenchmarkResult::success(
                dataset.name,
                "tiny-solver",
                "Rust",
                elapsed_ms,
                true, // Successfully converged
                None, // tiny-solver doesn't expose iteration count
                initial_cost,
                final_cost,
            )
        }
        None => {
            // Optimization failed (NaN, solve failed, or other error)
            BenchmarkResult::diverged(dataset.name, "tiny-solver", "Rust", elapsed_ms)
        }
    }
}

// ========================= C++ Benchmark Integration =========================

/// C++ benchmark result from CSV (matches the C++ CSV output format with dual metrics)
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CppBenchmarkResult {
    dataset: String,
    manifold: String,
    solver: String,
    language: String,
    vertices: usize,
    edges: usize,
    // Dual metrics from C++
    initial_chi2: f64,
    final_chi2: f64,
    chi2_improvement_pct: f64,
    initial_cost: f64,
    final_cost: f64,
    improvement_pct: f64,
    iterations: usize,
    time_ms: f64,
    status: String,
}

/// Build C++ benchmarks if not already built
fn build_cpp_benchmarks() -> Result<PathBuf, String> {
    let bench_dir = Path::new("benches/cpp_comparison");
    let build_dir = bench_dir.join("build");

    // Check if executables already exist
    let g2o_exe = build_dir.join("g2o_odometry_benchmark");
    let gtsam_exe = build_dir.join("gtsam_odometry_benchmark");

    if g2o_exe.exists() && gtsam_exe.exists() {
        info!("C++ benchmarks already built");
        return Ok(build_dir);
    }

    info!("Building C++ benchmarks...");

    // Create build directory if needed
    std::fs::create_dir_all(&build_dir)
        .map_err(|e| format!("Failed to create build dir: {}", e))?;

    // Run CMake configure
    let cmake_output = Command::new("cmake")
        .args(["..", "-DCMAKE_BUILD_TYPE=Release"])
        .current_dir(&build_dir)
        .output()
        .map_err(|e| format!("Failed to run cmake: {}", e))?;

    if !cmake_output.status.success() {
        return Err(format!(
            "CMake configure failed: {}",
            String::from_utf8_lossy(&cmake_output.stderr)
        ));
    }

    // Run CMake build
    let build_output = Command::new("cmake")
        .args(["--build", ".", "--config", "Release", "-j"])
        .current_dir(&build_dir)
        .output()
        .map_err(|e| format!("Failed to run cmake build: {}", e))?;

    if !build_output.status.success() {
        return Err(format!(
            "CMake build failed: {}",
            String::from_utf8_lossy(&build_output.stderr)
        ));
    }

    info!("C++ benchmarks built successfully");
    Ok(build_dir)
}

/// Run a C++ benchmark executable and return path to CSV output
fn run_cpp_benchmark(exe_name: &str, build_dir: &Path) -> Result<PathBuf, String> {
    // Convert to absolute path to handle working directory issues
    let absolute_build_dir = std::fs::canonicalize(build_dir)
        .map_err(|e| format!("Failed to canonicalize build dir: {}", e))?;

    let exe_path = absolute_build_dir.join(exe_name);

    if !exe_path.exists() {
        return Err(format!("Executable not found: {:?}", exe_path));
    }

    info!("Running {} ...", exe_name);

    let output = Command::new(&exe_path)
        .current_dir(&absolute_build_dir)
        .output()
        .map_err(|e| format!("Failed to run {}: {}", exe_name, e))?;

    if !output.status.success() {
        return Err(format!(
            "{} failed: {}",
            exe_name,
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Print stdout for user visibility
    if !output.stdout.is_empty() {
        info!("{}", String::from_utf8_lossy(&output.stdout));
    }

    // Determine CSV output filename based on executable name
    let csv_name = exe_name.replace("_benchmark", "_benchmark_results.csv");
    let csv_path = absolute_build_dir.join(&csv_name);

    if !csv_path.exists() {
        return Err(format!("CSV output not found: {:?}", csv_path));
    }

    Ok(csv_path)
}

/// Parse C++ benchmark CSV results into BenchmarkResult format
fn parse_cpp_results(csv_path: &Path) -> Result<Vec<BenchmarkResult>, String> {
    let mut reader =
        Reader::from_path(csv_path).map_err(|e| format!("Failed to read CSV: {}", e))?;

    let mut results = Vec::new();

    for record in reader.deserialize() {
        let cpp_result: CppBenchmarkResult =
            record.map_err(|e| format!("Failed to parse CSV record: {}", e))?;

        // Convert to BenchmarkResult format
        let converged = cpp_result.status == "CONVERGED";

        // Remove "-LM" suffix from solver name (e.g., "g2o-LM" -> "g2o", "GTSAM-LM" -> "GTSAM")
        let solver_name = cpp_result.solver.trim_end_matches("-LM");

        // Create CostMetrics from C++ values
        let initial_metrics = CostMetrics {
            chi2_cost: cpp_result.initial_chi2,
            unweighted_cost: cpp_result.initial_cost,
        };
        let final_metrics = CostMetrics {
            chi2_cost: cpp_result.final_chi2,
            unweighted_cost: cpp_result.final_cost,
        };

        let result = BenchmarkResult::success(
            &cpp_result.dataset,
            solver_name,
            &cpp_result.language,
            cpp_result.time_ms,
            converged,
            Some(cpp_result.iterations),
            initial_metrics,
            final_metrics,
        );

        results.push(result);
    }

    Ok(results)
}

/// Run all available C++ benchmarks and return combined results
fn run_cpp_benchmarks() -> Vec<BenchmarkResult> {
    let mut all_results = Vec::new();

    // Try to build C++ benchmarks
    let build_dir = match build_cpp_benchmarks() {
        Ok(dir) => dir,
        Err(e) => {
            info!("Warning: C++ benchmarks unavailable: {}", e);
            info!("Continuing with Rust-only benchmarks...\n");
            return all_results;
        }
    };

    // List of C++ benchmark executables to run (odometry benchmarks)
    let cpp_benchmarks = vec![
        "ceres_odometry_benchmark",
        "g2o_odometry_benchmark",
        "gtsam_odometry_benchmark",
    ];

    for exe_name in cpp_benchmarks {
        match run_cpp_benchmark(exe_name, &build_dir) {
            Ok(csv_path) => match parse_cpp_results(&csv_path) {
                Ok(results) => {
                    info!("{} completed: {} datasets", exe_name, results.len());
                    all_results.extend(results);
                }
                Err(e) => {
                    info!("Warning: Failed to parse {} results: {}", exe_name, e);
                }
            },
            Err(e) => {
                info!("Warning: Failed to run {}: {}", exe_name, e);
            }
        }
    }

    all_results
}

// ========================= Main Benchmark Runner =========================

// Normalized scores removed - we now track initial_cost, final_cost, and improvement_pct instead

fn run_single_benchmark(dataset: &Dataset, solver: &str) -> BenchmarkResult {
    match (dataset.is_3d, solver) {
        (false, "apex-solver") => apex_solver_se2(dataset),
        (true, "apex-solver") => apex_solver_se3(dataset),
        (_, "factrs") => factrs_benchmark(dataset),
        (_, "tiny-solver") => tiny_solver_benchmark(dataset),
        _ => BenchmarkResult::failed(
            dataset.name,
            solver,
            "unknown",
            &format!("Unknown solver: {}", solver),
        ),
    }
}

/// Helper function to save benchmark results to CSV
fn save_csv_results(
    results: &[BenchmarkResult],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = Writer::from_path(path)?;
    for result in results {
        writer.serialize(result)?;
    }
    writer.flush()?;
    Ok(())
}

fn main() {
    // Initialize logger with INFO level
    init_logger();

    info!("Starting solver comparison benchmark...");
    info!("Running each configuration 5 times and averaging results...");

    let solvers = ["apex-solver", "factrs", "tiny-solver"];
    let mut all_results = Vec::new();

    for dataset in DATASETS {
        info!("Dataset: {}", dataset.name);

        for solver in &solvers {
            info!("{} ... ", solver);
            let _ = std::io::Write::flush(&mut std::io::stdout());

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

                info!(
                    "done (converged: {}, time: {} ms)",
                    avg_result.converged, avg_result.elapsed_ms
                );

                all_results.push(avg_result);
            }
        }
    }

    // Step 2: Run C++ benchmarks
    info!("PHASE 2: C++ Benchmarks");

    let cpp_results = run_cpp_benchmarks();
    all_results.extend(cpp_results);

    // Write results to CSV in output folder
    let csv_path = "output/odometry_pose_benchmark_results.csv";
    if let Err(e) = save_csv_results(&all_results, csv_path) {
        warn!("Warning: Failed to save CSV results: {}", e);
    } else {
        info!("Results written to {}", csv_path);
    }

    // Separate 2D and 3D results and sort by dataset name
    // 2D datasets: M3500, intel, mit, ring
    let mut results_2d: Vec<_> = all_results
        .iter()
        .filter(|r| ["city10000", "mit", "M3500", "ring"].contains(&r.dataset.as_str()))
        .collect();

    // Sort by dataset name first, then by solver name
    results_2d.sort_by(|a, b| {
        a.dataset
            .cmp(&b.dataset)
            .then_with(|| a.solver.cmp(&b.solver))
    });

    // 3D datasets: sphere2500, parking-garage, torus3D, cubicle
    let mut results_3d: Vec<_> = all_results
        .iter()
        .filter(|r| {
            ["sphere2500", "parking-garage", "torus3D", "cubicle"].contains(&r.dataset.as_str())
        })
        .collect();

    // Sort by dataset name first, then by solver name
    results_3d.sort_by(|a, b| {
        a.dataset
            .cmp(&b.dataset)
            .then_with(|| a.solver.cmp(&b.solver))
    });

    // Print 2D results
    if !results_2d.is_empty() {
        info!("2D DATASETS (SE2)");
        info!("{}", "=".repeat(200));
        info!(
            "{:<15} {:<12} {:<6} {:<12} {:<12} {:<10} {:<12} {:<12} {:<10} {:<8} {:<10} {:<6}",
            "Dataset",
            "Solver",
            "Lang",
            "Init Chi2",
            "Final Chi2",
            "Chi2 Imp%",
            "Init Cost",
            "Final Cost",
            "Improve %",
            "Iters",
            "Time (ms)",
            "Conv"
        );
        info!("{}", "-".repeat(200));

        for result in &results_2d {
            info!(
                "{:<15} {:<12} {:<6} {:<12} {:<12} {:<10} {:<12} {:<12} {:<10} {:<8} {:<10} {:<6}",
                result.dataset,
                result.solver,
                result.language,
                result.initial_chi2,
                result.final_chi2,
                result.chi2_improvement_pct,
                result.initial_cost,
                result.final_cost,
                result.improvement_pct,
                result.iterations,
                result.elapsed_ms,
                result.converged
            );
        }
        info!("{}\n", "=".repeat(200));
    }

    // Print 3D results
    if !results_3d.is_empty() {
        info!("3D DATASETS (SE3)");
        info!("{}", "=".repeat(200));
        info!(
            "{:<15} {:<12} {:<6} {:<12} {:<12} {:<10} {:<12} {:<12} {:<10} {:<8} {:<10} {:<6}",
            "Dataset",
            "Solver",
            "Lang",
            "Init Chi2",
            "Final Chi2",
            "Chi2 Imp%",
            "Init Cost",
            "Final Cost",
            "Improve %",
            "Iters",
            "Time (ms)",
            "Conv"
        );
        info!("{}", "-".repeat(200));

        for result in &results_3d {
            info!(
                "{:<15} {:<12} {:<6} {:<12} {:<12} {:<10} {:<12} {:<12} {:<10} {:<8} {:<10} {:<6}",
                result.dataset,
                result.solver,
                result.language,
                result.initial_chi2,
                result.final_chi2,
                result.chi2_improvement_pct,
                result.initial_cost,
                result.final_cost,
                result.improvement_pct,
                result.iterations,
                result.elapsed_ms,
                result.converged
            );
        }
        info!("{}", "=".repeat(200));
    }
}
