//! Integration tests for Apex Solver
//!
//! These tests verify end-to-end optimization performance on real G2O datasets.
//! They ensure that the optimizers converge correctly and produce expected results.
//!
//! # Test Coverage
//!
//! - **SE3 (3D) Datasets**: sphere2500.g2o, parking-garage.g2o
//! - **SE2 (2D) Datasets**: intel.g2o, ring.g2o
//!
//! # Metrics Verified
//!
//! Each test verifies:
//! - Number of vertices and edges match expected values
//! - Optimization converges successfully
//! - Cost improvement exceeds threshold (>85%)
//! - Execution time is reasonable
//! - Iterations don't hit maximum limit
//! - Final cost is finite (not NaN or Inf)
//!
//! # Running Tests
//!
//! ```bash
//! # Run fast tests only (ring, intel)
//! cargo test
//!
//! # Run all tests including slow ones (sphere2500, parking-garage)
//! cargo test -- --include-ignored
//!
//! # Run only slow tests
//! cargo test -- --ignored
//! ```

use apex_io::{G2oLoader, GraphLoader};
use apex_solver::ManifoldType;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BetweenFactor, PriorFactor};
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::dvector;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Test result capturing all optimization metrics
#[derive(Debug)]
struct TestResult {
    vertices: usize,
    edges: usize,
    final_cost: f64,
    improvement_pct: f64,
    iterations: usize,
    elapsed_time: Duration,
    status: OptimizationStatus,
    success: bool,
}

/// Run SE3 (3D) optimization on a dataset
///
/// # Arguments
///
/// * `dataset_name` - Name of the dataset file (without .g2o extension)
/// * `max_iterations` - Maximum number of optimization iterations
/// * `use_prior` - Whether to add a prior factor on the first vertex
///
/// # Returns
///
/// `TestResult` containing all optimization metrics
fn run_se3_optimization(
    dataset_name: &str,
    max_iterations: usize,
    use_prior: bool,
) -> Result<TestResult, Box<dyn std::error::Error>> {
    // Load the G2O graph file (from workspace root, relative to crate)
    let dataset_path = format!("../../data/odometry/{}.g2o", dataset_name);
    let graph = G2oLoader::load(&dataset_path)?;

    let num_vertices = graph.vertices_se3.len();
    let num_edges = graph.edges_se3.len();

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

    // Add prior factor if requested (for gauge freedom)
    if use_prior
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
    }

    // Add SE3 between factors
    for edge in &graph.edges_se3 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);
        let relative_pose = edge.measurement.clone();

        let between_factor = BetweenFactor::new(relative_pose);
        problem.add_residual_block(&[&id0, &id1], Box::new(between_factor), None);
    }

    // Configure and run Levenberg-Marquardt optimizer
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(max_iterations)
        .with_cost_tolerance(1e-4)
        .with_parameter_tolerance(1e-4)
        .with_damping(1e-3);

    let mut solver = LevenbergMarquardt::with_config(config);

    let start_time = Instant::now();
    let result = solver.optimize(&problem, &initial_values)?;
    let elapsed_time = start_time.elapsed();

    // Calculate improvement percentage
    let improvement_pct = if result.initial_cost > 0.0 {
        ((result.initial_cost - result.final_cost) / result.initial_cost) * 100.0
    } else {
        0.0
    };

    let success = matches!(
        result.status,
        OptimizationStatus::Converged
            | OptimizationStatus::CostToleranceReached
            | OptimizationStatus::ParameterToleranceReached
            | OptimizationStatus::GradientToleranceReached
    );

    let test_result = TestResult {
        vertices: num_vertices,
        edges: num_edges,
        final_cost: result.final_cost,
        improvement_pct,
        iterations: result.iterations,
        elapsed_time,
        status: result.status.clone(),
        success,
    };

    Ok(test_result)
}

/// Run SE2 (2D) optimization on a dataset
///
/// # Arguments
///
/// * `dataset_name` - Name of the dataset file (without .g2o extension)
/// * `max_iterations` - Maximum number of optimization iterations
/// * `use_prior` - Whether to add a prior factor on the first vertex
///
/// # Returns
///
/// `TestResult` containing all optimization metrics
fn run_se2_optimization(
    dataset_name: &str,
    max_iterations: usize,
    use_prior: bool,
) -> Result<TestResult, Box<dyn std::error::Error>> {
    // Load the G2O graph file (from workspace root, relative to crate)
    let dataset_path = format!("../../data/odometry/{}.g2o", dataset_name);
    let graph = G2oLoader::load(&dataset_path)?;

    let num_vertices = graph.vertices_se2.len();
    let num_edges = graph.edges_se2.len();

    // Create optimization problem
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add SE2 vertices as variables
    let mut vertex_ids: Vec<_> = graph.vertices_se2.keys().cloned().collect();
    vertex_ids.sort();

    for &id in &vertex_ids {
        if let Some(vertex) = graph.vertices_se2.get(&id) {
            let var_name = format!("x{}", id);
            let pose = &vertex.pose;
            let se2_data = dvector![pose.x(), pose.y(), pose.angle()];
            initial_values.insert(var_name, (ManifoldType::SE2, se2_data));
        }
    }

    // Add prior factor if requested (for gauge freedom)
    if use_prior
        && let Some(&first_id) = vertex_ids.first()
        && let Some(first_vertex) = graph.vertices_se2.get(&first_id)
    {
        let var_name = format!("x{}", first_id);
        let pose = &first_vertex.pose;
        let prior_value = dvector![pose.x(), pose.y(), pose.angle()];

        let prior_factor = PriorFactor {
            data: prior_value.clone(),
        };
        let huber_loss = HuberLoss::new(1.0)?;
        problem.add_residual_block(
            &[&var_name],
            Box::new(prior_factor),
            Some(Box::new(huber_loss)),
        );
    }

    // Add SE2 between factors
    for edge in &graph.edges_se2 {
        let id0 = format!("x{}", edge.from);
        let id1 = format!("x{}", edge.to);

        let between_factor = BetweenFactor::new(edge.measurement.clone());
        problem.add_residual_block(&[&id0, &id1], Box::new(between_factor), None);
    }

    // Configure and run Levenberg-Marquardt optimizer
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(max_iterations)
        .with_cost_tolerance(1e-4)
        .with_parameter_tolerance(1e-4)
        .with_damping(1e-3);

    let mut solver = LevenbergMarquardt::with_config(config);

    let start_time = Instant::now();
    let result = solver.optimize(&problem, &initial_values)?;
    let elapsed_time = start_time.elapsed();

    // Calculate improvement percentage
    let improvement_pct = if result.initial_cost > 0.0 {
        ((result.initial_cost - result.final_cost) / result.initial_cost) * 100.0
    } else {
        0.0
    };

    let success = matches!(
        result.status,
        OptimizationStatus::Converged
            | OptimizationStatus::CostToleranceReached
            | OptimizationStatus::ParameterToleranceReached
            | OptimizationStatus::GradientToleranceReached
    );

    let test_result = TestResult {
        vertices: num_vertices,
        edges: num_edges,
        final_cost: result.final_cost,
        improvement_pct,
        iterations: result.iterations,
        elapsed_time,
        status: result.status.clone(),
        success,
    };

    Ok(test_result)
}

// ============================================================================
// SE2 (2D) Integration Tests
// ============================================================================

/// Test optimization on ring.g2o (small SE2 dataset)
///
/// This is a fast test suitable for CI. The ring dataset has 434 vertices
/// and 459 edges, representing a small loop closure problem.
#[test]
fn test_ring_se2_converges() -> Result<(), Box<dyn std::error::Error>> {
    let result = run_se2_optimization("ring", 100, true)?;

    // Verify dataset size
    assert_eq!(
        result.vertices, 434,
        "ring.g2o should have 434 vertices, got {}",
        result.vertices
    );
    assert_eq!(
        result.edges, 459,
        "ring.g2o should have 459 edges, got {}",
        result.edges
    );

    // Verify convergence
    assert!(
        result.success,
        "Optimization did not converge. Status: {:?}, Iterations: {}, Final cost: {}",
        result.status, result.iterations, result.final_cost
    );

    // Verify cost improvement
    assert!(
        result.improvement_pct > 85.0,
        "Cost improvement too low: {:.2}% (expected >85%)",
        result.improvement_pct
    );

    // Verify iterations
    assert!(
        result.iterations < 100,
        "Hit maximum iterations: {}",
        result.iterations
    );

    // Verify numerical stability
    assert!(
        result.final_cost.is_finite(),
        "Final cost is not finite: {}",
        result.final_cost
    );

    // Verify performance (should complete in <2 seconds)
    assert!(
        result.elapsed_time.as_secs() < 2,
        "Optimization took too long: {:?}",
        result.elapsed_time
    );

    Ok(())
}

/// Test optimization on intel.g2o (medium SE2 dataset)
///
/// The Intel Research Lab dataset has 1,228 vertices and 1,483 edges.
/// This is a real-world indoor SLAM dataset.
#[test]
fn test_intel_se2_converges() -> Result<(), Box<dyn std::error::Error>> {
    let result = run_se2_optimization("intel", 100, true)?;

    // Verify dataset size
    assert_eq!(
        result.vertices, 1228,
        "intel.g2o should have 1228 vertices, got {}",
        result.vertices
    );
    assert_eq!(
        result.edges, 1483,
        "intel.g2o should have 1483 edges, got {}",
        result.edges
    );

    // Verify convergence
    assert!(
        result.success,
        "Optimization did not converge. Status: {:?}, Iterations: {}, Final cost: {}",
        result.status, result.iterations, result.final_cost
    );

    // Verify cost improvement
    assert!(
        result.improvement_pct > 85.0,
        "Cost improvement too low: {:.2}% (expected >85%)",
        result.improvement_pct
    );

    // Verify iterations
    assert!(
        result.iterations < 100,
        "Hit maximum iterations: {}",
        result.iterations
    );

    // Verify numerical stability
    assert!(
        result.final_cost.is_finite(),
        "Final cost is not finite: {}",
        result.final_cost
    );

    // Verify performance (should complete in <5 seconds)
    assert!(
        result.elapsed_time.as_secs() < 5,
        "Optimization took too long: {:?}",
        result.elapsed_time
    );

    Ok(())
}

// ============================================================================
// SE3 (3D) Integration Tests
// ============================================================================

/// Test optimization on sphere2500.g2o (medium SE3 dataset)
///
///
/// The sphere2500 dataset has 2,500 vertices and 4,949 edges, representing
/// a spherical topology commonly used for benchmarking.
#[test]
fn test_sphere2500_se3_converges() -> Result<(), Box<dyn std::error::Error>> {
    let result = run_se3_optimization("sphere2500", 100, true)?;

    // Verify dataset size
    assert_eq!(
        result.vertices, 2500,
        "sphere2500.g2o should have 2500 vertices, got {}",
        result.vertices
    );
    assert_eq!(
        result.edges, 4949,
        "sphere2500.g2o should have 4949 edges, got {}",
        result.edges
    );

    // Verify convergence
    assert!(
        result.success,
        "Optimization did not converge. Status: {:?}, Iterations: {}, Final cost: {}",
        result.status, result.iterations, result.final_cost
    );

    // Verify cost improvement
    assert!(
        result.improvement_pct > 99.0,
        "Cost improvement too low: {:.2}% (expected >99.0%)",
        result.improvement_pct
    );

    // Verify iterations
    assert!(
        result.iterations < 20,
        "Should complete in < 20 iterations: {}",
        result.iterations
    );

    // Verify numerical stability
    assert!(
        result.final_cost.is_finite(),
        "Final cost is not finite: {}",
        result.final_cost
    );

    // Verify performance (should complete in < 30 seconds)
    assert!(
        result.elapsed_time.as_secs() < 30,
        "Optimization took too long: {:?}",
        result.elapsed_time
    );

    Ok(())
}

/// Test optimization on parking-garage.g2o (medium SE3 dataset)
///
///
/// The parking-garage dataset has 1,661 vertices and 6,275 edges,
/// representing a real-world indoor SLAM scenario.
#[test]
fn test_parking_garage_se3_converges() -> Result<(), Box<dyn std::error::Error>> {
    let result = run_se3_optimization("parking-garage", 100, true)?;

    // Verify dataset size
    assert_eq!(
        result.vertices, 1661,
        "parking-garage.g2o should have 1661 vertices, got {}",
        result.vertices
    );
    assert_eq!(
        result.edges, 6275,
        "parking-garage.g2o should have 6275 edges, got {}",
        result.edges
    );

    // Verify convergence
    assert!(
        result.success,
        "Optimization did not converge. Status: {:?}, Iterations: {}, Final cost: {}",
        result.status, result.iterations, result.final_cost
    );

    // Verify cost improvement
    assert!(
        result.improvement_pct > 99.00,
        "Cost improvement too low: {:.2}% (expected >99.00%)",
        result.improvement_pct
    );

    // Verify iterations
    assert!(
        result.iterations < 20,
        "Should complete in < 20 iterations: {}",
        result.iterations
    );

    // Verify numerical stability
    assert!(
        result.final_cost.is_finite(),
        "Final cost is not finite: {}",
        result.final_cost
    );

    // Verify performance (should complete in < 20 seconds)
    assert!(
        result.elapsed_time.as_secs() < 20,
        "Optimization took too long: {:?}",
        result.elapsed_time
    );

    Ok(())
}
