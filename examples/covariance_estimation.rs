//! Example demonstrating covariance (uncertainty) estimation in pose graph optimization.
//!
//! This example shows how to:
//! 1. Enable covariance computation in optimizer configuration
//! 2. Access per-variable covariance matrices from optimization results
//! 3. Interpret covariance as uncertainty in the tangent space
//!
//! Covariance estimation is essential for:
//! - State estimation and sensor fusion (e.g., SLAM, navigation)
//! - Uncertainty propagation in robotics
//! - Active loop closure and exploration
//! - Data association and outlier rejection

use apex_solver::{
    core::problem::Problem,
    factors::BetweenFactorSE2,
    linalg::LinearSolverType,
    manifold::ManifoldType,
    optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig},
};
use nalgebra::DVector;
use std::collections::HashMap;

fn main() {
    println!("=== Covariance Estimation Example ===\n");

    // Create a simple 2D pose graph with 4 poses forming a chain
    let mut problem = Problem::new();

    println!("Creating pose graph with 4 SE2 poses...");

    // Add odometry constraints (between consecutive poses)
    problem.add_residual_block(
        &["x0", "x1"],
        Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.0)), // Move 1m forward
        None,
    );

    problem.add_residual_block(
        &["x1", "x2"],
        Box::new(BetweenFactorSE2::new(1.0, 0.0, 0.1)), // Move 1m forward, turn 0.1 rad
        None,
    );

    problem.add_residual_block(
        &["x2", "x3"],
        Box::new(BetweenFactorSE2::new(1.0, 0.0, -0.1)), // Move 1m forward, turn -0.1 rad
        None,
    );

    // Add a loop closure constraint
    problem.add_residual_block(
        &["x0", "x3"],
        Box::new(BetweenFactorSE2::new(3.0, 0.0, 0.0)), // Direct measurement from x0 to x3
        None,
    );

    println!("Added 3 odometry constraints and 1 loop closure\n");

    // Initial values (noisy estimates)
    let mut initial_values = HashMap::new();
    initial_values.insert(
        "x0".to_string(),
        (ManifoldType::SE2, DVector::from_vec(vec![0.0, 0.0, 0.0])),
    );
    initial_values.insert(
        "x1".to_string(),
        (ManifoldType::SE2, DVector::from_vec(vec![0.95, 0.05, 0.02])),
    );
    initial_values.insert(
        "x2".to_string(),
        (ManifoldType::SE2, DVector::from_vec(vec![1.9, 0.1, 0.12])),
    );
    initial_values.insert(
        "x3".to_string(),
        (ManifoldType::SE2, DVector::from_vec(vec![2.85, 0.15, 0.05])),
    );

    // Create optimizer with covariance computation ENABLED
    println!("Configuring Levenberg-Marquardt optimizer with covariance estimation...");
    let config = LevenbergMarquardtConfig::new()
        .with_linear_solver_type(LinearSolverType::SparseCholesky)
        .with_max_iterations(50)
        .with_cost_tolerance(1e-6)
        .with_parameter_tolerance(1e-6)
        .with_compute_covariances(true); // ← KEY: Enable covariance computation

    let mut solver = LevenbergMarquardt::with_config(config);

    // Solve the optimization problem
    println!("Running optimization...\n");
    match solver.optimize(&problem, &initial_values) {
        Ok(result) => {
            println!("✓ Optimization succeeded!");
            println!("  Status: {:?}", result.status);
            println!("  Iterations: {}", result.iterations);
            println!("  Initial cost: {:.6e}", result.initial_cost);
            println!("  Final cost: {:.6e}", result.final_cost);
            println!(
                "  Cost reduction: {:.2}%\n",
                100.0 * (result.initial_cost - result.final_cost) / result.initial_cost
            );

            // Display optimized poses
            println!("Optimized poses:");
            for (name, variable) in result.parameters.iter() {
                let vec = variable.to_vector();
                println!(
                    "  {}: x={:.4}, y={:.4}, theta={:.4}",
                    name, vec[0], vec[1], vec[2]
                );
            }

            // Check if covariances were computed
            if let Some(covariances) = &result.covariances {
                println!("\n=== Covariance Analysis ===");
                println!("Covariances computed for {} variables\n", covariances.len());

                // Display covariances in order
                for name in ["x0", "x1", "x2", "x3"] {
                    if let Some(cov) = covariances.get(name) {
                        println!("Variable '{}':", name);
                        println!("  Full 3×3 covariance matrix (tangent space):");
                        for i in 0..3 {
                            print!("    [");
                            for j in 0..3 {
                                print!(" {:9.6e}", cov[(i, j)]);
                            }
                            println!(" ]");
                        }

                        // Extract standard deviations (uncertainty)
                        let std_x = cov[(0, 0)].sqrt();
                        let std_y = cov[(1, 1)].sqrt();
                        let std_theta = cov[(2, 2)].sqrt();

                        println!("  Standard deviations (1-sigma uncertainty):");
                        println!("    σ_x     = {:.6} m", std_x);
                        println!("    σ_y     = {:.6} m", std_y);
                        println!(
                            "    σ_theta = {:.6} rad ({:.3}°)",
                            std_theta,
                            std_theta.to_degrees()
                        );

                        // 95% confidence intervals (2-sigma)
                        println!("  95% confidence intervals (±2σ):");
                        println!("    x:     ±{:.6} m", 2.0 * std_x);
                        println!("    y:     ±{:.6} m", 2.0 * std_y);
                        println!(
                            "    theta: ±{:.6} rad (±{:.3}°)",
                            2.0 * std_theta,
                            (2.0 * std_theta).to_degrees()
                        );
                        println!();
                    }
                }

                // Interpretation
                println!("=== Interpretation ===");
                println!("- Covariance matrices are in the tangent space (local coordinates)");
                println!("- Diagonal elements are variances; off-diagonal are correlations");
                println!("- Smaller values = higher confidence (less uncertainty)");
                println!("- First pose (x0) typically has smallest uncertainty (anchor)");
                println!("- Uncertainty typically grows with distance from constraints");
                println!("- Loop closures reduce uncertainty in the graph");
            } else {
                println!("\n⚠ No covariances computed!");
                println!("Make sure .with_compute_covariances(true) is set in config.");
            }
        }
        Err(e) => {
            eprintln!("✗ Optimization failed: {:?}", e);
        }
    }

    println!("\n=== Usage Notes ===");
    println!("To enable covariance estimation in your code:");
    println!("  1. Set .with_compute_covariances(true) in optimizer config");
    println!("  2. Access result.covariances after optimization");
    println!("  3. Each variable gets its own tangent-space covariance matrix");
    println!("  4. For SE2: 3×3 matrix [x, y, theta]");
    println!("  5. For SE3: 6×6 matrix [trans_xyz, rot_xyz]");
    println!("\nNote: Covariance computation adds ~10-20% overhead.");
}
