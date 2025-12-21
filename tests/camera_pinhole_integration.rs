//! Integration test for Pinhole camera with SelfCalibration
//!
//! This test verifies end-to-end optimization of:
//! - Camera pose (SE3)
//! - 3D landmarks (RN)
//! - Camera intrinsics (fx, fy, cx, cy)
//!
//! The pinhole model is the simplest camera model with no distortion,
//! making it an ideal starting point for testing self-calibration.

// Allow expect() in test code
#![allow(clippy::expect_used)]
#![allow(clippy::unwrap_used)]

use apex_solver::core::problem::Problem;
use apex_solver::factors::camera::pinhole::PinholeCamera;
use apex_solver::factors::camera::{CameraModel, ProjectionFactor, SelfCalibration};
use apex_solver::manifold::ManifoldType;
use apex_solver::manifold::se3::SE3;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DVector, Matrix2xX};
use std::collections::HashMap;
use tracing::info;

mod camera_test_utils;
use camera_test_utils::*;

#[test]
fn test_pinhole_self_calibration_50_points() {
    // ============================================================================
    // 1. Ground Truth Setup
    // ============================================================================

    // Pinhole camera with typical parameters (VGA resolution, ~60° FOV)
    let true_camera = PinholeCamera::new(
        520.0, // fx
        520.0, // fy
        320.0, // cx
        240.0, // cy
    );

    // Camera at identity pose (world frame = camera frame)
    let true_pose = SE3::identity();

    // Generate 50 3D points in realistic scene (2-5m depth, ±2m XY spread)
    let true_landmarks = generate_scene_points(50, 42);

    // ============================================================================
    // 2. Generate Perfect Observations
    // ============================================================================

    let mut observations = Vec::with_capacity(true_landmarks.len());

    for landmark in &true_landmarks {
        // Project to image (landmarks already in camera frame at identity pose)
        let pixel = true_camera
            .project(landmark)
            .expect("Projection should succeed");
        observations.push(pixel);
    }

    // Convert to Matrix2xX format (2 rows, N columns)
    let observations_matrix = Matrix2xX::from_columns(&observations);

    // ============================================================================
    // 3. Add Noise to Create Initial Estimates
    // ============================================================================

    // Noise levels: 5cm landmarks, 2° rotation, 5% intrinsics
    let noisy_pose = perturb_pose(&true_pose, 0.05, 2.0, 123);
    let noisy_landmarks = perturb_landmarks(&true_landmarks, 0.05, 456);
    let noisy_intrinsics = perturb_intrinsics(&[520.0, 520.0, 320.0, 240.0], 0.05, 789);

    // ============================================================================
    // 4. Build Optimization Problem
    // ============================================================================

    let mut problem = Problem::new();

    // Create projection factor with SelfCalibration (optimizes all parameters)
    let factor: ProjectionFactor<PinholeCamera, SelfCalibration> =
        ProjectionFactor::new(observations_matrix, true_camera);

    // Add residual block with three variables: [pose, landmarks, intrinsics]
    problem.add_residual_block(
        &["camera_0", "landmarks", "intrinsics"],
        Box::new(factor),
        None, // No robust loss function
    );

    // ============================================================================
    // 5. Initialize Variables
    // ============================================================================

    let mut initial_values = HashMap::new();

    // Camera pose (SE3 manifold)
    initial_values.insert(
        "camera_0".to_string(),
        (ManifoldType::SE3, noisy_pose.into()),
    );

    // Landmarks (RN manifold, flattened [x0,y0,z0,x1,y1,z1,...])
    initial_values.insert(
        "landmarks".to_string(),
        (ManifoldType::RN, flatten_landmarks(&noisy_landmarks)),
    );

    // Intrinsics (RN manifold, [fx, fy, cx, cy])
    initial_values.insert(
        "intrinsics".to_string(),
        (
            ManifoldType::RN,
            DVector::from_vec(noisy_intrinsics.clone()),
        ),
    );

    // ============================================================================
    // 6. Optimize with Levenberg-Marquardt
    // ============================================================================

    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-6)
        .with_parameter_tolerance(1e-6)
        .with_damping(1e-3);

    let mut solver = LevenbergMarquardt::with_config(config);

    let result = solver
        .optimize(&problem, &initial_values)
        .expect("Optimization should succeed");

    // ============================================================================
    // 7. Verify Convergence
    // ============================================================================

    info!("\n========================================");
    info!("Pinhole Self-Calibration Results:");
    info!("========================================");
    info!("  Initial cost:       {:.6e}", result.initial_cost);
    info!("  Final cost:         {:.6e}", result.final_cost);
    info!("  Iterations:         {}", result.iterations);
    info!("  Status:             {:?}", result.status);
    info!(
        "  Final gradient norm: {:.6e}",
        result
            .convergence_info
            .as_ref()
            .map(|ci| ci.final_gradient_norm)
            .unwrap_or(0.0)
    );
    info!(
        "  Final param update:  {:.6e}",
        result
            .convergence_info
            .as_ref()
            .map(|ci| ci.final_parameter_update_norm)
            .unwrap_or(0.0)
    );
    info!("========================================");

    // Check convergence status
    assert!(
        matches!(
            result.status,
            OptimizationStatus::Converged
                | OptimizationStatus::CostToleranceReached
                | OptimizationStatus::ParameterToleranceReached
        ),
        "Optimization should converge, got: {:?}",
        result.status
    );

    // ============================================================================
    // 8. Check Cost Reduction
    // ============================================================================

    let cost_reduction = (result.initial_cost - result.final_cost) / result.initial_cost;
    info!(
        "  Cost reduction:     {:.4} ({:.2}%)\n",
        cost_reduction,
        cost_reduction * 100.0
    );

    assert!(
        cost_reduction > 0.90,
        "Cost should reduce by >90%, got {:.4}",
        cost_reduction
    );

    // ============================================================================
    // 9. Verify Parameter Recovery (Optional)
    // ============================================================================

    // Extract optimized intrinsics
    let final_intrinsics = result
        .parameters
        .get("intrinsics")
        .expect("Should have intrinsics")
        .to_vector();
    let true_intrinsics = [520.0, 520.0, 320.0, 240.0];

    info!("Intrinsics Recovery:");
    info!(
        "  True intrinsics:  fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
        true_intrinsics[0], true_intrinsics[1], true_intrinsics[2], true_intrinsics[3]
    );
    info!(
        "  Initial (noisy):  fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
        noisy_intrinsics[0], noisy_intrinsics[1], noisy_intrinsics[2], noisy_intrinsics[3]
    );
    info!(
        "  Final (optimized): fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
        final_intrinsics[0], final_intrinsics[1], final_intrinsics[2], final_intrinsics[3]
    );

    // Check intrinsic parameter recovery
    for i in 0..4 {
        let param_error = (final_intrinsics[i] - true_intrinsics[i]).abs() / true_intrinsics[i];
        info!("  Param {} relative error: {:.4}%", i, param_error * 100.0);

        // Pinhole should recover intrinsics very accurately (no distortion, well-conditioned)
        assert!(
            param_error < 0.10,
            "Parameter {} should recover within 10%, got {:.4}%",
            i,
            param_error * 100.0
        );
    }

    info!("========================================\n");
}
