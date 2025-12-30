//! Integration test for DoubleSphere camera with SelfCalibration
//!
//! This test verifies end-to-end optimization of:
//! - Camera pose (SE3)
//! - 3D landmarks (RN)
//! - Camera intrinsics (fx, fy, cx, cy)
//!
//! The double_sphere model is the simplest camera model with no distortion,
//! making it an ideal starting point for testing self-calibration.

use apex_solver::core::problem::Problem;
use apex_solver::factors::ProjectionFactor;
use apex_solver::factors::camera::double_sphere::DoubleSphereCamera;
use apex_solver::factors::camera::{CameraModel, SelfCalibration};
use apex_solver::manifold::ManifoldType;
use apex_solver::manifold::se3::SE3;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DVector, Matrix2xX};
use std::collections::HashMap;

mod camera_test_utils;
use camera_test_utils::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn test_double_sphere_self_calibration_50_points() -> TestResult {
    // ============================================================================
    // 1. Ground Truth Setup
    // ============================================================================

    // DoubleSphere camera with typical parameters (VGA resolution, ~60° FOV)
    let true_camera = DoubleSphereCamera::new(
        500.0, 500.0, 320.0, 240.0, // fx, fy, cx, cy
        0.5, 0.8, // xi, alpha
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
        let pixel = true_camera.project(landmark).ok_or("Projection failed")?;
        observations.push(pixel);
    }

    // Convert to Matrix2xX format (2 rows, N columns)
    let observations_matrix = Matrix2xX::from_columns(&observations);

    // ============================================================================
    // 3. Add Noise to Create Initial Estimates
    // ============================================================================

    // Noise levels: 5cm landmarks, 2° rotation, 5% intrinsics
    let noisy_pose = perturb_pose(&true_pose, 0.03, 1.5, 123);
    let noisy_landmarks = perturb_landmarks(&true_landmarks, 0.03, 456);
    let noisy_intrinsics = perturb_intrinsics(&[500.0, 500.0, 320.0, 240.0, 0.5, 0.8], 0.04, 789);

    // ============================================================================
    // 4. Build Optimization Problem
    // ============================================================================

    let mut problem = Problem::new();

    // Create projection factor with SelfCalibration (optimizes all parameters)
    let factor: ProjectionFactor<DoubleSphereCamera, SelfCalibration> =
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

    let result = solver.optimize(&problem, &initial_values)?;

    // ============================================================================
    // 7. Verify Convergence
    // ============================================================================

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
        .ok_or("Missing intrinsics parameter")?
        .to_vector();
    let true_intrinsics = [500.0, 500.0, 320.0, 240.0, 0.5, 0.8];

    // Check intrinsic parameter recovery
    for i in 0..4 {
        let param_error = (final_intrinsics[i] - true_intrinsics[i]).abs() / true_intrinsics[i];

        // DoubleSphere should recover intrinsics very accurately (no distortion, well-conditioned)
        assert!(
            param_error < 0.10,
            "Parameter {} should recover within 10%, got {:.4}%",
            i,
            param_error * 100.0
        );
    }

    Ok(())
}
