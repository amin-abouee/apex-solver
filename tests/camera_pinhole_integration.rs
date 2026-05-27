//! Integration test for Pinhole camera with multi-camera calibration
//!
//! Simulates a realistic camera calibration scenario:
//! - 200 calibration points on planar wall
//! - 5 cameras viewing the target from different positions
//! - Simultaneous optimization of poses, landmarks, and intrinsics
//! - Tests the simplest camera model with no distortion
//!
//! Pinhole Camera Model:
//! - 4 intrinsic parameters: fx, fy, cx, cy
//! - Linear projection model (no distortion)
//! - Best parameter recovery expected among all camera models

use apex_camera_models::{CameraModel, DistortionModel, PinholeCamera, PinholeParams};
use apex_manifolds::LieGroup;
use apex_solver::JacobianMode;
use apex_solver::ManifoldType;
use apex_solver::core::VarKey;
use apex_solver::core::problem::Problem;
use apex_solver::factors::ProjectionFactor;
use apex_solver::factors::SelfCalibration;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DVector, Matrix2xX, Vector2};

mod camera_test_utils;
use camera_test_utils::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Test Pinhole camera self-calibration with multi-camera setup.
///
/// Scenario: 5 cameras on a horizontal arc viewing 200 calibration points on
/// a planar wall. All cameras share the same intrinsics.
///
/// Ground truth camera: 600x400 image, pinhole model (no distortion)
#[test]
fn test_pinhole_multi_camera_calibration_200_points() -> TestResult {
    // ============================================================================
    // 1. Ground Truth Setup - 600x400 Pinhole Camera
    // ============================================================================

    // Pinhole camera parameters (simplest model, no distortion)
    // - Focal length 200px gives wide FOV to see entire wall
    // - No distortion parameters
    let true_camera = PinholeCamera::new(
        PinholeParams {
            fx: 200.0,
            fy: 200.0,
            cx: 300.0,
            cy: 200.0,
        },
        DistortionModel::None,
    )?;

    // Image bounds for projection validation
    let img_width = 600.0;
    let img_height = 400.0;

    // ============================================================================
    // 2. Generate Calibration Target (200 Points on Wall at Z=3m)
    // ============================================================================

    // Planar wall calibration target
    let true_landmarks = generate_wall_calibration_points(20, 10, 0.1, 3.0);
    assert_eq!(
        true_landmarks.len(),
        200,
        "Should generate exactly 200 calibration points"
    );

    // ============================================================================
    // 3. Generate 5 Camera Poses with Wider Baseline
    // ============================================================================

    let true_poses = generate_arc_camera_poses(5, 0.8, 3.0);
    assert_eq!(true_poses.len(), 5, "Should generate 5 camera poses");

    // ============================================================================
    // 4. Project Points and Verify ALL Points Visible
    // ============================================================================

    let mut all_observations: Vec<Vec<Vector2<f64>>> = Vec::new();

    for (cam_idx, pose) in true_poses.iter().enumerate() {
        let mut cam_observations = Vec::with_capacity(true_landmarks.len());

        for (lm_idx, landmark) in true_landmarks.iter().enumerate() {
            let p_cam = pose.act(landmark, None, None);

            assert!(
                true_camera.project(&p_cam).is_ok(),
                "Camera {} cannot see landmark {}: p_cam = {:?}",
                cam_idx,
                lm_idx,
                p_cam
            );

            let uv = true_camera.project(&p_cam)?;

            assert!(
                uv.x >= 0.0 && uv.x < img_width && uv.y >= 0.0 && uv.y < img_height,
                "Camera {} landmark {} projects outside image: uv = ({:.1}, {:.1})",
                cam_idx,
                lm_idx,
                uv.x,
                uv.y
            );

            cam_observations.push(uv);
        }

        assert_eq!(
            cam_observations.len(),
            true_landmarks.len(),
            "Camera {} should see all {} landmarks",
            cam_idx,
            true_landmarks.len()
        );

        all_observations.push(cam_observations);
    }

    // ============================================================================
    // 5. Add Noise to Create Initial Estimates
    // ============================================================================

    let noisy_landmarks = perturb_landmarks(&true_landmarks, 0.01, 100);

    let noisy_poses: Vec<_> = true_poses
        .iter()
        .enumerate()
        .map(|(i, p)| perturb_pose(p, 0.02, 1.0, 200 + i as u64 * 10))
        .collect();

    let true_intrinsics = [200.0, 200.0, 300.0, 200.0];
    let noisy_intrinsics = perturb_intrinsics(&true_intrinsics, 0.02, 300);

    // ============================================================================
    // 6. Build Optimization Problem
    // ============================================================================

    let mut problem = Problem::new(JacobianMode::Sparse);

    // Add variables first (poses, landmarks, intrinsics)
    let pose_keys: Vec<VarKey> = noisy_poses
        .iter()
        .map(|pose| {
            problem.add_variable(
                ManifoldType::SE3,
                DVector::from_column_slice(pose.as_param_slice()),
            )
        })
        .collect();
    let landmarks_key = problem.add_variable(ManifoldType::RN, flatten_landmarks(&noisy_landmarks));
    let intrinsics_key = problem.add_variable(
        ManifoldType::RN,
        DVector::from_vec(noisy_intrinsics.clone()),
    );

    for (cam_idx, observations) in all_observations.iter().enumerate() {
        let obs_matrix = Matrix2xX::from_columns(observations);

        let factor: ProjectionFactor<PinholeCamera, SelfCalibration> =
            ProjectionFactor::new(obs_matrix, true_camera);

        problem.add_residual_block(
            &[pose_keys[cam_idx], landmarks_key, intrinsics_key],
            Box::new(factor),
            None,
        );
    }

    for dof in 0..6 {
        problem.fix_variable(pose_keys[0], dof);
    }

    // ============================================================================
    // 8. Configure and Run Optimization
    // ============================================================================

    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-8)
        .with_parameter_tolerance(1e-8)
        .with_gradient_tolerance(1e-10)
        .with_damping(1e-3);

    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&mut problem)?;

    // ============================================================================
    // 9. Verify Convergence
    // ============================================================================

    assert!(
        matches!(
            result.status,
            OptimizationStatus::Converged
                | OptimizationStatus::CostToleranceReached
                | OptimizationStatus::ParameterToleranceReached
                | OptimizationStatus::GradientToleranceReached
        ),
        "Optimization should converge, got: {:?}",
        result.status
    );

    // ============================================================================
    // 10. Verify Cost Reduction
    // ============================================================================

    let cost_reduction = (result.initial_cost - result.final_cost) / result.initial_cost;

    // Pinhole should achieve excellent cost reduction (no distortion)
    assert!(
        cost_reduction > 0.95,
        "Cost should reduce by >95%, got {:.2}% reduction",
        cost_reduction * 100.0
    );

    // ============================================================================
    // 11. Verify Reprojection RMSE
    // ============================================================================

    let total_observations: usize = all_observations.iter().map(|o| o.len()).sum();
    let rmse = (result.final_cost / total_observations as f64).sqrt();

    // Pinhole should achieve excellent RMSE (tightest tolerance)
    assert!(
        rmse < 1.0,
        "Reprojection RMSE should be < 1 pixel, got {:.4} pixels",
        rmse
    );

    // ============================================================================
    // 12. Verify Intrinsic Parameter Recovery
    // ============================================================================

    let final_intrinsics = result.parameters[intrinsics_key].to_dvector();

    let param_names = ["fx", "fy", "cx", "cy"];

    // All parameters should be well-conditioned (no distortion)
    let tolerances = [0.05, 0.05, 0.05, 0.05];

    for i in 0..4 {
        let relative_error =
            (final_intrinsics[i] - true_intrinsics[i]).abs() / true_intrinsics[i].abs();

        assert!(
            relative_error < tolerances[i],
            "{} should recover within {:.0}% of ground truth, got {:.2}% error",
            param_names[i],
            tolerances[i] * 100.0,
            relative_error * 100.0
        );
    }

    Ok(())
}

/// Test with 3 cameras for faster execution (good for CI)
#[test]
fn test_pinhole_3_cameras_calibration() -> TestResult {
    let true_camera = PinholeCamera::new(
        PinholeParams {
            fx: 200.0,
            fy: 200.0,
            cx: 300.0,
            cy: 200.0,
        },
        DistortionModel::None,
    )?;

    let img_width = 600.0;
    let img_height = 400.0;

    let true_landmarks = generate_wall_calibration_points(20, 10, 0.1, 3.0);
    let true_poses = generate_arc_camera_poses(3, 0.6, 3.0);

    let mut all_observations: Vec<Vec<Vector2<f64>>> = Vec::new();

    for pose in &true_poses {
        let mut cam_obs = Vec::new();
        for landmark in &true_landmarks {
            let p_cam = pose.act(landmark, None, None);
            if let Ok(uv) = true_camera.project(&p_cam)
                && uv.x >= 0.0
                && uv.x < img_width
                && uv.y >= 0.0
                && uv.y < img_height
            {
                cam_obs.push(uv);
            }
        }
        all_observations.push(cam_obs);
    }

    let noisy_landmarks = perturb_landmarks(&true_landmarks, 0.01, 100);
    let noisy_poses: Vec<_> = true_poses
        .iter()
        .enumerate()
        .map(|(i, p)| perturb_pose(p, 0.02, 1.0, 200 + i as u64 * 10))
        .collect();
    let true_intrinsics = [200.0, 200.0, 300.0, 200.0];
    let noisy_intrinsics = perturb_intrinsics(&true_intrinsics, 0.02, 300);

    let mut problem = Problem::new(JacobianMode::Sparse);

    // Add variables first
    let pose_keys: Vec<VarKey> = noisy_poses
        .iter()
        .map(|pose| {
            problem.add_variable(
                ManifoldType::SE3,
                DVector::from_column_slice(pose.as_param_slice()),
            )
        })
        .collect();
    let landmarks_key = problem.add_variable(ManifoldType::RN, flatten_landmarks(&noisy_landmarks));
    let intrinsics_key =
        problem.add_variable(ManifoldType::RN, DVector::from_vec(noisy_intrinsics));

    for (cam_idx, observations) in all_observations.iter().enumerate() {
        let obs_matrix = Matrix2xX::from_columns(observations);
        let factor: ProjectionFactor<PinholeCamera, SelfCalibration> =
            ProjectionFactor::new(obs_matrix, true_camera);

        problem.add_residual_block(
            &[pose_keys[cam_idx], landmarks_key, intrinsics_key],
            Box::new(factor),
            None,
        );
    }

    for dof in 0..6 {
        problem.fix_variable(pose_keys[0], dof);
    }

    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-8)
        .with_parameter_tolerance(1e-8)
        .with_damping(1e-3);

    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&mut problem)?;

    assert!(
        matches!(
            result.status,
            OptimizationStatus::Converged
                | OptimizationStatus::CostToleranceReached
                | OptimizationStatus::ParameterToleranceReached
                | OptimizationStatus::GradientToleranceReached
        ),
        "3-camera calibration should converge, got: {:?}",
        result.status
    );

    let cost_reduction = (result.initial_cost - result.final_cost) / result.initial_cost;
    assert!(
        cost_reduction > 0.70,
        "Cost should reduce by >70%, got {:.2}%",
        cost_reduction * 100.0
    );

    Ok(())
}
