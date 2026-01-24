//! Integration test for RadTan camera with multi-camera calibration
//!
//! Simulates realistic camera calibration scenario:
//! - 200 calibration points on planar wall
//! - 5 cameras on horizontal arc
//! - Simultaneous optimization of poses, landmarks, intrinsics
//!
//! RadTan (Radial-Tangential) Camera Model:
//! - 9 intrinsic parameters: fx, fy, cx, cy, k1, k2, p1, p2, k3
//! - Most widely used distortion model (OpenCV, MATLAB, SLAM systems)
//! - Also known as Brown-Conrady or Plumb Bob model
//! - Radial distortion (k1, k2, k3) + Tangential distortion (p1, p2)
//!
//! Expected performance:
//! - Cost reduction: >85% (complex 9-parameter model)
//! - RMSE: <2 pixels
//! - fx, fy, cx, cy: <5% error
//! - k1: <10% error (primary radial distortion)
//! - k2, p1, p2: <15% error (secondary distortion terms)
//! - k3: <0.3 absolute error (higher-order term, poorly constrained by planar target)

use apex_solver::core::problem::Problem;
use apex_solver::factors::ProjectionFactor;
use apex_solver::factors::camera::rad_tan::RadTanCamera;
use apex_solver::factors::camera::{CameraModel, SelfCalibration};
use apex_solver::manifold::LieGroup;
use apex_solver::manifold::ManifoldType;
use apex_solver::manifold::se3::SE3;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DVector, Matrix2xX, Vector2};
use std::collections::HashMap;

mod camera_test_utils;
use camera_test_utils::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn test_radtan_multi_camera_calibration_200_points() -> TestResult {
    // ============================================================================
    // 1. Ground Truth Setup
    // ============================================================================

    // RadTan camera with moderate distortion parameters
    // Use weaker distortion for better convergence with planar calibration
    let true_camera = RadTanCamera::new(
        200.0, 200.0, // fx, fy (focal lengths - wide FOV)
        300.0, 200.0, // cx, cy (principal point - centered for 600x400 image)
        0.1, -0.05, // k1, k2 (moderate radial distortion)
        0.001, -0.001, // p1, p2 (small tangential distortion)
        0.0,    // k3 (set to zero - not observable from planar target)
    );

    let img_width = 600.0;
    let img_height = 400.0;

    // ============================================================================
    // 2. Generate Calibration Target (200 points on planar wall)
    // ============================================================================

    // Generate wall: 20x10 grid, 0.1m spacing, at 3.0m depth
    let true_landmarks = generate_wall_calibration_points(20, 10, 0.1, 3.0);
    assert_eq!(true_landmarks.len(), 200, "Expected 200 calibration points");

    // ============================================================================
    // 3. Generate Camera Poses (5 cameras on horizontal arc)
    // ============================================================================

    // 5 cameras with 0.8m baseline, looking at wall from 3.0m
    let true_poses = generate_arc_camera_poses(5, 0.8, 3.0);
    assert_eq!(true_poses.len(), 5, "Expected 5 camera poses");

    // ============================================================================
    // 4. Project Points and Verify Visibility
    // ============================================================================

    let mut all_observations: Vec<Vec<Vector2<f64>>> = Vec::new();

    for (cam_idx, pose) in true_poses.iter().enumerate() {
        let mut cam_observations = Vec::with_capacity(true_landmarks.len());

        for (lm_idx, landmark) in true_landmarks.iter().enumerate() {
            // Transform landmark to camera frame
            let p_cam = pose.act(landmark, None, None);

            // Verify point is in front of camera
            assert!(
                true_camera.is_valid_point(&p_cam),
                "Camera {} cannot see landmark {}: p_cam = {:?}",
                cam_idx,
                lm_idx,
                p_cam
            );

            // Project to image
            let uv = true_camera.project(&p_cam).unwrap_or_else(|| {
                panic!(
                    "Projection failed for camera {} landmark {}: p_cam = {:?}",
                    cam_idx, lm_idx, p_cam
                )
            });

            // Verify projection is inside image bounds
            assert!(
                uv.x >= 0.0 && uv.x < img_width && uv.y >= 0.0 && uv.y < img_height,
                "Camera {} landmark {} projects outside image: uv = ({:.1}, {:.1}), bounds = [0, {}] x [0, {}]",
                cam_idx,
                lm_idx,
                uv.x,
                uv.y,
                img_width,
                img_height
            );

            cam_observations.push(uv);
        }

        assert_eq!(
            cam_observations.len(),
            true_landmarks.len(),
            "Camera {} should see all landmarks",
            cam_idx
        );
        all_observations.push(cam_observations);
    }

    println!(
        "✓ All {} landmarks visible from all {} cameras",
        true_landmarks.len(),
        true_poses.len()
    );

    // ============================================================================
    // 5. Add Noise to Create Initial Estimates
    // ============================================================================

    // Noise levels: 1cm landmarks, 2cm pose translation, 1° rotation, 2% intrinsics
    let noisy_landmarks = perturb_landmarks(&true_landmarks, 0.01, 100);

    let noisy_poses: Vec<_> = true_poses
        .iter()
        .enumerate()
        .map(|(i, p)| perturb_pose(p, 0.02, 1.0, 200 + i as u64 * 10))
        .collect();

    let true_intrinsics = [200.0, 200.0, 300.0, 200.0, 0.1, -0.05, 0.001, -0.001, 0.0];
    let noisy_intrinsics = perturb_intrinsics(&true_intrinsics, 0.02, 300);

    // ============================================================================
    // 6. Build Optimization Problem
    // ============================================================================

    let mut problem = Problem::new();

    // Add projection factors for each camera
    for (cam_idx, observations) in all_observations.iter().enumerate() {
        // Convert observations to Matrix2xX format
        let obs_matrix = Matrix2xX::from_columns(observations);

        // Create projection factor with SelfCalibration
        let factor: ProjectionFactor<RadTanCamera, SelfCalibration> =
            ProjectionFactor::new(obs_matrix, true_camera);

        // Add residual block: [pose_i, landmarks, intrinsics]
        let pose_name = format!("pose_{}", cam_idx);
        problem.add_residual_block(
            &[&pose_name, "landmarks", "intrinsics"],
            Box::new(factor),
            None,
        );
    }

    // Fix first camera pose to anchor gauge freedom (prevent drift)
    for dof in 0..6 {
        problem.fix_variable("pose_0", dof);
    }

    println!(
        "✓ Built problem: {} cameras, {} landmarks, 9 intrinsics",
        true_poses.len(),
        true_landmarks.len()
    );

    // ============================================================================
    // 7. Initialize Variables
    // ============================================================================

    let mut initial_values = HashMap::new();

    // Camera poses (SE3 manifold)
    for (i, pose) in noisy_poses.iter().enumerate() {
        initial_values.insert(
            format!("pose_{}", i),
            (ManifoldType::SE3, pose.clone().into()),
        );
    }

    // Landmarks (RN manifold, flattened [x0,y0,z0,x1,y1,z1,...])
    initial_values.insert(
        "landmarks".to_string(),
        (ManifoldType::RN, flatten_landmarks(&noisy_landmarks)),
    );

    // Intrinsics (RN manifold, 9 parameters)
    initial_values.insert(
        "intrinsics".to_string(),
        (
            ManifoldType::RN,
            DVector::from_vec(noisy_intrinsics.clone()),
        ),
    );

    // ============================================================================
    // 8. Optimize with Levenberg-Marquardt
    // ============================================================================

    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-8)
        .with_parameter_tolerance(1e-8)
        .with_gradient_tolerance(1e-10)
        .with_damping(1e-3);

    let mut solver = LevenbergMarquardt::with_config(config);

    println!("\nStarting optimization...");
    let result = solver.optimize(&problem, &initial_values)?;

    // ============================================================================
    // 9. Verify Convergence
    // ============================================================================

    println!("\n=== Optimization Results ===");
    println!("Status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Initial cost: {:.6e}", result.initial_cost);
    println!("Final cost: {:.6e}", result.final_cost);

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
    // 10. Check Cost Reduction
    // ============================================================================

    let cost_reduction = (result.initial_cost - result.final_cost) / result.initial_cost;
    println!("Cost reduction: {:.2}%", cost_reduction * 100.0);

    assert!(
        cost_reduction > 0.85,
        "Cost should reduce by >85% (RadTan has 9 parameters, complex model), got {:.2}%",
        cost_reduction * 100.0
    );

    // ============================================================================
    // 11. Compute Reprojection RMSE
    // ============================================================================

    let final_intrinsics = result
        .parameters
        .get("intrinsics")
        .ok_or("Missing intrinsics")?
        .to_vector();

    let final_camera = RadTanCamera::new(
        final_intrinsics[0],
        final_intrinsics[1],
        final_intrinsics[2],
        final_intrinsics[3],
        final_intrinsics[4],
        final_intrinsics[5],
        final_intrinsics[6],
        final_intrinsics[7],
        final_intrinsics[8],
    );

    let final_landmarks_vec = result
        .parameters
        .get("landmarks")
        .ok_or("Missing landmarks")?
        .to_vector();

    let final_landmarks = unflatten_landmarks(&final_landmarks_vec);

    let mut total_error_sq = 0.0;
    let mut total_observations = 0;

    for (cam_idx, true_observations) in all_observations.iter().enumerate() {
        let pose_name = format!("pose_{}", cam_idx);
        let final_pose_vec = result
            .parameters
            .get(&pose_name)
            .ok_or_else(|| format!("Missing pose {}", pose_name))?
            .to_vector();
        let final_pose = SE3::from(final_pose_vec);

        for (lm_idx, true_obs) in true_observations.iter().enumerate() {
            let p_cam = final_pose.act(&final_landmarks[lm_idx], None, None);
            if let Some(pred_obs) = final_camera.project(&p_cam) {
                let error = (pred_obs - true_obs).norm();
                total_error_sq += error * error;
                total_observations += 1;
            }
        }
    }

    let rmse = (total_error_sq / total_observations as f64).sqrt();
    println!("Reprojection RMSE: {:.4} pixels", rmse);

    assert!(
        rmse < 2.5,
        "RMSE should be <2.5 pixels (RadTan with noise), got {:.4}",
        rmse
    );

    // ============================================================================
    // 12. Verify Parameter Recovery
    // ============================================================================

    println!("\n=== Parameter Recovery ===");

    // Focal lengths and principal point (well-conditioned)
    let param_names = ["fx", "fy", "cx", "cy"];
    for i in 0..4 {
        let error = (final_intrinsics[i] - true_intrinsics[i]).abs() / true_intrinsics[i];
        println!(
            "{}: true={:.2}, final={:.2}, error={:.2}%",
            param_names[i],
            true_intrinsics[i],
            final_intrinsics[i],
            error * 100.0
        );

        assert!(
            error < 0.05,
            "{} should recover within 5%, got {:.2}%",
            param_names[i],
            error * 100.0
        );
    }

    // k1 (primary radial distortion)
    let k1_error = (final_intrinsics[4] - true_intrinsics[4]).abs() / true_intrinsics[4].abs();
    println!(
        "k1: true={:.4}, final={:.4}, error={:.2}%",
        true_intrinsics[4],
        final_intrinsics[4],
        k1_error * 100.0
    );
    assert!(
        k1_error < 0.10,
        "k1 should recover within 10%, got {:.2}%",
        k1_error * 100.0
    );

    // k2 (secondary radial distortion - use percentage error)
    let k2_error = (final_intrinsics[5] - true_intrinsics[5]).abs() / true_intrinsics[5].abs();
    println!(
        "k2: true={:.4}, final={:.4}, error={:.2}%",
        true_intrinsics[5],
        final_intrinsics[5],
        k2_error * 100.0
    );
    assert!(
        k2_error < 0.15,
        "k2 should recover within 15%, got {:.2}%",
        k2_error * 100.0
    );

    // p1, p2 (tangential distortion - use absolute error since values are small)
    let tangential_names = ["p1", "p2"];
    for (idx, i) in [6, 7].iter().enumerate() {
        let abs_error = (final_intrinsics[*i] - true_intrinsics[*i]).abs();
        println!(
            "{}: true={:.4}, final={:.4}, abs_error={:.4}",
            tangential_names[idx], true_intrinsics[*i], final_intrinsics[*i], abs_error
        );

        assert!(
            abs_error < 0.015,
            "{} absolute error should be <0.015, got {:.4}",
            tangential_names[idx],
            abs_error
        );
    }

    // k3 (higher-order term, poorly constrained - use absolute error)
    let k3_abs_error = (final_intrinsics[8] - true_intrinsics[8]).abs();
    println!(
        "k3: true={:.4}, final={:.4}, abs_error={:.4}",
        true_intrinsics[8], final_intrinsics[8], k3_abs_error
    );
    assert!(
        k3_abs_error < 0.3,
        "k3 absolute error should be <0.3 (poorly constrained by planar calibration), got {:.4}",
        k3_abs_error
    );

    println!("\n✓ All RadTan multi-camera calibration tests passed!");

    Ok(())
}

/// Fast 3-camera variant for CI testing
///
/// Uses smaller problem (3 cameras instead of 5) for faster execution
/// while still validating core functionality.
#[test]
fn test_radtan_3_cameras_calibration() -> TestResult {
    // Same setup as main test but with 3 cameras
    let true_camera = RadTanCamera::new(200.0, 200.0, 300.0, 200.0, 0.1, -0.05, 0.001, -0.001, 0.0);

    let img_width = 600.0;
    let img_height = 400.0;

    // Generate calibration target
    let true_landmarks = generate_wall_calibration_points(20, 10, 0.1, 3.0);
    assert_eq!(true_landmarks.len(), 200);

    // Generate 3 camera poses (instead of 5)
    let true_poses = generate_arc_camera_poses(3, 0.8, 3.0);
    assert_eq!(true_poses.len(), 3);

    // Project points
    let mut all_observations: Vec<Vec<Vector2<f64>>> = Vec::new();

    for (cam_idx, pose) in true_poses.iter().enumerate() {
        let mut cam_observations = Vec::with_capacity(true_landmarks.len());

        for (lm_idx, landmark) in true_landmarks.iter().enumerate() {
            let p_cam = pose.act(landmark, None, None);

            assert!(
                true_camera.is_valid_point(&p_cam),
                "Camera {} cannot see landmark {}",
                cam_idx,
                lm_idx
            );

            let uv = true_camera.project(&p_cam).unwrap_or_else(|| {
                panic!(
                    "Projection failed for camera {} landmark {}",
                    cam_idx, lm_idx
                )
            });

            assert!(
                uv.x >= 0.0 && uv.x < img_width && uv.y >= 0.0 && uv.y < img_height,
                "Camera {} landmark {} outside image",
                cam_idx,
                lm_idx
            );

            cam_observations.push(uv);
        }

        all_observations.push(cam_observations);
    }

    // Add noise
    let noisy_landmarks = perturb_landmarks(&true_landmarks, 0.01, 100);
    let noisy_poses: Vec<_> = true_poses
        .iter()
        .enumerate()
        .map(|(i, p)| perturb_pose(p, 0.02, 1.0, 200 + i as u64 * 10))
        .collect();

    let true_intrinsics = [200.0, 200.0, 300.0, 200.0, 0.1, -0.05, 0.001, -0.001, 0.0];
    let noisy_intrinsics = perturb_intrinsics(&true_intrinsics, 0.02, 300);

    // Build problem
    let mut problem = Problem::new();

    for (cam_idx, observations) in all_observations.iter().enumerate() {
        let obs_matrix = Matrix2xX::from_columns(observations);
        let factor: ProjectionFactor<RadTanCamera, SelfCalibration> =
            ProjectionFactor::new(obs_matrix, true_camera);

        let pose_name = format!("pose_{}", cam_idx);
        problem.add_residual_block(
            &[&pose_name, "landmarks", "intrinsics"],
            Box::new(factor),
            None,
        );
    }

    // Fix first camera
    for dof in 0..6 {
        problem.fix_variable("pose_0", dof);
    }

    // Initialize variables
    let mut initial_values = HashMap::new();

    for (i, pose) in noisy_poses.iter().enumerate() {
        initial_values.insert(
            format!("pose_{}", i),
            (ManifoldType::SE3, pose.clone().into()),
        );
    }

    initial_values.insert(
        "landmarks".to_string(),
        (ManifoldType::RN, flatten_landmarks(&noisy_landmarks)),
    );
    initial_values.insert(
        "intrinsics".to_string(),
        (ManifoldType::RN, DVector::from_vec(noisy_intrinsics)),
    );

    // Optimize
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-8)
        .with_parameter_tolerance(1e-8)
        .with_gradient_tolerance(1e-10)
        .with_damping(1e-3);

    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&problem, &initial_values)?;

    // Verify convergence
    assert!(
        matches!(
            result.status,
            OptimizationStatus::Converged
                | OptimizationStatus::CostToleranceReached
                | OptimizationStatus::ParameterToleranceReached
        ),
        "3-camera test should converge"
    );

    // Verify cost reduction
    let cost_reduction = (result.initial_cost - result.final_cost) / result.initial_cost;
    assert!(
        cost_reduction > 0.70,
        "3-camera test should achieve >70% cost reduction, got {:.2}%",
        cost_reduction * 100.0
    );

    println!("✓ RadTan 3-camera calibration test passed!");

    Ok(())
}
