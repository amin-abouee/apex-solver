//! Integration test for KannalaBrandt camera with multi-camera calibration
//!
//! Simulates a realistic camera calibration scenario:
//! - 200+ 3D scene points with depth variation (hemisphere distribution)
//! - 5 cameras viewing the scene from different positions
//! - Simultaneous optimization of poses, landmarks, and intrinsics
//! - Tests Kannala-Brandt model's ability to handle fisheye distortion
//!
//! Kannala-Brandt Camera Model:
//! - 8 intrinsic parameters: fx, fy, cx, cy, k1, k2, k3, k4
//! - Polynomial angle-based distortion model
//! - Popular for wide-angle fisheye cameras (OpenCV fisheye model)

use apex_camera_models::{
    CameraModel, DistortionModel, KannalaBrandtCamera, PinholeParams, SelfCalibration,
};
use apex_manifolds::LieGroup;
use apex_solver::ManifoldType;
use apex_solver::core::problem::Problem;
use apex_solver::factors::ProjectionFactor;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DVector, Matrix2xX, Vector2};
use std::collections::HashMap;

mod camera_test_utils;
use camera_test_utils::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Test Kannala-Brandt camera self-calibration with multi-camera setup.
///
/// Scenario: 5 cameras on a horizontal arc viewing 200 3D points distributed
/// on a hemisphere. All cameras share the same intrinsics.
///
/// Ground truth camera: 600x400 image, fisheye with polynomial distortion
#[test]
fn test_kannala_brandt_multi_camera_calibration_200_points() -> TestResult {
    // ============================================================================
    // 1. Ground Truth Setup - 600x400 Fisheye Camera
    // ============================================================================

    // Kannala-Brandt camera parameters for fisheye lens
    // - Shorter focal length (200px) gives wider FOV (~85° horizontal)
    // - k1=0.5: primary radial distortion coefficient
    // - k2=0.1: secondary distortion
    // - k3, k4=0: higher-order terms set to zero (not observable from planar target)
    //
    // Note: Real calibration often finds k3, k4 close to zero or poorly constrained
    let true_camera = KannalaBrandtCamera::new(
        PinholeParams {
            fx: 200.0,
            fy: 200.0,
            cx: 300.0,
            cy: 200.0,
        },
        DistortionModel::KannalaBrandt {
            k1: 0.5,
            k2: 0.1,
            k3: 0.0,
            k4: 0.0,
        },
    )?;

    // Image bounds for projection validation
    let img_width = 600.0;
    let img_height = 400.0;

    // ============================================================================
    // 2. Generate Calibration Target (200 Points on Wall at Z=3m)
    // ============================================================================

    // For Kannala-Brandt fisheye, use planar wall to ensure all points visible
    // Wall at fixed depth ensures points stay within FOV
    let true_landmarks = generate_wall_calibration_points(20, 10, 0.1, 3.0);
    assert_eq!(
        true_landmarks.len(),
        200,
        "Should generate exactly 200 calibration points"
    );

    // ============================================================================
    // 3. Generate 5 Camera Poses with Wider Baseline
    // ============================================================================

    // IMPORTANT: Wider baseline (arc_spread=0.8m) provides better geometric
    // constraints for recovering distortion parameters k1-k4.
    let true_poses = generate_arc_camera_poses(5, 0.8, 3.0);
    assert_eq!(true_poses.len(), 5, "Should generate 5 camera poses");

    // ============================================================================
    // 4. Project Points and Verify ALL Points Visible
    // ============================================================================

    // For ProjectionFactor to work correctly, ALL landmarks must be visible
    // from ALL cameras.
    let mut all_observations: Vec<Vec<Vector2<f64>>> = Vec::new();

    for (cam_idx, pose) in true_poses.iter().enumerate() {
        let mut cam_observations = Vec::with_capacity(true_landmarks.len());

        for (lm_idx, landmark) in true_landmarks.iter().enumerate() {
            // Transform point from world to camera frame
            let p_cam = pose.act(landmark, None, None);

            // Verify point is valid for projection
            assert!(
                true_camera.project(&p_cam).is_ok(),
                "Camera {} cannot see landmark {}: p_cam = {:?}",
                cam_idx,
                lm_idx,
                p_cam
            );

            // Project to image coordinates
            let uv = true_camera.project(&p_cam)?;

            // Verify within image bounds
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
    // 5. Add Noise to Create Initial Estimates (Simulate Real Calibration)
    // ============================================================================

    // Noise levels for camera calibration:
    // - Landmark noise: 1cm (tight 3D reconstruction)
    // - Pose translation: 2cm (small positioning error)
    // - Pose rotation: 1.0° (small angular error)
    // - Intrinsic parameters: 2% relative error
    //
    // These values worked well for Double Sphere test.

    let noisy_landmarks = perturb_landmarks(&true_landmarks, 0.01, 100);

    let noisy_poses: Vec<_> = true_poses
        .iter()
        .enumerate()
        .map(|(i, p)| perturb_pose(p, 0.02, 1.0, 200 + i as u64 * 10))
        .collect();

    let true_intrinsics = [200.0, 200.0, 300.0, 200.0, 0.5, 0.1, 0.0, 0.0];
    let noisy_intrinsics = perturb_intrinsics(&true_intrinsics, 0.02, 300);

    // ============================================================================
    // 6. Build Optimization Problem
    // ============================================================================

    let mut problem = Problem::new();

    // Add one projection factor per camera
    // Each factor observes ALL landmarks from its viewpoint
    for (cam_idx, observations) in all_observations.iter().enumerate() {
        let obs_matrix = Matrix2xX::from_columns(observations);

        // SelfCalibration optimizes: pose + landmarks + intrinsics
        let factor: ProjectionFactor<KannalaBrandtCamera, SelfCalibration> =
            ProjectionFactor::new(obs_matrix, true_camera);

        let pose_name = format!("pose_{}", cam_idx);

        // Variables order: [pose, landmarks, intrinsics]
        problem.add_residual_block(
            &[&pose_name, "landmarks", "intrinsics"],
            Box::new(factor),
            None, // No robust loss for clean synthetic data
        );
    }

    // Fix first camera pose for gauge freedom (anchor the coordinate system)
    // This prevents the solution from drifting in SE3 space
    for dof in 0..6 {
        problem.fix_variable("pose_0", dof);
    }

    // ============================================================================
    // 7. Initialize Variables with Noisy Values
    // ============================================================================

    let mut initial_values = HashMap::new();

    // Camera poses (SE3 manifold)
    for (i, pose) in noisy_poses.iter().enumerate() {
        initial_values.insert(
            format!("pose_{}", i),
            (ManifoldType::SE3, pose.clone().into()),
        );
    }

    // Landmarks (RN manifold, flattened [x0, y0, z0, x1, y1, z1, ...])
    initial_values.insert(
        "landmarks".to_string(),
        (ManifoldType::RN, flatten_landmarks(&noisy_landmarks)),
    );

    // Intrinsics (RN manifold, [fx, fy, cx, cy, k1, k2, k3, k4])
    initial_values.insert(
        "intrinsics".to_string(),
        (
            ManifoldType::RN,
            DVector::from_vec(noisy_intrinsics.clone()),
        ),
    );

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
    let result = solver.optimize(&problem, &initial_values)?;

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

    // Kannala-Brandt with planar target: distortion params harder to constrain
    // Relax requirement to 85% (vs 95% for Double Sphere with 3D scene)
    assert!(
        cost_reduction > 0.85,
        "Cost should reduce by >85%, got {:.2}% reduction (initial={:.4e}, final={:.4e})",
        cost_reduction * 100.0,
        result.initial_cost,
        result.final_cost
    );

    // ============================================================================
    // 11. Verify Reprojection RMSE
    // ============================================================================

    let total_observations: usize = all_observations.iter().map(|o| o.len()).sum();
    let rmse = (result.final_cost / total_observations as f64).sqrt();

    // Print diagnostic info
    println!("\n=== Optimization Results ===");
    println!("Status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Initial cost: {:.4e}", result.initial_cost);
    println!("Final cost: {:.4e}", result.final_cost);
    println!("Cost reduction: {:.2}%", cost_reduction * 100.0);
    println!("Total observations: {}", total_observations);
    println!("Reprojection RMSE: {:.4} pixels", rmse);

    assert!(
        rmse < 2.0,
        "Reprojection RMSE should be < 2 pixels, got {:.4} pixels",
        rmse
    );

    // ============================================================================
    // 12. Verify Intrinsic Parameter Recovery
    // ============================================================================

    let final_intrinsics = result
        .parameters
        .get("intrinsics")
        .ok_or("Missing intrinsics in result")?
        .to_vector();

    let param_names = ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"];

    // Different tolerances for different parameters:
    // - fx, fy, cx, cy: 5% (well-conditioned)
    // - k1: 10% (primary distortion, reasonably well-constrained)
    // - k2, k3, k4: 20% (higher-order terms harder to recover from planar target)
    let tolerances = [0.05, 0.05, 0.05, 0.05, 0.10, 0.20, 0.20, 0.20];

    println!("\nIntrinsic Recovery:");
    for i in 0..8 {
        // For k3 and k4 (indices 6, 7), use absolute error since they're often near zero
        // For other params, use relative error
        let (error, error_type) = if i >= 6 {
            // Absolute error for k3, k4 (max 0.25 absolute deviation acceptable)
            // These higher-order terms are NOT well-constrained by planar calibration
            let abs_error = (final_intrinsics[i] - true_intrinsics[i]).abs();
            (abs_error, "abs")
        } else {
            // Relative error for fx, fy, cx, cy, k1, k2
            let rel_error = (final_intrinsics[i] - true_intrinsics[i]).abs()
                / true_intrinsics[i].abs().max(0.01);
            (rel_error, "rel")
        };

        if error_type == "abs" {
            println!(
                "  {}: true={:.4}, final={:.4}, error={:.4} (absolute)",
                param_names[i], true_intrinsics[i], final_intrinsics[i], error
            );
            assert!(
                error < 0.25,
                "{} absolute error should be < 0.25, got {:.4} (true={:.4}, final={:.4})",
                param_names[i],
                error,
                true_intrinsics[i],
                final_intrinsics[i]
            );
        } else {
            println!(
                "  {}: true={:.4}, final={:.4}, error={:.2}%",
                param_names[i],
                true_intrinsics[i],
                final_intrinsics[i],
                error * 100.0
            );
            assert!(
                error < tolerances[i],
                "{} should recover within {:.0}% of ground truth, got {:.2}% error \
                 (true={:.4}, final={:.4})",
                param_names[i],
                tolerances[i] * 100.0,
                error * 100.0,
                true_intrinsics[i],
                final_intrinsics[i]
            );
        }
    }

    // ============================================================================
    // 13. Print Summary (for debugging when run with --nocapture)
    // ============================================================================

    println!("\n=== Kannala-Brandt Multi-Camera Calibration Results ===");
    println!("Status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Initial cost: {:.4e}", result.initial_cost);
    println!("Final cost: {:.4e}", result.final_cost);
    println!("Cost reduction: {:.2}%", cost_reduction * 100.0);
    println!("Reprojection RMSE: {:.4} pixels", rmse);

    Ok(())
}

/// Test with 3 cameras for faster execution (good for CI)
#[test]
fn test_kannala_brandt_3_cameras_calibration() -> TestResult {
    // Simpler setup: 3 cameras, 200 points
    // Uses same camera params as 5-camera test for consistency
    let true_camera = KannalaBrandtCamera::new(
        PinholeParams {
            fx: 200.0,
            fy: 200.0,
            cx: 300.0,
            cy: 200.0,
        },
        DistortionModel::KannalaBrandt {
            k1: 0.5,
            k2: 0.1,
            k3: 0.0,
            k4: 0.0,
        },
    )?;

    let img_width = 600.0;
    let img_height = 400.0;

    // Generate calibration points on wall (not hemisphere for Kannala-Brandt)
    let true_landmarks = generate_wall_calibration_points(20, 10, 0.1, 3.0);
    let true_poses = generate_arc_camera_poses(3, 0.6, 3.0); // 3 cameras, wider arc

    // Project and collect observations
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

    // Add noise (same levels as 5-camera test)
    let noisy_landmarks = perturb_landmarks(&true_landmarks, 0.01, 100);
    let noisy_poses: Vec<_> = true_poses
        .iter()
        .enumerate()
        .map(|(i, p)| perturb_pose(p, 0.02, 1.0, 200 + i as u64 * 10))
        .collect();
    let true_intrinsics = [200.0, 200.0, 300.0, 200.0, 0.5, 0.1, 0.0, 0.0];
    let noisy_intrinsics = perturb_intrinsics(&true_intrinsics, 0.02, 300);

    // Build problem
    let mut problem = Problem::new();

    for (cam_idx, observations) in all_observations.iter().enumerate() {
        let obs_matrix = Matrix2xX::from_columns(observations);
        let factor: ProjectionFactor<KannalaBrandtCamera, SelfCalibration> =
            ProjectionFactor::new(obs_matrix, true_camera);

        problem.add_residual_block(
            &[&format!("pose_{}", cam_idx), "landmarks", "intrinsics"],
            Box::new(factor),
            None,
        );
    }

    // Fix first pose
    for dof in 0..6 {
        problem.fix_variable("pose_0", dof);
    }

    // Initialize
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
                | OptimizationStatus::GradientToleranceReached
        ),
        "3-camera calibration should converge, got: {:?}",
        result.status
    );

    // Verify cost reduction (relaxed for planar scene)
    let cost_reduction = (result.initial_cost - result.final_cost) / result.initial_cost;
    assert!(
        cost_reduction > 0.60,
        "Cost should reduce by >60%, got {:.2}%",
        cost_reduction * 100.0
    );

    Ok(())
}
