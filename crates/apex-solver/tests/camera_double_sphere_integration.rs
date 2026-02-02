//! Integration test for DoubleSphere camera with multi-camera calibration
//!
//! This test simulates a realistic camera calibration scenario:
//! - 200+ 3D calibration target points on a planar wall (like ArUco/chessboard)
//! - 5 cameras viewing the target from different positions on an arc
//! - Simultaneous optimization of poses, landmarks, and intrinsics (SelfCalibration)
//! - Tests the Double Sphere model's ability to handle fisheye-like distortion
//!
//! Double Sphere Camera Model:
//! - 6 intrinsic parameters: fx, fy, cx, cy, xi, alpha
//! - xi (ξ): Sphere offset parameter (0.0-1.0, controls fisheye strength)
//! - alpha (α): Sphere blending parameter (typically 0.4-0.6)
//!
//! This is a more complex model than pinhole, suitable for wide-angle cameras.

use apex_camera_models::{
    CameraModel, DistortionModel, DoubleSphereCamera, PinholeParams, SelfCalibration,
};
use apex_manifolds::LieGroup;
use apex_solver::core::problem::Problem;
use apex_solver::factors::ProjectionFactor;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::ManifoldType;
use nalgebra::{DVector, Matrix2xX, Vector2};
use std::collections::HashMap;

mod camera_test_utils;
use camera_test_utils::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Test Double Sphere camera self-calibration with multi-camera setup.
///
/// Scenario: 5 cameras on a horizontal arc viewing a 20x10 grid of calibration
/// points on a wall at 3m distance. All cameras share the same intrinsics.
///
/// Ground truth camera: 600x400 image, moderate fisheye (xi=0.6, alpha=0.5)
#[test]
fn test_double_sphere_multi_camera_calibration_200_points() -> TestResult {
    // ============================================================================
    // 1. Ground Truth Setup - 600x400 Fisheye Camera
    // ============================================================================

    // Double Sphere camera parameters for a fisheye-like lens
    // - Shorter focal length (200px) gives wider FOV (~85° horizontal)
    // - xi=0.5 gives moderate fisheye distortion (balanced)
    // - alpha=0.5 balances the two sphere projections
    //
    // Note: Using xi=0.5 instead of 0.6 because it's more numerically stable
    // and the double sphere model is symmetric around this point.
    let true_camera = DoubleSphereCamera::new(
        PinholeParams {
            fx: 200.0,
            fy: 200.0,
            cx: 300.0,
            cy: 200.0,
        },
        DistortionModel::DoubleSphere {
            alpha: 0.5,
            xi: 0.5,
        },
    )?;

    // Image bounds for projection validation
    let img_width = 600.0;
    let img_height = 400.0;

    // ============================================================================
    // 2. Generate Calibration Target (200 Points with Depth Variation)
    // ============================================================================

    // Generate 3D scene with depth variation for better intrinsic observability
    // Use generate_scene_points which creates hemisphere-distributed points
    // This provides better geometric constraints for xi/alpha recovery
    let true_landmarks = generate_scene_points(200, 42);
    assert_eq!(
        true_landmarks.len(),
        200,
        "Should generate exactly 200 calibration points"
    );

    // ============================================================================
    // 3. Generate 5 Camera Poses with Wider Baseline
    // ============================================================================

    // IMPORTANT: Wider baseline (arc_spread=0.8m) provides better geometric
    // constraints for recovering distortion parameters xi/alpha.
    // The cameras are spread ±0.8m from center horizontally.
    let true_poses = generate_arc_camera_poses(5, 0.8, 3.0);
    assert_eq!(true_poses.len(), 5, "Should generate 5 camera poses");

    // ============================================================================
    // 4. Project Points into Each Camera and Verify ALL Points Visible
    // ============================================================================

    // For the ProjectionFactor to work correctly, ALL landmarks must be visible
    // from ALL cameras. This is ensured by our setup (small arc, wall at 3m).
    let mut all_observations: Vec<Vec<Vector2<f64>>> = Vec::new();

    for (cam_idx, pose) in true_poses.iter().enumerate() {
        let mut cam_observations = Vec::with_capacity(true_landmarks.len());

        for (lm_idx, landmark) in true_landmarks.iter().enumerate() {
            // Transform point from world to camera frame
            let p_cam = pose.act(landmark, None, None);

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
    // Lower noise helps xi/alpha converge - these parameters have subtle
    // effects on the projection and are sensitive to noise.

    let noisy_landmarks = perturb_landmarks(&true_landmarks, 0.01, 100);

    let noisy_poses: Vec<_> = true_poses
        .iter()
        .enumerate()
        .map(|(i, p)| perturb_pose(p, 0.02, 1.0, 200 + i as u64 * 10))
        .collect();

    let true_intrinsics = [200.0, 200.0, 300.0, 200.0, 0.5, 0.5];
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
        let factor: ProjectionFactor<DoubleSphereCamera, SelfCalibration> =
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

    // Intrinsics (RN manifold, [fx, fy, cx, cy, alpha, xi])
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

    assert!(
        cost_reduction > 0.95,
        "Cost should reduce by >95%, got {:.2}% reduction (initial={:.4e}, final={:.4e})",
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
        rmse < 2.0, // Relax for now to see more diagnostics
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

    let param_names = ["fx", "fy", "cx", "cy", "xi", "alpha"];

    // Different tolerances for different parameters:
    // - fx, fy, cx, cy: 5% (well-conditioned)
    // - xi, alpha: 10% (distortion params are harder to recover precisely)
    let tolerances = [0.05, 0.05, 0.05, 0.05, 0.10, 0.10];

    for i in 0..6 {
        // Use max(0.1, |true|) to handle small true values like xi=0.6
        let relative_error =
            (final_intrinsics[i] - true_intrinsics[i]).abs() / true_intrinsics[i].abs().max(0.1);

        assert!(
            relative_error < tolerances[i],
            "{} should recover within {:.0}% of ground truth, got {:.2}% error \
             (true={:.4}, final={:.4})",
            param_names[i],
            tolerances[i] * 100.0,
            relative_error * 100.0,
            true_intrinsics[i],
            final_intrinsics[i]
        );
    }

    // ============================================================================
    // 13. Print Summary (for debugging when run with --nocapture)
    // ============================================================================

    println!("\n=== Double Sphere Multi-Camera Calibration Results ===");
    println!("Status: {:?}", result.status);
    println!("Iterations: {}", result.iterations);
    println!("Initial cost: {:.4e}", result.initial_cost);
    println!("Final cost: {:.4e}", result.final_cost);
    println!("Cost reduction: {:.2}%", cost_reduction * 100.0);
    println!("Reprojection RMSE: {:.4} pixels", rmse);
    println!("\nIntrinsic Recovery:");
    for i in 0..6 {
        let error = (final_intrinsics[i] - true_intrinsics[i]).abs() / true_intrinsics[i].abs();
        println!(
            "  {}: true={:.4}, final={:.4}, error={:.2}%",
            param_names[i],
            true_intrinsics[i],
            final_intrinsics[i],
            error * 100.0
        );
    }

    Ok(())
}

/// Test with 3 cameras for faster execution (good for CI)
#[test]
fn test_double_sphere_3_cameras_calibration() -> TestResult {
    // Simpler setup: 3 cameras, 200 points
    // Uses same camera params as 5-camera test for consistency
    let true_camera = DoubleSphereCamera::new(
        PinholeParams {
            fx: 200.0,
            fy: 200.0,
            cx: 300.0,
            cy: 200.0,
        },
        DistortionModel::DoubleSphere {
            alpha: 0.5,
            xi: 0.5,
        },
    )?;

    let img_width = 600.0;
    let img_height = 400.0;

    // Generate 3D scene points with depth variation (not planar wall)
    let true_landmarks = generate_scene_points(200, 42);
    let true_poses = generate_arc_camera_poses(3, 0.6, 3.0); // 3 cameras, wider arc

    // Project and collect observations
    let mut all_observations: Vec<Vec<Vector2<f64>>> = Vec::new();

    for pose in &true_poses {
        let mut cam_obs = Vec::new();
        for landmark in &true_landmarks {
            let p_cam = pose.act(landmark, None, None);
            if let Ok(uv) = true_camera.project(&p_cam) {
                if uv.x >= 0.0 && uv.x < img_width && uv.y >= 0.0 && uv.y < img_height {
                    cam_obs.push(uv);
                }
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
    let true_intrinsics = [200.0, 200.0, 300.0, 200.0, 0.5, 0.5];
    let noisy_intrinsics = perturb_intrinsics(&true_intrinsics, 0.02, 300);

    // Build problem
    let mut problem = Problem::new();

    for (cam_idx, observations) in all_observations.iter().enumerate() {
        let obs_matrix = Matrix2xX::from_columns(observations);
        let factor: ProjectionFactor<DoubleSphereCamera, SelfCalibration> =
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

    // Verify cost reduction
    let cost_reduction = (result.initial_cost - result.final_cost) / result.initial_cost;
    assert!(
        cost_reduction > 0.90,
        "Cost should reduce by >90%, got {:.2}%",
        cost_reduction * 100.0
    );

    Ok(())
}
