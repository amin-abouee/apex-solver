//! Integration test for bundle adjustment on a real BAL dataset.
//!
//! Loads the Trafalgar problem-21-11315-pre dataset (21 cameras, 11315 points)
//! and runs self-calibration optimization to verify convergence, RMSE reduction,
//! and monitor execution time.

use apex_solver::JacobianMode;
use apex_solver::apex_camera_models::{BALPinholeCameraStrict, DistortionModel, PinholeParams};
use apex_solver::apex_io::BalLoader;
use apex_solver::apex_manifolds::se3::SE3;
use apex_solver::apex_manifolds::so3::SO3;
use apex_solver::apex_manifolds::{LieGroup, ManifoldType};
use apex_solver::core::VarKey;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{ProjectionFactor, SelfCalibration};
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DVector, Matrix2xX, Vector2, Vector3};

/// Convert axis-angle rotation vector to SO3.
fn axis_angle_to_so3(axis_angle: &Vector3<f64>) -> SO3 {
    let angle = axis_angle.norm();
    if angle < 1e-10 {
        SO3::identity()
    } else {
        let axis = axis_angle / angle;
        SO3::from_axis_angle(&axis, angle)
    }
}

#[test]
fn test_trafalgar_21_self_calibration() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure the dataset is present, downloading it if necessary.
    apex_solver::apex_io::ensure_ba_dataset("trafalgar", 21, 11315)?;

    // Load BAL dataset
    let dataset = BalLoader::load("data/bundle_adjustment/trafalgar/problem-21-11315-pre.txt")?;

    let num_observations = dataset.observations.len();

    // Build problem
    let mut problem = Problem::new(JacobianMode::Sparse);

    // Track VarKeys by index
    let mut pose_keys: Vec<VarKey> = Vec::with_capacity(dataset.cameras.len());
    let mut intr_keys: Vec<VarKey> = Vec::with_capacity(dataset.cameras.len());
    let mut pt_keys: Vec<VarKey> = Vec::with_capacity(dataset.points.len());

    // Add camera poses (SE3) and intrinsics (RN)
    for (i, cam) in dataset.cameras.iter().enumerate() {
        let axis_angle = Vector3::new(cam.rotation.x, cam.rotation.y, cam.rotation.z);
        let translation = Vector3::new(cam.translation.x, cam.translation.y, cam.translation.z);
        let so3 = axis_angle_to_so3(&axis_angle);
        let pose = SE3::from_translation_so3(translation, so3);

        let pose_key = problem.add_variable(
            ManifoldType::SE3,
            DVector::from_column_slice(pose.as_param_slice()),
        );
        pose_keys.push(pose_key);

        let intrinsics_vec = DVector::from_vec(vec![cam.focal_length, cam.k1, cam.k2]);
        let intr_key = problem.add_variable(ManifoldType::RN, intrinsics_vec);
        intr_keys.push(intr_key);
        let _ = i;
    }

    // Add landmarks (RN3)
    for point in &dataset.points {
        let point_vec =
            DVector::from_vec(vec![point.position.x, point.position.y, point.position.z]);
        let pt_key = problem.add_variable(ManifoldType::RN, point_vec);
        problem.mark_as_schur_landmark(pt_key);
        pt_keys.push(pt_key);
    }

    // Add projection factors (self-calibration: pose + landmark + intrinsics)
    for obs in &dataset.observations {
        let cam = &dataset.cameras[obs.camera_index];
        let camera = BALPinholeCameraStrict::new(
            PinholeParams {
                fx: cam.focal_length,
                fy: cam.focal_length,
                cx: 0.0,
                cy: 0.0,
            },
            DistortionModel::Radial {
                k1: cam.k1,
                k2: cam.k2,
            },
        )?;

        let observations = Matrix2xX::from_columns(&[Vector2::new(obs.x, obs.y)]);
        let factor: ProjectionFactor<BALPinholeCameraStrict, SelfCalibration> =
            ProjectionFactor::new(observations, camera);

        let pose_key = pose_keys[obs.camera_index];
        let pt_key = pt_keys[obs.point_index];
        let intr_key = intr_keys[obs.camera_index];

        let loss = HuberLoss::new(1.0)?;
        problem.add_residual_block(
            &[pose_key, pt_key, intr_key],
            Box::new(factor),
            Some(Box::new(loss)),
        );
    }

    // Fix first camera pose for gauge freedom
    let first_pose_key = pose_keys[0];
    for dof in 0..6 {
        problem.fix_variable(first_pose_key, dof);
    }

    // Iterative Schur is required for SelfCalibration because intrinsics (RN, 3 DOF) and
    // landmarks (RN, 3 DOF) are indistinguishable for the Sparse Schur block classifier.
    let config = LevenbergMarquardtConfig::for_bundle_adjustment().with_max_iterations(50);

    let mut solver = LevenbergMarquardt::with_config(config);

    // Optimize
    let result = solver.optimize(&mut problem)?;

    // Compute RMSE
    let num_obs = num_observations as f64;
    let initial_rmse = (result.initial_cost / num_obs).sqrt();
    let final_rmse = (result.final_cost / num_obs).sqrt();

    // Assert convergence
    let converged = matches!(
        result.status,
        OptimizationStatus::Converged
            | OptimizationStatus::CostToleranceReached
            | OptimizationStatus::ParameterToleranceReached
            | OptimizationStatus::GradientToleranceReached
    );
    assert!(
        converged,
        "Solver did not converge. Status: {:?}",
        result.status
    );

    // Assert optimization improved the cost
    assert!(
        result.final_cost < result.initial_cost,
        "Final cost ({:.6e}) should be less than initial cost ({:.6e})",
        result.final_cost,
        result.initial_cost
    );

    // Assert RMSE decreased
    assert!(
        final_rmse < initial_rmse,
        "Final RMSE ({:.4}) should be less than initial RMSE ({:.4})",
        final_rmse,
        initial_rmse
    );

    Ok(())
}
