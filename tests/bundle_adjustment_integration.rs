//! Integration test for bundle adjustment on a real BAL dataset.
//!
//! Loads the Ladybug problem-21-11315-pre dataset (21 cameras, 11315 points)
//! and runs self-calibration optimization to verify convergence, RMSE reduction,
//! and monitor execution time.

use apex_solver::apex_camera_models::{BALPinholeCameraStrict, DistortionModel, PinholeParams};
use apex_solver::apex_io::BalLoader;
use apex_solver::apex_manifolds::ManifoldType;
use apex_solver::apex_manifolds::se3::SE3;
use apex_solver::apex_manifolds::so3::SO3;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{ProjectionFactor, SelfCalibration};
use apex_solver::JacobianMode;
use apex_solver::linalg::SchurVariant;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DVector, Matrix2xX, Vector2, Vector3};
use std::collections::HashMap;

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
fn test_ladybug_21_self_calibration() -> Result<(), Box<dyn std::error::Error>> {
    // Load BAL dataset
    let dataset = BalLoader::load("data/bundle_adjustment/Ladybug/problem-21-11315-pre.txt")?;

    let num_observations = dataset.observations.len();

    // Build problem
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut initial_values: HashMap<String, (ManifoldType, DVector<f64>)> = HashMap::new();

    // Add camera poses (SE3) and intrinsics (RN)
    for (i, cam) in dataset.cameras.iter().enumerate() {
        let axis_angle = Vector3::new(cam.rotation.x, cam.rotation.y, cam.rotation.z);
        let translation = Vector3::new(cam.translation.x, cam.translation.y, cam.translation.z);
        let so3 = axis_angle_to_so3(&axis_angle);
        let pose = SE3::from_translation_so3(translation, so3);

        let pose_name = format!("pose_{:04}", i);
        initial_values.insert(pose_name, (ManifoldType::SE3, DVector::from(pose)));

        let intrinsics_name = format!("intr_{:04}", i);
        let intrinsics_vec = DVector::from_vec(vec![cam.focal_length, cam.k1, cam.k2]);
        initial_values.insert(intrinsics_name, (ManifoldType::RN, intrinsics_vec));
    }

    // Add landmarks (RN3)
    for (j, point) in dataset.points.iter().enumerate() {
        let var_name = format!("pt_{:05}", j);
        let point_vec =
            DVector::from_vec(vec![point.position.x, point.position.y, point.position.z]);
        initial_values.insert(var_name, (ManifoldType::RN, point_vec));
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

        let pose_name = format!("pose_{:04}", obs.camera_index);
        let pt_name = format!("pt_{:05}", obs.point_index);
        let intr_name = format!("intr_{:04}", obs.camera_index);

        let loss = HuberLoss::new(1.0)?;
        problem.add_residual_block(
            &[&pose_name, &pt_name, &intr_name],
            Box::new(factor),
            Some(Box::new(loss)),
        );
    }

    // Fix first camera pose for gauge freedom
    for dof in 0..6 {
        problem.fix_variable("pose_0000", dof);
    }

    // Configure solver (explicit Schur for small problem, more iterations for convergence)
    let mut config = LevenbergMarquardtConfig::for_bundle_adjustment();
    config.schur_variant = SchurVariant::Sparse;
    config = config.with_max_iterations(50);

    let mut solver = LevenbergMarquardt::with_config(config);

    // Optimize
    let result = solver.optimize(&problem, &initial_values)?;

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
