//! Schur-vs-Cholesky agreement on real bundle-adjustment data.
//!
//! `schur_optimizers.rs` proves optimizer-level agreement on a synthetic
//! chain; this test proves it on real BAL data (ladybug, first 8 cameras).
//! All three linear paths must land on the same final cost — after the
//! assembly row-order fix, a misaligned sparse Jacobian can no longer make
//! the Schur paths diverge from Cholesky.

use apex_solver::JacobianMode;
use apex_solver::apex_camera_models::{BALPinholeCameraStrict, DistortionModel, PinholeParams};
use apex_solver::apex_io::BalLoader;
use apex_solver::apex_manifolds::se3::SE3;
use apex_solver::apex_manifolds::so3::SO3;
use apex_solver::apex_manifolds::{LieGroup, ManifoldType};
use apex_solver::core::VarKey;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BundleAdjustment, ProjectionFactor};
use apex_solver::linalg::{LinearSolverType, SchurVariant};
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DVector, Matrix2xX, Vector2, Vector3};
use std::collections::HashMap;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn axis_angle_to_so3(axis_angle: &Vector3<f64>) -> SO3 {
    let angle = axis_angle.norm();
    if angle < 1e-10 {
        SO3::identity()
    } else {
        let axis = axis_angle / angle;
        SO3::from_axis_angle(&axis, angle)
    }
}

/// First 8 cameras of ladybug + all points they observe (~6k observations).
fn build_mini_ba() -> Result<(Problem, usize), Box<dyn std::error::Error>> {
    apex_solver::apex_io::ensure_ba_dataset("ladybug", 1723, 156502)?;
    let dataset = BalLoader::load("data/bundle_adjustment/ladybug/problem-1723-156502-pre.txt")?;

    let n_cams = 8;
    let obs: Vec<_> = dataset
        .observations
        .iter()
        .filter(|o| o.camera_index < n_cams)
        .collect();
    let mut used_pts: Vec<usize> = obs.iter().map(|o| o.point_index).collect();
    used_pts.sort_unstable();
    used_pts.dedup();

    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut pose_keys = Vec::new();
    for cam in dataset.cameras.iter().take(n_cams) {
        let so3 = axis_angle_to_so3(&Vector3::new(
            cam.rotation.x,
            cam.rotation.y,
            cam.rotation.z,
        ));
        let pose = SE3::from_translation_so3(
            Vector3::new(cam.translation.x, cam.translation.y, cam.translation.z),
            so3,
        );
        pose_keys.push(problem.add_variable(
            ManifoldType::SE3,
            DVector::from_column_slice(pose.as_param_slice()),
        ));
    }
    let mut pt_map: HashMap<usize, VarKey> = HashMap::new();
    for old in used_pts {
        let point = &dataset.points[old];
        let key = problem.add_variable(
            ManifoldType::RN,
            DVector::from_vec(vec![point.position.x, point.position.y, point.position.z]),
        );
        problem.mark_for_elimination(key);
        pt_map.insert(old, key);
    }
    for o in &obs {
        let cam = &dataset.cameras[o.camera_index];
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
        let observations = Matrix2xX::from_columns(&[Vector2::new(o.x, o.y)]);
        let factor: ProjectionFactor<BALPinholeCameraStrict, BundleAdjustment> =
            ProjectionFactor::new(observations, camera);
        problem.add_residual_block(
            &[pose_keys[o.camera_index], pt_map[&o.point_index]],
            Box::new(factor),
            None,
        );
    }
    for dof in 0..6 {
        problem.fix_variable(pose_keys[0], dof);
    }
    Ok((problem, obs.len()))
}

#[test]
fn schur_variants_agree_with_cholesky_on_ladybug8() -> TestResult {
    let mut finals = Vec::new();
    for (solver_type, variant) in [
        (LinearSolverType::SparseCholesky, SchurVariant::Sparse),
        (
            LinearSolverType::SparseSchurComplement,
            SchurVariant::Sparse,
        ),
        (
            LinearSolverType::SparseSchurComplement,
            SchurVariant::ChunkedSparse,
        ),
    ] {
        let (mut problem, _) = build_mini_ba()?;
        let config = LevenbergMarquardtConfig::new()
            .with_max_iterations(20)
            .with_linear_solver_type(solver_type)
            .with_schur_variant(variant);
        let mut solver = LevenbergMarquardt::with_config(config);
        let result = solver.optimize(&mut problem)?;
        assert!(
            result.final_cost < result.initial_cost,
            "cost must improve (initial {:.6e}, final {:.6e})",
            result.initial_cost,
            result.final_cost
        );
        finals.push(result.final_cost);
    }
    for (i, cost) in finals.iter().enumerate().skip(1) {
        let scale = finals[0].abs().max(1.0);
        assert!(
            (finals[0] - cost).abs() / scale < 1e-6,
            "Schur variant {i} final cost {cost:.6e} disagrees with Cholesky {:.6e}",
            finals[0]
        );
    }
    Ok(())
}
