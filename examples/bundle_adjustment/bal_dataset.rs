//! # 9 · A real bundle-adjustment dataset (BAL)
//!
//! The [Bundle Adjustment in the Large](https://grail.cs.washington.edu/projects/bal/)
//! collection is the standard BA benchmark. This example loads its smallest
//! Ladybug problem (49 cameras, 7 776 points, 31 843 observations; downloaded
//! on first use) and solves it the way the BAL authors define it:
//!
//! - poses are world-to-camera, rotation stored as an axis-angle vector,
//! - each camera has its own `[f, k1, k2]` (focal length and two radial
//!   distortion terms), optimized together with pose and points
//!   (`SelfCalibration`),
//! - the camera looks down its **negative** z axis, which
//!   [`BALPinholeCameraStrict`] implements.
//!
//! With 23 000 point unknowns against 441 camera unknowns this is the case the
//! Schur complement exists for; the BA preset uses the matrix-free implicit
//! Schur solver. The `bundle_adjustment` binary exposes every option of this
//! pipeline (solver variants, preconditioners, all five parameterizations).
//! Example 12 feeds the same dataset to a sliding window one camera at a
//! time and compares against this batch solution.
//!
//! ```bash
//! cargo run --release --example bal_dataset
//! ```

use std::io::Write;
use std::time::Instant;

use apex_solver::apex_camera_models::{BALPinholeCameraStrict, DistortionModel, PinholeParams};
use apex_solver::apex_io::{BalDataset, BalLoader};
use apex_solver::core::VarKey;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::SelfCalibration;
use apex_solver::factors::visual::ProjectionFactor;
use apex_solver::{
    JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig, LieGroup, ManifoldType, SE3, SO3,
};
use nalgebra::{DVector, Matrix2xX, Vector2, Vector3};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// BAL stores rotations as axis-angle vectors.
fn axis_angle(rotation: &Vector3<f64>) -> SO3 {
    let angle = rotation.norm();
    if angle < 1e-12 {
        SO3::identity()
    } else {
        SO3::from_axis_angle(&(rotation / angle), angle)
    }
}

fn main() -> Result<()> {
    let mut out = std::io::stdout().lock();
    let path = apex_solver::apex_io::ensure_ba_dataset("ladybug", 49, 7776)?;
    let dataset: BalDataset = BalLoader::load(path.to_string_lossy().as_ref())?;
    writeln!(
        out,
        "{}: {} cameras, {} points, {} observations",
        path.display(),
        dataset.cameras.len(),
        dataset.points.len(),
        dataset.observations.len()
    )?;

    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut poses: Vec<VarKey> = Vec::with_capacity(dataset.cameras.len());
    let mut intrinsics: Vec<VarKey> = Vec::with_capacity(dataset.cameras.len());
    for camera in &dataset.cameras {
        let pose = SE3::from_translation_so3(camera.translation, axis_angle(&camera.rotation));
        poses.push(problem.add_variable(
            ManifoldType::SE3,
            DVector::from_column_slice(pose.as_param_slice()),
        ));
        intrinsics.push(problem.add_variable(
            ManifoldType::RN,
            DVector::from_vec(vec![camera.focal_length, camera.k1, camera.k2]),
        ));
    }
    let points: Vec<VarKey> = dataset
        .points
        .iter()
        .map(|point| {
            let key = problem.add_variable(
                ManifoldType::RN,
                DVector::from_column_slice(point.position.as_slice()),
            );
            problem.mark_for_elimination(key);
            key
        })
        .collect();

    for observation in &dataset.observations {
        let camera = &dataset.cameras[observation.camera_index];
        // The camera model the factor starts from; with `SelfCalibration` the
        // optimized `[f, k1, k2]` variable replaces it at every evaluation.
        let model = BALPinholeCameraStrict::new(
            PinholeParams {
                fx: camera.focal_length,
                fy: camera.focal_length,
                cx: 0.0,
                cy: 0.0,
            },
            DistortionModel::Radial {
                k1: camera.k1,
                k2: camera.k2,
            },
        )?;
        let factor: ProjectionFactor<BALPinholeCameraStrict, SelfCalibration> =
            ProjectionFactor::new(
                Matrix2xX::from_columns(&[Vector2::new(observation.x, observation.y)]),
                model,
            );
        problem.add_residual_block(
            &[
                poses[observation.camera_index],
                points[observation.point_index],
                intrinsics[observation.camera_index],
            ],
            Box::new(factor),
            Some(Box::new(HuberLoss::new(1.0)?)),
        )?;
    }
    // Gauge: hold the first camera's pose.
    for coordinate in 0..6 {
        problem.fix_variable(poses[0], coordinate)?;
    }

    let config = LevenbergMarquardtConfig::for_bundle_adjustment();
    let start = Instant::now();
    let result = LevenbergMarquardt::with_config(config).optimize(&mut problem)?;
    let observations = dataset.observations.len() as f64;
    writeln!(out, "status      {}", result.status)?;
    writeln!(out, "iterations  {}", result.iterations)?;
    writeln!(
        out,
        "RMSE        {:.3} px → {:.3} px (per image coordinate)",
        (result.initial_cost / observations).sqrt(),
        (result.final_cost / observations).sqrt()
    )?;
    writeln!(out, "time        {:.2} s", start.elapsed().as_secs_f64())?;
    Ok(())
}
