//! # 8 · What to optimize — one factor, five parameterizations
//!
//! [`ProjectionFactor`]`<Camera, OP>` decides at **compile time** which of
//! camera pose, landmark and intrinsics are unknowns. The choice changes the
//! variables a block is registered with, and nothing else:
//!
//! | alias | optimizes | variables, in order | typical task |
//! |---|---|---|---|
//! | `BundleAdjustment` | pose, landmark | `[pose, point]` | structure from motion (example 7) |
//! | `OnlyPose` | pose | `[pose]` | localizing a camera in a known map (PnP refinement) |
//! | `OnlyLandmarks` | landmark | `[point]` | triangulating from known cameras |
//! | `SelfCalibration` | all three | `[pose, point, intrinsics]` | calibrating the lens while mapping |
//! | `OnlyIntrinsics` | intrinsics | `[intrinsics]` | calibration against a known target |
//!
//! Whatever is not optimized is given to the factor up front, with
//! `with_fixed_pose` or `with_fixed_landmarks`; fixed intrinsics are the
//! camera model the factor is constructed with. There is no runtime branch
//! and no unused Jacobian column. This example runs the three tasks that
//! example 7 did not: localization, triangulation and self-calibration, on the
//! same synthetic ring of cameras.
//!
//! ```bash
//! cargo run --release --example ba_parameterizations
//! ```

use std::io::Write;

use apex_solver::core::VarKey;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::factors::visual::ProjectionFactor;
use apex_solver::factors::{OnlyLandmarks, OnlyPose, SelfCalibration};
use apex_solver::{
    CameraModel, JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig, LieGroup,
    LinearSolverType, ManifoldType, PinholeCamera, SE3,
};
use nalgebra::{
    DVector, Matrix2xX, Matrix3, Matrix3xX, Rotation3, UnitQuaternion, Vector2, Vector3,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const CAMERAS: usize = 20;
const LANDMARKS: usize = 300;
const PIXEL_SIGMA: f64 = 1.0;
const IMAGE: (f64, f64) = (640.0, 480.0);
const TRUE_INTRINSICS: [f64; 4] = [500.0, 500.0, 320.0, 240.0];

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn gaussian(rng: &mut StdRng, sigma: f64) -> f64 {
    let u1: f64 = rng.random_range(f64::EPSILON..1.0);
    let u2: f64 = rng.random();
    sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn look_at_origin(center: Vector3<f64>) -> SE3 {
    let forward = (-center).normalize();
    let right = forward.cross(&Vector3::z()).normalize();
    let down = forward.cross(&right);
    let rotation = Rotation3::from_matrix_unchecked(Matrix3::from_rows(&[
        right.transpose(),
        down.transpose(),
        forward.transpose(),
    ]));
    let quaternion = UnitQuaternion::from_rotation_matrix(&rotation);
    SE3::new(-(quaternion * center), quaternion)
}

fn jitter(rng: &mut StdRng, translation: f64, rotation: f64) -> SE3 {
    SE3::from_translation_euler(
        gaussian(rng, translation),
        gaussian(rng, translation),
        gaussian(rng, translation),
        gaussian(rng, rotation),
        gaussian(rng, rotation),
        gaussian(rng, rotation),
    )
}

/// The ring of example 7: world-to-camera poses, points, and per-camera
/// observations `(landmark index, pixel)`.
struct Scene {
    camera: PinholeCamera,
    poses: Vec<SE3>,
    points: Vec<Vector3<f64>>,
    seen: Vec<Vec<(usize, Vector2<f64>)>>,
}

fn simulate(rng: &mut StdRng) -> Result<Scene> {
    let camera = PinholeCamera::from(TRUE_INTRINSICS);
    let poses: Vec<SE3> = (0..CAMERAS)
        .map(|i| {
            let angle = std::f64::consts::TAU * i as f64 / CAMERAS as f64;
            // Heights vary around the ring: see `self_calibrate` for why.
            look_at_origin(Vector3::new(
                8.0 * angle.cos(),
                8.0 * angle.sin(),
                1.0 + 3.0 * (2.0 * angle).sin(),
            ))
        })
        .collect();
    let points: Vec<Vector3<f64>> = (0..LANDMARKS)
        .map(|_| {
            Vector3::new(
                rng.random_range(-2.0..2.0),
                rng.random_range(-2.0..2.0),
                rng.random_range(-1.0..1.0),
            )
        })
        .collect();
    let mut seen = vec![Vec::new(); CAMERAS];
    for (c, pose) in poses.iter().enumerate() {
        for (l, point) in points.iter().enumerate() {
            let pixel = camera.project(&pose.act(point, None, None))?;
            let noisy =
                pixel + Vector2::new(gaussian(rng, PIXEL_SIGMA), gaussian(rng, PIXEL_SIGMA));
            if noisy.x >= 0.0 && noisy.x < IMAGE.0 && noisy.y >= 0.0 && noisy.y < IMAGE.1 {
                seen[c].push((l, noisy));
            }
        }
    }
    Ok(Scene {
        camera,
        poses,
        points,
        seen,
    })
}

/// Camera-centre distance between two world-to-camera poses.
fn centre_error(a: &SE3, b: &SE3) -> f64 {
    (a.inverse(None).translation() - b.inverse(None).translation()).norm()
}

/// The solved variables, as `optimize` returns them.
type Values = slotmap::SlotMap<VarKey, Box<dyn apex_solver::core::variable::ManifoldVariable>>;

/// One solved variable's parameters.
fn param(result: &apex_solver::optimizer::SolverResult<Values>, key: VarKey) -> Result<&[f64]> {
    Ok(result
        .parameters
        .get(key)
        .ok_or("a variable is missing from the result")?
        .as_param_slice())
}

/// Localization: the map is known, one camera pose is not.
///
/// All of a camera's observations go into **one** factor: its observation
/// matrix is `2 × N` and its fixed landmarks `3 × N`, so the problem has a
/// single 6-DoF variable and one residual block of `2N` rows. Dense assembly
/// fits such a small, fully coupled problem best.
fn localize(scene: &Scene, rng: &mut StdRng, out: &mut impl Write) -> Result<()> {
    let camera = 7;
    let truth = &scene.poses[camera];
    let start = jitter(rng, 0.3, 0.05).compose(truth, None, None);

    let observations = Matrix2xX::from_columns(
        &scene.seen[camera]
            .iter()
            .map(|(_, px)| *px)
            .collect::<Vec<_>>(),
    );
    let landmarks = Matrix3xX::from_columns(
        &scene.seen[camera]
            .iter()
            .map(|(l, _)| scene.points[*l])
            .collect::<Vec<_>>(),
    );
    let factor: ProjectionFactor<PinholeCamera, OnlyPose> =
        ProjectionFactor::new(observations, scene.camera).with_fixed_landmarks(landmarks);

    let mut problem = Problem::new(JacobianMode::Dense);
    let pose = problem.add_variable(
        ManifoldType::SE3,
        DVector::from_column_slice(start.as_param_slice()),
    );
    problem.add_residual_block(
        &[pose],
        Box::new(factor),
        Some(Box::new(HuberLoss::new(1.0)?)),
    )?;

    let config =
        LevenbergMarquardtConfig::new().with_linear_solver_type(LinearSolverType::DenseCholesky);
    let result = LevenbergMarquardt::with_config(config).optimize(&mut problem)?;
    let estimate = SE3::from_param_slice(param(&result, pose)?);
    writeln!(
        out,
        "OnlyPose        camera {camera} from {} points: centre error {:.3} m → {:.4} m ({} iterations)",
        scene.seen[camera].len(),
        centre_error(&start, truth),
        centre_error(&estimate, truth),
        result.iterations
    )?;
    Ok(())
}

/// Triangulation: the cameras are known, the points are not.
///
/// Each block holds one observation and the pose that made it; the variables
/// are the points alone. Points do not interact, so the problem is a set of
/// independent 3-unknown problems solved together.
fn triangulate(scene: &Scene, rng: &mut StdRng, out: &mut impl Write) -> Result<()> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut keys = Vec::with_capacity(scene.points.len());
    let mut before = 0.0;
    for point in &scene.points {
        let start =
            point + Vector3::new(gaussian(rng, 0.3), gaussian(rng, 0.3), gaussian(rng, 0.3));
        before += (start - point).norm();
        keys.push(problem.add_variable(
            ManifoldType::RN,
            DVector::from_column_slice(start.as_slice()),
        ));
    }
    for (c, observations) in scene.seen.iter().enumerate() {
        for (l, pixel) in observations {
            let factor: ProjectionFactor<PinholeCamera, OnlyLandmarks> =
                ProjectionFactor::new(Matrix2xX::from_columns(&[*pixel]), scene.camera)
                    .with_fixed_pose(scene.poses[c].clone());
            problem.add_residual_block(&[keys[*l]], Box::new(factor), None)?;
        }
    }
    let result = LevenbergMarquardt::new().optimize(&mut problem)?;
    let mut after = 0.0;
    for (key, point) in keys.iter().zip(&scene.points) {
        after += (Vector3::from_column_slice(param(&result, *key)?) - point).norm();
    }
    let n = scene.points.len() as f64;
    writeln!(
        out,
        "OnlyLandmarks   {} points from known cameras: error {:.3} m → {:.4} m ({} iterations)",
        scene.points.len(),
        before / n,
        after / n,
        result.iterations
    )?;
    Ok(())
}

/// Self-calibration: poses, points *and* the intrinsics `[fx, fy, cx, cy]`.
///
/// Every camera of the ring is the same physical camera, so there is one
/// intrinsics variable shared by every block. Sharing a variable is just
/// registering the same key in many blocks.
///
/// Self-calibration needs the right motion. With every camera at the same
/// height (example 7's ring), stretching the world vertically while shrinking
/// `fy` leaves almost every pixel where it was, and `fy` comes out tens of
/// pixels wrong with a perfectly small reprojection error. That is a
/// *critical motion*, a property of the data rather than of the solver; the
/// cameras here change height around the ring to break it.
fn self_calibrate(scene: &Scene, rng: &mut StdRng, out: &mut impl Write) -> Result<()> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let wrong = [540.0, 470.0, 330.0, 230.0];
    let intrinsics = problem.add_variable(ManifoldType::RN, DVector::from_column_slice(&wrong));

    let scale_camera = CAMERAS / 2;
    let poses: Vec<VarKey> = scene
        .poses
        .iter()
        .enumerate()
        .map(|(c, pose)| {
            let start = if c == 0 || c == scale_camera {
                pose.clone()
            } else {
                jitter(rng, 0.1, 0.02).compose(pose, None, None)
            };
            problem.add_variable(
                ManifoldType::SE3,
                DVector::from_column_slice(start.as_param_slice()),
            )
        })
        .collect();
    let points: Vec<VarKey> = scene
        .points
        .iter()
        .map(|point| {
            let start =
                point + Vector3::new(gaussian(rng, 0.2), gaussian(rng, 0.2), gaussian(rng, 0.2));
            let key = problem.add_variable(
                ManifoldType::RN,
                DVector::from_column_slice(start.as_slice()),
            );
            problem.mark_for_elimination(key);
            key
        })
        .collect();

    for (c, observations) in scene.seen.iter().enumerate() {
        for (l, pixel) in observations {
            // The camera passed here only seeds the factor; the optimized
            // intrinsics come from the third variable.
            let factor: ProjectionFactor<PinholeCamera, SelfCalibration> =
                ProjectionFactor::new(Matrix2xX::from_columns(&[*pixel]), scene.camera);
            problem.add_residual_block(
                &[poses[c], points[*l], intrinsics],
                Box::new(factor),
                Some(Box::new(HuberLoss::new(1.0)?)),
            )?;
        }
    }
    for coordinate in 0..6 {
        problem.fix_variable(poses[0], coordinate)?;
    }
    problem.fix_variable(poses[scale_camera], 0)?;

    let config = LevenbergMarquardtConfig::for_bundle_adjustment()
        .with_linear_solver_type(LinearSolverType::ExplicitSparseSchur)
        .with_max_iterations(50);
    let result = LevenbergMarquardt::with_config(config).optimize(&mut problem)?;
    let recovered = param(&result, intrinsics)?;
    writeln!(out, "SelfCalibration ({} iterations)", result.iterations)?;
    for (index, name) in ["fx", "fy", "cx", "cy"].iter().enumerate() {
        writeln!(
            out,
            "    {name}: start {:>6.1}  →  {:>7.2}   (true {:.0})",
            wrong[index], recovered[index], TRUE_INTRINSICS[index]
        )?;
    }
    let worst = recovered
        .iter()
        .zip(TRUE_INTRINSICS)
        .map(|(estimate, truth)| (estimate - truth).abs())
        .fold(0.0, f64::max);
    if worst > 5.0 {
        return Err(format!("self-calibration missed the intrinsics by {worst:.1} px").into());
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut out = std::io::stdout().lock();
    let mut rng = StdRng::seed_from_u64(3);
    let scene = simulate(&mut rng)?;
    localize(&scene, &mut rng, &mut out)?;
    triangulate(&scene, &mut rng, &mut out)?;
    self_calibrate(&scene, &mut rng, &mut out)?;
    Ok(())
}
