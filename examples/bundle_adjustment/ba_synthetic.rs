//! # 7 · Bundle adjustment — cameras, landmarks and the Schur complement
//!
//! Twenty cameras on a ring look at 300 points. Each camera sees each point
//! once; the measurement is a pixel. Bundle adjustment refines camera poses
//! and 3-D points together so every point reprojects onto its pixels.
//!
//! What is new compared with the pose graphs:
//!
//! - **Two kinds of variables**: `SE3` camera poses and `ℝ³` landmarks.
//! - **[`ProjectionFactor`]**, parameterized at compile time by *what it
//!   optimizes*. `ProjectionFactor<PinholeCamera, BundleAdjustment>` optimizes
//!   pose and landmark and keeps the intrinsics fixed; example 8 shows the
//!   other choices. The pose is **world-to-camera** (`T_cw`, BAL's convention):
//!   `p_cam = R·p_world + t`.
//! - **A 7-DoF gauge**: a monocular reconstruction can be moved, rotated *and
//!   scaled* without changing a single pixel. Fixing camera 0 removes six
//!   degrees of freedom; fixing one coordinate of a second camera removes the
//!   scale. *Which* coordinate matters: it should lie along a long baseline.
//!   Here it is the `x` of the camera opposite camera 0, 16 m away. Fixing the
//!   `x` of the neighbouring camera instead (0.4 m of `x` baseline) is just as
//!   valid mathematically, but it pins the scale through a short lever and
//!   leaves the map with a few percent of scale error.
//! - **The Schur complement**: there are far more landmark unknowns than pose
//!   unknowns, and landmarks only connect to cameras. Marking them with
//!   `mark_for_elimination` lets the solver eliminate them first and solve a
//!   small camera-only system. Compare the three linear solvers in the table.
//!
//! ```bash
//! cargo run --release --example ba_synthetic
//! ```

use std::io::Write;
use std::time::Instant;

use apex_solver::core::VarKey;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::core::variable::ManifoldVariable;
use apex_solver::factors::BundleAdjustment;
use apex_solver::factors::visual::ProjectionFactor;
use apex_solver::{
    CameraModel, JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig, LieGroup,
    LinearSolverType, ManifoldType, PinholeCamera, SE3,
};
use nalgebra::{DVector, Matrix2xX, Matrix3, Rotation3, UnitQuaternion, Vector2, Vector3};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use slotmap::SlotMap;

const CAMERAS: usize = 20;
const LANDMARKS: usize = 300;
const PIXEL_SIGMA: f64 = 1.0;
const IMAGE: (f64, f64) = (640.0, 480.0);

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn gaussian(rng: &mut StdRng, sigma: f64) -> f64 {
    let u1: f64 = rng.random_range(f64::EPSILON..1.0);
    let u2: f64 = rng.random();
    sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// `T_cw` of a camera at `center` looking at the origin, `y` pointing down.
fn look_at_origin(center: Vector3<f64>) -> SE3 {
    let forward = (-center).normalize();
    let right = forward.cross(&Vector3::z()).normalize();
    let down = forward.cross(&right);
    // Rows of R_cw are the camera axes expressed in the world frame.
    let rotation = Rotation3::from_matrix_unchecked(Matrix3::from_rows(&[
        right.transpose(),
        down.transpose(),
        forward.transpose(),
    ]));
    let quaternion = UnitQuaternion::from_rotation_matrix(&rotation);
    SE3::new(-(quaternion * center), quaternion)
}

/// A small random rigid motion, to perturb a pose.
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

struct Scene {
    camera: PinholeCamera,
    poses: Vec<SE3>,
    points: Vec<Vector3<f64>>,
    /// `(camera index, landmark index, measured pixel)`
    observations: Vec<(usize, usize, Vector2<f64>)>,
}

fn simulate(rng: &mut StdRng) -> Result<Scene> {
    let camera = PinholeCamera::from([500.0, 500.0, IMAGE.0 / 2.0, IMAGE.1 / 2.0]);
    let poses: Vec<SE3> = (0..CAMERAS)
        .map(|i| {
            let angle = std::f64::consts::TAU * i as f64 / CAMERAS as f64;
            look_at_origin(Vector3::new(8.0 * angle.cos(), 8.0 * angle.sin(), 1.0))
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
    let mut observations = Vec::new();
    for (c, pose) in poses.iter().enumerate() {
        for (l, point) in points.iter().enumerate() {
            let pixel = camera.project(&pose.act(point, None, None))?;
            let noisy =
                pixel + Vector2::new(gaussian(rng, PIXEL_SIGMA), gaussian(rng, PIXEL_SIGMA));
            if noisy.x >= 0.0 && noisy.x < IMAGE.0 && noisy.y >= 0.0 && noisy.y < IMAGE.1 {
                observations.push((c, l, noisy));
            }
        }
    }
    Ok(Scene {
        camera,
        poses,
        points,
        observations,
    })
}

struct Built {
    problem: Problem,
    poses: Vec<VarKey>,
    points: Vec<VarKey>,
}

/// The camera whose `x` coordinate fixes the scale: opposite camera 0.
const SCALE_CAMERA: usize = CAMERAS / 2;

/// Every camera but the gauge pair and every point start from a perturbed guess.
fn build(scene: &Scene, rng: &mut StdRng) -> Result<Built> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let poses: Vec<VarKey> = scene
        .poses
        .iter()
        .enumerate()
        .map(|(c, pose)| {
            let start = if c == 0 || c == SCALE_CAMERA {
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
    let mut points = Vec::with_capacity(scene.points.len());
    for point in &scene.points {
        let start =
            point + Vector3::new(gaussian(rng, 0.2), gaussian(rng, 0.2), gaussian(rng, 0.2));
        let key = problem.add_variable(
            ManifoldType::RN,
            DVector::from_column_slice(start.as_slice()),
        );
        // Landmarks connect only to cameras, never to each other: exactly
        // the structure the Schur complement needs.
        problem.mark_for_elimination(key);
        points.push(key);
    }

    for (c, l, pixel) in &scene.observations {
        // One observation per factor here; a factor can also hold several
        // observations of several landmarks from the same camera.
        let factor: ProjectionFactor<PinholeCamera, BundleAdjustment> =
            ProjectionFactor::new(Matrix2xX::from_columns(&[*pixel]), scene.camera);
        // The residual is in pixels and the noise is 1 px, so no noise model
        // is needed; Huber guards against the odd bad match.
        problem.add_residual_block(
            &[poses[*c], points[*l]],
            Box::new(factor),
            Some(Box::new(HuberLoss::new(1.0)?)),
        )?;
    }

    // Gauge: camera 0 completely (tangent coordinates 0..3 translation,
    // 3..6 rotation), plus the scale camera's first translation coordinate.
    // For a world-to-camera pose that coordinate moves the camera centre along
    // world x, the direction of the 16 m baseline to camera 0.
    for coordinate in 0..6 {
        problem.fix_variable(poses[0], coordinate)?;
    }
    problem.fix_variable(poses[SCALE_CAMERA], 0)?;
    Ok(Built {
        problem,
        poses,
        points,
    })
}

/// The solved variables, as `optimize` returns them.
type Values = SlotMap<VarKey, Box<dyn ManifoldVariable>>;

/// One solved variable's parameters.
fn value(values: &Values, key: VarKey) -> Result<&[f64]> {
    Ok(values
        .get(key)
        .ok_or("a variable is missing from the result")?
        .as_param_slice())
}

/// Mean distance of the camera centres and of the points from the truth.
fn errors(scene: &Scene, built: &Built, values: &Values) -> Result<(f64, f64)> {
    let mut camera_error = 0.0;
    for (key, truth) in built.poses.iter().zip(&scene.poses) {
        let estimate = SE3::from_param_slice(value(values, *key)?);
        // The centre of a world-to-camera pose is the translation of its inverse.
        camera_error +=
            (estimate.inverse(None).translation() - truth.inverse(None).translation()).norm();
    }
    let mut point_error = 0.0;
    for (key, truth) in built.points.iter().zip(&scene.points) {
        point_error += (Vector3::from_column_slice(value(values, *key)?) - truth).norm();
    }
    Ok((
        camera_error / scene.poses.len() as f64,
        point_error / scene.points.len() as f64,
    ))
}

fn main() -> Result<()> {
    let mut out = std::io::stdout().lock();
    let mut rng = StdRng::seed_from_u64(3);
    let scene = simulate(&mut rng)?;
    writeln!(
        out,
        "{CAMERAS} cameras, {LANDMARKS} landmarks, {} observations ({} pose and {} landmark unknowns)\n",
        scene.observations.len(),
        6 * CAMERAS,
        3 * LANDMARKS
    )?;
    writeln!(
        out,
        "{:<22} {:>9} {:>9} {:>6} {:>9} {:>11} {:>11}",
        "linear solver", "RMSE₀[px]", "RMSE[px]", "iters", "time [s]", "camera [m]", "point [m]"
    )?;

    for (label, solver) in [
        ("sparse Cholesky", LinearSolverType::SparseCholesky),
        ("explicit Schur", LinearSolverType::ExplicitSparseSchur),
        (
            "implicit Schur (PCG)",
            LinearSolverType::ImplicitSparseSchur,
        ),
    ] {
        // The same seed per run, so every solver starts from the same guess.
        let mut start_rng = StdRng::seed_from_u64(11);
        let mut built = build(&scene, &mut start_rng)?;
        // The BA preset: tuned damping and stopping rules; only the linear
        // solver is overridden here.
        let config = LevenbergMarquardtConfig::for_bundle_adjustment()
            .with_linear_solver_type(solver)
            .with_max_iterations(50);
        let start = Instant::now();
        let result = LevenbergMarquardt::with_config(config).optimize(&mut built.problem)?;
        let seconds = start.elapsed().as_secs_f64();
        let observations = scene.observations.len() as f64;
        let (camera_error, point_error) = errors(&scene, &built, &result.parameters)?;
        writeln!(
            out,
            "{label:<22} {:>9.2} {:>9.3} {:>6} {:>9.3} {:>11.4} {:>11.4}",
            (result.initial_cost / observations).sqrt(),
            (result.final_cost / observations).sqrt(),
            result.iterations,
            seconds,
            camera_error,
            point_error
        )?;
        if (result.final_cost / observations).sqrt() > 2.0 * PIXEL_SIGMA {
            return Err(
                format!("{label}: the reprojection error did not reach the noise level").into(),
            );
        }
    }
    writeln!(
        out,
        "\nRMSE is per image coordinate; the pixel noise is {PIXEL_SIGMA} px."
    )?;
    Ok(())
}
