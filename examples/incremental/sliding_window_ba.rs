//! # 12 · Sliding-window bundle adjustment — a real dataset, one camera at a time
//!
//! Example 9 solved the Ladybug sequence in one batch. A visual odometry
//! system cannot: cameras arrive one by one, and the problem must stay small.
//! This example streams the same 49 cameras in order and runs a **local
//! bundle adjustment** after each one, the scheme used by keyframe-based
//! visual SLAM:
//!
//! - the newest `window` cameras and the points they see are **free**;
//! - older cameras that still see one of those points stay in the problem
//!   but are **fixed**: they anchor the gauge and the scale and keep their
//!   observations of the active points;
//! - a point no window camera sees any more is **finalized** (removed with
//!   its observations), and so is an old camera with nothing left to observe.
//!
//! Two rules about points matter as much as the window itself, and the
//! example is a record of why: without them the streamed map scores far worse
//! than the initial guess.
//!
//! 1. **A point is free only once two cameras see it.** One observation
//!    constrains a ray, not a depth; a free single-view point slides along its
//!    ray and is finalized wherever it ended up. Until its second view it is
//!    held fixed at its initial estimate.
//! 2. **A finalized point that is seen again returns fixed.** Ladybug is a
//!    camera rig moving through a street, and its cameras are not ordered so
//!    that a point is seen by consecutive cameras only: thousands of points
//!    leave the window and come back. Re-opening such a point as a free
//!    variable with only its new observations throws away everything its
//!    earlier observers said; returning it as a fixed map point localizes the
//!    new camera against it instead.
//!
//! The problem stays bounded however long the sequence is, and it needs no
//! marginal prior: fixing anchors is what keeps it sparse (a camera marginal
//! over hundreds of shared points would be dense). Example 11 is the
//! marginalization alternative, for state that has no such sparsity.
//!
//! At the end the streamed map is scored on *all* observations, the same way
//! as the batch solution, for two window sizes. Expect the window to land
//! between the initial guess and the batch optimum, closer with a larger
//! window. On a 49-camera dataset the batch solve is faster in total; the
//! window's point is that its cost per camera does not grow with the map.
//!
//! ```bash
//! cargo run --release --example sliding_window_ba
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::time::{Duration, Instant};

use apex_solver::apex_camera_models::{BALPinholeCameraStrict, DistortionModel, PinholeParams};
use apex_solver::apex_io::{BalDataset, BalLoader};
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::core::problem::Problem;
use apex_solver::core::{FactorKey, VarKey};
use apex_solver::factors::SelfCalibration;
use apex_solver::factors::visual::ProjectionFactor;
use apex_solver::{
    JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig, LieGroup, LinearSolverType,
    ManifoldType, SE3, SO3,
};
use nalgebra::{DVector, Matrix2xX, Vector2, Vector3};

const ITERATIONS_PER_STEP: usize = 10;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// A camera's estimate: world-to-camera pose and `[f, k1, k2]`.
#[derive(Clone)]
struct CameraState {
    pose: SE3,
    intrinsics: [f64; 3],
}

/// A complete estimate of the scene, the output of a batch or windowed run.
struct Map {
    cameras: Vec<CameraState>,
    points: Vec<Vector3<f64>>,
}

fn axis_angle(rotation: &Vector3<f64>) -> SO3 {
    let angle = rotation.norm();
    if angle < 1e-12 {
        SO3::identity()
    } else {
        SO3::from_axis_angle(&(rotation / angle), angle)
    }
}

/// The dataset's own starting estimate, as a [`Map`].
fn initial_map(dataset: &BalDataset) -> Map {
    Map {
        cameras: dataset
            .cameras
            .iter()
            .map(|camera| CameraState {
                pose: SE3::from_translation_so3(camera.translation, axis_angle(&camera.rotation)),
                intrinsics: [camera.focal_length, camera.k1, camera.k2],
            })
            .collect(),
        points: dataset.points.iter().map(|point| point.position).collect(),
    }
}

/// One observation as a self-calibrating BAL projection block.
fn add_observation(
    problem: &mut Problem,
    keys: [VarKey; 3],
    camera: &CameraState,
    pixel: Vector2<f64>,
) -> Result<FactorKey> {
    let model = BALPinholeCameraStrict::new(
        PinholeParams {
            fx: camera.intrinsics[0],
            fy: camera.intrinsics[0],
            cx: 0.0,
            cy: 0.0,
        },
        DistortionModel::Radial {
            k1: camera.intrinsics[1],
            k2: camera.intrinsics[2],
        },
    )?;
    let factor: ProjectionFactor<BALPinholeCameraStrict, SelfCalibration> =
        ProjectionFactor::new(Matrix2xX::from_columns(&[pixel]), model);
    Ok(problem.add_residual_block(
        &keys,
        Box::new(factor),
        Some(Box::new(HuberLoss::new(1.0)?)),
    )?)
}

/// Every camera, point and observation of the dataset, starting at `map`.
/// Returns the problem and its variable keys (cameras: pose and intrinsics).
fn full_problem(
    dataset: &BalDataset,
    map: &Map,
) -> Result<(Problem, Vec<[VarKey; 2]>, Vec<VarKey>)> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let cameras: Vec<[VarKey; 2]> = map
        .cameras
        .iter()
        .map(|camera| {
            [
                problem.add_variable(
                    ManifoldType::SE3,
                    DVector::from_column_slice(camera.pose.as_param_slice()),
                ),
                problem.add_variable(
                    ManifoldType::RN,
                    DVector::from_column_slice(&camera.intrinsics),
                ),
            ]
        })
        .collect();
    let points: Vec<VarKey> = map
        .points
        .iter()
        .map(|point| {
            let key = problem.add_variable(
                ManifoldType::RN,
                DVector::from_column_slice(point.as_slice()),
            );
            problem.mark_for_elimination(key);
            key
        })
        .collect();
    for observation in &dataset.observations {
        let [pose, intrinsics] = cameras[observation.camera_index];
        add_observation(
            &mut problem,
            [pose, points[observation.point_index], intrinsics],
            &map.cameras[observation.camera_index],
            Vector2::new(observation.x, observation.y),
        )?;
    }
    for coordinate in 0..6 {
        problem.fix_variable(cameras[0][0], coordinate)?;
    }
    Ok((problem, cameras, points))
}

/// RMS reprojection error of `map` over every observation of the dataset.
fn rmse(dataset: &BalDataset, map: &Map) -> Result<f64> {
    let (mut problem, _, _) = full_problem(dataset, map)?;
    let cost = apex_solver::optimizer::initialize_optimization_state(&mut problem)?.current_cost;
    Ok((cost / dataset.observations.len() as f64).sqrt())
}

/// The reference: example 9's batch solve.
fn batch(dataset: &BalDataset) -> Result<(Map, Duration)> {
    let start_map = initial_map(dataset);
    let (mut problem, cameras, points) = full_problem(dataset, &start_map)?;
    let start = Instant::now();
    let result = LevenbergMarquardt::with_config(LevenbergMarquardtConfig::for_bundle_adjustment())
        .optimize(&mut problem)?;
    let elapsed = start.elapsed();
    problem.update_values_from(&result.parameters)?;
    Ok((read_map(&problem, &cameras, &points)?, elapsed))
}

fn read_camera(problem: &Problem, keys: [VarKey; 2]) -> Result<CameraState> {
    let pose = SE3::from_param_slice(
        problem
            .variable_params(keys[0])
            .ok_or("a camera pose is missing")?,
    );
    let intrinsics = problem
        .variable_params(keys[1])
        .ok_or("camera intrinsics are missing")?;
    Ok(CameraState {
        pose,
        intrinsics: [intrinsics[0], intrinsics[1], intrinsics[2]],
    })
}

fn read_point(problem: &Problem, key: VarKey) -> Result<Vector3<f64>> {
    Ok(Vector3::from_column_slice(
        problem.variable_params(key).ok_or("a point is missing")?,
    ))
}

fn read_map(problem: &Problem, cameras: &[[VarKey; 2]], points: &[VarKey]) -> Result<Map> {
    Ok(Map {
        cameras: cameras
            .iter()
            .map(|keys| read_camera(problem, *keys))
            .collect::<Result<_>>()?,
        points: points
            .iter()
            .map(|key| read_point(problem, *key))
            .collect::<Result<_>>()?,
    })
}

/// Statistics of one windowed run.
struct WindowRun {
    map: Map,
    total: Duration,
    slowest_step: Duration,
    largest_problem: usize,
    reactivated: usize,
}

/// Stream the cameras through a window of `window` free cameras.
fn sliding_window(dataset: &BalDataset, window: usize) -> Result<WindowRun> {
    let start_map = initial_map(dataset);
    let mut observed_by: Vec<Vec<(usize, Vector2<f64>)>> = vec![Vec::new(); dataset.cameras.len()];
    for observation in &dataset.observations {
        observed_by[observation.camera_index].push((
            observation.point_index,
            Vector2::new(observation.x, observation.y),
        ));
    }

    let mut problem = Problem::new(JacobianMode::Sparse);
    let config = LevenbergMarquardtConfig::for_bundle_adjustment()
        .with_linear_solver_type(LinearSolverType::ExplicitSparseSchur)
        .with_max_iterations(ITERATIONS_PER_STEP);

    // What is in the problem right now.
    let mut camera_keys: HashMap<usize, [VarKey; 2]> = HashMap::new();
    let mut point_keys: HashMap<usize, VarKey> = HashMap::new();
    let mut free: VecDeque<usize> = VecDeque::new();
    // Bookkeeping to remove things later: the blocks of each point, and the
    // points each camera in the problem observes.
    let mut blocks_of_point: HashMap<usize, Vec<(usize, FactorKey)>> = HashMap::new();
    let mut points_of_camera: HashMap<usize, HashSet<usize>> = HashMap::new();
    // Points seen by one camera so far, held fixed until a second one sees them.
    let mut pending: HashSet<usize> = HashSet::new();
    // What has left the problem for good.
    let mut final_cameras: Vec<Option<CameraState>> = vec![None; dataset.cameras.len()];
    let mut final_points: Vec<Option<Vector3<f64>>> = vec![None; dataset.points.len()];

    let mut total = Duration::ZERO;
    let mut slowest_step = Duration::ZERO;
    let mut largest_problem = 0;
    let mut reactivated = 0;

    for (camera, observations) in observed_by.iter().enumerate() {
        let step_start = Instant::now();

        // 1. The new camera. A real system would predict it (constant
        //    velocity, PnP); BAL ships an initial estimate, which stands in.
        let state = start_map.cameras[camera].clone();
        let keys = [
            problem.add_variable(
                ManifoldType::SE3,
                DVector::from_column_slice(state.pose.as_param_slice()),
            ),
            problem.add_variable(
                ManifoldType::RN,
                DVector::from_column_slice(&state.intrinsics),
            ),
        ];
        if camera == 0 {
            for coordinate in 0..6 {
                problem.fix_variable(keys[0], coordinate)?;
            }
        }
        camera_keys.insert(camera, keys);
        free.push_back(camera);

        // 2. Its observations. A point seen for the first time enters at the
        //    dataset's estimate (a real system would triangulate it); a point
        //    finalized earlier and seen again is re-opened at its final value.
        for (point, pixel) in observations {
            let point_key = match point_keys.get(point) {
                Some(key) => *key,
                None => {
                    let key = match final_points[*point] {
                        // Seen again after it was finalized: it re-enters as a
                        // *fixed* map point, so the new camera is localized
                        // against it but its estimate does not change.
                        Some(previous) => {
                            reactivated += 1;
                            let key = problem.add_variable(
                                ManifoldType::RN,
                                DVector::from_column_slice(previous.as_slice()),
                            );
                            for coordinate in 0..3 {
                                problem.fix_variable(key, coordinate)?;
                            }
                            key
                        }
                        // New: held fixed until a second camera sees it. One
                        // observation constrains only a ray, not a depth.
                        None => {
                            let start = start_map.points[*point];
                            let key = problem.add_variable(
                                ManifoldType::RN,
                                DVector::from_column_slice(start.as_slice()),
                            );
                            for coordinate in 0..3 {
                                problem.fix_variable(key, coordinate)?;
                            }
                            pending.insert(*point);
                            key
                        }
                    };
                    point_keys.insert(*point, key);
                    key
                }
            };
            let block =
                add_observation(&mut problem, [keys[0], point_key, keys[1]], &state, *pixel)?;
            // The second view: the point is triangulable and becomes free.
            if blocks_of_point
                .get(point)
                .is_some_and(|blocks| !blocks.is_empty())
                && pending.remove(point)
            {
                problem.unfix_variable(point_key);
                problem.mark_for_elimination(point_key);
            }
            blocks_of_point
                .entry(*point)
                .or_default()
                .push((camera, block));
            points_of_camera.entry(camera).or_default().insert(*point);
        }

        // 3. Local bundle adjustment, warm-started, answer written back.
        //    The first camera alone has nothing free to solve for: it is the
        //    gauge anchor and its points wait for a second view.
        largest_problem = largest_problem.max(problem.num_residual_blocks());
        if camera > 0 {
            let result = LevenbergMarquardt::with_config(config.clone()).optimize(&mut problem)?;
            problem.update_values_from(&result.parameters)?;
        }

        // 4. The oldest free camera becomes a fixed anchor.
        if free.len() > window {
            let leaving = free.pop_front().ok_or("the window is empty")?;
            let [pose, intrinsics] = camera_keys[&leaving];
            for coordinate in 0..6 {
                problem.fix_variable(pose, coordinate)?;
            }
            for coordinate in 0..3 {
                problem.fix_variable(intrinsics, coordinate)?;
            }

            // 5. Finalize the points no free camera sees any more ...
            let candidates: Vec<usize> = points_of_camera
                .get(&leaving)
                .into_iter()
                .flatten()
                .copied()
                .collect();
            for point in candidates {
                let seen_by_window = free.iter().any(|c| {
                    points_of_camera
                        .get(c)
                        .is_some_and(|seen| seen.contains(&point))
                });
                if seen_by_window {
                    continue;
                }
                let Some(key) = point_keys.remove(&point) else {
                    continue;
                };
                pending.remove(&point);
                final_points[point] = Some(read_point(&problem, key)?);
                for (observer, block) in blocks_of_point.remove(&point).unwrap_or_default() {
                    problem.remove_residual_block(block);
                    if let Some(seen) = points_of_camera.get_mut(&observer) {
                        seen.remove(&point);
                    }
                }
                problem.try_remove_variable(key)?;
            }

            // ... and the anchors that no longer observe anything.
            let idle: Vec<usize> = camera_keys
                .keys()
                .copied()
                .filter(|c| {
                    !free.contains(c) && points_of_camera.get(c).is_none_or(HashSet::is_empty)
                })
                .collect();
            for old in idle {
                let keys = camera_keys
                    .remove(&old)
                    .ok_or("an idle camera has no keys")?;
                final_cameras[old] = Some(read_camera(&problem, keys)?);
                problem.try_remove_variable(keys[0])?;
                problem.try_remove_variable(keys[1])?;
                points_of_camera.remove(&old);
            }
        }

        let step = step_start.elapsed();
        total += step;
        slowest_step = slowest_step.max(step);
    }

    // Whatever is still in the problem holds its last estimate.
    for (camera, keys) in &camera_keys {
        final_cameras[*camera] = Some(read_camera(&problem, *keys)?);
    }
    for (point, key) in &point_keys {
        final_points[*point] = Some(read_point(&problem, *key)?);
    }
    let map = Map {
        cameras: final_cameras
            .into_iter()
            .map(|c| c.ok_or("a camera was never estimated"))
            .collect::<std::result::Result<_, _>>()?,
        points: final_points
            .into_iter()
            .map(|p| p.ok_or("a point was never estimated"))
            .collect::<std::result::Result<_, _>>()?,
    };
    Ok(WindowRun {
        map,
        total,
        slowest_step,
        largest_problem,
        reactivated,
    })
}

/// Mean distance between the camera centres of two maps.
fn centre_difference(a: &Map, b: &Map) -> f64 {
    let sum: f64 = a
        .cameras
        .iter()
        .zip(&b.cameras)
        .map(|(x, y)| {
            (x.pose.inverse(None).translation() - y.pose.inverse(None).translation()).norm()
        })
        .sum();
    sum / a.cameras.len() as f64
}

fn main() -> Result<()> {
    let mut out = std::io::stdout().lock();
    let path = apex_solver::apex_io::ensure_ba_dataset("ladybug", 49, 7776)?;
    let dataset = BalLoader::load(path.to_string_lossy().as_ref())?;
    writeln!(
        out,
        "{} cameras, {} points, {} observations\n",
        dataset.cameras.len(),
        dataset.points.len(),
        dataset.observations.len()
    )?;

    let initial = initial_map(&dataset);
    let (reference, batch_time) = batch(&dataset)?;
    writeln!(
        out,
        "{:<18} {:>10} {:>12} {:>11} {:>14} {:>12}",
        "estimator", "RMSE [px]", "Δ centre", "total [s]", "max step [ms]", "max blocks"
    )?;
    writeln!(
        out,
        "{:<18} {:>10.3} {:>12} {:>11}",
        "initial guess",
        rmse(&dataset, &initial)?,
        "",
        ""
    )?;
    writeln!(
        out,
        "{:<18} {:>10.3} {:>12} {:>11.2} {:>14} {:>12}",
        "batch",
        rmse(&dataset, &reference)?,
        "—",
        batch_time.as_secs_f64(),
        "—",
        dataset.observations.len()
    )?;
    for window in [5, 10] {
        let run = sliding_window(&dataset, window)?;
        writeln!(
            out,
            "{:<18} {:>10.3} {:>12.4} {:>11.2} {:>14.0} {:>12}",
            format!("window of {window}"),
            rmse(&dataset, &run.map)?,
            centre_difference(&run.map, &reference),
            run.total.as_secs_f64(),
            1e3 * run.slowest_step.as_secs_f64(),
            run.largest_problem
        )?;
        if run.reactivated > 0 {
            writeln!(
                out,
                "{:<18} {} times a finalized point was seen again (re-entered fixed)",
                "", run.reactivated
            )?;
        }
    }
    writeln!(
        out,
        "\nΔ centre: mean camera-centre distance to the batch solution, in the dataset's units."
    )?;
    Ok(())
}
