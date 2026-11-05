//! # 2 · A 2-D pose graph — poses on a manifold, noise models, gauge freedom
//!
//! A robot drives five laps of a 10 m square. Its wheel odometry drifts; every
//! time it passes a place it has seen before it also gets a *loop closure*, a
//! relative-pose measurement to the earlier visit. The pose graph combines the
//! two:
//!
//! - one `SE2` variable per pose (`[x, y, θ]`, optimized on the manifold, so
//!   the angle never needs wrapping by hand),
//! - one [`BetweenFactor`] per odometry step and per loop closure, each with a
//!   [`NoiseModel`] that says how much to trust it,
//! - one [`PriorFactor`] on the first pose to remove the **gauge freedom**:
//!   relative measurements are unchanged if the whole map is moved, so without
//!   an anchor the problem has infinitely many solutions.
//!
//! The example also shows the alternative anchor, `fix_variable`, and checks
//! that both give the same map.
//!
//! ```bash
//! cargo run --release --example pose_graph_2d
//! ```

use std::io::Write;

use apex_solver::core::VarKey;
use apex_solver::core::noise::NoiseModel;
use apex_solver::core::problem::Problem;
use apex_solver::core::variable::ManifoldVariable;
use apex_solver::factors::pose::{BetweenFactor, PriorFactor};
use apex_solver::{
    JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig, LieGroup, ManifoldType, SE2,
};
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use slotmap::SlotMap;

/// Steps per side of the square, one metre each.
const SIDE: usize = 10;
const LAPS: usize = 5;
/// Odometry noise per step: `[x, y, θ]`, metres and radians.
const ODOMETRY_SIGMAS: [f64; 3] = [0.05, 0.05, 0.02];
/// Loop closures (e.g. from place recognition) are more precise.
const LOOP_SIGMAS: [f64; 3] = [0.02, 0.02, 0.005];

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn gaussian(rng: &mut StdRng, sigma: f64) -> f64 {
    let u1: f64 = rng.random_range(f64::EPSILON..1.0);
    let u2: f64 = rng.random();
    sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A noisy copy of a relative motion.
fn perturb(rng: &mut StdRng, motion: &SE2, sigmas: &[f64; 3]) -> SE2 {
    SE2::from_xy_angle(
        motion.x() + gaussian(rng, sigmas[0]),
        motion.y() + gaussian(rng, sigmas[1]),
        motion.angle() + gaussian(rng, sigmas[2]),
    )
}

/// The measurements a robot would record, generated from a known ground truth.
struct Dataset {
    truth: Vec<SE2>,
    /// `(i, i + 1, measured T_i⁻¹·T_{i+1})`
    odometry: Vec<(usize, usize, SE2)>,
    /// `(earlier visit, current pose, measured relative pose)`
    loops: Vec<(usize, usize, SE2)>,
}

fn simulate(rng: &mut StdRng) -> Dataset {
    let steps = 4 * SIDE * LAPS;
    let mut truth = vec![SE2::identity()];
    for step in 1..=steps {
        // Drive one metre, and turn left at every corner.
        let turn = if step % SIDE == 0 {
            std::f64::consts::FRAC_PI_2
        } else {
            0.0
        };
        let previous = truth[step - 1].clone();
        truth.push(previous.compose(&SE2::from_xy_angle(1.0, 0.0, turn), None, None));
    }
    let relative = |a: usize, b: usize| truth[a].between(&truth[b], None, None);
    let odometry = (0..steps)
        .map(|i| {
            (
                i,
                i + 1,
                perturb(rng, &relative(i, i + 1), &ODOMETRY_SIGMAS),
            )
        })
        .collect();
    // One lap is `4·SIDE` poses: pose i revisits the place of pose i − 4·SIDE.
    let lap = 4 * SIDE;
    let loops = (lap..=steps)
        .step_by(3)
        .map(|i| {
            (
                i - lap,
                i,
                perturb(rng, &relative(i - lap, i), &LOOP_SIGMAS),
            )
        })
        .collect();
    Dataset {
        truth,
        odometry,
        loops,
    }
}

/// Initial guess: integrate the odometry (dead reckoning).
fn dead_reckoning(data: &Dataset) -> Vec<SE2> {
    let mut poses = vec![SE2::identity()];
    for (_, _, step) in &data.odometry {
        let last = poses[poses.len() - 1].clone();
        poses.push(last.compose(step, None, None));
    }
    poses
}

/// Absolute trajectory error: RMS position error against ground truth.
fn ate(estimate: &[SE2], truth: &[SE2]) -> f64 {
    let sum: f64 = estimate
        .iter()
        .zip(truth)
        .map(|(e, t)| (e.translation() - t.translation()).norm_squared())
        .sum();
    (sum / truth.len() as f64).sqrt()
}

#[derive(Clone, Copy)]
enum Gauge {
    /// A strong prior factor on pose 0 (soft constraint).
    Prior,
    /// Pose 0 removed from the optimization altogether (hard constraint).
    FixedVariable,
}

fn build(data: &Dataset, initial: &[SE2], gauge: Gauge) -> Result<(Problem, Vec<VarKey>)> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    // `add_variable` returns a `VarKey`: a small, copyable handle. Keep them
    // in your own index (here a Vec indexed by pose number).
    let keys: Vec<VarKey> = initial
        .iter()
        .map(|pose| {
            problem.add_variable(
                ManifoldType::SE2,
                DVector::from_column_slice(pose.as_param_slice()),
            )
        })
        .collect();

    // A noise model whitens a residual by its square-root information, so a
    // residual is measured in standard deviations. Build it once and clone
    // it per block.
    let odometry_noise = NoiseModel::from_sigmas(&ODOMETRY_SIGMAS)?;
    let loop_noise = NoiseModel::from_sigmas(&LOOP_SIGMAS)?;
    for (i, j, measurement) in &data.odometry {
        problem.add_residual_block_with_noise(
            &[keys[*i], keys[*j]],
            Box::new(BetweenFactor::new(measurement.clone())),
            None,
            odometry_noise.clone(),
        )?;
    }
    for (i, j, measurement) in &data.loops {
        problem.add_residual_block_with_noise(
            &[keys[*i], keys[*j]],
            Box::new(BetweenFactor::new(measurement.clone())),
            None,
            loop_noise.clone(),
        )?;
    }

    match gauge {
        Gauge::Prior => {
            problem.add_residual_block_with_noise(
                &[keys[0]],
                Box::new(PriorFactor::new(SE2::identity())),
                None,
                NoiseModel::from_sigmas(&[1e-3, 1e-3, 1e-4])?,
            )?;
        }
        Gauge::FixedVariable => {
            // Each call fixes one tangent coordinate; an SE2 has three.
            for coordinate in 0..3 {
                problem.fix_variable(keys[0], coordinate)?;
            }
        }
    }
    Ok((problem, keys))
}

/// Read the optimized poses back from a solver result, in pose order.
fn poses_from(
    keys: &[VarKey],
    values: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
) -> Result<Vec<SE2>> {
    keys.iter()
        .map(|key| {
            let value = values
                .get(*key)
                .ok_or("a pose is missing from the result")?;
            Ok(SE2::from_param_slice(value.as_param_slice()))
        })
        .collect()
}

fn main() -> Result<()> {
    let mut out = std::io::stdout().lock();
    let mut rng = StdRng::seed_from_u64(42);
    let data = simulate(&mut rng);
    let initial = dead_reckoning(&data);

    writeln!(
        out,
        "{} poses, {} odometry and {} loop-closure measurements",
        data.truth.len(),
        data.odometry.len(),
        data.loops.len()
    )?;
    writeln!(
        out,
        "dead-reckoning ATE      {:.3} m",
        ate(&initial, &data.truth)
    )?;

    // A preset tuned for 2-D pose graphs; every field can still be overridden.
    let config = LevenbergMarquardtConfig::for_2d_pose_graph();

    let mut maps = Vec::new();
    for (name, gauge) in [
        ("prior factor", Gauge::Prior),
        ("fixed variable", Gauge::FixedVariable),
    ] {
        let (mut problem, keys) = build(&data, &initial, gauge)?;
        let result = LevenbergMarquardt::with_config(config.clone()).optimize(&mut problem)?;
        let optimized = poses_from(&keys, &result.parameters)?;
        writeln!(
            out,
            "optimized ATE ({name:<14}) {:.3} m   {} iterations, cost {:.1} → {:.1}",
            ate(&optimized, &data.truth),
            result.iterations,
            result.initial_cost,
            result.final_cost
        )?;
        maps.push(optimized);
    }

    // The two anchors describe the same constraint, so the maps agree to the
    // precision of the (very stiff) prior.
    let disagreement = maps[0]
        .iter()
        .zip(&maps[1])
        .map(|(a, b)| (a.translation() - b.translation()).norm())
        .fold(0.0, f64::max);
    writeln!(
        out,
        "max difference between the two anchors  {disagreement:.2e} m"
    )?;
    if ate(&maps[0], &data.truth) >= ate(&initial, &data.truth) {
        return Err("optimization did not reduce the trajectory error".into());
    }
    Ok(())
}
