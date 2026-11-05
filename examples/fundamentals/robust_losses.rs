//! # 3 · Robust losses — surviving wrong measurements
//!
//! Place recognition makes mistakes. A single false loop closure tells the
//! optimizer that two distant places are the same, and plain least squares,
//! which squares every residual, bends the whole map to satisfy it.
//!
//! A robust loss `ρ(s)` replaces `½·s` (with `s = ‖S·r‖²`, the whitened
//! squared residual) by a function that grows slower for large `s`, so a
//! measurement many standard deviations off loses its influence. This example
//! adds 20 false loop closures to the square-lap pose graph of example 2 and
//! compares the losses side by side.
//!
//! ```bash
//! cargo run --release --example robust_losses
//! ```
//!
//! What to look for:
//! - **L2** (no loss) is pulled far from the truth.
//! - **Huber / Cauchy** keep growing, just slowly: robust and well behaved
//!   from a poor starting point.
//! - **DCS / Geman–McClure / Tukey** *redescend*: a gross outlier ends up with
//!   (almost) zero weight. Strongest rejection, but they are non-convex: from
//!   dead reckoning *every* loop closure looks like an outlier and they stay
//!   put. The last row shows the fix, a graduated solve (Cauchy first, then
//!   DCS warm-started from its answer).
//!
//! The loss is passed per residual block: odometry stays plain least squares,
//! only the loop closures, the measurements that can be wrong, get one.

use std::io::Write;

use apex_solver::core::VarKey;
use apex_solver::core::loss_functions::{
    CauchyLoss, DcsLoss, GemanMcClureLoss, HuberLoss, LossFunction, TukeyBiweightLoss,
};
use apex_solver::core::noise::NoiseModel;
use apex_solver::core::problem::Problem;
use apex_solver::factors::pose::{BetweenFactor, PriorFactor};
use apex_solver::{
    JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig, LieGroup, ManifoldType, SE2,
};
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const SIDE: usize = 10;
const LAPS: usize = 5;
const ODOMETRY_SIGMAS: [f64; 3] = [0.05, 0.05, 0.02];
const LOOP_SIGMAS: [f64; 3] = [0.02, 0.02, 0.005];
const FALSE_LOOPS: usize = 20;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
/// A constructor for one loss: each residual block owns its own instance.
type LossFactory = fn() -> Result<Option<Box<dyn LossFunction + Send + Sync>>>;

fn gaussian(rng: &mut StdRng, sigma: f64) -> f64 {
    let u1: f64 = rng.random_range(f64::EPSILON..1.0);
    let u2: f64 = rng.random();
    sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn perturb(rng: &mut StdRng, motion: &SE2, sigmas: &[f64; 3]) -> SE2 {
    SE2::from_xy_angle(
        motion.x() + gaussian(rng, sigmas[0]),
        motion.y() + gaussian(rng, sigmas[1]),
        motion.angle() + gaussian(rng, sigmas[2]),
    )
}

struct Dataset {
    truth: Vec<SE2>,
    odometry: Vec<(usize, usize, SE2)>,
    loops: Vec<(usize, usize, SE2)>,
}

fn simulate(rng: &mut StdRng) -> Dataset {
    let steps = 4 * SIDE * LAPS;
    let mut truth = vec![SE2::identity()];
    for step in 1..=steps {
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
    let lap = 4 * SIDE;
    let mut loops: Vec<_> = (lap..=steps)
        .step_by(3)
        .map(|i| {
            (
                i - lap,
                i,
                perturb(rng, &relative(i - lap, i), &LOOP_SIGMAS),
            )
        })
        .collect();
    // False positives: two random poses claimed to be at the same place.
    for _ in 0..FALSE_LOOPS {
        let a = rng.random_range(0..steps - 20);
        let b = rng.random_range(a + 15..=steps);
        let wrong = SE2::from_xy_angle(gaussian(rng, 0.1), gaussian(rng, 0.1), gaussian(rng, 0.05));
        loops.push((a, b, wrong));
    }
    Dataset {
        truth,
        odometry,
        loops,
    }
}

fn dead_reckoning(data: &Dataset) -> Vec<SE2> {
    let mut poses = vec![SE2::identity()];
    for (_, _, step) in &data.odometry {
        let last = poses[poses.len() - 1].clone();
        poses.push(last.compose(step, None, None));
    }
    poses
}

fn ate(estimate: &[SE2], truth: &[SE2]) -> f64 {
    let sum: f64 = estimate
        .iter()
        .zip(truth)
        .map(|(e, t)| (e.translation() - t.translation()).norm_squared())
        .sum();
    (sum / truth.len() as f64).sqrt()
}

/// Build the graph from `initial` with `loss` on every loop closure, solve it,
/// and return the optimized poses and the iteration count.
fn solve_with(data: &Dataset, initial: &[SE2], loss: LossFactory) -> Result<(Vec<SE2>, usize)> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let keys: Vec<VarKey> = initial
        .iter()
        .map(|pose| {
            problem.add_variable(
                ManifoldType::SE2,
                DVector::from_column_slice(pose.as_param_slice()),
            )
        })
        .collect();
    problem.add_residual_block_with_noise(
        &[keys[0]],
        Box::new(PriorFactor::new(SE2::identity())),
        None,
        NoiseModel::from_sigmas(&[1e-3, 1e-3, 1e-4])?,
    )?;
    let odometry_noise = NoiseModel::from_sigmas(&ODOMETRY_SIGMAS)?;
    for (i, j, measurement) in &data.odometry {
        problem.add_residual_block_with_noise(
            &[keys[*i], keys[*j]],
            Box::new(BetweenFactor::new(measurement.clone())),
            None,
            odometry_noise.clone(),
        )?;
    }
    let loop_noise = NoiseModel::from_sigmas(&LOOP_SIGMAS)?;
    for (i, j, measurement) in &data.loops {
        problem.add_residual_block_with_noise(
            &[keys[*i], keys[*j]],
            Box::new(BetweenFactor::new(measurement.clone())),
            loss()?,
            loop_noise.clone(),
        )?;
    }

    let config = LevenbergMarquardtConfig::for_2d_pose_graph().with_max_iterations(100);
    let result = LevenbergMarquardt::with_config(config).optimize(&mut problem)?;
    let poses = keys
        .iter()
        .map(|key| {
            let value = result
                .parameters
                .get(*key)
                .ok_or("a pose is missing from the result")?;
            Ok(SE2::from_param_slice(value.as_param_slice()))
        })
        .collect::<Result<Vec<SE2>>>()?;
    Ok((poses, result.iterations))
}

fn main() -> Result<()> {
    let mut out = std::io::stdout().lock();
    let mut rng = StdRng::seed_from_u64(42);
    let data = simulate(&mut rng);
    let initial = dead_reckoning(&data);

    writeln!(
        out,
        "{} poses, {} loop closures of which {FALSE_LOOPS} are false",
        data.truth.len(),
        data.loops.len()
    )?;
    writeln!(
        out,
        "dead-reckoning ATE {:.3} m\n",
        ate(&initial, &data.truth)
    )?;
    writeln!(
        out,
        "{:<22} {:>10} {:>11}",
        "loss on loop closures", "ATE [m]", "iterations"
    )?;

    // The scale is in standard deviations of the whitened residual: a closure
    // more than ~1–3 σ off starts to lose weight.
    let losses: [(&str, LossFactory); 6] = [
        ("L2 (none)", || Ok(None)),
        ("Huber(1.345)", || {
            Ok(Some(Box::new(HuberLoss::new(1.345)?)))
        }),
        ("Cauchy(1.0)", || Ok(Some(Box::new(CauchyLoss::new(1.0)?)))),
        ("DCS(1.0)", || Ok(Some(Box::new(DcsLoss::new(1.0)?)))),
        ("Geman-McClure(1.0)", || {
            Ok(Some(Box::new(GemanMcClureLoss::new(1.0)?)))
        }),
        ("Tukey(4.685)", || {
            Ok(Some(Box::new(TukeyBiweightLoss::new(4.685)?)))
        }),
    ];
    let mut l2 = f64::NAN;
    for (name, loss) in losses {
        let (poses, iterations) = solve_with(&data, &initial, loss)?;
        let error = ate(&poses, &data.truth);
        writeln!(out, "{name:<22} {error:>10.3} {iterations:>11}")?;
        if name.starts_with("L2") {
            l2 = error;
        }
    }

    // From dead reckoning, every loop closure — true or false — is many σ
    // off, so a redescending loss gives them all zero weight and the solver
    // stays at the initial guess. The standard remedy is graduated: solve
    // with a convex-ish loss first, then refine with the redescending one
    // starting from that answer.
    let (cauchy, first) = solve_with(&data, &initial, || {
        Ok(Some(Box::new(CauchyLoss::new(1.0)?)))
    })?;
    let (refined, second) = solve_with(&data, &cauchy, || Ok(Some(Box::new(DcsLoss::new(1.0)?))))?;
    let staged = ate(&refined, &data.truth);
    writeln!(
        out,
        "{:<22} {staged:>10.3} {:>11}",
        "Cauchy, then DCS",
        first + second
    )?;
    if staged >= l2 {
        return Err("the staged robust solve did not beat plain least squares".into());
    }
    Ok(())
}
