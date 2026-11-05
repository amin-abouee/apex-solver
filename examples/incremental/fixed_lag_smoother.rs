//! # 11 · A fixed-lag smoother — constant cost per step, by marginalization
//!
//! Example 10's problem grows with every pose, and so does each solve. A
//! **fixed-lag smoother** keeps only the last `WINDOW` poses. When a new pose
//! arrives and the window is full, the oldest pose must leave. There are two
//! ways to let it go:
//!
//! - **drop** it: delete the pose and its factors, and fix the new oldest pose
//!   so the window keeps an anchor. Cheap, but everything those factors said
//!   about the poses that stay is thrown away;
//! - **marginalize** it: [`Marginalizer::apply`] eliminates the pose exactly
//!   and leaves a Gaussian *prior* on the poses it was connected to (see the
//!   cookbook's marginalization chapter). Nothing is lost at the moment of
//!   removal; the prior is linear, so it is exact only at the estimate it was
//!   built at.
//!
//! The third variant adds **first-estimate Jacobians** (FEJ): the poses a
//! prior touches keep their Jacobians at the point the prior was built, so the
//! prior and the factors around it agree about where they were linearized.
//! That keeps the estimator *consistent*: it does not grow confident in
//! directions nothing measured.
//!
//! The data is a simulated visual-odometry-like stream: every pose is related
//! to its three predecessors (`i−1`, `i−2`, `i−3`), so a window of 10 sees
//! every measurement. The batch solution of the whole stream is the best any
//! of them can do; a smoother that marginalizes should land close to it, one
//! that drops should not.
//!
//! ```bash
//! cargo run --release --example fixed_lag_smoother
//! ```

use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::time::{Duration, Instant};

use apex_solver::core::marginalization::Marginalizer;
use apex_solver::core::noise::NoiseModel;
use apex_solver::core::problem::Problem;
use apex_solver::core::{FactorKey, VarKey};
use apex_solver::factors::pose::{BetweenFactor, PriorFactor};
use apex_solver::{
    JacobianEvaluation, JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig, LieGroup,
    ManifoldType, SE2,
};
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const POSES: usize = 1000;
const WINDOW: usize = 10;
/// Each pose is measured against this many predecessors.
const SPAN: usize = 3;
/// Noise of a one-step relative measurement; a k-step one has √k times it.
const STEP_SIGMAS: [f64; 3] = [0.05, 0.05, 0.01];
const ITERATIONS_PER_STEP: usize = 5;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn gaussian(rng: &mut StdRng, sigma: f64) -> f64 {
    let u1: f64 = rng.random_range(f64::EPSILON..1.0);
    let u2: f64 = rng.random();
    sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// One relative measurement `T_from⁻¹·T_to`.
struct Edge {
    from: usize,
    to: usize,
    measured: SE2,
    sigmas: [f64; 3],
}

struct Stream {
    truth: Vec<SE2>,
    /// Indexed by the younger end: `edges[i]` arrive together with pose `i`.
    edges: Vec<Vec<Edge>>,
}

fn simulate(rng: &mut StdRng) -> Stream {
    let mut truth = vec![SE2::identity()];
    for i in 1..POSES {
        let turn = 0.15 * (i as f64 / 25.0).sin();
        let previous = truth[i - 1].clone();
        truth.push(previous.compose(&SE2::from_xy_angle(1.0, 0.0, turn), None, None));
    }
    let edges = (0..POSES)
        .map(|to| {
            (to.saturating_sub(SPAN)..to)
                .map(|from| {
                    let scale = ((to - from) as f64).sqrt();
                    let sigmas = STEP_SIGMAS.map(|sigma| sigma * scale);
                    let exact = truth[from].between(&truth[to], None, None);
                    let measured = SE2::from_xy_angle(
                        exact.x() + gaussian(rng, sigmas[0]),
                        exact.y() + gaussian(rng, sigmas[1]),
                        exact.angle() + gaussian(rng, sigmas[2]),
                    );
                    Edge {
                        from,
                        to,
                        measured,
                        sigmas,
                    }
                })
                .collect()
        })
        .collect();
    Stream { truth, edges }
}

fn ate(estimate: &[SE2], reference: &[SE2]) -> f64 {
    let sum: f64 = estimate
        .iter()
        .zip(reference)
        .map(|(a, b)| (a.translation() - b.translation()).norm_squared())
        .sum();
    (sum / reference.len() as f64).sqrt()
}

#[derive(Clone, Copy, PartialEq)]
enum Policy {
    Drop,
    Marginalize,
    MarginalizeFej,
}

/// What a run produced: the final estimate of every pose and the timing.
struct Run {
    poses: Vec<SE2>,
    total: Duration,
    slowest_step: Duration,
}

fn pose_of(problem: &Problem, key: VarKey) -> Result<SE2> {
    Ok(SE2::from_param_slice(
        problem
            .variable_params(key)
            .ok_or("a window pose is missing")?,
    ))
}

fn add_edge(
    problem: &mut Problem,
    keys: &HashMap<usize, VarKey>,
    edge: &Edge,
) -> Result<FactorKey> {
    Ok(problem.add_residual_block_with_noise(
        &[keys[&edge.from], keys[&edge.to]],
        Box::new(BetweenFactor::new(edge.measured.clone())),
        None,
        NoiseModel::from_sigmas(&edge.sigmas)?,
    )?)
}

/// The reference: every pose and every measurement in one problem.
fn batch(stream: &Stream) -> Result<Run> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut keys = HashMap::new();
    let mut guess = SE2::identity();
    for i in 0..POSES {
        if let Some(odometry) = stream.edges[i].iter().find(|edge| edge.from + 1 == i) {
            guess = guess.compose(&odometry.measured, None, None);
        }
        keys.insert(
            i,
            problem.add_variable(
                ManifoldType::SE2,
                DVector::from_column_slice(guess.as_param_slice()),
            ),
        );
    }
    problem.add_residual_block_with_noise(
        &[keys[&0]],
        Box::new(PriorFactor::new(SE2::identity())),
        None,
        NoiseModel::from_sigmas(&[1e-3, 1e-3, 1e-4])?,
    )?;
    for edge in stream.edges.iter().flatten() {
        add_edge(&mut problem, &keys, edge)?;
    }
    let start = Instant::now();
    let result = LevenbergMarquardt::with_config(LevenbergMarquardtConfig::for_2d_pose_graph())
        .optimize(&mut problem)?;
    problem.update_values_from(&result.parameters)?;
    let poses = (0..POSES)
        .map(|i| pose_of(&problem, keys[&i]))
        .collect::<Result<_>>()?;
    let total = start.elapsed();
    Ok(Run {
        poses,
        total,
        slowest_step: total,
    })
}

fn smooth(stream: &Stream, policy: Policy) -> Result<Run> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    if policy == Policy::MarginalizeFej {
        problem.set_jacobian_evaluation(JacobianEvaluation::FirstEstimate);
    }
    let marginalizer = Marginalizer::new();
    let config =
        LevenbergMarquardtConfig::for_2d_pose_graph().with_max_iterations(ITERATIONS_PER_STEP);

    let mut keys: HashMap<usize, VarKey> = HashMap::new();
    let mut window: VecDeque<usize> = VecDeque::new();
    // The blocks touching each pose: only the drop policy needs them,
    // because it deletes factors itself. The problem does not index blocks
    // by variable, so the application keeps the bookkeeping.
    let mut blocks_of: HashMap<usize, Vec<FactorKey>> = HashMap::new();
    let mut finished: Vec<Option<SE2>> = vec![None; POSES];
    let mut total = Duration::ZERO;
    let mut slowest_step = Duration::ZERO;

    for i in 0..POSES {
        let start = Instant::now();

        // A new pose, predicted from the window's newest estimate.
        let prediction = match stream.edges[i].iter().find(|edge| edge.from + 1 == i) {
            Some(odometry) => {
                pose_of(&problem, keys[&odometry.from])?.compose(&odometry.measured, None, None)
            }
            None => SE2::identity(),
        };
        let key = problem.add_variable(
            ManifoldType::SE2,
            DVector::from_column_slice(prediction.as_param_slice()),
        );
        keys.insert(i, key);
        window.push_back(i);
        if i == 0 {
            // A prior rather than fixed coordinates: the marginalizer can
            // absorb a prior, and passes the anchor on to the next prior.
            let anchor = problem.add_residual_block_with_noise(
                &[key],
                Box::new(PriorFactor::new(SE2::identity())),
                None,
                NoiseModel::from_sigmas(&[1e-3, 1e-3, 1e-4])?,
            )?;
            blocks_of.entry(0).or_default().push(anchor);
        }
        for edge in &stream.edges[i] {
            // An edge back to a pose that has left the window is lost. With
            // WINDOW > SPAN that never happens here.
            if !keys.contains_key(&edge.from) {
                continue;
            }
            let block = add_edge(&mut problem, &keys, edge)?;
            blocks_of.entry(edge.from).or_default().push(block);
            blocks_of.entry(edge.to).or_default().push(block);
        }

        // Solve the window, warm-started, and keep the answer.
        let result = LevenbergMarquardt::with_config(config.clone()).optimize(&mut problem)?;
        problem.update_values_from(&result.parameters)?;

        // Let the oldest pose go once the window is over-full.
        if window.len() > WINDOW {
            let oldest = window.pop_front().ok_or("the window is empty")?;
            let oldest_key = keys.remove(&oldest).ok_or("the oldest pose has no key")?;
            finished[oldest] = Some(pose_of(&problem, oldest_key)?);
            match policy {
                Policy::Drop => {
                    let blocks = blocks_of.remove(&oldest).unwrap_or_default();
                    problem.remove_residual_blocks(&blocks);
                    problem.try_remove_variable(oldest_key)?;
                    // The new oldest pose becomes the window's anchor.
                    let anchor = window
                        .front()
                        .and_then(|id| keys.get(id))
                        .ok_or("the window is empty")?;
                    for coordinate in 0..3 {
                        problem.fix_variable(*anchor, coordinate)?;
                    }
                }
                Policy::Marginalize | Policy::MarginalizeFej => {
                    let applied = marginalizer.apply(&mut problem, &[oldest_key])?;
                    if policy == Policy::MarginalizeFej {
                        for kept in &applied.kept {
                            if !problem.is_linearization_point_frozen(*kept) {
                                problem.freeze_linearization_point(*kept)?;
                            }
                        }
                    }
                }
            }
        }
        let step = start.elapsed();
        total += step;
        slowest_step = slowest_step.max(step);
    }
    for id in window {
        finished[id] = Some(pose_of(&problem, keys[&id])?);
    }
    let poses = finished
        .into_iter()
        .map(|pose| pose.ok_or("a pose was never estimated"))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(Run {
        poses,
        total,
        slowest_step,
    })
}

fn main() -> Result<()> {
    let mut out = std::io::stdout().lock();
    let stream = simulate(&mut StdRng::seed_from_u64(5));
    let reference = batch(&stream)?;

    writeln!(
        out,
        "{POSES} poses, window {WINDOW}, {} measurements\n",
        stream.edges.iter().map(Vec::len).sum::<usize>()
    )?;
    writeln!(
        out,
        "{:<26} {:>12} {:>14} {:>11} {:>14}",
        "estimator", "ATE vs truth", "diff vs batch", "total [ms]", "max step [ms]"
    )?;
    let mut report = |label: &str, run: &Run| -> Result<()> {
        writeln!(
            out,
            "{label:<26} {:>10.3} m {:>12.3} m {:>11.0} {:>14.2}",
            ate(&run.poses, &stream.truth),
            ate(&run.poses, &reference.poses),
            1e3 * run.total.as_secs_f64(),
            1e3 * run.slowest_step.as_secs_f64()
        )?;
        Ok(())
    };
    report("batch (everything)", &reference)?;
    let drop = smooth(&stream, Policy::Drop)?;
    report("window, drop oldest", &drop)?;
    let marginal = smooth(&stream, Policy::Marginalize)?;
    report("window, marginalize", &marginal)?;
    let fej = smooth(&stream, Policy::MarginalizeFej)?;
    report("window, marginalize + FEJ", &fej)?;

    if ate(&marginal.poses, &reference.poses) >= ate(&drop.poses, &reference.poses) {
        return Err(
            "marginalization did not stay closer to the batch solution than dropping".into(),
        );
    }
    Ok(())
}
