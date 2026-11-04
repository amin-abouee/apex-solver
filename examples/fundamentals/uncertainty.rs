//! # 4 · Uncertainty — how sure is the estimate?
//!
//! An optimizer returns a point estimate; the covariance says how far to
//! trust it. For whitened residuals (every block has a noise model) the
//! covariance is the inverse of the Gauss–Newton Hessian `JᵀJ` at the
//! solution, and apex-solver computes it on request:
//!
//! ```text
//! LevenbergMarquardtConfig::new().with_compute_covariances(true)
//! result.covariances  →  one tangent-space matrix per variable
//! ```
//!
//! The scene is one lap of the square from example 2. Along an odometry chain
//! uncertainty accumulates (it dips near the end only because a heading error
//! acts through a lever arm, and the path turns back towards the start); a
//! loop closure back to the start pins the end of the lap down again. The table prints the 1-σ position uncertainty of every
//! tenth pose, without and with that closure.
//!
//! Two details worth knowing:
//! - The covariance lives in the **tangent space** of each variable (for `SE2`
//!   `[x, y, θ]` in the pose's own frame), not in global coordinates.
//! - Damping and Jacobi scaling do not affect it: the Hessian is rebuilt
//!   without them at the final estimate.
//!
//! ```bash
//! cargo run --release --example uncertainty
//! ```

use std::io::Write;

use apex_solver::core::VarKey;
use apex_solver::core::noise::NoiseModel;
use apex_solver::core::problem::Problem;
use apex_solver::factors::pose::{BetweenFactor, PriorFactor};
use apex_solver::{
    JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig, LieGroup, ManifoldType, SE2,
};
use nalgebra::DVector;

const SIDE: usize = 10;
const ODOMETRY_SIGMAS: [f64; 3] = [0.05, 0.05, 0.02];
const LOOP_SIGMAS: [f64; 3] = [0.02, 0.02, 0.005];

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// One lap: `4·SIDE` one-metre steps with a left turn at every corner.
fn one_lap() -> Vec<SE2> {
    let mut poses = vec![SE2::identity()];
    for step in 1..=4 * SIDE {
        let turn = if step % SIDE == 0 {
            std::f64::consts::FRAC_PI_2
        } else {
            0.0
        };
        let previous = poses[step - 1].clone();
        poses.push(previous.compose(&SE2::from_xy_angle(1.0, 0.0, turn), None, None));
    }
    poses
}

/// The 1-σ position uncertainty of every pose, with or without the loop closure.
fn position_sigmas(poses: &[SE2], close_the_loop: bool) -> Result<Vec<f64>> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let keys: Vec<VarKey> = poses
        .iter()
        .map(|pose| {
            problem.add_variable(
                ManifoldType::SE2,
                DVector::from_column_slice(pose.as_param_slice()),
            )
        })
        .collect();
    // The prior's σ is the uncertainty of the first pose itself: everything
    // else is measured relative to it.
    problem.add_residual_block_with_noise(
        &[keys[0]],
        Box::new(PriorFactor::new(poses[0].clone())),
        None,
        NoiseModel::from_sigmas(&[0.01, 0.01, 0.001])?,
    )?;
    let odometry = NoiseModel::from_sigmas(&ODOMETRY_SIGMAS)?;
    for i in 0..poses.len() - 1 {
        let step = poses[i].between(&poses[i + 1], None, None);
        problem.add_residual_block_with_noise(
            &[keys[i], keys[i + 1]],
            Box::new(BetweenFactor::new(step)),
            None,
            odometry.clone(),
        )?;
    }
    if close_the_loop {
        let last = poses.len() - 1;
        let closure = poses[0].between(&poses[last], None, None);
        problem.add_residual_block_with_noise(
            &[keys[0], keys[last]],
            Box::new(BetweenFactor::new(closure)),
            None,
            NoiseModel::from_sigmas(&LOOP_SIGMAS)?,
        )?;
    }

    let config = LevenbergMarquardtConfig::new().with_compute_covariances(true);
    let result = LevenbergMarquardt::with_config(config).optimize(&mut problem)?;
    let covariances = result.covariances.ok_or("covariances were not computed")?;
    keys.iter()
        .map(|key| {
            let sigma = covariances.get(*key).ok_or("a pose has no covariance")?;
            // Tangent order is [x, y, θ]: the top-left 2×2 block is position.
            Ok((sigma[(0, 0)] + sigma[(1, 1)]).sqrt())
        })
        .collect()
}

fn main() -> Result<()> {
    let mut out = std::io::stdout().lock();
    let lap = one_lap();
    let open = position_sigmas(&lap, false)?;
    let closed = position_sigmas(&lap, true)?;

    writeln!(out, "1-σ position uncertainty [cm] along one lap\n")?;
    writeln!(
        out,
        "{:>5} {:>12} {:>14}",
        "pose", "odometry", "+ loop closure"
    )?;
    for index in (0..lap.len()).step_by(SIDE / 2) {
        writeln!(
            out,
            "{index:>5} {:>12.1} {:>14.1}",
            100.0 * open[index],
            100.0 * closed[index]
        )?;
    }

    // The last pose returns to the start: with the closure it inherits (almost)
    // the start's certainty, without it it carries a whole lap of drift.
    let last = lap.len() - 1;
    writeln!(
        out,
        "\nend of the lap: {:.1} cm → {:.1} cm with the closure",
        100.0 * open[last],
        100.0 * closed[last]
    )?;
    if closed[last] >= open[last] {
        return Err("the loop closure did not reduce the uncertainty".into());
    }
    Ok(())
}
