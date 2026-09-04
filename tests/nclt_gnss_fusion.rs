//! Odometry + GNSS fusion on a real public dataset.
//!
//! Everything else in the suite runs on a synthetic trajectory, where the
//! measurements are generated from the same models the factors implement. That
//! catches modelling and Jacobian errors but cannot say whether the factors
//! behave on data that was actually recorded — with real noise, dropouts, and
//! sensor biases nobody chose.
//!
//! This runs a pose graph over the
//! [NCLT dataset](https://robots.engin.umich.edu/nclt/) (University of Michigan
//! North Campus Long-Term), session `2013-01-10`: a Segway driving a ~500 m
//! campus loop for about 17 minutes, carrying wheel/gyro odometry, consumer
//! GNSS, and laser-fused ground truth.
//!
//! * [`BetweenFactor`] over the real odometry increments — accurate locally,
//!   drifting to ~44 m by the end of the session on its own.
//! * [`PoseTranslationPrior`] at each GNSS fix — unbiased but noisy, ~6 m
//!   median against ground truth, and absent wherever the receiver drops to a
//!   2D fix.
//!
//! Fusing them should keep the odometry's local accuracy while removing its
//! drift, which is what the test asserts. NCLT publishes each stream
//! separately, so only ~40 MB of CSV is downloaded — not the ~100 GB of
//! imagery. The download happens once, on first run, like the other
//! dataset-backed tests.

// The run summary below is the point of a real-data test: the numbers say how
// far the fusion actually got, not merely that an assertion held.
#![allow(clippy::print_stderr)]

mod common;
use common::nclt;

use apex_solver::JacobianMode;
use apex_solver::apex_manifolds::LieGroup;
use apex_solver::apex_manifolds::ManifoldType;
use apex_solver::apex_manifolds::se3::SE3;
use apex_solver::core::noise::NoiseModel;
use apex_solver::core::problem::Problem;
use apex_solver::factors::pose::{BetweenFactor, PoseTranslationPrior, PriorFactor};
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DVector, Vector3};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Odometry epochs per keyframe. The stream is ~5 Hz; five epochs makes a
/// ~1 s keyframe interval and a graph of ~1000 poses.
const STRIDE: usize = 5;

#[test]
fn nclt_odometry_and_gnss_fusion_removes_drift() -> TestResult {
    let dir = apex_io::ensure_sensor_dataset("nclt-2013-01-10")?;
    let ground_truth = nclt::read_ground_truth(&dir.join("groundtruth.csv"))?;
    let odometry = nclt::read_odometry(&dir.join("2013-01-10/odometry_mu.csv"))?;
    let gps = nclt::read_gps(&dir.join("2013-01-10/gps.csv"))?;

    assert!(
        ground_truth.len() > 100_000 && odometry.len() > 4_000 && gps.len() > 1_000,
        "unexpected dataset size: gt={} odo={} gps={}",
        ground_truth.len(),
        odometry.len(),
        gps.len()
    );

    // ── Keyframes: compose the odometry increments in groups of STRIDE ───────
    let mut keyframe_times = Vec::new();
    let mut between: Vec<SE3> = Vec::new();
    let mut accumulated = SE3::identity();
    keyframe_times.push(odometry[0].utime);
    for (i, step) in odometry.iter().enumerate() {
        accumulated = accumulated.compose(&step.delta, None, None);
        if (i + 1) % STRIDE == 0 {
            between.push(accumulated.clone());
            keyframe_times.push(step.utime);
            accumulated = SE3::identity();
        }
    }
    let n = keyframe_times.len();
    assert_eq!(between.len(), n - 1);

    // ── Dead reckoning: the initial guess, and the baseline to beat ──────────
    let start = nclt::nearest_pose(&ground_truth, keyframe_times[0])
        .pose
        .clone();
    let mut dead_reckoned = Vec::with_capacity(n);
    dead_reckoned.push(start);
    for delta in &between {
        let previous = dead_reckoned
            .last()
            .unwrap_or_else(|| unreachable!("seeded above"));
        dead_reckoned.push(previous.compose(delta, None, None));
    }

    // ── GNSS fixes, projected into the ground-truth frame ────────────────────
    let offset = nclt::local_offset(&gps, &ground_truth);
    // One fix per keyframe: the nearest in time, if it is close enough.
    let mut fix_for_keyframe: Vec<Option<Vector3<f64>>> = vec![None; n];
    for fix in &gps {
        let idx = keyframe_times.partition_point(|t| *t < fix.utime);
        let idx = idx.min(n - 1);
        if (keyframe_times[idx] - fix.utime).abs() > 600_000 {
            continue; // no keyframe within 0.6 s
        }
        let truth_z = nclt::nearest_pose(&ground_truth, fix.utime)
            .pose
            .translation()
            .z;
        fix_for_keyframe[idx] = Some(Vector3::new(
            fix.north_east.0 + offset.0,
            fix.north_east.1 + offset.1,
            truth_z, // the receiver's altitude is unusable; z is held, not tested
        ));
    }
    let fix_count = fix_for_keyframe.iter().filter(|f| f.is_some()).count();
    assert!(
        fix_count > n / 4,
        "only {fix_count} of {n} keyframes received a GNSS fix"
    );

    // ── Build and solve ──────────────────────────────────────────────────────
    let mut problem = Problem::new(JacobianMode::Sparse);
    let keys: Vec<_> = dead_reckoned
        .iter()
        .map(|p| {
            problem.add_variable(
                ManifoldType::SE3,
                DVector::from_column_slice(p.as_param_slice()),
            )
        })
        .collect();

    // Odometry: locally accurate, so tight. Heading drifts fastest, hence the
    // looser yaw term.
    let odometry_noise = NoiseModel::from_sigmas(&[0.05, 0.05, 0.05, 0.01, 0.01, 0.02])?;
    for (i, delta) in between.iter().enumerate() {
        problem.add_residual_block_with_noise(
            &[keys[i], keys[i + 1]],
            Box::new(BetweenFactor::new(delta.clone())),
            None,
            odometry_noise.clone(),
        );
    }

    // GNSS: unbiased but noisy. The z term is slack because the altitude here
    // is the held ground-truth value, not a measurement.
    let gnss_noise = NoiseModel::from_sigmas(&[6.0, 6.0, 1.0e3])?;
    for (i, fix) in fix_for_keyframe.iter().enumerate() {
        if let Some(position) = fix {
            problem.add_residual_block_with_noise(
                &[keys[i]],
                Box::new(PoseTranslationPrior::new(*position)),
                None,
                gnss_noise.clone(),
            );
        }
    }

    // Position-only measurements leave absolute heading weakly observable;
    // anchor the first pose, as an initial alignment would.
    problem.add_residual_block_with_noise(
        &[keys[0]],
        Box::new(PriorFactor::new(dead_reckoned[0].clone())),
        None,
        NoiseModel::from_sigmas(&[0.1, 0.1, 0.1, 0.01, 0.01, 0.01])?,
    );

    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(50)
        .with_cost_tolerance(1e-8)
        .with_parameter_tolerance(1e-8);
    let result = LevenbergMarquardt::with_config(config).optimize(&mut problem)?;

    // ── Compare both trajectories against ground truth ───────────────────────
    let horizontal_error = |pose: &SE3, utime: i64| {
        let truth = nclt::nearest_pose(&ground_truth, utime).pose.translation();
        let p = pose.translation();
        ((p.x - truth.x).powi(2) + (p.y - truth.y).powi(2)).sqrt()
    };

    let mut odometry_errors: Vec<f64> = dead_reckoned
        .iter()
        .zip(&keyframe_times)
        .map(|(p, t)| horizontal_error(p, *t))
        .collect();
    let mut fused_errors: Vec<f64> = keys
        .iter()
        .zip(&keyframe_times)
        .map(|(k, t)| {
            let pose = SE3::from_param_slice(result.parameters[*k].as_param_slice());
            horizontal_error(&pose, *t)
        })
        .collect();

    let odometry_final = *odometry_errors
        .last()
        .unwrap_or_else(|| unreachable!("non-empty"));
    let fused_final = *fused_errors
        .last()
        .unwrap_or_else(|| unreachable!("non-empty"));
    let odometry_median = nclt::median(&mut odometry_errors);
    let fused_median = nclt::median(&mut fused_errors);

    eprintln!(
        "NCLT 2013-01-10: {n} keyframes, {fix_count} GNSS fixes\n\
         cost {:.4e} -> {:.4e}\n\
         odometry  median {odometry_median:.1} m  final {odometry_final:.1} m\n\
         fused     median {fused_median:.1} m  final {fused_final:.1} m",
        result.initial_cost, result.final_cost
    );

    assert!(
        result.final_cost < result.initial_cost,
        "optimizer must reduce cost: {:.6e} -> {:.6e}",
        result.initial_cost,
        result.final_cost
    );

    // Dead reckoning accumulates error without bound; fusing GNSS must not.
    assert!(
        fused_final < odometry_final * 0.5,
        "fusion should cut end-of-run drift: odometry {odometry_final:.1} m, \
         fused {fused_final:.1} m"
    );
    assert!(
        fused_median < odometry_median,
        "fusion should beat odometry overall: odometry {odometry_median:.1} m, \
         fused {fused_median:.1} m"
    );
    // The result should sit at the GNSS noise floor, not merely be "better".
    assert!(
        fused_median < 12.0,
        "fused median error {fused_median:.1} m is above the GNSS noise floor"
    );
    Ok(())
}
