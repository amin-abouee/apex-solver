//! Tests for the four SGal(3) IMU factors.
//!
//! The predecessors of these tests were `#[ignore]`d ("tangent Jacobian chain
//! under investigation"). They are enabled here, and they linearize *away* from
//! ground truth — the point at which the previous SE_2(3) tests were unable to
//! see a Jacobian error at all.

use apex_manifolds::se3::{SE3, SE3Tangent};
use apex_manifolds::sgal3::{SGal3, SGal3Tangent};
use apex_manifolds::{LieGroup, Tangent, rn::Rn};
use nalgebra::{DMatrix, DVector, Vector3};

use super::factors::{
    Sgal3CombinedImuFactor, Sgal3CombinedStateImuFactor, Sgal3ImuFactor, Sgal3StateImuFactor,
};
use crate::core::variable::{ManifoldVariable, Variable};
use crate::factors::Factor;
use crate::factors::inertial::preintegration::ImuPreintegration;
use crate::factors::inertial::types::{
    ImuMeasurement, ImuParameters, ImuSensorReadings, SpeedAndBias, SpeedAndBiasExt,
};

const FD_EPS: f64 = 1e-5;
/// See the SE_2(3) tests for why this is 1e-3: the floor is the `Q`-block
/// coupling of the group's right Jacobians, not the factor code.
const FD_TOL: f64 = 1e-3;

fn euroc_params() -> ImuParameters {
    ImuParameters {
        sigma_g_c: 1.6968e-04,
        sigma_a_c: 2.0000e-03,
        sigma_gw_c: 1.9393e-05,
        sigma_aw_c: 3.0000e-03,
        g: 9.81,
        ..ImuParameters::default()
    }
}

/// `(preintegration, pose_j, speed_and_bias_j, Δt)` for a yawing interval.
fn scenario() -> (ImuPreintegration, SE3, SpeedAndBias, f64) {
    let params = euroc_params();
    let g = params.g;
    let dt_step = 0.005_f64;
    let n = 201_usize;
    let t1 = (n - 1) as f64 * dt_step;

    let measurements: Vec<_> = (0..n)
        .map(|i| {
            ImuMeasurement::new(
                i as f64 * dt_step,
                ImuSensorReadings {
                    gyroscopes: Vector3::new(0.0, 0.0, 0.1),
                    accelerometers: Vector3::new(0.0, 0.0, g),
                },
            )
        })
        .collect();

    let sb_zero = SpeedAndBias::zeros();
    let preint = ImuPreintegration::new(measurements.clone(), params.clone(), 0.0, t1, &sb_zero);

    let mut pose_j = SE3::identity();
    let mut sb_j = SpeedAndBias::zeros();
    ImuPreintegration::propagation(&measurements, &params, &mut pose_j, &mut sb_j, 0.0, t1);

    (preint, pose_j, sb_j, t1)
}

fn pose_blocks(p: &SE3) -> DVector<f64> {
    DVector::from_column_slice(p.as_param_slice())
}

fn vec3(v: Vector3<f64>) -> DVector<f64> {
    DVector::from_vec(vec![v.x, v.y, v.z])
}

/// The `SGal3` state equivalent to a `(pose, velocity, time)` triple.
fn sgal3_of(pose: &SE3, v: Vector3<f64>, t: f64) -> DVector<f64> {
    let s = SGal3::new(pose.translation(), v, pose.rotation_quaternion(), t);
    DVector::from_column_slice(s.as_param_slice())
}

fn residual_of<F: Factor>(f: &F, params: &[&[f64]]) -> Vec<f64> {
    let mut r = vec![0.0f64; f.residual_dim()];
    f.linearize(params, &mut r, None);
    r
}

#[derive(Clone, Copy)]
enum Chart {
    Se3,
    Sgal3,
    Euclidean,
}

impl Chart {
    fn dof(self, block: &[f64]) -> usize {
        match self {
            Chart::Se3 => 6,
            Chart::Sgal3 => 10,
            Chart::Euclidean => block.len(),
        }
    }

    fn perturb(self, block: &[f64], k: usize, eps: f64) -> DVector<f64> {
        match self {
            Chart::Se3 => {
                let mut t = [0.0f64; 6];
                t[k] = eps;
                let m = SE3::from_param_slice(block).right_plus(
                    &SE3Tangent::from_slice(&t),
                    None,
                    None,
                );
                DVector::from_column_slice(m.as_param_slice())
            }
            Chart::Sgal3 => {
                let mut t = [0.0f64; 10];
                t[k] = eps;
                let m = SGal3::from_param_slice(block).right_plus(
                    &SGal3Tangent::from_slice(&t),
                    None,
                    None,
                );
                DVector::from_column_slice(m.as_param_slice())
            }
            Chart::Euclidean => {
                let mut out = DVector::from_column_slice(block);
                out[k] += eps;
                out
            }
        }
    }
}

fn check_jacobian<F: Factor>(f: &F, blocks: &[DVector<f64>], charts: &[Chart], label: &str) {
    let params: Vec<&[f64]> = blocks.iter().map(|b| b.as_slice()).collect();
    let (rows, cols) = f.jacobian_shape();
    let mut r = vec![0.0; rows];
    let mut buf = vec![0.0; rows * cols];
    let jm = faer::mat::MatMut::from_column_major_slice_mut(&mut buf, rows, cols);
    f.linearize(&params, &mut r, Some(jm));
    let analytic = DMatrix::from_column_slice(rows, cols, &buf);

    let mut col = 0usize;
    for (b, &chart) in charts.iter().enumerate() {
        for k in 0..chart.dof(blocks[b].as_slice()) {
            let mut p_plus = blocks.to_vec();
            p_plus[b] = chart.perturb(blocks[b].as_slice(), k, FD_EPS);
            let mut p_minus = blocks.to_vec();
            p_minus[b] = chart.perturb(blocks[b].as_slice(), k, -FD_EPS);

            let rp = residual_of(f, &p_plus.iter().map(|v| v.as_slice()).collect::<Vec<_>>());
            let rm = residual_of(f, &p_minus.iter().map(|v| v.as_slice()).collect::<Vec<_>>());

            for row in 0..rows {
                let fd = (rp[row] - rm[row]) / (2.0 * FD_EPS);
                let a = analytic[(row, col)];
                let scale = 1.0 + a.abs().max(fd.abs());
                assert!(
                    (a - fd).abs() / scale < FD_TOL,
                    "{label}: block {b} dof {k} (col {col}) row {row}: \
                     analytic={a:.8} fd={fd:.8}"
                );
            }
            col += 1;
        }
    }
    assert_eq!(col, cols, "{label}: column count");
}

// ── Shapes ───────────────────────────────────────────────────────────────────

#[test]
fn split_block_factors_drop_the_time_row() {
    let (preint, _, _, _) = scenario();
    let a = Sgal3ImuFactor::new(preint.clone());
    let b = Sgal3CombinedImuFactor::new(preint);
    assert_eq!(a.residual_dim(), 9);
    assert_eq!(a.jacobian_shape(), (9, 24));
    assert_eq!(b.residual_dim(), 15);
    assert_eq!(b.jacobian_shape(), (15, 30));
}

#[test]
fn native_state_factors_carry_the_time_row() {
    let (preint, _, _, _) = scenario();
    let a = Sgal3StateImuFactor::new(preint.clone());
    let b = Sgal3CombinedStateImuFactor::new(preint);
    assert_eq!(a.residual_dim(), 10);
    assert_eq!(a.jacobian_shape(), (10, 26));
    assert_eq!(b.residual_dim(), 16);
    assert_eq!(b.jacobian_shape(), (16, 32));
}

// ── Zero residual at ground truth ────────────────────────────────────────────

#[test]
fn split_block_residuals_vanish_at_ground_truth() {
    let (preint, pose_j, sb_j, _) = scenario();
    let pi = pose_blocks(&SE3::identity());
    let vi = DVector::zeros(3);
    let pj = pose_blocks(&pose_j);
    let vj = vec3(sb_j.velocity());
    let b = DVector::zeros(6);

    let f = Sgal3ImuFactor::new(preint.clone());
    for (i, v) in residual_of(
        &f,
        &[
            pi.as_slice(),
            vi.as_slice(),
            pj.as_slice(),
            vj.as_slice(),
            b.as_slice(),
        ],
    )
    .iter()
    .enumerate()
    {
        assert!(v.abs() < 1e-6, "Sgal3ImuFactor residual[{i}] = {v:.3e}");
    }

    let f = Sgal3CombinedImuFactor::new(preint);
    for (i, v) in residual_of(
        &f,
        &[
            pi.as_slice(),
            vi.as_slice(),
            b.as_slice(),
            pj.as_slice(),
            vj.as_slice(),
            b.as_slice(),
        ],
    )
    .iter()
    .enumerate()
    {
        assert!(
            v.abs() < 1e-6,
            "Sgal3CombinedImuFactor residual[{i}] = {v:.3e}"
        );
    }
}

#[test]
fn native_state_residuals_vanish_at_ground_truth() {
    let (preint, pose_j, sb_j, dt) = scenario();
    // Frame i at t = 0, frame j at t = Δt: the time row is satisfied exactly.
    let si = sgal3_of(&SE3::identity(), Vector3::zeros(), 0.0);
    let sj = sgal3_of(&pose_j, sb_j.velocity(), dt);
    let b = DVector::zeros(6);

    let f = Sgal3StateImuFactor::new(preint.clone());
    for (i, v) in residual_of(&f, &[si.as_slice(), sj.as_slice(), b.as_slice()])
        .iter()
        .enumerate()
    {
        assert!(
            v.abs() < 1e-6,
            "Sgal3StateImuFactor residual[{i}] = {v:.3e}"
        );
    }

    let f = Sgal3CombinedStateImuFactor::new(preint);
    for (i, v) in residual_of(
        &f,
        &[si.as_slice(), b.as_slice(), sj.as_slice(), b.as_slice()],
    )
    .iter()
    .enumerate()
    {
        assert!(
            v.abs() < 1e-6,
            "Sgal3CombinedStateImuFactor residual[{i}] = {v:.3e}"
        );
    }
}

/// The whole point of the native-state factors: the time row must respond to a
/// wrong inter-keyframe interval, where the split-block factors cannot.
#[test]
fn time_row_penalizes_a_wrong_interval() {
    let (preint, pose_j, sb_j, dt) = scenario();
    let f = Sgal3StateImuFactor::new(preint).with_time_sigma(1e-3);
    let b = DVector::zeros(6);
    let si = sgal3_of(&SE3::identity(), Vector3::zeros(), 0.0);

    let exact = residual_of(
        &f,
        &[
            si.as_slice(),
            sgal3_of(&pose_j, sb_j.velocity(), dt).as_slice(),
            b.as_slice(),
        ],
    );
    assert!(exact[9].abs() < 1e-9, "time row at truth: {:.3e}", exact[9]);

    let skew = 5e-3;
    let skewed = residual_of(
        &f,
        &[
            si.as_slice(),
            sgal3_of(&pose_j, sb_j.velocity(), dt + skew).as_slice(),
            b.as_slice(),
        ],
    );
    // Weighted by 1/σ_t = 1e3, so a 5 ms error is a residual of ~5.
    assert!(
        (skewed[9] - skew * 1e3).abs() < 1e-6,
        "time row should scale as Δ/σ_t, got {:.6}",
        skewed[9]
    );
}

#[test]
fn time_sigma_scales_the_time_row_only() {
    let (preint, pose_j, sb_j, dt) = scenario();
    let b = DVector::zeros(6);
    let si = sgal3_of(&SE3::identity(), Vector3::zeros(), 0.0);
    let sj = sgal3_of(&pose_j, sb_j.velocity(), dt + 2e-3);

    let loose = Sgal3StateImuFactor::new(preint.clone()).with_time_sigma(1e-2);
    let tight = Sgal3StateImuFactor::new(preint).with_time_sigma(1e-4);
    let rl = residual_of(&loose, &[si.as_slice(), sj.as_slice(), b.as_slice()]);
    let rt = residual_of(&tight, &[si.as_slice(), sj.as_slice(), b.as_slice()]);

    for row in 0..9 {
        assert!(
            (rl[row] - rt[row]).abs() < 1e-12,
            "kinematic row {row} must not depend on the time sigma"
        );
    }
    assert!(
        (rt[9] / rl[9] - 100.0).abs() < 1e-6,
        "time row should scale"
    );
}

// ── Finite-difference Jacobians (previously #[ignore]d) ──────────────────────

fn perturbed_split_blocks(pose_j: &SE3, sb_j: &SpeedAndBias) -> Vec<DVector<f64>> {
    let pose_i = SE3::from_param_slice(&[0.05, -0.02, 0.01, 0.99875, 0.03, 0.02, 0.02]);
    vec![
        pose_blocks(&pose_i),
        DVector::from_vec(vec![0.03, -0.01, 0.02]),
        pose_blocks(pose_j),
        vec3(sb_j.velocity() + Vector3::new(0.01, -0.02, 0.005)),
        DVector::from_vec(vec![1e-3, -2e-3, 5e-4, 1e-2, -5e-3, 2e-3]),
    ]
}

#[test]
fn sgal3_imu_factor_jacobians_match_finite_differences() {
    let (preint, pose_j, sb_j, _) = scenario();
    let f = Sgal3ImuFactor::new(preint);
    check_jacobian(
        &f,
        &perturbed_split_blocks(&pose_j, &sb_j),
        &[
            Chart::Se3,
            Chart::Euclidean,
            Chart::Se3,
            Chart::Euclidean,
            Chart::Euclidean,
        ],
        "Sgal3ImuFactor",
    );
}

#[test]
fn sgal3_combined_imu_factor_jacobians_match_finite_differences() {
    let (preint, pose_j, sb_j, _) = scenario();
    let f = Sgal3CombinedImuFactor::new(preint);
    let p = perturbed_split_blocks(&pose_j, &sb_j);
    let blocks = vec![
        p[0].clone(),
        p[1].clone(),
        p[4].clone(),
        p[2].clone(),
        p[3].clone(),
        DVector::from_vec(vec![-5e-4, 1e-3, 2e-3, -8e-3, 4e-3, 1e-3]),
    ];
    check_jacobian(
        &f,
        &blocks,
        &[
            Chart::Se3,
            Chart::Euclidean,
            Chart::Euclidean,
            Chart::Se3,
            Chart::Euclidean,
            Chart::Euclidean,
        ],
        "Sgal3CombinedImuFactor",
    );
}

fn perturbed_state_blocks(
    pose_j: &SE3,
    sb_j: &SpeedAndBias,
    dt: f64,
) -> (DVector<f64>, DVector<f64>) {
    let pose_i = SE3::from_param_slice(&[0.05, -0.02, 0.01, 0.99875, 0.03, 0.02, 0.02]);
    (
        sgal3_of(&pose_i, Vector3::new(0.03, -0.01, 0.02), 1e-3),
        sgal3_of(
            pose_j,
            sb_j.velocity() + Vector3::new(0.01, -0.02, 0.005),
            dt + 2e-3,
        ),
    )
}

#[test]
fn sgal3_state_imu_factor_jacobians_match_finite_differences() {
    let (preint, pose_j, sb_j, dt) = scenario();
    let f = Sgal3StateImuFactor::new(preint);
    let (si, sj) = perturbed_state_blocks(&pose_j, &sb_j, dt);
    let blocks = vec![
        si,
        sj,
        DVector::from_vec(vec![1e-3, -2e-3, 5e-4, 1e-2, -5e-3, 2e-3]),
    ];
    check_jacobian(
        &f,
        &blocks,
        &[Chart::Sgal3, Chart::Sgal3, Chart::Euclidean],
        "Sgal3StateImuFactor",
    );
}

#[test]
fn sgal3_combined_state_imu_factor_jacobians_match_finite_differences() {
    let (preint, pose_j, sb_j, dt) = scenario();
    let f = Sgal3CombinedStateImuFactor::new(preint);
    let (si, sj) = perturbed_state_blocks(&pose_j, &sb_j, dt);
    let blocks = vec![
        si,
        DVector::from_vec(vec![1e-3, -2e-3, 5e-4, 1e-2, -5e-3, 2e-3]),
        sj,
        DVector::from_vec(vec![-5e-4, 1e-3, 2e-3, -8e-3, 4e-3, 1e-3]),
    ];
    check_jacobian(
        &f,
        &blocks,
        &[
            Chart::Sgal3,
            Chart::Euclidean,
            Chart::Sgal3,
            Chart::Euclidean,
        ],
        "Sgal3CombinedStateImuFactor",
    );
}

// ── validate_variables ───────────────────────────────────────────────────────

#[test]
fn factors_validate_their_layouts() {
    let (preint, _, _, _) = scenario();
    let pose = Variable::new(SE3::identity());
    let vel = Variable::new(Rn::new(DVector::zeros(3)));
    let bias = Variable::new(Rn::new(DVector::zeros(6)));
    let state = Variable::new(SGal3::identity());

    let split = Sgal3ImuFactor::new(preint.clone());
    let good: Vec<&dyn ManifoldVariable> = vec![&pose, &vel, &pose, &vel, &bias];
    assert!(split.validate_variables(&good).is_ok());
    let wrong: Vec<&dyn ManifoldVariable> = vec![&state, &state, &bias];
    assert!(split.validate_variables(&wrong).is_err());

    let native = Sgal3StateImuFactor::new(preint.clone());
    let good: Vec<&dyn ManifoldVariable> = vec![&state, &state, &bias];
    assert!(native.validate_variables(&good).is_ok());
    // An SE3 pose is not an SGal3 state, even though both are "a pose".
    let wrong: Vec<&dyn ManifoldVariable> = vec![&pose, &state, &bias];
    assert!(native.validate_variables(&wrong).is_err());

    let native_combined = Sgal3CombinedStateImuFactor::new(preint);
    let good: Vec<&dyn ManifoldVariable> = vec![&state, &bias, &state, &bias];
    assert!(native_combined.validate_variables(&good).is_ok());
}
