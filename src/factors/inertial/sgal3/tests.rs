//! Tests for the two SGal(3) IMU factors.
//!
//! Like the SE_2(3) tests, Jacobians are checked away from ground truth. The
//! predecessors of these tests were `#[ignore]`d for a Jacobian chain that has
//! since been fixed.

use apex_manifolds::se3::SE3;
use apex_manifolds::sgal3::{SGal3, SGal3Tangent};
use apex_manifolds::{LieGroup, Tangent, rn::Rn};
use nalgebra::{DMatrix, DVector, Vector3};

use super::factors::{CombinedImuFactor, ImuFactor};
use crate::core::variable::{ManifoldVariable, Variable};
use crate::factors::Factor;
use crate::factors::inertial::preintegration::ImuPreintegration;
use crate::factors::inertial::types::{
    ImuMeasurement, ImuParameters, ImuSensorReadings, SpeedAndBias, SpeedAndBiasExt,
};

const FD_EPS: f64 = 1e-5;
/// See the SE_2(3) tests: the floor is the group's `Q`-block coupling.
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

/// The `SGal3` state for a `(pose, velocity, time)` triple.
fn state_of(pose: &SE3, v: Vector3<f64>, t: f64) -> DVector<f64> {
    let s = SGal3::new(pose.translation(), v, pose.rotation_quaternion(), t);
    DVector::from_column_slice(s.as_param_slice())
}

fn residual_of<F: Factor>(f: &F, params: &[&[f64]]) -> Vec<f64> {
    let mut r = vec![0.0f64; f.residual_dim()];
    f.linearize(params, &mut r, None);
    r
}

fn perturb(block: &[f64], k: usize, eps: f64, is_state: bool) -> DVector<f64> {
    if is_state {
        let mut t = [0.0f64; 10];
        t[k] = eps;
        let moved =
            SGal3::from_param_slice(block).right_plus(&SGal3Tangent::from_slice(&t), None, None);
        DVector::from_column_slice(moved.as_param_slice())
    } else {
        let mut out = DVector::from_column_slice(block);
        out[k] += eps;
        out
    }
}

fn check_jacobian<F: Factor>(f: &F, blocks: &[DVector<f64>], state_blocks: &[bool], label: &str) {
    let params: Vec<&[f64]> = blocks.iter().map(|b| b.as_slice()).collect();
    let (rows, cols) = f.jacobian_shape();
    let mut r = vec![0.0; rows];
    let mut buf = vec![0.0; rows * cols];
    let jm = faer::mat::MatMut::from_column_major_slice_mut(&mut buf, rows, cols);
    f.linearize(&params, &mut r, Some(jm));
    let analytic = DMatrix::from_column_slice(rows, cols, &buf);

    let mut col = 0usize;
    for (b, &is_state) in state_blocks.iter().enumerate() {
        let dof = if is_state { 10 } else { blocks[b].len() };
        for k in 0..dof {
            let mut plus = blocks.to_vec();
            plus[b] = perturb(blocks[b].as_slice(), k, FD_EPS, is_state);
            let mut minus = blocks.to_vec();
            minus[b] = perturb(blocks[b].as_slice(), k, -FD_EPS, is_state);

            let rp = residual_of(f, &plus.iter().map(|v| v.as_slice()).collect::<Vec<_>>());
            let rm = residual_of(f, &minus.iter().map(|v| v.as_slice()).collect::<Vec<_>>());

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

fn perturbed(
    pose_j: &SE3,
    sb_j: &SpeedAndBias,
    dt: f64,
) -> (DVector<f64>, DVector<f64>, DVector<f64>) {
    let pose_i = SE3::from_param_slice(&[0.05, -0.02, 0.01, 0.99875, 0.03, 0.02, 0.02]);
    (
        state_of(&pose_i, Vector3::new(0.03, -0.01, 0.02), 1e-3),
        state_of(
            pose_j,
            sb_j.velocity() + Vector3::new(0.01, -0.02, 0.005),
            dt + 2e-3,
        ),
        DVector::from_vec(vec![1e-3, -2e-3, 5e-4, 1e-2, -5e-3, 2e-3]),
    )
}

// ── Shapes ───────────────────────────────────────────────────────────────────

#[test]
fn factor_shapes_carry_the_time_row() {
    let (preint, _, _, _) = scenario();
    let imu = ImuFactor::new(preint.clone());
    assert_eq!(imu.residual_dim(), 10, "9 kinematic rows plus time");
    assert_eq!(imu.jacobian_shape(), (10, 26));

    let combined = CombinedImuFactor::new(preint);
    assert_eq!(combined.residual_dim(), 16, "15 rows plus time");
    assert_eq!(combined.jacobian_shape(), (16, 32));
}

// ── Zero residual at propagated ground truth ─────────────────────────────────

#[test]
fn residuals_vanish_at_ground_truth() {
    let (preint, pose_j, sb_j, dt) = scenario();
    // Frame i at t = 0 and frame j at t = Δt satisfies the time row exactly.
    let state_i = state_of(&SE3::identity(), Vector3::zeros(), 0.0);
    let state_j = state_of(&pose_j, sb_j.velocity(), dt);
    let bias = DVector::zeros(6);

    let imu = ImuFactor::new(preint.clone());
    for (i, v) in residual_of(
        &imu,
        &[state_i.as_slice(), state_j.as_slice(), bias.as_slice()],
    )
    .iter()
    .enumerate()
    {
        assert!(v.abs() < 1e-6, "ImuFactor residual[{i}] = {v:.3e}");
    }

    let combined = CombinedImuFactor::new(preint);
    for (i, v) in residual_of(
        &combined,
        &[
            state_i.as_slice(),
            bias.as_slice(),
            state_j.as_slice(),
            bias.as_slice(),
        ],
    )
    .iter()
    .enumerate()
    {
        assert!(v.abs() < 1e-6, "CombinedImuFactor residual[{i}] = {v:.3e}");
    }
}

/// The reason to choose SGal(3): the time row responds to a wrong interval,
/// which an SE_2(3) factor structurally cannot see.
#[test]
fn time_row_penalizes_a_wrong_interval() {
    let (preint, pose_j, sb_j, dt) = scenario();
    let factor = ImuFactor::new(preint).with_time_sigma(1e-3);
    let bias = DVector::zeros(6);
    let state_i = state_of(&SE3::identity(), Vector3::zeros(), 0.0);

    let exact = residual_of(
        &factor,
        &[
            state_i.as_slice(),
            state_of(&pose_j, sb_j.velocity(), dt).as_slice(),
            bias.as_slice(),
        ],
    );
    assert!(exact[9].abs() < 1e-9, "time row at truth: {:.3e}", exact[9]);

    let skew = 5e-3;
    let skewed = residual_of(
        &factor,
        &[
            state_i.as_slice(),
            state_of(&pose_j, sb_j.velocity(), dt + skew).as_slice(),
            bias.as_slice(),
        ],
    );
    // Weighted by 1/σ_t = 1e3, so a 5 ms error becomes a residual of ~5.
    assert!(
        (skewed[9] - skew * 1e3).abs() < 1e-6,
        "time row should scale as Δ/σ_t, got {:.6}",
        skewed[9]
    );
}

#[test]
fn time_sigma_scales_only_the_time_row() {
    let (preint, pose_j, sb_j, dt) = scenario();
    let bias = DVector::zeros(6);
    let state_i = state_of(&SE3::identity(), Vector3::zeros(), 0.0);
    let state_j = state_of(&pose_j, sb_j.velocity(), dt + 2e-3);

    let loose = ImuFactor::new(preint.clone()).with_time_sigma(1e-2);
    let tight = ImuFactor::new(preint).with_time_sigma(1e-4);
    let rl = residual_of(
        &loose,
        &[state_i.as_slice(), state_j.as_slice(), bias.as_slice()],
    );
    let rt = residual_of(
        &tight,
        &[state_i.as_slice(), state_j.as_slice(), bias.as_slice()],
    );

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

// ── Finite-difference Jacobians ──────────────────────────────────────────────

#[test]
fn imu_factor_jacobians_match_finite_differences() {
    let (preint, pose_j, sb_j, dt) = scenario();
    let factor = ImuFactor::new(preint);
    let (si, sj, bias) = perturbed(&pose_j, &sb_j, dt);
    check_jacobian(
        &factor,
        &[si, sj, bias],
        &[true, true, false],
        "sgal3::ImuFactor",
    );
}

#[test]
fn combined_imu_factor_jacobians_match_finite_differences() {
    let (preint, pose_j, sb_j, dt) = scenario();
    let factor = CombinedImuFactor::new(preint);
    let (si, sj, bias_i) = perturbed(&pose_j, &sb_j, dt);
    let bias_j = DVector::from_vec(vec![-5e-4, 1e-3, 2e-3, -8e-3, 4e-3, 1e-3]);
    check_jacobian(
        &factor,
        &[si, bias_i, sj, bias_j],
        &[true, false, true, false],
        "sgal3::CombinedImuFactor",
    );
}

// ── validate_variables ───────────────────────────────────────────────────────

#[test]
fn factors_validate_their_layouts() {
    let (preint, _, _, _) = scenario();
    let state = Variable::new(SGal3::identity());
    let bias = Variable::new(Rn::new(DVector::zeros(6)));
    let pose = Variable::new(SE3::identity());

    let imu = ImuFactor::new(preint.clone());
    let good: Vec<&dyn ManifoldVariable> = vec![&state, &state, &bias];
    assert!(imu.validate_variables(&good).is_ok());
    // An SE3 pose carries no time coordinate, so it is not an SGal3 state.
    let wrong: Vec<&dyn ManifoldVariable> = vec![&pose, &state, &bias];
    assert!(imu.validate_variables(&wrong).is_err());

    let combined = CombinedImuFactor::new(preint);
    let good: Vec<&dyn ManifoldVariable> = vec![&state, &bias, &state, &bias];
    assert!(combined.validate_variables(&good).is_ok());
    let wrong: Vec<&dyn ManifoldVariable> = vec![&state, &state, &bias];
    assert!(combined.validate_variables(&wrong).is_err());
}
