//! Tests for the two SE_2(3) IMU factors.
//!
//! Jacobians are checked at a linearization point *away* from ground truth.
//! That matters: the residual passes through `J_r⁻¹`, which tends to the
//! identity as the residual tends to zero, so a Jacobian checked only at the
//! solution can be badly wrong and still pass.

use apex_manifolds::se3::SE3;
use apex_manifolds::se23::{SE23, SE23Tangent};
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
/// Relative tolerance. The floor is the `Q`-block coupling of `SE23`'s right
/// Jacobians, which still tracks finite differences to ~8e-3 absolute;
/// everything else is exact.
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

/// A yawing, gravity-compensated interval and the state it propagates to.
///
/// Frame i sits at the identity with zero velocity and bias, so the residual at
/// these values is the preintegration's own consistency error and must be ~0.
fn scenario() -> (ImuPreintegration, SE3, SpeedAndBias) {
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

    (preint, pose_j, sb_j)
}

/// The `SE23` state for a `(pose, velocity)` pair, as a parameter vector.
fn state_of(pose: &SE3, v: Vector3<f64>) -> DVector<f64> {
    let s = SE23::new(pose.translation(), v, pose.rotation_quaternion());
    DVector::from_column_slice(s.as_param_slice())
}

fn residual_of<F: Factor>(f: &F, params: &[&[f64]]) -> Vec<f64> {
    let mut r = vec![0.0f64; f.residual_dim()];
    f.linearize(params, &mut r, None);
    r
}

/// Perturb an `SE23` state block, or a plain vector block, by `eps` in slot `k`.
fn perturb(block: &[f64], k: usize, eps: f64, is_state: bool) -> DVector<f64> {
    if is_state {
        let mut t = [0.0f64; 9];
        t[k] = eps;
        let moved =
            SE23::from_param_slice(block).right_plus(&SE23Tangent::from_slice(&t), None, None);
        DVector::from_column_slice(moved.as_param_slice())
    } else {
        let mut out = DVector::from_column_slice(block);
        out[k] += eps;
        out
    }
}

/// Compare every analytic Jacobian column against a central finite difference.
///
/// `state_blocks` marks which parameter blocks live on `SE23`.
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
        let dof = if is_state { 9 } else { blocks[b].len() };
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

/// A state pair and bias deliberately off the solution.
fn perturbed(pose_j: &SE3, sb_j: &SpeedAndBias) -> (DVector<f64>, DVector<f64>, DVector<f64>) {
    let pose_i = SE3::from_param_slice(&[0.05, -0.02, 0.01, 0.99875, 0.03, 0.02, 0.02]);
    (
        state_of(&pose_i, Vector3::new(0.03, -0.01, 0.02)),
        state_of(pose_j, sb_j.velocity() + Vector3::new(0.01, -0.02, 0.005)),
        DVector::from_vec(vec![1e-3, -2e-3, 5e-4, 1e-2, -5e-3, 2e-3]),
    )
}

// ── Shapes ───────────────────────────────────────────────────────────────────

#[test]
fn factor_shapes() {
    let (preint, _, _) = scenario();
    let imu = ImuFactor::new(preint.clone());
    assert_eq!(imu.residual_dim(), 9);
    assert_eq!(imu.jacobian_shape(), (9, 24));

    let combined = CombinedImuFactor::new(preint);
    assert_eq!(combined.residual_dim(), 15);
    assert_eq!(combined.jacobian_shape(), (15, 30));
}

// ── Zero residual at propagated ground truth ─────────────────────────────────

#[test]
fn imu_factor_residual_vanishes_at_ground_truth() {
    let (preint, pose_j, sb_j) = scenario();
    let factor = ImuFactor::new(preint);

    let state_i = state_of(&SE3::identity(), Vector3::zeros());
    let state_j = state_of(&pose_j, sb_j.velocity());
    let bias = DVector::zeros(6);

    let r = residual_of(
        &factor,
        &[state_i.as_slice(), state_j.as_slice(), bias.as_slice()],
    );
    assert_eq!(r.len(), 9);
    for (i, v) in r.iter().enumerate() {
        assert!(v.abs() < 1e-6, "residual[{i}] = {v:.3e}");
    }
}

#[test]
fn combined_imu_factor_residual_vanishes_at_ground_truth() {
    let (preint, pose_j, sb_j) = scenario();
    let factor = CombinedImuFactor::new(preint);

    let state_i = state_of(&SE3::identity(), Vector3::zeros());
    let state_j = state_of(&pose_j, sb_j.velocity());
    let bias = DVector::zeros(6);

    let r = residual_of(
        &factor,
        &[
            state_i.as_slice(),
            bias.as_slice(),
            state_j.as_slice(),
            bias.as_slice(),
        ],
    );
    assert_eq!(r.len(), 15);
    for (i, v) in r.iter().enumerate() {
        assert!(v.abs() < 1e-6, "residual[{i}] = {v:.3e}");
    }
}

/// The combined factor's trailing rows are the bias difference, and they are
/// the only thing distinguishing it from the shared-bias form.
#[test]
fn combined_factor_penalizes_a_bias_step() {
    let (preint, pose_j, sb_j) = scenario();
    let factor = CombinedImuFactor::new(preint);

    let state_i = state_of(&SE3::identity(), Vector3::zeros());
    let state_j = state_of(&pose_j, sb_j.velocity());
    let bias_i = DVector::zeros(6);
    let mut bias_j = DVector::zeros(6);
    bias_j[0] = 1e-3;

    let r = residual_of(
        &factor,
        &[
            state_i.as_slice(),
            bias_i.as_slice(),
            state_j.as_slice(),
            bias_j.as_slice(),
        ],
    );
    let walk_rows: f64 = r[9..15].iter().map(|v| v * v).sum();
    assert!(
        walk_rows > 0.0,
        "a bias step between keyframes must show in the walk rows"
    );
}

// ── Finite-difference Jacobians ──────────────────────────────────────────────

#[test]
fn imu_factor_jacobians_match_finite_differences() {
    let (preint, pose_j, sb_j) = scenario();
    let factor = ImuFactor::new(preint);
    let (si, sj, bias) = perturbed(&pose_j, &sb_j);
    check_jacobian(
        &factor,
        &[si, sj, bias],
        &[true, true, false],
        "se23::ImuFactor",
    );
}

#[test]
fn combined_imu_factor_jacobians_match_finite_differences() {
    let (preint, pose_j, sb_j) = scenario();
    let factor = CombinedImuFactor::new(preint);
    let (si, sj, bias_i) = perturbed(&pose_j, &sb_j);
    let bias_j = DVector::from_vec(vec![-5e-4, 1e-3, 2e-3, -8e-3, 4e-3, 1e-3]);
    check_jacobian(
        &factor,
        &[si, bias_i, sj, bias_j],
        &[true, false, true, false],
        "se23::CombinedImuFactor",
    );
}

// ── Weighting ────────────────────────────────────────────────────────────────

#[test]
fn kinematic_information_excludes_the_bias_random_walk() {
    // The shared-bias form drops the random-walk terms, so it is strictly more
    // confident on those rows — the bias edge accounts for them instead.
    let (preint, _, _) = scenario();
    let kinematic = preint.kinematic_square_root_information();
    let combined = preint.square_root_information();

    assert!(kinematic.iter().all(|v| v.is_finite()));
    assert!(
        kinematic.norm() > combined.fixed_view::<9, 9>(0, 0).norm(),
        "kinematic-only information should exceed the combined block"
    );
}

// ── validate_variables ───────────────────────────────────────────────────────

#[test]
fn factors_validate_their_layouts() {
    let (preint, _, _) = scenario();
    let state = Variable::new(SE23::identity());
    let bias = Variable::new(Rn::new(DVector::zeros(6)));
    let pose = Variable::new(SE3::identity());

    let imu = ImuFactor::new(preint.clone());
    let good: Vec<&dyn ManifoldVariable> = vec![&state, &state, &bias];
    assert!(imu.validate_variables(&good).is_ok());
    // An SE3 pose is not an SE23 state, even though both are "a pose".
    let wrong: Vec<&dyn ManifoldVariable> = vec![&pose, &state, &bias];
    assert!(imu.validate_variables(&wrong).is_err());
    // The combined 4-block layout is not interchangeable.
    let wrong: Vec<&dyn ManifoldVariable> = vec![&state, &bias, &state, &bias];
    assert!(imu.validate_variables(&wrong).is_err());

    let combined = CombinedImuFactor::new(preint);
    let good: Vec<&dyn ManifoldVariable> = vec![&state, &bias, &state, &bias];
    assert!(combined.validate_variables(&good).is_ok());
    let wrong: Vec<&dyn ManifoldVariable> = vec![&state, &state, &bias];
    assert!(combined.validate_variables(&wrong).is_err());
}
