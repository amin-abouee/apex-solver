//! Tests for the motion-model constraints.
//!
//! Residuals are checked against hand-computed values — a finite-difference
//! test compares a Jacobian to its *own* residual and so cannot catch a
//! residual that is self-consistently wrong. Jacobians are then checked at
//! linearization points away from zero, where a sign or convention error
//! actually shows.

use apex_manifolds::se3::{SE3, SE3Tangent};
use apex_manifolds::se23::{SE23, SE23Tangent};
use apex_manifolds::{LieGroup, Tangent, rn::Rn};
use nalgebra::{DMatrix, DVector, UnitQuaternion, Vector3};

use super::{NonholonomicFactor, PlanarMotionFactor, ZeroAngularRateFactor, ZeroVelocityFactor};
use crate::core::variable::{ManifoldVariable, Variable};
use crate::factors::Factor;

const FD_EPS: f64 = 1e-6;
const FD_TOL: f64 = 1e-6;

#[derive(Clone, Copy)]
enum Chart {
    Se3,
    Se23,
    Euclidean,
}

fn perturb(chart: Chart, block: &[f64], k: usize, eps: f64) -> DVector<f64> {
    match chart {
        Chart::Se3 => {
            let mut t = [0.0f64; 6];
            t[k] = eps;
            let m =
                SE3::from_param_slice(block).right_plus(&SE3Tangent::from_slice(&t), None, None);
            DVector::from_column_slice(m.as_param_slice())
        }
        Chart::Se23 => {
            let mut t = [0.0f64; 9];
            t[k] = eps;
            let m =
                SE23::from_param_slice(block).right_plus(&SE23Tangent::from_slice(&t), None, None);
            DVector::from_column_slice(m.as_param_slice())
        }
        Chart::Euclidean => {
            let mut out = DVector::from_column_slice(block);
            out[k] += eps;
            out
        }
    }
}

fn residual_of<F: Factor>(f: &F, params: &[&[f64]]) -> Vec<f64> {
    let mut r = vec![0.0f64; f.residual_dim()];
    f.linearize(params, &mut r, None);
    r
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
        let dof = match chart {
            Chart::Se3 => 6,
            Chart::Se23 => 9,
            Chart::Euclidean => blocks[b].len(),
        };
        for k in 0..dof {
            let mut plus = blocks.to_vec();
            plus[b] = perturb(chart, blocks[b].as_slice(), k, FD_EPS);
            let mut minus = blocks.to_vec();
            minus[b] = perturb(chart, blocks[b].as_slice(), k, -FD_EPS);
            let rp = residual_of(f, &plus.iter().map(|v| v.as_slice()).collect::<Vec<_>>());
            let rm = residual_of(f, &minus.iter().map(|v| v.as_slice()).collect::<Vec<_>>());
            for row in 0..rows {
                let fd = (rp[row] - rm[row]) / (2.0 * FD_EPS);
                let a = analytic[(row, col)];
                assert!(
                    (a - fd).abs() / (1.0 + a.abs().max(fd.abs())) < FD_TOL,
                    "{label}: block {b} dof {k} (col {col}) row {row}: \
                     analytic={a:.8} fd={fd:.8}"
                );
            }
            col += 1;
        }
    }
    assert_eq!(col, cols, "{label}: column count");
}

/// A tilted, moving state: nothing about it is axis-aligned.
fn moving_state() -> DVector<f64> {
    let s = SE23::new(
        Vector3::new(1.0, -2.0, 0.5),
        Vector3::new(0.7, -0.3, 0.2),
        UnitQuaternion::from_euler_angles(0.15, -0.2, 0.4),
    );
    DVector::from_column_slice(s.as_param_slice())
}

// ── ZUPT ─────────────────────────────────────────────────────────────────────

#[test]
fn zero_velocity_residual_is_the_world_velocity() {
    let factor = ZeroVelocityFactor::new();
    let state = moving_state();
    let r = residual_of(&factor, &[state.as_slice()]);
    // Velocity is stored in the world frame, so the residual is it verbatim.
    assert!((r[0] - 0.7).abs() < 1e-12, "{r:?}");
    assert!((r[1] + 0.3).abs() < 1e-12, "{r:?}");
    assert!((r[2] - 0.2).abs() < 1e-12, "{r:?}");
}

#[test]
fn zero_velocity_vanishes_when_stationary() {
    let s = SE23::new(
        Vector3::new(3.0, 1.0, -2.0),
        Vector3::zeros(),
        UnitQuaternion::from_euler_angles(0.3, 0.1, -0.2),
    );
    let state = DVector::from_column_slice(s.as_param_slice());
    let r = residual_of(&ZeroVelocityFactor::new(), &[state.as_slice()]);
    assert!(r.iter().all(|v| v.abs() < 1e-14), "{r:?}");
}

#[test]
fn zero_velocity_jacobian_matches_finite_differences() {
    check_jacobian(
        &ZeroVelocityFactor::new(),
        &[moving_state()],
        &[Chart::Se23],
        "ZeroVelocityFactor",
    );
}

// ── ZARU ─────────────────────────────────────────────────────────────────────

#[test]
fn zero_angular_rate_residual_is_the_bias_error() {
    let measured = Vector3::new(0.01, -0.02, 0.005);
    let factor = ZeroAngularRateFactor::new(measured);
    let bias = DVector::from_vec(vec![0.01, 0.0, 0.0, 0.5, 0.5, 0.5]);
    let r = residual_of(&factor, &[bias.as_slice()]);
    // Only the gyro half participates; the accelerometer bias is untouched.
    assert!(r[0].abs() < 1e-15, "gyro x should match exactly: {r:?}");
    assert!((r[1] + 0.02).abs() < 1e-15, "{r:?}");
    assert!((r[2] - 0.005).abs() < 1e-15, "{r:?}");
}

#[test]
fn zero_angular_rate_jacobian_matches_finite_differences() {
    check_jacobian(
        &ZeroAngularRateFactor::new(Vector3::new(0.01, -0.02, 0.005)),
        &[DVector::from_vec(vec![
            0.002, -0.001, 0.004, 0.1, -0.2, 0.3,
        ])],
        &[Chart::Euclidean],
        "ZeroAngularRateFactor",
    );
}

// ── Nonholonomic ─────────────────────────────────────────────────────────────

/// Driving straight along the body x-axis satisfies the constraint at any yaw;
/// sliding sideways does not. This is what pins the body-frame convention.
#[test]
fn nonholonomic_accepts_forward_motion_and_rejects_sideways() {
    let yaw = 0.6_f64;
    let rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, yaw);
    let forward_world = rotation * Vector3::new(2.0, 0.0, 0.0);

    let driving = SE23::new(Vector3::new(1.0, 2.0, 0.0), forward_world, rotation);
    let r = residual_of(
        &NonholonomicFactor::new(),
        &[DVector::from_column_slice(driving.as_param_slice()).as_slice()],
    );
    assert!(
        r.iter().all(|v| v.abs() < 1e-12),
        "forward motion must satisfy the constraint: {r:?}"
    );

    let sideways_world = rotation * Vector3::new(0.0, 1.5, 0.0);
    let sliding = SE23::new(Vector3::new(1.0, 2.0, 0.0), sideways_world, rotation);
    let r = residual_of(
        &NonholonomicFactor::new(),
        &[DVector::from_column_slice(sliding.as_param_slice()).as_slice()],
    );
    assert!(
        (r[0] - 1.5).abs() < 1e-12,
        "lateral slip should appear verbatim in the residual: {r:?}"
    );
}

#[test]
fn nonholonomic_jacobian_matches_finite_differences() {
    check_jacobian(
        &NonholonomicFactor::new(),
        &[moving_state()],
        &[Chart::Se23],
        "NonholonomicFactor",
    );
}

// ── Planar motion ────────────────────────────────────────────────────────────

#[test]
fn planar_motion_vanishes_for_a_level_pose_at_height() {
    let pose = SE3::new(
        Vector3::new(4.0, -1.0, 1.5),
        UnitQuaternion::from_euler_angles(0.0, 0.0, 1.1), // yaw only
    );
    let params = DVector::from_column_slice(pose.as_param_slice());
    let r = residual_of(&PlanarMotionFactor::new(1.5), &[params.as_slice()]);
    assert!(
        r.iter().all(|v| v.abs() < 1e-12),
        "a level pose at the plane height must be free: {r:?}"
    );
}

#[test]
fn planar_motion_penalizes_height_and_tilt() {
    // Rolled by 0.2 rad and 0.3 m above the plane.
    let roll = 0.2_f64;
    let pose = SE3::new(
        Vector3::new(0.0, 0.0, 0.3),
        UnitQuaternion::from_euler_angles(roll, 0.0, 0.0),
    );
    let params = DVector::from_column_slice(pose.as_param_slice());
    let r = residual_of(&PlanarMotionFactor::ground(), &[params.as_slice()]);
    assert!((r[0] - 0.3).abs() < 1e-12, "height row: {r:?}");
    // Rolling about x leaves the body x-axis level, and tips the y-axis so its
    // world-z component is R[2,1] = sin(roll).
    assert!(r[1].abs() < 1e-12, "x-axis stays level under roll: {r:?}");
    assert!(
        (r[2] - roll.sin()).abs() < 1e-12,
        "y-axis tilt should be sin(roll): {r:?}"
    );
}

#[test]
fn planar_motion_jacobian_matches_finite_differences() {
    let pose = SE3::new(
        Vector3::new(2.0, -3.0, 0.4),
        UnitQuaternion::from_euler_angles(0.12, -0.18, 0.9),
    );
    check_jacobian(
        &PlanarMotionFactor::new(0.25),
        &[DVector::from_column_slice(pose.as_param_slice())],
        &[Chart::Se3],
        "PlanarMotionFactor",
    );
}

// ── validate_variables ───────────────────────────────────────────────────────

#[test]
fn factors_validate_their_layouts() {
    let state = Variable::new(SE23::identity());
    let pose = Variable::new(SE3::identity());
    let bias = Variable::new(Rn::new(DVector::zeros(6)));

    let zupt = ZeroVelocityFactor::new();
    assert!(
        zupt.validate_variables(&[&state as &dyn ManifoldVariable])
            .is_ok()
    );
    assert!(
        zupt.validate_variables(&[&pose as &dyn ManifoldVariable])
            .is_err()
    );

    let zaru = ZeroAngularRateFactor::new(Vector3::zeros());
    assert!(
        zaru.validate_variables(&[&bias as &dyn ManifoldVariable])
            .is_ok()
    );
    assert!(
        zaru.validate_variables(&[&state as &dyn ManifoldVariable])
            .is_err()
    );

    let planar = PlanarMotionFactor::ground();
    assert!(
        planar
            .validate_variables(&[&pose as &dyn ManifoldVariable])
            .is_ok()
    );
    // An SE23 state is not an SE3 pose, even though both carry a pose.
    assert!(
        planar
            .validate_variables(&[&state as &dyn ManifoldVariable])
            .is_err()
    );

    let nonholonomic = NonholonomicFactor::new();
    assert!(
        nonholonomic
            .validate_variables(&[&state as &dyn ManifoldVariable])
            .is_ok()
    );
}
