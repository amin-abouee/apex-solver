//! Tests for the extrinsics-aware and time-offset reprojection factors.
//!
//! Both are checked at linearization points away from the solution: the
//! quantities they add — a camera-to-body transform, a clock offset — appear in
//! the residual only through terms that vanish at zero, so a Jacobian verified
//! only at the truth would say nothing.

use apex_camera_models::{CameraModel, PinholeCamera};
use apex_manifolds::se3::{SE3, SE3Tangent};
use apex_manifolds::se23::{SE23, SE23Tangent};
use apex_manifolds::so3::SO3Tangent;
use apex_manifolds::{LieGroup, Tangent, rn::Rn};
use nalgebra::{DMatrix, DVector, UnitQuaternion, Vector2, Vector3};

use super::{ExtrinsicProjectionFactor, TimeOffsetProjectionFactor};
use crate::core::variable::{ManifoldVariable, Variable};
use crate::factors::Factor;

const FD_EPS: f64 = 1e-6;
const FD_TOL: f64 = 1e-5;

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

fn camera() -> PinholeCamera {
    PinholeCamera::from([500.0, 500.0, 320.0, 240.0])
}

fn body_pose() -> SE3 {
    SE3::new(
        Vector3::new(1.0, -0.5, 0.2),
        UnitQuaternion::from_euler_angles(0.05, -0.03, 0.4),
    )
}

/// A realistic rig: camera forward-right of the body, slightly toed in.
fn extrinsics() -> SE3 {
    SE3::new(
        Vector3::new(0.15, -0.06, 0.10),
        UnitQuaternion::from_euler_angles(0.02, 0.04, -0.03),
    )
}

fn params_of(pose: &SE3) -> DVector<f64> {
    DVector::from_column_slice(pose.as_param_slice())
}

// ── ExtrinsicProjectionFactor ────────────────────────────────────────────────

/// The residual must vanish for the observation the model itself predicts, with
/// the projection built through an independent path: compose, invert, project.
#[test]
fn extrinsic_projection_vanishes_at_the_predicted_observation() {
    let (t_wb, t_bc) = (body_pose(), extrinsics());
    let landmark = Vector3::new(3.0, 1.2, 5.0);

    let t_wc = t_wb.compose(&t_bc, None, None);
    let p_cam = t_wc.inverse(None).act(&landmark, None, None);
    let uv = camera()
        .project(&p_cam)
        .unwrap_or_else(|e| panic!("test setup projects: {e}"));

    let factor = ExtrinsicProjectionFactor::new(camera(), uv);
    let lm = DVector::from_vec(vec![landmark.x, landmark.y, landmark.z]);
    let r = residual_of(
        &factor,
        &[
            params_of(&t_wb).as_slice(),
            params_of(&t_bc).as_slice(),
            lm.as_slice(),
        ],
    );
    assert!(r.iter().all(|v| v.abs() < 1e-9), "{r:?}");
}

/// A wrong extrinsic rotation must move the projection — otherwise the
/// extrinsics block is not actually connected to the residual.
#[test]
fn extrinsic_projection_responds_to_a_calibration_error() {
    let (t_wb, t_bc) = (body_pose(), extrinsics());
    let landmark = Vector3::new(3.0, 1.2, 5.0);
    let t_wc = t_wb.compose(&t_bc, None, None);
    let uv = camera()
        .project(&t_wc.inverse(None).act(&landmark, None, None))
        .unwrap_or_else(|e| panic!("{e}"));

    let factor = ExtrinsicProjectionFactor::new(camera(), uv);
    let lm = DVector::from_vec(vec![landmark.x, landmark.y, landmark.z]);
    // 1° of extrinsic yaw error.
    let mistaken = perturb(Chart::Se3, params_of(&t_bc).as_slice(), 5, 0.0175);
    let r = residual_of(
        &factor,
        &[
            params_of(&t_wb).as_slice(),
            mistaken.as_slice(),
            lm.as_slice(),
        ],
    );
    let magnitude = (r[0] * r[0] + r[1] * r[1]).sqrt();
    // ~f·tan(1°) ≈ 8.7 px on the optical axis, less for this off-axis
    // landmark. The point is that it is pixels, not noise.
    assert!(
        magnitude > 3.0,
        "1° of extrinsic error should move the projection by several pixels, got {magnitude:.3}"
    );
}

#[test]
fn extrinsic_projection_jacobians_match_finite_differences() {
    let landmark = Vector3::new(2.5, -0.8, 4.0);
    let t_wc = body_pose().compose(&extrinsics(), None, None);
    let uv = camera()
        .project(&t_wc.inverse(None).act(&landmark, None, None))
        .unwrap_or_else(|e| panic!("{e}"));
    let factor = ExtrinsicProjectionFactor::new(camera(), uv + Vector2::new(1.5, -2.0));

    // Away from the solution: the observation is offset and the blocks nudged.
    let blocks = vec![
        perturb(Chart::Se3, params_of(&body_pose()).as_slice(), 0, 0.05),
        perturb(Chart::Se3, params_of(&extrinsics()).as_slice(), 4, 0.03),
        DVector::from_vec(vec![landmark.x + 0.1, landmark.y - 0.05, landmark.z + 0.2]),
    ];
    check_jacobian(
        &factor,
        &blocks,
        &[Chart::Se3, Chart::Se3, Chart::Euclidean],
        "ExtrinsicProjectionFactor",
    );
}

// ── TimeOffsetProjectionFactor ───────────────────────────────────────────────

fn moving_state() -> SE23 {
    SE23::new(
        Vector3::new(1.0, -0.5, 0.2),
        Vector3::new(1.4, 0.3, -0.2), // moving, so the offset is observable
        UnitQuaternion::from_euler_angles(0.05, -0.03, 0.4),
    )
}

const BODY_RATE: Vector3<f64> = Vector3::new(0.10, -0.05, 0.30);

/// Predict the observation the model implies, built independently of the
/// factor: shift the state by `t_d`, compose with the extrinsics, project.
fn predicted_uv(state: &SE23, t_bc: &SE3, landmark: &Vector3<f64>, t_d: f64) -> Vector2<f64> {
    let delta_rot = SO3Tangent::new(BODY_RATE * t_d).exp(None);
    let t_wb = SE3::new(
        state.translation() + state.velocity() * t_d,
        state
            .rotation_so3()
            .compose(&delta_rot, None, None)
            .quaternion(),
    );
    let t_wc = t_wb.compose(t_bc, None, None);
    camera()
        .project(&t_wc.inverse(None).act(landmark, None, None))
        .unwrap_or_else(|e| panic!("test setup projects: {e}"))
}

#[test]
fn time_offset_projection_vanishes_at_the_predicted_observation() {
    let (state, t_bc) = (moving_state(), extrinsics());
    let landmark = Vector3::new(3.0, 1.2, 5.0);
    let t_d = 0.021;

    let factor = TimeOffsetProjectionFactor::new(
        camera(),
        predicted_uv(&state, &t_bc, &landmark, t_d),
        BODY_RATE,
    );
    let lm = DVector::from_vec(vec![landmark.x, landmark.y, landmark.z]);
    let offset = DVector::from_vec(vec![t_d]);
    let r = residual_of(
        &factor,
        &[
            DVector::from_column_slice(state.as_param_slice()).as_slice(),
            params_of(&t_bc).as_slice(),
            lm.as_slice(),
            offset.as_slice(),
        ],
    );
    assert!(r.iter().all(|v| v.abs() < 1e-9), "{r:?}");
}

/// The offset is only observable through motion: on a moving platform a wrong
/// `t_d` must move the projection, and with the platform at rest it must not.
#[test]
fn time_offset_is_observable_only_while_moving() {
    let (state, t_bc) = (moving_state(), extrinsics());
    let landmark = Vector3::new(3.0, 1.2, 5.0);
    let lm = DVector::from_vec(vec![landmark.x, landmark.y, landmark.z]);
    let truth = 0.020;

    let factor = TimeOffsetProjectionFactor::new(
        camera(),
        predicted_uv(&state, &t_bc, &landmark, truth),
        BODY_RATE,
    );
    let wrong = DVector::from_vec(vec![0.0]); // 20 ms of unmodelled offset
    let r = residual_of(
        &factor,
        &[
            DVector::from_column_slice(state.as_param_slice()).as_slice(),
            params_of(&t_bc).as_slice(),
            lm.as_slice(),
            wrong.as_slice(),
        ],
    );
    let magnitude = (r[0] * r[0] + r[1] * r[1]).sqrt();
    assert!(
        magnitude > 1.0,
        "20 ms of offset on a moving platform should show, got {magnitude:.3} px"
    );

    // Same offset error, but stationary and not rotating: nothing to see.
    let still = SE23::new(
        state.translation(),
        Vector3::zeros(),
        state.rotation_quaternion(),
    );
    let still_factor = TimeOffsetProjectionFactor::new(
        camera(),
        {
            let t_wc = SE3::new(still.translation(), still.rotation_quaternion())
                .compose(&t_bc, None, None);
            camera()
                .project(&t_wc.inverse(None).act(&landmark, None, None))
                .unwrap_or_else(|e| panic!("{e}"))
        },
        Vector3::zeros(),
    );
    let r = residual_of(
        &still_factor,
        &[
            DVector::from_column_slice(still.as_param_slice()).as_slice(),
            params_of(&t_bc).as_slice(),
            lm.as_slice(),
            DVector::from_vec(vec![0.05]).as_slice(),
        ],
    );
    assert!(
        r.iter().all(|v| v.abs() < 1e-9),
        "a stationary platform cannot observe the offset: {r:?}"
    );
}

#[test]
fn time_offset_projection_jacobians_match_finite_differences() {
    let (state, t_bc) = (moving_state(), extrinsics());
    let landmark = Vector3::new(2.5, -0.8, 4.0);
    let factor = TimeOffsetProjectionFactor::new(
        camera(),
        predicted_uv(&state, &t_bc, &landmark, 0.02) + Vector2::new(1.0, -1.5),
        BODY_RATE,
    );

    let blocks = vec![
        perturb(
            Chart::Se23,
            DVector::from_column_slice(state.as_param_slice()).as_slice(),
            0,
            0.04,
        ),
        perturb(Chart::Se3, params_of(&t_bc).as_slice(), 4, 0.03),
        DVector::from_vec(vec![landmark.x + 0.1, landmark.y - 0.05, landmark.z + 0.2]),
        DVector::from_vec(vec![0.013]),
    ];
    check_jacobian(
        &factor,
        &blocks,
        &[Chart::Se23, Chart::Se3, Chart::Euclidean, Chart::Euclidean],
        "TimeOffsetProjectionFactor",
    );
}

// ── validate_variables ───────────────────────────────────────────────────────

#[test]
fn factors_validate_their_layouts() {
    let pose = Variable::new(SE3::identity());
    let state = Variable::new(SE23::identity());
    let landmark = Variable::new(Rn::new(DVector::zeros(3)));
    let offset = Variable::new(Rn::new(DVector::zeros(1)));

    let extrinsic = ExtrinsicProjectionFactor::new(camera(), Vector2::zeros());
    let good: Vec<&dyn ManifoldVariable> = vec![&pose, &pose, &landmark];
    assert!(extrinsic.validate_variables(&good).is_ok());
    // Forgetting the extrinsics block is the likely mistake, and must be caught.
    let wrong: Vec<&dyn ManifoldVariable> = vec![&pose, &landmark];
    assert!(extrinsic.validate_variables(&wrong).is_err());

    let timed = TimeOffsetProjectionFactor::new(camera(), Vector2::zeros(), Vector3::zeros());
    let good: Vec<&dyn ManifoldVariable> = vec![&state, &pose, &landmark, &offset];
    assert!(timed.validate_variables(&good).is_ok());
    // An SE3 pose carries no velocity, so it cannot stand in for the state.
    let wrong: Vec<&dyn ManifoldVariable> = vec![&pose, &pose, &landmark, &offset];
    assert!(timed.validate_variables(&wrong).is_err());
}
