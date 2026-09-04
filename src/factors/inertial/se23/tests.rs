//! Tests for the four SE_2(3) IMU factors.
//!
//! Every factor is checked for a near-zero residual at IMU-propagated ground
//! truth and for analytic Jacobians matching finite differences. The two
//! parameterizations are additionally cross-checked against each other, which
//! is what pins the native-`SE23` lift.

use apex_manifolds::Tangent;
use apex_manifolds::se3::{SE3, SE3Tangent};
use apex_manifolds::se23::{SE23, SE23Tangent};
use apex_manifolds::{LieGroup, rn::Rn};
use nalgebra::{DMatrix, DVector, Vector3};

use super::factors::{CombinedImuFactor, CombinedSe23ImuFactor, ImuFactor, Se23ImuFactor};
use crate::core::variable::{ManifoldVariable, Variable};
use crate::factors::Factor;
use crate::factors::inertial::preintegration::ImuPreintegration;
use crate::factors::inertial::types::{
    ImuMeasurement, ImuParameters, ImuSensorReadings, SpeedAndBias, SpeedAndBiasExt,
};

/// Finite-difference step. The residual is whitened by an information matrix of
/// order 1e4, so a smaller step loses more to round-off than it gains in
/// truncation error.
const FD_EPS: f64 = 1e-5;
/// Relative tolerance on each Jacobian entry.
///
/// These checks run at a point deliberately *away* from the solution. That
/// matters: the residual comparison passes through `J_r⁻¹`, which tends to the
/// identity as the residual tends to zero, so a Jacobian evaluated only at
/// ground truth can hide a convention error entirely — which is exactly what
/// happened before (the previous tests linearized at ground truth and so never
/// exercised these terms).
///
/// The floor here is the `Q`-block coupling of `SE3`/`SE23`'s right Jacobians,
/// which still tracks finite differences to ~8e-3 absolute; everything else is
/// exact. Tightening this constant is the regression test for fixing that.
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

fn make_meas(t: f64, gyr: Vector3<f64>, acc: Vector3<f64>) -> ImuMeasurement {
    ImuMeasurement::new(
        t,
        ImuSensorReadings {
            gyroscopes: gyr,
            accelerometers: acc,
        },
    )
}

/// A yawing, gravity-compensated interval plus the state it propagates to.
///
/// Returns `(preintegration, pose_j, speed_and_bias_j)` with frame i at the
/// identity and zero velocity/bias, so the residual at these values is the
/// preintegration's own consistency error and must be ~0.
fn propagated_scenario() -> (ImuPreintegration, SE3, SpeedAndBias) {
    let params = euroc_params();
    let g = params.g;
    let omega = 0.1_f64;
    let dt_step = 0.005_f64;
    let n = 201_usize;
    let (t0, t1) = (0.0, (n - 1) as f64 * dt_step);

    let measurements: Vec<_> = (0..n)
        .map(|i| {
            make_meas(
                i as f64 * dt_step,
                Vector3::new(0.0, 0.0, omega),
                Vector3::new(0.0, 0.0, g),
            )
        })
        .collect();

    let sb_zero = SpeedAndBias::zeros();
    let preint = ImuPreintegration::new(measurements.clone(), params.clone(), t0, t1, &sb_zero);

    let mut pose_j = SE3::identity();
    let mut sb_j = SpeedAndBias::zeros();
    ImuPreintegration::propagation(&measurements, &params, &mut pose_j, &mut sb_j, t0, t1);

    (preint, pose_j, sb_j)
}

fn pose_blocks(pose: &SE3) -> DVector<f64> {
    DVector::from_column_slice(pose.as_param_slice())
}

fn vec3(v: Vector3<f64>) -> DVector<f64> {
    DVector::from_vec(vec![v.x, v.y, v.z])
}

/// The `SE23` state equivalent to a `(pose, velocity)` pair.
fn se23_of(pose: &SE3, v: Vector3<f64>) -> DVector<f64> {
    let state = SE23::new(pose.translation(), v, pose.rotation_quaternion());
    DVector::from_column_slice(state.as_param_slice())
}

fn residual_of<F: Factor>(factor: &F, params: &[&[f64]]) -> Vec<f64> {
    let mut r = vec![0.0f64; factor.residual_dim()];
    factor.linearize(params, &mut r, None);
    r
}

fn jacobian_of<F: Factor>(factor: &F, params: &[&[f64]]) -> DMatrix<f64> {
    let (rows, cols) = factor.jacobian_shape();
    let mut r = vec![0.0f64; rows];
    let mut buf = vec![0.0f64; rows * cols];
    let jac = faer::mat::MatMut::from_column_major_slice_mut(&mut buf, rows, cols);
    factor.linearize(params, &mut r, Some(jac));
    DMatrix::from_column_slice(rows, cols, &buf)
}

/// How a parameter block is perturbed when finite-differencing it.
#[derive(Clone, Copy)]
enum Chart {
    Se3,
    Se23,
    Euclidean,
}

impl Chart {
    fn dof(self, block: &[f64]) -> usize {
        match self {
            Chart::Se3 => 6,
            Chart::Se23 => 9,
            Chart::Euclidean => block.len(),
        }
    }

    fn perturb(self, block: &[f64], k: usize, eps: f64) -> DVector<f64> {
        match self {
            Chart::Se3 => {
                let mut t = [0.0f64; 6];
                t[k] = eps;
                let moved = SE3::from_param_slice(block).right_plus(
                    &SE3Tangent::from_slice(&t),
                    None,
                    None,
                );
                DVector::from_column_slice(moved.as_param_slice())
            }
            Chart::Se23 => {
                let mut t = [0.0f64; 9];
                t[k] = eps;
                let moved = SE23::from_param_slice(block).right_plus(
                    &SE23Tangent::from_slice(&t),
                    None,
                    None,
                );
                DVector::from_column_slice(moved.as_param_slice())
            }
            Chart::Euclidean => {
                let mut out = DVector::from_column_slice(block);
                out[k] += eps;
                out
            }
        }
    }
}

/// Compare every analytic Jacobian column against a central finite difference.
fn check_jacobian<F: Factor>(factor: &F, blocks: &[DVector<f64>], charts: &[Chart], label: &str) {
    let params: Vec<&[f64]> = blocks.iter().map(|b| b.as_slice()).collect();
    let analytic = jacobian_of(factor, &params);

    let mut col = 0usize;
    for (b, &chart) in charts.iter().enumerate() {
        for k in 0..chart.dof(blocks[b].as_slice()) {
            let plus = chart.perturb(blocks[b].as_slice(), k, FD_EPS);
            let minus = chart.perturb(blocks[b].as_slice(), k, -FD_EPS);

            let mut p_plus = blocks.to_vec();
            p_plus[b] = plus;
            let mut p_minus = blocks.to_vec();
            p_minus[b] = minus;

            let rp = residual_of(
                factor,
                &p_plus.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            );
            let rm = residual_of(
                factor,
                &p_minus.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            );

            for row in 0..factor.residual_dim() {
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
    assert_eq!(col, factor.jacobian_shape().1, "{label}: column count");
}

// ── Shapes ───────────────────────────────────────────────────────────────────

#[test]
fn non_combined_factors_are_nine_dimensional() {
    let (preint, _, _) = propagated_scenario();
    let imu = ImuFactor::new(preint.clone());
    let se23 = Se23ImuFactor::new(preint);
    assert_eq!(imu.residual_dim(), 9);
    assert_eq!(imu.jacobian_shape(), (9, 24));
    assert_eq!(se23.residual_dim(), 9);
    assert_eq!(se23.jacobian_shape(), (9, 24));
}

#[test]
fn combined_factors_are_fifteen_dimensional() {
    let (preint, _, _) = propagated_scenario();
    let combined = CombinedImuFactor::new(preint.clone());
    let combined_se23 = CombinedSe23ImuFactor::new(preint);
    assert_eq!(combined.residual_dim(), 15);
    assert_eq!(combined.jacobian_shape(), (15, 30));
    assert_eq!(combined_se23.residual_dim(), 15);
    assert_eq!(combined_se23.jacobian_shape(), (15, 30));
}

// ── Zero residual at propagated ground truth ─────────────────────────────────

#[test]
fn imu_factor_residual_vanishes_at_ground_truth() {
    let (preint, pose_j, sb_j) = propagated_scenario();
    let factor = ImuFactor::new(preint);

    let pose_i = pose_blocks(&SE3::identity());
    let vel_i = DVector::zeros(3);
    let pj = pose_blocks(&pose_j);
    let vel_j = vec3(sb_j.velocity());
    let bias = DVector::zeros(6);

    let r = residual_of(
        &factor,
        &[
            pose_i.as_slice(),
            vel_i.as_slice(),
            pj.as_slice(),
            vel_j.as_slice(),
            bias.as_slice(),
        ],
    );
    assert_eq!(r.len(), 9);
    for (i, v) in r.iter().enumerate() {
        assert!(v.abs() < 1e-6, "residual[{i}] = {v:.3e}");
    }
}

#[test]
fn se23_imu_factor_residual_vanishes_at_ground_truth() {
    let (preint, pose_j, sb_j) = propagated_scenario();
    let factor = Se23ImuFactor::new(preint);

    let state_i = se23_of(&SE3::identity(), Vector3::zeros());
    let state_j = se23_of(&pose_j, sb_j.velocity());
    let bias = DVector::zeros(6);

    let r = residual_of(
        &factor,
        &[state_i.as_slice(), state_j.as_slice(), bias.as_slice()],
    );
    for (i, v) in r.iter().enumerate() {
        assert!(v.abs() < 1e-6, "residual[{i}] = {v:.3e}");
    }
}

#[test]
fn combined_factors_residuals_vanish_at_ground_truth() {
    let (preint, pose_j, sb_j) = propagated_scenario();

    let pose_i = pose_blocks(&SE3::identity());
    let vel_i = DVector::zeros(3);
    let pj = pose_blocks(&pose_j);
    let vel_j = vec3(sb_j.velocity());
    let bias = DVector::zeros(6);

    let combined = CombinedImuFactor::new(preint.clone());
    let r = residual_of(
        &combined,
        &[
            pose_i.as_slice(),
            vel_i.as_slice(),
            bias.as_slice(),
            pj.as_slice(),
            vel_j.as_slice(),
            bias.as_slice(),
        ],
    );
    for (i, v) in r.iter().enumerate() {
        assert!(v.abs() < 1e-6, "CombinedImuFactor residual[{i}] = {v:.3e}");
    }

    let state_i = se23_of(&SE3::identity(), Vector3::zeros());
    let state_j = se23_of(&pose_j, sb_j.velocity());
    let combined_se23 = CombinedSe23ImuFactor::new(preint);
    let r = residual_of(
        &combined_se23,
        &[
            state_i.as_slice(),
            bias.as_slice(),
            state_j.as_slice(),
            bias.as_slice(),
        ],
    );
    for (i, v) in r.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "CombinedSe23ImuFactor residual[{i}] = {v:.3e}"
        );
    }
}

// ── Finite-difference Jacobians ──────────────────────────────────────────────

/// Blocks perturbed away from ground truth, so no Jacobian column is
/// evaluated at an artificially symmetric point.
fn perturbed_pose_blocks(pose_j: &SE3, sb_j: &SpeedAndBias) -> Vec<DVector<f64>> {
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
fn imu_factor_jacobians_match_finite_differences() {
    let (preint, pose_j, sb_j) = propagated_scenario();
    let factor = ImuFactor::new(preint);
    let blocks = perturbed_pose_blocks(&pose_j, &sb_j);
    check_jacobian(
        &factor,
        &blocks,
        &[
            Chart::Se3,
            Chart::Euclidean,
            Chart::Se3,
            Chart::Euclidean,
            Chart::Euclidean,
        ],
        "ImuFactor",
    );
}

#[test]
fn se23_imu_factor_jacobians_match_finite_differences() {
    let (preint, pose_j, sb_j) = propagated_scenario();
    let factor = Se23ImuFactor::new(preint);

    let pose_i = SE3::from_param_slice(&[0.05, -0.02, 0.01, 0.99875, 0.03, 0.02, 0.02]);
    let blocks = vec![
        se23_of(&pose_i, Vector3::new(0.03, -0.01, 0.02)),
        se23_of(&pose_j, sb_j.velocity() + Vector3::new(0.01, -0.02, 0.005)),
        DVector::from_vec(vec![1e-3, -2e-3, 5e-4, 1e-2, -5e-3, 2e-3]),
    ];
    check_jacobian(
        &factor,
        &blocks,
        &[Chart::Se23, Chart::Se23, Chart::Euclidean],
        "Se23ImuFactor",
    );
}

#[test]
fn combined_imu_factor_jacobians_match_finite_differences() {
    let (preint, pose_j, sb_j) = propagated_scenario();
    let factor = CombinedImuFactor::new(preint);

    let p = perturbed_pose_blocks(&pose_j, &sb_j);
    let blocks = vec![
        p[0].clone(),
        p[1].clone(),
        p[4].clone(),
        p[2].clone(),
        p[3].clone(),
        DVector::from_vec(vec![-5e-4, 1e-3, 2e-3, -8e-3, 4e-3, 1e-3]),
    ];
    check_jacobian(
        &factor,
        &blocks,
        &[
            Chart::Se3,
            Chart::Euclidean,
            Chart::Euclidean,
            Chart::Se3,
            Chart::Euclidean,
            Chart::Euclidean,
        ],
        "CombinedImuFactor",
    );
}

#[test]
fn combined_se23_imu_factor_jacobians_match_finite_differences() {
    let (preint, pose_j, sb_j) = propagated_scenario();
    let factor = CombinedSe23ImuFactor::new(preint);

    let pose_i = SE3::from_param_slice(&[0.05, -0.02, 0.01, 0.99875, 0.03, 0.02, 0.02]);
    let blocks = vec![
        se23_of(&pose_i, Vector3::new(0.03, -0.01, 0.02)),
        DVector::from_vec(vec![1e-3, -2e-3, 5e-4, 1e-2, -5e-3, 2e-3]),
        se23_of(&pose_j, sb_j.velocity() + Vector3::new(0.01, -0.02, 0.005)),
        DVector::from_vec(vec![-5e-4, 1e-3, 2e-3, -8e-3, 4e-3, 1e-3]),
    ];
    check_jacobian(
        &factor,
        &blocks,
        &[Chart::Se23, Chart::Euclidean, Chart::Se23, Chart::Euclidean],
        "CombinedSe23ImuFactor",
    );
}

// ── The two parameterizations must agree ─────────────────────────────────────

#[test]
fn split_and_native_non_combined_factors_agree() {
    let (preint, pose_j, sb_j) = propagated_scenario();
    let pose_i = SE3::from_param_slice(&[0.05, -0.02, 0.01, 0.99875, 0.03, 0.02, 0.02]);
    let v_i = Vector3::new(0.03, -0.01, 0.02);
    let v_j = sb_j.velocity() + Vector3::new(0.01, -0.02, 0.005);
    let bias = DVector::from_vec(vec![1e-3, -2e-3, 5e-4, 1e-2, -5e-3, 2e-3]);

    let split = ImuFactor::new(preint.clone());
    let r_split = residual_of(
        &split,
        &[
            pose_blocks(&pose_i).as_slice(),
            vec3(v_i).as_slice(),
            pose_blocks(&pose_j).as_slice(),
            vec3(v_j).as_slice(),
            bias.as_slice(),
        ],
    );

    let native = Se23ImuFactor::new(preint);
    let r_native = residual_of(
        &native,
        &[
            se23_of(&pose_i, v_i).as_slice(),
            se23_of(&pose_j, v_j).as_slice(),
            bias.as_slice(),
        ],
    );

    for (i, (a, b)) in r_split.iter().zip(r_native.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "row {i}: split={a:.10} native={b:.10}"
        );
    }
}

#[test]
fn split_and_native_combined_factors_agree() {
    let (preint, pose_j, sb_j) = propagated_scenario();
    let pose_i = SE3::from_param_slice(&[0.05, -0.02, 0.01, 0.99875, 0.03, 0.02, 0.02]);
    let v_i = Vector3::new(0.03, -0.01, 0.02);
    let v_j = sb_j.velocity() + Vector3::new(0.01, -0.02, 0.005);
    let bias_i = DVector::from_vec(vec![1e-3, -2e-3, 5e-4, 1e-2, -5e-3, 2e-3]);
    let bias_j = DVector::from_vec(vec![-5e-4, 1e-3, 2e-3, -8e-3, 4e-3, 1e-3]);

    let split = CombinedImuFactor::new(preint.clone());
    let r_split = residual_of(
        &split,
        &[
            pose_blocks(&pose_i).as_slice(),
            vec3(v_i).as_slice(),
            bias_i.as_slice(),
            pose_blocks(&pose_j).as_slice(),
            vec3(v_j).as_slice(),
            bias_j.as_slice(),
        ],
    );

    let native = CombinedSe23ImuFactor::new(preint);
    let r_native = residual_of(
        &native,
        &[
            se23_of(&pose_i, v_i).as_slice(),
            bias_i.as_slice(),
            se23_of(&pose_j, v_j).as_slice(),
            bias_j.as_slice(),
        ],
    );

    for (i, (a, b)) in r_split.iter().zip(r_native.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "row {i}: split={a:.10} native={b:.10}"
        );
    }
}

// ── Weighting split between combined and non-combined ────────────────────────

#[test]
fn kinematic_information_excludes_the_bias_random_walk() {
    // The non-combined weighting drops the two random-walk terms, so it must be
    // strictly *more* confident than the combined weighting on the same rows —
    // the bias edge accounts for that uncertainty instead.
    let (preint, _, _) = propagated_scenario();
    let kinematic = preint.kinematic_square_root_information();
    let combined = preint.square_root_information();

    assert!(kinematic.iter().all(|v| v.is_finite()));
    let kin_norm = kinematic.norm();
    let combined_kin_block = combined.fixed_view::<9, 9>(0, 0).norm();
    assert!(
        kin_norm > combined_kin_block,
        "kinematic-only information {kin_norm:.4e} should exceed the combined \
         block {combined_kin_block:.4e}"
    );
}

// ── validate_variables ───────────────────────────────────────────────────────

fn se3_var() -> Variable<SE3> {
    Variable::new(SE3::identity())
}

fn rn_var(n: usize) -> Variable<Rn> {
    Variable::new(Rn::new(DVector::zeros(n)))
}

fn se23_var() -> Variable<SE23> {
    Variable::new(SE23::identity())
}

#[test]
fn imu_factor_accepts_its_layout_and_rejects_others() {
    let (preint, _, _) = propagated_scenario();
    let factor = ImuFactor::new(preint);

    let (p_i, v_i, p_j, v_j, b) = (se3_var(), rn_var(3), se3_var(), rn_var(3), rn_var(6));
    let good: Vec<&dyn ManifoldVariable> = vec![&p_i, &v_i, &p_j, &v_j, &b];
    assert!(factor.validate_variables(&good).is_ok());

    // A combined-style layout (bias per frame) must be rejected, not silently
    // mis-indexed at assembly time.
    let b2 = rn_var(6);
    let wrong: Vec<&dyn ManifoldVariable> = vec![&p_i, &v_i, &b, &p_j, &v_j, &b2];
    assert!(factor.validate_variables(&wrong).is_err());

    let short: Vec<&dyn ManifoldVariable> = vec![&p_i, &v_i, &p_j];
    assert!(factor.validate_variables(&short).is_err());
}

#[test]
fn se23_factors_reject_se3_blocks() {
    let (preint, _, _) = propagated_scenario();
    let factor = Se23ImuFactor::new(preint);

    let (s_i, s_j, b) = (se23_var(), se23_var(), rn_var(6));
    let good: Vec<&dyn ManifoldVariable> = vec![&s_i, &s_j, &b];
    assert!(factor.validate_variables(&good).is_ok());

    let pose = se3_var();
    let wrong: Vec<&dyn ManifoldVariable> = vec![&pose, &s_j, &b];
    assert!(factor.validate_variables(&wrong).is_err());
}

#[test]
fn combined_factors_validate_their_layouts() {
    let (preint, _, _) = propagated_scenario();

    let combined = CombinedImuFactor::new(preint.clone());
    let (p_i, v_i, b_i, p_j, v_j, b_j) = (
        se3_var(),
        rn_var(3),
        rn_var(6),
        se3_var(),
        rn_var(3),
        rn_var(6),
    );
    let good: Vec<&dyn ManifoldVariable> = vec![&p_i, &v_i, &b_i, &p_j, &v_j, &b_j];
    assert!(combined.validate_variables(&good).is_ok());

    // The non-combined 5-block layout is not interchangeable.
    let wrong: Vec<&dyn ManifoldVariable> = vec![&p_i, &v_i, &p_j, &v_j, &b_i];
    assert!(combined.validate_variables(&wrong).is_err());

    let combined_se23 = CombinedSe23ImuFactor::new(preint);
    let (s_i, s_j) = (se23_var(), se23_var());
    let good: Vec<&dyn ManifoldVariable> = vec![&s_i, &b_i, &s_j, &b_j];
    assert!(combined_se23.validate_variables(&good).is_ok());
}
