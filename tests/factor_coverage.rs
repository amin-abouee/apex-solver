//! Integration coverage for the factors the scenario suite does not reach.
//!
//! `tests/factor_integration.rs` exercises the factors a VIO/SLAM pipeline uses
//! end to end. This file covers the rest, so that **every** exported factor
//! appears in at least one solved graph rather than only in its own unit tests
//! — a unit test checks a residual and its Jacobian in isolation, but it cannot
//! catch a factor that is correct on its own yet cannot be registered, assembled
//! or driven to a solution.
//!
//! `every_exported_factor_is_exercised` at the bottom enforces that claim
//! mechanically, by reading the modules' own re-export lists.
//!
//! Each scenario builds measurements from ground truth, starts the optimizer
//! from a perturbed state, and asserts the truth is recovered.

#![allow(clippy::print_stderr)]

mod common;
use common::{Trajectory, anchor_rn, anchor_se3, build_imu_dataset, imu_params, lm_solver};

use apex_solver::JacobianMode;
use apex_solver::apex_manifolds::se3::{SE3, SE3Tangent};
use apex_solver::apex_manifolds::se23::{SE23, SE23Tangent};
use apex_solver::apex_manifolds::sgal3::{SGal3, SGal3Tangent};
use apex_solver::apex_manifolds::{LieGroup, ManifoldType, Tangent};
use apex_solver::core::noise::NoiseModel;
use apex_solver::core::problem::Problem;
use apex_solver::factors::imu::{
    ImuPreintegration, SpeedAndBias, SpeedAndBiasExt, bias_random_walk, bias_random_walk_noise,
    se23, sgal3,
};
use apex_solver::factors::lidar::{
    IcpFactor, LidarEdgeFactor, PrecomputedPlane, lidar_plane_factor_isotropic,
};
use apex_solver::factors::navigation::{GpsAsyncFactor, GpsFactor};
use apex_solver::factors::ranging::bearing::BearingFactor;
use apex_solver::factors::ranging::range::PosePoseRangeFactor;
use apex_solver::factors::visual::{
    DepthFactor, EssentialMatrixConstraint, EssentialMatrixFactor, HomogeneousPointFactor,
    OneSidedDepthFactor,
};
use nalgebra::{DVector, UnitQuaternion, Vector3, Vector4};

/// Perturb an SE3 pose in its tangent so the optimizer has work to do.
fn nudge_se3(pose: &SE3, tangent: [f64; 6]) -> DVector<f64> {
    let moved = pose.right_plus(&SE3Tangent::from_slice(&tangent), None, None);
    DVector::from_column_slice(moved.as_param_slice())
}

fn se3_of(key: &[f64]) -> SE3 {
    SE3::from_param_slice(key)
}

/// The SE23 navigation state for a keyframe.
fn state_of(pose: &SE3, sb: &SpeedAndBias) -> SE23 {
    SE23::new(
        pose.translation(),
        sb.velocity(),
        pose.rotation_quaternion(),
    )
}

// ── 1. IMU: shared-bias form + bias random walk ──────────────────────────────

/// `se23::ImuFactor` shares one bias per interval and needs a companion random
/// walk edge; `se23::CombinedImuFactor` embeds that walk. Both formulations
/// should land on the same trajectory — that equivalence is the whole reason
/// two factors exist, so it is worth asserting rather than documenting.
#[test]
fn shared_bias_imu_form_matches_the_combined_form() {
    let key_times: Vec<f64> = (0..=5).map(|k| k as f64 * 0.5).collect();
    let (truth, segments) = build_imu_dataset(&key_times);
    let n = key_times.len();

    let truth_states: Vec<SE23> = truth.iter().map(|(p, sb)| state_of(p, sb)).collect();
    let bias_truth = DVector::from_vec(vec![
        common::BG_TRUE[0],
        common::BG_TRUE[1],
        common::BG_TRUE[2],
        common::BA_TRUE[0],
        common::BA_TRUE[1],
        common::BA_TRUE[2],
    ]);

    /// Build the chain either with the shared-bias form + walk edges, or the
    /// combined form, and return the recovered states.
    fn solve(
        truth_states: &[SE23],
        segments: &[Vec<apex_solver::factors::imu::ImuMeasurement>],
        key_times: &[f64],
        bias_truth: &DVector<f64>,
        shared_bias: bool,
    ) -> Vec<SE23> {
        let n = truth_states.len();
        let mut problem = Problem::new(JacobianMode::Sparse);
        let mut states = Vec::new();
        let mut biases = Vec::new();

        for (k, s) in truth_states.iter().enumerate() {
            let init = if k == 0 {
                s.clone()
            } else {
                let mut t = [0.0f64; 9];
                t[0] = 0.05;
                t[5] = 0.02;
                t[6] = 0.08;
                s.right_plus(&SE23Tangent::from_slice(&t), None, None)
            };
            states.push(problem.add_variable(
                ManifoldType::SE23,
                DVector::from_column_slice(init.as_param_slice()),
            ));
            biases.push(problem.add_variable(ManifoldType::RN, DVector::zeros(6)));
        }

        // Anchor the first state and give the biases a prior for observability.
        problem.add_residual_block_with_noise(
            &[states[0]],
            Box::new(apex_solver::factors::pose::PriorFactor::new(
                truth_states[0].clone(),
            )),
            None,
            NoiseModel::from_sigmas(&[1e-4; 9]).unwrap_or_else(|e| panic!("{e}")),
        );
        for b in &biases {
            problem.add_residual_block_with_noise(
                &[*b],
                Box::new(apex_solver::factors::pose::EuclideanPriorFactor::new(
                    bias_truth.clone(),
                )),
                None,
                NoiseModel::from_sigmas(&[0.05; 6]).unwrap_or_else(|e| panic!("{e}")),
            );
        }

        let params = imu_params();
        for k in 0..n - 1 {
            let mut sb_ref = SpeedAndBias::zeros();
            for i in 0..6 {
                sb_ref[3 + i] = bias_truth[i];
            }
            let preint = ImuPreintegration::new(
                segments[k].clone(),
                params.clone(),
                key_times[k],
                key_times[k + 1],
                &sb_ref,
            );
            let dt = key_times[k + 1] - key_times[k];

            if shared_bias {
                problem.add_residual_block(
                    &[states[k], states[k + 1], biases[k]],
                    Box::new(se23::ImuFactor::new(preint)),
                    None,
                );
                // The companion edge the shared-bias form requires.
                problem.add_residual_block_with_noise(
                    &[biases[k], biases[k + 1]],
                    Box::new(bias_random_walk()),
                    None,
                    bias_random_walk_noise(&params, dt).unwrap_or_else(|e| panic!("{e}")),
                );
            } else {
                problem.add_residual_block(
                    &[states[k], biases[k], states[k + 1], biases[k + 1]],
                    Box::new(se23::CombinedImuFactor::new(preint)),
                    None,
                );
            }
        }

        // Loose position/velocity aiding so the chain is observable.
        for (k, key) in states.iter().enumerate() {
            problem.add_residual_block_with_noise(
                &[*key],
                Box::new(apex_solver::factors::pose::PriorFactor::new(
                    truth_states[k].clone(),
                )),
                None,
                NoiseModel::from_sigmas(&[0.1; 9]).unwrap_or_else(|e| panic!("{e}")),
            );
        }

        let mut solver = lm_solver(200);
        let result = solver
            .optimize(&mut problem)
            .unwrap_or_else(|e| panic!("IMU chain failed: {e}"));
        states
            .iter()
            .map(|k| SE23::from_param_slice(result.parameters[*k].as_param_slice()))
            .collect()
    }

    let shared = solve(&truth_states, &segments, &key_times, &bias_truth, true);
    let combined = solve(&truth_states, &segments, &key_times, &bias_truth, false);

    for k in 0..n {
        let pos_err = (shared[k].translation() - truth_states[k].translation()).norm();
        assert!(
            pos_err < 0.15,
            "shared-bias form: keyframe {k} position error {pos_err:.4}"
        );
        let agree = (shared[k].translation() - combined[k].translation()).norm();
        assert!(
            agree < 0.1,
            "the two bias formulations disagree at keyframe {k} by {agree:.4}"
        );
    }
}

// ── 2. SGal(3) IMU factors, including the time coordinate ────────────────────

/// The SGal(3) factors estimate a keyframe's timestamp alongside its state, so
/// a wrong interval is corrected by the time residual rather than absorbed.
///
/// Deliberately a **single interval** with `s_i = 0`. SGal(3)'s group law is
/// `t = R₁·(t₂ + s₁·v₂) + t₁`, so the left operand's time coordinate couples the
/// right operand's velocity into translation: `gc_i⁻¹ ∘ state_j` depends on the
/// absolute `s_i`, not only on `s_j − s_i`, while the preintegrated delta
/// corresponds to `s_i = 0`. Chaining these factors over absolute timestamps is
/// therefore not yet correct — see the note in `factors::imu::sgal3`.
#[test]
fn sgal3_imu_factors_recover_state_and_timestamp() {
    run_sgal3_interval(false);
    run_sgal3_interval(true);
}

fn run_sgal3_interval(combined: bool) {
    let dt = 0.5_f64;
    let key_times = [0.0_f64, dt];
    let (truth, segments) = build_imu_dataset(&key_times);
    let params = imu_params();

    let bias_truth = DVector::from_vec(vec![
        common::BG_TRUE[0],
        common::BG_TRUE[1],
        common::BG_TRUE[2],
        common::BA_TRUE[0],
        common::BA_TRUE[1],
        common::BA_TRUE[2],
    ]);

    // Interval-relative timestamps: frame i at 0, frame j at Δt.
    let truth_sgal: Vec<SGal3> = truth
        .iter()
        .enumerate()
        .map(|(k, (pose, sb))| {
            SGal3::new(
                pose.translation(),
                sb.velocity(),
                pose.rotation_quaternion(),
                k as f64 * dt,
            )
        })
        .collect();

    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut states = Vec::new();
    let mut biases = Vec::new();
    for (k, state) in truth_sgal.iter().enumerate() {
        let init = if k == 0 {
            state.clone()
        } else {
            let mut t = [0.0f64; 10];
            t[0] = 0.05; // ρ
            t[3] = 0.08; // ν
            t[9] = 0.01; // a 10 ms timestamp error
            state.right_plus(&SGal3Tangent::from_slice(&t), None, None)
        };
        states.push(problem.add_variable(
            ManifoldType::SGal3,
            DVector::from_column_slice(init.as_param_slice()),
        ));
        biases.push(problem.add_variable(ManifoldType::RN, DVector::zeros(6)));
    }

    problem.add_residual_block_with_noise(
        &[states[0]],
        Box::new(apex_solver::factors::pose::PriorFactor::new(
            truth_sgal[0].clone(),
        )),
        None,
        NoiseModel::from_sigmas(&[1e-4; 10]).unwrap_or_else(|e| panic!("{e}")),
    );
    for b in &biases {
        problem.add_residual_block_with_noise(
            &[*b],
            Box::new(apex_solver::factors::pose::EuclideanPriorFactor::new(
                bias_truth.clone(),
            )),
            None,
            NoiseModel::from_sigmas(&[0.02; 6]).unwrap_or_else(|e| panic!("{e}")),
        );
    }

    let mut sb_ref = SpeedAndBias::zeros();
    for i in 0..6 {
        sb_ref[3 + i] = bias_truth[i];
    }
    let preint = ImuPreintegration::new(
        segments[0].clone(),
        params.clone(),
        key_times[0],
        key_times[1],
        &sb_ref,
    );

    if combined {
        problem.add_residual_block(
            &[states[0], biases[0], states[1], biases[1]],
            Box::new(sgal3::CombinedImuFactor::new(preint)),
            None,
        );
    } else {
        problem.add_residual_block(
            &[states[0], states[1], biases[0]],
            Box::new(sgal3::ImuFactor::new(preint)),
            None,
        );
        problem.add_residual_block_with_noise(
            &[biases[0], biases[1]],
            Box::new(bias_random_walk()),
            None,
            bias_random_walk_noise(&params, dt).unwrap_or_else(|e| panic!("{e}")),
        );
    }

    let mut solver = lm_solver(200);
    let result = solver
        .optimize(&mut problem)
        .unwrap_or_else(|e| panic!("SGal3 interval (combined={combined}) failed: {e}"));

    let hat = SGal3::from_param_slice(result.parameters[states[1]].as_param_slice());
    let pos_err = (hat.translation() - truth_sgal[1].translation()).norm();
    assert!(
        pos_err < 0.05,
        "combined={combined}: position error {pos_err:.4}"
    );
    let dt_err = (hat.time() - dt).abs();
    assert!(
        dt_err < 2e-3,
        "combined={combined}: timestamp error {dt_err:.5} s — \
         the time row should pull it back"
    );
}

// ── 3. GPS in its own frame ──────────────────────────────────────────────────

/// `GpsFactor` estimates the world→GPS frame transform alongside the poses:
/// `z = C_GW·(t_WS + C_WS·r_SA) + t_GW`.
#[test]
fn gps_factor_recovers_poses_and_the_frame_transform() {
    let times: Vec<f64> = (0..6).map(|k| k as f64 * 0.4).collect();
    let poses: Vec<SE3> = times.iter().map(|t| Trajectory::pose(*t)).collect();
    let lever = Vector3::new(0.12, -0.05, 0.30);

    // Ground-truth world → GPS frame transform.
    let t_gw = SE3::new(
        Vector3::new(2.0, -1.0, 0.5),
        UnitQuaternion::from_euler_angles(0.0, 0.0, 0.35),
    );
    let c_gw = t_gw.rotation_so3().rotation_matrix();

    let mut problem = Problem::new(JacobianMode::Sparse);
    let pose_keys: Vec<_> = poses
        .iter()
        .map(|p| {
            problem.add_variable(
                ManifoldType::SE3,
                nudge_se3(p, [0.2, -0.15, 0.1, 0.02, -0.01, 0.03]),
            )
        })
        .collect();
    let frame_key = problem.add_variable(
        ManifoldType::SE3,
        nudge_se3(&t_gw, [0.1, 0.1, -0.1, 0.0, 0.0, 0.05]),
    );

    // Rotation is not observable from position fixes alone; anchor it.
    for (k, key) in pose_keys.iter().enumerate() {
        problem.add_residual_block_with_noise(
            &[*key],
            Box::new(apex_solver::factors::pose::PoseRotationPrior::new(
                poses[k].clone(),
            )),
            None,
            NoiseModel::from_sigmas(&[1e-3; 3]).unwrap_or_else(|e| panic!("{e}")),
        );
    }
    // Observability: each pose owns 3 translation DOF and receives exactly one
    // 3D fix, so the poses can absorb *any* frame transform — with a free
    // T_GW the graph has a 6-DOF null space and a zero-cost family of
    // solutions. Surveying the frame's rotation (the usual case) and anchoring
    // one pose leaves the frame translation as the only free frame DOF.
    problem.add_residual_block_with_noise(
        &[frame_key],
        Box::new(apex_solver::factors::pose::PoseRotationPrior::new(
            t_gw.clone(),
        )),
        None,
        NoiseModel::from_sigmas(&[1e-4; 3]).unwrap_or_else(|e| panic!("{e}")),
    );
    problem.add_residual_block_with_noise(
        &[pose_keys[0]],
        Box::new(apex_solver::factors::pose::PriorFactor::new(
            poses[0].clone(),
        )),
        None,
        NoiseModel::from_sigmas(&[1e-3; 6]).unwrap_or_else(|e| panic!("{e}")),
    );

    for (k, key) in pose_keys.iter().enumerate() {
        let antenna_world =
            poses[k].translation() + poses[k].rotation_so3().rotation_matrix() * lever;
        let measured = c_gw * antenna_world + t_gw.translation();
        problem.add_residual_block(
            &[*key, frame_key],
            Box::new(GpsFactor::new_isotropic(measured, lever, 0.05)),
            None,
        );
    }

    let mut solver = lm_solver(300);
    let result = solver
        .optimize(&mut problem)
        .unwrap_or_else(|e| panic!("GPS graph failed: {e}"));

    for (k, key) in pose_keys.iter().enumerate() {
        let hat = se3_of(result.parameters[*key].as_param_slice());
        let err = (hat.translation() - poses[k].translation()).norm();
        assert!(err < 0.05, "pose {k} position error {err:.4}");
    }
    let frame_hat = se3_of(result.parameters[frame_key].as_param_slice());
    let frame_err = (frame_hat.translation() - t_gw.translation()).norm();
    assert!(
        frame_err < 0.1,
        "world→GPS translation error {frame_err:.4}"
    );
}

/// `GpsAsyncFactor` carries a preintegration from the keyframe to the GPS
/// timestamp, so a fix that lands between keyframes still constrains the state.
#[test]
fn gps_async_factor_uses_a_fix_between_keyframes() {
    let key_times = [0.0f64, 0.5];
    let (truth, segments) = build_imu_dataset(&key_times);
    let params = imu_params();
    let lever = Vector3::new(0.1, 0.0, 0.2);

    // Preintegrate only part of the interval: the GPS sample is mid-segment.
    let t_gps = 0.3;
    let partial: Vec<_> = segments[0]
        .iter()
        .filter(|m| m.timestamp <= t_gps + 1e-9)
        .cloned()
        .collect();
    let mut sb_ref = SpeedAndBias::zeros();
    let v0 = truth[0].1.velocity();
    sb_ref[0] = v0.x;
    sb_ref[1] = v0.y;
    sb_ref[2] = v0.z;
    for i in 0..3 {
        sb_ref[3 + i] = common::BG_TRUE[i];
        sb_ref[6 + i] = common::BA_TRUE[i];
    }
    let preint = ImuPreintegration::new(
        partial.clone(),
        params.clone(),
        key_times[0],
        t_gps,
        &sb_ref,
    );

    // Where the antenna actually is at t_gps, by independent propagation.
    let mut pose_at_gps = truth[0].0.clone();
    let mut sb_at_gps = sb_ref;
    ImuPreintegration::propagation(
        &partial,
        &params,
        &mut pose_at_gps,
        &mut sb_at_gps,
        key_times[0],
        t_gps,
    );
    let measured = pose_at_gps.translation() + pose_at_gps.rotation_so3().rotation_matrix() * lever;

    let mut problem = Problem::new(JacobianMode::Sparse);
    let pose_key = problem.add_variable(
        ManifoldType::SE3,
        nudge_se3(&truth[0].0, [0.25, -0.2, 0.15, 0.0, 0.0, 0.0]),
    );
    let sb_key = problem.add_variable(ManifoldType::RN, {
        let mut v = DVector::zeros(9);
        for i in 0..9 {
            v[i] = sb_ref[i];
        }
        v
    });
    // The GPS frame coincides with the world frame here.
    let frame_key = problem.add_variable(
        ManifoldType::SE3,
        DVector::from_column_slice(SE3::identity().as_param_slice()),
    );

    anchor_se3(&mut problem, frame_key, &SE3::identity());
    anchor_rn(&mut problem, sb_key, sb_ref.as_slice());
    problem.add_residual_block_with_noise(
        &[pose_key],
        Box::new(apex_solver::factors::pose::PoseRotationPrior::new(
            truth[0].0.clone(),
        )),
        None,
        NoiseModel::from_sigmas(&[1e-5; 3]).unwrap_or_else(|e| panic!("{e}")),
    );

    problem.add_residual_block(
        &[pose_key, sb_key, frame_key],
        Box::new(GpsAsyncFactor::new_isotropic(measured, lever, 0.02, preint)),
        None,
    );

    let mut solver = lm_solver(150);
    let result = solver
        .optimize(&mut problem)
        .unwrap_or_else(|e| panic!("async GPS graph failed: {e}"));

    let hat = se3_of(result.parameters[pose_key].as_param_slice());
    let err = (hat.translation() - truth[0].0.translation()).norm();
    assert!(
        err < 0.05,
        "keyframe position error {err:.4}: the mid-segment fix should pull it back"
    );
}

// ── 4. Bearing-only and pose-pose range ──────────────────────────────────────

/// A landmark seen only as a direction from several poses, plus a baseline
/// constraint between two of the poses.
#[test]
fn bearing_and_pose_pose_range_recover_a_landmark() {
    let poses: Vec<SE3> = (0..4)
        .map(|k| {
            SE3::new(
                Vector3::new(k as f64 * 1.5, 0.0, 0.0),
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0.1 * k as f64),
            )
        })
        .collect();
    let landmark = Vector3::new(3.0, 5.0, 1.5);

    let mut problem = Problem::new(JacobianMode::Sparse);
    let pose_keys: Vec<_> = poses
        .iter()
        .map(|p| {
            problem.add_variable(
                ManifoldType::SE3,
                DVector::from_column_slice(p.as_param_slice()),
            )
        })
        .collect();
    // Only the landmark is unknown; the poses are anchored.
    for (k, key) in pose_keys.iter().enumerate() {
        anchor_se3(&mut problem, *key, &poses[k]);
    }
    let lm_key = problem.add_variable(
        ManifoldType::RN,
        DVector::from_vec(vec![2.0, 3.0, 0.5]), // deliberately off
    );

    for (k, key) in pose_keys.iter().enumerate() {
        let delta = landmark - poses[k].translation();
        let bearing =
            poses[k].rotation_so3().rotation_matrix().transpose() * (delta / delta.norm());
        problem.add_residual_block(
            &[*key, lm_key],
            Box::new(BearingFactor::new_isotropic(bearing, 0.01)),
            None,
        );
    }

    // A baseline constraint between two poses — a pure pose-pose range edge.
    let baseline = (poses[0].translation() - poses[3].translation()).norm();
    problem.add_residual_block_with_noise(
        &[pose_keys[0], pose_keys[3]],
        Box::new(PosePoseRangeFactor::new(baseline)),
        None,
        NoiseModel::from_sigmas(&[0.01]).unwrap_or_else(|e| panic!("{e}")),
    );

    let mut solver = lm_solver(150);
    let result = solver
        .optimize(&mut problem)
        .unwrap_or_else(|e| panic!("bearing graph failed: {e}"));

    let hat = result.parameters[lm_key].as_param_slice();
    let err = (Vector3::new(hat[0], hat[1], hat[2]) - landmark).norm();
    assert!(err < 0.05, "landmark error {err:.4} from bearing-only rays");
}

// ── 5. Epipolar geometry ─────────────────────────────────────────────────────

/// Both essential-matrix factors constrain a relative pose from 2D–2D matches.
/// Scale is unobservable from epipolar geometry, so rotation and the
/// translation *direction* are what must be recovered.
#[test]
fn essential_matrix_factors_recover_rotation_and_direction() {
    let t_21 = SE3::new(
        Vector3::new(0.8, -0.2, 0.1),
        UnitQuaternion::from_euler_angles(0.03, 0.08, -0.05),
    );
    let rot = t_21.rotation_so3().rotation_matrix();
    let trans = t_21.translation();

    // Normalized coordinates of a spread of points in both cameras.
    let mut p1 = Vec::new();
    let mut p2 = Vec::new();
    for i in 0..12 {
        let a = i as f64 * 0.5;
        let x1 = Vector3::new(0.4 * a.cos(), 0.3 * a.sin(), 3.0 + 0.2 * a);
        let x2 = rot * x1 + trans;
        p1.push(x1 / x1.z);
        p2.push(x2 / x2.z);
    }

    let mut problem = Problem::new(JacobianMode::Sparse);
    let key = problem.add_variable(
        ManifoldType::SE3,
        nudge_se3(&t_21, [0.05, 0.05, -0.05, 0.04, -0.03, 0.05]),
    );

    problem.add_residual_block_with_noise(
        &[key],
        Box::new(EssentialMatrixFactor::new(p1, p2).unwrap_or_else(|e| panic!("{e}"))),
        None,
        NoiseModel::from_sigmas(&[1e-3; 12]).unwrap_or_else(|e| panic!("{e}")),
    );
    // The measured essential matrix as a 6D constraint on the same pose.
    problem.add_residual_block_with_noise(
        &[key],
        Box::new(
            EssentialMatrixConstraint::new(t_21.rotation_so3(), trans / trans.norm())
                .unwrap_or_else(|e| panic!("{e}")),
        ),
        None,
        NoiseModel::from_sigmas(&[1e-2; 6]).unwrap_or_else(|e| panic!("{e}")),
    );

    let mut solver = lm_solver(150);
    let result = solver
        .optimize(&mut problem)
        .unwrap_or_else(|e| panic!("epipolar graph failed: {e}"));

    let hat = se3_of(result.parameters[key].as_param_slice());
    let rot_err = hat
        .rotation_so3()
        .between(&t_21.rotation_so3(), None, None)
        .log(None)
        .axis_angle()
        .norm();
    assert!(rot_err < 1e-2, "relative rotation error {rot_err:.5} rad");

    let dir_hat = hat.translation() / hat.translation().norm();
    let dir_err = (dir_hat - trans / trans.norm()).norm();
    assert!(dir_err < 5e-2, "translation direction error {dir_err:.5}");
}

// ── 6. Homogeneous landmark: depth measurements ──────────────────────────────

/// `DepthFactor`, its one-sided variant, and `HomogeneousPointFactor` all act
/// on an R⁴ homogeneous landmark. An RGB-D-style depth reading plus a weak
/// Euclidean prior should place it.
#[test]
fn depth_factors_place_a_homogeneous_landmark() {
    let t_ws = SE3::new(
        Vector3::new(0.5, -0.2, 1.0),
        UnitQuaternion::from_euler_angles(0.02, -0.03, 0.2),
    );
    let t_sc = SE3::new(
        Vector3::new(0.05, 0.0, 0.1),
        UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0),
    );
    let landmark = Vector3::new(1.6, 0.8, 2.4);
    let hp_true = Vector4::new(landmark.x, landmark.y, landmark.z, 1.0);

    // Depth of the landmark in the camera frame, computed independently.
    let hp_c = t_sc.inverse(None).matrix() * (t_ws.inverse(None).matrix() * hp_true);
    let depth_true = hp_c[2] / hp_c[3];
    assert!(depth_true > 0.0, "test setup: landmark must be in front");

    let mut problem = Problem::new(JacobianMode::Sparse);
    let pose_key = problem.add_variable(
        ManifoldType::SE3,
        DVector::from_column_slice(t_ws.as_param_slice()),
    );
    let extrinsics_key = problem.add_variable(
        ManifoldType::SE3,
        DVector::from_column_slice(t_sc.as_param_slice()),
    );
    let lm_key = problem.add_variable(
        ManifoldType::RN,
        DVector::from_vec(vec![1.2, 0.4, 1.6, 1.0]), // off
    );
    anchor_se3(&mut problem, pose_key, &t_ws);
    anchor_se3(&mut problem, extrinsics_key, &t_sc);

    problem.add_residual_block(
        &[pose_key, lm_key, extrinsics_key],
        Box::new(DepthFactor::<false>::new_from_stdev(depth_true, 0.02)),
        None,
    );
    // The one-sided variant is inactive beyond the measurement, so it must not
    // fight the reading above; it only guards against the landmark coming closer.
    problem.add_residual_block(
        &[pose_key, lm_key, extrinsics_key],
        Box::new(OneSidedDepthFactor::new_from_stdev(depth_true, 0.05)),
        None,
    );
    // Lateral position comes from the dehomogenized-point prior.
    problem.add_residual_block(
        &[lm_key],
        Box::new(HomogeneousPointFactor::new_isotropic(landmark, 0.05)),
        None,
    );

    let mut solver = lm_solver(150);
    let result = solver
        .optimize(&mut problem)
        .unwrap_or_else(|e| panic!("depth graph failed: {e}"));

    let hat = result.parameters[lm_key].as_param_slice();
    let recovered = Vector3::new(hat[0], hat[1], hat[2]) / hat[3];
    let err = (recovered - landmark).norm();
    assert!(err < 0.05, "homogeneous landmark error {err:.4}");
}

// ── 7. Distance-field ICP and the LOAM plane alias ───────────────────────────

/// `IcpFactor` aligns two poses through a distance field; `LidarPlaneFactor` is
/// that same factor against a precomputed plane. Three non-parallel planes
/// determine the translation between the scans.
#[test]
fn icp_and_lidar_plane_factors_align_two_scans() {
    let t_wa = SE3::identity();
    let t_wb = SE3::new(
        Vector3::new(0.35, -0.2, 0.15),
        UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0),
    );

    // A corner: three orthogonal planes in frame A.
    let planes = [
        (Vector3::new(1.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0)),
        (Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 3.0, 0.0)),
        (Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 0.0, 1.0)),
    ];

    let mut problem = Problem::new(JacobianMode::Sparse);
    let key_a = problem.add_variable(
        ManifoldType::SE3,
        DVector::from_column_slice(t_wa.as_param_slice()),
    );
    let key_b = problem.add_variable(
        ManifoldType::SE3,
        nudge_se3(&t_wb, [0.2, 0.15, -0.1, 0.0, 0.0, 0.0]),
    );
    anchor_se3(&mut problem, key_a, &t_wa);
    // Rotation is not observable from three planes plus one point each.
    problem.add_residual_block_with_noise(
        &[key_b],
        Box::new(apex_solver::factors::pose::PoseRotationPrior::new(
            t_wb.clone(),
        )),
        None,
        NoiseModel::from_sigmas(&[1e-5; 3]).unwrap_or_else(|e| panic!("{e}")),
    );

    for (i, (normal, point_on_plane)) in planes.iter().enumerate() {
        // A scan point that lies exactly on the plane when B is at truth.
        let p_a = *point_on_plane + Vector3::new(0.3, 0.4, 0.2);
        let corrected = p_a - *normal * normal.dot(&(p_a - *point_on_plane));
        let p_b = t_wb
            .inverse(None)
            .act(&t_wa.act(&corrected, None, None), None, None);

        if i == 0 {
            // Through the generic distance-field entry point.
            problem.add_residual_block(
                &[key_a, key_b],
                Box::new(IcpFactor::new(
                    PrecomputedPlane::new(*point_on_plane, *normal),
                    p_b,
                    0.02,
                )),
                None,
            );
        } else {
            // Through the LOAM plane helper (the same factor, aliased).
            problem.add_residual_block(
                &[key_a, key_b],
                Box::new(lidar_plane_factor_isotropic(
                    p_b,
                    *point_on_plane,
                    *normal,
                    0.02,
                )),
                None,
            );
        }
    }

    // A LOAM edge feature: a point on a line, constraining the two directions
    // perpendicular to it.
    {
        let edge_point = Vector3::new(1.0, 1.0, 0.0);
        let edge_dir = Vector3::new(0.0, 0.0, 1.0);
        let on_line = edge_point + edge_dir * 0.7;
        let p_b = t_wb
            .inverse(None)
            .act(&t_wa.act(&on_line, None, None), None, None);
        problem.add_residual_block(
            &[key_a, key_b],
            Box::new(LidarEdgeFactor::new_isotropic(
                p_b, edge_point, edge_dir, 0.02,
            )),
            None,
        );
    }

    let mut solver = lm_solver(150);
    let result = solver
        .optimize(&mut problem)
        .unwrap_or_else(|e| panic!("ICP graph failed: {e}"));

    let hat = se3_of(result.parameters[key_b].as_param_slice());
    let err = (hat.translation() - t_wb.translation()).norm();
    assert!(err < 0.02, "scan pose translation error {err:.4}");
}

// ── 8. The coverage guard ────────────────────────────────────────────────────

/// Every factor a domain module exports must appear in one of the integration
/// suites.
///
/// The list is not hand-maintained: it is read from the modules' own `pub use`
/// lines at compile time, so exporting a new factor without adding a scenario
/// fails here rather than going unnoticed.
#[test]
fn every_exported_factor_is_exercised_by_an_integration_test() {
    const MODULES: &[&str] = &[
        include_str!("../src/factors/pose/mod.rs"),
        include_str!("../src/factors/visual/mod.rs"),
        include_str!("../src/factors/lidar/mod.rs"),
        include_str!("../src/factors/navigation/mod.rs"),
        include_str!("../src/factors/ranging/mod.rs"),
        include_str!("../src/factors/marginal/mod.rs"),
        include_str!("../src/factors/imu/se23/mod.rs"),
        include_str!("../src/factors/imu/sgal3/mod.rs"),
    ];
    const SUITES: &[&str] = &[
        include_str!("factor_integration.rs"),
        include_str!("factor_coverage.rs"),
    ];

    // Aliases that are the same type under another name; covering one covers all.
    const ALIASES: &[&str] = &["RegularDepthFactor"];

    let mut exported: Vec<String> = Vec::new();
    for source in MODULES {
        for line in source.lines() {
            let line = line.trim();
            if !line.starts_with("pub use ") {
                continue;
            }
            for name in line
                .trim_start_matches("pub use ")
                .trim_end_matches(';')
                .replace(['{', '}'], " ")
                .split([',', ' '])
            {
                let name = name.rsplit("::").next().unwrap_or(name).trim();
                if name.ends_with("Factor") && !ALIASES.contains(&name) {
                    exported.push(name.to_string());
                }
            }
        }
    }
    exported.sort();
    exported.dedup();
    assert!(
        exported.len() > 20,
        "the export scrape found only {} factors — the parser is probably broken",
        exported.len()
    );

    let missing: Vec<&String> = exported
        .iter()
        .filter(|name| !SUITES.iter().any(|s| s.contains(name.as_str())))
        .collect();
    assert!(
        missing.is_empty(),
        "these exported factors have no integration coverage: {missing:?}\n\
         Add a scenario to tests/factor_coverage.rs."
    );
}
