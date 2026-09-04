//! Shared simulation harness for the factor integration suites.
//!
//! One analytic trajectory, one IMU stream, and the anchoring helpers every
//! scenario needs. `tests/factor_integration.rs` and `tests/factor_coverage.rs`
//! are separate binaries, so each includes this module and uses the parts it
//! needs — hence the blanket `dead_code` allowance.

#![allow(dead_code)]

pub mod nclt;

use apex_solver::apex_manifolds::LieGroup;
use apex_solver::apex_manifolds::se3::SE3;
use apex_solver::core::VarKey;
use apex_solver::core::noise::NoiseModel;
use apex_solver::core::problem::Problem;
use apex_solver::factors::imu::{
    ImuMeasurement, ImuParameters, ImuPreintegration, ImuSensorReadings, SpeedAndBias,
};
use apex_solver::factors::pose::PriorFactor;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{DVector, UnitQuaternion, Vector3};

/// Gravity magnitude used by the IMU model (positive z in the world frame;
/// the preintegration subtracts `g·dt` from the velocity increment).
pub const G: f64 = 9.81;
pub const DT_IMU: f64 = 0.005;

/// Smooth circular-climb trajectory with analytic derivatives.
pub struct Trajectory;

impl Trajectory {
    pub fn position(t: f64) -> Vector3<f64> {
        Vector3::new(4.0 * (0.5 * t).cos(), 4.0 * (0.5 * t).sin(), 0.5 * t)
    }
    pub fn velocity(t: f64) -> Vector3<f64> {
        Vector3::new(-2.0 * (0.5 * t).sin(), 2.0 * (0.5 * t).cos(), 0.5)
    }
    pub fn acceleration(t: f64) -> Vector3<f64> {
        Vector3::new(-(0.5 * t).cos(), -(0.5 * t).sin(), 0.0)
    }
    pub fn rotation(t: f64) -> UnitQuaternion<f64> {
        // Slow yaw sweep with a gentle roll/pitch wobble.
        UnitQuaternion::from_euler_angles(0.05 * (0.7 * t).sin(), 0.04 * (0.9 * t).cos(), 0.5 * t)
    }
    /// World-to-body pose at time `t` (p_body = R·p_world + t).
    pub fn pose(t: f64) -> SE3 {
        SE3::new(Self::position(t), Self::rotation(t))
    }
}

/// IMU parameters with small but non-zero biases.
pub fn imu_params() -> ImuParameters {
    ImuParameters {
        sigma_g_c: 1.6968e-04,
        sigma_a_c: 2.0000e-03,
        sigma_gw_c: 1.9393e-05,
        sigma_aw_c: 3.0000e-03,
        g: G,
        ..ImuParameters::default()
    }
}

/// Gyro / accel biases used when synthesizing the IMU stream.
pub const BG_TRUE: [f64; 3] = [0.01, -0.005, 0.002];
pub const BA_TRUE: [f64; 3] = [0.05, -0.03, 0.04];

/// Analytic body rates (without bias) at time `t`.
pub fn gyro_truth(t: f64) -> Vector3<f64> {
    Vector3::new(
        0.5 * 0.05 * (0.7 * t).cos(),
        -0.5 * 0.04 * (0.9 * t).sin(),
        0.5,
    )
}

/// Sequentially consistent dataset generator.
///
/// The IMU stream is produced step by step from the *propagated* state: the
/// accelerometer reading is built from the orientation the integrator
/// itself maintains, and the keyframe states come from the crate's own
/// `propagation` over exactly the same samples. Measurements, ground truth
/// and factor model are therefore consistent by construction.
pub fn build_imu_dataset(times: &[f64]) -> (Vec<(SE3, SpeedAndBias)>, Vec<Vec<ImuMeasurement>>) {
    let params = imu_params();
    let mut state = (
        SE3::new(
            Trajectory::position(times[0]),
            Trajectory::rotation(times[0]),
        ),
        {
            let mut sb = SpeedAndBias::zeros();
            let v = Trajectory::velocity(times[0]);
            sb[0] = v.x;
            sb[1] = v.y;
            sb[2] = v.z;
            sb
        },
    );
    let mut keyframes = vec![state.clone()];
    let mut segments = Vec::new();

    for w in times.windows(2) {
        let n = ((w[1] - w[0]) / DT_IMU).round() as usize;
        let mut samples: Vec<ImuMeasurement> = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let t = w[0] + i as f64 * DT_IMU;
            // Body-frame specific force from the CURRENT propagated state:
            // a_body = Rᵀ(a_world + g) under the convention
            // v_j = v_i + ∫R·a_body − g·dt.
            let r = state.0.rotation_so3().rotation_matrix();
            let a_world = Trajectory::acceleration(t);
            let a_body = r * (a_world + Vector3::new(0.0, 0.0, G));
            samples.push(ImuMeasurement::new(
                t,
                ImuSensorReadings {
                    gyroscopes: gyro_truth(t) + Vector3::new(BG_TRUE[0], BG_TRUE[1], BG_TRUE[2]),
                    accelerometers: a_body + Vector3::new(BA_TRUE[0], BA_TRUE[1], BA_TRUE[2]),
                },
            ));
            if samples.len() >= 2 {
                // Advance the state so the next sample uses the propagated
                // orientation (the propagation re-reads the same samples).
                let start = samples.len() - 2;
                ImuPreintegration::propagation(
                    &samples[start..],
                    &params,
                    &mut state.0,
                    &mut state.1,
                    t - DT_IMU,
                    t,
                );
            }
        }
        segments.push(samples);
        keyframes.push(state.clone());
    }
    (keyframes, segments)
}

pub fn pose_to_dvector(pose: &SE3) -> DVector<f64> {
    DVector::from_column_slice(pose.as_param_slice())
}

pub fn lm_solver(max_iterations: usize) -> LevenbergMarquardt {
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(max_iterations)
        .with_cost_tolerance(1e-12)
        .with_parameter_tolerance(1e-10);
    LevenbergMarquardt::with_config(config)
}

pub fn velocity_dvector(v: &Vector3<f64>) -> DVector<f64> {
    DVector::from_vec(vec![v.x, v.y, v.z])
}

/// Tight prior anchoring a variable — used instead of `fix_variable` (whose
/// discarded-step semantics under-correct free variables that share factors
/// with the fixed one).
pub fn anchor_rn(problem: &mut Problem, key: VarKey, params: &[f64]) {
    problem.add_residual_block_with_noise(
        &[key],
        Box::new(apex_solver::factors::pose::EuclideanPriorFactor::new(
            DVector::from_vec(params.to_vec()),
        )),
        None,
        NoiseModel::from_sigmas(&vec![1e-6; params.len()]).unwrap_or_else(|e| panic!("{e}")),
    );
}

/// Tight tangent-space prior anchoring an SE3 pose.
pub fn anchor_se3(problem: &mut Problem, key: VarKey, pose: &SE3) {
    problem.add_residual_block_with_noise(
        &[key],
        Box::new(PriorFactor::new(pose.clone())),
        None,
        NoiseModel::from_sigmas(&[1e-6; 6]).unwrap_or_else(|e| panic!("{e}")),
    );
}
