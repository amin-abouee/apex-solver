//! The combined IMU factor in an estimated gravity frame.
//!
//! [`CombinedImuFactor`] takes gravity along the world `z`. A map built from
//! a still start inherits that start's tilt: at rest, an accelerometer bias
//! across gravity is indistinguishable from tilt, and a LiDAR map is held in
//! the world frame thereafter. Here the world frame stays the map's and
//! gravity's direction in it is a variable `R_g ∈ SO3` (gravity is
//! `R_g·(0, 0, −g)`): the IMU factor is evaluated on the states expressed in
//! the gravity frame,
//!
//! ```text
//! X' = T_g⁻¹·X,     T_g = SE₂(3)(R_g, 0, 0)
//! ```
//!
//! Rotation of the rig makes the tilt observable and the bias separable.
//!
//! # Jacobians
//!
//! A right perturbation of `X` is a right perturbation of `X'`, so the state
//! columns are the inner factor's. For `R_g ← R_g·Exp(ψ)`,
//! `X' ← Exp(−ξ)·X' = X'·Exp(−Ad(X'⁻¹)·ξ)` with `ξ = (0, ψ, 0)`, so
//! `∂r/∂ψ = Σ_{i,j} ∂r/∂X'·(−Ad(X'⁻¹))[:, θ]`. The `R_g` yaw is unobservable
//! (gravity is invariant to it): hold or prior it.

use apex_manifolds::LieGroup;
use apex_manifolds::se23::SE23;
use apex_manifolds::so3::SO3;
use faer::Mat;
use faer::prelude::ReborrowMut;
use nalgebra::{SMatrix, Vector3};

use super::factors::CombinedImuFactor;
use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::validate::expect_block_sizes;

/// [`CombinedImuFactor`] with gravity's direction a variable; see the module
/// docs.
///
/// # Parameter layout (5 blocks, 33 minimal DOF)
///
/// ```text
/// params[0]: SE23 state i   params[2]: SE23 state j   params[4]: SO3 R_g
/// params[1]: bias i         params[3]: bias j
/// ```
pub struct GravityFrameImuFactor {
    inner: CombinedImuFactor,
}

impl GravityFrameImuFactor {
    /// Wrap a combined IMU factor.
    pub fn new(inner: CombinedImuFactor) -> Self {
        Self { inner }
    }
}

/// `T_g⁻¹·X` for gravity rotation `rotation`.
pub fn into_gravity_frame(rotation: &SO3, state: &SE23) -> SE23 {
    let frame = SE23::from_components(Vector3::zeros(), Vector3::zeros(), rotation.clone());
    frame.inverse(None).compose(state, None, None)
}

impl Factor for GravityFrameImuFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let rotation = SO3::from_param_slice(params[4]);
        let state_i = into_gravity_frame(&rotation, &SE23::from_param_slice(params[0]));
        let state_j = into_gravity_frame(&rotation, &SE23::from_param_slice(params[2]));
        let inner_params = [
            state_i.as_param_slice(),
            params[1],
            state_j.as_param_slice(),
            params[3],
        ];
        let Some(mut jac) = jacobian else {
            self.inner.linearize(&inner_params, residual, None);
            return;
        };
        let mut inner = Mat::<f64>::zeros(15, 30);
        self.inner
            .linearize(&inner_params, residual, Some(inner.as_mut()));
        for row in 0..15 {
            for col in 0..30 {
                *jac.rb_mut().get_mut(row, col) = inner[(row, col)];
            }
        }
        // −Ad(X'⁻¹) restricted to the rotation columns, for both states.
        let lift = |state: &SE23| -> SMatrix<f64, 9, 3> {
            -state
                .inverse(None)
                .adjoint()
                .fixed_view::<9, 3>(0, 3)
                .into_owned()
        };
        let (lift_i, lift_j) = (lift(&state_i), lift(&state_j));
        for row in 0..15 {
            for k in 0..3 {
                let mut value = 0.0;
                for m in 0..9 {
                    value +=
                        inner[(row, m)] * lift_i[(m, k)] + inner[(row, 15 + m)] * lift_j[(m, k)];
                }
                *jac.rb_mut().get_mut(row, 30 + k) = value;
            }
        }
    }

    fn residual_dim(&self) -> usize {
        15
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (15, 33)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE23::REP_SIZE, 6, SE23::REP_SIZE, 6, SO3::REP_SIZE],
            "GravityFrameImuFactor expects [SE23 state_i, bias_i, SE23 state_j, bias_j, SO3 R_g]",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::imu::preintegration::ImuPreintegration;
    use crate::factors::imu::types::{
        ImuMeasurement, ImuParameters, ImuSensorReadings, SpeedAndBias,
    };
    use apex_manifolds::Tangent;
    use apex_manifolds::se23::SE23Tangent;
    use apex_manifolds::so3::SO3Tangent;
    use nalgebra::UnitQuaternion;

    /// One second of a rig yawing and accelerating sideways in a frame whose
    /// gravity is tilted by `tilt`: the specific force is `Rᵀ(a − g_w)`.
    fn scenario(tilt: &UnitQuaternion<f64>) -> (CombinedImuFactor, SE23, SE23) {
        let params = ImuParameters {
            sigma_g_c: 0.005,
            sigma_a_c: 0.05,
            sigma_gw_c: 0.01,
            sigma_aw_c: 0.01,
            g: 9.81,
            ..ImuParameters::default()
        };
        let dt = 0.005;
        let gravity = tilt * Vector3::new(0.0, 0.0, -9.81);
        let accel = Vector3::new(0.2, 0.1, 0.0);
        let rate = Vector3::new(0.0, 0.0, 0.3);
        let attitude = |t: f64| UnitQuaternion::from_scaled_axis(rate * t);
        let samples: Vec<ImuMeasurement> = (0..=200)
            .map(|k| {
                let t = f64::from(k) * dt;
                ImuMeasurement::new(
                    t,
                    ImuSensorReadings {
                        gyroscopes: rate,
                        accelerometers: attitude(t).inverse() * (accel - gravity),
                    },
                )
            })
            .collect();
        let preint =
            ImuPreintegration::new(samples.into(), params, 0.0, 1.0, &SpeedAndBias::zeros());
        let state_i = SE23::new(
            Vector3::zeros(),
            Vector3::zeros(),
            UnitQuaternion::identity(),
        );
        let state_j = SE23::new(accel * 0.5, accel, attitude(1.0));
        (CombinedImuFactor::new(preint), state_i, state_j)
    }

    fn residual(f: &GravityFrameImuFactor, blocks: &[Vec<f64>]) -> Vec<f64> {
        let params: Vec<&[f64]> = blocks.iter().map(Vec::as_slice).collect();
        let mut r = vec![0.0; 15];
        f.linearize(&params, &mut r, None);
        r
    }

    fn cost(r: &[f64]) -> f64 {
        r.iter().map(|v| v * v).sum()
    }

    /// The tilted world's residual is large with gravity along `z` and
    /// vanishes at the true gravity rotation.
    #[test]
    fn the_true_gravity_rotation_explains_a_tilted_world() {
        let tilt = UnitQuaternion::from_euler_angles(0.02, -0.015, 0.0);
        let (inner, state_i, state_j) = scenario(&tilt);
        let factor = GravityFrameImuFactor::new(inner);
        let blocks = |rotation: &SO3| {
            vec![
                state_i.as_param_slice().to_vec(),
                vec![0.0; 6],
                state_j.as_param_slice().to_vec(),
                vec![0.0; 6],
                rotation.as_param_slice().to_vec(),
            ]
        };
        let level = cost(&residual(&factor, &blocks(&SO3::identity())));
        let tilted = cost(&residual(&factor, &blocks(&SO3::new(tilt))));
        assert!(tilted < 1e-3 * level, "tilted {tilted} vs level {level}");
        let shapes = (factor.residual_dim(), factor.jacobian_shape());
        assert_eq!(shapes, (15, (15, 33)));
    }

    /// Every Jacobian column against central finite differences, away from
    /// the solution.
    #[test]
    fn gravity_frame_jacobians_match_finite_differences() {
        let tilt = UnitQuaternion::from_euler_angles(0.02, -0.015, 0.0);
        let (inner, state_i, state_j) = scenario(&tilt);
        let factor = GravityFrameImuFactor::new(inner);
        let nudge = |s: &SE23, k: f64| {
            let t: Vec<f64> = (0..9).map(|i| 0.01 * k * ((i % 4) as f64 - 1.5)).collect();
            s.right_plus(&SE23Tangent::from_slice(&t), None, None)
        };
        let base = vec![
            nudge(&state_i, 1.0).as_param_slice().to_vec(),
            vec![0.002, -0.001, 0.003, 0.02, -0.01, 0.03],
            nudge(&state_j, -1.0).as_param_slice().to_vec(),
            vec![0.001, 0.0, -0.002, 0.01, 0.02, -0.01],
            SO3::new(UnitQuaternion::from_euler_angles(0.01, 0.005, 0.2))
                .as_param_slice()
                .to_vec(),
        ];
        let params: Vec<&[f64]> = base.iter().map(Vec::as_slice).collect();
        let mut r = vec![0.0; 15];
        let mut analytic = Mat::<f64>::zeros(15, 33);
        factor.linearize(&params, &mut r, Some(analytic.as_mut()));
        let eps = 1e-6;
        let shifted = |block: usize, k: usize, sign: f64| -> Vec<f64> {
            let mut blocks = base.clone();
            blocks[block] = match block {
                0 | 2 => {
                    let mut t = [0.0; 9];
                    t[k] = sign * eps;
                    SE23::from_param_slice(&base[block])
                        .right_plus(&SE23Tangent::from_slice(&t), None, None)
                        .as_param_slice()
                        .to_vec()
                }
                4 => {
                    let mut t = [0.0; 3];
                    t[k] = sign * eps;
                    SO3::from_param_slice(&base[4])
                        .right_plus(&SO3Tangent::from_slice(&t), None, None)
                        .as_param_slice()
                        .to_vec()
                }
                _ => {
                    let mut b = base[block].clone();
                    b[k] += sign * eps;
                    b
                }
            };
            residual(&factor, &blocks)
        };
        let layout = [(0, 9), (1, 6), (2, 9), (3, 6), (4, 3)];
        let mut col = 0;
        for (block, dof) in layout {
            for k in 0..dof {
                let (rp, rm) = (shifted(block, k, 1.0), shifted(block, k, -1.0));
                for row in 0..15 {
                    let fd = (rp[row] - rm[row]) / (2.0 * eps);
                    let a = analytic[(row, col)];
                    let scale = 1.0 + a.abs().max(fd.abs());
                    assert!(
                        (a - fd).abs() / scale < 1e-3,
                        "block {block} dof {k} row {row}: analytic {a} fd {fd}"
                    );
                }
                col += 1;
            }
        }
    }
}
