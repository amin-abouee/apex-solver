//! Stationarity constraints: zero-velocity (ZUPT) and zero-angular-rate (ZARU).
//!
//! Whenever a platform is detected to be at rest — a stopped vehicle, a foot in
//! stance phase — the fact that it is *not moving* is a measurement, and a very
//! precise one. Applying it bounds the drift that inertial integration would
//! otherwise accumulate through the stop, at no sensor cost.
//!
//! Stationarity detection itself is upstream: these factors are added by
//! whatever decides the platform is at rest, exactly as a data-association
//! front end supplies correspondences to a matching factor.

use apex_manifolds::LieGroup;
use apex_manifolds::se23::SE23;
use faer::prelude::ReborrowMut;
use nalgebra::Vector3;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::validate::expect_block_sizes;

/// Zero-velocity update (ZUPT): the platform's world velocity is zero.
///
/// ```text
/// r = v_world        (3D)
/// ```
///
/// # Parameter layout (1 block, 9 DOF)
///
/// ```text
/// params[0]: SE23 navigation state — 10D, 9 DOF
/// ```
///
/// Under the group's right perturbation `v ← v + R·δν`, so the Jacobian is
/// `[0 | 0 | R]` over the `[ρ, θ, ν]` tangent.
#[derive(Debug, Clone, Default)]
pub struct ZeroVelocityFactor;

impl ZeroVelocityFactor {
    /// Create the constraint. The measurement is implicit — velocity is zero.
    pub fn new() -> Self {
        Self
    }
}

impl Factor for ZeroVelocityFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let state = SE23::from_param_slice(params[0]);
        let velocity = state.velocity();
        residual[..3].copy_from_slice(velocity.as_slice());

        let Some(mut jac) = jacobian else { return };
        let rotation = state.rotation_matrix();
        for row in 0..3 {
            for col in 0..9 {
                *jac.rb_mut().get_mut(row, col) = 0.0;
            }
            for col in 0..3 {
                *jac.rb_mut().get_mut(row, 6 + col) = rotation[(row, col)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, 9)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE23::REP_SIZE],
            "ZeroVelocityFactor expects [SE23 state]",
        )
    }
}

/// Zero-angular-rate update (ZARU): while at rest, the gyroscope reads its own
/// bias.
///
/// ```text
/// r = ω_measured − b_g        (3D)
/// ```
///
/// This is what makes a stop *observe* the gyro bias rather than merely stop
/// accumulating it, so it complements [`ZeroVelocityFactor`] rather than
/// duplicating it.
///
/// # Parameter layout (1 block, 6 DOF)
///
/// ```text
/// params[0]: imu bias — 6D [bg, ba]
/// ```
#[derive(Debug, Clone)]
pub struct ZeroAngularRateFactor {
    /// Gyroscope reading averaged over the stationary interval [rad/s].
    measured_rate: Vector3<f64>,
}

impl ZeroAngularRateFactor {
    /// Create the constraint from the gyro reading during the stop.
    pub fn new(measured_rate: Vector3<f64>) -> Self {
        Self { measured_rate }
    }
}

impl Factor for ZeroAngularRateFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let bias_gyro = Vector3::new(params[0][0], params[0][1], params[0][2]);
        let r = self.measured_rate - bias_gyro;
        residual[..3].copy_from_slice(r.as_slice());

        let Some(mut jac) = jacobian else { return };
        for row in 0..3 {
            for col in 0..6 {
                *jac.rb_mut().get_mut(row, col) = 0.0;
            }
            *jac.rb_mut().get_mut(row, row) = -1.0;
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, 6)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(variables, &[6], "ZeroAngularRateFactor expects [bias (6D)]")
    }
}
