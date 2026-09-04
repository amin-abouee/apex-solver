//! The IMU bias random-walk edge used with the non-combined factors.
//!
//! [`ImuFactor`](super::se23::ImuFactor) and
//! [`Se23ImuFactor`](super::se23::Se23ImuFactor) produce a 9D kinematic
//! residual and share one bias variable per edge, exactly as GTSAM's
//! `ImuFactor` does. Bias *evolution* between keyframes is then a separate
//! constraint, and this module builds it: a zero-mean Gauss–Markov random walk
//! over the 6D bias `[b_g, b_a]`,
//!
//! ```text
//! b_j = b_i + w,     w ~ N(0, diag(σ_gw², σ_aw²) · Δt)
//! ```
//!
//! which is a [`BetweenFactor<Rn>`] with an identity relative value and the
//! noise below — no new residual code needed.
//!
//! The combined factors ([`CombinedImuFactor`](super::se23::CombinedImuFactor),
//! [`CombinedSe23ImuFactor`](super::se23::CombinedSe23ImuFactor)) embed this
//! walk in their trailing six residual rows. **Do not** add this edge alongside
//! one of them: the random walk would then be counted twice.

use apex_manifolds::rn::Rn;
use nalgebra::DVector;

use crate::core::CoreResult;
use crate::core::noise::NoiseModel;
use crate::factors::pose::BetweenFactor;

use super::types::ImuParameters;

/// Number of degrees of freedom in an IMU bias block: `[b_g(3), b_a(3)]`.
pub const BIAS_DIM: usize = 6;

/// The zero-mean relative constraint `b_j ⊖ b_i = 0` over a 6D bias variable.
///
/// Pair it with [`bias_random_walk_noise`] and register both on the two bias
/// variables the interval connects.
pub fn bias_random_walk() -> BetweenFactor<Rn> {
    BetweenFactor::new(Rn::new(DVector::zeros(BIAS_DIM)))
}

/// Noise model for the bias random walk over an interval of `dt` seconds.
///
/// Standard deviations are `σ_gw·√Δt` on the gyro bias and `σ_aw·√Δt` on the
/// accelerometer bias, taken from `params`.
///
/// # Errors
///
/// Returns an error if `dt` is not positive and finite, or if either
/// random-walk density in `params` is non-positive — a zero sigma would make
/// the information matrix singular.
pub fn bias_random_walk_noise(params: &ImuParameters, dt: f64) -> CoreResult<NoiseModel> {
    let sigmas = bias_random_walk_sigmas(params, dt)?;
    NoiseModel::from_sigmas(&sigmas)
}

/// The six standard deviations used by [`bias_random_walk_noise`].
fn bias_random_walk_sigmas(params: &ImuParameters, dt: f64) -> CoreResult<[f64; BIAS_DIM]> {
    use crate::core::CoreError;

    if !dt.is_finite() || dt <= 0.0 {
        return Err(CoreError::InvalidInput(format!(
            "bias random walk needs a positive, finite interval, got dt={dt}"
        )));
    }
    if params.sigma_gw_c <= 0.0 || params.sigma_aw_c <= 0.0 {
        return Err(CoreError::InvalidInput(format!(
            "bias random-walk densities must be positive, got sigma_gw_c={}, sigma_aw_c={}",
            params.sigma_gw_c, params.sigma_aw_c
        )));
    }

    let scale = dt.sqrt();
    let sg = params.sigma_gw_c * scale;
    let sa = params.sigma_aw_c * scale;
    Ok([sg, sg, sg, sa, sa, sa])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::Factor;

    #[test]
    fn random_walk_residual_is_zero_for_equal_biases() {
        let factor = bias_random_walk();
        let b = [0.01, -0.02, 0.03, 0.4, -0.5, 0.6];
        let mut r = vec![0.0; factor.residual_dim()];
        factor.linearize(&[&b, &b], &mut r, None);
        assert!(
            r.iter().all(|v| v.abs() < 1e-12),
            "expected zero residual, got {r:?}"
        );
    }

    #[test]
    fn random_walk_residual_is_the_bias_difference() {
        let factor = bias_random_walk();
        let b_i = [0.0; 6];
        let b_j = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let mut r = vec![0.0; 6];
        factor.linearize(&[&b_i, &b_j], &mut r, None);
        for (k, expected) in b_j.iter().enumerate() {
            assert!(
                (r[k].abs() - expected.abs()).abs() < 1e-12,
                "row {k}: {r:?}"
            );
        }
    }

    #[test]
    fn sigmas_scale_with_sqrt_dt() {
        let params = ImuParameters::default();
        let one = bias_random_walk_sigmas(&params, 1.0).unwrap_or_else(|e| panic!("{e}"));
        let four = bias_random_walk_sigmas(&params, 4.0).unwrap_or_else(|e| panic!("{e}"));
        for k in 0..BIAS_DIM {
            assert!(
                (four[k] - 2.0 * one[k]).abs() < 1e-15,
                "sigma[{k}] should double when dt quadruples"
            );
        }
    }

    #[test]
    fn rejects_non_positive_interval() {
        let params = ImuParameters::default();
        assert!(bias_random_walk_noise(&params, 0.0).is_err());
        assert!(bias_random_walk_noise(&params, -1.0).is_err());
        assert!(bias_random_walk_noise(&params, f64::NAN).is_err());
    }

    #[test]
    fn rejects_zero_random_walk_density() {
        let params = ImuParameters {
            sigma_gw_c: 0.0,
            ..ImuParameters::default()
        };
        assert!(bias_random_walk_noise(&params, 0.1).is_err());
    }
}
