//! Anchored inverse-**range** reprojection on the unit sphere, between two
//! `SE_2(3)` navigation states.
//!
//! The sibling [`super::InverseDepthSe23Factor`] measures on the normalized
//! plane, `(P_x/P_z, P_y/P_z)`. That is the VINS-Mono form and it is correct
//! for a pinhole. It fails on a wide field of view in two ways, and the
//! spectacular one is not the expensive one:
//!
//! - **Past 90°** the ray has `P_z < 0`. A `z`-axis cheirality test calls it
//!   "behind the camera" even though a 182° double sphere images it, and
//!   rescaling the anchor to `z = 1` mirrors it through the origin. Loud, but
//!   only a thin annulus of the image.
//! - **Everywhere outside the centre** the gnomonic Jacobian stretches by
//!   `1/cos²θ` — 4× at 60°, 33× at 80°, 131× at 85° — while the whitening
//!   `1 px / f` stays uniform, because that conversion is a small-angle
//!   identity. Quiet, and it deforms most of the Hessian.
//!
//! This factor measures on `S²` instead, so nothing about it degrades with the
//! angle off the optical axis:
//!
//! ```text
//! f     = m_host / ρ                      point in the host camera, m_host UNIT
//! p_bk  = R_bc·f + p_bc                   host body
//! p_w   = R_k·p_bk + p_k                  world
//! p_bi  = R_iᵀ·(p_w − p_i)                observer body
//! P     = R_bcᵀ·(p_bi − p_bc)             observer camera
//! n     = P / ‖P‖                         predicted bearing
//! r     = S·E(m_obs)ᵀ·(n − m_obs)         tangent plane at the measurement, 2 rows
//! ```
//!
//! where `E` is [`crate::factors::common::math::tangent_basis`], an orthonormal
//! `3×2` basis of `T_{m_obs}(S²)`.
//!
//! # `ρ` is an inverse *range*, not an inverse depth
//!
//! `m_host` is a **unit** bearing here, where the planar factor rescales it to
//! `z = 1`. So `1/ρ` is the distance along the ray rather than the `z`
//! coordinate — the two agree on the optical axis and diverge as `1/cos θ`,
//! without bound at the rim. A caller must use the same convention throughout:
//! triangulation gates, depth culling and re-anchoring all become range
//! questions.
//!
//! # Parameter layout (4 blocks, 25 minimal DOF)
//!
//! ```text
//! params[0]: SE23 host state     — 10D, 9 DOF
//! params[1]: SE23 observer state — 10D, 9 DOF
//! params[2]: SE3  T_bc           —  7D, 6 DOF   body-from-camera
//! params[3]: Rn(1) rho           —  1D, 1 DOF
//! ```
//!
//! # Noise
//!
//! The factor whitens internally, like every factor in
//! [`crate::factors::ranging::bearing`] — register it with
//! [`crate::core::noise::NoiseModel::null`]. `sqrt_information` is in
//! **radians⁻¹**, so an isotropic `sigma_px / focal` is the same number the
//! planar factor used, now meaning what it says.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::se23::SE23;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::math::tangent_basis;
use crate::factors::common::validate::expect_block_sizes;

use super::inverse_depth_se23::{Chain, JACOBIAN_COLUMNS, point_jacobians, write_jacobian};

/// Residual assigned to a landmark whose inverse range is not positive, in
/// tangent units **before** whitening.
///
/// `‖n − m‖ ≤ 2` for two unit vectors, so 4.0 is twice the largest error an
/// honest observation can produce. That is enough to make an infeasible `ρ`
/// strictly worse than any feasible configuration, and it is deliberately not
/// the `1e4` the planar factor uses: whitened by `1/σ ≈ 200 rad⁻¹` that would
/// be `2e6`, which under a robust loss says nothing more than 4.0 does while
/// making the reported cost unreadable.
const INFEASIBLE_RANGE_PENALTY: f64 = 4.0;

/// The penalty must beat the worst an honest observation can produce, or an
/// infeasible range becomes a cheap place for the solve to sit. `‖n − m‖ ≤ 2`
/// for two unit vectors, so the bound is exact rather than empirical.
const _: () = assert!(INFEASIBLE_RANGE_PENALTY > 2.0);

/// Norm below which the observer-frame point carries no direction.
const MIN_POINT_NORM: f64 = 1.0e-16;

/// Reprojection of an inverse-range landmark, anchored in a host keyframe and
/// measured in the tangent plane of the unit sphere.
///
/// See the module documentation for the geometry and for why this exists
/// alongside [`super::InverseDepthSe23Factor`].
pub struct InverseDepthBearingSe23Factor {
    /// Unit bearing of the feature in the host camera.
    host_bearing: Vector3<f64>,
    /// Unit bearing measured in the observer camera.
    measured_bearing: Vector3<f64>,
    /// Orthonormal basis of the tangent plane at `measured_bearing`.
    tangent_basis: SMatrix<f64, 3, 2>,
    /// Square-root information, in radians⁻¹.
    sqrt_information: SMatrix<f64, 2, 2>,
}

impl InverseDepthBearingSe23Factor {
    /// Build the factor from two bearings and a `2×2` square-root information.
    ///
    /// # Arguments
    ///
    /// * `host_bearing` — direction in the host camera; normalized here.
    /// * `measured_bearing` — direction in the observer camera; normalized here.
    /// * `sqrt_information` — `2×2`, in radians⁻¹.
    ///
    /// # Errors
    ///
    /// A bearing whose norm is not finite and positive has no direction, and
    /// normalizing it would produce a factor whose residual is `NaN` for every
    /// input. Rejecting it here is the only place the caller can still act.
    pub fn new(
        host_bearing: Vector3<f64>,
        measured_bearing: Vector3<f64>,
        sqrt_information: SMatrix<f64, 2, 2>,
    ) -> Result<Self, String> {
        let host = unit(&host_bearing).ok_or_else(|| {
            format!("InverseDepthBearingSe23Factor: host bearing {host_bearing:?} has no direction")
        })?;
        let measured = unit(&measured_bearing).ok_or_else(|| {
            format!(
                "InverseDepthBearingSe23Factor: measured bearing {measured_bearing:?} has no direction"
            )
        })?;

        Ok(Self {
            host_bearing: host,
            measured_bearing: measured,
            tangent_basis: tangent_basis(&measured),
            sqrt_information,
        })
    }

    /// Build with isotropic noise, `sigma` in radians.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], plus a non-finite or non-positive `sigma`.
    pub fn new_isotropic(
        host_bearing: Vector3<f64>,
        measured_bearing: Vector3<f64>,
        sigma: f64,
    ) -> Result<Self, String> {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(format!(
                "InverseDepthBearingSe23Factor: sigma must be finite and positive, got {sigma}"
            ));
        }
        let sqrt_information = SMatrix::<f64, 2, 2>::identity() * (1.0 / sigma);
        Self::new(host_bearing, measured_bearing, sqrt_information)
    }
}

/// `Some(v/‖v‖)` when `v` has a usable direction.
fn unit(v: &Vector3<f64>) -> Option<Vector3<f64>> {
    let norm = v.norm();
    (norm.is_finite() && norm > MIN_POINT_NORM).then(|| v / norm)
}

impl Factor for InverseDepthBearingSe23Factor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let [host_params, observer_params, extrinsic_params, rho_params] = params else {
            return;
        };

        let host = SE23::from_param_slice(host_params);
        let observer = SE23::from_param_slice(observer_params);
        let extrinsic = SE3::from_param_slice(extrinsic_params);
        let rho = rho_params[0];

        // An inverse range at or behind the anchor is not a point at all, so it
        // gets a flat penalty and a **zero** Jacobian: leaving a gradient here
        // would make an invalid anchor a cheap direction to escape along.
        if !rho.is_finite() || rho <= 0.0 {
            let penalty =
                self.sqrt_information * SVector::<f64, 2>::repeat(INFEASIBLE_RANGE_PENALTY);
            residual[0] = penalty[0];
            residual[1] = penalty[1];
            if let Some(mut jac) = jacobian {
                write_jacobian(&SMatrix::<f64, 2, JACOBIAN_COLUMNS>::zeros(), &mut jac);
            }
            return;
        }

        let r_bc = extrinsic.rotation_so3().rotation_matrix();
        let p_bc = extrinsic.translation();
        let r_k = host.rotation_matrix();
        let r_i = observer.rotation_matrix();

        let f = self.host_bearing / rho;
        let p_bk = r_bc * f + p_bc;
        let p_w = r_k * p_bk + host.translation();
        let p_bi = r_i.transpose() * (p_w - observer.translation());
        let point = r_bc.transpose() * (p_bi - p_bc);

        let norm = point.norm();
        if !norm.is_finite() || norm < MIN_POINT_NORM {
            // The landmark sits on the observer's optical centre, where no
            // direction is defined. Zero residual and zero Jacobian, matching
            // `ranging::bearing`, rather than a penalty pointing nowhere.
            residual[0] = 0.0;
            residual[1] = 0.0;
            if let Some(mut jac) = jacobian {
                write_jacobian(&SMatrix::<f64, 2, JACOBIAN_COLUMNS>::zeros(), &mut jac);
            }
            return;
        }

        let n_est = point / norm;
        let error = self.sqrt_information
            * (self.tangent_basis.transpose() * (n_est - self.measured_bearing));
        residual[0] = error[0];
        residual[1] = error[1];

        let Some(mut jac) = jacobian else {
            return;
        };

        // ∂n/∂P = (I₃ − n·nᵀ)/‖P‖, the unit-vector normalization Jacobian. It
        // is what replaces the planar factor's ∂π/∂P, and unlike ∂π/∂P it is
        // bounded everywhere and has no preferred axis.
        let dn_dp = (Matrix3::identity() - n_est * n_est.transpose()) / norm;
        let prefix = self.sqrt_information * self.tangent_basis.transpose() * dn_dp;

        let chain = Chain {
            a: r_bc.transpose() * r_i.transpose() * r_k,
            r_bc,
            p_bk,
            p_bi,
            point,
            f,
            host_bearing: self.host_bearing,
            rho,
        };
        write_jacobian(&(prefix * point_jacobians(&chain)), &mut jac);
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2, JACOBIAN_COLUMNS)
    }

    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE23::REP_SIZE, SE23::REP_SIZE, SE3::REP_SIZE, 1],
            "InverseDepthBearingSe23Factor expects [SE23 host, SE23 observer, SE3 T_bc, Rn(1) rho]",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::visual::InverseDepthSe23Factor;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::SE3Tangent;
    use apex_manifolds::se23::SE23Tangent;
    use nalgebra::{DMatrix, UnitQuaternion, Vector2};

    /// Host state, observer state, `T_bc` and `rho`, as parameter vectors.
    #[derive(Clone)]
    struct Setup {
        host: Vec<f64>,
        observer: Vec<f64>,
        extrinsic: Vec<f64>,
        rho: Vec<f64>,
    }

    fn se23(t: Vector3<f64>, v: Vector3<f64>, q: UnitQuaternion<f64>) -> Vec<f64> {
        SE23::new(t, v, q).as_param_slice().to_vec()
    }

    fn se3(t: Vector3<f64>, q: UnitQuaternion<f64>) -> Vec<f64> {
        SE3::new(t, q).as_param_slice().to_vec()
    }

    fn setup(rho: f64) -> Setup {
        Setup {
            host: se23(
                Vector3::new(0.1, -0.2, 0.05),
                Vector3::new(0.8, 0.1, -0.2),
                UnitQuaternion::from_euler_angles(0.03, -0.07, 0.11),
            ),
            observer: se23(
                Vector3::new(0.9, 0.15, -0.1),
                Vector3::new(0.7, -0.3, 0.05),
                UnitQuaternion::from_euler_angles(-0.05, 0.09, -0.04),
            ),
            extrinsic: se3(
                Vector3::new(0.02, -0.065, 0.01),
                UnitQuaternion::from_euler_angles(0.01, -1.55, 0.02),
            ),
            rho: vec![rho],
        }
    }

    /// A bearing 7° off the optical axis — the regime every test of the planar
    /// factor lives in.
    fn near_axis() -> Vector3<f64> {
        Vector3::new(0.12, -0.05, 1.0).normalize()
    }

    /// A bearing 150° off the host optical axis, which a 182° double sphere
    /// images and a normalized-plane residual cannot represent.
    ///
    /// Chosen so that the **observer**-frame point also lands past the equator
    /// for every `rho` used below — being wide in the host frame is not enough,
    /// since the chain can rotate the ray back in front of the second camera.
    /// `a_landmark_behind_the_image_plane_is_an_ordinary_observation` asserts
    /// that it still holds.
    fn past_the_equator() -> Vector3<f64> {
        Vector3::new(-0.4813, 0.1356, -0.866).normalize()
    }

    /// The transform chain, derived independently of the factor so a
    /// zero-residual test is a real check rather than a tautology.
    fn observer_point(s: &Setup, host_bearing: &Vector3<f64>) -> Vector3<f64> {
        let host = SE23::from_param_slice(&s.host);
        let observer = SE23::from_param_slice(&s.observer);
        let extrinsic = SE3::from_param_slice(&s.extrinsic);
        let r_bc = extrinsic.rotation_so3().rotation_matrix();
        let p_bc = extrinsic.translation();

        let f = host_bearing.normalize() / s.rho[0];
        let p_bk = r_bc * f + p_bc;
        let p_w = host.rotation_matrix() * p_bk + host.translation();
        let p_bi = observer.rotation_matrix().transpose() * (p_w - observer.translation());
        r_bc.transpose() * (p_bi - p_bc)
    }

    /// The factor whose measurement is the predicted bearing rotated by
    /// `offset` radians inside the tangent plane, so `offset == 0` sits exactly
    /// at the minimum and anything else exercises the Jacobian away from it.
    fn factor_at(
        s: &Setup,
        host_bearing: &Vector3<f64>,
        offset: f64,
    ) -> InverseDepthBearingSe23Factor {
        let predicted = observer_point(s, host_bearing).normalize();
        let nudge = tangent_basis(&predicted).column(0) * offset;
        let measured = (predicted + nudge).normalize();
        match InverseDepthBearingSe23Factor::new_isotropic(*host_bearing, measured, 0.005) {
            Ok(factor) => factor,
            Err(message) => panic!("{message}"),
        }
    }

    fn residual_of(factor: &InverseDepthBearingSe23Factor, s: &Setup) -> Vec<f64> {
        let mut residual = vec![0.0f64; 2];
        factor.linearize(
            &[&s.host, &s.observer, &s.extrinsic, &s.rho],
            &mut residual,
            None,
        );
        residual
    }

    fn jacobian_of(factor: &InverseDepthBearingSe23Factor, s: &Setup) -> DMatrix<f64> {
        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0f64; rows];
        let mut buf = vec![0.0f64; rows * cols];
        let jac = faer::mat::MatMut::from_column_major_slice_mut(&mut buf, rows, cols);
        factor.linearize(
            &[&s.host, &s.observer, &s.extrinsic, &s.rho],
            &mut residual,
            Some(jac),
        );
        DMatrix::from_column_slice(rows, cols, &buf)
    }

    #[test]
    fn zero_residual_at_the_exact_solution() {
        for bearing in [near_axis(), past_the_equator()] {
            let s = setup(0.4);
            let factor = factor_at(&s, &bearing, 0.0);
            let r = residual_of(&factor, &s);
            assert!(
                r[0].abs() < 1e-12 && r[1].abs() < 1e-12,
                "residual {r:?} should vanish at the generating configuration"
            );
        }
    }

    /// The point this factor exists for: a landmark past the observer's equator
    /// is a normal observation, not a cheirality violation.
    ///
    /// The assertion on `point.z` is the load-bearing half — without it a
    /// change to `setup` could quietly move the case back in front of the
    /// camera and the test would keep passing while testing nothing.
    #[test]
    fn a_landmark_behind_the_image_plane_is_an_ordinary_observation() {
        let s = setup(0.9);
        let point = observer_point(&s, &past_the_equator());
        assert!(
            point.z < 0.0,
            "the fixture must place the landmark past the equator, got z = {}",
            point.z
        );

        let factor = factor_at(&s, &past_the_equator(), 0.0);
        let r = residual_of(&factor, &s);
        assert!(r[0].abs() < 1e-12 && r[1].abs() < 1e-12);

        let jacobian = jacobian_of(&factor, &s);
        assert!(jacobian.iter().all(|v| v.is_finite()));
        assert!(
            jacobian.norm() > 1e-6,
            "a valid observation must carry a gradient, got {}",
            jacobian.norm()
        );
    }

    /// The same configuration through the planar factor, so the improvement is
    /// recorded as a number rather than asserted in prose.
    ///
    /// The failure is quieter than a cheirality penalty and worse for it.
    /// `InverseDepthSe23Factor::new` rescales its anchor to `z = 1`; for a
    /// bearing with `z < 0` that division **mirrors the ray through the
    /// origin**, so the factor models a landmark on the opposite side of the
    /// host camera and never says so. Fed the measurement its own geometry
    /// generates, it should return zero and does not.
    #[test]
    fn the_planar_factor_mismodels_a_past_the_equator_anchor() {
        let s = setup(0.9);
        let point = observer_point(&s, &past_the_equator());
        assert!(point.z < 0.0, "fixture must be past the equator");

        let truth = Vector2::new(point.x / point.z, point.y / point.z);
        let planar = InverseDepthSe23Factor::new(past_the_equator(), truth);
        let mut residual = vec![0.0f64; 2];
        planar.linearize(
            &[&s.host, &s.observer, &s.extrinsic, &s.rho],
            &mut residual,
            None,
        );

        let error = (residual[0] * residual[0] + residual[1] * residual[1]).sqrt();
        assert!(
            error > 0.1,
            "the planar factor should mismodel this anchor; residual {residual:?}"
        );

        // The bearing factor, given the same truth as a direction, is exact.
        let bearing = factor_at(&s, &past_the_equator(), 0.0);
        let r = residual_of(&bearing, &s);
        assert!(r[0].abs() < 1e-12 && r[1].abs() < 1e-12);
    }

    /// The whole Jacobian, block by block, at both a near-axis and a
    /// past-the-equator bearing. Without this the `T_bc` sum, the `rho` term
    /// and the tangent-space prefix are unverifiable by inspection.
    #[test]
    fn jacobian_matches_finite_differences() {
        const EPS: f64 = 1e-7;

        // The upper bound is 1.0 rather than 5.0: at a 0.2 m range the
        // landmark sits inside the 0.8 m baseline, and `past_the_equator`
        // stops being past the equator in the observer frame.
        for bearing in [near_axis(), past_the_equator()] {
            for rho in [0.2, 0.5, 1.0] {
                let s = setup(rho);
                let factor = factor_at(&s, &bearing, 0.03);
                let analytic = jacobian_of(&factor, &s);

                // Host and observer: 9 DOF each, in their own SE23 chart.
                for (block, base) in [(0usize, 0usize), (1, 9)] {
                    for axis in 0..9 {
                        let mut tangent = [0.0f64; 9];
                        let source = if block == 0 { &s.host } else { &s.observer };

                        tangent[axis] = EPS;
                        let plus = SE23::from_param_slice(source)
                            .right_plus(&SE23Tangent::from_slice(&tangent), None, None)
                            .as_param_slice()
                            .to_vec();
                        tangent[axis] = -EPS;
                        let minus = SE23::from_param_slice(source)
                            .right_plus(&SE23Tangent::from_slice(&tangent), None, None)
                            .as_param_slice()
                            .to_vec();

                        let mut perturbed = s.clone();
                        if block == 0 {
                            perturbed.host = plus;
                        } else {
                            perturbed.observer = plus;
                        }
                        let rp = residual_of(&factor, &perturbed);
                        if block == 0 {
                            perturbed.host = minus;
                        } else {
                            perturbed.observer = minus;
                        }
                        let rm = residual_of(&factor, &perturbed);

                        for row in 0..2 {
                            let fd = (rp[row] - rm[row]) / (2.0 * EPS);
                            let col = base + axis;
                            assert!(
                                (fd - analytic[(row, col)]).abs() < 1e-4,
                                "rho {rho}, column {col}, row {row}: fd {fd} vs analytic {}",
                                analytic[(row, col)]
                            );
                        }
                    }
                }

                // T_bc: 6 DOF in the SE3 chart, columns 18..24.
                for axis in 0..6 {
                    let mut tangent = [0.0f64; 6];
                    let mut perturbed = s.clone();

                    tangent[axis] = EPS;
                    perturbed.extrinsic = SE3::from_param_slice(&s.extrinsic)
                        .right_plus(&SE3Tangent::from_slice(&tangent), None, None)
                        .as_param_slice()
                        .to_vec();
                    let rp = residual_of(&factor, &perturbed);

                    tangent[axis] = -EPS;
                    perturbed.extrinsic = SE3::from_param_slice(&s.extrinsic)
                        .right_plus(&SE3Tangent::from_slice(&tangent), None, None)
                        .as_param_slice()
                        .to_vec();
                    let rm = residual_of(&factor, &perturbed);

                    for row in 0..2 {
                        let fd = (rp[row] - rm[row]) / (2.0 * EPS);
                        let col = 18 + axis;
                        assert!(
                            (fd - analytic[(row, col)]).abs() < 1e-4,
                            "rho {rho}, T_bc column {col}, row {row}: fd {fd} vs analytic {}",
                            analytic[(row, col)]
                        );
                    }
                }

                // rho: column 24.
                let mut perturbed = s.clone();
                perturbed.rho = vec![rho + EPS];
                let rp = residual_of(&factor, &perturbed);
                perturbed.rho = vec![rho - EPS];
                let rm = residual_of(&factor, &perturbed);
                for row in 0..2 {
                    let fd = (rp[row] - rm[row]) / (2.0 * EPS);
                    assert!(
                        (fd - analytic[(row, 24)]).abs() < 1e-4,
                        "rho {rho}, rho column, row {row}: fd {fd} vs analytic {}",
                        analytic[(row, 24)]
                    );
                }
            }
        }
    }

    /// The two `T_bc` contributions cancel exactly for a self-observation, and
    /// that is correct: reprojecting a landmark into its own host frame is the
    /// identity map, so the extrinsic cannot move it. Asserted because a zero
    /// block otherwise reads as a missing term.
    #[test]
    fn the_extrinsic_cancels_on_a_self_observation() {
        let mut s = setup(0.5);
        s.observer = s.host.clone();
        let factor = factor_at(&s, &near_axis(), 0.02);
        let analytic = jacobian_of(&factor, &s);

        let extrinsic_block = analytic.view((0, 18), (2, 6)).norm();
        assert!(
            extrinsic_block < 1e-10,
            "expected exact cancellation, got {extrinsic_block:.3e}"
        );
    }

    /// A bearing measures direction only, so it says nothing about how fast the
    /// body carrying the camera is moving.
    #[test]
    fn velocity_columns_are_zero() {
        let s = setup(0.7);
        let factor = factor_at(&s, &past_the_equator(), 0.02);
        let analytic = jacobian_of(&factor, &s);

        assert!(analytic.view((0, 6), (2, 3)).norm() < 1e-14);
        assert!(analytic.view((0, 15), (2, 3)).norm() < 1e-14);
    }

    #[test]
    fn a_non_positive_rho_gets_a_flat_penalty_and_no_gradient() {
        let mut s = setup(0.5);
        let factor = factor_at(&s, &near_axis(), 0.0);
        s.rho = vec![-0.2];

        let r = residual_of(&factor, &s);
        // sqrt_information is I/0.005, so the whitened penalty is 4/0.005.
        assert!((r[0] - INFEASIBLE_RANGE_PENALTY / 0.005).abs() < 1e-9);
        assert!((r[1] - INFEASIBLE_RANGE_PENALTY / 0.005).abs() < 1e-9);
        assert!(jacobian_of(&factor, &s).norm() < 1e-14);
    }

    /// Optical infinity must stay representable — that is the whole reason the
    /// landmark is a scalar inverse range rather than a point.
    #[test]
    fn a_landmark_at_optical_infinity_still_has_a_finite_residual() {
        let mut s = setup(1.0);
        let factor = factor_at(&s, &near_axis(), 0.01);
        s.rho = vec![1.0e-6];

        let r = residual_of(&factor, &s);
        assert!(r.iter().all(|v| v.is_finite()), "residual {r:?}");
        assert!(jacobian_of(&factor, &s).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_bearing_without_a_direction_is_rejected() {
        let Err(message) =
            InverseDepthBearingSe23Factor::new_isotropic(Vector3::zeros(), Vector3::z(), 0.005)
        else {
            panic!("expected a zero host bearing to be rejected");
        };
        assert!(message.contains("host bearing"), "{message}");

        let Err(message) =
            InverseDepthBearingSe23Factor::new_isotropic(Vector3::z(), Vector3::zeros(), 0.005)
        else {
            panic!("expected a zero measured bearing to be rejected");
        };
        assert!(message.contains("measured bearing"), "{message}");
    }

    #[test]
    fn a_non_positive_sigma_is_rejected() {
        assert!(
            InverseDepthBearingSe23Factor::new_isotropic(Vector3::z(), Vector3::z(), 0.0).is_err()
        );
    }

    #[test]
    fn the_factor_whitens_internally() {
        let factor = factor_at(&setup(0.5), &near_axis(), 0.0);
        assert!(factor.whitens_internally());
        assert_eq!(factor.residual_dim(), 2);
        assert_eq!(factor.jacobian_shape(), (2, 25));
    }
}
