//! LiDAR factors on a navigation state `SE₂(3)` and a LiDAR extrinsic.
//!
//! The other LiDAR factors in this module act on bare `SE3` poses. A
//! visual-inertial window carries `X = (R, p, v) ∈ SE₂(3)` per frame (the IMU
//! pose plus velocity) and the LiDAR sits at its own extrinsic `T_SL`, so a
//! scan point reaches the world through both:
//!
//! ```text
//! s = R_SL·l + t_SL          (LiDAR → body/IMU)
//! w = R·s + p                (body → world)
//! ```
//!
//! # Jacobians (right perturbations, the apex-manifolds convention)
//!
//! `X ← X·Exp([δρ, δθ, δν])` moves `p ← p + R·δρ`, `R ← R·Exp(δθ)`,
//! `v ← v + R·δν`; `T_SL ← T_SL·Exp([δρₑ, δθₑ])`:
//!
//! ```text
//! ∂w/∂δρ  = R          ∂w/∂δθ  = −R·[s]×      ∂w/∂δν = 0
//! ∂w/∂δρₑ = R·R_SL     ∂w/∂δθₑ = −R·R_SL·[l]×
//! ```
//!
//! Velocity columns are exactly zero: a LiDAR point constrains the pose only.

use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, Vector3};

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::se23::SE23;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::math::skew;
use crate::factors::common::validate::expect_block_sizes;

use super::plane::Plane;

/// A LiDAR point carried to the world, with its derivatives.
#[derive(Clone, Debug)]
pub struct LidarPointChain {
    /// The point in the body (IMU) frame, `s = R_SL·l + t_SL`.
    pub body: Vector3<f64>,
    /// The point in the world, `w = R·s + p`.
    pub world: Vector3<f64>,
    /// `∂w/∂X`, columns `[δρ, δθ, δν]` of the `SE₂(3)` tangent.
    pub d_state: SMatrix<f64, 3, 9>,
    /// `∂w/∂T_SL`, columns `[δρₑ, δθₑ]` of the `SE3` tangent.
    pub d_extrinsic: SMatrix<f64, 3, 6>,
}

/// Carry LiDAR point `point_l` through `T_SL` and the state `X` to the world.
pub fn lidar_point_chain(state: &SE23, extrinsic: &SE3, point_l: &Vector3<f64>) -> LidarPointChain {
    let rotation: Matrix3<f64> = state.rotation_matrix();
    let r_sl: Matrix3<f64> = extrinsic.rotation_so3().rotation_matrix();
    let body = r_sl * point_l + extrinsic.translation();
    let world = rotation * body + state.translation();

    let mut d_state = SMatrix::<f64, 3, 9>::zeros();
    d_state.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    d_state
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-rotation * skew(&body)));

    let r_wl = rotation * r_sl;
    let mut d_extrinsic = SMatrix::<f64, 3, 6>::zeros();
    d_extrinsic.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_wl);
    d_extrinsic
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-r_wl * skew(point_l)));

    LidarPointChain {
        body,
        world,
        d_state,
        d_extrinsic,
    }
}

/// Standard deviation of a point-to-plane residual from the plane's and the
/// point's uncertainty — the VoxelMap / FAST-LIVO2 weighting:
///
/// ```text
/// σ² = nᵀ·R_WL·Σ_l·R_WLᵀ·n  +  J_π·Σ_π·J_πᵀ,     J_π = [(w − q)ᵀ, −nᵀ]
/// ```
///
/// `plane_covariance` is the `6×6` covariance of `(n, q)` — normal, then the
/// plane's centroid `q` — from the map points that fit it; `point_covariance`
/// is the scan point's `3×3` covariance in the LiDAR frame (range and bearing
/// noise); `world` the point carried to the world at the predicted pose and
/// `rotation_wl = R·R_SL`.
///
/// Returns `None` when the variance is not finite and positive.
pub fn point_to_plane_sigma(
    plane: &Plane,
    centroid: &Vector3<f64>,
    plane_covariance: &SMatrix<f64, 6, 6>,
    point_covariance: &Matrix3<f64>,
    world: &Vector3<f64>,
    rotation_wl: &Matrix3<f64>,
) -> Option<f64> {
    let n = plane.normal;
    let point_term =
        (n.transpose() * rotation_wl * point_covariance * rotation_wl.transpose() * n)[(0, 0)];
    let mut j_plane = SMatrix::<f64, 1, 6>::zeros();
    j_plane
        .fixed_view_mut::<1, 3>(0, 0)
        .copy_from(&(world - centroid).transpose());
    j_plane
        .fixed_view_mut::<1, 3>(0, 3)
        .copy_from(&(-n.transpose()));
    let plane_term = (j_plane * plane_covariance * j_plane.transpose())[(0, 0)];
    let variance = point_term + plane_term;
    (variance.is_finite() && variance > 0.0).then(|| variance.sqrt())
}

/// Point-to-plane, scan-to-map: a LiDAR point against a fixed map plane.
///
/// The FAST-LIO2 / FAST-LIVO2 LiDAR residual, on `[X ∈ SE₂(3), T_SL ∈ SE3]`:
///
/// ```text
/// r = (nᵀ·w + d) / σ          w = R·(R_SL·l + t_SL) + p
/// ```
///
/// The plane comes from the map (data association upstream) and is held
/// constant within a solve. `σ` is either fixed ([`Self::new`], FAST-LIO2) or
/// derived once from the plane's and the point's uncertainty at the predicted
/// pose ([`Self::with_uncertainty`], VoxelMap / FAST-LIVO2) and then frozen, so
/// residual and Jacobian stay consistent.
///
/// Whitens internally: register with `NoiseModel::null()`.
#[derive(Clone, Debug)]
pub struct LidarPointToPlaneSe23Factor {
    /// The scan point in the LiDAR frame, already deskewed to the state's time.
    point_l: Vector3<f64>,
    /// The map plane `nᵀ·x + d = 0`, unit normal, in the world.
    plane: Plane,
    /// Residual standard deviation [m].
    sigma: f64,
}

impl LidarPointToPlaneSe23Factor {
    /// A point against a plane with a fixed residual standard deviation.
    ///
    /// # Errors
    ///
    /// A non-finite point, or a `sigma` that is not finite and positive.
    pub fn new(point_l: Vector3<f64>, plane: Plane, sigma: f64) -> Result<Self, String> {
        if !point_l.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "LidarPointToPlaneSe23Factor: point {point_l:?} is not finite"
            ));
        }
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(format!(
                "LidarPointToPlaneSe23Factor: sigma must be finite and positive, got {sigma}"
            ));
        }
        Ok(Self {
            point_l,
            plane,
            sigma,
        })
    }

    /// A point against a plane, weighted by both uncertainties at the
    /// predicted state — see [`point_to_plane_sigma`].
    ///
    /// # Errors
    ///
    /// As [`Self::new`], plus covariances giving no finite positive variance.
    pub fn with_uncertainty(
        point_l: Vector3<f64>,
        plane: Plane,
        centroid: &Vector3<f64>,
        plane_covariance: &SMatrix<f64, 6, 6>,
        point_covariance: &Matrix3<f64>,
        predicted_state: &SE23,
        extrinsic: &SE3,
    ) -> Result<Self, String> {
        let chain = lidar_point_chain(predicted_state, extrinsic, &point_l);
        let rotation_wl =
            predicted_state.rotation_matrix() * extrinsic.rotation_so3().rotation_matrix();
        let sigma = point_to_plane_sigma(
            &plane,
            centroid,
            plane_covariance,
            point_covariance,
            &chain.world,
            &rotation_wl,
        )
        .ok_or("LidarPointToPlaneSe23Factor: covariances give no positive variance")?;
        Self::new(point_l, plane, sigma)
    }

    /// The residual standard deviation the factor whitens with [m].
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

/// Columns of the `[X, T_SL]` Jacobian.
const COLUMNS: usize = SE23::DOF + SE3::DOF;

impl Factor for LidarPointToPlaneSe23Factor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let state = SE23::from_param_slice(params[0]);
        let extrinsic = SE3::from_param_slice(params[1]);
        let chain = lidar_point_chain(&state, &extrinsic, &self.point_l);
        let information = 1.0 / self.sigma;
        let normal = self.plane.normal;

        residual[0] = information * (normal.dot(&chain.world) + self.plane.offset);

        let Some(mut jac) = jacobian else {
            return;
        };
        let d_state = information * normal.transpose() * chain.d_state;
        let d_extrinsic = information * normal.transpose() * chain.d_extrinsic;
        for col in 0..SE23::DOF {
            *jac.rb_mut().get_mut(0, col) = d_state[(0, col)];
        }
        for col in 0..SE3::DOF {
            *jac.rb_mut().get_mut(0, SE23::DOF + col) = d_extrinsic[(0, col)];
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, COLUMNS)
    }

    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE23::REP_SIZE, SE3::REP_SIZE],
            "LidarPointToPlaneSe23Factor expects [SE23 state, SE3 T_SL]",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::SE3Tangent;
    use apex_manifolds::se23::SE23Tangent;
    use faer::Mat;
    use nalgebra::UnitQuaternion;

    fn state() -> SE23 {
        SE23::new(
            Vector3::new(1.0, -2.0, 0.5),
            Vector3::new(0.3, 0.1, -0.2),
            UnitQuaternion::from_euler_angles(0.1, -0.3, 0.7),
        )
    }

    fn extrinsic() -> SE3 {
        SE3::new(
            Vector3::new(-0.007, -0.007, 0.05),
            UnitQuaternion::from_euler_angles(3.1, 0.02, -1.56),
        )
    }

    fn point() -> Vector3<f64> {
        Vector3::new(4.0, -1.5, 0.8)
    }

    /// A plane through the true world point, so the residual is zero there.
    fn plane_through(world: &Vector3<f64>) -> Result<Plane, String> {
        let normal = Vector3::new(0.3, -0.5, 0.8).normalize();
        Plane::new(normal, -normal.dot(world))
    }

    fn evaluate(
        factor: &LidarPointToPlaneSe23Factor,
        state: &SE23,
        extrinsic: &SE3,
    ) -> (f64, Mat<f64>) {
        let s = state.as_param_slice().to_vec();
        let e = extrinsic.as_param_slice().to_vec();
        let mut residual = [0.0];
        let mut jacobian = Mat::<f64>::zeros(1, COLUMNS);
        factor.linearize(&[&s, &e], &mut residual, Some(jacobian.as_mut()));
        (residual[0], jacobian)
    }

    #[test]
    fn zero_residual_on_the_plane() -> Result<(), String> {
        let world = lidar_point_chain(&state(), &extrinsic(), &point()).world;
        let factor = LidarPointToPlaneSe23Factor::new(point(), plane_through(&world)?, 0.02)?;
        let (residual, _) = evaluate(&factor, &state(), &extrinsic());
        assert!(residual.abs() < 1e-12, "{residual}");
        Ok(())
    }

    #[test]
    fn the_residual_is_the_signed_distance_over_sigma() -> Result<(), String> {
        let world = lidar_point_chain(&state(), &extrinsic(), &point()).world;
        let plane = plane_through(&world)?;
        let shifted = Plane::new(plane.normal, plane.offset + 0.1)?;
        let factor = LidarPointToPlaneSe23Factor::new(point(), shifted, 0.02)?;
        let (residual, _) = evaluate(&factor, &state(), &extrinsic());
        assert!((residual - 0.1 / 0.02).abs() < 1e-9, "{residual}");
        Ok(())
    }

    #[test]
    fn jacobian_matches_finite_differences() -> Result<(), String> {
        const EPS: f64 = 1e-6;
        let world = lidar_point_chain(&state(), &extrinsic(), &point()).world;
        // Off the plane, so the check is not at a stationary point only.
        let plane = Plane::new(Vector3::new(0.3, -0.5, 0.8), 0.4 - 0.5 * world.norm())?;
        let factor = LidarPointToPlaneSe23Factor::new(point(), plane, 0.05)?;
        let (_, analytic) = evaluate(&factor, &state(), &extrinsic());

        for axis in 0..SE23::DOF {
            let mut tangent = [0.0; 9];
            tangent[axis] = EPS;
            let plus = state().right_plus(&SE23Tangent::from_slice(&tangent), None, None);
            tangent[axis] = -EPS;
            let minus = state().right_plus(&SE23Tangent::from_slice(&tangent), None, None);
            let fd = (evaluate(&factor, &plus, &extrinsic()).0
                - evaluate(&factor, &minus, &extrinsic()).0)
                / (2.0 * EPS);
            let a = analytic[(0, axis)];
            assert!((fd - a).abs() < 1e-5, "state column {axis}: fd {fd} vs {a}");
        }
        for axis in 0..SE3::DOF {
            let mut tangent = [0.0; 6];
            tangent[axis] = EPS;
            let plus = extrinsic().right_plus(&SE3Tangent::from_slice(&tangent), None, None);
            tangent[axis] = -EPS;
            let minus = extrinsic().right_plus(&SE3Tangent::from_slice(&tangent), None, None);
            let fd = (evaluate(&factor, &state(), &plus).0 - evaluate(&factor, &state(), &minus).0)
                / (2.0 * EPS);
            let a = analytic[(0, SE23::DOF + axis)];
            assert!(
                (fd - a).abs() < 1e-5,
                "extrinsic column {axis}: fd {fd} vs {a}"
            );
        }
        Ok(())
    }

    #[test]
    fn velocity_columns_are_zero() -> Result<(), String> {
        let world = lidar_point_chain(&state(), &extrinsic(), &point()).world;
        let factor = LidarPointToPlaneSe23Factor::new(point(), plane_through(&world)?, 0.02)?;
        let (_, jacobian) = evaluate(&factor, &state(), &extrinsic());
        for col in 6..9 {
            assert_eq!(jacobian[(0, col)], 0.0);
        }
        Ok(())
    }

    /// Moving the world, the state and the map together changes nothing.
    #[test]
    fn a_common_change_of_world_leaves_the_residual() -> Result<(), String> {
        let world = lidar_point_chain(&state(), &extrinsic(), &point()).world;
        let plane = Plane::new(Vector3::new(0.3, -0.5, 0.8), 0.25 - world.norm())?;
        let factor = LidarPointToPlaneSe23Factor::new(point(), plane.clone(), 0.02)?;
        let (before, _) = evaluate(&factor, &state(), &extrinsic());

        let g_rot = UnitQuaternion::from_euler_angles(0.4, 0.2, -1.1);
        let g_t = Vector3::new(3.0, 1.0, -2.0);
        let moved_state = SE23::new(
            g_rot * state().translation() + g_t,
            g_rot * state().velocity(),
            g_rot * state().rotation_quaternion(),
        );
        // nᵀx + d = 0 in the old world ⇒ (G n)ᵀ x' + (d − (G n)ᵀ t) = 0.
        let moved_normal = g_rot * plane.normal;
        let moved_plane = Plane::new(moved_normal, plane.offset - moved_normal.dot(&g_t))?;
        let moved = LidarPointToPlaneSe23Factor::new(point(), moved_plane, 0.02)?;
        let (after, _) = evaluate(&moved, &moved_state, &extrinsic());
        assert!((before - after).abs() < 1e-9, "{before} vs {after}");
        Ok(())
    }

    #[test]
    fn uncertainty_weighting_grows_with_the_point_noise_and_the_distance() -> Result<(), String> {
        let world = lidar_point_chain(&state(), &extrinsic(), &point()).world;
        let plane = plane_through(&world)?;
        let plane_cov = SMatrix::<f64, 6, 6>::identity() * 1e-6;
        let near = world + Vector3::new(0.1, 0.0, 0.0);
        let r_wl = state().rotation_matrix() * extrinsic().rotation_so3().rotation_matrix();
        let quiet = point_to_plane_sigma(
            &plane,
            &near,
            &plane_cov,
            &(Matrix3::identity() * 1e-6),
            &world,
            &r_wl,
        )
        .ok_or("no sigma")?;
        let noisy = point_to_plane_sigma(
            &plane,
            &near,
            &plane_cov,
            &(Matrix3::identity() * 1e-3),
            &world,
            &r_wl,
        )
        .ok_or("no sigma")?;
        let far = point_to_plane_sigma(
            &plane,
            &(world + Vector3::new(5.0, 0.0, 0.0)),
            &plane_cov,
            &(Matrix3::identity() * 1e-6),
            &world,
            &r_wl,
        )
        .ok_or("no sigma")?;
        assert!(noisy > quiet, "{noisy} vs {quiet}");
        // A point far from the plane's centroid inherits more normal error.
        assert!(far > quiet, "{far} vs {quiet}");

        let factor = LidarPointToPlaneSe23Factor::with_uncertainty(
            point(),
            plane,
            &near,
            &plane_cov,
            &(Matrix3::identity() * 1e-4),
            &state(),
            &extrinsic(),
        )?;
        assert!(factor.sigma() > 0.0);
        Ok(())
    }

    #[test]
    fn bad_inputs_are_rejected() -> Result<(), String> {
        let plane = Plane::new(Vector3::z(), 0.0)?;
        assert!(LidarPointToPlaneSe23Factor::new(point(), plane.clone(), 0.0).is_err());
        assert!(LidarPointToPlaneSe23Factor::new(point(), plane.clone(), f64::NAN).is_err());
        assert!(
            LidarPointToPlaneSe23Factor::new(Vector3::new(f64::NAN, 0.0, 0.0), plane, 0.02)
                .is_err()
        );
        let zero = SMatrix::<f64, 6, 6>::zeros();
        assert!(
            point_to_plane_sigma(
                &Plane::new(Vector3::z(), 0.0)?,
                &Vector3::zeros(),
                &zero,
                &Matrix3::zeros(),
                &Vector3::zeros(),
                &Matrix3::identity()
            )
            .is_none()
        );
        Ok(())
    }

    #[test]
    fn the_factor_whitens_internally_and_has_fifteen_columns() -> Result<(), String> {
        let factor =
            LidarPointToPlaneSe23Factor::new(point(), Plane::new(Vector3::z(), 0.0)?, 0.02)?;
        assert!(factor.whitens_internally());
        assert_eq!(factor.jacobian_shape(), (1, 15));
        Ok(())
    }
}
