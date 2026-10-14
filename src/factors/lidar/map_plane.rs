//! LiDAR points against a map, on any pose group that acts on 3-D points.
//!
//! The other LiDAR factors in this module act on bare `SE3` poses. Here the
//! world pose `X` is generic: any [`LieGroup`] whose action moves a 3-D point
//! — `SE3`, `SE₂(3)` (pose + velocity), `Sim3`, `SO3` — while the
//! LiDAR extrinsic `T_SL` stays a rigid `SE3`:
//!
//! ```text
//! s = R_SL·l + t_SL          (LiDAR → body)
//! w = X ⊙ s                  (body → world, the group action)
//! ```
//!
//! # Jacobians
//!
//! Everything group-specific comes from [`LieGroup::act`], which returns
//! `∂(X⊙s)/∂X` in the group's own right-perturbation chart and `∂(X⊙s)/∂s`:
//!
//! ```text
//! ∂w/∂X   = act's jacobian_self (top 3 rows)
//! ∂w/∂δρₑ = (∂w/∂s)·R_SL      ∂w/∂δθₑ = −(∂w/∂s)·R_SL·[l]×
//! ```
//!
//! So `SE₂(3)` gets exact zero velocity columns and `Sim3` a scale column (the
//! LiDAR then constrains scale), none of it written by hand here. Groups
//! without a 3-D point action — see
//! [`acts_on_points`](crate::factors::common::group_point::acts_on_points) —
//! are rejected at construction.

use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, Vector3};
use std::marker::PhantomData;

use apex_manifolds::se3::SE3;
use apex_manifolds::{LieGroup, Tangent};

use crate::core::variable::ManifoldVariable;
use crate::core::{CoreError, CoreResult};
use crate::factors::Factor;
use crate::factors::common::group_point::acts_on_points;
use crate::factors::common::math::skew;
use crate::factors::common::validate::expect_manifolds;

use super::plane::Plane;

/// A LiDAR point carried to the world, with its derivatives.
#[derive(Clone, Debug)]
pub struct LidarPointChain {
    /// The point in the body frame, `s = R_SL·l + t_SL`.
    pub body: Vector3<f64>,
    /// The point in the world, `w = X ⊙ s`.
    pub world: Vector3<f64>,
    /// `∂w/∂X`, `3 × dof(G)`, row-major by point coordinate.
    pub d_state: Vec<[f64; 3]>,
    /// `∂w/∂T_SL`, columns `[δρₑ, δθₑ]` of the `SE3` tangent.
    pub d_extrinsic: SMatrix<f64, 3, 6>,
}

impl LidarPointChain {
    /// `rowᵀ·∂w/∂X` — one row of a scalar residual's state Jacobian.
    pub fn project_state(&self, row: &Vector3<f64>) -> Vec<f64> {
        self.d_state
            .iter()
            .map(|column| row.x * column[0] + row.y * column[1] + row.z * column[2])
            .collect()
    }
}

/// Carry LiDAR point `point_l` through `T_SL` and the pose `X` to the world.
pub fn lidar_point_chain<G: LieGroup>(
    state: &G,
    extrinsic: &SE3,
    point_l: &Vector3<f64>,
) -> LidarPointChain {
    let r_sl: Matrix3<f64> = extrinsic.rotation_so3().rotation_matrix();
    let body = r_sl * point_l + extrinsic.translation();

    let mut jacobian_state = G::zero_jacobian();
    let mut d_world_d_body = Matrix3::zeros();
    let world = state.act(&body, Some(&mut jacobian_state), Some(&mut d_world_d_body));
    let d_state = (0..state.tangent_dim())
        .map(|col| {
            [
                jacobian_state[(0, col)],
                jacobian_state[(1, col)],
                jacobian_state[(2, col)],
            ]
        })
        .collect();

    let d_body = d_world_d_body * r_sl;
    let mut d_extrinsic = SMatrix::<f64, 3, 6>::zeros();
    d_extrinsic.fixed_view_mut::<3, 3>(0, 0).copy_from(&d_body);
    d_extrinsic
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-d_body * skew(point_l)));

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
/// σ² = nᵀ·A·Σ_l·Aᵀ·n  +  J_π·Σ_π·J_πᵀ,     J_π = [(w − q)ᵀ, −nᵀ]
/// ```
///
/// `plane_covariance` is the `6×6` covariance of `(n, q)` — normal, then the
/// plane's centroid `q` — from the map points that fit it; `point_covariance`
/// the scan point's `3×3` covariance in the LiDAR frame (range and bearing
/// noise); `world` the point carried to the world at the predicted pose and
/// `d_world_d_point` the `3×3` map from a LiDAR-frame displacement to a world
/// one (`R·R_SL`, or `s·R·R_SL` for `Sim3`).
///
/// Returns `None` when the variance is not finite and positive.
pub fn point_to_plane_sigma(
    plane: &Plane,
    centroid: &Vector3<f64>,
    plane_covariance: &SMatrix<f64, 6, 6>,
    point_covariance: &Matrix3<f64>,
    world: &Vector3<f64>,
    d_world_d_point: &Matrix3<f64>,
) -> Option<f64> {
    let n = plane.normal;
    let a = d_world_d_point;
    let point_term = (n.transpose() * a * point_covariance * a.transpose() * n)[(0, 0)];
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
/// The FAST-LIO2 / FAST-LIVO2 LiDAR residual, on `[X ∈ G, T_SL ∈ SE3]`:
///
/// ```text
/// r = (nᵀ·w + d) / σ          w = X ⊙ (R_SL·l + t_SL)
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
pub struct LidarMapPlaneFactor<G: LieGroup> {
    /// The scan point in the LiDAR frame, already deskewed to the pose's time.
    point_l: Vector3<f64>,
    /// The map plane `nᵀ·x + d = 0`, unit normal, in the world.
    plane: Plane,
    /// Residual standard deviation [m].
    sigma: f64,
    /// The pose group.
    group: PhantomData<fn() -> G>,
}

impl<G: LieGroup> LidarMapPlaneFactor<G> {
    /// A point against a plane with a fixed residual standard deviation.
    ///
    /// # Errors
    ///
    /// A group that does not act on 3-D points, a non-finite point, or a
    /// `sigma` that is not finite and positive.
    pub fn new(point_l: Vector3<f64>, plane: Plane, sigma: f64) -> CoreResult<Self> {
        if !acts_on_points::<G>() {
            return Err(CoreError::InvalidInput(format!(
                "LidarMapPlaneFactor: {} does not act on 3-D points",
                G::NAME
            )));
        }
        if !point_l.iter().all(|value| value.is_finite()) {
            return Err(CoreError::InvalidInput(format!(
                "LidarMapPlaneFactor: point {point_l:?} is not finite"
            )));
        }
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(CoreError::InvalidInput(format!(
                "LidarMapPlaneFactor: sigma must be finite and positive, got {sigma}"
            )));
        }
        Ok(Self {
            point_l,
            plane,
            sigma,
            group: PhantomData,
        })
    }

    /// A point against a plane, weighted by both uncertainties at the
    /// predicted pose — see [`point_to_plane_sigma`].
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
        predicted_state: &G,
        extrinsic: &SE3,
    ) -> CoreResult<Self> {
        let chain = lidar_point_chain(predicted_state, extrinsic, &point_l);
        // ∂w/∂l = ∂w/∂s·R_SL = the extrinsic Jacobian's translation block.
        let d_world_d_point: Matrix3<f64> = chain.d_extrinsic.fixed_view::<3, 3>(0, 0).into();
        let sigma = point_to_plane_sigma(
            &plane,
            centroid,
            plane_covariance,
            point_covariance,
            &chain.world,
            &d_world_d_point,
        )
        .ok_or_else(|| {
            CoreError::InvalidInput(
                "LidarMapPlaneFactor: covariances give no positive variance".to_string(),
            )
        })?;
        Self::new(point_l, plane, sigma)
    }

    /// The residual standard deviation the factor whitens with [m].
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl<G: LieGroup + Send + Sync + 'static> Factor for LidarMapPlaneFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let state = G::from_param_slice(params[0]);
        let extrinsic = SE3::from_param_slice(params[1]);
        let chain = lidar_point_chain(&state, &extrinsic, &self.point_l);
        let information = 1.0 / self.sigma;
        let normal = self.plane.normal;

        residual[0] = information * (normal.dot(&chain.world) + self.plane.offset);

        let Some(mut jac) = jacobian else {
            return;
        };
        let d_state = chain.project_state(&normal);
        for (col, value) in d_state.iter().enumerate() {
            *jac.rb_mut().get_mut(0, col) = information * value;
        }
        let d_extrinsic = information * normal.transpose() * chain.d_extrinsic;
        let offset = d_state.len();
        for col in 0..6 {
            *jac.rb_mut().get_mut(0, offset + col) = d_extrinsic[(0, col)];
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, G::TangentVector::DIM + 6)
    }

    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_manifolds(
            variables,
            &[G::NAME, "SE3"],
            "LidarMapPlaneFactor expects [pose, SE3 T_SL]",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::rn::Rn;
    use apex_manifolds::se2::SE2;
    use apex_manifolds::se3::SE3Tangent;
    use apex_manifolds::se23::SE23;
    use apex_manifolds::sim3::Sim3;
    use apex_manifolds::so3::SO3;
    use faer::Mat;
    use nalgebra::UnitQuaternion;

    fn rotation() -> UnitQuaternion<f64> {
        UnitQuaternion::from_euler_angles(0.1, -0.3, 0.7)
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

    fn normal() -> Vector3<f64> {
        Vector3::new(0.3, -0.5, 0.8).normalize()
    }

    /// A plane `offset` metres from the true world point of `point()`.
    fn plane_near<G: LieGroup>(state: &G, offset: f64) -> Result<Plane, CoreError> {
        let world = lidar_point_chain(state, &extrinsic(), &point()).world;
        Plane::new(normal(), offset - normal().dot(&world))
    }

    fn evaluate<G: LieGroup + Send + Sync + 'static>(
        factor: &LidarMapPlaneFactor<G>,
        state: &G,
        extrinsic: &SE3,
    ) -> (f64, Mat<f64>) {
        let s = state.as_param_slice().to_vec();
        let e = extrinsic.as_param_slice().to_vec();
        let mut residual = [0.0];
        let (rows, cols) = factor.jacobian_shape();
        let mut jacobian = Mat::<f64>::zeros(rows, cols);
        factor.linearize(&[&s, &e], &mut residual, Some(jacobian.as_mut()));
        (residual[0], jacobian)
    }

    /// Zero on the plane, the signed distance over σ off it, and analytic
    /// Jacobians equal to central differences in the group's own chart — for
    /// any group `G`.
    fn check_group<G: LieGroup + Send + Sync + 'static>(state: G) -> Result<(), CoreError> {
        const EPS: f64 = 1e-6;
        let dof = G::TangentVector::DIM;

        let on = LidarMapPlaneFactor::<G>::new(point(), plane_near(&state, 0.0)?, 0.02)?;
        let (residual, _) = evaluate(&on, &state, &extrinsic());
        assert!(
            residual.abs() < 1e-12,
            "{}: on-plane residual {residual}",
            G::NAME
        );

        let off = LidarMapPlaneFactor::<G>::new(point(), plane_near(&state, 0.1)?, 0.02)?;
        let (residual, analytic) = evaluate(&off, &state, &extrinsic());
        assert!(
            (residual - 5.0).abs() < 1e-9,
            "{}: off-plane residual {residual}",
            G::NAME
        );
        assert_eq!(analytic.ncols(), dof + 6);

        for axis in 0..dof {
            let mut tangent = vec![0.0; dof];
            tangent[axis] = EPS;
            let plus = state.right_plus(&G::TangentVector::from_slice(&tangent), None, None);
            tangent[axis] = -EPS;
            let minus = state.right_plus(&G::TangentVector::from_slice(&tangent), None, None);
            let fd = (evaluate(&off, &plus, &extrinsic()).0
                - evaluate(&off, &minus, &extrinsic()).0)
                / (2.0 * EPS);
            let a = analytic[(0, axis)];
            assert!(
                (fd - a).abs() < 1e-5,
                "{} pose column {axis}: fd {fd} vs {a}",
                G::NAME
            );
        }
        for axis in 0..6 {
            let mut tangent = [0.0; 6];
            tangent[axis] = EPS;
            let plus = extrinsic().right_plus(&SE3Tangent::from_slice(&tangent), None, None);
            tangent[axis] = -EPS;
            let minus = extrinsic().right_plus(&SE3Tangent::from_slice(&tangent), None, None);
            let fd =
                (evaluate(&off, &state, &plus).0 - evaluate(&off, &state, &minus).0) / (2.0 * EPS);
            let a = analytic[(0, dof + axis)];
            assert!(
                (fd - a).abs() < 1e-5,
                "{} extrinsic column {axis}: fd {fd} vs {a}",
                G::NAME
            );
        }
        Ok(())
    }

    #[test]
    fn se3_poses() -> Result<(), CoreError> {
        check_group(SE3::new(Vector3::new(1.0, -2.0, 0.5), rotation()))
    }

    #[test]
    fn se23_navigation_states() -> Result<(), CoreError> {
        check_group(SE23::new(
            Vector3::new(1.0, -2.0, 0.5),
            Vector3::new(0.3, 0.1, -0.2),
            rotation(),
        ))
    }

    #[test]
    fn sim3_similarities() -> Result<(), CoreError> {
        check_group(Sim3::new(Vector3::new(1.0, -2.0, 0.5), rotation(), 1.3))
    }

    #[test]
    fn so3_rotations() -> Result<(), CoreError> {
        check_group(SO3::new(rotation()))
    }

    /// A navigation state's velocity never moves a LiDAR point.
    #[test]
    fn se23_velocity_columns_are_zero() -> Result<(), CoreError> {
        let state = SE23::new(
            Vector3::new(1.0, -2.0, 0.5),
            Vector3::new(0.3, 0.1, -0.2),
            rotation(),
        );
        let factor = LidarMapPlaneFactor::<SE23>::new(point(), plane_near(&state, 0.1)?, 0.02)?;
        let (_, jacobian) = evaluate(&factor, &state, &extrinsic());
        for col in 6..9 {
            assert_eq!(jacobian[(0, col)], 0.0);
        }
        Ok(())
    }

    /// Moving the world, the pose and the map together changes nothing.
    #[test]
    fn a_common_change_of_world_leaves_the_residual() -> Result<(), CoreError> {
        let state = SE23::new(
            Vector3::new(1.0, -2.0, 0.5),
            Vector3::new(0.3, 0.1, -0.2),
            rotation(),
        );
        let plane = plane_near(&state, 0.25)?;
        let factor = LidarMapPlaneFactor::<SE23>::new(point(), plane.clone(), 0.02)?;
        let (before, _) = evaluate(&factor, &state, &extrinsic());

        let g_rot = UnitQuaternion::from_euler_angles(0.4, 0.2, -1.1);
        let g_t = Vector3::new(3.0, 1.0, -2.0);
        let moved_state = SE23::new(
            g_rot * state.translation() + g_t,
            g_rot * state.velocity(),
            g_rot * state.rotation_quaternion(),
        );
        // nᵀx + d = 0 in the old world ⇒ (G n)ᵀ x' + (d − (G n)ᵀ t) = 0.
        let moved_normal = g_rot * plane.normal;
        let moved_plane = Plane::new(moved_normal, plane.offset - moved_normal.dot(&g_t))?;
        let moved = LidarMapPlaneFactor::<SE23>::new(point(), moved_plane, 0.02)?;
        let (after, _) = evaluate(&moved, &moved_state, &extrinsic());
        assert!((before - after).abs() < 1e-9, "{before} vs {after}");
        Ok(())
    }

    #[test]
    fn uncertainty_weighting_grows_with_the_point_noise_and_the_distance() -> Result<(), CoreError>
    {
        let state = SE23::new(Vector3::new(1.0, -2.0, 0.5), Vector3::zeros(), rotation());
        let chain = lidar_point_chain(&state, &extrinsic(), &point());
        let a: Matrix3<f64> = chain.d_extrinsic.fixed_view::<3, 3>(0, 0).into();
        let plane = plane_near(&state, 0.0)?;
        let plane_cov = SMatrix::<f64, 6, 6>::identity() * 1e-6;
        let near = chain.world + Vector3::new(0.1, 0.0, 0.0);
        let far = chain.world + Vector3::new(5.0, 0.0, 0.0);
        let sigma = |centroid: &Vector3<f64>, noise: f64| {
            point_to_plane_sigma(
                &plane,
                centroid,
                &plane_cov,
                &(Matrix3::identity() * noise),
                &chain.world,
                &a,
            )
            .ok_or_else(|| CoreError::InvalidInput("no sigma".to_string()))
        };
        let quiet = sigma(&near, 1e-6)?;
        assert!(sigma(&near, 1e-3)? > quiet);
        // A point far from the plane's centroid inherits more normal error.
        assert!(sigma(&far, 1e-6)? > quiet);

        let factor = LidarMapPlaneFactor::<SE23>::with_uncertainty(
            point(),
            plane.clone(),
            &near,
            &plane_cov,
            &(Matrix3::identity() * 1e-4),
            &state,
            &extrinsic(),
        )?;
        assert!(factor.sigma() > 0.0);
        Ok(())
    }

    #[test]
    fn groups_that_do_not_act_on_3d_points_are_rejected() -> Result<(), CoreError> {
        let plane = Plane::new(Vector3::z(), 0.0)?;
        assert!(LidarMapPlaneFactor::<SE2>::new(point(), plane.clone(), 0.02).is_err());
        assert!(LidarMapPlaneFactor::<Rn>::new(point(), plane, 0.02).is_err());
        assert!(acts_on_points::<SE3>() && acts_on_points::<Sim3>() && !acts_on_points::<SE2>());
        Ok(())
    }

    #[test]
    fn bad_inputs_are_rejected() -> Result<(), CoreError> {
        let plane = Plane::new(Vector3::z(), 0.0)?;
        assert!(LidarMapPlaneFactor::<SE3>::new(point(), plane.clone(), 0.0).is_err());
        assert!(LidarMapPlaneFactor::<SE3>::new(point(), plane.clone(), f64::NAN).is_err());
        assert!(
            LidarMapPlaneFactor::<SE3>::new(Vector3::new(f64::NAN, 0.0, 0.0), plane, 0.02).is_err()
        );
        let zero = SMatrix::<f64, 6, 6>::zeros();
        let none = point_to_plane_sigma(
            &Plane::new(Vector3::z(), 0.0)?,
            &Vector3::zeros(),
            &zero,
            &Matrix3::zeros(),
            &Vector3::zeros(),
            &Matrix3::identity(),
        );
        assert!(none.is_none());
        Ok(())
    }

    #[test]
    fn the_factor_whitens_internally() -> Result<(), CoreError> {
        let factor =
            LidarMapPlaneFactor::<SE23>::new(point(), Plane::new(Vector3::z(), 0.0)?, 0.02)?;
        assert!(factor.whitens_internally());
        assert_eq!(factor.jacobian_shape(), (1, 15));
        Ok(())
    }
}
