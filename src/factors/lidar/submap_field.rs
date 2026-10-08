//! LiDAR points against an occupancy field anchored to a keyframe — the
//! OKVIS2-X LiDAR factors, generic over the pose group.
//!
//! A submap is an occupancy (or distance) field `L` in the body frame of its
//! anchor state `X_a`. A world point reaches it through the anchor's inverse:
//!
//! ```text
//! p_A = X_a⁻¹ ⊙ w
//! e   = L(p_A) / σ,        σ = √(L_min²/9 + σ_d²·‖∇L‖²)     (OKVIS2-X)
//! ```
//!
//! [`LidarSubmapFieldFactor`] is frame-to-submap: `w` is a scan point carried
//! through `T_SL` and the scan's state `X_k`. [`SubmapAlignmentFactor`] is
//! submap-to-submap: `w = X_b ⊙ q_B` for a point `q_B` of submap B.
//!
//! `σ` is evaluated once, at the predicted poses, and then frozen, so residual
//! and Jacobian stay consistent. A point outside the field's domain (or on a
//! flat stretch of it) contributes nothing: zero residual, zero Jacobian.
//!
//! The anchor's Jacobian is `∂(Y⊙w)/∂Y · ∂(X_a⁻¹)/∂X_a` with `Y = X_a⁻¹`, both
//! from [`LieGroup`], so every group with a point action gets it for free.

use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, Vector3};
use std::marker::PhantomData;
use std::sync::Arc;

use apex_manifolds::se3::SE3;
use apex_manifolds::{LieGroup, Tangent};

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::validate::expect_manifolds;

use super::distance_field::DistanceField;
use super::map_plane::{acts_on_points, lidar_point_chain};

/// Gradients below this norm carry no direction: the point is ignored.
const MIN_GRADIENT: f64 = 1e-9;

/// A world point carried into an anchor's frame, with its derivatives.
#[derive(Clone, Debug)]
pub struct AnchorChain {
    /// The point in the anchor frame, `p_A = X_a⁻¹ ⊙ w`.
    pub point: Vector3<f64>,
    /// `∂p_A/∂X_a`, `3 × dof(G)`, one `[x, y, z]` column per tangent axis.
    pub d_anchor: Vec<[f64; 3]>,
    /// `∂p_A/∂w`.
    pub d_world: Matrix3<f64>,
}

/// Carry world point `world` into the frame of `anchor`.
pub fn into_anchor<G: LieGroup>(anchor: &G, world: &Vector3<f64>) -> AnchorChain {
    let mut d_inverse = G::zero_jacobian();
    let inverse = anchor.inverse(Some(&mut d_inverse));
    let mut d_act = G::zero_jacobian();
    let mut d_world = Matrix3::zeros();
    let point = inverse.act(world, Some(&mut d_act), Some(&mut d_world));
    let dof = anchor.tangent_dim();
    let d_anchor = (0..dof)
        .map(|col| {
            let mut column = [0.0; 3];
            for (row, value) in column.iter_mut().enumerate() {
                *value = (0..dof)
                    .map(|k| d_act[(row, k)] * d_inverse[(k, col)])
                    .sum();
            }
            column
        })
        .collect();
    AnchorChain {
        point,
        d_anchor,
        d_world,
    }
}

/// The OKVIS2-X residual scale `√(L_min²/9 + σ_d²·‖∇L‖²)`: the field's own
/// uncertainty (a third of the occupancy clamp `L_min`) plus the point's range
/// noise `σ_d` mapped through the field's slope.
///
/// Returns `None` unless the result is finite and positive.
pub fn occupancy_sigma(gradient: &Vector3<f64>, log_odds_min: f64, sigma_d: f64) -> Option<f64> {
    let variance = log_odds_min * log_odds_min / 9.0 + sigma_d * sigma_d * gradient.norm_squared();
    (variance.is_finite() && variance > 0.0).then(|| variance.sqrt())
}

fn check_inputs<G: LieGroup>(name: &str, point: &Vector3<f64>, sigma: f64) -> Result<(), String> {
    if !acts_on_points::<G>() {
        return Err(format!("{name}: {} does not act on 3-D points", G::NAME));
    }
    if !point.iter().all(|value| value.is_finite()) {
        return Err(format!("{name}: point {point:?} is not finite"));
    }
    if !(sigma.is_finite() && sigma > 0.0) {
        return Err(format!(
            "{name}: sigma must be finite and positive, got {sigma}"
        ));
    }
    Ok(())
}

/// Field value and gradient at `point`, if the point is in the domain and the
/// field has a direction there.
fn query<F: DistanceField>(field: &F, point: &Vector3<f64>) -> Option<(f64, Vector3<f64>)> {
    field
        .query(point)
        .filter(|(value, gradient)| value.is_finite() && gradient.norm() > MIN_GRADIENT)
}

/// The point is outside the field: zero residual, zero Jacobian row.
fn zero_out(residual: &mut [f64], jacobian: Option<faer::mat::MatMut<'_, f64>>) {
    residual[0] = 0.0;
    if let Some(mut jac) = jacobian {
        for col in 0..jac.ncols() {
            *jac.rb_mut().get_mut(0, col) = 0.0;
        }
    }
}

/// `rowᵀ·columns` for a `3 × n` column list.
fn project(row: &Vector3<f64>, columns: &[[f64; 3]]) -> impl Iterator<Item = f64> {
    columns
        .iter()
        .map(move |c| row.x * c[0] + row.y * c[1] + row.z * c[2])
}

/// Frame-to-submap: a scan point against the occupancy field of a submap.
///
/// Blocks `[X_a ∈ G, X_k ∈ G, T_SL ∈ SE3]`: the submap's anchor, the scan's
/// state, the LiDAR extrinsic. `X_a` and `X_k` must be different variables —
/// a scan inside its own submap constrains nothing.
///
/// Whitens internally: register with `NoiseModel::null()`.
pub struct LidarSubmapFieldFactor<G: LieGroup, F: DistanceField> {
    field: Arc<F>,
    /// The scan point in the LiDAR frame, already deskewed to `X_k`'s time.
    point_l: Vector3<f64>,
    /// Residual scale, in field units.
    sigma: f64,
    group: PhantomData<fn() -> G>,
}

impl<G: LieGroup, F: DistanceField> LidarSubmapFieldFactor<G, F> {
    /// A scan point against `field` with a fixed residual scale.
    ///
    /// # Errors
    ///
    /// A group without a 3-D point action, a non-finite point, or a `sigma`
    /// that is not finite and positive.
    pub fn new(field: Arc<F>, point_l: Vector3<f64>, sigma: f64) -> Result<Self, String> {
        check_inputs::<G>("LidarSubmapFieldFactor", &point_l, sigma)?;
        Ok(Self {
            field,
            point_l,
            sigma,
            group: PhantomData,
        })
    }

    /// A scan point weighted the OKVIS2-X way — see [`occupancy_sigma`] —
    /// at the predicted anchor, state and extrinsic.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], plus a predicted point outside the field.
    pub fn with_occupancy_weighting(
        field: Arc<F>,
        point_l: Vector3<f64>,
        log_odds_min: f64,
        sigma_d: f64,
        predicted_anchor: &G,
        predicted_state: &G,
        extrinsic: &SE3,
    ) -> Result<Self, String> {
        let world = lidar_point_chain(predicted_state, extrinsic, &point_l).world;
        let point_a = into_anchor(predicted_anchor, &world).point;
        let (_, gradient) = query(field.as_ref(), &point_a)
            .ok_or("LidarSubmapFieldFactor: predicted point is outside the field")?;
        let sigma = occupancy_sigma(&gradient, log_odds_min, sigma_d)
            .ok_or("LidarSubmapFieldFactor: occupancy weighting gives no positive scale")?;
        Self::new(field, point_l, sigma)
    }

    /// The residual scale the factor whitens with.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl<G, F> Factor for LidarSubmapFieldFactor<G, F>
where
    G: LieGroup + Send + Sync + 'static,
    F: DistanceField + 'static,
{
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let anchor = G::from_param_slice(params[0]);
        let state = G::from_param_slice(params[1]);
        let extrinsic = SE3::from_param_slice(params[2]);
        let scan = lidar_point_chain(&state, &extrinsic, &self.point_l);
        let chain = into_anchor(&anchor, &scan.world);
        let Some((value, gradient)) = query(self.field.as_ref(), &chain.point) else {
            zero_out(residual, jacobian);
            return;
        };
        residual[0] = value / self.sigma;

        let Some(mut jac) = jacobian else {
            return;
        };
        let d_point = gradient / self.sigma;
        let d_world = chain.d_world.transpose() * d_point;
        let d_extrinsic = d_world.transpose() * scan.d_extrinsic;
        let row = project(&d_point, &chain.d_anchor)
            .chain(project(&d_world, &scan.d_state))
            .chain(d_extrinsic.iter().copied());
        for (col, value) in row.enumerate() {
            *jac.rb_mut().get_mut(0, col) = value;
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, 2 * G::TangentVector::DIM + 6)
    }

    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_manifolds(
            variables,
            &[G::NAME, G::NAME, "SE3"],
            "LidarSubmapFieldFactor expects [anchor, state, SE3 T_SL]",
        )
    }
}

/// Submap-to-submap: a point of submap B against the field of submap A.
///
/// Blocks `[X_a ∈ G, X_b ∈ G]`, the two anchors; `q_B` is a point of B in its
/// anchor's body frame (e.g. an occupied voxel centre or a kept scan point).
///
/// Whitens internally: register with `NoiseModel::null()`.
pub struct SubmapAlignmentFactor<G: LieGroup, F: DistanceField> {
    field: Arc<F>,
    point_b: Vector3<f64>,
    sigma: f64,
    group: PhantomData<fn() -> G>,
}

impl<G: LieGroup, F: DistanceField> SubmapAlignmentFactor<G, F> {
    /// A point of submap B against `field` (submap A) with a fixed scale.
    ///
    /// # Errors
    ///
    /// As [`LidarSubmapFieldFactor::new`].
    pub fn new(field: Arc<F>, point_b: Vector3<f64>, sigma: f64) -> Result<Self, String> {
        check_inputs::<G>("SubmapAlignmentFactor", &point_b, sigma)?;
        Ok(Self {
            field,
            point_b,
            sigma,
            group: PhantomData,
        })
    }

    /// A point weighted the OKVIS2-X way at the predicted anchors.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], plus a predicted point outside the field.
    pub fn with_occupancy_weighting(
        field: Arc<F>,
        point_b: Vector3<f64>,
        log_odds_min: f64,
        sigma_d: f64,
        predicted_a: &G,
        predicted_b: &G,
    ) -> Result<Self, String> {
        let world = predicted_b.act(&point_b, None, None);
        let point_a = into_anchor(predicted_a, &world).point;
        let (_, gradient) = query(field.as_ref(), &point_a)
            .ok_or("SubmapAlignmentFactor: predicted point is outside the field")?;
        let sigma = occupancy_sigma(&gradient, log_odds_min, sigma_d)
            .ok_or("SubmapAlignmentFactor: occupancy weighting gives no positive scale")?;
        Self::new(field, point_b, sigma)
    }

    /// The residual scale the factor whitens with.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl<G, F> Factor for SubmapAlignmentFactor<G, F>
where
    G: LieGroup + Send + Sync + 'static,
    F: DistanceField + 'static,
{
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let anchor_a = G::from_param_slice(params[0]);
        let anchor_b = G::from_param_slice(params[1]);
        let mut d_b = G::zero_jacobian();
        let world = anchor_b.act(&self.point_b, Some(&mut d_b), None);
        let chain = into_anchor(&anchor_a, &world);
        let Some((value, gradient)) = query(self.field.as_ref(), &chain.point) else {
            zero_out(residual, jacobian);
            return;
        };
        residual[0] = value / self.sigma;

        let Some(mut jac) = jacobian else {
            return;
        };
        let d_point = gradient / self.sigma;
        let d_world = chain.d_world.transpose() * d_point;
        let d_b_row = (0..anchor_b.tangent_dim()).map(|col| {
            d_world.x * d_b[(0, col)] + d_world.y * d_b[(1, col)] + d_world.z * d_b[(2, col)]
        });
        for (col, value) in project(&d_point, &chain.d_anchor)
            .chain(d_b_row)
            .enumerate()
        {
            *jac.rb_mut().get_mut(0, col) = value;
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, 2 * G::TangentVector::DIM)
    }

    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_manifolds(
            variables,
            &[G::NAME, G::NAME],
            "SubmapAlignmentFactor expects [anchor A, anchor B]",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::se2::SE2;
    use apex_manifolds::se23::SE23;
    use apex_manifolds::sgal3::SGal3;
    use apex_manifolds::sim3::Sim3;
    use apex_manifolds::so3::SO3;
    use faer::Mat;
    use nalgebra::UnitQuaternion;

    /// `L(p) = pᵀ·A·p − 1`, an ellipsoid with distinct axes (so every pose
    /// direction is seen), defined within `radius` of the origin.
    struct Ellipsoid {
        radius: f64,
    }

    impl DistanceField for Ellipsoid {
        fn query(&self, point: &Vector3<f64>) -> Option<(f64, Vector3<f64>)> {
            let ap = Vector3::new(1.0 / 9.0, 1.0 / 4.0, 1.0).component_mul(point);
            (point.norm() < self.radius).then(|| (point.dot(&ap) - 1.0, 2.0 * ap))
        }
    }

    fn field() -> Arc<Ellipsoid> {
        Arc::new(Ellipsoid { radius: 10.0 })
    }

    fn rotation(roll: f64) -> UnitQuaternion<f64> {
        UnitQuaternion::from_euler_angles(roll, -0.3, 0.7)
    }

    fn extrinsic() -> SE3 {
        SE3::new(
            Vector3::new(-0.007, -0.007, 0.05),
            UnitQuaternion::from_euler_angles(3.1, 0.02, -1.56),
        )
    }

    fn point() -> Vector3<f64> {
        Vector3::new(1.5, -0.8, 0.6)
    }

    fn central_difference<G: LieGroup>(at: &G, axis: usize, evaluate: impl Fn(&G) -> f64) -> f64 {
        const EPS: f64 = 1e-6;
        let mut tangent = vec![0.0; G::TangentVector::DIM];
        tangent[axis] = EPS;
        let plus = at.right_plus(&G::TangentVector::from_slice(&tangent), None, None);
        tangent[axis] = -EPS;
        let minus = at.right_plus(&G::TangentVector::from_slice(&tangent), None, None);
        (evaluate(&plus) - evaluate(&minus)) / (2.0 * EPS)
    }

    fn linearize(factor: &dyn Factor, params: &[Vec<f64>]) -> (f64, Mat<f64>) {
        let slices: Vec<&[f64]> = params.iter().map(Vec::as_slice).collect();
        let (rows, cols) = factor.jacobian_shape();
        let mut jacobian = Mat::<f64>::zeros(rows, cols);
        let mut residual = [0.0];
        factor.linearize(&slices, &mut residual, Some(jacobian.as_mut()));
        (residual[0], jacobian)
    }

    fn assert_close(fd: f64, analytic: f64, what: &str) {
        assert!(
            (fd - analytic).abs() < 1e-5 * (1.0 + analytic.abs()),
            "{what}: fd {fd} vs {analytic}"
        );
    }

    /// Frame-to-submap and submap-to-submap Jacobians against central
    /// differences, every block, in group `G`'s own chart.
    fn check_group<G: LieGroup + Send + Sync + 'static>(anchor: G, state: G) -> Result<(), String> {
        let dof = G::TangentVector::DIM;
        let triple = |a: &G, k: &G, e: &SE3| {
            vec![
                a.as_param_slice().to_vec(),
                k.as_param_slice().to_vec(),
                e.as_param_slice().to_vec(),
            ]
        };

        let frame = LidarSubmapFieldFactor::<G, _>::new(field(), point(), 0.1)?;
        let (_, analytic) = linearize(&frame, &triple(&anchor, &state, &extrinsic()));
        let residual = |a: &G, k: &G, e: &SE3| linearize(&frame, &triple(a, k, e)).0;
        for axis in 0..dof {
            let fd = central_difference(&anchor, axis, |a| residual(a, &state, &extrinsic()));
            assert_close(
                fd,
                analytic[(0, axis)],
                &format!("{} anchor {axis}", G::NAME),
            );
            let fd = central_difference(&state, axis, |k| residual(&anchor, k, &extrinsic()));
            assert_close(
                fd,
                analytic[(0, dof + axis)],
                &format!("{} state {axis}", G::NAME),
            );
        }
        for axis in 0..6 {
            let fd = central_difference(&extrinsic(), axis, |e| residual(&anchor, &state, e));
            let a = analytic[(0, 2 * dof + axis)];
            assert_close(fd, a, &format!("{} extrinsic {axis}", G::NAME));
        }

        let submap = SubmapAlignmentFactor::<G, _>::new(field(), point(), 0.1)?;
        let pair = |a: &G, b: &G| vec![a.as_param_slice().to_vec(), b.as_param_slice().to_vec()];
        let (_, analytic) = linearize(&submap, &pair(&anchor, &state));
        for axis in 0..dof {
            let fd = central_difference(&anchor, axis, |a| linearize(&submap, &pair(a, &state)).0);
            assert_close(
                fd,
                analytic[(0, axis)],
                &format!("{} submap A {axis}", G::NAME),
            );
            let fd = central_difference(&state, axis, |b| linearize(&submap, &pair(&anchor, b)).0);
            let a = analytic[(0, dof + axis)];
            assert_close(fd, a, &format!("{} submap B {axis}", G::NAME));
        }
        Ok(())
    }

    #[test]
    fn se3_poses() -> Result<(), String> {
        check_group(
            SE3::new(Vector3::new(0.2, 0.1, -0.1), rotation(0.1)),
            SE3::new(Vector3::new(0.5, -0.3, 0.2), rotation(0.3)),
        )
    }

    #[test]
    fn se23_navigation_states() -> Result<(), String> {
        check_group(
            SE23::new(
                Vector3::new(0.2, 0.1, -0.1),
                Vector3::new(1.0, 0.0, 0.2),
                rotation(0.1),
            ),
            SE23::new(
                Vector3::new(0.5, -0.3, 0.2),
                Vector3::new(0.3, 0.1, -0.2),
                rotation(0.3),
            ),
        )
    }

    #[test]
    fn sim3_similarities() -> Result<(), String> {
        check_group(
            Sim3::new(Vector3::new(0.2, 0.1, -0.1), rotation(0.1), 0.9),
            Sim3::new(Vector3::new(0.5, -0.3, 0.2), rotation(0.3), 1.2),
        )
    }

    #[test]
    fn sgal3_galilean_poses() -> Result<(), String> {
        check_group(
            SGal3::new(
                Vector3::new(0.2, 0.1, -0.1),
                Vector3::new(1.0, 0.0, 0.2),
                rotation(0.1),
                0.2,
            ),
            SGal3::new(
                Vector3::new(0.5, -0.3, 0.2),
                Vector3::new(0.3, 0.1, -0.2),
                rotation(0.3),
                0.5,
            ),
        )
    }

    #[test]
    fn so3_rotations() -> Result<(), String> {
        check_group(SO3::new(rotation(0.1)), SO3::new(rotation(0.3)))
    }

    /// With anchor and state equal, the residual is the field at the point's
    /// body position, over σ.
    #[test]
    fn the_residual_is_the_field_over_sigma() -> Result<(), String> {
        let pose = SE3::new(Vector3::new(0.5, -0.3, 0.2), rotation(0.3));
        let factor = LidarSubmapFieldFactor::<SE3, _>::new(field(), point(), 0.5)?;
        let params = vec![
            pose.as_param_slice().to_vec(),
            pose.as_param_slice().to_vec(),
            extrinsic().as_param_slice().to_vec(),
        ];
        let (residual, _) = linearize(&factor, &params);
        let body = extrinsic().act(&point(), None, None);
        let (value, _) = field().query(&body).ok_or("outside")?;
        assert!((residual - value / 0.5).abs() < 1e-12, "{residual}");
        Ok(())
    }

    #[test]
    fn points_outside_the_field_contribute_nothing() -> Result<(), String> {
        let small = Arc::new(Ellipsoid { radius: 0.1 });
        let pose = SE3::new(Vector3::new(0.5, -0.3, 0.2), rotation(0.3));
        let factor = LidarSubmapFieldFactor::<SE3, _>::new(small.clone(), point(), 0.1)?;
        let params = [
            SE3::identity().as_param_slice().to_vec(),
            pose.as_param_slice().to_vec(),
            extrinsic().as_param_slice().to_vec(),
        ];
        let slices: Vec<&[f64]> = params.iter().map(Vec::as_slice).collect();
        let mut jacobian = Mat::<f64>::from_fn(1, 18, |_, _| 7.0);
        let mut residual = [7.0];
        factor.linearize(&slices, &mut residual, Some(jacobian.as_mut()));
        assert_eq!(residual[0], 0.0);
        assert!((0..18).all(|col| jacobian[(0, col)] == 0.0));

        let weighted = LidarSubmapFieldFactor::<SE3, _>::with_occupancy_weighting(
            small,
            point(),
            -2.0,
            0.02,
            &SE3::identity(),
            &pose,
            &extrinsic(),
        );
        assert!(weighted.is_err());
        Ok(())
    }

    #[test]
    fn occupancy_weighting_follows_okvis2x() -> Result<(), String> {
        let gradient = Vector3::new(3.0, 0.0, 4.0);
        let sigma = occupancy_sigma(&gradient, -3.0, 0.1).ok_or("no sigma")?;
        assert!((sigma - 1.25_f64.sqrt()).abs() < 1e-12, "{sigma}");
        assert!(occupancy_sigma(&gradient, 0.0, 0.0).is_none());

        let anchor = SE23::new(
            Vector3::new(0.2, 0.1, -0.1),
            Vector3::zeros(),
            rotation(0.1),
        );
        let state = SE23::new(
            Vector3::new(0.5, -0.3, 0.2),
            Vector3::zeros(),
            rotation(0.3),
        );
        let frame = LidarSubmapFieldFactor::<SE23, _>::with_occupancy_weighting(
            field(),
            point(),
            -2.0,
            0.02,
            &anchor,
            &state,
            &extrinsic(),
        )?;
        let submap = SubmapAlignmentFactor::<SE23, _>::with_occupancy_weighting(
            field(),
            point(),
            -2.0,
            0.02,
            &anchor,
            &state,
        )?;
        // Never below the map term L_min/3.
        assert!(frame.sigma() >= 2.0 / 3.0 && submap.sigma() >= 2.0 / 3.0);
        Ok(())
    }

    #[test]
    fn groups_and_inputs_are_checked() -> Result<(), String> {
        assert!(LidarSubmapFieldFactor::<SE2, _>::new(field(), point(), 0.1).is_err());
        assert!(SubmapAlignmentFactor::<SE2, _>::new(field(), point(), 0.1).is_err());
        assert!(LidarSubmapFieldFactor::<SE3, _>::new(field(), point(), 0.0).is_err());
        let nan = Vector3::new(f64::NAN, 0.0, 0.0);
        assert!(SubmapAlignmentFactor::<SE3, _>::new(field(), nan, 0.1).is_err());
        let frame = LidarSubmapFieldFactor::<SE23, _>::new(field(), point(), 0.1)?;
        assert_eq!(frame.jacobian_shape(), (1, 24));
        assert!(frame.whitens_internally());
        let submap = SubmapAlignmentFactor::<SE23, _>::new(field(), point(), 0.1)?;
        assert_eq!(submap.jacobian_shape(), (1, 18));
        assert!(submap.whitens_internally());
        Ok(())
    }

    /// `into_anchor` inverts the group action, and `∂p_A/∂w` undoes its linear
    /// part — here with a scale.
    #[test]
    fn into_anchor_inverts_the_action() {
        let anchor = Sim3::new(Vector3::new(0.2, 0.1, -0.1), rotation(0.1), 1.3);
        let local = Vector3::new(1.0, 2.0, -0.5);
        let chain = into_anchor(&anchor, &anchor.act(&local, None, None));
        assert!((chain.point - local).norm() < 1e-12);
        let step =
            anchor.act(&Vector3::x(), None, None) - anchor.act(&Vector3::zeros(), None, None);
        assert!((chain.d_world * step - Vector3::x()).norm() < 1e-12);
    }
}
