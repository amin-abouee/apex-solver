//! Moving 3-D points with a generic pose group, derivatives from the group.
//!
//! Factors generic over a [`LieGroup`] carry points through `X ⊙ p` and
//! `X⁻¹ ⊙ p`; both, with their Jacobians in the group's own chart, come from
//! the group's `act` and `inverse`.

use nalgebra::{Matrix3, SMatrix, Vector3};

use apex_manifolds::capability::ActsOnPoints3;

use apex_manifolds::LieGroup;

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
    let mut d_inverse = anchor.zero_jacobian_for();
    let inverse = anchor.inverse(Some(&mut d_inverse));
    let mut d_act = anchor.zero_jacobian_for();
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

/// Largest tangent dimension of a group that moves 3-D points (`SE23`).
pub const MAX_ACT_DOF: usize = 9;

/// A body point carried to the world, with its derivatives.
#[derive(Clone, Debug)]
pub struct PointAct {
    /// `X ⊙ p`.
    pub world: Vector3<f64>,
    /// `∂(X ⊙ p)/∂X`, one column per tangent axis (the rest zero).
    pub d_state: SMatrix<f64, 3, MAX_ACT_DOF>,
    /// `∂(X ⊙ p)/∂p`.
    pub d_point: Matrix3<f64>,
}

/// Carry body point `point` to the world by `state`.
pub fn act_point<G: ActsOnPoints3>(state: &G, point: &Vector3<f64>) -> PointAct {
    const { assert!(G::DOF <= MAX_ACT_DOF) };
    let mut jacobian = state.zero_jacobian_for();
    let mut d_point = Matrix3::zeros();
    let world = state.act(point, Some(&mut jacobian), Some(&mut d_point));
    PointAct {
        world,
        d_state: SMatrix::from_fn(|row, col| {
            if col < G::DOF {
                jacobian[(row, col)]
            } else {
                0.0
            }
        }),
        d_point,
    }
}

/// A point of frame B seen from frame A, `p_A = X_a⁻¹ ⊙ (X_b ⊙ p_B)`.
#[derive(Clone, Debug)]
pub struct RelativePoint {
    /// `p_A`.
    pub point: Vector3<f64>,
    /// `∂p_A/∂X_a`.
    pub d_a: SMatrix<f64, 3, MAX_ACT_DOF>,
    /// `∂p_A/∂X_b`.
    pub d_b: SMatrix<f64, 3, MAX_ACT_DOF>,
}

/// Carry `point_b` from frame B into frame A.
pub fn relative_point<G: ActsOnPoints3>(a: &G, b: &G, point_b: &Vector3<f64>) -> RelativePoint {
    let act_b = act_point(b, point_b);
    let chain = into_anchor(a, &act_b.world);
    RelativePoint {
        point: chain.point,
        d_a: SMatrix::from_fn(|row, col| chain.d_anchor.get(col).map_or(0.0, |axis| axis[row])),
        d_b: chain.d_world * act_b.d_state,
    }
}

/// Registration check for a `[state, 3-D point]` factor.
pub(crate) fn expect_state_and_point<G: ActsOnPoints3>(
    variables: &[&dyn crate::core::variable::ManifoldVariable],
    name: &str,
) -> Result<(), String> {
    match variables {
        [state, point]
            if state.manifold_type_name() == G::NAME && point.as_param_slice().len() == 3 =>
        {
            Ok(())
        }
        _ => Err(format!("{name} expects [{} state, 3D point]", G::NAME)),
    }
}

/// Registration check for a `[state, state]` factor.
pub(crate) fn expect_two_states<G: ActsOnPoints3>(
    variables: &[&dyn crate::core::variable::ManifoldVariable],
    name: &str,
) -> Result<(), String> {
    match variables {
        [a, b] if a.manifold_type_name() == G::NAME && b.manifold_type_name() == G::NAME => Ok(()),
        _ => Err(format!("{name} expects [{0} state, {0} state]", G::NAME)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::se3::SE3;
    use apex_manifolds::se23::SE23;
    use apex_manifolds::sgal3::SGal3;
    use apex_manifolds::sim3::Sim3;
    use apex_manifolds::so3::SO3;
    use nalgebra::UnitQuaternion;

    fn rotation(roll: f64) -> UnitQuaternion<f64> {
        UnitQuaternion::from_euler_angles(roll, -0.3, 0.7)
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

    /// The law that admits a group: its inverse undoes its action.
    fn inverse_undoes_act<G: LieGroup>(g: &G) -> bool {
        let x = Vector3::new(1.0, 2.0, -0.5);
        (into_anchor(g, &g.act(&x, None, None)).point - x).norm() < 1e-12
    }

    #[test]
    fn admitted_groups_are_actions_and_sgal3_is_not() {
        let r = rotation(0.2);
        assert!(inverse_undoes_act(&SE3::new(
            Vector3::new(1.0, 0.0, 2.0),
            r
        )));
        assert!(inverse_undoes_act(&SE23::new(
            Vector3::new(1.0, 0.0, 2.0),
            Vector3::new(0.5, 0.1, 0.0),
            r
        )));
        assert!(inverse_undoes_act(&Sim3::new(
            Vector3::new(1.0, 0.0, 2.0),
            r,
            1.4
        )));
        assert!(inverse_undoes_act(&SO3::new(r)));
        let sgal3 = SGal3::new(
            Vector3::new(1.0, 0.0, 2.0),
            Vector3::new(0.5, 0.1, 0.0),
            r,
            0.3,
        );
        assert!(!inverse_undoes_act(&sgal3));
    }
}
