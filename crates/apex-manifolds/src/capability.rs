//! Capabilities a factor can ask of a group, decided by the type system.
//!
//! A factor states what the mathematics needs (`G: ActsOnPoints3`) instead of
//! checking the group's name at run time, so a group that cannot do the job
//! fails to compile rather than failing at registration.

use crate::LieGroup;
use crate::se2::SE2;
use crate::se3::SE3;
use crate::se23::SE23;
use crate::sgal3::SGal3;
use crate::sim3::Sim3;
use crate::so2::SO2;
use crate::so3::SO3;

/// A group whose [`LieGroup::act`] is a true action on 3-D points:
/// `X⁻¹ ⊙ (X ⊙ x) = x` for every element and point.
///
/// Not implemented for `SE2`/`SO2` (they act in the plane), `Rn` (runtime
/// dimension) or `SGal3` (its `act` adds `t·v`, which its inverse does not
/// undo).
pub trait ActsOnPoints3: ConstDoF {}

impl ActsOnPoints3 for SO3 {}
impl ActsOnPoints3 for SE3 {}
impl ActsOnPoints3 for SE23 {}
impl ActsOnPoints3 for Sim3 {}

/// A group with a tangent dimension known at compile time (everything but
/// `Rn`).
pub trait ConstDoF: LieGroup {
    /// Degrees of freedom of the tangent space.
    const DOF: usize;
}

impl ConstDoF for SO2 {
    const DOF: usize = 1;
}
impl ConstDoF for SE2 {
    const DOF: usize = 3;
}
impl ConstDoF for SO3 {
    const DOF: usize = 3;
}
impl ConstDoF for SE3 {
    const DOF: usize = 6;
}
impl ConstDoF for Sim3 {
    const DOF: usize = 7;
}
impl ConstDoF for SE23 {
    const DOF: usize = 9;
}
impl ConstDoF for SGal3 {
    const DOF: usize = 10;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tangent;
    use nalgebra::Vector3;

    fn dof_matches_tangent<G: ConstDoF>() -> bool {
        G::DOF == <G::TangentVector as Tangent<G>>::DIM && G::random().tangent_dim() == G::DOF
    }

    fn action_is_invertible<G: ActsOnPoints3>() -> bool {
        (0..16).all(|_| {
            let g = G::random();
            let x = Vector3::new(0.7, -1.3, 2.1);
            let moved = g.act(&x, None, None);
            let back = g.inverse(None).act(&moved, None, None);
            (back - x).norm() < 1e-9
        })
    }

    #[test]
    fn const_dof_agrees_with_the_tangent_dimension() {
        assert!(dof_matches_tangent::<SO2>());
        assert!(dof_matches_tangent::<SE2>());
        assert!(dof_matches_tangent::<SO3>());
        assert!(dof_matches_tangent::<SE3>());
        assert!(dof_matches_tangent::<Sim3>());
        assert!(dof_matches_tangent::<SE23>());
        assert!(dof_matches_tangent::<SGal3>());
    }

    #[test]
    fn point_actions_are_invertible_for_every_admitted_group() {
        assert!(action_is_invertible::<SO3>());
        assert!(action_is_invertible::<SE3>());
        assert!(action_is_invertible::<SE23>());
        assert!(action_is_invertible::<Sim3>());
    }

    #[test]
    fn sgal3_is_excluded_because_its_act_is_not_an_action() {
        let g = SGal3::random();
        let x = Vector3::new(0.7, -1.3, 2.1);
        let moved = g.act(&x, None, None);
        let back = g.inverse(None).act(&moved, None, None);
        assert!((back - x).norm() > 1e-9);
    }
}
