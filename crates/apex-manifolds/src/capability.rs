//! Capabilities a factor can ask of a group, decided by the type system.
//!
//! A factor states what the mathematics needs (`G: ActsOnPoints3`) instead of
//! checking the group's name at run time, so a group that cannot do the job
//! fails to compile rather than failing at registration.

use nalgebra::{Matrix3, Vector3};

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

/// A rigid pose whose tangent starts `[δρ, δθ, …]` under right perturbation:
/// the translation moves as `t ← t + R·δρ`, the rotation as `R ← R·Exp(δθ)`,
/// and every later tangent axis leaves the point action untouched.
///
/// That is the layout closed-form reprojection Jacobians are written against;
/// `Sim3` (a scale axis scales the action) and `SO3` (no translation) do not
/// fit it.
pub trait RigidPose: ActsOnPoints3 {
    /// Floats in the parameter block.
    const REP_SIZE: usize;
    /// The rotation matrix `R`.
    fn rotation(&self) -> Matrix3<f64>;
    /// The position `t`.
    fn position(&self) -> Vector3<f64>;
}

impl RigidPose for SE3 {
    const REP_SIZE: usize = SE3::REP_SIZE;
    fn rotation(&self) -> Matrix3<f64> {
        self.rotation_so3().rotation_matrix()
    }
    fn position(&self) -> Vector3<f64> {
        self.translation()
    }
}

impl RigidPose for SE23 {
    const REP_SIZE: usize = SE23::REP_SIZE;
    fn rotation(&self) -> Matrix3<f64> {
        self.rotation_matrix()
    }
    fn position(&self) -> Vector3<f64> {
        self.translation()
    }
}

/// A group whose tangent carries a frame velocity block: `SE23` and `SGal3`.
///
/// Under the right perturbation the velocity moves as `v ← v + R·δν`, which is
/// what the zero-velocity and nonholonomic constraints differentiate.
pub trait HasVelocity: ConstDoF {
    /// First tangent index of the velocity block `ν`.
    const VELOCITY_AT: usize;
    /// First tangent index of the rotation block `θ`.
    const ROTATION_AT: usize;
    /// World-frame velocity.
    fn world_velocity(&self) -> Vector3<f64>;
    /// Body-to-world rotation.
    fn body_to_world(&self) -> Matrix3<f64>;
}

impl HasVelocity for SE23 {
    const VELOCITY_AT: usize = 6;
    const ROTATION_AT: usize = 3;
    fn world_velocity(&self) -> Vector3<f64> {
        self.velocity()
    }
    fn body_to_world(&self) -> Matrix3<f64> {
        self.rotation_matrix()
    }
}

impl HasVelocity for SGal3 {
    const VELOCITY_AT: usize = 3;
    const ROTATION_AT: usize = 6;
    fn world_velocity(&self) -> Vector3<f64> {
        self.velocity()
    }
    fn body_to_world(&self) -> Matrix3<f64> {
        self.rotation_matrix()
    }
}

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

    fn rigid_layout_holds<G: RigidPose>() -> bool {
        (0..16).all(|_| {
            let g = G::random();
            let p = Vector3::new(0.4, -0.9, 1.7);
            let mut jacobian = g.zero_jacobian_for();
            let _ = g.act(&p, Some(&mut jacobian), None);
            let r = g.rotation();
            let skew = crate::util::skew(&p);
            let head = (0..3).all(|row| {
                (0..3).all(|col| {
                    (jacobian[(row, col)] - r[(row, col)]).abs() < 1e-12
                        && (jacobian[(row, 3 + col)] + (r * skew)[(row, col)]).abs() < 1e-12
                })
            });
            let tail = (0..3).all(|row| (6..G::DOF).all(|col| jacobian[(row, col)] == 0.0));
            let acts = (g.act(&p, None, None) - (r * p + g.position())).norm() < 1e-12;
            head && tail && acts
        })
    }

    #[test]
    fn rigid_poses_share_the_tangent_layout() {
        assert!(rigid_layout_holds::<SE3>());
        assert!(rigid_layout_holds::<SE23>());
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
