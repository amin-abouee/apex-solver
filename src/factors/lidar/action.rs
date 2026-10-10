//! A group's action on 3-D points as an affine map `s ↦ A·s + b`, with its
//! Jacobian `∂(X⊙s)/∂X = J(0) + Σ_k s_k·(J(e_k) − J(0))` from four
//! evaluations — shared by the LiDAR factors, whose thousands of points or
//! planes all act through the same few states.

use std::any::TypeId;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use apex_manifolds::LieGroup;
use nalgebra::{Matrix3, SMatrix, Vector3};

/// A state's affine map and action Jacobians; see [`affine_action`].
pub(crate) type Action = (Matrix3<f64>, Vector3<f64>, ActionJacobian);

/// Largest tangent dimension of a group acting on points (`SE₂(3)`).
pub(crate) const MAX_DOF: usize = 9;

/// `∂(X⊙s)/∂X` of the affine map `s ↦ A·s + b`: at the origin (`∂b`), and
/// its change per unit axis (`∂(A·e_k)`), zero past the group's dimension.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ActionJacobian {
    /// `∂b/∂X`.
    pub(crate) origin: SMatrix<f64, 3, MAX_DOF>,
    /// `∂(A·e_k)/∂X`, `k = x, y, z`.
    pub(crate) axes: [SMatrix<f64, 3, MAX_DOF>; 3],
}

/// Recent states' actions per thread, keyed by group and exact parameters.
const ACTION_CACHE: usize = 32;

/// A cached action: its group, the state's parameters, the action.
type CachedAction = (TypeId, Vec<f64>, Rc<Action>);

thread_local! {
    static ACTIONS: RefCell<VecDeque<CachedAction>> =
        RefCell::new(VecDeque::with_capacity(ACTION_CACHE));
}

/// The affine map `s ↦ A·s + b` of `state`'s action, and the action's
/// Jacobian `∂(X⊙s)/∂X` (3 × dof) at the origin and at each unit axis.
pub(crate) fn affine_action<G: LieGroup>(state: &G) -> Action {
    let dof = state.tangent_dim().min(MAX_DOF);
    let at = |s: &Vector3<f64>| {
        let mut jacobian = G::zero_jacobian();
        let world = state.act(s, Some(&mut jacobian), None);
        let columns =
            SMatrix::<f64, 3, MAX_DOF>::from_fn(
                |r, c| if c < dof { jacobian[(r, c)] } else { 0.0 },
            );
        (world, columns)
    };
    let (b, j0) = at(&Vector3::zeros());
    let (x, jx) = at(&Vector3::x());
    let (y, jy) = at(&Vector3::y());
    let (z, jz) = at(&Vector3::z());
    let a = Matrix3::from_columns(&[x - b, y - b, z - b]);
    let jacobian = ActionJacobian {
        axes: [jx - j0, jy - j0, jz - j0],
        origin: j0,
    };
    (a, b, jacobian)
}

/// [`affine_action`] of the state stored as `param`, shared by every plane
/// a window state sees: a window has a handful of states and thousands of
/// planes, all linearized at the same parameters.
pub(crate) fn cached_action<G: LieGroup + 'static>(param: &[f64]) -> Rc<Action> {
    let group = TypeId::of::<G>();
    ACTIONS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some((_, _, action)) = cache
            .iter()
            .find(|(g, key, _)| *g == group && key.as_slice() == param)
        {
            return Rc::clone(action);
        }
        let action = Rc::new(affine_action(&G::from_param_slice(param)));
        if cache.len() == ACTION_CACHE {
            cache.pop_front();
        }
        cache.push_back((group, param.to_vec(), Rc::clone(&action)));
        action
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::se3::SE3;
    use apex_manifolds::sim3::Sim3;
    use nalgebra::UnitQuaternion;

    /// The cache returns exactly the computed action, keeps groups with the
    /// same parameter length apart, and survives eviction.
    #[test]
    fn cached_actions_equal_computed_ones() {
        let state = |i: usize| {
            SE3::new(
                Vector3::new(0.1 * i as f64, 0.2, -0.3),
                UnitQuaternion::from_euler_angles(0.01 * i as f64, 0.2, -0.1),
            )
        };
        for i in 0..2 * ACTION_CACHE {
            let s = state(i);
            let cached = cached_action::<SE3>(s.as_param_slice());
            assert_eq!(*cached, affine_action(&s));
            assert_eq!(*cached_action::<SE3>(s.as_param_slice()), affine_action(&s));
        }
        let first = state(0);
        assert_eq!(
            *cached_action::<SE3>(first.as_param_slice()),
            affine_action(&first)
        );
        // A different group at the same pose: a distinct entry.
        let sim = Sim3::new(first.translation(), first.rotation_quaternion(), 1.5);
        assert_eq!(
            *cached_action::<Sim3>(sim.as_param_slice()),
            affine_action(&sim)
        );
    }

    /// The affine map is the action, and its Jacobian at a point is
    /// `J(0) + Σ s_k·∂(A·e_k)`.
    #[test]
    fn the_affine_map_reproduces_the_action() {
        let state = SE3::new(
            Vector3::new(0.4, -1.2, 2.0),
            UnitQuaternion::from_euler_angles(0.3, -0.2, 1.1),
        );
        let (a, b, jacobian) = affine_action(&state);
        let s = Vector3::new(1.5, -0.7, 3.2);
        let mut expected = SE3::zero_jacobian();
        let world = state.act(&s, Some(&mut expected), None);
        assert!((a * s + b - world).norm() < 1e-12);
        let at = jacobian.origin
            + jacobian.axes[0] * s.x
            + jacobian.axes[1] * s.y
            + jacobian.axes[2] * s.z;
        for r in 0..3 {
            for c in 0..6 {
                assert!((at[(r, c)] - expected[(r, c)]).abs() < 1e-12);
            }
            for c in 6..MAX_DOF {
                assert_eq!(at[(r, c)], 0.0);
            }
        }
    }
}
