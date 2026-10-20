//! Planar-motion constraint for ground vehicles.

use std::marker::PhantomData;

use apex_manifolds::capability::ActsOnPoints3;
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, Vector3};

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::group_point::act_point;

/// A vehicle confined to a horizontal plane: fixed height, no roll, no pitch.
///
/// ```text
/// r = [ t.z − height,          altitude
///       e_zᵀ·R·e_x,            body x-axis is level
///       e_zᵀ·R·e_y ]           body y-axis is level      (3D)
/// ```
///
/// The two tilt rows are the world-z components of the body's x and y axes,
/// rather than roll and pitch angles: extracting Euler angles introduces a
/// gimbal singularity and a discontinuity that a least-squares residual should
/// not have, while these rows are smooth everywhere and vanish on exactly the
/// same set.
///
/// Three constraints per pose. On an indoor or road vehicle they remove the
/// three least-observable degrees of freedom in a 6-DOF estimate.
///
/// # Parameter layout (1 block)
///
/// ```text
/// params[0]: a state that moves 3-D points (`SE3`, `SE23`, `Sim3`, `SO3`)
/// ```
#[derive(Debug, Clone)]
pub struct PlanarMotionFactor<G: ActsOnPoints3> {
    /// World height of the plane the vehicle travels on [m].
    height: f64,
    group: PhantomData<fn() -> G>,
}

impl<G: ActsOnPoints3> PlanarMotionFactor<G> {
    /// Constrain the pose to the plane `z = height`.
    pub fn new(height: f64) -> Self {
        Self {
            height,
            group: PhantomData,
        }
    }

    /// Constrain the pose to the plane `z = 0`.
    pub fn ground() -> Self {
        Self::new(0.0)
    }
}

impl<G: ActsOnPoints3 + Send + Sync + 'static> Factor for PlanarMotionFactor<G> {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let state = G::from_param_slice(params[0]);
        let origin = act_point(&state, &Vector3::zeros());
        residual[0] = origin.world.z - self.height;

        let mut jac = jacobian;
        if let Some(jac) = jac.as_mut() {
            for row in 0..3 {
                for col in 0..G::DOF {
                    *jac.rb_mut().get_mut(row, col) = 0.0;
                }
            }
            // Height: the z row of the origin's chain.
            for col in 0..G::DOF {
                *jac.rb_mut().get_mut(0, col) = origin.d_state[(2, col)];
            }
        }

        // World-z component of each rotated body axis, normalized so a scale
        // (`Sim3`) does not leak in: the rotation part of the action is the
        // difference of the action on the axis and on the origin.
        for (row, axis) in [Vector3::x(), Vector3::y()].iter().enumerate() {
            let at_axis = act_point(&state, axis);
            let rotated = at_axis.world - origin.world;
            let norm = rotated.norm();
            let unit = rotated / norm;
            residual[1 + row] = unit.z;

            if let Some(jac) = jac.as_mut() {
                // ∂n = (I − nnᵀ)/‖v‖ · ∂v, ∂v the axis chain minus the origin's.
                let projector = (Matrix3::identity() - unit * unit.transpose()) / norm;
                let d_n = projector * (at_axis.d_state - origin.d_state);
                for col in 0..G::DOF {
                    *jac.rb_mut().get_mut(row + 1, col) = d_n[(2, col)];
                }
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, G::DOF)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        match variables {
            [state] if state.manifold_type_name() == G::NAME => Ok(()),
            _ => Err(format!("PlanarMotionFactor expects [{} state]", G::NAME)),
        }
    }
}
