//! Planar-motion constraint for ground vehicles.

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use faer::prelude::ReborrowMut;
use nalgebra::Vector3;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::math::skew;
use crate::factors::common::validate::expect_block_sizes;

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
/// # Parameter layout (1 block, 6 DOF)
///
/// ```text
/// params[0]: SE3 pose — 7D, 6 DOF
/// ```
#[derive(Debug, Clone)]
pub struct PlanarMotionFactor {
    /// World height of the plane the vehicle travels on [m].
    height: f64,
}

impl PlanarMotionFactor {
    /// Constrain the pose to the plane `z = height`.
    pub fn new(height: f64) -> Self {
        Self { height }
    }

    /// Constrain the pose to the plane `z = 0`.
    pub fn ground() -> Self {
        Self::new(0.0)
    }
}

impl Factor for PlanarMotionFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let pose = SE3::from_param_slice(params[0]);
        let rotation = pose.rotation_so3().rotation_matrix();

        residual[0] = pose.translation().z - self.height;
        residual[1] = rotation[(2, 0)]; // e_zᵀ R e_x
        residual[2] = rotation[(2, 1)]; // e_zᵀ R e_y

        let Some(mut jac) = jacobian else { return };
        for row in 0..3 {
            for col in 0..6 {
                *jac.rb_mut().get_mut(row, col) = 0.0;
            }
        }

        // Height: t ← t + R·δρ, so ∂(t.z)/∂δρ is the third row of R.
        for col in 0..3 {
            *jac.rb_mut().get_mut(0, col) = rotation[(2, col)];
        }

        // Tilt: R ← R·Exp(δθ) ≈ R(I + [δθ]ₓ), so
        //     ∂(e_zᵀ R e_a)/∂δθ = −e_zᵀ R [e_a]ₓ.
        let z_row = rotation.row(2).transpose(); // (Rᵀe_z) as a column
        for (row, axis) in [Vector3::x(), Vector3::y()].iter().enumerate() {
            let d = -(z_row.transpose() * skew(axis));
            for col in 0..3 {
                *jac.rb_mut().get_mut(row + 1, 3 + col) = d[(0, col)];
            }
        }
    }

    fn residual_dim(&self) -> usize {
        3
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (3, 6)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE3::REP_SIZE],
            "PlanarMotionFactor expects [SE3 pose]",
        )
    }
}
