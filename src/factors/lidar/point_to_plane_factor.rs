//! Point-to-plane correspondence factor (LIO-SAM `PoseToPlane` analogue).
//!
//! Constrains a body-to-world pose `T_wr` against a matched plane in the
//! target frame:
//!
//! ```text
//! r = nᵀ·(T_wr · p_body) + d        (1D)
//! ```
//!
//! with the target plane `nᵀ·x + d = 0` measured upstream from the target
//! cloud (plane fitting is data association, not optimization). This is the
//! standard lidar-odometry residual that needs no distance field — unlike
//! [`IcpFactor`](crate::factors::icp_factor::IcpFactor) it evaluates a single
//! matched correspondence.

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, Vector3};

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;

/// A plane `nᵀ·x + d = 0` with a unit normal.
#[derive(Clone, Debug)]
pub struct Plane {
    /// Unit normal.
    pub normal: Vector3<f64>,
    /// Offset: points on the plane satisfy `normalᵀ·x + offset = 0`.
    pub offset: f64,
}

impl Plane {
    /// Create a plane and normalize it. Rejects near-zero normals.
    pub fn new(normal: Vector3<f64>, offset: f64) -> Result<Self, String> {
        let norm = normal.norm();
        if !(norm.is_finite() && norm > 1e-12) {
            return Err("plane normal must be non-zero and finite".into());
        }
        Ok(Self {
            normal: normal / norm,
            offset: offset / norm,
        })
    }
}

/// Point-to-plane factor over `[T_wr, p_body]`.
///
/// The source point is a variable block (it may be refined or held fixed by
/// giving the optimizer a non-optimized variable), matching the LIO-SAM
/// formulation where scan points are treated as measurements with fixed
/// position but the factor still exposes the block for flexibility.
#[derive(Clone)]
pub struct PointToPlaneFactor {
    /// Target-frame plane.
    pub plane: Plane,
}

impl PointToPlaneFactor {
    /// Create the factor from the target plane.
    pub fn new(plane: Plane) -> Self {
        Self { plane }
    }
}

impl Factor for PointToPlaneFactor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        debug_assert_eq!(params.len(), 2, "PointToPlaneFactor expects [T_wr, p_body]");
        debug_assert_eq!(params[0].len(), 7, "params[0] must be SE3 (7D)");
        debug_assert_eq!(params[1].len(), 3, "params[1] must be a 3D point");

        let pose = SE3::from_param_slice(params[0]);
        let p_body = Vector3::new(params[1][0], params[1][1], params[1][2]);

        let predicted = pose.act(&p_body, None, None);
        residual[0] = self.plane.normal.dot(&predicted) + self.plane.offset;

        let Some(mut jac) = jacobian else { return };

        let rotation = pose.rotation_so3().rotation_matrix();
        let p_x = Matrix3::new(
            0.0,
            -p_body.z,
            p_body.y,
            p_body.z,
            0.0,
            -p_body.x,
            -p_body.y,
            p_body.x,
            0.0,
        );

        // ∂r/∂(δρ, δθ) = [nᵀR | −nᵀR·p̂_body];  ∂r/∂p_body = Rᵀn
        let n_r = self.plane.normal.transpose() * rotation; // 1×3
        let r_t_n = rotation.transpose() * self.plane.normal; // 3×1
        for col in 0..3 {
            *jac.rb_mut().get_mut(0, col) = n_r[col];
            *jac.rb_mut().get_mut(0, 3 + col) = -(n_r * p_x)[(0, col)];
            *jac.rb_mut().get_mut(0, 6 + col) = r_t_n[col];
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, 9)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        if variables.len() != 2
            || variables[0].as_param_slice().len() != SE3::REP_SIZE
            || variables[1].as_param_slice().len() != 3
        {
            return Err("PointToPlaneFactor expects [SE3 pose, 3D point]".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::SE3Tangent;

    type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

    #[test]
    fn zero_residual_on_plane() -> TestResult<()> {
        // Plane z = 1  →  n = (0,0,1), d = −1.
        let plane = Plane::new(Vector3::new(0.0, 0.0, 1.0), -1.0)?;
        let p_body = Vector3::new(0.5, -0.4, 0.7);
        // Choose the translation so the transformed point sits exactly on the
        // plane for this rotation.
        let rotation_q = nalgebra::UnitQuaternion::from_euler_angles(0.02, -0.01, 0.03);
        let r = rotation_q.to_rotation_matrix().into_inner();
        let p_world = Vector3::new(0.4, -0.6, 1.0);
        let translation = p_world - r * p_body;
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::from(translation),
            rotation_q,
        ));
        let factor = PointToPlaneFactor::new(plane);

        let mut residual = vec![0.0; 1];
        factor.linearize(
            &[pose.as_param_slice(), &[p_body.x, p_body.y, p_body.z]],
            &mut residual,
            None,
        );
        assert!(residual[0].abs() < 1e-12, "residual = {}", residual[0]);
        Ok(())
    }

    #[test]
    fn finite_difference_jacobians() -> TestResult<()> {
        let plane = Plane::new(Vector3::new(0.3, -0.5, 0.8), 0.2)?;
        let pose = SE3::from_isometry(nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(0.4, -0.2, 0.9),
            nalgebra::UnitQuaternion::from_euler_angles(0.03, 0.05, -0.02),
        ));
        let p_body = Vector3::new(0.2, 0.6, -0.3);
        let factor = PointToPlaneFactor::new(plane);
        let pose_vec: Vec<f64> = pose.as_param_slice().to_vec();
        let body_vec = [p_body.x, p_body.y, p_body.z];

        let mut r0 = vec![0.0; 1];
        let mut jac_buf = vec![0.0; 9];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 1, 9);
        factor.linearize(&[&pose_vec, &body_vec], &mut r0, Some(jac_mut));

        const EPS: f64 = 1e-6;
        const TOL: f64 = 1e-4;
        for col in 0..6 {
            let mut tan = [0.0f64; 6];
            tan[col] = EPS;
            let perturbed: Vec<f64> = pose
                .right_plus(&SE3Tangent::from_slice(&tan), None, None)
                .as_param_slice()
                .to_vec();
            let mut r_pert = vec![0.0; 1];
            factor.linearize(&[&perturbed, &body_vec], &mut r_pert, None);
            let fd = (r_pert[0] - r0[0]) / EPS;
            let ana = jac_buf[col];
            assert!(
                (fd - ana).abs() < TOL,
                "pose[{col}]: analytical={ana:.6} fd={fd:.6}"
            );
        }
        for col in 0..3 {
            let mut plus = body_vec;
            let mut minus = body_vec;
            plus[col] += EPS;
            minus[col] -= EPS;
            let mut r_plus = vec![0.0; 1];
            let mut r_minus = vec![0.0; 1];
            factor.linearize(&[&pose_vec, &plus], &mut r_plus, None);
            factor.linearize(&[&pose_vec, &minus], &mut r_minus, None);
            let fd = (r_plus[0] - r_minus[0]) / (2.0 * EPS);
            let ana = jac_buf[6 + col];
            assert!(
                (fd - ana).abs() < TOL,
                "point[{col}]: analytical={ana:.6} fd={fd:.6}"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_degenerate_normal() {
        assert!(Plane::new(Vector3::zeros(), 0.0).is_err());
        assert!(Plane::new(Vector3::new(f64::NAN, 0.0, 1.0), 0.0).is_err());
    }

    #[test]
    fn normal_is_normalized_on_construction() -> TestResult<()> {
        let plane = Plane::new(Vector3::new(0.0, 0.0, 2.0), -4.0)?;
        assert!((plane.normal.norm() - 1.0).abs() < 1e-12);
        assert!((plane.offset - (-2.0)).abs() < 1e-12);
        Ok(())
    }
}
