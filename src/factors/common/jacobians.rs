//! Jacobian blocks shared by more than one factor family.

use nalgebra::{Matrix3, SMatrix, Vector3};

use super::math::skew;

/// Jacobian of a rigid transform applied to a point, `∂(T · p) / ∂(δρ, δθ)`,
/// under the **right** SE(3) perturbation model.
///
/// ```text
/// ∂(T p)/∂(δρ, δθ) = [ R | −R·[p]× ]        (3×6)
/// ```
///
/// The companion point block is simply `∂(T p)/∂p = R`, returned by
/// [`pose_point_point_block`].
pub fn pose_point_jacobian(rotation: &Matrix3<f64>, point: &Vector3<f64>) -> SMatrix<f64, 3, 6> {
    let mut jac = SMatrix::<f64, 3, 6>::zeros();
    jac.fixed_view_mut::<3, 3>(0, 0).copy_from(rotation);
    jac.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-rotation * skew(point)));
    jac
}

/// Jacobian of a rigid transform applied to a point with respect to the point:
/// `∂(T · p) / ∂p = R`.
///
/// Trivial on its own, but named so call sites read symmetrically alongside
/// [`pose_point_jacobian`].
pub fn pose_point_point_block(rotation: &Matrix3<f64>) -> Matrix3<f64> {
    *rotation
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::LieGroup;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::{SE3, SE3Tangent};

    /// The analytic block must match a finite-difference of `T ⊞ δ` acting on p.
    #[test]
    fn pose_point_jacobian_matches_finite_difference() {
        let pose = SE3::from_param_slice(&[0.3, -0.4, 0.9, 0.9238795, 0.0, 0.3826834, 0.0]);
        let point = Vector3::new(0.7, -1.3, 2.1);
        let rotation = pose.rotation_so3().rotation_matrix();
        let analytic = pose_point_jacobian(&rotation, &point);

        let base = pose.act(&point, None, None);
        let eps = 1e-7;
        for k in 0..6 {
            let mut tangent = [0.0f64; 6];
            tangent[k] = eps;
            let perturbed = pose.right_plus(&SE3Tangent::from_slice(&tangent), None, None);
            let fd = (perturbed.act(&point, None, None) - base) / eps;
            for row in 0..3 {
                assert!(
                    (analytic[(row, k)] - fd[row]).abs() < 1e-6,
                    "col {k} row {row}: analytic={} fd={}",
                    analytic[(row, k)],
                    fd[row]
                );
            }
        }
    }

    #[test]
    fn point_block_is_the_rotation() {
        let pose = SE3::from_param_slice(&[0.0, 0.0, 0.0, 0.9238795, 0.0, 0.0, 0.3826834]);
        let rotation = pose.rotation_so3().rotation_matrix();
        assert!((pose_point_point_block(&rotation) - rotation).norm() < 1e-15);
    }
}
