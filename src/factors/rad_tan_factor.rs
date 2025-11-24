//! Radial-Tangential (RadTan) projection factors for apex-solver optimization.
//!
//! This module provides two factor implementations for the apex-solver framework:
//! 1. `RadTanCameraParamsFactor`: Optimizes camera intrinsics (fx, fy, cx, cy, k1, k2, p1, p2, k3)
//! 2. `RadTanProjectionFactor`: Optimizes 3D point positions
//!
//! Both factors share the same residual computation but differ in their Jacobians.
//!
//! # References
//!
//! Implementation based on standard Radial-Tangential distortion model
//! and granite-headers pinhole camera: https://github.com/DLR-RM/granite/blob/master/thirdparty/granite-headers/include/granite/camera/pinhole_camera.hpp

use super::Factor;
use nalgebra::{
    DMatrix, DVector, Matrix, Matrix2xX, Matrix3xX, RawStorage, U1, U2, U3, Vector2, Vector3,
};

/// Compute residual for RadTan camera model.
///
/// # Residual Formulation
///
/// For each 3D-2D point correspondence, the residual is computed as:
/// ```text
/// residual_x = (fx * x_distorted + cx) - gt_u
/// residual_y = (fy * y_distorted + cy) - gt_v
/// ```
///
/// where distortion is applied as:
/// ```text
/// r² = x_prime² + y_prime²
/// d = 1 + k1*r² + k2*r⁴ + k3*r⁶
/// x_distorted = d*x_prime + 2*p1*x_prime*y_prime + p2*(r² + 2*x_prime²)
/// y_distorted = d*y_prime + 2*p2*x_prime*y_prime + p1*(r² + 2*y_prime²)
/// ```
///
/// # Arguments
///
/// * `point_3d` - 3D point in camera coordinates [x, y, z]
/// * `point_2d` - Observed 2D point in image coordinates [u, v]
/// * `camera_params` - Camera parameters [fx, fy, cx, cy, k1, k2, p1, p2, k3]
///
/// # Returns
///
/// `Some(residual)` if projection is valid, `None` otherwise
#[inline]
fn compute_residual_rad_tan<S3, S2>(
    point_3d: Matrix<f64, U3, U1, S3>,
    point_2d: Matrix<f64, U2, U1, S2>,
    camera_params: &DVector<f64>,
) -> Option<Vector2<f64>>
where
    S3: RawStorage<f64, U3, U1>,
    S2: RawStorage<f64, U2, U1>,
{
    let fx = camera_params[0];
    let fy = camera_params[1];
    let cx = camera_params[2];
    let cy = camera_params[3];
    let k1 = camera_params[4];
    let k2 = camera_params[5];
    let p1 = camera_params[6];
    let p2 = camera_params[7];
    let k3 = camera_params[8];
    let obs_x = point_3d[0];
    let obs_y = point_3d[1];
    let obs_z = point_3d[2];

    // Check if point is at camera center (z too small)
    if obs_z < f64::EPSILON.sqrt() {
        return None;
    }

    // Calculate normalized image coordinates
    let x_prime = obs_x / obs_z;
    let y_prime = obs_y / obs_z;

    // Compute distortion terms
    let r2 = x_prime.powi(2) + y_prime.powi(2);
    let r4 = r2.powi(2);
    let r6 = r4 * r2;

    // Radial distortion factor
    let d = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

    // Apply radial and tangential distortion
    let x_distorted =
        d * x_prime + 2.0 * p1 * x_prime * y_prime + p2 * (r2 + 2.0 * x_prime.powi(2));
    let y_distorted =
        d * y_prime + 2.0 * p2 * x_prime * y_prime + p1 * (r2 + 2.0 * y_prime.powi(2));

    // Project to image coordinates
    let u = fx * x_distorted + cx;
    let v = fy * y_distorted + cy;

    // Compute residual (difference from observed)
    Some(Vector2::new(u - point_2d[0], v - point_2d[1]))
}

/// Factor for optimizing RadTan camera intrinsic parameters.
///
/// This factor computes the reprojection error and provides Jacobians
/// with respect to camera parameters: [fx, fy, cx, cy, k1, k2, p1, p2, k3].
///
/// # Parameters
///
/// The factor optimizes 9 camera parameters:
/// - fx, fy: Focal lengths
/// - cx, cy: Principal point
/// - k1, k2, k3: Radial distortion coefficients
/// - p1, p2: Tangential distortion coefficients
#[derive(Debug, Clone)]
pub struct RadTanCameraParamsFactor {
    /// 3D points in camera coordinate system
    pub points_3d: Matrix3xX<f64>,
    /// Corresponding observed 2D points in image coordinates
    pub points_2d: Matrix2xX<f64>,
}

impl RadTanCameraParamsFactor {
    /// Creates a new RadTan camera parameters factor.
    ///
    /// # Arguments
    ///
    /// * `points_3d` - Matrix of 3D points in camera coordinates (3×N)
    /// * `points_2d` - Matrix of corresponding 2D observed points (2×N)
    ///
    /// # Panics
    ///
    /// Panics if the number of 3D and 2D points don't match.
    pub fn new(points_3d: Matrix3xX<f64>, points_2d: Matrix2xX<f64>) -> Self {
        assert_eq!(
            points_3d.ncols(),
            points_2d.ncols(),
            "Number of 3D and 2D points must match"
        );
        Self {
            points_3d,
            points_2d,
        }
    }
}

impl Factor for RadTanCameraParamsFactor {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        let camera_params = &params[0];
        let num_points = self.points_2d.ncols();
        let residual_dim = num_points * 2;

        let mut residuals = DVector::zeros(residual_dim);
        let mut jacobian_matrix = if compute_jacobian {
            Some(DMatrix::zeros(residual_dim, 9))
        } else {
            None
        };

        for i in 0..num_points {
            let point_3d = self.points_3d.column(i);
            let point_2d = self.points_2d.column(i);

            let residual = compute_residual_rad_tan(point_3d, point_2d, camera_params);

            if let Some(res) = residual {
                residuals[i * 2] = res[0];
                residuals[i * 2 + 1] = res[1];

                if let Some(ref mut jac_matrix) = jacobian_matrix {
                    // Extract camera parameters
                    let fx = camera_params[0];
                    let fy = camera_params[1];
                    let k1 = camera_params[4];
                    let k2 = camera_params[5];
                    let p1 = camera_params[6];
                    let p2 = camera_params[7];
                    let k3 = camera_params[8];

                    // Extract 3D coordinates
                    let obs_x = point_3d[0];
                    let obs_y = point_3d[1];
                    let obs_z = point_3d[2];

                    let x_prime = obs_x / obs_z;
                    let y_prime = obs_y / obs_z;

                    let r2 = x_prime.powi(2) + y_prime.powi(2);
                    let r4 = r2.powi(2);
                    let r6 = r4 * r2;

                    let d = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

                    let x_distorted = d * x_prime
                        + 2.0 * p1 * x_prime * y_prime
                        + p2 * (r2 + 2.0 * x_prime.powi(2));
                    let y_distorted = d * y_prime
                        + 2.0 * p2 * x_prime * y_prime
                        + p1 * (r2 + 2.0 * y_prime.powi(2));

                    // Jacobian w.r.t. camera parameters (inline computation)
                    // ∂residual / ∂fx
                    jac_matrix[(i * 2, 0)] = x_distorted;
                    jac_matrix[(i * 2 + 1, 0)] = 0.0;

                    // ∂residual / ∂fy
                    jac_matrix[(i * 2, 1)] = 0.0;
                    jac_matrix[(i * 2 + 1, 1)] = y_distorted;

                    // ∂residual / ∂cx
                    jac_matrix[(i * 2, 2)] = 1.0;
                    jac_matrix[(i * 2 + 1, 2)] = 0.0;

                    // ∂residual / ∂cy
                    jac_matrix[(i * 2, 3)] = 0.0;
                    jac_matrix[(i * 2 + 1, 3)] = 1.0;

                    // ∂residual / ∂k1
                    jac_matrix[(i * 2, 4)] = fx * x_prime * r2;
                    jac_matrix[(i * 2 + 1, 4)] = fy * y_prime * r2;

                    // ∂residual / ∂k2
                    jac_matrix[(i * 2, 5)] = fx * x_prime * r4;
                    jac_matrix[(i * 2 + 1, 5)] = fy * y_prime * r4;

                    // ∂residual / ∂p1
                    jac_matrix[(i * 2, 6)] = fx * 2.0 * x_prime * y_prime;
                    jac_matrix[(i * 2 + 1, 6)] = fy * (r2 + 2.0 * y_prime.powi(2));

                    // ∂residual / ∂p2
                    jac_matrix[(i * 2, 7)] = fx * (r2 + 2.0 * x_prime.powi(2));
                    jac_matrix[(i * 2 + 1, 7)] = fy * 2.0 * x_prime * y_prime;

                    // ∂residual / ∂k3
                    jac_matrix[(i * 2, 8)] = fx * x_prime * r6;
                    jac_matrix[(i * 2 + 1, 8)] = fy * y_prime * r6;
                }
            } else {
                // Invalid projection - large residual
                residuals[i * 2] = 1e6;
                residuals[i * 2 + 1] = 1e6;
            }
        }

        (residuals, jacobian_matrix)
    }

    fn get_dimension(&self) -> usize {
        self.points_2d.ncols() * 2
    }
}

/// Factor for optimizing 3D point positions with fixed RadTan camera parameters.
///
/// This factor computes the reprojection error and provides Jacobians
/// with respect to 3D point positions. Camera parameters are held constant.
///
/// The Jacobian accounts for both the pinhole projection and the RadTan distortion.
#[derive(Debug, Clone)]
pub struct RadTanProjectionFactor {
    /// Corresponding observed 2D points in image coordinates
    pub points_2d: Matrix2xX<f64>,
    /// Fixed camera parameters: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    pub camera_params: DVector<f64>,
}

impl RadTanProjectionFactor {
    /// Creates a new RadTan projection factor.
    ///
    /// # Arguments
    ///
    /// * `points_2d` - Matrix of corresponding 2D observed points (2×N)
    /// * `camera_params` - Fixed camera parameters [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    ///
    /// # Panics
    ///
    /// When the number of camera parameters is not 8 or the number of 2D points is zero.
    pub fn new(points_2d: Matrix2xX<f64>, camera_params: DVector<f64>) -> Self {
        assert_eq!(
            camera_params.len(),
            9,
            "Double Sphere model requires 6 camera parameters"
        );
        assert!(
            points_2d.ncols() > 0,
            "Number of 2D points must be greater than zero"
        );
        Self {
            points_2d,
            camera_params,
        }
    }

    pub fn num_points(&self) -> usize {
        self.points_2d.ncols()
    }
}

impl Factor for RadTanProjectionFactor {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        let num_points = self.num_points();
        let residual_dim = num_points * 2;
        let param_dim = num_points * 3;

        let mut residuals = DVector::zeros(residual_dim);
        let mut jacobian_matrix = if compute_jacobian {
            Some(DMatrix::zeros(residual_dim, param_dim))
        } else {
            None
        };

        let fx = self.camera_params[0];
        let fy = self.camera_params[1];
        let k1 = self.camera_params[4];
        let k2 = self.camera_params[5];
        let p1 = self.camera_params[6];
        let p2 = self.camera_params[7];
        let k3 = self.camera_params[8];

        for i in 0..num_points {
            let point_3d =
                Vector3::new(params[0][i * 3], params[0][i * 3 + 1], params[0][i * 3 + 2]);
            let point_2d = self.points_2d.column(i);

            let residual = compute_residual_rad_tan(point_3d, point_2d, &self.camera_params);

            if let Some(res) = residual {
                residuals[i * 2] = res[0];
                residuals[i * 2 + 1] = res[1];

                if let Some(ref mut jac_matrix) = jacobian_matrix {
                    // Compute d_proj_d_p3d (derivative of projection w.r.t. 3D point)
                    let x = point_3d[0];
                    let y = point_3d[1];
                    let z = point_3d[2];

                    // Normalized coordinates
                    let x_prime = x / z;
                    let y_prime = y / z;

                    let r2 = x_prime * x_prime + y_prime * y_prime;
                    let r4 = r2 * r2;
                    let r6 = r4 * r2;

                    // Radial distortion factor and derivative
                    let d = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
                    let d_d_r2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;

                    // Derivatives of r² w.r.t. normalized coords
                    let d_r2_dx_prime = 2.0 * x_prime;
                    let d_r2_dy_prime = 2.0 * y_prime;

                    // Derivatives of distorted coordinates w.r.t. normalized coords
                    // x_dist = d*x' + 2*p1*x'*y' + p2*(r² + 2*x'²)
                    let d_xdist_dx_prime = d
                        + x_prime * d_d_r2 * d_r2_dx_prime
                        + 2.0 * p1 * y_prime
                        + p2 * (d_r2_dx_prime + 4.0 * x_prime);
                    let d_xdist_dy_prime =
                        x_prime * d_d_r2 * d_r2_dy_prime + 2.0 * p1 * x_prime + p2 * d_r2_dy_prime;

                    // y_dist = d*y' + 2*p2*x'*y' + p1*(r² + 2*y'²)
                    let d_ydist_dx_prime =
                        y_prime * d_d_r2 * d_r2_dx_prime + 2.0 * p2 * y_prime + p1 * d_r2_dx_prime;
                    let d_ydist_dy_prime = d
                        + y_prime * d_d_r2 * d_r2_dy_prime
                        + 2.0 * p2 * x_prime
                        + p1 * (d_r2_dy_prime + 4.0 * y_prime);

                    // Derivatives of normalized coords w.r.t. 3D point
                    // x' = x/z, y' = y/z
                    let d_xprime_dx = 1.0 / z;
                    let d_xprime_dz = -x / (z * z);
                    let d_yprime_dy = 1.0 / z;
                    let d_yprime_dz = -y / (z * z);

                    // Chain rule: ∂u/∂[x,y,z] = fx * ∂x_dist/∂[x,y,z]
                    jac_matrix[(i * 2, i * 3)] = fx * d_xdist_dx_prime * d_xprime_dx;
                    jac_matrix[(i * 2, i * 3 + 1)] = fx * d_xdist_dy_prime * d_yprime_dy;
                    jac_matrix[(i * 2, i * 3 + 2)] =
                        fx * (d_xdist_dx_prime * d_xprime_dz + d_xdist_dy_prime * d_yprime_dz);

                    // Chain rule: ∂v/∂[x,y,z] = fy * ∂y_dist/∂[x,y,z]
                    jac_matrix[(i * 2 + 1, i * 3)] = fy * d_ydist_dx_prime * d_xprime_dx;
                    jac_matrix[(i * 2 + 1, i * 3 + 1)] = fy * d_ydist_dy_prime * d_yprime_dy;
                    jac_matrix[(i * 2 + 1, i * 3 + 2)] =
                        fy * (d_ydist_dx_prime * d_xprime_dz + d_ydist_dy_prime * d_yprime_dz);
                }
            } else {
                // Invalid projection - large residual
                residuals[i * 2] = 1e6;
                residuals[i * 2 + 1] = 1e6;
            }
        }

        (residuals, jacobian_matrix)
    }

    fn get_dimension(&self) -> usize {
        self.points_2d.ncols() * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_camera_params_factor_creation() {
        let points_3d_vec = vec![
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.1, 0.0, 1.0),
            Vector3::new(0.0, 0.1, 1.0),
        ];
        let points_2d_vec = vec![
            Vector2::new(362.0, 246.0),
            Vector2::new(392.0, 246.0),
            Vector2::new(362.0, 276.0),
        ];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = RadTanCameraParamsFactor::new(points_3d, points_2d);
        assert_eq!(factor.get_dimension(), 6);
    }

    #[test]
    fn test_projection_factor_creation() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(362.680, 246.049)];
        let _points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);
        let camera_params = DVector::from_vec(vec![
            461.629,
            460.152,
            362.680,
            246.049,
            -0.28340811,
            0.07395907,
            0.00019359,
            1.76187114e-05,
            0.0,
        ]);

        let factor = RadTanProjectionFactor::new(points_2d, camera_params);
        assert_eq!(factor.get_dimension(), 2);
    }

    #[test]
    fn test_camera_params_linearize_dimensions() -> TestResult {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.1, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(362.0, 246.0), Vector2::new(392.0, 246.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = RadTanCameraParamsFactor::new(points_3d, points_2d);
        let params = vec![DVector::from_vec(vec![
            461.629,
            460.152,
            362.680,
            246.049,
            -0.28340811,
            0.07395907,
            0.00019359,
            1.76187114e-05,
            0.0,
        ])];

        let (residual, jacobian) = factor.linearize(&params, true);

        assert_eq!(residual.len(), 4);
        assert!(jacobian.is_some());
        let jac = jacobian.ok_or("Expected jacobian to be Some")?;
        assert_eq!(jac.nrows(), 4);
        assert_eq!(jac.ncols(), 9);
        Ok(())
    }

    #[test]
    fn test_projection_linearize_dimensions() -> TestResult {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(362.680, 246.049)];
        let _points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);
        let camera_params = DVector::from_vec(vec![
            461.629,
            460.152,
            362.680,
            246.049,
            -0.28340811,
            0.07395907,
            0.00019359,
            1.76187114e-05,
            0.0,
        ]);

        let factor = RadTanProjectionFactor::new(points_2d, camera_params);
        let params = vec![DVector::from_vec(vec![0.0, 0.0, 1.0])];

        let (residual, jacobian) = factor.linearize(&params, true);

        assert_eq!(residual.len(), 2);
        assert!(jacobian.is_some());
        let jac = jacobian.ok_or("Expected jacobian to be Some")?;
        assert_eq!(jac.nrows(), 2);
        assert_eq!(jac.ncols(), 3);
        Ok(())
    }

    #[test]
    fn test_residual_computation() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(362.680, 246.049)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = RadTanCameraParamsFactor::new(points_3d, points_2d);
        let params = vec![DVector::from_vec(vec![
            461.629,
            460.152,
            362.680,
            246.049,
            -0.28340811,
            0.07395907,
            0.00019359,
            1.76187114e-05,
            0.0,
        ])];

        let (residual, _) = factor.linearize(&params, false);

        assert!(residual[0].abs() < 10.0);
        assert!(residual[1].abs() < 10.0);
    }

    #[test]
    fn test_jacobian_non_zero() -> TestResult {
        let points_3d_vec = vec![Vector3::new(0.1, 0.1, 1.0)];
        let points_2d_vec = vec![Vector2::new(370.0, 256.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = RadTanCameraParamsFactor::new(points_3d, points_2d);
        let params = vec![DVector::from_vec(vec![
            461.629,
            460.152,
            362.680,
            246.049,
            -0.28340811,
            0.07395907,
            0.00019359,
            1.76187114e-05,
            0.0,
        ])];

        let (_, jacobian) = factor.linearize(&params, true);

        assert!(jacobian.is_some());
        let jac = jacobian.ok_or("Expected jacobian to be Some")?;
        let has_nonzero = jac.iter().any(|&x| x.abs() > 1e-10);
        assert!(has_nonzero);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "Number of 3D and 2D points must match")]
    fn test_mismatched_points_panic() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(362.0, 246.0), Vector2::new(370.0, 256.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        RadTanCameraParamsFactor::new(points_3d, points_2d);
    }
}
