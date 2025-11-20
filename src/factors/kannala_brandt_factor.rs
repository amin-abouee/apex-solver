//! Kannala-Brandt projection factors for apex-solver optimization.
//!
//! This module provides two factor implementations for the apex-solver framework:
//! 1. `KannalaBrandtCameraParamsFactor`: Optimizes camera intrinsics (fx, fy, cx, cy, k1, k2, k3, k4)
//! 2. `KannalaBrandtProjectionFactor`: Optimizes 3D point positions
//!
//! Both factors share the same residual computation but differ in their Jacobians.
//!
//! # References
//!
//! Implementation based on:
//! - https://github.com/DLR-RM/granite/blob/master/thirdparty/granite-headers/include/granite/camera/kannala_brandt_camera4.hpp

use super::Factor;
use nalgebra::{
    DMatrix, DVector, Matrix, Matrix2xX, Matrix3xX, RawStorage, U1, U2, U3, Vector2, Vector3,
};

/// Compute residual for Kannala-Brandt fisheye camera model.
///
/// # Residual Formulation
///
/// For each 3D-2D point correspondence, the residual is computed as:
/// ```text
/// theta = atan2(r, z) where r = sqrt(x² + y²)
/// theta_d = theta + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
/// residual_x = fx * theta_d * (x/r) + cx - u_observed
/// residual_y = fy * theta_d * (y/r) + cy - v_observed
/// ```
///
/// # Arguments
///
/// * `point_3d` - 3D point in camera coordinates [x, y, z]
/// * `point_2d` - Observed 2D point in image coordinates [u, v]
/// * `camera_params` - Camera parameters [fx, fy, cx, cy, k1, k2, k3, k4]
///
/// # Returns
///
/// `Some(residual)` if projection is valid, `None` otherwise
#[inline]
fn compute_residual_kannala_brandt<S3, S2>(
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
    let k3 = camera_params[6];
    let k4 = camera_params[7];
    let x = point_3d[0];
    let y = point_3d[1];
    let z = point_3d[2];

    // Check for invalid projections
    if z < f64::EPSILON {
        return None;
    }

    let r_squared = x * x + y * y;
    let r = r_squared.sqrt();
    let theta = r.atan2(z);

    // Compute polynomial terms
    let theta2 = theta * theta;
    let theta3 = theta2 * theta;
    let theta5 = theta3 * theta2;
    let theta7 = theta5 * theta2;
    let theta9 = theta7 * theta2;

    // Distorted angle
    let theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

    // Normalized coordinates
    let (x_r, y_r) = if r < f64::EPSILON {
        (0.0, 0.0)
    } else {
        (x / r, y / r)
    };

    // Projected point
    let projected_x = fx * theta_d * x_r + cx;
    let projected_y = fy * theta_d * y_r + cy;

    // Residual: projected - observed
    Some(Vector2::new(
        projected_x - point_2d[0],
        projected_y - point_2d[1],
    ))
}

/// Factor for optimizing Kannala-Brandt camera intrinsic parameters.
///
/// This factor computes the reprojection error and provides Jacobians
/// with respect to camera parameters: [fx, fy, cx, cy, k1, k2, k3, k4].
///
/// # Parameters
///
/// The factor optimizes 8 camera parameters:
/// - fx, fy: Focal lengths
/// - cx, cy: Principal point
/// - k1, k2, k3, k4: Distortion coefficients
#[derive(Debug, Clone)]
pub struct KannalaBrandtCameraParamsFactor {
    /// 3D points in camera coordinate system
    pub points_3d: Matrix3xX<f64>,
    /// Corresponding observed 2D points in image coordinates
    pub points_2d: Matrix2xX<f64>,
}

impl KannalaBrandtCameraParamsFactor {
    /// Creates a new Kannala-Brandt camera parameters factor.
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

impl Factor for KannalaBrandtCameraParamsFactor {
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
            Some(DMatrix::zeros(residual_dim, 8))
        } else {
            None
        };

        for i in 0..num_points {
            let point_3d = self.points_3d.column(i);
            let point_2d = self.points_2d.column(i);

            let residual = compute_residual_kannala_brandt(point_3d, point_2d, camera_params);

            if let Some(res) = residual {
                residuals[i * 2] = res[0];
                residuals[i * 2 + 1] = res[1];

                if let Some(ref mut jac_matrix) = jacobian_matrix {
                    // Extract camera parameters
                    let fx = camera_params[0];
                    let fy = camera_params[1];
                    let k1 = camera_params[4];
                    let k2 = camera_params[5];
                    let k3 = camera_params[6];
                    let k4 = camera_params[7];

                    // Extract 3D coordinates
                    let x = point_3d[0];
                    let y = point_3d[1];
                    let z = point_3d[2];

                    let r_squared = x * x + y * y;
                    let r = r_squared.sqrt();
                    let theta = r.atan2(z);

                    let theta2 = theta * theta;
                    let theta3 = theta2 * theta;
                    let theta5 = theta3 * theta2;
                    let theta7 = theta5 * theta2;
                    let theta9 = theta7 * theta2;

                    let theta_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

                    let (x_r, y_r) = if r < f64::EPSILON {
                        (0.0, 0.0)
                    } else {
                        (x / r, y / r)
                    };

                    // Jacobian w.r.t. camera parameters (inline computation)
                    // ∂u/∂fx = theta_d * x_r
                    jac_matrix[(i * 2, 0)] = theta_d * x_r;
                    jac_matrix[(i * 2 + 1, 0)] = 0.0;

                    // ∂v/∂fy = theta_d * y_r
                    jac_matrix[(i * 2, 1)] = 0.0;
                    jac_matrix[(i * 2 + 1, 1)] = theta_d * y_r;

                    // ∂u/∂cx = 1, ∂v/∂cx = 0
                    jac_matrix[(i * 2, 2)] = 1.0;
                    jac_matrix[(i * 2 + 1, 2)] = 0.0;

                    // ∂u/∂cy = 0, ∂v/∂cy = 1
                    jac_matrix[(i * 2, 3)] = 0.0;
                    jac_matrix[(i * 2 + 1, 3)] = 1.0;

                    // Derivatives w.r.t. distortion coefficients
                    // ∂theta_d/∂k1 = θ³, ∂u/∂k1 = fx * θ³ * x_r
                    jac_matrix[(i * 2, 4)] = fx * theta3 * x_r;
                    jac_matrix[(i * 2 + 1, 4)] = fy * theta3 * y_r;

                    // ∂theta_d/∂k2 = θ⁵, ∂u/∂k2 = fx * θ⁵ * x_r
                    jac_matrix[(i * 2, 5)] = fx * theta5 * x_r;
                    jac_matrix[(i * 2 + 1, 5)] = fy * theta5 * y_r;

                    // ∂theta_d/∂k3 = θ⁷, ∂u/∂k3 = fx * θ⁷ * x_r
                    jac_matrix[(i * 2, 6)] = fx * theta7 * x_r;
                    jac_matrix[(i * 2 + 1, 6)] = fy * theta7 * y_r;

                    // ∂theta_d/∂k4 = θ⁹, ∂u/∂k4 = fx * θ⁹ * x_r
                    jac_matrix[(i * 2, 7)] = fx * theta9 * x_r;
                    jac_matrix[(i * 2 + 1, 7)] = fy * theta9 * y_r;
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

/// Factor for optimizing 3D point positions with fixed Kannala-Brandt camera parameters.
///
/// This factor computes the reprojection error and provides Jacobians
/// with respect to 3D point positions. Camera parameters are held constant.
///
/// The Jacobian computation is based on granite-headers implementation:
/// https://github.com/DLR-RM/granite/blob/master/thirdparty/granite-headers/include/granite/camera/kannala_brandt_camera4.hpp
#[derive(Debug, Clone)]
pub struct KannalaBrandtProjectionFactor {
    /// Corresponding observed 2D points in image coordinates
    pub points_2d: Matrix2xX<f64>,
    /// Fixed camera parameters: [fx, fy, cx, cy, k1, k2, k3, k4]
    pub camera_params: DVector<f64>,
}

impl KannalaBrandtProjectionFactor {
    /// Creates a new Kannala-Brandt projection factor.
    ///
    /// # Arguments
    ///
    /// * `points_2d` - Matrix of corresponding 2D observed points (2×N)
    /// * `camera_params` - Fixed camera parameters [fx, fy, cx, cy, k1, k2, k3, k4]
    ///
    /// # Panics
    ///
    /// When the number of camera parameters is not 8 or the number of 2D points is zero.
    pub fn new(points_2d: Matrix2xX<f64>, camera_params: DVector<f64>) -> Self {
        assert_eq!(
            camera_params.len(),
            8,
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

impl Factor for KannalaBrandtProjectionFactor {
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
        let k3 = self.camera_params[6];
        let k4 = self.camera_params[7];

        for i in 0..num_points {
            let point_3d =
                Vector3::new(params[0][i * 3], params[0][i * 3 + 1], params[0][i * 3 + 2]);
            let point_2d = self.points_2d.column(i);

            let residual = compute_residual_kannala_brandt(point_3d, point_2d, &self.camera_params);

            if let Some(res) = residual {
                residuals[i * 2] = res[0];
                residuals[i * 2 + 1] = res[1];

                if let Some(ref mut jac_matrix) = jacobian_matrix {
                    // Inline Jacobian computation w.r.t. 3D point
                    let x = point_3d[0];
                    let y = point_3d[1];
                    let z = point_3d[2];

                    let r_squared = x * x + y * y;
                    let r = r_squared.sqrt();

                    if r < f64::EPSILON {
                        // Degenerate case - use pinhole approximation
                        jac_matrix[(i * 2, i * 3)] = fx / z;
                        jac_matrix[(i * 2, i * 3 + 1)] = 0.0;
                        jac_matrix[(i * 2, i * 3 + 2)] = -fx * x / (z * z);

                        jac_matrix[(i * 2 + 1, i * 3)] = 0.0;
                        jac_matrix[(i * 2 + 1, i * 3 + 1)] = fy / z;
                        jac_matrix[(i * 2 + 1, i * 3 + 2)] = -fy * y / (z * z);
                    } else {
                        let theta = r.atan2(z);
                        let theta2 = theta * theta;

                        // Compute distorted radius
                        let theta3 = theta2 * theta;
                        let theta5 = theta3 * theta2;
                        let theta7 = theta5 * theta2;
                        let theta9 = theta7 * theta2;
                        let r_theta = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

                        // Compute d_r_theta_d_theta = 1 + 3k1θ² + 5k2θ⁴ + 7k3θ⁶ + 9k4θ⁸
                        let mut d_r_theta_d_theta = 9.0 * k4 * theta2;
                        d_r_theta_d_theta += 7.0 * k3;
                        d_r_theta_d_theta *= theta2;
                        d_r_theta_d_theta += 5.0 * k2;
                        d_r_theta_d_theta *= theta2;
                        d_r_theta_d_theta += 3.0 * k1;
                        d_r_theta_d_theta *= theta2;
                        d_r_theta_d_theta += 1.0;

                        // Derivatives of r and theta w.r.t. x, y, z
                        let d_r_d_x = x / r;
                        let d_r_d_y = y / r;

                        let tmp = z * z + r_squared;
                        let d_theta_d_x = d_r_d_x * z / tmp;
                        let d_theta_d_y = d_r_d_y * z / tmp;
                        let d_theta_d_z = -r / tmp;

                        // Jacobian entries from granite-headers formula
                        jac_matrix[(i * 2, i * 3)] = fx
                            * (r_theta * r + x * r * d_r_theta_d_theta * d_theta_d_x
                                - x * x * r_theta / r)
                            / r_squared;
                        jac_matrix[(i * 2 + 1, i * 3)] =
                            fy * y * (d_r_theta_d_theta * d_theta_d_x * r - x * r_theta / r)
                                / r_squared;

                        jac_matrix[(i * 2, i * 3 + 1)] =
                            fx * x * (d_r_theta_d_theta * d_theta_d_y * r - y * r_theta / r)
                                / r_squared;
                        jac_matrix[(i * 2 + 1, i * 3 + 1)] = fy
                            * (r_theta * r + y * r * d_r_theta_d_theta * d_theta_d_y
                                - y * y * r_theta / r)
                            / r_squared;

                        jac_matrix[(i * 2, i * 3 + 2)] =
                            fx * x * d_r_theta_d_theta * d_theta_d_z / r;
                        jac_matrix[(i * 2 + 1, i * 3 + 2)] =
                            fy * y * d_r_theta_d_theta * d_theta_d_z / r;
                    }
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
            Vector2::new(320.0, 240.0),
            Vector2::new(350.0, 240.0),
            Vector2::new(320.0, 270.0),
        ];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = KannalaBrandtCameraParamsFactor::new(points_3d, points_2d);
        assert_eq!(factor.get_dimension(), 6);
    }

    #[test]
    fn test_projection_factor_creation() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0)];
        let _points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);
        let camera_params =
            DVector::from_vec(vec![460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04]);

        let factor = KannalaBrandtProjectionFactor::new(points_2d, camera_params);
        assert_eq!(factor.get_dimension(), 2);
    }

    #[test]
    fn test_camera_params_linearize_dimensions() -> TestResult {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.1, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0), Vector2::new(350.0, 240.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = KannalaBrandtCameraParamsFactor::new(points_3d, points_2d);
        let params = vec![DVector::from_vec(vec![
            460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04,
        ])];

        let (residual, jacobian) = factor.linearize(&params, true);

        assert_eq!(residual.len(), 4);
        assert!(jacobian.is_some());
        let jac = jacobian.ok_or("Expected jacobian to be Some")?;
        assert_eq!(jac.nrows(), 4);
        assert_eq!(jac.ncols(), 8);
        Ok(())
    }

    #[test]
    fn test_projection_linearize_dimensions() -> TestResult {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0)];
        let _points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);
        let camera_params =
            DVector::from_vec(vec![460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04]);

        let factor = KannalaBrandtProjectionFactor::new(points_2d, camera_params);
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
        let points_2d_vec = vec![Vector2::new(320.0, 240.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = KannalaBrandtCameraParamsFactor::new(points_3d, points_2d);
        let params = vec![DVector::from_vec(vec![
            460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04,
        ])];

        let (residual, _) = factor.linearize(&params, false);

        assert!(residual[0].abs() < 1.0);
        assert!(residual[1].abs() < 1.0);
    }

    #[test]
    fn test_jacobian_non_zero() -> TestResult {
        let points_3d_vec = vec![Vector3::new(0.1, 0.1, 1.0)];
        let points_2d_vec = vec![Vector2::new(330.0, 250.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = KannalaBrandtCameraParamsFactor::new(points_3d, points_2d);
        let params = vec![DVector::from_vec(vec![
            460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04,
        ])];

        let (_, jacobian) = factor.linearize(&params, true);

        assert!(jacobian.is_some());
        let jac = jacobian.ok_or("Expected jacobian to be Some")?;
        let has_nonzero = jac.iter().any(|&x| x.abs() > 1e-10);
        assert!(has_nonzero);
        Ok(())
    }

    #[test]
    fn test_jacobian_structure() -> TestResult {
        let points_3d_vec = vec![Vector3::new(0.2, 0.15, 1.0)];
        let points_2d_vec = vec![Vector2::new(380.0, 280.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = KannalaBrandtCameraParamsFactor::new(points_3d, points_2d);
        let params = vec![DVector::from_vec(vec![
            460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04,
        ])];

        let (_, jacobian) = factor.linearize(&params, true);

        let jac = jacobian.ok_or("Expected jacobian to be Some")?;

        // Check specific Jacobian structure
        // ∂u/∂fy should be 0
        assert!(jac[(0, 1)].abs() < 1e-12);
        // ∂v/∂fx should be 0
        assert!(jac[(1, 0)].abs() < 1e-12);
        // ∂u/∂cx should be 1
        assert!((jac[(0, 2)] - 1.0).abs() < 1e-12);
        // ∂v/∂cy should be 1
        assert!((jac[(1, 3)] - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "Number of 3D and 2D points must match")]
    fn test_mismatched_points_panic() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0), Vector2::new(330.0, 250.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        KannalaBrandtCameraParamsFactor::new(points_3d, points_2d);
    }
}
