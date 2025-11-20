//! Unified Camera Model (UCM) projection factors for apex-solver optimization.
//!
//! This module provides two factor implementations for the apex-solver framework:
//! 1. `UcmCameraParamsFactor`: Optimizes camera intrinsics (fx, fy, cx, cy, alpha)
//! 2. `UcmProjectionFactor`: Optimizes 3D point positions
//!
//! Both factors share the same residual computation but differ in their Jacobians.
//!
//! # References
//!
//! Implementation based on:
//! - https://github.com/eowjd0512/fisheye-calib-adapter/blob/main/include/model/UCM.hpp
//! - https://github.com/DLR-RM/granite/blob/master/thirdparty/granite-headers/include/granite/camera/unified_camera.hpp
//! - "A Unifying Theory for Central Panoramic Systems" by Geyer and Daniilidis

use super::Factor;
use nalgebra::{
    DMatrix, DVector, Matrix, Matrix2xX, Matrix3xX, RawStorage, U1, U2, U3, Vector2, Vector3,
};

/// Compute residual for UCM camera model.
///
/// # Residual Formulation
///
/// For each 3D-2D point correspondence, the residual is computed as:
/// ```text
/// residual_x = fx * x - (u - cx) * denom
/// residual_y = fy * y - (v - cy) * denom
/// ```
///
/// where `denom = alpha * d + (1 - alpha) * z` from the UCM model,
/// and `d = sqrt(x² + y² + z²)`.
///
/// # Arguments
///
/// * `point_3d` - 3D point in camera coordinates [x, y, z]
/// * `point_2d` - Observed 2D point in image coordinates [u, v]
/// * `camera_params` - Camera parameters [fx, fy, cx, cy, alpha]
///
/// # Returns
///
/// `Some(residual)` if projection is valid, `None` otherwise
#[inline]
fn compute_residual_ucm<S3, S2>(
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
    let alpha = camera_params[4];
    const PRECISION: f64 = 1e-3;

    let x = point_3d[0];
    let y = point_3d[1];
    let z = point_3d[2];

    // Compute distance from origin
    let d = (x * x + y * y + z * z).sqrt();

    // UCM projection denominator: alpha * d + (1 - alpha) * z
    let denom = alpha * d + (1.0 - alpha) * z;

    // Check projection validity
    let w = if alpha <= 0.5 {
        alpha / (1.0 - alpha)
    } else {
        (1.0 - alpha) / alpha
    };
    let check_projection = z > -w * d;

    if denom < PRECISION || !check_projection {
        return None;
    }

    // Compute residual using formulation: fx * x - (u - cx) * denom
    let u_cx = point_2d[0] - cx;
    let v_cy = point_2d[1] - cy;

    Some(Vector2::new(fx * x - u_cx * denom, fy * y - v_cy * denom))
}

/// Factor for optimizing UCM camera intrinsic parameters.
///
/// This factor computes the reprojection error and provides Jacobians
/// with respect to camera parameters: [fx, fy, cx, cy, alpha].
///
/// # Parameters
///
/// The factor optimizes 5 camera parameters:
/// - fx, fy: Focal lengths
/// - cx, cy: Principal point
/// - alpha: UCM distortion parameter
#[derive(Debug, Clone)]
pub struct UcmCameraParamsFactor {
    /// 3D points in camera coordinate system
    pub points_3d: Matrix3xX<f64>,
    /// Corresponding observed 2D points in image coordinates
    pub points_2d: Matrix2xX<f64>,
}

impl UcmCameraParamsFactor {
    /// Creates a new UCM camera parameters factor.
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

impl Factor for UcmCameraParamsFactor {
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
            Some(DMatrix::zeros(residual_dim, 5))
        } else {
            None
        };

        for i in 0..num_points {
            let point_3d = self.points_3d.column(i);
            let point_2d = self.points_2d.column(i);

            let residual = compute_residual_ucm(point_3d, point_2d, camera_params);

            if let Some(res) = residual {
                residuals[i * 2] = res[0];
                residuals[i * 2 + 1] = res[1];

                if let Some(ref mut jac_matrix) = jacobian_matrix {
                    // Extract camera parameters
                    let cx = camera_params[2];
                    let cy = camera_params[3];
                    let alpha = camera_params[4];

                    // Extract 3D coordinates
                    let x = point_3d[0];
                    let y = point_3d[1];
                    let z = point_3d[2];

                    // Compute UCM denominator
                    let d = (x * x + y * y + z * z).sqrt();
                    let denom = alpha * d + (1.0 - alpha) * z;

                    // Observation residuals
                    let u_cx = point_2d[0] - cx;
                    let v_cy = point_2d[1] - cy;

                    // Jacobian w.r.t. camera parameters (inline computation)
                    // ∂residual / ∂fx = [x, 0]^T
                    jac_matrix[(i * 2, 0)] = x;
                    jac_matrix[(i * 2 + 1, 0)] = 0.0;

                    // ∂residual / ∂fy = [0, y]^T
                    jac_matrix[(i * 2, 1)] = 0.0;
                    jac_matrix[(i * 2 + 1, 1)] = y;

                    // ∂residual / ∂cx = [denom, 0]^T
                    jac_matrix[(i * 2, 2)] = denom;
                    jac_matrix[(i * 2 + 1, 2)] = 0.0;

                    // ∂residual / ∂cy = [0, denom]^T
                    jac_matrix[(i * 2, 3)] = 0.0;
                    jac_matrix[(i * 2 + 1, 3)] = denom;

                    // ∂residual / ∂alpha
                    // ∂denom / ∂alpha = d - z
                    let ddenom_dalpha = d - z;
                    jac_matrix[(i * 2, 4)] = -u_cx * ddenom_dalpha;
                    jac_matrix[(i * 2 + 1, 4)] = -v_cy * ddenom_dalpha;
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

/// Factor for optimizing 3D point positions with fixed UCM camera parameters.
///
/// This factor computes the reprojection error and provides Jacobians
/// with respect to 3D point positions. Camera parameters are held constant.
///
#[derive(Debug, Clone)]
pub struct UcmProjectionFactor {
    /// Corresponding observed 2D points in image coordinates
    pub points_2d: Matrix2xX<f64>,
    /// Fixed camera parameters: [fx, fy, cx, cy, alpha]
    pub camera_params: DVector<f64>,
}

impl UcmProjectionFactor {
    /// Creates a new UCM projection factor.
    ///
    /// # Arguments
    ///
    /// * `points_2d` - Matrix of corresponding 2D observed points (2×N)
    /// * `camera_params` - Fixed camera parameters [fx, fy, cx, cy, alpha]
    ///
    /// # Panics
    ///
    /// When the number of camera parameters is not 8 or the number of 2D points is zero.
    pub fn new(points_2d: Matrix2xX<f64>, camera_params: DVector<f64>) -> Self {
        assert_eq!(
            camera_params.len(),
            5,
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

impl Factor for UcmProjectionFactor {
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
        let cx = self.camera_params[2];
        let cy = self.camera_params[3];
        let alpha = self.camera_params[4];

        for i in 0..num_points {
            let point_3d =
                Vector3::new(params[0][i * 3], params[0][i * 3 + 1], params[0][i * 3 + 2]);
            let point_2d = self.points_2d.column(i);

            let residual = compute_residual_ucm(point_3d, point_2d, &self.camera_params);

            if let Some(res) = residual {
                residuals[i * 2] = res[0];
                residuals[i * 2 + 1] = res[1];

                if let Some(ref mut jac_matrix) = jacobian_matrix {
                    // Compute d_proj_d_p3d (derivative of projection w.r.t. 3D point)
                    let x = point_3d[0];
                    let y = point_3d[1];
                    let z = point_3d[2];

                    let rho = (x * x + y * y + z * z).sqrt();

                    // Jacobian of residual w.r.t. 3D point (inline computation)
                    // residual = [fx * x - (u - cx) * denom, fy * y - (v - cy) * denom]
                    // We need ∂residual / ∂[x, y, z]

                    // From granite: projection π(x,y,z) = [fx·x/(αρ+(1-α)z) + cx, fy·y/(αρ+(1-α)z) + cy]
                    // d_proj_d_p3d gives ∂π / ∂[x,y,z]

                    // Since residual = [fx·x - (u-cx)·denom, fy·y - (v-cy)·denom]
                    // We need to apply chain rule carefully

                    // Let's compute d_proj_d_p3d from granite formulation

                    // For residual formulation: r = [fx·x - (u-cx)·D, fy·y - (v-cy)·D]
                    // where D = denom from UCM
                    // We need ∂r / ∂[x,y,z]

                    // ∂r_x / ∂x = fx - (u-cx)·∂D/∂x
                    // ∂r_x / ∂y = -(u-cx)·∂D/∂y
                    // ∂r_x / ∂z = -(u-cx)·∂D/∂z

                    let u_cx = point_2d[0] - cx;
                    let v_cy = point_2d[1] - cy;

                    // ∂D/∂x where D = α·ρ + (1-α)·z and ρ = sqrt(x²+y²+z²)
                    let d_denom_dx = alpha * x / rho;
                    let d_denom_dy = alpha * y / rho;
                    let d_denom_dz = alpha * z / rho + (1.0 - alpha);

                    jac_matrix[(i * 2, i * 3)] = fx - u_cx * d_denom_dx;
                    jac_matrix[(i * 2, i * 3 + 1)] = -u_cx * d_denom_dy;
                    jac_matrix[(i * 2, i * 3 + 2)] = -u_cx * d_denom_dz;

                    jac_matrix[(i * 2 + 1, i * 3)] = -v_cy * d_denom_dx;
                    jac_matrix[(i * 2 + 1, i * 3 + 1)] = fy - v_cy * d_denom_dy;
                    jac_matrix[(i * 2 + 1, i * 3 + 2)] = -v_cy * d_denom_dz;
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

        let factor = UcmCameraParamsFactor::new(points_3d, points_2d);
        assert_eq!(factor.get_dimension(), 6);
    }

    #[test]
    fn test_projection_factor_creation() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0)];
        let _points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);
        let camera_params = DVector::from_vec(vec![300.0, 300.0, 320.0, 240.0, 0.5]);

        let factor = UcmProjectionFactor::new(points_2d, camera_params);
        assert_eq!(factor.get_dimension(), 2);
    }

    #[test]
    fn test_camera_params_linearize_dimensions() -> TestResult {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.1, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0), Vector2::new(350.0, 240.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = UcmCameraParamsFactor::new(points_3d, points_2d);
        let params = vec![DVector::from_vec(vec![300.0, 300.0, 320.0, 240.0, 0.5])];

        let (residual, jacobian) = factor.linearize(&params, true);

        assert_eq!(residual.len(), 4);
        assert!(jacobian.is_some());
        let jac = jacobian.ok_or("Expected jacobian to be Some")?;
        assert_eq!(jac.nrows(), 4);
        assert_eq!(jac.ncols(), 5);
        Ok(())
    }

    #[test]
    fn test_projection_linearize_dimensions() -> TestResult {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0)];
        let _points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);
        let camera_params = DVector::from_vec(vec![300.0, 300.0, 320.0, 240.0, 0.5]);

        let factor = UcmProjectionFactor::new(points_2d, camera_params);
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

        let factor = UcmCameraParamsFactor::new(points_3d, points_2d);
        let params = vec![DVector::from_vec(vec![300.0, 300.0, 320.0, 240.0, 0.5])];

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

        let factor = UcmCameraParamsFactor::new(points_3d, points_2d);
        let params = vec![DVector::from_vec(vec![300.0, 300.0, 320.0, 240.0, 0.5])];

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
        let points_2d_vec = vec![Vector2::new(320.0, 240.0), Vector2::new(330.0, 250.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        UcmCameraParamsFactor::new(points_3d, points_2d);
    }
}
