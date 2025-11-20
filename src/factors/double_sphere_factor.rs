//! Double Sphere camera model factors for apex-solver optimization.
//!
//! This module provides two factor implementations for the Double Sphere camera model:
//!
//! 1. [`DoubleSphereCameraParamsFactor`] - Optimizes camera intrinsic parameters
//! 2. [`DoubleSphereProjectionFactor`] - Optimizes 3D point positions or poses
//!
//! Both factors share the same residual computation but differ in their Jacobians.
//!
//! # Double Sphere Model
//!
//! Parameters: `[fx, fy, cx, cy, alpha, xi]`
//!
//! The Double Sphere model is suitable for wide-angle and fisheye cameras.

use super::Factor;
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX, RawStorage, U1, U2, U3, Vector2, Vector3};

const PRECISION: f64 = 1e-3;

// ============================================================================
// SHARED RESIDUAL COMPUTATION
// ============================================================================

/// Compute Double Sphere projection residual for a single point.
///
/// # Arguments
///
/// * `point_3d` - 3D point in camera coordinates
/// * `point_2d_obs` - Observed 2D point in image coordinates
/// * `camera_params` - Camera parameters [fx, fy, cx, cy, alpha, xi]
///
/// # Returns
///
/// Residual vector (2D) or None if projection is invalid
#[inline]
fn compute_residual_double_sphere<S3, S2>(
    point_3d: nalgebra::Matrix<f64, U3, U1, S3>,
    point_2d_obs: nalgebra::Matrix<f64, U2, U1, S2>,
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
    let xi = camera_params[5];
    let x = point_3d[0];
    let y = point_3d[1];
    let z = point_3d[2];

    let r_squared = x * x + y * y;
    let d1 = (r_squared + z * z).sqrt();
    let gamma = xi * d1 + z;
    let d2 = (r_squared + gamma * gamma).sqrt();
    let m_alpha = 1.0 - alpha;
    let denom = alpha * d2 + m_alpha * gamma;

    // Check projection validity
    let w1 = if alpha <= 0.5 {
        alpha / m_alpha
    } else {
        m_alpha / alpha
    };
    let w2 = (w1 + xi) / (2.0 * w1 * xi + xi * xi + 1.0).sqrt();
    let check_projection = z > -w2 * d1;

    if denom < PRECISION || !check_projection {
        return None;
    }

    let u_cx = point_2d_obs[0] - cx;
    let v_cy = point_2d_obs[1] - cy;

    Some(Vector2::new(fx * x - u_cx * denom, fy * y - v_cy * denom))
}

// ============================================================================
// FACTOR 1: CAMERA PARAMETERS FACTOR (Optimize intrinsics)
// ============================================================================

/// Camera parameters factor for Double Sphere model.
///
/// Optimizes camera intrinsic parameters: `[fx, fy, cx, cy, alpha, xi]`
///
/// # Use Case
/// Camera calibration where 3D points and 2D observations are known.
#[derive(Debug, Clone)]
pub struct DoubleSphereCameraParamsFactor {
    /// 3D points in camera coordinate system (3×N matrix)
    pub points_3d: Matrix3xX<f64>,
    /// Corresponding observed 2D points in image coordinates (2×N matrix)
    pub points_2d: Matrix2xX<f64>,
}

impl DoubleSphereCameraParamsFactor {
    /// Creates a new Double Sphere camera parameters factor.
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

impl Factor for DoubleSphereCameraParamsFactor {
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
            Some(DMatrix::zeros(residual_dim, 6))
        } else {
            None
        };

        for i in 0..num_points {
            let point_3d = self.points_3d.column(i);
            let point_2d = self.points_2d.column(i);

            match compute_residual_double_sphere(point_3d, point_2d, camera_params) {
                Some(point_residual) => {
                    residuals[i * 2] = point_residual[0];
                    residuals[i * 2 + 1] = point_residual[1];

                    if let Some(ref mut jac_matrix) = jacobian_matrix {
                        // Inline Jacobian computation w.r.t. camera parameters
                        let cx = camera_params[2];
                        let cy = camera_params[3];
                        let alpha = camera_params[4];
                        let xi = camera_params[5];
                        let x = point_3d[0];
                        let y = point_3d[1];
                        let z = point_3d[2];

                        let r_squared = x * x + y * y;
                        let d1 = (r_squared + z * z).sqrt();
                        let gamma = xi * d1 + z;
                        let d2 = (r_squared + gamma * gamma).sqrt();
                        let m_alpha = 1.0 - alpha;
                        let denom = alpha * d2 + m_alpha * gamma;

                        let u_cx = point_2d[0] - cx;
                        let v_cy = point_2d[1] - cy;

                        // ∂residual / ∂fx
                        jac_matrix[(i * 2, 0)] = x;
                        jac_matrix[(i * 2 + 1, 0)] = 0.0;

                        // ∂residual / ∂fy
                        jac_matrix[(i * 2, 1)] = 0.0;
                        jac_matrix[(i * 2 + 1, 1)] = y;

                        // ∂residual / ∂cx
                        jac_matrix[(i * 2, 2)] = denom;
                        jac_matrix[(i * 2 + 1, 2)] = 0.0;

                        // ∂residual / ∂cy
                        jac_matrix[(i * 2, 3)] = 0.0;
                        jac_matrix[(i * 2 + 1, 3)] = denom;

                        // ∂residual / ∂alpha
                        jac_matrix[(i * 2, 4)] = (gamma - d2) * u_cx;
                        jac_matrix[(i * 2 + 1, 4)] = (gamma - d2) * v_cy;

                        // ∂residual / ∂xi
                        let coeff = (alpha * d1 * gamma) / d2 + (m_alpha * d1);
                        jac_matrix[(i * 2, 5)] = -u_cx * coeff;
                        jac_matrix[(i * 2 + 1, 5)] = -v_cy * coeff;
                    }
                }
                None => {
                    residuals[i * 2] = 1e6;
                    residuals[i * 2 + 1] = 1e6;
                    if let Some(ref mut jac_matrix) = jacobian_matrix {
                        jac_matrix.view_mut((i * 2, 0), (2, 6)).fill(0.0);
                    }
                }
            }
        }

        (residuals, jacobian_matrix)
    }

    fn get_dimension(&self) -> usize {
        self.points_2d.ncols() * 2
    }
}

// ============================================================================
// FACTOR 2: PROJECTION FACTOR (Optimize 3D points)
// ============================================================================

/// Projection factor for Double Sphere model with fixed camera parameters.
///
/// Optimizes 3D point positions with known camera intrinsics.
///
/// # Use Case
/// Bundle adjustment, structure-from-motion where camera is calibrated.
#[derive(Debug, Clone)]
pub struct DoubleSphereProjectionFactor {
    /// Observed 2D points in image coordinates (2×N matrix)
    pub points_2d: Matrix2xX<f64>,
    /// Fixed camera parameters: [fx, fy, cx, cy, alpha, xi]
    pub camera_params: DVector<f64>,
}

impl DoubleSphereProjectionFactor {
    /// Creates a new Double Sphere projection factor.
    pub fn new(points_2d: Matrix2xX<f64>, camera_params: DVector<f64>) -> Self {
        assert_eq!(
            camera_params.len(),
            6,
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

impl Factor for DoubleSphereProjectionFactor {
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
        let alpha = self.camera_params[4];
        let xi = self.camera_params[5];

        for i in 0..num_points {
            let point_3d =
                Vector3::new(params[0][i * 3], params[0][i * 3 + 1], params[0][i * 3 + 2]);
            let point_2d = self.points_2d.column(i);

            let residual = compute_residual_double_sphere(point_3d, point_2d, &self.camera_params);

            if let Some(res) = residual {
                residuals[i * 2] = res[0];
                residuals[i * 2 + 1] = res[1];

                if let Some(ref mut jac_matrix) = jacobian_matrix {
                    // Inline Jacobian computation w.r.t. 3D point
                    let x = point_3d[0];
                    let y = point_3d[1];
                    let z = point_3d[2];

                    let r2 = x * x + y * y;
                    let d1 = (r2 + z * z).sqrt();
                    let k = xi * d1 + z;
                    let d2 = (r2 + k * k).sqrt();
                    let norm = alpha * d2 + (1.0 - alpha) * k;
                    let norm2 = norm * norm;

                    // Intermediate terms from granite
                    let d_norm_d_r2 =
                        (xi * (1.0 - alpha) / d1 + alpha * (xi * k / d1 + 1.0) / d2) / norm2;
                    let tt2 = xi * z / d1 + 1.0;
                    let tmp2 = ((1.0 - alpha) * tt2 + alpha * k * tt2 / d2) / norm2;

                    // ∂u/∂x, ∂u/∂y, ∂u/∂z
                    jac_matrix[(i * 2, i * 3)] = fx * (1.0 / norm - x * x * d_norm_d_r2);
                    jac_matrix[(i * 2, i * 3 + 1)] = -fx * x * y * d_norm_d_r2;
                    jac_matrix[(i * 2, i * 3 + 2)] = -fx * x * tmp2;

                    // ∂v/∂x, ∂v/∂y, ∂v/∂z
                    jac_matrix[(i * 2 + 1, i * 3)] = -fy * x * y * d_norm_d_r2;
                    jac_matrix[(i * 2 + 1, i * 3 + 1)] = fy * (1.0 / norm - y * y * d_norm_d_r2);
                    jac_matrix[(i * 2 + 1, i * 3 + 2)] = -fy * y * tmp2;
                }
            } else {
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

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_camera_params_factor() -> TestResult {
        let points_3d =
            Matrix3xX::from_columns(&[Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.1, 0.0, 1.0)]);
        let points_2d =
            Matrix2xX::from_columns(&[Vector2::new(320.0, 240.0), Vector2::new(350.0, 240.0)]);

        let factor = DoubleSphereCameraParamsFactor::new(points_3d, points_2d);
        let params = vec![DVector::from_vec(vec![
            300.0, 300.0, 320.0, 240.0, 0.5, 0.1,
        ])];

        let (residual, jacobian) = factor.linearize(&params, true);
        assert_eq!(residual.len(), 4);
        assert!(jacobian.is_some());
        assert_eq!(jacobian.ok_or("Expected jacobian to be Some")?.ncols(), 6);
        Ok(())
    }

    #[test]
    fn test_projection_factor() -> TestResult {
        let points_2d =
            Matrix2xX::from_columns(&[Vector2::new(320.0, 240.0), Vector2::new(350.0, 240.0)]);
        let camera_params = DVector::from_vec(vec![300.0, 300.0, 320.0, 240.0, 0.5, 0.1]);

        let factor = DoubleSphereProjectionFactor::new(points_2d, camera_params);
        let params = vec![DVector::from_vec(vec![0.0, 0.0, 1.0, 0.1, 0.0, 1.0])];

        let (residual, jacobian) = factor.linearize(&params, true);
        assert_eq!(residual.len(), 4);
        assert!(jacobian.is_some());
        assert_eq!(jacobian.ok_or("Expected jacobian to be Some")?.ncols(), 6); // 2 points × 3 coords
        Ok(())
    }

    #[test]
    fn test_residual_consistency() {
        // Both factors should compute same residuals
        let points_3d = Matrix3xX::from_columns(&[Vector3::new(0.1, 0.1, 1.0)]);
        let points_2d = Matrix2xX::from_columns(&[Vector2::new(330.0, 250.0)]);
        let camera_params_vec = vec![300.0, 300.0, 320.0, 240.0, 0.5, 0.1];

        // Camera params factor
        let factor1 = DoubleSphereCameraParamsFactor::new(points_3d.clone(), points_2d.clone());
        let (res1, _) = factor1.linearize(&[DVector::from_vec(camera_params_vec.clone())], false);

        // Projection factor
        let camera_params = DVector::from_vec(camera_params_vec);
        let factor2 = DoubleSphereProjectionFactor::new(points_2d, camera_params);
        let (res2, _) = factor2.linearize(&[DVector::from_vec(vec![0.1, 0.1, 1.0])], false);

        assert!((res1[0] - res2[0]).abs() < 1e-10);
        assert!((res1[1] - res2[1]).abs() < 1e-10);
    }
}
