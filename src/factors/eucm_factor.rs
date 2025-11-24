//! Extended Unified Camera Model (EUCM) factors for apex-solver optimization.
//!
//! This module provides two factor implementations for the EUCM camera model:
//!
//! 1. [`EucmCameraParamsFactor`] - Optimizes camera intrinsic parameters
//! 2. [`EucmProjectionFactor`] - Optimizes 3D point positions or poses
//!
//! Both factors share the same residual computation but differ in their Jacobians.
//!
//! # EUCM Model
//!
//! Parameters: `[fx, fy, cx, cy, alpha, beta]`
//!
//! The Extended Unified Camera Model is suitable for fisheye and wide-angle cameras.

use super::Factor;
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX, RawStorage, U1, U2, U3, Vector2, Vector3};

const PRECISION: f64 = 1e-3;

// ============================================================================
// SHARED RESIDUAL COMPUTATION
// ============================================================================

/// Compute EUCM projection residual for a single point.
///
/// # Arguments
///
/// * `point_3d` - 3D point in camera coordinates
/// * `point_2d_obs` - Observed 2D point in image coordinates
/// * `camera_params` - Camera parameters [fx, fy, cx, cy, alpha, beta]
#[inline]
fn compute_residual_eucm<S3, S2>(
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
    let beta = camera_params[5];
    let x = point_3d[0];
    let y = point_3d[1];
    let z = point_3d[2];

    let r_squared = x * x + y * y;
    let d = (beta * r_squared + z * z).sqrt();
    let denom = alpha * d + (1.0 - alpha) * z;

    // Check projection validity
    if alpha > 0.5 {
        let c = (alpha - 1.0) / (2.0 * alpha - 1.0);
        if z < denom * c {
            return None;
        }
    }

    if denom < PRECISION {
        return None;
    }

    let u_cx = point_2d_obs[0] - cx;
    let v_cy = point_2d_obs[1] - cy;

    Some(Vector2::new(fx * x - u_cx * denom, fy * y - v_cy * denom))
}

// ============================================================================
// FACTOR 1: CAMERA PARAMETERS FACTOR
// ============================================================================

#[derive(Debug, Clone)]
pub struct EucmCameraParamsFactor {
    pub points_3d: Matrix3xX<f64>,
    pub points_2d: Matrix2xX<f64>,
}

impl EucmCameraParamsFactor {
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

impl Factor for EucmCameraParamsFactor {
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

            match compute_residual_eucm(point_3d, point_2d, camera_params) {
                Some(point_residual) => {
                    residuals[i * 2] = point_residual[0];
                    residuals[i * 2 + 1] = point_residual[1];

                    if let Some(ref mut jac_matrix) = jacobian_matrix {
                        // Inline Jacobian computation w.r.t. camera parameters
                        let cx = camera_params[2];
                        let cy = camera_params[3];
                        let alpha = camera_params[4];
                        let beta = camera_params[5];
                        let x = point_3d[0];
                        let y = point_3d[1];
                        let z = point_3d[2];

                        let r_squared = x * x + y * y;
                        let d = (beta * r_squared + z * z).sqrt();
                        let denom = alpha * d + (1.0 - alpha) * z;

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
                        jac_matrix[(i * 2, 4)] = (z - d) * u_cx;
                        jac_matrix[(i * 2 + 1, 4)] = (z - d) * v_cy;

                        // ∂residual / ∂beta
                        if d > PRECISION {
                            jac_matrix[(i * 2, 5)] = -(alpha * r_squared * u_cx) / (2.0 * d);
                            jac_matrix[(i * 2 + 1, 5)] = -(alpha * r_squared * v_cy) / (2.0 * d);
                        } else {
                            jac_matrix[(i * 2, 5)] = 0.0;
                            jac_matrix[(i * 2 + 1, 5)] = 0.0;
                        }
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
// FACTOR 2: PROJECTION FACTOR
// ============================================================================

#[derive(Debug, Clone)]
pub struct EucmProjectionFactor {
    pub points_2d: Matrix2xX<f64>,
    pub camera_params: DVector<f64>,
}

impl EucmProjectionFactor {
    pub fn new(points_2d: Matrix2xX<f64>, camera_params: DVector<f64>) -> Self {
        assert_eq!(
            camera_params.len(),
            6,
            "EUCM model requires 6 camera parameters"
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

impl Factor for EucmProjectionFactor {
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
        let beta = self.camera_params[5];

        for i in 0..num_points {
            let point_3d =
                Vector3::new(params[0][i * 3], params[0][i * 3 + 1], params[0][i * 3 + 2]);
            let point_2d = self.points_2d.column(i);

            match compute_residual_eucm(point_3d, point_2d, &self.camera_params) {
                Some(point_residual) => {
                    residuals[i * 2] = point_residual[0];
                    residuals[i * 2 + 1] = point_residual[1];

                    if let Some(ref mut jac_matrix) = jacobian_matrix {
                        // Inline Jacobian computation w.r.t. 3D point (from granite)
                        let x = point_3d[0];
                        let y = point_3d[1];
                        let z = point_3d[2];

                        let r2 = x * x + y * y;
                        let rho = (beta * r2 + z * z).sqrt();
                        let norm = alpha * rho + (1.0 - alpha) * z;
                        let denom = norm * norm * rho;

                        let mid = -(alpha * beta * x * y);
                        let add = norm * rho;
                        let addz = alpha * z + (1.0 - alpha) * rho;

                        jac_matrix[(i * 2, i * 3)] = fx * (add - x * x * alpha * beta) / denom;
                        jac_matrix[(i * 2, i * 3 + 1)] = fx * mid / denom;
                        jac_matrix[(i * 2, i * 3 + 2)] = -fx * x * addz / denom;

                        jac_matrix[(i * 2 + 1, i * 3)] = fy * mid / denom;
                        jac_matrix[(i * 2 + 1, i * 3 + 1)] =
                            fy * (add - y * y * alpha * beta) / denom;
                        jac_matrix[(i * 2 + 1, i * 3 + 2)] = -fy * y * addz / denom;
                    }
                }
                None => {
                    residuals[i * 2] = 1e6;
                    residuals[i * 2 + 1] = 1e6;
                    if let Some(ref mut jac_matrix) = jacobian_matrix {
                        jac_matrix.view_mut((i * 2, i * 3), (2, 3)).fill(0.0);
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

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_camera_params_factor() -> TestResult {
        let points_3d =
            Matrix3xX::from_columns(&[Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.1, 0.0, 1.0)]);
        let points_2d =
            Matrix2xX::from_columns(&[Vector2::new(960.0, 546.0), Vector2::new(990.0, 546.0)]);

        let factor = EucmCameraParamsFactor::new(points_3d, points_2d);
        let params = vec![DVector::from_vec(vec![
            1313.83, 1313.27, 960.471, 546.981, 1.01674, 0.5,
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
            Matrix2xX::from_columns(&[Vector2::new(960.0, 546.0), Vector2::new(990.0, 546.0)]);
        let camera_params =
            DVector::from_vec(vec![1313.83, 1313.27, 960.471, 546.981, 1.01674, 0.5]);

        let factor = EucmProjectionFactor::new(points_2d, camera_params);
        let params = vec![DVector::from_vec(vec![0.0, 0.0, 1.0, 0.1, 0.0, 1.0])];

        let (residual, jacobian) = factor.linearize(&params, true);
        assert_eq!(residual.len(), 4);
        assert!(jacobian.is_some());
        assert_eq!(jacobian.ok_or("Expected jacobian to be Some")?.ncols(), 6);
        Ok(())
    }

    #[test]
    fn test_residual_consistency() {
        let points_3d = Matrix3xX::from_columns(&[Vector3::new(0.1, 0.1, 1.0)]);
        let points_2d = Matrix2xX::from_columns(&[Vector2::new(970.0, 556.0)]);
        let camera_params_vec = vec![1313.83, 1313.27, 960.471, 546.981, 1.01674, 0.5];

        let factor1 = EucmCameraParamsFactor::new(points_3d.clone(), points_2d.clone());
        let (res1, _) = factor1.linearize(&[DVector::from_vec(camera_params_vec.clone())], false);

        let camera_params = DVector::from_vec(camera_params_vec);
        let factor2 = EucmProjectionFactor::new(points_2d, camera_params);
        let (res2, _) = factor2.linearize(&[DVector::from_vec(vec![0.1, 0.1, 1.0])], false);

        assert!((res1[0] - res2[0]).abs() < 1e-10);
        assert!((res1[1] - res2[1]).abs() < 1e-10);
    }
}
