//! Field-of-View (FOV) camera model factors for apex-solver optimization.
//!
//! This module provides two factor implementations for the FOV camera model:
//!
//! 1. [`FovCameraParamsFactor`] - Optimizes camera intrinsic parameters
//! 2. [`FovProjectionFactor`] - Optimizes 3D point positions or poses
//!
//! Both factors share the same residual computation but differ in their Jacobians.
//!
//! # FOV Model
//!
//! Parameters: `[fx, fy, cx, cy, w]`
//!
//! The FOV model is suitable for fisheye cameras with radial distortion.

use super::Factor;
use nalgebra::{DMatrix, DVector, Matrix2xX, Matrix3xX, RawStorage, U1, U2, U3, Vector2, Vector3};

const EPS_SQRT: f64 = 1e-7;

// ============================================================================
// SHARED RESIDUAL COMPUTATION
// ============================================================================

/// Compute FOV projection residual for a single point.
#[inline]
fn compute_residual_fov<S3, S2>(
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
    let w = camera_params[4];
    let x = point_3d[0];
    let y = point_3d[1];
    let z = point_3d[2];

    if z < EPS_SQRT {
        return None;
    }

    let r2 = x * x + y * y;
    let r = r2.sqrt();

    let tan_w_half = (w / 2.0).tan();
    let atan_wrd = (2.0 * tan_w_half * r).atan2(z);

    let rd = if r2 < EPS_SQRT {
        2.0 * tan_w_half / w
    } else {
        atan_wrd / (r * w)
    };

    let mx = x * rd;
    let my = y * rd;

    let u = fx * mx + cx;
    let v = fy * my + cy;

    Some(Vector2::new(u - point_2d_obs[0], v - point_2d_obs[1]))
}

// ============================================================================
// FACTOR 1: CAMERA PARAMETERS FACTOR
// ============================================================================

#[derive(Debug, Clone)]
pub struct FovCameraParamsFactor {
    pub points_3d: Matrix3xX<f64>,
    pub points_2d: Matrix2xX<f64>,
}

impl FovCameraParamsFactor {
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

impl Factor for FovCameraParamsFactor {
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

            match compute_residual_fov(point_3d, point_2d, camera_params) {
                Some(point_residual) => {
                    residuals[i * 2] = point_residual[0];
                    residuals[i * 2 + 1] = point_residual[1];

                    if let Some(ref mut jac_matrix) = jacobian_matrix {
                        // Inline Jacobian computation w.r.t. camera parameters
                        let fx = camera_params[0];
                        let fy = camera_params[1];
                        let w = camera_params[4];
                        let x = point_3d[0];
                        let y = point_3d[1];
                        let z = point_3d[2];

                        let r2 = x * x + y * y;
                        let r = r2.sqrt();

                        let tan_w_half = (w / 2.0).tan();
                        let atan_wrd = (2.0 * tan_w_half * r).atan2(z);

                        let rd = if r2 >= EPS_SQRT {
                            atan_wrd / (r * w)
                        } else {
                            2.0 * tan_w_half / w
                        };

                        let d_rd_d_w = if r2 >= EPS_SQRT {
                            let tmp1 = 1.0 / (w / 2.0).cos();
                            let d_tanwhalf_d_w = 0.5 * tmp1 * tmp1;
                            let tmp = z * z + 4.0 * tan_w_half * tan_w_half * r2;
                            let d_atan_wrd_d_w = 2.0 * r * d_tanwhalf_d_w * z / tmp;
                            (d_atan_wrd_d_w * w - atan_wrd) / (r * w * w)
                        } else {
                            let tmp1 = 1.0 / (w / 2.0).cos();
                            let d_tanwhalf_d_w = 0.5 * tmp1 * tmp1;
                            2.0 * (d_tanwhalf_d_w * w - tan_w_half) / (w * w)
                        };

                        let mx = x * rd;
                        let my = y * rd;

                        jac_matrix[(i * 2, 0)] = mx;
                        jac_matrix[(i * 2 + 1, 0)] = 0.0;

                        jac_matrix[(i * 2, 1)] = 0.0;
                        jac_matrix[(i * 2 + 1, 1)] = my;

                        jac_matrix[(i * 2, 2)] = 1.0;
                        jac_matrix[(i * 2 + 1, 2)] = 0.0;

                        jac_matrix[(i * 2, 3)] = 0.0;
                        jac_matrix[(i * 2 + 1, 3)] = 1.0;

                        jac_matrix[(i * 2, 4)] = fx * x * d_rd_d_w;
                        jac_matrix[(i * 2 + 1, 4)] = fy * y * d_rd_d_w;
                    }
                }
                None => {
                    residuals[i * 2] = 1e6;
                    residuals[i * 2 + 1] = 1e6;
                    if let Some(ref mut jac_matrix) = jacobian_matrix {
                        jac_matrix.view_mut((i * 2, 0), (2, 5)).fill(0.0);
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
pub struct FovProjectionFactor {
    pub points_2d: Matrix2xX<f64>,
    pub camera_params: DVector<f64>,
}

impl FovProjectionFactor {
    pub fn new(points_2d: Matrix2xX<f64>, camera_params: DVector<f64>) -> Self {
        assert_eq!(
            camera_params.len(),
            5,
            "FOV model requires 5 camera parameters"
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

impl Factor for FovProjectionFactor {
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
        let w = self.camera_params[4];

        for i in 0..num_points {
            let point_3d =
                Vector3::new(params[0][i * 3], params[0][i * 3 + 1], params[0][i * 3 + 2]);
            let point_2d = self.points_2d.column(i);

            match compute_residual_fov(point_3d, point_2d, &self.camera_params) {
                Some(point_residual) => {
                    residuals[i * 2] = point_residual[0];
                    residuals[i * 2 + 1] = point_residual[1];

                    if let Some(ref mut jac_matrix) = jacobian_matrix {
                        // Inline Jacobian computation w.r.t. 3D point (from granite)
                        let x = point_3d[0];
                        let y = point_3d[1];
                        let z = point_3d[2];

                        let r2 = x * x + y * y;
                        let r = r2.sqrt();

                        let tan_w_half = (w / 2.0).tan();
                        let two_tan_w_half = 2.0 * tan_w_half;

                        let rd = if r2 >= EPS_SQRT {
                            let atan_wrd = (two_tan_w_half * r).atan2(z);
                            atan_wrd / (r * w)
                        } else {
                            two_tan_w_half / w
                        };

                        let (d_rd_d_x, d_rd_d_y, d_rd_d_z) = if r2 >= EPS_SQRT {
                            let denom_atan = z * z + 4.0 * tan_w_half * tan_w_half * r2;

                            let d_rd_d_x =
                                (two_tan_w_half * x * z / denom_atan - rd * x / r) / (r * w);
                            let d_rd_d_y =
                                (two_tan_w_half * y * z / denom_atan - rd * y / r) / (r * w);
                            let d_rd_d_z = -two_tan_w_half * r / (w * denom_atan);

                            (d_rd_d_x, d_rd_d_y, d_rd_d_z)
                        } else {
                            (0.0, 0.0, 0.0)
                        };

                        jac_matrix[(i * 2, i * 3)] = fx * (d_rd_d_x * x + rd);
                        jac_matrix[(i * 2, i * 3 + 1)] = fx * d_rd_d_y * x;
                        jac_matrix[(i * 2, i * 3 + 2)] = fx * d_rd_d_z * x;

                        jac_matrix[(i * 2 + 1, i * 3)] = fy * d_rd_d_x * y;
                        jac_matrix[(i * 2 + 1, i * 3 + 1)] = fy * (d_rd_d_y * y + rd);
                        jac_matrix[(i * 2 + 1, i * 3 + 2)] = fy * d_rd_d_z * y;
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

    #[test]
    fn test_camera_params_factor() {
        let points_3d =
            Matrix3xX::from_columns(&[Vector3::new(0.1, 0.1, 1.0), Vector3::new(0.2, 0.2, 1.0)]);
        let points_2d =
            Matrix2xX::from_columns(&[Vector2::new(400.0, 300.0), Vector2::new(410.0, 310.0)]);

        let factor = FovCameraParamsFactor::new(points_3d, points_2d);
        let params = vec![DVector::from_vec(vec![400.0, 400.0, 376.0, 240.0, 1.0])];

        let (residual, jacobian) = factor.linearize(&params, true);
        assert_eq!(residual.len(), 4);
        assert!(jacobian.is_some());
        assert_eq!(jacobian.unwrap().ncols(), 5);
    }

    #[test]
    fn test_projection_factor() {
        let points_2d =
            Matrix2xX::from_columns(&[Vector2::new(400.0, 300.0), Vector2::new(410.0, 310.0)]);
        let camera_params = DVector::from_vec(vec![400.0, 400.0, 376.0, 240.0, 1.0]);

        let factor = FovProjectionFactor::new(points_2d, camera_params);
        let params = vec![DVector::from_vec(vec![0.1, 0.1, 1.0, 0.2, 0.2, 1.0])];

        let (residual, jacobian) = factor.linearize(&params, true);
        assert_eq!(residual.len(), 4);
        assert!(jacobian.is_some());
        assert_eq!(jacobian.unwrap().ncols(), 6);
    }

    #[test]
    fn test_residual_consistency() {
        let points_3d = Matrix3xX::from_columns(&[Vector3::new(0.1, 0.1, 1.0)]);
        let points_2d = Matrix2xX::from_columns(&[Vector2::new(410.0, 310.0)]);
        let camera_params_vec = vec![400.0, 400.0, 376.0, 240.0, 1.0];

        let factor1 = FovCameraParamsFactor::new(points_3d.clone(), points_2d.clone());
        let (res1, _) = factor1.linearize(&[DVector::from_vec(camera_params_vec.clone())], false);

        let camera_params = DVector::from_vec(camera_params_vec);
        let factor2 = FovProjectionFactor::new(points_2d, camera_params);
        let (res2, _) = factor2.linearize(&[DVector::from_vec(vec![0.1, 0.1, 1.0])], false);

        assert!((res1[0] - res2[0]).abs() < 1e-10);
        assert!((res1[1] - res2[1]).abs() < 1e-10);
    }
}
