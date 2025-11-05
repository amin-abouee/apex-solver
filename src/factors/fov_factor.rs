//! Field-of-View (FOV) camera projection factor for apex-solver optimization.
//!
//! This module provides a factor implementation for the apex-solver framework
//! that computes reprojection errors and analytical Jacobians for the Field-of-View
//! camera model. This allows using apex-solver's Levenberg-Marquardt optimizer
//! with hand-derived analytical derivatives.
//!
//! # References
//!
//! Implementation based on:
//! - Granite headers: https://github.com/DLR-RM/granite/blob/master/thirdparty/granite-headers/include/granite/camera/fov_camera.hpp
//! - "Simultaneous Localization and Mapping with Fisheye Cameras" by Zhang et al.

use super::Factor;
use nalgebra::{
    DMatrix, DVector, Matrix, Matrix2xX, Matrix3xX, RawStorage, SVector, U1, U2, U3, Vector2,
};

/// Projection factor for Field-of-View camera model optimization with apex-solver.
///
/// This factor computes the reprojection error between observed 2D points and
/// the projection of 3D points using the FOV camera model. It provides
/// analytical Jacobians for efficient optimization.
///
/// # Residual Formulation
///
/// For each 3D-2D point correspondence, the residual is computed as:
/// ```text
/// residual_x = fx * mx - (u - cx)
/// residual_y = fy * my - (v - cy)
/// ```
///
/// where `mx = x * rd`, `my = y * rd`, and `rd` is the FOV distortion factor.
///
/// # Parameters
///
/// The factor optimizes 5 camera parameters: `[fx, fy, cx, cy, w]`
#[derive(Debug, Clone)]
pub struct FovProjectionFactor {
    /// 3D points in camera coordinate system (3×N matrix)
    pub points_3d: Matrix3xX<f64>,
    /// Corresponding observed 2D points in image coordinates (2×N matrix)
    pub points_2d: Matrix2xX<f64>,
}

impl FovProjectionFactor {
    /// Creates a new FOV projection factor.
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

    /// Compute residual and analytical Jacobian for a single point.
    ///
    /// # Arguments
    ///
    /// * `point_3d` - 3D point in camera coordinates (column view)
    /// * `point_2d` - Observed 2D point (column view)
    /// * `params` - Camera parameters [fx, fy, cx, cy, w]
    /// * `compute_jacobian` - Whether to compute the Jacobian
    ///
    /// # Returns
    ///
    /// Tuple of (residual_vector, optional_jacobian_matrix)
    #[inline]
    fn compute_point_residual_jacobian<S3, S2>(
        point_3d: Matrix<f64, U3, U1, S3>,
        point_2d: Matrix<f64, U2, U1, S2>,
        params: &SVector<f64, 5>,
        compute_jacobian: bool,
    ) -> (Vector2<f64>, Option<nalgebra::Matrix2x5<f64>>)
    where
        S3: RawStorage<f64, U3, U1>,
        S2: RawStorage<f64, U2, U1>,
    {
        // Extract camera parameters
        let fx = params[0];
        let fy = params[1];
        let cx = params[2];
        let cy = params[3];
        let w = params[4];

        const EPS_SQRT: f64 = 1e-7; // sqrt(epsilon)

        let x = point_3d[0];
        let y = point_3d[1];
        let z = point_3d[2];

        // Check if z is valid (point in front of camera)
        if z < EPS_SQRT {
            // Invalid projection - return large residual
            let residual = Vector2::new(1e6, 1e6);
            let jacobian = if compute_jacobian {
                Some(nalgebra::Matrix2x5::zeros())
            } else {
                None
            };
            return (residual, jacobian);
        }

        let r2 = x * x + y * y;
        let r = r2.sqrt();

        let tan_w_half = (w / 2.0).tan();
        let atan_wrd = (2.0 * tan_w_half * r).atan2(z);

        // Compute distortion factor rd
        let rd = if r2 < EPS_SQRT {
            // Near optical axis - use Taylor expansion
            2.0 * tan_w_half / w
        } else {
            atan_wrd / (r * w)
        };

        let mx = x * rd;
        let my = y * rd;

        // Observed pixel coordinates
        let u = point_2d[0];
        let v = point_2d[1];

        // Compute residual
        let residual = Vector2::new(fx * mx - (u - cx), fy * my - (v - cy));

        // Compute analytical Jacobian if requested
        let jacobian = if compute_jacobian {
            let mut jac = nalgebra::Matrix2x5::zeros();

            // Compute derivative of rd with respect to w (d_rd_d_w)
            let d_rd_d_w = if r2 >= EPS_SQRT {
                // Standard case: r is not near zero
                // d_tanwhalf_d_w = 0.5 * sec²(w/2)
                let tmp1 = 1.0 / (w / 2.0).cos();
                let d_tanwhalf_d_w = 0.5 * tmp1 * tmp1;

                // tmp = z² + 4*tan²(w/2)*r²
                let tmp = z * z + 4.0 * tan_w_half * tan_w_half * r2;

                // d_atan_wrd_d_w = 2*r*d_tanwhalf_d_w*z / tmp
                let d_atan_wrd_d_w = 2.0 * r * d_tanwhalf_d_w * z / tmp;

                // d_rd_d_w = (d_atan_wrd_d_w * w - atan_wrd) / (r * w²)
                (d_atan_wrd_d_w * w - atan_wrd) / (r * w * w)
            } else {
                // Near optical axis case
                let tmp1 = 1.0 / (w / 2.0).cos();
                let d_tanwhalf_d_w = 0.5 * tmp1 * tmp1;
                2.0 * (d_tanwhalf_d_w * w - tan_w_half) / (w * w)
            };

            // ∂residual / ∂fx
            jac[(0, 0)] = mx;
            jac[(1, 0)] = 0.0;

            // ∂residual / ∂fy
            jac[(0, 1)] = 0.0;
            jac[(1, 1)] = my;

            // ∂residual / ∂cx
            jac[(0, 2)] = 1.0;
            jac[(1, 2)] = 0.0;

            // ∂residual / ∂cy
            jac[(0, 3)] = 0.0;
            jac[(1, 3)] = 1.0;

            // ∂residual / ∂w
            jac[(0, 4)] = fx * x * d_rd_d_w;
            jac[(1, 4)] = fy * y * d_rd_d_w;

            Some(jac)
        } else {
            None
        };

        (residual, jacobian)
    }
}

impl Factor for FovProjectionFactor {
    /// Compute residuals and Jacobians for all point correspondences.
    ///
    /// # Arguments
    ///
    /// * `params` - Slice containing one parameter vector [fx, fy, cx, cy, w]
    /// * `compute_jacobian` - Whether to compute the Jacobian matrix
    ///
    /// # Returns
    ///
    /// Tuple of (residual_vector, optional_jacobian_matrix)
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        assert_eq!(
            params.len(),
            1,
            "FOV factor expects exactly 1 parameter vector"
        );
        assert_eq!(
            params[0].len(),
            5,
            "FOV factor expects 5 parameters [fx, fy, cx, cy, w]"
        );

        let num_points = self.points_3d.ncols();
        let residual_dim = num_points * 2;

        // Convert parameter vector to fixed-size vector for efficiency
        let param_vec = SVector::<f64, 5>::from_column_slice(params[0].as_slice());

        // Allocate residual vector
        let mut residuals = DVector::zeros(residual_dim);

        // Allocate Jacobian matrix if needed
        let mut jacobian_matrix = if compute_jacobian {
            Some(DMatrix::zeros(residual_dim, 5))
        } else {
            None
        };

        // Process each point
        for i in 0..num_points {
            let point_3d = self.points_3d.column(i);
            let point_2d = self.points_2d.column(i);

            let (residual, jacobian) = Self::compute_point_residual_jacobian(
                point_3d,
                point_2d,
                &param_vec,
                compute_jacobian,
            );

            // Store residual
            residuals[i * 2] = residual[0];
            residuals[i * 2 + 1] = residual[1];

            // Store Jacobian if computed
            if let (Some(jac_matrix), Some(jac)) = (&mut jacobian_matrix, jacobian) {
                jac_matrix.view_mut((i * 2, 0), (2, 5)).copy_from(&jac);
            }
        }

        (residuals, jacobian_matrix)
    }

    /// Get the dimension of the residual vector.
    ///
    /// # Returns
    ///
    /// Number of residuals = 2 * number of points
    fn get_dimension(&self) -> usize {
        self.points_3d.ncols() * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Vector2, Vector3};

    #[test]
    fn test_fov_factor_creation() {
        let points_3d = Matrix3xX::from_columns(&[
            Vector3::new(0.1, 0.1, 1.0),
            Vector3::new(0.2, 0.2, 1.0),
            Vector3::new(0.3, 0.3, 1.0),
        ]);
        let points_2d = Matrix2xX::from_columns(&[
            Vector2::new(400.0, 300.0),
            Vector2::new(410.0, 310.0),
            Vector2::new(420.0, 320.0),
        ]);

        let factor = FovProjectionFactor::new(points_3d, points_2d);
        assert_eq!(factor.get_dimension(), 6); // 3 points * 2 = 6 residuals
    }

    #[test]
    fn test_fov_factor_linearize() {
        let points_3d =
            Matrix3xX::from_columns(&[Vector3::new(0.1, 0.1, 1.0), Vector3::new(0.2, 0.2, 1.0)]);
        let points_2d =
            Matrix2xX::from_columns(&[Vector2::new(400.0, 300.0), Vector2::new(410.0, 310.0)]);

        let factor = FovProjectionFactor::new(points_3d, points_2d);

        // Test parameters: [fx, fy, cx, cy, w]
        let params = vec![DVector::from_vec(vec![400.0, 400.0, 376.0, 240.0, 1.0])];

        // Test without Jacobian
        let (residual, jac) = factor.linearize(&params, false);
        assert_eq!(residual.len(), 4); // 2 points * 2
        assert!(jac.is_none());

        // Test with Jacobian
        let (residual, jac) = factor.linearize(&params, true);
        assert_eq!(residual.len(), 4);
        assert!(jac.is_some());
        let jac_matrix = jac.unwrap();
        assert_eq!(jac_matrix.nrows(), 4); // 2 points * 2
        assert_eq!(jac_matrix.ncols(), 5); // 5 parameters
    }

    #[test]
    #[should_panic(expected = "Number of 3D and 2D points must match")]
    fn test_fov_factor_mismatched_points() {
        let points_3d = Matrix3xX::from_columns(&[
            Vector3::new(0.1, 0.1, 1.0),
            Vector3::new(0.2, 0.2, 1.0),
            Vector3::new(0.3, 0.3, 1.0),
        ]);
        let points_2d =
            Matrix2xX::from_columns(&[Vector2::new(400.0, 300.0), Vector2::new(410.0, 310.0)]);

        // Should panic: 3 3D points but only 2 2D points
        FovProjectionFactor::new(points_3d, points_2d);
    }
}
