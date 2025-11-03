//! Unified Camera Model (UCM) projection factor for apex-solver optimization.
//!
//! This module provides a factor implementation for the apex-solver framework
//! that computes reprojection errors and analytical Jacobians for the Unified
//! Camera Model (UCM). This allows using apex-solver's Levenberg-Marquardt optimizer
//! with hand-derived analytical derivatives.
//!
//! # References
//!
//! Implementation based on:
//! - https://github.com/eowjd0512/fisheye-calib-adapter/blob/main/include/model/UCM.hpp
//! - "A Unifying Theory for Central Panoramic Systems" by Geyer and Daniilidis

use super::Factor;
use nalgebra::{
    DMatrix, DVector, Matrix, Matrix2xX, Matrix3xX, RawStorage, SVector, U1, U2, U3, Vector2,
};

/// Projection factor for Unified Camera Model (UCM) optimization with apex-solver.
///
/// This factor computes the reprojection error between observed 2D points and
/// the projection of 3D points using the UCM camera model. It provides
/// analytical Jacobians for efficient optimization.
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
/// # Parameters
///
/// The factor optimizes 5 camera parameters: `[fx, fy, cx, cy, alpha]`
#[derive(Debug, Clone)]
pub struct UcmProjectionFactor {
    /// 3D points in camera coordinate system
    pub points_3d: Matrix3xX<f64>,
    /// Corresponding observed 2D points in image coordinates
    pub points_2d: Matrix2xX<f64>,
}

impl UcmProjectionFactor {
    /// Creates a new UCM projection factor.
    ///
    /// # Arguments
    ///
    /// * `points_3d` - Vector of 3D points in camera coordinates
    /// * `points_2d` - Vector of corresponding 2D observed points
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
    /// * `params` - Camera parameters [fx, fy, cx, cy, alpha]
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
        let alpha = params[4];
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
            // Invalid projection - return large residual
            let residual = Vector2::new(1e6, 1e6);
            let jacobian = if compute_jacobian {
                Some(nalgebra::Matrix2x5::zeros())
            } else {
                None
            };
            return (residual, jacobian);
        }

        // Compute residual using formulation: fx * x - (u - cx) * denom
        let u_cx = point_2d[0] - cx;
        let v_cy = point_2d[1] - cy;

        let residual = Vector2::new(fx * x - u_cx * denom, fy * y - v_cy * denom);

        // Compute analytical Jacobian if requested
        let jacobian = if compute_jacobian {
            let mut jac = nalgebra::Matrix2x5::zeros();

            // ∂residual / ∂fx
            // From residual_x = fx * x - (u - cx) * denom
            // ∂residual_x / ∂fx = x
            jac[(0, 0)] = x;
            jac[(1, 0)] = 0.0;

            // ∂residual / ∂fy
            // From residual_y = fy * y - (v - cy) * denom
            // ∂residual_y / ∂fy = y
            jac[(0, 1)] = 0.0;
            jac[(1, 1)] = y;

            // ∂residual / ∂cx
            // ∂residual_x / ∂cx = ∂[fx * x - (u - cx) * denom] / ∂cx
            //                    = -∂(u - cx) / ∂cx * denom
            //                    = -(-1) * denom = denom
            jac[(0, 2)] = denom;
            jac[(1, 2)] = 0.0;

            // ∂residual / ∂cy
            // ∂residual_y / ∂cy = ∂[fy * y - (v - cy) * denom] / ∂cy
            //                    = -∂(v - cy) / ∂cy * denom
            //                    = -(-1) * denom = denom
            jac[(0, 3)] = 0.0;
            jac[(1, 3)] = denom;

            // ∂residual / ∂alpha
            // From denom = alpha * d + (1 - alpha) * z
            // ∂denom / ∂alpha = d - z
            //
            // ∂residual_x / ∂alpha = -∂[(u - cx) * denom] / ∂alpha
            //                       = -(u - cx) * ∂denom / ∂alpha
            //                       = -(u - cx) * (d - z)
            // ∂residual_y / ∂alpha = -(v - cy) * (d - z)
            let ddenom_dalpha = d - z;
            jac[(0, 4)] = -u_cx * ddenom_dalpha;
            jac[(1, 4)] = -v_cy * ddenom_dalpha;

            Some(jac)
        } else {
            None
        };

        (residual, jacobian)
    }
}

impl Factor for UcmProjectionFactor {
    /// Compute residuals and analytical Jacobians for all point correspondences.
    ///
    /// # Arguments
    ///
    /// * `params` - Slice containing camera parameters as a single DVector:
    ///   `params[0] = [fx, fy, cx, cy, alpha]`
    /// * `compute_jacobian` - Whether to compute the Jacobian matrix
    ///
    /// # Returns
    ///
    /// Tuple of (residual_vector, optional_jacobian_matrix) where:
    /// - `residual_vector` has dimension `2 * num_points`
    /// - `jacobian_matrix` has dimension `(2 * num_points) × 5`
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // Extract camera parameters into SVector
        let cam_params = &params[0];
        let camera_params = SVector::<f64, 5>::from_row_slice(&[
            cam_params[0], // fx
            cam_params[1], // fy
            cam_params[2], // cx
            cam_params[3], // cy
            cam_params[4], // alpha
        ]);

        let num_points = self.points_2d.ncols();
        let residual_dim = num_points * 2;

        // Initialize residual vector
        let mut residuals = DVector::zeros(residual_dim);

        // Initialize Jacobian if needed
        let mut jacobian_matrix = if compute_jacobian {
            Some(DMatrix::zeros(residual_dim, 5))
        } else {
            None
        };

        // Process each point
        for i in 0..num_points {
            let (point_residual, point_jacobian) = Self::compute_point_residual_jacobian(
                self.points_3d.column(i),
                self.points_2d.column(i),
                &camera_params,
                compute_jacobian,
            );

            // Fill residual vector
            residuals[i * 2] = point_residual[0];
            residuals[i * 2 + 1] = point_residual[1];

            // Fill Jacobian matrix if computed
            if let (Some(ref mut jac_matrix), Some(point_jac)) =
                (jacobian_matrix.as_mut(), point_jacobian)
            {
                jac_matrix
                    .view_mut((i * 2, 0), (2, 5))
                    .copy_from(&point_jac);
            }
        }

        (residuals, jacobian_matrix)
    }

    /// Returns the dimension of the residual vector.
    ///
    /// For N point correspondences, the residual dimension is 2N
    /// (2 residuals per point: x and y).
    fn get_dimension(&self) -> usize {
        self.points_2d.ncols() * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_factor_creation() {
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

        let factor = UcmProjectionFactor::new(points_3d, points_2d);
        assert_eq!(factor.get_dimension(), 6); // 3 points × 2 residuals
    }

    #[test]
    fn test_linearize_dimensions() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.1, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0), Vector2::new(350.0, 240.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = UcmProjectionFactor::new(points_3d, points_2d);

        // Camera parameters: [fx, fy, cx, cy, alpha]
        let params = vec![DVector::from_vec(vec![300.0, 300.0, 320.0, 240.0, 0.5])];

        let (residual, jacobian) = factor.linearize(&params, true);

        assert_eq!(residual.len(), 4); // 2 points × 2 residuals
        assert!(jacobian.is_some());
        let jac = jacobian.unwrap();
        assert_eq!(jac.nrows(), 4); // 2 points × 2 residuals
        assert_eq!(jac.ncols(), 5); // 5 camera parameters
    }

    #[test]
    fn test_residual_computation() {
        // Test with a simple case where 3D point at (0,0,1) should project to (cx,cy)
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = UcmProjectionFactor::new(points_3d, points_2d);

        // Parameters with perfect projection at center
        let params = vec![DVector::from_vec(vec![300.0, 300.0, 320.0, 240.0, 0.5])];

        let (residual, _) = factor.linearize(&params, false);

        // For point at (0,0,1) with the given parameters, residual should be small
        assert!(residual[0].abs() < 1.0);
        assert!(residual[1].abs() < 1.0);
    }

    #[test]
    fn test_jacobian_non_zero() {
        let points_3d_vec = vec![Vector3::new(0.1, 0.1, 1.0)];
        let points_2d_vec = vec![Vector2::new(330.0, 250.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = UcmProjectionFactor::new(points_3d, points_2d);

        let params = vec![DVector::from_vec(vec![300.0, 300.0, 320.0, 240.0, 0.5])];

        let (_, jacobian) = factor.linearize(&params, true);

        assert!(jacobian.is_some());
        let jac = jacobian.unwrap();

        // Check that Jacobian has non-zero entries
        let has_nonzero = jac.iter().any(|&x| x.abs() > 1e-10);
        assert!(has_nonzero, "Jacobian should have non-zero entries");
    }

    #[test]
    #[should_panic(expected = "Number of 3D and 2D points must match")]
    fn test_mismatched_points_panic() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0), Vector2::new(330.0, 250.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        UcmProjectionFactor::new(points_3d, points_2d);
    }

    #[test]
    fn test_edge_case_projection() {
        // Test projection with a point that has negative Z (behind camera in standard coordinates)
        // UCM can still compute a mathematically valid projection for such points
        let points_3d_vec = vec![Vector3::new(0.1, 0.1, -1.0)];
        let points_2d_vec = vec![Vector2::new(330.0, 250.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = UcmProjectionFactor::new(points_3d, points_2d);

        let params = vec![DVector::from_vec(vec![300.0, 300.0, 320.0, 240.0, 0.5])];

        let (residual, _) = factor.linearize(&params, false);

        // UCM model computes a projection even for negative Z
        // The residual will depend on how well the projection matches the observed 2D point
        assert_eq!(residual.len(), 2);
    }
}
