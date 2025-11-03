//! Kannala-Brandt projection factor for apex-solver optimization.
//!
//! This module provides a factor implementation for the apex-solver framework
//! that computes reprojection errors and analytical Jacobians for the Kannala-Brandt
//! fisheye camera model. This allows using apex-solver's Levenberg-Marquardt optimizer
//! with hand-derived analytical derivatives.

use super::Factor;
use nalgebra::{
    DMatrix, DVector, Matrix, Matrix2xX, Matrix3xX, RawStorage, SVector, U1, U2, U3, Vector2,
};

/// Projection factor for Kannala-Brandt camera model optimization with apex-solver.
///
/// This factor computes the reprojection error between observed 2D points and
/// the projection of 3D points using the Kannala-Brandt fisheye camera model.
/// It provides analytical Jacobians for efficient optimization.
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
/// # Parameters
///
/// The factor optimizes 8 camera parameters: `[fx, fy, cx, cy, k1, k2, k3, k4]`
#[derive(Debug, Clone)]
pub struct KannalaBrandtProjectionFactor {
    /// 3D points in camera coordinate system
    pub points_3d: Matrix3xX<f64>,
    /// Corresponding observed 2D points in image coordinates
    pub points_2d: Matrix2xX<f64>,
}

impl KannalaBrandtProjectionFactor {
    /// Creates a new Kannala-Brandt projection factor.
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
    /// * `params` - Camera parameters [fx, fy, cx, cy, k1, k2, k3, k4]
    /// * `compute_jacobian` - Whether to compute the Jacobian
    ///
    /// # Returns
    ///
    /// Tuple of (residual_vector, optional_jacobian_matrix)
    #[inline]
    fn compute_point_residual_jacobian<S3, S2>(
        point_3d: Matrix<f64, U3, U1, S3>,
        point_2d: Matrix<f64, U2, U1, S2>,
        params: &SVector<f64, 8>,
        compute_jacobian: bool,
    ) -> (Vector2<f64>, Option<nalgebra::SMatrix<f64, 2, 8>>)
    where
        S3: RawStorage<f64, U3, U1>,
        S2: RawStorage<f64, U2, U1>,
    {
        // Extract camera parameters
        let fx = params[0];
        let fy = params[1];
        let cx = params[2];
        let cy = params[3];
        let k1 = params[4];
        let k2 = params[5];
        let k3 = params[6];
        let k4 = params[7];
        let x = point_3d[0];
        let y = point_3d[1];
        let z = point_3d[2];

        // Check for invalid projections
        if z < f64::EPSILON {
            // Point behind or at camera center
            let residual = Vector2::new(1e6, 1e6);
            let jacobian = if compute_jacobian {
                Some(nalgebra::SMatrix::<f64, 2, 8>::zeros())
            } else {
                None
            };
            return (residual, jacobian);
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
        let residual = Vector2::new(projected_x - point_2d[0], projected_y - point_2d[1]);

        // Compute analytical Jacobian if requested
        let jacobian = if compute_jacobian {
            let mut jac = nalgebra::SMatrix::<f64, 2, 8>::zeros();

            // ∂u/∂fx = theta_d * x_r
            jac[(0, 0)] = theta_d * x_r;
            jac[(1, 0)] = 0.0;

            // ∂v/∂fy = theta_d * y_r
            jac[(0, 1)] = 0.0;
            jac[(1, 1)] = theta_d * y_r;

            // ∂u/∂cx = 1, ∂v/∂cx = 0
            jac[(0, 2)] = 1.0;
            jac[(1, 2)] = 0.0;

            // ∂u/∂cy = 0, ∂v/∂cy = 1
            jac[(0, 3)] = 0.0;
            jac[(1, 3)] = 1.0;

            // Derivatives w.r.t. distortion coefficients
            // ∂theta_d/∂k1 = θ³
            // ∂u/∂k1 = fx * θ³ * x_r
            jac[(0, 4)] = fx * theta3 * x_r;
            jac[(1, 4)] = fy * theta3 * y_r;

            // ∂theta_d/∂k2 = θ⁵
            // ∂u/∂k2 = fx * θ⁵ * x_r
            jac[(0, 5)] = fx * theta5 * x_r;
            jac[(1, 5)] = fy * theta5 * y_r;

            // ∂theta_d/∂k3 = θ⁷
            // ∂u/∂k3 = fx * θ⁷ * x_r
            jac[(0, 6)] = fx * theta7 * x_r;
            jac[(1, 6)] = fy * theta7 * y_r;

            // ∂theta_d/∂k4 = θ⁹
            // ∂u/∂k4 = fx * θ⁹ * x_r
            jac[(0, 7)] = fx * theta9 * x_r;
            jac[(1, 7)] = fy * theta9 * y_r;

            Some(jac)
        } else {
            None
        };

        (residual, jacobian)
    }
}

impl Factor for KannalaBrandtProjectionFactor {
    /// Compute residuals and analytical Jacobians for all point correspondences.
    ///
    /// # Arguments
    ///
    /// * `params` - Slice containing camera parameters as a single DVector:
    ///   `params[0] = [fx, fy, cx, cy, k1, k2, k3, k4]`
    /// * `compute_jacobian` - Whether to compute the Jacobian matrix
    ///
    /// # Returns
    ///
    /// Tuple of (residual_vector, optional_jacobian_matrix) where:
    /// - `residual_vector` has dimension `2 * num_points`
    /// - `jacobian_matrix` has dimension `(2 * num_points) × 8`
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // Extract camera parameters into SVector
        let cam_params = &params[0];
        let camera_params = SVector::<f64, 8>::from_row_slice(&[
            cam_params[0], // fx
            cam_params[1], // fy
            cam_params[2], // cx
            cam_params[3], // cy
            cam_params[4], // k1
            cam_params[5], // k2
            cam_params[6], // k3
            cam_params[7], // k4
        ]);

        let num_points = self.points_2d.ncols();
        let residual_dim = num_points * 2;

        // Initialize residual vector
        let mut residuals = DVector::zeros(residual_dim);

        // Initialize Jacobian if needed
        let mut jacobian_matrix = if compute_jacobian {
            Some(DMatrix::zeros(residual_dim, 8))
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
                    .view_mut((i * 2, 0), (2, 8))
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

        let factor = KannalaBrandtProjectionFactor::new(points_3d, points_2d);
        assert_eq!(factor.get_dimension(), 6); // 3 points × 2 residuals
    }

    #[test]
    fn test_linearize_dimensions() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.1, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0), Vector2::new(350.0, 240.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = KannalaBrandtProjectionFactor::new(points_3d, points_2d);

        // Camera parameters: [fx, fy, cx, cy, k1, k2, k3, k4]
        let params = vec![DVector::from_vec(vec![
            460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04,
        ])];

        let (residual, jacobian) = factor.linearize(&params, true);

        assert_eq!(residual.len(), 4); // 2 points × 2 residuals
        assert!(jacobian.is_some());
        let jac = jacobian.unwrap();
        assert_eq!(jac.nrows(), 4); // 2 points × 2 residuals
        assert_eq!(jac.ncols(), 8); // 8 camera parameters
    }

    #[test]
    fn test_residual_computation() {
        // Test with a point on optical axis
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = KannalaBrandtProjectionFactor::new(points_3d, points_2d);

        // Parameters where center projects to (cx, cy)
        let params = vec![DVector::from_vec(vec![
            460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04,
        ])];

        let (residual, _) = factor.linearize(&params, false);

        // For point at (0,0,1), should project near (cx,cy)
        assert!(residual[0].abs() < 1.0);
        assert!(residual[1].abs() < 1.0);
    }

    #[test]
    fn test_jacobian_non_zero() {
        let points_3d_vec = vec![Vector3::new(0.1, 0.1, 1.0)];
        let points_2d_vec = vec![Vector2::new(330.0, 250.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = KannalaBrandtProjectionFactor::new(points_3d, points_2d);

        let params = vec![DVector::from_vec(vec![
            460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04,
        ])];

        let (_, jacobian) = factor.linearize(&params, true);

        assert!(jacobian.is_some());
        let jac = jacobian.unwrap();

        // Check that Jacobian has non-zero entries
        let has_nonzero = jac.iter().any(|&x| x.abs() > 1e-10);
        assert!(has_nonzero, "Jacobian should have non-zero entries");
    }

    #[test]
    fn test_jacobian_structure() {
        let points_3d_vec = vec![Vector3::new(0.2, 0.15, 1.0)];
        let points_2d_vec = vec![Vector2::new(380.0, 280.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = KannalaBrandtProjectionFactor::new(points_3d, points_2d);

        let params = vec![DVector::from_vec(vec![
            460.0, 460.0, 320.0, 240.0, -0.01, 0.05, -0.08, 0.04,
        ])];

        let (_, jacobian) = factor.linearize(&params, true);

        let jac = jacobian.unwrap();

        // Check specific Jacobian structure
        // ∂u/∂fy should be 0
        assert!(jac[(0, 1)].abs() < 1e-12);
        // ∂v/∂fx should be 0
        assert!(jac[(1, 0)].abs() < 1e-12);
        // ∂u/∂cx should be 1
        assert!((jac[(0, 2)] - 1.0).abs() < 1e-12);
        // ∂v/∂cy should be 1
        assert!((jac[(1, 3)] - 1.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "Number of 3D and 2D points must match")]
    fn test_mismatched_points_panic() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0), Vector2::new(330.0, 250.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        KannalaBrandtProjectionFactor::new(points_3d, points_2d);
    }
}
