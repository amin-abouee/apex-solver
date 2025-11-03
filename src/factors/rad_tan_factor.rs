//! Radial-Tangential (RadTan) projection factor for apex-solver optimization.
//!
//! This module provides a factor implementation for the apex-solver framework
//! that computes reprojection errors and analytical Jacobians for the Radial-Tangential
//! camera model. This allows using apex-solver's Levenberg-Marquardt optimizer
//! with hand-derived analytical derivatives.

use super::Factor;
use nalgebra::{
    DMatrix, DVector, Matrix, Matrix2xX, Matrix3xX, RawStorage, SVector, U1, U2, U3, Vector2,
};

/// Projection factor for RadTan camera model optimization with apex-solver.
///
/// This factor computes the reprojection error between observed 2D points and
/// the projection of 3D points using the Radial-Tangential camera model. It provides
/// analytical Jacobians for efficient optimization.
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
/// # Parameters
///
/// The factor optimizes 9 camera parameters: `[fx, fy, cx, cy, k1, k2, p1, p2, k3]`
///
/// Note: The parameter order follows the Rust implementation where distortions are
/// stored as `[k1, k2, p1, p2, k3]`, which differs from some C++ implementations
/// that use `[k1, k2, k3, p1, p2]`.
#[derive(Debug, Clone)]
pub struct RadTanProjectionFactor {
    /// 3D points in camera coordinate system
    pub points_3d: Matrix3xX<f64>,
    /// Corresponding observed 2D points in image coordinates
    pub points_2d: Matrix2xX<f64>,
}

impl RadTanProjectionFactor {
    /// Creates a new RadTan projection factor.
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
    /// * `params` - Camera parameters [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    /// * `compute_jacobian` - Whether to compute the Jacobian
    ///
    /// # Returns
    ///
    /// Tuple of (residual_vector, optional_jacobian_matrix)
    ///
    /// # Parameter Ordering
    ///
    /// The Jacobian follows Rust's parameter order: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    #[inline]
    fn compute_point_residual_jacobian<S3, S2>(
        point_3d: Matrix<f64, U3, U1, S3>,
        point_2d: Matrix<f64, U2, U1, S2>,
        params: &SVector<f64, 9>,
        compute_jacobian: bool,
    ) -> (Vector2<f64>, Option<nalgebra::SMatrix<f64, 2, 9>>)
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
        let p1 = params[6];
        let p2 = params[7];
        let k3 = params[8];
        // Extract 3D coordinates
        let obs_x = point_3d[0];
        let obs_y = point_3d[1];
        let obs_z = point_3d[2];

        // Extract 2D coordinates (ground truth/observed)
        let gt_u = point_2d[0];
        let gt_v = point_2d[1];

        // Check if point is at camera center (z too small)
        if obs_z < f64::EPSILON.sqrt() {
            // Invalid projection - return large residual
            let residual = Vector2::new(1e6, 1e6);
            let jacobian = if compute_jacobian {
                Some(nalgebra::SMatrix::<f64, 2, 9>::zeros())
            } else {
                None
            };
            return (residual, jacobian);
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
        let residual = Vector2::new(u - gt_u, v - gt_v);

        // Compute analytical Jacobian if requested
        let jacobian = if compute_jacobian {
            let mut jac = nalgebra::SMatrix::<f64, 2, 9>::zeros();

            // ∂residual / ∂fx (column 0)
            jac[(0, 0)] = x_distorted;
            jac[(1, 0)] = 0.0;

            // ∂residual / ∂fy (column 1)
            jac[(0, 1)] = 0.0;
            jac[(1, 1)] = y_distorted;

            // ∂residual / ∂cx (column 2)
            jac[(0, 2)] = 1.0;
            jac[(1, 2)] = 0.0;

            // ∂residual / ∂cy (column 3)
            jac[(0, 3)] = 0.0;
            jac[(1, 3)] = 1.0;

            // ∂residual / ∂k1 (column 4)
            jac[(0, 4)] = fx * x_prime * r2;
            jac[(1, 4)] = fy * y_prime * r2;

            // ∂residual / ∂k2 (column 5)
            jac[(0, 5)] = fx * x_prime * r4;
            jac[(1, 5)] = fy * y_prime * r4;

            // ∂residual / ∂p1 (column 6)
            // Note: This is column 6 in Rust order [k1, k2, p1, p2, k3]
            jac[(0, 6)] = fx * 2.0 * x_prime * y_prime;
            jac[(1, 6)] = fy * (r2 + 2.0 * y_prime.powi(2));

            // ∂residual / ∂p2 (column 7)
            jac[(0, 7)] = fx * (r2 + 2.0 * x_prime.powi(2));
            jac[(1, 7)] = fy * 2.0 * x_prime * y_prime;

            // ∂residual / ∂k3 (column 8)
            jac[(0, 8)] = fx * x_prime * r6;
            jac[(1, 8)] = fy * y_prime * r6;

            Some(jac)
        } else {
            None
        };

        (residual, jacobian)
    }
}

impl Factor for RadTanProjectionFactor {
    /// Compute residuals and analytical Jacobians for all point correspondences.
    ///
    /// # Arguments
    ///
    /// * `params` - Slice containing camera parameters as a single DVector:
    ///   `params[0] = [fx, fy, cx, cy, k1, k2, p1, p2, k3]`
    /// * `compute_jacobian` - Whether to compute the Jacobian matrix
    ///
    /// # Returns
    ///
    /// Tuple of (residual_vector, optional_jacobian_matrix) where:
    /// - `residual_vector` has dimension `2 * num_points`
    /// - `jacobian_matrix` has dimension `(2 * num_points) × 9`
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // Extract camera parameters into SVector
        let cam_params = &params[0];
        let camera_params = SVector::<f64, 9>::from_row_slice(&[
            cam_params[0], // fx
            cam_params[1], // fy
            cam_params[2], // cx
            cam_params[3], // cy
            cam_params[4], // k1
            cam_params[5], // k2
            cam_params[6], // p1
            cam_params[7], // p2
            cam_params[8], // k3
        ]);

        let num_points = self.points_2d.ncols();
        let residual_dim = num_points * 2;

        // Initialize residual vector
        let mut residuals = DVector::zeros(residual_dim);

        // Initialize Jacobian if needed
        let mut jacobian_matrix = if compute_jacobian {
            Some(DMatrix::zeros(residual_dim, 9))
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
                    .view_mut((i * 2, 0), (2, 9))
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
            Vector2::new(362.0, 246.0),
            Vector2::new(392.0, 246.0),
            Vector2::new(362.0, 276.0),
        ];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = RadTanProjectionFactor::new(points_3d, points_2d);
        assert_eq!(factor.get_dimension(), 6); // 3 points × 2 residuals
    }

    #[test]
    fn test_linearize_dimensions() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.1, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(362.0, 246.0), Vector2::new(392.0, 246.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = RadTanProjectionFactor::new(points_3d, points_2d);

        // Camera parameters: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
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

        assert_eq!(residual.len(), 4); // 2 points × 2 residuals
        assert!(jacobian.is_some());
        let jac = jacobian.unwrap();
        assert_eq!(jac.nrows(), 4); // 2 points × 2 residuals
        assert_eq!(jac.ncols(), 9); // 9 camera parameters
    }

    #[test]
    fn test_residual_computation() {
        // Test with a simple case where 3D point at (0,0,1) should project near (cx,cy)
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(362.680, 246.049)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = RadTanProjectionFactor::new(points_3d, points_2d);

        // Parameters from samples/rad_tan.yaml
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

        // For point at (0,0,1) with the given parameters, residual should be small
        assert!(residual[0].abs() < 10.0);
        assert!(residual[1].abs() < 10.0);
    }

    #[test]
    fn test_jacobian_non_zero() {
        let points_3d_vec = vec![Vector3::new(0.1, 0.1, 1.0)];
        let points_2d_vec = vec![Vector2::new(370.0, 256.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = RadTanProjectionFactor::new(points_3d, points_2d);

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
        let jac = jacobian.unwrap();

        // Check that Jacobian has non-zero entries
        let has_nonzero = jac.iter().any(|&x| x.abs() > 1e-10);
        assert!(has_nonzero, "Jacobian should have non-zero entries");
    }

    #[test]
    #[should_panic(expected = "Number of 3D and 2D points must match")]
    fn test_mismatched_points_panic() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(362.0, 246.0), Vector2::new(370.0, 256.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        RadTanProjectionFactor::new(points_3d, points_2d);
    }
}
