//! Extended Unified Camera Model (EUCM) projection factor for apex-solver optimization.
//!
//! This module provides a factor implementation for the apex-solver framework
//! that computes reprojection errors and analytical Jacobians for the Extended
//! Unified Camera Model (EUCM). This allows using apex-solver's Levenberg-Marquardt optimizer
//! with hand-derived analytical derivatives.

use super::Factor;
use nalgebra::{
    DMatrix, DVector, Matrix, Matrix2xX, Matrix3xX, RawStorage, SVector, U1, U2, U3, Vector2,
};

/// Projection factor for EUCM camera model optimization with apex-solver.
///
/// This factor computes the reprojection error between observed 2D points and
/// the projection of 3D points using the Extended Unified Camera Model. It provides
/// analytical Jacobians for efficient optimization.
///
/// # Residual Formulation
///
/// For each 3D-2D point correspondence, the residual is computed as:
/// ```text
/// residual_x = fx * obs_x - (u - cx) * denom
/// residual_y = fy * obs_y - (v - cy) * denom
/// ```
///
/// where `denom = alpha * d + (1 - alpha) * obs_z` and `d = sqrt(beta * r² + obs_z²)`
///
/// # Parameters
///
/// The factor optimizes 6 camera parameters: `[fx, fy, cx, cy, alpha, beta]`
#[derive(Debug, Clone)]
pub struct EucmProjectionFactor {
    /// 3D points in camera coordinate system
    pub points_3d: Matrix3xX<f64>,
    /// Corresponding observed 2D points in image coordinates
    pub points_2d: Matrix2xX<f64>,
}

impl EucmProjectionFactor {
    /// Creates a new EUCM projection factor.
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
    /// * `params` - Camera parameters [fx, fy, cx, cy, alpha, beta]
    /// * `compute_jacobian` - Whether to compute the Jacobian
    ///
    /// # Returns
    ///
    /// Tuple of (residual_vector, optional_jacobian_matrix)
    #[inline]
    fn compute_point_residual_jacobian<S3, S2>(
        point_3d: Matrix<f64, U3, U1, S3>,
        point_2d: Matrix<f64, U2, U1, S2>,
        params: &SVector<f64, 6>,
        compute_jacobian: bool,
    ) -> (Vector2<f64>, Option<nalgebra::SMatrix<f64, 2, 6>>)
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
        let beta = params[5];
        const PRECISION: f64 = 1e-3;

        // Extract 3D coordinates
        let obs_x = point_3d[0];
        let obs_y = point_3d[1];
        let obs_z = point_3d[2];

        // Extract 2D coordinates (ground truth/observed)
        let gt_u = point_2d[0];
        let gt_v = point_2d[1];

        // Compute intermediate variables
        let u_cx = gt_u - cx;
        let v_cy = gt_v - cy;
        let r_squared = obs_x * obs_x + obs_y * obs_y;
        let d = (beta * r_squared + obs_z * obs_z).sqrt();
        let denom = alpha * d + (1.0 - alpha) * obs_z;

        // Check projection validity using EUCM's projection condition
        // Inline check: for alpha > 0.5, verify z >= denom * c where c = (alpha - 1) / (2*alpha - 1)
        let mut valid_projection = true;
        if alpha > 0.5 {
            let c = (alpha - 1.0) / (2.0 * alpha - 1.0);
            if obs_z < denom * c {
                valid_projection = false;
            }
        }

        if denom < PRECISION || !valid_projection {
            // Invalid projection - return large residual
            let residual = Vector2::new(1e6, 1e6);
            let jacobian = if compute_jacobian {
                Some(nalgebra::SMatrix::<f64, 2, 6>::zeros())
            } else {
                None
            };
            return (residual, jacobian);
        }

        // Compute residual using the C++ reference formulation
        let residual = Vector2::new(fx * obs_x - u_cx * denom, fy * obs_y - v_cy * denom);

        // Compute analytical Jacobian if requested
        let jacobian = if compute_jacobian {
            let mut jac = nalgebra::SMatrix::<f64, 2, 6>::zeros();

            // ∂residual / ∂fx
            jac[(0, 0)] = obs_x;
            jac[(1, 0)] = 0.0;

            // ∂residual / ∂fy
            jac[(0, 1)] = 0.0;
            jac[(1, 1)] = obs_y;

            // ∂residual / ∂cx
            jac[(0, 2)] = denom;
            jac[(1, 2)] = 0.0;

            // ∂residual / ∂cy
            jac[(0, 3)] = 0.0;
            jac[(1, 3)] = denom;

            // ∂residual / ∂alpha
            jac[(0, 4)] = (obs_z - d) * u_cx;
            jac[(1, 4)] = (obs_z - d) * v_cy;

            // ∂residual / ∂beta
            // Handle the case where d might be very small
            if d > PRECISION {
                jac[(0, 5)] = -(alpha * r_squared * u_cx) / (2.0 * d);
                jac[(1, 5)] = -(alpha * r_squared * v_cy) / (2.0 * d);
            } else {
                jac[(0, 5)] = 0.0;
                jac[(1, 5)] = 0.0;
            }

            Some(jac)
        } else {
            None
        };

        (residual, jacobian)
    }
}

impl Factor for EucmProjectionFactor {
    /// Compute residuals and analytical Jacobians for all point correspondences.
    ///
    /// # Arguments
    ///
    /// * `params` - Slice containing camera parameters as a single DVector:
    ///   `params[0] = [fx, fy, cx, cy, alpha, beta]`
    /// * `compute_jacobian` - Whether to compute the Jacobian matrix
    ///
    /// # Returns
    ///
    /// Tuple of (residual_vector, optional_jacobian_matrix) where:
    /// - `residual_vector` has dimension `2 * num_points`
    /// - `jacobian_matrix` has dimension `(2 * num_points) × 6`
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // Extract camera parameters into SVector
        let cam_params = &params[0];
        let camera_params = SVector::<f64, 6>::from_row_slice(&[
            cam_params[0], // fx
            cam_params[1], // fy
            cam_params[2], // cx
            cam_params[3], // cy
            cam_params[4], // alpha
            cam_params[5], // beta
        ]);

        let num_points = self.points_2d.ncols();
        let residual_dim = num_points * 2;

        // Initialize residual vector
        let mut residuals = DVector::zeros(residual_dim);

        // Initialize Jacobian if needed
        let mut jacobian_matrix = if compute_jacobian {
            Some(DMatrix::zeros(residual_dim, 6))
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
                    .view_mut((i * 2, 0), (2, 6))
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
            Vector2::new(960.0, 546.0),
            Vector2::new(990.0, 546.0),
            Vector2::new(960.0, 576.0),
        ];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = EucmProjectionFactor::new(points_3d, points_2d);
        assert_eq!(factor.get_dimension(), 6); // 3 points × 2 residuals
    }

    #[test]
    fn test_linearize_dimensions() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.1, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(960.0, 546.0), Vector2::new(990.0, 546.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = EucmProjectionFactor::new(points_3d, points_2d);

        // Camera parameters: [fx, fy, cx, cy, alpha, beta]
        let params = vec![DVector::from_vec(vec![
            1313.83, 1313.27, 960.471, 546.981, 1.01674, 0.5,
        ])];

        let (residual, jacobian) = factor.linearize(&params, true);

        assert_eq!(residual.len(), 4); // 2 points × 2 residuals
        assert!(jacobian.is_some());
        let jac = jacobian.unwrap();
        assert_eq!(jac.nrows(), 4); // 2 points × 2 residuals
        assert_eq!(jac.ncols(), 6); // 6 camera parameters
    }

    #[test]
    fn test_residual_computation() {
        // Test with a simple case where 3D point at (0,0,1) should project near (cx,cy)
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(960.471, 546.981)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = EucmProjectionFactor::new(points_3d, points_2d);

        // Parameters: reasonable EUCM values
        let params = vec![DVector::from_vec(vec![
            1313.83, 1313.27, 960.471, 546.981, 1.01674, 0.5,
        ])];

        let (residual, _) = factor.linearize(&params, false);

        // For point at (0,0,1) with the given parameters, residual should be small
        assert!(residual[0].abs() < 10.0);
        assert!(residual[1].abs() < 10.0);
    }

    #[test]
    fn test_jacobian_non_zero() {
        let points_3d_vec = vec![Vector3::new(0.1, 0.1, 1.0)];
        let points_2d_vec = vec![Vector2::new(970.0, 556.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = EucmProjectionFactor::new(points_3d, points_2d);

        let params = vec![DVector::from_vec(vec![
            1313.83, 1313.27, 960.471, 546.981, 1.01674, 0.5,
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
        let points_2d_vec = vec![Vector2::new(960.0, 546.0), Vector2::new(970.0, 556.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        EucmProjectionFactor::new(points_3d, points_2d);
    }
}
