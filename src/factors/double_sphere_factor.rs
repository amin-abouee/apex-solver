//! Double Sphere projection factor for apex-solver optimization.
//!
//! This module provides a factor implementation for the apex-solver framework
//! that computes reprojection errors and analytical Jacobians for the Double Sphere
//! camera model. This allows using apex-solver's Levenberg-Marquardt optimizer
//! with hand-derived analytical derivatives.

use super::Factor;
use nalgebra::{
    DMatrix, DVector, Matrix, Matrix2xX, Matrix3xX, RawStorage, SVector, U1, U2, U3, Vector2,
};

/// Projection factor for Double Sphere camera model optimization with apex-solver.
///
/// This factor computes the reprojection error between observed 2D points and
/// the projection of 3D points using the Double Sphere camera model. It provides
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
/// where `denom = alpha * d2 + (1 - alpha) * gamma` from the Double Sphere model.
///
/// # Parameters
///
/// The factor optimizes 6 camera parameters: `[fx, fy, cx, cy, alpha, xi]`
#[derive(Debug, Clone)]
pub struct DoubleSphereProjectionFactor {
    /// 3D points in camera coordinate system (3×N matrix)
    pub points_3d: Matrix3xX<f64>,
    /// Corresponding observed 2D points in image coordinates (2×N matrix)
    pub points_2d: Matrix2xX<f64>,
}

impl DoubleSphereProjectionFactor {
    /// Creates a new Double Sphere projection factor.
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
    /// * `params` - Camera parameters [fx, fy, cx, cy, alpha, xi]
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
    ) -> (Vector2<f64>, Option<nalgebra::Matrix2x6<f64>>)
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
        let xi = params[5];
        const PRECISION: f64 = 1e-3;

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
            // Invalid projection - return large residual
            let residual = Vector2::new(1e6, 1e6);
            let jacobian = if compute_jacobian {
                Some(nalgebra::Matrix2x6::zeros())
            } else {
                None
            };
            return (residual, jacobian);
        }

        // Compute residual
        let u_cx = point_2d[0] - cx;
        let v_cy = point_2d[1] - cy;

        let residual = Vector2::new(fx * x - u_cx * denom, fy * y - v_cy * denom);

        // Compute analytical Jacobian if requested
        let jacobian = if compute_jacobian {
            let mut jac = nalgebra::Matrix2x6::zeros();

            // ∂residual / ∂fx
            jac[(0, 0)] = x;
            jac[(1, 0)] = 0.0;

            // ∂residual / ∂fy
            jac[(0, 1)] = 0.0;
            jac[(1, 1)] = y;

            // ∂residual / ∂cx
            jac[(0, 2)] = denom;
            jac[(1, 2)] = 0.0;

            // ∂residual / ∂cy
            jac[(0, 3)] = 0.0;
            jac[(1, 3)] = denom;

            // ∂residual / ∂alpha
            jac[(0, 4)] = (gamma - d2) * u_cx;
            jac[(1, 4)] = (gamma - d2) * v_cy;

            // ∂residual / ∂xi
            let coeff = (alpha * d1 * gamma) / d2 + (m_alpha * d1);
            jac[(0, 5)] = -u_cx * coeff;
            jac[(1, 5)] = -v_cy * coeff;

            Some(jac)
        } else {
            None
        };

        (residual, jacobian)
    }
}

impl Factor for DoubleSphereProjectionFactor {
    /// Compute residuals and analytical Jacobians for all point correspondences.
    ///
    /// # Arguments
    ///
    /// * `params` - Slice containing camera parameters as a single DVector:
    ///   `params[0] = [fx, fy, cx, cy, alpha, xi]`
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
            cam_params[5], // xi
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
            Vector2::new(320.0, 240.0),
            Vector2::new(350.0, 240.0),
            Vector2::new(320.0, 270.0),
        ];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = DoubleSphereProjectionFactor::new(points_3d, points_2d);
        assert_eq!(factor.get_dimension(), 6); // 3 points × 2 residuals
    }

    #[test]
    fn test_linearize_dimensions() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.1, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0), Vector2::new(350.0, 240.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        let factor = DoubleSphereProjectionFactor::new(points_3d, points_2d);

        // Camera parameters: [fx, fy, cx, cy, alpha, xi]
        let params = vec![DVector::from_vec(vec![
            300.0, 300.0, 320.0, 240.0, 0.5, 0.1,
        ])];

        let (residual, jacobian) = factor.linearize(&params, true);

        assert_eq!(residual.len(), 4); // 2 points × 2 residuals
        assert!(jacobian.is_some());
        let jac = jacobian.unwrap();
        assert_eq!(jac.nrows(), 4); // 2 points × 2 residuals
        assert_eq!(jac.ncols(), 6); // 6 camera parameters
    }

    #[test]
    #[should_panic(expected = "Number of 3D and 2D points must match")]
    fn test_mismatched_points_panic() {
        let points_3d_vec = vec![Vector3::new(0.0, 0.0, 1.0)];
        let points_2d_vec = vec![Vector2::new(320.0, 240.0), Vector2::new(350.0, 240.0)];
        let points_3d = Matrix3xX::from_columns(&points_3d_vec);
        let points_2d = Matrix2xX::from_columns(&points_2d_vec);

        DoubleSphereProjectionFactor::new(points_3d, points_2d);
    }
}
