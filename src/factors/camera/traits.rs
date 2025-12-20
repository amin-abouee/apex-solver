//! Core traits and types for camera projection.

use crate::manifold::se3::SE3;
use nalgebra::{DVector, Matrix2xX, Matrix3, Matrix3xX, SMatrix, Vector2, Vector3};

/// Configuration for which parameters to optimize.
///
/// Uses const generic booleans for compile-time optimization selection.
/// This allows zero-cost abstraction - the compiler generates specialized
/// code for each configuration.
///
/// # Examples
///
/// ```
/// use apex_solver::factors::camera::{OptimizeParams, BundleAdjustment};
///
/// // Use predefined aliases
/// type BA = BundleAdjustment;  // OptimizeParams<true, true, false>
///
/// // Or create custom configurations
/// type PoseOnly = OptimizeParams<true, false, false>;
/// type IntrinsicsOnly = OptimizeParams<false, false, true>;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct OptimizeParams<const POSE: bool, const LANDMARK: bool, const INTRINSIC: bool>;

impl<const P: bool, const L: bool, const I: bool> OptimizeParams<P, L, I> {
    /// Whether to optimize camera pose
    pub const POSE: bool = P;
    /// Whether to optimize 3D landmarks
    pub const LANDMARK: bool = L;
    /// Whether to optimize camera intrinsics
    pub const INTRINSIC: bool = I;
}

/// Bundle Adjustment: optimize pose + landmarks (fixed intrinsics)
pub type BundleAdjustment = OptimizeParams<true, true, false>;

/// Self-Calibration: optimize pose + landmarks + intrinsics
pub type SelfCalibration = OptimizeParams<true, true, true>;

/// Calibration only: optimize intrinsics (fixed pose + landmarks)
pub type OnlyIntrinsics = OptimizeParams<false, false, true>;

/// Visual Odometry: optimize pose only (fixed landmarks + intrinsics)
pub type OnlyPose = OptimizeParams<true, false, false>;

/// Triangulation/Structure: optimize landmarks only (fixed pose + intrinsics)
pub type OnlyLandmarks = OptimizeParams<false, true, false>;

/// Pose + Intrinsics: optimize pose and intrinsics (fixed landmarks)
pub type PoseAndIntrinsics = OptimizeParams<true, false, true>;

/// Landmarks + Intrinsics: optimize landmarks and intrinsics (fixed pose)
pub type LandmarksAndIntrinsics = OptimizeParams<false, true, true>;

/// Trait for camera projection models.
///
/// Provides a unified interface for different camera models (Pinhole, Fisheye,
/// Double Sphere, EUCM, etc.) to compute projections and Jacobians for bundle
/// adjustment and structure from motion.
///
/// # Implementation Requirements
///
/// All methods assume points are in the camera coordinate frame (right-handed,
/// Z-axis pointing forward). Invalid projections (e.g., points behind camera)
/// should return `None` or appropriate error indicators.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` for parallel residual evaluation.
pub trait CameraModel: Send + Sync + Clone + std::fmt::Debug + 'static {
    /// Number of intrinsic parameters (compile-time constant).
    ///
    /// Examples:
    /// - Pinhole: 4 (fx, fy, cx, cy)
    /// - RadTan: 9 (fx, fy, cx, cy, k1, k2, p1, p2, k3)
    /// - Double Sphere: 6 (fx, fy, cx, cy, alpha, xi)
    const INTRINSIC_DIM: usize;

    /// Jacobian type for intrinsics: 2 × INTRINSIC_DIM
    type IntrinsicJacobian: Clone
        + std::fmt::Debug
        + Default
        + std::ops::Index<(usize, usize), Output = f64>;

    /// Jacobian type for 3D point: 2 × 3
    type PointJacobian: Clone
        + std::fmt::Debug
        + Default
        + std::ops::Index<(usize, usize), Output = f64>
        + std::ops::Mul<SMatrix<f64, 3, 6>, Output = SMatrix<f64, 2, 6>>
        + std::ops::Mul<nalgebra::Matrix3<f64>, Output = SMatrix<f64, 2, 3>>;

    /// Project a single 3D point in camera frame to 2D image coordinates.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame (x, y, z)
    ///
    /// # Returns
    ///
    /// * `Some((u, v))` - 2D image coordinates in pixels
    /// * `None` - If projection is invalid (e.g., point behind camera, outside FOV)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let camera = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
    /// let p_cam = Vector3::new(0.1, 0.2, 1.0);  // Point 1m in front
    /// let uv = camera.project(&p_cam).unwrap();
    /// ```
    fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>>;

    /// Check if a 3D point is valid for projection.
    ///
    /// This is faster than `project()` when you only need to check validity.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame
    ///
    /// # Returns
    ///
    /// `true` if the point can be projected, `false` otherwise
    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool;

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    ///
    /// Computes ∂(u,v)/∂(x,y,z) where (x,y,z) is the point in camera frame.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera frame (must be valid for projection)
    ///
    /// # Returns
    ///
    /// 2×3 matrix of partial derivatives
    ///
    /// # Panics
    ///
    /// May panic or return incorrect results if `p_cam` is not valid.
    /// Call `is_valid_point()` first.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian;

    /// Jacobian of projection w.r.t. camera pose (via chain rule).
    ///
    /// Returns both the projection Jacobian and the pose transformation Jacobian
    /// so the factor can apply the chain rule efficiently:
    ///
    /// ```text
    /// ∂(u,v)/∂(pose) = ∂(u,v)/∂(p_cam) * ∂(p_cam)/∂(pose)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `p_world` - 3D point in world coordinates
    /// * `pose` - Camera pose (world-to-camera transform, SE3)
    ///
    /// # Returns
    ///
    /// Tuple of:
    /// - `jacobian_projection`: ∂(u,v)/∂(p_cam) (2×3)
    /// - `jacobian_transform`: ∂(p_cam)/∂(pose) (3×6) for SE3 tangent space
    ///
    /// # Note
    ///
    /// The pose Jacobian is for right perturbation in SE3 tangent space:
    /// - First 3 columns: translation (∂p_cam/∂t)
    /// - Last 3 columns: rotation (∂p_cam/∂ω)
    fn jacobian_pose(
        &self,
        p_world: &Vector3<f64>,
        pose: &SE3,
    ) -> (Self::PointJacobian, SMatrix<f64, 3, 6>);

    /// Jacobian of projection w.r.t. intrinsic parameters.
    ///
    /// Computes ∂(u,v)/∂(intrinsics) where intrinsics are camera-model specific.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera frame (must be valid for projection)
    ///
    /// # Returns
    ///
    /// 2×INTRINSIC_DIM matrix of partial derivatives
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian;

    /// Project multiple 3D points (batch operation).
    ///
    /// Default implementation calls `project()` for each point.
    /// Override for performance-critical applications.
    ///
    /// # Arguments
    ///
    /// * `points_cam` - 3×N matrix of 3D points in camera frame
    ///
    /// # Returns
    ///
    /// 2×N matrix of 2D projections. Invalid projections are marked with
    /// large values (1e6, 1e6) to indicate errors.
    fn project_batch(&self, points_cam: &Matrix3xX<f64>) -> Matrix2xX<f64> {
        let n = points_cam.ncols();
        let mut result = Matrix2xX::zeros(n);
        for i in 0..n {
            let p = Vector3::new(points_cam[(0, i)], points_cam[(1, i)], points_cam[(2, i)]);
            match self.project(&p) {
                Some(uv) => result.set_column(i, &uv),
                None => result.set_column(i, &Vector2::new(1e6, 1e6)),
            }
        }
        result
    }

    /// Get intrinsic parameters as dynamic vector.
    ///
    /// # Returns
    ///
    /// INTRINSIC_DIM-dimensional vector of camera parameters
    fn intrinsics_vec(&self) -> DVector<f64>;

    /// Create camera from parameter slice.
    ///
    /// # Arguments
    ///
    /// * `params` - Slice containing at least INTRINSIC_DIM values
    ///
    /// # Returns
    ///
    /// New camera instance
    ///
    /// # Panics
    ///
    /// Panics if `params.len() < INTRINSIC_DIM`
    fn from_params(params: &[f64]) -> Self;
}

/// Compute skew-symmetric matrix from a 3D vector.
///
/// For a vector v = [x, y, z], returns:
/// ```text
/// [  0  -z   y ]
/// [  z   0  -x ]
/// [ -y   x   0 ]
/// ```
///
/// This is used for the cross product: `[v]× * w = v × w`
#[inline]
pub fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_params_constants() {
        assert!(BundleAdjustment::POSE);
        assert!(BundleAdjustment::LANDMARK);
        assert!(!BundleAdjustment::INTRINSIC);

        assert!(SelfCalibration::POSE);
        assert!(SelfCalibration::LANDMARK);
        assert!(SelfCalibration::INTRINSIC);

        assert!(!OnlyIntrinsics::POSE);
        assert!(!OnlyIntrinsics::LANDMARK);
        assert!(OnlyIntrinsics::INTRINSIC);
    }

    #[test]
    fn test_skew_symmetric() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let skew = skew_symmetric(&v);

        // Test skew-symmetric property: skew^T = -skew
        let skew_t = skew.transpose();
        assert!((skew + skew_t).norm() < 1e-10);

        // Test cross product: [v]× * w = v × w
        let w = Vector3::new(4.0, 5.0, 6.0);
        let cross1 = skew * w;
        let cross2 = v.cross(&w);
        assert!((cross1 - cross2).norm() < 1e-10);
    }
}
