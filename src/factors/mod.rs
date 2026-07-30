//! Factor implementations for graph-based optimization problems.
//!
//! Factors (also called constraints or error functions) represent measurements or relationships
//! between variables in a factor graph. Each factor computes a residual (error) vector and its
//! Jacobian with respect to the connected variables.
//!
//! # Factor Graph Formulation
//!
//! In graph-based SLAM and bundle adjustment, the optimization problem is represented as:
//!
//! ```text
//! minimize Σ_i ||r_i(x)||²
//! ```
//!
//! where:
//! - `x` is the set of variables (poses, landmarks, etc.)
//! - `r_i(x)` is the residual function for factor i
//! - Each factor connects one or more variables
//!
//! # Factor Types
//!
//! ## Pose Factors
//! - **Between factors**: Relative pose constraints (SE2, SE3)
//! - **Prior factors**: Unary constraints on single variables
//!
//! ## Camera Projection Factors
//!
//! Use [`ProjectionFactor`] with a specific
//! [`CameraModel`](apex_camera_models::CameraModel).
//!
//! Supported camera models:
//! - [`PinholeCamera`](camera::PinholeCamera)
//! - [`DoubleSphereCamera`](camera::DoubleSphereCamera)
//! - [`EucmCamera`](camera::EucmCamera)
//! - [`FovCamera`](camera::FovCamera)
//! - [`KannalaBrandtCamera`](camera::KannalaBrandtCamera)
//! - [`RadTanCamera`](camera::RadTanCamera)
//! - [`UcmCamera`](camera::UcmCamera)
//!
//! # Linearization
//!
//! Each factor must provide a `linearize` method that writes the residual and Jacobian
//! directly into caller-provided buffers — no heap allocation for the output.
//!
//! This information is used by the optimizer to compute parameter updates via Newton-type methods.

use thiserror::Error;

// Pose factors
pub mod between_factor;
pub mod prior_factor;
pub mod projection_factor;

pub use between_factor::BetweenFactor;
pub use prior_factor::PriorFactor;
pub use projection_factor::ProjectionFactor;

// Optimization configuration types

/// Configuration for which parameters to optimize.
///
/// Uses const generic booleans for compile-time optimization selection.
///
/// # Type Parameters
///
/// - `POSE`: Whether to optimize camera pose (SE3 transformation)
/// - `LANDMARK`: Whether to optimize 3D landmark positions
/// - `INTRINSIC`: Whether to optimize camera intrinsic parameters
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

/// Bundle Adjustment: optimize pose + landmarks (intrinsics fixed).
pub type BundleAdjustment = OptimizeParams<true, true, false>;

/// Self-Calibration: optimize pose + landmarks + intrinsics.
pub type SelfCalibration = OptimizeParams<true, true, true>;

/// Only Intrinsics: optimize intrinsics (pose and landmarks fixed).
pub type OnlyIntrinsics = OptimizeParams<false, false, true>;

/// Only Pose: optimize pose (landmarks and intrinsics fixed).
pub type OnlyPose = OptimizeParams<true, false, false>;

/// Only Landmarks: optimize landmarks (pose and intrinsics fixed).
pub type OnlyLandmarks = OptimizeParams<false, true, false>;

/// Pose and Intrinsics: optimize pose + intrinsics (landmarks fixed).
pub type PoseAndIntrinsics = OptimizeParams<true, false, true>;

/// Landmarks and Intrinsics: optimize landmarks + intrinsics (pose fixed).
pub type LandmarksAndIntrinsics = OptimizeParams<false, true, true>;

// Camera module alias for backward compatibility
// Re-exports the apex-camera-models crate as `camera` module
pub mod camera {
    pub use apex_camera_models::*;
}

/// Factor-specific error types for apex-solver
#[derive(Debug, Clone, Error)]
pub enum FactorError {
    /// Invalid dimension mismatch between expected and actual
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    /// Invalid projection (point behind camera or outside valid range)
    #[error("Invalid projection: {0}")]
    InvalidProjection(String),

    /// Jacobian computation failed
    #[error("Jacobian computation failed: {0}")]
    JacobianFailed(String),

    /// Invalid parameter values
    #[error("Invalid parameter values: {0}")]
    InvalidParameters(String),

    /// Numerical instability detected
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
}

/// Result type for factor operations
pub type FactorResult<T> = Result<T, FactorError>;

/// Trait for factor (constraint) implementations in factor graph optimization.
///
/// A factor represents a measurement or constraint connecting one or more variables.
/// It writes the residual and Jacobian directly into caller-provided buffers — no heap
/// allocation for the output. Parameters arrive as zero-copy `&[f64]` slices from
/// manifold storage.
///
/// # Thread Safety
///
/// Factors must be `Send + Sync` to enable parallel residual/Jacobian evaluation.
///
/// # Example
///
/// ```
/// use apex_solver::factors::Factor;
/// use faer::prelude::ReborrowMut;
///
/// // Simple 1D range measurement factor
/// struct RangeFactor {
///     measurement: f64,
/// }
///
/// impl Factor for RangeFactor {
///     fn linearize(
///         &self,
///         params: &[&[f64]],
///         residual: &mut [f64],
///         jacobian: Option<faer::mat::MatMut<'_, f64>>,
///     ) {
///         let x = params[0][0];
///         let y = params[0][1];
///         let dist = (x * x + y * y).sqrt();
///         residual[0] = self.measurement - dist;
///         if let Some(mut jac) = jacobian {
///             *jac.rb_mut().get_mut(0, 0) = -x / dist;
///             *jac.rb_mut().get_mut(0, 1) = -y / dist;
///         }
///     }
///     fn residual_dim(&self) -> usize { 1 }
///     fn jacobian_shape(&self) -> (usize, usize) { (1, 2) }
/// }
/// ```
pub trait Factor: Send + Sync {
    /// Write residual and (optionally) Jacobian into pre-allocated buffers.
    ///
    /// - `params`: one `&[f64]` slice per connected variable (from `ManifoldVariable::as_param_slice`)
    /// - `residual`: pre-allocated output buffer of length `residual_dim()`
    /// - `jacobian`: optional column-major `MatMut` of shape `jacobian_shape()`
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    );

    /// Number of residual rows (length of the `residual` buffer).
    fn residual_dim(&self) -> usize;

    /// `(rows, cols)` of the Jacobian — `rows == residual_dim()`, `cols == sum of variable DOFs`.
    fn jacobian_shape(&self) -> (usize, usize);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorLogging;
    use faer::prelude::ReborrowMut;
    use nalgebra::dvector;

    // -------------------------------------------------------------------------
    // OptimizeParams const generic flags — all 7 type aliases
    // -------------------------------------------------------------------------

    #[test]
    fn test_optimize_params_bundle_adjustment_flags() {
        const { assert!(BundleAdjustment::POSE) };
        const { assert!(BundleAdjustment::LANDMARK) };
        const { assert!(!BundleAdjustment::INTRINSIC) };
    }

    #[test]
    fn test_optimize_params_self_calibration_flags() {
        const { assert!(SelfCalibration::POSE) };
        const { assert!(SelfCalibration::LANDMARK) };
        const { assert!(SelfCalibration::INTRINSIC) };
    }

    #[test]
    fn test_optimize_params_only_intrinsics_flags() {
        const { assert!(!OnlyIntrinsics::POSE) };
        const { assert!(!OnlyIntrinsics::LANDMARK) };
        const { assert!(OnlyIntrinsics::INTRINSIC) };
    }

    #[test]
    fn test_optimize_params_only_pose_flags() {
        const { assert!(OnlyPose::POSE) };
        const { assert!(!OnlyPose::LANDMARK) };
        const { assert!(!OnlyPose::INTRINSIC) };
    }

    #[test]
    fn test_optimize_params_only_landmarks_flags() {
        const { assert!(!OnlyLandmarks::POSE) };
        const { assert!(OnlyLandmarks::LANDMARK) };
        const { assert!(!OnlyLandmarks::INTRINSIC) };
    }

    #[test]
    fn test_optimize_params_pose_and_intrinsics_flags() {
        const { assert!(PoseAndIntrinsics::POSE) };
        const { assert!(!PoseAndIntrinsics::LANDMARK) };
        const { assert!(PoseAndIntrinsics::INTRINSIC) };
    }

    #[test]
    fn test_optimize_params_landmarks_and_intrinsics_flags() {
        const { assert!(!LandmarksAndIntrinsics::POSE) };
        const { assert!(LandmarksAndIntrinsics::LANDMARK) };
        const { assert!(LandmarksAndIntrinsics::INTRINSIC) };
    }

    // -------------------------------------------------------------------------
    // FactorError Display — one per variant
    // -------------------------------------------------------------------------

    #[test]
    fn test_factor_error_invalid_dimension_display() {
        let e = FactorError::InvalidDimension {
            expected: 3,
            actual: 6,
        };
        let s = e.to_string();
        assert!(s.contains("3"), "{s}");
        assert!(s.contains("6"), "{s}");
    }

    #[test]
    fn test_factor_error_invalid_projection_display() {
        let e = FactorError::InvalidProjection("behind camera".into());
        assert!(e.to_string().contains("behind camera"));
    }

    #[test]
    fn test_factor_error_jacobian_failed_display() {
        let e = FactorError::JacobianFailed("singular".into());
        assert!(e.to_string().contains("singular"));
    }

    #[test]
    fn test_factor_error_invalid_parameters_display() {
        let e = FactorError::InvalidParameters("nan detected".into());
        assert!(e.to_string().contains("nan detected"));
    }

    #[test]
    fn test_factor_error_numerical_instability_display() {
        let e = FactorError::NumericalInstability("overflow".into());
        assert!(e.to_string().contains("overflow"));
    }

    // -------------------------------------------------------------------------
    // log() / log_with_source() return self
    // -------------------------------------------------------------------------

    #[test]
    fn test_factor_error_log_returns_self() {
        let e = FactorError::JacobianFailed("test_log".into());
        let returned = e.log();
        assert!(returned.to_string().contains("test_log"));
    }

    #[test]
    fn test_factor_error_log_with_source_returns_self() {
        let e = FactorError::InvalidProjection("proj_log".into());
        let source = std::io::Error::other("src");
        let returned = e.log_with_source(source);
        assert!(returned.to_string().contains("proj_log"));
    }

    // -------------------------------------------------------------------------
    // Factor trait — local implementation
    // -------------------------------------------------------------------------

    struct ConstantFactor {
        value: f64,
    }

    impl Factor for ConstantFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = params[0][0] - self.value;
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = 1.0;
            }
        }

        fn residual_dim(&self) -> usize {
            1
        }

        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 1)
        }
    }

    #[test]
    fn test_factor_compute_with_jacobian() {
        let f = ConstantFactor { value: 3.0 };
        let p = dvector![5.0];
        let params: Vec<&[f64]> = vec![p.as_slice()];
        let mut residual = vec![0.0f64; 1];
        let mut jac_buf = vec![0.0f64; 1];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, 1, 1);
        f.linearize(&params, &mut residual, Some(jac_mut));
        assert!((residual[0] - 2.0).abs() < 1e-12);
        assert!((jac_buf[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_factor_compute_without_jacobian() {
        let f = ConstantFactor { value: 3.0 };
        let p = dvector![5.0];
        let params: Vec<&[f64]> = vec![p.as_slice()];
        let mut residual = vec![0.0f64; 1];
        f.linearize(&params, &mut residual, None);
        assert!((residual[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_factor_residual_dim() {
        let f = ConstantFactor { value: 0.0 };
        assert_eq!(f.residual_dim(), 1);
    }

    // -------------------------------------------------------------------------
    // FactorResult type alias
    // -------------------------------------------------------------------------

    #[test]
    fn test_factor_result_ok() {
        let r: FactorResult<f64> = Ok(1.0);
        assert!(r.is_ok());
    }

    #[test]
    fn test_factor_result_err() {
        let r: FactorResult<f64> = Err(FactorError::InvalidParameters("bad".into()));
        assert!(r.is_err());
    }
}
