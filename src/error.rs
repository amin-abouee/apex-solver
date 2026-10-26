//! Error types for the apex-solver library
//!
//! This module provides the main error and result types used throughout the library.
//! All errors use the `thiserror` crate for automatic trait implementations.
//!
//! # Error Hierarchy
//!
//! The library uses a strict three-layer error hierarchy with bubble-up propagation
//! using the `?` operator:
//!
//! - **Layer A (Top/API)**: `ApexSolverError` — exposed to end-users, wraps all module errors
//! - **Layer B (Logic)**: `OptimizerError`, `ObserverError` — wrap Layer C errors with context
//! - **Layer C (Deep/Math)**: `CoreError`, `LinAlgError`, `ManifoldError`, `FactorError` —
//!   module-specific errors that must **never** return `ApexSolverError` directly
//!
//! # Error Propagation Convention
//!
//! Deep modules (Layer C) must return their own module-specific error type
//! (e.g., `Result<T, CoreError>`, not `Result<T, ApexSolverError>`). The `?` operator
//! and `#[from]` attributes handle automatic conversions at each layer boundary.
//!
//! Example error chain (Layer C → B → A):
//!
//! ```text
//! LinAlgError::SingularMatrix
//!   → OptimizerError::LinAlg (via #[from] at Layer B)
//!     → ApexSolverError::Optimizer (via #[from] at Layer A)
//! ```
//!
//! # Dual-Path Convention for LinAlgError
//!
//! `LinAlgError` can convert to `ApexSolverError` via two paths:
//! 1. Direct: `LinAlgError → ApexSolverError::LinearAlgebra(...)` — for standalone linalg usage
//! 2. Through optimizer: `LinAlgError → OptimizerError::LinAlg → ApexSolverError::Optimizer(...)` — during optimization
//!
//! When a `LinAlgError` occurs inside the optimizer, it should propagate through the
//! optimizer layer (path 2) to preserve optimization context. Standalone linalg usage
//! should use path 1 directly.

use crate::{
    core::CoreError, core::marginalization::MarginalizationError, factors::FactorError,
    factors::imu::PreintegrationError, linalg::LinAlgError, linalg::covariance::CovarianceError,
    linearizer::LinearizerError, observers::ObserverError, optimizer::OptimizerError,
};
use apex_camera_models::CameraModelError;
use apex_io::IoError;
use apex_manifolds::ManifoldError;
use thiserror::Error;

pub use apex_error::{ErrorChainExt, ErrorLogging};

/// Main result type used throughout the apex-solver library
pub type ApexSolverResult<T> = Result<T, ApexSolverError>;

/// Root error of the apex-solver library.
///
/// Every variant transparently wraps one module error, so `Display` prints the
/// module's own message and [`std::error::Error::source`] walks down from
/// there. Render the whole chain with [`ErrorChainExt::chain`] or
/// [`ErrorChainExt::chain_compact`]:
///
/// ```
/// use apex_solver::error::{ApexSolverError, ErrorChainExt};
/// use apex_solver::apex_manifolds::{ManifoldError, RnError};
///
/// let leaf = RnError::DimensionMismatch { expected: 3, actual: 2 };
/// let error = ApexSolverError::from(ManifoldError::from(leaf));
/// assert_eq!(
///     error.chain_compact(),
///     "Rn dimension mismatch, expected 3 but got 2"
/// );
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ApexSolverError {
    /// Problem construction and variable bookkeeping.
    #[error(transparent)]
    Core(#[from] CoreError),

    /// Optimization algorithms.
    #[error(transparent)]
    Optimizer(#[from] OptimizerError),

    /// Linear solvers and factorizations.
    #[error(transparent)]
    LinearAlgebra(#[from] LinAlgError),

    /// Manifold operations.
    #[error(transparent)]
    Manifold(#[from] ManifoldError),

    /// File loading and parsing.
    #[error(transparent)]
    Io(#[from] IoError),

    /// Observers and visualization.
    #[error(transparent)]
    Observer(#[from] ObserverError),

    /// Factor construction and registration.
    #[error(transparent)]
    Factor(#[from] FactorError),

    /// Residual and Jacobian assembly.
    #[error(transparent)]
    Linearizer(#[from] LinearizerError),

    /// Camera models.
    #[error(transparent)]
    Camera(#[from] CameraModelError),

    /// Marginalization of variables into a prior.
    #[error(transparent)]
    Marginalization(#[from] MarginalizationError),

    /// Covariance estimation.
    #[error(transparent)]
    Covariance(#[from] CovarianceError),

    /// IMU preintegration.
    #[error(transparent)]
    Preintegration(#[from] PreintegrationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_camera_models::{
        CameraModelResult, ConversionError, ConversionResult, EstimationError, ParamError,
        ParamResult, ProjectionError, UnprojectionError,
    };
    use apex_manifolds::RnError;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn a_factor_error_reaches_the_root_through_core_with_its_context() -> TestResult {
        fn validate() -> crate::factors::FactorResult<()> {
            Err(FactorError::VariableCount {
                factor: "GpsFactor",
                expected: 2,
                actual: 3,
            })
        }
        fn register() -> crate::core::CoreResult<()> {
            validate()?;
            Ok(())
        }
        fn api() -> ApexSolverResult<()> {
            register()?;
            Ok(())
        }
        let error = api().err().ok_or("expected an error")?;
        assert!(matches!(
            error,
            ApexSolverError::Core(CoreError::Factor(FactorError::VariableCount { .. }))
        ));
        assert_eq!(
            error.chain(),
            "residual block registration rejected\n  → GpsFactor expects 2 variables, got 3"
        );
        Ok(())
    }

    #[test]
    fn a_manifold_error_converts_transparently() {
        let leaf = RnError::DimensionMismatch {
            expected: 3,
            actual: 2,
        };
        let error = ApexSolverError::from(ManifoldError::from(leaf));
        assert!(matches!(error, ApexSolverError::Manifold(_)));
        assert_eq!(
            error.to_string(),
            "Rn dimension mismatch, expected 3 but got 2"
        );
    }

    // ========================================================================
    // CameraModelError → ApexSolverError::Camera (transparent wrap)
    // ========================================================================

    #[test]
    fn test_camera_error_point_behind_camera_direct() -> TestResult {
        let cam_err = CameraModelError::from(ProjectionError::PointBehindCamera {
            z: -0.5,
            min_z: 1e-6,
        });
        let apex_err: ApexSolverError = cam_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Camera(_)),
            "Expected Camera variant, got {:?}",
            apex_err
        );
        let compact = apex_err.chain_compact();
        assert!(compact.contains("behind camera"), "compact: {}", compact);
        assert!(
            compact.contains("z=-0.5"),
            "compact should preserve structured field z: {}",
            compact
        );
        Ok(())
    }

    #[test]
    fn test_camera_error_focal_length_preserves_fields() -> TestResult {
        let cam_err = CameraModelError::from(ParamError::FocalLengthNotPositive {
            fx: -1.0,
            fy: 500.0,
        });
        let apex_err: ApexSolverError = cam_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Camera(_)),
            "Expected Camera variant, got {:?}",
            apex_err
        );
        let msg = apex_err.to_string();
        assert!(msg.contains("fx=-1"), "msg should contain fx: {}", msg);
        assert!(msg.contains("fy=500"), "msg should contain fy: {}", msg);
        Ok(())
    }

    #[test]
    fn test_camera_error_numerical_preserves_fields() -> TestResult {
        let cam_err = CameraModelError::from(ProjectionError::DenominatorTooSmall {
            denom: 1e-15,
            threshold: 1e-6,
        });
        let apex_err: ApexSolverError = cam_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Camera(_)),
            "Expected Camera variant, got {:?}",
            apex_err
        );
        let msg = apex_err.to_string();
        assert!(msg.contains("denom"), "msg: {}", msg);
        assert!(
            msg.contains("threshold"),
            "msg should contain threshold: {}",
            msg
        );
        Ok(())
    }

    #[test]
    fn test_camera_error_parameter_out_of_range() -> TestResult {
        let cam_err = CameraModelError::from(ParamError::ParameterOutOfRange {
            param: "alpha",
            value: 1.5,
            min: 0.0,
            max: 1.0,
        });
        let apex_err: ApexSolverError = cam_err.into();
        assert!(
            matches!(apex_err, ApexSolverError::Camera(_)),
            "Expected Camera variant, got {:?}",
            apex_err
        );
        let msg = apex_err.to_string();
        assert!(msg.contains("alpha"), "msg: {}", msg);
        assert!(msg.contains("1.5"), "msg should preserve value: {}", msg);
        Ok(())
    }

    /// A camera leaf error reaches `ApexSolverError` through every hop with a
    /// bare `?`, and each hop's message appears once in the chain.
    #[test]
    fn test_camera_error_chain_from_leaf_to_root() -> TestResult {
        fn validate() -> ParamResult<()> {
            Err(ParamError::ParamCount {
                model: "pinhole",
                expected: 4,
                actual: 5,
            })
        }
        fn convert() -> ConversionResult<()> {
            validate()?;
            Ok(())
        }
        fn camera() -> CameraModelResult<()> {
            convert()?;
            Ok(())
        }
        fn solver() -> ApexSolverResult<()> {
            camera()?;
            Ok(())
        }
        let error = solver().err().ok_or("expected an error")?;
        assert!(matches!(
            error,
            ApexSolverError::Camera(CameraModelError::Conversion(
                ConversionError::InvalidInitial(ParamError::ParamCount { .. })
            ))
        ));
        let expected = "the initial camera is not valid → pinhole has 4 parameters, got 5";
        assert_eq!(apex_error::ErrorChainExt::chain_compact(&error), expected);
        assert_eq!(error.chain_compact(), expected);
        Ok(())
    }

    #[test]
    fn test_camera_error_all_variants_accessible() -> TestResult {
        let errors: Vec<ApexSolverError> = vec![
            CameraModelError::from(ProjectionError::PointBehindCamera {
                z: -0.5,
                min_z: 1e-6,
            })
            .into(),
            CameraModelError::from(ProjectionError::PointAtCameraCenter).into(),
            CameraModelError::from(ProjectionError::OutOfDomain {
                model: "fov",
                z: 0.0,
            })
            .into(),
            CameraModelError::from(ProjectionError::DenominatorTooSmall {
                denom: 1e-15,
                threshold: 1e-6,
            })
            .into(),
            CameraModelError::from(ProjectionError::DistortionFold { radial: -0.1 }).into(),
            CameraModelError::from(UnprojectionError::PointOutsideImage { x: 100.0, y: 200.0 })
                .into(),
            CameraModelError::from(UnprojectionError::NotConverged {
                iterations: 10,
                residual: 0.5,
            })
            .into(),
            CameraModelError::from(UnprojectionError::SingularJacobian).into(),
            CameraModelError::from(ParamError::FocalLengthNotPositive {
                fx: -1.0,
                fy: 500.0,
            })
            .into(),
            CameraModelError::from(ParamError::FocalLengthNotFinite {
                fx: f64::INFINITY,
                fy: 500.0,
            })
            .into(),
            CameraModelError::from(ParamError::PrincipalPointNotFinite {
                cx: f64::NAN,
                cy: 240.0,
            })
            .into(),
            CameraModelError::from(ParamError::DistortionNotFinite {
                name: "k1",
                value: f64::NAN,
            })
            .into(),
            CameraModelError::from(ParamError::ParameterOutOfRange {
                param: "alpha",
                value: 1.5,
                min: 0.0,
                max: 1.0,
            })
            .into(),
            CameraModelError::from(ParamError::DistortionVariant {
                model: "fov",
                expected: "FOV",
                actual: "None",
            })
            .into(),
            CameraModelError::from(EstimationError::SvdSolve { reason: "rank" }).into(),
            CameraModelError::from(ConversionError::TooFewRays { got: 1, need: 6 }).into(),
        ];

        for err in &errors {
            assert!(matches!(err, ApexSolverError::Camera(_)));
            assert!(
                !err.to_string().is_empty(),
                "Error Display should not be empty"
            );
            assert!(
                !err.chain_compact().is_empty(),
                "chain_compact should not be empty"
            );
        }
        Ok(())
    }
}
