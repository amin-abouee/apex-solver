//! Error types of the apex-solver library.
//!
//! [`ApexSolverError`] is the root. Each variant transparently wraps the error
//! of one module, so `?` lifts a module error to the root with no wrapper text
//! and [`std::error::Error::source`] walks down from there:
//!
//! - `CoreError`: problem construction, registration and variable
//!   bookkeeping; wraps `FactorError`, `ManifoldError` and `LinearizerError`.
//! - `FactorError`: factor construction and `Factor::validate_variables`.
//! - `LinearizerError`: residual and Jacobian assembly, including a
//!   non-finite residual at a linearization point.
//! - `LinAlgError`: factorizations and Schur solvers, with the faer failure
//!   as its source.
//! - `OptimizerError`: the optimizers; wraps `LinAlgError`, `CoreError` and
//!   `LinearizerError`.
//! - `MarginalizationError`, `CovarianceError`, `PreintegrationError`,
//!   `ObserverError`, plus the dependency roots `ManifoldError`, `IoError` and
//!   `CameraModelError`.
//!
//! Modules return their own error, never the root. A module error that wraps
//! another adds a short context message ("linear solve failed") and keeps the
//! wrapped one as its source, so [`ErrorChainExt::chain`] prints every hop
//! exactly once:
//!
//! ```text
//! linear solve failed
//!   → factorization of the normal equations failed, the matrix is singular or not positive definite
//!   → NonPositivePivot { index: 3 }
//! ```
//!
//! Errors are logged once, where they are built, with
//! [`ErrorLogging::log`]; everything above propagates them with a bare `?`.
//! Expected per-residual outcomes (a point that does not project, a scan point
//! outside a distance field) are the exception: they are returned unlogged
//! and the caller decides.

use crate::{
    core::CoreError, core::marginalization::MarginalizationError, factors::FactorError,
    factors::imu::PreintegrationError, linalg::LinAlgError, linalg::covariance::CovarianceError,
    linearizer::LinearizerError, observers::ObserverError, optimizer::OptimizerError,
};
use apex_camera_models::CameraModelError;
use apex_io::IoError;
use apex_manifolds::ManifoldError;
use thiserror::Error;

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

pub use logging::{ErrorChainExt, ErrorLogging};

/// Log-at-origin and error-chain helpers. Each crate keeps its own copy, so no
/// crate depends on another for them.
///
/// - [`ErrorLogging`]: log an error exactly once, where it is first built, with
///   the caller's file and line; everything above propagates it with a bare `?`.
///   An expected outcome a caller may retry is not logged where it is built;
///   the caller that gives up on it logs it once with [`ErrorLogging::log_chain`].
/// - [`ErrorChainExt`]: render an error and its `source()` chain at the
///   boundary that reports it.
mod logging {
    use std::error::Error;
    use std::fmt::Debug;
    use std::panic::Location;

    /// Log an error once, at the site that constructs it, and hand it back.
    ///
    /// The blanket impl covers every `std::error::Error`, not every `Display`
    /// type: Lie groups and tangents are `Display` and have a `log` method of
    /// their own, and a `Display` blanket impl would capture their calls.
    pub trait ErrorLogging: Error + Sized {
        /// Emit `self` at `error` level with the caller's location and return it.
        #[track_caller]
        fn log(self) -> Self {
            let at = Location::caller();
            tracing::error!(file = at.file(), line = at.line(), "{self}");
            self
        }

        /// Like [`ErrorLogging::log`], with an underlying cause that has no
        /// `#[from]` edge (a third-party error kept only as context).
        #[track_caller]
        fn log_with_source<S: Debug>(self, source: S) -> Self {
            let at = Location::caller();
            tracing::error!(file = at.file(), line = at.line(), source = ?source, "{self}");
            self
        }

        /// Like [`ErrorLogging::log`], with the whole `source()` chain as the
        /// message (`outer → cause → root cause`). For an error logged by the
        /// caller that gives up on it rather than where it was built.
        #[track_caller]
        fn log_chain(self) -> Self {
            let at = Location::caller();
            tracing::error!(
                file = at.file(),
                line = at.line(),
                "{}",
                self.chain_compact()
            );
            self
        }
    }

    impl<T: Error> ErrorLogging for T {}

    /// Render an error together with every error in its `source()` chain.
    pub trait ErrorChainExt: Error {
        /// One line per hop, outermost first: `outer\n  → cause\n  → root cause`.
        fn chain(&self) -> String {
            let mut text = self.to_string();
            let mut source = self.source();
            while let Some(cause) = source {
                text.push_str("\n  → ");
                text.push_str(&cause.to_string());
                source = cause.source();
            }
            text
        }

        /// The same hops on one line: `outer → cause → root cause`.
        fn chain_compact(&self) -> String {
            let mut text = self.to_string();
            let mut source = self.source();
            while let Some(cause) = source {
                text.push_str(" → ");
                text.push_str(&cause.to_string());
                source = cause.source();
            }
            text
        }
    }

    impl<T: Error + ?Sized> ErrorChainExt for T {}

    #[cfg(test)]
    mod tests {
        use super::*;

        #[derive(Debug, thiserror::Error)]
        #[error("leaf failed at {at}")]
        struct Leaf {
            at: usize,
        }

        #[derive(Debug, thiserror::Error)]
        enum Middle {
            #[error("middle step failed")]
            Leaf(#[from] Leaf),
        }

        #[derive(Debug, thiserror::Error)]
        enum Root {
            #[error(transparent)]
            Middle(#[from] Middle),
        }

        fn leaf() -> Result<(), Leaf> {
            Err(Leaf { at: 7 }.log())
        }

        fn middle() -> Result<(), Middle> {
            leaf()?;
            Ok(())
        }

        fn root() -> Result<(), Root> {
            middle()?;
            Ok(())
        }

        #[test]
        fn the_chain_lists_every_hop_once() {
            let chain = root().err().map(|error| error.chain());
            assert_eq!(
                chain.as_deref(),
                Some("middle step failed\n  → leaf failed at 7")
            );
        }

        #[test]
        fn the_compact_chain_is_one_line() {
            let chain = root().err().map(|error| error.chain_compact());
            assert_eq!(
                chain.as_deref(),
                Some("middle step failed → leaf failed at 7")
            );
        }

        #[test]
        fn logging_returns_the_same_error() {
            let error = Leaf { at: 2 }.log_with_source("context");
            assert_eq!(error.at, 2);
        }

        #[test]
        fn logging_the_chain_returns_the_same_error() {
            let error = Middle::from(Leaf { at: 4 }).log_chain();
            assert_eq!(
                error.chain_compact(),
                "middle step failed → leaf failed at 4"
            );
        }

        #[test]
        fn a_boxed_error_has_a_chain_too() {
            let boxed: Box<dyn Error> = Box::new(Middle::from(Leaf { at: 1 }));
            assert_eq!(
                boxed.chain_compact(),
                "middle step failed → leaf failed at 1"
            );
        }
    }
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
        assert_eq!(crate::error::ErrorChainExt::chain_compact(&error), expected);
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
