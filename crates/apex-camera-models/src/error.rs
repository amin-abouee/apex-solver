//! Errors of the camera-model crate.
//!
//! [`CameraModelError`] is the root. Each variant transparently wraps one
//! per-concern enum, so `?` lifts a leaf error to the root with no wrapper text
//! and [`std::error::Error::source`] walks down to the leaf:
//!
//! - [`ParamError`]: intrinsics or a distortion model that do not form a valid
//!   camera (bad values, wrong slice length, wrong distortion variant).
//! - [`ProjectionError`]: a camera-frame point outside the model's projection
//!   domain.
//! - [`UnprojectionError`]: a pixel the model cannot turn back into a ray.
//! - [`EstimationError`]: a failed linear estimation of the distortion.
//! - [`ConversionError`]: a failed fit of one model to another.
//!
//! Projection and unprojection run once per observation and residual, so their
//! errors are expected outcomes: they carry only `Copy` fields, never allocate
//! and are never logged where they are built. The callers that turn them into
//! a decision log them if they abort something. Parameter, estimation and
//! conversion errors are logged once by the public constructor that rejects
//! the input.

/// Result type for any camera-model operation.
pub type CameraModelResult<T> = Result<T, CameraModelError>;

/// Result of a check on camera parameters.
pub type ParamResult<T> = Result<T, ParamError>;

/// Result of a projection.
pub type ProjectionResult<T> = Result<T, ProjectionError>;

/// Result of an unprojection.
pub type UnprojectionResult<T> = Result<T, UnprojectionError>;

/// Result of a linear estimation.
pub type EstimationResult<T> = Result<T, EstimationError>;

/// Result of a camera-model conversion.
pub type ConversionResult<T> = Result<T, ConversionError>;

/// Root error of the camera-model crate.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CameraModelError {
    #[error(transparent)]
    Param(#[from] ParamError),

    #[error(transparent)]
    Projection(#[from] ProjectionError),

    #[error(transparent)]
    Unprojection(#[from] UnprojectionError),

    #[error(transparent)]
    Estimation(#[from] EstimationError),

    #[error(transparent)]
    Conversion(#[from] ConversionError),
}

/// Intrinsics or a distortion model that do not form a valid camera.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ParamError {
    #[error("focal length must be positive: fx={fx}, fy={fy}")]
    FocalLengthNotPositive { fx: f64, fy: f64 },

    #[error("focal length must be finite: fx={fx}, fy={fy}")]
    FocalLengthNotFinite { fx: f64, fy: f64 },

    #[error("principal point must be finite: cx={cx}, cy={cy}")]
    PrincipalPointNotFinite { cx: f64, cy: f64 },

    #[error("distortion coefficient {name} must be finite, got {value}")]
    DistortionNotFinite { name: &'static str, value: f64 },

    #[error("parameter {param} must be in range [{min}, {max}], got {value}")]
    ParameterOutOfRange {
        param: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },

    #[error("{model} has {expected} parameters, got {actual}")]
    ParamCount {
        model: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("{model} needs DistortionModel::{expected}, got DistortionModel::{actual}")]
    DistortionVariant {
        model: &'static str,
        expected: &'static str,
        actual: &'static str,
    },

    #[error("{model} has a single focal length, but fx={fx} and fy={fy}")]
    UnequalFocalLengths {
        model: &'static str,
        fx: f64,
        fy: f64,
    },

    #[error("{model} has its principal point at the origin, got cx={cx}, cy={cy}")]
    PrincipalPointNotZero {
        model: &'static str,
        cx: f64,
        cy: f64,
    },

    #[error("fisheye624 has non-zero thin-prism terms, fisheye62 cannot represent it")]
    ThinPrismNotZero,
}

/// A camera-frame point outside the projection domain of the model.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ProjectionError {
    #[error("point behind camera: z={z} (must be > {min_z})")]
    PointBehindCamera { z: f64, min_z: f64 },

    #[error("point at camera center: 3D point too close to optical axis")]
    PointAtCameraCenter,

    #[error("projection denominator too small: denom={denom} (threshold={threshold})")]
    DenominatorTooSmall { denom: f64, threshold: f64 },

    #[error("point with z={z} is outside the {model} projection domain")]
    OutOfDomain { model: &'static str, z: f64 },

    #[error("radial distortion factor {radial} is not positive, the lens model folds over")]
    DistortionFold { radial: f64 },
}

/// A pixel the model cannot turn back into a ray.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum UnprojectionError {
    #[error("point outside image bounds: ({x}, {y}) not in valid region")]
    PointOutsideImage { x: f64, y: f64 },

    #[error("normalized radius {radius} is outside the model's field of view (at most {max})")]
    RadiusOutOfDomain { radius: f64, max: f64 },

    #[error("unprojection did not converge after {iterations} iterations, residual {residual}")]
    NotConverged { iterations: usize, residual: f64 },

    #[error("unprojection diverged to incidence angle {theta}")]
    Diverged { theta: f64 },

    #[error("singular distortion Jacobian in unprojection")]
    SingularJacobian,

    #[error("unprojection denominator too small: denom={denom} (threshold={threshold})")]
    DenominatorTooSmall { denom: f64, threshold: f64 },

    #[error("pixel coordinates are not finite")]
    NonFinitePixel,
}

/// A failed linear estimation of the distortion parameters.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum EstimationError {
    #[error("{points_3d} 3D points but {points_2d} 2D points")]
    PointCountMismatch { points_3d: usize, points_2d: usize },

    #[error("{got} point correspondences, linear estimation needs at least {need}")]
    TooFewPoints { got: usize, need: usize },

    #[error("focal length {focal} is too small for linear estimation")]
    DegenerateFocalLength { focal: f64 },

    #[error("linear least-squares solve failed: {reason}")]
    SvdSolve { reason: &'static str },

    #[error("estimated parameters are not a valid camera")]
    InvalidEstimate(#[from] ParamError),
}

/// A failed fit of one camera model to another.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ConversionError {
    #[error("{got} usable rays for {need} parameters")]
    TooFewRays { got: usize, need: usize },

    #[error("the initial camera is not valid")]
    InvalidInitial(#[from] ParamError),

    #[error("the initial camera cannot project sample {sample}")]
    InitialCannotProject { sample: usize },
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

    use std::error::Error;

    type TestResult = Result<(), Box<dyn Error>>;

    fn leaf() -> ParamResult<()> {
        Err(ParamError::ParameterOutOfRange {
            param: "alpha",
            value: 1.5,
            min: 0.0,
            max: 1.0,
        })
    }

    fn estimate() -> EstimationResult<()> {
        leaf()?;
        Ok(())
    }

    fn root() -> CameraModelResult<()> {
        estimate()?;
        Ok(())
    }

    #[test]
    fn a_leaf_error_reaches_the_root_with_bare_question_marks() -> TestResult {
        let error = root().err().ok_or("expected an error")?;
        assert!(matches!(
            error,
            CameraModelError::Estimation(EstimationError::InvalidEstimate(
                ParamError::ParameterOutOfRange { param: "alpha", .. }
            ))
        ));
        Ok(())
    }

    #[test]
    fn the_chain_shows_each_hop_once() -> TestResult {
        let error = root().err().ok_or("expected an error")?;
        assert_eq!(
            error.chain(),
            "estimated parameters are not a valid camera\n  \
             → parameter alpha must be in range [0, 1], got 1.5"
        );
        assert_eq!(
            error.chain_compact(),
            "estimated parameters are not a valid camera → \
             parameter alpha must be in range [0, 1], got 1.5"
        );
        Ok(())
    }

    #[test]
    fn every_root_variant_is_transparent() {
        let errors: [CameraModelError; 5] = [
            ParamError::ThinPrismNotZero.into(),
            ProjectionError::PointAtCameraCenter.into(),
            UnprojectionError::NonFinitePixel.into(),
            EstimationError::TooFewPoints { got: 1, need: 4 }.into(),
            ConversionError::TooFewRays { got: 3, need: 6 }.into(),
        ];
        let messages: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        assert_eq!(
            messages,
            [
                "fisheye624 has non-zero thin-prism terms, fisheye62 cannot represent it",
                "point at camera center: 3D point too close to optical axis",
                "pixel coordinates are not finite",
                "1 point correspondences, linear estimation needs at least 4",
                "3 usable rays for 6 parameters",
            ]
        );
    }

    #[test]
    fn display_strings_name_their_values() {
        let count = ParamError::ParamCount {
            model: "pinhole",
            expected: 4,
            actual: 5,
        };
        assert_eq!(count.to_string(), "pinhole has 4 parameters, got 5");
        let variant = ParamError::DistortionVariant {
            model: "fov",
            expected: "FOV",
            actual: "None",
        };
        assert_eq!(
            variant.to_string(),
            "fov needs DistortionModel::FOV, got DistortionModel::None"
        );
        let behind = ProjectionError::PointBehindCamera {
            z: -0.5,
            min_z: 1e-6,
        };
        assert_eq!(
            behind.to_string(),
            "point behind camera: z=-0.5 (must be > 0.000001)"
        );
        let converged = UnprojectionError::NotConverged {
            iterations: 10,
            residual: 0.25,
        };
        assert_eq!(
            converged.to_string(),
            "unprojection did not converge after 10 iterations, residual 0.25"
        );
    }

    #[test]
    fn a_conversion_error_keeps_its_parameter_source() -> TestResult {
        let error = ConversionError::from(ParamError::FocalLengthNotPositive { fx: -1.0, fy: 2.0 });
        let source = error.source().ok_or("expected a source")?;
        assert_eq!(
            source.to_string(),
            "focal length must be positive: fx=-1, fy=2"
        );
        Ok(())
    }

    #[test]
    fn hot_path_errors_are_copy() {
        fn is_copy<T: Copy>() {}
        is_copy::<ProjectionError>();
        is_copy::<UnprojectionError>();
    }
}
