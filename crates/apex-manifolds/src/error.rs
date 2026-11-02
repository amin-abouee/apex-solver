//! Errors of the manifold layer.
//!
//! [`ManifoldError`] is the root. Each variant transparently wraps one
//! per-concern enum, so `?` lifts a leaf error to the root with no wrapper text
//! and [`std::error::Error::source`] walks down to the leaf:
//!
//! - [`ElementError`]: a parameter block or constructor argument that is not a
//!   valid group element.
//! - [`TangentError`]: a tangent slice or generator index that does not fit its
//!   space.
//! - [`RnError`]: operands of `Rn` with different dimensions.
//! - [`ManifoldTypeError`]: a manifold name that is not known.
//!
//! The checked constructors build and log these errors once, where the bad
//! input is found. The group operations on the optimization hot path are
//! infallible and never produce them.

/// Result type for manifold operations.
pub type ManifoldResult<T> = Result<T, ManifoldError>;

/// Result of a check on a group element.
pub type ElementResult<T> = Result<T, ElementError>;

/// Result of a check on a tangent vector.
pub type TangentResult<T> = Result<T, TangentError>;

/// Result of an `Rn` operation that needs matching dimensions.
pub type RnResult<T> = Result<T, RnError>;

/// Result of resolving a manifold name.
pub type ManifoldTypeResult<T> = Result<T, ManifoldTypeError>;

/// Root error of the manifold crate.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ManifoldError {
    #[error(transparent)]
    Element(#[from] ElementError),

    #[error(transparent)]
    Tangent(#[from] TangentError),

    #[error(transparent)]
    Rn(#[from] RnError),

    #[error(transparent)]
    ManifoldType(#[from] ManifoldTypeError),
}

/// A value that is not a valid element of its group.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ElementError {
    #[error("{group} parameter block has {actual} values, expected {expected}")]
    ParamLength {
        group: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("rotation quaternion has norm {norm}, expected 1")]
    NonUnitQuaternion { norm: f64 },

    #[error("similarity scale {scale} is not positive")]
    NonPositiveScale { scale: f64 },

    #[error("{group} value contains a NaN or infinite number")]
    NonFinite { group: &'static str },

    #[error("rotation axis has zero length")]
    ZeroAxis,
}

/// A value that does not fit a tangent space.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TangentError {
    #[error("{group} tangent has {actual} values, expected {expected}")]
    TangentLength {
        group: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("{group} tangent contains a NaN or infinite number")]
    NonFinite { group: &'static str },

    #[error("{group} has no generator {index}, its tangent space has {dim} dimensions")]
    GeneratorIndex {
        group: &'static str,
        index: usize,
        dim: usize,
    },
}

/// Operands of the dynamically sized `Rn` that disagree on dimension.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RnError {
    #[error("Rn dimension mismatch, expected {expected} but got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// A manifold name that does not match any [`ManifoldType`](crate::ManifoldType).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ManifoldTypeError {
    #[error("unknown manifold {name}")]
    UnknownManifold { name: String },
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

    fn leaf() -> ElementResult<()> {
        Err(ElementError::NonPositiveScale { scale: -2.0 })
    }

    fn root() -> ManifoldResult<()> {
        leaf()?;
        Ok(())
    }

    #[test]
    fn a_leaf_error_reaches_the_root_with_bare_question_marks() -> TestResult {
        let error = root().err().ok_or("expected an error")?;
        assert_eq!(
            error,
            ManifoldError::Element(ElementError::NonPositiveScale { scale: -2.0 })
        );
        Ok(())
    }

    #[test]
    fn the_chain_shows_the_leaf_message_once() -> TestResult {
        let error = root().err().ok_or("expected an error")?;
        assert_eq!(error.to_string(), "similarity scale -2 is not positive");
        assert_eq!(error.chain(), "similarity scale -2 is not positive");
        Ok(())
    }

    #[test]
    fn every_root_variant_is_transparent() {
        let errors: [ManifoldError; 4] = [
            ElementError::ZeroAxis.into(),
            TangentError::GeneratorIndex {
                group: "SO3",
                index: 5,
                dim: 3,
            }
            .into(),
            RnError::DimensionMismatch {
                expected: 3,
                actual: 2,
            }
            .into(),
            ManifoldTypeError::UnknownManifold {
                name: "SE4".to_string(),
            }
            .into(),
        ];
        let messages: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        assert_eq!(
            messages,
            [
                "rotation axis has zero length",
                "SO3 has no generator 5, its tangent space has 3 dimensions",
                "Rn dimension mismatch, expected 3 but got 2",
                "unknown manifold SE4",
            ]
        );
    }

    #[test]
    fn display_strings_name_their_values() {
        let length = ElementError::ParamLength {
            group: "SE3",
            expected: 7,
            actual: 6,
        };
        assert_eq!(
            length.to_string(),
            "SE3 parameter block has 6 values, expected 7"
        );
        let norm = ElementError::NonUnitQuaternion { norm: 2.0 };
        assert_eq!(
            norm.to_string(),
            "rotation quaternion has norm 2, expected 1"
        );
        let tangent = TangentError::TangentLength {
            group: "SE2",
            expected: 3,
            actual: 4,
        };
        assert_eq!(tangent.to_string(), "SE2 tangent has 4 values, expected 3");
        assert_eq!(
            ElementError::NonFinite { group: "Rn" }.to_string(),
            "Rn value contains a NaN or infinite number"
        );
        assert_eq!(
            TangentError::NonFinite { group: "Rn" }.to_string(),
            "Rn tangent contains a NaN or infinite number"
        );
    }

    #[test]
    fn errors_clone_and_compare() {
        let error = ManifoldError::from(RnError::DimensionMismatch {
            expected: 1,
            actual: 2,
        });
        assert_eq!(error.clone(), error);
        assert_ne!(error, ManifoldError::from(ElementError::ZeroAxis));
    }
}
