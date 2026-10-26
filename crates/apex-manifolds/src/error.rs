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

#[cfg(test)]
mod tests {
    use super::*;
    use apex_error::ErrorChainExt;
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
