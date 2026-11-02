//! Core optimization components for the apex-solver library
//!
//! This module contains the fundamental building blocks for nonlinear least squares optimization:
//! - Problem formulation and management
//! - Residual blocks
//! - Variables and manifold handling
//! - Loss functions for robust estimation
//! - Correctors for applying loss functions

pub mod corrector;
pub mod loss_functions;
pub mod marginalization;
pub mod noise;
pub mod problem;
pub mod residual_block;
pub mod variable;

use crate::factors::FactorError;
use crate::linearizer::LinearizerError;
use apex_manifolds::ManifoldError;
use slotmap::new_key_type;
use thiserror::Error;

new_key_type! {
    /// Stable generational handle for a variable in the Problem.
    pub struct VarKey;
    /// Stable generational handle for a residual block (factor) in the Problem.
    pub struct FactorKey;
}

/// Errors of problem construction and variable bookkeeping.
///
/// Built and logged once where the bad input is found. The wrapped module
/// errors were logged by their own origin and are carried here only for the
/// context the hop adds.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum CoreError {
    /// A key that is not (or no longer) in the problem.
    #[error("unknown variable key {key:?}")]
    UnknownVariable { key: VarKey },

    /// A variable that residual blocks still reference.
    #[error("variable {key:?} is still used by {blocks} residual block(s); remove them first")]
    VariableInUse { key: VarKey, blocks: usize },

    /// A value whose length disagrees with the slot it is written to.
    #[error("{what} has {actual} values, expected {expected}")]
    DimensionMismatch {
        what: &'static str,
        expected: usize,
        actual: usize,
    },

    /// A tangent index past the variable's degrees of freedom.
    #[error("index {index} is out of range for a variable with {dof} degrees of freedom")]
    IndexOutOfRange { index: usize, dof: usize },

    /// Bounds that are not an ordered pair of numbers.
    #[error("bounds [{lower}, {upper}] on index {index} are not an ordered pair of numbers")]
    InvalidBounds {
        index: usize,
        lower: f64,
        upper: f64,
    },

    /// A variable type that cannot hold a frozen linearization point.
    #[error("this variable does not support a frozen linearization point")]
    FreezeUnsupported,

    /// A self-whitening factor registered with a noise model as well.
    #[error(
        "this factor applies its own square-root information; register it with \
         NoiseModel::null(), a second noise model would whiten the residual twice"
    )]
    DoubleWhitening,

    /// A scalar parameter outside its admissible range.
    #[error("{owner} {param} must be {requirement}, got {value}")]
    InvalidParameter {
        owner: &'static str,
        param: &'static str,
        requirement: &'static str,
        value: f64,
    },

    /// A matrix argument of the wrong shape or with non-finite entries.
    #[error("{what} is {rows}×{cols}; it must be non-empty, square and finite")]
    InvalidMatrix {
        what: &'static str,
        rows: usize,
        cols: usize,
    },

    /// A name that matches no registered option.
    #[error("unknown {kind} {name:?}, expected one of: {valid}")]
    UnknownName {
        kind: &'static str,
        name: String,
        valid: String,
    },

    /// The factor rejected the variables it was registered with.
    #[error("residual block registration rejected")]
    Factor(#[from] FactorError),

    /// A parameter block that is not a valid element of its manifold.
    #[error("invalid variable value")]
    Manifold(#[from] ManifoldError),

    /// Residual or Jacobian assembly failed.
    #[error("problem linearization failed")]
    Linearizer(#[from] LinearizerError),
}

/// Result type for core module operations
pub type CoreResult<T> = Result<T, CoreError>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{ApexSolverError, ApexSolverResult, ErrorChainExt};
    use crate::factors::FactorResult;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn validate() -> FactorResult<()> {
        Err(FactorError::VariableCount {
            factor: "BetweenFactor",
            expected: 2,
            actual: 1,
        })
    }

    fn register() -> CoreResult<()> {
        validate()?;
        Ok(())
    }

    fn api() -> ApexSolverResult<()> {
        register()?;
        Ok(())
    }

    #[test]
    fn a_factor_error_reaches_the_root_through_core() -> TestResult {
        let error = api().err().ok_or("expected an error")?;
        assert!(matches!(
            error,
            ApexSolverError::Core(CoreError::Factor(FactorError::VariableCount { .. }))
        ));
        assert_eq!(
            error.chain_compact(),
            "residual block registration rejected → BetweenFactor expects 2 variables, got 1"
        );
        Ok(())
    }

    #[test]
    fn display_strings_name_their_values() {
        let mismatch = CoreError::DimensionMismatch {
            what: "parameter block",
            expected: 7,
            actual: 6,
        };
        assert_eq!(
            mismatch.to_string(),
            "parameter block has 6 values, expected 7"
        );
        let index = CoreError::IndexOutOfRange { index: 7, dof: 3 };
        assert_eq!(
            index.to_string(),
            "index 7 is out of range for a variable with 3 degrees of freedom"
        );
        let parameter = CoreError::InvalidParameter {
            owner: "HuberLoss",
            param: "scale",
            requirement: "finite and positive",
            value: -1.0,
        };
        assert_eq!(
            parameter.to_string(),
            "HuberLoss scale must be finite and positive, got -1"
        );
    }
}
