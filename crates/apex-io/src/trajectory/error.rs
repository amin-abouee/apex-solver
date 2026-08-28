//! Errors raised while reading or writing a trajectory file.

use std::path::PathBuf;

/// Why a trajectory could not be read or written.
#[derive(Debug, thiserror::Error)]
pub enum TrajectoryError {
    /// The file could not be read or written.
    #[error("io error on '{path}': {source}")]
    Io {
        /// The file involved.
        path: PathBuf,
        /// Underlying failure.
        #[source]
        source: std::io::Error,
    },

    /// A row had the wrong number of fields.
    #[error("{path}: row {line} has {got} fields, expected {expected}")]
    MissingColumns {
        /// The file.
        path: PathBuf,
        /// Row index among non-comment rows, 1-based.
        line: usize,
        /// Fields required.
        expected: usize,
        /// Fields present.
        got: usize,
    },

    /// A row had a field count matching no known layout.
    ///
    /// Distinct from [`Self::MissingColumns`]: accepting a 9-column ASL row by
    /// reading its first 8 fields is exactly how the inertial columns went
    /// unnoticed for so long.
    #[error("{path}: row {line} has {got} fields, which is neither {a} nor {b}")]
    UnexpectedColumnCount {
        /// The file.
        path: PathBuf,
        /// Row index among non-comment rows, 1-based.
        line: usize,
        /// Fields present.
        got: usize,
        /// One accepted layout.
        a: usize,
        /// The other accepted layout.
        b: usize,
    },

    /// A field did not parse as a number.
    #[error("{path}: row {line}, field '{field}': '{value}' is not a number")]
    InvalidNumber {
        /// The file.
        path: PathBuf,
        /// Row index among non-comment rows, 1-based.
        line: usize,
        /// Which column.
        field: String,
        /// The offending text.
        value: String,
    },

    /// A quaternion was not unit norm.
    ///
    /// Rejected rather than normalised: a quaternion that is far from unit is
    /// evidence the columns were read in the wrong order or the file is
    /// corrupt, and silently repairing it hides both.
    #[error("{path}: row {line}: quaternion norm {norm} is not 1")]
    InvalidQuaternion {
        /// The file.
        path: PathBuf,
        /// Row index among non-comment rows, 1-based.
        line: usize,
        /// The norm actually found.
        norm: f64,
    },

    /// A timestamp was negative or non-finite.
    ///
    /// Trajectory timestamps are `u64` nanoseconds since the epoch, so there is
    /// no representation for a negative value and wrapping would produce a
    /// plausible far-future stamp.
    #[error("{path}: row {line}: timestamp {value} is negative or not finite")]
    NegativeTimestamp {
        /// The file.
        path: PathBuf,
        /// Row index among non-comment rows, 1-based.
        line: usize,
        /// The offending value.
        value: f64,
    },

    /// Parallel inputs disagreed in length.
    #[error("{what}: {got} supplied, {expected} required")]
    LengthMismatch {
        /// What was mismatched.
        what: &'static str,
        /// Length supplied.
        got: usize,
        /// Length required.
        expected: usize,
    },

    /// A 17-column ASL write was asked for on a pose-only trajectory.
    ///
    /// Padding the velocity and bias columns with zeros would claim the body
    /// was stationary with perfect sensors — a measurement nobody made.
    #[error("cannot write EuRoC 17-column layout: this trajectory has no inertial columns")]
    MissingInertialColumns,

    /// The file extension named no known format.
    #[error("unsupported trajectory format: '{0}'")]
    UnsupportedFormat(String),
}

/// Result alias for trajectory operations.
pub type Result<T> = std::result::Result<T, TrajectoryError>;
