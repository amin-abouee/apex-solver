//! Errors raised while reading or writing a trajectory file.

use std::path::PathBuf;

/// Why a trajectory could not be read or written.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TrajectoryError {
    /// The file could not be read or written.
    #[error("failed to access trajectory file {}", path.display())]
    Io {
        /// The file involved.
        path: PathBuf,
        /// Underlying failure.
        #[source]
        source: std::io::Error,
    },

    /// A row had the wrong number of fields.
    #[error("{}: row {line} has {got} fields, expected {expected}", path.display())]
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
    #[error("{}: row {line} has {got} fields, which is neither {a} nor {b}", path.display())]
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
    #[error("{}: row {line}, field {field}: {value:?} is not a number", path.display())]
    InvalidNumber {
        /// The file.
        path: PathBuf,
        /// Row index among non-comment rows, 1-based.
        line: usize,
        /// Which column.
        field: &'static str,
        /// The offending text.
        value: String,
    },

    /// A quaternion was not unit norm.
    ///
    /// Rejected rather than normalised: a quaternion that is far from unit is
    /// evidence the columns were read in the wrong order or the file is
    /// corrupt, and silently repairing it hides both.
    #[error("{}: row {line}: quaternion norm {norm} is not 1", path.display())]
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
    #[error("{}: row {line}: timestamp {value} is negative or not finite", path.display())]
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
    #[error("{}: unsupported trajectory extension {extension:?}", path.display())]
    UnsupportedFormat {
        /// The file.
        path: PathBuf,
        /// Its lower-cased extension, empty when it has none.
        extension: String,
    },
}

/// Result alias for trajectory operations.
pub type Result<T> = std::result::Result<T, TrajectoryError>;

/// Result alias for trajectory operations, named like the other module aliases.
pub type TrajectoryResult<T> = Result<T>;
