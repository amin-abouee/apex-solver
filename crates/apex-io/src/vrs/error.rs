//! Errors for VRS reading and writing.

use std::path::PathBuf;

/// Result alias for VRS operations.
pub type Result<T> = std::result::Result<T, VrsError>;

/// Failure while reading or writing a VRS file.
#[derive(Debug, thiserror::Error)]
pub enum VrsError {
    /// Underlying I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The file does not start with the VRS magic values.
    #[error("'{path}' is not a VRS file")]
    NotVrs {
        /// Offending path.
        path: PathBuf,
    },

    /// A structure on disk is inconsistent or truncated.
    #[error("corrupt VRS file: {reason}")]
    Corrupt {
        /// What was wrong.
        reason: String,
    },

    /// The file uses a feature this implementation does not handle.
    #[error("unsupported VRS feature: {what}")]
    Unsupported {
        /// The unsupported feature.
        what: String,
    },

    /// A record payload failed to decompress.
    #[error("decompression failed: {0}")]
    Decompress(String),

    /// A `DataLayout`/`RecordFormat` description or access was invalid.
    #[error("data layout error: {0}")]
    Layout(String),

    /// A JSON description failed to parse.
    #[error("invalid JSON description: {0}")]
    Json(#[from] serde_json::Error),

    /// Caller error (bad stream, closed writer, ...).
    #[error("{0}")]
    Usage(String),
}

impl VrsError {
    /// Shorthand for [`VrsError::Corrupt`].
    pub fn corrupt(reason: impl Into<String>) -> Self {
        Self::Corrupt {
            reason: reason.into(),
        }
    }

    /// Shorthand for [`VrsError::Unsupported`].
    pub fn unsupported(what: impl Into<String>) -> Self {
        Self::Unsupported { what: what.into() }
    }
}
