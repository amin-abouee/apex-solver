//! Errors raised while reading or writing a pose-graph file.

use std::io;
use thiserror::Error;
use tracing::error;

/// Errors that can occur during graph file parsing
#[derive(Error, Debug)]
pub enum IoError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    #[error("Unsupported vertex type: {0}")]
    UnsupportedVertexType(String),

    #[error("Unsupported edge type: {0}")]
    UnsupportedEdgeType(String),

    #[error("Invalid number format at line {line}: {value}")]
    InvalidNumber { line: usize, value: String },

    #[error("Missing required fields at line {line}")]
    MissingFields { line: usize },

    #[error("Duplicate vertex ID: {id}")]
    DuplicateVertex { id: usize },

    #[error("Invalid quaternion at line {line}: norm = {norm:.6}, expected ~1.0")]
    InvalidQuaternion { line: usize, norm: f64 },

    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("Failed to create file '{path}': {reason}")]
    FileCreationFailed { path: String, reason: String },
}

impl IoError {
    /// Log the error using tracing::error and return self for chaining
    pub fn log(self) -> Self {
        error!("{}", self);
        self
    }

    /// Log the error with source error information using tracing::error and return self for chaining
    pub fn log_with_source<E: std::fmt::Debug>(self, source_error: E) -> Self {
        error!("{} | Source: {:?}", self, source_error);
        self
    }
}
