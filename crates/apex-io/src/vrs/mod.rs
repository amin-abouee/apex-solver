//! Reader and writer for Meta's VRS container (Project Aria recordings).
//!
//! VRS stores per-device streams of timestamped records. This module is a
//! from-scratch, safe-Rust implementation of the `VRS2` file format derived
//! from the published documentation and verified against real Aria files.

pub mod error;
pub mod format;

pub use error::{Result, VrsError};
pub use format::{
    Compression, FILE_HEADER_SIZE, FileHeader, IndexEntry, RECORD_HEADER_SIZE, RecordHeader,
    RecordType, StreamId,
};
