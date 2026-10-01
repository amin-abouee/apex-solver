//! Reader and writer for Meta's VRS container (Project Aria recordings).
//!
//! VRS stores per-device streams of timestamped records. This module is a
//! from-scratch, safe-Rust implementation of the `VRS2` file format derived
//! from the published documentation and verified against real Aria files.

pub mod aria;
pub mod codec;
pub mod copy;
pub mod datalayout;
pub mod description;
pub mod error;
pub mod format;
pub mod index;
pub mod readahead;
pub mod reader;
pub mod recordformat;
pub mod writer;

pub use copy::{CopyOptions, copy_filtered};
pub use datalayout::{DataLayout, LayoutView, LayoutWriter, Piece, PieceKind, Scalar};
pub use description::{Description, StreamTags, TagMap};
pub use error::{Result, VrsError};
pub use format::{
    Compression, FILE_HEADER_SIZE, FileHeader, IndexEntry, RECORD_HEADER_SIZE, RecordHeader,
    RecordType, StreamId,
};
pub use index::{IndexSource, RecordInfo};
pub use readahead::SequentialReader;
pub use reader::{Cursor, Record, RecordFilter, RecordMeta, StoredRecord, VrsReader};
pub use recordformat::{Block, BlockSpec, ImageFormat, ImageSpec, RecordFormat, RecordLayout};
pub use writer::{VrsWriteOptions, VrsWriter, WriteSummary};
