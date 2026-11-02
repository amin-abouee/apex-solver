//! Errors for VRS reading and writing.
//!
//! [`VrsError`] covers the container: the file and record headers, the index,
//! compression and the writer's preconditions. [`LayoutError`] covers the
//! typed views inside a record (`DataLayout` and `RecordFormat`); it is a
//! separate enum because its accessors are also used to probe a record, so
//! they never log. Container errors are logged once where they are built;
//! layout errors are logged by the caller that gives up on the record.

use super::format::StreamId;
use std::path::PathBuf;

/// Result alias for VRS operations.
pub type Result<T> = std::result::Result<T, VrsError>;

/// Result alias for VRS operations, named like the other module aliases.
pub type VrsResult<T> = Result<T>;

/// Failure while reading or writing a VRS file.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VrsError {
    /// Underlying I/O failure on an already open file.
    #[error("VRS file i/o failed")]
    Io(#[from] std::io::Error),

    /// The file could not be opened or created.
    #[error("cannot open VRS file {}", path.display())]
    Open {
        /// Offending path.
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The file does not start with the VRS magic values.
    #[error("{} is not a VRS file", path.display())]
    NotVrs {
        /// Offending path.
        path: PathBuf,
    },

    /// A structure on disk is shorter than its fixed size.
    #[error("truncated {what}: {len} bytes, need {need}")]
    Truncated {
        /// The structure, e.g. `record header`.
        what: &'static str,
        /// Bytes available.
        len: usize,
        /// Bytes required.
        need: usize,
    },

    /// A record header declares a size smaller than the header itself.
    #[error("record size {size} is smaller than its {header}-byte header")]
    RecordTooSmall {
        /// Declared whole-record size.
        size: u32,
        /// Header size.
        header: usize,
    },

    /// A record or offset lies outside the file.
    #[error(
        "{what} at offset {offset} with {len} bytes runs past the end of the {file_len}-byte file"
    )]
    OutOfFile {
        /// The structure, e.g. `index record`.
        what: &'static str,
        /// Its start offset.
        offset: u64,
        /// Its length.
        len: u64,
        /// File size.
        file_len: u64,
    },

    /// A header offset is negative.
    #[error("negative {what} offset {offset}")]
    NegativeOffset {
        /// Which offset.
        what: &'static str,
        /// Its value.
        offset: i64,
    },

    /// A record read through the index is not the record the index names.
    #[error("record {index} at offset {offset} does not match the index")]
    IndexMismatch {
        /// Record index.
        index: usize,
        /// Its offset.
        offset: u64,
    },

    /// The index record cannot be decoded.
    #[error("index record unusable: {reason}")]
    BadIndex {
        /// What is wrong with it.
        reason: &'static str,
    },

    /// A payload decompressed to a different size than its header says.
    #[error("payload decompressed to {actual} bytes, header says {expected}")]
    SizeMismatch {
        /// Size announced by the record header.
        expected: usize,
        /// Size produced.
        actual: usize,
    },

    /// The record header names an unknown compression type.
    #[error("unsupported VRS compression type {value}")]
    UnknownCompression {
        /// The on-disk byte.
        value: u8,
    },

    /// The file is written in an unsupported VRS format version.
    #[error("unsupported VRS format version {version:?}")]
    FormatVersion {
        /// The four version bytes, as text where printable.
        version: String,
    },

    /// A header size other than the one this implementation reads.
    #[error("unsupported VRS {what} {value}")]
    HeaderSize {
        /// Which size, e.g. `file header size`.
        what: &'static str,
        /// Its value.
        value: u32,
    },

    /// A content block without a size that is not the last block.
    #[error("content block {block} has no size and is not the last block")]
    UnsizedBlock {
        /// Block index in the record format.
        block: usize,
    },

    /// A record payload failed to decompress.
    #[error("{codec} decompression failed")]
    Decompress {
        /// `zstd` or `lz4`.
        codec: &'static str,
        #[source]
        source: std::io::Error,
    },

    /// A record payload failed to compress.
    #[error("{codec} compression failed")]
    Compress {
        /// `zstd` or `lz4`.
        codec: &'static str,
        #[source]
        source: std::io::Error,
    },

    /// The lz4 frame could not be finished.
    #[error("lz4 frame could not be finished")]
    Lz4Frame(#[source] lz4_flex::frame::Error),

    /// A JSON description failed to parse.
    #[error("invalid JSON description")]
    Json(#[from] serde_json::Error),

    /// A `DataLayout`/`RecordFormat` description or access was invalid.
    #[error(transparent)]
    Layout(#[from] LayoutError),

    /// A record index past the end of the file's records.
    #[error("record index {index} out of range, the file has {count} records")]
    RecordOutOfRange {
        /// Requested index.
        index: usize,
        /// Records in the file.
        count: usize,
    },

    /// A stream has no configuration record.
    #[error("{sensor} has no configuration record")]
    NoConfiguration {
        /// The sensor's label.
        sensor: &'static str,
    },

    /// A sensor stream decoded to a different kind of sample than asked for.
    #[error("{sensor} is not a {expected} stream")]
    WrongSensor {
        /// The sensor's label.
        sensor: &'static str,
        /// What the caller asked for.
        expected: &'static str,
    },

    /// A writer call made after the first record was written.
    #[error("cannot {what} after the first record was written")]
    AlreadyStarted {
        /// The refused call.
        what: &'static str,
    },

    /// A stream id in the reserved range.
    #[error("stream type ids 1 and 2 are reserved, got {stream}")]
    ReservedStream {
        /// The stream.
        stream: StreamId,
    },

    /// A stream declared twice.
    #[error("stream {stream} declared twice")]
    DuplicateStream {
        /// The stream.
        stream: StreamId,
    },

    /// A record for a stream that was never declared.
    #[error("stream {stream} was not declared")]
    UndeclaredStream {
        /// The stream.
        stream: StreamId,
    },

    /// A call on a writer that was already finished.
    #[error("the VRS writer is already finished")]
    WriterFinished,

    /// A record or index too large for the format's 32-bit sizes.
    #[error("{what} of {len} bytes does not fit the 4 GiB limit")]
    TooLarge {
        /// `record` or `index`.
        what: &'static str,
        /// Its size.
        len: usize,
    },

    /// The read-ahead worker thread is gone.
    #[error("the VRS read-ahead worker has stopped")]
    ReadAheadStopped,

    /// The read-ahead worker thread could not be started.
    #[error("cannot spawn the VRS read-ahead thread")]
    ReadAheadSpawn(#[source] std::io::Error),
}

/// A `DataLayout` or `RecordFormat` that does not fit a record, or a field
/// access that does not fit the layout.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum LayoutError {
    #[error("duplicate DataLayout piece {name}")]
    DuplicatePiece { name: String },

    #[error("DataLayout piece {name} has neither a valid offset nor an index")]
    PiecePlacement { name: String },

    #[error("record of {len} bytes is shorter than its {need}-byte DataLayout")]
    RecordTooShort { len: usize, need: usize },

    #[error("variable-size piece {index} ends past the {len}-byte record")]
    VarOutOfBounds { index: usize, len: usize },

    #[error("no field {name} in this DataLayout")]
    NoField { name: String },

    #[error("field {name} is not a {expected}")]
    WrongKind {
        name: String,
        expected: &'static str,
    },

    #[error("field {name} is not a {scalar}[{len}] array")]
    WrongArray {
        name: String,
        scalar: &'static str,
        len: usize,
    },

    #[error("string vector field {name} is truncated")]
    StringVectorTruncated { name: String },

    #[error("unknown content block {block:?}")]
    UnknownBlock { block: String },

    #[error("missing tag DL:{record_type}:{version}:{index}")]
    MissingLayoutTag {
        record_type: &'static str,
        version: u32,
        index: usize,
    },

    #[error("record of {len} bytes ends before content block {block}")]
    RecordShorterThanFormat { len: usize, block: usize },

    #[error(
        "content block {block} of {size} bytes exceeds the {available} bytes left in the record"
    )]
    BlockExceedsRecord {
        block: usize,
        size: usize,
        available: usize,
    },

    #[error("record format says data_layout/size={declared} but the layout is {actual} bytes")]
    LayoutSize { declared: usize, actual: usize },

    #[error("stream {stream} has no layout for {record_type} records of format {version}")]
    NoRecordLayout {
        stream: StreamId,
        record_type: &'static str,
        version: u32,
    },

    #[error("{record_type} record has no DataLayout block")]
    NoLayoutBlock { record_type: &'static str },

    #[error("camera record has no image block")]
    NoImageBlock,
}

/// Result alias for `DataLayout` and `RecordFormat` operations.
pub type LayoutResult<T> = std::result::Result<T, LayoutError>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorChainExt;

    #[test]
    fn a_layout_error_is_transparent_in_the_container_error() {
        let error = VrsError::from(LayoutError::NoField {
            name: "gain".to_string(),
        });
        assert_eq!(error.chain_compact(), "no field gain in this DataLayout");
    }

    #[test]
    fn a_codec_error_keeps_its_source() {
        let error = VrsError::Decompress {
            codec: "zstd",
            source: std::io::Error::other("unknown frame descriptor"),
        };
        assert_eq!(
            error.chain_compact(),
            "zstd decompression failed → unknown frame descriptor"
        );
    }
}
