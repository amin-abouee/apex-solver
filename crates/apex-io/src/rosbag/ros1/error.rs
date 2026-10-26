//! Errors of the ROS1 bag (v2.0) reader, writer and message codec.
//!
//! The record parsers and the message deserializer build their errors without
//! logging: a failed message decode is an expected outcome for a caller that
//! probes a payload, and the record parsers run per record. The public
//! [`Ros1Reader`](super::Ros1Reader) and [`Ros1Writer`](super::Ros1Writer)
//! methods log an error once, where they give up.

use super::format::OpCode;
use std::path::PathBuf;

/// Result of a ROS1 bag operation.
pub type Ros1Result<T> = Result<T, Ros1Error>;

/// A ROS1 bag that cannot be read or written, or a payload that cannot be decoded.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Ros1Error {
    #[error("ROS1 bag i/o failed")]
    Io(#[from] std::io::Error),

    #[error("ROS1 bag not found: {}", path.display())]
    BagNotFound { path: PathBuf },

    #[error("ROS1 bag already exists: {}", path.display())]
    BagAlreadyExists { path: PathBuf },

    #[error("cannot open ROS1 bag {}", path.display())]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("the ROS1 bag is not open")]
    NotOpen,

    #[error("the ROS1 bag is already open")]
    AlreadyOpen,

    #[error("ROS1 bag magic invalid: expected `#ROSBAG V2.0`, got {found:?}")]
    InvalidMagic { found: String },

    #[error("ROS1 bag version {found:?} is not supported")]
    UnsupportedVersion { found: String },

    #[error("unknown ROS1 record op-code 0x{op:02x}")]
    UnknownOpCode { op: u8 },

    #[error("ROS1 record header has no field {field}")]
    MissingField { field: String },

    #[error("ROS1 record field {field} has {actual} bytes, expected {expected}")]
    FieldSize {
        field: String,
        expected: usize,
        actual: usize,
    },

    #[error("ROS1 record field {field} is not UTF-8")]
    FieldNotUtf8 { field: String },

    #[error("ROS1 record header is truncated: a field needs {need} bytes, {available} remain")]
    HeaderTruncated { need: usize, available: usize },

    #[error("ROS1 record header field has no `=`")]
    FieldWithoutEquals,

    #[error("ROS1 record header field name is not UTF-8")]
    FieldNameNotUtf8,

    #[error("ROS1 {what} is truncated: need {need} bytes, {available} available")]
    Truncated {
        what: &'static str,
        need: u64,
        available: u64,
    },

    #[error("ROS1 {what} at byte {position} is missing")]
    MissingRecord { what: &'static str, position: u64 },

    #[error("expected a ROS1 {expected:?} record at byte {position}, found {found:?}")]
    UnexpectedRecord {
        expected: OpCode,
        found: OpCode,
        position: u64,
    },

    #[error("ROS1 bag header has index_pos 0, the writer never called close()")]
    Unfinished,

    #[error("ROS1 bag header claims {claimed} {what} but the tail holds {found}")]
    CountMismatch {
        what: &'static str,
        claimed: u32,
        found: usize,
    },

    #[error("ROS1 IndexData record after the chunk at {chunk} lacks conn, ver or count")]
    IndexDataFields { chunk: u64 },

    #[error("ROS1 IndexData version {version} is not supported (want 1)")]
    IndexDataVersion { version: u32 },

    #[error("ROS1 IndexData for connection {conn} holds {count} entries in {len} bytes")]
    IndexDataTruncated { conn: u32, count: usize, len: usize },

    #[error("ROS1 chunk at {chunk} lists a different message count for connection {conn}")]
    ChunkCountMismatch { chunk: u64, conn: u32 },

    #[error("ROS1 ChunkInfo holds {count} entries in {len} bytes")]
    ChunkInfoTruncated { count: usize, len: usize },

    #[error("ROS1 index references unknown connection {conn}")]
    UnknownConnection { conn: u32 },

    #[error("message offset {offset} is past the end of the ROS1 chunk at {chunk}")]
    OffsetOutOfChunk { offset: u32, chunk: u64 },

    #[error("unsupported ROS1 chunk compression {format:?}")]
    UnsupportedCompression { format: String },

    #[error("{codec} chunk decompression failed")]
    Decompress {
        codec: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error("lz4 block decompression failed")]
    Lz4Block(#[source] lz4_flex::block::DecompressError),

    #[error("{codec} chunk compression failed")]
    Compress {
        codec: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error("lz4 frame could not be finished")]
    Lz4Frame(#[source] lz4_flex::frame::Error),

    #[error("ROS1 chunk inflated to {actual} bytes, its header says {expected}")]
    ChunkSize { expected: usize, actual: usize },

    #[error("cannot {what} after the ROS1 bag was opened")]
    AlreadyStarted { what: &'static str },

    #[error("a ROS1 connection for topic {topic} already exists")]
    ConnectionAlreadyExists { topic: String },

    #[error("no ROS1 connection for topic {topic}")]
    ConnectionNotFound { topic: String },

    #[error("{what} of {len} bytes exceeds the ROS1 4 GiB limit")]
    TooLarge { what: &'static str, len: usize },

    #[error("the ROS1 bag header needs {len} bytes, more than the {reserved} reserved")]
    HeaderTooLarge { len: usize, reserved: usize },

    #[error("a ROS1 chunk compression worker panicked")]
    WorkerPanicked,

    #[error("the ROS1 chunk compression workers disconnected")]
    WorkersDisconnected,

    #[error("ROS1 message needs {need} bytes at byte {position}, {available} remain")]
    MessageTruncated {
        position: usize,
        need: usize,
        available: usize,
    },
}

impl Ros1Error {
    pub(crate) fn missing_field(field: &str) -> Self {
        Self::MissingField {
            field: field.to_string(),
        }
    }

    pub(crate) fn field_size(field: &str, expected: usize, actual: usize) -> Self {
        Self::FieldSize {
            field: field.to_string(),
            expected,
            actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_error::ErrorChainExt;

    #[test]
    fn display_strings_name_their_values() {
        let error = Ros1Error::field_size("conn", 4, 2);
        assert_eq!(
            error.to_string(),
            "ROS1 record field conn has 2 bytes, expected 4"
        );
        let open = Ros1Error::Open {
            path: PathBuf::from("a.bag"),
            source: std::io::Error::other("denied"),
        };
        assert_eq!(open.chain_compact(), "cannot open ROS1 bag a.bag → denied");
    }
}
