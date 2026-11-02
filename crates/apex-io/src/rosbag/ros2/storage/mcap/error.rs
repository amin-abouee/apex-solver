//! Errors of the MCAP container reader and writer.
//!
//! The container parsers build their errors without logging; the storage
//! backend logs one where it aborts a read or a write.

use std::path::PathBuf;

/// Result of an MCAP container operation.
pub type McapResult<T> = Result<T, McapError>;

/// An MCAP container that cannot be read or written.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum McapError {
    #[error("cannot open MCAP file {}", path.display())]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("MCAP file i/o failed while {what}")]
    Io {
        what: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error("MCAP file of {len} bytes is too small for the magic")]
    TooSmall { len: u64 },

    #[error("the {position} MCAP magic is missing")]
    BadMagic { position: &'static str },

    #[error("expected an MCAP {expected} record at byte {offset}, found opcode 0x{found:02x}")]
    UnexpectedOpcode {
        expected: &'static str,
        found: u8,
        offset: u64,
    },

    #[error("MCAP footer content length is {len}, expected 20")]
    FooterLength { len: u64 },

    #[error(
        "MCAP footer summary offsets are inconsistent: start {start}, end {end}, file {file_len}"
    )]
    SummaryOffsets { start: u64, end: u64, file_len: u64 },

    #[error("MCAP {what} truncated: need {need} bytes at offset {offset}, have {available}")]
    Truncated {
        what: &'static str,
        offset: u64,
        need: u64,
        available: u64,
    },

    #[error("MCAP record at byte {offset} declares a length of {len} bytes that overflows")]
    LengthOverflow { offset: u64, len: u64 },

    #[error("MCAP string is not UTF-8")]
    NotUtf8(#[source] std::string::FromUtf8Error),

    #[error("unsupported MCAP chunk compression {compression:?}")]
    UnsupportedCompression { compression: String },

    #[error("{codec} MCAP chunk decompression failed")]
    Decompress {
        codec: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error("MCAP chunk decompressed to {actual} bytes, its header declared {expected}")]
    ChunkSize { expected: u64, actual: usize },

    #[error("MCAP chunk CRC mismatch: header {expected:#010x}, payload {actual:#010x}")]
    ChunkCrc { expected: u32, actual: u32 },

    #[error(
        "MCAP chunk at byte {offset} declares compression {found:?}, its index said {indexed:?}"
    )]
    ChunkCompressionMismatch {
        offset: u64,
        found: String,
        indexed: String,
    },

    #[error(
        "MCAP chunk at byte {offset} declares {found} uncompressed bytes, its index said {indexed}"
    )]
    ChunkSizeMismatch {
        offset: u64,
        found: u64,
        indexed: u64,
    },

    #[error("MCAP message record of {len} bytes is shorter than its 22-byte header")]
    ShortMessage { len: u64 },

    #[error("MCAP message references unknown channel {id}")]
    UnknownChannel { id: u16 },

    #[error("an MCAP chunk worker stopped unexpectedly")]
    WorkerStopped,

    #[error("the zstd context could not be created")]
    ZstdInit(#[source] std::io::Error),

    #[error("zstd MCAP chunk compression failed")]
    Compress(#[source] std::io::Error),

    #[error("too many MCAP {what}, the format allows 65535")]
    TooMany { what: &'static str },

    #[error("the MCAP writer is already finished")]
    WriterFinished,

    #[error("an MCAP chunk compression worker panicked")]
    WorkerPanicked,

    #[error("the MCAP chunk compression workers disconnected")]
    WorkersDisconnected,

    #[error("{missing} MCAP chunks were never written")]
    ChunksMissing { missing: u64 },
}

impl McapError {
    pub(crate) fn io(what: &'static str) -> impl FnOnce(std::io::Error) -> Self {
        move |source| Self::Io { what, source }
    }
}
