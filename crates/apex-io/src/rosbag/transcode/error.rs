//! Errors of the ROS1 <-> ROS2 payload transcoder.
//!
//! Definition and planning errors carry the offending type or line. The
//! per-message conversion errors ([`TranscodeError::Truncated`] and the ones
//! after it) carry only `Copy` fields and are never logged here: a bag
//! converter decides per message whether to skip or abort.

/// Result of a transcoder operation.
pub type TranscodeResult<T> = Result<T, TranscodeError>;

/// A message definition that cannot be compiled, a topic that cannot be
/// planned, or a payload that does not match its definition.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TranscodeError {
    #[error("IDL message definitions are not supported")]
    IdlUnsupported,

    #[error("definition section without a `MSG:` name")]
    SectionWithoutName,

    #[error("message type {name} is not defined")]
    UndefinedType { name: String },

    #[error("malformed definition line {line:?}")]
    MalformedLine { line: String },

    #[error("malformed array type {ty:?}")]
    MalformedArrayType { ty: String },

    #[error("malformed array size in {ty:?}")]
    MalformedArraySize { ty: String },

    #[error("wstring/wchar fields are not supported")]
    WideStringUnsupported,

    #[error("message {name} nests deeper than {max} levels")]
    TooDeep { name: String, max: usize },

    #[error("unsupported payload encoding {format:?}")]
    UnsupportedEncoding { format: String },

    #[error("no message definition for {message_type} (not in the bag, not a standard type)")]
    NoDefinition { message_type: String },

    #[error("{message_type} depends on undefined types: {}", missing.join(", "))]
    UndefinedDependencies {
        message_type: String,
        missing: Vec<String>,
    },

    #[error("message truncated: need {need} bytes at byte {position}, {available} remain")]
    Truncated {
        position: usize,
        need: usize,
        available: usize,
    },

    #[error("CDR string at byte {position} lacks its NUL terminator")]
    MissingNul { position: usize },

    #[error("big-endian CDR payloads are not supported")]
    BigEndian,

    #[error("CDR payload of {len} bytes is too short for its header")]
    PayloadTooShort { len: usize },

    #[error("array of {count} elements of {size} bytes overflows")]
    ArrayOverflow { count: usize, size: usize },

    #[error("{count} unread bytes after the message (definition mismatch?)")]
    TrailingBytes { count: usize },
}
