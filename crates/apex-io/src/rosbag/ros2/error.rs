//! Errors of the ROS2 bag reader, writer, metadata and storage backends.

use super::cdr::CdrError;
use super::storage::mcap::McapError;
use std::path::PathBuf;

/// Result of a ROS2 bag operation.
pub type Ros2BagResult<T> = Result<T, Ros2BagError>;

/// A ROS2 bag (directory with `metadata.yaml`, or a bare `.mcap` file) that
/// cannot be read or written.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Ros2BagError {
    #[error("ROS2 bag i/o failed")]
    Io(#[from] std::io::Error),

    #[error("cannot read ROS2 bag metadata {}", path.display())]
    MetadataNotFound {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("ROS2 bag metadata is not valid YAML")]
    Yaml(#[from] serde_norway::Error),

    #[error("ROS2 bag database query failed")]
    Sqlite(#[from] rusqlite::Error),

    #[error(transparent)]
    Mcap(#[from] McapError),

    #[error(transparent)]
    Cdr(#[from] CdrError),

    #[error("ROS2 bag not found: {}", path.display())]
    BagNotFound { path: PathBuf },

    #[error("ROS2 bag already exists: {}", path.display())]
    BagAlreadyExists { path: PathBuf },

    #[error("ROS2 bag storage file not found: {}", path.display())]
    StorageFileNotFound { path: PathBuf },

    #[error("bare bag file {} must be .mcap (a ROS2 bag directory needs metadata.yaml, a ROS1 .bag needs Ros1Reader)", path.display())]
    BareFileNotMcap { path: PathBuf },

    #[error("ROS2 bag metadata version {version} is not supported")]
    UnsupportedVersion { version: u32 },

    #[error("unsupported ROS2 storage format {format:?}")]
    UnsupportedStorageFormat { format: String },

    #[error("ROS2 bag has no .db3 or .mcap storage file")]
    NoStorageFiles,

    #[error("unsupported ROS2 bag compression format {format:?}")]
    UnsupportedCompressionFormat { format: String },

    #[error("unsupported ROS2 serialization format {format:?}")]
    UnsupportedSerializationFormat { format: String },

    #[error("the SQLite3 storage does not support storage-side compression")]
    SqliteStorageCompression,

    #[error("the ROS2 bag is not open")]
    NotOpen,

    #[error("the ROS2 bag is already open")]
    AlreadyOpen,

    #[error("no ROS2 connection for topic {topic}")]
    ConnectionNotFound { topic: String },

    #[error("a ROS2 connection for topic {topic} already exists")]
    ConnectionAlreadyExists { topic: String },

    #[error("bag database {} lacks the messages or topics table", path.display())]
    MissingTables { path: PathBuf },

    #[error("bag database topic id {id} does not fit a connection id")]
    TopicIdOutOfRange { id: i64 },

    #[error("bag database message {message} references unknown topic id {topic_id}")]
    UnknownTopicId { message: i64, topic_id: i64 },

    #[error("bag path {} has no file name", path.display())]
    NoFileName { path: PathBuf },

    #[error("storage file {} has no extension", path.display())]
    NoExtension { path: PathBuf },

    #[error("the zstd context could not be created")]
    ZstdInit(#[source] std::io::Error),

    #[error("zstd {what} failed")]
    Zstd {
        what: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error("MESSAGE-compressed payload does not start with a zstd frame")]
    NotZstdFrame,

    #[error("unsupported ROS2 message type {message_type}")]
    UnsupportedMessageType { message_type: String },
}

impl Ros2BagError {
    pub(crate) fn connection_not_found(topic: &str) -> Self {
        Self::ConnectionNotFound {
            topic: topic.to_string(),
        }
    }

    pub(crate) fn no_file_name(path: &std::path::Path) -> Self {
        Self::NoFileName {
            path: path.to_path_buf(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_error::ErrorChainExt;

    #[test]
    fn a_cdr_error_is_transparent() {
        let error = Ros2BagError::from(CdrError {
            kind: super::super::cdr::CdrErrorKind::Truncated { need: 8 },
            position: 20,
            len: 24,
        });
        assert_eq!(
            error.chain_compact(),
            "CDR field needs 8 bytes at byte 20 of 24"
        );
    }

    #[test]
    fn metadata_errors_keep_their_source() {
        let error = Ros2BagError::MetadataNotFound {
            path: PathBuf::from("bag/metadata.yaml"),
            source: std::io::Error::other("missing"),
        };
        assert_eq!(
            error.chain_compact(),
            "cannot read ROS2 bag metadata bag/metadata.yaml → missing"
        );
    }
}
