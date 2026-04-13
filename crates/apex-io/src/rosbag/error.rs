//! Error types for rosbag I/O operations

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for rosbag operations
pub type Result<T> = std::result::Result<T, BagError>;

/// Result type alias for reader operations (backwards compatibility)
pub type ReaderResult<T> = std::result::Result<T, BagError>;

/// Result type alias for writer operations
pub type WriterResult<T> = std::result::Result<T, BagError>;

/// Errors that can occur when working with ROS2 bag files
#[derive(Error, Debug)]
pub enum BagError {
    /// IO error when accessing files
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Error parsing YAML metadata
    #[error("Failed to parse metadata YAML: {0}")]
    YamlParse(#[from] serde_yaml::Error),

    /// Database error when reading SQLite files
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Compression/decompression error
    #[error("Compression error: {0}")]
    Compression(String),

    /// Bag file not found
    #[error("Bag file not found: {path}")]
    BagNotFound { path: PathBuf },

    /// Bag already exists
    #[error("Bag already exists: {path}")]
    BagAlreadyExists { path: PathBuf },

    /// Metadata file not found
    #[error("Metadata file not found: {path}")]
    MetadataNotFound { path: PathBuf },

    /// Storage file not found
    #[error("Storage file not found: {path}")]
    StorageFileNotFound { path: PathBuf },

    /// Unsupported bag version
    #[error("Unsupported bag version: {version}")]
    UnsupportedVersion { version: u32 },

    /// Unsupported storage format
    #[error("Unsupported storage format: {format}")]
    UnsupportedStorageFormat { format: String },

    /// Unsupported compression format
    #[error("Unsupported compression format: {format}")]
    UnsupportedCompressionFormat { format: String },

    /// Unsupported serialization format
    #[error("Unsupported serialization format: {format}")]
    UnsupportedSerializationFormat { format: String },

    /// Bag is not open
    #[error("Bag is not open - call open() first")]
    BagNotOpen,

    /// Bag is already open
    #[error("Bag is already open")]
    BagAlreadyOpen,

    /// Invalid message data
    #[error("Invalid message data: {reason}")]
    InvalidMessageData { reason: String },

    /// CDR deserialization error
    #[error("CDR deserialization error at position {position}/{data_length}: {message}")]
    CdrDeserialization {
        message: String,
        position: usize,
        data_length: usize,
    },

    /// Message type not found in type registry
    #[error("Message type not found: {message_type}")]
    MessageTypeNotFound { message_type: String },

    /// Schema validation error
    #[error("Schema validation error: {reason}")]
    SchemaValidation { reason: String },

    /// Connection not found
    #[error("Connection not found for topic: {topic}")]
    ConnectionNotFound { topic: String },

    /// Connection already exists
    #[error("Connection already exists for topic: {topic}")]
    ConnectionAlreadyExists { topic: String },

    /// Invalid QoS profile
    #[error("Invalid QoS profile: {reason}")]
    InvalidQosProfile { reason: String },

    /// Writer error with custom message
    #[error("Writer error: {message}")]
    Writer { message: String },

    /// Generic error with custom message
    #[error("Bag error: {message}")]
    Generic { message: String },
}

/// Type alias for backwards compatibility
pub type ReaderError = BagError;

impl BagError {
    /// Create a new generic error with a custom message
    pub fn generic(message: impl Into<String>) -> Self {
        Self::Generic {
            message: message.into(),
        }
    }

    /// Create a writer error
    pub fn writer(message: impl Into<String>) -> Self {
        Self::Writer {
            message: message.into(),
        }
    }

    /// Create a compression error
    pub fn compression(message: impl Into<String>) -> Self {
        Self::Compression(message.into())
    }

    /// Create an invalid message data error
    pub fn invalid_message_data(reason: impl Into<String>) -> Self {
        Self::InvalidMessageData {
            reason: reason.into(),
        }
    }

    /// Create a CDR deserialization error
    pub fn cdr_deserialization(
        message: impl Into<String>,
        position: usize,
        data_length: usize,
    ) -> Self {
        Self::CdrDeserialization {
            message: message.into(),
            position,
            data_length,
        }
    }

    /// Create a message type not found error
    pub fn message_type_not_found(message_type: impl Into<String>) -> Self {
        Self::MessageTypeNotFound {
            message_type: message_type.into(),
        }
    }

    /// Create a schema validation error
    pub fn schema_validation(reason: impl Into<String>) -> Self {
        Self::SchemaValidation {
            reason: reason.into(),
        }
    }

    /// Create a connection not found error
    pub fn connection_not_found(topic: impl Into<String>) -> Self {
        Self::ConnectionNotFound {
            topic: topic.into(),
        }
    }

    /// Create a connection already exists error
    pub fn connection_already_exists(topic: impl Into<String>) -> Self {
        Self::ConnectionAlreadyExists {
            topic: topic.into(),
        }
    }

    /// Create an invalid QoS profile error
    pub fn invalid_qos_profile(reason: impl Into<String>) -> Self {
        Self::InvalidQosProfile {
            reason: reason.into(),
        }
    }
}
