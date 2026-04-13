//! Core data types for ROS2 bag files

use serde::{Deserialize, Serialize};

/// Represents a connection to a topic in the bag file
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Connection {
    /// Unique connection ID
    pub id: u32,
    /// Topic name (e.g., "/camera/image_raw")
    pub topic: String,
    /// Message type (e.g., "sensor_msgs/msg/Image")
    pub message_type: String,
    /// Message definition (for type information)
    pub message_definition: MessageDefinition,
    /// Type description hash
    pub type_description_hash: String,
    /// Number of messages on this connection
    pub message_count: u64,
    /// Serialization format (typically "cdr")
    pub serialization_format: String,
    /// QoS profiles offered for this topic
    pub offered_qos_profiles: Vec<QosProfile>,
}

/// Message definition format and content
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageDefinition {
    /// Format of the definition (MSG, IDL, or None)
    pub format: MessageDefinitionFormat,
    /// The actual definition content
    pub data: String,
}

/// Format of message definitions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageDefinitionFormat {
    /// No definition available
    None,
    /// ROS message format (.msg files)
    Msg,
    /// Interface Definition Language format
    Idl,
}

/// QoS (Quality of Service) profile for a topic
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QosProfile {
    /// History policy
    pub history: QosHistory,
    /// Queue depth for KEEP_LAST history
    pub depth: u32,
    /// Reliability policy
    pub reliability: QosReliability,
    /// Durability policy
    pub durability: QosDurability,
    /// Deadline constraint
    pub deadline: QosTime,
    /// Lifespan constraint
    pub lifespan: QosTime,
    /// Liveliness policy
    pub liveliness: QosLiveliness,
    /// Liveliness lease duration
    pub liveliness_lease_duration: QosTime,
    /// Whether to avoid ROS namespace conventions
    pub avoid_ros_namespace_conventions: bool,
}

/// QoS History policy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QosHistory {
    SystemDefault,
    KeepLast,
    KeepAll,
    Unknown,
}

/// QoS Reliability policy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QosReliability {
    SystemDefault,
    Reliable,
    BestEffort,
    Unknown,
    BestAvailable,
}

/// QoS Durability policy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QosDurability {
    SystemDefault,
    TransientLocal,
    Volatile,
    Unknown,
    BestAvailable,
}

/// QoS Liveliness policy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QosLiveliness {
    SystemDefault,
    Automatic,
    ManualByNode,
    ManualByTopic,
    Unknown,
    BestAvailable,
}

/// Time specification for QoS constraints
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct QosTime {
    /// Seconds component
    pub sec: i32,
    /// Nanoseconds component
    pub nsec: u32,
}

/// Information about a topic in the bag
#[derive(Debug, Clone)]
pub struct TopicInfo {
    /// Topic name
    pub name: String,
    /// Message type
    pub message_type: String,
    /// Message definition
    pub message_definition: MessageDefinition,
    /// Number of messages
    pub message_count: u64,
    /// Connections for this topic
    pub connections: Vec<Connection>,
}

/// A message from the bag file
#[derive(Debug, Clone)]
pub struct Message {
    /// Connection this message belongs to
    pub connection: Connection,
    /// Topic name (convenience field)
    pub topic: String,
    /// Timestamp in nanoseconds since epoch
    pub timestamp: u64,
    /// Raw message data (serialized)
    pub data: Vec<u8>,
}

/// A raw message from the bag file with minimal processing overhead
/// This is equivalent to ROS2's SerializedBagMessage for high-performance operations
#[derive(Debug, Clone)]
pub struct RawMessage {
    /// Connection this message belongs to
    pub connection: Connection,
    /// Timestamp in nanoseconds since epoch
    pub timestamp: u64,
    /// Raw serialized message data (CDR format)
    pub raw_data: Vec<u8>,
}

/// Time duration in nanoseconds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Duration {
    pub nanoseconds: u64,
}

/// Starting time in nanoseconds since epoch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartingTime {
    pub nanoseconds_since_epoch: u64,
}

/// Compression mode for bag files
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMode {
    /// No compression
    None,
    /// Compress individual messages
    Message,
    /// Compress entire file
    File,
    /// Storage-specific compression
    Storage,
}

/// Compression format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionFormat {
    /// No compression format specified
    None,
    /// Zstandard compression
    Zstd,
}

/// Storage plugin type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoragePlugin {
    /// SQLite3 storage
    Sqlite3,
    /// MCAP storage
    Mcap,
}

impl Default for MessageDefinition {
    fn default() -> Self {
        Self {
            format: MessageDefinitionFormat::None,
            data: String::new(),
        }
    }
}

impl Default for QosProfile {
    fn default() -> Self {
        Self {
            history: QosHistory::SystemDefault,
            depth: 0,
            reliability: QosReliability::SystemDefault,
            durability: QosDurability::SystemDefault,
            deadline: QosTime::default(),
            lifespan: QosTime::default(),
            liveliness: QosLiveliness::SystemDefault,
            liveliness_lease_duration: QosTime::default(),
            avoid_ros_namespace_conventions: false,
        }
    }
}

impl Connection {
    /// Get the message type (compatibility alias for message_type)
    pub fn msgtype(&self) -> &str {
        &self.message_type
    }

    /// Get the message count (compatibility alias for message_count)
    pub fn msgcount(&self) -> u64 {
        self.message_count
    }
}

impl CompressionMode {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            CompressionMode::None => "",
            CompressionMode::Message => "message",
            CompressionMode::File => "file",
            CompressionMode::Storage => "storage",
        }
    }
}

impl CompressionFormat {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            CompressionFormat::None => "",
            CompressionFormat::Zstd => "zstd",
        }
    }
}

impl StoragePlugin {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            StoragePlugin::Sqlite3 => "sqlite3",
            StoragePlugin::Mcap => "mcap",
        }
    }
}
