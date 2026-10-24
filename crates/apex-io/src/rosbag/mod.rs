//! ROS1 and ROS2 bag reading and writing.
//!
//! This module provides comprehensive functionality to read and write both
//! ROS1 (v2.0, single `.bag` file) and ROS2 bag files (SQLite3 and MCAP storage
//! formats). The two formats are exposed through parallel APIs rather than a
//! single unified one: use [`Reader`]/[`Writer`] for ROS2 and
//! [`ros1::Ros1Reader`]/[`ros1::Ros1Writer`] for ROS1. The design mirrors the
//! upstream Python [`rosbags`](https://gitlab.com/ternaris/rosbags) library.
//!
//! ## Bare MCAP files (ROS1-profile bags)
//!
//! [`Reader`] accepts either a bag directory (with `metadata.yaml`) or a
//! bare `.mcap` file. Single-file bags are common for datasets (e.g. the
//! HILTI SLAM Challenge bags), and many are *ROS1-profile* MCAPs: the
//! container is MCAP, but payloads are ROS1-serialized (`profile: ros1`,
//! channel `message_encoding: ros1`, `ros1msg` schemas). The container layer
//! is shared; only payload decoding differs, so dispatch on the
//! connection's `serialization_format`:
//!
//! ```no_run
//! use apex_io::rosbag::{Reader, cdr::CdrDeserializer, messages::FromCdr,
//!                      ros1::deserialize::Ros1Deserializer, ros1::messages::Image as Ros1Image};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut reader = Reader::new("/data/recording.mcap")?;
//! reader.open()?;
//! for message in reader.raw_messages()? {
//!     let message = message?;
//!     match message.connection.serialization_format.as_str() {
//!         "ros1" => {
//!             let mut d = Ros1Deserializer::new(&message.raw_data);
//!             let image = Ros1Image::from_ros1(&mut d)?;
//!             println!("ros1 image {}x{}", image.width, image.height);
//!         }
//!         "cdr" => {
//!             let mut d = CdrDeserializer::new(&message.raw_data)?;
//!             let image = apex_io::rosbag::messages::Image::from_cdr(&mut d)?;
//!             println!("cdr image {}x{}", image.width, image.height);
//!         }
//!         other => panic!("unknown encoding {other}"),
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! A ros1-profile bag's connections report the schema name (e.g.
//! `sensor_msgs/Image`) as `message_type` and the channel encoding (`ros1`)
//! as `serialization_format` -- read from the MCAP summary, never a full
//! message scan.
//!
//!
//! ## Features
//!
//! - Read ROS2 bag files in SQLite3 and MCAP formats
//! - Write ROS1 bags, and ROS2 bags with SQLite3 or MCAP storage
//! - Parse `metadata.yaml` files with full validation
//! - Filter messages by topic and time range
//! - Compression support (zstd)
//! - Schema-driven ROS1 <-> ROS2 payload conversion ([`transcode`]) for every type with a definition
//! - One streaming reader/writer over all containers ([`bag::BagReader`], [`bag::BagWriter`])
//! - Cross-compatible with Python rosbags library
//!
//! ## Quick Start
//!
//! ### Reading a bag
//! ```no_run
//! use apex_io::rosbag::Reader;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut reader = Reader::new(Path::new("path/to/bag"))?;
//! reader.open()?;
//!
//! println!("Bag duration: {:.2}s", reader.duration() as f64 / 1e9);
//! println!("Topics: {}", reader.topics().len());
//!
//! for message_result in reader.messages()? {
//!     let message = message_result?;
//!     println!("Topic: {}, Time: {}", message.connection.topic, message.timestamp);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Writing a bag
//! ```no_run
//! use apex_io::rosbag::{Writer, StoragePlugin};
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut writer = Writer::new("output_bag", None, Some(StoragePlugin::Sqlite3))?;
//! writer.open()?;
//!
//! let connection = writer.add_connection(
//!     "/my_topic".to_string(),
//!     "std_msgs/msg/String".to_string(),
//!     None, None, None, None
//! )?;
//!
//! writer.write(&connection, 1_000_000_000u64, b"hello")?;
//! writer.close()?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Fast metadata reading
//! ```no_run
//! use apex_io::rosbag::read_bag_metadata_fast;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let metadata = read_bag_metadata_fast(Path::new("path/to/bag"))?;
//! println!("Duration: {:.2}s", metadata.duration() as f64 / 1e9);
//! println!("Message count: {}", metadata.message_count());
//! # Ok(())
//! # }
//! ```

/// Error types for bag I/O operations.
pub mod error;
pub mod point_cloud;

/// Format-agnostic reader/writer over every supported container.
pub mod bag;

/// Schema-driven ROS1 <-> ROS2 payload conversion.
pub mod transcode;

#[cfg(feature = "vrs")]
pub(crate) mod vrs_bag;

/// Core data types and structures.
pub mod types;

/// ROS1 bag (v2.0) reader and writer.
pub mod ros1;

/// ROS2 bag reader, writer, CDR deserialization, and message catalog.
pub mod ros2;

// Backward-compatible module re-exports so existing code using
// `apex_io::rosbag::cdr::*`, `::messages::*`, `::storage::*`, `::metadata::*` continues to compile.
pub use ros2::cdr;
pub use ros2::messages;
pub use ros2::metadata;
pub use ros2::storage;

// Re-export main types for convenience
pub use error::{BagError, ReaderError, Result, WriterResult};
pub use ros2::metadata::{BagMetadata, TopicMetadata};
pub use ros2::reader::Reader;
pub use ros2::writer::Writer;
pub use types::{
    CompressionFormat, CompressionMode, Connection, Message, StoragePlugin, TopicInfo,
};

pub use bag::{
    BagIterator, BagKind, BagMessage, BagReader, BagWriter, ConnectionHandle, ContainerParams,
    NewConnection, OutputCompression, Selection, SensorConfig, SensorParams, detect_kind,
};
pub use ros1::{Ros1Compression, Ros1Reader, Ros1Writer};

/// Read bag metadata from `metadata.yaml` without opening storage files.
///
/// This is ideal for quickly inspecting a bag's duration, message count,
/// and topic list without the overhead of opening storage.
///
/// # Example
/// ```no_run
/// use apex_io::rosbag::read_bag_metadata_fast;
/// use std::path::Path;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let metadata = read_bag_metadata_fast(Path::new("path/to/bag"))?;
///
/// println!("Duration: {:.2}s", metadata.duration() as f64 / 1e9);
/// println!("Message count: {}", metadata.message_count());
///
/// for topic in &metadata.info().topics_with_message_count {
///     println!("Topic: {} ({}), Count: {}",
///         topic.topic_metadata.name,
///         topic.topic_metadata.message_type,
///         topic.message_count
///     );
/// }
/// # Ok(())
/// # }
/// ```
pub fn read_bag_metadata_fast<P: AsRef<std::path::Path>>(bag_path: P) -> Result<BagMetadata> {
    let bag_path = bag_path.as_ref();
    let metadata_path = bag_path.join("metadata.yaml");
    BagMetadata::from_file(metadata_path)
}
