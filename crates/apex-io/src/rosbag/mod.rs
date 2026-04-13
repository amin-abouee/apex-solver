//! ROS2 bag reading and writing (sqlite3 and MCAP formats).
//!
//! This module provides comprehensive functionality to read and write ROS2 bag files,
//! supporting both SQLite3 and MCAP storage formats with guaranteed compatibility
//! with the Python rosbags library.
//!
//! ## Features
//!
//! - Read ROS2 bag files in SQLite3 and MCAP formats
//! - Write ROS2 bag files with SQLite3 storage
//! - Parse `metadata.yaml` files with full validation
//! - Filter messages by topic and time range
//! - Compression support (zstd)
//! - 94+ ROS2 message types with CDR deserialization
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

/// CDR (Common Data Representation) deserialization.
pub mod cdr;

/// Error types for bag I/O operations.
pub mod error;

/// ROS2 message type definitions with CDR deserialization support.
pub mod messages;

/// Metadata parsing and validation (`metadata.yaml`).
pub mod metadata;

/// Main reader interface.
pub mod reader;

/// Main writer interface.
pub mod writer;

/// Storage backend implementations (SQLite3 and MCAP).
pub mod storage;

/// Core data types and structures.
pub mod types;

// Re-export main types for convenience
pub use error::{BagError, ReaderError, Result, WriterResult};
pub use metadata::{BagMetadata, TopicMetadata};
pub use reader::Reader;
pub use types::{
    CompressionFormat, CompressionMode, Connection, Message, StoragePlugin, TopicInfo,
};
pub use writer::Writer;

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
