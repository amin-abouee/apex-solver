//! ROS1 bag (v2.0) reader and writer.
//!
//! ROS1 bags are a single binary file with the layout:
//!
//! ```text
//! #ROSBAG V2.0\n
//! <bag header record>
//! <chunk record>*           // each holds connection + message records,
//!                           // optionally bz2/lz4 compressed
//! <index data record>*
//! <chunk info record>*
//! ```
//!
//! Unlike ROS2 bags, there is no `metadata.yaml` sidecar — all metadata is
//! embedded in the file itself.  This module exposes a parallel API to the
//! ROS2 reader/writer; types like [`Connection`](crate::rosbag::types::Connection)
//! and [`Message`](crate::rosbag::types::Message) are shared.

pub mod deserialize;
pub mod format;
pub mod messages;
pub mod reader;
pub mod record;
pub mod writer;

pub use deserialize::{Ros1Deserializer, Ros1Serializer};
pub use format::{OpCode, Ros1Compression};
pub use reader::Ros1Reader;
pub use writer::Ros1Writer;
