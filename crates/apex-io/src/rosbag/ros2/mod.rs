//! ROS2 bag format reader, writer, and message catalog.

pub mod cdr;
pub mod error;
pub mod messages;
pub mod metadata;
pub mod reader;
pub mod storage;
pub mod writer;

#[cfg(feature = "dds")]
pub mod dds;

pub use cdr::{CdrError, CdrErrorKind, CdrResult};
pub use error::{Ros2BagError, Ros2BagResult};
pub use metadata::{BagMetadata, TopicMetadata};
pub use reader::Reader;
pub use writer::Writer;
