//! Errors of the rosbag module.
//!
//! [`BagError`] is the root of the rosbag errors and the error of the
//! container-agnostic facade ([`BagReader`](super::BagReader),
//! [`BagWriter`](super::BagWriter), [`detect_kind`](super::detect_kind)).
//! Each submodule returns its own error, wrapped transparently here:
//!
//! - [`Ros1Error`]: ROS1 bag reader, writer and message codec.
//! - [`Ros2BagError`]: ROS2 bag reader, writer, metadata and storage backends;
//!   it wraps [`McapError`] (the MCAP container) and [`CdrError`].
//! - [`CdrError`]: a CDR payload that does not decode.
//! - [`PointCloudError`]: a point-cloud field that cannot be read.
//! - [`TranscodeError`]: ROS1 <-> ROS2 payload conversion.
//! - `DdsError` (feature `dds`): live DDS subscriptions.
//! - `VrsError` (feature `vrs`): VRS containers read or written as bags.

use std::path::PathBuf;

pub use super::point_cloud::{PointCloudError, PointCloudResult};
pub use super::ros1::error::{Ros1Error, Ros1Result};
pub use super::ros2::cdr::{CdrError, CdrErrorKind, CdrResult};
#[cfg(feature = "dds")]
pub use super::ros2::dds::DdsError;
pub use super::ros2::error::{Ros2BagError, Ros2BagResult};
pub use super::ros2::storage::mcap::{McapError, McapResult};
pub use super::transcode::{TranscodeError, TranscodeResult};

/// Result type alias for rosbag operations
pub type Result<T> = std::result::Result<T, BagError>;

/// Result type alias for rosbag operations, named like the other module aliases.
pub type BagResult<T> = Result<T>;

/// Result type alias for reader operations (backwards compatibility)
pub type ReaderResult<T> = Result<T>;

/// Result type alias for writer operations
pub type WriterResult<T> = Result<T>;

/// Type alias for backwards compatibility
pub type ReaderError = BagError;

/// Root error of the rosbag module.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BagError {
    #[error(transparent)]
    Ros1(#[from] Ros1Error),

    #[error(transparent)]
    Ros2(#[from] Ros2BagError),

    #[error(transparent)]
    Mcap(#[from] McapError),

    #[error(transparent)]
    Cdr(#[from] CdrError),

    #[error(transparent)]
    PointCloud(#[from] PointCloudError),

    #[error(transparent)]
    Transcode(#[from] TranscodeError),

    #[cfg(feature = "dds")]
    #[error(transparent)]
    Dds(#[from] DdsError),

    #[cfg(feature = "vrs")]
    #[error(transparent)]
    Vrs(#[from] crate::vrs::VrsError),

    #[error("bag not found: {}", path.display())]
    BagNotFound { path: PathBuf },

    #[error("bag already exists: {}", path.display())]
    BagAlreadyExists { path: PathBuf },

    #[error("cannot read {}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{} is not a ROS1 bag, MCAP or VRS file", path.display())]
    NotABag { path: PathBuf },

    #[error("directory {} contains no bag", path.display())]
    NoBagInDirectory { path: PathBuf },

    #[error("unsupported storage format {format:?}")]
    UnsupportedStorageFormat { format: String },

    #[error("VRS containers need apex-io built with the `vrs` feature")]
    VrsDisabled,

    #[error("{container} does not support {what}")]
    Unsupported {
        container: &'static str,
        what: &'static str,
    },

    #[error("too many connections: {count}")]
    TooManyConnections { count: usize },

    #[error("no connection for handle {handle}")]
    UnknownHandle { handle: usize },

    #[error("no output connection for topic {topic}")]
    ConnectionNotFound { topic: String },

    #[error("VRS stream {type_id}-{instance} lacks its {tag} tag")]
    VrsStreamTag {
        type_id: i32,
        instance: u16,
        tag: &'static str,
    },

    #[error("VRS stream {type_id}-{instance} carries ROS messages but no data format")]
    VrsNoDataFormat { type_id: i32, instance: u16 },

    #[error("VRS record {index} is not a well-formed ROS message record")]
    VrsMalformedRecord { index: usize },

    #[error("VRS record {index} is not a data record")]
    VrsRecordNotData { index: usize },

    #[error("VRS output needs every topic declared before the first message")]
    VrsTopicsAfterStart,
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_error::ErrorChainExt;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    fn decode() -> CdrResult<()> {
        Err(CdrError {
            kind: CdrErrorKind::Truncated { need: 8 },
            position: 12,
            len: 16,
        })
    }

    fn storage() -> Ros2BagResult<()> {
        decode()?;
        Ok(())
    }

    fn facade() -> Result<()> {
        storage()?;
        Ok(())
    }

    #[test]
    fn a_leaf_error_reaches_the_root_with_bare_question_marks() -> TestResult {
        let error = facade().err().ok_or("expected an error")?;
        assert!(matches!(error, BagError::Ros2(Ros2BagError::Cdr(_))));
        assert_eq!(
            error.chain_compact(),
            "CDR field needs 8 bytes at byte 12 of 16"
        );
        Ok(())
    }

    #[test]
    fn the_crate_root_wraps_the_rosbag_root() -> TestResult {
        let error = crate::IoError::from(BagError::UnknownHandle { handle: 3 });
        assert_eq!(error.chain(), "no connection for handle 3");
        Ok(())
    }

    #[test]
    fn facade_errors_name_their_values() {
        let error = BagError::Unsupported {
            container: "vrs",
            what: "bz2 compression",
        };
        assert_eq!(error.to_string(), "vrs does not support bz2 compression");
    }
}
