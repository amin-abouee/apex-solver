//! Metadata parsing for ROS2 bag files

use crate::rosbag::error::{ReaderError, Result};
use crate::rosbag::types::{Duration, QosProfile, StartingTime};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Complete bag metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BagMetadata {
    pub rosbag2_bagfile_information: BagFileInformation,
}

/// Main bag file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BagFileInformation {
    /// Bag format version
    pub version: u32,
    /// Storage plugin identifier (e.g., "sqlite3", "mcap")
    pub storage_identifier: String,
    /// Relative paths to storage files
    pub relative_file_paths: Vec<String>,
    /// Bag duration
    pub duration: Duration,
    /// Starting time
    pub starting_time: StartingTime,
    /// Total message count
    pub message_count: u64,
    /// Compression format (e.g., "zstd", empty string for none)
    #[serde(default)]
    pub compression_format: String,
    /// Compression mode (e.g., "FILE", "MESSAGE", empty string for none)
    #[serde(default)]
    pub compression_mode: String,
    /// Topics with message counts
    pub topics_with_message_count: Vec<TopicWithMessageCount>,
    /// Per-file information (version 5+)
    #[serde(default)]
    pub files: Vec<FileInformation>,
    /// Custom metadata (version 6+)
    #[serde(default)]
    pub custom_data: Option<HashMap<String, String>>,
    /// ROS distribution (version 8+)
    #[serde(default)]
    pub ros_distro: Option<String>,
}

/// Topic information with message count
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicWithMessageCount {
    /// Number of messages for this topic
    pub message_count: u64,
    /// Topic metadata
    pub topic_metadata: TopicMetadata,
}

/// Metadata for a single topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicMetadata {
    /// Topic name (e.g., "/camera/image_raw")
    pub name: String,
    /// Message type (e.g., "sensor_msgs/msg/Image")
    #[serde(rename = "type")]
    pub message_type: String,
    /// Serialization format (typically "cdr")
    pub serialization_format: String,
    /// QoS profiles (can be string or list depending on version)
    #[serde(default)]
    pub offered_qos_profiles: QosProfilesField,
    /// Type description hash (version 7+)
    #[serde(default)]
    pub type_description_hash: String,
}

/// QoS profiles field that can be either a string or a list
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum QosProfilesField {
    /// String representation (older versions)
    String(String),
    /// List of QoS profiles (newer versions)
    List(Vec<QosProfile>),
}

impl Default for QosProfilesField {
    fn default() -> Self {
        Self::String(String::new())
    }
}

/// Per-file information (version 5+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInformation {
    /// File path
    pub path: String,
    /// Starting time for this file
    pub starting_time: StartingTime,
    /// Duration of this file
    pub duration: Duration,
    /// Message count in this file
    pub message_count: u64,
}

impl BagMetadata {
    /// Load metadata from a metadata.yaml file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|_| ReaderError::MetadataNotFound {
            path: path.to_path_buf(),
        })?;

        let metadata: BagMetadata = serde_yaml::from_str(&content)?;

        // Validate the metadata
        metadata.validate()?;

        Ok(metadata)
    }

    /// Validate the metadata structure
    pub fn validate(&self) -> Result<()> {
        let info = &self.rosbag2_bagfile_information;

        // Check supported version
        if info.version > 9 {
            return Err(ReaderError::UnsupportedVersion {
                version: info.version,
            });
        }

        // Check supported storage formats
        match info.storage_identifier.as_str() {
            "sqlite3" | "mcap" => {}
            "" => {
                // Auto-detect storage format from file extensions when storage_identifier is empty
                let has_db3 = info
                    .relative_file_paths
                    .iter()
                    .any(|path| path.ends_with(".db3"));
                let has_mcap = info
                    .relative_file_paths
                    .iter()
                    .any(|path| path.ends_with(".mcap"));

                if !has_db3 && !has_mcap {
                    return Err(ReaderError::UnsupportedStorageFormat {
                        format: "unknown (no .db3 or .mcap files found)".to_string(),
                    });
                }
            }
            _ => {
                return Err(ReaderError::UnsupportedStorageFormat {
                    format: info.storage_identifier.clone(),
                });
            }
        }

        // Check compression format if specified
        if !info.compression_format.is_empty() && info.compression_format != "zstd" {
            return Err(ReaderError::UnsupportedCompressionFormat {
                format: info.compression_format.clone(),
            });
        }

        // Check serialization formats
        for topic in &info.topics_with_message_count {
            if topic.topic_metadata.serialization_format != "cdr" {
                return Err(ReaderError::UnsupportedSerializationFormat {
                    format: topic.topic_metadata.serialization_format.clone(),
                });
            }
        }

        Ok(())
    }

    /// Get the bag file information
    pub fn info(&self) -> &BagFileInformation {
        &self.rosbag2_bagfile_information
    }

    /// Get the duration in nanoseconds
    pub fn duration(&self) -> u64 {
        self.info().duration.nanoseconds
    }

    /// Get the start time in nanoseconds since epoch
    pub fn start_time(&self) -> u64 {
        self.info().starting_time.nanoseconds_since_epoch
    }

    /// Get the end time in nanoseconds since epoch
    pub fn end_time(&self) -> u64 {
        if self.info().message_count == 0 {
            0
        } else {
            self.start_time() + self.duration()
        }
    }

    /// Get the total message count
    pub fn message_count(&self) -> u64 {
        self.info().message_count
    }

    /// Check if compression is enabled
    pub fn is_compressed(&self) -> bool {
        !self.info().compression_format.is_empty()
    }

    /// Get compression mode
    pub fn compression_mode(&self) -> Option<&str> {
        if self.info().compression_mode.is_empty() {
            None
        } else {
            Some(&self.info().compression_mode)
        }
    }
}
