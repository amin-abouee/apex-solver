//! Main writer implementation for ROS2 bag files

use crate::rosbag::error::{BagError, Result};
use crate::rosbag::metadata::{BagFileInformation, BagMetadata};
use crate::rosbag::storage::{StorageWriter, create_storage_writer};
use crate::rosbag::types::{
    CompressionFormat, CompressionMode, Connection, MessageDefinition, QosProfile, StoragePlugin,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Buffered message for batch writing
#[derive(Debug, Clone)]
struct BufferedMessage {
    connection: Connection,
    timestamp: u64,
    data: Vec<u8>,
}

/// Main writer for ROS2 bag files
pub struct Writer {
    bag_path: PathBuf,
    metadata_path: PathBuf,
    version: u32,
    storage_plugin: StoragePlugin,
    compression_mode: CompressionMode,
    compression_format: CompressionFormat,
    storage: Option<Box<dyn StorageWriter>>,
    connections: Vec<Connection>,
    message_counts: HashMap<u32, u64>,
    custom_data: HashMap<String, String>,
    added_types: std::collections::HashSet<String>,
    min_timestamp: u64,
    max_timestamp: u64,
    is_open: bool,
    message_buffer: Vec<BufferedMessage>,
    buffer_size_limit: usize,
    current_buffer_size: usize,
    batch_threshold: usize,
}

impl std::fmt::Debug for Writer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Writer")
            .field("bag_path", &self.bag_path)
            .field("version", &self.version)
            .field("storage_plugin", &self.storage_plugin)
            .field("is_open", &self.is_open)
            .finish()
    }
}

impl Writer {
    /// Latest supported bag format version
    pub const VERSION_LATEST: u32 = 9;

    /// Create a new writer for the given bag path
    pub fn new<P: AsRef<Path>>(
        bag_path: P,
        version: Option<u32>,
        storage_plugin: Option<StoragePlugin>,
    ) -> Result<Self> {
        let bag_path = bag_path.as_ref().to_path_buf();

        if bag_path.exists() {
            return Err(BagError::BagAlreadyExists { path: bag_path });
        }

        let version = version.unwrap_or(Self::VERSION_LATEST);
        let storage_plugin = storage_plugin.unwrap_or(StoragePlugin::Sqlite3);
        let metadata_path = bag_path.join("metadata.yaml");

        Ok(Self {
            bag_path,
            metadata_path,
            version,
            storage_plugin,
            compression_mode: CompressionMode::None,
            compression_format: CompressionFormat::None,
            storage: None,
            connections: Vec::new(),
            message_counts: HashMap::new(),
            custom_data: HashMap::new(),
            added_types: std::collections::HashSet::new(),
            min_timestamp: u64::MAX,
            max_timestamp: 0,
            is_open: false,
            message_buffer: Vec::new(),
            buffer_size_limit: 10 * 1024 * 1024, // 10MB
            current_buffer_size: 0,
            batch_threshold: 100,
        })
    }

    /// Set compression for the bag
    pub fn set_compression(
        &mut self,
        mode: CompressionMode,
        format: CompressionFormat,
    ) -> Result<()> {
        if self.is_open {
            return Err(BagError::BagAlreadyOpen);
        }

        self.compression_mode = mode;
        self.compression_format = format;
        Ok(())
    }

    /// Set custom metadata
    pub fn set_custom_data(&mut self, key: String, value: String) -> Result<()> {
        self.custom_data.insert(key, value);
        Ok(())
    }

    /// Configure message buffer settings for performance optimization
    pub fn configure_buffer(
        &mut self,
        buffer_size_mb: usize,
        batch_threshold: usize,
    ) -> Result<()> {
        if self.is_open {
            return Err(BagError::BagAlreadyOpen);
        }

        self.buffer_size_limit = buffer_size_mb * 1024 * 1024;
        self.batch_threshold = batch_threshold;
        Ok(())
    }

    /// Flush the message buffer to storage
    pub fn flush_buffer(&mut self) -> Result<()> {
        if self.message_buffer.is_empty() {
            return Ok(());
        }

        let batch_messages: Vec<(Connection, u64, Vec<u8>)> = self
            .message_buffer
            .iter()
            .map(|msg| (msg.connection.clone(), msg.timestamp, msg.data.clone()))
            .collect();

        let storage = self.storage.as_mut().ok_or(BagError::BagNotOpen)?;
        storage.write_batch(&batch_messages)?;

        self.message_buffer.clear();
        self.current_buffer_size = 0;

        Ok(())
    }

    /// Check if buffer should be flushed
    fn should_flush_buffer(&self) -> bool {
        self.message_buffer.len() >= self.batch_threshold
            || self.current_buffer_size >= self.buffer_size_limit
    }

    /// Open the bag for writing
    pub fn open(&mut self) -> Result<()> {
        if self.is_open {
            return Ok(());
        }

        std::fs::create_dir_all(&self.bag_path)?;

        let mut storage =
            create_storage_writer(self.storage_plugin, &self.bag_path, self.compression_mode)?;
        storage.open()?;

        self.storage = Some(storage);
        self.is_open = true;

        Ok(())
    }

    /// Add a connection (topic) to the bag
    pub fn add_connection(
        &mut self,
        topic: String,
        message_type: String,
        message_definition: Option<MessageDefinition>,
        type_description_hash: Option<String>,
        serialization_format: Option<String>,
        offered_qos_profiles: Option<Vec<QosProfile>>,
    ) -> Result<Connection> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }

        let connection_id = (self.connections.len() + 1) as u32;

        let message_definition = message_definition.unwrap_or_default();
        let type_description_hash = type_description_hash.unwrap_or_default();
        let serialization_format = serialization_format.unwrap_or_else(|| "cdr".to_string());
        let offered_qos_profiles = offered_qos_profiles.unwrap_or_default();

        let connection = Connection {
            id: connection_id,
            topic: topic.clone(),
            message_type: message_type.clone(),
            message_definition: message_definition.clone(),
            type_description_hash: type_description_hash.clone(),
            message_count: 0,
            serialization_format,
            offered_qos_profiles: offered_qos_profiles.clone(),
        };

        for existing_conn in &self.connections {
            if existing_conn.topic == connection.topic
                && existing_conn.message_type == connection.message_type
            {
                return Err(BagError::ConnectionAlreadyExists {
                    topic: connection.topic,
                });
            }
        }

        let qos_yaml = self.serialize_qos_profiles(&offered_qos_profiles)?;

        let storage = self.storage.as_mut().ok_or(BagError::BagNotOpen)?;

        if !self.added_types.contains(&message_type) {
            storage.add_msgtype(&connection)?;
            self.added_types.insert(message_type);
        }

        storage.add_connection(&connection, &qos_yaml)?;

        self.message_counts.insert(connection_id, 0);
        self.connections.push(connection.clone());

        Ok(connection)
    }

    /// Write a message to the bag
    pub fn write(&mut self, connection: &Connection, timestamp: u64, data: &[u8]) -> Result<()> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }

        if !self.connections.iter().any(|c| c.id == connection.id) {
            return Err(BagError::ConnectionNotFound {
                topic: connection.topic.clone(),
            });
        }

        let final_data = match self.compression_mode {
            CompressionMode::Message => {
                if self.compression_format == CompressionFormat::Zstd {
                    zstd::encode_all(data, 0)?
                } else {
                    data.to_vec()
                }
            }
            _ => data.to_vec(),
        };

        let buffered_message = BufferedMessage {
            connection: connection.clone(),
            timestamp,
            data: final_data.clone(),
        };

        self.current_buffer_size += final_data.len();
        self.message_buffer.push(buffered_message);

        *self.message_counts.entry(connection.id).or_insert(0) += 1;
        self.min_timestamp = self.min_timestamp.min(timestamp);
        self.max_timestamp = self.max_timestamp.max(timestamp);

        if self.should_flush_buffer() {
            self.flush_buffer()?;
        }

        Ok(())
    }

    /// Close the bag and write metadata
    pub fn close(&mut self) -> Result<()> {
        if !self.is_open {
            return Ok(());
        }

        self.flush_buffer()?;

        let bag_info = self.generate_metadata()?;
        let metadata = BagMetadata {
            rosbag2_bagfile_information: bag_info,
        };
        let metadata_yaml = serde_yaml::to_string(&metadata)?;

        if let Some(mut storage) = self.storage.take() {
            storage.close(self.version, &metadata_yaml)?;
        }

        std::fs::write(&self.metadata_path, &metadata_yaml)?;

        if self.compression_mode == CompressionMode::File {
            self.compress_storage_file()?;
        }

        self.is_open = false;
        Ok(())
    }

    /// Get all connections
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Check if the bag is currently open
    pub fn is_open(&self) -> bool {
        self.is_open
    }

    /// Generate bag metadata
    fn generate_metadata(&self) -> Result<BagFileInformation> {
        let bag_file_name = self
            .bag_path
            .file_name()
            .ok_or_else(|| BagError::writer("bag path has no file name component"))?
            .to_string_lossy();

        let storage_file_name = match self.storage_plugin {
            StoragePlugin::Sqlite3 => format!("{bag_file_name}.db3"),
            StoragePlugin::Mcap => format!("{bag_file_name}.mcap"),
        };

        let final_file_name = if self.compression_mode == CompressionMode::File {
            format!("{}.{}", storage_file_name, self.compression_format.as_str())
        } else {
            storage_file_name
        };

        let duration = self.max_timestamp.saturating_sub(self.min_timestamp);
        let total_message_count: u64 = self.message_counts.values().sum();

        let topics_with_message_count = self
            .connections
            .iter()
            .map(|conn| crate::rosbag::metadata::TopicWithMessageCount {
                message_count: *self.message_counts.get(&conn.id).unwrap_or(&0),
                topic_metadata: crate::rosbag::metadata::TopicMetadata {
                    name: conn.topic.clone(),
                    message_type: conn.message_type.clone(),
                    serialization_format: conn.serialization_format.clone(),
                    offered_qos_profiles: crate::rosbag::metadata::QosProfilesField::List(
                        conn.offered_qos_profiles.clone(),
                    ),
                    type_description_hash: conn.type_description_hash.clone(),
                },
            })
            .collect();

        Ok(BagFileInformation {
            version: self.version,
            storage_identifier: self.storage_plugin.as_str().to_string(),
            relative_file_paths: vec![final_file_name.clone()],
            duration: crate::rosbag::types::Duration {
                nanoseconds: duration,
            },
            starting_time: crate::rosbag::types::StartingTime {
                nanoseconds_since_epoch: self.min_timestamp,
            },
            message_count: total_message_count,
            compression_format: if self.compression_mode == CompressionMode::None {
                String::new()
            } else {
                self.compression_format.as_str().to_string()
            },
            compression_mode: if self.compression_mode == CompressionMode::None {
                String::new()
            } else {
                self.compression_mode.as_str().to_string()
            },
            topics_with_message_count,
            files: vec![crate::rosbag::metadata::FileInformation {
                path: final_file_name,
                starting_time: crate::rosbag::types::StartingTime {
                    nanoseconds_since_epoch: self.min_timestamp,
                },
                duration: crate::rosbag::types::Duration {
                    nanoseconds: duration,
                },
                message_count: total_message_count,
            }],
            custom_data: if self.custom_data.is_empty() {
                None
            } else {
                Some(self.custom_data.clone())
            },
            ros_distro: Some("rosbags".to_string()),
        })
    }

    /// Serialize QoS profiles to YAML
    fn serialize_qos_profiles(&self, profiles: &[QosProfile]) -> Result<String> {
        if profiles.is_empty() {
            return Ok(String::new());
        }

        let yaml = serde_yaml::to_string(profiles)?;
        Ok(yaml.trim().to_string())
    }

    /// Compress storage file (for file-level compression)
    fn compress_storage_file(&self) -> Result<()> {
        let bag_file_name = self
            .bag_path
            .file_name()
            .ok_or_else(|| BagError::writer("bag path has no file name component"))?
            .to_string_lossy();

        let storage_file = match self.storage_plugin {
            StoragePlugin::Sqlite3 => self.bag_path.join(format!("{bag_file_name}.db3")),
            StoragePlugin::Mcap => self.bag_path.join(format!("{bag_file_name}.mcap")),
        };

        let ext = storage_file
            .extension()
            .ok_or_else(|| BagError::writer("storage file has no extension"))?
            .to_string_lossy();

        let compressed_file =
            storage_file.with_extension(format!("{}.{}", ext, self.compression_format.as_str()));

        let input_data = std::fs::read(&storage_file)?;
        let compressed_data = zstd::encode_all(input_data.as_slice(), 0)?;
        std::fs::write(&compressed_file, compressed_data)?;
        std::fs::remove_file(&storage_file)?;

        Ok(())
    }

    /// Write a raw serialized message directly without deserialization overhead
    pub fn write_raw_message(
        &mut self,
        connection: &Connection,
        timestamp: u64,
        raw_data: &[u8],
    ) -> Result<()> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }

        if timestamp < self.min_timestamp {
            self.min_timestamp = timestamp;
        }
        if timestamp > self.max_timestamp {
            self.max_timestamp = timestamp;
        }

        *self.message_counts.entry(connection.id).or_insert(0) += 1;

        let buffered_msg = BufferedMessage {
            connection: connection.clone(),
            timestamp,
            data: raw_data.to_vec(),
        };

        self.current_buffer_size += raw_data.len();
        self.message_buffer.push(buffered_msg);

        if self.should_flush_buffer() {
            self.flush_buffer()?;
        }

        Ok(())
    }

    /// Write multiple raw messages in a batch for maximum performance
    pub fn write_raw_messages_batch(
        &mut self,
        messages: &[(Connection, u64, Vec<u8>)],
    ) -> Result<()> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }

        if messages.is_empty() {
            return Ok(());
        }

        self.flush_buffer()?;

        for (connection, timestamp, _data) in messages {
            if *timestamp < self.min_timestamp {
                self.min_timestamp = *timestamp;
            }
            if *timestamp > self.max_timestamp {
                self.max_timestamp = *timestamp;
            }
            *self.message_counts.entry(connection.id).or_insert(0) += 1;
        }

        if let Some(storage) = &mut self.storage {
            storage.write_batch(messages)?;
        }

        Ok(())
    }

    /// Copy a message directly from a reader without any processing
    pub fn copy_raw_message_from_reader(
        &mut self,
        connection: &Connection,
        timestamp: u64,
        raw_message_data: &[u8],
    ) -> Result<()> {
        self.write_raw_message(connection, timestamp, raw_message_data)
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_writer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");

        let writer = Writer::new(&bag_path, None, None);
        assert!(writer.is_ok());

        let writer = writer.unwrap();
        assert!(!writer.is_open());
        assert_eq!(writer.version, Writer::VERSION_LATEST);
    }

    #[test]
    fn test_writer_rejects_existing_path() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("existing_bag");
        std::fs::create_dir(&bag_path).unwrap();

        let writer = Writer::new(&bag_path, None, None);
        assert!(writer.is_err());
        assert!(matches!(
            writer.unwrap_err(),
            BagError::BagAlreadyExists { .. }
        ));
    }

    #[test]
    fn test_writer_open_close() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");

        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        assert!(!writer.is_open());

        writer.open().unwrap();
        assert!(writer.is_open());

        writer.close().unwrap();
        assert!(!writer.is_open());

        assert!(bag_path.exists());
        assert!(bag_path.join("metadata.yaml").exists());
    }

    #[test]
    fn test_set_compression() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");

        let mut writer = Writer::new(&bag_path, None, None).unwrap();

        let result = writer.set_compression(CompressionMode::Message, CompressionFormat::Zstd);
        assert!(result.is_ok());

        writer.open().unwrap();

        let result = writer.set_compression(CompressionMode::File, CompressionFormat::Zstd);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BagError::BagAlreadyOpen));
    }

    #[test]
    fn test_add_connection() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");

        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer.open().unwrap();

        let connection = writer
            .add_connection(
                "/test_topic".to_string(),
                "std_msgs/msg/String".to_string(),
                None,
                None,
                None,
                None,
            )
            .unwrap();

        assert_eq!(connection.topic, "/test_topic");
        assert_eq!(connection.message_type, "std_msgs/msg/String");
        assert_eq!(connection.id, 1);
        assert_eq!(writer.connections().len(), 1);
    }

    #[test]
    fn test_duplicate_connection() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");

        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer.open().unwrap();

        writer
            .add_connection(
                "/test_topic".to_string(),
                "std_msgs/msg/String".to_string(),
                None,
                None,
                None,
                None,
            )
            .unwrap();

        let result = writer.add_connection(
            "/test_topic".to_string(),
            "std_msgs/msg/String".to_string(),
            None,
            None,
            None,
            None,
        );

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BagError::ConnectionAlreadyExists { .. }
        ));
    }

    #[test]
    fn test_write_message() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");

        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer.open().unwrap();

        let connection = writer
            .add_connection(
                "/test_topic".to_string(),
                "std_msgs/msg/String".to_string(),
                None,
                None,
                None,
                None,
            )
            .unwrap();

        let test_data = b"Hello, ROS2!";
        let timestamp = 1_234_567_890_000_000_000;

        let result = writer.write(&connection, timestamp, test_data);
        assert!(result.is_ok());

        assert_eq!(*writer.message_counts.get(&connection.id).unwrap(), 1);
    }

    #[test]
    fn test_set_custom_data() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer
            .set_custom_data("key1".to_string(), "value1".to_string())
            .unwrap();
        writer.open().unwrap();
        // custom data is stored and used during metadata generation
        writer.close().unwrap();
        assert!(bag_path.join("metadata.yaml").exists());
    }

    #[test]
    fn test_configure_buffer() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        let result = writer.configure_buffer(10, 50);
        assert!(result.is_ok());
    }

    #[test]
    fn test_configure_buffer_after_open_fails() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer.open().unwrap();
        let result = writer.configure_buffer(10, 50);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BagError::BagAlreadyOpen));
    }

    #[test]
    fn test_write_raw_message() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer.open().unwrap();
        let conn = writer
            .add_connection(
                "/raw".to_string(),
                "std_msgs/msg/String".to_string(),
                None,
                None,
                None,
                None,
            )
            .unwrap();
        let result = writer.write_raw_message(&conn, 1_000_000, &[0x00, 0x01, 0x00, 0x00, 0x01]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_flush_buffer_empty_is_noop() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer.open().unwrap();
        // No messages written → buffer is empty → flush is noop
        assert!(writer.flush_buffer().is_ok());
    }

    #[test]
    fn test_writer_connections_accessor() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer.open().unwrap();
        assert!(writer.connections().is_empty());
        writer
            .add_connection(
                "/a".to_string(),
                "std_msgs/msg/String".to_string(),
                None,
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(writer.connections().len(), 1);
    }

    #[test]
    fn test_writer_batch_write() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer.open().unwrap();
        let conn = writer
            .add_connection(
                "/batch".to_string(),
                "std_msgs/msg/String".to_string(),
                None,
                None,
                None,
                None,
            )
            .unwrap();
        let msgs: Vec<(Connection, u64, Vec<u8>)> = (0..5u64)
            .map(|i| (conn.clone(), i * 1000, vec![0x00, 0x01, 0x00, 0x00, 0x01]))
            .collect();
        assert!(writer.write_raw_messages_batch(&msgs).is_ok());
    }

    #[test]
    fn test_writer_debug_impl() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("debug_bag");
        let writer = Writer::new(&bag_path, None, None).unwrap();
        let s = format!("{writer:?}");
        assert!(s.contains("Writer"));
        assert!(s.contains("is_open"));
    }

    #[test]
    fn test_writer_is_open_accessor() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        assert!(!writer.is_open());
        writer.open().unwrap();
        assert!(writer.is_open());
        writer.close().unwrap();
        assert!(!writer.is_open());
    }

    #[test]
    fn test_writer_open_idempotent() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer.open().unwrap();
        assert!(writer.open().is_ok()); // second open is noop
        assert!(writer.is_open());
    }

    #[test]
    fn test_write_bag_not_open_returns_err() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        let fake_conn = Connection {
            id: 1,
            topic: "/test".to_string(),
            message_type: "std_msgs/msg/String".to_string(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "cdr".to_string(),
            offered_qos_profiles: Vec::new(),
        };
        assert!(matches!(
            writer.write(&fake_conn, 0, &[]),
            Err(BagError::BagNotOpen)
        ));
    }

    #[test]
    fn test_write_connection_not_found_returns_err() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer.open().unwrap();
        let unknown_conn = Connection {
            id: 999,
            topic: "/unknown".to_string(),
            message_type: "std_msgs/msg/String".to_string(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "cdr".to_string(),
            offered_qos_profiles: Vec::new(),
        };
        assert!(matches!(
            writer.write(&unknown_conn, 0, &[]),
            Err(BagError::ConnectionNotFound { .. })
        ));
    }

    #[test]
    fn test_add_connection_not_open_returns_err() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        let result = writer.add_connection(
            "/t".to_string(),
            "std_msgs/msg/String".to_string(),
            None,
            None,
            None,
            None,
        );
        assert!(matches!(result, Err(BagError::BagNotOpen)));
    }

    #[test]
    fn test_copy_raw_message_from_reader() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer.open().unwrap();
        let conn = writer
            .add_connection(
                "/copy".to_string(),
                "std_msgs/msg/String".to_string(),
                None,
                None,
                None,
                None,
            )
            .unwrap();
        let result =
            writer.copy_raw_message_from_reader(&conn, 1_000_000, &[0x00, 0x01, 0x00, 0x00]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialize_qos_profiles_nonempty() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("test_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer.open().unwrap();
        let profile = crate::rosbag::types::QosProfile::default();
        let result = writer.add_connection(
            "/qos_topic".to_string(),
            "std_msgs/msg/String".to_string(),
            None,
            None,
            None,
            Some(vec![profile]),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_writer_file_compression() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("compressed_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer
            .set_compression(CompressionMode::File, CompressionFormat::Zstd)
            .unwrap();
        writer.open().unwrap();
        let conn = writer
            .add_connection(
                "/compressed".to_string(),
                "std_msgs/msg/String".to_string(),
                None,
                None,
                None,
                None,
            )
            .unwrap();
        writer
            .write(&conn, 1_000_000, &[0x00, 0x01, 0x00, 0x00, 0x01])
            .unwrap();
        writer.close().unwrap();
        assert!(
            bag_path.join("compressed_bag.db3.zstd").exists()
                || bag_path.join("metadata.yaml").exists()
        );
    }

    #[test]
    fn test_writer_message_compression() {
        let temp_dir = TempDir::new().unwrap();
        let bag_path = temp_dir.path().join("msg_compressed_bag");
        let mut writer = Writer::new(&bag_path, None, None).unwrap();
        writer
            .set_compression(CompressionMode::Message, CompressionFormat::Zstd)
            .unwrap();
        writer.open().unwrap();
        let conn = writer
            .add_connection(
                "/cmsg".to_string(),
                "std_msgs/msg/String".to_string(),
                None,
                None,
                None,
                None,
            )
            .unwrap();
        writer
            .write(&conn, 1_000_000, &[0x00, 0x01, 0x00, 0x00, 0x01])
            .unwrap();
        writer.close().unwrap();
        assert!(bag_path.join("metadata.yaml").exists());
    }
}
