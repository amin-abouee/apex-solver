//! Storage backend implementations for ROS2 bag files

use crate::rosbag::error::Result;
use crate::rosbag::types::{
    CompressionMode, Connection, Message, MessageDefinition, RawMessage, StoragePlugin,
};
use std::collections::HashMap;
use std::path::Path;

pub mod mcap;
pub mod sqlite;

/// Trait for storage backend implementations (reading)
pub trait StorageReader {
    /// Open the storage files for reading
    fn open(&mut self) -> Result<()>;

    /// Close the storage files
    fn close(&mut self) -> Result<()>;

    /// Get message definitions from the storage
    fn get_definitions(&self) -> Result<HashMap<String, MessageDefinition>>;

    /// Iterate over messages, optionally filtered by connections, start time, and stop time
    fn messages_filtered(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Box<dyn Iterator<Item = Result<Message>> + '_>>;

    /// Iterate over raw messages without deserialization for maximum performance
    fn raw_messages(&self) -> Result<Box<dyn Iterator<Item = Result<RawMessage>> + '_>>;

    /// Iterate over filtered raw messages without deserialization
    fn raw_messages_filtered(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Box<dyn Iterator<Item = Result<RawMessage>> + '_>>;

    /// Read all raw messages as a batch for bulk operations
    fn read_raw_messages_batch(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Vec<RawMessage>>;

    /// Check if the storage is currently open
    fn is_open(&self) -> bool;

    /// Get a reference to the concrete type for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Storage writer trait for writing bag data
pub trait StorageWriter: std::any::Any {
    /// Open the storage for writing
    fn open(&mut self) -> Result<()>;

    /// Close the storage and write any final metadata
    fn close(&mut self, version: u32, metadata: &str) -> Result<()>;

    /// Add a message type definition
    fn add_msgtype(&mut self, connection: &Connection) -> Result<()>;

    /// Add a connection (topic) to the storage
    fn add_connection(&mut self, connection: &Connection, offered_qos_profiles: &str)
    -> Result<()>;

    /// Write a single message to the storage
    fn write(&mut self, connection: &Connection, timestamp: u64, data: &[u8]) -> Result<()>;

    /// Write multiple messages in a batch for better performance
    /// Default implementation falls back to individual writes
    fn write_batch(&mut self, messages: &[(Connection, u64, Vec<u8>)]) -> Result<()> {
        for (connection, timestamp, data) in messages {
            self.write(connection, *timestamp, data)?;
        }
        Ok(())
    }

    /// Check if the storage is open
    fn is_open(&self) -> bool;

    /// Get type-erased reference for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Create a storage reader for the given storage identifier
pub fn create_storage_reader(
    storage_id: &str,
    paths: Vec<&Path>,
    #[allow(unused_variables)] connections: Vec<Connection>,
) -> Result<Box<dyn StorageReader>> {
    match storage_id {
        "sqlite3" => Ok(Box::new(sqlite::SqliteReader::new(paths, connections)?)),
        "mcap" => Ok(Box::new(mcap::McapStorageReader::new(paths, connections)?)),
        "" => {
            // Auto-detect storage format from file extensions when storage_identifier is empty
            let has_db3 = paths
                .iter()
                .any(|path| path.extension().is_some_and(|ext| ext == "db3"));
            let has_mcap = paths
                .iter()
                .any(|path| path.extension().is_some_and(|ext| ext == "mcap"));

            if has_db3 {
                Ok(Box::new(sqlite::SqliteReader::new(paths, connections)?))
            } else if has_mcap {
                Ok(Box::new(mcap::McapStorageReader::new(paths, connections)?))
            } else {
                Err(crate::rosbag::error::BagError::UnsupportedStorageFormat {
                    format: "unknown (no .db3 or .mcap files found)".to_string(),
                })
            }
        }
        _ => Err(crate::rosbag::error::BagError::UnsupportedStorageFormat {
            format: storage_id.to_string(),
        }),
    }
}

/// Create a storage writer for the given storage plugin
pub fn create_storage_writer(
    storage_plugin: StoragePlugin,
    path: &Path,
    compression_mode: CompressionMode,
) -> Result<Box<dyn StorageWriter>> {
    match storage_plugin {
        StoragePlugin::Sqlite3 => Ok(Box::new(sqlite::SqliteWriter::new(path, compression_mode)?)),
        StoragePlugin::Mcap => Ok(Box::new(mcap::McapWriter::new(path, compression_mode)?)),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn create_storage_reader_sqlite3_identifier() {
        let path = PathBuf::from("test.db3");
        let result = create_storage_reader("sqlite3", vec![path.as_path()], vec![]);
        assert!(result.is_ok());
    }

    #[test]
    fn create_storage_reader_mcap_identifier() {
        let path = PathBuf::from("test.mcap");
        let result = create_storage_reader("mcap", vec![path.as_path()], vec![]);
        assert!(result.is_ok());
    }

    #[test]
    fn create_storage_reader_unknown_identifier_returns_err() {
        let path = PathBuf::from("test.hdf5");
        let result = create_storage_reader("hdf5", vec![path.as_path()], vec![]);
        assert!(matches!(
            result,
            Err(crate::rosbag::error::BagError::UnsupportedStorageFormat { .. })
        ));
    }

    #[test]
    fn create_storage_reader_auto_detect_db3() {
        let path = PathBuf::from("autodetect.db3");
        let result = create_storage_reader("", vec![path.as_path()], vec![]);
        assert!(result.is_ok());
    }

    #[test]
    fn create_storage_reader_auto_detect_mcap() {
        let path = PathBuf::from("autodetect.mcap");
        let result = create_storage_reader("", vec![path.as_path()], vec![]);
        assert!(result.is_ok());
    }

    #[test]
    fn create_storage_reader_auto_detect_no_known_extension_returns_err() {
        let path = PathBuf::from("unknown.xyz");
        let result = create_storage_reader("", vec![path.as_path()], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn create_storage_writer_sqlite3_plugin() {
        let path = PathBuf::from("/tmp/test_writer_sqlite3.db3");
        let result = create_storage_writer(StoragePlugin::Sqlite3, &path, CompressionMode::None);
        assert!(result.is_ok());
    }

    #[test]
    fn create_storage_writer_mcap_plugin() {
        let path = PathBuf::from("/tmp/test_writer_mcap.mcap");
        let result = create_storage_writer(StoragePlugin::Mcap, &path, CompressionMode::None);
        assert!(result.is_ok());
    }

    #[test]
    fn storage_writer_write_batch_default_impl() {
        use tempfile::tempdir;
        let dir = tempdir().unwrap();
        let bag_path = dir.path().join("batch_bag");
        std::fs::create_dir_all(&bag_path).unwrap();
        let mut writer =
            create_storage_writer(StoragePlugin::Sqlite3, &bag_path, CompressionMode::None)
                .unwrap();
        writer.open().unwrap();

        let conn = Connection {
            id: 1,
            topic: "/test".to_string(),
            message_type: "std_msgs/msg/String".to_string(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "cdr".to_string(),
            offered_qos_profiles: Vec::new(),
        };
        writer.add_msgtype(&conn).unwrap();
        writer.add_connection(&conn, "").unwrap();

        let msgs: Vec<(Connection, u64, Vec<u8>)> = vec![
            (conn.clone(), 100, vec![0x00, 0x01, 0x00, 0x00, 0x01]),
            (conn.clone(), 200, vec![0x00, 0x01, 0x00, 0x00, 0x02]),
        ];
        writer.write_batch(&msgs).unwrap();
        assert!(writer.is_open());
    }
}
