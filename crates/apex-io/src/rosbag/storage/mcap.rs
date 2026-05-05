//! MCAP storage backend for reading ROS2 bag files
//!
//! This module provides support for reading ROS2 bag files stored in MCAP format.
//! MCAP is a modern, efficient container format for multimodal log data.

use crate::rosbag::error::{BagError, ReaderError, Result};
use crate::rosbag::storage::StorageReader;
use crate::rosbag::types::{Connection, Message, MessageDefinition, RawMessage};
use mcap::MessageStream;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// MCAP storage reader implementation
pub struct McapStorageReader {
    /// Paths to MCAP files
    mcap_paths: Vec<PathBuf>,
    /// Topic connections discovered from MCAP files
    topic_connections: Vec<Connection>,
    /// Whether the storage is currently open
    is_open: bool,
    /// Loaded MCAP file contents (read fully into memory; deref to `&[u8]`).
    mapped_files: Vec<Box<[u8]>>,
}

impl McapStorageReader {
    /// Create a new MCAP storage reader
    pub fn new(paths: Vec<&Path>, connections: Vec<Connection>) -> Result<Self> {
        let mcap_paths: Vec<PathBuf> = paths.iter().map(|p| p.to_path_buf()).collect();

        Ok(Self {
            mcap_paths,
            topic_connections: connections,
            is_open: false,
            mapped_files: Vec::new(),
        })
    }

    /// Get all topics and their message counts directly from MCAP files
    pub fn get_topics_from_mcap(&self) -> Result<Vec<Connection>> {
        let mut topic_map: HashMap<String, (String, u64)> = HashMap::new();

        for mapped_file in &self.mapped_files {
            let message_stream = MessageStream::new(mapped_file.as_ref()).map_err(|e| {
                ReaderError::generic(format!("Failed to create message stream: {e}"))
            })?;

            for message_result in message_stream {
                match message_result {
                    Ok(message) => {
                        let topic_name = &message.channel.topic;
                        let message_type = &message.channel.message_encoding;

                        let entry = topic_map
                            .entry(topic_name.clone())
                            .or_insert((message_type.clone(), 0));
                        entry.1 += 1;
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to read MCAP message: {e}");
                    }
                }
            }
        }

        let all_connections = topic_map
            .into_iter()
            .enumerate()
            .map(|(idx, (topic_name, (message_type, count)))| Connection {
                id: (idx + 1) as u32,
                topic: topic_name,
                message_type,
                message_definition: MessageDefinition::default(),
                type_description_hash: String::new(),
                message_count: count,
                serialization_format: "cdr".to_string(),
                offered_qos_profiles: Vec::new(),
            })
            .collect();

        Ok(all_connections)
    }
}

impl StorageReader for McapStorageReader {
    fn open(&mut self) -> Result<()> {
        self.mapped_files.clear();

        for path in &self.mcap_paths {
            let bytes: Box<[u8]> = std::fs::read(path)
                .map_err(|e| {
                    ReaderError::generic(format!(
                        "Failed to read MCAP file {}: {}",
                        path.display(),
                        e
                    ))
                })?
                .into_boxed_slice();

            self.mapped_files.push(bytes);
        }

        self.is_open = true;
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.mapped_files.clear();
        self.is_open = false;
        Ok(())
    }

    fn get_definitions(&self) -> Result<HashMap<String, MessageDefinition>> {
        Ok(HashMap::new())
    }

    fn messages_filtered(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Box<dyn Iterator<Item = Result<Message>> + '_>> {
        let mut all_messages = Vec::new();

        for mapped_file in &self.mapped_files {
            let message_stream = MessageStream::new(mapped_file).map_err(|e| {
                ReaderError::generic(format!("Failed to create message stream: {e}"))
            })?;

            for message_result in message_stream {
                match message_result {
                    Ok(message) => {
                        if let Some(conns) = connections {
                            if !conns.iter().any(|c| c.topic == message.channel.topic) {
                                continue;
                            }
                        }

                        let timestamp = message.log_time;
                        if start.is_some_and(|t| timestamp < t) {
                            continue;
                        }
                        if stop.is_some_and(|t| timestamp > t) {
                            continue;
                        }

                        let connection = self
                            .topic_connections
                            .iter()
                            .find(|c| c.topic == message.channel.topic)
                            .cloned()
                            .unwrap_or_else(|| Connection {
                                id: 1,
                                topic: message.channel.topic.clone(),
                                message_type: message.channel.message_encoding.clone(),
                                message_definition: MessageDefinition::default(),
                                type_description_hash: String::new(),
                                message_count: 0,
                                serialization_format: "cdr".to_string(),
                                offered_qos_profiles: Vec::new(),
                            });

                        let msg = Message {
                            connection,
                            topic: message.channel.topic.clone(),
                            timestamp,
                            data: message.data.to_vec(),
                        };
                        all_messages.push(Ok(msg));
                    }
                    Err(e) => {
                        all_messages.push(Err(ReaderError::generic(format!(
                            "Failed to read MCAP message: {e}"
                        ))));
                    }
                }
            }
        }

        all_messages.sort_by(|a, b| match (a, b) {
            (Ok(msg_a), Ok(msg_b)) => msg_a.timestamp.cmp(&msg_b.timestamp),
            _ => std::cmp::Ordering::Equal,
        });

        Ok(Box::new(all_messages.into_iter()))
    }

    fn is_open(&self) -> bool {
        self.is_open
    }

    fn raw_messages(&self) -> Result<Box<dyn Iterator<Item = Result<RawMessage>> + '_>> {
        self.raw_messages_filtered(None, None, None)
    }

    fn raw_messages_filtered(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Box<dyn Iterator<Item = Result<RawMessage>> + '_>> {
        let mut all_messages = Vec::new();

        for mapped_file in &self.mapped_files {
            let message_stream = MessageStream::new(mapped_file).map_err(|e| {
                ReaderError::generic(format!("Failed to create message stream: {e}"))
            })?;

            for message_result in message_stream {
                match message_result {
                    Ok(message) => {
                        if let Some(conns) = connections {
                            if !conns.iter().any(|c| c.topic == message.channel.topic) {
                                continue;
                            }
                        }

                        let timestamp = message.log_time;
                        if start.is_some_and(|t| timestamp < t) {
                            continue;
                        }
                        if stop.is_some_and(|t| timestamp > t) {
                            continue;
                        }

                        let connection = self
                            .topic_connections
                            .iter()
                            .find(|c| c.topic == message.channel.topic)
                            .cloned()
                            .unwrap_or_else(|| Connection {
                                id: 1,
                                topic: message.channel.topic.clone(),
                                message_type: message.channel.message_encoding.clone(),
                                message_definition: MessageDefinition::default(),
                                type_description_hash: String::new(),
                                message_count: 0,
                                serialization_format: "cdr".to_string(),
                                offered_qos_profiles: Vec::new(),
                            });

                        let raw_msg = RawMessage {
                            connection,
                            timestamp,
                            raw_data: message.data.to_vec(),
                        };
                        all_messages.push(Ok(raw_msg));
                    }
                    Err(e) => {
                        all_messages.push(Err(ReaderError::generic(format!(
                            "Failed to read MCAP message: {e}"
                        ))));
                    }
                }
            }
        }

        all_messages.sort_by(|a, b| match (a, b) {
            (Ok(msg_a), Ok(msg_b)) => msg_a.timestamp.cmp(&msg_b.timestamp),
            _ => std::cmp::Ordering::Equal,
        });

        Ok(Box::new(all_messages.into_iter()))
    }

    fn read_raw_messages_batch(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Vec<RawMessage>> {
        let mut all_messages = Vec::new();

        for mapped_file in &self.mapped_files {
            let message_stream = MessageStream::new(mapped_file).map_err(|e| {
                ReaderError::generic(format!("Failed to create message stream: {e}"))
            })?;

            for message_result in message_stream {
                match message_result {
                    Ok(message) => {
                        if let Some(conns) = connections {
                            if !conns.iter().any(|c| c.topic == message.channel.topic) {
                                continue;
                            }
                        }

                        let timestamp = message.log_time;
                        if start.is_some_and(|t| timestamp < t) {
                            continue;
                        }
                        if stop.is_some_and(|t| timestamp > t) {
                            continue;
                        }

                        let connection = self
                            .topic_connections
                            .iter()
                            .find(|c| c.topic == message.channel.topic)
                            .cloned()
                            .unwrap_or_else(|| Connection {
                                id: 1,
                                topic: message.channel.topic.clone(),
                                message_type: message.channel.message_encoding.clone(),
                                message_definition: MessageDefinition::default(),
                                type_description_hash: String::new(),
                                message_count: 0,
                                serialization_format: "cdr".to_string(),
                                offered_qos_profiles: Vec::new(),
                            });

                        all_messages.push(RawMessage {
                            connection,
                            timestamp,
                            raw_data: message.data.to_vec(),
                        });
                    }
                    Err(e) => {
                        return Err(ReaderError::generic(format!(
                            "Failed to read MCAP message: {e}"
                        )));
                    }
                }
            }
        }

        all_messages.sort_by_key(|a| a.timestamp);
        Ok(all_messages)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// MCAP storage writer (minimal placeholder — MCAP reading is fully supported)
pub struct McapWriter {
    mcap_path: PathBuf,
    writer: Option<File>,
    _compression_mode: crate::rosbag::types::CompressionMode,
    is_open: bool,
    schemas: Vec<Schema>,
    channels: Vec<Channel>,
    next_schema_id: u64,
    next_channel_id: u64,
    channel_id_map: HashMap<String, u64>,
}

impl McapWriter {
    /// Create a new MCAP writer
    pub fn new(
        path: &Path,
        compression_mode: crate::rosbag::types::CompressionMode,
    ) -> Result<Self> {
        let file_name = path
            .file_name()
            .ok_or_else(|| BagError::writer("bag path has no file name component"))?
            .to_string_lossy();
        let mcap_path = path.join(format!("{file_name}.mcap"));

        Ok(Self {
            mcap_path,
            writer: None,
            _compression_mode: compression_mode,
            is_open: false,
            schemas: Vec::new(),
            channels: Vec::new(),
            next_schema_id: 1,
            next_channel_id: 1,
            channel_id_map: HashMap::new(),
        })
    }

    fn write_header(&mut self) -> Result<()> {
        if let Some(writer) = &mut self.writer {
            writer.write_all(b"\x89MCAP0\r\n")?;
        }
        Ok(())
    }
}

impl crate::rosbag::storage::StorageWriter for McapWriter {
    fn open(&mut self) -> Result<()> {
        if self.is_open {
            return Err(BagError::BagAlreadyOpen);
        }

        let file = File::create(&self.mcap_path)?;
        self.writer = Some(file);
        self.write_header()?;
        self.is_open = true;
        Ok(())
    }

    fn close(&mut self, _version: u32, _metadata: &str) -> Result<()> {
        if !self.is_open {
            return Ok(());
        }

        if let Some(writer) = &mut self.writer {
            writer.flush()?;
        }

        self.writer = None;
        self.is_open = false;
        self.schemas.clear();
        self.channels.clear();
        self.channel_id_map.clear();
        self.next_schema_id = 1;
        self.next_channel_id = 1;

        Ok(())
    }

    fn add_msgtype(&mut self, connection: &Connection) -> Result<()> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }

        let schema = Schema {
            id: self.next_schema_id,
            name: connection.message_type.clone(),
            _encoding: "ros2msg".to_string(),
            _data: connection.message_definition.data.as_bytes().to_vec(),
        };

        self.schemas.push(schema);
        self.next_schema_id += 1;
        Ok(())
    }

    fn add_connection(
        &mut self,
        connection: &Connection,
        _offered_qos_profiles: &str,
    ) -> Result<()> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }

        let schema_id = self
            .schemas
            .iter()
            .find(|s| s.name == connection.message_type)
            .map(|s| s.id)
            .unwrap_or(0);

        let channel = Channel {
            _id: self.next_channel_id,
            _schema_id: schema_id,
            _topic: connection.topic.clone(),
            _message_encoding: connection.message_type.clone(),
            _metadata: HashMap::new(),
        };

        self.channel_id_map
            .insert(connection.topic.clone(), self.next_channel_id);
        self.channels.push(channel);
        self.next_channel_id += 1;
        Ok(())
    }

    fn write(&mut self, connection: &Connection, _timestamp: u64, _data: &[u8]) -> Result<()> {
        if !self.is_open {
            return Err(BagError::BagNotOpen);
        }

        let _channel_id = self
            .channel_id_map
            .get(&connection.topic)
            .ok_or_else(|| BagError::connection_not_found(&connection.topic))?;

        if self.writer.is_none() {
            return Err(BagError::BagNotOpen);
        }

        Ok(())
    }

    fn is_open(&self) -> bool {
        self.is_open
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::rosbag::storage::StorageWriter;
    use crate::rosbag::types::{CompressionMode, MessageDefinition};
    use tempfile::tempdir;

    fn make_connection(topic: &str, msg_type: &str) -> Connection {
        Connection {
            id: 1,
            topic: topic.to_string(),
            message_type: msg_type.to_string(),
            message_definition: MessageDefinition::default(),
            type_description_hash: String::new(),
            message_count: 0,
            serialization_format: "cdr".to_string(),
            offered_qos_profiles: Vec::new(),
        }
    }

    #[test]
    fn mcap_reader_new_stores_paths() {
        let path = std::path::PathBuf::from("test.mcap");
        let reader = McapStorageReader::new(vec![path.as_path()], vec![]).unwrap();
        assert!(!reader.is_open());
    }

    #[test]
    fn mcap_writer_not_open_initially() {
        let dir = tempdir().unwrap();
        let bag_path = dir.path().join("test_bag");
        let writer = McapWriter::new(&bag_path, CompressionMode::None).unwrap();
        assert!(!writer.is_open());
    }

    #[test]
    fn mcap_writer_open_and_close() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("test_bag")).unwrap();
        let bag_path = dir.path().join("test_bag");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None).unwrap();
        writer.open().unwrap();
        assert!(writer.is_open());
        writer.close(9, "{}").unwrap();
        assert!(!writer.is_open());
    }

    #[test]
    fn mcap_writer_double_open_returns_err() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("bag2")).unwrap();
        let bag_path = dir.path().join("bag2");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None).unwrap();
        writer.open().unwrap();
        assert!(writer.open().is_err());
    }

    #[test]
    fn mcap_writer_write_without_open_returns_err() {
        let dir = tempdir().unwrap();
        let bag_path = dir.path().join("bag3");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None).unwrap();
        let conn = make_connection("/test", "std_msgs/msg/String");
        assert!(writer.write(&conn, 0, b"data").is_err());
    }

    #[test]
    fn mcap_writer_add_msgtype_and_connection() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("bag4")).unwrap();
        let bag_path = dir.path().join("bag4");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None).unwrap();
        writer.open().unwrap();
        let conn = make_connection("/imu", "sensor_msgs/msg/Imu");
        writer.add_msgtype(&conn).unwrap();
        writer.add_connection(&conn, "").unwrap();
        assert!(writer.is_open());
    }

    #[test]
    fn mcap_reader_open_missing_file_returns_err() {
        let path = std::path::PathBuf::from("/nonexistent/path/to.mcap");
        let mut reader = McapStorageReader::new(vec![path.as_path()], vec![]).unwrap();
        assert!(reader.open().is_err());
    }

    #[test]
    fn mcap_reader_get_definitions_returns_empty() {
        let path = std::path::PathBuf::from("test.mcap");
        let reader = McapStorageReader::new(vec![path.as_path()], vec![]).unwrap();
        let defs = reader.get_definitions().unwrap();
        assert!(defs.is_empty());
    }

    #[test]
    fn mcap_reader_as_any_returns_self() {
        let path = std::path::PathBuf::from("test.mcap");
        let reader = McapStorageReader::new(vec![path.as_path()], vec![]).unwrap();
        let any = reader.as_any();
        assert!(any.is::<McapStorageReader>());
    }

    #[test]
    fn mcap_writer_as_any_returns_self() {
        let dir = tempdir().unwrap();
        let bag_path = dir.path().join("as_any_bag");
        let writer = McapWriter::new(&bag_path, CompressionMode::None).unwrap();
        let any = <McapWriter as crate::rosbag::storage::StorageWriter>::as_any(&writer);
        assert!(any.is::<McapWriter>());
    }

    #[test]
    fn mcap_reader_close_when_not_open_ok() {
        let path = std::path::PathBuf::from("test.mcap");
        let mut reader = McapStorageReader::new(vec![path.as_path()], vec![]).unwrap();
        // Close without opening should be a noop
        reader.close().unwrap();
        assert!(!reader.is_open());
    }

    #[test]
    fn mcap_writer_add_msgtype_without_open_fails() {
        let dir = tempdir().unwrap();
        let bag_path = dir.path().join("closed_bag");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None).unwrap();
        let conn = make_connection("/imu", "sensor_msgs/msg/Imu");
        assert!(writer.add_msgtype(&conn).is_err());
    }

    #[test]
    fn mcap_writer_add_connection_without_open_fails() {
        let dir = tempdir().unwrap();
        let bag_path = dir.path().join("closed_bag2");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None).unwrap();
        let conn = make_connection("/imu", "sensor_msgs/msg/Imu");
        assert!(writer.add_connection(&conn, "").is_err());
    }

    #[test]
    fn mcap_writer_close_when_not_open_is_noop() {
        let dir = tempdir().unwrap();
        let bag_path = dir.path().join("noop_close");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None).unwrap();
        // Close without opening should be a noop
        writer.close(9, "{}").unwrap();
        assert!(!writer.is_open());
    }
}

#[derive(Debug, Clone)]
struct Schema {
    id: u64,
    name: String,
    _encoding: String,
    _data: Vec<u8>,
}

#[derive(Debug, Clone)]
struct Channel {
    _id: u64,
    _schema_id: u64,
    _topic: String,
    _message_encoding: String,
    _metadata: HashMap<String, String>,
}
