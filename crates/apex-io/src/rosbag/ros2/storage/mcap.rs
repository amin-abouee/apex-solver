//! MCAP storage backend for reading ROS2 bag files
//!
//! This module provides support for reading ROS2 bag files stored in MCAP format.
//! MCAP is a modern, efficient container format for multimodal log data.

use super::StorageReader;
use crate::rosbag::error::{BagError, ReaderError, Result};
use crate::rosbag::types::{Connection, Message, MessageDefinition, RawMessage};
use mcap::MessageStream;
use std::collections::HashMap;
use std::fs::File;
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
                        let entry = topic_map
                            .entry(topic_name.clone())
                            .or_insert((channel_type(&message.channel), 0));
                        entry.1 += 1;
                    }
                    Err(e) => {
                        tracing::warn!("Failed to read MCAP message: {e}");
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
                        if let Some(conns) = connections
                            && !conns.iter().any(|c| c.topic == message.channel.topic)
                        {
                            continue;
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
                                message_type: channel_type(&message.channel),
                                message_definition: MessageDefinition::default(),
                                type_description_hash: String::new(),
                                message_count: 0,
                                serialization_format: message.channel.message_encoding.clone(),
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
                        if let Some(conns) = connections
                            && !conns.iter().any(|c| c.topic == message.channel.topic)
                        {
                            continue;
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
                                message_type: channel_type(&message.channel),
                                message_definition: MessageDefinition::default(),
                                type_description_hash: String::new(),
                                message_count: 0,
                                serialization_format: message.channel.message_encoding.clone(),
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
                        if let Some(conns) = connections
                            && !conns.iter().any(|c| c.topic == message.channel.topic)
                        {
                            continue;
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
                                message_type: channel_type(&message.channel),
                                message_definition: MessageDefinition::default(),
                                type_description_hash: String::new(),
                                message_count: 0,
                                serialization_format: message.channel.message_encoding.clone(),
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

/// MCAP storage writer: ROS 2 messages (`cdr`) on one channel per topic,
/// each with its `ros2msg` schema, in chunks, with the summary section `mcap`
/// readers index by. Under [`CompressionMode::Message`] each chunk is
/// zstd-compressed — MCAP's per-message compression; under `File` the bag
/// writer compresses the finished file itself, so the chunks stay plain.
///
/// [`CompressionMode::Message`]: crate::rosbag::types::CompressionMode::Message
///
/// Streams: memory holds the open chunk, not the file.
pub struct McapWriter {
    mcap_path: PathBuf,
    compression_mode: crate::rosbag::types::CompressionMode,
    writer: Option<mcap::Writer<'static, std::io::BufWriter<File>>>,
    /// Schemas added, by message type.
    schemas: HashMap<String, std::sync::Arc<mcap::Schema<'static>>>,
    /// Channel ID and next sequence number, by topic.
    channels: HashMap<String, (u16, u32)>,
}

impl McapWriter {
    /// Create a new MCAP writer for the bag directory `path`; the file is
    /// `<path>/<name>.mcap`.
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
            compression_mode,
            writer: None,
            schemas: HashMap::new(),
            channels: HashMap::new(),
        })
    }
}

fn mcap_error(error: mcap::McapError) -> BagError {
    BagError::writer(format!("mcap: {error}"))
}

impl crate::rosbag::storage::StorageWriter for McapWriter {
    fn open(&mut self) -> Result<()> {
        if self.writer.is_some() {
            return Err(BagError::BagAlreadyOpen);
        }
        let file = std::io::BufWriter::new(File::create(&self.mcap_path)?);
        let compression = match self.compression_mode {
            crate::rosbag::types::CompressionMode::Message => Some(mcap::Compression::Zstd),
            _ => None,
        };
        let writer = mcap::WriteOptions::new()
            .compression(compression)
            .profile("ros2")
            .create(file)
            .map_err(mcap_error)?;
        self.writer = Some(writer);
        Ok(())
    }

    fn close(&mut self, _version: u32, _metadata: &str) -> Result<()> {
        let Some(mut writer) = self.writer.take() else {
            return Ok(());
        };
        writer.finish().map_err(mcap_error)?;
        self.schemas.clear();
        self.channels.clear();
        Ok(())
    }

    fn add_msgtype(&mut self, connection: &Connection) -> Result<()> {
        if self.writer.is_none() {
            return Err(BagError::BagNotOpen);
        }
        self.schemas
            .entry(connection.message_type.clone())
            .or_insert_with(|| {
                std::sync::Arc::new(mcap::Schema {
                    name: connection.message_type.clone(),
                    encoding: "ros2msg".to_string(),
                    data: std::borrow::Cow::Owned(
                        connection.message_definition.data.as_bytes().to_vec(),
                    ),
                })
            });
        Ok(())
    }

    fn add_connection(
        &mut self,
        connection: &Connection,
        _offered_qos_profiles: &str,
    ) -> Result<()> {
        let Some(writer) = self.writer.as_mut() else {
            return Err(BagError::BagNotOpen);
        };
        let channel = mcap::Channel {
            topic: connection.topic.clone(),
            schema: self.schemas.get(&connection.message_type).cloned(),
            message_encoding: "cdr".to_string(),
            metadata: std::collections::BTreeMap::new(),
        };
        let id = writer.add_channel(&channel).map_err(mcap_error)?;
        self.channels.insert(connection.topic.clone(), (id, 0));
        Ok(())
    }

    fn write(&mut self, connection: &Connection, timestamp: u64, data: &[u8]) -> Result<()> {
        let Some(writer) = self.writer.as_mut() else {
            return Err(BagError::BagNotOpen);
        };
        let (channel_id, sequence) = self
            .channels
            .get_mut(&connection.topic)
            .ok_or_else(|| BagError::connection_not_found(&connection.topic))?;
        let header = mcap::records::MessageHeader {
            channel_id: *channel_id,
            sequence: *sequence,
            log_time: timestamp,
            publish_time: timestamp,
        };
        *sequence = sequence.wrapping_add(1);
        writer
            .write_to_known_channel(&header, data)
            .map_err(mcap_error)
    }

    fn is_open(&self) -> bool {
        self.writer.is_some()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The message type a channel carries — its schema name, e.g.
/// `geometry_msgs/TransformStamped` — falling back to the encoding for a
/// channel recorded without a schema.
fn channel_type(channel: &mcap::Channel<'_>) -> String {
    channel.schema.as_ref().map_or_else(
        || channel.message_encoding.clone(),
        |schema| schema.name.clone(),
    )
}

#[cfg(test)]
mod tests {

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

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
    fn mcap_reader_new_stores_paths() -> TestResult {
        let path = std::path::PathBuf::from("test.mcap");
        let reader = McapStorageReader::new(vec![path.as_path()], vec![])?;
        assert!(!reader.is_open());
        Ok(())
    }

    #[test]
    fn mcap_writer_not_open_initially() -> TestResult {
        let dir = tempdir()?;
        let bag_path = dir.path().join("test_bag");
        let writer = McapWriter::new(&bag_path, CompressionMode::None)?;
        assert!(!writer.is_open());
        Ok(())
    }

    #[test]
    fn mcap_writer_open_and_close() -> TestResult {
        let dir = tempdir()?;
        std::fs::create_dir_all(dir.path().join("test_bag"))?;
        let bag_path = dir.path().join("test_bag");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None)?;
        writer.open()?;
        assert!(writer.is_open());
        writer.close(9, "{}")?;
        assert!(!writer.is_open());
        Ok(())
    }

    #[test]
    fn mcap_writer_double_open_returns_err() -> TestResult {
        let dir = tempdir()?;
        std::fs::create_dir_all(dir.path().join("bag2"))?;
        let bag_path = dir.path().join("bag2");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None)?;
        writer.open()?;
        assert!(writer.open().is_err());
        Ok(())
    }

    #[test]
    fn mcap_writer_write_without_open_returns_err() -> TestResult {
        let dir = tempdir()?;
        let bag_path = dir.path().join("bag3");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None)?;
        let conn = make_connection("/test", "std_msgs/msg/String");
        assert!(writer.write(&conn, 0, b"data").is_err());
        Ok(())
    }

    #[test]
    fn mcap_writer_add_msgtype_and_connection() -> TestResult {
        let dir = tempdir()?;
        std::fs::create_dir_all(dir.path().join("bag4"))?;
        let bag_path = dir.path().join("bag4");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None)?;
        writer.open()?;
        let conn = make_connection("/imu", "sensor_msgs/msg/Imu");
        writer.add_msgtype(&conn)?;
        writer.add_connection(&conn, "")?;
        assert!(writer.is_open());
        Ok(())
    }

    #[test]
    fn mcap_reader_open_missing_file_returns_err() -> TestResult {
        let path = std::path::PathBuf::from("/nonexistent/path/to.mcap");
        let mut reader = McapStorageReader::new(vec![path.as_path()], vec![])?;
        assert!(reader.open().is_err());
        Ok(())
    }

    #[test]
    fn mcap_reader_get_definitions_returns_empty() -> TestResult {
        let path = std::path::PathBuf::from("test.mcap");
        let reader = McapStorageReader::new(vec![path.as_path()], vec![])?;
        let defs = reader.get_definitions()?;
        assert!(defs.is_empty());
        Ok(())
    }

    #[test]
    fn mcap_reader_as_any_returns_self() -> TestResult {
        let path = std::path::PathBuf::from("test.mcap");
        let reader = McapStorageReader::new(vec![path.as_path()], vec![])?;
        let any = reader.as_any();
        assert!(any.is::<McapStorageReader>());
        Ok(())
    }

    #[test]
    fn mcap_writer_as_any_returns_self() -> TestResult {
        let dir = tempdir()?;
        let bag_path = dir.path().join("as_any_bag");
        let writer = McapWriter::new(&bag_path, CompressionMode::None)?;
        let any = <McapWriter as crate::rosbag::storage::StorageWriter>::as_any(&writer);
        assert!(any.is::<McapWriter>());
        Ok(())
    }

    #[test]
    fn mcap_reader_close_when_not_open_ok() -> TestResult {
        let path = std::path::PathBuf::from("test.mcap");
        let mut reader = McapStorageReader::new(vec![path.as_path()], vec![])?;
        // Close without opening should be a noop
        reader.close()?;
        assert!(!reader.is_open());
        Ok(())
    }

    #[test]
    fn mcap_writer_add_msgtype_without_open_fails() -> TestResult {
        let dir = tempdir()?;
        let bag_path = dir.path().join("closed_bag");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None)?;
        let conn = make_connection("/imu", "sensor_msgs/msg/Imu");
        assert!(writer.add_msgtype(&conn).is_err());
        Ok(())
    }

    #[test]
    fn mcap_writer_add_connection_without_open_fails() -> TestResult {
        let dir = tempdir()?;
        let bag_path = dir.path().join("closed_bag2");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None)?;
        let conn = make_connection("/imu", "sensor_msgs/msg/Imu");
        assert!(writer.add_connection(&conn, "").is_err());
        Ok(())
    }

    #[test]
    fn mcap_writer_close_when_not_open_is_noop() -> TestResult {
        let dir = tempdir()?;
        let bag_path = dir.path().join("noop_close");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None)?;
        // Close without opening should be a noop
        writer.close(9, "{}")?;
        assert!(!writer.is_open());
        Ok(())
    }

    #[test]
    fn mcap_writer_write_with_valid_connection() -> TestResult {
        let dir = tempdir()?;
        std::fs::create_dir_all(dir.path().join("write_bag"))?;
        let bag_path = dir.path().join("write_bag");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None)?;
        writer.open()?;
        let conn = make_connection("/test", "std_msgs/msg/String");
        writer.add_msgtype(&conn)?;
        writer.add_connection(&conn, "")?;
        let result = writer.write(&conn, 1_000_000, &[0x00, 0x01, 0x00, 0x00]);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn mcap_writer_write_unknown_connection_errors() -> TestResult {
        let dir = tempdir()?;
        std::fs::create_dir_all(dir.path().join("unknown_bag"))?;
        let bag_path = dir.path().join("unknown_bag");
        let mut writer = McapWriter::new(&bag_path, CompressionMode::None)?;
        writer.open()?;
        let unknown_conn = make_connection("/nonexistent", "std_msgs/msg/String");
        assert!(writer.write(&unknown_conn, 0, &[]).is_err());
        Ok(())
    }

    #[test]
    fn mcap_reader_open_and_close() -> TestResult {
        let dir = tempdir()?;
        let mcap_path = dir.path().join("test.mcap");
        std::fs::write(&mcap_path, b"\x89MCAP0\r\n")?;
        let mut reader = McapStorageReader::new(vec![mcap_path.as_path()], vec![])?;
        reader.open()?;
        assert!(reader.is_open());
        reader.close()?;
        assert!(!reader.is_open());
        Ok(())
    }

    /// What the writer writes, the reader reads back: every message, on its
    /// topic, at its time, byte for byte — across several chunks, and with
    /// or without compression.
    #[test]
    fn written_messages_are_read_back() -> TestResult {
        use crate::rosbag::types::{CompressionFormat, CompressionMode, StoragePlugin};
        use crate::rosbag::{Reader, Writer};

        for compression in [CompressionMode::None, CompressionMode::Message] {
            let dir = tempfile::tempdir()?;
            let bag = dir.path().join("round_trip");
            let mut writer = Writer::new(&bag, None, Some(StoragePlugin::Mcap))?;
            writer.set_compression(compression, CompressionFormat::None)?;
            writer.open()?;
            let imu = writer.add_connection(
                "/imu".to_string(),
                "sensor_msgs/msg/Imu".to_string(),
                None,
                None,
                None,
                None,
            )?;
            let cloud = writer.add_connection(
                "/cloud".to_string(),
                "sensor_msgs/msg/PointCloud2".to_string(),
                None,
                None,
                None,
                None,
            )?;
            let mut written = Vec::new();
            for k in 0..400u64 {
                let (connection, size) = match k % 4 {
                    0 => (&cloud, 4096),
                    _ => (&imu, 64),
                };
                let data: Vec<u8> = (0..size).map(|i| ((i as u64 + k) % 251) as u8).collect();
                writer.write(connection, 1_000 + k, &data)?;
                written.push((connection.topic.clone(), 1_000 + k, data));
            }
            writer.close()?;

            let mut reader = Reader::new(&bag)?;
            reader.open()?;
            let read: Vec<(String, u64, Vec<u8>)> = reader
                .messages()?
                .map(|m| m.map(|m| (m.topic, m.timestamp, m.data)))
                .collect::<std::result::Result<_, _>>()?;
            assert_eq!(read.len(), written.len(), "{compression:?}");
            let mut read = read;
            read.sort_by_key(|(_, t, _)| *t);
            assert_eq!(read, written, "{compression:?}");
        }
        Ok(())
    }

    #[test]
    fn mcap_reader_is_open_initially_false() -> TestResult {
        let path = std::path::PathBuf::from("test.mcap");
        let reader = McapStorageReader::new(vec![path.as_path()], vec![])?;
        assert!(!reader.is_open());
        Ok(())
    }
}
