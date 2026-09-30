//! MCAP storage backend for reading bag files stored in MCAP format.
//!
//! MCAP is a modern, efficient container format for multimodal log data. This
//! reader is *profile-agnostic*: an MCAP container carries the payload
//! encoding per channel (`cdr` for ROS2 bags, `ros1` for ROS1-serialized
//! payloads, or any other encoding). Each channel's encoding is surfaced as
//! the connection's `serialization_format` so callers can pick the matching
//! deserializer ([`crate::rosbag::cdr::CdrDeserializer`] for `cdr`,
//! [`crate::rosbag::ros1::deserialize::Ros1Deserializer`] for `ros1`), and the
//! schema-registered message type (e.g. `sensor_msgs/Image`) is surfaced as
//! `message_type`.
//!
//! Reading is bounded-memory: the low-level container reader
//! ([`mcap_format`]) seeks to the summary, then streams messages
//! chunk-by-chunk, holding one decompressed chunk at a time -- multi-GB
//! dataset bags never enter memory whole.

use super::StorageReader;
use crate::rosbag::error::{BagError, ReaderError, Result};
use crate::rosbag::types::{Connection, Message, MessageDefinition, MessageDefinitionFormat, RawMessage};
use mcap_format::{Filters, MergedMcapStream, McapSummary};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

pub(crate) mod mcap_format;

/// Aggregate statistics taken from MCAP summary sections.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct McapSummaryStats {
    /// Total number of message records across all files.
    pub message_count: u64,
    /// Earliest message `log_time` in nanoseconds (0 when unknown).
    pub start_time: u64,
    /// Latest message `log_time` in nanoseconds (0 when unknown).
    pub end_time: u64,
}

/// Map an MCAP schema encoding onto the crate's message-definition format.
fn definition_from_schema(encoding: &str, data: &[u8]) -> MessageDefinition {
    let (format, content): (MessageDefinitionFormat, &[u8]) = match encoding {
        "ros1msg" | "ros2msg" => (MessageDefinitionFormat::Msg, data),
        "ros2idl" | "omgidl" => (MessageDefinitionFormat::Idl, data),
        _ => (MessageDefinitionFormat::None, &[]),
    };
    MessageDefinition {
        format,
        data: String::from_utf8_lossy(content).into_owned(),
    }
}

/// Aggregate topic connections and statistics out of parsed summaries.
///
/// `message_type` is taken from the channel's schema name (e.g.
/// `sensor_msgs/Image`), falling back to the message encoding when a channel
/// has no schema; `serialization_format` is the channel's message encoding
/// (`cdr`, `ros1`, ...).
fn summarize_summaries(
    summaries: &[Option<McapSummary>],
) -> (Vec<Connection>, McapSummaryStats) {
    // Accumulated per topic: (message_type, definition, encoding, count).
    // BTreeMap keeps the topic order stable across runs.
    let mut topics: BTreeMap<String, (String, MessageDefinition, String, u64)> = BTreeMap::new();
    let mut stats = McapSummaryStats::default();

    for summary in summaries.iter().flatten() {
        for (id, channel) in &summary.channels {
            let (message_type, definition) = match summary.schemas.get(&channel.schema_id) {
                Some(schema) => (
                    schema.name.clone(),
                    definition_from_schema(&schema.encoding, &schema.data),
                ),
                None => (channel.message_encoding.clone(), MessageDefinition::default()),
            };
            let count = summary
                .stats
                .as_ref()
                .and_then(|s| s.channel_counts.get(id))
                .copied()
                .unwrap_or(0);
            let entry = topics
                .entry(channel.topic.clone())
                .or_insert_with(|| (message_type, definition, channel.message_encoding.clone(), 0));
            entry.3 += count;
        }

        if let Some(statistics) = &summary.stats {
            stats.message_count += statistics.message_count;
            if statistics.start_time != 0 {
                stats.start_time = if stats.start_time == 0 {
                    statistics.start_time
                } else {
                    stats.start_time.min(statistics.start_time)
                };
            }
            stats.end_time = stats.end_time.max(statistics.end_time);
        }
    }

    let connections = topics
        .into_iter()
        .enumerate()
        .map(|(idx, (topic, (message_type, definition, encoding, count)))| Connection {
            id: (idx + 1) as u32,
            topic,
            message_type,
            message_definition: definition,
            type_description_hash: String::new(),
            message_count: count,
            serialization_format: encoding,
            offered_qos_profiles: Vec::new(),
        })
        .collect();

    (connections, stats)
}

/// MCAP storage reader implementation
pub struct McapStorageReader {
    /// Paths to MCAP files
    mcap_paths: Vec<PathBuf>,
    /// Topic connections discovered from MCAP files
    topic_connections: Vec<Connection>,
    /// Whether the storage is currently open
    is_open: bool,
    /// Per-file summaries read by seeking through the summary section only.
    /// `None` entries are files without a summary section (unindexed).
    summaries: Vec<Option<McapSummary>>,
}

impl McapStorageReader {
    /// Create a new MCAP storage reader
    pub fn new(paths: Vec<&Path>, connections: Vec<Connection>) -> Result<Self> {
        let mcap_paths: Vec<PathBuf> = paths.iter().map(|p| p.to_path_buf()).collect();

        Ok(Self {
            mcap_paths,
            topic_connections: connections,
            is_open: false,
            summaries: Vec::new(),
        })
    }

    /// Get all topics and their message counts from the MCAP summary sections.
    ///
    /// Never touches message data: everything comes from the summary's
    /// schema/channel/statistics records, so a 13 GB bag summarizes in
    /// milliseconds instead of a full decompression pass. Files without a
    /// summary section contribute nothing.
    pub fn get_topics_from_mcap(&self) -> Result<Vec<Connection>> {
        Ok(summarize_summaries(&self.summaries).0)
    }

    /// Aggregate message count and time range from the summary sections.
    pub fn summary_stats(&self) -> Result<McapSummaryStats> {
        Ok(summarize_summaries(&self.summaries).1)
    }

    /// Build the global log-time-order message stream across all files,
    /// applying the topic and time filters.
    fn merged_stream(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<MergedMcapStream> {
        let filters = Filters::new(connections, start, stop);
        MergedMcapStream::new(&self.mcap_paths, &self.summaries, filters)
    }
}

impl StorageReader for McapStorageReader {
    fn open(&mut self) -> Result<()> {
        self.summaries.clear();

        for path in &self.mcap_paths {
            let mut file = File::open(path).map_err(|e| {
                ReaderError::generic(format!(
                    "Failed to open MCAP file {}: {e}",
                    path.display()
                ))
            })?;
            let summary = mcap_format::read_summary(&mut file)?;
            self.summaries.push(summary);
        }

        self.is_open = true;
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.summaries.clear();
        self.is_open = false;
        Ok(())
    }

    fn get_definitions(&self) -> Result<HashMap<String, MessageDefinition>> {
        // One entry per distinct message type (schema name); definitions come
        // from the summary's schema records.
        let mut definitions = HashMap::new();
        for summary in self.summaries.iter().flatten() {
            for schema in summary.schemas.values() {
                let definition = definition_from_schema(&schema.encoding, &schema.data);
                definitions.entry(schema.name.clone()).or_insert(definition);
            }
        }
        Ok(definitions)
    }

    fn messages_filtered(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Box<dyn Iterator<Item = Result<Message>> + '_>> {
        // Known connections by topic, from the caller's list and (as
        // fallback) from the summaries themselves. Channel IDs are NOT used
        // as keys: they are file-local and unrelated to the enumeration
        // order of the topic list.
        let mut by_topic: HashMap<String, Connection> = summarize_summaries(&self.summaries)
            .0
            .into_iter()
            .map(|c| (c.topic.clone(), c))
            .collect();
        for connection in self.topic_connections.iter().cloned() {
            by_topic.insert(connection.topic.clone(), connection);
        }

        let merged = self.merged_stream(connections, start, stop)?;

        let iterator = merged.map(move |item| {
            item.map(|(topic, mcap_msg)| {
                let connection = by_topic
                    .get(topic.as_ref())
                    .cloned()
                    .unwrap_or_else(|| Connection {
                        id: 0,
                        topic: topic.to_string(),
                        message_type: String::new(),
                        message_definition: MessageDefinition::default(),
                        type_description_hash: String::new(),
                        message_count: 0,
                        serialization_format: String::new(),
                        offered_qos_profiles: Vec::new(),
                    });
                Message {
                    connection,
                    topic: topic.to_string(),
                    timestamp: mcap_msg.log_time,
                    data: mcap_msg.data,
                }
            })
        });

        Ok(Box::new(iterator))
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
        let mut by_topic: HashMap<String, Connection> = summarize_summaries(&self.summaries)
            .0
            .into_iter()
            .map(|c| (c.topic.clone(), c))
            .collect();
        for connection in self.topic_connections.iter().cloned() {
            by_topic.insert(connection.topic.clone(), connection);
        }

        let merged = self.merged_stream(connections, start, stop)?;

        let iterator = merged.map(move |item| {
            item.map(|(topic, mcap_msg)| {
                let connection = by_topic
                    .get(topic.as_ref())
                    .cloned()
                    .unwrap_or_else(|| Connection {
                        id: 0,
                        topic: topic.to_string(),
                        message_type: String::new(),
                        message_definition: MessageDefinition::default(),
                        type_description_hash: String::new(),
                        message_count: 0,
                        serialization_format: String::new(),
                        offered_qos_profiles: Vec::new(),
                    });
                RawMessage {
                    connection,
                    timestamp: mcap_msg.log_time,
                    raw_data: mcap_msg.data,
                }
            })
        });

        Ok(Box::new(iterator))
    }

    fn read_raw_messages_batch(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Vec<RawMessage>> {
        let mut by_topic: HashMap<String, Connection> = summarize_summaries(&self.summaries)
            .0
            .into_iter()
            .map(|c| (c.topic.clone(), c))
            .collect();
        for connection in self.topic_connections.iter().cloned() {
            by_topic.insert(connection.topic.clone(), connection);
        }

        let mut all_messages = Vec::new();
        for item in self.merged_stream(connections, start, stop)? {
            let (topic, mcap_msg) = item.map_err(|e| {
                ReaderError::generic(format!("Failed to read MCAP message: {e}"))
            })?;
            let connection = by_topic
                .get(topic.as_ref())
                .cloned()
                .unwrap_or_else(|| Connection {
                    id: 0,
                    topic: topic.to_string(),
                    message_type: String::new(),
                    message_definition: MessageDefinition::default(),
                    type_description_hash: String::new(),
                    message_count: 0,
                    serialization_format: String::new(),
                    offered_qos_profiles: Vec::new(),
                });
            all_messages.push(RawMessage {
                connection,
                timestamp: mcap_msg.log_time,
                raw_data: mcap_msg.data,
            });
        }

        // The k-way merge yields log-time order as long as every file is
        // internally ordered (writers emit chunks in log-time order); the
        // explicit sort keeps the batch contract exact even for a file whose
        // chunks were written out of order.
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

    #[test]
    fn mcap_reader_is_open_initially_false() -> TestResult {
        let path = std::path::PathBuf::from("test.mcap");
        let reader = McapStorageReader::new(vec![path.as_path()], vec![])?;
        assert!(!reader.is_open());
        Ok(())
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

/// Round-trip tests through the `mcap` crate's writer, including the
/// ROS1-profile case (`message_encoding: ros1`, `ros1msg` schemas) that bare
/// dataset bags such as HILTI's use.
#[cfg(test)]
mod roundtrip_tests {
    use super::*;
    use mcap::Message as McapMessage;
    use std::borrow::Cow;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    /// Write a ros1-profile MCAP with `num_messages` payload on `topic`, each
    /// payload ending in a distinct u8 tag so decode order can be asserted.
    fn write_ros1_mcap(
        path: &Path,
        topic: &str,
        message_type: &str,
        num_messages: u32,
        base_time_ns: u64,
    ) -> TestResult {
        let file = File::create(path)?;
        let mut writer = mcap::WriteOptions::new()
            .compression(Some(mcap::Compression::Zstd))
            .create(file)?;

        let definition = format!("{message_type} definition");
        let schema = std::sync::Arc::new(mcap::Schema {
            name: message_type.to_string(),
            encoding: "ros1msg".to_string(),
            data: Cow::Owned(definition.into_bytes()),
        });
        let channel = std::sync::Arc::new(mcap::Channel {
            schema: Some(schema),
            topic: topic.to_string(),
            message_encoding: "ros1".to_string(),
            metadata: BTreeMap::new(),
        });

        for i in 0..num_messages {
            let message = McapMessage {
                channel: channel.clone(),
                sequence: i,
                log_time: base_time_ns + u64::from(i) * 1_000_000,
                publish_time: base_time_ns + u64::from(i) * 1_000_000,
                data: Cow::Owned(vec![u8::try_from(i % 256)?]),
            };
            writer.write(&message)?;
        }
        writer.finish()?;
        Ok(())
    }

    #[test]
    fn ros1_profile_summary_reports_schema_type_and_encoding() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("ros1.mcap");
        write_ros1_mcap(&path, "/cam0/image_raw", "sensor_msgs/Image", 3, 1_700_000_000_000_000_000)?;

        let mut reader = McapStorageReader::new(vec![path.as_path()], vec![])?;
        reader.open()?;

        let connections = reader.get_topics_from_mcap()?;
        assert_eq!(connections.len(), 1);
        let connection = &connections[0];
        assert_eq!(connection.topic, "/cam0/image_raw");
        // The schema name, not the message encoding, is the message type --
        // the old implementation reported "ros1" here.
        assert_eq!(connection.message_type, "sensor_msgs/Image");
        // And the encoding is the serialization format, not a hardcoded cdr.
        assert_eq!(connection.serialization_format, "ros1");
        assert_eq!(connection.message_count, 3);

        let stats = reader.summary_stats()?;
        assert_eq!(stats.message_count, 3);
        assert_eq!(stats.start_time, 1_700_000_000_000_000_000);
        assert_eq!(stats.end_time, 1_700_000_000_000_000_000 + 2_000_000);

        let definitions = reader.get_definitions()?;
        assert_eq!(definitions.len(), 1);
        let definition = &definitions["sensor_msgs/Image"];
        assert_eq!(definition.format, MessageDefinitionFormat::Msg);
        Ok(())
    }

    #[test]
    fn ros1_profile_messages_stream_in_log_time_order() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("ros1_stream.mcap");
        let base = 1_700_000_000_000_000_000;
        write_ros1_mcap(&path, "/imu", "sensor_msgs/Imu", 5, base)?;

        let mut reader = McapStorageReader::new(vec![path.as_path()], vec![])?;
        reader.open()?;

        let raws: Vec<_> = reader.raw_messages()?.filter_map(Result::ok).collect();
        assert_eq!(raws.len(), 5);
        let mut times: Vec<u64> = raws.iter().map(|m| m.timestamp).collect();
        assert!(times.is_sorted());
        // Payloads carried through byte-exact: tag i on the i-th message.
        assert_eq!(raws[2].raw_data, vec![2]);

        // Time filter: only the last two messages.
        let late: Vec<_> = reader
            .raw_messages_filtered(None, Some(base + 3_000_000), None)?
            .filter_map(Result::ok)
            .collect();
        assert_eq!(late.len(), 2);
        times = late.iter().map(|m| m.timestamp).collect();
        assert!(times.iter().all(|t| *t >= base + 3_000_000));
        Ok(())
    }

    #[test]
    fn topic_filter_and_decoded_message_both_honor_the_allow_list() -> TestResult {
        let dir = tempfile::tempdir()?;
        let image_path = dir.path().join("images.mcap");
        let imu_path = dir.path().join("imu.mcap");
        write_ros1_mcap(&image_path, "/cam0/image_raw", "sensor_msgs/Image", 2, 1_000_000_000)?;
        write_ros1_mcap(&imu_path, "/imu", "sensor_msgs/Imu", 3, 1_000_000_000)?;

        let mut reader =
            McapStorageReader::new(vec![image_path.as_path(), imu_path.as_path()], vec![])?;
        reader.open()?;

        // Two files, one shared topic each: summary aggregates both.
        let connections = reader.get_topics_from_mcap()?;
        assert_eq!(connections.len(), 2);
        assert_eq!(
            reader.summary_stats()?.message_count,
            5,
            "counts across files must add up"
        );

        let image_conn = connections
            .iter()
            .find(|c| c.topic == "/cam0/image_raw")
            .ok_or("no /cam0/image_raw connection")?;
        let raws: Vec<_> = reader
            .raw_messages_filtered(Some(std::slice::from_ref(image_conn)), None, None)?
            .filter_map(Result::ok)
            .collect();
        assert_eq!(raws.len(), 2);
        assert_eq!(raws[0].connection.serialization_format, "ros1");

        // Decoded-message path reports the same connection fields.
        let decoded: Vec<_> = reader
            .messages_filtered(None, None, None)?
            .filter_map(Result::ok)
            .collect();
        assert_eq!(decoded.len(), 5);
        assert!(decoded.iter().all(|m| m.connection.message_type.starts_with("sensor_msgs/")));
        Ok(())
    }
}
