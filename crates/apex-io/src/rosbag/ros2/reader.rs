//! Main reader implementation for ROS2 bag files
//!
//! Two bag layouts are supported:
//!
//! - a **bag directory** with a `metadata.yaml` describing its storage files
//!   (the ROS2 bag format); and
//! - a **bare file**, e.g. a standalone `record.mcap` with no metadata.yaml.
//!   Its topics, message counts, and time range are read from the MCAP
//!   summary section on [`Reader::open`], so it works for datasets shipped as
//!   single files (HILTI SLAM Challenge bags, for example). Such a reader's
//!   connections report each channel's message encoding (e.g. `ros1`) as
//!   `serialization_format` and its schema name (e.g. `sensor_msgs/Image`)
//!   as `message_type`; [`Reader::metadata`] returns `None`.

use super::metadata::BagMetadata;
use super::storage::mcap::{McapStorageReader, McapSummaryStats};
use super::storage::{StorageReader, create_storage_reader};
use crate::rosbag::error::{BagError, ReaderError, Result};
use crate::rosbag::types::{Connection, Message, MessageDefinition, RawMessage, TopicInfo};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Largest frame we pre-size a decode buffer for (guards corrupt headers).
const MAX_MESSAGE_FRAME: u64 = 1 << 30;

/// Decodes rosbag2 MESSAGE-compressed payloads with one reused zstd context.
struct MessageDecoder {
    decompressor: zstd::bulk::Decompressor<'static>,
}

impl MessageDecoder {
    fn new() -> Result<Self> {
        Ok(Self {
            decompressor: zstd::bulk::Decompressor::new()
                .map_err(|e| ReaderError::generic(format!("zstd init failed: {e}")))?,
        })
    }

    /// Decode one payload. A payload that is not a zstd frame is a corrupt
    /// bag and an error -- never silently passed through.
    fn decode(&mut self, data: Vec<u8>) -> Result<Vec<u8>> {
        let fail =
            |e: std::io::Error| ReaderError::generic(format!("MESSAGE-compressed payload: {e}"));
        match zstd::zstd_safe::get_frame_content_size(&data) {
            // Fast path: the frame announces its size (rosbag2 one-shot compress).
            Ok(Some(size)) if size <= MAX_MESSAGE_FRAME => self
                .decompressor
                .decompress(&data, size as usize)
                .map_err(fail),
            // Streaming frames carry no size; let the decoder grow the output.
            Ok(_) => zstd::stream::decode_all(&data[..]).map_err(fail),
            Err(_) => Err(ReaderError::generic(
                "MESSAGE-compressed payload does not start with a zstd frame",
            )),
        }
    }

    fn raw(&mut self, mut m: RawMessage) -> Result<RawMessage> {
        m.raw_data = self.decode(std::mem::take(&mut m.raw_data))?;
        Ok(m)
    }

    fn message(&mut self, mut m: Message) -> Result<Message> {
        m.data = self.decode(std::mem::take(&mut m.data))?;
        Ok(m)
    }
}

/// Main reader for ROS2 bag files
pub struct Reader {
    /// Path to the bag directory, or to the bare bag file itself
    bag_path: PathBuf,
    /// Parsed metadata (`None` for bare-file bags)
    metadata: Option<BagMetadata>,
    /// Storage backend
    storage: Option<Box<dyn StorageReader>>,
    /// Connections (topics) in the bag
    connections: Vec<Connection>,
    /// Whether the reader is currently open
    is_open: bool,
    /// Summary-derived aggregates for bare-file bags (no metadata.yaml).
    standalone_stats: Option<McapSummaryStats>,
    /// rosbag2 MESSAGE compression: every payload is a zstd frame
    /// (declared only in metadata.yaml), decoded transparently on read.
    message_compression: bool,
}

impl Reader {
    /// Create a new reader for the given bag path
    ///
    /// `bag_path` may be a bag directory (must contain `metadata.yaml`) or a
    /// bare `.mcap` file. Anything else is an error: a bare `.bag` (ROS1)
    /// file belongs to [`crate::rosbag::ros1::Ros1Reader`], and a directory
    /// without `metadata.yaml` is not a ROS2 bag.
    pub fn new<P: AsRef<Path>>(bag_path: P) -> Result<Self> {
        let bag_path = bag_path.as_ref().to_path_buf();

        if !bag_path.exists() {
            return Err(ReaderError::BagNotFound { path: bag_path });
        }

        if bag_path.is_file() {
            if !bag_path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("mcap"))
            {
                return Err(ReaderError::UnsupportedStorageFormat {
                    format: format!(
                        "bare bag file '{}' must be .mcap (a bag directory with metadata.yaml, \
                         or a ROS1 .bag via Ros1Reader, otherwise)",
                        bag_path.display()
                    ),
                });
            }
            return Ok(Self {
                bag_path,
                metadata: None,
                storage: None,
                connections: Vec::new(),
                is_open: false,
                standalone_stats: None,
                message_compression: false,
            });
        }

        let metadata_path = bag_path.join("metadata.yaml");
        let metadata = BagMetadata::from_file(&metadata_path)?;

        Ok(Self {
            bag_path,
            metadata: Some(metadata),
            storage: None,
            connections: Vec::new(),
            is_open: false,
            standalone_stats: None,
            message_compression: false,
        })
    }

    /// Open the bag for reading
    pub fn open(&mut self) -> Result<()> {
        if self.is_open {
            return Ok(());
        }

        // Bare-file path: no metadata.yaml -- the MCAP summary is the source
        // of truth for topics, counts, and the time range.
        if self.metadata.is_none() {
            let file_path = self.bag_path.clone();
            let mut storage = create_storage_reader("mcap", vec![file_path.as_path()], Vec::new())?;
            storage.open()?;

            if let Some(mcap_storage) = storage.as_any().downcast_ref::<McapStorageReader>() {
                self.connections = mcap_storage.get_topics_from_mcap()?;
                self.standalone_stats = Some(mcap_storage.summary_stats()?);
            }

            self.storage = Some(storage);
            self.is_open = true;
            return Ok(());
        }

        let metadata = self.metadata.as_ref().ok_or(BagError::BagNotOpen)?;
        let info = metadata.info();

        // MESSAGE compression: payloads are zstd frames. Anything else is
        // rejected loudly rather than handing out compressed bytes.
        let message_compressed = info.compression_mode.eq_ignore_ascii_case("message");
        if message_compressed && info.compression_format != "zstd" {
            return Err(ReaderError::UnsupportedCompressionFormat {
                format: info.compression_format.clone(),
            });
        }

        self.connections = info
            .topics_with_message_count
            .iter()
            .enumerate()
            .map(|(idx, topic)| {
                let qos_profiles = match &topic.topic_metadata.offered_qos_profiles {
                    super::metadata::QosProfilesField::String(_) => Vec::new(),
                    super::metadata::QosProfilesField::List(profiles) => profiles.clone(),
                };

                Connection {
                    id: (idx + 1) as u32,
                    topic: topic.topic_metadata.name.clone(),
                    message_type: topic.topic_metadata.message_type.clone(),
                    message_definition: MessageDefinition::default(),
                    type_description_hash: topic.topic_metadata.type_description_hash.clone(),
                    message_count: topic.message_count,
                    serialization_format: topic.topic_metadata.serialization_format.clone(),
                    offered_qos_profiles: qos_profiles,
                }
            })
            .collect();

        let storage_paths: Vec<PathBuf> = info
            .relative_file_paths
            .iter()
            .map(|path| self.bag_path.join(path))
            .collect();

        for path in &storage_paths {
            if !path.exists() {
                return Err(ReaderError::StorageFileNotFound { path: path.clone() });
            }
        }

        let storage_path_refs: Vec<&Path> = storage_paths.iter().map(|p| p.as_path()).collect();
        let mut storage = create_storage_reader(
            &info.storage_identifier,
            storage_path_refs,
            self.connections.clone(),
        )?;

        storage.open()?;
        self.message_compression = message_compressed;

        // Get actual topics from the storage (may be more complete than metadata)
        if let Some(sqlite_storage) = storage
            .as_any()
            .downcast_ref::<super::storage::sqlite::SqliteReader>()
        {
            match sqlite_storage.get_topics_from_database() {
                Ok(db_connections) => {
                    if !db_connections.is_empty() {
                        self.connections = db_connections;
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to read topics from database: {e}");
                }
            }
        }

        if let Some(mcap_storage) = storage
            .as_any()
            .downcast_ref::<super::storage::mcap::McapStorageReader>()
        {
            match mcap_storage.get_topics_from_mcap() {
                Ok(mcap_connections) => {
                    if !mcap_connections.is_empty() {
                        for mcap_conn in &mcap_connections {
                            if let Some(metadata_conn) = self
                                .connections
                                .iter_mut()
                                .find(|c| c.topic == mcap_conn.topic)
                            {
                                metadata_conn.message_count = mcap_conn.message_count;
                            } else {
                                self.connections.push(mcap_conn.clone());
                            }
                        }

                        if self.connections.is_empty() {
                            self.connections = mcap_connections;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to read topics from MCAP: {e}");
                }
            }
        }

        let definitions = storage.get_definitions()?;
        for connection in &mut self.connections {
            if let Some(def) = definitions.get(&connection.message_type) {
                connection.message_definition = def.clone();
            }
        }

        self.storage = Some(storage);
        self.is_open = true;

        Ok(())
    }

    /// Close the bag
    pub fn close(&mut self) -> Result<()> {
        if !self.is_open {
            return Ok(());
        }

        if let Some(mut storage) = self.storage.take() {
            storage.close()?;
        }

        self.is_open = false;
        Ok(())
    }

    /// Get the bag duration in nanoseconds
    ///
    /// For bare-file bags this comes from the MCAP summary, read at
    /// [`open`](Self::open); 0 before open.
    pub fn duration(&self) -> u64 {
        if let Some(stats) = &self.standalone_stats {
            return stats.end_time.saturating_sub(stats.start_time);
        }
        self.metadata.as_ref().map_or(0, |m| m.duration())
    }

    /// Get the start time in nanoseconds since epoch
    pub fn start_time(&self) -> u64 {
        if let Some(stats) = &self.standalone_stats {
            return stats.start_time;
        }
        self.metadata.as_ref().map_or(0, |m| m.start_time())
    }

    /// Get the end time in nanoseconds since epoch
    pub fn end_time(&self) -> u64 {
        if let Some(stats) = &self.standalone_stats {
            return stats.end_time;
        }
        self.metadata.as_ref().map_or(0, |m| m.end_time())
    }

    /// Get the total message count
    pub fn message_count(&self) -> u64 {
        if let Some(stats) = &self.standalone_stats {
            return stats.message_count;
        }
        self.metadata.as_ref().map_or(0, |m| m.message_count())
    }

    /// Get information about all topics in the bag
    pub fn topics(&self) -> Vec<TopicInfo> {
        if !self.is_open {
            return Vec::new();
        }

        let mut topic_map: HashMap<String, TopicInfo> = HashMap::new();

        for connection in &self.connections {
            let topic_info = topic_map
                .entry(connection.topic.clone())
                .or_insert_with(|| TopicInfo {
                    name: connection.topic.clone(),
                    message_type: connection.message_type.clone(),
                    message_definition: connection.message_definition.clone(),
                    message_count: 0,
                    connections: Vec::new(),
                });

            topic_info.message_count += connection.message_count;
            topic_info.connections.push(connection.clone());
        }

        let mut topics: Vec<TopicInfo> = topic_map.into_values().collect();
        topics.sort_by(|a, b| a.name.cmp(&b.name));
        topics
    }

    /// Get all connections
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Iterate over all messages in the bag
    pub fn messages(&self) -> Result<Box<dyn Iterator<Item = Result<Message>> + '_>> {
        self.messages_filtered(None, None, None)
    }

    /// Iterate over messages with optional filters
    pub fn messages_filtered(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Box<dyn Iterator<Item = Result<Message>> + '_>> {
        if !self.is_open {
            return Err(ReaderError::BagNotOpen);
        }

        let storage = self.storage.as_ref().ok_or(BagError::BagNotOpen)?;
        let iterator = storage.messages_filtered(connections, start, stop)?;
        if self.message_compression {
            let mut decoder = MessageDecoder::new()?;
            return Ok(Box::new(
                iterator.map(move |m| m.and_then(|m| decoder.message(m))),
            ));
        }
        Ok(iterator)
    }

    /// Get raw message data without deserialization for maximum performance
    pub fn raw_messages(&self) -> Result<Box<dyn Iterator<Item = Result<RawMessage>> + '_>> {
        if !self.is_open {
            return Err(ReaderError::BagNotOpen);
        }

        let storage = self.storage.as_ref().ok_or(BagError::BagNotOpen)?;
        let iterator = storage.raw_messages()?;
        if self.message_compression {
            let mut decoder = MessageDecoder::new()?;
            return Ok(Box::new(
                iterator.map(move |m| m.and_then(|m| decoder.raw(m))),
            ));
        }
        Ok(iterator)
    }

    /// Get filtered raw message data without deserialization
    pub fn raw_messages_filtered(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Box<dyn Iterator<Item = Result<RawMessage>> + '_>> {
        if !self.is_open {
            return Err(ReaderError::BagNotOpen);
        }

        let storage = self.storage.as_ref().ok_or(BagError::BagNotOpen)?;
        let iterator = storage.raw_messages_filtered(connections, start, stop)?;
        if self.message_compression {
            let mut decoder = MessageDecoder::new()?;
            return Ok(Box::new(
                iterator.map(move |m| m.and_then(|m| decoder.raw(m))),
            ));
        }
        Ok(iterator)
    }

    /// Read all messages in raw format as a batch for bulk operations
    pub fn read_raw_messages_batch(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Vec<RawMessage>> {
        if !self.is_open {
            return Err(ReaderError::BagNotOpen);
        }

        let storage = self.storage.as_ref().ok_or(BagError::BagNotOpen)?;
        let batch = storage.read_raw_messages_batch(connections, start, stop)?;
        if self.message_compression {
            let mut decoder = MessageDecoder::new()?;
            return batch.into_iter().map(|m| decoder.raw(m)).collect();
        }
        Ok(batch)
    }

    /// Check if the bag is open
    pub fn is_open(&self) -> bool {
        self.is_open
    }

    /// Get the metadata
    ///
    /// `None` for bare-file bags, which have no metadata.yaml; use the
    /// summary-derived [`duration`](Self::duration) /
    /// [`start_time`](Self::start_time) / [`end_time`](Self::end_time) /
    /// [`message_count`](Self::message_count) and
    /// [`connections`](Self::connections) instead.
    pub fn metadata(&self) -> Option<&BagMetadata> {
        self.metadata.as_ref()
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests {

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_metadata() -> String {
        r#"
rosbag2_bagfile_information:
  version: 4
  storage_identifier: sqlite3
  relative_file_paths:
    - test.db3
  duration:
    nanoseconds: 1000000000
  starting_time:
    nanoseconds_since_epoch: 1234567890000000000
  message_count: 10
  compression_format: ""
  compression_mode: ""
  topics_with_message_count:
    - topic_metadata:
        name: /test_topic
        type: std_msgs/msg/String
        serialization_format: cdr
        offered_qos_profiles: ""
        type_description_hash: ""
      message_count: 10
"#
        .trim()
        .to_string()
    }

    #[test]
    fn test_reader_creation_with_missing_bag() -> TestResult {
        let result = Reader::new("/nonexistent/path");
        assert!(matches!(result, Err(ReaderError::BagNotFound { .. })));
        Ok(())
    }

    #[test]
    fn test_reader_creation_with_missing_metadata() -> TestResult {
        let temp_dir = TempDir::new()?;
        let result = Reader::new(temp_dir.path());
        assert!(matches!(result, Err(ReaderError::MetadataNotFound { .. })));
        Ok(())
    }

    #[test]
    fn test_reader_creation_success() -> TestResult {
        let temp_dir = TempDir::new()?;
        let metadata_path = temp_dir.path().join("metadata.yaml");
        fs::write(&metadata_path, create_test_metadata())?;

        let db_path = temp_dir.path().join("test.db3");
        fs::write(&db_path, b"")?;

        let reader = Reader::new(temp_dir.path());
        assert!(reader.is_ok());

        let reader = reader?;
        assert!(!reader.is_open());
        assert_eq!(reader.duration(), 1000000000);
        assert_eq!(reader.message_count(), 10);
        Ok(())
    }

    const SQLITE3_BAG: &str = "tests/test_bags/test_bag_sqlite3";

    #[test]
    fn test_reader_topics() -> TestResult {
        let mut reader = Reader::new(SQLITE3_BAG)?;
        reader.open()?;
        let topics = reader.topics();
        assert!(!topics.is_empty());
        assert_eq!(topics.len(), 94);
        Ok(())
    }

    #[test]
    fn test_reader_start_end_time() -> TestResult {
        let mut reader = Reader::new(SQLITE3_BAG)?;
        reader.open()?;
        assert!(reader.start_time() > 0);
        assert!(reader.end_time() > reader.start_time());
        Ok(())
    }

    #[test]
    fn test_reader_raw_messages_filtered_by_connection() -> TestResult {
        let mut reader = Reader::new(SQLITE3_BAG)?;
        reader.open()?;
        let conns: Vec<_> = reader
            .connections()
            .iter()
            .filter(|c| c.topic == "/test/std_msgs/string")
            .cloned()
            .collect();
        let count = reader
            .raw_messages_filtered(Some(&conns), None, None)?
            .filter_map(|r| r.ok())
            .count();
        assert_eq!(count, 2);
        Ok(())
    }

    #[test]
    fn test_reader_messages_without_open_returns_err() -> TestResult {
        let temp_dir = TempDir::new()?;
        let metadata_path = temp_dir.path().join("metadata.yaml");
        fs::write(&metadata_path, create_test_metadata())?;
        let db_path = temp_dir.path().join("test.db3");
        fs::write(&db_path, b"")?;
        let reader = Reader::new(temp_dir.path())?;
        // Not opened → storage is None → messages() should return Err
        assert!(reader.messages().is_err());
        Ok(())
    }

    /// Write a minimal but valid ros1-profile MCAP with `n` messages.
    fn write_bare_mcap(path: &std::path::Path, n: u32) -> TestResult {
        use std::borrow::Cow;
        use std::collections::BTreeMap;
        use std::sync::Arc;

        let file = fs::File::create(path)?;
        let mut writer = mcap::WriteOptions::new()
            .compression(Some(mcap::Compression::Zstd))
            .create(file)?;
        let schema = Arc::new(mcap::Schema {
            name: "sensor_msgs/Imu".to_string(),
            encoding: "ros1msg".to_string(),
            data: Cow::Borrowed(&b"sensor_msgs/Imu definition"[..]),
        });
        let channel = Arc::new(mcap::Channel {
            schema: Some(schema),
            topic: "/imu".to_string(),
            message_encoding: "ros1".to_string(),
            metadata: BTreeMap::new(),
        });
        for i in 0..n {
            writer.write(&mcap::Message {
                channel: channel.clone(),
                sequence: i,
                log_time: 1_700_000_000_000_000_000 + u64::from(i),
                publish_time: 1_700_000_000_000_000_000 + u64::from(i),
                data: Cow::Owned(vec![0; 4]),
            })?;
        }
        writer.finish()?;
        Ok(())
    }

    #[test]
    fn test_reader_new_with_bare_mcap_file_is_standalone() -> TestResult {
        let temp_dir = TempDir::new()?;
        let mcap_path = temp_dir.path().join("record.mcap");
        write_bare_mcap(&mcap_path, 3)?;

        let mut reader = Reader::new(&mcap_path)?;
        assert!(reader.metadata().is_none());
        assert_eq!(reader.duration(), 0, "summary is only read on open");

        reader.open()?;
        assert_eq!(reader.message_count(), 3);
        assert_eq!(reader.start_time(), 1_700_000_000_000_000_000);
        assert_eq!(reader.end_time(), 1_700_000_000_000_000_000 + 2);
        assert_eq!(reader.duration(), 2);

        let connections = reader.connections();
        assert_eq!(connections.len(), 1);
        // Schema name as message type, encoding as serialization format.
        assert_eq!(connections[0].message_type, "sensor_msgs/Imu");
        assert_eq!(connections[0].serialization_format, "ros1");
        assert_eq!(connections[0].message_count, 3);

        let count = reader.raw_messages()?.filter_map(Result::ok).count();
        assert_eq!(count, 3);
        Ok(())
    }

    #[test]
    fn test_reader_new_with_bare_non_mcap_file_errors() -> TestResult {
        let temp_dir = TempDir::new()?;
        let bag_path = temp_dir.path().join("record.bag");
        fs::write(&bag_path, b"#ROSBAG V2.0")?;
        assert!(matches!(
            Reader::new(&bag_path),
            Err(ReaderError::UnsupportedStorageFormat { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_reader_raw_messages_without_open_returns_err() -> TestResult {
        let temp_dir = TempDir::new()?;
        let metadata_path = temp_dir.path().join("metadata.yaml");
        fs::write(&metadata_path, create_test_metadata())?;
        let db_path = temp_dir.path().join("test.db3");
        fs::write(&db_path, b"")?;
        let reader = Reader::new(temp_dir.path())?;
        assert!(reader.raw_messages().is_err());
        Ok(())
    }
}
