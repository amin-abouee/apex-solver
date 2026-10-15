//! MCAP storage backend for ROS2 bag files.
//!
//! Reading streams: [`McapMessageStream`] pulls the file through mcap's
//! sans-io [`LinearReader`](mcap::sans_io::linear_reader::LinearReader) a
//! record at a time, so memory holds the chunk being read, not the file — a
//! Hilti recording is 5-20 GB. Messages come out in file order, which is
//! log-time order for a file written in time order, as recorders and
//! [`McapWriter`] write them.

use super::StorageReader;
use crate::rosbag::error::{BagError, ReaderError, Result};
use crate::rosbag::types::{Connection, Message, MessageDefinition, RawMessage};
use mcap::sans_io::linear_reader::{LinearReadEvent, LinearReader, LinearReaderOptions};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// One message read from an MCAP file, with what its channel says about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamedMessage {
    /// Its channel's topic.
    pub topic: String,
    /// Its schema's name — the message type, e.g. `sensor_msgs/msg/Imu` —
    /// or the channel's encoding when it has no schema.
    pub message_type: String,
    /// Its channel's message encoding, e.g. `cdr` or `ros1`.
    pub encoding: String,
    /// Log time \[ns\].
    pub log_time: u64,
    /// The serialized message.
    pub data: Vec<u8>,
}

/// Which messages of an MCAP file a stream yields. A message outside it is
/// passed over where it sits in its chunk, before its payload is copied.
#[derive(Debug, Clone, Default)]
pub struct StreamFilter {
    /// The topics wanted; every topic when `None`.
    pub topics: Option<HashSet<String>>,
    /// The earliest log time wanted \[ns\].
    pub start: Option<u64>,
    /// The latest log time wanted \[ns\].
    pub stop: Option<u64>,
}

impl StreamFilter {
    fn wants_topic(&self, topic: &str) -> bool {
        self.topics
            .as_ref()
            .is_none_or(|topics| topics.contains(topic))
    }

    fn wants_time(&self, log_time: u64) -> bool {
        self.start.is_none_or(|t| log_time >= t) && self.stop.is_none_or(|t| log_time <= t)
    }
}

/// A chunk read from the file and not yet decompressed.
struct PendingChunk {
    compression: String,
    uncompressed_size: u64,
    data: Vec<u8>,
}

/// The schemas and channels seen so far, and the wanted messages found.
struct Demux {
    filter: StreamFilter,
    /// Schema names by ID.
    schemas: HashMap<u16, String>,
    /// Topic, message type and encoding by channel ID.
    channels: HashMap<u16, (String, String, String)>,
    /// The channels whose topic the filter wants.
    wanted: HashSet<u16>,
    /// Wanted messages, in file order, not yet handed out.
    ready: VecDeque<StreamedMessage>,
}

impl Demux {
    fn new(filter: StreamFilter) -> Self {
        Self {
            filter,
            schemas: HashMap::new(),
            channels: HashMap::new(),
            wanted: HashSet::new(),
            ready: VecDeque::new(),
        }
    }

    /// Take one record: remember a schema or a channel; queue a message if
    /// its channel and time are wanted.
    fn absorb(&mut self, record: mcap::records::Record<'_>) -> std::result::Result<(), String> {
        match record {
            mcap::records::Record::Schema { header, .. } => {
                self.schemas.insert(header.id, header.name);
            }
            mcap::records::Record::Channel(channel) => {
                let message_type = self
                    .schemas
                    .get(&channel.schema_id)
                    .cloned()
                    .unwrap_or_else(|| channel.message_encoding.clone());
                if self.filter.wants_topic(&channel.topic) {
                    self.wanted.insert(channel.id);
                } else {
                    self.wanted.remove(&channel.id);
                }
                self.channels.insert(
                    channel.id,
                    (channel.topic, message_type, channel.message_encoding),
                );
            }
            mcap::records::Record::Message { header, data } => {
                let Some((topic, message_type, encoding)) = self.channels.get(&header.channel_id)
                else {
                    return Err(format!("message on unknown channel {}", header.channel_id));
                };
                if self.wanted.contains(&header.channel_id)
                    && self.filter.wants_time(header.log_time)
                {
                    self.ready.push_back(StreamedMessage {
                        topic: topic.clone(),
                        message_type: message_type.clone(),
                        encoding: encoding.clone(),
                        log_time: header.log_time,
                        data: data.into_owned(),
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// The most a chunk's claimed uncompressed size reserves up front \[bytes\];
/// a larger real chunk just grows the buffer.
const MAX_CHUNK_RESERVE: u64 = 256 << 20;

/// The records of `chunk`, decompressed.
fn decompress(chunk: PendingChunk) -> std::result::Result<Vec<u8>, String> {
    let reserve = chunk.uncompressed_size.min(MAX_CHUNK_RESERVE) as usize;
    let mut out = Vec::with_capacity(reserve);
    match chunk.compression.as_str() {
        "" => return Ok(chunk.data),
        "zstd" => {
            let mut decoder = zstd::stream::read::Decoder::new(chunk.data.as_slice())
                .map_err(|e| format!("zstd chunk: {e}"))?;
            decoder
                .read_to_end(&mut out)
                .map_err(|e| format!("zstd chunk: {e}"))?;
        }
        "lz4" => {
            lz4_flex::frame::FrameDecoder::new(chunk.data.as_slice())
                .read_to_end(&mut out)
                .map_err(|e| format!("lz4 chunk: {e}"))?;
        }
        other => return Err(format!("unsupported chunk compression {other:?}")),
    }
    Ok(out)
}

/// Feed the records of a decompressed chunk, back to back, to `demux`.
fn absorb_chunk(demux: &mut Demux, records: &[u8]) -> std::result::Result<(), String> {
    let mut at = 0usize;
    while at < records.len() {
        let head = records
            .get(at..at + 9)
            .ok_or("truncated record header in a chunk")?;
        let opcode = head[0];
        let length = <[u8; 8]>::try_from(&head[1..9])
            .map(u64::from_le_bytes)
            .map_err(|e| e.to_string())?;
        let start = at + 9;
        let end = usize::try_from(length)
            .ok()
            .and_then(|length| start.checked_add(length))
            .filter(|&end| end <= records.len())
            .ok_or("a record overruns its chunk")?;
        let body = records
            .get(start..end)
            .ok_or("a record overruns its chunk")?;
        let record = mcap::parse_record(opcode, body).map_err(|e| e.to_string())?;
        demux.absorb(record)?;
        at = end;
    }
    Ok(())
}

/// Decompress the pending chunks on the thread pool and feed them to
/// `demux` in file order.
fn flush(pending: &mut Vec<PendingChunk>, demux: &mut Demux) -> std::result::Result<(), String> {
    if pending.is_empty() {
        return Ok(());
    }
    let chunks = std::mem::take(pending);
    let decoded: Vec<std::result::Result<Vec<u8>, String>> =
        chunks.into_par_iter().map(decompress).collect();
    for records in decoded {
        absorb_chunk(demux, &records?)?;
    }
    Ok(())
}

/// The messages of one MCAP file, read a record at a time. Its chunks are
/// read in batches and each batch is decompressed in parallel; messages
/// still come out in file order.
pub struct McapMessageStream {
    file: File,
    reader: LinearReader,
    demux: Demux,
    pending: Vec<PendingChunk>,
    /// Chunks read before they are decompressed together.
    batch: usize,
    /// The most chunks decompressed together so far.
    largest_batch: usize,
    finished: bool,
}

impl McapMessageStream {
    /// Stream every message of the MCAP file at `path`.
    ///
    /// # Errors
    ///
    /// [`ReaderError`] if the file cannot be opened.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_filtered(path, StreamFilter::default())
    }

    /// Stream the messages of the MCAP file at `path` that `filter` wants.
    ///
    /// # Errors
    ///
    /// [`ReaderError`] if the file cannot be opened.
    pub fn open_filtered(path: &Path, filter: StreamFilter) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            ReaderError::generic(format!("Failed to open MCAP file {}: {e}", path.display()))
        })?;
        let options = LinearReaderOptions::default().with_emit_chunks(true);
        Ok(Self {
            file,
            reader: LinearReader::new_with_options(options),
            demux: Demux::new(filter),
            pending: Vec::new(),
            batch: (rayon::current_num_threads() * 2).clamp(4, 64),
            largest_batch: 0,
            finished: false,
        })
    }

    /// The most chunks decompressed together so far: how parallel the
    /// reading has been. 1 means every chunk was decompressed alone.
    #[must_use]
    pub fn largest_batch(&self) -> usize {
        self.largest_batch
    }

    /// Read until a wanted message is queued or the file ends.
    fn advance(&mut self) -> std::result::Result<(), String> {
        loop {
            let event = match self.reader.next_event() {
                None => {
                    flush(&mut self.pending, &mut self.demux)?;
                    self.finished = true;
                    return Ok(());
                }
                Some(Err(error)) => return Err(error.to_string()),
                Some(Ok(event)) => event,
            };
            match event {
                LinearReadEvent::ReadRequest(need) => {
                    let read = self
                        .file
                        .read(self.reader.insert(need))
                        .map_err(|e| e.to_string())?;
                    self.reader.notify_read(read);
                }
                LinearReadEvent::Record { opcode, data } => {
                    match mcap::parse_record(opcode, data).map_err(|e| e.to_string())? {
                        mcap::records::Record::Chunk { header, data } => {
                            self.pending.push(PendingChunk {
                                compression: header.compression,
                                uncompressed_size: header.uncompressed_size,
                                data: data.into_owned(),
                            });
                            if self.pending.len() >= self.batch {
                                self.largest_batch = self.largest_batch.max(self.pending.len());
                                flush(&mut self.pending, &mut self.demux)?;
                            }
                        }
                        record @ (mcap::records::Record::Schema { .. }
                        | mcap::records::Record::Channel(_)
                        | mcap::records::Record::Message { .. }) => {
                            // A schema, a channel or a message between chunks
                            // comes after the chunks before it. The index
                            // records between chunks (one per channel, after
                            // every chunk) say nothing the stream uses, so
                            // they must not end a batch.
                            self.largest_batch = self.largest_batch.max(self.pending.len());
                            flush(&mut self.pending, &mut self.demux)?;
                            self.demux.absorb(record)?;
                        }
                        _ => {}
                    }
                }
            }
            if !self.demux.ready.is_empty() {
                return Ok(());
            }
        }
    }
}

impl Iterator for McapMessageStream {
    type Item = Result<StreamedMessage>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(message) = self.demux.ready.pop_front() {
                return Some(Ok(message));
            }
            if self.finished {
                return None;
            }
            if let Err(error) = self.advance() {
                self.finished = true;
                self.pending.clear();
                return Some(Err(ReaderError::generic(format!(
                    "Failed to read MCAP message: {error}"
                ))));
            }
        }
    }
}

/// The message type and message count of each topic in the MCAP file at
/// `path`, from its summary section; `None` when the file has no summary or
/// its summary carries no statistics (a recorder that was killed, say).
///
/// A topic counts only if it holds messages, as in
/// [`topics_by_scanning`]; its type is its schema's name, or its channel's
/// encoding without a schema.
fn topics_from_summary(path: &Path) -> Result<Option<HashMap<String, (String, u64)>>> {
    use mcap::sans_io::summary_reader::{SummaryReadEvent, SummaryReader, SummaryReaderOptions};
    use std::io::Seek;

    let fail = |error: &dyn std::fmt::Display| {
        ReaderError::generic(format!(
            "Failed to read the MCAP summary of {}: {error}",
            path.display()
        ))
    };
    let mut file = File::open(path).map_err(|e| fail(&e))?;
    let size = file.metadata().map_err(|e| fail(&e))?.len();
    let mut reader =
        SummaryReader::new_with_options(SummaryReaderOptions::default().with_file_size(size));
    while let Some(event) = reader.next_event() {
        match event.map_err(|e| fail(&e))? {
            SummaryReadEvent::ReadRequest(want) => {
                let read = file.read(reader.insert(want)).map_err(|e| fail(&e))?;
                reader.notify_read(read);
            }
            SummaryReadEvent::SeekRequest(to) => {
                let at = file.seek(to).map_err(|e| fail(&e))?;
                reader.notify_seeked(at);
            }
        }
    }
    let Some(summary) = reader.finish() else {
        return Ok(None);
    };
    let Some(stats) = summary.stats else {
        return Ok(None);
    };
    let mut topics: HashMap<String, (String, u64)> = HashMap::new();
    for (id, &count) in &stats.channel_message_counts {
        if count == 0 {
            continue;
        }
        let Some(channel) = summary.channels.get(id) else {
            return Ok(None);
        };
        let message_type = channel.schema.as_ref().map_or_else(
            || channel.message_encoding.clone(),
            |schema| schema.name.clone(),
        );
        topics
            .entry(channel.topic.clone())
            .or_insert((message_type, 0))
            .1 += count;
    }
    Ok(Some(topics))
}

/// The same table as [`topics_from_summary`], by streaming every message of
/// the file; unreadable messages are skipped with a warning.
fn topics_by_scanning(path: &Path) -> HashMap<String, (String, u64)> {
    let mut topics: HashMap<String, (String, u64)> = HashMap::new();
    let stream = match McapMessageStream::open(path) {
        Ok(stream) => stream,
        Err(e) => {
            tracing::warn!("Failed to read MCAP topics: {e}");
            return topics;
        }
    };
    for message in stream {
        match message {
            Ok(message) => {
                topics
                    .entry(message.topic)
                    .or_insert((message.message_type, 0))
                    .1 += 1;
            }
            Err(e) => tracing::warn!("Failed to read MCAP message: {e}"),
        }
    }
    topics
}

/// MCAP storage reader implementation
pub struct McapStorageReader {
    /// Paths to MCAP files
    mcap_paths: Vec<PathBuf>,
    /// Topic connections discovered from MCAP files
    topic_connections: Vec<Connection>,
    /// Whether the storage is currently open
    is_open: bool,
}

impl McapStorageReader {
    /// Create a new MCAP storage reader
    pub fn new(paths: Vec<&Path>, connections: Vec<Connection>) -> Result<Self> {
        let mcap_paths: Vec<PathBuf> = paths.iter().map(|p| p.to_path_buf()).collect();

        Ok(Self {
            mcap_paths,
            topic_connections: connections,
            is_open: false,
        })
    }

    /// Get all topics and their message counts directly from the MCAP
    /// files: from each file's summary section when it has statistics (a
    /// seek to the footer and a read of the index, however large the file),
    /// by one streaming pass over the file when it has none.
    pub fn get_topics_from_mcap(&self) -> Result<Vec<Connection>> {
        let mut topic_map: HashMap<String, (String, u64)> = HashMap::new();
        for path in &self.mcap_paths {
            let counted = match topics_from_summary(path) {
                Ok(Some(counted)) => counted,
                Ok(None) => topics_by_scanning(path),
                Err(e) => {
                    tracing::warn!(
                        "No usable MCAP summary in {} ({e}); scanning the file",
                        path.display()
                    );
                    topics_by_scanning(path)
                }
            };
            for (topic, (message_type, count)) in counted {
                topic_map.entry(topic).or_insert((message_type, 0)).1 += count;
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

    /// Every file's messages in turn, those on `connections`' topics in
    /// `[start, stop]` \[ns\] (all when `None`).
    fn stream(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<impl Iterator<Item = Result<StreamedMessage>> + use<>> {
        if !self.is_open {
            return Err(ReaderError::BagNotOpen);
        }
        let filter = StreamFilter {
            topics: connections.map(|cs| cs.iter().map(|c| c.topic.clone()).collect()),
            start,
            stop,
        };
        let streams = self
            .mcap_paths
            .iter()
            .map(|path| McapMessageStream::open_filtered(path, filter.clone()))
            .collect::<Result<Vec<_>>>()?;
        Ok(streams.into_iter().flatten())
    }

    /// The connection a message on `topic` belongs to.
    fn connection(&self, message: &StreamedMessage) -> Connection {
        self.topic_connections
            .iter()
            .find(|c| c.topic == message.topic)
            .cloned()
            .unwrap_or_else(|| Connection {
                id: 1,
                topic: message.topic.clone(),
                message_type: message.message_type.clone(),
                message_definition: MessageDefinition::default(),
                type_description_hash: String::new(),
                message_count: 0,
                serialization_format: message.encoding.clone(),
                offered_qos_profiles: Vec::new(),
            })
    }
}

impl StorageReader for McapStorageReader {
    fn open(&mut self) -> Result<()> {
        for path in &self.mcap_paths {
            if !path.is_file() {
                return Err(ReaderError::generic(format!(
                    "Failed to read MCAP file {}: not a file",
                    path.display()
                )));
            }
        }
        self.is_open = true;
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
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
        let stream = self.stream(connections, start, stop)?;
        Ok(Box::new(stream.map(move |message| {
            let message = message?;
            Ok(Message {
                connection: self.connection(&message),
                topic: message.topic,
                timestamp: message.log_time,
                data: message.data,
            })
        })))
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
        let stream = self.stream(connections, start, stop)?;
        Ok(Box::new(stream.map(move |message| {
            let message = message?;
            Ok(RawMessage {
                connection: self.connection(&message),
                timestamp: message.log_time,
                raw_data: message.data,
            })
        })))
    }

    fn read_raw_messages_batch(
        &self,
        connections: Option<&[Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Result<Vec<RawMessage>> {
        let mut all_messages = self
            .raw_messages_filtered(connections, start, stop)?
            .collect::<Result<Vec<_>>>()?;
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
    writer: Option<mcap::Writer<std::io::BufWriter<File>>>,
    /// Schema IDs, by message type.
    schemas: HashMap<String, u16>,
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
        let Some(writer) = self.writer.as_mut() else {
            return Err(BagError::BagNotOpen);
        };
        if !self.schemas.contains_key(&connection.message_type) {
            let id = writer
                .add_schema(
                    &connection.message_type,
                    "ros2msg",
                    connection.message_definition.data.as_bytes(),
                )
                .map_err(mcap_error)?;
            self.schemas.insert(connection.message_type.clone(), id);
        }
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
        // Schema 0 is MCAP's "no schema", for a type never added.
        let schema_id = self
            .schemas
            .get(&connection.message_type)
            .copied()
            .unwrap_or(0);
        let id = writer
            .add_channel(
                schema_id,
                &connection.topic,
                "cdr",
                &std::collections::BTreeMap::new(),
            )
            .map_err(mcap_error)?;
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

    /// A message as written: topic, log time \[ns\], payload.
    type WrittenMessage = (String, u64, Vec<u8>);

    /// One recording the readers below are compared on: `messages` messages
    /// round-robin on `/imu`, `/cloud` and `/cam`, in chunks of about
    /// `chunk_size` bytes. `(topic, log time, data)` in write order.
    fn write_recording(
        path: &Path,
        messages: u64,
        chunk_size: u64,
        compression: Option<mcap::Compression>,
        statistics: bool,
        chunked: bool,
    ) -> std::result::Result<Vec<WrittenMessage>, Box<dyn std::error::Error>> {
        let file = std::io::BufWriter::new(File::create(path)?);
        let mut writer = mcap::WriteOptions::new()
            .compression(compression)
            .chunk_size(Some(chunk_size))
            .emit_statistics(statistics)
            .use_chunks(chunked)
            .profile("ros2")
            .create(file)?;
        let schema = writer.add_schema("pkg/msg/Sample", "ros2msg", b"int32 value")?;
        let channels = [
            (
                "/imu",
                writer.add_channel(schema, "/imu", "cdr", &Default::default())?,
            ),
            (
                "/cloud",
                writer.add_channel(schema, "/cloud", "cdr", &Default::default())?,
            ),
            (
                "/cam",
                writer.add_channel(0, "/cam", "cdr", &Default::default())?,
            ),
        ];
        let mut written = Vec::new();
        for k in 0..messages {
            let (topic, channel_id) = channels[(k % 3) as usize];
            let size = 16 + (k % 7) as usize * 40;
            let data: Vec<u8> = (0..size)
                .map(|i| ((i as u64 * 7 + k) % 251) as u8)
                .collect();
            let log_time = 1_000 + 10 * k;
            let header = mcap::records::MessageHeader {
                channel_id,
                sequence: k as u32,
                log_time,
                publish_time: log_time,
            };
            writer.write_to_known_channel(&header, &data)?;
            written.push((topic.to_string(), log_time, data));
        }
        writer.finish()?;
        Ok(written)
    }

    fn sorted(topics: HashMap<String, (String, u64)>) -> Vec<(String, String, u64)> {
        let mut rows: Vec<_> = topics
            .into_iter()
            .map(|(topic, (message_type, count))| (topic, message_type, count))
            .collect();
        rows.sort();
        rows
    }

    #[test]
    fn the_summary_lists_the_topics_a_scan_finds() -> TestResult {
        let dir = tempdir()?;
        for compression in [None, Some(mcap::Compression::Zstd)] {
            let path = dir.path().join("recording.mcap");
            write_recording(&path, 90, 1_024, compression, true, true)?;
            let from_summary =
                topics_from_summary(&path)?.ok_or("a written file carries statistics")?;
            let scanned = topics_by_scanning(&path);
            assert_eq!(sorted(from_summary), sorted(scanned), "{compression:?}");
        }
        Ok(())
    }

    #[test]
    fn the_summary_types_a_schemaless_channel_by_its_encoding() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("recording.mcap");
        write_recording(&path, 9, 1_024, None, true, true)?;
        let topics = topics_from_summary(&path)?.ok_or("statistics expected")?;
        assert_eq!(topics.get("/cam"), Some(&("cdr".to_string(), 3)));
        assert_eq!(topics.get("/imu"), Some(&("pkg/msg/Sample".to_string(), 3)));
        Ok(())
    }

    #[test]
    fn a_file_without_statistics_is_counted_by_scanning() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("recording.mcap");
        write_recording(&path, 30, 1_024, None, false, true)?;
        assert!(topics_from_summary(&path)?.is_none());

        let reader = McapStorageReader::new(vec![path.as_path()], Vec::new())?;
        let connections = reader.get_topics_from_mcap()?;
        let counts: HashMap<_, _> = connections
            .iter()
            .map(|c| (c.topic.as_str(), c.message_count))
            .collect();
        assert_eq!(counts.get("/imu"), Some(&10));
        assert_eq!(counts.get("/cloud"), Some(&10));
        assert_eq!(counts.get("/cam"), Some(&10));
        Ok(())
    }

    #[test]
    fn topics_sum_over_several_files() -> TestResult {
        let dir = tempdir()?;
        let first = dir.path().join("a.mcap");
        let second = dir.path().join("b.mcap");
        write_recording(&first, 12, 1_024, None, true, true)?;
        write_recording(
            &second,
            30,
            1_024,
            Some(mcap::Compression::Zstd),
            true,
            true,
        )?;
        let reader = McapStorageReader::new(vec![first.as_path(), second.as_path()], Vec::new())?;
        let connections = reader.get_topics_from_mcap()?;
        let imu = connections
            .iter()
            .find(|c| c.topic == "/imu")
            .ok_or("imu topic")?;
        assert_eq!(imu.message_count, 4 + 10);
        Ok(())
    }

    #[test]
    fn a_missing_file_has_no_summary_and_no_topics() -> TestResult {
        let dir = tempdir()?;
        let missing = dir.path().join("none.mcap");
        assert!(topics_from_summary(&missing).is_err());
        assert!(topics_by_scanning(&missing).is_empty());
        Ok(())
    }

    fn stream_all(
        path: &Path,
        filter: StreamFilter,
    ) -> std::result::Result<Vec<WrittenMessage>, Box<dyn std::error::Error>> {
        McapMessageStream::open_filtered(path, filter)?
            .map(|m| m.map(|m| (m.topic, m.log_time, m.data)).map_err(Into::into))
            .collect()
    }

    #[test]
    fn a_stream_yields_every_message_in_file_order() -> TestResult {
        let dir = tempdir()?;
        for compression in [
            None,
            Some(mcap::Compression::Zstd),
            Some(mcap::Compression::Lz4),
        ] {
            let path = dir.path().join("recording.mcap");
            let written = write_recording(&path, 900, 1_024, compression, true, true)?;
            let read = stream_all(&path, StreamFilter::default())?;
            assert_eq!(read.len(), written.len(), "{compression:?}");
            assert_eq!(read, written, "{compression:?}");
        }
        Ok(())
    }

    #[test]
    fn a_stream_spans_more_chunks_than_one_batch() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("recording.mcap");
        let written =
            write_recording(&path, 3_000, 256, Some(mcap::Compression::Zstd), true, true)?;
        let stream = McapMessageStream::open(&path)?;
        assert!(stream.batch >= 4);
        let chunks = {
            let mut reader = LinearReader::new_with_options(
                LinearReaderOptions::default().with_emit_chunks(true),
            );
            let mut file = File::open(&path)?;
            let mut chunks = 0usize;
            while let Some(event) = reader.next_event() {
                match event? {
                    LinearReadEvent::ReadRequest(need) => {
                        let read = file.read(reader.insert(need))?;
                        reader.notify_read(read);
                    }
                    LinearReadEvent::Record { opcode, .. } => {
                        if opcode == mcap::records::op::CHUNK {
                            chunks += 1;
                        }
                    }
                }
            }
            chunks
        };
        assert!(
            chunks > 4 * stream.batch,
            "{chunks} chunks, batch {}",
            stream.batch
        );
        let read: Vec<WrittenMessage> = stream
            .map(|m| m.map(|m| (m.topic, m.log_time, m.data)))
            .collect::<Result<_>>()?;
        assert_eq!(read, written);
        Ok(())
    }

    #[test]
    fn a_filter_keeps_the_wanted_topics_and_times() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("recording.mcap");
        let written = write_recording(&path, 600, 512, Some(mcap::Compression::Zstd), true, true)?;
        let filter = StreamFilter {
            topics: Some(["/imu".to_string(), "/cam".to_string()].into()),
            start: Some(2_000),
            stop: Some(5_000),
        };
        let expected: Vec<WrittenMessage> = written
            .into_iter()
            .filter(|(topic, time, _)| topic != "/cloud" && (2_000..=5_000).contains(time))
            .collect();
        assert!(!expected.is_empty());
        assert_eq!(stream_all(&path, filter)?, expected);
        Ok(())
    }

    #[test]
    fn a_filter_that_wants_no_topic_yields_nothing() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("recording.mcap");
        write_recording(&path, 60, 512, None, true, true)?;
        let filter = StreamFilter {
            topics: Some(HashSet::from(["/absent".to_string()])),
            ..StreamFilter::default()
        };
        assert!(stream_all(&path, filter)?.is_empty());
        Ok(())
    }

    #[test]
    fn an_unchunked_file_streams_its_top_level_messages() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("recording.mcap");
        let written = write_recording(&path, 90, 1_024, None, true, false)?;
        assert_eq!(stream_all(&path, StreamFilter::default())?, written);
        Ok(())
    }

    #[test]
    fn a_truncated_file_ends_the_stream_with_an_error() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("recording.mcap");
        write_recording(&path, 600, 512, Some(mcap::Compression::Zstd), true, true)?;
        let bytes = std::fs::read(&path)?;
        std::fs::write(&path, &bytes[..bytes.len() / 2])?;

        let mut errors = 0usize;
        let mut messages = 0usize;
        for item in McapMessageStream::open(&path)? {
            match item {
                Ok(_) => messages += 1,
                Err(_) => errors += 1,
            }
        }
        assert!(messages > 0, "the intact half is still read");
        assert_eq!(errors, 1, "one error, then the stream is finished");
        Ok(())
    }

    #[test]
    fn a_stream_of_a_missing_file_is_an_open_error() {
        assert!(McapMessageStream::open(Path::new("/nonexistent/none.mcap")).is_err());
    }

    #[test]
    fn a_chunk_with_an_unknown_compression_is_an_error() {
        let chunk = PendingChunk {
            compression: "brotli".to_string(),
            uncompressed_size: 0,
            data: Vec::new(),
        };
        assert!(decompress(chunk).is_err());
    }

    #[test]
    fn a_chunk_whose_record_overruns_it_is_an_error() {
        let mut demux = Demux::new(StreamFilter::default());
        let mut records = vec![0x05u8];
        records.extend_from_slice(&1_000u64.to_le_bytes());
        records.extend_from_slice(&[0, 1, 2]);
        assert!(absorb_chunk(&mut demux, &records).is_err());
        assert!(absorb_chunk(&mut demux, &[0x05, 1, 2]).is_err());
    }

    #[test]
    fn a_message_on_an_unknown_channel_is_an_error() {
        let mut demux = Demux::new(StreamFilter::default());
        let header = mcap::records::MessageHeader {
            channel_id: 9,
            sequence: 0,
            log_time: 1,
            publish_time: 1,
        };
        let record = mcap::records::Record::Message {
            header,
            data: std::borrow::Cow::Borrowed(&[1, 2, 3]),
        };
        assert!(demux.absorb(record).is_err());
    }

    #[test]
    fn chunks_are_decompressed_in_batches_despite_the_index_records_between_them() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("recording.mcap");
        write_recording(&path, 3_000, 256, Some(mcap::Compression::Zstd), true, true)?;
        let mut stream = McapMessageStream::open(&path)?;
        let batch = stream.batch;
        let mut count = 0usize;
        for message in stream.by_ref() {
            message?;
            count += 1;
        }
        assert_eq!(count, 3_000);
        assert_eq!(stream.largest_batch(), batch);
        assert!(batch > 1);
        Ok(())
    }
}
