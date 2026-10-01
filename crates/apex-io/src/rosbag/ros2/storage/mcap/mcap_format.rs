//! Low-level seek-based MCAP container parsing.
//!
//! A minimal, allocation-bounded reader for the [MCAP container
//! format](https://mcap.dev/spec): it walks the summary section by seeking
//! (footer → summary), then streams messages chunk-by-chunk, decompressing
//! and holding **one chunk at a time** rather than the whole file. This is
//! what lets the rosbag layer open multi-gigabyte dataset bags without
//! reading them into RAM first.
//!
//! It is deliberately profile-agnostic: the container layer knows nothing
//! about `cdr` vs `ros1` payload encodings. Callers dispatch on the channel's
//! `message_encoding` (surfaced by the storage layer as the connection's
//! `serialization_format`).
//!
//! Scope notes, relative to the full spec:
//!
//! - CRCs are validated for chunk `records` payloads when a chunk declares
//!   one; summary/footer CRCs are not re-checked.
//! - Attachment/Metadata records are not surfaced.
//! - Files without a summary are read by a linear data-section walk
//!   (unindexed fallback), since there is no chunk index to drive seeks.
//! - Message Index records are not consulted: chunk-level
//!   `message_start_time`/`message_end_time` already give time-skip, and the
//!   chunk payload is decompressed whole anyway.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::rosbag::error::ReaderError;

/// MCAP file magic (`0x89 "MCAP0" CR LF`).
const MAGIC: &[u8] = b"\x89MCAP0\r\n";

/// Footer record: opcode (1) + length (8) + 20 content bytes, then magic (8).
const FOOTER_SPAN: u64 = 9 + 20 + MAGIC.len() as u64;

/// Opcodes used by this reader.
const OP_HEADER: u8 = 0x01;
const OP_SCHEMA: u8 = 0x03;
const OP_CHANNEL: u8 = 0x04;
const OP_MESSAGE: u8 = 0x05;
const OP_CHUNK: u8 = 0x06;
const OP_STATISTICS: u8 = 0x0B;
const OP_CHUNK_INDEX: u8 = 0x08;
const OP_DATA_END: u8 = 0x0F;

/// Fixed schema record: name/encoding plus the definition payload.
#[derive(Debug, Clone)]
pub(crate) struct SchemaInfo {
    pub name: String,
    pub encoding: String,
    pub data: Vec<u8>,
}

/// Channel record: topic plus the schema and payload encoding it references.
#[derive(Debug, Clone)]
pub(crate) struct ChannelInfo {
    pub schema_id: u16,
    pub topic: String,
    pub message_encoding: String,
}

/// Aggregates from a file's Statistics record.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct McapStats {
    pub message_count: u64,
    pub start_time: u64,
    pub end_time: u64,
    /// Channel ID -> message count, as declared by the writer.
    pub channel_counts: BTreeMap<u16, u64>,
}

/// Chunk Index record: where a chunk lives and what time range it covers.
#[derive(Debug, Clone)]
pub(crate) struct ChunkIndexLite {
    pub message_start_time: u64,
    pub message_end_time: u64,
    pub chunk_start_offset: u64,
    pub compression: String,
    pub compressed_size: u64,
    pub uncompressed_size: u64,
}

/// Everything the reader needs from a file's summary section.
#[derive(Debug, Clone, Default)]
pub(crate) struct McapSummary {
    pub schemas: BTreeMap<u16, SchemaInfo>,
    pub channels: BTreeMap<u16, ChannelInfo>,
    pub stats: Option<McapStats>,
    pub chunk_indexes: Vec<ChunkIndexLite>,
}

impl McapSummary {
    /// Schema name for a channel, falling back to its message encoding --
    /// the (schema, encoding) pair the storage layer maps onto
    /// `message_type` / `serialization_format`.
    pub(crate) fn message_type_for(&self, channel: &ChannelInfo) -> String {
        self.schemas
            .get(&channel.schema_id)
            .map(|s| s.name.clone())
            .unwrap_or_else(|| channel.message_encoding.clone())
    }
}

/// A message yielded by [`ChunkMessageStream`], with its payload copied out
/// of the transient chunk buffer.
#[derive(Debug, Clone)]
pub(crate) struct RawMcapMessage {
    pub channel_id: u16,
    /// Part of the container record; `Message`/`RawMessage` don't carry
    /// these two, but they're kept on the parsed form for format fidelity.
    #[allow(dead_code)]
    pub sequence: u32,
    pub log_time: u64,
    #[allow(dead_code)]
    pub publish_time: u64,
    pub data: Vec<u8>,
}

/// Little-endian u64 from exactly 8 bytes.
fn le_u64(bytes: &[u8]) -> Result<u64, ReaderError> {
    let array: [u8; 8] = bytes
        .try_into()
        .map_err(|_| ReaderError::generic("MCAP record truncated: u64 needs 8 bytes"))?;
    Ok(u64::from_le_bytes(array))
}

/// IEEE CRC-32 (the polynomial MCAP and zlib share), SIMD-accelerated.
pub(crate) fn crc32(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

/// Cursor over an in-memory record stream: little-endian fixed-width reads
/// plus MCAP length-prefixed strings/maps/bytes.
struct Buf<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Buf<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], ReaderError> {
        if self.remaining() < n {
            return Err(ReaderError::generic(format!(
                "MCAP record truncated: need {n} bytes at offset {}, have {}",
                self.pos,
                self.remaining()
            )));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u16(&mut self) -> Result<u16, ReaderError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, ReaderError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, ReaderError> {
        le_u64(self.take(8)?)
    }

    fn read_str(&mut self) -> Result<String, ReaderError> {
        let len = self.read_u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| ReaderError::generic(format!("MCAP string is not UTF-8: {e}")))
    }

    /// Skip a `Map<K, V>` by its byte length (readers may ignore entries).
    fn skip_map(&mut self) -> Result<(), ReaderError> {
        let len = self.read_u32()? as usize;
        self.take(len)?;
        Ok(())
    }
}

/// Parse one summary-section record into the summary accumulator.
fn parse_summary_record(
    op: u8,
    content: &[u8],
    summary: &mut McapSummary,
) -> Result<(), ReaderError> {
    match op {
        OP_SCHEMA => {
            let mut buf = Buf::new(content);
            let id = buf.read_u16()?;
            let name = buf.read_str()?;
            let encoding = buf.read_str()?;
            let len = buf.read_u32()? as usize;
            let data = buf.take(len)?.to_vec();
            summary.schemas.insert(
                id,
                SchemaInfo {
                    name,
                    encoding,
                    data,
                },
            );
        }
        OP_CHANNEL => {
            let mut buf = Buf::new(content);
            let id = buf.read_u16()?;
            let schema_id = buf.read_u16()?;
            let topic = buf.read_str()?;
            let message_encoding = buf.read_str()?;
            buf.skip_map()?;
            summary.channels.insert(
                id,
                ChannelInfo {
                    schema_id,
                    topic,
                    message_encoding,
                },
            );
        }
        OP_STATISTICS => {
            let mut buf = Buf::new(content);
            let message_count = buf.read_u64()?;
            let _schema_count = buf.read_u16()?;
            let _channel_count = buf.read_u32()?;
            let _attachment_count = buf.read_u32()?;
            let _metadata_count = buf.read_u32()?;
            let _chunk_count = buf.read_u32()?;
            let start_time = buf.read_u64()?;
            let end_time = buf.read_u64()?;
            let mut channel_counts = BTreeMap::new();
            if buf.remaining() >= 4 {
                let byte_len = buf.read_u32()? as usize;
                let end = buf.pos + byte_len;
                while buf.pos + 10 <= end.min(content.len()) {
                    let id = buf.read_u16()?;
                    let count = buf.read_u64()?;
                    channel_counts.insert(id, count);
                }
            }
            summary.stats = Some(McapStats {
                message_count,
                start_time,
                end_time,
                channel_counts,
            });
        }
        OP_CHUNK_INDEX => {
            let mut buf = Buf::new(content);
            let message_start_time = buf.read_u64()?;
            let message_end_time = buf.read_u64()?;
            let chunk_start_offset = buf.read_u64()?;
            let _chunk_length = buf.read_u64()?;
            buf.skip_map()?; // message_index_offsets
            let _message_index_length = buf.read_u64()?;
            let compression = buf.read_str()?;
            let compressed_size = buf.read_u64()?;
            let uncompressed_size = buf.read_u64()?;
            summary.chunk_indexes.push(ChunkIndexLite {
                message_start_time,
                message_end_time,
                chunk_start_offset,
                compression,
                compressed_size,
                uncompressed_size,
            });
        }
        // Attachment indexes, metadata indexes, summary offsets, and anything
        // newer: irrelevant to message reading.
        _ => {}
    }
    Ok(())
}

/// Read the summary section of the MCAP file behind `file` by seeking.
///
/// `Ok(None)` means the file declares no summary (`summary_start == 0`).
/// Never reads message data: the region parsed is bounded by the footer's
/// own offsets, and its size is validated against the file length before
/// being read into memory.
pub(crate) fn read_summary(file: &mut File) -> Result<Option<McapSummary>, ReaderError> {
    let file_len = file
        .metadata()
        .map_err(|e| ReaderError::generic(format!("Failed to stat MCAP file: {e}")))?
        .len();

    if file_len < MAGIC.len() as u64 {
        return Err(ReaderError::generic(format!(
            "File too small ({file_len} bytes) to be an MCAP container"
        )));
    }

    // Leading magic.
    file.seek(SeekFrom::Start(0))
        .map_err(|e| ReaderError::generic(format!("Failed to seek MCAP file: {e}")))?;
    let mut magic = [0u8; MAGIC.len()];
    file.read_exact(&mut magic)
        .map_err(|e| ReaderError::generic(format!("Failed to read MCAP magic: {e}")))?;
    if magic != MAGIC {
        return Err(ReaderError::generic(
            "Leading bytes are not the MCAP magic; not an MCAP file",
        ));
    }

    // Structure sanity for anything claiming to have a footer: magic +
    // header record + footer + magic. Smaller files have no summary to read.
    if file_len < 8 + 9 + FOOTER_SPAN {
        return Ok(None);
    }

    // Trailing magic.
    file.seek(SeekFrom::Start(file_len - MAGIC.len() as u64))
        .map_err(|e| ReaderError::generic(format!("Failed to seek MCAP file: {e}")))?;
    let mut magic = [0u8; MAGIC.len()];
    file.read_exact(&mut magic)
        .map_err(|e| ReaderError::generic(format!("Failed to read MCAP magic: {e}")))?;
    if magic != MAGIC {
        return Err(ReaderError::generic(
            "Trailing bytes are not the MCAP magic; file may be truncated",
        ));
    }

    // Footer record: 9-byte prefix + 20 content bytes, ending right before
    // the trailing magic.
    file.seek(SeekFrom::Start(file_len - FOOTER_SPAN))
        .map_err(|e| ReaderError::generic(format!("Failed to seek MCAP file: {e}")))?;
    let mut footer = [0u8; 9 + 20];
    file.read_exact(&mut footer)
        .map_err(|e| ReaderError::generic(format!("Failed to read MCAP footer: {e}")))?;
    if footer[0] != 0x02 {
        return Err(ReaderError::generic(format!(
            "Expected Footer record before trailing magic, found opcode 0x{:02x}",
            footer[0]
        )));
    }
    let content_len = le_u64(&footer[1..9])?;
    if content_len != 20 {
        return Err(ReaderError::generic(format!(
            "Footer content length is {content_len}, expected 20"
        )));
    }
    let summary_start = le_u64(&footer[9..17])?;
    let summary_offset_start = le_u64(&footer[17..25])?;

    if summary_start == 0 {
        return Ok(None);
    }

    // The summary section ends at the summary offset section (if any) or at
    // the footer record itself.
    let summary_end = if summary_offset_start != 0 {
        summary_offset_start
    } else {
        file_len - FOOTER_SPAN
    };
    if summary_start > summary_end || summary_end > file_len - FOOTER_SPAN {
        return Err(ReaderError::generic(format!(
            "Footer summary offsets are inconsistent: start {summary_start}, end {summary_end}, file {file_len}"
        )));
    }

    let summary_len = (summary_end - summary_start) as usize;
    file.seek(SeekFrom::Start(summary_start))
        .map_err(|e| ReaderError::generic(format!("Failed to seek MCAP summary: {e}")))?;
    let mut section = vec![0u8; summary_len];
    file.read_exact(&mut section)
        .map_err(|e| ReaderError::generic(format!("Failed to read MCAP summary: {e}")))?;

    let mut summary = McapSummary::default();
    let mut cursor = 0usize;
    while cursor + 9 <= section.len() {
        let op = section[cursor];
        let len = le_u64(&section[cursor + 1..cursor + 9])? as usize;
        cursor += 9;
        let end = cursor + len as usize;
        if end > section.len() {
            return Err(ReaderError::generic("Summary section truncated mid-record"));
        }
        parse_summary_record(op, &section[cursor..end], &mut summary)?;
        cursor = end;
    }

    Ok(Some(summary))
}

/// Locate the data section start (just past the Header record) for the
/// unindexed linear walk.
fn data_section_start(file: &mut File) -> Result<u64, ReaderError> {
    file.seek(SeekFrom::Start(MAGIC.len() as u64))
        .map_err(|e| ReaderError::generic(format!("Failed to seek MCAP file: {e}")))?;
    let mut prefix = [0u8; 9];
    file.read_exact(&mut prefix)
        .map_err(|e| ReaderError::generic(format!("Failed to read MCAP header: {e}")))?;
    if prefix[0] != OP_HEADER {
        return Err(ReaderError::generic(format!(
            "Expected Header record after magic, found opcode 0x{:02x}",
            prefix[0]
        )));
    }
    let len = le_u64(&prefix[1..9])?;
    Ok(MAGIC.len() as u64 + 9 + len)
}

/// Decompress a chunk's `records` payload according to its declared
/// compression, validating the declared uncompressed size (and CRC, when
/// the writer provided one).
fn decompress_chunk_records(
    compression: &str,
    compressed: &[u8],
    uncompressed_size: u64,
    uncompressed_crc: u32,
) -> Result<Vec<u8>, ReaderError> {
    let expected = uncompressed_size as usize;
    let records = match compression {
        "" => compressed.to_vec(),
        "zstd" => zstd::bulk::decompress(compressed, expected).map_err(|e| {
            ReaderError::generic(format!("Failed to decompress zstd MCAP chunk: {e}"))
        })?,
        "lz4" => {
            let mut reader = lz4_flex::frame::FrameDecoder::new(std::io::Cursor::new(compressed));
            let mut out = Vec::with_capacity(expected);
            reader.read_to_end(&mut out).map_err(|e| {
                ReaderError::generic(format!("Failed to decompress lz4 MCAP chunk: {e}"))
            })?;
            out
        }
        other => {
            return Err(ReaderError::generic(format!(
                "Unsupported MCAP chunk compression: {other:?}"
            )));
        }
    };
    if records.len() != expected {
        return Err(ReaderError::generic(format!(
            "MCAP chunk decompressed to {} bytes, header declared {expected}",
            records.len()
        )));
    }
    if uncompressed_crc != 0 && crc32(&records) != uncompressed_crc {
        return Err(ReaderError::generic(
            "MCAP chunk CRC mismatch (payload corrupt)",
        ));
    }
    Ok(records)
}

/// Read one chunk record from `file` and return its decompressed, CRC-checked
/// message records. Self-contained so it can run on any thread with its own
/// file handle.
fn read_chunk_records(
    file: &mut File,
    file_len: u64,
    chunk: &ChunkIndexLite,
) -> Result<Vec<u8>, ReaderError> {
    let record_len = 8 + 8 + 8 + 4 + 4 + chunk.compression.len() as u64 + 8 + chunk.compressed_size;
    let end = chunk
        .chunk_start_offset
        .checked_add(9 + record_len)
        .ok_or_else(|| ReaderError::generic("MCAP chunk offset overflow"))?;
    if end > file_len {
        return Err(ReaderError::generic(format!(
            "MCAP chunk at {} extends past end of file",
            chunk.chunk_start_offset
        )));
    }

    file.seek(SeekFrom::Start(chunk.chunk_start_offset))
        .map_err(|e| ReaderError::generic(format!("Failed to seek MCAP chunk: {e}")))?;
    let mut prefix = [0u8; 9];
    file.read_exact(&mut prefix)
        .map_err(|e| ReaderError::generic(format!("Failed to read MCAP chunk: {e}")))?;
    if prefix[0] != OP_CHUNK {
        return Err(ReaderError::generic(format!(
            "Chunk index points to opcode 0x{:02x}, expected a Chunk record",
            prefix[0]
        )));
    }
    let mut content = vec![0u8; record_len as usize];
    file.read_exact(&mut content)
        .map_err(|e| ReaderError::generic(format!("Failed to read MCAP chunk: {e}")))?;

    let mut buf = Buf::new(&content);
    let _message_start_time = buf.read_u64()?;
    let _message_end_time = buf.read_u64()?;
    let uncompressed_size = buf.read_u64()?;
    let uncompressed_crc = buf.read_u32()?;
    let compression = buf.read_str()?;
    let records_len = buf.read_u64()? as usize;
    let compressed = buf.take(records_len)?;

    if compression != chunk.compression {
        return Err(ReaderError::generic(format!(
            "Chunk declares compression {compression:?}, its index said {:?}",
            chunk.compression
        )));
    }
    if uncompressed_size != chunk.uncompressed_size {
        return Err(ReaderError::generic(format!(
            "Chunk decompressed size {uncompressed_size} disagrees with its index ({})",
            chunk.uncompressed_size
        )));
    }

    decompress_chunk_records(
        &compression,
        compressed,
        uncompressed_size,
        uncompressed_crc,
    )
}

/// Fewer chunks than this are read inline: thread start-up would not pay off.
const PREFETCH_MIN_CHUNKS: usize = 4;
/// Decompressed chunks each worker may hold ready ahead of the consumer.
const PREFETCH_DEPTH: usize = 2;

/// Order-preserving parallel chunk decompression.
///
/// Worker `w` of `W` handles chunks `w, w + W, w + 2W, ...` on its own file
/// handle and sends them through a bounded channel; the consumer takes chunk
/// `i` from worker `i % W`. Order is therefore exact without any reordering
/// buffer, and memory is bounded by `W * (PREFETCH_DEPTH + 1)` chunks.
struct Prefetch {
    receivers: Vec<std::sync::mpsc::Receiver<Result<Vec<u8>, ReaderError>>>,
    handles: Vec<std::thread::JoinHandle<()>>,
    next: usize,
    total: usize,
}

impl Prefetch {
    /// Start the workers, or `None` if no thread could be started (the
    /// caller then reads inline).
    fn spawn(path: &Path, file_len: u64, chunks: Vec<ChunkIndexLite>) -> Option<Self> {
        let total = chunks.len();
        let cores = std::thread::available_parallelism().map_or(2, |n| n.get());
        let workers = cores.saturating_sub(1).clamp(1, 6).min(total);

        let mut receivers = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for w in 0..workers {
            let (tx, rx) = std::sync::mpsc::sync_channel(PREFETCH_DEPTH);
            let mine: Vec<ChunkIndexLite> =
                chunks.iter().skip(w).step_by(workers).cloned().collect();
            let path = path.to_path_buf();
            let spawned = std::thread::Builder::new()
                .name(format!("mcap-chunk-{w}"))
                .spawn(move || {
                    let mut file = match File::open(&path) {
                        Ok(f) => f,
                        Err(e) => {
                            let _ = tx.send(Err(ReaderError::generic(format!(
                                "Failed to open MCAP file {}: {e}",
                                path.display()
                            ))));
                            return;
                        }
                    };
                    for chunk in &mine {
                        let result = read_chunk_records(&mut file, file_len, chunk);
                        let failed = result.is_err();
                        // A closed channel means the reader was dropped.
                        if tx.send(result).is_err() || failed {
                            return;
                        }
                    }
                });
            match spawned {
                Ok(handle) => {
                    receivers.push(rx);
                    handles.push(handle);
                }
                Err(_) => {
                    // Cannot start this worker: abandon the pool.
                    drop(rx);
                    drop(receivers);
                    for h in handles {
                        let _ = h.join();
                    }
                    return None;
                }
            }
        }
        Some(Self {
            receivers,
            handles,
            next: 0,
            total,
        })
    }

    fn next_chunk(&mut self) -> Result<Option<Vec<u8>>, ReaderError> {
        if self.next >= self.total {
            return Ok(None);
        }
        let worker = self.next % self.receivers.len();
        self.next += 1;
        match self.receivers[worker].recv() {
            Ok(result) => result.map(Some),
            Err(_) => Err(ReaderError::generic(
                "MCAP chunk worker stopped unexpectedly",
            )),
        }
    }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        // Closing the channels unblocks workers waiting to send.
        self.receivers.clear();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Streaming message reader over a single MCAP file.
///
/// Indexed files (with chunk indexes) are read chunk-by-chunk: each chunk is
/// seeked to, decompressed whole, and drained message-by-message, so peak
/// memory is one decompressed chunk -- never the file. Chunks outside a
/// requested time range are skipped without being read at all. Unindexed
/// files fall back to a linear data-section walk.
pub(crate) struct ChunkMessageStream {
    path: PathBuf,
    file: File,
    file_len: u64,
    summary: Option<McapSummary>,
    /// Channel topics resolved per channel ID (owned, so they can outlive a
    /// chunk buffer borrow). Serves both filtering and connection mapping.
    topics: BTreeMap<u16, Rc<str>>,
    /// Message encodings per channel ID, same lifecycle as `topics`.
    encodings: BTreeMap<u16, Rc<str>>,
    /// Message types per channel ID (schema name, or encoding when the
    /// channel has no schema).
    types: BTreeMap<u16, Rc<str>>,
    /// Schemas seen only in chunk/data-section records (unindexed files
    /// carry no summary, so their schemas arrive inline).
    local_schemas: BTreeMap<u16, SchemaInfo>,

    /// Remaining indexed chunks, in file (log-time) order.
    chunk_queue: std::collections::VecDeque<ChunkIndexLite>,
    /// Decompressed records of the chunk currently being drained.
    inner: Vec<u8>,
    /// Parse cursor into `inner`.
    cursor: usize,

    /// Unindexed-fallback state: next offset to read from the data section,
    /// `Some` only when walking a file with no chunk indexes.
    section_cursor: Option<u64>,

    /// Parallel chunk decompression, started on the first chunk load.
    prefetch: Option<Prefetch>,
    prefetch_tried: bool,
}

impl ChunkMessageStream {
    /// Open the file and prepare the stream from its summary.
    pub(crate) fn open(path: &Path, summary: Option<McapSummary>) -> Result<Self, ReaderError> {
        let file = File::open(path).map_err(|e| {
            ReaderError::generic(format!("Failed to open MCAP file {}: {e}", path.display()))
        })?;
        let file_len = file
            .metadata()
            .map_err(|e| ReaderError::generic(format!("Failed to stat MCAP file: {e}")))?
            .len();

        let mut stream = Self {
            path: path.to_path_buf(),
            file,
            file_len,
            summary,
            topics: BTreeMap::new(),
            encodings: BTreeMap::new(),
            types: BTreeMap::new(),
            local_schemas: BTreeMap::new(),
            chunk_queue: std::collections::VecDeque::new(),
            inner: Vec::new(),
            cursor: 0,
            section_cursor: None,
            prefetch: None,
            prefetch_tried: false,
        };

        if let Some(summary) = &stream.summary {
            stream
                .chunk_queue
                .extend(summary.chunk_indexes.iter().cloned());
            // Seed the per-channel caches from the summary so lookups do not
            // depend on chunk-embedded Channel records.
            let channels = summary.channels.clone();
            for (id, channel) in &channels {
                let message_type = summary.message_type_for(channel);
                stream
                    .topics
                    .entry(*id)
                    .or_insert_with(|| Rc::from(channel.topic.as_str()));
                stream
                    .encodings
                    .entry(*id)
                    .or_insert_with(|| Rc::from(channel.message_encoding.as_str()));
                stream
                    .types
                    .entry(*id)
                    .or_insert_with(|| Rc::from(message_type.as_str()));
            }
        }

        if stream.chunk_queue.is_empty() {
            // No chunk indexes: fall back to the linear data-section walk.
            stream.section_cursor = Some(data_section_start(&mut stream.file)?);
        }

        Ok(stream)
    }

    /// Topic for a channel ID, from the summary or from chunk-embedded
    /// Channel records seen so far.
    pub(crate) fn topic(&self, channel_id: u16) -> Option<Rc<str>> {
        self.topics.get(&channel_id).cloned()
    }

    /// Register schema/channel records encountered while draining a chunk
    /// (files may embed them rather than relying on the summary).
    fn register_record(&mut self, op: u8, content: &[u8]) -> Result<(), ReaderError> {
        match op {
            OP_CHANNEL => {
                let mut buf = Buf::new(content);
                let id = buf.read_u16()?;
                let schema_id = buf.read_u16()?;
                let topic = buf.read_str()?;
                let message_encoding = buf.read_str()?;
                let message_type = self
                    .local_schemas
                    .get(&schema_id)
                    .map(|s| s.name.clone())
                    .or_else(|| {
                        self.summary
                            .as_ref()
                            .and_then(|s| s.schemas.get(&schema_id))
                            .map(|s| s.name.clone())
                    })
                    .unwrap_or_else(|| message_encoding.clone());
                self.topics
                    .entry(id)
                    .or_insert_with(|| Rc::from(topic.as_str()));
                self.encodings
                    .entry(id)
                    .or_insert_with(|| Rc::from(message_encoding.as_str()));
                self.types
                    .entry(id)
                    .or_insert_with(|| Rc::from(message_type.as_str()));
            }
            OP_SCHEMA => {
                let mut buf = Buf::new(content);
                let id = buf.read_u16()?;
                let name = buf.read_str()?;
                let encoding = buf.read_str()?;
                let len = buf.read_u32()? as usize;
                let data = buf.take(len)?.to_vec();
                self.local_schemas.entry(id).or_insert(SchemaInfo {
                    name,
                    encoding,
                    data,
                });
            }
            _ => {}
        }
        Ok(())
    }

    /// Pull the next eligible chunk's records into `inner`, skipping chunks
    /// outside the time filter entirely.
    ///
    /// With enough chunks to read, decompression runs on a small worker pool
    /// (see [`Prefetch`]); otherwise it happens inline.
    fn load_next_chunk(&mut self, filters: &Filters) -> Result<bool, ReaderError> {
        if self.prefetch.is_none() && !self.prefetch_tried {
            self.prefetch_tried = true;
            let eligible: Vec<ChunkIndexLite> = self
                .chunk_queue
                .iter()
                .filter(|c| {
                    !(filters.start.is_some_and(|t| c.message_end_time < t)
                        || filters.stop.is_some_and(|t| c.message_start_time >= t))
                })
                .cloned()
                .collect();
            if eligible.len() >= PREFETCH_MIN_CHUNKS {
                self.prefetch = Prefetch::spawn(&self.path, self.file_len, eligible);
                if self.prefetch.is_some() {
                    self.chunk_queue.clear();
                }
            }
        }

        if let Some(prefetch) = &mut self.prefetch {
            return match prefetch.next_chunk()? {
                Some(records) => {
                    self.inner = records;
                    self.cursor = 0;
                    Ok(true)
                }
                None => Ok(false),
            };
        }

        while let Some(chunk) = self.chunk_queue.pop_front() {
            // Chunk-level time skip: nothing inside can pass the filter.
            if filters.start.is_some_and(|t| chunk.message_end_time < t)
                || filters.stop.is_some_and(|t| chunk.message_start_time >= t)
            {
                continue;
            }
            self.inner = read_chunk_records(&mut self.file, self.file_len, &chunk)?;
            self.cursor = 0;
            return Ok(true);
        }
        Ok(false)
    }

    /// Parse the next message record out of `inner`, registering any
    /// schema/channel records that precede it.
    fn next_inner_message(
        &mut self,
        filters: &Filters,
    ) -> Result<Option<RawMcapMessage>, ReaderError> {
        loop {
            if self.cursor + 9 > self.inner.len() {
                self.inner = Vec::new();
                self.cursor = 0;
                return Ok(None);
            }
            let op = self.inner[self.cursor];
            let len = le_u64(&self.inner[self.cursor + 1..self.cursor + 9])? as usize;
            let end = self.cursor + 9 + len;
            if end > self.inner.len() {
                return Err(ReaderError::generic(
                    "MCAP chunk records truncated mid-record",
                ));
            }
            // Own the record content before touching `self` again: the slice
            // borrows `self.inner`, and the arms below register records (a
            // `&mut self` call) or copy data out.
            let content = self.inner[self.cursor + 9..end].to_vec();
            self.cursor = end;

            match op {
                OP_MESSAGE => {
                    let mut buf = Buf::new(&content);
                    let channel_id = buf.read_u16()?;
                    let sequence = buf.read_u32()?;
                    let log_time = buf.read_u64()?;
                    let publish_time = buf.read_u64()?;
                    let data = content[22..].to_vec();

                    if filters.start.is_some_and(|t| log_time < t)
                        || filters.stop.is_some_and(|t| log_time >= t)
                    {
                        continue;
                    }
                    if let Some(topics) = &filters.topics {
                        let topic = self.topic(channel_id).ok_or_else(|| {
                            ReaderError::generic(format!(
                                "MCAP message references unknown channel id {channel_id}"
                            ))
                        })?;
                        if !topics.contains(topic.as_ref()) {
                            continue;
                        }
                    }
                    return Ok(Some(RawMcapMessage {
                        channel_id,
                        sequence,
                        log_time,
                        publish_time,
                        data,
                    }));
                }
                OP_SCHEMA | OP_CHANNEL => self.register_record(op, &content)?,
                // Chunks nest records of these kinds per spec; anything else
                // is skipped.
                _ => {}
            }
        }
    }

    /// Parse the next record out of the data section for the unindexed walk.
    fn next_section_message(
        &mut self,
        filters: &Filters,
    ) -> Result<Option<RawMcapMessage>, ReaderError> {
        loop {
            let Some(mut offset) = self.section_cursor else {
                return Ok(None);
            };
            if offset + 9 > self.file_len {
                return Ok(None);
            }
            self.file
                .seek(SeekFrom::Start(offset))
                .map_err(|e| ReaderError::generic(format!("Failed to seek MCAP record: {e}")))?;
            let mut prefix = [0u8; 9];
            match self.file.read_exact(&mut prefix) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    self.section_cursor = None;
                    return Ok(None);
                }
                Err(e) => {
                    return Err(ReaderError::generic(format!(
                        "Failed to read MCAP record: {e}"
                    )));
                }
            }
            let op = prefix[0];
            let len = le_u64(&prefix[1..9])?;
            offset += 9 + len;
            self.section_cursor = Some(offset);

            match op {
                OP_MESSAGE => {
                    let mut content = vec![0u8; len as usize];
                    self.file.seek(SeekFrom::Start(offset - len)).map_err(|e| {
                        ReaderError::generic(format!("Failed to seek MCAP message: {e}"))
                    })?;
                    self.file.read_exact(&mut content).map_err(|e| {
                        ReaderError::generic(format!("Failed to read MCAP message: {e}"))
                    })?;
                    let mut buf = Buf::new(&content);
                    let channel_id = buf.read_u16()?;
                    let sequence = buf.read_u32()?;
                    let log_time = buf.read_u64()?;
                    let publish_time = buf.read_u64()?;
                    let data = content[22..].to_vec();

                    if filters.start.is_some_and(|t| log_time < t)
                        || filters.stop.is_some_and(|t| log_time >= t)
                    {
                        continue;
                    }
                    if let Some(topics) = &filters.topics {
                        let topic = self.topic(channel_id).ok_or_else(|| {
                            ReaderError::generic(format!(
                                "MCAP message references unknown channel id {channel_id}"
                            ))
                        })?;
                        if !topics.contains(topic.as_ref()) {
                            continue;
                        }
                    }
                    return Ok(Some(RawMcapMessage {
                        channel_id,
                        sequence,
                        log_time,
                        publish_time,
                        data,
                    }));
                }
                OP_SCHEMA | OP_CHANNEL => {
                    let mut content = vec![0u8; len as usize];
                    self.file.seek(SeekFrom::Start(offset - len)).map_err(|e| {
                        ReaderError::generic(format!("Failed to seek MCAP record: {e}"))
                    })?;
                    self.file.read_exact(&mut content).map_err(|e| {
                        ReaderError::generic(format!("Failed to read MCAP record: {e}"))
                    })?;
                    self.register_record(op, &content)?;
                }
                OP_CHUNK => {
                    // Read the whole chunk record, decompress, and drain it
                    // before resuming the section walk.
                    let chunk_start = offset - 9 - len;
                    let mut content = vec![0u8; len as usize];
                    self.file
                        .seek(SeekFrom::Start(chunk_start + 9))
                        .map_err(|e| {
                            ReaderError::generic(format!("Failed to seek MCAP chunk: {e}"))
                        })?;
                    self.file.read_exact(&mut content).map_err(|e| {
                        ReaderError::generic(format!("Failed to read MCAP chunk: {e}"))
                    })?;
                    let mut buf = Buf::new(&content);
                    let _message_start_time = buf.read_u64()?;
                    let _message_end_time = buf.read_u64()?;
                    let uncompressed_size = buf.read_u64()?;
                    let uncompressed_crc = buf.read_u32()?;
                    let compression = buf.read_str()?;
                    let records_len = buf.read_u64()? as usize;
                    let compressed = buf.take(records_len)?;
                    self.inner = decompress_chunk_records(
                        &compression,
                        compressed,
                        uncompressed_size,
                        uncompressed_crc,
                    )?;
                    self.cursor = 0;
                    if let Some(message) = self.next_inner_message(filters)? {
                        return Ok(Some(message));
                    }
                }
                OP_DATA_END => {
                    self.section_cursor = None;
                    return Ok(None);
                }
                _ => {}
            }
        }
    }

    /// Next message passing the filters, in the file's own log-time order.
    fn next_filtered(&mut self, filters: &Filters) -> Result<Option<RawMcapMessage>, ReaderError> {
        loop {
            if self.section_cursor.is_some() {
                if let Some(message) = self.next_section_message(filters)? {
                    return Ok(Some(message));
                }
                return Ok(None);
            }
            if let Some(message) = self.next_inner_message(filters)? {
                return Ok(Some(message));
            }
            if !self.load_next_chunk(filters)? {
                return Ok(None);
            }
        }
    }
}

/// Topic and time filters shared by the single-file and merged streams.
#[derive(Debug, Clone, Default)]
pub(crate) struct Filters {
    pub topics: Option<HashSet<String>>,
    pub start: Option<u64>,
    pub stop: Option<u64>,
}

impl Filters {
    pub(crate) fn new(
        connections: Option<&[crate::rosbag::types::Connection]>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> Self {
        Self {
            topics: connections.map(|conns| conns.iter().map(|c| c.topic.clone()).collect()),
            start,
            stop,
        }
    }
}

/// Global log-time-order message stream across several MCAP files.///
/// K-way merge over per-file streams: each file emits its own messages in
/// log-time order (writers record chunks in order), and the heap interleaves
/// them without buffering everything first.
pub(crate) struct MergedMcapStream {
    streams: Vec<Option<ChunkMessageStream>>,
    /// Next in-filter message per stream, with its topic resolved at
    /// advance time (the stream is still alive to answer the lookup).
    heads: Vec<Option<(Rc<str>, RawMcapMessage)>>,
    heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize)>>,
    filters: Filters,
    /// First stream error, surfaced as the next yielded item.
    pending_error: Option<ReaderError>,
}

impl MergedMcapStream {
    /// Build the merged stream over `paths`, whose summaries pair by index.
    pub(crate) fn new(
        paths: &[std::path::PathBuf],
        summaries: &[Option<McapSummary>],
        filters: Filters,
    ) -> Result<Self, ReaderError> {
        let mut streams = Vec::with_capacity(paths.len());
        for (path, summary) in paths.iter().zip(summaries) {
            streams.push(Some(ChunkMessageStream::open(path, summary.clone())?));
        }
        let stream_count = streams.len();
        let mut merged = Self {
            heads: vec![None; stream_count],
            streams,
            heap: std::collections::BinaryHeap::with_capacity(stream_count),
            filters,
            pending_error: None,
        };
        for idx in 0..merged.streams.len() {
            merged.advance(idx)?;
        }
        Ok(merged)
    }

    /// Pull stream `idx` forward to its next in-filter message and schedule
    /// it on the heap.
    fn advance(&mut self, idx: usize) -> Result<(), ReaderError> {
        let Some(stream) = self.streams[idx].as_mut() else {
            return Ok(());
        };
        match stream.next_filtered(&self.filters) {
            Ok(Some(message)) => {
                let topic = stream
                    .topic(message.channel_id)
                    .unwrap_or_else(|| Rc::from(""));
                self.heap.push(std::cmp::Reverse((message.log_time, idx)));
                self.heads[idx] = Some((topic, message));
                Ok(())
            }
            Ok(None) => {
                self.streams[idx] = None;
                Ok(())
            }
            Err(e) => {
                if self.pending_error.is_none() {
                    self.pending_error = Some(e);
                }
                self.streams[idx] = None;
                Ok(())
            }
        }
    }
}

impl Iterator for MergedMcapStream {
    type Item = Result<(Rc<str>, RawMcapMessage), ReaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(error) = self.pending_error.take() {
            return Some(Err(error));
        }
        while let Some(std::cmp::Reverse((_, idx))) = self.heap.pop() {
            if let Some((topic, message)) = self.heads[idx].take() {
                if let Err(e) = self.advance(idx) {
                    return Some(Err(e));
                }
                return Some(Ok((topic, message)));
            }
        }
        None
    }
}
