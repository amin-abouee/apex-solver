//! Random-access and streaming reader for VRS files.
//!
//! Opening a file reads only the header, the description record and the
//! index (a few MB even for 20 GB recordings); record payloads are read on
//! demand. Data records are addressable by timestamp (nearest / range
//! queries) and can be pulled one at a time in time order across any set of
//! streams with a [`Cursor`], which is independent of the file handle so a
//! consumer (for example a VIO front end) can interleave sequential pulls
//! with random lookups.
//!
//! ```no_run
//! use apex_io::vrs::{RecordFilter, StreamId, VrsReader};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut vrs = VrsReader::open("recording.vrs")?;
//! let imu = StreamId::new(1202, 1);
//! // Pull IMU + left camera in time order, one record at a time.
//! let filter = RecordFilter::streams([imu, StreamId::new(1201, 1)]);
//! let mut cursor = vrs.cursor(&filter);
//! let mut buf = Vec::new();
//! while let Some(meta) = vrs.next_record_into(&mut cursor, &mut buf)? {
//!     println!("{} t={:.6} {} bytes", meta.stream, meta.timestamp, buf.len());
//! }
//! // Random access: the IMU sample closest to t = 530.0 s.
//! if let Some(i) = vrs.find_nearest(imu, 530.0) {
//!     let _meta = vrs.read_record_into(i, &mut buf)?;
//! }
//! # Ok(())
//! # }
//! ```

use super::codec;
use super::description::{Description, StreamTags, TagMap};
use super::error::{Result, VrsError};
use super::format::{
    Compression, FILE_HEADER_SIZE, FileHeader, RECORD_HEADER_SIZE, RecordHeader, RecordType,
    StreamId,
};
use super::index::{IndexSource, RecordInfo, read_index, read_special_record, rebuild_index};
use super::recordformat::RecordLayout;
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Metadata of a record that has been read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecordMeta {
    /// Position in [`VrsReader::records`].
    pub index: usize,
    /// Owning stream.
    pub stream: StreamId,
    /// Kind of record.
    pub record_type: RecordType,
    /// Capture time in seconds.
    pub timestamp: f64,
    /// `RecordFormat`/`DataLayout` version used by the payload.
    pub format_version: u32,
}

/// A record together with its decompressed payload.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    /// Metadata.
    pub meta: RecordMeta,
    /// Decompressed payload (DataLayout bytes followed by content blocks).
    pub payload: Vec<u8>,
}

/// Which data records a [`Cursor`] visits.
#[derive(Debug, Clone, Default)]
pub struct RecordFilter {
    /// Streams to include; empty means every stream.
    pub streams: HashSet<StreamId>,
    /// First timestamp to visit (seconds, inclusive).
    pub start: Option<f64>,
    /// Timestamp to stop at (seconds, exclusive).
    pub end: Option<f64>,
}

impl RecordFilter {
    /// Visit everything.
    pub fn all() -> Self {
        Self::default()
    }

    /// Visit only the given streams.
    pub fn streams(ids: impl IntoIterator<Item = StreamId>) -> Self {
        Self {
            streams: ids.into_iter().collect(),
            ..Self::default()
        }
    }

    /// Restrict to `[start, end)` seconds.
    pub fn between(mut self, start: f64, end: f64) -> Self {
        self.start = Some(start);
        self.end = Some(end);
        self
    }
}

/// Where the `previous_record_size` chain of the file first breaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainBreak {
    /// Position in [`VrsReader::records`] of the offending record.
    pub record_index: usize,
    /// File offset of its header.
    pub offset: u64,
    /// Which header field disagreed with the index.
    pub field: &'static str,
    /// Value the index implies.
    pub expected: u32,
    /// Value found in the record header.
    pub found: u32,
}

/// A position in the time-ordered sequence of data records.
///
/// Holds no reference to the reader, so it can be stored next to it.
#[derive(Debug, Clone)]
pub struct Cursor {
    filter: RecordFilter,
    pos: usize,
    end: usize,
}

impl Cursor {
    /// Reposition at the first matching record with `timestamp >= t`.
    pub fn seek_time(&mut self, reader: &VrsReader, t: f64) {
        self.pos = reader
            .data_order
            .partition_point(|&i| reader.records[i as usize].timestamp < t);
    }

    /// Records left to visit (upper bound: ignores the stream filter).
    pub fn remaining_upper_bound(&self) -> usize {
        self.end.saturating_sub(self.pos)
    }
}

/// A record exactly as stored on disk: header metadata plus the still
/// (optionally) compressed payload bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredRecord {
    /// Record identity and timing.
    pub meta: RecordMeta,
    /// How the payload below is compressed.
    pub compression: Compression,
    /// Decompressed payload size (0 for stored payloads).
    pub uncompressed_size: u32,
    /// Stored payload (the bytes right after the record header).
    pub body: Vec<u8>,
}

#[derive(Debug, Default)]
struct StreamIndex {
    /// Data records, sorted by timestamp.
    data: Vec<u32>,
    configuration: Vec<u32>,
    state: Vec<u32>,
}

/// An open VRS file.
pub struct VrsReader {
    path: PathBuf,
    file: File,
    file_len: u64,
    header: FileHeader,
    description: Description,
    records: Vec<RecordInfo>,
    index_source: IndexSource,
    by_stream: BTreeMap<StreamId, StreamIndex>,
    /// All data records sorted by (timestamp, file order).
    data_order: Vec<u32>,
    /// Parsed `RF:`/`DL:` tags per (stream, record type, version).
    layouts: BTreeMap<(StreamId, RecordType, u32), RecordLayout>,
    raw: Vec<u8>,
    /// Reused decompression context (a fresh zstd `DCtx` per record would
    /// dominate the cost of pulling small, high-rate records).
    codec: codec::Decompressor,
}

impl VrsReader {
    /// Open a VRS file: header, description and index (rebuilt if missing).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)?;
        let file_len = file.metadata()?.len();

        let mut head = [0u8; FILE_HEADER_SIZE];
        let n = file.read(&mut head)?;
        let header = FileHeader::parse(&head[..n])?
            .ok_or_else(|| VrsError::NotVrs { path: path.clone() })?;

        let desc_offset = u64::try_from(header.description_record_offset)
            .map_err(|_| VrsError::corrupt("negative description record offset"))?;
        let (_, desc_payload) = read_special_record(&mut file, desc_offset, file_len)?;
        let description = Description::parse(&desc_payload)?;

        let (records, index_source) = match read_index(&mut file, &header, file_len)? {
            Some(records) => (records, IndexSource::File),
            None => (
                rebuild_index(&mut file, &header, file_len)?,
                IndexSource::Rebuilt,
            ),
        };

        let mut reader = Self {
            path,
            file,
            file_len,
            header,
            description,
            records,
            index_source,
            by_stream: BTreeMap::new(),
            data_order: Vec::new(),
            layouts: BTreeMap::new(),
            raw: Vec::new(),
            codec: codec::Decompressor::new(),
        };
        reader.build_time_indexes();
        reader.build_layouts();
        Ok(reader)
    }

    fn build_time_indexes(&mut self) {
        let mut by_stream: BTreeMap<StreamId, StreamIndex> = BTreeMap::new();
        for (i, r) in self.records.iter().enumerate() {
            let entry = by_stream.entry(r.stream).or_default();
            match r.record_type {
                RecordType::Data => entry.data.push(i as u32),
                RecordType::Configuration => entry.configuration.push(i as u32),
                RecordType::State => entry.state.push(i as u32),
                _ => {}
            }
        }
        let records = &self.records;
        let key = |i: &u32| {
            let r = &records[*i as usize];
            (r.timestamp, r.offset)
        };
        for idx in by_stream.values_mut() {
            idx.data.sort_by(|a, b| {
                key(a)
                    .partial_cmp(&key(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let mut all: Vec<u32> = by_stream
            .values()
            .flat_map(|s| s.data.iter().copied())
            .collect();
        all.sort_by(|a, b| {
            key(a)
                .partial_cmp(&key(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.by_stream = by_stream;
        self.data_order = all;
    }

    fn build_layouts(&mut self) {
        for (&stream, tags) in &self.description.streams {
            for record_type in [
                RecordType::Configuration,
                RecordType::State,
                RecordType::Data,
            ] {
                let prefix = format!("RF:{}:", record_type.name());
                for key in tags.vrs.keys().filter(|k| k.starts_with(&prefix)) {
                    let Ok(version) = key[prefix.len()..].parse::<u32>() else {
                        continue;
                    };
                    match RecordLayout::from_tags(tags, record_type, version) {
                        Some(Ok(layout)) => {
                            self.layouts.insert((stream, record_type, version), layout);
                        }
                        Some(Err(e)) => tracing::warn!(
                            %stream,
                            "ignoring undecodable record format {key}: {e}"
                        ),
                        None => {}
                    }
                }
            }
        }
    }

    /// Layout of the records of `stream` with this type and format version,
    /// as announced in the stream tags.
    pub fn record_layout(
        &self,
        stream: StreamId,
        record_type: RecordType,
        version: u32,
    ) -> Option<&RecordLayout> {
        self.layouts.get(&(stream, record_type, version))
    }

    // ---- file level -------------------------------------------------------

    /// Path the reader was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Size of the file in bytes.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// File header.
    pub fn header(&self) -> &FileHeader {
        &self.header
    }

    /// Whether the index came from the file or was rebuilt.
    pub fn index_source(&self) -> IndexSource {
        self.index_source
    }

    /// Tags of the whole file.
    pub fn file_tags(&self) -> &TagMap {
        &self.description.file_tags
    }

    /// Streams in the file, in id order.
    pub fn streams(&self) -> Vec<StreamId> {
        self.description.streams.keys().copied().collect()
    }

    /// Tags of one stream.
    pub fn stream_tags(&self, stream: StreamId) -> Option<&StreamTags> {
        self.description.streams.get(&stream)
    }

    /// The parsed description record.
    pub fn description(&self) -> &Description {
        &self.description
    }

    /// Every indexed record, in file order.
    pub fn records(&self) -> &[RecordInfo] {
        &self.records
    }

    // ---- per-stream queries -------------------------------------------------

    /// Indices (into [`Self::records`]) of the stream's data records, sorted
    /// by timestamp.
    pub fn data_records(&self, stream: StreamId) -> &[u32] {
        self.by_stream
            .get(&stream)
            .map_or(&[][..], |s| s.data.as_slice())
    }

    /// Index of the stream's first configuration record, if any.
    pub fn configuration_record(&self, stream: StreamId) -> Option<usize> {
        self.by_stream
            .get(&stream)
            .and_then(|s| s.configuration.first())
            .map(|&i| i as usize)
    }

    /// Index of the stream's first state record, if any.
    pub fn state_record(&self, stream: StreamId) -> Option<usize> {
        self.by_stream
            .get(&stream)
            .and_then(|s| s.state.first())
            .map(|&i| i as usize)
    }

    /// First and last data timestamp of a stream.
    pub fn stream_time_range(&self, stream: StreamId) -> Option<(f64, f64)> {
        let d = self.data_records(stream);
        Some((
            self.records[*d.first()? as usize].timestamp,
            self.records[*d.last()? as usize].timestamp,
        ))
    }

    /// Data records of `stream` with `start <= timestamp < end`.
    pub fn data_records_between(&self, stream: StreamId, start: f64, end: f64) -> &[u32] {
        let d = self.data_records(stream);
        let lo = d.partition_point(|&i| self.records[i as usize].timestamp < start);
        let hi = d.partition_point(|&i| self.records[i as usize].timestamp < end);
        &d[lo..hi.max(lo)]
    }

    /// First data record with `timestamp >= t`.
    pub fn find_at_or_after(&self, stream: StreamId, t: f64) -> Option<usize> {
        let d = self.data_records(stream);
        let p = d.partition_point(|&i| self.records[i as usize].timestamp < t);
        d.get(p).map(|&i| i as usize)
    }

    /// Last data record with `timestamp <= t`.
    pub fn find_at_or_before(&self, stream: StreamId, t: f64) -> Option<usize> {
        let d = self.data_records(stream);
        let p = d.partition_point(|&i| self.records[i as usize].timestamp <= t);
        p.checked_sub(1).map(|p| d[p] as usize)
    }

    /// Data record whose timestamp is closest to `t` (ties favour the earlier).
    pub fn find_nearest(&self, stream: StreamId, t: f64) -> Option<usize> {
        let before = self.find_at_or_before(stream, t);
        let after = self.find_at_or_after(stream, t);
        match (before, after) {
            (Some(b), Some(a)) => {
                let db = t - self.records[b].timestamp;
                let da = self.records[a].timestamp - t;
                Some(if da < db { a } else { b })
            }
            (b, a) => b.or(a),
        }
    }

    // ---- reading payloads ---------------------------------------------------

    /// Read record `index`, decompressing its payload into `buf` (cleared
    /// first, so one buffer can be reused for the whole file).
    pub fn read_record_into(&mut self, index: usize, buf: &mut Vec<u8>) -> Result<RecordMeta> {
        let (info, header) = read_raw_record(&mut self.file, &self.records, index, &mut self.raw)?;
        self.codec.decompress(
            header.compression,
            &self.raw[RECORD_HEADER_SIZE..],
            header.uncompressed_size as usize,
            buf,
        )?;
        Ok(RecordMeta {
            index,
            stream: info.stream,
            record_type: info.record_type,
            timestamp: info.timestamp,
            format_version: header.format_version,
        })
    }

    /// Read record `index` into a freshly allocated [`Record`].
    pub fn read_record(&mut self, index: usize) -> Result<Record> {
        let mut payload = Vec::new();
        let meta = self.read_record_into(index, &mut payload)?;
        Ok(Record { meta, payload })
    }

    /// Read record `index` as stored on disk, without decompressing it.
    /// Copy operations use this to move payload bytes through verbatim.
    pub fn read_stored_record_into(
        &mut self,
        index: usize,
        raw: &mut Vec<u8>,
    ) -> Result<StoredRecord> {
        let (info, header) = read_raw_record(&mut self.file, &self.records, index, raw)?;
        let meta = RecordMeta {
            index,
            stream: info.stream,
            record_type: info.record_type,
            timestamp: info.timestamp,
            format_version: header.format_version,
        };
        Ok(StoredRecord {
            meta,
            compression: header.compression,
            uncompressed_size: header.uncompressed_size,
            body: raw[RECORD_HEADER_SIZE..].to_vec(),
        })
    }

    // ---- sequential access ----------------------------------------------------

    /// A cursor over the data records matching `filter`, in timestamp order.
    pub fn cursor(&self, filter: &RecordFilter) -> Cursor {
        let order = &self.data_order;
        let pos = filter.start.map_or(0, |s| {
            order.partition_point(|&i| self.records[i as usize].timestamp < s)
        });
        let end = filter.end.map_or(order.len(), |e| {
            order.partition_point(|&i| self.records[i as usize].timestamp < e)
        });
        Cursor {
            filter: filter.clone(),
            pos,
            end: end.max(pos),
        }
    }

    /// A time-ordered scan with a background thread reading and
    /// decompressing ahead of the consumer. Yields exactly the records of
    /// the equivalent synchronous cursor loop.
    pub fn sequential(
        &mut self,
        filter: &RecordFilter,
    ) -> Result<super::readahead::SequentialReader<'_>> {
        super::readahead::SequentialReader::new(self, filter)
    }

    /// Advance `cursor` to the next matching record and return its index
    /// into [`Self::records`] without reading the payload. Returns `None` at
    /// the end. Raw-read consumers (copy, prefetch) build on this.
    pub fn next_index(&self, cursor: &mut Cursor) -> Option<usize> {
        while cursor.pos < cursor.end {
            let index = self.data_order[cursor.pos] as usize;
            cursor.pos += 1;
            let stream = self.records[index].stream;
            if cursor.filter.streams.is_empty() || cursor.filter.streams.contains(&stream) {
                return Some(index);
            }
        }
        None
    }

    /// Advance `cursor` and read the next matching record into `buf`.
    /// Returns `None` at the end.
    pub fn next_record_into(
        &mut self,
        cursor: &mut Cursor,
        buf: &mut Vec<u8>,
    ) -> Result<Option<RecordMeta>> {
        while cursor.pos < cursor.end {
            let index = self.data_order[cursor.pos] as usize;
            cursor.pos += 1;
            let stream = self.records[index].stream;
            if cursor.filter.streams.is_empty() || cursor.filter.streams.contains(&stream) {
                return self.read_record_into(index, buf).map(Some);
            }
        }
        Ok(None)
    }

    /// Advance `cursor` and return the next matching record (allocates the
    /// payload; use [`Self::next_record_into`] in hot loops).
    pub fn next_record(&mut self, cursor: &mut Cursor) -> Result<Option<Record>> {
        let mut payload = Vec::new();
        Ok(self
            .next_record_into(cursor, &mut payload)?
            .map(|meta| Record { meta, payload }))
    }

    /// Rebuild the index from the records and report whether it equals the
    /// loaded one (a consistency check for tools).
    pub fn index_matches_scan(&mut self) -> Result<bool> {
        let rebuilt = rebuild_index(&mut self.file, &self.header, self.file_len)?;
        Ok(rebuilt == self.records)
    }

    /// Walk every record header in file order and check it against the
    /// index: `record_size` and stream must match, and `previous_record_size`
    /// must equal the size of the record before. Returns the first
    /// inconsistency, or `None` when the file is intact.
    ///
    /// Sequential and header-only (the payloads are skipped), so it costs one
    /// small read per record; the hot random-access path never pays for it.
    pub fn verify_record_chain(&mut self) -> Result<Option<ChainBreak>> {
        let Some(first) = self.records.first() else {
            return Ok(None);
        };
        self.file.seek(SeekFrom::Start(first.offset))?;
        let mut reader = std::io::BufReader::with_capacity(64 * 1024, &mut self.file);
        let mut head = [0u8; RECORD_HEADER_SIZE];
        for (i, info) in self.records.iter().enumerate() {
            reader.read_exact(&mut head)?;
            let h = RecordHeader::parse(&head)?;
            let mismatch = if h.record_size != info.size {
                Some(("record_size", info.size, h.record_size))
            } else if h.stream_id() != info.stream {
                Some(("stream", 0, 0))
            } else if i > 0 && h.previous_record_size != self.records[i - 1].size {
                Some((
                    "previous_record_size",
                    self.records[i - 1].size,
                    h.previous_record_size,
                ))
            } else {
                None
            };
            if let Some((field, expected, found)) = mismatch {
                return Ok(Some(ChainBreak {
                    record_index: i,
                    offset: info.offset,
                    field,
                    expected,
                    found,
                }));
            }
            reader.seek_relative(i64::from(info.size) - RECORD_HEADER_SIZE as i64)?;
        }
        Ok(None)
    }

    /// The underlying file handle (used by the read-ahead worker to take its
    /// own copy).
    pub(crate) fn file_handle(&self) -> &File {
        &self.file
    }
}

/// Read the on-disk record `index` (header + stored payload) into `raw` and
/// validate it against the index. Shared by the decompressing and raw read
/// paths.
fn read_raw_record(
    file: &mut File,
    records: &[RecordInfo],
    index: usize,
    raw: &mut Vec<u8>,
) -> Result<(RecordInfo, RecordHeader)> {
    let info = *records
        .get(index)
        .ok_or_else(|| VrsError::Usage(format!("record index {index} out of range")))?;
    file.seek(SeekFrom::Start(info.offset))?;
    raw.resize(info.size as usize, 0);
    file.read_exact(raw)?;

    let header = RecordHeader::parse(raw)?;
    if header.record_size != info.size || header.stream_id() != info.stream {
        return Err(VrsError::corrupt(format!(
            "record {index} at offset {} does not match the index",
            info.offset
        )));
    }
    Ok((info, header))
}
