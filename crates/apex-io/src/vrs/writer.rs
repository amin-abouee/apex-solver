//! Streaming VRS writer.
//!
//! Output layout (single pass, no temporary files):
//!
//! ```text
//! FileHeader | description record | user records ... | index record
//! ```
//!
//! Declare streams and tags first; the description record is written when
//! the first record arrives (after that the set of streams is fixed). On
//! [`VrsWriter::finish`] the index record is appended and the file header is
//! patched with the final offsets.

use super::codec;
use super::datalayout::DataLayout;
use super::description::{Description, StreamTags};
use super::error::{Result, VrsError};
use super::format::{
    Compression, DESCRIPTION_TYPE_ID, FILE_HEADER_SIZE, FileHeader, INDEX_TYPE_ID, IndexEntry,
    RECORD_HEADER_SIZE, RecordHeader, RecordType, StreamId,
};
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Format version of the description and index records.
const SPECIAL_RECORD_VERSION: u32 = 2;

/// Writer settings.
#[derive(Debug, Clone)]
pub struct VrsWriteOptions {
    /// Default compression for records.
    pub compression: Compression,
    /// Payloads smaller than this are stored uncompressed (compressing tiny
    /// records only wastes time).
    pub min_compress_size: usize,
}

impl Default for VrsWriteOptions {
    fn default() -> Self {
        Self {
            compression: Compression::Zstd,
            min_compress_size: 128,
        }
    }
}

/// What was written, reported by [`VrsWriter::finish`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteSummary {
    /// Final file path.
    pub path: PathBuf,
    /// Stream records written (excluding description and index).
    pub records: u64,
    /// Final file size in bytes.
    pub bytes: u64,
}

/// Writes a VRS file record by record.
pub struct VrsWriter {
    path: PathBuf,
    out: Option<BufWriter<std::fs::File>>,
    options: VrsWriteOptions,
    description: Description,
    started: bool,
    pos: u64,
    prev_size: u32,
    first_user_offset: u64,
    entries: Vec<IndexEntry>,
}

impl VrsWriter {
    /// Create `path` (which must not exist) and write a placeholder header.
    pub fn create(path: impl AsRef<Path>, options: VrsWriteOptions) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        out.write_all(&[0u8; FILE_HEADER_SIZE])?;
        Ok(Self {
            path,
            out: Some(out),
            options,
            description: Description::default(),
            started: false,
            pos: FILE_HEADER_SIZE as u64,
            prev_size: 0,
            first_user_offset: 0,
            entries: Vec::new(),
        })
    }

    fn ensure_not_started(&self, what: &str) -> Result<()> {
        if self.started {
            Err(VrsError::Usage(format!(
                "cannot {what} after the first record was written"
            )))
        } else {
            Ok(())
        }
    }

    /// Set a file-level tag.
    pub fn set_file_tag(&mut self, key: impl Into<String>, value: impl Into<String>) -> Result<()> {
        self.ensure_not_started("change file tags")?;
        self.description.file_tags.insert(key.into(), value.into());
        Ok(())
    }

    /// Declare a stream with its tags.
    pub fn add_stream(&mut self, id: StreamId, tags: StreamTags) -> Result<()> {
        self.ensure_not_started("add streams")?;
        if id.type_id <= DESCRIPTION_TYPE_ID {
            return Err(VrsError::Usage(format!(
                "stream type ids 1 and 2 are reserved (got {id})"
            )));
        }
        if self.description.streams.insert(id, tags).is_some() {
            return Err(VrsError::Usage(format!("stream {id} declared twice")));
        }
        Ok(())
    }

    /// Declare the format of records `(record_type, version)` of a stream:
    /// the `RecordFormat` text and one layout per `data_layout` block.
    pub fn define_record_format(
        &mut self,
        stream: StreamId,
        record_type: RecordType,
        version: u32,
        format: &str,
        layouts: &[&DataLayout],
    ) -> Result<()> {
        self.ensure_not_started("define record formats")?;
        let tags = self
            .description
            .streams
            .get_mut(&stream)
            .ok_or_else(|| VrsError::Usage(format!("stream {stream} was not declared")))?;
        let name = record_type.name();
        tags.vrs
            .insert(format!("RF:{name}:{version}"), format.to_string());
        for (i, layout) in layouts.iter().enumerate() {
            tags.vrs
                .insert(format!("DL:{name}:{version}:{i}"), layout.to_json());
        }
        Ok(())
    }

    fn out(&mut self) -> Result<&mut BufWriter<std::fs::File>> {
        self.out
            .as_mut()
            .ok_or_else(|| VrsError::Usage("writer already finished".into()))
    }

    fn write_framed(&mut self, header: &RecordHeader, payload: &[u8]) -> Result<()> {
        let bytes = header.to_bytes();
        let out = self.out()?;
        out.write_all(&bytes)?;
        out.write_all(payload)?;
        self.pos += u64::from(header.record_size);
        self.prev_size = header.record_size;
        Ok(())
    }

    fn start(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }
        let payload = self.description.to_bytes();
        let header = RecordHeader {
            record_size: record_size(payload.len())?,
            previous_record_size: 0,
            type_id: DESCRIPTION_TYPE_ID,
            format_version: SPECIAL_RECORD_VERSION,
            timestamp: f64::MAX,
            instance: 0,
            record_type: RecordType::Data,
            compression: Compression::None,
            uncompressed_size: 0,
        };
        self.write_framed(&header, &payload)?;
        self.first_user_offset = self.pos;
        self.started = true;
        Ok(())
    }

    /// Write one record with the default compression policy.
    pub fn write_record(
        &mut self,
        stream: StreamId,
        record_type: RecordType,
        version: u32,
        timestamp: f64,
        payload: &[u8],
    ) -> Result<()> {
        self.write_record_with(
            stream,
            record_type,
            version,
            timestamp,
            payload,
            self.options.compression,
        )
    }

    /// Write one record with an explicit compression (use
    /// [`Compression::None`] for already-compressed content such as JPEG).
    pub fn write_record_with(
        &mut self,
        stream: StreamId,
        record_type: RecordType,
        version: u32,
        timestamp: f64,
        payload: &[u8],
        compression: Compression,
    ) -> Result<()> {
        if !self.description.streams.contains_key(&stream) {
            return Err(VrsError::Usage(format!("stream {stream} was not declared")));
        }
        self.start()?;

        let mut stored = None;
        let mut used = Compression::None;
        if compression != Compression::None && payload.len() >= self.options.min_compress_size {
            let packed = codec::compress(compression, payload)?;
            if packed.len() < payload.len() {
                stored = Some(packed);
                used = compression;
            }
        }
        let body: &[u8] = stored.as_deref().unwrap_or(payload);
        let header = RecordHeader {
            record_size: record_size(body.len())?,
            previous_record_size: self.prev_size,
            type_id: stream.type_id,
            format_version: version,
            timestamp,
            instance: stream.instance,
            record_type,
            compression: used,
            uncompressed_size: if used == Compression::None {
                0
            } else {
                u32::try_from(payload.len())
                    .map_err(|_| VrsError::Usage("record larger than 4 GiB".into()))?
            },
        };
        self.write_framed(&header, body)?;
        self.entries.push(IndexEntry {
            timestamp,
            record_size: header.record_size,
            record_type,
            stream,
        });
        Ok(())
    }

    /// Append the index, patch the header and flush.
    pub fn finish(&mut self) -> Result<WriteSummary> {
        self.start()?;

        // Index payload: stream ids, record count, one zstd frame of entries.
        let streams: Vec<StreamId> = self.description.streams.keys().copied().collect();
        let mut prefix = Vec::with_capacity(8 + streams.len() * 6);
        prefix.extend_from_slice(&(streams.len() as u32).to_le_bytes());
        for s in &streams {
            prefix.extend_from_slice(&s.to_bytes());
        }
        prefix.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        let entries: Vec<u8> = self.entries.iter().flat_map(|e| e.to_bytes()).collect();
        let mut payload = prefix.clone();
        payload.extend_from_slice(&codec::compress(Compression::Zstd, &entries)?);

        let index_offset = self.pos;
        let header = RecordHeader {
            record_size: record_size(payload.len())?,
            previous_record_size: self.prev_size,
            type_id: INDEX_TYPE_ID,
            format_version: SPECIAL_RECORD_VERSION,
            timestamp: f64::MAX,
            instance: 0,
            record_type: RecordType::Data,
            compression: Compression::Zstd,
            uncompressed_size: u32::try_from(prefix.len() + entries.len())
                .map_err(|_| VrsError::Usage("index larger than 4 GiB".into()))?,
        };
        self.write_framed(&header, &payload)?;

        let file_header = FileHeader {
            creation_id: creation_id(),
            file_header_size: FILE_HEADER_SIZE as u32,
            record_header_size: RECORD_HEADER_SIZE as u32,
            index_record_offset: index_offset as i64,
            description_record_offset: FILE_HEADER_SIZE as i64,
            first_user_record_offset: self.first_user_offset as i64,
        };
        let total = self.pos;
        let mut out = self
            .out
            .take()
            .ok_or_else(|| VrsError::Usage("writer already finished".into()))?;
        out.flush()?;
        let file = out.get_mut();
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&file_header.to_bytes())?;
        file.flush()?;
        Ok(WriteSummary {
            path: self.path.clone(),
            records: self.entries.len() as u64,
            bytes: total,
        })
    }
}

impl Drop for VrsWriter {
    fn drop(&mut self) {
        if self.out.is_some()
            && let Err(e) = self.finish()
        {
            tracing::warn!(
                "VRS file {} not finalised cleanly: {e}",
                self.path.display()
            );
        }
    }
}

fn record_size(payload_len: usize) -> Result<u32> {
    u32::try_from(payload_len + RECORD_HEADER_SIZE)
        .map_err(|_| VrsError::Usage("record larger than 4 GiB".into()))
}

fn creation_id() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0x5652_5300, |d| d.as_nanos() as u64)
}

/// Per-stream tag helper used by converters.
pub fn stream_tags(name: &str, user: BTreeMap<String, String>) -> StreamTags {
    let mut tags = StreamTags {
        user,
        ..StreamTags::default()
    };
    tags.vrs
        .insert("VRS_Original_Recordable_Name".into(), name.to_string());
    tags
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::vrs::datalayout::LayoutWriter;
    use crate::vrs::reader::{RecordFilter, VrsReader};
    use crate::vrs::{Block, IndexSource};
    use tempfile::tempdir;

    const LAYOUT: &str = r#"{"data_layout":[
        {"name":"seq","type":"DataPieceValue<uint32_t>","offset":0},
        {"name":"label","type":"DataPieceString","index":0}]}"#;

    fn write_sample(path: &Path, n: u32) -> Vec<(StreamId, f64, Vec<u8>)> {
        let layout = DataLayout::from_json(LAYOUT).unwrap();
        let a = StreamId::new(500, 1);
        let b = StreamId::new(501, 1);
        let mut w = VrsWriter::create(path, VrsWriteOptions::default()).unwrap();
        w.set_file_tag("session_id", "test").unwrap();
        for id in [a, b] {
            w.add_stream(id, stream_tags("Test Stream", BTreeMap::new()))
                .unwrap();
            w.define_record_format(id, RecordType::Data, 1, "data_layout+custom", &[&layout])
                .unwrap();
        }
        let mut expected = Vec::new();
        for i in 0..n {
            for (id, t0) in [(a, 0.0), (b, 0.0005)] {
                let ts = f64::from(i) * 0.001 + t0;
                let mut lw = LayoutWriter::new(&layout);
                lw.set_u32("seq", i).unwrap();
                lw.set_string("label", &format!("{id}#{i}")).unwrap();
                let mut payload = lw.finish();
                payload.extend_from_slice(&vec![(i % 251) as u8; 300]); // custom block, compressible
                w.write_record(id, RecordType::Data, 1, ts, &payload)
                    .unwrap();
                expected.push((id, ts, payload));
            }
        }
        w.finish().unwrap();
        expected
    }

    #[test]
    fn round_trip_through_the_reader() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.vrs");
        let expected = write_sample(&path, 200);

        let mut r = VrsReader::open(&path).unwrap();
        assert_eq!(r.index_source(), IndexSource::File);
        assert!(r.index_matches_scan().unwrap());
        assert_eq!(
            r.file_tags().get("session_id").map(String::as_str),
            Some("test")
        );
        assert_eq!(r.streams(), [StreamId::new(500, 1), StreamId::new(501, 1)]);
        assert_eq!(r.data_records(StreamId::new(500, 1)).len(), 200);

        // Pull everything in time order: interleaves the two streams exactly
        // as written, with payloads intact.
        let mut cur = r.cursor(&RecordFilter::all());
        let mut buf = Vec::new();
        for (id, ts, payload) in &expected {
            let meta = r.next_record_into(&mut cur, &mut buf).unwrap().unwrap();
            assert_eq!((meta.stream, meta.timestamp), (*id, *ts));
            assert_eq!(&buf, payload);
        }
        assert!(r.next_record_into(&mut cur, &mut buf).unwrap().is_none());

        // Typed access through the stored RF/DL tags.
        let idx = r.find_nearest(StreamId::new(501, 1), 0.0505).unwrap();
        let meta = r.read_record_into(idx, &mut buf).unwrap();
        let layout = r.record_layout(meta.stream, RecordType::Data, 1).unwrap();
        let blocks = layout.split(&buf).unwrap();
        match (&blocks[0], &blocks[1]) {
            (Block::Layout(v), Block::Custom(c)) => {
                assert_eq!(v.get_u32("seq").unwrap(), 50);
                assert_eq!(v.get_string("label").unwrap(), "501-1#50");
                assert_eq!(c.len(), 300);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn small_records_are_stored_and_large_ones_compressed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("c.vrs");
        let id = StreamId::new(700, 1);
        let mut w = VrsWriter::create(&path, VrsWriteOptions::default()).unwrap();
        w.add_stream(id, StreamTags::default()).unwrap();
        w.write_record(id, RecordType::Data, 1, 0.0, &[1u8; 16])
            .unwrap();
        w.write_record(id, RecordType::Data, 1, 1.0, &[2u8; 100_000])
            .unwrap();
        w.write_record_with(
            id,
            RecordType::Data,
            1,
            2.0,
            &[3u8; 100_000],
            Compression::None,
        )
        .unwrap();
        let summary = w.finish().unwrap();
        assert_eq!(summary.records, 3);
        assert_eq!(summary.bytes, std::fs::metadata(&path).unwrap().len());

        let mut r = VrsReader::open(&path).unwrap();
        let sizes: Vec<u32> = r.records().iter().map(|i| i.size).collect();
        assert_eq!(sizes[0] as usize, RECORD_HEADER_SIZE + 16);
        assert!(
            sizes[1] < 2_000,
            "compressible record not compressed: {}",
            sizes[1]
        );
        assert_eq!(sizes[2] as usize, RECORD_HEADER_SIZE + 100_000);
        let mut buf = Vec::new();
        r.read_record_into(1, &mut buf).unwrap();
        assert_eq!(buf, vec![2u8; 100_000]);
    }

    #[test]
    fn truncated_file_is_recovered_by_rebuilding_the_index() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cut.vrs");
        write_sample(&path, 100);
        let full = std::fs::metadata(&path).unwrap().len();
        // Chop off the index record and part of the last data record.
        let r = VrsReader::open(&path).unwrap();
        let last = r.records().last().copied().unwrap();
        drop(r);
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(last.offset + 10).unwrap();
        assert!(full > last.offset + 10);

        let r = VrsReader::open(&path).unwrap();
        assert_eq!(r.index_source(), IndexSource::Rebuilt);
        assert_eq!(r.records().len(), 199, "all complete records survive");
    }

    #[test]
    fn index_at_the_end_is_not_mistaken_for_a_record_by_a_scan() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("e.vrs");
        write_sample(&path, 20);
        let mut r = VrsReader::open(&path).unwrap();
        assert!(r.index_matches_scan().unwrap());
    }

    #[test]
    fn usage_errors_are_reported() {
        let dir = tempdir().unwrap();
        let id = StreamId::new(900, 1);
        let mut w =
            VrsWriter::create(dir.path().join("u.vrs"), VrsWriteOptions::default()).unwrap();
        assert!(
            w.add_stream(StreamId::new(2, 1), StreamTags::default())
                .is_err()
        );
        w.add_stream(id, StreamTags::default()).unwrap();
        assert!(w.add_stream(id, StreamTags::default()).is_err());
        assert!(
            w.write_record(StreamId::new(901, 1), RecordType::Data, 1, 0.0, b"x")
                .is_err()
        );
        w.write_record(id, RecordType::Data, 1, 0.0, b"x").unwrap();
        assert!(
            w.add_stream(StreamId::new(902, 1), StreamTags::default())
                .is_err()
        );
        assert!(w.set_file_tag("a", "b").is_err());
        w.finish().unwrap();
        assert!(w.write_record(id, RecordType::Data, 1, 1.0, b"y").is_err());
        // refuses to overwrite
        assert!(VrsWriter::create(dir.path().join("u.vrs"), VrsWriteOptions::default()).is_err());
    }

    #[test]
    fn dropping_an_unfinished_writer_still_yields_a_valid_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("d.vrs");
        {
            let id = StreamId::new(800, 1);
            let mut w = VrsWriter::create(&path, VrsWriteOptions::default()).unwrap();
            w.add_stream(id, StreamTags::default()).unwrap();
            w.write_record(id, RecordType::Data, 1, 0.5, b"hello")
                .unwrap();
        }
        let r = VrsReader::open(&path).unwrap();
        assert_eq!(r.index_source(), IndexSource::File);
        assert_eq!(r.records().len(), 1);
    }
}
