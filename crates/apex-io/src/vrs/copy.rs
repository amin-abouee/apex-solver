//! Copy (and optionally trim / re-compress) a VRS file.
//!
//! Tags, record formats and payload bytes are preserved; only the container
//! framing and compression are rewritten. Configuration and state records of
//! the selected streams are always kept; data records are filtered by
//! stream and time window and written in capture-time order.

use super::error::Result;
use super::format::{Compression, RecordType, StreamId};
use super::reader::{RecordFilter, VrsReader};
use super::writer::{VrsWriteOptions, VrsWriter, WriteSummary};
use std::path::Path;

/// Which part of the source to copy.
#[derive(Debug, Clone, Default)]
pub struct CopyOptions {
    /// Streams to keep; empty means all.
    pub streams: Vec<StreamId>,
    /// First data timestamp to keep (seconds, inclusive).
    pub start: Option<f64>,
    /// Data timestamp to stop at (seconds, exclusive).
    pub end: Option<f64>,
    /// Compression of the output records.
    pub compression: Option<Compression>,
}

/// Copy `src` to `dst` (which must not exist).
pub fn copy_filtered(
    src: &mut VrsReader,
    dst: impl AsRef<Path>,
    options: &CopyOptions,
) -> Result<WriteSummary> {
    let wanted: Vec<StreamId> = if options.streams.is_empty() {
        src.streams()
    } else {
        options
            .streams
            .iter()
            .copied()
            .filter(|s| src.streams().contains(s))
            .collect()
    };

    let mut write_opts = VrsWriteOptions::default();
    if let Some(c) = options.compression {
        write_opts.compression = c;
    }
    let mut out = VrsWriter::create(dst, write_opts)?;
    for (k, v) in src.file_tags() {
        out.set_file_tag(k.clone(), v.clone())?;
    }
    for &id in &wanted {
        let tags = src.stream_tags(id).cloned().unwrap_or_default();
        out.add_stream(id, tags)?;
    }

    let mut buf = Vec::new();
    // Configuration and state records first, as a recorder would write them:
    // globally ordered by timestamp (then original file order), because
    // readers expect the record sequence to be time-sorted.
    let mut special: Vec<(f64, u64, usize)> = Vec::new();
    for &id in &wanted {
        for index in [src.configuration_record(id), src.state_record(id)]
            .into_iter()
            .flatten()
        {
            let info = src.records()[index];
            special.push((info.timestamp, info.offset, index));
        }
    }
    special.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    for (_, _, index) in special {
        let meta = src.read_record_into(index, &mut buf)?;
        out.write_record(
            meta.stream,
            meta.record_type,
            meta.format_version,
            meta.timestamp,
            &buf,
        )?;
    }

    let mut filter = RecordFilter::streams(wanted.iter().copied());
    filter.start = options.start;
    filter.end = options.end;
    let mut cursor = src.cursor(&filter);
    while let Some(meta) = src.next_record_into(&mut cursor, &mut buf)? {
        debug_assert_eq!(meta.record_type, RecordType::Data);
        out.write_record(
            meta.stream,
            meta.record_type,
            meta.format_version,
            meta.timestamp,
            &buf,
        )?;
    }
    out.finish()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::vrs::description::StreamTags;

    fn make(path: &Path) {
        let mut w = VrsWriter::create(path, VrsWriteOptions::default()).unwrap();
        w.set_file_tag("session_id", "s1").unwrap();
        for t in [600, 601] {
            let mut tags = StreamTags::default();
            tags.user.insert("n".into(), t.to_string());
            w.add_stream(StreamId::new(t, 1), tags).unwrap();
        }
        for t in [600, 601] {
            let id = StreamId::new(t, 1);
            w.write_record(id, RecordType::Configuration, 1, 0.0, b"config")
                .unwrap();
            for i in 0..100 {
                w.write_record(
                    id,
                    RecordType::Data,
                    1,
                    f64::from(i) * 0.01,
                    &[i as u8; 200],
                )
                .unwrap();
            }
        }
        w.finish().unwrap();
    }

    #[test]
    fn trims_streams_and_time_and_keeps_tags() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a.vrs"), dir.path().join("b.vrs"));
        make(&a);
        let mut src = VrsReader::open(&a).unwrap();
        let opts = CopyOptions {
            streams: vec![StreamId::new(601, 1)],
            start: Some(0.20),
            end: Some(0.50),
            compression: Some(Compression::Lz4),
        };
        let summary = copy_filtered(&mut src, &b, &opts).unwrap();
        assert_eq!(summary.records, 31, "config + 30 data");

        let mut dst = VrsReader::open(&b).unwrap();
        assert_eq!(dst.streams(), [StreamId::new(601, 1)]);
        assert_eq!(dst.file_tags()["session_id"], "s1");
        assert_eq!(
            dst.stream_tags(StreamId::new(601, 1)).unwrap().user["n"],
            "601"
        );
        let d = dst.data_records(StreamId::new(601, 1));
        assert_eq!(d.len(), 30);
        let first = dst.read_record(d[0] as usize).unwrap();
        assert!((first.meta.timestamp - 0.20).abs() < 1e-9);
        assert_eq!(first.payload, vec![20u8; 200]);
        let cfg = dst.configuration_record(StreamId::new(601, 1)).unwrap();
        assert_eq!(dst.read_record(cfg).unwrap().payload, b"config");
    }
}
