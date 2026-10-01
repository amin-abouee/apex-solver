//! The description record: stream tags and file tags.
//!
//! Payload layout (little-endian, uncompressed):
//!
//! ```text
//! u32 nStreams
//! nStreams x { i32 typeId, u16 instance, TagMap user, TagMap vrs }
//! TagMap fileTags
//! TagMap = u32 n, n x { String key, String value }
//! String = u32 len, len bytes UTF-8
//! ```

use super::error::{Result, VrsError};
use super::format::StreamId;
use std::collections::BTreeMap;

/// String-to-string tag map (sorted, so output is deterministic).
pub type TagMap = BTreeMap<String, String>;

/// Tags attached to one stream.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamTags {
    /// Free-form tags set by the recorder (`device_serial`, `camera_id`, ...).
    pub user: TagMap,
    /// Tags VRS itself maintains (`RF:...`, `DL:...`, recordable name, ...).
    pub vrs: TagMap,
}

/// Parsed description record.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Description {
    /// Tags per stream.
    pub streams: BTreeMap<StreamId, StreamTags>,
    /// Tags of the whole file (`calib_json`, `session_id`, ...).
    pub file_tags: TagMap,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| VrsError::corrupt("description record truncated"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        // Tag values are UTF-8 by convention; tolerate stray bytes.
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn map(&mut self) -> Result<TagMap> {
        let n = self.u32()?;
        let mut map = TagMap::new();
        for _ in 0..n {
            let key = self.string()?;
            let value = self.string()?;
            map.insert(key, value);
        }
        Ok(map)
    }
}

fn put_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_map(out: &mut Vec<u8>, map: &TagMap) {
    out.extend_from_slice(&(map.len() as u32).to_le_bytes());
    for (k, v) in map {
        put_string(out, k);
        put_string(out, v);
    }
}

impl Description {
    /// Parse a description record payload.
    pub fn parse(payload: &[u8]) -> Result<Self> {
        let mut c = Cursor {
            buf: payload,
            pos: 0,
        };
        let n = c.u32()?;
        let mut streams = BTreeMap::new();
        for _ in 0..n {
            let id = StreamId::from_bytes(c.take(6)?)?;
            let user = c.map()?;
            let vrs = c.map()?;
            streams.insert(id, StreamTags { user, vrs });
        }
        let file_tags = c.map()?;
        Ok(Self { streams, file_tags })
    }

    /// Serialise to a description record payload.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.streams.len() as u32).to_le_bytes());
        for (id, tags) in &self.streams {
            out.extend_from_slice(&id.to_bytes());
            put_map(&mut out, &tags.user);
            put_map(&mut out, &tags.vrs);
        }
        put_map(&mut out, &self.file_tags);
        out
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn sample() -> Description {
        let mut d = Description::default();
        let mut tags = StreamTags::default();
        tags.user.insert("camera_id".into(), "2".into());
        tags.vrs
            .insert("RF:Data:2".into(), "data_layout/size=92+image/jpg".into());
        d.streams.insert(StreamId::new(214, 1), tags);
        d.streams
            .insert(StreamId::new(1202, 2), StreamTags::default());
        d.file_tags.insert("session_id".into(), "abc".into());
        d.file_tags.insert("unicode".into(), "héllo".into());
        d
    }

    #[test]
    fn round_trips() {
        let d = sample();
        assert_eq!(Description::parse(&d.to_bytes()).unwrap(), d);
    }

    #[test]
    fn empty_description_round_trips() {
        let d = Description::default();
        assert_eq!(d.to_bytes().len(), 8);
        assert_eq!(Description::parse(&d.to_bytes()).unwrap(), d);
    }

    #[test]
    fn truncation_is_an_error_not_a_panic() {
        let bytes = sample().to_bytes();
        for cut in 0..bytes.len() {
            assert!(Description::parse(&bytes[..cut]).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn absurd_counts_do_not_allocate_or_hang() {
        // nStreams = u32::MAX with no data behind it.
        assert!(Description::parse(&u32::MAX.to_le_bytes()).is_err());
    }
}
