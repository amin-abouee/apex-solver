//! Low-level ROS1 record I/O.
//!
//! Every record on disk has the shape:
//!
//! ```text
//! <u32 header_len LE>
//! <header_len bytes>     // sequence of `<u32 field_len LE><name=value bytes>`
//! <u32 data_len LE>
//! <data_len bytes>
//! ```
//!
//! Header fields are length-prefixed `key=value` byte strings; values may be
//! arbitrary bytes (including binary numerics) so we keep them as `Vec<u8>`.

use crate::rosbag::error::{BagError, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Read, Write};

use super::format::OpCode;

/// Parsed header: maps field name to raw value bytes.  Insertion order is not
/// preserved (ROS1 readers must not rely on it).
#[derive(Debug, Clone, Default)]
pub struct RecordHeader {
    fields: HashMap<String, Vec<u8>>,
}

impl RecordHeader {
    /// Create an empty header.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a field, replacing any existing value.
    pub fn put(&mut self, name: impl Into<String>, value: impl Into<Vec<u8>>) -> &mut Self {
        self.fields.insert(name.into(), value.into());
        self
    }

    /// Convenience: insert a `u8` op-code as the `op` field.
    pub fn put_op(&mut self, op: OpCode) -> &mut Self {
        self.put("op", vec![op.as_u8()])
    }

    /// Convenience: insert a little-endian `u32`.
    pub fn put_u32(&mut self, name: impl Into<String>, value: u32) -> &mut Self {
        self.put(name, value.to_le_bytes().to_vec())
    }

    /// Convenience: insert a little-endian `u64`.
    pub fn put_u64(&mut self, name: impl Into<String>, value: u64) -> &mut Self {
        self.put(name, value.to_le_bytes().to_vec())
    }

    /// Convenience: insert a little-endian `u8`.
    pub fn put_u8(&mut self, name: impl Into<String>, value: u8) -> &mut Self {
        self.put(name, vec![value])
    }

    /// Convenience: insert a string (no NUL terminator, no length prefix —
    /// the outer field length covers it).
    pub fn put_str(&mut self, name: impl Into<String>, value: &str) -> &mut Self {
        self.put(name, value.as_bytes().to_vec())
    }

    /// Convenience: insert a ROS1 timestamp encoded as `<u32 sec LE><u32 nsec LE>`.
    pub fn put_time(&mut self, name: impl Into<String>, secs: u32, nsecs: u32) -> &mut Self {
        let mut buf = Vec::with_capacity(8);
        buf.extend_from_slice(&secs.to_le_bytes());
        buf.extend_from_slice(&nsecs.to_le_bytes());
        self.put(name, buf)
    }

    /// Read a field as raw bytes.
    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.fields.get(name).map(Vec::as_slice)
    }

    /// Read a field as a UTF-8 string.
    pub fn get_str(&self, name: &str) -> Result<&str> {
        let bytes = self.require(name)?;
        std::str::from_utf8(bytes).map_err(|_| BagError::Ros1MalformedRecord {
            reason: format!("field `{name}` is not valid UTF-8"),
        })
    }

    /// Read a required field, returning an error if it is missing.
    pub fn require(&self, name: &str) -> Result<&[u8]> {
        self.get(name).ok_or_else(|| BagError::Ros1MalformedRecord {
            reason: format!("missing required header field `{name}`"),
        })
    }

    /// Read a little-endian `u8`.
    pub fn get_u8(&self, name: &str) -> Result<u8> {
        let bytes = self.require(name)?;
        if bytes.len() != 1 {
            return Err(BagError::Ros1MalformedRecord {
                reason: format!("field `{name}` expected 1 byte, got {}", bytes.len()),
            });
        }
        Ok(bytes[0])
    }

    /// Read a little-endian `u32`.
    pub fn get_u32(&self, name: &str) -> Result<u32> {
        let bytes = self.require(name)?;
        if bytes.len() != 4 {
            return Err(BagError::Ros1MalformedRecord {
                reason: format!("field `{name}` expected 4 bytes, got {}", bytes.len()),
            });
        }
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read a little-endian `u64`.
    pub fn get_u64(&self, name: &str) -> Result<u64> {
        let bytes = self.require(name)?;
        if bytes.len() != 8 {
            return Err(BagError::Ros1MalformedRecord {
                reason: format!("field `{name}` expected 8 bytes, got {}", bytes.len()),
            });
        }
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(arr))
    }

    /// Read a ROS1 timestamp as nanoseconds since epoch.
    pub fn get_time_nanos(&self, name: &str) -> Result<u64> {
        let bytes = self.require(name)?;
        if bytes.len() != 8 {
            return Err(BagError::Ros1MalformedRecord {
                reason: format!(
                    "field `{name}` expected 8 bytes (time), got {}",
                    bytes.len()
                ),
            });
        }
        let secs = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64;
        let nsecs = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as u64;
        Ok(secs.saturating_mul(1_000_000_000).saturating_add(nsecs))
    }

    /// Read the `op` field as a typed [`OpCode`].
    pub fn op(&self) -> Result<OpCode> {
        let op = self.get_u8("op")?;
        OpCode::from_u8(op).ok_or(BagError::Ros1UnknownOpCode { op })
    }

    /// Serialize the header to its on-wire representation (without the outer
    /// `u32 header_len` prefix).
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        // Sort field names so writers are deterministic; readers don't care.
        let mut keys: Vec<&String> = self.fields.keys().collect();
        keys.sort();
        for k in keys {
            let v = &self.fields[k];
            let field_len = (k.len() + 1 + v.len()) as u32;
            out.extend_from_slice(&field_len.to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            out.push(b'=');
            out.extend_from_slice(v);
        }
        out
    }
}

/// A single record read from a bag.  `data` is the raw payload; for chunk
/// records this is the (possibly compressed) inner record stream.
#[derive(Debug, Clone)]
pub struct Record {
    pub header: RecordHeader,
    pub data: Vec<u8>,
}

/// Parse a header block (without the outer length prefix) into a [`RecordHeader`].
pub fn parse_header(mut bytes: &[u8]) -> Result<RecordHeader> {
    let mut header = RecordHeader::new();
    while !bytes.is_empty() {
        if bytes.len() < 4 {
            return Err(BagError::Ros1MalformedRecord {
                reason: "truncated header field length".into(),
            });
        }
        let field_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        bytes = &bytes[4..];
        if bytes.len() < field_len {
            return Err(BagError::Ros1MalformedRecord {
                reason: format!(
                    "header field truncated (need {field_len}, have {})",
                    bytes.len()
                ),
            });
        }
        let field = &bytes[..field_len];
        bytes = &bytes[field_len..];
        let eq =
            field
                .iter()
                .position(|&b| b == b'=')
                .ok_or_else(|| BagError::Ros1MalformedRecord {
                    reason: "header field missing `=`".into(),
                })?;
        let name =
            std::str::from_utf8(&field[..eq]).map_err(|_| BagError::Ros1MalformedRecord {
                reason: "header field name not UTF-8".into(),
            })?;
        let value = field[eq + 1..].to_vec();
        header.put(name.to_string(), value);
    }
    Ok(header)
}

/// Read one record (header + data) from `r`.  Returns `Ok(None)` on clean EOF
/// before the next header length, propagates other I/O errors.
pub fn read_record<R: Read>(r: &mut R) -> Result<Option<Record>> {
    let header_len = match r.read_u32::<LittleEndian>() {
        Ok(v) => v as usize,
        Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(BagError::Io(e)),
    };
    let mut header_bytes = vec![0u8; header_len];
    r.read_exact(&mut header_bytes)?;
    let header = parse_header(&header_bytes)?;

    let data_len = r.read_u32::<LittleEndian>()? as usize;
    let mut data = vec![0u8; data_len];
    r.read_exact(&mut data)?;
    Ok(Some(Record { header, data }))
}

/// Write a record (header + data) to `w`.
pub fn write_record<W: Write>(w: &mut W, header: &RecordHeader, data: &[u8]) -> Result<()> {
    let header_bytes = header.encode();
    w.write_u32::<LittleEndian>(header_bytes.len() as u32)?;
    w.write_all(&header_bytes)?;
    w.write_u32::<LittleEndian>(data.len() as u32)?;
    w.write_all(data)?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn header_round_trip() -> Result<()> {
        let mut h = RecordHeader::new();
        h.put_op(OpCode::BagHeader)
            .put_u32("chunk_count", 7)
            .put_u64("index_pos", 12345)
            .put_str("compression", "bz2");

        let encoded = h.encode();
        let decoded = parse_header(&encoded)?;

        assert_eq!(decoded.op()?, OpCode::BagHeader);
        assert_eq!(decoded.get_u32("chunk_count")?, 7);
        assert_eq!(decoded.get_u64("index_pos")?, 12345);
        assert_eq!(decoded.get_str("compression")?, "bz2");
        Ok(())
    }

    #[test]
    fn record_round_trip() -> Result<()> {
        let mut h = RecordHeader::new();
        h.put_op(OpCode::MessageData)
            .put_u32("conn", 3)
            .put_time("time", 100, 200);
        let payload = b"hello world".to_vec();

        let mut buf = Vec::new();
        write_record(&mut buf, &h, &payload)?;

        let mut cur = Cursor::new(buf);
        let rec = read_record(&mut cur)?.expect("record");
        assert_eq!(rec.header.op()?, OpCode::MessageData);
        assert_eq!(rec.header.get_u32("conn")?, 3);
        assert_eq!(
            rec.header.get_time_nanos("time")?,
            100 * 1_000_000_000 + 200
        );
        assert_eq!(rec.data, payload);

        // Clean EOF after the record.
        assert!(read_record(&mut cur)?.is_none());
        Ok(())
    }

    #[test]
    fn parse_header_rejects_missing_equals() {
        let bad = {
            let mut out = Vec::new();
            let f = b"nooequalshere";
            out.extend_from_slice(&(f.len() as u32).to_le_bytes());
            out.extend_from_slice(f);
            out
        };
        assert!(parse_header(&bad).is_err());
    }

    #[test]
    fn missing_required_field_errors() {
        let h = RecordHeader::new();
        assert!(matches!(
            h.require("op"),
            Err(BagError::Ros1MalformedRecord { .. })
        ));
    }
}
