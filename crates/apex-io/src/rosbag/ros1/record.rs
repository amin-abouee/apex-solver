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
//!
//! The lengths are untrusted: a record is read into a buffer that grows with
//! the bytes actually present, never pre-sized from a length field, so a
//! corrupt 4 GB length fails as a truncated record instead of an allocation
//! abort.

use super::error::{Ros1Error, Ros1Result as Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

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
        std::str::from_utf8(bytes).map_err(|_| Ros1Error::FieldNotUtf8 {
            field: name.to_string(),
        })
    }

    /// Read a required field, returning an error if it is missing.
    pub fn require(&self, name: &str) -> Result<&[u8]> {
        self.get(name).ok_or_else(|| Ros1Error::missing_field(name))
    }

    /// Read a required field of exactly `N` bytes.
    fn require_array<const N: usize>(&self, name: &str) -> Result<[u8; N]> {
        let bytes = self.require(name)?;
        let mut array = [0u8; N];
        if bytes.len() != N {
            return Err(Ros1Error::field_size(name, N, bytes.len()));
        }
        array.copy_from_slice(bytes);
        Ok(array)
    }

    /// Read a little-endian `u8`.
    pub fn get_u8(&self, name: &str) -> Result<u8> {
        Ok(self.require_array::<1>(name)?[0])
    }

    /// Read a little-endian `u32`.
    pub fn get_u32(&self, name: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.require_array::<4>(name)?))
    }

    /// Read a little-endian `u64`.
    pub fn get_u64(&self, name: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.require_array::<8>(name)?))
    }

    /// Read a ROS1 timestamp as nanoseconds since epoch.
    pub fn get_time_nanos(&self, name: &str) -> Result<u64> {
        let bytes = self.require_array::<8>(name)?;
        let secs = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64;
        let nsecs = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as u64;
        Ok(secs.saturating_mul(1_000_000_000).saturating_add(nsecs))
    }

    /// Read the `op` field as a typed [`OpCode`].
    pub fn op(&self) -> Result<OpCode> {
        let op = self.get_u8("op")?;
        OpCode::from_u8(op).ok_or(Ros1Error::UnknownOpCode { op })
    }

    /// Serialize the header to its on-wire representation (without the outer
    /// `u32 header_len` prefix).
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        // Sort field names so writers are deterministic; readers don't care.
        let mut keys: Vec<(&String, &Vec<u8>)> = self.fields.iter().collect();
        keys.sort_by_key(|(k, _)| *k);
        for (k, v) in keys {
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

/// Split the next `<u32 len><len bytes>` header field off `bytes`.
fn next_field<'a>(bytes: &mut &'a [u8]) -> Result<(&'a [u8], &'a [u8])> {
    let (len_bytes, rest) = bytes
        .split_first_chunk::<4>()
        .ok_or(Ros1Error::HeaderTruncated {
            need: 4,
            available: bytes.len(),
        })?;
    let len = u32::from_le_bytes(*len_bytes) as usize;
    let field = rest.get(..len).ok_or(Ros1Error::HeaderTruncated {
        need: len,
        available: rest.len(),
    })?;
    *bytes = &rest[len..];
    let eq = field
        .iter()
        .position(|&b| b == b'=')
        .ok_or(Ros1Error::FieldWithoutEquals)?;
    Ok((&field[..eq], &field[eq + 1..]))
}

/// Parse a header block (without the outer length prefix) into a [`RecordHeader`].
pub fn parse_header(mut bytes: &[u8]) -> Result<RecordHeader> {
    let mut header = RecordHeader::new();
    while !bytes.is_empty() {
        let (name, value) = next_field(&mut bytes)?;
        let name = std::str::from_utf8(name).map_err(|_| Ros1Error::FieldNameNotUtf8)?;
        header.put(name.to_string(), value.to_vec());
    }
    Ok(header)
}

/// Read exactly `len` bytes of `what` into `buf` (cleared first).
///
/// The buffer grows with the bytes actually read, so a corrupt `len` costs
/// no more memory than the stream holds.
pub(crate) fn read_bounded<R: Read>(
    r: &mut R,
    len: usize,
    what: &'static str,
    buf: &mut Vec<u8>,
) -> Result<()> {
    buf.clear();
    let read = r.take(len as u64).read_to_end(buf)?;
    if read != len {
        return Err(Ros1Error::Truncated {
            what,
            need: len as u64,
            available: read as u64,
        });
    }
    Ok(())
}

/// Read a record's header length, `None` on a clean end of stream.
fn read_header_len<R: Read>(r: &mut R) -> Result<Option<usize>> {
    match r.read_u32::<LittleEndian>() {
        Ok(v) => Ok(Some(v as usize)),
        Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(Ros1Error::Io(e)),
    }
}

/// Read one record (header + data) from `r`.  Returns `Ok(None)` on clean EOF
/// before the next header length, propagates other I/O errors.
pub fn read_record<R: Read>(r: &mut R) -> Result<Option<Record>> {
    let Some(header_len) = read_header_len(r)? else {
        return Ok(None);
    };
    let mut header_bytes = Vec::new();
    read_bounded(r, header_len, "record header", &mut header_bytes)?;
    let header = parse_header(&header_bytes)?;

    let data_len = r.read_u32::<LittleEndian>()? as usize;
    let mut data = Vec::new();
    read_bounded(r, data_len, "record data", &mut data)?;
    Ok(Some(Record { header, data }))
}

/// Read one record's header and skip its data without loading it.
///
/// Returns the header and the data length; the stream is left positioned
/// after the record. Used to hop over large chunk payloads while indexing.
pub fn read_record_skip_data<R: Read + Seek>(r: &mut R) -> Result<Option<(RecordHeader, u32)>> {
    let Some(header_len) = read_header_len(r)? else {
        return Ok(None);
    };
    let mut header_bytes = Vec::new();
    read_bounded(r, header_len, "record header", &mut header_bytes)?;
    let header = parse_header(&header_bytes)?;
    let data_len = r.read_u32::<LittleEndian>()?;
    r.seek(SeekFrom::Current(i64::from(data_len)))?;
    Ok(Some((header, data_len)))
}

/// Return the payload of the record starting at `buf[0]` as a sub-slice,
/// without parsing its header (the index already supplies conn and time).
pub fn record_data_slice(buf: &[u8]) -> Result<&[u8]> {
    let truncated = |what: &'static str, need: usize| Ros1Error::Truncated {
        what,
        need: need as u64,
        available: buf.len() as u64,
    };
    let (len_bytes, rest) = buf
        .split_first_chunk::<4>()
        .ok_or_else(|| truncated("record header length", 4))?;
    let header_len = u32::from_le_bytes(*len_bytes) as usize;
    let after_header = rest
        .get(header_len..)
        .ok_or_else(|| truncated("record header", header_len.saturating_add(4)))?;
    let (len_bytes, data) = after_header
        .split_first_chunk::<4>()
        .ok_or_else(|| truncated("record data length", header_len.saturating_add(8)))?;
    let data_len = u32::from_le_bytes(*len_bytes) as usize;
    data.get(..data_len).ok_or_else(|| {
        truncated(
            "record data",
            header_len.saturating_add(8).saturating_add(data_len),
        )
    })
}

/// Walk the `name=value` fields of an encoded header without allocating.
///
/// `f` receives each `(name, value)` pair. This is the hot-path alternative
/// to [`parse_header`] for records whose few fields of interest are known.
pub fn scan_header_fields<'a>(
    mut bytes: &'a [u8],
    mut f: impl FnMut(&'a [u8], &'a [u8]),
) -> Result<()> {
    while !bytes.is_empty() {
        let (name, value) = next_field(&mut bytes)?;
        f(name, value);
    }
    Ok(())
}

/// Read a record's raw header bytes into `header` and return the length of
/// its data, leaving the stream at the start of the data. `None` on clean EOF.
pub fn read_record_prefix<R: Read>(r: &mut R, header: &mut Vec<u8>) -> Result<Option<u32>> {
    let Some(header_len) = read_header_len(r)? else {
        return Ok(None);
    };
    read_bounded(r, header_len, "record header", header)?;
    Ok(Some(r.read_u32::<LittleEndian>()?))
}

/// Write a record (header + data) to `w`.
pub fn write_record<W: Write>(w: &mut W, header: &RecordHeader, data: &[u8]) -> Result<()> {
    let header_bytes = header.encode();
    let data_len = u32::try_from(data.len()).map_err(|_| Ros1Error::TooLarge {
        what: "record data",
        len: data.len(),
    })?;
    w.write_u32::<LittleEndian>(header_bytes.len() as u32)?;
    w.write_all(&header_bytes)?;
    w.write_u32::<LittleEndian>(data_len)?;
    w.write_all(data)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn header_round_trip() -> TestResult {
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
    fn record_round_trip() -> TestResult {
        let mut h = RecordHeader::new();
        h.put_op(OpCode::MessageData)
            .put_u32("conn", 3)
            .put_time("time", 100, 200);
        let payload = b"hello world".to_vec();

        let mut buf = Vec::new();
        write_record(&mut buf, &h, &payload)?;

        let mut cur = Cursor::new(buf);
        let rec = read_record(&mut cur)?.ok_or("expected a record")?;
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
            Err(Ros1Error::MissingField { .. })
        ));
    }

    #[test]
    fn put_u8_get_u8_round_trip() -> TestResult {
        let mut h = RecordHeader::new();
        h.put_u8("flag", 0xAB);
        assert_eq!(h.get_u8("flag")?, 0xAB);
        Ok(())
    }

    #[test]
    fn get_u8_wrong_size_errors() {
        let mut h = RecordHeader::new();
        h.put_u32("big", 42);
        assert!(matches!(h.get_u8("big"), Err(Ros1Error::FieldSize { .. })));
    }

    #[test]
    fn get_str_valid_utf8() -> TestResult {
        let mut h = RecordHeader::new();
        h.put_str("name", "hello");
        assert_eq!(h.get_str("name")?, "hello");
        Ok(())
    }

    #[test]
    fn get_str_invalid_utf8_errors() {
        let mut h = RecordHeader::new();
        h.put("bad", vec![0xFF, 0xFE]);
        assert!(matches!(
            h.get_str("bad"),
            Err(Ros1Error::FieldNotUtf8 { .. })
        ));
    }

    #[test]
    fn get_u32_wrong_size_errors() {
        let mut h = RecordHeader::new();
        h.put_u8("small", 1);
        assert!(matches!(
            h.get_u32("small"),
            Err(Ros1Error::FieldSize { .. })
        ));
    }

    #[test]
    fn get_u64_wrong_size_errors() {
        let mut h = RecordHeader::new();
        h.put_u32("half", 1);
        assert!(matches!(
            h.get_u64("half"),
            Err(Ros1Error::FieldSize { .. })
        ));
    }

    #[test]
    fn get_time_nanos_round_trip() -> TestResult {
        let mut h = RecordHeader::new();
        h.put_time("stamp", 100, 500);
        assert_eq!(h.get_time_nanos("stamp")?, 100 * 1_000_000_000 + 500);
        Ok(())
    }

    #[test]
    fn get_time_nanos_wrong_size_errors() {
        let mut h = RecordHeader::new();
        h.put_u32("short_time", 1);
        assert!(matches!(
            h.get_time_nanos("short_time"),
            Err(Ros1Error::FieldSize { .. })
        ));
    }

    #[test]
    fn parse_header_truncated_field_length() {
        let bad = vec![0x01, 0x00];
        assert!(parse_header(&bad).is_err());
    }

    #[test]
    fn parse_header_truncated_field_data() {
        let mut bad = Vec::new();
        bad.extend_from_slice(&100u32.to_le_bytes());
        bad.extend_from_slice(&[0x01, 0x02]);
        assert!(parse_header(&bad).is_err());
    }

    #[test]
    fn put_op_and_op_round_trip() -> TestResult {
        let mut h = RecordHeader::new();
        h.put_op(OpCode::ChunkInfo);
        assert_eq!(h.op()?, OpCode::ChunkInfo);
        Ok(())
    }

    #[test]
    fn op_unknown_op_code_errors() {
        let mut h = RecordHeader::new();
        h.put_u8("op", 0xFF);
        assert!(matches!(h.op(), Err(Ros1Error::UnknownOpCode { op: 0xFF })));
    }

    #[test]
    fn encode_deterministic_order() {
        let mut h1 = RecordHeader::new();
        h1.put_str("b", "2").put_str("a", "1");

        let mut h2 = RecordHeader::new();
        h2.put_str("a", "1").put_str("b", "2");

        assert_eq!(h1.encode(), h2.encode());
    }

    #[test]
    fn get_optional_field_returns_none() {
        let h = RecordHeader::new();
        assert!(h.get("nonexistent").is_none());
    }

    #[test]
    fn a_huge_record_length_fails_as_truncated_without_allocating() -> TestResult {
        // header_len = u32::MAX with four bytes behind it: the old reader
        // allocated 4 GB before noticing.
        let mut bytes = u32::MAX.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0u8; 4]);
        let result = read_record(&mut Cursor::new(bytes.clone()));
        assert!(
            matches!(
                result,
                Err(Ros1Error::Truncated {
                    what: "record header",
                    available: 4,
                    ..
                })
            ),
            "got {result:?}"
        );
        let mut header = Vec::new();
        assert!(read_record_prefix(&mut Cursor::new(bytes), &mut header).is_err());
        Ok(())
    }

    #[test]
    fn a_huge_data_length_fails_as_truncated() -> TestResult {
        let mut h = RecordHeader::new();
        h.put_op(OpCode::MessageData);
        let mut bytes = Vec::new();
        write_record(&mut bytes, &h, b"")?;
        // Overwrite the data length with u32::MAX.
        let at = bytes.len() - 4;
        bytes[at..].copy_from_slice(&u32::MAX.to_le_bytes());
        let result = read_record(&mut Cursor::new(bytes));
        assert!(
            matches!(
                result,
                Err(Ros1Error::Truncated {
                    what: "record data",
                    ..
                })
            ),
            "got {result:?}"
        );
        Ok(())
    }

    #[test]
    fn record_data_slice_rejects_lengths_past_the_buffer() {
        let mut buf = 2u32.to_le_bytes().to_vec();
        buf.extend_from_slice(b"ab");
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            record_data_slice(&buf),
            Err(Ros1Error::Truncated {
                what: "record data",
                ..
            })
        ));
        assert!(record_data_slice(&u32::MAX.to_le_bytes()).is_err());
    }
}
