//! ROS Message serialization format deserializer (used by ROS1 bags).
//!
//! Differences from CDR (ROS2):
//! - no 4-byte encapsulation header,
//! - no per-field alignment padding,
//! - little-endian throughout,
//! - strings: `u32 length + bytes` (no NUL terminator),
//! - time/duration: `u32 sec + u32 nsec`,
//! - variable arrays: `u32 length + elements`,
//! - fixed arrays: elements only.

use crate::rosbag::error::{BagError, Result};

/// Cursor-based deserializer for the ROS1 wire format.
pub struct Ros1Deserializer<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Ros1Deserializer<'a> {
    /// Wrap a buffer.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Current cursor position.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Number of bytes remaining.
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(BagError::invalid_message_data(format!(
                "ros1: needed {n} bytes at offset {}, have {}",
                self.pos,
                self.remaining()
            )));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// Read a `u8`.
    pub fn read_u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// Read a `bool` (1 byte, 0 = false).
    pub fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    /// Read an `i8`.
    pub fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    /// Read a little-endian `u16`.
    pub fn read_u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    /// Read a little-endian `i16`.
    pub fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    /// Read a little-endian `u32`.
    pub fn read_u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a little-endian `i32`.
    pub fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    /// Read a little-endian `u64`.
    pub fn read_u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a little-endian `i64`.
    pub fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    /// Read a little-endian `f32`.
    pub fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    /// Read a little-endian `f64`.
    pub fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    /// Read a length-prefixed UTF-8 string.
    pub fn read_string(&mut self) -> Result<String> {
        let len = self.read_u32()? as usize;
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Read a ROS `time`/`duration` (sec + nsec) as nanoseconds.
    pub fn read_time_nanos(&mut self) -> Result<u64> {
        let secs = self.read_u32()? as u64;
        let nsecs = self.read_u32()? as u64;
        Ok(secs.saturating_mul(1_000_000_000).saturating_add(nsecs))
    }

    /// Read a variable-length sequence of `T` via a per-element callback.
    pub fn read_seq<T, F>(&mut self, mut read_elem: F) -> Result<Vec<T>>
    where
        F: FnMut(&mut Self) -> Result<T>,
    {
        let len = self.read_u32()? as usize;
        let mut out = Vec::with_capacity(len.min(1 << 16));
        for _ in 0..len {
            out.push(read_elem(self)?);
        }
        Ok(out)
    }

    /// Read a fixed-length sequence of `T`.
    pub fn read_array<T, F>(&mut self, len: usize, mut read_elem: F) -> Result<Vec<T>>
    where
        F: FnMut(&mut Self) -> Result<T>,
    {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(read_elem(self)?);
        }
        Ok(out)
    }

    /// Read a raw byte buffer (variable length).
    pub fn read_bytes(&mut self) -> Result<Vec<u8>> {
        let len = self.read_u32()? as usize;
        Ok(self.take(len)?.to_vec())
    }
}

/// Symmetric serializer used by [`super::messages`] to write standard messages.
pub struct Ros1Serializer {
    buf: Vec<u8>,
}

impl Ros1Serializer {
    /// Create a new serializer.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Consume the serializer and return the encoded bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Borrow the encoded bytes without consuming.
    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    /// Append a `u8`.
    pub fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    /// Append a `bool` as a single byte.
    pub fn write_bool(&mut self, v: bool) {
        self.write_u8(v as u8);
    }

    /// Append an `i8`.
    pub fn write_i8(&mut self, v: i8) {
        self.write_u8(v as u8);
    }

    /// Append a little-endian `u16`.
    pub fn write_u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a little-endian `i16`.
    pub fn write_i16(&mut self, v: i16) {
        self.write_u16(v as u16);
    }

    /// Append a little-endian `u32`.
    pub fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a little-endian `i32`.
    pub fn write_i32(&mut self, v: i32) {
        self.write_u32(v as u32);
    }

    /// Append a little-endian `u64`.
    pub fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a little-endian `i64`.
    pub fn write_i64(&mut self, v: i64) {
        self.write_u64(v as u64);
    }

    /// Append a little-endian `f32`.
    pub fn write_f32(&mut self, v: f32) {
        self.write_u32(v.to_bits());
    }

    /// Append a little-endian `f64`.
    pub fn write_f64(&mut self, v: f64) {
        self.write_u64(v.to_bits());
    }

    /// Append a length-prefixed UTF-8 string.
    pub fn write_string(&mut self, s: &str) {
        self.write_u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }

    /// Append a ROS time/duration from nanoseconds.
    pub fn write_time_nanos(&mut self, ns: u64) {
        let secs = (ns / 1_000_000_000) as u32;
        let nsecs = (ns % 1_000_000_000) as u32;
        self.write_u32(secs);
        self.write_u32(nsecs);
    }

    /// Append a variable-length sequence using a per-element callback.
    pub fn write_seq<T, F>(&mut self, items: &[T], mut write_elem: F)
    where
        F: FnMut(&mut Self, &T),
    {
        self.write_u32(items.len() as u32);
        for it in items {
            write_elem(self, it);
        }
    }

    /// Append a raw byte buffer (variable length).
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.write_u32(bytes.len() as u32);
        self.buf.extend_from_slice(bytes);
    }
}

impl Default for Ros1Serializer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn primitives_round_trip() -> Result<()> {
        let mut s = Ros1Serializer::new();
        s.write_u8(7);
        s.write_i32(-1234);
        s.write_f64(3.14159265);
        s.write_string("hello");
        s.write_time_nanos(1_500_000_007);
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        assert_eq!(d.read_u8()?, 7);
        assert_eq!(d.read_i32()?, -1234);
        assert!((d.read_f64()? - 3.14159265).abs() < 1e-12);
        assert_eq!(d.read_string()?, "hello");
        assert_eq!(d.read_time_nanos()?, 1_500_000_007);
        assert_eq!(d.remaining(), 0);
        Ok(())
    }

    #[test]
    fn seq_round_trip() -> Result<()> {
        let mut s = Ros1Serializer::new();
        s.write_seq(&[1.5f64, 2.5, 3.5], |s, v| s.write_f64(*v));
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        let v = d.read_seq(|d| d.read_f64())?;
        assert_eq!(v, vec![1.5, 2.5, 3.5]);
        Ok(())
    }

    #[test]
    fn truncated_buffer_error() {
        let mut d = Ros1Deserializer::new(&[0u8; 2]);
        assert!(d.read_u32().is_err());
    }

    #[test]
    fn position_accessor() {
        let d = Ros1Deserializer::new(&[0u8; 8]);
        assert_eq!(d.position(), 0);
        assert_eq!(d.remaining(), 8);
    }

    #[test]
    fn position_advances_after_read() -> Result<()> {
        let mut d = Ros1Deserializer::new(&[0u8; 8]);
        d.read_u32()?;
        assert_eq!(d.position(), 4);
        assert_eq!(d.remaining(), 4);
        Ok(())
    }

    #[test]
    fn bool_round_trip() -> Result<()> {
        let mut s = Ros1Serializer::new();
        s.write_bool(true);
        s.write_bool(false);
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        assert!(d.read_bool()?);
        assert!(!d.read_bool()?);
        Ok(())
    }

    #[test]
    fn i8_round_trip() -> Result<()> {
        let mut s = Ros1Serializer::new();
        s.write_i8(-42);
        s.write_i8(127);
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        assert_eq!(d.read_i8()?, -42);
        assert_eq!(d.read_i8()?, 127);
        Ok(())
    }

    #[test]
    fn u16_round_trip() -> Result<()> {
        let mut s = Ros1Serializer::new();
        s.write_u16(0);
        s.write_u16(65535);
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        assert_eq!(d.read_u16()?, 0);
        assert_eq!(d.read_u16()?, 65535);
        Ok(())
    }

    #[test]
    fn i16_round_trip() -> Result<()> {
        let mut s = Ros1Serializer::new();
        s.write_i16(-32768);
        s.write_i16(32767);
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        assert_eq!(d.read_i16()?, -32768);
        assert_eq!(d.read_i16()?, 32767);
        Ok(())
    }

    #[test]
    fn i64_round_trip() -> Result<()> {
        let mut s = Ros1Serializer::new();
        s.write_i64(i64::MIN);
        s.write_i64(i64::MAX);
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        assert_eq!(d.read_i64()?, i64::MIN);
        assert_eq!(d.read_i64()?, i64::MAX);
        Ok(())
    }

    #[test]
    fn f32_round_trip() -> Result<()> {
        let mut s = Ros1Serializer::new();
        s.write_f32(std::f32::consts::PI);
        s.write_f32(-0.0);
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        assert!((d.read_f32()? - std::f32::consts::PI).abs() < 1e-6);
        assert_eq!(d.read_f32()?, -0.0);
        Ok(())
    }

    #[test]
    fn read_array_fixed_length() -> Result<()> {
        let mut s = Ros1Serializer::new();
        s.write_f64(1.0);
        s.write_f64(2.0);
        s.write_f64(3.0);
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        let arr = d.read_array(3, |d| d.read_f64())?;
        assert_eq!(arr, vec![1.0, 2.0, 3.0]);
        assert_eq!(d.remaining(), 0);
        Ok(())
    }

    #[test]
    fn read_array_empty() -> Result<()> {
        let d_bytes: &[u8] = &[];
        let mut d = Ros1Deserializer::new(d_bytes);
        let arr: Vec<f64> = d.read_array(0, |d| d.read_f64())?;
        assert!(arr.is_empty());
        Ok(())
    }

    #[test]
    fn bytes_round_trip() -> Result<()> {
        let mut s = Ros1Serializer::new();
        s.write_bytes(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        let result = d.read_bytes()?;
        assert_eq!(result, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        Ok(())
    }

    #[test]
    fn bytes_empty_round_trip() -> Result<()> {
        let mut s = Ros1Serializer::new();
        s.write_bytes(&[]);
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        let result = d.read_bytes()?;
        assert!(result.is_empty());
        Ok(())
    }

    #[test]
    fn serializer_default_matches_new() {
        let a = Ros1Serializer::new();
        let b = Ros1Serializer::default();
        assert_eq!(a.into_bytes(), b.into_bytes());
    }

    #[test]
    fn serializer_as_slice() {
        let mut s = Ros1Serializer::new();
        s.write_u8(42);
        assert_eq!(s.as_slice(), &[42]);
    }

    #[test]
    fn read_bool_truncated_error() {
        let mut d = Ros1Deserializer::new(&[]);
        assert!(d.read_bool().is_err());
    }

    #[test]
    fn read_string_truncated_error() {
        let mut s = Ros1Serializer::new();
        s.write_u32(100);
        let bytes = s.into_bytes();
        let mut d = Ros1Deserializer::new(&bytes);
        assert!(d.read_string().is_err());
    }

    #[test]
    fn read_bytes_truncated_error() {
        let mut s = Ros1Serializer::new();
        s.write_u32(100);
        let bytes = s.into_bytes();
        let mut d = Ros1Deserializer::new(&bytes);
        assert!(d.read_bytes().is_err());
    }

    #[test]
    fn read_seq_empty() -> Result<()> {
        let mut s = Ros1Serializer::new();
        s.write_u32(0);
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        let v: Vec<u32> = d.read_seq(|d| d.read_u32())?;
        assert!(v.is_empty());
        Ok(())
    }

    #[test]
    fn write_seq_empty() -> Result<()> {
        let mut s = Ros1Serializer::new();
        let empty: Vec<f64> = vec![];
        s.write_seq(&empty, |s, v| s.write_f64(*v));
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        let v = d.read_seq(|d| d.read_f64())?;
        assert!(v.is_empty());
        Ok(())
    }

    #[test]
    fn all_primitives_combined_round_trip() -> Result<()> {
        let mut s = Ros1Serializer::new();
        s.write_bool(true);
        s.write_i8(-1);
        s.write_u16(1000);
        s.write_i16(-1000);
        s.write_u32(123456);
        s.write_i32(-123456);
        s.write_u64(u64::MAX);
        s.write_i64(i64::MIN);
        s.write_f32(2.5);
        s.write_f64(3.14);
        s.write_string("test");
        s.write_bytes(&[1, 2, 3]);
        let bytes = s.into_bytes();

        let mut d = Ros1Deserializer::new(&bytes);
        assert!(d.read_bool()?);
        assert_eq!(d.read_i8()?, -1);
        assert_eq!(d.read_u16()?, 1000);
        assert_eq!(d.read_i16()?, -1000);
        assert_eq!(d.read_u32()?, 123456);
        assert_eq!(d.read_i32()?, -123456);
        assert_eq!(d.read_u64()?, u64::MAX);
        assert_eq!(d.read_i64()?, i64::MIN);
        assert!((d.read_f32()? - 2.5).abs() < 1e-6);
        assert!((d.read_f64()? - 3.14).abs() < 1e-12);
        assert_eq!(d.read_string()?, "test");
        assert_eq!(d.read_bytes()?, vec![1, 2, 3]);
        assert_eq!(d.remaining(), 0);
        Ok(())
    }
}
