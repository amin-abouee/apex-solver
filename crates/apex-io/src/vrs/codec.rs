//! Record payload compression and decompression.
//!
//! [`Decompressor`] and [`Compressor`] own reusable zstd contexts: creating a
//! fresh `DCtx`/`CCtx` per record dominated the cost of pulling small records
//! (IMU streams run at about 1 kHz). The free functions [`decompress`] and
//! [`compress`] are one-shot equivalents for rare, one-off uses such as
//! parsing the index record.

use super::error::{Result, VrsError};
use super::format::Compression;
use std::io::Read;

/// Cap on up-front reservation, so a corrupt size field cannot exhaust memory.
const MAX_PREALLOC: usize = 1 << 28;

/// Reusable decompression contexts (zstd `DCtx`).
pub struct Decompressor {
    zstd: Option<zstd::bulk::Decompressor<'static>>,
}

impl Default for Decompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Decompressor {
    pub fn new() -> Self {
        Self {
            zstd: zstd::bulk::Decompressor::new().ok(),
        }
    }

    /// Decompress `src` into `out` (overwritten).
    ///
    /// `expected` is the size announced in the record header; for stored
    /// payloads it is ignored. Unlike the one-shot [`decompress`], this hot
    /// path is strict: no trailing data after the compressed frame.
    pub fn decompress(
        &mut self,
        compression: Compression,
        src: &[u8],
        expected: usize,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        out.clear();
        match compression {
            Compression::None => out.extend_from_slice(src),
            Compression::Zstd => {
                out.reserve(expected.min(MAX_PREALLOC));
                match self.zstd.as_mut() {
                    // The DCtx writes from index 0 and sets the final length.
                    Some(ctx) => {
                        let n = ctx
                            .decompress_to_buffer(src, out)
                            .map_err(|e| VrsError::Decompress(e.to_string()))?;
                        out.truncate(n);
                    }
                    None => {
                        *out = zstd::bulk::decompress(src, expected.min(MAX_PREALLOC))
                            .map_err(|e| VrsError::Decompress(e.to_string()))?;
                    }
                }
            }
            Compression::Lz4 => {
                out.reserve(expected.min(MAX_PREALLOC));
                lz4_flex::frame::FrameDecoder::new(src)
                    .read_to_end(out)
                    .map_err(|e| VrsError::Decompress(e.to_string()))?;
            }
        }
        if compression != Compression::None && out.len() != expected {
            return Err(VrsError::corrupt(format!(
                "payload decompressed to {} bytes, header says {expected}",
                out.len()
            )));
        }
        Ok(())
    }
}

/// Reusable compression context (zstd `CCtx`).
pub struct Compressor {
    zstd: Option<zstd::bulk::Compressor<'static>>,
}

impl Default for Compressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor {
    pub fn new() -> Self {
        Self {
            zstd: zstd::bulk::Compressor::new(1).ok(),
        }
    }

    /// Compress `src` for storage; returns the bytes to write.
    pub fn compress(&mut self, compression: Compression, src: &[u8]) -> Result<Vec<u8>> {
        match compression {
            Compression::None => Ok(src.to_vec()),
            Compression::Zstd => match self.zstd.as_mut() {
                Some(ctx) => {
                    let mut out =
                        Vec::with_capacity(zstd::zstd_safe::compress_bound(src.len()));
                    let n = ctx
                        .compress_to_buffer(src, &mut out)
                        .map_err(|e| VrsError::Decompress(e.to_string()))?;
                    out.truncate(n);
                    Ok(out)
                }
                None => zstd::bulk::compress(src, 1).map_err(|e| VrsError::Decompress(e.to_string())),
            },
            Compression::Lz4 => {
                use std::io::Write;
                let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
                enc.write_all(src)?;
                enc.finish()
                    .map_err(|e| VrsError::Decompress(e.to_string()))
            }
        }
    }
}

/// One-shot decompression with a fresh, tolerant decoder: trailing bytes
/// after a single frame are ignored. Used for rare records that carry extra
/// data after the compressed payload (Meta's index record does); hot loops
/// should use [`Decompressor`], which is strict and much faster.
pub fn decompress(
    compression: Compression,
    src: &[u8],
    expected: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    out.clear();
    match compression {
        Compression::None => out.extend_from_slice(src),
        Compression::Zstd => {
            out.reserve(expected.min(MAX_PREALLOC));
            let mut decoder = zstd::stream::read::Decoder::new(src)
                .map_err(|e| VrsError::Decompress(e.to_string()))?
                .single_frame();
            decoder
                .read_to_end(out)
                .map_err(|e| VrsError::Decompress(e.to_string()))?;
        }
        Compression::Lz4 => {
            out.reserve(expected.min(MAX_PREALLOC));
            lz4_flex::frame::FrameDecoder::new(src)
                .read_to_end(out)
                .map_err(|e| VrsError::Decompress(e.to_string()))?;
        }
    }
    if compression != Compression::None && out.len() != expected {
        return Err(VrsError::corrupt(format!(
            "payload decompressed to {} bytes, header says {expected}",
            out.len()
        )));
    }
    Ok(())
}

/// One-shot compression with a fresh context. Hot loops should hold a
/// [`Compressor`] instead.
pub fn compress(compression: Compression, src: &[u8]) -> Result<Vec<u8>> {
    Compressor::new().compress(compression, src)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn every_codec_round_trips() {
        let data: Vec<u8> = (0..50_000u32)
            .flat_map(|i| (i % 97).to_le_bytes())
            .collect();
        for c in [Compression::None, Compression::Zstd, Compression::Lz4] {
            let packed = compress(c, &data).unwrap();
            let mut out = Vec::new();
            decompress(c, &packed, data.len(), &mut out).unwrap();
            assert_eq!(out, data, "{c:?}");

            // Reused contexts must behave identically, across several records.
            let mut de = Decompressor::new();
            let mut co = Compressor::new();
            for _ in 0..3 {
                let packed = co.compress(c, &data).unwrap();
                out.clear();
                de.decompress(c, &packed, data.len(), &mut out).unwrap();
                assert_eq!(out, data, "{c:?} (reused)");
            }
        }
    }

    #[test]
    fn decompress_out_is_overwritten_not_appended() {
        let data = vec![9u8; 1000];
        let packed = compress(Compression::Zstd, &data).unwrap();
        let mut de = Decompressor::new();
        let mut out = vec![0xEE; 500];
        de.decompress(Compression::Zstd, &packed, data.len(), &mut out)
            .unwrap();
        assert_eq!(out, data);
        // A second, smaller record also fully replaces the buffer.
        let small = compress(Compression::Zstd, &[1u8; 10]).unwrap();
        de.decompress(Compression::Zstd, &small, 10, &mut out).unwrap();
        assert_eq!(out, vec![1u8; 10]);
    }

    #[test]
    fn trailing_bytes_after_a_zstd_frame_are_ignored() {
        let data = vec![7u8; 1000];
        let mut packed = compress(Compression::Zstd, &data).unwrap();
        packed.extend_from_slice(&[0xAB; 100]);
        let mut out = Vec::new();
        decompress(Compression::Zstd, &packed, data.len(), &mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn size_mismatch_and_garbage_are_errors() {
        let packed = compress(Compression::Zstd, &[1u8; 100]).unwrap();
        let mut out = Vec::new();
        assert!(decompress(Compression::Zstd, &packed, 99, &mut out).is_err());
        assert!(decompress(Compression::Zstd, &[1, 2, 3, 4], 4, &mut out).is_err());
        assert!(decompress(Compression::Lz4, &[1, 2, 3, 4], 4, &mut out).is_err());
    }
}
