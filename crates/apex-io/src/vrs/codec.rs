//! Record payload decompression.

use super::error::{Result, VrsError};
use super::format::Compression;
use std::io::Read;

/// Cap on up-front reservation, so a corrupt size field cannot exhaust memory.
const MAX_PREALLOC: usize = 1 << 28;

/// Decompress `src` into `out` (cleared first).
///
/// `expected` is the size announced in the record header; for stored
/// payloads it is ignored. Trailing bytes after a single compressed frame
/// are tolerated (the index record carries some).
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

/// Compress `src` for storage; returns the bytes to write.
pub fn compress(compression: Compression, src: &[u8]) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(src.to_vec()),
        Compression::Zstd => {
            zstd::bulk::compress(src, 1).map_err(|e| VrsError::Decompress(e.to_string()))
        }
        Compression::Lz4 => {
            use std::io::Write;
            let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
            enc.write_all(src)?;
            enc.finish()
                .map_err(|e| VrsError::Decompress(e.to_string()))
        }
    }
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
        }
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
