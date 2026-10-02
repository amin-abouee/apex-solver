//! `RecordFormat`: how a record payload splits into content blocks.
//!
//! The stream tag `RF:<Type>:<version>` holds a `+`-separated list of block
//! specs, for example `data_layout/size=92+image/jpg`. Each `data_layout`
//! block has a layout description in `DL:<Type>:<version>:<n>` (n counts the
//! `data_layout` blocks of that format).

use super::datalayout::{DataLayout, LayoutView};
use super::description::StreamTags;
use super::error::{Result, VrsError};
use super::format::RecordType;

/// Encoding of an image block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageFormat {
    Raw,
    Jpg,
    Png,
    Video,
    Jxl,
    /// Anything else, verbatim.
    Other(String),
}

/// Parameters of an image block spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSpec {
    /// Encoding.
    pub format: ImageFormat,
    /// Width in pixels, when the spec says.
    pub width: Option<u32>,
    /// Height in pixels, when the spec says.
    pub height: Option<u32>,
    /// Pixel format name (`grey8`, `rgb8`, ...), for raw images.
    pub pixel: Option<String>,
    /// Explicit block size in bytes.
    pub size: Option<usize>,
}

/// One block of a record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockSpec {
    /// A [`DataLayout`]; `size` is the fixed size announced by the spec.
    DataLayout { size: Option<usize> },
    /// Image pixels.
    Image(ImageSpec),
    /// Audio samples (spec kept verbatim).
    Audio { spec: String, size: Option<usize> },
    /// Opaque application bytes.
    Custom { size: Option<usize> },
    /// No data.
    Empty,
}

/// A parsed `RF:` tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordFormat {
    blocks: Vec<BlockSpec>,
    text: String,
}

fn size_param(parts: &[&str]) -> Option<usize> {
    parts
        .iter()
        .find_map(|p| p.strip_prefix("size="))
        .and_then(|v| v.parse().ok())
}

impl RecordFormat {
    /// Parse the text of an `RF:` tag.
    pub fn parse(text: &str) -> Result<Self> {
        let mut blocks = Vec::new();
        for block in text.split('+') {
            let parts: Vec<&str> = block.split('/').collect();
            let spec = match parts[0] {
                "data_layout" | "datalayout" => BlockSpec::DataLayout {
                    size: size_param(&parts[1..]),
                },
                "image" => {
                    let format = match parts.get(1).copied() {
                        Some("raw") => ImageFormat::Raw,
                        Some("jpg") => ImageFormat::Jpg,
                        Some("png") => ImageFormat::Png,
                        Some("video") => ImageFormat::Video,
                        Some("jxl") => ImageFormat::Jxl,
                        Some(p) if p.contains('=') || p.contains('x') => ImageFormat::Raw,
                        Some(other) => ImageFormat::Other(other.to_string()),
                        None => ImageFormat::Raw,
                    };
                    let dims = parts.iter().skip(1).find_map(|p| {
                        let (w, h) = p.split_once('x')?;
                        Some((w.parse().ok()?, h.parse().ok()?))
                    });
                    BlockSpec::Image(ImageSpec {
                        format,
                        width: dims.map(|d| d.0),
                        height: dims.map(|d| d.1),
                        pixel: parts
                            .iter()
                            .find_map(|p| p.strip_prefix("pixel="))
                            .map(str::to_string),
                        size: size_param(&parts[1..]),
                    })
                }
                "audio" => BlockSpec::Audio {
                    spec: block.to_string(),
                    size: size_param(&parts[1..]),
                },
                "custom" => BlockSpec::Custom {
                    size: size_param(&parts[1..]),
                },
                "empty" => BlockSpec::Empty,
                other => {
                    return Err(VrsError::Layout(format!("unknown content block '{other}'")));
                }
            };
            blocks.push(spec);
        }
        Ok(Self {
            blocks,
            text: text.to_string(),
        })
    }

    /// The blocks in order.
    pub fn blocks(&self) -> &[BlockSpec] {
        &self.blocks
    }

    /// The original text.
    pub fn as_str(&self) -> &str {
        &self.text
    }
}

/// One block of a decoded record payload.
#[derive(Debug, Clone, Copy)]
pub enum Block<'a> {
    /// Typed header fields.
    Layout(LayoutView<'a>),
    /// Image bytes (encoded per the spec; JPEG is returned untouched).
    Image {
        /// Block spec.
        spec: &'a ImageSpec,
        /// Image bytes.
        bytes: &'a [u8],
    },
    /// Audio bytes.
    Audio(&'a [u8]),
    /// Opaque bytes.
    Custom(&'a [u8]),
    /// Empty block.
    Empty,
}

/// A record format together with the layouts of its `data_layout` blocks.
#[derive(Debug, Clone)]
pub struct RecordLayout {
    format: RecordFormat,
    layouts: Vec<DataLayout>,
}

impl RecordLayout {
    /// Build from a stream's tags for `(record_type, version)`; `None` if the
    /// stream has no `RF:` tag for it.
    pub fn from_tags(
        tags: &StreamTags,
        record_type: RecordType,
        version: u32,
    ) -> Option<Result<Self>> {
        let name = record_type.name();
        let rf = tags.vrs.get(&format!("RF:{name}:{version}"))?;
        Some(Self::build(tags, name, version, rf))
    }

    fn build(tags: &StreamTags, name: &str, version: u32, rf: &str) -> Result<Self> {
        let format = RecordFormat::parse(rf)?;
        let wanted = format
            .blocks()
            .iter()
            .filter(|b| matches!(b, BlockSpec::DataLayout { .. }))
            .count();
        let mut layouts = Vec::with_capacity(wanted);
        for i in 0..wanted {
            let json = tags
                .vrs
                .get(&format!("DL:{name}:{version}:{i}"))
                .ok_or_else(|| VrsError::Layout(format!("missing tag DL:{name}:{version}:{i}")))?;
            layouts.push(DataLayout::from_json(json)?);
        }
        Ok(Self { format, layouts })
    }

    /// The parsed record format.
    pub fn format(&self) -> &RecordFormat {
        &self.format
    }

    /// The first `data_layout` block's layout, if any.
    pub fn first_layout(&self) -> Option<&DataLayout> {
        self.layouts.first()
    }

    /// Split `payload` into its blocks.
    pub fn split<'a>(&'a self, payload: &'a [u8]) -> Result<Vec<Block<'a>>> {
        let mut out = Vec::with_capacity(self.format.blocks.len());
        self.split_into(payload, &mut out)?;
        Ok(out)
    }

    /// Split `payload` into `blocks` (cleared first), so callers in hot loops
    /// can reuse the same vector for every record.
    pub fn split_into<'a>(&'a self, payload: &'a [u8], blocks: &mut Vec<Block<'a>>) -> Result<()> {
        blocks.clear();
        let out = blocks;
        let mut pos = 0usize;
        let mut layout_i = 0usize;
        let last = self.format.blocks.len().saturating_sub(1);

        for (bi, spec) in self.format.blocks.iter().enumerate() {
            let rest = payload
                .get(pos..)
                .ok_or_else(|| VrsError::Layout("record shorter than its format".into()))?;
            match spec {
                BlockSpec::DataLayout { size } => {
                    let layout = &self.layouts[layout_i];
                    layout_i += 1;
                    let view = layout.bind(rest)?;
                    if let Some(sz) = size
                        && layout.var_count() == 0
                        && *sz != layout.fixed_size()
                    {
                        return Err(VrsError::Layout(format!(
                            "format says data_layout/size={sz} but the layout is {} bytes",
                            layout.fixed_size()
                        )));
                    }
                    pos += view.total_len();
                    out.push(Block::Layout(view));
                }
                BlockSpec::Empty => out.push(Block::Empty),
                BlockSpec::Image(_) | BlockSpec::Audio { .. } | BlockSpec::Custom { .. } => {
                    let explicit = match spec {
                        BlockSpec::Image(i) => i.size,
                        BlockSpec::Audio { size, .. } | BlockSpec::Custom { size } => *size,
                        _ => None,
                    };
                    let len = match explicit {
                        Some(n) => n,
                        None if bi == last => rest.len(),
                        None => {
                            return Err(VrsError::unsupported(
                                "content block without a size that is not the last block",
                            ));
                        }
                    };
                    let bytes = rest
                        .get(..len)
                        .ok_or_else(|| VrsError::Layout("content block exceeds record".into()))?;
                    pos += len;
                    out.push(match spec {
                        BlockSpec::Image(i) => Block::Image { spec: i, bytes },
                        BlockSpec::Audio { .. } => Block::Audio(bytes),
                        _ => Block::Custom(bytes),
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_aria_formats() {
        let f = RecordFormat::parse("data_layout/size=92+image/jpg").unwrap();
        assert_eq!(f.blocks().len(), 2);
        assert_eq!(f.blocks()[0], BlockSpec::DataLayout { size: Some(92) });
        assert!(matches!(&f.blocks()[1], BlockSpec::Image(i) if i.format == ImageFormat::Jpg));
        assert_eq!(f.as_str(), "data_layout/size=92+image/jpg");

        let f = RecordFormat::parse("data_layout+custom").unwrap();
        assert_eq!(f.blocks()[1], BlockSpec::Custom { size: None });
        assert_eq!(
            RecordFormat::parse("data_layout").unwrap().blocks(),
            [BlockSpec::DataLayout { size: None }]
        );
    }

    #[test]
    fn parses_raw_image_parameters() {
        let f = RecordFormat::parse("image/raw/640x480/pixel=grey8").unwrap();
        match &f.blocks()[0] {
            BlockSpec::Image(i) => {
                assert_eq!(i.format, ImageFormat::Raw);
                assert_eq!((i.width, i.height), (Some(640), Some(480)));
                assert_eq!(i.pixel.as_deref(), Some("grey8"));
            }
            other => panic!("{other:?}"),
        }
        assert!(RecordFormat::parse("audio/pcm/uint24be/rate=32000/channels=1").is_ok());
        assert!(RecordFormat::parse("bogus/thing").is_err());
    }

    fn tags(rf: &str) -> StreamTags {
        let mut t = StreamTags::default();
        t.vrs.insert("RF:Data:1".into(), rf.into());
        t.vrs.insert(
            "DL:Data:1:0".into(),
            r#"{"data_layout":[{"name":"n","type":"DataPieceValue<uint32_t>","offset":0}]}"#.into(),
        );
        t
    }

    #[test]
    fn splits_layout_plus_image_and_layout_plus_custom() {
        let t = tags("data_layout/size=4+image/jpg");
        let rl = RecordLayout::from_tags(&t, RecordType::Data, 1)
            .unwrap()
            .unwrap();
        let mut payload = 7u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&[0xFF, 0xD8, 1, 2, 3, 0xFF, 0xD9]);
        let blocks = rl.split(&payload).unwrap();
        match (&blocks[0], &blocks[1]) {
            (Block::Layout(v), Block::Image { bytes, spec }) => {
                assert_eq!(v.get_u32("n").unwrap(), 7);
                assert_eq!(*bytes, &[0xFF, 0xD8, 1, 2, 3, 0xFF, 0xD9]);
                assert_eq!(spec.format, ImageFormat::Jpg);
            }
            other => panic!("{other:?}"),
        }

        let t = tags("data_layout+custom");
        let rl = RecordLayout::from_tags(&t, RecordType::Data, 1)
            .unwrap()
            .unwrap();
        let blocks = rl.split(&[1, 0, 0, 0, 9, 9]).unwrap();
        assert!(matches!(blocks[1], Block::Custom(b) if b == [9, 9]));
    }

    #[test]
    fn mismatches_are_errors() {
        // announced size disagrees with the layout
        let t = tags("data_layout/size=8");
        let rl = RecordLayout::from_tags(&t, RecordType::Data, 1)
            .unwrap()
            .unwrap();
        assert!(rl.split(&[0; 8]).is_err());
        // payload shorter than the layout
        let t = tags("data_layout");
        let rl = RecordLayout::from_tags(&t, RecordType::Data, 1)
            .unwrap()
            .unwrap();
        assert!(rl.split(&[0; 2]).is_err());
        // no RF tag for that version
        assert!(RecordLayout::from_tags(&t, RecordType::Data, 9).is_none());
        // RF without its DL tag
        let mut t2 = StreamTags::default();
        t2.vrs.insert("RF:Data:1".into(), "data_layout".into());
        assert!(
            RecordLayout::from_tags(&t2, RecordType::Data, 1)
                .unwrap()
                .is_err()
        );
    }
}
