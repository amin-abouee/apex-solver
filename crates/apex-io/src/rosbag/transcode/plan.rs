//! Schema-compiled ROS1 <-> CDR transcoding.
//!
//! A message definition is compiled once into a flat list of [`Op`]s; each
//! message is then converted by walking the ops over the input bytes and
//! appending to a reusable output buffer. There are no typed message
//! structs: runs of fixed-size fields and primitive arrays (image and cloud
//! payloads) are copied as byte slices.

use super::msgdef::{ArrayKind, Dialect, FieldType, Registry};
use crate::rosbag::error::{BagError, Result};

/// Maximum nesting depth while compiling (guards recursive definitions).
const MAX_DEPTH: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Op {
    /// `count` consecutive scalars of `size` bytes (CDR alignment = `size`).
    Scalars { size: usize, count: usize },
    /// A string.
    Str,
    /// Read and discard one scalar (e.g. the ROS1 header `seq`).
    Drop(usize),
    /// Write one zero scalar (e.g. a synthesised ROS1 header `seq`).
    Fill(usize),
    /// A fixed-size array or a length-prefixed sequence.
    Array {
        elem: Vec<Op>,
        fixed: Option<usize>,
        /// Element size when `elem` is a single scalar (bulk-copy path).
        scalar: Option<usize>,
    },
}

/// A message converter between two wire dialects.
#[derive(Debug, Clone)]
pub struct Transcoder {
    ops: Vec<Op>,
    from: Dialect,
    to: Dialect,
}

fn push_op(ops: &mut Vec<Op>, op: Op) {
    if let (Some(Op::Scalars { size, count }), Op::Scalars { size: s, count: c }) =
        (ops.last_mut(), &op)
        && size == s
    {
        *count += c;
        return;
    }
    ops.push(op);
}

fn compile_fields(
    reg: &Registry,
    name: &str,
    from: Dialect,
    to: Dialect,
    depth: usize,
) -> Result<Vec<Op>> {
    if depth > MAX_DEPTH {
        return Err(BagError::schema_validation(format!(
            "message '{name}' nests deeper than {MAX_DEPTH} levels"
        )));
    }
    let def = reg.get(name).ok_or_else(|| {
        BagError::schema_validation(format!("message type '{name}' is not defined"))
    })?;
    let mut ops = Vec::new();
    for field in &def.fields {
        let one: Vec<Op> = match &field.ty {
            FieldType::Prim(p) => vec![Op::Scalars {
                size: p.size(),
                count: 1,
            }],
            FieldType::String => vec![Op::Str],
            FieldType::Time | FieldType::Duration => vec![Op::Scalars { size: 4, count: 2 }],
            FieldType::Header => {
                let stamp = Op::Scalars { size: 4, count: 2 };
                match (from, to) {
                    (Dialect::Ros1, Dialect::Ros2) => vec![Op::Drop(4), stamp, Op::Str],
                    (Dialect::Ros2, Dialect::Ros1) => vec![Op::Fill(4), stamp, Op::Str],
                    (Dialect::Ros1, Dialect::Ros1) => {
                        vec![Op::Scalars { size: 4, count: 1 }, stamp, Op::Str]
                    }
                    (Dialect::Ros2, Dialect::Ros2) => vec![stamp, Op::Str],
                }
            }
            FieldType::Msg(dep) => compile_fields(reg, dep, from, to, depth + 1)?,
        };
        match field.array {
            ArrayKind::Scalar => {
                for op in one {
                    push_op(&mut ops, op);
                }
            }
            ArrayKind::Fixed(n) => {
                let n = n as usize;
                if n == 0 {
                    continue;
                }
                match one.as_slice() {
                    [Op::Scalars { size, count }] => push_op(
                        &mut ops,
                        Op::Scalars {
                            size: *size,
                            count: count * n,
                        },
                    ),
                    _ => ops.push(Op::Array {
                        elem: one,
                        fixed: Some(n),
                        scalar: None,
                    }),
                }
            }
            ArrayKind::Dynamic => {
                let scalar = match one.as_slice() {
                    [Op::Scalars { size, count: 1 }] => Some(*size),
                    _ => None,
                };
                ops.push(Op::Array {
                    elem: one,
                    fixed: None,
                    scalar,
                });
            }
        }
    }
    Ok(ops)
}

struct Input<'a> {
    buf: &'a [u8],
    pos: usize,
    origin: usize,
    aligned: bool,
}

impl<'a> Input<'a> {
    fn align(&mut self, size: usize) {
        if self.aligned && size > 1 {
            let rem = (self.pos - self.origin) % size;
            if rem != 0 {
                self.pos += size - rem;
            }
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| BagError::invalid_message_data("message truncated"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        self.align(4);
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

struct Output<'a> {
    buf: &'a mut Vec<u8>,
    origin: usize,
    aligned: bool,
}

impl Output<'_> {
    fn align(&mut self, size: usize) {
        if self.aligned && size > 1 {
            let rem = (self.buf.len() - self.origin) % size;
            if rem != 0 {
                self.buf.resize(self.buf.len() + size - rem, 0);
            }
        }
    }

    fn u32(&mut self, v: u32) {
        self.align(4);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn run(ops: &[Op], inp: &mut Input<'_>, out: &mut Output<'_>) -> Result<()> {
    for op in ops {
        match op {
            Op::Scalars { size, count } => {
                inp.align(*size);
                let bytes = inp.take(size * count)?;
                out.align(*size);
                out.buf.extend_from_slice(bytes);
            }
            Op::Str => {
                let len = inp.u32()? as usize;
                let raw = inp.take(len)?;
                let text = if inp.aligned {
                    // CDR: length counts the terminating NUL.
                    match raw.split_last() {
                        Some((0, rest)) => rest,
                        Some(_) => {
                            return Err(BagError::invalid_message_data(
                                "CDR string missing NUL terminator",
                            ));
                        }
                        None => &[][..],
                    }
                } else {
                    raw
                };
                if out.aligned {
                    out.u32(text.len() as u32 + 1);
                    out.buf.extend_from_slice(text);
                    out.buf.push(0);
                } else {
                    out.u32(text.len() as u32);
                    out.buf.extend_from_slice(text);
                }
            }
            Op::Drop(size) => {
                inp.align(*size);
                inp.take(*size)?;
            }
            Op::Fill(size) => {
                out.align(*size);
                out.buf.resize(out.buf.len() + size, 0);
            }
            Op::Array {
                elem,
                fixed,
                scalar,
            } => {
                let n = match fixed {
                    Some(n) => *n,
                    None => {
                        let n = inp.u32()? as usize;
                        out.u32(n as u32);
                        n
                    }
                };
                if n == 0 {
                    continue;
                }
                if let Some(size) = scalar {
                    inp.align(*size);
                    let total = n
                        .checked_mul(*size)
                        .ok_or_else(|| BagError::invalid_message_data("array size overflow"))?;
                    let bytes = inp.take(total)?;
                    out.align(*size);
                    out.buf.extend_from_slice(bytes);
                } else if !elem.is_empty() {
                    for _ in 0..n {
                        run(elem, inp, out)?;
                    }
                }
            }
        }
    }
    Ok(())
}

impl Transcoder {
    /// Compile the converter for message `root` from `from` to `to`.
    pub fn compile(reg: &Registry, root: &str, from: Dialect, to: Dialect) -> Result<Self> {
        Ok(Self {
            ops: compile_fields(reg, root, from, to, 0)?,
            from,
            to,
        })
    }

    /// Convert one message. `out` is cleared first and reused across calls.
    pub fn convert(&self, input: &[u8], out: &mut Vec<u8>) -> Result<()> {
        out.clear();
        let in_aligned = self.from == Dialect::Ros2;
        let out_aligned = self.to == Dialect::Ros2;

        let origin_in = if in_aligned {
            match input {
                [_, 1, _, _, ..] => 4,
                [_, 0, _, _, ..] => {
                    return Err(BagError::invalid_message_data(
                        "big-endian CDR payloads are not supported",
                    ));
                }
                _ => return Err(BagError::invalid_message_data("CDR payload too short")),
            }
        } else {
            0
        };
        if out_aligned {
            out.extend_from_slice(&[0, 1, 0, 0]);
        }

        let mut inp = Input {
            buf: input,
            pos: origin_in,
            origin: origin_in,
            aligned: in_aligned,
        };
        let mut o = Output {
            buf: out,
            origin: if out_aligned { 4 } else { 0 },
            aligned: out_aligned,
        };
        run(&self.ops, &mut inp, &mut o)?;

        let trailing = input.len() - inp.pos.min(input.len());
        let allowed = if in_aligned { 3 } else { 0 };
        if trailing > allowed {
            return Err(BagError::invalid_message_data(format!(
                "{trailing} unread bytes after message (definition mismatch?)"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::catalog::catalog;
    use super::super::msgdef::{ArrayKind, FieldType, Registry};
    use super::*;

    /// xorshift64: deterministic, no dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Independent reference encoders: the same random value written both as
    /// packed ROS1 and as aligned CDR, straight from the definition.
    struct Both {
        ros1: Vec<u8>,
        cdr: Vec<u8>,
        rng: Rng,
    }

    impl Both {
        fn new(seed: u64) -> Self {
            Self {
                ros1: Vec::new(),
                cdr: vec![0, 1, 0, 0],
                rng: Rng(seed | 1),
            }
        }

        fn scalar(&mut self, size: usize) {
            let v = self.rng.next().to_le_bytes();
            self.ros1.extend_from_slice(&v[..size]);
            let rem = (self.cdr.len() - 4) % size;
            if rem != 0 {
                self.cdr.resize(self.cdr.len() + size - rem, 0);
            }
            self.cdr.extend_from_slice(&v[..size]);
        }

        fn u32(&mut self, v: u32) {
            self.ros1.extend_from_slice(&v.to_le_bytes());
            let rem = (self.cdr.len() - 4) % 4;
            if rem != 0 {
                self.cdr.resize(self.cdr.len() + 4 - rem, 0);
            }
            self.cdr.extend_from_slice(&v.to_le_bytes());
        }

        fn string(&mut self) {
            let len = (self.rng.next() % 7) as usize;
            let text: Vec<u8> = (0..len).map(|i| b'a' + (i as u8)).collect();
            self.ros1.extend_from_slice(&(len as u32).to_le_bytes());
            self.ros1.extend_from_slice(&text);
            self.u32_cdr_only(len as u32 + 1);
            self.cdr.extend_from_slice(&text);
            self.cdr.push(0);
        }

        fn u32_cdr_only(&mut self, v: u32) {
            let rem = (self.cdr.len() - 4) % 4;
            if rem != 0 {
                self.cdr.resize(self.cdr.len() + 4 - rem, 0);
            }
            self.cdr.extend_from_slice(&v.to_le_bytes());
        }

        fn msg(&mut self, reg: &Registry, name: &str) {
            let def = reg.get(name).expect("defined");
            for f in &def.fields {
                let count = match f.array {
                    ArrayKind::Scalar => 1,
                    ArrayKind::Fixed(n) => n as usize,
                    ArrayKind::Dynamic => {
                        let n = (self.rng.next() % 4) as u32;
                        self.u32(n);
                        n as usize
                    }
                };
                for _ in 0..count {
                    match &f.ty {
                        FieldType::Prim(p) => self.scalar(p.size()),
                        FieldType::String => self.string(),
                        FieldType::Time | FieldType::Duration => {
                            self.scalar(4);
                            self.scalar(4);
                        }
                        FieldType::Header => {
                            // ROS1 seq = 0 so the ROS2 -> ROS1 direction (which
                            // synthesises seq = 0) reproduces it exactly.
                            self.ros1.extend_from_slice(&0u32.to_le_bytes());
                            self.scalar(4);
                            self.scalar(4);
                            self.string();
                        }
                        FieldType::Msg(dep) => self.msg(reg, dep),
                    }
                }
            }
        }
    }

    /// Every standard type, random contents: the transcoder must reproduce
    /// the independently-encoded bytes in both directions.
    #[test]
    fn matches_reference_encoders_for_every_catalog_type() {
        let reg = catalog().unwrap_or_else(|e| panic!("{e}"));
        let mut checked = 0;
        for name in reg.names() {
            let to_cdr = Transcoder::compile(reg, &name, Dialect::Ros1, Dialect::Ros2)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let to_ros1 = Transcoder::compile(reg, &name, Dialect::Ros2, Dialect::Ros1)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            for seed in 1..=20u64 {
                let mut b = Both::new(seed * 0x9E37_79B9 + checked);
                b.msg(reg, &name);
                let mut out = Vec::new();
                to_cdr
                    .convert(&b.ros1, &mut out)
                    .unwrap_or_else(|e| panic!("{name} ros1->cdr: {e}"));
                assert_eq!(out, b.cdr, "{name} ros1->cdr (seed {seed})");
                to_ros1
                    .convert(&b.cdr, &mut out)
                    .unwrap_or_else(|e| panic!("{name} cdr->ros1: {e}"));
                assert_eq!(out, b.ros1, "{name} cdr->ros1 (seed {seed})");
            }
            checked += 1;
        }
        assert!(checked > 100, "only {checked} types checked");
    }

    #[test]
    fn rejects_truncated_and_mismatched_input() -> Result<()> {
        let reg = Registry::parse("t/S", "string a\nint32 b\n")?;
        let t = Transcoder::compile(&reg, "t/S", Dialect::Ros1, Dialect::Ros2)?;
        let mut out = Vec::new();
        // string claims 100 bytes but only 3 follow
        let mut bad = 100u32.to_le_bytes().to_vec();
        bad.extend_from_slice(b"abc");
        assert!(t.convert(&bad, &mut out).is_err());
        // trailing garbage after a complete message
        let mut extra = Vec::new();
        extra.extend_from_slice(&0u32.to_le_bytes());
        extra.extend_from_slice(&7i32.to_le_bytes());
        extra.push(9);
        assert!(t.convert(&extra, &mut out).is_err());
        // CDR input: too short / big-endian
        let back = Transcoder::compile(&reg, "t/S", Dialect::Ros2, Dialect::Ros1)?;
        assert!(back.convert(&[0, 1], &mut out).is_err());
        assert!(
            back.convert(&[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], &mut out)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn empty_arrays_do_not_emit_alignment_padding() -> Result<()> {
        // CDR omits element alignment for empty sequences.
        let reg = Registry::parse("t/A", "uint8 a\nfloat64[] v\n")?;
        let to_cdr = Transcoder::compile(&reg, "t/A", Dialect::Ros1, Dialect::Ros2)?;
        let mut ros1 = vec![5u8];
        ros1.extend_from_slice(&0u32.to_le_bytes());
        let mut out = Vec::new();
        to_cdr.convert(&ros1, &mut out)?;
        // header(4) + a(1) + pad(3) + count(4); no 8-alignment padding.
        assert_eq!(out, [0, 1, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0]);
        Ok(())
    }
}
