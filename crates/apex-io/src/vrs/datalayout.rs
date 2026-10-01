//! `DataLayout`: the typed, named header of a record.
//!
//! A layout is described in the stream tags (`DL:<Type>:<version>:<block>`)
//! as JSON. On disk a layout instance is
//!
//! ```text
//! fixed section        (pieces with an "offset": scalars, arrays)
//! var-size index       nVar x { u32 offset, u32 byteLength }   (offsets relative to var data)
//! var-size data        strings, vectors
//! ```
//!
//! `DataPieceVector<string>` stores `u32 count` followed by
//! `count x { u32 len, bytes }`; other vectors are packed elements.

use super::error::{Result, VrsError};
use serde::Deserialize;
use std::collections::HashMap;

/// Scalar element types found in layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scalar {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    /// `Point3Df`: three `f32`.
    Point3F,
}

impl Scalar {
    /// Size in bytes.
    pub fn size(self) -> usize {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
            Self::Point3F => 12,
        }
    }

    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "Bool" | "bool" => Self::Bool,
            "int8_t" => Self::I8,
            "uint8_t" => Self::U8,
            "int16_t" => Self::I16,
            "uint16_t" => Self::U16,
            "int32_t" => Self::I32,
            "uint32_t" => Self::U32,
            "int64_t" => Self::I64,
            "uint64_t" => Self::U64,
            "float" => Self::F32,
            "double" => Self::F64,
            "Point3Df" => Self::Point3F,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::Bool => "Bool",
            Self::I8 => "int8_t",
            Self::U8 => "uint8_t",
            Self::I16 => "int16_t",
            Self::U16 => "uint16_t",
            Self::I32 => "int32_t",
            Self::U32 => "uint32_t",
            Self::I64 => "int64_t",
            Self::U64 => "uint64_t",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::Point3F => "Point3Df",
        }
    }
}

/// What a piece holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PieceKind {
    /// One scalar at a fixed offset.
    Value(Scalar),
    /// `size` scalars at a fixed offset.
    Array(Scalar, usize),
    /// Variable-length string.
    String,
    /// Variable-length vector of scalars.
    Vector(Scalar),
    /// Variable-length vector of strings.
    StringVector,
    /// A type this implementation does not decode (its bytes stay accessible
    /// through the raw payload).
    Unknown(String),
}

/// One named field of a layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Piece {
    /// Field name.
    pub name: String,
    /// Field type.
    pub kind: PieceKind,
    /// Byte offset in the fixed section (fixed pieces only).
    pub offset: Option<usize>,
    /// Slot in the var-size index (variable pieces only).
    pub index: Option<usize>,
}

#[derive(Deserialize)]
struct JsonLayout {
    data_layout: Vec<JsonPiece>,
}

#[derive(Deserialize)]
struct JsonPiece {
    name: String,
    #[serde(rename = "type")]
    ty: String,
    offset: Option<usize>,
    index: Option<usize>,
    size: Option<usize>,
}

fn template_arg(ty: &str, prefix: &str) -> Option<String> {
    ty.strip_prefix(prefix)?
        .strip_prefix('<')?
        .strip_suffix('>')
        .map(str::to_string)
}

fn parse_kind(ty: &str, size: Option<usize>) -> PieceKind {
    if ty == "DataPieceString" {
        return PieceKind::String;
    }
    if let Some(inner) = template_arg(ty, "DataPieceValue") {
        return Scalar::parse(&inner).map_or(PieceKind::Unknown(ty.into()), PieceKind::Value);
    }
    if let Some(inner) = template_arg(ty, "DataPieceArray") {
        return match (Scalar::parse(&inner), size) {
            (Some(s), Some(n)) => PieceKind::Array(s, n),
            _ => PieceKind::Unknown(ty.into()),
        };
    }
    if let Some(inner) = template_arg(ty, "DataPieceVector") {
        if inner == "string" {
            return PieceKind::StringVector;
        }
        return Scalar::parse(&inner).map_or(PieceKind::Unknown(ty.into()), PieceKind::Vector);
    }
    PieceKind::Unknown(ty.into())
}

/// A parsed layout description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataLayout {
    pieces: Vec<Piece>,
    by_name: HashMap<String, usize>,
    fixed_size: usize,
    var_count: usize,
}

impl DataLayout {
    /// Parse the JSON stored in a `DL:` stream tag.
    pub fn from_json(json: &str) -> Result<Self> {
        let parsed: JsonLayout = serde_json::from_str(json)?;
        let mut pieces = Vec::with_capacity(parsed.data_layout.len());
        for p in parsed.data_layout {
            pieces.push(Piece {
                kind: parse_kind(&p.ty, p.size),
                name: p.name,
                offset: p.offset,
                index: p.index,
            });
        }
        Self::from_pieces(pieces)
    }

    /// Build from pieces (used by the writer).
    pub fn from_pieces(pieces: Vec<Piece>) -> Result<Self> {
        let mut by_name = HashMap::new();
        let mut fixed_size = 0usize;
        let mut var_count = 0usize;
        for (i, p) in pieces.iter().enumerate() {
            if by_name.insert(p.name.clone(), i).is_some() {
                return Err(VrsError::Layout(format!("duplicate piece '{}'", p.name)));
            }
            match (&p.kind, p.offset, p.index) {
                (PieceKind::Value(s), Some(o), _) => fixed_size = fixed_size.max(o + s.size()),
                (PieceKind::Array(s, n), Some(o), _) => {
                    fixed_size = fixed_size.max(o + s.size() * n)
                }
                (PieceKind::Unknown(_), Some(_), _) => {}
                (_, None, Some(ix)) => var_count = var_count.max(ix + 1),
                _ => {
                    return Err(VrsError::Layout(format!(
                        "piece '{}' has neither a valid offset nor an index",
                        p.name
                    )));
                }
            }
        }
        Ok(Self {
            pieces,
            by_name,
            fixed_size,
            var_count,
        })
    }

    /// Serialise to the JSON form stored in tags.
    pub fn to_json(&self) -> String {
        let fields: Vec<serde_json::Value> = self
            .pieces
            .iter()
            .map(|p| {
                let ty = match &p.kind {
                    PieceKind::Value(s) => format!("DataPieceValue<{}>", s.name()),
                    PieceKind::Array(s, _) => format!("DataPieceArray<{}>", s.name()),
                    PieceKind::String => "DataPieceString".to_string(),
                    PieceKind::Vector(s) => format!("DataPieceVector<{}>", s.name()),
                    PieceKind::StringVector => "DataPieceVector<string>".to_string(),
                    PieceKind::Unknown(t) => t.clone(),
                };
                let mut m = serde_json::Map::new();
                m.insert("name".into(), p.name.clone().into());
                m.insert("type".into(), ty.into());
                if let Some(o) = p.offset {
                    m.insert("offset".into(), o.into());
                }
                if let Some(i) = p.index {
                    m.insert("index".into(), i.into());
                }
                if let PieceKind::Array(_, n) = p.kind {
                    m.insert("size".into(), n.into());
                }
                serde_json::Value::Object(m)
            })
            .collect();
        serde_json::json!({ "data_layout": fields }).to_string()
    }

    /// Pieces in declaration order.
    pub fn pieces(&self) -> &[Piece] {
        &self.pieces
    }

    /// Size of the fixed section.
    pub fn fixed_size(&self) -> usize {
        self.fixed_size
    }

    /// Number of variable-size pieces.
    pub fn var_count(&self) -> usize {
        self.var_count
    }

    /// Does the layout define `name`?
    pub fn has(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// Bind the layout to the start of `data`, validating that the fixed
    /// section, var index and var data all lie inside it.
    pub fn bind<'a>(&'a self, data: &'a [u8]) -> Result<LayoutView<'a>> {
        let index_end = self
            .fixed_size
            .checked_add(self.var_count * 8)
            .filter(|&e| e <= data.len())
            .ok_or_else(|| VrsError::Layout("record shorter than its DataLayout".into()))?;
        let mut var_end = 0usize;
        for k in 0..self.var_count {
            let at = self.fixed_size + k * 8;
            let off =
                u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]) as usize;
            let len = u32::from_le_bytes([data[at + 4], data[at + 5], data[at + 6], data[at + 7]])
                as usize;
            let end = off
                .checked_add(len)
                .and_then(|e| e.checked_add(index_end))
                .filter(|&e| e <= data.len())
                .ok_or_else(|| VrsError::Layout("variable-size data out of bounds".into()))?;
            var_end = var_end.max(end);
        }
        Ok(LayoutView {
            layout: self,
            data,
            index_end,
            total_len: var_end.max(index_end),
        })
    }
}

/// A layout bound to record bytes: typed access by field name.
#[derive(Debug, Clone, Copy)]
pub struct LayoutView<'a> {
    layout: &'a DataLayout,
    data: &'a [u8],
    index_end: usize,
    total_len: usize,
}

macro_rules! scalar_getter {
    ($(#[$m:meta])* $fn:ident, $ret:ty, $scalar:ident, $n:expr) => {
        $(#[$m])*
        pub fn $fn(&self, name: &str) -> Result<$ret> {
            let b = self.scalar_bytes(name, Scalar::$scalar)?;
            let mut a = [0u8; $n];
            a.copy_from_slice(b);
            Ok(<$ret>::from_le_bytes(a))
        }
    };
}

impl<'a> LayoutView<'a> {
    /// Bytes this layout instance occupies (fixed + index + var data); the
    /// next content block starts here.
    pub fn total_len(&self) -> usize {
        self.total_len
    }

    fn piece(&self, name: &str) -> Result<&'a Piece> {
        let layout: &'a DataLayout = self.layout;
        layout
            .by_name
            .get(name)
            .map(|&i| &layout.pieces[i])
            .ok_or_else(|| VrsError::Layout(format!("no field '{name}' in this DataLayout")))
    }

    fn scalar_bytes(&self, name: &str, want: Scalar) -> Result<&'a [u8]> {
        let p = self.piece(name)?;
        match (&p.kind, p.offset) {
            (PieceKind::Value(s), Some(o)) if *s == want => Ok(&self.data[o..o + s.size()]),
            _ => Err(VrsError::Layout(format!(
                "field '{name}' is not a {} value",
                want.name()
            ))),
        }
    }

    scalar_getter!(
        /// Read a `uint8_t` field.
        get_u8, u8, U8, 1
    );
    scalar_getter!(
        /// Read a `uint32_t` field.
        get_u32, u32, U32, 4
    );
    scalar_getter!(
        /// Read an `int32_t` field.
        get_i32, i32, I32, 4
    );
    scalar_getter!(
        /// Read a `uint64_t` field.
        get_u64, u64, U64, 8
    );
    scalar_getter!(
        /// Read an `int64_t` field.
        get_i64, i64, I64, 8
    );
    scalar_getter!(
        /// Read a `float` field.
        get_f32, f32, F32, 4
    );
    scalar_getter!(
        /// Read a `double` field.
        get_f64, f64, F64, 8
    );

    /// Read a `Bool` field.
    pub fn get_bool(&self, name: &str) -> Result<bool> {
        Ok(self.scalar_bytes(name, Scalar::Bool)?[0] != 0)
    }

    /// Read a `Point3Df` field.
    pub fn get_point3f(&self, name: &str) -> Result<[f32; 3]> {
        let b = self.scalar_bytes(name, Scalar::Point3F)?;
        Ok(std::array::from_fn(|i| {
            f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
        }))
    }

    /// Read a fixed-size `float` array of exactly `N` elements.
    pub fn get_f32_array<const N: usize>(&self, name: &str) -> Result<[f32; N]> {
        let p = self.piece(name)?;
        match (&p.kind, p.offset) {
            (PieceKind::Array(Scalar::F32, n), Some(o)) if *n == N => {
                Ok(std::array::from_fn(|i| {
                    let at = o + 4 * i;
                    f32::from_le_bytes([
                        self.data[at],
                        self.data[at + 1],
                        self.data[at + 2],
                        self.data[at + 3],
                    ])
                }))
            }
            _ => Err(VrsError::Layout(format!(
                "field '{name}' is not a float[{N}] array"
            ))),
        }
    }

    /// Read a fixed-size `double` array of exactly `N` elements.
    pub fn get_f64_array<const N: usize>(&self, name: &str) -> Result<[f64; N]> {
        let p = self.piece(name)?;
        match (&p.kind, p.offset) {
            (PieceKind::Array(Scalar::F64, n), Some(o)) if *n == N => {
                Ok(std::array::from_fn(|i| {
                    let at = o + 8 * i;
                    let mut a = [0u8; 8];
                    a.copy_from_slice(&self.data[at..at + 8]);
                    f64::from_le_bytes(a)
                }))
            }
            _ => Err(VrsError::Layout(format!(
                "field '{name}' is not a double[{N}] array"
            ))),
        }
    }

    fn var_bytes(&self, p: &Piece) -> Result<&'a [u8]> {
        let k = p
            .index
            .ok_or_else(|| VrsError::Layout(format!("field '{}' is not variable-size", p.name)))?;
        let at = self.layout.fixed_size + k * 8;
        let off = u32::from_le_bytes([
            self.data[at],
            self.data[at + 1],
            self.data[at + 2],
            self.data[at + 3],
        ]) as usize;
        let len = u32::from_le_bytes([
            self.data[at + 4],
            self.data[at + 5],
            self.data[at + 6],
            self.data[at + 7],
        ]) as usize;
        // `bind` already checked these bounds.
        Ok(&self.data[self.index_end + off..self.index_end + off + len])
    }

    /// Read a string field.
    pub fn get_string(&self, name: &str) -> Result<String> {
        let p = self.piece(name)?;
        if p.kind != PieceKind::String {
            return Err(VrsError::Layout(format!("field '{name}' is not a string")));
        }
        Ok(String::from_utf8_lossy(self.var_bytes(p)?).into_owned())
    }

    /// Read a vector-of-`float` field.
    pub fn get_f32_vector(&self, name: &str) -> Result<Vec<f32>> {
        let p = self.piece(name)?;
        if p.kind != PieceKind::Vector(Scalar::F32) {
            return Err(VrsError::Layout(format!(
                "field '{name}' is not a float vector"
            )));
        }
        let (chunks, _) = self.var_bytes(p)?.as_chunks::<4>();
        Ok(chunks.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }

    /// Read a vector-of-string field.
    pub fn get_string_vector(&self, name: &str) -> Result<Vec<String>> {
        let p = self.piece(name)?;
        if p.kind != PieceKind::StringVector {
            return Err(VrsError::Layout(format!(
                "field '{name}' is not a string vector"
            )));
        }
        let b = self.var_bytes(p)?;
        if b.is_empty() {
            return Ok(Vec::new());
        }
        let read_u32 = |at: usize| -> Result<usize> {
            b.get(at..at + 4)
                .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]) as usize)
                .ok_or_else(|| VrsError::Layout("string vector truncated".into()))
        };
        let count = read_u32(0)?;
        let mut pos = 4usize;
        let mut out = Vec::new();
        for _ in 0..count {
            let len = read_u32(pos)?;
            pos += 4;
            let s = b
                .get(pos..pos.saturating_add(len))
                .ok_or_else(|| VrsError::Layout("string vector truncated".into()))?;
            out.push(String::from_utf8_lossy(s).into_owned());
            pos += len;
        }
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The IMU data layout of real Aria files (79 bytes).
    const IMU_JSON: &str = r#"{"data_layout":[
        {"name":"accelerometer_valid","type":"DataPieceValue<Bool>","offset":0},
        {"name":"gyroscope_valid","type":"DataPieceValue<Bool>","offset":1},
        {"name":"magnetometer_valid","type":"DataPieceValue<Bool>","offset":2},
        {"name":"temperature_deg_c","type":"DataPieceValue<double>","offset":3},
        {"name":"capture_timestamp_ns","type":"DataPieceValue<int64_t>","offset":11},
        {"name":"accelerometer","type":"DataPieceArray<float>","offset":43,"size":3}]}"#;

    #[test]
    fn imu_layout_has_the_documented_size_and_values() {
        let l = DataLayout::from_json(IMU_JSON).unwrap();
        assert_eq!(l.fixed_size(), 55);
        assert_eq!(l.var_count(), 0);

        let mut d = vec![0u8; 55];
        d[0] = 1;
        d[3..11].copy_from_slice(&36.5f64.to_le_bytes());
        d[11..19].copy_from_slice(&(-7i64).to_le_bytes());
        for (i, v) in [1.0f32, -2.0, 9.81].iter().enumerate() {
            d[43 + 4 * i..47 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        let v = l.bind(&d).unwrap();
        assert!(v.get_bool("accelerometer_valid").unwrap());
        assert!(!v.get_bool("gyroscope_valid").unwrap());
        assert_eq!(v.get_f64("temperature_deg_c").unwrap(), 36.5);
        assert_eq!(v.get_i64("capture_timestamp_ns").unwrap(), -7);
        assert_eq!(
            v.get_f32_array::<3>("accelerometer").unwrap(),
            [1.0, -2.0, 9.81]
        );
        assert_eq!(v.total_len(), 55);
    }

    #[test]
    fn wrong_type_or_name_is_an_error() {
        let l = DataLayout::from_json(IMU_JSON).unwrap();
        let d = vec![0u8; 55];
        let v = l.bind(&d).unwrap();
        assert!(v.get_f64("capture_timestamp_ns").is_err());
        assert!(v.get_i64("nope").is_err());
        assert!(v.get_f32_array::<4>("accelerometer").is_err());
        assert!(v.get_string("accelerometer").is_err());
    }

    #[test]
    fn short_record_is_rejected() {
        let l = DataLayout::from_json(IMU_JSON).unwrap();
        assert!(l.bind(&[0u8; 54]).is_err());
    }

    #[test]
    fn var_size_pieces_decode_like_a_real_wifi_record() {
        let json = r#"{"data_layout":[
            {"name":"rssi","type":"DataPieceValue<float>","offset":0},
            {"name":"ssid","type":"DataPieceString","index":0},
            {"name":"antennas","type":"DataPieceVector<float>","index":1},
            {"name":"raw","type":"DataPieceVector<string>","index":2}]}"#;
        let l = DataLayout::from_json(json).unwrap();
        assert_eq!((l.fixed_size(), l.var_count()), (4, 3));

        let mut vardata = Vec::new();
        vardata.extend_from_slice(b"eth-visitors"); // 0..12
        vardata.extend_from_slice(&1.5f32.to_le_bytes()); // 12..16
        vardata.extend_from_slice(&(-2.5f32).to_le_bytes()); // 16..20
        let strings_at = vardata.len();
        vardata.extend_from_slice(&2u32.to_le_bytes());
        for s in ["ab", "cde"] {
            vardata.extend_from_slice(&(s.len() as u32).to_le_bytes());
            vardata.extend_from_slice(s.as_bytes());
        }
        let strings_len = vardata.len() - strings_at;

        let mut d = Vec::new();
        d.extend_from_slice(&(-60.0f32).to_le_bytes());
        for (o, n) in [
            (0u32, 12u32),
            (12, 8),
            (strings_at as u32, strings_len as u32),
        ] {
            d.extend_from_slice(&o.to_le_bytes());
            d.extend_from_slice(&n.to_le_bytes());
        }
        d.extend_from_slice(&vardata);
        d.extend_from_slice(b"TRAILING-CONTENT-BLOCK");

        let v = l.bind(&d).unwrap();
        assert_eq!(v.get_f32("rssi").unwrap(), -60.0);
        assert_eq!(v.get_string("ssid").unwrap(), "eth-visitors");
        assert_eq!(v.get_f32_vector("antennas").unwrap(), [1.5, -2.5]);
        assert_eq!(v.get_string_vector("raw").unwrap(), ["ab", "cde"]);
        // Content blocks begin right after the var data.
        assert_eq!(&d[v.total_len()..], b"TRAILING-CONTENT-BLOCK");
    }

    #[test]
    fn hostile_var_index_is_rejected() {
        let l = DataLayout::from_json(
            r#"{"data_layout":[{"name":"s","type":"DataPieceString","index":0}]}"#,
        )
        .unwrap();
        let mut d = Vec::new();
        d.extend_from_slice(&u32::MAX.to_le_bytes());
        d.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(l.bind(&d).is_err());
    }

    #[test]
    fn json_round_trips_and_unknown_types_survive() {
        let json = r#"{"data_layout":[
            {"name":"a","type":"DataPieceValue<double>","offset":0},
            {"name":"w","type":"DataPieceWeird<x>","offset":8}]}"#;
        let l = DataLayout::from_json(json).unwrap();
        assert!(matches!(l.pieces()[1].kind, PieceKind::Unknown(_)));
        let again = DataLayout::from_json(&l.to_json()).unwrap();
        assert_eq!(again, l);
    }

    #[test]
    fn duplicate_or_malformed_pieces_are_rejected() {
        assert!(DataLayout::from_json(
            r#"{"data_layout":[{"name":"a","type":"DataPieceValue<double>","offset":0},{"name":"a","type":"DataPieceValue<double>","offset":8}]}"#
        )
        .is_err());
        assert!(
            DataLayout::from_json(
                r#"{"data_layout":[{"name":"a","type":"DataPieceValue<double>"}]}"#
            )
            .is_err()
        );
        assert!(DataLayout::from_json("not json").is_err());
    }
}
