//! `.msg` definition parsing, rendering and ROS1 md5 computation.
//!
//! A [`Registry`] holds message definitions in a dialect-neutral form:
//! type names are normalised to `pkg/Type` (ROS2's `pkg/msg/Type` is folded),
//! `time`/`builtin_interfaces/Time` are one [`FieldType::Time`], and
//! `Header`/`std_msgs/Header` are one [`FieldType::Header`] whose wire shape
//! (ROS1 has a `seq` field, ROS2 does not) is decided per dialect later.
//! Both the ROS1 concatenated form (`MSG: pkg/Type` sections) and the ROS2
//! `ros2msg` form parse with the same code.

use crate::rosbag::error::{BagError, Result};
use std::collections::HashMap;

/// md5sum of `std_msgs/Header` (a message type, so its sum feeds parents).
const HEADER_MD5: &str = "2176decaecbce78abc3b96ef049fabed";

/// Wire dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// ROS1: packed little-endian, `u32` length prefixes, no NUL in strings.
    Ros1,
    /// ROS2 CDR: aligned little-endian, NUL-terminated strings.
    Ros2,
}

/// Fixed-size scalar types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prim {
    Bool,
    /// ROS1 `byte` (signed 8-bit alias).
    Byte,
    /// ROS1 `char` (unsigned 8-bit alias).
    Char,
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
}

impl Prim {
    /// Size in bytes (also the CDR alignment).
    pub fn size(self) -> usize {
        match self {
            Self::Bool | Self::Byte | Self::Char | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Byte => "byte",
            Self::Char => "char",
            Self::I8 => "int8",
            Self::U8 => "uint8",
            Self::I16 => "int16",
            Self::U16 => "uint16",
            Self::I32 => "int32",
            Self::U32 => "uint32",
            Self::I64 => "int64",
            Self::U64 => "uint64",
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "bool" => Self::Bool,
            "byte" => Self::Byte,
            "char" => Self::Char,
            "int8" => Self::I8,
            "uint8" => Self::U8,
            "int16" => Self::I16,
            "uint16" => Self::U16,
            "int32" => Self::I32,
            "uint32" => Self::U32,
            "int64" => Self::I64,
            "uint64" => Self::U64,
            "float32" => Self::F32,
            "float64" => Self::F64,
            _ => return None,
        })
    }
}

/// Type of a field, ignoring array-ness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldType {
    Prim(Prim),
    String,
    /// `time` / `builtin_interfaces/Time`.
    Time,
    /// `duration` / `builtin_interfaces/Duration`.
    Duration,
    /// `std_msgs/Header`.
    Header,
    /// Another message, as `pkg/Type`.
    Msg(String),
}

/// Array shape of a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayKind {
    Scalar,
    Fixed(u32),
    /// Unbounded or bounded (`[<=N]`) sequence.
    Dynamic,
}

/// One message field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub ty: FieldType,
    pub array: ArrayKind,
}

/// A message constant, kept verbatim for md5 purposes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constant {
    pub ty: String,
    pub name: String,
    pub value: String,
}

/// A parsed message type.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MsgDef {
    pub fields: Vec<Field>,
    pub constants: Vec<Constant>,
}

/// A set of message definitions keyed by `pkg/Type`.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    defs: HashMap<String, MsgDef>,
}

fn bad(reason: impl Into<String>) -> BagError {
    BagError::schema_validation(reason.into())
}

/// Fold `pkg/msg/Type` to `pkg/Type`.
pub fn normalize_type_name(name: &str) -> String {
    let parts: Vec<&str> = name.split('/').collect();
    match parts.as_slice() {
        [pkg, "msg", ty] => format!("{pkg}/{ty}"),
        _ => name.to_string(),
    }
}

/// `pkg/Type` in the target dialect's spelling (`pkg/msg/Type` for ROS2).
pub fn dialect_type_name(name: &str, dialect: Dialect) -> String {
    let norm = normalize_type_name(name);
    match (dialect, norm.split_once('/')) {
        (Dialect::Ros2, Some((pkg, ty))) => format!("{pkg}/msg/{ty}"),
        _ => norm,
    }
}

fn is_separator(line: &str) -> bool {
    line.len() >= 3 && line.bytes().all(|b| b == b'=')
}

impl Registry {
    /// Parse a (possibly multi-section) definition whose first section
    /// defines `root`.
    pub fn parse(root: &str, text: &str) -> Result<Self> {
        let mut reg = Self::default();
        reg.add_text(Some(root), text)?;
        Ok(reg)
    }

    /// Parse a text made only of `MSG: pkg/Type` sections.
    pub fn parse_sections(text: &str) -> Result<Self> {
        let mut reg = Self::default();
        reg.add_text(None, text)?;
        Ok(reg)
    }

    fn add_text(&mut self, root: Option<&str>, text: &str) -> Result<()> {
        let mut sections: Vec<(Option<String>, Vec<&str>)> = vec![(None, Vec::new())];
        let mut expect_name = false;
        for line in text.lines() {
            let trimmed = line.trim();
            if is_separator(trimmed) {
                sections.push((None, Vec::new()));
                expect_name = true;
                continue;
            }
            if expect_name {
                expect_name = false;
                if let Some(name) = trimmed.strip_prefix("MSG:") {
                    if let Some(last) = sections.last_mut() {
                        last.0 = Some(normalize_type_name(name.trim()));
                    }
                    continue;
                }
                if trimmed.starts_with("IDL:") {
                    return Err(bad("IDL message definitions are not supported"));
                }
            }
            if let Some(last) = sections.last_mut() {
                last.1.push(line);
            }
        }
        for (i, (name, lines)) in sections.into_iter().enumerate() {
            let name = match (name, root) {
                (Some(n), _) => n,
                (None, Some(r)) if i == 0 => normalize_type_name(r),
                (None, _)
                    if lines
                        .iter()
                        .all(|l| l.trim().is_empty() || l.trim().starts_with('#')) =>
                {
                    continue;
                }
                (None, _) => return Err(bad("definition section without a `MSG:` name")),
            };
            // First definition wins (embedded definitions override the catalog).
            if self.defs.contains_key(&name) {
                continue;
            }
            let pkg = name.split('/').next().unwrap_or("").to_string();
            let def = parse_body(&pkg, &lines)?;
            self.defs.insert(name, def);
        }
        Ok(())
    }

    /// Definition of `name` (`pkg/Type`).
    pub fn get(&self, name: &str) -> Option<&MsgDef> {
        self.defs.get(name)
    }

    /// All defined type names, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.defs.keys().cloned().collect();
        v.sort();
        v
    }

    /// Does the registry define `name`?
    pub fn contains(&self, name: &str) -> bool {
        self.defs.contains_key(name)
    }

    /// Copy every definition of `other` that this registry lacks.
    pub fn fill_missing_from(&mut self, other: &Registry) {
        for (k, v) in &other.defs {
            self.defs.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }

    /// Names of message types reachable from `root` that are not defined.
    pub fn missing_from(&self, root: &str) -> Vec<String> {
        let mut missing = Vec::new();
        let mut seen = Vec::new();
        self.walk(root, &mut seen, &mut missing);
        missing
    }

    fn walk(&self, name: &str, seen: &mut Vec<String>, missing: &mut Vec<String>) {
        if seen.iter().any(|s| s == name) {
            return;
        }
        seen.push(name.to_string());
        let Some(def) = self.defs.get(name) else {
            missing.push(name.to_string());
            return;
        };
        for f in &def.fields {
            if let FieldType::Msg(dep) = &f.ty {
                self.walk(dep, seen, missing);
            }
        }
    }

    /// Render `root` (and its dependencies) as a definition text in `dialect`.
    pub fn render(&self, root: &str, dialect: Dialect) -> Result<String> {
        let mut order = Vec::new();
        let mut flags = Uses::default();
        self.collect(root, &mut order, &mut flags)?;

        let mut out = String::new();
        self.render_body(root, dialect, &mut out)?;
        for dep in order.iter().skip(1) {
            push_section(&mut out, dep);
            self.render_body(dep, dialect, &mut out)?;
        }
        if flags.header {
            push_section(&mut out, "std_msgs/Header");
            match dialect {
                Dialect::Ros1 => out.push_str("uint32 seq\ntime stamp\nstring frame_id\n"),
                Dialect::Ros2 => out.push_str("builtin_interfaces/Time stamp\nstring frame_id\n"),
            }
        }
        if dialect == Dialect::Ros2 {
            if flags.time || flags.header {
                push_section(&mut out, "builtin_interfaces/Time");
                out.push_str("int32 sec\nuint32 nanosec\n");
            }
            if flags.duration {
                push_section(&mut out, "builtin_interfaces/Duration");
                out.push_str("int32 sec\nuint32 nanosec\n");
            }
        }
        Ok(out.trim_end().to_string())
    }

    fn collect(&self, name: &str, order: &mut Vec<String>, flags: &mut Uses) -> Result<()> {
        if order.iter().any(|s| s == name) {
            return Ok(());
        }
        let def = self
            .defs
            .get(name)
            .ok_or_else(|| bad(format!("message type '{name}' is not defined")))?;
        order.push(name.to_string());
        for f in &def.fields {
            match &f.ty {
                FieldType::Time => flags.time = true,
                FieldType::Duration => flags.duration = true,
                FieldType::Header => flags.header = true,
                FieldType::Msg(dep) => self.collect(dep, order, flags)?,
                _ => {}
            }
        }
        Ok(())
    }

    fn render_body(&self, name: &str, dialect: Dialect, out: &mut String) -> Result<()> {
        let def = self
            .defs
            .get(name)
            .ok_or_else(|| bad(format!("message type '{name}' is not defined")))?;
        for c in &def.constants {
            out.push_str(&format!("{} {}={}\n", c.ty, c.name, c.value));
        }
        for f in &def.fields {
            out.push_str(&field_type_text(&f.ty, f.array, dialect, true));
            out.push(' ');
            out.push_str(&f.name);
            out.push('\n');
        }
        Ok(())
    }

    /// ROS1 md5sum of `root` as rendered by genmsg.
    pub fn md5(&self, root: &str) -> Result<String> {
        let mut cache = HashMap::new();
        self.md5_inner(root, &mut cache)
    }

    fn md5_inner(&self, name: &str, cache: &mut HashMap<String, String>) -> Result<String> {
        if name == "std_msgs/Header" {
            return Ok(HEADER_MD5.to_string());
        }
        if let Some(h) = cache.get(name) {
            return Ok(h.clone());
        }
        let def = self
            .defs
            .get(name)
            .ok_or_else(|| bad(format!("message type '{name}' is not defined")))?;
        let mut lines: Vec<String> = def
            .constants
            .iter()
            .map(|c| format!("{} {}={}", c.ty, c.name, c.value))
            .collect();
        for f in &def.fields {
            match &f.ty {
                FieldType::Header => lines.push(format!("{HEADER_MD5} {}", f.name)),
                FieldType::Msg(dep) => {
                    let sub = self.md5_inner(dep, cache)?;
                    lines.push(format!("{sub} {}", f.name));
                }
                other => lines.push(format!(
                    "{} {}",
                    field_type_text(other, f.array, Dialect::Ros1, false),
                    f.name
                )),
            }
        }
        let digest = md5::compute(lines.join("\n"));
        let hex = format!("{digest:x}");
        cache.insert(name.to_string(), hex.clone());
        Ok(hex)
    }
}

#[derive(Default)]
struct Uses {
    time: bool,
    duration: bool,
    header: bool,
}

fn push_section(out: &mut String, name: &str) {
    out.push_str(&"=".repeat(80));
    out.push_str("\nMSG: ");
    out.push_str(name);
    out.push('\n');
}

fn field_type_text(
    ty: &FieldType,
    array: ArrayKind,
    dialect: Dialect,
    with_header: bool,
) -> String {
    let base = match ty {
        FieldType::Prim(p) => p.name().to_string(),
        FieldType::String => "string".to_string(),
        FieldType::Time => match dialect {
            Dialect::Ros1 => "time".into(),
            Dialect::Ros2 => "builtin_interfaces/Time".into(),
        },
        FieldType::Duration => match dialect {
            Dialect::Ros1 => "duration".into(),
            Dialect::Ros2 => "builtin_interfaces/Duration".into(),
        },
        FieldType::Header => match (dialect, with_header) {
            (Dialect::Ros1, true) => "Header".into(),
            _ => "std_msgs/Header".into(),
        },
        FieldType::Msg(n) => n.clone(),
    };
    match array {
        ArrayKind::Scalar => base,
        ArrayKind::Fixed(n) => format!("{base}[{n}]"),
        ArrayKind::Dynamic => format!("{base}[]"),
    }
}

fn parse_body(pkg: &str, lines: &[&str]) -> Result<MsgDef> {
    let mut def = MsgDef::default();
    for raw in lines {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.splitn(2, char::is_whitespace);
        let ty_text = it.next().unwrap_or("");
        let rest = it.next().unwrap_or("").trim();
        if rest.is_empty() {
            return Err(bad(format!("malformed definition line '{line}'")));
        }

        // Constant: `TYPE NAME=value` (the `=` may be spaced).
        if let Some(eq) = rest.find('=') {
            let name = rest[..eq].trim();
            if !name.is_empty() && !name.contains(char::is_whitespace) {
                let mut value = rest[eq + 1..].trim().to_string();
                if ty_text != "string"
                    && let Some(hash) = value.find('#')
                {
                    value.truncate(hash);
                    value = value.trim().to_string();
                }
                def.constants.push(Constant {
                    ty: ty_text.to_string(),
                    name: name.to_string(),
                    value,
                });
                continue;
            }
        }

        // Field: `TYPE NAME [default]`; defaults and trailing comments ignored.
        let name = rest
            .split(char::is_whitespace)
            .next()
            .unwrap_or("")
            .to_string();
        let (base, array) = split_array(ty_text)?;
        def.fields.push(Field {
            name,
            ty: resolve_type(pkg, base)?,
            array,
        });
    }
    Ok(def)
}

fn split_array(ty: &str) -> Result<(&str, ArrayKind)> {
    let Some(open) = ty.find('[') else {
        return Ok((ty, ArrayKind::Scalar));
    };
    let inner = ty[open + 1..]
        .strip_suffix(']')
        .ok_or_else(|| bad(format!("malformed array type '{ty}'")))?;
    let base = &ty[..open];
    if inner.is_empty() || inner.starts_with("<=") {
        return Ok((base, ArrayKind::Dynamic));
    }
    let n = inner
        .parse::<u32>()
        .map_err(|_| bad(format!("malformed array size in '{ty}'")))?;
    Ok((base, ArrayKind::Fixed(n)))
}

fn resolve_type(pkg: &str, base: &str) -> Result<FieldType> {
    if let Some(p) = Prim::parse(base) {
        return Ok(FieldType::Prim(p));
    }
    // `string<=N` is a bounded string.
    if base == "string" || base.starts_with("string<=") {
        return Ok(FieldType::String);
    }
    if base == "wstring" || base.starts_with("wstring<=") || base == "wchar" {
        return Err(bad("wstring/wchar fields are not supported"));
    }
    Ok(match base {
        "time" | "builtin_interfaces/Time" | "builtin_interfaces/msg/Time" => FieldType::Time,
        "duration" | "builtin_interfaces/Duration" | "builtin_interfaces/msg/Duration" => {
            FieldType::Duration
        }
        "Header" | "std_msgs/Header" | "std_msgs/msg/Header" => FieldType::Header,
        other if other.contains('/') => FieldType::Msg(normalize_type_name(other)),
        other => FieldType::Msg(format!("{pkg}/{other}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    const IMU_ROS1: &str = "Header header\ngeometry_msgs/Quaternion orientation\nfloat64[9] orientation_covariance\ngeometry_msgs/Vector3 angular_velocity\nfloat64[9] angular_velocity_covariance\ngeometry_msgs/Vector3 linear_acceleration\nfloat64[9] linear_acceleration_covariance\n\n================================================================================\nMSG: std_msgs/Header\nuint32 seq\ntime stamp\nstring frame_id\n\n================================================================================\nMSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n\n================================================================================\nMSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n";

    #[test]
    fn normalizes_type_names() {
        assert_eq!(
            normalize_type_name("sensor_msgs/msg/Imu"),
            "sensor_msgs/Imu"
        );
        assert_eq!(normalize_type_name("sensor_msgs/Imu"), "sensor_msgs/Imu");
        assert_eq!(
            dialect_type_name("sensor_msgs/Imu", Dialect::Ros2),
            "sensor_msgs/msg/Imu"
        );
        assert_eq!(
            dialect_type_name("sensor_msgs/msg/Imu", Dialect::Ros1),
            "sensor_msgs/Imu"
        );
    }

    /// Known md5sums of standard ROS1 messages (from the ROS1 distribution).
    #[test]
    fn md5_matches_ros1_for_standard_messages() -> TestResult {
        let imu = Registry::parse("sensor_msgs/Imu", IMU_ROS1)?;
        assert_eq!(
            imu.md5("sensor_msgs/Imu")?,
            "6a62c6daae103f4ff57a132d6f95cec2"
        );
        assert_eq!(
            imu.md5("geometry_msgs/Vector3")?,
            "4a842b65f413084dc2b10fb484ea7f17"
        );

        let string = Registry::parse("std_msgs/String", "string data\n")?;
        assert_eq!(
            string.md5("std_msgs/String")?,
            "992ce8a1687cec8c8bd883ec73ca41d1"
        );

        let image = Registry::parse(
            "sensor_msgs/Image",
            "Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n",
        )?;
        assert_eq!(
            image.md5("sensor_msgs/Image")?,
            "060021388200f6f0f447d0fcd9c64743"
        );
        Ok(())
    }

    #[test]
    fn constants_are_part_of_the_md5() -> TestResult {
        // std_msgs/Bool-like with a constant
        let r = Registry::parse("t/A", "uint8 FOO=1\nuint8 data\n")?;
        let def = r.get("t/A").ok_or("missing")?;
        assert_eq!(def.constants.len(), 1);
        assert_eq!(def.fields.len(), 1);
        assert_ne!(
            r.md5("t/A")?,
            Registry::parse("t/A", "uint8 data\n")?.md5("t/A")?
        );
        Ok(())
    }

    #[test]
    fn parses_ros2_syntax() -> TestResult {
        let r = Registry::parse(
            "t/B",
            "std_msgs/Header header\nstring<=5 s\nfloat64[<=3] v\nint32 x 7\nint32 K=3 # c\nbuiltin_interfaces/Time t\nt/Sub[2] subs\n",
        )?;
        let d = r.get("t/B").ok_or("missing")?;
        assert_eq!(d.fields.len(), 6);
        assert_eq!(d.fields[0].ty, FieldType::Header);
        assert_eq!(d.fields[1].ty, FieldType::String);
        assert_eq!(d.fields[2].array, ArrayKind::Dynamic);
        assert_eq!(d.fields[4].ty, FieldType::Time);
        assert_eq!(d.fields[5].ty, FieldType::Msg("t/Sub".into()));
        assert_eq!(d.fields[5].array, ArrayKind::Fixed(2));
        assert_eq!(d.constants[0].value, "3");
        Ok(())
    }

    #[test]
    fn unqualified_types_resolve_in_the_parent_package() -> TestResult {
        let r = Registry::parse("pkg/A", "B b\n")?;
        assert_eq!(
            r.get("pkg/A").ok_or("missing")?.fields[0].ty,
            FieldType::Msg("pkg/B".into())
        );
        assert_eq!(r.missing_from("pkg/A"), vec!["pkg/B".to_string()]);
        Ok(())
    }

    #[test]
    fn render_round_trips_between_dialects() -> TestResult {
        let r = Registry::parse("sensor_msgs/Imu", IMU_ROS1)?;
        let ros2 = r.render("sensor_msgs/Imu", Dialect::Ros2)?;
        assert!(ros2.starts_with("std_msgs/Header header"));
        assert!(ros2.contains("MSG: builtin_interfaces/Time"));
        assert!(!ros2.contains("uint32 seq"));

        let ros1 = r.render("sensor_msgs/Imu", Dialect::Ros1)?;
        assert!(ros1.starts_with("Header header"));
        assert!(ros1.contains("uint32 seq"));

        // Re-parse the ROS1 rendering: md5 is stable.
        let again = Registry::parse("sensor_msgs/Imu", &ros1)?;
        assert_eq!(
            again.md5("sensor_msgs/Imu")?,
            "6a62c6daae103f4ff57a132d6f95cec2"
        );
        Ok(())
    }

    #[test]
    fn rejects_wstring_and_idl() {
        assert!(Registry::parse("t/A", "wstring s\n").is_err());
        assert!(Registry::parse("t/A", "int32 a\n====\nIDL: x\n").is_err());
    }
}
