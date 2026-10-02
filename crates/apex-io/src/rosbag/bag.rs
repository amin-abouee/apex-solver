//! Format-agnostic bag access.
//!
//! One reader and one writer over every supported container, so tools (and
//! users) can copy, filter and convert without caring which backend sits
//! behind a path:
//!
//! | [`BagKind`]  | On disk                         | Payload encoding |
//! |--------------|---------------------------------|------------------|
//! | `Ros1Bag`    | `*.bag` (ROS1 v2.0)             | `ros1`           |
//! | `Ros2Sqlite` | directory + `metadata.yaml`     | `cdr`            |
//! | `Ros2Mcap`   | directory + `metadata.yaml`     | `cdr`            |
//! | `McapFile`   | bare `*.mcap` (ROS2 or ROS1 profile) | per channel |
//!
//! Both sides stream: messages reach the callback as borrowed slices and the
//! writers write straight through, so memory use does not grow with bag size.

use super::error::{BagError, Result};
use super::ros1::reader::Ros1MessageIter;
use super::ros1::{Ros1Compression, Ros1Reader, Ros1Writer};
use super::ros2::reader::Reader;
use super::ros2::storage::StorageWriter;
use super::ros2::storage::mcap::McapWriter;
use super::ros2::writer::Writer;
use super::types::{
    CompressionFormat, CompressionMode, Connection, MessageDefinition, MessageDefinitionFormat,
    QosProfile, StoragePlugin,
};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

const ROS1_MAGIC: &[u8] = b"#ROSBAG V2.0\n";
const MCAP_MAGIC: &[u8] = b"\x89MCAP0\r\n";
const VRS_MAGIC: &[u8] = b"VisionRe";

/// Container flavour of a bag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BagKind {
    /// ROS1 `.bag` file.
    Ros1Bag,
    /// ROS2 bag directory with sqlite3 storage.
    Ros2Sqlite,
    /// ROS2 bag directory with MCAP storage.
    Ros2Mcap,
    /// A single bare `.mcap` file (no `metadata.yaml`).
    McapFile,
    /// A Meta VRS file (Project Aria recording); needs the `vrs` feature.
    Vrs,
}

impl BagKind {
    /// Parse a user-facing name: `bag`, `sqlite3`, `mcap` (directory) or
    /// `mcap-file`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "bag" | "ros1" => Some(Self::Ros1Bag),
            "sqlite3" | "db3" => Some(Self::Ros2Sqlite),
            "mcap" => Some(Self::Ros2Mcap),
            "mcap-file" | "mcap_file" => Some(Self::McapFile),
            "vrs" => Some(Self::Vrs),
            _ => None,
        }
    }

    /// Guess the kind to create at `path` from its extension: `.bag` →
    /// ROS1, `.mcap` → bare MCAP, anything else → `default` (a directory).
    pub fn from_output_path(path: &Path, default: Self) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some(e) if e.eq_ignore_ascii_case("bag") => Self::Ros1Bag,
            Some(e) if e.eq_ignore_ascii_case("mcap") => Self::McapFile,
            Some(e) if e.eq_ignore_ascii_case("vrs") => Self::Vrs,
            _ => default,
        }
    }

    /// Short name, as accepted by [`BagKind::parse`].
    pub fn name(self) -> &'static str {
        match self {
            Self::Ros1Bag => "bag",
            Self::Ros2Sqlite => "sqlite3",
            Self::Ros2Mcap => "mcap",
            Self::McapFile => "mcap-file",
            Self::Vrs => "vrs",
        }
    }
}

/// Identify an existing bag by looking at the file magic / directory layout.
pub fn detect_kind(path: &Path) -> Result<BagKind> {
    if !path.exists() {
        return Err(BagError::BagNotFound {
            path: path.to_path_buf(),
        });
    }
    if path.is_file() {
        let mut magic = [0u8; 13];
        let n = std::fs::File::open(path)?.read(&mut magic)?;
        if magic[..n].starts_with(ROS1_MAGIC) {
            return Ok(BagKind::Ros1Bag);
        }
        if magic[..n].starts_with(MCAP_MAGIC) {
            return Ok(BagKind::McapFile);
        }
        if magic[..n].starts_with(VRS_MAGIC) {
            return Ok(BagKind::Vrs);
        }
        return Err(BagError::UnsupportedStorageFormat {
            format: format!("'{}' is not a ROS1 bag, MCAP or VRS file", path.display()),
        });
    }
    let metadata = path.join("metadata.yaml");
    if metadata.exists() {
        let meta = super::ros2::metadata::BagMetadata::from_file(&metadata)?;
        return match meta.info().storage_identifier.as_str() {
            "mcap" => Ok(BagKind::Ros2Mcap),
            "sqlite3" => Ok(BagKind::Ros2Sqlite),
            other => Err(BagError::UnsupportedStorageFormat {
                format: other.to_string(),
            }),
        };
    }
    let has_ext = |ext: &str| {
        std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| e.path().extension().is_some_and(|x| x == ext))
    };
    if has_ext("db3") {
        Ok(BagKind::Ros2Sqlite)
    } else if has_ext("mcap") {
        Ok(BagKind::Ros2Mcap)
    } else {
        Err(BagError::UnsupportedStorageFormat {
            format: format!("'{}' contains no bag", path.display()),
        })
    }
}

/// Which messages to read.
#[derive(Debug, Clone, Default)]
pub struct Selection {
    /// Topics to include; empty means all.
    pub include: Vec<String>,
    /// Topics to exclude.
    pub exclude: Vec<String>,
    /// First timestamp (ns, inclusive).
    pub start_ns: Option<u64>,
    /// Last timestamp (ns, exclusive).
    pub end_ns: Option<u64>,
}

impl Selection {
    /// Does `topic` pass the include/exclude lists?
    pub fn accepts_topic(&self, topic: &str) -> bool {
        (self.include.is_empty() || self.include.iter().any(|t| t == topic))
            && !self.exclude.iter().any(|t| t == topic)
    }
}

/// Configuration of one sensor stream (cameras and IMUs of a VRS file).
#[derive(Debug, Clone, PartialEq)]
pub enum SensorConfig {
    /// A camera.
    Camera {
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
        /// Row stride in bytes.
        stride: u32,
        /// Nominal frame rate in Hz.
        nominal_rate_hz: f64,
    },
    /// An IMU or magnetometer unit.
    Imu {
        /// Nominal sampling rate in Hz.
        nominal_rate_hz: f64,
        /// Has an accelerometer.
        has_accelerometer: bool,
        /// Has a gyroscope.
        has_gyroscope: bool,
        /// Has a magnetometer.
        has_magnetometer: bool,
    },
}

/// One sensor's configuration, keyed by the topic it is exposed as.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorParams {
    /// Topic the sensor's messages appear on.
    pub topic: String,
    /// `frame_id` used in the messages' headers.
    pub frame_id: String,
    /// Device/sensor serial number, when known.
    pub serial: String,
    /// The configuration.
    pub config: SensorConfig,
}

/// Parameters carried by a container, gathered without reading messages.
///
/// Only the fields the container actually has are filled; the rest stay
/// empty. Calibration is passed through as JSON so nothing is lost when a
/// vendor adds fields.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ContainerParams {
    /// ROS2 directory bags: storage plugin (`sqlite3`/`mcap`).
    pub storage_identifier: Option<String>,
    /// ROS2 directory bags: ROS distribution, when recorded.
    pub ros_distro: Option<String>,
    /// ROS2 directory bags: compression as `mode/format`, when compressed.
    pub compression: Option<String>,
    /// ROS2 directory bags: the `custom_data` map of `metadata.yaml`.
    pub custom_data: BTreeMap<String, String>,
    /// VRS: file-level tags (device, session id, capture time, ...).
    pub file_tags: BTreeMap<String, String>,
    /// VRS: factory calibration (cameras, IMUs, extrinsics) as JSON.
    pub calibration: Option<serde_json::Value>,
    /// VRS: camera and IMU configurations.
    pub sensors: Vec<SensorParams>,
}

/// One message pulled from a [`BagIterator`].
#[derive(Debug, Clone, PartialEq)]
pub struct BagMessage {
    /// Id of the message's connection (see [`BagIterator::connection`]).
    pub connection_id: u32,
    /// Timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Serialized payload.
    pub data: Vec<u8>,
}

enum IterImpl<'a> {
    Ros1(Ros1MessageIter<'a>),
    Ros2(Box<dyn Iterator<Item = Result<super::types::RawMessage>> + 'a>),
    #[cfg(feature = "vrs")]
    Vrs {
        reader: &'a mut super::vrs_bag::VrsBagReader,
        state: super::vrs_bag::VrsPull,
    },
    Empty,
}

/// Pulls the messages of a [`Selection`] one at a time, in timestamp order,
/// from any container. Created by [`BagReader::iter`].
///
/// Unlike [`BagReader::for_each`] it keeps its own position, so a consumer
/// can interleave pulls with its own state:
///
/// ```no_run
/// # use apex_io::rosbag::{BagReader, Selection};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut bag = BagReader::open("recording.bag")?;
/// let selection = Selection { include: vec!["/imu0".into()], ..Selection::default() };
/// let mut it = bag.iter(&selection)?;
/// let mut buf = Vec::new();
/// while let Some((conn, t_ns)) = it.next_into(&mut buf)? {
///     let _topic = &it.connection(conn).map(|c| c.topic.clone());
///     let _ = (t_ns, buf.len());
/// }
/// # Ok(())
/// # }
/// ```
pub struct BagIterator<'a> {
    inner: IterImpl<'a>,
    connections: Vec<Connection>,
}

impl BagIterator<'_> {
    /// The connections this iterator draws from.
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Look up a connection by the id returned with each message.
    pub fn connection(&self, id: u32) -> Option<&Connection> {
        self.connections.iter().find(|c| c.id == id)
    }

    /// Move the next message's payload into `buf` (its previous contents are
    /// replaced) and return `(connection id, timestamp ns)`; `None` at the
    /// end. Reuses the caller's buffer where the backend allows.
    pub fn next_into(&mut self, buf: &mut Vec<u8>) -> Result<Option<(u32, u64)>> {
        match &mut self.inner {
            IterImpl::Empty => Ok(None),
            IterImpl::Ros1(it) => it.next_into(buf),
            IterImpl::Ros2(it) => match it.next() {
                Some(msg) => {
                    let msg = msg?;
                    *buf = msg.raw_data;
                    Ok(Some((msg.connection.id, msg.timestamp)))
                }
                None => Ok(None),
            },
            #[cfg(feature = "vrs")]
            IterImpl::Vrs { reader, state } => Ok(reader
                .pull(state, buf)?
                .map(|(i, ts)| (reader.connection_at(i).id, ts))),
        }
    }

    /// Pull the next message as an owned value.
    pub fn next_message(&mut self) -> Result<Option<BagMessage>> {
        let mut data = Vec::new();
        Ok(self
            .next_into(&mut data)?
            .map(|(connection_id, timestamp_ns)| BagMessage {
                connection_id,
                timestamp_ns,
                data,
            }))
    }
}

impl Iterator for BagIterator<'_> {
    type Item = Result<BagMessage>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_message().transpose()
    }
}

enum ReaderImpl {
    Ros1(Box<Ros1Reader>),
    Ros2(Box<Reader>),
    #[cfg(feature = "vrs")]
    Vrs(Box<super::vrs_bag::VrsBagReader>),
}

/// Reader over any [`BagKind`].
pub struct BagReader {
    kind: BagKind,
    inner: ReaderImpl,
    connections: Vec<Connection>,
    start_ns: u64,
    end_ns: u64,
    message_count: u64,
}

impl BagReader {
    /// Detect the container at `path` and open it.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let kind = detect_kind(path)?;
        match kind {
            BagKind::Ros1Bag => {
                let mut r = Ros1Reader::new(path)?;
                r.open()?;
                Ok(Self {
                    kind,
                    connections: r.connections().to_vec(),
                    start_ns: r.start_time(),
                    end_ns: r.end_time(),
                    message_count: r.message_count(),
                    inner: ReaderImpl::Ros1(Box::new(r)),
                })
            }
            BagKind::Vrs => {
                #[cfg(feature = "vrs")]
                {
                    let r = super::vrs_bag::VrsBagReader::open(path)?;
                    Ok(Self {
                        kind,
                        connections: r.connections().to_vec(),
                        start_ns: r.start_ns(),
                        end_ns: r.end_ns(),
                        message_count: r.message_count(),
                        inner: ReaderImpl::Vrs(Box::new(r)),
                    })
                }
                #[cfg(not(feature = "vrs"))]
                {
                    Err(BagError::UnsupportedStorageFormat {
                        format: "VRS (rebuild apex-io with the `vrs` feature)".into(),
                    })
                }
            }
            _ => {
                let mut r = Reader::new(path)?;
                r.open()?;
                Ok(Self {
                    kind,
                    connections: r.connections().to_vec(),
                    start_ns: r.start_time(),
                    end_ns: r.end_time(),
                    message_count: r.message_count(),
                    inner: ReaderImpl::Ros2(Box::new(r)),
                })
            }
        }
    }

    /// Container flavour.
    pub fn kind(&self) -> BagKind {
        self.kind
    }

    /// All connections (one per topic/type pair).
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Earliest message time (ns).
    pub fn start_time(&self) -> u64 {
        self.start_ns
    }

    /// Latest message time (ns).
    pub fn end_time(&self) -> u64 {
        self.end_ns
    }

    /// Total number of messages.
    pub fn message_count(&self) -> u64 {
        self.message_count
    }

    /// Connections that pass `selection`'s topic lists.
    pub fn selected_connections(&self, selection: &Selection) -> Vec<Connection> {
        self.connections
            .iter()
            .filter(|c| selection.accepts_topic(&c.topic))
            .cloned()
            .collect()
    }

    /// Pull the messages of `selection` one at a time in timestamp order.
    /// See [`BagIterator`].
    pub fn iter(&mut self, selection: &Selection) -> Result<BagIterator<'_>> {
        let connections = self.selected_connections(selection);
        if connections.is_empty() {
            return Ok(BagIterator {
                inner: IterImpl::Empty,
                connections,
            });
        }
        let inner = match &mut self.inner {
            ReaderImpl::Ros1(r) => IterImpl::Ros1(r.raw_iter(
                Some(&connections),
                selection.start_ns,
                selection.end_ns,
            )?),
            ReaderImpl::Ros2(r) => IterImpl::Ros2(r.raw_messages_filtered(
                Some(&connections),
                selection.start_ns,
                selection.end_ns,
            )?),
            #[cfg(feature = "vrs")]
            ReaderImpl::Vrs(r) => {
                let state = r.begin(&connections, selection.start_ns, selection.end_ns);
                IterImpl::Vrs { reader: r, state }
            }
        };
        Ok(BagIterator { inner, connections })
    }

    /// Parameters the container carries: ROS2 `metadata.yaml` facts, or for
    /// VRS the file tags, factory calibration and sensor configurations.
    pub fn params(&mut self) -> Result<ContainerParams> {
        let mut params = ContainerParams::default();
        match &mut self.inner {
            ReaderImpl::Ros1(_) => {}
            ReaderImpl::Ros2(r) => {
                if let Some(meta) = r.metadata() {
                    let info = meta.info();
                    params.storage_identifier = Some(info.storage_identifier.clone());
                    params.ros_distro = info.ros_distro.clone();
                    if !info.compression_mode.is_empty() {
                        params.compression = Some(format!(
                            "{}/{}",
                            info.compression_mode.to_lowercase(),
                            info.compression_format
                        ));
                    }
                    if let Some(custom) = &info.custom_data {
                        params.custom_data =
                            custom.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                    }
                }
            }
            #[cfg(feature = "vrs")]
            ReaderImpl::Vrs(r) => r.fill_params(&mut params)?,
        }
        Ok(params)
    }

    /// Stream the messages of `connections` inside `selection`'s time window
    /// in timestamp order. `f` receives `(connection, timestamp_ns, payload)`.
    pub fn for_each<F>(
        &mut self,
        connections: &[Connection],
        selection: &Selection,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(&Connection, u64, &[u8]) -> Result<()>,
    {
        if connections.is_empty() {
            return Ok(());
        }
        match &mut self.inner {
            ReaderImpl::Ros1(r) => {
                r.for_each_raw_filtered(Some(connections), selection.start_ns, selection.end_ns, f)
            }
            ReaderImpl::Ros2(r) => {
                for msg in r.raw_messages_filtered(
                    Some(connections),
                    selection.start_ns,
                    selection.end_ns,
                )? {
                    let msg = msg?;
                    f(&msg.connection, msg.timestamp, &msg.raw_data)?;
                }
                Ok(())
            }
            #[cfg(feature = "vrs")]
            ReaderImpl::Vrs(r) => r.for_each(connections, selection.start_ns, selection.end_ns, f),
        }
    }
}

/// A connection to create in an output bag.
#[derive(Debug, Clone)]
pub struct NewConnection {
    /// Topic name.
    pub topic: String,
    /// Type name in the *target* convention (`pkg/Type` for ROS1,
    /// `pkg/msg/Type` for ROS2).
    pub message_type: String,
    /// Message definition text (`.msg` style); may be empty.
    pub definition: String,
    /// ROS1: md5sum; ROS2: type description hash. May be empty.
    pub type_hash: String,
    /// Payload encoding: `ros1` or `cdr`.
    pub serialization_format: String,
    /// QoS profiles (ROS2 only).
    pub qos: Vec<QosProfile>,
}

impl NewConnection {
    /// Copy a source connection unchanged.
    pub fn from_connection(c: &Connection) -> Self {
        Self {
            topic: c.topic.clone(),
            message_type: c.message_type.clone(),
            definition: c.message_definition.data.clone(),
            type_hash: c.type_description_hash.clone(),
            serialization_format: c.serialization_format.clone(),
            qos: c.offered_qos_profiles.clone(),
        }
    }
}

/// Output compression, applied where the container supports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputCompression {
    /// Store uncompressed.
    None,
    /// zstd: MCAP chunks, or the whole file for sqlite3 bags.
    Zstd,
    /// LZ4 chunks (ROS1 only).
    Lz4,
    /// bzip2 chunks (ROS1 only).
    Bz2,
}

impl OutputCompression {
    /// Parse `none|zstd|lz4|bz2`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "none" => Some(Self::None),
            "zstd" => Some(Self::Zstd),
            "lz4" => Some(Self::Lz4),
            "bz2" => Some(Self::Bz2),
            _ => None,
        }
    }
}

enum WriterImpl {
    Ros1(Box<Ros1Writer>),
    Ros2(Box<Writer>),
    McapFile(Box<McapWriter>),
    #[cfg(feature = "vrs")]
    Vrs(Box<super::vrs_bag::VrsBagWriter>),
}

/// Writer for any [`BagKind`].
pub struct BagWriter {
    kind: BagKind,
    inner: WriterImpl,
    connections: Vec<Connection>,
    closed: bool,
}

/// Handle to a connection added with [`BagWriter::add_connection`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConnectionHandle(usize);

impl BagWriter {
    /// Create a new bag of `kind` at `path` (which must not exist).
    pub fn create(
        path: impl AsRef<Path>,
        kind: BagKind,
        compression: OutputCompression,
    ) -> Result<Self> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let unsupported =
            |what: &str| BagError::writer(format!("{} does not support {what}", kind.name()));
        let inner = match kind {
            BagKind::Ros1Bag => {
                let mut w = Ros1Writer::new(&path)?;
                w.set_compression(match compression {
                    OutputCompression::None => Ros1Compression::None,
                    OutputCompression::Lz4 => Ros1Compression::Lz4,
                    OutputCompression::Bz2 => Ros1Compression::Bz2,
                    OutputCompression::Zstd => return Err(unsupported("zstd compression")),
                })?;
                w.open()?;
                WriterImpl::Ros1(Box::new(w))
            }
            BagKind::Ros2Sqlite | BagKind::Ros2Mcap => {
                let plugin = if kind == BagKind::Ros2Mcap {
                    StoragePlugin::Mcap
                } else {
                    StoragePlugin::Sqlite3
                };
                let mut w = Writer::new(&path, None, Some(plugin))?;
                match (compression, kind) {
                    (OutputCompression::None, _) => {}
                    (OutputCompression::Zstd, BagKind::Ros2Mcap) => {
                        w.set_compression(CompressionMode::Storage, CompressionFormat::Zstd)?;
                    }
                    (OutputCompression::Zstd, _) => {
                        w.set_compression(CompressionMode::File, CompressionFormat::Zstd)?;
                    }
                    _ => return Err(unsupported("lz4/bz2 compression")),
                }
                w.open()?;
                WriterImpl::Ros2(Box::new(w))
            }
            BagKind::Vrs => {
                #[cfg(feature = "vrs")]
                {
                    WriterImpl::Vrs(Box::new(super::vrs_bag::VrsBagWriter::create(
                        &path,
                        compression,
                    )?))
                }
                #[cfg(not(feature = "vrs"))]
                {
                    return Err(BagError::UnsupportedStorageFormat {
                        format: "VRS (rebuild apex-io with the `vrs` feature)".into(),
                    });
                }
            }
            BagKind::McapFile => {
                if path.exists() {
                    return Err(BagError::BagAlreadyExists { path });
                }
                let mode = match compression {
                    OutputCompression::None => CompressionMode::None,
                    OutputCompression::Zstd => CompressionMode::Storage,
                    _ => return Err(unsupported("lz4/bz2 compression")),
                };
                let mut w = McapWriter::with_file(path, "ros2", mode);
                w.open()?;
                WriterImpl::McapFile(Box::new(w))
            }
        };
        Ok(Self {
            kind,
            inner,
            connections: Vec::new(),
            closed: false,
        })
    }

    /// Container flavour being written.
    pub fn kind(&self) -> BagKind {
        self.kind
    }

    /// Register a topic. Returns a handle for [`Self::write`].
    pub fn add_connection(&mut self, new: &NewConnection) -> Result<ConnectionHandle> {
        // Bags such as rosbag2 sqlite store no schema; fill it in for
        // standard types so the output is self-describing.
        let mut new = std::borrow::Cow::Borrowed(new);
        if new.definition.is_empty() {
            let dialect = if new.serialization_format == "ros1" {
                super::transcode::Dialect::Ros1
            } else {
                super::transcode::Dialect::Ros2
            };
            if let Some(text) = super::transcode::catalog_definition(&new.message_type, dialect) {
                new.to_mut().definition = text;
            }
        }
        let new = new.as_ref();
        let definition = MessageDefinition {
            format: if new.definition.is_empty() {
                MessageDefinitionFormat::None
            } else {
                MessageDefinitionFormat::Msg
            },
            data: new.definition.clone(),
        };
        let connection = match &mut self.inner {
            WriterImpl::Ros1(w) => w.add_connection(
                new.topic.clone(),
                new.message_type.clone(),
                new.type_hash.clone(),
                new.definition.clone(),
                false,
            )?,
            WriterImpl::Ros2(w) => w.add_connection(
                new.topic.clone(),
                new.message_type.clone(),
                Some(definition),
                Some(new.type_hash.clone()),
                Some(new.serialization_format.clone()),
                Some(new.qos.clone()),
            )?,
            WriterImpl::McapFile(w) => {
                let connection = Connection {
                    id: u32::try_from(self.connections.len() + 1)
                        .map_err(|_| BagError::writer("too many connections"))?,
                    topic: new.topic.clone(),
                    message_type: new.message_type.clone(),
                    message_definition: definition,
                    type_description_hash: new.type_hash.clone(),
                    message_count: 0,
                    serialization_format: new.serialization_format.clone(),
                    offered_qos_profiles: new.qos.clone(),
                };
                w.add_msgtype(&connection)?;
                w.add_connection(&connection, "")?;
                connection
            }
            #[cfg(feature = "vrs")]
            WriterImpl::Vrs(w) => {
                w.add_stream(
                    &new.topic,
                    &new.message_type,
                    &new.definition,
                    &new.serialization_format,
                    &new.type_hash,
                )?;
                Connection {
                    id: u32::try_from(self.connections.len() + 1)
                        .map_err(|_| BagError::writer("too many connections"))?,
                    topic: new.topic.clone(),
                    message_type: new.message_type.clone(),
                    message_definition: definition,
                    type_description_hash: new.type_hash.clone(),
                    message_count: 0,
                    serialization_format: new.serialization_format.clone(),
                    offered_qos_profiles: new.qos.clone(),
                }
            }
        };
        self.connections.push(connection);
        Ok(ConnectionHandle(self.connections.len() - 1))
    }

    /// Write one message verbatim.
    pub fn write(
        &mut self,
        handle: ConnectionHandle,
        timestamp_ns: u64,
        data: &[u8],
    ) -> Result<()> {
        let connection = self
            .connections
            .get(handle.0)
            .ok_or_else(|| BagError::connection_not_found(format!("handle {}", handle.0)))?;
        match &mut self.inner {
            WriterImpl::Ros1(w) => w.write(connection, timestamp_ns, data),
            WriterImpl::Ros2(w) => w.write_raw_message(connection, timestamp_ns, data),
            WriterImpl::McapFile(w) => w.write(connection, timestamp_ns, data),
            #[cfg(feature = "vrs")]
            WriterImpl::Vrs(w) => w.write(handle.0, timestamp_ns, data),
        }
    }

    /// Finish the bag (index/summary/metadata). Idempotent.
    pub fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        match &mut self.inner {
            WriterImpl::Ros1(w) => w.close(),
            WriterImpl::Ros2(w) => w.close(),
            WriterImpl::McapFile(w) => w.close(0, ""),
            #[cfg(feature = "vrs")]
            WriterImpl::Vrs(w) => w.finish(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    fn conn(topic: &str, ty: &str, fmt: &str) -> NewConnection {
        NewConnection {
            topic: topic.into(),
            message_type: ty.into(),
            definition: String::new(),
            type_hash: String::new(),
            serialization_format: fmt.into(),
            qos: Vec::new(),
        }
    }

    /// Every kind can be written, sniffed, and read back identically.
    #[test]
    fn every_kind_round_trips() -> TestResult {
        #[allow(unused_mut)]
        let mut cases = vec![
            (BagKind::Ros1Bag, "out.bag", "pkg/Type", "ros1"),
            (BagKind::Ros2Sqlite, "sq", "pkg/msg/Type", "cdr"),
            (BagKind::Ros2Mcap, "mc", "pkg/msg/Type", "cdr"),
            (BagKind::McapFile, "out.mcap", "pkg/msg/Type", "cdr"),
        ];
        #[cfg(feature = "vrs")]
        cases.push((BagKind::Vrs, "out.vrs", "pkg/msg/Type", "cdr"));
        for (kind, name, ty, fmt) in cases {
            let dir = tempdir()?;
            let path = dir.path().join(name);
            let mut w = BagWriter::create(&path, kind, OutputCompression::None)?;
            let a = w.add_connection(&conn("/a", ty, fmt))?;
            let b = w.add_connection(&conn("/b", ty, fmt))?;
            for i in 0..300u64 {
                w.write(a, 10 + i, &i.to_le_bytes())?;
                w.write(b, 10 + i, &[7u8; 5])?;
            }
            w.close()?;

            assert_eq!(detect_kind(&path)?, kind, "{kind:?}");
            let mut r = BagReader::open(&path)?;
            assert_eq!(r.message_count(), 600, "{kind:?}");
            let sel = Selection {
                include: vec!["/a".into()],
                start_ns: Some(110),
                end_ns: Some(210),
                ..Selection::default()
            };
            let conns = r.selected_connections(&sel);
            assert_eq!(conns.len(), 1);
            let mut got = Vec::new();
            r.for_each(&conns, &sel, |c, ts, d| {
                assert_eq!(c.topic, "/a");
                got.push((ts, d.to_vec()));
                Ok(())
            })?;
            assert_eq!(got.len(), 100, "{kind:?}");
            assert_eq!(got[0], (110, 100u64.to_le_bytes().to_vec()), "{kind:?}");
        }
        Ok(())
    }

    #[test]
    fn compressed_outputs_read_back() -> TestResult {
        for (kind, name, c) in [
            (BagKind::Ros1Bag, "z.bag", OutputCompression::Lz4),
            (BagKind::Ros1Bag, "b.bag", OutputCompression::Bz2),
            (BagKind::McapFile, "z.mcap", OutputCompression::Zstd),
        ] {
            let dir = tempdir()?;
            let path = dir.path().join(name);
            let fmt = if kind == BagKind::Ros1Bag {
                "ros1"
            } else {
                "cdr"
            };
            let mut w = BagWriter::create(&path, kind, c)?;
            let h = w.add_connection(&conn("/t", "pkg/Type", fmt))?;
            for i in 0..1000u64 {
                w.write(h, i, &[i as u8; 64])?;
            }
            w.close()?;
            let mut r = BagReader::open(&path)?;
            let conns = r.connections().to_vec();
            let mut n = 0u64;
            r.for_each(&conns, &Selection::default(), |_, ts, d| {
                assert_eq!(d, &[ts as u8; 64][..]);
                n += 1;
                Ok(())
            })?;
            assert_eq!(n, 1000, "{name}");
        }
        Ok(())
    }

    #[test]
    fn standard_types_get_a_definition_when_the_source_has_none() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("d.mcap");
        let mut w = BagWriter::create(&path, BagKind::McapFile, OutputCompression::None)?;
        w.add_connection(&conn("/imu", "sensor_msgs/msg/Imu", "cdr"))?;
        w.add_connection(&conn("/custom", "my_pkg/msg/Thing", "cdr"))?;
        w.close()?;
        let r = BagReader::open(&path)?;
        let by_topic = |t: &str| r.connections().iter().find(|c| c.topic == t).cloned();
        let imu = by_topic("/imu").ok_or("imu")?;
        assert!(imu.message_definition.data.contains("angular_velocity"));
        let custom = by_topic("/custom").ok_or("custom")?;
        assert!(custom.message_definition.data.is_empty());
        Ok(())
    }

    #[test]
    fn unsupported_combinations_are_rejected() {
        let dir = tempdir().unwrap_or_else(|e| panic!("{e}"));
        assert!(
            BagWriter::create(
                dir.path().join("a.bag"),
                BagKind::Ros1Bag,
                OutputCompression::Zstd
            )
            .is_err()
        );
        assert!(
            BagWriter::create(
                dir.path().join("b.mcap"),
                BagKind::McapFile,
                OutputCompression::Bz2
            )
            .is_err()
        );
    }

    /// `(connection id, stamp, payload)` triples via the callback API.
    fn via_for_each(r: &mut BagReader, sel: &Selection) -> Vec<(u32, u64, Vec<u8>)> {
        let conns = r.selected_connections(sel);
        let mut out = Vec::new();
        r.for_each(&conns, sel, |c, ts, d| {
            out.push((c.id, ts, d.to_vec()));
            Ok(())
        })
        .unwrap();
        out
    }

    /// The pull iterator is the same stream as `for_each`, for every
    /// container, whatever the filter.
    #[test]
    fn pull_iterator_matches_for_each_for_every_kind() -> TestResult {
        #[allow(unused_mut)]
        let mut cases = vec![
            (BagKind::Ros1Bag, "out.bag", "pkg/Type", "ros1"),
            (BagKind::Ros2Sqlite, "sq", "pkg/msg/Type", "cdr"),
            (BagKind::Ros2Mcap, "mc", "pkg/msg/Type", "cdr"),
            (BagKind::McapFile, "out.mcap", "pkg/msg/Type", "cdr"),
        ];
        #[cfg(feature = "vrs")]
        cases.push((BagKind::Vrs, "out.vrs", "pkg/msg/Type", "cdr"));

        for (kind, name, ty, fmt) in cases {
            let dir = tempdir()?;
            let path = dir.path().join(name);
            let mut w = BagWriter::create(&path, kind, OutputCompression::None)?;
            let a = w.add_connection(&conn("/a", ty, fmt))?;
            let b = w.add_connection(&conn("/b", ty, fmt))?;
            for i in 0..400u64 {
                w.write(a, 1_000 + i * 10, &i.to_le_bytes())?;
                if i % 3 == 0 {
                    w.write(b, 1_005 + i * 10, &vec![i as u8; 1 + (i as usize % 50)])?;
                }
            }
            w.close()?;

            let mut r = BagReader::open(&path)?;
            let selections = [
                Selection::default(),
                Selection {
                    include: vec!["/b".into()],
                    ..Selection::default()
                },
                Selection {
                    start_ns: Some(2_000),
                    end_ns: Some(3_000),
                    ..Selection::default()
                },
                Selection {
                    exclude: vec!["/a".into()],
                    start_ns: Some(1_500),
                    ..Selection::default()
                },
            ];
            for sel in &selections {
                let expect = via_for_each(&mut r, sel);
                let mut pulled = Vec::new();
                let mut it = r.iter(sel)?;
                let mut buf = Vec::new();
                while let Some((id, ts)) = it.next_into(&mut buf)? {
                    assert!(it.connection(id).is_some(), "{kind:?}: unknown id {id}");
                    pulled.push((id, ts, buf.clone()));
                }
                assert_eq!(pulled, expect, "{kind:?} {sel:?}");
                assert!(!expect.is_empty(), "{kind:?} {sel:?} selected nothing");
            }

            // owned-message and Iterator forms agree too
            let all = Selection::default();
            let via_iter: Vec<_> = r.iter(&all)?.collect::<Result<Vec<_>>>()?;
            let expect = via_for_each(&mut r, &all);
            assert_eq!(via_iter.len(), expect.len(), "{kind:?}");
            assert_eq!(via_iter[7].data, expect[7].2, "{kind:?}");

            // nothing selected => an empty iterator, not an error
            let none = Selection {
                include: vec!["/nope".into()],
                ..Selection::default()
            };
            assert!(r.iter(&none)?.next_message()?.is_none());
        }
        Ok(())
    }

    #[test]
    fn ros2_directory_params_expose_metadata() -> TestResult {
        use super::super::types::{CompressionFormat, CompressionMode};
        let dir = tempdir()?;
        let path = dir.path().join("p");
        let mut w = Writer::new(&path, None, Some(StoragePlugin::Mcap))?;
        w.set_compression(CompressionMode::Message, CompressionFormat::Zstd)?;
        w.set_custom_data("operator".into(), "amin".into())?;
        w.open()?;
        let c = w.add_connection("/t".into(), "pkg/msg/T".into(), None, None, None, None)?;
        w.write(&c, 5, b"x")?;
        w.close()?;

        let mut r = BagReader::open(&path)?;
        let p = r.params()?;
        assert_eq!(p.storage_identifier.as_deref(), Some("mcap"));
        assert_eq!(p.compression.as_deref(), Some("message/zstd"));
        assert_eq!(
            p.custom_data.get("operator").map(String::as_str),
            Some("amin")
        );
        assert!(p.sensors.is_empty() && p.calibration.is_none());
        Ok(())
    }
}
