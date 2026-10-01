//! VRS as a bag container (feature `vrs`).
//!
//! **Reading** a Project Aria recording exposes its sensors as standard ROS
//! topics with ROS1-encoded payloads (the usual transcoder then produces CDR
//! for ROS2 outputs):
//!
//! | VRS stream            | Topic                         | Type                         |
//! |-----------------------|-------------------------------|------------------------------|
//! | 214-1 RGB camera      | `/aria/camera_rgb/compressed` | `sensor_msgs/CompressedImage`|
//! | 1201-1/2 SLAM cameras | `/aria/camera_slam_{left,right}/compressed` | same           |
//! | 1202-1/2 IMUs         | `/aria/imu_{right,left}`      | `sensor_msgs/Imu`            |
//! | 1203-1 magnetometer   | `/aria/magnetometer`          | `sensor_msgs/MagneticField`  |
//! | 247-1 barometer       | `/aria/barometer`             | `sensor_msgs/FluidPressure`  |
//! | 281-1 GPS             | `/aria/gps`                   | `sensor_msgs/NavSatFix`      |
//!
//! Message stamps are the sensors' capture times on the device clock.
//! Other streams (Wi-Fi, Bluetooth, time-domain map, ...) are listed with
//! `serialization_format = "vrs"` so converters can warn about them.
//!
//! **Writing** stores each ROS topic as a VRS stream holding the raw
//! message bytes together with topic, type, definition and encoding, so a
//! bag survives a round trip through VRS byte for byte, whatever the types.

use super::bag::OutputCompression;
use super::error::{BagError, Result};
use super::transcode::{Dialect, catalog_definition, catalog_md5};
use super::types::{Connection, MessageDefinition, MessageDefinitionFormat};
use crate::vrs::aria::{AriaRecording, AriaSensor, Sample};
use crate::vrs::datalayout::{DataLayout, LayoutWriter, Piece, PieceKind, Scalar};
use crate::vrs::{
    Block, Compression, Cursor, RecordFilter, RecordType, StreamId, StreamTags, VrsError,
    VrsWriteOptions, VrsWriter,
};
use std::collections::BTreeMap;
use std::path::Path;

fn vrs_err(e: VrsError) -> BagError {
    BagError::generic(format!("VRS: {e}"))
}

const TAG_TOPIC: &str = "apex_ros_topic";
const TAG_TYPE: &str = "apex_ros_type";
const TAG_DEFINITION: &str = "apex_ros_definition";
const TAG_ENCODING: &str = "apex_ros_serialization";
const TAG_HASH: &str = "apex_ros_hash";
/// First stream type id used for ROS topics written by [`VrsBagWriter`].
const WRITER_BASE_TYPE_ID: i32 = 7000;

/// How a connection's records are turned into messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// A known Aria sensor mapped to a standard ROS type.
    Aria(AriaSensor),
    /// A stream written by [`VrsBagWriter`]: raw message bytes.
    Native,
    /// Anything else; reported, but not convertible.
    Raw,
}

struct Mapped {
    topic: &'static str,
    message_type: &'static str,
    frame_id: &'static str,
}

fn aria_mapping(sensor: AriaSensor) -> Option<Mapped> {
    let m = |topic, message_type, frame_id| {
        Some(Mapped {
            topic,
            message_type,
            frame_id,
        })
    };
    match sensor {
        AriaSensor::Rgb => m(
            "/aria/camera_rgb/compressed",
            "sensor_msgs/CompressedImage",
            "camera-rgb",
        ),
        AriaSensor::SlamLeft => m(
            "/aria/camera_slam_left/compressed",
            "sensor_msgs/CompressedImage",
            "camera-slam-left",
        ),
        AriaSensor::SlamRight => m(
            "/aria/camera_slam_right/compressed",
            "sensor_msgs/CompressedImage",
            "camera-slam-right",
        ),
        AriaSensor::ImuRight => m("/aria/imu_right", "sensor_msgs/Imu", "imu-right"),
        AriaSensor::ImuLeft => m("/aria/imu_left", "sensor_msgs/Imu", "imu-left"),
        AriaSensor::Magnetometer => m("/aria/magnetometer", "sensor_msgs/MagneticField", "mag"),
        AriaSensor::Barometer => m("/aria/barometer", "sensor_msgs/FluidPressure", "baro"),
        AriaSensor::Gps => m("/aria/gps", "sensor_msgs/NavSatFix", "gps"),
        _ => None,
    }
}

struct Entry {
    stream: StreamId,
    source: Source,
    frame_id: &'static str,
    seq: u32,
}

/// Reader side: an Aria (or ROS-in-VRS) file seen as a bag.
pub(crate) struct VrsBagReader {
    rec: AriaRecording,
    entries: Vec<Entry>,
    connections: Vec<Connection>,
    start_ns: u64,
    end_ns: u64,
    message_count: u64,
}

impl VrsBagReader {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let rec = AriaRecording::open(path).map_err(vrs_err)?;
        let mut entries = Vec::new();
        let mut connections = Vec::new();
        let (mut start, mut end, mut total) = (f64::INFINITY, f64::NEG_INFINITY, 0u64);

        for stream in rec.vrs().streams() {
            let count = rec.vrs().data_records(stream).len() as u64;
            let Some((first, last)) = rec.vrs().stream_time_range(stream) else {
                continue; // no data records: nothing to expose
            };
            let tags = rec.vrs().stream_tags(stream);
            let user = |k: &str| tags.and_then(|t| t.user.get(k)).cloned();

            let (source, conn, frame_id) = if let Some(topic) = user(TAG_TOPIC) {
                let definition = user(TAG_DEFINITION).unwrap_or_default();
                let conn = connection(
                    topic,
                    user(TAG_TYPE).unwrap_or_default(),
                    definition,
                    user(TAG_HASH).unwrap_or_default(),
                    user(TAG_ENCODING).unwrap_or_else(|| "ros1".into()),
                );
                (Source::Native, conn, "")
            } else if let Some((sensor, map)) =
                AriaSensor::from_stream(stream).and_then(|s| aria_mapping(s).map(|m| (s, m)))
            {
                let conn = connection(
                    map.topic.to_string(),
                    map.message_type.to_string(),
                    catalog_definition(map.message_type, Dialect::Ros1).unwrap_or_default(),
                    catalog_md5(map.message_type).unwrap_or_default(),
                    "ros1".into(),
                );
                (Source::Aria(sensor), conn, map.frame_id)
            } else {
                let conn = connection(
                    format!("/vrs/stream_{}_{}", stream.type_id, stream.instance),
                    format!("vrs/Stream{}", stream.type_id),
                    String::new(),
                    String::new(),
                    "vrs".into(),
                );
                (Source::Raw, conn, "")
            };

            let mut conn = conn;
            conn.id = entries.len() as u32 + 1;
            conn.message_count = count;
            connections.push(conn);
            entries.push(Entry {
                stream,
                source,
                frame_id,
                seq: 0,
            });
            start = start.min(first);
            end = end.max(last);
            total += count;
        }

        let to_ns = |s: f64| {
            if s.is_finite() {
                (s.max(0.0) * 1e9).round() as u64
            } else {
                0
            }
        };
        Ok(Self {
            rec,
            entries,
            connections,
            start_ns: to_ns(start),
            end_ns: to_ns(end),
            message_count: total,
        })
    }

    pub(crate) fn connections(&self) -> &[Connection] {
        &self.connections
    }
    pub(crate) fn start_ns(&self) -> u64 {
        self.start_ns
    }
    pub(crate) fn end_ns(&self) -> u64 {
        self.end_ns
    }
    pub(crate) fn message_count(&self) -> u64 {
        self.message_count
    }

    pub(crate) fn for_each<F>(
        &mut self,
        connections: &[Connection],
        start_ns: Option<u64>,
        end_ns: Option<u64>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(&Connection, u64, &[u8]) -> Result<()>,
    {
        let wanted: Vec<usize> = connections
            .iter()
            .filter_map(|c| (c.id as usize).checked_sub(1))
            .filter(|&i| i < self.entries.len())
            .collect();
        let by_stream: BTreeMap<StreamId, usize> = wanted
            .iter()
            .map(|&i| (self.entries[i].stream, i))
            .collect();

        // Widen the record-time window slightly; exact ns checks follow.
        let mut filter = RecordFilter::streams(by_stream.keys().copied());
        filter.start = start_ns.map(|s| s as f64 / 1e9 - 1e-6);
        filter.end = end_ns.map(|e| e as f64 / 1e9 + 1e-6);
        let mut cursor: Cursor = self.rec.vrs().cursor(&filter);

        let mut buf = Vec::new();
        let mut out = Vec::new();
        while let Some(meta) = self
            .rec
            .vrs_mut()
            .next_record_into(&mut cursor, &mut buf)
            .map_err(vrs_err)?
        {
            let Some(&i) = by_stream.get(&meta.stream) else {
                continue;
            };
            out.clear();
            let ts_ns = match self.entries[i].source {
                Source::Native => match native_message(&self.rec, &meta, &buf, &mut out)? {
                    Some(ts) => ts,
                    None => continue,
                },
                Source::Aria(_) => {
                    let sample = self.rec.decode_record(meta, &buf).map_err(vrs_err)?;
                    let seq = {
                        let e = &mut self.entries[i];
                        e.seq += 1;
                        e.seq - 1
                    };
                    match aria_message(&sample, seq, self.entries[i].frame_id, &mut out) {
                        Some(ts) => ts,
                        None => continue,
                    }
                }
                Source::Raw => {
                    out.extend_from_slice(&buf);
                    (meta.timestamp.max(0.0) * 1e9).round() as u64
                }
            };
            if start_ns.is_some_and(|s| ts_ns < s) || end_ns.is_some_and(|e| ts_ns >= e) {
                continue;
            }
            f(&self.connections[i], ts_ns, &out)?;
        }
        Ok(())
    }
}

fn connection(
    topic: String,
    message_type: String,
    definition: String,
    hash: String,
    serialization_format: String,
) -> Connection {
    Connection {
        id: 0,
        topic,
        message_type,
        message_definition: MessageDefinition {
            format: if definition.is_empty() {
                MessageDefinitionFormat::None
            } else {
                MessageDefinitionFormat::Msg
            },
            data: definition,
        },
        type_description_hash: hash,
        message_count: 0,
        serialization_format,
        offered_qos_profiles: Vec::new(),
    }
}

// ---- Aria sample -> ROS1 message bytes ----------------------------------------

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_f64(b: &mut Vec<u8>, v: f64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u32(b, s.len() as u32);
    b.extend_from_slice(s.as_bytes());
}
fn put_header(b: &mut Vec<u8>, seq: u32, capture_ns: i64, frame_id: &str) {
    let ns = capture_ns.max(0) as u64;
    put_u32(b, seq);
    put_u32(b, (ns / 1_000_000_000) as u32);
    put_u32(b, (ns % 1_000_000_000) as u32);
    put_str(b, frame_id);
}
fn put_vec3(b: &mut Vec<u8>, v: [f32; 3]) {
    for x in v {
        put_f64(b, f64::from(x));
    }
}
fn put_cov(b: &mut Vec<u8>, first: f64) {
    put_f64(b, first);
    for _ in 1..9 {
        put_f64(b, 0.0);
    }
}

/// Build the ROS1 message for `sample`; returns its stamp in ns, or `None`
/// when the sample carries no usable data (for example an invalid IMU read).
fn aria_message(sample: &Sample, seq: u32, frame_id: &str, out: &mut Vec<u8>) -> Option<u64> {
    match sample {
        Sample::Image(f) => {
            put_header(out, seq, f.capture_ns, frame_id);
            put_str(out, "jpeg");
            put_u32(out, f.jpeg.len() as u32);
            out.extend_from_slice(&f.jpeg);
            Some(f.capture_ns.max(0) as u64)
        }
        Sample::Imu(s) if s.stream.type_id == 1202 => {
            let (accel, gyro) = (s.accel?, s.gyro?);
            put_header(out, seq, s.capture_ns, frame_id);
            for q in [0.0, 0.0, 0.0, 1.0] {
                put_f64(out, q); // orientation unknown...
            }
            put_cov(out, -1.0); // ...flagged by covariance[0] = -1 (REP 145)
            put_vec3(out, gyro);
            put_cov(out, 0.0);
            put_vec3(out, accel);
            put_cov(out, 0.0);
            Some(s.capture_ns.max(0) as u64)
        }
        Sample::Imu(s) => {
            // Magnetometer (stored in microtesla).
            let mag = s.mag?;
            put_header(out, seq, s.capture_ns, frame_id);
            put_vec3(out, [mag[0] * 1e-6, mag[1] * 1e-6, mag[2] * 1e-6]);
            put_cov(out, 0.0);
            Some(s.capture_ns.max(0) as u64)
        }
        Sample::Baro(s) => {
            put_header(out, seq, s.capture_ns, frame_id);
            put_f64(out, s.pressure_pa);
            put_f64(out, 0.0);
            Some(s.capture_ns.max(0) as u64)
        }
        Sample::Gps(s) => {
            put_header(out, seq, s.capture_ns, frame_id);
            out.push(0); // NavSatStatus.status = STATUS_FIX
            out.extend_from_slice(&1u16.to_le_bytes()); // service = GPS
            put_f64(out, f64::from(s.latitude));
            put_f64(out, f64::from(s.longitude));
            put_f64(out, f64::from(s.altitude));
            let var = f64::from(s.accuracy) * f64::from(s.accuracy);
            for i in 0..9 {
                put_f64(out, if i % 4 == 0 { var } else { 0.0 });
            }
            out.push(2); // COVARIANCE_TYPE_DIAGONAL_KNOWN
            Some(s.capture_ns.max(0) as u64)
        }
        Sample::TimeDomain(_) | Sample::Other { .. } => None,
    }
}

/// Decode a record written by [`VrsBagWriter`].
fn native_message(
    rec: &AriaRecording,
    meta: &crate::vrs::RecordMeta,
    payload: &[u8],
    out: &mut Vec<u8>,
) -> Result<Option<u64>> {
    let layout = rec
        .vrs()
        .record_layout(meta.stream, RecordType::Data, meta.format_version)
        .ok_or_else(|| BagError::generic("VRS: ROS stream without a data format"))?;
    let blocks = layout.split(payload).map_err(vrs_err)?;
    let (Some(Block::Layout(v)), Some(Block::Custom(bytes))) = (blocks.first(), blocks.get(1))
    else {
        return Err(BagError::generic("VRS: malformed ROS message record"));
    };
    out.extend_from_slice(bytes);
    Ok(Some(v.get_u64("log_time_ns").map_err(vrs_err)?))
}

// ---- writer -------------------------------------------------------------------------

struct StreamInfo {
    id: StreamId,
}

/// Writer side: ROS topics stored losslessly in a VRS file.
pub(crate) struct VrsBagWriter {
    writer: VrsWriter,
    data_layout: DataLayout,
    config_layout: DataLayout,
    streams: Vec<StreamInfo>,
    pending_config: Vec<Vec<u8>>,
    started: bool,
    compression: Compression,
}

fn string_piece(name: &str, index: usize) -> Piece {
    Piece {
        name: name.to_string(),
        kind: PieceKind::String,
        offset: None,
        index: Some(index),
    }
}

impl VrsBagWriter {
    pub(crate) fn create(path: &Path, settings: OutputCompression) -> Result<Self> {
        let compression = match settings {
            OutputCompression::None => Compression::None,
            OutputCompression::Zstd => Compression::Zstd,
            OutputCompression::Lz4 => Compression::Lz4,
            OutputCompression::Bz2 => {
                return Err(BagError::writer("vrs does not support bz2 compression"));
            }
        };
        let writer = VrsWriter::create(
            path,
            VrsWriteOptions {
                compression,
                ..VrsWriteOptions::default()
            },
        )
        .map_err(vrs_err)?;
        let data_layout = DataLayout::from_pieces(vec![Piece {
            name: "log_time_ns".into(),
            kind: PieceKind::Value(Scalar::U64),
            offset: Some(0),
            index: None,
        }])
        .map_err(vrs_err)?;
        let config_layout = DataLayout::from_pieces(vec![
            string_piece("topic", 0),
            string_piece("message_type", 1),
            string_piece("definition", 2),
            string_piece("serialization_format", 3),
            string_piece("type_hash", 4),
        ])
        .map_err(vrs_err)?;
        Ok(Self {
            writer,
            data_layout,
            config_layout,
            streams: Vec::new(),
            pending_config: Vec::new(),
            started: false,
            compression,
        })
    }

    pub(crate) fn add_stream(
        &mut self,
        topic: &str,
        message_type: &str,
        definition: &str,
        serialization_format: &str,
        type_hash: &str,
    ) -> Result<usize> {
        if self.started {
            return Err(BagError::writer(
                "VRS output needs every topic declared before the first message",
            ));
        }
        let index = self.streams.len();
        let id = StreamId::new(
            WRITER_BASE_TYPE_ID
                + i32::try_from(index).map_err(|_| BagError::writer("too many topics"))?,
            1,
        );
        let mut tags = StreamTags::default();
        tags.vrs
            .insert("VRS_Original_Recordable_Name".into(), topic.to_string());
        tags.user.insert(TAG_TOPIC.into(), topic.to_string());
        tags.user.insert(TAG_TYPE.into(), message_type.to_string());
        tags.user
            .insert(TAG_DEFINITION.into(), definition.to_string());
        tags.user
            .insert(TAG_ENCODING.into(), serialization_format.to_string());
        tags.user.insert(TAG_HASH.into(), type_hash.to_string());
        self.writer.add_stream(id, tags).map_err(vrs_err)?;
        self.writer
            .define_record_format(
                id,
                RecordType::Configuration,
                1,
                "data_layout",
                &[&self.config_layout],
            )
            .map_err(vrs_err)?;
        self.writer
            .define_record_format(
                id,
                RecordType::Data,
                1,
                "data_layout/size=8+custom",
                &[&self.data_layout],
            )
            .map_err(vrs_err)?;

        let mut cfg = LayoutWriter::new(&self.config_layout);
        cfg.set_string("topic", topic).map_err(vrs_err)?;
        cfg.set_string("message_type", message_type)
            .map_err(vrs_err)?;
        cfg.set_string("definition", definition).map_err(vrs_err)?;
        cfg.set_string("serialization_format", serialization_format)
            .map_err(vrs_err)?;
        cfg.set_string("type_hash", type_hash).map_err(vrs_err)?;
        self.pending_config.push(cfg.finish());
        self.streams.push(StreamInfo { id });
        Ok(index)
    }

    fn write_configurations(&mut self) -> Result<()> {
        if !self.started {
            for (info, cfg) in self.streams.iter().zip(&self.pending_config) {
                self.writer
                    .write_record_with(
                        info.id,
                        RecordType::Configuration,
                        1,
                        0.0,
                        cfg,
                        Compression::None,
                    )
                    .map_err(vrs_err)?;
            }
            self.started = true;
        }
        Ok(())
    }

    pub(crate) fn write(&mut self, stream: usize, timestamp_ns: u64, data: &[u8]) -> Result<()> {
        self.write_configurations()?;
        let id = self
            .streams
            .get(stream)
            .ok_or_else(|| BagError::connection_not_found(format!("stream {stream}")))?
            .id;
        let mut payload = Vec::with_capacity(8 + data.len());
        payload.extend_from_slice(&timestamp_ns.to_le_bytes());
        payload.extend_from_slice(data);
        self.writer
            .write_record_with(
                id,
                RecordType::Data,
                1,
                timestamp_ns as f64 / 1e9,
                &payload,
                self.compression,
            )
            .map_err(vrs_err)
    }

    pub(crate) fn finish(&mut self) -> Result<()> {
        self.write_configurations()?;
        self.writer.finish().map(|_| ()).map_err(vrs_err)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::rosbag::ros1::deserialize::Ros1Deserializer;
    use crate::rosbag::ros1::messages as r1;
    use crate::rosbag::transcode::plan_topic;
    use crate::vrs::aria::{BaroSample, GpsSample, ImageFrame, ImuSample};

    fn conn_for(m: &Mapped) -> Connection {
        connection(
            m.topic.to_string(),
            m.message_type.to_string(),
            catalog_definition(m.message_type, Dialect::Ros1).unwrap(),
            catalog_md5(m.message_type).unwrap(),
            "ros1".into(),
        )
    }

    fn imu_sample(stream: StreamId) -> ImuSample {
        ImuSample {
            stream,
            time_s: 12.5,
            capture_ns: 12_500_000_123,
            temperature_c: 30.0,
            accel: Some([1.0, -2.0, 9.5]),
            gyro: Some([0.1, 0.2, -0.3]),
            mag: Some([20.0, -5.0, 40.0]),
        }
    }

    /// Every message we synthesise must be exactly what its ROS definition
    /// says: the strict transcoder rejects truncated or over-long payloads.
    #[test]
    fn aria_messages_are_valid_for_their_ros_types() {
        let samples = [
            (
                AriaSensor::ImuRight,
                Sample::Imu(imu_sample(StreamId::new(1202, 1))),
            ),
            (
                AriaSensor::Magnetometer,
                Sample::Imu(imu_sample(StreamId::new(1203, 1))),
            ),
            (
                AriaSensor::Barometer,
                Sample::Baro(BaroSample {
                    time_s: 1.0,
                    capture_ns: 1_000_000_000,
                    temperature_c: 25.0,
                    pressure_pa: 101_325.0,
                    altitude_m: 10.0,
                }),
            ),
            (
                AriaSensor::Gps,
                Sample::Gps(GpsSample {
                    time_s: 1.0,
                    capture_ns: 1_000_000_000,
                    utc_time_ms: 1,
                    latitude: 47.4,
                    longitude: 8.5,
                    altitude: 500.0,
                    accuracy: 3.0,
                    provider: "gps".into(),
                }),
            ),
            (
                AriaSensor::SlamLeft,
                Sample::Image(ImageFrame {
                    stream: StreamId::new(1201, 1),
                    time_s: 2.0,
                    capture_ns: 2_000_000_000,
                    frame_number: 1,
                    exposure_s: 0.01,
                    gain: 2.0,
                    temperature_c: 30.0,
                    jpeg: vec![0xFF, 0xD8, 1, 2, 3, 0xFF, 0xD9],
                }),
            ),
        ];
        for (sensor, sample) in samples {
            let map = aria_mapping(sensor).unwrap();
            let mut bytes = Vec::new();
            let ts = aria_message(&sample, 7, map.frame_id, &mut bytes).unwrap();
            assert!(ts > 0);
            let plan = plan_topic(&conn_for(&map), Dialect::Ros2)
                .unwrap_or_else(|e| panic!("{}: {e}", map.topic));
            let mut cdr = Vec::new();
            plan.transcoder
                .convert(&bytes, &mut cdr)
                .unwrap_or_else(|e| panic!("{}: {e}", map.topic));
        }
    }

    #[test]
    fn imu_message_decodes_with_the_typed_ros1_decoder() {
        let mut bytes = Vec::new();
        let sample = Sample::Imu(imu_sample(StreamId::new(1202, 1)));
        aria_message(&sample, 5, "imu-right", &mut bytes).unwrap();
        let imu = r1::Imu::from_ros1(&mut Ros1Deserializer::new(&bytes)).unwrap();
        assert_eq!(imu.header.seq, 5);
        assert_eq!(imu.header.stamp_ns, 12_500_000_123);
        assert_eq!(imu.header.frame_id, "imu-right");
        assert_eq!(imu.linear_acceleration.z, 9.5);
        assert_eq!(imu.angular_velocity.y, f64::from(0.2f32));
        assert_eq!(imu.orientation_covariance[0], -1.0);
    }

    #[test]
    fn invalid_channels_are_dropped_not_zero_filled() {
        let mut s = imu_sample(StreamId::new(1202, 1));
        s.gyro = None;
        let mut out = Vec::new();
        assert!(aria_message(&Sample::Imu(s), 0, "x", &mut out).is_none());
        let mut m = imu_sample(StreamId::new(1203, 1));
        m.mag = None;
        assert!(aria_message(&Sample::Imu(m), 0, "x", &mut out).is_none());
    }

    #[test]
    fn mapping_covers_the_documented_sensors() {
        for sensor in [
            AriaSensor::Rgb,
            AriaSensor::SlamLeft,
            AriaSensor::SlamRight,
            AriaSensor::ImuRight,
            AriaSensor::ImuLeft,
            AriaSensor::Magnetometer,
            AriaSensor::Barometer,
            AriaSensor::Gps,
        ] {
            let m = aria_mapping(sensor).unwrap();
            assert!(catalog_definition(m.message_type, Dialect::Ros1).is_some());
            assert!(catalog_md5(m.message_type).is_some());
        }
        assert!(aria_mapping(AriaSensor::Wifi).is_none());
    }
}
