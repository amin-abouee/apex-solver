//! Typed access to Project Aria recordings.
//!
//! [`AriaRecording`] wraps a [`VrsReader`] and decodes the standard Aria
//! sensors (cameras, IMUs, magnetometer, barometer, GPS, time-domain map)
//! into plain structs. It is built for being embedded in another pipeline
//! such as visual-inertial odometry:
//!
//! * pull samples one by one, merged in capture-time order across sensors
//!   ([`AriaRecording::next_sample`]);
//! * or ask by time: [`imu_between`](AriaRecording::imu_between),
//!   [`image_nearest`](AriaRecording::image_nearest), ...
//!
//! Camera frames are returned as the original JPEG bytes (no decode).
//!
//! ```no_run
//! use apex_io::vrs::aria::{AriaRecording, AriaSensor, Sample};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut rec = AriaRecording::open("recording.vrs")?;
//! let mut cursor = rec.cursor(&[AriaSensor::SlamLeft, AriaSensor::ImuRight]);
//! while let Some(sample) = rec.next_sample(&mut cursor)? {
//!     match sample {
//!         Sample::Imu(imu) => { let _ = (imu.time_s, imu.accel, imu.gyro); }
//!         Sample::Image(img) => { let _ = (img.time_s, img.jpeg.len()); }
//!         _ => {}
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use super::datalayout::LayoutView;
use super::error::{Result, VrsError};
use super::format::{RecordType, StreamId};
use super::reader::{Cursor, RecordFilter, RecordMeta, VrsReader};
use super::recordformat::Block;
use std::path::Path;

/// The sensors of an Aria device, mapped to their VRS stream ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AriaSensor {
    /// RGB camera, stream `214-1`.
    Rgb,
    /// Left SLAM camera, `1201-1`.
    SlamLeft,
    /// Right SLAM camera, `1201-2`.
    SlamRight,
    /// Eye-tracking camera, `211-1`.
    EyeTracking,
    /// Right IMU (about 1 kHz), `1202-1`.
    ImuRight,
    /// Left IMU (about 800 Hz), `1202-2`.
    ImuLeft,
    /// Magnetometer, `1203-1`.
    Magnetometer,
    /// Barometer, `247-1`.
    Barometer,
    /// GPS, `281-1`.
    Gps,
    /// Wi-Fi beacons, `282-1`.
    Wifi,
    /// Bluetooth beacons, `283-1`.
    Bluetooth,
    /// Monotonic-to-real-time clock map, `285-1`.
    TimeDomain,
    /// Attention (eye-gaze) data, `286-1`.
    Attention,
}

impl AriaSensor {
    /// All known sensors.
    pub const ALL: [Self; 13] = [
        Self::Rgb,
        Self::SlamLeft,
        Self::SlamRight,
        Self::EyeTracking,
        Self::ImuRight,
        Self::ImuLeft,
        Self::Magnetometer,
        Self::Barometer,
        Self::Gps,
        Self::Wifi,
        Self::Bluetooth,
        Self::TimeDomain,
        Self::Attention,
    ];

    /// The VRS stream id.
    pub fn stream_id(self) -> StreamId {
        let (t, i) = match self {
            Self::Rgb => (214, 1),
            Self::SlamLeft => (1201, 1),
            Self::SlamRight => (1201, 2),
            Self::EyeTracking => (211, 1),
            Self::ImuRight => (1202, 1),
            Self::ImuLeft => (1202, 2),
            Self::Magnetometer => (1203, 1),
            Self::Barometer => (247, 1),
            Self::Gps => (281, 1),
            Self::Wifi => (282, 1),
            Self::Bluetooth => (283, 1),
            Self::TimeDomain => (285, 1),
            Self::Attention => (286, 1),
        };
        StreamId::new(t, i)
    }

    /// Sensor for a stream id, if it is a known Aria stream.
    pub fn from_stream(id: StreamId) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.stream_id() == id)
    }

    /// Label used by Aria tooling (`camera-slam-left`, `imu-right`, ...).
    pub fn label(self) -> &'static str {
        match self {
            Self::Rgb => "camera-rgb",
            Self::SlamLeft => "camera-slam-left",
            Self::SlamRight => "camera-slam-right",
            Self::EyeTracking => "camera-et",
            Self::ImuRight => "imu-right",
            Self::ImuLeft => "imu-left",
            Self::Magnetometer => "mag",
            Self::Barometer => "baro",
            Self::Gps => "gps",
            Self::Wifi => "wps",
            Self::Bluetooth => "bluetooth",
            Self::TimeDomain => "time-domain",
            Self::Attention => "attention",
        }
    }
}

/// An IMU or magnetometer measurement. A channel is `None` when the device
/// flagged it invalid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuSample {
    /// Source stream.
    pub stream: StreamId,
    /// Record timestamp in seconds.
    pub time_s: f64,
    /// Capture timestamp in nanoseconds (same clock as `time_s`).
    pub capture_ns: i64,
    /// Sensor temperature in degrees Celsius.
    pub temperature_c: f64,
    /// Accelerometer, m/s^2.
    pub accel: Option<[f32; 3]>,
    /// Gyroscope, rad/s.
    pub gyro: Option<[f32; 3]>,
    /// Magnetometer, microtesla.
    pub mag: Option<[f32; 3]>,
}

/// A barometer reading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaroSample {
    /// Record timestamp in seconds.
    pub time_s: f64,
    /// Capture timestamp in nanoseconds.
    pub capture_ns: i64,
    /// Temperature in degrees Celsius.
    pub temperature_c: f64,
    /// Pressure in pascal.
    pub pressure_pa: f64,
    /// Derived altitude in metres.
    pub altitude_m: f64,
}

/// A GPS fix.
#[derive(Debug, Clone, PartialEq)]
pub struct GpsSample {
    /// Record timestamp in seconds.
    pub time_s: f64,
    /// Capture timestamp in nanoseconds.
    pub capture_ns: i64,
    /// UTC time in milliseconds.
    pub utc_time_ms: i64,
    /// Latitude in degrees.
    pub latitude: f32,
    /// Longitude in degrees.
    pub longitude: f32,
    /// Altitude in metres.
    pub altitude: f32,
    /// Horizontal accuracy in metres.
    pub accuracy: f32,
    /// Position provider (for example `gps`).
    pub provider: String,
}

/// A camera frame: the JPEG exactly as recorded.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageFrame {
    /// Source stream.
    pub stream: StreamId,
    /// Record timestamp in seconds.
    pub time_s: f64,
    /// Capture timestamp in nanoseconds.
    pub capture_ns: i64,
    /// Frame counter.
    pub frame_number: u64,
    /// Exposure time in seconds.
    pub exposure_s: f64,
    /// Sensor gain.
    pub gain: f64,
    /// Sensor temperature in degrees Celsius.
    pub temperature_c: f64,
    /// Encoded image bytes (JPEG).
    pub jpeg: Vec<u8>,
}

/// A mapping between the device's monotonic clock and real time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeDomainSample {
    /// Monotonic clock, ns.
    pub monotonic_ns: i64,
    /// Real-time (UTC) clock, ns.
    pub real_ns: i64,
}

/// One decoded data record.
#[derive(Debug, Clone, PartialEq)]
pub enum Sample {
    /// IMU or magnetometer.
    Imu(ImuSample),
    /// Barometer.
    Baro(BaroSample),
    /// GPS.
    Gps(GpsSample),
    /// Camera frame.
    Image(ImageFrame),
    /// Clock mapping.
    TimeDomain(TimeDomainSample),
    /// A stream without a typed decoder (Wi-Fi, Bluetooth, attention, ...):
    /// the raw decompressed payload.
    Other {
        /// Source stream.
        stream: StreamId,
        /// Record timestamp in seconds.
        time_s: f64,
        /// Decompressed payload.
        payload: Vec<u8>,
    },
}

impl Sample {
    /// Record timestamp in seconds.
    pub fn time_s(&self) -> f64 {
        match self {
            Self::Imu(s) => s.time_s,
            Self::Baro(s) => s.time_s,
            Self::Gps(s) => s.time_s,
            Self::Image(s) => s.time_s,
            Self::TimeDomain(_) => f64::NAN,
            Self::Other { time_s, .. } => *time_s,
        }
    }
}

/// Camera parameters from the stream's configuration record.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraConfig {
    /// Device camera id.
    pub camera_id: u32,
    /// Nominal frame rate in Hz.
    pub nominal_rate: f64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Row stride in bytes.
    pub stride: u32,
    /// Sensor model.
    pub sensor_model: String,
    /// Device serial number.
    pub device_serial: String,
    /// Factory calibration (JSON text), possibly empty.
    pub factory_calibration: String,
}

/// IMU/magnetometer parameters from the configuration record.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuConfig {
    /// Nominal sampling rate in Hz.
    pub nominal_rate: f64,
    /// Whether the unit has an accelerometer.
    pub has_accelerometer: bool,
    /// Whether the unit has a gyroscope.
    pub has_gyroscope: bool,
    /// Whether the unit has a magnetometer.
    pub has_magnetometer: bool,
    /// Sensor model.
    pub sensor_model: String,
}

/// An Aria recording opened for typed access.
pub struct AriaRecording {
    vrs: VrsReader,
    buf: Vec<u8>,
}

impl AriaRecording {
    /// Open a `.vrs` file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::from_reader(VrsReader::open(path)?))
    }

    /// Wrap an already-open reader.
    pub fn from_reader(vrs: VrsReader) -> Self {
        Self {
            vrs,
            buf: Vec::new(),
        }
    }

    /// The underlying generic reader.
    pub fn vrs(&self) -> &VrsReader {
        &self.vrs
    }

    /// Mutable access to the generic reader.
    pub fn vrs_mut(&mut self) -> &mut VrsReader {
        &mut self.vrs
    }

    /// Known Aria sensors present in the file.
    pub fn sensors(&self) -> Vec<AriaSensor> {
        let present = self.vrs.streams();
        AriaSensor::ALL
            .into_iter()
            .filter(|s| present.contains(&s.stream_id()))
            .collect()
    }

    /// Number of data records of a sensor.
    pub fn sample_count(&self, sensor: AriaSensor) -> usize {
        self.vrs.data_records(sensor.stream_id()).len()
    }

    /// First and last data time of a sensor, in seconds.
    pub fn time_range(&self, sensor: AriaSensor) -> Option<(f64, f64)> {
        self.vrs.stream_time_range(sensor.stream_id())
    }

    /// The factory calibration JSON stored in the file tags, if any.
    pub fn calibration_json(&self) -> Option<&str> {
        self.vrs.file_tags().get("calib_json").map(String::as_str)
    }

    /// Camera configuration of a camera sensor.
    pub fn camera_config(&mut self, sensor: AriaSensor) -> Result<CameraConfig> {
        self.with_config(sensor, |v| {
            Ok(CameraConfig {
                camera_id: v.get_u32("camera_id")?,
                nominal_rate: v.get_f64("nominal_rate")?,
                width: v.get_u32("image_width")?,
                height: v.get_u32("image_height")?,
                stride: v.get_u32("image_stride")?,
                sensor_model: v.get_string("sensor_model")?,
                device_serial: v.get_string("device_serial")?,
                factory_calibration: v.get_string("factory_calibration")?,
            })
        })
    }

    /// IMU/magnetometer configuration.
    pub fn imu_config(&mut self, sensor: AriaSensor) -> Result<ImuConfig> {
        self.with_config(sensor, |v| {
            Ok(ImuConfig {
                nominal_rate: v.get_f64("nominal_rate")?,
                has_accelerometer: v.get_bool("has_accelerometer")?,
                has_gyroscope: v.get_bool("has_gyroscope")?,
                has_magnetometer: v.get_bool("has_magnetometer")?,
                sensor_model: v.get_string("sensor_model")?,
            })
        })
    }

    fn with_config<T>(
        &mut self,
        sensor: AriaSensor,
        f: impl FnOnce(&LayoutView<'_>) -> Result<T>,
    ) -> Result<T> {
        let stream = sensor.stream_id();
        let index = self.vrs.configuration_record(stream).ok_or_else(|| {
            VrsError::Usage(format!("{} has no configuration record", sensor.label()))
        })?;
        let mut buf = std::mem::take(&mut self.buf);
        let result = (|| {
            let meta = self.vrs.read_record_into(index, &mut buf)?;
            let layout = self
                .vrs
                .record_layout(stream, RecordType::Configuration, meta.format_version)
                .ok_or_else(|| VrsError::Layout("no configuration layout".into()))?;
            match layout.split(&buf)?.first() {
                Some(Block::Layout(v)) => f(v),
                _ => Err(VrsError::Layout("configuration has no DataLayout".into())),
            }
        })();
        self.buf = buf;
        result
    }

    // ---- sequential access ------------------------------------------------

    /// A cursor over the given sensors in capture-time order (all sensors
    /// when `sensors` is empty).
    pub fn cursor(&self, sensors: &[AriaSensor]) -> Cursor {
        self.vrs.cursor(&Self::filter(sensors))
    }

    /// A cursor over `[start, end)` seconds.
    pub fn cursor_between(&self, sensors: &[AriaSensor], start: f64, end: f64) -> Cursor {
        self.vrs.cursor(&Self::filter(sensors).between(start, end))
    }

    fn filter(sensors: &[AriaSensor]) -> RecordFilter {
        RecordFilter::streams(sensors.iter().map(|s| s.stream_id()))
    }

    /// Next sample in time order, or `None` at the end.
    pub fn next_sample(&mut self, cursor: &mut Cursor) -> Result<Option<Sample>> {
        let mut buf = std::mem::take(&mut self.buf);
        let result = match self.vrs.next_record_into(cursor, &mut buf) {
            Ok(Some(meta)) => self.decode(meta, &buf).map(Some),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        };
        self.buf = buf;
        result
    }

    /// Decode data record `index` (an index into [`VrsReader::records`]).
    pub fn sample_at(&mut self, index: usize) -> Result<Sample> {
        let mut buf = std::mem::take(&mut self.buf);
        let result = match self.vrs.read_record_into(index, &mut buf) {
            Ok(meta) => self.decode(meta, &buf),
            Err(e) => Err(e),
        };
        self.buf = buf;
        result
    }

    // ---- time-based access ----------------------------------------------------

    /// All IMU/magnetometer samples with `start <= t < end`.
    pub fn imu_between(
        &mut self,
        sensor: AriaSensor,
        start: f64,
        end: f64,
    ) -> Result<Vec<ImuSample>> {
        let indices: Vec<u32> = self
            .vrs
            .data_records_between(sensor.stream_id(), start, end)
            .to_vec();
        indices
            .into_iter()
            .map(|i| match self.sample_at(i as usize)? {
                Sample::Imu(s) => Ok(s),
                _ => Err(VrsError::Usage(format!(
                    "{} is not an IMU stream",
                    sensor.label()
                ))),
            })
            .collect()
    }

    /// The camera frame closest in time to `t`.
    pub fn image_nearest(&mut self, sensor: AriaSensor, t: f64) -> Result<Option<ImageFrame>> {
        match self.vrs.find_nearest(sensor.stream_id(), t) {
            Some(i) => match self.sample_at(i)? {
                Sample::Image(f) => Ok(Some(f)),
                _ => Err(VrsError::Usage(format!(
                    "{} is not a camera",
                    sensor.label()
                ))),
            },
            None => Ok(None),
        }
    }

    /// The first camera frame at or after `t`.
    pub fn image_at_or_after(&mut self, sensor: AriaSensor, t: f64) -> Result<Option<ImageFrame>> {
        match self.vrs.find_at_or_after(sensor.stream_id(), t) {
            Some(i) => match self.sample_at(i)? {
                Sample::Image(f) => Ok(Some(f)),
                _ => Err(VrsError::Usage(format!(
                    "{} is not a camera",
                    sensor.label()
                ))),
            },
            None => Ok(None),
        }
    }

    /// The IMU/magnetometer sample closest in time to `t`.
    pub fn imu_nearest(&mut self, sensor: AriaSensor, t: f64) -> Result<Option<ImuSample>> {
        match self.vrs.find_nearest(sensor.stream_id(), t) {
            Some(i) => match self.sample_at(i)? {
                Sample::Imu(s) => Ok(Some(s)),
                _ => Err(VrsError::Usage(format!(
                    "{} is not an IMU stream",
                    sensor.label()
                ))),
            },
            None => Ok(None),
        }
    }

    // ---- decoding ---------------------------------------------------------------

    fn decode(&self, meta: RecordMeta, payload: &[u8]) -> Result<Sample> {
        let layout = self
            .vrs
            .record_layout(meta.stream, RecordType::Data, meta.format_version)
            .ok_or_else(|| {
                VrsError::Layout(format!(
                    "stream {} has no layout for data format {}",
                    meta.stream, meta.format_version
                ))
            })?;
        let blocks = layout.split(payload)?;
        let Some(Block::Layout(v)) = blocks.first() else {
            return Err(VrsError::Layout("data record has no DataLayout".into()));
        };
        let time_s = meta.timestamp;

        match meta.stream.type_id {
            214 | 1201 | 211 => {
                let jpeg = blocks
                    .iter()
                    .find_map(|b| match b {
                        Block::Image { bytes, .. } => Some(bytes.to_vec()),
                        _ => None,
                    })
                    .ok_or_else(|| VrsError::Layout("camera record has no image block".into()))?;
                Ok(Sample::Image(ImageFrame {
                    stream: meta.stream,
                    time_s,
                    capture_ns: v.get_i64("capture_timestamp_ns")?,
                    frame_number: v.get_u64("frame_number")?,
                    exposure_s: v.get_f64("exposure_duration_s")?,
                    gain: v.get_f64("gain")?,
                    temperature_c: v.get_f64("temperature_deg_c")?,
                    jpeg,
                }))
            }
            1202 | 1203 => {
                let channel = |valid: &str, field: &str| -> Result<Option<[f32; 3]>> {
                    if v.get_bool(valid)? {
                        v.get_f32_array::<3>(field).map(Some)
                    } else {
                        Ok(None)
                    }
                };
                Ok(Sample::Imu(ImuSample {
                    stream: meta.stream,
                    time_s,
                    capture_ns: v.get_i64("capture_timestamp_ns")?,
                    temperature_c: v.get_f64("temperature_deg_c")?,
                    accel: channel("accelerometer_valid", "accelerometer")?,
                    gyro: channel("gyroscope_valid", "gyroscope")?,
                    mag: channel("magnetometer_valid", "magnetometer")?,
                }))
            }
            247 => Ok(Sample::Baro(BaroSample {
                time_s,
                capture_ns: v.get_i64("capture_timestamp_ns")?,
                temperature_c: v.get_f64("temperature")?,
                pressure_pa: v.get_f64("pressure")?,
                altitude_m: v.get_f64("altitude")?,
            })),
            281 => Ok(Sample::Gps(GpsSample {
                time_s,
                capture_ns: v.get_i64("capture_timestamp_ns")?,
                utc_time_ms: v.get_i64("utc_time_ms")?,
                latitude: v.get_f32("latitude")?,
                longitude: v.get_f32("longitude")?,
                altitude: v.get_f32("altitude")?,
                accuracy: v.get_f32("accuracy")?,
                provider: v.get_string("provider")?,
            })),
            285 => Ok(Sample::TimeDomain(TimeDomainSample {
                monotonic_ns: v.get_i64("monotonic_timestamp_ns")?,
                real_ns: v.get_i64("real_timestamp_ns")?,
            })),
            _ => Ok(Sample::Other {
                stream: meta.stream,
                time_s,
                payload: payload.to_vec(),
            }),
        }
    }
}
