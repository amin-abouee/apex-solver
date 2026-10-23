//! LaMAria calibration files (`aria_calibrations/<sequence>.json`).
//!
//! One file per sequence, with the two SLAM cameras and the IMU of the Aria
//! glasses:
//!
//! ```text
//! { "cam0": { "model", "resolution", "params", "T_b_s": { "qvec", "tvec" } },
//!   "cam1": { ... },
//!   "imu0": { "T_b_s", "gravity_magnitude", "*_noise_density",
//!             "*_bias_random_walk_sigma", "*_saturation_max", "imu_rate" } }
//! ```
//!
//! `T_b_s` is the sensor's pose in the body frame, `qvec` a quaternion stored
//! **`x y z w`** (the IMU's identity is `[0, 0, 0, 1]`) and `tvec` metres. The
//! camera `params` are the model's own order, kept as read: for
//! `RAD_TAN_THIN_PRISM_FISHEYE`, `fx fy cx cy` then twelve distortion terms.
//!
//! The same data lives in each recording's VRS tags as Meta's factory
//! calibration (`vrs::aria::AriaRecording::calibration_json`, feature `vrs`); this is
//! the dataset's own, separate file, in a different schema.

use std::path::{Path, PathBuf};

use apex_manifolds::se3::SE3;
use apex_manifolds::so3::SO3;
use nalgebra::Vector3;
use serde::Deserialize;

/// Why a calibration file could not be read.
#[derive(Debug, thiserror::Error)]
pub enum LamariaError {
    /// The file could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The text is not the expected JSON.
    #[error("cannot parse {path}: {source}")]
    Json {
        /// The file.
        path: PathBuf,
        /// The underlying error.
        source: serde_json::Error,
    },
    /// A value is present but unusable.
    #[error("{path}: {what}")]
    Invalid {
        /// The file.
        path: PathBuf,
        /// What is wrong, with the sensor.
        what: String,
    },
}

/// Result alias for this module.
pub type Result<T> = std::result::Result<T, LamariaError>;

/// One camera: its model, image size, intrinsics and mounting.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraCalibration {
    /// The camera model's name, e.g. `RAD_TAN_THIN_PRISM_FISHEYE`.
    pub model: String,
    /// Image width \[px\].
    pub width: u32,
    /// Image height \[px\].
    pub height: u32,
    /// Intrinsics and distortion, in the model's order.
    pub params: Vec<f64>,
    /// Sensor position in the body frame \[m\].
    pub translation: Vector3<f64>,
    /// Sensor orientation in the body frame.
    pub rotation: SO3,
}

impl CameraCalibration {
    /// `T_body_sensor`.
    pub fn t_body_sensor(&self) -> SE3 {
        SE3::from_translation_so3(self.translation, self.rotation.clone())
    }
}

/// The IMU: mounting and noise.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuCalibration {
    /// Sensor position in the body frame \[m\].
    pub translation: Vector3<f64>,
    /// Sensor orientation in the body frame.
    pub rotation: SO3,
    /// Gravity magnitude \[m/s²\].
    pub gravity_magnitude: f64,
    /// Accelerometer noise density \[m/s²/√Hz\].
    pub acc_noise_density: f64,
    /// Gyroscope noise density \[rad/s/√Hz\].
    pub gyro_noise_density: f64,
    /// Accelerometer bias random walk \[m/s³/√Hz\].
    pub acc_bias_random_walk_sigma: f64,
    /// Gyroscope bias random walk \[rad/s²/√Hz\].
    pub gyro_bias_random_walk_sigma: f64,
    /// Accelerometer saturation \[m/s²\].
    pub acc_saturation_max: f64,
    /// Gyroscope saturation \[rad/s\].
    pub gyro_saturation_max: f64,
    /// Nominal rate \[Hz\].
    pub imu_rate: f64,
}

impl ImuCalibration {
    /// `T_body_sensor`.
    pub fn t_body_sensor(&self) -> SE3 {
        SE3::from_translation_so3(self.translation, self.rotation.clone())
    }
}

/// A sequence's calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// The first SLAM camera.
    pub cam0: CameraCalibration,
    /// The second SLAM camera.
    pub cam1: CameraCalibration,
    /// The IMU.
    pub imu: ImuCalibration,
}

impl Calibration {
    /// Read `aria_calibrations/<sequence>.json`.
    ///
    /// # Errors
    ///
    /// See [`LamariaError`].
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| LamariaError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text, path)
    }

    /// Parse calibration JSON. `path` labels errors only.
    ///
    /// # Errors
    ///
    /// See [`LamariaError`].
    pub fn parse(text: &str, path: &Path) -> Result<Self> {
        let raw: RawFile = serde_json::from_str(text).map_err(|source| LamariaError::Json {
            path: path.to_path_buf(),
            source,
        })?;
        let invalid = |what: String| LamariaError::Invalid {
            path: path.to_path_buf(),
            what,
        };
        let camera = |name: &str, raw: RawCamera| -> Result<CameraCalibration> {
            if raw.params.is_empty() || raw.params.iter().any(|p| !p.is_finite()) {
                return Err(invalid(format!("{name}: parameters empty or not finite")));
            }
            let (translation, rotation) = raw.t_b_s.pose().ok_or_else(|| {
                invalid(format!(
                    "{name}: T_b_s is not a unit quaternion and a vector"
                ))
            })?;
            Ok(CameraCalibration {
                model: raw.model,
                width: raw.resolution.width,
                height: raw.resolution.height,
                params: raw.params,
                translation,
                rotation,
            })
        };
        let (translation, rotation) =
            raw.imu0.t_b_s.pose().ok_or_else(|| {
                invalid("imu0: T_b_s is not a unit quaternion and a vector".into())
            })?;
        let imu = ImuCalibration {
            translation,
            rotation,
            gravity_magnitude: raw.imu0.gravity_magnitude,
            acc_noise_density: raw.imu0.acc_noise_density,
            gyro_noise_density: raw.imu0.gyro_noise_density,
            acc_bias_random_walk_sigma: raw.imu0.acc_bias_random_walk_sigma,
            gyro_bias_random_walk_sigma: raw.imu0.gyro_bias_random_walk_sigma,
            acc_saturation_max: raw.imu0.acc_saturation_max,
            gyro_saturation_max: raw.imu0.gyro_saturation_max,
            imu_rate: raw.imu0.imu_rate,
        };
        Ok(Self {
            cam0: camera("cam0", raw.cam0)?,
            cam1: camera("cam1", raw.cam1)?,
            imu,
        })
    }
}

#[derive(Deserialize)]
struct RawFile {
    cam0: RawCamera,
    cam1: RawCamera,
    imu0: RawImu,
}

#[derive(Deserialize)]
struct RawCamera {
    model: String,
    resolution: RawResolution,
    params: Vec<f64>,
    #[serde(rename = "T_b_s")]
    t_b_s: RawPose,
}

#[derive(Deserialize)]
struct RawResolution {
    width: u32,
    height: u32,
}

#[derive(Deserialize)]
struct RawImu {
    #[serde(rename = "T_b_s")]
    t_b_s: RawPose,
    gravity_magnitude: f64,
    acc_noise_density: f64,
    gyro_noise_density: f64,
    acc_bias_random_walk_sigma: f64,
    gyro_bias_random_walk_sigma: f64,
    acc_saturation_max: f64,
    gyro_saturation_max: f64,
    imu_rate: f64,
}

#[derive(Deserialize)]
struct RawPose {
    /// `x y z w`.
    qvec: [f64; 4],
    tvec: [f64; 3],
}

impl RawPose {
    fn pose(&self) -> Option<(Vector3<f64>, SO3)> {
        let [x, y, z, w] = self.qvec;
        let rotation = crate::trajectory::text_orientation(w, x, y, z)?;
        let translation = Vector3::new(self.tvec[0], self.tvec[1], self.tvec[2]);
        translation
            .iter()
            .all(|v| v.is_finite())
            .then_some((translation, rotation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    /// `R_06_medium/aria_calibrations/R_06_medium.json`, verbatim.
    const SAMPLE: &str = r#"{
    "cam0": {
        "model": "RAD_TAN_THIN_PRISM_FISHEYE",
        "resolution": {
            "width": 640,
            "height": 480
        },
        "params": [
            241.6048925567284,
            241.6048925567284,
            322.895555771704,
            240.4445599560633,
            -0.02484243376501498,
            0.09628938752134535,
            -0.063325466945624,
            0.006200726341569274,
            0.003488761399963396,
            -0.0007297520056856479,
            0.00089532032215904,
            0.000361066580790014,
            -0.00136879244387001,
            -0.0001054083415853531,
            -0.0004635611298488213,
            -0.0001049856461262643
        ],
        "T_b_s": {
            "qvec": [
                0.7871966367212387,
                0.08407781810505843,
                -0.05361927390966591,
                0.6085863530369895
            ],
            "tvec": [
                0.01678296927310889,
                -0.10892921838506506,
                0.07605583988186104
            ]
        }
    },
    "cam1": {
        "model": "RAD_TAN_THIN_PRISM_FISHEYE",
        "resolution": {
            "width": 640,
            "height": 480
        },
        "params": [
            241.3272717603338,
            241.3272717603338,
            321.6189294210335,
            239.6221110493106,
            -0.02540327399199527,
            0.09654066714494701,
            -0.06397303271827755,
            0.006223153894704157,
            0.003741072868202308,
            -0.0007998563476126205,
            0.001725615932096635,
            -0.0009989475964546615,
            -0.002626656620305424,
            4.621087294408381e-05,
            0.001412977671638857,
            -8.439958970429877e-05
        ],
        "T_b_s": {
            "qvec": [
                0.9969920181758002,
                0.012669053334945501,
                -0.07641622358987671,
                0.0026403699746858265
            ],
            "tvec": [
                -0.0021144839732690456,
                -0.0001812898201488805,
                -0.0063552451861506876
            ]
        }
    },
    "imu0": {
        "T_b_s": {
            "qvec": [
                0,
                0,
                0,
                1
            ],
            "tvec": [
                0,
                0,
                0
            ]
        },
        "gravity_magnitude": 9.806,
        "acc_noise_density": 0.0007844799999999999,
        "gyro_noise_density": 0.00017453292519943296,
        "acc_bias_random_walk_sigma": 0.0064483304620110765,
        "gyro_bias_random_walk_sigma": 0.00024437102728225855,
        "acc_saturation_max": 78.448,
        "gyro_saturation_max": 17.453292519943297,
        "imu_rate": 1000
    }
}"#;

    fn path() -> &'static Path {
        Path::new("R_06_medium.json")
    }

    #[test]
    fn a_real_file_parses_into_two_cameras_and_an_imu() -> TestResult {
        let calibration = Calibration::parse(SAMPLE, path())?;
        assert_eq!(calibration.cam0.model, "RAD_TAN_THIN_PRISM_FISHEYE");
        assert_eq!(
            (calibration.cam0.width, calibration.cam0.height),
            (640, 480)
        );
        assert_eq!(calibration.cam0.params.len(), 16);
        assert!((calibration.cam0.params[0] - 241.604_892_556_728_4).abs() < 1e-12);
        assert_eq!(calibration.cam1.params.len(), 16);
        assert!((calibration.imu.gravity_magnitude - 9.806).abs() < 1e-12);
        assert!((calibration.imu.imu_rate - 1000.0).abs() < 1e-12);
        Ok(())
    }

    /// `qvec` is `x y z w`: the IMU's `[0, 0, 0, 1]` is the identity rotation.
    #[test]
    fn the_quaternion_is_read_x_y_z_w() -> TestResult {
        let calibration = Calibration::parse(SAMPLE, path())?;
        let [w, x, y, z] = calibration.imu.rotation.coeffs();
        assert!((w - 1.0).abs() < 1e-12 && x.abs() + y.abs() + z.abs() < 1e-12);
        let [w, x, y, z] = calibration.cam0.rotation.coeffs();
        assert!((w - 0.608_586_353_036_989_5).abs() < 1e-12);
        assert!((x - 0.787_196_636_721_238_7).abs() < 1e-12);
        assert!((y - 0.084_077_818_107_238_6).abs() < 1e-9);
        assert!((z + 0.053_619_273_903_478_95).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn a_missing_sensor_is_a_json_error() {
        assert!(matches!(
            Calibration::parse(r#"{"cam0": {}}"#, path()),
            Err(LamariaError::Json { .. })
        ));
    }

    #[test]
    fn a_non_unit_quaternion_is_refused() {
        let broken = SAMPLE.replacen("0.7871966367212387", "7.871966367212387", 1);
        assert!(matches!(
            Calibration::parse(&broken, path()),
            Err(LamariaError::Invalid { .. })
        ));
    }
}
