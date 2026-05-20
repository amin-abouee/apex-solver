pub mod error;
pub mod types;

pub use error::{AslError, Result};
pub use types::{AslDataset, CameraData, CameraFrame, GroundTruthPose, ImuMeasurement};

/// Cursor-based streaming interface over one camera and the IMU in an ASL dataset.
///
/// Open with [`AslStream::open`] (camera 0) or [`AslStream::open_camera`] (specific index),
/// then call [`next_image`](AslStream::next_image) / [`next_imu`](AslStream::next_imu) to
/// consume frames one at a time, or [`next_n_images`](AslStream::next_n_images) /
/// [`next_n_imu`](AslStream::next_n_imu) for batches.
pub struct AslStream {
    dataset: AslDataset,
    cam_idx: usize,
    image_cursor: usize,
    imu_cursor: usize,
}

impl AslStream {
    /// Open a mav0 directory and stream camera 0.
    pub fn open<P: AsRef<Path>>(mav0_path: P) -> Result<Self> {
        Self::open_camera(mav0_path, 0)
    }

    /// Open a mav0 directory and stream the camera at `cam_idx`.
    pub fn open_camera<P: AsRef<Path>>(mav0_path: P, cam_idx: usize) -> Result<Self> {
        let dataset = AslReader::load(mav0_path)?;
        if cam_idx >= dataset.cameras.len() {
            return Err(AslError::CameraNotFound { index: cam_idx });
        }
        Ok(Self {
            dataset,
            cam_idx,
            image_cursor: 0,
            imu_cursor: 0,
        })
    }

    /// Total number of image frames in the selected camera stream.
    pub fn image_count(&self) -> usize {
        self.dataset
            .cameras
            .get(self.cam_idx)
            .map(|c| c.frames.len())
            .unwrap_or(0)
    }

    /// Total number of IMU measurements in the dataset.
    pub fn imu_count(&self) -> usize {
        self.dataset.imu_measurements.len()
    }

    /// Yield the next image frame `(timestamp_ns, image)`.
    /// Returns `None` when the stream is exhausted.
    pub fn next_image(&mut self) -> Option<Result<(u64, image::DynamicImage)>> {
        let frame = self
            .dataset
            .cameras
            .get(self.cam_idx)?
            .frames
            .get(self.image_cursor)?;
        let path = frame.image_path.clone();
        let ts = frame.timestamp_ns;
        self.image_cursor += 1;
        Some(
            image::open(&path)
                .map(|img| (ts, img))
                .map_err(AslError::ImageLoad),
        )
    }

    /// Yield the next IMU measurement. Returns `None` when exhausted.
    pub fn next_imu(&mut self) -> Option<ImuMeasurement> {
        let m = self.dataset.imu_measurements.get(self.imu_cursor)?.clone();
        self.imu_cursor += 1;
        Some(m)
    }

    /// Yield the next `n` image frames. May return fewer when the stream is nearly exhausted.
    /// Returns `Err` immediately if any image fails to load.
    pub fn next_n_images(&mut self, n: usize) -> Result<Vec<(u64, image::DynamicImage)>> {
        let mut frames = Vec::with_capacity(n);
        for _ in 0..n {
            match self.next_image() {
                Some(Ok(f)) => frames.push(f),
                Some(Err(e)) => return Err(e),
                None => break,
            }
        }
        Ok(frames)
    }

    /// Yield the next `n` IMU measurements. May return fewer when exhausted.
    pub fn next_n_imu(&mut self, n: usize) -> Vec<ImuMeasurement> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            match self.next_imu() {
                Some(m) => out.push(m),
                None => break,
            }
        }
        out
    }

    /// Reset the image cursor to the beginning of the stream.
    pub fn reset_images(&mut self) {
        self.image_cursor = 0;
    }

    /// Reset the IMU cursor to the beginning of the stream.
    pub fn reset_imu(&mut self) {
        self.imu_cursor = 0;
    }
}

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

pub struct AslReader;

impl AslReader {
    pub fn load<P: AsRef<Path>>(mav0_path: P) -> Result<AslDataset> {
        let mav0 = mav0_path.as_ref();
        if !mav0.is_dir() {
            return Err(AslError::InvalidMav0Directory {
                path: mav0.to_path_buf(),
            });
        }

        let sensor_entries = discover_sensors(mav0)?;

        let results: Vec<Result<LoadedSensor>> =
            sensor_entries.par_iter().map(load_sensor).collect();

        let mut cameras: Vec<CameraData> = Vec::new();
        let mut imu_measurements: Vec<ImuMeasurement> = Vec::new();
        let mut ground_truth: Option<Vec<GroundTruthPose>> = None;

        for result in results {
            match result? {
                LoadedSensor::Camera(cam_data) => cameras.push(cam_data),
                LoadedSensor::Imu(measurements) => imu_measurements = measurements,
                LoadedSensor::GroundTruth(poses) => ground_truth = Some(poses),
            }
        }

        cameras.sort_by_key(|c| c.index);

        Ok(AslDataset {
            cameras,
            imu_measurements,
            ground_truth,
            base_path: mav0.to_path_buf(),
        })
    }
}

impl AslDataset {
    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    pub fn imu_sample_count(&self) -> usize {
        self.imu_measurements.len()
    }

    pub fn has_ground_truth(&self) -> bool {
        self.ground_truth.is_some()
    }

    pub fn load_image(&self, cam_idx: usize, frame_idx: usize) -> Result<image::DynamicImage> {
        let path = self.frame_path(cam_idx, frame_idx)?;
        image::open(&path).map_err(AslError::ImageLoad)
    }

    #[cfg(feature = "asl-async")]
    pub async fn load_image_async(
        &self,
        cam_idx: usize,
        frame_idx: usize,
    ) -> Result<image::DynamicImage> {
        let path = self.frame_path(cam_idx, frame_idx)?;
        tokio::task::spawn_blocking(move || image::open(&path).map_err(AslError::ImageLoad))
            .await
            .map_err(|e| AslError::Io(std::io::Error::other(e.to_string())))?
    }

    fn frame_path(&self, cam_idx: usize, frame_idx: usize) -> Result<PathBuf> {
        let cam = self
            .cameras
            .get(cam_idx)
            .ok_or(AslError::CameraNotFound { index: cam_idx })?;
        let frame = cam
            .frames
            .get(frame_idx)
            .ok_or(AslError::FrameOutOfBounds {
                cam_idx,
                frame_idx,
                total: cam.frames.len(),
            })?;
        Ok(frame.image_path.clone())
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

enum SensorEntry {
    Camera { index: usize, path: PathBuf },
    Imu { path: PathBuf },
    GroundTruth { path: PathBuf },
}

enum LoadedSensor {
    Camera(CameraData),
    Imu(Vec<ImuMeasurement>),
    GroundTruth(Vec<GroundTruthPose>),
}

// ---------------------------------------------------------------------------
// Sensor discovery
// ---------------------------------------------------------------------------

fn discover_sensors(mav0: &Path) -> Result<Vec<SensorEntry>> {
    let mut entries: Vec<SensorEntry> = Vec::new();

    for dir_entry in std::fs::read_dir(mav0)? {
        let dir_entry = dir_entry?;
        if !dir_entry.file_type()?.is_dir() {
            continue;
        }
        let name = dir_entry.file_name();
        let name = name.to_string_lossy();

        if let Some(idx) = parse_sensor_index(&name, "cam") {
            entries.push(SensorEntry::Camera {
                index: idx,
                path: dir_entry.path(),
            });
        } else if parse_sensor_index(&name, "imu").is_some() {
            entries.push(SensorEntry::Imu {
                path: dir_entry.path(),
            });
        } else if parse_sensor_index(&name, "mocap").is_some() {
            entries.push(SensorEntry::GroundTruth {
                path: dir_entry.path(),
            });
        }
    }

    if entries.is_empty() {
        return Err(AslError::NoSensorsFound {
            path: mav0.to_path_buf(),
        });
    }

    Ok(entries)
}

fn parse_sensor_index(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?.parse::<usize>().ok()
}

// ---------------------------------------------------------------------------
// Sensor loading dispatcher
// ---------------------------------------------------------------------------

fn load_sensor(entry: &SensorEntry) -> Result<LoadedSensor> {
    match entry {
        SensorEntry::Camera { index, path } => {
            let csv_path = path.join("data.csv");
            let data_dir = path.join("data");
            let frames = parse_camera_csv(&csv_path, &data_dir)?;
            Ok(LoadedSensor::Camera(CameraData {
                index: *index,
                frames,
                data_dir,
            }))
        }
        SensorEntry::Imu { path } => {
            let csv_path = path.join("data.csv");
            let measurements = parse_imu_csv(&csv_path)?;
            Ok(LoadedSensor::Imu(measurements))
        }
        SensorEntry::GroundTruth { path } => {
            let csv_path = path.join("data.csv");
            let poses = parse_ground_truth_csv(&csv_path)?;
            Ok(LoadedSensor::GroundTruth(poses))
        }
    }
}

// ---------------------------------------------------------------------------
// Shared CSV helper
// ---------------------------------------------------------------------------

fn parse_csv_rows(path: &Path) -> Result<Vec<Vec<String>>> {
    let content = std::fs::read_to_string(path)?;
    let rows = content
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .map(|line| {
            line.split(',')
                .map(|field| field.trim().to_owned())
                .collect::<Vec<_>>()
        })
        .collect();
    Ok(rows)
}

fn parse_f64(path: &Path, line: usize, field: &str, value: &str) -> Result<f64> {
    value.parse::<f64>().map_err(|_| AslError::InvalidNumber {
        path: path.to_path_buf(),
        line,
        field: field.to_owned(),
        value: value.to_owned(),
    })
}

fn parse_u64(path: &Path, line: usize, field: &str, value: &str) -> Result<u64> {
    value.parse::<u64>().map_err(|_| AslError::InvalidNumber {
        path: path.to_path_buf(),
        line,
        field: field.to_owned(),
        value: value.to_owned(),
    })
}

// ---------------------------------------------------------------------------
// Per-sensor parsers
// ---------------------------------------------------------------------------

fn parse_imu_csv(csv_path: &Path) -> Result<Vec<ImuMeasurement>> {
    let rows = parse_csv_rows(csv_path)?;
    let mut measurements = Vec::with_capacity(rows.len());

    for (row_idx, fields) in rows.iter().enumerate() {
        let line = row_idx + 1;
        if fields.len() < 7 {
            return Err(AslError::MissingCsvColumns {
                path: csv_path.to_path_buf(),
                line,
                expected: 7,
                got: fields.len(),
            });
        }
        let timestamp_ns = parse_u64(csv_path, line, "timestamp", &fields[0])?;
        let wx = parse_f64(csv_path, line, "wx", &fields[1])?;
        let wy = parse_f64(csv_path, line, "wy", &fields[2])?;
        let wz = parse_f64(csv_path, line, "wz", &fields[3])?;
        let ax = parse_f64(csv_path, line, "ax", &fields[4])?;
        let ay = parse_f64(csv_path, line, "ay", &fields[5])?;
        let az = parse_f64(csv_path, line, "az", &fields[6])?;

        measurements.push(ImuMeasurement {
            timestamp_ns,
            angular_velocity: Vector3::new(wx, wy, wz),
            linear_acceleration: Vector3::new(ax, ay, az),
        });
    }

    Ok(measurements)
}

fn parse_camera_csv(csv_path: &Path, data_dir: &Path) -> Result<Vec<CameraFrame>> {
    let rows = parse_csv_rows(csv_path)?;
    let mut frames = Vec::with_capacity(rows.len());

    for (row_idx, fields) in rows.iter().enumerate() {
        let line = row_idx + 1;
        if fields.len() < 2 {
            return Err(AslError::MissingCsvColumns {
                path: csv_path.to_path_buf(),
                line,
                expected: 2,
                got: fields.len(),
            });
        }
        let timestamp_ns = parse_u64(csv_path, line, "timestamp", &fields[0])?;
        let image_path = data_dir.join(&fields[1]);

        frames.push(CameraFrame {
            timestamp_ns,
            image_path,
        });
    }

    Ok(frames)
}

fn parse_ground_truth_csv(csv_path: &Path) -> Result<Vec<GroundTruthPose>> {
    let rows = parse_csv_rows(csv_path)?;
    let mut poses = Vec::with_capacity(rows.len());

    for (row_idx, fields) in rows.iter().enumerate() {
        let line = row_idx + 1;
        if fields.len() < 8 {
            return Err(AslError::MissingCsvColumns {
                path: csv_path.to_path_buf(),
                line,
                expected: 8,
                got: fields.len(),
            });
        }
        let timestamp_ns = parse_u64(csv_path, line, "timestamp", &fields[0])?;
        let px = parse_f64(csv_path, line, "px", &fields[1])?;
        let py = parse_f64(csv_path, line, "py", &fields[2])?;
        let pz = parse_f64(csv_path, line, "pz", &fields[3])?;
        let qw = parse_f64(csv_path, line, "qw", &fields[4])?;
        let qx = parse_f64(csv_path, line, "qx", &fields[5])?;
        let qy = parse_f64(csv_path, line, "qy", &fields[6])?;
        let qz = parse_f64(csv_path, line, "qz", &fields[7])?;

        // nalgebra::Quaternion::new(w, i, j, k)
        let q = Quaternion::new(qw, qx, qy, qz);
        let norm = q.norm();
        if (norm - 1.0).abs() > 0.01 {
            return Err(AslError::InvalidQuaternion {
                path: csv_path.to_path_buf(),
                line,
                norm,
            });
        }

        poses.push(GroundTruthPose {
            timestamp_ns,
            position: Vector3::new(px, py, pz),
            orientation: UnitQuaternion::from_quaternion(q.normalize()),
        });
    }

    Ok(poses)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    // --- IMU CSV ---

    #[test]
    fn parse_imu_csv_valid() {
        let tmp = TempDir::new().unwrap();
        let csv = write_file(
            tmp.path(),
            "data.csv",
            "#timestamp [ns],wx,wy,wz,ax,ay,az\n\
             100,1.0,2.0,3.0,4.0,5.0,6.0\n\
             200,0.1,0.2,0.3,0.4,0.5,0.6\n",
        );
        let rows = parse_imu_csv(&csv).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].timestamp_ns, 100);
        assert!((rows[0].angular_velocity.x - 1.0).abs() < 1e-10);
        assert!((rows[1].linear_acceleration.z - 0.6).abs() < 1e-10);
    }

    #[test]
    fn parse_imu_csv_skips_comment_lines() {
        let tmp = TempDir::new().unwrap();
        let csv = write_file(
            tmp.path(),
            "data.csv",
            "# first comment\n\
             # second comment\n\
             100,1.0,2.0,3.0,4.0,5.0,6.0\n",
        );
        let rows = parse_imu_csv(&csv).unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn parse_imu_csv_missing_fields_returns_error() {
        let tmp = TempDir::new().unwrap();
        let csv = write_file(tmp.path(), "data.csv", "100,1.0,2.0\n");
        let result = parse_imu_csv(&csv);
        assert!(matches!(
            result,
            Err(AslError::MissingCsvColumns {
                expected: 7,
                got: 3,
                ..
            })
        ));
    }

    #[test]
    fn parse_imu_csv_invalid_number_returns_error() {
        let tmp = TempDir::new().unwrap();
        let csv = write_file(tmp.path(), "data.csv", "100,NaN_bad,2.0,3.0,4.0,5.0,6.0\n");
        let result = parse_imu_csv(&csv);
        assert!(matches!(result, Err(AslError::InvalidNumber { .. })));
    }

    // --- Camera CSV ---

    #[test]
    fn parse_camera_csv_valid() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let csv = write_file(
            tmp.path(),
            "data.csv",
            "#timestamp [ns],filename\n\
             100,100.png\n\
             200,200.png\n",
        );
        let frames = parse_camera_csv(&csv, &data_dir).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].timestamp_ns, 100);
        assert_eq!(frames[0].image_path, data_dir.join("100.png"));
        assert_eq!(frames[1].timestamp_ns, 200);
    }

    #[test]
    fn parse_camera_csv_missing_fields_returns_error() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let csv = write_file(tmp.path(), "data.csv", "100\n");
        let result = parse_camera_csv(&csv, &data_dir);
        assert!(matches!(
            result,
            Err(AslError::MissingCsvColumns {
                expected: 2,
                got: 1,
                ..
            })
        ));
    }

    // --- Ground truth CSV ---

    #[test]
    fn parse_ground_truth_valid() {
        let tmp = TempDir::new().unwrap();
        let csv = write_file(
            tmp.path(),
            "data.csv",
            "#timestamp,px,py,pz,qw,qx,qy,qz\n\
             100,1.0,2.0,3.0,1.0,0.0,0.0,0.0\n",
        );
        let poses = parse_ground_truth_csv(&csv).unwrap();
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].timestamp_ns, 100);
        assert!((poses[0].position.x - 1.0).abs() < 1e-10);
        // identity quaternion
        assert!((poses[0].orientation.quaternion().w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn parse_ground_truth_invalid_quaternion_returns_error() {
        let tmp = TempDir::new().unwrap();
        let csv = write_file(tmp.path(), "data.csv", "100,0.0,0.0,0.0,0.1,0.0,0.0,0.0\n");
        let result = parse_ground_truth_csv(&csv);
        assert!(matches!(result, Err(AslError::InvalidQuaternion { .. })));
    }

    // --- Sensor discovery ---

    #[test]
    fn sensor_discovery_finds_cameras_and_imu() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("cam0")).unwrap();
        std::fs::create_dir_all(tmp.path().join("cam1")).unwrap();
        std::fs::create_dir_all(tmp.path().join("imu0")).unwrap();

        let entries = discover_sensors(tmp.path()).unwrap();
        let cam_count = entries
            .iter()
            .filter(|e| matches!(e, SensorEntry::Camera { .. }))
            .count();
        let imu_count = entries
            .iter()
            .filter(|e| matches!(e, SensorEntry::Imu { .. }))
            .count();
        assert_eq!(cam_count, 2);
        assert_eq!(imu_count, 1);
    }

    #[test]
    fn sensor_discovery_empty_dir_returns_error() {
        let tmp = TempDir::new().unwrap();
        let result = discover_sensors(tmp.path());
        assert!(matches!(result, Err(AslError::NoSensorsFound { .. })));
    }

    #[test]
    fn invalid_mav0_path_returns_error() {
        let result = AslReader::load("/this/path/does/not/exist");
        assert!(matches!(result, Err(AslError::InvalidMav0Directory { .. })));
    }

    // --- AslDataset accessors ---

    fn make_synthetic_dataset() -> AslDataset {
        let tmp = TempDir::new().unwrap();
        let mav0 = tmp.path().to_path_buf();

        // cam0
        let cam0_dir = mav0.join("cam0");
        let cam0_data = cam0_dir.join("data");
        std::fs::create_dir_all(&cam0_data).unwrap();
        write_file(
            &cam0_dir,
            "data.csv",
            "#timestamp,filename\n100,100.png\n200,200.png\n",
        );

        // imu0
        let imu_dir = mav0.join("imu0");
        std::fs::create_dir_all(&imu_dir).unwrap();
        write_file(
            &imu_dir,
            "data.csv",
            "#ts,wx,wy,wz,ax,ay,az\n100,1.0,0.0,0.0,0.0,0.0,9.8\n",
        );

        // mocap0
        let mocap_dir = mav0.join("mocap0");
        std::fs::create_dir_all(&mocap_dir).unwrap();
        write_file(
            &mocap_dir,
            "data.csv",
            "#ts,px,py,pz,qw,qx,qy,qz\n100,0.0,0.0,0.0,1.0,0.0,0.0,0.0\n",
        );

        // Keep tmp alive by leaking for this test (ok in tests)
        let path = mav0.clone();
        std::mem::forget(tmp);
        AslReader::load(&path).expect("synthetic dataset should load")
    }

    #[test]
    fn dataset_accessors_correct() {
        let ds = make_synthetic_dataset();
        assert_eq!(ds.camera_count(), 1);
        assert_eq!(ds.imu_sample_count(), 1);
        assert!(ds.has_ground_truth());
    }
}
