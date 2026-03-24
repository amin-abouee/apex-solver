//! Bundle Adjustment in the Large (BAL) dataset loader.
//!
//! This module provides functionality to load BAL format files, which are used
//! for bundle adjustment benchmarks in computer vision.
//!
//! ## Format
//!
//! BAL files contain bundle adjustment problems with the following sequential structure:
//! 1. **Header line**: `<num_cameras> <num_points> <num_observations>`
//! 2. **Observations block**: One observation per line with format `<camera_idx> <point_idx> <x> <y>`
//! 3. **Cameras block**: 9 sequential parameter lines per camera (one value per line)
//! 4. **Points block**: 3 sequential coordinate lines per point (one value per line)
//!
//! ## Camera Model
//!
//! Uses Snavely's 9-parameter camera model from the Bundler structure-from-motion system:
//! - **Rotation**: 3D axis-angle representation (rx, ry, rz) - 3 parameters
//! - **Translation**: 3D vector (tx, ty, tz) - 3 parameters
//! - **Focal length**: Single parameter (f) - 1 parameter
//! - **Radial distortion**: Two coefficients (k1, k2) - 2 parameters
//!
//! For more details, see: <https://grail.cs.washington.edu/projects/bal/>
//!
//! ## Example
//!
//! ```no_run
//! use apex_io::BalLoader;
//!
//! let dataset = BalLoader::load("data/bundle_adjustment/problem-21-11315-pre.txt")?;
//! println!("Loaded {} cameras, {} points, {} observations",
//!          dataset.cameras.len(),
//!          dataset.points.len(),
//!          dataset.observations.len());
//! # Ok::<(), apex_io::IoError>(())
//! ```

use super::IoError;
use nalgebra::Vector3;
use std::fs::File;
use std::path::Path;

/// Represents a camera using Snavely's 9-parameter camera model.
///
/// The camera model from Bundler uses:
/// - Axis-angle rotation representation (compact 3-parameter rotation)
/// - 3D translation vector
/// - Single focal length parameter
/// - Two radial distortion coefficients
#[derive(Debug, Clone)]
pub struct BalCamera {
    /// Rotation as axis-angle representation (rx, ry, rz)
    pub rotation: Vector3<f64>,
    /// Translation vector (tx, ty, tz)
    pub translation: Vector3<f64>,
    /// Focal length
    pub focal_length: f64,
    /// First radial distortion coefficient
    pub k1: f64,
    /// Second radial distortion coefficient
    pub k2: f64,
}

/// Represents a 3D point (landmark) in the bundle adjustment problem.
#[derive(Debug, Clone)]
pub struct BalPoint {
    /// 3D position (x, y, z)
    pub position: Vector3<f64>,
}

/// Represents an observation of a 3D point by a camera.
///
/// Each observation links a camera to a 3D point via a 2D pixel measurement.
#[derive(Debug, Clone)]
pub struct BalObservation {
    /// Index of the observing camera
    pub camera_index: usize,
    /// Index of the observed 3D point
    pub point_index: usize,
    /// Pixel x-coordinate
    pub x: f64,
    /// Pixel y-coordinate
    pub y: f64,
}

/// Complete bundle adjustment dataset loaded from a BAL file.
#[derive(Debug, Clone)]
pub struct BalDataset {
    /// All cameras in the dataset
    pub cameras: Vec<BalCamera>,
    /// All 3D points (landmarks) in the dataset
    pub points: Vec<BalPoint>,
    /// All observations (camera-point correspondences)
    pub observations: Vec<BalObservation>,
}

/// Loader for BAL (Bundle Adjustment in the Large) dataset files.
pub struct BalLoader;

/// Default focal length used for cameras with negative or non-finite values.
/// This value is used during BAL dataset loading to normalize invalid focal lengths.
pub const DEFAULT_FOCAL_LENGTH: f64 = 500.0;

impl BalCamera {
    /// Normalizes the focal length to ensure it's valid for optimization.
    ///
    /// Replaces negative or non-finite focal lengths with DEFAULT_FOCAL_LENGTH,
    /// while preserving all positive values regardless of magnitude.
    fn normalize_focal_length(focal_length: f64) -> f64 {
        if focal_length > 0.0 && focal_length.is_finite() {
            focal_length
        } else {
            DEFAULT_FOCAL_LENGTH
        }
    }
}

impl BalLoader {
    /// Loads a BAL dataset from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the BAL format file
    ///
    /// # Returns
    ///
    /// Returns a `BalDataset` containing all cameras, points, and observations,
    /// or an `IoError` if parsing fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use apex_io::BalLoader;
    ///
    /// let dataset = BalLoader::load("data/bundle_adjustment/problem-21-11315-pre.txt")?;
    /// assert_eq!(dataset.cameras.len(), 21);
    /// assert_eq!(dataset.points.len(), 11315);
    /// # Ok::<(), apex_io::IoError>(())
    /// ```
    pub fn load(path: impl AsRef<Path>) -> Result<BalDataset, IoError> {
        // Open file with error context
        let file = File::open(path.as_ref()).map_err(|e| {
            IoError::Io(e).log_with_source(format!("Failed to open BAL file: {:?}", path.as_ref()))
        })?;

        // Memory-map file for performance (following g2o.rs pattern)
        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|e| {
                IoError::Io(e).log_with_source("Failed to memory-map BAL file".to_string())
            })?
        };

        // Convert to UTF-8 string
        let content = std::str::from_utf8(&mmap).map_err(|_| IoError::Parse {
            line: 0,
            message: "File is not valid UTF-8".to_string(),
        })?;

        // Create line iterator (skip empty lines, trim whitespace)
        let mut lines = content
            .lines()
            .enumerate()
            .map(|(idx, line)| (idx + 1, line.trim()))
            .filter(|(_, line)| !line.is_empty());

        // Parse header
        let (num_cameras, num_points, num_observations) = Self::parse_header(&mut lines)?;

        // Parse observations
        let observations = Self::parse_observations(&mut lines, num_observations)?;

        // Parse cameras
        let cameras = Self::parse_cameras(&mut lines, num_cameras)?;

        // Parse points
        let points = Self::parse_points(&mut lines, num_points)?;

        // Validate counts match header
        if cameras.len() != num_cameras {
            return Err(IoError::Parse {
                line: 0,
                message: format!(
                    "Camera count mismatch: header says {}, got {}",
                    num_cameras,
                    cameras.len()
                ),
            });
        }

        if points.len() != num_points {
            return Err(IoError::Parse {
                line: 0,
                message: format!(
                    "Point count mismatch: header says {}, got {}",
                    num_points,
                    points.len()
                ),
            });
        }

        if observations.len() != num_observations {
            return Err(IoError::Parse {
                line: 0,
                message: format!(
                    "Observation count mismatch: header says {}, got {}",
                    num_observations,
                    observations.len()
                ),
            });
        }

        Ok(BalDataset {
            cameras,
            points,
            observations,
        })
    }

    /// Parses the header line containing dataset dimensions.
    fn parse_header<'a>(
        lines: &mut impl Iterator<Item = (usize, &'a str)>,
    ) -> Result<(usize, usize, usize), IoError> {
        let (line_num, header_line) = lines.next().ok_or(IoError::Parse {
            line: 1,
            message: "Missing header line".to_string(),
        })?;

        let parts: Vec<&str> = header_line.split_whitespace().collect();
        if parts.len() != 3 {
            return Err(IoError::MissingFields { line: line_num });
        }

        let num_cameras = parts[0]
            .parse::<usize>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[0].to_string(),
            })?;

        let num_points = parts[1]
            .parse::<usize>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[1].to_string(),
            })?;

        let num_observations = parts[2]
            .parse::<usize>()
            .map_err(|_| IoError::InvalidNumber {
                line: line_num,
                value: parts[2].to_string(),
            })?;

        Ok((num_cameras, num_points, num_observations))
    }

    /// Parses the observations block.
    fn parse_observations<'a>(
        lines: &mut impl Iterator<Item = (usize, &'a str)>,
        num_observations: usize,
    ) -> Result<Vec<BalObservation>, IoError> {
        let mut observations = Vec::with_capacity(num_observations);

        for _ in 0..num_observations {
            let (line_num, line) = lines.next().ok_or(IoError::Parse {
                line: 0,
                message: "Unexpected end of file in observations section".to_string(),
            })?;

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 4 {
                return Err(IoError::MissingFields { line: line_num });
            }

            let camera_index = parts[0]
                .parse::<usize>()
                .map_err(|_| IoError::InvalidNumber {
                    line: line_num,
                    value: parts[0].to_string(),
                })?;

            let point_index = parts[1]
                .parse::<usize>()
                .map_err(|_| IoError::InvalidNumber {
                    line: line_num,
                    value: parts[1].to_string(),
                })?;

            let x = parts[2]
                .parse::<f64>()
                .map_err(|_| IoError::InvalidNumber {
                    line: line_num,
                    value: parts[2].to_string(),
                })?;

            let y = parts[3]
                .parse::<f64>()
                .map_err(|_| IoError::InvalidNumber {
                    line: line_num,
                    value: parts[3].to_string(),
                })?;

            observations.push(BalObservation {
                camera_index,
                point_index,
                x,
                y,
            });
        }

        Ok(observations)
    }

    /// Parses the cameras block.
    ///
    /// Each camera has 9 parameters on sequential lines:
    /// - 3 lines for rotation (rx, ry, rz)
    /// - 3 lines for translation (tx, ty, tz)
    /// - 1 line for focal length (f)
    /// - 2 lines for radial distortion (k1, k2)
    fn parse_cameras<'a>(
        lines: &mut impl Iterator<Item = (usize, &'a str)>,
        num_cameras: usize,
    ) -> Result<Vec<BalCamera>, IoError> {
        let mut cameras = Vec::with_capacity(num_cameras);

        for camera_idx in 0..num_cameras {
            let mut params = Vec::with_capacity(9);

            // Read 9 consecutive lines for camera parameters
            for param_idx in 0..9 {
                let (line_num, line) = lines.next().ok_or(IoError::Parse {
                    line: 0,
                    message: format!(
                        "Unexpected end of file in camera {} parameter {}",
                        camera_idx, param_idx
                    ),
                })?;

                let value = line
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| IoError::InvalidNumber {
                        line: line_num,
                        value: line.to_string(),
                    })?;

                params.push(value);
            }

            cameras.push(BalCamera {
                rotation: Vector3::new(params[0], params[1], params[2]),
                translation: Vector3::new(params[3], params[4], params[5]),
                focal_length: BalCamera::normalize_focal_length(params[6]),
                k1: params[7],
                k2: params[8],
            });
        }

        Ok(cameras)
    }

    /// Parses the points block.
    ///
    /// Each point has 3 coordinates on sequential lines (x, y, z).
    fn parse_points<'a>(
        lines: &mut impl Iterator<Item = (usize, &'a str)>,
        num_points: usize,
    ) -> Result<Vec<BalPoint>, IoError> {
        let mut points = Vec::with_capacity(num_points);

        for point_idx in 0..num_points {
            let mut coords = Vec::with_capacity(3);

            // Read 3 consecutive lines for point coordinates
            for coord_idx in 0..3 {
                let (line_num, line) = lines.next().ok_or(IoError::Parse {
                    line: 0,
                    message: format!(
                        "Unexpected end of file in point {} coordinate {}",
                        point_idx, coord_idx
                    ),
                })?;

                let value = line
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| IoError::InvalidNumber {
                        line: line_num,
                        value: line.to_string(),
                    })?;

                coords.push(value);
            }

            points.push(BalPoint {
                position: Vector3::new(coords[0], coords[1], coords[2]),
            });
        }

        Ok(points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Writes a minimal BAL file: 1 camera, 1 point, 1 observation.
    fn write_minimal_bal() -> Result<NamedTempFile, Box<dyn std::error::Error>> {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1 1")?; // header
        writeln!(f, "0 0 -123.456 456.789")?; // observation
        // Camera params (9 values, one per line): rx ry rz tx ty tz f k1 k2
        for v in [0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 500.0, -0.1, 0.05] {
            writeln!(f, "{v}")?;
        }
        // Point coords (3 values, one per line): x y z
        for v in [1.0f64, 2.0, 3.0] {
            writeln!(f, "{v}")?;
        }
        f.flush()?;
        Ok(f)
    }

    /// Writes a BAL file with a custom focal length value (all other params are zeros).
    fn write_bal_with_focal(focal: f64) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1 1")?;
        writeln!(f, "0 0 0.0 0.0")?; // observation
        // Camera params: rx ry rz tx ty tz f k1 k2
        for v in [0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, focal, 0.0, 0.0] {
            writeln!(f, "{v}")?;
        }
        // Point coords
        for v in [0.0f64, 0.0, 0.0] {
            writeln!(f, "{v}")?;
        }
        f.flush()?;
        Ok(f)
    }

    #[test]
    fn test_load_minimal_dataset() -> TestResult {
        let f = write_minimal_bal()?;
        let ds = BalLoader::load(f.path())?;
        assert_eq!(ds.cameras.len(), 1);
        assert_eq!(ds.points.len(), 1);
        assert_eq!(ds.observations.len(), 1);
        Ok(())
    }

    #[test]
    fn test_load_camera_values() -> TestResult {
        let f = write_minimal_bal()?;
        let ds = BalLoader::load(f.path())?;
        let cam = &ds.cameras[0];
        assert!((cam.rotation.x - 0.1).abs() < 1e-12);
        assert!((cam.rotation.y - 0.2).abs() < 1e-12);
        assert!((cam.rotation.z - 0.3).abs() < 1e-12);
        assert!((cam.translation.x - 0.4).abs() < 1e-12);
        assert!((cam.translation.y - 0.5).abs() < 1e-12);
        assert!((cam.translation.z - 0.6).abs() < 1e-12);
        assert!((cam.focal_length - 500.0).abs() < 1e-12);
        assert!((cam.k1 - (-0.1)).abs() < 1e-12);
        assert!((cam.k2 - 0.05).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_load_observation_values() -> TestResult {
        let f = write_minimal_bal()?;
        let ds = BalLoader::load(f.path())?;
        let obs = &ds.observations[0];
        assert_eq!(obs.camera_index, 0);
        assert_eq!(obs.point_index, 0);
        assert!((obs.x - (-123.456)).abs() < 1e-10);
        assert!((obs.y - 456.789).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_load_point_values() -> TestResult {
        let f = write_minimal_bal()?;
        let ds = BalLoader::load(f.path())?;
        let pt = &ds.points[0];
        assert!((pt.position.x - 1.0).abs() < 1e-12);
        assert!((pt.position.y - 2.0).abs() < 1e-12);
        assert!((pt.position.z - 3.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_normalize_focal_length_negative_uses_default() -> TestResult {
        let f = write_bal_with_focal(-100.0)?;
        let ds = BalLoader::load(f.path())?;
        assert!(
            (ds.cameras[0].focal_length - DEFAULT_FOCAL_LENGTH).abs() < 1e-12,
            "negative focal length should be replaced with DEFAULT_FOCAL_LENGTH"
        );
        Ok(())
    }

    #[test]
    fn test_normalize_focal_length_zero_uses_default() -> TestResult {
        let f = write_bal_with_focal(0.0)?;
        let ds = BalLoader::load(f.path())?;
        assert!(
            (ds.cameras[0].focal_length - DEFAULT_FOCAL_LENGTH).abs() < 1e-12,
            "zero focal length should be replaced with DEFAULT_FOCAL_LENGTH"
        );
        Ok(())
    }

    #[test]
    fn test_normalize_focal_length_positive_preserved() -> TestResult {
        let f = write_bal_with_focal(300.0)?;
        let ds = BalLoader::load(f.path())?;
        assert!(
            (ds.cameras[0].focal_length - 300.0).abs() < 1e-12,
            "positive focal length should be preserved"
        );
        Ok(())
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = BalLoader::load("/nonexistent/path/file.bal");
        assert!(result.is_err(), "loading a missing file should return Err");
    }

    #[test]
    fn test_load_empty_file() -> TestResult {
        let f = NamedTempFile::new()?;
        let result = BalLoader::load(f.path());
        assert!(result.is_err(), "empty file should fail (missing header)");
        Ok(())
    }

    #[test]
    fn test_load_header_wrong_field_count() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1")?; // only 2 fields, need 3
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(result.is_err(), "header with 2 fields should fail");
        Ok(())
    }

    #[test]
    fn test_load_header_invalid_number() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 abc 1")?;
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(result.is_err(), "non-numeric header field should fail");
        Ok(())
    }

    #[test]
    fn test_load_truncated_observations() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1 2")?; // claims 2 observations
        writeln!(f, "0 0 1.0 1.0")?; // only 1 provided
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(result.is_err(), "truncated observation block should fail");
        Ok(())
    }

    #[test]
    fn test_load_truncated_cameras() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1 1")?;
        writeln!(f, "0 0 1.0 1.0")?; // observation
        // Only 5 of the 9 required camera params
        for v in [0.0f64, 0.0, 0.0, 0.0, 0.0] {
            writeln!(f, "{v}")?;
        }
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(result.is_err(), "truncated camera block should fail");
        Ok(())
    }

    #[test]
    fn test_load_multiple_cameras_and_points() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "2 2 3")?; // 2 cameras, 2 points, 3 observations
        writeln!(f, "0 0 1.0 1.0")?;
        writeln!(f, "0 1 2.0 2.0")?;
        writeln!(f, "1 0 3.0 3.0")?;
        // Camera 0
        for v in [0.0f64; 9] {
            writeln!(f, "{v}")?;
        }
        // Camera 1
        for _ in 0..8 {
            writeln!(f, "0.0")?;
        }
        writeln!(f, "200.0")?; // focal_length = 200
        // Point 0
        for v in [1.0f64, 2.0, 3.0] {
            writeln!(f, "{v}")?;
        }
        // Point 1
        for v in [4.0f64, 5.0, 6.0] {
            writeln!(f, "{v}")?;
        }
        f.flush()?;
        let ds = BalLoader::load(f.path())?;
        assert_eq!(ds.cameras.len(), 2);
        assert_eq!(ds.points.len(), 2);
        assert_eq!(ds.observations.len(), 3);
        Ok(())
    }

    #[test]
    fn test_load_observation_invalid_number() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1 1")?;
        writeln!(f, "0 0 bad_x 1.0")?; // bad x coordinate
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(
            result.is_err(),
            "invalid observation coordinate should fail"
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // parse_header additional error paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_header_invalid_num_cameras() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "bad 1 1")?;
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid num_cameras should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_load_header_invalid_num_observations() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1 bad")?;
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid num_observations should return InvalidNumber"
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // parse_observations additional error paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_observation_missing_fields() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1 1")?;
        writeln!(f, "0 1.0")?; // only 2 fields, need 4
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::MissingFields { .. })),
            "observation with too few fields should return MissingFields"
        );
        Ok(())
    }

    #[test]
    fn test_load_observation_invalid_camera_index() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1 1")?;
        writeln!(f, "bad 0 1.0 2.0")?;
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid camera_index in observation should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_load_observation_invalid_point_index() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1 1")?;
        writeln!(f, "0 bad 1.0 2.0")?;
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid point_index in observation should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_load_observation_invalid_y() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1 1")?;
        writeln!(f, "0 0 1.0 bad")?;
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid y in observation should return InvalidNumber"
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // parse_cameras and parse_points additional error paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_load_camera_invalid_parameter() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1 1")?;
        writeln!(f, "0 0 1.0 1.0")?; // observation
        writeln!(f, "bad")?; // invalid first camera parameter (rx)
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid camera parameter should return InvalidNumber"
        );
        Ok(())
    }

    #[test]
    fn test_load_truncated_points() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1 1")?;
        writeln!(f, "0 0 1.0 1.0")?; // observation
        // Full camera block
        for v in [0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 0.0, 0.0] {
            writeln!(f, "{v}")?;
        }
        // Only 2 of 3 point coordinates
        writeln!(f, "1.0")?;
        writeln!(f, "2.0")?;
        // missing z
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(result.is_err(), "truncated point block should fail");
        Ok(())
    }

    #[test]
    fn test_load_point_invalid_coordinate() -> TestResult {
        let mut f = NamedTempFile::new()?;
        writeln!(f, "1 1 1")?;
        writeln!(f, "0 0 1.0 1.0")?; // observation
        // Full camera block
        for v in [0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 0.0, 0.0] {
            writeln!(f, "{v}")?;
        }
        writeln!(f, "bad")?; // invalid point x
        f.flush()?;
        let result = BalLoader::load(f.path());
        assert!(
            matches!(result, Err(IoError::InvalidNumber { .. })),
            "invalid point coordinate should return InvalidNumber"
        );
        Ok(())
    }
}
