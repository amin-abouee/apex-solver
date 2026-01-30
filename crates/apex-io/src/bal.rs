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
                focal_length: params[6],
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
