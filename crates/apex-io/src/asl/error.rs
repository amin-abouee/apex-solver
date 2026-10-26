//! Errors raised while reading an ASL (EuRoC-style) dataset folder.

use std::path::PathBuf;

/// An ASL dataset folder, CSV or image that cannot be read.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AslError {
    #[error("failed to read {}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("not a valid mav0 directory: {}", path.display())]
    InvalidMav0Directory { path: PathBuf },

    #[error("no sensors found in {}", path.display())]
    NoSensorsFound { path: PathBuf },

    #[error("{}:{line}: field {field} is not a number: {value:?}", path.display())]
    InvalidNumber {
        path: PathBuf,
        line: usize,
        field: &'static str,
        value: String,
    },

    #[error("{}:{line}: {got} CSV columns, expected at least {expected}", path.display())]
    MissingCsvColumns {
        path: PathBuf,
        line: usize,
        expected: usize,
        got: usize,
    },

    #[error("camera index {index} not found, the dataset has {count} cameras")]
    CameraNotFound { index: usize, count: usize },

    #[error("frame {frame_idx} out of bounds for camera {cam_idx} (total: {total})")]
    FrameOutOfBounds {
        cam_idx: usize,
        frame_idx: usize,
        total: usize,
    },

    #[error("failed to load image {}", path.display())]
    ImageLoad {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },

    #[cfg(feature = "asl-async")]
    #[error("the image loading task for {} did not finish", path.display())]
    ImageTask {
        path: PathBuf,
        #[source]
        source: tokio::task::JoinError,
    },

    /// Ground-truth CSV parsing is delegated to the trajectory reader, which
    /// is the one implementation of the ASL pose formats.
    #[error(transparent)]
    GroundTruth(#[from] crate::trajectory::TrajectoryError),
}

impl AslError {
    pub(crate) fn read(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Read {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn image(path: impl Into<PathBuf>, source: image::ImageError) -> Self {
        Self::ImageLoad {
            path: path.into(),
            source,
        }
    }
}

pub type Result<T> = std::result::Result<T, AslError>;

/// Result alias for ASL dataset operations, named like the other module aliases.
pub type AslResult<T> = Result<T>;
