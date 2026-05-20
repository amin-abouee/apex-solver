use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum AslError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("not a valid mav0 directory: {path}")]
    InvalidMav0Directory { path: PathBuf },

    #[error("no sensors found in: {path}")]
    NoSensorsFound { path: PathBuf },

    #[error("invalid CSV at {path}:{line}: {message}")]
    InvalidCsvFormat {
        path: PathBuf,
        line: usize,
        message: String,
    },

    #[error("invalid number at {path}:{line} field '{field}': '{value}'")]
    InvalidNumber {
        path: PathBuf,
        line: usize,
        field: String,
        value: String,
    },

    #[error("missing CSV columns at {path}:{line}: expected {expected}, got {got}")]
    MissingCsvColumns {
        path: PathBuf,
        line: usize,
        expected: usize,
        got: usize,
    },

    #[error("invalid quaternion at {path}:{line}: norm={norm:.6}")]
    InvalidQuaternion {
        path: PathBuf,
        line: usize,
        norm: f64,
    },

    #[error("camera index {index} not found")]
    CameraNotFound { index: usize },

    #[error("frame {frame_idx} out of bounds for camera {cam_idx} (total: {total})")]
    FrameOutOfBounds {
        cam_idx: usize,
        frame_idx: usize,
        total: usize,
    },

    #[error("image load error: {0}")]
    ImageLoad(#[from] image::ImageError),
}

pub type Result<T> = std::result::Result<T, AslError>;
