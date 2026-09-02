use std::path::PathBuf;

use nalgebra::Vector3;

use crate::trajectory::TrajectoryPose;

/// One ground-truth pose from an ASL dataset.
///
/// The trajectory module's [`TrajectoryPose`] under a dataset-reader name:
/// same fields, same meaning, so a ground-truth track converts to a
/// [`Trajectory`](crate::trajectory::Trajectory) by copying rather than by
/// converting.
pub type GroundTruthPose = TrajectoryPose;

#[derive(Debug, Clone)]
pub struct ImuMeasurement {
    pub timestamp_ns: u64,
    pub angular_velocity: Vector3<f64>,
    pub linear_acceleration: Vector3<f64>,
}

#[derive(Debug, Clone)]
pub struct CameraFrame {
    pub timestamp_ns: u64,
    pub image_path: PathBuf,
}

#[derive(Debug)]
pub struct CameraData {
    pub index: usize,
    pub frames: Vec<CameraFrame>,
    pub data_dir: PathBuf,
}

#[derive(Debug)]
pub struct AslDataset {
    pub cameras: Vec<CameraData>,
    pub imu_measurements: Vec<ImuMeasurement>,
    pub ground_truth: Option<Vec<GroundTruthPose>>,
    pub base_path: PathBuf,
}
