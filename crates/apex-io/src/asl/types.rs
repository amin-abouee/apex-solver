use nalgebra::{UnitQuaternion, Vector3};
use std::path::PathBuf;

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

#[derive(Debug, Clone)]
pub struct GroundTruthPose {
    pub timestamp_ns: u64,
    pub position: Vector3<f64>,
    pub orientation: UnitQuaternion<f64>,
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
