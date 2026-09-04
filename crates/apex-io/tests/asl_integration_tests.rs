//! Integration tests against real EuRoC/ASL data from the apex-calibration project.
//! Tests are skipped when the dataset is not present on the current machine.
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Test diagnostics below print under `--nocapture` only.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use apex_io::AslError;
use apex_io::asl::{AslDataset, AslReader};
use std::path::Path;

const EUROC_MAV0: &str =
    "/Volumes/External/Workspace/rust/apex-calibration/data/dataset-calib-imu1_1024_16/mav0";

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn load_dataset() -> Option<AslDataset> {
    if !Path::new(EUROC_MAV0).exists() {
        return None;
    }
    Some(AslReader::load(EUROC_MAV0).expect("failed to load EuRoC dataset"))
}

macro_rules! require_dataset {
    ($ds:ident) => {
        let Some($ds) = load_dataset() else {
            eprintln!("Skipping: dataset not available at {EUROC_MAV0}");
            return Ok(());
        };
    };
}

#[test]
fn discovers_two_cameras_imu_and_mocap() -> TestResult {
    require_dataset!(ds);
    assert_eq!(ds.camera_count(), 2, "expected 2 cameras (cam0 + cam1)");
    assert!(ds.imu_sample_count() > 0, "expected IMU measurements");
    assert!(ds.has_ground_truth(), "expected mocap ground truth");
    Ok(())
}

#[test]
fn imu_sample_count_matches_csv() -> TestResult {
    require_dataset!(ds);
    assert_eq!(ds.imu_sample_count(), 10345);
    Ok(())
}

#[test]
fn camera_frame_counts() -> TestResult {
    require_dataset!(ds);
    assert_eq!(ds.cameras[0].frames.len(), 1038, "cam0 frame count");
    assert_eq!(ds.cameras[1].frames.len(), 1038, "cam1 frame count");
    Ok(())
}

#[test]
fn ground_truth_sample_count() -> TestResult {
    require_dataset!(ds);
    let gt = ds.ground_truth.as_ref().unwrap();
    assert_eq!(gt.len(), 5696);
    Ok(())
}

#[test]
fn imu_first_sample_values() -> TestResult {
    require_dataset!(ds);
    let m = &ds.imu_measurements[0];
    assert_eq!(m.timestamp_ns, 1520527958474741167);
    assert!((m.angular_velocity.x - (-0.1007931761)).abs() < 1e-7);
    assert!((m.angular_velocity.y - 0.0516897516).abs() < 1e-7);
    assert!((m.angular_velocity.z - 0.0603467801).abs() < 1e-7);
    assert!((m.linear_acceleration.x - 0.9296921410).abs() < 1e-7);
    assert!((m.linear_acceleration.z - 9.7992666975).abs() < 1e-7);
    Ok(())
}

#[test]
fn camera_frame_timestamps() -> TestResult {
    require_dataset!(ds);
    assert_eq!(ds.cameras[0].frames[0].timestamp_ns, 1520527958463478167);
    assert_eq!(ds.cameras[1].frames[0].timestamp_ns, 1520527958463478167);
    Ok(())
}

#[test]
fn camera_frame_paths_are_absolute_and_exist() -> TestResult {
    require_dataset!(ds);
    for cam in &ds.cameras {
        for frame in cam.frames.iter().take(5) {
            assert!(
                frame.image_path.is_absolute(),
                "image path must be absolute: {}",
                frame.image_path.display()
            );
            assert!(
                frame.image_path.exists(),
                "image file must exist: {}",
                frame.image_path.display()
            );
        }
    }
    Ok(())
}

#[test]
fn ground_truth_first_pose_values() -> TestResult {
    require_dataset!(ds);
    let pose = &ds.ground_truth.as_ref().unwrap()[0];
    assert_eq!(pose.timestamp_ns, 1520527960237865414);
    assert!((pose.position.x - (-0.1986969748)).abs() < 1e-6);
    assert!((pose.position.y - (-0.1762659604)).abs() < 1e-6);
    assert!((pose.position.z - 0.5823374105).abs() < 1e-6);
    let q = pose.orientation.quaternion();
    assert!((q.w - 0.9994042349).abs() < 1e-6);
    Ok(())
}

#[test]
fn load_image_first_frame_succeeds() -> TestResult {
    require_dataset!(ds);
    let img = ds.load_image(0, 0)?;
    assert!(img.width() > 0 && img.height() > 0);
    Ok(())
}

#[test]
fn camera_not_found_returns_error() -> TestResult {
    require_dataset!(ds);
    let result = ds.load_image(99, 0);
    assert!(
        matches!(result, Err(AslError::CameraNotFound { index: 99 })),
        "unexpected: {result:?}"
    );
    Ok(())
}

#[test]
fn frame_out_of_bounds_returns_error() -> TestResult {
    require_dataset!(ds);
    let result = ds.load_image(0, 999_999);
    assert!(
        matches!(
            result,
            Err(AslError::FrameOutOfBounds {
                cam_idx: 0,
                frame_idx: 999_999,
                ..
            })
        ),
        "unexpected: {result:?}"
    );
    Ok(())
}

#[test]
fn invalid_mav0_path_returns_error() -> TestResult {
    let result = AslReader::load("/this/path/does/not/exist/mav0");
    assert!(
        matches!(result, Err(AslError::InvalidMav0Directory { .. })),
        "unexpected: {result:?}"
    );
    Ok(())
}
