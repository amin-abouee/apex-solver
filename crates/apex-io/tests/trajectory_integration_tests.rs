//! Integration tests for `apex_io::trajectory`.
//!
//! These exercise only the public API — format dispatch, cross-format
//! agreement, and real datasets when present — so they belong beside the crate
//! rather than inflating the modules they test.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use apex_io::asl::{AslLayout, AslTrajectoryLoader, load_mav0_trajectory};
use apex_io::trajectory::{
    Trajectory, TrajectoryError, TrajectoryFormat, TrajectoryLoader, TrajectoryPose, TumLoader,
    load_trajectory, load_trajectory_as,
};
use apex_manifolds::so3::SO3;
use nalgebra::Vector3;
use tempfile::TempDir;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A machine-local dataset. Tests needing it skip when it is absent.
const EUROC_MAV0: &str =
    "/Volumes/External/Workspace/rust/apex-calibration/data/dataset-calib-imu1_1024_16/mav0";

macro_rules! require_dataset {
    ($path:expr) => {
        if !Path::new($path).is_dir() {
            eprintln!("Skipping: dataset not available at {}", $path);
            return Ok(());
        }
    };
}

/// A rotation with four distinct components of different magnitude, so any
/// permutation of the quaternion fields is detectable.
fn asymmetric_rotation() -> SO3 {
    let axis = Vector3::new(0.3, 0.7, 0.1);
    SO3::from_axis_angle(&axis, 0.7)
}

fn sample_trajectory() -> Trajectory {
    Trajectory::from_poses(vec![
        TrajectoryPose::new(
            1_403_636_579_763_555_584,
            Vector3::new(1.0, -2.0, 3.0),
            asymmetric_rotation(),
        ),
        TrajectoryPose::new(
            1_403_636_579_813_555_584,
            Vector3::new(1.5, -2.5, 3.5),
            SO3::identity(),
        ),
    ])
}

/// **The w-first / w-last regression test.**
///
/// A symmetric flip — writer and reader both wrong in the same way — round-trips
/// perfectly inside a single codec, so no per-format test can catch it. Routing
/// the *same* rotation through both codecs does: a flip in either one makes the
/// two disagree.
#[test]
fn cross_format_round_trip_preserves_rotation() -> TestResult {
    let dir = TempDir::new()?;
    let original = sample_trajectory();

    let tum_path = dir.path().join("t.tum");
    let asl_path = dir.path().join("t.csv");
    TumLoader::write(&original, &tum_path)?;
    AslTrajectoryLoader::write(&original, &asl_path)?;

    let via_tum = TumLoader::load(&tum_path)?;
    let via_asl = AslTrajectoryLoader::load(&asl_path)?;

    assert_eq!(via_tum.len(), via_asl.len());
    for (a, b) in via_tum.poses().iter().zip(via_asl.poses().iter()) {
        let angle = a.orientation.distance(&b.orientation);
        assert!(
            angle < 1e-8,
            "the two codecs disagree by {angle:e} rad — one of them has its \
             quaternion fields in the wrong order"
        );
        assert!((a.position - b.position).norm() < 1e-8);
    }
    Ok(())
}

#[test]
fn dispatch_reads_a_tum_file_by_extension() -> TestResult {
    let dir = TempDir::new()?;
    let path = dir.path().join("traj.tum");
    TumLoader::write(&sample_trajectory(), &path)?;
    assert_eq!(load_trajectory(&path)?.len(), 2);
    Ok(())
}

#[test]
fn dispatch_reads_an_asl_csv_by_extension() -> TestResult {
    let dir = TempDir::new()?;
    let path = dir.path().join("data.csv");
    AslTrajectoryLoader::write(&sample_trajectory(), &path)?;
    assert_eq!(load_trajectory(&path)?.len(), 2);
    Ok(())
}

#[test]
fn dispatch_rejects_an_unsupported_extension() -> TestResult {
    let Err(TrajectoryError::UnsupportedFormat(ext)) = load_trajectory("trajectory.bin") else {
        return Err("an unknown extension must be refused".into());
    };
    assert_eq!(ext, "bin");
    Ok(())
}

/// Extension dispatch is weaker for trajectories than for pose graphs, because
/// `.csv` and `.txt` are not format names. What makes that safe is that a
/// misdispatch fails loudly: the separators disagree, so the row splits into a
/// single field and the column check rejects it.
#[test]
fn an_asl_csv_fed_to_the_tum_loader_errors_rather_than_misparsing() -> TestResult {
    let dir = TempDir::new()?;
    let path = dir.path().join("data.csv");
    AslTrajectoryLoader::write(&sample_trajectory(), &path)?;

    let Err(TrajectoryError::MissingColumns { .. }) =
        load_trajectory_as(&path, TrajectoryFormat::Tum)
    else {
        return Err("an ASL file must not parse as TUM".into());
    };
    Ok(())
}

#[test]
fn a_tum_file_fed_to_the_asl_loader_errors_rather_than_misparsing() -> TestResult {
    let dir = TempDir::new()?;
    let path = dir.path().join("traj.tum");
    TumLoader::write(&sample_trajectory(), &path)?;

    let Err(TrajectoryError::UnexpectedColumnCount { .. }) =
        load_trajectory_as(&path, TrajectoryFormat::Asl)
    else {
        return Err("a TUM file must not parse as ASL".into());
    };
    Ok(())
}

#[test]
fn mav0_prefers_state_groundtruth_over_mocap() -> TestResult {
    let dir = TempDir::new()?;
    let mav0 = dir.path().join("mav0");

    // mocap0 is pose-only; state_groundtruth_estimate0 carries the inertial
    // columns. Preferring the richer source is what the caller relies on.
    for (sensor, row) in [
        ("mocap0", "1000,1,2,3,1,0,0,0\n"),
        (
            "state_groundtruth_estimate0",
            "1000,1,2,3,1,0,0,0,4,5,6,7,8,9,10,11,12\n",
        ),
    ] {
        let sensor_dir = mav0.join(sensor);
        std::fs::create_dir_all(&sensor_dir)?;
        std::fs::write(sensor_dir.join("data.csv"), row)?;
    }

    let trajectory = load_mav0_trajectory(dir.path())?.ok_or("ground truth must be found")?;
    assert!(
        trajectory.has_inertial(),
        "the 17-column source must win over mocap0"
    );
    Ok(())
}

#[test]
fn mav0_falls_back_to_mocap() -> TestResult {
    let dir = TempDir::new()?;
    let sensor_dir = dir.path().join("mav0").join("mocap0");
    std::fs::create_dir_all(&sensor_dir)?;
    std::fs::write(sensor_dir.join("data.csv"), "1000,1,2,3,1,0,0,0\n")?;

    let trajectory = load_mav0_trajectory(dir.path())?.ok_or("mocap0 must be found")?;
    assert_eq!(trajectory.len(), 1);
    assert!(!trajectory.has_inertial());
    Ok(())
}

#[test]
fn mav0_without_ground_truth_is_none() -> TestResult {
    let dir = TempDir::new()?;
    std::fs::create_dir_all(dir.path().join("mav0").join("cam0"))?;
    assert!(load_mav0_trajectory(dir.path())?.is_none());
    Ok(())
}

/// The `mav0` sub-directory is optional, matching `AslReader::load`.
#[test]
fn mav0_accepts_either_the_root_or_the_mav0_directory() -> TestResult {
    let dir = TempDir::new()?;
    let sensor_dir = dir.path().join("mocap0");
    std::fs::create_dir_all(&sensor_dir)?;
    std::fs::write(sensor_dir.join("data.csv"), "1000,1,2,3,1,0,0,0\n")?;

    assert!(load_mav0_trajectory(dir.path())?.is_some());
    Ok(())
}

/// A file that exists but is malformed must be an error, never a silent `None`
/// — otherwise a corrupt ground truth reads as "this sequence has none".
#[test]
fn a_malformed_ground_truth_file_is_an_error_not_none() -> TestResult {
    let dir = TempDir::new()?;
    let sensor_dir = dir.path().join("mav0").join("mocap0");
    std::fs::create_dir_all(&sensor_dir)?;
    std::fs::write(sensor_dir.join("data.csv"), "1000,1,2\n")?;

    if load_mav0_trajectory(dir.path()).is_ok() {
        return Err("a malformed file must not read as absent ground truth".into());
    }
    Ok(())
}

#[test]
fn asl_euroc_layout_round_trips_through_a_real_file() -> TestResult {
    let dir = TempDir::new()?;
    let path = dir.path().join("gt.csv");

    let poses = vec![TrajectoryPose::new(
        1_403_636_579_763_555_584,
        Vector3::new(1.0, 2.0, 3.0),
        asymmetric_rotation(),
    )];
    let inertial = vec![apex_io::trajectory::InertialState {
        velocity: Vector3::new(10.0, 11.0, 12.0),
        gyro_bias: Vector3::new(20.0, 21.0, 22.0),
        accel_bias: Vector3::new(30.0, 31.0, 32.0),
    }];
    let original = Trajectory::from_poses_and_inertial(poses, inertial)?;

    AslTrajectoryLoader::write_as(&original, &path, AslLayout::Euroc17)?;
    let parsed = AslTrajectoryLoader::load(&path)?;

    let a = original.inertial().ok_or("a")?.first().ok_or("a0")?;
    let b = parsed.inertial().ok_or("b")?.first().ok_or("b0")?;
    assert!((a.velocity - b.velocity).norm() < 1e-12);
    assert!((a.gyro_bias - b.gyro_bias).norm() < 1e-12);
    assert!((a.accel_bias - b.accel_bias).norm() < 1e-12);
    Ok(())
}

#[test]
fn a_real_asl_mocap_track_loads_pose_only() -> TestResult {
    require_dataset!(EUROC_MAV0);
    let trajectory =
        load_mav0_trajectory(EUROC_MAV0)?.ok_or("this sequence should ship ground truth")?;

    assert!(!trajectory.is_empty());
    let (first, last) = trajectory.time_span_ns().ok_or("a non-empty span")?;
    assert!(last > first, "timestamps must be ascending");

    // A query inside the span resolves; one outside is refused.
    assert!(trajectory.position_at(first + (last - first) / 2).is_some());
    assert!(trajectory.position_at(first - 1).is_none());
    Ok(())
}
