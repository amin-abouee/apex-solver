//! The unified pull API over real recordings.
//!
//! Skipped unless the sample variables are set:
//!
//! ```text
//! APEX_VRS_SAMPLE=/data/lamaria/.../R_06_medium.vrs \
//! APEX_BENCH_MCAP=/data/uzh_fpv/mcap/indoor_forward_10_snapdragon_with_gt.mcap \
//!   cargo test -p apex-io --features rosbag,vrs --release --test bag_pull_real
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use apex_io::rosbag::{BagReader, Selection, SensorConfig};

fn pulled(r: &mut BagReader, sel: &Selection) -> Vec<(u32, u64, Vec<u8>)> {
    let mut it = r.iter(sel).unwrap();
    let mut buf = Vec::new();
    let mut out = Vec::new();
    while let Some((id, ts)) = it.next_into(&mut buf).unwrap() {
        out.push((id, ts, buf.clone()));
    }
    out
}

fn callback(r: &mut BagReader, sel: &Selection) -> Vec<(u32, u64, Vec<u8>)> {
    let conns = r.selected_connections(sel);
    let mut out = Vec::new();
    r.for_each(&conns, sel, |c, ts, d| {
        out.push((c.id, ts, d.to_vec()));
        Ok(())
    })
    .unwrap();
    out
}

#[test]
fn aria_vrs_params_and_pull_match_the_callback_api() {
    let Ok(path) = std::env::var("APEX_VRS_SAMPLE") else {
        eprintln!("APEX_VRS_SAMPLE not set; skipping");
        return;
    };
    let mut bag = BagReader::open(&path).unwrap();

    let params = bag.params().unwrap();
    assert_eq!(
        params.file_tags.get("device_type").map(String::as_str),
        Some("Ariane")
    );
    let calib = params.calibration.as_ref().expect("factory calibration");
    assert!(calib.get("CameraCalibrations").is_some());
    let slam = params
        .sensors
        .iter()
        .find(|s| s.topic == "/aria/camera_slam_left/compressed")
        .expect("left SLAM camera");
    assert!(matches!(
        slam.config,
        SensorConfig::Camera {
            width: 640,
            height: 480,
            ..
        }
    ));
    let imu = params
        .sensors
        .iter()
        .find(|s| s.topic == "/aria/imu_right")
        .unwrap();
    assert!(matches!(
        imu.config,
        SensorConfig::Imu {
            has_accelerometer: true,
            has_gyroscope: true,
            ..
        }
    ));

    // 5 s of IMU + left camera, 10 s into the recording.
    let start = bag.start_time() + 10_000_000_000;
    let sel = Selection {
        include: vec![
            "/aria/imu_right".into(),
            "/aria/camera_slam_left/compressed".into(),
        ],
        start_ns: Some(start),
        end_ns: Some(start + 5_000_000_000),
        ..Selection::default()
    };
    let a = callback(&mut bag, &sel);
    let b = pulled(&mut bag, &sel);
    assert_eq!(a, b);
    assert!(
        (4_900..5_200).contains(&a.len()),
        "~1 kHz IMU + 20 Hz camera for 5 s, got {}",
        a.len()
    );
    // Emitted stamps are capture times; ordering follows record times, so
    // sub-millisecond inversions are the most that can happen.
    assert!(b.windows(2).all(|w| w[1].1 + 2_000_000 >= w[0].1));
}

#[test]
fn real_mcap_pull_matches_the_callback_api() {
    let Ok(path) = std::env::var("APEX_BENCH_MCAP") else {
        eprintln!("APEX_BENCH_MCAP not set; skipping");
        return;
    };
    let mut bag = BagReader::open(&path).unwrap();
    let topic = bag
        .connections()
        .iter()
        .max_by_key(|c| c.message_count)
        .map(|c| c.topic.clone())
        .unwrap();
    let sel = Selection {
        include: vec![topic],
        ..Selection::default()
    };
    let a = callback(&mut bag, &sel);
    let b = pulled(&mut bag, &sel);
    assert!(!a.is_empty());
    assert_eq!(a, b);
    assert!(
        b.windows(2).all(|w| w[1].1 >= w[0].1),
        "pull must be time ordered"
    );
}
