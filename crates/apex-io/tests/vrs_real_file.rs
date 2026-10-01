//! Checks the VRS reader against a real Project Aria recording.
//!
//! Skipped unless `APEX_VRS_SAMPLE` points at a `.vrs` file, e.g.
//!
//! ```text
//! APEX_VRS_SAMPLE=/data/lamaria/training/R_06_medium/raw_data/R_06_medium.vrs \
//!   cargo test -p apex-io --features vrs --release --test vrs_real_file
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use apex_io::vrs::{IndexSource, RecordFilter, RecordType, StreamId, VrsReader};

fn sample() -> Option<VrsReader> {
    let path = std::env::var("APEX_VRS_SAMPLE").ok()?;
    Some(VrsReader::open(path).expect("open sample"))
}

#[test]
fn index_streams_and_counts() {
    let Some(mut vrs) = sample() else {
        eprintln!("APEX_VRS_SAMPLE not set; skipping");
        return;
    };
    assert_eq!(vrs.index_source(), IndexSource::File);
    assert!(
        vrs.index_matches_scan().unwrap(),
        "index disagrees with a scan"
    );
    assert!(!vrs.streams().is_empty());

    // Every stream's data list is sorted and consistent with the records.
    let mut total_data = 0;
    for id in vrs.streams() {
        let d = vrs.data_records(id);
        total_data += d.len();
        assert!(d.windows(2).all(|w| {
            vrs.records()[w[0] as usize].timestamp <= vrs.records()[w[1] as usize].timestamp
        }));
        assert!(d.iter().all(|&i| {
            let r = &vrs.records()[i as usize];
            r.stream == id && r.record_type == RecordType::Data
        }));
    }
    let data_in_index = vrs
        .records()
        .iter()
        .filter(|r| r.record_type == RecordType::Data)
        .count();
    assert_eq!(total_data, data_in_index);
}

#[test]
fn cursor_visits_everything_once_in_time_order() {
    let Some(mut vrs) = sample() else { return };
    // Use a cheap subset: the busiest non-image stream.
    let stream = vrs
        .streams()
        .into_iter()
        .max_by_key(|s| vrs.data_records(*s).len())
        .unwrap();
    let expect = vrs.data_records(stream).len();
    let mut cur = vrs.cursor(&RecordFilter::streams([stream]));
    let (mut n, mut last) = (0usize, f64::MIN);
    let mut buf = Vec::new();
    while let Some(m) = vrs.next_record_into(&mut cur, &mut buf).unwrap() {
        assert_eq!(m.stream, stream);
        assert!(m.timestamp >= last, "timestamps went backwards");
        last = m.timestamp;
        n += 1;
    }
    assert_eq!(n, expect);
}

#[test]
fn nearest_and_range_queries_match_brute_force() {
    let Some(vrs) = sample() else { return };
    let stream = vrs
        .streams()
        .into_iter()
        .max_by_key(|s| vrs.data_records(*s).len())
        .unwrap();
    let (t0, t1) = vrs.stream_time_range(stream).unwrap();
    let times: Vec<f64> = vrs
        .data_records(stream)
        .iter()
        .map(|&i| vrs.records()[i as usize].timestamp)
        .collect();

    for k in 0..200 {
        let q = t0 - 0.5 + (t1 - t0 + 1.0) * f64::from(k) / 199.0;
        // nearest
        let got = vrs.find_nearest(stream, q).unwrap();
        let got_t = vrs.records()[got].timestamp;
        let best = times
            .iter()
            .map(|t| (t - q).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(((got_t - q).abs() - best).abs() < 1e-12, "q={q}");
        // at-or-before / at-or-after
        let before = vrs
            .find_at_or_before(stream, q)
            .map(|i| vrs.records()[i].timestamp);
        assert_eq!(
            before,
            times
                .iter()
                .copied()
                .filter(|t| *t <= q)
                .fold(None, |a: Option<f64>, t| Some(a.map_or(t, |x| x.max(t))))
        );
        // range
        let (a, b) = (q, q + 0.25);
        let in_range = vrs.data_records_between(stream, a, b).len();
        assert_eq!(
            in_range,
            times.iter().filter(|t| **t >= a && **t < b).count()
        );
    }
}

#[test]
fn payloads_decode_for_one_sample_per_stream() {
    let Some(mut vrs) = sample() else { return };
    let mut buf = Vec::new();
    for id in vrs.streams() {
        let Some(&mid) = vrs.data_records(id).get(vrs.data_records(id).len() / 2) else {
            continue;
        };
        let meta = vrs.read_record_into(mid as usize, &mut buf).unwrap();
        assert_eq!(meta.stream, id);
        assert!(!buf.is_empty(), "{id}: empty payload");
        let format = vrs
            .stream_tags(id)
            .and_then(|t| t.vrs.get(&format!("RF:Data:{}", meta.format_version)))
            .cloned()
            .unwrap_or_default();
        if format.contains("image/jpg") {
            // DataLayout first, then JPEG: find the SOI marker after the layout.
            let size: usize = format
                .split("size=")
                .nth(1)
                .and_then(|s| s.split('+').next())
                .and_then(|s| s.parse().ok())
                .unwrap();
            assert_eq!(
                &buf[size..size + 2],
                &[0xFF, 0xD8],
                "{id}: no JPEG after layout"
            );
            assert_eq!(
                &buf[buf.len() - 2..],
                &[0xFF, 0xD9],
                "{id}: JPEG not terminated"
            );
        }
    }
    // Wrong index is an error, not a panic.
    assert!(vrs.read_record_into(usize::MAX, &mut buf).is_err());
    let _ = StreamId::new(0, 0);
}

mod aria {
    use super::sample;
    use apex_io::vrs::VrsReader;
    use apex_io::vrs::aria::{AriaRecording, AriaSensor, Sample};

    fn recording() -> Option<AriaRecording> {
        let path = std::env::var("APEX_VRS_SAMPLE").ok()?;
        Some(AriaRecording::open(path).expect("open"))
    }

    #[test]
    fn configs_and_calibration() {
        let Some(mut rec) = recording() else { return };
        let cam = rec.camera_config(AriaSensor::SlamLeft).unwrap();
        assert_eq!((cam.width, cam.height), (640, 480));
        assert!(cam.nominal_rate > 0.0);
        let imu = rec.imu_config(AriaSensor::ImuRight).unwrap();
        assert!(imu.has_accelerometer && imu.has_gyroscope);
        let calib = rec.calibration_json().expect("calib_json file tag");
        assert!(calib.contains("CameraCalibrations"));
        assert!(rec.sensors().contains(&AriaSensor::ImuRight));
        let _ = sample;
    }

    #[test]
    fn imu_range_and_nearest_agree_with_the_index() {
        let Some(mut rec) = recording() else { return };
        let (t0, t1) = rec.time_range(AriaSensor::ImuRight).unwrap();
        let mid = (t0 + t1) / 2.0;
        let window = rec
            .imu_between(AriaSensor::ImuRight, mid, mid + 1.0)
            .unwrap();
        // ~1 kHz for one second.
        assert!(
            (900..=1100).contains(&window.len()),
            "{} samples",
            window.len()
        );
        assert!(window.windows(2).all(|w| w[0].time_s <= w[1].time_s));
        assert!(window.iter().all(|s| s.accel.is_some() && s.gyro.is_some()));
        // Gravity: |accel| about 9.8 m/s^2 for a handheld/walking recording.
        let a = window[0].accel.unwrap();
        let norm = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        assert!((5.0..15.0).contains(&norm), "|accel| = {norm}");
        // capture_ns and record time are the same clock.
        assert!((window[0].capture_ns as f64 * 1e-9 - window[0].time_s).abs() < 1e-3);

        let near = rec
            .imu_nearest(AriaSensor::ImuRight, mid + 0.5)
            .unwrap()
            .unwrap();
        assert!((near.time_s - (mid + 0.5)).abs() < 2e-3);
    }

    #[test]
    fn image_nearest_returns_a_jpeg_with_metadata() {
        let Some(mut rec) = recording() else { return };
        let (t0, t1) = rec.time_range(AriaSensor::SlamLeft).unwrap();
        let t = (t0 + t1) / 2.0;
        let f = rec.image_nearest(AriaSensor::SlamLeft, t).unwrap().unwrap();
        assert!((f.time_s - t).abs() < 0.06, "20 Hz camera");
        assert_eq!(&f.jpeg[..2], &[0xFF, 0xD8]);
        assert!(f.exposure_s > 0.0 && f.gain > 0.0);
        // Decodes as a real 640x480 image (the `image` crate is a dependency).
        let img = image::load_from_memory_with_format(&f.jpeg, image::ImageFormat::Jpeg).unwrap();
        assert_eq!((img.width(), img.height()), (640, 480));
    }

    #[test]
    fn merged_pull_is_time_ordered_and_complete() {
        let Some(mut rec) = recording() else { return };
        let (t0, _) = rec.time_range(AriaSensor::ImuRight).unwrap();
        let sensors = [
            AriaSensor::ImuRight,
            AriaSensor::SlamLeft,
            AriaSensor::Barometer,
        ];
        let mut cur = rec.cursor_between(&sensors, t0 + 10.0, t0 + 20.0);
        let (mut imu, mut img, mut baro, mut last) = (0, 0, 0, f64::MIN);
        let start = std::time::Instant::now();
        while let Some(s) = rec.next_sample(&mut cur).unwrap() {
            assert!(s.time_s() >= last);
            last = s.time_s();
            match s {
                Sample::Imu(_) => imu += 1,
                Sample::Image(_) => img += 1,
                Sample::Baro(b) => {
                    assert!((50_000.0..110_000.0).contains(&b.pressure_pa));
                    baro += 1;
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        println!(
            "10 s merged pull: {imu} imu, {img} images, {baro} baro in {:?}",
            start.elapsed()
        );
        assert!((9_000..=11_000).contains(&imu));
        assert!((190..=210).contains(&img));
        assert!(baro > 400);
    }

    #[test]
    fn all_imu_samples_pull_quickly() {
        let Some(mut rec) = recording() else { return };
        let mut cur = rec.cursor(&[AriaSensor::ImuRight]);
        let n = rec.sample_count(AriaSensor::ImuRight);
        let start = std::time::Instant::now();
        let mut got = 0usize;
        while let Some(Sample::Imu(_)) = rec.next_sample(&mut cur).unwrap() {
            got += 1;
        }
        let dt = start.elapsed();
        println!(
            "pulled {got} IMU samples in {dt:?} ({:.0}k samples/s)",
            got as f64 / dt.as_secs_f64() / 1e3
        );
        assert_eq!(got, n);
        let _: Option<VrsReader> = None;
    }
}
