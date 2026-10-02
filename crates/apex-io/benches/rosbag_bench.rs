#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Criterion benchmarks for the ROS1/ROS2 bag stack.
//!
//! Groups (all over the same synthetic VIO-like recording — 2 x 10 Hz
//! 320x240 cameras, 200 Hz IMU, 5 Hz 10k-point lidar, 20 s, ~48 MB):
//!
//! - `rosbag_info`     — open + summary (no data read)
//! - `rosbag_read`     — full / two-topic / two-second-window scans
//! - `rosbag_write`    — create + write + close per container
//! - `rosbag_convert`  — full container + payload-dialect conversions
//! - `rosbag_transcode`— per-message ROS1 <-> CDR conversion
//! - `real_mcap`       — the same operations on a real MCAP when
//!   `APEX_BENCH_MCAP` points at one (e.g.
//!   /Volumes/External/datasets/uzh_fpv/mcap/indoor_forward_3_snapdragon_with_gt.mcap)
//!
//! ```bash
//! cargo bench -p apex-io --features rosbag --bench rosbag_bench
//! ```

mod helpers;

use apex_io::rosbag::bag::{BagKind, BagReader, OutputCompression, Selection};
use apex_io::rosbag::transcode::{Dialect, Transcoder, dialect_of, plan_connections};
use apex_io::rosbag::types::Connection;
use apex_io::rosbag::{BagWriter, NewConnection};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::collections::HashMap;
use std::hint::black_box;
use std::path::{Path, PathBuf};

struct Fixtures {
    ros1: PathBuf,
    sqlite: PathBuf,
    mcap: PathBuf,
    scenario: helpers::Scenario,
}

fn fixtures() -> &'static Fixtures {
    static F: std::sync::OnceLock<Fixtures> = std::sync::OnceLock::new();
    F.get_or_init(|| {
        helpers::reset_bench_dir();
        let root = helpers::bench_root();
        let scenario = helpers::Scenario::build();
        let ros1 = root.join("synth.bag");
        let sqlite = root.join("sqlite");
        let mcap = root.join("synth.mcap");
        helpers::write_bag(&scenario, BagKind::Ros1Bag, &ros1);
        helpers::write_bag(&scenario, BagKind::Ros2Sqlite, &sqlite);
        helpers::write_bag(&scenario, BagKind::McapFile, &mcap);
        Fixtures {
            ros1,
            sqlite,
            mcap,
            scenario,
        }
    })
}

fn bench_open_info(c: &mut Criterion) {
    let f = fixtures();
    let mut g = c.benchmark_group("rosbag_info");
    for (name, path) in [("ros1", &f.ros1), ("sqlite3", &f.sqlite), ("mcap", &f.mcap)] {
        g.bench_function(name, |b| {
            b.iter(|| {
                let r = BagReader::open(black_box(path)).unwrap();
                black_box((
                    r.message_count(),
                    r.start_time(),
                    r.end_time(),
                    r.connections().len(),
                ));
            })
        });
    }
    g.finish();
}

fn scan(path: &Path, selection: &Selection) -> apex_io::rosbag::Result<u64> {
    let mut r = BagReader::open(path)?;
    let conns = r.selected_connections(selection);
    let mut bytes = 0u64;
    r.for_each(&conns, selection, |_, _, data| {
        bytes += data.len() as u64;
        Ok(())
    })?;
    Ok(bytes)
}

fn bench_read(c: &mut Criterion) {
    let f = fixtures();
    let vio = Selection {
        include: helpers::Scenario::vio_topics()
            .iter()
            .map(|s| s.to_string())
            .collect(),
        ..Selection::default()
    };
    let (start, end) = helpers::Scenario::window();
    let window = Selection {
        start_ns: Some(start),
        end_ns: Some(end),
        ..Selection::default()
    };
    let mut g = c.benchmark_group("rosbag_read");
    g.sample_size(20);
    for (name, path) in [("ros1", &f.ros1), ("sqlite3", &f.sqlite), ("mcap", &f.mcap)] {
        g.bench_function(format!("full/{name}"), |b| {
            b.iter(|| black_box(scan(black_box(path), &Selection::default()).unwrap()))
        });
        g.bench_function(format!("two_topics/{name}"), |b| {
            b.iter(|| black_box(scan(black_box(path), &vio).unwrap()))
        });
        g.bench_function(format!("two_second_window/{name}"), |b| {
            b.iter(|| black_box(scan(black_box(path), &window).unwrap()))
        });
    }
    g.finish();
}

fn bench_write(c: &mut Criterion) {
    let f = fixtures();
    let mut g = c.benchmark_group("rosbag_write");
    g.sample_size(10);
    g.measurement_time(std::time::Duration::from_secs(10));
    let cases = [
        ("ros1", BagKind::Ros1Bag, "out.bag"),
        ("sqlite3", BagKind::Ros2Sqlite, "out"),
        ("mcap", BagKind::McapFile, "out.mcap"),
    ];
    for (name, kind, file) in cases {
        g.bench_function(name, |b| {
            b.iter_batched(
                || helpers::scratch_dir("bench"),
                |s| {
                    let path = s.0.join(file);
                    helpers::write_bag(black_box(&f.scenario), kind, &path);
                    black_box(std::fs::metadata(&path).unwrap().len());
                },
                BatchSize::PerIteration,
            )
        });
    }
    g.bench_function("mcap_zstd", |b| {
        b.iter_batched(
            || helpers::scratch_dir("bench"),
            |s| {
                let path = s.0.join("out.mcap");
                write_bag_zstd(black_box(&f.scenario), &path);
                black_box(std::fs::metadata(&path).unwrap().len());
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}

fn write_bag_zstd(scenario: &helpers::Scenario, path: &Path) {
    let mut w = BagWriter::create(path, BagKind::McapFile, OutputCompression::Zstd).unwrap();
    let handles: Vec<_> = scenario
        .topics
        .iter()
        .map(|t| {
            w.add_connection(&NewConnection {
                topic: t.topic.to_string(),
                message_type: helpers::ros2_type(t.ros1_type),
                definition: apex_io::rosbag::transcode::catalog_definition(
                    t.ros1_type,
                    Dialect::Ros2,
                )
                .unwrap_or_default(),
                type_hash: String::new(),
                serialization_format: "cdr".to_string(),
                qos: Vec::new(),
            })
            .unwrap()
        })
        .collect();
    for m in &scenario.messages {
        w.write(handles[m.topic_idx], m.timestamp_ns, &m.cdr)
            .unwrap();
    }
    w.close().unwrap();
}

enum Action {
    Copy(BagWriterHandle),
    Convert(BagWriterHandle, Box<Transcoder>),
}
use apex_io::rosbag::ConnectionHandle as BagWriterHandle;

/// Full conversion of `input` to `kind`, mirroring the bag_convert CLI loop.
fn convert_once(input: &Path, kind: BagKind, out: &Path) -> u64 {
    let mut reader = BagReader::open(input).unwrap();
    let selection = Selection::default();
    let selected = reader.selected_connections(&selection);
    let to = if kind == BagKind::Ros1Bag {
        Dialect::Ros1
    } else {
        Dialect::Ros2
    };
    let (same, different): (Vec<Connection>, Vec<Connection>) = selected
        .iter()
        .cloned()
        .partition(|c| dialect_of(&c.serialization_format) == Some(to));
    let (planned, _unsupported) = plan_connections(&different, to);

    let mut writer = BagWriter::create(out, kind, OutputCompression::None).unwrap();
    let mut actions: HashMap<u32, Action> = HashMap::new();
    let mut readable = Vec::new();
    for c in &same {
        let h = writer
            .add_connection(&NewConnection::from_connection(c))
            .unwrap();
        actions.insert(c.id, Action::Copy(h));
        readable.push(c.clone());
    }
    for (c, plan) in planned {
        let h = writer.add_connection(&plan.target).unwrap();
        actions.insert(c.id, Action::Convert(h, Box::new(plan.transcoder)));
        readable.push(c);
    }

    let mut count = 0u64;
    let mut scratch = Vec::new();
    reader
        .for_each(&readable, &selection, |c, ts, data| {
            match actions.get(&c.id).expect("action") {
                Action::Copy(h) => writer.write(*h, ts, data)?,
                Action::Convert(h, t) => {
                    t.convert(data, &mut scratch)?;
                    writer.write(*h, ts, &scratch)?;
                }
            }
            count += 1;
            Ok(())
        })
        .unwrap();
    writer.close().unwrap();
    count
}

fn bench_convert(c: &mut Criterion) {
    let f = fixtures();
    let mut g = c.benchmark_group("rosbag_convert");
    g.sample_size(10);
    g.measurement_time(std::time::Duration::from_secs(10));
    let cases = [
        ("ros1_to_mcap", &f.ros1, BagKind::McapFile, "out.mcap"),
        ("mcap_to_ros1", &f.mcap, BagKind::Ros1Bag, "out.bag"),
        ("mcap_to_sqlite", &f.mcap, BagKind::Ros2Sqlite, "out"),
    ];
    for (name, input, kind, file) in cases {
        g.bench_function(name, |b| {
            b.iter_batched(
                || helpers::scratch_dir("bench"),
                |s| {
                    let out = s.0.join(file);
                    let n = convert_once(black_box(input), kind, &out);
                    black_box(n);
                },
                BatchSize::PerIteration,
            )
        });
    }
    g.finish();
}

fn bench_transcode(c: &mut Criterion) {
    let f = fixtures();
    let reg = apex_io::rosbag::transcode::catalog().unwrap();
    let mut g = c.benchmark_group("rosbag_transcode");

    let cases = [
        ("imu_ros1_to_cdr", "/imu0", Dialect::Ros1, Dialect::Ros2),
        (
            "image_ros1_to_cdr",
            "/cam0/image_raw",
            Dialect::Ros1,
            Dialect::Ros2,
        ),
        (
            "pointcloud2_ros1_to_cdr",
            "/lidar/points",
            Dialect::Ros1,
            Dialect::Ros2,
        ),
        (
            "image_cdr_to_ros1",
            "/cam0/image_raw",
            Dialect::Ros2,
            Dialect::Ros1,
        ),
    ];
    for (name, topic, from, to) in cases {
        let spec = f
            .scenario
            .topics
            .iter()
            .find(|t| t.topic == topic)
            .expect("topic");
        let ty = spec.ros1_type;
        let t = Transcoder::compile(reg, ty, from, to).unwrap();
        let payloads: Vec<&Vec<u8>> = f
            .scenario
            .messages
            .iter()
            .filter(|m| f.scenario.topics[m.topic_idx].topic == topic)
            .take(10)
            .map(|m| {
                if from == Dialect::Ros1 {
                    &m.ros1
                } else {
                    &m.cdr
                }
            })
            .collect();
        let mut out = Vec::new();
        g.bench_function(name, |b| {
            b.iter(|| {
                for p in &payloads {
                    t.convert(black_box(p), &mut out).unwrap();
                    black_box(&out);
                }
            })
        });
    }
    g.finish();
}

fn bench_real_mcap(c: &mut Criterion) {
    let Some(path) = helpers::env_path("APEX_BENCH_MCAP") else {
        return;
    };
    let topics: Vec<String> = {
        let r = BagReader::open(&path).unwrap();
        r.connections()
            .iter()
            .take(2)
            .map(|c| c.topic.clone())
            .collect()
    };
    let two_topics = Selection {
        include: topics,
        ..Selection::default()
    };
    let mut g = c.benchmark_group("real_mcap");
    g.sample_size(10);
    g.bench_function("open_info", |b| {
        b.iter(|| {
            let r = BagReader::open(black_box(&path)).unwrap();
            black_box((r.message_count(), r.connections().len()));
        })
    });
    g.bench_function("read_full", |b| {
        b.iter(|| black_box(scan(black_box(&path), &Selection::default()).unwrap()))
    });
    g.bench_function("read_two_topics", |b| {
        b.iter(|| black_box(scan(black_box(&path), &two_topics).unwrap()))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_open_info,
    bench_read,
    bench_write,
    bench_convert,
    bench_transcode,
    bench_real_mcap
);
criterion_main!(benches);
