//! Criterion benchmarks for streaming an MCAP recording:
//!
//! - `mcap_stream/all_topics`: every message, parallel chunk decompression
//! - `mcap_stream/imu_and_cloud`: a topic filter, the shape of a LiDAR-inertial
//!   run — the camera payloads are skipped before they are copied
//! - `mcap_stream/open_topics`: topics and counts from the summary section
//!
//! ```bash
//! cargo bench -p apex-io --bench mcap_stream
//! ```
//!
//! The recording is synthetic and written once into a temporary directory:
//! zstd chunks of about 1 MiB carrying a small IMU topic and two large
//! payload topics, so the cost measured is the container's, not a dataset's.

use std::collections::HashSet;
use std::fs::File;
use std::hint::black_box;
use std::io::BufWriter;
use std::path::Path;

use apex_io::rosbag::ros2::storage::mcap::{McapMessageStream, McapStorageReader, StreamFilter};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

type SetupResult<T> = Result<T, Box<dyn std::error::Error>>;

const IMU_MESSAGES: u64 = 4_000;
const CLOUD_BYTES: usize = 160 * 1024;
const CAMERA_BYTES: usize = 360 * 1024;
const FRAMES: u64 = 100;

fn write_recording(path: &Path) -> SetupResult<u64> {
    let mut writer = mcap::WriteOptions::new()
        .compression(Some(mcap::Compression::Zstd))
        .chunk_size(Some(1 << 20))
        .emit_statistics(true)
        .profile("ros2")
        .create(BufWriter::new(File::create(path)?))?;
    let schema = writer.add_schema("pkg/msg/Sample", "ros2msg", b"int32 value")?;
    let imu = writer.add_channel(schema, "/imu", "cdr", &Default::default())?;
    let cloud = writer.add_channel(schema, "/cloud", "cdr", &Default::default())?;
    let camera = writer.add_channel(schema, "/camera", "cdr", &Default::default())?;

    let mut bytes = 0u64;
    let mut emit = |channel: u16, time: u64, size: usize, seed: u64| -> SetupResult<()> {
        // Structured rather than random, so zstd has something to do.
        let data: Vec<u8> = (0..size)
            .map(|i| ((i as u64 / 3 + seed) % 61) as u8)
            .collect();
        let header = mcap::records::MessageHeader {
            channel_id: channel,
            sequence: 0,
            log_time: time,
            publish_time: time,
        };
        writer.write_to_known_channel(&header, &data)?;
        bytes += size as u64;
        Ok(())
    };
    for frame in 0..FRAMES {
        let t = frame * 100_000_000;
        emit(cloud, t, CLOUD_BYTES, frame)?;
        emit(camera, t + 1, CAMERA_BYTES, frame + 7)?;
        for k in 0..IMU_MESSAGES / FRAMES {
            emit(imu, t + 2 + k, 64, k)?;
        }
    }
    writer.finish()?;
    Ok(bytes)
}

fn bench(c: &mut Criterion) -> SetupResult<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("synthetic.mcap");
    let payload_bytes = write_recording(&path)?;

    let mut group = c.benchmark_group("mcap_stream");
    group.sample_size(15);
    group.throughput(Throughput::Bytes(payload_bytes));
    group.bench_function("all_topics", |b| {
        b.iter(|| {
            let mut total = 0usize;
            if let Ok(stream) = McapMessageStream::open(&path) {
                for message in stream.flatten() {
                    total += message.data.len();
                }
            }
            black_box(total)
        });
    });
    let wanted: HashSet<String> = ["/imu", "/cloud"]
        .iter()
        .map(|t| (*t).to_string())
        .collect();
    group.bench_function("imu_and_cloud", |b| {
        b.iter(|| {
            let filter = StreamFilter {
                topics: Some(wanted.clone()),
                ..StreamFilter::default()
            };
            let mut total = 0usize;
            if let Ok(stream) = McapMessageStream::open_filtered(&path, filter) {
                for message in stream.flatten() {
                    total += message.data.len();
                }
            }
            black_box(total)
        });
    });
    group.finish();

    let mut group = c.benchmark_group("mcap_stream");
    group.throughput(Throughput::Elements(1));
    group.bench_function("open_topics", |b| {
        b.iter(|| {
            let counted = McapStorageReader::new(vec![path.as_path()], Vec::new())
                .and_then(|reader| reader.get_topics_from_mcap());
            black_box(counted.map(|topics| topics.len()).unwrap_or(0))
        });
    });
    group.finish();
    Ok(())
}

fn benches(c: &mut Criterion) {
    if let Err(error) = bench(c) {
        tracing::error!("mcap_stream: setup failed: {error}");
    }
}

criterion_group!(mcap_stream, benches);
criterion_main!(mcap_stream);
