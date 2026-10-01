#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Criterion benchmarks for the VRS container stack.
//!
//! Synthetic groups run over a generated recording (200 Hz IMU-like stream,
//! 10 Hz camera-like stream with 30 KB incompressible frames, 20 s, ~13 MB).
//! The `real_vrs` group runs the identical operations on a real Aria file
//! when `APEX_BENCH_VRS` points at one (e.g. a LaMaria `.vrs`); without it
//! the group is skipped.
//!
//! ```bash
//! cargo bench -p apex-io --features vrs --bench vrs_bench
//! APEX_BENCH_VRS=/Volumes/External/datasets/lamaria/test/sequence_1_18/raw_data/sequence_1_18.vrs \
//!     cargo bench -p apex-io --features vrs --bench vrs_bench -- real_vrs
//! ```

mod helpers;

use apex_io::vrs::aria::{AriaRecording, AriaSensor, Sample};
use apex_io::vrs::{CopyOptions, RecordFilter, VrsReader, copy_filtered};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

struct Fixtures {
    vrs: helpers::VrsFixture,
}

fn fixtures() -> &'static Fixtures {
    static F: std::sync::OnceLock<Fixtures> = std::sync::OnceLock::new();
    F.get_or_init(|| {
        helpers::reset_bench_dir();
        let vrs = helpers::write_vrs(&helpers::bench_root());
        Fixtures { vrs }
    })
}

fn bench_open(c: &mut Criterion) {
    let f = fixtures();
    let mut g = c.benchmark_group("vrs_open");
    g.bench_function("synthetic", |b| {
        b.iter(|| {
            let r = VrsReader::open(black_box(&f.vrs.path)).unwrap();
            black_box((r.records().len(), r.streams().len()));
        })
    });
    g.finish();
}

fn bench_read(c: &mut Criterion) {
    let f = fixtures();
    let mut g = c.benchmark_group("vrs_read");
    g.sample_size(20);
    let both = RecordFilter::all();
    let imu_only = RecordFilter::streams([f.vrs.imu_stream]);
    let images_only = RecordFilter::streams([f.vrs.image_stream]);
    for (name, filter) in [("full", &both), ("imu_only", &imu_only), ("images_only", &images_only)] {
        g.bench_function(name, |b| {
            b.iter(|| {
                let mut r = VrsReader::open(&f.vrs.path).unwrap();
                let mut cur = r.cursor(black_box(filter));
                let mut buf = Vec::new();
                let mut bytes = 0u64;
                while r
                    .next_record_into(&mut cur, &mut buf)
                    .unwrap()
                    .is_some()
                {
                    bytes += buf.len() as u64;
                }
                black_box(bytes);
            })
        });
    }
    g.finish();
}

fn bench_write(c: &mut Criterion) {
    let _f = fixtures();
    let mut g = c.benchmark_group("vrs_write");
    g.sample_size(10);
    g.bench_function("synthetic", |b| {
        b.iter_batched(
            || helpers::scratch_dir("bench"),
            |s| {
                let vrs = helpers::write_vrs(&s.0);
                black_box(std::fs::metadata(&vrs.path).unwrap().len());
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}

fn bench_copy(c: &mut Criterion) {
    let f = fixtures();
    let mut g = c.benchmark_group("vrs_copy");
    g.sample_size(10);
    g.bench_function("full", |b| {
        b.iter_batched(
            || helpers::scratch_dir("bench"),
            |s| {
                let mut src = VrsReader::open(&f.vrs.path).unwrap();
                let summary =
                    copy_filtered(black_box(&mut src), s.0.join("copy.vrs"), &CopyOptions::default())
                        .unwrap();
                black_box(summary.bytes);
            },
            BatchSize::PerIteration,
        )
    });
    g.bench_function("trim_two_seconds", |b| {
        b.iter_batched(
            || helpers::scratch_dir("bench"),
            |s| {
                let mut src = VrsReader::open(&f.vrs.path).unwrap();
                let opts = CopyOptions {
                    start: Some(5.0),
                    end: Some(7.0),
                    ..CopyOptions::default()
                };
                let summary = copy_filtered(black_box(&mut src), s.0.join("trim.vrs"), &opts).unwrap();
                black_box(summary.bytes);
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}

fn bench_real_vrs(c: &mut Criterion) {
    let Some(path) = helpers::env_path("APEX_BENCH_VRS") else {
        return;
    };
    // A 10-second window from the start of the IMU clock, so the heavy real
    // file is only partially walked.
    let probe = AriaRecording::open(&path).expect("open real vrs");
    let imus: Vec<AriaSensor> = probe
        .sensors()
        .into_iter()
        .filter(|s| matches!(s, AriaSensor::ImuRight | AriaSensor::ImuLeft))
        .collect();
    let window = imus
        .first()
        .and_then(|s| probe.time_range(*s))
        .map(|(t0, _)| (t0, t0 + 10.0))
        .expect("real file has an IMU stream");
    drop(probe);

    let mut g = c.benchmark_group("real_vrs");
    g.sample_size(10);
    g.bench_function("open", |b| {
        b.iter(|| {
            let r = AriaRecording::open(black_box(&path)).unwrap();
            black_box(r.sensors().len());
        })
    });
    g.bench_function("read_full_raw", |b| {
        b.iter(|| {
            let mut r = VrsReader::open(&path).unwrap();
            let mut cur = r.cursor(&RecordFilter::all());
            let mut buf = Vec::new();
            let mut n = 0u64;
            while r.next_record_into(&mut cur, &mut buf).unwrap().is_some() {
                n += 1;
            }
            black_box(n);
        })
    });
    g.bench_function("decode_window", |b| {
        b.iter(|| {
            let mut rec = AriaRecording::open(&path).unwrap();
            let mut cur = rec.cursor_between(&[], black_box(window.0), black_box(window.1));
            let mut n = 0u64;
            while let Some(s) = rec.next_sample(&mut cur).unwrap() {
                black_box(&s);
                n += 1;
            }
            black_box(n);
        })
    });
    g.bench_function("decode_imu_window", |b| {
        b.iter(|| {
            let mut rec = AriaRecording::open(&path).unwrap();
            let mut cur =
                rec.cursor_between(&imus, black_box(window.0), black_box(window.1));
            let mut n = 0u64;
            while let Some(Sample::Imu(s)) = rec.next_sample(&mut cur).unwrap() {
                black_box(s.capture_ns);
                n += 1;
            }
            black_box(n);
        })
    });
    g.bench_function("imu_between_one_second", |b| {
        b.iter(|| {
            let mut rec = AriaRecording::open(&path).unwrap();
            let (s, _) = imus.first().copied().map(|s| (s, ())).expect("imu");
            let v = rec
                .imu_between(s, black_box(window.0) + 5.0, black_box(window.0) + 6.0)
                .unwrap();
            black_box(v.len());
        })
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_open,
    bench_read,
    bench_write,
    bench_copy,
    bench_real_vrs
);
criterion_main!(benches);
