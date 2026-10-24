# VRS and LaMAria

[LaMAria](https://lamaria.ethz.ch) records with Project Aria glasses, whose
recordings are **VRS** files. A sequence is three parts, each with its own reader:

| Part | File | Reader |
|---|---|---|
| Sensor data | `raw_data/<seq>.vrs` | [`vrs::aria::AriaRecording`](#reading-a-recording) (feature `vrs`) |
| Calibration | `aria_calibrations/<seq>.json` | [`lamaria::Calibration`](#calibration) |
| Ground truth | `ground_truth/pGT/<seq>.txt` | [`TumNanosLoader`](#ground-truth) |

Ground truth ships for `training/` sequences only.

## Reading a recording

```rust
use apex_io::vrs::aria::{AriaRecording, AriaSensor, Sample};

let mut rec = AriaRecording::open("R_06_medium.vrs")?;

// Samples one at a time, merged in capture-time order across sensors.
let mut cursor = rec.cursor(&[AriaSensor::SlamLeft, AriaSensor::ImuRight]);
while let Some(sample) = rec.next_sample(&mut cursor)? {
    match sample {
        Sample::Imu(imu) => { /* imu.time_s, imu.accel, imu.gyro */ }
        Sample::Image(img) => { /* img.jpeg: the original bytes, undecoded */ }
        _ => {}
    }
}

// Or by time.
let (t0, _) = rec.time_range(AriaSensor::ImuRight).ok_or("no IMU")?;
let imu = rec.imu_between(AriaSensor::ImuRight, t0 + 10.0, t0 + 10.5)?;
```

Opening reads the header, tags and index only; a 2.3 GB recording opens in
milliseconds. Timestamps are the **device clock** in seconds, the same clock the
ground truth uses.

`vrs_info --verify-index` checks the index against a full scan and the record
chain; `R_06_medium` (749,793 records, 12 streams) passes both.

## Writing

`VrsWriter` writes streams, tags and per-record zstd or lz4, and builds the
index when it closes. `vrs_copy` trims by stream and time and can recompress; a
copy is byte-identical per record. As a bag container, VRS stores every ROS
topic losslessly, so any bag survives ROS → VRS → ROS byte for byte
(`bag_convert in.bag out.vrs`).

## Calibration

```rust
use apex_io::lamaria::Calibration;

let calibration = Calibration::load("aria_calibrations/R_06_medium.json")?;
let t_body_cam0 = calibration.cam0.t_body_sensor();   // SE3
let intrinsics = &calibration.cam0.params;             // model's own order
let sigma_a = calibration.imu.acc_noise_density;
```

`T_b_s` is the sensor's pose in the body frame; the file stores its quaternion
as `x y z w`. The cameras are `RAD_TAN_THIN_PRISM_FISHEYE` with sixteen
parameters, `fx fy cx cy` then twelve distortion terms. All 86 files in the
dataset parse.

> The same data is in each recording's VRS tags as Meta's factory calibration,
> in a different schema: `AriaRecording::calibration_json`.

## Ground truth

```rust
use apex_io::{TrajectoryLoader, TumNanosLoader};

let truth = TumNanosLoader::load("ground_truth/pGT/R_06_medium.txt")?;
let pose = truth.pose_at(timestamp_ns);
```

The rows are TUM's (`tx ty tz qx qy qz qw`, w last) with the timestamp an
**integer count of nanoseconds** instead of decimal seconds. `TumLoader` reads
such a bare integer as seconds and overflows; use `TumNanosLoader`, or
`load_trajectory_as(path, TrajectoryFormat::TumNanos)`.

## Checking against the dataset

```bash
APEX_LAMARIA_ROOT=/data/lamaria \
  cargo test -p apex-io --features vrs --release --test lamaria_real_files
```

parses every calibration file and loads every ground truth, and asserts that
each ground truth lies inside its recording's time range.
