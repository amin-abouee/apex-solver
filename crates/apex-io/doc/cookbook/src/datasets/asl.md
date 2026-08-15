# ASL / EuRoC

The **ASL** format is the MAV0 sensor layout used by the EuRoC MAV benchmark:
one or more cameras, an IMU, and (optionally) ground-truth poses, each described
by a `data.csv` alongside its raw data. `apex-io` offers two access styles:

- [`AslReader`](#aslreader-eager) — eager: load the whole dataset into memory.
- [`AslStream`](#aslstream-streaming) — streaming: pull images and IMU samples on
  demand with independent cursors.

Both return `Result<_, AslError>` (aliased as `asl::Result<T>`).

## Directory layout

```text
mav0/
├── cam0/  data.csv  data/*.png
├── cam1/  data.csv  data/*.png
├── imu0/  data.csv
└── state_groundtruth_estimate0/  data.csv
```

## Data types

| Type | Fields |
|---|---|
| `ImuMeasurement` | `timestamp_ns: u64`, `angular_velocity: Vector3`, `linear_acceleration: Vector3` |
| `CameraFrame` | `timestamp_ns: u64`, `image_path: PathBuf` |
| `GroundTruthPose` | `timestamp_ns: u64`, `position: Vector3`, `orientation: UnitQuaternion` |
| `CameraData` | `index: usize`, `frames: Vec<CameraFrame>`, `data_dir: PathBuf` |
| `AslDataset` | `cameras: Vec<CameraData>`, `imu_measurements: Vec<ImuMeasurement>`, `ground_truth: Option<Vec<GroundTruthPose>>`, `base_path: PathBuf` |

<a id="aslreader-eager"></a>
## `AslReader` (eager)

```rust
pub struct AslReader;
```

| Method | Signature | Description |
|---|---|---|
| `load` | `fn load<P: AsRef<Path>>(mav0_path: P) -> Result<AslDataset>` | Parse every sensor CSV into one `AslDataset`. |
| `camera_count` | `fn camera_count(&self) -> usize` | Number of cameras. |
| `imu_sample_count` | `fn imu_sample_count(&self) -> usize` | Number of IMU rows. |
| `has_ground_truth` | `fn has_ground_truth(&self) -> bool` | Whether ground truth is present. |
| `load_image` | `fn load_image(&self, cam_idx, frame_idx) -> Result<image::DynamicImage>` | Decode a single frame. |

<a id="aslstream-streaming"></a>
## `AslStream` (streaming)

For long sequences you rarely want every image in memory at once. `AslStream`
keeps independent image and IMU cursors:

```rust
pub struct AslStream { /* … */ }
```

| Method | Signature | Description |
|---|---|---|
| `open` | `fn open<P: AsRef<Path>>(mav0_path: P) -> Result<Self>` | Open the dataset (cam0 by default). |
| `open_camera` | `fn open_camera<P: AsRef<Path>>(mav0_path: P, cam_idx: usize) -> Result<Self>` | Open a specific camera. |
| `image_count` / `imu_count` | `-> usize` | Totals for each stream. |
| `next_image` | `-> Option<Result<(u64, DynamicImage)>>` | Next `(timestamp_ns, image)`. |
| `next_imu` | `-> Option<ImuMeasurement>` | Next IMU sample. |
| `next_n_images` | `(n) -> Result<Vec<(u64, DynamicImage)>>` | Batch of images. |
| `next_n_imu` | `(n) -> Vec<ImuMeasurement>` | Batch of IMU samples. |
| `reset_images` / `reset_imu` | `(&mut self)` | Rewind a cursor to the start. |

## Errors

`AslError` covers directory-structure problems (`InvalidMav0Directory`,
`NoSensorsFound`, `CameraNotFound`, `FrameOutOfBounds`), CSV problems
(`InvalidCsvFormat`, `InvalidNumber`, `MissingCsvColumns`, `InvalidQuaternion`,
each with `path:line` context), and image decoding (`ImageLoad`).

## Example

```rust
use apex_io::{AslReader, AslStream};

// Eager: metadata + counts.
let ds = AslReader::load("data/euroc/MH_01/mav0")?;
println!("{} cameras, {} IMU samples", ds.cameras.len(), ds.imu_measurements.len());

// Streaming: process frames one at a time.
let mut stream = AslStream::open("data/euroc/MH_01/mav0")?;
while let Some(frame) = stream.next_image() {
    let (ts_ns, img) = frame?;
    // … feed `img` to a front-end …
    let _ = (ts_ns, img);
}
# Ok::<(), apex_io::AslError>(())
```

## References

- Burri, M. et al. (2016). *The EuRoC micro aerial vehicle datasets*. IJRR 35(10), 1157–1163.
