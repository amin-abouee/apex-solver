# Trajectories

The `trajectory` module reads and writes timestamped pose trajectories: TUM and the ASL pose CSV.
It is one implementation shared by every consumer of the crate, replacing per-project readers
that had drifted apart.

| Loader | Format | Quaternion | Separator |
|---|---|---|---|
| `TumLoader` | `timestamp tx ty tz qx qy qz qw` | **w last** | whitespace |
| `TumNanosLoader` | the same, with `timestamp` an integer of nanoseconds (LaMAria `pGT`) | **w last** | whitespace |
| `AslTrajectoryLoader` | EuRoC 17-column or `mocap0` 8-column CSV | **w first** | `,` |

```rust
use apex_io::trajectory::{load_trajectory, load_trajectory_as, TrajectoryFormat};

let truth = load_trajectory("groundtruth.tum")?;                   // by extension: .tum/.txt, .csv
let nanos = load_trajectory_as("pGT/seq.txt", TrajectoryFormat::TumNanos)?; // stated, extension ignored
```

`load_trajectory` dispatches on the extension (`.tum` and `.txt` are TUM, `.csv` is ASL) and fails
loudly on a mismatch rather than misparsing; any other extension is
`TrajectoryError::UnsupportedFormat`.

## The model

A `Trajectory` is a time-sorted list of `TrajectoryPose { timestamp_ns, position, orientation }`
(`position` a `Vector3`, `orientation` an `SO3` from apex-manifolds), optionally with the
`InertialState` columns EuRoC ships (velocity, gyroscope and accelerometer bias). Both formats
store a body pose in a world frame, `T_WB`: neither file says which body or world frame, which is
the dataset's business. Poses are **sorted by timestamp on construction**, because a
concatenated or re-exported log can arrive out of order and every query assumes ascending time;
`Trajectory::from_poses_and_inertial` applies the *same* permutation to the inertial columns.

Queries: `pose_at(ns)`, `pose_at_seconds`, `position_at`, `se3_at`, `inertial_at`, `time_span_ns`,
`time_span_seconds`, `timestamps_seconds`, `se3_poses`; constructors from `SE3` and seconds or
nanoseconds (`from_se3_seconds`, `from_se3_nanos`).

## Timestamps are `u64` nanoseconds

At epoch magnitudes (~1.4e9 s) an `f64` second has an ulp of about **476 ns**, so a trajectory
routed through `f64` seconds does not round-trip. The model therefore stores nanoseconds, and the
TUM codec formats and parses its timestamps as integers; only a foreign file in scientific
notation falls back to `f64`, which is lossy. `timestamp_seconds()` exists for arithmetic against
other `f64` times, never as a key or for equality.

## Quaternions from text

A quaternion read from a text file must be within `1e-2` of unit norm and is then normalized:
real ground truth is not exactly unit (EuRoC's `MH_03` is off by up to `3.3e-4`), while a scaled,
truncated or garbage row is off by far more and is rejected (`TrajectoryError::InvalidQuaternion`)
rather than repaired. A permuted unit quaternion is still unit, so no norm check can catch a wrong
column order: mind the w-first against w-last difference in the table.

## Writing

`TumLoader::render(&trajectory)` and `write` produce TUM text with nine decimals on the pose
fields (a nanometre and a nanoradian: below any sensor, and the width TUM's own tools emit),
and the timestamp as `seconds.nanoseconds`, exact. Inertial columns are dropped, as TUM has no
place for them. The format is what `evo` and the TUM scripts read.

Evaluating a trajectory against ground truth is the job of `apex-evaluation`, whose `read_tum`
parses through this module and converts with `TryFrom<&apex_io::trajectory::Trajectory>`.
