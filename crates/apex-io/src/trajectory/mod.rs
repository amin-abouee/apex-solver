//! Trajectory reading and writing: TUM and the ASL pose CSV (via [`crate::asl`]).
//!
//! | Loader | Format | Quaternion | Separator |
//! |---|---|---|---|
//! | [`TumLoader`] | `timestamp tx ty tz qx qy qz qw` | **w last** | whitespace |
//! | [`crate::asl::AslTrajectoryLoader`] | EuRoC 17-column or `mocap0` 8-column CSV | **w first** | `,` |
//!
//! One implementation shared by every consumer of this crate, replacing the
//! per-project trajectory readers that had drifted apart.
//!
//! # Timestamps are `u64` nanoseconds
//!
//! Matching the ASL dataset reader, and matching what both ASL layouts store
//! on disk.
//! `f64` seconds is a *presentation* format belonging to the TUM codec, which
//! is the only place in this module that converts. That matters: near epoch
//! magnitudes (~1.4e9 s) an `f64` second has an ulp of about **476 ns**, so a
//! trajectory routed through `f64` seconds does not round-trip. The TUM codec
//! therefore formats and parses its timestamps as integers, and only falls back
//! to `f64` for foreign files in scientific notation.
//!
//! # Frames
//!
//! Both formats store a body pose in a world frame, `T_WB` — position and
//! orientation *of* the body *in* the world. Neither file says which body frame
//! or which world; that is the dataset's business, not this module's.

pub mod error;
pub mod tum;
pub mod types;

use std::path::Path;

use apex_manifolds::se3::SE3;
use nalgebra::Vector3;

pub use error::{Result, TrajectoryError};
pub use tum::TumLoader;
pub use types::{InertialState, TrajectoryFormat, TrajectoryPose};

use types::NANOS_PER_SECOND;

/// Trait for trajectory file loaders and writers.
pub trait TrajectoryLoader {
    /// Load a trajectory from a file.
    ///
    /// # Errors
    ///
    /// See [`TrajectoryError`].
    fn load<P: AsRef<Path>>(path: P) -> Result<Trajectory>;

    /// Write a trajectory to a file.
    ///
    /// # Errors
    ///
    /// See [`TrajectoryError`].
    fn write<P: AsRef<Path>>(trajectory: &Trajectory, path: P) -> Result<()>;
}

/// A time-ordered trajectory.
///
/// # Why the inertial columns are one `Option` for the whole trajectory
///
/// A file either carries velocity and biases for every row (EuRoC's 17 columns)
/// or for none (TUM, TUM VI `mocap0`). Optionality is a property of the
/// *format*, not of any individual sample. An `Option` per sample would make
/// every pose-only consumer — ATE, RPE, plotting, the majority — pay for an
/// unwrap on each access, and would make "velocity present, gyro bias missing"
/// representable when no format can produce it.
///
/// So the `Option` sits here, once. [`Self::poses`] is infallible;
/// [`Self::inertial`] is the single place a consumer needing the richer data
/// asks for it.
///
/// # Invariant
///
/// When `inertial` is `Some`, its length equals `poses.len()` and its entries
/// correspond index-for-index. Private fields plus checked constructors are
/// what enforce that — do not add a `pub` field.
#[derive(Debug, Clone, Default)]
pub struct Trajectory {
    poses: Vec<TrajectoryPose>,
    inertial: Option<Vec<InertialState>>,
}

impl Trajectory {
    /// Build a pose-only trajectory, sorting by timestamp.
    ///
    /// File order is not trusted: a concatenated or re-exported log can arrive
    /// out of order, and every query here assumes ascending time.
    pub fn from_poses(mut poses: Vec<TrajectoryPose>) -> Self {
        poses.sort_by_key(|p| p.timestamp_ns);
        Self {
            poses,
            inertial: None,
        }
    }

    /// Build a trajectory carrying the inertial columns.
    ///
    /// Both vectors are reordered by the *same* permutation. Sorting the poses
    /// alone would pair every pose with another sample's velocity — a bug no
    /// round-trip test on already-sorted input could catch.
    ///
    /// # Errors
    ///
    /// [`TrajectoryError::LengthMismatch`] if the two disagree in length.
    pub fn from_poses_and_inertial(
        poses: Vec<TrajectoryPose>,
        inertial: Vec<InertialState>,
    ) -> Result<Self> {
        if poses.len() != inertial.len() {
            return Err(TrajectoryError::LengthMismatch {
                what: "trajectory poses vs inertial states",
                got: inertial.len(),
                expected: poses.len(),
            });
        }

        let mut order: Vec<usize> = (0..poses.len()).collect();
        order.sort_by_key(|&i| poses[i].timestamp_ns);

        let sorted_poses = order.iter().map(|&i| poses[i].clone()).collect();
        let sorted_inertial = order.iter().map(|&i| inertial[i].clone()).collect();

        Ok(Self {
            poses: sorted_poses,
            inertial: Some(sorted_inertial),
        })
    }

    /// Build from `f64` seconds and [`SE3`] poses.
    ///
    /// This is the whole seconds→nanoseconds boundary for producers.
    ///
    /// # Errors
    ///
    /// [`TrajectoryError::LengthMismatch`] if the slices differ in length;
    /// [`TrajectoryError::NegativeTimestamp`] for a negative or non-finite time.
    pub fn from_se3_seconds(timestamps_s: &[f64], poses: &[SE3]) -> Result<Self> {
        if timestamps_s.len() != poses.len() {
            return Err(TrajectoryError::LengthMismatch {
                what: "timestamps vs poses",
                got: poses.len(),
                expected: timestamps_s.len(),
            });
        }

        let mut samples = Vec::with_capacity(poses.len());
        for (index, (seconds, pose)) in timestamps_s.iter().zip(poses.iter()).enumerate() {
            if !seconds.is_finite() || *seconds < 0.0 {
                return Err(TrajectoryError::NegativeTimestamp {
                    path: Path::new("<memory>").to_path_buf(),
                    line: index + 1,
                    value: *seconds,
                });
            }
            let nanos = (seconds * NANOS_PER_SECOND).round();
            samples.push(TrajectoryPose::from_se3(nanos as u64, pose));
        }
        Ok(Self::from_poses(samples))
    }

    /// Build from `u64` nanoseconds and [`SE3`] poses.
    ///
    /// # Errors
    ///
    /// [`TrajectoryError::LengthMismatch`] if the slices differ in length.
    pub fn from_se3_nanos(timestamps_ns: &[u64], poses: &[SE3]) -> Result<Self> {
        if timestamps_ns.len() != poses.len() {
            return Err(TrajectoryError::LengthMismatch {
                what: "timestamps vs poses",
                got: poses.len(),
                expected: timestamps_ns.len(),
            });
        }
        Ok(Self::from_poses(
            timestamps_ns
                .iter()
                .zip(poses.iter())
                .map(|(ns, pose)| TrajectoryPose::from_se3(*ns, pose))
                .collect(),
        ))
    }

    /// The pose samples, ascending in time.
    pub fn poses(&self) -> &[TrajectoryPose] {
        &self.poses
    }

    /// The inertial columns, when the source format carried them.
    pub fn inertial(&self) -> Option<&[InertialState]> {
        self.inertial.as_deref()
    }

    /// Whether this trajectory carries velocity and bias columns.
    pub fn has_inertial(&self) -> bool {
        self.inertial.is_some()
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.poses.len()
    }

    /// Whether the trajectory holds no samples.
    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    /// First and last timestamps \[ns\], or `None` when empty.
    pub fn time_span_ns(&self) -> Option<(u64, u64)> {
        Some((
            self.poses.first()?.timestamp_ns,
            self.poses.last()?.timestamp_ns,
        ))
    }

    /// First and last timestamps \[s\], or `None` when empty.
    pub fn time_span_seconds(&self) -> Option<(f64, f64)> {
        let (first, last) = self.time_span_ns()?;
        Some((
            first as f64 / NANOS_PER_SECOND,
            last as f64 / NANOS_PER_SECOND,
        ))
    }

    /// Every timestamp in seconds, for consumers that batch them.
    pub fn timestamps_seconds(&self) -> Vec<f64> {
        self.poses.iter().map(|p| p.timestamp_seconds()).collect()
    }

    /// Every pose as an [`SE3`].
    pub fn se3_poses(&self) -> Vec<SE3> {
        self.poses.iter().map(|p| p.se3()).collect()
    }

    /// Interpolated pose at `timestamp_ns`.
    ///
    /// Position is linearly interpolated; rotation is **slerped**. Returns
    /// `None` outside the covered span: extrapolating ground truth fabricates
    /// the very reference an estimate is being scored against, and clamping at
    /// the ends silently flattens the motion under test. The span is closed —
    /// a query equal to the first or last sample returns that sample.
    pub fn pose_at(&self, timestamp_ns: u64) -> Option<TrajectoryPose> {
        let (before, after, alpha) = self.bracket(timestamp_ns)?;
        let a = self.poses.get(before)?;
        let b = self.poses.get(after)?;

        Some(TrajectoryPose {
            timestamp_ns,
            position: a.position + (b.position - a.position) * alpha,
            orientation: a.orientation.slerp(&b.orientation, alpha),
        })
    }

    /// As [`Self::pose_at`], for a query in seconds.
    ///
    /// `None` for a negative or non-finite `timestamp_s`.
    pub fn pose_at_seconds(&self, timestamp_s: f64) -> Option<TrajectoryPose> {
        self.pose_at(Self::seconds_to_nanos(timestamp_s)?)
    }

    /// Interpolated position at `timestamp_ns`.
    pub fn position_at(&self, timestamp_ns: u64) -> Option<Vector3<f64>> {
        self.pose_at(timestamp_ns).map(|p| p.position)
    }

    /// Interpolated position at a query in seconds.
    pub fn position_at_seconds(&self, timestamp_s: f64) -> Option<Vector3<f64>> {
        self.pose_at_seconds(timestamp_s).map(|p| p.position)
    }

    /// Interpolated pose at `timestamp_ns` as an [`SE3`].
    pub fn se3_at(&self, timestamp_ns: u64) -> Option<SE3> {
        self.pose_at(timestamp_ns).map(|p| p.se3())
    }

    /// Interpolated inertial state at `timestamp_ns`.
    ///
    /// `None` both when the query is out of span **and** when the source format
    /// carried no inertial columns; call [`Self::has_inertial`] to distinguish.
    pub fn inertial_at(&self, timestamp_ns: u64) -> Option<InertialState> {
        let inertial = self.inertial.as_ref()?;
        let (before, after, alpha) = self.bracket(timestamp_ns)?;
        let a = inertial.get(before)?;
        let b = inertial.get(after)?;

        Some(InertialState {
            velocity: a.velocity + (b.velocity - a.velocity) * alpha,
            gyro_bias: a.gyro_bias + (b.gyro_bias - a.gyro_bias) * alpha,
            accel_bias: a.accel_bias + (b.accel_bias - a.accel_bias) * alpha,
        })
    }

    /// Convert a query in seconds to nanoseconds, refusing invalid input.
    fn seconds_to_nanos(timestamp_s: f64) -> Option<u64> {
        if !timestamp_s.is_finite() || timestamp_s < 0.0 {
            return None;
        }
        Some((timestamp_s * NANOS_PER_SECOND).round() as u64)
    }

    /// Indices bracketing `timestamp_ns`, with the interpolation fraction.
    ///
    /// `None` outside the closed span. A zero-length span — duplicate
    /// timestamps — yields `alpha = 0`, returning the earlier sample rather
    /// than dividing by zero.
    fn bracket(&self, timestamp_ns: u64) -> Option<(usize, usize, f64)> {
        let (first, last) = self.time_span_ns()?;
        if timestamp_ns < first || timestamp_ns > last {
            return None;
        }

        // `partition_point` gives the count of samples strictly before the
        // query, so it is the index of the first sample at or after it.
        let upper = self
            .poses
            .partition_point(|p| p.timestamp_ns < timestamp_ns);
        if upper == 0 {
            return Some((0, 0, 0.0));
        }
        let before = upper - 1;
        let after = upper.min(self.poses.len() - 1);

        let t0 = self.poses.get(before)?.timestamp_ns;
        let t1 = self.poses.get(after)?.timestamp_ns;
        let span = t1.saturating_sub(t0);
        let alpha = if span == 0 {
            0.0
        } else {
            (timestamp_ns - t0) as f64 / span as f64
        };
        Some((before, after, alpha))
    }
}

// ---------------------------------------------------------------------------
// Shared parsing and IO helpers for the trajectory codecs
// ---------------------------------------------------------------------------

/// Read a trajectory file into memory, mapping failure to [`TrajectoryError::Io`].
pub(crate) fn read_file(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|source| TrajectoryError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Write trajectory text to `path`, mapping failure to [`TrajectoryError::Io`].
pub(crate) fn write_file(path: &Path, text: &str) -> Result<()> {
    std::fs::write(path, text).map_err(|source| TrajectoryError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Parse row field `index` as a finite `f64`, named for error messages.
pub(crate) fn parse_field(
    row: &[&str],
    index: usize,
    name: &str,
    path: &Path,
    line: usize,
) -> Result<f64> {
    let raw = row.get(index).copied().unwrap_or_default();
    raw.parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| TrajectoryError::InvalidNumber {
            path: path.to_path_buf(),
            line,
            field: name.to_owned(),
            value: raw.to_owned(),
        })
}

/// Parse row field `index` as a `u64`, named for error messages.
///
/// ASL timestamps are integer nanoseconds, so a fractional stamp is refused
/// rather than silently rounded.
pub(crate) fn parse_u64_field(raw: &str, name: &str, path: &Path, line: usize) -> Result<u64> {
    raw.parse::<u64>()
        .map_err(|_| TrajectoryError::InvalidNumber {
            path: path.to_path_buf(),
            line,
            field: name.to_owned(),
            value: raw.to_owned(),
        })
}

// ---------------------------------------------------------------------------
// Bridge to the ASL dataset reader's pose type
// ---------------------------------------------------------------------------

impl From<&[crate::asl::GroundTruthPose]> for Trajectory {
    fn from(poses: &[crate::asl::GroundTruthPose]) -> Self {
        Self::from_poses(poses.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Format dispatch
// ---------------------------------------------------------------------------

/// Load a trajectory, choosing the format from the file extension.
///
/// `.tum` and `.txt` are TUM; `.csv` is ASL. A misdispatch fails loudly rather
/// than misparsing — an ASL row split on whitespace yields one field, and a TUM
/// row split on commas yields one field, so either way the column check
/// rejects it.
///
/// # Errors
///
/// [`TrajectoryError::UnsupportedFormat`] for any other extension, plus
/// whatever the chosen loader raises.
pub fn load_trajectory<P: AsRef<Path>>(path: P) -> Result<Trajectory> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .unwrap_or_default();

    match extension.as_str() {
        "tum" | "txt" => TumLoader::load(path),
        "csv" => crate::asl::AslTrajectoryLoader::load(path),
        other => Err(TrajectoryError::UnsupportedFormat(other.to_owned())),
    }
}

/// Load a trajectory in a stated format, ignoring the extension.
///
/// # Errors
///
/// Whatever the chosen loader raises.
pub fn load_trajectory_as<P: AsRef<Path>>(path: P, format: TrajectoryFormat) -> Result<Trajectory> {
    match format {
        TrajectoryFormat::Tum => TumLoader::load(path),
        TrajectoryFormat::Asl => crate::asl::AslTrajectoryLoader::load(path),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use apex_manifolds::so3::SO3;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    fn pose(ns: u64, x: f64) -> TrajectoryPose {
        TrajectoryPose::new(ns, Vector3::new(x, 0.0, 0.0), SO3::identity())
    }

    fn two_sample() -> Trajectory {
        Trajectory::from_poses(vec![pose(1_000, 0.0), pose(2_000, 10.0)])
    }

    #[test]
    fn interpolating_at_a_sample_returns_that_sample() -> TestResult {
        let t = two_sample();
        let at_first = t.position_at(1_000).ok_or("first sample must resolve")?;
        assert!((at_first.x - 0.0).abs() < 1e-12);
        let at_last = t.position_at(2_000).ok_or("last sample must resolve")?;
        assert!((at_last.x - 10.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn interpolating_midway_is_the_linear_midpoint() -> TestResult {
        let mid = two_sample().position_at(1_500).ok_or("midpoint")?;
        assert!((mid.x - 5.0).abs() < 1e-12, "got {}", mid.x);
        Ok(())
    }

    /// Rotation must be slerped, not lerped componentwise. Halfway between
    /// identity and a 180° turn is 90°; a component lerp cannot produce it.
    #[test]
    fn rotation_is_slerped_not_lerped() -> TestResult {
        let half_turn = SO3::from_axis_angle(&Vector3::x(), std::f64::consts::PI);
        let t = Trajectory::from_poses(vec![
            TrajectoryPose::new(0, Vector3::zeros(), SO3::identity()),
            TrajectoryPose::new(1_000, Vector3::zeros(), half_turn),
        ]);

        let mid = t.pose_at(500).ok_or("midpoint")?;
        let angle = mid.orientation.quaternion().angle();
        assert!(
            (angle - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
            "expected 90 deg, got {} deg",
            angle.to_degrees()
        );
        Ok(())
    }

    /// Extrapolating ground truth fabricates the reference an estimate is being
    /// scored against, so both ends refuse rather than clamp.
    #[test]
    fn queries_outside_the_span_are_refused() -> TestResult {
        let t = two_sample();
        if t.position_at(999).is_some() {
            return Err("a query before the first sample must be refused".into());
        }
        if t.position_at(2_001).is_some() {
            return Err("a query after the last sample must be refused".into());
        }
        Ok(())
    }

    #[test]
    fn duplicate_timestamps_return_the_earlier_sample() -> TestResult {
        let t = Trajectory::from_poses(vec![pose(1_000, 1.0), pose(1_000, 2.0)]);
        let at = t.position_at(1_000).ok_or("duplicate span must resolve")?;
        assert!(
            at.x.is_finite(),
            "a zero-length span must not divide by zero"
        );
        Ok(())
    }

    #[test]
    fn an_empty_trajectory_answers_none_to_everything() -> TestResult {
        let t = Trajectory::default();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(t.time_span_ns().is_none());
        assert!(t.position_at(0).is_none());
        assert!(t.inertial_at(0).is_none());
        Ok(())
    }

    /// The parallel-array trap: sorting poses without applying the same
    /// permutation to the inertial states pairs each pose with another
    /// sample's velocity.
    #[test]
    fn unsorted_input_keeps_poses_and_inertial_aligned() -> TestResult {
        let poses = vec![pose(2_000, 2.0), pose(1_000, 1.0)];
        let inertial = vec![
            InertialState {
                velocity: Vector3::new(2.0, 0.0, 0.0),
                ..InertialState::default()
            },
            InertialState {
                velocity: Vector3::new(1.0, 0.0, 0.0),
                ..InertialState::default()
            },
        ];

        let t = Trajectory::from_poses_and_inertial(poses, inertial)?;
        let states = t.inertial().ok_or("inertial must be present")?;

        // After sorting, sample 0 is t=1000 with x=1.0, so its velocity must
        // also be the one that arrived paired with x=1.0.
        assert!((t.poses()[0].position.x - 1.0).abs() < 1e-12);
        assert!(
            (states[0].velocity.x - 1.0).abs() < 1e-12,
            "pose and inertial were reordered independently"
        );
        assert!((t.poses()[1].position.x - 2.0).abs() < 1e-12);
        assert!((states[1].velocity.x - 2.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn from_poses_and_inertial_rejects_a_length_mismatch() -> TestResult {
        let Err(TrajectoryError::LengthMismatch { got, expected, .. }) =
            Trajectory::from_poses_and_inertial(vec![pose(0, 0.0)], Vec::new())
        else {
            return Err("a length mismatch must be refused".into());
        };
        assert_eq!((got, expected), (0, 1));
        Ok(())
    }

    #[test]
    fn inertial_columns_are_interpolated() -> TestResult {
        let poses = vec![pose(0, 0.0), pose(1_000, 1.0)];
        let inertial = vec![
            InertialState::default(),
            InertialState {
                velocity: Vector3::new(10.0, 0.0, 0.0),
                gyro_bias: Vector3::new(2.0, 0.0, 0.0),
                accel_bias: Vector3::new(4.0, 0.0, 0.0),
            },
        ];
        let t = Trajectory::from_poses_and_inertial(poses, inertial)?;
        let mid = t.inertial_at(500).ok_or("midpoint")?;
        assert!((mid.velocity.x - 5.0).abs() < 1e-12);
        assert!((mid.gyro_bias.x - 1.0).abs() < 1e-12);
        assert!((mid.accel_bias.x - 2.0).abs() < 1e-12);
        Ok(())
    }

    /// A pose-only trajectory answers `None` for inertial queries even inside
    /// the span, which is why `has_inertial` exists.
    #[test]
    fn a_pose_only_trajectory_has_no_inertial_state() -> TestResult {
        let t = two_sample();
        assert!(!t.has_inertial());
        assert!(t.inertial().is_none());
        assert!(t.inertial_at(1_500).is_none());
        assert!(t.position_at(1_500).is_some(), "the pose is still there");
        Ok(())
    }

    #[test]
    fn from_se3_seconds_rejects_a_negative_timestamp() -> TestResult {
        let Err(TrajectoryError::NegativeTimestamp { value, .. }) =
            Trajectory::from_se3_seconds(&[-1.0], &[SE3::identity()])
        else {
            return Err("a negative timestamp must be refused".into());
        };
        assert!(value < 0.0);
        Ok(())
    }

    #[test]
    fn from_se3_seconds_rejects_a_length_mismatch() -> TestResult {
        let Err(TrajectoryError::LengthMismatch { .. }) =
            Trajectory::from_se3_seconds(&[0.0, 1.0], &[SE3::identity()])
        else {
            return Err("a length mismatch must be refused".into());
        };
        Ok(())
    }

    /// The bridge from the dataset reader's pose type still builds a
    /// trajectory by copy.
    #[test]
    fn ground_truth_pose_slice_builds_a_trajectory() -> TestResult {
        let t = two_sample();
        let back = Trajectory::from(t.poses());
        assert_eq!(back.len(), 2);
        assert!((back.poses()[1].position.x - 10.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn se3_conversion_preserves_position_and_rotation() -> TestResult {
        let q = SO3::from_scaled_axis(Vector3::new(0.3, 0.7, 0.1));
        let sample = TrajectoryPose::new(42, Vector3::new(1.0, -2.0, 3.0), q);

        let round = TrajectoryPose::from_se3(42, &sample.se3());
        assert!((round.position - sample.position).norm() < 1e-15);
        assert!(round.orientation.distance(&sample.orientation) < 1e-15);
        Ok(())
    }

    #[test]
    fn dispatch_rejects_an_unsupported_extension() -> TestResult {
        let Err(TrajectoryError::UnsupportedFormat(ext)) = load_trajectory("a.bin") else {
            return Err("an unknown extension must be refused".into());
        };
        assert_eq!(ext, "bin");
        Ok(())
    }
}
