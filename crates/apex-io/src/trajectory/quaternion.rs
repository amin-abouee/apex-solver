//! Quaternion field ordering, isolated to one file.
//!
//! ```text
//! ASL / EuRoC / SE3 :  q_w, q_x, q_y, q_z    — w FIRST
//! TUM               :  q_x, q_y, q_z, q_w    — w LAST
//! nalgebra          :  Quaternion::new(w, i, j, k)  — matches ASL, not TUM
//! ```
//!
//! Getting this wrong does not produce garbage. It produces a different,
//! valid-looking, unit-norm rotation that every downstream tool reads without
//! complaint and every number quietly wrong. It has happened in this codebase
//! before.
//!
//! These four functions are the only place in the crate that names
//! `Quaternion::new` or reads `.w`/`.i`/`.j`/`.k` on a trajectory quaternion.
//! They are named after the **file layout**, not the maths, so the argument
//! names at each call site line up with the parameter names and a swap is a
//! one-word diff in review.

use std::path::Path;

use nalgebra::{Quaternion, UnitQuaternion};

use super::error::{Result, TrajectoryError};

/// Accepted deviation from unit norm.
///
/// Matches the tolerance the ASL dataset reader has always applied, so a file
/// accepted by one path is accepted by the other.
pub(crate) const QUATERNION_NORM_TOLERANCE: f64 = 0.01;

/// Validate and build a unit quaternion, rejecting a non-unit input.
fn checked(path: &Path, line: usize, q: Quaternion<f64>) -> Result<UnitQuaternion<f64>> {
    let norm = q.norm();
    if !norm.is_finite() || (norm - 1.0).abs() > QUATERNION_NORM_TOLERANCE {
        return Err(TrajectoryError::InvalidQuaternion {
            path: path.to_path_buf(),
            line,
            norm,
        });
    }
    Ok(UnitQuaternion::from_quaternion(q))
}

/// Read a quaternion stored **w first** (ASL / EuRoC).
///
/// # Errors
///
/// [`TrajectoryError::InvalidQuaternion`] if the norm is not 1 within
/// [`QUATERNION_NORM_TOLERANCE`].
pub(crate) fn from_wxyz(
    path: &Path,
    line: usize,
    w: f64,
    x: f64,
    y: f64,
    z: f64,
) -> Result<UnitQuaternion<f64>> {
    checked(path, line, Quaternion::new(w, x, y, z))
}

/// Read a quaternion stored **w last** (TUM).
///
/// # Errors
///
/// [`TrajectoryError::InvalidQuaternion`] if the norm is not 1 within
/// [`QUATERNION_NORM_TOLERANCE`].
pub(crate) fn from_xyzw(
    path: &Path,
    line: usize,
    x: f64,
    y: f64,
    z: f64,
    w: f64,
) -> Result<UnitQuaternion<f64>> {
    checked(path, line, Quaternion::new(w, x, y, z))
}

/// Emit fields in **w-first** order (ASL / EuRoC).
pub(crate) fn to_wxyz(q: &UnitQuaternion<f64>) -> [f64; 4] {
    [q.w, q.i, q.j, q.k]
}

/// Emit fields in **w-last** order (TUM).
pub(crate) fn to_xyzw(q: &UnitQuaternion<f64>) -> [f64; 4] {
    [q.i, q.j, q.k, q.w]
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    fn path() -> &'static Path {
        Path::new("test.csv")
    }

    /// The whole point of two constructors: given the same four numbers they
    /// must produce *different* rotations. If they ever agree, one of them has
    /// been edited into a copy of the other and the ordering guard is gone.
    #[test]
    fn wxyz_and_xyzw_disagree_for_an_asymmetric_rotation() -> TestResult {
        // Four distinct components, none equal in magnitude.
        let (a, b, c, d) = (0.8, 0.4, 0.3, 0.32659863237109041);
        let first = from_wxyz(path(), 1, a, b, c, d)?;
        let last = from_xyzw(path(), 1, a, b, c, d)?;
        assert!(
            first.angle_to(&last) > 1e-3,
            "the two orderings produced the same rotation"
        );
        Ok(())
    }

    #[test]
    fn both_constructors_reject_a_non_unit_quaternion() -> TestResult {
        let Err(TrajectoryError::InvalidQuaternion { norm, .. }) =
            from_wxyz(path(), 7, 2.0, 0.0, 0.0, 0.0)
        else {
            return Err("w-first must reject a norm-2 quaternion".into());
        };
        assert!((norm - 2.0).abs() < 1e-12);

        let Err(TrajectoryError::InvalidQuaternion { line, .. }) =
            from_xyzw(path(), 7, 0.0, 0.0, 0.0, 2.0)
        else {
            return Err("w-last must reject a norm-2 quaternion".into());
        };
        assert_eq!(line, 7);
        Ok(())
    }

    /// Each writer must be the inverse of its own reader.
    #[test]
    fn to_wxyz_and_to_xyzw_are_inverses_of_their_readers() -> TestResult {
        let q = from_wxyz(path(), 1, 0.8, 0.4, 0.3, 0.32659863237109041)?;

        let [w, x, y, z] = to_wxyz(&q);
        let round_first = from_wxyz(path(), 1, w, x, y, z)?;
        assert!(q.angle_to(&round_first) < 1e-15);

        let [x, y, z, w] = to_xyzw(&q);
        let round_last = from_xyzw(path(), 1, x, y, z, w)?;
        assert!(q.angle_to(&round_last) < 1e-15);
        Ok(())
    }

    /// A NaN component must be refused, not turned into a NaN rotation.
    #[test]
    fn a_non_finite_component_is_rejected() -> TestResult {
        let Err(TrajectoryError::InvalidQuaternion { .. }) =
            from_wxyz(path(), 1, f64::NAN, 0.0, 0.0, 0.0)
        else {
            return Err("a NaN component must be rejected".into());
        };
        Ok(())
    }
}
