//! Small shared helpers used by the group implementations and by their consumers.

use crate::error::{ElementError, ElementResult};
use crate::so3::SO3;
use nalgebra::{Matrix3, Vector3};

/// Skew-symmetric (cross-product) matrix: `[v]× w = v × w`.
#[inline]
#[must_use]
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Validates a flat parameter block: `len` values, all finite and, when
/// `quaternion_at` is set, a unit quaternion `[w, x, y, z]` starting there.
pub(crate) fn check_param_block(
    group: &'static str,
    s: &[f64],
    len: usize,
    quaternion_at: Option<usize>,
) -> ElementResult<()> {
    if s.len() != len {
        return Err(ElementError::ParamLength {
            group,
            expected: len,
            actual: s.len(),
        });
    }
    if s.iter().any(|value| !value.is_finite()) {
        return Err(ElementError::NonFinite { group });
    }
    if let Some(q) = quaternion_at.and_then(|at| s.get(at..at + 4)) {
        check_unit_quaternion(group, q)?;
    }
    Ok(())
}

/// Validates a finite `[w, x, y, z]` quaternion of unit norm within
/// [`SO3::QUATERNION_NORM_TOLERANCE`].
pub(crate) fn check_unit_quaternion(group: &'static str, q: &[f64]) -> ElementResult<()> {
    let norm = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    if !norm.is_finite() {
        return Err(ElementError::NonFinite { group });
    }
    if (norm - 1.0).abs() > SO3::QUATERNION_NORM_TOLERANCE {
        return Err(ElementError::NonUnitQuaternion { norm });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skew_matches_the_cross_product() {
        let v = Vector3::new(1.0, -2.0, 3.0);
        let w = Vector3::new(-0.5, 4.0, 0.25);
        assert_eq!(skew(&v) * w, v.cross(&w));
    }

    #[test]
    fn skew_is_antisymmetric() {
        let m = skew(&Vector3::new(0.3, 0.7, -1.1));
        assert_eq!(m + m.transpose(), Matrix3::zeros());
    }
}
