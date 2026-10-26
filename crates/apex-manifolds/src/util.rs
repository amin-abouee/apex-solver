//! Small shared helpers used by the group implementations and by their consumers.

use crate::error::{ElementError, ElementResult};
use crate::so3::SO3;
use nalgebra::{Matrix3, Vector3};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Skew-symmetric (cross-product) matrix: `[v]× w = v × w`.
#[inline]
#[must_use]
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Rate limiter for diagnostics raised from per-residual code.
///
/// Every call is counted, but [`WarnOnce::hit`] returns the running count only
/// for the first call and then for each power of ten (1, 10, 100, ...), so a
/// fallback hit millions of times logs a handful of lines instead of one per
/// call.
pub(crate) struct WarnOnce {
    hits: AtomicUsize,
}

impl WarnOnce {
    pub(crate) const fn new() -> Self {
        Self {
            hits: AtomicUsize::new(0),
        }
    }

    /// Counts one occurrence; `Some(count)` when it is time to log.
    pub(crate) fn hit(&self) -> Option<usize> {
        let count = self.hits.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        is_power_of_ten(count).then_some(count)
    }

    /// Occurrences counted so far.
    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.hits.load(Ordering::Relaxed)
    }
}

fn is_power_of_ten(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }
    while n.is_multiple_of(10) {
        n /= 10;
    }
    n == 1
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
    fn warn_once_fires_on_the_first_hit_and_each_power_of_ten() {
        let latch = WarnOnce::new();
        let fired: Vec<usize> = (0..1000).filter_map(|_| latch.hit()).collect();
        assert_eq!(fired, [1, 10, 100, 1000]);
        assert_eq!(latch.count(), 1000);
    }

    #[test]
    fn warn_once_counts_silently_between_logs() {
        let latch = WarnOnce::new();
        assert_eq!(latch.hit(), Some(1));
        assert!((2..10).all(|_| latch.hit().is_none()));
        assert_eq!(latch.hit(), Some(10));
        assert_eq!(latch.count(), 10);
    }

    #[test]
    fn warn_once_counts_across_threads() {
        let latch = WarnOnce::new();
        let logged: usize = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| (0..250).filter(|_| latch.hit().is_some()).count()))
                .collect();
            workers.into_iter().map(|w| w.join().unwrap_or(0)).sum()
        });
        assert_eq!(latch.count(), 1000);
        assert_eq!(logged, 4);
    }

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
