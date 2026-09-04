//! Regularization policy for near-singular linear algebra.
//!
//! Three levels, three different jobs — unified here so the rule is stated
//! once instead of drifting per call site:
//!
//! 1. **Damping** ([`Damping`](crate::linalg::Damping)): `λ·clamp(H_jj)` added
//!    *before* the solve, chosen by the optimizer (LM schedule, GN floor,
//!    DogLeg μ). This shapes convergence; it is not a rescue path.
//! 2. **Block retry** ([`invert_with_retry_1`]/[`invert_with_retry_3`]/
//!    [`invert_with_retry_dyn`]): when one small diagonal block fails to
//!    invert, retry once with a trace-scaled Tikhonov shift
//!    ([`tikhonov_scale`]). The shift is proportional to the block's own
//!    scale, so well-scaled problems barely notice it.
//! 3. **System retry** (`solve_with_cholesky`'s 5-level loop): when the whole
//!    reduced system fails to factorize, retry with exponentially growing
//!    regularization. Coarse, last-resort, loud (debug-logged per attempt).
//!
//! Level 2 exists because level 3 is the wrong tool for a single bad landmark:
//! re-factorizing `S` five times over one unobserved point wastes a full
//! symbolic+numeric cycle per attempt. Conversely, the matrix-free
//! (`Iterative`) path keeps its own eigen-gated per-block policy
//! (`regularize_landmark_block` in `implicit_schur.rs`): it already computes
//! eigenvalues for the Schur-Jacobi preconditioner, so reusing that
//! information is cheaper than a blind retry there. Both block policies are
//! pinned by the tests in this module to never fail spuriously and never
//! produce NaN — beyond that, their exact shift amounts are allowed to differ,
//! and the cross-solver agreement tests bound the resulting step differences.

use nalgebra::{DMatrix, Matrix3};

/// Tikhonov shift for a block whose diagonal sums (in absolute value) to
/// `diag_abs_sum`: `max(1e-6 · mean, 1e-8)`.
///
/// The floor keeps exactly-zero blocks invertible; the proportional term keeps
/// the shift invisible next to healthy curvature.
pub(crate) fn tikhonov_scale(diag_abs_sum: f64, n: usize) -> f64 {
    (1e-6 * diag_abs_sum / n.max(1) as f64).max(1e-8)
}

/// Invert a 1×1 block, retrying with [`tikhonov_scale`] on near-zero input.
///
/// Sign is preserved for any usable value (a nonzero indefinite scalar is
/// still invertible); the shift only rescues magnitudes at or below
/// floating-point noise.
pub(crate) fn invert_with_retry_1(v: f64) -> Option<f64> {
    if v.abs() > f64::EPSILON {
        return Some(1.0 / v);
    }
    // Tikhonov shift toward +inf, mirroring the matrix retries.
    let shifted = v + tikhonov_scale(v.abs(), 1);
    if shifted.abs() > f64::EPSILON {
        Some(1.0 / shifted)
    } else {
        None
    }
}

/// Invert a 3×3 block, retrying with [`tikhonov_scale`] on the trace.
pub(crate) fn invert_with_retry_3(m: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    if let Some(inv) = m.try_inverse() {
        return Some(inv);
    }
    let reg = tikhonov_scale(m.diagonal().iter().sum::<f64>().abs(), 3);
    (m + Matrix3::identity() * reg).try_inverse()
}

/// Dimension-generic counterpart of [`invert_with_retry_3`].
pub(crate) fn invert_with_retry_dyn(m: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if let Some(inv) = m.clone().try_inverse() {
        return Some(inv);
    }
    let n = m.nrows();
    let reg = tikhonov_scale(m.diagonal().iter().sum::<f64>().abs(), n);
    (m + DMatrix::identity(n, n) * reg).try_inverse()
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn tikhonov_scale_floors_and_scales() -> TestResult {
        // Exactly zero still gets the floor: invertible, not infinite.
        assert_eq!(tikhonov_scale(0.0, 3), 1e-8);
        // Healthy curvature: proportional and invisible next to it.
        assert!((tikhonov_scale(300.0, 3) - 1e-4).abs() < 1e-12);
        // Degenerate n is guarded, never divides by zero.
        assert_eq!(tikhonov_scale(0.0, 0), 1e-8);
        Ok(())
    }

    #[test]
    fn retry_helpers_agree_with_direct_inverse_when_healthy() -> TestResult {
        // On well-conditioned input the retry path must be a no-op: every
        // helper returns the exact inverse, so policies cannot drift apart
        // where it matters.
        assert!((invert_with_retry_1(4.0).ok_or("scalar")? - 0.25).abs() < 1e-15);
        let m3 = Matrix3::new(4.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0);
        let inv3 = invert_with_retry_3(&m3).ok_or("3x3")?;
        let identity = m3 * inv3;
        for r in 0..3 {
            for c in 0..3 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((identity[(r, c)] - want).abs() < 1e-12);
            }
        }
        let md = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let invd = invert_with_retry_dyn(&md).ok_or("dyn")?;
        let identity = md * invd;
        for r in 0..2 {
            for c in 0..2 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((identity[(r, c)] - want).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn retry_helpers_rescue_singular_blocks_without_nan() -> TestResult {
        let inv0 = invert_with_retry_1(0.0).ok_or("zero scalar must take the shift")?;
        assert!(inv0.is_finite() && inv0 > 0.0);
        let z3 = Matrix3::<f64>::zeros();
        let invz = invert_with_retry_3(&z3).ok_or("zero 3x3 must take the shift")?;
        assert!(invz.iter().all(|v| v.is_finite()));
        let zd = DMatrix::<f64>::zeros(2, 2);
        let invd = invert_with_retry_dyn(&zd).ok_or("zero dyn must take the shift")?;
        assert!(invd.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn implicit_and_shared_policies_agree_where_healthy() -> TestResult {
        use crate::linalg::sparse::implicit_schur::regularize_landmark_block;

        // Below every threshold both policies take the plain inverse, so they
        // must return bit-comparable answers — this is the precise sense in
        // which "all policies agree where it matters".
        let healthy = Matrix3::new(4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0);
        let shared = invert_with_retry_3(&healthy).ok_or("shared")?;
        let implicit = regularize_landmark_block(&healthy).map_err(|e| format!("implicit: {e}"))?;
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (shared[(r, c)] - implicit[(r, c)]).abs() < 1e-15,
                    "healthy-block divergence at ({r},{c})"
                );
            }
        }

        // Past the thresholds the amounts legitimately differ (eigen-gated
        // tiers vs trace-scaled retry); both must still succeed finite.
        // cond ~1e12 trips the implicit severe branch.
        let ill = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e-12);
        let shared_ill = invert_with_retry_3(&ill).ok_or("shared ill")?;
        let implicit_ill =
            regularize_landmark_block(&ill).map_err(|e| format!("implicit ill: {e}"))?;
        assert!(shared_ill.iter().all(|v| v.is_finite()));
        assert!(implicit_ill.iter().all(|v| v.is_finite()));
        Ok(())
    }
}
