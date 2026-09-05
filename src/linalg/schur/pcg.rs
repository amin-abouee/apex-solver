//! Generic preconditioned conjugate gradient, shared by every Schur solver
//! that needs to solve `S·x = b` without a direct factorization.
//!
//! The two PCG loops that used to live separately — `ExplicitSparseSchur`'s
//! `Iterative` variant (applying `S` explicitly) and `ImplicitSparseSchur`
//! (applying `S` matrix-free) — run the identical classical algorithm; they
//! differ only in how `A·p` and `M⁻¹·r` are computed. Factoring the loop out
//! here means both are pinned by one set of tests instead of two copies that
//! could silently drift apart.

use faer::Mat;

/// Stopping criteria for [`pcg`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgParams {
    pub max_iterations: usize,
    /// Relative tolerance: PCG stops once `‖r‖ < tolerance · max(‖b‖, 1)`.
    pub tolerance: f64,
}

impl Default for PcgParams {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-6,
        }
    }
}

impl PcgParams {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
}

/// Outcome of a [`pcg`] solve.
#[derive(Debug, Clone)]
pub struct PcgResult {
    pub x: Mat<f64>,
    pub iterations: usize,
    pub final_residual: f64,
}

/// Solve `A·x = b` with preconditioned conjugate gradients.
///
/// `apply_operator(p, ap)` must overwrite `ap` with `A·p`. `ap` arrives
/// zeroed on every call, so an operator that only ever accumulates (as the
/// matrix-free Schur operator does) is correct without an explicit reset.
///
/// `apply_preconditioner(r, z)` must overwrite `z` with `M⁻¹·r`.
///
/// `x` starts at the zero vector, matching every caller: each solves for a
/// Newton *step*, not a general linear system that would benefit from a warm
/// start.
pub fn pcg(
    b: &Mat<f64>,
    params: &PcgParams,
    mut apply_operator: impl FnMut(&Mat<f64>, &mut Mat<f64>),
    mut apply_preconditioner: impl FnMut(&Mat<f64>, &mut Mat<f64>),
) -> PcgResult {
    let n = b.nrows();
    let mut x = Mat::<f64>::zeros(n, 1);

    // r = b - A*x (x starts at 0, so r = b)
    let mut r = b.clone();
    let tol = params.tolerance * r.norm_l2().max(1.0);

    let mut z = Mat::<f64>::zeros(n, 1);
    apply_preconditioner(&r, &mut z);

    let mut p = z.clone();
    let mut rz_old: f64 = (r.transpose() * &z)[(0, 0)];

    let mut ap = Mat::<f64>::zeros(n, 1);
    let mut iterations = 0usize;

    for iter in 0..params.max_iterations {
        iterations = iter + 1;
        ap.fill(0.0);
        apply_operator(&p, &mut ap);

        let p_ap: f64 = (p.transpose() * &ap)[(0, 0)];
        if p_ap.abs() < 1e-20 {
            iterations = iter;
            break;
        }
        let alpha = rz_old / p_ap;

        faer::zip!(&mut x, &p).for_each(|faer::unzip!(x, p)| *x += alpha * p);
        faer::zip!(&mut r, &ap).for_each(|faer::unzip!(r, ap)| *r -= alpha * ap);

        if r.norm_l2() < tol {
            break;
        }

        apply_preconditioner(&r, &mut z);
        let rz_new: f64 = (r.transpose() * &z)[(0, 0)];
        if rz_old.abs() < 1e-30 {
            break;
        }
        let beta = rz_new / rz_old;

        faer::zip!(&mut p, &z).for_each(|faer::unzip!(p, z)| *p = *z + beta * *p);
        rz_old = rz_new;
    }

    PcgResult {
        final_residual: r.norm_l2(),
        x,
        iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PCG on a tiny dense SPD system, decoupled from any Schur-specific
    /// setup — the sharpest possible test of the primitive itself.
    #[test]
    fn pcg_solves_small_spd_system_with_identity_preconditioner() {
        // A = [[4, 1], [1, 3]], b = [1, 2] -> x = [1/11, 7/11]
        let a = [[4.0, 1.0], [1.0, 3.0]];
        let b = Mat::from_fn(2, 1, |i, _| [1.0, 2.0][i]);

        let result = pcg(
            &b,
            &PcgParams::new(50, 1e-12),
            |p, ap| {
                for i in 0..2 {
                    let mut acc = 0.0;
                    for j in 0..2 {
                        acc += a[i][j] * p[(j, 0)];
                    }
                    ap[(i, 0)] += acc;
                }
            },
            |r, z| {
                z[(0, 0)] = r[(0, 0)];
                z[(1, 0)] = r[(1, 0)];
            },
        );

        assert!((result.x[(0, 0)] - 1.0 / 11.0).abs() < 1e-8);
        assert!((result.x[(1, 0)] - 7.0 / 11.0).abs() < 1e-8);
        assert!(
            result.iterations <= 2,
            "SPD 2x2 converges in at most 2 CG steps"
        );
    }

    /// A Jacobi preconditioner must still land on the same solution as the
    /// unpreconditioned case — only the convergence path differs.
    #[test]
    fn pcg_jacobi_preconditioner_matches_identity_solution() {
        let a = [[10.0, 1.0], [1.0, 8.0]];
        let b = Mat::from_fn(2, 1, |i, _| [3.0, 5.0][i]);
        let inv_diag = [1.0 / a[0][0], 1.0 / a[1][1]];

        let result = pcg(
            &b,
            &PcgParams::new(50, 1e-12),
            |p, ap| {
                for i in 0..2 {
                    let mut acc = 0.0;
                    for j in 0..2 {
                        acc += a[i][j] * p[(j, 0)];
                    }
                    ap[(i, 0)] += acc;
                }
            },
            |r, z| {
                z[(0, 0)] = inv_diag[0] * r[(0, 0)];
                z[(1, 0)] = inv_diag[1] * r[(1, 0)];
            },
        );

        // Direct solve for reference.
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        let want0 = (b[(0, 0)] * a[1][1] - a[0][1] * b[(1, 0)]) / det;
        let want1 = (a[0][0] * b[(1, 0)] - b[(0, 0)] * a[1][0]) / det;

        assert!((result.x[(0, 0)] - want0).abs() < 1e-8);
        assert!((result.x[(1, 0)] - want1).abs() < 1e-8);
    }

    /// `p^T A p ≈ 0` on the very first iteration (a zero RHS) must not panic
    /// or divide into a NaN.
    #[test]
    fn pcg_handles_zero_rhs() {
        let b = Mat::<f64>::zeros(3, 1);
        let result = pcg(
            &b,
            &PcgParams::default(),
            |_p, _ap| {},
            |r, z| {
                z[(0, 0)] = r[(0, 0)];
                z[(1, 0)] = r[(1, 0)];
                z[(2, 0)] = r[(2, 0)];
            },
        );
        for i in 0..3 {
            assert!(result.x[(i, 0)].is_finite());
        }
    }
}
