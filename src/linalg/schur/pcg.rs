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
///
/// Two rules, either of which stops the solve — this is Ceres's arrangement of
/// `r_tolerance` plus `q_tolerance`, and the second one matters far more than
/// the first for a Newton step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgParams {
    pub max_iterations: usize,
    /// Relative residual tolerance: stop once `‖r‖ < tolerance · max(‖b‖, 1)`.
    pub tolerance: f64,
    /// Forcing-sequence parameter η for the quadratic-model rule; `0.0`
    /// disables that rule and leaves only the residual one.
    ///
    /// See [`pcg`] for what the rule is and why it is the one that matters.
    pub q_tolerance: f64,
}

/// Default forcing-sequence parameter η.
///
/// Ceres uses `1e-1`; this is deliberately tighter, because `1e-1` measured
/// badly here. Sweep on the four BAL datasets (`ImplicitSparseSchur`, final
/// RMSE against the direct solver, speed against it in parentheses):
///
/// | η | Ladybug | Trafalgar | Dubrovnik | Venice |
/// |---|---|---|---|---|
/// | `1e-1` | **+0.0849** (5.8×) | −0.0128 (1.4×) | −0.0226 (1.3×) | +0.0079 (3.4×) |
/// | `1e-2` | +0.0013 (4.0×) | −0.0104 (0.4×) | −0.0189 (1.3×) | +0.0045 (2.7×) |
/// | `1e-3` | +0.0005 (2.8×) | +0.0022 (0.5×) | +0.0001 (1.1×) | −0.0014 (1.9×) |
///
/// At `1e-1` Ladybug loses 9.7% accuracy and does not recover with more
/// optimizer iterations — the truncated steps converge somewhere worse, not
/// merely more slowly. `1e-2` buys that accuracy back (0.14% off the exact
/// solver) while keeping 4.0×/1.3×/2.7× on the three large datasets, which is
/// where this solver is used at all. `1e-3` is near-exact but gives back most
/// of the speed.
pub const DEFAULT_ETA: f64 = 1e-2;

impl Default for PcgParams {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-6,
            q_tolerance: DEFAULT_ETA,
        }
    }
}

impl PcgParams {
    /// Residual tolerance and iteration cap, with the default forcing sequence.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            ..Self::default()
        }
    }

    /// Override the forcing-sequence parameter. `0.0` turns the
    /// quadratic-model rule off, so only the residual rule and the iteration
    /// cap remain — what a caller wanting an exact solve should ask for.
    pub fn with_q_tolerance(mut self, q_tolerance: f64) -> Self {
        self.q_tolerance = q_tolerance;
        self
    }
}

/// Why [`pcg`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcgTermination {
    /// `‖r‖` fell below the residual tolerance.
    Residual,
    /// The quadratic model stopped improving fast enough — the forcing
    /// sequence is satisfied and further iterations would not pay for
    /// themselves.
    QuadraticModel,
    /// The iteration cap was reached without either rule firing.
    MaxIterations,
    /// The search direction degenerated (`pᵀAp ≈ 0`), which on an SPD system
    /// means the residual is already numerically zero.
    Breakdown,
}

/// Outcome of a [`pcg`] solve.
#[derive(Debug, Clone)]
pub struct PcgResult {
    pub x: Mat<f64>,
    pub iterations: usize,
    pub final_residual: f64,
    /// Which rule ended the solve. [`PcgTermination::MaxIterations`] on a
    /// Newton step means the step is truncated, which is worth logging.
    pub termination: PcgTermination,
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
///
/// # Stopping: why the residual rule is not enough
///
/// CG minimizes the quadratic model `φ(x) = ½·xᵀAx − bᵀx`, and at iterate `x_i`
/// that value is available for free from vectors already on hand:
///
/// ```text
/// φ(x) = ½·xᵀ(A·x) − bᵀx = −½·xᵀ(b + r)        since A·x = b − r
/// ```
///
/// A trust-region optimizer does not need `A·x = b` solved *accurately*; it
/// needs a step that reduces the model. Insisting on a small residual on an
/// ill-conditioned `S` burns iterations long after the model has stopped
/// moving — measured here, every solve on Ladybug ran the full iteration cap
/// without the residual rule ever firing.
///
/// So, following Ceres, the solve also stops when the *relative model
/// improvement per iteration* falls below the forcing sequence `η/i`:
///
/// ```text
/// Q_i = −xᵀ(b + r) ;    i·(Q_i − Q_{i−1}) / Q_i  <  η
/// ```
///
/// The `η/i` shape tightens the requirement as iterations accumulate, so early
/// Newton steps are cheap and approximate while later ones — where the
/// optimizer is close and needs accuracy — are solved harder. Setting
/// [`PcgParams::q_tolerance`] to `0.0` disables it.
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
    let mut termination = PcgTermination::MaxIterations;
    // Q at x = 0 is zero, so the first iteration always has a model to improve.
    let mut q_old = 0.0f64;

    for iter in 0..params.max_iterations {
        iterations = iter + 1;
        ap.fill(0.0);
        apply_operator(&p, &mut ap);

        let p_ap: f64 = (p.transpose() * &ap)[(0, 0)];
        if p_ap.abs() < 1e-20 {
            iterations = iter;
            termination = PcgTermination::Breakdown;
            break;
        }
        let alpha = rz_old / p_ap;

        faer::zip!(&mut x, &p).for_each(|faer::unzip!(x, p)| *x += alpha * p);
        faer::zip!(&mut r, &ap).for_each(|faer::unzip!(r, ap)| *r -= alpha * ap);

        if r.norm_l2() < tol {
            termination = PcgTermination::Residual;
            break;
        }

        // Quadratic model rule: Q_i = −xᵀ(b + r), stop when the relative
        // improvement per iteration drops below the forcing sequence η/i.
        if params.q_tolerance > 0.0 {
            let q_new = -((x.transpose() * (&r + b))[(0, 0)]);
            if q_new.abs() > f64::MIN_POSITIVE {
                let zeta = iterations as f64 * (q_new - q_old) / q_new;
                if zeta < params.q_tolerance {
                    termination = PcgTermination::QuadraticModel;
                    break;
                }
            }
            q_old = q_new;
        }

        apply_preconditioner(&r, &mut z);
        let rz_new: f64 = (r.transpose() * &z)[(0, 0)];
        if rz_old.abs() < 1e-30 {
            termination = PcgTermination::Breakdown;
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
        termination,
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

    /// An ill-conditioned system where the residual rule never fires within
    /// the iteration budget — the exact situation measured on Ladybug, where
    /// every solve ran the full cap. The forcing sequence must cut it short
    /// while still producing a step that reduces the quadratic model.
    #[test]
    fn forcing_sequence_stops_early_on_a_stagnating_residual() {
        // Diagonal, condition number 1e8: CG needs many iterations to drive
        // the residual down but improves the model almost immediately.
        let n = 60;
        let diag: Vec<f64> = (0..n)
            .map(|i| 1.0 + (i as f64 / (n - 1) as f64) * 1e8)
            .collect();
        let b = Mat::from_fn(n, 1, |i, _| ((i % 7) as f64) - 3.0);
        let op = |p: &Mat<f64>, ap: &mut Mat<f64>| {
            for i in 0..n {
                ap[(i, 0)] += diag[i] * p[(i, 0)];
            }
        };
        let identity = |r: &Mat<f64>, z: &mut Mat<f64>| {
            for i in 0..n {
                z[(i, 0)] = r[(i, 0)];
            }
        };

        let tight = pcg(
            &b,
            &PcgParams::new(200, 1e-12).with_q_tolerance(0.0),
            op,
            identity,
        );
        let forced = pcg(&b, &PcgParams::new(200, 1e-12), op, identity);

        assert_eq!(
            forced.termination,
            PcgTermination::QuadraticModel,
            "the forcing sequence should be what stops this solve"
        );
        assert!(
            forced.iterations < tight.iterations,
            "forcing sequence must save iterations: {} vs {}",
            forced.iterations,
            tight.iterations
        );

        // The truncated step must still descend the model φ(x) = ½xᵀAx − bᵀx.
        let phi = |x: &Mat<f64>| {
            let mut acc = 0.0;
            for i in 0..n {
                acc += 0.5 * diag[i] * x[(i, 0)] * x[(i, 0)] - b[(i, 0)] * x[(i, 0)];
            }
            acc
        };
        assert!(phi(&forced.x) < 0.0, "truncated step must reduce the model");
    }

    /// `q_tolerance = 0.0` must reproduce the old behaviour exactly, so callers
    /// that need an exact solve still get one.
    #[test]
    fn zero_q_tolerance_disables_the_forcing_sequence() {
        let a = [[4.0, 1.0], [1.0, 3.0]];
        let b = Mat::from_fn(2, 1, |i, _| [1.0, 2.0][i]);
        let result = pcg(
            &b,
            &PcgParams::new(50, 1e-12).with_q_tolerance(0.0),
            |p, ap| {
                for i in 0..2 {
                    for j in 0..2 {
                        ap[(i, 0)] += a[i][j] * p[(j, 0)];
                    }
                }
            },
            |r, z| {
                z[(0, 0)] = r[(0, 0)];
                z[(1, 0)] = r[(1, 0)];
            },
        );
        assert_ne!(result.termination, PcgTermination::QuadraticModel);
        assert!((result.x[(0, 0)] - 1.0 / 11.0).abs() < 1e-8);
        assert!((result.x[(1, 0)] - 7.0 / 11.0).abs() < 1e-8);
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
