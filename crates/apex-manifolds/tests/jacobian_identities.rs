//! Jacobian identity battery across the fixed-size Lie groups.
//!
//! Every check here is built from a *different* code path than the analytic
//! Jacobian it is checking:
//!
//! * **Definition FD** — `Jr(ξ)` and `Jl(ξ)` are *defined* by
//!   `exp(ξ+δ) = exp(ξ)∘exp(Jr(ξ)·δ)` and `exp(ξ+δ) = exp(Jl(ξ)·δ)∘exp(ξ)`,
//!   so their columns are central differences of
//!   `log(exp(ξ)⁻¹exp(ξ+δ))` and `log(exp(ξ+δ)exp(ξ)⁻¹)`. Only `exp`,
//!   `inverse`, `compose` and `log` *values* are used — never another
//!   Jacobian — so a `right_jacobian` that agrees with its own
//!   `right_jacobian_inv` but not with the exponential map cannot cancel
//!   itself out here (the trap `subgroup_law.rs` documents for SGal(3)).
//!   `SE2` and `SGal3` had no such FD coverage at all before this file.
//! * **Algebraic identities** — `Jl(ξ) = Jr(−ξ)`, `Jl(ξ) = Ad(exp ξ)·Jr(ξ)`,
//!   `Jr·Jr⁻¹ = I`, `Jl·Jl⁻¹ = I`: pure matrix identities tying the four
//!   Jacobians and `adjoint()` together, with no finite differences in play.
//! * **Retraction chains** — the Jacobians that `compose`, `right_plus`,
//!   `right_minus`, `left_plus` and `left_minus` hand back, checked against
//!   central differences of those very operations. `right_plus` and
//!   `right_minus` are the ones production requests
//!   (`src/factors/imu/se23/factors.rs`, `src/factors/imu/sgal3/factors.rs`);
//!   `compose` is what `BetweenFactor` chains onto `between`.
//!
//! # Conventions
//!
//! Group-returning operations are differentiated in the output's **right
//! local coordinates** (an output perturbation `η` in `result·exp(η)`).
//! Tangent-returning operations are differentiated as plain tangent-vector
//! differences. Inputs are always perturbed on the **right** — the
//! convention `compose`, `between` and `right_plus` implement and that
//! production factors depend on.
//!
//! All failures for a group are collected and reported together, so one run
//! shows the whole picture instead of stopping at the first mismatch.

use apex_manifolds::se2::SE2;
use apex_manifolds::se3::SE3;
use apex_manifolds::se23::SE23;
use apex_manifolds::sgal3::SGal3;
use apex_manifolds::sim3::Sim3;
use apex_manifolds::{LieGroup, Tangent};

/// Step for every central difference. Large enough to stay clear of f64
/// rounding in `log`, small enough that the O(eps²) truncation lands well
/// under [`FD_TOL`].
const FD_EPS: f64 = 1e-6;
/// Tolerance for finite-difference comparisons, relative to the entry
/// magnitude (`|a-b| / (1 + |a|)`).
const FD_TOL: f64 = 1e-6;
/// Tolerance for exact algebraic identities — no truncation, so the only
/// error is rounding. The analytically-computed groups (SE2, SE3, SE23,
/// Sim3) land near machine precision; SGal3 lands near 1e-10 because its
/// `right_jacobian`/`left_jacobian` are themselves central differences
/// through `exp`/`log` (see `sgal3.rs`) and inherit that stencil's
/// roundoff, `eps·|log|/EPS ≈ 2e-10`. 1e-9 stays far below any structural
/// mistake — a mis-ordered adjoint block shows up at 1e-1 or worse — while
/// not failing on that known stencil noise.
const ALG_TOL: f64 = 1e-9;

/// Fixed generator sequence; a group of dimension `n` takes `n` entries
/// starting at a caller-chosen offset, so different offsets give directions
/// that are neither equal nor collinear (which would make two-element
/// checks such as `between` degenerate).
const SEQ: [f64; 16] = [
    0.42, -0.31, 0.27, 0.55, -0.44, 0.19, 0.36, -0.52, 0.23, 0.47, -0.29, 0.34, 0.17, -0.38, 0.49,
    -0.21,
];

/// Deterministic tangent of the group's own dimension.
fn tangent<G: LieGroup>(start: usize) -> G::TangentVector {
    let buf: Vec<f64> = (0..G::TangentVector::DIM)
        .map(|i| SEQ[(start + i) % SEQ.len()])
        .collect();
    G::TangentVector::from_slice(&buf)
}

/// A deterministic, non-identity group element — `exp` of a fixed tangent,
/// so it works for every group without knowing any constructor.
fn element<G: LieGroup>(start: usize) -> G {
    tangent::<G>(start).exp(None)
}

/// `ξ` with component `k` shifted by `step`.
fn shifted<G: LieGroup>(t: &G::TangentVector, k: usize, step: f64) -> G::TangentVector {
    let mut buf = t.as_slice().to_vec();
    buf[k] += step;
    G::TangentVector::from_slice(&buf)
}

/// The pure right perturbation `±eps·e_k`, used to move a *group element*
/// rather than a tangent.
fn unit<G: LieGroup>(k: usize, step: f64) -> G::TangentVector {
    let mut buf = vec![0.0; G::TangentVector::DIM];
    buf[k] = step;
    G::TangentVector::from_slice(&buf)
}

/// `a - b` in the tangent's flat representation.
fn sub<G: LieGroup>(a: &G::TangentVector, b: &G::TangentVector) -> Vec<f64> {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| x - y)
        .collect()
}

fn scaled(v: Vec<f64>, factor: f64) -> Vec<f64> {
    v.into_iter().map(|x| x * factor).collect()
}

fn negate(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| -x).collect()
}

/// `fd[k][i]` is entry `(i, k)` of the finite-difference Jacobian.
fn fd_entry_err<G: LieGroup>(analytic: &G::JacobianMatrix, fd: &[Vec<f64>]) -> f64 {
    let n = G::TangentVector::DIM;
    let mut worst: f64 = 0.0;
    for k in 0..n {
        for i in 0..n {
            worst = worst.max((analytic[(i, k)] - fd[k][i]).abs() / (1.0 + analytic[(i, k)].abs()));
        }
    }
    worst
}

fn matrix_err<G: LieGroup>(a: &G::JacobianMatrix, b: &G::JacobianMatrix) -> f64 {
    let n = G::TangentVector::DIM;
    let mut worst: f64 = 0.0;
    for i in 0..n {
        for k in 0..n {
            worst = worst.max((a[(i, k)] - b[(i, k)]).abs() / (1.0 + a[(i, k)].abs()));
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// Finite-difference builders. Each returns one column per tangent direction.
// ---------------------------------------------------------------------------

/// Columns of `Jr(ξ)` from its definition:
/// `log(exp(ξ)⁻¹·exp(ξ+δ)) = Jr(ξ)·δ`.
fn fd_right_jacobian<G: LieGroup>(xi: &G::TangentVector) -> Vec<Vec<f64>> {
    let base = xi.exp(None);
    (0..G::TangentVector::DIM)
        .map(|k| {
            let dp = shifted::<G>(xi, k, FD_EPS)
                .exp(None)
                .right_minus(&base, None, None);
            let dm = shifted::<G>(xi, k, -FD_EPS)
                .exp(None)
                .right_minus(&base, None, None);
            scaled(sub::<G>(&dp, &dm), 1.0 / (2.0 * FD_EPS))
        })
        .collect()
}

/// Columns of `Jl(ξ)` from its definition:
/// `log(exp(ξ+δ)·exp(ξ)⁻¹) = Jl(ξ)·δ`.
fn fd_left_jacobian<G: LieGroup>(xi: &G::TangentVector) -> Vec<Vec<f64>> {
    let base = xi.exp(None);
    (0..G::TangentVector::DIM)
        .map(|k| {
            let dp = shifted::<G>(xi, k, FD_EPS)
                .exp(None)
                .left_minus(&base, None, None);
            let dm = shifted::<G>(xi, k, -FD_EPS)
                .exp(None)
                .left_minus(&base, None, None);
            scaled(sub::<G>(&dp, &dm), 1.0 / (2.0 * FD_EPS))
        })
        .collect()
}

/// Columns of `∂(a∘b)/∂a` and `∂(a∘b)/∂b`: perturb an operand on the right,
/// re-read the product in its own right-local coordinates.
fn fd_compose<G: LieGroup>(a: &G, b: &G) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let r0 = a.compose(b, None, None);
    let n = G::TangentVector::DIM;
    let mut wrt_a = Vec::with_capacity(n);
    let mut wrt_b = Vec::with_capacity(n);
    for k in 0..n {
        let a_p = a.right_plus(&unit::<G>(k, FD_EPS), None, None);
        let a_m = a.right_plus(&unit::<G>(k, -FD_EPS), None, None);
        let eta_p = a_p.compose(b, None, None).right_minus(&r0, None, None);
        let eta_m = a_m.compose(b, None, None).right_minus(&r0, None, None);
        wrt_a.push(scaled(sub::<G>(&eta_p, &eta_m), 1.0 / (2.0 * FD_EPS)));

        let b_p = b.right_plus(&unit::<G>(k, FD_EPS), None, None);
        let b_m = b.right_plus(&unit::<G>(k, -FD_EPS), None, None);
        let eta_p = a.compose(&b_p, None, None).right_minus(&r0, None, None);
        let eta_m = a.compose(&b_m, None, None).right_minus(&r0, None, None);
        wrt_b.push(scaled(sub::<G>(&eta_p, &eta_m), 1.0 / (2.0 * FD_EPS)));
    }
    (wrt_a, wrt_b)
}

/// Columns of `∂(g ⊞ φ)/∂φ` and `∂(g ⊞ φ)/∂g`.
fn fd_right_plus<G: LieGroup>(g: &G, phi: &G::TangentVector) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let r0 = g.right_plus(phi, None, None);
    let n = G::TangentVector::DIM;
    let mut wrt_tangent = Vec::with_capacity(n);
    let mut wrt_self = Vec::with_capacity(n);
    for k in 0..n {
        let r_p = g.right_plus(&shifted::<G>(phi, k, FD_EPS), None, None);
        let r_m = g.right_plus(&shifted::<G>(phi, k, -FD_EPS), None, None);
        let eta_p = r_p.right_minus(&r0, None, None);
        let eta_m = r_m.right_minus(&r0, None, None);
        wrt_tangent.push(scaled(sub::<G>(&eta_p, &eta_m), 1.0 / (2.0 * FD_EPS)));

        let g_p = g.right_plus(&unit::<G>(k, FD_EPS), None, None);
        let g_m = g.right_plus(&unit::<G>(k, -FD_EPS), None, None);
        let eta_p = g_p.right_plus(phi, None, None).right_minus(&r0, None, None);
        let eta_m = g_m.right_plus(phi, None, None).right_minus(&r0, None, None);
        wrt_self.push(scaled(sub::<G>(&eta_p, &eta_m), 1.0 / (2.0 * FD_EPS)));
    }
    (wrt_tangent, wrt_self)
}

/// Columns of `∂(g₁ ⊟ g₂)/∂g₁` and `∂(g₁ ⊟ g₂)/∂g₂`. The output is a
/// tangent vector, so the differences are plain — no re-localization.
fn fd_right_minus<G: LieGroup>(g1: &G, g2: &G) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = G::TangentVector::DIM;
    let mut wrt_g1 = Vec::with_capacity(n);
    let mut wrt_g2 = Vec::with_capacity(n);
    for k in 0..n {
        let g1_p = g1.right_plus(&unit::<G>(k, FD_EPS), None, None);
        let g1_m = g1.right_plus(&unit::<G>(k, -FD_EPS), None, None);
        let h_p = g1_p.right_minus(g2, None, None);
        let h_m = g1_m.right_minus(g2, None, None);
        wrt_g1.push(scaled(sub::<G>(&h_p, &h_m), 1.0 / (2.0 * FD_EPS)));

        let g2_p = g2.right_plus(&unit::<G>(k, FD_EPS), None, None);
        let g2_m = g2.right_plus(&unit::<G>(k, -FD_EPS), None, None);
        let h_p = g1.right_minus(&g2_p, None, None);
        let h_m = g1.right_minus(&g2_m, None, None);
        wrt_g2.push(scaled(sub::<G>(&h_p, &h_m), 1.0 / (2.0 * FD_EPS)));
    }
    (wrt_g1, wrt_g2)
}

/// Columns of `∂(φ ⊞ g)/∂φ` and `∂(φ ⊞ g)/∂g`.
fn fd_left_plus<G: LieGroup>(g: &G, phi: &G::TangentVector) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let r0 = g.left_plus(phi, None, None);
    let n = G::TangentVector::DIM;
    let mut wrt_tangent = Vec::with_capacity(n);
    let mut wrt_self = Vec::with_capacity(n);
    for k in 0..n {
        let r_p = g.left_plus(&shifted::<G>(phi, k, FD_EPS), None, None);
        let r_m = g.left_plus(&shifted::<G>(phi, k, -FD_EPS), None, None);
        let eta_p = r_p.right_minus(&r0, None, None);
        let eta_m = r_m.right_minus(&r0, None, None);
        wrt_tangent.push(scaled(sub::<G>(&eta_p, &eta_m), 1.0 / (2.0 * FD_EPS)));

        let g_p = g.right_plus(&unit::<G>(k, FD_EPS), None, None);
        let g_m = g.right_plus(&unit::<G>(k, -FD_EPS), None, None);
        let eta_p = g_p.left_plus(phi, None, None).right_minus(&r0, None, None);
        let eta_m = g_m.left_plus(phi, None, None).right_minus(&r0, None, None);
        wrt_self.push(scaled(sub::<G>(&eta_p, &eta_m), 1.0 / (2.0 * FD_EPS)));
    }
    (wrt_tangent, wrt_self)
}

/// Columns of `∂(g₁ ⊟_left g₂)/∂g₁` and `∂(g₁ ⊟_left g₂)/∂g₂`.
fn fd_left_minus<G: LieGroup>(g1: &G, g2: &G) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = G::TangentVector::DIM;
    let mut wrt_g1 = Vec::with_capacity(n);
    let mut wrt_g2 = Vec::with_capacity(n);
    for k in 0..n {
        let g1_p = g1.right_plus(&unit::<G>(k, FD_EPS), None, None);
        let g1_m = g1.right_plus(&unit::<G>(k, -FD_EPS), None, None);
        let h_p = g1_p.left_minus(g2, None, None);
        let h_m = g1_m.left_minus(g2, None, None);
        wrt_g1.push(scaled(sub::<G>(&h_p, &h_m), 1.0 / (2.0 * FD_EPS)));

        let g2_p = g2.right_plus(&unit::<G>(k, FD_EPS), None, None);
        let g2_m = g2.right_plus(&unit::<G>(k, -FD_EPS), None, None);
        let h_p = g1.left_minus(&g2_p, None, None);
        let h_m = g1.left_minus(&g2_m, None, None);
        wrt_g2.push(scaled(sub::<G>(&h_p, &h_m), 1.0 / (2.0 * FD_EPS)));
    }
    (wrt_g1, wrt_g2)
}

// ---------------------------------------------------------------------------
// The battery
// ---------------------------------------------------------------------------

/// Run every identity and FD chain for `G`, collecting all mismatches so a
/// single run reports the whole picture.
fn run_battery<G: LieGroup>() {
    let mut failures: Vec<String> = Vec::new();
    let n = G::TangentVector::DIM;
    let xi = tangent::<G>(0);
    let g = element::<G>(0);
    let h = element::<G>(5);

    // --- 1. Definition-level FD of Jr and Jl -----------------------------
    // Note: SGal3 implements both Jacobians *as* this very central
    // difference (see `sgal3.rs`), so for that group this check compares
    // two stencils rather than an analytic formula against one — its real
    // independent coverage comes from the `Ad` identity and the retraction
    // chains below. For SE2/SE3/SE23/Sim3 it checks the closed-form
    // formulas against exp/log.
    let jr = xi.right_jacobian();
    let jl = xi.left_jacobian();
    note(
        &mut failures,
        "right_jacobian vs exp/log central difference",
        fd_entry_err::<G>(&jr, &fd_right_jacobian::<G>(&xi)),
        FD_TOL,
    );
    note(
        &mut failures,
        "left_jacobian vs exp/log central difference",
        fd_entry_err::<G>(&jl, &fd_left_jacobian::<G>(&xi)),
        FD_TOL,
    );

    // --- 2. Algebraic identities ----------------------------------------
    let neg_xi = G::TangentVector::from_slice(&negate(xi.as_slice()));
    note(
        &mut failures,
        "Jl(xi) == Jr(-xi)",
        matrix_err::<G>(&neg_xi.right_jacobian(), &jl),
        ALG_TOL,
    );
    note(
        &mut failures,
        "Jl(xi) == Ad(exp xi) * Jr(xi)",
        matrix_err::<G>(&(g.adjoint() * jr.clone()), &jl),
        ALG_TOL,
    );
    note(
        &mut failures,
        "Jr * Jr_inv == I",
        matrix_err::<G>(&(jr * xi.right_jacobian_inv()), &G::jacobian_identity()),
        ALG_TOL,
    );
    note(
        &mut failures,
        "Jl * Jl_inv == I",
        matrix_err::<G>(&(jl * xi.left_jacobian_inv()), &G::jacobian_identity()),
        ALG_TOL,
    );

    // --- 3. Retraction chains -------------------------------------------
    let (fd_a, fd_b) = fd_compose(&g, &h);
    let mut js = G::zero_jacobian();
    let mut jo = G::zero_jacobian();
    let _ = g.compose(&h, Some(&mut js), Some(&mut jo));
    note(
        &mut failures,
        "compose wrt self",
        fd_entry_err::<G>(&js, &fd_a),
        FD_TOL,
    );
    note(
        &mut failures,
        "compose wrt other",
        fd_entry_err::<G>(&jo, &fd_b),
        FD_TOL,
    );

    let (fd_tan, fd_self) = fd_right_plus(&g, &xi);
    let mut jt = G::zero_jacobian();
    let mut js = G::zero_jacobian();
    let _ = g.right_plus(&xi, Some(&mut js), Some(&mut jt));
    note(
        &mut failures,
        "right_plus wrt tangent",
        fd_entry_err::<G>(&jt, &fd_tan),
        FD_TOL,
    );
    note(
        &mut failures,
        "right_plus wrt self",
        fd_entry_err::<G>(&js, &fd_self),
        FD_TOL,
    );

    let (fd_g1, fd_g2) = fd_right_minus(&g, &h);
    let mut j1 = G::zero_jacobian();
    let mut j2 = G::zero_jacobian();
    let _ = g.right_minus(&h, Some(&mut j1), Some(&mut j2));
    note(
        &mut failures,
        "right_minus wrt self",
        fd_entry_err::<G>(&j1, &fd_g1),
        FD_TOL,
    );
    note(
        &mut failures,
        "right_minus wrt other",
        fd_entry_err::<G>(&j2, &fd_g2),
        FD_TOL,
    );

    let (fd_tan, fd_self) = fd_left_plus(&g, &xi);
    let mut jt = G::zero_jacobian();
    let mut js = G::zero_jacobian();
    let _ = g.left_plus(&xi, Some(&mut jt), Some(&mut js));
    note(
        &mut failures,
        "left_plus wrt tangent",
        fd_entry_err::<G>(&jt, &fd_tan),
        FD_TOL,
    );
    note(
        &mut failures,
        "left_plus wrt self",
        fd_entry_err::<G>(&js, &fd_self),
        FD_TOL,
    );

    let (fd_g1, fd_g2) = fd_left_minus(&g, &h);
    let mut j1 = G::zero_jacobian();
    let mut j2 = G::zero_jacobian();
    let _ = g.left_minus(&h, Some(&mut j1), Some(&mut j2));
    note(
        &mut failures,
        "left_minus wrt self",
        fd_entry_err::<G>(&j1, &fd_g1),
        FD_TOL,
    );
    note(
        &mut failures,
        "left_minus wrt other",
        fd_entry_err::<G>(&j2, &fd_g2),
        FD_TOL,
    );

    assert!(
        failures.is_empty(),
        "{} (dim {n}) battery failures:\n  {}",
        G::NAME,
        failures.join("\n  ")
    );
}

/// Record `what` as a failure unless `err` is within `tol`.
fn note(failures: &mut Vec<String>, what: &str, err: f64, tol: f64) {
    if err.is_nan() || err > tol {
        failures.push(format!("{what}: err={err:.3e} > tol={tol:.1e}"));
    }
}

#[test]
fn jacobian_identity_battery_se2() {
    run_battery::<SE2>();
}

#[test]
fn jacobian_identity_battery_se3() {
    run_battery::<SE3>();
}

#[test]
fn jacobian_identity_battery_se23() {
    run_battery::<SE23>();
}

#[test]
fn jacobian_identity_battery_sgal3() {
    run_battery::<SGal3>();
}

#[test]
fn jacobian_identity_battery_sim3() {
    run_battery::<Sim3>();
}
