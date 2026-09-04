//! The one-parameter subgroup law: `exp(a·ξ) ∘ exp(b·ξ) == exp((a+b)·ξ)`.
//!
//! This is the property that ties `exp` to `compose`. It is independent of
//! `log` (no round-trip escape hatch), so an `exp` that is merely a
//! retraction — smooth, exp(0)=I, first-order correct — passes every
//! round-trip and FD-Jacobian test while still failing here.
//!
//! Reference: the defect that motivated this suite (SGal(3)) is documented in
//! `doc/claude_review/02-confirmed-critical.md` §1.

use apex_manifolds::rn::RnTangent;
use apex_manifolds::se2::SE2Tangent;
use apex_manifolds::se3::SE3Tangent;
use apex_manifolds::se23::SE23Tangent;
use apex_manifolds::sgal3::SGal3Tangent;
use apex_manifolds::sim3::Sim3Tangent;
use apex_manifolds::so2::SO2Tangent;
use apex_manifolds::so3::SO3Tangent;
use apex_manifolds::{LieGroup, Tangent};
use nalgebra::{DVector, Vector3};

const TOL: f64 = 1e-9;
/// Split so that a+b stays inside the canonical range of the rotation groups.
const A: f64 = 0.3;
const B: f64 = 0.5;

#[test]
fn subgroup_law_so2() {
    let xi = 0.7;
    let left = SO2Tangent::new(A * xi).exp(None);
    let right = SO2Tangent::new(B * xi).exp(None);
    let both = left.compose(&right, None, None);
    let direct = SO2Tangent::new((A + B) * xi).exp(None);
    assert!(
        both.is_approx(&direct, TOL),
        "SO2: exp(a)∘exp(b) != exp(a+b) — {both} vs {direct}"
    );
}

#[test]
fn subgroup_law_so3() {
    let xi = Vector3::new(0.4, -0.25, 0.6);
    let scale = |t: f64| SO3Tangent::new(xi * t);
    let left = scale(A).exp(None);
    let right = scale(B).exp(None);
    let both = left.compose(&right, None, None);
    let direct = scale(A + B).exp(None);
    assert!(
        both.is_approx(&direct, TOL),
        "SO3: exp(a)∘exp(b) != exp(a+b)"
    );
}

#[test]
fn subgroup_law_se2() {
    let (x, y, theta) = (0.4, -0.3, 0.25);
    let scale = |t: f64| SE2Tangent::new(x * t, y * t, theta * t);
    let left = scale(A).exp(None);
    let right = scale(B).exp(None);
    let both = left.compose(&right, None, None);
    let direct = scale(A + B).exp(None);
    assert!(
        both.is_approx(&direct, TOL),
        "SE2: exp(a)∘exp(b) != exp(a+b)"
    );
}

#[test]
fn subgroup_law_se3() {
    let (rho, theta) = (Vector3::new(0.3, -0.5, 0.2), Vector3::new(0.4, 0.25, -0.6));
    let scale = |t: f64| SE3Tangent::new(rho * t, theta * t);
    let left = scale(A).exp(None);
    let right = scale(B).exp(None);
    let both = left.compose(&right, None, None);
    let direct = scale(A + B).exp(None);
    assert!(
        both.is_approx(&direct, TOL),
        "SE3: exp(a)∘exp(b) != exp(a+b)"
    );
}

#[test]
fn subgroup_law_se23() {
    let (rho, theta, nu) = (
        Vector3::new(0.3, -0.5, 0.2),
        Vector3::new(0.4, 0.25, -0.6),
        Vector3::new(-0.35, 0.15, 0.45),
    );
    let scale = |t: f64| SE23Tangent::new(rho * t, theta * t, nu * t);
    let left = scale(A).exp(None);
    let right = scale(B).exp(None);
    let both = left.compose(&right, None, None);
    let direct = scale(A + B).exp(None);
    assert!(
        both.is_approx(&direct, TOL),
        "SE23: exp(a)∘exp(b) != exp(a+b)"
    );
}

#[test]
fn subgroup_law_sim3() {
    let (rho, theta, sigma) = (
        Vector3::new(0.3, -0.5, 0.2),
        Vector3::new(0.4, 0.25, -0.6),
        0.3,
    );
    let scale = |t: f64| Sim3Tangent::new(rho * t, theta * t, sigma * t);
    let left = scale(A).exp(None);
    let right = scale(B).exp(None);
    let both = left.compose(&right, None, None);
    let direct = scale(A + B).exp(None);
    assert!(
        both.is_approx(&direct, TOL),
        "Sim3: exp(a)∘exp(b) != exp(a+b)"
    );
}

#[test]
fn subgroup_law_rn() {
    let xi = DVector::from_vec(vec![0.4, -0.7, 1.2, 0.05]);
    let scale = |t: f64| RnTangent::new(DVector::from_iterator(xi.len(), xi.iter().map(|v| v * t)));
    let left = scale(A).exp(None);
    let right = scale(B).exp(None);
    let both = left.compose(&right, None, None);
    let direct = scale(A + B).exp(None);
    assert!(
        both.is_approx(&direct, TOL),
        "Rn: exp(a)∘exp(b) != exp(a+b)"
    );
}

/// SGal(3): passes now that `exp` integrates the time–velocity coupling
/// `s·M(θ)·ν` (see `s_nu_coupling`). This test was the one that caught the
/// original defect (documented in `doc/claude_review/02-confirmed-critical.md`
/// §1) — keep it green.
#[test]
fn subgroup_law_sgal3() {
    let (rho, nu, theta, s) = (
        Vector3::new(0.3, -0.5, 0.2),
        Vector3::new(-0.35, 0.15, 0.45),
        Vector3::new(0.4, 0.25, -0.6),
        0.3,
    );
    let scale = |t: f64| SGal3Tangent::new(rho * t, nu * t, theta * t, s * t);
    let left = scale(A).exp(None);
    let right = scale(B).exp(None);
    let both = left.compose(&right, None, None);
    let direct = scale(A + B).exp(None);
    assert!(
        both.is_approx(&direct, TOL),
        "SGal3: exp(a)∘exp(b) != exp(a+b)"
    );
}

/// FD-validate the SGal(3) exp Jacobian on a tangent with a strong
/// time–velocity coupling (s·ν ≠ 0) — exactly the region where the old,
/// uncoupled exponential's Jacobian tables disagreed with the map itself.
#[test]
fn sgal3_exp_jacobian_matches_fd_with_coupling() {
    use apex_manifolds::sgal3::SGal3Tangent;
    use apex_manifolds::sgal3::Vector10;

    let xi = Vector10::from_iterator([0.4, -0.6, 0.25, 0.8, -0.3, 0.5, 0.35, -0.45, 0.2, 1.1]);
    let eps = 1e-6;

    let tangent = SGal3Tangent::new(
        Vector3::new(xi[0], xi[1], xi[2]),
        Vector3::new(xi[3], xi[4], xi[5]),
        Vector3::new(xi[6], xi[7], xi[8]),
        xi[9],
    );

    // The right Jacobian is *defined* by Exp(ξ+δ) ≈ Exp(ξ)∘Exp(Jr(ξ)·δ), so the
    // composition check below uses Jr itself. (This previously used Jr⁻¹, which
    // only held because `right_jacobian` was computing the wrong quantity —
    // inverting one error cancelled the other.)
    let jac = tangent.right_jacobian();

    let base = tangent.exp(None);
    for k in 0..10 {
        let mut step_p = xi;
        let mut step_m = xi;
        // one-parameter law: perturb the tangent, not the element
        step_p[k] += eps;
        step_m[k] -= eps;
        let tan_p = SGal3Tangent::new(
            Vector3::new(step_p[0], step_p[1], step_p[2]),
            Vector3::new(step_p[3], step_p[4], step_p[5]),
            Vector3::new(step_p[6], step_p[7], step_p[8]),
            step_p[9],
        );
        let _ = step_m;
        let ep = tan_p.exp(None);
        // First-order identity: Exp(ξ+δξ) ≈ Exp(ξ) ∘ Exp(Jr(ξ)·δξ).
        let col = jac.column(k);
        let expected_ep = base.compose(
            &SGal3Tangent::new(
                Vector3::new(col[0] * eps, col[1] * eps, col[2] * eps),
                Vector3::new(col[3] * eps, col[4] * eps, col[5] * eps),
                Vector3::new(col[6] * eps, col[7] * eps, col[8] * eps),
                col[9] * eps,
            )
            .exp(None),
            None,
            None,
        );
        assert!(
            ep.is_approx(&expected_ep, 1e-6),
            "SGal3 exp Jacobian col {k}: composition check failed"
        );
    }
}

#[test]
fn matrix10_inverse_is_supported() {
    use apex_manifolds::sgal3::Matrix10;
    let mut m = Matrix10::identity();
    m[(0, 1)] = 0.3;
    m[(3, 7)] = -0.5;
    let inv = m
        .try_inverse()
        .unwrap_or_else(|| panic!("Matrix10 must support try_inverse"));
    let product = m * inv;
    let mut max_err: f64 = 0.0;
    for i in 0..10 {
        for k in 0..10 {
            let target: f64 = if i == k { 1.0 } else { 0.0 };
            max_err = max_err.max((product[(i, k)] - target).abs());
        }
    }
    assert!(max_err < 1e-10, "inverse residual {max_err}");
}
