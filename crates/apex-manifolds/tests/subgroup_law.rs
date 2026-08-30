//! The one-parameter subgroup law: `exp(a·ξ) ∘ exp(b·ξ) == exp((a+b)·ξ)`.
//!
//! This is the property that ties `exp` to `compose`. It is independent of
//! `log` (no round-trip escape hatch), so an `exp` that is merely a
//! retraction — smooth, exp(0)=I, first-order correct — passes every
//! round-trip and FD-Jacobian test while still failing here.
//!
//! Reference: the defect that motivated this suite (SGal(3)) is documented in
//! `doc/claude_review/02-confirmed-critical.md` §1.

use apex_manifolds::rn::{Rn, RnTangent};
use apex_manifolds::se2::SE2;
use apex_manifolds::se2::SE2Tangent;
use apex_manifolds::se3::SE3;
use apex_manifolds::se3::SE3Tangent;
use apex_manifolds::se23::SE23;
use apex_manifolds::se23::SE23Tangent;
use apex_manifolds::sgal3::SGal3;
use apex_manifolds::sgal3::SGal3Tangent;
use apex_manifolds::sim3::Sim3;
use apex_manifolds::sim3::Sim3Tangent;
use apex_manifolds::so2::SO2;
use apex_manifolds::so2::SO2Tangent;
use apex_manifolds::so3::SO3;
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

/// SGal(3) currently FAILS this law: its `exp` has no time-velocity coupling
/// while its `compose` does. The discrepancy is exactly a·b·s·ν (see
/// `doc/claude_review/02-confirmed-critical.md` §1). Ignored until the true
/// exponential lands; remove this attribute together with that fix.
#[test]
#[ignore = "SGal(3) exp is a retraction, not the group exponential (missing s·ν coupling)"]
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
