# apex-manifolds Crate Analysis

## Overview

The `apex-manifolds` crate provides Lie group implementations (SE2, SE3, SO2, SO3, Rn) for non-Euclidean optimization. It defines the `LieGroup` and `Tangent` traits and implements `plus()`, `minus()`, `compose()`, `inverse()`, and analytic Jacobians for each manifold type.

**Files analyzed:** `lib.rs`, `rn.rs`, `se2.rs`, `se3.rs`, `so2.rs`, `so3.rs`

---

## Performance Issues

### P1. Missing `#[inline]` on Accessor Methods (20+ sites)

Small, frequently-called accessor methods lack `#[inline]`, preventing the compiler from inlining across crate boundaries.

| File | Methods |
|------|---------|
| `so2.rs` (SO2Tangent) | `angle()`, `new()`, `zero()` |
| `so3.rs` (SO3Tangent) | `x()`, `y()`, `z()`, `coeffs()`, `axis()`, `angle()` |
| `se2.rs` (SE2Tangent) | `x()`, `y()`, `angle()`, `translation()`, `new()` |
| `se3.rs` (SE3/SE3Tangent) | `x()`, `y()`, `z()`, `translation()` |
| `rn.rs` (Rn) | `component()`, `dim()`, `norm()`, `norm_squared()` |

**Impact:** These are called millions of times during optimization. Without `#[inline]`, each call is a function call overhead across crate boundaries.

### P2. Unnecessary Intermediate Allocations in SE3 Jacobian

`se3.rs` — `SE3Tangent::q_block_jacobian_matrix()` creates 5 intermediate 3x3 matrices (`m1` through `m4` + result accumulation). These could be computed in-place with pre-allocated mutable buffers.

### P3. Clone in `Rn::to_vector()`

`rn.rs` — `self.data.clone()` in `to_vector()`. For Rn manifolds used as landmarks in bundle adjustment, this clone is called per-variable per-iteration. Consider returning a reference or using `Cow`.

### P4. Redundant Jacobian Computation in SE3 Compose

`se3.rs` — `other.inverse(None).adjoint()` in `compose()` creates an intermediate Jacobian allocation that could be fused.

---

## Code Quality Issues

### Q1. `Rn::identity()` Hardcoded to 3D

`rn.rs` — `identity()` always returns a 3-dimensional zero vector despite `Rn` being conceptually dynamic-dimensional. Same issue affects `RnTangent::zero()` and `RnTangent::random()`. Users working with non-3D Euclidean spaces (e.g., 2D landmarks) will get dimension mismatches.

**Related:** `Rn::DIM`, `Rn::DOF`, `Rn::REP_SIZE` are all `0` (sentinel for "dynamic"), but `identity()` ignores this and returns 3D.

### Q2. `assert_eq!` Instead of `Result` in `Rn::compose()`

`rn.rs` — Dimension mismatch in `compose()` triggers `assert_eq!` which **panics** instead of returning an error. This violates the project's own rules (`unwrap_used = "deny"` in workspace lints). Should return `ManifoldResult<Rn>`.

### Q3. Redundant SO3 Accessors

`so3.rs` — Both `quaternion()` and `to_quaternion()` exist and return the same thing. Redundant API surface that creates confusion about which to use.

### Q4. Inconsistent Naming Across Manifolds

| Concept | SE2 | SE3 | SO3 | Rn |
|---------|-----|-----|-----|----|
| Rotation access | `.angle()` | `.rotation_so3()` | `.quaternion()` | N/A |
| Component access | `.x()`, `.y()` | `.x()`, `.y()`, `.z()` | axis/angle | `.component()` |

The naming conventions differ enough to require remembering per-manifold conventions.

### Q5. Unused Imports

- `se3.rs`: `Matrix4` imported but only used in tests
- `so3.rs`: `Matrix4` imported but only used in tests

These should be moved inside `#[cfg(test)]` blocks.

---

## Readability & Maintainability

### R1. `SE3Tangent::q_block_jacobian_matrix()` — 55 Lines, High Complexity

`se3.rs` — This function computes the Q-block of the SE3 Jacobian with coefficients `a`, `b`, `c`, `d` in large-angle and small-angle branches. The formula references "Eq. 180" but the actual coefficient formulas are hard to verify against the reference.

**Recommendation:** Split into `compute_q_coefficients(theta: f64) -> (f64, f64, f64, f64)` and `assemble_q_matrix(coeffs, ...)`.

### R2. `SO3::log()` — Complex Conditional Logic

`so3.rs` — The `log()` implementation has nested if/else for small-angle, near-pi, and general cases with unclear variable naming. `sin_angle_squared` doesn't actually hold sin(angle)^2 — it holds the squared norm of the imaginary quaternion part.

### R3. Duplicated "Taylor vs Euler" Logic

`se2.rs` has 4 functions that each repeat the same pattern:
```
if theta_sq < f64::EPSILON { use_taylor_approximation } else { use_exact_formula }
```
This same pattern appears in `se3.rs` and `so3.rs`. Each copy has slightly different threshold handling.

### R4. Magic Numbers in Random Generation

- `se3.rs`: Random translation uses `0.1` range, rotation uses `0.2` range — no justification
- `so3.rs`: Random tangent uses `0.2 - 0.1` range — unexplained
- These should be named constants or parameters

---

## Redundancy / DRY Violations

### D1. Small-Angle Approximation Logic (Repeated 12+ Times)

The pattern `if theta_sq < EPSILON { taylor_approx } else { exact }` is repeated across:
- `se2.rs`: `right_jacobian()`, `left_jacobian()`, `right_jacobian_inverse()`, `left_jacobian_inverse()`
- `se3.rs`: Same 4 functions
- `so3.rs`: `exp()`, `log()`, right/left Jacobians

Each instance uses slightly different threshold values and coefficient formulas, making it error-prone.

**Recommendation:** Extract `fn angle_coefficients(theta_sq: f64) -> AngleCoeffs` shared utility.

### D2. Test Structure Duplication

All manifold files have nearly identical test patterns:
- `test_*_exp_log_consistency()`
- `test_*_compose()`
- `test_*_random()`
- `test_*_jacobian_dimensions()`

These could use a shared test macro `manifold_test_suite!` that generates common tests.

### D3. `normalize()` Reimplemented per Manifold

Each manifold has its own `normalize()` with similar structure but different thresholds (`f64::EPSILON` in some, `1e-12` in Rn).

---

## Safety & Numerical Stability

### S1. Inconsistent Epsilon Thresholds

| File | Context | Threshold |
|------|---------|-----------|
| `se2.rs` | Small-angle check | `f64::EPSILON` (~2.2e-16) |
| `se3.rs` | Small-angle check | `f64::EPSILON` |
| `so3.rs` | Small-angle check | `f64::EPSILON` |
| `rn.rs` | Normalize check | `1e-12` |
| `so2.rs` | Normalize check | `f64::EPSILON` |

Using `f64::EPSILON` for small-angle detection is extremely tight — angles of ~1e-8 radians may fall through. A threshold of `1e-10` or `1e-12` is more robust for Taylor expansion switching.

### S2. SO3 Angle Wrapping

`so3.rs` — `log()` uses `atan2(-sin_angle, -cos_angle)` for the negative cosine case. This wraps the angle to a potentially unexpected range. The implementation should be verified against the reference (manif library).

### S3. No Quaternion Normalization Validation

`so3.rs` — `log()` and `compose()` assume the internal quaternion is normalized but never validate this invariant. If a quaternion drifts due to numerical accumulation, results will silently degrade.

### S4. No Input Validation in SE3 Constructors

`se3.rs` — `compose()` and `From<DVector>` don't validate that the rotation component is a valid unit quaternion. External data (from file I/O) could introduce denormalized quaternions.

---

## API Design

### A1. `LieGroup::act()` Forces 3D

The `act()` method in the `LieGroup` trait requires `Vector3<f64>`, making it impossible for `Rn` to act on arbitrary-dimensional vectors. This is a fundamental limitation for the dynamic-dimensional manifold.

### A2. Missing Factory Methods

- `SE2Tangent` has no `from_components(x, y, angle)` factory (SE3Tangent does have one)
- `Rn` has no `with_dimension(n)` constructor for creating zero/identity of specific size
- `SO3::from_quaternion_coeffs(x, y, z, w)` uses unusual parameter order (most conventions use `w` first)

### A3. `Tangent::DIM` Is Zero for Dynamic Types

`RnTangent::DIM = 0` as a sentinel for "dynamic" is confusing and could cause bugs if used in array sizing. Consider using `Option<usize>` or a separate `is_dynamic()` method.

---

## Testing Gaps

### T1. No Jacobian Numerical Verification Tests

None of the manifolds verify that analytic Jacobians match finite-difference Jacobians. This is the most important missing test category — incorrect Jacobians cause silent optimization failures.

**Missing tests:**
- `right_jacobian()` vs numerical differentiation
- `left_jacobian()` vs numerical differentiation
- `lplus_jacobian()` / `rminus_jacobian()` vs numerical differentiation
- `q_block_jacobian_matrix()` correctness verification

### T2. Missing Edge Cases

- **SO3:** No test for quaternion antipodal equivalence (`q` and `-q` represent the same rotation)
- **SE2:** No test for angle wrap-around (angle > 2pi)
- **SE3:** No test for very large translations + small rotations
- **Rn:** No test for dimension mismatch (2D vs 3D composition)

### T3. No Accumulated Error Tests

No tests for numerical drift over composition chains (e.g., compose 1000 small rotations and verify result).

### T4. Missing Jacobian Inverse Identity Tests

`Jr * Jr_inv` should approximate the identity matrix, but this relationship is not tested.

---

## Prioritized Recommendations

### Critical
1. **Add Jacobian numerical verification tests** — incorrect Jacobians silently break optimization
2. **Fix Rn dimension design** — hardcoded 3D defaults cause runtime panics for non-3D usage
3. **Replace `assert_eq!` with `Result`** in `Rn::compose()` — panics violate workspace lint rules

### High
4. **Add `#[inline]` to 20+ accessor methods** — measurable performance impact in optimization loops
5. **Standardize epsilon thresholds** — create `const SMALL_ANGLE_THRESHOLD: f64 = 1e-10` shared across all manifolds
6. **Verify SO3 log() angle wrapping** against reference implementation

### Medium
7. **Extract `angle_coefficients()` helper** to eliminate 12+ duplicated small-angle patterns
8. **Split `q_block_jacobian_matrix()`** into coefficient computation + matrix assembly
9. **Add edge case tests** (antipodal quaternions, angle wrapping, dimension mismatches)
10. **Remove redundant `to_quaternion()` alias** in SO3

### Low
11. **Create `manifold_test_suite!` macro** for shared test patterns
12. **Move test-only imports** (`Matrix4`) inside `#[cfg(test)]` blocks
13. **Document random generation ranges** with named constants
