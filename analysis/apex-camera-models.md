# apex-camera-models Crate Analysis

## Overview

The `apex-camera-models` crate provides 8 camera projection models for bundle adjustment and SfM: Pinhole, Radial-Tangential, Kannala-Brandt, FOV, UCM, EUCM, Double Sphere, and BAL Pinhole. Each model implements the `CameraModel` trait with `project()`, `unproject()`, analytic Jacobians, and `linear_estimation()`.

**Files analyzed:** `lib.rs`, `pinhole.rs`, `rad_tan.rs`, `kannala_brandt.rs`, `fov.rs`, `ucm.rs`, `eucm.rs`, `double_sphere.rs`, `bal_pinhole.rs`

---

## Performance Issues

### P1. Missing `#[inline]` on Hot-Path Utility

`lib.rs` — `skew_symmetric()` is a small 3x3 matrix construction called in every camera model's `jacobian_pose()`. Without `#[inline]`, it's a cross-crate function call at 9+ call sites in tight optimization loops.

### P2. Repeated `distortion_params()` Extraction

All camera models call `self.distortion_params()` (which pattern-matches on the `DistortionModel` enum) multiple times within the same function. For example, `rad_tan.rs` calls it 3 times in its Jacobian computation.

**Fix:** Cache the result once at the top of each function:
```rust
let (k1, k2, p1, p2, k3) = self.distortion_params();
```

### P3. Redundant Normalization in Unprojection

`fov.rs`, `ucm.rs`, `eucm.rs`, `double_sphere.rs` — All call `.normalize()` at the end of `unproject()`, which internally computes `.norm()`. In many cases, the norm was already computed earlier in the function but discarded. Reusing the pre-computed norm would save a square root per call.

### P4. Double Square Root in Double Sphere

`double_sphere.rs` (line ~562) — `r2 = x*x + y*y` and `d1 = (r2 + z*z).sqrt()` are computed, then `r2` is reused for `d2`. The intermediate `r2` could be more aggressively reused to avoid recomputation in the Jacobian path.

---

## Code Quality Issues

### Q1. `jacobian_pose()` Duplicated Across 8 Files (CRITICAL)

The **exact same** ~90-line `jacobian_pose()` implementation is copy-pasted across all 8 camera model files:

| File | Approximate Lines |
|------|-------------------|
| `pinhole.rs` | 315-405 |
| `rad_tan.rs` | 596-686 |
| `kannala_brandt.rs` | 773-863 |
| `fov.rs` | 449-539 |
| `ucm.rs` | 441-531 |
| `eucm.rs` | 495-585 |
| `double_sphere.rs` | 783-873 |
| `bal_pinhole.rs` | 370-460 |

**Total duplicated lines: ~720**

This is the single largest DRY violation in the entire codebase. The `jacobian_pose()` depends only on `skew_symmetric()` and the `project()` + `jacobian_point()` methods, which are already part of the `CameraModel` trait.

**Fix:** Implement as a default trait method in `CameraModel`:
```rust
trait CameraModel {
    fn jacobian_pose(&self, p_world: &Vector3<f64>, pose: &SE3) -> ... {
        // Default implementation using self.project() and self.jacobian_point()
    }
}
```

### Q2. `From<&[f64]>` Creates Invalid Cameras Without Validation

All models implement `From<&[f64]>` and `From<[f64; N]>` without calling `validate_params()`:

```rust
// rad_tan.rs line 269 — Can create camera with negative focal length
impl From<&[f64]> for RadTanCamera {
    fn from(params: &[f64]) -> Self {
        Self { pinhole: PinholeParams::new(params[0], ...), ... }
    }
}
```

Users who construct cameras via `From` conversion get no validation. Invalid cameras will silently produce garbage projections.

**Fix:** Either validate in `From` or rename to `from_unchecked()` and provide a validating constructor.

### Q3. `linear_estimation()` Methods Completely Untested

Six camera models have `linear_estimation()` methods (for estimating intrinsics from point correspondences) with **zero test coverage**:
- `rad_tan.rs`
- `kannala_brandt.rs`
- `fov.rs`
- `ucm.rs`
- `eucm.rs`
- `double_sphere.rs`

These contain SVD-based solvers and numerical conditioning logic that is entirely unverified.

### Q4. Inconsistent Convergence Thresholds

Different models use different constants for the same purpose:

| Model | Convergence Check | Constant Used |
|-------|-------------------|---------------|
| `kannala_brandt.rs` | Unprojection loop | `crate::CONVERGENCE_THRESHOLD` |
| `rad_tan.rs` | Unprojection loop | `crate::GEOMETRIC_PRECISION` |
| `double_sphere.rs` | Denominator check | `crate::GEOMETRIC_PRECISION` |

The same concept (numerical precision threshold) has two different names and potentially different values.

### Q5. Undocumented Clamping in Unprojection

`kannala_brandt.rs` (line ~492) — `ru = ru.min(std::f64::consts::PI / 2.0)` silently clamps the result without any user feedback or documentation explaining why PI/2 is the limit.

`fov.rs` (line ~491) — Similar silent clamping occurs.

---

## Readability & Maintainability

### R1. Verbose Jacobian Derivation Comments (150-200 Lines Each)

`rad_tan.rs` has 150+ lines of mathematical derivation in `jacobian_point()` comments. `kannala_brandt.rs` has 200+ lines. While mathematically rigorous, these make navigation difficult.

**Recommendation:** Move detailed derivations to `doc/camera_jacobians.md` and keep inline comments concise with references.

### R2. Repeated Test Setup

Every test file repeats camera construction:
```rust
let camera = RadTanCamera::new(
    PinholeParams::new(fx, fy, cx, cy),
    DistortionModel::RadTan { k1, k2, p1, p2, k3 },
);
```

A shared `test_utils` module with factory functions would reduce boilerplate across 8 files.

### R3. Nearly Identical Validation Patterns

All models validate `fx > 0`, `fy > 0`, `is_finite()` with only minor variations for distortion parameter ranges. This ~30-line pattern is repeated 8 times.

**Fix:** Extract common validation into `PinholeParams::validate()` and model-specific validation into per-model methods.

### R4. Nearly Identical `From` Trait Implementations

All models implement `From<&[f64]>` and `From<[f64; N]>` with the same pattern — only the parameter count and field mapping differ. A macro could generate these.

---

## Redundancy / DRY Violations

### D1. `jacobian_pose()` Duplicated 8 Times (~720 Lines) — See Q1

### D2. Validation Logic Duplicated 8 Times (~240 Lines)

Each model's `validate_params()` repeats:
```rust
if !self.pinhole.fx.is_finite() || self.pinhole.fx <= 0.0 { return Err(...) }
if !self.pinhole.fy.is_finite() || self.pinhole.fy <= 0.0 { return Err(...) }
if !self.pinhole.cx.is_finite() { return Err(...) }
if !self.pinhole.cy.is_finite() { return Err(...) }
// Then model-specific checks
```

### D3. `From` Trait Implementations Duplicated 8 Times (~160 Lines)

Same conversion pattern with only field counts varying.

### D4. Test Assertion Patterns Duplicated

All projection tests check the same conditions with the same tolerance (`1e-10`). All Jacobian tests use the same `NUMERICAL_DERIVATIVE_EPS` pattern.

**Total estimated redundancy: ~1,200 lines** that could be reduced to ~200 lines with proper trait defaults, macros, and shared utilities.

---

## Safety & Numerical Stability

### S1. Division by Zero in Jacobian Calculations

- `kannala_brandt.rs` (line ~569): `let inv_r = 1.0 / r;` — `r` is checked earlier with early return, but the check and usage are separated by many lines, making the safety fragile
- `fov.rs` (line ~377): Denominator `z*z + mul2tanwby2*mul2tanwby2*r*r` could be zero if both `z` and `r` are very small
- `double_sphere.rs` (line ~592): `denom < GEOMETRIC_PRECISION` check uses 1e-6, which may be too large for a denominator guard

### S2. No Input Validation on 3D Points

All `project()` methods assume valid input coordinates. No model validates `!p_cam.x.is_nan()` or `!p_cam.z.is_finite()` before computation. NaN input propagates silently through the pipeline.

### S3. Iterative Unprojection Failure Handling

`rad_tan.rs` (line ~435) — When unprojection fails to converge after `MAX_ITERATIONS`, it returns an error rather than the best-effort result. In optimization contexts, a "best effort with high residual" is often more useful than an error.

### S4. No Overflow Protection

Large coordinates (e.g., `z = 1e300`) passed to `project()` could cause overflow in `x/z`, `x*x + y*y`, etc. No models guard against extreme input values.

---

## API Design

### A1. Missing `unproject_batch()` in Trait

The `CameraModel` trait has `project_batch()` (lib.rs line ~485) but no corresponding `unproject_batch()`. Unprojection is common in initialization pipelines.

### A2. Missing Typed Distortion Accessors

`get_distortion()` returns a `DistortionModel` enum, requiring pattern matching:
```rust
match camera.get_distortion() {
    DistortionModel::RadTan { k1, k2, p1, p2, k3 } => { ... }
    _ => panic!("wrong type"),
}
```

Per-model typed accessors (`get_k1()`, `get_k2()`) would be more ergonomic.

### A3. No "When to Use" Guidance

No documentation explains when to choose one model over another. Users must research externally to decide between Double Sphere vs UCM vs Kannala-Brandt for their fisheye lens.

### A4. Inconsistent `inv_z` Naming

- `pinhole.rs`: `inv_z = 1.0 / p_cam.z` (standard Z-forward)
- `bal_pinhole.rs`: `inv_neg_z = -1.0 / p_cam.z` (-Z forward convention)

The naming is technically correct but the convention difference should be more prominent.

---

## Testing Gaps

### T1. Missing `jacobian_pose()` Numerical Tests (CRITICAL)

All models have numerical tests for `jacobian_point()` and `jacobian_intrinsics()`, but **none** test `jacobian_pose()` against finite differences. This is the most commonly used Jacobian in SLAM/VO pipelines. An incorrect pose Jacobian would silently break all pose optimization.

### T2. `linear_estimation()` Completely Untested (6 Models)

Six complex SVD-based estimation methods with no test coverage. Degenerate inputs (collinear points, insufficient correspondences) are entirely unverified.

### T3. Missing Round-Trip Projection Tests

No model systematically tests: `project(p) -> pixel -> unproject(pixel) -> ray -> verify alignment with original p`. This is the fundamental correctness invariant for camera models.

### T4. Missing Edge Case Tests

- Points exactly at `z == MIN_DEPTH` boundary
- Very large focal lengths (numerical stability)
- Points on the optical axis (`x = 0, y = 0`)
- Points at extreme field-of-view angles
- Negative principal points (unusual but valid)

### T5. No `From`/`Into` Round-Trip Validation Tests

Converting camera to parameter vector and back doesn't verify that `validate_params()` passes on the reconstructed camera. This could silently mask validation bugs.

### T6. Jacobian Tests Use Loose Tolerance

`JACOBIAN_TEST_TOLERANCE = 1e-5` — For double precision derivatives, 1e-7 or 1e-8 would catch more bugs. The current tolerance may pass even with significant Jacobian errors.

### T7. No NaN/Inf Output Verification in Jacobian Tests

Tests compute Jacobians but never verify the output is `is_finite()`. Edge-case inputs could produce NaN Jacobians that pass tolerance checks (NaN comparisons are always false).

---

## Prioritized Recommendations

### Critical
1. **Extract `jacobian_pose()` as trait default method** — eliminates 720 lines of duplication and ensures consistency
2. **Add `jacobian_pose()` numerical tests** — untested in all 8 models, critical for optimization correctness
3. **Add validation to `From` conversions** — or rename to `from_unchecked()` to signal the lack of validation

### High
4. **Add `linear_estimation()` tests** for all 6 models — complex numerical code with zero coverage
5. **Add round-trip projection tests** — fundamental correctness invariant
6. **Standardize convergence thresholds** — use a single named constant across all models
7. **Guard against division by zero** in Jacobian calculations with explicit checks

### Medium
8. **Extract common validation** into `PinholeParams::validate()` + per-model extension
9. **Add `unproject_batch()`** to trait for API completeness
10. **Tighten Jacobian test tolerance** from 1e-5 to 1e-7
11. **Move verbose derivation comments** to external documentation
12. **Add NaN/Inf checks** in Jacobian test assertions

### Low
13. **Create macro for `From` implementations** — reduce 160 lines of boilerplate
14. **Add "model selection guide"** documentation
15. **Create shared test factory functions** to reduce test setup duplication
16. **Add `#[inline]` to `skew_symmetric()`** and `check_projection_condition()`
