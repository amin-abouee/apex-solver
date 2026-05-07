# Changelog

All notable changes to `apex-manifolds` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-29

First release prepared for publishing to crates.io. This version promotes the crate from an
internal workspace dependency to a standalone publishable library, and consolidates all
improvements made since v0.1.0: three new higher-dimensional manifolds, coordinate convention
alignment, Jacobian correctness fixes, numerical stability improvements, and a complete unit
test suite.

### Added

- **`SE_2(3)`** (`se23` module) — Extended special Euclidean group encoding rotation,
  translation, and velocity (9 DOF, 10-scalar representation). Natural state manifold for
  IMU preintegration in visual-inertial odometry. Based on Barrau & Bonnabel (2017).

- **`SGal(3)`** (`sgal3` module) — Special Galilean group extending SE_2(3) with a scalar
  time parameter (10 DOF, 11-scalar representation). Used in inertial navigation systems
  where time is part of the optimization state.

- **`Sim(3)`** (`sim3` module) — Similarity transformation group extending SE(3) with a
  positive scale factor λ (7 DOF, 8-scalar representation). Group action is λRx + t.
  Intended for monocular SLAM and structure-from-motion where metric scale is unobservable.

- **`ManifoldType`** enum — variants `SE2`, `SE3`, `SO2`, `SO3`, `SE23`, `SGal3`, `Sim3`,
  `RN`; used by the solver to dispatch variable construction at graph-build time.

- **`RnTangent`** `From<DVector<f64>>` / `Into<DVector<f64>>` for ergonomic conversion
  between the Euclidean tangent type and nalgebra dynamic vectors.

- **Unit test suite** for all eight manifolds:
  - Identity, composition, inverse, and `between` correctness
  - Round-trip `exp ∘ log` and `log ∘ exp` consistency
  - Numerical Jacobian verification (finite-difference vs. analytic) for `plus`, `minus`,
    `compose`, `act`, and `inverse`
  - `ManifoldError` display, clone, and equality

### Changed

- **SO(3) quaternion convention** aligned to w-first (Hamilton) `[qw, qx, qy, qz]` —
  was previously inconsistent between construction and serialization paths.
- **SO(3) Jacobian inverse** numerical stability improved near θ = 0 and θ = π.
- **SE(3) Q-matrix** sign error in `right_minus` Jacobian block corrected.
- **Sim(3) Jacobian and V-matrix** computations refactored; safety docs added for the
  inverse Jacobian near degenerate scale values.
- **SGal(3) tangent space adjoint** representation fixed.
- Module `se_2_3` renamed to `se23` for naming consistency.
- Version bumped from `0.1.0` to `0.2.0` in `Cargo.toml`.
- Workspace `Cargo.toml` dependency updated to `apex-manifolds = "0.2.0"`.
- README updated with new manifold table, usage examples for SE_2(3) / SGal(3) / Sim(3),
  and additional academic references.

---

## [0.1.0] - 2026-01-30

Initial creation of the `apex-manifolds` crate as part of the `apex-solver` workspace
restructuring ([apex-solver v1.1.0](../../doc/CHANGELOG.md#110---2026-02-21)). Extracted
from the monolithic `apex-solver` crate to be independently publishable and reusable by
downstream robotics and computer vision crates.

### Added

- **`LieGroup` trait** — core interface for all manifold types:
  - `identity()`, `random()`, `inverse()`, `compose()`, `log()`, `act()`
  - `plus()` / `minus()` (right perturbation, default)
  - `left_plus()` / `left_minus()` (left perturbation)
  - `between()`, `adjoint()`, `normalize()`, `is_valid()`
  - All operations accept optional `&mut JacobianMatrix` for analytic derivatives

- **`Tangent` trait** — Lie algebra operations:
  - `exp()`, `hat()`, `small_adj()`, `lie_bracket()`, `generator()`
  - `right_jacobian()`, `left_jacobian()`, `right_jacobian_inv()`, `left_jacobian_inv()`

- **`Interpolatable` trait** — `interp()` and `slerp()` for smooth manifold interpolation.

- **Five manifold implementations**:
  - **`SE3`** — 3D rigid transformations (6 DOF, quaternion + translation)
  - **`SO3`** — 3D rotations (3 DOF, unit quaternion)
  - **`SE2`** — 2D rigid transformations (3 DOF, angle + translation)
  - **`SO2`** — 2D rotations (1 DOF, unit complex number)
  - **`Rn`** — Euclidean vector space with dynamic dimension (n DOF)

- **`ManifoldError`** — structured error type: `InvalidTangentDimension`,
  `NumericalInstability`, `InvalidElement`, `DimensionMismatch`, `InvalidNumber`,
  `NormalizationFailed`.

- **`SMALL_ANGLE_THRESHOLD`** (1e-10) — guards Taylor series approximations in all
  trigonometric operations to prevent division-by-zero near θ = 0.

---

*For the top-level apex-solver workspace changelog see [../../doc/CHANGELOG.md](../../doc/CHANGELOG.md)*
