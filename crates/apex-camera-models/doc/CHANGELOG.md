# Changelog

All notable changes to `apex-camera-models` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-29

First release prepared for publishing to crates.io. This version promotes the crate from an
internal workspace dependency to a standalone publishable library, and consolidates the
improvements made since v0.1.0: a tenth camera model, analytic pose Jacobians, centralized
parameter validation, a world-to-camera coordinate convention alignment, and a full unit
test suite across all models.

### Added

- **`FThetaCamera`** — NVIDIA f-theta fisheye camera model for automotive surround-view
  and robotics wide-angle use cases:
  - Supports field-of-view up to 220°
  - Polynomial projection: r = k₁·θ + k₂·θ³ + k₃·θ⁵ + k₄·θ⁷ with isotropic intrinsics
  - Six parameters: cx, cy, k₁, k₂, k₃, k₄
  - `FTheta` variant added to `DistortionModel` enum
  - Analytic `jacobian_intrinsics` and `jacobian_point` implementations
  - Based on the NVIDIA DriveWorks f-theta specification

- **`jacobian_pose`** on `CameraModel` trait — analytic ∂(u,v)/∂ξ Jacobian with respect
  to a SE(3) pose perturbation in the Lie algebra:
  - Returns `(PointJacobian, SMatrix<f64, 3, 6>)` — the 2×6 pixel Jacobian and the
    intermediate 3×6 point-in-camera Jacobian for point-cloud applications
  - Implemented for all ten camera models
  - Replaces the previously removed stub

- **`CameraModelError` variants**:
  - `ParameterOutOfRange { param, value, min, max }` — structured replacement for ad-hoc
    `InvalidParams` strings for range-checked coefficients (UCM α, Double Sphere ξ and α)
  - `ProjectionOutOfBounds` — raised when a projected pixel falls outside the image sensor

- **Comprehensive unit test suite** covering all ten camera models:
  - Round-trip projection / unprojection with sub-pixel tolerance
  - Numerical Jacobian verification (finite-difference vs. analytic) for all three Jacobians
    (`jacobian_point`, `jacobian_pose`, `jacobian_intrinsics`)
  - Parameter validation edge cases (zero focal length, out-of-range distortion coefficients)
  - Batch projection consistency (`project_batch` matches individual `project` results)

### Changed

- **World-to-camera coordinate convention** aligned consistently across all models — the
  input point `p_cam` is expected in the camera frame (Z-forward). Previously some models
  implicitly assumed a different frame origin. All methods now document this convention.

- **`TryFrom<&[f64]>` replaces `From<&[f64]>`** for all camera model structs — parameter
  slice construction is now fallible and returns `CameraModelError` on invalid inputs
  rather than panicking. The `try_from_params(&[f64])` associated function mirrors this.

- **Centralized parameter validation** — each model's `validate_params()` delegates to
  `PinholeParams::validate()` and `DistortionModel::validate()` rather than duplicating
  range checks. Validation is called automatically from all constructors.

- **`unwrap()` eliminated from all test code** — assertions use `Result` propagation with
  `?`; no panicking unwraps remain in the test suite.

- **Batch projection loop optimized** — `project_batch` now iterates column-wise over the
  input `Matrix3xX` to exploit nalgebra's column-major storage layout.

- Version bumped from `0.1.0` to `0.2.0` in `Cargo.toml`
- Workspace `Cargo.toml` dependency updated to `apex-camera-models = "0.2.0"`
- README expanded with an updated model comparison table, per-model academic references,
  and cargo publish instructions

### Removed

- **`OptimizeParams<const N: usize>`** compile-time generic — the experimental const-generic
  parameterization for enabling/disabling intrinsic optimization was removed in favour of a
  simpler runtime approach that does not appear in the public type signature.

- **`is_valid_point` free functions** — replaced by the `validate_point_in_front(z)`
  utility and the structured `CameraModelError::PointBehindCamera` variant.

---

## [0.1.0] - 2026-01-30

Initial creation of the `apex-camera-models` crate as part of the `apex-solver` workspace
restructuring. Extracted as an independent, reusable camera projection library so that
downstream crates (bundle adjustment, visual odometry, sensor fusion) can depend on camera
models without pulling in the full solver.

### Added

- **`CameraModel` trait** — unified interface implemented by all projection models:
  - `project(p_cam) -> Result<Vector2<f64>, CameraModelError>` — 3D-to-2D projection
  - `unproject(point_2d) -> Result<Vector3<f64>, CameraModelError>` — 2D-to-3D bearing ray
  - `jacobian_point(p_cam) -> PointJacobian` — analytic ∂(u,v)/∂(x,y,z) (2×3)
  - `jacobian_intrinsics(p_cam) -> IntrinsicJacobian` — analytic ∂(u,v)/∂(intrinsic params)
  - `project_batch(points_cam) -> Matrix2xX<f64>` — vectorized batch projection
  - `validate_params() -> Result<(), CameraModelError>` — parameter self-check
  - `get_pinhole_params() -> PinholeParams` — extract focal length and principal point
  - `get_distortion() -> DistortionModel` — retrieve the distortion variant
  - `get_model_name() -> &'static str` — model identifier string

- **Nine camera projection models**:

  - **`PinholeCamera`** (4 params: fx, fy, cx, cy) — standard perspective projection with
    no distortion. Baseline model; all other models reduce to this in the limit of zero
    distortion coefficients.

  - **`RadTanCamera`** (9 params: fx, fy, cx, cy, k₁, k₂, p₁, p₂, k₃) — Brown–Conrady
    (OpenCV) radial-tangential distortion. Compatible with `cv::calibrateCamera` output.
    Newton solver for iterative unprojection.

  - **`KannalaBrandtCamera`** (8 params: fx, fy, cx, cy, k₁, k₂, k₃, k₄) — polynomial
    fisheye model (Kannala & Brandt, PAMI 2006). r = θ + k₁θ³ + k₂θ⁵ + k₃θ⁷ + k₄θ⁹.
    Supports up to ~180° FOV; used for GoPro-style and automotive fisheye cameras.

  - **`FovCamera`** (5 params: fx, fy, cx, cy, w) — field-of-view model (Devernay &
    Faugeras, MVA 2001). Single parameter w ∈ (0, π). Closed-form atan-based
    projection/unprojection; suitable for SLAM with wide-angle lenses.

  - **`UcmCamera`** (5 params: fx, fy, cx, cy, α) — Unified Camera Model (Geyer &
    Daniilidis, ECCV 2000; Mei & Rives, ICRA 2007). Sphere–plane projection; α ∈ [0, 1].
    Handles FOV > 90° including catadioptric (mirror + lens) systems.

  - **`EucmCamera`** (6 params: fx, fy, cx, cy, α, β) — Extended Unified Camera Model
    (Khomutenko et al., LRA 2016). Second sphere parameter β > 0 improves accuracy on
    extreme-distortion fisheye lenses with FOV > 180°.

  - **`DoubleSphereCamera`** (6 params: fx, fy, cx, cy, ξ, α) — Double Sphere model
    (Usenko et al., 3DV 2018). Two-sphere projection; ξ ∈ [−1, 1], α ∈ [0, 1]. Achieves
    the highest accuracy on extreme-FOV lenses among closed-form models.

  - **`BALPinholeCameraStrict`** (3 params: f, k₁, k₂) — Snavely's BAL dataset camera
    format. Single focal length, no principal point offset, two radial distortion
    coefficients. Projects in the −z convention used by the BAL benchmark datasets.

  - **`BALPinholeCamera`** (6 params: fx, fy, cx, cy, k₁, k₂) — extended BAL variant with
    separate horizontal and vertical focal lengths and a non-zero principal point. Used when
    re-calibrating BAL scenes with a standard calibration tool.

- **`PinholeParams`** struct — type-safe (fx, fy, cx, cy) storage:
  - `new(fx, fy, cx, cy) -> Result<Self, CameraModelError>` — validated constructor
  - `validate() -> Result<(), CameraModelError>` — reusable validation for focal lengths
    (positive, finite) and principal point (finite)

- **`DistortionModel`** enum — one variant per model carrying all distortion coefficients:
  `None`, `Radial { k1, k2 }`, `BrownConrady { k1, k2, p1, p2, k3 }`,
  `KannalaBrandt { k1, k2, k3, k4 }`, `FOV { w }`, `UCM { alpha }`,
  `EUCM { alpha, beta }`, `DoubleSphere { xi, alpha }`. Each variant implements
  `validate() -> Result<(), CameraModelError>`.

- **`CameraModelError`** — structured error type (via `thiserror`) with typed variants
  that include the offending parameter values for diagnostics:
  `FocalLengthNotPositive { fx, fy }`, `FocalLengthNotFinite { fx, fy }`,
  `PrincipalPointNotFinite { cx, cy }`, `DistortionNotFinite { name, value }`,
  `PointBehindCamera { z, min_z }`, `PointAtCameraCenter`, `DenominatorTooSmall`,
  `NumericalError { operation, details }`, `InvalidParams(String)`

- **Constants**: `GEOMETRIC_PRECISION` (1e-6), `NUMERICAL_DERIVATIVE_EPS` (1e-7),
  `MIN_DEPTH` (1e-6), `CONVERGENCE_THRESHOLD` (1e-6), `JACOBIAN_TEST_TOLERANCE` (1e-5),
  `PROJECTION_TEST_TOLERANCE` (1e-10)

---

*For the top-level apex-solver workspace changelog see [../../doc/CHANGELOG.md](../../doc/CHANGELOG.md)*
