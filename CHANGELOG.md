# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking Changes

- **`apex-io`: `rosbag` is now an opt-in feature (was built unconditionally).**
  Depending on `apex-io` no longer compiles `rusqlite` (bundled SQLite),
  `mcap`, `zstd`, `lz4_flex`, `serde_yaml`, `byteorder`, `hex`. Bag I/O users
  must opt in:
  ```toml
  apex-io = { version = "0.3", features = ["rosbag"] }
  # via the solver crate:
  apex-solver = { version = "1.4", features = ["rosbag"] }
  ```
  (Use the newest 0.x / 1.x release; the `rosbag` feature name is stable.)
  `download` (dataset fetching via `ureq`/`bzip2`/`flate2`/`tar`) stays on by
  default; `--no-default-features` disables it, in which case the
  `ensure_*_dataset` helpers only serve already-downloaded files.
  `clap` is now behind the default-on `cli` feature in both crates
  (bins requiring it declare `required-features`).
  The `bag_*` binaries now require `--features rosbag`, `download_datasets`
  requires `--features download`.
- **`apex-solver` bins/examples moved behind the default-on `cli` feature.**
  `--no-default-features` builds the library without `clap`; the
  `pose_graph_g2o` / `bundle_adjustment` binaries and the `clap`-based
  examples are skipped in that configuration.

### Changed

- **Changelog moved from `doc/CHANGELOG.md` to the repository root**
  (`CHANGELOG.md`), following Keep-a-Changelog conventions. Each sub-crate's
  changelog moved likewise from `crates/*/doc/CHANGELOG.md` to its crate root
  (`crates/*/CHANGELOG.md`).
- **Reference PDFs relocated** from `doc/*.pdf` to `doc/references/`.

### Added

- **Factor library expansion** (GTSAM-audited; see `doc/factor-catalog.md` for the full
  keep/skip rationale):
  - **IMU**: model-suffixed files (`imu_se23_factors.rs`, `combined_imu_se23_factors.rs`,
    `imu_sgal3_factors.rs`, `combined_imu_sgal3_factors.rs`) plus `Sgal3ImuFactor` /
    `Sgal3CombinedImuFactor` — SGal(3)-based kinematic constraint over the same shared
    `ImuPreintegration` (SGal(3) increment = SE(2)₃ delta + time coordinate; the time row
    of the log-residual vanishes identically). `ImuPreintegration::delta_sgal3()` added.
    The SGal(3) FD-Jacobian tests are currently `#[ignore]`d (tangent Jacobian chain under
    investigation; the zero-residual and group-composition formulation tests pass).
  - **Visual**: `StereoFactor` (rectified stereo), `InverseDepthFactor` (anchor pixel +
    inverse-depth landmarks), `EssentialMatrixFactor` / `EssentialMatrixConstraint`
    (2D–2D epipolar geometry), and `SmartProjectionFactor` — pose-only structure-less
    multi-view factor with DLT + Gauss-Newton re-triangulation and the exact implicit-Schur
    (point-marginalized) Jacobian; registers with `NoiseModel::null()` (internal whitening)
    and falls back to a bounded cheirality penalty on degeneracy.
  - **LiDAR**: `PoseToPointFactor`, `PointToPlaneFactor` (target plane, no distance field),
    and `GicpFactor` (plane-to-plane via combined-covariance whitening, frozen at
    construction with a rotation hint).
  - **Navigation**: `GpsVelocityFactor`, `PseudorangeFactor`, `DopplerFactor`,
    `BarometricFactor` (SE3 pose + scalar bias), `AttitudeFactor` (gravity/magnetometer
    direction constraint).
  - **Range family**: `PosePoseRangeFactor`, `PosePointRangeFactor`, `BearingRangeFactor`
    (4D: 3 bearing rows + 1 range row).
  - **Marginalization**: `MarginalPriorFactor` (GTSAM iSAM2 `LinearContainerFactor`
    analogue — Gaussian marginal over eliminated variables, manifold-agnostic via a
    caller-supplied local-log closure) plus `PoseRotationPrior` / `PoseTranslationPrior`
    partial-pose priors for loop-closure initialization.
- **Integration suite** (`tests/factor_integration.rs`): a synthetic multi-sensor dataset
  (circular-climb trajectory with sequentially self-consistent IMU propagation) driving
  end-to-end LM solves for VIO, monocular BA with cheirality violations, stereo VO,
  lidar scan matching (point-to-plane + GICP + point correspondences), a GNSS
  constellation solve (pseudorange + Doppler), marginalization consistency, loop
  closure with partial priors, pose-only smart-factor BA, and auxiliary barometer/
  attitude anchoring.

### Fixed

- **`BearingRangeFactor` never constrained the range** — the residual was bearing-only
  (3 rows), so range measurements were silently ignored. The residual is now 4D with the
  range row and its Jacobian (∂d/∂δρ = −q̂ᵀR, point block q̂ᵀ).
- **`BarometricFactor` took an R³ position** instead of an SE3 pose (GTSAM layout); the
  z-Jacobian follows the world-to-body retraction (`∂z/∂δρ = R[2,:]`).
- **`AttitudeFactor` used the wrong hemisphere** of the world-to-body rotation
  (`Rᵀ·d_world` instead of `R·d_world` for the crate's `T_wc` convention) and a wrong
  rotation-Jacobian sign.
- **`Sgal3` adjoint was stale** relative to the corrected group exponential
  (`apex-manifolds`): the ρ-row ν-column sign, the θ-column (`ρ̂R` instead of
  `(ρ−tν)̂R`), and the s-column (`−ν` instead of `+ν`) are fixed and now verified against
  `Log(g∘exp(ξ)∘g⁻¹)` by `adjoint_matches_group_conjugation`.
- `EuclideanPriorFactor` re-exported at `apex_solver::factors` level (was only reachable
  via the module path).

### Known limitations

- **`fix_variable` under-corrects free variables** that share factors with the fixed one:
  the linear solve still treats fixed coordinates as free and their step is discarded
  after application. Anchor with tight `EuclideanPriorFactor` / `PriorFactor` priors
  instead (see `tests/factor_integration.rs`); fully-fixed variables can also surface as
  "structurally empty diagonal" errors in the sparse LM damping.
- The SGal(3) IMU factors' analytical Jacobians are pending FD validation (tests
  `#[ignore]`d); the residual formulation itself is exact.


### Breaking Changes

- **`PriorFactor` is now a tangent-space anchor and generic over the manifold.**
  `r = Log(T_prior⁻¹ ∘ X) ∈ ℝ^dof` with the full between-chain Jacobian — no quaternion
  double-cover ambiguity, no dropped rotation–translation coupling, correct SE(2) angle wrap.
  The old ambient parameter-space factor is renamed **`EuclideanPriorFactor`** and is
  restricted to `Rn` variables at registration (anything else returns a
  `DimensionMismatch` error). Struct-literal construction (`PriorFactor { data }`) becomes
  `PriorFactor::new(prior)` / `EuclideanPriorFactor::new(data)`.
  ```rust
  // before — ambient, SE(3)-incorrect
  problem.add_residual_block(&[k], Box::new(PriorFactor { data: pose7 }), loss);

  // after — tangent anchor on SE(3)
  problem.add_residual_block(&[k], Box::new(PriorFactor::<SE3>::new(prior_pose)), loss);
  ```
- **`KannalaBrandtCamera::unproject` now returns `NumericalError`** for pixels outside the
  model's valid domain (`ru > π/2`) and for non-converged Newton iterations; the old code
  silently clamped `ru` and returned an unconverged ray.
- **`Rn::DIM` / `Rn::DOF` / `Rn::REP_SIZE` are deprecated** — they are `0` sentinels for a
  dynamic manifold, not dimensions. Use `is_dynamic()` / `tangent_dim()`.

### Added

- **Noise model layer** — measurement uncertainty per residual block:
  `NoiseModel` (`Null` identity default | `Diagonal` | `Dense`, sqrt-information domain),
  registered via `add_residual_block_with_noise` / `try_add_residual_block_with_noise`.
  Residuals and Jacobians are whitened (`r̃ = S·r`, `J̃ = S·J`) upstream of the robust-loss
  corrector, so the optimized objective is `Σ ½·ρ(‖S·r‖²)` — the Ω-weighted objective g2o
  reports. `pose_graph_g2o` and the odometry benchmark weight edges with the parsed
  information matrices **by default**; `--no-noise` restores the unweighted solve.
  `NoiseModel::from_information` tolerates rank-deficient/slightly indefinite Ω (negative
  eigenvalues clamped with a warning) — required by real g2o data such as sphere2500.
- **One-parameter subgroup law tests for all eight manifolds**
  (`exp(aξ)∘exp(bξ) = exp((a+b)ξ)`) plus an SGal(3) Jacobian composition check with strong
  time–velocity coupling.

### Fixed

- **`IterativeSchurSolver` published `−Jᵀr` from `get_gradient`** while every other backend
  published `+Jᵀr`, the documented contract. Levenberg-Marquardt and Dog Leg build their
  predicted cost reduction from that vector, so on the implicit-Schur path the step-quality
  ratio ρ came out sign-inverted. It also cached the *damped* `JᵀJ + λI` as `get_hessian`,
  which Dog Leg uses for the Cauchy point and the true quadratic model. `tests/
  linear_solver_contract.rs` now pins both conventions across all six backends.

- **`compute_step_quality` accepted steps that increased the cost.** A negative predicted
  reduction divided by a negative actual reduction yields ρ > 0, so a step the quadratic model
  itself expected to make things worse was accepted. Ceres treats a non-positive
  `model_cost_change` as an invalid step; it is now rejected. The near-zero case is unchanged,
  since at the solution both reductions legitimately vanish.

- **λ, ν, the trust-region radius and μ were stored in the config and mutated during a solve**,
  so a second `optimize()` call on the same solver silently started from wherever the previous
  run finished. They are now solver run state, re-seeded from the configuration on every solve.


- **SGal(3) `exp` is now the group exponential.** The old map dropped the time–velocity
  coupling entirely: `exp(aξ)∘exp(bξ)` differed from `exp((a+b)ξ)` by exactly `a·b·s·ν`,
  and `exp∘log ≠ id` whenever `s·ν ≠ 0`. `exp` now integrates the subgroup flow
  (`ρ' = Jl(θ)·ρ + s·M(θ)·ν` with `M(ω) = ½I + α·ω̂ + β·ω̂²`), `log` inverts it exactly, and
  `right/left_jacobian(±inv)` are computed as the derivative-by-definition of the corrected
  map (central differences through the crate's own compose/log) — validated by the subgroup
  law and a coupling-region composition test.
- **`SO3::log` returned the wrong sign near the negative-`w` identity.** For a rotation by
  `−s` (`w < 0`, `|s| < 2e-5`) the small-angle branch returned `+s`. Now sign-correct with
  regression tests.
- **Sim(3) `right/left_jacobian_inv` and `V⁻¹` no longer silently fall back to identity**
  on singular inputs; they use a Tikhonov-regularized inverse and emit a `warn!`.
- **Kannala-Brandt `unproject` validates its Newton iterations** post-loop (finite,
  converged) and rejects out-of-domain radii, matching `ftheta`.

### Changed

Eleven configuration fields were declared, documented, given builder setters — and never read.
Each now drives behaviour or is deprecated:

- `min_diagonal` / `max_diagonal` (LM) — the Marquardt damping diagonal, plus the
  `with_diagonal_bounds` setter they never had.
- `min_relative_decrease` (LM, Dog Leg) — the step-acceptance threshold.
- `max_condition_number` (LM, GN, Dog Leg) — checks `κ₂(JᵀJ) ≥ max_j H_jj / min_j H_jj`, a
  rigorous lower bound computed from the column norms already available, and terminates with
  `OptimizationStatus::IllConditionedJacobian` — a variant nothing previously constructed.
  Exceeding the threshold is proof of ill-conditioning; staying below it is not proof of the
  converse, and the doc says so. Its most useful case is a variable no residual constrains,
  which gives a zero column and an infinite bound; that previously surfaced as an opaque
  "JᵀJ has structurally empty diagonal entries" from inside the linear solver.
- `damping_increase_factor` / `damping_decrease_factor` / `min_step_quality` /
  `good_step_quality` (LM) — read by the new `DampingUpdate::Marquardt` policy. The default
  `DampingUpdate::Nielsen` derives both directions from ρ and ignores them, which is why they
  were inert; each field's doc now names the policy that reads it.
- `min_diagonal` (GN) — the uniform regularizer its doc-comment always promised, applied as
  `(JᵀJ + min_diagonal·I)`. Set to `0.0` for the un-regularized normal equations.
- `trust_region_increase_factor` (Dog Leg) — replaces the hardcoded `3.0` in the radius
  growth rule (same default, so unchanged behaviour). The radius growth now measures the step
  in the *scaled* space the radius bounds; previously it mixed the scaled radius with an
  un-scaled step norm whenever Jacobi scaling was on, which is Dog Leg's default.
- `min_step_quality` (Dog Leg) and `enable_visualization` (GN, Dog Leg) are `#[deprecated]`:
  the first duplicates `min_relative_decrease`, the second is superseded by the observer
  pattern (`solver.add_observer(RerunObserver::new(true)?)`).

Also: Dog Leg's `update_trust_region` return value was discarded, so a rejected step with
`0 < ρ < 1e-4` took the "moderate" branch and cleared the step-reuse cache as though it had
been accepted. Levenberg-Marquardt's predicted reduction moved from the `λI`-specific identity
`½·δᵀ(λδ − g)` to the policy-independent `−δᵀg − ½·δᵀHδ`, shared with Dog Leg.


- **Odometry benchmarks now solve the Ω-weighted objective end-to-end** — the optimized
  number is the χ² the harness reports. Measured impact on eight pose graphs: every final
  χ² improves (torus3D 1.8×, cubicle 240×, sphere2500 2.5×, mit 12×), five of eight are
  also faster. Details in [`noise-round-results.md`](noise-round-results.md).

- **`LinearSolver::solve_augmented_equation` takes `&Damping` instead of `lambda: f64`.**
  Custom `LinearSolver` implementations must update the signature. The augmented system it
  describes is now `(JᵀJ + λ·D)·dx = −Jᵀr` with `D_jj = clamp(JᵀJ_jj, min_diagonal,
  max_diagonal)` — Ceres' `LevenbergMarquardtStrategy`. `Damping::identity(lambda)` reproduces
  the previous uniform `λI` behaviour exactly:
  ```rust
  // before
  solver.solve_augmented_equation(&residuals, &jacobian, lambda)?;

  // after — same numerics
  solver.solve_augmented_equation(&residuals, &jacobian, &Damping::identity(lambda))?;
  ```

- **Levenberg-Marquardt now damps with `λ·D` and starts from `λ = 1e-4`** (was `λI` and
  `1e-3`). Iterates change on every problem. Measurements across ten pose graphs, two BAL
  datasets and three camera-calibration problems are in
  [`ceres-params-validation.md`](ceres-params-validation.md): order-of-magnitude wins wherever
  parameter scales are heterogeneous (calibration, bundle adjustment), bounded constant-factor
  costs on homogeneous pose graphs. To restore the old behaviour:
  `LevenbergMarquardtConfig::new().with_diagonal_bounds(1.0, 1.0).with_damping(1e-3)`.

- **Step acceptance is now gated on `min_relative_decrease`** in both Levenberg-Marquardt
  (was a hardcoded `rho > 0.0`) and Dog Leg (was `rho > 1e-4`). The default of `1e-3` matches
  Ceres, so marginal steps that used to be accepted are now rejected and the damping raised.

## [1.4.0] - 2026-07-30

### Breaking Changes

This release changes the public API. Code written against `1.3.0` will not compile without
the edits below. (The version number is `1.4.0` rather than `2.0.0` by project decision.)

- **`Problem` uses handles instead of string names.** `add_variable` now returns a `VarKey`,
  and `add_residual_block` takes `&[VarKey]` and returns a `FactorKey` — previously `&[&str]`
  and `usize`. Keep the returned key and pass it where you used to pass a name:
  ```rust
  // 1.3.0
  problem.add_variable("pose_0", ManifoldType::SE3, params);
  problem.add_residual_block(&["pose_0", "pose_1"], factor, loss);

  // 1.4.0
  let k0 = problem.add_variable(ManifoldType::SE3, params);
  let k1 = problem.add_variable(ManifoldType::SE3, params_1);
  problem.add_residual_block(&[k0, k1], factor, loss);
  ```
- **`Factor::get_dimension` renamed to `Factor::residual_dim`.** Custom factor implementations
  must rename the method; there is no default implementation.
- **`OptimizationStatus` gained the `StalledNoProgress` variant.** Exhaustive `match`
  expressions over this enum need a new arm. Treat it as a *successful* termination — it means
  the solver reached a point where the cost can no longer improve (verified to return the same
  final cost as running to the iteration limit).

### Changed
- **Slot-map `Problem` data structure (faster).** `Problem` now stores variables and
  residual blocks in `slotmap::SlotMap`s keyed by stable, generational `VarKey` / `FactorKey`
  handles, replacing the previous `HashMap<String, _>` design. Per-variable side data (fixed
  indices, bounds, column offsets) moved to matching `SecondaryMap`s. Benefits:
  - O(1) generational access on the assembly hot path — no string hashing or comparison.
  - No per-key allocation; `VarKey`/`FactorKey` are `Copy` 8-byte handles.
  - Cache-friendly, dense-array iteration during residual/Jacobian assembly.
  - Generational safety: a stale handle returns `None` instead of aliasing a reused slot.
- **Zero-copy nalgebra ↔ faer boundary.** Manifold parameters stay in contiguous `nalgebra`
  storage and are handed to factors as `&[f64]` slices that `faer` views directly
  (`from_column_major_slice`), removing `DVector`↔`Mat` conversions from the inner loop. The
  symbolic sparsity structure is built once and reused every iteration; parallel assembly is
  lock-free over disjoint buffers (rayon).

### Added
- **`LevenbergMarquardt` stall detection.** New config field
  `max_consecutive_rejected_steps` (default `5`, matching Ceres'
  `max_num_consecutive_invalid_steps`) and the matching
  `OptimizationStatus::StalledNoProgress`. LM stops once trial steps are rejected repeatedly
  *and* damping has saturated at `damping_max` — the point past which the step is negligible
  and the cost provably cannot change. Both conditions are required: a run of rejections alone
  is normal LM behaviour (damping rises, the next step succeeds).
- **`init_logger_with_directives(level, directives)`** — installs the tracing subscriber with
  fallback filter directives, so a noisy dependency can be quieted without mutating the process
  environment. `RUST_LOG` still takes precedence when set.

### Fixed
- **SE3 cost was over-reported by the C++ benchmark harness.** `SO3LogMap` in
  `benches/cpp_comparison/common/src/unified_cost.cpp` ignored quaternion double cover: when
  `q.w() < 0` the angle landed near `2π` instead of the equivalent short rotation, and the
  inverse left Jacobian (which divides by `sin θ`) then blew up the translation residual. On
  parking-garage the initial cost read `1.22e8` instead of `8.36e3` (~14,000×); on sphere2500
  `8.26e7` instead of `1.28e5`. All six benchmarked solvers now agree on initial cost for both
  SE2 and SE3, making cross-implementation comparison valid for the first time.

### Performance
- **Wasted LM iterations eliminated on stalled problems**, with bit-identical final cost —
  no tolerance was loosened. Measured on the pose-graph benchmark:
  - `torus3D`: 101 → 23 iterations, 5877 → 1347 ms (4.3× faster)
  - `cubicle`: 101 → 21 iterations, 7466 → 1579 ms (4.7× faster)
  - All other datasets unchanged in both iteration count and cost.

### Documentation
- **Documentation cookbooks** (mdBook, KaTeX) for the three sub-crates, each under
  `crates/<crate>/doc/cookbook`:
  - `apex-manifolds` — every group and operation (exp/log, adjoint, Jacobians, ⊞/⊟) with
    formulas derived from the implementation and a shared Conventions page.
  - `apex-camera-models` — a unified eight-section template per model with validity merged
    into projection/inverse-projection and corrected inverse-projection formulas.
  - `apex-io` — every public functionality (pose-graph formats, ASL, ROS1/ROS2 bags, DDS,
    CLI tools) organized by domain.
- **Performance benchmarks** (`doc/performance.md`) rewritten: 5 independent runs reported as
  mean ± std, with plotly figures for cost and runtime. Metrics changed from "% cost
  improvement" to the literature-standard final objective value plus runtime (SE-Sync,
  Rosen et al. IJRR 2019; Carlone et al. ICRA 2015).

### Notes
- Sub-crate versions bumped to `0.3.0` (`apex-manifolds`, `apex-io`, `apex-camera-models`).

## [1.3.0] - 2026-04-29

### Added
- **Three new Lie group manifolds** in `apex-manifolds` (v0.2.0):
  - `SE_2(3)` — extended pose with velocity for IMU preintegration (9 DOF)
  - `SGal(3)` — special Galilean group for time-coupled inertial navigation (10 DOF)
  - `Sim(3)` — similarity transforms with scale for monocular SLAM (7 DOF)
- **`FThetaCamera`** in `apex-camera-models` (v0.2.0) — NVIDIA DriveWorks f-theta fisheye
  model for 220° FOV surround-view cameras
- **`jacobian_pose`** on `CameraModel` trait — analytic ∂(u,v)/∂ξ for all 10 camera models
- **Comprehensive unit test suite** across all workspace crates:
  - `apex-manifolds`: identity, compose, inverse, round-trip exp/log, numerical Jacobian
    verification for all 8 manifolds
  - `apex-camera-models`: projection/unprojection round-trip, Jacobian verification,
    parameter validation, batch projection consistency for all 10 models
  - `apex-solver`: extended integration tests and factor Jacobian checks

### Changed
- **SO(3) quaternion convention** aligned to w-first (Hamilton) `[qw, qx, qy, qz]` —
  previously inconsistent between construction and serialization paths
- **`TryFrom<&[f64]>`** replaces `From<&[f64]>` for all camera model structs — construction
  is now fallible with structured `CameraModelError`
- Sub-crate versions bumped: `apex-manifolds 0.2.0`, `apex-camera-models 0.2.0`
- Workspace `Cargo.toml` dependencies updated to new sub-crate versions

### Fixed
- **SE(3) Q-matrix** sign error in `right_minus` Jacobian block
- **SO(3) Jacobian inverse** numerical stability near θ = 0 and θ = π
- **Sim(3) Jacobian and V-matrix** computations near degenerate scale values
- **SGal(3) tangent space adjoint** representation

## [1.2.1] - 2026-03-07

### Fixed
- **Visualization for pose_graph_g2o binary** - Fixed `--with-visualizer` flag to work properly with Rerun visualization (requires `visualization` feature). Now displays both initial and optimized pose graphs.
- **Visualization for bundle_adjustment binary** - Fixed and improved visualization:
  - Added documentation and usage examples for `--with-visualizer` flag
  - Changed 3D point colors to white (255,255,255) for better visibility
  - Reduced camera frustum scale
  - Now displays both initial and optimized states
- **Binary instructions in README** - Fixed outdated command examples and added proper usage documentation for visualization features
- **Git LFS setup** - Added clear instructions in README Quick Start section reminding users to pull data files using `git lfs pull` before running examples

## [1.2.0] - 2026-02-22

### Changed
- **Workspace layout flattened** - `apex-solver` crate moved from `crates/apex-solver/`
  to the repository root, following the standard pattern of major Rust projects (tokio,
  serde, axum). Sub-directories `src/`, `bin/`, `benches/`, `examples/`, `tests/` now
  live at the top level.
- Root `Cargo.toml` is now both the workspace manifest and the `apex-solver` crate
  manifest (combined `[workspace]` + `[package]` sections).
- All data file paths in benchmarks and integration tests updated to be relative to the
  workspace root (removed `../../` prefix).

### No API Changes
All public APIs, types, imports, and behavior are identical to v1.1.0.

## [1.1.0] - 2026-02-21

### Added
- **Cargo workspace restructuring** - Codebase split into four focused, independently publishable crates:
  - `apex-manifolds` (v0.1.0) - Lie group manifolds (SE2, SE3, SO2, SO3, Rn) with analytic Jacobians
  - `apex-io` (v0.1.0) - File I/O for pose graphs (G2O, TORO, BAL formats) with SE2/SE3 support
  - `apex-camera-models` (v0.1.0) - Camera projection models (pinhole, fisheye, omnidirectional) for bundle adjustment
  - `apex-solver` (v1.1.0) - Core nonlinear least squares optimizer, now depends on the above crates
- **`apex-manifolds` crate** - Standalone Lie group library usable independently of the optimizer
- **`apex-io` crate** - Standalone pose graph I/O library with G2O, TORO, and BAL format support
- **`apex-camera-models` crate** - Standalone camera model library with 9 projection models and analytic Jacobians

### Changed
- `apex-solver` now uses workspace dependencies for internal crates (dual `path + version` for local dev and publishing)
- Benchmark damping explicitly set to `1e-4` in `odometry_pose_benchmark` to match README baseline (global LM default is `1e-3`, optimized for bundle adjustment)

## [1.0.0] - 2026-01-24

### Added
- **Bundle Adjustment with Camera Intrinsic Optimization** - Full support for simultaneous optimization of camera poses, 3D landmarks, and camera intrinsics
  - **`ProjectionFactor<CameraModel, OptConfig>` generic system** - Type-safe bundle adjustment with compile-time optimization configuration
    - Optimization modes: `SelfCalibration` (pose + landmarks + intrinsics), `BundleAdjustment` (pose + landmarks), `OnlyPose`, `OnlyLandmarks`, `OnlyIntrinsics`, `PoseAndIntrinsics`, `LandmarksAndIntrinsics`
    - Batch projection support for multiple observations per factor
    - Automatic cheirality checking for points behind camera
    - Analytic Jacobians for all camera models (no auto-differentiation overhead)
  - **9 Camera Projection Models** with analytic Jacobians w.r.t. pose, point, and intrinsics:
    - `BALPinholeCameraStrict` - Bundle Adjustment in the Large format (focal, k1, k2)
    - `RadialTangential` - Brown-Conrady distortion model (fx, fy, cx, cy, k1, k2, p1, p2) - OpenCV compatible
    - `Equidistant` - Fisheye lens model (fx, fy, cx, cy, k1, k2, k3, k4)
    - `FOV` - Field-of-view distortion (fx, fy, cx, cy, omega)
    - `UnifiedCamera (UCM)` - Unified camera model for wide FOV (fx, fy, cx, cy, alpha)
    - `ExtendedUnified (EUCM)` - Extended unified model for >180° FOV (fx, fy, cx, cy, alpha, beta)
    - `DoubleSphere` - Double sphere projection for fisheye (fx, fy, cx, cy, xi, alpha)
    - `KannalaBrandt` - Fisheye polynomial model (fx, fy, cx, cy, k1, k2, k3, k4) - GoPro-style cameras
    - `Orthographic` - Orthographic projection (fx, fy, cx, cy)
  - **`CameraModel` trait** with compile-time constant `INTRINSIC_DIM`
    - Associated types: `IntrinsicJacobian`, `PointJacobian` for static-sized matrices
    - Methods: `project()`, `is_valid_point()`, `jacobian_point()`, `jacobian_pose()`, `jacobian_intrinsics()`
    - Batch processing with `project_batch()` for vectorized computation
- **Schur Complement Solvers** - Specialized linear algebra backends for bundle adjustment that exploit problem structure
  - **Explicit Schur Complement** (`ExplicitSchurComplementSolver`) - Direct sparse Cholesky factorization
    - Constructs reduced camera matrix S = B - E·C⁻¹·Eᵀ explicitly in memory
    - Suitable for medium-to-large BA problems (<10,000 cameras)
    - High accuracy with moderate memory usage
    - Supports block reordering and symbolic factorization caching
  - **Implicit Schur Complement** (`IterativeSchurSolver`) - Matrix-free Preconditioned Conjugate Gradients (PCG)
    - Memory-efficient for very large problems (10,000+ cameras)
    - Never constructs S explicitly - computes only matrix-vector products S·x
    - Three preconditioner types: `None`, `BlockDiagonal`, `SchurJacobi` (recommended, matches Ceres behavior)
    - Configurable CG parameters: max_iterations (default: 500), tolerance (default: 1e-9)
    - Linear memory growth with problem size
  - **`StructuredSparseLinearSolver` trait** - Extends `SparseLinearSolver` with variable structure awareness
    - Methods: `initialize_structure()` for BA-specific setup, `solve_normal_equation()`, `solve_augmented_equation()`
    - Enables Schur complement solvers to distinguish camera/landmark blocks
    - Required for exploiting sparsity structure in BA problems
- **BAL (Bundle Adjustment in the Large) File Format Support** (`src/io/bal.rs`)
  - `BalLoader::load()` - Parse BAL format datasets
  - `BalDataset` struct with cameras, points, observations
  - Supports large-scale structure-from-motion datasets (Dubrovnik, Ladybug, Trafalgar, Venice)
  - Git LFS integration for multi-GB dataset storage
- **New Binary: `bundle_adjustment`** - Professional CLI tool for BA optimization
  - Command-line options:
    - `--solver [explicit|implicit]` - Choose Schur complement solver variant
    - `--type [self-calibration|bundle-adjustment|only-pose|only-landmarks|only-intrinsics]` - Optimization configuration
    - `-n NUM_POINTS` - Limit dataset size for testing/profiling
    - `-v/--verbose` - Detailed optimization output
  - Supports all camera models and optimization modes
  - Real-time RMSE reporting and convergence diagnostics
- **New Binary: `pose_graph_g2o`** - Unified pose graph optimization tool
  - Replaces separate `optimize_2d_graph` and `optimize_3d_graph` binaries
  - Single CLI handles both SE2 and SE3 datasets automatically
  - Enhanced command-line interface with dataset selection, optimizer choice, loss functions
  - Support for real-time Rerun visualization with `--with-visualizer` flag
  - Output saving with `--save-output` option
- **Comprehensive Bundle Adjustment Benchmarks** (`benches/bundle_adjustment_benchmark.rs`)
  - Compares apex-solver vs Ceres, GTSAM, g2o on 4 BAL datasets
  - Datasets tested:
    - Dubrovnik: 356 cameras, 226,730 landmarks, 1,255,268 observations
    - Ladybug: 1,723 cameras, 156,502 landmarks, 678,718 observations
    - Trafalgar: 257 cameras, 65,132 landmarks, 225,911 observations
    - Venice: 1,778 cameras, 993,923 landmarks, 5,001,946 observations (largest)
  - Metrics: Initial/final RMSE, wall-clock time, iterations, convergence status
  - Automated CSV export for reproducibility
- **Refactored Odometry Pose Benchmarks** (`benches/odometry_pose_benchmark.rs`)
  - Renamed from `solver_comparison_benchmark` for clarity
  - Cleaner structure with consistent metrics reporting across 2D and 3D datasets
  - Enhanced output formatting with summary statistics
- **C++ Comparison Benchmarks** (`benches/cpp_comparison/`)
  - Reference Ceres, g2o, GTSAM implementations for BA and odometry
  - Enables side-by-side performance validation with identical datasets
  - CMake build system for cross-platform compatibility
- **Test Datasets**
  - BAL datasets via Git LFS: Dubrovnik (356 cams), Ladybug (1,723 cams), Trafalgar (257 cams), Venice (1,778 cams)
  - `city10000.g2o` - Large SE2 pose graph dataset (10,000 poses) for scalability testing

### Changed
- **Binary Consolidation** - Streamlined executable structure for better user experience
  - **Replaced** `optimize_2d_graph` and `optimize_3d_graph` with unified `pose_graph_g2o` binary
  - Single binary automatically detects SE2/SE3 datasets and applies appropriate optimizer
  - Cleaner codebase with reduced duplication
  - **BREAKING**: Users must update scripts to use `pose_graph_g2o` instead of old binary names
- **Linear Algebra Improvements**
  - **Removed** PowerSeries Schur complement solver (inferior performance and accuracy compared to explicit/implicit variants)
  - Enhanced `LevenbergMarquardtConfig::for_bundle_adjustment()` - Preset configuration optimized for BA
    - Pre-configured Schur solver selection, appropriate tolerances, and damping parameters
    - Reduces boilerplate for common BA use cases
  - Improved `SchurBlockStructure` with better variable classification (camera vs landmark blocks)
- **Camera Model System Refactoring**
  - Unified `CameraModel` trait with compile-time intrinsic dimensions
  - Associated types (`IntrinsicJacobian`, `PointJacobian`) for zero-cost abstractions
  - Consistent API across all 9 camera models
  - Batch projection methods for efficient multi-observation processing
- **Code Quality Enhancements**
  - Improved error handling with structured `LinAlgError` types (SchurDecompositionFailed, PCGConvergenceFailed, etc.)
  - Enhanced logging throughout optimization pipeline with tracing instrumentation
  - Removed deprecated bundle adjustment examples (replaced with production-ready `bundle_adjustment` binary)
  - Better separation of concerns in factor graph construction
- **Benchmark Infrastructure**
  - Renamed `solver_comparison_benchmark` → `odometry_pose_benchmark` for semantic clarity
  - Consistent metrics reporting format across all benchmarks (time, iterations, cost, RMSE)
  - Better separation between 2D (SE2) and 3D (SE3) dataset results
  - Enhanced summary statistics with convergence rate tracking

### Performance
- **Bundle Adjustment - Production-Grade Scalability**
  - **100% convergence rate** on all 4 BAL datasets (4/4 successful optimizations)
  - **Superior scalability**: Only solver alongside g2o to complete Venice dataset (5M observations)
    - Ceres and GTSAM timeout after 10 minutes on Venice
    - apex-solver completes in 83 seconds with 0.458 RMSE (vs g2o: 252s with 10.126 RMSE)
  - **Best accuracy on largest dataset**: Achieves 0.458 RMSE on Venice in only 2 iterations
    - 22x better accuracy than g2o (10.126 RMSE in 20 iterations)
  - **Speed advantage over Ceres**:
    - Dubrovnik: 61x faster (47s vs 2,879s), better RMSE (0.533 vs 1.004)
    - Trafalgar: 4.2x faster (10s vs 44s), better RMSE (0.679 vs 1.320)
  - **Competitive with GTSAM**: Similar convergence speed and accuracy on smaller datasets
  - **Memory efficiency**: Implicit Schur solver handles 10,000+ camera problems with linear memory growth
    - Venice (1,778 cameras): ~2GB peak memory vs Ceres/GTSAM >8GB
- **Pose Graph Optimization - Maintained Excellence**
  - Maintained 100% convergence rate on all 8 odometry datasets (4 SE2 + 4 SE3)
  - Consistent performance across problem scales: 434 poses (ring) to 5,000 poses (torus3D)
  - 2-10x faster than Ceres on most datasets while achieving equivalent or better final cost

## [0.1.6] - 2025-11-29

### Added
- **Comprehensive benchmark infrastructure** comparing 6 optimization libraries across 8 standard datasets
  - Rust solvers: apex-solver, factrs, tiny-solver
  - C++ solvers: Ceres Solver, g2o, GTSAM
  - Benchmark results on 4 SE2 datasets (intel, mit, M3500, ring) and 4 SE3 datasets (sphere2500, parking-garage, torus3D, cubicle)
  - Automated benchmark runner with CSV output for reproducibility
  - Performance metrics: execution time, iterations, cost improvement, convergence status
- **Integration test suite** in `tests/integration_tests.rs`
  - End-to-end optimization verification on real G2O datasets
  - Metrics tracked: convergence status, cost improvement, execution time, iteration count
  - Fast tests (ring, intel) and slow tests (sphere2500, parking-garage) with `#[ignore]` annotation

### Changed
- **Improved logging infrastructure**
  - All `println!`/`eprintln!` replaced with `tracing` macros for structured logging
  - Centralized logger configuration in `src/logger.rs` with custom formatter
  - Color-coded log levels and environment variable control (`RUST_LOG`)
  - Consistent logging levels across all modules (info, warn, error, debug, trace)
- **Enhanced code quality**
  - Removed all unwrap() and expect() calls from production code paths
  - Comprehensive error handling with Result types throughout the codebase
  - Cargo.toml lints enforce `unwrap_used = "deny"` and `expect_used = "deny"`
  - Zero panic-inducing calls in hot optimization paths

### Documentation
- Added comprehensive benchmark comparison table to README.md with analysis
  - Convergence reliability comparison across all 6 solvers
  - Performance highlights and cost improvement quality assessment
  - Instructions for reproducing benchmarks
- Moved project status section from README.md to CHANGELOG.md for better organization
- Enhanced Key Features section with v0.1.6 highlights

### Performance
- apex-solver achieves 100% convergence rate (8/8 datasets) - most reliable Rust solver
- Competitive performance: 2-10x faster than Ceres on most datasets
- Excellent cost reduction quality (>99% on well-conditioned problems)

## [0.1.5] - 2025-11-03

### Added
- **Camera Projection Factors** - 5 camera models for calibration and bundle adjustment
  - `DoubleSphereProjectionFactor` - Wide FOV fisheye cameras (6 params: fx, fy, cx, cy, α, ξ)
  - `EucmProjectionFactor` - Extended unified camera model (6 params: fx, fy, cx, cy, α, β)
  - `KannalaBrandtProjectionFactor` - Fisheye polynomial model (8 params: fx, fy, cx, cy, k1-k4)
  - `RadTanProjectionFactor` - Brown-Conrady distortion (9 params: fx, fy, cx, cy, k1, k2, p1, p2, k3)
  - `UcmProjectionFactor` - Unified camera model (5 params: fx, fy, cx, cy, α)
- **Factors Module Restructuring** - Dedicated `src/factors/` module with improved organization
  - Separated pose factors (SE2, SE3, Prior) from camera projection factors
  - Better code organization and discoverability
- **Factor Trait Enhancement** - Updated `Factor` trait with `compute_jacobian` parameter for optional Jacobian computation
- **Analytical Jacobians** - All camera factors use hand-derived analytical gradients (no auto-diff overhead)
- **Batch Processing Support** - Efficient vectorized computation for multiple point correspondences
- **Validity Checking** - Automatic detection of invalid projections in all camera models

### Changed
- **Code Quality Improvements** - Streamlined imports, renamed `Loss` trait to `LossFunction`, reduced Debug bounds

## [0.1.4] - 2025.10.26

### Added
- **15 Robust Loss Functions** - Comprehensive outlier rejection (Huber, Cauchy, Tukey, Welsch, Barron, and more)
- **Enhanced Termination Criteria** - 8-9 comprehensive convergence checks with relative tolerances
- **Prior Factors** - Anchor poses with known values and incorporate GPS/sensor measurements
- **Fixed Variables** - Hard-constrain specific parameter indices during optimization
- **Relative Tolerances** - Parameter and cost tolerances that scale with problem magnitude
- **New OptimizationStatus Variants** - Better diagnostics with `TrustRegionRadiusTooSmall`, `MinCostThresholdReached`, `IllConditionedJacobian`, `InvalidNumericalValues`
- **New Examples** - `loss_function_comparison.rs` and `compare_constraint_scenarios_3d.rs`

### Changed
- **Updated Defaults** - max_iterations: 50, cost_tolerance: 1e-6, gradient_tolerance: 1e-10

## [0.1.3] - 2025.10.20

### Added
- **Persistent symbolic factorization** - 10-15% performance boost via cached symbolic decomposition
- **Covariance for both Cholesky and QR** - Complete uncertainty quantification for all linear solvers
- **G2O file writing** - Export optimized graphs with `G2oWriter::write()`
- **Enhanced error messages** - Structured errors (`OptimizerError`) with numeric context
- **Binary executables** - Professional CLI tools: `optimize_3d_graph` and `optimize_2d_graph`
- **Real-time Rerun visualization** - Live optimization debugging with time series plots, Hessian/gradient heat maps
- **Jacobi preconditioning** - Automatic column scaling for robustness (enabled by default)
- **Improved examples** - `covariance_estimation.rs` and `visualize_optimization.rs`

### Changed
- **Updated dependencies** - Rerun v0.26, improved Glam integration

---

*For detailed usage examples and API documentation, see [README.md](README.md)*
