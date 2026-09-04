# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Apex Solver is a Rust workspace for nonlinear least squares optimization (SLAM, bundle
adjustment, pose graph optimization). Root crate `apex-solver` (edition 2024, MSRV 1.93) plus
three independently-publishable sub-crates in `crates/`.

## Commands

### Build / check

```bash
cargo check --workspace --all-targets --all-features   # fast compile check (mirrors CI)
cargo build --release                                    # optimized build
```

### Test

```bash
cargo test --workspace --all-features --release   # full suite, matches CI exactly
cargo test test_name                               # filter by name across targets
cargo test --test integration_tests                 # one integration test file (tests/*.rs)
cargo test -p apex-manifolds                        # a single workspace crate
cargo test -- --nocapture                           # show println!/tracing output
```

- `[profile.test]` in `Cargo.toml` inherits `release` — `cargo test` is already optimized
  (large-graph integration tests are ~50x slower under plain `debug`). The tradeoff:
  `debug_assert!` and integer-overflow checks are **not** exercised by `cargo test`; use a debug
  `cargo run`/`cargo check` when chasing that class of bug.
- Integration tests in `tests/` (e.g. `integration_tests.rs`, `schur_ba_agreement.rs`) pull real
  G2O/BAL datasets on first run via `apex_io::ensure_*`/`ODOMETRY_DATA_DIR_*` — first run needs
  network access; subsequent runs use the cached files under `data/`.

### Lint / format

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Workspace-wide (`[workspace.lints]` in `Cargo.toml`, non-negotiable — CI enforces `-D warnings`):
`unsafe_code = "forbid"`; clippy `unwrap_used`, `expect_used`, `print_stdout`, `print_stderr` are
all `deny`. Library code must return `Result` and log via `tracing`, never `println!`/`unwrap()`.
The only opt-outs are CLI binaries under `crates/apex-io/bin/` and a few test diagnostics, which
disable these lints file-by-file for user-facing console output.

### Benchmarks

```bash
cargo bench --bench odometry_pose_benchmark      # vs factrs, tiny-solver, Ceres, GTSAM, g2o
cargo bench --bench bundle_adjustment_benchmark  # vs Ceres, GTSAM, g2o
cargo bench --bench micro_kernels
```

C++ comparisons (`benches/cpp_comparison/`) build on first run and need CMake 3.15+, Eigen3,
Ceres, GTSAM, g2o (`brew install ceres-solver gtsam g2o eigen` on macOS); missing solvers are
skipped with a warning. See `benches/README.md` for the repeated-run/plotting workflow and
`doc/performance.md` for published numbers.

### Datasets

```bash
cargo run --release -p apex-io --bin download_datasets -- --list
cargo run --release -p apex-io --bin download_datasets -- --select 10   # everything benches need
cargo run --release -p apex-io --bin download_datasets -- --select 12   # IMU/GNSS sensor datasets
```

Lands in `data/odometry/{2d,3d}/`, `data/bundle_adjustment/<name>/` and
`data/sensor/<name>/`. No Git LFS.

`tests/nclt_gnss_fusion.rs` runs odometry+GNSS fusion on the real
[NCLT](https://robots.engin.umich.edu/nclt/) dataset; it pulls ~40 MB of CSV on
first run via `apex_io::ensure_sensor_dataset` (NCLT publishes each stream
separately, so no imagery is downloaded).

### Run binaries / examples

```bash
cargo run --release --bin pose_graph_g2o -- --dataset sphere2500
cargo run --release --bin bundle_adjustment -- --dataset ladybug
cargo run --release --example compare_solvers
cargo run --release --features visualization --bin pose_graph_g2o -- --dataset sphere2500 --with-visualizer
```

`visualize_graph_file` and `visualize_optimization` examples require `--features visualization`.

### Profiling

```bash
cargo build --profile=profiling --example <name>
samply record ./target/profiling/examples/<name>
```

## Architecture

### Workspace layout

```
apex-solver/            # root crate: problem formulation, optimizers, linear algebra
├── src/core/            # Problem (slotmap arena), residual blocks, loss functions, variables
├── src/factors/         # Factors grouped by sensor modality, addressed by that path
│   │                    #   (factors::visual::StereoFactor, factors::imu::ImuFactor)
│   ├── pose/            # between, prior, partial pose priors
│   ├── visual/          # projection, stereo, inverse depth, smart, essential, depth
│   ├── lidar/           # distance field/ICP, edge, plane, point-to-point, GICP
│   ├── imu/             # IMU preintegration on SE23/SGal3 states; each group has
│   │                    #   exactly ImuFactor (shared bias) + CombinedImuFactor (per-frame)
│   ├── navigation/      # GNSS position/velocity/raw, barometric, attitude
│   ├── ranging/         # range, bearing, bearing-range
│   ├── marginal/        # Gaussian marginal prior
│   └── common/          # shared math, cheirality, validation (NOT manifold derivatives)
├── src/linearizer/      # Jacobian assembly bridging factors -> linear system (cpu/{dense,sparse})
├── src/linalg/          # Linear solvers: dense/sparse Cholesky & QR, explicit/implicit Schur
├── src/optimizer/       # Levenberg-Marquardt, Gauss-Newton, Dog Leg
├── src/observers/       # Optimizer callbacks (Rerun visualization, custom hooks)
├── bin/, examples/, benches/, tests/
└── crates/
    ├── apex-manifolds/   # Lie groups: SE2, SE3, SO2, SO3, SE_2(3), SGal(3), Sim(3), Rn
    ├── apex-io/          # G2O/TORO/BAL/ASL/rosbag file I/O, dataset downloader CLI
    └── apex-camera-models/ # 10 camera projection models with analytic Jacobians
```

### Optimization pipeline

`Problem` (in `core/problem.rs`) stores variables and residual blocks in `slotmap::SlotMap`s,
referenced by stable generational `VarKey`/`FactorKey` handles (not strings) — O(1) access, no
hashing, `Copy` 8-byte keys. Per-variable side data (fixed indices, bounds, column offsets) lives
in matching `SecondaryMap`s. Manifold parameters stay in contiguous `nalgebra` storage that
`faer` views without copying (`MatRef`/`MatMut::from_column_major_slice`) across the whole hot
path.

Each solver iteration flows:

```
Problem (factor graph)
    │  AssemblyBackend::assemble()   [src/linearizer]
    ▼
(residual: Mat<f64>, Jacobian: M::Jacobian)   ← M: LinearizationMode (Dense | Sparse)
    │
    ▼
LinearSolver<M>   [src/linalg]  — Cholesky | QR | Explicit Schur | Implicit Schur (matrix-free PCG)
    │
    ▼
dx: Mat<f64>  →  manifold ⊞ update (apex-manifolds)  →  Optimizer step accept/reject [src/optimizer]
```

Symbolic sparsity structure (`SymbolicStructure`) is built once and reused every iteration —
never recomputed inside the loop. Assembly is parallelized over disjoint buffers with `rayon`.

### Manifold derivatives come from `apex-manifolds`

Every group (`so3`, `se3`, `se23`, `sgal3`, …) reports the Jacobians of its own
operations through optional output arguments — `act`, `compose`, `log`,
`right_plus`, `right_minus`, `between` all take `Option<&mut JacobianMatrix>`
slots and fill them using that group's right/left conventions.

**Factors must ask the manifold rather than re-deriving blocks by hand.** A
hand-written `[R | −R·[p]ₓ]` is a second copy of a convention that can silently
drift from the group's — and did: `src/factors/common/` deliberately contains
no manifold derivatives, only skew/sinc/matrix-root helpers, the cheirality
penalty, block validation, and test utilities.

Two conventions to know: poses are body-in-world (`T_wb`), so a body-frame
direction is `Rᵀ(p_world − t)`; and a factor either whitens internally or takes
a `NoiseModel`, never both (`Factor::whitens_internally()` — registering an
internally-whitened factor with a non-null model is rejected).

### Error hierarchy (`src/error.rs`)

Strict three-layer, bubble-up via `?` and `#[from]` — never construct `ApexSolverError` in a
deep module:

- **Layer A** (`ApexSolverError`, top): the only type exposed by the public API.
- **Layer B** (`OptimizerError`, `ObserverError`): wrap Layer C errors with call-site context.
- **Layer C** (`CoreError`, `LinAlgError`, `ManifoldError`, `FactorError`, `LinearizerError`,
  `CameraModelError`, `IoError`): module-specific; must return their own type, not
  `ApexSolverError`.

`LinAlgError` has a dual conversion path: direct (`LinAlgError → ApexSolverError::LinearAlgebra`)
for standalone linalg usage, or through the optimizer (`LinAlgError → OptimizerError::LinAlg →
ApexSolverError::Optimizer`) to preserve optimization context when it fails mid-solve. Use
`ApexSolverError::chain()` / `chain_compact()` to print the full cause chain.

### Bundle-adjustment parameterization (const-generic type state)

`ProjectionFactor<Camera, OptimizeParams<POSE, LANDMARK, INTRINSIC>>` selects at compile time
(zero runtime cost) which of camera pose / 3D landmark / camera intrinsics a factor optimizes.
Named aliases in `src/factors/mod.rs`: `BundleAdjustment` (pose+landmark), `SelfCalibration`
(pose+landmark+intrinsic), `OnlyPose`, `OnlyLandmarks`, `OnlyIntrinsics`,
`PoseAndIntrinsics`, `LandmarksAndIntrinsics`.

### Feature flags

`visualization` (off by default) pulls in `rerun` for real-time optimization debugging
(`src/observers/visualization.rs`); zero overhead when disabled. Enable with `--features
visualization`; requires the `RerunObserver` to be added via `solver.add_observer(...)`.

## Working Style

These guidelines bias toward caution over speed; for trivial tasks, use judgment.

1. **Think before coding.** State assumptions explicitly (e.g. "assumes the objective is
   convex"). Present tradeoffs when choosing between algorithms or linear solvers (Cholesky vs
   QR vs Schur: convergence, memory, numerical precision). Push back on complexity that isn't
   justified by the request. Stop and ask if something is genuinely unclear.
2. **Simplicity first.** No speculative generics, abstractions, or configurability beyond what
   was asked. Follow existing patterns in `core/`/`linalg`/`apex-manifolds` rather than inventing
   new ones.
3. **Surgical changes.** Match existing style (including the Layer A/B/C error hierarchy above).
   Every changed line should trace to the request — don't reformat or "improve" unrelated code.
   Only remove imports/dead code that your own change made unused.
4. **Goal-driven execution, test-first.** Bug fixes start with a failing test in `tests/` or a
   `#[cfg(test)]` module. Numerical/optimization changes must state a success criterion
   (convergence within an epsilon, matching a golden value) and verify it. Performance-sensitive
   changes in `linalg/` or the manifold crates should be checked with `cargo bench` before
   finalizing.
