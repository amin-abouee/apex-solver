# Code Map

Where everything lives, and which chapter covers it. The crate is 120 source
files, and the call graph fans out from the optimizer into the linearizer
into the factors — this page is the index for going from a concept in the
book to the file that implements it.

## 1. The tree

<div class="apx apx-table">
<span class="apx-title">`src/` by module</span>

| Path | Contents | Chapter |
|---|---|---|
| `lib.rs` | the crate root: re-exports, the feature flags, the prelude | [Errors, Logging and the Prelude](./errors-and-logging.md) |
| `error.rs` | the three-layer error hierarchy, `ErrorLogging` | [Errors, Logging and the Prelude](./errors-and-logging.md) |
| `logger.rs` | the `tracing` setup for the binaries | [Errors, Logging and the Prelude](./errors-and-logging.md) |
| `prelude.rs` | the curated import surface | [Errors, Logging and the Prelude](./errors-and-logging.md) |
| `core/problem.rs` | the graph: variables, blocks, assembly, the linearization point | [The Problem](../core/problem.md) |
| `core/variable.rs` | `ManifoldVariable`, the fixed coordinates, the covariance slot | [Variables on Manifolds](../core/variables.md) |
| `core/residual_block.rs` | a factor plus its loss and its noise model | [The Problem](../core/problem.md) |
| `core/noise.rs` | `NoiseModel`, the information repair, the repair summary | [Noise Models](../core/noise.md) |
| `core/loss_functions.rs` | the whole robust-kernel family | [Robust Loss Functions](../core/losses.md) |
| `core/corrector.rs` | the Triggs re-weighting | [The Corrector](../core/corrector.md) |
| `core/marginalization.rs` | `Marginalizer`, `Marginal`, the sign convention | [Marginalization](../marginal/marginalization.md) |
| `factors/mod.rs` | the `Factor` trait, `NormalBlocks`, `OptimizeParams` | [Writing a Factor](../factors/writing.md) |
| `factors/common/` | `math`, `cheirality`, `group_point`, `validate` | [Common Utilities](../factors/common.md) |
| `factors/pose/` | `BetweenFactor`, the priors, `Relative4DofFactor` | [Pose and Prior Factors](../factors/pose.md) |
| `factors/visual/` | the seventeen visual factors, geometry and photometric | [Visual Geometry](../factors/visual-geometry.md), [Photometric](../factors/visual-photometric.md) |
| `factors/imu/` | preintegration, the two groups, gravity and scale | [IMU Factors](../factors/imu.md) |
| `factors/lidar/` | the plane, field and GICP factors; the affine action | [LiDAR Factors](../factors/lidar.md) |
| `factors/ranging/` | range and bearing factors | [Range and Bearing](../factors/ranging.md) |
| `factors/motion/` | nonholonomic, planar, stationarity | [Motion Model Factors](../factors/motion.md) |
| `factors/navigation/` | GPS, raw GNSS, barometric, attitude | [Navigation Factors](../factors/navigation.md) |
| `factors/marginal/` | `MarginalPriorFactor` | [The Marginal Prior](../marginal/prior-factor.md) |
| `factors/aggregate.rs` | `AggregateFactor` — many factors, one block | [Decorators](../factors/decorators.md) |
| `factors/held.rs` | `HeldBlocks` — pinned blocks | [Decorators](../factors/decorators.md) |
| `factors/offset.rs` | `RightOffset` — a block at a fixed offset | [Decorators](../factors/decorators.md) |
| `factors/projected.rs` | `ProjectedFactor` — a Jacobian on a subspace | [Decorators](../factors/decorators.md) |
| `linalg/mod.rs` | `Damping`, `LinearSolverType`, `StructureAware` | [Damping](../optimizer/damping.md), [Choosing a Solver](../optimizer/choosing.md) |
| `linalg/dense/` | dense Cholesky, QR, the dense Schur | [Dense Linear Solvers](../linalg/dense.md) |
| `linalg/sparse/` | sparse Cholesky, QR, the normal-equations cache, the Schur solvers | [Sparse Linear Solvers](../linalg/sparse.md) |
| `linalg/schur/` | the partition, the ordering, the PCG, the preconditioners | [The Schur Complement](../linalg/schur.md) |
| `linalg/covariance.rs` | `Covariance`, the two inversion strategies | [Covariance](../linalg/covariance.md) |
| `linearizer/` | the per-iteration evaluation, sparse and dense, the evaluation policy | [Linearization](../core/linearization.md) |
| `optimizer/mod.rs` | the `Optimizer` trait, the convergence checks, the shared machinery | [The Optimizer Contract](../optimizer/contract.md) |
| `optimizer/gauss_newton.rs` | the undamped optimizer | [Gauss–Newton](../optimizer/gauss-newton.md) |
| `optimizer/levenberg_marquardt.rs` | the damped optimizer, `DampingUpdate`, the presets | [Levenberg–Marquardt](../optimizer/levenberg-marquardt.md) |
| `optimizer/dog_leg.rs` | the trust-region optimizer | [Dog Leg](../optimizer/dog-leg.md) |
| `observers/` | the observer pattern and the Rerun observer | [Observers](../tools/observers.md) |

</div>

## 2. Outside `src/`

<div class="apx apx-table">
<span class="apx-title">The rest of the workspace</span>

| Path | Contents |
|---|---|
| `crates/apex-manifolds/` | the Lie groups and their Jacobians — its own cookbook |
| `crates/apex-camera-models/` | the camera projections — its own cookbook |
| `crates/apex-io/` | the recording formats — its own cookbook |
| `benches/` | the criterion benches: pose graphs, bundle adjustment, micro-kernels, window kernels |
| `tests/` | the integration suites (the multi-sensor one is `factor_integration.rs`) |
| `doc/performance.md` | the full benchmark tables, with the methodology |
| `doc/factors.md`, `doc/factor-catalog.md` | the older documentation surface — see the catch |
| `analysis/` | the measurement write-ups behind the presets and the forcing sequence |

</div>

## 3. Concept → code

| If you are looking for… | Start at |
|---|---|
| how a residual becomes a step | `core/residual_block.rs` → `linearizer/cpu/sparse.rs` → `optimizer/mod.rs` |
| why a factor whitens twice | `factors/mod.rs::whitens_internally` and `core/problem.rs`'s registration check |
| where the damping is applied | `linalg/mod.rs::Damping` and `optimizer/levenberg_marquardt.rs` |
| where the Schur fill-in is decided | `linalg/schur/ordering.rs` |
| why the marginal prior's sign is what it is | `core/marginalization.rs`'s sign section |
| how a factor's Jacobian is tested | the `*_tests.rs` file beside the factor, and `factors/common/test_utils.rs` |
| what a config field actually does | `optimizer/*.rs`'s `with_*` builders and the `#[deprecated]` notes |

## 4. Reading order for a newcomer

1. `core/problem.rs` — the graph, end to end.
2. `core/residual_block.rs` and `core/noise.rs` — what a block *is*.
3. `linearizer/mod.rs` — how a block becomes a linear system.
4. `optimizer/mod.rs` — how a linear system becomes a step.
5. `linalg/schur/mod.rs` — the one genuinely intricate piece.
6. One factor family, end to end: `factors/pose/between.rs` is the shortest.

## 5. The catch

<div class="apx-warning">
<span class="apx-title">Warning — there are two documentation surfaces</span>

The cookbook is the book, but the repository also carries `doc/factors.md`,
`doc/factor-catalog.md`, `doc/examples.md` and `doc/performance.md` beside
it. They are *not* part of the book and they can drift: the cookbook is
written against the crate, while those pages are a separate, older set of
notes. When the two disagree, the cookbook and the code win — and the
right fix is to change the code or the cookbook, not to reconcile the notes
by hand.

</div>

<div class="apx-note">
<span class="apx-title">Note — the `analysis/` directory is where the presets come from</span>

The optimiser presets, the forcing-sequence parameter $\\eta$ and the
"uniform damping is faster for 2D pose graphs" claim are not folklore: they
are measurements, and the write-ups are in `analysis/` and in the comments of
the code that uses them. A chapter that quotes a number should be able to
point at the sweep that produced it.

</div>
