# Migrating 1.4 → 1.5 → 1.6

The crate's API has changed twice recently, in ways that affect every
consumer. This chapter is the migration guide: what changed, why, and the
exact edits a caller needs. The crate's `CHANGELOG.md` is the record; this is
the working checklist.

## 1. 1.3 → 1.4 — handles instead of names

The version stayed at 1.x by project decision, but the API broke.

**Variables and blocks are keyed.** `add_variable` returns a `VarKey`, and
`add_residual_block` takes `&[VarKey]` and returns a `FactorKey` — previously
`&[&str]` and `usize`:

```rust
// 1.3.0
problem.add_variable("pose_0", ManifoldType::SE3, params);
problem.add_residual_block(&["pose_0", "pose_1"], factor, loss);

// 1.4.0
let k0 = problem.add_variable(ManifoldType::SE3, params);
let k1 = problem.add_variable(ManifoldType::SE3, params_1);
problem.add_residual_block(&[k0, k1], factor, loss);
```

The generational keys are the reason the change was worth making: a key to a
removed variable resolves to `None` rather than to another variable's data.

**`Factor::get_dimension` → `Factor::residual_dim`.** No default
implementation: every custom factor must be renamed.

**`OptimizationStatus` gained `StalledNoProgress`.** An exhaustive `match`
over the enum needs a new arm. The new variant is a *successful* termination
— the solver reached a point where the cost can no longer improve.

## 2. 1.4 → 1.5 — the big one

### Breaking changes

<div class="apx-table">
<span class="apx-title">The 1.5 breaking changes</span>

| Change | Migration |
|---|---|
| `PriorFactor` is now a **tangent-space anchor**, generic over the manifold; the ambient version is `EuclideanPriorFactor` (`Rn` only) | construct with `PriorFactor::new(prior)` / `EuclideanPriorFactor::new(data)` instead of a struct literal; re-check the *meaning* of your prior, not just the name |
| `apex-io`'s `rosbag` is now opt-in; `download` and `clap` moved behind default-on features | `apex-solver = { version = "1.5", features = ["rosbag"] }` |
| `LinearSolver::solve_augmented_equation` takes `&Damping` instead of `lambda: f64` | `Damping::identity(lambda)` reproduces the old uniform $\lambda I$ |
| Schur solvers renamed and consolidated, **no deprecated aliases** | `SparseSchurComplementSolver` → `ExplicitSparseSchur`; `IterativeSchurSolver` → `ImplicitSparseSchur`; `SchurVariant` → `ExplicitSchurVariant`; `LinearSolverType::SparseSchurComplement` → `ExplicitSparseSchur`; new `ExplicitDenseSchur` |
| `for_bundle_adjustment` now selects `ImplicitSparseSchur` (2.2× faster) | pass `.with_linear_solver_type(LinearSolverType::ExplicitSparseSchur)` for the exact reduced solve |

</div>

The `PriorFactor` change is the one that is not a rename. A pose prior and a
parameter prior are different measurements, and code that was written against
the old ambient `PriorFactor` needs its *meaning* re-examined, not its
spelling.

### Behaviour changes

- **Levenberg–Marquardt damps with $\lambda D$** (Ceres-style) instead of
  $\lambda = 10^{-4}$, and both LM and Dog Leg gate step acceptance on
  `min_relative_decrease`. **The iterates change on every problem** — a
  trajectory that used to converge in 8 iterations may take 9, and the
  differences are not small. The crate's own analysis lives in
  `analysis/ceres-params-validation.md`.
- **Eleven previously-inert configuration fields now drive behaviour** — the
  diagonal bounds, the relative-decrease thresholds, the condition-number
  checks and the Marquardt damping policy. A configuration that was
  decorative becomes real; a configuration whose defaults were wrong now
  bites.
- **Ω-weighted objectives end to end.** The `pose_graph_g2o` binary and the
  odometry benchmark weight edges with the parsed information matrices by
  default; `--no-noise` restores the unweighted solve. A benchmark number
  computed under 1.4 and one computed under 1.5 are not comparable unless
  both used `--no-noise`.

### Additions

- The **factor library** grew: IMU (`SE23` / `SGal3`), visual, LiDAR,
  GNSS/navigation, range and bearing, motion-model and marginalization
  factors (chapter [The Factor Catalogue](../factors/catalogue.md)).
- The **noise model layer**: `NoiseModel` (`Null` / `Diagonal` / `Dense`, in
  the square-root-information domain), registered through
  `add_residual_block_with_noise`, with whitening applied *upstream* of the
  robust corrector (chapter [Noise Models](../core/noise.md)).
- The **new Schur solvers**: the matrix-free `ImplicitSparseSchur` (5.9–13.4×
  faster), `ExplicitDenseSchur`, a PCG forcing sequence (Ceres' $\eta$), and
  shared preconditioner handling across all Schur paths.

### Fixes worth knowing about

- The SGal(3) group exponential and adjoint, the Sim(3) action/adjoint/
  Jacobians, the `SE3`/`SE23` right Jacobians, and `SO3::log`'s small-angle
  sign were all corrected. A consumer whose trajectory changed slightly
  across the upgrade was not imagining it.
- `BearingRangeFactor` now constrains the range (a 4-D residual);
  `BarometricFactor` takes an `SE3` pose; `AttitudeFactor` uses the correct
  hemisphere.
- The optimizer state ($\lambda$, $\nu$, the trust-region radius, $\mu$) is
  re-seeded per solve, so one solve's damping no longer leaks into the next.

### Known limitations, stated by the crate

- **`fix_variable` under-corrects free variables** that share factors with the
  fixed one. Anchor with a tight `EuclideanPriorFactor` / `PriorFactor`
  prior instead.
- The SGal(3) IMU factors' analytic Jacobians are **pending finite-difference
  validation** (the tests are `#[ignore]`d); the residual formulation itself
  is exact.

## 3. 1.5 → 1.6 — the staged deprecations

The 1.6 cycle has not cut a release yet, but the tree already carries its
deprecations, annotated `since = "1.6.0"`. Ten items are marked, and they
cluster into three groups.

<div class="apx-table">
<span class="apx-title">The 1.6 deprecations</span>

| Item | Replacement |
|---|---|
| `Problem::mark_as_schur_landmark` | `mark_for_elimination` — elimination is not restricted to landmarks |
| `linalg::utils`: two sparse-to-dense helpers | "no callers outside this module's own tests; densifying a sparse Jacobian defeats the sparse pipeline" — build sparse matrices from the symbolic structure instead |
| `GaussNewtonConfig`: two inert fields | "never read" / "no-op; attach a `RerunObserver` with `add_observer` instead" |
| `optimizer::mod`: two scaling helpers | "quadratic in the number of columns" / "builds a diagonal scaling matrix and multiplies" — the optimizers scale the columns directly |
| `optimizer::mod`: one solver constructor | "no optimizer calls this; construct the desired solver directly" |
| `DogLegConfig`: three inert fields | two are "never read"; one is a "no-op" — the acceptance threshold is `min_relative_decrease` |
| `LevenbergMarquardtConfig`: the `radius` argument and a `radius` field | "the `radius` argument is ignored — Levenberg-Marquardt here is damping-controlled" / "never read" |

</div>

The pattern is consistent: the crate removed configuration that did not do
what it said, and every removal says what to do instead. The `radius`
deprecations are the clearest case — LM has no trust region, and a `radius`
field on an LM configuration is a lie with a default value.

## 4. A migration checklist

<div class="apx apx-check">
<span class="apx-title">Check — moving a consumer across 1.4 → 1.6</span>

1. Rename `get_dimension` → `residual_dim` in every custom factor.
2. Switch from string names to `VarKey` / `FactorKey`.
3. Decide, for every prior, whether it was a pose prior or a parameter
   prior — and pick `PriorFactor` or `EuclideanPriorFactor` accordingly.
4. Move from `lambda: f64` to `&Damping` (`Damping::identity(λ)` for the old
   behaviour).
5. Rename the Schur solvers; there are no aliases, so the compiler finds
   every one.
6. Re-check every "inert" configuration field — the eleven that became live
   in 1.5 are the ones most likely to change your iterates.
7. Decide whether the bundle-adjustment default change (implicit Schur) is
   acceptable, or pass `ExplicitSparseSchur` explicitly.
8. If the benchmark numbers are quoted anywhere, re-run them: the objective
   is now Ω-weighted by default.
9. Expect the iterates to change. Compare trajectories, not just costs.

</div>

## 5. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a deprecation is a promise about the future, not the present</span>

The 1.6 deprecations are annotated `since = "1.6.0"` while the crate's
version is still `1.5.0`. That is intentional staging — the deprecations
land in the tree first, and the release that removes the items will be 1.6 —
but it means a `1.5.0` build already carries warnings about a version that
does not exist yet. A consumer that treats warnings as errors should expect
them, and a consumer that pins 1.5 must know that 1.6 will remove those
items rather than merely warn about them.

</div>

<div class="apx-note">
<span class="apx-title">Note — the changelog is the record</span>

`CHANGELOG.md` at the repository root carries the full text of every release,
including the analysis documents behind the behaviour changes. The crate's
migration page in the older documentation surface covers the same ground; this
chapter is the version that includes the 1.6 staging.

</div>
