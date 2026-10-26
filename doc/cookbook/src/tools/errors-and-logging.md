# Errors, Logging and the Prelude

A crate that refuses to panic, forbids `unsafe`, and denies `unwrap` has to
say what it does instead. This chapter is that contract: the error hierarchy,
the propagation convention, the logging helpers, and the prelude that a
consumer normally imports.

## 1. The hierarchy

<div class="apx apx-table">
<span class="apx-title">The errors</span>

| Type | Module | Wraps |
|---|---|---|
| `ApexSolverError` | crate root | every module error below, transparently |
| `CoreError` | `core` | `FactorError`, `ManifoldError`, `LinearizerError` |
| `FactorError` | `factors` | `CameraModelError`, `ManifoldError`, `PreintegrationError` |
| `LinearizerError` | `linearizer` | the faer failure behind a pattern copy |
| `LinAlgError` | `linalg` | the faer failure behind a factorization |
| `OptimizerError` | `optimizer` | `LinAlgError`, `CoreError`, `LinearizerError` |
| `MarginalizationError` | `core::marginalization` | `FactorError`, `LinearizerError`, `CoreError` |
| `CovarianceError` | `linalg::covariance` | `LinAlgError`, `LinearizerError` |
| `ObserverError` | `observers` | the Rerun error |
| `PreintegrationError` | `factors::imu` | — |

</div>

A module returns *its own* error — the optimizers return `OptimizerError`, not
the root — and `#[from]` converts at each boundary, so `?` is all a caller
writes. Every enum is `#[non_exhaustive]` and carries structured fields
(`VariableCount { factor, expected, actual }`), not strings.

A module error that wraps another adds a short context message and keeps the
original as its `source()`; the root wraps transparently. The chain therefore
names each hop once:

```text
linear solve failed
  → factorization of the normal equations failed, the matrix is singular or not positive definite
  → NonPositivePivot { index: 3 }
```

`ApexSolverResult<T>` is the alias for `Result<T, ApexSolverError>`; each
module has its own (`CoreResult`, `FactorResult`, `OptimizerResult`, ...).

## 2. Reading the chain

`ErrorChainExt` (re-exported from `apex_solver::error`, implemented for every
`std::error::Error`) renders the chain:

```rust
use apex_solver::ErrorChainExt;

if let Err(e) = solver.optimize(&mut problem) {
    tracing::warn!("{}", e.chain());         // one line per hop
    tracing::warn!("{}", e.chain_compact()); // "a → b → c"
}
```

`Display` on its own prints only the outermost message, so report the chain at
the boundary that gives up.

## 3. Logging once

`ErrorLogging` (also re-exported from `apex_solver::error`) logs an error at
`error` level with the caller's file and line, and returns it:

```rust
return Err(CoreError::UnknownVariable { key }.log());
```

The rule is *log at origin*: the function that detects the failure builds the
error and calls `.log()` (or `.log_with_source(e)` when a third-party error
has no `#[from]` edge); every caller above propagates with a bare `?` and does
not log again. Expected per-residual outcomes — a point that does not project,
a scan point outside a distance field — are returned unlogged, because the
caller decides whether they matter.

## 4. Registration never panics

`Problem::add_residual_block`, `add_residual_block_with_noise`,
`fix_variable` and `set_variable_bounds` return `CoreResult`; there is no
panicking form. `try_add_variable` validates a parameter vector through the
manifold's checked constructor; `add_variable` remains for parameters known
to be valid.

## 5. The logger

The crate ships a small, opinionated `tracing` setup for its binaries and
examples:

```rust
init_logger();                            // INFO by default
init_logger_with_level(Level::DEBUG);
init_logger_with_directives("apex_solver=trace,info");
```

The format is `[LEVEL YYYY-MM-DD HH:MM:SS module]` for INFO/WARN/ERROR and
`[LEVEL YYYY-MM-DD HH:MM:SS file:line]` for DEBUG/TRACE — the source
location appears only when it is useful, which is what makes a busy INFO log
readable.

The level is overridable by the `RUST_LOG` environment variable, in the usual
`tracing` syntax:

```console
$ RUST_LOG=debug cargo run --bin pose_graph_g2o
$ RUST_LOG=apex_solver=trace cargo run
```

A *library* should not install a subscriber — that is a decision for the
binary. `init_logger` is re-exported at the crate root for the crate's own
executables and for consumers that want the same format.

## 6. The prelude

```rust
use apex_solver::prelude::*;
```

The prelude is the crate's recommended import surface, and the source is
explicit about its policy: it carries "only the pieces almost every program
needs — the problem, the optimizers, the manifolds, and the three factors
that appear in nearly any graph".

| Group | Names |
|---|---|
| the graph | `Problem`, `Variable`, `NoiseModel` |
| the factors | `Factor`, `BetweenFactor`, `PriorFactor`, `EuclideanPriorFactor`, `ProjectionFactor` |
| the optimisation presets | `OptimizeParams` and its aliases `BundleAdjustment`, `SelfCalibration`, `OnlyPose`, `OnlyLandmarks`, `OnlyIntrinsics`, `PoseAndIntrinsics`, `LandmarksAndIntrinsics` |
| the solvers | `GaussNewton`, `LevenbergMarquardt`, `LevenbergMarquardtConfig`, `DogLeg`, `Optimizer` |
| the linear algebra | `JacobianMode`, `LinearSolverType` |
| the manifolds (re-exported from `apex-manifolds`) | `LieGroup`, `ManifoldType`, `Tangent`, `SE3`, `SE2`, `SE23`, `SGal3`, `SO2`, `SO3`, `Rn` |
| the errors | `ApexSolverError`, `ApexSolverResult` (`ErrorChainExt`, `ErrorLogging` at the crate root) |

The deliberate omissions are as informative as the list. The *other* factor
families are not in the prelude, because a factor lives under a module named
for its sensor modality — `factors::visual::StereoFactor`,
`factors::imu::ImuFactor` — and the import itself is documentation of where
the measurement comes from. The marginalization types, the observers, the
logger and the linear solvers are reached by their own paths.

The crate root re-exports a wider set (`apex_solver::NoiseModel` works as
well as `apex_solver::prelude::NoiseModel`), but the prelude is the *curated*
list: it is the one a consumer can rely on staying stable.

## 7. The lint contract

<div class="apx apx-table">
<span class="apx-title">What the crate forbids</span>

| Forbidden | Enforcement |
|---|---|
| `unwrap` | `clippy::unwrap_used = "deny"`, crate-wide, tests included |
| `expect` | `clippy::expect_used = "deny"` |
| `panic!`, `todo!`, `unimplemented!` | denied |
| `unsafe` | `rust::unsafe_code = "forbid"` |
| `#[allow(...)]` to silence a finding | a project rule — findings are fixed, not silenced |

</div>

The crate's own test profile inherits `release`, which means the
`debug_assert!`s that factors carry are *compiled out* of `cargo test`. That
is why the registration-time checks are real typed errors rather than
assertions: they are the only shape checks that survive into the test suite.

## 8. Worked example

A consumer that wants to know *why* a window solve failed:

```rust
match optimizer.optimize(&mut problem) {
    Ok(result) => tracing::info!("cost {}", result.final_cost),
    Err(e) => {
        tracing::error!("{}", e.chain_compact());
        // "linear solve failed → factorization of the normal equations
        //  failed, the matrix is singular or not positive definite → …"
    }
}
```

The chain tells the story in one line: the *optimizer* failed, because the
*linear algebra* failed, because the *matrix was singular*. The hops are the
difference between "the solve failed" and "the factorization of a
rank-deficient normal matrix failed".

## 9. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a typed error is not a diagnosis</span>

The hierarchy tells you *where* a failure came from, not *why the model was
wrong*. A `CoreError::InvalidInput` at registration means the graph and the
factor disagree — but only the factor's own test says which of the two is
wrong. And a successful solve reports no error at all, however wrong the
residual is. The error machinery is about *reporting*, not about
*correctness*, and the two are not the same discipline.

</div>

<div class="apx-note">
<span class="apx-title">Note — the crate never installs a subscriber for you</span>

`init_logger` is opt-in. A library that called it would take the subscriber
away from the host application, and the level would be a global decision. The
crate's binaries call it; a consumer's binary calls it if it wants the same
format; a library that embeds the solver does not.

</div>
