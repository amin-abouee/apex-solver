# Errors, Logging and the Prelude

A crate that refuses to panic, forbids `unsafe`, and denies `unwrap` has to
say what it does instead. This chapter is that contract: the error hierarchy,
the propagation convention, the logging helpers, and the prelude that a
consumer normally imports.

## 1. The three-layer hierarchy

<div class="apx apx-table">
<span class="apx-title">The layers</span>

| Layer | Kind | Types | Rule |
|---|---|---|---|
| **A** — top / API | exposed to end users | `ApexSolverError` | the only type a consumer's `?` should meet |
| **B** — logic | wraps Layer C with context | `OptimizerError`, `ObserverError` | adds *where* it happened |
| **C** — deep / math | module-specific | `CoreError`, `LinAlgError`, `ManifoldError`, `FactorError`, `LinearizerError`, `MarginalizationError`, `CovarianceError`, `PreintegrationError` | must **never** return `ApexSolverError` directly |

</div>

The rule that makes the hierarchy work: a deep module returns *its own* error
type, and the `?` operator plus `#[from]` attributes convert at each layer
boundary. The example chain the source gives:

```text
LinAlgError::SingularMatrix
  → OptimizerError::LinAlg        (via #[from] at Layer B)
    → ApexSolverError::Optimizer   (via #[from] at Layer A)
```

`ApexSolverResult<T>` is the alias for `Result<T, ApexSolverError>`.

## 2. The dual path for `LinAlgError`

`LinAlgError` reaches the top level by two routes, and the crate is explicit
about which to use:

1. **Direct** — `LinAlgError → ApexSolverError::LinearAlgebra(…)` — for
   standalone linear-algebra usage, where no optimization context exists;
2. **Through the optimizer** — `LinAlgError → OptimizerError::LinAlg →
   ApexSolverError::Optimizer(…)` — during an optimization, where the
   optimizer's context is worth keeping.

A linear-algebra error that happens *inside* an optimizer should take the
second path, because the context ("the solve failed in iteration 7 of a
window solve") is what makes the report useful.

## 3. Reading the chain

`ApexSolverError` implements `std::error::Error`, so the standard chain
access works:

```rust
if let Err(e) = solver_optimize() {
    tracing::warn!("Error: {}", e);
    tracing::warn!("Full chain: {}", e.chain());
}
```

For a single line, `chain_compact()` joins the chain with `" → "`:

```text
Optimizer error → Linear algebra error → Singular matrix
```

## 4. `ErrorLogging`

The crate's one ergonomic addition to the error story:

```rust
pub trait ErrorLogging: Sized + Display {
    fn log(self) -> Self;                       // tracing::error! then self
    fn log_with_source(self, source: &dyn Debug) -> Self;
}
```

`log` returns `self`, so it composes with `?`:

```rust
let result = do_something().map_err(|e| e.log())?;
```

The point is that a failure is *reported at the point it happens*, with its
context, and the error still propagates — the crate never swallows a failure
to make a caller's life easier.

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
| the errors | `ApexSolverError`, `ApexSolverResult` |

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
    Ok(result) => println!("cost {}", result.final_cost),
    Err(e) => {
        tracing::error!("{}", e.chain_compact());
        // the chain names the layer: "Optimizer error → Linear algebra error
        // → Singular matrix"
    }
}
```

The chain tells the story in one line: the *optimizer* failed, because the
*linear algebra* failed, because the *matrix was singular*. The three layers
are not bureaucracy — they are the difference between "the solve failed" and
"the factorization of a rank-deficient normal matrix failed".

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
