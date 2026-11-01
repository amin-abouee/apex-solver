# Error propagation plan (solver, cv, vio, model)

Spec: `error_rules.md` (primary). Taken from `ERROR_HANDLING.md` only where it
does not conflict: `#[non_exhaustive]`, named fields, `#[source]` + constructor
when a hop adds context, the diamond rules, `try_new`. Repo rules stay stricter
than both: no `unwrap`/`expect`, no `#[allow]`/`#[expect]` anywhere, tests return
`Result`.

Work lives in `/Volumes/External/Workspace/rust/error-propagation/` — one
worktree per repo on branch `error-propagation`, side by side so the path
dependencies (`../apex-solver`, `../apex-cv`, `../apex-model`) resolve to each
other's branch. Nothing is pushed.

## Decisions

**D1. One shared crate, `apex-error`** (superseded: the crate was removed and each crate keeps the two traits in its own `error.rs`).
Two unrelated bottom crates need the logging trait (apex-manifolds and
apex-model), so a copy per crate would drift. It holds:
- `ErrorLogging` with `.log()` / `.log_with_source(e)`, `#[track_caller]`, and
  the caller's `file:line` as a structured field. Its blanket impl is bounded on
  `T: std::error::Error`, **not** `Display`: every Lie group and tangent is
  `Display`, and a `Display` blanket impl makes `g.log(None)` resolve to the
  logging method (the `LieGroup::log` call breaks).
- `ErrorChainExt` with `.chain()` (multi-line) and `.chain_compact()` (`A → B → C`)
  for any `std::error::Error`, so every root error has them without a copy.
- `apex_solver::error` re-exports both; nothing else.

**D2. Root names stay the ones consumers already match on**: `ManifoldError`,
`CameraModelError`, `IoError`, `ApexSolverError` (renaming to `SolverError`
clashes with the existing `optimizer::SolverResult<T>` struct), `GpuError`,
`EvaluationError`, `VioError`. New roots: `CvError`, `ModelError`. The current
apex-io `IoError` is really the graph-file error and becomes `GraphError`; the
facade `BagError` keeps its name.

**D3. Hot paths are an explicit exception to log-at-origin.** Expected,
per-element outcomes are *not* logged and do not allocate: projection behind the
camera, unprojection outside the FOV, a lost track, a rejected RANSAC sample, a
failed per-point Cholesky in the patch tracker. Their variants carry only `Copy`
fields (no `String`, no `format!`). Logging belongs to the caller that turns
them into a decision, and only when it aborts something. Without this rule the
log floods (millions of residuals per run) and runs slow down.

**D4. Manifold and factor traits stay infallible on the hot path.** Validation
moves to construction: `try_from_param_slice`, `SO3::try_from_quaternion_wxyz`,
`try_from_axis_angle`, `Sim3::try_new`, `Problem::try_add_variable`. The
unchecked forms stay for internal use, with their precondition documented.
`Factor::validate_variables` returns `FactorResult<()>` (63 impls).
`Factor::linearize` stays `()` in this pass (98 impls, per-block cost); a single
`is_finite` check in the linearizer turns NaN residuals into
`LinearizerError::NonFiniteResidual { block }`.

**D5. Modules return their own error, never the root.** The optimizer's
`OptimizeResult` uses `OptimizerError`; apex-vio config uses `ConfigError`, the
frontend `FrontendError`, and so on. The root is returned only by top-level
entry points (`pipeline`, `app`, `Problem` facade).

**D6. No string payloads for structured facts.** `Validation(String)`,
`Internal(String)`, `Generic { message }`, `Factor(e.to_string())`,
`map_err(|_| ..)` and `.ok()?` on errors that mean something are replaced. A
`String` remains only for opaque third-party text that has no error type
(`ort::Error` is not `Send`).

**D7. Swallowed errors become a decision on record.** Each `.ok()`, `let _ =`,
`unwrap_or(default)` on a real failure becomes: propagate, or `warn!` with the
error and continue (deliberate fallback), or stays silent with a comment when it
is an expected outcome (D3).

**D8. Lints**: `print_stdout`, `print_stderr`, `dbg_macro` = "deny" in every
crate (apex-cv, apex-vio and apex-model lack them). Binaries print through
`writeln!(std::io::stdout().lock(), ..)` and return `Result`, so no `#[allow]`
is needed.

## Correctness bugs found by the audit (fixed in their crate's phase)

- apex-io g2o: `EDGE_SE3:QUAT` checks `len < 10` but reads `parts[10..31]`;
  a 10–30 token line panics (also inside rayon).
- apex-io BAL / ROS1 / MCAP: `with_capacity` and `vec![0; n]` from untrusted
  header counts (up to 4 GB, alloc abort); MCAP `cursor + len` overflows.
- apex-io ROS2 Odometry/Imu decoders fabricate frame ids, identity poses and zero
  covariance on short reads and return `Ok`.
- apex-manifolds: `from_param_slice` only `debug_assert`s the length (panic in
  release via nalgebra); `Sim3::new` asserts; `Sim3::normalize` silently resets
  a non-positive scale; `SE3Tangent::normalize` divides by zero.
- apex-camera-models: `TryFrom<&[f64]>` accepts too-long slices; a mismatched
  distortion variant projects with zero distortion (FOV divides by zero);
  `project_batch` returns `(1e6, 1e6)` sentinels; `convert` discards the real
  validation error.
- apex-solver: LM/GN re-log already-logged linear-solve errors and stringify
  them; `CoreError` flattens `LinearizerError` by hand; marginalization stringifies
  `CoreError`; `add_residual_block` panics; dead `OptimizerError` variants.
- apex-cv: every solver error discarded (`map_err(|_| ..)`); the tracker panics
  on pyramids of different depth; `image_grad` underflows at the border; GPU
  errors stringified, losing the `Overflow` fallback signal.
- apex-vio: `numerical.rs` checks `OptimizerError::MaxIterationsReached`, which
  is never constructed (always false); the initializer logs an error and `app`
  logs it again; Display strings embed `{source}` so the chain prints it twice;
  sweeps and frames dropped silently with `.ok()?`.
- apex-model: about 20 `expect`/`assert!` panics, one reachable from the public
  API (nanosam "set_image first"); `Ort(String)` used for shape and I/O errors.

## Phases (each: build + tests of every crate above it, then commit)

| # | Crate(s) | Root / module enums | Model |
|---|---|---|---|
| 0 | `apex-error` (new) | `ErrorLogging`, `ErrorChainExt` | Sonnet |
| 1 | apex-manifolds | `ManifoldError` ← `ElementError`, `TangentError`, `RnError`, `JacobianError`; checked constructors | Sonnet |
| 2 | apex-camera-models | `CameraModelError` ← `ParamError`, `ProjectionError`, `UnprojectionError`, `EstimationError`, `ConversionError` | Opus (hot path, 6 downstream matches) |
| 3 | apex-io | `IoError` ← `GraphError`, `BalError`, `TrajectoryError`, `AslError`, `LamariaError`, `DatasetError`, `VrsError`, `BagError` ← `Ros1Error`, `Ros2BagError`, `McapError`, `CdrError`, `PointCloudError`, `TranscodeError`, `DdsError` | Opus (31k lines, input-safety bugs) |
| 4 | apex-solver | `ApexSolverError` ← `CoreError`, `FactorError`, `LinearizerError`, `LinAlgError`, `CovarianceError`, `MarginalizationError`, `OptimizerError`, `ObserverError`, `PreintegrationError` | Opus |
| 5 | apex-cv, apex-gpu, apex-evaluation | `CvError` (new) ← per module; `GpuError` ← `ContextError`, `ReadbackError`, kernels; `EvaluationError` ← io, trajectory, options, metric, alignment, report | Opus |
| 6 | apex-model (+8 crates) | `ModelError` (rename of `Error`) + `XfeatError`, `AnylocError` | Sonnet |
| 7 | apex-vio | `VioError` ← `ConfigError`, `IoError`, `FrontendError`, `BackendError`, `InitializationError` (owns `NumericalError`), `LioError`, `SlamError`, `EvalError`, `ObserverError`, `PipelineError`; `src/error/` → `src/error.rs`; binaries print `.chain()` once | Opus |

Phase 6 depends only on phase 0 and can run beside phases 1–5.

## Gates per phase

1. `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`
   in the crate **and every crate above it** in the worktree folder.
2. Each root error has a test that builds a leaf error, propagates it with `?`
   through every hop to the root, and asserts `.chain()` shows each hop exactly
   once.
3. A test per fixed bug (malformed g2o line, huge BAL header, short CDR, wrong
   param length, mismatched pyramids, ...).
4. Behaviour neutral: apex-solver benchmark problems give identical costs;
   apex-vio MH_01 and room1 (seed 42) give the same ATE as `main`; ms/frame
   within noise.

## Overlap

The quality agent's uncommitted work in `apex-vio-quality` adds
`VioError::FrameSource` and edits `pipeline.rs`, `error/mod.rs`,
`frontend/mod.rs` and `backend/estimator.rs`. Phase 7 rewrites the same files
(`FrameSource` becomes `PipelineError::FrameSource(#[from] IoError)`). It is not
touched; the two branches will need a merge.
