# apex-solver — architecture findings

Every finding: **Evidence** (where to look), **Why it matters**, **Proposal**,
**Gate** (what must hold before it lands).

---

## S-A1 · One optimizer loop, not three · P1 · L

**Evidence**
- `optimize_with_mode` exists three times: `src/optimizer/levenberg_marquardt.rs:1141`
  (233 lines), `src/optimizer/dog_leg.rs:1320` (226), `src/optimizer/gauss_newton.rs:691`
  (193). The duplicate detector finds 355 shared 8-line windows between the three
  files (dog_leg/gauss_newton 184, dog_leg/LM 92, GN/LM 79).
- The shared skeleton is the same each time: counters, maxima and
  `IterationStats` bookkeeping; `iteration_preamble`; observer notification;
  `ConvergenceParams` (16 fields, rebuilt every iteration); debug summary;
  covariance; `build_solver_result`.
- The extracted helpers pay for the copying with long signatures, each carrying
  `#[allow(clippy::too_many_arguments)]`: `create_optimizer_summary` (17
  parameters, `optimizer/mod.rs:1196`), `notify_observers` and
  `notify_observers_generic` (9 each, `:974`, `:1003`; the second ignores its
  `_linear_solver` argument), `build_solver_result` (9, `:1044`),
  `iteration_preamble` (7, `:819`).
- 15 `#[allow(deprecated)]` in these files exist only because the deprecated
  Ceres-compatibility config fields are still read internally.

**Why it matters.** A fix to convergence, stall detection or observer timing
has to be made three times and verified three times. Copies already differ: the
stall rule (`consecutive_rejected` plus saturated damping,
`levenberg_marquardt.rs:1317`) exists only in LM; the other two optimizers have
no equivalent early stop. This is also the single biggest obstacle to S-P1 (warm-started solves).

**Proposal**
1. `SolveTracker` struct owning the counters, maxima, `iteration_stats`,
   timers and `previous_cost`. `tracker.record(&step, &eval)`,
   `tracker.summary(status) -> OptimizerSummary`. Deletes the 17-parameter
   function and most of the 9-parameter ones.
2. `trait StepStrategy<M: AssemblyBackend>` with three methods:
   `propose(&mut self, residual, jacobian, solver) -> Proposal`,
   `judge(&mut self, proposal, state, problem) -> StepVerdict` (apply or
   revert, update λ / trust radius) and `extra_stop(&self, tracker) -> Option<Status>`
   (LM's stall rule; GN returns `None`).
3. One generic `fn run<S: StepStrategy<M>>(...)` loop. `LevenbergMarquardt`,
   `DogLeg` and `GaussNewton` keep their public types and configs and only
   implement `StepStrategy`.
4. Split `ConvergenceParams` into an immutable `Tolerances` (built once from
   config) and a per-iteration `IterationFacts` (4 numbers).
5. Remove the deprecated config fields (one minor release of `#[deprecated]`
   first, but note the repo forbids `#[allow]`, so the internal reads must go
   first), then the 15 `allow(deprecated)`.

**Gate.** `ba_benchmark` golden-cost guard passes unchanged for every dataset
and both Schur variants; g2o pose-graph final costs identical to the last
digit (the loop order of floating-point operations must not change: keep the
same operation sequence); criterion within noise; apex-vio EuRoC/Hilti
trajectories byte-identical.

---

## S-A2 · `Problem` is a god object · P1 · L

**Evidence.** `src/core/problem.rs` (1,997 lines, 968 of them production code;
the rest is tests). One struct (`:28`) owns the
variable store, the residual blocks, fixed indices, bounds, Schur-landmark
marks, manifold types, row-layout dirtiness, the Jacobian mode and the
evaluation mode. Its methods mix four concerns: model building
(`add_*`, `remove_*`), constraints (`fix_variable`, `set_variable_bounds`),
evaluation (`compute_residual_and_cost_sparse`,
`compute_residual_and_jacobian_{sparse,dense}`) and diagnostics
(`log_residual_to_file`, `log_sparse_jacobian_to_file`, `log_variables_to_file`).
Three public methods panic on bad input (`add_residual_block_with_noise`,
`fix_variable`, `set_variable_bounds`: `unwrap_or_else(|e| panic!(...))`).
`log_sparse_jacobian_to_file` writes only the header: it is a stub under a
name that promises a matrix. None of the `log_*_to_file` methods uses `self`.

**Why it matters.** Anything that touches one concern recompiles and
re-reviews the others; the evaluation path cannot be unit-tested without
building a whole `Problem`; the public surface is hard to keep stable.

**Proposal.** Keep `Problem` as the façade and move state behind it:
- `VariableStore` (variables, manifold types, fixed mask, bounds).
- `BlockStore` (residual blocks, row layout, dirty flag, `compact_rows`).
- `Evaluator<'p>` (borrows both; owns `compute_*`, the workspace
  and the cost-only path from S-P2).
- `debug` module: the three `log_*_to_file` as free functions taking the data
  they print. Either write the real triplets in `log_sparse_jacobian_to_file`
  or delete it.
- The panicking wrappers become `try_*` only (see S-E1); keep a thin
  `add_residual_block` that returns `Result`.

**Gate.** No behaviour change: the complete test suite and the same golden costs.
This is a move-and-rename refactor and should land as several small commits
(one concern per commit).

---

## S-A3 · Fixed-index and bounds state exist twice · P2 · M

**Evidence.** `Problem.fixed_variable_indexes: SecondaryMap<VarKey, HashSet<usize>>`
and `Problem.variable_bounds` (`core/problem.rs:33-34`) are the source of truth;
each `Variable` has its own `fixed_indices: HashSet<usize>` and
`bounds: HashMap<usize,(f64,f64)>` (`core/variable.rs:123-125`). Every solve
clones the whole variable map (`optimizer/mod.rs:608`) and then copies both
containers back into each variable (`apply_constraints_to_variables`,
`core/problem.rs:730`, with `indexes.clone()` per variable). The public trait
`ManifoldVariable` exposes `&HashSet<usize>` / `&HashMap` (`variable.rs:327-330`),
so the container choice is part of the API.

**Why it matters.** Two owners can disagree; hash containers allocate and
hash on a path that runs per solve for every constrained variable; the trait
leaks an implementation detail.

**Proposal.** One owner (the variable). Represent "fixed" as a small bitmask
(`u16` covers every group; `SmallVec<[usize; 6]>` is the fallback for `Rn`),
expose it as `fn is_fixed(&self, idx) -> bool` and `fn free_dof(&self)`.
Remove the problem-level copy and `apply_constraints_to_variables`.

**Gate.** `ISSUE-0003` regression tests (fixed DOFs solved as free) stay green;
no change in solver output.

---

## S-A4 · Variable bounds: accepted, stored, then rejected at solve time · P1 · M

**Evidence.** `Problem::set_variable_bounds` validates and stores. Then
`initialize_optimization_state` (`optimizer/mod.rs:588`) scans every variable
and returns `InvalidConstraint` if any bound is non-trivial (comment:
"no optimizer currently enforces variable bounds ... ISSUE-0004"). So the
public API offers a feature that can only fail at the last moment.

**Proposal (pick one, in this order of preference).**
1. Implement box constraints for the trust-region optimizers: after computing
   the tangent step, clamp the post-retraction value per component (projected
   LM), and treat active components as fixed for the next normal equations.
   Only meaningful for `Rn` components and Euclidean-chart groups; reject
   bounds on rotation components at registration with a typed error.
2. If there is no consumer (none in apex-cv or apex-vio today), delete
   `set_variable_bounds`, `variable_bounds`, the bounds half of `Variable`
   and ISSUE-0004. Fewer states, smaller API.

**Gate.** Option 1 needs a constrained least-squares benchmark with a known
optimum on the boundary; option 2 needs a semver note.

---

## S-A5 · Four overlapping "group acts on a point" helpers · P2 · M

**Evidence.**
- `src/factors/lidar/action.rs`: `affine_action` plus a thread-local,
  32-entry `Rc<Action>` cache keyed by `TypeId` and the exact parameter
  vector; `MAX_DOF = 9`.
- `src/factors/common/group_point.rs`: `act_point`, `into_anchor`,
  `relative_point`; `MAX_ACT_DOF = 9`.
- `src/factors/lidar/map_plane.rs`: `lidar_point_chain`, `project_state`.
- `src/factors/offset.rs`: `MAX_DOF = 16`; `inverse_depth_plane`/`photometric`
  `MAX_COLUMNS`.

All four answer "carry a point through a pose group and give the Jacobian in
the group's tangent chart".

**Why it matters.** The same maximum dimension is stated in four places; a
future group with a larger tangent has to be found by grep. The thread-local
cache is never cleared and its hit-rate is not measured.

**Proposal.**
- One module `factors/common/point_action.rs` owning `MAX_ACT_DOF`
  (derived from `ConstDoF`, with a `const` assertion per supported group).
- Express `affine_action` (the 4-evaluation trick) as a strategy of that
  module and benchmark it against plain `act_point` per point; keep the cache
  only where it wins, document its lifetime, and add a `clear` hook called at
  solve start.
- `lidar_point_chain` becomes a thin caller.

**Gate.** `tests/allocation_free_wrappers.rs` stays at zero; LiDAR factor
Jacobians conform under `factors/common/conformance.rs`; apex-vio LIO output
byte-identical.

---

## S-A6 · Two logger modules; library crates own a global subscriber · P2 · S

**Evidence.** `src/logger.rs` (373 lines) and `crates/apex-io/src/logger.rs`
(286) are copies with different doc comments (28 shared windows).
`tracing-subscriber` and `chrono` are unconditional dependencies of the
`apex-solver` library (`Cargo.toml:[dependencies]`), only so that a library
can install a global subscriber.

**Why it matters.** A library should not initialise global state; consumers
(apex-vio) pull `tracing-subscriber` and `chrono` for nothing. Two copies drift.

**Proposal.** Keep one `init_logger*` family, in a feature `logging`
(on for the CLI/examples, off by default), living in `apex-io` or a tiny
`apex-log` helper that both depend on. The libraries then depend on `tracing`
only.

**Gate.** `cargo tree -p apex-solver --no-default-features` loses
`tracing-subscriber` and `chrono`.

---

## S-A7 · Visualization module is 1,950 lines with four 120+ line methods · P2 · M

**Evidence.** `src/observers/visualization.rs`: `log_final_state` (154 lines,
`:1019`), `log_manifolds` (131, `:1533`), `log_initial_state` (124, `:1408`),
`log_initial_ba_state` (122, `:886`); the file duplicates itself in 63
windows. It is behind the `visualization` feature, so the cost is
maintenance, not compile time for most users.

**Proposal.** One submodule per entity (`pose`, `landmark`, `cost_plot`,
`manifold`, `matrix`) with a shared `EntityPath` builder, and table-driven
logging (a slice of `(label, color, data)`) instead of repeated blocks.
Hide `rerun` types behind the module so a rerun upgrade touches one file.

**Gate.** Visual regression is manual; add a unit test that the module builds
the expected entity paths for a synthetic problem.

---

## S-A8 · CLI binaries in the library repo; `cli` and `download` are default features · P1 · M

**Evidence.**
- `bin/pose_graph_g2o.rs` (1,260 lines; `test_se2_dataset` 388 lines and
  `test_se3_dataset` 354 lines are near-copies, 168 shared windows),
  `bin/bundle_adjustment.rs` (594; `run_bundle_adjustment` 256),
  `crates/apex-io/bin/*` (7 binaries, 3,002 lines, about 100 `println!`).
- `Cargo.toml`: `default = ["cli", "download"]` in both `apex-solver` and
  `apex-io`. In the apex-vio dependency tree this brings `clap`, `ureq`,
  `bzip2`, `flate2` and `tar` (`cargo tree -i ureq` leads to `apex-io` leads
  to `apex-solver`). apex-cv already opts out (`default-features = false`);
  apex-vio does not.

**Why it matters.** Consumers compile and ship an HTTP client and archive
tooling to run an optimizer. The binaries and their tests carry 12 `print_*` lint suppressions, which the no-suppression rule forbids.

**Proposal.**
1. `default = []` in both crates; keep `cli`/`download` as opt-in.
2. Move the 9 binaries into a new workspace member `apex-cli` with
   `publish = false`. The library crates keep only `examples/` that need no
   extra dependencies.
3. In the new crate, collapse the dataset runners into one table-driven
   function (`DatasetSpec { name, kind, expected_cost }`), which removes the
   two 350-line functions.

**Gate.** `cargo build -p apex-solver --no-default-features` and a downstream
`cargo tree` in apex-vio show no `ureq`, `clap`, `tar`, `bzip2`, `flate2`.
Needs a coordinated one-line change in apex-vio's `Cargo.toml` if it relied
on a default feature (it does not appear to).
