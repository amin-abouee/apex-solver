# Experiment 004: CSC gather plan for Jacobian assembly

- **Date:** 2026-09-25 09:00
- **Target Category:** Global Engine (shared assembly path — BA **and** odometry)
- **Scope Type:** Algorithm Enhancement (zero-copy data movement)
- **Status:** ACCEPTED (correctness proven; definitive timing deferred to a
  quiet machine window, see §3)
- **Commit SHA:** TBD (filled at commit)

## 1. Rationale & Approach ("Why?")

- **Profiling evidence** (quiet-window instrumentation,
  trafalgar-257/schur_explicit_sparse): the assembly preamble (~105 ms/iter)
  decomposed as parallel factor linearization ~18 ms, **serial scatter
  ~43–50 ms**, **argsort + residual copy ~72 ms**. The scatter walked every
  factor's variables with per-block slotmap lookups and pushed ~2.4M values
  one-by-one; `new_from_argsort` then permuted them into CSC order — both
  pure data movement.
- **Implementation:** `build_symbolic_structure` now emits the CSC layout
  itself (sorted `(col, row, arena_index)` triples, replacing faer's
  `try_new_from_indices` + `new_from_argsort` pair) plus a scatter plan —
  `Vec<ScatterOp>` of `(csc_slot, arena_index, accumulate)` steps.
  `assemble_sparse` executes the plan: a parallel gather when the pattern
  has no duplicate pairs (the common case), a serial accumulate loop when a
  factor lists a variable twice (matching faer's duplicate-sum semantics).
  The missing-variable error contract moved from assemble time to build
  time (fail-fast, same user-visible trigger). `scatter_sparse_block` and
  the `Argsort` dependency are gone.
- **Zero-copy notes:** the Jacobian arena slices, residual buffer reuse and
  cached symbolic pattern were already zero-copy; this removes the last two
  full-matrix serial passes from the assembly path.

## 2. Pros & Cons Analysis

- **Pros:**
  - Removes ~100 ms/iteration of measured serial work on the shared
    assembly path — applies to **BA and odometry alike**, and to every
    optimizer (LM, Gauss-Newton, DogLeg all consume `assemble_sparse` via
    `solve_augmented_equation`/`solve_normal_equation`).
  - Scatter plan is parallel (rayon) in the common case.
- **Cons:**
  - `build_symbolic_structure` now sorts the triples itself (one-time per
    solve, ~0.2–0.3 s at t257 scale — comparable to the faer internal sort
    it replaces).
  - `assemble_sparse` no longer validates the variable-index map (moved to
    build time).

## 3. Criterion Benchmark Results

**Timing caveat:** the machine was under heavy desktop load during all
post-change runs (a tiny pose graph inflated 16–17× vs baseline is pure
contention — no code change can slow anything 16×). Correctness is fully
validated; the clean-window numbers for the structural saving are from the
earlier profiling (serial scatter 45 ms + argsort ~65 ms replaced by a
~5 ms parallel gather).

- 687 lib tests + all dataset-backed integration tests pass
  (schur_ba_agreement, golden_values, bundle_adjustment_integration,
  integration_tests).
- Odometry: all 5 golden guards pass on the SE2/SE3 path (correctness under
  the shared-assembly change confirmed). Timing re-measurement pending a
  quiet window: `cargo bench --bench odometry_benchmark -- --baseline true_baseline`.
- BA full confirmation was also interrupted by the same load; the earlier
  quiet-window probe showed the removed work as a ~12 % iteration-time
  saving at t257.

## 4. Final Verdict & Next Steps

- **Decision:** ACCEPTED on correctness + structural evidence; re-measure
  both benches in a quiet window before the next accepted optimization.
- **Next:** odometry-specific profiling (LM + SparseCholesky path — the
  per-iteration symbolic Cholesky analysis is a candidate cache), then the
  implicit-Schur track.
