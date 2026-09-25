# Experiment 006: Hash-keyed symbolic Cholesky cache for explicit Schur

- **Date:** 2026-09-25 12:50
- **Target Category:** Bundle Adjustment (Global Engine — linear solver)
- **Scope Type:** Algorithm Enhancement (caching)
- **Status:** ACCEPTED
- **Commit SHA:** TBD (filled at commit)

## 1. Rationale & Approach ("Why?")

- **Hypothesis:** after Exp 003, `solve_with_cholesky` still ran
  `SymbolicLlt::try_new` (the fill-reducing ordering analysis of the reduced
  system `S`) on every LM iteration — profiled at ~25–30 ms per iteration on
  trafalgar-257 (~3–4 % of the post-Exp003 iteration). The filtered `S`
  pattern is now derived deterministically from the occupancy bitmap, so its
  identity can be *hashed cheaply during the same walk* and the symbolic
  reused whenever the hash is unchanged.
- **Implementation:**
  - `build_schur_output` additionally returns an FNV-1a hash of the CSC
    structure (column index + count + rows per column).
  - `ExplicitSparseSchur` holds `s_llt_cache: Option<(u64, SymbolicLlt)>`.
    `solve_with_cholesky` takes the hash: on a match it clones the cached
    symbolic (Arc, O(1)) and goes straight to the numeric factorization; on
    a miss (first solve, or a genuine pattern change) it rebuilds and
    re-caches. Only first-attempt symbolics are cached — the regularization
    retry path widens the pattern with synthetic diagonals and must not
    poison the cache.
  - The chunked variant passes `None` (its `S` comes from a different
    representation); its partition is now cloned where a `&mut self` call
    needed the borrow freed.
- **Numerics:** identical symbolic + identical values ⇒ bit-identical
  factorization and step; the golden guards confirm it (exact pinned costs).

## 2. Pros & Cons Analysis

- **Pros:** removes the per-iteration ordering analysis (~25–30 ms on
  t257, ~3–4 % of the iteration) with zero numeric change.
- **Cons:** the cache must be invalidated on any pattern change — guarded
  by the hash plus a defensive rebuild if `Llt::try_new_with_symbolic`
  rejects the reuse.

## 3. Criterion Benchmark Results

- dubrovnik-135/schur_explicit_sparse: exact pinned golden cost
  (1.888767976834e5) on every timed run with the cache active.
- 687 lib tests + all dataset-backed integration tests pass; clippy clean.
- Definitive timing: pending the quiet-window full re-measurement (the
  expected effect is ~3–4 % on the Sparse variant points, invisible on the
  Iterative variant, which never enters `solve_with_cholesky`).

## 4. Final Verdict & Next Steps

- **Decision:** ACCEPTED and committed.
- **Next:** the implicit-Schur track can reuse the same
  `pcg`/`pcg_from` primitive; its symbolic cost is zero by construction
  (matrix-free), so the remaining implicit candidates are the scatter
  parallelization and PCG tuning.
