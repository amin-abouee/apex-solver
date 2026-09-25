# Experiment 002: Structural-pattern Schur complement (REJECTED)

- **Date:** 2026-09-25 02:30
- **Target Category:** Bundle Adjustment (Global Engine — linear solver)
- **Scope Type:** Algorithm Enhancement
- **Status:** REJECTED (reverted to H1 state, 9e4d851)
- **Commit SHA:** N/A

## 1. Rationale & Approach ("Why?")

Profiling after H1 showed, per LM iteration at trafalgar-257
(schur_explicit_sparse, ≈1010 ms):

| stage | time |
|---|---|
| Jacobian+residual assembly | ~105 ms |
| JᵀJ + Jᵀr formation | ~213 ms |
| H_kk/H_ke extraction | ~196 ms → ~5 ms after H1 |
| eliminated blocks | ~11 ms |
| **S formation + reduced gradient** | **~333 ms** |
| Cholesky (symbolic ~28 ms + numeric ~50 ms) | ~78 ms |
| back-substitution | ~10 ms |
| step evaluation | ~16 ms |

Within S formation: **rank-`dof` updates ≈ 220 ms (serial)**, triplet
re-sparsification sort ≈ 54–74 ms, dense scan ≈ 17–20 ms, symmetrize ≈ 15 ms,
reduced gradient ≈ 10 ms.

Three variants were built and measured:

1. **H2a — accumulate `S` on its structural CSC pattern** (no dense buffer):
   **+62 % slower** at t257 (17.5 s vs 10.8 s). The positional scatter
   destroys the locality the dense row-strip update loop has; the O(n²)
   overheads it removes are not what dominates.
2. **H2b — parallelize the dense updates across chunk-private dense
   buffers** (pooled across iterations): **no win** — the update loop is
   bound by random 8-byte write bandwidth into a buffer larger than L3;
   16 threads split the same DRAM traffic (measured 140–320 ms with high
   variance vs 220 ms serial, and the fill+reduce overhead is additive).
3. **H2c — keep serial dense updates, output through the structural pattern**
   (no triplets/sort/symmetrize; averaged-at-read): **−4 to −7 % vs the
   original baseline, but +24 % vs H1** at t21 in an interleaved same-thermal
   A/B. Root cause: `ExtractionCache::build` gained the S-pattern
   construction, whose clique enumeration is DOF-level — ~19 rows per
   landmark → ~361 pairs per landmark → a multi-million-entry sort
   (~400 ms at t21, more at t257) that amortizes poorly over only 20 LM
   iterations.

## 2. Learnings (the valuable part)

- The dense row-strip update loop is near the memory-bandwidth roof; only a
  blocked/tiled reformulation (grouping landmarks by camera locality into
  tile-sized dense sub-problems) can beat it — a substantial algorithmic
  project, not a quick win.
- Any per-structure cache whose build cost approaches the per-iteration
  saving × iteration count (~20) loses. The S pattern is too expensive to
  build exactly; a cheaper conservative approximation would be needed.
- Cross-session timing comparisons on this laptop drift >20 % with thermal
  state. Interleaved same-window A/B (candidate ↔ baseline alternating) is
  the only trustworthy micro-comparison; the 2-pass warm protocol handles
  the full-bench case.

## 3. Criterion Benchmark Results

All variants failed the gate (≥ max(2 %, noise) improvement, no regression):
H2a +62 %, H2b ~0 % net, H2c +24 % vs H1 at the probe. Reverted; golden
guards and the 687-test lib suite stayed green throughout.

## 4. Final Verdict & Next Steps

- **Decision:** REJECTED, working tree restored to 9e4d851 (H1).
- **Next candidates** (from the profile, in value order):
  1. Tiled/blocked Schur accumulation (big project, the only real path at
     the update stage).
  2. PCG warm-starting + forcing-sequence tuning for
     `schur_explicit_iterative` (t257 iterative is 21.9 s vs 12.0 s sparse —
     PCG iteration count dominates).
  3. Jacobian assembly (~105 ms/iter) — allocation/SIMD work in the
     linearizer.
  4. Symbolic-Cholesky caching (~28 ms/iter, needs an exact-pattern guard;
     marginal vs the 2 % gate).
