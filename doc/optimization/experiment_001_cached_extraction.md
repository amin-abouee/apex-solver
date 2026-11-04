# Experiment 001: Pattern-cached H_kk/H_ke extraction

- **Date:** 2026-09-24 23:59
- **Target Category:** Bundle Adjustment (Global Engine — linear solver)
- **Scope Type:** Algorithm Enhancement (structural caching)
- **Status:** ACCEPTED
- **Commit SHA:** 9e4d851

## 1. Rationale & Approach ("Why?")

- **Hypothesis:** `ExplicitSparseSchur` re-extracts `H_kk` (retained block)
  and `H_ke` (coupling block) from `JᵀJ` on every LM iteration by walking all
  nonzeros, pushing triplets, and re-building CSC matrices (allocate + sort).
  The sparsity pattern of `JᵀJ` is static within one optimizer run, so the
  extraction can be reduced to a value-position map built once per pattern
  and two parallel gathers per solve.
- **Profiling evidence** (temporary instrumentation, trafalgar-257 /
  schur_explicit_sparse, steady state): per LM iteration ≈ 1010 ms, of which
  extraction ≈ 196 ms (19 %), JᵀJ formation ≈ 213 ms (24 %), S formation +
  reduced gradient ≈ 333 ms (37 %), Cholesky ≈ 75 ms (7 %), Jacobian
  assembly ≈ 105 ms (11 %).
- **Implementation:** `src/linalg/sparse/schur/explicit.rs` — new
  `ExtractionCache` (per-pattern maps `H_kk`/`H_ke` value slot → `JᵀJ` value
  index, built via one sorted pass; rebuilt only when the pattern
  fingerprint changes), new `ExplicitSparseSchur::extract_kept_and_coupling`,
  both solve paths wired through it. The direct extractors remain as
  test-only references; a unit test pins the cached output to be identical
  (pattern and values) to the reference.

## 2. Pros & Cons Analysis

- **Pros:**
  - Removes ~196 ms/iteration of triplet rebuild + CSC sort + allocation on
    trafalgar-257 (the second-largest solve-stage cost).
  - Parallel gathers replace a serial walk.
  - Same building block the S-formation restructure (next experiment) needs.
- **Cons:**
  - Two `u32` position arrays sized by the extracted nnz (a few MB at
    trafalgar-257 scale); one-time build cost per pattern.

## 3. Criterion Benchmark Results

Protocol: 2-pass warm (first pass discarded), comparison via criterion
`--baseline true_baseline`, golden cost guards asserted in every timed run.

### Small probe (trafalgar-21, recorded pass)

| Benchmark | Baseline median | Candidate median | Change | p | Verdict |
|---|---|---|---|---|---|
| trafalgar-21/schur_explicit_sparse | 2.198 s | 1.657 s | **−24.6 %** | <0.05 | PASS |
| trafalgar-21/schur_explicit_iterative | 2.117 s | 1.648 s | **−22.1 %** | <0.05 | PASS |

### Full confirmation (all 4 datasets, recorded pass, criterion median)

| Benchmark | Baseline median | Candidate median | Change | p | Verdict |
|---|---|---|---|---|---|
| trafalgar-21/schur_explicit_sparse | 2.198 s | 1.658 s | **−24.6 %** | <0.05 | PASS |
| trafalgar-21/schur_explicit_iterative | 2.117 s | 1.632 s | **−22.9 %** | <0.05 | PASS |
| trafalgar-257/schur_explicit_sparse | 11.99 s | 10.78 s | **−10.1 %** | <0.05 | PASS |
| trafalgar-257/schur_explicit_iterative | 21.92 s | 19.34 s | **−11.7 %** | <0.05 | PASS |
| dubrovnik-135/schur_explicit_sparse | 68.89 s | 61.73 s | **−10.4 %** | <0.05 | PASS |
| dubrovnik-135/schur_explicit_iterative | 70.10 s | 62.64 s | **−10.6 %** | <0.05 | PASS |
| venice-52/schur_explicit_sparse | 37.47 s | 32.87 s | **−12.3 %** | <0.05 | PASS |
| venice-52/schur_explicit_iterative | 37.36 s | 32.61 s | **−12.7 %** | <0.05 | PASS |

Cumulative vs `true_baseline`: 1.11x–1.33x across all eight points.

### Accuracy & Convergence Parity

All in-bench golden guards passed on every timed run (cost ≤ golden × 1.0005,
RMSE ≤ golden + 0.01 px). Unit test `cached_extraction_matches_direct_reference`
pins the cached extraction to the direct reference exactly.

## 4. Final Verdict & Next Steps

- **Decision:** TBD after full confirmation.
- **Next:** H2 — accumulate the Schur complement directly on its structural
  pattern (removes the dense kept_dof² buffer, the O(n²) symmetrize/filter
  scans and the per-iteration triplet conversion; 37 % of iteration time).
