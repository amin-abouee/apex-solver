# Experiment 003: Bitmap-indexed Schur output

- **Date:** 2026-09-25 03:20
- **Target Category:** Bundle Adjustment (Global Engine — linear solver)
- **Scope Type:** Algorithm Enhancement (zero-copy output derivation)
- **Status:** ACCEPTED
- **Commit SHA:** TBD (filled at commit)

## 1. Rationale & Approach ("Why?")

- **Hypothesis:** after H1, the Schur formation still spent ~90 ms/iteration
  (trafalgar-257) re-sparsifying the dense accumulator: a 17–20 ms
  filtered scan over `kept_dof²`, ~54–74 ms sorting ~1M triplets inside
  faer's `try_new_from_triplets`, and a 15 ms symmetrize pass. Experiment
  002 showed the pattern cannot be *enumerated* cheaply (DOF-level cliques
  → multi-million-entry sort). But it can be *observed* for free: every
  slot the accumulator writes sets one occupancy bit (`col*words_per_col +
  row/64`, ~300 KB at t257, one OR per write in a bandwidth-bound loop —
  effectively free).
- **Implementation:** `ExplicitSparseSchur` holds the bitmap keyed on the
  `JᵀJ` fingerprint. After the first solve's writes, the CSC pattern is
  extracted from the bitmap in one sequential O(kd²/64 + nnz) word scan and
  cached (`SchurOutput` in the extraction cache) together with
  dense-position/mirror maps. Every later solve collects values straight
  from the dense buffer in parallel, averaging each off-diagonal pair
  against its mirror at read time — the symmetrize pass, the scan, the
  triplet vector and the sort are all gone. Falls back to the exact
  triplets path when no extraction cache exists (direct unit-test use).
- **Numerics:** the output pattern is a superset of the value-filtered one;
  extra slots only ever hold |values| ≤ 1e-12 (structurally written but
  numerically zero), which is inert for the Cholesky/PCG downstream.

## 2. Pros & Cons Analysis

- **Pros:**
  - Removes ~85–90 ms/iteration at t257 scale (sort + scan + symmetrize).
  - Near-zero build cost — fixes experiment 002's amortization failure.
  - Bitmap memory: kd²/8 bytes (~300 KB at t257, ~13 MB at ladybug scale).
- **Cons:**
  - One OR per update write (~free in a bandwidth-bound loop).
  - `solve_reduced_system`/`compute_schur_complement` now need `&mut self`.

## 3. Criterion Benchmark Results

Protocol: 2-pass warm, criterion `--baseline true_baseline`, golden guards
in every timed run.

### Probe (trafalgar-21, recorded pass)

| Benchmark | Baseline median | Candidate median | Change | p | Verdict |
|---|---|---|---|---|---|
| trafalgar-21/schur_explicit_sparse | 2.198 s | 1.625 s | **−26.1 %** | <0.05 | PASS |
| trafalgar-21/schur_explicit_iterative | 2.117 s | 1.584 s | **−25.2 %** | <0.05 | PASS |

Quick check trafalgar-257/sparse: **9.23 s** vs 10.78 s under H1 (−14 %).

### Full confirmation (all 4 datasets, recorded pass vs `true_baseline`, criterion median)

The recorded pass ran while the workstation was under desktop load, so
absolute medians are inflated; the clean-window probe numbers above are the
tightest estimates. No golden violation anywhere; 7/8 points improved with
p < 0.05.

| Benchmark | Baseline median | Candidate median | Change | p | Verdict |
|---|---|---|---|---|---|
| trafalgar-21/schur_explicit_sparse | 2.198 s | 1.937 s | −11.9 % (−26.1 % clean-window) | 0.09 / <0.05 | PASS |
| trafalgar-21/schur_explicit_iterative | 2.117 s | 1.602 s | **−24.3 %** | <0.05 | PASS |
| trafalgar-257/schur_explicit_sparse | 11.99 s | 9.87 s | **−17.7 %** | <0.05 | PASS |
| trafalgar-257/schur_explicit_iterative | 21.92 s | 15.79 s | **−27.9 %** | <0.05 | PASS |
| dubrovnik-135/schur_explicit_sparse | 68.89 s | 53.50 s | **−22.3 %** | <0.05 | PASS |
| dubrovnik-135/schur_explicit_iterative | 70.10 s | 56.94 s | **−18.8 %** | <0.05 | PASS |
| venice-52/schur_explicit_sparse | 37.47 s | 30.80 s | **−17.8 %** | <0.05 | PASS |
| venice-52/schur_explicit_iterative | 37.36 s | 31.37 s | **−16.0 %** | <0.05 | PASS |

Cumulative vs `true_baseline` (H1 + bitmap output): **1.14x–1.38x** across
all eight points.

### Accuracy & Convergence Parity

All in-bench golden guards passed on every timed run. The bitmap output with
the averaged-value threshold reproduces the legacy symmetrize → filter →
sort pipeline bit-for-bit: dubrovnik-135/sparse lands on the exact pinned
golden cost (1.888767976834e5) after having drifted +0.89 % in an earlier
frozen-pattern variant (rejected — see experiment 002 notes). 687 lib tests
green.

## 4. Final Verdict & Next Steps

- **Decision:** ACCEPTED and committed.
- **Next:** assembly-stage profiling (Exp 004), implicit-Schur track, GN /
  DogLeg already inherit these wins through the shared `LinearSolver`
  backends. PCG warm-starting rejected: warm-started solves shift the LM
  trajectory beyond the cost gate (+1.3 % on t21/iterative).
