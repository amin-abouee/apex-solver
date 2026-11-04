# Experiment 008: row-parallel F·v and E·u in the implicit Schur operator (ACCEPTED)

- **Date:** 2026-09-27
- **Target:** bundle adjustment, `ImplicitSparseSchur` (the library's BA default)
- **Commit:** `69f717e`
- **Host:** Apple Mac mini M4 (10 cores)

## 1. Hypothesis

Every PCG iteration applies `S·v = Fᵀ(F·v − E·(EᵀE+λD_e)⁻¹·Eᵀ·F·v) + λD_k·v` from
`J`. Two of its four passes, `y = F·v` and `y ← y − E·u`, were **serial**
column-by-column scatters (each column writes rows other columns also write),
while the two gathers were already parallel.

Temporary `Instant` instrumentation of `apply_schur_operator` on trafalgar-257
(21 LM iterations, most with a 200-iteration PCG solve) — per LM iteration:

| stage | time | parallel? |
|---|---|---|
| `y = F·v` | 407 ms | no |
| `t = Eᵀ·y` | 70 ms | yes |
| `u = H_ee⁻¹·t`, `y ← y − E·u` | 259 ms | no |
| `S·v = Fᵀ·y + λD_k·v` | 142 ms | yes |

The serial scatters were 75 % of the operator and ~70 % of the whole LM iteration.

## 2. Implementation

Split both scatters over **fixed row chunks** instead of columns. Each chunk owns
its slice of `y`. At structure-build time (once per sparsity pattern) the solver
records, per chunk, the `[lo, hi)` value range of every column that touches it —
pushed in exactly the order the column-wise scatter visited columns. Each chunk
replays that list, so every row receives the same additions in the same sequence:
**bit-identical output**. `u = (EᵀE+λD_e)⁻¹·t` becomes a per-entry parallel gather.

Cost: one 12-byte `ColumnSpan` per (column, row-chunk) intersection; `u32` indices,
with an explicit error if a Jacobian exceeds that range. Chunks = 8 × threads,
minimum 4096 rows.

## 3. Verification

- **Unit test** `row_chunked_operator_is_bit_identical_to_column_scatter`: the old
  column scatter is kept verbatim as a reference; bitwise equality for chunk
  lengths 1, 2, 3, 5, 7, `nrows`, `nrows + 4` on an irregular Jacobian whose rows
  touch two retained blocks (so addition order matters).
- `cargo test --release --workspace`: 2282 passed, 0 failed. fmt / clippy
  `-D warnings` clean.
- **Solver datasets:** `bundle_adjustment -s implicit` on trafalgar-21,
  trafalgar-257, dubrovnik-356, ladybug-1723, venice-1778 — iterations, initial
  cost, final cost and RMSE identical to the pre-change build on all five; the
  trafalgar-257 per-iteration LM trace (cost, gradient, step, ρ, λ) is identical
  line by line.

## 4. Results

Criterion `implicit_ba_benchmark`, pre (`feature/criterion` before this change)
vs post, interleaved, golden guards on:

| id | pre | post | Δ |
|---|---|---|---|
| trafalgar-21 | 2.365 s | 1.395 s | −41.0 % |
| trafalgar-257 | 17.74 s | 12.90 s | −27.3 % |
| venice-52 | 19.82 s | 14.82 s | −25.2 % |
| dubrovnik-135 | 42.85 s | 32.21 s | −24.8 % |

All four intervals disjoint. On the full `doc/performance.md` problems (3 rounds):
Ladybug −15.6 %, Trafalgar −25.6 %, Dubrovnik −31.1 %, Venice −4.5 % (2 LM
iterations only), accuracy identical — see `baseline_vs_current_m4.md` §2.

## 5. Verdict

**Accepted.** Bit-identical, faster on every BA dataset, no effect on other
solvers (odometry and the explicit Schur paths do not use this operator).
Remaining operator time is the two gathers (`Eᵀ·y`, `Fᵀ·y`), both already
parallel and bandwidth-bound.
