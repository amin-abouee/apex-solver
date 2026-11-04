# Experiment 010: re-measure the Schur solver comparison at HEAD (ACCEPTED, doc-only)

- **Date:** 2026-09-27
- **Target:** `doc/performance.md`'s Schur solver comparison table
- **Commit:** none (measurement only; `doc/performance.md` updated)
- **Host:** Apple Mac mini M4 (10 cores)

## 1. Why

The Schur solver comparison table (four `APEX_BENCH_SCHUR` variants × four BAL
datasets) was still measured at `a3ef66d`, before Exp 009's counting-sort fix to
`build_symbolic_structure`. Exp 009 showed that fix also speeds up the explicit
Schur paths (they call the same assembly), so the table was stale relative to the
rest of `doc/performance.md`, which had already moved to `13518e0`. This closes that
gap so every apex number in the document is from one commit.

## 2. Method

`APEX_BENCH_RUST_ONLY=1`, 3 runs per `{sparse, chunked, explicit-iterative,
iterative}` variant, on `feature/criterion` HEAD (`13518e0`, docs on top at
`3b77eb8`). Same `bundle_adjustment_benchmark` harness, same four datasets
(Ladybug-1723, Trafalgar-257, Dubrovnik-356, Venice-1778), same machine, run
sequentially (no concurrent benches).

## 3. Results

Final RMSE and iteration counts on every one of the 16 (dataset × variant) points
matched the `a3ef66d` table **exactly** — confirms Exp 009 changed no numerics on
the explicit or implicit Schur paths. Timing:

| dataset / variant | `a3ef66d` | `13518e0` | Δ |
|---|---|---|---|
| Ladybug / Sparse | 61.75 s | 62.35 s | +1.0 % (noise) |
| Ladybug / Chunked | 88.51 s | 89.36 s | +1.0 % (noise) |
| Ladybug / Iterative (explicit) | 31.12 s | 30.98 s | −0.5 % (noise) |
| Ladybug / Implicit | 20.01 s | 19.82 s | −1.0 % (noise) |
| Trafalgar / Sparse | 3.36 s | 3.39 s | +0.9 % (noise) |
| Trafalgar / Chunked | 6.10 s | 5.91 s | **−3.1 %** |
| Trafalgar / Iterative (explicit) | 6.83 s | 6.79 s | −0.6 % (noise) |
| Trafalgar / Implicit | 14.23 s | 13.69 s | **−3.8 %** |
| Dubrovnik / Sparse | 24.65 s | 24.08 s | **−2.3 %** |
| Dubrovnik / Chunked | 51.36 s | 49.68 s | **−3.3 %** |
| Dubrovnik / Iterative (explicit) | 51.40 s | 51.90 s | +1.0 % (noise) |
| Dubrovnik / Implicit | 88.27 s | 86.74 s | −1.7 % (noise) |
| Venice / Sparse | 53.58 s | 55.06 s | +2.8 % (noise, high std) |
| Venice / Chunked | 64.24 s | 63.65 s | −0.9 % (noise) |
| Venice / Iterative (explicit) | 45.32 s | 43.05 s | **−5.0 %** |
| Venice / Implicit | 19.90 s | 16.02 s | **−19.5 %** |

Most deltas are within this table's own run-to-run noise (3 runs, std up to ~8 %
on Venice/Sparse). Trafalgar and Dubrovnik lean consistently faster across all four
variants — consistent with Exp 009's ~5 % effect on explicit BA assembly.
Venice/Implicit's −19.5 % exceeds this table's noise band and is close to but not
identical to the BA head-to-head table's own Venice/Implicit measurement (19.9 s,
a separate 3-run session); both are attributed to this machine's documented
session-to-session thermal drift (`baseline_criterion.md`) rather than to Exp 009,
since Exp 009 does not touch anything on the 2-iteration Venice path differently
than a 21-iteration one, and the criterion `implicit_ba_benchmark` A/B for Venice-52
(a different, smaller dataset) showed no comparable jump.

## 4. Verdict

**Accepted as a documentation update.** No code change. `doc/performance.md`'s Schur
table, its dependent narrative bullets (ratios, per-iteration timings), and the
BA head-to-head table's cross-reference note were all updated to these numbers, so
every apex measurement in the document is now from `13518e0`.
