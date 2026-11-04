# Baseline vs current — criterion suite and the large datasets (Mac mini M4)

- **Date:** 2026-09-27
- **Host:** Apple Mac mini M4 (10 cores, 32 GB), macOS, rust 1.98.0
- **baseline:** `84e609b` — the optimization program's `true_baseline` (before
  every experiment).
- **current:** `feature/criterion` at `69f717e` — Exp 001/003/004/006, the Round-2
  audit fixes, and the row-parallel implicit Schur operator (Exp 008).
- **Why a new baseline table:** `baseline_criterion.md` was measured on the
  i7-11800H Linux laptop; absolute numbers from this machine are not comparable to
  it. Both revisions here were built and measured on the same machine in one
  session, with **identical harness files** (HEAD's `benches/*.rs` copied into the
  baseline worktree), so only library code differs.
- **Accuracy:** solver numerics are identical between the two revisions on every
  dataset below (golden guards passed on every criterion run; the large-dataset
  runs match final cost / RMSE and iteration count exactly).

## 1. Criterion suite (`odometry_benchmark`, `ba_benchmark`, `implicit_ba_benchmark`)

Per id: 15 s warm-up profile of each revision, then `--save-baseline base`
(baseline) immediately followed by `--save-baseline cur` (current). Criterion
median with its 95 % interval. Acceptance threshold: max(2 %, 2× noise).

### Odometry (LM + sparse Cholesky)

| id | baseline | current | Δ | verdict |
|---|---|---|---|---|
| intel | 5.791 ms [5.783, 5.799] | 5.845 ms [5.815, 5.885] | +0.9 % | noise |
| M3500 | 64.13 ms [64.10, 64.16] | 62.19 ms [62.12, 62.26] | −3.0 % | faster |
| parking-garage | 97.71 ms [97.54, 98.00] | 99.72 ms [99.50, 100.20] | **+2.1 %** | slower |
| sphere2500 | 161.8 ms [161.4, 162.2] | 166.1 ms [165.8, 166.4] | **+2.6 %** | slower |
| torus3D | 1.271 s [1.270, 1.273] | 1.253 s [1.251, 1.256] | −1.4 % | noise |

### Bundle adjustment — explicit Schur

| id | baseline | current | Δ |
|---|---|---|---|
| trafalgar-21 / sparse | 857.4 ms | 691.2 ms | **−19.4 %** |
| trafalgar-21 / iterative | 878.8 ms | 708.8 ms | **−19.3 %** |
| trafalgar-257 / sparse | 3.907 s | 3.317 s | **−15.1 %** |
| trafalgar-257 / iterative | 8.036 s | 6.918 s | **−13.9 %** |
| venice-52 / sparse | 11.98 s | 11.37 s | **−5.1 %** |
| venice-52 / iterative | 12.18 s | 11.46 s | **−5.9 %** |
| dubrovnik-135 / sparse | 21.80 s | 21.34 s | −2.1 % |
| dubrovnik-135 / iterative | 22.37 s | 22.16 s | −0.9 % (intervals overlap) |

### Bundle adjustment — implicit (matrix-free) Schur, the library default

| id | baseline | current | Δ |
|---|---|---|---|
| trafalgar-21 | 2.403 s | 1.399 s | **−41.8 %** |
| trafalgar-257 | 18.01 s | 12.83 s | **−28.8 %** |
| venice-52 | 20.45 s | 14.64 s | **−28.4 %** |
| dubrovnik-135 | 43.43 s | 32.22 s | **−25.8 %** |

## 2. Large datasets (the `doc/performance.md` problems)

The comparison harnesses `bundle_adjustment_benchmark` (library default =
`ImplicitSparseSchur`, full BAL problems) and `odometry_pose_benchmark` (8 pose
graphs), `APEX_BENCH_RUST_ONLY=1`, apex rows only. Baseline and current
alternate run by run (odometry 5 rounds, each already the mean of 5 inner solves;
BA 3 rounds).

### Odometry — apex-solver (LM), 5 base / 5 cur runs

| dataset | baseline (ms) | current (ms) | Δ | final cost base → cur | iters base → cur | accuracy |
|---|---|---|---|---|---|---|
| M3500 | 37.8 ± 5.1 | 38.0 ± 6.3 | +0.5 % | 1.523814e0 → 1.523814e0 | 6 → 6 | identical |
| mit | 10.4 ± 0.1 | 10.0 ± 0.3 | -4.3 % | 4.996966e1 → 4.996966e1 | 15 → 15 | identical |
| city10000 | 114.1 ± 2.9 | 118.8 ± 5.8 | +4.1 % | 4.432983e0 → 4.432983e0 | 5 → 5 | identical |
| ring | 2.6 ± 0.1 | 2.6 ± 0.1 | -0.5 % | 3.017645e-2 → 3.017645e-2 | 5 → 5 | identical |
| sphere2500 | 136.0 ± 6.6 | 139.7 ± 5.0 | +2.7 % | 3.491247e1 → 3.491247e1 | 5 → 5 | identical |
| parking-garage | 45.9 ± 6.2 | 51.2 ± 5.3 | +11.7 % | 6.280947e-1 → 6.280947e-1 | 2 → 2 | identical |
| torus3D | 2028.8 ± 51.4 | 1960.2 ± 60.0 | -3.4 % | 1.248790e2 → 1.248790e2 | 38 → 38 | identical |
| cubicle | 360.6 ± 15.0 | 386.9 ± 32.4 | +7.3 % | 9.349116e0 → 9.349116e0 | 5 → 5 | identical |

### Bundle adjustment — apex-solver implicit (matrix-free) Schur, 3 base / 3 cur runs

| dataset | cameras / points / obs | baseline (s) | current (s) | Δ | final RMSE base → cur | iters base → cur | accuracy |
|---|---|---|---|---|---|---|---|
| Ladybug | 1,723 / 156,502 / 678,718 | 23.71 ± 1.23 | 20.01 ± 0.22 | -15.6 % | 0.874681 → 0.874681 | 21 → 21 | identical |
| Trafalgar | 257 / 65,132 / 225,911 | 19.12 ± 0.31 | 14.23 ± 0.44 | -25.6 % | 0.772831 → 0.772831 | 21 → 21 | identical |
| Dubrovnik | 356 / 226,730 / 1,255,268 | 128.20 ± 3.36 | 88.27 ± 0.04 | -31.1 % | 0.743193 → 0.743193 | 21 → 21 | identical |
| Venice | 1,778 / 993,923 / 5,001,946 | 20.83 ± 1.51 | 19.90 ± 0.41 | -4.5 % | 0.745052 → 0.745052 | 2 → 2 | identical |

## 3. Reading

- **Bundle adjustment gets faster everywhere it iterates.** The matrix-free default
  gains 16–31 % on the full Ladybug / Trafalgar / Dubrovnik problems and 26–42 % on the
  criterion subsets, all from Exp 008. Venice gains only 4.5 %: it stops after 2 LM
  iterations (parameter-tolerance test, see `doc/performance.md`), so few operator
  applications are left to speed up. The explicit path gains 13–19 % on the small and
  medium sets from Exp 001/003/004/006, and 1–5 % on the largest ones.
- **Odometry did not improve.** The criterion suite, which has tight intervals, shows
  parking-garage +2.1 % and sphere2500 +2.6 % (disjoint intervals, just past the 2 %
  gate); the large-dataset harness is noisier (±5–30 ms) but leans the same way on
  parking-garage, city10000, sphere2500 and cubicle. The only commit in the range on the
  shared odometry path is `28e910c` (Exp 004, CSC gather assembly) — investigated in
  `experiment_009_odometry_assembly_regression.md`.

## 4. Reproducing

```bash
git worktree add ../apex-solver-base 84e609b   # then copy HEAD's benches/*.rs into it
# criterion, per id, interleaved:
BASE --bench <id> --save-baseline base ; CUR --bench <id> --save-baseline cur
# large datasets, alternating per round:
APEX_BENCH_RUST_ONLY=1 cargo bench --bench bundle_adjustment_benchmark   # in each worktree
APEX_BENCH_RUST_ONLY=1 cargo bench --bench odometry_pose_benchmark
```

## 5. After Exp 009 (`13518e0`): baseline vs current, odometry fixed

Exp 009 (`experiment_009_odometry_assembly_regression.md`) removed the per-solve
cost Exp 004 had added. Numerics unchanged (48 g2o runs bit-identical, 10 BAL runs
identical). Criterion medians: baseline from §1, current from the Exp 009 session
(same machine, same day, same harness; each column interleaved within its own
session).

| id | baseline `84e609b` | current `13518e0` | Δ |
|---|---|---|---|
| odometry / intel | 5.791 ms | 5.406 ms | **−6.6 %** |
| odometry / M3500 | 64.13 ms | 60.38 ms | **−5.8 %** |
| odometry / parking-garage | 97.71 ms | 89.59 ms | **−8.3 %** |
| odometry / sphere2500 | 161.8 ms | 156.4 ms | **−3.3 %** |
| odometry / torus3D | 1.271 s | 1.234 s | **−2.9 %** |
| BA explicit / trafalgar-21 sparse | 857.4 ms | 651.6 ms | **−24.0 %** |
| BA explicit / trafalgar-21 iterative | 878.8 ms | 672.2 ms | **−23.5 %** |
| BA explicit / trafalgar-257 sparse | 3.907 s | 3.074 s | **−21.3 %** |
| BA explicit / trafalgar-257 iterative | 8.036 s | 6.590 s | **−18.0 %** |
| BA explicit / venice-52 sparse | 11.98 s | 10.67 s | **−10.9 %** |
| BA explicit / venice-52 iterative | 12.18 s | 10.78 s | **−11.5 %** |
| BA explicit / dubrovnik-135 sparse | 21.80 s | 19.98 s | **−8.3 %** |
| BA explicit / dubrovnik-135 iterative | 22.37 s | 20.70 s | **−7.5 %** |
| BA implicit / trafalgar-21 | 2.403 s | 1.392 s | **−42.1 %** |
| BA implicit / trafalgar-257 | 18.01 s | 12.88 s | **−28.5 %** |
| BA implicit / venice-52 | 20.45 s | 14.38 s | **−29.7 %** |
| BA implicit / dubrovnik-135 | 43.43 s | 31.85 s | **−26.7 %** |

Large odometry harness, apex (ms, mean ± std over 5 rounds): baseline from §2,
current from the Exp 009 session.

| dataset | baseline `84e609b` | current `13518e0` | Δ |
|---|---|---|---|
| M3500 | 37.8 ± 5.1 | 34.9 ± 6.5 | -7.6 % |
| mit | 10.4 ± 0.1 | 9.5 ± 0.2 | -9.2 % |
| city10000 | 114.1 ± 2.9 | 108.3 ± 1.2 | -5.1 % |
| ring | 2.6 ± 0.1 | 2.4 ± 0.1 | -6.3 % |
| sphere2500 | 136.0 ± 6.6 | 137.1 ± 1.9 | +0.8 % |
| parking-garage | 45.9 ± 6.2 | 38.9 ± 0.5 | -15.3 % |
| torus3D | 2028.8 ± 51.4 | 1891.1 ± 3.8 | -6.8 % |
| cubicle | 360.6 ± 15.0 | 334.9 ± 1.1 | -7.1 % |

Every criterion id is now faster than `true_baseline`; on the noisier large
harness 7 of 8 graphs are faster and sphere2500 is within its spread.
