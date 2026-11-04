# Experiment 009: the odometry cost of Exp 004, and a counting-sort CSC plan (ACCEPTED)

- **Date:** 2026-09-27
- **Target:** Jacobian assembly setup (`build_symbolic_structure`) — shared by every
  solver; the regression showed on pose graphs
- **Commit:** `13518e0`
- **Host:** Apple Mac mini M4 (10 cores)

## 1. The regression

`baseline_vs_current_m4.md` found odometry *not* faster than `true_baseline` —
parking-garage +2.1 %, sphere2500 +2.6 % on the criterion suite (disjoint intervals).
Of the commits since `84e609b`, only `28e910c` (Exp 004, CSC gather plan) changes
runtime code a pose-graph solve executes.

**Localized** — criterion odometry ids on four revisions, run in both orders
(A B C D, then D C B A):

| id | `84e609b` | `e34a117` | `28e910c` | HEAD | step at `28e910c` (pass 1 / 2) |
|---|---|---|---|---|---|
| parking-garage | 98.05 / 97.21 | 98.26 / 97.89 | 99.86 / 99.65 | 99.80 / 99.76 ms | **+1.6 % / +1.8 %** |
| sphere2500 | 166.3 / 163.7 | 166.6 / 164.7 | 169.5 / 165.8 | 169.2 / 166.5 ms | +1.7 % / +0.6 % |
| M3500 | 64.18 / 64.20 | 64.83 / 64.70 | 62.52 / 63.79 | 63.19 / 62.62 ms | −3.6 % / −1.4 % |
| torus3D | 1.301 / 1.277 | 1.308 / 1.274 | 1.291 / 1.251 | 1.274 / 1.256 s | −1.3 % / −1.8 % |

`28e910c` is the only step that moves: it costs the graphs that converge in 2–5 LM
iterations and helps the ones that iterate longer — a one-time cost up, a
per-iteration cost down.

**Proved** — temporary `Instant` instrumentation around `build_symbolic_structure`
(once per solve) and `assemble_sparse` (per linearization), `pose_graph_g2o`, 3 runs:

| graph | build `e34a117` | build HEAD | assembly/call `e34a117` | assembly/call HEAD |
|---|---|---|---|---|
| parking-garage | 4.13 ms | **14.53 ms** | 2.61 ms | 1.50 ms |
| sphere2500 | 2.66 ms | **11.01 ms** | 2.20 ms | 1.22 ms |
| torus3D | 4.94 ms | **20.90 ms** | 3.60 ms | 1.76 ms |
| M3500 | 0.90 ms | **3.01 ms** | 0.76 ms | 0.51 ms |

Exp 004 halved assembly but made the structure build 3–4× slower: a stable
comparison sort over every `(col, row, arena_index)` Jacobian triple.

## 2. Fix

The sort is unnecessary. Blocks are visited in ascending row order and each pushes
its rows ascending, so a **stable counting sort by column** (bucket placement in push
order) already yields rows ascending within each column and duplicate pairs in push
order — exactly the comparison sort's output, hence the same pattern, destinations
and duplicate summation order. A per-column `is_sorted` check falls back to a stable
sort if that invariant ever breaks. Also: the free-column list is hoisted out of the
per-row loop, and the triple buffer is reserved per block.

Structure build after the fix: parking-garage 3.76 ms, sphere2500 2.90, torus3D 5.24,
M3500 1.01 — back to pre-Exp-004 cost, with Exp 004's faster assembly kept (and a
little faster still: 1.32 / 1.12 / 1.68 / 0.48 ms per call).

## 3. Verification

- Test `counting_sort_plan_matches_stable_comparison_sort`: the old comparison-sort
  construction kept as the reference; equal pattern, destinations and duplicate
  order on row-ordered input with duplicated variables, and on shuffled input (the
  fallback path); out-of-range columns are rejected.
- `cargo test --release --workspace`: 2283 passed, 0 failed; fmt / clippy `-D warnings`
  clean.
- **Solver datasets:** 16 g2o graphs × {LM, GN, DL} = 48 runs, optimized graphs
  numerically bit-identical to the previous build; BAL trafalgar-21 / trafalgar-257 /
  dubrovnik-356 / ladybug-1723 / venice-1778 × {implicit, explicit}: iterations and
  costs identical on all 10.

## 4. Results

Criterion, current (`69f717e`) vs fix, interleaved, golden guards on every run:

| id | before | after | Δ |
|---|---|---|---|
| odometry / intel | 5.857 ms | 5.406 ms | −7.7 % |
| odometry / M3500 | 62.29 ms | 60.38 ms | −3.1 % |
| odometry / parking-garage | 99.25 ms | 89.59 ms | **−9.7 %** |
| odometry / sphere2500 | 165.7 ms | 156.4 ms | **−5.6 %** |
| odometry / torus3D | 1.249 s | 1.234 s | −1.2 % |
| BA explicit / trafalgar-21 sparse | 689.6 ms | 651.6 ms | −5.5 % |
| BA explicit / trafalgar-21 iterative | 708.5 ms | 672.2 ms | −5.1 % |
| BA explicit / trafalgar-257 sparse | 3.304 s | 3.074 s | −6.9 % |
| BA explicit / trafalgar-257 iterative | 6.900 s | 6.590 s | −4.5 % |
| BA explicit / venice-52 sparse | 11.29 s | 10.67 s | −5.5 % |
| BA explicit / venice-52 iterative | 11.40 s | 10.78 s | −5.5 % |
| BA explicit / dubrovnik-135 sparse | 21.09 s | 19.98 s | −5.3 % |
| BA explicit / dubrovnik-135 iterative | 21.79 s | 20.70 s | −5.0 % |
| BA implicit / trafalgar-21 | 1.416 s | 1.392 s | −1.7 % |
| BA implicit / trafalgar-257 | 13.12 s | 12.88 s | −1.9 % |
| BA implicit / venice-52 | 14.74 s | 14.38 s | −2.4 % |
| BA implicit / dubrovnik-135 | 31.86 s | 31.85 s | 0.0 % |

Large odometry harness (`odometry_pose_benchmark`, the 8 `doc/performance.md`
graphs, 5 alternating rounds): parking-garage 48.3 → 38.9 ms (−19.6 %), M3500
−8.0 %, cubicle −7.5 %, mit −7.2 %, city10000 −6.8 %, sphere2500 −4.9 %, ring
−2.7 %, torus3D −0.9 %; final cost and iterations identical on all 8.

## 5. Verdict

**Accepted.** Exp 004's odometry regression is gone and every id is faster at
identical numerics. Lesson for the program: a per-solve setup cost has to be timed
on problems that converge in a few iterations, not only per iteration — Exp 004
was accepted on per-iteration BA numbers and never timed on odometry.
