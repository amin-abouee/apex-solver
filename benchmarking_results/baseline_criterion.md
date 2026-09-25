# Criterion Baseline — `feature/criterion` @ 84e609b

True performance baseline for the optimization loop. Recorded 2026-09-24.

## Benchmark targets

- `benches/odometry_benchmark.rs` — g2o pose graphs, LM + SparseCholesky,
  configs matching `tests/golden_values.rs`; golden final costs pinned per
  dataset (1e-6 relative) and asserted in every timed run.
- `benches/ba_benchmark.rs` — BAL, `for_bundle_adjustment()` preset; Schur
  solver variants; per-(dataset, variant) golden final costs asserted in every
  timed run (cost ≤ golden × 1.0005, RMSE ≤ golden + 0.01 px).

Datasets: odometry = M3500, intel, sphere2500, parking-garage, torus3D;
BA = trafalgar-21 (4 Schur variants), trafalgar-257 / dubrovnik-135 /
venice-52 (implicit + explicit-sparse each).

## Measurement protocol (thermal)

Host is an i7-11800H laptop, `powersave` governor with EPP=performance.
Sustained all-core load drifts >20 % with thermal state between runs (the
same trafalgar-21/schur_implicit bench measured 5.61 s cool-start vs 4.09 s
on a later cool-start run). Therefore:

- **Every benchmark session runs twice back-to-back; the first pass is the
  warm-up and is discarded, the second is recorded.** All comparisons are
  hot-steady vs hot-steady.
- Acceptance threshold for a candidate = max(2 %, 2x measured run-to-run
  noise on the warm protocol).

## Baseline numbers

Official `true_baseline` (criterion `--save-baseline`, warm protocol,
2026-09-24, all golden guards passed):

### Bundle adjustment (criterion median)

| dataset | variant | median | golden final cost |
|---|---|---|---|
| trafalgar-21 | schur_explicit_sparse | 2.198 s | 1.370013056594e4 |
| trafalgar-21 | schur_explicit_iterative | 2.117 s | 1.370020699431e4 |
| trafalgar-257 | schur_explicit_sparse | 11.99 s | 6.863327074594e4 |
| trafalgar-257 | schur_explicit_iterative | 21.92 s | 6.799201774758e4 |
| dubrovnik-135 | schur_explicit_sparse | 68.89 s | 1.888767976834e5 |
| dubrovnik-135 | schur_explicit_iterative | 70.10 s | 1.905530913190e5 |
| venice-52 | schur_explicit_sparse | 37.47 s | 9.716652695390e4 |
| venice-52 | schur_explicit_iterative | 37.36 s | 9.195236768263e4 |

### Odometry (criterion median)

| dataset | median | golden final cost |
|---|---|---|
| M3500 | 125.91 ms | 1.510940460434e0 |
| intel | 9.49 ms | 3.893032054396e-1 |
| sphere2500 | 530.87 ms | 2.129064817909e1 |
| parking-garage | 203.19 ms | 6.245093871665e-1 |
| torus3D | 4.19 s | 1.201016810957e2 |

### Historical (cool-start runs, pre-protocol — for the record only)

| dataset | variant | pinning run | baseline pass 1 |
|---|---|---|---|
| trafalgar-21 | schur_implicit | 5.61 s | 4.09 s |
| trafalgar-257 | schur_implicit | 81.2 s | 59.6 s |
| dubrovnik-135 | schur_implicit | 148.2 s | 135.1 s |
| venice-52 | schur_implicit | 59.3 s | — |
| trafalgar-257 | schur_explicit_sparse | 15.2 s | 11.8 s |

The cool-start vs warm spread (up to 35 % on identical code) is why the
2-pass protocol exists.

## Notes

- On this host `schur_implicit` (the library default) is slower than
  `schur_explicit_sparse` on **every** BA dataset — 5.3x on trafalgar-257
  (81 s vs 15 s), 2.2x on dubrovnik-135 (148 s vs 68 s), 2.0x on trafalgar-21
  (5.6 s vs 2.8 s). This inverts the reference-machine numbers quoted in
  `for_bundle_adjustment()`'s doc comment (implicit 2.2x faster in total).
  Plausible cause: the implicit operator's two serial scatter passes per PCG
  iteration don't scale with cores, while the explicit path's `JᵀJ` formation
  is fully parallel — a 16-thread host favors the parallel path. First
  profiling target: PCG iteration counts and the serial scatter kernels in
  `ImplicitSparseSchur::apply_schur_operator`.
- dubrovnik-135 implicit (148 s) is slower than the camera-larger
  trafalgar-257 implicit (81 s) despite fewer cameras — 553k observations
  vs 226k; observation-count-dominated cost in the PCG operator.
