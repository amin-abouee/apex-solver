# Benchmarks and Performance

Numbers tell you where a solver's time goes, and the crate measures itself
against the same public datasets the literature uses. This chapter is what
was measured, on what, and how to reproduce it.

## 1. The setup

<div class="apx-table">
<span class="apx-title">The measurement conditions</span>

| Item | Value |
|---|---|
| hardware | Apple Mac mini M4 (10 cores), 32 GB RAM, macOS |
| build | Rust 1.98 release (`opt-level=3`, LTO); C++ `-O3 -DNDEBUG -march=native` |
| solver version | apex-solver 1.5.0 |
| comparators | Ceres 2.2.0, GTSAM 4.3.0, g2o 20241228, factrs, tiny-solver (Eigen 5.0.1) |
| method | mean ± std over independent runs; apex rows 5 fresh runs, 3 per Schur configuration |
| timing scope | the `optimize()` call only — setup and metric computation excluded |
| BA timeout | 10 minutes per solver |

</div>

The metrics follow the pose-graph literature (SE-Sync, Rosen et al., *IJRR*
2019; Carlone et al., *ICRA* 2015): the final objective value and the runtime.
Bundle adjustment uses reprojection RMSE and runtime.

Two honesty notes from the crate's own performance document, reproduced here
because they are the kind of thing that quietly poisons a comparison:

- **The GTSAM rows before 2026-09-27 were wrong.** The C++ harness compiled
  GTSAM against eigen@3 while a system-Eigen GTSAM is built on Eigen 5; the
  ABI mismatch built cleanly but broke the linearization. Fixed in `14d3c82`;
  every GTSAM number in the tables is from the fixed harness.
- **Solution cost is deterministic** for a fixed input and algorithm, so its
  standard deviation is zero; the error bars reflect runtime variation only.

## 2. Pose graph optimization

Levenberg–Marquardt throughout; sparse Cholesky for the linear solve. The
cost is computed by the harness directly from the G2O file for every solver,
so the values are comparable across implementations.

| dataset | vertices | edges | apex time | final $\chi^2$ |
|---|---:|---:|---:|---:|
| ring | 434 | 459 | ~4 ms | 2.2e-2 |
| M3500 | 3 500 | 5 453 | ~60 ms | 1.5 |
| city10000 | 10 000 | 20 687 | ~110 ms | 4.4 |
| sphere2500 | 2 500 | 4 949 | ~190 ms | 21.3 |
| parking-garage | 1 661 | 6 275 | ~85 ms | 0.62 |
| torus3D | 5 000 | 9 048 | ~930 ms | 236 |
| cubicle | 5 750 | 16 869 | ~990 ms | 1 692 |

The shape of the table is the interesting part: the time does **not** scale
with the vertex count alone. `city10000` (10 000 vertices) solves in ~110 ms
while `torus3D` (5 000 vertices) takes ~930 ms — a factor of eight for half
the size. The difference is the *fill-in*: a city graph is locally connected
and its sparse Cholesky stays sparse, while a torus's long-range loop
closures make the factorization dense in the worst possible way. This is the
practical form of the ordering discussion in
[The Schur Complement](../linalg/schur.md): the arithmetic depends on the
graph's shape, not only its size.

## 3. Bundle adjustment

The matrix-free Schur complement (`ImplicitSparseSchur`: neither $J^\top J$ nor
$S$ is formed; the reduced operator is applied straight from $J$ inside PCG,
with the Schur–Jacobi preconditioner and a forcing sequence $\eta = 10^{-2}$),
capped at 20 LM iterations.

| dataset | cameras | points | observations | time | final RMSE |
|---|---:|---:|---:|---:|---:|
| Trafalgar | 257 | 65 132 | 225 911 | ~4 s | 0.81 px |
| Dubrovnik | 356 | 226 730 | 1 255 268 | ~48 s | 0.79 px |
| Ladybug | 1 723 | 156 502 | 678 718 | ~77 s | 0.88 px |
| Venice | 1 778 | 993 923 | 5 001 946 | ~56 s | 0.75 px |

Venice is the interesting row: five million observations and it solves faster
than Ladybug's 678 thousand. The cost is in the *reduced* system, not in the
observations — the matrix-free operator's cost scales with `nnz(J)`, and the
fill-in of an explicit $S$ would have scaled with the number of kept
variables squared.

## 4. The Schur comparison

The crate ships both Schur flavours and measures them against each other. The
1.5 changelog records the outcome:

> `LevenbergMarquardtConfig::for_bundle_adjustment` now selects
> `ImplicitSparseSchur` (2.2× faster across the BAL datasets); pass
> `.with_linear_solver_type(LinearSolverType::ExplicitSparseSchur)` for the
> exact reduced solve.

So the default for bundle adjustment is the *matrix-free* variant, at 2.2× the
speed, and the exact variant is one builder call away. The measurement is
reproduced by `benches/ba_benchmark.rs` and `benches/implicit_ba_benchmark.rs`.

## 5. The benchmark suite

| Bench | What it measures |
|---|---|
| `odometry_pose_benchmark` | pose graphs, against the C++ solvers |
| `bundle_adjustment_benchmark` | bundle adjustment, apex rows only with `APEX_BENCH_RUST_ONLY=1` |
| `implicit_ba_benchmark` | the matrix-free Schur path |
| `ba_benchmark` | the bundle-adjustment comparison table |
| `micro_kernels` | normal-equation formation, assembly, Cholesky, QR, PCG, the Schur variants |
| `window_kernels` | the IMU, LiDAR and window kernels of a sliding window |

```console
$ cargo bench --bench odometry_pose_benchmark
$ APEX_BENCH_RUST_ONLY=1 cargo bench --bench bundle_adjustment_benchmark
$ bash benches/tools/run_repeated.sh odometry_pose_benchmark 5
$ cargo bench --bench micro_kernels
```

## 6. Worked example

Choosing a solver for a new problem using these numbers. A consumer has a
window with 2 000 landmarks and 30 keyframes, and it must run at 20 Hz. The
pose-graph table says a 10 000-vertex graph solves in ~110 ms, so the pose
side is fine. The BA table says 65 000 points costs ~4 s, so a
full bundle adjustment is *not* fine — but the window has 2 000 points and a
moving horizon, and the marginalization means only the newest window is
solved at full size. The decision: `ImplicitSparseSchur`, 20 LM iterations,
and a window that slides rather than a global solve.

## 7. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a benchmark on someone else's problem is not your problem</span>

These numbers are from *these* datasets, on *this* machine, with *this*
configuration. The crossover between dense and sparse, between explicit and
implicit Schur, and between GN and LM depends on the problem's shape, its
conditioning, and the number of iterations actually needed. The crate ships
the harness because the honest procedure is to measure the problem you have —
and the number to quote is then yours, with your conditions attached.

</div>

<div class="apx-note">
<span class="apx-title">Note — the full tables live in the repository</span>

`doc/performance.md` at the repository root carries the complete tables,
including the comparator rows, the iteration counts, and the methodology
notes. `doc/plots/` holds the rendered figures, with an interactive HTML
version for each. This chapter is the summary; that document is the record.

</div>
