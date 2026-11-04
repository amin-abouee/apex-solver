# Hotpath Profile of apex-solver (odometry + bundle adjustment) — hotpath-rs

**Profiled**: apex-solver `1.5.0` working tree (post Exp-001..009 pipeline) with
[hotpath-rs](https://github.com/pawurb/hotpath-rs) `0.28.4` (`hotpath` + `hotpath-alloc`
features — per-function timing **and** allocation attribution).
**Host**: Apple M4 (4P+6E, 10 logical cores), 32 GB, macOS; release profile (`opt-level=3`, thin LTO).
**Workloads**: the exact criterion benchmark constructions — pose-graph odometry
(`benches/odometry_benchmark.rs`: g2o datasets, SE2/SE3 `BetweenFactor`, L2, `SparseCholesky`)
and bundle adjustment (`benches/ba_benchmark.rs`: BAL datasets, `ProjectionFactor` + Huber(1px),
`for_bundle_adjustment()` with the explicit-Schur variants; implicit variant added).
Every run converged to the pinned golden final cost (e.g. dubrovnik-135 `1.888768e5`,
venice-52 `9.716653e4`), so the profiled solves are numerically identical to the benchmarked ones.

## TL;DR — the hotpaths

- **Odometry (pose graph) is a sparse-Cholesky problem.** 59–90 % of solve time on the
  larger graphs is inside `SparseCholeskySolver`'s numeric LLT factorization + solve of the
  (damped) normal equations — the `faer` `Llt` call following the already-instrumented
  `JᵀJ` build. Only on tiny graphs (intel, 5 iterations) does Jacobian/factor evaluation
  and trial-cost evaluation become comparable.
- **Bundle adjustment (explicit sparse Schur) splits into two regimes:**
  - *camera-heavy* (trafalgar-257, 257 cameras / 1 542 kept DOF): **Cholesky of the reduced
    system S is 60 %** of solve time; forming S is 14 %.
  - *landmark-heavy* (dubrovnik-135, 90 642 landmarks / 810 kept DOF): **forming the Schur
    complement is 67 %** of solve time; its Cholesky is only 2.4 %. The dense
    `kept_dof²` accumulator + filter in `compute_schur_complement` is the bottleneck the
    docstring already warns about.
- **Implicit (matrix-free) Schur** spends **80 %** of its time in `pcg` — and on
  trafalgar-21 it is **4× slower wall-clock** than explicit sparse Schur (3.42 s vs 0.84 s
  per solve), a caution for the `for_bundle_adjustment()` default on small/medium problems.
- **Allocation hotspots** mirror the timing hotspots: the per-iteration damped-Hessian copy
  + LLT arrays (odometry: 52–68 % of all bytes), the `JᵀJ` CSC values rebuild (BA: ~35 %),
  and the per-iteration Jacobian values array (implicit BA: 47 %).

## Methodology

Instrumentation: 20 feature-gated `#[cfg_attr(feature = "hotpath", hotpath::measure)]`
attributes on the pipeline below (zero-cost noops unless the feature is enabled; verified
`cargo check` is unchanged without the feature):

| Stage | Function |
|---|---|
| Whole solve | `levenberg_marquardt::optimize_with_mode` |
| Per-iteration assembly | `optimizer::iteration_preamble` → `sparse::assemble_sparse` → `linearizer::compute_block_into` (per-block factor `linearize` + noise whitening + loss corrector) |
| One-time setup | `optimizer::initialize_optimization_state` → `sparse::build_symbolic_structure` |
| Linear-solve phase | `levenberg_marquardt::compute_step_generic` → `normal_eq::compute` (JᵀJ, `LazyNormalEquations` + `NormalEquationsCache`), Schur: `partition::gather` / `invert_in_place` / `new`, `explicit::solve_reduced_system` → `compute_schur_complement` / `solve_with_cholesky` / `compute_reduced_gradient` / `back_substitute`; implicit: `implicit::solve_from_jacobian`; `pcg::pcg` |
| Trial evaluation | `levenberg_marquardt::evaluate_and_apply_step` → `optimizer::apply_parameter_step` |

Harness: `examples/profile_hotpath.rs` (mirrors the bench builders/configs verbatim),
run with `--features hotpath,hotpath-alloc`, one dataset × variant per process
(`hotpath` aggregates across the whole process), 20 solves per odometry dataset and
2 per BA dataset/variant; raw reports in `benchmarking_results/hotpath/raw_*.txt`.

Reading the numbers (important):

1. **`% Total` is inclusive time** — a caller's row contains its callees'. Derived rows
   ("LLT of normal equations") are differences of instrumented rows.
2. **`compute_block_into` is parallel** (rayon): its Total sums *worker* durations, so its
   % is inflated vs wall-clock by the parallel speedup (2.9–6.5× here). The wall-clock
   share of assembly is the `assemble_sparse` row.
3. **`normal_eq::compute` merges two nested instrumented functions**
   (`LazyNormalEquations::compute` → `NormalEquationsCache::compute`, same key), so its
   real time ≈ half the reported Total. The tables below already halve it.
4. Per-call guard overhead (~50–100 ns) slightly inflates the per-block
   `compute_block_into` average; it does not affect the stage-level shares.
5. Absolute wall times here include profiling overhead — for absolute speed comparisons
   use `doc/performance.md`; this report is about *relative* hotpaths.

## Odometry (pose graph, SparseCholesky)

Timing shares of total process time (% of solve = `optimize_with_mode`):

| Stage | M3500 (3.5k poses, 14 it, 82 ms/solve) | intel (1.2k, 5 it, 8.0 ms) | sphere2500 (2.5k, 6 it, 329 ms) | parking-garage (1.7k, 7 it, 120 ms) | torus3D (25 it, 2.01 s) |
|---|---|---|---|---|---|
| Linear-solve phase (`compute_step_generic`) | 72.7 | 46.8 | 92.2 | 80.0 | 94.7 |
| ├ JᵀJ build (`normal_eq`, real) | 13.3 | 16.0 | 5.6 | 17.7 | 4.6 |
| └ **numeric LLT factorize+solve (derived)** | **59.4** | **30.8** | **86.6** | **62.3** | **90.1** |
| Jacobian assembly (wall, `assemble_sparse`) | 8.1 | 16.3 | 3.2 | 7.4 | 2.6 |
| ├ of which factor `linearize` (worker-sum) | 48.7 | 46.9 | 11.2 | 48.1 | 14.0 |
| Trial cost + step apply (`evaluate_and_apply_step`) | 15.0 | 24.5 | 2.9 | 6.2 | 2.1 |
| Setup incl. symbolic structure | 2.4 | 7.5 | 1.3 | 5.1 | 0.5 |

`compute_block_into` averages 263 ns/call (P50 250 ns) over 3.2 M calls (M3500) — the
`BetweenFactor` linearize + whitening + L2 corrector loop parallelizes ~6× on M4
(worker-sum 833 ms vs 139 ms wall inside `assemble_sparse`).

**Allocations** (exclusive bytes; M3500 total 175 MB/solve, sphere2500 475 MB/solve):

| Function | M3500 | sphere2500 |
|---|---|---|
| `compute_step_generic` (damped-H copy + LLT arrays + step) | 52.0 % | 68.4 % |
| `normal_eq::compute` (JᵀJ values) | 20.5 % | 13.2 % |
| `assemble_sparse` (Jacobian values array) | 15.1 % | 7.9 % |
| `build_symbolic_structure` (once per solve) | 6.1 % | 8.3 % |

Threads: 19 threads (main + 18 rayon workers); main ≈ 70 % CPU (the serial LLT), workers
≈ 14 % each during parallel assembly/`JᵀJ` phases.

## Bundle adjustment (explicit sparse Schur)

Timing shares (% of solve):

| Stage | trafalgar-21 (0.84 s/solve, 21 it) | trafalgar-257 (11 it) | dubrovnik-135 (24 s/solve, 21 it) | venice-52 (12.8 s/solve, 21 it) |
|---|---|---|---|---|
| Linear-solve phase | 84.2 | 89.9 | 94.6 | 92.2 |
| ├ JᵀJ build (real) | 18.7 | 6.9 | 12.4 | 15.9 |
| ├ **Schur complement formation** (`compute_schur_complement`) | — | **14.0** | **67.4** | — |
| ├ **Cholesky of S** (`solve_with_cholesky`) | — | **60.2** | **2.4** | — |
| ├ back-substitute + reduced gradient | — | 0.7 | 1.4 | — |
| └ extraction/extraction-misc (uninstrumented) | ~19 | ~8 | ~10 | ~11 |
| Jacobian assembly (wall) | 6.6 | 5.2 | 2.5 | 3.1 |
| ├ factor `linearize` (worker-sum) | 28.6 | 5.6 | 15.3 | 18.6 |
| Trial cost + step apply | 5.7 | 3.4 | 1.6 | 2.5 |
| Partition gather + invert | 1.9 | 1.5 | 1.1 | 1.3 |
| Setup | 1.3 | 1.2 | 1.0 | 1.0 |

Phase-level splits were instrumented in a second pass and captured for trafalgar-257 and
dubrovnik-135 (`raw_ba_*_phases.txt`); the trafalgar-21/venice-52 rows come from the first
pass (solve-phase split not captured there).

Two regimes, driven by `kept_dof` (cameras × 6) vs landmark count:

- **trafalgar-257**: 1 542 kept DOF → the sparse LLT of S dominates (987 ms/iteration,
  60 % of everything). S formation is only 230 ms/iteration.
- **dubrovnik-135**: 810 kept DOF but 90 642 eliminated blocks → S formation dominates
  (844 ms/iteration, 67 %) while its LLT is 30 ms. The rank-6 scatter of every
  inverted landmark block into the dense `kept_dof²` accumulator (then filtered back to
  sparse — the scaling limit the docstring calls out) is the single hottest function in
  the whole library on this workload.

`partition::gather` + `invert_in_place` (H_ee extraction and inversion) stay cheap
(≈1–2 %) thanks to the chunk eliminator.

**Iterative variant** (`ExplicitSchurVariant::Iterative`, PCG on formed S): per-iteration
cost is within ~5 % of sparse on dubrovnik-135/venice-52 (PCG 1–5 %); on trafalgar-257 PCG
is 22 % of time. Note the iterative variant took 42 vs 22 LM steps to converge on
trafalgar-257 (different trajectory, allowed by its own golden) — a convergence-robustness
observation, not a per-iteration perf one.

**Implicit variant** (matrix-free, `for_bundle_adjustment()` default), trafalgar-21:

| Stage | share |
|---|---|
| `implicit::solve_from_jacobian` | 94.4 % |
| ├ `pcg` (matrix-free JᵀJ operators over cached J) | **80.0 %** |
| Jacobian assembly (wall) | 1.8 % |
| Trial cost + apply | 1.7 % |

Wall-clock per solve: implicit 3.42 s vs explicit sparse 0.84 s (**4.1× slower**) on this
problem — with a tiny kept set (21 cameras), the matrix-free operator's extra J/Jᵀ sweeps
buy nothing while S would be only 126×126. The implicit default pays off only when
`kept_dof` is large enough that S's factorization would dominate (the doc/performance.md
large-BA regime).

**Allocations** (exclusive bytes; trafalgar-257 sparse: 7.5 GB/solve(!), trafalgar-21
implicit: 686 MB/solve):

| Function | trafalgar-257 sparse | trafalgar-21 implicit |
|---|---|---|
| `normal_eq::compute` (JᵀJ CSC values rebuild) | 34.6 % | — |
| `compute_step_generic` (S + factorization buffers) | 20.0 % | 3.3 % |
| `solve_reduced_system` | 18.5 % | — |
| `assemble_sparse` (Jacobian values array) | 13.9 % | **46.6 %** |
| `build_symbolic_structure` (once) | 6.5 % | 15.4 % |
| `implicit::solve_from_jacobian` workspaces | — | 17.8 % |

Max RSS: 92 MB (M3500) → 1.7 GB (trafalgar-257); allocation churn is ~two orders of
magnitude above resident size (17.7 GB allocated over a 2-run trafalgar-257 process).

## Ranked optimization candidates (evidence-based)

1. **Sparse numeric LLT of the (damped) normal equations — odometry's hotspot.**
   59–90 % of solve time on graphs ≥ ~2.5k vertices (`faer::Llt::try_new_with_symbolic` +
   `solve` inside `SparseCholeskySolver::solve_augmented_equation`). Levers: fill-reducing
   ordering quality (what does faer's `SymbolicLlt` default to — can AMD/COLAMD be
   supplied?), supernodal/block kernels, or exploiting the constant pattern to cache more
   than the symbolic factor.
2. **`compute_schur_complement` dense-accumulator path — landmark-heavy BA's hotspot**
   (67 % on dubrovnik-135). The dense `kept_dof²` accumulation + filter-back-to-sparse is
   O(landmarks × kept_dof) work in a bandwidth-bound buffer. Levers: direct sparse scatter
   (the bitmap-Schur output idea from `experiment_003`), splitting the dense accumulator
   into per-thread tiles, or capping the accumulator via camera grouping.
3. **Cholesky of S — camera-heavy BA's hotspot** (60 % on trafalgar-257). Same lever
   family as (1); additionally a parallel factorization would overlap with the serial
   main-thread profile seen here.
4. **Per-iteration allocation churn.** Damped-Hessian copy + LLT arrays (odometry,
   52–68 % of bytes), JᵀJ CSC values rebuild (BA 35 %), Jacobian values array (47 % of
   bytes on implicit BA). Persistent scratch buffers in the solver hot paths would cut
   allocator traffic roughly in half on the solve phase.
5. **Trial-cost re-evaluation** (`evaluate_and_apply_step`, 15–25 % on small/medium pose
   graphs): it runs a second full parallel residual evaluation per iteration. Reusing the
   accepted trial's residuals as the next iteration's `r` (the classic Ceres trick) would
   remove ~one residual sweep per accepted step.
6. **Solver-preset guidance.** `for_bundle_adjustment()` defaults to
   `ImplicitSparseSchur`, measured here at 4× slower than explicit sparse on a
   21-camera/11k-landmark problem; the crossover is worth documenting (or the preset made
   size-aware).

## Reproduce

```bash
cargo run --release --features hotpath,hotpath-alloc --example profile_hotpath -- odometry M3500 20
cargo run --release --features hotpath,hotpath-alloc --example profile_hotpath -- ba dubrovnik-135 schur_explicit_sparse 2
cargo run --release --features hotpath,hotpath-alloc --example profile_hotpath -- ba trafalgar-21 implicit 2
HP_FORMAT=json-pretty HP_OUTPUT=/tmp/r.json cargo run ... -- odometry M3500 20
```

Raw hotpath tables: `benchmarking_results/hotpath/raw_*.txt` (16 runs; `*_phases` files
include the second-pass `solve_reduced_system` phase split).
