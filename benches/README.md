# Apex Solver Benchmarks

Two comparison benchmarks:

1. **Pose Graph Optimization** (`odometry_pose_benchmark.rs`) — apex-solver vs factrs,
   tiny-solver, Ceres, GTSAM and g2o on 8 standard G2O datasets (SE2 + SE3).
2. **Bundle Adjustment** (`bundle_adjustment_benchmark.rs`) — apex-solver vs Ceres, GTSAM and
   g2o on 4 BAL datasets.

**→ Results live in [`doc/performance.md`](../doc/performance.md)** — tables (mean ± std over
5 runs) and figures. Numbers are not duplicated here so they cannot drift.

---

## Prerequisites

### Datasets

Datasets are downloaded on demand by the `download_datasets` tool in the `apex-io` crate.
No Git LFS is involved.

```bash
# List all available datasets and selection numbers
cargo run --release -p apex-io --bin download_datasets -- --list

# Download everything the benchmarks need
# (all odometry g2o + the largest problem from each BA dataset)
cargo run --release -p apex-io --bin download_datasets -- --select 10

# Interactive mode (prompts for selection)
cargo run --release -p apex-io --bin download_datasets
```

Files land in `data/odometry/{2d,3d}/` and `data/bundle_adjustment/<name>/`. A benchmark skips
any dataset whose file is missing, so a partial download still runs.

### C++ solvers (optional)

The Ceres / GTSAM / g2o comparisons build from `benches/cpp_comparison/` on first run and need
CMake 3.15+, Eigen3, and the solvers themselves:

```bash
brew install ceres-solver gtsam g2o eigen   # macOS
```

If they are unavailable the benchmarks run with the Rust solvers only and log a warning.

---

## Running

```bash
# single run of either benchmark
cargo bench --bench odometry_pose_benchmark
cargo bench --bench bundle_adjustment_benchmark
```

Each run writes an aggregate CSV to `output/`:
`odometry_pose_benchmark_results.csv` and `ba_comparison_results.csv`.

### Repeated runs and figures

`doc/performance.md` is generated from 5 repetitions of each benchmark:

```bash
# run N times, archiving every run's raw CSV + log to output/runs/
bash benches/tools/run_repeated.sh odometry_pose_benchmark 5
bash benches/tools/run_repeated.sh bundle_adjustment_benchmark 5

# aggregate to output/*_aggregated.csv and render doc/plots/*.{html,png}
uv run --with plotly --with kaleido --with pandas benches/tools/plot_benchmarks.py
```

The bundle-adjustment sweep is slow — roughly 3.5 hours for 5 runs, since Ceres and GTSAM each
consume the full 10-minute timeout on Venice.

---

## Methodology

- **Timing** covers the `optimize()` call only. Problem construction, cost evaluation and CSV
  writing are all outside the timed region.
- **Console logging is off** during measurement, for the solvers and the harness alike.
- **Cost** is computed by the harness directly from the source file for every solver, so values
  are comparable across implementations rather than reflecting each library's internal
  bookkeeping.
- **Metrics** are the final objective value and runtime, following the pose-graph literature
  (SE-Sync, Rosen et al. IJRR 2019; Carlone et al. ICRA 2015). Bundle adjustment uses
  reprojection RMSE and runtime.
- One sample per invocation — `run_repeated.sh` supplies the repetitions, so Rust and C++
  solvers contribute the same number of independent samples.

## Configuration

| | Max iters | Cost tol | Param tol | Grad tol | Damping |
|---|---|---|---|---|---|
| apex-solver SE2 | 150 | 1e-4 | 1e-4 | 1e-10 | 1e-4 |
| apex-solver SE3 | 100 | 1e-4 | 1e-4 | 1e-12 | 1e-4 |
| apex-solver BA | 20 | 1e-6 | 1e-8 | 1e-10 | 1e-3 |

Bundle adjustment runs in SelfCalibration mode (pose + landmarks + intrinsics) with an
iterative Schur complement solver (PCG + Schur-Jacobi preconditioner) and a 10-minute timeout
per solver.

---

## Dataset sources

- **BAL** — [Bundle Adjustment in the Large](https://grail.cs.washington.edu/projects/bal/)
- **G2O** — standard SLAM benchmarks from the g2o and SE-Sync distributions
