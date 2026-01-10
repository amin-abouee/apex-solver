# Bundle Adjustment Benchmark

This benchmark compares Apex Solver against Ceres, GTSAM, and g2o on large-scale bundle adjustment problems.

## Running the Benchmark

### Quick Start

```bash
# Run the bundle adjustment benchmark
cargo bench --bench bundle_adjustment_comparison

# Or run with release optimizations (recommended)
cargo bench --bench bundle_adjustment_comparison --release
```

### What It Does

The benchmark will:

1. **Load the dataset**: `data/bundle_adjustment/problem-1723-156502-pre.txt`
   - 1,723 cameras
   - 156,502 3D points
   - 678,718 observations

2. **Run Apex Solver** (Rust):
   - Schur-Iterative linear solver
   - Huber loss function
   - Levenberg-Marquardt optimizer

3. **Build and Run C++ Benchmarks**:
   - **Ceres Solver**: ITERATIVE_SCHUR + OpenMP
   - **GTSAM**: LM optimizer + Intel OneTBB
   - **g2o**: SBA types + OpenMP

4. **Output Results**:
   - Console: Comparison table with metrics
   - CSV: `ba_benchmark_results.csv`

### Metrics Reported

For each solver:
- Initial MSE (Mean Squared Error in pixels²)
- Final MSE
- Initial RMSE (Root Mean Squared Error in pixels)
- Final RMSE
- Runtime (seconds, optimization only)
- Number of iterations
- Convergence status

### Prerequisites

**C++ Libraries** (optional but recommended):
```bash
# macOS
brew install ceres-solver gtsam g2o tbb eigen@3

# The benchmark will run Apex Solver even if C++ libraries are missing
```

**Build Configuration**:
- Ceres & g2o: Compiled with OpenMP
- GTSAM: Compiled with Intel OneTBB
- All: Optimized with `-O3 -march=native`

### Example Output

```
=== BUNDLE ADJUSTMENT COMPARISON ===
Dataset              | Solver      | Lang | Cameras | Points | Obs    | Init RMSE | Final RMSE | Iters | Time (s) | Status
----------------------------------------------------------------------------------------------------------------------------
problem-1723-156502  | apex-solver | Rust |  1723   | 156502 | 678718 |   11.23   |    0.87    |  42   |   45.0   | CONVERGED
problem-1723-156502  | Ceres       | C++  |  1723   | 156502 | 678718 |   11.23   |    0.79    |  38   |   32.5   | CONVERGED
problem-1723-156502  | GTSAM       | C++  |  1723   | 156502 | 678718 |   11.23   |    0.81    |  41   |   34.2   | CONVERGED
problem-1723-156502  | g2o         | C++  |  1723   | 156502 | 678718 |   11.23   |    0.80    |  39   |   31.8   | CONVERGED
```

### Troubleshooting

**C++ benchmarks fail to build:**
- Check that libraries are installed: `brew list | grep -E 'ceres|gtsam|g2o'`
- The benchmark will still run Apex Solver if C++ libraries are missing

**Dataset not found:**
- Ensure you're running from the repository root
- Dataset should be at: `data/bundle_adjustment/problem-1723-156502-pre.txt`

**Benchmark takes too long:**
- This is expected for large datasets (may take 30-60 seconds per solver)
- Consider testing on smaller datasets first (e.g., problem-21-11315-pre.txt)

### Configuration Details

All solvers use equivalent settings for fair comparison:

| Parameter | Value |
|-----------|-------|
| Algorithm | Levenberg-Marquardt |
| Max Iterations | 100 |
| Linear Solver | Schur Complement (Iterative) |
| Preconditioner | Block Diagonal (Jacobi) |
| Loss Function | Huber (δ=1.0) |
| Gauge Fixing | First camera fixed (6 DOF) |

### Paper Reference

Compilation flags follow the settings from:
**"Robust Bundle Adjustment Revisited"** (https://arxiv.org/html/2409.12190v2)

- GTSAM: Compiled with Intel OneTBB
- Ceres & g2o: Compiled with OpenMP
- All: `-O3` optimization
