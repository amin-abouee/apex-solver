# Apex Solver Benchmarks

This directory contains two comprehensive benchmarks for evaluating Apex Solver performance:

1. **Bundle Adjustment Benchmark** (`bundle_adjustment_benchmark.rs`) - BAL datasets, Apex vs C++ solvers
2. **Pose Graph Optimization Benchmark** (`solver_comparison.rs`) - G2O datasets, Apex vs Rust/C++ solvers

---

## 1. Bundle Adjustment Benchmark

Compares **Apex Solver (Iterative Schur)** against **Ceres**, **GTSAM**, and **g2o** on 4 large-scale BAL datasets.

### Datasets Tested

| Dataset | Cameras | Landmarks | Observations | Size |
|---------|---------|-----------|--------------|------|
| Ladybug | 1,723 | 156,502 | 678,718 | Large |
| Trafalgar | 257 | 65,132 | 225,911 | Medium |
| Dubrovnik | 356 | 226,730 | 1,255,268 | Large |
| Venice | 1,778 | 993,923 | 5,001,946 | Very Large |

### Apex Solver Configuration

- **Mode**: SelfCalibration (optimizes pose + intrinsics + landmarks)
- **Linear Solver**: Iterative Schur Complement (PCG with Schur-Jacobi preconditioner)
- **Timeout**: 10 minutes per solver
- **Parameters**: 
  - Max iterations: 50
  - Cost tolerance: 1e-6
  - Parameter tolerance: 1e-8
  - Damping: 1e-3

### Usage

```bash
# Run all 4 datasets
cargo bench --bench bundle_adjustment_benchmark
```

### Output

Results are saved to **`output/ba_comparison_results.csv`** with:
- Dataset name
- Solver (Apex-Iterative, Ceres, GTSAM, g2o)
- Language (Rust, C++)
- Initial/Final RMSE (pixels)
- Time (seconds)
- Iterations
- Status (CONVERGED / TIMEOUT)

### Example Output

```
Dataset: Ladybug: Cameras: 1723, Landmarks: 156502, Observations: 678718
--------------------------------------------------------------------------------------
Solver               Language   Initial RMSE    Final RMSE      Time (s)    Iters      Status
--------------------------------------------------------------------------------------
Apex-Iterative       Rust       49.234          14.675          70.14       12         CONVERGED
Ceres                C++        49.234          1.892           45.32       40         CONVERGED
GTSAM                C++        49.234          1.889           52.18       38         CONVERGED
g2o                  C++        49.234          1.891           48.67       41         CONVERGED

Dataset: Trafalgar: Cameras: 257, Landmarks: 65132, Observations: 225911
...
```

---

## 2. Pose Graph Optimization Benchmark

Compares **apex-solver**, **factrs**, **tiny-solver** (Rust) and **g2o**, **GTSAM** (C++) on standard pose graph datasets.

### Datasets Tested

**SE2 (2D Pose Graphs)**:
- M3500 (3,500 poses)
- mit (1,045 poses)
- intel (943 poses)
- ring (3,500 poses)

**SE3 (3D Pose Graphs)**:
- sphere2500 (2,500 poses)
- parking-garage (1,661 poses)
- torus3D (5,000 poses)
- cubicle (5,750 poses)

### Apex Solver Configuration

#### SE2 (2D):
- Max iterations: 150
- Cost tolerance: 1e-4
- Parameter tolerance: 1e-4
- Gradient tolerance: 1e-10

#### SE3 (3D):
- Max iterations: 100
- Cost tolerance: 1e-4
- Parameter tolerance: 1e-4
- Gradient tolerance: 1e-12

### Usage

```bash
# Run all datasets across all solvers
cargo run --release --bin solver_comparison
```

### Output

CSV files with convergence metrics:
- **Converged**: true/false
- **Time**: Average milliseconds (5 runs)
- **Iterations**: Number of iterations
- **Initial/Final Cost**: Optimization cost
- **Improvement**: Percentage cost reduction

---

## C++ Solver Integration

Both benchmarks support C++ solvers (Ceres, GTSAM, g2o) if available.

### Requirements

- CMake 3.10+
- Ceres Solver
- GTSAM
- g2o
- Eigen3

### Build Process

C++ benchmarks are located in `benches/cpp_comparison/` and build automatically on first run.

If C++ solvers are unavailable, benchmarks run with Rust solvers only (with warnings).

### Troubleshooting

**"C++ benchmarks unavailable"**:
1. Install required libraries (Ceres, GTSAM, g2o)
2. Ensure CMake is in PATH
3. Check `benches/cpp_comparison/CMakeLists.txt` for build requirements

**OR** ignore and run Rust-only benchmarks.

---

## Output Files

All benchmark results are saved to the **`output/`** directory:

- **`ba_comparison_results.csv`**: Bundle adjustment benchmark results
- **`solver_comparison_*.csv`**: Pose graph benchmark results (multiple files)

The `output/` directory is gitignored and created automatically.

---

## Quick Start

### Run Bundle Adjustment Benchmark
```bash
cargo bench --bench bundle_adjustment_benchmark
```

### Run Pose Graph Benchmark
```bash
cargo run --release --bin solver_comparison
```

### View Results
```bash
# Open CSV in spreadsheet software
open output/ba_comparison_results.csv

# Or view in terminal
cat output/ba_comparison_results.csv
```

---

## Performance Notes

- **Bundle adjustment**: Iterative Schur scales well to large problems (1,000+ cameras)
- **Pose graphs**: apex-solver matches or exceeds factrs/tiny-solver on most datasets
- **Timeouts**: BA benchmark enforces 10-minute timeout per solver to handle very large datasets
- **Averaging**: Pose graph benchmark runs 5 iterations per dataset for stable timing

---

## Dataset Sources

- **BAL datasets**: [Bundle Adjustment in the Large](https://grail.cs.washington.edu/projects/bal/)
- **G2O datasets**: Standard SLAM benchmarks from g2o repository

Datasets are expected in:
- `data/bundle_adjustment/[dataset]/problem-*.txt` (BAL format)
- `data/*.g2o` (G2O format)
