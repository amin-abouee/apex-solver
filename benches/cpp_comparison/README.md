# C++ Benchmark Suite for Apex-Solver

This directory contains C++ benchmarks comparing apex-solver against industry-standard optimization libraries: **Ceres Solver**, **GTSAM**, and **g2o**.

> **✅ Status:** g2o benchmark successfully built and tested!  
> **⚠️ Note:** Ceres has an Eigen version conflict (see SETUP.md for solutions). GTSAM is not yet installed.  
> **📖 Quick Start:** See [SETUP.md](SETUP.md) for detailed setup instructions.

## Overview

The benchmark suite evaluates pose graph optimization performance on both 2D (SE2) and 3D (SE3) datasets using Levenberg-Marquardt optimization with consistent termination criteria across all solvers.

### Benchmarked Libraries

| Library | Version | Language | Description |
|---------|---------|----------|-------------|
| **Ceres Solver** | Homebrew | C++ | Google's optimization library for non-linear least squares |
| **GTSAM** | Homebrew | C++ | Georgia Tech's Smoothing and Mapping library |
| **g2o** | Homebrew | C++ | General Graph Optimization framework |
| **apex-solver** | Latest | Rust | This project (Rust-based nonlinear optimization) |

## Directory Structure

```
cpp-bench/
├── CMakeLists.txt              # CMake build configuration
├── README.md                   # This file
├── run_all_benchmarks.sh       # Automation script
├── common/
│   ├── read_g2o.h              # G2O file parser header
│   ├── read_g2o.cpp            # G2O file parser implementation
│   └── benchmark_utils.h       # Timing and CSV output utilities
├── ceres_benchmark.cpp         # Ceres Solver benchmark
├── gtsam_benchmark.cpp         # GTSAM benchmark
├── g2o_benchmark.cpp           # g2o benchmark
└── main.cpp                    # Info executable
```

## Prerequisites

### Install Dependencies (macOS with Homebrew)

```bash
# Install all required libraries
brew install ceres-solver
brew install gtsam
brew install g2o
brew install eigen
brew install cmake
```

Verify installations:
```bash
brew list ceres-solver gtsam g2o eigen cmake
```

### System Requirements

- macOS 10.15+ (or Linux with equivalent packages)
- CMake 3.15+
- C++17 compatible compiler (Clang, GCC)
- Eigen 3.3+

## Building the Benchmarks

### Quick Build (Recommended)

```bash
# From the cpp-bench directory
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(sysctl -n hw.ncpu)
```

### Build Options

**Release build (default, for benchmarking):**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Debug build (for development):**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

## Running Benchmarks

### Individual Benchmarks

Run each solver independently:

```bash
cd build

# Ceres Solver benchmark
./ceres_benchmark

# GTSAM benchmark
./gtsam_benchmark

# g2o benchmark
./g2o_benchmark
```

Each benchmark outputs:
- Console table with results
- CSV file: `{solver}_benchmark_results.csv`

### Automated Benchmark Suite

Run all C++ benchmarks at once:

```bash
# From cpp-bench directory
bash run_all_benchmarks.sh
```

This script:
1. Builds all C++ benchmarks in Release mode
2. Runs Ceres, GTSAM, and g2o benchmarks
3. Optionally runs Rust benchmarks (apex-solver)
4. Copies all CSV results to `../benchmark_results/`

## Benchmark Configuration

All solvers use **consistent configuration** for fair comparison:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer** | Levenberg-Marquardt | Trust region method |
| **Max Iterations** | 100 | Termination limit |
| **Cost Tolerance** | 1e-3 | Relative cost change threshold |
| **Parameter Tolerance** | 1e-3 | Parameter update threshold |
| **Linear Solver** | Sparse Cholesky | Fast for positive definite systems |
| **Gauge Fix** | Fix first pose | Prevents rank-deficient Hessian |

## Datasets

Benchmarks run on 6 datasets included in `../data/`:

### SE3 (3D Pose Graphs)

| Dataset | Vertices | Edges | Description |
|---------|----------|-------|-------------|
| **sphere2500** | 2,500 | 4,949 | Sphere surface |
| **parking-garage** | 1,661 | 6,275 | Indoor parking |
| **torus3D** | 5,000 | 9,048 | Torus shape |

### SE2 (2D Pose Graphs)

| Dataset | Vertices | Edges | Description |
|---------|----------|-------|-------------|
| **intel** | 1,228 | 1,483 | Intel Research Lab |
| **mit** | 808 | 827 | MIT Killian Court |
| **manhattanOlson3500** | 3,500 | 5,598 | Manhattan grid |

## Output Format

### CSV Format

All benchmarks write results to CSV with the following columns:

```csv
dataset,manifold,solver,language,vertices,edges,init_cost,final_cost,improvement_pct,iterations,time_ms,status
rim,SE3,Ceres-LM,C++,10195,10663,1.234e6,8.456e3,99.31,42,1234,CONVERGED
```

**Columns:**
- `dataset`: Dataset name (rim, cubicle, intel, mit)
- `manifold`: SE2 or SE3
- `solver`: Ceres-LM, GTSAM-LM, g2o-LM
- `language`: C++ (or Rust for apex-solver)
- `vertices`: Number of poses
- `edges`: Number of constraints
- `init_cost`: Initial objective value
- `final_cost`: Final objective value after optimization
- `improvement_pct`: `(init_cost - final_cost) / init_cost * 100`
- `iterations`: Number of optimization iterations
- `time_ms`: Wall-clock time in milliseconds
- `status`: CONVERGED or NOT_CONVERGED

### Console Output

Benchmarks also print formatted tables to the console:

```
Dataset      | Manifold | Solver     | Vertices | Edges  | Init Cost    | Final Cost   | Improvement | Iters | Time(ms) | Status    
rim          | SE3      | Ceres-LM   | 10195    | 10663  | 1.234000e+06 | 8.456000e+03 |      99.31% | 42    | 1234     | CONVERGED
```

## Combining Results

Merge all CSV results into a single file:

```bash
cd ../benchmark_results

# Create combined CSV with header
cat *_benchmark_results.csv | head -1 > combined_results.csv

# Append all data rows (skip headers)
tail -n +2 -q *_benchmark_results.csv >> combined_results.csv
```

## Comparing with Rust Benchmarks

To compare C++ solvers with apex-solver:

```bash
# Run C++ benchmarks
cd cpp-bench
bash run_all_benchmarks.sh

# Run Rust benchmarks
cd ..
cargo run --release --example compare_optimizers -- \
    --max-iterations 100 \
    --cost-tolerance 1e-3 \
    --parameter-tolerance 1e-3

# Results are now in benchmark_results/
```

## Troubleshooting

### CMake Cannot Find Libraries

If CMake fails to find Ceres, GTSAM, or g2o:

```bash
# Check Homebrew installation paths
brew --prefix ceres-solver
brew --prefix gtsam
brew --prefix g2o

# Set CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH="$(brew --prefix)/opt:$(brew --prefix)"
cmake .. -DCMAKE_BUILD_TYPE=Release
```

### Link Errors with g2o

If you encounter linker errors with g2o targets:

```cmake
# In CMakeLists.txt, use alternative linking:
target_link_libraries(g2o_benchmark PRIVATE
    bench_common
    Eigen3::Eigen
    ${G2O_CORE_LIBRARY}
    ${G2O_STUFF_LIBRARY}
    ${G2O_TYPES_SLAM2D_LIBRARY}
    ${G2O_TYPES_SLAM3D_LIBRARY}
    ${G2O_SOLVER_EIGEN_LIBRARY}
)
```

### Dataset Not Found Errors

Ensure you're running benchmarks from the `build/` directory:

```bash
cd cpp-bench/build
./ceres_benchmark  # Expects ../data/rim.g2o
```

Or adjust dataset paths in the source code.

## Performance Tips

1. **Always use Release builds** for benchmarking (`-DCMAKE_BUILD_TYPE=Release`)
2. **Close background applications** to reduce CPU contention
3. **Run multiple times** and average results for consistency
4. **Disable Turbo Boost** on Mac for reproducible timings:
   ```bash
   sudo pmset -a disablesleep 1
   ```

## Extending the Benchmarks

### Adding New Datasets

1. Place `.g2o` file in `../data/`
2. Add to benchmark main functions:
   ```cpp
   results.push_back(BenchmarkSE3("new_dataset", "../data/new_dataset.g2o"));
   ```

### Adding New Solvers

1. Create `{solver}_benchmark.cpp`
2. Implement `BenchmarkSE2()` and `BenchmarkSE3()` functions
3. Add executable to `CMakeLists.txt`:
   ```cmake
   add_executable(new_solver_benchmark new_solver_benchmark.cpp)
   target_link_libraries(new_solver_benchmark PRIVATE bench_common ...)
   ```

### Modifying Optimization Parameters

Edit the solver configuration in each benchmark file:

**Ceres:**
```cpp
options.max_num_iterations = 100;
options.function_tolerance = 1e-3;
options.parameter_tolerance = 1e-3;
```

**GTSAM:**
```cpp
params.setMaxIterations(100);
params.setRelativeErrorTol(1e-3);
params.setAbsoluteErrorTol(1e-3);
```

**g2o:**
```cpp
optimizer.optimize(100);  // max iterations
```

## References

- [Ceres Solver](http://ceres-solver.org/)
- [GTSAM](https://gtsam.org/)
- [g2o](https://github.com/RainerKuemmerle/g2o)
- [G2O File Format](https://github.com/RainerKuemmerle/g2o/wiki/File-Format)

## License

Same as apex-solver project.
