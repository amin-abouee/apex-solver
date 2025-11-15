# Apex-Solver Benchmarks

This directory contains both Rust and C++ benchmarks for comprehensive performance evaluation of apex-solver against other optimization libraries.

## Overview

The benchmark suite is organized into two tiers following the [factrs-bench](https://github.com/rpl-cmu/fact-rs/tree/dev/factrs-bench) pattern:

1. **Rust Benchmarks**: Compare apex-solver with other Rust libraries (factrs, tiny-solver)
2. **C++ Benchmarks**: Compare against industry-standard C++ libraries (Ceres, GTSAM, g2o)

## Directory Structure

```
benches/
├── solver_comparison.rs          # Rust benchmark (Criterion-based)
├── cpp_comparison/               # C++ benchmarks (standalone executables)
│   ├── CMakeLists.txt           # C++ build configuration
│   ├── README.md                # Detailed C++ benchmark docs
│   ├── build/                   # Build directory (generated)
│   ├── common/                  # Shared G2O parser and utilities
│   ├── ceres_benchmark.cpp      # Ceres Solver benchmark
│   ├── gtsam_benchmark.cpp      # GTSAM benchmark
│   ├── g2o_benchmark.cpp        # g2o benchmark
│   └── run_all_benchmarks.sh    # Legacy C++ benchmark runner
├── run_all_benchmarks.sh        # Legacy unified runner for both Rust and C++
└── README.md                    # This file
```

**Note:** Rust binaries (`run_benchmarks` and `run_cpp_benchmarks`) are defined in `../bin/` and provide a modern alternative to the bash scripts.

## Quick Start

### Run All Benchmarks

#### Option 1: Bash Scripts (Legacy)
```bash
# From project root
bash benches/run_all_benchmarks.sh
```

#### Option 2: Rust Binaries (Recommended)
```bash
# From project root
cargo run --bin run_benchmarks
```

This will:
1. Run Rust benchmarks with Criterion
2. Build and run C++ benchmarks
3. Generate reports for both

### Run Rust Benchmarks Only

```bash
cargo bench --bench solver_comparison
```

**Results:** HTML reports in `target/criterion/report/index.html`

### Run C++ Benchmarks Only

```bash
cd benches/cpp_comparison
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Run individual benchmarks
./g2o_benchmark
./ceres_benchmark
./gtsam_benchmark
```

**Results:** CSV files: `*_benchmark_results.csv`

## Rust Benchmarks (`solver_comparison.rs`)

### What It Tests
- apex-solver (LM, GN, Dog Leg optimizers)
- factrs (if available)
- tiny-solver (if available)

### Configuration
- Uses Criterion for statistical benchmarking
- Tests on multiple SE2/SE3 datasets
- Measures optimization time and convergence

### Running
```bash
cargo bench --bench solver_comparison

# Run specific benchmark
cargo bench --bench solver_comparison -- sphere2500

# Save baseline for comparison
cargo bench --bench solver_comparison -- --save-baseline my-baseline
```

## C++ Benchmarks (`cpp_comparison/`)

### What It Tests
- ✅ **GTSAM**: Georgia Tech's Smoothing and Mapping library  
- ✅ **g2o**: General Graph Optimization framework
- ❌ **Ceres Solver**: Currently unavailable due to Eigen version conflicts (see `cpp_comparison/CERES_STATUS.md`)

### Configuration
All solvers use consistent parameters:
- Optimizer: Levenberg-Marquardt
- Max iterations: 100
- Cost tolerance: 1e-3
- Parameter tolerance: 1e-3
- Gauge fix: First pose fixed

### Prerequisites

Install C++ dependencies (macOS):
```bash
brew install eigen@3 gtsam g2o cmake
```

**Note:** Ceres Solver has Eigen version compatibility issues. The current setup uses:
- Eigen 3.4.1 (via `eigen@3`) for GTSAM and g2o
- Both GTSAM and g2o benchmarks are fully functional
- For Ceres status, see `cpp_comparison/CERES_STATUS.md` and `cpp_comparison/BENCHMARK_RESULTS.md`

### Building

```bash
cd benches/cpp_comparison
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(sysctl -n hw.ncpu)
```

### Running

#### Option 1: Bash Script (Legacy)
```bash
# From benches/cpp_comparison/
bash run_all_benchmarks.sh
```

#### Option 2: Rust Binary (Recommended)
```bash
# From project root
cargo run --bin run_cpp_benchmarks
```

#### Manual Execution
```bash
# From benches/cpp_comparison/build/
./g2o_benchmark       # Run g2o benchmark (✅ working)
./gtsam_benchmark     # Run GTSAM benchmark (✅ working)
# ./ceres_benchmark   # Ceres currently unavailable (❌ Eigen conflicts)
```

### Output

Each benchmark generates:
- **Console table**: Human-readable results
- **CSV file**: Machine-readable for analysis

Example CSV:
```csv
dataset,manifold,solver,language,vertices,edges,init_cost,final_cost,improvement_pct,iterations,time_ms,status
sphere2500,SE3,g2o-LM,C++,2500,4949,0.0,727.15,0.0,26,4176,CONVERGED
```

## Rust Benchmark Runners

As an alternative to bash scripts, this project provides Rust binaries for running benchmarks:

### `run_benchmarks` Binary

Unified runner that orchestrates both Rust and C++ benchmarks:

```bash
cargo run --bin run_benchmarks
```

**Features:**
- Colored output with progress indicators
- Automatic directory management
- Error handling and status reporting
- Cross-platform compatibility

### `run_cpp_benchmarks` Binary

Specialized runner for C++ benchmarks that handles:

- CMake build configuration and compilation
- Parallel building using all available CPU cores
- Execution of individual C++ benchmark executables
- Result collection and organization
- Rust example execution for comparison

```bash
cargo run --bin run_cpp_benchmarks
```

**Advantages over bash scripts:**
- Better error handling and recovery
- Type safety and compile-time checks
- Cross-platform path handling
- Structured output and logging
- Easier maintenance and testing

## Datasets

Both Rust and C++ benchmarks use datasets from `../data/`:

### SE3 (3D Pose Graphs)
| Dataset | Vertices | Edges | Description |
|---------|----------|-------|-------------|
| sphere2500 | 2,500 | 4,949 | Sphere surface |
| parking-garage | 1,661 | 6,275 | Indoor parking |
| torus3D | 5,000 | 9,048 | Torus shape |

### SE2 (2D Pose Graphs)
| Dataset | Vertices | Edges | Description |
|---------|----------|-------|-------------|
| intel | 1,228 | 1,483 | Intel Research Lab |
| mit | 808 | 827 | MIT Killian Court |
| manhattanOlson3500 | 3,500 | 5,598 | Manhattan grid |

## Comparing Results

### Rust Results
View Criterion reports:
```bash
open target/criterion/report/index.html
```

### C++ Results
View CSV files:
```bash
cat benches/cpp_comparison/build/*_benchmark_results.csv
```

### Combined Analysis
Merge all results:
```bash
cd benches/cpp_comparison/build
cat *_benchmark_results.csv | head -1 > combined.csv
tail -n +2 -q *_benchmark_results.csv >> combined.csv
```

## Architecture Notes

### Why Separate Rust and C++ Benchmarks?

Following the factrs-bench pattern, we keep Rust and C++ benchmarks as separate executables rather than using FFI integration because:

1. **Simplicity**: No build.rs complexity, no bindgen, no linking issues
2. **Independence**: Rust benchmarks work even if C++ libs aren't installed
3. **Fairness**: External process timing is more realistic
4. **Maintainability**: Each ecosystem evolves independently
5. **Proven**: factrs-bench team chose this approach for good reasons

### No FFI, No build.rs

Unlike projects that call C++ from Rust, we:
- ✅ Build C++ benchmarks with CMake (not cargo)
- ✅ Run C++ benchmarks as separate processes
- ✅ Compare results via CSV/JSON output files
- ❌ No FFI bindings
- ❌ No build.rs script
- ❌ No runtime integration

This is the **factrs-bench pattern** and is intentional.

## Troubleshooting

### C++ Build Issues

**Problem:** `Ceres not found or has Eigen version conflict`

**Solution:**
```bash
brew reinstall ceres-solver
```

**Problem:** `GTSAM not found`

**Solution:**
```bash
brew install gtsam
```

### Rust Benchmark Issues

**Problem:** `factrs` or `tiny-solver` not found

**Solution:** These are optional dependencies. Benchmarks will run with apex-solver only.

### Dataset Not Found

**Problem:** `Cannot open file ../../../data/sphere2500.g2o`

**Solution:** Ensure you're running from the correct directory:
```bash
cd benches/cpp_comparison/build
./g2o_benchmark
```

## Performance Expectations

### Rust Benchmarks
- Typically run in seconds per dataset
- Criterion provides statistical analysis with confidence intervals
- HTML reports include comparison graphs

### C++ Benchmarks
- SE2: 30-100ms (small 2D problems)
- SE3: 500ms-30s (larger 3D problems)
- CSV output for easy comparison

## Contributing

When adding new benchmarks:

1. **Rust**: Add to `solver_comparison.rs` using Criterion
2. **C++**: Add new solver in `cpp_comparison/` following existing pattern
3. **Datasets**: Place in `../data/` directory
4. **Documentation**: Update this README

## References

- [factrs-bench](https://github.com/rpl-cmu/fact-rs/tree/dev/factrs-bench) - Inspiration for this architecture
- [Criterion.rs](https://github.com/bheisler/criterion.rs) - Rust benchmarking framework
- [Ceres Solver](http://ceres-solver.org/)
- [GTSAM](https://gtsam.org/)
- [g2o](https://github.com/RainerKuemmerle/g2o)
