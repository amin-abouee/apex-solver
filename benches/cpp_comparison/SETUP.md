# C++ Benchmark Setup Guide

This document provides step-by-step instructions for setting up and running the C++ benchmarks for apex-solver.

## Quick Start

```bash
# 1. Install dependencies
cd cpp-bench
bash install_dependencies.sh

# 2. Build benchmarks
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(sysctl -n hw.ncpu)

# 3. Run benchmarks
./g2o_benchmark       # Run g2o benchmark
./gtsam_benchmark     # Run GTSAM benchmark (if installed)
./ceres_benchmark     # Run Ceres benchmark (if installed)
```

## What Was Created

The C++ benchmark suite includes:

### Directory Structure
```
cpp-bench/
├── CMakeLists.txt                    # Build configuration
├── README.md                         # Detailed documentation
├── SETUP.md                          # This file
├── install_dependencies.sh           # Dependency installation script
├── run_all_benchmarks.sh            # Automation script
├── common/
│   ├── read_g2o.h                   # G2O file parser header
│   ├── read_g2o.cpp                 # G2O file parser implementation
│   └── benchmark_utils.h            # Timing and CSV utilities
├── ceres_benchmark.cpp              # Ceres Solver benchmark
├── gtsam_benchmark.cpp              # GTSAM benchmark
├── g2o_benchmark.cpp                # g2o benchmark
└── main.cpp                         # Info executable
```

### Features

1. **Flexible Build System**: CMake automatically detects which libraries are installed and builds only the available benchmarks
2. **Consistent Configuration**: All solvers use identical parameters (LM optimizer, 100 iterations, 1e-3 tolerances)
3. **Common G2O Parser**: Shared parser for SE2 and SE3 pose graphs
4. **CSV Output**: Results exported in standardized format for comparison
5. **Multiple Datasets**: Tests on 6 datasets (3 SE3, 3 SE2)

## Current Status

✅ **Working:**
- g2o benchmark (successfully tested)
- CMake build system with optional dependencies
- G2O file parser for SE2 and SE3
- CSV output generation

⚠️ **Partially Working:**
- Ceres benchmark (Eigen version conflict: Ceres needs 5.0.0, system has 5.0.1)
- GTSAM benchmark (not installed on this system)

## Installation Status

On your Mac, the following are installed:
- ✅ Eigen 5.0.1
- ✅ g2o (via Homebrew)
- ✅ Ceres Solver 2.2.0 (with Eigen 5.0.0 dependency issue)
- ❌ GTSAM (not installed)

## Fixing Ceres-Eigen Version Conflict

The Ceres Solver on your system was compiled with Eigen 5.0.0, but Eigen 5.0.1 is currently installed. This causes a CMake configuration error.

**Solution Options:**

### Option 1: Reinstall Ceres (Recommended)
```bash
brew reinstall ceres-solver
```
This rebuilds Ceres against the current Eigen version (5.0.1).

### Option 2: Use Only g2o and GTSAM
The CMakeLists.txt is configured to skip Ceres if it can't be found, so you can proceed with just g2o and GTSAM benchmarks.

### Option 3: Downgrade Eigen
```bash
brew uninstall eigen
brew install eigen@5.0.0  # if available
```
Not recommended as it may break other packages.

## Installing GTSAM

To add GTSAM benchmarks:

```bash
brew install gtsam

# Rebuild
cd cpp-bench/build
rm -rf *
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Datasets Used

The benchmarks test on these datasets from `../data/`:

**SE3 (3D Pose Graphs):**
| Dataset | Vertices | Edges | Description |
|---------|----------|-------|-------------|
| sphere2500 | 2,500 | 4,949 | Sphere surface |
| parking-garage | 1,661 | 6,275 | Indoor parking |
| torus3D | 5,000 | 9,048 | Torus shape |

**SE2 (2D Pose Graphs):**
| Dataset | Vertices | Edges | Description |
|---------|----------|-------|-------------|
| intel | 1,228 | 1,483 | Intel Research Lab |
| mit | 808 | 827 | MIT Killian Court |
| manhattanOlson3500 | 3,500 | 5,598 | Manhattan grid |

## Example g2o Benchmark Results

From the successful test run:

```
Dataset            | Manifold | Solver  | Vertices | Edges | Final Cost   | Iters | Time(ms) | Status
sphere2500         | SE3      | g2o-LM  | 2500     | 4949  | 7.271497e+02 | 26    | 4176     | CONVERGED
parking-garage     | SE3      | g2o-LM  | 1661     | 6275  | 1.238691e+00 | 41    | 532      | CONVERGED
torus3D            | SE3      | g2o-LM  | 5000     | 9048  | 1.997223e+04 | 100   | 25823    | NOT_CONVERGED
intel              | SE2      | g2o-LM  | 1228     | 1483  | 1.608737e+05 | 100   | 77       | NOT_CONVERGED
mit                | SE2      | g2o-LM  | 808      | 827   | 5.263310e+02 | 100   | 34       | CONVERGED
manhattanOlson3500 | SE2      | g2o-LM  | 3500     | 5598  | 1.460766e+02 | 28    | 96       | CONVERGED
```

Results saved to: `cpp-bench/build/g2o_benchmark_results.csv`

## Comparing with Rust Benchmarks

To compare C++ solvers with apex-solver:

```bash
# Run C++ benchmarks
cd cpp-bench/build
./g2o_benchmark

# Run Rust benchmarks
cd ../..
cargo run --release --example compare_optimizers

# Results will be in:
# - cpp-bench/build/g2o_benchmark_results.csv
# - (Rust results printed to console)
```

## Troubleshooting

### "No C++ optimization libraries found"
**Cause:** None of Ceres, GTSAM, or g2o are installed/detected.
**Solution:** Run `bash install_dependencies.sh`

### "Failed to find Ceres - Missing required Ceres dependency: Eigen version 5.0.0"
**Cause:** Eigen version mismatch
**Solution:** `brew reinstall ceres-solver`

### "Cannot open file ../../data/sphere2500.g2o"
**Cause:** Running benchmark from wrong directory
**Solution:** Always run from `cpp-bench/build/` directory

### Compilation errors with g2o
**Cause:** g2o headers not found
**Solution:** Check g2o installation: `brew info g2o`

## Next Steps

1. **Fix Ceres**: `brew reinstall ceres-solver`
2. **Install GTSAM**: `brew install gtsam`
3. **Rebuild benchmarks**: `cd build && rm -rf * && cmake .. && make`
4. **Run all benchmarks**: `bash ../run_all_benchmarks.sh`
5. **Compare results**: Analyze CSV outputs against apex-solver performance

## Performance Notes

From the g2o test run:
- **Fast 2D optimization**: 34-96ms for SE2 datasets
- **Slower 3D optimization**: 532-25,823ms for SE3 datasets
- **Convergence**: 4/6 datasets converged within 100 iterations
- **Large problems**: torus3D (5000 vertices) took 25 seconds

## References

- [g2o GitHub](https://github.com/RainerKuemmerle/g2o)
- [Ceres Solver Documentation](http://ceres-solver.org/)
- [GTSAM Documentation](https://gtsam.org/)
- [Homebrew](https://brew.sh/)
