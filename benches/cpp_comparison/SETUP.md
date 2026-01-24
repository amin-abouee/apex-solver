# C++ Benchmark Setup Guide

This document provides step-by-step instructions for setting up and running the C++ benchmarks for apex-solver.

## Prerequisites

### Git LFS (Required)

The benchmark datasets are stored using Git LFS. You must have Git LFS installed and initialized:

```bash
# Install Git LFS (if not already installed)
brew install git-lfs  # macOS
# or
sudo apt-get install git-lfs  # Ubuntu/Debian

# Initialize Git LFS in your repository
git lfs install

# Pull LFS files (if cloning existing repo)
git lfs pull
```

**Verify datasets are downloaded:**
```bash
# Check file sizes (should show actual MB, not KB pointers)
ls -lh data/odometry/
ls -lh data/bundle_adjustment/
```

If files show as ~1KB, they are LFS pointers and need to be pulled:
```bash
git lfs fetch --all
git lfs checkout
```

## Quick Start

```bash
# 1. Install dependencies (if not already installed)
brew install eigen ceres-solver gtsam g2o tbb

# 2. Build benchmarks
cd benches/cpp_comparison
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(sysctl -n hw.ncpu)

# 3. Run benchmarks
./ceres_odometry_benchmark     # Ceres odometry (pose graph)
./gtsam_odometry_benchmark     # GTSAM odometry
./g2o_odometry_benchmark       # g2o odometry

./ceres_ba_benchmark           # Ceres bundle adjustment
./gtsam_ba_benchmark           # GTSAM bundle adjustment
./g2o_ba_benchmark             # g2o bundle adjustment
```

## Directory Structure

```
cpp_comparison/
├── CMakeLists.txt                # Build configuration
├── SETUP.md                      # This file
├── bundle_adjustment/
│   ├── include/                  # BA-specific headers (if any)
│   └── src/
│       ├── ceres.cpp             # Ceres BA benchmark
│       ├── g2o.cpp               # g2o BA benchmark
│       └── gtsam.cpp             # GTSAM BA benchmark
├── odometry/
│   ├── include/                  # Odometry-specific headers (if any)
│   └── src/
│       ├── ceres.cpp             # Ceres odometry benchmark
│       ├── g2o.cpp               # g2o odometry benchmark
│       └── gtsam.cpp             # GTSAM odometry benchmark
├── common/
│   ├── include/                  # Shared headers
│   │   ├── ba_benchmark_utils.h  # BA benchmark utilities
│   │   ├── ba_cost.h             # BA cost computation
│   │   ├── benchmark_utils.h     # Common benchmark utilities
│   │   ├── read_bal.h            # BAL file parser
│   │   ├── read_g2o.h            # G2O file parser
│   │   └── unified_cost.h        # Unified cost functions
│   └── src/                      # Shared implementations
│       ├── ba_cost.cpp
│       ├── read_bal.cpp
│       ├── read_g2o.cpp
│       └── unified_cost.cpp
└── build/                        # Build directory (generated)
```

## Executables

After building, the following executables are available:

| Executable | Description | Threading |
|------------|-------------|-----------|
| `ceres_odometry_benchmark` | Ceres pose graph optimization | Multi-threaded |
| `gtsam_odometry_benchmark` | GTSAM pose graph optimization | Multi-threaded (TBB) |
| `g2o_odometry_benchmark` | g2o pose graph optimization | Single-threaded |
| `ceres_ba_benchmark` | Ceres bundle adjustment | Multi-threaded (OpenMP) |
| `gtsam_ba_benchmark` | GTSAM bundle adjustment | Multi-threaded (TBB) |
| `g2o_ba_benchmark` | g2o bundle adjustment | Single-threaded |

## Features

1. **Flexible Build System**: CMake automatically detects which libraries are installed and builds only the available benchmarks
2. **Consistent Configuration**: All solvers use identical parameters (LM optimizer, 100 iterations, 1e-3 tolerances)
3. **Common Parsers**: Shared parsers for G2O and BAL file formats
4. **CSV Output**: Results exported in standardized format for comparison
5. **Multiple Datasets**: Tests on multiple odometry and bundle adjustment datasets

## Eigen Version Requirements

The benchmark suite handles the Eigen version mismatch between solvers:

| Library | Eigen Version Required |
|---------|----------------------|
| Ceres Solver | Eigen 5.0.x |
| g2o | Eigen 5.0.x |
| GTSAM | Eigen 3.4.x |

The CMakeLists.txt automatically manages these dependencies by:
- Using Eigen 5.0.x for Ceres and g2o benchmarks
- Using Eigen@3 (3.4.x) for GTSAM benchmarks
- Building separate common libraries for each Eigen version

## Installation

### macOS (Homebrew)

```bash
# Install Eigen (both versions)
brew install eigen
brew install eigen@3

# Install optimization libraries
brew install ceres-solver
brew install gtsam
brew install g2o

# Install threading libraries
brew install tbb       # For GTSAM
brew install libomp    # For OpenMP (optional)
```

### Rebuild after installation

```bash
cd benches/cpp_comparison/build
rm -rf *
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(sysctl -n hw.ncpu)
```

## Datasets

**Note**: All datasets are stored via Git LFS. Ensure you have run `git lfs pull` before running benchmarks.

### Odometry Datasets (Git LFS)

Located in `data/odometry/` (all `.g2o` files tracked via LFS):

**SE3 (3D Pose Graphs):**
| Dataset | Vertices | Edges | Description |
|---------|----------|-------|-------------|
| sphere2500 | 2,500 | 4,949 | Sphere surface |
| parking-garage | 1,661 | 6,275 | Indoor parking |
| torus3D | 5,000 | 9,048 | Torus shape |
| cubicle | 5,750 | 16,869 | Indoor environment |

**SE2 (2D Pose Graphs):**
| Dataset | Vertices | Edges | Description |
|---------|----------|-------|-------------|
| intel | 1,228 | 1,483 | Intel Research Lab |
| mit | 808 | 827 | MIT Killian Court |
| ring | 901 | 991 | Ring topology |
| M3500 | 3,500 | 5,453 | Manhattan grid |

### Bundle Adjustment Datasets (Git LFS)

Located in `data/bundle_adjustment/` (all `.txt` files tracked via LFS):

| Dataset | Cameras | Points | Observations |
|---------|---------|--------|--------------|
| problem-1723-156502-pre | 1,723 | 156,502 | 1,044,414 |

## Running Benchmarks

The executables run from the build directory and use relative paths to find data files:

```bash
cd benches/cpp_comparison/build

# Run individual benchmarks
./ceres_odometry_benchmark
./g2o_odometry_benchmark
./gtsam_odometry_benchmark

./ceres_ba_benchmark
./g2o_ba_benchmark
./gtsam_ba_benchmark

# Or with custom dataset paths
./ceres_ba_benchmark /path/to/custom/bal_file.txt
```

Results are saved to CSV files in the build directory:
- `ceres_odometry_benchmark_results.csv`
- `gtsam_odometry_benchmark_results.csv`
- `g2o_odometry_benchmark_results.csv`
- `ceres_ba_benchmark_results.csv`
- `gtsam_ba_benchmark_results.csv`
- `g2o_ba_benchmark_results.csv`

## Comparing with Rust Benchmarks

```bash
# Run C++ benchmarks
cd benches/cpp_comparison/build
./ceres_odometry_benchmark
./g2o_odometry_benchmark

# Run Rust benchmark suite
cd ../../..
cargo bench solver_comparison
```

## Troubleshooting

### "Cannot open file" or datasets appear corrupted
**Cause:** Git LFS files not downloaded (files are LFS pointers instead of actual data)
**Solution:** 
```bash
# Check if files are LFS pointers (they'll be ~1KB instead of MB/GB)
ls -lh data/odometry/sphere2500.g2o  # Should be ~1MB, not 1KB

# Pull LFS files
git lfs fetch --all
git lfs checkout

# Verify LFS status
git lfs ls-files  # Should list all .g2o and .txt files
```

### "No C++ optimization libraries found"
**Cause:** None of Ceres, GTSAM, or g2o are installed/detected.
**Solution:** Install at least one library using Homebrew.

### "Failed to find Ceres - Missing required Ceres dependency: Eigen version X.X.X"
**Cause:** Eigen version mismatch
**Solution:** `brew reinstall ceres-solver`

### "Cannot open file ../../data/..."
**Cause:** Running benchmark from wrong directory
**Solution:** Always run from `benches/cpp_comparison/build/` directory

### Compilation errors with headers not found
**Cause:** Library not properly installed or detected
**Solution:** Check installation with `brew info <library>` and ensure paths are correct

### OpenMP not found warning
**Cause:** OpenMP not installed on macOS
**Solution:** `brew install libomp` (optional - benchmarks will run single-threaded)

## References

- [Ceres Solver Documentation](http://ceres-solver.org/)
- [GTSAM Documentation](https://gtsam.org/)
- [g2o GitHub](https://github.com/RainerKuemmerle/g2o)
- [BAL Dataset Format](https://grail.cs.washington.edu/projects/bal/)
- [Homebrew](https://brew.sh/)
