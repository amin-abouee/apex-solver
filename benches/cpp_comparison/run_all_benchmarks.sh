#!/usr/bin/env bash

# Script to build and run all C++ benchmarks (Ceres, GTSAM, g2o)
# and compare with Rust benchmarks (apex-solver, factrs, tiny-solver)

set -e  # Exit on error (disabled for benchmark runs)
set +e  # Disable exit on error for benchmark execution

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== APEX-SOLVER COMPREHENSIVE BENCHMARK SUITE ===${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${YELLOW}Project root: ${PROJECT_ROOT}${NC}"
echo ""

# ============================================================================
# Step 1: Build C++ Benchmarks
# ============================================================================

echo -e "${GREEN}[1/4] Building C++ benchmarks...${NC}"

BUILD_DIR="${SCRIPT_DIR}/build"

if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Running CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "Building executables..."
cmake --build . --config Release -j$(sysctl -n hw.ncpu) || echo -e "${YELLOW}⚠ Some C++ benchmarks failed to build${NC}"

echo -e "${GREEN}✓ C++ benchmarks build completed${NC}"
echo ""

# ============================================================================
# Step 2: Run C++ Benchmarks
# ============================================================================

echo -e "${GREEN}[2/4] Running C++ benchmarks...${NC}"

cd "$BUILD_DIR"

# Run Ceres benchmark
if [ -f "./ceres_benchmark" ]; then
    echo -e "${BLUE}Running Ceres benchmark...${NC}"
    ./ceres_benchmark
    echo ""
else
    echo -e "${YELLOW}⚠ Ceres benchmark not available (skipping)${NC}"
    echo ""
fi

# Run GTSAM benchmark
if [ -f "./gtsam_benchmark" ]; then
    echo -e "${BLUE}Running GTSAM benchmark...${NC}"
    ./gtsam_benchmark
    echo ""
else
    echo -e "${YELLOW}⚠ GTSAM benchmark not available (skipping)${NC}"
    echo ""
fi

# Run g2o benchmark
if [ -f "./g2o_benchmark" ]; then
    echo -e "${BLUE}Running g2o benchmark...${NC}"
    ./g2o_benchmark
    echo ""
else
    echo -e "${YELLOW}⚠ g2o benchmark not available (skipping)${NC}"
    echo ""
fi

echo -e "${GREEN}✓ C++ benchmarks completed${NC}"
echo ""

# ============================================================================
# Step 3: Run Rust Benchmarks
# ============================================================================

echo -e "${GREEN}[3/4] Running Rust benchmarks...${NC}"

cd "$PROJECT_ROOT"

echo -e "${BLUE}Running apex-solver benchmark (compare_optimizers)...${NC}"
cargo run --release --example compare_optimizers -- --max-iterations 100 --cost-tolerance 1e-3 --parameter-tolerance 1e-3
echo ""

echo -e "${GREEN}✓ Rust benchmarks completed${NC}"
echo ""

# ============================================================================
# Step 4: Combine Results
# ============================================================================

echo -e "${GREEN}[4/4] Combining results...${NC}"

RESULTS_DIR="${PROJECT_ROOT}/benchmark_results"
mkdir -p "$RESULTS_DIR"

# Copy C++ results to results directory
if [ -f "${BUILD_DIR}/ceres_benchmark_results.csv" ]; then
    cp "${BUILD_DIR}/ceres_benchmark_results.csv" "$RESULTS_DIR/"
    echo "✓ Copied Ceres results"
fi

if [ -f "${BUILD_DIR}/gtsam_benchmark_results.csv" ]; then
    cp "${BUILD_DIR}/gtsam_benchmark_results.csv" "$RESULTS_DIR/"
    echo "✓ Copied GTSAM results"
fi

if [ -f "${BUILD_DIR}/g2o_benchmark_results.csv" ]; then
    cp "${BUILD_DIR}/g2o_benchmark_results.csv" "$RESULTS_DIR/"
    echo "✓ Copied g2o results"
fi

echo ""
echo -e "${GREEN}✓ All results copied to: ${RESULTS_DIR}${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo -e "${BLUE}=== BENCHMARK SUMMARY ===${NC}"
echo ""
echo "Results available in:"

# Only list results that actually exist
RESULTS_FOUND=0
if [ -f "${RESULTS_DIR}/ceres_benchmark_results.csv" ]; then
    echo "  - ${RESULTS_DIR}/ceres_benchmark_results.csv"
    RESULTS_FOUND=1
fi
if [ -f "${RESULTS_DIR}/gtsam_benchmark_results.csv" ]; then
    echo "  - ${RESULTS_DIR}/gtsam_benchmark_results.csv"
    RESULTS_FOUND=1
fi
if [ -f "${RESULTS_DIR}/g2o_benchmark_results.csv" ]; then
    echo "  - ${RESULTS_DIR}/g2o_benchmark_results.csv"
    RESULTS_FOUND=1
fi

if [ $RESULTS_FOUND -eq 1 ]; then
    echo ""
    echo "To merge all results into a single CSV:"
    echo "  cd ${RESULTS_DIR}"
    echo "  cat *_benchmark_results.csv | head -1 > combined_results.csv"
    echo "  tail -n +2 -q *_benchmark_results.csv >> combined_results.csv"
fi
echo ""
echo -e "${GREEN}Benchmark suite completed!${NC}"
