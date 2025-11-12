#!/usr/bin/env bash

# Unified script to run both Rust and C++ benchmarks

set -e
set +e  # Disable exit on error for benchmark execution

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== APEX-SOLVER COMPREHENSIVE BENCHMARK SUITE ===${NC}"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# ============================================================================
# Step 1: Run Rust Benchmarks
# ============================================================================

echo -e "${GREEN}[1/2] Running Rust Benchmarks${NC}"
echo ""

echo -e "${BLUE}Running Rust solver comparison (apex-solver, factrs, tiny-solver)...${NC}"
if cargo bench --bench solver_comparison; then
    echo -e "${GREEN}✓ Rust benchmarks completed${NC}"
else
    echo -e "${YELLOW}⚠ Rust benchmarks encountered issues${NC}"
fi
echo ""

# ============================================================================
# Step 2: Run C++ Benchmarks
# ============================================================================

echo -e "${GREEN}[2/2] Running C++ Benchmarks${NC}"
echo ""

cd "$SCRIPT_DIR/cpp_comparison"

if [ ! -f "run_all_benchmarks.sh" ]; then
    echo -e "${YELLOW}⚠ cpp_comparison/run_all_benchmarks.sh not found (skipping C++ benchmarks)${NC}"
else
    echo -e "${BLUE}Running C++ benchmarks (Ceres, GTSAM, g2o)...${NC}"
    if bash run_all_benchmarks.sh; then
        echo -e "${GREEN}✓ C++ benchmarks completed${NC}"
    else
        echo -e "${YELLOW}⚠ C++ benchmarks encountered issues${NC}"
    fi
fi

cd "$PROJECT_ROOT"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${BLUE}=== BENCHMARK SUMMARY ===${NC}"
echo ""
echo "Results available in:"
echo "  Rust benchmarks: target/criterion/"
echo "  C++ benchmarks: benches/cpp_comparison/build/*_benchmark_results.csv"
echo ""
echo "To compare results:"
echo "  1. Check Criterion HTML reports: target/criterion/report/index.html"
echo "  2. View C++ CSV results: cat benches/cpp_comparison/build/*.csv"
echo ""
echo -e "${GREEN}Benchmark suite completed!${NC}"
