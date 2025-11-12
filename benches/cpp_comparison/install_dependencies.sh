#!/usr/bin/env bash

# Installation script for C++ optimization library dependencies
# Tested on macOS with Homebrew

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== C++ Optimization Libraries Installation Script ===${NC}"
echo ""
echo "This script will install Ceres Solver, GTSAM, and g2o via Homebrew."
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo -e "${RED}ERROR: Homebrew is not installed!${NC}"
    echo "Please install Homebrew first: https://brew.sh"
    exit 1
fi

echo -e "${GREEN}✓ Homebrew is installed${NC}"
echo ""

# Function to check and install a package
install_package() {
    local package=$1
    local name=$2

    echo -e "${YELLOW}Checking ${name}...${NC}"

    if brew list "$package" &>/dev/null; then
        echo -e "${GREEN}✓ ${name} is already installed${NC}"
        brew info "$package" | head -3
    else
        echo -e "${YELLOW}Installing ${name}...${NC}"
        if brew install "$package"; then
            echo -e "${GREEN}✓ ${name} installed successfully${NC}"
        else
            echo -e "${RED}✗ Failed to install ${name}${NC}"
            return 1
        fi
    fi
    echo ""
}

# Install dependencies
echo -e "${BLUE}=== Installing Dependencies ===${NC}"
echo ""

# Install Eigen (required by all)
install_package "eigen" "Eigen"

# Install g2o
install_package "g2o" "g2o"

# Install GTSAM
install_package "gtsam" "GTSAM"

# Install Ceres Solver
echo -e "${YELLOW}Checking Ceres Solver...${NC}"
if brew list "ceres-solver" &>/dev/null; then
    echo -e "${GREEN}✓ Ceres Solver is already installed${NC}"

    # Check for Eigen version conflict
    EIGEN_VERSION=$(brew list --versions eigen | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    echo "Current Eigen version: $EIGEN_VERSION"

    echo ""
    echo -e "${YELLOW}NOTE: Ceres was compiled with Eigen 5.0.0 but you may have Eigen $EIGEN_VERSION${NC}"
    echo "If you encounter CMake errors about Eigen version mismatch, you have two options:"
    echo ""
    echo "  Option 1: Rebuild Ceres with current Eigen (recommended)"
    echo "    brew reinstall ceres-solver"
    echo ""
    echo "  Option 2: Build only GTSAM and g2o benchmarks"
    echo "    Ceres will be skipped automatically during CMake configuration"
    echo ""
else
    echo -e "${YELLOW}Installing Ceres Solver...${NC}"
    if brew install "ceres-solver"; then
        echo -e "${GREEN}✓ Ceres Solver installed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to install Ceres Solver${NC}"
    fi
    echo ""
fi

# Install CMake
install_package "cmake" "CMake"

echo -e "${BLUE}=== Installation Summary ===${NC}"
echo ""
echo "Installed packages:"
brew list | grep -E "eigen|g2o|gtsam|ceres-solver|cmake" || echo "No packages found"
echo ""

echo -e "${GREEN}✓ Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. cd cpp-bench"
echo "  2. mkdir build && cd build"
echo "  3. cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "  4. cmake --build . --config Release"
echo "  5. ./g2o_benchmark  # or ./gtsam_benchmark, ./ceres_benchmark"
echo ""
echo "Or use the automation script:"
echo "  bash run_all_benchmarks.sh"
