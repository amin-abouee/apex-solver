#!/bin/bash
# Profile all SE3 datasets with samply
# Usage: ./scripts/profile_all_datasets.sh

set -e

# Colors for output
GREEN='\033[0.32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Navigate to project root
cd "$(dirname "$0")/.."

echo -e "${BLUE}=== Apex-Solver Profiling Suite ===${NC}"
echo

# Create profiles directory
mkdir -p profiles

# Build profiling binary
echo -e "${GREEN}Building profiling binary...${NC}"
cargo build --profile profiling --example profile_datasets
echo

# List of datasets to profile
DATASETS=("sphere2500" "parking-garage" "cubicle" "torus3D" "rim" "grid3D")

# Profile each dataset
for dataset in "${DATASETS[@]}"; do
    echo -e "${GREEN}Profiling ${dataset}...${NC}"
    samply record --save-only -o "profiles/${dataset}.json" \
        ./target/profiling/examples/profile_datasets "${dataset}" 2>&1 | \
        grep -E "(Loading|Loaded|Problem setup|Optimization Complete|Status|Time:|Iterations:)"

    # Check file size
    size=$(ls -lh "profiles/${dataset}.json" | awk '{print $5}')
    echo -e "  ${BLUE}✓ Saved profile: profiles/${dataset}.json (${size})${NC}"
    echo
done

echo -e "${GREEN}=== Profiling Complete ===${NC}"
echo
echo "Profile files saved in profiles/ directory:"
ls -lh profiles/*.json
echo
echo "To view profiles:"
echo "  1. Open https://profiler.firefox.com/"
echo "  2. Click 'Load a profile from file'"
echo "  3. Select a profile from profiles/ directory"
echo
echo "Or use: samply load profiles/<dataset>.json"
