#!/usr/bin/env bash
#
# Run a benchmark N times and archive every run's raw output.
#
# Usage:
#   bash benches/tools/run_repeated.sh <bench-name> [runs]
#
# Examples:
#   bash benches/tools/run_repeated.sh odometry_pose_benchmark 5
#   bash benches/tools/run_repeated.sh bundle_adjustment_benchmark 5
#
# Per-run artifacts land in output/runs/:
#   <bench>_run<i>.csv        aggregate result CSV for that run
#   <bench>_run<i>.log        full console output for that run
#   <bench>_run<i>_<exe>.csv  per-solver C++ CSVs for that run
#
# output/ and *.csv are gitignored, so this data stays local for investigation.

set -euo pipefail

BENCH="${1:?usage: run_repeated.sh <bench-name> [runs]}"
RUNS="${2:-5}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

case "$BENCH" in
    odometry_pose_benchmark)   RESULT_CSV="output/odometry_pose_benchmark_results.csv" ;;
    bundle_adjustment_benchmark) RESULT_CSV="output/ba_comparison_results.csv" ;;
    *) echo "unknown bench: $BENCH" >&2; exit 1 ;;
esac

CPP_BUILD="benches/cpp_comparison/build"
RUN_DIR="output/runs"
mkdir -p "$RUN_DIR"

echo "Running '$BENCH' $RUNS times -> $RUN_DIR"

for i in $(seq 1 "$RUNS"); do
    echo "--- run $i/$RUNS  ($(date +%H:%M:%S)) ---"

    # Remove the previous CSV so a failed run cannot masquerade as a fresh result.
    rm -f "$RESULT_CSV"

    cargo bench --bench "$BENCH" > "$RUN_DIR/${BENCH}_run${i}.log" 2>&1 || {
        echo "  run $i FAILED (see $RUN_DIR/${BENCH}_run${i}.log)" >&2
    }

    if [[ -f "$RESULT_CSV" ]]; then
        cp "$RESULT_CSV" "$RUN_DIR/${BENCH}_run${i}.csv"
        echo "  saved $RUN_DIR/${BENCH}_run${i}.csv"
    else
        echo "  WARNING: $RESULT_CSV not produced by run $i" >&2
    fi

    # Keep the per-solver C++ CSVs too; they carry vertices/edges and per-solver detail.
    for csv in "$CPP_BUILD"/*_results.csv; do
        [[ -e "$csv" ]] || continue
        cp "$csv" "$RUN_DIR/${BENCH}_run${i}_$(basename "$csv")"
    done
done

echo "Done. $(ls -1 "$RUN_DIR/${BENCH}"_run*.csv 2>/dev/null | wc -l | tr -d ' ') run CSVs in $RUN_DIR"
