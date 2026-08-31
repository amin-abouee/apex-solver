#!/usr/bin/env bash
#
# Fast correctness + performance probe used to gate incremental changes.
#
# Runs a fixed set of small/medium solves covering every hot path that the
# code-review fixes touch, and prints one line per probe:
#
#   <probe>  status=<...>  iters=<n>  cost=<final>  ms=<median of N>
#
# Correctness invariants are `status`, `iters` and `cost` — they must not move
# for a pure performance change. `ms` is the perf signal.
#
# Usage:
#   bash benches/tools/sanity_check.sh [runs] > output/sanity_before.txt
#   # ...make a change, rebuild...
#   bash benches/tools/sanity_check.sh [runs] > output/sanity_after.txt
#   diff <(cut -d'm' -f1 output/sanity_before.txt) \
#        <(cut -d'm' -f1 output/sanity_after.txt)   # correctness diff, ignores timing

set -uo pipefail

RUNS="${1:-3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PG=./target/release/pose_graph_g2o
BA=./target/release/bundle_adjustment
TRAFALGAR=data/bundle_adjustment/trafalgar/problem-21-11315-pre.txt
LADYBUG=data/bundle_adjustment/ladybug/problem-1723-156502-pre.txt

for b in "$PG" "$BA"; do
    [[ -x "$b" ]] || { echo "missing $b — run: cargo build --release --bins" >&2; exit 1; }
done

median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }

# Extract the invariants from a probe log.
parse_pose_graph() {
    local log="$1"
    local status iters cost
    status=$(grep -o 'Status: [A-Za-z]*' "$log" | tail -1 | cut -d' ' -f2)
    iters=$(grep -o 'Iterations: [0-9]*' "$log" | tail -1 | cut -d' ' -f2)
    cost=$(grep -o 'Final chi2: [0-9.e+-]*' "$log" | tail -1 | cut -d' ' -f3)
    echo "${status:-NA} ${iters:-NA} ${cost:-NA}"
}

parse_ba() {
    local log="$1"
    local status iters cost
    status=$(grep -o 'Status: [A-Za-z]*' "$log" | tail -1 | cut -d' ' -f2)
    iters=$(grep -o 'Iterations: [0-9]*' "$log" | tail -1 | cut -d' ' -f2)
    cost=$(grep -o 'Final cost: [0-9.e+-]*' "$log" | tail -1 | cut -d' ' -f3)
    echo "${status:-NA} ${iters:-NA} ${cost:-NA}"
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

probe() {
    local name="$1" parser="$2"; shift 2
    local times=() invariants="" log="$TMP/${name//\//_}.log"
    for _ in $(seq 1 "$RUNS"); do
        local t0 t1
        t0=$(python3 -c 'import time;print(int(time.time()*1000))')
        "$@" > "$log" 2>&1
        t1=$(python3 -c 'import time;print(int(time.time()*1000))')
        times+=($((t1 - t0)))
    done
    invariants=$("$parser" "$log")
    # shellcheck disable=SC2086
    printf '%-26s status=%-28s iters=%-4s cost=%-14s ms=%s\n' \
        "$name" $invariants "$(median "${times[@]}")"
}

echo "# apex-solver sanity check — $(date -u +%Y-%m-%dT%H:%M:%SZ) — median of $RUNS runs"
echo "# commit $(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty')"
echo

# --- odometry: covers LM/GN/DL step paths, sparse assembly, Cholesky ---
probe "odom/M3500/lm"       parse_pose_graph "$PG" --dataset M3500       --optimizer lm
probe "odom/sphere2500/lm"  parse_pose_graph "$PG" --dataset sphere2500  --optimizer lm
probe "odom/ring/gn"        parse_pose_graph "$PG" --dataset ring        --optimizer gn
probe "odom/ring/dl"        parse_pose_graph "$PG" --dataset ring        --optimizer dl

# --- bundle adjustment: covers both Schur paths + projection factor ---
probe "ba/trafalgar/implicit" parse_ba "$BA" "$TRAFALGAR" -s implicit -t bundle-adjustment
probe "ba/trafalgar/explicit" parse_ba "$BA" "$TRAFALGAR" -s explicit -t bundle-adjustment
probe "ba/trafalgar/selfcal"  parse_ba "$BA" "$TRAFALGAR" -s implicit -t self-calibration
