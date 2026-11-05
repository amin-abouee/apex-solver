#!/usr/bin/env python3
"""List production functions at or above a line threshold.

usage: longfns.py <repo-root> <fns.json from fnscan.py> [threshold=90]
Test modules (from the first #[cfg(test)] line), tests/, benches/, examples/
are excluded. Prints a markdown table.
"""
import json, os, sys

root, data = sys.argv[1], sys.argv[2]
threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 90
fns = json.load(open(data))
first_test = {}


def test_line(path):
    if path not in first_test:
        first_test[path] = 10**9
        with open(os.path.join(root, path), encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip().startswith("#[cfg(test)]"):
                    first_test[path] = number
                    break
    return first_test[path]


rows = []
for f in fns:
    path = f["file"]
    if path.startswith(("tests/", "benches/", "examples/")) or "/tests/" in path or "/benches/" in path or path.endswith("tests.rs"):
        continue
    if f["line"] >= test_line(path) or f["len"] < threshold:
        continue
    rows.append(f)
rows.sort(key=lambda f: -f["len"])
print("| Lines | Branch pts | Params | Function | Location |")
print("|---:|---:|---:|---|---|")
for f in rows:
    print(f"| {f['len']} | {f['branches']} | {f['params']} | `{f['name']}` | `{f['file']}:{f['line']}` |")
