# Benchmarks & Performance

Headline numbers (Apple M4, release profile, median of 5 runs; methodology and
full tables in the repository's
[`doc/performance.md`](https://github.com/amin-abouee/apex-solver/blob/main/doc/performance.md)):
timing covers `optimize()` only; costs are computed by the harness directly
from the input files so values are comparable across implementations.

## Pose graph optimization (LM)

| dataset | vertices | edges | apex time | final χ² |
|---|---:|---:|---:|---:|
| ring | 434 | 459 | ~4 ms | 2.2e-2 |
| M3500 | 3 500 | 5 453 | ~60 ms | 1.5 |
| city10000 | 10 000 | 20 687 | ~110 ms | 4.4 |
| sphere2500 | 2 500 | 4 949 | ~190 ms | 21.3 |
| parking-garage | 1 661 | 6 275 | ~85 ms | 0.62 |
| torus3D | 5 000 | 9 048 | ~930 ms | 236 |
| cubicle | 5 750 | 16 869 | ~990 ms | 1 692 |

## Bundle adjustment (LM + iterative Schur, reprojection RMSE)

| dataset | cameras | points | observations | time | final RMSE |
|---|---:|---:|---:|---:|---:|
| Trafalgar | 257 | 65 132 | 225 911 | ~4 s | 0.81 px |
| Dubrovnik | 356 | 226 730 | 1 255 268 | ~48 s | 0.79 px |
| Ladybug | 1 723 | 156 502 | 678 718 | ~77 s | 0.88 px |
| Venice | 1 778 | 993 923 | 5 001 946 | ~56 s | 0.75 px |

## Reproducing

```bash
cargo bench --bench odometry_pose_benchmark   # pose graphs (Rust + C++ solvers)
APEX_BENCH_RUST_ONLY=1 cargo bench --bench bundle_adjustment_benchmark   # apex rows only
bash benches/tools/run_repeated.sh odometry_pose_benchmark 5            # N repeats, archived CSVs
```

Kernel-level micro-benchmarks (normal-equation formation, assembly, PCG) live
in `benches/micro_kernels.rs` and run under Criterion:

```bash
cargo bench --bench micro_kernels
```
