# Performance Benchmarks

**Hardware**: Apple Mac Mini M4, 64GB RAM
**Build**: Rust release (`opt-level=3`, LTO); C++ `-O3 -DNDEBUG -march=native`
**Methodology**: 5 independent runs per benchmark, reported as **mean ± std**. Timing covers the `optimize()` call only — problem setup and metric computation are excluded. Bundle adjustment uses a 10-minute timeout per solver.
**Metrics**: final objective value (cost) and runtime, following the pose graph optimization literature ([SE-Sync, Rosen et al. IJRR 2019](https://david-m-rosen.github.io/publication/sesync-ijrr/SESync-IJRR.pdf); [Carlone et al. ICRA 2015](https://dellaert.github.io/files/Carlone15icra1.pdf)). Bundle adjustment uses reprojection RMSE and runtime ([arXiv:2409.12190](https://arxiv.org/abs/2409.12190)).

Solution cost is deterministic for a fixed input and algorithm, so its std is zero; the error bars reflect runtime variation.

## Pose Graph Optimization

Six solvers, Levenberg-Marquardt throughout. Cost is computed by the benchmark harness directly from the G2O file for every solver, so values are comparable across implementations. `cost/(m−n)` normalizes by degrees of freedom (m = edges, n = poses) so datasets of different size are comparable.

![Pose graph benchmark](plots/odometry_benchmark.png)

*[Interactive version](plots/odometry_benchmark.html)*

### 2D Datasets (SE2)

| Dataset | Solver | Final Cost | cost/(m−n) | Time (ms) | Iters |
|---------|--------|-----------|------------|-----------|-------|
| **M3500** (3500 poses, 5453 edges) |
| | apex-solver | 1.5109e+00 | **7.737e-04** | 69.9 ± 10.4 | 10 |
| | factrs | 1.5238e+00 | 7.802e-04 | **59.0 ± 1.0** | - |
| | tiny-solver | 2.8604e+04 | 1.465e+01 | 205.4 ± 2.1 | - |
| | Ceres | 4.5437e+03 | 2.327e+00 | 76.3 ± 0.4 | 18 |
| | GTSAM | 1.5111e+00 | **7.737e-04** | 69.1 ± 1.2 | 6 |
| | g2o | 1.5109e+00 | **7.737e-04** | 108.0 ± 0.5 | 33 |
| **mit** (808 poses, 827 edges) |
| | apex-solver | 1.1454e+02 | **6.028e+00** | 82.3 ± 1.0 | 110 |
| | factrs | 1.4831e+04 | 7.806e+02 | **3.3 ± 0.0** | - |
| | tiny-solver | 1.1933e+04 | 6.280e+02 | 5.8 ± 0.2 | - |
| | Ceres | 3.4865e+02 | 1.835e+01 | 11.5 ± 0.1 | 29 |
| | GTSAM | 8.3309e+04 | 4.385e+03 | 42.1 ± 0.7 | 4 |
| | g2o | 1.2571e+03 | 6.616e+01 | 46.3 ± 0.3 | 100 |
| **city10000** (10000 poses, 20687 edges) |
| | apex-solver | 4.3620e+00 | **4.082e-04** | **123.6 ± 0.8** | 5 |
| | factrs | 4.4330e+00 | 4.148e-04 | 226.9 ± 2.7 | - |
| | tiny-solver | 1.2237e+05 | 1.145e+01 | 1097.7 ± 10.9 | - |
| | Ceres | 1.8045e+04 | 1.689e+00 | 393.7 ± 1.7 | 27 |
| | GTSAM | 2.9292e+05 | 2.741e+01 | 2040.8 ± 20.9 | 22 |
| | g2o | 4.4232e+02 | 4.139e-02 | 4178.6 ± 2.8 | 100 |
| **ring** (434 poses, 459 edges) |
| | apex-solver | 2.2179e-02 | **8.872e-04** | 5.0 ± 0.2 | 10 |
| | factrs | 3.0176e-02 | 1.207e-03 | 4.4 ± 0.1 | - |
| | tiny-solver | 9.8712e+02 | 3.948e+01 | 20.8 ± 0.2 | - |
| | Ceres | 2.2188e-02 | 8.875e-04 | **3.2 ± 0.0** | 14 |
| | GTSAM | 2.2179e-02 | **8.872e-04** | 11.1 ± 0.4 | 6 |
| | g2o | 2.2179e-02 | **8.872e-04** | 6.3 ± 0.0 | 34 |

### 3D Datasets (SE3)

| Dataset | Solver | Final Cost | cost/(m−n) | Time (ms) | Iters |
|---------|--------|-----------|------------|-----------|-------|
| **sphere2500** (2500 poses, 4949 edges) |
| | apex-solver | 2.1320e+01 | 8.706e-03 | 209.5 ± 1.5 | 7 |
| | factrs | - | - | - | ✗ |
| | tiny-solver | 4.0584e+04 | 1.657e+01 | 2063.3 ± 10.4 | - |
| | Ceres | 1.1654e+05 | 4.759e+01 | 1120.6 ± 3.7 | 90 |
| | GTSAM | 2.1298e+01 | **8.697e-03** | **131.9 ± 1.5** | 7 |
| | g2o | 6.4554e+01 | 2.636e-02 | 10864.3 ± 18.5 | 84 |
| **parking-garage** (1661 poses, 6275 edges) |
| | apex-solver | 6.2449e-01 | **1.353e-04** | 112.9 ± 0.4 | 6 |
| | factrs | 6.2777e-01 | 1.361e-04 | 445.1 ± 1.8 | - |
| | tiny-solver | 1.2116e+05 | 2.626e+01 | 843.3 ± 11.5 | - |
| | Ceres | 2.0103e+05 | 4.357e+01 | 270.8 ± 1.0 | 34 |
| | GTSAM | 6.2471e-01 | 1.354e-04 | **33.3 ± 0.3** | 3 |
| | g2o | 6.2869e-01 | 1.363e-04 | 628.0 ± 3.3 | 56 |
| **torus3D** (5000 poses, 9048 edges) |
| | apex-solver | 1.0124e+03 | 2.501e-01 | 1347.5 ± 3.5 | 23 |
| | factrs | - | - | - | ✗ |
| | tiny-solver | - | - | - | ✗ |
| | Ceres | 2.3940e+04 | 5.914e+00 | 1014.3 ± 4.4 | 38 |
| | GTSAM | 1.2032e+02 | **2.972e-02** | **638.7 ± 5.1** | 12 |
| | g2o | 1.4131e+02 | 3.491e-02 | 31073.9 ± 28.2 | 96 |
| **cubicle** (5750 poses, 16869 edges) |
| | apex-solver | 5.1827e+02 | 4.661e-02 | 1565.5 ± 6.3 | 21 |
| | factrs | - | - | - | ✗ |
| | tiny-solver | 9.9185e+03 | 8.920e-01 | 1948.2 ± 28.7 | - |
| | Ceres | 1.7144e+04 | 1.542e+00 | 965.4 ± 7.0 | 29 |
| | GTSAM | 5.3897e+00 | **4.847e-04** | **551.0 ± 3.7** | 5 |
| | g2o | 1.2771e+01 | 1.149e-03 | 8483.6 ± 7.6 | 47 |

**Observations**:
- **apex-solver** reaches the lowest cost on all four 2D datasets and on parking-garage, and is fastest on city10000 (3.2× vs Ceres, 34× vs g2o). On mit it is 3× better than the next-best solver.
- **apex-solver is weakest on torus3D and cubicle**, where GTSAM reaches 8× and 96× lower cost respectively. These are the clear optimization targets.
- **GTSAM** is the fastest 3D solver and the most accurate on torus3D and cubicle, but is the *worst* solver on mit (4.385e+03) and city10000 (2.741e+01).
- **g2o** reaches competitive cost but is consistently the slowest — 31 s on torus3D, 10.9 s on sphere2500.
- **factrs** is fast in 2D but fails on three of four 3D datasets; **tiny-solver** rarely reaches a good solution.
- **Ceres** trails on cost throughout; its odometry configuration uses `function_tolerance = 1e-3`, looser than the other solvers, so this reflects benchmark configuration rather than a Ceres limitation.

## Bundle Adjustment (Self-Calibration)

Large-scale BAL datasets, optimizing **camera poses, 3D landmarks, and camera intrinsics simultaneously**. apex-solver uses iterative Schur complement (PCG + Schur-Jacobi preconditioner).

![Bundle adjustment benchmark](plots/ba_benchmark.png)

*[Interactive version](plots/ba_benchmark.html)*

| Dataset | Solver | Cameras | Landmarks | Observations | Final RMSE (px) | Time (s) | Iters |
|---------|--------|---------|-----------|--------------|-----------------|----------|-------|
| **Ladybug** |
| | apex-solver | 1,723 | 156,502 | 678,718 | **0.7700 ± 0.0000** | 86.3 ± 0.6 | 21 |
| | Ceres | 1,723 | 156,502 | 678,718 | 1.1673 ± 0.0008 | **17.9 ± 0.9** | 101 |
| | GTSAM | 1,723 | 156,502 | 678,718 | 0.9812 ± 0.0000 | 82.7 ± 0.2 | 2 |
| | g2o | 1,723 | 156,502 | 678,718 | 13.5074 ± 0.0000 | 150.4 ± 0.1 | 20 |
| **Trafalgar** |
| | apex-solver | 257 | 65,132 | 225,911 | 0.7844 ± 0.0000 | **5.2 ± 0.0** | 10 |
| | Ceres | 257 | 65,132 | 225,911 | 1.3184 ± 0.0173 | 42.7 ± 5.3 | 101 |
| | GTSAM | 257 | 65,132 | 225,911 | **0.6259 ± 0.0000** | 57.6 ± 0.1 | 100 |
| | g2o | 257 | 65,132 | 225,911 | 8.1506 ± 0.0000 | 16.4 ± 0.1 | 20 |
| **Dubrovnik** |
| | apex-solver | 356 | 226,730 | 1,255,268 | 0.7237 ± 0.0000 | 37.6 ± 0.2 | 11 |
| | Ceres | 356 | 226,730 | 1,255,268 | 1.0036 ± 0.0001 | 86.2 ± 14.9 | 101 |
| | GTSAM | 356 | 226,730 | 1,255,268 | **0.5622 ± 0.0000** | 117.9 ± 0.2 | 31 |
| | g2o | 356 | 226,730 | 1,255,268 | 12.1678 ± 0.0000 | **34.8 ± 0.1** | 20 |
| **Venice** (largest) |
| | apex-solver | 1,778 | 993,923 | 5,001,946 | **0.6503 ± 0.0000** | **48.7 ± 0.5** | 2 |
| | Ceres | 1,778 | 993,923 | 5,001,946 | TIMEOUT | TIMEOUT | - |
| | GTSAM | 1,778 | 993,923 | 5,001,946 | TIMEOUT | TIMEOUT | - |
| | g2o | 1,778 | 993,923 | 5,001,946 | 10.1261 ± 0.0000 | 244.9 ± 0.4 | 20 |

**Observations**:
- **Scalability**: apex-solver is the only solver to produce a usable Venice result (5M observations, 0.650 px in 48.7 s). Ceres and GTSAM exceed the 10-minute timeout on all 5 runs; g2o finishes but barely moves the solution (10.128 → 10.126 px).
- **Accuracy**: apex-solver reaches sub-pixel RMSE on all four datasets — best on Ladybug and Venice. GTSAM is more accurate on Trafalgar (0.626) and Dubrovnik (0.562).
- **Speed**: apex-solver is fastest on Trafalgar (8.2× vs Ceres) and Venice. Ceres is 4.8× faster on Ladybug.
- **g2o** never meaningfully reduces reprojection error within its 20-iteration cap.

---

## Reproducing

```bash
# 5 runs each, raw per-run CSVs archived to output/runs/
bash benches/tools/run_repeated.sh odometry_pose_benchmark 5
bash benches/tools/run_repeated.sh bundle_adjustment_benchmark 5

# aggregate to output/*_aggregated.csv and render doc/plots/*.{html,png}
uv run --with plotly --with kaleido --with pandas benches/tools/plot_benchmarks.py
```

---

*Back to [README](../README.md)*
