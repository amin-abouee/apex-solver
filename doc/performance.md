# Performance Benchmarks

**Hardware**: Apple Mac Mini M4, 64GB RAM
**Build**: Rust release (`opt-level=3`, LTO); C++ `-O3 -DNDEBUG -march=native`
**Methodology**: Pose graph — average of 5 runs. Bundle adjustment — single run, 10-minute timeout per solver. Timing covers `optimize()` only; problem setup and metric computation are excluded.

## Pose Graph Optimization

Six solvers on standard pose graph datasets, Levenberg-Marquardt throughout. Cost columns use the unweighted metric (`0.5 · Σ‖r‖²`) computed from the raw G2O file.

### 2D Datasets (SE2)

| Dataset | Solver | Lang | Time (ms) | Iters | Init Cost | Final Cost | Improve % | Conv |
|---------|--------|------|-----------|-------|-----------|------------|-----------|------|
| **M3500** (3500 vertices, 5453 edges) |
| | apex-solver | Rust | **56.3** | 10 | 2.86e4 | 1.51e0 | 99.99 | ✓ |
| | factrs | Rust | 57.5 | - | 2.86e4 | 1.52e0 | 99.99 | ✓ |
| | tiny-solver | Rust | 196.7 | - | 3.65e4 | 2.86e4 | 21.67 | ✓ |
| | Ceres | C++ | 77.0 | 18 | 2.86e4 | 4.54e3 | 84.14 | ✓ |
| | g2o | C++ | 121.7 | 33 | 2.86e4 | 1.51e0 | 99.99 | ✓ |
| | GTSAM | C++ | 70.8 | 6 | 2.86e4 | 1.51e0 | 99.99 | ✓ |
| **mit** (808 vertices, 827 edges) |
| | apex-solver | Rust | 82.0 | 110 | 1.63e5 | 1.15e2 | 99.93 | ✓ |
| | factrs | Rust | **3.3** | - | 1.63e5 | 1.48e4 | 90.91 | ✓ |
| | tiny-solver | Rust | 5.7 | - | 5.78e4 | 1.19e4 | 79.34 | ✓ |
| | Ceres | C++ | 11.5 | 29 | 1.63e5 | 3.49e2 | 99.79 | ✓ |
| | g2o | C++ | 59.6 | 100 | 1.63e5 | 1.26e3 | 99.23 | ✓ |
| | GTSAM | C++ | 46.9 | 4 | 1.63e5 | 8.33e4 | 48.94 | ✓ |
| **city10000** (10000 vertices, 20687 edges) |
| | apex-solver | Rust | **122.0** | 5 | 7.18e6 | 4.36e0 | 100.00 | ✓ |
| | factrs | Rust | 222.2 | - | 7.18e6 | 4.43e0 | 100.00 | ✓ |
| | tiny-solver | Rust | 1078.2 | - | 4.96e6 | 1.22e5 | 97.53 | ✓ |
| | Ceres | C++ | 403.8 | 27 | 7.18e6 | 1.80e4 | 99.75 | ✓ |
| | g2o | C++ | 4342.1 | 100 | 7.18e6 | 4.42e2 | 99.99 | ✓ |
| | GTSAM | C++ | 2087.6 | 22 | 7.18e6 | 2.93e5 | 95.92 | ✓ |
| **ring** (434 vertices, 459 edges) |
| | apex-solver | Rust | 5.3 | 10 | 1.02e4 | 2.22e-2 | 100.00 | ✓ |
| | factrs | Rust | 4.2 | - | 1.02e4 | 3.02e-2 | 100.00 | ✓ |
| | tiny-solver | Rust | 20.3 | - | 3.17e3 | 9.87e2 | 68.81 | ✓ |
| | Ceres | C++ | **3.2** | 14 | 1.02e4 | 2.22e-2 | 100.00 | ✓ |
| | g2o | C++ | 8.3 | 34 | 1.02e4 | 2.22e-2 | 100.00 | ✓ |
| | GTSAM | C++ | 13.8 | 6 | 1.02e4 | 2.22e-2 | 100.00 | ✓ |

### 3D Datasets (SE3)

| Dataset | Solver | Lang | Time (ms) | Iters | Init Cost | Final Cost | Improve % | Conv |
|---------|--------|------|-----------|-------|-----------|------------|-----------|------|
| **sphere2500** (2500 vertices, 4949 edges) |
| | apex-solver | Rust | 208.4 | 7 | 1.28e5 | 2.13e1 | 99.98 | ✓ |
| | factrs | Rust | - | - | - | - | - | ✗ |
| | tiny-solver | Rust | 2037.9 | - | 4.08e4 | 4.06e4 | 0.48 | ✓ |
| | Ceres | C++ | 1131.9 | 90 | 8.26e7 | 1.43e5 | 99.83 | ✓ |
| | g2o | C++ | 11089.9 | 84 | 8.26e7 | 3.89e3 | 100.00 | ✓ |
| | GTSAM | C++ | **142.2** | 7 | 8.26e7 | 1.01e4 | 99.99 | ✓ |
| **parking-garage** (1661 vertices, 6275 edges) |
| | apex-solver | Rust | 112.6 | 6 | 8.36e3 | 6.24e-1 | 99.99 | ✓ |
| | factrs | Rust | 453.3 | - | 8.36e3 | 6.28e-1 | 99.99 | ✓ |
| | tiny-solver | Rust | 901.6 | - | 1.21e5 | 1.21e5 | -0.05 | ✓ |
| | Ceres | C++ | 269.4 | 34 | 1.22e8 | 1.17e6 | 99.04 | ✓ |
| | g2o | C++ | 631.2 | 56 | 1.22e8 | 2.82e6 | 97.70 | ✓ |
| | GTSAM | C++ | **37.8** | 3 | 1.22e8 | 4.79e6 | 96.08 | ✓ |
| **torus3D** (5000 vertices, 9048 edges) |
| | apex-solver | Rust | 6127.4 | 101 | 1.91e4 | 1.01e3 | 94.70 | ✓ |
| | factrs | Rust | - | - | - | - | - | ✗ |
| | tiny-solver | Rust | - | - | - | - | - | ✗ |
| | Ceres | C++ | 1026.5 | 38 | 2.30e5 | 3.90e4 | 83.02 | ✓ |
| | g2o | C++ | 31796.7 | 96 | 2.30e5 | 1.52e5 | 34.04 | ✓ |
| | GTSAM | C++ | **645.2** | 12 | 2.30e5 | 3.10e5 | -34.88 | ✗ |
| **cubicle** (5750 vertices, 16869 edges) |
| | apex-solver | Rust | 7774.8 | 101 | 3.19e4 | 5.18e2 | 98.37 | ✓ |
| | factrs | Rust | - | - | - | - | - | ✗ |
| | tiny-solver | Rust | 1976.9 | - | 1.14e4 | 9.92e3 | 12.62 | ✓ |
| | Ceres | C++ | 978.4 | 29 | 8.41e6 | 1.96e4 | 99.77 | ✓ |
| | g2o | C++ | 8598.4 | 47 | 8.41e6 | 2.17e5 | 97.42 | ✓ |
| | GTSAM | C++ | **563.1** | 5 | 8.41e6 | 7.52e5 | 91.05 | ✓ |

**Observations**:
- **apex-solver**: fastest on M3500 and city10000, lowest final cost on all 2D datasets. On city10000 it is 3.3× faster than Ceres and 36× faster than g2o. Slower on torus3D and cubicle, where it uses its full 101-iteration budget.
- **GTSAM**: fastest 3D solver, but weakest 3D accuracy — increases cost on torus3D (−34.88%).
- **g2o**: reaches good final costs but is consistently the slowest, up to 31.8 s on torus3D.
- **factrs**: competitive in 2D, fastest on mit; fails on three of four 3D datasets.
- **tiny-solver**: weak convergence — 0.48% improvement on sphere2500, −0.05% on parking-garage.

## Bundle Adjustment (Self-Calibration)

Large-scale BAL datasets, optimizing **camera poses, 3D landmarks, and camera intrinsics simultaneously**. apex-solver uses iterative Schur complement (PCG + Schur-Jacobi preconditioner). Timeout: 10 minutes per solver.

| Dataset | Solver | Lang | Cameras | Landmarks | Observations | Init RMSE | Final RMSE | Time (s) | Iters |
|---------|--------|------|---------|-----------|--------------|-----------|------------|----------|-------|
| **Ladybug** |
| | apex-solver | Rust | 1,723 | 156,502 | 678,718 | 1.955 | **0.770** | 88.62 | 21 |
| | Ceres | C++ | 1,723 | 156,502 | 678,718 | 13.518 | 1.166 | **17.26** | 101 |
| | GTSAM | C++ | 1,723 | 156,502 | 678,718 | 1.857 | 0.981 | 83.79 | 2 |
| | g2o | C++ | 1,723 | 156,502 | 678,718 | 13.518 | 13.507 | 152.49 | 20 |
| **Trafalgar** |
| | apex-solver | Rust | 257 | 65,132 | 225,911 | 2.874 | 0.784 | **5.48** | 10 |
| | Ceres | C++ | 257 | 65,132 | 225,911 | 14.753 | 1.305 | 50.32 | 101 |
| | GTSAM | C++ | 257 | 65,132 | 225,911 | 2.798 | **0.626** | 58.83 | 100 |
| | g2o | C++ | 257 | 65,132 | 225,911 | 14.753 | 8.151 | 16.31 | 20 |
| **Dubrovnik** |
| | apex-solver | Rust | 356 | 226,730 | 1,255,268 | 2.890 | 0.724 | **40.26** | 11 |
| | Ceres | C++ | 356 | 226,730 | 1,255,268 | 12.975 | 1.004 | 93.34 | 101 |
| | GTSAM | C++ | 356 | 226,730 | 1,255,268 | 2.812 | **0.562** | 119.51 | 31 |
| | g2o | C++ | 356 | 226,730 | 1,255,268 | 12.975 | 12.168 | 34.54 | 20 |
| **Venice** (largest) |
| | apex-solver | Rust | 1,778 | 993,923 | 5,001,946 | 2.371 | **0.650** | **51.92** | 2 |
| | Ceres | C++ | 1,778 | 993,923 | 5,001,946 | - | - | TIMEOUT | - |
| | GTSAM | C++ | 1,778 | 993,923 | 5,001,946 | - | - | TIMEOUT | - |
| | g2o | C++ | 1,778 | 993,923 | 5,001,946 | 10.128 | 10.126 | 249.09 | 20 |

**Observations**:
- **Scalability**: apex-solver is the only solver to produce a usable Venice result (5M observations, 0.650 px in 51.92 s). Ceres and GTSAM exceed the 10-minute timeout; g2o finishes but barely moves the solution (10.128 → 10.126 px).
- **Accuracy**: apex-solver reaches sub-pixel RMSE on all four datasets — best on Ladybug and Venice; GTSAM is better on Trafalgar (0.626) and Dubrovnik (0.562).
- **Speed**: apex-solver is fastest on Trafalgar (9.2× vs Ceres), Dubrovnik (2.3× vs Ceres) and Venice. Ceres is 5× faster on Ladybug.
- **g2o**: never meaningfully reduces reprojection error on Ladybug, Dubrovnik or Venice within its 20-iteration cap.

---

*Back to [README](../README.md)*
