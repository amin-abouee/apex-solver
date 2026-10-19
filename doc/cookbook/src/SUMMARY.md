# Summary

# Front Matter

[Introduction](./introduction.md)
[How to Read This Book](./how-to-read.md)
[Notation & Conventions](./conventions.md)

---

# Part I — The Problem and Its Machinery

- [What a Factor Graph Is](./core/factor-graph.md)
- [The Problem: Variables, Blocks, Assembly](./core/problem.md)
- [Variables on Manifolds](./core/variables.md)
- [Noise Models and Whitening](./core/noise.md)
- [Robust Loss Functions](./core/losses.md)
- [The Corrector](./core/corrector.md)
- [From Factor Graph to Linear System](./core/linearization.md)
- [The First-Estimate Policy](./core/first-estimate.md)

---

# Part II — Optimizers and Linear Solvers

- [The Optimizer Contract](./optimizer/contract.md)
- [Gauss–Newton](./optimizer/gauss-newton.md)
- [Levenberg–Marquardt](./optimizer/levenberg-marquardt.md)
- [Dog Leg](./optimizer/dog-leg.md)
- [Damping and Column Scaling](./optimizer/damping.md)
- [Dense Linear Solvers](./linalg/dense.md)
- [Sparse Linear Solvers](./linalg/sparse.md)
- [The Schur Complement](./linalg/schur.md)
- [Choosing a Solver](./optimizer/choosing.md)

---

# Part III — The Factors

- [Writing a Factor](./factors/writing.md)
- [Pose and Prior Factors](./factors/pose.md)
- [Visual Factors: Geometry](./factors/visual-geometry.md)
- [Visual Factors: Photometric](./factors/visual-photometric.md)
- [IMU Factors](./factors/imu.md)
- [LiDAR Factors](./factors/lidar.md)
- [Range and Bearing Factors](./factors/ranging.md)
- [Motion Model Factors](./factors/motion.md)
- [Navigation Factors](./factors/navigation.md)
- [Decorators: Aggregate, Held, Offset, Projected](./factors/decorators.md)
- [Common Utilities](./factors/common.md)
- [The Factor Catalogue](./factors/catalogue.md)

---

# Part IV — Marginalization

- [Marginalization: From Factors to a Prior](./marginal/marginalization.md)
- [The Marginal Prior as a Factor](./marginal/prior-factor.md)

---

# Part V — Tools and Appendices

- [Observers: Rerun](./tools/observers.md)
- [Benchmarks and Performance](./tools/benchmarks.md)
- [Migrating 1.4 → 1.5 → 1.6](./tools/migration.md)
- [The Crate Cookbooks](./tools/crate-cookbooks.md)
- [Symbol Glossary](./appendix/glossary.md)
- [Exercises](./appendix/exercises.md)
- [Troubleshooting](./appendix/troubleshooting.md)
