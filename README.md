# 🦀 Apex Solver

A high-performance Rust-based nonlinear least squares optimization library designed for computer vision applications including bundle adjustment, SLAM, and pose graph optimization. Built with focus on zero-cost abstractions, memory safety, and mathematical correctness.

Apex Solver is a comprehensive optimization library that bridges the gap between theoretical robotics and practical implementation. It provides manifold-aware optimization for Lie groups commonly used in computer vision, multiple optimization algorithms with unified interfaces, flexible linear algebra backends supporting both sparse Cholesky and QR decompositions, and industry-standard file format support for seamless integration with existing workflows.

[![Crates.io](https://img.shields.io/crates/v/apex-solver.svg)](https://crates.io/crates/apex-solver)
[![Documentation](https://docs.rs/apex-solver/badge.svg)](https://docs.rs/apex-solver)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## ⚠️ Upgrading to 1.4.0 — breaking API changes

`1.4.0` changes the public API. Code written against `1.3.0` will not compile until you make
the edits below. Full detail in the [changelog](doc/CHANGELOG.md).

**1. `Problem` uses handles instead of string names.** `add_variable` returns a `VarKey`;
`add_residual_block` takes `&[VarKey]` and returns a `FactorKey` (previously `&[&str]` and
`usize`). Keep the returned key and pass it where you used to pass a name:

```rust
// 1.3.0
problem.add_variable("pose_0", ManifoldType::SE3, params);
problem.add_residual_block(&["pose_0", "pose_1"], factor, loss);

// 1.4.0
let k0 = problem.add_variable(ManifoldType::SE3, params);
let k1 = problem.add_variable(ManifoldType::SE3, params_1);
problem.add_residual_block(&[k0, k1], factor, loss);
```

If you need to look variables up later, keep your own `HashMap<YourId, VarKey>` — the
[Quick Start](#quick-start) below shows the pattern.

**2. `Factor::get_dimension` is renamed to `Factor::residual_dim`.** Custom factor
implementations must rename the method; there is no default implementation.

```rust
// 1.3.0                              // 1.4.0
fn get_dimension(&self) -> usize      fn residual_dim(&self) -> usize
```

**3. `OptimizationStatus` gained a `StalledNoProgress` variant.** Exhaustive `match`
expressions need a new arm. Treat it as a *successful* termination — it means the solver
reached a point where the cost can no longer improve.

## Key Features (v1.4.0)

- **Slot-Map Problem Structure (faster)**: Variables and factors are stored in a
  [`slotmap`](https://docs.rs/slotmap)-backed arena and referenced by stable, generational
  `VarKey` / `FactorKey` handles instead of string keys. This gives O(1) access with no
  hashing or per-key allocation on the hot path, and keeps manifold parameters in contiguous
  `nalgebra` storage that `faer` views **without copying** — minimizing data movement between
  the two linear-algebra backends. See [Performance](#performance--data-structure).
- **Bundle Adjustment with Camera Intrinsic Optimization**: Simultaneous optimization of camera poses, 3D landmarks, and camera intrinsics (10 camera models via apex-camera-models crate) [apex-camera-models](crates/apex-camera-models/README.md)
- **Explicit & Implicit Schur Complement Solvers**: Memory-efficient matrix-free PCG for large-scale problems (10,000+ cameras) alongside traditional explicit formulation
- **15 Robust Loss Functions**: Comprehensive outlier rejection (Huber, Cauchy, Tukey, Welsch, Barron, and more)
- **Manifold-Aware**: Full Lie group support (SE2, SE3, SO2, SO3, SE_2(3), SGal(3), Sim(3), Rn) with analytic Jacobians [apex-manifolds](crates/apex-manifolds/README.md)
- **Three Optimization Algorithms**: Levenberg-Marquardt, Gauss-Newton, and Dog Leg with unified interface
- **Prior Factors & Fixed Variables**: Anchor poses with known values and constrain specific parameter indices
- **Uncertainty Quantification**: Covariance estimation for both Cholesky and QR solvers
- **Real-time Visualization**: Integrated [Rerun](https://rerun.io/) support for live debugging of optimization progress
- **I/O**: Read and write G2O, Toro, BAL format files for seamless integration with SLAM ecosystems [apex-io](crates/apex-io/README.md)
- **High Performance**: Sparse linear algebra with persistent symbolic factorization
- **Mathematical Cookbooks**: Full derivations and explanations for [apex-manifolds](crates/apex-manifolds/doc/cookbook/src/introduction.md), [apex-camera-models](crates/apex-camera-models/doc/cookbook/src/introduction.md), and [apex-io](crates/apex-io/doc/cookbook/src/introduction.md)
- **Production-Grade**: Comprehensive error handling, structured tracing, integration test suite

---

## Quick Start

```toml
[dependencies]
apex-solver = "1.4.0"
```

```rust
use apex_solver::core::problem::Problem;
use apex_solver::factors::BetweenFactor;
use apex_solver::{G2oLoader, JacobianMode, ManifoldType};
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::dvector;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load pose graph from G2O file
    let graph = G2oLoader::load("data/odometry/3d/sphere2500.g2o")?;

    // Create optimization problem
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut var_keys = HashMap::new();

    // Add SE3 poses as variables -- returns stable VarKey handles
    for (&id, vertex) in &graph.vertices_se3 {
        let quat = vertex.pose.rotation_quaternion();
        let trans = vertex.pose.translation();
        let se3_data = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];
        let key = problem.add_variable(ManifoldType::SE3, se3_data);
        var_keys.insert(id, key);
    }

    // Add between factors (relative pose constraints) using VarKey handles
    for edge in &graph.edges_se3 {
        let k_from = var_keys[&edge.from];
        let k_to = var_keys[&edge.to];
        problem.add_residual_block(
            &[k_from, k_to],
            Box::new(BetweenFactor::new(edge.measurement.clone())),
            None,  // Optional: add HuberLoss for robustness
        );
    }

    // Configure and run optimizer
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-6)
        .with_compute_covariances(true);  // Enable uncertainty estimation

    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&mut problem)?;

    println!("Status: {:?}", result.status);
    println!("Initial cost: {:.3e}", result.initial_cost);
    println!("Final cost: {:.3e}", result.final_cost);
    println!("Iterations: {}", result.iterations);

    Ok(())
}
```

**Result**:
```
Status: CostToleranceReached
Initial cost: 1.280e+05
Final cost: 2.130e+01
Iterations: 5
```

---

## Architecture

The workspace root is the `apex-solver` crate. Sub-crates for manifolds, I/O, and camera models live in `crates/`:

```
apex-solver/                # workspace root = apex-solver crate
├── src/
│   ├── core/               # Problem formulation, factors, residuals
│   ├── factors/            # Factor implementations (projection, between, prior)
│   ├── optimizer/          # LM, GN, Dog Leg algorithms
│   ├── linalg/             # Cholesky, QR, Explicit/Implicit Schur
│   └── observers/          # Optimization observers and callbacks
├── bin/                    # Executable binaries
├── benches/                # Benchmarks
├── examples/               # Example programs
├── tests/                  # Integration tests
├── doc/                    # Extended documentation
└── crates/
    ├── apex-manifolds/     # Lie groups: SE2, SE3, SO2, SO3, SE_2(3), SGal(3), Sim(3), Rn
    ├── apex-io/            # File I/O: G2O, TORO, BAL formats
    └── apex-camera-models/ # 8 camera projection models
```

**Core Modules** (in `src/`):
- **`core/`**: Optimization problem definitions, residual blocks, robust loss functions, and variable management
- **`optimizer/`**: Three optimization algorithms (Levenberg-Marquardt with adaptive damping, Gauss-Newton, Dog Leg trust region) with real-time visualization support
- **`linalg/`**: Linear algebra backends including sparse Cholesky decomposition, QR factorization, explicit Schur complement, and implicit Schur complement (matrix-free PCG)
- **`observers/`**: Optimization observers and callbacks (Rerun visualization, custom hooks)

**Workspace Sub-crates** (in `crates/`):
- **`apex-manifolds`**: Lie group implementations (SE2, SE3, SO2, SO3, SE_2(3), SGal(3), Sim(3), Rn) with analytic Jacobians
- **`apex-io`**: File format parsers for G2O, TORO, and BAL formats
- **`apex-camera-models`**: Camera projection models with analytic Jacobians (10 models)

**Low-level Dependencies**:
- **`faer`** / **`nalgebra`**: High-performance linear algebra backends

---

## Performance & Data Structure

Apex Solver stores the optimization problem in a **slot-map arena**. `Problem` keeps its
variables and residual blocks in `slotmap::SlotMap`s and returns stable, generational
`VarKey` / `FactorKey` handles; per-variable side data (fixed indices, bounds, column
offsets) lives in matching `SecondaryMap`s.

Why it's faster than the previous string-keyed design:

- **O(1) generational access, no hashing.** Looking up a variable during residual/Jacobian
  assembly is an index + generation check, not a `HashMap<String, _>` hash and compare.
- **No per-key allocation.** `VarKey`/`FactorKey` are `Copy` 8-byte handles; there are no
  `String` keys to allocate, clone, or compare.
- **Cache-friendly iteration.** Values live in a dense backing array, so assembly sweeps
  contiguous memory.
- **Generational safety.** A removed handle can never alias a reused slot — stale keys
  return `None` instead of silently pointing at a different variable.

The handles also enable a **zero-copy nalgebra ↔ faer boundary**: manifold parameters stay in
contiguous `nalgebra` column-major storage and are handed to factors as `&[f64]` slices that
`faer` views directly (`MatRef`/`MatMut::from_column_major_slice`) — no `DVector`↔`Mat`
conversion in the inner loop. Combined with a persistent symbolic factorization (built once,
reused every iteration) and lock-free parallel assembly over disjoint buffers (rayon), the
per-iteration hot path is allocation- and copy-free for the manifold data.

→ **[Full performance benchmarks](doc/performance.md)**

---

## Datasets

Datasets are downloaded on demand using the built-in `download_datasets` tool in the `apex-io` crate. No Git LFS required.

```bash
# List all available datasets and selection numbers
cargo run --release -p apex-io --bin download_datasets -- --list

# Download benchmark datasets (all odometry g2o + largest from each BA dataset)
cargo run --release -p apex-io --bin download_datasets -- --select 10

# Download all odometry g2o datasets (2D + 3D)
cargo run --release -p apex-io --bin download_datasets -- --select 3

# Interactive mode (prompts for selection)
cargo run --release -p apex-io --bin download_datasets
```

Datasets are saved to `data/odometry/` (g2o files) and `data/bundle_adjustment/` (BAL format).

Available datasets:
- **Pose Graph SE2** (2D): `M3500`, `mit`, `city10000`, `ring`
- **Pose Graph SE3** (3D): `sphere2500`, `parking-garage`, `torus3D`, `cubicle`
- **Bundle Adjustment** (UW BAL): `ladybug`, `trafalgar`, `dubrovnik`, `venice`, `final`

---

## Workspace Crates

Apex Solver is organized as a Cargo workspace with specialized sub-crates that can be used independently:

| Crate | Description | Docs | Cookbook |
|-------|-------------|------|----------|
| **[apex-manifolds](crates/apex-manifolds)** | Lie group manifolds (SE2, SE3, SO2, SO3, SE_2(3), SGal(3), Sim(3), Rn) with analytic Jacobians | [README](crates/apex-manifolds/README.md) | [Cookbook](crates/apex-manifolds/doc/cookbook/src/introduction.md) |
| **[apex-camera-models](crates/apex-camera-models)** | 10 camera projection models for bundle adjustment and SLAM | [README](crates/apex-camera-models/README.md) | [Cookbook](crates/apex-camera-models/doc/cookbook/src/introduction.md) |
| **[apex-io](crates/apex-io)** | File I/O utilities for G2O, TORO, and BAL formats | [README](crates/apex-io/README.md) | [Cookbook](crates/apex-io/doc/cookbook/src/introduction.md) |

### Cookbooks

Each sub-crate ships an mdBook cookbook (KaTeX-rendered) that is the mathematical reference
for its domain — derived from the implementation, not restated from papers:

- **[apex-manifolds](crates/apex-manifolds/doc/cookbook/src/introduction.md)** — every group
  and operation: exp/log, adjoints, left/right Jacobians and inverses, ⊞/⊟, plus a shared
  [Conventions](crates/apex-manifolds/doc/cookbook/src/manifolds/conventions.md) page
  documenting w-first quaternions and `[ρ, θ]` twist order.
- **[apex-camera-models](crates/apex-camera-models/doc/cookbook/src/introduction.md)** — one
  chapter per model on an eight-section template (Parameters → Projection → Inverse
  Projection → Point Jacobian → Intrinsic Jacobian → Linear Estimation → Example →
  References), with validity conditions merged into the projection sections.
- **[apex-io](crates/apex-io/doc/cookbook/src/introduction.md)** — every public capability by
  domain: pose-graph formats, ASL/EuRoC, ROS1/ROS2 bags, DDS, CLI tools, and a feature-flag
  reference.

Build any of them locally:

```bash
cargo install mdbook mdbook-katex
mdbook build crates/apex-manifolds/doc/cookbook      # then open book/index.html
```

**Using sub-crates independently:**

```toml
[dependencies]
apex-manifolds = "0.3.0"

[dependencies]
apex-camera-models = "0.3.0"

[dependencies]
apex-io = "0.3.0"
```

---

## Performance Benchmarks

Detailed benchmark tables comparing apex-solver against Ceres, GTSAM, g2o, factrs, and
tiny-solver on 8 pose-graph datasets (SE2/SE3) and 4 BAL bundle-adjustment datasets.

→ **[Full performance benchmarks](doc/performance.md)**

---

## Examples

Usage examples covering pose graph optimization, custom factor implementation, and
self-calibration bundle adjustment.

→ **[Full examples](doc/examples.md)**

---

## Technical Implementation

### Robust Loss Functions

15 robust loss functions for handling outliers in optimization:

- **L2Loss**: Standard least squares (no outliers)
- **L1Loss**: Linear growth (light outliers)
- **HuberLoss**: Quadratic near zero, linear after threshold (moderate outliers)
- **CauchyLoss**: Logarithmic growth (heavy outliers)
- **FairLoss**, **GemanMcClureLoss**, **WelschLoss**, **TukeyBiweightLoss**, **AndrewsWaveLoss**: Various robustness profiles
- **RamsayEaLoss**: Asymmetric outliers
- **TrimmedMeanLoss**: Ignores worst residuals
- **LpNormLoss**: Generalized Lp norm
- **BarronGeneralLoss**, **AdaptiveBarronLoss**: Adaptive robustness
- **TDistributionLoss**: Statistical outliers

**Usage**:
```rust
use apex_solver::core::loss_functions::HuberLoss;

let loss = HuberLoss::new(1.345);  // 95% efficiency threshold
problem.add_residual_block(Box::new(factor), Some(Box::new(loss)));
```

### Optimization Algorithms

#### Levenberg-Marquardt (Recommended)
- Adaptive damping between gradient descent and Gauss-Newton
- Robust convergence from poor initial estimates
- Supports covariance estimation for uncertainty quantification
- 9 comprehensive termination criteria (gradient norm, cost change, trust region radius, etc.)

#### Gauss-Newton
- Fast convergence near solution
- Minimal memory requirements
- Best for well-initialized problems

#### Dog Leg Trust Region
- Combines steepest descent and Gauss-Newton
- Global convergence guarantees
- Adaptive trust region management

### Linear Algebra Backends

Four sparse linear solvers for different use cases:

- **Sparse Cholesky**: Direct factorization of J^T J + λI - fast, moderate memory, best for well-conditioned problems
- **Sparse QR**: QR factorization of Jacobian - robust for rank-deficient systems, slightly slower
- **Explicit Schur Complement**: Constructs reduced camera matrix S = B - E C⁻¹ Eᵀ explicitly in memory - most accurate for bundle adjustment, moderate memory usage
- **Implicit Schur Complement**: Matrix-free PCG solver computing only S·x products - memory-efficient for large-scale problems (10,000+ cameras), highly scalable

Configure via `LinearSolverType` in optimizer config:
```rust
config.with_linear_solver_type(LinearSolverType::ExplicitSchur)  // For bundle adjustment
config.with_linear_solver_type(LinearSolverType::ImplicitSchur)  // For very large BA
```

---

## Interactive Visualization

Real-time optimization debugging with integrated [Rerun](https://rerun.io/) visualization using the observer pattern:

```rust
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};

let config = LevenbergMarquardtConfig::new()
    .with_max_iterations(100);

let mut solver = LevenbergMarquardt::with_config(config);

// Add Rerun visualization observer (requires `visualization` feature)
#[cfg(feature = "visualization")]
{
    use apex_solver::observers::RerunObserver;
    solver.add_observer(RerunObserver::new(true)?);  // true = spawn viewer
}

let result = solver.optimize(&mut problem)?;
```

**Visualized Metrics**:
- Time series: Cost, gradient norm, damping (λ), step quality (ρ), step norm
- Matrix visualizations: Hessian heat map, gradient vector
- 3D poses: SE3 camera frusta, SE2 2D points

**Run Examples**:
```bash
# Enable visualization feature and run
cargo run --release --features visualization --bin pose_graph_g2o -- --dataset sphere2500 --with-visualizer
cargo run --release --features visualization --bin pose_graph_g2o -- --dataset intel --with-visualizer
```

> **Note:** The data files (e.g., `sphere2500.g2o`) must be downloaded first.
> See [Datasets](#datasets) — run `cargo run --release -p apex-io --bin download_datasets -- --select 10` to get all benchmark datasets.

Zero overhead when disabled (feature-gated).

---

## Learning Resources

### Computer Vision Background
- [Multiple View Geometry](https://www.robots.ox.ac.uk/~vgg/hzbook/) (Hartley & Zisserman) - Mathematical foundations
- [Visual SLAM algorithms](http://www.robots.ox.ac.uk/~ian/Teaching/SLAMLect/) (Durrant-Whyte & Bailey) - Probabilistic robotics
- [g2o documentation](https://github.com/RainerKuemmerle/g2o) - Reference C++ implementation

### Lie Group Theory
- [A micro Lie theory](https://arxiv.org/abs/1812.01537) (Solà et al.) - Practical introduction
- [manif library](https://github.com/artivis/manif) - C++ reference we follow
- [State Estimation for Robotics](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf) (Barfoot) - SO(3) and SE(3)

### Optimization Theory
- [Numerical Optimization](https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf) (Nocedal & Wright) - Standard reference
- [Trust Region Methods](https://doi.org/10.1137/1.9780898719857) - Dog Leg theory
- [Ceres Solver Tutorial](http://ceres-solver.org/nnls_tutorial.html) - Practical guide

---

## Acknowledgments

Apex Solver draws inspiration and reference implementations from:

- **[Ceres Solver](http://ceres-solver.org/)** - Google's C++ optimization library
- **[g2o](https://github.com/RainerKuemmerle/g2o)** - General framework for graph optimization
- **[GTSAM](https://gtsam.org/)** - Georgia Tech Smoothing and Mapping library
- **[tiny-solver](https://github.com/ceres-solver/tiny-solver)** - Lightweight nonlinear least squares solver
- **[factrs](https://github.com/msabate00/factrs)** - Rust factor graph optimization library
- **[faer](https://github.com/sarah-ek/faer-rs)** - High-performance linear algebra library for Rust
- **[manif](https://github.com/artivis/manif)** - C++ Lie theory library (for manifold conventions)
- **[nalgebra](https://nalgebra.org/)** - Geometry and linear algebra primitives

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---
