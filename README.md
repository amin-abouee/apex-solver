# 🦀 Apex Solver

A high-performance Rust-based nonlinear least squares optimization library designed for computer vision applications including bundle adjustment, SLAM, and pose graph optimization. Built with focus on zero-cost abstractions, memory safety, and mathematical correctness.

Apex Solver is a comprehensive optimization library that bridges the gap between theoretical robotics and practical implementation. It provides manifold-aware optimization for Lie groups commonly used in computer vision, multiple optimization algorithms with unified interfaces, flexible linear algebra backends supporting both sparse Cholesky and QR decompositions, and industry-standard file format support for seamless integration with existing workflows.

[![Crates.io](https://img.shields.io/crates/v/apex-solver.svg)](https://crates.io/crates/apex-solver)
[![Documentation](https://docs.rs/apex-solver/badge.svg)](https://docs.rs/apex-solver)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Key Features (v1.0.0)

- **📷 Bundle Adjustment with Camera Intrinsic Optimization**: Simultaneous optimization of camera poses, 3D landmarks, and camera intrinsics (11 camera models via apex-camera-models crate)
- **🔧 Explicit & Implicit Schur Complement Solvers**: Memory-efficient matrix-free PCG for large-scale problems (10,000+ cameras) alongside traditional explicit formulation
- **🛡️ 15 Robust Loss Functions**: Comprehensive outlier rejection (Huber, Cauchy, Tukey, Welsch, Barron, and more)
- **📐 Manifold-Aware**: Full Lie group support (SE2, SE3, SO2, SO3) with analytic Jacobians
- **🚀 Three Optimization Algorithms**: Levenberg-Marquardt, Gauss-Newton, and Dog Leg with unified interface
- **📌 Prior Factors & Fixed Variables**: Anchor poses with known values and constrain specific parameter indices
- **📊 Uncertainty Quantification**: Covariance estimation for both Cholesky and QR solvers
- **🎨 Real-time Visualization**: Integrated [Rerun](https://rerun.io/) support for live debugging of optimization progress
- **📝 G2O I/O**: Read and write G2O format files for seamless integration with SLAM ecosystems
- **⚡ High Performance**: Sparse linear algebra with persistent symbolic factorization
- **✅ Production-Grade**: Comprehensive error handling, structured tracing, integration test suite

---

## 🚀 Quick Start

```rust
use std::collections::HashMap;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BetweenFactor, PriorFactor};
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::linalg::LinearSolverType;
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::dvector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load pose graph from G2O file
    let graph = G2oLoader::load("data/odometry/sphere2500.g2o")?;
    
    // Create optimization problem
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();
    
    // Add SE3 poses as variables
    for (&id, vertex) in &graph.vertices_se3 {
        let var_name = format!("x{}", id);
        let quat = vertex.pose.rotation_quaternion();
        let trans = vertex.pose.translation();
        let se3_data = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];
        initial_values.insert(var_name, (ManifoldType::SE3, se3_data));
    }
    
    // Add between factors (relative pose constraints)
    for edge in &graph.edges_se3 {
        let factor = BetweenFactor::new(edge.measurement.clone());
        problem.add_residual_block(
            &[&format!("x{}", edge.from), &format!("x{}", edge.to)],
            Box::new(factor),
            None,  // Optional: add HuberLoss for robustness
        );
    }
    
    // Configure and run optimizer
    let config = LevenbergMarquardtConfig::new()
        .with_linear_solver_type(LinearSolverType::SparseCholesky)
        .with_max_iterations(100)
        .with_cost_tolerance(1e-6)
        .with_compute_covariances(true);  // Enable uncertainty estimation
    
    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&problem, &initial_values)?;
    
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

## 🏗️ Architecture

The library is organized into five core modules:

```
apex-solver/
├── src/
│   ├── core/           # Problem formulation, factors, residuals
│   ├── factors/        # Factor implementations (projection, between, prior)
│   ├── optimizer/      # LM, GN, Dog Leg algorithms
│   ├── linalg/         # Cholesky, QR, Explicit/Implicit Schur
│   ├── manifold/       # SE2, SE3, SO2, SO3, Rn
│   ├── observers/      # Optimization observers and callbacks
│   └── io/             # G2O, TORO, TUM file formats
├── bin/                # Executable binaries
├── benches/            # Benchmarks (Rust + C++ comparisons)
├── examples/           # Example programs
├── tests/              # Integration tests
└── doc/                # Extended documentation
```

**Core Modules**:
- **`core/`**: Optimization problem definitions, residual blocks, robust loss functions, and variable management
- **`optimizer/`**: Three optimization algorithms (Levenberg-Marquardt with adaptive damping, Gauss-Newton, Dog Leg trust region) with real-time visualization support
- **`linalg/`**: Linear algebra backends including sparse Cholesky decomposition, QR factorization, explicit Schur complement, and implicit Schur complement (matrix-free PCG)
- **`manifold/`**: Lie group implementations (SE2/SE3 for rigid transformations, SO2/SO3 for rotations, Rn for Euclidean space) with analytic Jacobians
- **`io/`**: File format parsers for G2O, TORO, and TUM trajectory formats

**Core Dependencies**:
- **`apex-manifolds`**: Lie group manifolds (SE2, SE3, SO2, SO3) with exponential/logarithmic maps
- **`apex-camera-models`**: Camera projection models with analytic Jacobians (11 models: Pinhole, RadTan, Kannala-Brandt, Double Sphere, FOV, UCM, EUCM, Equidistant, Orthographic, BAL variants)
- **`faer`** / **`nalgebra`**: High-performance linear algebra backends

---

## 📊 Performance Benchmarks

**Hardware**: Apple Mac Mini M4, 64GB RAM  
**Methodology**: Average over multiple runs

### Pose Graph Optimization

Performance comparison across 6 optimization libraries on standard pose graph datasets. All benchmarks use Levenberg-Marquardt algorithm with consistent parameters (max_iterations=100, cost_tolerance=1e-4).

**Metrics**: Wall-clock time (ms), iterations, initial/final cost, convergence status

#### 2D Datasets (SE2)

| Dataset | Solver | Lang | Time (ms) | Iters | Init Cost | Final Cost | Improve % | Conv |
|---------|--------|------|-----------|-------|-----------|------------|-----------|------|
| **intel** (1228 vertices, 1483 edges) |
| | apex-solver | Rust | 28.5 | 12 | 3.68e4 | 3.89e-1 | 100.00 | ✓ |
| | factrs | Rust | 2.9 | - | 3.68e4 | 8.65e3 | 76.47 | ✓ |
| | tiny-solver | Rust | 87.9 | - | 1.97e4 | 4.56e3 | 76.91 | ✓ |
| | Ceres | C++ | 9.0 | 13 | 3.68e4 | 2.34e2 | 99.36 | ✓ |
| | g2o | C++ | 74.0 | 100 | 3.68e4 | 3.15e0 | 99.99 | ✓ |
| | GTSAM | C++ | 39.0 | 11 | 3.68e4 | 3.89e-1 | 100.00 | ✓ |
| **mit** (808 vertices, 827 edges) |
| | apex-solver | Rust | 140.7 | 107 | 1.63e5 | 1.10e2 | 99.93 | ✓ |
| | factrs | Rust | 3.5 | - | 1.63e5 | 1.48e4 | 90.91 | ✓ |
| | tiny-solver | Rust | 5.7 | - | 5.78e4 | 1.19e4 | 79.34 | ✓ |
| | Ceres | C++ | 11.0 | 29 | 1.63e5 | 3.49e2 | 99.79 | ✓ |
| | g2o | C++ | 46.0 | 100 | 1.63e5 | 1.26e3 | 99.23 | ✓ |
| | GTSAM | C++ | 39.0 | 4 | 1.63e5 | 8.33e4 | 48.94 | ✓ |
| **M3500** (3500 vertices, 5453 edges) |
| | apex-solver | Rust | 103.5 | 10 | 2.86e4 | 1.51e0 | 99.99 | ✓ |
| | factrs | Rust | 62.6 | - | 2.86e4 | 1.52e0 | 99.99 | ✓ |
| | tiny-solver | Rust | 200.1 | - | 3.65e4 | 2.86e4 | 21.67 | ✓ |
| | Ceres | C++ | 77.0 | 18 | 2.86e4 | 4.54e3 | 84.14 | ✓ |
| | g2o | C++ | 108.0 | 33 | 2.86e4 | 1.51e0 | 99.99 | ✓ |
| | GTSAM | C++ | 67.0 | 6 | 2.86e4 | 1.51e0 | 99.99 | ✓ |
| **ring** (434 vertices, 459 edges) |
| | apex-solver | Rust | 8.5 | 10 | 1.02e4 | 2.22e-2 | 100.00 | ✓ |
| | factrs | Rust | 4.8 | - | 1.02e4 | 3.02e-2 | 100.00 | ✓ |
| | tiny-solver | Rust | 21.0 | - | 3.17e3 | 9.87e2 | 68.81 | ✓ |
| | Ceres | C++ | 3.0 | 14 | 1.02e4 | 2.22e-2 | 100.00 | ✓ |
| | g2o | C++ | 6.0 | 34 | 1.02e4 | 2.22e-2 | 100.00 | ✓ |
| | GTSAM | C++ | 10.0 | 6 | 1.02e4 | 2.22e-2 | 100.00 | ✓ |

#### 3D Datasets (SE3)

| Dataset | Solver | Lang | Time (ms) | Iters | Init Cost | Final Cost | Improve % | Conv |
|---------|--------|------|-----------|-------|-----------|------------|-----------|------|
| **sphere2500** (2500 vertices, 4949 edges) |
| | apex-solver | Rust | 176.3 | 5 | 1.28e5 | 2.13e1 | 99.98 | ✓ |
| | factrs | Rust | 334.8 | - | 1.28e5 | 3.49e1 | 99.97 | ✓ |
| | tiny-solver | Rust | 2020.3 | - | 4.08e4 | 4.06e4 | 0.48 | ✓ |
| | Ceres | C++ | 1447.0 | 101 | 8.26e7 | 8.25e5 | 99.00 | ✓ |
| | g2o | C++ | 10919.0 | 84 | 8.26e7 | 3.89e3 | 100.00 | ✓ |
| | GTSAM | C++ | 138.0 | 7 | 8.26e7 | 1.01e4 | 99.99 | ✓ |
| **parking-garage** (1661 vertices, 6275 edges) |
| | apex-solver | Rust | 153.1 | 6 | 8.36e3 | 6.24e-1 | 99.99 | ✓ |
| | factrs | Rust | 453.1 | - | 8.36e3 | 6.28e-1 | 99.99 | ✓ |
| | tiny-solver | Rust | 849.2 | - | 1.21e5 | 1.21e5 | -0.05 | ✓ |
| | Ceres | C++ | 344.0 | 36 | 1.22e8 | 4.84e5 | 99.60 | ✓ |
| | g2o | C++ | 635.0 | 56 | 1.22e8 | 2.82e6 | 97.70 | ✓ |
| | GTSAM | C++ | 31.0 | 3 | 1.22e8 | 4.79e6 | 96.08 | ✓ |
| **torus3D** (5000 vertices, 9048 edges) |
| | apex-solver | Rust | 1780.5 | 27 | 1.91e4 | 1.20e2 | 99.37 | ✓ |
| | factrs | Rust | - | - | - | - | - | ✗ |
| | tiny-solver | Rust | - | - | - | - | - | ✗ |
| | Ceres | C++ | 1063.0 | 34 | 2.30e5 | 3.85e4 | 83.25 | ✓ |
| | g2o | C++ | 31279.0 | 96 | 2.30e5 | 1.52e5 | 34.04 | ✓ |
| | GTSAM | C++ | 647.0 | 12 | 2.30e5 | 3.10e5 | -34.88 | ✗ |
| **cubicle** (5750 vertices, 16869 edges) |
| | apex-solver | Rust | 512.0 | 5 | 3.19e4 | 5.38e0 | 99.98 | ✓ |
| | factrs | Rust | - | - | - | - | - | ✗ |
| | tiny-solver | Rust | 1975.8 | - | 1.14e4 | 9.92e3 | 12.62 | ✓ |
| | Ceres | C++ | 1457.0 | 36 | 8.41e6 | 1.95e4 | 99.77 | ✓ |
| | g2o | C++ | 8533.0 | 47 | 8.41e6 | 2.17e5 | 97.42 | ✓ |
| | GTSAM | C++ | 558.0 | 5 | 8.41e6 | 7.52e5 | 91.05 | ✓ |

**Key Observations**:
- **apex-solver**: 100% convergence rate (8/8 datasets), most reliable Rust solver
- **Ceres/g2o**: 100% convergence but often slower (especially g2o)
- **GTSAM**: Fast when it converges, but diverged on torus3D (87.5% rate)
- **factrs**: Fast on 2D but panics on large 3D problems (62.5% rate)
- **tiny-solver**: Convergence issues on several datasets (75% rate)

---

### Bundle Adjustment (Self-Calibration)

Large-scale bundle adjustment benchmarks optimizing **camera poses, 3D landmarks, and camera intrinsics simultaneously**. Tests self-calibration capability on real-world structure-from-motion datasets from the Bundle Adjustment in the Large (BAL) collection.

| Dataset | Solver | Lang | Cameras | Landmarks | Observations | Init RMSE | Final RMSE | Time (s) | Iters | Status |
|---------|--------|------|---------|-----------|--------------|-----------|------------|----------|-------|--------|
| **Dubrovnik** |
| | Apex-Iterative | Rust | 356 | 226,730 | 1,255,268 | 2.043 | 0.533 | 47.16 | 9 | ✓ |
| | Ceres | C++ | 356 | 226,730 | 1,255,268 | 12.975 | 1.004 | 2879.23 | 101 | ✗ |
| | GTSAM | C++ | 356 | 226,730 | 1,255,268 | 2.812 | 0.562 | 196.72 | 31 | ✓ |
| | g2o | C++ | 356 | 226,730 | 1,255,268 | 12.975 | 12.168 | 34.67 | 20 | ✓ |
| **Ladybug** |
| | Apex-Iterative | Rust | 1,723 | 156,502 | 678,718 | 1.382 | 0.537 | 146.69 | 30 | ✓ |
| | Ceres | C++ | 1,723 | 156,502 | 678,718 | 13.518 | 1.168 | 17.53 | 101 | ✗ |
| | GTSAM | C++ | 1,723 | 156,502 | 678,718 | 1.857 | 0.981 | 95.46 | 2 | ✓ |
| | g2o | C++ | 1,723 | 156,502 | 678,718 | 13.518 | 13.507 | 150.46 | 20 | ✓ |
| **Trafalgar** |
| | Apex-Iterative | Rust | 257 | 65,132 | 225,911 | 2.033 | 0.679 | 10.39 | 14 | ✓ |
| | Ceres | C++ | 257 | 65,132 | 225,911 | 14.753 | 1.320 | 44.14 | 101 | ✗ |
| | GTSAM | C++ | 257 | 65,132 | 225,911 | 2.798 | 0.626 | 77.64 | 100 | ✓ |
| | g2o | C++ | 257 | 65,132 | 225,911 | 14.753 | 8.151 | 16.11 | 20 | ✓ |
| **Venice** (Largest) |
| | Apex-Iterative | Rust | 1,778 | 993,923 | 5,001,946 | 1.676 | 0.458 | 83.17 | 2 | ✓ |
| | Ceres | C++ | 1,778 | 993,923 | 5,001,946 | - | - | TIMEOUT | - | ✗ |
| | GTSAM | C++ | 1,778 | 993,923 | 5,001,946 | - | - | TIMEOUT | - | ✗ |
| | g2o | C++ | 1,778 | 993,923 | 5,001,946 | 10.128 | 10.126 | 252.17 | 20 | ✓ |

**Key Results**:
- **Apex-Iterative**: 100% convergence rate (4/4 datasets), handles up to 5M observations efficiently
- **Superior scalability**: Only solver alongside g2o to complete Venice dataset; Ceres and GTSAM timeout after 10 minutes
- **Best accuracy on largest dataset**: Achieves 0.458 RMSE on Venice (5M observations) in only 2 iterations
- **Speed advantage**: 61x faster than Ceres on Dubrovnik, 4x faster on Trafalgar (where Ceres converged)

---

## 📊 Examples

### Example 1: Basic Pose Graph Optimization

```rust
use std::collections::HashMap;
use apex_solver::core::problem::Problem;
use apex_solver::factors::BetweenFactor;
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::dvector;

// Load pose graph
let graph = G2oLoader::load("data/odometry/sphere2500.g2o")?;

// Build optimization problem
let mut problem = Problem::new();
let mut initial_values = HashMap::new();

// Add SE3 poses as variables
for (&id, vertex) in &graph.vertices_se3 {
    let quat = vertex.pose.rotation_quaternion();
    let trans = vertex.pose.translation();
    initial_values.insert(
        format!("x{}", id),
        (ManifoldType::SE3, dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k])
    );
}

// Add between factors
for edge in &graph.edges_se3 {
    problem.add_residual_block(
        &[&format!("x{}", edge.from), &format!("x{}", edge.to)],
        Box::new(BetweenFactor::new(edge.measurement.clone())),
        None,
    );
}

// Configure and solve
let config = LevenbergMarquardtConfig::new()
    .with_max_iterations(100)
    .with_cost_tolerance(1e-6);

let mut solver = LevenbergMarquardt::with_config(config);
let result = solver.optimize(&problem, &initial_values)?;

println!("Optimized {} poses in {} iterations", 
    result.parameters.len(), result.iterations);
```

### Example 2: Custom Factor Implementation

Create custom factors by implementing the `Factor` trait:

```rust
use apex_solver::factors::Factor;
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct MyRangeFactor {
    pub measurement: f64,
    pub information: f64,
}

impl Factor for MyRangeFactor {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // Extract 2D point parameters [x, y]
        let x = params[0][0];
        let y = params[0][1];

        // Compute predicted measurement
        let predicted_distance = (x * x + y * y).sqrt();

        // Compute residual: measurement - prediction
        let residual = DVector::from_vec(vec![
            self.information.sqrt() * (self.measurement - predicted_distance)
        ]);

        // Compute analytic Jacobian
        let jacobian = if compute_jacobian {
            if predicted_distance > 1e-8 {
                let scale = -self.information.sqrt() / predicted_distance;
                Some(DMatrix::from_row_slice(1, 2, &[scale * x, scale * y]))
            } else {
                Some(DMatrix::zeros(1, 2))
            }
        } else {
            None
        };

        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        1  // One scalar residual
    }
}

// Use in optimization
problem.add_residual_block(
    &["point"],
    Box::new(MyRangeFactor { measurement: 5.0, information: 1.0 }),
    None
);
```

### Example 3: Self-Calibration Bundle Adjustment

Optimize camera poses, 3D landmarks, AND camera intrinsics simultaneously. See the [apex-camera-models](crates/apex-camera-models) crate for detailed camera model documentation.

```rust
use std::collections::HashMap;
use apex_solver::core::problem::Problem;
use apex_solver::factors::ProjectionFactor;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardt;
// Use any camera model from apex-camera-models crate

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();
    
    // Add camera poses (SE3), 3D landmarks, and per-camera intrinsics
    // See apex-camera-models documentation for camera model options
    
    // Add projection factors with compile-time optimization config
    // ProjectionFactor<CameraModel, OptConfig> links poses + landmarks + intrinsics
    for observation in &observations {
        problem.add_residual_block(
            &[&format!("pose_{}", obs.camera_id),
              &format!("landmark_{}", obs.point_id),
              &format!("intrinsics_{}", obs.camera_id)],
            Box::new(projection_factor),
            Some(Box::new(HuberLoss::new(1.0))),
        );
    }
    
    // Fix first camera for gauge freedom
    for dof in 0..6 {
        problem.fix_variable("pose_0000", dof);
    }
    
    // Configure solver with Schur complement (best for BA)
    let mut solver = LevenbergMarquardt::for_bundle_adjustment();
    let result = solver.optimize(&problem, &initial_values)?;
    
    Ok(())
}
```

**Optimization Types** (compile-time configuration):
- `SelfCalibration`: Optimize pose + landmarks + intrinsics
- `BundleAdjustment`: Optimize pose + landmarks (fixed intrinsics)
- `OnlyPose`: Visual odometry (fixed landmarks and intrinsics)
- `OnlyLandmarks`: Triangulation (known poses)
- `OnlyIntrinsics`: Camera calibration (known structure)

See [apex-camera-models documentation](crates/apex-camera-models/README.md) for complete camera model reference and advanced examples.

---

## 🧮 Technical Implementation

### Manifold Operations

Apex Solver implements mathematically rigorous Lie group operations following the [manif](https://github.com/artivis/manif) library conventions. All manifold types provide `plus()` (retraction), `minus()` (inverse retraction), `compose()` (group composition), `inverse()` (group inverse), and analytic Jacobians for efficient optimization.

**Supported Manifolds**:

| Manifold | Description | DOF | Use Case |
|----------|-------------|-----|----------|
| **SE(3)** | 3D rigid transformations | 6 | 3D SLAM, visual odometry |
| **SO(3)** | 3D rotations | 3 | Orientation tracking |
| **SE(2)** | 2D rigid transformations | 3 | 2D SLAM, mobile robots |
| **SO(2)** | 2D rotations | 1 | 2D orientation |
| **R^n** | Euclidean space | n | Landmarks, camera parameters |

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

## 🎨 Interactive Visualization

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

let result = solver.optimize(&problem, &initial_values)?;
```

**Visualized Metrics**:
- Time series: Cost, gradient norm, damping (λ), step quality (ρ), step norm
- Matrix visualizations: Hessian heat map, gradient vector
- 3D poses: SE3 camera frusta, SE2 2D points

**Run Examples**:
```bash
# Enable visualization feature and run
cargo run --release --bin optimize_3d_graph -- --dataset sphere2500 --with-visualizer
cargo run --release --bin optimize_2d_graph -- --dataset intel --with-visualizer
```

Zero overhead when disabled (feature-gated).

---

## 🧠 Learning Resources

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

## 🙏 Acknowledgments

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

## 📜 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---
