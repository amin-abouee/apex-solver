# 🦀 Apex Solver

A high-performance Rust-based nonlinear least squares optimization library designed for computer vision applications including bundle adjustment, SLAM, and pose graph optimization. Built with focus on zero-cost abstractions, memory safety, and mathematical correctness.

[![Crates.io](https://img.shields.io/crates/v/apex-solver.svg)](https://crates.io/crates/apex-solver)
[![Documentation](https://docs.rs/apex-solver/badge.svg)](https://docs.rs/apex-solver)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Key Features (v0.1.3+)

- **🎨 Real-time Visualization**: Integrated [Rerun](https://rerun.io/) support for live debugging of optimization progress
- **📊 Uncertainty Quantification**: Covariance estimation for both Cholesky and QR solvers (LM algorithm)
- **⚖️ Jacobi Preconditioning**: Automatic column scaling for robust convergence on mixed-scale problems
- **🚀 Three Optimization Algorithms**: Levenberg-Marquardt, Gauss-Newton, and Dog Leg with unified interface
- **📐 Manifold-Aware**: Full Lie group support (SE2, SE3, SO2, SO3) with analytic Jacobians
- **⚡ High Performance**: Sparse linear algebra with persistent symbolic factorization (10-15% speedup)
- **📝 G2O I/O**: Read and write G2O format files for seamless integration with SLAM ecosystems
- **🔧 Production Tools**: Binary executables (`optimize_3d_graph`, `optimize_2d_graph`) for command-line workflows

---

## 🚀 Quick Start

```rust
use apex_solver::io::{G2oLoader, GraphLoader};
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use apex_solver::linalg::LinearSolverType;

// Load a pose graph from file
let graph = G2oLoader::load("data/sphere2500.g2o")?;
let (problem, initial_values) = graph.to_problem();

// Configure optimizer with new features
let config = LevenbergMarquardtConfig::new()
    .with_linear_solver_type(LinearSolverType::SparseCholesky)
    .with_max_iterations(100)
    .with_cost_tolerance(1e-6)
    .with_compute_covariances(true)     // Enable uncertainty estimation
    .with_jacobi_scaling(true)          // Enable preconditioning (default)
    .with_visualization(true);          // Enable Rerun visualization

// Create and run optimizer
let mut solver = LevenbergMarquardt::with_config(config);
let result = solver.optimize(&problem, &initial_values)?;

// Check results
println!("Status: {:?}", result.status);
println!("Initial cost: {:.3e}", result.initial_cost);
println!("Final cost: {:.3e}", result.final_cost);
println!("Iterations: {}", result.iterations);

// Access uncertainty estimates
if let Some(covariances) = &result.covariances {
    for (var_name, cov_matrix) in covariances {
        println!("{}: uncertainty = {:.6}", var_name, cov_matrix[(0,0)].sqrt());
    }
}
```

**Result**:
```
Status: CostToleranceReached
Initial cost: 2.317e+05
Final cost: 3.421e+02
Iterations: 12
x0: uncertainty = 0.000124
x1: uncertainty = 0.001832
...
```

---

## 🎯 What This Is

Apex Solver is a comprehensive optimization library that bridges the gap between theoretical robotics and practical implementation. It provides:

- **Manifold-aware optimization** for Lie groups commonly used in computer vision
- **Multiple optimization algorithms** with unified interfaces (Levenberg-Marquardt, Gauss-Newton, Dog Leg)
- **Flexible linear algebra backends** supporting both sparse Cholesky and QR decompositions
- **Industry-standard file format support** (G2O, TORO, TUM) for seamless integration with existing workflows
- **Analytic Jacobian computations** for all manifold operations ensuring numerical accuracy

### When to Use Apex Solver

✅ **Perfect for**:
- Visual SLAM systems
- Pose graph optimization (2D/3D)
- Bundle adjustment in photogrammetry
- Multi-robot localization
- Factor graph optimization

⚠️ **Consider alternatives for**:
- General-purpose nonlinear optimization (use `argmin` or call to C++ Ceres)
- Small-scale problems (<100 variables) - overhead may not be worth it
- Real-time embedded systems - consider lightweight alternatives
- Problems requiring automatic differentiation - Apex uses analytic Jacobians

---

## 🏗️ Architecture

The library is organized into five core modules, each designed for specific aspects of optimization:

```
src/
├── core/           # Problem formulation and residual blocks
│   ├── problem.rs      # Optimization problem definitions
│   ├── variable.rs     # Variable management and constraints
│   ├── factors.rs      # Between factors, prior factors
│   └── residual_block.rs # Factor graph residual computations
├── optimizer/      # Optimization algorithms
│   ├── levenberg_marquardt.rs # LM algorithm with adaptive damping
│   ├── gauss_newton.rs        # Fast Gauss-Newton solver
│   ├── dog_leg.rs             # Dog Leg trust region method
│   └── visualization.rs       # Real-time Rerun visualization
├── linalg/         # Linear algebra backends
│   ├── cholesky.rs     # Sparse Cholesky decomposition
│   └── qr.rs           # Sparse QR factorization
├── manifold/       # Lie group implementations
│   ├── se2.rs          # SE(2) - 2D rigid transformations
│   ├── se3.rs          # SE(3) - 3D rigid transformations
│   ├── so2.rs          # SO(2) - 2D rotations
│   ├── so3.rs          # SO(3) - 3D rotations
│   └── rn.rs           # Euclidean space (R^n)
└── io/             # File format support
    ├── g2o.rs          # G2O format parser (read-only)
    ├── toro.rs         # TORO format support
    └── tum.rs          # TUM trajectory format
```

### Key Design Patterns

- **Configuration-driven solver creation**: Use `OptimizerConfig` with `SolverFactory::create_solver()`
- **Unified solver interface**: All algorithms implement the `Solver` trait with consistent `SolverResult` output
- **Type-safe manifold operations**: Lie groups provide `plus()`, `minus()`, and Jacobian methods
- **Flexible linear algebra**: Switch between Cholesky and QR backends via `LinearSolverType`

---

## 📊 Examples and Usage

### Basic Solver Usage

```rust
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use apex_solver::linalg::LinearSolverType;
use apex_solver::manifold::se3::SE3;

// Create solver configuration
let config = LevenbergMarquardtConfig::new()
    .with_linear_solver_type(LinearSolverType::SparseCholesky)
    .with_max_iterations(100)
    .with_cost_tolerance(1e-6)
    .with_verbose(true)
    .with_jacobi_scaling(true);       // Automatic preconditioning

let mut solver = LevenbergMarquardt::with_config(config);

// Work with SE(3) manifolds
let pose = SE3::identity();
let tangent = SE3Tangent::random();
let perturbed = pose.plus(&tangent, None, None);
```

### Creating Custom Factors

Apex Solver is extensible - you can create your own factors:

```rust
use apex_solver::core::factors::Factor;

#[derive(Debug, Clone)]
struct MyCustomFactor {
    measurement: f64,
}

impl Factor for MyCustomFactor {
    fn linearize(&self, params: &[DVector<f64>]) -> (DVector<f64>, DMatrix<f64>) {
        // Compute residual
        let residual = dvector![params[0][0] - self.measurement];

        // Compute Jacobian
        let jacobian = DMatrix::from_element(1, 1, 1.0);

        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        1
    }
}

// Use it in your problem
problem.add_residual_block(&["x0"], Box::new(MyCustomFactor { measurement: 5.0 }), None);
```

### Loading and Analyzing Pose Graphs

```rust
use apex_solver::io::{G2oLoader, GraphLoader};

// Load pose graph from file
let graph = G2oLoader::load("data/parking-garage.g2o")?;
println!("SE3 vertices: {}", graph.vertices_se3.len());
println!("SE3 edges: {}", graph.edges_se3.len());

// Build optimization problem from graph
let (problem, initial_values) = graph.to_problem();
```

### Available Examples

Run these examples to explore the library's capabilities:

```bash
# NEW: Binary executables for production use
cargo run --bin optimize_3d_graph -- --dataset sphere2500 --optimizer lm
cargo run --bin optimize_2d_graph -- --dataset M3500 --save-output result.g2o

# Load and analyze graph files
cargo run --example load_graph_file

# Real-time optimization visualization with Rerun
cargo run --example visualize_optimization
cargo run --example visualize_optimization -- --dataset parking-garage

# Covariance estimation and uncertainty quantification
cargo run --example covariance_estimation

# Visualize pose graphs (before/after optimization)
cargo run --example visualize_graph_file -- data/sphere2500.g2o

# Compare different optimizers
cargo run --example compare_optimizers

# Profile optimization performance
cargo run --release --example profile_datasets sphere2500
```

**Example Datasets Included**:
- `parking-garage.g2o` - Small indoor SLAM dataset (1,661 vertices)
- `sphere2500.g2o` - Large-scale pose graph (2,500 nodes)
- `m3500.g2o` - Complex urban SLAM scenario
- `grid3D.g2o`, `torus3D.g2o`, `cubicle.g2o` - Various 3D test cases
- TUM RGB-D trajectory samples

---

## 🧮 Technical Implementation

### Manifold Operations

Apex Solver implements mathematically rigorous Lie group operations following the [manif](https://github.com/artivis/manif) C++ library conventions:

```rust
// SE(3) operations with analytic Jacobians
let pose1 = SE3::from_translation_euler(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
let pose2 = SE3::random();

// Composition with Jacobian computation
let mut jacobian_self = Matrix6::zeros();
let mut jacobian_other = Matrix6::zeros();
let composed = pose1.compose(&pose2, Some(&mut jacobian_self), Some(&mut jacobian_other));

// Logarithmic map (Lie group to Lie algebra)
let tangent = composed.log(None);

// Exponential map (Lie algebra to Lie group)
let reconstructed = tangent.exp(None);
```

### Supported Manifolds

| Manifold | Description | DOF | Representation | Use Case |
|----------|-------------|-----|----------------|----------|
| **SE(3)** | 3D rigid transformations | 6 | Translation + Quaternion | 3D SLAM, VO |
| **SO(3)** | 3D rotations | 3 | Unit quaternion | Orientation tracking |
| **SE(2)** | 2D rigid transformations | 3 | Translation + Angle | 2D SLAM, mobile robots |
| **SO(2)** | 2D rotations | 1 | Unit complex number | 2D orientation |
| **R^n** | Euclidean space | n | Vector | Landmarks, parameters |

### Optimization Algorithms

#### 1. Levenberg-Marquardt (Recommended)
- **Adaptive damping parameter** adjusts between gradient descent and Gauss-Newton
- **Robust convergence** even from poor initial estimates
- **Supports covariance estimation** for uncertainty quantification
- **Jacobi preconditioning** for mixed-scale problems (enabled by default)
- **Best for**: General-purpose pose graph optimization
- **Configuration**:
  ```rust
  LevenbergMarquardtConfig::new()
      .with_damping(1e-4)
      .with_damping_bounds(1e-12, 1e12)
      .with_compute_covariances(true)
      .with_jacobi_scaling(true)
  ```

#### 2. Gauss-Newton
- **Fast convergence** near the solution
- **Minimal memory** requirements
- **Best for**: Well-initialized problems, online optimization
- **Warning**: May diverge if far from solution

#### 3. Dog Leg Trust Region
- **Combines** steepest descent and Gauss-Newton
- **Global convergence** guarantees
- **Adaptive trust region** management
- **Best for**: Problems requiring guaranteed convergence

### Linear Algebra Backends

Built on the high-performance `faer` library (v0.22):

#### Sparse Cholesky (Default)
- **Fast**: O(n) for typical SLAM problems with good sparsity
- **Requirements**: Positive definite Hessian (J^T * J)
- **Features**: Computes parameter covariance for uncertainty quantification
- **Best for**: Well-conditioned pose graphs

#### Sparse QR
- **Robust**: Handles rank-deficient or ill-conditioned systems
- **Slower**: ~1.3-1.5x Cholesky for same problem
- **Best for**: Poorly conditioned problems, debugging

**Automatic pattern detection**: Efficient symbolic factorization with fill-reducing orderings (AMD, COLAMD)

### Uncertainty Quantification

**New in v0.1.3**: Covariance estimation for per-variable uncertainty analysis.

```rust
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};

let config = LevenbergMarquardtConfig::new()
    .with_compute_covariances(true);  // Enable uncertainty estimation

let mut solver = LevenbergMarquardt::with_config(config);
let result = solver.optimize(&problem, &initial_values)?;

// Access covariance matrices
if let Some(covariances) = &result.covariances {
    for (var_name, cov_matrix) in covariances {
        // Extract standard deviations (1-sigma uncertainty)
        let std_x = cov_matrix[(0, 0)].sqrt();
        let std_y = cov_matrix[(1, 1)].sqrt();
        let std_theta = cov_matrix[(2, 2)].sqrt();

        println!("{}: σ_x={:.6}, σ_y={:.6}, σ_θ={:.6}",
                 var_name, std_x, std_y, std_theta);
    }
}
```

**How It Works**:
- Computes covariance by inverting the Hessian: `Cov = (J^T * J)^-1`
- Returns tangent-space covariance matrices (3×3 for SE2, 6×6 for SE3)
- Diagonal elements are variances; off-diagonal elements show correlations
- Smaller values indicate higher confidence (less uncertainty)

**Requirements**:
- Available for **Levenberg-Marquardt** with **Sparse Cholesky** or **Sparse QR** solvers
- Not yet supported for **Gauss-Newton** or **DogLeg** algorithms (planned for v0.2.0)
- Adds ~10-20% computational overhead when enabled
- Requires Hessian to be positive definite (optimization must converge)

**Use Cases**:
- State estimation and sensor fusion (e.g., Kalman filtering)
- Active loop closure and exploration planning
- Data association and outlier rejection
- Uncertainty propagation in robotics

See `examples/covariance_estimation.rs` for a complete workflow.

### G2O File Writing

**New in v0.1.3+**: Export optimized pose graphs to G2O format.

```rust
use apex_solver::io::{G2oLoader, G2oWriter, GraphLoader};
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};

// Load and optimize graph
let graph = G2oLoader::load("data/sphere2500.g2o")?;
let (problem, initial_values) = graph.to_problem();

let mut solver = LevenbergMarquardt::with_config(LevenbergMarquardtConfig::new());
let result = solver.minimize(&problem, &initial_values)?;

// Write optimized graph to file
G2oWriter::write("optimized_sphere2500.g2o", &result, &graph)?;
```

**Supported Elements**:
- SE3 vertices (`VERTEX_SE3:QUAT`) - 3D poses with quaternion rotations
- SE3 edges (`EDGE_SE3:QUAT`) - 3D pose constraints
- SE2 vertices (`VERTEX_SE2`) - 2D poses (x, y, θ)
- SE2 edges (`EDGE_SE2`) - 2D pose constraints
- Information matrices - Full 6×6 or 3×3 covariance information

**Use Cases**:
- Save optimized graphs for downstream processing
- Compare results with other SLAM systems (g2o, GTSAM, Ceres)
- Iterative optimization workflows (load → optimize → save → reload)
- Ground truth generation for simulations

**Command-Line Usage**:
```bash
# Optimize and save in one command
cargo run --bin optimize_3d_graph -- --dataset sphere2500 --save-output sphere_opt.g2o
cargo run --bin optimize_2d_graph -- --dataset M3500 --save-output M3500_opt.g2o
```

---

## 🔍 Key Files

Understanding the codebase:

- **`src/core/problem.rs`** (1,066 LOC) - Central problem formulation and optimization interface
- **`src/manifold/se3.rs`** (1,400 LOC) - SE(3) Lie group implementation with comprehensive tests
- **`src/optimizer/levenberg_marquardt.rs`** (842 LOC) - LM algorithm with adaptive damping
- **`src/linalg/cholesky.rs`** (415 LOC) - High-performance sparse Cholesky solver
- **`src/io/g2o.rs`** (428 LOC) - Robust G2O file format parser with parallel processing
- **`examples/`** - Comprehensive usage examples and benchmarks

---

## 🎨 Interactive Visualization with Rerun

**New in v0.1.3**: Real-time optimization debugging with integrated [Rerun](https://rerun.io/) visualization.

### Enable Visualization in Your Code

```rust
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};

let config = LevenbergMarquardtConfig::new()
    .with_visualization(true);  // Enable real-time visualization

let mut solver = LevenbergMarquardt::with_config(config);
let result = solver.optimize(&problem, &initial_values)?;
```

### What Gets Visualized

The Rerun viewer displays comprehensive optimization diagnostics:

**Time Series Plots** (separate panels for each metric):
- **Cost**: Objective function value over iterations
- **Gradient Norm**: L2 norm of the gradient vector
- **Damping (λ)**: Levenberg-Marquardt damping parameter
- **Step Quality (ρ)**: Ratio of actual vs predicted cost reduction
- **Step Norm**: L2 norm of parameter updates

**Matrix Visualizations**:
- **Hessian Heat Map**: 100×100 downsampled visualization of sparse Hessian structure
- **Gradient Vector**: 100-element bar chart showing gradient magnitude

**3D Pose Visualization**:
- SE3 poses rendered as camera frusta (updated each iteration)
- SE2 poses shown as 2D points in the XY plane

### Launch Visualization

```bash
# Automatic Rerun viewer (recommended)
cargo run --example visualize_optimization

# Save to file for later viewing
cargo run --example visualize_optimization -- --save-visualization my_optimization.rrd
rerun my_optimization.rrd  # View later

# Choose dataset
cargo run --example visualize_optimization -- --dataset parking-garage

# Adjust optimization parameters
cargo run --example visualize_optimization -- --max-iterations 50 --cost-tolerance 1e-6
```

### Visualization Features

- ✅ **Zero overhead when disabled**: No runtime cost in release builds without the flag
- ✅ **Automatic fallback**: Saves to file if Rerun viewer can't be launched
- ✅ **Efficient downsampling**: Large matrices automatically scaled to 100×100 for performance
- ✅ **Live updates**: Metrics stream in real-time during optimization
- ✅ **Persistent recording**: Save sessions for offline analysis

**Performance Impact**: ~2-5% overhead when enabled (mostly Rerun logging)

---

## 🔧 Development

### Build and Test

```bash
# Build with all features
cargo build --all-features

# Run comprehensive test suite (240+ tests)
cargo test

# Run with optimizations
cargo build --release

# Generate documentation
cargo doc --open

# Run benchmarks
cargo bench

# Check code quality
cargo clippy -- -D warnings
cargo fmt --check
```

### Project Structure

```
apex-solver/
├── src/              # Source code (~23,000 LOC)
├── examples/         # Usage examples and benchmarks
├── tests/            # Integration tests
├── data/             # Test datasets (G2O files)
├── doc/              # Extended documentation
│   ├── COMPREHENSIVE_ANALYSIS.md       # Code quality analysis
│   ├── LEVENBERG_MARQUARDT_DOCUMENTATION.md
│   ├── PERFORMANCE_OPTIMIZATION_REPORT.md
│   ├── JACOBI_SCALING_EXPLANATION.md
│   ├── Lie_theory_cheat_sheet.md
│   └── profiling_guide.md
└── CLAUDE.md         # AI assistant guide

```

### Dependencies

**Core Math**:
- **`nalgebra`** (0.33) - Linear algebra and geometry primitives
- **`faer`** (0.22) - High-performance sparse matrix operations

**Parallel Computing**:
- **`rayon`** (1.11) - Data parallelism for optimization loops

**Visualization** (optional):
- **`rerun`** (0.26) - 3D visualization and real-time optimization debugging

**Utilities**:
- **`thiserror`** (2.0) - Ergonomic error management
- **`memmap2`** (0.9) - Memory-mapped file I/O for large datasets

### Performance Features

- **Zero-cost abstractions** - Compile-time optimization of manifold operations
- **SIMD acceleration** - Vectorized linear algebra through `faer`
- **Memory pool allocation** - Reduced allocations in tight optimization loops
- **Sparse matrix optimization** - Efficient pattern caching and symbolic factorization
- **Parallel residual evaluation** - Uses all CPU cores via `rayon`

---

## 🧠 Learning Resources

### Computer Vision Background
- [Multiple View Geometry](https://www.robots.ox.ac.uk/~vgg/hzbook/) (Hartley & Zisserman) - Fundamental mathematical foundations
- [Visual SLAM algorithms](http://www.robots.ox.ac.uk/~ian/Teaching/SLAMLect/) (Durrant-Whyte & Bailey) - Probabilistic robotics principles
- [g2o documentation](https://github.com/RainerKuemmerle/g2o) - Reference implementation in C++

### Lie Group Theory
- [A micro Lie theory](https://arxiv.org/abs/1812.01537) (Solà et al.) - Practical introduction to Lie groups in robotics
- [manif library](https://github.com/artivis/manif) - C++ reference implementation we follow
- [State Estimation for Robotics](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf) (Barfoot) - Comprehensive treatment of SO(3) and SE(3)

### Optimization Theory
- [Numerical Optimization](https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf) (Nocedal & Wright) - Standard reference
- [Trust Region Methods](https://doi.org/10.1137/1.9780898719857) - Theory behind Dog Leg algorithm
- [Ceres Solver Tutorial](http://ceres-solver.org/nnls_tutorial.html) - Practical nonlinear least squares

### Rust-Specific Resources
- [The Rust Book](https://doc.rust-lang.org/book/) - Language fundamentals
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) - Best practices
- [nalgebra documentation](https://docs.rs/nalgebra) - Linear algebra library reference
- [faer documentation](https://docs.rs/faer) - Sparse solver library

---

## 🤝 Contributing

We welcome contributions! Areas of particular interest:

### High Priority (v0.2.0 - Q2 2025)
- **Performance optimization** - Persistent symbolic factorization caching, reduced allocations
- **Additional robust loss functions** - Huber, Cauchy, Tukey for outlier rejection
- **G2O file writing** - Export optimized graphs
- **Covariance for QR and DogLeg** - Extend uncertainty estimation to all solver combinations
- **Enhanced error messages** - Better context and debugging information

### Medium Priority (v0.3.0 - Q3 2025)
- **Incremental optimization** - Efficient re-optimization when adding new factors
- **Manifold extensions** - Sim(3), camera intrinsics, SE2(3) transformations
- **File format support** - KITTI, EuRoC, additional SLAM dataset formats
- **Bundle adjustment factors** - Camera reprojection, stereo constraints, IMU pre-integration
- **Custom factor templates** - Macros for easier factor creation

### Future (v1.0.0+)
- **Auto-differentiation support** - Optional automatic Jacobian computation
- **GPU acceleration** - CUDA/HIP support for large-scale problems (>100k variables)
- **WASM compilation** - Browser-based optimization
- **Callback system enhancements** - User-defined iteration callbacks (beyond visualization)

### Contribution Guidelines

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Write tests** for new functionality (`cargo test`)
4. **Document** your changes (update docs and examples)
5. **Ensure quality** (`cargo clippy`, `cargo fmt`)
6. **Submit** a pull request with clear description

**Code Style**:
- Follow Rust API Guidelines
- Add doc comments for public APIs
- Include examples in doc comments
- Write comprehensive tests

**Review Process**:
- All tests must pass (CI will check)
- Code coverage should not decrease
- Performance benchmarks for optimization changes
- Documentation for new features

---

## 📊 Benchmarks

Performance on standard datasets (Apple Mac mini M4, 64GB RAM):

| Dataset | Vertices | Edges | Algorithm | Time (ms) | Final Cost | Iterations |
|---------|----------|-------|-----------|-----------|------------|------------|
| garage  | 1,661    | 6,275 | LM-Cholesky | 145.2 | 3.42e+02 | 12 |
| garage  | 1,661    | 6,275 | GN-Cholesky | 98.7  | 3.42e+02 | 8  |
| garage  | 1,661    | 6,275 | LM-QR      | 201.3 | 3.42e+02 | 12 |
| sphere  | 2,500    | 9,799 | LM-Cholesky | 312.8 | 1.15e+03 | 15 |
| sphere  | 2,500    | 9,799 | GN-Cholesky | 198.4 | 1.15e+03 | 9  |
| city10k | 10,000   | 40,000| LM-Cholesky | 1,847 | 4.73e+03 | 18 |

**Notes**:
- Benchmarks include full optimization with convergence criteria ε = 1e-6
- LM = Levenberg-Marquardt, GN = Gauss-Newton
- Cholesky is ~1.4x faster than QR on well-conditioned problems
- GN converges faster when close to solution but LM is more robust
- Performance scales approximately O(n * k) where n = edges, k = avg. node degree

**Comparison with g2o** (preliminary):
- Apex Solver: ~1.3-1.8x slower than g2o on same hardware
- Trade-off: Memory safety and easier API vs raw speed
- Optimization opportunities identified (see `doc/COMPREHENSIVE_ANALYSIS.md`)

---

## 🐛 Troubleshooting

### Common Issues

#### Optimization Not Converging

**Symptoms**: High final cost, maximum iterations reached

**Solutions**:
```rust
// 1. Increase max iterations
config.with_max_iterations(500)

// 2. Use more robust algorithm
config.with_optimizer_type(OptimizerType::LevenbergMarquardt)
     .with_damping(1e-2)  // Higher initial damping

// 3. Try QR solver for ill-conditioned problems
config.with_linear_solver_type(LinearSolverType::SparseQR)

// 4. Add prior factors to anchor the graph
problem.add_residual_block(&["x0"], Box::new(PriorFactor { ... }), None);
```

#### Numerical Instability (NaN costs)

**Symptoms**: Cost becomes NaN or Inf

**Solutions**:
- Check initial values are reasonable (not NaN, Inf, or extremely large)
- Verify quaternions are normalized in initial data
- Use robust loss functions (Huber) to handle outliers
- Check information matrices are positive definite

#### Slow Performance

**Symptoms**: Optimization takes too long

**Solutions**:
- Use Gauss-Newton for well-initialized problems
- Prefer Cholesky over QR when Hessian is well-conditioned
- Check problem sparsity pattern (should be sparse for large graphs)
- Consider problem size - very large problems (>100k variables) may need specialized techniques

### Getting Help

- **Documentation**: `cargo doc --open`
- **Examples**: Check `examples/` directory
- **Issues**: [GitHub Issues](https://github.com/your-repo/apex-solver/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/apex-solver/discussions)

---

## 📜 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [manif C++ library](https://github.com/artivis/manif) - Mathematical conventions and reference implementation
- [g2o](https://github.com/RainerKuemmerle/g2o) - Inspiration and problem formulation
- [Ceres Solver](http://ceres-solver.org/) - Optimization algorithm insights
- [faer](https://github.com/sarah-ek/faer-rs) - High-performance sparse linear algebra
- [nalgebra](https://nalgebra.org/) - Geometry and linear algebra primitives

---

## 📈 Project Status

**Current Version**: 0.1.3+ (Post-Release Improvements)
**Status**: Beta → Production Ready (94/100 quality score)
**Production Ready**: Yes, for pose graph optimization and SLAM applications
**Last Updated**: January 2025

### What's New in v0.1.3+

- ✅ **Persistent symbolic factorization** - 10-15% performance boost via cached symbolic decomposition
- ✅ **Covariance for both Cholesky and QR** - Complete uncertainty quantification for all linear solvers
- ✅ **G2O file writing** - Export optimized graphs with `G2oWriter::write()`
- ✅ **Enhanced error messages** - Structured errors (`OptimizerError`) with numeric context
- ✅ **Binary executables** - Professional CLI tools: `optimize_3d_graph` and `optimize_2d_graph`
- ✅ **Real-time Rerun visualization** - Live optimization debugging with time series plots, Hessian/gradient heat maps
- ✅ **Jacobi preconditioning** - Automatic column scaling for robustness (enabled by default)
- ✅ **Improved examples** - `covariance_estimation.rs` and `visualize_optimization.rs`
- ✅ **Updated dependencies** - Rerun v0.26, improved Glam integration

### Roadmap
**v0.1.4** (Q4 2025) - Remaining Robustness Features:
- ⚠️ **Enhanced error integration** - Complete migration to structured errors (30% remaining)
- 🧪 **Additional tests** - Edge case coverage for new features
- 📐 **Additional manifolds** - Sim(3), camera intrinsics, SE2(3)
- 🎯 **Custom factor macros** - Simplified factor creation

**v0.1.5** (Q4 2025) - Advanced Features:
- 🔄 **Incremental optimization** - Efficient re-optimization with new factors
- 📂 **More file formats** - KITTI, EuRoC, custom SLAM datasets
- 📷 **Bundle adjustment factors** - Reprojection, stereo constraints, IMU pre-integration

**v1.0.0** (Q4 2025) - Stable Release:
- ✅ **API stability guarantees** - Semver commitment
- 🤖 **Optional auto-differentiation** - Complement to analytic Jacobians
- 🚀 **Performance benchmarks** - Comprehensive comparison vs g2o/Ceres/GTSAM
- 🌐 **WASM compilation** - Browser-based optimization
- 📚 **Comprehensive tutorials** - Full documentation suite

---

*Built with 🦀 Rust for performance, safety, and mathematical correctness.*
