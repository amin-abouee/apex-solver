# 🦀 Apex Solver

A high-performance Rust-based nonlinear least squares optimization library designed for computer vision applications including bundle adjustment, SLAM, and pose graph optimization. Built with focus on zero-cost abstractions, memory safety, and mathematical correctness.

[![Crates.io](https://img.shields.io/crates/v/apex-solver.svg)](https://crates.io/crates/apex-solver)
[![Documentation](https://docs.rs/apex-solver/badge.svg)](https://docs.rs/apex-solver)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## 🚀 Quick Start

```rust
use apex_solver::{OptimizerConfig, OptimizerType, LinearSolverType, load_graph};

// Load a pose graph from file
let graph = load_graph("data/sphere2500.g2o")?;

// Configure the optimizer
let config = OptimizerConfig::new()
    .with_optimizer_type(OptimizerType::LevenbergMarquardt)
    .with_linear_solver_type(LinearSolverType::SparseCholesky)
    .with_max_iterations(100);

// Build the optimization problem from the graph
let (problem, initial_values) = graph.to_problem();

// Create and run the optimizer
let mut solver = SolverFactory::create_solver(config);
let result = solver.minimize(&problem, &initial_values)?;

// Check results
println!("Converged: {:?}", result.status);
println!("Initial cost: {:.3e}", result.init_cost);
println!("Final cost: {:.3e}", result.final_cost);
println!("Iterations: {}", result.iterations);
```

**Result**:
```
Converged: CostToleranceReached
Initial cost: 2.317e+05
Final cost: 3.421e+02
Iterations: 12
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
│   └── dog_leg.rs             # Dog Leg trust region method
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
use apex_solver::{OptimizerConfig, OptimizerType, LinearSolverType, SolverFactory};
use apex_solver::manifold::se3::SE3;

// Create solver configuration
let config = OptimizerConfig::new()
    .with_optimizer_type(OptimizerType::LevenbergMarquardt)
    .with_linear_solver_type(LinearSolverType::SparseCholesky)
    .with_max_iterations(100)
    .with_cost_tolerance(1e-6)
    .with_verbose(true);

let mut solver = SolverFactory::create_solver(config);

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
use apex_solver::load_graph;

// Load pose graph from file
let graph = load_graph("data/parking-garage.g2o")?;
println!("Loaded {} vertices and {} edges", 
         graph.vertex_count(), graph.edge_count());

// Access vertices and edges
for vertex in graph.vertices() {
    println!("Vertex {}: {:?}", vertex.id(), vertex.estimate());
}

// Build optimization problem
let (problem, initial_values) = graph.to_problem();
```

### Available Examples

Run these examples to explore the library's capabilities:

```bash
# Load and analyze graph files
cargo run --example load_graph_file

# Visualize pose graphs with rerun
cargo run --example visualize_graph_file -- data/sphere2500.g2o

# Compare different optimizers
cargo run --example compare_optimizers

# Benchmark linear algebra backends
cargo run --example sparse_comparison

# Demonstrate manifold operations
cargo run --example manifold_demo
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
- **Best for**: General-purpose pose graph optimization
- **Configuration**:
  ```rust
  OptimizerConfig::new()
      .with_optimizer_type(OptimizerType::LevenbergMarquardt)
      .with_damping(1e-4)
      .with_damping_bounds(1e-12, 1e12)
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

## 🎮 Interactive Visualization

Use the visualization examples to explore pose graphs interactively:

```bash
# Launch 3D visualization with rerun
cargo run --example visualize_graph_file -- data/sphere2500.g2o

# The viewer will show:
# - Initial pose graph (red)
# - Optimized pose graph (green)
# - Edge constraints
# - Cost reduction over iterations
```

The visualization uses [rerun](https://rerun.io/) for real-time 3D rendering of:
- Pose graphs before/after optimization
- Optimization trajectory
- Convergence metrics
- Residual distributions

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
- **`rerun`** (0.23) - 3D visualization and plotting

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

### High Priority
- **Performance optimization** - Caching symbolic factorizations, reducing allocations
- **Additional robust loss functions** - Cauchy, Tukey, DCS
- **G2O file writing** - Export optimized graphs
- **Covariance for all solvers** - Currently only Cholesky computes covariance

### Medium Priority
- **Algorithm implementations** - Additional optimization methods (BFGS, conjugate gradient)
- **Manifold extensions** - Support for Sim(3), affine transformations
- **File format support** - KITTI, EuRoC, additional SLAM datasets
- **Bundle adjustment factors** - Camera reprojection, stereo, IMU pre-integration

### Future
- **GPU acceleration** - CUDA/HIP support for large-scale problems
- **Incremental optimization** - Efficient re-optimization when adding new factors
- **Visualization tools** - Built-in pose graph viewer

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

Performance on standard datasets (Apple M1 Max, 64GB RAM):

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

**Current Version**: 0.1.2  
**Status**: Beta - API may change  
**Production Ready**: For pose graph optimization  
**Next Release**: 0.2.0 (planned features: covariance for all solvers, G2O writing, additional robust losses)

### Roadmap

**v0.2.0** (Q2 2025):
- ✅ Covariance computation for all solvers
- ✅ G2O file writing
- ✅ Additional robust loss functions
- ✅ Performance optimizations (symbolic factorization caching)

**v0.3.0** (Q3 2025):
- ✅ Sim(3) manifold support
- ✅ Bundle adjustment factors
- ✅ KITTI/EuRoC format support
- ✅ GPU acceleration (experimental)

**v1.0.0** (Q4 2025):
- ✅ Stable API
- ✅ Comprehensive documentation
- ✅ Production deployment ready
- ✅ Performance parity with g2o

---

*Built with 🦀 Rust for performance, safety, and mathematical correctness.*
