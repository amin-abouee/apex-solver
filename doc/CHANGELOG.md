# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-24

### Added
- **Bundle Adjustment with Camera Intrinsic Optimization** - Full support for simultaneous optimization of camera poses, 3D landmarks, and camera intrinsics
  - **`ProjectionFactor<CameraModel, OptConfig>` generic system** - Type-safe bundle adjustment with compile-time optimization configuration
    - Optimization modes: `SelfCalibration` (pose + landmarks + intrinsics), `BundleAdjustment` (pose + landmarks), `OnlyPose`, `OnlyLandmarks`, `OnlyIntrinsics`, `PoseAndIntrinsics`, `LandmarksAndIntrinsics`
    - Batch projection support for multiple observations per factor
    - Automatic cheirality checking for points behind camera
    - Analytic Jacobians for all camera models (no auto-differentiation overhead)
  - **9 Camera Projection Models** with analytic Jacobians w.r.t. pose, point, and intrinsics:
    - `BALPinholeCameraStrict` - Bundle Adjustment in the Large format (focal, k1, k2)
    - `RadialTangential` - Brown-Conrady distortion model (fx, fy, cx, cy, k1, k2, p1, p2) - OpenCV compatible
    - `Equidistant` - Fisheye lens model (fx, fy, cx, cy, k1, k2, k3, k4)
    - `FOV` - Field-of-view distortion (fx, fy, cx, cy, omega)
    - `UnifiedCamera (UCM)` - Unified camera model for wide FOV (fx, fy, cx, cy, alpha)
    - `ExtendedUnified (EUCM)` - Extended unified model for >180° FOV (fx, fy, cx, cy, alpha, beta)
    - `DoubleSphere` - Double sphere projection for fisheye (fx, fy, cx, cy, xi, alpha)
    - `KannalaBrandt` - Fisheye polynomial model (fx, fy, cx, cy, k1, k2, k3, k4) - GoPro-style cameras
    - `Orthographic` - Orthographic projection (fx, fy, cx, cy)
  - **`CameraModel` trait** with compile-time constant `INTRINSIC_DIM`
    - Associated types: `IntrinsicJacobian`, `PointJacobian` for static-sized matrices
    - Methods: `project()`, `is_valid_point()`, `jacobian_point()`, `jacobian_pose()`, `jacobian_intrinsics()`
    - Batch processing with `project_batch()` for vectorized computation
- **Schur Complement Solvers** - Specialized linear algebra backends for bundle adjustment that exploit problem structure
  - **Explicit Schur Complement** (`ExplicitSchurComplementSolver`) - Direct sparse Cholesky factorization
    - Constructs reduced camera matrix S = B - E·C⁻¹·Eᵀ explicitly in memory
    - Suitable for medium-to-large BA problems (<10,000 cameras)
    - High accuracy with moderate memory usage
    - Supports block reordering and symbolic factorization caching
  - **Implicit Schur Complement** (`IterativeSchurSolver`) - Matrix-free Preconditioned Conjugate Gradients (PCG)
    - Memory-efficient for very large problems (10,000+ cameras)
    - Never constructs S explicitly - computes only matrix-vector products S·x
    - Three preconditioner types: `None`, `BlockDiagonal`, `SchurJacobi` (recommended, matches Ceres behavior)
    - Configurable CG parameters: max_iterations (default: 500), tolerance (default: 1e-9)
    - Linear memory growth with problem size
  - **`StructuredSparseLinearSolver` trait** - Extends `SparseLinearSolver` with variable structure awareness
    - Methods: `initialize_structure()` for BA-specific setup, `solve_normal_equation()`, `solve_augmented_equation()`
    - Enables Schur complement solvers to distinguish camera/landmark blocks
    - Required for exploiting sparsity structure in BA problems
- **BAL (Bundle Adjustment in the Large) File Format Support** (`src/io/bal.rs`)
  - `BalLoader::load()` - Parse BAL format datasets
  - `BalDataset` struct with cameras, points, observations
  - Supports large-scale structure-from-motion datasets (Dubrovnik, Ladybug, Trafalgar, Venice)
  - Git LFS integration for multi-GB dataset storage
- **New Binary: `bundle_adjustment`** - Professional CLI tool for BA optimization
  - Command-line options:
    - `--solver [explicit|implicit]` - Choose Schur complement solver variant
    - `--type [self-calibration|bundle-adjustment|only-pose|only-landmarks|only-intrinsics]` - Optimization configuration
    - `-n NUM_POINTS` - Limit dataset size for testing/profiling
    - `-v/--verbose` - Detailed optimization output
  - Supports all camera models and optimization modes
  - Real-time RMSE reporting and convergence diagnostics
- **New Binary: `pose_graph_g2o`** - Unified pose graph optimization tool
  - Replaces separate `optimize_2d_graph` and `optimize_3d_graph` binaries
  - Single CLI handles both SE2 and SE3 datasets automatically
  - Enhanced command-line interface with dataset selection, optimizer choice, loss functions
  - Support for real-time Rerun visualization with `--with-visualizer` flag
  - Output saving with `--save-output` option
- **Comprehensive Bundle Adjustment Benchmarks** (`benches/bundle_adjustment_benchmark.rs`)
  - Compares apex-solver vs Ceres, GTSAM, g2o on 4 BAL datasets
  - Datasets tested:
    - Dubrovnik: 356 cameras, 226,730 landmarks, 1,255,268 observations
    - Ladybug: 1,723 cameras, 156,502 landmarks, 678,718 observations
    - Trafalgar: 257 cameras, 65,132 landmarks, 225,911 observations
    - Venice: 1,778 cameras, 993,923 landmarks, 5,001,946 observations (largest)
  - Metrics: Initial/final RMSE, wall-clock time, iterations, convergence status
  - Automated CSV export for reproducibility
- **Refactored Odometry Pose Benchmarks** (`benches/odometry_pose_benchmark.rs`)
  - Renamed from `solver_comparison_benchmark` for clarity
  - Cleaner structure with consistent metrics reporting across 2D and 3D datasets
  - Enhanced output formatting with summary statistics
- **C++ Comparison Benchmarks** (`benches/cpp_comparison/`)
  - Reference Ceres, g2o, GTSAM implementations for BA and odometry
  - Enables side-by-side performance validation with identical datasets
  - CMake build system for cross-platform compatibility
- **Test Datasets**
  - BAL datasets via Git LFS: Dubrovnik (356 cams), Ladybug (1,723 cams), Trafalgar (257 cams), Venice (1,778 cams)
  - `city10000.g2o` - Large SE2 pose graph dataset (10,000 poses) for scalability testing

### Changed
- **Binary Consolidation** - Streamlined executable structure for better user experience
  - **Replaced** `optimize_2d_graph` and `optimize_3d_graph` with unified `pose_graph_g2o` binary
  - Single binary automatically detects SE2/SE3 datasets and applies appropriate optimizer
  - Cleaner codebase with reduced duplication
  - **BREAKING**: Users must update scripts to use `pose_graph_g2o` instead of old binary names
- **Linear Algebra Improvements**
  - **Removed** PowerSeries Schur complement solver (inferior performance and accuracy compared to explicit/implicit variants)
  - Enhanced `LevenbergMarquardtConfig::for_bundle_adjustment()` - Preset configuration optimized for BA
    - Pre-configured Schur solver selection, appropriate tolerances, and damping parameters
    - Reduces boilerplate for common BA use cases
  - Improved `SchurBlockStructure` with better variable classification (camera vs landmark blocks)
- **Camera Model System Refactoring**
  - Unified `CameraModel` trait with compile-time intrinsic dimensions
  - Associated types (`IntrinsicJacobian`, `PointJacobian`) for zero-cost abstractions
  - Consistent API across all 9 camera models
  - Batch projection methods for efficient multi-observation processing
- **Code Quality Enhancements**
  - Improved error handling with structured `LinAlgError` types (SchurDecompositionFailed, PCGConvergenceFailed, etc.)
  - Enhanced logging throughout optimization pipeline with tracing instrumentation
  - Removed deprecated bundle adjustment examples (replaced with production-ready `bundle_adjustment` binary)
  - Better separation of concerns in factor graph construction
- **Benchmark Infrastructure**
  - Renamed `solver_comparison_benchmark` → `odometry_pose_benchmark` for semantic clarity
  - Consistent metrics reporting format across all benchmarks (time, iterations, cost, RMSE)
  - Better separation between 2D (SE2) and 3D (SE3) dataset results
  - Enhanced summary statistics with convergence rate tracking

### Performance
- **Bundle Adjustment - Production-Grade Scalability**
  - **100% convergence rate** on all 4 BAL datasets (4/4 successful optimizations)
  - **Superior scalability**: Only solver alongside g2o to complete Venice dataset (5M observations)
    - Ceres and GTSAM timeout after 10 minutes on Venice
    - apex-solver completes in 83 seconds with 0.458 RMSE (vs g2o: 252s with 10.126 RMSE)
  - **Best accuracy on largest dataset**: Achieves 0.458 RMSE on Venice in only 2 iterations
    - 22x better accuracy than g2o (10.126 RMSE in 20 iterations)
  - **Speed advantage over Ceres**:
    - Dubrovnik: 61x faster (47s vs 2,879s), better RMSE (0.533 vs 1.004)
    - Trafalgar: 4.2x faster (10s vs 44s), better RMSE (0.679 vs 1.320)
  - **Competitive with GTSAM**: Similar convergence speed and accuracy on smaller datasets
  - **Memory efficiency**: Implicit Schur solver handles 10,000+ camera problems with linear memory growth
    - Venice (1,778 cameras): ~2GB peak memory vs Ceres/GTSAM >8GB
- **Pose Graph Optimization - Maintained Excellence**
  - Maintained 100% convergence rate on all 8 odometry datasets (4 SE2 + 4 SE3)
  - Consistent performance across problem scales: 434 poses (ring) to 5,000 poses (torus3D)
  - 2-10x faster than Ceres on most datasets while achieving equivalent or better final cost

## [0.1.6] - 2024-11-29

### Added
- **Comprehensive benchmark infrastructure** comparing 6 optimization libraries across 8 standard datasets
  - Rust solvers: apex-solver, factrs, tiny-solver
  - C++ solvers: Ceres Solver, g2o, GTSAM
  - Benchmark results on 4 SE2 datasets (intel, mit, M3500, ring) and 4 SE3 datasets (sphere2500, parking-garage, torus3D, cubicle)
  - Automated benchmark runner with CSV output for reproducibility
  - Performance metrics: execution time, iterations, cost improvement, convergence status
- **Integration test suite** in `tests/integration_tests.rs`
  - End-to-end optimization verification on real G2O datasets
  - Metrics tracked: convergence status, cost improvement, execution time, iteration count
  - Fast tests (ring, intel) and slow tests (sphere2500, parking-garage) with `#[ignore]` annotation

### Changed
- **Improved logging infrastructure**
  - All `println!`/`eprintln!` replaced with `tracing` macros for structured logging
  - Centralized logger configuration in `src/logger.rs` with custom formatter
  - Color-coded log levels and environment variable control (`RUST_LOG`)
  - Consistent logging levels across all modules (info, warn, error, debug, trace)
- **Enhanced code quality**
  - Removed all unwrap() and expect() calls from production code paths
  - Comprehensive error handling with Result types throughout the codebase
  - Cargo.toml lints enforce `unwrap_used = "deny"` and `expect_used = "deny"`
  - Zero panic-inducing calls in hot optimization paths

### Documentation
- Added comprehensive benchmark comparison table to README.md with analysis
  - Convergence reliability comparison across all 6 solvers
  - Performance highlights and cost improvement quality assessment
  - Instructions for reproducing benchmarks
- Moved project status section from README.md to CHANGELOG.md for better organization
- Enhanced Key Features section with v0.1.6 highlights

### Performance
- apex-solver achieves 100% convergence rate (8/8 datasets) - most reliable Rust solver
- Competitive performance: 2-10x faster than Ceres on most datasets
- Excellent cost reduction quality (>99% on well-conditioned problems)

## [0.1.5] - 2024-XX-XX

### Added
- **Camera Projection Factors** - 5 camera models for calibration and bundle adjustment
  - `DoubleSphereProjectionFactor` - Wide FOV fisheye cameras (6 params: fx, fy, cx, cy, α, ξ)
  - `EucmProjectionFactor` - Extended unified camera model (6 params: fx, fy, cx, cy, α, β)
  - `KannalaBrandtProjectionFactor` - Fisheye polynomial model (8 params: fx, fy, cx, cy, k1-k4)
  - `RadTanProjectionFactor` - Brown-Conrady distortion (9 params: fx, fy, cx, cy, k1, k2, p1, p2, k3)
  - `UcmProjectionFactor` - Unified camera model (5 params: fx, fy, cx, cy, α)
- **Factors Module Restructuring** - Dedicated `src/factors/` module with improved organization
  - Separated pose factors (SE2, SE3, Prior) from camera projection factors
  - Better code organization and discoverability
- **Factor Trait Enhancement** - Updated `Factor` trait with `compute_jacobian` parameter for optional Jacobian computation
- **Analytical Jacobians** - All camera factors use hand-derived analytical gradients (no auto-diff overhead)
- **Batch Processing Support** - Efficient vectorized computation for multiple point correspondences
- **Validity Checking** - Automatic detection of invalid projections in all camera models

### Changed
- **Code Quality Improvements** - Streamlined imports, renamed `Loss` trait to `LossFunction`, reduced Debug bounds

## [0.1.4] - 2024-XX-XX

### Added
- **15 Robust Loss Functions** - Comprehensive outlier rejection (Huber, Cauchy, Tukey, Welsch, Barron, and more)
- **Enhanced Termination Criteria** - 8-9 comprehensive convergence checks with relative tolerances
- **Prior Factors** - Anchor poses with known values and incorporate GPS/sensor measurements
- **Fixed Variables** - Hard-constrain specific parameter indices during optimization
- **Relative Tolerances** - Parameter and cost tolerances that scale with problem magnitude
- **New OptimizationStatus Variants** - Better diagnostics with `TrustRegionRadiusTooSmall`, `MinCostThresholdReached`, `IllConditionedJacobian`, `InvalidNumericalValues`
- **New Examples** - `loss_function_comparison.rs` and `compare_constraint_scenarios_3d.rs`

### Changed
- **Updated Defaults** - max_iterations: 50, cost_tolerance: 1e-6, gradient_tolerance: 1e-10

## [0.1.3] - 2024-XX-XX

### Added
- **Persistent symbolic factorization** - 10-15% performance boost via cached symbolic decomposition
- **Covariance for both Cholesky and QR** - Complete uncertainty quantification for all linear solvers
- **G2O file writing** - Export optimized graphs with `G2oWriter::write()`
- **Enhanced error messages** - Structured errors (`OptimizerError`) with numeric context
- **Binary executables** - Professional CLI tools: `optimize_3d_graph` and `optimize_2d_graph`
- **Real-time Rerun visualization** - Live optimization debugging with time series plots, Hessian/gradient heat maps
- **Jacobi preconditioning** - Automatic column scaling for robustness (enabled by default)
- **Improved examples** - `covariance_estimation.rs` and `visualize_optimization.rs`

### Changed
- **Updated dependencies** - Rerun v0.26, improved Glam integration

---

*For detailed usage examples and API documentation, see [README.md](../README.md)*
