# Architecture and Structure Analysis

## Directory Structure

```
/Volumes/External/Workspace/rust/apex-solver/src/
├── lib.rs                           # Main library entry point
├── logger.rs                        # Logging utilities
├── error.rs                         # Top-level error types
│
├── core/                            # Core optimization components
│   ├── mod.rs                       # Module exports, CoreError
│   ├── problem.rs                   # Problem formulation (~1,667 LOC)
│   ├── residual_block.rs            # Residual block definitions
│   ├── variable.rs                  # Variable abstractions
│   ├── corrector.rs                 # Loss function application
│   └── loss_functions.rs            # 16 loss function implementations (~2,004 LOC)
│
├── manifold/                        # Lie group implementations
│   ├── mod.rs                       # LieGroup, Tangent traits
│   ├── rn.rs                        # Euclidean space Rn (~1,016 LOC)
│   ├── se2.rs                       # 2D rigid transformations (~1,437 LOC)
│   ├── se3.rs                       # 3D rigid transformations (~1,431 LOC)
│   ├── so2.rs                       # 2D rotations (~600 LOC)
│   └── so3.rs                       # 3D rotations (~1,305 LOC)
│
├── optimizer/                       # Optimization algorithms
│   ├── mod.rs                       # Solver trait, types (~320 LOC)
│   ├── levenberg_marquardt.rs       # LM algorithm (~1,625 LOC)
│   ├── gauss_newton.rs              # Gauss-Newton algorithm (~1,367 LOC)
│   └── dog_leg.rs                   # Dog Leg/Trust region (~2,032 LOC)
│
├── linalg/                          # Linear algebra backends
│   ├── mod.rs                       # SparseLinearSolver trait
│   ├── cholesky.rs                  # Sparse Cholesky (~649 LOC)
│   └── qr.rs                        # Sparse QR (~646 LOC)
│
├── factors/                         # Factor/constraint implementations
│   ├── mod.rs                       # Factor trait
│   ├── prior_factor.rs              # Unary priors
│   ├── se2_factor.rs                # SE2 between factors
│   ├── se3_factor.rs                # SE3 between factors
│   ├── double_sphere_factor.rs      # Double Sphere camera model
│   ├── eucm_factor.rs               # EUCM camera model
│   ├── fov_factor.rs                # FOV camera model
│   ├── kannala_brandt_factor.rs     # Kannala-Brandt camera model
│   ├── rad_tan_factor.rs            # Radial-Tangential camera model
│   └── ucm_factor.rs                # Unified Camera Model
│
├── io/                              # File I/O and graph formats
│   ├── mod.rs                       # Graph data structures, GraphLoader trait
│   ├── g2o.rs                       # G2O format loader/writer
│   └── toro.rs                      # TORO format loader
│
└── observers/                       # Optimization monitoring
    ├── mod.rs                       # OptObserver trait
    ├── visualization.rs             # Rerun visualization (feature-gated)
    └── conversions.rs               # Manifold to Rerun conversions
```

## Module Dependency Graph

```
                    lib.rs (public API)
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    manifold/         core/          optimizer/
        │                │                │
        │         ┌──────┴──────┐         │
        │         │             │         │
        │         ▼             │         ▼
        │     factors/          │     linalg/
        │         │             │         │
        └─────────┴─────────────┴─────────┘
                         │
                    ┌────┴────┐
                    │         │
                    ▼         ▼
                  io/     observers/
```

### Dependency Details

| Module | Dependencies | Dependents |
|--------|-------------|------------|
| **manifold/** | nalgebra | core, factors, io, observers |
| **core/** | manifold, factors | optimizer |
| **factors/** | manifold | core |
| **linalg/** | faer | optimizer |
| **optimizer/** | core, linalg, observers | lib.rs |
| **io/** | manifold | lib.rs |
| **observers/** | manifold (feature-gated) | optimizer |

## Public API Design (lib.rs)

### Core Types
```rust
pub use core::variable::Variable;
pub use error::{ApexSolverError, ApexSolverResult};
```

### Factors
```rust
pub use factors::{
    BetweenFactorSE2, BetweenFactorSE3, PriorFactor,
    DoubleSphereCameraParamsFactor, DoubleSphereProjectionFactor,
    EucmCameraParamsFactor, EucmProjectionFactor,
    FovCameraParamsFactor, FovProjectionFactor,
    KannalaBrandtCameraParamsFactor, KannalaBrandtProjectionFactor,
    RadTanCameraParamsFactor, RadTanProjectionFactor,
    UcmCameraParamsFactor, UcmProjectionFactor,
};
```

### Linear Algebra
```rust
pub use linalg::{
    LinearSolverType,
    SparseCholeskySolver,
    SparseLinearSolver,
    SparseQRSolver,
};
```

### Optimizers
```rust
pub use optimizer::{
    LevenbergMarquardt, LevenbergMarquardtConfig,
    OptObserver, OptObserverVec,
    OptimizerType, Solver,
};
```

### Feature-Gated
```rust
#[cfg(feature = "visualization")]
pub use observers::visualization::RerunObserver;
```

## Module Size Analysis

| Module | Files | Total Lines | Avg per File | Largest File |
|--------|-------|-------------|--------------|--------------|
| **core** | 6 | 5,250 | 875 | loss_functions.rs (2,004) |
| **manifold** | 6 | 6,677 | 1,113 | se2.rs (1,437) |
| **optimizer** | 4 | 5,401 | 1,350 | dog_leg.rs (2,032) |
| **factors** | 10 | 3,874 | 387 | kannala_brandt.rs (480) |
| **io** | 3 | 1,537 | 512 | g2o.rs (692) |
| **linalg** | 3 | 1,508 | 503 | cholesky.rs (649) |
| **observers** | 3 | 1,744 | 581 | visualization.rs (840) |
| **root** | 3 | 389 | 130 | lib.rs (47) |
| **TOTAL** | 38 | ~25,000 | 658 | - |

## Key Architectural Patterns

### 1. Trait-Based Abstractions

**Core Traits:**
- `Solver` (optimizer/mod.rs) - Unified optimization interface
- `SparseLinearSolver` (linalg/mod.rs) - Linear algebra backends
- `Factor` (factors/mod.rs) - Constraint definitions
- `LieGroup` + `Tangent` (manifold/mod.rs) - Manifold operations
- `OptObserver` (observers/mod.rs) - Monitoring callbacks
- `GraphLoader` (io/mod.rs) - File format support

### 2. Enum-Based Static Dispatch

**VariableEnum** (core/problem.rs:110-248):
```rust
pub enum VariableEnum {
    Rn(Variable<rn::Rn>),
    SE2(Variable<se2::SE2>),
    SE3(Variable<se3::SE3>),
    SO2(Variable<so2::SO2>),
    SO3(Variable<so3::SO3>),
}
```
- Enables mixed manifold types in same optimization problem
- Uses match statements for static dispatch
- Avoids trait object overhead for variables

### 3. Builder Pattern for Configuration

All optimizers use fluent builder configuration:
```rust
let config = LevenbergMarquardtConfig::new()
    .with_max_iterations(100)
    .with_cost_tolerance(1e-6)
    .with_damping(1e-3);
```

### 4. Error Hierarchy

Each module has its own error type:
```rust
ApexSolverError
├── CoreError
├── OptimizerError
├── LinAlgError
├── ManifoldError
├── IoError
└── ObserverError
```

All implement `.log()` and `.log_with_source()` for consistent error handling.

### 5. Feature Gating

Visualization is optional:
```rust
#[cfg(feature = "visualization")]
pub mod visualization;
```

Prevents Rerun dependency for core optimization use.

## Code Organization Quality

### Strengths

1. **Single Responsibility** - Each file has one clear purpose
2. **Clear Module Boundaries** - Dependencies flow downward
3. **Public API Curation** - Careful re-exports in lib.rs
4. **Feature Isolation** - Visualization cleanly separated
5. **Consistent Patterns** - Same structure across modules

### Improvement Opportunities

1. **Optimizer Duplication** - 60-70% shared code between LM/GN/DogLeg
2. **Camera Factor Duplication** - Similar Jacobian patterns repeated
3. **No Common Solver Base** - Could extract shared functionality
4. **Missing Abstractions** - No generic camera model trait

## Recommended Refactoring

### 1. Extract Solver Base

```rust
// Proposed structure
pub struct SolverBase {
    config: SolverConfig,
    linear_solver: Box<dyn SparseLinearSolver>,
    observers: OptObserverVec,
}

impl SolverBase {
    fn initialize_state(&self, ...) -> OptimizationState { ... }
    fn check_convergence(&self, ...) -> Option<OptimizationStatus> { ... }
    fn apply_step(&self, ...) -> StepResult { ... }
}
```

### 2. Camera Model Trait

```rust
pub trait CameraModel {
    fn project(&self, point: &Vector3<f64>) -> Vector2<f64>;
    fn project_with_jacobian(&self, point: &Vector3<f64>) -> (Vector2<f64>, Matrix2x3<f64>);
    fn unproject(&self, pixel: &Vector2<f64>) -> Vector3<f64>;
}
```

Would reduce duplication in camera factor implementations.
