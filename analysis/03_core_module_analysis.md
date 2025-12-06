# Core Module Analysis

## Overview

The `core/` module is the heart of Apex Solver, providing problem formulation, variable management, residual blocks, and robust loss functions.

**Files:**
- `mod.rs` - Module exports, `CoreError` definition
- `problem.rs` - `Problem` struct and `VariableEnum` (~1,667 LOC)
- `variable.rs` - Generic `Variable<M>` wrapper (~350 LOC)
- `residual_block.rs` - Factor wrapper with loss function (~305 LOC)
- `corrector.rs` - Loss function Jacobian correction (~225 LOC)
- `loss_functions.rs` - 16 robust loss implementations (~2,004 LOC)

## Problem Formulation (problem.rs)

### VariableEnum - Mixed Manifold Support

**Location:** `src/core/problem.rs:110-248`

```rust
pub enum VariableEnum {
    Rn(Variable<rn::Rn>),
    SE2(Variable<se2::SE2>),
    SE3(Variable<se3::SE3>),
    SO2(Variable<so2::SO2>),
    SO3(Variable<so3::SO3>),
}
```

**Design Pattern:** Enum-based static dispatch instead of `dyn Manifold` trait objects.

**Key Methods (all use exhaustive match):**
- `to_vector()` (lines 112-131) - Convert to DVector
- `get_size()` (lines 135-154) - Get tangent space dimension
- `get_manifold_type()` (lines 158-177) - Get ManifoldType enum
- `clone_inner()` (lines 181-200) - Deep clone
- `right_plus()` (lines 204-223) - Manifold retraction
- `right_minus()` (lines 227-246) - Manifold inverse retraction

**Benefits:**
- Zero virtual function overhead
- Compile-time exhaustiveness checking
- Clear type information preserved

### Problem Struct

**Location:** `src/core/problem.rs:278-625`

```rust
pub struct Problem {
    residual_blocks: HashMap<String, ResidualBlock>,
    fixed_indices: HashMap<String, HashSet<usize>>,
    // Symbolic structure computed once
    symbolic_structure: Option<SymbolicStructure>,
}
```

**Core Operations:**

1. **add_residual_block()** (lines 590-625)
   - Accepts `Box<dyn Factor + Send>` and `Option<Box<dyn LossFunction + Send>>`
   - Creates unique block ID
   - Stores in HashMap for O(1) access

2. **build_symbolic_structure()** (lines 682-789)
   - Pre-computes Jacobian sparsity pattern
   - Called once before optimization
   - Maps variable names to column indices
   - Enables efficient sparse matrix construction

3. **compute_residual_sparse()** (lines 826-900)
   - Fast residual-only computation
   - Used for cost evaluation (step acceptance)
   - No Jacobian computation

4. **compute_residual_and_jacobian_sparse()** (lines 902-1009)
   - Full linearization with parallel evaluation
   - Uses `rayon::par_iter()` for residual blocks
   - Thread-safe via `Arc<Mutex<>>`

5. **apply_tangent_step()** (lines 1076-1161)
   - Applies optimization step to variables
   - Respects fixed indices (zero update for fixed DOF)
   - Handles each manifold type via match statement

### Parallel Computation Pattern

**Location:** `src/core/problem.rs:988-1008`

```rust
residual_blocks
    .par_iter()
    .try_for_each(|(block_id, block)| -> CoreResult<()> {
        // Evaluate factor
        let (residual, jacobian) = block.linearize(...)?;
        
        // Thread-safe accumulation
        let mut results = results_arc.lock().map_err(...)?;
        results.push((block_id.clone(), residual, jacobian));
        Ok(())
    })?;
```

**Thread Safety:**
- `Arc<Mutex<Vec<...>>>` for result collection
- Proper error propagation via `map_err()`
- No data races possible

## Variable System (variable.rs)

### Generic Variable<M>

**Location:** `src/core/variable.rs:101-182`

```rust
pub struct Variable<M>
where
    M: LieGroup + Clone + 'static,
    M::TangentVector: Tangent<M>,
{
    value: M,
    fixed_indices: HashSet<usize>,
}
```

**Key Methods:**
- `new(value: M)` - Create from manifold element
- `get_value()` / `set_value()` - Access underlying manifold
- `right_plus(tangent)` - Apply manifold update
- `to_vector()` - Convert to DVector for optimization

**Type Bounds:**
- `M: LieGroup + Clone + 'static`
- `M::TangentVector: Tangent<M>`

These ensure:
- Full manifold operations available
- Cloneable for state snapshots
- Static lifetime for storage in collections

## Residual Block (residual_block.rs)

### Structure

**Location:** `src/core/residual_block.rs:38-42`

```rust
pub struct ResidualBlock {
    pub variable_names: Vec<String>,
    pub factor: Box<dyn Factor + Send>,
    pub loss_func: Option<Box<dyn LossFunction + Send>>,
}
```

**Dynamic Dispatch Usage:**
- `Box<dyn Factor + Send>` - Enables any factor type
- `Option<Box<dyn LossFunction + Send>>` - Optional robust loss

**Rationale:** Factors and loss functions are unknown at compile time in general optimization problems. Dynamic dispatch is appropriate here.

### Linearization Flow

**Location:** `src/core/residual_block.rs:78-170`

```rust
pub fn linearize(&self, variables: &[&VariableEnum], compute_jacobian: bool) 
    -> CoreResult<(DVector<f64>, Option<DMatrix<f64>>)> 
{
    // 1. Extract variable vectors
    let params: Vec<DVector<f64>> = variables.iter()
        .map(|v| v.to_vector())
        .collect();
    
    // 2. Call factor linearization
    let (residual, jacobian) = self.factor.linearize(&params, compute_jacobian);
    
    // 3. Apply loss function correction (if present)
    if let Some(ref loss) = self.loss_func {
        if let Some(ref jac) = jacobian {
            let corrector = Corrector::new(&residual, loss.as_ref());
            let corrected_jac = corrector.correct_jacobian(jac)?;
            let corrected_res = corrector.correct_residuals()?;
            return Ok((corrected_res, Some(corrected_jac)));
        }
    }
    
    Ok((residual, jacobian))
}
```

## Loss Functions (loss_functions.rs)

### LossFunction Trait

**Location:** `src/core/loss_functions.rs:67-85`

```rust
pub trait LossFunction: Send + Sync {
    /// Evaluate loss function at squared residual s = r^T r
    /// Returns [rho(s), rho'(s), rho''(s)]
    fn evaluate(&self, s: f64) -> [f64; 3];
}
```

**Design Notes:**
- Returns array of 3 values: ρ(s), ρ'(s), ρ''(s)
- Used by Corrector for Jacobian scaling
- `Send + Sync` for parallel evaluation

### Implemented Loss Functions

| Loss Function | Parameters | Use Case |
|---------------|------------|----------|
| `L2Loss` | None | Standard least squares |
| `L1Loss` | None | Robust, handles outliers |
| `HuberLoss` | `a` (threshold) | Smooth L1 transition |
| `SoftL1Loss` | `a` (scale) | Smooth absolute loss |
| `CauchyLoss` | `c` (scale) | Heavy-tailed distributions |
| `GemanMcClureLoss` | `a` (scale) | Computer vision applications |
| `WelschLoss` | `c` (scale) | Smooth redescending |
| `TukeyBiweightLoss` | `c` (scale) | Hard rejection of outliers |
| `ArcsinhSmooth` | `a, s` (params) | Smooth, bounded |
| `ScaledError` | `a` (scale) | Simple scaling |
| `TDistributionLoss` | `nu` (DOF) | Student-t distribution |
| `BarronLoss` | `alpha, c` (params) | Generalized loss family |
| `AdaptiveLoss` | Various | Self-adapting |
| `ComposedLoss` | Two losses | Composition |
| `HuberCharbonnier` | `epsilon` | Edge-aware |
| `FairLoss` | `c` (scale) | Fair penalty |

### Example Implementation (HuberLoss)

**Location:** `src/core/loss_functions.rs:168-287`

```rust
impl HuberLoss {
    pub fn new(a: f64) -> Result<Self, String> {
        if a <= 0.0 {
            return Err("Huber parameter 'a' must be positive".to_string());
        }
        Ok(HuberLoss { a })
    }
}

impl LossFunction for HuberLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let a = self.a;
        let sqrt_s = s.sqrt();
        
        if sqrt_s <= a {
            // Quadratic region
            [s, 1.0, 0.0]
        } else {
            // Linear region
            let rho = 2.0 * a * sqrt_s - a * a;
            let drho = a / sqrt_s;
            let ddrho = -0.5 * a / (s * sqrt_s);
            [rho, drho, ddrho]
        }
    }
}
```

**Quality Notes:**
- Parameter validation in constructor
- Numerical stability for small/large values
- Clear mathematical documentation

## Corrector (corrector.rs)

### Purpose

Applies loss function derivatives to scale Jacobians and residuals, implementing the IRLS (Iteratively Reweighted Least Squares) formulation.

**Location:** `src/core/corrector.rs:40-225`

### Key Algorithm

**correct_jacobian()** (lines 103-183):

```rust
pub fn correct_jacobian(&self, jacobian: &DMatrix<f64>) -> CoreResult<DMatrix<f64>> {
    // Compute loss function derivatives
    let [rho, drho, ddrho] = self.loss_func.evaluate(self.s);
    
    // Scale factor for Jacobian
    let alpha = if self.s > f64::EPSILON {
        (drho / self.s).sqrt()
    } else {
        1.0
    };
    
    // Apply correction
    let corrected = jacobian * alpha;
    Ok(corrected)
}
```

**References Ceres Solver algorithm** for proper Jacobian scaling with robust loss functions.

## Error Handling

### CoreError Enum

**Location:** `src/core/mod.rs:18-50`

```rust
#[derive(Debug, Clone, Error)]
pub enum CoreError {
    #[error("Residual block error: {0}")]
    ResidualBlock(String),
    
    #[error("Variable error: {0}")]
    Variable(String),
    
    #[error("Factor linearization failed: {0}")]
    FactorLinearization(String),
    
    #[error("Symbolic structure computation failed: {0}")]
    SymbolicStructure(String),
    
    #[error("Parallel computation error: {0}")]
    ParallelComputation(String),
    
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    
    #[error("Invalid constraint: {0}")]
    InvalidConstraint(String),
    
    #[error("Loss function error: {0}")]
    LossFunction(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}
```

### Error Methods

**Location:** `src/core/mod.rs:35-68`

```rust
impl CoreError {
    #[must_use]
    pub fn log(self) -> Self {
        error!("{}", self);
        self
    }
    
    #[must_use]
    pub fn log_with_source<E: std::error::Error>(self, source: E) -> Self {
        error!("{}: {}", self, source);
        self
    }
}
```

**Pattern:** `#[must_use]` ensures errors are not silently dropped.

## Code Quality Assessment

### Strengths

| Aspect | Rating | Notes |
|--------|--------|-------|
| Error Handling | Excellent | Custom types, logging, no unwrap |
| Type Safety | Excellent | Generic Variable<M>, VariableEnum |
| Documentation | Very Good | Clear algorithm descriptions |
| Testing | Good | Embedded tests, could be more comprehensive |
| Thread Safety | Excellent | Proper Arc<Mutex<>> usage |

### Areas for Improvement

1. **Fixed Index Handling** (problem.rs:1088-1094)
   - Manual bounds checking could use iterator patterns
   - Current: `if fixed_idx < 6 { step_data[fixed_idx] = 0.0; }`

2. **Jacobian Assembly** (problem.rs:1081)
   - Allocates temporary Vec for each variable
   - Could use iterator views

3. **Loss Function Trait**
   - Single `evaluate()` method returns array
   - Could split into separate methods for clarity

## Dynamic Dispatch Analysis

### Current Usage

| Location | Type | Frequency |
|----------|------|-----------|
| `ResidualBlock.factor` | `Box<dyn Factor + Send>` | Per block |
| `ResidualBlock.loss_func` | `Option<Box<dyn LossFunction + Send>>` | Per block |

### Static Dispatch Alternative

**LossFunction could use enum dispatch:**

```rust
pub enum LossFunctionEnum {
    L2(L2Loss),
    Huber(HuberLoss),
    Cauchy(CauchyLoss),
    // ... 13 more variants
}

impl LossFunctionEnum {
    pub fn evaluate(&self, s: f64) -> [f64; 3] {
        match self {
            Self::L2(l) => l.evaluate(s),
            Self::Huber(l) => l.evaluate(s),
            // ...
        }
    }
}
```

**Trade-offs:**
- Pro: ~5% performance gain from inlining
- Con: Larger binary, less flexible for custom losses
- Verdict: **Not recommended** - loss evaluation is not a bottleneck

### Factor Trait

**Cannot easily convert to enum dispatch:**
- 10+ factor types (camera models, pose factors)
- Users may define custom factors
- Dynamic dispatch overhead negligible (once per block per iteration)

**Verdict:** Keep `Box<dyn Factor + Send>` - appropriate design choice.
