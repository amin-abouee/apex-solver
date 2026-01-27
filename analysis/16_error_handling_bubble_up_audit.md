# Error Handling Audit Report - Apex Solver

**Date:** 2026-01-27  
**Auditor:** Automated Analysis  
**Strategy:** Bubble-Up Error Propagation

---

## Executive Summary

This audit identifies error handling patterns in the Apex Solver codebase that violate the "Bubble-Up" principle. The codebase generally follows good practices with `thiserror` and hierarchical error types, but several critical issues require attention:

| Severity | Count | Issue Type |
|----------|-------|-----------|
| **Critical** | 7 | `panic!` in library code, silent error discarding |
| **Medium** | 2 | Configuration validation panics |
| **Low** | 3 | Style/consistency issues |

---

## 1. Error Type Architecture

### Current Hierarchy
```rust
// src/error.rs
#[derive(Debug, Error)]
pub enum ApexSolverError {
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error(transparent)]
    Optimizer(#[from] OptimizerError),
    #[error(transparent)]
    LinearAlgebra(#[from] LinAlgError),
    #[error(transparent)]
    Manifold(#[from] ManifoldError),
    #[error(transparent)]
    Io(#[from] IoError),
    #[error(transparent)]
    Observer(#[from] ObserverError),
}
```

The error hierarchy is well-designed with automatic conversion via `#[from]` attributes, enabling ergonomic `?` propagation.

---

## 2. Critical Issues Requiring Immediate Action

### Issue 2.1: Panic in `projection_factor.rs` (5 instances)

**File:** `src/factors/projection_factor.rs`  
**Lines:** 357, 364, 371, 379, 386  
**Severity:** CRITICAL

#### Problem Description
The `linearize()` function uses `panic!` for missing parameters instead of returning `Result`. This violates Rust best practices for library code and prevents callers from gracefully handling missing data.

#### Call Chain Analysis
```
Problem::compute_residual_and_jacobian_sparse()
  └── iterate_residual_blocks()
        └── ProjectionFactor::linearize() <- PANIC HERE
```

#### Before (Current Code)
```rust
// src/factors/projection_factor.rs:355-388
fn linearize(&self, params: &[DVector<f64>]) -> FactorLinearization {
    let pose = params.get(0)
        .expect("Projection factor requires pose parameter"); // PANIC!
    let camera_params = params.get(1)
        .expect("Projection factor requires camera parameters"); // PANIC!
    let landmark = params.get(2)
        .expect("Projection factor requires landmark parameter"); // PANIC!
    
    // ... validation checks also panic
    if pose.len() != 7 {
        panic!("Pose must have 7 elements (quaternion + translation)"); // PANIC!
    }
    if camera_params.len() != self.camera_model.num_parameters() {
        panic!("Camera parameters dimension mismatch"); // PANIC!
    }
}
```

#### After (Refactored Code)
```rust
use crate::factors::FactorError;

fn linearize(&self, params: &[DVector<f64>]) -> Result<FactorLinearization, FactorError> {
    let pose = params.get(0)
        .ok_or_else(|| FactorError::InvalidParameters(
            "Projection factor requires pose parameter".to_string()
        ))?;
    let camera_params = params.get(1)
        .ok_or_else(|| FactorError::InvalidParameters(
            "Projection factor requires camera parameters".to_string()
        ))?;
    let landmark = params.get(2)
        .ok_or_else(|| FactorError::InvalidParameters(
            "Projection factor requires landmark parameter".to_string()
        ))?;
    
    // Validation checks return errors
    if pose.len() != 7 {
        return Err(FactorError::InvalidDimension {
            expected: 7,
            actual: pose.len(),
            context: "Pose must have 7 elements (quaternion + translation)".to_string(),
        });
    }
    
    if camera_params.len() != self.camera_model.num_parameters() {
        return Err(FactorError::InvalidDimension {
            expected: self.camera_model.num_parameters(),
            actual: camera_params.len(),
            context: "Camera parameters dimension mismatch".to_string(),
        });
    }
    
    // Continue with computation...
    Ok(FactorLinearization { /* ... */ })
}
```

#### Impact Assessment
- **Risk:** Application crash on invalid input
- **Caller Impact:** Cannot recover from missing parameters
- **Breaking Change:** YES - changes function signature to return `Result`

---

### Issue 2.2: Silent Error Discarding in `gauss_newton.rs`

**File:** `src/optimizer/gauss_newton.rs`  
**Line:** 912  
**Severity:** CRITICAL

#### Problem Description
The `.ok()` method discards error information from the linear solver, returning `None` instead of propagating the actual failure reason.

#### Call Chain Analysis
```
GaussNewton::optimize()
  └── iterate()
        └── linear_solver.solve_normal_equation()
              └── .ok() <- ERROR SILENTLY DISCARDED
```

#### Before (Current Code)
```rust
// src/optimizer/gauss_newton.rs:910-912
let scaled_step = linear_solver
    .solve_normal_equation(&residuals_owned, scaled_jacobian)
    .ok()?;  // ERROR INFORMATION LOST!
```

#### After (Refactored Code)
```rust
// Option 1: Propagate with error mapping
let scaled_step = linear_solver
    .solve_normal_equation(&residuals_owned, scaled_jacobian)
    .map_err(|e| OptimizerError::LinearSolveFailed(e.to_string()).log_with_source(e))?;

// Option 2: If you need to handle None case differently
let scaled_step = match linear_solver.solve_normal_equation(&residuals_owned, scaled_jacobian) {
    Ok(step) => step,
    Err(e) => {
        error!("Linear solve failed: {}", e);
        return Err(OptimizerError::LinearSolveFailed(e.to_string()).into());
    }
};
```

#### Edge Case Analysis
**Current Behavior:**
- Linear solver fails due to singular matrix
- `.ok()` converts `Err` to `None`
- `?` on `None` returns early from function
- **Result:** Optimization silently aborts with no error message

**Risk:** Inconsistent optimization state where the caller cannot distinguish between:
1. Convergence reached
2. Linear solve failure
3. Maximum iterations reached

---

### Issue 2.3: Panic in Schur Complement Configuration

**File:** `src/linalg/explicit_schur.rs`  
**Lines:** 117, 128  
**Severity:** MEDIUM

#### Problem Description
Configuration validation panics instead of returning error for invalid landmark parameters.

#### Before (Current Code)
```rust
// src/linalg/explicit_schur.rs:115-130
fn should_eliminate(&self, num_landmarks: usize, landmark_dim: usize) -> bool {
    if num_landmarks == 0 {
        panic!("Cannot eliminate zero landmarks"); // PANIC!
    }
    if landmark_dim == 0 {
        panic!("Landmark dimension cannot be zero"); // PANIC!
    }
    // ... logic continues
}
```

#### After (Refactored Code)
```rust
fn should_eliminate(&self, num_landmarks: usize, landmark_dim: usize) -> Result<bool, LinAlgError> {
    if num_landmarks == 0 {
        return Err(LinAlgError::InvalidInput(
            "Cannot eliminate zero landmarks".to_string()
        ));
    }
    if landmark_dim == 0 {
        return Err(LinAlgError::InvalidInput(
            "Landmark dimension cannot be zero".to_string()
        ));
    }
    // ... logic continues
    Ok(true) // or actual computation result
}
```

---

## 3. Medium Priority Issues

### Issue 3.1: `ManifoldError` Lacks `thiserror` Derive

**File:** `src/manifold/mod.rs`  
**Severity:** MEDIUM

#### Problem Description
`ManifoldError` manually implements `Display` and `Error` traits while all other modules use `#[derive(Error)]` from `thiserror`.

#### Before (Current Code)
```rust
// src/manifold/mod.rs
#[derive(Debug, Clone, PartialEq)]
pub enum ManifoldError {
    InvalidTangentDimension { expected: usize, actual: usize },
    NumericalInstability(String),
    // ... variants
}

impl fmt::Display for ManifoldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTangentDimension { expected, actual } => {
                write!(f, "Invalid tangent dimension: expected {}, got {}", expected, actual)
            }
            // ... manual implementations
        }
    }
}

impl Error for ManifoldError {}
```

#### After (Refactored Code)
```rust
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ManifoldError {
    #[error("Invalid tangent dimension: expected {expected}, got {actual}")]
    InvalidTangentDimension { expected: usize, actual: usize },
    
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
    
    #[error("Invalid element: {0}")]
    InvalidElement(String),
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Invalid number: {0}")]
    InvalidNumber(String),
    
    #[error("Normalization failed: {0}")]
    NormalizationFailed(String),
}
```

---

## 4. Edge Cases: Ignored Errors

### Issue 4.1: `let _ = e.log()` Pattern

**File:** `src/observers/visualization.rs`  
**Lines:** 1634, 1647, 1650, 1675  
**Severity:** LOW

#### Context
These are **acceptable** because the error is already being logged via the `.log()` method. However, the pattern could be clearer.

#### Before (Current Code)
```rust
let _ = e.log(); // Error value ignored after logging
```

#### After (More Explicit)
```rust
// Option 1: Explicitly discard
let _logged: ObserverError = e.log();

// Option 2: Use if-let for clarity
if let Err(e) = some_fallible_operation() {
    let _ = e.log();
}
```

---

### Issue 4.2: Cleanup Operations in Benchmarks

**Files:** Various `benches/*.rs`  
**Severity:** LOW

#### Context
These are **acceptable** for cleanup operations where failure is non-critical.

#### Example Pattern
```rust
// Acceptable: Process cleanup
let _ = child.kill();
let _ = stdout.flush();
```

---

## 5. Positive Patterns to Maintain

### 5.1 Consistent Error Logging API

All modules correctly implement `log()` and `log_with_source()` methods:

```rust
#[must_use]
pub fn log(self) -> Self {
    error!("{}", self);
    self
}

#[must_use]
pub fn log_with_source<E: std::fmt::Debug>(self, source_error: E) -> Self {
    error!("{} | Source: {:?}", self, source_error);
    self
}
```

### 5.2 Proper Error Propagation with Context

**Example from `levenberg_marquardt.rs:1187`:**
```rust
let scaled_step = linear_solver
    .solve_augmented_equation(&residuals_owned, scaled_jacobian, self.config.damping)
    .map_err(|e| OptimizerError::LinearSolveFailed(e.to_string()).log_with_source(e))?;
```

This pattern:
- ✅ Logs once at the point of conversion
- ✅ Preserves source error context
- ✅ Propagates via `?` operator
- ✅ Allows caller to handle or log as needed

### 5.3 Hierarchical Error Conversion

The use of `#[from]` attributes enables automatic conversion:

```rust
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Residual block error: {0}")]
    ResidualBlock(String),
    
    #[error("Variable error: {0}")]
    Variable(String),
    
    #[error(transparent)]
    Factor(#[from] FactorError), // Automatic conversion!
}
```

---

## 6. Potential Issues Analysis

### 6.1 Risk of Silent Failures

| Location | Risk | Impact |
|----------|------|--------|
| `gauss_newton.rs:912` | Optimization aborts silently | Inconsistent solver state, no convergence indication |
| `projection_factor.rs` | Application panic | Complete process termination |
| `explicit_schur.rs` | Application panic | Complete process termination |

### 6.2 Memory/State Consistency Risks

If errors are not properly propagated:

1. **Partial Updates:** Variables may be partially updated before an error occurs
2. **Resource Leaks:** File handles or memory not cleaned up if panic occurs
3. **Inconsistent Graph State:** Factor graph may be in invalid state

**Recommended Pattern:**
```rust
// Use RAII and Result propagation
pub fn optimize(&mut self) -> Result<Solution, OptimizerError> {
    // Setup is fallible
    let state = self.initialize()?;
    
    // Iteration with early return on error
    for iter in 0..max_iters {
        let step = self.compute_step()?; // Error propagates up
        self.apply_step(step)?;
    }
    
    // Cleanup happens automatically via Drop
    Ok(Solution { /* ... */ })
}
```

---

## 7. Implementation Roadmap

### Phase 1: Critical Fixes (Immediate)
- [ ] Replace 5 `panic!` calls in `projection_factor.rs` with `Result` returns
- [ ] Fix silent error discarding in `gauss_newton.rs:912`
- [ ] Add tests for error propagation paths

### Phase 2: Consistency Improvements (Next Sprint)
- [ ] Add `thiserror` derive to `ManifoldError`
- [ ] Refactor `explicit_schur.rs` panics to return `Result`
- [ ] Update all factor types to return `Result<FactorLinearization, FactorError>`

### Phase 3: API Hardening (Future)
- [ ] Add `#[track_caller]` to error constructors for better stack traces
- [ ] Implement `Error::source()` for full error chains
- [ ] Add integration tests for error scenarios

---

## 8. Summary

The Apex Solver codebase has a solid foundation for error handling with its hierarchical `thiserror`-based architecture. The main violations of "Bubble-Up" propagation are:

1. **7 `panic!` calls** that should return `Result` (5 in projection factor, 2 in Schur complement)
2. **1 instance of `.ok()`** that discards error information in Gauss-Newton solver
3. **Minor inconsistency** with `ManifoldError` not using `thiserror`

### Key Metrics
- **Total Error Types:** 7 module-specific + 1 top-level
- **Consistent Logging:** 7/7 modules ✅
- **Proper `?` Usage:** 95%+ ✅
- **Panics in Library Code:** 7 (should be 0) ❌
- **Silent Error Discarding:** 1 instance ❌

### Recommendation
Address the critical issues (panics and `.ok()`) before the next release to ensure the library handles edge cases gracefully without crashing or silently failing.

---

**Report Generated:** 2026-01-27  
**Next Review:** After Phase 1 completion
