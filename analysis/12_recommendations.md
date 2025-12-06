# Recommendations

## Executive Summary

Apex Solver is a well-designed, production-quality optimization library. The recommendations below are improvements rather than critical fixes. The codebase is already suitable for production use.

---

## Priority Levels

- **P0 - Critical**: Should be fixed immediately (none identified)
- **P1 - High**: Significant improvement, recommend addressing
- **P2 - Medium**: Worthwhile improvement when time permits
- **P3 - Low**: Nice to have, minor benefit

---

## P1 - High Priority Recommendations

### 1. Extract Common Solver Base

**Current State:** 60-70% code duplication across LM, GN, and Dog Leg optimizers.

**Duplicated Code:**
- Convergence checking (~90 lines × 3)
- Parameter norm computation (~20 lines × 3)
- Jacobi scaling creation (~40 lines × 3)
- Optimization state initialization (~60 lines × 3)
- Summary/logging infrastructure (~100 lines × 3)

**Proposed Solution:**

```rust
// src/optimizer/common.rs

pub struct SolverBase {
    linear_solver: Box<dyn SparseLinearSolver>,
    observers: OptObserverVec,
    iteration_stats: Vec<IterationStats>,
}

impl SolverBase {
    pub fn check_convergence(
        &self, 
        config: &ConvergenceConfig,
        state: &OptimizationState
    ) -> Option<OptimizationStatus> {
        // Shared convergence logic
    }
    
    pub fn compute_parameter_norm(
        &self, 
        vars: &HashMap<String, VariableEnum>
    ) -> f64 {
        // Shared implementation
    }
    
    pub fn create_jacobi_scaling(
        &self, 
        jacobian: &SparseColMat<usize, f64>
    ) -> Vec<f64> {
        // Shared implementation
    }
    
    pub fn initialize_state(
        &self,
        problem: &Problem,
        initial_params: &HashMap<String, (ManifoldType, DVector<f64>)>
    ) -> OptimizerResult<OptimizationState> {
        // Shared initialization
    }
}
```

**Benefits:**
- ~1000 lines of code removed
- Single point for bug fixes
- Easier maintenance
- Consistent behavior across solvers

**Effort:** Medium (2-3 days)
**Impact:** High (maintainability)

---

### 2. Improve Test Coverage for Camera Factors

**Current State:**

| Factor | Tests | Coverage |
|--------|-------|----------|
| Kannala-Brandt | 9 | Excellent |
| RadTan | 6 | Good |
| UCM | 6 | Good |
| Double Sphere | 3 | Basic |
| EUCM | 3 | Basic |
| FOV | 3 | Basic |

**Recommended Tests for All Camera Models:**

1. **Dimension verification** - Residual and Jacobian dimensions
2. **Non-zero Jacobian validation** - Jacobian elements are computed
3. **Jacobian structure tests** - Verify expected zeros (e.g., ∂u/∂fy = 0)
4. **Numerical Jacobian comparison** - Compare with finite differences
5. **Invalid projection handling** - Test edge cases (behind camera, at origin)
6. **Panic tests** - Mismatched observation/point counts

**Template from Kannala-Brandt:**

```rust
#[test]
fn test_jacobian_structure() {
    let factor = create_test_factor();
    let params = create_test_params();
    let (_, jacobian) = factor.linearize(&params, true);
    let jac = jacobian.unwrap();
    
    // Verify ∂u/∂fy = 0
    assert!((jac[(0, 1)]).abs() < 1e-10);
    // Verify ∂v/∂fx = 0
    assert!((jac[(1, 0)]).abs() < 1e-10);
}

#[test]
fn test_numerical_jacobian_matches_analytical() {
    let factor = create_test_factor();
    let params = create_test_params();
    let (_, analytical) = factor.linearize(&params, true);
    let numerical = compute_numerical_jacobian(&factor, &params, 1e-7);
    
    assert_matrix_approx(analytical.unwrap(), numerical, 1e-5);
}
```

**Effort:** Low (1-2 days)
**Impact:** High (correctness confidence)

---

### 3. Centralize Numerical Constants

**Current State:** Scattered precision constants:

```rust
// double_sphere_factor.rs
const PRECISION: f64 = 1e-3;

// eucm_factor.rs
const PRECISION: f64 = 1e-3;

// fov_factor.rs
const EPS_SQRT: f64 = 1e-7;

// kannala_brandt_factor.rs
const EPSILON: f64 = 1e-10;
```

**Proposed Solution:**

```rust
// src/factors/mod.rs

/// Numerical thresholds for factor computations
pub mod thresholds {
    /// Threshold for considering a value as zero
    pub const EPSILON: f64 = 1e-10;
    
    /// Threshold for small-angle approximations
    pub const SMALL_ANGLE: f64 = 1e-7;
    
    /// Threshold for invalid projection detection
    pub const PROJECTION_MIN: f64 = 1e-3;
    
    /// Large residual for invalid projections
    pub const INVALID_RESIDUAL: f64 = 1e6;
}
```

**Benefits:**
- Single source of truth
- Easier to tune
- Consistent behavior

**Effort:** Low (1 day)
**Impact:** Medium (maintainability)

---

## P2 - Medium Priority Recommendations

### 4. Linear Solver Enum Dispatch

**Current State:**
```rust
linear_solver: Box<dyn SparseLinearSolver>
```

**Proposed Change:**
```rust
pub enum LinearSolver {
    Cholesky(SparseCholeskySolver),
    QR(SparseQRSolver),
}

impl SparseLinearSolver for LinearSolver {
    fn solve_normal_equation(&mut self, ...) -> ... {
        match self {
            Self::Cholesky(s) => s.solve_normal_equation(...),
            Self::QR(s) => s.solve_normal_equation(...),
        }
    }
}
```

**Benefits:**
- No vtable overhead (2-5% improvement)
- Still supports runtime selection
- Minimal code change

**Effort:** Low (0.5 days)
**Impact:** Low-Medium (performance)

---

### 5. Thread-Local Result Collection

**Current State:**
```rust
let results_arc = Arc::new(Mutex::new(Vec::new()));
residual_blocks.par_iter().try_for_each(|block| {
    let mut results = results_arc.lock()?;
    results.push(...);
})?;
```

**Proposed Change:**
```rust
let results: Vec<_> = residual_blocks
    .par_iter()
    .map(|(id, block)| {
        let (r, j) = block.linearize(...)?;
        Ok((id.clone(), r, j))
    })
    .collect::<Result<Vec<_>, _>>()?;
```

**Benefits:**
- No mutex contention
- 5-10% speedup for many small factors
- Cleaner code

**Effort:** Low (0.5 days)
**Impact:** Medium (performance)

---

### 6. Add Numerical Jacobian Verification Utility

**Proposed Addition:**

```rust
// src/core/testing.rs

/// Compute numerical Jacobian using finite differences
pub fn numerical_jacobian<F>(
    f: F,
    x: &DVector<f64>,
    epsilon: f64,
) -> DMatrix<f64>
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    let n = x.len();
    let y0 = f(x);
    let m = y0.len();
    
    let mut jacobian = DMatrix::zeros(m, n);
    let mut x_perturbed = x.clone();
    
    for i in 0..n {
        x_perturbed[i] = x[i] + epsilon;
        let y_plus = f(&x_perturbed);
        x_perturbed[i] = x[i] - epsilon;
        let y_minus = f(&x_perturbed);
        x_perturbed[i] = x[i];
        
        let dy = (y_plus - y_minus) / (2.0 * epsilon);
        jacobian.set_column(i, &dy);
    }
    
    jacobian
}

/// Assert two matrices are approximately equal
pub fn assert_matrix_approx(a: &DMatrix<f64>, b: &DMatrix<f64>, tol: f64) {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let diff = (a[(i, j)] - b[(i, j)]).abs();
            assert!(diff < tol, 
                "Matrix mismatch at ({}, {}): {} vs {} (diff: {})",
                i, j, a[(i, j)], b[(i, j)], diff);
        }
    }
}
```

**Benefits:**
- Standard tool for verifying analytical Jacobians
- Can be used in tests and debugging
- Catches Jacobian bugs early

**Effort:** Low (0.5 days)
**Impact:** Medium (correctness)

---

### 7. Add SECURITY.md

**Proposed Content:**

```markdown
# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Security Considerations

### Input Validation
- G2O/TORO files are validated during parsing
- Quaternion norms are checked and normalized
- Invalid projections return large residuals

### Memory Safety
- Zero unsafe code in core algorithms
- Only `memmap2` for file I/O (audited crate)

### Denial of Service
- Use `max_iterations` to limit computation
- Consider file size limits for untrusted input

## Reporting a Vulnerability

Please report security vulnerabilities to [security@example.com].
```

**Effort:** Low (0.5 days)
**Impact:** Low (documentation)

---

## P3 - Low Priority Recommendations

### 8. Use More Iterator Patterns

**Example Improvement:**

```rust
// Current
for &fixed_idx in &var.fixed_indices {
    if fixed_idx < 6 {
        step_data[fixed_idx] = 0.0;
    }
}

// Improved
var.fixed_indices.iter()
    .filter(|&&idx| idx < step_data.len())
    .for_each(|&idx| step_data[idx] = 0.0);
```

**Effort:** Low (scattered changes)
**Impact:** Low (style)

---

### 9. Add Camera Model Trait

**Proposed:**

```rust
pub trait CameraModel {
    fn project(&self, point: &Vector3<f64>) -> Option<Vector2<f64>>;
    fn project_with_jacobian(&self, point: &Vector3<f64>) 
        -> Option<(Vector2<f64>, Matrix2x3<f64>)>;
    fn unproject(&self, pixel: &Vector2<f64>, depth: f64) -> Vector3<f64>;
    fn num_params(&self) -> usize;
}
```

**Benefits:**
- Common interface for camera operations
- Could simplify factor implementations
- Useful for calibration

**Effort:** Medium (2-3 days)
**Impact:** Low (code organization)

---

### 10. Documentation Improvements

**Add missing examples in:**
- `LieGroup` trait methods
- Observer trait usage
- Custom factor creation

**Explain Jacobian derivations:**
- Add LaTeX equations in doc comments
- Reference papers for complex derivations (Q-block, etc.)

**Effort:** Medium (ongoing)
**Impact:** Low (user experience)

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
- [ ] Centralize numerical constants
- [ ] Add numerical Jacobian utility
- [ ] Linear solver enum dispatch
- [ ] Thread-local result collection

### Phase 2: Test Coverage (1 week)
- [ ] Add camera factor tests to match Kannala-Brandt level
- [ ] Add numerical Jacobian verification tests

### Phase 3: Refactoring (2-3 weeks)
- [ ] Extract common solver base
- [ ] Add documentation improvements
- [ ] Add SECURITY.md

### Phase 4: Optional Enhancements (as needed)
- [ ] Camera model trait
- [ ] Iterator pattern cleanup
- [ ] Additional file format support

---

## Summary

| Priority | Count | Effort | Impact |
|----------|-------|--------|--------|
| P1 - High | 3 | Medium | High |
| P2 - Medium | 4 | Low-Medium | Medium |
| P3 - Low | 3 | Low | Low |

**Key Takeaways:**
1. The codebase is already production-quality
2. Main improvement is reducing optimizer duplication
3. Test coverage for camera factors should match Kannala-Brandt
4. Performance optimizations have diminishing returns
5. Dynamic dispatch is appropriate and shouldn't be changed (except LinearSolver enum)

**Total Estimated Effort:** 3-4 weeks for all recommendations
**Recommended Initial Focus:** P1 items (solver base, test coverage, constants)
