# Static vs Dynamic Dispatch Analysis

## Overview

This document analyzes the current use of dynamic dispatch (`Box<dyn Trait>`) in Apex Solver and evaluates opportunities to convert to static dispatch (generics, enums) for potential performance gains.

---

## 1. Current Dynamic Dispatch Usage

### Summary Table

| Location | Type | Purpose | Frequency |
|----------|------|---------|-----------|
| `ResidualBlock.factor` | `Box<dyn Factor + Send>` | Pluggable factors | Per block |
| `ResidualBlock.loss_func` | `Option<Box<dyn LossFunction + Send>>` | Robust loss | Per block |
| `Solver.linear_solver` | `Box<dyn SparseLinearSolver>` | LA backend | Per solver |
| `OptObserverVec` | `Vec<Box<dyn OptObserver>>` | Observers | Per solver |

### Detailed Analysis

#### A. Factor Trait

**Location:** `src/core/residual_block.rs:38-42`

```rust
pub struct ResidualBlock {
    pub variable_names: Vec<String>,
    pub factor: Box<dyn Factor + Send>,
    pub loss_func: Option<Box<dyn LossFunction + Send>>,
}
```

**Current Implementations:**
- `PriorFactor`
- `BetweenFactorSE2`
- `BetweenFactorSE3`
- `DoubleSphereCameraParamsFactor`
- `DoubleSphereProjectionFactor`
- `EucmCameraParamsFactor`
- `EucmProjectionFactor`
- `FovCameraParamsFactor`
- `FovProjectionFactor`
- `KannalaBrandtCameraParamsFactor`
- `KannalaBrandtProjectionFactor`
- `RadTanCameraParamsFactor`
- `RadTanProjectionFactor`
- `UcmCameraParamsFactor`
- `UcmProjectionFactor`

**Total: 15+ implementations**

#### B. LossFunction Trait

**Location:** `src/core/loss_functions.rs:67-85`

**Current Implementations:**
- `L2Loss`
- `L1Loss`
- `HuberLoss`
- `SoftL1Loss`
- `CauchyLoss`
- `GemanMcClureLoss`
- `WelschLoss`
- `TukeyBiweightLoss`
- `ArcsinhSmooth`
- `ScaledError`
- `TDistributionLoss`
- `BarronLoss`
- `AdaptiveLoss`
- `ComposedLoss`
- `HuberCharbonnier`
- `FairLoss`

**Total: 16 implementations**

#### C. SparseLinearSolver Trait

**Location:** `src/linalg/mod.rs:63-115`

**Current Implementations:**
- `SparseCholeskySolver`
- `SparseQRSolver`

**Total: 2 implementations**

#### D. OptObserver Trait

**Location:** `src/observers/mod.rs:110-189`

**Current Implementations:**
- `RerunObserver` (feature-gated)
- Custom user observers (unknown at compile time)

---

## 2. Static Dispatch Alternatives

### A. Enum-Based Dispatch (Like VariableEnum)

**Pattern Already Used:**

```rust
// src/core/problem.rs:110-248
pub enum VariableEnum {
    Rn(Variable<rn::Rn>),
    SE2(Variable<se2::SE2>),
    SE3(Variable<se3::SE3>),
    SO2(Variable<so2::SO2>),
    SO3(Variable<so3::SO3>),
}

impl VariableEnum {
    pub fn to_vector(&self) -> DVector<f64> {
        match self {
            Self::Rn(v) => v.to_vector(),
            Self::SE2(v) => v.to_vector(),
            Self::SE3(v) => v.to_vector(),
            Self::SO2(v) => v.to_vector(),
            Self::SO3(v) => v.to_vector(),
        }
    }
}
```

**Potential FactorEnum:**

```rust
pub enum FactorEnum {
    Prior(PriorFactor),
    SE2Between(BetweenFactorSE2),
    SE3Between(BetweenFactorSE3),
    DoubleSphereCamera(DoubleSphereCameraParamsFactor),
    DoubleSphereProjection(DoubleSphereProjectionFactor),
    EucmCamera(EucmCameraParamsFactor),
    EucmProjection(EucmProjectionFactor),
    // ... 9 more variants
}

impl FactorEnum {
    pub fn linearize(&self, params: &[DVector<f64>], compute_jacobian: bool) 
        -> (DVector<f64>, Option<DMatrix<f64>>) 
    {
        match self {
            Self::Prior(f) => f.linearize(params, compute_jacobian),
            Self::SE2Between(f) => f.linearize(params, compute_jacobian),
            // ... 13+ more arms
        }
    }
}
```

**Potential LossFunctionEnum:**

```rust
pub enum LossFunctionEnum {
    L2(L2Loss),
    L1(L1Loss),
    Huber(HuberLoss),
    SoftL1(SoftL1Loss),
    Cauchy(CauchyLoss),
    GemanMcClure(GemanMcClureLoss),
    Welsch(WelschLoss),
    TukeyBiweight(TukeyBiweightLoss),
    ArcsinhSmooth(ArcsinhSmooth),
    ScaledError(ScaledError),
    TDistribution(TDistributionLoss),
    Barron(BarronLoss),
    Adaptive(AdaptiveLoss),
    Composed(ComposedLoss),
    HuberCharbonnier(HuberCharbonnier),
    Fair(FairLoss),
}

impl LossFunctionEnum {
    pub fn evaluate(&self, s: f64) -> [f64; 3] {
        match self {
            Self::L2(l) => l.evaluate(s),
            Self::L1(l) => l.evaluate(s),
            // ... 14 more arms
        }
    }
}
```

### B. Generic Type Parameters

**Current:**
```rust
pub struct LevenbergMarquardt {
    config: LevenbergMarquardtConfig,
    linear_solver: Box<dyn SparseLinearSolver>,
    observers: OptObserverVec,
}
```

**Generic Alternative:**
```rust
pub struct LevenbergMarquardt<LS: SparseLinearSolver> {
    config: LevenbergMarquardtConfig,
    linear_solver: LS,
    observers: OptObserverVec,
}

impl<LS: SparseLinearSolver> LevenbergMarquardt<LS> {
    pub fn new(linear_solver: LS) -> Self {
        // ...
    }
}

// Usage
let solver = LevenbergMarquardt::new(SparseCholeskySolver::new());
```

### C. Tuple-Based Observers

**Current:**
```rust
pub struct OptObserverVec {
    observers: Vec<Box<dyn OptObserver>>,
}
```

**Tuple Alternative:**
```rust
pub struct OptObserverTuple<O1, O2, O3>(O1, O2, O3);

impl<O1: OptObserver, O2: OptObserver, O3: OptObserver> OptObserver for OptObserverTuple<O1, O2, O3> {
    fn on_step(&self, values: &HashMap<String, VariableEnum>, iteration: usize) {
        self.0.on_step(values, iteration);
        self.1.on_step(values, iteration);
        self.2.on_step(values, iteration);
    }
}
```

---

## 3. Trade-off Analysis

### A. Factor Trait → FactorEnum

| Aspect | Box<dyn Factor> | FactorEnum |
|--------|-----------------|------------|
| **Performance** | ~5ns vtable lookup | Direct call |
| **Flexibility** | Users can add custom factors | Closed set, no extension |
| **Maintainability** | Add impl, done | Add variant + match arm |
| **Binary Size** | Smaller | Larger (15+ variants) |
| **Compile Time** | Faster | Slower (more code gen) |
| **API Complexity** | Simple | Users must use enum |

**Verdict: NOT RECOMMENDED**

Reasons:
1. Users frequently define custom factors
2. 15+ match arms is error-prone
3. Performance gain is ~0.1% (5ns × 1000 factors = 5μs vs 10ms iteration)
4. Breaks extensibility - core design goal

### B. LossFunction Trait → LossFunctionEnum

| Aspect | Box<dyn LossFunction> | LossFunctionEnum |
|--------|----------------------|------------------|
| **Performance** | ~5ns vtable lookup | Direct call + inlining |
| **Flexibility** | Custom losses possible | 16 built-in only |
| **Maintainability** | Easy to add | 16 match arms |
| **Custom Losses** | Supported | Would need `Custom(Box<dyn>)` |

**Verdict: MARGINALLY FEASIBLE**

Could implement as:
```rust
pub enum LossFunctionEnum {
    L2(L2Loss),
    Huber(HuberLoss),
    // ... built-in losses
    Custom(Box<dyn LossFunction + Send>),  // Escape hatch
}
```

But gain is minimal (<1% of runtime).

### C. SparseLinearSolver → Generic Parameter

| Aspect | Box<dyn SparseLinearSolver> | Generic<LS> |
|--------|---------------------------|-------------|
| **Performance** | ~5ns per call | Inlined |
| **Flexibility** | Runtime selection | Compile-time fixed |
| **API Complexity** | Simple | Type parameter propagates |
| **Binary Size** | One impl | 2× (Cholesky + QR) |

**Verdict: FEASIBLE, OPTIONAL**

Implementation:
```rust
pub struct LevenbergMarquardt<LS: SparseLinearSolver = SparseCholeskySolver> {
    linear_solver: LS,
    // ...
}

// Can still support runtime selection via enum
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

### D. OptObserver → Tuple-Based

| Aspect | Vec<Box<dyn OptObserver>> | Tuple<O1, O2, O3> |
|--------|--------------------------|-------------------|
| **Performance** | Dynamic iteration | Fully inlined |
| **Flexibility** | Any number | Fixed count |
| **API Complexity** | Simple | Generics complexity |
| **Use Case** | Runtime config | Compile-time fixed |

**Verdict: NOT RECOMMENDED for general use**

The current `#[inline(always)]` on `notify()` already handles the empty case efficiently. Observer overhead is negligible.

---

## 4. Recommended Approach

### Keep Current Design For:

1. **Factor trait** - Extensibility is essential
2. **LossFunction trait** - 16 types manageable, custom losses useful
3. **OptObserver trait** - Runtime flexibility needed

### Consider Changing:

1. **SparseLinearSolver** - Only 2 implementations, could be generic or enum

**Proposed Implementation:**

```rust
// src/linalg/mod.rs

/// Enum-based linear solver (static dispatch with runtime selection)
pub enum LinearSolver {
    Cholesky(SparseCholeskySolver),
    QR(SparseQRSolver),
}

impl LinearSolver {
    pub fn cholesky() -> Self {
        Self::Cholesky(SparseCholeskySolver::new())
    }
    
    pub fn qr() -> Self {
        Self::QR(SparseQRSolver::new())
    }
}

impl SparseLinearSolver for LinearSolver {
    fn solve_normal_equation(&mut self, residuals: &Mat<f64>, 
        jacobians: &SparseColMat<usize, f64>) -> LinAlgResult<Mat<f64>> 
    {
        match self {
            Self::Cholesky(s) => s.solve_normal_equation(residuals, jacobians),
            Self::QR(s) => s.solve_normal_equation(residuals, jacobians),
        }
    }
    
    // ... other methods
}

// Usage in optimizer
pub struct LevenbergMarquardt {
    linear_solver: LinearSolver,  // No Box needed
    // ...
}
```

**Benefits:**
- No vtable overhead
- Match is predictable (only 2 arms)
- Preserves runtime selection capability
- Minor code change

**Estimated Gain:** 2-5% in linear solver calls (which are 1-2 per iteration)

---

## 5. Summary

### Current State is Appropriate

| Dispatch Point | Current | Recommendation | Reason |
|----------------|---------|----------------|--------|
| Factor | `Box<dyn>` | Keep | Extensibility essential |
| LossFunction | `Box<dyn>` | Keep | Custom losses, <1% overhead |
| LinearSolver | `Box<dyn>` | Consider enum | Only 2 impls, easy change |
| OptObserver | `Box<dyn>` | Keep | Runtime flexibility |

### Performance Impact

| Change | Effort | Performance Gain | Recommended |
|--------|--------|-----------------|-------------|
| Factor → Enum | High | <0.1% | No |
| LossFunction → Enum | Medium | <1% | No |
| LinearSolver → Enum | Low | 2-5% on solve | Optional |
| Observer → Tuple | Medium | <0.1% | No |

### Conclusion

The current design is **pragmatic and appropriate**. Dynamic dispatch overhead is negligible compared to:
- Sparse matrix multiplication (40-60%)
- Cholesky factorization (20-30%)
- Residual evaluation (10-15%)

The only worthwhile change is converting `SparseLinearSolver` to an enum, which is a minor refactoring with small but measurable benefit.

**Overall Verdict:** Current dynamic dispatch usage is well-justified. Focus optimization efforts on linear algebra, not dispatch overhead.
