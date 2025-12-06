# Linear Algebra and Factors Analysis

## Part 1: Linear Algebra Module

### Overview

The `linalg/` module provides sparse linear solvers for the normal equations.

**Files:**
- `mod.rs` - `SparseLinearSolver` trait, types (~213 LOC)
- `cholesky.rs` - Sparse Cholesky solver (~649 LOC)
- `qr.rs` - Sparse QR solver (~646 LOC)

### SparseLinearSolver Trait

**Location:** `src/linalg/mod.rs:63-115`

```rust
pub trait SparseLinearSolver {
    /// Solve J^T J x = -J^T r
    fn solve_normal_equation(
        &mut self, 
        residuals: &Mat<f64>, 
        jacobians: &SparseColMat<usize, f64>
    ) -> LinAlgResult<Mat<f64>>;
    
    /// Solve (J^T J + λI) x = -J^T r
    fn solve_augmented_equation(
        &mut self, 
        residuals: &Mat<f64>, 
        jacobians: &SparseColMat<usize, f64>, 
        lambda: f64
    ) -> LinAlgResult<Mat<f64>>;
    
    /// Cached Hessian (J^T J)
    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>>;
    
    /// Cached gradient (-J^T r)
    fn get_gradient(&self) -> Option<&Mat<f64>>;
    
    /// Compute covariance matrix (H^-1)
    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>>;
    
    /// Get cached covariance
    fn get_covariance_matrix(&self) -> Option<&Mat<f64>>;
}
```

### SparseCholeskySolver

**Location:** `src/linalg/cholesky.rs`

#### Structure

```rust
pub struct SparseCholeskySolver {
    factorizer: Option<Llt<usize, f64>>,
    symbolic_factorization: Option<SymbolicLlt<usize>>,
    hessian: Option<SparseColMat<usize, f64>>,
    gradient: Option<Mat<f64>>,
    covariance_matrix: Option<Mat<f64>>,
    standard_errors: Option<Mat<f64>>,
}
```

#### Symbolic Factorization Caching

**Location:** Lines 79-117

```rust
fn solve_normal_equation(&mut self, residuals: &Mat<f64>, 
    jacobians: &SparseColMat<usize, f64>) -> LinAlgResult<Mat<f64>> 
{
    // Form H = J^T J
    let jt = jacobians.transpose();
    let hessian = jt.to_col_major() * jacobians;
    
    // Reuse symbolic factorization if available
    let symbolic = match &self.symbolic_factorization {
        Some(s) => s.clone(),  // Cheap clone (reference-counted)
        None => {
            let s = hessian.symbolic_llt(Side::Lower, Default::default())?;
            self.symbolic_factorization = Some(s.clone());
            s
        }
    };
    
    // Numeric factorization
    let numeric = symbolic.factorize_numeric(&hessian)?;
    
    // Solve
    let gradient = -(&jt.to_col_major() * residuals);
    let solution = numeric.solve(&gradient);
    
    // Cache results
    self.hessian = Some(hessian);
    self.gradient = Some(gradient);
    
    Ok(solution)
}
```

#### Augmented Equation (LM)

**Location:** Lines 119-175

```rust
fn solve_augmented_equation(&mut self, residuals: &Mat<f64>, 
    jacobians: &SparseColMat<usize, f64>, lambda: f64) -> LinAlgResult<Mat<f64>> 
{
    // Form H = J^T J
    let jt = jacobians.transpose();
    let hessian = jt.to_col_major() * jacobians;
    
    // Create λI as sparse matrix
    let n = hessian.nrows();
    let triplets: Vec<_> = (0..n)
        .map(|i| Triplet::new(i, i, lambda))
        .collect();
    let lambda_i = SparseColMat::from_triplets(n, n, &triplets);
    
    // H + λI
    let augmented = &hessian + &lambda_i;
    
    // Factorize and solve
    // ... similar to normal equation
}
```

#### Covariance Computation

**Location:** Lines 207-224

```rust
fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>> {
    let hessian = self.hessian.as_ref()?;
    
    // Solve H * X = I to get H^-1
    let n = hessian.nrows();
    let identity = Mat::identity(n, n);
    
    // Reuse cached factorization
    let symbolic = self.symbolic_factorization.as_ref()?;
    let numeric = symbolic.factorize_numeric(hessian).ok()?;
    
    let covariance = numeric.solve(&identity);
    self.covariance_matrix = Some(covariance);
    
    // Standard errors = sqrt(diag(cov))
    self.standard_errors = Some(compute_std_errors(&covariance));
    
    self.covariance_matrix.as_ref()
}
```

### SparseQRSolver

**Location:** `src/linalg/qr.rs`

Nearly identical structure to Cholesky with:
- `Qr<usize, f64>` instead of `Llt<usize, f64>`
- `SymbolicQr<usize>` instead of `SymbolicLlt<usize>`

#### Key Differences

1. **Rank-Deficient Handling**: QR is more robust for ill-conditioned systems
2. **Slightly Slower**: More operations than Cholesky
3. **Same Caching Strategy**: Symbolic pattern reuse

### Faer Library Usage

The module uses `faer` for sparse linear algebra:

```rust
use faer::sparse::{SparseColMat, SymbolicLlt, SymbolicQr};
use faer::linalg::solvers::{Solve, Llt, Qr};
use faer::{Mat, Side};
```

**Key faer features used:**
- `SparseColMat` - Sparse column-major matrices
- `.transpose()` - Zero-copy transpose
- `.to_col_major()` - Format conversion
- `SymbolicLlt/Qr` - Separated symbolic/numeric factorization
- `Solve` trait - Generic solving interface

### Error Handling

**Location:** `src/linalg/mod.rs:29-51`

```rust
#[derive(Debug, Clone, Error)]
pub enum LinAlgError {
    #[error("Factorization failed: {0}")]
    FactorizationFailed(String),
    
    #[error("Singular matrix")]
    SingularMatrix,
    
    #[error("Sparse matrix creation failed: {0}")]
    SparseMatrixCreation(String),
    
    #[error("Matrix conversion failed: {0}")]
    MatrixConversion(String),
}
```

### Performance Analysis

| Operation | Bottleneck Level | Notes |
|-----------|-----------------|-------|
| J^T J multiplication | **HIGH** (40-60%) | Sparse matrix multiply |
| Cholesky factorization | **MEDIUM** (20-30%) | Per-iteration |
| Symbolic factorization | **LOW** (cached) | Once per problem |
| Solve step | **LOW** | Fast back-substitution |

---

## Part 2: Factors Module

### Overview

The `factors/` module provides constraint implementations for optimization.

**Files:**
- `mod.rs` - `Factor` trait (~133 LOC)
- `prior_factor.rs` - Unary priors (~103 LOC)
- `se2_factor.rs` - SE2 between factors (~166 LOC)
- `se3_factor.rs` - SE3 between factors (~166 LOC)
- `double_sphere_factor.rs` - Double Sphere camera (~380 LOC)
- `eucm_factor.rs` - EUCM camera (~290 LOC)
- `fov_factor.rs` - FOV camera (~320 LOC)
- `kannala_brandt_factor.rs` - Kannala-Brandt camera (~480 LOC)
- `rad_tan_factor.rs` - RadTan camera (~480 LOC)
- `ucm_factor.rs` - UCM camera (~280 LOC)

### Factor Trait

**Location:** `src/factors/mod.rs:47-133`

```rust
pub trait Factor: Send + Sync {
    /// Linearize the factor at given parameter values
    /// 
    /// # Arguments
    /// * `params` - Parameter vectors for each connected variable
    /// * `compute_jacobian` - Whether to compute Jacobian
    /// 
    /// # Returns
    /// (residual, optional_jacobian)
    fn linearize(&self, params: &[DVector<f64>], compute_jacobian: bool) 
        -> (DVector<f64>, Option<DMatrix<f64>>);
    
    /// Get residual dimension
    fn get_dimension(&self) -> usize;
}
```

**Design Notes:**
- `Send + Sync` for parallel evaluation
- Generic over number of connected variables
- Optional Jacobian for cost-only iterations

### Prior Factor

**Location:** `src/factors/prior_factor.rs`

```rust
pub struct PriorFactor {
    pub data: DVector<f64>,
}

impl Factor for PriorFactor {
    fn linearize(&self, params: &[DVector<f64>], compute_jacobian: bool) 
        -> (DVector<f64>, Option<DMatrix<f64>>) 
    {
        let residual = &params[0] - &self.data;
        
        let jacobian = if compute_jacobian {
            Some(DMatrix::identity(self.data.len(), self.data.len()))
        } else {
            None
        };
        
        (residual, jacobian)
    }
    
    fn get_dimension(&self) -> usize {
        self.data.len()
    }
}
```

**Use Cases:**
- Anchoring variables
- GPS measurements
- Regularization

### Pose Factors (SE2, SE3)

#### BetweenFactorSE2

**Location:** `src/factors/se2_factor.rs`

```rust
pub struct BetweenFactorSE2 {
    pub relative_pose: SE2,
}

impl BetweenFactorSE2 {
    pub fn new(dx: f64, dy: f64, dtheta: f64) -> Self {
        Self {
            relative_pose: SE2::new(dx, dy, dtheta),
        }
    }
}
```

**Linearization Algorithm** (lines 100-166):

```rust
impl Factor for BetweenFactorSE2 {
    fn linearize(&self, params: &[DVector<f64>], compute_jacobian: bool) 
        -> (DVector<f64>, Option<DMatrix<f64>>) 
    {
        // 1. Convert params to SE2
        let se2_k0 = SE2::from_vector(&params[0]);
        let se2_k1 = SE2::from_vector(&params[1]);
        
        // 2. Compute residual: log(T_meas^-1 * T_0^-1 * T_1)
        let mut j_inv = Matrix3::zeros();
        let se2_k1_inv = se2_k1.inverse(Some(&mut j_inv));
        
        let mut j_compose1_k1_inv = Matrix3::zeros();
        let mut j_compose1_k0 = Matrix3::zeros();
        let se2_temp = se2_k1_inv.compose(&se2_k0, 
            Some(&mut j_compose1_k1_inv), Some(&mut j_compose1_k0));
        
        let mut j_compose2_temp = Matrix3::zeros();
        let se2_diff = se2_temp.compose(&self.relative_pose, 
            Some(&mut j_compose2_temp), None);
        
        let mut j_log = Matrix3::zeros();
        let residual = se2_diff.log(Some(&mut j_log));
        
        if compute_jacobian {
            // Chain rule for Jacobian
            let j_k0 = j_log * j_compose2_temp * j_compose1_k0;
            let j_k1 = j_log * j_compose2_temp * j_compose1_k1_inv * j_inv;
            
            // Assemble 3x6 Jacobian
            let mut jacobian = DMatrix::zeros(3, 6);
            jacobian.fixed_view_mut::<3, 3>(0, 0).copy_from(&j_k0);
            jacobian.fixed_view_mut::<3, 3>(0, 3).copy_from(&j_k1);
            
            (residual.to_vector(), Some(jacobian))
        } else {
            (residual.to_vector(), None)
        }
    }
}
```

#### BetweenFactorSE3

Nearly identical structure with 6x12 Jacobians instead of 3x6.

### Camera Model Factors

All camera factors follow a common pattern with two factor types:

1. **CameraParamsFactor** - Optimizes camera intrinsics (fixed 3D points)
2. **ProjectionFactor** - Optimizes 3D points (fixed intrinsics)

#### Camera Model Comparison

| Model | Parameters | Complexity | Use Case |
|-------|------------|------------|----------|
| Double Sphere | 6 (fx,fy,cx,cy,α,ξ) | High | Fisheye, wide-angle |
| EUCM | 6 (fx,fy,cx,cy,α,β) | Medium | Fisheye, unified |
| FOV | 5 (fx,fy,cx,cy,w) | Medium | Fisheye distortion |
| Kannala-Brandt | 8 (fx,fy,cx,cy,k1-k4) | Very High | High-order fisheye |
| RadTan | 9 (fx,fy,cx,cy,k1-k3,p1,p2) | Very High | OpenCV standard |
| UCM | 5 (fx,fy,cx,cy,α) | Low | Minimal distortion |

#### Example: Double Sphere

**Location:** `src/factors/double_sphere_factor.rs`

```rust
pub struct DoubleSphereCameraParamsFactor {
    pub points_3d: Vec<Vector3<f64>>,
    pub observations: Vec<Vector2<f64>>,
}

pub struct DoubleSphereProjectionFactor {
    pub camera_params: DVector<f64>,  // [fx, fy, cx, cy, alpha, xi]
    pub observations: Vec<Vector2<f64>>,
}
```

**Projection Model** (lines 14-60):

```rust
fn compute_residual_double_sphere(
    point: &Vector3<f64>,
    observation: &Vector2<f64>,
    params: &DVector<f64>,
) -> Vector2<f64> {
    let fx = params[0];
    let fy = params[1];
    let cx = params[2];
    let cy = params[3];
    let alpha = params[4];
    let xi = params[5];
    
    let x = point.x;
    let y = point.y;
    let z = point.z;
    
    let r2 = x * x + y * y;
    let d1 = (r2 + z * z).sqrt();
    let gamma = xi * d1 + z;
    let d2 = (r2 + gamma * gamma).sqrt();
    
    let denom = alpha * d2 + (1.0 - alpha) * gamma;
    
    if denom.abs() < 1e-10 {
        return Vector2::new(1e6, 1e6);  // Invalid projection
    }
    
    let u = fx * x / denom + cx;
    let v = fy * y / denom + cy;
    
    Vector2::new(u - observation.x, v - observation.y)
}
```

**Jacobian Computation** (lines 96-170):

Analytical Jacobians are computed for both:
- Camera parameters: `∂residual/∂[fx,fy,cx,cy,α,ξ]`
- 3D points: `∂residual/∂[x,y,z]`

### Test Coverage Analysis

| Factor | Test Count | Coverage Level |
|--------|------------|----------------|
| Prior | 2 | Basic |
| SE2 | 3 | Basic |
| SE3 | 3 | Basic |
| Double Sphere | 3 | Basic |
| EUCM | 3 | Basic |
| FOV | 3 | Basic |
| Kannala-Brandt | 9 | **Excellent** |
| RadTan | 6 | Good |
| UCM | 6 | Good |

**Observation:** Kannala-Brandt has best test coverage with:
- Dimension verification
- Non-zero Jacobian validation
- Jacobian structure tests
- Panic tests for mismatched inputs

### Code Quality Issues

#### 1. Inconsistent Precision Constants

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

**Recommendation:** Centralize thresholds in `factors/mod.rs`.

#### 2. Magic Return Values

```rust
// Invalid projection handling
if denom.abs() < 1e-10 {
    return Vector2::new(1e6, 1e6);  // Magic value
}
```

**Recommendation:** Use `Result<Vector2<f64>, FactorError>` or `Option`.

#### 3. Inlined Jacobian Computation

All camera models compute Jacobians inline, leading to:
- Code duplication
- Hard-to-verify derivatives
- Maintenance burden

**Recommendation:** Extract common patterns or use automatic differentiation.

### Dynamic Dispatch Analysis

#### Current Design

```rust
// In ResidualBlock
pub factor: Box<dyn Factor + Send>,
```

#### Static Dispatch Alternative

```rust
pub enum FactorEnum {
    Prior(PriorFactor),
    SE2Between(BetweenFactorSE2),
    SE3Between(BetweenFactorSE3),
    DoubleSphereCamera(DoubleSphereCameraParamsFactor),
    DoubleSphereProjection(DoubleSphereProjectionFactor),
    // ... 10+ more variants
}
```

**Trade-offs:**

| Aspect | Box<dyn Factor> | FactorEnum |
|--------|-----------------|------------|
| Flexibility | High (custom factors) | Low (closed set) |
| Performance | vtable lookup | Direct call |
| Binary size | Smaller | Larger |
| Maintenance | Easy | 10+ match arms |

**Verdict:** Keep `Box<dyn Factor>` - flexibility outweighs ~1% performance gain.

### Recommendations

1. **Centralize Precision Constants**
   ```rust
   // factors/mod.rs
   pub const PROJECTION_EPSILON: f64 = 1e-10;
   pub const SMALL_ANGLE_THRESHOLD: f64 = 1e-7;
   ```

2. **Improve Test Coverage**
   - Match Kannala-Brandt level for all camera models
   - Add numerical Jacobian verification

3. **Consider Camera Model Trait**
   ```rust
   pub trait CameraModel {
       fn project(&self, point: &Vector3<f64>) -> Option<Vector2<f64>>;
       fn project_with_jacobian(&self, point: &Vector3<f64>) 
           -> Option<(Vector2<f64>, Matrix2x3<f64>)>;
       fn unproject(&self, pixel: &Vector2<f64>) -> Vector3<f64>;
   }
   ```

4. **Document Projection Models**
   - Add references to papers
   - Include mathematical formulas in docs
