# 1. Current Linear Algebra Architecture

## 1.1 Trait Hierarchy

The linear algebra subsystem lives in `src/linalg/` and is built around two traits:

### Primary Trait: `SparseLinearSolver`

**File**: `src/linalg/mod.rs:184-229`

```rust
pub trait SparseLinearSolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>>;

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>>;

    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>>;
    fn get_gradient(&self) -> Option<&Mat<f64>>;
    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>>;
    fn get_covariance_matrix(&self) -> Option<&Mat<f64>>;
}
```

**Key observation**: The trait is tightly coupled to `SparseColMat<usize, f64>` in both inputs (Jacobian) and cached outputs (Hessian). This is the central tension for adding dense/GPU backends.

### Secondary Trait: `StructuredSparseLinearSolver`

**File**: `src/linalg/mod.rs:148-182`

Used exclusively by Schur complement solvers. Requires `initialize_structure()` with variable partition info. Not affected by dense/GPU additions.

## 1.2 Concrete Implementations

### SparseCholeskySolver (`src/linalg/cholesky.rs`)

```
Input:  SparseColMat (Jacobian J)
Step 1: H = J^T * J  (sparse × sparse → sparse)
Step 2: g = J^T * r  (sparse^T × dense → dense)
Step 3: Symbolic Cholesky on sparsity pattern (cached across iterations)
Step 4: Numeric Cholesky: L*L^T = H (or H + λI for augmented)
Step 5: Solve L*L^T * dx = -g
Output: Mat<f64> (dense step vector dx)
```

Caches: `hessian: Option<SparseColMat>`, `gradient: Option<Mat>`, `symbolic_factorization: Option<SymbolicLlt>`

### SparseQRSolver (`src/linalg/qr.rs`)

Same interface, uses `faer::sparse::linalg::solvers::Qr` and `SymbolicQr`. More robust for rank-deficient, slightly slower.

### SchurSolverAdapter (`src/linalg/explicit_schur.rs`)

Wraps `SparseSchurComplementSolver` to implement `SparseLinearSolver`. Partitions variables into camera/landmark blocks, eliminates landmarks via Schur complement.

### IterativeSchurSolver (`src/linalg/implicit_schur.rs`)

Matrix-free PCG solver. Never forms the Schur complement explicitly.

## 1.3 Configuration & Factory

**Enum** (`src/linalg/mod.rs:16-22`):
```rust
pub enum LinearSolverType {
    #[default]
    SparseCholesky,
    SparseQR,
    SparseSchurComplement,
}
```

**Factory** (`src/optimizer/mod.rs:696-705`):
```rust
pub fn create_linear_solver(solver_type: &LinearSolverType) -> Box<dyn SparseLinearSolver> {
    match solver_type {
        SparseCholesky => Box::new(SparseCholeskySolver::new()),
        SparseQR => Box::new(SparseQRSolver::new()),
        SparseSchurComplement => Box::new(SparseCholeskySolver::new()), // fallback
    }
}
```

LM has its own inline creation (`src/optimizer/levenberg_marquardt.rs:833-851`) to handle Schur adapter initialization.

## 1.4 How Optimizers Consume Linear Solvers

All three optimizers (LM, GN, DogLeg) store `Box<dyn SparseLinearSolver>`.

### Levenberg-Marquardt
- Calls `solve_augmented_equation(residuals, jacobians, lambda)`
- Uses `get_gradient()` for convergence checks and predicted reduction
- Uses `get_hessian()` but currently discards it (`let _hessian = ...`)

### Gauss-Newton
- Calls `solve_normal_equation(residuals, jacobians)`
- Uses `get_gradient()` and `get_hessian()` via `notify_observers()`

### Dog Leg (**CRITICAL**)
- Calls `solve_normal_equation(residuals, jacobians)` (or augmented with mu)
- **Directly uses sparse Hessian** in `compute_predicted_reduction()`:
  ```rust
  fn compute_predicted_reduction(
      &self,
      step: &Mat<f64>,
      gradient: &Mat<f64>,
      hessian: &SparseColMat<usize, f64>,  // ← sparse type!
  ) -> f64 {
      let hessian_step = hessian * step;  // sparse × dense multiply
      // ...
  }
  ```
- This means `get_hessian()` **must return Some** for DogLeg to work.

### Observer System (`src/optimizer/mod.rs:724-729`)
```rust
if let (Some(hessian), Some(gradient)) =
    (linear_solver.get_hessian(), linear_solver.get_gradient())
{
    observers.set_matrix_data(Some(hessian.clone()), Some(gradient.clone()));
}
```
Gracefully handles `None` via `if let`.

## 1.5 Data Flow Summary

```
Problem::compute_residual_and_jacobian_sparse()
    ↓
(Mat<f64>, SparseColMat<usize, f64>)  ← residual + sparse Jacobian
    ↓
SparseLinearSolver::solve_*_equation(residuals, jacobians, [lambda])
    ↓
Mat<f64>  ← dense step vector dx
    ↓
VariableEnum::plus(dx)  ← manifold update
```

The Problem **always produces sparse Jacobians**. The step vector is always dense. This asymmetry is key: dense/GPU solvers need to handle sparse input but produce dense output.

## 1.6 Critical Constraints for New Backends

1. **DogLeg requires sparse Hessian** from `get_hessian()` — dense solvers must store a sparse copy
2. **Observers clone the sparse Hessian** — gracefully handles None, but loses functionality
3. **Covariance computation** uses the Cholesky factorizer — dense solvers need their own path
4. **All optimizers use `Box<dyn SparseLinearSolver>`** — new solvers must implement this trait
5. **Problem only produces `SparseColMat` Jacobians** — conversion to dense is solver's responsibility
