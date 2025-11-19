# Sparse Cholesky Decomposition

The default and fastest linear solver for positive definite systems.

## Mathematical Background

### Cholesky Factorization

For a symmetric positive definite matrix \\(H\\), Cholesky decomposition finds a lower triangular matrix \\(L\\) such that:

$$
H = L L^T
$$

### Solving Linear Systems

To solve \\(H \cdot \Delta x = -g\\):

1. Factorize: \\(H = L L^T\\)
2. Forward substitution: \\(L y = -g\\)
3. Back substitution: \\(L^T \Delta x = y\\)

### Normal Equations

In nonlinear least squares, we solve:

$$
J^T J \cdot \Delta x = -J^T r
$$

where:
- \\(H = J^T J\\) is the Hessian (Gauss-Newton approximation)
- \\(g = J^T r\\) is the gradient
- \\(J\\) is the \\(m \times n\\) Jacobian
- \\(r\\) is the \\(m \times 1\\) residual vector

### Augmented Equations (Levenberg-Marquardt)

For LM damping, solve:

$$
(J^T J + \lambda I) \Delta x = -J^T r
$$

Adding \\(\lambda I\\) to the diagonal preserves positive definiteness and improves conditioning.

## Covariance Computation

The parameter covariance matrix is:

$$
\Sigma = H^{-1} = (J^T J)^{-1}
$$

This gives uncertainty estimates for each parameter. Standard errors are the square roots of the diagonal:

$$
\sigma_i = \sqrt{\Sigma_{ii}}
$$

## Usage

```rust
use apex_solver::linalg::LinearSolverType;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;

// Cholesky is the default
let config = LevenbergMarquardtConfig::new()
    .with_linear_solver_type(LinearSolverType::SparseCholesky);
```

## Implementation Details

### Sparse Matrix Storage

Uses the `faer` library for sparse LLT decomposition:
- Compressed Sparse Column (CSC) format
- Efficient for sparse Jacobians from factor graphs

### Symbolic Factorization Caching

The sparsity pattern is analyzed once and reused:

1. **First iteration**: Compute symbolic factorization (expensive)
2. **Subsequent iterations**: Reuse symbolic structure (10-15% speedup)

This works because the Jacobian sparsity pattern doesn't change—only values change.

### Numerical Stability

Cholesky requires \\(H\\) to be:
- **Symmetric**: Guaranteed by \\(H = J^T J\\)
- **Positive definite**: Usually satisfied, but can fail with:
  - Rank-deficient Jacobians
  - Numerical errors
  - Very ill-conditioned problems

## Advantages

- **Fast**: \\(O(n^3/3)\\) for dense, much better for sparse
- **Numerically stable** for well-conditioned problems
- **Memory efficient**: Only stores \\(L\\), not full \\(H\\)
- **Natural for least squares**: \\(J^T J\\) is always symmetric

## Limitations

- **Requires positive definite**: Fails on singular systems
- **Less robust** than QR for ill-conditioned problems
- **No rank detection**: Doesn't identify rank deficiency

## When to Use

**Use Cholesky when:**
- Problem is well-conditioned
- Speed is important
- Normal equations are appropriate

**Use [QR](./qr.md) instead when:**
- System may be rank-deficient
- Numerical robustness is critical
- Ill-conditioned Jacobian expected

## Performance

For typical SLAM/bundle adjustment problems:

| Problem Size | Cholesky Time | Notes |
|--------------|---------------|-------|
| 100 poses | ~1ms | Very fast |
| 1000 poses | ~10ms | Still fast |
| 10000 poses | ~100ms | Sparse structure helps |

The sparse structure of factor graphs (local connectivity) makes Cholesky very efficient.
