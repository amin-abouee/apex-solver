# Sparse QR Decomposition

A more robust linear solver for rank-deficient or ill-conditioned systems.

## Mathematical Background

### QR Factorization

For a matrix \\(A\\), QR decomposition finds:

$$
A = QR
$$

where:
- \\(Q\\): Orthogonal matrix (\\(Q^T Q = I\\))
- \\(R\\): Upper triangular matrix

### Solving Linear Systems

To solve \\(H \cdot \Delta x = -g\\) where \\(H = J^T J\\):

1. Factorize: \\(H = QR\\)
2. Compute: \\(y = Q^T (-g)\\)
3. Back substitution: \\(R \Delta x = y\\)

### Normal Equations

In nonlinear least squares:

$$
J^T J \cdot \Delta x = -J^T r
$$

QR can also solve the original least squares problem directly:

$$
\min_{\Delta x} \|J \Delta x + r\|^2
$$

by factorizing \\(J = QR\\) and solving \\(R \Delta x = -Q^T r\\).

### Augmented Equations (Levenberg-Marquardt)

For LM damping:

$$
(J^T J + \lambda I) \Delta x = -J^T r
$$

QR handles the augmented system with the same sparsity pattern since \\(\lambda I\\) only modifies the diagonal.

## Covariance Computation

The parameter covariance matrix is:

$$
\Sigma = H^{-1} = (J^T J)^{-1}
$$

QR provides this through:

$$
H^{-1} = R^{-1} R^{-T}
$$

## Usage

```rust
use apex_solver::linalg::LinearSolverType;
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;

let config = LevenbergMarquardtConfig::new()
    .with_linear_solver_type(LinearSolverType::SparseQR);
```

## Implementation Details

### Sparse Matrix Storage

Uses the `faer` library for sparse QR decomposition:
- Compressed Sparse Column (CSC) format
- Column pivoting for numerical stability

### Symbolic Factorization Caching

Like Cholesky, the sparsity pattern is analyzed once:

1. **First iteration**: Compute symbolic factorization
2. **Subsequent iterations**: Reuse symbolic structure (10-15% speedup)

### Numerical Advantages

QR is more robust because:
- **No squaring**: Avoids condition number squaring (\\(\kappa(J^T J) = \kappa(J)^2\\))
- **Orthogonal transformation**: Preserves numerical precision
- **Rank revealing**: Can detect rank deficiency with pivoting

## Advantages

- **More robust** than Cholesky for ill-conditioned problems
- **Handles rank deficiency** better
- **Better numerical stability** for problematic Jacobians
- **No positive definiteness requirement**

## Limitations

- **Slightly slower** than Cholesky (typically 20-50% more time)
- **More memory** for storing \\(Q\\) and \\(R\\)
- **Overkill** for well-conditioned problems

## When to Use

**Use QR when:**
- System may be rank-deficient
- Jacobian is ill-conditioned
- Numerical robustness is critical
- Cholesky fails or gives poor results

**Use [Cholesky](./cholesky.md) instead when:**
- Problem is well-conditioned
- Speed is important
- Memory is constrained

## Comparison with Cholesky

| Aspect | Cholesky | QR |
|--------|----------|-----|
| Speed | Faster | Slower |
| Robustness | Good | Better |
| Rank deficiency | Fails | Handles |
| Condition sensitivity | \\(\kappa^2\\) | \\(\kappa\\) |
| Memory | Less | More |
| Default | Yes | No |

## Condition Number

For a matrix \\(A\\), the condition number is:

$$
\kappa(A) = \|A\| \cdot \|A^{-1}\|
$$

When solving \\(H \cdot \Delta x = -g\\) with \\(H = J^T J\\):

- **Cholesky** sees: \\(\kappa(J^T J) = \kappa(J)^2\\)
- **QR** sees: \\(\kappa(J)\\)

If \\(\kappa(J) = 10^6\\), then:
- Cholesky works with \\(\kappa = 10^{12}\\) (potentially problematic)
- QR works with \\(\kappa = 10^6\\) (more manageable)

## Performance

For typical problems, QR is 20-50% slower than Cholesky:

| Problem Size | Cholesky | QR | Ratio |
|--------------|----------|-----|-------|
| 100 poses | ~1ms | ~1.5ms | 1.5x |
| 1000 poses | ~10ms | ~15ms | 1.5x |
| 10000 poses | ~100ms | ~140ms | 1.4x |

The extra cost is usually acceptable when robustness is needed.
