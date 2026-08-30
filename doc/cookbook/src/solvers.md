# Linear Solvers

The normal equations `H = JᵀJ` are formed once per iteration with faer's
parallel sparse kernels; the sparsity pattern of `Jᵀ`, of `H` and the value
permutation linking them are cached across iterations, so each evaluation is a
parallel value gather plus a parallel sparse product.

## Sparse Cholesky

`SparseCholeskySolver` — `LDLᵀ`/`LLᵀ` factorization of `H` (or `H + λD` under
LM). The symbolic factorization is cached; only the numeric pass repeats.
The default for pose graphs.

## Sparse QR

`SparseQRSolver` — factorizes `J` directly, more numerically robust on
ill-conditioned problems at a higher cost per iteration.

## Schur complement

Bundle adjustment decomposes into camera and landmark blocks:

$$
\mathbf{S} = \mathbf{H}_{cc} - \mathbf{H}_{cp}\,\mathbf{H}_{pp}^{-1}\,\mathbf{H}_{pc}
$$

- **`SparseSchurComplementSolver`** — forms `S` explicitly and Cholesky-factors
  it. Best when the camera count is moderate.
- **`IterativeSchurSolver`** — never forms `S`; applies it matrix-free inside
  preconditioned conjugate gradients with a block-Jacobi (Schur–Jacobi)
  preconditioner. The choice for large BA (10,000+ cameras).

Both variants support the `StructureAware::initialize_structure` step that
partitions variables into cameras and landmarks (manual marks via
`Problem::mark_as_schur_landmark`, or opt-in auto-detection — see
[Problem Construction](./problem.md)).

## Covariance estimation

`Covariance::compute` re-linearizes the problem at a point, forms a clean
`H = JᵀJ` and inverts via sparse Cholesky or dense SVD:

```rust
use apex_solver::linalg::covariance::{Covariance, CovarianceAlgorithm, CovarianceOptions};

let cov = Covariance::compute(
    CovarianceOptions::new(CovarianceAlgorithm::SparseCholesky),
    &problem,
    &variables,
)?;
let block = cov.block(key); // dof × dof marginal in tangent space
```

The covariance is a property of the problem at a point — it never consults the
optimizer's last linear system.
