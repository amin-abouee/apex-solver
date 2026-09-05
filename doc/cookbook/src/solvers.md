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

Bundle adjustment decomposes into retained ("kept") and eliminated blocks —
classically cameras and landmarks, but the partition works for any DOF and
any manifold type (3-D points, inverse-depth landmarks, marginalized poses):

$$
\mathbf{S} = \mathbf{H}_{cc} - \mathbf{H}_{cp}\,\mathbf{H}_{pp}^{-1}\,\mathbf{H}_{pc}
$$

- **`ExplicitSparseSchur`** (`LinearSolverType::ExplicitSparseSchur`) — forms
  `S` explicitly over a sparse Hessian. `ExplicitSchurVariant` selects how it
  is then solved: `Sparse` (Cholesky, default), `Iterative` (PCG on the
  formed `S`), or `Chunked` (built chunk-by-chunk straight from `J`, never
  materializing `JᵀJ` — Ceres's `SchurEliminator` strategy). Best when the
  camera count is moderate. Equivalent to Ceres's `SPARSE_SCHUR`.
- **`ExplicitDenseSchur`** (`LinearSolverType::ExplicitDenseSchur`) — the same
  explicit construction over a dense Hessian, for small-to-medium problems
  (`JacobianMode::Dense`). Equivalent to Ceres's `DENSE_SCHUR`.
- **`ImplicitSparseSchur`** (`LinearSolverType::ImplicitSparseSchur`) — forms
  neither `S` nor `JᵀJ`; applies the reduced operator matrix-free from `J`
  inside preconditioned conjugate gradients, so its cost scales with `nnz(J)`.
  `SchurPreconditioner` selects `None` (Ceres `IDENTITY`), `BlockDiagonal`
  (Ceres `JACOBI`), or `SchurJacobi` (Ceres `SCHUR_JACOBI`; the default, and
  usually the best convergence). The choice for large BA (10,000+ cameras).
  Equivalent to Ceres's `ITERATIVE_SCHUR`.

All three support the `StructureAware::initialize_structure` step that
partitions variables into the kept and eliminated sets (manual marks via
`Problem::mark_for_elimination`, or opt-in auto-detection via
`SchurOrdering::with_auto_detect` — see [Problem Construction](./problem.md)).

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
