# Sparse Linear Solvers

Most factor-graph problems are large and very sparse: a window with a hundred
poses and a thousand landmarks has a Jacobian with a few thousand non-zeros in
a matrix a thousand times wider. The sparse path keeps that structure all the
way to the factorization. This chapter is the crate's sparse machinery: the
solvers, the pattern bookkeeping, and the requirement that makes Schur
elimination possible at all.

## 1. The sparse path

`JacobianMode::Sparse` assembles into `faer::SparseColMat` using a
**symbolic structure** computed once from the block layout. The structure is
the pattern of non-zeros; the numerics are refilled each iteration. Keeping
them apart is what makes a per-iteration rebuild affordable.

The solvers:

| Solver | Notes |
|---|---|
| `SparseCholeskySolver` | the default; needs $H$ positive definite |
| `SparseQRSolver` | least-squares directly, no squaring of the condition number |
| `ExplicitSparseSchur` | eliminates the marked variables, then Cholesky (or PCG) on the reduced system |
| `ImplicitSparseSchur` | matrix-free Schur with PCG; never forms $S$ or $J^\top J$ |

## 2. The pattern

`PatternFingerprint` is the crate's handle on the sparsity structure: a
fingerprint of the pattern, used to decide whether a cached structure is still
valid. The structure itself is `SymbolicStructure`, re-exported from
`core::problem` so a consumer can inspect or persist it.

The pattern is derived from the block layout at registration: which variables
each residual block touches, and therefore which (block, variable) pairs
exist. A block's *rows* always exist; which columns are non-zero depends on
the factor's Jacobian, and the crate's structure is conservative about that —
it reserves the columns the factor declares in `jacobian_shape`.

## 3. Normal equations versus least squares

The sparse Cholesky solves $H\delta = -g$ with $H = J^\top J$. It is the
fastest option and the most fragile: $H$ squares the condition number, so a
problem that is ill-conditioned in $J$ can be numerically singular in $H$. The
QR solver works on $J$ directly and does not have that weakness, at the cost of
a larger factorization.

The choice is the same one the dense path makes, and the same advice applies:
if a sparse Cholesky fails on a problem that should be fine, try the QR.

## 4. The Schur requirement

Eliminating a set of variables is only possible when the eliminated block is
**block-diagonal** in the normal equations: no two eliminated variables may
share a residual row. The crate enforces this structurally:

- `ChunkLayout` rejects two eliminated variables that share a residual row;
- `SchurPartition::verify_block_diagonal` checks the assembled Hessian
  directly and returns a `LinAlgResult` if the condition is violated.

The consequence is a hard design constraint on the *problems*, not on the
solver: a visual-inertial window eliminates $\{state_0, bias_0, \text{hosted
landmarks}\}$, and `CombinedImuFactor` puts `state_0` and `bias_0` in the
*same fifteen rows*. The eliminated block is therefore not block-diagonal, and
the Schur machinery does not apply — which is exactly why the crate has a
separate, dense marginalization module for that case (Part IV).

<div class="apx-note">
<span class="apx-title">Note — the two Schur flavours</span>

`ExplicitSparseSchur` forms $S$ and factorizes it: most accurate and fastest
per iteration, but $S$ is accumulated into a dense `kept_dof²` buffer, so
memory grows quadratically with the retained set. `ImplicitSparseSchur` never
forms $S$ or $J^\top J$: the operator reads $J$ directly and the cost scales
with $\text{nnz}(J)$, solved by PCG with a preconditioner
(`SchurPreconditioner::{None, BlockDiagonal, SchurJacobi}`, the last being
what Ceres uses). The trade is memory against iteration count and accuracy.

</div>

## 5. Worked example

A window with 7 keyframes, 200 landmarks, 800 residual rows, 3 405 non-zeros.
The sparse Jacobian is $800\times305$ with 3 405 stored values — about 26 kB.
A sparse Cholesky of the $305\times305$ normal matrix touches a few hundred
thousand non-zeros; the dense equivalent would be $800\times305$ doubles
(1.9 MB) and a $305^3/3$ factorization (9.5 Mflops) — two orders of magnitude
more arithmetic for a problem that is 99 % empty.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a stale pattern is a wrong answer, not a crash</span>

The symbolic structure is computed at registration. If the graph is mutated
without telling it — a block added by a path that bypasses the problem's own
methods, a variable removed directly — the numeric values are written into the
wrong slots, and the solve converges to a wrong answer with no error. The
problem's own mutators (`add_residual_block`, `remove_residual_block`,
`try_remove_variable`) keep the structure in step; a consumer that mutates
the graph behind its back has to keep the structure in step itself.

</div>

<div class="apx-note">
<span class="apx-title">Note — sparse is not automatically faster</span>

For a few hundred unknowns the sparse path's bookkeeping costs more than the
dense factorization saves. The crate's guidance is a size threshold (about 500
DoF), and the honest advice is to measure: the benchmark suite exists precisely
because the crossover depends on the problem's shape, not only its size.

</div>
