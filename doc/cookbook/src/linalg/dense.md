# Dense Linear Solvers

A dense problem is one the solver can hold in a single matrix: a few hundred
unknowns, a handful of residual blocks, no structure worth exploiting. The
crate's dense path is deliberately simple — Cholesky, QR, and a dense Schur —
and this chapter is what each one assumes, what it costs, and when the
assembly mode itself is the wrong choice.

## 1. The dense path

`JacobianMode::Dense` builds the Jacobian as a `faer::Mat<f64>` in full. The
doc comment's guidance: "Best for small-to-medium problems (< ~500 DOF)". The
linear solvers available on it:

| Solver | Assumes | Notes |
|---|---|---|
| `DenseCholeskySolver` | $H = J^\top J$ is symmetric positive definite | the fast path; fails on a semi-definite $H$ |
| `DenseQRSolver` | nothing | more robust, more expensive; solves the least-squares problem directly |
| `ExplicitDenseSchur` | an eliminable subset exists | Ceres' `DENSE_SCHUR` equivalent |

## 2. Cholesky

The normal equations $H\delta = -g$ are symmetric by construction, so a
Cholesky factorization $H = LL^\top$ costs $\tfrac13 n^3$ flops and yields
$\delta = -L^{-T}L^{-1}g$. It requires $H$ to be positive definite, which a
least-squares problem's $H$ is only when the system is well determined.

A rank-deficient $H$ — a gauge freedom, a variable no residual observes —
makes the factorization fail. The dense path's answer is not to regularize
silently: it is to use the QR solver, or to fix the gauge.

## 3. QR

Solving the least-squares problem directly,

$$
\min_\delta \|J\delta + \tilde r\|^2,
$$

by a QR factorization of $J$ does not square the condition number and does
not require $H$ to be definite. The price is roughly twice the work and a
denser factorization. For a small, ill-conditioned problem this is often the
right trade; for a large one, the sparse path's QR (`SparseQRSolver`) does the
same job without the density.

## 4. Dense Schur

`ExplicitDenseSchur` eliminates the marked variables and solves the reduced
system densely. It is the dense counterpart of the sparse Schur of the next
chapter: same mathematics, same requirement that the eliminated block be
block-diagonal, but with no sparse machinery. Its use is for small problems
where the sparsity would not pay for itself.

## 5. Worked example

A 300-DoF problem: 6 poses (36 DoF) and 264 landmarks. The dense Jacobian is
$800\times300$ — 1.9 MB, unremarkable — and a dense Cholesky is $9$ Mflops,
under a millisecond. Sparse assembly would spend more time in the symbolic
bookkeeping than the factorization. At 10 000 DoF the same reasoning inverts:
the dense Jacobian is $80\,000 \times 10\,000$ — 6.4 GB — and the sparse path
is the only option.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — dense hides the structure you paid for</span>

A dense assembly of a sparse problem is not just slower; it changes the
numerics. The normal equations square the condition number of $J$, so a
problem that is perfectly conditioned in the least-squares sense can be
numerically singular in $H$, and the dense Cholesky will fail where the
sparse QR would have succeeded. If a dense solve reports a factorization
failure on a problem that "should" be fine, the condition number is the
story.

</div>

<div class="apx-note">
<span class="apx-title">Note — the mode is chosen once</span>

`JacobianMode` is a property of the `Problem`, chosen at construction, and
the optimizer reads it to dispatch. The same problem cannot be solved densely
and sparsely in turn without rebuilding it — the assembly path and the solver
type are coupled by design, so that a dense Jacobian is never fed to a
sparse solver.

</div>
