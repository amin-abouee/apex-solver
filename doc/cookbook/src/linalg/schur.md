# The Schur Complement

Eliminating a variable from a linear system is exact, not an approximation,
and it is the mechanism that makes a bundle adjustment tractable: a thousand
landmarks can be removed from the linear system and restored afterwards. This
chapter is the mathematics, the crate's implementation, and the structural
condition that decides when the machinery applies at all.

## 1. The mathematics

Partition the normal equations

$$
\begin{bmatrix} H_{kk} & H_{ke}\ H_{ek} & H_{ee}\end{bmatrix}
\begin{bmatrix}\delta_k\ \delta_e\end{bmatrix}
=
\begin{bmatrix}g_k\ g_e\end{bmatrix}
$$

into the kept variables $k$ and the eliminated ones $e$. Solving the second
block row and substituting gives the reduced system

$$
\underbrace{\left(H_{kk} - H_{ke}H_{ee}^{-1}H_{ek}\right)}_{S}\delta_k
= g_k - H_{ke}H_{ee}^{-1}g_e .
$$

$S$ is the **Schur complement**. The eliminated variables are recovered by
back-substitution once $\delta_k$ is known. Nothing is approximated: the
elimination is algebra, and the reduced solve is exact for the original
system.

## 2. The condition

The machinery requires the eliminated block $H_{ee}$ to be **block-diagonal**:
two eliminated variables must not share a residual row. If they do, $H_{ee}$ is
not invertible block-wise, and the "eliminate each block independently" step
that makes the computation cheap is not valid.

The crate enforces this rather than assuming it:

- `ChunkLayout` rejects two eliminated variables sharing a residual row;
- `SchurPartition::verify_block_diagonal` inspects the assembled Hessian and
  returns a typed error if the condition is violated.

<div class="apx-warning">
<span class="apx-title">Warning — the condition is a property of the problem</span>

Whether the eliminated set is block-diagonal depends on *which* variables are
eliminated, not on the solver. A visual-inertial window that eliminates
$\{state_0, bias_0, \text{landmarks}\}$ fails the test, because the IMU factor
puts `state_0` and `bias_0` in the same rows. The crate's answer is a separate
dense marginalization path for exactly that case (Part IV), not a relaxation
of the condition.

</div>

## 3. The implementation

<div class="apx-widget" data-widget="schur" data-title="Sparsity, fill-in, and the Schur solve" data-foot="Watch the eliminated blocks' fill-in and how the reduced system's size and cost scale with the landmark count."></div>

`ChunkedSchurEliminator` walks the residual rows that share an eliminated
variable and accumulates three quantities per chunk:

$$
\texttt{ete} = E^\top E\ (dof\times dof),
\qquad
\texttt{etf} = E^\top F\ (dof\times|F_c|),
\qquad
\texttt{etr} = E^\top r_c\ (dof),
$$

then subtracts the corrections

$$
S \mathrel{-}= \texttt{etf}^\top\,\texttt{ete}^{-1}\,\texttt{etf},
\qquad
\mathbf{g}_{\text{red}} \mathrel{-}= \texttt{etf}^\top\,\texttt{ete}^{-1}\,\texttt{etr}.
$$

The scratch buffers are reused across chunks, and $\texttt{ete}^{-1}$ is
**retained** for the back-substitution that restores the eliminated
variables.

`SchurPartition` owns the bookkeeping: the kept and eliminated block spans,
the global-to-local column maps for both spaces, and the offsets that place
each block in the reduced system's column space. `SchurOrdering` chooses the
elimination order, which is what controls the fill-in; `PatternFingerprint`
identifies the sparsity pattern so a cached ordering can be reused.

`ReducedSystem` is the result: $S$ (dense in the retained variables — it
typically is), the reduced gradient, and the retained inverses and right-hand
sides for back-substitution.

## 4. Explicit and implicit

<div class="apx-table">
<span class="apx-title">Two ways to use the same mathematics</span>

| | `ExplicitSparseSchur` | `ImplicitSparseSchur` |
|---|---|---|
| forms $S$ | yes, dense in `kept_dof²` | never |
| solves the reduced system | sparse Cholesky (or PCG) | PCG on a matrix-free operator |
| memory | quadratic in the retained set | linear in `nnz(J)` |
| cost | fastest per iteration | more iterations, cheaper each |

</div>

The implicit variant's preconditioner is `SchurPreconditioner`:
`None`, `BlockDiagonal` (of $H_{cc}$ alone — fast, less effective), or
`SchurJacobi` (the true Schur-Jacobi preconditioner, what Ceres uses, and
much better conditioned). The default is `SchurJacobi`.

## 5. Worked example

A window with 7 keyframes (105 DoF) and 200 landmarks (200 DoF), each seen by
two keyframes. Eliminating the landmarks:

$$
S = H_{kk} - H_{ke}H_{ee}^{-1}H_{ek}
$$

has size $105\times105$ instead of $305\times305$. The back-substitution
restores the 200 landmark steps from the retained $H_{ee}^{-1}$ and the right-hand
sides. The dense $S$ is 88 kB — comfortably dense, which is why the explicit
path keeps it dense and converts only at factorization time.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — the reduced system is dense, and that is the memory bound</span>

$S$ is typically dense: eliminating a landmark connects the keyframes that
observe it, and every pair of keyframes ends up coupled. The explicit path
therefore pays $O(\text{kept}^2)$ memory even though the input was sparse. A
window with a thousand kept variables needs 8 MB; with ten thousand, 800 MB.
The implicit path exists for exactly this regime, and the crossover is a
measurement, not a rule of thumb.

</div>

<div class="apx-note">
<span class="apx-title">Note — ordering is where fill-in is won or lost</span>

`SchurOrdering` decides the order in which eliminated variables are processed.
Eliminate a landmark whose keyframes are far apart and the intermediate
structure fills in; eliminate a spatially coherent one and it does not. The
crate computes an ordering from the pattern and fingerprints it, so a repeated
solve reuses it. The order does not change the mathematics — it changes the
arithmetic.

</div>

<div class="apx-paper">
<span class="apx-title">Papers — the technique</span>

Schur, "Über die Auflösung linearer Gleichungen mittels quadratischer
Iterationen", *Mathematische Annalen*, 1911. Hartley & Zisserman, *Multiple
View Geometry*, 2004 — the reduced camera system in §12.1. Agarwal, Mierle
et al., "Ceres Solver" — the sparse and implicit Schur variants, the
preconditioner choices, and the `SPARSE_SCHUR` / `ITERATIVE_SCHUR` vocabulary.
</div>
