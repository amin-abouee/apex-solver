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

## 5. The iterative solve: PCG

Both Schur paths that do not factorize $S$ solve the reduced system with
**preconditioned conjugate gradients**, and the crate's `linalg::schur::pcg`
is the single implementation both use — the explicit variant applies $S$
explicitly, the implicit one applies it matrix-free, and the loop itself is
identical.

The parameters are `PcgParams`:

| Field | Default | Meaning |
|---|---|---|
| `max_iterations` | 200 | the cap |
| `tolerance` | $10^{-6}$ | stop once $\lVert r\rVert <$ `tolerance` $\cdot\max(\lVert b\rVert, 1)$ |
| `q_tolerance` | `DEFAULT_ETA` = $10^{-2}$ | the forcing-sequence parameter $\eta$; `0.0` disables the quadratic-model rule |

### Why the residual rule is not enough

CG minimizes the quadratic model $\phi(x) = \tfrac12 x^\top A x - b^\top x$,
and at an iterate $x_i$ that value is free from vectors already on hand:

$$
\phi(x) = \tfrac12 x^\top(Ax) - b^\top x = -\tfrac12 x^\top(b + r)
\qquad\text{since}\quad Ax = b - r .
$$

A trust-region optimizer does not need $Ax = b$ solved *accurately*; it needs a
step that reduces the model. Insisting on a small residual on an
ill-conditioned $S$ burns iterations long after the model has stopped moving —
measured on the BAL datasets, every solve on Ladybug ran the full iteration
cap without the residual rule ever firing.

So the solve also stops when the **relative model improvement per iteration**
falls below the forcing sequence $\eta/i$:

$$
Q_i = -x^\top(b + r), \qquad \frac{i\,(Q_i - Q_{i-1})}{Q_i} < \eta .
$$

The $\eta/i$ shape tightens the requirement as iterations accumulate: early
Newton steps are cheap and approximate, later ones — where the optimizer is
close and needs accuracy — are solved harder.

### Why the default is $10^{-2}$ and not Ceres' $10^{-1}$

The source's own sweep, over the four BAL datasets with the implicit Schur,
final RMSE against the exact solver (and the speed against it in
parentheses):

| $\eta$ | Ladybug | Trafalgar | Dubrovnik | Venice |
|---|---|---|---|---|
| $10^{-1}$ | **+0.0849** (5.8×) | −0.0128 (1.4×) | −0.0226 (1.3×) | +0.0079 (3.4×) |
| $10^{-2}$ | +0.0013 (4.0×) | −0.0104 (0.4×) | −0.0189 (1.3×) | +0.0045 (2.7×) |
| $10^{-3}$ | +0.0005 (2.8×) | +0.0022 (0.5×) | +0.0001 (1.1×) | −0.0014 (1.9×) |

At $\eta = 10^{-1}$ Ladybug loses 9.7 % accuracy and does not recover with
more optimizer iterations — the truncated steps converge somewhere *worse*,
not merely more slowly. $10^{-2}$ buys that accuracy back (0.14 % off the
exact solver) while keeping 4.0× / 1.3× / 2.7× on the three large datasets,
which is where the solver is used at all.

### The outcome

`PcgResult` reports the solution, the iteration count, the final residual, and
**why the solve stopped** — `PcgTermination::{Residual, QuadraticModel,
MaxIterations, Breakdown}`. A `MaxIterations` termination on a Newton step
means the step was truncated, which is worth logging: it is the difference
between "the reduced solve converged" and "the reduced solve gave up".

### The ordering and the partition

Two pieces of bookkeeping make the elimination work, and both are public:

- **`SchurOrdering`** decides *which* variables are eliminated and in what
  order. `with_auto_detect(true)` adds a structural rule to the explicitly
  marked set: a variable is eliminated if its manifold type is in
  `eliminate_types` and, for `Rn`, its DoF matches `eliminate_rn_size` — by
  default an `Rn` with exactly 3 DoF is a landmark.
  `should_eliminate(manifold_type, size)` is the test itself, and
  `should_eliminate_by_name(name, size)` is the same test for a variable
  identified by its manifold's name (`"Rn"`, `"SE3"`, …); an unknown name
  never eliminates. `effective_landmark_keys(variables, marked, ordering)`
  returns the set that actually got eliminated — the marked ones plus the
  auto-detected ones. The order is what controls the fill-in.
- **`SchurPartition`** owns the index bookkeeping: `kept_blocks` and
  `eliminated_blocks` (the variable sets), `kept_dof` / `eliminated_dof` /
  `total_dof` (the sizes), `kept_local` / `eliminated_local` (the per-variable
  local index of a block's first column), `kept_offset` / `eliminated_offset`
  (where the block lands in the reduced column space), and `BlockSpan` /
  `ColSlot` / `EliminatedBlocks` (the span types). `verify_block_diagonal`
  checks the assembled Hessian against the structural requirement.

## 6. Worked example

A window with 7 keyframes (105 DoF) and 200 landmarks (200 DoF), each seen by
two keyframes. Eliminating the landmarks:

$$
S = H_{kk} - H_{ke}H_{ee}^{-1}H_{ek}
$$

has size $105\times105$ instead of $305\times305$. The back-substitution
restores the 200 landmark steps from the retained $H_{ee}^{-1}$ and the right-hand
sides. The dense $S$ is 88 kB — comfortably dense, which is why the explicit
path keeps it dense and converts only at factorization time.

## 7. The catch

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
