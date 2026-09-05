# Linear Solvers

Every optimizer iteration reduces to one linear solve. This chapter documents
the backends in `src/linalg/`: what each one computes, the equations as
implemented, and how to configure them. Formulas name the module they come from.

Two traits define the contract (`linalg/mod.rs`):

| Method | Returns |
|---|---|
| `solve_normal_equation(r, J)` | $h$ solving $J^{\mathsf T}J\,h = -J^{\mathsf T}r$ |
| `solve_augmented_equation(r, J, damping)` | $h$ solving $(J^{\mathsf T}J + \lambda D)\,h = -J^{\mathsf T}r$ |
| `hessian_vec_product(v)` | $H v$ for the **un-damped** $H$ |
| `get_gradient()` | $+J^{\mathsf T}r$ — note the sign |
| `get_hessian()` | $J^{\mathsf T}J$ *when the backend happens to hold one*; `None` is valid |

`get_hessian()` returning `None` is not a failure. A backend that eliminates
straight from $J$ never materialises $J^{\mathsf T}J$, and callers must
degrade to `hessian_vec_product` rather than error — that is what makes the
matrix-free paths expressible.

`JacobianMode` picks the assembly: `Sparse` (default) or `Dense` (below ~500
DOF). `LinearSolverType` then picks the algorithm within that mode, and the
optimizer rejects mismatched combinations (`levenberg_marquardt.rs`).

## Normal equations

For the non-Schur backends, $H = J^{\mathsf T}J$ and $g = J^{\mathsf T}r$
are formed once per iteration (`sparse/normal_eq.rs`). The sparsity of
$J^{\mathsf T}$, of $H$, and the value permutation linking them are
cached across iterations, so each evaluation is a parallel value gather plus
faer's parallel sparse product. The cache is keyed on $J$'s sparsity, which
must not change between iterations.

## Sparse Cholesky

`SparseCholeskySolver` — $LL^{\mathsf T}$ factorization of $H$ (or
$H + \lambda D$). Symbolic factorization cached, numeric pass repeated.
The default for pose graphs.

## Sparse QR

`SparseQRSolver` — factorizes $J$ directly rather than forming $H$,
which squares the condition number. More robust on ill-conditioned problems, at
a higher per-iteration cost.

## Dense Cholesky / QR

`DenseCholeskySolver`, `DenseQRSolver` — for `JacobianMode::Dense`. Both route
their SPD solves through one shared `solve_spd` helper (`dense/cholesky.rs`)
so the factorization and error mapping exist in exactly one place.

---

# The Schur complement

## Why

In bundle adjustment most variables are landmarks. Ladybug has 1,723 cameras
and 156,502 landmarks: 15,507 retained DOF against 469,506 eliminated. Landmarks
are mutually unconnected — no factor touches two of them — so the landmark block
of the Hessian is **block-diagonal** and can be eliminated cheaply, leaving a
system in the cameras alone.

## The algebra

Partition the variables into an eliminated set (**e**, Ceres's "group 0") and a
retained set (**k**, "group 1"). The system becomes
(`linalg/schur/partition.rs`):

$$
\begin{bmatrix} H_{kk} & H_{ke} \\\\ H_{ke}^{\mathsf T} & H_{ee} \end{bmatrix}
\begin{bmatrix} \delta_k \\\\ \delta_e \end{bmatrix}
=
\begin{bmatrix} g_k \\\\ g_e \end{bmatrix}
$$

Eliminating $\delta_e$ gives the **reduced system**:

$$
\underbrace{\bigl(H_{kk} - H_{ke} H_{ee}^{-1} H_{ke}^{\mathsf T}\bigr)}_{S}\,\delta_k
\;=\; g_k - H_{ke} H_{ee}^{-1} g_e
$$

and back-substitution recovers the eliminated half:

$$
\delta_e \;=\; H_{ee}^{-1}\bigl(g_e - H_{ke}^{\mathsf T}\delta_k\bigr)
$$

Nothing here mentions cameras or points — the partition is agnostic to what the
variables mean, which is why the same code covers classic BA, inverse-depth
parameterisations, LiDAR features and marginalization.

### The precondition, and how it is enforced

The elimination is exact **only** when $H_{ee}$ is block-diagonal, i.e. the
eliminated variables are mutually unconnected. Violating it yields a wrong step
with no other symptom, so it is checked rather than assumed:

- Solvers that form $H$ call `SchurPartition::verify_block_diagonal`
  (`partition.rs`) — one pass over the eliminated columns' nonzeros.
- The matrix-free path never forms $H$, so it checks against $J$'s rows
  instead: a residual row touching two eliminated variables is the same
  violation (`sparse/schur/implicit.rs`, `ensure_structure`).

Both report a typed error naming the two offending variables.

### Damping under Schur

$\lambda D$ is applied to **both** blocks before elimination — damping the
reduced system instead would not be the same problem
(`levenberg_marquardt.rs`, and `explicit.rs`):

$$
S \;=\; (H_{kk} + \lambda D_k) - H_{ke}\,(H_{ee} + \lambda D_e)^{-1} H_{ke}^{\mathsf T}
$$

### Landmark block inversion

$H_{ee}^{-1}$ is formed blockwise. Every Schur solver shares one
regularized-retry policy, `EliminatedBlocks::invert_in_place`
(`partition.rs`), so a landmark is regularized identically regardless of
which solver eliminated it.

## Choosing what to eliminate

`SchurOrdering` (`linalg/schur/ordering.rs`) classifies variables. **Manual
marks always apply**:

```rust
problem.mark_for_elimination(landmark_key);
```

Auto-classification by manifold type and DOF is available but **off by
default**:

```rust
let ordering = SchurOrdering::new().with_auto_detect(true);
```

> The default is off for a specific reason (`ordering.rs`): `Rn(3)` is also
> how self-calibration represents intrinsics `[focal, k1, k2]`. Auto-detecting
> "any `Rn(3)` is a landmark" would silently eliminate intrinsic blocks and
> corrupt $S$. Turn it on only when you know every `Rn(3)` in the problem is
> a landmark.

---

## The four Schur solvers

| `LinearSolverType` | Sub-config | Mode | Ceres equivalent |
|---|---|---|---|
| `ExplicitSparseSchur` | `ExplicitSchurVariant::Sparse` | Sparse | `SPARSE_SCHUR` |
| `ExplicitSparseSchur` | `ExplicitSchurVariant::Chunked` | Sparse | `SchurEliminator` |
| `ExplicitSparseSchur` | `ExplicitSchurVariant::Iterative` | Sparse | `ITERATIVE_SCHUR` + `use_explicit_schur_complement` |
| `ImplicitSparseSchur` | preconditioner | Sparse | `ITERATIVE_SCHUR` |
| `ExplicitDenseSchur` | — | **Dense** | `DENSE_SCHUR` |

### `ExplicitSchurVariant::Sparse`

Forms $H = J^{\mathsf T}J$, builds $S$ explicitly, factorizes with
sparse Cholesky. The most accurate: it solves the reduced system exactly, and
is the reference the other paths are measured against.

$S$ is accumulated into a dense $\text{kept\_dof}^2$ buffer and filtered
back to sparse (`explicit.rs`). That buffer is the current scaling limit —
1.9 GB at Ladybug's 15,507 retained DOF.

### `ExplicitSchurVariant::Chunked`

Algebraically **identical** to `Sparse` — the benchmark confirms bit-identical
RMSE on all four datasets — but builds $S$ chunk by chunk directly from
$J$, never forming $J^{\mathsf T}J$, $J^{\mathsf T}$, or the value
permutation. On the largest BAL problem those account for ~24 GB of the ~32 GB
needed before elimination can start.

Per chunk $c$ (the rows sharing one eliminated variable), with
$E = J[\text{rows}_c, e]$ and $F = J[\text{rows}_c, k]$:

$$
S \mathrel{-}= (E^{\mathsf T}F)^{\mathsf T} (E^{\mathsf T}E)^{-1} (E^{\mathsf T}F),
\qquad
g_{\text{red}} \mathrel{-}= (E^{\mathsf T}F)^{\mathsf T} (E^{\mathsf T}E)^{-1} (E^{\mathsf T}r_c)
$$

> **This trades time for memory and is slower in wall clock** — 1.2×–2.1× the
> `Sparse` variant. The chunk sweep is inherently serial: reading rows out of a
> column-major $J$ without building a CSR copy relies on one forward-only
> cursor per retained column, which forces chunks to be visited in increasing
> row order. The $J^{\mathsf T}J$ path it replaces is fully parallel, so on a
> many-core machine the lost parallelism outweighs the saved work. Requires each
> eliminated variable's rows to be contiguous, which
> `Problem::group_rows_for_elimination` arranges.

### `ExplicitSchurVariant::Iterative`

Forms $S$ explicitly like `Sparse`, then solves it with PCG instead of
Cholesky. Carries the same $\text{kept\_dof}^2$ memory cost while solving
less exactly, so it buys nothing on memory — but it measures 0.5×–1.3× `Sparse`
on time.

### `ImplicitSparseSchur` — the matrix-free path

**Forms neither $S$ nor $J^{\mathsf T}J$.** Writing
$J = [\,E \mid F\,]$, PCG only ever needs $S$'s action on a vector, and
expanding that action so it reads $J$ alone gives the whole solver
(`implicit.rs`):

$$
\begin{aligned}
y &= F v \\\\
t &= E^{\mathsf T} y \\\\
u &= (E^{\mathsf T}E + \lambda D_e)^{-1} t \\\\
y &\leftarrow y - E u \\\\
S v &= F^{\mathsf T} y + \lambda D_k v
\end{aligned}
$$

Four passes over $J$'s nonzeros, no intermediate matrix. This is why it
exists: $J^{\mathsf T}J$ for BA carries a dense block for **every pair of
cameras sharing a landmark** — fill-in $J$ does not have — so cost scales
with $\mathrm{nnz}(J)$ rather than $\mathrm{nnz}(J^{\mathsf T}J)$. Ceres
states the same cost model for `ITERATIVE_SCHUR`.

Everything else it needs also comes from $J$ (`linalg/schur/jacobian_ops.rs`):
the gradient $J^{\mathsf T}r$, the damping diagonal
$\operatorname{diag}(J^{\mathsf T}J)_j = \lVert J_{:,j}\rVert^2$, and the
$E^{\mathsf T}E$ blocks as column dot products.

`get_hessian()` returns `None` here; the quadratic model is served exactly as
$J^{\mathsf T}(Jv)$.

**This is what `for_bundle_adjustment()` selects.**

### `ExplicitDenseSchur`

The same explicit construction over a dense Hessian, for `JacobianMode::Dense`.
Targets a few thousand DOF — a dense $J^{\mathsf T}J$ for the smallest BAL
dataset would be ~485k × 485k.

---

## Preconditioners

`SchurPreconditioner` applies to both PCG paths. The three match Ceres's
options for `ITERATIVE_SCHUR`:

| This crate | Ceres | Built from |
|---|---|---|
| `None` | `IDENTITY` | — (unpreconditioned) |
| `BlockDiagonal` | `JACOBI` | diagonal blocks of $F^{\mathsf T}F$ (i.e. $H_{kk}$) |
| `SchurJacobi` *(default)* | `SCHUR_JACOBI` | diagonal blocks of $S$ itself |

`SchurJacobi` is the true one — it includes the elimination correction, so it
preconditions $S$ rather than the unreduced retained block:

$$
S_{ii} \;=\; (F^{\mathsf T}F + \lambda D)_{ii} \;-\; \sum_{j\,\in\,\text{visible}(i)} (F^{\mathsf T}E)_{ij}\,H_{ee,jj}^{-1}\,(F^{\mathsf T}E)_{ij}^{\mathsf T}
$$

summed over the eliminated blocks visible to $i$ — captured by a visibility
index cached per sparsity pattern, so this costs $O(\text{observations})$
rather than $O(\text{kept} \times \text{eliminated})$. Each block is inverted
with the shared retry policy; a block that will not invert falls back to
identity, because a preconditioner is a convergence aid and must not fail the
solve.

## PCG and the forcing sequence

Both PCG paths share one loop (`linalg/schur/pcg.rs`), starting from
$x = 0$ — every caller solves for a Newton *step*, not a general system that
would benefit from a warm start.

**Two stopping rules, and the second matters more.** The residual rule is

$$
\lVert r \rVert < \varepsilon_r \cdot \max(\lVert b\rVert, 1)
$$

but CG minimises the quadratic model $\varphi(x) = \tfrac12 x^{\mathsf T}Ax - b^{\mathsf T}x$,
and that value is available for free from vectors already on hand:

$$
\varphi(x) \;=\; -\tfrac12\, x^{\mathsf T}(b + r) \qquad \text{since } Ax = b - r
$$

A trust-region optimizer does not need the system solved *accurately*; it needs
a step that reduces the model. Insisting on a small residual on an
ill-conditioned $S$ burns iterations long after the model stops moving —
**measured on Ladybug, every solve ran the full iteration cap without the
residual rule ever firing.** So, following Ceres, the solve also stops on the
forcing sequence with $Q_i = -x^{\mathsf T}(b + r)$:

$$
i \cdot \frac{Q_i - Q_{i-1}}{Q_i} \;<\; \eta
$$

The $\eta/i$ shape tightens the requirement as iterations accumulate, so
early Newton steps are cheap and approximate while later ones — where the
optimizer is close and needs accuracy — are solved harder.

**$\eta$ defaults to `1e-2`, deliberately tighter than Ceres's `1e-1`**,
because `1e-1` measured 9.7% worse RMSE on Ladybug and that loss did not recover
with more optimizer iterations. The sweep is tabulated at `DEFAULT_ETA`
(`pcg.rs`). Set `with_schur_cg_q_tolerance(0.0)` to disable the rule and get an
exact solve — what the cross-solver agreement tests do.

`PcgResult::termination` reports which rule fired: `Residual`,
`QuadraticModel`, `MaxIterations` (the step is truncated — worth logging) or
`Breakdown`.

## Configuration

```rust
use apex_solver::linalg::{LinearSolverType, ExplicitSchurVariant, SchurPreconditioner};
use apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig;

// The BA preset: matrix-free, Schur-Jacobi, forcing sequence on.
let config = LevenbergMarquardtConfig::for_bundle_adjustment();

// Exact reduced solve instead, when the step must not depend on a tolerance.
let exact = LevenbergMarquardtConfig::for_bundle_adjustment()
    .with_linear_solver_type(LinearSolverType::ExplicitSparseSchur)
    .with_schur_variant(ExplicitSchurVariant::Sparse);

// Memory-bound: never form JᵀJ, accept the slower serial sweep.
let low_memory = LevenbergMarquardtConfig::for_bundle_adjustment()
    .with_linear_solver_type(LinearSolverType::ExplicitSparseSchur)
    .with_schur_variant(ExplicitSchurVariant::Chunked);

// Tune the iterative paths.
let tuned = LevenbergMarquardtConfig::for_bundle_adjustment()
    .with_schur_preconditioner(SchurPreconditioner::SchurJacobi)
    .with_schur_cg_params(200, 1e-6)   // max iterations, residual tolerance
    .with_schur_cg_q_tolerance(1e-2);  // forcing sequence; 0.0 disables
```

Mark what to eliminate on the problem itself:

```rust
let point = problem.add_variable(ManifoldType::RN, position);
problem.mark_for_elimination(point);
```

## Which to use

| Situation | Choice |
|---|---|
| Large BA (the default) | `ImplicitSparseSchur` — fastest on the three large BAL sets, forms neither $J^{\mathsf T}J$ nor $S$ |
| The step must be exact | `ExplicitSparseSchur` / `Sparse` — no tolerance dependence |
| Small problems | `ExplicitSparseSchur` / `Sparse` — direct factorization wins when $S$ is small |
| $J^{\mathsf T}J$ will not fit | `ExplicitSparseSchur` / `Chunked` — same answer, serial, memory-lean |
| Dense mode, few thousand DOF | `ExplicitDenseSchur` |
| Pose graphs | Not Schur at all — `SparseCholesky` |

Measured numbers for each are in [Benchmarks](./benchmarks.md).

---

## Covariance estimation

`Covariance::compute` re-linearizes the problem at a point, forms a clean
$H = J^{\mathsf T}J$ and inverts via sparse Cholesky or dense SVD:

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
