# Noise Models and Whitening

Every measurement has an uncertainty, and the solver must weigh it. This
chapter is the crate's representation of that uncertainty — a square-root
information matrix — how it is built, how it is applied, and the two
conventions (whiten here or whiten there) that a factor author has to choose
between.

## 1. The problem

The objective of a factor-graph solve is not $\sum\|\mathbf{r}\|^2$ but the
information-weighted one:

$$
F(\mathbf{x}) = \sum_i \tfrac12\,\mathbf{r}_i^\top \Omega_i\,\mathbf{r}_i,
\qquad \Omega_i = S_i^\top S_i .
$$

Applying $S$ to the residual and the Jacobian *before* the robust corrector
turns the problem into an ordinary least-squares one in the whitened
coordinates:

$$
\tilde r_i = S_i r_i,\qquad \tilde J_i = S_i J_i
\quad\Longrightarrow\quad
F(\mathbf{x}) = \sum_i \tfrac12\,\rho(\|S_i r_i\|^2),
$$

which is exactly the $\Omega$-weighted objective that g2o-style benchmarks
report — and it means the optimizer, the corrector and the covariance
machinery all work in one coordinate system.

## 2. The model

<div class="apx-symbols apx-table">
<span class="apx-title">`NoiseModel`</span>

| Variant | Holds | Used for |
|---|---|---|
| `Null` | nothing | unweighted measurements; the zero-cost identity |
| `Diagonal(DVector)` | one square root per residual row | the block-diagonal $\Omega$ most SLAM graphs have |
| `Dense(DMatrix)` | the full $S$ | correlated measurements |

</div>

`NoiseModel::Null` is not "no information": it is the identity, and it is
*zero-cost* — no allocation, no arithmetic. A block registered with it behaves
exactly as if it had been whitened by $I$.

## 3. Building a model

```rust
let noise = NoiseModel::from_sigmas(&[σ1, σ2, σ3])?;      // diagonal
let noise = NoiseModel::from_information(omega)?;           // dense, from Ω
let noise = NoiseModel::from_sqrt_info(s)?;                // dense, from S
```

`from_sigmas` rejects non-finite or non-positive sigmas — a negative sigma is
not a measurement.

`from_information` takes $\Omega$, symmetrizes it, eigendecomposes it, and
builds

$$
S = \sum_{\lambda_k>0} \sqrt{\lambda_k}\,\mathbf{u}_k\mathbf{u}_k^\top .
$$

Directions with $\lambda_k \le 0$ contribute nothing to $S$: an exactly-zero
eigenvalue is an unobserved degree of freedom, and a negative one is
indefinite data. g2o and GTSAM tolerate indefinite $\Omega$ in the same way —
their pivoting LDLᵀ ignores the offending directions — so the crate clamps
rather than rejecting.

But "clamping" and "this was fine" are different claims, and the crate keeps
them apart. `from_information` warns when the repair was **material**; the
reporting variant returns the detail:

```rust
let (model, repair) = NoiseModel::from_information_reporting(omega)?;
if repair.is_material() { /* the edge may be ill-formed */ }
```

`InformationRepair` says how many directions were clamped, the smallest
eigenvalue before clamping, and the largest (the scale the smallest is judged
against). `is_material` distinguishes a *rank-deficient* $\Omega$ — legitimate,
an unobserved direction — from an *indefinite* one, which is large relative
to the matrix's own scale (the tolerance is `INDEFINITE_TOLERANCE = 1e-9`).
The repair can also be *policy* rather than arithmetic. `RepairStrategy` is
the shared vocabulary — `Clamp` (the default: keep the nearest-PSD
projection) or `UnitWeight` (substitute an identity weight for an edge whose
Ω needed a *material* repair) — with `RepairStrategy::build` producing the
model under a strategy and `RepairSummary` tallying a whole graph
(`record`, `is_clean`, `edges`, `materially_repaired`, `unit_weighted`). A
harness can then report how many edges were ill-formed rather than repairing
them in silence.

## 4. Applying a model

```rust
noise.whiten_residual(&mut residual);   // r ← S·r
noise.whiten_jacobian(&mut jac, rows, cols);   // J ← S·J
```

Both are in-place and allocation-free for the diagonal case (a row scaling).
`whiten_jacobian` multiplies row $i$ of every column by $s_i$, which is the
diagonal case of $S J$.

## 5. Two conventions, one trap

<div class="apx-table">
<span class="apx-title">Who whitens</span>

| Convention | Who applies $S$ | Registration |
|---|---|---|
| the factor returns the raw residual | the problem, from the noise model | give a `NoiseModel` |
| the factor whitens internally | the factor | **must** be `NoiseModel::null()` |

</div>

A factor that whitens internally does so because its weighting is bound up with
the measurement: GICP's combined point covariance, a smart factor's internal
Schur elimination, a preintegration's information matrix. Such a factor
declares `whitens_internally() -> true`, and the registration path rejects a
non-null noise model:

```rust
if factor.whitens_internally() && !matches!(noise, NoiseModel::Null) {
    return Err(CoreError::InvalidInput(
        "this factor applies its own square-root information; register it with \
         NoiseModel::null(). Supplying a noise model as well would whiten the \
         residual twice."));
}
```

Double whitening is invisible in the residual: the block is simply over-weighted
by the square of its information, the solve converges confidently, and the
answer is wrong.

<div class="apx-note">
<span class="apx-title">Note — dimension zero means "no model"</span>

`NoiseModel::Null.dim()` is 0, and the registration check accepts a model of
dimension 0 for any factor. That is what lets `add_residual_block` (no noise
argument) register a block whose factor whitens internally, with the check
above still firing if the *same* factor is later given a real model.

</div>

## 6. Worked example

A measurement with a standard deviation of 0.02 m in $z$ and 0.05 m in $x$:

```rust
let noise = NoiseModel::from_sigmas(&[0.05, 0.05, 0.02])?;
```

The diagonal model whitens the residual to $r/\sigma$ per row, so a 2 cm error
in $z$ becomes a residual of 1 in whitened units — the same scale as every
other measurement's, which is what makes a single Huber threshold meaningful
across the graph.

## 7. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a clamped direction is unobserved, not zero-information</span>

When `from_information` clamps a negative eigenvalue to zero, the corresponding
residual direction stops constraining anything — the factor becomes blind
there. That is the right behaviour for an unobserved direction, but it is
*silently* blind if the eigenvalue was negative because the edge is
ill-formed. The crate's answer is the reporting path: `is_material` and the
graph-level `is_clean` summary exist so a run can say "12 edges were repaired"
rather than converging to a slightly over-confident answer.

</div>

<div class="apx-note">
<span class="apx-title">Note — the noise model is the measurement's, not the variable's</span>

The information matrix belongs to the *edge*. The variable's uncertainty is
the variable's covariance (`ManifoldVariable::set_covariance`), and the
difference between the two is the difference between "how well this
measurement is known" and "how well this unknown is known". Conflating them —
using a measurement's $\Omega$ as a variable's prior — is a common and quiet
error.

</div>
