# Noise Models

A factor produces a raw residual in measurement units — metres, pixels,
radians. A noise model converts it into a *statistically comparable* quantity so
that a 2 cm range error and a 0.5 px reprojection error can be summed in one
objective. This is the piece that makes the least-squares problem a maximum
likelihood estimate rather than an arbitrary weighted sum.

## What it stores

`NoiseModel` (`core::noise`) holds the **square-root information**, not the
covariance. Writing $\Omega = \Sigma^{-1}$ for the information matrix and
$\Omega = S^{\mathsf T}S$ for its factor, whitening is

$$
\tilde r = S\,r, \qquad \tilde J = S\,J
$$

so that

$$
\lVert \tilde r \rVert^2 = r^{\mathsf T} \Omega\, r
$$

which is the Mahalanobis distance. Storing $S$ rather than $\Sigma$ means
whitening is a multiply, with no factorization on the hot path.

Three representations, chosen by what the measurement actually needs:

| Variant | Stores | Cost |
|---|---|---|
| `Null` | nothing | Zero — no allocation, no arithmetic |
| `Diagonal(DVector)` | one entry per residual row | One multiply per row |
| `Dense(DMatrix)` | full `dim × dim` factor | Triangular multiply |

`Diagonal` covers the block-diagonal $\Omega$ that most SLAM graphs use.
Reach for `Dense` only when residual components are genuinely correlated.

## Constructing one

```rust
use apex_solver::core::noise::NoiseModel;

// From standard deviations — the common case. sqrt_info_i = 1/σ_i.
let noise = NoiseModel::from_sigmas(&[0.05, 0.05, 0.02])?;

// From square-root information directly (1/σ per row).
let noise = NoiseModel::from_diagonal_sqrt_info(&[20.0, 20.0, 50.0])?;

// From a full information matrix Ω (symmetric PSD).
let noise = NoiseModel::from_information(omega)?;

// From an already-factored S.
let noise = NoiseModel::from_sqrt_info(s)?;

// Unweighted.
let noise = NoiseModel::null();
```

Register it with the block:

```rust
problem.add_residual_block_with_noise(&[key_a, key_b], factor, loss, noise);
```

## Rank-deficient and indefinite information

Real datasets carry information matrices that are not positive definite. A g2o
file may encode zero roll/pitch information for a planar robot, giving a
rank-deficient $\Omega$; floating-point round-trips can leave slightly
negative eigenvalues.

`from_information` handles this by clamping negative eigenvalues to zero — the
nearest-PSD projection — and taking the symmetric square root
$S = V\sqrt{\Lambda^{+}}V^{\mathsf T}$, which satisfies
$S^{\mathsf T}S = \Omega$. Clamped directions whiten to zero, i.e. the
measurement stops constraining them.

**That clamp is silent, and on a genuinely ill-formed $\Omega$ it deletes real
constraints rather than empty ones.** When the difference matters, use the
reporting form:

```rust
let (noise, repair) = NoiseModel::from_information_reporting(omega)?;
if repair.is_material() {
    // `repair.clamped_directions`, `repair.min_eigenvalue` and the scale it is
    // judged against tell you whether this was floating-point noise or a
    // measurement whose Ω is wrong.
}
```

`InformationRepair` reports how many directions were clamped, the smallest
eigenvalue before clamping, and the largest eigenvalue it is judged against —
`relative_indefiniteness` is the ratio that distinguishes round-off from a
broken measurement.

`RepairStrategy` then decides what to do with a materially-repaired edge:

| Strategy | Effect |
|---|---|
| `Clamp` *(default)* | Keep the PSD projection. The objective stays the Ω-weighted χ², minus the clamped directions |
| `UnitWeight` | Substitute an identity weight for that edge. Trades χ² for unweighted cost — **the reported objective must be labelled accordingly** |

This is not hypothetical: the `cubicle` pose-graph dataset ships indefinite
$\Omega$ on some edges, and switching those to unit weight moves the
unweighted cost by a factor of ~7. Whichever you choose, say which objective
you are reporting.

## Whiten internally, or take a noise model — never both

Most factors return a raw residual and let the registered `NoiseModel` whiten
it. Some cannot: their weighting is inseparable from the measurement — GICP's
combined point covariance, a smart factor's internal elimination, an IMU
preintegration's propagated information matrix. Those whiten as they go and
report `Factor::whitens_internally() == true`.

Pairing an internally-whitened factor with a non-null noise model whitens
**twice** and silently over-weights the block. The two conventions are
indistinguishable by inspecting a residual, so registration rejects the
combination outright rather than letting it through:

```rust
// Rejected at registration — the factor already applied its information.
problem.try_add_residual_block_with_noise(&keys, gicp_factor, None, dense_noise)?;

// Correct.
problem.try_add_residual_block_with_noise(&keys, gicp_factor, None, NoiseModel::null())?;
```

## Where whitening happens in the pipeline

Whitening runs **after** the factor linearizes and **before** the robust loss —
see [From Factor Graph to Linear System](./linearization.md). The ordering is
not incidental: a robust loss decides whether a residual is an outlier, and
that is a question about standard deviations, not raw units. A 3 px reprojection
error is an outlier only relative to the measurement's σ.

## Choosing σ

The noise model is a modelling decision, not a tuning knob, and getting it wrong
biases the estimate rather than just slowing convergence:

- **Too small σ** (over-confident) — the factor dominates and pulls the solution
  toward that measurement, over-riding better-informed ones.
- **Too large σ** (under-confident) — the factor is effectively ignored.
- **Relative scale is what matters.** Multiplying every σ in the graph by a
  constant leaves the optimum unchanged; it only rescales the cost. What changes
  the answer is the *ratio* between different sensors' models.

For a robust loss to work as intended, σ should reflect the inlier
distribution — the loss then handles the tail. Inflating σ to "cover" outliers
and *also* applying a robust loss down-weights inliers twice.
