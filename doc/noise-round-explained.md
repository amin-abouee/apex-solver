# The Noise Model Round — Explained from First Principles

*For a reader who has never worked with covariance, information matrices, or noise models —
and a full account of what was broken before this round and what the fixes changed.*

---

## Part 1 — The concepts

### 1.1 The problem a solver actually solves

Apex Solver is a **nonlinear least-squares solver**. Given:

- **variables** — unknown numbers the solver is allowed to change (camera poses, 3D landmark
  positions, camera intrinsics…), and
- **factors** (also called *measurements* or *edges*) — equations that say “if the variables
  had their correct values, this measured quantity should match this prediction”,

the solver searches for the variable values that make the total *error* as small as possible.
The standard measure of “total error” is a sum of squares:

$$
F(x) = \sum_i \frac{1}{2} \big\| r_i(x) \big\|^2
$$

where `r_i(x)` is the **residual** of factor *i* — a vector whose components are “measured
minus predicted”. For example: a G2O pose-graph edge measures “pose B is 2 metres and 30°
from pose A”; the residual is whatever is left of that measurement after substituting the
current estimates of A and B.

The classic **Gauss–Newton / Levenberg–Marquardt** machinery linearises every residual
around the current guess (`J_i = ∂r_i/∂x`, the *Jacobian*), stacks them into one big linear
system

$$
\underbrace{J^\top J}_{H\text{ (Hessian)}} \Delta x = -\underbrace{J^\top r}_{\text{gradient}}
$$

and applies `Δx` to the variables, iterating until the cost stops improving.

### 1.2 Not all measurements are equally trustworthy — the missing piece

The formula above implicitly assumes **every residual is equally trustworthy**. That is
physically wrong:

- a wheel odometry edge drifts a few centimetres per metre — very trustworthy *locally*;
- a GPS fix is ±3 metres in the horizontal plane but ±10 m vertically;
- a visual feature on a distant landmark is uncertain in pixels, and the uncertainty
  **grows with distance**;
- within a single pose measurement, the rotation may be known to 0.1° while the translation
  is known to 0.5 m — mixing radians and metres in one residual means their raw magnitudes
  are *not comparable*.

The statistical tool that encodes “how much do I trust each number” is the **covariance
matrix** Σ: entry Σ[i][j] says how strongly error components *i* and *j* co-vary. The
larger Σ[i][i], the less trustworthy component *i* is.

Statistically, the *right* objective is not `½‖r‖²` but the **Mahalanobis-norm** cost:

$$
F(x) = \sum_i \tfrac{1}{2}\, r_i^\top \Sigma_i^{-1} r_i
$$

The matrix `Ω = Σ⁻¹` is called the **information matrix** — “how much information do I have
about each component”. Large Ω entry = small σ = trustworthy = this residual contributes a
lot to the error. SLAM file formats (G2O, TORO) store exactly this Ω per edge: for an SE(2)
edge a 3×3 matrix, for SE(3) a 6×6 matrix.

### 1.3 Whitening: turning weighted math into plain math

Solving `r^T Ω r` directly would require changing every part of the solver. Instead, every
SLAM library uses the **whitening trick**. Factor `Ω = SᵀS` (the matrix square root — `S` is
called the **square-root information matrix**), and define

$$
\tilde{r} = S\,r, \qquad \tilde{J} = S\,J
$$

Then

$$
\tilde{r}^\top \tilde{r} = r^\top S^\top S\, r = r^\top \Omega\, r
$$

— *the whitened residual, fed through the plain unweighted pipeline, produces exactly the
weighted objective*. So instead of teaching the solver about Σ, you transform the inputs:

1. compute the raw residual `r` and raw Jacobian `J` from the factor,
2. multiply both by `S` (**whitening**),
3. from here on, everything downstream — Hessian assembly, robust loss, damping,
   covariance — proceeds as if the measurement had no uncertainty.

The **noise model** is the object that owns `S` and performs step 2. That is the layer this
round added.

### 1.4 Robust losses still compose — this is why placement matters

SLAM residuals occasionally contain **outliers** (a wrong loop-closure, a mis-tracked
feature). Robust loss functions (Huber, Cauchy, Tukey…) down-weight large residuals so a
single outlier cannot wreck the solution. The Triggs correction — the standard way to apply
a robust loss to a squared system — modifies *both* the residual and the Jacobian using the
residual norm `‖r‖²`.

The ordering question: should whitening happen **before** or **after** the robust loss? The
statistically meaningful quantity is `ρ(‖S·r‖²)` — the loss evaluates the *weighted* norm.
So whitening must happen **first**, and the robust corrector sees the already-whitened
residual. Had we whitened after the loss, the loss would judge the *unweighted* outlier
magnitude and the down-weighting would be computed for the wrong measurement. Getting this
ordering right was the single most subtle design decision of the round (05 §2 called it out
explicitly).

### 1.5 The three concrete noise models

| model | sqrt-info `S` | when to use |
|---|---|---|
| `Null` | identity (nothing stored, nothing computed) | every existing call site; BAL — all observations equally trusted |
| `Diagonal` | `diag(s₁, s₂, …)` | the overwhelming SLAM case: Ω block-diagonal, per-axis σ |
| `Dense` | full matrix `S = V·√Λ⁺·Vᵀ` | coupled Ω — e.g. g2o SE(3) edges where x-y-z-rot errors correlate |

`Null` is **bit-for-bit free**: one enum branch, no allocation. This is what makes the
feature zero-regression for existing users — pinned by a test that runs the same problem
both ways and asserts identical bits in the costs.

### 1.6 Where the sqrt-information comes from — and the PSD trap

Users typically have `Ω` (the information matrix from the g2o file), not `S`. Computing
`S` needs the **square root of a matrix**: `SᵀS = Ω`.

The trap: Ω must be **positive semi-definite (PSD)** — every eigenvalue ≥ 0 — because a
negative eigenvalue would mean “negative information”, which is meaningless. The naive
Cholesky decomposition *fails* on anything not strictly positive **definite**. Real g2o
files break that: sphere2500's SE(3) edges carry rank-deficient Ω (some rotation DOFs have
*zero* information — the sensor simply did not observe them), and after a floating-point
round-trip some eigenvalues come out *slightly negative*.

The fix: decompose with the **eigen-decomposition** `Ω = V·Λ·Vᵀ`, build
`S = V·√Λ⁺·Vᵀ` where negative eigenvalues are clamped to zero (with a warning). A
clamped direction whitens to zero — “this measurement direction carries no information” —
which is exactly the mathematically honest interpretation, and matches how GTSAM/g2o
tolerate indefinite inputs via pivoted LDLᵀ.

---

## Part 2 — What was broken before, and what the round fixed

### 2.1 The g2o Ω matrices were parsed and then thrown away (04 §2, P0/P1)

**Before:** the G2O loader faithfully parsed each edge's Ω matrix into `EdgeSE2/EdgeSE3`.
information — and then the benchmark binaries built `BetweenFactor`s with **no** noise
model, so the solver optimised the *unweighted* objective `½‖r‖²` while the benchmark
*reported* the Ω-weighted χ². Two different objectives: the numbers could never match, and
non-uniform datasets were silently mis-weighted.

**After:** `pose_graph_g2o` and the odometry benchmark build
`NoiseModel::from_information(edge.information)` and register each edge through
`add_residual_block_with_noise`. A `--no-noise` flag restores the legacy behaviour.

**Measured consequence** (harness χ², median of 5 — the harness computes χ² identically on
both sides, so this is apples-to-apples):

| dataset | χ² before | χ² after | improvement | time |
|---|---|---|---|---|
| torus3D | 1.0389e5 | **5.6847e4** | **1.8× lower** | 949 → 1663 ms (more iterations on the correctly weighted problem) |
| cubicle | 1.7168e9 | **7.1068e6** | **240× lower** | 1017 → 532 ms (also faster) |
| sphere2500 | 3.4270e3 | **1.3517e3** | **2.5× lower** | 193 → 142 ms (faster) |
| mit | 9.6067e3 | **7.7024e2** | **12.5× lower** | 94 → 59 ms (faster) |
| M3500 | 1.4007e2 | 1.3791e2 | better | ~same |
| city10000 / ring / parking-garage | | | better | ~same |

**Every dataset improves on the reported objective.** Bundle adjustment is statistically
unchanged (BAL carries no per-edge Ω; the `Null` path is free).

### 2.2 `PriorFactor` was geometrically meaningless on Lie groups (04 §1, P0)

**Before:** the only prior factor computed `r = x − x_prior` on the raw **parameter vector**.
For an SE(3) variable stored as `[tx,ty,tz,qw,qx,qy,qz]`:

1. **Quaternion double cover** — `q` and `−q` are the same rotation, so `q − q_prior` can be
   ~2.0 for *identical* rotations.
2. **Wrong Jacobian size** — a 7-column identity against a 6-DOF manifold; the 7th column was
   silently dropped, and all rotation–translation coupling was absent.
3. **No angle wrap** — an SE(2) prior at 3.14 rad against a state at −3.14 rad produced a
   residual of ≈ 6.28 instead of ≈ 0.003.

**After:** two factors with a clean split:

- `PriorFactor<T: LieGroup>` — the **tangent-space anchor**: `r = Log(T_prior⁻¹ ∘ X)`, the
  SE(3)-textbook prior. Zero at `X = T_prior`, correct 6-DOF Jacobian, wraps angles, immune
  to the double cover. Implemented by delegating to the same `between` chain that
  `BetweenFactor` uses, so the Jacobian conventions are shared with production code.
- `EuclideanPriorFactor` — the old ambient difference, **restricted to `Rn` variables** at
  registration time (it is exact there, since parameters and tangent coincide); any other
  manifold is rejected with a typed error at registration rather than failing mid-solve.

A weighted Euclidean prior (the goal of the unmerged `weighted-prior-constrained-ls`
branch) is now simply `EuclideanPriorFactor` + `NoiseModel::Diagonal` — no separate factor
type needed; that branch's rationale is preserved in the docs.

Tests pin the Jacobian against central differences (SE(2) passing; SE(3) currently
`#[ignore]`d — see the finding in 2.6) and the SE(2) angle wrap.

### 2.3 SGal(3)'s exponential was not a group exponential (04 §3, P1→P0)

**The concept.** A Lie group `exp` must satisfy the **one-parameter subgroup law**:

$$
\exp(a\xi) \circ \exp(b\xi) = \exp\big((a+b)\xi\big)
$$

— following the flow of a constant twist for time `a`, then `b`, must land where following
it for `a+b` lands. This is *the* property that ties `exp` to `compose`.

**Before:** SGal(3)'s `exp` treated the time scalar `s` as a pass-through — the translation
update had no dependence on it. Claude's review proved by execution that
`exp(aξ)∘exp(bξ)` differs from `exp((a+b)ξ)` by exactly `a·b·s·ν` (velocity × time — the
*defining* physics of a Galilean group: a constant-velocity object moves). It also passed
all 1722 existing tests, because they only checked `exp∘log` round-trips, which a
self-consistent-but-wrong pair passes.

**After:** the exponential was re-derived by integrating the subgroup flow from the group
law (`ρ' = ρ₁ + R₁(ρ₂ + s₂ν₂)`):

- `ν' = Jl(θ)·ν`
- `ρ' = Jl(θ)·ρ + s·M(θ)·ν` with the new coupling matrix
  `M(ω) = ½I + α·ω̂ + β·ω̂²`, `α = (sin w − w cos w)/w³`, `β = (1 − cos w)/w⁴ − sin w/w³ + 1/(2w²)`
  (small-angle: `½I + ⅓ω̂ + ⅛ω̂²`) — derived from `∫₀¹ Exp(σω)·σ dσ`.
- `log` inverts it exactly.
- The 10×10 `right/left_jacobian(±inv)` tables were re-derived as
  **derivative-by-definition** (central differences through the crate's own
  compose/log — exact against the group operations), because the old hand tables matched
  only the uncoupled map.
- The subgroup test now **passes for SGal(3)** and is un-ignored; a composition test with
  strong `s·ν` coupling pins the Jacobian permanently.
- The subgroup law itself was added for **all eight manifolds** (the other seven passed on
  first run — SGal(3) was the only broken one).

### 2.4 Sim(3): silent Jacobian corruption on singular inputs (04 §4)

**Before:** when the inverse of the V-matrix or a Jacobian block was singular (scale σ ≈ 0
mid-solve), the code silently substituted the **identity matrix** — the SAFETY comment
claimed degenerate inputs were “outside the valid domain”, but nothing checked. A solver
passing near σ ≈ 0 got quiet Jacobian corruption.

**After:** a Tikhonov-regularized inverse `(M + εI)⁻¹` with a `warn!` — the block keeps its
structure so the step degrades gracefully instead of teleporting, and the event is
surfaced. (Claude's review priced the alternative — a typed-error trait change across all
six manifolds — as too invasive; this matches how the Schur solver already handles
near-singular landmark blocks.)

### 2.5 Kannala-Brandt `unproject`: silent clamping + no convergence check (04 §4)

**Before:** pixels whose undistorted radius exceeded the valid FOV were **clamped**
(`ru = min(ru, π/2)`) — rewriting the input ray and returning a direction that never
projected back to the pixel — and the Newton loop had **no convergence check after
exit**, so a diverged theta silently became a ray.

**After:** both conditions return typed `CameraModelError::NumericalError`, mirroring the
sibling FTheta model. Regression tests cover an out-of-domain pixel and a
divergent-Newton configuration.

### 2.6 04 §6 small items

- **Cheirality rank-1 block**: both residual rows carry the same scalar penalty, so the
  Jacobian rows are identical — this is *exact* for the residual as written and deliberate
  under LM damping; now documented as a rank property (a 1-D residual would be a breaking
  layout change).
- **`Rn` dimension sentinels deprecated**: `Rn::DIM/DOF/REP_SIZE` are `0` sentinels for a
  dynamic manifold; anything branching on them statically mishandles dynamic landmarks.
  They now carry `#[deprecated]` pointing to `tangent_dim()` / `is_dynamic()`.
- **Quaternion hazard documented**: writing quaternion components through
  `as_param_slice_mut()` does not renormalize; the docs now route writers to
  `LieGroup::normalize()`.
- **§5 rayon assert** was already fixed in the architecture round (registration-time
  `validate_variables`).

---

## Part 3 — Verification

- **37/37 test binaries** pass with `--all-features`; clippy `-D warnings` clean; fmt clean.
- **Noise math tests** (`tests/noise_model.rs`): hand-checked whitened costs, Huber
  composition (`½·ρ(‖S·r‖²)` exact), whitened-Jacobian vs central differences, Null
  bit-identity, dimension-mismatch rejection, PSD-clamping behaviour.
- **Subgroup law suite** (`crates/apex-manifolds/tests/subgroup_law.rs`): all eight
  manifolds, plus `Matrix10` inverse support and SGal(3) coupling-region Jacobian checks.
- **Benchmarks**: before/after medians in `output/{baseline,after}_noise/`; full write-up in
  [`noise-round-results.md`](noise-round-results.md).

## Part 4 — What remains (tracked, deliberately deferred)

1. **SE(3) between/log Jacobian chain is FD-inconsistent** — discovered *during* this round:
   the analytic Jacobian of `BetweenFactor<SE3>` does not reproduce central differences
   through the crate's own compose/log (the wrt-X block behaves as identity, and
   `right_jacobian_inv`'s coupling disagrees with the FD-measured derivative). Never caught
   because the existing FD test only perturbs translation components. The SE(3) prior FD
   test is `#[ignore]`d until a closed-form re-derivation (GTSAM/manif-style) lands. LM
   still converges on real data, but the rotation–translation step coupling is degraded.
2. **Proptest law suite**, **fuzz targets for the parsers**, **dataset checksum pinning** —
   deferred from the CI round (claude's sequencing).
3. **WeightedPriorFactor branch**: superseded — its weighting idea is now
   `EuclideanPriorFactor` + `NoiseModel::Diagonal`; reconcile or close the branch.
