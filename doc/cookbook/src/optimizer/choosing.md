# Choosing a Solver

Three optimizers and seven linear-solver types is more choice than most
problems need. This chapter is the decision procedure: what to pick, in what
order to consider, and how to tell that the choice was wrong from the solve's
own report.

## 1. The short answer

<div class="apx-table">
<span class="apx-title">Default choices, and when to change them</span>

| Situation | Optimizer | Linear solver | Assembly |
|---|---|---|---|
| a typical factor graph, large | Levenberg–Marquardt | `SparseCholesky` | `Sparse` |
| a graph with a few hundred unknowns | Levenberg–Marquardt | `DenseCholesky` | `Dense` |
| a graph with landmarks to eliminate | Levenberg–Marquardt | `ExplicitSparseSchur` | `Sparse` |
| a graph with very many kept variables | Levenberg–Marquardt | `ImplicitSparseSchur` | `Sparse` |
| a small, well-conditioned refinement | Gauss–Newton | `DenseCholesky` | `Dense` |
| a poor initial guess, or a hard step bound | Dog Leg | `SparseCholesky` | `Sparse` |
| a rank-deficient normal matrix | any | `SparseQRSolver` / `DenseQRSolver` | matching mode |

</div>

The default is the crate's default: Levenberg–Marquardt, `SparseCholesky`,
`Sparse`. It is right for most problems because most problems are large,
sparse, and not perfectly conditioned.

## 2. The crate's own presets

Before choosing anything, look at what the crate already measured. The LM
config carries three presets, each backed by a sweep rather than a guess.

<div class="apx apx-table">
<span class="apx-title">`LevenbergMarquardtConfig` presets</span>

| Preset | Sets | Why |
|---|---|---|
| `for_2d_pose_graph()` | damping $10^{-4}$, diagonal bounds $(1, 1)$ | on the 2D suite (M3500, mit, city10000, ring) and cubicle, **scalar** $\lambda I$ damping is 2–7× faster than $\lambda\operatorname{diag}(H)$ at unchanged cost. The equal bounds *are* the uniform damping |
| `for_large_3d_pose_graph()` | damping $10^{-5}$, cost and parameter tolerances $10^{-3}$ | for graphs with an early cost plateau (parking-garage class): the weighted objective plateaus while the unweighted cost is already accurate, so looser tolerances stop the plateau early (~2× faster at identical unweighted cost), and the smaller initial damping keeps the accurate basin |
| `for_bundle_adjustment()` | `ImplicitSparseSchur`, `ExplicitSchurVariant::Sparse`, `SchurJacobi` preconditioner, damping $10^{-3}$, 20 iterations, Ceres' tolerances | the matrix-free Schur, measured 2.2× faster in total on the BAL datasets |

</div>

The bundle-adjustment preset's own measurement table, from the source:

| dataset | implicit | explicit/Sparse | ΔRMSE |
|---|---|---|---|
| Ladybug | 18.76 s, 0.876538 | 75.24 s, 0.875283 | +0.14 % |
| Trafalgar | 6.27 s, 0.798081 | 2.51 s, 0.808522 | −1.29 % |
| Dubrovnik | 31.34 s, 0.768611 | 41.91 s, 0.787517 | −2.40 % |
| Venice | 20.23 s, 0.752080 | 51.15 s, 0.747589 | +0.60 % |

It forms neither $J^\top J$ nor $S$, so it is the only one of the four that
keeps scaling. It loses only on Trafalgar, the smallest set, where the
absolute cost is about 4 s against about 56 s saved on Ladybug. And
`ExplicitSparseSchur` remains the choice when the reduced solve must be
*exact*: its step does not depend on a tolerance, and it is the reference
every other path is measured against.

<div class="apx apx-note">
<span class="apx-title">Note — the 2D preset contradicts the general advice</span>

This book says elsewhere that Marquardt's diagonal scaling is the default
because it is scale-relative. The 2D preset says otherwise *for 2D pose
graphs*: uniform $\lambda I$ is 2–7× faster there, at unchanged cost. The
reason is that a 2D pose graph's columns are already comparable in magnitude
— a translation in metres and an angle in radians, on a map of a few hundred
metres — so the diagonal scaling has nothing to correct, and computing it
costs. A preset is a measured statement about a shape of problem, and the
shape matters.

</div>

## 3. The decision procedure

**Step 1 — how big?** Below roughly 500 DoF, use the dense path: the
bookkeeping of the sparse path costs more than the factorization saves. Above
it, use sparse.

**Step 2 — can anything be eliminated?** If the graph has variables that are
touched by a few factors and nowhere else — landmarks, in the classical
bundle adjustment — and the eliminated set is block-diagonal (the crate
verifies this), use a Schur solver. The reduced system is smaller, and the
landmark steps are recovered by back-substitution.

**Step 3 — is the normal matrix definite?** If the problem has a gauge
freedom or a variable no residual observes, $H$ is singular and Cholesky
fails. The answer is not to damp harder: it is to *constrain* the direction
(fix the gauge, add a prior) or to use the QR solver, which solves the
least-squares problem without forming $H$.

**Step 4 — is the model good near the solution?** If the initial guess may
be far enough that the quadratic model is useless, use Dog Leg (a hard bound
on the step) or Levenberg–Marquardt (an adaptive one). Gauss–Newton assumes
the model is good; it does not check.

**Step 5 — is the problem mixed-scale?** Positions in metres next to angles
in radians, or a parameter with a wildly different magnitude, call for
Jacobi scaling — or for a per-block parameterization that puts them on
similar scales.

## 4. Reading the result

The solve's own report tells you when the choice was wrong:

<div class="apx-table">
<span class="apx-title">Symptoms and their causes</span>

| Symptom | Likely cause | Change |
|---|---|---|
| the factorization fails at iteration 0 | $H$ singular — an unconstrained direction | fix the gauge, or switch to QR |
| `max_iterations` reached, cost still falling | the model is hard, or the damping is too high | more iterations, or Dog Leg |
| cost falls then rises, steps rejected | the model is bad far from the solution | Dog Leg, or a better initialization |
| the solve converges in one or two iterations | the problem is nearly linear | Gauss–Newton, and a cheaper linear solver |
| the solve is fast but the answer is wrong | the residual is wrong | not a solver problem |

</div>

## 5. The interaction with the rest of the pipeline

Two decisions outside the solver change what the solver should be:

**Whitening.** A block registered with a noise model has its residual scaled
to standard deviations; a block that whitens internally must be registered with
`NoiseModel::null()`. Mixing the two over-weights a block, and no solver
setting repairs it (chapter [Noise Models](../core/noise.md)).

**Robustness.** A robust loss changes the objective but not the model. If the
data contains gross outliers that the loss cannot absorb — a coherent cluster
of mismatches — the fix is selection, upstream: RANSAC in the consumer crate,
an inlier mask, a gate. The solver's job is to minimize the objective it is
given.

## 6. Worked example

A visual-inertial window: 7 keyframes (105 DoF), 200 landmarks, an IMU factor
per pair. The landmarks are eliminable, but the IMU factor puts `state_0` and
`bias_0` in the same rows, so the eliminated set is **not** block-diagonal and
the Schur machinery refuses. The right answer here is not a different solver:
it is the dense marginalization path (Part IV), which summarizes the evicted
frame into a prior instead of eliminating it.

The same graph with the IMU factor replaced by a pose-only prior *is*
block-diagonal, and `ExplicitSparseSchur` then reduces the solve to 105
unknowns with the landmark steps recovered by back-substitution.

## 7. The catch

<div class="apx-warning">
<span class="apx-title">Warning — the cheapest solver is not always the fastest</span>

For a small problem, the sparse path's per-iteration bookkeeping can exceed a
dense factorization's cost. For a large one, the dense path is not an option
at all. The crossover depends on the problem's shape — how many blocks, how
local they are, how many variables each touches — so the crate ships a bench
suite rather than a rule, and the honest procedure is to measure the problem you
have.

</div>

<div class="apx-note">
<span class="apx-title">Note — every solver sees the same linearization</span>

Whichever linear solver runs, the residual, the Jacobian and the robust
correction are produced by the same linearizer. A solver choice changes the
arithmetic of one step, never the objective. That is why the choice can be
made late, and why the objective's correctness is a separate question from
the solver's efficiency.

</div>
