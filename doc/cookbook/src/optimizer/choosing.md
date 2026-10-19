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

## 2. The decision procedure

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

## 3. Reading the result

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

## 4. The interaction with the rest of the pipeline

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

## 5. Worked example

A visual-inertial window: 7 keyframes (105 DoF), 200 landmarks, an IMU factor
per pair. The landmarks are eliminable, but the IMU factor puts `state_0` and
`bias_0` in the same rows, so the eliminated set is **not** block-diagonal and
the Schur machinery refuses. The right answer here is not a different solver:
it is the dense marginalization path (Part IV), which summarizes the evicted
frame into a prior instead of eliminating it.

The same graph with the IMU factor replaced by a pose-only prior *is*
block-diagonal, and `ExplicitSparseSchur` then reduces the solve to 105
unknowns with the landmark steps recovered by back-substitution.

## 6. The catch

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
