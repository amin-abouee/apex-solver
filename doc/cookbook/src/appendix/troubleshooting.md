# Troubleshooting

Symptoms, their likely causes, and where to look. Every entry names the file
that answers the question.

## 1. Registration errors

<div class="apx apx-table">
<span class="apx-title">What the problem refuses</span>

| Symptom | Cause | Where |
|---|---|---|
| `CoreError::DimensionMismatch` at registration | the noise model's dimension ≠ the factor's `residual_dim()` | `core::noise.rs` |
| `CoreError::InvalidInput` about whitening | an internally-whitening factor registered with a non-null noise model | `core::problem.rs`, `factors/mod.rs` |
| a factor's own validation message | the variables' dimensions disagree with what the factor expects | `factors/common/validate.rs` |
| a panic at registration | the panicking path (`add_residual_block`) was used with an invalid registration | use the `try_` variants |
| the block measures nothing | an unknown `VarKey` was registered | the `try_` path rejects it with `CoreError::Variable` |
| a `DimensionMismatch` about the Jacobian shape | the factor's column count is not the sum of the blocks' DoF | `core/problem.rs` |

</div>

The registration checks exist because the alternatives are silent. A
double-whitened block is over-weighted; a mismatched block is skipped. Both
produce a confident, wrong answer.

## 2. Solve failures

<div class="apx apx-table">
<span class="apx-title">What the solve reports</span>

| Symptom | Likely cause | First thing to try |
|---|---|---|
| a factorization failure at iteration 0 | $H$ is singular — an unconstrained direction | fix the gauge with `fix_variable`, or switch to the QR solver |
| `max_iterations` reached, cost still falling | the problem is hard, or the damping is too high | more iterations, or Dog Leg |
| the cost falls, then rises, steps rejected | the model is bad far from the solution | Dog Leg, or a better initialization |
| `InvalidNumericalValues` | a non-finite cost, step or gradient | look for a NaN in a factor's Jacobian — a division by a near-zero residual |
| the solve converges in one iteration | the problem is nearly linear | nothing is wrong; consider Gauss–Newton |
| the cost oscillates | a robust loss and a bad model fighting each other | Huber for the early iterations, or better initialization |

</div>

## 3. Wrong answers

The hardest class, because the solve looks perfectly healthy.

<div class="apx apx-table">
<span class="apx-title">Symptoms of a wrong model</span>

| Symptom | Likely cause | Where to look |
|---|---|---|
| the trajectory is consistently offset or rotated | a frame convention: world-to-camera vs camera-to-world, or a lever arm | the factor's documentation; the conventions chapter |
| the solve converges, and the answer is wrong by a constant | a residual's **sign** | the factor's residual definition |
| the answer drifts with the problem's size | a wrong Jacobian that is close enough to be plausible | the finite-difference test |
| a marginal makes things *worse* | the sign of $b$ | `core/marginalization.rs` — the sign section |
| the prior pushes the state the wrong way after a window step | the marginal's offset sign, or a stale prior | `factors/marginal/marginal_prior.rs` |
| the estimate is over-confident in an unobserved direction | a re-linearized prior instead of a first-estimate one | `linearizer::JacobianEvaluation` |
| a block is over-weighted | a factor whitened twice | `Factor::whitens_internally` |

</div>

The order of suspicion is worth remembering: **frame, then sign, then
Jacobian, then the solver**. The solver is the last suspect, and the most
expensive one to check.

## 4. Performance

| Symptom | Likely cause | Remedy |
|---|---|---|
| the solve is slower than the benchmark suggests | the problem's shape differs — fill-in, not size | measure with `micro_kernels` |
| the dense path is used on a large problem | `JacobianMode::Dense` was chosen | switch to `Sparse` above ~500 DoF |
| memory grows with the number of kept variables | the explicit Schur's dense $S$ | `ImplicitSparseSchur` |
| the linearization dominates the runtime | every block touches a frozen variable | check the FEJ policy's scope |
| a window solve is slower than the sum of its parts | the aggregate factor is not in use | `AggregateFactor` |

## 5. Marginalization

| Symptom | Likely cause | Where |
|---|---|---|
| `MarginalizationError::Unobserved` | a dropped variable appears in no block | drop it instead of marginalizing it |
| `MarginalizationError::NoKeptVariables` | the absorbed factors touch nothing else | there is nothing to summarize |
| `MarginalizationError::SingularEliminatedBlock` | a marginalized variable is unobserved, or its observations are degenerate | drop the variable and its factors |
| `MarginalizationError::RankDeficient` | `SqrtInformation::Cholesky` on a rank-deficient $\Lambda_p$ | use the default `Eigen` variant |
| the covariance shrinks in an unobserved direction | the prior is being re-linearized | use the first-estimate policy |
| the window's trajectory changes when the marginal is rebuilt | the drift is large | rebuild more often — the model has aged |

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — "the solve converged" is not a diagnosis</span>

Every symptom in this chapter has a failure mode that *converges*. The
optimizer minimizes the objective it is given, faithfully, and reports a
confidence that belongs to the objective rather than to the world. The
diagnostic that actually separates the two is outside the solver: an
independent evaluation of the residual at the solution, and an agreement
check against something that does not share the factor's assumptions.

</div>

<div class="apx-note">
<span class="apx-title">Note — three tools that answer most questions</span>

1. `Problem::compute_residual_and_jacobian_sparse` — see what the graph
   actually produced.
2. `Problem::log_residual_to_file` / `log_sparse_jacobian_to_file` — see it
   in a form an editor can read.
3. `RerunObserver` — watch the solve as it happens (chapter
   [Observers](../tools/observers.md)).

</div>
