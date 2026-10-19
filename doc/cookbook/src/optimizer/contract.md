# The Optimizer Contract

Three optimizers ship in this crate — Gauss–Newton, Levenberg–Marquardt, Dog
Leg — and they differ in one thing: how they choose the step. Everything else
they share is the **contract**, and the contract is what a consumer of the
crate should know: what goes in, what comes out, and what the numbers in the
result mean.

## 1. The problem

An optimizer is handed a `Problem` and a configuration, and it mutates the
problem's variables until it decides to stop. Everything else — what a step
means, when a step is rejected, how the damping adapts — is internal.

The contract exists so that a *caller* can be written once and work with any
of the three.

## 2. The trait

```rust
pub trait Optimizer {
    fn optimize(&mut self, problem: &mut Problem) -> OptimizeResult;
}
```

One method, one mutable borrow of the problem. The optimizer owns no state
about the problem beyond what its configuration says; the problem owns the
variables.

`OptimizerType` names the three (`LevenbergMarquardt` is the default):

```rust
pub enum OptimizerType { LevenbergMarquardt, GaussNewton, DogLeg }
```

## 3. The configuration

Every optimizer's config carries the same skeleton:

| Field | Meaning |
|---|---|
| `linear_solver_type` | which `LinearSolverType` solves the (possibly reduced) system |
| `max_iterations` | the iteration cap |
| `cost_tolerance` | stop when the cost stops changing |
| `parameter_tolerance` | stop when the step stops changing |
| `gradient_tolerance` | stop when the gradient is small |
| `timeout` | wall-clock stop |
| `min_cost_threshold` | early exit once the cost is good enough |

Levenberg–Marquardt adds `damping`, `damping_min`, `damping_max` and the
update factors; Dog Leg adds the trust-region radius and its bounds; Gauss–
Newton adds `use_jacobi_scaling` and `min_diagonal`.

## 4. The result

<div class="apx-table">
<span class="apx-title">What comes back</span>

| Type | Contents |
|---|---|
| `OptimizeResult` | the alias every `optimize()` returns: `Result<SolverResult<…>, ApexSolverError>` |
| `SolverResult<T>` | `parameters`, `status`, `initial_cost`, `final_cost`, `iterations`, `elapsed_time`, `convergence_info`, `covariances` |
| `OptimizationStatus` | *why* the solve stopped: converged, the iteration cap, a timeout, invalid numbers, ... |
| `ConvergenceInfo` | the detail behind the status |
| `IterationStats` | per-iteration cost, step norm, gradient norm, elapsed time, accepted flag |
| `OptimizerSummary` | the run's aggregate: iterations, final cost, time |

</div>

The distinction that matters in practice: **"converged" is not "succeeded"**.
`OptimizationStatus` names the stopping reason, and the crate checks the
dangerous cases first — a non-finite cost, step or gradient ends the solve as
`InvalidNumericalValues` rather than continuing with NaNs.

The `covariances` field is `None` unless the configuration asked for it
(`compute_covariances: true`); when present it maps each variable to its
covariance **in tangent space** — a $6\times6$ matrix for an `SE3`, not a
$7\times7$ one.

## 5. Convergence

`check_convergence(&ConvergenceParams) -> Option<OptimizationStatus>` is the
shared criterion, and its order is deliberate:

1. **numerical validity** — a non-finite cost, step or gradient stops
   immediately;
2. **timeout**;
3. **the cost floor** (`min_cost_threshold`);
4. **gradient norm** below `gradient_tolerance`;
5. **parameter update** below `parameter_tolerance`;
6. **cost change** below `cost_tolerance`.

The cost that is compared is the **robust** cost the linearizer accumulated,
not $\tfrac12\|r\|^2$ — the number the user asked to minimize.

## 6. Applying a step

```rust
pub fn apply_parameter_step(variables, step, variable_order) -> f64
```

is the one place a step becomes motion. It slices the flat step vector
according to each variable's tangent dimension and applies it through the
variable's own `plus` — a right perturbation on the manifold. The returned
value is the step's norm, computed over the **free columns only**, so a
frozen coordinate cannot inflate it.

The rollback path is the same function with a sign: `apply_signed_parameter_step`
folds the sign into the per-variable copy it has to make anyway, so a rollback
does not allocate and fill a negated copy of the whole step vector.

<div class="apx-note">
<span class="apx-title">Note — the step is not the update</span>

The linear solver returns $\delta$ — a tangent-space step. The *update* is
$X \leftarrow X\,\mathrm{Exp}(\delta)$, which is not $\delta$ and not
$X + \delta$. The two are the same to first order and differ at second order,
which is exactly where a solve stops agreeing with its own model. The crate
applies the exponential; a consumer that adds the tangent vector directly has
introduced a bias that grows with the step size.

</div>

## 7. Worked example

```rust
let config = LevenbergMarquardtConfig::default()
    .with_linear_solver_type(LinearSolverType::ExplicitSparseSchur)
    .with_max_iterations(20);
let mut optimizer = LevenbergMarquardt::with_config(config);
let result = optimizer.optimize(&mut problem)?;
println!("{:?}", result.convergence);
```

The same call with `GaussNewton` or `DogLeg` swaps the step rule and nothing
else — the problem, the factors, the linearization and the result type are
identical. That is the point of the trait.

## 8. The catch

<div class="apx-warning">
<span class="apx-title">Warning — the optimizer cannot fix a bad problem</span>

A wrong residual sign, a factor whose Jacobian disagrees with it, a variable
with no constraint: the optimizer will happily minimize the objective it was
given and report a confident convergence. Every number in the result describes
the *solver's* state, not the correctness of the model. The checks that matter
are outside the crate: the residual's own test, the Jacobian's finite-difference
test, and — in a full system — the trajectory's agreement with something
independent.

</div>

<div class="apx-note">
<span class="apx-title">Note — the iteration cap is not convergence</span>

`max_iterations` reached with `ConvergenceInfo` reporting the cap is a
different outcome from convergence, and the crate keeps them distinct. A solve
that always hits the cap is a problem that is too hard for the configured
budget — raise the budget, or change the optimizer — not a solve that
succeeded.

</div>
