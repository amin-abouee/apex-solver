# The Problem: Variables, Blocks, Assembly

`Problem` is the graph: the variables, the residual blocks registered against
them, and the machinery that turns the graph into a linear system. This chapter
is its API — what each call guarantees, which ones can fail, and the two
registration paths (panicking and typed) that decide how a mistake surfaces.

## 1. The problem

The graph is assembled once and solved many times. Between solves the
structure does not change; only the variable values do. So `Problem` holds
structure (keys, layouts, the symbolic sparsity pattern) and values (the
current estimate of each variable), and the optimizer mutates only the latter.

## 2. Construction and assembly mode

```rust
let mut problem = Problem::new(JacobianMode::Sparse);   // or Dense
```

`JacobianMode` selects the assembly strategy:

| Mode | When |
|---|---|
| `Sparse` (default) | large problems; symbolic structure, `faer::SparseColMat` |
| `Dense` | small-to-medium problems, roughly under 500 DoF |

`Problem::default()` is `Sparse`. The mode is not a performance knob alone:
the dense path builds the whole Jacobian as a `faer::Mat`, the sparse path
never materializes it.

## 3. Variables

```rust
let key: VarKey = problem.add_variable(ManifoldType::SE3, params);
```

`params` is the variable's **parameter vector** in the manifold's layout
(`SE3` → `[tx, ty, tz, qw, qx, qy, qz]`, 7 values for 6 DoF; `SE23` →
10 values for 9 DoF; `Rn(n)` → $n$). The key is a generational handle: a key
to a removed variable resolves to `None` rather than to another variable's
data.

Variables can be constrained:

- `fix_variable(key, idx)` — freeze one tangent component (index 0–2 is
  translation for `SE3`, 3–5 rotation);
- `unfix_variable(key)` — release them;
- `set_variable_bounds(key, lower, upper)` — box constraints, applied by
  `apply_constraints_to_variables` after a step.

Fixed coordinates are **excluded from the step vector** entirely, so a step's
norm is the norm of the motion actually applied (the crate's own note on
`apply_parameter_step` records this as the fix for a step norm inflated by
frozen coordinates).

## 4. Residual blocks

```rust
let block = problem.add_residual_block(&[a, b], Box::new(MyFactor{..}), None);
let block = problem.add_residual_block_with_noise(&[a, b], Box::new(MyFactor{..}), None, noise);
let block = problem.try_add_residual_block(&[a, b], Box::new(MyFactor{..}), None)?;   // typed
```

The arguments are the variables the factor touches, the factor, an optional
robust loss, and (in the `_with_noise` forms) the measurement's noise model.

Two registration paths exist on purpose:

- `add_residual_block` **panics** on an invalid registration. It is the
  ergonomic path for code whose validity is a compile-time-ish fact.
- `try_add_residual_block` returns `CoreResult`, and it validates more:
  - the factor's `validate_variables` hook runs, so a shape mismatch (a
    landmark whose parameter count disagrees with the factor's observations)
    is a typed error rather than a panic inside the parallel assembly;
  - unknown variable keys are rejected — otherwise they would be *silently
    skipped* during linearization, and the block would quietly measure
    nothing.

<div class="apx-warning">
<span class="apx-title">Warning — the panicking path is a footgun in a library</span>

`add_residual_block` is convenient and, in a library, hostile: a consumer's
bad factor crashes the consumer's process. Use the `try_` forms in anything
that builds a graph from configuration or from another crate's data, and keep
the panicking forms for statically-guaranteed registrations inside the crate
itself.

</div>

### What registration validates

<div class="apx-table">
<span class="apx-title">The three checks</span>

| Check | Failure |
|---|---|
| the noise model's dimension equals `residual_dim()` | `CoreError::DimensionMismatch` |
| an internally-whitening factor is registered with `NoiseModel::Null` | `CoreError::InvalidInput` |
| the factor validates against the variables it names | the factor's own message |

</div>

The second check is the crate's answer to the double-whitening bug that has
plagued every factor-graph codebase: `Factor::whitens_internally()` says
"this factor applies its own square-root information", and pairing it with a
non-null noise model is rejected at registration rather than silently
over-weighting the block forever.

## 5. Elimination, freezing, and the evaluation policy

Three structural operations the problem supports directly:

- `mark_for_elimination(key)` — opt a variable into the Schur path
  (`mark_as_schur_landmark` is the deprecated spelling; the new name says
  what it does, because elimination is not restricted to landmarks);
- `set_jacobian_evaluation(policy)` — choose the linearization policy
  (`JacobianEvaluation`, chapter [The First-Estimate
  Policy](./first-estimate.md));
- `freeze_linearization_point(key)` / `set_linearization_point` /
  `clear_linearization_point` / `is_linearization_point_frozen` — the FEJ
  machinery: a variable's linearization point can be pinned, so its factors
  keep reporting the Jacobian they reported when the prior was built.

The last group is what makes a sliding window consistent, and it is the reason
`Problem` holds *both* the current value and a possible frozen linearization
point per variable.

<div class="apx apx-example">
<span class="apx-title">Example — registering a block with a noise model</span>

```rust
let key = problem.add_variable(ManifoldType::SE3, pose_params);
let noise = NoiseModel::from_sigmas(&[0.05, 0.05, 0.02])?;
let block = problem.try_add_residual_block_with_noise(
    &[key],
    Box::new(BearingFactor { bearing, sigma }),
    Some(Box::new(HuberLoss::new(1.345)?)),
    noise,
)?;
```

The factor returns a raw bearing residual; the noise model scales it to
standard deviations; the loss then thresholds the *whitened* residual, so the
1.345 means 1.345 standard deviations in every block of the graph.

</div>

## 6. Solving and inspection

```rust
let result = LevenbergMarquardt::with_config(config).optimize(&mut problem)?;
```

The optimizer needs the problem mutable; the result carries the outcome
(`SolverResult`, `ConvergenceInfo`, `IterationStats`).

For diagnosis the problem can dump itself: `log_residual_to_file`,
`log_sparse_jacobian_to_file`, `log_variables_to_file`, and the direct
accessors `compute_residual_sparse`, `compute_residual_and_cost_sparse`,
`compute_residual_and_jacobian_sparse`, `compute_residual_and_jacobian_dense`.
The last two are the fastest way to see what a factor actually contributed
before blaming the optimizer.

## 7. Worked example

A pose graph with 300 keyframes:

```rust
let mut problem = Problem::new(JacobianMode::Sparse);
let mut keys = Vec::new();
for _ in 0..300 {
    keys.push(problem.add_variable(ManifoldType::SE3, pose_params));
}
for pair in odometry_edges {
    problem.add_residual_block(
        &[keys[pair.0], keys[pair.1]],
        Box::new(BetweenFactor::new(...)),
        None,
    );
}
```

Every registration is against known variables, so the panicking path is safe
here; a graph built from user configuration would use the `try_` forms.

## 8. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a block that references an unknown key measures nothing</span>

The typed path rejects an unknown `VarKey`; the panicking path rejects it too,
by panicking. But a *stale* key (a variable removed earlier in the build) is
the dangerous case, because the generational key can resolve to a *different*
variable than the one the author meant. The crate's rule is that a key's
identity survives removal: a key to a removed variable returns `None` from
`problem.variable(key)`, and the registration path treats that as an error
rather than as "not mine".

</div>

<div class="apx-note">
<span class="apx-title">Note — the problem is not a solver</span>

`Problem` does not choose a step, does not damp anything, and does not know
what a trust region is. It assembles; the optimizer decides. That separation
is why the same problem can be solved by Gauss–Newton, Levenberg–Marquardt or
Dog Leg without a change, and why a factor author never has to know which one
will run.

</div>
