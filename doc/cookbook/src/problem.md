# Problem Construction

## Variables and keys

Variables are stored in a
[`slotmap`](https://docs.rs/slotmap)-backed arena. `add_variable` returns a
stable, generational `VarKey`; keep the key and pass it wherever a factor needs
the variable. There are no string lookups on the hot path.

```rust
use apex_solver::{ManifoldType};
use nalgebra::dvector;

let pose = problem.add_variable(ManifoldType::SE3,
    dvector![tx, ty, tz, qw, qx, qy, qz]);
let point = problem.add_variable(ManifoldType::RN, dvector![x, y, z]);
```

Supported manifold types: `SO2`, `SO3`, `SE2`, `SE3`, `SE23`, `SGal3`, `Sim3`,
`RN`. Parameters are stored in the manifold's **representation size** (e.g. 7
doubles for SE(3)); optimization happens in the **tangent space** (6 DOF for
SE(3)) via a right-perturbation retraction.

That distinction runs through everything downstream: Jacobians have tangent
width, the step is applied with $\boxplus$ rather than addition, and the
linear system is built in tangent coordinates. See
[From Factor Graph to Linear System](./linearization.md).

## Residual blocks

A residual block binds one factor to the variables it reads:

```rust
problem.add_residual_block(&[k_from, k_to], Box::new(between_factor), loss);
```

- The factor implements `Factor::linearize(params, residual, jacobian)` — see
  [Factors](./factors/index.md).
- `loss` is an optional `Box<dyn LossFunction>`; see
  [Robust Loss Functions](./losses.md).
- To weight the block by its measurement uncertainty, use
  `add_residual_block_with_noise` and pass a
  [`NoiseModel`](./noise.md). Without one the block is unweighted, which is
  only correct when every residual in the graph is already in comparable
  units.
- Use `try_add_residual_block` for a `Result`-returning registration that runs
  the factor's `validate_variables` hook — shape mismatches are caught at
  registration time instead of during parallel evaluation.

## Gauge freedom

A pose graph is unconstrained up to a global rigid transform. Anchor it by
fixing DOFs of one pose:

```rust
for dof in 0..6 {
    problem.fix_variable(pose_keys[0], dof);
}
```

Alternatively, register a `PriorFactor` on the first pose.

## Schur landmarks

Schur-complement solvers eliminate landmark blocks. Mark the variables to
eliminate:

```rust
problem.mark_for_elimination(point_key);
```

Every Schur solver (`ExplicitSparseSchur`, `ExplicitDenseSchur`,
`ImplicitSparseSchur`) can also classify automatically from manifold type and
size when the ordering is explicitly opted in:

```rust
use apex_solver::linalg::SchurOrdering;

let ordering = SchurOrdering::default().with_auto_detect(true);
```

Auto-detection is **off by default**: `Rn(3)` is ambiguous between 3-D
landmarks and self-calibration intrinsics (`[focal, k1, k2]`), and eliminating
intrinsics as landmarks silently corrupts the Schur complement.

What elimination buys, and which Schur solver to pick, is in
[Linear Solvers & the Schur Complement](./solvers.md).
