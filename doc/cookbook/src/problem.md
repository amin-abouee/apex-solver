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

## Residual blocks

A residual block binds one factor to the variables it reads:

```rust
problem.add_residual_block(&[k_from, k_to], Box::new(between_factor), loss);
```

- The factor implements `Factor::linearize(params, residual, jacobian)`.
- `loss` is an optional `Box<dyn LossFunction>`; see
  [Robust Loss Functions](./losses.md).
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
problem.mark_as_schur_landmark(point_key);
```

`SparseSchurComplementSolver` can also classify automatically from manifold
type and size when the ordering is explicitly opted in:

```rust
use apex_solver::linalg::SchurOrdering;

let ordering = SchurOrdering::default().with_auto_detect(true);
```

Auto-detection is **off by default**: `Rn(3)` is ambiguous between 3-D
landmarks and self-calibration intrinsics (`[focal, k1, k2]`), and eliminating
intrinsics as landmarks silently corrupts the Schur complement.
