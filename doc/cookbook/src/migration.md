# Migrating from 1.3

`1.4.0` changes the public API. Code written against `1.3.0` will not compile
until you make the edits below. Full detail in the
[changelog](https://github.com/amin-abouee/apex-solver/blob/main/CHANGELOG.md).

## 1. `Problem` uses handles instead of string names

`add_variable` returns a `VarKey`; `add_residual_block` takes `&[VarKey]` and
returns a `FactorKey` (previously `&[&str]` and `usize`). Keep the returned key
and pass it where you used to pass a name:

```rust
// 1.3.0
problem.add_variable("pose_0", ManifoldType::SE3, params);
problem.add_residual_block(&["pose_0", "pose_1"], factor, loss);

// 1.4.0
let k0 = problem.add_variable(ManifoldType::SE3, params);
let k1 = problem.add_variable(ManifoldType::SE3, params_1);
problem.add_residual_block(&[k0, k1], factor, loss);
```

If you need to look variables up later, keep your own `HashMap<YourId, VarKey>`.

## 2. `Factor::get_dimension` is renamed to `Factor::residual_dim`

Custom factor implementations must rename the method; there is no default
implementation.

```rust
// 1.3.0                              // 1.4.0
fn get_dimension(&self) -> usize      fn residual_dim(&self) -> usize
```

## 3. `OptimizationStatus` gained a `StalledNoProgress` variant

Exhaustive `match` expressions need a new arm. Treat it as a *successful*
termination — it means the solver reached a point where the cost can no longer
improve.
