# Migrating from 1.4

`1.5.0` changes the public API. Code written against `1.4.0` will not compile
until you make the edits below. Full detail in the
[changelog](https://github.com/amin-abouee/apex-solver/blob/main/CHANGELOG.md).

## 1. `PriorFactor` is a tangent-space anchor

`PriorFactor` now anchors in the manifold's tangent space,
$r = \mathrm{Log}(T_{\text{prior}}^{-1} \circ X) \in \mathbb{R}^{\mathrm{dof}}$, and is
generic over the manifold — no quaternion double-cover ambiguity, no dropped
rotation–translation coupling, correct SE(2) angle wrap. The old ambient
parameter-space factor is renamed **`EuclideanPriorFactor`** and is restricted to
`Rn` variables at registration (anything else returns a `DimensionMismatch`
error). Struct-literal construction is gone:

```rust
// 1.4.0 — ambient
problem.add_residual_block(&[k], Box::new(PriorFactor { data: pose7 }), loss);

// 1.5.0 — tangent anchor on SE(3)
problem.add_residual_block(&[k], Box::new(PriorFactor::<SE3>::new(prior_pose)), loss);
```

For the old behaviour on an `Rn` variable, use
`EuclideanPriorFactor::new(data)`.

## 2. Feature flags: `rosbag`, `download`, `cli`

`apex-io`'s `rosbag` feature is now **opt-in** (it was built unconditionally).
Depending on `apex-io` no longer compiles `rusqlite` (bundled SQLite), `mcap`,
`zstd`, `lz4_flex`, `serde_yaml`, `byteorder`, `hex`. Bag I/O users must opt in:

```toml
apex-io = { version = "0.4", features = ["rosbag"] }
# via the solver crate:
apex-solver = { version = "1.5", features = ["rosbag"] }
```

`download` (dataset fetching) stays on by default; `--no-default-features`
disables it, in which case the `ensure_*_dataset` helpers only serve
already-downloaded files. `clap` is behind the default-on `cli` feature in both
crates; the `apex-solver` bins/examples likewise require `cli`. The `bag_*`
binaries need `--features rosbag`, `download_datasets` needs
`--features download`.

## 3. `LinearSolver::solve_augmented_equation` takes `&Damping`

The augmented system is now $(J^\top J + \lambda D)\,\delta = -J^\top r$ with
$D_{jj} = \mathrm{clamp}(J^\top J_{jj},\, d_{\min},\, d_{\max})$ — Ceres'
`LevenbergMarquardtStrategy`. Custom `LinearSolver` implementations must update
the signature:

```rust
// 1.4.0
solver.solve_augmented_equation(&residuals, &jacobian, lambda)?;

// 1.5.0 — same numerics as the old uniform λI
solver.solve_augmented_equation(&residuals, &jacobian, &Damping::identity(lambda))?;
```

## 4. Schur solver renames

The Schur solvers were consolidated around explicit/implicit names mirroring
Ceres. Old names are gone outright — no deprecated aliases:

| 1.4.0 | 1.5.0 |
|---|---|
| `SparseSchurComplementSolver` | `ExplicitSparseSchur` |
| `IterativeSchurSolver` | `ImplicitSparseSchur` |
| — | `ExplicitDenseSchur` (new) |
| `SchurVariant` | `ExplicitSchurVariant` (`Sparse`, `Iterative`, `Chunked`) |
| `LinearSolverType::SparseSchurComplement` | `LinearSolverType::ExplicitSparseSchur` |

`SchurPartition`, `EliminatedBlocks`, `SchurOrdering` and `SchurPreconditioner`
moved to the top-level `src/linalg/schur/` module (previously under
`src/linalg/sparse/`). `LinearSolverType` gained `ImplicitSparseSchur` and
`ExplicitDenseSchur` discriminants; the old `SchurVariant::Iterative`
special-case is gone.

## 5. `for_bundle_adjustment` selects `ImplicitSparseSchur`

`LevenbergMarquardtConfig::for_bundle_adjustment` now selects the matrix-free
implicit Schur solver instead of `ExplicitSparseSchur` — 2.2× faster across the
four BAL datasets. The step is inexact by construction; restore the exact
reduced solve with:

```rust
config.with_linear_solver_type(LinearSolverType::ExplicitSparseSchur)
```

**Numerics changed everywhere else too**: Levenberg-Marquardt now damps with
$\lambda \cdot D$ from $\lambda = 10^{-4}$ (was $\lambda I$ from $10^{-3}$),
and step acceptance in both LM and Dog Leg is gated on
`min_relative_decrease`. Iterates change on every problem. To restore the old
behaviour:

```rust
LevenbergMarquardtConfig::new().with_diagonal_bounds(1.0, 1.0).with_damping(1e-3)
```
