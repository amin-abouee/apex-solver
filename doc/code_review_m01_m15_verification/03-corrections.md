# Corrections — claims that are wrong, mis-scoped, or mis-severitied

Six items. In each case the underlying concern is at least partly real; what is wrong is the
description, the root cause, the severity, or — in one case — the proposed remedy.

---

<a id="c1"></a>
## X-1 · `serde` is **not** an unused dev-dependency

**Review says** (m11-ecosystem.md, M11-H1):

> `serde` (declared in **both** dependencies and dev-dependencies; no use found in either)

**Actually:** the `[dev-dependencies]` entry is used.

```
benches/odometry_pose_benchmark.rs:85:use serde::{Deserialize, Serialize};
benches/odometry_pose_benchmark.rs:359:#[derive(Debug, Clone, Serialize)]
benches/odometry_pose_benchmark.rs:856:#[derive(Debug, Deserialize)]
benches/bundle_adjustment_benchmark.rs:68:use serde::{Deserialize, Serialize};
benches/bundle_adjustment_benchmark.rs:102:#[derive(Debug, Clone, Serialize)]
benches/bundle_adjustment_benchmark.rs:370:#[derive(Debug, Deserialize)]
```

Benchmarks are a dev-dependency consumer, so `Cargo.toml:167` must stay. Only the
`[dependencies]` entry at `:147` is dead.

**The rest of M11-H1 holds.** Confirmed unused in root `src/ bin/ examples/ benches/ tests/`:
`colored`, `serde_json`, `ureq`, `bzip2`, `num_cpus`, `rand`. `memmap2` is declared at
`Cargo.toml:33` and imported by **no crate in the workspace** — the review is right, and it
is worth removing precisely because it implies a memory-mapping capability that does not
exist (the same false claim appears in the ROS1 reader's module doc, M12-M1). `chrono` and
`tracing-subscriber` are used only in `src/logger.rs:110` — real, but legitimate.

**Net:** drop six deps plus workspace `memmap2`; keep `serde` under `[dev-dependencies]`.

---

<a id="c2"></a>
## X-2 · The `Rn` Jacobian bug has a narrower root cause than described

**Review says** (m09-domain.md, M09-H1) that both `jacobian_identity()` and
`zero_jacobian()` are at fault.

**Actually:** only `jacobian_identity()` propagates. In `between_factor.rs:183–196` all four
`zero_jacobian()` values are passed as `Some(&mut …)` out-parameters and **overwritten
before they are read** — `Rn::compose` assigns `DMatrix::identity(dim, dim)` (`rn.rs:236`),
`Rn::log` likewise (`rn.rs:258`). Their initial 3×3 shape never reaches the arithmetic.

The fault enters one level up, in the `LieGroup::between` **default implementation**
(`apex-manifolds/src/lib.rs:444–450`):

```rust
if let Some(jac_self) = jacobian_self {
    *jac_self = -result.inverse(None).adjoint();   // n×n — correct, adjoint() is dim-aware
}
if let Some(jac_other) = jacobian_other {
    *jac_other = Self::jacobian_identity();        // 3×3 — always, for Rn
}
```

`jacobian_identity()` is an **associated function with no `self`**, so it cannot consult the
runtime dimension. `between_factor.rs:204` then computes
`j_diff_wrt_k1_k0 (n×n) * j_k1_k0_wrt_k0 (3×3)` and nalgebra panics.

**Why this matters for the fix:** patching `zero_jacobian` alone changes nothing. The fix
must either reach the instance (a `&self` trait method) or restrict the bound. The review's
roadmap item 3.1 ("derive `Rn` Jacobian dimensions from actual dim") is directionally right
but has to target the associated-function signature to work.

**Also missed:** `Rn::apply_to_vector` (`rn.rs:286–292`) hardcodes `DMatrix::identity(3, 3)`
for `jac_self` with the same consequence.

---

<a id="c3"></a>
## X-3 · M05-H1's proposed fix was already tried and reverted as harmful

**Review says** (m05-type-driven.md, M05-H1):

> **Fix direction:** validate at `optimize()` entry when a Schur solver is selected
> (every RN/DOF-3 var deliberately classified)

**Actually:** the arch round implemented exactly this and backed it out. From
`doc/arch-round-results.md`, "Why SchurOrdering auto-detection is opt-in":

> The first attempt eliminated every unmarked `Rn(3)` variable eagerly. The benchmark probe
> caught it immediately: self-calibration represents intrinsics as `Rn(3)`
> (`[focal, k1, k2]`), and eliminating those as landmarks corrupted the Schur complement —
> Trafalgar RMSE stayed at its initial 2.963 px.

`SchurOrdering::auto_detect` (`explicit_schur.rs:100`) therefore defaults to `false`
(`:108`), and manual `mark_as_schur_landmark` marks always apply.

**The gap is still real** — nothing links `schur_landmark_keys` to
`LinearSolverType::SparseSchurComplement`, and a partially-marked problem silently produces
the wrong block structure. But "DOF == 3 implies landmark" is a false premise. A safe
remedy warns on *shape*, not on classification: when a Schur solver is selected and
`schur_landmark_keys` is non-empty, report how many DOF-3 variables were left unmarked and
let the user confirm, rather than reclassifying them.

---

## X-4 · `create_jacobi_scaling` is dead code, not a performance finding

**Review says** (m04-zero-cost.md M04-4, m10-performance.md M10-6): MEDIUM, "this path
remains **public API**", O(cols × nnz) quadratic scan.

**Actually:** the quadratic scan at `optimizer/mod.rs:484–503` is real, but the function has
**no production callers**. Every reference in the workspace:

```
src/optimizer/mod.rs:1354:  let scaling = create_jacobi_scaling(&jac)?;   // #[cfg(test)]
src/optimizer/mod.rs:1370:  let scaling = create_jacobi_scaling(&jac)?;   // #[cfg(test)]
```

The optimizers use the linear, parallel `AssemblyBackend::compute_column_norms`. So this
belongs with the hygiene sweep (deprecate or delete alongside `sparse_to_dense` /
`dense_to_sparse`, M01-L4), not with the hot-path work. Filing it as MED perf inflates the
performance backlog with an item that cannot move a benchmark.

---

## X-5 · Damping is implemented two ways now, not four

**Review says** (m15-anti-patterns.md, M15-H2): damping "is implemented **four different
ways**: linear find-loop over triplets (explicit Schur), λ·I matrix addition (Cholesky/QR),
dense diagonal in-place add (dense — the correct one), extra-triplet insertion (implicit)."

**Actually:** the perf round unified three of them onto
`NormalEquationsCache::damped_hessian`. Cholesky (`cholesky.rs:117`), QR (`qr.rs:116`) and
implicit Schur (`implicit_schur.rs:1043`) all call it; the dense solver keeps its in-place
diagonal add. Only explicit Schur's triplet loop (`explicit_schur.rs:1188–1205`) is still
divergent.

The structural argument survives — the drift the review predicted *did* happen — but the
remediation is now one site, not four, which changes its cost estimate substantially.

---

## X-6 · Every line reference in the review is stale

The audit is dated 26 Aug; `main` is 31 Aug, with three rounds of changes in between. Offsets
for the most-cited files:

| File | Shift | Example |
|---|---|---|
| `optimizer/levenberg_marquardt.rs` | **+135** | `to_owned()` :757 → **:892** |
| `optimizer/dog_leg.rs` | +33 | :1038 → **:1071** |
| `optimizer/gauss_newton.rs` | +12 | :514 → **:526** |
| `linalg/sparse/explicit_schur.rs` | +25 | :1079 → **:1104** |
| `apex-manifolds/src/rn.rs` | +9 | :310 → **:319** |
| `core/problem.rs` | varies (functions moved) | `set_variable_bounds` :115 → **:197** |
| `observers/visualization.rs` | −7 | :1860 → **:1853** |

Several cited snippets no longer exist verbatim at all — `let hessian = jt.mul(jacobians)`
(M01-H4) and the `lambda_i_triplets` block (M10-P3) were replaced by `ne_cache`. Use
[00-disposition-table.md](00-disposition-table.md) for current locations.
