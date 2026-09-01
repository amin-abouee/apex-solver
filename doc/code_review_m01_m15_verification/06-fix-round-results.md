# Fix Round — results

Applied against `main` @ `9e193a0`. Every change was gated on
`benches/tools/sanity_check.sh`, which reports solver status, iteration count and final
cost per probe; **all three were bit-identical to baseline after every commit**, so the
performance deltas below carry no accuracy cost.

Machine: Apple Mac Mini M4 (10 cores), 64 GB.

---

## Measured result

Wall clock, median of 5, baseline `9e193a0` → `7117655`:

| Probe | before | after | Δ |
|---|---|---|---|
| ba/trafalgar/implicit | 600 ms | 571 ms | **−4.8%** |
| ba/trafalgar/explicit | 594 ms | 570 ms | **−4.0%** |
| ba/trafalgar/selfcal | 919 ms | 885 ms | **−3.7%** |
| odom/M3500/lm | 125 ms | 125 ms | ±0 |
| odom/sphere2500/lm | 744 ms | 745 ms | ±0 |
| odom/ring/dl | 10.0 ms | 10.6 ms | ±0 (noise; 10 ms probe) |

Odometry rows are solver-only time (`Optimization time:`), which excludes process start and
g2o parsing; the wall-clock harness numbers for those probes are dominated by I/O.

Separately, on ladybug (1723 cameras, 8k points, explicit Schur):

| Change | before | after | Δ |
|---|---|---|---|
| Drain-loop ordering ([NF-7](04-new-findings.md#nf7)) | 15.86 s | 8.69 s | **−45%** |
| H_cc damping ([C-3](01-confirmed.md)) | 17.35 s | 15.92 s | **−8.3%** |

Odometry is unchanged by design: it runs sparse Cholesky, which the perf round of 28 Aug
had already optimized. Every fix here lands on the Schur/BA path or on correctness.

---

## What was fixed

### Correctness
| Item | Change |
|---|---|
| [C-4](01-confirmed.md) | `Rn` between-Jacobian sized from the runtime dimension. Added `LieGroup::jacobian_identity_for(&self)`, defaulting to the associated function and overridden by `Rn`. `BetweenFactor<Rn>` panicked for every dimension ≠ 3; now covered by dimension and finite-difference tests at Rn(1,2,3,4,7). |
| [C-6](01-confirmed.md) | LM rejects a linear solver that does not match the problem's Jacobian mode, matching GN and Dog Leg. **This exposed a latent gap**: `covariance_available_for_every_linear_solver` claimed to cover all four solvers but built a sparse problem, so the two dense cases had been silently running sparse Cholesky. The test now pairs each solver with its mode and genuinely exercises the dense path. |
| [C-5](01-confirmed.md), M12-M1 | ROS1 chunk cache bounded to 2 payloads (LRU); `close()` + `Drop` added; the module's memory-mapping claim corrected. Test asserts the cache stays bounded and guards against going vacuous. |
| [C-14](01-confirmed.md#m06-1) | The gather loop returns `LinearizerError::Variable` for an unresolved key instead of silently shortening `param_slices` and letting the factor index past its end. |
| M05-M2, M05-L1 | `try_set_variable_bounds` / `try_fix_variable` return `CoreResult<()>`; inverted ranges and out-of-DOF indices are rejected rather than warned-and-dropped. The panicking wrappers follow the existing `add_*` / `try_add_*` convention. |
| M15-M2 | The two parallel loss-function tables in `pose_graph_g2o` collapsed into one, removing the `unreachable!()` that existed only to bridge them. |
| M06-L2 | Divide-by-zero guards on user-facing BA metrics. |

### Performance
| Item | Change |
|---|---|
| [NF-7](04-new-findings.md#nf7) | Schur complement drained in row-major order to match `s_dense`'s layout. **The round's largest win.** |
| [C-3](01-confirmed.md) | H_cc damping as a cached-diagonal edit instead of a linear `find` per camera index over a rebuilt triplet list. |
| [C-1](01-confirmed.md) | `residuals.as_ref().to_owned()` deleted in all three optimizers. |
| [C-2](01-confirmed.md) | Explicit-Schur H and g published by move; the `gradient` clone was dead outright. |
| [C-7](01-confirmed.md) | `system_hessian` removed entirely — the damped operator is threaded down the call chain instead of stored, so neither Schur path copies a `JᵀJ`. |
| [C-9](01-confirmed.md) | LM builds `InitializedState` once; the Schur path was paying for a full variable clone, symbolic build and residual/cost pass twice. |
| [NF-1](04-new-findings.md), M05-M1 | Projection factor borrows the landmark buffer and the camera. |
| M01-M4 | Rollback folds the sign into the per-variable copy instead of allocating a negated step vector. |
| [NF-4](04-new-findings.md) | Jacobian scatter buffer moved into `AssemblyWorkspace`. |
| [C-10](01-confirmed.md) | Visibility index rebuilt only when the sparsity fingerprint changes. |

### Hygiene
Unused root dependencies removed (`colored`, `serde_json`, `ureq`, `bzip2`, `num_cpus`,
`rand`, and the `[dependencies]` `serde` — the dev-dependency stays, the benches use it),
workspace `memmap2` dropped; `create_jacobi_scaling`, `process_jacobian`, `sparse_to_dense`
and `dense_to_sparse` deprecated with pointers to their replacements; commented-out
`IterativeState` deleted; `Sync` restored at the `Box<dyn Factor>` storage site;
`DdsError::{ThreadSpawn, RuntimeCreation}` split out of `ThreadJoin`; `ManifoldError`
converted to `thiserror`; the false `RefCell` multi-thread comment corrected.

---

## Estimates that measurement overturned

Recorded because the reasoning was wrong, not just the number.

- **[NF-1](04-new-findings.md) — I rated this HIGH.** ~14 M small allocations per Ladybug
  solve is a real count, but each is 24 bytes, the allocator handles that size well, and the
  factor's projection arithmetic dominates. Measured within noise. Kept as hygiene.
- **[C-7](01-confirmed.md) and [C-10](01-confirmed.md) measured *exactly* zero** — which is
  what led to [NF-6](04-new-findings.md#nf6): `implicit_schur.rs` is not on any production
  path. Code on a hot path does not respond to a removed `JᵀJ` copy with no change at all.
- **[C-11](01-confirmed.md) not fixed.** Sized it first: ladybug8k has ~907 k nnz, so the
  per-assemble pattern clone is ~7 MB, about **0.09%** of runtime. Avoiding it means
  reimplementing faer's argsort accumulation, including its duplicate-summing bit trick.
  Not worth the fragility.
- **M10-L1 not fixed.** `Mat::from_fn` with a trivial closure already lowers to a memcpy.

The pattern: the wins were where an operation had the wrong *complexity* or the wrong
*memory access order* (C-3, NF-7), not where it merely allocated.

---

## Not fixed — needs a decision

**[NF-6](04-new-findings.md#nf6): the matrix-free Schur solver is dead code.**
`IterativeSchurSolver` (~1500 lines) is constructed as a delegate and never read; both
`SchurVariant`s form the full explicit Schur complement. Either wire it up — matrix-free PCG
never forms `S`, so it sidesteps the O(cam²) buffer that dominates large-camera solves — or
delete it. Both are architecture decisions.

Also open: `enable_visualization` (M11-M1) is dead but correctly `#[deprecated]`; removing
public API belongs in a major version.

---

## Verification

`cargo test --workspace --release`: **1886 passed, 0 failed**.
`cargo clippy --workspace --all-targets`: clean.
Sanity check status/iterations/cost identical to baseline on all 7 probes.
