# apex-solver — performance findings

Where this crate sits in the stack: apex-vio's own profile
(`doc/runtime-optimization.md`, C7) found that **93 % of a keyframe solve is
inside this crate's Levenberg-Marquardt**, and a fresh run today on EuRoC
V1_03_difficult (release, Apple M4) gives, per frame:

| stage | ms/frame |
|---|---:|
| frontend (apex-cv) | 6.74 |
| solve (this crate's LM) | 5.79 |
| **marginalize (this crate's `Marginalizer` + a second reduction in apex-vio)** | **2.44** |
| reacquire / locate / other | 0.70 / 0.57 / 0.25 |

So two stages of the VIO frame, about 8 ms of roughly 17, run in this crate.
That is why the items below matter beyond solver benchmarks. Hot-path
linearization of the *factors* is already tight (analytic Jacobians, SmallVec
scratch, cached normal equations); what remains is per-solve overhead,
avoidable copies, dense-algebra choices, and the SGal3 group.

Measurement discipline for every item: a Criterion benchmark first
(`ba_benchmark` for BAL, `odometry_*` for pose graphs, a new window-shaped
benchmark for S-P1), benchmarks run strictly one at a time, and a downstream
byte-identical check on apex-vio EuRoC/Hilti trajectories.

---

## S-P1 · Per-solve setup is rebuilt from scratch on every `optimize` · P1 · L

**Evidence.** `optimizer/mod.rs:588 initialize_optimization_state`:
`problem.variables.clone()` (line 608, a deep clone of a
`SlotMap<_, Box<dyn ManifoldVariable>>`, one heap allocation per variable plus
two hash containers inside each), `apply_constraints_to_variables`, a sort of
the variable keys, a fresh symbolic Jacobian pattern
(`build_symbolic_structure`), a fresh `AssemblyWorkspace::build` (block order,
offsets, residual and Jacobian arenas) and one full residual evaluation. For a
sliding window the structure is identical from one keyframe to the next except
at its edges, yet apex-vio builds a new `Problem` and pays all of this every
keyframe.

**Unknown, to measure first.** How much of the 5.8 ms solve is setup versus
assembly versus the linear solve. Add per-phase timers to the existing
`OptimizerSummary` (debug level only) and a `window_solve` benchmark that
builds a 10-keyframe / 300-landmark / 1,400-observation problem.

**Proposal.**
1. `Problem::structure_version: u64`, bumped by `add_*`, `remove_*`,
   `fix/unfix`. A `SolverSession` owns the `InitializedState` pieces that
   depend only on structure (index map, symbolic pattern, workspace) and
   reuses them while the version is unchanged.
2. Replace the deep variable clone with a parameter snapshot: one contiguous
   `Vec<f64>` holding all variables' parameters (see S-A3 for the owner
   change), restored on failure. Variables then need not be `Clone` as boxed
   trait objects on the hot path.
3. Expose it as `optimizer.optimize_warm(&mut session, &mut problem)`; the
   existing `optimize` stays and builds a throw-away session.

**Gate.** Identical iterates (bitwise) between warm and cold solves of the same
problem; benchmark shows the setup term falling; apex-vio parity.

---

## S-P2 · Trial-step cost copies the whole residual and throws it away · P1 · S

**Evidence.** `core/problem.rs:796-821 compute_residual_and_cost_sparse_with_workspace`
returns `(Mat<f64>, f64)` and builds the `Mat` with
`Mat::from_fn(n, 1, |i, _| workspace.residual_buf[i])`: an O(n) allocation
and copy. `levenberg_marquardt.rs:1090` (and the equivalents in GN and DL) bind
it to `_new_residual` and drop it. It runs once per trial step. The same
function also collects per-block results into a `Vec<CoreResult<f64>>` before
summing.

**Proposal.** Add `Problem::cost_with_workspace(&self, vars, ws) -> CoreResult<f64>`
that evaluates into `ws.residual_buf` and returns only the cost. Keep the
existing function for callers that want the residual (implemented on top of
it). For the reduction, keep a fixed chunk order (as the factor code does
elsewhere) with `try_fold` per chunk, so the sum stays bitwise deterministic
and independent of thread count.

**Gate.** Bitwise equal cost; one fewer allocation per trial step (extend the
dhat budget test from S-P9).

---

## S-P3 · A rejected step is undone by applying the negative step · P2 · S

**Evidence.** `levenberg_marquardt.rs:1116 apply_negative_parameter_step`.
Mathematically `X ⊞ δ ⊞ (−δ) = X`, but in floating point it is not exactly
`X`; after many rejections the state drifts by rounding, and for quaternion
groups the intermediate normalization compounds it.

**Proposal.** Snapshot the parameters touched by the step (a `Vec<f64>` reused
across iterations; with S-P1 it is the same snapshot) and restore them on
reject. Also makes a rejected iteration exactly a no-op, which the L2/L3
rungs of apex-vio's `estimator_consistency` test depend on.

**Gate.** The existing re-solve and idempotence tests plus a new one: reject 100
steps and require bitwise-equal parameters.

---

## S-P4 · `ImuPreintegration::append` copies all samples on every call · P3 · S

**Evidence.** `factors/imu/preintegration.rs:171-185`: `self.measurements`
is `Arc<[ImuMeasurement]>`; `append` does `self.measurements.to_vec()`,
`extend`, and `.into()` on each call, O(n) per append and O(n²) over a stream.
Not on a current hot path: apex-vio calls `try_new` once per interval (grep
shows no call to `append` outside the crate's tests), which is why this is
P3, but `append` is public API meant for streaming.

**Proposal.** Hold `Vec<ImuMeasurement>` inside an `Arc` and use
`Arc::make_mut` (one copy only if shared), or keep `Vec` with `Arc` created
lazily when the factor is registered. Add a benchmark appending 1,000 samples
in steps of 10.

---

## S-P5 · SGal3 Jacobians by central differences · P2 · L

**Evidence.** `crates/apex-manifolds/src/sgal3.rs:660-710`: `right_jacobian`
and `left_jacobian` loop over 10 tangent components, and for each perturbs a
cloned `Vec` twice (3 `to_vec`/`clone` allocations per column), builds two
tangents, and computes `inverse ∘ compose ∘ log` for each (comment:
"by central differences through the crate's own compose/log"), step `1e-6`.
`exp(Some(jac))` (`:652`) calls it, and `right_jacobian_inv`/`left_jacobian_inv`
(`:716`, `:724`) invert the result. That is about 70 group operations
(three `exp`, two `compose`, two `log` per column, ten columns) and 30 small
allocations per Jacobian request, with an accuracy
ceiling near 1e-9 to 1e-10 relative (FD error plus the 1e-6 step).

apex-vio does not use SGal3 today, so the effect is on library users and on
the correctness of the SGal3 IMU factor's derivatives.

**Proposal.** Closed-form Jacobians. SGal3 is a semi-direct product
`SO(3) ⋉ (R³ × R³ × R)`; the right and left Jacobians have the same block
structure as SE₂(3) with an extra time column. Derive from the series
(`Σ ad^k/(k+1)!`), implement with the `SMALL_ANGLE_THRESHOLD` fallback used
by SE₂(3), and keep the FD code **in tests only** as the oracle (the crate
already has such oracles for Sim3). Remove the FD implementation afterwards.

**Gate.** The existing FD comparison tests (`sgal3.rs:1466-1487`) turn into
the oracle; tolerance tightens from the FD floor to 1e-12.

---

## S-P6 · Marginalization forms a dense JᵀJ and an explicit pseudo-inverse · P2 · M

**Evidence.** `core/marginalization.rs:120-390`: builds a dense
`total_rows x total_dof` Jacobian (every absorbed block), forms
`information = JᵀJ`, extracts four dense blocks, computes
`lambda_mm_inv = pseudo_inverse(&lambda_mm, 1e-12)` (explicit, `:297`) and
`Λ_kk − Λ_km Λ_mm⁻¹ Λ_mk`. apex-vio then runs a second dense reduction
(`backend/marginal.rs:185 reduce_out_dropped`) on the result. On V1_03 the
stage costs 2.44 ms per frame, at 320 landmarks and 1,385 observations before
marginalization. For landmark-heavy windows `Λ_mm` is block-diagonal; a dense
inverse ignores that.

**Proposal.**
1. Accumulate `Λ` and `g` directly per residual block into the dense
   `Λ` (an outer product of each block's Jacobian slices), as
   `NormalBlocks` already does for the normal equations; no `J` materialization.
2. Eliminate landmark blocks by their small diagonal blocks (inverse of
   `dof x dof` tiles), and only densify the kept part. Use the existing
   `SchurPartition`.
3. For the remaining dense block use an LDLT/eigen solve of `Λ_mm X = Λ_mk`,
   not an inverse (this also makes the code match its own comment, S-C1).
4. Let the caller ask for the final kept set up front so the second reduction
   in apex-vio is not needed (`Marginalizer::compute_keeping(&problem, eliminate, keep)`).

**Gate.** The linear-chain exact-equivalence tests in
`core/marginalization.rs` hold to the same 1e-6; apex-vio marginal parity
(`backend/marginal.rs` tests) and a trajectory comparison within 1e-9 (not
bitwise: the operation order changes, so this one needs the accuracy sweep
rather than a byte compare).

---

## S-P7 · Column scaling clones the Jacobian every iteration · P2 · S

**Evidence.** `linearizer/mod.rs:429-460`: `apply_column_scaling` copies the
value array (`to_vec`), rebuilds the symbolic structure via
`jacobian.symbolic().to_owned()` and returns a new `SparseColMat`; on the error
path it returns `jacobian.clone()`, silently dropping the scaling. Dense
version clones the matrix (`:510`).

**Proposal.** Scale in place on the owned Jacobian that the iteration already
owns (the preamble creates it), keep the symbolic structure shared (`Arc`), and
turn the `Err(_) => jacobian.clone()` branch into a typed `LinAlgError` (it can
only fail on malformed structure, which is a bug, not a fallback).

---

## S-P8 · Hash sets and maps on hot lookups · P2 · S

**Evidence.** `Problem.schur_landmark_keys: HashSet<VarKey>`
(`core/problem.rs:35`), looked up for every variable when building the
partition (`explicit.rs:496`, `:1147`); `SchurPartition` stores its maps as
`Vec<u32>` already. Fixed indices are `HashSet<usize>` per variable (S-A3).
`SchurPartition` is `.clone()`d on each Schur complement computation
(`explicit.rs:861`, `:1363`, `implicit.rs:735`): three `Vec<u32>` of length
`ncols` plus the block-span vectors, per iteration, to satisfy a borrow of
`&mut self`.

**Proposal.** `SecondaryMap<VarKey, bool>` or a bitset for the marks;
`Arc<SchurPartition>` (clone is a reference-count bump) or pass the partition
by reference after splitting `self` into disjoint fields. For BAL
dubrovnik-135 (about 270k columns) the clone is roughly 3 MB of memcpy per
iteration; small next to the solve, but free to remove.

---

## S-P9 · Allocation budgets exist for wrappers, not for the solver loop · P2 · M

**Evidence.** `tests/allocation_free_wrappers.rs` (added with the wrapper work)
and `tests/allocation_free_dense_solve.rs` guard two paths. apex-vio has
`tests/allocation_budget.rs` for its frontend. Nothing bounds heap blocks per
LM iteration, per `linearize` of the common factor families, or per
`Marginalizer::compute`.

**Proposal.** A `dhat` budget test per layer, each set from a measured run with
headroom and lowered when the number drops (the convention already stated in
apex-vio's test):
1. `linearize` of Between/Prior/Bearing/ICP/IMU factors: zero.
2. One LM iteration on a 20-pose pose graph after warm-up: a measured ceiling.
3. `Marginalizer::compute` on the window-shaped problem from S-P1.

Also fixes the repository's **flaky `a_warm_solve_of_the_same_shape_allocates_nothing`**
(fails at baseline `6e1e19d`; see S-T3): a counting test must be hermetic. Root
cause to confirm: dhat counts blocks from other threads (rayon pool start-up).
Run that test with `--test-threads=1` and pin the rayon pool size to 1 inside
the test.

---

## Not recommended (considered and rejected)

- **Enum fast-lane for built-in factors** (replace `Box<dyn Factor>`): was in
  the August review. With per-block work in the hundreds of nanoseconds and
  the dispatch dwarfed by the Jacobian math, the measured upside is small and
  it fights the open factor trait. Revisit only if S-P1 leaves dispatch visible
  in a profile.
- **GPU path**: the only trace left is `linearizer/cpu/mod.rs:4`, a doc
  comment linking `super::gpu`, a module that does not exist (a broken link
  to delete, see S-T7). Do not start a GPU path before the CPU session API
  (S-P1) is stable.
