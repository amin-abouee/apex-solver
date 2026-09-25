# Factor Correctness Audit & Potential-Issue Scenarios

> Audit date: 2026-09-25. Scope: every factor family in `src/factors/`, the
> noise/corrector math in `src/core/`, the manifold Jacobians in
> `crates/apex-manifolds`, and the hot-path Rust/optimization concerns found
> while optimizing the solver.
>
> **Method.** Each item below was checked against (a) the source, (b) the
> reference math (Ceres `corrector.cc`, GTSAM/OKVIS IMU formulation,
> Barfoot's SE(3) Jacobians), and (c) the existing test coverage. Items are
> graded: **CONFIRMED** (bug reproduced by reading/known-repro), **RISK**
> (unverified code path with a concrete failure mechanism), **VERIFIED OK**
> (checked, correct), **OPT** (optimization opportunity, not a bug).

---

## Part I — Correctness scenarios (math)

### S1. IMU end-boundary interpolation uses the already-mutated start sample — **CONFIRMED**

`src/factors/imu/preintegration.rs::integrate_from`. When both the window
start `t0` and the window end `t1` fall inside the *same* measurement
interval `[t_i, t_{i+1}]`, the start-boundary interpolation at line ~173
overwrites the local `omega_s_0`/`acc_s_0`, and the end-boundary
interpolation at line ~186 then computes
`omega_s_1 ← (1−r)·omega_s_0 + r·omega_s_1` between the *interpolated
start* and the raw end sample instead of the two raw samples. The blended
endpoint is only first-order-equal at `r` measured from the wrong base; the
integration weight `dt = end − time` is correct, so the error is bounded but
systematic for short windows (window ≪ measurement interval).

- Trigger: window shorter than the IMU period (fast re-linearization
  windows, sliding-window marginalization with 1-sample windows).
- Repro: the untracked WIP test `tests/imu_preintegration_oracles.rs`
  documents exactly this ("end-boundary interpolation reusing mutated start
  sample").
- Fix: keep raw `omega_s_0/acc_s_0` for the end interpolation (compute both
  interpolations from the raw samples), or clamp the window to sample
  boundaries.

### S2. IMU `try_new` / interval coverage is unchecked — **CONFIRMED (by WIP test)**

The same WIP test file notes `try_new` accepts a `[t0, t1]` whose
measurement span does not actually cover the interval: `delta_t()` reports
`t1 − t0`, but the integration loop silently stops at the last measurement
(`if next_time >= end { break; }` — fine) and `continue`s intervals before
`t0`. A window extending past the last loaded measurement produces a
`delta_t` larger than the integrated span with no error. Any factor using
`delta_t()` for the bias random-walk noise (e.g. `bias_random_walk_noise`)
then under-states the information for the un-integrated tail.

- Fix: in `new`/`append`, assert `last integrated time == t1` (or clamp and
  report the uncovered gap).

### S3. IMU factors self-whiten but do not override `whitens_internally()` — **CONFIRMED (API trap, verified)**

All four IMU factors multiply their own residual and Jacobian by the
preintegration's sqrt-information inside `linearize`
(`se23/factors.rs:210` and `:305`, `sgal3/factors.rs:261` and `:370`).
But none overrides `Factor::whitens_internally()` — while every *other*
self-whitening factor in the crate does (`navigation/gps.rs`,
`visual/depth.rs`, `visual/smart_projection.rs`, `ranging/bearing.rs`,
`lidar/{gicp,edge,distance_field}.rs`, `visual/homogeneous_point.rs`), and
`Problem::try_add_residual_block_with_noise` (`src/core/problem.rs:144`)
uses that override to **reject** attaching an external noise model to a
self-whitening factor. The IMU factors are therefore the one family where
`try_add_residual_block_with_noise(imu_factor, Some(noise))` silently
double-whitens (extra `S·S` on every such edge) instead of erroring.

- Fix: `fn whitens_internally(&self) -> bool { true }` on the four IMU
  factors (+ a registration-rejection test), making the contract uniform.

### S4. Accelerometer-noise covariance constant for the position block is unvalidated — **RISK**

`integrate_from`'s per-step covariance adds `k1(0,0) = ½·dt³·σ_a²·I` for the
position block (and `k1(6,6) = dt·σ_a²·I` for velocity). The velocity block
is empirically validated (`tests/imu_covariance_monte_carlo.rs`, ratio
1 ± 0.12), but **the position block is not**. Double-integrating white
acceleration noise gives `Var[Δp] = σ_a²·T³/3`; whether the per-step
`½dt³` recursion reproduces `T³/3` after N steps (vs `½·T³`, a 1.5× gap)
depends on the F-matrix coupling — the Monte-Carlo harness that already
exists covers only Δv. Extending that harness to Δp would settle it in one
run. A 1.5× sigma error here over-weights IMU position constraints ~1.5×.

### S5. Saturation multiplier magnitude (100× vs 100²×) — **RISK (verify vs OKVIS)**

`gyr_sat_mult`/`acc_sat_mult` are set to `100.0` when any sample exceeds
`g_max`/`a_max`, and applied *linearly* to the K-matrix noise entries
(`k0 = gyr_sat_mult·dt·I`), which are later squared by the outer `sgw2`
scaling — an effective ×100 noise inflation. In OKVIS the same `100.0`
constant multiplies the *sigma* (which squares to ×10 000 on covariance).
If OKVIS squares it, saturated samples here are under-inflated by 100× —
exactly the samples the mechanism exists to suppress. Needs a one-line
check against `okvis/kinematics/ImuError.cc` (or a hardware-run sanity
check on saturated data).

### S6. IMU F-matrix off-diagonal signs — **RISK (verify against OKVIS source)**

The FΔ blocks (`f_delta(0,9) = +dp_db_g_step`, `(0,12) = −c_integral_old·dt
+ ¼·c_mid·dt²`, `(3,9) = −dt·c_after`, `(6,9) = +dv_db_g_step`,
`(6,12) = −c_mid·dt`) match the OKVIS formulation *as far as sign
bookkeeping can be checked from the residual direction* (Δp/Δα/Δv ordering,
gravity sign `g = +9.81 on z` with `gc_i = T_i + v·dt − ½g·dt²`). The
wrongness of any single sign would show up as an asymmetric `p_delta` or a
non-monotone covariance — the symmetrization at the end of `integrate_from`
would mask it. A cheap check: run the existing Monte-Carlo harness with
*correlated* blocks enabled (compare full 15×15 empirical covariance, not
just Δv).

### S7. SE23/SGal3 first-order bias correction outside the linearization domain — **RISK (standard, bounded)**

`se23::evaluate` corrects the preintegrated delta with
`delta.right_plus(correction)` — the Forster-style first-order bias
correction. This is exact only to first order in `‖b − b_ref‖`; a large
bias jump between re-linearizations (fast motion, poor initialization)
makes the residual inconsistent with the reported Jacobian (the second-order
term that Forster's paper appendix and GTSAM's `ImuFactor::Evaluate`
second-order variant include is dropped). Standard practice, but the
solver offers no warning or bound check on `‖b − b_ref‖`. A guard
(warn when `‖db_g‖ > σ_g·3` or so) would surface silent degradation.

### S8. SGal3 `dalpha_db_g` sign convention vs SE23 — **VERIFIED consistent, but test it once**

`sgal3::ImuFactor` was audited as part of the recent timestamp fix
(`dbbf945` rejects non-interval-relative timestamps). The correction block
signs (`+dp_db_g·db_g − c_doubleintegral·db_a`, `−dalpha_db_g·db_g`,
`+dv_db_g·db_g − c_integral·db_a`) match SE23 exactly in
`se23::evaluate`. The two implementations were written against the same
reference, so a shared error would be invisible — a single numerical oracle
(se23 vs sgal3 on the same measurements with zero gravity offset and the
Galilean transform recovered) would tie them together.

### S9. BetweenFactor residual convention — **VERIFIED OK**

Residual is `Log(T_j⁻¹ · T_i · T_measured)` (via
`k1.between(k0).compose(measured)` then `.log()`), with Jacobians chained
through the manifold ops' own ∂-functions and finite-difference-verified in
`between.rs` tests (SE2, SE3, Rⁿ with variable dimension). The convention
differs from GTSAM's `Log(M⁻¹·(T_i⁻¹T_j))` by `Log(D) → −Log(D⁻¹)`-type
duality which is *not* a pure sign flip in SE(3) — but it is
self-consistent (residual and Jacobian come from the same chain), so any
problem built entirely with this factor is correct. Mixing factors from
different libraries against the same measurement would silently
mismatch — worth a convention note in the factor docs.

### S10. SE(3) Jr/Jl consistency — **VERIFIED OK (with a precision caveat)**

`right_jacobian` = `left_jacobian(−ξ)` block-for-block (the code computes
`q_block_jacobian_matrix(−ρ, −θ)` for Jr and `(ρ, θ)` for Jl, satisfying
`Jr(ξ) = Jl(−ξ)`), and `Jr·Jr⁻¹ ≈ I` is asserted with documented
precision bands (θ<0.01 → 1e-6; larger θ degrades). The Q-block closed
form uses Taylor coefficients `b = (θ−sinθ)/θ³`, `c = (1−θ²/2−cosθ)/θ⁴`,
`d = (c−3e)/2` with series fallbacks below the threshold — consistent with
the Barfoot Q-matrix expansion. Caveat carried in the docs: Jr⁻¹ precision
degrades for large rotations (documented 0.01 at large θ) — the
`ComputeStep` gradient-tolerance path and the IMU `dalpha_db_g` both consume
it, so large-rotation IMU windows carry that error into the Jacobians.

### S11. ProjectionFactor layout per optimization mode — **VERIFIED OK**

For all four `OP` flag combinations (POSE/LANDMARK/INTRINSIC on/off), the
column offsets written by `evaluate_internal` mirror `jacobian_shape`
exactly (pose block advances only when `OP::POSE`, landmark stride only
when `OP::LANDMARK`). The registered variable layout is
`[pose?, landmark?, intrinsics?]` in the same order. Checked each
combination by hand; `validate_variables` additionally enforces the
parameter sizes.

### S12. Non-cheirality invalid projection returns a free zero residual — **CONFIRMED (incentive bug, same class as the one already fixed)**

`evaluate_internal`'s fallback for camera-model errors *other than*
`PointBehindCamera` (line ~277) writes `residual = 0`, Jacobian rows = 0.
The doc comment on `write_cheirality_penalty` itself explains why a zero
residual is exploitable: a point that currently costs a large residual can
be driven *into* the invalid region, dropping its cost contribution to
zero — the optimizer is rewarded for creating invalid projections. The
cheirality penalty fixed this for `PointBehindCamera`; the other
singularity classes (e.g. atan/fisheye models at the principal axis, null
focal length) still have the hole.

- Fix: give every invalid class a constant-or-growing penalty (gradient
  optional — even a constant penalty with zero Jacobian removes the
  *incentive gradient* toward the singularity, though a smooth barrier
  before it is better).

### S13. Corrector (robust loss) — **VERIFIED OK against Ceres**

`Corrector` reproduces Ceres `corrector.cc` exactly: α solves
`½α² − α − (ρ''/ρ')·s = 0` with `α = 1 − √(max(1 + 2sρ''/ρ', 0))`;
`r̃ = √ρ'/(1−α)·r`; `J̃ = √ρ'·(J − (α/s)·r·(rᵀJ))`; cost `0.5·ρ(s)` (not the
corrected-residual norm — there is an explicit test for this). The
`ρ'' ≤ 0 ∨ ρ' ≤ 0 ∨ s = 0` guard correctly routes Huber's outlier region
(ρ'' = −δ/(2s√s) < 0) to the pure-reweighting path, avoiding the
`1−α = 0` division. Huber derivatives themselves verified: `ρ = 2δ√s − δ²`,
`ρ' = δ/√s`, `ρ'' = −ρ'/(2s)`.

### S14. Noise whitening — **VERIFIED OK; one OPT finding**

`NoiseModel::Diagonal` stores 1/σ (`from_sigmas`), so `r̃ = S·r`,
`J̃ = S·J` with S = √information is correct for both dense and diagonal.
`from_information` eigen-decomposes and takes √, clamping negative
eigenvalues with a warning. **OPT:** the `Dense` whitening path allocates a
temp `Vec` per call (per factor per iteration); a preallocated workspace
buffer or chunked in-place row transform would remove the allocation
(not on the BA hot path — BA uses `Null` — but on any Dense-noise odometry
edge it is one heap alloc per factor per iteration).

### S15. Jacobian-shape/landmark-stride assert — **VERIFIED OK (good hardening)**

`linearize`'s hard `assert_eq!(landmarks.len(), 3n)` (not `debug_assert` —
deliberate, `profile.test` inherits release) protects against the silent
stride-mismatch class; the comment documents why. Positive example of the
assert discipline this codebase should keep.

---

## Part II — Rust / optimization issues (from the performance program)

### S16. Dense noise whitening allocates per call — **OPT**
`NoiseModel::Dense` `whiten_residual`/`whiten_jacobian` allocate a temp
`Vec` per factor per iteration (§S14). Not on the BA hot path (`Null`
there), but any Dense-noise edge pays a heap alloc per iteration.

### S17. Residual buffer → `faer::Mat` copy in `assemble_sparse` — **OPT**
`Mat::from_fn(n, 1, |i, _| residual_buf[i])` copies ~3.6 MB per iteration
at t257 scale (~1–2 % of the iteration). Ownership-swap of
`workspace.residual_buf` (same pattern as `jacobian_values`) removes it.

### S18. `problem.variables.clone()` deep-copies every variable per solve — **OPT / documented**
`src/optimizer/mod.rs:604` deep-clones the `SlotMap<VarKey, Box<dyn
ManifoldVariable>>` (one heap allocation per variable, ~66k allocations at
t257) so the optimizer can mutate its own copy while the `Problem` keeps
the initial state. ~0.2 % per solve. Removing it changes the public
contract (who owns the final values) — needs a design decision, not a
drive-by fix.

### S19. `factor.linearize` dynamic dispatch per block — **accepted cost**
One vtable call per residual block per iteration (~1–2 ms total at t257
scale). An enum-based factor dispatch would remove it at a large API cost.
Documented as an accepted trade-off.

### S20. `free_local_columns` dynamic-dispatch per entry — **OPT (small)**
`build_symbolic_structure` and the (now-removed) scatter called
`variable.local_free_offset(col)` per column per row through the
`ManifoldVariable` vtable. Build-time only after Exp 004 (the scatter is a
precomputed plan), so this is now off the hot path — noted for completeness.

### S21. The Schur update stage is bandwidth-bound — ** architectural finding**
`compute_schur_complement`'s rank-`dof` updates (220 ms/iteration at t257,
the largest single stage) are bound by scattered 8-byte writes into a
buffer larger than L3. Verified by experiment: 16 threads splitting the
same DRAM traffic gained nothing (H2b, rejected). The only real fix is a
tiled accumulation (group landmarks by camera locality so updates hit
cache-resident tiles) — a substantial algorithmic project, sketched in
`AGENTIC_OPTIMIZATION_PLAN.md` §9.3.

### S22. `NormalEquationsCache::compute` clones the full JᵀJ value array — **OPT**
`hessian_values.clone()` per solve (one `nnz(JᵀJ)` memcpy, ~2–5 ms at
t257). Ownership lives in the published `SparseColMat`; restructuring the
publish path (own the buffer in the cache, hand out `SparseColMatRef`, or
swap two buffers) would remove it. Low priority, mechanical.

### S23. Chunked-variant partition borrow forced a clone — **code-health note**
Exp 006's `&mut self` solve-with-cholesky signature required cloning the
`SchurPartition` in the chunked path's closure (one clone per solve, small).
If the chunked path becomes hot again, refactor
`back_substitute_chunked`/`combine_updates` to take the partition
first to avoid the clone.

### S24. `load`-related measurement validity — **protocol, not code**
All post-change odometry/BA numbers taken while the desktop was active are
inflated 30–100 %+ (worst on the shortest benchmarks; torus3D +1550 %).
Every number quoted in this audit that was not explicitly flagged
"quiet-window" carries that caveat. The 2-pass warm protocol + interleaved
A/B are mandatory before accepting or rejecting anything timing-related.

---

## Part III — Verified-correct inventory (audit summary)

| component | file | verdict |
|---|---|---|
| BetweenFactor residual/Jacobian chain | `factors/pose/between.rs` | OK (FD-verified tests, SE2/SE3/Rⁿ) |
| PriorFactor (via shared chain) | `factors/pose/between.rs` | OK (identical chain) |
| SE(3) exp/log/Jr/Jl/Q-block | `apex-manifolds/src/se3.rs` | OK (Jr = Jl(−ξ) identity; precision bands documented) |
| ProjectionFactor layout & Jacobians | `factors/visual/projection.rs` | OK (mode matrix checked, S11) |
| Cheirality penalty (PointBehindCamera) | `factors/visual/projection.rs` | OK (rank-1 deliberate, gradient correct) |
| Non-cheirality invalid fallback | `factors/visual/projection.rs` | **S12 confirmed incentive bug** |
| Corrector (Triggs/Ceres) | `core/corrector.rs` | OK (matches Ceres, S13) |
| Huber derivatives | `core/loss_functions.rs` | OK (S13) |
| Noise whitening diagonal/dense | `core/noise.rs` | OK + S16 opt |
| SE23 evaluate (bias correction chain) | `factors/imu/se23/factors.rs` | OK structurally; S7 risk on correction domain |
| IMU preintegration (OKVIS F/K recursion) | `factors/imu/preintegration.rs` | S1 confirmed bug, S2 confirmed gap, S4/S5/S6 risks |
| IMU self-whitening + registration trap | `factors/imu/*`, `factors/mod.rs:214` | S3 API gap |
| Schur complement algebra (dense/explicit) | `linalg/sparse/schur/explicit.rs` | OK (cross-variant agreement test at 1e-6) |
| Damping `λ·diag(H)` with clamp | `Damping` | OK (diagonal clamps bounded, tested) |
| Predicted-reduction formula | `optimizer/levenberg_marquardt.rs` | OK (scale-consistency test from GitHub #57 fix) |

---

## Part IV — Recommended fix order

1. **S1 + S2** (IMU window boundaries): smallest change, real correctness
   impact for sliding-window users; the WIP oracle test file already
   specifies the expected behavior — finish it.
2. **S3** (`whitens_internally` overrides): four one-line overrides plus a
   registration-rejection test; converts a silent failure into a compile-
   time-visible contract.
3. **S4** (position-covariance Monte-Carlo): extend the existing harness;
   either validates ½dt³ or finds a 1.5× sigma error.
4. **S12** (invalid-projection penalty): mirror the cheirality penalty for
   the remaining error classes.
5. **S5/S6** (OKVIS cross-check): one-time source comparison.
6. **S16–S18** (allocations): mechanical, measure before/after.
7. **S21** (tiled Schur): the big one — see the plan document.
