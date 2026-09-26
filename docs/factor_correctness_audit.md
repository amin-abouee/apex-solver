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
interpolation at line ~187 then computes
`omega_s_1 ← (1−r)·omega_s_0 + r·omega_s_1` between the *interpolated
start* and the raw end sample instead of the two raw samples. The blended
endpoint is only first-order-equal at `r` measured from the wrong base; the
integration weight `dt = end − time` is correct, so the error is bounded but
systematic for short windows (window ≪ measurement interval).

- Trigger: window shorter than the IMU period (fast re-linearization
  windows, sliding-window marginalization with 1-sample windows).
- Repro **(code inspection — see F7 in Part V: the test file this item used
  to cite never existed)**: the start interpolation at
  `preintegration.rs:173-174` overwrites the local `omega_s_0`/`acc_s_0`,
  and the end interpolation at `:187-188` then reads those *same locals* —
  both happen in one loop iteration exactly when the whole window sits
  inside a single measurement interval. The end value becomes
  `(1−r_end)·[(1−r_start)·ω₀ + r_start·ω₁] + r_end·ω₁` instead of
  `(1−r_end)·ω₀ + r_end·ω₁`, i.e. an error of
  `(1−r_end)·r_start·(ω₁ − ω₀)`.
- Boundaries of the defect (also verified): the locals are re-read from
  `measurements[i]` at the top of every iteration, so the mutation does not
  persist past that one iteration; `dt = end − time` is computed from
  `time`, which the start interpolation sets *before* the end
  interpolation, so the integration weight is correct. Only the blended
  endpoint value is wrong.
- The same pattern is duplicated in `propagation`
  (`:409-410` start, `:421-422` end).
- Fix: keep raw `omega_s_0/acc_s_0` for the end interpolation (compute both
  interpolations from the raw samples), or clamp the window to sample
  boundaries — in both functions.

### S2. IMU `new` / interval coverage is unchecked — **CONFIRMED (by code inspection)**

The constructor is `Preintegration::new(t0, t1, …)` — there is no `try_new`
and no fallible path at all — and it accepts a `[t0, t1]` whose measurement
span does not actually cover the interval, with nothing checking it after
the fact:

- `delta_t()` (`preintegration.rs:482`) reports `t1 − t0` unconditionally;
- the integration loop skips intervals before `t0` (`:162`) and stops at
  the last measurement with `if next_time >= end { break; }` (`:333-335`),
  so it integrates only the covered prefix;
- `integrate_from` returns a step count, but no caller compares it against
  the requested span (there are no external callers of
  `redo_preintegration`/`new` that do).

A window extending past the last loaded measurement therefore produces a
`delta_t` larger than the integrated span with no error. Any factor using
`delta_t()` for the bias random-walk noise (e.g. `bias_random_walk_noise`)
then under-states the information for the un-integrated tail.

- Fix: in `new`/`append`, assert `last integrated time == t1` (or clamp and
  report the uncovered gap).

*(This item previously claimed confirmation "by WIP test" — the file it
named never existed; see F7 in Part V. The finding itself is unchanged and
is supported by the lines above.)*

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

### S4. Accelerometer-noise covariance constants (position *and* velocity) are unvalidated — **RISK**

`integrate_from`'s per-step covariance adds `k1(0,0) = ½·dt³·σ_a²·I` for the
position block (and `k1(6,6) = dt·σ_a²·I` for velocity). Double-integrating
white acceleration noise gives `Var[Δp] = σ_a²·T³/3`; whether the per-step
`½dt³` recursion reproduces `T³/3` after N steps (vs `½·T³`, a 1.5× gap)
depends on the F-matrix coupling. A 1.5× sigma error here over-weights IMU
position constraints ~1.5×.

**Evidence correction (F7, Part V).** This item previously claimed that "the
velocity block is empirically validated (`tests/imu_covariance_monte_carlo.rs`,
ratio 1 ± 0.12)" and that "the Monte-Carlo harness that already exists covers
only Δv". No such file has ever existed in this repository — `grep -ri monte`
returns nothing outside this document, and `git log --all --diff-filter=D
-- tests/` shows no such path was ever deleted. So **neither block has been
validated**: the Δv claim, its `1 ± 0.12` ratio, and the "harness that
already exists" were all unsupported. Both blocks are open, and the work is
to *write* that harness (Δv first as the control — the simpler recursion —
then Δp), not to extend one.

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
would mask it. A cheap check: run a Monte-Carlo harness with *correlated*
blocks enabled (compare full 15×15 empirical covariance, not just Δv) —
note no such harness exists today, see the F7 correction under S4.

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

> **Status (2026-09-26): finding stands, fix REVERTED.** `1c4f56e`
> implemented exactly this (per-model `projection_deficit` → barrier on
> every non-cheirality error) and the criterion benchmark goldens rejected
> it: 5 of 8 BA ids regressed (trafalgar-257 and dubrovnik-135 initial cost
> 3.29e12 / 2.20e10 against 6.86e4 / 1.89e5 goldens, solve then made no
> progress). Reverted in `b2014ea`; evidence in Part V (F8) and
> `benchmarking_results/experiment_007_projection_domain_barrier_REJECTED.md`.
> The hole described above is therefore still open.

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
| Non-cheirality invalid fallback | `factors/visual/projection.rs` | **S12 confirmed incentive bug**; fix tried & reverted (`b2014ea`) |
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
   impact for sliding-window users; the expected behavior is spelled out
   under S1 (both boundary interpolations from the *raw* samples, in
   `integrate_from` and `propagation`). No oracle test exists yet — the one
   this item used to cite never did (F7) — so write it with the fix.
2. **S3** (`whitens_internally` overrides): four one-line overrides plus a
   registration-rejection test; converts a silent failure into a compile-
   time-visible contract.
3. **S4** (position-covariance Monte-Carlo): *write* the harness (Δv as the
   control first, then Δp); either validates ½dt³ or finds a 1.5× sigma
   error.
4. **S12** (invalid-projection penalty): **attempted, measured, reverted** —
   `1c4f56e` implemented it, the criterion benchmark goldens rejected it
   (5 of 8 BA ids regressed), `b2014ea` reverted it. See
   `benchmarking_results/experiment_007_projection_domain_barrier_REJECTED.md`;
   a retry needs a different formulation, not the same one.
5. **S5/S6** (OKVIS cross-check): one-time source comparison.
6. **S16–S18** (allocations): mechanical, measure before/after.
7. **S21** (tiled Schur): the big one — see the plan document.

---

## Part V — Round-2 fix log, measurements, and rejected work (2026-09-26)

Parts I–IV graded S1–S24 and recommended an order. Round 2 landed what
could be landed safely, measured it, and rejected one fix. This part is the
record: what changed, what it cost, what it proved, and what is still open.

### Method

- **Per fix:** implementation → full gate → one commit, explicit paths
  only. Gates on every change:
  `cargo fmt --all -- --check`,
  `cargo clippy --all-targets --all-features -- -D warnings`,
  `cargo test --release`.
- **Accuracy gate = the benchmark goldens.** `benches/*.rs` assert a pinned
  final cost inside *every timed run* of `odometry_benchmark` and
  `ba_benchmark`. That gate is not part of `cargo test` — which is how F8
  passed 707 unit tests and still broke 5 of the 8 bundle-adjustment ids.
  Any future change that touches a factor now runs it.
- **Performance gate = a criterion A/B** over the default `odometry` +
  `bundle_adjustment` targets (13 ids: 5 odometry, 8 BA — the implicit /
  `variants_for()` set). Both revisions were built once and interleaved per
  id (warm, warm, record, record), and the odometry suite was repeated with
  the order swapped as a noise control. Full protocol, all numbers and the
  verdict: `benchmarking_results/criterion_ab_round2_factor_fixes.md`.

### F1–F9: what landed

| id | scenario | change | commit |
|---|---|---|---|
| F1 | S3 (marginal-prior instance) | `MarginalPriorFactor` now overrides `whitens_internally()`: its `sqrt_info` is folded into residual *and* Jacobian inside `linearize`, so attaching an external noise model whitened the block twice; registration with a non-null noise model is now rejected | `1cf5413` |
| F2 | registration contract (the discipline S15 praises, applied to registration) | `validate_variables` checks each declared block dim against the registered variable's `dof`. `sqrt_info` is indexed by *dims* while the sparse scatter indexes columns by *dof*, so a mismatch silently landed Jacobian columns in the wrong place | `1cf5413` |
| F3 | registration contract | `jacobian_shape() == (residual_dim, Σ dof)` is enforced at registration. Previously only a `debug_assert` — stripped in release — so a hostile or incorrect factor could assemble a corrupted Jacobian | `1cf5413` |
| F4 | new in Round 2 (frozen-S container Jacobian) | `MarginalPriorFactor`'s constant-S Jacobian FD-checked on all 12 columns with central differences, and the drift budget measured instead of asserted in prose (table below) | `05e6af0` |
| F5 | S13 coverage gap | the derivative contract (ρ, ρ′, ρ″ finite, ρ ≥ 0, ρ′ ≥ 0, ρ′/ρ″ = real derivatives) swept over the **whole** loss registry, plus the `Corrector` identity `J̃ᵀ r̃ = ρ′·⟨r, J⟩ = d(robust_cost)` for every kernel — previously 4 kernels and one loss | `d19a97c` |
| F6 | S11 coverage gap | finite-difference pinning of `ProjectionFactor` across 7 perturbation modes (double-sphere intrinsics included), both parameter blocks and both residual rows; the cheirality barrier's FD test extended to the pose block | `1c4f56e` |
| F7 | **a defect in this document** | S1/S2/S4/S6 and Part IV cited test files that never existed; corrected below | this part |
| F8 | S12 | charge every non-cheirality projection-domain error → implemented, measured, **rejected and reverted** | `1c4f56e` → `b2014ea` |
| F9 | new (`left_plus::jacobian_self`) | returned `self.adjoint()`, which is the identity under *no* convention (not `I`, not `Ad(g)⁻¹`); now returns the identity the documented right-local convention implies, with a 473-line cross-group Jacobian-identity battery | `802c437` |

`f8680fc` (format-only sweep) landed first so the fix commits stay
reviewable. Test suite after Round 2: **703 apex-solver unit tests**
(707 − F8's four, removed with the revert), 166 + 10 doctests in
`apex-camera-models`, 474 + 5 + 10 in `apex-manifolds`, all 26 test
binaries green.

### F4 — the frozen-S drift budget, measured

`MarginalPriorFactor` holds S constant while recomputing the residual from
`theta(x − x0)` (GTSAM `LinearContainerFactor` semantics): the residual is
exact everywhere, `d theta / d delta = I` only at `x0`, and only the *step
direction* is approximated. Measured as the first block is displaced along
an all-six tangent:

| drift | relative error | cosine |
|---:|---:|---:|
| 0.00 | 0.0000 | 1.00000 |
| 0.02 | 0.0114 | 0.99994 |
| 0.05 | 0.0285 | 0.99959 |
| 0.15 | 0.0856 | 0.99635 |

Error is **linear** in drift (~0.57 per unit), not quadratic — the residual
value stays exact. The test now pins: exact at `x0` (1e-6), monotone growth,
a budget (< 2.5 % at drift 0.02, < 6 % at 0.05) that turns "rebuild when the
estimate drifts" into a checkable number, and cosine > 0.99 throughout so
the approximated step can never become an ascent step.

### F5 — coverage actually obtained

20 configurations (18 registry-resolved through `loss_from_name`, plus
LpNorm p=3 and Barron α=3, appended because no default kernel has positive
curvature) × 14 grid points = **280 samples**, asserting per sample that
ρ, ρ′, ρ″ are finite, ρ ≥ 0, ρ′ ≥ 0, and that ρ′/ρ″ are the actual
derivatives by central difference (except within 1e-4 of a declared
derivative cliff — huber, dcs, andrews, trimmed — where finiteness is still
required). Every kernel must leave ≥ 10 of 14 samples admissible (L2 leaves
13; the cliff-bearers leave 12), so a drifting grid cannot pass vacuously.
Measured: **21 samples** hit the ρ′ ≤ 0 suppression path, **26** the ρ″ > 0
alpha path, **235 / 280** produced a cost change large enough to
differentiate. **No kernel was changed — all eighteen pass as specified.**

### F7 — citations to test files that never existed

Round 2's own defect, in this document:

- **S1** cited "the untracked WIP test `tests/imu_preintegration_oracles.rs`"
  as the repro, and **S2** derived its **CONFIRMED** grade from "the same
  WIP test file". Neither that file nor any parent of it has ever existed
  (`git log --all --diff-filter=D -- tests/` shows no such path).
- **S4** stated "the velocity block is empirically validated
  (`tests/imu_covariance_monte_carlo.rs`, ratio 1 ± 0.12)" and that "the
  Monte-Carlo harness that already exists covers only Δv"; **S6** told the
  reader to "run the existing Monte-Carlo harness". `grep -ri monte` over
  the repository returns nothing outside this document, so **no
  Monte-Carlo harness exists at all** — the Δv validation, its ratio, and
  the harness were all unsupported, and both Δv *and* Δp are unvalidated.

Corrections made here: S1's repro is now derived from the source with exact
line references and the error term `(1−r_end)·r_start·(ω₁ − ω₀)` (and the
defect's boundaries — locals are re-read per iteration, `dt` is correct,
`propagation` duplicates it); S2 is downgraded to "CONFIRMED (by code
inspection)", with `try_new` corrected to the actual constructor `new` (there
is no fallible path); S4 now says *write* the harness rather than *extend*
one; S6 points at S4; Part IV items 1 and 3 no longer promise a file that
was never written.

**Not done:** creating `tests/imu_preintegration_oracles.rs` and
`tests/imu_covariance_monte_carlo.rs`. They are open work (Part IV 1 and 3),
not evidence — until they exist, S1/S2 rest on source inspection and S4
rests on nothing.

### F8 — implemented, measured, rejected

`1c4f56e` routed every non-camera error in `ProjectionFactor` through a
charged barrier, closing S12's free-zero-residual incentive. The mechanism is
sound (FD-verified, correct per-model domain geometry) and on venice-52 it
even improved the optimum (9.7167e4 → 9.4988e4) — but BAL's start pose puts
~3e4 observations out of domain, so the barrier was charged *at the initial
guess* and dwarfed the whole problem: initial cost 3.29e12 (t257) /
2.20e10 (dubrovnik-135) against goldens of 6.86e4 / 1.89e5, and the LM solve
then made no progress at all (final/initial = 1 − 1e-7). **5 of 8 BA ids
regressed, 1 improved, 2 unchanged; all 5 odometry ids bit-identical.**
Lowering the penalty 1000× scaled the initial cost by exactly 1000×
(Huber, cost linear in the penalty) and changed nothing about the outcome —
so this is an irreducible barrier cost at the initial guess, not a
conditioning artifact. Reverted surgically in `b2014ea` (only the
`Err(cam_err)` arm plus F8's four tests). Full matrix, the discriminating
experiment and a next formulation:
`benchmarking_results/experiment_007_projection_domain_barrier_REJECTED.md`.

Kept with the revert: F6's FD tests, the generalized
`write_projection_barrier` (bit-identical to the old
`write_cheirality_penalty` on the cheirality arm), and the camera-models
`projection_deficit` API — documented as **not currently charged**.
**S12 remains open.**

### Performance and accuracy verdict (criterion A/B)

**Interpretation used.** "Improve in all datasets or revert" was applied as
the repository's own acceptance rule from `benchmarking_results/baseline_criterion.md`:
a dataset counts as *not improved* — and forces a revert — only when it
**regresses beyond `max(2 %, 2× measured run-to-run noise)`**. Faster-is-good
and noise-band-is-neutral both count as no regression.

- **Accuracy: unchanged and green.** 52 + 20 criterion runs, `rc=0` on both
  revisions → every golden guard passed; initial *and* final costs are
  bit-identical pre ↔ post on all 13 ids. None of the golden datasets
  exercises a marginal prior, a dof / `jacobian_shape` mismatch, or
  `left_plus`, so the fixed defects cannot show up there — the fixes change
  what is *rejected* on hostile input, not the numerics of valid input.
- **Bundle adjustment (8 ids): max +0.965 %** (venice-52 sparse, disjoint
  CI) against a 2.00 % floor; 7 of 8 CIs overlap. Nothing regressed.
- **Odometry (5 ids, two order-swapped passes):** 8 of 10 measurements
  favour `post`. The only delta above 2 % — sphere2500 **+7.30 %** — has
  overlapping CIs in both passes (16.6 % CI half-width in pass 1, so its
  own floor that pass was 33 %) and **flips sign to −2.43 %** when the order
  is swapped: measured noise, not a regression. M3500 (−1.76 % / −1.91 %)
  and parking-garage (−6.96 % / −2.36 %) improve in both passes.
- **No fix is on the timed region** (registration, tests, docs, a reverted
  arm, and a manifold method with no caller in `src/` — see §1 of the
  results doc), so non-zero deltas are binary-layout and thermal effects.
- **Conclusion: nothing to revert beyond F8**, which the benchmark goldens
  rejected and `b2014ea` removed before this A/B ran.

### Status of every scenario after Round 2

| scenario | status |
|---|---|
| S1, S2 (IMU window boundaries) | **open** — fix not attempted; it changes preintegration numerics, so it needs its own golden/accuracy evidence and an oracle test |
| S3 (`whitens_internally`) | **partly fixed** — marginal prior done (F1); the four IMU factors still lack the override |
| S4, S5, S6 (IMU noise covariance / saturation / F-matrix signs) | **open** — S4's supporting evidence was corrected by F7; still needs the harness and an OKVIS cross-check |
| S7, S8 (bias-correction domain, SGal3 sign) | unchanged: bounded/standard, structurally consistent |
| S9, S10, S11, S13, S14, S15 | **verified OK**; F5/F6 closed the coverage gaps they named |
| S12 (free invalid projection) | **confirmed; fix attempted and rejected** — open, next formulation in experiment_007 |
| S16–S24 (OPT / architectural / protocol) | out of scope for a correctness-only round; S24 in particular still governs how any future timing is read |
| F7 (this document) | **fixed in this part** |
