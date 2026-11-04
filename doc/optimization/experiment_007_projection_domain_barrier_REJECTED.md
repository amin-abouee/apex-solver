# Experiment 007: Charge non-cheirality projection-domain violations (REJECTED)

- **Date:** 2026-09-26
- **Target Category:** Bundle Adjustment + Visual factors (residual semantics)
- **Scope Type:** Correctness (optimizer incentive), evaluated for regressions
- **Status:** REJECTED — reverted in `b2014ea`
- **Introduced by:** `1c4f56e` (F8 of the factor-correctness audit)
- **Reverted at:** `b2014ea`

## 1. Rationale & Approach ("Why?")

- **Hypothesis:** `ProjectionFactor` returned a hard zero residual *and* zero
  Jacobian for every camera error other than an explicit
  `PointBehindCamera { z, min_z }`. Because `fov`, `bal_pinhole` and
  `double_sphere` report their cheirality failure as
  `ProjectionOutOfBounds`, for those models a behind-camera point was
  completely free — so the optimizer had a standing incentive to make a point
  invalid rather than fit it (a valid-but-grazing point can carry a large
  residual; pushing it past the boundary makes it zero). Audit scenario S12.
- **Implementation:** `CameraModel::projection_deficit(p_cam) ->
  (deficit, ∂deficit/∂p_cam)` was added to the camera-models trait (with
  z-forward default, plus `fov`/`double_sphere`/`bal_pinhole` overrides for
  their non-planar / z-backward domains). `ProjectionFactor` routed *every*
  non-cheirality error through `write_projection_barrier`, charging
  `CHEIRALITY_BASE_PENALTY (1e4) + CHEIRALITY_DEPTH_SCALE (1e3)·deficit` in
  both residual rows with a real gradient through pose and landmark.
- **Test coverage added:** 4 tests in `projection.rs`
  (`charged_domain_violation_{fov,double_sphere,bal_pinhole}`,
  `degenerate_projection_failure_gets_constant_barrier`) plus F6's
  finite-difference verification of the barrier Jacobian. All green.

## 2. Pros & Cons Analysis

- **Pros:** closes a real, documented incentive bug; the barrier math is
  correct (verified by finite differences) and the per-model domain geometry
  is right; on venice-52/sparse the fix genuinely *improved* the optimum
  (9.7167e4 → 9.4988e4) because free-riding observations stopped being
  discarded.
- **Cons:** the penalty must dominate any plausible in-image residual to do
  its job, i.e. it must dwarf the whole problem's cost — and with the BAL
  start pose containing ~3e4 out-of-domain observations the barrier is
  charged *at the initial guess*, before the optimizer has any chance to
  clear it. Result: initial cost 3.29e12 (t257) / 2.20e10 (dubrovnik-135)
  against goldens of 6.86e4 / 1.89e5, and the LM solve then makes no
  progress at all.

## 3. Criterion Benchmark Results

Both revisions built and run in one session (criterion test mode, one
execution per id, `RUST_LOG=debug` for the cost trace). Pre = `894b105`
(identical `src/` to the last pre-fix state), post = the F8 state.

| dataset / variant | pre-fix final cost | post-F8 initial | post-F8 final | verdict |
|---|---|---|---|---|
| trafalgar-21 / explicit_sparse | 1.370013056594e4 | 2.771703495085e5 | 1.370013056594e4 | unchanged (identical) |
| trafalgar-21 / explicit_iterative | 1.370020699431e4 | 2.771703495085e5 | 1.370020699431e4 | unchanged (identical) |
| trafalgar-257 / explicit_sparse | 6.863327074594e4 | 3.292363270052e12 | 3.292363e12 | **FAIL** (golden 6.863327e4) |
| trafalgar-257 / explicit_iterative | 6.799201774758e4 | 3.292363270052e12 | 3.292363e12 | **FAIL** (golden 6.799202e4) |
| venice-52 / explicit_sparse | 9.716652695390e4 | 1.552697117121e6 | 9.498807429930e4 | **improved** |
| venice-52 / explicit_iterative | 9.195236768263e4 | 1.552697117121e6 | 9.412961e4 | **FAIL** (golden 9.195237e4) |
| dubrovnik-135 / explicit_sparse | 1.888767976834e5 | 2.196853406152e10 | 2.192345e10 | **FAIL** (golden 1.888768e5) |
| dubrovnik-135 / explicit_iterative | 1.905530913190e5 | 2.196853406152e10 | 2.192345e10 | **FAIL** (golden 1.905531e5) |
| odometry M3500 / intel / parking-garage / sphere2500 / torus3D | — | — | — | all 5 bit-identical |

Summary: **5 of 8 BA ids regress, 1 improves, 2 unchanged; all 5 odometry
ids unchanged.**

- **Discriminating experiment:** lowering `CHEIRALITY_BASE_PENALTY`
  1e4 → 10 and `CHEIRALITY_DEPTH_SCALE` 1e3 → 1 scaled the t257 initial cost
  by exactly the expected 1000× (3.292363e12 → 3.294130e9) but changed
  nothing about the outcome — final/initial = 1 − 1.5e-8, still no
  convergence, still 6.8e4 golden unreachable. So this is **not** a
  conditioning artifact of large residuals in the normal equations; the
  barrier cost at the initial guess is simply irreducible within the solve
  (those observations cannot all be pulled back inside the domain from the
  BAL start pose).
- The F8 state was also confirmed to make **zero progress** on t257 even
  though LM reported success: relative cost change 1e-7 over the whole run,
  i.e. the trust region never accepted a useful step.
- The reverted state (`b2014ea`) reproduces the pre-fix costs **bit-identically
  on all 13 benchmark ids**, and `cargo fmt --check` / `clippy -D warnings` /
  `cargo test --release` (703 unit tests + 26 test binaries + camera-models
  166 + 10 doctests) are green.

## 4. Final Verdict & Next Steps

- **Decision:** REJECTED and reverted in `b2014ea`. The zero residual is
  restored for non-`PointBehindCamera` errors; S12's incentive issue stays
  open in the audit.
- **Kept:** the per-model `projection_deficit` API and its tests (correct
  domain geometry, documented as *not charged*), the generalized
  `write_projection_barrier` (bit-identical to the old
  `write_cheirality_penalty` on the cheirality arm), F6's finite-difference
  tests, and this record.
- **Next (if S12 is to be closed properly):** a formulation whose cost does
  not have to dominate the whole problem at the initial guess — e.g. a
  slack/relaxed (interior-point) treatment, or charging only observations
  that *cross* the boundary during the solve rather than ones that start
  outside it. Any such attempt must be validated against the full BA
  benchmark matrix in §3, not only against unit tests: the regression here
  was invisible to `cargo test` and only the bench golden guards caught it.
