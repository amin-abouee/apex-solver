# Audit Fix Report

Branch: `fix/estimation_issue`. Scope: sanity-check every issue in `codex/issues/`
(an 11-issue third-party correctness audit) and GitHub issue
[#57](https://github.com/amin-abouee/apex-solver/issues/57) ("Minor issues discovered
in step quality"), fix everything confirmed, one commit per issue.

**Verdict: all 11 audit issues and all 3 GitHub #57 claims were confirmed still
present in the tree and have been fixed.** Nothing was found to be already-fixed or a
false positive. One additional latent bug (`chunk_eliminator.rs` Phase B skipping RHS
elimination for zero-free-DOF chunks) was discovered and fixed as a direct consequence
of ISSUE-0003 — it was unreachable before that fix and became reachable only once
fixed columns started being correctly excluded from the linear system.

Fixes were applied in risk order (Sim(3)/Barron/DCS first — silent corruption risk —
then fixed-DOF/bounds, then SE(3) family, then remaining losses, then
validity/SGal3/Jacobi), each as its own commit, verified with targeted tests plus a
full `cargo test --workspace --all-features --release` + `clippy -D warnings` + `fmt
--check` pass before commit. No commits were pushed.

---

## ISSUE-0002 — Sim(3) differential geometry (Critical)

- **Verdict:** Confirmed. `act()`, `adjoint()`, and `right_jacobian()`/`left_jacobian()`
  all had sign/structural errors that corrupt any Sim(3)-based solve (similarity-pose
  SLAM, scale-drift correction, loop closure with scale).
- **Root cause:** `act()`'s translation-perturbation block used `Matrix3::identity()`
  instead of `scale · R`, and its scale column omitted the leading `scale` factor.
  `adjoint()`'s rotation-coupling block had a spurious extra `scale` factor, and its
  scale column was left at zero instead of `-translation`. `right_jacobian()` reused a
  broken closed-form `q_matrix` that never actually depended on `sigma`.
- **Fix:** Corrected `act()`'s translation and scale blocks; corrected `adjoint()`'s
  rotation-coupling block and filled in its scale column. Replaced the closed-form
  `right_jacobian()`/`left_jacobian()` with numerical integration of the adjoint of the
  negated exponential (Simpson's rule, 256 intervals) — the standard, provably-correct
  construction for the right Jacobian of any matrix Lie group — removing the broken
  closed-form `q_matrix` entirely.
- **Files:** `crates/apex-manifolds/src/sim3.rs`
- **Tests added:** independent 4×4-matrix `vee()`/central-difference Jacobian tests
  (general, near-π, θ=0, small-angle cases), `act` FD test at scale≠1, adjoint-via-
  conjugation FD test, near-zero continuity test.
- **Commit:** `43b361d fix: correct Sim(3) action, adjoint, and Jacobian differentials`

## ISSUE-0006 — Barron general loss formula (Critical)

- **Verdict:** Confirmed. Reproduced the audit's claim directly: `alpha=-2, c=1, s=4`
  produced `rho=-1.6` — a negative robust cost, with both `rho'`/`rho''` signs wrong.
- **Root cause:** `rho'`/`rho''` were not algebraically consistent with the `rho`
  expression actually being returned, and the `alpha≈0`/`alpha≈2` special cases weren't
  continuous with the general branch.
- **Fix:** Re-derived `rho`, `rho'`, `rho''` symbolically from one consistent canonical
  form (`b=|α−2|`, `inner=s/(c²b)+1`, `rho=(b/α)(inner^(α/2)−1)`) with correctly-limiting
  special branches at `α≈0` (Cauchy-equivalent) and `α≈2` (L2). `AdaptiveBarronLoss`
  inherits the fix via delegation.
- **Files:** `src/core/loss_functions.rs`
- **Tests added:** `test_barron_general_loss_derivative_contract` (central-difference
  check across `alpha ∈ {2,1,0,-2,large negative}`), `rho(0)=0`, cost non-negativity,
  `test_barron_general_loss_continuous_at_shape_limits`.
- **Commit:** `c169f22 fix: correct Barron general loss formula and derivatives`

## ISSUE-0010 — DCS loss cost/gradient mismatch (High)

- **Verdict:** Confirmed. `DcsLoss::evaluate` let `rho(s)` decline back toward 0 past
  the saturation threshold `phi`, while `Corrector` (correctly) clamps the applied
  Jacobian/residual to zero once `rho'<0` — so the optimizer's modeled gradient froze
  while the actual scalar cost kept moving, breaking step-quality (predicted vs. actual
  reduction) evaluation for any factor with a large outlier.
- **Root cause:** DCS's own peak-then-decline cost curve was inconsistent with the
  corrector's existing zero-gradient-past-peak handling (same pattern already used
  correctly for Tukey's outlier branch).
- **Fix:** For `s > phi`, freeze `DcsLoss::evaluate` at the peak (`rho=phi, rho'=0,
  rho''=0`), turning it into a genuine plateauing saturating loss consistent with what
  the corrector already assumes.
- **Files:** `src/core/loss_functions.rs`, `src/core/corrector.rs`
- **Tests added:** `test_dcs_saturates_past_phi_instead_of_declining`,
  `test_dcs_cost_gradient_consistency_across_threshold` (numerical block-cost-vs-state
  derivative compared to `J̃ᵀr̃` on both sides of `s=phi`).
- **Commit:** `4df615d fix: make DCS loss cost consistent with its zero-gradient outlier region`

## ISSUE-0003 — Fixed DOF solved as free (High) + GitHub #57 predicted-reduction claim

- **Verdict:** Confirmed, and confirmed as the root cause of GitHub #57's
  predicted-reduction complaint. Variables with `fixed_indices` set (gauge-fixing,
  held poses/landmarks) still had their fixed columns allocated in the linear system
  and solved as free, silently perturbing values that should have stayed fixed and
  polluting `gradient`/`predicted_reduction`.
- **Root cause:** `build_variable_index_map` allocated columns using `.dof()` instead
  of a free-DOF count, and neither the dense/sparse linearizers nor the Schur block
  spans ever excluded fixed local indices from the assembled Jacobian.
- **Fix:** Added `free_dof()`, `local_free_offset()`, `expand_free_step()` as default
  `ManifoldVariable` methods. Column allocation, dense/sparse Jacobian scatter (kept in
  lockstep via a shared `free_local_columns` helper so the symbolic sparsity pattern
  can never desync from the value scatter), and all three `BlockSpan` construction
  sites now use free DOF. The solved free-only step is expanded back to full local
  tangent space (zeros at fixed indices) before `.plus()`. `Covariance::block`/
  `block_pair` rewritten to place explicit zeros at fixed-touching rows/cols instead of
  silently returning biased marginals.
- **Additional bug found and fixed in the same commit:** `chunk_eliminator.rs`'s Phase
  B was skipping the `eliminated_rhs` computation entirely for chunks with zero free
  columns (`data.cols.is_empty()`), not just the correction term — a bug that was
  unreachable before this fix (a fully-fixed landmark's chunk never used to have zero
  columns) and was caught by a Schur-variant cross-check regression
  (`schur_ba_agreement.rs`) once ISSUE-0003 made it reachable.
- **Files:** `src/core/variable.rs`, `src/optimizer/mod.rs`,
  `src/linearizer/cpu/dense.rs`, `src/linearizer/cpu/sparse.rs`,
  `src/linalg/dense/schur/explicit.rs`, `src/linalg/sparse/schur/explicit.rs`,
  `src/linalg/sparse/schur/implicit.rs`, `src/linalg/covariance.rs`,
  `src/linalg/sparse/schur/chunk_eliminator.rs`, `tests/bundle_adjustment_integration.rs`
- **Tests added:** `Rn(2)` fixed-index repro across GN/LM/DogLeg, fully-fixed variable,
  partially-fixed SE3 tangent, mixed fixed/free across two variables in one factor,
  dense/sparse agreement, Schur with a fixed kept pose and a fixed eliminated landmark
  cross-checked against plain Cholesky, step-norm/predicted-reduction vs. a hand-computed
  free-only value, covariance vs. a hand-inverted reduced Hessian,
  `chunked_elimination_handles_zero_dof_kept_block`,
  `chunked_elimination_handles_mixed_zero_dof_row_in_chunk`,
  `chunked_elimination_computes_delta_e_for_chunk_with_no_kept_coupling`.
- **Side effect (expected, not a bug):** `test_trafalgar_21_self_calibration` now
  needs 150 iterations instead of 50 — correctly eliminating the gauge-fixed pose's 6
  DOF makes the solve mathematically harder, matching the external GH#57 report's
  observation (38→67 iterations for its own repro).
- **Commit:** `38b1537 fix: eliminate fixed tangent columns from the linear solve`

## ISSUE-0004 — Variable bounds silently ignored (High)

- **Verdict:** Confirmed. Zero call sites outside the trait definition read
  `get_bounds()`; any bounds set via `try_set_variable_bounds` were silently discarded
  by every optimizer, so a "bounded" problem was actually solved unconstrained with no
  warning.
- **Scope decision (user-approved):** typed rejection rather than a full constrained
  optimizer, matching the audit's own recommended interim fix.
- **Fix:** `optimize()` now scans all variables up front and returns a typed
  `OptimizerError`/`CoreError::InvalidConstraint` if any variable has a non-trivial
  bound (`lower > -inf || upper < inf`), instead of silently solving the unconstrained
  problem. Also tightened `try_set_variable_bounds` to reject `NaN` bounds (previously
  passed the `lower > upper` check since NaN comparisons are always false).
- **Files:** `src/optimizer/mod.rs`, `src/core/problem.rs`, `src/error.rs`
- **Tests added:** `initialize_optimization_state_rejects_active_variable_bounds`,
  `initialize_optimization_state_allows_unbounded_interval`,
  `initialize_optimization_state_unaffected_without_bounds`, NaN/inverted-bounds
  rejection at `set_variable_bounds`.
- **Commit:** `ddfa2e5 fix: reject optimization when variable bounds are set but unsupported`

## ISSUE-0001 — SE(3)/SE_2(3) Q-block Jacobian coefficient (High)

- **Verdict:** Confirmed — this is exactly why
  `prior_factor_se3_jacobian_matches_central_difference` was checked in as `#[ignore]`d
  and failing (analytic `0.1038...` vs. FD `0.4412...`).
- **Root cause:** The `d` coefficient in the SE(3)/SE_2(3) left/right-Jacobian `Q`-block
  was `(c − 3.0) · e` instead of `0.5 · (c − 3.0·e)`, and the block was missing the
  `T²PT` term (only `TPT²` was present).
- **Fix:** Corrected the coefficient formula and added the missing `T²PT` term in both
  `se3.rs` and `se23.rs` (which duplicates the helper for position and velocity
  blocks); corrected the small-angle series constant to match. Un-ignored the
  previously-failing prior-factor test.
- **Files:** `crates/apex-manifolds/src/se3.rs`, `crates/apex-manifolds/src/se23.rs`,
  `src/factors/pose/prior.rs`
- **Tests added:** matrix-group central-difference Jacobian sweep (small-angle through
  near-π, nonzero translation) for both types; SE_2(3) position and velocity blocks
  verified independently.
- **Commit:** `97f8192 fix: correct SE3 and SE23 Q-block Jacobian coupling term`

## ISSUE-0005 — Robust loss derivative contract (High) + GitHub #57 Cauchy claim

- **Verdict:** Confirmed for Cauchy (matches GH#57's independent report — "rho is 0.5
  instead of 1.0"), Fair, Tukey, and Andrews.
- **Root cause / fix per loss:**
  - **Cauchy:** `rho` carried a spurious `/2.0` inconsistent with its own (correct)
    `rho'`/`rho''`. Removed the `/2.0`.
  - **Fair:** `rho'` was missing the `c` scale factor; `rho''` used the wrong sign and
    denominator. Corrected to `rho'=c/(2(c+x))`, `rho''=-c/(4x(c+x)²)` with `x=√s`.
  - **Tukey:** inlier-branch `rho''` carried a spurious `√s/c` multiplier. Removed it.
  - **Andrews:** `rho'` was missing the `c/x` factor and `rho''` was missing a term;
    added a small-`x` Taylor branch so `rho'(0)` correctly limits to `1/2` (it
    previously returned `0`).
- **Files:** `src/core/loss_functions.rs`
- **Tests added:** `test_robust_loss_derivative_contract_table` (table-driven central
  difference `rho'`/`rho''` vs. `rho` across log-spaced `s`, `s=0`, both sides of every
  branch threshold), `test_andrews_wave_loss_small_s_continuity`.
- **Commit:** `77e0bfe fix: correct Cauchy, Fair, Tukey, and Andrews loss derivatives`

## ISSUE-0007 — Non-finite loss constructor parameters accepted (Medium)

- **Verdict:** Confirmed for every parameterized loss except `DcsLoss` (already
  correct — used as the reference pattern).
- **Root cause:** Constructors used bare `<= 0.0` guards, which pass for `NaN`
  (comparisons with NaN are always false) and don't reject `+inf`/`-inf`.
- **Fix:** Added shared `finite_positive(name, value)`/`finite(name, value)`
  validators and applied them to every parameterized-loss constructor (Huber, Cauchy,
  Fair, GemanMcClure, Welsch, Tukey, Andrews, Ramsay, TrimmedMean, Lp, Barron,
  TDistribution); `AdaptiveBarronLoss` inherits the fix via delegation.
- **Files:** `src/core/loss_functions.rs`
- **Tests added:** `test_constructor_rejects_nonfinite_scale_parameters`
  (table-tested against `NaN`, `+inf`, `-inf`, `-0.0`, `0.0`, negative finite, valid
  finite for every constructor).
- **Commit:** `b3c2416 fix: validate loss constructor parameters are finite`

## ISSUE-0011 — L1/Lp origin discontinuity (Medium)

- **Verdict:** Confirmed. Both losses branch-and-spliced a linear approximation for
  `s < f64::EPSILON`, jumping several orders of magnitude in value/derivative right at
  that boundary — which corrupts step-quality evaluation for any residual that crosses
  the boundary during a solve.
- **Root cause:** A hard branch at an arbitrary threshold instead of a single smooth
  formula valid for all `s ≥ 0`.
- **Fix:** Replaced the branch with Charbonnier-style shifted-power smoothing
  (`rho(s) = 2·(√(s+ε) − √ε)` for L1, `rho(s) = (s+ε)^(p/2) − ε^(p/2)` for Lp, `ε =
  1e-12` internal constant, not new public API), C∞ at `s=0`, matching true L1/Lp for
  `s ≫ ε`.
- **Files:** `src/core/loss_functions.rs`
- **Tests added:** `test_l1_lp_continuous_across_old_guard_boundary`,
  `test_lp_norm_smooth_sweep_through_origin` (`p ∈ {0.5,1,1.5,2}`),
  `test_l1_robust_cost_continuous_across_guard_boundary`.
- **Test-suite note:** `test_loss_registry_resolves_every_canonical_name` previously
  asserted `rho'(0) ∈ [0,1]` for every registered loss — mathematically impossible to
  satisfy for a genuinely-singular-at-origin loss (true L1/Lp with `p<2` has infinite
  slope at `s=0`) while also keeping the smoothing bias negligible away from the
  origin. Special-cased `"l1"`/`"lp"` in that test to require finite+positive instead,
  with an inline explanation.
- **Commit:** `7f1f524 fix: smooth L1 and Lp loss near the origin to remove cost discontinuity`

## ISSUE-0008 — Manifold `is_valid` ignores non-rotation fields (Medium)

- **Verdict:** Confirmed for SE(2), SE(3), SE_2(3), SGal(3), Sim(3) — each checked only
  the rotation subobject's finiteness, letting `NaN`/`±inf` translation, velocity,
  time, or scale silently pass validation.
- **Fix:** Extended each `is_valid` to check every stored scalar: SE2 also checks
  `tx,ty`; SE3/SE_2(3)/SGal(3) also check translation (+ velocity for SE_2(3)/SGal(3),
  + time for SGal(3)); Sim(3) also checks translation and strengthens the scale check
  to `is_finite() && > 0.0` (previously bare `> 0.0`, which let `+infinity` through).
  `SO3::is_valid` also gained an explicit `!tolerance.is_finite()` rejection (an
  infinite tolerance previously made any quaternion, including NaN-containing ones,
  pass).
- **Scope note (as planned, not a silent gap):** `Problem::add_variable`/
  `create_variable` remain infallible in this pass — wiring `is_valid` into them would
  be a public-API break (every call site across examples/bin/io/tests) beyond what
  "fix `is_valid`" implies. Flagged here as a follow-up, not done in this fix set.
- **Files:** `crates/apex-manifolds/src/{se2,se3,se23,sgal3,sim3,so3}.rs`
- **Tests added:** `test_*_is_valid_rejects_nonfinite_at_every_index` for each of the
  5 composite types (every parameter index × `{NaN, +inf, -inf}`), plus
  invalid-tolerance rejection.
- **Commit:** `142f502 fix: check all state fields for finiteness in manifold is_valid`

## ISSUE-0009 — SGal(3) IMU absolute-time coupling (High)

- **Verdict:** Confirmed. The residual translation term contains an unremoved
  `s_i`-dependent component (`R_i⁻¹·(t_j − s_i·v_j)`), so a common timestamp-epoch
  shift changes the spatial residual whenever velocities differ — breaking any
  multi-keyframe chain on a real (non-zero-origin) clock. No runtime guard existed;
  the limitation was documented only in the module doc.
- **Fix (per the audit's own interim recommendation):** Added a typed rejection —
  `ImuFactor`/`CombinedImuFactor::validate_variables` now return a `FactorError` when
  `state_i.time()` is not interval-relative (nonzero beyond `1e-6` tolerance), instead
  of silently computing a corrupted residual.
- **Files:** `src/factors/imu/sgal3/factors.rs`, `src/factors/imu/sgal3/tests.rs`
- **Tests added:** `imu_factors_reject_absolute_state_i_time`; existing
  single-interval (`s_i=0`) coverage verified to keep passing unchanged.
- **Commit:** `5f9041b fix: reject SGal3 IMU factors with non-interval-relative timestamps`

## GitHub #57 — Jacobi-scaling gradient tolerance (new, not in the audit)

- **Verdict:** Confirmed. `ConvergenceParams.gradient_norm` was computed from
  `linear_solver.get_gradient()`, i.e. `J̃ᵀr̃` for the (possibly Jacobi-scaled) `J̃`
  actually handed to the solver — but the documented contract and
  `gradient_tolerance`'s calibration assume the unscaled `Jᵀr`. Most impactful for Dog
  Leg, where Jacobi scaling is on by default; matches GH#57's report of convergence
  decisions that don't match an unscaled reference run.
- **Fix:** Added `optimizer::unscaled_gradient_norm(gradient, scaling)` and used it in
  all three optimizers (`gauss_newton.rs`, `dog_leg.rs` — two call sites,
  `levenberg_marquardt.rs`) to un-scale the gradient by the per-column Jacobi diagonal
  before it's compared against `gradient_tolerance`, only when
  `config.use_jacobi_scaling` is set. All other internal uses of the (still scaled)
  gradient — steepest descent, Cauchy point, Hessian-vector product, predicted
  reduction — were deliberately left unchanged, since those legitimately operate in
  scaled space.
- **Files:** `src/optimizer/mod.rs`, `src/optimizer/gauss_newton.rs`,
  `src/optimizer/dog_leg.rs`, `src/optimizer/levenberg_marquardt.rs`
- **Tests added:** `test_lm_gradient_norm_matches_unscaled_jacobian_without_scaling`/
  `with_scaling`, `test_dl_gradient_norm_matches_unscaled_jacobian_without_scaling`/
  `with_scaling`, using a purpose-built `IllScaledFactor` to make scaled/unscaled
  gradient norms diverge sharply if the fix regresses.
- **Commit:** `2a1e9ab fix: compare gradient tolerance against unscaled gradient under Jacobi scaling`

---

## GitHub #57 cross-reference summary

| #57 symptom | Shared root cause with audit | Resolution |
|---|---|---|
| Predicted vs. actual step-quality mismatch on self-calibration BA | ISSUE-0003 (fixed DOF solved as free) | Fixed in `38b1537` |
| Cauchy loss magnitude off by 2× | ISSUE-0005 (Cauchy `rho` normalization) | Fixed in `77e0bfe` |
| Convergence decision disagrees with an unscaled reference run | New, not in the audit | Fixed in `2a1e9ab` |

## Verification

Every commit was preceded by its own targeted test run and followed, once all 12 were
in, by:

```
cargo fmt --all -- --check                                            # clean
cargo clippy --workspace --all-targets --all-features -- -D warnings  # clean
cargo test --workspace --all-features --release                       # 0 failures
```

(The only clippy output is a pre-existing, unrelated `binrw` future-incompatibility
warning from a dependency, not this codebase.)

## Commit list

```
43b361d fix: correct Sim(3) action, adjoint, and Jacobian differentials
c169f22 fix: correct Barron general loss formula and derivatives
4df615d fix: make DCS loss cost consistent with its zero-gradient outlier region
38b1537 fix: eliminate fixed tangent columns from the linear solve
ddfa2e5 fix: reject optimization when variable bounds are set but unsupported
97f8192 fix: correct SE3 and SE23 Q-block Jacobian coupling term
77e0bfe fix: correct Cauchy, Fair, Tukey, and Andrews loss derivatives
b3c2416 fix: validate loss constructor parameters are finite
7f1f524 fix: smooth L1 and Lp loss near the origin to remove cost discontinuity
142f502 fix: check all state fields for finiteness in manifold is_valid
5f9041b fix: reject SGal3 IMU factors with non-interval-relative timestamps
2a1e9ab fix: compare gradient tolerance against unscaled gradient under Jacobi scaling
```

None of these commits have been pushed.
