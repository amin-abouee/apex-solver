# Roadmap: deferred work after 1.6.0

Things we looked at during the 1.6.0 release and chose **not** to do in it, with
enough detail to pick each one up cold. Newest decisions first. When an item is
done, delete it here and put it in the changelog.

## 1. Static dispatch over the pose group (one factor, not one module per manifold)

**Principle (decided 2026-11-04):** where the mathematics works for several
groups, a factor is generic over a capability trait from
`apex_manifolds::capability` (`ActsOnPoints3`, `RigidPose`, `HasVelocity`,
`ConstDoF`), not duplicated in a module per group. Most already are (the LiDAR,
ranging, motion and GNSS-position factors, and the photometric and inverse-range
ones); `StereoFactor<G>` joined them in 1.6.0 (checked
against finite differences for `SE3`, `SE23`, `Sim3`, `SO3`). The rest below are
deferred. Use `factors::common::group_point` (`act_point`, `into_anchor`,
`expect_state_and_point`) and `factors::common::conformance`
(`finite_diff_jacobian_parity`, run over several groups as in
`StereoFactor`'s tests) as the pattern.

### 1a. The IMU twins (the real "module per manifold")

`imu::se23::{ImuFactor, CombinedImuFactor}` (+ `GravityFrameImuFactor`,
`ScaleGravityImuFactor`) and `imu::sgal3::{ImuFactor, CombinedImuFactor}`
duplicate each other: ~820 lines of factor code and ~730 lines of tests, plus
shared helpers in `imu/common.rs`, `imu/bias.rs`, `imu/types.rs`.

Why it is not a swap like `StereoFactor`:

- SGal(3) has a **time coordinate**: residual 10 rows (`[ρ, ν, θ, s]`) against 9
  (`[ρ, θ, ν]`), combined 16 against 15, and a different tangent order
  (`HasVelocity::VELOCITY_AT` is 3 for SGal3 and 6 for SE23).
- SGal(3)'s group law couples the left operand's velocity into translation, so
  `evaluate` needs a `correction_element` to cancel a spurious
  `v · (s_j − s_i)` term (see the module docs of `imu/sgal3/factors.rs`).
- Stable Rust cannot size a `SVector<f64, { G::DOF }>` from a trait constant, so
  the residual and Jacobian types have to be associated types.

Proposed design: an `ImuState` trait (implemented for `SE23` and `SGal3`) with
the residual dimension, the per-group `evaluate` (residual plus Jacobians) as an
associated function, and one `ImuFactor<G: ImuState>` / `CombinedImuFactor<G>`
that owns what is genuinely shared: block validation, bias split, noise
weighting, the `Factor` impl. The per-group math stays per group, as it must.
Keep the existing `se23` and `sgal3` test batteries running **unchanged** against
the generic types: they are the regression net.

Impact: breaking for apex-slam (`imu::se23::` appears ~18 times in its source).
Decide then whether to keep `imu::se23::ImuFactor` as a type alias for one
release (safer, but leaves the per-manifold module names in the API).

### 1b. Factors still tied to `SE3`

All are used only by the solver's own tests; no downstream repository calls them.

| Factor | What stands in the way |
|---|---|
| `DepthFactor` | `T_WS` could be `G`; `T_SC` is an `SE3` extrinsic, so it composes a `G` with an `SE3` (see `ExtrinsicProjectionFactor`) |
| `PoseRotationPrior`, `PoseTranslationPrior` | Could be `G: RigidPose` (`rotation()`, `position()`); the rotation and translation tangent blocks are guaranteed first by `RigidPose`'s layout. Their constructors take an `SE3` measurement today |
| `ExtrinsicProjectionFactor` | Needs a decision on composing a body pose `G` with an `SE3` extrinsic: `T_WC = X · T_BC` is an `SE3`, so `RigidPose` probably needs a `to_se3()`, and `∂T_WC/∂X = Ad(T_BC⁻¹)` padded with zeros beyond the first 6 tangent axes |
| `InverseDepthFactor` | **Hand-derived Jacobians** (`[R_j \| −R_j p̂_w]`). Re-derive through `into_anchor` / `act_point`, which is what the project rules ask for anyway |
| `SmartProjectionFactor` | Hand-derived Jacobians and a DLT triangulation that reads `rotation` and `translation`; uses `Vec<SE3>` |

Not candidates, and why: `EssentialMatrixFactor` / `EssentialMatrixConstraint`
(a relative pose between two cameras is an `SE3` by definition of the epipolar
geometry), `TimeOffsetProjectionFactor` (needs a velocity, so `HasVelocity`; it
builds an `SE3` from an `SE23` state), `PseudorangeFactor` / `DopplerFactor` /
`GpsVelocityFactor` (no pose group), `Relative4DofFactor` (`Rn(4)` nodes).

Source-compatibility note: a defaulted parameter (`G = SE3`) keeps the type name
valid in signatures, but a constructor call (`StereoFactor::<SE3>::new`) must
name the group, exactly as `ProjectionFactor` already requires.

## 2. A first-class window / fixed-lag smoother

1.6.0 ships the **building blocks** (`Marginalizer`, first-estimate Jacobians,
`Problem` variable and block removal, `update_values_from`, `AggregateFactor`)
and three examples (`incremental_pose_graph`, `fixed_lag_smoother`,
`sliding_window_ba`), but no smoother type: each example owns its window
bookkeeping (which blocks touch which pose, what to drop, when to marginalize).
`Problem` does not index blocks by variable, so every caller keeps that map.

Candidate: a `Window` helper that owns that index, a keyframe policy
(drop / marginalize / marginalize with FEJ, as in the example) and the
write-back. Build it only after a second real consumer shows the shared shape;
the examples are the spec.

## 3. Open quality-audit findings

`plan/quality/` holds the 2026-10-20 audit of apex-solver (48 findings, `S-*`);
`plan/quality/STATUS.md` says which are fixed, which are still open and which were
not re-checked at 1.6.0. The biggest open ones: one optimizer loop instead of
three (S-A1), default `cli` and `download` features (S-A8 / S-T1), the stale
"solved, not inverted" marginalization (S-C1), the negative-step undo (S-P3),
dataset-download integrity (S-E6) and parser fuzzing (S-E2).

## 4. Profiling instrumentation (`hotpath`)

`wip/solver-optimization` (a local branch, never pushed) carries an optional
`hotpath` feature: `#[cfg_attr(feature = "hotpath", hotpath::measure)]` on ~20
functions across the optimizer, linearizer and Schur solvers, an optional
`hotpath = "0.28"` dependency, and `examples/profile_hotpath.rs`. It produced
`doc/optimization/hotpath_profile_report.md`. It is **not** in 1.6.0: a new
optional dependency and attribute noise on hot functions is a decision for its
own change, and `samply` (see `CLAUDE.md`) already covers ad-hoc profiling. The
branch also keeps scratch notes (`.zcode/plans/`, `AGENTIC_OPTIMIZATION_PLAN.md`,
`agentic_optimization.md`) and 16 raw profiler outputs that were deliberately
left out. Delete the branch once this item is decided.

## 5. Smaller items

- **`Rn::DIM`, `Rn::DOF`, `Rn::REP_SIZE`** are still deprecated, not removed:
  `Tangent` and `LieGroup` declare the same constants (a `0` sentinel for `Rn`),
  so deleting only the inherent ones would leave `Rn::DIM` compiling and
  returning 0 without the warning. Removing them means reworking the trait
  constants.
- **rustdoc**: `cargo doc --workspace --no-deps --all-features` succeeds but
  emits ~50 warnings (broken intra-doc links such as `[m]` units read as links,
  stale `camera::PinholeCamera` and `super::se23::Se23ImuFactor` paths, links to
  private items). docs.rs tolerates them. Fix them and add
  `RUSTDOCFLAGS="-D warnings" cargo doc` to CI so they stay fixed. (Two in
  apex-manifolds were fixed in 1.6.0.)
- **`mdbook build`** is only run in CI (`docs.yml`); it was not run locally for
  1.6.0 because `mdbook` is not installed on the release machine (`mdbook-katex`
  and `mdbook-admonish` are).
- **Slow convergence on `intel`**: Levenberg-Marquardt (all three presets) and
  Dog Leg are still short of the optimum after 300 iterations where Gauss-Newton
  converges in 5; M3500, mit and sphere2500 are fine. See the `optimizers`
  example. Worth an issue and a look at the damping initialization.
- **Benchmarks**: `doc/performance.md` and `tools/benchmarks.md` still describe
  1.5.0 measurements. Re-run `odometry_pose_benchmark` and
  `bundle_adjustment_benchmark` and update both at the same time.
- **Icons** exist for apex-solver, apex-manifolds, apex-camera-models and
  apex-io only.

## 6. After 1.6.0 is published

- apex-cv, apex-slam, apex-calibration: the requirements already name
  apex-solver 1.6.0 and apex-manifolds / apex-camera-models 0.5.0; remove their
  temporary `[patch.crates-io]` tables, refresh apex-cv's tracked `Cargo.lock`,
  and re-run their gates against the published crates.
- Publish order: apex-manifolds, then apex-camera-models and apex-io, then
  apex-solver; tag `v1.6.0`.
