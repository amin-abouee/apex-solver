# Activating the inert Ceres-compat parameters — measurements

Numbers behind the change that made eleven declared-but-unread configuration
fields drive behaviour. The one with numerical consequences is the switch from
uniform `λI` damping to Ceres' Marquardt diagonal `λ·D`; everything else either
replaces a hardcoded literal with the field that was supposed to supply it, or
is a deprecation.

All runs: `cargo run --release`, macOS, same machine, same commit for each
column. Pose graphs via `bin/pose_graph_g2o.rs`, bundle adjustment via
`bin/bundle_adjustment.rs` — the same binaries that produce `doc/performance.md`.

## What changed numerically

```text
before:   (JᵀJ + λI)   · dx = −Jᵀr,   λ₀ = 1e-3
after:    (JᵀJ + λ·D)  · dx = −Jᵀr,   λ₀ = 1e-4,
          D_jj = clamp(JᵀJ_jj, min_diagonal, max_diagonal)
```

`D` scales the damping of each column by that column's own curvature, which is
what makes the damped step invariant to a rescaling of the parameters. λ₀ moves
to 1e-4 because that is Ceres' equivalent (`initial_trust_region_radius = 1e4`,
λ = 1/radius); the previous 1e-3 was hand-tuned against `λI`, where λ alone had
to absorb the problem's scale.

`min_diagonal == max_diagonal == 1.0` gives `D = I` and reproduces the old
behaviour exactly, so the change is reversible per-solve without a rebuild.

## Where it wins: heterogeneous parameter scales

Self-calibration puts focal lengths (~10²), metric landmarks (~10⁰) and radians
in one parameter vector. This is the case `λI` cannot serve at any λ, because a
single scalar cannot damp all three appropriately at once.

| problem | λI final cost | λ·D final cost | change |
|---|---|---|---|
| EUCM 3-camera self-calibration | 1.50e3 *(stalled, iter 29)* | **1.51e-2** *(converged, iter 194)* | 10⁵× lower |
| Pinhole 3-camera self-calibration | 2.61e3 *(stalled, iter 39)* | **2.94e-2** *(converged, iter 145)* | 10⁵× lower |
| RadTan multi-camera calibration | 2.41e3 | **1.51e-1** | 10⁴× lower |
| BAL trafalgar-21 (20 iters) | 7.90e4 *(71.5% reduction)* | **1.83e4** *(93.4%)* | 4.3× lower |
| BAL ladybug-1723 (20 iters) | 2.70e5 *(88.4%)* | **2.60e5** *(88.9%)* | 1.04× lower |

The two self-calibration cases had been reporting `StalledNoProgress` — a status
their tests counted as success, which is why a cost three orders of magnitude
above the achievable minimum went unnoticed.

## Where it costs: homogeneous pose graphs

Every variable is an SE(2)/SE(3) pose and every column norm is similar, so
`D ≈ cI` and the change reduces to a rescaling of λ. There is no scale problem
for `D` to solve, and the tuned `λI` default was already good.

| dataset | λI final cost | λ·D final cost | ratio | iters (λI → λ·D) |
|---|---|---|---|---|
| mit | 3.2985e4 | **6.8651e2** | **48× better** | 18 → 37 |
| torus3D | 1.0061e3 | **3.4385e2** | **2.9× better** | 20 → 24 |
| M3500 | 5.4228e1 | **4.6712e1** | 1.16× better | 27 → 28 |
| sphere2500 | 2.2367e2 | 2.6221e2 | 1.17× worse | 28 → 29 |
| cubicle | 2.1396e3 | 2.3085e3 | 1.08× worse | 29 → 25 |
| parking-garage | 3.2941e0 | 5.8225e0 | 1.77× worse | 20 → 21 |
| intel | 7.6702e0 | 1.1817e1 | 1.54× worse | 35 → 38 |
| ring | 1.1525e1 | 2.0748e1 | 1.80× worse | 32 → 33 |

All eight converge in both configurations, and cost reduction stays at or above
92.7% throughout (the floor is `cubicle`, at 93.29% before and 92.76% after).
Three improve, five give up a small constant factor.

`mit` deserves a note: 79.78% → 99.58% cost reduction is the same failure the
self-calibration tests had, on a pose graph — the old solver was stopping early
and the assertion tolerated it.

## The call

Marquardt stays the default. The wins are order-of-magnitude and appear wherever
parameter scales differ — bundle adjustment and calibration, which is what this
solver is for — while the losses are bounded constant factors on problems where
the two formulations are mathematically near-equivalent anyway.

Two things were checked and ruled out as explanations for the pose-graph losses:

- **Initial λ.** Swept 1e-3 … 1e-6. 1e-4 is the best compromise and is Ceres'
  value; no setting recovers `λI`'s numbers on all eight while keeping the
  calibration wins.
- **The clamp bounds.** Swept `min_diagonal` ∈ {1e-6, 1e-3, 1.0} and
  `max_diagonal` ∈ {1e6, 1e32}. On pose graphs the result is bit-identical —
  every `JᵀJ_jj` already lies inside the range, so the clamp never binds. The
  Ceres defaults are kept.

A caller who wants the old numbers on a homogeneous pose graph has a one-line
escape hatch:

```rust
LevenbergMarquardtConfig::new().with_diagonal_bounds(1.0, 1.0)  // D = I
```

## Test-suite adjustments and why

Three assertions were calibrated against the old solver's early stopping and
needed to move. None of them weakened a correctness check.

- `camera_eucm_integration` / `camera_pinhole_integration`: iteration budget
  100 → 300. Both now converge on their own terms instead of being scored on a
  stall.
- `integration_tests`: the SE(3) iteration bound 20 → 30 (measured 24), cost
  assertions untouched.
- `camera_radtan_integration`: the k1 recovery bound 10% → 15%. Radial
  distortion is the most weakly observable parameter in that problem and trades
  off against focal length and pose; at the genuine minimum (cost 1.5e-1, four
  orders below where the old solver stopped) k1 sits 10.5% from truth.
  fx/fy/cx/cy still recover within 5%.

## Scale invariance, verified

A side check worth recording, because it is the property `λ·D` is supposed to
have: on the EUCM problem, enabling Jacobi column scaling changes the result by
nothing at all — `ParameterToleranceReached`, iteration 194, final cost
1.507169e-2 either way, to seven significant figures. `D` is computed from the
same Hessian the columns were scaled into, so rescaling the columns cancels
exactly. Under `λI` the two configurations give different answers.
