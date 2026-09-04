# Factor Reference

A factor is one measurement's contribution to the objective: given the
variables it connects, it produces a residual vector and its Jacobian. Apex
Solver ships about forty of them, grouped by the sensor they model.

$$
\min_{\mathbf{x}} \; \tfrac{1}{2}\sum_i \big\lVert \mathbf{r}_i(\mathbf{x}_i) \big\rVert^2_{\Sigma_i}
$$

## The contract

```rust
pub trait Factor: Send + Sync {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    );
    fn residual_dim(&self) -> usize;
    fn jacobian_shape(&self) -> (usize, usize);
    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> { Ok(()) }
    fn whitens_internally(&self) -> bool { false }
}
```

`linearize` writes into caller-owned buffers and allocates nothing — it runs
once per factor per iteration, in parallel across disjoint buffers. `params`
arrives as one zero-copy `&[f64]` slice per connected variable, in the order
the block was registered. `jacobian` is column-major with shape
`jacobian_shape()`, whose columns are the **minimal** (tangent) dimensions, not
the parameter dimensions: an `SE3` block is 7 parameters but 6 columns.

## Four conventions worth knowing

**Poses are body-in-world.** `T_wb` places the body in the world, so
`pose.translation()` *is* the body's world position and a body-frame direction
is $R^\top(\mathbf{p} - \mathbf{t})$. Some camera factors take a
world-to-camera pose instead; each says so.

**Jacobians are with respect to the right perturbation.** For a Lie group
variable the solver updates $X \leftarrow X \boxplus \delta = X \cdot
\mathrm{Exp}(\delta)$, so every Jacobian column is
$\partial \mathbf{r} / \partial \delta$ at $\delta = 0$. Take these derivatives
from `apex-manifolds` — every group reports the Jacobians of its own
operations (`act`, `compose`, `log`, `right_plus`, `right_minus`) through
optional output arguments. Re-deriving a block such as
$[\,R \mid -R[\mathbf{p}]_\times\,]$ by hand creates a second copy of a
convention that can silently drift from the group's.

**A factor either whitens internally or takes a `NoiseModel` — never both.**
Most return the raw residual and let the noise model supplied at registration
whiten it. A few cannot, because their weighting is bound up with the
measurement itself; they report `whitens_internally() == true` and must be
registered with `NoiseModel::null()`. Registering one with a noise model is
rejected rather than silently double-weighted.

**`validate_variables` is the real shape check.** `[profile.test]` inherits
`release`, so the `debug_assert!`s factors also carry are compiled out of
`cargo test`. Every factor implements this hook, so a mismatched registration
is a typed error instead of a panic inside the parallel assembly.

## Writing your own

Implement the trait, then test it in this order — the order matters:

1. **A hand-computed residual.** Pick a configuration whose answer you can work
   out on paper and assert it. This is the only test that pins your
   *convention*; everything below is self-referential without it.
2. **Finite differences on the Jacobian, away from the solution.** Perturb
   through the manifold's `right_plus`, not the raw parameters. Evaluate at a
   point where the residual is *not* near zero: residual comparisons pass
   through $J_r^{-1}$, which tends to the identity as the residual tends to
   zero, so a Jacobian checked only at the truth can be badly wrong and still
   pass.
3. **A solved graph.** `tests/factor_coverage.rs` has one scenario per factor,
   and a guard test that reads these modules' `pub use` lines and fails if an
   exported factor has no scenario.

Step 1 is not optional. A finite-difference test compares a Jacobian against
its *own* residual, so a residual that is self-consistently wrong passes it —
which is exactly how a `BearingRangeFactor` that rotated the wrong way, and a
`GpsAsyncFactor` missing gravity, both survived in this repo.

## Index

| Group | Module | Covers |
|---|---|---|
| [Pose & priors](./pose.md) | `factors::pose` | relative constraints, anchors |
| [Visual](./visual.md) | `factors::visual` | reprojection, stereo, epipolar, calibration |
| [IMU](./imu.md) | `factors::imu` | preintegration on SE₂(3) and SGal(3) |
| [LiDAR](./lidar.md) | `factors::lidar` | ICP, LOAM edge/plane, GICP |
| [GNSS & navigation](./navigation.md) | `factors::navigation` | GNSS, barometer, attitude |
| [Range & bearing](./ranging.md) | `factors::ranging` | UWB, radar, bearing-only |
| [Motion models](./motion.md) | `factors::motion` | ZUPT, ZARU, nonholonomic, planar |
| [Marginalization](./marginal.md) | `factors::marginal` | sliding-window priors |

Factors are addressed by their module path — `factors::visual::StereoFactor`,
`factors::imu::se23::ImuFactor` — so an import says which sensor a measurement
came from. `apex_solver::prelude` re-exports only the handful nearly every
program needs.
