# Capabilities: What a Group Can Do

A factor needs a group to do something specific: move 3-D points, read a rotation and a
position, carry a velocity. The crate states those needs as **capability traits**
(`apex_manifolds::capability`), so a factor can say `G: ActsOnPoints3` and work for every
group that qualifies, with static dispatch and no runtime check of the group's name. A
group that cannot do the job fails to **compile**, instead of failing at registration.

| Trait | What it promises | Implemented for |
|---|---|---|
| `ConstDoF` | the tangent dimension is a compile-time constant, `G::DOF` | `SO2`, `SE2`, `SO3`, `SE3`, `SE23`, `SGal3`, `Sim3` (not `Rn`, whose dimension is a runtime value) |
| `ActsOnPoints3` | `act` is a true action on 3-D points: `X⁻¹ ⊙ (X ⊙ x) = x` for every element and point | `SO3`, `SE3`, `SE23`, `Sim3` |
| `RigidPose` | a rigid pose whose tangent starts `[δρ, δθ, …]` under right perturbation: the translation moves as `t ← t + R·δρ`, the rotation as `R ← R·Exp(δθ)`, and every later axis leaves the point action untouched; offers `rotation()` and `position()` | `SE3`, `SE23` |
| `HasVelocity` | the tangent carries a velocity block `ν`, at `VELOCITY_AT`, next to the rotation block at `ROTATION_AT`; offers `world_velocity()` and `body_to_world()` | `SE23` (`VELOCITY_AT = 6`, `ROTATION_AT = 3`), `SGal3` (`3`, `6`) |

`ActsOnPoints3: ConstDoF`, and `RigidPose: ActsOnPoints3`.

## Why the exclusions are deliberate

- **`SE2` and `SO2`** act in the plane, not on 3-D points.
- **`Rn`** has a runtime dimension, so it has no `DOF` constant.
- **`SGal3`** is not `ActsOnPoints3`: its `act` adds `t·v`, which its inverse does not undo, so
  `X⁻¹ ⊙ (X ⊙ x) ≠ x`. It is `HasVelocity` instead.
- **`Sim3` and `SO3`** are not `RigidPose`: a scale axis scales the point action, and `SO3` has no
  translation, so neither fits the `[δρ, δθ, …]` layout that closed-form reprojection
  Jacobians are written against.

## Using them

```rust
use apex_manifolds::capability::ActsOnPoints3;

/// Works for SE3, SE23, Sim3 and SO3, with the group's own Jacobian columns.
fn jacobian_columns<G: ActsOnPoints3>() -> usize {
    G::DOF
}
```

A generic factor takes its Jacobians from the group instead of re-deriving them: `act`,
`inverse`, `compose` and friends each report their derivatives through optional output
arguments, in that group's right convention. `LieGroup::zero_jacobian_for(&self)` gives a
correctly sized zero matrix for the group at hand, which a generic caller needs because the
Jacobian type differs per group. This is how the solver's factors (`ProjectionFactor`,
`StereoFactor`, the LiDAR plane factors, `BearingFactor`, …) are written once and run on
several groups; the group-specific part is only the capability bound.

## `skew`

`apex_manifolds::util::skew` is the one public home of the skew-symmetric (cross-product)
matrix, `skew(v) · w = v × w`. The solver crates used to carry three copies.
