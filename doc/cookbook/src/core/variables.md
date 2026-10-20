# Variables on Manifolds

A pose is not a point in $\mathbb{R}^6$ that happens to satisfy orthonormality;
it is a point on a manifold, and a least-squares solver that treats it as a
vector will leave the manifold. This chapter is the crate's variable model:
the tangent space, the plus/minus operations every step uses, and the
constraints (fixed coordinates, bounds) that live on the variable rather than
in the factor.

## 1. The problem

A rotation has three parameters and lives on $SO(3)$; a pose has six and
lives on $SE(3)$. Adding a small vector to a rotation matrix is not a
rotation, and renormalizing afterwards is a projection, not a step. The
manifold formulation avoids the problem: the solver works in the **tangent
space** — a small Euclidean space attached to the manifold point — and the
step is applied by the manifold's own `plus`:

$$
X' = X \circ \mathrm{Exp}(\xi).
$$

Every Jacobian in this crate is with respect to that $\xi$.

## 2. The variable

<div class="apx-symbols apx-table">
<span class="apx-title">`ManifoldVariable`</span>

| Operation | Meaning |
|---|---|
| `plus(tangent)` | $X \circ \mathrm{Exp}(\xi)$ — the step |
| `minus(other)` | $\log(X^{-1} \circ Y)$ — the difference as a tangent vector |
| `get_size()` | tangent dimension (6 for `SE3`) |
| `to_vector()` / `from_vector()` | parameter-vector round trip |
| `update_variable(delta)` | apply a tangent step in place |

</div>

`get_size()` returning 6 for an `SE3` is worth internalizing, because the
variable *stores* 7 parameters. The *parameter* count and the *DoF* count
differ wherever a quaternion is stored, and the crate uses the two sizes for
two different things: a factor's `params` carry the **parameters** (7 values
for an `SE3`), while the step vector and the Jacobian's columns are indexed
by the **DoF** (6 for an `SE3`).

## 3. The manifold types

| `ManifoldType` | Element | Parameters | DoF |
|---|---|---|---|
| `RN(n)` | a plain vector | $n$ | $n$ |
| `SO2` / `SO3` | a rotation | 1 / 4 | 1 / 3 |
| `SE2` / `SE3` | a rigid transform | 3 / 7 | 3 / 6 |
| `SE23` | pose + velocity | 10 | 9 |
| `SGal3` | pose + velocity + time | 11 | 10 |
| `Sim3` | a similarity | 8 | 7 |

The layouts are fixed by `apex-manifolds` and every factor in this crate
assumes them: `SE3` is `[tx, ty, tz, qw, qx, qy, qz]`, `SE2` is
`[tx, ty, θ]`, `SE23` is `[ρ(3), θ(3), ν(3)]`.

<div class="apx-note">
<span class="apx-title">Note — `SE23` is a group, not a vector with a velocity</span>

`SE23` is the group whose element is $(R, t, v)$: composing two of them
composes their velocities too, and the tangent is $[\delta\rho, \delta\theta,
\delta\nu]$. That is why the IMU factor can compare two states with one
right-minus: the velocity's effect is part of the composition, not bolted on.
The neighbours' cookbook (`apex-manifolds`) derives the group law; this
chapter only records what a factor author needs.

</div>

## 4. Constraints on a variable

Two kinds of constraint live on the variable, and both are enforced by the
problem rather than by a factor:

**Fixed coordinates.** `fix_variable(key, idx)` freezes one tangent
component. The frozen components are removed from the step vector, so the
solver never proposes a motion in them and the step's norm reflects only what
can actually move. This is how a gauge is held: fix the first keyframe's
position and yaw, and the gauge freedom leaves the problem entirely rather
than being held by a stiff prior.

**Bounds.** `set_variable_bounds(key, lower, upper)` imposes box constraints
on the parameter vector; `apply_constraints_to_variables` clamps after a step.
Bounds are a blunt instrument — they apply in *parameter* space, so a bound on
a quaternion component is not a bound on an angle — and they exist for the
cases where a physical limit is natural (a scale factor, a time offset).

## 5. Covariance

A variable can carry a covariance: `get_covariance`, `set_covariance`,
`clear_covariance`. The solver does not use it; it is the place where a
marginal prior or a solved problem records what is known about the variable.
The chain of custody matters: a variable's covariance is *its* uncertainty,
while a residual block's noise model is the *measurement's* uncertainty. Mixing
the two is a classic error, and the crate keeps them in different types for
exactly that reason.

## 6. Worked example

A pose-graph node:

```rust
let mut var = ManifoldVariable::<se3::SE3>::new(SE3::new(t, q));
let tangent = SE3Tangent::new(rho, theta);
let moved = var.plus(&tangent);     // right perturbation
let delta = moved.minus(&var);       // back to the tangent, for a check
```

A step of $\xi = ([\rho], [\theta])$ is applied as $X\,\mathrm{Exp}(\xi)$; the
inverse operation recovers $\xi$ exactly (up to the branch cut, which the
manifold's own `log` handles), and the crate's factor tests use that round
trip to check their Jacobians.

## 7. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a left perturbation is a different Jacobian</span>

Every Jacobian here is with respect to the **right** perturbation
$X' = X\,\mathrm{Exp}(\xi)$. A factor that differentiates a left perturbation
$X' = \mathrm{Exp}(\xi)X$ produces a Jacobian that is *not* merely a
different convention: the two differ by the adjoint, and mixing them yields a
step that looks reasonable and converges in the wrong direction. When a factor
is ported from another library, check which side it perturbs.

</div>

<div class="apx-note">
<span class="apx-title">Note — the parameter vector is not the tangent vector</span>

`SE3` stores 7 parameters and has 6 DoF. A factor that writes a Jacobian
column count of **7** for an `SE3` block is wrong: the column count must
equal the variable's `dof()`, which is 6. The registration path enforces it
(`jacobian_shape().1` must equal the sum of the connected variables' DoF),
so the mistake is a typed error rather than a corrupted Jacobian. Meanwhile
the factor's `params` — the slices it reads to compute the residual — carry
the 7 *parameters*, and it is from those that it builds the pose.

</div>
