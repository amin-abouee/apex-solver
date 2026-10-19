# Damping and Column Scaling

Damping is the knob every damped optimizer turns, and the crate makes it a
typed object rather than a bare `f64`, because the details matter: a scalar
damping applied to a system with mixed units is not the same as a diagonal one,
and neither is the same as a clamp. This chapter is the `Damping` type, the
two scalings, and when each is right.

## 1. The problem

The damped normal equations are

$$
\left(J^\top J + \lambda D\right)\delta = -J^\top \tilde r .
$$

$D$ can be:

- $I$ — Levenberg's original uniform damping;
- $\operatorname{diag}(J^\top J)$ — Marquardt's scaling;
- a clamped version of either.

The choice changes *which* directions the damping holds back, and therefore
what the step does.

## 2. Marquardt versus uniform

With $D = I$, the damping is the same absolute quantity in every direction.
Consider a problem with a translation in metres and a rotation in radians: the
rotation's column has a much smaller numerical magnitude, so a uniform
$\lambda I$ barely damps it at all, and the step in that direction is
essentially undamped. With $D = \operatorname{diag}(J^\top J)$, the damping
is *relative* to each column's own curvature, and a step of one "column unit"
is damped equally in every direction.

Marquardt's scaling is what Ceres and most modern implementations use, and
it is the crate's default (`Damping::identity` is the uniform variant,
available for compatibility).

## 3. The clamp

`Damping` carries the scalar **and** a clamp range for the diagonal:

<div class="apx-table">
<span class="apx-title">`Damping` fields</span>

| Field | Default (Ceres) | Meaning |
|---|---|---|
| `lambda` | — | the damping parameter; must be finite and non-negative |
| `min_diagonal` | $10^{-6}$ | lower clamp on the damping diagonal |
| `max_diagonal` | $10^{32}$ | upper clamp on the damping diagonal |

</div>

The lower clamp exists because a column with *little* curvature would
otherwise receive no damping at all: its diagonal entry is near zero, and
adding $\lambda\cdot 0$ to it damps nothing, leaving an unconstrained
direction free to move. The upper clamp keeps one very stiff column from
dominating the damped system.

The constructor validates: a negative or non-finite $\lambda$, or a
non-positive or non-finite bound, is `LinAlgError::InvalidInput`.

## 4. Jacobi scaling

Separate from damping: `use_jacobi_scaling` (Gauss–Newton's config) and the
optimizer's own `jacobi_scaling` normalize the Jacobian columns by their
$L_2$ norm *before* solving. This is preconditioning, not damping: it
changes the metric of the step, not the system's conditioning. The crate
documents the cost as 5–10 % overhead, which is why it is off by default.

## 5. Worked example

A pose-graph step with a translation column of norm $10^3$ (metres in a
large-scale map) and a rotation column of norm $10^0$ (radians):

- uniform damping $\lambda = 10^{-3}$ adds $10^{-3}$ to both diagonals: the
  translation is damped by a factor $10^{-6}$ relative to its curvature, the
  rotation by $10^{-3}$ — the rotation is effectively undamped and can swing
  wildly;
- Marquardt's scaling adds $\lambda\cdot 10^3$ to the translation diagonal
  and $\lambda\cdot 10^0$ to the rotation one: the damping is proportional to
  each column's own scale, and the step's shape is governed by the model
  rather than by the units.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — damping does not add information</span>

A damped direction is a direction the data does not determine; the damping
only keeps the step small. If the direction is genuinely unobservable, the
estimate along it is decided by the prior, the initial value, or the
regularization's tiny gradient — not by the measurements. The honest fix is to
constrain the direction explicitly (fix the gauge, add a prior), not to damp
it and hope.

</div>

<div class="apx-note">
<span class="apx-title">Note — the API changed for a reason</span>

`solve_augmented_equation` takes `&Damping` rather than a bare `lambda:
f64`; the old uniform behaviour is `Damping::identity(lambda)`. This is the
1.5-era change the migration chapter documents, and it is the shape that lets
the clamp live with the scalar instead of in the optimizer.

</div>
