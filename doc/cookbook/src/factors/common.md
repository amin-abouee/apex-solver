# Common Utilities

A handful of small pieces are used by more than one factor family. They live
in `factors::common` rather than being re-derived per module, and this
chapter is what they are, when each is the right tool, and — for one of them
— why it exists at all.

## 1. The module

<div class="apx-table">
<span class="apx-title">`factors::common`</span>

| Item | Purpose |
|---|---|
| `math` | numerical helpers: a stable `sinc`, skew-symmetric constructors, matrix roots |
| `cheirality` | the bounded penalty for a point behind the camera |
| `group_point` | moving points with a generic pose group, and the Jacobians |
| `validate` | shared parameter-block validation |
| `test_utils` | boilerplate shared by the factor tests |

</div>

The module's own comment states the rule: "Anything used by more than one
factor domain lives here rather than being re-derived per module: the
skew-symmetric and matrix-root helpers, the cheirality penalty, and
parameter-block validation."

## 2. `math`

**`sinc(x) = sin(x)/x`**, with a Taylor expansion for $|x|^2 < 10^{-10}$ to
avoid dividing by a near-zero. It appears wherever a factor differentiates a
normalized vector — the derivative of $\mathbf{v}/\lVert\mathbf{v}\rVert$
contains $\mathrm{sinc}$, and the naive form loses precision exactly where
the factor is most sensitive: a point close to the camera.

**Skew-symmetric constructors** turn a 3-vector into the matrix $[\mathbf{v}]_\times$
with $\mathbf{v} \times \mathbf{w} = [\mathbf{v}]_\times \mathbf{w}$. Every
factor that rotates a vector needs one, and the sign convention (which side
the vector is on) is a classic source of sign errors, so the constructor is
shared.

**Matrix roots** — a symmetric matrix's positive-definite square root
$L$ with $L L^\top = M$ — appear wherever a factor needs a square-root
information matrix: the plane-moment factor's $S = LL^\top$, the covariance
machinery, the aggregate factor's eigendecomposition.

## 3. `cheirality`

A projection factor cannot evaluate a landmark that is behind the camera:
the projection is undefined, the depth is negative, and the "measurement" is
meaningless. Returning zero there would make an invalid configuration look
*optimal* — the optimizer would find a pose that puts the point behind the
camera and stop, because the cost went to zero.

Every camera factor therefore emits a **bounded, strictly-decreasing-toward-validity
penalty** instead: the residual grows as the point moves further behind the
camera, up to a cap, so the optimizer is pushed back into the valid region
rather than rewarded for leaving it.

The penalty is *bounded* on purpose. An unbounded penalty (a real
$1/z$-shaped term) would make the problem's cost unbounded below as well, and
the solver would chase the singularity. A bounded penalty trades exactness at
the boundary for a well-posed objective.

The module's constant is in "pixels-equivalent" units, because the visual
factors that use it work in image space: the penalty's scale is comparable to
a pixel, so a point that drifts behind the camera produces a residual of the
same order as a one-pixel reprojection error.

## 4. `group_point`

Factors that are generic over a `LieGroup` carry points through $X\odot\mathbf{p}$
and $X^{-1}\odot\mathbf{p}$. Both the actions and their Jacobians come from
the group's own `act` and `inverse` — the module's comment is explicit that
the manifold derivatives are *not* re-derived here, because every group in
`apex-manifolds` already reports the Jacobians of its own operations.

This is a good design decision with a visible consequence: a new group needs
no changes in the factors. The `SE23` and `SGal3` IMU factors, the `SE2`
bearing factors and the `SE3` pose factors share the same point-moving code,
and each family's factor is written once per group rather than once per
(group, point) pair.

## 5. `validate`

`validate` checks that a factor is being registered with the variables it
expects: `expected` lists the parameter-vector length of each block, in
order, and the check compares it with the variables' actual sizes. This is
the shared implementation behind the `Factor::validate_variables` hook
(chapter [Writing a Factor](./writing.md)), and it is what turns "the
landmark's parameter count disagrees with the factor's observations" into a
typed registration error rather than a panic inside the parallel assembly.

## 6. `test_utils`

The boilerplate the factor tests repeat verbatim: constructing a perturbed
parameter block for the finite-difference check. What it deliberately does
*not* do is decide *how* to perturb — a manifold's `right_plus` and a plain
vector's `+= eps` differ per factor, and that decision stays inline in each
test, where the reader can see it.

## 7. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a shared helper is a shared assumption</span>

`cheirality`'s penalty scale is in pixel-equivalents, and it is applied by
factors that work in *bearing* space as well as pixel space. A factor that
uses the pixel-scaled penalty in a bearing-space residual is applying a
penalty whose magnitude means something else — the penalty's size is
dimensionally tied to the residual's units, and the units differ between the
families. A factor ported between the two must convert, and the conversion is
the part that is easy to forget.

</div>

<div class="apx-note">
<span class="apx-title">Note — the module's boundaries are deliberate</span>

Manifold derivatives are not in `common`, even though several factors need
them, because they belong to the group that defines them. The rule is worth
stating because it is the reason the crate's factor code is as short as it is:
a factor asks its group for a derivative, and never re-derives one.

</div>
