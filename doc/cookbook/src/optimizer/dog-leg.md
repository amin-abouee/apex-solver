# Dog Leg

Levenberg–Marquardt bounds the *curvature* a step may use; a **trust region**
bounds the *length* of the step. Dog Leg is the trust-region method in this
crate, and it is the right tool when the model may be far from the truth — a
large initial error, a first frame, a re-localization. This chapter is the
three-point construction, the radius adaptation, and the comparison with LM.

## 1. The construction

Powell's Dog Leg builds a piecewise path inside a spherical trust region of
radius $\Delta$, connecting three points:

1. the **origin** — the current estimate;
2. the **Cauchy point** $p_c = -\alpha g$, the optimal step along the
   steepest-descent direction, with
   $\alpha = \frac{g^\top g}{g^\top H g}$ and $H = J^\top J$;
3. the **Gauss–Newton point** $h_{gn}$, the full Newton step.

The step is then chosen by where the trust region boundary falls:

<div class="apx-table">
<span class="apx-title">The three cases</span>

| Condition | Step |
|---|---|
| $\|h_{gn}\| \le \Delta$ | $h = h_{gn}$ — the full Newton step |
| $\|p_c\| \ge \Delta$ | $h = \frac{\Delta}{\|g\|}(-g)$ — steepest descent to the boundary |
| $\|p_c\| < \Delta < \|h_{gn}\|$ | $h(\beta) = p_c + \beta (h_{gn} - p_c)$, with $\beta$ chosen so $\|h(\beta)\| = \Delta$ |

</div>

The middle case is the "dog leg": out to the Cauchy point, then along the
direction to the Newton point, stopping at the boundary. The shape is what
gives the method its name and its behaviour — it always contains a downhill
direction, and it approaches the Newton step as the model becomes good.

## 2. Adapting the radius

The radius $\Delta$ is not fixed. It responds to the gain ratio
$\rho = \frac{\text{actual decrease}}{\text{predicted decrease}}$, in three
bands:

| Band | Action |
|---|---|
| $\rho >$ `good_step_quality` (0.75) | $\Delta \leftarrow \Delta\cdot$`trust_region_increase_factor` |
| $\rho <$ `poor_step_quality` | $\Delta \leftarrow \Delta\cdot$`trust_region_decrease_factor` |
| in between | $\Delta$ unchanged |

bounded by `trust_region_min` and `trust_region_max`. A step that improves
the cost much less than predicted means the model is stale, and the region
shrinks; a step that matches the prediction means the model is good, and the
region grows.

The implementation also carries Ceres-style **adaptive $\mu$ regularization**,
so a trust-region step in a nearly singular direction is regularized rather
than allowed to run to the boundary.

## 3. Configuration

| Field | Meaning |
|---|---|
| `trust_region_radius` | the initial $\Delta$ |
| `trust_region_min`, `trust_region_max` | its bounds |
| `trust_region_increase_factor`, `trust_region_decrease_factor` | the adaptation factors |
| `min_step_quality`, `good_step_quality`, `poor_step_quality` | the $\rho$ bands |

plus the shared skeleton (`linear_solver_type`, `max_iterations`, the three
tolerances, `timeout`).

## 4. Dog Leg versus Levenberg–Marquardt

<div class="apx-table">
<span class="apx-title">Two ways to distrust a model</span>

| | Levenberg–Marquardt | Dog Leg |
|---|---|---|
| controls | the curvature the step may use | the length of the step |
| failure mode | a step that is too long in a bad direction | a region that is too small and stalls |
| reaction to a bad model | damping grows, step shrinks | radius shrinks, step shortens |
| cost per iteration | one damped solve | one Newton solve + a Cauchy point |
| best for | well-behaved models with occasional bad steps | models that may be far from the truth |

</div>

Practically: LM is the default because it is cheap and rarely fails badly;
Dog Leg is the choice when the initial guess may be poor enough that the
quadratic model is useless, and when a *hard* bound on the motion is
desirable.

## 5. Worked example

A relocalization: a pose is known to within 30° of the truth, and the model is
a bearing factor over a few landmarks. The Newton step would be enormous and
wrong; LM would damp it, possibly by a factor of 100, and take several
iterations to find the right length. Dog Leg computes the Cauchy point, finds
it outside a 5° radius, and takes a scaled gradient step to the boundary —
a bounded, always-downhill move. Three or four iterations of that land inside
the basin where the model is valid, and the Newton steps take over.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a trust region needs a metric</span>

The region is a sphere in the *step* space, and the step space is a tangent
space whose coordinates are the variables' parameters — metres for a
translation, radians for a rotation. A radius of 1 means one metre of
translation *and* one radian of rotation at the same time, which is rarely
what anyone means. The crate mitigates this with optional Jacobi scaling, and
a consumer with mixed units should think about the radius in those terms
before choosing one.

</div>

<div class="apx-note">
<span class="apx-title">Note — the Cauchy point needs a positive curvature</span>

$\alpha = \frac{g^\top g}{g^\top H g}$ divides by the curvature along the
gradient. When the model is flat in that direction, the Cauchy point is far
away (or unbounded) and the trust region takes over. The implementation
handles the degenerate case by falling back to the boundary step; the
mathematical limit is the same one LM's damping handles.

</div>

<div class="apx-paper">
<span class="apx-title">Papers — the method</span>

Powell, "Some algorithms for constrained maximization or minimization", 1970 —
the dog-leg construction. Charles & LeGault, "Gauss-Newton and Dog-Leg
Trust-Region Methods", 2019 — the modern formulation used here. Ceres Solver —
the reference implementation of the radius adaptation and the $\mu$
regularization.
</div>
