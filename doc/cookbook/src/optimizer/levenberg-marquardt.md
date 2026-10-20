# Levenberg–Marquardt

Levenberg–Marquardt is the crate's default optimizer, and the reason is a
property no other method in the crate has: **it never takes a step it does
not believe**. The step comes from a damped system, the step is *tested*, and
if it does not improve the cost it is rejected and the damping raised. This
chapter is the damped step, the gain ratio, and the two damping rules.

## 1. The damped step

$$
\left(J^\top J + \lambda D\right)\delta = -J^\top \tilde r,
\qquad
D = \operatorname{diag}(J^\top J)\ \text{(Marquardt)},
$$

with $\lambda$ adaptive. The two limits are the familiar ones: $\lambda \to 0$
gives the Gauss–Newton step (fast, local), $\lambda \to \infty$ gives a scaled
gradient step (slow, always downhill). LM interpolates, and the interpolation
is chosen by the data.

The crate's `Damping` type carries the scalar plus a **clamp range** for the
damping diagonal: `min_diagonal` (Ceres' `min_lm_diagonal`, $10^{-6}$) keeps
the damping away from zero for columns with little curvature — without it, an
unconstrained direction would receive *no* damping at all — and
`max_diagonal` (Ceres' `max_lm_diagonal`, $10^{32}$) keeps one very stiff
column from dominating the damped system.

## 2. The gain ratio

The step is accepted or rejected by comparing what happened with what the
model promised:

$$
\rho = \frac{f(x_k) - f(x_k + h)}{f(x_k) - L(h)},
\qquad
L(h) = f(x_k) + h^\top g + \tfrac12 h^\top H h .
$$

$\rho > 0$ means the cost fell: accept, and trust the model more (lower
$\lambda$). $\rho \le 0$ means it rose: reject, roll the step back, and be
more conservative (higher $\lambda$).

<div class="apx apx-derive">
<span class="apx-title">Derivation — where the gain ratio comes from</span>

The damped step $\delta$ solves $(H + \lambda D)\delta = -\mathbf{g}$, so
$\delta = -(H + \lambda D)^{-1}\mathbf{g}$ and the *predicted* decrease of the
quadratic model is

$$
L(0) - L(\delta) = -\mathbf{g}^\top\delta - \tfrac12\delta^\top H\delta
= \tfrac12\delta^\top(H + \lambda D)\delta
= \tfrac12\delta^\top(-\mathbf{g})
$$

using $(H+\lambda D)\delta = -\mathbf{g}$. The *actual* decrease is
$F(\mathbf{x}_k) - F(\mathbf{x}_k + \delta)$. Their ratio is the gain ratio
$\rho$: it is one when the model is exact, zero when the model predicted no
descent, and negative when the step made things worse. That is why the
acceptance test is a comparison of two quantities the solver can both
compute.

</div>

## 3. Two damping rules

<div class="apx-widget" data-widget="lm" data-title="A Levenberg–Marquardt walk on the Rosenbrock valley: move the damping slider and watch the path change from gradient descent to the Newton step." data-foot="Gauss–Newton jumps to the valley floor and overshoots the curved walls; λ interpolates between gradient descent (large λ) and Gauss–Newton (small λ). LM accepts a step only when the cost falls, and raises λ otherwise — the rule in <code>apex_solver::optimizer::levenberg_marquardt</code>."></div>


<div class="apx-table">
<span class="apx-title">`DampingUpdate`</span>

| Rule | Behaviour |
|---|---|
| `Nielsen` (default) | accepted: $\lambda \leftarrow \lambda\max(\tfrac13, 1-(2\rho-1)^3)$, $\nu$ resets; rejected: $\lambda \leftarrow \lambda\nu$, $\nu \leftarrow 2\nu$ |
| `Marquardt` | $\lambda \leftarrow \lambda\cdot$`damping_decrease_factor` (0.3) when $\rho \ge$ `good_step_quality`; $\lambda \leftarrow \lambda\cdot$`damping_increase_factor` (10) when $\rho \le$ `min_step_quality` or the step was rejected; unchanged in between |

</div>

Nielsen's cubic is smooth in the step quality: $\rho = 1$ (a perfect
prediction) shrinks $\lambda$ by a factor of $1/3$ in one step, and the
shrink is gentle when the model is barely better than useless. The escalation
$\nu$ makes consecutive rejections grow geometrically, so a step that keeps
failing stops being attempted at full size.

Marquardt's three-band rule is coarser and easier to reason about: the
configuration's factors decide how fast the damping moves. The two rules
ignore each other's fields, and the crate says so in the enum's documentation.

<div class="apx apx-note">
<span class="apx-title">Note — Marquardt's third band is unreachable by default</span>

At the default `min_step_quality` of 0.0, a step with $\rho \le 0$ is
rejected *before* the threshold is consulted, so Marquardt's "raise the
damping" branch never fires — only the decrease and the rejection branches
are live. Raising `min_step_quality` above `min_relative_decrease` (the
textbook value is 0.25) makes the middle band do something. Under Nielsen —
the default — the question does not arise, because the cubic reacts to the
step quality continuously.

</div>

## 4. The configuration

| Field | Default | Meaning |
|---|---|---|
| `damping` | — | the initial $\lambda$ |
| `damping_min`, `damping_max` | — | bounds on $\lambda$ |
| `damping_increase_factor` | 10.0 | Marquardt's rejection multiplier |
| `damping_decrease_factor` | 0.3 | Marquardt's acceptance multiplier |
| `damping_nu` | 2.0 | Nielsen's escalation factor |

plus the shared skeleton: `linear_solver_type`, `max_iterations`, the three
tolerances, `timeout`, `min_cost_threshold`.

## 5. Worked example

A window solve with 111 unknowns, 800 rows, 30 % inlier rate. The first
iteration's model is decent ($\rho \approx 0.6$): Nielsen's rule gives
$\lambda \leftarrow \lambda\max(\tfrac13, 1-(0.2)^3) \approx 0.33\lambda$ — a
modest shrink, and the next step is nearly a Newton step. When the poses jump
and the photometric residuals blow up ($\rho \approx -0.4$), the step is
rejected, $\lambda \leftarrow 2\lambda$, and the retry is half as long. After
three consecutive rejections $\lambda$ has grown by 8× and the step is small
enough to be inside the model's trustworthy region.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — LM's guarantee is about the objective, not the model</span>

LM guarantees the cost decreases (for a well-behaved model, with the right
step-length treatment). It does not guarantee the *model* is right. A factor
with a wrong sign produces a smooth, well-behaved objective that the solver
minimizes happily. This is the difference between "the solve converged" and
"the answer is right", and no optimizer can bridge it.

</div>

<div class="apx-note">
<span class="apx-title">Note — the damping is not the trust region</span>

Dog Leg bounds the *step length*; LM bounds the *curvature contribution*.
The two are different controls with different failure modes: a trust region can
be too small and stall, a damping can be too small and diverge. LM's
`damping_min` and `damping_max` bound the damping itself, not the step — the
step length falls out of the damped system's condition number. Consumers who
want a hard bound on the step use Dog Leg.

</div>

<div class="apx-paper">
<span class="apx-title">Papers — the method</span>

Levenberg, "A Method for the Solution of Certain Non-Linear Problems in Least
Squares", 1944. Marquardt, "An Algorithm for Least-Squares Estimation of
Nonlinear Parameters", *SIAM*, 1963 — the diagonal scaling. Nielsen, "Damping
Parameter in Marquardt's Method", 1999 — the cubic damping rule this crate
defaults to. Ceres Solver — the reference implementation whose `min_lm_diagonal`
/ `max_lm_diagonal` clamps the crate reproduces.
</div>
