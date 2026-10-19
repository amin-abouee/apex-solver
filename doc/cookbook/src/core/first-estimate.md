# The First-Estimate Policy

A sliding window summarizes what it drops into a Gaussian prior. That prior is
a *linear* statement about the survivors, valid at the point where it was
built. If the solve then re-linearizes it at the current estimate, the same
information is counted at two different points and the estimator becomes
confident about directions no measurement constrains. The fix is the
**first-estimate** convention, and this chapter is the crate's mechanism for
it.

## 1. The problem

Consider a window that marginalizes its oldest keyframe. The prior it creates
is

$$
p(\delta x) \propto \exp\!\left(-\tfrac12\,\delta x^\top \Lambda_p\,\delta x\right),
$$

linearized at the estimate $\mathbf{x}_0$ of the time the prior was built. In
the next solve, the survivors have moved to $\mathbf{x}$; the prior still
describes the world *as it was at $\mathbf{x}_0$*.

If the prior's Jacobian is recomputed at $\mathbf{x}$ — as a factor would be,
were the prior an ordinary factor — the information it carries is re-expressed
at the new point, and the difference between the two linearizations is
counted as *new* information. Repeated every window step, this accumulates
**rank in the null directions** of the problem: the gauge directions (global
position, yaw about gravity) that no measurement constrains acquire a small but
non-zero information, and the estimator's covariance in those directions
shrinks as if it had been measured.

## 2. The convention

> A measurement is evaluated at the linearization point where it entered the
> problem, not at the current one.

For the residual this is harmless — a residual is a function of the state, and
the *state* is current. For the Jacobian it is everything: the Jacobian is the
derivative at a point, and the point is the one the measurement was built at.

## 3. The crate's mechanism

<div class="apx-table">
<span class="apx-title">The three pieces</span>

| Piece | Where |
|---|---|
| `JacobianEvaluation::CurrentEstimate` (default) / `FirstEstimate` | `linearizer::JacobianEvaluation` |
| `Problem::set_jacobian_evaluation` / `jacobian_evaluation` | the problem's policy |
| `freeze_linearization_point` / `set_linearization_point` / `clear_linearization_point` / `is_linearization_point_frozen` | per variable |

</div>

A variable carries an optional **frozen linearization point**: the parameter
values at which its factors should report their Jacobians. With the policy
set to `FirstEstimate`:

- the **residual** of every block is evaluated at the current estimate;
- the **Jacobian** of a block is evaluated at each variable's frozen point
  where one exists, and at the current estimate for the variables that have
  none.

The doc comment states the cost exactly: "one extra `Factor::linearize` call
per block that touches a frozen variable, and only when a Jacobian is actually
requested. A block whose variables are all unfrozen takes the ordinary path
unchanged."

<figure class="apx-figure">
<svg viewBox="0 0 900 190" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif" font-size="12">
  <defs>
    <marker id="a08" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#555"/>
    </marker>
  </defs>
  <rect x="14" y="40" width="190" height="90" rx="7" fill="#eef4ff" stroke="#446"/>
  <text x="109" y="64" text-anchor="middle" font-weight="600">x_a (frozen at x₀)</text>
  <text x="109" y="84" text-anchor="middle" font-size="11">residual at the current</text>
  <text x="109" y="101" text-anchor="middle" font-size="11">Jacobian at x₀</text>
  <rect x="240" y="40" width="190" height="90" rx="7" fill="#eef4ff" stroke="#446"/>
  <text x="335" y="64" text-anchor="middle" font-weight="600">x_b (unfrozen)</text>
  <text x="335" y="84" text-anchor="middle" font-size="11">residual at the current</text>
  <text x="335" y="101" text-anchor="middle" font-size="11">Jacobian at the current</text>
  <rect x="520" y="40" width="360" height="90" rx="7" fill="#fff7e6" stroke="#a80"/>
  <text x="700" y="64" text-anchor="middle" font-weight="600">one block, two evaluation points</text>
  <text x="700" y="84" text-anchor="middle" font-size="11">the residual is a function of the state;</text>
  <text x="700" y="101" text-anchor="middle" font-size="11">the Jacobian is a derivative at a point</text>
  <line x1="204" y1="85" x2="232" y2="85" stroke="#555" marker-end="url(#a08)"/>
  <line x1="430" y1="85" x2="512" y2="85" stroke="#555" marker-end="url(#a08)"/>
  <text x="450" y="160" text-anchor="middle" font-size="11.5" fill="#555">the marginal prior is the block whose Jacobian must stay where it was built</text>
</svg>
<figcaption><b>Figure 8.1</b> — One block, two evaluation points: the residual follows the
state, the Jacobian follows the frozen point.</figcaption>
</figure>

## 4. Who uses it

A consumer that maintains a sliding window does the following at each step:

1. **marginalize** the oldest keyframe into a dense prior, built at the current
   estimate — and record that estimate as the prior's linearization point;
2. **freeze** the linearization point of every variable the prior touches;
3. **set** the problem's policy to `FirstEstimate`;
4. solve; the prior's block reports the Jacobian it was built with, and the
   other blocks report theirs.

`apex-vio` does exactly this, and its marginalization chapter documents the
consequences; this chapter documents the mechanism that makes it possible.

## 5. Worked example

A window of 7 keyframes. The oldest is marginalized; the prior touches
keyframes 1–6 and their biases, and is built at the estimate $\mathbf{x}_0$.
The next solve moves the estimate to $\mathbf{x} = \mathbf{x}_0 + \delta$.

With `CurrentEstimate`, the prior's Jacobian is recomputed at $\mathbf{x}$: the
prior's linear model is re-expanded about a different point, and the
difference between the two expansions is charged as information — the drift the
chapters call "the first estimate migrates".

With `FirstEstimate`, the prior's Jacobian stays at $\mathbf{x}_0$. The prior
is a *linear* model, and linear models do not need re-expanding; what they
need is to be applied at the point they were built. The information the prior
carries is therefore counted once, at the point it was measured, and the
gauge directions keep the zero rows the marginalization gave them.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — FEJ is not a free lunch for the whole graph</span>

The policy is global: with `FirstEstimate`, *every* block that touches a
frozen variable pays an extra `linearize` call. A graph that freezes a few
variables pays only for the blocks that touch them — but a graph that freezes
most of its variables pays for most of its blocks, and the linearization
becomes the dominant cost. The policy is a tool for windows, not a default for
static problems.

</div>

<div class="apx-warning">
<span class="apx-title">Warning — a frozen point is a promise you must keep</span>

`freeze_linearization_point` records where a variable's factors should report
their Jacobians. If the variable is later re-initialized (a re-detection, a
relocalization) and the frozen point is not cleared or updated, the factors
keep reporting a Jacobian about a state that no longer exists — and the step
is wrong in a way that looks like poor conditioning. `clear_linearization_point`
and `set_linearization_point` exist for that transition.

</div>

<div class="apx-note">
<span class="apx-title">Note — the residual is always current</span>

Only the Jacobian is frozen. A residual evaluated at the old point would
describe a world that is not the one being estimated, and the solve would
converge to the wrong state. The asymmetry — residual current, Jacobian
frozen — is the whole content of the convention.

</div>

<div class="apx-paper">
<span class="apx-title">Papers — the convention</span>

Dong-Si & Mourikis, "Motion Tracking with Fixed-lag Smoothing", and the
first-estimate Jacobian literature that followed (Dong-Si, Kelly & Leonard,
"Concurrent Real-time Visual-Inertial Odometry and Dense 3D Reconstruction",
*IROS*, 2018) — the convention, and the rank story that makes it necessary.

</div>
