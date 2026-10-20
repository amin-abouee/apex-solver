# From Factor Graph to Linear System

Every optimizer iteration needs the same three things: the residual vector,
the Jacobian, and the cost. The **linearizer** produces them, once per
iteration, for the whole graph — and it is where the crate's two assembly
strategies (sparse and dense) and the factor-evaluation policy live. This
chapter is that stage.

## 1. The problem

A factor's residual is a function of the variables; the optimizer needs it
evaluated at the current estimate, and the Jacobian of that residual with
respect to every variable the factor touches. Assembling the global system is
then a matter of placing each block's rows and columns in the right slots.

Three things make this more than a loop over factors:

- **the Jacobian is optional.** Some factor shapes have a much cheaper normal
  equations than a Jacobian (`Factor::accumulate_normal`), and the solver
  should not pay for a matrix it will not use;
- **the evaluation point may differ per variable** — the first-estimate policy
  of the next chapter;
- **the sparsity is structural.** Which (block, variable) pairs exist is known
  at registration, so the layout can be computed once and reused every
  iteration.

## 2. The pipeline

<figure class="apx-figure">
<svg viewBox="0 0 900 210" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif" font-size="12">
  <defs>
    <marker id="a07" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#555"/>
    </marker>
  </defs>
  <rect x="10" y="60" width="150" height="66" rx="7" fill="#eef4ff" stroke="#446"/>
  <text x="85" y="84" text-anchor="middle" font-weight="600">Problem</text>
  <text x="85" y="102" text-anchor="middle" font-size="11">blocks + variables</text>
  <rect x="195" y="60" width="170" height="66" rx="7" fill="#fff7e6" stroke="#a80"/>
  <text x="280" y="84" text-anchor="middle" font-weight="600">compute_block_into</text>
  <text x="280" y="102" text-anchor="middle" font-size="11">per block: r, J, cost</text>
  <rect x="400" y="60" width="180" height="66" rx="7" fill="#eafbea" stroke="#484"/>
  <text x="490" y="84" text-anchor="middle" font-weight="600">whiten + correct</text>
  <text x="490" y="102" text-anchor="middle" font-size="11">S·r, S·J, then ρ</text>
  <rect x="615" y="60" width="150" height="66" rx="7" fill="#fdeef2" stroke="#a46"/>
  <text x="690" y="84" text-anchor="middle" font-weight="600">assemble</text>
  <text x="690" y="102" text-anchor="middle" font-size="11">sparse or dense</text>
  <rect x="790" y="60" width="100" height="66" rx="7" fill="#f2f2f2" stroke="#666"/>
  <text x="840" y="84" text-anchor="middle" font-weight="600">solve</text>
  <text x="840" y="102" text-anchor="middle" font-size="11">H δ = −g</text>
  <line x1="160" y1="93" x2="187" y2="93" stroke="#555" marker-end="url(#a07)"/>
  <line x1="365" y1="93" x2="392" y2="93" stroke="#555" marker-end="url(#a07)"/>
  <line x1="580" y1="93" x2="607" y2="93" stroke="#555" marker-end="url(#a07)"/>
  <line x1="765" y1="93" x2="782" y2="93" stroke="#555" marker-end="url(#a07)"/>
  <text x="450" y="30" text-anchor="middle" font-size="12" fill="#555">per iteration: evaluate → whiten → correct → assemble → solve</text>
  <text x="450" y="180" text-anchor="middle" font-size="11.5" fill="#555">the per-block evaluation is parallel across blocks; the assembly is where the sparse structure is exploited</text>
</svg>
<figcaption><b>Figure 7.1</b> — The linearization stage. Everything the optimizer needs is
produced here, once per iteration.</figcaption>
</figure>

The crate's own architecture comment states the flow:

```text
Problem (factor graph)
    │  AssemblyBackend::assemble()
    ▼
(r: Mat<f64>, J: M::Jacobian)   ← M: LinearizationMode
    │
    ▼
LinearSolver<M>
    │
    ▼
dx: Mat<f64>  → manifold update
```

`LinearizationMode` is the sparse/dense choice; `AssemblyBackend` is the trait
that bridges the linearizer to the optimizer's solver types.

<div class="apx apx-derive">
<span class="apx-title">Derivation — the normal equations</span>

Write the objective as $F(\mathbf{x}) = \tfrac12\lVert\tilde r(\mathbf{x})\rVert^2$ and
expand it to second order about the current estimate $\mathbf{x}_k$:

$$
F(\mathbf{x}_k + \delta) \approx F(\mathbf{x}_k) + \mathbf{g}^\top\delta + \tfrac12\delta^\top H\delta,
$$

with $\mathbf{g} = J^\top\tilde r$ and $H = J^\top J$. Differentiating and
setting the derivative to zero gives $\mathbf{g} + H\delta = 0$, i.e.

$$
H\,\delta = -\mathbf{g}, \qquad \delta = -H^{-1}\mathbf{g}.
$$

The minus sign is not a convention: the gradient is $J^\top r$ with $r$ the
predicted-minus-measured residual, so a positive gradient points *up* the
cost, and the step must go down it. The crate's `NormalEquations::gradient`
is positive, and the system it solves carries the minus on the right-hand
side.

</div>

## 3. Per-block evaluation

`compute_block_into` evaluates one residual block: it fetches the variables'
parameter slices, calls the factor's `linearize`, applies the noise model's
whitening, applies the corrector if a loss is attached, and returns the block's
rows, columns and cost.

The factor's two other hooks matter here:

- `Factor::accumulate_normal` — "add this factor's $J^\top J$ and $J^\top r$
  into `blocks` and return its cost $\|\mathbf{r}\|^2$, without forming $J$".
  A factor whose structure makes the normal equations far cheaper than the
  Jacobian (a plane over many frames: a few small diagonal blocks of a
  mostly-zero $J$) implements it, and the solver then never materializes $J$
  for it.
- `Factor::validate_variables` — called at registration, not here.

## 4. Sparse assembly

The sparse path keeps a **symbolic structure** — the pattern of non-zeros —
computed once from the block layout and reused every iteration. `SymbolicStructure`
is re-exported from `core::problem`, so a consumer can inspect or persist it.
The numeric values are then filled in each iteration; the pattern is not
recomputed, which is what makes a per-iteration rebuild affordable.

The problem's own accessors expose the same products for diagnosis:
`compute_residual_sparse`, `compute_residual_and_cost_sparse`,
`compute_residual_and_jacobian_sparse`, and the dense twin
`compute_residual_and_jacobian_dense`.

## 5. Worked example

A window with 7 keyframes (15 DoF each) and 200 landmarks (1 DoF each):
111 305 unknowns, 800 residual rows. The dense Jacobian is $800 \times 111305$ —
712 million doubles, 5.7 GB. The sparse one holds the same numbers in the
block layout: 7 IMU blocks of 15 rows, and 200 bearing blocks of 2 rows each
touching 2 keyframes and 1 landmark. The nonzero count is a few thousand. This
is the whole reason the crate has a sparse path, and why `JacobianMode::Sparse`
is the default.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — the pattern is fixed at registration</span>

The symbolic structure is derived from the blocks registered when the problem
was built. Adding or removing a block changes the structure, and a problem
whose structure is stale assembles into the wrong slots. The crate rebuilds the
structure when blocks are added or removed (`add_residual_block`,
`remove_residual_block`); a consumer that mutates the graph between solves
must let those calls do it.

</div>

<div class="apx-note">
<span class="apx-title">Note — the cost the linearizer reports is the robust one</span>

The cost accumulated per block is $\rho(\|S r\|^2)$, not
$\tfrac12\|S r\|^2$. That is the number to compare against a reference
implementation, and it is why the solver's convergence tests can be stated in
"the cost stopped changing" rather than "the step stopped changing" — the two
are not the same criterion, and the robust cost is the one the user asked for.

</div>
