# What a Factor Graph Is

A factor graph is a way of writing down "a set of unknowns, and a set of
measurements that relate them" such that the structure of the problem — which
unknowns are coupled, which are local, which can be eliminated — is visible
on the page. This chapter is that idea, the vocabulary the crate uses for it,
and the two properties of the graph that decide how the solve is organized.

## 1. The problem

A trajectory estimation problem looks like this in prose: "the camera pose at
each keyframe is unknown; the relative pose between consecutive keyframes is
measured; the landmarks are unknown; each landmark is seen by several
keyframes." Written as equations, it is a stack of residuals, each touching a
handful of unknowns.

The questions a solver must answer are structural:

- **How big is the linear system?** A hundred poses and a thousand landmarks
  is 100 × 6 + 1000 unknowns, but the matrix is *sparse*: a landmark is
  coupled to the few keyframes that see it, and nothing else.
- **What can be eliminated?** A landmark is local — it appears in a few
  residuals and nowhere else. Eliminating it exactly (a Schur complement)
  leaves a smaller system in the poses alone, and the landmark's contribution
  comes back at the end.
- **What is each measurement's weight?** Every residual has an information
  matrix, and the objective is $\Omega$-weighted.

A factor graph is the data structure that answers all three, and this crate's
`Problem` is its implementation.

## 2. The vocabulary

<div class="apx-symbols apx-table">
<span class="apx-title">The objects</span>

| Object | Meaning | Code |
|---|---|---|
| variable | an unknown on a manifold (a pose, a bias, a landmark) | `ManifoldVariable`, keyed by `VarKey` |
| factor | a measurement: a residual and its Jacobian | `Factor` |
| residual block | a factor plus an optional robust loss | `ResidualBlock` |
| problem | the whole graph: variables, blocks, and the assembly machinery | `Problem` |
| noise model | the square-root information $\Omega^{1/2}$ of one block | `NoiseModel` |
| loss | a robust kernel $\rho$ | `core::loss_functions` |
| corrector | the re-weighting that turns $\rho$ into a least-squares problem | `core::corrector` |

</div>

The split of a factor from a residual block matters: the *factor* is geometry
(the residual formula and its Jacobian), the *block* is the geometry plus the
robustness policy. A factor is registered once and may be shared; a block
carries the loss and the noise model for one registration.

## 3. The two structural properties

<figure class="apx-figure">
<svg viewBox="0 0 900 250" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif" font-size="12">
  <defs>
    <marker id="a01" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#555"/>
    </marker>
  </defs>
  <circle cx="120" cy="60" r="17" fill="#eef4ff" stroke="#446"/>
  <text x="120" y="64" text-anchor="middle" font-size="11">x₀</text>
  <circle cx="300" cy="60" r="17" fill="#eef4ff" stroke="#446"/>
  <text x="300" y="64" text-anchor="middle" font-size="11">x₁</text>
  <circle cx="480" cy="60" r="17" fill="#eef4ff" stroke="#446"/>
  <text x="480" y="64" text-anchor="middle" font-size="11">x₂</text>
  <circle cx="210" cy="190" r="17" fill="#fdeef2" stroke="#a46"/>
  <text x="210" y="194" text-anchor="middle" font-size="11">ρ</text>
  <circle cx="390" cy="190" r="17" fill="#fdeef2" stroke="#a46"/>
  <text x="390" y="194" text-anchor="middle" font-size="11">ρ</text>
  <rect x="145" y="88" width="105" height="30" rx="5" fill="#fff7e6" stroke="#a80"/>
  <text x="197" y="107" text-anchor="middle" font-size="10.5">IMU</text>
  <rect x="325" y="88" width="105" height="30" rx="5" fill="#fff7e6" stroke="#a80"/>
  <text x="377" y="107" text-anchor="middle" font-size="10.5">IMU</text>
  <rect x="160" y="140" width="100" height="30" rx="5" fill="#eafbea" stroke="#484"/>
  <text x="210" y="159" text-anchor="middle" font-size="10.5">bearing</text>
  <rect x="340" y="140" width="100" height="30" rx="5" fill="#eafbea" stroke="#484"/>
  <text x="390" y="159" text-anchor="middle" font-size="10.5">bearing</text>
  <line x1="120" y1="60" x2="145" y2="88" stroke="#888"/>
  <line x1="300" y1="60" x2="145" y2="88" stroke="#888"/>
  <line x1="300" y1="60" x2="325" y2="88" stroke="#888"/>
  <line x1="480" y1="60" x2="325" y2="88" stroke="#888"/>
  <line x1="300" y1="60" x2="210" y2="140" stroke="#888"/>
  <line x1="480" y1="60" x2="390" y2="140" stroke="#888"/>
  <line x1="210" y1="173" x2="210" y2="140" stroke="#888"/>
  <line x1="390" y1="173" x2="390" y2="140" stroke="#888"/>
  <text x="450" y="30" text-anchor="middle" font-size="12" fill="#555">variables (poses, landmarks) · factors (measurements)</text>
  <text x="450" y="228" text-anchor="middle" font-size="11.5" fill="#555">a landmark touches two poses and nothing else — that locality is what makes elimination possible</text>
</svg>
<figcaption><b>Figure 1.1</b> — A factor graph: variables as circles, measurements as
squares, and the edges that say which variables a measurement constrains.</figcaption>
</figure>

**Sparsity.** The graph's edges are the Jacobian's non-zeros. A factor that
touches three variables contributes a $3 \times 3$ block of $J$ and nothing
else. The whole linear system inherits that sparsity, and the linearizer's job
is to exploit it rather than to densify it.

**Locality.** A variable that appears in a few factors and nowhere else is
*eliminable*: it can be removed from the linear system exactly (a Schur
complement) and recovered afterwards. Landmarks are the standard case, and the
crate's `mark_for_elimination` is the explicit opt-in that says so. The
deprecation note on the old name makes the policy clear: elimination is not
restricted to landmarks; any variable that satisfies the structural
requirements may be marked.

## 4. What the crate does with the graph

<figure class="apx-figure">
<svg viewBox="0 0 900 200" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif" font-size="12">
  <defs>
    <marker id="a01b" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#555"/>
    </marker>
  </defs>
  <rect x="10" y="60" width="140" height="60" rx="7" fill="#eef4ff" stroke="#446"/>
  <text x="80" y="84" text-anchor="middle" font-weight="600">Problem</text>
  <text x="80" y="102" text-anchor="middle" font-size="11">variables + blocks</text>
  <rect x="190" y="60" width="150" height="60" rx="7" fill="#fff7e6" stroke="#a80"/>
  <text x="265" y="84" text-anchor="middle" font-weight="600">Linearizer</text>
  <text x="265" y="102" text-anchor="middle" font-size="11">r, J (sparse or dense)</text>
  <rect x="380" y="60" width="160" height="60" rx="7" fill="#eafbea" stroke="#484"/>
  <text x="460" y="84" text-anchor="middle" font-weight="600">Optimizer</text>
  <text x="460" y="102" text-anchor="middle" font-size="11">GN · LM · Dog Leg</text>
  <rect x="580" y="60" width="150" height="60" rx="7" fill="#fdeef2" stroke="#a46"/>
  <text x="655" y="84" text-anchor="middle" font-weight="600">Linear solver</text>
  <text x="655" y="102" text-anchor="middle" font-size="11">Cholesky · QR · Schur</text>
  <rect x="770" y="60" width="120" height="60" rx="7" fill="#f2f2f2" stroke="#666"/>
  <text x="830" y="84" text-anchor="middle" font-weight="600">Step</text>
  <text x="830" y="102" text-anchor="middle" font-size="11">manifold plus</text>
  <line x1="150" y1="90" x2="182" y2="90" stroke="#555" marker-end="url(#a01b)"/>
  <line x1="340" y1="90" x2="372" y2="90" stroke="#555" marker-end="url(#a01b)"/>
  <line x1="540" y1="90" x2="572" y2="90" stroke="#555" marker-end="url(#a01b)"/>
  <line x1="730" y1="90" x2="762" y2="90" stroke="#555" marker-end="url(#a01b)"/>
  <path d="M 830 60 L 830 20 L 265 20 L 265 52" fill="none" stroke="#999" stroke-dasharray="4 4" marker-end="url(#a01b)"/>
  <text x="548" y="14" text-anchor="middle" font-size="11" fill="#666">repeat until convergence</text>
</svg>
<figcaption><b>Figure 1.2</b> — One optimizer iteration: the linearizer produces the
residual and Jacobian, the linear solver produces the step, the step is applied
through the manifold.</figcaption>
</figure>

<div class="apx-widget" data-widget="solver-graph" data-title="The topologies the crate's factors form" data-foot="Switch between the four shapes; hover a node for its incident factors, a factor for its residual."></div>

The crate separates the three stages deliberately: the **linearizer** owns
"evaluate every factor and assemble the linear system", the **optimizer** owns
"choose a step, accept or reject it", and the **linear solver** owns "solve
the (possibly reduced) linear system". A factor author touches only the first.

## 5. Worked example

A pose graph: 300 keyframes (6 DoF each) and 20 loop closures, plus 280
odometry edges. Variables: 300. The odometry edges chain the poses
(band structure); the loop edges connect distant poses (fill-in). The
linearizer assembles a sparse $1800 \times 1800$ system with about
$300 \times 12$ non-zeros from the odometry edges and $20 \times 72$ from the
loops. A dense Cholesky would cost $O(1800^3)$; a sparse one costs
$O(\text{nnz}^{1.5})$. The Schur path is not applicable here — no variable is
eliminable — which is exactly the decision [Choosing a
Solver](../optimizer/choosing.md) teaches.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a graph's structure is not a promise about conditioning</span>

Sparsity says which unknowns are coupled; it says nothing about whether the
coupling is *well determined*. A corridor couples the rig's position to its
orientation in a way that is sparse and singular; a factor graph can be
perfectly structured and still have a direction no measurement constrains.
That is what the degeneracy analysis in the consumer crates is for, and what
the marginalization chapter's rank story is about: structure tells you what
*can* be eliminated, conditioning tells you what *is* known.

</div>

<div class="apx-note">
<span class="apx-title">Note — "factor" in this crate is a measurement, not a term of art</span>

`Factor` implements one residual and its Jacobian. It is not a probability
factor, and it does not carry a prior or a likelihood. The prior-like objects
in the crate (`PriorFactor`, `EuclideanPriorFactor`, `MarginalPriorFactor`)
are factors of a particular kind: a prior on a variable, a prior in ambient
coordinates, and a summary of eliminated variables respectively. Chapter
[Pose and Prior Factors](../factors/pose.md) tells them apart.

</div>
