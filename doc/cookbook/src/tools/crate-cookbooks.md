# The Crate Cookbooks

`apex-solver` is one of four crates in the workspace, and the other three
carry their own cookbooks. This page is the map: what each one covers, and
where the boundaries are.

## 1. The neighbourhood

<div class="apx-table">
<span class="apx-title">The four crates</span>

| Crate | What it is | Its cookbook |
|---|---|---|
| `apex-solver` | the factor-graph engine: variables, factors, optimizers, linear solvers, marginalization | **this book** |
| `apex-manifolds` | the Lie groups: `SE3`, `SE23`, `SGal3`, `SO3`, `SE2`, `Sim3`, and their Jacobians | `crates/apex-manifolds/doc/cookbook/` |
| `apex-camera-models` | projection and unprojection: pinhole, radtan, Brown-Conrady, double sphere, EUCM, Kannala–Brandt, FOV, UCM, f-theta | `crates/apex-camera-models/doc/cookbook/` |
| `apex-io` | the recording formats: ASL, EuRoC, TUM, rosbag, CDR, mcap, pose graphs | `crates/apex-io/doc/cookbook/` |

</div>

## 2. What belongs where

The division is by *kind* of knowledge, and it is worth being strict about
it:

- **A manifold owns its own derivatives.** The exponential, the logarithm, the
  adjoint, the right Jacobian of `SE3` — these are properties of the group,
  and they live in `apex-manifolds`. The solver's factors *use* them; they do
  not re-derive them (chapter [Common Utilities](../factors/common.md)).
- **A camera model owns its own projection.** The pinhole's `K` and its
  distortion polynomial, the fisheye's tangent mapping, the double sphere's
  two-sphere parameterization — these live in `apex-camera-models`, and a
  visual factor is generic over them.
- **A format owns its own quirks.** A rosbag's timestamps, an ASL dataset's
  naming, a TUM trajectory's format — these live in `apex-io`, and the solver
  never sees a timestamp.
- **The solver owns the graph.** The problem, the blocks, the noise models,
  the robust losses, the optimizers, the linear solvers, the Schur
  complement, the marginalization, and the first-estimate convention. That is
  this book.

## 3. What this book assumes

Three things are assumed and never re-derived here:

1. the right-perturbation convention and the tangent-space calculus
   (`apex-manifolds`);
2. the camera projection conventions — a camera's pose is world-to-camera,
   $\mathbf{p}_C = R\mathbf{p}_W + \mathbf{t}$ (`apex-camera-models`);
3. the dataset and trajectory formats (`apex-io`).

Where this book *does* restate a piece of that theory, it is because the
solver's correctness depends on it: the right-perturbation convention
(chapter [Variables on Manifolds](../core/variables.md)), the parameter
layouts (the same chapter), and the camera pose direction (chapter
[Notation & Conventions](../conventions.md)).

## 4. The two application books

Above the solver sit two more books, in the sibling repositories:

| Book | Covers |
|---|---|
| **The Apex VIO Cookbook** (`apex-vio`) | visual-inertial and LiDAR-inertial odometry: front ends, back ends, the window, the marginalization policy, the SLAM layer, and the accuracy results |
| **The Apex CV Cookbook** (`apex-cv`) | the computer-vision crate: feature detection, tracking, multi-view geometry, mapping, registration, photometric estimation, and detection |

They are the natural place to read how the solver's factors are *used*: what
the residuals mean in the context of a camera rig, which noise model a
particular measurement deserves, and what the system's own conventions are.

## 5. A note on duplication

The shared theory — manifolds, camera models, robust losses, Levenberg–
Marquardt, the Schur complement, noise models — appears in more than one of
these books, because each of them is written to be readable on its own. This
book is the authority for the *solver's* half of that theory: the noise
model, the corrector, the optimizers, the linear solvers, the Schur
complement, the marginalization, and the first-estimate convention.

When the two disagree, the code wins — and the code is the one place both
books point at.

## 6. Worked example

A consumer needs a factor that measures a point's bearing in a fisheye
camera. Three crates are involved:

- `apex-camera-models` provides the unprojection that turns the pixel into a
  unit bearing, and the right Jacobian of that unprojection;
- `apex-solver` provides the factor that compares the bearing with the
  measurement, the noise model that scales it, and the optimizer that
  minimizes it;
- the consumer's own book — VIO or CV — says which camera model is in use and
  what the residuals mean physically.

No crate needs to know about another's internals, and the factor is generic
over the camera model: a double sphere would work unchanged.

## 7. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a book that contradicts the code is worse than no book</span>

These books are written against a specific tree, and the tree moves. The rule
this book follows is that the code always wins: where a document and the
source disagree, the source is described and the disagreement is stated.
When you read something here that the code does not do, it is a bug in this
book — and the fix is to change the book, in the same commit that changes
the code.

</div>

<div class="apx-note">
<span class="apx-title">Note — the version pins</span>

This book describes the tree at `apex-solver` **1.5.0**, with the 1.6
deprecations already staged in the source. The three sibling crates are at
0.4.0. The migration chapter lists what changed and when.

</div>
