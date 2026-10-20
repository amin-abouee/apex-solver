# Introduction

`apex-solver` is the optimization engine under the Apex stack: the crate that
turns a graph of measurements and unknowns into a trajectory, a map, and an
uncertainty estimate. It is used by `apex-vio` for visual-inertial and
LiDAR-inertial odometry, by `apex-cv` for its refinement steps, and by
`apex-calibration` for target poses. This book teaches it the way the other
books in the family teach their crates: from the problem, through the
derivation, to the code that ships.

| Part | What it covers |
|---|---|
| **I — The Problem and Its Machinery** | what a factor graph is, variables on manifolds, noise models, robust losses, the corrector, linearization, and the first-estimate policy |
| **II — Optimizers and Linear Solvers** | Gauss–Newton, Levenberg–Marquardt, Dog Leg, damping, the dense and sparse linear solvers, the Schur complement, and how to choose between them |
| **III — The Factors** | every factor family in the crate: pose and priors, visual geometry, photometric, IMU, LiDAR, ranging, motion models, navigation, and the four decorators that reshape a factor's blocks |
| **IV — Marginalization** | turning a set of variables into a Gaussian prior, and the FEJ consistency rule that keeps a sliding window honest |
| **V — Tools and Appendices** | Rerun observers, benchmarks, the 1.4 → 1.5 → 1.6 migration, and the crate's neighbours |

## Why this crate deserves a book

A factor-graph solver looks like plumbing: a `Problem`, a `Factor` trait, an
optimizer, a linear solver. The plumbing is easy to misuse, and the misuse is
silent. A factor that whitens its own residual and is registered with a noise
model is over-weighted; a marginalization that re-linearizes at the current
estimate makes the estimator confident about directions no measurement
constrains; a Gauss–Newton step on a rank-deficient system either fails or
returns a direction chosen by rounding noise.

Each of those is a *convention*, and this crate's conventions are documented
in its source. This book makes them teachable: what the convention is, what
breaks when it is violated, and which line of code enforces it.

## What the crate is

<div class="apx-table">
<span class="apx-title">The crate at a glance</span>

| Area | Lines | What lives there |
|---|---|---|
| `core/` | 8 283 | `Problem`, `Variable`, `ResidualBlock`, `NoiseModel`, the loss family, the `Corrector`, marginalization |
| `factors/` | 25 717 | nine factor families, the four decorators, shared utilities |
| `linalg/` | 12 230 | dense and sparse linear solvers, the Schur complement, covariance, damping |
| `linearizer/` | 2 026 | residual/Jacobian assembly, sparse and dense, the evaluation policy |
| `optimizer/` | 8 310 | Gauss–Newton, Levenberg–Marquardt, Dog Leg, the `Optimizer` trait |
| `observers/` | 3 125 | Rerun visualisation of the solve |
| crates | — | `apex-manifolds`, `apex-camera-models`, `apex-io` |

</div>

The crate's own quality bars are strict and worth stating, because they shape
the code you read: no `unwrap` or `expect` (denied crate-wide, tests
included), no `unsafe`, typed errors throughout, and a test suite of roughly
850 tests over 61 000 lines.

## The neighbours

Three sub-crates carry their own cookbooks and should be read alongside this
one: `apex-manifolds` (the groups and their Jacobians),
`apex-camera-models` (projection and unprojection), and `apex-io` (the
recording formats). This book assumes their vocabulary — right perturbations,
$SE_2(3)$ tangents, unit bearings — and derives only what the solver itself
needs.

## Running the code

```console
$ cargo test                       # the whole suite
$ cargo test -p apex-solver        # the solver only
$ cargo bench                      # criterion benches
$ cargo doc --open --no-deps       # the API docs
```

The book's "Check" callouts name the test or bench to run after a chapter.

## What this book is not

It is not a copy of the API documentation — that is `cargo doc`, and it is
excellent. It is not a survey of optimization theory. It is a map of *this*
crate: what each piece is for, what it guarantees, and how the pieces compose
into a solve that is correct rather than merely convergent.
