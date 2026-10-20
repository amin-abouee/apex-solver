# How to Read This Book

## Pick a path

| If you want… | Read |
|---|---|
| to write a factor | [Writing a Factor](./factors/writing.md), then [The Problem](./core/problem.md) and [Noise Models](./core/noise.md) |
| to understand the solve | [What a Factor Graph Is](./core/factor-graph.md) → [Linearization](./core/linearization.md) → [LM](./optimizer/levenberg-marquardt.md) |
| to pick an optimizer or a linear solver | [Choosing a Solver](./optimizer/choosing.md) |
| to implement marginalization | Part IV, and [The First-Estimate Policy](./core/first-estimate.md) |
| to use one of the crate's factors | Part III, by family |
| to migrate | [Migrating 1.4 → 1.5 → 1.6](./tools/migration.md) |
| to see a solve happen | [Observers: Rerun](./tools/observers.md) |

<div class="apx apx-note">
<span class="apx-title">Note — the dependency graph</span>

```text
factor graph ── variables ── manifolds ── camera models (neighbour crate)
      │             │
      │             └── noise ── losses ── corrector
      │                                    │
      └── linearization ── optimizers ── linear solvers ── Schur
                                   │
                                   └── marginalization ── FEJ
```

The arrow means "assumes". Part I is the foundation; Parts II–IV are the
machinery; Part III is a catalogue of what people actually put in the graph.

</div>

## The anatomy of a chapter

1. **The problem** — what breaks without it.
2. **The math** — definitions, then derivation, with algebra in `Derivation`
   callouts.
3. **The algorithm** — numbered steps, in the order the code takes them.
4. **In the code** — files, types, functions, constants.
5. **Worked example** — one number, end to end.
6. **The catch** — the failure the code guards against.

### Callouts

| Colour | Kind | Meaning |
|---|---|---|
| blue | **Note** | an insight, a cross-reference, or a "why this way" |
| green | **Example** | a concrete numeric instance with the arithmetic done |
| purple | **Paper** | the source of an algorithm, and what was taken from it |
| red | **Warning** | a trap, and what the code does about it |
| teal | **Derivation** | the algebra, out of the way of the prose |
| amber | **Check** | something you can verify yourself: a test, a bench, a number |

### Interactive widgets

The book embeds a few interactive diagrams — plain HTML/SVG/JS committed into
the book, no network, no dependencies.

| Widget | Chapter | Try this |
|---|---|---|
| Schur complement | [The Schur Complement](./linalg/schur.md) | watch the fill-in appear as variables are eliminated |
| LM walk | [Levenberg–Marquardt](./optimizer/levenberg-marquardt.md) | move the damping and watch the step change from gradient descent to Newton |
| Epipolar transfer | [Visual Factors: Geometry](./factors/visual-geometry.md) | move a point off the epipolar line and watch the Sampson distance |
| Triangulation | [Visual Factors: Geometry](./factors/visual-geometry.md) | shrink the parallax and watch the depth uncertainty grow |
| Plane fit | [LiDAR Factors](./factors/lidar.md) | add points to a voxel and see the normal lock in |
| Scan match | [LiDAR Factors](./factors/lidar.md) | perturb the pose and watch the point-to-plane residuals |
| Factor graph | [What a Factor Graph Is](./core/factor-graph.md) | switch between the four topologies the crate's factors form |

### Code anchors

Code references name real files and real identifiers. `src/core/problem.rs` is a
path, `NoiseModel` is a type, `Damping::min_diagonal` is a field, `Problem::add_variable`
is a function. Where a chapter quotes a value, the value is the one in this
tree; where a value exists only in code, the chapter names the file so you can
find it after a refactor.

## Building the book

```console
$ cargo install mdbook mdbook-katex        # once
$ cd doc/cookbook
$ mdbook serve                             # http://localhost:3000
$ mdbook build                             # -> book/html
```

Math is pre-rendered at build time by `mdbook-katex`; the KaTeX stylesheet
and fonts are vendored under `assets/katex/` and `src/assets/katex/fonts/`, so
the built book renders fully offline. Figures are inline SVG committed into
the chapters.

## About the numbers in this book

- **Exact** — a constant read from the code. Verifiable by `grep`.
- **Measured** — a number from a bench or a test, quoted with the scene and
  the machine.
- **Illustrative** — a number invented to make an example concrete.

<div class="apx apx-check">
<span class="apx-title">Check — the numbers are greppable</span>

```console
$ grep -rn "fn whitens_internally" src/core/    # the whitening contract
$ grep -rn "JacobianEvaluation" src/linearizer/  # the FEJ policy
$ grep -rn "pub struct .*Factor" src/factors/   # the catalogue
```

</div>
