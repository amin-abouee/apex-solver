# Apex Manifolds Cookbook

The mathematical reference for the `apex-manifolds` crate — the Lie groups and
manifold operations used by Apex Solver. It documents every group and every
operation (exp/log, compose, inverse, adjoint, left/right Jacobians, `⊞`/`⊟`)
with formulas derived from the implementation.

## Build

```bash
cargo install mdbook --locked
cargo install mdbook-katex --locked

mdbook serve crates/apex-manifolds/doc/cookbook --open   # live preview
mdbook build crates/apex-manifolds/doc/cookbook          # render to book/
```

## Contents

- **Conventions** — notation, storage layout (quaternion / twist order), the
  `⊞`/`⊟` operators, and the shared SO(3) Jacobians.
- **Manifold Reference** — one chapter per group, all on the same template
  (Representation → Group Operations → Hat/Vee → Exp → Log → Adjoint →
  Jacobians → Plus/Minus → Example → References):
  - SO(2), SO(3) — rotations
  - SE(2), SE(3) — rigid transforms
  - SE₂(3) — extended poses (rotation + position + velocity)
  - SGal(3) — Galilean group (pose + velocity + time)
  - Sim(3) — similarity transforms (rotation + translation + scale)
  - Rⁿ — Euclidean vectors (trivial manifold)

## Add a Page

1. Create the `.md` under `src/manifolds/`.
2. Link it in the **Manifold Reference** section of `src/SUMMARY.md`.
3. Run `mdbook build`.

Math is rendered with the `mdbook-katex` preprocessor (inline `$...$`, display
`$$...$$`).

## License

See the LICENSE file in the project root.
