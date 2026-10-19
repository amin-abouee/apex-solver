# Authoring guide — The Apex Solver Cookbook

Internal contract for anybody writing a chapter of this book. Follow it
exactly; the book is meant to read as one voice with the VIO and CV books in
the family.

## Audience and promise

The reader is a strong programmer who may know *nothing* about nonlinear least
squares, factor graphs, or Lie groups. Every chapter therefore

1. **motivates** the problem in one paragraph,
2. **defines every symbol** before it is used — in a *Where* table,
3. **derives or justifies** the formula the code implements,
4. **states the algorithm** as a numbered list,
5. **anchors it to the code** (file, type, function, constant),
6. ends with a **catch** — the pitfall the implementation guards against.

Never assume the reader remembers an earlier chapter: restate the one-line
version and link. Never write "simply", "just", "obviously", or "trivially".

## Chapter template

```markdown
# Chapter title

One-paragraph motivation.

## 1. The problem
## 2. The math (symbols table, derivation)
## 3. The algorithm (numbered steps)
## 4. In the code (paths, types, functions)
## 5. Worked example
## 6. The catch
```

The order is fixed: *idea → math → algorithm → code → numbers → pitfalls*.

## House style

* Plain, direct, active voice. Short sentences.
* One idea per paragraph; none longer than five lines.
* Callouts: `apx-note`, `apx-example`, `apx-paper`, `apx-warning`,
  `apx-derive`, `apx-check`. A callout body must be separated from the tags
  by **blank lines**, or the Markdown inside it is not parsed:

  ```html
  <div class="apx apx-note">
  <span class="apx-title">Note — title</span>

  Body text with **markdown**, $math$ and [links](./other-page.md).

  </div>
  ```

* Math: `$...$` inline, `$$ ... $$` display, KaTeX only (no `\begin{align}`
  numbering, no `\label`). Multi-line equations in `\begin{aligned}`.
* Code: fenced blocks with a language tag (`rust`, `toml`, `text`, `console`).
  Quote real identifiers — never invent one.
* Tables: plain Markdown pipes, or `<div class="apx-table">` when wide; symbol
  tables use `<div class="apx apx-symbols apx-table">`.
* Figures: `figure.apx-figure` with a numbered caption
  (`<b>Figure 12.3</b> — ...`). The first number is the chapter's position
  in `SUMMARY.md`, counting content chapters from 1 (front matter is chapter
  0); the second counts the figures inside one file, except in the front
  matter, where it continues across `introduction.md`, `conventions.md` and
  `how-to-read.md` (0.1, 0.2, …). Figures are inline SVG, committed in the
  chapter.
* Interactive widgets: `<div class="apx-widget" data-widget="NAME"
  data-title="..." data-foot="..."></div>`. Introduce each in prose *before*
  it appears.
* Links: repository paths in backticks, chapter links relative
  (`[Schur complement](../linalg/schur.md)`). Every forward reference must
  point at a file that exists in `SUMMARY.md`.

## Accuracy rules (non-negotiable)

1. **Every constant, key, function name and default must be checked against the
   source** in this repository (`src/**`, `crates/**`, `Cargo.toml`). When the
   crate and an older document disagree, the code wins; say so explicitly.
2. **Never describe behaviour the code does not have.** Deprecated items are
   labelled deprecated; feature-gated paths are labelled as such.
3. **Equations must be implementable**: a reader should be able to type them
   into Rust and get the same residual the code computes, sign and frame
   included.
4. Where the implementation deviates from its paper, present both and label
   the deviation.
5. Cite the origin of each algorithm in an `apx-paper` callout.

## Available interactive widgets

Shipped in `assets/`, self-contained (no network).

| `data-widget` | What it shows | Suggested chapter |
|---|---|---|
| `solver-graph` | the four topologies the crate's factors form | core/factor-graph |
| `schur` | sparsity, fill-in, Schur solve cost | linalg/schur |
| `lm` | the Levenberg–Marquardt damping walk on a 2-D cost surface | optimizer/levenberg-marquardt |
| `epipolar` | epipolar transfer and Sampson distance | factors/visual-geometry |
| `triangulation` | rays, depth uncertainty, parallax | factors/visual-geometry |
| `plane-fit` | points in a voxel → eigen-decomposition → plane | factors/lidar |
| `scan-match` | point-to-plane residuals of one scan against a map | factors/lidar |
| `deskew` | a rotating sweep, motion-compensated | (neighbour crates) |

## Figure generation

Figures are **inline SVG inside the chapter** — no build step, no external
files. If a figure ever needs to be generated (a data plot from a measured
run), commit the generator next to the image so the figure stays reproducible.
