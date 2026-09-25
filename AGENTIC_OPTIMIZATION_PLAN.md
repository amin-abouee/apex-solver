# AGENTIC_OPTIMIZATION_PLAN.md — Apex-Solver Performance Program (v2)

> **Handoff document.** Extends `~/Downloads/AGENTIC_OPTIMIZATION.md` (v1)
> with everything learned from the actual code review, profiling, and
> experiments on `feature/criterion`. Written so a fresh agent session can
> resume the optimization program without re-deriving any of it.
>
> **Companion artifacts (all in-repo):**
> - `benchmarking_results/baseline_criterion.md` — measured baselines + protocol
> - `benchmarking_results/experiment_00[1-6]_*.md` — per-experiment evidence
> - `benches/odometry_benchmark.rs`, `benches/ba_benchmark.rs` — the Criterion harness
> - `tests/golden_values.rs` — the odometry goldens this harness also guards
>
> **Status snapshot:** 4 accepted optimizations committed (H1, bitmap Schur
> output, CSC gather assembly, symbolic-Cholesky cache); 3 rejected with
> evidence. BA suite ~1.2–1.4× faster than the recorded baseline; odometry
> improved on the shared assembly path (timing re-measurement pending).

---

## 1. Primary Directive (unchanged in spirit, corrected in mechanics)

Keep iterating until the Criterion benchmarks stop improving and the crate
is as fast as it can be — **without ever trading accuracy**. Every candidate
change passes through the gates in §8. "Faster but less accurate" is a
rejected change, not a win.

Two corrections to v1 that this program learned the hard way:

1. **v1's benchmark template referenced APIs that do not exist**
   (`apex_solver::odometry`, `load_bal_dataset`, `NonLinearSolver`,
   KITTI/TUM-VI datasets). The real APIs are
   `Problem` + `Optimizer`/`LevenbergMarquardt`, `apex_io::{BalLoader,
   G2oLoader, ensure_*_dataset}`, and the datasets are BAL + g2o pose
   graphs. The benches in `benches/` are the corrected realization.
2. **v1's absolute timing gates are meaningless on a thermally unstable
   laptop.** All comparisons run under the thermal protocol in §2.

---

## 2. Environment & Measurement Protocol

### 2.1 Host quirks (must know before running anything)

- **rustup proxies break under the ZCode sandbox**: every command is exec'd
  with `argv[0]` set to the ZCode AppImage name, so `cargo`/`rustc`/clippy
  die with `unknown proxy name`. Workaround (already proven):
  ```bash
  printf '#!/bin/bash\nexec -a cargo /home/aabouee/.cargo/bin/cargo "$@"\n' \
      > /tmp/cargo-wrap && chmod +x /tmp/cargo-wrap
  /tmp/cargo-wrap bench --bench ba_benchmark -- --baseline true_baseline
  ```
  Child processes (rustc spawned by cargo) are unaffected.
- **`perf` is blocked** (`perf_event_paranoid = 4`, no root). Profiling is
  done with temporary `std::time::Instant` instrumentation + `tracing::debug!`
  in the target stage, one filtered bench run, then `git checkout --` the file.
  Stage *ratios* are trustworthy even when absolute times are not.
- **Python is broken** in the sandbox — do all edits with editor tooling.

### 2.2 Thermal protocol (mandatory)

Host: i7-11800H laptop, `powersave` governor with EPP=performance. Sustained
all-core load drifts **>20 % between runs** (same binary measured 5.61 s
cool-start vs 4.09 s on a later cool-start; single points swung up to 3×).

- **Every benchmark session runs twice back-to-back; pass 1 is the warm-up
  and is discarded, pass 2 is recorded.**
- Cross-session comparisons are only valid via criterion's
  `--baseline true_baseline` (saved data) **plus** the 2-pass discipline.
- For micro-decisions, use **interleaved same-window A/B**: candidate and
  baseline code alternate (`git stash` / `git stash pop`) on one filtered
  bench point; compare medians pairwise.
- **Desktop load is fatal to measurement**: Chrome/renderer at 100 %+ CPU
  inflated a 0.5 s solve to 11.8 s (20×). Check `uptime`/`ps aux --sort=-%cpu`
  before trusting any number. If the user is active, defer timing, keep
  working on correctness/code quality, and re-measure in a quiet window.
- Acceptance gate: ≥ max(2 %, 2× measured noise) median improvement.

### 2.3 Sequential execution (v1 Rule 3, still in force)

Never run two bench processes concurrently; never compile while a bench runs
(the compile steals cores and pollutes samples). Pipeline: bench → edit →
compile → bench.

---

## 3. Benchmark Harness (what exists now)

- `benches/odometry_benchmark.rs` — g2o pose graphs (M3500, intel 2D;
  sphere2500, parking-garage, torus3D 3D). LM + SparseCholesky with the
  exact configs from `tests/golden_values.rs`. Golden final costs asserted
  inside every timed run at 1e-6 relative.
- `benches/ba_benchmark.rs` — BAL (trafalgar-21, trafalgar-257,
  dubrovnik-135, venice-52) × {`schur_explicit_sparse`,
  `schur_explicit_iterative`} via the `for_bundle_adjustment()` preset.
  Per-(dataset, variant) golden costs asserted in every timed run:
  cost ≤ golden × 1.0005 AND reprojection-RMSE ≤ golden + 0.01 px
  (RMSE = √(2·cost/observations), the suite-wide convention).
- Both rebuild the problem in the untimed `iter_batched` setup
  (`BatchSize::PerIteration`) — every timed solve starts from the identical
  initial state (Rule 5).
- Filters: `cargo bench --bench ba_benchmark -- trafalgar-21` (fast probe),
  `-- "trafalgar-257|dubrovnik"` etc. Criterion `--save-baseline
  true_baseline` / `--baseline true_baseline` manage comparisons.
- **Bench files are read-only after the baseline** (v1 Rule 2): goldens and
  configs are frozen; new datasets or variants mean a new experiment and a
  re-pinned baseline.

---

## 4. Baseline & Achieved State

Official `true_baseline` (warm protocol, all golden guards passing) — see
`benchmarking_results/baseline_criterion.md` for the full table. Key
medians: BA 2.198 s (t21) … 68.9 s (d135) sparse; odometry 9.5 ms (intel) …
4.19 s (torus3D).

Committed accepted optimizations (newest last):

| commit | experiment | scope | effect |
|---|---|---|---|
| `9e4d851` | H1 — pattern-cached H_kk/H_ke extraction | BA (explicit Schur) | −10…−25 % |
| `e34a117` | Exp 003 — bitmap-derived Schur output | BA (explicit Schur) | cumulative −16…−28 % |
| `28e910c` | Exp 004 — CSC gather plan for assembly | **shared: BA + odometry, all optimizers** | ~100 ms/iter serial work removed |
| `a3ef66d` | Exp 006 — hash-keyed symbolic Cholesky cache | BA (explicit Schur, Sparse variant) | ~25–30 ms/iter |

Cumulative BA vs `true_baseline`: **1.14×–1.38×** per point (verified
7/8 points at p<0.05; the eighth was load-noised but its clean-window probe
improved −26 %).

---

## 5. Architecture Map (where the hot paths live)

Per LM iteration on trafalgar-257/schur_explicit_sparse (~1010 ms before
optimization; ~790 ms after; stage times from temporary instrumentation):

| stage | file | before | after |
|---|---|---|---|
| residual+Jacobian assembly | `src/linearizer/cpu/sparse.rs::assemble_sparse` | ~105 ms | **~25 ms** (Exp 004) |
| JᵀJ + Jᵀr formation | `src/linalg/sparse/normal_eq.rs::NormalEquationsCache` | ~213 ms | unchanged (faer-parallel, already cached symbolic) |
| H_kk/H_ke extraction | `explicit.rs::extract_kept_and_coupling` → `ExtractionCache` | ~196 ms | **~5 ms** (H1) |
| H_ee gather+damp+invert | `explicit.rs` → `EliminatedBlocks` | ~11 ms | unchanged |
| S formation (dense accumulator) | `explicit.rs::compute_schur_complement` | ~220 ms updates (bandwidth-bound) | unchanged — see §7.1 |
| S output (dense→CSC) | `explicit.rs::build_schur_output` | ~90 ms (scan+sort+sym) | **~10 ms** (Exp 003 bitmap) |
| Cholesky solve | `explicit.rs::solve_with_cholesky` | ~78 ms (sym 28 + num 50) | **~50 ms** (Exp 006 cache) |
| back-substitution | `explicit.rs::back_substitute` | ~10 ms | unchanged |
| step evaluation | `src/optimizer/levenberg_marquardt.rs` | ~16 ms | unchanged |

Odometry path (pose graphs): `assemble_sparse` (improved by Exp 004) +
`SparseCholeskySolver` (`src/linalg/sparse/cholesky.rs` — **already caches**
its symbolic factorization) + SE2/SE3 factor `linearize`. No Schur involved.

---

## 6. Zero-Copy Audit (Problem / Factor / Graph)

Audited for hidden copies; results:

- **`compute_block_into`** (`src/linearizer/mod.rs`) — clean. `param_slices`
  and `variable_local_idx_size_list` are inline `SmallVec`s;
  `variable.as_param_slice()` hands zero-copy slices to
  `factor.linearize`; the Jacobian is written through a `MatMut` view into
  the pre-split arena; noise whitening and the robust corrector run
  in-place; the returned `BlockLinearization` uses `SmallVec<[_; 8]>`
  (no heap for ≤4-variable factors). **No action needed.**
- **Jacobian arena** — pre-split into per-block slices
  (`split_by_row_offsets_mut`), written directly by `factor.linearize`
  (zero-copy), gathered into CSC by Exp 004's plan. **No action needed.**
- **`SymbolicStructure`** — block order, offsets, pattern, and now the CSC
  scatter plan are built once per solve and reused every iteration.
- **Residual buffer → `faer::Mat`** — one 3.6 MB copy per iteration
  (`Mat::from_fn` over the workspace buffer). Known, small (~1–2 % of the
  iteration); a zero-copy return would need an ownership swap of the
  workspace buffer — candidate micro-project, low priority.
- **`problem.variables.clone()`** (`src/optimizer/mod.rs:604`) — deep-clones
  every variable (via `ManifoldVariable::clone_box`) once per `optimize()`.
  Semantics: the optimizer mutates its own copy; the caller reads optimized
  values from `SolverResult.parameters`, while `problem.variables` keeps the
  initial state. Removing it would change that public contract (the
  `Problem` would need move-out/move-back of its `SlotMap`). Cost is
  ~0.2 % per solve at t257. **Documented, intentionally kept.**
- **Factor dispatch** — `Box<dyn Factor>` virtual call per block per
  iteration (~1–2 ms total at t257). Acceptable; a generic/enum dispatch
  rewrite is not worth the API churn.

---

## 7. Accepted Experiments (implementation notes for maintainers)

### 7.1 H1 — `ExtractionCache` (`9e4d851`)
`ExplicitSparseSchur::extract_kept_and_coupling` maps each `JᵀJ` value to
its `H_kk`/`H_ke` slot once per pattern (sorted triplet build + hash lookup
at build, `u32` source-index arrays) — every later solve is two parallel
gathers. The direct `extract_kept_block`/`extract_coupling_block` remain as
test-only references; `cached_extraction_matches_direct_reference` pins
them to exact equality.

### 7.2 Exp 003 — bitmap Schur output (`e34a117`)
The dense accumulator sets one occupancy bit per write
(`bits[col*wpc + row/64]`, column-major word layout). After the updates,
`build_schur_output` walks the bitmap (two passes: count, then fill) and
emits the CSC pattern **and values** in one go — the symmetrize is folded
in by averaging `(dense[r,c] + dense[c,r])·0.5` at read, and the noise
filter (`|v| > 1e-12`, applied to the averaged value) matches the legacy
pipeline exactly, making the output bit-identical to the pre-bitmap path.
Bitmap memory: kd²/8 bytes (~300 KB at t257). Keyed on the `JᵀJ`
fingerprint; a structure change resets it.

### 7.3 Exp 004 — CSC scatter plan (`28e910c`)
`build_symbolic_structure` sorts `(col, row, arena_index)` triples itself
(stable, so duplicate summation order matches push order) and emits
`scatter_ops: Vec<ScatterOp>` — `(csc_slot, arena_index, accumulate)`.
`assemble_sparse` executes the plan: parallel gather in the common
no-duplicate case, serial accumulate loop when a factor lists a variable
twice (faer's duplicate-sum semantics, preserved). The missing-variable
error moved from assemble time to build time (fail-fast). This is the
**shared-path** win: BA, odometry, and every optimizer (LM/GaussNewton/
DogLeg all call `solve_augmented_equation`/`solve_normal_equation`).

### 7.4 Exp 006 — hash-keyed symbolic Cholesky cache (`a3ef66d`)
`build_schur_output` FNV-1a hashes the filtered CSC structure during the
same walk; `solve_with_cholesky` reuses `SymbolicLlt` while the hash is
unchanged (identical symbolic + values ⇒ bit-identical factorization), with
a defensive rebuild if `Llt::try_new_with_symbolic` rejects. Only
first-attempt symbolics are cached; the regularization-retry path (which
widens the pattern) never poisons the cache. The chunked variant passes
`None` and stays cold.

---

## 8. Rejected Experiments (do NOT retry without new evidence)

1. **Structural-position Schur accumulation** (H2a, exp 002): building `S`
   on its structural CSC pattern via positional scatter measured **+62 %**
   (17.5 s vs 10.8 s at t257). The dense row-strip update loop is
   bandwidth-bound; positional scatter destroys its locality.
2. **Parallel chunk-private dense accumulation** (H2b): 16 threads splitting
   the same DRAM traffic (random 8-byte writes into a >L3 buffer) gained
   nothing and added fill+reduce overhead (140–320 ms, high variance vs
   220 ms serial). The update stage is **bandwidth-bound, not
   compute-bound** — thread count does not help it.
3. **PCG warm-starting from the previous LM step** (Exp 005): the primitive
   (`pcg_from`) works and is correct, but warm-started steps shift the LM
   trajectory; t21/iterative converged +1.3 % above its pinned golden —
   the cost gate (×1.0005) rejected it. Revisit only together with the
   accuracy-gate policy (§8 note in v1) or on problems whose goldens are
   not pinned this tightly.

---

## 9. Remaining Roadmap (ranked, with entry points)

1. **Quiet-window re-measurement** (first action of the next session):
   both benches × 2 passes against `--baseline true_baseline`; fill the
   "deferred" notes in experiments 004/006 logs.
2. **Odometry deep-dive**: profile one torus3D/parking-garage iteration
   (same Instant method): split assemble / cholesky / residual-eval.
   Assembly is already optimized (Exp 004); `SparseCholeskySolver` already
   caches its symbolic. The remaining candidate is the SE2/SE3 Jacobian
   math in the pose `BetweenFactor::linearize` (`src/factors/pose/`) and
   the manifold right-Jacobian code in `apex-manifolds`.
3. **Tiled Schur accumulation** (large project, the only real path past the
   bandwidth wall on the 220 ms update stage): group landmarks by camera
   locality into tile-sized dense sub-accumulators so the rank-`dof`
   updates hit cache-resident tiles. Sketch: sort eliminated blocks by
   their minimum kept-row; process tiles of `~L1-sized` camera sets;
   requires a tile-aware `S` output pattern (the bitmap from Exp 003
   extends naturally — bits are already per-write). Entry:
   `explicit.rs::compute_schur_complement`.
4. **Implicit-Schur track (SfM phase, per user direction)**: profile
   `ImplicitSparseSchur::apply_schur_operator` — the `F·v` and `E·u`
   scatter passes are serial (write-aliasing). Candidates: row-block
   partitioned accumulation with per-chunk partials + reduce (same lesson
   as H2b — check bandwidth first), and PCG tuning (`PcgParams`,
   forcing-sequence η). Benchmark implicit-only benches (per user
   instruction) before touching anything.
5. **`hessian_values.clone()`** in `NormalEquationsCache::compute`
   (`src/linalg/sparse/normal_eq.rs:248`) — one full JᵀJ value memcpy per
   solve; ownership lives in the published `SparseColMat`. Candidate:
   transfer ownership into the cache and re-materialize per iteration
   (measure first; ~2–5 ms/iter at t257).
6. **Residual buffer → `Mat` copy** in `assemble_sparse` (~1–2 ms/iter):
   ownership swap of `workspace.residual_buf`, same pattern as
   `jacobian_values`.
7. **`problem.variables.clone()`** (optimizer/mod.rs:604, ~0.2 %/solve):
   only worth it together with an API contract change — needs a user
   decision.

### 9.1 simsimd evaluation (requested; decision documented)

`simsimd` (SIMD dot/cosine/L2 for f64) was evaluated against the actual hot
kernels and **not integrated**, because:

- The dominant arithmetic (`JᵀJ` numeric product, PCG SpMV, Cholesky) is
  already inside **faer's** SIMD/parallel kernels; wrapping it with simsimd
  cannot beat it and risks regression.
- The hand-rolled loops that remain (rank-3 Schur updates) are
  **memory-bandwidth-bound** (measured — §8.1/8.2) and hand-unrolled;
  SIMD would not move the bottleneck.
- The vector ops PCG does per iteration (`norm_l2`, axpy) are tiny
  (`kept_dof` f64, few KB) — launch overhead dominates any SIMD gain.

**Revisit trigger:** if the tiled Schur accumulation (§9.3) lands, its
tile-level `contrib · H_ke` products become cache-resident medium-size
operations — that is where simsimd (or `std::simd`) becomes worthwhile.
Defer until then.

---

## 10. Gates & Rules (consolidated, binding)

- Accuracy first: any golden violation = rejected change, no exceptions.
- BA gates: final cost ≤ golden × 1.0005 AND RMSE ≤ golden + 0.01 px,
  asserted inside every timed run. Odometry gate: 1e-6 relative on pinned
  costs.
- ≥ max(2 %, 2× noise) median improvement, p < 0.05, no point regressing.
- Iteration-count parity is enforced implicitly by the golden cost gates
  (a changed trajectory moves the converged cost).
- Sequential benching; 2-pass thermal protocol; interleaved A/B for micro
  decisions; never trust cross-session absolute times on this laptop.
- Library code only (`src/`, `crates/`); benches/tests frozen post-baseline.
- No `unsafe` (workspace-forbidden), no `.unwrap()`/`.expect()`, no
  `println!` in library code, no `target-cpu=native`, no tolerance loosening.
- Every experiment gets a `benchmarking_results/experiment_NNN_*.md` log,
  accepted or rejected.

---

## 11. Playbook for the Next Agent

1. Read `benchmarking_results/baseline_criterion.md` and the six experiment
   logs — they contain every number in this document.
2. Verify the environment: `/tmp/cargo-wrap` exists (or recreate it, §2.1);
   `uptime` shows a quiet machine before any measurement.
3. Pick the top roadmap item that fits the session budget; write the
   hypothesis into a new experiment log *before* implementing.
4. Instrument (temporary `Instant` + `debug!`) if the stage split is
   unknown; revert instrumentation before benchmarking.
5. Gate in this exact order: `clippy --all-targets` (excluding the known
   broken WIP test) → `cargo test --release` targeted dataset tests →
   2-pass probe on trafalgar-21 → interleaved A/B if the delta is < 10 % →
   full 4-dataset confirmation → commit → fill the experiment log.
6. On any golden violation: revert, log the rejection with the measured
   evidence, move to the next hypothesis. Do not re-pin goldens to admit a
   slower-converging candidate.
