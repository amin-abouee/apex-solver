# Solver optimization experiments

The written record of the 2026-09 performance work on the Schur solvers and the
Jacobian assembly: what was hypothesized, measured, accepted or rejected, and on
which machine (Apple Mac mini M4, 10 cores). The code that was accepted is in the
tree; this folder is the reasoning and the numbers, so a rejected idea is not
tried twice.

Method, gates and the six hypotheses (H1-H6): [`optimization_plan.md`](optimization_plan.md).
A change was accepted at a median improvement of at least `max(2 %, 2 x measured
noise)` with unchanged iteration counts and golden costs; everything else was
reverted and logged.

| # | Experiment | Status | Where it lives now |
|---|---|---|---|
| 001 | [Cache the `H_kk` / `H_ke` extraction per structure](experiment_001_cached_extraction.md) | accepted | `ExtractionCache` in `linalg/sparse/schur/explicit.rs` |
| 002 | [Build `S` on its structural pattern](experiment_002_structural_schur_REJECTED.md) | **rejected**, reverted | not in the tree; read it before trying again |
| 003 | [Occupancy bitmap seeds the sparse `S` output](experiment_003_bitmap_schur_output.md) | accepted | `s_bitmap` in `explicit.rs` |
| 004 | [CSC gather assembly](experiment_004_csc_gather_assembly.md) | accepted | `scatter_has_duplicates` and the plan in `linearizer/cpu/sparse.rs` |
| 006 | [Cache the symbolic Cholesky of `S`](experiment_006_symbolic_cholesky_cache.md) | accepted | `s_llt_cache` in `explicit.rs` |
| 007 | [Charge non-cheirality projection-domain violations](experiment_007_projection_domain_barrier_REJECTED.md) | **rejected**, reverted | `projection_deficit` is defined in apex-camera-models but deliberately **not** charged by `ProjectionFactor` |
| 008 | [Row-parallel `F·v` and `E·u` in the implicit Schur operator](experiment_008_implicit_row_parallel_operator.md) | accepted | `RowSpans` in `linalg/sparse/schur/implicit.rs` |
| 009 | [Odometry cost of 004, and a counting-sort CSC plan](experiment_009_odometry_assembly_regression.md) | accepted | `build_symbolic_structure` |
| 010 | [Re-measure the Schur solver table](experiment_010_schur_table_remeasure.md) | accepted (doc only) | `doc/performance.md` |

(There is no experiment 005.)

Baselines and profiles the experiments refer to:

- [`baseline_criterion.md`](baseline_criterion.md), [`baseline_vs_current_m4.md`](baseline_vs_current_m4.md): the Criterion suite and the large datasets, before and after.
- [`criterion_ab_round2_factor_fixes.md`](criterion_ab_round2_factor_fixes.md): A/B of the round-2 factor-correctness fixes (F1-F9).
- [`verification_post_a3ef66d.md`](verification_post_a3ef66d.md): the commits after `a3ef66d`.
- [`hotpath_profile_report.md`](hotpath_profile_report.md): per-function timing of odometry and bundle adjustment, taken with [hotpath-rs](https://github.com/pawurb/hotpath-rs). The instrumentation itself (an optional `hotpath` feature and `#[hotpath::measure]` attributes) is **not** in the tree; see `doc/roadmap.md`.

The numbers are as measured on 2026-09-25 to 2026-09-27 at the commits each note
names; `doc/performance.md` is the maintained comparison.
