# Optimization Plan — explicit-Schur-first track

Per user direction: explicit Schur (VIO path) first, implicit (SfM path) later.
Ranked hypotheses; profiling (temporary `Instant` instrumentation, reverted
afterwards) re-ranks before any implementation.

## Profiling instrumentation points

Outer, `src/optimizer/levenberg_marquardt.rs` `optimize_with_mode` loop:
- `iteration_preamble` (residual + Jacobian assembly, conditioning, scaling)
- `compute_step_generic` (linear solve)
- `evaluate_and_apply_step` (trial evaluation)

Inner, `src/linalg/sparse/schur/explicit.rs`:
- `solve_augmented_equation`: `ne_cache.compute` / extract kk+ke+g+damp /
  eliminated gather+damp+invert
- `solve_reduced_system`: `compute_schur_complement`+`compute_reduced_gradient`
  / `solve_with_cholesky` / `back_substitute`

## Hypotheses (explicit-sparse variant)

1. **H1 — structural caching of H_kk/H_ke extraction.** Both extractors walk
   the full JᵀJ and rebuild `Vec<Triplet>` + `try_new_from_triplets` (sort +
   allocate) on every LM iteration, though the sparsity pattern is static per
   structure. Cache the CSC structure + value-position mapping keyed on the
   pattern fingerprint; update values in place per iteration.
2. **H2 — build S on its structural pattern, not a dense kept_dof² buffer.**
   `compute_schur_complement` accumulates a dense `kept_dof²` buffer every
   iteration, scans it O(n²), re-sparsifies through triplets, and re-sorts.
   S's pattern (H_kk pattern ∪ per-landmark clique fill) is static per
   structure: precompute positions once, accumulate into cached CSC values.
   Also removes the kept_dof² memory ceiling (855 MB at ladybug-1723 scale),
   which is what currently forces big problems onto the implicit path.
3. **H3 — cache the symbolic Cholesky of S across iterations.**
   `solve_with_cholesky` runs `SymbolicLlt::try_new` (fill-reducing ordering
   analysis) every iteration. Same caveat as H2: today's S pattern is
   value-dependent because of the 1e-12 filter; on a structural pattern the
   symbolic is cacheable and the filter becomes unnecessary.
4. **H4 — per-iteration `hessian_values.clone()`** in
   `NormalEquationsCache::compute` (one full JᵀJ value memcpy per solve).
   Ownership constraints (solver publishes `Some(hessian)`); measure first.
5. **H5 — `damp_camera_block`**: verify it reuses the pattern rather than
   rebuilding; fold into H1's structure cache if not.
6. **H6 — factor/linearizer evaluation** (outside the linear solver): if the
   profiling shows assembly dominating the iteration, optimize
   `ProjectionFactor` linearize / `assemble_sparse` — pays off across every
   solver variant and every problem type including odometry.

## Gates (per AGENTIC_OPTIMIZATION.md, adapted to measured noise)

- Accept: ≥ max(2 %, 2x warm-protocol noise) median improvement on the
  targeted bench points, no golden violation, iteration counts unchanged
  (golden cost drift ≤ 1e-6 rel / ≤ 1.0005x catches it).
- Routine evaluation: 2-pass small BA (trafalgar-21 explicit variants) +
  odometry. Confirmation: full BA before any commit.
- Revert on failure; log every experiment in this directory.
