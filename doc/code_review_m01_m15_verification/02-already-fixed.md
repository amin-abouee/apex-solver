# Already Fixed — do not act on these

Thirteen findings, including **three of the review's five HIGH performance items**, were
closed by rounds that landed after the audit was written. Each entry names the commit and
the code that replaced it.

Timeline: audit **26 Aug** → perf round 1 **28 Aug** (`doc/perf-round1-results.md`) → arch
round **29 Aug** (`doc/arch-round-results.md`) → noise round **31 Aug**.

---

## F-1 · Jacobi column scaling — was triplets + sparse product, now in place
*(M01-M1 · M04-3 · M10-7)* — perf round, commit `277e97c`

The review found a diagonal `SparseColMat` built from triplets and multiplied into the
Jacobian just to scale each column. `linearizer/mod.rs:339–370` now does:

```rust
let mut values = jacobian.as_ref().val().to_vec();
let columns = split_by_row_offsets_mut(&mut values, &offsets_lens);
columns.into_par_iter().zip((0..ncols).into_par_iter())
    .for_each(|(column, c)| { let s = scaling[c]; for v in column { *v *= s; } });
```

O(nnz), parallel, pattern untouched. Column norms were parallelised in the same commit,
which also supersedes the `compute_column_norms` half of M04-4/M10-6.

## F-2 · Per-iteration block re-sort and scratch rebuild — now a cached workspace
*(M04-1 · M10-P5 · M12-M2 · M12-L4)* — perf round, commit `1b7ab76`

`AssemblyWorkspace` (`linearizer/mod.rs:135–175`) holds `block_order`, `offsets_lens`,
`jac_arena`, `jac_offsets` and `residual_buf`, built once per solve by
`AssemblyWorkspace::build` and threaded through `assemble_sparse`, `assemble_dense` and
`Problem::compute_residual_and_cost_sparse_with_workspace`. The sort the review flagged in
three places now runs once, at `linearizer/mod.rs:153`.

The comments the review called out as false ("These buffers are reused") are now accurate.

## F-3 · λ·I rebuilt from triplets plus a sparse add — now a diagonal edit
*(M10-P3 · M12-L3)* — perf round, commit `c9c1238`

`cholesky.rs:117` and `qr.rs:116` both call `self.ne_cache.damped_hessian(damping)?`.
`NormalEquationsCache` (`linalg/sparse/normal_eq.rs`) caches `diag_pos` — the offset of each
column's diagonal entry within the cached `JᵀJ` value array — so damping is an in-place
value edit on an unchanged pattern. This is exactly the "canonical diagonal damping on CSC
value arrays" the review's roadmap proposed as Wave 1.4.

## F-4 · `Jᵀ` materialised twice per solve — now cached
*(M10-9 · M04-L1)* — perf round, commit `c9c1238`

`transpose().to_col_major()` no longer appears in any solver. `NormalEquationsCache` stores
`jt_pattern` plus a `value_perm` permutation, so each evaluation is a parallel O(nnz) gather
followed by faer's parallel `sparse_sparse_matmul_numeric` into a cached product pattern.

## F-5 · `Mat::zeros` allocated inside the CG loop
*(M10-L2)* — perf round, commit `6736355`

`explicit_schur.rs:711` hoists `let mut ap = Mat::<f64>::zeros(n, 1);` above the iteration
loop, and the CG body now uses faer SIMD kernels (`faer::zip!`, `sparse_dense_matmul`)
instead of scalar `(i, 0)` indexing.

## F-6 · Sim3 silent identity fallbacks and misused `// SAFETY:` labels
*(M13-3)* — noise/correctness round

`apex-manifolds/src/sim3.rs:68–103` replaces `try_inverse().unwrap_or(identity)` with
`regularized_inverse_3` / `regularized_inverse_7`: a Tikhonov-regularised retry that
`warn!`s, and only falls back to identity after an `error!`. A workspace-wide grep for
`SAFETY` now returns exactly one hit — `optimizer/mod.rs:622`, `// CRITICAL SAFETY CHECKS`,
which is not an unsafe-block annotation. The convention hazard the review raised is gone.

---

## Partially fixed — the remainder is tracked in [01-confirmed.md](01-confirmed.md)

| Review ID | Closed | Still open |
|---|---|---|
| **M05-H2** | GN (`gauss_newton.rs:818`) and DogLeg (`dog_leg.rs:1459`) reject unsupported mode×solver pairs (arch round `cd160a0`) | LM still coerces silently in both arms → [C-6](01-confirmed.md) |
| **M06-1 / M09-M3** | `try_add_residual_block_impl` rejects unknown keys; the duplicate gather loop in `problem.rs` was unified into `compute_block_into` (arch round `8073dee`) | the skip itself, now unreachable → [C-14](01-confirmed.md#m06-1) |
| **M01-H5 / M10-P4** | the E0502 clone-to-appease-borrowck is gone (`ne_cache` restructure) | two redundant CSC clones → [C-7](01-confirmed.md) |
| **M15-H2** | `JᵀJ`/`Jᵀr` formation shared via `NormalEquationsCache`; damping unified for Cholesky, QR and implicit Schur | explicit Schur's triplet loop → [C-3](01-confirmed.md) |
| **M11-M1 / M15-M4** | setters now carry `Deprecated: no-op` docs | the field is still dead → LOW |
| **M05-L2** | sentinel wrapped in an `is_dynamic()` predicate (`112f3ae`) | the magic value itself |

## Obsolete

**M06-L4** — "reachable hot-path assert message lacks factor identification". The assert is
`debug_assert_eq!` (`projection_factor.rs:451–458`), compiled out in release, and the arch
round's `Factor::validate_variables` hook (`projection_factor.rs:482–510`, commit `75705cb`)
now rejects the same mismatch at registration with a message naming both counts. Nothing
left to fix.

---

## What this means for the roadmap

`../code_review_m01_m15/fix-roadmap.md` Wave 1 items **1.4** and **1.5** and Wave 2 items
**2.1** and **2.4** are already done. Wave 1.1–1.3, 1.6, 1.7 and Wave 2.2, 2.3, 2.5, 2.6
remain. See [05-revised-roadmap.md](05-revised-roadmap.md).
