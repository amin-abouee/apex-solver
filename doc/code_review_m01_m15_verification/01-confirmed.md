# Confirmed Findings — still real on `main` @ `9e193a0`

Re-rated by severity as measured today, not as the review filed them. Every location was
re-read immediately before quoting.

---

## HIGH

### C-1 · Full residual vector deep-copied every iteration, in all three optimizers
*(review: M01-H1/H2/H3 · M04-2 · M10-P1)*

- `optimizer/levenberg_marquardt.rs:892`
- `optimizer/gauss_newton.rs:526`
- `optimizer/dog_leg.rs:1071`

```rust
let residuals_owned = residuals.as_ref().to_owned();
let scaled_step = linear_solver
    .solve_augmented_equation(&residuals_owned, scaled_jacobian, &damping)
```

`residuals` is already `&Mat<f64>`, and both trait methods take `&Mat<f64>`
(`linalg/mod.rs:249,259`). The binding buys nothing and copies the whole residual column —
millions of `f64` on BAL-sized problems — once per iteration, plus once per damping retry
in DogLeg's `while mu_attempts < 10` loop (`dog_leg.rs:1076`).

**Fix:** delete the binding, pass `residuals` directly. Three one-line edits, no API change,
bit-identical results.

### C-2 · Explicit Schur clones the Hessian and gradient it is about to stop using
*(review: M01-H4)*

`linalg/sparse/explicit_schur.rs:1104,1107` (normal) and `:1159,1162` (augmented):

```rust
self.hessian = Some(hessian.clone());
self.gradient = Some(gradient.clone());

// 2. Extract blocks
let h_cc = self.extract_camera_block(&hessian)?;
```

Two observations the review did not make:

1. **The `gradient` clone is dead outright.** `gradient` is never read after the assignment
   — the solve path uses `neg_gradient`, built at `:1099`/`:1154` *before* the clone. It can
   become `self.gradient = Some(gradient);` with no reordering at all.
2. The `hessian` clone only exists because block extraction borrows the local afterwards.
   `extract_camera_block` / `extract_coupling_block` / `extract_landmark_blocks` all take
   `&self`, so moving the two assignments below the extraction block (and below the `debug!`
   dimension logging at `:1174–1178` in the augmented variant) removes it.

This clones the largest object in the solver — sparse `JᵀJ` — twice per iteration.

**Fix:** drop the `gradient` clone; sink the `hessian` assignment past the last use.

### C-3 · λ damping via a linear `find` over the whole H_cc triplet list
*(review: M10-P2)*

`linalg/sparse/explicit_schur.rs:1188–1205`:

```rust
let mut h_cc_triplets = Vec::new();
for col in 0..h_cc.ncols() { /* … push every (row, col, val) … */ }
for i in 0..cam_size {
    if let Some(entry) = h_cc_triplets.iter_mut().find(|t| t.row == i && t.col == i) {
        *entry = Triplet::new(i, i, entry.val + damping.diagonal_term(entry.val));
    } else {
        h_cc_triplets.push(Triplet::new(i, i, damping.diagonal_term(0.0)));
    }
}
```

The full triplet list is rebuilt each solve, then scanned linearly **once per camera column
index** — O(cam_size × nnz(H_cc)) — and finally re-sorted by
`try_new_from_triplets`. For a 10k-camera BA this dominates the Schur solve.

Note the review's suggested one-liner (`if row == col { val + lambda }`) needs adjusting:
damping is `damping.diagonal_term(val)`, a clamp, and structurally-absent diagonals still
need the `else` branch. The cleaner fix is the one already used elsewhere in the codebase:
`NormalEquationsCache::damped_hessian` (`linalg/sparse/normal_eq.rs`) caches `diag_pos` —
the position of each column's diagonal entry — and edits values in place. Apply the same
pattern to H_cc, precomputing its diagonal positions in `initialize_structure`.

### C-4 · `BetweenFactor<Rn>` panics for any dimension ≠ 3 — **reproduced**
*(review: M09-H1)*

`apex-manifolds/src/rn.rs:319–327`:

```rust
fn jacobian_identity() -> Self::JacobianMatrix {
    // Default to 3D identity for compatibility
    DMatrix::identity(3, 3)
}
```

Verbatim output of a throwaway integration test calling `BetweenFactor::<Rn>::linearize`
(control at `dim = 3`, then 2 and 4):

```
running 4 tests
test scratch_rn_between_dim2 ...
thread 'scratch_rn_between_dim2' panicked at
  nalgebra-0.33.3/src/base/blas_uninit.rs:142:5:
Gemv: dimensions mismatch.
FAILED
test scratch_rn_between_dim3 ... dim=3: OK, jacobian_shape=(3,6), residual=[-0.5, -0.5, -0.5]
ok
test scratch_rn_between_dim4 ...
thread 'scratch_rn_between_dim4' panicked at
  nalgebra-0.33.3/src/base/blas_uninit.rs:142:5:
Gemv: dimensions mismatch.
FAILED

test result: FAILED. 2 passed; 2 failed
```

**The propagation path is narrower than the review states** — see
[03-corrections.md](03-corrections.md#c2). `zero_jacobian()`'s 3×3 shape is harmless
(always overwritten); the fault enters through `LieGroup::between`'s default impl
(`apex-manifolds/src/lib.rs:449`), which assigns `Self::jacobian_identity()` — a
dimensionless associated function — into `jacobian_other`, while the sibling assignment at
`:445` correctly yields n×n via `adjoint()`. `between_factor.rs:204` then multiplies the
n×n and 3×3 operands.

This sits behind a documented API claim: `between_factor.rs` advertises support for
"SE(2), SE(3), SO(2), SO(3), and Rⁿ". There is **no test anywhere** exercising
`BetweenFactor<Rn>` — grep for it returns nothing.

**Fix:** give `Rn` a dimension-aware Jacobian. `Rn::jacobian_identity_with_dim(dim)` already
exists (`rn.rs:764`) but is unreachable from the trait's associated-function signature, so
either add a `&self`-taking trait method or restrict `BetweenFactor`'s bound to fixed-DOF
groups. `Rn::apply_to_vector` (`rn.rs:290`) hardcodes 3×3 for the same reason and needs the
same treatment. Add regression tests at `Rn(2)` and `Rn(4)`.

### C-5 · Unbounded decompression cache in the ROS1 reader
*(review: M12-H1)*

`apex-io/src/rosbag/ros1/reader.rs:41`, `:359–363`:

```rust
chunk_cache: HashMap<u64, Vec<u8>>,
…
if !self.chunk_cache.contains_key(&chunk_pos) {
    let inflated = self.inflate_chunk(chunk_pos)?;
    self.chunk_cache.insert(chunk_pos, inflated);
}
```

Every decompressed chunk is retained for the reader's lifetime. Chunks are typically MBs and
bags are GBs, so a full pass accumulates the entire *uncompressed* bag in RAM. Sequential
iteration only ever needs the current chunk.

**Fix:** LRU of depth 1–2, which preserves the temporal locality the cache was added for.

---

## MEDIUM

### C-6 · LM silently substitutes a different linear solver than configured
*(review: M05-H2 — partial)*

The arch round fixed two of three optimizers. `gauss_newton.rs:818` and `dog_leg.rs:1459`
now return `InvalidParameters` naming the requested solver. **LM was not updated** and still
coerces in both arms:

```rust
// levenberg_marquardt.rs:1240 (Dense)
_ => { let mut solver = DenseCholeskySolver::new(); … }
// levenberg_marquardt.rs:1270 (Sparse)
_ => { let mut solver = SparseCholeskySolver::new(); … }
```

Requesting `DenseCholesky` on a sparse problem runs sparse Cholesky with no signal. The
inconsistency is now worse than when the review was written, because the same mistake is
loud in two optimizers and silent in the third.

**Fix:** mirror the GN/DogLeg arms. Mechanical, and the error text can be copied.

### C-7 · Cached Hessian cloned twice per undamped iterative-Schur solve
*(review: M01-H5 / M10-P4 — partial, downgraded from HIGH)*

The borrow-checker workaround the review quoted no longer exists. What remains
(`implicit_schur.rs:1019–1020`):

```rust
self.system_hessian = Some(hessian.clone());
self.hessian = Some(hessian.clone());
self.gradient = Some(gradient);

self.solve_with_system(&hessian, &neg_gradient)
```

Two full CSC copies of the same matrix. `solve_with_system` takes `&SparseColMat` and
`&mut self` over disjoint fields, so calling it first and moving `hessian` into one of the
two slots afterwards removes one clone; removing both needs `Arc` or a single field with an
"is damped" flag (the augmented path at `:1045–1047` already stores distinct matrices).

### C-8 · Schur complement accumulated row-major, read column-major
*(review: M10-8 · M12-L1 · M12-L2)*

`explicit_schur.rs:778–932`. `s_dense` is indexed `[row * cam_size + col]` throughout
accumulation, then drained by

```rust
for col in 0..cam_size {
    for row in 0..cam_size {
        let val = s_dense[row * cam_size + col];
```

— an inner loop striding by `cam_size` on every element, into a `Vec::new()` with no
capacity, which `try_new_from_triplets` then re-sorts. The symmetrisation loop at `:910–916`
touches the buffer in both orders. Separately, `extract_camera_block` (`:476`) and
`extract_coupling_block` (`:509`) rebuild triplet lists from `Vec::new()` and re-sort every
solve although H_cc/H_cp sparsity is invariant.

**Fix:** swap the drain loop nesting (or store column-major), `Vec::with_capacity`, and
precompute the extraction index maps in `initialize_structure`.

### C-9 · LM's Schur path initialises the whole optimization twice
*(review: M10-10 · M04-L4)*

`levenberg_marquardt.rs:1251` calls `initialize_optimization_state(problem)` to obtain
`variables`/`variable_index_map` for `solver.initialize_structure(…)`; `optimize_with_mode`
then calls it again at `:1007`. Each call (`optimizer/mod.rs:559–592`) clones the entire
variable slotmap, rebuilds the symbolic structure, allocates a fresh `AssemblyWorkspace`,
**and runs a full parallel residual + cost evaluation**.

This is the default path for `for_bundle_adjustment()`. On a 20-iteration BA solve it is
roughly one extra iteration of setup plus a duplicated symbolic build.

**Fix:** build once at `:1251` and thread the state into `optimize_with_mode`.

### C-10 · Visibility index rebuilt inside every iterative-Schur solve
*(review: M10-11)*

`implicit_schur.rs:860` calls `build_visibility_index(hessian)` (defined `:783`) from inside
`solve_with_system`, i.e. once per LM iteration, repopulating
`camera_to_landmark_visibility: Vec<Vec<usize>>` (`:100`) by scanning H_cp. The structure is
sparsity-invariant, and the nested-`Vec` layout is cache-hostile.

**Fix:** build once in `initialize_structure`; store flat CSR-style arrays.

### C-11 · CSC pattern cloned on every assemble call
*(review: M01-H6)*

`linearizer/cpu/sparse.rs:134` — `SparseColMat::new_from_argsort` consumes the pattern by
value, so `col_ptr` and `row_idx` (O(nnz)) are copied every iteration although they never
change.

**Fix:** own the pattern in solver state, or keep a spare clone to alternate against.

### C-12 · Per-call allocations in the projection factor's hot path
*(review: M01-M6, broadened)*

`projection_factor.rs:428,437,447`. The review flagged only `fixed_landmarks.clone()` — the
`LANDMARK = false` branch. See [04-new-findings.md](04-new-findings.md#nf1) for the branch
that actually runs in bundle adjustment.

### C-13 · Remaining MEDIUM items, verified present

| ID | Location | One-line |
|---|---|---|
| M01-M2 | `dog_leg.rs:886,1052,1126` | step clones held only for a later reborrow |
| M01-M3 | `residual_block.rs:282–299` | legacy owned-value API clones every variable + double-buffers |
| M01-M4 | `optimizer/mod.rs:357–367` | `Mat::zeros(total_dof,1)` per rejected step; use a sign flag |
| M01-M5 | `between_factor.rs:183–196` | Jacobians computed even when the caller passes `None` |
| M02-1 | `residual_block.rs:125` | `Box<dyn Factor + Send>` erases the trait's `Sync` |
| M02-2 / M12-M3 | `sqlite.rs:225–235` | `Box<dyn Iterator>` facade over full materialisation |
| M04-L2 | `optimizer/mod.rs:329–339`, `variable.rs:359–371` | step copied twice per variable |
| M05-M1 | `projection_factor.rs:428,437` | missing fixed state silently becomes identity / empty |
| M05-M2 | `problem.rs:197–212` | invalid bounds warn-and-drop instead of `Result` |
| M05-M3 | `problem.rs:34–35` vs `variable.rs` | constraints have two sources of truth |
| M06-2 | `projection_factor.rs:443–445` | `.ok()` swallows intrinsics decode failure |
| M07-1 | `dds/subscriber.rs:79–94` | `listen()` returns `Ok` before the thread initialises |
| M09-M1 | `core/problem.rs` | aggregate root also does file I/O and evaluation |
| M09-M2 | `problem.rs:69–72`, `se3.rs:420–425` | parameter length checked only by `debug_assert` |
| M10-11 | see C-10 | |
| M11-H1 | `Cargo.toml:143–154`, `:33` | six dead root deps + workspace `memmap2` |
| M11-M2 | `Cargo.lock` | 3× faer, 3× rand from comparison dev-deps |
| M12-M1 | `ros1/reader.rs:3` | no `Drop`; doc claims memory-mapping that does not exist |
| M13-1 | `apex-manifolds/src/lib.rs:74–131` | only error enum not using thiserror |
| M13-2 | `src/error.rs:197–200` | `log_with_source` drops the source; `source()` stays `None` |
| M13-4 | `dds/subscriber.rs:91,161` | `ThreadJoin` reused for two unrelated failures |
| M15-H1 / M15-M1 | LM `:993` (235 ln), DL `:1201` (227), GN `:591` (195), `explicit_schur.rs:778` (155) | loops have **grown** since the review |
| M15-H2 | `explicit_schur.rs:1188` | last damping implementation not yet unified |
| M15-M2 | `pose_graph_g2o.rs:323` | `unreachable!()` between stringly-typed tables |
| M15-M3 | `optimizer/mod.rs:146–159` | commented-out `IterativeState` |

---

## LOW

<a id="m06-1"></a>
### C-14 · Silent missing-key skip — real code, **unreachable today**
*(review: M06-1 / M09-M3 — downgraded from MEDIUM)*

The gather loop still skips silently (`linearizer/mod.rs:207–214`):

```rust
for &var_key in &residual_block.variable_keys {
    if let Some(variable) = variables.get(var_key) {
        param_slices.push(variable.as_param_slice());
        …
    }
}
```

But the reachability argument the review rested on no longer holds. Every registration path
funnels through `try_add_residual_block_impl` (`core/problem.rs:141–172`), which rejects
unknown keys, and `Problem` exposes **no `remove_variable`**, so a registered key cannot
later dangle. Probed directly:

```
A: REJECTED at registration -> Variable error: residual block references
   unknown variable key VarKey(6v1)
```

**Keep the fix, lower the priority** — and note *why* it matters: the sliding-window design
in `doc/sliding_window/` introduces variable removal, which makes this reachable. Convert
the skip to an error before that work lands, not after.

### C-15 · Remaining LOW items, verified present

| ID | Location | One-line |
|---|---|---|
| M01-L1 | `implicit_schur.rs:408,579` | `block.clone().try_inverse()` — clone wasted on the common success path |
| M01-L2 | `levenberg_marquardt.rs:1112,1189` | debug-gated stats clones |
| M01-L3 | `problem.rs:476` | `set_covariance(cov.clone())` while `per_var` is also returned |
| M01-L4 | `linalg/utils.rs:14,41` | `sparse_to_dense`/`dense_to_sparse` have no non-test callers |
| M03-1 | `visualization.rs:1853` | comment claims `RefCell` permits multi-thread access; it does not |
| M05-L1 | `problem.rs:183–191` | `fix_variable` accepts `idx ≥ DOF`, silently no-ops forever |
| M05-L2 | `apex-manifolds/src/lib.rs:493–496` | `DIM == 0` sentinel, now behind `is_dynamic()` |
| M06-L1 | `variable.rs:365,374` | `step[..dof]` unguarded; document `# Panics` |
| M06-L2 | `bin/bundle_adjustment.rs:409–431` | float division by possibly-zero `num_obs` / `initial_cost` / `iterations` |
| M06-L3 | `apex-io/bin/bag_convert.rs` | 21 `-> Option<…>` helpers erase the failure reason |
| M09-L1 | `problem.rs:174–181` | `remove_residual_block` leaves permanent row-space gaps |
| M09-L2 / M15-L2 | `examples/loss_function_comparison.rs:465` | HashSet→Vec→sort; cosmetic |
| M10-L1 | `cpu/sparse.rs:131`, `cpu/dense.rs:71`, `problem.rs:335` | `Mat::from_fn` element-wise residual copy |
| M11-M1 / M15-M4 | `gauss_newton.rs:244,382`; `dog_leg.rs:373,593` | dead `enable_visualization` field (setters now documented as no-ops) |
| M12-L1 | `explicit_schur.rs:476,509` | H_cc/H_cp extraction rebuilds triplets per solve |
