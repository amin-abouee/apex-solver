# New Findings — issues the m01–m15 audit did not report

Five items found while verifying, all present on `main` @ `9e193a0`. Four are hot-path
waste in the same class the audit was hunting; one is a correctness hazard that its m05
"typed keys" verdict actively missed.

---

<a id="nf1"></a>
## NF-1 · LOW (re-rated after measurement) — Two heap allocations per observation per iteration in the BA hot path

**Location:** `src/factors/projection_factor.rs:431–435` and `:447`

The audit's M01-M6 flagged `fixed_landmarks.clone()` — the `OP::LANDMARK == false` branch,
used only under `OnlyIntrinsics` / `OnlyPose`. The branch that actually runs in bundle
adjustment is the *other* one:

```rust
let landmarks: Matrix3xX<f64> = if OP::LANDMARK {
    let flat = params[param_idx];
    let n = flat.len() / 3;
    param_idx += 1;
    Matrix3xX::from_fn(n, |r, c| flat[c * 3 + r])      // :435  heap alloc
} else {
    self.fixed_landmarks.clone().unwrap_or_else(|| Matrix3xX::zeros(0))
};

let camera: CAM = if OP::INTRINSIC { … } else {
    self.camera.clone()                                 // :447
};
```

`Matrix3xX<f64>` is `Matrix<f64, Const<3>, Dyn, VecStorage>` — a heap `Vec`. So every call
copies the landmark parameters out of the zero-copy `&[f64]` slice into a fresh allocation,
immediately defeating the `as_param_slice()` contract that `core/variable.rs:307`
documents as a *"hot-path contract — must never heap-allocate"*.

**Scale.** BA builds **one factor per observation** — `bin/bundle_adjustment.rs:466` and
`benches/bundle_adjustment_benchmark.rs:303` both construct
`Matrix2xX::from_columns(&[Vector2::new(obs.x, obs.y)])`, so `n == 1`. On BAL Ladybug
(~680k observations, 21 iterations) that is on the order of **14 million small heap
allocations per solve**, made from inside the rayon-parallel assembly loop where they
contend on the global allocator.

**Fix (applied):** `evaluate_internal` now accepts `&[f64]`. Both landmark sources are
already column-major triples — the optimizer's parameter slice, and `Matrix3xX::as_slice`
for the fixed branch — so both callers borrow and neither allocates. The camera fallback
became `Option<CAM>` + `unwrap_or(&self.camera)`, removing the clone on every path.

**Measured impact: within noise, contrary to the estimate above.** Median of 5 on
trafalgar-21, before → after: bundle-adjustment 0.56 s → 0.56 s, self-calibration
0.88 s → 0.87 s; ladybug (8k points) 16.5 s → 16.4 s. Final costs bit-identical.

The allocation count was right but the severity was not: each allocation is 24 bytes,
the allocator handles that size well, and the factor's projection arithmetic dominates.
The change is worth keeping — it is strictly less work, removes the clone, and restores
the `as_param_slice()` no-allocation contract that `core/variable.rs:307` documents — but
it is a hygiene fix, not a performance one. Recorded here so the estimate is not repeated.

---

## NF-2 · MEDIUM — `VarKey`s from two different `Problem`s silently collide

**Location:** `src/core/problem.rs:141–172` (`try_add_residual_block_impl`)

m05-type-driven.md opens by praising the key design:

> `VarKey`/`FactorKey` distinct slotmap key types make cross-key mixups compile errors

That is true across key *families*, but not across `Problem` *instances*. Slotmap keys are
`(index, version)` pairs scoped to one map, and two freshly-created `SlotMap`s hand out
identical keys. Probed directly:

```
B: foreign == a (same slot+version)? true
B: ACCEPTED colliding key -> resolves to p's own variable
```

The registration guard added by the arch round only catches keys that are *out of range*:

```
A: REJECTED at registration -> Variable error: residual block references
   unknown variable key VarKey(6v1)
```

So passing a `VarKey` minted by another `Problem` does not error — it silently binds the
factor to whatever variable occupies that slot in the target problem. The result is a
well-formed factor graph wired to the wrong variables, converging to a wrong answer with no
diagnostic anywhere.

**Why it matters now:** `doc/sliding_window/` describes marginalisation across problem
instances, which is exactly the workflow that mints keys in one `Problem` and uses them in
another.

**Fix:** give `Problem` a cheap identity (a `u64` from a process-wide counter) and either
tag `VarKey` with it in debug builds, or assert it in `try_add_residual_block_impl`. A
newtype wrapping `(ProblemId, VarKey)` is the type-driven version.

---

## NF-3 · MEDIUM — Redundant Hessian clone on the undamped iterative-Schur path

**Location:** `src/linalg/sparse/implicit_schur.rs:1019–1020`

```rust
self.system_hessian = Some(hessian.clone());
self.hessian = Some(hessian.clone());
```

The same matrix is cloned twice into two fields. On the undamped path they are by definition
identical — the comment two lines up says so: *"Undamped solve: the operator matrix and the
published Hessian coincide."* One of the two copies is pure waste, at full `JᵀJ` size.

This is adjacent to, but distinct from, the audit's M01-H5: that finding was about a
borrow-checker workaround that no longer exists (see
[02-already-fixed.md](02-already-fixed.md)). This clone is not a borrow workaround, just
duplication.

**Fix:** call `solve_with_system(&hessian, …)` first, then move `hessian` into one field and
leave the other holding the clone — halving the copies. Removing both needs the two fields
to share an `Arc`, or a single field plus an `is_damped` flag.

---

<a id="nf6"></a>
## NF-6 · HIGH — The entire matrix-free Schur solver is dead code

**Location:** `src/linalg/sparse/explicit_schur.rs:219, 234, 1071`;
`src/linalg/sparse/implicit_schur.rs` (whole file, ~1500 lines)

`SparseSchurComplementSolver` holds a delegate:

```rust
iterative_solver: Option<IterativeSchurSolver>,   // :219
```

`initialize_structure` builds one for `SchurVariant::Iterative` (`:1071`) and stores it.
**Nothing ever reads the field** — a workspace-wide grep for `iterative_solver` returns
exactly three hits: the declaration, the `None` initializer, and that one write.

The solve never delegates either. Step 6 of both `solve_normal_equation` and
`solve_augmented_equation` is:

```rust
let delta_c = match self.variant {
    SchurVariant::Iterative => self.solve_with_pcg(&s, &g_reduced)?,
    _ => self.solve_with_cholesky(&s, &g_reduced)?,
};
```

— `self.solve_with_pcg`, a local Jacobi-preconditioned CG on the **already formed** `s`.

So both `SchurVariant`s form the full explicit Schur complement through
`compute_schur_complement`, including its O(cam_size²) dense accumulator; the variant only
chooses the inner linear solve. `IterativeSchurSolver`'s matrix-free operator, its block and
Schur-Jacobi preconditioners, and its visibility index are never executed outside unit
tests. Constructing the delegate is pure waste — `initialize_structure` on it allocates a
second full block structure per solve.

**How this was found:** measuring. Two fixes applied to `implicit_schur.rs`
([C-7](01-confirmed.md), [C-10](01-confirmed.md)) produced *exactly* zero change on every
BA probe, including one that removed a full `JᵀJ` copy per iteration. Code on a hot path
does not behave like that.

**This also corrects the original review.** Its M01-H5 / M10-P4 rated the
`implicit_schur.rs` Hessian clones HIGH, "potentially hundreds of MB each iteration under
`SchurVariant::Iterative`". Those clones cost nothing, because nothing runs them.

**Not fixed here** — deliberately. There are two defensible resolutions and they point in
opposite directions:

- **Wire it up.** Matrix-free PCG never forms `S`, so it sidesteps the O(cam²) dense
  buffer that dominates large-camera solves. Plausibly the better algorithm at Ladybug
  scale and above. But it changes BA numerics and needs its own convergence validation.
- **Delete it.** ~1500 lines, plus the delegate field and half of `SchurVariant`'s meaning.

Either is an architecture decision, not a mechanical fix. Left for a dedicated round.

---

## NF-4 · LOW — The one per-iteration allocation the workspace refactor missed

**Location:** `src/linearizer/cpu/sparse.rs:119`

```rust
let mut jacobian_values = Vec::with_capacity(total_nnz);
```

`AssemblyWorkspace` (perf round `1b7ab76`) absorbed the block order, offsets, Jacobian arena
and residual buffer, but this O(nnz) scatter target is still allocated fresh on every
`assemble_sparse` call. It is capacity-correct, so this is one large allocation rather than a
growth cascade — but it belongs in the workspace next to `jac_arena` for consistency, and it
is the last thing standing between the assembly path and being genuinely allocation-free.

---

<a id="nf7"></a>
## NF-7 · resolved — The Schur drain loop was the dominant cost at scale

**Location:** `src/linalg/sparse/explicit_schur.rs` (drain loop following the
symmetrisation)

`s_dense` is row-major (`[row * cam_size + col]`), but the loop converting it back to
triplets iterated **columns outermost**, striding by `cam_size` on every one of the
`cam_size²` elements.

Fixed by swapping the nesting — `try_new_from_triplets` sorts, so emission order is free —
plus `Vec::with_capacity`.

**Measured: 15.86 s → 8.69 s on ladybug (8k points, explicit Schur), a 45% cut**, final
cost bit-identical at `2.114155e4`. This was the single largest win of the round, and the
original review had it filed as one clause inside a MEDIUM (M10-8).

---

## NF-5 · LOW — Schur symmetrisation walks an O(cam²) buffer in both orders

**Location:** `src/linalg/sparse/explicit_schur.rs:910–916`

```rust
for i in 0..cam_size {
    for j in (i + 1)..cam_size {
        let avg = (s_dense[i * cam_size + j] + s_dense[j * cam_size + i]) * 0.5;
        s_dense[i * cam_size + j] = avg;
        s_dense[j * cam_size + i] = avg;
    }
}
```

Every iteration of the inner loop touches one cache-friendly element and one that strides by
`cam_size`, over a buffer the audit already notes is O(cam²) (M10-8). It is serial, and it
runs on every solve. Blocking the traversal (tiles of ~64) or fusing symmetrisation into the
triplet drain that immediately follows would remove the second stream.

Related to M10-8 but a distinct loop — the audit cited only the accumulation and drain
phases.

---

## Not findings — checked and clean

While verifying, these were swept and confirmed sound, in case they come up later:

- **Rayon determinism** — the audit's claim holds at all `par_iter` sites; the cost reduction
  at `core/problem.rs:332` is still summed serially via
  `results.into_iter().sum::<CoreResult<f64>>()`, so results stay bit-reproducible.
- **Zero `unsafe`** — re-swept workspace-wide, still zero.
- **Zero non-test `unwrap`/`expect`** — 50 raw matches, every one inside a `tests/` target.
- **`normal_eq.rs`'s cache invalidation** — `LazyNormalEquations::ensure` re-keys on the
  Jacobian's pattern fingerprint (`nrows`/`ncols`/`nnz`), and `Clone` resets the cache rather
  than copying a stale one. Correct.
