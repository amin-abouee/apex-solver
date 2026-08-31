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
