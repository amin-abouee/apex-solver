# Exercises

The crate is the best teacher of its own conventions, because its tests
encode the rules. These exercises are built around reading and extending the
code, and each one names the file to start in.

## 1. Reading exercises

### 1.1 Trace a residual from the graph to the step

Start at `src/core/problem.rs::compute_residual_and_jacobian_sparse` and
follow one block all the way to a step:

1. where is the block's residual written?
2. where is it whitened?
3. where is the corrector applied?
4. where does the block's contribution to $H$ and $\mathbf{g}$ land?
5. where is the step applied to the variable?

**Answer to check:** each stage is in a different module — `linearizer`,
`core::corrector`, `optimizer`. The separation is the point of the design.

### 1.2 Find every internally-whitening factor

The `Factor` trait has a `whitens_internally()` hook. Find every factor
that overrides it, and for each, name the measurement whose weighting is
bound up with the geometry.

**Answer to check:** the IMU factors (the preintegration's information
matrix), the LiDAR factors that apply their own $\sigma$, and the
photometric family (the weights are part of the measurement).

### 1.3 Follow a marginal

Start at `src/core/marginalization.rs::compute` and follow:

1. which blocks are absorbed, and why not the others?
2. how is the local column layout built, and why is "kept first" said to
   matter?
3. where does the sign of $b$ come from?
4. what happens to a direction nobody observed?

**Answer to check:** the absorbed set is the Markov blanket's *factors*; the
layout is kept-first so $\Lambda$ is literally a 2×2 block; $b = -\Lambda_p^+g_p$;
an unobserved direction becomes a **zero row** of $S$.

## 2. Small implementation exercises

### 2.1 Write a factor

Write a factor measuring the **height** of a pose above a known plane
$\mathbf{n}^\top\mathbf{x} + d = 0$:

- `residual_dim` → 1; `jacobian_shape` → `(1, 6)` for an `SE3` block (its
  DoF, not its 7 stored parameters);
- the residual is $\mathbf{n}^\top(R\mathbf{t}) + d - 0$, i.e. the signed
  height of the camera origin;
- the Jacobian is the row $\mathbf{n}^\top R$ against the translation part
  and $\mathbf{n}^\top(\mathbf{r}\times)$ against the rotation part.

Then write the finite-difference test, perturbing through the manifold's own
`plus`.

### 2.2 Make a factor whiten internally

Take the factor from 2.1 and give it its own uncertainty — a height
measurement with a standard deviation of $\sigma$ — by dividing the residual
by $\sigma$ and declaring `whitens_internally() -> true`. Register it with
`NoiseModel::null()`.

Then register it with a non-null noise model and confirm that the crate
**rejects the registration** (chapter [Noise Models](../core/noise.md)).

### 2.3 Compare the optimizers

Build the same problem three ways — `GaussNewton`, `LevenbergMarquardt`,
`DogLeg` — and record the iteration count, the final cost, and the number of
*rejected* steps. Try it twice: once from a good initialization, once from one
perturbed by 20°.

**Answer to check:** Gauss–Newton does not reject steps (it has no acceptance
machinery in the same sense); LM and Dog Leg do, and the perturbed start is
where the difference shows.

## 3. Design exercises

### 3.1 Choose a solver for a corridor

A robot drives down a 200 m corridor. The LiDAR scan cannot tell where along
the corridor it is.

1. What is the degeneracy, and in which direction?
2. What does the Hessian look like in that direction?
3. Which crate's mechanism addresses it directly, and how is it configured?
4. What is the cost of using it when the direction is *not* degenerate?

**Answer to check:** `ProjectedFactor` (chapter
[Decorators](../factors/decorators.md)) projects the Jacobian onto the
well-constrained directions; the cost is that real information is discarded
when the assumption is wrong.

### 3.2 Design a window's marginalization policy

A window of 10 keyframes marginalizes its oldest every step.

1. Which factors are absorbed, and which are re-absorbed on the next step?
2. Why is the eliminated block not block-diagonal, and what does that rule
   out?
3. When should the linearization point be re-frozen, and when should the
   marginal be rebuilt?
4. Which direction is guaranteed to be unobservable, and how should the prior
   represent it?

**Answer to check:** the previous prior is re-absorbed; `CombinedImuFactor`
shares rows between a state and its bias, so the Schur path does not apply;
the first-estimate policy freezes the Jacobian, and a rebuild is needed when
the drift is large; the global heading is unobservable, and it becomes a zero
row.

### 3.3 Budget a solve

A window solve must finish in 8 ms.

1. Count the residual rows and the non-zeros for your own problem.
2. Choose the optimizer and the linear solver, and justify each choice.
3. Decide between the explicit and the implicit Schur, and estimate the
   memory of each.
4. Say what would make you change the answer.

**Answer to check:** the answer is a measurement, not an argument — see
[Benchmarks](../tools/benchmarks.md) and the `micro_kernels` bench.

## 4. Debugging exercises

### 4.1 Break a Jacobian on purpose

Take a working factor and change the sign of one Jacobian column. Write the
finite-difference test and watch it fail; then watch what the *solve* does.
It converges, and the cost is low.

**Answer to check:** a wrong Jacobian produces a plausible, confident, wrong
answer. No optimizer can detect it — which is why the finite-difference test
is not optional.

### 4.2 Flip the marginal's sign

Set $b = +\Lambda_p^+ g_p$ and solve.

**Answer to check:** the prior pushes the state the wrong way by exactly
$2\Lambda^{-1}g$, and the solve still converges. The crate's test
`offset_sign_reproduces_the_conditional_minimizer` exists for exactly this.

### 4.3 Whiten twice

Register an internally-whitening factor with a non-null noise model, using
the panicking registration path, and look at what the cost does.

**Answer to check:** the block's information is squared; the solve converges
faster and more confidently, and the answer is wrong. The typed registration
path rejects it — which is why it exists.

## 5. Project exercises

### 5.1 Add a factor family

Add a factor that measures a **bearing to a landmark with a known position**
from a moving platform — the `BearingFactor` of chapter
[Range and Bearing](../factors/ranging.md), extended to `SE23`:

1. write the residual and the Jacobian;
2. write the finite-difference test;
3. document the frame of every input;
4. add it to the catalogue chapter of this book.

### 5.2 Add a linear solver

Implement a **Lanczos**-style iterative solver for the reduced system, behind
the `LinearSolverType` enum, and benchmark it against the PCG path on the BAL
datasets. The `micro_kernels` bench has the harness.

### 5.3 Write a book chapter

The best exercise in understanding a part of the crate is writing its
chapter: pick a module, follow the house style in `AUTHORING.md`, and check
every constant against the source. The crate's own rule applies — **the code
wins** — and a chapter that contradicts the source is a bug in the chapter.

## 6. The catch

<div class="apx-warning">
<span class="apx-title">Warning — an exercise without a check is a guess</span>

Every exercise above names the file to open and the thing to look for. An
exercise that cannot be checked against the code is a puzzle, not an
exercise: the crate's tests are the arbiter, and where a test is missing,
writing one is the exercise.

</div>

<div class="apx-note">
<span class="apx-title">Note — run the suite first</span>

```console
$ cargo test                    # the whole suite, ~850 tests
$ cargo test -p apex-solver     # the solver only
$ cargo bench --bench micro_kernels
```

A green suite before you start means a red suite afterwards is *yours*.

</div>
