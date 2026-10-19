# Writing a Factor

A factor is the crate's extension point: one measurement, one residual, one
Jacobian. Everything else — the problem, the linearizer, the optimizers, the
linear solvers — is fixed. This chapter is the contract a factor must
satisfy, the traps that make a factor silently wrong, and the test recipe
that catches them.

## 1. The trait

```rust
pub trait Factor: Send + Sync {
    fn linearize(&self, params: &[&[f64]], residual: &mut [f64],
                 jacobian: Option<faer::mat::MatMut<'_, f64>>);

    fn residual_dim(&self) -> usize;
    fn jacobian_shape(&self) -> (usize, usize);

    fn whitens_internally(&self) -> bool { false }
    fn validate_variables(&self, variables: &[&dyn ManifoldVariable])
        -> Result<(), String> { Ok(()) }
    fn accumulate_normal(&self, ...) -> Option<f64> { None }
}
```

The three required methods are the geometry: the residual, its dimension, and
the shape of its Jacobian. The optional ones are the policies.

## 2. What the arguments mean

- `params` — one parameter slice per connected variable, in the order the
  factor was registered with. The slices are the variables' **parameter**
  vectors, not tangent vectors: an `SE3` contributes 7 values (six of which
  are its tangent dimension).
- `residual` — a pre-allocated buffer of `residual_dim()` elements. The
  factor writes into it; it never allocates.
- `jacobian` — an optional, **column-major** buffer of `jacobian_shape()`. A
  factor that is only ever asked for residuals may skip it; a factor asked for
  a Jacobian must fill it completely.

`Send + Sync` is required, because the linearizer evaluates blocks in
parallel.

## 3. The conventions a factor must respect

<div class="apx-table">
<span class="apx-title">The four rules</span>

| Rule | Why |
|---|---|
| residual is **predicted − measured** | one sign across the crate; the sign of a factor is a property of the data |
| Jacobian is with respect to the **right** perturbation | the solver's step is applied as $X\leftarrow X\mathrm{Exp}(\delta)$ |
| column count is the **DoF** count, not the parameter count | the step vector is indexed by tangent dimensions; an `SE3` block has 6 columns, though the variable stores 7 parameters |
| the residual is in the **model's own units** | whitening is the problem's job, not the factor's |

</div>

The last rule is the one that decides whether the factor whitens internally:
a factor whose weighting is bound up with the measurement (a preintegration's
information matrix, GICP's combined covariance) declares
`whitens_internally() -> true` and must be registered with
`NoiseModel::null()`.

## 4. The optional hooks

**`validate_variables`.** Called once at registration, with the variables the
factor names. It is where a factor checks that the shapes it expects are the
shapes it got — a landmark whose parameter count disagrees with the factor's
observations. Without it, the mismatch surfaces as a panic inside the parallel
assembly, far from the cause.

**`accumulate_normal`.** "Add this factor's $J^\top J$ and $J^\top r$ into
`blocks` and return its cost $\|\mathbf{r}\|^2$, without forming $J$." For a
factor whose structure makes the normal equations far cheaper than the
Jacobian — a plane over many frames, whose Jacobian is mostly zeros — this
turns a matrix operation into a handful of small diagonal products. Returning
`None` means "not provided" and the solver falls back to `linearize`.

## 5. A worked factor

A 1-D range measurement to a point known in the world:

```rust
struct RangeFactor { point: Vector3<f64>, measurement: f64 }

impl Factor for RangeFactor {
    fn linearize(&self, params: &[&[f64]], residual: &mut [f64],
                 jacobian: Option<faer::mat::MatMut<'_, f64>>) {
        let pose = se3_from_params(params[0]);            // world → camera
        let p = pose.act(&self.point);
        let d = p.norm();
        residual[0] = d - self.measurement;               // predicted − measured
        if let Some(mut jac) = jacobian {
            // d/dδ of the translation part, right-perturbation convention
            jac.rb_mut().get_mut(0, 0) = p.x / d;
            jac.rb_mut().get_mut(0, 1) = p.y / d;
            jac.rb_mut().get_mut(0, 2) = p.z / d;
        }
    }
    fn residual_dim(&self) -> usize { 1 }
    fn jacobian_shape(&self) -> (usize, usize) { (1, 6) }
}
```

Three details a reader should check in it: the residual's sign, the
right-perturbation derivative, and the column count (**6** — the `SE3`
tangent dimension, not the 7 stored parameters; the registration path rejects
a mismatch). The crate's own `Factor` documentation carries a similar
example, and the crate's factor tests check the Jacobian against finite
differences.

## 6. Testing a factor

<div class="apx apx-check">
<span class="apx-title">Check — finite differences, every time</span>

A factor's Jacobian is the part most likely to be wrong and least likely to
show up in a solve. The test that catches it: perturb each variable's
tangent by $\pm h$, re-evaluate the residual, and compare the difference
quotient with the analytic column. The crate's factor modules each carry such
a test, and `apex-cv`'s factors (which differ numerically) carry the
equivalent.

</div>

The pattern the crate uses: difference **through the manifold's own `plus`**,
not by adding to the parameter vector. A factor whose Jacobian is taken with
respect to a left perturbation differs from the right one by the adjoint, and
the finite-difference test catches that immediately — if the test differences
the same way the factor claims to.

## 7. The catch

<div class="apx-warning">
<span class="apx-title">Warning — the frame is the part that is usually wrong</span>

The algebra of a factor is easy to get right and the *frame* is easy to get
wrong: a pose that maps world to camera instead of camera to world, a bearing
in the body frame instead of the camera frame, a point in the camera frame
instead of the world frame. Such a factor compiles, converges, and produces a
trajectory that is confidently wrong. Every factor in the crate documents the
frame of its inputs, and the ecosystem books (apex-vio, apex-cv) document the
same conventions in the context of the systems that use them.

</div>

<div class="apx-note">
<span class="apx-title">Note — a factor is not a probability factor</span>

The name is graph-theoretic: a factor is a *piece* of the problem, a function
of the variables. It carries no likelihood, no prior semantics, no
normalization. A prior is a factor of a particular kind (a tangent-space
anchor, an ambient prior), and the marginalization chapter covers the third
kind. Nothing about a factor says how confident it is; that is the noise
model's business.

</div>
