# LiDAR

`factors::lidar` — scan registration. Correspondences are established upstream:
these factors evaluate one already-matched pair, so the data association is the
front end's job.

Two graph topologies coexist deliberately. The **two-pose** family
(`distance_field`, `edge`, and the `LidarPlaneFactor` alias) registers a scan
against a reference frame with the query point baked into the factor. The
**pose-and-point** family (`plane`, `point_to_point`, `gicp`) treats the
body-frame point as a variable. Pick the one matching how your front end
parameterizes the scan.

The derivative every pose-and-point factor needs is `SE3::act`'s:

$$
\frac{\partial (T\mathbf{p})}{\partial(\delta\boldsymbol{\rho},\delta\boldsymbol{\theta})}
= \big[\;R \;\big|\; -R[\mathbf{p}]_\times\;\big],
\qquad
\frac{\partial (T\mathbf{p})}{\partial \mathbf{p}} = R
$$

---

## `IcpFactor<F>` — point-to-field

Aligns a query point against a distance or occupancy field defined in another
frame. Generic over `F: DistanceField`, which returns the field value and its
gradient at a query point.

**Blocks** $[T_{WA},\ T_{WB}]$ — residual 1, Jacobian $1\times12$. Whitens
internally with $1/\sigma$.

### Error

$$
T_{AB} = T_{WA}^{-1}T_{WB},
\qquad
\mathbf{p}_A = T_{AB}\,\mathbf{p}_B,
\qquad
r = \frac{1}{\sigma}\cdot\frac{f(\mathbf{p}_A)}{\lVert\nabla f(\mathbf{p}_A)\rVert}
$$

Dividing by $\lVert\nabla f\rVert$ converts the field value into an approximate
*metric* distance to the surface, so one $\sigma$ is meaningful across fields
of different scaling.

### Jacobian

$$
\frac{\partial r}{\partial \mathbf{p}_A} = \frac{1}{\sigma}\cdot\frac{\nabla f}{\lVert\nabla f\rVert},
$$

$$
\frac{\partial \mathbf{p}_A}{\partial T_{WA}} = \big[\,-C_{WA}^\top \;\big|\; C_{WA}^\top[C_{WB}\mathbf{p}_B + \mathbf{t}_{WB} - \mathbf{t}_{WA}]_\times\,\big],
\qquad
\frac{\partial \mathbf{p}_A}{\partial T_{WB}} = \big[\,C_{WA}^\top \;\big|\; -C_{WA}^\top[C_{WB}\mathbf{p}_B]_\times\,\big]
$$

The residual and Jacobian are zeroed where the field query fails or
$\lVert\nabla f\rVert < 10^{-3}$ — an undefined gradient carries no information,
and a huge one is numerical noise.

---

## `LidarPlaneFactor` — LOAM point-to-plane, two-pose

Not a distinct residual: it is the type alias
`IcpFactor<PrecomputedPlane>`, where the field is a plane through
$\mathbf{q}$ with unit normal $\mathbf{n}$:

$$
f(\mathbf{p}) = \mathbf{n}^\top(\mathbf{p} - \mathbf{q}),
\qquad
\nabla f = \mathbf{n},
\qquad
\lVert\nabla f\rVert = 1
$$

so $r = \sigma^{-1}\mathbf{n}^\top(\mathbf{p}_A - \mathbf{q})$ and the Jacobian
above specializes exactly. Build one with `lidar_plane_factor_isotropic`.

---

## `LidarEdgeFactor` — LOAM point-to-line

The only genuinely distinct LiDAR residual here.

**Blocks** $[T_{WA},\ T_{WB}]$ — residual 3, Jacobian $3\times12$. Whitens
internally.

### Error

With a matched edge line through $\mathbf{q}$ with unit direction $\mathbf{d}$,

$$
\mathbf{e} = W\left(I - \mathbf{d}\mathbf{d}^\top\right)\left(\mathbf{p}_A - \mathbf{q}\right)
$$

the **vector rejection** of the offset onto the line's orthogonal complement.
The scalar cross-product form $\lVert(\mathbf{p}_A - \mathbf{q})\times\mathbf{d}\rVert$
would measure the same distance but is non-differentiable at its own zero,
which is precisely where the optimizer converges.

### Jacobian

The projector $P = I - \mathbf{d}\mathbf{d}^\top$ is constant, so
$\partial\mathbf{e}/\partial\mathbf{p}_A = W P$, chained with the same
$\partial\mathbf{p}_A/\partial T$ blocks as `IcpFactor`.

---

## `PointToPlaneFactor` — plane as a measurement

**Blocks** $[T_{wr},\ \mathbf{p}_\text{body}]$ — residual 1, Jacobian $1\times9$.
Takes an external `NoiseModel`.

$$
r = \mathbf{n}^\top\!\left(T_{wr}\,\mathbf{p}_\text{body}\right) + d
$$

for a target plane $\mathbf{n}^\top\mathbf{x} + d = 0$ with $\lVert\mathbf{n}\rVert = 1$.

$$
\frac{\partial r}{\partial(\delta\boldsymbol{\rho},\delta\boldsymbol{\theta})}
= \big[\;\mathbf{n}^\top R \;\big|\; -\mathbf{n}^\top R\,[\mathbf{p}_\text{body}]_\times\;\big],
\qquad
\frac{\partial r}{\partial \mathbf{p}_\text{body}} = \mathbf{n}^\top R
$$

One constraint per correspondence, along the normal only — which is why three
non-parallel planes are needed to fix a translation.

---

## `PoseToPointFactor`

**Blocks** $[T_{wr},\ \mathbf{p}_\text{body}]$ — residual 3, Jacobian $3\times9$.

$$
\mathbf{r} = T_{wr}\,\mathbf{p}_\text{body} - \tilde{\mathbf{p}}_\text{world},
\qquad
\frac{\partial\mathbf{r}}{\partial\delta} = \big[\,R \mid -R[\mathbf{p}_\text{body}]_\times\,\big],
\qquad
\frac{\partial\mathbf{r}}{\partial\mathbf{p}_\text{body}} = R
$$

The plain point-to-point correspondence, and the $C = I$ special case of GICP.

---

## `GicpFactor` — plane-to-plane

Generalized ICP: the same point-to-point residual, but Mahalanobis-whitened by
the *combined* surface covariance, so tangential directions (flat in both
clouds) contribute almost nothing and the normal direction dominates.
Plane-to-plane registration emerges from point-to-point with the right metric.

**Blocks** $[T_{wr},\ \mathbf{p}_\text{body}]$ — residual 3, Jacobian $3\times9$.
Whitens internally.

### Error

$$
C = C_\text{target} + R\,C_\text{body}R^\top,
\qquad
\mathbf{e} = T_{wr}\mathbf{p}_\text{body} - \mathbf{p}_\text{target},
\qquad
\mathbf{r} = C^{-1/2}\,\mathbf{e}
$$

$C^{-1/2}$ is the symmetric inverse square root, with eigenvalues below
$10^{-12}$ clamped to zero so a rank-deficient covariance yields a
pseudo-inverse root rather than non-finite entries.

### Jacobian

$$
\frac{\partial\mathbf{r}}{\partial\delta} = C^{-1/2}\big[\,R \mid -R[\mathbf{p}_\text{body}]_\times\,\big],
\qquad
\frac{\partial\mathbf{r}}{\partial\mathbf{p}_\text{body}} = C^{-1/2}R
$$

$C$ is evaluated **once at construction** from a caller-supplied rotation hint
and then held fixed, so the rotation-dependence of $C$ is dropped from the
Jacobian. That is deliberate: callers rebuild correspondences (and these
factors) every scan-matching iteration, so the frozen covariance never goes
stale, and freezing it keeps the residual/Jacobian pair exactly consistent.
