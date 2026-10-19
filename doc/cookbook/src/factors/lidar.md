# LiDAR Factors

The LiDAR family is where a solver earns its keep: a sweep brings tens of
thousands of points, and a factor per point would be a factor graph with tens
of thousands of blocks. The crate's answer is a family of factors that take
*many* points at once, with the point-to-plane geometry compressed into a
handful of rows. This chapter is the family, its two topologies, and the
compression that makes it possible.

## 1. Two topologies, deliberately

<div class="apx-table">
<span class="apx-title">The LiDAR module's structure</span>

| Topology | Factors | The unknown |
|---|---|---|
| **two-pose** | `distance_field` (`IcpFactor`), `edge` (`LidarEdgeFactor`) | a scan registered against a reference frame, the query point baked into the factor |
| **pose-and-point** | `plane` (`LidarPlaneFactor`, `PointToPlaneFactor`), `point_to_point` (`PoseToPointFactor`), `gicp` (`GicpFactor`), `map_plane`, `map_planes`, `plane_bundle`, `plane_moments`, `submap_field` | the body-frame point is a *variable* |

</div>

The two-pose family treats a measurement as a relation between two poses; the
pose-and-point family treats the point as an unknown in the graph. The
module's own comment calls the coexistence deliberate: the two answer
different questions, and a consumer that registers both must be clear about
which is which.

## 2. The plane factors

The workhorse is a point-to-plane residual:

$$
r_k = \frac{\mathbf{n}_k^\top (X \odot \mathbf{s}_k) + d_k}{\sigma_k},
$$

where $\mathbf{s}_k$ is the point in the body frame, $(\mathbf{n}_k, d_k)$ the
map plane, and $\sigma_k$ the match's uncertainty.

<div class="apx-table">
<span class="apx-title">The plane family</span>

| Factor | Shape |
|---|---|
| `LidarMapPlaneFactor` | one point against one map plane |
| `LidarMapPlanesFactor` | a whole sweep's matches in one factor — `PlaneMatch` list, one block on the pose |
| `LidarPlaneBundleFactor` | one plane over several frames, the plane eliminated |
| `LidarPlaneMomentFactor` | the same, with each frame's points summarized by their moments |
| `PrecomputedPlane`, `Plane`, `LidarPointChain` | the shared plane types and the point-chain helper |

</div>

<div class="apx-widget" data-widget="plane-fit" data-title="From points to a plane" data-foot="Add points to a voxel and watch the normal lock in; the smallest eigenvalue is the residual thickness."></div>

## 3. The moment compression

`LidarPlaneMomentFactor` is Voxel-SLAM's local-mapping factor, and the
interesting part is its algebra. A frame's points reach the world affinely,
$\mathbf{w}_k = A\mathbf{s}_k + \mathbf{b}$, so the frame's total squared
distance to the plane is a quadratic form in the moments:

$$
\sum_k (\mathbf{n}^\top\mathbf{w}_k + d)^2 = \mathbf{c}^\top M\mathbf{c},
\qquad
M = \sum_k \tilde{\mathbf{s}}_k\tilde{\mathbf{s}}_k^\top .
$$

Centring on the frame's mean $\boldsymbol{\mu}$ kills the cross term, and the
sum splits into a spread term and a mean term:

$$
(A^\top\mathbf{n})^\top S(A^\top\mathbf{n}) + N\,(\mathbf{n}^\top(A\boldsymbol{\mu}+\mathbf{b})+d)^2 ,
$$

with $S$ the centred scatter. Writing $S = LL^\top$, the frame contributes
**four** rows:

$$
\left[\,L^\top A^\top\mathbf{n}\;;\;\sqrt{N}\,(\mathbf{n}^\top(A\boldsymbol{\mu}+\mathbf{b})+d)\,\right]/\sigma ,
$$

whatever the number of points. A leaf carrying 100 000 mapped points
contributes the same four rows per participating frame as one with five.

The plane itself is eliminated — re-fit on every evaluation — which is what
makes the factor's Jacobian exact rather than a linearization about a stale
plane: at the optimum, the derivative through the plane parameters vanishes
(the envelope theorem), so the gradient of the reduced cost is exact.

Two details make the compression usable. First, the moments are **centred** on
the frame's mean: the raw $4 \times 4$ moment matrix of points tens of metres
away spans ten orders of magnitude, and a square root of it is accurate only
to about $10^{-4}$ relative — residuals in the thousands for a plane whose true
cost is around $10^{-2}$. Second, world points of frames already outside the
window ("fixed" frames) contribute the same four residuals with $A = I$,
$b = 0$ and **no Jacobian**, so a frame that has left the window still
constrains the plane.

## 4. Why the action is affine

Every factor that moves a body-frame point into the world needs
$\partial(X\odot\mathbf{s})/\partial X$. The crate computes it from four
group-action evaluations:

$$
\frac{\partial (X\odot\mathbf{s})}{\partial X}
= J(\mathbf{0}) + \sum_j s_j\left(J(\mathbf{e}_j) - J(\mathbf{0})\right),
$$

using the fact that the action's Jacobian is affine in the point for every
group that acts on points. One formula, every group, four evaluations per
factor rather than per point.

## 5. Distance fields and submap fields

Two factors work against a *field* rather than against discrete planes:

- `IcpFactor` (with the `DistanceField` trait) registers a scan against a
  reference frame's field — the log-odds occupancy field the LIO stack builds
  — with a residual normalized by the field's gradient;
- `LidarSubmapFieldFactor` and `SubmapAlignmentFactor` do the same against a
  submap's field, with the alignment as a variable.

The field's residual is $L(\mathbf{p}) / \sqrt{L_{\min}^2/9 + \sigma_d^2
\|\nabla L\|^2}$ — a log-odds level divided by a scale built from the field's
own slope and the measurement noise, which turns a level set into an
approximate metric distance.

## 6. Worked example

A sweep with 4 000 matches against the map's planes. `LidarMapPlanesFactor`
takes the `PlaneMatch` list and the pose, and produces 4 000 rows with one
block — the whole sweep in one residual block, with the Jacobian accumulated
in 512-row chunks so the matrix is never materialized. The odometry's problem
therefore has one block per sweep rather than one per point, and the
per-iteration linearization cost is bounded by the sweep count, not the point
count.

## 7. The catch

<div class="apx-warning">
<span class="apx-title">Warning — the moment compression changes what the residual means</span>

Four rows per frame are not four measurements; they are a *summary* of
thousands, with the summary's own weighting (the centred scatter). A factor
consumer that wants to weight individual points differently — by range, by
incidence angle, by a learned quality — cannot use the moment factor, and
must use the point-level one and pay the cost. The compression is a
*modelling* choice about which points matter, not a numerical trick.

</div>

<div class="apx-note">
<span class="apx-title">Note — whitening inside the LiDAR factors</span>

Several of these factors apply their own weighting — GICP's combined point
covariance, the preintegration-style information, the moment factor's $\sigma$ —
and therefore declare `whitens_internally() -> true`. They must be registered
with `NoiseModel::null()`, and the crate rejects the combination that would
whiten twice (chapter [Noise Models](../core/noise.md)).

</div>
