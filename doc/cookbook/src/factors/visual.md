# Visual

`factors::visual` — everything that goes through a camera.

All projection factors share a **smooth cheirality penalty**. A landmark behind
the camera cannot be projected, but returning zero would make an invalid
configuration look optimal, so they emit

$$
r = \Pi_\text{base} + \Pi_\text{scale}\,\max(z_\text{min} - z_\text{cam},\,0),
\qquad
\frac{\partial r}{\partial z_\text{cam}} = -\Pi_\text{scale}
$$

with $\Pi_\text{base} = 10^4$, $\Pi_\text{scale} = 10^3$ — large enough to
dominate any plausible in-image residual, small enough not to ill-condition the
normal equations, and carrying a gradient that pushes the point back in front.

Two derivatives recur throughout. For a **world-to-camera** pose acting
directly, $\mathbf{p}_c = R\mathbf{p} + \mathbf{t}$:

$$
\frac{\partial \mathbf{p}_c}{\partial(\delta\boldsymbol{\rho},\delta\boldsymbol{\theta})}
= \big[\;R \;\big|\; -R\,[\mathbf{p}]_\times\;\big],
\qquad
\frac{\partial \mathbf{p}_c}{\partial \mathbf{p}} = R
$$

For an **inverted** pose, $\mathbf{q} = T^{-1}\mathbf{p}$, a right perturbation
gives $\mathbf{q} \leftarrow \mathrm{Exp}(-\delta)\mathbf{q}
= \mathbf{q} - \delta\boldsymbol{\rho} + [\mathbf{q}]_\times\delta\boldsymbol{\theta}$:

$$
\frac{\partial \mathbf{q}}{\partial(\delta\boldsymbol{\rho},\delta\boldsymbol{\theta})}
= \big[\;-I \;\big|\; [\mathbf{q}]_\times\;\big],
\qquad
\frac{\partial \mathbf{q}}{\partial \mathbf{p}} = R^\top
$$

Both come from `SE3::act`, which reports them itself.

---

## `ProjectionFactor<CAM, OP>`

Monocular reprojection over $N$ observations of one pose, generic over the
camera model and over which blocks are optimized.

**Blocks** $[\,\text{pose}?,\ \text{landmarks}?,\ \text{intrinsics}?\,]$,
selected at compile time by `OP: OptimizationConfig` — `BundleAdjustment`,
`SelfCalibration`, `OnlyPose`, `OnlyLandmarks`, `OnlyIntrinsics`,
`PoseAndIntrinsics`, `LandmarksAndIntrinsics`. Unused blocks cost nothing; the
const-generic flags remove their columns entirely.

**Residual** $2N$, **Jacobian** $2N \times (6\,\text{P} + 3N\,\text{L} + d_\kappa\,\text{I})$.

### Error

$$
\mathbf{p}_{c,k} = R\,\mathbf{p}_k + \mathbf{t},
\qquad
\mathbf{r}_k = \pi(\mathbf{p}_{c,k};\ \boldsymbol{\kappa}) - \tilde{\mathbf{u}}_k
\;\in\; \mathbb{R}^2
$$

### Jacobian

$$
\frac{\partial \mathbf{r}_k}{\partial(\delta\boldsymbol{\rho},\delta\boldsymbol{\theta})}
= \frac{\partial\pi}{\partial\mathbf{p}_c}\big[\,R \mid -R[\mathbf{p}_k]_\times\,\big],
\qquad
\frac{\partial \mathbf{r}_k}{\partial \mathbf{p}_k}
= \frac{\partial\pi}{\partial\mathbf{p}_c}\,R,
\qquad
\frac{\partial \mathbf{r}_k}{\partial \boldsymbol{\kappa}}
= \frac{\partial\pi}{\partial\boldsymbol{\kappa}}
$$

The camera model supplies $\partial\pi/\partial\mathbf{p}_c$ (2×3) and
$\partial\pi/\partial\boldsymbol{\kappa}$ (2×$d_\kappa$) analytically. For a
pinhole model,

$$
\frac{\partial\pi}{\partial\mathbf{p}_c}
= \frac{1}{z}\begin{bmatrix} f_x & 0 & -f_x x/z \\ 0 & f_y & -f_y y/z \end{bmatrix}
$$

Nine models are available (`Pinhole`, `RadTan`, `KannalaBrandt`,
`DoubleSphere`, `Eucm`, `Ucm`, `Fov`, …).

---

## `ExtrinsicProjectionFactor<CAM>`

Reprojection with the **camera-to-body transform as a variable**. On a real rig
the extrinsics are the least trustworthy part of the setup and are shared by
every observation from that camera, so they belong in the graph where many
observations determine them.

**Blocks** $[T_{WB},\ T_{BC},\ \mathbf{p}_w]$ — residual 2, Jacobian $2\times15$.
Both poses read the way a calibration file does: body-in-world,
camera-in-body.

### Error

$$
T_{WC} = T_{WB}\,T_{BC},
\qquad
\mathbf{p}_c = T_{WC}^{-1}\,\mathbf{p}_w,
\qquad
\mathbf{r} = \pi(\mathbf{p}_c) - \tilde{\mathbf{u}}
$$

### Jacobian

With $D = \dfrac{\partial\pi}{\partial\mathbf{p}_c}\big[\,-I \mid [\mathbf{p}_c]_\times\,\big]$ (2×6),

$$
\frac{\partial\mathbf{r}}{\partial \delta_{WB}} = D\;\mathrm{Ad}_{T_{BC}^{-1}},
\qquad
\frac{\partial\mathbf{r}}{\partial \delta_{BC}} = D,
\qquad
\frac{\partial\mathbf{r}}{\partial \mathbf{p}_w} = \frac{\partial\pi}{\partial\mathbf{p}_c}R_{WC}^\top
$$

The $\mathrm{Ad}_{T_{BC}^{-1}}$ factor is $\partial T_{WC}/\partial T_{WB}$,
which `compose` reports directly. The extrinsics block gets $D$ unchanged
because $\partial T_{WC}/\partial T_{BC} = I$.

---

## `TimeOffsetProjectionFactor<CAM>`

Reprojection with the **camera-to-IMU clock offset as a variable**. The two
clocks differ by a constant lag $t_d$; unmodelled it is indistinguishable from
an extrinsic rotation error at constant angular rate, and 20 ms at 1 m/s is
2 cm of position error.

**Blocks** $[\,\text{SE}_2(3)\ \text{state},\ T_{BC},\ \mathbf{p}_w,\ t_d\,]$ —
residual 2, Jacobian $2\times19$.

### Error

A first-order expansion of the state to the exposure instant,

$$
\mathbf{p}(t{+}t_d) = \mathbf{p} + \mathbf{v}\,t_d,
\qquad
R(t{+}t_d) = R\,R_d, \quad R_d = \mathrm{Exp}(\boldsymbol{\omega}\,t_d)
$$

then $T_{WC} = T_{WB}(t{+}t_d)\,T_{BC}$ and the same reprojection as above. The
state is `SE23` because it already carries $\mathbf{v}$; $\boldsymbol{\omega}$
is the bias-corrected gyro sample, a **measurement**, since the graph has no
angular rate to estimate.

### Jacobian

The `SE23` tangent $(\delta\boldsymbol{\rho}, \delta\boldsymbol{\theta}, \delta\boldsymbol{\nu})$
pushes to the exposure pose's SE(3) tangent as

$$
\delta\boldsymbol{\rho}' = R_d^\top\!\left(\delta\boldsymbol{\rho} + t_d\,\delta\boldsymbol{\nu}\right),
\qquad
\delta\boldsymbol{\theta}' = R_d^\top \delta\boldsymbol{\theta}
$$

$$
\frac{\partial T_{WB}}{\partial \text{state}} =
\begin{bmatrix} R_d^\top & 0 & t_d R_d^\top \\ 0 & R_d^\top & 0 \end{bmatrix} \in \mathbb{R}^{6\times9}
$$

and the offset column is the exposure pose's own tangent velocity,

$$
\frac{\partial T_{WB}}{\partial t_d} =
\begin{bmatrix} R_c^\top\,\mathbf{v} \\ \boldsymbol{\omega} \end{bmatrix} \in \mathbb{R}^{6\times1},
\qquad R_c = R\,R_d
$$

Both are then chained with $D\,\mathrm{Ad}_{T_{BC}^{-1}}$ exactly as above. Note
the offset column vanishes when $\mathbf{v} = 0$ and $\boldsymbol{\omega} = 0$:
**a stationary platform cannot observe $t_d$**, which the tests assert directly.

---

## `StereoFactor`

Rectified stereo, so disparity fixes depth in a single observation.

**Blocks** $[\text{pose},\ \mathbf{p}]$ — residual 3, Jacobian $3\times9$.

### Error

$$
u_L = c_x + f_x\frac{x}{z}, \qquad
u_R = c_x + f_x\frac{x-b}{z}, \qquad
v   = c_y + f_y\frac{y}{z}
$$

$$
\mathbf{r} = (u_L, u_R, v) - (\tilde{u}_L, \tilde{u}_R, \tilde{v})
$$

### Jacobian

$$
\frac{\partial(u_L,u_R,v)}{\partial \mathbf{p}_c}
= \frac{1}{z}\begin{bmatrix}
f_x & 0 & -f_x x / z \\
f_x & 0 & -f_x (x-b)/z \\
0 & f_y & -f_y y / z
\end{bmatrix}
$$

chained with $[\,R \mid -R[\mathbf{p}]_\times\,]$ and $R$. Calibration is fixed
at construction — this is the one camera factor not generic over `CameraModel`.

---

## `InverseDepthFactor<CAM>`

A landmark parameterized in its **anchor camera** by pixel $(u,v)$ and inverse
depth $d = 1/z$, which stays well-conditioned for distant, low-parallax points
where a Euclidean 3D point blows up.

**Blocks** $[T_i,\ (u,v,d),\ T_j]$ — residual 2, Jacobian $2\times15$.

$$
\mathbf{X}_i = \frac{\pi^{-1}(u,v)}{d},
\qquad
\mathbf{p}_w = T_{wc,i}^{-1}\mathbf{X}_i,
\qquad
\mathbf{r} = \pi_j\!\left(T_{wc,j}\,\mathbf{p}_w\right) - \tilde{\mathbf{u}}
$$

The anchor block's Jacobian carries $\partial\mathbf{X}_i/\partial d = -\pi^{-1}(u,v)/d^2$,
which is what makes the parameterization behave as $d \to 0$: the *inverse*
depth stays finite and its derivative bounded where $z \to \infty$ would not.

---

## `SmartProjectionFactor<CAM>`

Structure-less: connects $N$ poses to a landmark that is never a variable. The
point is re-triangulated by DLT from the current poses at every linearization,
then eliminated exactly by the implicit Schur complement.

**Blocks** $N$ poses — residual $2N$, Jacobian $2N \times 6N$. Whitens
internally; register with `NoiseModel::null()`.

### Jacobian

Writing $A_i = \sigma_i^{-1}\partial\mathbf{u}_i/\partial T_i$ (2×6) and
$B_i = \sigma_i^{-1}\partial\mathbf{u}_i/\partial\mathbf{p}$ (2×3),

$$
J_{ij} = \delta_{ij}A_i - B_i M_j,
\qquad
M_j = \Big(\sum_k B_k^\top B_k\Big)^{-1} B_j^\top A_j \;\in \mathbb{R}^{3\times6}
$$

The triangulated point satisfies the first-order optimality condition
$\sum_i B_i^\top \mathbf{r}_i = 0$, so this dense $J$ *is* the total derivative
of the re-triangulated residual, and

$$
J^\top J = \sum_i A_i^\top A_i - \Big(\sum_i A_i^\top B_i\Big)\Big(\sum_k B_k^\top B_k\Big)^{-1}\Big(\sum_j B_j^\top A_j\Big)
$$

is exactly the Schur complement. Degenerate geometry (pure rotation,
rank-deficient DLT) returns a bounded penalty; `status()` reports which.

---

## `EssentialMatrixFactor` / `EssentialMatrixConstraint`

2D–2D epipolar geometry over a relative pose, for initialization and
loop-closure verification before any landmark exists. Points are **normalized**
camera coordinates.

$$
\textbf{Factor:}\quad
r_i = \mathbf{p}_{2,i}^\top\,[\hat{\mathbf{t}}]_\times R\;\mathbf{p}_{1,i},
\qquad
\hat{\mathbf{t}} = \frac{\mathbf{t}}{\lVert\mathbf{t}\rVert},
\qquad i = 1\dots N
$$

$$
\textbf{Constraint:}\quad
\mathbf{r} = \begin{bmatrix} \mathrm{Log}(R_E^\top R) \\ \hat{\mathbf{t}} - \mathbf{u}_E \end{bmatrix} \in \mathbb{R}^6
$$

The direction normalization contributes
$\partial\hat{\mathbf{t}}/\partial\mathbf{t} = (I - \hat{\mathbf{t}}\hat{\mathbf{t}}^\top)/\lVert\mathbf{t}\rVert$,
which is rank 2 — both factors are **scale-free by construction**. Only the
translation direction is observable from epipolar geometry, so a graph
containing only these has a one-dimensional null space; the constraint's six
rows are effectively rank 5.

---

## `DepthFactor<ONESIDED>` / `HomogeneousPointFactor`

Both act on a homogeneous 4D landmark $\mathbf{h} = [x,y,z,w]^\top$, the
parameterization that keeps points at infinity representable.

### `DepthFactor` — an RGB-D-style depth reading

**Blocks** $[T_{WS},\ \mathbf{h}_W,\ T_{SC}]$ — residual 1, Jacobian $1\times16$.

$$
\mathbf{h}_C = T_{CS}\,T_{SW}\,\mathbf{h}_W,
\qquad
z = \frac{\mathbf{h}_C[2]}{\mathbf{h}_C[3]},
\qquad
r = W\,(\tilde{z} - z)
$$

With `ONESIDED = true` (`OneSidedDepthFactor`) both residual and Jacobian are
zeroed when $z > \tilde{z}$, penalizing only points that come *too close* —
a free-space constraint rather than a depth measurement.

### `HomogeneousPointFactor` — a unary prior on the dehomogenized position

**Blocks** $[\mathbf{h}]$ — residual 3, Jacobian $3\times4$.

$$
\mathbf{p}_\text{est} = \frac{\mathbf{h}_{0:3}}{w},
\qquad
\mathbf{r} = W\left(\tilde{\mathbf{p}} - \mathbf{p}_\text{est}\right),
\qquad
\frac{\partial\mathbf{r}}{\partial\mathbf{h}}
= -\frac{W}{w}\big[\;I_3 \;\big|\; -\mathbf{p}_\text{est}\;\big]
$$

Both whiten internally.
