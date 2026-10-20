# Symbol Glossary

Every symbol this book uses, in one place. The conventions chapter states the
rules; this page is the lookup table.

## 1. Algebra and geometry

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | the full state vector of the problem |
| $\mathbf{r}$ | a residual — predicted minus measured |
| $J$ | a Jacobian, $\partial\mathbf{r}/\partial\mathbf{x}$ |
| $H$ | a normal matrix, $J^\top J$ |
| $\mathbf{g}$ | a gradient, $J^\top\mathbf{r}$ — **positive** in this crate |
| $\Lambda$ | an information matrix |
| $S$ | a square-root information, $S^\top S = \Lambda$; also the Schur complement's reduced Hessian |
| $C$ | a cost, $\tfrac12\mathbf{r}^\top\Omega\mathbf{r}$ |
| $\Omega$ | an information matrix in the *noise model's* notation |
| $\Delta$ | a trust-region radius; also a rigid offset |
| $\lambda$ | the Levenberg–Marquardt damping parameter |
| $D$ | Marquardt's column scaling, $\operatorname{diag}(J^\top J)$ |
| $\delta$ | a step in the tangent space |
| $\xi$ | a tangent-space perturbation |
| $\rho$ | a robust loss function |
| $\alpha^2$ | the corrector's Jacobian correction factor, $\rho''/\rho'$ |

## 2. Manifolds

| Symbol | Meaning |
|---|---|
| $X$ | a variable's value on its manifold |
| $\oplus$, $\ominus$ | manifold plus and minus |
| $\odot$ | a group acting on a point, $X \odot \mathbf{p}$ |
| $\circ$ | a group composition, $X \circ Y$ |
| $\mathrm{Exp}$, $\log$ | the exponential and logarithm maps |
| $\mathrm{Ad}$ | the adjoint, $\mathrm{Ad}(X)$ |
| $\boxplus$, $\boxminus$ | the `SE23`-family plus/minus, used by the IMU factors |
| $\xi(\delta b)$ | the IMU factor's first-order bias correction |
| $SE(3)$ | a rigid transform, 7 parameters, 6 DoF |
| $SE_2(3)$ | pose and velocity, 9 parameters |
| $SGal(3)$ | pose, velocity and time, 10 parameters |
| $Sim(3)$ | a similarity, 8 parameters, 7 DoF |

## 3. The problem

| Symbol | Meaning | Code |
|---|---|---|
| $\mathcal{V}$ | the variables | `VarKey` |
| $\mathcal{F}$ | the residual blocks | `FactorKey` |
| $M$ | a factor's measurement vector | — |
| $h(x)$ | the prediction function | — |
| $\mathcal{F}_{ij}$ | the factor connecting variables $i$ and $j$ | — |
| $\theta$ | a variable's local coordinates | — |
| $x_0$ | a linearization point | `linearization_point` |
| $\Lambda_p$ | a marginal's information | `Marginal::information` |
| $g_p$ | a marginal's gradient | `Marginal::gradient` |
| $b$ | a prior's offset, $-\Lambda_p^+ g_p$ | `Marginal::offset` |
| $\Lambda_{kk}$ | a block of the information matrix | — |
| $\Lambda_{km}$ | a cross block | — |

## 4. Optimization

| Symbol | Meaning |
|---|---|
| $\rho_{\text{gain}}$ | the step-quality ratio (actual over predicted decrease) |
| $L(h)$ | the local quadratic model of the cost |
| $p_c$ | the Cauchy point |
| $h_{gn}$ | the Gauss–Newton point |
| $\Delta$ | the trust-region radius |
| $\nu$ | Nielsen's damping escalation factor |
| $\eta$ | the PCG forcing sequence parameter |
| $\mathrm{nnz}$ | the number of stored non-zeros |

## 5. Sensors

| Symbol | Meaning |
|---|---|
| $K$ | a camera's calibration matrix |
| $\mathbf{b}$ | a unit bearing |
| $\rho$ | an inverse depth or an inverse range (context) |
| $\sigma$ | a standard deviation |
| $\delta$ | a Huber threshold (in whitened units) |
| $\mathbf{n}$ | a unit normal |
| $d$ | a plane's offset, $\mathbf{n}^\top\mathbf{x} + d = 0$ |
| $\mathbf{q}$ | a plane's centroid |
| $b_g$, $b_a$ | the gyroscope and accelerometer biases |
| $\Delta R$, $\Delta\mathbf{v}$, $\Delta\mathbf{p}$ | the preintegrated increments |
| $\mathbf{C}_1$, $\mathbf{C}_2$ | the preintegration's first and second moment integrals |
| $r$ | a range measurement; also a residual (context) |

## 6. Code names

<div class="apx-table">
<span class="apx-title">The names that recur</span>

| Name | What it is |
|---|---|
| `Problem` | the graph |
| `ManifoldVariable` | one variable |
| `Factor` | one measurement's residual and Jacobian |
| `ResidualBlock` | a factor plus its loss and noise model |
| `NoiseModel` | a square-root information |
| `LossFunction` | a robust kernel |
| `Corrector` | the re-weighting that makes $\rho$ a least-squares problem |
| `Linearizer` | the per-iteration evaluation and assembly |
| `JacobianEvaluation` | the evaluation policy: current or first estimate |
| `Optimizer` | the trait the three optimizers implement |
| `LevenbergMarquardt`, `GaussNewton`, `DogLeg` | the three optimizers |
| `Damping` | the damping scalar and its clamps |
| `SchurPartition`, `ChunkedSchurEliminator` | the Schur bookkeeping and the elimination |
| `Marginalizer`, `MarginalPriorFactor` | the marginalization and its factor |
| `OptObserver`, `RerunObserver` | the monitoring hooks |

</div>
