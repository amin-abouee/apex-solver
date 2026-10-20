# Robust Loss Functions

Least squares assumes every residual is a draw from a Gaussian. Real data
contains wrong correspondences, and a squared cost lets them dominate. A
**robust loss** (an M-estimator) replaces $\tfrac12 s$ with a slower-growing
$\rho(s)$, and the crate ships a catalogue of them behind one trait. This
chapter is the family, the parameters each kernel takes, and the choice.

## 1. The problem

With $s = \|\mathbf{r}\|^2$, the standard cost is $\tfrac12 s$: a residual ten
times too large contributes a hundred times as much. A robust cost grows
sub-quadratically, so an outlier's influence is bounded.

Every kernel in the crate is specified by three functions:

<div class="apx-symbols apx-table">
<span class="apx-title">The `Loss` trait</span>

| Function | Meaning |
|---|---|
| $\rho(s)$ | the cost |
| $\rho'(s)$ | the weight function |
| $\rho''(s)$ | the second derivative the corrector needs |

</div>

The input is the **squared** norm, not the norm. That convention is what lets a
kernel with a threshold $\delta$ (Huber) or a scale $c$ (Cauchy) act on a
multi-dimensional residual as a whole.

## 2. The family

The crate's `core::loss_functions` module ships, among others:

| Kernel | Cost | Character |
|---|---|---|
| `L2Loss` | $\tfrac12 s$ | no robustness — the baseline |
| `L1Loss` | $\sqrt{s}$ | linear influence, non-smooth at the origin |
| `HuberLoss(δ)` | quadratic inside, linear outside | the default choice |
| `CauchyLoss(c)` | $\tfrac{c^2}{2}\log(1+s/c^2)$ | redescending |
| `TukeyBiweightLoss(c)` | biweight | redescending, hard cut beyond $c$ |
| `WelschLoss`, `FairLoss`, `GemanMcClureLoss`, `DcsLoss` | — | further shapes |
| `AndrewsWaveLoss`, `RamsayEaLoss`, `TrimmedMeanLoss`, `LpNormLoss` | — | bounded, trimmed and norm-family shapes |
| `TDistributionLoss` | the Student's t | heavy-tailed |
| `BarronGeneralLoss`, `AdaptiveBarronLoss` | the one-parameter family | shape and scale in one number |

Two axes matter when choosing:

- **where the influence stops growing** (Huber: never; Cauchy/Tukey: it
  falls back to zero), and
- **how the scale is set** (a fixed $\delta$ in whitened units, or a data-driven
  one).

## 3. Huber in particular

Huber's $\delta = 1.345$ (in whitened units) is the 95 %-efficient choice: at
that threshold the loss has 95 % of the efficiency of least squares on
Gaussian data, while the influence of a gross outlier is capped at $\delta$.

The crate's own documentation recommends `HuberLoss::new(1.345)` and names it
as the standard choice, and the consumer crates use the same number. A threshold in
*raw* residual units is meaningless — 1.345 what? — which is why the loss is
applied **after** whitening: the corrector sees the whitened residual, so one
threshold means "1.345 standard deviations" everywhere in the graph.

## 4. Robustness is not rejection

A robust loss down-weights; it does not remove. The distinction shows up in
what the solve does with a badly wrong measurement:

- a **kernel** keeps it, with a bounded influence;
- a **gate** (a hard threshold, or a RANSAC inlier set) removes it.

The crate offers both, at different levels: the loss lives on the residual
block (per measurement, per solve), while selection lives in the consumer (the
RANSAC driver in `apex-cv`, the inlier masks in `apex-vio`). A graph can use
either, both, or neither.

## 5. Choosing a kernel

<div class="apx-table">
<span class="apx-title">A rule of thumb</span>

| Data | Kernel |
|---|---|
| mostly Gaussian, a few gross errors | Huber |
| many gross errors, or a mixture | Cauchy or Tukey (redescending) |
| bounded residual range is known a priori | Trimmed |
| the shape should adapt to the data | a Barron variant |

</div>

The redescending property is the substantive difference. Huber's influence grows
linearly forever, so a *coherent* group of wrong measurements still pulls the
estimate; Cauchy's influence falls back to zero, so a group of outliers stops
pulling once it is far enough out. The cost of redescending is fragility: a
genuine measurement that is slightly unusual can be discarded, and the solution
can depend on the initial guess.

## 6. Worked example

A whitened residual of 10 (ten sigma) under each kernel, with the loss's
influence relative to least squares:

| Kernel | influence / least squares |
|---|---|
| least squares | 10 |
| Huber ($\delta=1.345$) | 0.135 |
| Cauchy ($c=2.385$) | 0.0016 |
| Tukey ($c=4.685$) | 0 |

The numbers are the decision: a Huber graph still feels a ten-sigma outlier
(mildly), a Cauchy graph barely does, and a Tukey graph ignores it entirely.

## 7. The catch

<div class="apx-warning">
<span class="apx-title">Warning — a redescending kernel can discard the signal</span>

Cauchy and Tukey reduce the influence of a residual that is far from the
current estimate. During the first iterations of a solve, the estimate *is*
far from the truth, and a redescending kernel can decide the true solution is
the outlier. The usual defences are a good initialization, a Huber kernel for
the early iterations, and — in this crate's design — the RANSAC stage in the
consumer, which produces the initialization.

</div>

<div class="apx-note">
<span class="apx-title">Note — a loss does not repair a wrong Jacobian</span>

The kernel re-weights the residual; it cannot fix a factor whose Jacobian
disagrees with its residual. A sign error in a factor produces a residual the
kernel happily down-weights, and the solve converges to a wrong answer with a
comfortable cost. The crate's factor tests check Jacobians against finite
differences precisely because no amount of robustness can substitute for that
check.

</div>
