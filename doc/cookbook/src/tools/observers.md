# Observers: Rerun

An optimizer that fails tells you only that it failed. An observer lets you
*watch* it: the cost curve, the gradient, the damping, the state moving. The
crate's observer pattern separates the monitoring from the optimizing, so the
same observer works with Gauss–Newton, Levenberg–Marquardt and Dog Leg. This
chapter is the pattern, the Rerun observer, and what to look at when the
solve misbehaves.

## 1. The pattern

<div class="apx apx-note">
<span class="apx-title">Note — the design philosophy, from the module</span>

> The observer pattern provides complete separation between optimization
> algorithms and monitoring/visualization logic:
>
> - **Decoupling**: optimization logic is independent of how progress is
>   monitored
> - **Extensibility**: add observers (Rerun, CSV, metrics, dashboards) without
>   touching the optimizer
> - **Composability**: several observers run simultaneously
> - **Zero overhead**: with no observers registered, notification is a no-op

</div>

```text
┌─────────────────┐
│   Optimizer     │
│  (LM/GN/DogLeg) │
└────────┬────────┘
         │ observers.notify(values, iteration)
         ├──────────────┬──────────────┬──────────────┬──────────────┐
         ▼              ▼              ▼              ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │  Rerun   │   │   CSV    │   │ Metrics  │   │  Custom  │
   │ Observer │   │ Observer │   │ Observer │   │ Observer │
   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

## 2. The trait

```rust
pub trait OptObserver: Send {
    fn on_step(&self, values: &SlotMap<VarKey, Box<dyn ManifoldVariable>>, iteration: usize);
    fn set_iteration_metrics(&self, metrics: &IterationMetrics) { /* optional */ }
    fn on_complete(&self, ...) { /* optional */ }
}
```

`on_step` is called after each iteration, with the current variable values
(iteration 0 is the *initial* values, before any step) and the iteration
number. The trait's own guidance to implementors is worth repeating: keep the
method fast, handle errors internally (log, don't panic), do not mutate the
values, and buffer expensive operations.

The optional hooks carry the per-iteration metrics — cost, gradient norm,
damping parameter, step quality — which is what a plot needs.

`OptObserverVec` is the multi-observer container: `add`, `notify`,
`notify_complete`, `len`, `is_empty`.

## 3. Attaching an observer

```rust
let mut solver = LevenbergMarquardt::with_config(config);
solver.add_observer(RerunObserver::new(true)?);
let result = solver.optimize(&mut problem)?;
```

`add_observer` exists on the optimizers that have a meaningful per-iteration
story. The crate deprecated several older "visualization" fields with the
note: *"never read; attach a `RerunObserver` with `add_observer` instead"* —
the observer pattern replaced the ad-hoc hooks.

## 4. The Rerun observer

`RerunObserver` lives behind the **`visualization` feature**:

```toml
apex-solver = { version = "1.5", features = ["visualization"] }
```

What it shows:

<div class="apx-table">
<span class="apx-title">What the Rerun observer renders</span>

| Channel | Contents |
|---|---|
| time-series plots | cost, gradient norm, damping parameter, step quality |
| Hessian | a heat map of the sparse normal matrix — structure *and* values |
| gradient | vectors with magnitude encoded |
| state | live poses for `SE2` and `SE3` problems |
| initial graph | the starting configuration, for comparison |

</div>

`VisualizationConfig` controls what appears: `with_show_cameras`,
`with_show_landmarks`, `with_show_se2_poses`, `with_show_plots`,
`with_show_matrices`, plus the camera's field of view, aspect ratio and
frustum scale, the landmark point size and colors (initial versus
optimized), the `SE2` pose radius, box half-size and colors, and the Hessian
downsample size.

## 5. The presets

`VisualizationConfig` has three named configurations, and
`RerunObserver` has matching constructors:

| Preset | What it shows |
|---|---|
| `VisualizationConfig::cameras_only()` | the cameras, no landmarks, no `SE2` poses |
| `VisualizationConfig::landmarks_only()` | the 3-D landmarks alone — the structure-from-motion point cloud |
| `VisualizationConfig::for_pose_graph()` | the cameras, no landmarks, poses **not** inverted |
| `VisualizationConfig::for_bundle_adjustment()` | the cameras **inverted** ($T_{wc} \to T_{cw}$, which is what a viewer expects) and the `SE2` poses off — they are not used in bundle adjustment |

`RerunObserver::new_for_bundle_adjustment(enabled, save_path, invert_camera_poses)`
combines the last of these with the observer's own options.

`VisualizationMode` decides *when* the state is published:
`Iterative` (every iteration — detailed but slower, for analysing the
trajectory) and `InitialAndFinal` (the default — the initial and the final
state only, for production runs).

## 6. Reading a solve

<div class="apx-table">
<span class="apx-title">What a plot tells you</span>

| What you see | What it means |
|---|---|
| the cost falls smoothly and stops | a healthy solve |
| the cost falls, then jumps up | a rejected step; the damping is adapting |
| the cost plateaus at a high value | the model is wrong, or the outliers dominate |
| the gradient spikes and does not recover | a factor's Jacobian disagrees with its residual |
| the damping climbs monotonically | the model is bad everywhere in the basin |
| the Hessian heat map has an empty row | an unconstrained direction — fix the gauge |
| the states do not move at all | the fixed coordinates were set wrongly |

</div>

The Hessian view is the most useful of these for a *new* problem: an empty row
or an isolated block in the heat map is a gauge or a missing factor, visible
before the solve rather than after.

## 7. Worked example

Debugging a window solve that converges to the wrong answer. The Rerun
observer shows the cost falling normally and the states moving smoothly — so
the optimizer is doing its job. The Hessian heat map shows a block for every
landmark with the expected two connections, and the plot of the gradient
shows one landmark's gradient an order of magnitude above the others. That
landmark is the suspect: its residual is an outlier, and the robust kernel is
down-weighting it, but the *step* still moves the states because the kernel's
influence, while bounded, is not zero. The fix is upstream — a gate.

## 8. The catch

<div class="apx-warning">
<span class="apx-title">Warning — the observer is not a debugger</span>

The observer shows the *solver's* state: the cost it computed, the Jacobian it
assembled, the step it took. It cannot show that the residual was wrong, only
that it was large. When the plot looks healthy and the answer is wrong, the
problem is in the factor — and the checks that find it are the factor's own
finite-difference test and an independent evaluation of the residual at the
solution.

</div>

<div class="apx-warning">
<span class="apx-title">Warning — the feature pulls a dependency</span>

`visualization` enables `rerun` and `apex-io/visualization`. It is **not** a
default feature: the default set is `["cli", "download"]`. A library that
embeds the solver should not enable `visualization` in production, and a
binary that wants it should enable it explicitly and pay the dependency.

</div>

<div class="apx-note">
<span class="apx-title">Note — `OptObserver` is `Send`, not `Sync`</span>

The trait requires `Send` but not `Sync`, because `on_step` takes `&self` and
an observer may hold interior state it mutates through a `Mutex` or an
atomic. An observer that is called from a parallel context must therefore
synchronize its own state — the crate's `OptObserverVec` does not do it for
you.

</div>
