# Visualization with Rerun

The `visualization` feature streams poses, landmarks, cost curves, damping
trajectories and Hessian sparsity patterns to a live
[Rerun](https://rerun.io) viewer while the optimizer runs.

```toml
[dependencies]
apex-solver = { version = "1.4.0", features = ["visualization"] }
```

```rust
use apex_solver::observers::{RerunObserver, VisualizationConfig};

let config = VisualizationConfig::new()
    .with_show_cameras(true)
    .with_show_landmarks(true)
    .with_show_plots(true);

let mut solver = LevenbergMarquardt::with_config(lm_config);
solver.add_observer(RerunObserver::with_config(true, config)?);
let result = solver.optimize(&mut problem)?;
```

Start the viewer first (`rerun` on the command line or `rerun::Session`), then
run your binary. Every iteration appears live: camera frustums, landmark points,
the cost curve, and the sparsity pattern of the current Hessian.

The `visualize_optimization` and `visualize_graph_file` examples show complete
setups for bundle adjustment and pose graphs respectively:

```bash
cargo run --release --example visualize_optimization --features visualization
```
