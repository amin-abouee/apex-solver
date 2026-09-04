# Installation & Quick Start

## Installation

```toml
[dependencies]
apex-solver = "1.4.0"
```

Optional feature for live Rerun visualization:

```toml
[dependencies]
apex-solver = { version = "1.4.0", features = ["visualization"] }
```

## Solving a pose graph from a G2O file

```rust
use apex_solver::core::problem::Problem;
use apex_solver::factors::pose::BetweenFactor;
use apex_solver::{G2oLoader, JacobianMode, ManifoldType};
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::dvector;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load pose graph from G2O file
    let graph = G2oLoader::load("data/odometry/3d/sphere2500.g2o")?;

    // Create optimization problem
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut var_keys = HashMap::new();

    // Add SE3 poses as variables -- returns stable VarKey handles
    for (&id, vertex) in &graph.vertices_se3 {
        let quat = vertex.pose.rotation_quaternion();
        let trans = vertex.pose.translation();
        let se3_data = dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k];
        let key = problem.add_variable(ManifoldType::SE3, se3_data);
        var_keys.insert(id, key);
    }

    // Add between factors (relative pose constraints) using VarKey handles
    for edge in &graph.edges_se3 {
        let k_from = var_keys[&edge.from];
        let k_to = var_keys[&edge.to];
        problem.add_residual_block(
            &[k_from, k_to],
            Box::new(BetweenFactor::new(edge.measurement.clone())),
            None,  // Optional: add HuberLoss for robustness
        );
    }

    // Configure and run optimizer
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-6)
        .with_compute_covariances(true);  // Enable uncertainty estimation

    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&mut problem)?;

    println!("Status: {:?}", result.status);
    println!("Initial cost: {:.3e}", result.initial_cost);
    println!("Final cost: {:.3e}", result.final_cost);
    println!("Iterations: {}", result.iterations);

    Ok(())
}
```

**Result**:
```text
Status: CostToleranceReached
Initial cost: 1.280e+05
Final cost: 2.130e+01
Iterations: 5
```

## Solving a BAL bundle adjustment problem

The `bundle_adjustment` binary solves BAL datasets with the iterative Schur
solver and self-calibration:

```bash
cargo run --release --bin bundle_adjustment -- \
    data/bundle_adjustment/trafalgar/problem-257-65132-pre.txt
```

Programmatically the same shape is: SE(3) pose variable per camera, `Rn(3)`
variable per landmark, optional `Rn(3)` intrinsics per camera
(`mark_as_schur_landmark` on the point keys), `ProjectionFactor` with a
`HuberLoss`. See [Problem Construction](./problem.md) next.
