# Examples

## Example 1: Basic Pose Graph Optimization

```rust
use std::collections::HashMap;
use apex_solver::core::problem::Problem;
use apex_solver::factors::BetweenFactor;
use apex_solver::{G2oLoader, ManifoldType};
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::dvector;

// Load pose graph
let graph = G2oLoader::load("data/odometry/sphere2500.g2o")?;

// Build optimization problem
let mut problem = Problem::new();
let mut initial_values = HashMap::new();

// Add SE3 poses as variables
for (&id, vertex) in &graph.vertices_se3 {
    let quat = vertex.pose.rotation_quaternion();
    let trans = vertex.pose.translation();
    initial_values.insert(
        format!("x{}", id),
        (ManifoldType::SE3, dvector![trans.x, trans.y, trans.z, quat.w, quat.i, quat.j, quat.k])
    );
}

// Add between factors
for edge in &graph.edges_se3 {
    problem.add_residual_block(
        &[&format!("x{}", edge.from), &format!("x{}", edge.to)],
        Box::new(BetweenFactor::new(edge.measurement.clone())),
        None,
    );
}

// Configure and solve
let config = LevenbergMarquardtConfig::new()
    .with_max_iterations(100)
    .with_cost_tolerance(1e-6);

let mut solver = LevenbergMarquardt::with_config(config);
let result = solver.optimize(&problem, &initial_values)?;

println!("Optimized {} poses in {} iterations",
    result.parameters.len(), result.iterations);
```

---

## Example 2: Custom Factor Implementation

Create custom factors by implementing the `Factor` trait:

```rust
use apex_solver::factors::Factor;
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct MyRangeFactor {
    pub measurement: f64,
    pub information: f64,
}

impl Factor for MyRangeFactor {
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>) {
        // Extract 2D point parameters [x, y]
        let x = params[0][0];
        let y = params[0][1];

        // Compute predicted measurement
        let predicted_distance = (x * x + y * y).sqrt();

        // Compute residual: measurement - prediction
        let residual = DVector::from_vec(vec![
            self.information.sqrt() * (self.measurement - predicted_distance)
        ]);

        // Compute analytic Jacobian
        let jacobian = if compute_jacobian {
            if predicted_distance > 1e-8 {
                let scale = -self.information.sqrt() / predicted_distance;
                Some(DMatrix::from_row_slice(1, 2, &[scale * x, scale * y]))
            } else {
                Some(DMatrix::zeros(1, 2))
            }
        } else {
            None
        };

        (residual, jacobian)
    }

    fn get_dimension(&self) -> usize {
        1  // One scalar residual
    }
}

// Use in optimization
problem.add_residual_block(
    &["point"],
    Box::new(MyRangeFactor { measurement: 5.0, information: 1.0 }),
    None
);
```

---

## Example 3: Self-Calibration Bundle Adjustment

Optimize camera poses, 3D landmarks, AND camera intrinsics simultaneously. See the
[apex-camera-models](../crates/apex-camera-models/README.md) crate for detailed camera
model documentation.

```rust
use std::collections::HashMap;
use apex_solver::core::problem::Problem;
use apex_solver::factors::ProjectionFactor;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
// Use any camera model from apex-camera-models crate

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut problem = Problem::new();
    let mut initial_values = HashMap::new();

    // Add camera poses (SE3), 3D landmarks, and per-camera intrinsics
    // See apex-camera-models documentation for camera model options

    // Add projection factors with compile-time optimization config
    // ProjectionFactor<CameraModel, OptConfig> links poses + landmarks + intrinsics
    for observation in &observations {
        problem.add_residual_block(
            &[&format!("pose_{}", obs.camera_id),
              &format!("landmark_{}", obs.point_id),
              &format!("intrinsics_{}", obs.camera_id)],
            Box::new(projection_factor),
            Some(Box::new(HuberLoss::new(1.0)?)),
        );
    }

    // Fix first camera for gauge freedom
    for dof in 0..6 {
        problem.fix_variable("pose_0000", dof);
    }

    // Configure solver with Schur complement (best for BA)
    let config = LevenbergMarquardtConfig::for_bundle_adjustment();
    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&problem, &initial_values)?;

    Ok(())
}
```

**Optimization Types** (compile-time configuration):
- `SelfCalibration`: Optimize pose + landmarks + intrinsics
- `BundleAdjustment`: Optimize pose + landmarks (fixed intrinsics)
- `OnlyPose`: Visual odometry (fixed landmarks and intrinsics)
- `OnlyLandmarks`: Triangulation (known poses)
- `OnlyIntrinsics`: Camera calibration (known structure)

See [apex-camera-models documentation](../crates/apex-camera-models/README.md) for complete
camera model reference and advanced examples.

---

*Back to [README](../README.md)*
