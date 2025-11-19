# Practical Examples

Complete working examples for common use cases.

## Loading and Optimizing G2O Pose Graph

```rust
use apex_solver::io::G2oLoader;
use apex_solver::core::problem::Problem;
use apex_solver::factors::BetweenFactorSE3;
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::dvector;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load graph from file
    let graph = G2oLoader::load("data/sphere2500.g2o")?;

    // Build problem
    let mut problem = Problem::new();
    let mut initial = HashMap::new();

    // Add vertices as variables
    for (id, vertex) in &graph.vertices_se3 {
        let name = format!("x{}", id);
        let q = vertex.pose.rotation_quaternion();
        let t = vertex.pose.translation();
        
        initial.insert(
            name,
            (ManifoldType::SE3, dvector![t.x, t.y, t.z, q.w, q.i, q.j, q.k])
        );
    }

    // Add edges as factors
    for edge in &graph.edges_se3 {
        let from = format!("x{}", edge.from);
        let to = format!("x{}", edge.to);
        let factor = Box::new(BetweenFactorSE3::new(edge.measurement.clone()));
        problem.add_residual_block(&[&from, &to], factor, None);
    }

    // Fix first pose (gauge freedom)
    for i in 0..6 {
        problem.fix_variable("x0", i);
    }

    // Optimize
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_verbose(true);

    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&problem, &initial)?;

    println!("Status: {:?}", result.status);
    println!("Cost: {:.6} → {:.6}", result.initial_cost, result.final_cost);
    println!("Iterations: {}", result.iterations);

    Ok(())
}
```

## 2D SLAM with SE2

```rust
use apex_solver::core::problem::Problem;
use apex_solver::factors::{BetweenFactorSE2, PriorFactor};
use apex_solver::manifold::se2::SE2;
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::{dvector, Vector2};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut problem = Problem::new();
    let mut initial = HashMap::new();

    // Robot trajectory: 10 poses
    for i in 0..10 {
        let name = format!("x{}", i);
        let x = i as f64 * 1.0;
        initial.insert(
            name,
            (ManifoldType::SE2, dvector![x, 0.0, 0.0])
        );
    }

    // Odometry constraints
    for i in 0..9 {
        let from = format!("x{}", i);
        let to = format!("x{}", i + 1);
        
        // Measurement: move 1m forward, slight rotation
        let meas = SE2::new(Vector2::new(1.0, 0.0), 0.05);
        let factor = Box::new(BetweenFactorSE2::new(meas));
        problem.add_residual_block(&[&from, &to], factor, None);
    }

    // Loop closure: x9 observes x0
    let loop_meas = SE2::new(Vector2::new(-9.0, 0.0), -0.45);
    let loop_factor = Box::new(BetweenFactorSE2::new(loop_meas));
    problem.add_residual_block(&["x9", "x0"], loop_factor, None);

    // Fix first pose
    for i in 0..3 {
        problem.fix_variable("x0", i);
    }

    // Optimize
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(50)
        .with_verbose(true);

    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&problem, &initial)?;

    println!("Optimization complete: {:?}", result.status);
    
    Ok(())
}
```

## Bundle Adjustment (Poses + Landmarks)

```rust
use apex_solver::core::problem::Problem;
use apex_solver::core::loss_functions::HuberLoss;
use apex_solver::factors::ProjectionFactor;
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::dvector;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut problem = Problem::new();
    let mut initial = HashMap::new();

    // Camera poses (SE3)
    for i in 0..5 {
        let pose_data = dvector![
            i as f64 * 0.5, 0.0, 0.0,  // translation
            1.0, 0.0, 0.0, 0.0         // quaternion (identity)
        ];
        initial.insert(
            format!("cam{}", i),
            (ManifoldType::SE3, pose_data)
        );
    }

    // 3D landmarks (Rn)
    let landmarks = vec![
        dvector![2.0, 1.0, 5.0],
        dvector![2.0, -1.0, 5.0],
        dvector![3.0, 0.0, 6.0],
    ];
    
    for (i, lm) in landmarks.iter().enumerate() {
        initial.insert(
            format!("lm{}", i),
            (ManifoldType::Rn, lm.clone())
        );
    }

    // Camera intrinsics
    let fx = 500.0;
    let fy = 500.0;
    let cx = 320.0;
    let cy = 240.0;

    // Projection observations (simulated)
    // Each camera observes each landmark
    for cam_idx in 0..5 {
        for lm_idx in 0..3 {
            let cam_name = format!("cam{}", cam_idx);
            let lm_name = format!("lm{}", lm_idx);
            
            // Simulated 2D observation (u, v)
            let u = 320.0 + (lm_idx as f64 - 1.0) * 50.0;
            let v = 240.0 + (cam_idx as f64 - 2.0) * 30.0;
            
            let factor = Box::new(ProjectionFactor::new(
                u, v,
                fx, fy, cx, cy,
            ));
            
            // Robust loss for outlier rejection
            let huber = Box::new(HuberLoss::new(1.0)?);
            problem.add_residual_block(
                &[&cam_name, &lm_name],
                factor,
                Some(huber),
            );
        }
    }

    // Fix first camera (gauge freedom)
    for i in 0..6 {
        problem.fix_variable("cam0", i);
    }

    // Optimize
    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_verbose(true);

    let mut solver = LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&problem, &initial)?;

    println!("Bundle adjustment complete: {:?}", result.status);
    println!("Final cost: {}", result.final_cost);

    Ok(())
}
```

## Robust Optimization with Outliers

```rust
use apex_solver::core::problem::Problem;
use apex_solver::core::loss_functions::{HuberLoss, CauchyLoss, TukeyLoss};
use apex_solver::factors::BetweenFactorSE3;

let mut problem = Problem::new();

// Normal measurements with Huber loss
for edge in &normal_edges {
    let factor = Box::new(BetweenFactorSE3::new(edge.measurement.clone()));
    let huber = Box::new(HuberLoss::new(1.0)?);
    problem.add_residual_block(&[&edge.from, &edge.to], factor, Some(huber));
}

// Potentially outlier measurements with Cauchy (heavier tails)
for edge in &loop_closures {
    let factor = Box::new(BetweenFactorSE3::new(edge.measurement.clone()));
    let cauchy = Box::new(CauchyLoss::new(0.5)?);
    problem.add_residual_block(&[&edge.from, &edge.to], factor, Some(cauchy));
}
```

## Incremental Optimization

Add variables and factors incrementally:

```rust
let mut problem = Problem::new();
let mut initial = HashMap::new();

// Initial pose
initial.insert("x0".to_string(), (ManifoldType::SE3, identity_pose.clone()));
problem.fix_variable("x0", 0);

// Add poses incrementally
for i in 1..=100 {
    // Add new variable
    let name = format!("x{}", i);
    initial.insert(name.clone(), (ManifoldType::SE3, /* estimate */));
    
    // Add odometry factor
    let prev_name = format!("x{}", i - 1);
    let factor = Box::new(BetweenFactorSE3::new(odom_measurement));
    problem.add_residual_block(&[&prev_name, &name], factor, None);
    
    // Optimize periodically
    if i % 10 == 0 {
        let result = solver.optimize(&problem, &initial)?;
        // Update initial with optimized values for warm start
        initial = result.parameters.iter()
            .map(|(k, v)| (k.clone(), v.to_init_format()))
            .collect();
    }
}
```

## Covariance Extraction

Extract uncertainty estimates after optimization:

```rust
let config = LevenbergMarquardtConfig::new()
    .with_max_iterations(100)
    .with_compute_covariances(true);

let mut solver = LevenbergMarquardt::with_config(config);
let result = solver.optimize(&problem, &initial)?;

if let Some(covs) = &result.covariances {
    for (name, cov) in covs {
        // For SE3: 6×6 covariance matrix
        // Indices 0-2: translation, 3-5: rotation
        
        let pos_cov = cov.submatrix(0, 0, 3, 3);
        let rot_cov = cov.submatrix(3, 3, 3, 3);
        
        // Position uncertainty (3-sigma)
        let sigma_x = (cov[(0, 0)]).sqrt() * 3.0;
        let sigma_y = (cov[(1, 1)]).sqrt() * 3.0;
        let sigma_z = (cov[(2, 2)]).sqrt() * 3.0;
        
        println!("{}: position 3σ = [{:.4}, {:.4}, {:.4}]", 
            name, sigma_x, sigma_y, sigma_z);
    }
}
```
