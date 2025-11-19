# Problem Setup

This chapter covers creating optimization problems, adding factors, and initializing variables.

## Creating a Problem

```rust
use apex_solver::core::problem::Problem;

let mut problem = Problem::new();
```

## Adding Factors

### Unary Factor (Prior)

Prior factors constrain a single variable toward a desired value:

```rust
use apex_solver::factors::PriorFactor;
use nalgebra::dvector;

let prior = Box::new(PriorFactor {
    data: dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], // SE3 identity
});
problem.add_residual_block(&["x0"], prior, None);
```

### Binary Factor (Between)

Between factors constrain the relative transformation between two variables:

```rust
use apex_solver::factors::BetweenFactorSE3;
use apex_solver::manifold::se3::SE3;
use nalgebra::Vector3;

let measurement = SE3::from_translation_and_rotation(
    Vector3::new(1.0, 0.0, 0.0),
    SO3::identity(),
);
let factor = Box::new(BetweenFactorSE3::new(measurement));
problem.add_residual_block(&["x0", "x1"], factor, None);
```

## Robust Loss Functions

Add robustness to outliers:

```rust
use apex_solver::core::loss_functions::{HuberLoss, CauchyLoss};

// Huber loss (threshold = 1.0)
let huber = Box::new(HuberLoss::new(1.0)?);
problem.add_residual_block(&["x0", "x1"], factor, Some(huber));

// Cauchy loss (threshold = 0.5)
let cauchy = Box::new(CauchyLoss::new(0.5)?);
problem.add_residual_block(&["x1", "x2"], factor2, Some(cauchy));
```

## Initializing Variables

Each variable requires a name, manifold type, and initial value:

```rust
use apex_solver::manifold::ManifoldType;
use nalgebra::dvector;
use std::collections::HashMap;

let mut initial = HashMap::new();

// SE3 pose: [tx, ty, tz, qw, qx, qy, qz]
initial.insert(
    "x0".to_string(),
    (ManifoldType::SE3, dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
);

// SE2 pose: [x, y, theta]
initial.insert(
    "pose2d".to_string(),
    (ManifoldType::SE2, dvector![1.0, 2.0, 0.5])
);

// SO3 rotation: [qw, qx, qy, qz]
initial.insert(
    "rotation".to_string(),
    (ManifoldType::SO3, dvector![1.0, 0.0, 0.0, 0.0])
);

// Rn point: [x, y, z]
initial.insert(
    "landmark".to_string(),
    (ManifoldType::Rn, dvector![5.0, 3.0, 2.0])
);
```

## Fixed Variables vs Prior Factors

Two ways to constrain variables:

### Fixed Variables (Hard Constraint)

Fixed variables receive **exactly zero update** during optimization. Use for gauge freedom.

```rust
// Fix entire SE3 pose
for i in 0..6 {
    problem.fix_variable("x0", i);
}

// Fix only position, rotation free
problem.fix_variable("x0", 0); // x
problem.fix_variable("x0", 1); // y
problem.fix_variable("x0", 2); // z
```

**Tangent space indices for SE3:**
- 0, 1, 2: Translation (vx, vy, vz)
- 3, 4, 5: Rotation (ωx, ωy, ωz)

### Prior Factors (Soft Constraint)

Prior factors pull variables toward a value but **allow movement** if other constraints demand it:

```rust
let prior = Box::new(PriorFactor {
    data: dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
});
problem.add_residual_block(&["x0"], prior, None);
```

### When to Use Which

| Scenario | Approach |
|----------|----------|
| Gauge freedom (first pose in SLAM) | Fixed variable |
| GPS with uncertainty | Prior factor |
| Known exact calibration | Fixed variable |
| Regularization | Prior factor |

## Variable Bounds

Constrain variables to stay within bounds:

```rust
// Set bounds for variable "x0", index 0 (first DOF)
problem.set_variable_bounds("x0", 0, -1.0, 1.0);

// Example: constrain translation to [-10, 10]
for i in 0..3 {
    problem.set_variable_bounds("pose", i, -10.0, 10.0);
}
```
