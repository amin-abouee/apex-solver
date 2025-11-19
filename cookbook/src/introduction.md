# Introduction

Apex Solver provides manifold-aware optimization for robotics and computer vision applications. Unlike standard optimizers that treat all variables as Euclidean vectors, Apex Solver understands the geometry of Lie groups (rotations, rigid transforms) and performs optimization directly on these manifolds.

## Why Manifold Optimization?

Consider optimizing a 3D rotation. A naive approach might represent it as a 3×3 matrix (9 parameters) with orthogonality constraints. This is:
- **Overconstrained**: 9 parameters for 3 degrees of freedom
- **Numerically unstable**: Must project back to SO(3) after each step
- **Inefficient**: Wasted computation on constraint satisfaction

Manifold optimization instead:
- Works in the **tangent space** (3 parameters for SO(3))
- Uses **exponential map** to stay on the manifold
- Computes **analytic Jacobians** for the geometry

## Key Concepts

| Term | Definition |
|------|------------|
| **Manifold** | A smooth surface where each point has a local Euclidean structure |
| **Lie Group** | A manifold that is also a group (has composition, inverse, identity) |
| **Tangent Space** | The local linear approximation at a point on the manifold |
| **Exponential Map** | Maps from tangent space to manifold: `exp: tangent → group` |
| **Logarithmic Map** | Maps from manifold to tangent space: `log: group → tangent` |
| **Plus/Minus** | Manifold operations: `g ⊞ δ = g ∘ exp(δ)`, `g₁ ⊟ g₂ = log(g₁⁻¹ ∘ g₂)` |

## Quick Start

```rust
use apex_solver::core::problem::Problem;
use apex_solver::factors::BetweenFactorSE3;
use apex_solver::manifold::ManifoldType;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use nalgebra::dvector;
use std::collections::HashMap;

// 1. Create problem
let mut problem = Problem::new();

// 2. Add factors (constraints)
let factor = Box::new(BetweenFactorSE3::new(measurement));
problem.add_residual_block(&["x0", "x1"], factor, None);

// 3. Initialize variables
let mut initial = HashMap::new();
initial.insert(
    "x0".to_string(),
    (ManifoldType::SE3, dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
);

// 4. Optimize
let mut solver = LevenbergMarquardt::new();
let result = solver.optimize(&problem, &initial)?;

println!("Final cost: {}", result.final_cost);
```

## Dependencies

- **faer**: Core sparse linear algebra (v0.22)
- **nalgebra**: Dense linear algebra and quaternions (v0.33)
- **rayon**: Parallel computation (v1.8)
