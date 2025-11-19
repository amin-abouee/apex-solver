# Rn - Euclidean Vectors

The trivial manifold for standard vector spaces.

## Mathematical Definition

$$
\mathbb{R}^n = \{x \in \mathbb{R}^n\} \text{ with vector addition}
$$

**Group Properties:**
- **Identity**: 0 (zero vector)
- **Composition**: x₁ + x₂
- **Inverse**: -x
- **Abelian**: x₁ + x₂ = x₂ + x₁

## Tangent Space

- **Dimension**: n DOF (same as manifold)
- **Representation**: Same as group element
- The tangent space **is** the manifold

## Internal Representation

```rust
// Group element: dynamic-size vector
pub struct Rn {
    value: DVector<f64>,
}

// Tangent element: same as group
pub struct RnTangent {
    value: DVector<f64>,
}
```

**Data format**: `DVector<f64>` of any size

## Exponential and Logarithmic Maps

Both are the identity:

$$
\exp(v) = v
$$

$$
\log(v) = v
$$

## Jacobians

All Jacobians are identity matrices:

$$
J_r = J_l = J_r^{-1} = J_l^{-1} = I_n
$$

## Common Operations

```rust
use apex_solver::manifold::rn::Rn;
use nalgebra::{dvector, DVector};

// Create vectors
let v1 = Rn::from_vec(vec![1.0, 2.0, 3.0]);
let v2 = Rn::from_dvector(dvector![4.0, 5.0, 6.0]);
let v3 = Rn::zeros(5); // 5D zero vector

// Dimension
let dim = v1.dim(); // 3

// Composition (addition)
let sum = v1.compose(&v2, None, None);

// Interpolation
let interp = v1.interp(&v2, 0.5); // Midpoint

// Plus/minus are standard vector ops
let delta = RnTangent::from_dvector(dvector![0.1, 0.1, 0.1]);
let updated = v1.right_plus(&delta, None, None);
```

## Usage in Optimization

Rn is used for:
- **Landmarks**: 3D points in bundle adjustment
- **Calibration parameters**: Camera intrinsics, IMU biases
- **Any Euclidean quantity**: Velocities, forces, etc.

```rust
use apex_solver::manifold::ManifoldType;
use nalgebra::dvector;
use std::collections::HashMap;

let mut initial = HashMap::new();

// 3D landmark
initial.insert(
    "landmark0".to_string(),
    (ManifoldType::Rn, dvector![5.0, 3.0, 2.0])
);

// Camera intrinsics (fx, fy, cx, cy)
initial.insert(
    "intrinsics".to_string(),
    (ManifoldType::Rn, dvector![500.0, 500.0, 320.0, 240.0])
);
```

## Mixed Manifold Problems

Rn works seamlessly with other manifolds in the same problem:

```rust
let mut initial = HashMap::new();

// SE3 camera pose
initial.insert(
    "cam0".to_string(),
    (ManifoldType::SE3, dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
);

// Rn landmark
initial.insert(
    "landmark0".to_string(),
    (ManifoldType::Rn, dvector![5.0, 3.0, 2.0])
);

// Both handled transparently by the optimizer
let result = solver.optimize(&problem, &initial)?;
```
