# apex-manifolds

Lie group manifolds (SE2, SE3, SO2, SO3, Rn) with analytic Jacobians for nonlinear optimization.

## Overview

This library provides efficient implementations of Lie group manifolds commonly used in robotics and computer vision applications. All implementations include analytic Jacobians for optimization algorithms.

## Features

- **SE(2)**: Special Euclidean group for 2D rigid transformations
- **SE(3)**: Special Euclidean group for 3D rigid transformations
- **SO(2)**: Special Orthogonal group for 2D rotations
- **SO(3)**: Special Orthogonal group for 3D rotations
- **Rⁿ**: Euclidean vector spaces
- Analytic Jacobians for all operations
- Right and left perturbation models
- Exponential and logarithmic maps
- Manifold plus/minus operations

## Installation

```toml
[dependencies]
apex-manifolds = "1.0.0"
```

## Usage

```rust
use apex_manifolds::{SE3, LieGroup};
use nalgebra::{Vector3, Vector6};

// Create an SE3 pose
let pose = SE3::from_translation(Vector3::new(1.0, 2.0, 3.0));

// Apply a tangent space perturbation
let delta = Vector6::new(0.1, 0.0, 0.0, 0.0, 0.0, 0.0);
let updated_pose = pose.plus(&delta, None, None);

// Compute relative transformation
let other_pose = SE3::identity();
let relative = pose.minus(&other_pose, None, None);
```

## Design

The design is inspired by the [manif C++ library](https://github.com/artivis/manif) and provides:

- Consistent API across all manifold types
- Type-safe operations
- Zero-cost abstractions
- Comprehensive test coverage

## License

Apache-2.0
