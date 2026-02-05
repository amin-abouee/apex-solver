# apex-manifolds

Lie group manifolds (SE2, SE3, SO2, SO3, Rn) with analytic Jacobians for nonlinear optimization.

## Overview

This library provides efficient implementations of Lie group manifolds commonly used in robotics and computer vision applications. All implementations include analytic Jacobians for optimization algorithms, following the conventions established by the [manif](https://github.com/artivis/manif) C++ library.

Lie groups are essential mathematical structures for representing:
- **Rigid body transformations**: Camera poses, robot configurations
- **Rotations**: Orientation tracking, attitude estimation
- **Euclidean vectors**: Landmarks, feature points, calibration parameters

The library provides:
- Type-safe manifold operations with compile-time guarantees
- Analytic Jacobians for all operations (no numerical differentiation)
- Right and left perturbation models for optimization
- Exponential and logarithmic maps between groups and algebras
- Composition, inverse, and action operations

## Supported Manifolds

| Manifold | Description | DOF | Rep Size | Use Case |
|----------|-------------|-----|----------|----------|
| **SE(3)** | 3D rigid transformations | 6 | 7 (quaternion + translation) | 3D SLAM, visual odometry, bundle adjustment |
| **SO(3)** | 3D rotations | 3 | 4 (unit quaternion) | Orientation tracking, IMU integration |
| **SE(2)** | 2D rigid transformations | 3 | 4 (angle + translation) | 2D SLAM, mobile robotics |
| **SO(2)** | 2D rotations | 1 | 2 (cos, sin) | 2D orientation, bearing-only SLAM |
| **Rⁿ** | Euclidean vector space | n | n | Landmarks, camera intrinsics, calibration |

### Mathematical Summary

```
Lie group M,° | size   | dim | X ∈ M                   | Constraint      | T_E M             | Exp(T)             | Comp. | Action
------------- | ------ | --- | ----------------------- | --------------- | ----------------- | ------------------ | ----- | ------
n-D vector    | Rⁿ,+   | n   | v ∈ Rⁿ                 | -               | v ∈ Rⁿ            | v = exp(v)         | v₁+v₂ | v + x
SO(2)         | S¹,.   | 1   | R ∈ ℝ²ˣ²              | RᵀR = I         | θ ∈ ℝ             | R = exp([θ]×)      | R₁R₂  | Rx
SE(2)         | -,.    | 3   | M = [R t; 0 1]         | RᵀR = I         | [v̂] ∈ ℝ³          | Exp([v̂])           | M₁M₂  | Rx+t
SO(3)         | S³,.   | 3   | R ∈ ℝ³ˣ³              | RᵀR = I         | [θ]× ∈ so(3)      | R = exp([θ]×)      | R₁R₂  | Rx
SE(3)         | -,.    | 6   | M = [R t; 0 1]         | RᵀR = I         | [v̂] ∈ se(3)       | Exp([v̂])           | M₁M₂  | Rx+t
```

## Installation

```toml
[dependencies]
apex-manifolds = "0.1.0"
```

## Usage

### SE(3) - 3D Rigid Transformations

SE(3) represents 3D rigid body transformations (rotation + translation), commonly used for camera poses and robot configurations.

```rust
use apex_manifolds::se3::SE3;
use apex_manifolds::LieGroup;
use nalgebra::{Vector3, Vector6};

// Create an SE3 pose from translation
let pose = SE3::from_translation(Vector3::new(1.0, 2.0, 3.0));

// Create identity transformation
let identity = SE3::identity();

// Compose two transformations
let composed = pose.compose(&identity, None, None);

// Apply tangent space perturbation (right plus)
let delta = Vector6::new(0.1, 0.0, 0.0, 0.0, 0.0, 0.01);  // [vx, vy, vz, ωx, ωy, ωz]
let perturbed = pose.plus(&delta, None, None);

// Compute relative transformation (right minus)
let relative = perturbed.minus(&pose, None, None);

// Transform a 3D point
let point = Vector3::new(1.0, 0.0, 0.0);
let transformed = pose.act(&point, None, None);

// Get inverse transformation
let inverse = pose.inverse(None);
```

### SO(3) - 3D Rotations

SO(3) represents 3D rotations using unit quaternions internally.

```rust
use apex_manifolds::so3::SO3;
use apex_manifolds::LieGroup;
use nalgebra::{Vector3};
use std::f64::consts::PI;

// Create rotation from axis-angle
let axis = Vector3::new(0.0, 0.0, 1.0);  // Z-axis
let angle = PI / 4.0;  // 45 degrees
let rotation = SO3::from_axis_angle(&axis, angle);

// Create identity rotation
let identity = SO3::identity();

// Compose rotations
let composed = rotation.compose(&rotation, None, None);  // 90 degree rotation

// Rotate a vector
let v = Vector3::new(1.0, 0.0, 0.0);
let rotated = rotation.act(&v, None, None);

// Logarithmic map (rotation to axis-angle vector)
let log_vec = rotation.log(None);
```

### SE(2) - 2D Rigid Transformations

SE(2) represents 2D rigid body transformations for planar robotics.

```rust
use apex_manifolds::se2::SE2;
use apex_manifolds::LieGroup;
use nalgebra::{Vector2, Vector3};
use std::f64::consts::PI;

// Create SE2 from position and angle
let pose = SE2::new(1.0, 2.0, PI / 4.0);  // x=1, y=2, θ=45°

// Create from translation only
let translation = SE2::from_translation(Vector2::new(5.0, 3.0));

// Compose transformations
let composed = pose.compose(&translation, None, None);

// Apply perturbation
let delta = Vector3::new(0.1, 0.0, 0.01);  // [dx, dy, dθ]
let perturbed = pose.plus(&delta, None, None);

// Transform a 2D point (lifted to 3D with z=0)
let point = Vector3::new(1.0, 0.0, 0.0);
let transformed = pose.act(&point, None, None);
```

### SO(2) - 2D Rotations

SO(2) represents 2D rotations (complex numbers on the unit circle).

```rust
use apex_manifolds::so2::SO2;
use apex_manifolds::LieGroup;
use nalgebra::Vector3;
use std::f64::consts::PI;

// Create rotation from angle
let rotation = SO2::from_angle(PI / 2.0);  // 90 degrees

// Compose rotations
let double = rotation.compose(&rotation, None, None);  // 180 degrees

// Get the angle
let angle = rotation.angle();

// Rotate a 2D vector (as 3D with z=0)
let v = Vector3::new(1.0, 0.0, 0.0);
let rotated = rotation.act(&v, None, None);  // [0, 1, 0]
```

### Rⁿ - Euclidean Vector Space

Rⁿ represents standard Euclidean vector spaces, useful for landmarks and calibration parameters.

```rust
use apex_manifolds::rn::Rn;
use apex_manifolds::LieGroup;
use nalgebra::DVector;

// Create a 3D point
let point = Rn::from_vector(DVector::from_vec(vec![1.0, 2.0, 3.0]));

// Create from slice
let landmark = Rn::from_slice(&[4.0, 5.0, 6.0]);

// Plus operation (vector addition)
let delta = DVector::from_vec(vec![0.1, 0.2, 0.3]);
let updated = point.plus(&delta, None, None);

// Minus operation (vector difference)
let diff = updated.minus(&point, None, None);
```

## Computing Jacobians

All manifold operations support optional Jacobian computation for optimization:

```rust
use apex_manifolds::se3::SE3;
use apex_manifolds::LieGroup;
use nalgebra::{Matrix6, Vector6};

let pose = SE3::random();
let delta = Vector6::new(0.1, 0.0, 0.0, 0.0, 0.0, 0.01);

// Compute Jacobians during plus operation
let mut jac_pose = Matrix6::zeros();
let mut jac_delta = Matrix6::zeros();
let result = pose.plus(&delta, Some(&mut jac_pose), Some(&mut jac_delta));

// Jacobians are now filled:
// jac_pose: ∂(pose ⊞ δ)/∂pose
// jac_delta: ∂(pose ⊞ δ)/∂δ

// Compute Jacobians during minus operation
let pose1 = SE3::random();
let pose2 = SE3::random();
let mut jac_pose1 = Matrix6::zeros();
let mut jac_pose2 = Matrix6::zeros();
let diff = pose1.minus(&pose2, Some(&mut jac_pose1), Some(&mut jac_pose2));

// jac_pose1: ∂(pose1 ⊟ pose2)/∂pose1
// jac_pose2: ∂(pose1 ⊟ pose2)/∂pose2
```

## Mathematical Background

### Lie Groups and Lie Algebras

A Lie group G is a smooth manifold with a group structure. Each Lie group has an associated Lie algebra 𝔤 (the tangent space at the identity), connected via:

- **Exponential map**: exp: 𝔤 → G (maps tangent vectors to group elements)
- **Logarithmic map**: log: G → 𝔤 (maps group elements to tangent vectors)

### Perturbation Models

For optimization on manifolds, we use perturbation models:

**Right Perturbation (default):**
```
g ⊞ δ = g ∘ exp(δ)     (plus)
g₁ ⊟ g₂ = log(g₂⁻¹ ∘ g₁)  (minus)
```

**Left Perturbation:**
```
δ ⊞ g = exp(δ) ∘ g     (plus)
g₁ ⊟ g₂ = log(g₁ ∘ g₂⁻¹)  (minus)
```

### Jacobians

The library provides analytic Jacobians for all operations:

- **Right Jacobian Jr(θ)**: Relates perturbations to the tangent space
- **Left Jacobian Jl(θ)**: Alternative convention for perturbations
- **Adjoint Ad(g)**: Maps tangent vectors between reference frames

## API Reference

### LieGroup Trait

Core operations provided by all manifold types:

| Method | Description | Jacobians |
|--------|-------------|-----------|
| `identity()` | Identity element | - |
| `random()` | Random element | - |
| `inverse(jac)` | Group inverse g⁻¹ | ∂g⁻¹/∂g |
| `compose(other, jac_self, jac_other)` | Group composition g₁ ∘ g₂ | ∂(g₁∘g₂)/∂g₁, ∂(g₁∘g₂)/∂g₂ |
| `log(jac)` | Logarithmic map to tangent space | ∂log(g)/∂g |
| `act(v, jac_self, jac_v)` | Action on vector g ⊙ v | ∂(g⊙v)/∂g, ∂(g⊙v)/∂v |
| `plus(delta, jac_self, jac_delta)` | Right plus g ⊞ δ = g ∘ exp(δ) | ∂(g⊞δ)/∂g, ∂(g⊞δ)/∂δ |
| `minus(other, jac_self, jac_other)` | Right minus g₁ ⊟ g₂ | ∂(g₁⊟g₂)/∂g₁, ∂(g₁⊟g₂)/∂g₂ |
| `adjoint()` | Adjoint matrix Ad(g) | - |
| `between(other, jac_self, jac_other)` | Relative transformation g₁⁻¹ ∘ g₂ | Jacobians w.r.t. both |
| `normalize()` | Project to manifold | - |
| `is_valid(tol)` | Check manifold constraints | - |

### Tangent Trait

Operations for tangent space vectors:

| Method | Description |
|--------|-------------|
| `exp(jac)` | Exponential map to group |
| `right_jacobian()` | Right Jacobian Jr |
| `left_jacobian()` | Left Jacobian Jl |
| `right_jacobian_inv()` | Inverse right Jacobian Jr⁻¹ |
| `left_jacobian_inv()` | Inverse left Jacobian Jl⁻¹ |
| `hat()` | Hat operator (vector to matrix) |
| `zero()` | Zero tangent vector |
| `random()` | Random tangent vector |

## Dependencies

- **[nalgebra](https://nalgebra.org/)**: Linear algebra primitives (vectors, matrices, quaternions)
- **[rand](https://docs.rs/rand/)**: Random number generation for testing
- **[thiserror](https://docs.rs/thiserror/)**: Error handling

## Design Philosophy

The design closely follows the [manif](https://github.com/artivis/manif) C++ library, providing:

- **Consistent API**: Same interface across all manifold types
- **Type safety**: Compile-time guarantees for manifold operations
- **Zero-cost abstractions**: No runtime overhead for type safety
- **Analytic Jacobians**: Exact derivatives, no numerical approximation
- **Comprehensive testing**: All operations verified against reference implementations

## References

- [A micro Lie theory for state estimation in robotics](https://arxiv.org/abs/1812.01537) (Solà et al., 2018)
- [manif: A small header-only library for Lie theory](https://github.com/artivis/manif)
- [State Estimation for Robotics](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf) (Barfoot, 2017)

## License

Apache-2.0
