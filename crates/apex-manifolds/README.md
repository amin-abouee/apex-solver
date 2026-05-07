# apex-manifolds

Lie group manifolds (SE2, SE3, SO2, SO3, SE_2(3), SGal(3), Sim(3), Rn) with analytic
Jacobians for nonlinear optimization.

## Overview

This library provides efficient implementations of Lie group manifolds commonly used in
robotics and computer vision applications. All implementations include analytic Jacobians
for optimization algorithms, following the conventions established by the
[manif](https://github.com/artivis/manif) C++ library.

Lie groups are essential mathematical structures for representing:
- **Rigid body transformations**: Camera poses, robot configurations
- **Rotations**: Orientation tracking, attitude estimation
- **Extended kinematics**: Velocity-augmented states for IMU preintegration
- **Similarity transforms**: Monocular SLAM with unknown scale
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
| **SE(3)** | 3D rigid transformations | 6 | 7 (quat + translation) | 3D SLAM, visual odometry, bundle adjustment |
| **SO(3)** | 3D rotations | 3 | 4 (unit quaternion) | Orientation tracking, IMU integration |
| **SE(2)** | 2D rigid transformations | 3 | 4 (angle + translation) | 2D SLAM, mobile robotics |
| **SO(2)** | 2D rotations | 1 | 2 (cos, sin) | 2D orientation, bearing-only SLAM |
| **SE_2(3)** | 3D pose + velocity | 9 | 10 (quat + translation + velocity) | Visual-inertial odometry, IMU preintegration |
| **SGal(3)** | Galilean group (pose + vel + time) | 10 | 11 | Inertial navigation, time-coupled kinematics |
| **Sim(3)** | Similarity transforms (pose + scale) | 7 | 8 (quat + translation + scale) | Monocular SLAM, SfM with unknown scale |
| **Rⁿ** | Euclidean vector space | n | n | Landmarks, camera intrinsics, calibration |

### Mathematical Summary

```
Lie group   | DOF | Rep | X ∈ M                         | Tangent space     | Comp.   | Action
----------- | --- | --- | ----------------------------- | ----------------- | ------- | ------
Rⁿ          | n   | n   | v ∈ Rⁿ                       | v ∈ Rⁿ            | v₁+v₂  | v + x
SO(2)       | 1   | 2   | R ∈ ℝ²ˣ²  (RᵀR=I)           | θ ∈ ℝ             | R₁R₂   | Rx
SE(2)       | 3   | 4   | (R,t), R∈SO(2), t∈ℝ²        | ξ ∈ ℝ³            | M₁M₂   | Rx+t
SO(3)       | 3   | 4   | R ∈ ℝ³ˣ³  (RᵀR=I)           | ω ∈ ℝ³            | R₁R₂   | Rx
SE(3)       | 6   | 7   | (R,t), R∈SO(3), t∈ℝ³        | ξ ∈ ℝ⁶            | M₁M₂   | Rx+t
SE_2(3)     | 9   | 10  | (R,t,v), R∈SO(3), t,v∈ℝ³    | ξ ∈ ℝ⁹            | M₁M₂   | Rx+t
SGal(3)     | 10  | 11  | (R,t,v,s), s∈ℝ               | ξ ∈ ℝ¹⁰           | M₁M₂   | Rx+t
Sim(3)      | 7   | 8   | (R,t,λ), R∈SO(3), λ∈ℝ>0     | ξ ∈ ℝ⁷            | M₁M₂   | λRx+t
```

## Installation

```toml
[dependencies]
apex-manifolds = "0.2.0"
```

## Usage

### SE(3) - 3D Rigid Transformations

SE(3) represents 3D rigid body transformations (rotation + translation), commonly used
for camera poses and robot configurations.

```rust
use apex_manifolds::se3::SE3;
use apex_manifolds::LieGroup;
use nalgebra::{Vector3, Vector6};

// Create an SE3 pose from translation
let pose = SE3::from_translation(Vector3::new(1.0, 2.0, 3.0));

// Compose two transformations
let composed = pose.compose(&SE3::identity(), None, None);

// Apply tangent space perturbation (right plus)
let delta = Vector6::new(0.1, 0.0, 0.0, 0.0, 0.0, 0.01);  // [vx, vy, vz, ωx, ωy, ωz]
let perturbed = pose.plus(&delta, None, None);

// Compute relative transformation (right minus)
let relative = perturbed.minus(&pose, None, None);

// Transform a 3D point
let transformed = pose.act(&Vector3::new(1.0, 0.0, 0.0), None, None);

// Get inverse
let inverse = pose.inverse(None);
```

### SO(3) - 3D Rotations

SO(3) represents 3D rotations, stored internally as a unit quaternion in w-first
(Hamilton) convention `[qw, qx, qy, qz]`.

```rust
use apex_manifolds::so3::SO3;
use apex_manifolds::LieGroup;
use nalgebra::Vector3;
use std::f64::consts::PI;

// Create rotation from axis-angle
let rotation = SO3::from_axis_angle(&Vector3::new(0.0, 0.0, 1.0), PI / 4.0);

// Compose rotations
let double = rotation.compose(&rotation, None, None);  // 90° rotation

// Rotate a vector
let rotated = rotation.act(&Vector3::new(1.0, 0.0, 0.0), None, None);

// Logarithmic map (rotation → axis-angle vector)
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

// Compose and perturb
let composed = pose.compose(&translation, None, None);
let delta = Vector3::new(0.1, 0.0, 0.01);  // [dx, dy, dθ]
let perturbed = pose.plus(&delta, None, None);
```

### SO(2) - 2D Rotations

SO(2) represents 2D rotations (unit complex numbers).

```rust
use apex_manifolds::so2::SO2;
use apex_manifolds::LieGroup;
use nalgebra::Vector3;
use std::f64::consts::PI;

let rotation = SO2::from_angle(PI / 2.0);  // 90°
let angle = rotation.angle();
let rotated = rotation.act(&Vector3::new(1.0, 0.0, 0.0), None, None);
```

### SE_2(3) - Extended Pose with Velocity

SE_2(3) is the "extended special Euclidean group" that augments SE(3) with a velocity
vector. It is the natural state manifold for IMU preintegration in visual-inertial
odometry (Barrau & Bonnabel, 2017).

Internal representation: `[tx, ty, tz, vx, vy, vz, qw, qx, qy, qz]` (10 scalars).

```rust
use apex_manifolds::se23::SE23;
use apex_manifolds::LieGroup;
use nalgebra::{Vector3, SVector};

// Identity: zero pose and zero velocity
let identity = SE23::identity();
let state = SE23::random();

// Compose two extended poses
let composed = state.compose(&identity, None, None);

// Apply 9-DOF tangent perturbation [δtranslation, δvelocity, δrotation]
let delta: SVector<f64, 9> = SVector::zeros();
let perturbed = state.plus(&delta, None, None);

// Compute relative state
let relative = state.minus(&identity, None, None);
```

### SGal(3) - Galilean Group

SGal(3) extends SE_2(3) with a scalar time parameter, forming the special Galilean group.
It captures the symmetry of Newtonian kinematics and is used in inertial navigation
systems where time is part of the state.

Internal representation: `[tx, ty, tz, vx, vy, vz, s, qw, qx, qy, qz]` (11 scalars).

```rust
use apex_manifolds::sgal3::SGal3;
use apex_manifolds::LieGroup;
use nalgebra::SVector;

let state = SGal3::random();
let identity = SGal3::identity();

// Compose two Galilean states
let composed = state.compose(&identity, None, None);

// Apply 10-DOF tangent perturbation
let delta: SVector<f64, 10> = SVector::zeros();
let perturbed = state.plus(&delta, None, None);
```

### Sim(3) - Similarity Transformations

Sim(3) extends SE(3) with a positive scale factor λ, making it the natural state space
for monocular SLAM and structure-from-motion where metric scale is unobservable. The
group action is g ⊙ x = λ R x + t.

Internal representation: `[tx, ty, tz, qw, qx, qy, qz, λ]` (8 scalars).

```rust
use apex_manifolds::sim3::Sim3;
use apex_manifolds::LieGroup;
use nalgebra::{Vector3, SVector};

// Identity: unit scale, zero pose
let identity = Sim3::identity();
let a = Sim3::random();

// Compose: scales multiply, translations accumulate
let b = Sim3::random();
let composed = a.compose(&b, None, None);

// Transform a 3D point: result = λ · R · x + t
let transformed = a.act(&Vector3::new(1.0, 0.0, 0.0), None, None);

// Apply 7-DOF perturbation [δt, δω, δσ]  (σ = log-scale)
let delta: SVector<f64, 7> = SVector::zeros();
let perturbed = a.plus(&delta, None, None);
```

### Rⁿ - Euclidean Vector Space

Rⁿ represents Euclidean vector spaces with dynamic dimension, useful for landmarks and
calibration parameters.

```rust
use apex_manifolds::rn::Rn;
use apex_manifolds::LieGroup;
use nalgebra::DVector;

let point = Rn::from_vector(DVector::from_vec(vec![1.0, 2.0, 3.0]));
let landmark = Rn::from_slice(&[4.0, 5.0, 6.0]);

let delta = DVector::from_vec(vec![0.1, 0.2, 0.3]);
let updated = point.plus(&delta, None, None);
let diff = updated.minus(&point, None, None);
```

## Computing Jacobians

All manifold operations accept optional mutable Jacobian references:

```rust
use apex_manifolds::se3::SE3;
use apex_manifolds::LieGroup;
use nalgebra::{Matrix6, Vector6};

let pose = SE3::random();
let delta = Vector6::new(0.1, 0.0, 0.0, 0.0, 0.0, 0.01);

// Jacobians of plus w.r.t. pose and delta
let mut jac_pose = Matrix6::zeros();
let mut jac_delta = Matrix6::zeros();
let result = pose.plus(&delta, Some(&mut jac_pose), Some(&mut jac_delta));
// jac_pose: ∂(pose ⊞ δ)/∂pose
// jac_delta: ∂(pose ⊞ δ)/∂δ

// Jacobians of minus w.r.t. both operands
let pose1 = SE3::random();
let pose2 = SE3::random();
let mut jac1 = Matrix6::zeros();
let mut jac2 = Matrix6::zeros();
let diff = pose1.minus(&pose2, Some(&mut jac1), Some(&mut jac2));
```

## Mathematical Background

### Lie Groups and Lie Algebras

A Lie group G is a smooth manifold with a group structure. Each Lie group has an
associated Lie algebra 𝔤 (the tangent space at the identity), connected via:

- **Exponential map**: exp: 𝔤 → G
- **Logarithmic map**: log: G → 𝔤

### Perturbation Models

**Right Perturbation (default):**
```
g ⊞ δ = g ∘ exp(δ)
g₁ ⊟ g₂ = log(g₂⁻¹ ∘ g₁)
```

**Left Perturbation:**
```
δ ⊞ g = exp(δ) ∘ g
g₁ ⊟ g₂ = log(g₁ ∘ g₂⁻¹)
```

### Numerical Stability

A small-angle threshold (`SMALL_ANGLE_THRESHOLD = 1e-10`) guards all Taylor approximations
to prevent catastrophic cancellation in sin(θ)/θ and (1−cos θ)/θ² near θ = 0.

## API Reference

### LieGroup Trait

| Method | Description | Jacobians |
|--------|-------------|-----------|
| `identity()` | Identity element | - |
| `random()` | Random element | - |
| `inverse(jac)` | Group inverse g⁻¹ | ∂g⁻¹/∂g |
| `compose(other, jl, jr)` | Composition g₁ ∘ g₂ | ∂/∂g₁, ∂/∂g₂ |
| `log(jac)` | Logarithmic map | ∂log(g)/∂g |
| `act(v, jg, jv)` | Action g ⊙ v | ∂/∂g, ∂/∂v |
| `plus(δ, jg, jδ)` | Right plus g ⊞ δ | ∂/∂g, ∂/∂δ |
| `minus(other, j1, j2)` | Right minus g₁ ⊟ g₂ | ∂/∂g₁, ∂/∂g₂ |
| `between(other, j1, j2)` | Relative g₁⁻¹ ∘ g₂ | ∂/∂g₁, ∂/∂g₂ |
| `adjoint()` | Adjoint Ad(g) | - |
| `normalize()` | Project to manifold | - |
| `is_valid(tol)` | Check constraints | - |

### Tangent Trait

| Method | Description |
|--------|-------------|
| `exp(jac)` | Exponential map |
| `right_jacobian()` | Right Jacobian Jr |
| `left_jacobian()` | Left Jacobian Jl |
| `right_jacobian_inv()` | Jr⁻¹ |
| `left_jacobian_inv()` | Jl⁻¹ |
| `hat()` | Hat operator (vector → matrix) |
| `zero()` | Zero tangent vector |
| `random()` | Random tangent vector |

## Dependencies

- **[nalgebra](https://nalgebra.org/)**: Linear algebra (vectors, matrices, quaternions)
- **[rand](https://docs.rs/rand/)**: Random number generation
- **[thiserror](https://docs.rs/thiserror/)**: Error handling

## Design Philosophy

- **Consistent API**: Same interface across all eight manifold types
- **Type safety**: Compile-time guarantees for manifold operations
- **Analytic Jacobians**: Exact derivatives, no numerical approximation
- **Numerical stability**: Small-angle threshold guards all Taylor approximations
- **Comprehensive testing**: All operations verified against reference implementations

## References

- [A micro Lie theory for state estimation in robotics](https://arxiv.org/abs/1812.01537) (Solà et al., 2018)
- [The Invariant Extended Kalman Filter as a stable observer](https://arxiv.org/abs/1410.1465) (Barrau & Bonnabel, 2017) — SE_2(3), SGal(3)
- [Lie Groups for Computer Vision](https://ethaneade.com/lie_groups.pdf) (Eade, 2017) — Sim(3)
- [manif: A small header-only library for Lie theory](https://github.com/artivis/manif)
- [State Estimation for Robotics](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf) (Barfoot, 2017)

## License

Apache-2.0
