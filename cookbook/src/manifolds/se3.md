# SE(3) - 3D Rigid Transforms

The most important group for 3D robotics—6 DOF rigid body transformations.

## Mathematical Definition

$$
SE(3) = \{(R, t) : R \in SO(3), t \in \mathbb{R}^3\}
$$

As a 4×4 matrix:

$$
T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}
$$

**Group Properties:**
- **Identity**: (I, 0)
- **Composition**: (R₁, t₁) ∘ (R₂, t₂) = (R₁R₂, R₁t₂ + t₁)
- **Inverse**: (R, t)⁻¹ = (Rᵀ, -Rᵀt)

## Tangent Space (Lie Algebra se(3))

- **Dimension**: 6 DOF
- **Representation**: [ρₓ, ρᵧ, ρᵤ, θₓ, θᵧ, θᵤ]
  - **ρ (rho)**: Translation component (linear velocity)
  - **θ (theta)**: Rotation component (angular velocity, axis-angle)
- **Physical meaning**: Spatial twist

**Important:** The tangent space ordering is [translation, rotation], which differs from some conventions.

## Internal Representation

```rust
// Group element
pub struct SE3 {
    rotation: SO3,           // UnitQuaternion
    translation: Vector3<f64>,
}

// Tangent element
pub struct SE3Tangent {
    value: Vector6<f64>, // [rho_x, rho_y, rho_z, theta_x, theta_y, theta_z]
}
```

**Data format** (as `DVector<f64>`): `[tx, ty, tz, qw, qx, qy, qz]` - 7 elements

## Exponential Map

$$
\exp(\rho, \theta) = (\exp(\theta), V(\theta) \cdot \rho)
$$

where exp(θ) is the SO(3) exponential and:

$$
V(\theta) = I + \frac{1 - \cos|\theta|}{|\theta|^2} [\theta]_\times + \frac{|\theta| - \sin|\theta|}{|\theta|^3} [\theta]_\times^2
$$

**Code:**
```rust
use apex_solver::manifold::se3::{SE3, SE3Tangent};
use nalgebra::{Vector3, Vector6};

// Translation + rotation tangent vector
let tangent = SE3Tangent::new(
    Vector3::new(1.0, 0.0, 0.0),  // translation (rho)
    Vector3::new(0.0, 0.0, 0.1),  // rotation (theta)
);

// Or from Vector6 directly
let tangent = SE3Tangent::from_vector(
    Vector6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.1)
);

let transform = tangent.exp(None);
```

## Logarithmic Map

$$
\log(R, t) = (V^{-1}(\theta) \cdot t, \log(R))
$$

**Code:**
```rust
let transform = SE3::from_translation_and_rotation(
    Vector3::new(1.0, 2.0, 3.0),
    SO3::from_euler_angles(0.1, 0.2, 0.3),
);
let tangent = transform.log(None);
```

## Right Jacobian Jr (SE(3))

The SE(3) Jacobian has a block structure:

$$
J_r(\rho, \theta) = \begin{bmatrix} J_{r_\theta}(\theta) & Q(\rho, \theta) \\ 0 & J_{r_\theta}(\theta) \end{bmatrix}
$$

where Jr_θ is the SO(3) right Jacobian and Q is a coupling term:

$$
Q(\rho, \theta) = \frac{1}{2}[\rho]_\times + c_1([\theta]_\times[\rho]_\times + [\rho]_\times[\theta]_\times + [\theta]_\times[\rho]_\times[\theta]_\times)
$$

$$
- c_2([\theta]_\times^2[\rho]_\times + [\rho]_\times[\theta]_\times^2 - 3[\theta]_\times[\rho]_\times[\theta]_\times)
$$

$$
+ c_3([\theta]_\times[\rho]_\times[\theta]_\times^2 + [\theta]_\times^2[\rho]_\times[\theta]_\times)
$$

with coefficients:
- c₁ = (|θ| - sin|θ|)/|θ|³
- c₂ = (|θ|² + 2cos|θ| - 2)/(2|θ|⁴)
- c₃ = (2|θ| - 3sin|θ| + |θ|cos|θ|)/(2|θ|⁵)

**Code:**
```rust
let tangent = SE3Tangent::new(
    Vector3::new(0.1, 0.2, 0.3),
    Vector3::new(0.01, 0.02, 0.03),
);

let jr = tangent.right_jacobian();      // 6×6 matrix
let jr_inv = tangent.right_jacobian_inv();
```

## Left Jacobian

$$
J_l(\rho, \theta) = J_r(-\rho, -\theta)
$$

## Common Operations

```rust
use apex_solver::manifold::se3::SE3;
use nalgebra::{Vector3, UnitQuaternion};

// Create transforms
let t1 = SE3::identity();
let t2 = SE3::from_translation_and_rotation(
    Vector3::new(1.0, 0.0, 0.0),
    SO3::from_euler_angles(0.0, 0.0, 0.1),
);

// Alternative constructors
let t3 = SE3::from_translation(Vector3::new(1.0, 2.0, 3.0));
let t4 = SE3::from_rotation(SO3::random());

// Composition with Jacobians
let mut jac_self = nalgebra::Matrix6::zeros();
let mut jac_other = nalgebra::Matrix6::zeros();
let t12 = t1.compose(&t2, Some(&mut jac_self), Some(&mut jac_other));

// Inverse
let t2_inv = t2.inverse(None);

// Transform a 3D point
let p = Vector3::new(1.0, 0.0, 0.0);
let mut jac_pose = nalgebra::Matrix3x6::zeros();
let mut jac_point = nalgebra::Matrix3::zeros();
let transformed = t2.act(&p, Some(&mut jac_pose), Some(&mut jac_point));

// Between (relative transform for factor graphs)
let t_rel = t1.between(&t2, None, None); // t1⁻¹ ∘ t2

// Manifold operations
let delta = SE3Tangent::new(
    Vector3::new(0.01, 0.0, 0.0),
    Vector3::new(0.0, 0.0, 0.001),
);
let updated = t2.right_plus(&delta, None, None);
let diff = t2.right_minus(&t1, None, None);

// Access components
let rotation = t2.rotation();
let translation = t2.translation();
```

## Adjoint Matrix

$$
\text{Ad}(R, t) = \begin{bmatrix} R & [t]_\times R \\ 0 & R \end{bmatrix}
$$

This 6×6 matrix transforms twists between frames.

**Code:**
```rust
let transform = SE3::from_translation_and_rotation(
    Vector3::new(1.0, 0.0, 0.0),
    SO3::identity(),
);
let adj = transform.adjoint(); // 6×6 matrix
```

## Usage in Optimization

**Variable initialization:**
```rust
use apex_solver::manifold::ManifoldType;
use nalgebra::dvector;
use std::collections::HashMap;

let mut initial = HashMap::new();
initial.insert(
    "pose0".to_string(),
    (ManifoldType::SE3, dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    //                          tx   ty   tz   qw   qx   qy   qz
);
```

**Fixed variable indices** (tangent space DOF):
- 0, 1, 2: Translation (vx, vy, vz)
- 3, 4, 5: Rotation (ωx, ωy, ωz)

```rust
problem.fix_variable("pose0", 0); // Fix x translation
problem.fix_variable("pose0", 1); // Fix y translation
problem.fix_variable("pose0", 2); // Fix z translation
// Rotation (3, 4, 5) remains free
```
