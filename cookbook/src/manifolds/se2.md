# SE(2) - 2D Rigid Transforms

Rigid body transformations in the plane (rotation + translation).

## Mathematical Definition

$$
SE(2) = \{(R, t) : R \in SO(2), t \in \mathbb{R}^2\}
$$

As a matrix:

$$
T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta & t_x \\ \sin\theta & \cos\theta & t_y \\ 0 & 0 & 1 \end{bmatrix}
$$

**Group Properties:**
- **Identity**: (I, 0) - no rotation, no translation
- **Composition**: (R₁, t₁) ∘ (R₂, t₂) = (R₁R₂, R₁t₂ + t₁)
- **Inverse**: (R, t)⁻¹ = (Rᵀ, -Rᵀt)

## Tangent Space (Lie Algebra se(2))

- **Dimension**: 3 DOF
- **Representation**: [ρₓ, ρᵧ, θ]
  - **ρ**: Linear velocity / translation component
  - **θ**: Angular velocity / rotation component
- **Physical meaning**: Twist (combined linear + angular velocity)

## Internal Representation

```rust
// Group element: UnitComplex (rotation) + Vector2 (translation)
pub struct SE2 {
    rotation: UnitComplex<f64>,
    translation: Vector2<f64>,
}

// Tangent element
pub struct SE2Tangent {
    value: Vector3<f64>, // [rho_x, rho_y, theta]
}
```

**Data format** (as `DVector<f64>`): `[x, y, θ]` - position and angle

## Exponential Map

$$
\exp(\rho, \theta) = (\exp(\theta), V(\theta) \cdot \rho)
$$

where:

$$
V(\theta) = \begin{bmatrix} \frac{\sin\theta}{\theta} & -\frac{1-\cos\theta}{\theta} \\ \frac{1-\cos\theta}{\theta} & \frac{\sin\theta}{\theta} \end{bmatrix}
$$

**Small angle (θ → 0):**

$$
V(\theta) \to I
$$

**Code:**
```rust
use apex_solver::manifold::se2::{SE2, SE2Tangent};
use nalgebra::Vector3;

// Move forward 1m and rotate 0.1 rad
let tangent = SE2Tangent::new(Vector3::new(1.0, 0.0, 0.1));
let transform = tangent.exp(None);
```

## Logarithmic Map

$$
\log(R, t) = (V^{-1}(\theta) \cdot t, \theta)
$$

where θ = log(R) and:

$$
V^{-1}(\theta) = \begin{bmatrix} \frac{\theta}{2\tan(\theta/2)} & \frac{\theta}{2} \\ -\frac{\theta}{2} & \frac{\theta}{2\tan(\theta/2)} \end{bmatrix}
$$

## Right Jacobian Jr(ρ, θ)

$$
J_r = \begin{bmatrix} a & b & \frac{\partial x}{\partial\theta} \\ -b & a & \frac{\partial y}{\partial\theta} \\ 0 & 0 & 1 \end{bmatrix}
$$

where:
- a = sin(θ)/θ
- b = (1-cos(θ))/θ

The ∂x/∂θ and ∂y/∂θ terms couple rotation to translation.

## Common Operations

```rust
use apex_solver::manifold::se2::SE2;
use nalgebra::Vector2;

// Create transforms
let t1 = SE2::identity();
let t2 = SE2::new(Vector2::new(1.0, 0.0), 0.5); // translation, angle

// Composition
let t12 = t1.compose(&t2, None, None);

// Inverse
let t2_inv = t2.inverse(None);

// Transform a 2D point
let p = Vector2::new(1.0, 0.0);
let transformed = t2.act(&p, None, None);

// Between (relative transform)
let t_rel = t1.between(&t2, None, None); // t1⁻¹ ∘ t2

// Manifold operations
let delta = SE2Tangent::new(Vector3::new(0.1, 0.0, 0.05));
let updated = t2.right_plus(&delta, None, None);
```

## Adjoint Matrix

$$
\text{Ad}(R, t) = \begin{bmatrix} R & [t]_\times \\ 0 & 1 \end{bmatrix}
$$

where [t]× = [-ty, tx]ᵀ is the 2D "cross product" (90° rotation of t).

## Usage in Optimization

```rust
use apex_solver::manifold::ManifoldType;
use std::collections::HashMap;

let mut initial = HashMap::new();
initial.insert(
    "pose".to_string(),
    (ManifoldType::SE2, dvector![1.0, 2.0, 0.5]) // x, y, theta
);
```
