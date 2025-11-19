# SO(2) - 2D Rotations

The simplest Lie group—rotations in the plane.

## Mathematical Definition

$$
SO(2) = \{R \in \mathbb{R}^{2 \times 2} : R^T R = I, \det(R) = 1\}
$$

Equivalently, represented as angles θ ∈ [-π, π] or unit complex numbers e^(iθ).

**Group Properties:**
- **Identity**: θ = 0 (or complex 1 + 0i)
- **Composition**: θ₁ ∘ θ₂ = θ₁ + θ₂ (wrapping to [-π, π])
- **Inverse**: θ⁻¹ = -θ
- **Abelian**: θ₁ ∘ θ₂ = θ₂ ∘ θ₁ (commutative)

## Tangent Space

- **Dimension**: 1 DOF
- **Representation**: Single angle δθ ∈ ℝ
- **Physical meaning**: Infinitesimal rotation angle

## Internal Representation

```rust
// Group element: UnitComplex (cos θ, sin θ)
pub struct SO2 {
    value: UnitComplex<f64>,
}

// Tangent element: single angle
pub struct SO2Tangent {
    value: f64,
}
```

**Data format** (as `DVector<f64>`): `[θ]` - single element

## Exponential Map

Maps angle to rotation:

$$
\exp(\delta\theta) = (\cos \delta\theta, \sin \delta\theta)
$$

**Code:**
```rust
use apex_solver::manifold::so2::{SO2, SO2Tangent};

let tangent = SO2Tangent::new(std::f64::consts::FRAC_PI_4); // 45 degrees
let rotation = tangent.exp(None); // Jacobian optional

// With Jacobian computation
let mut jacobian = nalgebra::Matrix1::zeros();
let rotation = tangent.exp(Some(&mut jacobian));
// jacobian = [1.0] (always identity for SO2)
```

## Logarithmic Map

Maps rotation to angle:

$$
\log(\cos \theta, \sin \theta) = \text{atan2}(\sin \theta, \cos \theta)
$$

**Code:**
```rust
let rotation = SO2::from_angle(0.5);
let tangent = rotation.log(None);
assert!((tangent.value() - 0.5).abs() < 1e-10);
```

## Jacobians

Since SO(2) is 1-dimensional and Abelian:

$$
J_r(\theta) = J_l(\theta) = J_r^{-1}(\theta) = J_l^{-1}(\theta) = 1
$$

All Jacobians are the identity (scalar 1).

## Common Operations

```rust
use apex_solver::manifold::so2::SO2;
use nalgebra::Vector2;

// Create rotations
let r1 = SO2::identity();
let r2 = SO2::from_angle(0.5);
let r3 = SO2::random();

// Composition
let r12 = r1.compose(&r2, None, None);

// Inverse
let r2_inv = r2.inverse(None);

// Rotate a 2D vector
let v = Vector2::new(1.0, 0.0);
let rotated = r2.act(&v, None, None);

// Right plus (manifold update)
let delta = SO2Tangent::new(0.1);
let updated = r2.right_plus(&delta, None, None);

// Right minus (manifold difference)
let diff = r2.right_minus(&r1, None, None); // Returns SO2Tangent
```

## Adjoint Matrix

For SO(2), the adjoint is simply 1:

$$
\text{Ad}(\theta) = 1
$$
