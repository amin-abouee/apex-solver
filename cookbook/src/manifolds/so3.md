# SO(3) - 3D Rotations

The group of 3D rotations, fundamental to robotics and computer vision.

## Mathematical Definition

$$
SO(3) = \{R \in \mathbb{R}^{3 \times 3} : R^T R = I, \det(R) = 1\}
$$

Represented internally as unit quaternions for numerical stability.

**Group Properties:**
- **Identity**: Quaternion (1, 0, 0, 0)
- **Composition**: q₁ ∘ q₂ = q₁ * q₂ (quaternion multiplication)
- **Inverse**: q⁻¹ = q* (quaternion conjugate)
- **Non-Abelian**: q₁ ∘ q₂ ≠ q₂ ∘ q₁

## Tangent Space (Lie Algebra so(3))

- **Dimension**: 3 DOF
- **Representation**: Axis-angle vector θ = [θₓ, θᵧ, θᵤ]
  - **Direction**: Rotation axis (unit vector when normalized)
  - **Magnitude**: Rotation angle |θ| in radians
- **Physical meaning**: Infinitesimal rotation / angular velocity

## Internal Representation

```rust
// Group element: UnitQuaternion
pub struct SO3 {
    value: UnitQuaternion<f64>,
}

// Tangent element: axis-angle vector
pub struct SO3Tangent {
    value: Vector3<f64>,
}
```

**Data format** (as `DVector<f64>`): `[qw, qx, qy, qz]` - quaternion (w, i, j, k)

## Hat Operator (Vector to Skew-Symmetric Matrix)

$$
[\theta]_\times = \hat{\theta} = \begin{bmatrix} 0 & -\theta_z & \theta_y \\ \theta_z & 0 & -\theta_x \\ -\theta_y & \theta_x & 0 \end{bmatrix}
$$

This is the cross-product matrix: `[θ]× v = θ × v`

## Exponential Map

Maps axis-angle to quaternion using Rodrigues' formula:

$$
\exp(\theta) = \cos\left(\frac{|\theta|}{2}\right) + \frac{\sin(|\theta|/2)}{|\theta|} \cdot \theta
$$

As a rotation matrix:

$$
\exp([\theta]_\times) = I + \frac{\sin|\theta|}{|\theta|} [\theta]_\times + \frac{1-\cos|\theta|}{|\theta|^2} [\theta]_\times^2
$$

**Small angle approximation** (|θ|² < ε):

$$
\exp(\theta) \approx 1 + \frac{\theta}{2} \quad \text{(quaternion)}
$$

**Code:**
```rust
use apex_solver::manifold::so3::{SO3, SO3Tangent};
use nalgebra::Vector3;

// Rotation of 0.1 radians about z-axis
let axis_angle = Vector3::new(0.0, 0.0, 0.1);
let tangent = SO3Tangent::new(axis_angle);
let rotation = tangent.exp(None);

// With Jacobian
let mut jac = nalgebra::Matrix3::zeros();
let rotation = tangent.exp(Some(&mut jac));
```

## Logarithmic Map

Maps quaternion to axis-angle:

$$
\log(q) = \frac{2 \cdot \text{atan2}(|v|, w)}{|v|} \cdot v
$$

where q = w + v (scalar + vector parts).

**Code:**
```rust
let rotation = SO3::from_axis_angle(
    &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
    0.5
);
let tangent = rotation.log(None);
// tangent ≈ [0, 0, 0.5]
```

## Right Jacobian Jr(θ)

The right Jacobian relates perturbations in tangent space to perturbations on the group:

$$
\exp(\theta + \delta\theta) \approx \exp(\theta) \circ \exp(J_r(\theta) \cdot \delta\theta)
$$

**Formula:**

$$
J_r(\theta) = I - \frac{1 - \cos|\theta|}{|\theta|^2} [\theta]_\times + \frac{|\theta| - \sin|\theta|}{|\theta|^3} [\theta]_\times^2
$$

**Small angle approximation:**

$$
J_r(\theta) \approx I - \frac{1}{2}[\theta]_\times
$$

**Code:**
```rust
let tangent = SO3Tangent::new(Vector3::new(0.1, 0.2, 0.3));
let jr = tangent.right_jacobian();      // 3×3 matrix
let jr_inv = tangent.right_jacobian_inv(); // Inverse
```

## Left Jacobian Jl(θ)

$$
J_l(\theta) = I + \frac{1 - \cos|\theta|}{|\theta|^2} [\theta]_\times + \frac{|\theta| - \sin|\theta|}{|\theta|^3} [\theta]_\times^2
$$

**Relationship to right Jacobian:**

$$
J_l(\theta) = J_r(-\theta) = J_r(\theta)^T
$$

## Common Operations

```rust
use apex_solver::manifold::so3::SO3;
use nalgebra::{Vector3, UnitQuaternion};

// Create rotations
let r1 = SO3::identity();
let r2 = SO3::from_quaternion(UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3));
let r3 = SO3::from_axis_angle(
    &nalgebra::Unit::new_normalize(Vector3::z()),
    std::f64::consts::FRAC_PI_2
);

// Composition (with Jacobians)
let mut jac_self = nalgebra::Matrix3::zeros();
let mut jac_other = nalgebra::Matrix3::zeros();
let r12 = r1.compose(&r2, Some(&mut jac_self), Some(&mut jac_other));

// Inverse
let r2_inv = r2.inverse(None);

// Rotate a 3D vector
let v = Vector3::new(1.0, 0.0, 0.0);
let rotated = r2.act(&v, None, None);

// Get rotation matrix
let mat = r2.rotation_matrix();

// Distance between rotations (geodesic)
let dist = r1.distance(&r2); // |log(r1⁻¹ ∘ r2)|

// Right plus/minus
let delta = SO3Tangent::new(Vector3::new(0.01, 0.02, 0.03));
let updated = r2.right_plus(&delta, None, None);
let diff = r2.right_minus(&r1, None, None);
```

## Adjoint Matrix

The adjoint maps between left and right tangent spaces:

$$
\text{Ad}(R) = R \quad \text{(the rotation matrix itself)}
$$

**Property:** `R ∘ exp(θ) = exp(Ad(R) · θ) ∘ R`
