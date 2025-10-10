//! Manifold representations for optimization on non-Euclidean spaces.
//!
//! This module provides manifold representations commonly used in computer vision and robotics:
//! - **SE(3)**: Special Euclidean group (rigid body transformations)
//! - **SO(3)**: Special Orthogonal group (rotations)
//! - **Sim(3)**: Similarity transformations
//! - **SE(2)**: Rigid transformations in 2D
//! - **SO(2)**: Rotations in 2D
//!
//! Lie group M,° | size   | dim | X ∈ M                   | Constraint      | T_E M             | T_X M                 | Exp(T)             | Comp. | Action
//! ------------- | ------ | --- | ----------------------- | --------------- | ----------------- | --------------------- | ------------------ | ----- | ------
//! n-D vector    | Rⁿ,+   | n   | n   | v ∈ Rⁿ            | |v-v|=0         | v ∈ Rⁿ            | v ∈ Rⁿ                | v = exp(v)         | v₁+v₂ | v + x
//! Circle        | S¹,.   | 2   | 1   | z ∈ C             | z*z = 1         | iθ ∈ iR           | θ ∈ R                 | z = exp(iθ)        | z₁z₂  | zx
//! Rotation      | SO(2),.| 4   | 1   | R                 | RᵀR = I         | [θ]x ∈ so(2)      | [θ] ∈ R²              | R = exp([θ]x)      | R₁R₂  | Rx
//! Rigid motion  | SE(2),.| 9   | 3   | M = [R t; 0 1]    | RᵀR = I         | [v̂] ∈ se(2)       | [v̂] ∈ R³              | Exp([v̂])           | M₁M₂  | Rx+t
//! 3-sphere      | S³,.   | 4   | 3   | q ∈ H             | q*q = 1         | θ/2 ∈ Hp          | θ ∈ R³                | q = exp(uθ/2)      | q₁q₂  | qxq*
//! Rotation      | SO(3),.| 9   | 3   | R                 | RᵀR = I         | [θ]x ∈ so(3)      | [θ] ∈ R³              | R = exp([θ]x)      | R₁R₂  | Rx
//! Rigid motion  | SE(3),.| 16  | 6   | M = [R t; 0 1]    | RᵀR = I         | [v̂] ∈ se(3)       | [v̂] ∈ R⁶              | Exp([v̂])           | M₁M₂  | Rx+t
//!
//! The design is inspired by the [manif](https://github.com/artivis/manif) C++ library
//! and provides:
//! - Analytic Jacobian computations for all operations
//! - Right and left perturbation models
//! - Composition and inverse operations
//! - Exponential and logarithmic maps
//! - Tangent space operations
//!
//! # Mathematical Background
//!
//! This module implements Lie group theory for robotics applications. Each manifold
//! represents a Lie group with its associated tangent space (Lie algebra).
//! Operations are differentiated with respect to perturbations on the local tangent space.
//!

use nalgebra::{Matrix3, Vector3};
use std::fmt::Debug;
use std::ops::{Mul, Neg};

pub mod quaternion;
pub mod rn;
pub mod se2;
pub mod se3;
pub mod so2;
pub mod so3;

/// Errors that can occur during manifold operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ManifoldError {
    /// Invalid tangent vector dimension
    InvalidTangentDimension { expected: usize, actual: usize },
    /// Numerical instability in computation
    NumericalInstability(String),
    /// Invalid manifold element
    InvalidElement(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ManifoldType {
    RN,
    SE2,
    SE3,
    SO2,
    SO3,
}

impl std::fmt::Display for ManifoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifoldError::InvalidTangentDimension { expected, actual } => {
                write!(
                    f,
                    "Invalid tangent dimension: expected {expected}, got {actual}"
                )
            }
            ManifoldError::NumericalInstability(msg) => {
                write!(f, "Numerical instability: {msg}")
            }
            ManifoldError::InvalidElement(msg) => {
                write!(f, "Invalid manifold element: {msg}")
            }
        }
    }
}

impl std::error::Error for ManifoldError {}

/// Result type for manifold operations.
pub type ManifoldResult<T> = Result<T, ManifoldError>;

/// Core trait for Lie group operations.
///
/// This trait provides the fundamental operations for Lie groups, including:
/// - Group operations (composition, inverse, identity)
/// - Exponential and logarithmic maps
/// - Lie group plus/minus operations with Jacobians
/// - Adjoint operations
/// - Random sampling and normalization
///
/// The design closely follows the [manif](https://github.com/artivis/manif) C++ library.
///
/// # Type Parameters
///
/// Associated types define the mathematical structure:
/// - `Element`: The Lie group element type (e.g., `Isometry3<f64>` for SE(3))
/// - `TangentVector`: The tangent space vector type (e.g., `Vector6<f64>` for SE(3))
/// - `JacobianMatrix`: The Jacobian matrix type for this Lie group
/// - `LieAlgebra`: Associated Lie algebra type
///
/// # Dimensions
///
/// Three key dimensions characterize each Lie group:
/// - `DIM`: Space dimension - dimension of ambient space (e.g., 3 for SE(3))
/// - `DOF`: Degrees of freedom - tangent space dimension (e.g., 6 for SE(3))
/// - `REP_SIZE`: Representation size - underlying data size (e.g., 7 for SE(3))
pub trait LieGroup: Clone + Debug + PartialEq {
    /// The tangent space vector type
    type TangentVector: Tangent<Self>;

    /// The Jacobian matrix type
    type JacobianMatrix: Clone
        + Debug
        + PartialEq
        + Neg<Output = Self::JacobianMatrix>
        + Mul<Output = Self::JacobianMatrix>;

    /// Associated Lie algebra type
    type LieAlgebra: Clone + Debug + PartialEq;

    // Core group operations

    /// Compute the inverse of this manifold element.
    ///
    /// For a group element g, returns g⁻¹ such that g ∘ g⁻¹ = e.
    ///
    /// # Arguments
    /// * `jacobian` - Optional mutable reference to store the Jacobian ∂(g⁻¹)/∂g
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self;

    /// Compose this element with another (group multiplication).
    ///
    /// Computes g₁ ∘ g₂ where ∘ is the group operation.
    ///
    /// # Arguments
    /// * `other` - The right operand for composition
    /// * `jacobian_self` - Optional Jacobian ∂(g₁ ∘ g₂)/∂g₁
    /// * `jacobian_other` - Optional Jacobian ∂(g₁ ∘ g₂)/∂g₂
    fn compose(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self;

    /// Logarithmic map from manifold to tangent space.
    ///
    /// Maps a group element g ∈ G to its tangent vector log(g)^∨ ∈ 𝔤.
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian ∂log(g)^∨/∂g
    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector;

    /// Vee operator: log(g)^∨.
    ///
    /// Maps a group element g ∈ G to its tangent vector log(g)^∨ ∈ 𝔤.
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian ∂log(g)^∨/∂g
    fn vee(&self) -> Self::TangentVector;

    /// Act on a vector v: g ⊙ v.
    ///
    /// Group action on vectors (e.g., rotation for SO(3), transformation for SE(3)).
    ///
    /// # Arguments
    /// * `vector` - Vector to transform
    /// * `jacobian_self` - Optional Jacobian ∂(g ⊙ v)/∂g
    /// * `jacobian_vector` - Optional Jacobian ∂(g ⊙ v)/∂v
    fn act(
        &self,
        vector: &Vector3<f64>,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_vector: Option<&mut Matrix3<f64>>,
    ) -> Vector3<f64>;

    // Adjoint operations

    /// Adjoint matrix Ad(g).
    ///
    /// The adjoint representation maps the group to linear transformations
    /// on the Lie algebra: Ad(g) φ = log(g ∘ exp(φ^∧) ∘ g⁻¹)^∨.
    fn adjoint(&self) -> Self::JacobianMatrix;

    // Utility operations

    /// Generate a random element (useful for testing and initialization).
    fn random() -> Self;

    /// Normalize/project the element to the manifold.
    ///
    /// Ensures the element satisfies manifold constraints (e.g., orthogonality for rotations).
    fn normalize(&mut self);

    /// Check if the element is approximately on the manifold.
    fn is_valid(&self, tolerance: f64) -> bool;

    /// Check if the element is approximately equal to another element.
    ///
    /// # Arguments
    /// * `other` - The other element to compare with
    /// * `tolerance` - The tolerance for the comparison
    fn is_approx(&self, other: &Self, tolerance: f64) -> bool;

    // Manifold plus/minus operations

    /// Right plus operation: g ⊞ φ = g ∘ exp(φ^∧).
    ///
    /// Applies a tangent space perturbation to this manifold element.
    ///
    /// # Arguments
    /// * `tangent` - Tangent vector perturbation
    /// * `jacobian_self` - Optional Jacobian ∂(g ⊞ φ)/∂g
    /// * `jacobian_tangent` - Optional Jacobian ∂(g ⊞ φ)/∂φ
    ///
    /// # Notes
    /// # Equation 148:
    /// J_R⊕θ_R = R(θ)ᵀ
    /// J_R⊕θ_θ = J_r(θ)
    fn right_plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        let exp_tangent = tangent.exp(None);

        if let Some(jac_tangent) = jacobian_tangent {
            *jac_tangent = tangent.right_jacobian();
        }

        self.compose(&exp_tangent, jacobian_self, None)
    }

    /// Right minus operation: g₁ ⊟ g₂ = log(g₂⁻¹ ∘ g₁)^∨.
    ///
    /// Computes the tangent vector that transforms g₂ to g₁.
    ///
    /// # Arguments
    /// * `other` - The reference element g₂
    /// * `jacobian_self` - Optional Jacobian ∂(g₁ ⊟ g₂)/∂g₁
    /// * `jacobian_other` - Optional Jacobian ∂(g₁ ⊟ g₂)/∂g₂
    ///
    /// # Notes
    /// # Equation 149:
    /// J_Q⊖R_Q = J_r⁻¹(θ)
    /// J_Q⊖R_R = -J_l⁻¹(θ)
    fn right_minus(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::TangentVector {
        let other_inverse = other.inverse(None);
        let result_group = other_inverse.compose(self, None, None);
        let result = result_group.log(None);

        if let Some(jac_self) = jacobian_self {
            *jac_self = -result.left_jacobian_inv();
        }

        if let Some(jac_other) = jacobian_other {
            *jac_other = result.right_jacobian_inv();
        }

        result
    }

    /// Left plus operation: φ ⊞ g = exp(φ^∧) ∘ g.
    ///
    /// # Arguments
    /// * `tangent` - Tangent vector perturbation
    /// * `jacobian_tangent` - Optional Jacobian ∂(φ ⊞ g)/∂φ
    /// * `jacobian_self` - Optional Jacobian ∂(φ ⊞ g)/∂g
    fn left_plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        // Left plus: τ ⊕ g = exp(τ) * g
        let exp_tangent = tangent.exp(None);
        let result = exp_tangent.compose(self, None, None);

        if let Some(jac_self) = jacobian_self {
            // Note: jacobian_identity() is now implemented in concrete types
            // This will be handled by the concrete implementation
            *jac_self = self.adjoint();
        }

        if let Some(jac_tangent) = jacobian_tangent {
            *jac_tangent = self.inverse(None).adjoint() * tangent.right_jacobian();
        }

        result
    }

    /// Left minus operation: g₁ ⊟ g₂ = log(g₁ ∘ g₂⁻¹)^∨.
    ///
    /// # Arguments
    /// * `other` - The reference element g₂
    /// * `jacobian_self` - Optional Jacobian ∂(g₁ ⊟ g₂)/∂g₁
    /// * `jacobian_other` - Optional Jacobian ∂(g₁ ⊟ g₂)/∂g₂
    fn left_minus(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::TangentVector {
        // Left minus: g1 ⊖ g2 = log(g1 * g2^{-1})
        let other_inverse = other.inverse(None);
        let result_group = self.compose(&other_inverse, None, None);
        let result = result_group.log(None);

        if let Some(jac_self) = jacobian_self {
            *jac_self = result.right_jacobian_inv() * other.adjoint();
        }

        if let Some(jac_other) = jacobian_other {
            *jac_other = -(result.right_jacobian_inv() * other.adjoint());
        }

        result
    }

    // Convenience methods (use right operations by default)

    /// Convenience method for right_plus. Equivalent to g ⊞ φ.
    fn plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        self.right_plus(tangent, jacobian_self, jacobian_tangent)
    }

    /// Convenience method for right_minus. Equivalent to g₁ ⊟ g₂.
    fn minus(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::TangentVector {
        self.right_minus(other, jacobian_self, jacobian_other)
    }

    // Additional operations

    /// Compute g₁⁻¹ ∘ g₂ (relative transformation).
    ///
    /// # Arguments
    /// * `other` - The target element g₂
    /// * `jacobian_self` - Optional Jacobian with respect to g₁
    /// * `jacobian_other` - Optional Jacobian with respect to g₂
    fn between(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        // Between: g1.between(g2) = g1^{-1} * g2
        let self_inverse = self.inverse(None);
        let result = self_inverse.compose(other, None, None);

        if let Some(jac_self) = jacobian_self {
            *jac_self = -result.inverse(None).adjoint();
        }

        if let Some(jac_other) = jacobian_other {
            // Note: jacobian_identity() is now implemented in concrete types
            // This will be handled by the concrete implementation
            *jac_other = other.adjoint();
        }

        result
    }
}

/// Trait for Lie algebra operations.
///
/// This trait provides operations for vectors in the Lie algebra of a Lie group,
/// including vector space operations, adjoint actions, and conversions to matrix form.
///
/// # Type Parameters
///
/// - `G`: The associated Lie group type
pub trait Tangent<Group: LieGroup>: Clone + Debug + PartialEq {
    // Dimension constants

    /// Dimension of the tangent space
    const DIM: usize;

    // Exponential map and Jacobians

    /// Exponential map to Lie group: exp(φ^∧).
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian ∂exp(φ^∧)/∂φ
    fn exp(&self, jacobian: Option<&mut Group::JacobianMatrix>) -> Group;

    /// Right Jacobian Jr.
    ///
    /// Matrix Jr such that for small δφ:
    /// exp((φ + δφ)^∧) ≈ exp(φ^∧) ∘ exp((Jr δφ)^∧)
    fn right_jacobian(&self) -> Group::JacobianMatrix;

    /// Left Jacobian Jl.
    ///
    /// Matrix Jl such that for small δφ:
    /// exp((φ + δφ)^∧) ≈ exp((Jl δφ)^∧) ∘ exp(φ^∧)
    fn left_jacobian(&self) -> Group::JacobianMatrix;

    /// Inverse of right Jacobian Jr⁻¹.
    fn right_jacobian_inv(&self) -> Group::JacobianMatrix;

    /// Inverse of left Jacobian Jl⁻¹.
    fn left_jacobian_inv(&self) -> Group::JacobianMatrix;

    // Matrix representations

    /// Hat operator: φ^∧ (vector to matrix).
    ///
    /// Maps the tangent vector to its matrix representation in the Lie algebra.
    /// For SO(3): 3×1 vector → 3×3 skew-symmetric matrix
    /// For SE(3): 6×1 vector → 4×4 transformation matrix
    fn hat(&self) -> Group::LieAlgebra;

    /// Small adjugate operator: adj(φ) = φ^∧.
    ///
    /// Maps the tangent vector to its matrix representation in the Lie algebra.
    /// For SO(3): 3×1 vector → 3×3 skew-symmetric matrix
    /// For SE(3): 6×1 vector → 4×4 transformation matrix
    fn small_adj(&self) -> Group::JacobianMatrix;

    /// Lie bracket: [φ, ψ] = φ ∘ ψ - ψ ∘ φ.
    ///
    /// Computes the Lie bracket of two tangent vectors in the Lie algebra.
    /// For SO(3): 3×1 vector → 3×1 vector
    /// For SE(3): 6×1 vector → 6×1 vector
    fn lie_bracket(&self, other: &Self) -> Group::TangentVector;

    /// Check if the tangent vector is approximately equal to another tangent vector.
    ///
    /// # Arguments
    /// * `other` - The other tangent vector to compare with
    /// * `tolerance` - The tolerance for the comparison
    fn is_approx(&self, other: &Self, tolerance: f64) -> bool;

    /// Get the i-th generator of the Lie algebra.
    fn generator(&self, i: usize) -> Group::LieAlgebra;

    // Utility functions

    /// Zero tangent vector.
    fn zero() -> Group::TangentVector;

    /// Random tangent vector (useful for testing).
    fn random() -> Group::TangentVector;

    /// Check if the tangent vector is approximately zero.
    fn is_zero(&self, tolerance: f64) -> bool;

    /// Normalize the tangent vector to unit norm.
    fn normalize(&mut self);

    /// Return a unit tangent vector in the same direction.
    fn normalized(&self) -> Group::TangentVector;
}

/// Trait for Lie groups that support interpolation.
pub trait Interpolatable: LieGroup {
    /// Linear interpolation in the manifold.
    ///
    /// For parameter t ∈ [0,1]: interp(g₁, g₂, 0) = g₁, interp(g₁, g₂, 1) = g₂.
    ///
    /// # Arguments
    /// * `other` - Target element for interpolation
    /// * `t` - Interpolation parameter in [0,1]
    fn interp(&self, other: &Self, t: f64) -> Self;

    /// Spherical linear interpolation (when applicable).
    fn slerp(&self, other: &Self, t: f64) -> Self;
}
