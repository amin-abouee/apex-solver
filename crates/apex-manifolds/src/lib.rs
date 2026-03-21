//! Manifold representations for optimization on non-Euclidean spaces.
//!
//! This module provides manifold representations commonly used in computer vision and robotics:
//! - **SE(3)**: Special Euclidean group (rigid body transformations)
//! - **SO(3)**: Special Orthogonal group (rotations)
//! - **Sim(3)**: Similarity transformations (rotation + translation + scale)
//! - **SGal(3)**: Special Galilean group (rotation + translation + velocity + time)
//! - **SE_2(3)**: Extended Special Euclidean group (rotation + translation + velocity)
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
//! Similarity    | Sim(3),.| 16 | 7   | M = [sR t; 0 1]   | RᵀR=I, s>0     | [v̂] ∈ sim(3)      | [ρ,θ,σ] ∈ R⁷          | Exp([v̂])           | M₁M₂  | sRx+t
//! Galilean      | SGal(3),.| 25| 10  | (R,t,v,s)         | RᵀR = I         | [v̂] ∈ sgal(3)     | [ρ,ν,θ,s] ∈ R¹⁰       | Exp([v̂])           | M₁M₂  | Rx+t+sv
//! Extended pose | SE_2(3),.| 25| 9   | (R,t,v)           | RᵀR = I         | [v̂] ∈ se_2_3      | [ρ,θ,ν] ∈ R⁹          | Exp([v̂])           | M₁M₂  | Rx+t
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
use std::ops::{Mul, Neg};
use std::{
    error, fmt,
    fmt::{Display, Formatter},
};

/// Threshold for switching between exact formulas and Taylor approximations
/// in small-angle computations.
///
/// `f64::EPSILON` (~2.2e-16) is too tight for small-angle detection — angles of ~1e-8 radians
/// are well within Taylor approximation validity but would fall through to the exact formula
/// path, where division by near-zero values causes numerical issues.
///
/// `1e-10` is chosen because:
/// - Taylor expansions for sin(θ)/θ, (1-cos(θ))/θ² are accurate to ~1e-20 at this scale
/// - Avoids catastrophic cancellation in exact formulas near zero
/// - Consistent with production SLAM libraries (Sophus, GTSAM)
///
/// **Note:** This threshold is compared against `θ²` (not `θ`), so the effective angle
/// threshold is `√(1e-10) ≈ 1e-5` radians (~0.00057°).
pub const SMALL_ANGLE_THRESHOLD: f64 = 1e-10;

pub mod rn;
pub mod se2;
pub mod se3;
pub mod se23;
pub mod sgal3;
pub mod sim3;
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
    /// Dimension validation failed during conversion
    DimensionMismatch { expected: usize, actual: usize },
    /// NaN or Inf detected in manifold element
    InvalidNumber,
    /// Normalization failed for manifold element
    NormalizationFailed(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ManifoldType {
    RN,
    SE2,
    SE3,
    SE23,
    SGal3,
    Sim3,
    SO2,
    SO3,
}

impl Display for ManifoldError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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
            ManifoldError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
            ManifoldError::InvalidNumber => {
                write!(f, "Invalid number: NaN or Inf detected")
            }
            ManifoldError::NormalizationFailed(msg) => {
                write!(f, "Normalization failed: {msg}")
            }
        }
    }
}

impl error::Error for ManifoldError {}

/// Result type for manifold operations.
pub type ManifoldResult<T> = Result<T, ManifoldError>;

/// Core trait for Lie group operations.
///
/// Provides group operations, exponential/logarithmic maps, plus/minus with Jacobians,
/// and adjoint representations.
///
/// # Associated Types
///
/// - `TangentVector`: Tangent space (Lie algebra) vector type
/// - `JacobianMatrix`: Jacobian matrix type for this group
/// - `LieAlgebra`: Matrix representation of the Lie algebra
pub trait LieGroup: Clone + PartialEq {
    /// The tangent space vector type
    type TangentVector: Tangent<Self>;

    /// The Jacobian matrix type
    type JacobianMatrix: Clone
        + PartialEq
        + Neg<Output = Self::JacobianMatrix>
        + Mul<Output = Self::JacobianMatrix>
        + std::ops::Index<(usize, usize), Output = f64>;

    /// Associated Lie algebra type
    type LieAlgebra: Clone + PartialEq;

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

    /// Get the identity matrix for Jacobians.
    ///
    /// Returns the identity matrix in the appropriate dimension for Jacobian computations.
    /// This is used to initialize Jacobian matrices in optimization algorithms.
    fn jacobian_identity() -> Self::JacobianMatrix;

    /// Get a zero Jacobian matrix.
    ///
    /// Returns a zero matrix in the appropriate dimension for Jacobian computations.
    /// This is used to initialize Jacobian matrices before optimization computations.
    fn zero_jacobian() -> Self::JacobianMatrix;

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
        let exp_tangent = tangent.exp(None);
        let result = exp_tangent.compose(self, None, None);

        if let Some(jac_self) = jacobian_self {
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
        let self_inverse = self.inverse(None);
        let result = self_inverse.compose(other, None, None);

        if let Some(jac_self) = jacobian_self {
            *jac_self = -result.inverse(None).adjoint();
        }

        if let Some(jac_other) = jacobian_other {
            *jac_other = Self::jacobian_identity();
        }

        result
    }

    /// Get the dimension of the tangent space for this manifold element.
    ///
    /// For most manifolds, this returns the compile-time constant from the TangentVector type.
    /// For dynamically-sized manifolds like Rⁿ, this method should be overridden to return
    /// the actual runtime dimension.
    ///
    /// # Returns
    /// The dimension of the tangent space (degrees of freedom)
    ///
    /// # Default Implementation
    /// Returns `Self::TangentVector::DIM` which works for fixed-size manifolds
    /// (SE2=3, SE3=6, SO2=1, SO3=3).
    fn tangent_dim(&self) -> usize {
        Self::TangentVector::DIM
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
pub trait Tangent<Group: LieGroup>: Clone + PartialEq {
    // Dimension constants

    /// Dimension of the tangent space.
    ///
    /// For fixed-size manifolds (SE2, SE3, SO2, SO3), this is the compile-time constant.
    /// For dynamic-size manifolds (Rn), this is `0` as a sentinel value — use the
    /// `is_dynamic()` method to check, and the `LieGroup::tangent_dim()` instance method
    /// to get the actual runtime dimension.
    const DIM: usize;

    /// Whether this tangent type has dynamic (runtime-determined) dimension.
    ///
    /// Returns `true` for `RnTangent` where `DIM == 0` is used as a sentinel.
    /// Returns `false` for all fixed-size tangent types (SE2, SE3, SO2, SO3).
    fn is_dynamic() -> bool {
        Self::DIM == 0
    }

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
