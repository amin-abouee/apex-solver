//! Manifold representations for optimization on non-Euclidean spaces.
//!
//! This module provides manifold representations commonly used in computer vision and robotics:
//! - **SE(3)**: Special Euclidean group (rigid body transformations)
//! - **SO(3)**: Special Orthogonal group (rotations)
//! - **Sim(3)**: Similarity transformations
//! - **SE(2)**: Rigid transformations in 2D
//! - **SO(2)**: Rotations in 2D
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
//! # Example
//!
//! ```rust,ignore
//! use nalgebra::Vector3;
//!
//! // Create a random SE(3) element
//! let pose = SE3::random();
//!
//! // Create a tangent vector (6-DoF: [rho, theta])
//! let tangent = SE3Tangent::new(Vector3::new(0.1, 0.0, 0.0), Vector3::new(0.0, 0.1, 0.0));
//!
//! // Apply perturbation with Jacobian computation
//! let mut jacobian = Matrix6::zeros();
//! let perturbed = pose.plus(&tangent, Some(&mut jacobian), None);
//! ```

use nalgebra::{DMatrix, Matrix3, Vector3};
use std::fmt::Debug;

pub mod se3;

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
    /// The Lie group element type
    type Element: Clone + Debug + PartialEq;

    /// The tangent space vector type
    type TangentVector: Clone + Debug + PartialEq;

    /// The Jacobian matrix type
    type JacobianMatrix: Clone + Debug + PartialEq;

    /// Associated Lie algebra type
    type LieAlgebra: LieAlgebra<Self>;

    // Dimension constants (following manif conventions)

    /// Space dimension - dimension of the ambient space that the group acts on
    const DIM: usize;

    /// Degrees of freedom - dimension of the tangent space
    const DOF: usize;

    /// Representation size - size of the underlying data representation
    const REP_SIZE: usize;

    // Core group operations

    /// Get the identity element of the group.
    ///
    /// Returns the neutral element e such that e ∘ g = g ∘ e = g for any group element g.
    fn identity() -> Self::Element;

    /// Compute the inverse of this manifold element.
    ///
    /// For a group element g, returns g⁻¹ such that g ∘ g⁻¹ = e.
    ///
    /// # Arguments
    /// * `jacobian` - Optional mutable reference to store the Jacobian ∂(g⁻¹)/∂g
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::Element;

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
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element;

    // Exponential and logarithmic maps

    /// Exponential map from tangent space to manifold.
    ///
    /// Maps a tangent vector φ ∈ 𝔤 to the group element exp(φ^∧) ∈ G.
    ///
    /// # Arguments
    /// * `tangent` - Tangent vector in the Lie algebra
    /// * `jacobian` - Optional Jacobian ∂exp(φ^∧)/∂φ
    fn exp(
        tangent: &Self::TangentVector,
        jacobian: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element;

    /// Logarithmic map from manifold to tangent space.
    ///
    /// Maps a group element g ∈ G to its tangent vector log(g)^∨ ∈ 𝔤.
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian ∂log(g)^∨/∂g
    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector;

    // Manifold plus/minus operations

    /// Right plus operation: g ⊞ φ = g ∘ exp(φ^∧).
    ///
    /// Applies a tangent space perturbation to this manifold element.
    ///
    /// # Arguments  
    /// * `tangent` - Tangent vector perturbation
    /// * `jacobian_self` - Optional Jacobian ∂(g ⊞ φ)/∂g
    /// * `jacobian_tangent` - Optional Jacobian ∂(g ⊞ φ)/∂φ
    fn right_plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element;

    /// Right minus operation: g₁ ⊟ g₂ = log(g₂⁻¹ ∘ g₁)^∨.
    ///
    /// Computes the tangent vector that transforms g₂ to g₁.
    ///
    /// # Arguments
    /// * `other` - The reference element g₂
    /// * `jacobian_self` - Optional Jacobian ∂(g₁ ⊟ g₂)/∂g₁
    /// * `jacobian_other` - Optional Jacobian ∂(g₁ ⊟ g₂)/∂g₂
    fn right_minus(
        &self,
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::TangentVector;

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
    ) -> Self::Element;

    /// Left minus operation: g₁ ⊟ g₂ = log(g₁ ∘ g₂⁻¹)^∨.
    ///
    /// # Arguments
    /// * `other` - The reference element g₂
    /// * `jacobian_self` - Optional Jacobian ∂(g₁ ⊟ g₂)/∂g₁
    /// * `jacobian_other` - Optional Jacobian ∂(g₁ ⊟ g₂)/∂g₂  
    fn left_minus(
        &self,
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::TangentVector;

    // Convenience methods (use right operations by default)

    /// Convenience method for right_plus. Equivalent to g ⊞ φ.
    fn plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        self.right_plus(tangent, jacobian_self, jacobian_tangent)
    }

    /// Convenience method for right_minus. Equivalent to g₁ ⊟ g₂.
    fn minus(
        &self,
        other: &Self::Element,
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
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element;

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
    fn random() -> Self::Element;

    /// Normalize/project the element to the manifold.
    ///
    /// Ensures the element satisfies manifold constraints (e.g., orthogonality for rotations).
    fn normalize(&mut self);

    /// Check if the element is approximately on the manifold.
    fn is_valid(&self, tolerance: f64) -> bool;

    // Distance and norms

    /// Riemannian distance between two manifold elements.
    ///
    /// Computes ||log(g₁⁻¹ ∘ g₂)||.
    fn distance(&self, other: &Self::Element) -> f64;

    /// Weighted distance with a metric tensor.
    fn weighted_distance(&self, other: &Self::Element, weight: &Self::JacobianMatrix) -> f64;

    /// Test if this element is approximately equal to another.
    ///
    /// Uses the Riemannian distance with a tolerance threshold.
    fn is_approx(&self, other: &Self::Element, tolerance: f64) -> bool {
        self.distance(other) < tolerance
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
pub trait LieAlgebra<G: LieGroup>: Clone + Debug + PartialEq {
    // Dimension constants

    /// Dimension of the tangent space (same as Lie group DOF)
    const DIM: usize = G::DOF;

    /// Vector space operations
    ///
    /// Vector space addition: φ₁ + φ₂.
    ///
    /// # Arguments
    /// * `other` - The tangent vector to add
    fn add(&self, other: &G::TangentVector) -> G::TangentVector;

    /// Scalar multiplication: α · φ.
    ///
    /// # Arguments  
    /// * `scalar` - Scalar multiplier
    fn scale(&self, scalar: f64) -> G::TangentVector;

    /// Additive inverse: -φ.
    fn negate(&self) -> G::TangentVector;

    /// Vector subtraction: φ₁ - φ₂.
    fn subtract(&self, other: &G::TangentVector) -> G::TangentVector;

    // Norms and inner products

    /// Euclidean norm: ||φ||.
    fn norm(&self) -> f64;

    /// Squared norm: ||φ||².
    fn squared_norm(&self) -> f64;

    /// Weighted norm: √(φᵀ W φ).
    ///
    /// # Arguments
    /// * `weight` - Weight matrix W
    fn weighted_norm(&self, weight: &G::JacobianMatrix) -> f64;

    /// Squared weighted norm: φᵀ W φ.
    fn squared_weighted_norm(&self, weight: &G::JacobianMatrix) -> f64;

    /// Inner product: ⟨φ₁, φ₂⟩.
    ///
    /// # Arguments
    /// * `other` - The second tangent vector
    fn inner(&self, other: &G::TangentVector) -> f64;

    /// Weighted inner product: ⟨φ₁, W φ₂⟩.
    fn weighted_inner(&self, other: &G::TangentVector, weight: &G::JacobianMatrix) -> f64;

    // Exponential map and Jacobians

    /// Exponential map to Lie group: exp(φ^∧).
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian ∂exp(φ^∧)/∂φ
    fn exp(&self, jacobian: Option<&mut G::JacobianMatrix>) -> G::Element;

    /// Right Jacobian Jr.
    ///
    /// Matrix Jr such that for small δφ:
    /// exp((φ + δφ)^∧) ≈ exp(φ^∧) ∘ exp((Jr δφ)^∧)
    fn right_jacobian(&self) -> G::JacobianMatrix;

    /// Left Jacobian Jl.  
    ///
    /// Matrix Jl such that for small δφ:
    /// exp((φ + δφ)^∧) ≈ exp((Jl δφ)^∧) ∘ exp(φ^∧)
    fn left_jacobian(&self) -> G::JacobianMatrix;

    /// Inverse of right Jacobian Jr⁻¹.
    fn right_jacobian_inv(&self) -> G::JacobianMatrix;

    /// Inverse of left Jacobian Jl⁻¹.
    fn left_jacobian_inv(&self) -> G::JacobianMatrix;

    // Matrix representations

    /// Hat operator: φ^∧ (vector to matrix).
    ///
    /// Maps the tangent vector to its matrix representation in the Lie algebra.
    /// For SO(3): 3×1 vector → 3×3 skew-symmetric matrix
    /// For SE(3): 6×1 vector → 4×4 transformation matrix
    fn hat(&self) -> DMatrix<f64>;

    /// Vee operator: φ^∨ (matrix to vector).
    ///
    /// Inverse of the hat operator.
    fn vee(matrix: &DMatrix<f64>) -> ManifoldResult<G::TangentVector>;

    // Adjoint operations

    /// Small adjoint: ad(φ).
    ///
    /// The adjoint representation of the Lie algebra: ad(φ) ψ = [φ^∧, ψ^∧]^∨.
    fn small_adjoint(&self) -> G::JacobianMatrix;

    // Utility functions

    /// Zero tangent vector.
    fn zero() -> G::TangentVector;

    /// Random tangent vector (useful for testing).
    fn random() -> G::TangentVector;

    /// Check if the tangent vector is approximately zero.
    fn is_zero(&self, tolerance: f64) -> bool;

    /// Normalize the tangent vector to unit norm.
    fn normalize(&mut self);

    /// Return a unit tangent vector in the same direction.
    fn normalized(&self) -> G::TangentVector;
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
    fn interp(&self, other: &Self::Element, t: f64) -> Self::Element;

    /// Spherical linear interpolation (when applicable).
    fn slerp(&self, other: &Self::Element, t: f64) -> Self::Element;
}
