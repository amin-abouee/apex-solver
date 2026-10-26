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
//! ```text
//! Lie group M,° | size   | dim | X ∈ M                   | Constraint      | T_E M             | T_X M                 | Exp(T)             | Comp. | Action
//! ------------- | ------ | --- | ----------------------- | --------------- | ----------------- | --------------------- | ------------------ | ----- | ------
//! n-D vector    | Rⁿ,+   | n   | n   | v ∈ Rⁿ            | |v-v|=0         | v ∈ Rⁿ            | v ∈ Rⁿ                | v = exp(v)         | v₁+v₂ | v + x
//! Circle        | S¹,.   | 2   | 1   | z ∈ C             | z*z = 1         | iθ ∈ iR           | θ ∈ R                 | z = exp(iθ)        | z₁z₂  | zx
//! `Rotation      | SO(2),.| 4   | 1   | R                 | RᵀR = I         | [θ]x ∈ so(2)      | [θ] ∈ R²              | R = exp([θ]x)      | R₁R₂  | Rx`
//! `Rigid motion  | SE(2),.| 9   | 3   | M = [R t; 0 1]    | RᵀR = I         | [v̂] ∈ se(2)       | [v̂] ∈ R³              | Exp([v̂])           | M₁M₂  | Rx+t`
//! 3-sphere      | S³,.   | 4   | 3   | q ∈ H             | q*q = 1         | θ/2 ∈ Hp          | θ ∈ R³                | q = exp(uθ/2)      | q₁q₂  | qxq*
//! `Rotation      | SO(3),.| 9   | 3   | R                 | RᵀR = I         | [θ]x ∈ so(3)      | [θ] ∈ R³              | R = exp([θ]x)      | R₁R₂  | Rx`
//! `Rigid motion  | SE(3),.| 16  | 6   | M = [R t; 0 1]    | RᵀR = I         | [v̂] ∈ se(3)       | [v̂] ∈ R⁶              | Exp([v̂])           | M₁M₂  | Rx+t`
//! `Similarity    | Sim(3),.| 16 | 7   | M = [sR t; 0 1]   | RᵀR=I, s>0     | [v̂] ∈ sim(3)      | [ρ,θ,σ] ∈ R⁷          | Exp([v̂])           | M₁M₂  | sRx+t`
//! `Galilean      | SGal(3),.| 25| 10  | (R,t,v,s)         | RᵀR = I         | [v̂] ∈ sgal(3)     | [ρ,ν,θ,s] ∈ R¹⁰       | Exp([v̂])           | M₁M₂  | Rx+t+sv`
//! `Extended pose | SE_2(3),.| 25| 9   | (R,t,v)           | RᵀR = I         | [v̂] ∈ se_2_3      | [ρ,θ,ν] ∈ R⁹          | Exp([v̂])           | M₁M₂  | Rx+t`
//! ```
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
//! # Errors
//!
//! [`ManifoldError`] is the root error. Every variant transparently wraps one
//! per-concern enum, so `?` lifts a leaf error to the root and `source()` still
//! reaches the leaf:
//!
//! - [`ElementError`]: bad group element (wrong parameter count, non-unit quaternion,
//!   non-positive `Sim3` scale, NaN, zero rotation axis)
//! - [`TangentError`]: bad tangent slice or generator index
//! - [`RnError`]: `Rn` operands of different dimension
//! - [`ManifoldTypeError`]: unknown manifold name
//!
//! The group operations (`compose`, `inverse`, `act`, `exp`, `log`, the Jacobians,
//! `from_param_slice`) are infallible and never log, because the solver calls them
//! per residual per iteration. Input is validated where it enters, by the checked
//! constructors: [`LieGroup::try_from_param_slice`], [`Tangent::try_from_slice`],
//! `SO3::try_from_quaternion_wxyz`, `SO3::try_from_axis_angle`, `Sim3::try_new`.
//! These log the error once with `apex_error::ErrorLogging` and return it.
//!

use apex_error::ErrorLogging;
use nalgebra::{Matrix3, Vector3};
use std::ops::{Mul, Neg};

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

pub mod capability;
pub mod error;
pub mod rn;
pub mod se2;
pub mod se23;
pub mod se3;
pub mod sgal3;
pub mod sim3;
pub mod so2;
pub mod so3;
pub mod util;

pub use error::{
    ElementError, ElementResult, ManifoldError, ManifoldResult, ManifoldTypeError,
    ManifoldTypeResult, RnError, RnResult, TangentError, TangentResult,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl ManifoldType {
    /// The manifold type whose [`LieGroup::NAME`] is `name` (`"Rn"`, `"SE3"`, …),
    /// or `None` if the name does not match a known manifold.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Rn" => Some(Self::RN),
            "SE2" => Some(Self::SE2),
            "SE3" => Some(Self::SE3),
            "SE23" => Some(Self::SE23),
            "SGal3" => Some(Self::SGal3),
            "Sim3" => Some(Self::Sim3),
            "SO2" => Some(Self::SO2),
            "SO3" => Some(Self::SO3),
            _ => None,
        }
    }

    /// Like [`Self::from_name`], reporting the unknown name as an error.
    pub fn try_from_name(name: &str) -> ManifoldTypeResult<Self> {
        Self::from_name(name).ok_or_else(|| {
            ManifoldTypeError::UnknownManifold {
                name: name.to_string(),
            }
            .log()
        })
    }
}

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
    /// Human-readable name for serialization and logging.
    const NAME: &'static str;

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

    /// Identity Jacobian sized for *this element's* tangent space.
    ///
    /// [`Self::jacobian_identity`] is an associated function, so it cannot see
    /// the runtime dimension of a dynamically sized manifold — `Rn` has to
    /// guess, and guesses wrong for every dimension but one. Any operation that
    /// needs an identity block conformant with a concrete element must use this
    /// method instead.
    ///
    /// The default forwards to [`Self::jacobian_identity`], which is correct for
    /// every fixed-size group (SE2, SE3, SO2, SO3, Sim3, SE23, SGal3); `Rn`
    /// overrides it with its runtime dimension.
    fn jacobian_identity_for(&self) -> Self::JacobianMatrix {
        Self::jacobian_identity()
    }

    /// Zero Jacobian sized for *this element's* tangent space; the instance
    /// counterpart of [`Self::zero_jacobian`] (see [`Self::jacobian_identity_for`]).
    fn zero_jacobian_for(&self) -> Self::JacobianMatrix {
        Self::zero_jacobian()
    }

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

    /// Returns the manifold parameters as a contiguous flat slice.
    ///
    /// Enables zero-copy faer views: `faer::col::from_slice(g.as_param_slice())`.
    fn as_param_slice(&self) -> &[f64];

    /// Mutably borrows the raw parameter storage for in-place retraction updates.
    fn as_param_slice_mut(&mut self) -> &mut [f64];

    /// Constructs from a raw parameter slice (same layout as `as_param_slice`).
    ///
    /// Infallible and unchecked: `s` must hold exactly the group's parameter
    /// count (a debug build asserts it, a release build may panic inside
    /// nalgebra on a short slice) and describe a valid element. Use
    /// [`Self::try_from_param_slice`] for input that is not already trusted.
    fn from_param_slice(s: &[f64]) -> Self;

    /// Checks that `s` is a valid parameter block of this group: the right
    /// length, all finite, a unit quaternion where the group has one and a
    /// positive scale for `Sim3`. Does not log.
    fn validate_params(s: &[f64]) -> ElementResult<()>;

    /// Checked [`Self::from_param_slice`]: validates `s`, logs the error once
    /// and returns it instead of building a corrupt element.
    ///
    /// The quaternion tolerance is [`so3::SO3::QUATERNION_NORM_TOLERANCE`].
    #[track_caller]
    fn try_from_param_slice(s: &[f64]) -> ManifoldResult<Self> {
        match Self::validate_params(s) {
            Ok(()) => Ok(Self::from_param_slice(s)),
            Err(error) => Err(error.log().into()),
        }
    }

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
            *jac_self = result.right_jacobian_inv();
        }

        if let Some(jac_other) = jacobian_other {
            *jac_other = -result.left_jacobian_inv();
        }

        result
    }

    /// Left plus operation: φ ⊞ g = exp(φ^∧) ∘ g.
    ///
    /// # Arguments
    /// * `tangent` - Tangent vector perturbation
    /// * `jacobian_tangent` - Optional Jacobian ∂(φ ⊞ g)/∂φ
    /// * `jacobian_self` - Optional Jacobian ∂(φ ⊞ g)/∂g
    ///
    /// # Notes
    /// Inputs are perturbed on the right and the result is read in its own
    /// right-local coordinates — the convention [`Self::right_plus`],
    /// [`Self::compose`] and [`Self::between`] implement. Under it
    /// `∂(φ ⊞ g)/∂g = I` and `∂(φ ⊞ g)/∂φ = Ad(g)⁻¹·J_r(φ)`; both are
    /// checked against central differences in `tests/jacobian_identities.rs`.
    fn left_plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        let exp_tangent = tangent.exp(None);
        let result = exp_tangent.compose(self, None, None);

        if let Some(jac_self) = jacobian_self {
            // The result is `exp(φ)∘g`. Perturbing `g` on the right gives
            // `exp(φ)∘(g·exp(δ)) = (exp(φ)∘g)·exp(δ)`, so in the result's
            // right-local coordinates the Jacobian is the identity — the
            // very value `compose` hands back for its `other` operand,
            // which is what this computation bottoms out in. `Rn`'s
            // override agrees.
            //
            // It used to return `self.adjoint()`, which is the identity
            // under no convention: not `I` (right perturbation of the
            // input, right-local output — the convention `compose`,
            // `between` and `right_plus` implement, and the one
            // `jacobian_tangent` below already uses), and not `Ad(g)⁻¹`
            // either (left perturbation of the input).
            *jac_self = self.jacobian_identity_for();
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
            // Sized from `self`, not from the type: `other` shares this
            // element's tangent dimension, and for dynamically sized manifolds
            // the type alone cannot supply it.
            *jac_other = self.jacobian_identity_for();
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
    ///
    /// Total: an index outside the tangent space yields the zero matrix. Use
    /// [`Self::try_generator`] to get an error for it.
    fn generator(&self, i: usize) -> Group::LieAlgebra;

    /// Checked [`Self::generator`]: returns [`TangentError::GeneratorIndex`]
    /// (logged once) when `i` is not smaller than the tangent dimension.
    #[track_caller]
    fn try_generator(&self, i: usize) -> TangentResult<Group::LieAlgebra> {
        let dim = self.as_slice().len();
        if i >= dim {
            return Err(TangentError::GeneratorIndex {
                group: Group::NAME,
                index: i,
                dim,
            }
            .log());
        }
        Ok(self.generator(i))
    }

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

    /// Borrows the raw tangent data as a flat slice. Zero allocation.
    fn as_slice(&self) -> &[f64];

    /// Constructs from a raw slice (same layout as `as_slice`).
    ///
    /// Infallible and unchecked: `s` must hold exactly [`Self::DIM`] values
    /// (any count for the dynamic `RnTangent`). Use [`Self::try_from_slice`] for
    /// input that is not already trusted.
    fn from_slice(s: &[f64]) -> Self;

    /// Checked [`Self::from_slice`]: rejects a wrong length or a NaN/infinite
    /// value, logging the error once.
    #[track_caller]
    fn try_from_slice(s: &[f64]) -> TangentResult<Self> {
        if !Self::is_dynamic() && s.len() != Self::DIM {
            return Err(TangentError::TangentLength {
                group: Group::NAME,
                expected: Self::DIM,
                actual: s.len(),
            }
            .log());
        }
        if s.iter().any(|value| !value.is_finite()) {
            return Err(TangentError::NonFinite { group: Group::NAME }.log());
        }
        Ok(Self::from_slice(s))
    }
}

/// Trait for Lie groups that support interpolation.
pub trait Interpolatable: LieGroup {
    /// Linear interpolation in the manifold.
    ///
    /// For parameter t ∈ `[0,1]`: interp(g₁, g₂, 0) = g₁, interp(g₁, g₂, 1) = g₂.
    ///
    /// # Arguments
    /// * `other` - Target element for interpolation
    /// * `t` - Interpolation parameter in `[0,1]`
    fn interp(&self, other: &Self, t: f64) -> Self;

    /// Spherical linear interpolation (when applicable).
    fn slerp(&self, other: &Self, t: f64) -> Self;
}

#[cfg(test)]
mod tests {
    use crate::LieGroup;
    use crate::Tangent;
    use crate::rn::{Rn, RnTangent};
    use crate::se2::SE2;
    use crate::se3::SE3;
    use crate::se23::SE23;
    use crate::sgal3::SGal3;
    use crate::sim3::Sim3;
    use crate::so2::{SO2, SO2Tangent};
    use crate::so3::{SO3, SO3Tangent};
    use crate::{ElementError, ManifoldError, ManifoldType, ManifoldTypeError, TangentError};
    use nalgebra::Matrix1;

    fn make_so2(angle: f64) -> SO2 {
        SO2::from_angle(angle)
    }

    fn make_so2_tangent(angle: f64) -> SO2Tangent {
        SO2Tangent::new(angle)
    }

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn param_length_error(group: &'static str, expected: usize, actual: usize) -> ManifoldError {
        ElementError::ParamLength {
            group,
            expected,
            actual,
        }
        .into()
    }

    /// Round trip, wrong length and NaN on a fixed-size group.
    fn check_checked_param_slice<G: LieGroup>() -> TestResult {
        let group = G::random();
        let params = group.as_param_slice().to_vec();
        let len = params.len();

        let back = G::try_from_param_slice(&params)?;
        assert!(group.is_approx(&back, 1e-12));

        assert_eq!(
            G::try_from_param_slice(&params[..len - 1]).err(),
            Some(param_length_error(G::NAME, len, len - 1))
        );
        let mut longer = params.clone();
        longer.push(0.0);
        assert_eq!(
            G::try_from_param_slice(&longer).err(),
            Some(param_length_error(G::NAME, len, len + 1))
        );
        assert_eq!(
            G::try_from_param_slice(&[]).err(),
            Some(param_length_error(G::NAME, len, 0))
        );

        let mut nan = params.clone();
        nan[0] = f64::NAN;
        assert_eq!(
            G::try_from_param_slice(&nan).err(),
            Some(ElementError::NonFinite { group: G::NAME }.into())
        );
        let mut inf = params;
        inf[len - 1] = f64::INFINITY;
        assert!(G::try_from_param_slice(&inf).is_err());
        Ok(())
    }

    /// A quaternion starting at `at` that is scaled off unit norm is refused.
    fn check_non_unit_quaternion<G: LieGroup>(at: usize) {
        let mut params = G::random().as_param_slice().to_vec();
        for component in &mut params[at..at + 4] {
            *component *= 2.0;
        }
        assert!(matches!(
            G::try_from_param_slice(&params).err(),
            Some(ManifoldError::Element(ElementError::NonUnitQuaternion { norm }))
                if (norm - 2.0).abs() < 1e-12
        ));
    }

    #[test]
    fn try_from_param_slice_checks_every_fixed_size_group() -> TestResult {
        check_checked_param_slice::<SO2>()?;
        check_checked_param_slice::<SE2>()?;
        check_checked_param_slice::<SO3>()?;
        check_checked_param_slice::<SE3>()?;
        check_checked_param_slice::<SE23>()?;
        check_checked_param_slice::<SGal3>()?;
        check_checked_param_slice::<Sim3>()?;
        Ok(())
    }

    #[test]
    fn try_from_param_slice_rejects_a_non_unit_quaternion_in_every_group() {
        check_non_unit_quaternion::<SO3>(0);
        check_non_unit_quaternion::<SE3>(3);
        check_non_unit_quaternion::<SE23>(3);
        check_non_unit_quaternion::<SGal3>(3);
        check_non_unit_quaternion::<Sim3>(3);
    }

    #[test]
    fn rn_try_from_param_slice_accepts_any_length_but_not_nan() -> TestResult {
        for dim in [0, 1, 3, 7] {
            let params = vec![0.5; dim];
            assert_eq!(Rn::try_from_param_slice(&params)?.as_param_slice(), params);
        }
        assert_eq!(
            Rn::try_from_param_slice(&[1.0, f64::NAN]).err(),
            Some(ElementError::NonFinite { group: "Rn" }.into())
        );
        Ok(())
    }

    /// `try_from_slice` on a fixed tangent: round trip, wrong length, NaN.
    fn check_checked_tangent_slice<G: LieGroup>() -> TestResult {
        let tangent = G::TangentVector::random();
        let values = tangent.as_slice().to_vec();
        let len = values.len();

        assert!(tangent.is_approx(&G::TangentVector::try_from_slice(&values)?, 1e-12));
        assert_eq!(
            G::TangentVector::try_from_slice(&values[..len - 1]).err(),
            Some(TangentError::TangentLength {
                group: G::NAME,
                expected: len,
                actual: len - 1
            })
        );
        let mut nan = values;
        nan[0] = f64::NAN;
        assert_eq!(
            G::TangentVector::try_from_slice(&nan).err(),
            Some(TangentError::NonFinite { group: G::NAME })
        );
        Ok(())
    }

    #[test]
    fn tangent_try_from_slice_checks_every_fixed_size_group() -> TestResult {
        check_checked_tangent_slice::<SO2>()?;
        check_checked_tangent_slice::<SE2>()?;
        check_checked_tangent_slice::<SO3>()?;
        check_checked_tangent_slice::<SE3>()?;
        check_checked_tangent_slice::<SE23>()?;
        check_checked_tangent_slice::<SGal3>()?;
        check_checked_tangent_slice::<Sim3>()?;
        Ok(())
    }

    #[test]
    fn rn_tangent_try_from_slice_accepts_any_length() -> TestResult {
        assert_eq!(
            RnTangent::try_from_slice(&[1.0, 2.0])?.as_slice(),
            [1.0, 2.0]
        );
        assert_eq!(
            RnTangent::try_from_slice(&[f64::INFINITY]).err(),
            Some(TangentError::NonFinite { group: "Rn" })
        );
        Ok(())
    }

    #[test]
    fn a_checked_constructor_error_reaches_the_root_with_a_one_line_chain() -> TestResult {
        use apex_error::ErrorChainExt;
        let error = SE3::try_from_param_slice(&[0.0; 3])
            .err()
            .ok_or("expected an error")?;
        assert_eq!(
            error.chain(),
            "SE3 parameter block has 3 values, expected 7"
        );
        Ok(())
    }

    #[test]
    fn try_from_name_resolves_every_group_name() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(ManifoldType::try_from_name("Rn")?, ManifoldType::RN);
        assert_eq!(ManifoldType::try_from_name("SE3")?, ManifoldType::SE3);
        assert_eq!(ManifoldType::try_from_name("Sim3")?, ManifoldType::Sim3);
        Ok(())
    }

    #[test]
    fn try_from_name_reports_an_unknown_name() {
        let error = ManifoldType::try_from_name("SE4");
        assert_eq!(
            error,
            Err(ManifoldTypeError::UnknownManifold {
                name: "SE4".to_string()
            })
        );
        assert_eq!(ManifoldType::from_name("SE4"), None);
        let root = error.map_err(ManifoldError::from);
        assert!(matches!(root, Err(ManifoldError::ManifoldType(_))));
    }

    #[test]
    fn manifold_type_variants_are_distinct() {
        let types = [
            ManifoldType::RN,
            ManifoldType::SE2,
            ManifoldType::SE3,
            ManifoldType::SE23,
            ManifoldType::SGal3,
            ManifoldType::Sim3,
            ManifoldType::SO2,
            ManifoldType::SO3,
        ];
        assert_eq!(types.len(), 8);
        assert_eq!(ManifoldType::SO3, ManifoldType::SO3);
        assert_ne!(ManifoldType::SO2, ManifoldType::SO3);
    }

    #[test]
    fn default_right_plus_no_jacobians() {
        let g = make_so2(0.3);
        let t = make_so2_tangent(0.1);
        let result = g.right_plus(&t, None, None);
        assert!(result.is_valid(1e-9));
    }

    #[test]
    fn default_right_plus_with_jacobians() {
        let g = make_so2(0.3);
        let t = make_so2_tangent(0.1);
        let mut j_self = Matrix1::zeros();
        let mut j_tan = Matrix1::zeros();
        let result = g.right_plus(&t, Some(&mut j_self), Some(&mut j_tan));
        assert!(result.is_valid(1e-9));
        assert!(j_tan[0].is_finite());
    }

    #[test]
    fn default_right_minus_no_jacobians() {
        let g1 = make_so2(0.5);
        let g2 = make_so2(0.2);
        let _t = g1.right_minus(&g2, None, None);
    }

    #[test]
    fn default_right_minus_with_jacobians() {
        let g1 = make_so2(0.5);
        let g2 = make_so2(0.2);
        let mut j_self = Matrix1::zeros();
        let mut j_other = Matrix1::zeros();
        let _t = g1.right_minus(&g2, Some(&mut j_self), Some(&mut j_other));
        assert!(j_self[0].is_finite());
        assert!(j_other[0].is_finite());
    }

    #[test]
    fn default_left_plus_no_jacobians() {
        let g = make_so2(0.3);
        let t = make_so2_tangent(0.1);
        let result = g.left_plus(&t, None, None);
        assert!(result.is_valid(1e-9));
    }

    #[test]
    fn default_left_plus_with_jacobians() {
        let g = make_so2(0.3);
        let t = make_so2_tangent(0.1);
        let mut j_tan = Matrix1::zeros();
        let mut j_self = Matrix1::zeros();
        let result = g.left_plus(&t, Some(&mut j_tan), Some(&mut j_self));
        assert!(result.is_valid(1e-9));
        assert!(j_tan[0].is_finite());
        assert!(j_self[0].is_finite());
    }

    #[test]
    fn default_left_minus_no_jacobians() {
        let g1 = make_so2(0.5);
        let g2 = make_so2(0.2);
        let _t = g1.left_minus(&g2, None, None);
    }

    #[test]
    fn default_left_minus_with_jacobians() {
        let g1 = make_so2(0.5);
        let g2 = make_so2(0.2);
        let mut j_self = Matrix1::zeros();
        let mut j_other = Matrix1::zeros();
        let _t = g1.left_minus(&g2, Some(&mut j_self), Some(&mut j_other));
        assert!(j_self[0].is_finite());
        assert!(j_other[0].is_finite());
    }

    #[test]
    fn default_plus_delegates_to_right_plus() {
        let g = make_so2(0.3);
        let t = make_so2_tangent(0.1);
        let r1 = g.plus(&t, None, None);
        let r2 = g.right_plus(&t, None, None);
        assert!(r1.is_approx(&r2, 1e-9));
    }

    #[test]
    fn default_minus_delegates_to_right_minus() {
        let g1 = make_so2(0.5);
        let g2 = make_so2(0.2);
        let t1 = g1.minus(&g2, None, None);
        let t2 = g1.right_minus(&g2, None, None);
        assert!(t1.is_approx(&t2, 1e-9));
    }

    #[test]
    fn default_between_no_jacobians() {
        let g1 = make_so2(0.3);
        let g2 = make_so2(0.7);
        let b = g1.between(&g2, None, None);
        assert!(b.is_valid(1e-9));
    }

    #[test]
    fn default_between_with_jacobians() {
        let g1 = make_so2(0.3);
        let g2 = make_so2(0.7);
        let mut j_self = Matrix1::zeros();
        let mut j_other = Matrix1::zeros();
        let b = g1.between(&g2, Some(&mut j_self), Some(&mut j_other));
        assert!(b.is_valid(1e-9));
        assert!(j_self[0].is_finite());
        assert!(j_other[0].is_finite());
    }

    #[test]
    fn default_tangent_dim_returns_dof() {
        let g = make_so2(0.0);
        assert_eq!(g.tangent_dim(), 1); // SO2 has 1 DOF
    }

    #[test]
    fn tangent_is_dynamic_false_for_so2() {
        assert!(!SO2Tangent::is_dynamic());
    }

    #[test]
    fn manifold_type_clone_and_eq() {
        let all_types = [
            ManifoldType::RN,
            ManifoldType::SE2,
            ManifoldType::SE3,
            ManifoldType::SE23,
            ManifoldType::SGal3,
            ManifoldType::Sim3,
            ManifoldType::SO2,
            ManifoldType::SO3,
        ];
        for t in &all_types {
            let t2 = *t;
            assert_eq!(t, &t2);
        }
        // Ensure different variants are not equal
        assert_ne!(ManifoldType::RN, ManifoldType::SE2);
        assert_ne!(ManifoldType::SE2, ManifoldType::SE3);
        assert_ne!(ManifoldType::SE3, ManifoldType::SE23);
        assert_ne!(ManifoldType::SE23, ManifoldType::SGal3);
        assert_ne!(ManifoldType::SGal3, ManifoldType::Sim3);
        assert_ne!(ManifoldType::Sim3, ManifoldType::SO2);
        assert_ne!(ManifoldType::SO2, ManifoldType::SO3);
    }

    #[test]
    fn manifold_type_debug() {
        let s = format!("{:?}", ManifoldType::RN);
        assert!(!s.is_empty());
        let s2 = format!("{:?}", ManifoldType::SE23);
        assert!(!s2.is_empty());
        let s3 = format!("{:?}", ManifoldType::SGal3);
        assert!(!s3.is_empty());
        let s4 = format!("{:?}", ManifoldType::Sim3);
        assert!(!s4.is_empty());
    }

    // Test SO3-based defaults to exercise more code paths

    #[test]
    fn so3_default_right_plus_with_jacobians() {
        use crate::LieGroup;
        let r = SO3::from_euler_angles(0.1, 0.2, 0.3);
        let t = SO3Tangent::new(nalgebra::Vector3::new(0.05, 0.0, 0.0));
        let mut j_self = nalgebra::Matrix3::zeros();
        let mut j_tan = nalgebra::Matrix3::zeros();
        let result = r.right_plus(&t, Some(&mut j_self), Some(&mut j_tan));
        assert!(result.is_valid(1e-6));
        assert!(j_self[(0, 0)].is_finite());
        assert!(j_tan[(0, 0)].is_finite());
    }

    #[test]
    fn so3_default_left_plus_with_jacobians() {
        use crate::LieGroup;
        let r = SO3::from_euler_angles(0.1, 0.2, 0.3);
        let t = SO3Tangent::new(nalgebra::Vector3::new(0.05, 0.0, 0.0));
        let mut j_tan = nalgebra::Matrix3::zeros();
        let mut j_self = nalgebra::Matrix3::zeros();
        let result = r.left_plus(&t, Some(&mut j_tan), Some(&mut j_self));
        assert!(result.is_valid(1e-6));
        assert!(j_tan[(0, 0)].is_finite());
        assert!(j_self[(0, 0)].is_finite());
    }

    #[test]
    fn so3_default_right_minus_with_jacobians() {
        use crate::LieGroup;
        let r1 = SO3::from_euler_angles(0.3, 0.1, 0.2);
        let r2 = SO3::from_euler_angles(0.1, 0.0, 0.1);
        let mut j_self = nalgebra::Matrix3::zeros();
        let mut j_other = nalgebra::Matrix3::zeros();
        let _t = r1.right_minus(&r2, Some(&mut j_self), Some(&mut j_other));
        assert!(j_self[(0, 0)].is_finite());
        assert!(j_other[(0, 0)].is_finite());
    }

    #[test]
    fn so3_default_left_minus_with_jacobians() {
        use crate::LieGroup;
        let r1 = SO3::from_euler_angles(0.3, 0.1, 0.2);
        let r2 = SO3::from_euler_angles(0.1, 0.0, 0.1);
        let mut j_self = nalgebra::Matrix3::zeros();
        let mut j_other = nalgebra::Matrix3::zeros();
        let _t = r1.left_minus(&r2, Some(&mut j_self), Some(&mut j_other));
        assert!(j_self[(0, 0)].is_finite());
        assert!(j_other[(0, 0)].is_finite());
    }

    #[test]
    fn so3_default_between_with_jacobians() {
        use crate::LieGroup;
        let r1 = SO3::from_euler_angles(0.1, 0.2, 0.3);
        let r2 = SO3::from_euler_angles(0.4, 0.1, 0.2);
        let mut j_self = nalgebra::Matrix3::zeros();
        let mut j_other = nalgebra::Matrix3::zeros();
        let b = r1.between(&r2, Some(&mut j_self), Some(&mut j_other));
        assert!(b.is_valid(1e-6));
        assert!(j_self[(0, 0)].is_finite());
        assert!(j_other[(0, 0)].is_finite());
    }

    #[test]
    fn so3_default_tangent_dim() {
        use crate::LieGroup;
        let r = SO3::identity();
        assert_eq!(r.tangent_dim(), 3);
    }
}
