//! Rn - n-dimensional Euclidean Space
//!
//! This module implements the n-dimensional Euclidean space Rⁿ with vector addition
//! as the group operation.
//!
//! Rⁿ elements are represented using nalgebra's nalgebra::DVector<f64> for dynamic sizing.
//! Rⁿ tangent elements are also represented as nalgebra::DVector<f64> since the tangent space
//! is isomorphic to the manifold itself.
//!
//! The implementation follows the [manif](https://github.com/artivis/manif) C++ library
//! conventions and provides all operations required by the LieGroup and Tangent traits.

use crate::manifold;
use crate::manifold::Tangent;
use nalgebra;
use std::fmt;

/// Rⁿ group element representing n-dimensional Euclidean vectors.
///
/// Internally represented using nalgebra's nalgebra::DVector<f64> for dynamic sizing.
#[derive(Clone, Debug, PartialEq)]
pub struct Rn {
    /// Internal representation as a dynamic vector
    data: nalgebra::DVector<f64>,
}

impl fmt::Display for Rn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rn(dim: {}, data: [", self.data.len())?;
        for (i, val) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", val)?;
        }
        write!(f, "])")
    }
}

// Conversion traits for integration with generic Problem
impl From<nalgebra::DVector<f64>> for Rn {
    fn from(data: nalgebra::DVector<f64>) -> Self {
        Rn::new(data)
    }
}

impl From<Rn> for nalgebra::DVector<f64> {
    fn from(rn: Rn) -> Self {
        rn.data
    }
}

/// Rⁿ tangent space element representing elements in the Lie algebra rⁿ.
///
/// For Euclidean space, the tangent space is isomorphic to the manifold itself,
/// so this is also represented as a nalgebra::DVector<f64>.
#[derive(Clone, Debug, PartialEq)]
pub struct RnTangent {
    /// Internal data: n-dimensional vector
    data: nalgebra::DVector<f64>,
}

impl fmt::Display for RnTangent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rn(dim: {}, data: [", self.data.len())?;
        for (i, val) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", val)?;
        }
        write!(f, "])")
    }
}

impl Rn {
    /// Space dimension - dimension of the ambient space that the group acts on
    /// Note: For Rⁿ this is dynamic and determined at runtime
    pub const DIM: usize = 0;

    /// Degrees of freedom - dimension of the tangent space
    /// Note: For Rⁿ this is dynamic and determined at runtime
    pub const DOF: usize = 0;

    /// Representation size - size of the underlying data representation
    /// Note: For Rⁿ this is dynamic and determined at runtime
    pub const REP_SIZE: usize = 0;

    /// Get the identity element of the group.
    ///
    /// Returns the neutral element e such that e ∘ g = g ∘ e = g for any group element g.
    /// Note: Default to 3D identity for compatibility, but this should be created with specific dimension
    pub fn identity() -> Self {
        Rn::new(nalgebra::DVector::zeros(3))
    }

    /// Get the identity matrix for Jacobians.
    ///
    /// Returns the identity matrix in the appropriate dimension for Jacobian computations.
    /// Note: Default to 3x3 identity, but this should be created with specific dimension
    pub fn jacobian_identity() -> nalgebra::DMatrix<f64> {
        nalgebra::DMatrix::identity(3, 3)
    }

    /// Create a new Rⁿ element from a vector.
    ///
    /// # Arguments
    /// * `data` - Vector data
    pub fn new(data: nalgebra::DVector<f64>) -> Self {
        Rn { data }
    }

    /// Create Rⁿ from a slice.
    ///
    /// # Arguments
    /// * `slice` - Data slice
    pub fn from_slice(slice: &[f64]) -> Self {
        Rn::new(nalgebra::DVector::from_row_slice(slice))
    }

    /// Create Rⁿ from individual components (up to 6D for convenience).
    pub fn from_vec(components: Vec<f64>) -> Self {
        Rn::new(nalgebra::DVector::from_vec(components))
    }

    /// Get the underlying vector.
    pub fn data(&self) -> &nalgebra::DVector<f64> {
        &self.data
    }

    /// Get the dimension of the space.
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Get a specific component.
    pub fn component(&self, index: usize) -> f64 {
        self.data[index]
    }

    /// Set a specific component.
    pub fn set_component(&mut self, index: usize, value: f64) {
        self.data[index] = value;
    }

    /// Get the norm (Euclidean length) of the vector.
    pub fn norm(&self) -> f64 {
        self.data.norm()
    }

    /// Get the squared norm of the vector.
    pub fn norm_squared(&self) -> f64 {
        self.data.norm_squared()
    }

    /// Convert Rn to a nalgebra::DVector
    pub fn to_vector(&self) -> nalgebra::DVector<f64> {
        self.data.clone()
    }
}

impl manifold::LieGroup for Rn {
    type TangentVector = RnTangent;
    type JacobianMatrix = nalgebra::DMatrix<f64>;
    type LieAlgebra = nalgebra::DMatrix<f64>;

    /// Rⁿ inverse (negation for additive group).
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian matrix of the inverse wrt self.
    ///
    /// # Notes
    /// For Euclidean space with addition: -v
    /// Jacobian of inverse: d(-v)/dv = -I
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self {
        let dim = self.data.len();
        if let Some(jac) = jacobian {
            *jac = -nalgebra::DMatrix::identity(dim, dim);
        }
        Rn::new(-&self.data)
    }

    /// Rⁿ composition (vector addition).
    ///
    /// # Arguments
    /// * `other` - Another Rⁿ element.
    /// * `jacobian_self` - Optional Jacobian matrix of the composition wrt self.
    /// * `jacobian_other` - Optional Jacobian matrix of the composition wrt other.
    ///
    /// # Notes
    /// For Euclidean space: v₁ + v₂
    /// Jacobians: d(v₁ + v₂)/dv₁ = I, d(v₁ + v₂)/dv₂ = I
    fn compose(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Rn elements must have the same dimension for composition"
        );

        let dim = self.data.len();
        if let Some(jac_self) = jacobian_self {
            *jac_self = nalgebra::DMatrix::identity(dim, dim);
        }
        if let Some(jac_other) = jacobian_other {
            *jac_other = nalgebra::DMatrix::identity(dim, dim);
        }

        Rn::new(&self.data + &other.data)
    }

    /// Logarithmic map from manifold to tangent space.
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian matrix of the tangent wrt to self.
    ///
    /// # Notes
    /// For Euclidean space, log is identity: log(v) = v
    /// Jacobian: dlog(v)/dv = I
    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector {
        let dim = self.data.len();
        if let Some(jac) = jacobian {
            *jac = nalgebra::DMatrix::identity(dim, dim);
        }
        RnTangent::new(self.data.clone())
    }

    /// Vee operator: log(g)^∨.
    ///
    /// For Euclidean space, this is the same as log().
    fn vee(&self) -> Self::TangentVector {
        self.log(None)
    }

    /// Action on a 3-vector (for compatibility with the trait).
    ///
    /// # Arguments
    /// * `vector` - A 3-vector.
    /// * `jacobian_self` - Optional Jacobian of the new object wrt this.
    /// * `jacobian_vector` - Optional Jacobian of the new object wrt input object.
    ///
    /// # Returns
    /// The transformed 3-vector.
    ///
    /// # Notes
    /// For Euclidean space, the action is translation: v + x
    /// This only works if this Rⁿ element is 3-dimensional.
    fn act(
        &self,
        vector: &nalgebra::Vector3<f64>,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_vector: Option<&mut nalgebra::Matrix3<f64>>,
    ) -> nalgebra::Vector3<f64> {
        if self.data.len() != 3 {
            panic!(
                "act() requires 3-dimensional Rⁿ element, got dimension {}",
                self.data.len()
            );
        }

        if let Some(jac_self) = jacobian_self {
            *jac_self = nalgebra::DMatrix::identity(3, 3);
        }
        if let Some(jac_vector) = jacobian_vector {
            *jac_vector = nalgebra::Matrix3::identity();
        }

        nalgebra::Vector3::new(
            self.data[0] + vector.x,
            self.data[1] + vector.y,
            self.data[2] + vector.z,
        )
    }

    /// Get the adjoint matrix of Rⁿ.
    ///
    /// # Notes
    /// For Euclidean space (abelian group), adjoint is identity.
    fn adjoint(&self) -> Self::JacobianMatrix {
        let dim = self.data.len();
        nalgebra::DMatrix::identity(dim, dim)
    }

    /// Generate a random element.
    fn random() -> Self {
        // Default to 3D random vector
        let data = nalgebra::DVector::from_fn(3, |_, _| rand::random::<f64>() * 10.0 - 5.0);
        Rn::new(data)
    }

    /// Normalize the vector (no-op for Euclidean space, but could normalize to unit length).
    fn normalize(&mut self) {
        // For Euclidean space, we could normalize to unit length
        let norm = self.data.norm();
        if norm > 1e-12 {
            self.data /= norm;
        }
    }

    /// Check if the element is valid (always true for Euclidean space).
    fn is_valid(&self, _tolerance: f64) -> bool {
        // All finite vectors are valid in Euclidean space
        self.data.iter().all(|x| x.is_finite())
    }

    /// Check if the element is approximately equal to another element.
    ///
    /// # Arguments
    /// * `other` - The other element to compare with
    /// * `tolerance` - The tolerance for the comparison
    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        if self.data.len() != other.data.len() {
            return false;
        }
        let difference = self.right_minus(other, None, None);
        difference.is_zero(tolerance)
    }

    // Explicit implementations of plus/minus operations for optimal performance
    // These override the default LieGroup implementations to provide correct
    // Jacobians for Euclidean space (simple identity matrices)

    /// Right plus operation: v ⊞ δ = v + δ.
    ///
    /// For Euclidean space, this is simple vector addition.
    ///
    /// # Arguments
    /// * `tangent` - Tangent vector perturbation
    /// * `jacobian_self` - Optional Jacobian ∂(v ⊞ δ)/∂v = I
    /// * `jacobian_tangent` - Optional Jacobian ∂(v ⊞ δ)/∂δ = I
    ///
    /// # Notes
    /// For Euclidean space: v ⊞ δ = v + δ
    /// Jacobians: ∂(v + δ)/∂v = I, ∂(v + δ)/∂δ = I
    fn right_plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        assert_eq!(
            self.data.len(),
            tangent.data.len(),
            "Rn element and tangent must have the same dimension"
        );

        let dim = self.data.len();

        if let Some(jac_self) = jacobian_self {
            *jac_self = nalgebra::DMatrix::identity(dim, dim);
        }

        if let Some(jac_tangent) = jacobian_tangent {
            *jac_tangent = nalgebra::DMatrix::identity(dim, dim);
        }

        Rn::new(&self.data + &tangent.data)
    }

    /// Right minus operation: v₁ ⊟ v₂ = v₁ - v₂.
    ///
    /// For Euclidean space, this is simple vector subtraction.
    ///
    /// # Arguments
    /// * `other` - The reference element v₂
    /// * `jacobian_self` - Optional Jacobian ∂(v₁ ⊟ v₂)/∂v₁ = I
    /// * `jacobian_other` - Optional Jacobian ∂(v₁ ⊟ v₂)/∂v₂ = -I
    ///
    /// # Notes
    /// For Euclidean space: v₁ ⊟ v₂ = v₁ - v₂
    /// Jacobians: ∂(v₁ - v₂)/∂v₁ = I, ∂(v₁ - v₂)/∂v₂ = -I
    fn right_minus(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::TangentVector {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Rn elements must have the same dimension"
        );

        let dim = self.data.len();

        if let Some(jac_self) = jacobian_self {
            *jac_self = nalgebra::DMatrix::identity(dim, dim);
        }

        if let Some(jac_other) = jacobian_other {
            *jac_other = -nalgebra::DMatrix::identity(dim, dim);
        }

        RnTangent::new(&self.data - &other.data)
    }

    /// Left plus operation: δ ⊞ v = δ + v.
    ///
    /// For Euclidean space (abelian group), left plus is the same as right plus.
    ///
    /// # Arguments
    /// * `tangent` - Tangent vector perturbation
    /// * `jacobian_tangent` - Optional Jacobian ∂(δ ⊞ v)/∂δ = I
    /// * `jacobian_self` - Optional Jacobian ∂(δ ⊞ v)/∂v = I
    ///
    /// # Notes
    /// For abelian groups: δ ⊞ v = v ⊞ δ = δ + v
    fn left_plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        assert_eq!(
            self.data.len(),
            tangent.data.len(),
            "Rn element and tangent must have the same dimension"
        );

        let dim = self.data.len();

        if let Some(jac_tangent) = jacobian_tangent {
            *jac_tangent = nalgebra::DMatrix::identity(dim, dim);
        }

        if let Some(jac_self) = jacobian_self {
            *jac_self = nalgebra::DMatrix::identity(dim, dim);
        }

        // For abelian groups, left plus is the same as right plus
        Rn::new(&tangent.data + &self.data)
    }

    /// Left minus operation: v₁ ⊟ v₂ = v₁ - v₂.
    ///
    /// For Euclidean space (abelian group), left minus is the same as right minus.
    ///
    /// # Arguments
    /// * `other` - The reference element v₂
    /// * `jacobian_self` - Optional Jacobian ∂(v₁ ⊟ v₂)/∂v₁ = I
    /// * `jacobian_other` - Optional Jacobian ∂(v₁ ⊟ v₂)/∂v₂ = -I
    ///
    /// # Notes
    /// For abelian groups: left minus = right minus
    fn left_minus(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::TangentVector {
        // For abelian groups, left minus is the same as right minus
        self.right_minus(other, jacobian_self, jacobian_other)
    }

    /// Get the dimension of the tangent space for this Rⁿ element.
    ///
    /// # Returns
    /// The actual runtime dimension of this Rⁿ element.
    ///
    /// # Notes
    /// Overrides the default implementation to return the dynamic size
    /// based on the actual data vector length, since Rⁿ has variable dimension.
    fn tangent_dim(&self) -> usize {
        self.data.len()
    }
}

impl RnTangent {
    /// Create a new RnTangent from a vector.
    ///
    /// # Arguments
    /// * `data` - Vector data
    pub fn new(data: nalgebra::DVector<f64>) -> Self {
        RnTangent { data }
    }

    /// Create RnTangent from a slice.
    ///
    /// # Arguments
    /// * `slice` - Data slice
    pub fn from_slice(slice: &[f64]) -> Self {
        RnTangent::new(nalgebra::DVector::from_row_slice(slice))
    }

    /// Create RnTangent from individual components.
    pub fn from_vec(components: Vec<f64>) -> Self {
        RnTangent::new(nalgebra::DVector::from_vec(components))
    }

    /// Get the underlying vector.
    pub fn data(&self) -> &nalgebra::DVector<f64> {
        &self.data
    }

    /// Get the dimension of the tangent space.
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Get a specific component.
    pub fn component(&self, index: usize) -> f64 {
        self.data[index]
    }

    /// Set a specific component.
    pub fn set_component(&mut self, index: usize, value: f64) {
        self.data[index] = value;
    }

    /// Get the norm (Euclidean length) of the tangent vector.
    pub fn norm(&self) -> f64 {
        self.data.norm()
    }

    /// Get the squared norm of the tangent vector.
    pub fn norm_squared(&self) -> f64 {
        self.data.norm_squared()
    }

    /// Convert RnTangent to a nalgebra::DVector
    pub fn to_vector(&self) -> nalgebra::DVector<f64> {
        self.data.clone()
    }

    /// Create RnTangent from a nalgebra::DVector
    pub fn from_vector(data: nalgebra::DVector<f64>) -> Self {
        RnTangent::new(data)
    }
}

impl manifold::Tangent<Rn> for RnTangent {
    /// Dimension of the tangent space
    /// Note: For Rⁿ this is dynamic and determined at runtime
    const DIM: usize = 0;

    /// Exponential map for Euclidean space (identity).
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian matrix of the Rn element wrt this.
    ///
    /// # Notes
    /// For Euclidean space: exp(v) = v
    /// Jacobian: dexp(v)/dv = I
    fn exp(&self, jacobian: Option<&mut <Rn as manifold::LieGroup>::JacobianMatrix>) -> Rn {
        let dim = self.data.len();
        if let Some(jac) = jacobian {
            *jac = nalgebra::DMatrix::identity(dim, dim);
        }
        Rn::new(self.data.clone())
    }

    /// Right Jacobian for Euclidean space (identity).
    fn right_jacobian(&self) -> <Rn as manifold::LieGroup>::JacobianMatrix {
        let dim = self.data.len();
        nalgebra::DMatrix::identity(dim, dim)
    }

    /// Left Jacobian for Euclidean space (identity).
    fn left_jacobian(&self) -> <Rn as manifold::LieGroup>::JacobianMatrix {
        let dim = self.data.len();
        nalgebra::DMatrix::identity(dim, dim)
    }

    /// Inverse of right Jacobian for Euclidean space (identity).
    fn right_jacobian_inv(&self) -> <Rn as manifold::LieGroup>::JacobianMatrix {
        let dim = self.data.len();
        nalgebra::DMatrix::identity(dim, dim)
    }

    /// Inverse of left Jacobian for Euclidean space (identity).
    fn left_jacobian_inv(&self) -> <Rn as manifold::LieGroup>::JacobianMatrix {
        let dim = self.data.len();
        nalgebra::DMatrix::identity(dim, dim)
    }

    /// Hat operator: v^∧ (vector to matrix).
    ///
    /// For Euclidean space, this could be interpreted as a diagonal matrix
    /// or simply return the vector as a column matrix.
    fn hat(&self) -> <Rn as manifold::LieGroup>::LieAlgebra {
        nalgebra::DMatrix::from_diagonal(&self.data)
    }

    /// Small adjugate operator for Euclidean space.
    ///
    /// For abelian groups, this is zero matrix.
    fn small_adj(&self) -> <Rn as manifold::LieGroup>::JacobianMatrix {
        let dim = self.data.len();
        nalgebra::DMatrix::zeros(dim, dim)
    }

    /// Lie bracket for Euclidean space.
    ///
    /// For abelian groups: [v, w] = 0
    fn lie_bracket(&self, _other: &Self) -> <Rn as manifold::LieGroup>::TangentVector {
        RnTangent::new(nalgebra::DVector::zeros(self.data.len()))
    }

    /// Check if the tangent vector is approximately equal to another tangent vector.
    ///
    /// # Arguments
    /// * `other` - The other tangent vector to compare with
    /// * `tolerance` - The tolerance for the comparison
    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        if self.data.len() != other.data.len() {
            return false;
        }
        (&self.data - &other.data).norm() < tolerance
    }

    /// Get the i-th generator of the Lie algebra.
    ///
    /// For Euclidean space, generators are standard basis vectors.
    fn generator(&self, i: usize) -> <Rn as manifold::LieGroup>::LieAlgebra {
        let dim = self.data.len();
        if i >= dim {
            panic!("Generator index {} out of bounds for dimension {}", i, dim);
        }
        let mut generator_matrix = nalgebra::DMatrix::zeros(dim, dim);
        generator_matrix[(i, i)] = 1.0;
        generator_matrix
    }

    /// Zero tangent vector.
    fn zero() -> <Rn as manifold::LieGroup>::TangentVector {
        // Default to 3D zero vector for compatibility
        RnTangent::new(nalgebra::DVector::zeros(3))
    }

    /// Random tangent vector.
    fn random() -> <Rn as manifold::LieGroup>::TangentVector {
        // Default to 3D random vector
        let data = nalgebra::DVector::from_fn(3, |_, _| rand::random::<f64>() * 10.0 - 5.0);
        RnTangent::new(data)
    }

    /// Check if the tangent vector is approximately zero.
    fn is_zero(&self, tolerance: f64) -> bool {
        self.data.norm() < tolerance
    }

    /// Normalize the tangent vector to unit norm.
    fn normalize(&mut self) {
        let norm = self.data.norm();
        if norm > 1e-12 {
            self.data /= norm;
        }
    }

    /// Return a unit tangent vector in the same direction.
    fn normalized(&self) -> <Rn as manifold::LieGroup>::TangentVector {
        let mut result = self.clone();
        result.normalize();
        result
    }
}

// Implement Interpolatable trait for Rn
impl manifold::Interpolatable for Rn {
    /// Linear interpolation in Euclidean space.
    ///
    /// For parameter t ∈ [0,1]: interp(v₁, v₂, 0) = v₁, interp(v₁, v₂, 1) = v₂.
    ///
    /// # Arguments
    /// * `other` - Target element for interpolation
    /// * `t` - Interpolation parameter in [0,1]
    fn interp(&self, other: &Self, t: f64) -> Self {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Rn elements must have the same dimension for interpolation"
        );

        let interpolated = &self.data * (1.0 - t) + &other.data * t;
        Rn::new(interpolated)
    }

    /// Spherical linear interpolation (same as linear for Euclidean space).
    fn slerp(&self, other: &Self, t: f64) -> Self {
        self.interp(other, t)
    }
}

// Additional convenience implementations
impl Rn {
    /// Create Rn with specific dimension filled with zeros.
    pub fn zeros(dim: usize) -> Self {
        Rn::new(nalgebra::DVector::zeros(dim))
    }

    /// Create Rn with specific dimension filled with ones.
    pub fn ones(dim: usize) -> Self {
        Rn::new(nalgebra::DVector::from_element(dim, 1.0))
    }

    /// Create Rn with specific dimension and random values.
    pub fn random_with_dim(dim: usize) -> Self {
        let data = nalgebra::DVector::from_fn(dim, |_, _| rand::random::<f64>() * 10.0 - 5.0);
        Rn::new(data)
    }

    /// Create identity matrix for Jacobians with specific dimension.
    pub fn jacobian_identity_with_dim(dim: usize) -> nalgebra::DMatrix<f64> {
        nalgebra::DMatrix::identity(dim, dim)
    }
}

impl RnTangent {
    /// Create RnTangent with specific dimension filled with zeros.
    pub fn zeros(dim: usize) -> Self {
        RnTangent::new(nalgebra::DVector::zeros(dim))
    }

    /// Create RnTangent with specific dimension filled with ones.
    pub fn ones(dim: usize) -> Self {
        RnTangent::new(nalgebra::DVector::from_element(dim, 1.0))
    }

    /// Create RnTangent with specific dimension and random values.
    pub fn random_with_dim(dim: usize) -> Self {
        let data = nalgebra::DVector::from_fn(dim, |_, _| rand::random::<f64>() * 2.0 - 1.0);
        RnTangent::new(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{Interpolatable, LieGroup, Tangent};

    #[test]
    fn test_rn_basic_operations() {
        // Test 3D Euclidean space
        let v1 = Rn::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = Rn::from_vec(vec![4.0, 5.0, 6.0]);

        // Test composition (addition)
        let sum = v1.compose(&v2, None, None);
        assert_eq!(sum.component(0), 5.0);
        assert_eq!(sum.component(1), 7.0);
        assert_eq!(sum.component(2), 9.0);

        // Test identity
        let identity = Rn::zeros(3);
        let result = v1.compose(&identity, None, None);
        assert_eq!(result.component(0), v1.component(0));
        assert_eq!(result.component(1), v1.component(1));
        assert_eq!(result.component(2), v1.component(2));

        // Test inverse
        let v1_inv = v1.inverse(None);
        assert_eq!(v1_inv.component(0), -1.0);
        assert_eq!(v1_inv.component(1), -2.0);
        assert_eq!(v1_inv.component(2), -3.0);

        // Test log/exp (should be identity for Euclidean space)
        let tangent = v1.log(None);
        let recovered = tangent.exp(None);
        assert!((recovered.component(0) - v1.component(0)).abs() < 1e-10);
        assert!((recovered.component(1) - v1.component(1)).abs() < 1e-10);
        assert!((recovered.component(2) - v1.component(2)).abs() < 1e-10);
    }

    #[test]
    fn test_rn_tangent_operations() {
        let t1 = RnTangent::from_vec(vec![1.0, 2.0, 3.0]);
        let t2 = RnTangent::from_vec(vec![4.0, 5.0, 6.0]);

        // Test Lie bracket (should be zero for abelian group)
        let bracket = t1.lie_bracket(&t2);
        assert!(bracket.is_zero(1e-10));

        // Test zero tangent
        let zero = RnTangent::zeros(3);
        assert!(zero.is_zero(1e-10));

        // Test normalization
        let mut t = RnTangent::from_vec(vec![3.0, 4.0, 0.0]);
        t.normalize();
        assert!((t.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rn_interpolation() {
        let v1 = Rn::from_vec(vec![0.0, 0.0, 0.0]);
        let v2 = Rn::from_vec(vec![10.0, 20.0, 30.0]);

        // Test interpolation at t=0.5
        let mid = v1.interp(&v2, 0.5);
        assert_eq!(mid.component(0), 5.0);
        assert_eq!(mid.component(1), 10.0);
        assert_eq!(mid.component(2), 15.0);

        // Test interpolation at endpoints
        let start = v1.interp(&v2, 0.0);
        let end = v1.interp(&v2, 1.0);
        assert!(v1.is_approx(&start, 1e-10));
        assert!(v2.is_approx(&end, 1e-10));
    }

    #[test]
    fn test_rn_different_dimensions() {
        // Test 2D
        let v2d = Rn::from_vec(vec![1.0, 2.0]);
        assert_eq!(v2d.dim(), 2);

        // Test 1D
        let v1d = Rn::from_vec(vec![5.0]);
        assert_eq!(v1d.dim(), 1);

        // Test higher dimensions
        let v5d = Rn::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(v5d.dim(), 5);
        assert_eq!(v5d.component(4), 5.0);
    }

    #[test]
    fn test_rn_action() {
        // Test action on 3D vector (translation)
        let translation = Rn::from_vec(vec![1.0, 2.0, 3.0]);
        let point = nalgebra::Vector3::new(4.0, 5.0, 6.0);
        let transformed = translation.act(&point, None, None);

        assert_eq!(transformed.x, 5.0);
        assert_eq!(transformed.y, 7.0);
        assert_eq!(transformed.z, 9.0);
    }

    #[test]
    fn test_rn_right_plus_operations() {
        let v = Rn::from_vec(vec![1.0, 2.0, 3.0]);
        let delta = RnTangent::from_vec(vec![0.1, 0.2, 0.3]);

        // Test right plus without Jacobians
        let result = v.right_plus(&delta, None, None);
        assert!((result.component(0) - 1.1).abs() < 1e-10);
        assert!((result.component(1) - 2.2).abs() < 1e-10);
        assert!((result.component(2) - 3.3).abs() < 1e-10);

        // Test right plus with Jacobians
        let mut jac_self = nalgebra::DMatrix::zeros(3, 3);
        let mut jac_tangent = nalgebra::DMatrix::zeros(3, 3);
        let result_jac = v.right_plus(&delta, Some(&mut jac_self), Some(&mut jac_tangent));

        // Verify result is the same
        assert!((result_jac.component(0) - result.component(0)).abs() < 1e-10);
        assert!((result_jac.component(1) - result.component(1)).abs() < 1e-10);
        assert!((result_jac.component(2) - result.component(2)).abs() < 1e-10);

        // Verify Jacobians are identity matrices
        let identity = nalgebra::DMatrix::identity(3, 3);
        assert!((jac_self - &identity).norm() < 1e-10);
        assert!((jac_tangent - &identity).norm() < 1e-10);
    }

    #[test]
    fn test_rn_right_minus_operations() {
        let v1 = Rn::from_vec(vec![5.0, 7.0, 9.0]);
        let v2 = Rn::from_vec(vec![1.0, 2.0, 3.0]);

        // Test right minus without Jacobians
        let result = v1.right_minus(&v2, None, None);
        assert!((result.component(0) - 4.0).abs() < 1e-10);
        assert!((result.component(1) - 5.0).abs() < 1e-10);
        assert!((result.component(2) - 6.0).abs() < 1e-10);

        // Test right minus with Jacobians
        let mut jac_self = nalgebra::DMatrix::zeros(3, 3);
        let mut jac_other = nalgebra::DMatrix::zeros(3, 3);
        let result_jac = v1.right_minus(&v2, Some(&mut jac_self), Some(&mut jac_other));

        // Verify result is the same
        assert!((result_jac.component(0) - result.component(0)).abs() < 1e-10);
        assert!((result_jac.component(1) - result.component(1)).abs() < 1e-10);
        assert!((result_jac.component(2) - result.component(2)).abs() < 1e-10);

        // Verify Jacobians
        let identity = nalgebra::DMatrix::identity(3, 3);
        let neg_identity = -&identity;
        assert!((jac_self - &identity).norm() < 1e-10);
        assert!((jac_other - &neg_identity).norm() < 1e-10);
    }

    #[test]
    fn test_rn_left_plus_operations() {
        let v = Rn::from_vec(vec![1.0, 2.0, 3.0]);
        let delta = RnTangent::from_vec(vec![0.1, 0.2, 0.3]);

        // Test left plus without Jacobians
        let result = v.left_plus(&delta, None, None);
        assert!((result.component(0) - 1.1).abs() < 1e-10);
        assert!((result.component(1) - 2.2).abs() < 1e-10);
        assert!((result.component(2) - 3.3).abs() < 1e-10);

        // Test left plus with Jacobians
        let mut jac_tangent = nalgebra::DMatrix::zeros(3, 3);
        let mut jac_self = nalgebra::DMatrix::zeros(3, 3);
        let result_jac = v.left_plus(&delta, Some(&mut jac_tangent), Some(&mut jac_self));

        // Verify result is the same
        assert!((result_jac.component(0) - result.component(0)).abs() < 1e-10);
        assert!((result_jac.component(1) - result.component(1)).abs() < 1e-10);
        assert!((result_jac.component(2) - result.component(2)).abs() < 1e-10);

        // Verify Jacobians are identity matrices
        let identity = nalgebra::DMatrix::identity(3, 3);
        assert!((jac_tangent - &identity).norm() < 1e-10);
        assert!((jac_self - &identity).norm() < 1e-10);
    }

    #[test]
    fn test_rn_left_minus_operations() {
        let v1 = Rn::from_vec(vec![5.0, 7.0, 9.0]);
        let v2 = Rn::from_vec(vec![1.0, 2.0, 3.0]);

        // Test left minus without Jacobians
        let result = v1.left_minus(&v2, None, None);
        assert!((result.component(0) - 4.0).abs() < 1e-10);
        assert!((result.component(1) - 5.0).abs() < 1e-10);
        assert!((result.component(2) - 6.0).abs() < 1e-10);

        // Test left minus with Jacobians
        let mut jac_self = nalgebra::DMatrix::zeros(3, 3);
        let mut jac_other = nalgebra::DMatrix::zeros(3, 3);
        let result_jac = v1.left_minus(&v2, Some(&mut jac_self), Some(&mut jac_other));

        // Verify result is the same
        assert!((result_jac.component(0) - result.component(0)).abs() < 1e-10);
        assert!((result_jac.component(1) - result.component(1)).abs() < 1e-10);
        assert!((result_jac.component(2) - result.component(2)).abs() < 1e-10);

        // Verify Jacobians
        let identity = nalgebra::DMatrix::identity(3, 3);
        let neg_identity = -&identity;
        assert!((jac_self - &identity).norm() < 1e-10);
        assert!((jac_other - &neg_identity).norm() < 1e-10);
    }

    #[test]
    fn test_rn_left_right_equivalence() {
        // For abelian groups, left and right operations should be equivalent
        let v1 = Rn::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = Rn::from_vec(vec![4.0, 5.0, 6.0]);
        let delta = RnTangent::from_vec(vec![0.1, 0.2, 0.3]);

        // Test plus operations equivalence
        let right_plus = v1.right_plus(&delta, None, None);
        let left_plus = v1.left_plus(&delta, None, None);
        assert!(right_plus.is_approx(&left_plus, 1e-10));

        // Test minus operations equivalence
        let right_minus = v1.right_minus(&v2, None, None);
        let left_minus = v1.left_minus(&v2, None, None);
        assert!(right_minus.is_approx(&left_minus, 1e-10));
    }

    #[test]
    fn test_rn_plus_minus_different_dimensions() {
        // Test 2D operations
        let v2d = Rn::from_vec(vec![1.0, 2.0]);
        let delta2d = RnTangent::from_vec(vec![0.5, 1.0]);
        let result2d = v2d.right_plus(&delta2d, None, None);
        assert_eq!(result2d.dim(), 2);
        assert!((result2d.component(0) - 1.5).abs() < 1e-10);
        assert!((result2d.component(1) - 3.0).abs() < 1e-10);

        // Test 5D operations
        let v5d = Rn::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let delta5d = RnTangent::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let result5d = v5d.right_plus(&delta5d, None, None);
        assert_eq!(result5d.dim(), 5);
        for i in 0..5 {
            assert!(
                (result5d.component(i) - (i as f64 + 1.0 + (i as f64 + 1.0) * 0.1)).abs() < 1e-10
            );
        }
    }

    #[test]
    fn test_rn_plus_minus_edge_cases() {
        let v = Rn::from_vec(vec![1.0, 2.0, 3.0]);

        // Test with zero tangent vector
        let zero_tangent = RnTangent::zeros(3);
        let result_zero = v.right_plus(&zero_tangent, None, None);
        assert!(v.is_approx(&result_zero, 1e-10));

        // Test minus with itself (should give zero)
        let self_minus = v.right_minus(&v, None, None);
        assert!(self_minus.is_zero(1e-10));

        // Test plus then minus (should recover original)
        let delta = RnTangent::from_vec(vec![0.5, 1.0, 1.5]);
        let plus_result = v.right_plus(&delta, None, None);
        let recovered_delta = plus_result.right_minus(&v, None, None);
        assert!(delta.is_approx(&recovered_delta, 1e-10));
    }

    #[test]
    fn test_rn_jacobian_dimensions() {
        // Test that Jacobians have correct dimensions for different vector sizes
        let v1d = Rn::from_vec(vec![1.0]);
        let delta1d = RnTangent::from_vec(vec![0.1]);
        let mut jac1d = nalgebra::DMatrix::zeros(1, 1);
        v1d.right_plus(&delta1d, Some(&mut jac1d), None);
        assert_eq!(jac1d.nrows(), 1);
        assert_eq!(jac1d.ncols(), 1);

        let v4d = Rn::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let delta4d = RnTangent::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let mut jac4d = nalgebra::DMatrix::zeros(4, 4);
        v4d.right_plus(&delta4d, Some(&mut jac4d), None);
        assert_eq!(jac4d.nrows(), 4);
        assert_eq!(jac4d.ncols(), 4);

        // Verify they are identity matrices
        assert!((jac1d - nalgebra::DMatrix::identity(1, 1)).norm() < 1e-10);
        assert!((jac4d - nalgebra::DMatrix::identity(4, 4)).norm() < 1e-10);
    }
}
