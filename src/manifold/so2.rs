//! SO(2) - Special Orthogonal Group in 2D
//!
//! This module implements the Special Orthogonal group SO(2), which represents
//! rotations in 2D space.
//!
//! SO(2) elements are represented using nalgebra's UnitComplex internally.
//! SO(2) tangent elements are represented as a single angle in radians.
//!
//! The implementation follows the [manif](https://github.com/artivis/manif) C++ library
//! conventions and provides all operations required by the LieGroup and Tangent traits.

use crate::manifold::{LieGroup, Tangent};
use nalgebra::{Matrix2, UnitComplex};
use std::fmt;

/// SO(2) group element representing rotations in 2D.
///
/// Internally represented using nalgebra's UnitComplex<f64> for efficient rotations.
#[derive(Clone, Debug, PartialEq)]
pub struct SO2 {
    /// Internal representation as a unit complex number
    complex: UnitComplex<f64>,
}

impl fmt::Display for SO2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SO2(angle: {:.4})", self.complex.angle())
    }
}

// Conversion traits for integration with generic Problem
impl From<nalgebra::DVector<f64>> for SO2 {
    fn from(data: nalgebra::DVector<f64>) -> Self {
        if data.len() != 1 {
            panic!("SO2::from expects 1-dimensional vector [angle]");
        }
        SO2::from_angle(data[0])
    }
}

impl From<SO2> for nalgebra::DVector<f64> {
    fn from(so2: SO2) -> Self {
        nalgebra::DVector::from_vec(vec![so2.complex.angle()])
    }
}

/// SO(2) tangent space element representing elements in the Lie algebra so(2).
///
/// Internally represented as a single scalar (angle in radians).
#[derive(Clone, Debug, PartialEq)]
pub struct SO2Tangent {
    /// Internal data: angle (radians)
    data: f64,
}

impl fmt::Display for SO2Tangent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "so2(angle: {:.4})", self.data)
    }
}

impl SO2 {
    /// Space dimension - dimension of the ambient space that the group acts on
    pub const DIM: usize = 2;

    /// Degrees of freedom - dimension of the tangent space
    pub const DOF: usize = 1;

    /// Representation size - size of the underlying data representation
    pub const REP_SIZE: usize = 2;

    /// Get the identity element of the group.
    ///
    /// Returns the neutral element e such that e ∘ g = g ∘ e = g for any group element g.
    pub fn identity() -> Self {
        SO2 {
            complex: UnitComplex::identity(),
        }
    }

    /// Get the identity matrix for Jacobians.
    ///
    /// Returns the identity matrix in the appropriate dimension for Jacobian computations.
    pub fn jacobian_identity() -> nalgebra::Matrix1<f64> {
        nalgebra::Matrix1::<f64>::identity()
    }

    /// Create a new SO(2) element from a unit complex number.
    ///
    /// # Arguments
    /// * `complex` - Unit complex number representing rotation
    pub fn new(complex: UnitComplex<f64>) -> Self {
        SO2 { complex }
    }

    /// Create SO(2) from an angle.
    ///
    /// # Arguments
    /// * `angle` - Rotation angle in radians
    pub fn from_angle(angle: f64) -> Self {
        SO2::new(UnitComplex::from_angle(angle))
    }

    /// Get the underlying unit complex number.
    pub fn complex(&self) -> UnitComplex<f64> {
        self.complex
    }

    /// Get the rotation angle in radians.
    pub fn angle(&self) -> f64 {
        self.complex.angle()
    }

    /// Get the rotation matrix (2x2).
    pub fn rotation_matrix(&self) -> Matrix2<f64> {
        self.complex.to_rotation_matrix().into_inner()
    }
}

impl LieGroup for SO2 {
    type TangentVector = SO2Tangent;
    type JacobianMatrix = nalgebra::Matrix1<f64>;
    type LieAlgebra = Matrix2<f64>;

    /// SO2 inverse.
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian matrix of the inverse wrt self.
    ///
    /// # Notes
    /// # Equation 118: SO(2) Inverse
    /// R(θ)⁻¹ = R(-θ)
    ///
    /// # Equation 124: Jacobian of Inverse for SO(2)
    /// J_R⁻¹_R = -I
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self {
        if let Some(jac) = jacobian {
            *jac = -self.adjoint();
        }
        SO2 {
            complex: self.complex.inverse(),
        }
    }

    /// SO2 composition.
    ///
    /// # Arguments
    /// * `other` - Another SO2 element.
    /// * `jacobian_self` - Optional Jacobian matrix of the composition wrt self.
    /// * `jacobian_other` - Optional Jacobian matrix of the composition wrt other.
    ///
    /// # Notes
    /// # Equation 125: Jacobian of Composition for SO(2)
    /// J_C_A = I
    /// J_C_B = I
    fn compose(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        if let Some(jac_self) = jacobian_self {
            *jac_self = other.inverse(None).adjoint();
        }
        if let Some(jac_other) = jacobian_other {
            *jac_other = nalgebra::Matrix1::identity();
        }
        SO2 {
            complex: self.complex * other.complex,
        }
    }

    /// Get the SO2 corresponding Lie algebra element in vector form.
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian matrix of the tangent wrt to self.
    ///
    /// # Notes
    /// # Equation 115: Logarithmic map for SO(2)
    /// θ = atan2(R(1,0), R(0,0))
    ///
    /// # Equation 126: Jacobian of Logarithmic map for SO(2)
    /// J_log_R = I
    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector {
        if let Some(jac) = jacobian {
            *jac = nalgebra::Matrix1::identity();
        }
        SO2Tangent {
            data: self.complex.angle(),
        }
    }

    /// Rotation action on a 3-vector.
    ///
    /// # Arguments
    /// * `v` - A 3-vector.
    /// * `jacobian_self` - Optional Jacobian of the new object wrt this.
    /// * `jacobian_vector` - Optional Jacobian of the new object wrt input object.
    ///
    /// # Returns
    /// The rotated 3-vector.
    ///
    /// # Notes
    /// This is a convenience function that treats the 3D vector as a 2D vector and ignores the z component.
    fn act(
        &self,
        vector: &nalgebra::Vector3<f64>,
        _jacobian_self: Option<&mut Self::JacobianMatrix>,
        _jacobian_vector: Option<&mut nalgebra::Matrix3<f64>>,
    ) -> nalgebra::Vector3<f64> {
        let point2d = nalgebra::Vector2::new(vector.x, vector.y);
        let rotated_point = self.complex * point2d;
        nalgebra::Vector3::new(rotated_point.x, rotated_point.y, vector.z)
    }

    /// Get the adjoint matrix of SO2 at this.
    ///
    /// # Notes
    /// See Eq. (123).
    fn adjoint(&self) -> Self::JacobianMatrix {
        nalgebra::Matrix1::identity()
    }

    /// Generate a random element.
    fn random() -> Self {
        SO2::from_angle(rand::random::<f64>() * 2.0 * std::f64::consts::PI)
    }

    /// Normalize the underlying complex number.
    fn normalize(&mut self) {
        self.complex.renormalize();
    }

    /// Check if the element is valid.
    fn is_valid(&self, tolerance: f64) -> bool {
        let norm_diff = (self.complex.norm() - 1.0).abs();
        norm_diff < tolerance
    }

    /// Vee operator: log(g)^∨.
    ///
    /// Maps a group element g ∈ G to its tangent vector log(g)^∨ ∈ 𝔤.
    /// For SO(2), this is the same as log().
    fn vee(&self) -> Self::TangentVector {
        self.log(None)
    }

    /// Check if the element is approximately equal to another element.
    ///
    /// # Arguments
    /// * `other` - The other element to compare with
    /// * `tolerance` - The tolerance for the comparison
    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        let difference = self.right_minus(other, None, None);
        difference.is_zero(tolerance)
    }
}

impl SO2Tangent {
    /// Create a new SO2Tangent from an angle.
    ///
    /// # Arguments
    /// * `angle` - Angle in radians
    pub fn new(angle: f64) -> Self {
        SO2Tangent { data: angle }
    }

    /// Get the angle.
    pub fn angle(&self) -> f64 {
        self.data
    }

    /// Create SO2Tangent from a 1-dimensional vector
    pub fn from_vector(vector: nalgebra::DVector<f64>) -> Self {
        if vector.len() != 1 {
            panic!("SO2Tangent::from_vector expects 1-dimensional vector");
        }
        SO2Tangent { data: vector[0] }
    }

    /// Convert SO2Tangent to a 1-dimensional vector
    pub fn to_vector(&self) -> nalgebra::DVector<f64> {
        nalgebra::DVector::from_element(1, self.data)
    }
}

impl Tangent<SO2> for SO2Tangent {
    /// Dimension of the tangent space
    const DIM: usize = 1;

    /// SO2 exponential map.
    ///
    /// # Arguments
    /// * `tangent` - Tangent vector (angle)
    /// * `jacobian` - Optional Jacobian matrix of the SE(3) element wrt this.
    fn exp(&self, jacobian: Option<&mut <SO2 as LieGroup>::JacobianMatrix>) -> SO2 {
        let angle = self.angle();
        let complex = UnitComplex::new(angle);

        if let Some(jac) = jacobian {
            *jac = nalgebra::Matrix1::identity();
        }

        SO2::new(complex)
    }

    /// Right Jacobian for SO(2) is identity.
    fn right_jacobian(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        nalgebra::Matrix1::identity()
    }

    /// Left Jacobian for SO(2) is identity.
    fn left_jacobian(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        nalgebra::Matrix1::identity()
    }

    /// Inverse of right Jacobian for SO(2) is identity.
    fn right_jacobian_inv(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        nalgebra::Matrix1::identity()
    }

    /// Inverse of left Jacobian for SO(2) is identity.
    fn left_jacobian_inv(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        nalgebra::Matrix1::identity()
    }

    /// Hat operator: θ^∧ (scalar to skew-symmetric matrix).
    fn hat(&self) -> <SO2 as LieGroup>::LieAlgebra {
        let theta = self.data;
        Matrix2::new(0.0, -theta, theta, 0.0)
    }

    /// Zero tangent vector for SO2
    fn zero() -> Self {
        SO2Tangent { data: 0.0 }
    }

    /// Random tangent vector for SO2
    fn random() -> Self {
        SO2Tangent {
            data: rand::random::<f64>() * 0.2 - 0.1,
        }
    }

    /// Check if tangent vector is zero
    fn is_zero(&self, tolerance: f64) -> bool {
        self.data.abs() < tolerance
    }

    /// Normalize tangent vector
    fn normalize(&mut self) {
        // Normalizing a scalar doesn't make much sense unless it's a direction.
        // For an angle, this is a no-op.
    }

    /// Return a unit tangent vector in the same direction.
    fn normalized(&self) -> Self {
        if self.data.abs() > f64::EPSILON {
            SO2Tangent::new(self.data.signum())
        } else {
            SO2Tangent::new(0.0)
        }
    }

    /// Small adjoint matrix for SO(2).
    ///
    /// For SO(2), the small adjoint is zero (since it's commutative).
    fn small_adj(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        nalgebra::Matrix1::zeros()
    }

    /// Lie bracket for SO(2).
    ///
    /// For SO(2), the Lie bracket is always zero since it's commutative.
    fn lie_bracket(&self, _other: &Self) -> <SO2 as LieGroup>::TangentVector {
        SO2Tangent::zero()
    }

    /// Check if this tangent vector is approximately equal to another.
    ///
    /// # Arguments
    /// * `other` - The other tangent vector to compare with
    /// * `tolerance` - The tolerance for the comparison
    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        (self.data - other.data).abs() < tolerance
    }

    /// Get the ith generator of the SO(2) Lie algebra.
    ///
    /// # Arguments
    /// * `i` - Index of the generator (must be 0 for SO(2))
    ///
    /// # Returns
    /// The generator matrix
    fn generator(&self, i: usize) -> <SO2 as LieGroup>::LieAlgebra {
        assert_eq!(i, 0, "SO(2) only has one generator (index 0)");
        // The generator for SO(2) is the skew-symmetric matrix:
        // E = | 0 -1 |
        //     | 1  0 |
        Matrix2::new(0.0, -1.0, 1.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOLERANCE: f64 = 1e-12;

    #[test]
    fn test_so2_identity() {
        let so2 = SO2::identity();
        assert!((so2.angle() - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_so2_inverse() {
        let so2 = SO2::from_angle(PI / 4.0);
        let so2_inv = so2.inverse(None);
        assert!((so2_inv.angle() + PI / 4.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_so2_compose() {
        let so2_a = SO2::from_angle(PI / 4.0);
        let so2_b = SO2::from_angle(PI / 2.0);
        let composed = so2_a.compose(&so2_b, None, None);
        assert!((composed.angle() - (3.0 * PI / 4.0)).abs() < TOLERANCE);
    }

    #[test]
    fn test_so2_exp_log_consistency() {
        let angle = PI / 4.0;
        let tangent = SO2Tangent::new(angle);
        let so2 = tangent.exp(None);
        let recovered_tangent = so2.log(None);

        assert!((tangent.angle() - recovered_tangent.angle()).abs() < 1e-10);
    }

    // New tests for the additional functions

    #[test]
    fn test_so2_vee() {
        let so2 = SO2::from_angle(PI / 3.0);
        let tangent_log = so2.log(None);
        let tangent_vee = so2.vee();

        assert!((tangent_log.angle() - tangent_vee.angle()).abs() < 1e-10);
    }

    #[test]
    fn test_so2_is_approx() {
        let so2_1 = SO2::from_angle(PI / 4.0);
        let so2_2 = SO2::from_angle(PI / 4.0 + 1e-12);
        let so2_3 = SO2::from_angle(PI / 2.0);

        assert!(so2_1.is_approx(&so2_1, 1e-10));
        assert!(so2_1.is_approx(&so2_2, 1e-10));
        assert!(!so2_1.is_approx(&so2_3, 1e-10));
    }

    #[test]
    fn test_so2_tangent_small_adj() {
        let tangent = SO2Tangent::new(PI / 6.0);
        let small_adj = tangent.small_adj();

        // For SO(2), small adjoint should be zero (commutative group)
        assert!((small_adj[(0, 0)]).abs() < 1e-10);
    }

    #[test]
    fn test_so2_tangent_lie_bracket() {
        let tangent_a = SO2Tangent::new(0.1);
        let tangent_b = SO2Tangent::new(0.2);

        let bracket = tangent_a.lie_bracket(&tangent_b);

        // For SO(2), Lie bracket should be zero (commutative group)
        assert!(bracket.is_zero(1e-10));

        // Anti-symmetry test: [a,b] = -[b,a]
        let bracket_ba = tangent_b.lie_bracket(&tangent_a);
        assert!(bracket.lie_bracket(&tangent_b).is_zero(1e-10)); // [a,a] = 0

        // Since SO(2) is commutative, both should be zero
        assert!(bracket_ba.is_zero(1e-10));
    }

    #[test]
    fn test_so2_tangent_is_approx() {
        let tangent_1 = SO2Tangent::new(0.5);
        let tangent_2 = SO2Tangent::new(0.5 + 1e-12);
        let tangent_3 = SO2Tangent::new(1.0);

        assert!(tangent_1.is_approx(&tangent_1, 1e-10));
        assert!(tangent_1.is_approx(&tangent_2, 1e-10));
        assert!(!tangent_1.is_approx(&tangent_3, 1e-10));
    }

    #[test]
    fn test_so2_generator() {
        let tangent = SO2Tangent::new(1.0);
        let generator = tangent.generator(0);

        // SO(2) generator should be the skew-symmetric matrix
        let expected = Matrix2::new(0.0, -1.0, 1.0, 0.0);

        assert!((generator - expected).norm() < 1e-10);
    }

    #[test]
    #[should_panic]
    fn test_so2_generator_invalid_index() {
        let tangent = SO2Tangent::new(1.0);
        let _generator = tangent.generator(1); // Should panic for SO(2)
    }

    #[test]
    fn test_so2_bracket_hat_relationship() {
        let a = SO2Tangent::new(0.1);
        let b = SO2Tangent::new(0.2);

        // For SO(2): [a,b]^ = a^ * b^ - b^ * a^ should be zero (commutative)
        let bracket_hat = a.lie_bracket(&b).hat();
        let expected = a.hat() * b.hat() - b.hat() * a.hat();

        assert!((bracket_hat - expected).norm() < 1e-10);
        assert!(expected.norm() < 1e-10); // Should be zero for SO(2)
    }
}
