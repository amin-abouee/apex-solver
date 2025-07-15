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
    type Element = SO2;
    type TangentVector = SO2Tangent;
    type JacobianMatrix = nalgebra::Matrix1<f64>;
    type LieAlgebra = Matrix2<f64>;

    const DIM: usize = 2; // Space dimension (2D space)
    const DOF: usize = 1; // Degrees of freedom (1 rotation parameter)
    const REP_SIZE: usize = 2; // Representation size (2 complex components)

    /// Get the identity element.
    fn identity() -> Self::Element {
        SO2 {
            complex: UnitComplex::identity(),
        }
    }

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
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::Element {
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
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
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

    /// Right plus: R ⊕ φ = R * exp(φ)
    ///
    /// # Arguments
    /// * `tangent` - Tangent vector
    /// * `jacobian_self` - Optional Jacobian matrix of the composition wrt self.
    /// * `jacobian_tangent` - Optional Jacobian matrix of the composition wrt tangent.
    fn right_plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        let result = self.compose(&tangent.exp(None), jacobian_self, None);
        if let Some(jac_tangent) = jacobian_tangent {
            *jac_tangent = tangent.right_jacobian();
        }
        result
    }

    /// Right minus: R1 ⊖ R2 = log(R2^T * R1)
    ///
    /// # Arguments
    /// * `other` - Another SO2 element.
    /// * `jacobian_self` - Optional Jacobian matrix of the composition wrt self.
    /// * `jacobian_other` - Optional Jacobian matrix of the composition wrt other.
    fn right_minus(
        &self,
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::TangentVector {
        let result = other.inverse(None).compose(self, None, None).log(None);
        if let Some(jac_self) = jacobian_self {
            *jac_self = result.right_jacobian_inv();
        }
        if let Some(jac_other) = jacobian_other {
            *jac_other = -result.left_jacobian_inv();
        }
        result
    }

    /// Left plus: φ ⊕ R = exp(φ) * R
    ///
    /// # Arguments
    /// * `tangent` - Tangent vector
    /// * `jacobian_tangent` - Optional Jacobian matrix of the composition wrt tangent.
    /// * `jacobian_self` - Optional Jacobian matrix of the composition wrt self.
    fn left_plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        let result = tangent.exp(None).compose(self, None, None);
        if let Some(jac_tangent) = jacobian_tangent {
            *jac_tangent = tangent.left_jacobian();
        }
        if let Some(jac_self) = jacobian_self {
            *jac_self = nalgebra::Matrix1::identity();
        }
        result
    }

    /// Left minus: R1 ⊖ R2 = log(R1 * R2^T)
    ///
    /// # Arguments
    /// * `other` - Another SO2 element.
    /// * `jacobian_self` - Optional Jacobian matrix of the composition wrt self.
    /// * `jacobian_other` - Optional Jacobian matrix of the composition wrt other.
    fn left_minus(
        &self,
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::TangentVector {
        let result = self.compose(&other.inverse(None), None, None).log(None);
        if let Some(jac_self) = jacobian_self {
            *jac_self = result.right_jacobian_inv();
        }
        if let Some(jac_other) = jacobian_other {
            *jac_other = -result.left_jacobian_inv();
        }
        result
    }

    /// Between: R1.between(R2) = R1^T * R2
    fn between(
        &self,
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        self.inverse(None)
            .compose(other, jacobian_self, jacobian_other)
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
    fn random() -> Self::Element {
        SO2::from_angle(rand::random::<f64>() * 2.0 * std::f64::consts::PI)
    }

    /// Normalize the underlying complex number.
    fn normalize(&mut self) {
        self.complex.renormalize_fast();
    }

    /// Check if the element is valid.
    fn is_valid(&self, tolerance: f64) -> bool {
        (self.complex.norm_sqr() - 1.0).abs() < tolerance
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
}

impl Tangent<SO2> for SO2Tangent {
    /// SO2 exponential map.
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian matrix of the SO2 element wrt self.
    ///
    /// # Notes
    /// # Equation 114: Exponential map for SO(2)
    /// R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    ///
    /// # Equation 128: Jacobian of Exponential map for SO(2)
    /// J_exp_θ = I
    fn exp(
        &self,
        jacobian: Option<&mut <SO2 as LieGroup>::JacobianMatrix>,
    ) -> <SO2 as LieGroup>::Element {
        if let Some(jac) = jacobian {
            *jac = self.right_jacobian();
        }
        SO2::from_angle(self.data)
    }

    /// Right Jacobian for SO2
    ///
    /// # Notes
    /// # Equation 126: Right Jacobian for SO(2)
    /// J_r(θ) = I
    fn right_jacobian(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        nalgebra::Matrix1::identity()
    }

    /// Left Jacobian for SO2
    ///
    /// # Notes
    /// # Equation 126: Left Jacobian for SO(2)
    /// J_l(θ) = I
    fn left_jacobian(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        nalgebra::Matrix1::identity()
    }

    /// Right Jacobian inverse for SO2
    ///
    /// # Notes
    /// # Equation 127: Right Jacobian inverse for SO(2)
    /// J_r(θ)⁻¹ = I
    fn right_jacobian_inv(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        nalgebra::Matrix1::identity()
    }

    /// Left Jacobian inverse for SO2
    ///
    /// # Notes
    /// # Equation 127: Left Jacobian inverse for SO(2)
    /// J_l(θ)⁻¹ = I
    fn left_jacobian_inv(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        nalgebra::Matrix1::identity()
    }

    /// Hat map for SO2
    ///
    /// # Notes
    /// # Equation 112: Hat map for so(2)
    /// [θ]ₓ = [0 -θ; θ 0]
    fn hat(&self) -> <SO2 as LieGroup>::LieAlgebra {
        Matrix2::new(0.0, -self.data, self.data, 0.0)
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

    /// Return a normalized tangent vector
    fn normalized(&self) -> Self {
        self.clone()
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
        let tangent = SO2Tangent::new(0.4);
        let so2 = tangent.exp(None);
        let recovered_tangent = so2.log(None);
        assert!((tangent.data - recovered_tangent.data).abs() < TOLERANCE);
    }
}