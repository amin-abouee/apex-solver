//! SO3 - Special Orthogonal Group in 3D
//!
//! This module implements the Special Orthogonal group SO(3), which represents
//! rotations in 3D space.
//!
//! SO(3) elements are represented using nalgebra's UnitQuaternion internally.
//! SO(3) tangent elements are represented as axis-angle vectors in R³,
//! where the direction gives the axis of rotation and the magnitude gives the angle.
//!
//! The implementation follows the [manif](https://github.com/artivis/manif) C++ library
//! conventions and provides all operations required by the LieGroup and Tangent traits.

use crate::manifold::{LieGroup, Tangent};
use nalgebra::{DVector, Matrix3, Matrix4, Quaternion, Unit, UnitQuaternion, Vector3};
use std::{
    fmt,
    fmt::{Display, Formatter},
};

/// SO(3) group element representing rotations in 3D.
///
/// Internally represented using nalgebra's UnitQuaternion<f64> for efficient rotations.
#[derive(Clone, PartialEq)]
pub struct SO3 {
    /// Internal representation as a unit quaternion
    quaternion: UnitQuaternion<f64>,
}

impl Display for SO3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let q = self.quaternion.quaternion();
        write!(
            f,
            "SO3(quaternion: [w: {:.4}, x: {:.4}, y: {:.4}, z: {:.4}])",
            q.w, q.i, q.j, q.k
        )
    }
}

impl SO3 {
    /// Space dimension - dimension of the ambient space that the group acts on
    pub const DIM: usize = 3;

    /// Degrees of freedom - dimension of the tangent space
    pub const DOF: usize = 3;

    /// Representation size - size of the underlying data representation
    pub const REP_SIZE: usize = 4;

    /// Get the identity element of the group.
    ///
    /// Returns the neutral element e such that e ∘ g = g ∘ e = g for any group element g.
    pub fn identity() -> Self {
        SO3 {
            quaternion: UnitQuaternion::identity(),
        }
    }

    /// Get the identity matrix for Jacobians.
    ///
    /// Returns the identity matrix in the appropriate dimension for Jacobian computations.
    pub fn jacobian_identity() -> Matrix3<f64> {
        Matrix3::<f64>::identity()
    }

    /// Create a new SO(3) element from a unit quaternion.
    ///
    /// # Arguments
    /// * `quaternion` - Unit quaternion representing rotation
    pub fn new(quaternion: UnitQuaternion<f64>) -> Self {
        SO3 { quaternion }
    }

    /// Create SO(3) from quaternion coefficients [x, y, z, w].
    ///
    /// # Arguments
    /// * `x` - i component of quaternion
    /// * `y` - j component of quaternion
    /// * `z` - k component of quaternion
    /// * `w` - w (real) component of quaternion
    pub fn from_quaternion_coeffs(x: f64, y: f64, z: f64, w: f64) -> Self {
        let q = Quaternion::new(w, x, y, z);
        SO3::new(UnitQuaternion::from_quaternion(q))
    }

    /// Create SO(3) from Euler angles (roll, pitch, yaw).
    pub fn from_euler_angles(roll: f64, pitch: f64, yaw: f64) -> Self {
        let quaternion = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
        SO3::new(quaternion)
    }

    /// Create SO(3) from axis-angle representation.
    pub fn from_axis_angle(axis: &Vector3<f64>, angle: f64) -> Self {
        let unit_axis = Unit::new_normalize(*axis);
        let quaternion = UnitQuaternion::from_axis_angle(&unit_axis, angle);
        SO3::new(quaternion)
    }

    /// Create SO(3) from scaled axis (axis-angle vector).
    pub fn from_scaled_axis(axis_angle: Vector3<f64>) -> Self {
        let quaternion = UnitQuaternion::from_scaled_axis(axis_angle);
        SO3::new(quaternion)
    }

    /// Get the quaternion representation.
    pub fn quaternion(&self) -> UnitQuaternion<f64> {
        self.quaternion
    }

    /// Create SO3 from quaternion (alias for new)
    pub fn from_quaternion(quaternion: UnitQuaternion<f64>) -> Self {
        Self::new(quaternion)
    }

    /// Get the quaternion representation (alias for quaternion)
    pub fn to_quaternion(&self) -> UnitQuaternion<f64> {
        self.quaternion
    }

    /// Get the raw quaternion coefficients.
    pub fn quat(&self) -> Quaternion<f64> {
        *self.quaternion.quaternion()
    }

    /// Get the x component of the quaternion.
    pub fn x(&self) -> f64 {
        self.quaternion.i
    }

    /// Get the y component of the quaternion.
    pub fn y(&self) -> f64 {
        self.quaternion.j
    }

    /// Get the z component of the quaternion.
    pub fn z(&self) -> f64 {
        self.quaternion.k
    }

    /// Get the w component of the quaternion.
    pub fn w(&self) -> f64 {
        self.quaternion.w
    }

    /// Get the rotation matrix (3x3).
    pub fn rotation_matrix(&self) -> Matrix3<f64> {
        self.quaternion.to_rotation_matrix().into_inner()
    }

    /// Get the homogeneous transformation matrix (4x4).
    pub fn transform(&self) -> Matrix4<f64> {
        self.quaternion.to_homogeneous()
    }

    /// Set the quaternion from coefficients array [w, x, y, z].
    pub fn set_quaternion(&mut self, coeffs: &[f64; 4]) {
        let q = Quaternion::new(coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
        self.quaternion = UnitQuaternion::from_quaternion(q);
    }

    /// Get coefficients as array [w, x, y, z].
    pub fn coeffs(&self) -> [f64; 4] {
        let q = self.quaternion.quaternion();
        [q.w, q.i, q.j, q.k]
    }

    /// Calculate the distance between two SO3 elements
    ///
    /// Computes the geodesic distance, which is the norm of the log map
    /// of the relative rotation between the two elements.
    pub fn distance(&self, other: &Self) -> f64 {
        self.between(other, None, None).log(None).angle()
    }
}

// Conversion traits for integration with generic Problem
impl From<DVector<f64>> for SO3 {
    fn from(data: DVector<f64>) -> Self {
        if data.len() != 4 {
            panic!("SO3::from expects 4-dimensional vector [w, i, j, k]");
        }
        SO3::from_quaternion_coeffs(data[0], data[1], data[2], data[3])
    }
}

impl From<SO3> for DVector<f64> {
    fn from(so3: SO3) -> Self {
        DVector::from_vec(so3.coeffs().to_vec())
    }
}

// Implement basic trait requirements for LieGroup
impl LieGroup for SO3 {
    type TangentVector = SO3Tangent;
    type JacobianMatrix = Matrix3<f64>;
    type LieAlgebra = Matrix3<f64>;

    /// SO3 inverse.
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian matrix of the inverse wrt self.
    ///
    /// # Notes
    /// R⁻¹ = Rᵀ, for quaternions: q⁻¹ = q*
    ///
    /// # Equation 140: Jacobian of Inverse for SO(3)
    /// J_R⁻¹_R = -Adj(R) = -R
    ///
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self {
        // For SO(3): R^{-1} = R^T, for quaternions: q^{-1} = q*
        let inverse_quat = self.quaternion.inverse();

        if let Some(jac) = jacobian {
            // Jacobian of inverse operation: -R^T = -R
            *jac = -self.adjoint();
        }

        SO3 {
            quaternion: inverse_quat,
        }
    }

    /// SO3 composition.
    ///
    /// # Arguments
    /// * `other` - Another SO3 element.
    /// * `jacobian_self` - Optional Jacobian matrix of the composition wrt self.
    /// * `jacobian_other` - Optional Jacobian matrix of the composition wrt other.
    ///
    /// # Notes
    /// # Equation 141: Jacobian of the composition wrt self.
    /// J_QR_R = Adj(R⁻¹) = Rᵀ
    ///
    /// # Equation 142: Jacobian of the composition wrt other.
    /// J_QR_Q = I
    ///
    fn compose(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        let result = SO3 {
            quaternion: self.quaternion * other.quaternion,
        };

        if let Some(jac_self) = jacobian_self {
            // Jacobian wrt first element: R2^T
            *jac_self = other.inverse(None).adjoint();
        }

        if let Some(jac_other) = jacobian_other {
            // Jacobian wrt second element: I (identity)
            *jac_other = Matrix3::identity();
        }

        result
    }

    /// Get the SO3 corresponding Lie algebra element in vector form.
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian matrix of the tangent wrt to self.
    ///
    /// # Notes
    /// # Equation 133: Logarithmic map for unit quaternions (S³)
    /// θu = Log(q) = (2 / ||v||) * v * arctan(||v||, w) ∈ R³
    ///
    /// # Equation 144: Inverse of Right Jacobian for SO(3) Exp map
    /// J_R⁻¹(θ) = I + (1/2) [θ]ₓ + (1/θ² - (1 + cos θ)/(2θ sin θ)) [θ]ₓ²
    ///
    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector {
        // let mut log_coeff;

        // Extract quaternion components
        let q = self.quaternion.quaternion();
        let sin_angle_squared = q.i * q.i + q.j * q.j + q.k * q.k;

        let log_coeff = if sin_angle_squared > f64::EPSILON {
            let sin_angle = sin_angle_squared.sqrt();
            let cos_angle = q.w;

            // Handle the case where cos_angle < 0, which means angle >= pi/2
            // In that case, we need to adjust the computation to get a normalized angle_axis vector
            let two_angle = 2.0
                * if cos_angle < 0.0 {
                    f64::atan2(-sin_angle, -cos_angle)
                } else {
                    f64::atan2(sin_angle, cos_angle)
                };

            two_angle / sin_angle
        } else {
            // Small-angle approximation
            2.0
        };

        // Compute the tangent vector (axis-angle representation)
        let axis_angle = SO3Tangent::new(Vector3::new(
            q.i * log_coeff,
            q.j * log_coeff,
            q.k * log_coeff,
        ));

        if let Some(jac) = jacobian {
            // Compute the right Jacobian inverse for SO(3)
            // Self::compute_right_jacobian_inverse(&axis_angle, jac);
            *jac = axis_angle.right_jacobian_inv();
        }

        axis_angle
    }

    fn act(
        &self,
        vector: &Vector3<f64>,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_vector: Option<&mut Matrix3<f64>>,
    ) -> Vector3<f64> {
        // Apply rotation to vector
        let result = self.quaternion * vector;

        if let Some(jac_self) = jacobian_self {
            // Jacobian wrt SO(3) element: -R * [v]×
            let vector_hat = SO3Tangent::new(*vector).hat();
            *jac_self = -self.rotation_matrix() * vector_hat;
        }

        if let Some(jac_vector) = jacobian_vector {
            // Jacobian wrt vector: R
            *jac_vector = self.rotation_matrix();
        }

        result
    }

    fn adjoint(&self) -> Self::JacobianMatrix {
        // Adjoint matrix for SO(3) is just the rotation matrix
        self.rotation_matrix()
    }

    fn random() -> Self {
        SO3 {
            quaternion: UnitQuaternion::from_scaled_axis(Vector3::new(
                rand::random::<f64>() * 2.0 - 1.0,
                rand::random::<f64>() * 2.0 - 1.0,
                rand::random::<f64>() * 2.0 - 1.0,
            )),
        }
    }

    fn normalize(&mut self) {
        // Normalize the quaternion
        let q = self.quaternion.into_inner().normalize();
        self.quaternion = UnitQuaternion::from_quaternion(q);
    }

    fn is_valid(&self, tolerance: f64) -> bool {
        // Check if the quaternion is normalized
        let norm_diff = (self.quaternion.norm() - 1.0).abs();
        norm_diff < tolerance
    }

    /// Vee operator: log(g)^∨.
    ///
    /// Maps a group element g ∈ G to its tangent vector log(g)^∨ ∈ 𝔤.
    /// For SO(3), this is the same as log().
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

/// SO(3) tangent space element representing elements in the Lie algebra so(3).
///
/// Internally represented as axis-angle vectors in R³ where:
/// - Direction: axis of rotation (unit vector)
/// - Magnitude: angle of rotation (radians)
#[derive(Clone, PartialEq)]
pub struct SO3Tangent {
    /// Internal data: axis-angle vector [θx, θy, θz]
    data: Vector3<f64>,
}

impl fmt::Display for SO3Tangent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "so3(axis-angle: [{:.4}, {:.4}, {:.4}])",
            self.data.x, self.data.y, self.data.z
        )
    }
}

// Conversion traits for integration with generic Problem
impl From<DVector<f64>> for SO3Tangent {
    fn from(data_vector: DVector<f64>) -> Self {
        if data_vector.len() != 3 {
            panic!("SO3Tangent::from expects 3-dimensional vector [θx, θy, θz]");
        }
        SO3Tangent {
            data: Vector3::new(data_vector[0], data_vector[1], data_vector[2]),
        }
    }
}

impl From<SO3Tangent> for DVector<f64> {
    fn from(so3_tangent: SO3Tangent) -> Self {
        DVector::from_vec(vec![
            so3_tangent.data.x,
            so3_tangent.data.y,
            so3_tangent.data.z,
        ])
    }
}

impl SO3Tangent {
    /// Create a new SO3Tangent from axis-angle vector.
    ///
    /// # Arguments
    /// * `axis_angle` - Axis-angle vector [θx, θy, θz]
    pub fn new(axis_angle: Vector3<f64>) -> Self {
        SO3Tangent { data: axis_angle }
    }

    /// Create SO3Tangent from individual components.
    pub fn from_components(x: f64, y: f64, z: f64) -> Self {
        SO3Tangent::new(Vector3::new(x, y, z))
    }

    /// Get the axis-angle vector.
    pub fn axis_angle(&self) -> Vector3<f64> {
        self.data
    }

    /// Get the angle of rotation.
    pub fn angle(&self) -> f64 {
        self.data.norm()
    }

    /// Get the axis of rotation (normalized).
    pub fn axis(&self) -> Vector3<f64> {
        let norm = self.data.norm();
        if norm < f64::EPSILON {
            Vector3::identity()
        } else {
            self.data / norm
        }
    }

    /// Get the x component.
    pub fn x(&self) -> f64 {
        self.data.x
    }

    /// Get the y component.
    pub fn y(&self) -> f64 {
        self.data.y
    }

    /// Get the z component.
    pub fn z(&self) -> f64 {
        self.data.z
    }

    /// Get the coefficients as a vector.
    pub fn coeffs(&self) -> Vector3<f64> {
        self.data
    }

    /// Get angular velocity representation (alias for axis_angle).
    pub fn ang(&self) -> Vector3<f64> {
        self.data
    }
}

// Implement LieAlgebra trait for SO3Tangent
impl Tangent<SO3> for SO3Tangent {
    /// Dimension of the tangent space
    const DIM: usize = 3;

    /// SO3 exponential map.
    ///
    /// # Arguments
    /// * `tangent` - Tangent vector [θx, θy, θz]
    /// * `jacobian` - Optional Jacobian matrix of the SO3 element wrt self.
    ///
    /// # Notes
    /// # Equation 132: Exponential map for unit quaternions (S³)
    /// q = Exp(θu) = cos(θ/2) + u sin(θ/2) ∈ H
    ///
    /// # Equation 143: Right Jacobian for SO(3) Exp map
    /// J_R(θ) = I - (1 - cos θ)/θ² [θ]ₓ + (θ - sin θ)/θ³ [θ]ₓ²
    ///
    fn exp(&self, jacobian: Option<&mut <SO3 as LieGroup>::JacobianMatrix>) -> SO3 {
        let theta_squared = self.data.norm_squared();

        let quaternion = if theta_squared > f64::EPSILON {
            UnitQuaternion::from_scaled_axis(self.data)
        } else {
            UnitQuaternion::from_quaternion(Quaternion::new(
                1.0,
                self.data.x / 2.0,
                self.data.y / 2.0,
                self.data.z / 2.0,
            ))
        };

        if let Some(jac) = jacobian {
            // Right Jacobian for SO(3)
            *jac = self.right_jacobian();
        }

        SO3 { quaternion }
    }

    /// Right Jacobian for SO(3)
    ///
    /// # Notes
    /// # Equation 143: Right Jacobian for SO(3) Exp map
    /// J_R(θ) = I - (1 - cos θ)/θ² [θ]ₓ + (θ - sin θ)/θ³ [θ]ₓ²
    ///
    fn right_jacobian(&self) -> <SO3 as LieGroup>::JacobianMatrix {
        self.left_jacobian().transpose()
    }

    /// Left Jacobian for SO(3)
    ///
    /// # Notes
    /// # Equation 144: Left Jacobian for SO(3) Exp map
    /// J_R⁻¹(θ) = I + (1 - cos θ)/θ² [θ]ₓ + (θ - sin θ)/θ³ [θ]ₓ²
    ///
    fn left_jacobian(&self) -> <SO3 as LieGroup>::JacobianMatrix {
        let angle = self.data.norm_squared();
        let tangent_skew = self.hat();

        if angle <= f64::EPSILON {
            Matrix3::identity() + 0.5 * tangent_skew
        } else {
            let theta = angle.sqrt(); // rotation angle
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            Matrix3::identity()
                + (1.0 - cos_theta) / angle * tangent_skew
                + (theta - sin_theta) / (angle * angle) * tangent_skew * tangent_skew
        }
    }

    /// Right Jacobian inverse for SO(3)
    ///
    /// # Notes
    /// # Equation 145: Right Jacobian inverse for SO(3) Exp map
    /// J_R⁻¹(θ) = I + (1 - cos θ)/θ² [θ]ₓ + (θ - sin θ)/θ³ [θ]ₓ²
    ///
    fn right_jacobian_inv(&self) -> <SO3 as LieGroup>::JacobianMatrix {
        self.left_jacobian_inv().transpose()
    }

    /// Left Jacobian inverse for SO(3)
    ///
    /// # Notes
    /// # Equation 146: Left Jacobian inverse for SO(3) Exp map
    /// J_R⁻¹(θ) = I - (1 - cos θ)/θ² [θ]ₓ + (θ - sin θ)/θ³ [θ]ₓ²
    ///
    fn left_jacobian_inv(&self) -> <SO3 as LieGroup>::JacobianMatrix {
        let angle = self.data.norm_squared();
        let tangent_skew = self.hat();

        if angle <= f64::EPSILON {
            Matrix3::identity() - 0.5 * tangent_skew
        } else {
            let theta = angle.sqrt(); // rotation angle
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            Matrix3::identity() - (0.5 * tangent_skew)
                + (1.0 / angle - (1.0 + cos_theta) / (2.0 * theta * sin_theta))
                    * tangent_skew
                    * tangent_skew
        }
    }

    /// Hat map for SO(3)
    ///
    /// # Notes
    /// [θ]ₓ = [0 -θz θy; θz 0 -θx; -θy θx 0]
    ///
    fn hat(&self) -> <SO3 as LieGroup>::LieAlgebra {
        Matrix3::new(
            0.0,
            -self.data.z,
            self.data.y,
            self.data.z,
            0.0,
            -self.data.x,
            -self.data.y,
            self.data.x,
            0.0,
        )
    }

    /// Zero tangent vector for SO(3)
    ///
    /// # Notes
    /// # Equation 147: Zero tangent vector for SO(3)
    /// [0, 0, 0]
    ///
    fn zero() -> <SO3 as LieGroup>::TangentVector {
        Self::new(Vector3::zeros())
    }

    /// Random tangent vector for SO(3)
    ///
    /// # Notes
    /// # Equation 147: Random tangent vector for SO(3)
    /// [0, 0, 0]
    ///
    fn random() -> <SO3 as LieGroup>::TangentVector {
        Self::new(Vector3::new(
            rand::random::<f64>() * 0.2 - 0.1,
            rand::random::<f64>() * 0.2 - 0.1,
            rand::random::<f64>() * 0.2 - 0.1,
        ))
    }

    /// Check if tangent vector is zero
    ///
    /// # Notes
    /// # Equation 147: Check if tangent vector is zero
    /// [0, 0, 0]
    ///
    fn is_zero(&self, tolerance: f64) -> bool {
        self.data.norm() < tolerance
    }

    /// Normalize tangent vector
    ///
    /// # Notes
    /// # Equation 147: Normalize tangent vector
    /// [0, 0, 0]
    ///
    fn normalize(&mut self) {
        let norm = self.data.norm();
        if norm > f64::EPSILON {
            self.data /= norm;
        }
    }

    fn normalized(&self) -> <SO3 as LieGroup>::TangentVector {
        let norm = self.data.norm();
        if norm > f64::EPSILON {
            SO3Tangent::new(self.data / norm)
        } else {
            Self::zero()
        }
    }

    /// Small adjoint matrix for SO(3).
    ///
    /// For SO(3), the small adjoint is the skew-symmetric matrix (hat operator).
    fn small_adj(&self) -> <SO3 as LieGroup>::JacobianMatrix {
        self.hat()
    }

    /// Lie bracket for SO(3).
    ///
    /// Computes the Lie bracket [this, other] = this.small_adj() * other.
    fn lie_bracket(&self, other: &Self) -> <SO3 as LieGroup>::TangentVector {
        let bracket_result = self.small_adj() * other.data;
        SO3Tangent::new(bracket_result)
    }

    /// Check if this tangent vector is approximately equal to another.
    ///
    /// # Arguments
    /// * `other` - The other tangent vector to compare with
    /// * `tolerance` - The tolerance for the comparison
    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        (self.data - other.data).norm() < tolerance
    }

    /// Get the ith generator of the SO(3) Lie algebra.
    ///
    /// # Arguments
    /// * `i` - Index of the generator (0, 1, or 2 for SO(3))
    ///
    /// # Returns
    /// The generator matrix
    fn generator(&self, i: usize) -> <SO3 as LieGroup>::LieAlgebra {
        assert!(i < 3, "SO(3) only has generators for indices 0, 1, 2");

        match i {
            0 => {
                // Generator E1 for x-axis rotation
                Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
            }
            1 => {
                // Generator E2 for y-axis rotation
                Matrix3::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0)
            }
            2 => {
                // Generator E3 for z-axis rotation
                Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64;
    use std::f64::consts::PI;

    const TOLERANCE: f64 = 1e-12;

    #[test]
    fn test_so3_constructor_datatype() {
        let so3 = SO3::from_quaternion_coeffs(0.0, 0.0, 0.0, 1.0);
        assert_eq!(0.0, so3.x());
        assert_eq!(0.0, so3.y());
        assert_eq!(0.0, so3.z());
        assert_eq!(1.0, so3.w());
    }

    #[test]
    fn test_so3_constructor_quat() {
        let quat = UnitQuaternion::identity();
        let so3 = SO3::new(quat);
        assert_eq!(0.0, so3.x());
        assert_eq!(0.0, so3.y());
        assert_eq!(0.0, so3.z());
        assert_eq!(1.0, so3.w());
    }

    #[test]
    fn test_so3_constructor_euler() {
        let so3 = SO3::from_euler_angles(0.0, 0.0, 0.0);
        assert_eq!(0.0, so3.x());
        assert_eq!(0.0, so3.y());
        assert_eq!(0.0, so3.z());
        assert_eq!(1.0, so3.w());
    }

    #[test]
    fn test_so3_identity() {
        let so3 = SO3::identity();
        assert_eq!(0.0, so3.x());
        assert_eq!(0.0, so3.y());
        assert_eq!(0.0, so3.z());
        assert_eq!(1.0, so3.w());
    }

    #[test]
    fn test_so3_coeffs() {
        // Create from normalized coefficients
        let so3 = SO3::from_quaternion_coeffs(0.0, 0.0, 0.0, 1.0);
        let coeffs = so3.coeffs();
        assert!((coeffs[0] - 1.0).abs() < TOLERANCE); // w
        assert!((coeffs[1] - 0.0).abs() < TOLERANCE); // x
        assert!((coeffs[2] - 0.0).abs() < TOLERANCE); // y
        assert!((coeffs[3] - 0.0).abs() < TOLERANCE); // z

        // Test with non-normalized input - should get normalized output
        let so3 = SO3::from_quaternion_coeffs(0.1, 0.2, 0.3, 0.4);
        let coeffs = so3.coeffs();
        let original_quat = Quaternion::new(0.4, 0.1, 0.2, 0.3);
        let normalized_quat = original_quat.normalize();
        assert!((coeffs[0] - normalized_quat.w).abs() < TOLERANCE);
        assert!((coeffs[1] - normalized_quat.i).abs() < TOLERANCE);
        assert!((coeffs[2] - normalized_quat.j).abs() < TOLERANCE);
        assert!((coeffs[3] - normalized_quat.k).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_random() {
        let so3 = SO3::random();
        assert!((so3.quaternion().norm() - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_transform() {
        let so3 = SO3::identity();
        let transform = so3.transform();

        assert_eq!(4, transform.nrows());
        assert_eq!(4, transform.ncols());

        // Check identity transform
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert!((transform[(i, j)] - 1.0).abs() < TOLERANCE);
                } else {
                    assert!(transform[(i, j)].abs() < TOLERANCE);
                }
            }
        }
    }

    #[test]
    fn test_so3_rotation() {
        let so3 = SO3::identity();
        let rotation = so3.rotation_matrix();

        assert_eq!(3, rotation.nrows());
        assert_eq!(3, rotation.ncols());

        // Check identity rotation
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((rotation[(i, j)] - 1.0).abs() < TOLERANCE);
                } else {
                    assert!(rotation[(i, j)].abs() < TOLERANCE);
                }
            }
        }
    }

    #[test]
    fn test_so3_inverse() {
        // inverse of identity is identity
        let so3 = SO3::identity();
        let so3_inv = so3.inverse(None);
        assert_eq!(0.0, so3_inv.x());
        assert_eq!(0.0, so3_inv.y());
        assert_eq!(0.0, so3_inv.z());
        assert_eq!(1.0, so3_inv.w());

        // inverse of random in quaternion form is conjugate
        let so3 = SO3::random();
        let so3_inv = so3.inverse(None);
        assert!((so3.x() + so3_inv.x()).abs() < TOLERANCE);
        assert!((so3.y() + so3_inv.y()).abs() < TOLERANCE);
        assert!((so3.z() + so3_inv.z()).abs() < TOLERANCE);
        assert!((so3.w() - so3_inv.w()).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_inverse_jacobian() {
        let so3 = SO3::identity();
        let mut jacobian = Matrix3::zeros();
        let so3_inv = so3.inverse(Some(&mut jacobian));

        // Check result
        assert_eq!(0.0, so3_inv.x());
        assert_eq!(0.0, so3_inv.y());
        assert_eq!(0.0, so3_inv.z());
        assert_eq!(1.0, so3_inv.w());

        // Check Jacobian is negative identity
        let expected_jac = -Matrix3::identity();
        assert!((jacobian - expected_jac).norm() < TOLERANCE);
    }

    #[test]
    fn test_so3_rplus() {
        // Adding zero to identity
        let so3a = SO3::identity();
        let so3b = SO3Tangent::new(Vector3::zeros());
        let so3c = so3a.right_plus(&so3b, None, None);
        assert_eq!(0.0, so3c.x());
        assert_eq!(0.0, so3c.y());
        assert_eq!(0.0, so3c.z());
        assert_eq!(1.0, so3c.w());

        // Adding zero to random
        let so3a = SO3::random();
        let so3c = so3a.right_plus(&so3b, None, None);
        assert!((so3a.x() - so3c.x()).abs() < TOLERANCE);
        assert!((so3a.y() - so3c.y()).abs() < TOLERANCE);
        assert!((so3a.z() - so3c.z()).abs() < TOLERANCE);
        assert!((so3a.w() - so3c.w()).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_lplus() {
        // Adding zero to identity
        let so3a = SO3::identity();
        let so3t = SO3Tangent::new(Vector3::zeros());
        let so3c = so3a.left_plus(&so3t, None, None);
        assert_eq!(0.0, so3c.x());
        assert_eq!(0.0, so3c.y());
        assert_eq!(0.0, so3c.z());
        assert_eq!(1.0, so3c.w());

        // Adding zero to random
        let so3a = SO3::random();
        let so3c = so3a.left_plus(&so3t, None, None);
        assert!((so3a.x() - so3c.x()).abs() < TOLERANCE);
        assert!((so3a.y() - so3c.y()).abs() < TOLERANCE);
        assert!((so3a.z() - so3c.z()).abs() < TOLERANCE);
        assert!((so3a.w() - so3c.w()).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_rminus() {
        // identity minus identity is zero
        let so3a = SO3::identity();
        let so3b = SO3::identity();
        let so3c = so3a.right_minus(&so3b, None, None);
        assert!(so3c.x().abs() < TOLERANCE);
        assert!(so3c.y().abs() < TOLERANCE);
        assert!(so3c.z().abs() < TOLERANCE);

        // random minus the same is zero
        let so3a = SO3::random();
        let so3b = so3a.clone();
        let so3c = so3a.right_minus(&so3b, None, None);
        assert!(so3c.data.norm() < TOLERANCE);
    }

    #[test]
    fn test_so3_minus() {
        // minus is the same as right_minus
        let so3a = SO3::random();
        let so3b = SO3::random();
        let so3c = so3a.minus(&so3b, None, None);
        let so3d = so3a.right_minus(&so3b, None, None);
        assert!((so3c.data - so3d.data).norm() < TOLERANCE);
    }

    #[test]
    fn test_so3_exp_log() {
        // exp of zero is identity
        let so3t = SO3Tangent::new(Vector3::zeros());
        let so3 = so3t.exp(None);
        assert_eq!(0.0, so3.x());
        assert_eq!(0.0, so3.y());
        assert_eq!(0.0, so3.z());
        assert_eq!(1.0, so3.w());

        // exp of negative is inverse of exp
        let so3t = SO3Tangent::new(Vector3::new(0.1, 0.2, 0.3));
        let so3 = so3t.exp(None);
        let so3n = SO3Tangent::new(Vector3::new(-0.1, -0.2, -0.3));
        let so3_inv = so3n.exp(None);
        assert!((so3_inv.x() + so3.x()).abs() < TOLERANCE);
        assert!((so3_inv.y() + so3.y()).abs() < TOLERANCE);
        assert!((so3_inv.z() + so3.z()).abs() < TOLERANCE);
        assert!((so3_inv.w() - so3.w()).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_log() {
        // log of identity is zero
        let so3 = SO3::identity();
        let so3_log = so3.log(None);
        assert!(so3_log.x().abs() < TOLERANCE);
        assert!(so3_log.y().abs() < TOLERANCE);
        assert!(so3_log.z().abs() < TOLERANCE);

        // log of inverse is negative log
        let so3 = SO3::random();
        let so3_log = so3.log(None);
        let so3_inv_log = so3.inverse(None).log(None);
        assert!((so3_inv_log.x() + so3_log.x()).abs() < TOLERANCE);
        assert!((so3_inv_log.y() + so3_log.y()).abs() < TOLERANCE);
        assert!((so3_inv_log.z() + so3_log.z()).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_tangent_hat() {
        let so3_tan = SO3Tangent::new(Vector3::new(1.0, 2.0, 3.0));
        let so3_lie = so3_tan.hat();

        assert!((so3_lie[(0, 0)] - 0.0).abs() < TOLERANCE);
        assert!((so3_lie[(0, 1)] + 3.0).abs() < TOLERANCE);
        assert!((so3_lie[(0, 2)] - 2.0).abs() < TOLERANCE);
        assert!((so3_lie[(1, 0)] - 3.0).abs() < TOLERANCE);
        assert!((so3_lie[(1, 1)] - 0.0).abs() < TOLERANCE);
        assert!((so3_lie[(1, 2)] + 1.0).abs() < TOLERANCE);
        assert!((so3_lie[(2, 0)] + 2.0).abs() < TOLERANCE);
        assert!((so3_lie[(2, 1)] - 1.0).abs() < TOLERANCE);
        assert!((so3_lie[(2, 2)] - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_act() {
        let so3 = SO3::identity();
        let transformed_point = so3.act(&Vector3::new(1.0, 1.0, 1.0), None, None);
        assert!((transformed_point.x - 1.0).abs() < TOLERANCE);
        assert!((transformed_point.y - 1.0).abs() < TOLERANCE);
        assert!((transformed_point.z - 1.0).abs() < TOLERANCE);

        let so3 = SO3::from_euler_angles(PI, PI / 2.0, PI / 4.0);
        let transformed_point = so3.act(&Vector3::new(1.0, 1.0, 1.0), None, None);
        assert!((transformed_point.x - 0.0).abs() < TOLERANCE);
        assert!((transformed_point.y + f64::consts::SQRT_2).abs() < 1e-10);
        assert!((transformed_point.z + 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_tangent_angular_velocity() {
        let so3tan = SO3Tangent::new(Vector3::new(1.0, 2.0, 3.0));
        let ang_vel = so3tan.ang();
        assert!((ang_vel - Vector3::new(1.0, 2.0, 3.0)).norm() < TOLERANCE);
    }

    #[test]
    fn test_so3_compose() {
        let so3_1 = SO3::random();
        let so3_2 = SO3::random();
        let composed = so3_1.compose(&so3_2, None, None);
        assert!(composed.is_valid(TOLERANCE));

        // Test composition with identity
        let identity = SO3::identity();
        let composed_with_identity = so3_1.compose(&identity, None, None);
        assert!((composed_with_identity.distance(&so3_1)).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_exp_log_consistency() {
        let tangent = SO3Tangent::new(Vector3::new(0.1, 0.2, 0.3));
        let so3 = tangent.exp(None);
        let recovered_tangent = so3.log(None);
        assert!((tangent.data - recovered_tangent.data).norm() < TOLERANCE);
    }

    #[test]
    fn test_so3_right_left_jacobian_relationship() {
        // For zero tangent, left and right Jacobians should be equal (both identity)
        let tangent = SO3Tangent::new(Vector3::zeros());
        let ljac = tangent.left_jacobian();
        let rjac = tangent.right_jacobian();
        assert!((ljac - rjac).norm() < TOLERANCE);
        assert!((ljac - Matrix3::identity()).norm() < TOLERANCE);

        // For non-zero tangent, test the general relationship
        let tangent = SO3Tangent::new(Vector3::new(0.1, 0.2, 0.3));
        let ljac = tangent.left_jacobian();
        let rjac = tangent.right_jacobian();

        // The correct relationship for SO(3) should be that both are transposes
        // when the tangent is small enough
        assert!((ljac - rjac.transpose()).norm() < TOLERANCE);
        assert!((rjac - ljac.transpose()).norm() < TOLERANCE);
    }

    #[test]
    fn test_so3_manifold_properties() {
        assert_eq!(SO3::DIM, 3);
        assert_eq!(SO3::DOF, 3);
        assert_eq!(SO3::REP_SIZE, 4);
    }

    #[test]
    fn test_so3_normalize() {
        let mut so3 = SO3::from_quaternion_coeffs(0.5, 0.5, 0.5, 0.5);
        so3.normalize();
        assert!(so3.is_valid(TOLERANCE));
    }

    #[test]
    fn test_so3_tangent_norms() {
        let tangent = SO3Tangent::new(Vector3::new(3.0, 4.0, 0.0));
        let norm = tangent.data.norm();
        assert!((norm - 5.0).abs() < TOLERANCE);

        let squared_norm = tangent.data.norm_squared();
        assert!((squared_norm - 25.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_tangent_zero() {
        let zero = SO3Tangent::zero();
        assert!(zero.data.norm() < TOLERANCE);

        let tangent = SO3Tangent::new(Vector3::zeros());
        assert!(tangent.is_zero(TOLERANCE));
    }

    #[test]
    fn test_so3_tangent_normalize() {
        let mut tangent = SO3Tangent::new(Vector3::new(3.0, 4.0, 0.0));
        tangent.normalize();
        assert!((tangent.data.norm() - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_adjoint() {
        let so3 = SO3::random();
        let adj = so3.adjoint();
        assert_eq!(adj.nrows(), 3);
        assert_eq!(adj.ncols(), 3);

        // For SO(3), adjoint is the rotation matrix, so det should be 1
        let det = adj.determinant();
        assert!((det - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_small_angle_approximations() {
        let small_tangent = SO3Tangent::new(Vector3::new(1e-8, 2e-8, 3e-8));
        let so3 = small_tangent.exp(None);
        let recovered = so3.log(None);
        assert!((small_tangent.data - recovered.data).norm() < TOLERANCE);
    }

    #[test]
    fn test_so3_specific_rotations() {
        // Test rotation around X axis
        let so3_x = SO3::from_axis_angle(&Vector3::x(), PI / 2.0);
        let point_y = Vector3::y();
        let rotated = so3_x.act(&point_y, None, None);
        assert!((rotated - Vector3::z()).norm() < TOLERANCE);

        // Test rotation around Z axis
        let so3_z = SO3::from_axis_angle(&Vector3::z(), PI / 2.0);
        let point_x = Vector3::x();
        let rotated = so3_z.act(&point_x, None, None);
        assert!((rotated - Vector3::y()).norm() < TOLERANCE);
    }

    #[test]
    fn test_so3_from_components() {
        let so3 = SO3::from_quaternion_coeffs(0.0, 0.0, 0.0, 1.0);
        assert_eq!(so3.x(), 0.0);
        assert_eq!(so3.y(), 0.0);
        assert_eq!(so3.z(), 0.0);
        assert_eq!(so3.w(), 1.0);
    }

    #[test]
    fn test_so3_tangent_from_components() {
        let tangent = SO3Tangent::from_components(1.0, 2.0, 3.0);
        assert_eq!(tangent.x(), 1.0);
        assert_eq!(tangent.y(), 2.0);
        assert_eq!(tangent.z(), 3.0);
    }

    #[test]
    fn test_so3_consistency_with_manif() {
        // Test that operations are consistent with manif library expectations
        let so3_1 = SO3::random();
        let so3_2 = SO3::random();

        // Test associativity: (R1 * R2) * R3 = R1 * (R2 * R3)
        let so3_3 = SO3::random();
        let left_assoc = so3_1
            .compose(&so3_2, None, None)
            .compose(&so3_3, None, None);
        let right_assoc = so3_1.compose(&so3_2.compose(&so3_3, None, None), None, None);

        assert!(left_assoc.distance(&right_assoc) < 1e-10);
    }

    #[test]
    fn test_so3_tangent_accessors() {
        let tangent = SO3Tangent::new(Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(tangent.x(), 1.0);
        assert_eq!(tangent.y(), 2.0);
        assert_eq!(tangent.z(), 3.0);

        let coeffs = tangent.coeffs();
        assert_eq!(coeffs, Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_so3_between() {
        let so3_1 = SO3::random();
        let so3_2 = SO3::random();
        let between = so3_1.between(&so3_2, None, None);

        // Check that so3_1 * between = so3_2
        let result = so3_1.compose(&between, None, None);
        assert!(result.distance(&so3_2) < TOLERANCE);
    }

    #[test]
    fn test_so3_distance() {
        let so3_1 = SO3::random();
        let so3_2 = SO3::random();
        let distance = so3_1.distance(&so3_2);
        assert!(distance >= 0.0);
        assert!(so3_1.distance(&so3_1) < TOLERANCE);
    }

    // New tests for the additional functions

    #[test]
    fn test_so3_vee() {
        let so3 = SO3::random();
        let tangent_log = so3.log(None);
        let tangent_vee = so3.vee();

        assert!((tangent_log.data - tangent_vee.data).norm() < 1e-10);
    }

    #[test]
    fn test_so3_is_approx() {
        let so3_1 = SO3::random();
        let so3_2 = so3_1.clone();

        assert!(so3_1.is_approx(&so3_1, 1e-10));
        assert!(so3_1.is_approx(&so3_2, 1e-10));

        // Test with small perturbation
        let small_tangent = SO3Tangent::new(Vector3::new(1e-12, 1e-12, 1e-12));
        let so3_perturbed = so3_1.right_plus(&small_tangent, None, None);
        assert!(so3_1.is_approx(&so3_perturbed, 1e-10));
    }

    #[test]
    fn test_so3_tangent_small_adj() {
        let axis_angle = Vector3::new(0.1, 0.2, 0.3);
        let tangent = SO3Tangent::new(axis_angle);
        let small_adj = tangent.small_adj();
        let hat_matrix = tangent.hat();

        // For SO(3), small adjoint equals hat matrix
        assert!((small_adj - hat_matrix).norm() < 1e-10);
    }

    #[test]
    fn test_so3_tangent_lie_bracket() {
        let tangent_a = SO3Tangent::new(Vector3::new(0.1, 0.0, 0.0));
        let tangent_b = SO3Tangent::new(Vector3::new(0.0, 0.2, 0.0));

        let bracket_ab = tangent_a.lie_bracket(&tangent_b);
        let bracket_ba = tangent_b.lie_bracket(&tangent_a);

        // Anti-symmetry test: [a,b] = -[b,a]
        assert!((bracket_ab.data + bracket_ba.data).norm() < 1e-10);

        // [a,a] = 0
        let bracket_aa = tangent_a.lie_bracket(&tangent_a);
        assert!(bracket_aa.is_zero(1e-10));

        // Verify bracket relationship with hat operator
        let bracket_hat = bracket_ab.hat();
        let expected = tangent_a.hat() * tangent_b.hat() - tangent_b.hat() * tangent_a.hat();
        assert!((bracket_hat - expected).norm() < 1e-10);
    }

    #[test]
    fn test_so3_tangent_is_approx() {
        let tangent_1 = SO3Tangent::new(Vector3::new(0.1, 0.2, 0.3));
        let tangent_2 = SO3Tangent::new(Vector3::new(0.1 + 1e-12, 0.2, 0.3));
        let tangent_3 = SO3Tangent::new(Vector3::new(0.5, 0.6, 0.7));

        assert!(tangent_1.is_approx(&tangent_1, 1e-10));
        assert!(tangent_1.is_approx(&tangent_2, 1e-10));
        assert!(!tangent_1.is_approx(&tangent_3, 1e-10));
    }

    #[test]
    fn test_so3_generators() {
        let tangent = SO3Tangent::new(Vector3::new(1.0, 1.0, 1.0));

        // Test all three generators
        for i in 0..3 {
            let generator = tangent.generator(i);

            // Generator should be skew-symmetric
            assert!((generator + generator.transpose()).norm() < 1e-10);

            // Generator should have trace zero
            assert!(generator.trace().abs() < 1e-10);
        }

        // Test specific values for the generators
        let e1 = tangent.generator(0);
        let e2 = tangent.generator(1);
        let e3 = tangent.generator(2);

        // Expected generators based on C++ manif implementation
        let expected_e1 = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
        let expected_e2 = Matrix3::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0);
        let expected_e3 = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        assert!((e1 - expected_e1).norm() < 1e-10);
        assert!((e2 - expected_e2).norm() < 1e-10);
        assert!((e3 - expected_e3).norm() < 1e-10);
    }

    #[test]
    #[should_panic]
    fn test_so3_generator_invalid_index() {
        let tangent = SO3Tangent::new(Vector3::new(1.0, 1.0, 1.0));
        let _generator = tangent.generator(3); // Should panic for SO(3)
    }

    #[test]
    fn test_so3_jacobi_identity() {
        // Test Jacobi identity: [x,[y,z]]+[y,[z,x]]+[z,[x,y]]=0
        let x = SO3Tangent::new(Vector3::new(0.1, 0.0, 0.0));
        let y = SO3Tangent::new(Vector3::new(0.0, 0.2, 0.0));
        let z = SO3Tangent::new(Vector3::new(0.0, 0.0, 0.3));

        let term1 = x.lie_bracket(&y.lie_bracket(&z));
        let term2 = y.lie_bracket(&z.lie_bracket(&x));
        let term3 = z.lie_bracket(&x.lie_bracket(&y));

        let jacobi_sum = SO3Tangent::new(term1.data + term2.data + term3.data);
        assert!(jacobi_sum.is_zero(1e-10));
    }
}
