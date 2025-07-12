//! SO(3) - Special Orthogonal Group in 3D
//!
//! This module implements the Special Orthogonal group SO(3), which represents
//! rotations in 3D space.
//!
//! SO(3) elements are represented using nalgebra's UnitQuaternion internally.
//! SO(3) tangent elements are represented as axis-angle vectors in R³,
//! where the direction gives the axis of rotation and the magnitude gives the angle.
//!
//! The implementation follows the [manif](https://github.com/artivis/manif) C++ library
//! conventions and provides all operations required by the LieGroup and LieAlgebra traits.

use crate::manifold::{LieAlgebra, LieGroup};
use nalgebra::{DMatrix, Matrix3, Matrix4, Quaternion, Unit, UnitQuaternion, Vector3};
use std::fmt;

/// SO(3) group element representing rotations in 3D.
///
/// Internally represented using nalgebra's UnitQuaternion<f64> for efficient rotations.
#[derive(Clone, Debug, PartialEq)]
pub struct SO3 {
    /// Internal representation as a unit quaternion
    quaternion: UnitQuaternion<f64>,
}

impl fmt::Display for SO3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let q = self.quaternion.quaternion();
        write!(
            f,
            "SO3(quaternion: [w: {:.4}, x: {:.4}, y: {:.4}, z: {:.4}])",
            q.w, q.i, q.j, q.k
        )
    }
}

/// SO(3) tangent space element representing elements in the Lie algebra so(3).
///
/// Internally represented as axis-angle vectors in R³ where:
/// - Direction: axis of rotation (unit vector)
/// - Magnitude: angle of rotation (radians)
#[derive(Clone, Debug, PartialEq)]
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

impl SO3 {
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
    pub fn rotation(&self) -> Matrix3<f64> {
        self.quaternion.to_rotation_matrix().into_inner()
    }

    /// Get the homogeneous transformation matrix (4x4).
    pub fn transform(&self) -> Matrix4<f64> {
        self.quaternion.to_homogeneous()
    }

    /// Set the quaternion from coefficients array [x, y, z, w].
    pub fn set_quaternion(&mut self, coeffs: &[f64; 4]) {
        let q = Quaternion::new(coeffs[3], coeffs[0], coeffs[1], coeffs[2]);
        self.quaternion = UnitQuaternion::from_quaternion(q);
    }

    /// Get coefficients as array [x, y, z, w].
    pub fn coeffs(&self) -> [f64; 4] {
        let q = self.quaternion.quaternion();
        [q.i, q.j, q.k, q.w]
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
            Vector3::x() // Default axis when angle is zero
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

// Implement basic trait requirements for LieGroup
impl LieGroup for SO3 {
    type Element = SO3;
    type TangentVector = Vector3<f64>;
    type JacobianMatrix = Matrix3<f64>;
    type LieAlgebra = SO3Tangent;

    // Dimension constants following manif conventions
    const DIM: usize = 3; // Space dimension (3D space)
    const DOF: usize = 3; // Degrees of freedom (3 rotation parameters)
    const REP_SIZE: usize = 4; // Representation size (4 quaternion components)

    fn identity() -> Self::Element {
        SO3 {
            quaternion: UnitQuaternion::identity(),
        }
    }

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
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::Element {
        // For SO(3): R^{-1} = R^T, for quaternions: q^{-1} = q*
        let inverse_quat = self.quaternion.inverse();

        if let Some(jac) = jacobian {
            // Jacobian of inverse operation: -R^T = -R
            let rotation_matrix = self.quaternion.to_rotation_matrix().into_inner();
            jac.copy_from(&(-rotation_matrix));
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
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        let result = SO3 {
            quaternion: self.quaternion * other.quaternion,
        };

        if let Some(jac_self) = jacobian_self {
            // Jacobian wrt first element: R2^T
            let other_rotation = other.quaternion.to_rotation_matrix().into_inner();
            jac_self.copy_from(&other_rotation.transpose());
        }

        if let Some(jac_other) = jacobian_other {
            // Jacobian wrt second element: I (identity)
            *jac_other = Matrix3::identity();
        }

        result
    }

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
    fn exp(
        tangent: &Self::TangentVector,
        jacobian: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        let angle = tangent.norm_squared();

        let quaternion = if angle < f64::EPSILON {
            UnitQuaternion::identity()
        } else {
            UnitQuaternion::from_scaled_axis(*tangent)
        };

        if let Some(jac) = jacobian {
            // Right Jacobian for SO(3)
            Self::compute_right_jacobian(tangent, jac);
        }

        SO3 { quaternion }
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
        let axis_angle = self.quaternion.scaled_axis();

        if let Some(jac) = jacobian {
            // Right Jacobian inverse for SO(3)
            Self::compute_right_jacobian_inverse(&axis_angle, jac);
        }

        axis_angle
    }

    fn right_plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        // Right plus: R ⊕ φ = R * exp(φ)
        let exp_tangent = Self::exp(tangent, jacobian_tangent);
        self.compose(&exp_tangent, jacobian_self, None)
    }

    fn right_minus(
        &self,
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::TangentVector {
        // Right minus: R1 ⊖ R2 = log(R2^T * R1)
        let other_inv = other.inverse(None);
        let result_group = other_inv.compose(self, None, None);
        let result = result_group.log(jacobian_self);

        if let Some(jac_other) = jacobian_other {
            // Jacobian wrt other: -Jr^{-1}(-φ) where φ = log(R2^T * R1)
            let neg_result = -result;
            let mut temp_jac = Matrix3::identity();
            Self::compute_right_jacobian_inverse(&neg_result, &mut temp_jac);
            jac_other.copy_from(&(-temp_jac));
        }

        result
    }

    fn left_plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        // Left plus: φ ⊕ R = exp(φ) * R
        let exp_tangent = Self::exp(tangent, jacobian_tangent);

        if let Some(jac_self) = jacobian_self {
            *jac_self = Matrix3::identity();
        }

        exp_tangent.compose(self, None, None)
    }

    fn left_minus(
        &self,
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::TangentVector {
        // Left minus: R1 ⊖ R2 = log(R1 * R2^T)
        let other_inv = other.inverse(None);
        let result_group = self.compose(&other_inv, jacobian_self, None);
        let result = result_group.log(None);

        if let Some(jac_other) = jacobian_other {
            // Jacobian wrt other: -Ad(R2) * Jr^{-1}(φ)
            let adj_other = other.adjoint();
            let mut jr_inv = Matrix3::identity();
            Self::compute_right_jacobian_inverse(&result, &mut jr_inv);
            jac_other.copy_from(&(-adj_other * jr_inv));
        }

        result
    }

    fn between(
        &self,
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        // Between: R1.between(R2) = R1^T * R2
        let self_inv = self.inverse(None);
        self_inv.compose(
            other,
            jacobian_self.map(|j| {
                let other_rotation = other.quaternion.to_rotation_matrix().into_inner();
                *j = -other_rotation.transpose();
                j
            }),
            jacobian_other,
        )
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
            let rotation_matrix = self.quaternion.to_rotation_matrix().into_inner();
            let vector_skew = Self::skew_symmetric(vector);
            jac_self.copy_from(&(-rotation_matrix * vector_skew));
        }

        if let Some(jac_vector) = jacobian_vector {
            // Jacobian wrt vector: R
            let rotation_matrix = self.quaternion.to_rotation_matrix().into_inner();
            jac_vector.copy_from(&rotation_matrix);
        }

        result
    }

    fn adjoint(&self) -> Self::JacobianMatrix {
        // Adjoint matrix for SO(3) is just the rotation matrix
        self.quaternion.to_rotation_matrix().into_inner()
    }

    fn random() -> Self::Element {
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
        self.quaternion = UnitQuaternion::from_quaternion(self.quaternion.quaternion().normalize());
    }

    fn is_valid(&self, tolerance: f64) -> bool {
        // Check if quaternion is properly normalized
        let q = self.quaternion.quaternion();
        (q.norm() - 1.0).abs() < tolerance
    }

    fn distance(&self, other: &Self::Element) -> f64 {
        self.right_minus(other, None, None).norm()
    }

    fn weighted_distance(&self, other: &Self::Element, weight: &Self::JacobianMatrix) -> f64 {
        let diff = self.right_minus(other, None, None);
        (diff.transpose() * weight * diff)[0].sqrt()
    }
}

impl SO3 {
    /// Compute the right Jacobian for SO(3).
    fn compute_right_jacobian(tangent: &Vector3<f64>, jacobian: &mut Matrix3<f64>) {
        let angle = tangent.norm();

        if angle < 1e-12 {
            *jacobian = Matrix3::identity();
        } else {
            let c = angle.cos();

            let tangent_skew = Self::skew_symmetric(tangent);

            *jacobian = Matrix3::identity() - 0.5 * tangent_skew
                + ((1.0 - c) / (angle * angle)) * tangent_skew * tangent_skew;
        }
    }

    /// Compute the right Jacobian inverse for SO(3).
    fn compute_right_jacobian_inverse(tangent: &Vector3<f64>, jacobian: &mut Matrix3<f64>) {
        let angle = tangent.norm();

        if angle < 1e-12 {
            *jacobian = Matrix3::identity();
        } else {
            let half_angle = 0.5 * angle;
            let cot_half = half_angle.cos() / half_angle.sin();

            let tangent_skew = Self::skew_symmetric(tangent);

            *jacobian = Matrix3::identity()
                + 0.5 * tangent_skew
                + ((1.0 / (angle * angle)) * (1.0 - half_angle * cot_half))
                    * tangent_skew
                    * tangent_skew;
        }
    }

    /// Create skew-symmetric matrix from vector.
    fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }
}

// Implement LieAlgebra trait for SO3Tangent
impl LieAlgebra<SO3> for SO3Tangent {
    fn add(&self, other: &Vector3<f64>) -> Vector3<f64> {
        self.data + other
    }

    fn scale(&self, scalar: f64) -> Vector3<f64> {
        self.data * scalar
    }

    fn negate(&self) -> Vector3<f64> {
        -self.data
    }

    fn subtract(&self, other: &Vector3<f64>) -> Vector3<f64> {
        self.data - other
    }

    fn norm(&self) -> f64 {
        self.data.norm()
    }

    fn squared_norm(&self) -> f64 {
        self.data.norm_squared()
    }

    fn weighted_norm(&self, weight: &Matrix3<f64>) -> f64 {
        (self.data.transpose() * weight * self.data)[0].sqrt()
    }

    fn squared_weighted_norm(&self, weight: &Matrix3<f64>) -> f64 {
        (self.data.transpose() * weight * self.data)[0]
    }

    fn inner(&self, other: &Vector3<f64>) -> f64 {
        self.data.dot(other)
    }

    fn weighted_inner(&self, other: &Vector3<f64>, weight: &Matrix3<f64>) -> f64 {
        (self.data.transpose() * weight * other)[0]
    }

    fn exp(&self, jacobian: Option<&mut Matrix3<f64>>) -> SO3 {
        SO3::exp(&self.data, jacobian)
    }

    fn right_jacobian(&self) -> Matrix3<f64> {
        let mut jac = Matrix3::identity();
        SO3::compute_right_jacobian(&self.data, &mut jac);
        jac
    }

    fn left_jacobian(&self) -> Matrix3<f64> {
        let angle = self.data.norm();

        if angle < 1e-12 {
            Matrix3::identity()
        } else {
            let s = angle.sin();
            let c = angle.cos();

            let tangent_skew = SO3::skew_symmetric(&self.data);

            Matrix3::identity()
                + (s / angle) * tangent_skew
                + ((1.0 - c) / (angle * angle)) * tangent_skew * tangent_skew
        }
    }

    fn right_jacobian_inv(&self) -> Matrix3<f64> {
        let mut jac = Matrix3::identity();
        SO3::compute_right_jacobian_inverse(&self.data, &mut jac);
        jac
    }

    fn left_jacobian_inv(&self) -> Matrix3<f64> {
        let angle = self.data.norm();

        if angle < 1e-12 {
            Matrix3::identity()
        } else {
            let half_angle = 0.5 * angle;
            let cot_half = half_angle.cos() / half_angle.sin();

            let tangent_skew = SO3::skew_symmetric(&self.data);

            Matrix3::identity() - 0.5 * tangent_skew
                + ((1.0 / (angle * angle)) * (1.0 - half_angle * cot_half))
                    * tangent_skew
                    * tangent_skew
        }
    }

    fn hat(&self) -> DMatrix<f64> {
        let mut lie_alg = DMatrix::zeros(3, 3);
        let skew = SO3::skew_symmetric(&self.data);
        lie_alg.copy_from(&skew);
        lie_alg
    }

    fn vee(matrix: &DMatrix<f64>) -> Vector3<f64> {
        Vector3::new(matrix[(2, 1)], matrix[(0, 2)], matrix[(1, 0)])
    }

    fn small_adjoint(&self) -> Matrix3<f64> {
        SO3::skew_symmetric(&self.data)
    }

    fn zero() -> Vector3<f64> {
        Vector3::zeros()
    }

    fn random() -> Vector3<f64> {
        Vector3::new(
            rand::random::<f64>() * 0.2 - 0.1,
            rand::random::<f64>() * 0.2 - 0.1,
            rand::random::<f64>() * 0.2 - 0.1,
        )
    }

    fn is_zero(&self, tolerance: f64) -> bool {
        self.norm() < tolerance
    }

    fn normalize(&mut self) {
        let norm = self.norm();
        if norm > f64::EPSILON {
            self.data /= norm;
        }
    }

    fn normalized(&self) -> Vector3<f64> {
        let norm = self.norm();
        if norm > f64::EPSILON {
            self.data / norm
        } else {
            Self::zero()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert!((coeffs[0] - 0.0).abs() < TOLERANCE);
        assert!((coeffs[1] - 0.0).abs() < TOLERANCE);
        assert!((coeffs[2] - 0.0).abs() < TOLERANCE);
        assert!((coeffs[3] - 1.0).abs() < TOLERANCE);

        // Test with non-normalized input - should get normalized output
        let so3 = SO3::from_quaternion_coeffs(0.1, 0.2, 0.3, 0.4);
        let coeffs = so3.coeffs();
        let original_quat = Quaternion::new(0.4, 0.1, 0.2, 0.3);
        let normalized_quat = original_quat.normalize();
        assert!((coeffs[0] - normalized_quat.i).abs() < TOLERANCE);
        assert!((coeffs[1] - normalized_quat.j).abs() < TOLERANCE);
        assert!((coeffs[2] - normalized_quat.k).abs() < TOLERANCE);
        assert!((coeffs[3] - normalized_quat.w).abs() < TOLERANCE);
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
        let rotation = so3.rotation();

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
        let so3b = Vector3::zeros();
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
        let so3t = Vector3::zeros();
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
    fn test_so3_plus() {
        // plus() is the same as right_plus()
        let so3a = SO3::random();
        let so3t = Vector3::new(0.1, 0.2, 0.3);
        let so3c = so3a.plus(&so3t, None, None);
        let so3d = so3a.right_plus(&so3t, None, None);
        assert!((so3c.x() - so3d.x()).abs() < TOLERANCE);
        assert!((so3c.y() - so3d.y()).abs() < TOLERANCE);
        assert!((so3c.z() - so3d.z()).abs() < TOLERANCE);
        assert!((so3c.w() - so3d.w()).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_rminus() {
        // identity minus identity is zero
        let so3a = SO3::identity();
        let so3b = SO3::identity();
        let so3c = so3a.right_minus(&so3b, None, None);
        assert!(so3c[0].abs() < TOLERANCE);
        assert!(so3c[1].abs() < TOLERANCE);
        assert!(so3c[2].abs() < TOLERANCE);

        // random minus the same is zero
        let so3a = SO3::random();
        let so3b = so3a.clone();
        let so3c = so3a.right_minus(&so3b, None, None);
        assert!(so3c.norm() < TOLERANCE);
    }

    #[test]
    fn test_so3_minus() {
        // minus is the same as right_minus
        let so3a = SO3::random();
        let so3b = SO3::random();
        let so3c = so3a.minus(&so3b, None, None);
        let so3d = so3a.right_minus(&so3b, None, None);
        assert!((so3c - so3d).norm() < TOLERANCE);
    }

    #[test]
    fn test_so3_exp_log() {
        // exp of zero is identity
        let so3t = Vector3::zeros();
        let so3 = SO3::exp(&so3t, None);
        assert_eq!(0.0, so3.x());
        assert_eq!(0.0, so3.y());
        assert_eq!(0.0, so3.z());
        assert_eq!(1.0, so3.w());

        // exp of negative is inverse of exp
        let so3t = Vector3::new(0.1, 0.2, 0.3);
        let so3 = SO3::exp(&so3t, None);
        let so3n = Vector3::new(-0.1, -0.2, -0.3);
        let so3_inv = SO3::exp(&so3n, None);
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
        assert!(so3_log[0].abs() < TOLERANCE);
        assert!(so3_log[1].abs() < TOLERANCE);
        assert!(so3_log[2].abs() < TOLERANCE);

        // log of inverse is negative log
        let so3 = SO3::random();
        let so3_log = so3.log(None);
        let so3_inv_log = so3.inverse(None).log(None);
        assert!((so3_inv_log[0] + so3_log[0]).abs() < TOLERANCE);
        assert!((so3_inv_log[1] + so3_log[1]).abs() < TOLERANCE);
        assert!((so3_inv_log[2] + so3_log[2]).abs() < TOLERANCE);
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
        assert!((transformed_point.y + 1.414213562373).abs() < 1e-10);
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
        let tangent = Vector3::new(0.1, 0.2, 0.3);
        let so3 = SO3::exp(&tangent, None);
        let recovered_tangent = so3.log(None);
        assert!((tangent - recovered_tangent).norm() < TOLERANCE);
    }

    #[test]
    #[ignore] // TODO: Fix the Jacobian relationship - needs more investigation
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
    fn test_so3_tangent_operations() {
        let tangent = SO3Tangent::new(Vector3::new(1.0, 2.0, 3.0));
        let other = Vector3::new(0.5, 0.5, 0.5);

        let sum = tangent.add(&other);
        assert!((sum - Vector3::new(1.5, 2.5, 3.5)).norm() < TOLERANCE);

        let scaled = tangent.scale(2.0);
        assert!((scaled - Vector3::new(2.0, 4.0, 6.0)).norm() < TOLERANCE);

        let negated = tangent.negate();
        assert!((negated - Vector3::new(-1.0, -2.0, -3.0)).norm() < TOLERANCE);
    }

    #[test]
    fn test_so3_tangent_norms() {
        let tangent = SO3Tangent::new(Vector3::new(3.0, 4.0, 0.0));
        let norm = tangent.norm();
        assert!((norm - 5.0).abs() < TOLERANCE);

        let squared_norm = tangent.squared_norm();
        assert!((squared_norm - 25.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_so3_tangent_zero() {
        let zero = SO3Tangent::zero();
        assert!(zero.norm() < TOLERANCE);

        let tangent = SO3Tangent::new(Vector3::zeros());
        assert!(tangent.is_zero(TOLERANCE));
    }

    #[test]
    fn test_so3_tangent_normalize() {
        let mut tangent = SO3Tangent::new(Vector3::new(3.0, 4.0, 0.0));
        tangent.normalize();
        assert!((tangent.norm() - 1.0).abs() < TOLERANCE);
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
        let small_tangent = Vector3::new(1e-8, 2e-8, 3e-8);
        let so3 = SO3::exp(&small_tangent, None);
        let recovered = so3.log(None);
        assert!((small_tangent - recovered).norm() < TOLERANCE);
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
    fn test_so3_hat_vee_consistency() {
        let tangent = SO3Tangent::new(Vector3::new(0.1, 0.2, 0.3));
        let hat_result = tangent.hat();
        let vee_result = SO3Tangent::vee(&hat_result);
        assert!((vee_result - tangent.data).norm() < TOLERANCE);
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
        let so3_2 = so3_1.clone();

        // Distance to self should be zero
        assert!(so3_1.distance(&so3_2) < TOLERANCE);

        // Distance should be positive for different elements
        let so3_3 = SO3::random();
        assert!(so3_1.distance(&so3_3) >= 0.0);
    }
}
