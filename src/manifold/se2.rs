//! SE(2) - Special Euclidean Group in 2D
//!
//! This module implements the Special Euclidean group SE(2), which represents
//! rigid body transformations in 2D space (rotation + translation).
//!
//! SE(2) elements are represented as a combination of 2D rotation and Vector2 translation.
//! SE(2) tangent elements are represented as [x, y, theta] = 3 components,
//! where x,y is the translational component and theta is the rotational component.
//!
//! The implementation follows the [manif](https://github.com/artivis/manif) C++ library
//! conventions and provides all operations required by the LieGroup and Tangent traits.

use crate::manifold::so2::SO2;
use crate::manifold::{LieGroup, Tangent};
use nalgebra::{Complex, Isometry2, Matrix2, Matrix3, Translation2, UnitComplex, Vector2, Vector3};
use std::fmt;

/// SE(2) group element representing rigid body transformations in 2D.
///
/// Represented as a combination of 2D rotation and Vector2 translation.
#[derive(Clone, Debug, PartialEq)]
pub struct SE2 {
    /// Translation part as Vector2
    translation: Vector2<f64>,
    /// Rotation part as UnitComplex
    rotation: UnitComplex<f64>,
}

impl fmt::Display for SE2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = self.translation();
        write!(
            f,
            "SE2(translation: [{:.4}, {:.4}], rotation: {:.4})",
            t.x,
            t.y,
            self.angle()
        )
    }
}

/// SE(2) tangent space element representing elements in the Lie algebra se(2).
///
/// Following manif conventions, internally represented as [x, y, theta] where:
/// - x, y: translational components
/// - theta: rotational component
#[derive(Clone, Debug, PartialEq)]
pub struct SE2Tangent {
    /// Internal data: [x, y, theta]
    data: Vector3<f64>,
}

impl fmt::Display for SE2Tangent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "se2(x: {:.4}, y: {:.4}, theta: {:.4})",
            self.x(),
            self.y(),
            self.angle()
        )
    }
}

impl SE2 {
    /// Space dimension - dimension of the ambient space that the group acts on
    pub const DIM: usize = 2;

    /// Degrees of freedom - dimension of the tangent space
    pub const DOF: usize = 3;

    /// Representation size - size of the underlying data representation
    pub const REP_SIZE: usize = 4;

    /// Get the identity element of the group.
    ///
    /// Returns the neutral element e such that e ∘ g = g ∘ e = g for any group element g.
    pub fn identity() -> Self {
        SE2 {
            translation: Vector2::zeros(),
            rotation: UnitComplex::identity(),
        }
    }

    /// Get the identity matrix for Jacobians.
    ///
    /// Returns the identity matrix in the appropriate dimension for Jacobian computations.
    pub fn jacobian_identity() -> Matrix3<f64> {
        Matrix3::<f64>::identity()
    }

    /// Create a new SE2 element from translation and rotation.
    ///
    /// # Arguments
    /// * `translation` - Translation vector [x, y]
    /// * `rotation` - Unit complex number representing rotation
    pub fn new(translation: Vector2<f64>, rotation: UnitComplex<f64>) -> Self {
        SE2 {
            translation,
            rotation,
        }
    }

    /// Create SE2 from translation components and angle.
    pub fn from_xy_angle(x: f64, y: f64, theta: f64) -> Self {
        let translation = Vector2::new(x, y);
        let rotation = UnitComplex::from_angle(theta);
        Self::new(translation, rotation)
    }

    /// Create SE2 from translation components and complex rotation.
    pub fn from_xy_complex(x: f64, y: f64, real: f64, imag: f64) -> Self {
        let translation = Vector2::new(x, y);
        let complex = Complex::new(real, imag);
        let rotation = UnitComplex::from_complex(complex);
        Self::new(translation, rotation)
    }

    /// Create SE2 directly from an Isometry2.
    pub fn from_isometry(isometry: Isometry2<f64>) -> Self {
        SE2 {
            translation: isometry.translation.vector,
            rotation: isometry.rotation,
        }
    }

    /// Create SE2 from Vector2 and SO2 components.
    pub fn from_translation_so2(translation: Vector2<f64>, rotation: SO2) -> Self {
        SE2 {
            translation,
            rotation: rotation.complex(),
        }
    }

    /// Get the translation part as a Vector2.
    pub fn translation(&self) -> Vector2<f64> {
        self.translation
    }

    /// Get the rotation part as UnitComplex.
    pub fn rotation_complex(&self) -> UnitComplex<f64> {
        self.rotation
    }

    /// Get the rotation angle.
    pub fn rotation_angle(&self) -> f64 {
        self.rotation.angle()
    }

    /// Get the rotation part as SO2.
    pub fn rotation_so2(&self) -> SO2 {
        SO2::new(self.rotation)
    }

    /// Get as an Isometry2 (convenience method).
    pub fn isometry(&self) -> Isometry2<f64> {
        Isometry2::from_parts(Translation2::from(self.translation), self.rotation)
    }

    /// Get the transformation matrix (3x3 homogeneous matrix).
    pub fn matrix(&self) -> Matrix3<f64> {
        self.isometry().to_homogeneous()
    }

    /// Get the rotation matrix (2x2).
    pub fn rotation_matrix(&self) -> Matrix2<f64> {
        self.rotation.to_rotation_matrix().into_inner()
    }

    /// Get the x component of translation.
    pub fn x(&self) -> f64 {
        self.translation.x
    }

    /// Get the y component of translation.
    pub fn y(&self) -> f64 {
        self.translation.y
    }

    /// Get the real part of the complex rotation.
    pub fn real(&self) -> f64 {
        self.rotation.re
    }

    /// Get the imaginary part of the complex rotation.
    pub fn imag(&self) -> f64 {
        self.rotation.im
    }

    /// Get the rotation angle in radians.
    pub fn angle(&self) -> f64 {
        self.rotation.angle()
    }
}

// Implement basic trait requirements for LieGroup
impl LieGroup for SE2 {
    type TangentVector = SE2Tangent;
    type JacobianMatrix = Matrix3<f64>;
    type LieAlgebra = Matrix3<f64>;

    /// Get the inverse.
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian matrix of the inverse wrt this.
    ///
    /// # Notes
    /// For SE(2): g^{-1} = [R^T, -R^T * t; 0, 1]
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self {
        let rot_inv = self.rotation.inverse();
        let trans_inv = -(rot_inv * self.translation);

        if let Some(jac) = jacobian {
            // Jacobian of inverse operation: -Ad(g)
            *jac = -self.adjoint();
        }

        SE2::new(trans_inv, rot_inv)
    }

    /// Composition of this and another SE2 element.
    fn compose(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        let composed_rotation = self.rotation * other.rotation;
        let composed_translation = self
            .rotation
            .transform_point(&nalgebra::Point2::from(other.translation))
            .coords
            + self.translation;

        let result = SE2::new(composed_translation, composed_rotation);

        if let Some(jac_self) = jacobian_self {
            *jac_self = other.inverse(None).adjoint();
        }

        if let Some(jac_other) = jacobian_other {
            *jac_other = Matrix3::identity();
        }

        result
    }

    /// Get the SE2 corresponding Lie algebra element in vector form.
    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector {
        let theta = self.angle();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let theta_sq = theta * theta;

        let (a, b) = if theta_sq < f64::EPSILON {
            // Taylor approximation
            let a = 1.0 - theta_sq / 6.0;
            let b = 0.5 * theta - theta * theta_sq / 24.0;
            (a, b)
        } else {
            // Euler
            let a = sin_theta / theta;
            let b = (1.0 - cos_theta) / theta;
            (a, b)
        };

        let den = 1.0 / (a * a + b * b);
        let a_scaled = a * den;
        let b_scaled = b * den;

        let x = a_scaled * self.x() + b_scaled * self.y();
        let y = -b_scaled * self.x() + a_scaled * self.y();

        let result = SE2Tangent::new(x, y, theta);

        if let Some(jac) = jacobian {
            *jac = result.right_jacobian_inv();
        }

        result
    }

    fn act(
        &self,
        vector: &Vector3<f64>,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_vector: Option<&mut Matrix3<f64>>,
    ) -> Vector3<f64> {
        // For SE(2), we operate on 2D vectors but maintain 3D interface compatibility
        let point2d = Vector2::new(vector.x, vector.y);
        let transformed_2d = self
            .rotation
            .transform_point(&nalgebra::Point2::from(point2d))
            .coords
            + self.translation;
        let result = Vector3::new(transformed_2d.x, transformed_2d.y, vector.z);

        if let Some(jac_self) = jacobian_self {
            let r = self.rotation_matrix();
            jac_self.fixed_view_mut::<2, 2>(0, 0).copy_from(&r);
            jac_self[(0, 2)] = -point2d.y;
            jac_self[(1, 2)] = point2d.x;
            jac_self[(2, 0)] = 0.0;
            jac_self[(2, 1)] = 0.0;
            jac_self[(2, 2)] = 1.0;
        }

        if let Some(jac_vector) = jacobian_vector {
            *jac_vector = Matrix3::identity();
            let r = self.rotation_matrix();
            jac_vector.fixed_view_mut::<2, 2>(0, 0).copy_from(&r);
        }

        result
    }

    fn adjoint(&self) -> Self::JacobianMatrix {
        let mut adjoint_matrix = Matrix3::identity();
        let rotation_matrix = self.rotation_matrix();

        adjoint_matrix
            .fixed_view_mut::<2, 2>(0, 0)
            .copy_from(&rotation_matrix);
        adjoint_matrix[(0, 2)] = self.y();
        adjoint_matrix[(1, 2)] = -self.x();

        adjoint_matrix
    }

    fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::rng();

        // Random translation in [-1, 1]²
        let translation = Vector2::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));

        // Random rotation
        let angle = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
        let rotation = UnitComplex::from_angle(angle);

        SE2::new(translation, rotation)
    }

    fn normalize(&mut self) {
        self.rotation.renormalize();
    }

    fn is_valid(&self, tolerance: f64) -> bool {
        (self.rotation.norm() - 1.0).abs() < tolerance
    }

    /// Vee operator: log(g)^∨.
    ///
    /// Maps a group element g ∈ G to its tangent vector log(g)^∨ ∈ 𝔤.
    /// For SE(2), this is the same as log().
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

impl SE2Tangent {
    /// Create a new SE2Tangent from x, y, and theta components.
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        SE2Tangent {
            data: Vector3::new(x, y, theta),
        }
    }

    /// Get the x (translational) component.
    pub fn x(&self) -> f64 {
        self.data[0]
    }

    /// Get the y (translational) component.
    pub fn y(&self) -> f64 {
        self.data[1]
    }

    /// Get the theta (rotational) component.
    pub fn angle(&self) -> f64 {
        self.data[2]
    }

    /// Get the translation part as Vector2.
    pub fn translation(&self) -> Vector2<f64> {
        Vector2::new(self.x(), self.y())
    }

    /// Create SE2Tangent from a 3-dimensional vector
    pub fn from_vector(vector: nalgebra::DVector<f64>) -> Self {
        if vector.len() != 3 {
            panic!("SE2Tangent::from_vector expects 3-dimensional vector");
        }
        SE2Tangent {
            data: Vector3::new(vector[0], vector[1], vector[2]),
        }
    }

    /// Convert SE2Tangent to a 3-dimensional vector
    pub fn to_vector(&self) -> nalgebra::DVector<f64> {
        nalgebra::DVector::from_vec(vec![self.data[0], self.data[1], self.data[2]])
    }
}

impl Tangent<SE2> for SE2Tangent {
    /// Dimension of the tangent space
    const DIM: usize = 3;

    /// Get the SE2 element.
    fn exp(&self, jacobian: Option<&mut <SE2 as LieGroup>::JacobianMatrix>) -> SE2 {
        let theta = self.angle();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let theta_sq = theta * theta;

        let (a, b) = if theta_sq < f64::EPSILON {
            // Taylor approximation
            let a = 1.0 - theta_sq / 6.0;
            let b = 0.5 * theta - theta * theta_sq / 24.0;
            (a, b)
        } else {
            // Euler
            let a = sin_theta / theta;
            let b = (1.0 - cos_theta) / theta;
            (a, b)
        };

        let translation = Vector2::new(a * self.x() - b * self.y(), b * self.x() + a * self.y());
        let rotation = UnitComplex::from_cos_sin_unchecked(cos_theta, sin_theta);

        if let Some(jac) = jacobian {
            *jac = self.right_jacobian();
        }

        SE2::new(translation, rotation)
    }

    /// Right Jacobian Jr.
    fn right_jacobian(&self) -> <SE2 as LieGroup>::JacobianMatrix {
        let theta = self.angle();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let theta_sq = theta * theta;

        let (a, b) = if theta_sq < f64::EPSILON {
            // Taylor approximation
            let a = 1.0 - theta_sq / 6.0;
            let b = 0.5 * theta - theta * theta_sq / 24.0;
            (a, b)
        } else {
            // Euler
            let a = sin_theta / theta;
            let b = (1.0 - cos_theta) / theta;
            (a, b)
        };

        let mut jac = Matrix3::identity();
        jac[(0, 0)] = a;
        jac[(0, 1)] = b;
        jac[(1, 0)] = -b;
        jac[(1, 1)] = a;

        if theta_sq < f64::EPSILON {
            jac[(0, 2)] = -self.y() / 2.0 + theta * self.x() / 6.0;
            jac[(1, 2)] = self.x() / 2.0 + theta * self.y() / 6.0;
        } else {
            jac[(0, 2)] = (-self.y() + theta * self.x() + self.y() * cos_theta
                - self.x() * sin_theta)
                / theta_sq;
            jac[(1, 2)] =
                (self.x() + theta * self.y() - self.x() * cos_theta - self.y() * sin_theta)
                    / theta_sq;
        }

        jac
    }

    /// Left Jacobian Jl.
    fn left_jacobian(&self) -> <SE2 as LieGroup>::JacobianMatrix {
        let theta = self.angle();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let theta_sq = theta * theta;

        let (a, b) = if theta_sq < f64::EPSILON {
            // Taylor approximation
            let a = 1.0 - theta_sq / 6.0;
            let b = 0.5 * theta - theta * theta_sq / 24.0;
            (a, b)
        } else {
            // Euler
            let a = sin_theta / theta;
            let b = (1.0 - cos_theta) / theta;
            (a, b)
        };

        let mut jac = Matrix3::identity();
        jac[(0, 0)] = a;
        jac[(0, 1)] = -b;
        jac[(1, 0)] = b;
        jac[(1, 1)] = a;

        if theta_sq < f64::EPSILON {
            jac[(0, 2)] = self.y() / 2.0 + theta * self.x() / 6.0;
            jac[(1, 2)] = -self.x() / 2.0 + theta * self.y() / 6.0;
        } else {
            jac[(0, 2)] =
                (self.y() + theta * self.x() - self.y() * cos_theta - self.x() * sin_theta)
                    / theta_sq;
            jac[(1, 2)] = (-self.x() + theta * self.y() + self.x() * cos_theta
                - self.y() * sin_theta)
                / theta_sq;
        }

        jac
    }

    /// Inverse of right Jacobian Jr⁻¹.
    fn right_jacobian_inv(&self) -> <SE2 as LieGroup>::JacobianMatrix {
        let theta = self.angle();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let theta_sq = theta * theta;

        let mut jac_inv = Matrix3::zeros();
        jac_inv[(0, 1)] = -theta * 0.5;
        jac_inv[(1, 0)] = -jac_inv[(0, 1)];
        jac_inv[(2, 2)] = 1.0;

        if theta_sq > f64::EPSILON {
            let a = theta * sin_theta;
            let b = theta * cos_theta;

            jac_inv[(0, 0)] = -a / (2.0 * cos_theta - 2.0);
            jac_inv[(1, 1)] = jac_inv[(0, 0)];

            let den = 2.0 * theta * (cos_theta - 1.0);
            jac_inv[(0, 2)] = (a * self.x() + b * self.y() - theta * self.y()
                + 2.0 * self.x() * cos_theta
                - 2.0 * self.x())
                / den;
            jac_inv[(1, 2)] =
                (-b * self.x() + a * self.y() + theta * self.x() + 2.0 * self.y() * cos_theta
                    - 2.0 * self.y())
                    / den;
        } else {
            jac_inv[(0, 0)] = 1.0 - theta_sq / 12.0;
            jac_inv[(1, 1)] = jac_inv[(0, 0)];

            jac_inv[(0, 2)] = self.y() / 2.0 + theta * self.x() / 12.0;
            jac_inv[(1, 2)] = -self.x() / 2.0 + theta * self.y() / 12.0;
        }

        jac_inv
    }

    /// Inverse of left Jacobian Jl⁻¹.
    fn left_jacobian_inv(&self) -> <SE2 as LieGroup>::JacobianMatrix {
        let theta = self.angle();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let theta_sq = theta * theta;

        let mut jac_inv = Matrix3::zeros();
        jac_inv[(0, 1)] = theta * 0.5;
        jac_inv[(1, 0)] = -jac_inv[(0, 1)];
        jac_inv[(2, 2)] = 1.0;

        if theta_sq > f64::EPSILON {
            let a = theta * sin_theta;
            let b = theta * cos_theta;

            jac_inv[(0, 0)] = -a / (2.0 * cos_theta - 2.0);
            jac_inv[(1, 1)] = jac_inv[(0, 0)];

            let den = 2.0 * theta * (cos_theta - 1.0);
            jac_inv[(0, 2)] =
                (a * self.x() - b * self.y() + theta * self.y() + 2.0 * self.x() * cos_theta
                    - 2.0 * self.x())
                    / den;
            jac_inv[(1, 2)] = (b * self.x() + a * self.y() - theta * self.x()
                + 2.0 * self.y() * cos_theta
                - 2.0 * self.y())
                / den;
        } else {
            jac_inv[(0, 0)] = 1.0 - theta_sq / 12.0;
            jac_inv[(1, 1)] = jac_inv[(0, 0)];

            jac_inv[(0, 2)] = -self.y() / 2.0 + theta * self.x() / 12.0;
            jac_inv[(1, 2)] = self.x() / 2.0 + theta * self.y() / 12.0;
        }

        jac_inv
    }

    /// Hat operator: φ^∧ (vector to matrix).
    fn hat(&self) -> <SE2 as LieGroup>::LieAlgebra {
        Matrix3::new(
            0.0,
            -self.angle(),
            self.x(),
            self.angle(),
            0.0,
            self.y(),
            0.0,
            0.0,
            0.0,
        )
    }

    /// Zero tangent vector.
    fn zero() -> <SE2 as LieGroup>::TangentVector {
        SE2Tangent::new(0.0, 0.0, 0.0)
    }

    /// Random tangent vector (useful for testing).
    fn random() -> <SE2 as LieGroup>::TangentVector {
        use rand::Rng;
        let mut rng = rand::rng();
        SE2Tangent::new(
            rng.random_range(-1.0..1.0),                                   // x
            rng.random_range(-1.0..1.0),                                   // y
            rng.random_range(-std::f64::consts::PI..std::f64::consts::PI), // theta
        )
    }

    /// Check if the tangent vector is approximately zero.
    fn is_zero(&self, tolerance: f64) -> bool {
        self.data.norm() < tolerance
    }

    /// Normalize the tangent vector to unit norm.
    fn normalize(&mut self) {
        let norm = self.data.norm();
        if norm > f64::EPSILON {
            self.data /= norm;
        }
    }

    /// Return a unit tangent vector in the same direction.
    fn normalized(&self) -> <SE2 as LieGroup>::TangentVector {
        let norm = self.data.norm();
        if norm > f64::EPSILON {
            SE2Tangent {
                data: self.data / norm,
            }
        } else {
            SE2Tangent::zero()
        }
    }

    /// Small adjoint matrix for SE(2).
    ///
    /// For SE(2), the small adjoint involves the angular component.
    fn small_adj(&self) -> <SE2 as LieGroup>::JacobianMatrix {
        let _theta = self.angle();
        let x = self.x();
        let y = self.y();

        let mut small_adj = Matrix3::zeros();

        // Following the C++ manif implementation structure:
        // smallAdj(0,1) = -angle();
        // smallAdj(1,0) =  angle();
        // smallAdj(0,2) =  y();
        // smallAdj(1,2) = -x();
        small_adj[(0, 1)] = -self.angle();
        small_adj[(1, 0)] = self.angle();
        small_adj[(0, 2)] = y;
        small_adj[(1, 2)] = -x;

        small_adj
    }

    /// Lie bracket for SE(2).
    ///
    /// Computes the Lie bracket [this, other] = this.small_adj() * other.
    fn lie_bracket(&self, other: &Self) -> <SE2 as LieGroup>::TangentVector {
        let bracket_result = self.small_adj() * other.data;
        SE2Tangent {
            data: bracket_result,
        }
    }

    /// Check if this tangent vector is approximately equal to another.
    ///
    /// # Arguments
    /// * `other` - The other tangent vector to compare with
    /// * `tolerance` - The tolerance for the comparison
    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        (self.data - other.data).norm() < tolerance
    }

    /// Get the ith generator of the SE(2) Lie algebra.
    ///
    /// # Arguments
    /// * `i` - Index of the generator (0, 1, or 2 for SE(2))
    ///
    /// # Returns
    /// The generator matrix
    fn generator(&self, i: usize) -> <SE2 as LieGroup>::LieAlgebra {
        assert!(i < 3, "SE(2) only has generators for indices 0, 1, 2");

        match i {
            0 => {
                // Generator E1 for x translation
                Matrix3::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
            1 => {
                // Generator E2 for y translation
                Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            }
            2 => {
                // Generator E3 for rotation
                Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOLERANCE: f64 = 1e-12;

    #[test]
    fn test_se2_tangent_basic() {
        let tangent = SE2Tangent::new(4.0, 2.0, PI);
        assert_eq!(tangent.x(), 4.0);
        assert_eq!(tangent.y(), 2.0);
        assert_eq!(tangent.angle(), PI);
    }

    #[test]
    fn test_se2_tangent_zero() {
        let zero = SE2Tangent::zero();
        assert_eq!(zero.data, Vector3::zeros());
        assert!(zero.is_zero(1e-10));
    }

    #[test]
    fn test_se2_identity() {
        let identity = SE2::identity();
        assert!(identity.is_valid(TOLERANCE));
        assert_eq!(identity.x(), 0.0);
        assert_eq!(identity.y(), 0.0);
        assert_eq!(identity.angle(), 0.0);
    }

    #[test]
    fn test_se2_new() {
        let translation = Vector2::new(1.0, 2.0);
        let rotation = UnitComplex::from_angle(PI / 4.0);
        let se2 = SE2::new(translation, rotation);

        assert!(se2.is_valid(TOLERANCE));
        assert_eq!(se2.x(), 1.0);
        assert_eq!(se2.y(), 2.0);
        assert!((se2.angle() - PI / 4.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_from_xy_angle() {
        let se2 = SE2::from_xy_angle(4.0, 2.0, 0.0);
        assert_eq!(se2.x(), 4.0);
        assert_eq!(se2.y(), 2.0);
        assert_eq!(se2.angle(), 0.0);
    }

    #[test]
    fn test_se2_from_xy_complex() {
        let se2 = SE2::from_xy_complex(4.0, 2.0, 1.0, 0.0);
        assert_eq!(se2.x(), 4.0);
        assert_eq!(se2.y(), 2.0);
        assert_eq!(se2.real(), 1.0);
        assert_eq!(se2.imag(), 0.0);
        assert_eq!(se2.angle(), 0.0);
    }

    #[test]
    fn test_se2_inverse() {
        let se2 = SE2::from_xy_angle(1.0, 1.0, PI);
        let se2_inv = se2.inverse(None);

        assert!((se2_inv.x() - 1.0).abs() < TOLERANCE);
        assert!((se2_inv.y() - 1.0).abs() < TOLERANCE);
        assert!((se2_inv.angle() + PI).abs() < TOLERANCE);

        // Test that g * g^-1 = identity
        let composed = se2.compose(&se2_inv, None, None);
        let identity = SE2::identity();

        assert!((composed.x() - identity.x()).abs() < TOLERANCE);
        assert!((composed.y() - identity.y()).abs() < TOLERANCE);
        assert!((composed.angle() - identity.angle()).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_compose() {
        let se2a = SE2::from_xy_angle(1.0, 1.0, PI / 2.0);
        let se2b = SE2::from_xy_angle(2.0, 2.0, PI / 2.0);
        let se2c = se2a.compose(&se2b, None, None);

        assert!((se2c.x() - (-1.0)).abs() < TOLERANCE);
        assert!((se2c.y() - 3.0).abs() < TOLERANCE);
        assert!((se2c.angle() - PI).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_exp_log() {
        let tangent = SE2Tangent::new(4.0, 2.0, PI);
        let se2 = tangent.exp(None);
        let recovered_tangent = se2.log(None);

        assert!((tangent.x() - recovered_tangent.x()).abs() < TOLERANCE);
        assert!((tangent.y() - recovered_tangent.y()).abs() < TOLERANCE);
        assert!((tangent.angle() - recovered_tangent.angle()).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_exp_zero() {
        let zero_tangent = SE2Tangent::zero();
        let se2 = zero_tangent.exp(None);
        let identity = SE2::identity();

        assert!((se2.x() - identity.x()).abs() < TOLERANCE);
        assert!((se2.y() - identity.y()).abs() < TOLERANCE);
        assert!((se2.angle() - identity.angle()).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_log_identity() {
        let identity = SE2::identity();
        let tangent = identity.log(None);

        assert!(tangent.data.norm() < TOLERANCE);
    }

    #[test]
    fn test_se2_act() {
        let se2 = SE2::from_xy_angle(1.0, 1.0, PI / 2.0);
        let point = Vector3::new(1.0, 1.0, 0.0);
        let transformed_point = se2.act(&point, None, None);

        assert!((transformed_point.x - 0.0).abs() < TOLERANCE);
        assert!((transformed_point.y - 2.0).abs() < TOLERANCE);
        assert!((transformed_point.z - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_between() {
        let se2a = SE2::from_xy_angle(1.0, 1.0, PI);
        let se2b = SE2::from_xy_angle(1.0, 1.0, PI);
        let se2c = se2a.between(&se2b, None, None);

        assert!((se2c.x() - 0.0).abs() < TOLERANCE);
        assert!((se2c.y() - 0.0).abs() < TOLERANCE);
        assert!((se2c.angle() - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_adjoint() {
        let se2 = SE2::random();
        let adj = se2.adjoint();

        assert_eq!(adj.nrows(), 3);
        assert_eq!(adj.ncols(), 3);
    }

    #[test]
    fn test_se2_manifold_properties() {
        assert_eq!(SE2::DIM, 2);
        assert_eq!(SE2::DOF, 3);
        assert_eq!(SE2::REP_SIZE, 4);
    }

    #[test]
    fn test_se2_random() {
        let se2 = SE2::random();
        assert!(se2.is_valid(TOLERANCE));
    }

    #[test]
    fn test_se2_normalize() {
        let mut se2 = SE2::from_xy_complex(1.0, 2.0, 0.5, 0.5); // Not normalized complex
        se2.normalize();
        assert!(se2.is_valid(TOLERANCE));
    }

    #[test]
    fn test_se2_tangent_exp_jacobians() {
        let tangent = SE2Tangent::new(1.0, 2.0, 0.1);

        let se2_element = tangent.exp(None);
        assert!(se2_element.is_valid(TOLERANCE));

        // Test Jacobians have correct dimensions
        let right_jac = tangent.right_jacobian();
        let left_jac = tangent.left_jacobian();
        let right_jac_inv = tangent.right_jacobian_inv();
        let left_jac_inv = tangent.left_jacobian_inv();

        assert_eq!(right_jac.nrows(), 3);
        assert_eq!(right_jac.ncols(), 3);
        assert_eq!(left_jac.nrows(), 3);
        assert_eq!(left_jac.ncols(), 3);
        assert_eq!(right_jac_inv.nrows(), 3);
        assert_eq!(right_jac_inv.ncols(), 3);
        assert_eq!(left_jac_inv.nrows(), 3);
        assert_eq!(left_jac_inv.ncols(), 3);
    }

    #[test]
    fn test_se2_tangent_hat() {
        let tangent = SE2Tangent::new(4.0, 2.0, PI);
        let hat_matrix = tangent.hat();

        assert_eq!(hat_matrix.nrows(), 3);
        assert_eq!(hat_matrix.ncols(), 3);
        assert_eq!(hat_matrix[(0, 2)], 4.0);
        assert_eq!(hat_matrix[(1, 2)], 2.0);
        assert_eq!(hat_matrix[(1, 0)], PI);
        assert_eq!(hat_matrix[(0, 1)], -PI);
    }

    #[test]
    fn test_se2_consistency() {
        // Test associativity: (g1 * g2) * g3 = g1 * (g2 * g3)
        let se2_1 = SE2::random();
        let se2_2 = SE2::random();
        let se2_3 = SE2::random();

        let left_assoc = se2_1
            .compose(&se2_2, None, None)
            .compose(&se2_3, None, None);
        let right_assoc = se2_1.compose(&se2_2.compose(&se2_3, None, None), None, None);

        let translation_diff = (left_assoc.translation() - right_assoc.translation()).norm();
        let angle_diff = (left_assoc.angle() - right_assoc.angle()).abs();

        assert!(translation_diff < 1e-10);
        assert!(angle_diff < 1e-10);
    }

    #[test]
    fn test_se2_isometry() {
        let translation = nalgebra::Translation2::new(1.0, 2.0);
        let rotation = UnitComplex::from_angle(PI / 4.0);
        let isometry = Isometry2::from_parts(translation, rotation);

        let se2 = SE2::from_isometry(isometry);
        let recovered_isometry = se2.isometry();

        let translation_diff =
            (isometry.translation.vector - recovered_isometry.translation.vector).norm();
        let angle_diff = (isometry.rotation.angle() - recovered_isometry.rotation.angle()).abs();

        assert!(translation_diff < TOLERANCE);
        assert!(angle_diff < TOLERANCE);
    }

    #[test]
    fn test_se2_matrix() {
        let se2 = SE2::random();
        let matrix = se2.matrix();

        // Check matrix is 3x3
        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 3);

        // Check bottom row is [0, 0, 1]
        assert!((matrix[(2, 0)]).abs() < TOLERANCE);
        assert!((matrix[(2, 1)]).abs() < TOLERANCE);
        assert!((matrix[(2, 2)] - 1.0).abs() < TOLERANCE);
    }

    // Additional comprehensive tests based on manif C++ test suite

    #[test]
    fn test_se2_constructor_copy() {
        let se2_original = SE2::from_xy_complex(4.0, 2.0, (PI / 4.0).cos(), (PI / 4.0).sin());
        let se2_copy = se2_original.clone();

        assert_eq!(se2_copy.x(), 4.0);
        assert_eq!(se2_copy.y(), 2.0);
        assert!((se2_copy.angle() - PI / 4.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_assign_op() {
        let _se2a = SE2::from_xy_angle(0.0, 0.0, 0.0);
        let se2b = SE2::from_xy_angle(4.0, 2.0, PI);

        let se2a = se2b.clone(); // Rust equivalent of assignment

        assert_eq!(se2a.x(), 4.0);
        assert_eq!(se2a.y(), 2.0);
        assert!((se2a.angle() - PI).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_inverse_detailed() {
        // Test with identity
        let se2 = SE2::identity();
        let se2_inv = se2.inverse(None);

        assert!((se2_inv.x() - 0.0).abs() < TOLERANCE);
        assert!((se2_inv.y() - 0.0).abs() < TOLERANCE);
        assert!((se2_inv.angle() - 0.0).abs() < TOLERANCE);
        assert!((se2_inv.real() - 1.0).abs() < TOLERANCE);
        assert!((se2_inv.imag() - 0.0).abs() < TOLERANCE);

        // Test with specific values
        let se2 = SE2::from_xy_angle(0.7, 2.3, PI / 3.0);
        let se2_inv = se2.inverse(None);

        assert!((se2_inv.x() - (-2.341858428704209)).abs() < 1e-10);
        assert!((se2_inv.y() - (-0.543782217350893)).abs() < 1e-10);
        assert!((se2_inv.angle() - (-PI / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_se2_rplus_zero() {
        let se2a = SE2::from_xy_angle(1.0, 1.0, PI / 2.0);
        let se2b = SE2Tangent::zero();

        let se2c = se2a.right_plus(&se2b, None, None);

        assert!((se2c.x() - 1.0).abs() < TOLERANCE);
        assert!((se2c.y() - 1.0).abs() < TOLERANCE);
        assert!((se2c.angle() - PI / 2.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_rplus() {
        let se2a = SE2::from_xy_angle(1.0, 1.0, PI / 2.0);
        let se2b = SE2Tangent::new(1.0, 1.0, PI / 2.0);

        let se2c = se2a.right_plus(&se2b, None, None);

        assert!((se2c.angle() - PI).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_lplus_zero() {
        let se2a = SE2::from_xy_angle(1.0, 1.0, PI / 2.0);
        let se2b = SE2Tangent::zero();

        let se2c = se2a.left_plus(&se2b, None, None);

        assert!((se2c.x() - 1.0).abs() < TOLERANCE);
        assert!((se2c.y() - 1.0).abs() < TOLERANCE);
        assert!((se2c.angle() - PI / 2.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_lplus() {
        let se2a = SE2::from_xy_angle(1.0, 1.0, PI / 2.0);
        let se2b = SE2Tangent::new(1.0, 1.0, PI / 2.0);

        let se2c = se2a.left_plus(&se2b, None, None);

        assert!((se2c.angle() - PI).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_rminus_zero() {
        let se2a = SE2::identity();
        let se2b = SE2::identity();

        let se2c = se2a.right_minus(&se2b, None, None);

        assert!((se2c.x() - 0.0).abs() < TOLERANCE);
        assert!((se2c.y() - 0.0).abs() < TOLERANCE);
        assert!((se2c.angle() - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_rminus() {
        let se2a = SE2::from_xy_angle(1.0, 1.0, PI);
        let se2b = SE2::from_xy_angle(2.0, 2.0, PI / 2.0);

        let se2c = se2a.right_minus(&se2b, None, None);

        assert!((se2c.angle() - PI / 2.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_lminus_identity() {
        let se2a = SE2::identity();
        let se2b = SE2::identity();

        let se2c = se2a.left_minus(&se2b, None, None);

        assert!((se2c.x() - 0.0).abs() < TOLERANCE);
        assert!((se2c.y() - 0.0).abs() < TOLERANCE);
        assert!((se2c.angle() - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_lminus() {
        let se2a = SE2::from_xy_angle(1.0, 1.0, PI);
        let se2b = SE2::from_xy_angle(2.0, 2.0, PI / 2.0);

        let se2c = se2a.left_minus(&se2b, None, None);

        assert!((se2c.angle() - PI / 2.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_lift() {
        let se2 = SE2::from_xy_angle(1.0, 1.0, PI);
        let se2_log = se2.log(None);

        assert!((se2_log.angle() - PI).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_compose_detailed() {
        let se2a = SE2::from_xy_angle(1.0, 1.0, PI / 2.0);
        let se2b = SE2::from_xy_angle(2.0, 2.0, PI / 2.0);

        let se2c = se2a.compose(&se2b, None, None);

        assert!((se2c.x() - (-1.0)).abs() < TOLERANCE);
        assert!((se2c.y() - 3.0).abs() < TOLERANCE);
        assert!((se2c.angle() - PI).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_between_identity() {
        let se2a = SE2::from_xy_angle(1.0, 1.0, PI);
        let se2b = SE2::from_xy_angle(1.0, 1.0, PI);

        let se2c = se2a.between(&se2b, None, None);

        assert!((se2c.x() - 0.0).abs() < TOLERANCE);
        assert!((se2c.y() - 0.0).abs() < TOLERANCE);
        assert!((se2c.angle() - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_between_detailed() {
        let se2a = SE2::from_xy_angle(1.0, 1.0, PI);
        let se2b = SE2::from_xy_angle(2.0, 2.0, PI / 2.0);

        let se2c = se2a.between(&se2b, None, None);

        assert!((se2c.x() - (-1.0)).abs() < TOLERANCE);
        assert!((se2c.y() - (-1.0)).abs() < TOLERANCE);
        assert!((se2c.angle() - (-PI / 2.0)).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_act_detailed() {
        let se2 = SE2::from_xy_angle(1.0, 1.0, PI / 2.0);
        let point = Vector3::new(1.0, 1.0, 0.0);
        let transformed_point = se2.act(&point, None, None);

        assert!((transformed_point.x - 0.0).abs() < TOLERANCE);
        assert!((transformed_point.y - 2.0).abs() < TOLERANCE);

        let se2 = SE2::from_xy_angle(1.0, 1.0, -PI / 2.0);
        let transformed_point = se2.act(&point, None, None);

        assert!((transformed_point.x - 2.0).abs() < TOLERANCE);
        assert!((transformed_point.y - 0.0).abs() < TOLERANCE);

        let se2 = SE2::identity();
        let transformed_point = se2.act(&point, None, None);

        assert!((transformed_point.x - 1.0).abs() < TOLERANCE);
        assert!((transformed_point.y - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_rotation_matrix() {
        let se2 = SE2::identity();
        let r = se2.rotation_matrix();

        assert_eq!(r.nrows(), 2);
        assert_eq!(r.ncols(), 2);

        // Should be identity matrix for zero rotation
        assert!((r[(0, 0)] - 1.0).abs() < TOLERANCE);
        assert!((r[(0, 1)] - 0.0).abs() < TOLERANCE);
        assert!((r[(1, 0)] - 0.0).abs() < TOLERANCE);
        assert!((r[(1, 1)] - 1.0).abs() < TOLERANCE);
    }

    // SE2Tangent specific tests

    #[test]
    fn test_se2_tangent_data() {
        let se2_tan = SE2Tangent::new(4.0, 2.0, PI);

        // Test access functions
        assert_eq!(se2_tan.x(), 4.0);
        assert_eq!(se2_tan.y(), 2.0);
        assert_eq!(se2_tan.angle(), PI);
    }

    #[test]
    fn test_se2_tangent_retract() {
        let se2_tan = SE2Tangent::new(4.0, 2.0, PI);

        assert_eq!(se2_tan.x(), 4.0);
        assert_eq!(se2_tan.y(), 2.0);
        assert_eq!(se2_tan.angle(), PI);

        let se2_exp = se2_tan.exp(None);

        assert!((se2_exp.real() - PI.cos()).abs() < TOLERANCE);
        assert!((se2_exp.imag() - PI.sin()).abs() < TOLERANCE);
        assert!((se2_exp.angle() - PI).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_tangent_retract_jac() {
        let se2_tan = SE2Tangent::new(4.0, 2.0, PI);

        let mut j_ret = Matrix3::zeros();
        let se2_exp = se2_tan.exp(Some(&mut j_ret));

        assert!((se2_exp.real() - PI.cos()).abs() < TOLERANCE);
        assert!((se2_exp.imag() - PI.sin()).abs() < TOLERANCE);
        assert!((se2_exp.angle() - PI).abs() < TOLERANCE);

        // Check Jacobian dimensions
        assert_eq!(j_ret.nrows(), 3);
        assert_eq!(j_ret.ncols(), 3);
    }

    #[test]
    fn test_se2_small_angle_approximations() {
        // Test behavior with very small angles
        let small_tangent = SE2Tangent::new(1e-8, 2e-8, 1e-9);

        let se2 = small_tangent.exp(None);
        let recovered = se2.log(None);

        let diff = (Vector3::new(small_tangent.x(), small_tangent.y(), small_tangent.angle())
            - Vector3::new(recovered.x(), recovered.y(), recovered.angle()))
        .norm();
        assert!(diff < TOLERANCE);
    }

    #[test]
    fn test_se2_tangent_norm() {
        let tangent = SE2Tangent::new(3.0, 4.0, 0.0);
        let norm = Vector3::new(tangent.x(), tangent.y(), tangent.angle()).norm();
        assert!((norm - 5.0).abs() < TOLERANCE); // sqrt(3^2 + 4^2) = 5
    }

    // New tests for the additional functions

    #[test]
    fn test_se2_vee() {
        let se2 = SE2::random();
        let tangent_log = se2.log(None);
        let tangent_vee = se2.vee();

        assert!((tangent_log.data - tangent_vee.data).norm() < 1e-10);
    }

    #[test]
    fn test_se2_is_approx() {
        let se2_1 = SE2::random();
        let se2_2 = se2_1.clone();

        assert!(se2_1.is_approx(&se2_1, 1e-10));
        assert!(se2_1.is_approx(&se2_2, 1e-10));

        // Test with small perturbation
        let small_tangent = SE2Tangent::new(1e-12, 1e-12, 1e-12);
        let se2_perturbed = se2_1.right_plus(&small_tangent, None, None);
        assert!(se2_1.is_approx(&se2_perturbed, 1e-10));
    }

    #[test]
    fn test_se2_tangent_small_adj() {
        let tangent = SE2Tangent::new(0.1, 0.2, 0.3);
        let small_adj = tangent.small_adj();

        // Verify the structure of the small adjoint matrix for SE(2)
        // Following C++ manif implementation:
        // [ 0  -θ   y ]
        // [ θ   0  -x ]
        // [ 0   0   0 ]
        assert_eq!(small_adj[(0, 0)], 0.0);
        assert_eq!(small_adj[(1, 1)], 0.0);
        assert_eq!(small_adj[(2, 2)], 0.0);
        assert_eq!(small_adj[(0, 1)], -tangent.angle());
        assert_eq!(small_adj[(1, 0)], tangent.angle());
        assert_eq!(small_adj[(0, 2)], tangent.y());
        assert_eq!(small_adj[(1, 2)], -tangent.x());
    }

    #[test]
    fn test_se2_tangent_lie_bracket() {
        let tangent_a = SE2Tangent::new(0.1, 0.0, 0.0); // Pure x translation
        let tangent_b = SE2Tangent::new(0.0, 0.0, 0.2); // Pure rotation

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
    fn test_se2_tangent_is_approx() {
        let tangent_1 = SE2Tangent::new(0.1, 0.2, 0.3);
        let tangent_2 = SE2Tangent::new(0.1 + 1e-12, 0.2, 0.3);
        let tangent_3 = SE2Tangent::new(0.5, 0.6, 0.7);

        assert!(tangent_1.is_approx(&tangent_1, 1e-10));
        assert!(tangent_1.is_approx(&tangent_2, 1e-10));
        assert!(!tangent_1.is_approx(&tangent_3, 1e-10));
    }

    #[test]
    fn test_se2_generators() {
        let tangent = SE2Tangent::new(1.0, 1.0, 1.0);

        // Test all three generators
        for i in 0..3 {
            let generator = tangent.generator(i);

            // Verify that generators are 3x3 matrices
            assert_eq!(generator.nrows(), 3);
            assert_eq!(generator.ncols(), 3);
        }

        // Test specific values for the generators
        let e1 = tangent.generator(0); // x translation
        let e2 = tangent.generator(1); // y translation  
        let e3 = tangent.generator(2); // rotation

        // Expected generators for SE(2)
        let expected_e1 = Matrix3::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let expected_e2 = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let expected_e3 = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        assert!((e1 - expected_e1).norm() < 1e-10);
        assert!((e2 - expected_e2).norm() < 1e-10);
        assert!((e3 - expected_e3).norm() < 1e-10);
    }

    #[test]
    #[should_panic]
    fn test_se2_generator_invalid_index() {
        let tangent = SE2Tangent::new(1.0, 1.0, 1.0);
        let _generator = tangent.generator(3); // Should panic for SE(2)
    }

    #[test]
    fn test_se2_jacobi_identity() {
        // Test Jacobi identity: [x,[y,z]]+[y,[z,x]]+[z,[x,y]]=0
        let x = SE2Tangent::new(0.1, 0.0, 0.0);
        let y = SE2Tangent::new(0.0, 0.2, 0.0);
        let z = SE2Tangent::new(0.0, 0.0, 0.3);

        let term1 = x.lie_bracket(&y.lie_bracket(&z));
        let term2 = y.lie_bracket(&z.lie_bracket(&x));
        let term3 = z.lie_bracket(&x.lie_bracket(&y));

        let jacobi_sum = SE2Tangent {
            data: term1.data + term2.data + term3.data,
        };
        assert!(jacobi_sum.is_zero(1e-10));
    }

    #[test]
    fn test_se2_hat_vee_consistency() {
        let tangent = SE2Tangent::new(0.1, 0.2, 0.3);
        let hat_matrix = tangent.hat();

        // For SE(2), verify hat matrix structure
        // The hat matrix should be 3x3, not 4x4 like SE(3)
        assert_eq!(hat_matrix[(0, 2)], tangent.x());
        assert_eq!(hat_matrix[(1, 2)], tangent.y());
        assert_eq!(hat_matrix[(0, 1)], -tangent.angle());
        assert_eq!(hat_matrix[(1, 0)], tangent.angle());
        assert_eq!(hat_matrix[(2, 0)], 0.0);
        assert_eq!(hat_matrix[(2, 1)], 0.0);
        assert_eq!(hat_matrix[(2, 2)], 0.0);
    }
}
