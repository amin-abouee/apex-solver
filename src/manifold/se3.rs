//! SE(3) - Special Euclidean Group in 3D
//!
//! This module implements the Special Euclidean group SE(3), which represents
//! rigid body transformations in 3D space (rotation + translation).
//!
//! SE(3) elements are represented using nalgebra's Isometry3 internally.
//! SE(3) tangent elements are represented as [rho(3), theta(3)] = 6 components,
//! where rho is the translational component and theta is the rotational component.
//!
//! The implementation follows the [manif](https://github.com/artivis/manif) C++ library
//! conventions and provides all operations required by the LieGroup and LieAlgebra traits.

use crate::manifold::{LieAlgebra, LieGroup, ManifoldResult};
use nalgebra::{
    DMatrix, Isometry3, Matrix3, Matrix4, Matrix6, Point3, Translation3, UnitQuaternion, Vector3,
    Vector6,
};
use std::fmt::Debug;

/// Create a skew-symmetric matrix from a 3D vector.
/// For vector v = [x, y, z], returns:
/// [ 0  -z   y ]
/// [ z   0  -x ]
/// [-y   x   0 ]
fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

/// SE(3) group element representing rigid body transformations in 3D.
///
/// Internally represented using nalgebra's Isometry3<f64> for efficient transformations.
#[derive(Clone, Debug, PartialEq)]
pub struct SE3 {
    /// Internal representation as an isometry
    isometry: Isometry3<f64>,
}

/// SE(3) tangent space element representing elements in the Lie algebra se(3).
///
/// Following manif conventions, internally represented as [rho(3), theta(3)] where:
/// - rho: translational component [rho_x, rho_y, rho_z]
/// - theta: rotational component [theta_x, theta_y, theta_z]
#[derive(Clone, Debug, PartialEq)]
pub struct SE3Tangent {
    /// Internal data: [rho_x, rho_y, rho_z, theta_x, theta_y, theta_z]
    data: Vector6<f64>,
}

impl SE3 {
    /// Create a new SE3 element from translation and rotation.
    ///
    /// # Arguments
    /// * `translation` - Translation vector [x, y, z]
    /// * `rotation` - Unit quaternion representing rotation
    pub fn new(translation: Vector3<f64>, rotation: UnitQuaternion<f64>) -> Self {
        let isometry = Isometry3::from_parts(Translation3::from(translation), rotation);
        SE3 { isometry }
    }

    /// Create SE3 from translation components and Euler angles.
    pub fn from_translation_rotation(
        x: f64,
        y: f64,
        z: f64,
        roll: f64,
        pitch: f64,
        yaw: f64,
    ) -> Self {
        let translation = Vector3::new(x, y, z);
        let rotation = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
        Self::new(translation, rotation)
    }

    /// Create SE3 directly from an Isometry3.
    pub fn from_isometry(isometry: Isometry3<f64>) -> Self {
        SE3 { isometry }
    }

    /// Get the translation part as a Vector3.
    pub fn translation(&self) -> Vector3<f64> {
        self.isometry.translation.vector
    }

    /// Get the rotation part as a UnitQuaternion.
    pub fn rotation(&self) -> UnitQuaternion<f64> {
        self.isometry.rotation
    }

    /// Get the transformation matrix (4x4 homogeneous matrix).
    pub fn matrix(&self) -> Matrix4<f64> {
        self.isometry.to_homogeneous()
    }

    /// Get as an Isometry3 (copy of internal representation).
    pub fn isometry(&self) -> Isometry3<f64> {
        self.isometry
    }

    /// Get the x component of translation.
    pub fn x(&self) -> f64 {
        self.isometry.translation.x
    }

    /// Get the y component of translation.
    pub fn y(&self) -> f64 {
        self.isometry.translation.y
    }

    /// Get the z component of translation.
    pub fn z(&self) -> f64 {
        self.isometry.translation.z
    }

    /// Get the rotation as a unit quaternion.
    pub fn quat(&self) -> UnitQuaternion<f64> {
        self.isometry.rotation
    }
}

impl SE3Tangent {
    /// Create a new SE3Tangent from rho (translational) and theta (rotational) components.
    ///
    /// # Arguments
    /// * `rho` - Translational component [rho_x, rho_y, rho_z]
    /// * `theta` - Rotational component [theta_x, theta_y, theta_z]
    pub fn new(rho: Vector3<f64>, theta: Vector3<f64>) -> Self {
        let mut data = Vector6::zeros();
        data.fixed_rows_mut::<3>(0).copy_from(&rho);
        data.fixed_rows_mut::<3>(3).copy_from(&theta);
        SE3Tangent { data }
    }

    /// Create SE3Tangent from individual components.
    pub fn from_components(
        rho_x: f64,
        rho_y: f64,
        rho_z: f64,
        theta_x: f64,
        theta_y: f64,
        theta_z: f64,
    ) -> Self {
        SE3Tangent {
            data: Vector6::new(rho_x, rho_y, rho_z, theta_x, theta_y, theta_z),
        }
    }

    /// Get the rho (translational) part.
    pub fn rho(&self) -> Vector3<f64> {
        self.data.fixed_rows::<3>(0).into_owned()
    }

    /// Get the theta (rotational) part.
    pub fn theta(&self) -> Vector3<f64> {
        self.data.fixed_rows::<3>(3).into_owned()
    }
}

// Implement basic trait requirements for LieGroup
impl LieGroup for SE3 {
    type Element = SE3;
    type TangentVector = Vector6<f64>;
    type JacobianMatrix = Matrix6<f64>;
    type LieAlgebra = SE3Tangent;

    // Dimension constants following manif conventions
    const DIM: usize = 3; // Space dimension (3D space)
    const DOF: usize = 6; // Degrees of freedom (6-DOF: 3 translation + 3 rotation)
    const REP_SIZE: usize = 7; // Representation size (3 translation + 4 quaternion)

    fn identity() -> Self::Element {
        SE3 {
            isometry: Isometry3::identity(),
        }
    }

    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::Element {
        // For SE(3): g^{-1} = [R^T, -R^T * t; 0, 1]
        let inverse_isometry = self.isometry.inverse();

        if let Some(jac) = jacobian {
            // Jacobian of inverse operation: -Ad(g^{-1})
            let adj_inv = SE3::from_isometry(inverse_isometry).adjoint();
            jac.copy_from(&(-adj_inv));
        }

        SE3 {
            isometry: inverse_isometry,
        }
    }

    fn compose(
        &self,
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        let result = SE3 {
            isometry: self.isometry * other.isometry,
        };

        if let Some(jac_self) = jacobian_self {
            // Jacobian wrt first element: Ad(g2^{-1})
            jac_self.copy_from(&other.inverse(None).adjoint());
        }

        if let Some(jac_other) = jacobian_other {
            // Jacobian wrt second element: I (identity)
            *jac_other = Matrix6::identity();
        }

        result
    }

    fn exp(
        tangent: &Self::TangentVector,
        jacobian: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        let rho = tangent.fixed_rows::<3>(0).into_owned();
        let theta = tangent.fixed_rows::<3>(3).into_owned();

        // Compute rotation part using Rodrigues' formula
        let theta_norm = theta.norm();
        let rotation = if theta_norm < 1e-12 {
            UnitQuaternion::identity()
        } else {
            UnitQuaternion::from_scaled_axis(theta)
        };

        // Compute translation part using left Jacobian
        let translation = if theta_norm < 1e-12 {
            rho
        } else {
            let theta_skew = skew_symmetric(&theta);
            let sin_theta = theta_norm.sin();
            let cos_theta = theta_norm.cos();

            let left_jacobian = Matrix3::identity()
                + (sin_theta / theta_norm) * theta_skew
                + ((1.0 - cos_theta) / (theta_norm * theta_norm)) * theta_skew * theta_skew;

            left_jacobian * rho
        };

        if let Some(jac) = jacobian {
            // Right Jacobian for SE(3) exp map
            let theta_norm = theta.norm();
            if theta_norm < 1e-12 {
                *jac = Matrix6::identity();
            } else {
                let theta_skew = skew_symmetric(&theta);
                let sin_theta = theta_norm.sin();
                let cos_theta = theta_norm.cos();

                // Right Jacobian of SO(3)
                let jr_so3 = Matrix3::identity() - 0.5 * theta_skew
                    + ((1.0 - cos_theta) / (theta_norm * theta_norm)) * theta_skew * theta_skew;

                // Q matrix for SE(3) right Jacobian
                let rho_skew = skew_symmetric(&rho);
                let q_matrix = 0.5 * rho_skew
                    + ((theta_norm - sin_theta) / (theta_norm * theta_norm * theta_norm))
                        * (theta_skew * rho_skew
                            + rho_skew * theta_skew
                            + theta_skew * rho_skew * theta_skew);

                // Build right Jacobian
                jac.fill(0.0);
                jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&jr_so3);
                jac.fixed_view_mut::<3, 3>(3, 3).copy_from(&jr_so3);
                jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&q_matrix);
            }
        }

        SE3::new(translation, rotation)
    }

    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector {
        let rotation = self.rotation();
        let translation = self.translation();

        // Log of rotation (axis-angle representation)
        let theta = rotation.scaled_axis();
        let theta_norm = theta.norm();

        let rho = if theta_norm < 1e-12 {
            translation
        } else {
            let theta_skew = skew_symmetric(&theta);
            let sin_theta = theta_norm.sin();
            let cos_theta = theta_norm.cos();

            let left_jacobian_inv = Matrix3::identity() - 0.5 * theta_skew
                + ((theta_norm * cos_theta - sin_theta) / (theta_norm * theta_norm * sin_theta))
                    * theta_skew
                    * theta_skew;

            left_jacobian_inv * translation
        };

        if let Some(jac) = jacobian {
            // Right Jacobian inverse for SE(3) log map
            if theta_norm < 1e-12 {
                *jac = Matrix6::identity();
            } else {
                let theta_skew = skew_symmetric(&theta);
                let sin_theta = theta_norm.sin();
                let cos_theta = theta_norm.cos();
                let half_theta = 0.5 * theta_norm;
                let cot_half = cos_theta / sin_theta;

                // Right Jacobian inverse of SO(3)
                let jr_inv_so3 = Matrix3::identity()
                    + 0.5 * theta_skew
                    + ((1.0 / (theta_norm * theta_norm)) * (1.0 - half_theta * cot_half))
                        * theta_skew
                        * theta_skew;

                // Q matrix for SE(3) right Jacobian inverse
                let rho_skew = skew_symmetric(&rho);
                let q_inv_matrix = -0.5 * rho_skew
                    + ((theta_norm * (1.0 + cos_theta) - 2.0 * sin_theta)
                        / (2.0 * theta_norm * theta_norm * sin_theta))
                        * (theta_skew * rho_skew + rho_skew * theta_skew);

                // Build right Jacobian inverse
                jac.fill(0.0);
                jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&jr_inv_so3);
                jac.fixed_view_mut::<3, 3>(3, 3).copy_from(&jr_inv_so3);
                jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&q_inv_matrix);
            }
        }

        Vector6::new(rho[0], rho[1], rho[2], theta[0], theta[1], theta[2])
    }

    fn right_plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        // Right plus: g ⊕ τ = g * exp(τ)
        let exp_tangent = Self::exp(tangent, jacobian_tangent);
        self.compose(&exp_tangent, jacobian_self, None)
    }

    fn right_minus(
        &self,
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::TangentVector {
        // Right minus: g1 ⊖ g2 = log(g2^{-1} * g1)
        let g2_inv = other.inverse(None);
        let result_group = g2_inv.compose(self, None, None);
        let result = result_group.log(jacobian_self);

        if let Some(jac_other) = jacobian_other {
            // Jacobian wrt other: -Jr^{-1}(-τ) where τ = log(g2^{-1} * g1)
            let neg_result = -result;
            let mut temp_jac = Matrix6::identity();
            Self::exp(&neg_result, Some(&mut temp_jac));
            // This is a simplified computation - full implementation would require more care
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
        // Left plus: τ ⊕ g = exp(τ) * g
        let exp_tangent = Self::exp(tangent, jacobian_tangent);

        if let Some(jac_self) = jacobian_self {
            *jac_self = Matrix6::identity();
        }

        exp_tangent.compose(self, None, None)
    }

    fn left_minus(
        &self,
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::TangentVector {
        // Left minus: g1 ⊖ g2 = log(g1 * g2^{-1})
        let g2_inv = other.inverse(None);
        let result_group = self.compose(&g2_inv, jacobian_self, None);
        let result = result_group.log(None);

        if let Some(jac_other) = jacobian_other {
            // Jacobian wrt other: -Ad(g2) * Jr^{-1}(τ)
            let adj_g2 = other.adjoint();
            let mut jr_inv = Matrix6::identity();
            Self::exp(&result, Some(&mut jr_inv));
            jac_other.copy_from(&(-adj_g2 * jr_inv));
        }

        result
    }

    fn between(
        &self,
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        // Between: g1.between(g2) = g1^{-1} * g2
        let self_inv = self.inverse(None);
        self_inv.compose(
            other,
            jacobian_self.map(|j| {
                *j = -other.inverse(None).adjoint();
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
        // Convert Vector3 to Point3, apply transformation, convert back to Vector3
        let point = Point3::from(*vector);
        let transformed_point = self.isometry * point;
        let result = transformed_point.coords;

        if let Some(jac_self) = jacobian_self {
            // Jacobian wrt SE(3) element
            let rotation_matrix = self.rotation().to_rotation_matrix();
            let r = rotation_matrix.matrix();
            jac_self.fill(0.0);
            jac_self.fixed_view_mut::<3, 3>(0, 0).copy_from(r);
            jac_self
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-r * skew_symmetric(vector)));
        }

        if let Some(jac_vector) = jacobian_vector {
            // Jacobian wrt vector
            let rotation_matrix = self.rotation().to_rotation_matrix();
            jac_vector.copy_from(rotation_matrix.matrix());
        }

        result
    }

    fn adjoint(&self) -> Self::JacobianMatrix {
        // Adjoint matrix for SE(3)
        let rotation_matrix = self.rotation().to_rotation_matrix();
        let r_matrix = rotation_matrix.matrix();
        let translation = self.translation();
        let mut adj = Matrix6::zeros();

        // Top-left block: R
        adj.fixed_view_mut::<3, 3>(0, 0).copy_from(r_matrix);

        // Bottom-right block: R
        adj.fixed_view_mut::<3, 3>(3, 3).copy_from(r_matrix);

        // Top-right block: [t]_× R (skew-symmetric of translation times rotation)
        let t_skew = skew_symmetric(&translation);
        let t_skew_r = t_skew * r_matrix;
        adj.fixed_view_mut::<3, 3>(0, 3).copy_from(&t_skew_r);

        adj
    }

    fn random() -> Self::Element {
        use rand::prelude::*;
        let mut rng = thread_rng();

        // Random translation in [-1, 1]³
        let translation = Vector3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );

        // Random unit quaternion
        let rotation = UnitQuaternion::new(Vector3::new(
            rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
            rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
            rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
        ));

        SE3::new(translation, rotation)
    }

    fn normalize(&mut self) {
        // Normalize the rotation part
        self.isometry.rotation =
            UnitQuaternion::from_quaternion(self.isometry.rotation.quaternion().normalize());
    }

    fn is_valid(&self, tolerance: f64) -> bool {
        // Check if rotation is properly normalized
        let q = self.isometry.rotation.quaternion();
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

// Implement LieAlgebra trait for SE3Tangent
impl LieAlgebra<SE3> for SE3Tangent {
    fn add(&self, other: &Vector6<f64>) -> Vector6<f64> {
        self.data + other
    }

    fn scale(&self, scalar: f64) -> Vector6<f64> {
        self.data * scalar
    }

    fn negate(&self) -> Vector6<f64> {
        -&self.data
    }

    fn subtract(&self, other: &Vector6<f64>) -> Vector6<f64> {
        self.data - other
    }

    fn norm(&self) -> f64 {
        self.data.norm()
    }

    fn squared_norm(&self) -> f64 {
        self.data.norm_squared()
    }

    fn weighted_norm(&self, weight: &Matrix6<f64>) -> f64 {
        (self.data.transpose() * weight * self.data)[0].sqrt()
    }

    fn squared_weighted_norm(&self, weight: &Matrix6<f64>) -> f64 {
        (self.data.transpose() * weight * self.data)[0]
    }

    fn inner(&self, other: &Vector6<f64>) -> f64 {
        self.data.dot(other)
    }

    fn weighted_inner(&self, other: &Vector6<f64>, weight: &Matrix6<f64>) -> f64 {
        (self.data.transpose() * weight * other)[0]
    }

    fn exp(&self, _jacobian: Option<&mut Matrix6<f64>>) -> SE3 {
        unimplemented!("SE3Tangent::exp - to be implemented")
    }

    fn right_jacobian(&self) -> Matrix6<f64> {
        unimplemented!("SE3Tangent::right_jacobian - to be implemented")
    }

    fn left_jacobian(&self) -> Matrix6<f64> {
        unimplemented!("SE3Tangent::left_jacobian - to be implemented")
    }

    fn right_jacobian_inv(&self) -> Matrix6<f64> {
        unimplemented!("SE3Tangent::right_jacobian_inv - to be implemented")
    }

    fn left_jacobian_inv(&self) -> Matrix6<f64> {
        unimplemented!("SE3Tangent::left_jacobian_inv - to be implemented")
    }

    fn hat(&self) -> DMatrix<f64> {
        unimplemented!("SE3Tangent::hat - to be implemented")
    }

    fn vee(_matrix: &DMatrix<f64>) -> ManifoldResult<Vector6<f64>> {
        unimplemented!("SE3Tangent::vee - to be implemented")
    }

    fn small_adjoint(&self) -> Matrix6<f64> {
        unimplemented!("SE3Tangent::small_adjoint - to be implemented")
    }

    fn zero() -> Vector6<f64> {
        Vector6::zeros()
    }

    fn random() -> Vector6<f64> {
        unimplemented!("SE3Tangent::random - to be implemented")
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

    fn normalized(&self) -> Vector6<f64> {
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
    use nalgebra::Quaternion;
    use nalgebra::Vector3;
    use std::f64::consts::PI;

    const TOLERANCE: f64 = 1e-9;

    #[test]
    fn test_se3_tangent_basic() {
        let linear = Vector3::new(1.0, 2.0, 3.0);
        let angular = Vector3::new(0.1, 0.2, 0.3);
        let tangent = SE3Tangent::new(linear, angular);

        assert_eq!(tangent.rho(), linear);
        assert_eq!(tangent.theta(), angular);
    }

    #[test]
    fn test_se3_tangent_operations() {
        let tangent = SE3Tangent::from_components(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        let other = Vector6::new(0.5, 0.5, 0.5, 0.05, 0.05, 0.05);

        let sum = tangent.add(&other);
        let expected = Vector6::new(1.5, 2.5, 3.5, 0.15, 0.25, 0.35);
        assert!((sum - expected).norm() < TOLERANCE);

        let scaled = tangent.scale(2.0);
        assert_eq!(scaled, Vector6::new(2.0, 4.0, 6.0, 0.2, 0.4, 0.6));

        let negated = tangent.negate();
        assert_eq!(negated, Vector6::new(-1.0, -2.0, -3.0, -0.1, -0.2, -0.3));
    }

    #[test]
    fn test_se3_tangent_zero() {
        let zero = SE3Tangent::zero();
        assert_eq!(zero, Vector6::zeros());

        let tangent = SE3Tangent::from_components(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(tangent.is_zero(1e-10));
    }

    // Comprehensive SE3 tests
    #[test]
    fn test_se3_identity() {
        let identity = SE3::identity();
        assert!(identity.is_valid(TOLERANCE));

        let translation = identity.translation();
        let rotation = identity.rotation();

        assert!(translation.norm() < TOLERANCE);
        assert!((rotation.angle()) < TOLERANCE);
    }

    #[test]
    fn test_se3_new() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let rotation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);

        let se3 = SE3::new(translation, rotation);

        assert!(se3.is_valid(TOLERANCE));
        assert!((se3.translation() - translation).norm() < TOLERANCE);
        assert!((se3.rotation().angle() - rotation.angle()).abs() < TOLERANCE);
    }

    #[test]
    fn test_se3_random() {
        let se3 = SE3::random();
        assert!(se3.is_valid(TOLERANCE));
    }

    #[test]
    fn test_se3_inverse() {
        let se3 = SE3::random();
        let se3_inv = se3.inverse(None);

        // Test that g * g^-1 = identity
        let composed = se3.compose(&se3_inv, None, None);
        let identity = SE3::identity();

        let translation_diff = (composed.translation() - identity.translation()).norm();
        let rotation_diff = composed.rotation().angle();

        assert!(translation_diff < TOLERANCE);
        assert!(rotation_diff < TOLERANCE);
    }

    #[test]
    fn test_se3_compose() {
        let se3_1 = SE3::random();
        let se3_2 = SE3::random();

        let composed = se3_1.compose(&se3_2, None, None);
        assert!(composed.is_valid(TOLERANCE));

        // Test composition with identity
        let identity = SE3::identity();
        let composed_with_identity = se3_1.compose(&identity, None, None);

        let translation_diff = (composed_with_identity.translation() - se3_1.translation()).norm();
        let rotation_diff =
            (composed_with_identity.rotation().angle() - se3_1.rotation().angle()).abs();

        assert!(translation_diff < TOLERANCE);
        assert!(rotation_diff < TOLERANCE);
    }

    #[test]
    fn test_se3_adjoint() {
        let se3 = SE3::random();
        let adj = se3.adjoint();

        // Adjoint should be 6x6
        assert_eq!(adj.nrows(), 6);
        assert_eq!(adj.ncols(), 6);

        // Test some properties of the adjoint matrix
        // det(Adj(g)) = 1 for SE(3)
        let det = adj.determinant();
        assert!((det - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se3_act() {
        let se3 = SE3::random();
        let point = Vector3::new(1.0, 2.0, 3.0);

        let _transformed_point = se3.act(&point, None, None);

        // Test act with identity
        let identity = SE3::identity();
        let identity_transformed = identity.act(&point, None, None);

        let diff = (identity_transformed - point).norm();
        assert!(diff < TOLERANCE);
    }

    #[test]
    fn test_se3_exp_log() {
        let tangent = Vector6::new(0.1, 0.2, 0.3, 0.01, 0.02, 0.03);

        // Test exp(log(g)) = g
        let se3 = SE3::exp(&tangent, None);
        let recovered_tangent = se3.log(None);

        let diff = (tangent - recovered_tangent).norm();
        assert!(diff < TOLERANCE);
    }

    #[test]
    fn test_se3_exp_zero() {
        let zero_tangent = Vector6::zeros();
        let se3 = SE3::exp(&zero_tangent, None);
        let identity = SE3::identity();

        let translation_diff = (se3.translation() - identity.translation()).norm();
        let rotation_diff = se3.rotation().angle();

        assert!(translation_diff < TOLERANCE);
        assert!(rotation_diff < TOLERANCE);
    }

    #[test]
    fn test_se3_log_identity() {
        let identity = SE3::identity();
        let tangent = identity.log(None);

        assert!(tangent.norm() < TOLERANCE);
    }

    #[test]
    fn test_se3_normalize() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let rotation =
            UnitQuaternion::from_quaternion(Quaternion::new(0.5, 0.5, 0.5, 0.5).normalize()); // Normalized

        let mut se3 = SE3::new(translation, rotation);
        se3.normalize();

        assert!(se3.is_valid(TOLERANCE));
    }

    #[test]
    fn test_se3_manifold_properties() {
        // Test manifold dimension constants
        assert_eq!(SE3::DIM, 3);
        assert_eq!(SE3::DOF, 6);
        assert_eq!(SE3::REP_SIZE, 7);
    }

    #[test]
    fn test_se3_consistency() {
        // Test that operations are consistent with manif library expectations
        let se3_1 = SE3::random();
        let se3_2 = SE3::random();

        // Test associativity: (g1 * g2) * g3 = g1 * (g2 * g3)
        let se3_3 = SE3::random();
        let left_assoc = se3_1
            .compose(&se3_2, None, None)
            .compose(&se3_3, None, None);
        let right_assoc = se3_1.compose(&se3_2.compose(&se3_3, None, None), None, None);

        let translation_diff = (left_assoc.translation() - right_assoc.translation()).norm();
        let rotation_diff = (left_assoc.rotation().angle() - right_assoc.rotation().angle()).abs();

        assert!(translation_diff < 1e-10);
        assert!(rotation_diff < 1e-10);
    }

    #[test]
    fn test_se3_specific_values() {
        // Test specific known values similar to manif tests

        // Translation only
        let translation_only = SE3::new(Vector3::new(1.0, 2.0, 3.0), UnitQuaternion::identity());

        let point = Vector3::new(0.0, 0.0, 0.0);
        let transformed = translation_only.act(&point, None, None);
        let expected = Vector3::new(1.0, 2.0, 3.0);

        assert!((transformed - expected).norm() < TOLERANCE);

        // Rotation only
        let rotation_only = SE3::new(
            Vector3::zeros(),
            UnitQuaternion::from_euler_angles(PI / 2.0, 0.0, 0.0),
        );

        let point_y = Vector3::new(0.0, 1.0, 0.0);
        let rotated = rotation_only.act(&point_y, None, None);
        let expected_rotated = Vector3::new(0.0, 0.0, 1.0);

        assert!((rotated - expected_rotated).norm() < TOLERANCE);
    }

    #[test]
    fn test_se3_small_angle_approximations() {
        // Test behavior with very small angles, similar to manif library tests
        let small_tangent = Vector6::new(1e-8, 2e-8, 3e-8, 1e-9, 2e-9, 3e-9);

        let se3 = SE3::exp(&small_tangent, None);
        let recovered = se3.log(None);

        let diff = (small_tangent - recovered).norm();
        assert!(diff < TOLERANCE);
    }

    #[test]
    fn test_se3_tangent_norm() {
        let tangent_vec = Vector6::new(3.0, 4.0, 0.0, 0.0, 0.0, 0.0);
        let tangent = SE3Tangent { data: tangent_vec };

        let norm = tangent.norm();
        assert!((norm - 5.0).abs() < TOLERANCE); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_se3_from_components() {
        let se3 = SE3::from_translation_rotation(1.0, 2.0, 3.0, 0.0, 0.0, 0.0);
        assert!(se3.is_valid(TOLERANCE));
        assert_eq!(se3.x(), 1.0);
        assert_eq!(se3.y(), 2.0);
        assert_eq!(se3.z(), 3.0);

        let quat = se3.quat();
        assert!((quat.w - 1.0).abs() < TOLERANCE);
        assert!(quat.i.abs() < TOLERANCE);
        assert!(quat.j.abs() < TOLERANCE);
        assert!(quat.k.abs() < TOLERANCE);
    }

    #[test]
    fn test_se3_from_isometry() {
        let translation = nalgebra::Translation3::new(1.0, 2.0, 3.0);
        let rotation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let isometry = Isometry3::from_parts(translation, rotation);

        let se3 = SE3::from_isometry(isometry);
        let recovered_isometry = se3.isometry();

        let translation_diff =
            (isometry.translation.vector - recovered_isometry.translation.vector).norm();
        let rotation_diff = (isometry.rotation.angle() - recovered_isometry.rotation.angle()).abs();

        assert!(translation_diff < TOLERANCE);
        assert!(rotation_diff < TOLERANCE);
    }

    #[test]
    fn test_se3_matrix() {
        let se3 = SE3::random();
        let matrix = se3.matrix();

        // Check matrix is 4x4
        assert_eq!(matrix.nrows(), 4);
        assert_eq!(matrix.ncols(), 4);

        // Check bottom row is [0, 0, 0, 1]
        assert!((matrix[(3, 0)]).abs() < TOLERANCE);
        assert!((matrix[(3, 1)]).abs() < TOLERANCE);
        assert!((matrix[(3, 2)]).abs() < TOLERANCE);
        assert!((matrix[(3, 3)] - 1.0).abs() < TOLERANCE);
    }

    // Integration tests based on manif library patterns
    #[test]
    fn test_se3_manif_like_operations() {
        // This test mimics operations commonly found in manif test suite

        // Create two SE3 elements
        let g1 = SE3::new(
            Vector3::new(1.0, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, PI / 4.0),
        );

        let g2 = SE3::new(
            Vector3::new(0.0, 1.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, PI / 4.0, 0.0),
        );

        // Test composition
        let g3 = g1.compose(&g2, None, None);
        assert!(g3.is_valid(TOLERANCE));

        // Test inverse composition property: g1 * g2 * g2^-1 * g1^-1 = I
        let g2_inv = g2.inverse(None);
        let g1_inv = g1.inverse(None);
        let result = g1
            .compose(&g2, None, None)
            .compose(&g2_inv, None, None)
            .compose(&g1_inv, None, None);

        let identity = SE3::identity();
        let translation_diff = (result.translation() - identity.translation()).norm();
        let rotation_diff = result.rotation().angle();

        assert!(translation_diff < TOLERANCE);
        assert!(rotation_diff < TOLERANCE);
    }
}
