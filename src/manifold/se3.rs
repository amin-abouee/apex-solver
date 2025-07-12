//! SE(3) - Special Euclidean Group in 3D
//!
//! This module implements the Special Euclidean group SE(3), which represents
//! rigid body transformations in 3D space (rotation + translation).
//!
//! SE(3) elements are represented as a combination of SO(3) rotation and Vector3 translation.
//! SE(3) tangent elements are represented as [rho(3), theta(3)] = 6 components,
//! where rho is the translational component and theta is the rotational component.
//!
//! The implementation follows the [manif](https://github.com/artivis/manif) C++ library
//! conventions and provides all operations required by the LieGroup and LieAlgebra traits.

use crate::manifold::so3::{SO3, SO3Tangent};
use crate::manifold::{LieAlgebra, LieGroup};
use nalgebra::{
    DMatrix, Isometry3, Matrix3, Matrix4, Matrix6, Quaternion, Translation3, UnitQuaternion,
    Vector3, Vector6,
};
use std::fmt;

/// SE(3) group element representing rigid body transformations in 3D.
///
/// Represented as a combination of SO(3) rotation and Vector3 translation.
#[derive(Clone, Debug, PartialEq)]
pub struct SE3 {
    /// Rotation part as SO(3) element
    rotation: SO3,
    /// Translation part as Vector3
    translation: Vector3<f64>,
}

impl fmt::Display for SE3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = self.translation();
        let q = self.rotation_quaternion();
        write!(
            f,
            "SE3(translation: [{:.4}, {:.4}, {:.4}], rotation: [w: {:.4}, x: {:.4}, y: {:.4}, z: {:.4}])",
            t.x, t.y, t.z, q.w, q.i, q.j, q.k
        )
    }
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

impl fmt::Display for SE3Tangent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rho = self.rho();
        let theta = self.theta();
        write!(
            f,
            "se3(rho: [{:.4}, {:.4}, {:.4}], theta: [{:.4}, {:.4}, {:.4}])",
            rho.x, rho.y, rho.z, theta.x, theta.y, theta.z
        )
    }
}

impl SE3 {
    /// Create a new SE3 element from translation and rotation.
    ///
    /// # Arguments
    /// * `translation` - Translation vector [x, y, z]
    /// * `rotation` - Unit quaternion representing rotation
    pub fn new(translation: Vector3<f64>, rotation: UnitQuaternion<f64>) -> Self {
        SE3 {
            rotation: SO3::new(rotation),
            translation,
        }
    }

    /// Create SE3 from translation components and Euler angles.
    pub fn from_translation_quaternion(
        x: f64,
        y: f64,
        z: f64,
        qw: f64,
        qx: f64,
        qy: f64,
        qz: f64,
    ) -> Self {
        let translation = Vector3::new(x, y, z);
        let quaternion =
            UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz).normalize());
        Self::new(translation, quaternion)
    }

    /// Create SE3 from translation components and Euler angles.
    pub fn from_translation_euler(x: f64, y: f64, z: f64, roll: f64, pitch: f64, yaw: f64) -> Self {
        let translation = Vector3::new(x, y, z);
        let rotation = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
        Self::new(translation, rotation)
    }

    /// Create SE3 directly from an Isometry3.
    pub fn from_isometry(isometry: Isometry3<f64>) -> Self {
        SE3 {
            rotation: SO3::new(isometry.rotation),
            translation: isometry.translation.vector,
        }
    }

    /// Create SE3 from SO3 and Vector3 components.
    pub fn from_translation_so3(translation: Vector3<f64>, rotation: SO3) -> Self {
        SE3 {
            rotation,
            translation,
        }
    }

    /// Get the translation part as a Vector3.
    pub fn translation(&self) -> Vector3<f64> {
        self.translation
    }

    /// Get the rotation part as SO3.
    pub fn rotation(&self) -> SO3 {
        self.rotation.clone()
    }

    /// Get the rotation part as a UnitQuaternion.
    pub fn rotation_quaternion(&self) -> UnitQuaternion<f64> {
        self.rotation.quaternion()
    }

    /// Get as an Isometry3 (convenience method).
    pub fn isometry(&self) -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::from(self.translation),
            self.rotation_quaternion(),
        )
    }

    /// Get the transformation matrix (4x4 homogeneous matrix).
    pub fn matrix(&self) -> Matrix4<f64> {
        self.isometry().to_homogeneous()
    }

    /// Get the x component of translation.
    pub fn x(&self) -> f64 {
        self.translation.x
    }

    /// Get the y component of translation.
    pub fn y(&self) -> f64 {
        self.translation.y
    }

    /// Get the z component of translation.
    pub fn z(&self) -> f64 {
        self.translation.z
    }

    /// Get the rotation as a unit quaternion.
    pub fn quat(&self) -> UnitQuaternion<f64> {
        self.rotation_quaternion()
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
            rotation: SO3::identity(),
            translation: Vector3::zeros(),
        }
    }

    /// Get the inverse.
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian matrix of the inverse wrt this.
    ///
    /// # Notes
    /// # Equation 170: Inverse of SE(3) matrix
    /// M⁻¹ = [ Rᵀ -Rᵀt ]
    ///       [ 0    1   ]
    ///
    /// # Equation 176: Jacobian of inverse operation
    /// J_M⁻¹_M = - [ R [t]ₓ R ]
    ///             [ 0    R   ]
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::Element {
        // For SE(3): g^{-1} = [R^T, -R^T * t; 0, 1]
        let rot_inv = self.rotation.inverse(None);
        let trans_inv = -rot_inv.act(&self.translation, None, None);

        // Eqs. 176
        if let Some(jac) = jacobian {
            // Jacobian of inverse operation: -Ad(g^)
            let adjoint_matrix = self.adjoint();
            jac.copy_from(&(-adjoint_matrix));
        }

        SE3::from_translation_so3(trans_inv, rot_inv)
    }

    /// Composition of this and another SE3 element.
    ///
    /// # Arguments
    /// * `other` - Another SE3 element.
    /// * `jacobian_self` - Optional Jacobian matrix of the composition wrt this.
    /// * `jacobian_other` - Optional Jacobian matrix of the composition wrt other.
    ///
    /// # Notes
    /// # Equation 171: Composition of SE(3) matrices
    /// M_a M_b = [ R_a*R_b   R_a*t_b + t_a ]
    ///           [ 0             1         ]
    ///
    /// # Equation 177: Jacobian of the composition wrt self.
    /// J_MaMb_Ma = [ R_bᵀ   -R_bᵀ*[t_b]ₓ ]
    ///             [ 0          R_bᵀ     ]
    ///
    /// # Equation 178: Jacobian of the composition wrt other.
    /// J_MaMb_Mb = I_6
    ///
    fn compose(
        &self,
        other: &Self::Element,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        // Eqs. 171
        let composed_rotation = self.rotation.compose(&other.rotation, None, None);
        let composed_translation =
            self.rotation.act(&other.translation, None, None) + self.translation;

        let result = SE3::from_translation_so3(composed_translation, composed_rotation);

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

    /// Get the SE3 element.
    ///
    /// # Arguments
    /// * `tangent` - Tangent vector [rho, theta]
    /// * `jacobian` - Optional Jacobian matrix of the SE3 element wrt this.
    ///
    /// # Notes
    /// # Equation 172: SE(3) exponential map
    /// M = exp(τ) = [ R(θ)   t(ρ) ]
    ///              [ 0       1   ]
    fn exp(
        tangent: &Self::TangentVector,
        jacobian: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        let rho = tangent.fixed_rows::<3>(0).into_owned();
        let theta = tangent.fixed_rows::<3>(3).into_owned();

        // Compute rotation part using SO(3) exponential
        let rotation = SO3::exp(&theta, None);

        // Compute translation part using left Jacobian
        let theta_norm = theta.norm();
        let translation = if theta_norm < 1e-12 {
            rho
        } else {
            let theta_hat = SO3Tangent::new(theta).hat();
            let theta_hat_matrix = Matrix3::new(
                theta_hat[(0, 0)],
                theta_hat[(0, 1)],
                theta_hat[(0, 2)],
                theta_hat[(1, 0)],
                theta_hat[(1, 1)],
                theta_hat[(1, 2)],
                theta_hat[(2, 0)],
                theta_hat[(2, 1)],
                theta_hat[(2, 2)],
            );
            let sin_theta = theta_norm.sin();
            let cos_theta = theta_norm.cos();

            let left_jacobian = Matrix3::identity()
                + (sin_theta / theta_norm) * theta_hat_matrix
                + ((1.0 - cos_theta) / (theta_norm * theta_norm))
                    * theta_hat_matrix
                    * theta_hat_matrix;

            left_jacobian * rho
        };

        if let Some(jac) = jacobian {
            // Right Jacobian for SE(3) exp map
            let theta_norm = theta.norm();
            if theta_norm < 1e-12 {
                *jac = Matrix6::identity();
            } else {
                let theta_hat = SO3Tangent::new(theta).hat();
                let theta_hat_matrix = Matrix3::new(
                    theta_hat[(0, 0)],
                    theta_hat[(0, 1)],
                    theta_hat[(0, 2)],
                    theta_hat[(1, 0)],
                    theta_hat[(1, 1)],
                    theta_hat[(1, 2)],
                    theta_hat[(2, 0)],
                    theta_hat[(2, 1)],
                    theta_hat[(2, 2)],
                );
                let sin_theta = theta_norm.sin();
                let cos_theta = theta_norm.cos();

                // Right Jacobian of SO(3)
                let jr_so3 = Matrix3::identity() - 0.5 * theta_hat_matrix
                    + ((1.0 - cos_theta) / (theta_norm * theta_norm))
                        * theta_hat_matrix
                        * theta_hat_matrix;

                // Q matrix for SE(3) right Jacobian
                let rho_hat = SO3Tangent::new(rho).hat();
                let rho_hat_matrix = Matrix3::new(
                    rho_hat[(0, 0)],
                    rho_hat[(0, 1)],
                    rho_hat[(0, 2)],
                    rho_hat[(1, 0)],
                    rho_hat[(1, 1)],
                    rho_hat[(1, 2)],
                    rho_hat[(2, 0)],
                    rho_hat[(2, 1)],
                    rho_hat[(2, 2)],
                );
                let q_matrix = 0.5 * rho_hat_matrix
                    + ((theta_norm - sin_theta) / (theta_norm * theta_norm * theta_norm))
                        * (theta_hat_matrix * rho_hat_matrix
                            + rho_hat_matrix * theta_hat_matrix
                            + theta_hat_matrix * rho_hat_matrix * theta_hat_matrix);

                // Build right Jacobian
                jac.fill(0.0);
                jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&jr_so3);
                jac.fixed_view_mut::<3, 3>(3, 3).copy_from(&jr_so3);
                jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&q_matrix);
            }
        }

        SE3::from_translation_so3(translation, rotation)
    }

    /// Get the SE3 corresponding Lie algebra element in vector form.
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian matrix of the tangent wrt to this.
    ///
    /// # Notes
    /// # Equation 173: SE(3) logarithmic map
    /// τ = log(M) = [ V⁻¹(θ) t ]
    ///              [ Log(R)  ]
    ///
    /// # Equation 174: V(θ) function for SE(3) Log/Exp maps
    /// V(θ) = I + (1 - cos θ)/θ² [θ]ₓ + (θ - sin θ)/θ³ [θ]ₓ²

    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector {
        // Log of rotation (axis-angle representation)
        let theta = self.rotation.log(None);
        let theta_norm = theta.norm();

        let rho = if theta_norm < 1e-12 {
            self.translation
        } else {
            let theta_hat = SO3Tangent::new(theta).hat();
            let theta_hat_matrix = Matrix3::new(
                theta_hat[(0, 0)],
                theta_hat[(0, 1)],
                theta_hat[(0, 2)],
                theta_hat[(1, 0)],
                theta_hat[(1, 1)],
                theta_hat[(1, 2)],
                theta_hat[(2, 0)],
                theta_hat[(2, 1)],
                theta_hat[(2, 2)],
            );
            let sin_theta = theta_norm.sin();
            let cos_theta = theta_norm.cos();

            let left_jacobian_inv = Matrix3::identity() - 0.5 * theta_hat_matrix
                + ((theta_norm * cos_theta - sin_theta) / (theta_norm * theta_norm * sin_theta))
                    * theta_hat_matrix
                    * theta_hat_matrix;

            left_jacobian_inv * self.translation
        };

        if let Some(jac) = jacobian {
            // Right Jacobian inverse for SE(3) log map
            if theta_norm < 1e-12 {
                *jac = Matrix6::identity();
            } else {
                let theta_hat = SO3Tangent::new(theta).hat();
                let theta_hat_matrix = Matrix3::new(
                    theta_hat[(0, 0)],
                    theta_hat[(0, 1)],
                    theta_hat[(0, 2)],
                    theta_hat[(1, 0)],
                    theta_hat[(1, 1)],
                    theta_hat[(1, 2)],
                    theta_hat[(2, 0)],
                    theta_hat[(2, 1)],
                    theta_hat[(2, 2)],
                );
                let sin_theta = theta_norm.sin();
                let cos_theta = theta_norm.cos();
                let half_theta = 0.5 * theta_norm;
                let cot_half = cos_theta / sin_theta;

                // Right Jacobian inverse of SO(3)
                let jr_inv_so3 = Matrix3::identity()
                    + 0.5 * theta_hat_matrix
                    + ((1.0 / (theta_norm * theta_norm)) * (1.0 - half_theta * cot_half))
                        * theta_hat_matrix
                        * theta_hat_matrix;

                // Q matrix for SE(3) right Jacobian inverse
                let rho_hat = SO3Tangent::new(rho).hat();
                let rho_hat_matrix = Matrix3::new(
                    rho_hat[(0, 0)],
                    rho_hat[(0, 1)],
                    rho_hat[(0, 2)],
                    rho_hat[(1, 0)],
                    rho_hat[(1, 1)],
                    rho_hat[(1, 2)],
                    rho_hat[(2, 0)],
                    rho_hat[(2, 1)],
                    rho_hat[(2, 2)],
                );
                let q_inv_matrix = -0.5 * rho_hat_matrix
                    + ((theta_norm * (1.0 + cos_theta) - 2.0 * sin_theta)
                        / (2.0 * theta_norm * theta_norm * sin_theta))
                        * (theta_hat_matrix * rho_hat_matrix + rho_hat_matrix * theta_hat_matrix);

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
        // Apply SE(3) transformation: R * v + t
        let result = self.rotation.act(vector, None, None) + self.translation;

        if let Some(jac_self) = jacobian_self {
            // Jacobian wrt SE(3) element
            let rotation_matrix = self.rotation.rotation();
            jac_self.fill(0.0);
            jac_self
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&rotation_matrix);

            let vector_hat = SO3Tangent::new(*vector).hat();
            let vector_hat_matrix = Matrix3::new(
                vector_hat[(0, 0)],
                vector_hat[(0, 1)],
                vector_hat[(0, 2)],
                vector_hat[(1, 0)],
                vector_hat[(1, 1)],
                vector_hat[(1, 2)],
                vector_hat[(2, 0)],
                vector_hat[(2, 1)],
                vector_hat[(2, 2)],
            );
            jac_self
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-rotation_matrix * vector_hat_matrix));
        }

        if let Some(jac_vector) = jacobian_vector {
            // Jacobian wrt vector
            let rotation_matrix = self.rotation.rotation();
            jac_vector.copy_from(&rotation_matrix);
        }

        result
    }

    fn adjoint(&self) -> Self::JacobianMatrix {
        // Adjoint matrix for SE(3)
        let rotation_matrix = self.rotation.rotation();
        let translation = self.translation;
        let mut adj = Matrix6::zeros();

        // Top-left block: R
        adj.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation_matrix);

        // Bottom-right block: R
        adj.fixed_view_mut::<3, 3>(3, 3).copy_from(&rotation_matrix);

        // Top-right block: [t]_× R (skew-symmetric of translation times rotation)
        let t_hat = SO3Tangent::new(translation).hat();
        let t_hat_matrix = Matrix3::new(
            t_hat[(0, 0)],
            t_hat[(0, 1)],
            t_hat[(0, 2)],
            t_hat[(1, 0)],
            t_hat[(1, 1)],
            t_hat[(1, 2)],
            t_hat[(2, 0)],
            t_hat[(2, 1)],
            t_hat[(2, 2)],
        );
        let t_hat_r = t_hat_matrix * rotation_matrix;
        adj.fixed_view_mut::<3, 3>(0, 3).copy_from(&t_hat_r);

        adj
    }

    fn random() -> Self::Element {
        use rand::Rng;
        let mut rng = rand::rng();

        // Random translation in [-1, 1]³
        let translation = Vector3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        );

        // Random rotation
        let rotation = SO3::random();

        SE3::from_so3_translation(rotation, translation)
    }

    fn normalize(&mut self) {
        // Normalize the rotation part
        self.rotation.normalize();
    }

    fn is_valid(&self, tolerance: f64) -> bool {
        // Check if rotation is valid
        self.rotation.is_valid(tolerance)
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
    // Vector space operations

    /// Vector space addition: φ₁ + φ₂.
    ///
    /// Adds two tangent vectors component-wise following standard vector addition.
    ///
    /// # Arguments
    /// * `other` - The tangent vector to add
    ///
    /// # Returns
    /// The sum of the two tangent vectors
    fn add(&self, other: &Vector6<f64>) -> Vector6<f64> {
        self.data + other
    }

    /// Scalar multiplication: α · φ.
    ///
    /// Multiplies the tangent vector by a scalar following standard scalar multiplication.
    ///
    /// # Arguments  
    /// * `scalar` - Scalar multiplier
    ///
    /// # Returns
    /// The scaled tangent vector
    fn scale(&self, scalar: f64) -> Vector6<f64> {
        self.data * scalar
    }

    /// Additive inverse: -φ.
    ///
    /// Returns the additive inverse of the tangent vector.
    ///
    /// # Returns
    /// The negated tangent vector
    fn negate(&self) -> Vector6<f64> {
        -self.data
    }

    /// Vector subtraction: φ₁ - φ₂.
    ///
    /// Subtracts the given tangent vector from this one.
    ///
    /// # Arguments
    /// * `other` - The tangent vector to subtract
    ///
    /// # Returns
    /// The difference of the two tangent vectors
    fn subtract(&self, other: &Vector6<f64>) -> Vector6<f64> {
        self.data - other
    }

    // Norms and inner products

    /// Euclidean norm: ||φ||.
    ///
    /// Computes the standard Euclidean norm of the tangent vector.
    ///
    /// # Returns
    /// The Euclidean norm of the tangent vector
    fn norm(&self) -> f64 {
        self.data.norm()
    }

    /// Squared norm: ||φ||².
    ///
    /// Computes the squared Euclidean norm for efficiency.
    ///
    /// # Returns
    /// The squared norm of the tangent vector
    fn squared_norm(&self) -> f64 {
        self.data.norm_squared()
    }

    /// Weighted norm: √(φᵀ W φ).
    ///
    /// Computes the weighted norm using the given weight matrix.
    ///
    /// # Arguments
    /// * `weight` - Weight matrix W (6x6 for SE(3))
    ///
    /// # Returns
    /// The weighted norm
    fn weighted_norm(&self, weight: &Matrix6<f64>) -> f64 {
        (self.data.transpose() * weight * self.data)[0].sqrt()
    }

    /// Squared weighted norm: φᵀ W φ.
    ///
    /// Computes the squared weighted norm for efficiency.
    ///
    /// # Arguments
    /// * `weight` - Weight matrix W (6x6 for SE(3))
    ///
    /// # Returns
    /// The squared weighted norm
    fn squared_weighted_norm(&self, weight: &Matrix6<f64>) -> f64 {
        (self.data.transpose() * weight * self.data)[0]
    }

    /// Inner product: ⟨φ₁, φ₂⟩.
    ///
    /// Computes the standard inner product between two tangent vectors.
    ///
    /// # Arguments
    /// * `other` - The second tangent vector
    ///
    /// # Returns
    /// The inner product
    fn inner(&self, other: &Vector6<f64>) -> f64 {
        self.data.dot(other)
    }

    /// Weighted inner product: ⟨φ₁, W φ₂⟩.
    ///
    /// Computes the weighted inner product using the given weight matrix.
    ///
    /// # Arguments
    /// * `other` - The second tangent vector
    /// * `weight` - Weight matrix W (6x6 for SE(3))
    ///
    /// # Returns
    /// The weighted inner product
    fn weighted_inner(&self, other: &Vector6<f64>, weight: &Matrix6<f64>) -> f64 {
        (self.data.transpose() * weight * other)[0]
    }

    // Exponential map and Jacobians

    /// Exponential map to Lie group: exp(φ^∧).
    ///
    /// Maps the tangent vector to the corresponding SE(3) group element using
    /// the exponential map. This is the fundamental connection between the
    /// Lie algebra and Lie group.
    ///
    /// # Arguments
    /// * `jacobian` - Optional Jacobian ∂exp(φ^∧)/∂φ (6x6)
    ///
    /// # Returns
    /// The corresponding SE(3) element
    fn exp(&self, jacobian: Option<&mut Matrix6<f64>>) -> SE3 {
        SE3::exp(&self.data, jacobian)
    }

    /// Right Jacobian Jr.
    ///
    /// Computes the right Jacobian matrix such that for small δφ:
    /// exp((φ + δφ)^∧) ≈ exp(φ^∧) ∘ exp((Jr δφ)^∧)
    ///
    /// For SE(3), this involves computing Jacobians for both translation and rotation parts.
    ///
    /// # Returns
    /// The right Jacobian matrix (6x6)
    fn right_jacobian(&self) -> Matrix6<f64> {
        let mut jac = Matrix6::identity();
        SE3::exp(&self.data, Some(&mut jac));
        jac
    }

    /// Left Jacobian Jl.
    ///
    /// Computes the left Jacobian matrix such that for small δφ:
    /// exp((φ + δφ)^∧) ≈ exp((Jl δφ)^∧) ∘ exp(φ^∧)
    ///
    /// Following manif conventions for SE(3) left Jacobian computation.
    ///
    /// # Returns
    /// The left Jacobian matrix (6x6)
    fn left_jacobian(&self) -> Matrix6<f64> {
        let theta = self.theta();
        let theta_norm = theta.norm();

        if theta_norm < 1e-12 {
            Matrix6::identity()
        } else {
            let theta_hat = SO3Tangent::new(theta).hat();
            let theta_hat_matrix = Matrix3::new(
                theta_hat[(0, 0)],
                theta_hat[(0, 1)],
                theta_hat[(0, 2)],
                theta_hat[(1, 0)],
                theta_hat[(1, 1)],
                theta_hat[(1, 2)],
                theta_hat[(2, 0)],
                theta_hat[(2, 1)],
                theta_hat[(2, 2)],
            );
            let sin_theta = theta_norm.sin();
            let cos_theta = theta_norm.cos();

            // Left Jacobian of SO(3)
            let jl_so3 = Matrix3::identity()
                + (sin_theta / theta_norm) * theta_hat_matrix
                + ((1.0 - cos_theta) / (theta_norm * theta_norm))
                    * theta_hat_matrix
                    * theta_hat_matrix;

            let mut jl = Matrix6::zeros();
            jl.fixed_view_mut::<3, 3>(0, 0).copy_from(&jl_so3);
            jl.fixed_view_mut::<3, 3>(3, 3).copy_from(&jl_so3);

            jl
        }
    }

    /// Inverse of right Jacobian Jr⁻¹.
    ///
    /// Computes the inverse of the right Jacobian. This is used for
    /// computing perturbations and derivatives.
    ///
    /// # Returns
    /// The inverse right Jacobian matrix (6x6)
    fn right_jacobian_inv(&self) -> Matrix6<f64> {
        let theta = self.theta();
        let theta_norm = theta.norm();

        if theta_norm < 1e-12 {
            Matrix6::identity()
        } else {
            let theta_hat = SO3Tangent::new(theta).hat();
            let theta_hat_matrix = Matrix3::new(
                theta_hat[(0, 0)],
                theta_hat[(0, 1)],
                theta_hat[(0, 2)],
                theta_hat[(1, 0)],
                theta_hat[(1, 1)],
                theta_hat[(1, 2)],
                theta_hat[(2, 0)],
                theta_hat[(2, 1)],
                theta_hat[(2, 2)],
            );
            let sin_theta = theta_norm.sin();
            let cos_theta = theta_norm.cos();

            // Right Jacobian inverse of SO(3)
            let jr_inv_so3 = Matrix3::identity()
                + 0.5 * theta_hat_matrix
                + ((1.0 / (theta_norm * theta_norm))
                    * (1.0 - (0.5 * theta_norm) * (cos_theta / sin_theta)))
                    * theta_hat_matrix
                    * theta_hat_matrix;

            let mut jr_inv = Matrix6::zeros();
            jr_inv.fixed_view_mut::<3, 3>(0, 0).copy_from(&jr_inv_so3);
            jr_inv.fixed_view_mut::<3, 3>(3, 3).copy_from(&jr_inv_so3);

            jr_inv
        }
    }

    /// Inverse of left Jacobian Jl⁻¹.
    ///
    /// Computes the inverse of the left Jacobian following manif conventions.
    ///
    /// # Returns
    /// The inverse left Jacobian matrix (6x6)
    fn left_jacobian_inv(&self) -> Matrix6<f64> {
        let theta = self.theta();
        let theta_norm = theta.norm();

        if theta_norm < 1e-12 {
            Matrix6::identity()
        } else {
            let theta_hat = SO3Tangent::new(theta).hat();
            let theta_hat_matrix = Matrix3::new(
                theta_hat[(0, 0)],
                theta_hat[(0, 1)],
                theta_hat[(0, 2)],
                theta_hat[(1, 0)],
                theta_hat[(1, 1)],
                theta_hat[(1, 2)],
                theta_hat[(2, 0)],
                theta_hat[(2, 1)],
                theta_hat[(2, 2)],
            );
            let half_theta = 0.5 * theta_norm;
            let cot_half = half_theta.cos() / half_theta.sin();

            // Left Jacobian inverse of SO(3)
            let jl_inv_so3 = Matrix3::identity() - 0.5 * theta_hat_matrix
                + ((1.0 / (theta_norm * theta_norm)) * (1.0 - half_theta * cot_half))
                    * theta_hat_matrix
                    * theta_hat_matrix;

            let mut jl_inv = Matrix6::zeros();
            jl_inv.fixed_view_mut::<3, 3>(0, 0).copy_from(&jl_inv_so3);
            jl_inv.fixed_view_mut::<3, 3>(3, 3).copy_from(&jl_inv_so3);

            jl_inv
        }
    }

    // Matrix representations

    /// Hat operator: φ^∧ (vector to matrix).
    ///
    /// Converts the SE(3) tangent vector to its 4x4 matrix representation in the Lie algebra.
    /// Following manif conventions, the structure is:
    /// ```
    /// [  [theta]_×   rho  ]
    /// [      0       0    ]
    /// ```
    /// where [theta]_× is the skew-symmetric matrix of the rotational part.
    ///
    /// # Returns
    /// The 4x4 matrix representation in the SE(3) Lie algebra
    fn hat(&self) -> DMatrix<f64> {
        let mut lie_alg = DMatrix::zeros(4, 4);

        // Top-left 3x3: skew-symmetric matrix of rotational part
        let theta_hat = SO3Tangent::new(self.theta()).hat();
        lie_alg.view_mut((0, 0), (3, 3)).copy_from(&theta_hat);

        // Top-right 3x1: translational part
        let rho = self.rho();
        lie_alg[(0, 3)] = rho[0];
        lie_alg[(1, 3)] = rho[1];
        lie_alg[(2, 3)] = rho[2];

        // Bottom row is already zero (initialized with zeros)

        lie_alg
    }

    /// Vee operator: φ^∨ (matrix to vector).
    ///
    /// Extracts the tangent vector from its 4x4 matrix representation.
    /// This is the inverse of the hat operator.
    ///
    /// # Arguments
    /// * `matrix` - The 4x4 matrix in the SE(3) Lie algebra
    ///
    /// # Returns
    /// The corresponding 6-dimensional tangent vector [rho, theta]
    fn vee(matrix: &DMatrix<f64>) -> Vector6<f64> {
        // Extract translational part from top-right 3x1
        let rho = Vector3::new(matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)]);

        // Extract rotational part from top-left 3x3 skew-symmetric matrix
        // Convert DMatrix view to Matrix3 for SO3Tangent::vee compatibility
        let theta_hat = DMatrix::from_fn(3, 3, |i, j| matrix[(i, j)]);
        let theta = SO3Tangent::vee(&theta_hat);

        let mut result = Vector6::zeros();
        result.fixed_rows_mut::<3>(0).copy_from(&rho);
        result.fixed_rows_mut::<3>(3).copy_from(&theta);

        result
    }

    // Adjoint operations

    /// Small adjoint: ad(φ).
    ///
    /// Computes the small adjoint representation of the Lie algebra element.
    /// For SE(3), this implements: ad(φ) ψ = [φ^∧, ψ^∧]^∨.
    ///
    /// The resulting 6x6 matrix has the structure:
    /// ```
    /// [ [theta]_×     [rho]_×  ]
    /// [    0       [theta]_×   ]
    /// ```
    ///
    /// # Returns
    /// The small adjoint matrix (6x6)
    fn small_adjoint(&self) -> Matrix6<f64> {
        let mut small_adj = Matrix6::zeros();

        let theta_hat = SO3Tangent::new(self.theta()).hat();
        let theta_hat_matrix = Matrix3::new(
            theta_hat[(0, 0)],
            theta_hat[(0, 1)],
            theta_hat[(0, 2)],
            theta_hat[(1, 0)],
            theta_hat[(1, 1)],
            theta_hat[(1, 2)],
            theta_hat[(2, 0)],
            theta_hat[(2, 1)],
            theta_hat[(2, 2)],
        );
        let rho_hat = SO3Tangent::new(self.rho()).hat();
        let rho_hat_matrix = Matrix3::new(
            rho_hat[(0, 0)],
            rho_hat[(0, 1)],
            rho_hat[(0, 2)],
            rho_hat[(1, 0)],
            rho_hat[(1, 1)],
            rho_hat[(1, 2)],
            rho_hat[(2, 0)],
            rho_hat[(2, 1)],
            rho_hat[(2, 2)],
        );

        // Top-left and bottom-right blocks: [theta]_×
        small_adj
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&theta_hat_matrix);
        small_adj
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&theta_hat_matrix);

        // Top-right block: [rho]_×
        small_adj
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&rho_hat_matrix);

        small_adj
    }

    // Utility functions

    /// Zero tangent vector.
    ///
    /// Returns the zero element of the SE(3) tangent space.
    ///
    /// # Returns
    /// A 6-dimensional zero vector
    fn zero() -> Vector6<f64> {
        Vector6::zeros()
    }

    /// Random tangent vector (useful for testing).
    ///
    /// Generates a random tangent vector with reasonable bounds.
    /// Translation components are in [-1, 1] and rotation components in [-0.1, 0.1].
    ///
    /// # Returns
    /// A random 6-dimensional tangent vector
    fn random() -> Vector6<f64> {
        use rand::Rng;
        let mut rng = rand::rng();
        Vector6::new(
            rng.random_range(-1.0..1.0), // rho_x
            rng.random_range(-1.0..1.0), // rho_y
            rng.random_range(-1.0..1.0), // rho_z
            rng.random_range(-0.1..0.1), // theta_x
            rng.random_range(-0.1..0.1), // theta_y
            rng.random_range(-0.1..0.1), // theta_z
        )
    }

    /// Check if the tangent vector is approximately zero.
    ///
    /// Compares the norm of the tangent vector to the given tolerance.
    ///
    /// # Arguments
    /// * `tolerance` - Tolerance for zero comparison
    ///
    /// # Returns
    /// True if the norm is below the tolerance
    fn is_zero(&self, tolerance: f64) -> bool {
        self.norm() < tolerance
    }

    /// Normalize the tangent vector to unit norm.
    ///
    /// Modifies this tangent vector to have unit norm. If the vector
    /// is near zero, it remains unchanged.
    fn normalize(&mut self) {
        let norm = self.norm();
        if norm > f64::EPSILON {
            self.data /= norm;
        }
    }

    /// Return a unit tangent vector in the same direction.
    ///
    /// Returns a new tangent vector with unit norm in the same direction.
    /// If the original vector is near zero, returns zero.
    ///
    /// # Returns
    /// A normalized copy of the tangent vector
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
        let rotation = identity.rotation_quaternion();

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
        assert!((se3.rotation_quaternion().angle() - rotation.angle()).abs() < TOLERANCE);
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
        let rotation_diff = composed.rotation_quaternion().angle();

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
        let rotation_diff = (composed_with_identity.rotation_quaternion().angle()
            - se3_1.rotation_quaternion().angle())
        .abs();

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
        let rotation_diff = se3.rotation_quaternion().angle();

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
        let rotation_diff = (left_assoc.rotation_quaternion().angle()
            - right_assoc.rotation_quaternion().angle())
        .abs();

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
        let rotation_diff = result.rotation_quaternion().angle();

        assert!(translation_diff < TOLERANCE);
        assert!(rotation_diff < TOLERANCE);
    }

    #[test]
    fn test_se3_tangent_lie_algebra_operations() {
        // Test vector space operations
        let tangent1 = SE3Tangent::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(0.1, 0.2, 0.3));
        let tangent2_vec = Vector6::new(0.5, 0.5, 0.5, 0.05, 0.05, 0.05);

        // Test add
        let sum = tangent1.add(&tangent2_vec);
        let expected_sum = Vector6::new(1.5, 2.5, 3.5, 0.15, 0.25, 0.35);
        assert!((sum - expected_sum).norm() < TOLERANCE);

        // Test scale
        let scaled = tangent1.scale(2.0);
        let expected_scaled = Vector6::new(2.0, 4.0, 6.0, 0.2, 0.4, 0.6);
        assert!((scaled - expected_scaled).norm() < TOLERANCE);

        // Test negate
        let negated = tangent1.negate();
        let expected_negated = Vector6::new(-1.0, -2.0, -3.0, -0.1, -0.2, -0.3);
        assert!((negated - expected_negated).norm() < TOLERANCE);

        // Test subtract
        let diff = tangent1.subtract(&tangent2_vec);
        let expected_diff = Vector6::new(0.5, 1.5, 2.5, 0.05, 0.15, 0.25);
        assert!((diff - expected_diff).norm() < TOLERANCE);
    }

    #[test]
    fn test_se3_tangent_norms_and_inner_products() {
        let tangent = SE3Tangent::new(Vector3::new(3.0, 4.0, 0.0), Vector3::new(0.0, 0.0, 0.0));

        // Test norm (should be 5.0 for [3, 4, 0, 0, 0, 0])
        let norm = tangent.norm();
        assert!((norm - 5.0).abs() < TOLERANCE);

        // Test squared norm
        let squared_norm = tangent.squared_norm();
        assert!((squared_norm - 25.0).abs() < TOLERANCE);

        // Test inner product
        let other_vec = Vector6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let inner = tangent.inner(&other_vec);
        assert!((inner - 3.0).abs() < TOLERANCE);

        // Test weighted norm with identity weight
        let weight = Matrix6::identity();
        let weighted_norm = tangent.weighted_norm(&weight);
        assert!((weighted_norm - 5.0).abs() < TOLERANCE);

        // Test weighted inner product
        let weighted_inner = tangent.weighted_inner(&other_vec, &weight);
        assert!((weighted_inner - 3.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se3_tangent_hat_vee_operations() {
        let tangent = SE3Tangent::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(0.1, 0.2, 0.3));

        // Test hat operator
        let hat_matrix = tangent.hat();
        assert_eq!(hat_matrix.nrows(), 4);
        assert_eq!(hat_matrix.ncols(), 4);

        // Check translational part (top-right column)
        assert!((hat_matrix[(0, 3)] - 1.0).abs() < TOLERANCE);
        assert!((hat_matrix[(1, 3)] - 2.0).abs() < TOLERANCE);
        assert!((hat_matrix[(2, 3)] - 3.0).abs() < TOLERANCE);

        // Check that bottom row is zero
        assert!(hat_matrix[(3, 0)].abs() < TOLERANCE);
        assert!(hat_matrix[(3, 1)].abs() < TOLERANCE);
        assert!(hat_matrix[(3, 2)].abs() < TOLERANCE);
        assert!(hat_matrix[(3, 3)].abs() < TOLERANCE);

        // Test vee operator (inverse of hat)
        let recovered_vec = SE3Tangent::vee(&hat_matrix);
        assert!((recovered_vec - tangent.data).norm() < TOLERANCE);
    }

    #[test]
    fn test_se3_tangent_exp_jacobians() {
        let tangent = SE3Tangent::new(Vector3::new(0.1, 0.0, 0.0), Vector3::new(0.0, 0.1, 0.0));

        // Test exponential map
        let se3_element = tangent.exp(None);
        assert!(se3_element.is_valid(TOLERANCE));

        // Test basic exp functionality - that we can convert tangent to SE3
        let another_tangent = SE3Tangent::new(
            Vector3::new(0.01, 0.02, 0.03),
            Vector3::new(0.001, 0.002, 0.003),
        );
        let another_se3 = another_tangent.exp(None);
        assert!(another_se3.is_valid(TOLERANCE));

        // Test that Jacobians can be computed without panicking
        let _right_jac = tangent.right_jacobian();
        let _left_jac = tangent.left_jacobian();
        let _right_jac_inv = tangent.right_jacobian_inv();
        let _left_jac_inv = tangent.left_jacobian_inv();

        // Test that Jacobians have correct dimensions
        assert_eq!(_right_jac.nrows(), 6);
        assert_eq!(_right_jac.ncols(), 6);
        assert_eq!(_left_jac.nrows(), 6);
        assert_eq!(_left_jac.ncols(), 6);
        assert_eq!(_right_jac_inv.nrows(), 6);
        assert_eq!(_right_jac_inv.ncols(), 6);
        assert_eq!(_left_jac_inv.nrows(), 6);
        assert_eq!(_left_jac_inv.ncols(), 6);
    }

    #[test]
    fn test_se3_tangent_small_adjoint() {
        let tangent = SE3Tangent::new(Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0));

        let small_adj = tangent.small_adjoint();
        assert_eq!(small_adj.nrows(), 6);
        assert_eq!(small_adj.ncols(), 6);

        // Test anti-symmetry property: ad(x) = -ad(x)^T for the skew-symmetric parts
        let theta_part = small_adj.fixed_view::<3, 3>(0, 0);
        let theta_part_transpose = theta_part.transpose();
        let antisym_check = (theta_part + theta_part_transpose).norm();
        assert!(antisym_check < TOLERANCE);
    }

    #[test]
    fn test_se3_tangent_utility_functions() {
        // Test zero
        let zero_vec = SE3Tangent::zero();
        assert!(zero_vec.norm() < TOLERANCE);

        // Test random
        let random_vec = SE3Tangent::random();
        assert!(random_vec.norm() > 0.0);

        // Test is_zero
        let tangent = SE3Tangent::new(Vector3::zeros(), Vector3::zeros());
        assert!(tangent.is_zero(1e-10));

        let non_zero_tangent = SE3Tangent::new(Vector3::new(1e-5, 0.0, 0.0), Vector3::zeros());
        assert!(!non_zero_tangent.is_zero(1e-10));

        // Test normalize and normalized
        let mut tangent_to_normalize =
            SE3Tangent::new(Vector3::new(3.0, 4.0, 0.0), Vector3::zeros());
        let original_norm = tangent_to_normalize.norm();

        tangent_to_normalize.normalize();
        assert!((tangent_to_normalize.norm() - 1.0).abs() < TOLERANCE);

        // Test normalized (without modifying original)
        let tangent_orig = SE3Tangent::new(Vector3::new(3.0, 4.0, 0.0), Vector3::zeros());
        let normalized_copy = tangent_orig.normalized();
        assert!((normalized_copy.norm() - 1.0).abs() < TOLERANCE);
        assert!((tangent_orig.norm() - original_norm).abs() < TOLERANCE); // Original unchanged
    }

    #[test]
    fn test_se3_tangent_zero_angle_cases() {
        // Test behavior with very small rotational components (should handle gracefully)
        let small_rotation_tangent = SE3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(1e-15, 1e-15, 1e-15),
        );

        // These should not panic and should return reasonable values
        let left_jac = small_rotation_tangent.left_jacobian();
        let right_jac = small_rotation_tangent.right_jacobian();
        let _left_jac_inv = small_rotation_tangent.left_jacobian_inv();
        let _right_jac_inv = small_rotation_tangent.right_jacobian_inv();

        // For very small angles, Jacobians should be approximately identity
        let identity_diff_left = (left_jac - Matrix6::identity()).norm();
        let identity_diff_right = (right_jac - Matrix6::identity()).norm();

        assert!(identity_diff_left < 1e-10);
        assert!(identity_diff_right < 1e-10);
    }

    #[test]
    fn test_se3_tangent_consistency_with_manif() {
        // Test consistency with manif library expectations
        let tangent = SE3Tangent::new(Vector3::new(0.1, 0.2, 0.3), Vector3::new(0.01, 0.02, 0.03));

        // Test that hat(vee(M)) = M for SE(3) matrices
        let hat_result = tangent.hat();
        let vee_result = SE3Tangent::vee(&hat_result);
        let hat_again = SE3Tangent { data: vee_result }.hat();

        // Check that we get back the same matrix
        let matrix_diff = (hat_result - hat_again).norm();
        assert!(matrix_diff < TOLERANCE);

        // Test exp-log consistency
        let se3_element = tangent.exp(None);
        let log_tangent = se3_element.log(None);
        let exp_again = SE3::exp(&log_tangent, None);

        let translation_diff = (se3_element.translation() - exp_again.translation()).norm();
        let rotation_diff = (se3_element.rotation_quaternion().angle()
            - exp_again.rotation_quaternion().angle())
        .abs();

        assert!(translation_diff < TOLERANCE);
        assert!(rotation_diff < TOLERANCE);
    }
}
