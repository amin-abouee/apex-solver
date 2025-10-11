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
//! conventions and provides all operations required by the LieGroup and Tangent traits.

use crate::manifold::so3::{SO3, SO3Tangent};
use crate::manifold::{LieGroup, Tangent, quaternion::Quaternion};
use faer::{Col, Mat, col};
use std::fmt;

/// SE(3) group element representing rigid body transformations in 3D.
///
/// Represented as a combination of SO(3) rotation and Vector3 translation.
#[derive(Clone, Debug, PartialEq)]
pub struct SE3 {
    /// Rotation part as SO(3) element
    rotation: SO3,
    /// Translation part as Vector3
    translation: Col<f64>,
}

impl fmt::Display for SE3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = self.translation();
        let q = self.rotation_quaternion();
        write!(
            f,
            "SE3(translation: [{:.4}, {:.4}, {:.4}], rotation: [w: {:.4}, x: {:.4}, y: {:.4}, z: {:.4}])",
            t[0],
            t[1],
            t[2],
            q.w(),
            q.x(),
            q.y(),
            q.z()
        )
    }
}

impl SE3 {
    /// Space dimension - dimension of the ambient space that the group acts on
    pub const DIM: usize = 3;

    /// Degrees of freedom - dimension of the tangent space
    pub const DOF: usize = 6;

    /// Representation size - size of the underlying data representation
    pub const REP_SIZE: usize = 7;

    /// Get the identity element of the group.
    ///
    /// Returns the neutral element e such that e ∘ g = g ∘ e = g for any group element g.
    pub fn identity() -> Self {
        SE3 {
            rotation: SO3::identity(),
            translation: Col::<f64>::zeros(3),
        }
    }

    /// Get the identity matrix for Jacobians.
    ///
    /// Returns the identity matrix in the appropriate dimension for Jacobian computations.
    pub fn jacobian_identity() -> Mat<f64> {
        Mat::<f64>::identity(6, 6)
    }

    /// Create a new SE3 element from translation and rotation.
    ///
    /// # Arguments
    /// * `translation` - Translation vector [x, y, z]
    /// * `rotation` - Unit quaternion representing rotation
    pub fn new(translation: Col<f64>, rotation: Quaternion) -> Self {
        SE3 {
            rotation: SO3::new(rotation),
            translation,
        }
    }

    /// Create SE3 from translation components and Euler angles.
    pub fn from_translation_quaternion(translation: Col<f64>, quaternion: Quaternion) -> Self {
        let mut normalized_quat = quaternion;
        normalized_quat.normalize();
        Self::new(translation, normalized_quat)
    }

    /// Create SE3 from translation components and Euler angles.
    pub fn from_translation_euler(x: f64, y: f64, z: f64, roll: f64, pitch: f64, yaw: f64) -> Self {
        let translation = col![x, y, z];
        let rotation = Quaternion::from_euler_angles(roll, pitch, yaw);
        Self::new(translation, rotation)
    }

    /// Create SE3 from SO3 and Vector3 components.
    pub fn from_translation_so3(translation: Col<f64>, rotation: SO3) -> Self {
        SE3 {
            rotation,
            translation,
        }
    }

    /// Get the translation part as a Vector3.
    pub fn translation(&self) -> Col<f64> {
        self.translation.clone()
    }

    /// Get the rotation part as SO3.
    pub fn rotation_so3(&self) -> SO3 {
        self.rotation.clone()
    }

    /// Get the rotation part as a UnitQuaternion.
    pub fn rotation_quaternion(&self) -> Quaternion {
        self.rotation.quaternion()
    }

    /// Get the transformation matrix (4x4 homogeneous matrix).
    pub fn matrix(&self) -> Mat<f64> {
        let mut matrix = Mat::<f64>::identity(4, 4);
        let rot_matrix = self.rotation.rotation_matrix();

        // Copy rotation part (top-left 3x3) using submatrix view
        matrix
            .as_mut()
            .submatrix_mut(0, 0, 3, 3)
            .copy_from(&rot_matrix);

        // Copy translation part (top-right 3x1)
        for i in 0..3 {
            matrix[(i, 3)] = self.translation[i];
        }

        matrix
    }

    /// Get the x component of translation.
    pub fn x(&self) -> f64 {
        self.translation[0]
    }

    /// Get the y component of translation.
    pub fn y(&self) -> f64 {
        self.translation[1]
    }

    /// Get the z component of translation.
    pub fn z(&self) -> f64 {
        self.translation[2]
    }

    /// Get coefficients as array [tx, ty, tz, qw, qx, qy, qz].
    pub fn coeffs(&self) -> [f64; 7] {
        let q = self.rotation.quaternion();
        [
            self.translation[0],
            self.translation[1],
            self.translation[2],
            q.w(),
            q.x(),
            q.y(),
            q.z(),
        ]
    }
}

// Conversion traits for integration with generic Problem
impl From<nalgebra::DVector<f64>> for SE3 {
    fn from(data: nalgebra::DVector<f64>) -> Self {
        if data.len() != 7 {
            panic!("SE3::from expects 7-dimensional vector [tx, ty, tz, qw, qx, qy, qz]");
        }
        let translation = col![data[0], data[1], data[2]];
        let quaternion = Quaternion::new(data[3], data[4], data[5], data[6]).unwrap();
        SE3::from_translation_quaternion(translation, quaternion)
    }
}

impl From<SE3> for nalgebra::DVector<f64> {
    fn from(se3: SE3) -> Self {
        let q = se3.rotation.quaternion();
        nalgebra::DVector::from_vec(vec![
            se3.translation[0], // tx first
            se3.translation[1], // ty second
            se3.translation[2], // tz third
            q.w(),              // qw fourth
            q.x(),              // qx fifth
            q.y(),              // qy sixth
            q.z(),              // qz seventh
        ])
    }
}

// Conversion traits using faer Col
impl From<Col<f64>> for SE3 {
    fn from(data: Col<f64>) -> Self {
        if data.nrows() != 7 {
            panic!("SE3::from expects 7-dimensional vector [tx, ty, tz, qw, qx, qy, qz]");
        }
        let translation = col![data[0], data[1], data[2]];
        let quaternion = Quaternion::new(data[3], data[4], data[5], data[6]).unwrap();
        SE3::from_translation_quaternion(translation, quaternion)
    }
}

impl From<&Col<f64>> for SE3 {
    fn from(data: &Col<f64>) -> Self {
        if data.nrows() != 7 {
            panic!("SE3::from expects 7-dimensional vector [tx, ty, tz, qw, qx, qy, qz]");
        }
        let translation = col![data[0], data[1], data[2]];
        let quaternion = Quaternion::new(data[3], data[4], data[5], data[6]).unwrap();
        SE3::from_translation_quaternion(translation, quaternion)
    }
}

impl From<SE3> for Col<f64> {
    fn from(se3: SE3) -> Self {
        let q = se3.rotation.quaternion();
        col![
            se3.translation[0], // tx
            se3.translation[1], // ty
            se3.translation[2], // tz
            q.w(),              // qw
            q.x(),              // qx
            q.y(),              // qy
            q.z()               // qz
        ]
    }
}

impl From<&SE3> for Col<f64> {
    fn from(se3: &SE3) -> Self {
        let q = se3.rotation.quaternion();
        col![
            se3.translation[0], // tx
            se3.translation[1], // ty
            se3.translation[2], // tz
            q.w(),              // qw
            q.x(),              // qx
            q.y(),              // qy
            q.z()               // qz
        ]
    }
}

// Implement basic trait requirements for LieGroup
impl LieGroup for SE3 {
    type TangentVector = SE3Tangent;
    type JacobianMatrix = Mat<f64>;
    type LieAlgebra = Mat<f64>;

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
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self {
        // For SE(3): g^{-1} = [R^T, -R^T * t; 0, 1]
        let rot_inv = self.rotation.inverse(None);
        let trans_inv = -rot_inv.act(&self.translation, None, None);

        // Eqs. 176
        if let Some(jac) = jacobian {
            *jac = -self.adjoint();
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
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        // Eqs. 171
        let composed_rotation = self.rotation.compose(&other.rotation, None, None);
        let composed_translation =
            &self.rotation.act(&other.translation, None, None) + &self.translation;

        let result = SE3::from_translation_so3(composed_translation, composed_rotation);

        if let Some(jac_self) = jacobian_self {
            // Jacobian wrt first element: Ad(g2^{-1})
            *jac_self = other.inverse(None).adjoint();
        }

        if let Some(jac_other) = jacobian_other {
            // Jacobian wrt second element: I (identity)
            *jac_other = Mat::<f64>::identity(6, 6);
        }

        result
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
    ///
    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector {
        // Log of rotation (axis-angle representation)
        let theta = self.rotation.log(None);
        let mut data = Col::<f64>::zeros(6);
        let translation_vector = &theta.left_jacobian_inv() * &self.translation;

        // Copy translation part
        data[0] = translation_vector[0];
        data[1] = translation_vector[1];
        data[2] = translation_vector[2];

        // Copy rotation part
        let theta_coeffs = theta.coeffs();
        data[3] = theta_coeffs[0];
        data[4] = theta_coeffs[1];
        data[5] = theta_coeffs[2];

        let result = SE3Tangent { data };
        if let Some(jac) = jacobian {
            *jac = result.right_jacobian_inv();
        }

        result
    }

    fn act(
        &self,
        vector: &Col<f64>,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_vector: Option<&mut Mat<f64>>,
    ) -> Col<f64> {
        // Apply SE(3) transformation: R * v + t
        let result = self.rotation.act(vector, None, None) + &self.translation;

        if let Some(jac_self) = jacobian_self {
            let rot_matrix = self.rotation.rotation_matrix();

            // Copy rotation matrix to top-left 3x3 block
            jac_self
                .as_mut()
                .submatrix_mut(0, 0, 3, 3)
                .copy_from(&rot_matrix);

            // Copy -R * [v]_x to top-right 3x3 block
            let top_right = -rot_matrix * SO3Tangent::new(vector.clone()).hat();
            jac_self
                .as_mut()
                .submatrix_mut(0, 3, 3, 3)
                .copy_from(&top_right);
        }

        if let Some(jac_vector) = jacobian_vector {
            // Jacobian wrt vector
            let rotation_matrix = self.rotation.rotation_matrix();
            jac_vector
                .as_mut()
                .submatrix_mut(0, 0, 3, 3)
                .copy_from(&rotation_matrix);
        }

        result
    }

    fn adjoint(&self) -> Self::JacobianMatrix {
        // Adjoint matrix for SE(3)
        let rotation_matrix = self.rotation.rotation_matrix();
        let translation = self.translation.clone();
        let mut adjoint_matrix = Mat::<f64>::zeros(6, 6);

        // Top-left block: R
        adjoint_matrix
            .as_mut()
            .submatrix_mut(0, 0, 3, 3)
            .copy_from(&rotation_matrix);

        // Bottom-right block: R
        adjoint_matrix
            .as_mut()
            .submatrix_mut(3, 3, 3, 3)
            .copy_from(&rotation_matrix);

        // Top-right block: [t]_× R (skew-symmetric of translation times rotation)
        let top_right = SO3Tangent::new(translation).hat() * rotation_matrix;
        adjoint_matrix
            .as_mut()
            .submatrix_mut(0, 3, 3, 3)
            .copy_from(&top_right);

        adjoint_matrix
    }

    fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::rng();

        // Random translation in [-1, 1]³
        let translation = col![
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        ];

        // Random rotation
        let rotation = SO3::random();

        SE3::from_translation_so3(translation, rotation)
    }

    fn normalize(&mut self) {
        // Normalize the rotation part
        self.rotation.normalize();
    }

    fn is_valid(&self, tolerance: f64) -> bool {
        // Check if rotation is valid
        self.rotation.is_valid(tolerance)
    }

    /// Vee operator: log(g)^∨.
    ///
    /// Maps a group element g ∈ G to its tangent vector log(g)^∨ ∈ 𝔤.
    /// For SE(3), this is the same as log().
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

/// SE(3) tangent space element representing elements in the Lie algebra se(3).
///
/// Following manif conventions, internally represented as [rho(3), theta(3)] where:
/// - rho: translational component [rho_x, rho_y, rho_z]
/// - theta: rotational component [theta_x, theta_y, theta_z]
#[derive(Clone, Debug, PartialEq)]
pub struct SE3Tangent {
    /// Internal data: [rho_x, rho_y, rho_z, theta_x, theta_y, theta_z]
    data: Col<f64>,
}

impl fmt::Display for SE3Tangent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rho = self.rho();
        let theta = self.theta();
        write!(
            f,
            "se3(rho: [{:.4}, {:.4}, {:.4}], theta: [{:.4}, {:.4}, {:.4}])",
            rho[0], rho[1], rho[2], theta[0], theta[1], theta[2]
        )
    }
}

impl From<nalgebra::DVector<f64>> for SE3Tangent {
    fn from(data_vector: nalgebra::DVector<f64>) -> Self {
        if data_vector.len() != 6 {
            panic!(
                "SE3Tangent::from expects 6-dimensional vector [rho_x, rho_y, rho_z, theta_x, theta_y, theta_z]"
            );
        }
        SE3Tangent {
            data: col![
                data_vector[0],
                data_vector[1],
                data_vector[2],
                data_vector[3],
                data_vector[4],
                data_vector[5],
            ],
        }
    }
}

// Conversion traits using faer Col
impl From<Col<f64>> for SE3Tangent {
    fn from(data: Col<f64>) -> Self {
        if data.nrows() != 6 {
            panic!(
                "SE3Tangent::from expects 6-dimensional vector [rho_x, rho_y, rho_z, theta_x, theta_y, theta_z]"
            );
        }
        SE3Tangent { data }
    }
}

impl From<SE3Tangent> for Col<f64> {
    fn from(tangent: SE3Tangent) -> Self {
        tangent.data
    }
}

impl From<&SE3Tangent> for Col<f64> {
    fn from(tangent: &SE3Tangent) -> Self {
        tangent.data.clone()
    }
}

// impl From<SE3Tangent> for nalgebra::DVector<f64> {
//     fn from(se3_tangent: SE3Tangent) -> Self {
//         DVector::from_vec(vec![
//             se3_tangent.data[0],
//             se3_tangent.data[1],
//             se3_tangent.data[2],
//             se3_tangent.data[3],
//             se3_tangent.data[4],
//             se3_tangent.data[5],
//         ])
//     }
// }

impl SE3Tangent {
    /// Create a new SE3Tangent from rho (translational) and theta (rotational) components.
    ///
    /// # Arguments
    /// * `rho` - Translational component [rho_x, rho_y, rho_z]
    /// * `theta` - Rotational component [theta_x, theta_y, theta_z]
    pub fn new(rho: Col<f64>, theta: Col<f64>) -> Self {
        let mut data = Col::<f64>::zeros(6);
        data[0] = rho[0];
        data[1] = rho[1];
        data[2] = rho[2];
        data[3] = theta[0];
        data[4] = theta[1];
        data[5] = theta[2];
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
            data: col![rho_x, rho_y, rho_z, theta_x, theta_y, theta_z],
        }
    }

    /// Get the rho (translational) part.
    pub fn rho(&self) -> Col<f64> {
        col![self.data[0], self.data[1], self.data[2]]
    }

    /// Get the theta (rotational) part.
    pub fn theta(&self) -> Col<f64> {
        col![self.data[3], self.data[4], self.data[5]]
    }

    /// Equation 180: Q(ρ, θ) function for SE(3) Jacobians
    /// Q(ρ, θ) = (1/2)ρₓ + (θ - sin θ)/θ³ (θₓρₓ + ρₓθₓ + θₓρₓθₓ)
    ///           - (1 - θ²/2 - cos θ)/θ⁴ (θ²ₓρₓ + ρₓθ²ₓ - 3θₓρₓθₓ)
    ///           - (1/2) * ( (1 - θ²/2 - cos θ)/θ⁴ - (3.0 * (θ - sin θ - θ³/6))/θ⁵ ) * (θₓρₓθ²ₓ + θ²ₓρₓθₓ)
    pub fn q_block_jacobian_matrix(rho: Col<f64>, theta: Col<f64>) -> Mat<f64> {
        let theta_squared = theta.squared_norm_l2();
        let rho_skew = SO3Tangent::new(rho).hat();
        let theta_skew = SO3Tangent::new(theta).hat();

        let a = 0.5;
        let mut b = 1.0 / 6.0 + 1.0 / 120.0 * theta_squared;
        let mut c = -1.0 / 24.0 + 1.0 / 720.0 * theta_squared;
        let mut d = -1.0 / 60.0;

        if theta_squared > f64::EPSILON {
            // --- Large Angle Path: Direct computation ---
            let theta_norm = theta_squared.sqrt();
            let theta_norm_3 = theta_norm * theta_squared;
            let theta_norm_4 = theta_squared * theta_squared;
            let theta_norm_5 = theta_norm_3 * theta_squared;
            let sin_theta = theta_norm.sin();
            let cos_theta = theta_norm.cos();

            // Calculate coefficients directly from the formula
            b = (theta_norm - sin_theta) / theta_norm_3;
            c = (1.0 - theta_squared / 2.0 - cos_theta) / theta_norm_4;
            d = (c - 3.0) * (theta_norm - sin_theta - theta_norm_3 / 6.0) / theta_norm_5;
        }

        let rho_skew_theta_skew = &rho_skew * &theta_skew;
        let theta_skew_rho_skew = &theta_skew * &rho_skew;
        let theta_skew_rho_skew_theta_skew = &theta_skew * &rho_skew * &theta_skew;
        let rho_skew_theta_skew_sq2 = &rho_skew * &theta_skew * &theta_skew;

        // Calculate matrix terms
        let m1 = &rho_skew;
        let m2 = &theta_skew_rho_skew + &rho_skew_theta_skew + &theta_skew_rho_skew_theta_skew;
        let m3 = &rho_skew_theta_skew_sq2
            - &rho_skew_theta_skew_sq2.transpose().to_owned()
            - 3.0 * &theta_skew_rho_skew_theta_skew;
        let m4 = &theta_skew_rho_skew_theta_skew * &theta_skew;

        // Assemble the final matrix
        m1 * a + m2 * b - m3 * c - m4 * d
    }
}

// Implement LieAlgebra trait for SE3Tangent
impl Tangent<SE3> for SE3Tangent {
    /// Dimension of the tangent space
    const DIM: usize = 6;

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
    fn exp(&self, jacobian: Option<&mut <SE3 as LieGroup>::JacobianMatrix>) -> SE3 {
        let rho = self.rho();
        let theta = self.theta();

        let theta_tangent = SO3Tangent::new(theta);
        // Compute rotation part using SO(3) exponential
        let rotation = theta_tangent.exp(None);
        let translation = theta_tangent.left_jacobian() * rho;

        if let Some(jac) = jacobian {
            *jac = self.right_jacobian();
        }

        SE3::from_translation_so3(translation, rotation)
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
    fn right_jacobian(&self) -> <SE3 as LieGroup>::JacobianMatrix {
        let mut jac = Mat::<f64>::zeros(6, 6);
        let rho = self.rho();
        let theta = self.theta();
        let theta_right_jac = SO3Tangent::new(-theta.clone()).right_jacobian();

        // Top-left 3x3
        jac.as_mut()
            .submatrix_mut(0, 0, 3, 3)
            .copy_from(&theta_right_jac);

        // Bottom-right 3x3
        jac.as_mut()
            .submatrix_mut(3, 3, 3, 3)
            .copy_from(&theta_right_jac);

        // Top-right 3x3
        let q_block = SE3Tangent::q_block_jacobian_matrix(-rho, -theta);
        jac.as_mut().submatrix_mut(0, 3, 3, 3).copy_from(&q_block);

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
    fn left_jacobian(&self) -> <SE3 as LieGroup>::JacobianMatrix {
        let mut jac = Mat::<f64>::zeros(6, 6);
        let theta_left_jac = SO3Tangent::new(self.theta()).left_jacobian();

        // Top-left 3x3
        jac.as_mut()
            .submatrix_mut(0, 0, 3, 3)
            .copy_from(&theta_left_jac);

        // Bottom-right 3x3
        jac.as_mut()
            .submatrix_mut(3, 3, 3, 3)
            .copy_from(&theta_left_jac);

        // Top-right 3x3
        let q_block = SE3Tangent::q_block_jacobian_matrix(self.rho(), self.theta());
        jac.as_mut().submatrix_mut(0, 3, 3, 3).copy_from(&q_block);

        jac
    }

    /// Inverse of right Jacobian Jr⁻¹.
    ///
    /// Computes the inverse of the right Jacobian. This is used for
    /// computing perturbations and derivatives.
    ///
    /// # Returns
    /// The inverse right Jacobian matrix (6x6)
    fn right_jacobian_inv(&self) -> <SE3 as LieGroup>::JacobianMatrix {
        let mut jac = Mat::<f64>::zeros(6, 6);
        let rho = self.rho();
        let theta = self.theta();
        let theta_left_inv_jac = SO3Tangent::new(-theta.clone()).left_jacobian_inv();
        let q_block_jac = SE3Tangent::q_block_jacobian_matrix(-rho, -theta);

        // Top-left 3x3
        jac.as_mut()
            .submatrix_mut(0, 0, 3, 3)
            .copy_from(&theta_left_inv_jac);

        // Bottom-right 3x3
        jac.as_mut()
            .submatrix_mut(3, 3, 3, 3)
            .copy_from(&theta_left_inv_jac);

        // Top-right 3x3
        let bottom_right = -1.0 * &theta_left_inv_jac * &q_block_jac * &theta_left_inv_jac;
        jac.as_mut()
            .submatrix_mut(0, 3, 3, 3)
            .copy_from(&bottom_right);

        jac
    }

    /// Inverse of left Jacobian Jl⁻¹.
    ///
    /// Computes the inverse of the left Jacobian following manif conventions.
    ///
    /// # Returns
    /// The inverse left Jacobian matrix (6x6)
    fn left_jacobian_inv(&self) -> <SE3 as LieGroup>::JacobianMatrix {
        let mut jac = Mat::<f64>::zeros(6, 6);
        let rho = self.rho();
        let theta = self.theta();
        let theta_left_inv_jac = SO3Tangent::new(theta.clone()).left_jacobian_inv();
        let q_block_jac = SE3Tangent::q_block_jacobian_matrix(rho, theta);
        let top_right_block = -1.0 * &theta_left_inv_jac * &q_block_jac * &theta_left_inv_jac;

        // Top-left 3x3
        jac.as_mut()
            .submatrix_mut(0, 0, 3, 3)
            .copy_from(&theta_left_inv_jac);

        // Bottom-right 3x3
        jac.as_mut()
            .submatrix_mut(3, 3, 3, 3)
            .copy_from(&theta_left_inv_jac);

        // Top-right 3x3
        jac.as_mut()
            .submatrix_mut(0, 3, 3, 3)
            .copy_from(&top_right_block);

        jac
    }

    // Matrix representations

    /// Hat operator: φ^∧ (vector to matrix).
    ///
    /// Converts the SE(3) tangent vector to its 4x4 matrix representation in the Lie algebra.
    /// Following manif conventions, the structure is:
    /// [  theta_×   rho  ]
    /// [      0       0    ]
    /// where theta_× is the skew-symmetric matrix of the rotational part.
    ///
    /// # Returns
    /// The 4x4 matrix representation in the SE(3) Lie algebra
    fn hat(&self) -> <SE3 as LieGroup>::LieAlgebra {
        let mut lie_alg = Mat::<f64>::zeros(4, 4);

        // Top-left 3x3: skew-symmetric matrix of rotational part
        let theta_hat = SO3Tangent::new(self.theta()).hat();
        lie_alg
            .as_mut()
            .submatrix_mut(0, 0, 3, 3)
            .copy_from(&theta_hat);

        // Top-right 3x1: translational part
        let rho = self.rho();
        lie_alg[(0, 3)] = rho[0];
        lie_alg[(1, 3)] = rho[1];
        lie_alg[(2, 3)] = rho[2];

        lie_alg
    }

    // Utility functions

    /// Zero tangent vector.
    ///
    /// Returns the zero element of the SE(3) tangent space.
    ///
    /// # Returns
    /// A 6-dimensional zero vector
    fn zero() -> <SE3 as LieGroup>::TangentVector {
        SE3Tangent::new(Col::<f64>::zeros(3), Col::<f64>::zeros(3))
    }

    /// Random tangent vector (useful for testing).
    ///
    /// Generates a random tangent vector with reasonable bounds.
    /// Translation components are in [-1, 1] and rotation components in [-0.1, 0.1].
    ///
    /// # Returns
    /// A random 6-dimensional tangent vector
    fn random() -> <SE3 as LieGroup>::TangentVector {
        use rand::Rng;
        let mut rng = rand::rng();
        SE3Tangent::from_components(
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
        self.data.norm_l2() < tolerance
    }

    /// Normalize the tangent vector to unit norm.
    ///
    /// Modifies this tangent vector to have unit norm. If the vector
    /// is near zero, it remains unchanged.
    fn normalize(&mut self) {
        let theta_norm = self.theta().norm_l2();
        self.data[3] /= theta_norm;
        self.data[4] /= theta_norm;
        self.data[5] /= theta_norm;
    }

    /// Return a unit tangent vector in the same direction.
    ///
    /// Returns a new tangent vector with unit norm in the same direction.
    /// If the original vector is near zero, returns zero.
    ///
    /// # Returns
    /// A normalized copy of the tangent vector
    fn normalized(&self) -> <SE3 as LieGroup>::TangentVector {
        let norm = self.theta().norm_l2();
        if norm > f64::EPSILON {
            SE3Tangent::new(self.rho(), self.theta() / norm)
        } else {
            SE3Tangent::new(self.rho(), Col::<f64>::zeros(3))
        }
    }

    /// Small adjoint matrix for SE(3).
    ///
    /// For SE(3), the small adjoint matrix has the structure:
    /// [ Omega  V   ]
    /// [   0  Omega ]
    /// where Omega is the skew-symmetric matrix of the angular part
    /// and V is the skew-symmetric matrix of the linear part.
    fn small_adj(&self) -> <SE3 as LieGroup>::JacobianMatrix {
        let mut small_adj = Mat::<f64>::zeros(6, 6);
        let rho_skew = SO3Tangent::new(self.rho()).hat();
        let theta_skew = SO3Tangent::new(self.theta()).hat();

        // Top-left block: theta_skew
        small_adj
            .as_mut()
            .submatrix_mut(0, 0, 3, 3)
            .copy_from(&theta_skew);

        // Bottom-right block: theta_skew
        small_adj
            .as_mut()
            .submatrix_mut(3, 3, 3, 3)
            .copy_from(&theta_skew);

        // Top-right block: rho_skew (skew-symmetric of linear part)
        small_adj
            .as_mut()
            .submatrix_mut(0, 3, 3, 3)
            .copy_from(&rho_skew);

        // Bottom-left block: zeros (already set)

        small_adj
    }

    /// Lie bracket for SE(3).
    ///
    /// Computes the Lie bracket [this, other] = this.small_adj() * other.
    fn lie_bracket(&self, other: &Self) -> <SE3 as LieGroup>::TangentVector {
        let bracket_result = &self.small_adj() * &other.data;
        SE3Tangent {
            data: bracket_result,
        }
    }

    /// Check if this tangent vector is approximately equal to another.
    ///
    /// # Arguments
    /// * `other` - The other tangent vector to compare with
    /// * `tolerance` - The tolerance for the comparison
    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        (&self.data - &other.data).norm_l2() < tolerance
    }

    /// Get the ith generator of the SE(3) Lie algebra.
    ///
    /// # Arguments
    /// * `i` - Index of the generator (0-5 for SE(3))
    ///
    /// # Returns
    /// The generator matrix
    fn generator(&self, i: usize) -> <SE3 as LieGroup>::LieAlgebra {
        assert!(i < 6, "SE(3) only has generators for indices 0-5");

        let mut generator = Mat::<f64>::zeros(4, 4);

        match i {
            0 => {
                // Generator for rho_x (translation in x)
                generator[(0, 3)] = 1.0;
            }
            1 => {
                // Generator for rho_y (translation in y)
                generator[(1, 3)] = 1.0;
            }
            2 => {
                // Generator for rho_z (translation in z)
                generator[(2, 3)] = 1.0;
            }
            3 => {
                // Generator for theta_x (rotation around x-axis)
                generator[(1, 2)] = -1.0;
                generator[(2, 1)] = 1.0;
            }
            4 => {
                // Generator for theta_y (rotation around y-axis)
                generator[(0, 2)] = 1.0;
                generator[(2, 0)] = -1.0;
            }
            5 => {
                // Generator for theta_z (rotation around z-axis)
                generator[(0, 1)] = -1.0;
                generator[(1, 0)] = 1.0;
            }
            _ => unreachable!(),
        }

        generator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOLERANCE: f64 = 1e-9;

    #[test]
    fn test_se3_tangent_basic() {
        let linear = col![1.0, 2.0, 3.0];
        let angular = col![0.1, 0.2, 0.3];
        let tangent = SE3Tangent::new(linear.clone(), angular.clone());

        assert_eq!(tangent.rho(), linear);
        assert_eq!(tangent.theta(), angular);
    }

    #[test]
    fn test_se3_tangent_zero() {
        let zero = SE3Tangent::zero();
        let zero_vec = Col::<f64>::zeros(6);
        assert_eq!(zero.data, zero_vec);

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

        assert!(translation.norm_l2() < TOLERANCE);
        assert!((rotation.angle()) < TOLERANCE);
    }

    #[test]
    fn test_se3_new() {
        let translation = col![1.0, 2.0, 3.0];
        let rotation = Quaternion::from_euler_angles(0.1, 0.2, 0.3);
        let rotation_angle = rotation.angle();

        let se3 = SE3::new(translation.clone(), rotation);

        assert!(se3.is_valid(TOLERANCE));
        assert!((&se3.translation() - &translation).norm_l2() < TOLERANCE);
        assert!((se3.rotation_quaternion().angle() - rotation_angle).abs() < TOLERANCE);
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

        let translation_diff = (&composed.translation() - &identity.translation()).norm_l2();
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

        let translation_diff =
            (&composed_with_identity.translation() - &se3_1.translation()).norm_l2();
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
        let point = col![1.0, 2.0, 3.0];

        let _transformed_point = se3.act(&point, None, None);

        // Test act with identity
        let identity = SE3::identity();
        let identity_transformed = identity.act(&point, None, None);

        let diff = (&identity_transformed - &point).norm_l2();
        assert!(diff < TOLERANCE);
    }

    #[test]
    fn test_se3_between() {
        let se3a = SE3::from_translation_euler(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        let se3b = se3a.clone();
        let se3_between_identity = se3a.between(&se3b, None, None);
        assert!(se3_between_identity.is_approx(&SE3::identity(), TOLERANCE));

        let se3c = SE3::from_translation_euler(4.0, 5.0, 6.0, 0.4, 0.5, 0.6);
        let se3_between = se3a.between(&se3c, None, None);
        let expected = se3a.inverse(None).compose(&se3c, None, None);
        assert!(se3_between.is_approx(&expected, TOLERANCE));
    }

    #[test]
    fn test_se3_exp_log() {
        let tangent_vec = col![0.1, 0.2, 0.3, 0.01, 0.02, 0.03];
        let tangent = SE3Tangent { data: tangent_vec };

        // Test exp(log(g)) = g
        let se3 = tangent.exp(None);
        let recovered_tangent = se3.log(None);

        let diff = (&tangent.data - &recovered_tangent.data).norm_l2();
        assert!(diff < TOLERANCE);
    }

    #[test]
    fn test_se3_exp_zero() {
        let zero_tangent = SE3Tangent::zero();
        let se3 = zero_tangent.exp(None);
        let identity = SE3::identity();

        let translation_diff = (&se3.translation() - &identity.translation()).norm_l2();
        let rotation_diff = se3.rotation_quaternion().angle();

        assert!(translation_diff < TOLERANCE);
        assert!(rotation_diff < TOLERANCE);
    }

    #[test]
    fn test_se3_log_identity() {
        let identity = SE3::identity();
        let tangent = identity.log(None);

        assert!(tangent.data.norm_l2() < TOLERANCE);
    }

    #[test]
    fn test_se3_normalize() {
        let translation = col![1.0, 2.0, 3.0];
        let mut rotation = Quaternion::new(0.5, 0.5, 0.5, 0.5).unwrap();
        rotation.normalize();

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

        let translation_diff = (&left_assoc.translation() - &right_assoc.translation()).norm_l2();
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
        let translation_only = SE3::new(col![1.0, 2.0, 3.0], Quaternion::identity());

        let point = col![0.0, 0.0, 0.0];
        let transformed = translation_only.act(&point, None, None);
        let expected = col![1.0, 2.0, 3.0];

        assert!((&transformed - &expected).norm_l2() < TOLERANCE);

        // Rotation only
        let rotation_only = SE3::new(
            Col::<f64>::zeros(3),
            Quaternion::from_euler_angles(PI / 2.0, 0.0, 0.0),
        );

        let point_y = col![0.0, 1.0, 0.0];
        let rotated = rotation_only.act(&point_y, None, None);
        let expected_rotated = col![0.0, 0.0, 1.0];

        assert!((&rotated - &expected_rotated).norm_l2() < TOLERANCE);
    }

    #[test]
    fn test_se3_small_angle_approximations() {
        // Test behavior with very small angles, similar to manif library tests
        let small_tangent = col![1e-8, 2e-8, 3e-8, 1e-9, 2e-9, 3e-9];

        let se3 = SE3::new(
            col![1e-8, 2e-8, 3e-8],
            Quaternion::from_euler_angles(1e-9, 2e-9, 3e-9),
        );
        let recovered = se3.log(None);

        let diff = (&small_tangent - &recovered.data).norm_l2();
        assert!(diff < TOLERANCE);
    }

    #[test]
    fn test_se3_tangent_norm() {
        let tangent_vec = col![3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let tangent = SE3Tangent { data: tangent_vec };

        let norm = tangent.data.norm_l2();
        assert!((norm - 5.0).abs() < TOLERANCE); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_se3_from_components() {
        let translation = col![1.0, 2.0, 3.0];
        let quaternion = Quaternion::new(1.0, 0.0, 0.0, 0.0).unwrap();
        let se3 = SE3::from_translation_quaternion(translation, quaternion);
        assert!(se3.is_valid(TOLERANCE));
        assert_eq!(se3.x(), 1.0);
        assert_eq!(se3.y(), 2.0);
        assert_eq!(se3.z(), 3.0);

        let quat = se3.rotation_quaternion();
        assert!((quat.w() - 1.0).abs() < TOLERANCE);
        assert!(quat.x().abs() < TOLERANCE);
        assert!(quat.y().abs() < TOLERANCE);
        assert!(quat.z().abs() < TOLERANCE);
    }

    // Note: test_se3_from_isometry removed - Isometry3 is nalgebra-specific
    // and not available in faer. SE3 can be constructed directly from
    // translation and rotation components.

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
            col![1.0, 0.0, 0.0],
            Quaternion::from_euler_angles(0.0, 0.0, PI / 4.0),
        );

        let g2 = SE3::new(
            col![0.0, 1.0, 0.0],
            Quaternion::from_euler_angles(0.0, PI / 4.0, 0.0),
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
        let translation_diff = (&result.translation() - &identity.translation()).norm_l2();
        let rotation_diff = result.rotation_quaternion().angle();

        assert!(translation_diff < TOLERANCE);
        assert!(rotation_diff < TOLERANCE);
    }

    #[test]
    fn test_se3_tangent_exp_jacobians() {
        let tangent = SE3Tangent::new(col![0.1, 0.0, 0.0], col![0.0, 0.1, 0.0]);

        // Test exponential map
        let se3_element = tangent.exp(None);
        assert!(se3_element.is_valid(TOLERANCE));

        // Test basic exp functionality - that we can convert tangent to SE3
        let another_tangent = SE3Tangent::new(col![0.01, 0.02, 0.03], col![0.001, 0.002, 0.003]);
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
    fn test_se3_tangent_utility_functions() {
        // Test zero
        let zero_vec = SE3Tangent::zero();
        assert!(zero_vec.data.norm_l2() < TOLERANCE);

        // Test random
        let random_vec = SE3Tangent::random();
        assert!(random_vec.data.norm_l2() > 0.0);

        // Test is_zero
        let tangent = SE3Tangent::new(Col::<f64>::zeros(3), Col::<f64>::zeros(3));
        assert!(tangent.is_zero(1e-10));

        let non_zero_tangent = SE3Tangent::new(col![1e-5, 0.0, 0.0], Col::<f64>::zeros(3));
        assert!(!non_zero_tangent.is_zero(1e-10));
    }

    // New tests for the additional functions

    #[test]
    fn test_se3_vee() {
        let se3 = SE3::random();
        let tangent_log = se3.log(None);
        let tangent_vee = se3.vee();

        assert!((&tangent_log.data - &tangent_vee.data).norm_l2() < 1e-10);
    }

    #[test]
    fn test_se3_is_approx() {
        let se3_1 = SE3::random();
        let se3_2 = se3_1.clone();

        assert!(se3_1.is_approx(&se3_1, 1e-10));
        assert!(se3_1.is_approx(&se3_2, 1e-10));

        // Test with small perturbation
        let small_tangent = SE3Tangent::new(col![1e-12, 1e-12, 1e-12], col![1e-12, 1e-12, 1e-12]);
        let se3_perturbed = se3_1.right_plus(&small_tangent, None, None);
        assert!(se3_1.is_approx(&se3_perturbed, 1e-10));
    }

    #[test]
    fn test_se3_tangent_small_adj() {
        let tangent = SE3Tangent::new(col![0.1, 0.2, 0.3], col![0.4, 0.5, 0.6]);
        let small_adj = tangent.small_adj();

        // Verify the structure of the small adjoint matrix for SE(3)
        // Should be:
        // [ Omega  V   ]
        // [   0  Omega ]
        let rho_skew = SO3Tangent::new(tangent.rho()).hat();
        let theta_skew = SO3Tangent::new(tangent.theta()).hat();

        // Check top-left block (theta_skew)
        let top_left = small_adj.submatrix(0, 0, 3, 3);
        assert!((&top_left - &theta_skew).norm_l2() < 1e-10);

        // Check bottom-right block (theta_skew)
        let bottom_right = small_adj.submatrix(3, 3, 3, 3);
        assert!((&bottom_right - &theta_skew).norm_l2() < 1e-10);

        // Check top-right block (rho_skew)
        let top_right = small_adj.submatrix(0, 3, 3, 3);
        assert!((&top_right - &rho_skew).norm_l2() < 1e-10);

        // Check bottom-left block (zeros)
        let bottom_left = small_adj.submatrix(3, 0, 3, 3);
        assert!(bottom_left.norm_l2() < 1e-10);
    }

    #[test]
    fn test_se3_tangent_lie_bracket() {
        let tangent_a = SE3Tangent::new(col![0.1, 0.0, 0.0], col![0.0, 0.2, 0.0]);
        let tangent_b = SE3Tangent::new(col![0.0, 0.3, 0.0], col![0.0, 0.0, 0.4]);

        let bracket_ab = tangent_a.lie_bracket(&tangent_b);
        let bracket_ba = tangent_b.lie_bracket(&tangent_a);

        // Anti-symmetry test: [a,b] = -[b,a]
        assert!((&bracket_ab.data + &bracket_ba.data).norm_l2() < 1e-10);

        // [a,a] = 0
        let bracket_aa = tangent_a.lie_bracket(&tangent_a);
        assert!(bracket_aa.is_zero(1e-10));

        // Verify bracket relationship with hat operator
        let bracket_hat = bracket_ab.hat();
        let expected = &tangent_a.hat() * &tangent_b.hat() - &tangent_b.hat() * &tangent_a.hat();
        assert!((&bracket_hat - &expected).norm_l2() < 1e-10);
    }

    #[test]
    fn test_se3_tangent_is_approx() {
        let tangent_1 = SE3Tangent::new(col![0.1, 0.2, 0.3], col![0.4, 0.5, 0.6]);
        let tangent_2 = SE3Tangent::new(col![0.1 + 1e-12, 0.2, 0.3], col![0.4, 0.5, 0.6]);
        let tangent_3 = SE3Tangent::new(col![0.7, 0.8, 0.9], col![1.0, 1.1, 1.2]);

        assert!(tangent_1.is_approx(&tangent_1, 1e-10));
        assert!(tangent_1.is_approx(&tangent_2, 1e-10));
        assert!(!tangent_1.is_approx(&tangent_3, 1e-10));
    }

    #[test]
    fn test_se3_generators() {
        let tangent = SE3Tangent::new(col![1.0, 1.0, 1.0], col![1.0, 1.0, 1.0]);

        // Test all six generators
        for i in 0..6 {
            let generator = tangent.generator(i);

            // Verify that generators are 4x4 matrices
            assert_eq!(generator.nrows(), 4);
            assert_eq!(generator.ncols(), 4);

            // Bottom row should always be zeros for SE(3) generators
            assert_eq!(generator[(3, 0)], 0.0);
            assert_eq!(generator[(3, 1)], 0.0);
            assert_eq!(generator[(3, 2)], 0.0);
            assert_eq!(generator[(3, 3)], 0.0);
        }

        // Test specific values for translation generators
        let e1 = tangent.generator(0); // rho_x
        let e2 = tangent.generator(1); // rho_y
        let e3 = tangent.generator(2); // rho_z

        assert_eq!(e1[(0, 3)], 1.0);
        assert_eq!(e2[(1, 3)], 1.0);
        assert_eq!(e3[(2, 3)], 1.0);

        // Test specific values for rotation generators
        let e4 = tangent.generator(3); // theta_x
        let e5 = tangent.generator(4); // theta_y
        let e6 = tangent.generator(5); // theta_z

        // Rotation generators should be skew-symmetric in top-left 3x3 block
        assert_eq!(e4[(1, 2)], -1.0);
        assert_eq!(e4[(2, 1)], 1.0);
        assert_eq!(e5[(0, 2)], 1.0);
        assert_eq!(e5[(2, 0)], -1.0);
        assert_eq!(e6[(0, 1)], -1.0);
        assert_eq!(e6[(1, 0)], 1.0);
    }

    #[test]
    #[should_panic]
    fn test_se3_generator_invalid_index() {
        let tangent = SE3Tangent::new(col![1.0, 1.0, 1.0], col![1.0, 1.0, 1.0]);
        let _generator = tangent.generator(6); // Should panic for SE(3)
    }

    #[test]
    fn test_se3_jacobi_identity() {
        // Test Jacobi identity: [x,[y,z]]+[y,[z,x]]+[z,[x,y]]=0
        let x = SE3Tangent::new(col![0.1, 0.0, 0.0], col![0.0, 0.1, 0.0]);
        let y = SE3Tangent::new(col![0.0, 0.2, 0.0], col![0.0, 0.0, 0.2]);
        let z = SE3Tangent::new(col![0.0, 0.0, 0.3], col![0.3, 0.0, 0.0]);

        let term1 = x.lie_bracket(&y.lie_bracket(&z));
        let term2 = y.lie_bracket(&z.lie_bracket(&x));
        let term3 = z.lie_bracket(&x.lie_bracket(&y));

        let jacobi_sum = SE3Tangent {
            data: &term1.data + &term2.data + &term3.data,
        };
        assert!(jacobi_sum.is_zero(1e-10));
    }

    #[test]
    fn test_se3_hat_matrix_structure() {
        let tangent = SE3Tangent::new(col![0.1, 0.2, 0.3], col![0.4, 0.5, 0.6]);
        let hat_matrix = tangent.hat();

        // Verify hat matrix structure for SE(3)
        // Top-right should be translation part
        assert_eq!(hat_matrix[(0, 3)], tangent.rho()[0]);
        assert_eq!(hat_matrix[(1, 3)], tangent.rho()[1]);
        assert_eq!(hat_matrix[(2, 3)], tangent.rho()[2]);

        // Top-left should be skew-symmetric matrix of rotation part
        let theta_hat = SO3Tangent::new(tangent.theta()).hat();
        let top_left = hat_matrix.submatrix(0, 0, 3, 3);
        assert!((&top_left - &theta_hat).norm_l2() < 1e-10);

        // Bottom row should be zeros
        assert_eq!(hat_matrix[(3, 0)], 0.0);
        assert_eq!(hat_matrix[(3, 1)], 0.0);
        assert_eq!(hat_matrix[(3, 2)], 0.0);
        assert_eq!(hat_matrix[(3, 3)], 0.0);
    }
}
