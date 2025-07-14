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
use crate::manifold::{LieGroup, Tangent};
use nalgebra::{
    Isometry3, Matrix3, Matrix4, Matrix6, Quaternion, Translation3, UnitQuaternion, Vector3,
    Vector6,
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
    pub fn rotation_so3(&self) -> SO3 {
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
}

// Implement basic trait requirements for LieGroup
impl LieGroup for SE3 {
    type Element = SE3;
    type TangentVector = SE3Tangent;
    type JacobianMatrix = Matrix6<f64>;
    type LieAlgebra = Matrix4<f64>;

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
            *jac = self.adjoint();
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
            *jac_self = -other.inverse(None).adjoint();
        }

        if let Some(jac_other) = jacobian_other {
            // Jacobian wrt second element: I (identity)
            *jac_other = Matrix6::identity();
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
        let mut data = Vector6::zeros();
        let translation_vector = theta.left_jacobian_inv() * self.translation;
        data.fixed_rows_mut::<3>(0).copy_from(&translation_vector);
        data.fixed_rows_mut::<3>(3).copy_from(&theta.coeffs());
        let result = SE3Tangent { data };
        if let Some(jac) = jacobian {
            *jac = result.left_jacobian_inv();
        }

        result
    }

    fn right_plus(
        &self,
        tangent: &Self::TangentVector,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_tangent: Option<&mut Self::JacobianMatrix>,
    ) -> Self::Element {
        // Right plus: g ⊕ τ = g * exp(τ)
        let exp_tangent = tangent.exp(None);
        let result = self.compose(&exp_tangent, None, None);

        if let Some(jac_self) = jacobian_self {
            *jac_self = tangent.right_jacobian() * tangent.left_jacobian_inv();
        }
        if let Some(jac_tangent) = jacobian_tangent {
            *jac_tangent = tangent.right_jacobian();
        }

        result
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
        let result = result_group.log(None);

        if let Some(jac_self) = jacobian_self {
            *jac_self = -result.left_jacobian_inv();
        }

        if let Some(jac_other) = jacobian_other {
            *jac_other = result.right_jacobian_inv();
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
        let exp_tangent = tangent.exp(None);
        let result = exp_tangent.compose(self, None, None);

        if let Some(jac_self) = jacobian_self {
            *jac_self = Matrix6::identity();
        }

        if let Some(jac_tangent) = jacobian_tangent {
            *jac_tangent = Matrix6::identity();
        }

        result
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
            *jac_other = Matrix6::identity();
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
            jac_self
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&self.rotation.rotation_matrix());
            jac_self
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-self.rotation.rotation_matrix() * SO3Tangent::new(*vector).hat()));
        }

        if let Some(jac_vector) = jacobian_vector {
            // Jacobian wrt vector
            let rotation_matrix = self.rotation.rotation_matrix();
            jac_vector.copy_from(&rotation_matrix);
        }

        result
    }

    fn adjoint(&self) -> Self::JacobianMatrix {
        // Adjoint matrix for SE(3)
        let rotation_matrix = self.rotation.rotation_matrix();
        let translation = self.translation;
        let mut adjoint_matrix = Matrix6::zeros();

        // Top-left block: R
        adjoint_matrix
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&rotation_matrix);

        // Bottom-right block: R
        adjoint_matrix
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&rotation_matrix);

        // Top-right block: [t]_× R (skew-symmetric of translation times rotation)
        let top_right = SO3Tangent::new(translation).hat() * rotation_matrix;
        adjoint_matrix
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&top_right);

        adjoint_matrix
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

    /// Equation 180: Q(ρ, θ) function for SE(3) Jacobians
    /// Q(ρ, θ) = (1/2)ρₓ + (θ - sin θ)/θ³ (θₓρₓ + ρₓθₓ + θₓρₓθₓ)
    ///           - (1 - θ²/2 - cos θ)/θ⁴ (θ²ₓρₓ + ρₓθ²ₓ - 3θₓρₓθₓ)
    ///           - (1/2) * ( (1 - θ²/2 - cos θ)/θ⁴ - (3.0 * (θ - sin θ - θ³/6))/θ⁵ ) * (θₓρₓθ²ₓ + θ²ₓρₓθₓ)
    pub fn q_block_jacobian_matrix(rho: Vector3<f64>, theta: Vector3<f64>) -> Matrix3<f64> {
        let rho_skew = SO3Tangent::new(rho).hat();
        let theta_skew = SO3Tangent::new(theta).hat();
        let theta_squared = theta.norm_squared();

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

        let rho_skew_theta_skew = rho_skew * theta_skew;
        let theta_skew_rho_skew = theta_skew * rho_skew;
        let theta_skew_rho_skew_theta_skew = theta_skew * rho_skew * theta_skew;
        let rho_skew_theta_skew_sq2 = rho_skew * theta_skew * theta_skew;

        // Calculate matrix terms
        let m1 = rho_skew;
        let m2 = theta_skew_rho_skew + rho_skew_theta_skew + theta_skew_rho_skew_theta_skew;
        let m3 = rho_skew_theta_skew_sq2
            - rho_skew_theta_skew_sq2.transpose()
            - 3.0 * theta_skew_rho_skew_theta_skew;
        let m4 = theta_skew_rho_skew_theta_skew * theta_skew;

        // Assemble the final matrix
        m1 * a + m2 * b - m3 * c - m4 * d
    }
}

// Implement LieAlgebra trait for SE3Tangent
impl Tangent<SE3> for SE3Tangent {
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
        &self,
        jacobian: Option<&mut <SE3 as LieGroup>::JacobianMatrix>,
    ) -> <SE3 as LieGroup>::Element {
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
        let mut jac = Matrix6::zeros();
        let rho = self.rho();
        let theta = self.theta();
        let theta_left_jac = SO3Tangent::new(-theta).left_jacobian();
        jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&theta_left_jac);
        jac.fixed_view_mut::<3, 3>(3, 3).copy_from(&theta_left_jac);
        jac.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&SE3Tangent::q_block_jacobian_matrix(-rho, -theta));
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
        let mut jac = Matrix6::zeros();
        let theta_left_jac = SO3Tangent::new(self.theta()).left_jacobian();
        jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&theta_left_jac);
        jac.fixed_view_mut::<3, 3>(3, 3).copy_from(&theta_left_jac);
        jac.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&SE3Tangent::q_block_jacobian_matrix(
                self.rho(),
                self.theta(),
            ));
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
        let mut jac = Matrix6::zeros();
        let rho = self.rho();
        let theta = self.theta();
        let theta_left_inv_jac = SO3Tangent::new(-theta).left_jacobian_inv();
        let q_block_jac = SE3Tangent::q_block_jacobian_matrix(-rho, -theta);
        jac.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&theta_left_inv_jac);
        jac.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&theta_left_inv_jac);
        // Bottom-right block: J_r^{-1}(theta)
        let bottom_right = -1.0 * theta_left_inv_jac * q_block_jac * theta_left_inv_jac;
        jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&bottom_right);
        jac
    }

    /// Inverse of left Jacobian Jl⁻¹.
    ///
    /// Computes the inverse of the left Jacobian following manif conventions.
    ///
    /// # Returns
    /// The inverse left Jacobian matrix (6x6)
    fn left_jacobian_inv(&self) -> <SE3 as LieGroup>::JacobianMatrix {
        let mut jac = Matrix6::zeros();
        let rho = self.rho();
        let theta = self.theta();
        let theta_left_inv_jac = SO3Tangent::new(theta).left_jacobian_inv();
        let q_block_jac = SE3Tangent::q_block_jacobian_matrix(rho, theta);
        let top_right_block = -1.0 * theta_left_inv_jac * q_block_jac * theta_left_inv_jac;
        jac.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&theta_left_inv_jac);
        jac.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&theta_left_inv_jac);
        jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&top_right_block);
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
        let mut lie_alg = Matrix4::zeros();

        // Top-left 3x3: skew-symmetric matrix of rotational part
        let theta_hat = SO3Tangent::new(self.theta()).hat();
        lie_alg.view_mut((0, 0), (3, 3)).copy_from(&theta_hat);

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
        SE3Tangent::new(Vector3::zeros(), Vector3::zeros())
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
        self.data.norm() < tolerance
    }

    /// Normalize the tangent vector to unit norm.
    ///
    /// Modifies this tangent vector to have unit norm. If the vector
    /// is near zero, it remains unchanged.
    fn normalize(&mut self) {
        let theta_norm = self.theta().norm();
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
        let norm = self.theta().norm();
        if norm > f64::EPSILON {
            SE3Tangent::new(self.rho(), self.theta() / norm)
        } else {
            SE3Tangent::new(self.rho(), Vector3::zeros())
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
    fn test_se3_tangent_zero() {
        let zero = SE3Tangent::zero();
        assert_eq!(zero.data, Vector6::zeros());

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
        let tangent_vec = Vector6::new(0.1, 0.2, 0.3, 0.01, 0.02, 0.03);
        let tangent = SE3Tangent { data: tangent_vec };

        // Test exp(log(g)) = g
        let se3 = tangent.exp(None);
        let recovered_tangent = se3.log(None);

        let diff = (tangent.data - recovered_tangent.data).norm();
        assert!(diff < TOLERANCE);
    }

    #[test]
    fn test_se3_exp_zero() {
        let zero_tangent = SE3Tangent::zero();
        let se3 = zero_tangent.exp(None);
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

        assert!(tangent.data.norm() < TOLERANCE);
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

        let se3 = SE3::new(
            Vector3::new(1e-8, 2e-8, 3e-8),
            UnitQuaternion::from_euler_angles(1e-9, 2e-9, 3e-9),
        );
        let recovered = se3.log(None);

        let diff = (small_tangent - recovered.data).norm();
        assert!(diff < TOLERANCE);
    }

    #[test]
    fn test_se3_tangent_norm() {
        let tangent_vec = Vector6::new(3.0, 4.0, 0.0, 0.0, 0.0, 0.0);
        let tangent = SE3Tangent { data: tangent_vec };

        let norm = tangent.data.norm();
        assert!((norm - 5.0).abs() < TOLERANCE); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_se3_from_components() {
        let se3 = SE3::from_translation_quaternion(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0);
        assert!(se3.is_valid(TOLERANCE));
        assert_eq!(se3.x(), 1.0);
        assert_eq!(se3.y(), 2.0);
        assert_eq!(se3.z(), 3.0);

        let quat = se3.rotation_quaternion();
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
    fn test_se3_tangent_utility_functions() {
        // Test zero
        let zero_vec = SE3Tangent::zero();
        assert!(zero_vec.data.norm() < TOLERANCE);

        // Test random
        let random_vec = SE3Tangent::random();
        assert!(random_vec.data.norm() > 0.0);

        // Test is_zero
        let tangent = SE3Tangent::new(Vector3::zeros(), Vector3::zeros());
        assert!(tangent.is_zero(1e-10));

        let non_zero_tangent = SE3Tangent::new(Vector3::new(1e-5, 0.0, 0.0), Vector3::zeros());
        assert!(!non_zero_tangent.is_zero(1e-10));
    }
}
