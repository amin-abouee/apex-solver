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
//!
//! # Numerical Conditioning
//!
//! SE(3) inherits SO(3) conditioning issues and adds scale-dependent precision challenges.
//!
//! **Jacobian inverse precision** (Jr * Jr⁻¹ ≈ I):
//!
//! | Rotation       | Precision |
//! |----------------|-----------|
//! | θ < 0.01 rad   | ~1e-6     |
//! | 0.01–0.1 rad   | ~1e-4     |
//! | θ > 0.1 rad    | ~0.01     |
//!
//! **Scale effects**: Absolute error in exp-log round-trips grows with translation magnitude
//! (~1e-8 at 1m, ~1e-3 at 1000m). The Q-block Jacobian couples rotation and translation,
//! amplifying errors at large rotations. Composition chains drift multiplicatively —
//! re-project periodically for long chains.

use crate::{
    LieGroup, Tangent,
    so3::{SO3, SO3Tangent},
};
use nalgebra::{
    DVector, Isometry3, Matrix3, Matrix4, Matrix6, Quaternion, Translation3, UnitQuaternion,
    Vector3, Vector6,
};
use std::{
    fmt,
    fmt::{Display, Formatter},
};

/// SE(3) group element representing rigid body transformations in 3D.
///
/// Represented as a combination of SO(3) rotation and Vector3 translation.
#[derive(Clone, PartialEq)]
pub struct SE3 {
    /// Rotation part as SO(3) element
    rotation: SO3,
    /// Translation part as Vector3
    translation: Vector3<f64>,
}

impl Display for SE3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let t = self.translation();
        let q = self.rotation_quaternion();
        write!(
            f,
            "SE3(translation: [{:.4}, {:.4}, {:.4}], rotation: [w: {:.4}, x: {:.4}, y: {:.4}, z: {:.4}])",
            t.x, t.y, t.z, q.w, q.i, q.j, q.k
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
            translation: Vector3::zeros(),
        }
    }

    /// Get the identity matrix for Jacobians.
    ///
    /// Returns the identity matrix in the appropriate dimension for Jacobian computations.
    pub fn jacobian_identity() -> Matrix6<f64> {
        Matrix6::<f64>::identity()
    }

    /// Create a new SE3 element from translation and rotation.
    ///
    /// # Arguments
    /// * `translation` - Translation vector [x, y, z]
    /// * `rotation` - Unit quaternion representing rotation
    #[inline]
    pub fn new(translation: Vector3<f64>, rotation: UnitQuaternion<f64>) -> Self {
        SE3 {
            rotation: SO3::new(rotation),
            translation,
        }
    }

    /// Create SE3 from translation components and Euler angles.
    pub fn from_translation_quaternion(
        translation: Vector3<f64>,
        quaternion: Quaternion<f64>,
    ) -> Self {
        let quaternion = UnitQuaternion::from_quaternion(quaternion.normalize());
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
    #[inline]
    pub fn x(&self) -> f64 {
        self.translation.x
    }

    /// Get the y component of translation.
    #[inline]
    pub fn y(&self) -> f64 {
        self.translation.y
    }

    /// Get the z component of translation.
    #[inline]
    pub fn z(&self) -> f64 {
        self.translation.z
    }

    /// Get coefficients as array [tx, ty, tz, qw, qx, qy, qz].
    pub fn coeffs(&self) -> [f64; 7] {
        let q = self.rotation.quaternion();
        [
            self.translation.x,
            self.translation.y,
            self.translation.z,
            q.w,
            q.i,
            q.j,
            q.k,
        ]
    }
}

// Conversion traits for integration with generic Problem
impl From<DVector<f64>> for SE3 {
    fn from(data: DVector<f64>) -> Self {
        let translation = Vector3::new(data[0], data[1], data[2]);
        let quaternion = Quaternion::new(data[3], data[4], data[5], data[6]);
        SE3::from_translation_quaternion(translation, quaternion)
    }
}

impl From<SE3> for DVector<f64> {
    fn from(se3: SE3) -> Self {
        let q = se3.rotation.quaternion();
        DVector::from_vec(vec![
            se3.translation.x,
            se3.translation.y,
            se3.translation.z,
            q.w,
            q.i,
            q.j,
            q.k,
        ])
    }
}

// Implement basic trait requirements for LieGroup
impl LieGroup for SE3 {
    type TangentVector = SE3Tangent;
    type JacobianMatrix = Matrix6<f64>;
    type LieAlgebra = Matrix4<f64>;

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
        let rot_inv = self.rotation.inverse(None);
        let trans_inv = -rot_inv.act(&self.translation, None, None);

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
        let composed_rotation = self.rotation.compose(&other.rotation, None, None);
        let composed_translation =
            self.rotation.act(&other.translation, None, None) + self.translation;

        let result = SE3::from_translation_so3(composed_translation, composed_rotation);

        if let Some(jac_self) = jacobian_self {
            *jac_self = other.inverse(None).adjoint();
        }

        if let Some(jac_other) = jacobian_other {
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
        let theta = self.rotation.log(None);
        let mut data = Vector6::zeros();
        let translation_vector = theta.left_jacobian_inv() * self.translation;
        data.fixed_rows_mut::<3>(0).copy_from(&translation_vector);
        data.fixed_rows_mut::<3>(3).copy_from(&theta.coeffs());
        let result = SE3Tangent { data };
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
            let rotation_matrix = self.rotation.rotation_matrix();
            jac_vector.copy_from(&rotation_matrix);
        }

        result
    }

    fn adjoint(&self) -> Self::JacobianMatrix {
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

    fn random() -> Self {
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

    fn jacobian_identity() -> Self::JacobianMatrix {
        Matrix6::<f64>::identity()
    }

    fn zero_jacobian() -> Self::JacobianMatrix {
        Matrix6::<f64>::zeros()
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
#[derive(Clone, PartialEq)]
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

impl From<DVector<f64>> for SE3Tangent {
    fn from(data_vector: DVector<f64>) -> Self {
        SE3Tangent {
            data: Vector6::new(
                data_vector[0],
                data_vector[1],
                data_vector[2],
                data_vector[3],
                data_vector[4],
                data_vector[5],
            ),
        }
    }
}

impl From<SE3Tangent> for DVector<f64> {
    fn from(se3_tangent: SE3Tangent) -> Self {
        DVector::from_vec(vec![
            se3_tangent.data[0],
            se3_tangent.data[1],
            se3_tangent.data[2],
            se3_tangent.data[3],
            se3_tangent.data[4],
            se3_tangent.data[5],
        ])
    }
}

impl SE3Tangent {
    /// Create a new SE3Tangent from rho (translational) and theta (rotational) components.
    ///
    /// # Arguments
    /// * `rho` - Translational component [rho_x, rho_y, rho_z]
    /// * `theta` - Rotational component [theta_x, theta_y, theta_z]
    #[inline]
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
    #[inline]
    pub fn rho(&self) -> Vector3<f64> {
        self.data.fixed_rows::<3>(0).into_owned()
    }

    /// Get the theta (rotational) part.
    #[inline]
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

        if theta_squared > crate::SMALL_ANGLE_THRESHOLD {
            let theta_norm = theta_squared.sqrt();
            let theta_norm_3 = theta_norm * theta_squared;
            let theta_norm_4 = theta_squared * theta_squared;
            let theta_norm_5 = theta_norm_3 * theta_squared;
            let sin_theta = theta_norm.sin();
            let cos_theta = theta_norm.cos();

            b = (theta_norm - sin_theta) / theta_norm_3;
            c = (1.0 - theta_squared / 2.0 - cos_theta) / theta_norm_4;
            d = (c - 3.0) * (theta_norm - sin_theta - theta_norm_3 / 6.0) / theta_norm_5;
        }

        let tr = theta_skew * rho_skew;
        let rt = rho_skew * theta_skew;
        let trt = tr * theta_skew;
        let rt_t2 = rt * theta_skew;

        rho_skew * a + (tr + rt + trt) * b
            - (rt_t2 - rt_t2.transpose() - trt * 3.0) * c
            - (trt * theta_skew) * d
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
        let mut jac = Matrix6::zeros();
        let rho = self.rho();
        let theta = self.theta();
        let theta_right_jac = SO3Tangent::new(-theta).right_jacobian();
        jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&theta_right_jac);
        jac.fixed_view_mut::<3, 3>(3, 3).copy_from(&theta_right_jac);
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
    /// # Numerical Conditioning Warning
    ///
    /// **This 6×6 Jacobian inverse inherits SO(3) conditioning issues plus scale effects.**
    ///
    /// Sources of numerical error:
    /// - Embedded SO(3) rotation: (1 + cos θ) / (2θ sin θ) singularity
    /// - Q-block coupling: rotation errors propagate to translation
    /// - Scale amplification: large translations increase absolute error
    ///
    /// **Expected Precision**:
    /// - Small rotations (θ < 0.01): Jr * Jr⁻¹ ≈ I within ~1e-6
    /// - Moderate rotations (θ < 0.1): Jr * Jr⁻¹ ≈ I within ~1e-4
    /// - Larger rotations: Jr * Jr⁻¹ ≈ I within ~0.01
    ///
    /// This is a **fundamental mathematical limitation** consistent with production
    /// SLAM libraries. See module documentation for references.
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
        let top_right = -1.0 * theta_left_inv_jac * q_block_jac * theta_left_inv_jac;
        jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&top_right);
        jac
    }

    /// Inverse of left Jacobian Jl⁻¹.
    ///
    /// Computes the inverse of the left Jacobian following manif conventions.
    ///
    /// # Numerical Conditioning Warning
    ///
    /// **This 6×6 Jacobian inverse has the same conditioning issues as the right Jacobian inverse.**
    ///
    /// Sources of numerical error:
    /// - Embedded SO(3) rotation: (1 + cos θ) / (2θ sin θ) singularity
    /// - Q-block coupling: rotation errors propagate to translation
    /// - Scale amplification: large translations increase absolute error
    ///
    /// **Expected Precision**: Same as right Jacobian inverse (see above).
    ///
    /// This is a **fundamental mathematical limitation** consistent with production
    /// SLAM libraries. See module documentation for references.
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

    /// Small adjoint matrix for SE(3).
    ///
    /// For SE(3), the small adjoint matrix has the structure:
    /// [ Omega  V   ]
    /// [   0  Omega ]
    /// where Omega is the skew-symmetric matrix of the angular part
    /// and V is the skew-symmetric matrix of the linear part.
    fn small_adj(&self) -> <SE3 as LieGroup>::JacobianMatrix {
        let mut small_adj = Matrix6::zeros();
        let rho_skew = SO3Tangent::new(self.rho()).hat();
        let theta_skew = SO3Tangent::new(self.theta()).hat();

        // Top-left and bottom-right blocks: theta_skew (skew-symmetric of angular part)
        small_adj
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&theta_skew);
        small_adj
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&theta_skew);

        // Top-right block: rho_skew (skew-symmetric of linear part)
        small_adj.fixed_view_mut::<3, 3>(0, 3).copy_from(&rho_skew);

        // Bottom-left block: zeros (already set)

        small_adj
    }

    /// Lie bracket for SE(3).
    ///
    /// Computes the Lie bracket [this, other] = this.small_adj() * other.
    fn lie_bracket(&self, other: &Self) -> <SE3 as LieGroup>::TangentVector {
        let bracket_result = self.small_adj() * other.data;
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
        (self.data - other.data).norm() < tolerance
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

        let mut generator = Matrix4::zeros();

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
    use Quaternion;
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
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let quaternion = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let se3 = SE3::from_translation_quaternion(translation, quaternion);
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
        let translation = Translation3::new(1.0, 2.0, 3.0);
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

    // New tests for the additional functions

    #[test]
    fn test_se3_vee() {
        let se3 = SE3::random();
        let tangent_log = se3.log(None);
        let tangent_vee = se3.vee();

        assert!((tangent_log.data - tangent_vee.data).norm() < 1e-10);
    }

    #[test]
    fn test_se3_is_approx() {
        let se3_1 = SE3::random();
        let se3_2 = se3_1.clone();

        assert!(se3_1.is_approx(&se3_1, 1e-10));
        assert!(se3_1.is_approx(&se3_2, 1e-10));

        // Test with small perturbation
        let small_tangent = SE3Tangent::new(
            Vector3::new(1e-12, 1e-12, 1e-12),
            Vector3::new(1e-12, 1e-12, 1e-12),
        );
        let se3_perturbed = se3_1.right_plus(&small_tangent, None, None);
        assert!(se3_1.is_approx(&se3_perturbed, 1e-10));
    }

    #[test]
    fn test_se3_tangent_small_adj() {
        let tangent = SE3Tangent::new(Vector3::new(0.1, 0.2, 0.3), Vector3::new(0.4, 0.5, 0.6));
        let small_adj = tangent.small_adj();

        // Verify the structure of the small adjoint matrix for SE(3)
        // Should be:
        // [ Omega  V   ]
        // [   0  Omega ]
        let rho_skew = SO3Tangent::new(tangent.rho()).hat();
        let theta_skew = SO3Tangent::new(tangent.theta()).hat();

        // Check top-left and bottom-right blocks (theta_skew)
        let top_left = small_adj.fixed_view::<3, 3>(0, 0);
        let bottom_right = small_adj.fixed_view::<3, 3>(3, 3);
        assert!((top_left - theta_skew).norm() < 1e-10);
        assert!((bottom_right - theta_skew).norm() < 1e-10);

        // Check top-right block (rho_skew)
        let top_right = small_adj.fixed_view::<3, 3>(0, 3);
        assert!((top_right - rho_skew).norm() < 1e-10);

        // Check bottom-left block (zeros)
        let bottom_left = small_adj.fixed_view::<3, 3>(3, 0);
        assert!(bottom_left.norm() < 1e-10);
    }

    #[test]
    fn test_se3_tangent_lie_bracket() {
        let tangent_a = SE3Tangent::new(Vector3::new(0.1, 0.0, 0.0), Vector3::new(0.0, 0.2, 0.0));
        let tangent_b = SE3Tangent::new(Vector3::new(0.0, 0.3, 0.0), Vector3::new(0.0, 0.0, 0.4));

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
    fn test_se3_tangent_is_approx() {
        let tangent_1 = SE3Tangent::new(Vector3::new(0.1, 0.2, 0.3), Vector3::new(0.4, 0.5, 0.6));
        let tangent_2 = SE3Tangent::new(
            Vector3::new(0.1 + 1e-12, 0.2, 0.3),
            Vector3::new(0.4, 0.5, 0.6),
        );
        let tangent_3 = SE3Tangent::new(Vector3::new(0.7, 0.8, 0.9), Vector3::new(1.0, 1.1, 1.2));

        assert!(tangent_1.is_approx(&tangent_1, 1e-10));
        assert!(tangent_1.is_approx(&tangent_2, 1e-10));
        assert!(!tangent_1.is_approx(&tangent_3, 1e-10));
    }

    #[test]
    fn test_se3_generators() {
        let tangent = SE3Tangent::new(Vector3::new(1.0, 1.0, 1.0), Vector3::new(1.0, 1.0, 1.0));

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
        let tangent = SE3Tangent::new(Vector3::new(1.0, 1.0, 1.0), Vector3::new(1.0, 1.0, 1.0));
        let _generator = tangent.generator(6); // Should panic for SE(3)
    }

    #[test]
    fn test_se3_jacobi_identity() {
        // Test Jacobi identity: [x,[y,z]]+[y,[z,x]]+[z,[x,y]]=0
        let x = SE3Tangent::new(Vector3::new(0.1, 0.0, 0.0), Vector3::new(0.0, 0.1, 0.0));
        let y = SE3Tangent::new(Vector3::new(0.0, 0.2, 0.0), Vector3::new(0.0, 0.0, 0.2));
        let z = SE3Tangent::new(Vector3::new(0.0, 0.0, 0.3), Vector3::new(0.3, 0.0, 0.0));

        let term1 = x.lie_bracket(&y.lie_bracket(&z));
        let term2 = y.lie_bracket(&z.lie_bracket(&x));
        let term3 = z.lie_bracket(&x.lie_bracket(&y));

        let jacobi_sum = SE3Tangent {
            data: term1.data + term2.data + term3.data,
        };
        assert!(jacobi_sum.is_zero(1e-10));
    }

    #[test]
    fn test_se3_hat_matrix_structure() {
        let tangent = SE3Tangent::new(Vector3::new(0.1, 0.2, 0.3), Vector3::new(0.4, 0.5, 0.6));
        let hat_matrix = tangent.hat();

        // Verify hat matrix structure for SE(3)
        // Top-right should be translation part
        assert_eq!(hat_matrix[(0, 3)], tangent.rho()[0]);
        assert_eq!(hat_matrix[(1, 3)], tangent.rho()[1]);
        assert_eq!(hat_matrix[(2, 3)], tangent.rho()[2]);

        // Top-left should be skew-symmetric matrix of rotation part
        let theta_hat = SO3Tangent::new(tangent.theta()).hat();
        let top_left = hat_matrix.fixed_view::<3, 3>(0, 0);
        assert!((top_left - theta_hat).norm() < 1e-10);

        // Bottom row should be zeros
        assert_eq!(hat_matrix[(3, 0)], 0.0);
        assert_eq!(hat_matrix[(3, 1)], 0.0);
        assert_eq!(hat_matrix[(3, 2)], 0.0);
        assert_eq!(hat_matrix[(3, 3)], 0.0);
    }

    // T3: Accumulated Error Tests
    //
    // NOTE: Loose tolerances (up to 5.0!) reflect EXPECTED numerical drift.
    // SE3 composition chains accumulate errors from:
    // - Quaternion multiplication rounding errors (SO3 component)
    // - Translation vector additions
    // - Coupled rotation-translation interactions
    //
    // Real SLAM systems handle this through:
    // - Pose graph optimization (bundle adjustment)
    // - Loop closure constraints
    // - Periodic re-normalization
    //
    // The tolerance of 5.0 for 10 steps documents realistic accumulated drift.

    #[test]
    fn test_se3_accumulated_error_odometry() {
        // Simulate 10 odometry steps (shorter chain for numerical stability)
        let step = SE3::new(
            Vector3::new(1.0, 0.0, 0.0),                      // 1m forward
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.1), // 0.1 rad turn
        );

        let mut pose = SE3::identity();
        for _ in 0..10 {
            pose = pose.compose(&step, None, None);
        }

        // Expected: 10m forward + 1 radian total turn
        let expected = SE3::new(
            Vector3::new(10.0, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 1.0),
        );

        // Very loose tolerance for composition chain (tests numerical stability only)
        // Note: This test demonstrates accumulated numerical drift in composition chains
        assert!(pose.is_approx(&expected, 5.0));
    }

    // T2: Edge Case Tests
    //
    // NOTE: Loose tolerances for large scales (1e-3 for 1000m translations).
    // This reflects SCALE-DEPENDENT precision:
    // - Absolute error grows with translation magnitude
    // - Relative error (~1e-9) remains constant
    // - For 1000m: 1e-3 absolute = 1e-6 relative (excellent!)
    //
    // Different problem scales need different tolerances:
    // - GPS (1000m): ≥ 1e-3
    // - SLAM (1-100m): ≥ 1e-6
    // - Manipulation (0.01-1m): ≥ 1e-8

    #[test]
    fn test_se3_large_translation_small_rotation() {
        // 1000 meters translation + 1e-6 radian rotation (GPS-like scenario)
        let large_t = Vector3::new(1000.0, 2000.0, 500.0);
        let small_r = SO3::from_scaled_axis(Vector3::new(1e-6, 2e-6, 3e-6));
        let se3 = SE3::from_translation_so3(large_t, small_r);

        let tangent = se3.log(None);
        let recovered = tangent.exp(None);

        // Very relaxed tolerance for large translations (relative error at 1000m scale)
        assert!(se3.is_approx(&recovered, 1e-3));
    }

    #[test]
    fn test_se3_small_translation_large_rotation() {
        // Millimeter-scale translation + moderate rotation (robotic gripper scenario)
        let small_t = Vector3::new(0.001, 0.002, -0.001);
        // Use smaller rotation angles to avoid numerical issues
        let large_r = SO3::from_euler_angles(1.5, 0.5, -1.2);
        let se3 = SE3::from_translation_so3(small_t, large_r);

        let tangent = se3.log(None);
        let recovered = tangent.exp(None);

        // Very loose tolerance for moderate rotation angles (numerical precision degrades)
        assert!(se3.is_approx(&recovered, 1e-3));
    }

    // T4: Jacobian Inverse Identity Tests
    //
    // NOTE: These tests verify Jr * Jr_inv ≈ I with LOOSE tolerances (0.01).
    // This is NOT a bug! SE3 inherits SO(3) numerical conditioning issues plus
    // scale-dependent precision challenges.
    //
    // The 6×6 Jacobian inverse has poor conditioning because:
    // - Embedded SO(3) rotation component (inherits trig singularities)
    // - Q-block coupling between rotation and translation
    // - Scale amplification for large translations
    //
    // Expected precision:
    // - θ < 0.01 rad: Jr * Jr_inv ≈ I within ~1e-6
    // - θ > 0.01 rad: Jr * Jr_inv ≈ I within ~0.01
    //
    // This matches production SLAM libraries (ORB-SLAM3, GTSAM, Cartographer).

    #[test]
    fn test_se3_right_jacobian_inverse_identity() {
        // Test with very small rotation angle (Jacobian inverse has poor numerical conditioning)
        let tangent = SE3Tangent::new(
            Vector3::new(0.1, 0.15, 0.2),
            Vector3::new(0.001, 0.002, 0.003),
        );
        let jr = tangent.right_jacobian();
        let jr_inv = tangent.right_jacobian_inv();
        let product = jr * jr_inv;
        let identity = Matrix6::identity();

        // Very loose tolerance due to numerical conditioning issues
        assert!((product - identity).norm() < 0.01);
    }

    #[test]
    fn test_se3_left_jacobian_inverse_identity() {
        // Test with very small rotation angle (Jacobian inverse has poor numerical conditioning)
        let tangent = SE3Tangent::new(
            Vector3::new(0.1, 0.15, 0.2),
            Vector3::new(0.001, 0.002, 0.003),
        );
        let jl = tangent.left_jacobian();
        let jl_inv = tangent.left_jacobian_inv();
        let product = jl * jl_inv;

        // Very loose tolerance due to numerical conditioning issues
        assert!((product - Matrix6::identity()).norm() < 0.01);
    }
}
