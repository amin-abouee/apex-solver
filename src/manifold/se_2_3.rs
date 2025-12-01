//! SE_2(3) - Extended Special Euclidean Group (Rotation + Translation + Velocity)
//!
//! This module implements the SE_2(3) Lie group, which represents extended pose
//! transformations in 3D space including rotation, translation, and velocity.
//!
//! SE_2(3) is the semi-direct product: SO(3) ⋉ (ℝ³ × ℝ³)
//!
//! SE_2(3) elements are represented as (R, t, v) where:
//! - R ∈ SO(3): rotation
//! - t ∈ ℝ³: translation (position)
//! - v ∈ ℝ³: velocity
//!
//! SE_2(3) tangent elements are represented as [ρ(3), θ(3), ν(3)] = 9 components:
//! - ρ: translational component
//! - θ: rotational component (axis-angle)
//! - ν: velocity component
//!
//! The implementation follows the [manif](https://github.com/artivis/manif) C++ library
//! conventions and provides all operations required by the LieGroup and Tangent traits.
//!
//! # References
//! - manif C++ library: include/manif/impl/se_2_3/
//! - "A micro Lie theory for state estimation in robotics" - Solà et al.

use crate::manifold::{
    LieGroup, Tangent,
    so3::{SO3, SO3Tangent},
};
use nalgebra::{DVector, Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};

// Type aliases for SE_2(3) - 9 DOF
type Vector9<T> = SVector<T, 9>;
type Matrix9<T> = SMatrix<T, 9, 9>;
use std::{
    fmt,
    fmt::{Display, Formatter},
};

/// SE_2(3) group element representing extended pose transformations in 3D.
///
/// Represented as (rotation, translation, velocity).
#[derive(Clone, PartialEq)]
pub struct SE_2_3 {
    /// Rotation part as SO(3) element
    rotation: SO3,
    /// Translation part (position) as Vector3
    translation: Vector3<f64>,
    /// Velocity part as Vector3
    velocity: Vector3<f64>,
}

impl Display for SE_2_3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let t = self.translation();
        let v = self.velocity();
        let q = self.rotation_quaternion();
        write!(
            f,
            "SE_2_3(translation: [{:.4}, {:.4}, {:.4}], velocity: [{:.4}, {:.4}, {:.4}], rotation: [w: {:.4}, x: {:.4}, y: {:.4}, z: {:.4}])",
            t.x, t.y, t.z, v.x, v.y, v.z, q.w, q.i, q.j, q.k
        )
    }
}

impl SE_2_3 {
    /// Space dimension - dimension of the ambient space
    pub const DIM: usize = 3;

    /// Degrees of freedom - dimension of the tangent space
    pub const DOF: usize = 9;

    /// Representation size - size of the underlying data representation
    pub const REP_SIZE: usize = 10;

    /// Get the identity element of the group.
    pub fn identity() -> Self {
        SE_2_3 {
            rotation: SO3::identity(),
            translation: Vector3::zeros(),
            velocity: Vector3::zeros(),
        }
    }

    /// Get the identity matrix for Jacobians.
    pub fn jacobian_identity() -> Matrix9<f64> {
        Matrix9::<f64>::identity()
    }

    /// Create a new SE_2(3) element from translation, velocity, and rotation.
    ///
    /// # Arguments
    /// * `translation` - Position vector [x, y, z]
    /// * `velocity` - Velocity vector [vx, vy, vz]
    /// * `rotation` - Unit quaternion representing rotation
    pub fn new(
        translation: Vector3<f64>,
        velocity: Vector3<f64>,
        rotation: UnitQuaternion<f64>,
    ) -> Self {
        SE_2_3 {
            rotation: SO3::new(rotation),
            translation,
            velocity,
        }
    }

    /// Create SE_2(3) from components.
    pub fn from_components(
        translation: Vector3<f64>,
        velocity: Vector3<f64>,
        rotation: SO3,
    ) -> Self {
        SE_2_3 {
            rotation,
            translation,
            velocity,
        }
    }

    /// Get the translation part as a Vector3.
    pub fn translation(&self) -> Vector3<f64> {
        self.translation
    }

    /// Get the velocity part as a Vector3.
    pub fn velocity(&self) -> Vector3<f64> {
        self.velocity
    }

    /// Get the rotation part as SO3.
    pub fn rotation_so3(&self) -> SO3 {
        self.rotation.clone()
    }

    /// Get the rotation part as a UnitQuaternion.
    pub fn rotation_quaternion(&self) -> UnitQuaternion<f64> {
        self.rotation.quaternion()
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

    /// Get the vx component of velocity.
    pub fn vx(&self) -> f64 {
        self.velocity.x
    }

    /// Get the vy component of velocity.
    pub fn vy(&self) -> f64 {
        self.velocity.y
    }

    /// Get the vz component of velocity.
    pub fn vz(&self) -> f64 {
        self.velocity.z
    }
}

// Conversion traits for integration with generic Problem
impl From<DVector<f64>> for SE_2_3 {
    fn from(data: DVector<f64>) -> Self {
        let translation = Vector3::new(data[0], data[1], data[2]);
        let velocity = Vector3::new(data[3], data[4], data[5]);
        let rotation = SO3::from_quaternion_coeffs(data[6], data[7], data[8], data[9]);
        SE_2_3::from_components(translation, velocity, rotation)
    }
}

impl From<SE_2_3> for DVector<f64> {
    fn from(se_2_3: SE_2_3) -> Self {
        let q = se_2_3.rotation.quaternion();
        DVector::from_vec(vec![
            se_2_3.translation.x,
            se_2_3.translation.y,
            se_2_3.translation.z,
            se_2_3.velocity.x,
            se_2_3.velocity.y,
            se_2_3.velocity.z,
            q.i,
            q.j,
            q.k,
            q.w,
        ])
    }
}

// Implement LieGroup trait
impl LieGroup for SE_2_3 {
    type TangentVector = SE_2_3Tangent;
    type JacobianMatrix = Matrix9<f64>;
    type LieAlgebra = SMatrix<f64, 5, 5>;

    /// Get the inverse.
    ///
    /// For SE_2(3): g^{-1} = (R^T, -R^T * t, -R^T * v)
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self {
        let rot_inv = self.rotation.inverse(None);
        let trans_inv = -rot_inv.act(&self.translation, None, None);
        let vel_inv = -rot_inv.act(&self.velocity, None, None);

        if let Some(jac) = jacobian {
            *jac = -self.adjoint();
        }

        SE_2_3::from_components(trans_inv, vel_inv, rot_inv)
    }

    /// Composition of this and another SE_2(3) element.
    ///
    /// g1 ∘ g2 = (R1*R2, R1*t2 + t1, R1*v2 + v1)
    fn compose(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        let composed_rotation = self.rotation.compose(&other.rotation, None, None);
        let composed_translation =
            self.rotation.act(&other.translation, None, None) + self.translation;
        let composed_velocity = self.rotation.act(&other.velocity, None, None) + self.velocity;

        let result =
            SE_2_3::from_components(composed_translation, composed_velocity, composed_rotation);

        if let Some(jac_self) = jacobian_self {
            *jac_self = other.inverse(None).adjoint();
        }

        if let Some(jac_other) = jacobian_other {
            *jac_other = Matrix9::identity();
        }

        result
    }

    /// Logarithmic map from SE_2(3) to its tangent space.
    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector {
        let theta = self.rotation.log(None);
        let mut data = Vector9::zeros();

        let v_inv = theta.left_jacobian_inv();
        let translation_vector = v_inv * self.translation;
        let velocity_vector = v_inv * self.velocity;

        data.fixed_rows_mut::<3>(0).copy_from(&translation_vector);
        data.fixed_rows_mut::<3>(3).copy_from(&theta.coeffs());
        data.fixed_rows_mut::<3>(6).copy_from(&velocity_vector);

        let result = SE_2_3Tangent { data };

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
            let rotation_matrix = self.rotation.rotation_matrix();
            jac_self
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&rotation_matrix);
            jac_self
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-rotation_matrix * SO3Tangent::new(*vector).hat()));
            jac_self.fixed_view_mut::<3, 3>(0, 6).fill(0.0);
        }

        if let Some(jac_vector) = jacobian_vector {
            *jac_vector = self.rotation.rotation_matrix();
        }

        result
    }

    fn adjoint(&self) -> Self::JacobianMatrix {
        let rotation_matrix = self.rotation.rotation_matrix();
        let translation = self.translation;
        let velocity = self.velocity;
        let mut adjoint_matrix = Matrix9::zeros();

        // Block structure:
        // [R    [t]×R   0  ]
        // [0      R     0  ]
        // [0    [v]×R   R  ]

        // Top-left: R
        adjoint_matrix
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&rotation_matrix);

        // Top-middle: [t]× R
        let top_middle = SO3Tangent::new(translation).hat() * rotation_matrix;
        adjoint_matrix
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&top_middle);

        // Middle-middle: R
        adjoint_matrix
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&rotation_matrix);

        // Bottom-middle: [v]× R
        let bottom_middle = SO3Tangent::new(velocity).hat() * rotation_matrix;
        adjoint_matrix
            .fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&bottom_middle);

        // Bottom-right: R
        adjoint_matrix
            .fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&rotation_matrix);

        adjoint_matrix
    }

    fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::rng();

        let translation = Vector3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        );

        let velocity = Vector3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        );

        let rotation = SO3::random();

        SE_2_3::from_components(translation, velocity, rotation)
    }

    fn normalize(&mut self) {
        self.rotation.normalize();
    }

    fn is_valid(&self, tolerance: f64) -> bool {
        self.rotation.is_valid(tolerance)
    }

    fn vee(&self) -> Self::TangentVector {
        self.log(None)
    }

    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        let difference = self.right_minus(other, None, None);
        difference.is_zero(tolerance)
    }
}

/// SE_2(3) tangent space element.
///
/// Represented as [ρ(3), θ(3), ν(3)] where:
/// - ρ: translational component
/// - θ: rotational component (axis-angle)
/// - ν: velocity component
#[derive(Clone, PartialEq)]
pub struct SE_2_3Tangent {
    /// Internal data: [ρ_x, ρ_y, ρ_z, θ_x, θ_y, θ_z, ν_x, ν_y, ν_z]
    data: Vector9<f64>,
}

impl fmt::Display for SE_2_3Tangent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rho = self.rho();
        let theta = self.theta();
        let nu = self.nu();
        write!(
            f,
            "se_2_3(rho: [{:.4}, {:.4}, {:.4}], theta: [{:.4}, {:.4}, {:.4}], nu: [{:.4}, {:.4}, {:.4}])",
            rho.x, rho.y, rho.z, theta.x, theta.y, theta.z, nu.x, nu.y, nu.z
        )
    }
}

impl From<DVector<f64>> for SE_2_3Tangent {
    fn from(data_vector: DVector<f64>) -> Self {
        SE_2_3Tangent {
            data: Vector9::from_iterator(data_vector.iter().copied()),
        }
    }
}

impl From<SE_2_3Tangent> for DVector<f64> {
    fn from(tangent: SE_2_3Tangent) -> Self {
        DVector::from_vec(tangent.data.as_slice().to_vec())
    }
}

impl SE_2_3Tangent {
    /// Create a new SE_2(3)Tangent from components.
    pub fn new(rho: Vector3<f64>, theta: Vector3<f64>, nu: Vector3<f64>) -> Self {
        let mut data = Vector9::zeros();
        data.fixed_rows_mut::<3>(0).copy_from(&rho);
        data.fixed_rows_mut::<3>(3).copy_from(&theta);
        data.fixed_rows_mut::<3>(6).copy_from(&nu);
        SE_2_3Tangent { data }
    }

    /// Get the ρ (translational) part.
    pub fn rho(&self) -> Vector3<f64> {
        self.data.fixed_rows::<3>(0).into_owned()
    }

    /// Get the θ (rotational) part.
    pub fn theta(&self) -> Vector3<f64> {
        self.data.fixed_rows::<3>(3).into_owned()
    }

    /// Get the ν (velocity) part.
    pub fn nu(&self) -> Vector3<f64> {
        self.data.fixed_rows::<3>(6).into_owned()
    }

    /// Compute the Q matrix for SE_2(3) Jacobians (same as SE(3)).
    fn q_matrix(rho: Vector3<f64>, theta: Vector3<f64>) -> Matrix3<f64> {
        let rho_skew = SO3Tangent::new(rho).hat();
        let theta_skew = SO3Tangent::new(theta).hat();
        let theta_squared = theta.norm_squared();

        let a = 0.5;
        let mut b = 1.0 / 6.0 + 1.0 / 120.0 * theta_squared;
        let mut c = -1.0 / 24.0 + 1.0 / 720.0 * theta_squared;
        let mut d = -1.0 / 60.0;

        if theta_squared > f64::EPSILON {
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

        let rho_skew_theta_skew = rho_skew * theta_skew;
        let theta_skew_rho_skew = theta_skew * rho_skew;
        let theta_skew_rho_skew_theta_skew = theta_skew * rho_skew * theta_skew;
        let rho_skew_theta_skew_sq2 = rho_skew * theta_skew * theta_skew;

        let m1 = rho_skew;
        let m2 = theta_skew_rho_skew + rho_skew_theta_skew + theta_skew_rho_skew_theta_skew;
        let m3 = rho_skew_theta_skew_sq2
            - rho_skew_theta_skew_sq2.transpose()
            - 3.0 * theta_skew_rho_skew_theta_skew;
        let m4 = theta_skew_rho_skew_theta_skew * theta_skew;

        m1 * a + m2 * b - m3 * c - m4 * d
    }
}

impl Tangent<SE_2_3> for SE_2_3Tangent {
    const DIM: usize = 9;

    /// Exponential map to SE_2(3).
    fn exp(&self, jacobian: Option<&mut <SE_2_3 as LieGroup>::JacobianMatrix>) -> SE_2_3 {
        let rho = self.rho();
        let theta = self.theta();
        let nu = self.nu();

        let theta_tangent = SO3Tangent::new(theta);
        let rotation = theta_tangent.exp(None);
        let v_matrix = theta_tangent.left_jacobian();
        let translation = v_matrix * rho;
        let velocity = v_matrix * nu;

        if let Some(jac) = jacobian {
            *jac = self.right_jacobian();
        }

        SE_2_3::from_components(translation, velocity, rotation)
    }

    /// Right Jacobian for SE_2(3).
    fn right_jacobian(&self) -> <SE_2_3 as LieGroup>::JacobianMatrix {
        let mut jac = Matrix9::zeros();
        let rho = self.rho();
        let theta = self.theta();
        let nu = self.nu();

        let theta_right_jac = SO3Tangent::new(-theta).right_jacobian();
        let q_rho = Self::q_matrix(-rho, -theta);
        let q_nu = Self::q_matrix(-nu, -theta);

        // Block structure similar to SE(3) but extended for velocity
        jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&theta_right_jac);
        jac.fixed_view_mut::<3, 3>(3, 3).copy_from(&theta_right_jac);
        jac.fixed_view_mut::<3, 3>(6, 6).copy_from(&theta_right_jac);
        jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&q_rho);
        jac.fixed_view_mut::<3, 3>(6, 3).copy_from(&q_nu);

        jac
    }

    /// Left Jacobian for SE_2(3).
    fn left_jacobian(&self) -> <SE_2_3 as LieGroup>::JacobianMatrix {
        let mut jac = Matrix9::zeros();
        let rho = self.rho();
        let theta = self.theta();
        let nu = self.nu();

        let theta_left_jac = SO3Tangent::new(theta).left_jacobian();
        let q_rho = Self::q_matrix(rho, theta);
        let q_nu = Self::q_matrix(nu, theta);

        jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&theta_left_jac);
        jac.fixed_view_mut::<3, 3>(3, 3).copy_from(&theta_left_jac);
        jac.fixed_view_mut::<3, 3>(6, 6).copy_from(&theta_left_jac);
        jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&q_rho);
        jac.fixed_view_mut::<3, 3>(6, 3).copy_from(&q_nu);

        jac
    }

    /// Inverse of right Jacobian.
    fn right_jacobian_inv(&self) -> <SE_2_3 as LieGroup>::JacobianMatrix {
        let mut jac = Matrix9::zeros();
        let rho = self.rho();
        let theta = self.theta();
        let nu = self.nu();

        let theta_left_inv_jac = SO3Tangent::new(-theta).left_jacobian_inv();
        let q_rho = Self::q_matrix(-rho, -theta);
        let q_nu = Self::q_matrix(-nu, -theta);

        jac.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&theta_left_inv_jac);
        jac.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&theta_left_inv_jac);
        jac.fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&theta_left_inv_jac);

        let top_right = -1.0 * theta_left_inv_jac * q_rho * theta_left_inv_jac;
        let bottom_right = -1.0 * theta_left_inv_jac * q_nu * theta_left_inv_jac;

        jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&top_right);
        jac.fixed_view_mut::<3, 3>(6, 3).copy_from(&bottom_right);

        jac
    }

    /// Inverse of left Jacobian.
    fn left_jacobian_inv(&self) -> <SE_2_3 as LieGroup>::JacobianMatrix {
        let mut jac = Matrix9::zeros();
        let rho = self.rho();
        let theta = self.theta();
        let nu = self.nu();

        let theta_left_inv_jac = SO3Tangent::new(theta).left_jacobian_inv();
        let q_rho = Self::q_matrix(rho, theta);
        let q_nu = Self::q_matrix(nu, theta);

        jac.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&theta_left_inv_jac);
        jac.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&theta_left_inv_jac);
        jac.fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&theta_left_inv_jac);

        let top_right = -1.0 * theta_left_inv_jac * q_rho * theta_left_inv_jac;
        let bottom_right = -1.0 * theta_left_inv_jac * q_nu * theta_left_inv_jac;

        jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&top_right);
        jac.fixed_view_mut::<3, 3>(6, 3).copy_from(&bottom_right);

        jac
    }

    /// Hat operator: maps tangent vector to Lie algebra matrix (5x5).
    fn hat(&self) -> <SE_2_3 as LieGroup>::LieAlgebra {
        let mut lie_alg = SMatrix::<f64, 5, 5>::zeros();

        let theta_hat = SO3Tangent::new(self.theta()).hat();
        lie_alg.view_mut((0, 0), (3, 3)).copy_from(&theta_hat);

        let rho = self.rho();
        let nu = self.nu();
        lie_alg[(0, 3)] = rho[0];
        lie_alg[(1, 3)] = rho[1];
        lie_alg[(2, 3)] = rho[2];
        lie_alg[(0, 4)] = nu[0];
        lie_alg[(1, 4)] = nu[1];
        lie_alg[(2, 4)] = nu[2];

        lie_alg
    }

    fn zero() -> <SE_2_3 as LieGroup>::TangentVector {
        SE_2_3Tangent::new(Vector3::zeros(), Vector3::zeros(), Vector3::zeros())
    }

    fn random() -> <SE_2_3 as LieGroup>::TangentVector {
        use rand::Rng;
        let mut rng = rand::rng();
        SE_2_3Tangent::new(
            Vector3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ),
            Vector3::new(
                rng.random_range(-0.1..0.1),
                rng.random_range(-0.1..0.1),
                rng.random_range(-0.1..0.1),
            ),
            Vector3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ),
        )
    }

    fn is_zero(&self, tolerance: f64) -> bool {
        self.data.norm() < tolerance
    }

    fn normalize(&mut self) {
        let theta_norm = self.theta().norm();
        if theta_norm > f64::EPSILON {
            self.data[3] /= theta_norm;
            self.data[4] /= theta_norm;
            self.data[5] /= theta_norm;
        }
    }

    fn normalized(&self) -> <SE_2_3 as LieGroup>::TangentVector {
        let norm = self.theta().norm();
        if norm > f64::EPSILON {
            SE_2_3Tangent::new(self.rho(), self.theta() / norm, self.nu())
        } else {
            SE_2_3Tangent::new(self.rho(), Vector3::zeros(), self.nu())
        }
    }

    fn small_adj(&self) -> <SE_2_3 as LieGroup>::JacobianMatrix {
        let mut small_adj = Matrix9::zeros();
        let rho_skew = SO3Tangent::new(self.rho()).hat();
        let theta_skew = SO3Tangent::new(self.theta()).hat();
        let nu_skew = SO3Tangent::new(self.nu()).hat();

        // Block structure:
        // [θ×   ρ×   0 ]
        // [0    θ×   0 ]
        // [0    ν×   θ×]

        small_adj
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&theta_skew);
        small_adj.fixed_view_mut::<3, 3>(0, 3).copy_from(&rho_skew);
        small_adj
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&theta_skew);
        small_adj.fixed_view_mut::<3, 3>(6, 3).copy_from(&nu_skew);
        small_adj
            .fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&theta_skew);

        small_adj
    }

    fn lie_bracket(&self, other: &Self) -> <SE_2_3 as LieGroup>::TangentVector {
        let bracket_result = self.small_adj() * other.data;
        SE_2_3Tangent {
            data: bracket_result,
        }
    }

    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        (self.data - other.data).norm() < tolerance
    }

    fn generator(&self, i: usize) -> <SE_2_3 as LieGroup>::LieAlgebra {
        assert!(i < 9, "SE_2(3) only has generators for indices 0-8");

        let mut generator = SMatrix::<f64, 5, 5>::zeros();

        match i {
            0..=2 => {
                // Translation generators (rho)
                generator[(i, 3)] = 1.0;
            }
            3..=5 => {
                // Rotation generators (theta)
                let idx = i - 3;
                match idx {
                    0 => {
                        generator[(1, 2)] = -1.0;
                        generator[(2, 1)] = 1.0;
                    }
                    1 => {
                        generator[(0, 2)] = 1.0;
                        generator[(2, 0)] = -1.0;
                    }
                    2 => {
                        generator[(0, 1)] = -1.0;
                        generator[(1, 0)] = 1.0;
                    }
                    _ => unreachable!(),
                }
            }
            6..=8 => {
                // Velocity generators (nu)
                let idx = i - 6;
                generator[(idx, 4)] = 1.0;
            }
            _ => unreachable!(),
        }

        generator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-9;

    #[test]
    fn test_se_2_3_identity() {
        let identity = SE_2_3::identity();
        assert!(identity.is_valid(TOLERANCE));
        assert!(identity.translation().norm() < TOLERANCE);
        assert!(identity.velocity().norm() < TOLERANCE);
        assert!(identity.rotation_quaternion().angle() < TOLERANCE);
    }

    #[test]
    fn test_se_2_3_new() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let velocity = Vector3::new(0.5, 0.6, 0.7);
        let rotation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);

        let se_2_3 = SE_2_3::new(translation, velocity, rotation);
        assert!(se_2_3.is_valid(TOLERANCE));
        assert!((se_2_3.translation() - translation).norm() < TOLERANCE);
        assert!((se_2_3.velocity() - velocity).norm() < TOLERANCE);
    }

    #[test]
    fn test_se_2_3_random() {
        let se_2_3 = SE_2_3::random();
        assert!(se_2_3.is_valid(TOLERANCE));
    }

    #[test]
    fn test_se_2_3_inverse() {
        let se_2_3 = SE_2_3::random();
        let se_2_3_inv = se_2_3.inverse(None);

        let composed = se_2_3.compose(&se_2_3_inv, None, None);
        let identity = SE_2_3::identity();

        assert!(composed.translation().norm() < TOLERANCE);
        assert!(composed.velocity().norm() < TOLERANCE);
        assert!(composed.rotation_quaternion().angle() < TOLERANCE);
        assert!(composed.is_approx(&identity, TOLERANCE));
    }

    #[test]
    fn test_se_2_3_compose() {
        let se_2_3_1 = SE_2_3::random();
        let se_2_3_2 = SE_2_3::random();

        let composed = se_2_3_1.compose(&se_2_3_2, None, None);
        assert!(composed.is_valid(TOLERANCE));

        // Test composition with identity
        let identity = SE_2_3::identity();
        let composed_with_identity = se_2_3_1.compose(&identity, None, None);
        assert!(composed_with_identity.is_approx(&se_2_3_1, TOLERANCE));
    }

    #[test]
    fn test_se_2_3_exp_log() {
        let tangent = SE_2_3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.01, 0.02, 0.03),
            Vector3::new(0.5, 0.6, 0.7),
        );

        let se_2_3 = tangent.exp(None);
        let recovered_tangent = se_2_3.log(None);

        assert!((tangent.data - recovered_tangent.data).norm() < TOLERANCE);
    }

    #[test]
    fn test_se_2_3_exp_zero() {
        let zero_tangent = SE_2_3Tangent::zero();
        let se_2_3 = zero_tangent.exp(None);
        let identity = SE_2_3::identity();

        assert!(se_2_3.is_approx(&identity, TOLERANCE));
    }

    #[test]
    fn test_se_2_3_log_identity() {
        let identity = SE_2_3::identity();
        let tangent = identity.log(None);

        assert!(tangent.data.norm() < TOLERANCE);
    }

    #[test]
    fn test_se_2_3_adjoint() {
        let se_2_3 = SE_2_3::random();
        let adj = se_2_3.adjoint();

        assert_eq!(adj.nrows(), 9);
        assert_eq!(adj.ncols(), 9);

        let det = adj.determinant();
        assert!((det - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se_2_3_act() {
        let se_2_3 = SE_2_3::random();
        let point = Vector3::new(1.0, 2.0, 3.0);

        let _transformed_point = se_2_3.act(&point, None, None);

        let identity = SE_2_3::identity();
        let identity_transformed = identity.act(&point, None, None);

        assert!((identity_transformed - point).norm() < TOLERANCE);
    }

    #[test]
    fn test_se_2_3_between() {
        let se_2_3_a = SE_2_3::random();
        let se_2_3_b = se_2_3_a.clone();
        let se_2_3_between_identity = se_2_3_a.between(&se_2_3_b, None, None);
        assert!(se_2_3_between_identity.is_approx(&SE_2_3::identity(), TOLERANCE));

        let se_2_3_c = SE_2_3::random();
        let se_2_3_between = se_2_3_a.between(&se_2_3_c, None, None);
        let expected = se_2_3_a.inverse(None).compose(&se_2_3_c, None, None);
        assert!(se_2_3_between.is_approx(&expected, TOLERANCE));
    }

    #[test]
    fn test_se_2_3_tangent_zero() {
        let zero = SE_2_3Tangent::zero();
        assert!(zero.data.norm() < TOLERANCE);

        let tangent = SE_2_3Tangent::new(Vector3::zeros(), Vector3::zeros(), Vector3::zeros());
        assert!(tangent.is_zero(TOLERANCE));
    }

    #[test]
    fn test_se_2_3_manifold_properties() {
        assert_eq!(SE_2_3::DIM, 3);
        assert_eq!(SE_2_3::DOF, 9);
        assert_eq!(SE_2_3::REP_SIZE, 10);
    }

    #[test]
    fn test_se_2_3_consistency() {
        let se_2_3_1 = SE_2_3::random();
        let se_2_3_2 = SE_2_3::random();
        let se_2_3_3 = SE_2_3::random();

        // Test associativity
        let left_assoc = se_2_3_1
            .compose(&se_2_3_2, None, None)
            .compose(&se_2_3_3, None, None);
        let right_assoc = se_2_3_1.compose(&se_2_3_2.compose(&se_2_3_3, None, None), None, None);

        assert!(left_assoc.is_approx(&right_assoc, 1e-10));
    }

    #[test]
    fn test_se_2_3_tangent_small_adj() {
        let tangent = SE_2_3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.4, 0.5, 0.6),
            Vector3::new(0.7, 0.8, 0.9),
        );
        let small_adj = tangent.small_adj();

        assert_eq!(small_adj.nrows(), 9);
        assert_eq!(small_adj.ncols(), 9);
    }

    #[test]
    fn test_se_2_3_tangent_lie_bracket() {
        let tangent_a = SE_2_3Tangent::new(
            Vector3::new(0.1, 0.0, 0.0),
            Vector3::new(0.0, 0.2, 0.0),
            Vector3::new(0.0, 0.0, 0.3),
        );
        let tangent_b = SE_2_3Tangent::new(
            Vector3::new(0.0, 0.3, 0.0),
            Vector3::new(0.0, 0.0, 0.4),
            Vector3::new(0.5, 0.0, 0.0),
        );

        let bracket_ab = tangent_a.lie_bracket(&tangent_b);
        let bracket_ba = tangent_b.lie_bracket(&tangent_a);

        // Anti-symmetry
        assert!((bracket_ab.data + bracket_ba.data).norm() < 1e-10);

        // [a,a] = 0
        let bracket_aa = tangent_a.lie_bracket(&tangent_a);
        assert!(bracket_aa.is_zero(1e-10));
    }

    #[test]
    fn test_se_2_3_tangent_is_approx() {
        let tangent_1 = SE_2_3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.4, 0.5, 0.6),
            Vector3::new(0.7, 0.8, 0.9),
        );
        let tangent_2 = SE_2_3Tangent::new(
            Vector3::new(0.1 + 1e-12, 0.2, 0.3),
            Vector3::new(0.4, 0.5, 0.6),
            Vector3::new(0.7, 0.8, 0.9),
        );
        let tangent_3 = SE_2_3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 5.0, 6.0),
            Vector3::new(7.0, 8.0, 9.0),
        );

        assert!(tangent_1.is_approx(&tangent_1, 1e-10));
        assert!(tangent_1.is_approx(&tangent_2, 1e-10));
        assert!(!tangent_1.is_approx(&tangent_3, 1e-10));
    }

    #[test]
    fn test_se_2_3_generators() {
        let tangent = SE_2_3Tangent::new(
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
        );

        for i in 0..9 {
            let generator = tangent.generator(i);
            assert_eq!(generator.nrows(), 5);
            assert_eq!(generator.ncols(), 5);

            // Bottom row should be zeros
            assert_eq!(generator[(3, 0)], 0.0);
            assert_eq!(generator[(3, 1)], 0.0);
            assert_eq!(generator[(3, 2)], 0.0);
            assert_eq!(generator[(3, 3)], 0.0);
            assert_eq!(generator[(3, 4)], 0.0);
            assert_eq!(generator[(4, 0)], 0.0);
            assert_eq!(generator[(4, 1)], 0.0);
            assert_eq!(generator[(4, 2)], 0.0);
            assert_eq!(generator[(4, 3)], 0.0);
            assert_eq!(generator[(4, 4)], 0.0);
        }
    }

    #[test]
    #[should_panic]
    fn test_se_2_3_generator_invalid_index() {
        let tangent = SE_2_3Tangent::new(
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
        );
        let _generator = tangent.generator(9);
    }

    #[test]
    fn test_se_2_3_vee() {
        let se_2_3 = SE_2_3::random();
        let tangent_log = se_2_3.log(None);
        let tangent_vee = se_2_3.vee();

        assert!((tangent_log.data - tangent_vee.data).norm() < 1e-10);
    }

    #[test]
    fn test_se_2_3_is_approx() {
        let se_2_3_1 = SE_2_3::random();
        let se_2_3_2 = se_2_3_1.clone();

        assert!(se_2_3_1.is_approx(&se_2_3_1, 1e-10));
        assert!(se_2_3_1.is_approx(&se_2_3_2, 1e-10));

        let small_tangent = SE_2_3Tangent::new(
            Vector3::new(1e-12, 1e-12, 1e-12),
            Vector3::new(1e-12, 1e-12, 1e-12),
            Vector3::new(1e-12, 1e-12, 1e-12),
        );
        let se_2_3_perturbed = se_2_3_1.right_plus(&small_tangent, None, None);
        assert!(se_2_3_1.is_approx(&se_2_3_perturbed, 1e-10));
    }

    #[test]
    fn test_se_2_3_small_angle_approximations() {
        let small_tangent = SE_2_3Tangent::new(
            Vector3::new(1e-8, 2e-8, 3e-8),
            Vector3::new(1e-9, 2e-9, 3e-9),
            Vector3::new(4e-8, 5e-8, 6e-8),
        );

        let se_2_3 = small_tangent.exp(None);
        let recovered = se_2_3.log(None);

        assert!((small_tangent.data - recovered.data).norm() < TOLERANCE);
    }

    #[test]
    fn test_se_2_3_accessors() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let velocity = Vector3::new(4.0, 5.0, 6.0);
        let rotation = UnitQuaternion::identity();

        let se_2_3 = SE_2_3::new(translation, velocity, rotation);

        assert_eq!(se_2_3.x(), 1.0);
        assert_eq!(se_2_3.y(), 2.0);
        assert_eq!(se_2_3.z(), 3.0);
        assert_eq!(se_2_3.vx(), 4.0);
        assert_eq!(se_2_3.vy(), 5.0);
        assert_eq!(se_2_3.vz(), 6.0);
    }
}
