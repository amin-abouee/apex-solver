//! SGal(3) - Special Galilean Group
//!
//! This module implements the Special Galilean group SGal(3), which represents
//! extended transformations including rotation, translation, velocity, and time.
//!
//! SGal(3) is a 10-dimensional Lie group representing the Galilean transformations
//! used in inertial navigation and kinematics.
//!
//! SGal(3) elements are represented as (R, t, v, s) where:
//! - R ∈ SO(3): rotation
//! - t ∈ ℝ³: translation (position)
//! - v ∈ ℝ³: velocity
//! - s ∈ ℝ: time/scale parameter
//!
//! SGal(3) tangent elements are represented as [ρ(3), ν(3), θ(3), s(1)] = 10 components.
//! **IMPORTANT**: Note the ordering - velocity ν comes BEFORE rotation θ!
//! - ρ: translational component
//! - ν: velocity component
//! - θ: rotational component (axis-angle)
//! - s: time parameter
//!
//! The implementation follows the [manif](https://github.com/artivis/manif) C++ library
//! conventions and provides all operations required by the LieGroup and Tangent traits.
//!
//! # References
//! - manif C++ library: include/manif/impl/sgal3/
//! - "All About the Galilean Group SGal(3)" (arXiv:2312.07555)

use crate::manifold::{
    LieGroup, Tangent,
    so3::{SO3, SO3Tangent},
};
use nalgebra::{DVector, Matrix3, SMatrix, UnitQuaternion, Vector3};
use std::{
    fmt,
    fmt::{Display, Formatter},
};

/// Type alias for 10x10 matrix
pub type Matrix10<T> = SMatrix<T, 10, 10>;
/// Type alias for 10x1 vector
pub type Vector10<T> = SMatrix<T, 10, 1>;

/// SGal(3) group element representing Galilean transformations.
///
/// Represented as (rotation, translation, velocity, time).
#[derive(Clone, PartialEq)]
pub struct SGal3 {
    /// Rotation part as SO(3) element
    rotation: SO3,
    /// Translation part (position) as Vector3
    translation: Vector3<f64>,
    /// Velocity part as Vector3
    velocity: Vector3<f64>,
    /// Time parameter
    time: f64,
}

impl Display for SGal3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let t = self.translation();
        let v = self.velocity();
        let s = self.time();
        let q = self.rotation_quaternion();
        write!(
            f,
            "SGal3(translation: [{:.4}, {:.4}, {:.4}], velocity: [{:.4}, {:.4}, {:.4}], time: {:.4}, rotation: [w: {:.4}, x: {:.4}, y: {:.4}, z: {:.4}])",
            t.x, t.y, t.z, v.x, v.y, v.z, s, q.w, q.i, q.j, q.k
        )
    }
}

impl SGal3 {
    /// Space dimension - dimension of the ambient space
    pub const DIM: usize = 3;

    /// Degrees of freedom - dimension of the tangent space
    pub const DOF: usize = 10;

    /// Representation size - size of the underlying data representation
    pub const REP_SIZE: usize = 11;

    /// Get the identity element of the group.
    pub fn identity() -> Self {
        SGal3 {
            rotation: SO3::identity(),
            translation: Vector3::zeros(),
            velocity: Vector3::zeros(),
            time: 0.0,
        }
    }

    /// Get the identity matrix for Jacobians.
    pub fn jacobian_identity() -> Matrix10<f64> {
        Matrix10::<f64>::identity()
    }

    /// Create a new SGal(3) element from components.
    ///
    /// # Arguments
    /// * `translation` - Position vector [x, y, z]
    /// * `velocity` - Velocity vector [vx, vy, vz]
    /// * `rotation` - Unit quaternion representing rotation
    /// * `time` - Time parameter
    pub fn new(
        translation: Vector3<f64>,
        velocity: Vector3<f64>,
        rotation: UnitQuaternion<f64>,
        time: f64,
    ) -> Self {
        SGal3 {
            rotation: SO3::new(rotation),
            translation,
            velocity,
            time,
        }
    }

    /// Create SGal(3) from components.
    pub fn from_components(
        translation: Vector3<f64>,
        velocity: Vector3<f64>,
        rotation: SO3,
        time: f64,
    ) -> Self {
        SGal3 {
            rotation,
            translation,
            velocity,
            time,
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

    /// Get the time parameter.
    pub fn time(&self) -> f64 {
        self.time
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
impl From<DVector<f64>> for SGal3 {
    fn from(data: DVector<f64>) -> Self {
        let translation = Vector3::new(data[0], data[1], data[2]);
        let velocity = Vector3::new(data[3], data[4], data[5]);
        let rotation = SO3::from_quaternion_coeffs(data[6], data[7], data[8], data[9]);
        let time = data[10];
        SGal3::from_components(translation, velocity, rotation, time)
    }
}

impl From<SGal3> for DVector<f64> {
    fn from(sgal3: SGal3) -> Self {
        let q = sgal3.rotation.quaternion();
        DVector::from_vec(vec![
            sgal3.translation.x,
            sgal3.translation.y,
            sgal3.translation.z,
            sgal3.velocity.x,
            sgal3.velocity.y,
            sgal3.velocity.z,
            q.i,
            q.j,
            q.k,
            q.w,
            sgal3.time,
        ])
    }
}

// Implement LieGroup trait
impl LieGroup for SGal3 {
    type TangentVector = SGal3Tangent;
    type JacobianMatrix = Matrix10<f64>;
    type LieAlgebra = SMatrix<f64, 6, 6>;

    /// Get the inverse.
    ///
    /// For SGal(3): g^{-1} = (R^T, -R^T * (t - s*v), -R^T * v, -s)
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self {
        let rot_inv = self.rotation.inverse(None);
        let trans_inv = -rot_inv.act(&(self.translation - self.time * self.velocity), None, None);
        let vel_inv = -rot_inv.act(&self.velocity, None, None);
        let time_inv = -self.time;

        if let Some(jac) = jacobian {
            *jac = -self.adjoint();
        }

        SGal3::from_components(trans_inv, vel_inv, rot_inv, time_inv)
    }

    /// Composition of this and another SGal(3) element.
    ///
    /// g1 ∘ g2 = (R1*R2, R1*(t2 + s1*v2) + t1, R1*v2 + v1, s1 + s2)
    fn compose(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        let composed_rotation = self.rotation.compose(&other.rotation, None, None);
        let composed_translation = self.rotation.act(
            &(other.translation + self.time * other.velocity),
            None,
            None,
        ) + self.translation;
        let composed_velocity = self.rotation.act(&other.velocity, None, None) + self.velocity;
        let composed_time = self.time + other.time;

        let result = SGal3::from_components(
            composed_translation,
            composed_velocity,
            composed_rotation,
            composed_time,
        );

        if let Some(jac_self) = jacobian_self {
            *jac_self = other.inverse(None).adjoint();
        }

        if let Some(jac_other) = jacobian_other {
            *jac_other = Matrix10::identity();
        }

        result
    }

    /// Logarithmic map from SGal(3) to its tangent space.
    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector {
        let theta = self.rotation.log(None);
        let mut data = Vector10::zeros();

        let v_inv = theta.left_jacobian_inv();
        let translation_vector = v_inv * self.translation;
        let velocity_vector = v_inv * self.velocity;

        // Order: [ρ, ν, θ, s]
        data.fixed_rows_mut::<3>(0).copy_from(&translation_vector);
        data.fixed_rows_mut::<3>(3).copy_from(&velocity_vector);
        data.fixed_rows_mut::<3>(6).copy_from(&theta.coeffs());
        data[9] = self.time;

        let result = SGal3Tangent { data };

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
        let result =
            self.rotation.act(vector, None, None) + self.translation + self.time * self.velocity;

        if let Some(jac_self) = jacobian_self {
            let rotation_matrix = self.rotation.rotation_matrix();
            jac_self
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&rotation_matrix);
            jac_self
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(self.time * rotation_matrix));
            jac_self
                .fixed_view_mut::<3, 3>(0, 6)
                .copy_from(&(-rotation_matrix * SO3Tangent::new(*vector).hat()));
            jac_self
                .fixed_view_mut::<3, 1>(0, 9)
                .copy_from(&self.velocity);
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
        let time = self.time;
        let mut adjoint_matrix = Matrix10::zeros();

        // Block structure for SGal(3):
        // [R    0    [t]×R    tv  ]
        // [0    R    [v]×R     v  ]
        // [0    0      R       0  ]
        // [0    0      0       1  ]

        // Top-left: R
        adjoint_matrix
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&rotation_matrix);

        // (0,6): [t]× R
        let block_06 = SO3Tangent::new(translation).hat() * rotation_matrix;
        adjoint_matrix
            .fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&block_06);

        // (0,9): t*v
        adjoint_matrix
            .fixed_view_mut::<3, 1>(0, 9)
            .copy_from(&(time * velocity));

        // (3,3): R
        adjoint_matrix
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&rotation_matrix);

        // (3,6): [v]× R
        let block_36 = SO3Tangent::new(velocity).hat() * rotation_matrix;
        adjoint_matrix
            .fixed_view_mut::<3, 3>(3, 6)
            .copy_from(&block_36);

        // (3,9): v
        adjoint_matrix
            .fixed_view_mut::<3, 1>(3, 9)
            .copy_from(&velocity);

        // (6,6): R
        adjoint_matrix
            .fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&rotation_matrix);

        // (9,9): 1
        adjoint_matrix[(9, 9)] = 1.0;

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
        let time = rng.random_range(-1.0..1.0);

        SGal3::from_components(translation, velocity, rotation, time)
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

/// SGal(3) tangent space element.
///
/// Represented as [ρ(3), ν(3), θ(3), s(1)] where:
/// - ρ: translational component
/// - ν: velocity component
/// - θ: rotational component (axis-angle)
/// - s: time parameter
///
/// **IMPORTANT**: Note the ordering - velocity ν comes BEFORE rotation θ!
#[derive(Clone, PartialEq)]
pub struct SGal3Tangent {
    /// Internal data: [ρ_x, ρ_y, ρ_z, ν_x, ν_y, ν_z, θ_x, θ_y, θ_z, s]
    data: Vector10<f64>,
}

impl fmt::Display for SGal3Tangent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rho = self.rho();
        let nu = self.nu();
        let theta = self.theta();
        let s = self.s();
        write!(
            f,
            "sgal3(rho: [{:.4}, {:.4}, {:.4}], nu: [{:.4}, {:.4}, {:.4}], theta: [{:.4}, {:.4}, {:.4}], s: {:.4})",
            rho.x, rho.y, rho.z, nu.x, nu.y, nu.z, theta.x, theta.y, theta.z, s
        )
    }
}

impl From<DVector<f64>> for SGal3Tangent {
    fn from(data_vector: DVector<f64>) -> Self {
        SGal3Tangent {
            data: Vector10::from_iterator(data_vector.iter().copied()),
        }
    }
}

impl From<SGal3Tangent> for DVector<f64> {
    fn from(tangent: SGal3Tangent) -> Self {
        DVector::from_vec(tangent.data.as_slice().to_vec())
    }
}

impl SGal3Tangent {
    /// Create a new SGal(3)Tangent from components.
    /// Order: [ρ, ν, θ, s]
    pub fn new(rho: Vector3<f64>, nu: Vector3<f64>, theta: Vector3<f64>, s: f64) -> Self {
        let mut data = Vector10::zeros();
        data.fixed_rows_mut::<3>(0).copy_from(&rho);
        data.fixed_rows_mut::<3>(3).copy_from(&nu);
        data.fixed_rows_mut::<3>(6).copy_from(&theta);
        data[9] = s;
        SGal3Tangent { data }
    }

    /// Get the ρ (translational) part.
    pub fn rho(&self) -> Vector3<f64> {
        self.data.fixed_rows::<3>(0).into_owned()
    }

    /// Get the ν (velocity) part.
    pub fn nu(&self) -> Vector3<f64> {
        self.data.fixed_rows::<3>(3).into_owned()
    }

    /// Get the θ (rotational) part.
    pub fn theta(&self) -> Vector3<f64> {
        self.data.fixed_rows::<3>(6).into_owned()
    }

    /// Get the s (time) part.
    pub fn s(&self) -> f64 {
        self.data[9]
    }

    /// Compute the Q matrix for SGal(3) Jacobians (same as SE(3)).
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

impl Tangent<SGal3> for SGal3Tangent {
    const DIM: usize = 10;

    /// Exponential map to SGal(3).
    fn exp(&self, jacobian: Option<&mut <SGal3 as LieGroup>::JacobianMatrix>) -> SGal3 {
        let rho = self.rho();
        let nu = self.nu();
        let theta = self.theta();
        let s = self.s();

        let theta_tangent = SO3Tangent::new(theta);
        let rotation = theta_tangent.exp(None);
        let v_matrix = theta_tangent.left_jacobian();
        let translation = v_matrix * rho;
        let velocity = v_matrix * nu;

        if let Some(jac) = jacobian {
            *jac = self.right_jacobian();
        }

        SGal3::from_components(translation, velocity, rotation, s)
    }

    /// Right Jacobian for SGal(3).
    fn right_jacobian(&self) -> <SGal3 as LieGroup>::JacobianMatrix {
        let mut jac = Matrix10::zeros();
        let rho = self.rho();
        let nu = self.nu();
        let theta = self.theta();

        let theta_right_jac = SO3Tangent::new(-theta).right_jacobian();
        let q_rho = Self::q_matrix(-rho, -theta);
        let q_nu = Self::q_matrix(-nu, -theta);

        // Block structure for SGal(3) with ordering [ρ, ν, θ, s]
        jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&theta_right_jac);
        jac.fixed_view_mut::<3, 3>(3, 3).copy_from(&theta_right_jac);
        jac.fixed_view_mut::<3, 3>(6, 6).copy_from(&theta_right_jac);
        jac.fixed_view_mut::<3, 3>(0, 6).copy_from(&q_rho);
        jac.fixed_view_mut::<3, 3>(3, 6).copy_from(&q_nu);
        jac[(9, 9)] = 1.0;

        jac
    }

    /// Left Jacobian for SGal(3).
    fn left_jacobian(&self) -> <SGal3 as LieGroup>::JacobianMatrix {
        let mut jac = Matrix10::zeros();
        let rho = self.rho();
        let nu = self.nu();
        let theta = self.theta();

        let theta_left_jac = SO3Tangent::new(theta).left_jacobian();
        let q_rho = Self::q_matrix(rho, theta);
        let q_nu = Self::q_matrix(nu, theta);

        jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&theta_left_jac);
        jac.fixed_view_mut::<3, 3>(3, 3).copy_from(&theta_left_jac);
        jac.fixed_view_mut::<3, 3>(6, 6).copy_from(&theta_left_jac);
        jac.fixed_view_mut::<3, 3>(0, 6).copy_from(&q_rho);
        jac.fixed_view_mut::<3, 3>(3, 6).copy_from(&q_nu);
        jac[(9, 9)] = 1.0;

        jac
    }

    /// Inverse of right Jacobian.
    fn right_jacobian_inv(&self) -> <SGal3 as LieGroup>::JacobianMatrix {
        let mut jac = Matrix10::zeros();
        let rho = self.rho();
        let nu = self.nu();
        let theta = self.theta();

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
        let mid_right = -1.0 * theta_left_inv_jac * q_nu * theta_left_inv_jac;

        jac.fixed_view_mut::<3, 3>(0, 6).copy_from(&top_right);
        jac.fixed_view_mut::<3, 3>(3, 6).copy_from(&mid_right);
        jac[(9, 9)] = 1.0;

        jac
    }

    /// Inverse of left Jacobian.
    fn left_jacobian_inv(&self) -> <SGal3 as LieGroup>::JacobianMatrix {
        let mut jac = Matrix10::zeros();
        let rho = self.rho();
        let nu = self.nu();
        let theta = self.theta();

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
        let mid_right = -1.0 * theta_left_inv_jac * q_nu * theta_left_inv_jac;

        jac.fixed_view_mut::<3, 3>(0, 6).copy_from(&top_right);
        jac.fixed_view_mut::<3, 3>(3, 6).copy_from(&mid_right);
        jac[(9, 9)] = 1.0;

        jac
    }

    /// Hat operator: maps tangent vector to Lie algebra matrix (6x6).
    fn hat(&self) -> <SGal3 as LieGroup>::LieAlgebra {
        let mut lie_alg = SMatrix::<f64, 6, 6>::zeros();

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
        lie_alg[(3, 5)] = 1.0;
        lie_alg[(4, 5)] = self.s();

        lie_alg
    }

    fn zero() -> <SGal3 as LieGroup>::TangentVector {
        SGal3Tangent::new(Vector3::zeros(), Vector3::zeros(), Vector3::zeros(), 0.0)
    }

    fn random() -> <SGal3 as LieGroup>::TangentVector {
        use rand::Rng;
        let mut rng = rand::rng();
        SGal3Tangent::new(
            Vector3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ),
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
            rng.random_range(-1.0..1.0),
        )
    }

    fn is_zero(&self, tolerance: f64) -> bool {
        self.data.norm() < tolerance
    }

    fn normalize(&mut self) {
        let theta_norm = self.theta().norm();
        if theta_norm > f64::EPSILON {
            self.data[6] /= theta_norm;
            self.data[7] /= theta_norm;
            self.data[8] /= theta_norm;
        }
    }

    fn normalized(&self) -> <SGal3 as LieGroup>::TangentVector {
        let norm = self.theta().norm();
        if norm > f64::EPSILON {
            SGal3Tangent::new(self.rho(), self.nu(), self.theta() / norm, self.s())
        } else {
            SGal3Tangent::new(self.rho(), self.nu(), Vector3::zeros(), self.s())
        }
    }

    fn small_adj(&self) -> <SGal3 as LieGroup>::JacobianMatrix {
        let mut small_adj = Matrix10::zeros();
        let rho_skew = SO3Tangent::new(self.rho()).hat();
        let nu_skew = SO3Tangent::new(self.nu()).hat();
        let theta_skew = SO3Tangent::new(self.theta()).hat();

        // Block structure for SGal(3) with ordering [ρ, ν, θ, s]:
        // [θ×   0    ρ×   ρ ]
        // [0    θ×   ν×   ν ]
        // [0    0    θ×   0 ]
        // [0    0    0    0 ]

        small_adj
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&theta_skew);
        small_adj.fixed_view_mut::<3, 3>(0, 6).copy_from(&rho_skew);
        small_adj
            .fixed_view_mut::<3, 1>(0, 9)
            .copy_from(&self.rho());

        small_adj
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&theta_skew);
        small_adj.fixed_view_mut::<3, 3>(3, 6).copy_from(&nu_skew);
        small_adj.fixed_view_mut::<3, 1>(3, 9).copy_from(&self.nu());

        small_adj
            .fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&theta_skew);

        small_adj
    }

    fn lie_bracket(&self, other: &Self) -> <SGal3 as LieGroup>::TangentVector {
        let bracket_result = self.small_adj() * other.data;
        SGal3Tangent {
            data: bracket_result,
        }
    }

    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        (self.data - other.data).norm() < tolerance
    }

    fn generator(&self, i: usize) -> <SGal3 as LieGroup>::LieAlgebra {
        assert!(i < 10, "SGal(3) only has generators for indices 0-9");

        let mut generator = SMatrix::<f64, 6, 6>::zeros();

        match i {
            0..=2 => {
                // Translation generators (rho)
                generator[(i, 3)] = 1.0;
            }
            3..=5 => {
                // Velocity generators (nu)
                let idx = i - 3;
                generator[(idx, 4)] = 1.0;
            }
            6..=8 => {
                // Rotation generators (theta)
                let idx = i - 6;
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
            9 => {
                // Time generator (s)
                generator[(3, 5)] = 1.0;
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
    fn test_sgal3_identity() {
        let identity = SGal3::identity();
        assert!(identity.is_valid(TOLERANCE));
        assert!(identity.translation().norm() < TOLERANCE);
        assert!(identity.velocity().norm() < TOLERANCE);
        assert!(identity.time().abs() < TOLERANCE);
        assert!(identity.rotation_quaternion().angle() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_new() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let velocity = Vector3::new(0.5, 0.6, 0.7);
        let rotation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let time = 0.5;

        let sgal3 = SGal3::new(translation, velocity, rotation, time);
        assert!(sgal3.is_valid(TOLERANCE));
        assert!((sgal3.translation() - translation).norm() < TOLERANCE);
        assert!((sgal3.velocity() - velocity).norm() < TOLERANCE);
        assert!((sgal3.time() - time).abs() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_random() {
        let sgal3 = SGal3::random();
        assert!(sgal3.is_valid(TOLERANCE));
    }

    #[test]
    fn test_sgal3_inverse() {
        let sgal3 = SGal3::random();
        let sgal3_inv = sgal3.inverse(None);

        let composed = sgal3.compose(&sgal3_inv, None, None);
        let identity = SGal3::identity();

        assert!(composed.is_approx(&identity, TOLERANCE));
    }

    #[test]
    fn test_sgal3_compose() {
        let sgal3_1 = SGal3::random();
        let sgal3_2 = SGal3::random();

        let composed = sgal3_1.compose(&sgal3_2, None, None);
        assert!(composed.is_valid(TOLERANCE));

        let identity = SGal3::identity();
        let composed_with_identity = sgal3_1.compose(&identity, None, None);
        assert!(composed_with_identity.is_approx(&sgal3_1, TOLERANCE));
    }

    #[test]
    fn test_sgal3_exp_log() {
        let tangent = SGal3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.5, 0.6, 0.7),
            Vector3::new(0.01, 0.02, 0.03),
            0.5,
        );

        let sgal3 = tangent.exp(None);
        let recovered_tangent = sgal3.log(None);

        assert!((tangent.data - recovered_tangent.data).norm() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_exp_zero() {
        let zero_tangent = SGal3Tangent::zero();
        let sgal3 = zero_tangent.exp(None);
        let identity = SGal3::identity();

        assert!(sgal3.is_approx(&identity, TOLERANCE));
    }

    #[test]
    fn test_sgal3_log_identity() {
        let identity = SGal3::identity();
        let tangent = identity.log(None);

        assert!(tangent.data.norm() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_adjoint() {
        let sgal3 = SGal3::random();
        let adj = sgal3.adjoint();

        assert_eq!(adj.nrows(), 10);
        assert_eq!(adj.ncols(), 10);
    }

    #[test]
    fn test_sgal3_act() {
        let sgal3 = SGal3::random();
        let point = Vector3::new(1.0, 2.0, 3.0);

        let _transformed_point = sgal3.act(&point, None, None);

        let identity = SGal3::identity();
        let identity_transformed = identity.act(&point, None, None);

        assert!((identity_transformed - point).norm() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_between() {
        let sgal3_a = SGal3::random();
        let sgal3_b = sgal3_a.clone();
        let sgal3_between_identity = sgal3_a.between(&sgal3_b, None, None);
        assert!(sgal3_between_identity.is_approx(&SGal3::identity(), TOLERANCE));

        let sgal3_c = SGal3::random();
        let sgal3_between = sgal3_a.between(&sgal3_c, None, None);
        let expected = sgal3_a.inverse(None).compose(&sgal3_c, None, None);
        assert!(sgal3_between.is_approx(&expected, TOLERANCE));
    }

    #[test]
    fn test_sgal3_tangent_zero() {
        let zero = SGal3Tangent::zero();
        assert!(zero.data.norm() < TOLERANCE);

        let tangent = SGal3Tangent::new(Vector3::zeros(), Vector3::zeros(), Vector3::zeros(), 0.0);
        assert!(tangent.is_zero(TOLERANCE));
    }

    #[test]
    fn test_sgal3_manifold_properties() {
        assert_eq!(SGal3::DIM, 3);
        assert_eq!(SGal3::DOF, 10);
        assert_eq!(SGal3::REP_SIZE, 11);
    }

    #[test]
    fn test_sgal3_consistency() {
        let sgal3_1 = SGal3::random();
        let sgal3_2 = SGal3::random();
        let sgal3_3 = SGal3::random();

        // Test associativity
        let left_assoc = sgal3_1
            .compose(&sgal3_2, None, None)
            .compose(&sgal3_3, None, None);
        let right_assoc = sgal3_1.compose(&sgal3_2.compose(&sgal3_3, None, None), None, None);

        assert!(left_assoc.is_approx(&right_assoc, 1e-10));
    }

    #[test]
    fn test_sgal3_tangent_small_adj() {
        let tangent = SGal3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.4, 0.5, 0.6),
            Vector3::new(0.7, 0.8, 0.9),
            0.5,
        );
        let small_adj = tangent.small_adj();

        assert_eq!(small_adj.nrows(), 10);
        assert_eq!(small_adj.ncols(), 10);
    }

    #[test]
    fn test_sgal3_tangent_lie_bracket() {
        let tangent_a = SGal3Tangent::new(
            Vector3::new(0.1, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.3),
            Vector3::new(0.0, 0.2, 0.0),
            0.1,
        );
        let tangent_b = SGal3Tangent::new(
            Vector3::new(0.0, 0.3, 0.0),
            Vector3::new(0.5, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.4),
            0.2,
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
    fn test_sgal3_tangent_is_approx() {
        let tangent_1 = SGal3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.4, 0.5, 0.6),
            Vector3::new(0.7, 0.8, 0.9),
            0.5,
        );
        let tangent_2 = SGal3Tangent::new(
            Vector3::new(0.1 + 1e-12, 0.2, 0.3),
            Vector3::new(0.4, 0.5, 0.6),
            Vector3::new(0.7, 0.8, 0.9),
            0.5,
        );
        let tangent_3 = SGal3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 5.0, 6.0),
            Vector3::new(7.0, 8.0, 9.0),
            1.0,
        );

        assert!(tangent_1.is_approx(&tangent_1, 1e-10));
        assert!(tangent_1.is_approx(&tangent_2, 1e-10));
        assert!(!tangent_1.is_approx(&tangent_3, 1e-10));
    }

    #[test]
    fn test_sgal3_generators() {
        let tangent = SGal3Tangent::new(
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            1.0,
        );

        for i in 0..10 {
            let generator = tangent.generator(i);
            assert_eq!(generator.nrows(), 6);
            assert_eq!(generator.ncols(), 6);
        }
    }

    #[test]
    #[should_panic]
    fn test_sgal3_generator_invalid_index() {
        let tangent = SGal3Tangent::new(
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            1.0,
        );
        let _generator = tangent.generator(10);
    }

    #[test]
    fn test_sgal3_vee() {
        let sgal3 = SGal3::random();
        let tangent_log = sgal3.log(None);
        let tangent_vee = sgal3.vee();

        assert!((tangent_log.data - tangent_vee.data).norm() < 1e-10);
    }

    #[test]
    fn test_sgal3_is_approx() {
        let sgal3_1 = SGal3::random();
        let sgal3_2 = sgal3_1.clone();

        assert!(sgal3_1.is_approx(&sgal3_1, 1e-10));
        assert!(sgal3_1.is_approx(&sgal3_2, 1e-10));

        let small_tangent = SGal3Tangent::new(
            Vector3::new(1e-12, 1e-12, 1e-12),
            Vector3::new(1e-12, 1e-12, 1e-12),
            Vector3::new(1e-12, 1e-12, 1e-12),
            1e-12,
        );
        let sgal3_perturbed = sgal3_1.right_plus(&small_tangent, None, None);
        assert!(sgal3_1.is_approx(&sgal3_perturbed, 1e-10));
    }

    #[test]
    fn test_sgal3_small_angle_approximations() {
        let small_tangent = SGal3Tangent::new(
            Vector3::new(1e-8, 2e-8, 3e-8),
            Vector3::new(4e-8, 5e-8, 6e-8),
            Vector3::new(1e-9, 2e-9, 3e-9),
            1e-8,
        );

        let sgal3 = small_tangent.exp(None);
        let recovered = sgal3.log(None);

        assert!((small_tangent.data - recovered.data).norm() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_accessors() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let velocity = Vector3::new(4.0, 5.0, 6.0);
        let rotation = UnitQuaternion::identity();
        let time = 0.5;

        let sgal3 = SGal3::new(translation, velocity, rotation, time);

        assert_eq!(sgal3.x(), 1.0);
        assert_eq!(sgal3.y(), 2.0);
        assert_eq!(sgal3.z(), 3.0);
        assert_eq!(sgal3.vx(), 4.0);
        assert_eq!(sgal3.vy(), 5.0);
        assert_eq!(sgal3.vz(), 6.0);
        assert_eq!(sgal3.time(), 0.5);
    }

    #[test]
    fn test_sgal3_tangent_ordering() {
        // Test that the tangent ordering is [ρ, ν, θ, s]
        let rho = Vector3::new(1.0, 2.0, 3.0);
        let nu = Vector3::new(4.0, 5.0, 6.0);
        let theta = Vector3::new(0.1, 0.2, 0.3);
        let s = 0.5;

        let tangent = SGal3Tangent::new(rho, nu, theta, s);

        assert_eq!(tangent.rho(), rho);
        assert_eq!(tangent.nu(), nu);
        assert_eq!(tangent.theta(), theta);
        assert_eq!(tangent.s(), s);

        // Check data ordering
        assert_eq!(tangent.data[0], 1.0); // ρ_x
        assert_eq!(tangent.data[1], 2.0); // ρ_y
        assert_eq!(tangent.data[2], 3.0); // ρ_z
        assert_eq!(tangent.data[3], 4.0); // ν_x
        assert_eq!(tangent.data[4], 5.0); // ν_y
        assert_eq!(tangent.data[5], 6.0); // ν_z
        assert_eq!(tangent.data[6], 0.1); // θ_x
        assert_eq!(tangent.data[7], 0.2); // θ_y
        assert_eq!(tangent.data[8], 0.3); // θ_z
        assert_eq!(tangent.data[9], 0.5); // s
    }
}
