//! Sim(3) - Similarity Transformations in 3D
//!
//! This module implements the Similarity group Sim(3), which represents
//! transformations including rotation, translation, and uniform scaling in 3D space.
//!
//! Sim(3) is the semi-direct product: (SO(3) × ℝ₊) ⋉ ℝ³
//!
//! Sim(3) elements are represented as (R, t, s) where:
//! - R ∈ SO(3): rotation
//! - t ∈ ℝ³: translation
//! - s ∈ ℝ₊: scale (positive real number)
//!
//! Sim(3) tangent elements are represented as [ρ(3), θ(3), σ(1)] = 7 components:
//! - ρ: translational component
//! - θ: rotational component (axis-angle)
//! - σ: logarithmic scale (log(s))
//!
//! The implementation follows conventions from:
//! - Ethan Eade's "Lie Groups for Computer Vision" (Section 6)
//! - Sophus library Sim3 implementation
//! - manif C++ library patterns
//!
//! # Use Cases
//! - Visual SLAM with scale ambiguity (monocular cameras)
//! - Structure from Motion
//! - 3D reconstruction with unknown scale
//!
//! # References
//! - Ethan Eade: "Lie Groups for Computer Vision" - https://www.ethaneade.com/lie.pdf
//! - Sophus library: sophus/sim3.hpp

use crate::{
    LieGroup, Tangent,
    so3::{SO3, SO3Tangent},
};
use nalgebra::{DVector, Matrix3, Matrix4, SMatrix, SVector, UnitQuaternion, Vector3};

// Type aliases for Sim(3) - 7 DOF
type Vector7<T> = SVector<T, 7>;
type Matrix7<T> = SMatrix<T, 7, 7>;
use std::{
    fmt,
    fmt::{Display, Formatter},
};

/// Sim(3) group element representing similarity transformations in 3D.
///
/// Represented as (rotation, translation, scale).
#[derive(Clone, PartialEq)]
pub struct Sim3 {
    /// Rotation part as SO(3) element
    rotation: SO3,
    /// Translation part as Vector3
    translation: Vector3<f64>,
    /// Scale factor (positive real number)
    scale: f64,
}

impl Display for Sim3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let t = self.translation();
        let s = self.scale();
        let q = self.rotation_quaternion();
        write!(
            f,
            "Sim3(translation: [{:.4}, {:.4}, {:.4}], scale: {:.4}, rotation: [w: {:.4}, x: {:.4}, y: {:.4}, z: {:.4}])",
            t.x, t.y, t.z, s, q.w, q.i, q.j, q.k
        )
    }
}

impl Sim3 {
    /// Space dimension - dimension of the ambient space
    pub const DIM: usize = 3;

    /// Degrees of freedom - dimension of the tangent space
    pub const DOF: usize = 7;

    /// Representation size - size of the underlying data representation
    pub const REP_SIZE: usize = 8;

    /// Get the identity element of the group.
    pub fn identity() -> Self {
        Sim3 {
            rotation: SO3::identity(),
            translation: Vector3::zeros(),
            scale: 1.0,
        }
    }

    /// Get the identity matrix for Jacobians.
    pub fn jacobian_identity() -> Matrix7<f64> {
        Matrix7::<f64>::identity()
    }

    /// Create a new Sim(3) element from translation, rotation, and scale.
    ///
    /// # Arguments
    /// * `translation` - Translation vector [x, y, z]
    /// * `rotation` - Unit quaternion representing rotation
    /// * `scale` - Scale factor (must be positive)
    pub fn new(translation: Vector3<f64>, rotation: UnitQuaternion<f64>, scale: f64) -> Self {
        assert!(scale > 0.0, "Scale must be positive");
        Sim3 {
            rotation: SO3::new(rotation),
            translation,
            scale,
        }
    }

    /// Create Sim(3) from components.
    pub fn from_components(translation: Vector3<f64>, rotation: SO3, scale: f64) -> Self {
        assert!(scale > 0.0, "Scale must be positive");
        Sim3 {
            rotation,
            translation,
            scale,
        }
    }

    /// Get the translation part as a Vector3.
    pub fn translation(&self) -> Vector3<f64> {
        self.translation
    }

    /// Get the scale factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Get the rotation part as SO3.
    pub fn rotation_so3(&self) -> SO3 {
        self.rotation.clone()
    }

    /// Get the rotation part as a UnitQuaternion.
    pub fn rotation_quaternion(&self) -> UnitQuaternion<f64> {
        self.rotation.quaternion()
    }

    /// Get the rotation matrix (3x3).
    pub fn rotation_matrix(&self) -> Matrix3<f64> {
        self.rotation.rotation_matrix()
    }

    /// Get the 4x4 homogeneous transformation matrix.
    pub fn matrix(&self) -> Matrix4<f64> {
        let mut mat = Matrix4::identity();
        let rot_mat = self.rotation_matrix();

        // Top-left 3x3: s*R
        for i in 0..3 {
            for j in 0..3 {
                mat[(i, j)] = self.scale * rot_mat[(i, j)];
            }
        }

        // Top-right 3x1: t
        mat[(0, 3)] = self.translation.x;
        mat[(1, 3)] = self.translation.y;
        mat[(2, 3)] = self.translation.z;

        mat
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

    /// Get the parameter vector [tx, ty, tz, qw, qx, qy, qz, s].
    pub fn coeffs(&self) -> [f64; 8] {
        let q = self.rotation.quaternion();
        [
            self.translation.x,
            self.translation.y,
            self.translation.z,
            q.w,
            q.i,
            q.j,
            q.k,
            self.scale,
        ]
    }
}

impl From<DVector<f64>> for Sim3 {
    fn from(data: DVector<f64>) -> Self {
        let translation = Vector3::new(data[0], data[1], data[2]);
        let rotation = SO3::from_quaternion_wxyz(data[3], data[4], data[5], data[6]);
        let scale = data[7];
        Sim3::from_components(translation, rotation, scale)
    }
}

impl From<Sim3> for DVector<f64> {
    fn from(sim3: Sim3) -> Self {
        let q = sim3.rotation.quaternion();
        DVector::from_vec(vec![
            sim3.translation.x,
            sim3.translation.y,
            sim3.translation.z,
            q.w,
            q.i,
            q.j,
            q.k,
            sim3.scale,
        ])
    }
}

impl LieGroup for Sim3 {
    type TangentVector = Sim3Tangent;
    type JacobianMatrix = Matrix7<f64>;
    type LieAlgebra = Matrix4<f64>;

    /// Get the inverse.
    ///
    /// For Sim(3): g^{-1} = (R^T, -R^T * t / s, 1/s)
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self {
        let rot_inv = self.rotation.inverse(None);
        let scale_inv = 1.0 / self.scale;
        let trans_inv = -rot_inv.act(&self.translation, None, None) * scale_inv;

        if let Some(jac) = jacobian {
            *jac = -self.adjoint();
        }

        Sim3::from_components(trans_inv, rot_inv, scale_inv)
    }

    /// Composition of this and another Sim(3) element.
    ///
    /// g1 ∘ g2 = (R1*R2, s1*R1*t2 + t1, s1*s2)
    fn compose(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        let composed_rotation = self.rotation.compose(&other.rotation, None, None);
        let composed_translation =
            self.scale * self.rotation.act(&other.translation, None, None) + self.translation;
        let composed_scale = self.scale * other.scale;

        let result = Sim3::from_components(composed_translation, composed_rotation, composed_scale);

        if let Some(jac_self) = jacobian_self {
            *jac_self = other.inverse(None).adjoint();
        }

        if let Some(jac_other) = jacobian_other {
            *jac_other = Matrix7::identity();
        }

        result
    }

    /// Logarithmic map from Sim(3) to its tangent space.
    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector {
        let theta = self.rotation.log(None);
        let sigma = self.scale.ln();
        let mut data = Vector7::zeros();

        // Compute the V^{-1} matrix for Sim(3)
        // V^{-1} = J_l^{-1}(θ) * (I - σ/2 * [θ]× + ...)
        let theta_tangent = SO3Tangent::new(theta.coeffs());
        let v_inv = Self::compute_v_inv(&theta_tangent, sigma);
        let translation_vector = v_inv * self.translation;

        data.fixed_rows_mut::<3>(0).copy_from(&translation_vector);
        data.fixed_rows_mut::<3>(3).copy_from(&theta.coeffs());
        data[6] = sigma;

        let result = Sim3Tangent { data };

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
        let result = self.scale * self.rotation.act(vector, None, None) + self.translation;

        if let Some(jac_self) = jacobian_self {
            let rotation_matrix = self.rotation.rotation_matrix();
            let rotated_vector = self.rotation.act(vector, None, None);

            // Jacobian w.r.t. translation
            jac_self
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&Matrix3::identity());

            // Jacobian w.r.t. rotation
            jac_self
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-self.scale * rotation_matrix * SO3Tangent::new(*vector).hat()));

            // Jacobian w.r.t. scale
            jac_self
                .fixed_view_mut::<3, 1>(0, 6)
                .copy_from(&rotated_vector);
        }

        if let Some(jac_vector) = jacobian_vector {
            *jac_vector = self.scale * self.rotation.rotation_matrix();
        }

        result
    }

    fn adjoint(&self) -> Self::JacobianMatrix {
        let rotation_matrix = self.rotation.rotation_matrix();
        let translation = self.translation;
        let scale = self.scale;
        let mut adjoint_matrix = Matrix7::zeros();

        // Block structure for Sim(3):
        // [sR   [t]×sR   0]
        // [0      R      0]
        // [0      0      1]

        // Top-left: s*R
        adjoint_matrix
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(scale * rotation_matrix));

        // Top-middle: [t]× * s*R
        let top_middle = SO3Tangent::new(translation).hat() * scale * rotation_matrix;
        adjoint_matrix
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&top_middle);

        // Middle-middle: R
        adjoint_matrix
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&rotation_matrix);

        // Bottom-right: 1
        adjoint_matrix[(6, 6)] = 1.0;

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

        let rotation = SO3::random();
        let scale = rng.random_range(0.5..2.0);

        Sim3::from_components(translation, rotation, scale)
    }

    fn normalize(&mut self) {
        self.rotation.normalize();
        if self.scale <= 0.0 {
            self.scale = 1.0;
        }
    }

    fn is_valid(&self, tolerance: f64) -> bool {
        self.rotation.is_valid(tolerance) && self.scale > 0.0
    }

    fn vee(&self) -> Self::TangentVector {
        self.log(None)
    }

    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        let difference = self.right_minus(other, None, None);
        difference.is_zero(tolerance)
    }

    fn jacobian_identity() -> Self::JacobianMatrix {
        Matrix7::<f64>::identity()
    }

    fn zero_jacobian() -> Self::JacobianMatrix {
        Matrix7::<f64>::zeros()
    }
}

impl Sim3 {
    /// Compute V^{-1} matrix for Sim(3) logarithm.
    ///
    /// Compute the inverse of the V matrix for Sim(3) logarithm.
    fn compute_v_inv(theta: &SO3Tangent, sigma: f64) -> Matrix3<f64> {
        let v = Sim3Tangent::v_matrix(theta, sigma);
        // SAFETY: V is non-singular for all valid Sim(3) elements (non-zero scale).
        // The identity fallback is only reached in degenerate near-zero-scale cases,
        // which are outside the valid domain of Sim(3).
        v.try_inverse().unwrap_or(Matrix3::identity())
    }
}

/// Sim(3) tangent space element.
///
/// Represented as [ρ(3), θ(3), σ(1)] where:
/// - ρ: translational component
/// - θ: rotational component (axis-angle)
/// - σ: logarithmic scale
#[derive(Clone, PartialEq)]
pub struct Sim3Tangent {
    /// Internal data: [ρ_x, ρ_y, ρ_z, θ_x, θ_y, θ_z, σ]
    data: Vector7<f64>,
}

impl fmt::Display for Sim3Tangent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rho = self.rho();
        let theta = self.theta();
        let sigma = self.sigma();
        write!(
            f,
            "sim3(rho: [{:.4}, {:.4}, {:.4}], theta: [{:.4}, {:.4}, {:.4}], sigma: {:.4})",
            rho.x, rho.y, rho.z, theta.x, theta.y, theta.z, sigma
        )
    }
}

impl From<DVector<f64>> for Sim3Tangent {
    fn from(data_vector: DVector<f64>) -> Self {
        Sim3Tangent {
            data: Vector7::from_iterator(data_vector.iter().copied()),
        }
    }
}

impl From<Sim3Tangent> for DVector<f64> {
    fn from(tangent: Sim3Tangent) -> Self {
        DVector::from_vec(tangent.data.as_slice().to_vec())
    }
}

impl Sim3Tangent {
    /// Create a new Sim(3)Tangent from components.
    pub fn new(rho: Vector3<f64>, theta: Vector3<f64>, sigma: f64) -> Self {
        let mut data = Vector7::zeros();
        data.fixed_rows_mut::<3>(0).copy_from(&rho);
        data.fixed_rows_mut::<3>(3).copy_from(&theta);
        data[6] = sigma;
        Sim3Tangent { data }
    }

    /// Get the ρ (translational) part.
    pub fn rho(&self) -> Vector3<f64> {
        self.data.fixed_rows::<3>(0).into_owned()
    }

    /// Get the θ (rotational) part.
    pub fn theta(&self) -> Vector3<f64> {
        self.data.fixed_rows::<3>(3).into_owned()
    }

    /// Get the σ (logarithmic scale) part.
    pub fn sigma(&self) -> f64 {
        self.data[6]
    }

    /// Create Sim3Tangent from individual scalar components.
    pub fn from_components(
        rho_x: f64,
        rho_y: f64,
        rho_z: f64,
        theta_x: f64,
        theta_y: f64,
        theta_z: f64,
        sigma: f64,
    ) -> Self {
        Sim3Tangent {
            data: Vector7::from_column_slice(&[
                rho_x, rho_y, rho_z, theta_x, theta_y, theta_z, sigma,
            ]),
        }
    }

    /// Compute the V matrix for Sim(3) exponential map.
    ///
    /// V = ∫₀¹ exp(σt) · exp(θt×) dt = A·I + B·[θ]× + C·[θ]×²
    ///
    /// Based on Ethan Eade's "Lie Groups for Computer Vision" formulation.
    fn v_matrix(theta: &SO3Tangent, sigma: f64) -> Matrix3<f64> {
        let theta_norm_sq = theta.coeffs().norm_squared();

        if theta_norm_sq < crate::SMALL_ANGLE_THRESHOLD
            && sigma.abs() < crate::SMALL_ANGLE_THRESHOLD
        {
            return Matrix3::identity();
        }

        let theta_hat = theta.hat();

        if theta_norm_sq < crate::SMALL_ANGLE_THRESHOLD {
            // Pure scale, no rotation: V = (e^σ - 1)/σ · I
            let a = (sigma.exp() - 1.0) / sigma;
            return a * Matrix3::identity();
        }

        let theta_norm = theta_norm_sq.sqrt();
        let sin_theta = theta_norm.sin();
        let cos_theta = theta_norm.cos();

        if sigma.abs() < crate::SMALL_ANGLE_THRESHOLD {
            // Pure rotation, no scale: V = SO(3) left Jacobian
            let a = 1.0;
            let b = (1.0 - cos_theta) / theta_norm_sq;
            let c = (theta_norm - sin_theta) / (theta_norm * theta_norm_sq);
            return a * Matrix3::identity() + b * theta_hat + c * theta_hat * theta_hat;
        }

        // General case: both sigma and theta nonzero
        let e_sigma = sigma.exp();
        let alpha_sq = sigma * sigma + theta_norm_sq;

        let a = (e_sigma - 1.0) / sigma;
        let b = (e_sigma * (sigma * sin_theta - theta_norm * cos_theta) + theta_norm)
            / (theta_norm * alpha_sq);
        let cos_integral =
            (e_sigma * (sigma * cos_theta + theta_norm * sin_theta) - sigma) / alpha_sq;
        let c = (a - cos_integral) / theta_norm_sq;

        a * Matrix3::identity() + b * theta_hat + c * theta_hat * theta_hat
    }

    /// Compute the Q matrix for Sim(3) Jacobians.
    fn q_matrix(rho: Vector3<f64>, theta: Vector3<f64>, sigma: f64) -> Matrix3<f64> {
        let rho_skew = SO3Tangent::new(rho).hat();
        let theta_skew = SO3Tangent::new(theta).hat();
        let theta_squared = theta.norm_squared();

        if theta_squared < crate::SMALL_ANGLE_THRESHOLD && sigma.abs() < f64::EPSILON {
            return 0.5 * rho_skew;
        }

        let a = 0.5;
        let mut b = 1.0 / 6.0;
        let mut c = -1.0 / 24.0;
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

impl Tangent<Sim3> for Sim3Tangent {
    const DIM: usize = 7;

    /// Exponential map to Sim(3).
    fn exp(&self, jacobian: Option<&mut <Sim3 as LieGroup>::JacobianMatrix>) -> Sim3 {
        let rho = self.rho();
        let theta = self.theta();
        let sigma = self.sigma();

        let theta_tangent = SO3Tangent::new(theta);
        let rotation = theta_tangent.exp(None);
        let v_matrix = Self::v_matrix(&theta_tangent, sigma);
        let translation = v_matrix * rho;
        let scale = sigma.exp();

        if let Some(jac) = jacobian {
            *jac = self.right_jacobian();
        }

        Sim3::from_components(translation, rotation, scale)
    }

    /// Right Jacobian for Sim(3).
    fn right_jacobian(&self) -> <Sim3 as LieGroup>::JacobianMatrix {
        let mut jac = Matrix7::zeros();
        let rho = self.rho();
        let theta = self.theta();
        let sigma = self.sigma();

        let theta_right_jac = SO3Tangent::new(-theta).right_jacobian();
        let q_block = Self::q_matrix(-rho, -theta, -sigma);

        // Block structure for Sim(3)
        jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&theta_right_jac);
        jac.fixed_view_mut::<3, 3>(3, 3).copy_from(&theta_right_jac);
        jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&q_block);

        // Scale part
        jac[(6, 6)] = 1.0;

        // Coupling between translation and scale
        let v_deriv = Self::v_matrix(&SO3Tangent::new(-theta), -sigma);
        jac.fixed_view_mut::<3, 1>(0, 6)
            .copy_from(&(v_deriv * (-rho)));

        jac
    }

    /// Left Jacobian for Sim(3).
    fn left_jacobian(&self) -> <Sim3 as LieGroup>::JacobianMatrix {
        let mut jac = Matrix7::zeros();
        let rho = self.rho();
        let theta = self.theta();
        let sigma = self.sigma();

        let theta_left_jac = SO3Tangent::new(theta).left_jacobian();
        let q_block = Self::q_matrix(rho, theta, sigma);

        jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&theta_left_jac);
        jac.fixed_view_mut::<3, 3>(3, 3).copy_from(&theta_left_jac);
        jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&q_block);

        jac[(6, 6)] = 1.0;

        let v_deriv = Self::v_matrix(&SO3Tangent::new(theta), sigma);
        jac.fixed_view_mut::<3, 1>(0, 6).copy_from(&(v_deriv * rho));

        jac
    }

    /// Inverse of right Jacobian.
    fn right_jacobian_inv(&self) -> <Sim3 as LieGroup>::JacobianMatrix {
        // SAFETY: The right Jacobian is non-singular for valid Sim(3) tangent elements.
        // Identity fallback only occurs at degenerate inputs outside the valid domain.
        self.right_jacobian()
            .try_inverse()
            .unwrap_or(Matrix7::identity())
    }

    /// Inverse of left Jacobian.
    fn left_jacobian_inv(&self) -> <Sim3 as LieGroup>::JacobianMatrix {
        // SAFETY: The left Jacobian is non-singular for valid Sim(3) tangent elements.
        // Identity fallback only occurs at degenerate inputs outside the valid domain.
        self.left_jacobian()
            .try_inverse()
            .unwrap_or(Matrix7::identity())
    }

    /// Hat operator: maps tangent vector to Lie algebra matrix (4x4).
    fn hat(&self) -> <Sim3 as LieGroup>::LieAlgebra {
        let mut lie_alg = Matrix4::zeros();

        let theta_hat = SO3Tangent::new(self.theta()).hat();

        // Top-left 3x3: [θ]× + σ*I
        for i in 0..3 {
            for j in 0..3 {
                lie_alg[(i, j)] = theta_hat[(i, j)];
            }
            lie_alg[(i, i)] += self.sigma();
        }

        // Top-right 3x1: ρ
        let rho = self.rho();
        lie_alg[(0, 3)] = rho[0];
        lie_alg[(1, 3)] = rho[1];
        lie_alg[(2, 3)] = rho[2];

        lie_alg
    }

    fn zero() -> <Sim3 as LieGroup>::TangentVector {
        Sim3Tangent::new(Vector3::zeros(), Vector3::zeros(), 0.0)
    }

    fn random() -> <Sim3 as LieGroup>::TangentVector {
        use rand::Rng;
        let mut rng = rand::rng();
        Sim3Tangent::new(
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
            rng.random_range(-0.5..0.5),
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

    fn normalized(&self) -> <Sim3 as LieGroup>::TangentVector {
        let norm = self.theta().norm();
        if norm > f64::EPSILON {
            Sim3Tangent::new(self.rho(), self.theta() / norm, self.sigma())
        } else {
            Sim3Tangent::new(self.rho(), Vector3::zeros(), self.sigma())
        }
    }

    fn small_adj(&self) -> <Sim3 as LieGroup>::JacobianMatrix {
        let mut small_adj = Matrix7::zeros();
        let rho_skew = SO3Tangent::new(self.rho()).hat();
        let theta_skew = SO3Tangent::new(self.theta()).hat();
        let sigma = self.sigma();

        // Block structure for Sim(3):
        // [θ× + σ*I   ρ×   ρ]
        // [   0       θ×   0]
        // [   0       0    0]

        for i in 0..3 {
            for j in 0..3 {
                small_adj[(i, j)] = theta_skew[(i, j)];
            }
            small_adj[(i, i)] += sigma;
        }

        small_adj.fixed_view_mut::<3, 3>(0, 3).copy_from(&rho_skew);
        small_adj
            .fixed_view_mut::<3, 1>(0, 6)
            .copy_from(&(-self.rho()));
        small_adj
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&theta_skew);

        small_adj
    }

    fn lie_bracket(&self, other: &Self) -> <Sim3 as LieGroup>::TangentVector {
        let bracket_result = self.small_adj() * other.data;
        Sim3Tangent {
            data: bracket_result,
        }
    }

    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        (self.data - other.data).norm() < tolerance
    }

    fn generator(&self, i: usize) -> <Sim3 as LieGroup>::LieAlgebra {
        assert!(i < 7, "Sim(3) only has generators for indices 0-6");

        let mut generator = Matrix4::zeros();

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
            6 => {
                // Scale generator (sigma)
                generator[(0, 0)] = 1.0;
                generator[(1, 1)] = 1.0;
                generator[(2, 2)] = 1.0;
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
    fn test_sim3_identity() {
        let identity = Sim3::identity();
        assert!(identity.is_valid(TOLERANCE));
        assert!(identity.translation().norm() < TOLERANCE);
        assert!((identity.scale() - 1.0).abs() < TOLERANCE);
        assert!(identity.rotation_quaternion().angle() < TOLERANCE);
    }

    #[test]
    fn test_sim3_new() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let rotation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let scale = 1.5;

        let sim3 = Sim3::new(translation, rotation, scale);
        assert!(sim3.is_valid(TOLERANCE));
        assert!((sim3.translation() - translation).norm() < TOLERANCE);
        assert!((sim3.scale() - scale).abs() < TOLERANCE);
    }

    #[test]
    #[should_panic(expected = "Scale must be positive")]
    fn test_sim3_new_negative_scale() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let rotation = UnitQuaternion::identity();
        let _sim3 = Sim3::new(translation, rotation, -1.0);
    }

    #[test]
    fn test_sim3_random() {
        let sim3 = Sim3::random();
        assert!(sim3.is_valid(TOLERANCE));
        assert!(sim3.scale() > 0.0);
    }

    #[test]
    fn test_sim3_inverse() {
        let sim3 = Sim3::random();
        let sim3_inv = sim3.inverse(None);

        let composed = sim3.compose(&sim3_inv, None, None);
        let identity = Sim3::identity();

        assert!(composed.is_approx(&identity, TOLERANCE));
    }

    #[test]
    fn test_sim3_compose() {
        let sim3_1 = Sim3::random();
        let sim3_2 = Sim3::random();

        let composed = sim3_1.compose(&sim3_2, None, None);
        assert!(composed.is_valid(TOLERANCE));

        let identity = Sim3::identity();
        let composed_with_identity = sim3_1.compose(&identity, None, None);
        assert!(composed_with_identity.is_approx(&sim3_1, TOLERANCE));
    }

    #[test]
    fn test_sim3_exp_log() {
        let tangent = Sim3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.01, 0.02, 0.03),
            0.1,
        );

        let sim3 = tangent.exp(None);
        let recovered_tangent = sim3.log(None);

        assert!((tangent.data - recovered_tangent.data).norm() < TOLERANCE);
    }

    #[test]
    fn test_sim3_exp_zero() {
        let zero_tangent = Sim3Tangent::zero();
        let sim3 = zero_tangent.exp(None);
        let identity = Sim3::identity();

        assert!(sim3.is_approx(&identity, TOLERANCE));
    }

    #[test]
    fn test_sim3_log_identity() {
        let identity = Sim3::identity();
        let tangent = identity.log(None);

        assert!(tangent.data.norm() < TOLERANCE);
    }

    #[test]
    fn test_sim3_adjoint() {
        let sim3 = Sim3::random();
        let adj = sim3.adjoint();

        assert_eq!(adj.nrows(), 7);
        assert_eq!(adj.ncols(), 7);
    }

    #[test]
    fn test_sim3_act() {
        let sim3 = Sim3::random();
        let point = Vector3::new(1.0, 2.0, 3.0);

        let _transformed_point = sim3.act(&point, None, None);

        let identity = Sim3::identity();
        let identity_transformed = identity.act(&point, None, None);

        assert!((identity_transformed - point).norm() < TOLERANCE);
    }

    #[test]
    fn test_sim3_between() {
        let sim3_a = Sim3::random();
        let sim3_b = sim3_a.clone();
        let sim3_between_identity = sim3_a.between(&sim3_b, None, None);
        assert!(sim3_between_identity.is_approx(&Sim3::identity(), TOLERANCE));

        let sim3_c = Sim3::random();
        let sim3_between = sim3_a.between(&sim3_c, None, None);
        let expected = sim3_a.inverse(None).compose(&sim3_c, None, None);
        assert!(sim3_between.is_approx(&expected, TOLERANCE));
    }

    #[test]
    fn test_sim3_tangent_zero() {
        let zero = Sim3Tangent::zero();
        assert!(zero.data.norm() < TOLERANCE);

        let tangent = Sim3Tangent::new(Vector3::zeros(), Vector3::zeros(), 0.0);
        assert!(tangent.is_zero(TOLERANCE));
    }

    #[test]
    fn test_sim3_manifold_properties() {
        assert_eq!(Sim3::DIM, 3);
        assert_eq!(Sim3::DOF, 7);
        assert_eq!(Sim3::REP_SIZE, 8);
    }

    #[test]
    fn test_sim3_consistency() {
        let sim3_1 = Sim3::random();
        let sim3_2 = Sim3::random();
        let sim3_3 = Sim3::random();

        // Test associativity
        let left_assoc = sim3_1
            .compose(&sim3_2, None, None)
            .compose(&sim3_3, None, None);
        let right_assoc = sim3_1.compose(&sim3_2.compose(&sim3_3, None, None), None, None);

        assert!(left_assoc.is_approx(&right_assoc, 1e-10));
    }

    #[test]
    fn test_sim3_scale_composition() {
        let sim3_1 = Sim3::new(Vector3::zeros(), UnitQuaternion::identity(), 2.0);
        let sim3_2 = Sim3::new(Vector3::zeros(), UnitQuaternion::identity(), 3.0);

        let composed = sim3_1.compose(&sim3_2, None, None);
        assert!((composed.scale() - 6.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_sim3_tangent_small_adj() {
        let tangent = Sim3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.4, 0.5, 0.6),
            0.1,
        );
        let small_adj = tangent.small_adj();

        assert_eq!(small_adj.nrows(), 7);
        assert_eq!(small_adj.ncols(), 7);
    }

    #[test]
    fn test_sim3_tangent_lie_bracket() {
        let tangent_a = Sim3Tangent::new(
            Vector3::new(0.1, 0.0, 0.0),
            Vector3::new(0.0, 0.2, 0.0),
            0.1,
        );
        let tangent_b = Sim3Tangent::new(
            Vector3::new(0.0, 0.3, 0.0),
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
    fn test_sim3_tangent_is_approx() {
        let tangent_1 = Sim3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.4, 0.5, 0.6),
            0.1,
        );
        let tangent_2 = Sim3Tangent::new(
            Vector3::new(0.1 + 1e-12, 0.2, 0.3),
            Vector3::new(0.4, 0.5, 0.6),
            0.1,
        );
        let tangent_3 = Sim3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 5.0, 6.0),
            1.0,
        );

        assert!(tangent_1.is_approx(&tangent_1, 1e-10));
        assert!(tangent_1.is_approx(&tangent_2, 1e-10));
        assert!(!tangent_1.is_approx(&tangent_3, 1e-10));
    }

    #[test]
    fn test_sim3_generators() {
        let tangent = Sim3Tangent::new(
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            1.0,
        );

        for i in 0..7 {
            let generator = tangent.generator(i);
            assert_eq!(generator.nrows(), 4);
            assert_eq!(generator.ncols(), 4);

            // Bottom row should be zeros
            assert_eq!(generator[(3, 0)], 0.0);
            assert_eq!(generator[(3, 1)], 0.0);
            assert_eq!(generator[(3, 2)], 0.0);
            assert_eq!(generator[(3, 3)], 0.0);
        }
    }

    #[test]
    #[should_panic]
    fn test_sim3_generator_invalid_index() {
        let tangent = Sim3Tangent::new(
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            1.0,
        );
        let _generator = tangent.generator(7);
    }

    #[test]
    fn test_sim3_vee() {
        let sim3 = Sim3::random();
        let tangent_log = sim3.log(None);
        let tangent_vee = sim3.vee();

        assert!((tangent_log.data - tangent_vee.data).norm() < 1e-10);
    }

    #[test]
    fn test_sim3_is_approx() {
        let sim3_1 = Sim3::random();
        let sim3_2 = sim3_1.clone();

        assert!(sim3_1.is_approx(&sim3_1, 1e-10));
        assert!(sim3_1.is_approx(&sim3_2, 1e-10));

        let small_tangent = Sim3Tangent::new(
            Vector3::new(1e-12, 1e-12, 1e-12),
            Vector3::new(1e-12, 1e-12, 1e-12),
            1e-12,
        );
        let sim3_perturbed = sim3_1.right_plus(&small_tangent, None, None);
        assert!(sim3_1.is_approx(&sim3_perturbed, 1e-10));
    }

    #[test]
    fn test_sim3_small_angle_approximations() {
        let small_tangent = Sim3Tangent::new(
            Vector3::new(1e-8, 2e-8, 3e-8),
            Vector3::new(1e-9, 2e-9, 3e-9),
            1e-8,
        );

        let sim3 = small_tangent.exp(None);
        let recovered = sim3.log(None);

        assert!((small_tangent.data - recovered.data).norm() < TOLERANCE);
    }

    #[test]
    fn test_sim3_accessors() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let rotation = UnitQuaternion::identity();
        let scale = 1.5;

        let sim3 = Sim3::new(translation, rotation, scale);

        assert_eq!(sim3.x(), 1.0);
        assert_eq!(sim3.y(), 2.0);
        assert_eq!(sim3.z(), 3.0);
        assert_eq!(sim3.scale(), 1.5);
    }

    #[test]
    fn test_sim3_matrix() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let rotation = UnitQuaternion::identity();
        let scale = 2.0;

        let sim3 = Sim3::new(translation, rotation, scale);
        let mat = sim3.matrix();

        // Check that top-left 3x3 is scaled rotation (2*I in this case)
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((mat[(i, j)] - 2.0).abs() < TOLERANCE);
                } else {
                    assert!(mat[(i, j)].abs() < TOLERANCE);
                }
            }
        }

        // Check translation
        assert!((mat[(0, 3)] - 1.0).abs() < TOLERANCE);
        assert!((mat[(1, 3)] - 2.0).abs() < TOLERANCE);
        assert!((mat[(2, 3)] - 3.0).abs() < TOLERANCE);

        // Check bottom row
        assert!(mat[(3, 0)].abs() < TOLERANCE);
        assert!(mat[(3, 1)].abs() < TOLERANCE);
        assert!(mat[(3, 2)].abs() < TOLERANCE);
        assert!((mat[(3, 3)] - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_sim3_scale_action() {
        let scale = 2.0;
        let sim3 = Sim3::new(Vector3::zeros(), UnitQuaternion::identity(), scale);
        let point = Vector3::new(1.0, 2.0, 3.0);

        let transformed = sim3.act(&point, None, None);

        assert!((transformed.x - 2.0).abs() < TOLERANCE);
        assert!((transformed.y - 4.0).abs() < TOLERANCE);
        assert!((transformed.z - 6.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_sim3_tangent_basic() {
        let rho = Vector3::new(0.1, 0.2, 0.3);
        let theta = Vector3::new(0.4, 0.5, 0.6);
        let sigma = 0.1;
        let tangent = Sim3Tangent::new(rho, theta, sigma);
        assert!((tangent.rho() - rho).norm() < TOLERANCE);
        assert!((tangent.theta() - theta).norm() < TOLERANCE);
        assert!((tangent.sigma() - sigma).abs() < TOLERANCE);
    }

    #[test]
    fn test_sim3_tangent_from_components() {
        let t1 = Sim3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(0.4, 0.5, 0.6),
            0.1,
        );
        let t2 = Sim3Tangent::from_components(1.0, 2.0, 3.0, 0.4, 0.5, 0.6, 0.1);
        assert!(t1.is_approx(&t2, TOLERANCE));
    }

    #[test]
    fn test_sim3_normalize() {
        let mut sim3 = Sim3::random();
        sim3.normalize();
        assert!(sim3.is_valid(TOLERANCE));
    }

    #[test]
    fn test_sim3_coeffs() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let rotation = UnitQuaternion::identity();
        let scale = 1.5;
        let sim3 = Sim3::new(translation, rotation, scale);
        let c = sim3.coeffs();
        assert!((c[0] - 1.0).abs() < TOLERANCE);
        assert!((c[1] - 2.0).abs() < TOLERANCE);
        assert!((c[2] - 3.0).abs() < TOLERANCE);
        assert!((c[3] - 1.0).abs() < TOLERANCE); // qw
        assert!((c[7] - 1.5).abs() < TOLERANCE); // scale
    }

    #[test]
    fn test_sim3_manif_like_operations() {
        let a = Sim3::random();
        let b = Sim3::random();

        let a_to_b = a.between(&b, None, None);
        let recovered_b = a.compose(&a_to_b, None, None);
        assert!(recovered_b.is_approx(&b, TOLERANCE));

        let tangent = Sim3Tangent::new(
            Vector3::new(0.01, 0.02, 0.03),
            Vector3::new(0.001, 0.002, 0.003),
            0.01,
        );
        let perturbed = a.plus(&tangent, None, None);
        let recovered_tangent = perturbed.minus(&a, None, None);
        assert!((tangent.data - recovered_tangent.data).norm() < 1e-6);
    }

    #[test]
    fn test_sim3_right_jacobian_inverse_identity() {
        let tangent = Sim3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.01, 0.02, 0.03),
            0.1,
        );
        let jr = tangent.right_jacobian();
        let jr_inv = tangent.right_jacobian_inv();
        let product = jr * jr_inv;
        let identity = Matrix7::<f64>::identity();
        assert!((product - identity).norm() < 0.01);
    }

    #[test]
    fn test_sim3_left_jacobian_inverse_identity() {
        let tangent = Sim3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.01, 0.02, 0.03),
            0.1,
        );
        let jl = tangent.left_jacobian();
        let jl_inv = tangent.left_jacobian_inv();
        let product = jl * jl_inv;
        let identity = Matrix7::<f64>::identity();
        assert!((product - identity).norm() < 0.01);
    }

    #[test]
    fn test_sim3_jacobi_identity() {
        let a = Sim3Tangent::random();
        let b = Sim3Tangent::random();
        let c = Sim3Tangent::random();

        let term1 = a.lie_bracket(&b.lie_bracket(&c));
        let term2 = b.lie_bracket(&c.lie_bracket(&a));
        let term3 = c.lie_bracket(&a.lie_bracket(&b));

        assert!((term1.data + term2.data + term3.data).norm() < 1e-8);
    }

    #[test]
    fn test_sim3_hat_matrix_structure() {
        let tangent = Sim3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(0.1, 0.2, 0.3),
            0.5,
        );
        let hat = tangent.hat();

        for j in 0..4 {
            assert_eq!(hat[(3, j)], 0.0);
        }

        assert!((hat[(0, 3)] - 1.0).abs() < TOLERANCE);
        assert!((hat[(1, 3)] - 2.0).abs() < TOLERANCE);
        assert!((hat[(2, 3)] - 3.0).abs() < TOLERANCE);

        // Diagonal includes sigma
        assert!((hat[(0, 0)] - 0.5).abs() < 0.5); // theta_skew + sigma
    }

    #[test]
    fn test_sim3_dvector_round_trip() {
        let sim3 = Sim3::random();
        let dvec: DVector<f64> = sim3.clone().into();
        let recovered: Sim3 = dvec.into();
        assert!(sim3.is_approx(&recovered, TOLERANCE));
    }

    // --- Additional coverage tests ---

    #[test]
    fn test_sim3_display() {
        let sim3 = Sim3::identity();
        let s = format!("{}", sim3);
        assert!(s.contains("Sim3"));

        let tangent = Sim3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(0.1, 0.2, 0.3),
            0.5,
        );
        let ts = format!("{}", tangent);
        assert!(ts.contains("sim3"));
    }

    #[test]
    fn test_sim3_jacobian_identity_static() {
        let jac = Sim3::jacobian_identity();
        assert!((jac - Matrix7::identity()).norm() < TOLERANCE);
    }

    #[test]
    fn test_sim3_rotation_so3() {
        let sim3 = Sim3::random();
        let so3 = sim3.rotation_so3();
        assert!(so3.is_valid(TOLERANCE));
    }

    #[test]
    fn test_sim3_inverse_with_jacobian() {
        let sim3 = Sim3::random();
        let mut jac = Matrix7::zeros();
        let inv = sim3.inverse(Some(&mut jac));
        let expected_jac = -sim3.adjoint();
        assert!((jac - expected_jac).norm() < TOLERANCE);
        let composed = sim3.compose(&inv, None, None);
        assert!(composed.is_approx(&Sim3::identity(), TOLERANCE));
    }

    #[test]
    fn test_sim3_compose_with_jacobians() {
        let a = Sim3::random();
        let b = Sim3::random();
        let mut jac_a = Matrix7::zeros();
        let mut jac_b = Matrix7::zeros();
        let _composed = a.compose(&b, Some(&mut jac_a), Some(&mut jac_b));
        assert!(jac_a.norm() > 0.0);
        assert!(jac_b.norm() > 0.0);
    }

    #[test]
    fn test_sim3_act_with_jacobians() {
        let sim3 = Sim3::random();
        let point = Vector3::new(1.0, 2.0, 3.0);
        let mut jac_self = Matrix7::zeros();
        let mut jac_point = Matrix3::<f64>::zeros();
        let result = sim3.act(&point, Some(&mut jac_self), Some(&mut jac_point));
        assert!(result.norm() > 0.0);
        assert!(jac_self.norm() > 0.0);
        assert!(jac_point.norm() > 0.0);
    }

    #[test]
    fn test_sim3_liegroup_jacobian_identity() {
        let jac = <Sim3 as LieGroup>::jacobian_identity();
        assert!((jac - Matrix7::identity()).norm() < TOLERANCE);

        let zero = <Sim3 as LieGroup>::zero_jacobian();
        assert!(zero.norm() < TOLERANCE);
    }

    #[test]
    fn test_sim3_tangent_dvector_conversions() {
        let tangent = Sim3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(0.1, 0.2, 0.3),
            0.5,
        );
        let dvec: DVector<f64> = tangent.clone().into();
        let recovered = Sim3Tangent::from(dvec);
        assert!(tangent.is_approx(&recovered, TOLERANCE));
    }

    #[test]
    fn test_sim3_exp_with_jacobian() {
        let tangent = Sim3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.01, 0.02, 0.03),
            0.1,
        );
        let mut jac = Matrix7::zeros();
        let _result = tangent.exp(Some(&mut jac));
        assert!(jac.norm() > 0.0);
    }

    #[test]
    fn test_sim3_exp_log_stress() {
        for _ in 0..100 {
            let tangent = Sim3Tangent::random();
            let sim3 = tangent.exp(None);
            let recovered = sim3.log(None);
            assert!(
                tangent.is_approx(&recovered, 1e-6),
                "exp/log round-trip failed: error = {}",
                (tangent.data - recovered.data).norm()
            );
        }
    }

    #[test]
    fn test_sim3_right_plus_minus_round_trip() {
        let a = Sim3::random();
        let b = Sim3::random();
        let diff = a.right_minus(&b, None, None);
        let recovered = b.right_plus(&diff, None, None);
        assert!(a.is_approx(&recovered, 1e-6));
    }

    #[test]
    fn test_sim3_left_plus_minus_round_trip() {
        let a = Sim3::random();
        let b = Sim3::random();
        let diff = a.left_minus(&b, None, None);
        let recovered = b.left_plus(&diff, None, None);
        assert!(a.is_approx(&recovered, 1e-6));
    }

    #[test]
    fn test_sim3_compose_associativity() {
        let a = Sim3::random();
        let b = Sim3::random();
        let c = Sim3::random();
        let ab_c = a.compose(&b, None, None).compose(&c, None, None);
        let a_bc = a.compose(&b.compose(&c, None, None), None, None);
        assert!(ab_c.is_approx(&a_bc, 1e-6));
    }

    #[test]
    fn test_sim3_inverse_twice() {
        let g = Sim3::random();
        let g_inv_inv = g.inverse(None).inverse(None);
        assert!(g.is_approx(&g_inv_inv, 1e-6));
    }

    #[test]
    fn test_sim3_pure_scale() {
        // exp/log with only sigma nonzero (theta=0)
        let tangent = Sim3Tangent::new(Vector3::zeros(), Vector3::zeros(), 0.5);
        let sim3 = tangent.exp(None);
        let recovered = sim3.log(None);
        assert!(tangent.is_approx(&recovered, TOLERANCE));
        assert!((sim3.scale() - 0.5_f64.exp()).abs() < TOLERANCE);
    }

    #[test]
    fn test_sim3_pure_rotation() {
        // exp/log with only theta nonzero (sigma=0)
        let tangent = Sim3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.05, 0.1, 0.15),
            0.0,
        );
        let sim3 = tangent.exp(None);
        let recovered = sim3.log(None);
        assert!(tangent.is_approx(&recovered, 1e-6));
        assert!((sim3.scale() - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_sim3_scale_compose() {
        let a = Sim3::from_components(Vector3::zeros(), SO3::identity(), 2.0);
        let b = Sim3::from_components(Vector3::zeros(), SO3::identity(), 3.0);
        let ab = a.compose(&b, None, None);
        assert!((ab.scale() - 6.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_sim3_tangent_normalize_zero_theta() {
        let tangent = Sim3Tangent::new(Vector3::new(1.0, 2.0, 3.0), Vector3::zeros(), 0.5);
        let normalized = tangent.normalized();
        assert!(normalized.theta().norm() < TOLERANCE);
    }

    #[test]
    fn test_sim3_tangent_generator_all() {
        let tangent = Sim3Tangent::zero();
        for i in 0..7 {
            let g = tangent.generator(i);
            assert!(g.norm() > 0.0, "Generator {} should be non-zero", i);
        }
    }

    #[test]
    fn test_sim3_v_matrix_sigma_zero_matches_so3_jl() {
        // When sigma=0, V-matrix should equal SO3 left Jacobian
        let theta = SO3Tangent::new(Vector3::new(0.3, 0.4, 0.5));
        let v = Sim3Tangent::v_matrix(&theta, 0.0);
        let jl = theta.left_jacobian();
        assert!(
            (v - jl).norm() < 1e-10,
            "V(θ,0) should equal Jl(θ), error = {}",
            (v - jl).norm()
        );
    }

    #[test]
    fn test_sim3_v_matrix_theta_zero() {
        // When theta=0, V-matrix should be (e^σ - 1)/σ * I
        let theta = SO3Tangent::new(Vector3::zeros());
        let sigma = 0.5;
        let v = Sim3Tangent::v_matrix(&theta, sigma);
        let expected = (sigma.exp() - 1.0) / sigma * Matrix3::<f64>::identity();
        assert!(
            (v - expected).norm() < 1e-10,
            "V(0,σ) should equal (e^σ-1)/σ * I, error = {}",
            (v - expected).norm()
        );
    }
}
