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

use crate::{
    LieGroup, Tangent,
    so3::{SO3, SO3Tangent},
};
use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
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
/// Stored as a flat `SVector<f64, 11>` = [tx, ty, tz, qw, qx, qy, qz, vx, vy, vz, time].
#[derive(Clone, PartialEq)]
pub struct SGal3 {
    /// Flat parameter storage: [tx, ty, tz, qw, qx, qy, qz, vx, vy, vz, time]
    params: SVector<f64, 11>,
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

    #[inline]
    fn translation_impl(&self) -> Vector3<f64> {
        Vector3::new(self.params[0], self.params[1], self.params[2])
    }

    #[inline]
    fn velocity_impl(&self) -> Vector3<f64> {
        Vector3::new(self.params[7], self.params[8], self.params[9])
    }

    #[inline]
    fn time_impl(&self) -> f64 {
        self.params[10]
    }

    #[inline]
    fn rotation_impl(&self) -> SO3 {
        SO3::from_quaternion_wxyz(
            self.params[3],
            self.params[4],
            self.params[5],
            self.params[6],
        )
    }

    #[inline]
    fn from_parts(t: Vector3<f64>, v: Vector3<f64>, time: f64, r: &SO3) -> Self {
        let q = r.params();
        SGal3 {
            params: SVector::<f64, 11>::from([
                t.x, t.y, t.z, q[0], q[1], q[2], q[3], v.x, v.y, v.z, time,
            ]),
        }
    }

    /// Get the identity element of the group.
    pub fn identity() -> Self {
        SGal3 {
            params: SVector::<f64, 11>::from([
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]),
        }
    }

    /// Get the identity matrix for Jacobians.
    pub fn jacobian_identity() -> Matrix10<f64> {
        Matrix10::<f64>::identity()
    }

    /// Create a new SGal(3) element from components.
    pub fn new(
        translation: Vector3<f64>,
        velocity: Vector3<f64>,
        rotation: UnitQuaternion<f64>,
        time: f64,
    ) -> Self {
        SGal3::from_parts(translation, velocity, time, &SO3::new(rotation))
    }

    /// Create SGal(3) from components.
    pub fn from_components(
        translation: Vector3<f64>,
        velocity: Vector3<f64>,
        rotation: SO3,
        time: f64,
    ) -> Self {
        SGal3::from_parts(translation, velocity, time, &rotation)
    }

    /// Get the translation part as a Vector3.
    pub fn translation(&self) -> Vector3<f64> {
        self.translation_impl()
    }
    /// Get the velocity part as a Vector3.
    pub fn velocity(&self) -> Vector3<f64> {
        self.velocity_impl()
    }
    /// Get the time parameter.
    pub fn time(&self) -> f64 {
        self.time_impl()
    }
    /// Get the rotation part as SO3.
    pub fn rotation_so3(&self) -> SO3 {
        self.rotation_impl()
    }
    /// Get the rotation part as a UnitQuaternion.
    pub fn rotation_quaternion(&self) -> UnitQuaternion<f64> {
        self.rotation_impl().quaternion()
    }

    /// Get the x component of translation.
    pub fn x(&self) -> f64 {
        self.params[0]
    }
    /// Get the y component of translation.
    pub fn y(&self) -> f64 {
        self.params[1]
    }
    /// Get the z component of translation.
    pub fn z(&self) -> f64 {
        self.params[2]
    }
    /// Get the vx component of velocity.
    pub fn vx(&self) -> f64 {
        self.params[7]
    }
    /// Get the vy component of velocity.
    pub fn vy(&self) -> f64 {
        self.params[8]
    }
    /// Get the vz component of velocity.
    pub fn vz(&self) -> f64 {
        self.params[9]
    }

    /// Get the rotation matrix (3x3).
    pub fn rotation_matrix(&self) -> Matrix3<f64> {
        self.rotation_impl().rotation_matrix()
    }

    /// Get the parameter vector [tx, ty, tz, qw, qx, qy, qz, vx, vy, vz, time].
    pub fn coeffs(&self) -> [f64; 11] {
        [
            self.params[0],
            self.params[1],
            self.params[2],
            self.params[3],
            self.params[4],
            self.params[5],
            self.params[6],
            self.params[7],
            self.params[8],
            self.params[9],
            self.params[10],
        ]
    }

    /// Get the 5x5 homogeneous matrix representation.
    pub fn matrix(&self) -> SMatrix<f64, 5, 5> {
        let mut mat = SMatrix::<f64, 5, 5>::identity();
        let rot = self.rotation_matrix();
        mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&rot);
        mat[(0, 3)] = self.params[0];
        mat[(1, 3)] = self.params[1];
        mat[(2, 3)] = self.params[2];
        mat[(0, 4)] = self.params[7];
        mat[(1, 4)] = self.params[8];
        mat[(2, 4)] = self.params[9];
        mat[(3, 4)] = self.params[10];
        mat
    }
}
impl LieGroup for SGal3 {
    const NAME: &'static str = "SGal3";

    type TangentVector = SGal3Tangent;
    type JacobianMatrix = Matrix10<f64>;
    type LieAlgebra = SMatrix<f64, 6, 6>;

    /// Get the inverse.
    ///
    /// For SGal(3): g^{-1} = (R^T, -R^T * (t - s*v), -R^T * v, -s)
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self {
        let rot = self.rotation_impl();
        let rot_inv = rot.inverse(None);
        let t = self.translation_impl();
        let v = self.velocity_impl();
        let s = self.time_impl();
        let trans_inv = -rot_inv.act(&(t - s * v), None, None);
        let vel_inv = -rot_inv.act(&v, None, None);
        let time_inv = -s;

        if let Some(jac) = jacobian {
            *jac = -self.adjoint();
        }

        SGal3::from_parts(trans_inv, vel_inv, time_inv, &rot_inv)
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
        let rot = self.rotation_impl();
        let s = self.time_impl();
        let composed_rotation = rot.compose(&other.rotation_impl(), None, None);
        let composed_translation = rot.act(
            &(other.translation_impl() + s * other.velocity_impl()),
            None,
            None,
        ) + self.translation_impl();
        let composed_velocity = rot.act(&other.velocity_impl(), None, None) + self.velocity_impl();
        let composed_time = s + other.time_impl();

        let result = SGal3::from_parts(
            composed_translation,
            composed_velocity,
            composed_time,
            &composed_rotation,
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
        let theta = self.rotation_impl().log(None);
        let mut data = Vector10::zeros();

        // Invert the exp relations: ν_x = Jl(θ)⁻¹·ν_e and
        // ρ_x = Jl(θ)⁻¹·(ρ_e − s·M(θ)·ν_x). The old log omitted the coupling
        // term, so exp∘log returned a different element whenever s·ν ≠ 0.
        let v_inv = theta.left_jacobian_inv();
        let velocity_vector = v_inv * self.velocity_impl();
        let coupling = SGal3Tangent::s_nu_coupling(&theta.coeffs());
        let translation_vector =
            v_inv * (self.translation_impl() - self.time_impl() * (coupling * velocity_vector));

        data.fixed_rows_mut::<3>(0).copy_from(&translation_vector);
        data.fixed_rows_mut::<3>(3).copy_from(&velocity_vector);
        data.fixed_rows_mut::<3>(6).copy_from(&theta.coeffs());
        data[9] = self.time_impl();

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
        let rot = self.rotation_impl();
        let rotation_matrix = rot.rotation_matrix();
        let velocity = self.velocity_impl();
        let time = self.time_impl();
        let result = rot.act(vector, None, None) + self.translation_impl() + time * velocity;

        if let Some(jac_self) = jacobian_self {
            jac_self
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&rotation_matrix);
            jac_self
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(time * rotation_matrix));
            jac_self
                .fixed_view_mut::<3, 3>(0, 6)
                .copy_from(&(-rotation_matrix * SO3Tangent::new(*vector).hat()));
            jac_self.fixed_view_mut::<3, 1>(0, 9).copy_from(&velocity);
        }

        if let Some(jac_vector) = jacobian_vector {
            *jac_vector = rotation_matrix;
        }

        result
    }

    fn adjoint(&self) -> Self::JacobianMatrix {
        let r = self.rotation_impl().rotation_matrix();
        let rho = self.translation_impl();
        let nu = self.velocity_impl();
        let t = self.time_impl();
        let mut adj = Matrix10::zeros();

        // Tangent ordering [ρ, ν, θ, s]. Derived from the corrected group law
        // g₁∘g₂ = (R₁R₂, R₁(t₂+s₁ν₂)+ρ₁, R₁ν₂+ν₁, t₁+t₂): conjugating
        // exp(ξ) ≈ (I+θ̂, ρ, ν, σ) by g gives
        //   ρ' = Rρ + t·Rν + ρ̂Rθ − ν·σ
        //   ν' = Rν + ν̂Rθ,  θ' = Rθ,  s' = σ.
        // Verified against Log(g ∘ exp(ξ) ∘ g⁻¹) by
        // `adjoint_matches_group_conjugation`.

        adj.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        adj.fixed_view_mut::<3, 3>(0, 3).copy_from(&(t * r));
        adj.fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&(SO3Tangent::new(rho).hat() * r));
        adj.fixed_view_mut::<3, 1>(0, 9).copy_from(&(-nu));

        adj.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        adj.fixed_view_mut::<3, 3>(3, 6)
            .copy_from(&(SO3Tangent::new(nu).hat() * r));

        adj.fixed_view_mut::<3, 3>(6, 6).copy_from(&r);

        adj[(9, 9)] = 1.0;

        adj
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

        SGal3::from_parts(translation, velocity, time, &rotation)
    }

    fn normalize(&mut self) {
        let mut rot = self.rotation_impl();
        rot.normalize();
        let q = rot.params();
        self.params[3] = q[0];
        self.params[4] = q[1];
        self.params[5] = q[2];
        self.params[6] = q[3];
    }

    fn is_valid(&self, tolerance: f64) -> bool {
        self.rotation_impl().is_valid(tolerance)
    }

    fn as_param_slice(&self) -> &[f64] {
        self.params.as_slice()
    }

    fn as_param_slice_mut(&mut self) -> &mut [f64] {
        self.params.as_mut_slice()
    }

    fn from_param_slice(s: &[f64]) -> Self {
        debug_assert_eq!(s.len(), 11);
        SGal3 {
            params: SVector::from_column_slice(s),
        }
    }

    fn vee(&self) -> Self::TangentVector {
        self.log(None)
    }

    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        let difference = self.right_minus(other, None, None);
        difference.is_zero(tolerance)
    }

    fn jacobian_identity() -> Self::JacobianMatrix {
        Matrix10::<f64>::identity()
    }

    fn zero_jacobian() -> Self::JacobianMatrix {
        Matrix10::<f64>::zeros()
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

    /// The ρ–ν coupling matrix of the SGal(3) exponential:
    /// `M(ω) = ½I + α·ω̂ + β·ω̂²` with
    /// `α = (sin w − w·cos w)/w³`, `β = (1 − cos w)/w⁴ − sin w/w³ + 1/(2w²)`,
    /// derived by integrating `∫₀¹ Exp(σω)·σ dσ` over the time flow. The
    /// small-angle limit is `½I + ⅓ω̂ + ⅛ω̂²`. This is the term the old,
    /// uncoupled exponential dropped entirely.
    pub(crate) fn s_nu_coupling(theta: &Vector3<f64>) -> Matrix3<f64> {
        let w_squared = theta.norm_squared();
        let w = w_squared.sqrt();
        let theta_skew = SO3Tangent::new(*theta).hat();

        if w_squared <= crate::SMALL_ANGLE_THRESHOLD {
            // Series: ½I + (1/3 − w²/30)ω̂ + (1/8 − w²/120)ω̂²
            return Matrix3::identity() * 0.5
                + theta_skew * (1.0 / 3.0 - w_squared / 30.0)
                + (theta_skew * theta_skew) * (1.0 / 8.0 - w_squared / 120.0);
        }

        let alpha = (w.sin() - w * w.cos()) / w.powi(3);
        let beta = (1.0 - w.cos()) / w.powi(4) - w.sin() / w.powi(3) + 1.0 / (2.0 * w_squared);
        Matrix3::identity() * 0.5 + theta_skew * alpha + (theta_skew * theta_skew) * beta
    }

    /// Get the s (time) part.
    pub fn s(&self) -> f64 {
        self.data[9]
    }

    /// Create SGal3Tangent from individual scalar components.
    ///
    /// Order: [ρ_x, ρ_y, ρ_z, ν_x, ν_y, ν_z, θ_x, θ_y, θ_z, s]
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        rho_x: f64,
        rho_y: f64,
        rho_z: f64,
        nu_x: f64,
        nu_y: f64,
        nu_z: f64,
        theta_x: f64,
        theta_y: f64,
        theta_z: f64,
        s: f64,
    ) -> Self {
        SGal3Tangent {
            data: Vector10::from_column_slice(&[
                rho_x, rho_y, rho_z, nu_x, nu_y, nu_z, theta_x, theta_y, theta_z, s,
            ]),
        }
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

        // One-parameter subgroup of the group law
        //   ρ' = ρ₁ + R₁(ρ₂ + s₂ν₂),  ν' = ν₁ + R₁ν₂,  s' = s₁ + s₂:
        // flowing from the identity under the twist (ρ, ν, θ, s) gives
        //   ν' = Jl(θ)·ν,   ρ' = Jl(θ)·ρ + s·M(θ)·ν
        // with M the time–velocity coupling integrated from ∫₀¹ Exp(σω)·σ dσ.
        // The old exp dropped the s·M(θ)·ν term entirely, which made exp fail
        // the one-parameter subgroup law.
        let v_matrix = theta_tangent.left_jacobian();
        let velocity = v_matrix * nu;
        let coupling = SGal3Tangent::s_nu_coupling(&theta);
        let translation = v_matrix * rho + s * (coupling * nu);

        if let Some(jac) = jacobian {
            *jac = self.right_jacobian();
        }

        SGal3::from_parts(translation, velocity, s, &rotation)
    }

    /// Right Jacobian for SGal(3).
    fn right_jacobian(&self) -> <SGal3 as LieGroup>::JacobianMatrix {
        // Jr(ξ) is defined by  Exp(ξ + δ) ≈ Exp(ξ) ∘ Exp(Jr(ξ)·δ), i.e.
        //
        //     Jr(ξ)·δ = Log( Exp(ξ)⁻¹ ∘ Exp(ξ + δ) )
        //
        // evaluated here by central differences through the crate's own
        // compose/log, so it tracks the group law by construction rather than
        // a hand-written block table derived for the uncoupled exponential.
        const EPS: f64 = 1e-6;
        let mut jac = Matrix10::zeros();
        for k in 0..10 {
            let base = self.data.as_slice().to_vec();
            let mut plus_k = base.clone();
            let mut minus_k = base.clone();
            plus_k[k] += EPS;
            minus_k[k] -= EPS;
            let tan_p = SGal3Tangent::from_slice(&plus_k);
            let tan_m = SGal3Tangent::from_slice(&minus_k);
            let element_inv = self.exp(None).inverse(None);
            let rp = element_inv.compose(&tan_p.exp(None), None, None).log(None);
            let rm = element_inv.compose(&tan_m.exp(None), None, None).log(None);
            for r in 0..10 {
                jac[(r, k)] = (rp.as_slice()[r] - rm.as_slice()[r]) / (2.0 * EPS);
            }
        }
        jac
    }

    /// Left Jacobian for SGal(3).
    fn left_jacobian(&self) -> <SGal3 as LieGroup>::JacobianMatrix {
        // Jl(ξ) is defined by  Exp(ξ + δ) ≈ Exp(Jl(ξ)·δ) ∘ Exp(ξ), i.e.
        //
        //     Jl(ξ)·δ = Log( Exp(ξ + δ) ∘ Exp(ξ)⁻¹ )
        //
        // by central differences (same rationale as `right_jacobian`).
        const EPS: f64 = 1e-6;
        let mut jac = Matrix10::zeros();
        for k in 0..10 {
            let base = self.data.as_slice().to_vec();
            let mut plus_k = base.clone();
            let mut minus_k = base.clone();
            plus_k[k] += EPS;
            minus_k[k] -= EPS;
            let tan_p = SGal3Tangent::from_slice(&plus_k);
            let tan_m = SGal3Tangent::from_slice(&minus_k);
            let element_inv = self.exp(None).inverse(None);
            let lp = tan_p.exp(None).compose(&element_inv, None, None).log(None);
            let lm = tan_m.exp(None).compose(&element_inv, None, None).log(None);
            for r in 0..10 {
                jac[(r, k)] = (lp.as_slice()[r] - lm.as_slice()[r]) / (2.0 * EPS);
            }
        }
        jac
    }

    /// Inverse of right Jacobian.
    fn right_jacobian_inv(&self) -> <SGal3 as LieGroup>::JacobianMatrix {
        self.right_jacobian().try_inverse().unwrap_or_else(|| {
            tracing::error!("SGal(3) right-Jacobian inverse failed; returning identity");
            Matrix10::identity()
        })
    }

    /// Inverse of left Jacobian.
    fn left_jacobian_inv(&self) -> <SGal3 as LieGroup>::JacobianMatrix {
        self.left_jacobian().try_inverse().unwrap_or_else(|| {
            tracing::error!("SGal(3) left-Jacobian inverse failed; returning identity");
            Matrix10::identity()
        })
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
        lie_alg[(3, 5)] = self.s();

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

    fn as_slice(&self) -> &[f64] {
        self.data.as_slice()
    }

    fn from_slice(s: &[f64]) -> Self {
        debug_assert_eq!(s.len(), 10);
        SGal3Tangent {
            data: Vector10::from_column_slice(s),
        }
    }

    fn small_adj(&self) -> <SGal3 as LieGroup>::JacobianMatrix {
        let mut small_adj = Matrix10::zeros();
        let rho_skew = SO3Tangent::new(self.rho()).hat();
        let nu_skew = SO3Tangent::new(self.nu()).hat();
        let theta_skew = SO3Tangent::new(self.theta()).hat();
        let s = self.s();

        // Block structure for SGal(3) with ordering [ρ, ν, θ, s]:
        // [θ×  -s·I   ρ×   ν ]
        // [0    θ×    ν×   0 ]
        // [0    0     θ×   0 ]
        // [0    0     0    0 ]

        small_adj
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&theta_skew);
        small_adj
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-s * Matrix3::identity()));
        small_adj.fixed_view_mut::<3, 3>(0, 6).copy_from(&rho_skew);
        small_adj.fixed_view_mut::<3, 1>(0, 9).copy_from(&self.nu());

        small_adj
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&theta_skew);
        small_adj.fixed_view_mut::<3, 3>(3, 6).copy_from(&nu_skew);

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
        let rho = Vector3::new(1.0, 2.0, 3.0);
        let nu = Vector3::new(4.0, 5.0, 6.0);
        let theta = Vector3::new(0.1, 0.2, 0.3);
        let s = 0.5;

        let tangent = SGal3Tangent::new(rho, nu, theta, s);

        assert_eq!(tangent.rho(), rho);
        assert_eq!(tangent.nu(), nu);
        assert_eq!(tangent.theta(), theta);
        assert_eq!(tangent.s(), s);

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

    #[test]
    fn test_sgal3_tangent_basic() {
        let tangent = SGal3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 5.0, 6.0),
            Vector3::new(0.1, 0.2, 0.3),
            0.5,
        );

        assert!((tangent.rho() - Vector3::new(1.0, 2.0, 3.0)).norm() < TOLERANCE);
        assert!((tangent.nu() - Vector3::new(4.0, 5.0, 6.0)).norm() < TOLERANCE);
        assert!((tangent.theta() - Vector3::new(0.1, 0.2, 0.3)).norm() < TOLERANCE);
        assert!((tangent.s() - 0.5).abs() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_tangent_from_components() {
        let tangent =
            SGal3Tangent::from_components(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3, 0.5);

        let expected = SGal3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 5.0, 6.0),
            Vector3::new(0.1, 0.2, 0.3),
            0.5,
        );

        assert!(tangent.is_approx(&expected, TOLERANCE));
    }

    #[test]
    fn test_sgal3_normalize() {
        let mut sgal3 = SGal3::random();
        sgal3.normalize();
        assert!(sgal3.is_valid(TOLERANCE));

        // Normalize identity
        let mut identity = SGal3::identity();
        identity.normalize();
        assert!(identity.is_valid(TOLERANCE));
    }

    #[test]
    fn test_sgal3_coeffs() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let velocity = Vector3::new(4.0, 5.0, 6.0);
        let rotation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let time = 0.5;

        let sgal3 = SGal3::new(translation, velocity, rotation, time);
        let c = sgal3.coeffs();

        // Layout: [tx, ty, tz, qw, qx, qy, qz, vx, vy, vz, time]
        assert!((c[0] - 1.0).abs() < TOLERANCE);
        assert!((c[1] - 2.0).abs() < TOLERANCE);
        assert!((c[2] - 3.0).abs() < TOLERANCE);
        // qw, qx, qy, qz
        let q = sgal3.rotation_quaternion();
        assert!((c[3] - q.w).abs() < TOLERANCE);
        assert!((c[4] - q.i).abs() < TOLERANCE);
        assert!((c[5] - q.j).abs() < TOLERANCE);
        assert!((c[6] - q.k).abs() < TOLERANCE);
        // vx, vy, vz
        assert!((c[7] - 4.0).abs() < TOLERANCE);
        assert!((c[8] - 5.0).abs() < TOLERANCE);
        assert!((c[9] - 6.0).abs() < TOLERANCE);
        // time
        assert!((c[10] - 0.5).abs() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_matrix() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let velocity = Vector3::new(4.0, 5.0, 6.0);
        let rotation = UnitQuaternion::identity();
        let time = 0.5;

        let sgal3 = SGal3::new(translation, velocity, rotation, time);
        let mat = sgal3.matrix();

        // Top-left 3x3 is rotation (identity here)
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((mat[(i, j)] - expected).abs() < TOLERANCE);
            }
        }
        // Translation column
        assert!((mat[(0, 3)] - 1.0).abs() < TOLERANCE);
        assert!((mat[(1, 3)] - 2.0).abs() < TOLERANCE);
        assert!((mat[(2, 3)] - 3.0).abs() < TOLERANCE);
        // Velocity column
        assert!((mat[(0, 4)] - 4.0).abs() < TOLERANCE);
        assert!((mat[(1, 4)] - 5.0).abs() < TOLERANCE);
        assert!((mat[(2, 4)] - 6.0).abs() < TOLERANCE);
        // Time
        assert!((mat[(3, 4)] - 0.5).abs() < TOLERANCE);
        // Bottom row
        assert!((mat[(4, 4)] - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_manif_like_operations() {
        let a = SGal3::random();
        let b = SGal3::random();
        let c = SGal3::random();

        // plus(minus(b, a)) == b
        let diff = b.right_minus(&a, None, None);
        let recovered = a.right_plus(&diff, None, None);
        assert!(recovered.is_approx(&b, 1e-8));

        // between chain: a.between(b).compose(b.between(c)) == a.between(c)
        let ab = a.between(&b, None, None);
        let bc = b.between(&c, None, None);
        let ac = a.between(&c, None, None);
        let chain = ab.compose(&bc, None, None);
        assert!(chain.is_approx(&ac, 1e-8));
    }

    #[test]
    fn test_sgal3_right_jacobian_inverse_identity() {
        let tangent = SGal3Tangent::new(
            Vector3::new(0.1, -0.2, 0.3),
            Vector3::new(0.4, 0.5, -0.6),
            Vector3::new(0.05, -0.03, 0.07),
            0.2,
        );

        let jr = tangent.right_jacobian();
        let jr_inv = tangent.right_jacobian_inv();
        let product = jr * jr_inv;

        let identity = Matrix10::<f64>::identity();
        assert!(
            (product - identity).norm() < 1e-8,
            "Jr * Jr_inv should be identity, error = {}",
            (product - identity).norm()
        );
    }

    #[test]
    fn test_sgal3_left_jacobian_inverse_identity() {
        let tangent = SGal3Tangent::new(
            Vector3::new(0.1, -0.2, 0.3),
            Vector3::new(0.4, 0.5, -0.6),
            Vector3::new(0.05, -0.03, 0.07),
            0.2,
        );

        let jl = tangent.left_jacobian();
        let jl_inv = tangent.left_jacobian_inv();
        let product = jl * jl_inv;

        let identity = Matrix10::<f64>::identity();
        assert!(
            (product - identity).norm() < 1e-8,
            "Jl * Jl_inv should be identity, error = {}",
            (product - identity).norm()
        );
    }

    #[test]
    fn test_sgal3_jacobi_identity() {
        // Jacobi identity: [a, [b, c]] + [b, [c, a]] + [c, [a, b]] = 0
        let a = SGal3Tangent::random();
        let b = SGal3Tangent::random();
        let c = SGal3Tangent::random();

        let bc = b.lie_bracket(&c);
        let ca = c.lie_bracket(&a);
        let ab = a.lie_bracket(&b);

        let term1 = a.lie_bracket(&bc);
        let term2 = b.lie_bracket(&ca);
        let term3 = c.lie_bracket(&ab);

        let sum = term1.data + term2.data + term3.data;
        assert!(
            sum.norm() < 1e-8,
            "Jacobi identity violated, norm = {}",
            sum.norm()
        );
    }

    #[test]
    fn test_sgal3_hat_matrix_structure() {
        let tangent = SGal3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 5.0, 6.0),
            Vector3::new(0.1, 0.2, 0.3),
            0.5,
        );
        let hat = tangent.hat();

        // Top-left 3x3 should be skew-symmetric (theta hat)
        let theta_hat = SO3Tangent::new(tangent.theta()).hat();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (hat[(i, j)] - theta_hat[(i, j)]).abs() < TOLERANCE,
                    "hat({},{}) mismatch",
                    i,
                    j
                );
            }
        }

        // Column 3 should be rho
        assert!((hat[(0, 3)] - 1.0).abs() < TOLERANCE);
        assert!((hat[(1, 3)] - 2.0).abs() < TOLERANCE);
        assert!((hat[(2, 3)] - 3.0).abs() < TOLERANCE);

        // Column 4 should be nu
        assert!((hat[(0, 4)] - 4.0).abs() < TOLERANCE);
        assert!((hat[(1, 4)] - 5.0).abs() < TOLERANCE);
        assert!((hat[(2, 4)] - 6.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_param_slice_round_trip() {
        let sgal3 = SGal3::random();
        let recovered = SGal3::from_param_slice(sgal3.as_param_slice());
        assert!(sgal3.is_approx(&recovered, TOLERANCE));
    }

    #[test]
    fn test_sgal3_tangent_norm() {
        let tangent = SGal3Tangent::new(
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::zeros(),
            Vector3::zeros(),
            0.0,
        );
        assert!((tangent.data.norm() - 1.0).abs() < TOLERANCE);

        let zero = SGal3Tangent::zero();
        assert!(zero.data.norm() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_tangent_normalize() {
        let mut tangent = SGal3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 5.0, 6.0),
            Vector3::new(0.3, 0.4, 0.0),
            0.5,
        );
        tangent.normalize();
        let theta_norm = tangent.theta().norm();
        assert!((theta_norm - 1.0).abs() < TOLERANCE);

        let normalized = tangent.normalized();
        assert!((normalized.theta().norm() - 1.0).abs() < TOLERANCE);
    }

    // --- Additional coverage tests ---

    #[test]
    fn test_sgal3_display() {
        let sgal3 = SGal3::identity();
        let s = format!("{}", sgal3);
        assert!(s.contains("SGal3"));

        let tangent = SGal3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 5.0, 6.0),
            Vector3::new(0.1, 0.2, 0.3),
            0.5,
        );
        let ts = format!("{}", tangent);
        assert!(ts.contains("sgal3"));
    }

    #[test]
    fn test_sgal3_jacobian_identity_static() {
        let jac = SGal3::jacobian_identity();
        assert!((jac - Matrix10::identity()).norm() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_rotation_so3() {
        let sgal3 = SGal3::random();
        let so3 = sgal3.rotation_so3();
        assert!(so3.is_valid(TOLERANCE));
    }

    #[test]
    fn test_sgal3_inverse_with_jacobian() {
        let sgal3 = SGal3::random();
        let mut jac = Matrix10::zeros();
        let inv = sgal3.inverse(Some(&mut jac));
        let expected_jac = -sgal3.adjoint();
        assert!((jac - expected_jac).norm() < TOLERANCE);
        let composed = sgal3.compose(&inv, None, None);
        assert!(composed.is_approx(&SGal3::identity(), TOLERANCE));
    }

    #[test]
    fn test_sgal3_compose_with_jacobians() {
        let a = SGal3::random();
        let b = SGal3::random();
        let mut jac_a = Matrix10::zeros();
        let mut jac_b = Matrix10::zeros();
        let _composed = a.compose(&b, Some(&mut jac_a), Some(&mut jac_b));
        assert!(jac_a.norm() > 0.0);
        assert!(jac_b.norm() > 0.0);
    }

    #[test]
    fn test_sgal3_act_with_jacobians() {
        let sgal3 = SGal3::random();
        let point = Vector3::new(1.0, 2.0, 3.0);
        let mut jac_self = Matrix10::zeros();
        let mut jac_point = Matrix3::<f64>::zeros();
        let result = sgal3.act(&point, Some(&mut jac_self), Some(&mut jac_point));
        assert!(result.norm() > 0.0);
        assert!(jac_self.norm() > 0.0);
        assert!(jac_point.norm() > 0.0);
    }

    #[test]
    fn test_sgal3_liegroup_jacobian_identity() {
        let jac = <SGal3 as LieGroup>::jacobian_identity();
        assert!((jac - Matrix10::identity()).norm() < TOLERANCE);

        let zero = <SGal3 as LieGroup>::zero_jacobian();
        assert!(zero.norm() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_tangent_slice_round_trip() {
        let tangent = SGal3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 5.0, 6.0),
            Vector3::new(0.1, 0.2, 0.3),
            0.5,
        );
        let recovered = SGal3Tangent::from_slice(tangent.as_slice());
        assert!(tangent.is_approx(&recovered, TOLERANCE));
    }

    #[test]
    fn test_sgal3_exp_with_jacobian() {
        let tangent = SGal3Tangent::new(
            Vector3::new(0.1, 0.2, 0.3),
            Vector3::new(0.4, 0.5, 0.6),
            Vector3::new(0.01, 0.02, 0.03),
            0.1,
        );
        let mut jac = Matrix10::zeros();
        let _result = tangent.exp(Some(&mut jac));
        assert!(jac.norm() > 0.0);
    }

    #[test]
    fn test_sgal3_exp_log_stress() {
        for _ in 0..100 {
            let tangent = SGal3Tangent::random();
            let sgal3 = tangent.exp(None);
            let recovered = sgal3.log(None);
            assert!(
                tangent.is_approx(&recovered, 1e-6),
                "exp/log round-trip failed: error = {}",
                (tangent.data - recovered.data).norm()
            );
        }
    }

    #[test]
    fn test_sgal3_right_plus_minus_round_trip() {
        let a = SGal3::random();
        let b = SGal3::random();
        let diff = a.right_minus(&b, None, None);
        let recovered = b.right_plus(&diff, None, None);
        assert!(a.is_approx(&recovered, 1e-6));
    }

    #[test]
    fn test_sgal3_left_plus_minus_round_trip() {
        let a = SGal3::random();
        let b = SGal3::random();
        let diff = a.left_minus(&b, None, None);
        let recovered = b.left_plus(&diff, None, None);
        assert!(a.is_approx(&recovered, 1e-6));
    }

    #[test]
    fn test_sgal3_compose_associativity() {
        let a = SGal3::random();
        let b = SGal3::random();
        let c = SGal3::random();
        let ab_c = a.compose(&b, None, None).compose(&c, None, None);
        let a_bc = a.compose(&b.compose(&c, None, None), None, None);
        assert!(ab_c.is_approx(&a_bc, 1e-6));
    }

    #[test]
    fn test_sgal3_inverse_twice() {
        let g = SGal3::random();
        let g_inv_inv = g.inverse(None).inverse(None);
        assert!(g.is_approx(&g_inv_inv, 1e-6));
    }

    #[test]
    fn test_sgal3_tangent_normalize_zero_theta() {
        let tangent = SGal3Tangent::new(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 5.0, 6.0),
            Vector3::zeros(),
            0.5,
        );
        let normalized = tangent.normalized();
        assert!(normalized.theta().norm() < TOLERANCE);
    }

    #[test]
    fn test_sgal3_tangent_generator_all() {
        let tangent = SGal3Tangent::zero();
        for i in 0..10 {
            let g = tangent.generator(i);
            assert!(g.norm() > 0.0, "Generator {} should be non-zero", i);
        }
    }

    #[test]
    fn test_sgal3_hat_linearity() {
        let a = SGal3Tangent::random();
        let b = SGal3Tangent::random();
        let alpha = 2.5;

        // hat(α*a) = α*hat(a)
        let scaled_tangent = SGal3Tangent {
            data: a.data * alpha,
        };
        let hat_scaled = scaled_tangent.hat();
        let scaled_hat = a.hat() * alpha;
        assert!(
            (hat_scaled - scaled_hat).norm() < TOLERANCE,
            "hat should be linear"
        );

        // hat(a + b) = hat(a) + hat(b)
        let sum_tangent = SGal3Tangent {
            data: a.data + b.data,
        };
        let hat_sum = sum_tangent.hat();
        let sum_hat = a.hat() + b.hat();
        assert!(
            (hat_sum - sum_hat).norm() < TOLERANCE,
            "hat should be additive"
        );
    }

    #[test]
    fn test_sgal3_pure_time() {
        let tangent = SGal3Tangent::new(Vector3::zeros(), Vector3::zeros(), Vector3::zeros(), 1.0);
        let sgal3 = tangent.exp(None);
        let recovered = sgal3.log(None);
        assert!(tangent.is_approx(&recovered, TOLERANCE));
    }

    #[test]
    fn sgal3_param_slice_round_trip() {
        let g = SGal3::random();
        let recovered = SGal3::from_param_slice(g.as_param_slice());
        assert!(g.is_approx(&recovered, 1e-14));
    }

    #[test]
    fn sgal3_tangent_slice_round_trip() {
        let t = SGal3Tangent::random();
        let recovered = SGal3Tangent::from_slice(t.as_slice());
        assert!(t.is_approx(&recovered, 1e-14));
    }
}

#[cfg(test)]
mod adjoint_check {
    use super::*;
    use crate::Tangent;

    // Ad(g) ξ must equal Log(g ∘ exp(ξ) ∘ g⁻¹).
    #[test]
    fn adjoint_matches_group_conjugation() {
        for _ in 0..20 {
            let g = SGal3::random();
            let xi = SGal3Tangent::random();
            let conj = g
                .compose(&xi.exp(None), None, None)
                .compose(&g.inverse(None), None, None);
            let lhs = conj.log(None);
            let rhs = g.adjoint() * xi.data;
            let err = (lhs.data - rhs).norm();
            assert!(err < 1e-8, "adjoint mismatch: err = {err:.3e}");
        }
    }
}
