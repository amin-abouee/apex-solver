//! SO(2) - Special Orthogonal Group in 2D
//!
//! This module implements the Special Orthogonal group SO(2), which represents
//! rotations in 2D space.
//!
//! SO(2) elements are stored as a single angle θ (radians). UnitComplex is
//! constructed on-the-fly for math operations. This gives contiguous single-float
//! storage compatible with zero-copy faer views.
//!
//! The implementation follows the [manif](https://github.com/artivis/manif) C++ library
//! conventions and provides all operations required by the LieGroup and Tangent traits.

use crate::{LieGroup, Tangent};
use nalgebra::{Matrix1, Matrix2, Matrix3, UnitComplex, Vector2, Vector3};
use std::{
    fmt,
    fmt::{Display, Formatter},
};

/// SO(2) group element representing rotations in 2D.
///
/// Stored as a single angle θ (radians). UnitComplex is derived on demand for
/// math operations, keeping storage minimal and contiguous.
#[derive(Clone, PartialEq)]
pub struct SO2 {
    /// Rotation angle in radians
    theta: f64,
}

impl Display for SO2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "SO2(angle: {:.4})", self.theta)
    }
}

impl SO2 {
    /// Space dimension - dimension of the ambient space that the group acts on
    pub const DIM: usize = 2;

    /// Degrees of freedom - dimension of the tangent space
    pub const DOF: usize = 1;

    /// Representation size - one angle θ
    pub const REP_SIZE: usize = 1;

    /// Get the identity element of the group.
    pub fn identity() -> Self {
        SO2 { theta: 0.0 }
    }

    /// Get the identity matrix for Jacobians.
    pub fn jacobian_identity() -> Matrix1<f64> {
        Matrix1::<f64>::identity()
    }

    /// Create a new SO(2) element from a unit complex number.
    #[inline]
    pub fn new(complex: UnitComplex<f64>) -> Self {
        SO2 {
            theta: complex.angle(),
        }
    }

    /// Create SO(2) from an angle.
    pub fn from_angle(angle: f64) -> Self {
        SO2 { theta: angle }
    }

    /// Derive a UnitComplex from the stored angle (on-the-fly).
    #[inline]
    fn unit_complex(&self) -> UnitComplex<f64> {
        UnitComplex::new(self.theta)
    }

    /// Get the underlying unit complex number.
    pub fn complex(&self) -> UnitComplex<f64> {
        self.unit_complex()
    }

    /// Get the rotation angle in radians.
    #[inline]
    pub fn angle(&self) -> f64 {
        self.theta
    }

    /// Get the rotation matrix (2x2).
    pub fn rotation_matrix(&self) -> Matrix2<f64> {
        self.unit_complex().to_rotation_matrix().into_inner()
    }
}

impl LieGroup for SO2 {
    const NAME: &'static str = "SO2";

    type TangentVector = SO2Tangent;
    type JacobianMatrix = Matrix1<f64>;
    type LieAlgebra = Matrix2<f64>;

    /// SO2 inverse: R(θ)⁻¹ = R(-θ).
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self {
        if let Some(jac) = jacobian {
            *jac = -self.adjoint();
        }
        SO2 { theta: -self.theta }
    }

    /// SO2 composition: angles add, wrapped via UnitComplex to stay in [-π, π].
    fn compose(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        if let Some(jac_self) = jacobian_self {
            *jac_self = other.inverse(None).adjoint();
        }
        if let Some(jac_other) = jacobian_other {
            *jac_other = Matrix1::identity();
        }
        SO2 {
            theta: (self.unit_complex() * other.unit_complex()).angle(),
        }
    }

    /// Logarithmic map: returns the canonical angle in [-π, π].
    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector {
        if let Some(jac) = jacobian {
            *jac = Matrix1::identity();
        }
        SO2Tangent {
            data: self.unit_complex().angle(),
        }
    }

    /// Rotation action on a 3-vector (operates on the xy-plane).
    fn act(
        &self,
        vector: &Vector3<f64>,
        _jacobian_self: Option<&mut Self::JacobianMatrix>,
        _jacobian_vector: Option<&mut Matrix3<f64>>,
    ) -> Vector3<f64> {
        let point2d = Vector2::new(vector.x, vector.y);
        let rotated = self.unit_complex() * point2d;
        Vector3::new(rotated.x, rotated.y, vector.z)
    }

    /// Adjoint for SO(2) is identity (abelian group).
    fn adjoint(&self) -> Self::JacobianMatrix {
        Matrix1::identity()
    }

    fn random() -> Self {
        SO2::from_angle(rand::random::<f64>() * 2.0 * std::f64::consts::PI)
    }

    fn jacobian_identity() -> Self::JacobianMatrix {
        Matrix1::<f64>::identity()
    }

    fn zero_jacobian() -> Self::JacobianMatrix {
        Matrix1::<f64>::zeros()
    }

    /// Normalize: wrap angle to [-π, π] using UnitComplex.
    fn normalize(&mut self) {
        self.theta = self.unit_complex().angle();
    }

    /// Any finite angle is valid.
    fn is_valid(&self, _tolerance: f64) -> bool {
        self.theta.is_finite()
    }

    fn vee(&self) -> Self::TangentVector {
        self.log(None)
    }

    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        let difference = self.right_minus(other, None, None);
        difference.is_zero(tolerance)
    }

    fn as_param_slice(&self) -> &[f64] {
        std::slice::from_ref(&self.theta)
    }

    fn as_param_slice_mut(&mut self) -> &mut [f64] {
        std::slice::from_mut(&mut self.theta)
    }

    fn from_param_slice(s: &[f64]) -> Self {
        debug_assert_eq!(s.len(), 1);
        SO2 { theta: s[0] }
    }
}

/// SO(2) tangent space element representing elements in the Lie algebra so(2).
///
/// Internally represented as a single scalar (angle in radians).
#[derive(Clone, PartialEq)]
pub struct SO2Tangent {
    /// Internal data: angle (radians)
    data: f64,
}

impl fmt::Display for SO2Tangent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "so2(angle: {:.4})", self.data)
    }
}

impl SO2Tangent {
    /// Create a new SO2Tangent from an angle.
    ///
    /// # Arguments
    /// * `angle` - Angle in radians
    #[inline]
    pub fn new(angle: f64) -> Self {
        SO2Tangent { data: angle }
    }

    /// Get the angle.
    #[inline]
    pub fn angle(&self) -> f64 {
        self.data
    }
}

impl Tangent<SO2> for SO2Tangent {
    /// Dimension of the tangent space
    const DIM: usize = 1;

    /// SO2 exponential map.
    ///
    /// # Arguments
    /// * `tangent` - Tangent vector (angle)
    /// * `jacobian` - Optional Jacobian matrix of the SE(3) element wrt this.
    fn exp(&self, jacobian: Option<&mut <SO2 as LieGroup>::JacobianMatrix>) -> SO2 {
        let angle = self.angle();
        let complex = UnitComplex::new(angle);

        if let Some(jac) = jacobian {
            *jac = Matrix1::identity();
        }

        SO2::new(complex)
    }

    /// Right Jacobian for SO(2) is identity.
    fn right_jacobian(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        Matrix1::identity()
    }

    /// Left Jacobian for SO(2) is identity.
    fn left_jacobian(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        Matrix1::identity()
    }

    /// Inverse of right Jacobian for SO(2) is identity.
    fn right_jacobian_inv(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        Matrix1::identity()
    }

    /// Inverse of left Jacobian for SO(2) is identity.
    fn left_jacobian_inv(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        Matrix1::identity()
    }

    /// Hat operator: θ^∧ (scalar to skew-symmetric matrix).
    fn hat(&self) -> <SO2 as LieGroup>::LieAlgebra {
        let theta = self.data;
        Matrix2::new(0.0, -theta, theta, 0.0)
    }

    /// Zero tangent vector for SO2
    fn zero() -> Self {
        SO2Tangent { data: 0.0 }
    }

    /// Random tangent vector for SO2
    fn random() -> Self {
        SO2Tangent {
            data: rand::random::<f64>() * 0.2 - 0.1,
        }
    }

    /// Check if tangent vector is zero
    fn is_zero(&self, tolerance: f64) -> bool {
        self.data.abs() < tolerance
    }

    /// Normalize tangent vector
    fn normalize(&mut self) {
        // Normalizing a scalar doesn't make much sense unless it's a direction.
        // For an angle, this is a no-op.
    }

    /// Return a unit tangent vector in the same direction.
    fn normalized(&self) -> Self {
        if self.data.abs() > f64::EPSILON {
            SO2Tangent::new(self.data.signum())
        } else {
            SO2Tangent::new(0.0)
        }
    }

    fn as_slice(&self) -> &[f64] {
        std::slice::from_ref(&self.data)
    }

    fn from_slice(s: &[f64]) -> Self {
        debug_assert_eq!(s.len(), 1);
        SO2Tangent { data: s[0] }
    }

    /// Small adjoint matrix for SO(2).
    ///
    /// For SO(2), the small adjoint is zero (since it's commutative).
    fn small_adj(&self) -> <SO2 as LieGroup>::JacobianMatrix {
        Matrix1::zeros()
    }

    /// Lie bracket for SO(2).
    ///
    /// For SO(2), the Lie bracket is always zero since it's commutative.
    fn lie_bracket(&self, _other: &Self) -> <SO2 as LieGroup>::TangentVector {
        SO2Tangent::zero()
    }

    /// Check if this tangent vector is approximately equal to another.
    ///
    /// # Arguments
    /// * `other` - The other tangent vector to compare with
    /// * `tolerance` - The tolerance for the comparison
    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        (self.data - other.data).abs() < tolerance
    }

    /// Get the ith generator of the SO(2) Lie algebra.
    ///
    /// # Arguments
    /// * `i` - Index of the generator (must be 0 for SO(2))
    ///
    /// # Returns
    /// The generator matrix
    fn generator(&self, i: usize) -> <SO2 as LieGroup>::LieAlgebra {
        assert_eq!(i, 0, "SO(2) only has one generator (index 0)");
        Matrix2::new(0.0, -1.0, 1.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use std::f64::consts::PI;

    const TOLERANCE: f64 = 1e-12;

    /// Numerically compute Jacobian using central difference
    fn numerical_jacobian<F>(
        func: F,
        point: &DVector<f64>,
        output_dim: usize,
        epsilon: f64,
    ) -> DMatrix<f64>
    where
        F: Fn(&DVector<f64>) -> DVector<f64>,
    {
        let input_dim = point.len();
        let mut jacobian = DMatrix::zeros(output_dim, input_dim);

        for i in 0..input_dim {
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();
            point_plus[i] += epsilon;
            point_minus[i] -= epsilon;

            let output_plus = func(&point_plus);
            let output_minus = func(&point_minus);
            let derivative = (output_plus - output_minus) / (2.0 * epsilon);

            jacobian.set_column(i, &derivative);
        }

        jacobian
    }

    #[test]
    fn test_so2_identity() {
        let so2 = SO2::identity();
        assert!((so2.angle() - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_so2_inverse() {
        let so2 = SO2::from_angle(PI / 4.0);
        let so2_inv = so2.inverse(None);
        assert!((so2_inv.angle() + PI / 4.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_so2_compose() {
        let so2_a = SO2::from_angle(PI / 4.0);
        let so2_b = SO2::from_angle(PI / 2.0);
        let composed = so2_a.compose(&so2_b, None, None);
        assert!((composed.angle() - (3.0 * PI / 4.0)).abs() < TOLERANCE);
    }

    #[test]
    fn test_so2_exp_log_consistency() {
        let angle = PI / 4.0;
        let tangent = SO2Tangent::new(angle);
        let so2 = tangent.exp(None);
        let recovered_tangent = so2.log(None);

        assert!((tangent.angle() - recovered_tangent.angle()).abs() < 1e-10);
    }

    // New tests for the additional functions

    #[test]
    fn test_so2_vee() {
        let so2 = SO2::from_angle(PI / 3.0);
        let tangent_log = so2.log(None);
        let tangent_vee = so2.vee();

        assert!((tangent_log.angle() - tangent_vee.angle()).abs() < 1e-10);
    }

    #[test]
    fn test_so2_is_approx() {
        let so2_1 = SO2::from_angle(PI / 4.0);
        let so2_2 = SO2::from_angle(PI / 4.0 + 1e-12);
        let so2_3 = SO2::from_angle(PI / 2.0);

        assert!(so2_1.is_approx(&so2_1, 1e-10));
        assert!(so2_1.is_approx(&so2_2, 1e-10));
        assert!(!so2_1.is_approx(&so2_3, 1e-10));
    }

    #[test]
    fn test_so2_tangent_small_adj() {
        let tangent = SO2Tangent::new(PI / 6.0);
        let small_adj = tangent.small_adj();

        // For SO(2), small adjoint should be zero (commutative group)
        assert!((small_adj[(0, 0)]).abs() < 1e-10);
    }

    #[test]
    fn test_so2_tangent_lie_bracket() {
        let tangent_a = SO2Tangent::new(0.1);
        let tangent_b = SO2Tangent::new(0.2);

        let bracket = tangent_a.lie_bracket(&tangent_b);

        // For SO(2), Lie bracket should be zero (commutative group)
        assert!(bracket.is_zero(1e-10));

        // Anti-symmetry test: [a,b] = -[b,a]
        let bracket_ba = tangent_b.lie_bracket(&tangent_a);
        assert!(bracket.lie_bracket(&tangent_b).is_zero(1e-10)); // [a,a] = 0

        // Since SO(2) is commutative, both should be zero
        assert!(bracket_ba.is_zero(1e-10));
    }

    #[test]
    fn test_so2_tangent_is_approx() {
        let tangent_1 = SO2Tangent::new(0.5);
        let tangent_2 = SO2Tangent::new(0.5 + 1e-12);
        let tangent_3 = SO2Tangent::new(1.0);

        assert!(tangent_1.is_approx(&tangent_1, 1e-10));
        assert!(tangent_1.is_approx(&tangent_2, 1e-10));
        assert!(!tangent_1.is_approx(&tangent_3, 1e-10));
    }

    #[test]
    fn test_so2_generator() {
        let tangent = SO2Tangent::new(1.0);
        let generator = tangent.generator(0);

        // SO(2) generator should be the skew-symmetric matrix
        let expected = Matrix2::new(0.0, -1.0, 1.0, 0.0);

        assert!((generator - expected).norm() < 1e-10);
    }

    #[test]
    #[should_panic]
    fn test_so2_generator_invalid_index() {
        let tangent = SO2Tangent::new(1.0);
        let _generator = tangent.generator(1); // Should panic for SO(2)
    }

    #[test]
    fn test_so2_bracket_hat_relationship() {
        let a = SO2Tangent::new(0.1);
        let b = SO2Tangent::new(0.2);

        // For SO(2): [a,b]^ = a^ * b^ - b^ * a^ should be zero (commutative)
        let bracket_hat = a.lie_bracket(&b).hat();
        let expected = a.hat() * b.hat() - b.hat() * a.hat();

        assert!((bracket_hat - expected).norm() < 1e-10);
        assert!(expected.norm() < 1e-10); // Should be zero for SO(2)
    }

    #[test]
    fn test_so2_right_jacobian_numerical() {
        let epsilon = 1e-7;
        let tolerance = 1e-4;

        let tangent = SO2Tangent::new(0.5);
        let jr_analytical = tangent.right_jacobian();

        // Numerical Jacobian through exp-log round trip
        let angle_vec = DVector::from_vec(vec![tangent.angle()]);
        let jr_numerical = numerical_jacobian(
            |theta| {
                let tang = SO2Tangent::new(theta[0]);
                let so2 = tang.exp(None);
                let log_result = so2.log(None);
                DVector::from_vec(vec![log_result.angle()])
            },
            &angle_vec,
            1,
            epsilon,
        );

        assert!(
            (jr_analytical - &jr_numerical).norm() < tolerance,
            "Right Jacobian mismatch: analytical = {}, numerical = {}",
            jr_analytical,
            jr_numerical
        );
    }

    #[test]
    fn test_so2_left_jacobian_numerical() {
        let epsilon = 1e-7;
        let tolerance = 1e-4;

        let tangent = SO2Tangent::new(0.5);
        let jl_analytical = tangent.left_jacobian();

        let angle_vec = DVector::from_vec(vec![tangent.angle()]);
        let jl_numerical = numerical_jacobian(
            |theta| {
                let tang = SO2Tangent::new(theta[0]);
                let so2 = tang.exp(None);
                let log_result = so2.log(None);
                DVector::from_vec(vec![log_result.angle()])
            },
            &angle_vec,
            1,
            epsilon,
        );

        assert!((jl_analytical - jl_numerical).norm() < tolerance);
    }

    // T4: Jacobian Inverse Identity Tests

    #[test]
    fn test_so2_jacobian_inverse_identity() {
        // SO2 Jacobians are scalars (1x1 matrices)
        let tangent = SO2Tangent::new(0.5);
        let jr = tangent.right_jacobian();
        let jr_inv = tangent.right_jacobian_inv();
        let product = jr * jr_inv;

        assert!((product[(0, 0)] - 1.0).abs() < 1e-10);

        // Also test left Jacobian
        let jl = tangent.left_jacobian();
        let jl_inv = tangent.left_jacobian_inv();
        let product_left = jl * jl_inv;

        assert!((product_left[(0, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_so2_display() {
        let r = SO2::from_angle(0.5);
        let s = format!("{r}");
        assert!(!s.is_empty(), "Display should produce output, got: {s}");

        let t = SO2Tangent::new(1.2);
        let st = format!("{t}");
        assert!(
            !st.is_empty(),
            "Tangent Display should produce output, got: {st}"
        );
    }

    #[test]
    fn test_so2_from_slice_and_back() {
        let angle = 1.0f64;
        let r = SO2::from_param_slice(&[angle]);
        let back = DVector::from_column_slice(r.as_param_slice());
        assert_eq!(back.len(), 1);
        assert!((back[0] - angle).abs() < 1e-9);
    }

    #[test]
    fn test_so2_tangent_from_slice_and_back() {
        let t = SO2Tangent::from_slice(&[0.7f64]);
        let v2 = DVector::from_column_slice(t.as_slice());
        assert!((v2[0] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_so2_rotation_matrix() {
        let r = SO2::from_angle(0.0);
        let mat = r.rotation_matrix();
        assert!((mat[(0, 0)] - 1.0).abs() < 1e-10);
        assert!(mat[(0, 1)].abs() < 1e-10);
    }

    #[test]
    fn test_so2_complex_angle_accessors() {
        let angle = std::f64::consts::FRAC_PI_4;
        let r = SO2::from_angle(angle);
        let c = r.complex();
        assert!((c.re - angle.cos()).abs() < 1e-9);
        assert!((c.im - angle.sin()).abs() < 1e-9);
        assert!((r.angle() - angle).abs() < 1e-9);
    }

    #[test]
    fn test_so2_normalize_is_valid() {
        let mut r = SO2::from_angle(0.3);
        r.normalize();
        assert!(r.is_valid(1e-6));
    }

    #[test]
    fn test_so2_tangent_normalized() {
        let t_pos = SO2Tangent::new(3.0);
        let tn = t_pos.normalized();
        assert!((tn.angle() - 1.0).abs() < 1e-9);

        let t_neg = SO2Tangent::new(-2.0);
        let tn_neg = t_neg.normalized();
        assert!((tn_neg.angle() - (-1.0)).abs() < 1e-9);

        let t_zero = SO2Tangent::new(0.0);
        let tn_zero = t_zero.normalized();
        assert!((tn_zero.angle()).abs() < 1e-9);
    }

    #[test]
    fn test_so2_tangent_is_zero() {
        let zero = SO2Tangent::new(0.0);
        assert!(zero.is_zero(1e-9));
        let nonzero = SO2Tangent::new(0.1);
        assert!(!nonzero.is_zero(1e-9));
    }

    #[test]
    fn test_so2_random() {
        let r = SO2::random();
        assert!(r.is_valid(1e-6));

        let t = SO2Tangent::random();
        let _ = t; // just verify it doesn't panic
    }

    #[test]
    fn test_so2_adjoint_zero_jacobian_identity() {
        let r = SO2::from_angle(0.5);
        let adj = r.adjoint();
        assert!((adj[0] - 1.0).abs() < 1e-10);

        let zj = SO2::zero_jacobian();
        assert!(zj[0].abs() < 1e-10);

        let ji = SO2::jacobian_identity();
        assert!((ji[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_so2_compose_with_jacobians() {
        let r1 = SO2::from_angle(0.3);
        let r2 = SO2::from_angle(0.2);
        let mut j_self = Matrix1::zeros();
        let mut j_other = Matrix1::zeros();
        let result = r1.compose(&r2, Some(&mut j_self), Some(&mut j_other));
        assert!(result.is_valid(1e-9));
        assert!(j_self[0].is_finite());
        assert!(j_other[0].is_finite());
    }

    #[test]
    fn test_so2_log_with_jacobian() {
        let r = SO2::from_angle(0.5);
        let mut jac = Matrix1::zeros();
        let t = r.log(Some(&mut jac));
        assert!((t.angle() - 0.5).abs() < 1e-9);
        assert!((jac[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_so2_inverse_with_jacobian() {
        let r = SO2::from_angle(0.4);
        let mut jac = Matrix1::zeros();
        let inv = r.inverse(Some(&mut jac));
        assert!(inv.is_valid(1e-9));
        assert!(jac[0].is_finite());
    }

    #[test]
    fn test_so2_tangent_hat() {
        let t = SO2Tangent::new(1.0);
        let hat = t.hat();
        // hat should be skew-symmetric: [[0, -theta], [theta, 0]]
        assert!(hat[(0, 0)].abs() < 1e-10);
        assert!((hat[(1, 0)] - 1.0).abs() < 1e-10);
        assert!((hat[(0, 1)] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_so2_tangent_exp_with_jacobian() {
        use crate::Tangent;
        let t = SO2Tangent::new(0.3);
        let mut jac = Matrix1::zeros();
        let r = t.exp(Some(&mut jac));
        assert!(r.is_valid(1e-9));
        assert!((jac[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_so2_tangent_zero() {
        use crate::Tangent;
        let zero = SO2Tangent::zero();
        assert!(zero.is_zero(1e-9));
    }

    #[test]
    fn so2_param_slice_round_trip() {
        let g = SO2::random();
        let recovered = SO2::from_param_slice(g.as_param_slice());
        assert!(g.is_approx(&recovered, 1e-14));
    }

    #[test]
    fn so2_tangent_slice_round_trip() {
        let t = SO2Tangent::random();
        let recovered = SO2Tangent::from_slice(t.as_slice());
        assert!(t.is_approx(&recovered, 1e-14));
    }
}
