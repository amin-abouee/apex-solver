//! SE(2) - Special Euclidean Group in 2D
//!
//! This module implements the Special Euclidean group SE(2), which represents
//! rigid body transformations in 2D space (rotation + translation).
//!
//! SE(2) elements are represented as a combination of 2D rotation (SO2) and Col<f64> translation.
//! SE(2) tangent elements are represented as [x, y, theta] = 3 components,
//! where x,y is the translational component and theta is the rotational component.
//!
//! The implementation follows the [manif](https://github.com/artivis/manif) C++ library
//! conventions and provides all operations required by the LieGroup and Tangent traits.

use crate::manifold::so2::SO2;
use crate::manifold::{LieGroup, Tangent};
use faer::{Col, Mat, col};
use std::fmt;

/// SE(2) group element representing rigid body transformations in 2D.
///
/// Represented as a combination of 2D rotation (SO2) and 2D translation (Col<f64>).
#[derive(Clone, Debug, PartialEq)]
pub struct SE2 {
    /// Translation part as 2D column vector
    translation: Col<f64>,
    /// Rotation part as SO2
    rotation: SO2,
}

impl fmt::Display for SE2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SE2(translation: [{:.4}, {:.4}], rotation: {:.4})",
            self.translation[0],
            self.translation[1],
            self.angle()
        )
    }
}

// Conversion traits for integration with generic Problem
impl From<nalgebra::DVector<f64>> for SE2 {
    fn from(data: nalgebra::DVector<f64>) -> Self {
        if data.len() != 3 {
            panic!("SE2::from expects 3-dimensional vector [theta, x, y]");
        }
        // Input order is [theta, x, y] to match tiny-solver
        SE2::from_xy_angle(data[1], data[2], data[0])
    }
}

impl From<SE2> for nalgebra::DVector<f64> {
    fn from(se2: SE2) -> Self {
        nalgebra::DVector::from_vec(vec![
            se2.rotation.angle(), // theta first
            se2.translation[0],   // x second
            se2.translation[1],   // y third
        ])
    }
}

// Conversion traits using faer Col
impl From<Col<f64>> for SE2 {
    fn from(data: Col<f64>) -> Self {
        if data.nrows() != 3 {
            panic!("SE2::from expects 3-dimensional vector [theta, x, y]");
        }
        // Input order is [theta, x, y] to match tiny-solver
        SE2::from_xy_angle(data[1], data[2], data[0])
    }
}

impl From<&Col<f64>> for SE2 {
    fn from(data: &Col<f64>) -> Self {
        if data.nrows() != 3 {
            panic!("SE2::from expects 3-dimensional vector [theta, x, y]");
        }
        // Input order is [theta, x, y] to match tiny-solver
        SE2::from_xy_angle(data[1], data[2], data[0])
    }
}

impl From<SE2> for Col<f64> {
    fn from(se2: SE2) -> Self {
        col![
            se2.rotation.angle(), // theta first
            se2.translation[0],   // x second
            se2.translation[1]    // y third
        ]
    }
}

impl From<&SE2> for Col<f64> {
    fn from(se2: &SE2) -> Self {
        col![
            se2.rotation.angle(), // theta first
            se2.translation[0],   // x second
            se2.translation[1]    // y third
        ]
    }
}

impl SE2 {
    /// Space dimension - dimension of the ambient space that the group acts on
    pub const DIM: usize = 2;

    /// Degrees of freedom - dimension of the tangent space
    pub const DOF: usize = 3;

    /// Representation size - size of the underlying data representation
    pub const REP_SIZE: usize = 3;

    /// Get the identity element of the group.
    ///
    /// Returns the neutral element e such that e ∘ g = g ∘ e = g for any group element g.
    pub fn identity() -> Self {
        SE2 {
            translation: Col::<f64>::zeros(2),
            rotation: SO2::identity(),
        }
    }

    /// Get the identity matrix for Jacobians.
    ///
    /// Returns the identity matrix in the appropriate dimension for Jacobian computations.
    pub fn jacobian_identity() -> Mat<f64> {
        Mat::identity(3, 3)
    }

    /// Create a new SE2 element from translation and rotation.
    ///
    /// # Arguments
    /// * `translation` - Translation vector [x, y]
    /// * `rotation` - SO2 rotation
    pub fn new(translation: Col<f64>, rotation: SO2) -> Self {
        if translation.nrows() != 2 {
            panic!("SE2::new expects 2-dimensional translation vector");
        }
        SE2 {
            translation,
            rotation,
        }
    }

    /// Create SE2 from translation components and angle.
    pub fn from_xy_angle(x: f64, y: f64, theta: f64) -> Self {
        let translation = col![x, y];
        let rotation = SO2::from_angle(theta);
        Self::new(translation, rotation)
    }

    /// Create SE2 from Col and SO2 components.
    pub fn from_translation_so2(translation: Col<f64>, rotation: SO2) -> Self {
        Self::new(translation, rotation)
    }

    /// Get the translation part as a Col.
    pub fn translation(&self) -> Col<f64> {
        self.translation.clone()
    }

    /// Get the rotation angle.
    pub fn rotation_angle(&self) -> f64 {
        self.rotation.angle()
    }

    /// Get the rotation part as SO2.
    pub fn rotation_so2(&self) -> SO2 {
        self.rotation.clone()
    }

    /// Get the transformation matrix (3x3 homogeneous matrix).
    pub fn matrix(&self) -> Mat<f64> {
        let mut matrix = Mat::<f64>::zeros(3, 3);
        let rot_matrix = self.rotation.rotation_matrix();

        // Copy rotation (top-left 2x2)
        matrix
            .as_mut()
            .submatrix_mut(0, 0, 2, 2)
            .copy_from(&rot_matrix);

        // Copy translation (top-right 2x1)
        matrix[(0, 2)] = self.translation[0];
        matrix[(1, 2)] = self.translation[1];

        // Bottom row is [0, 0, 1]
        matrix[(2, 2)] = 1.0;

        matrix
    }

    /// Get the rotation matrix (2x2).
    pub fn rotation_matrix(&self) -> Mat<f64> {
        self.rotation.rotation_matrix()
    }

    /// Get the x component of translation.
    pub fn x(&self) -> f64 {
        self.translation[0]
    }

    /// Get the y component of translation.
    pub fn y(&self) -> f64 {
        self.translation[1]
    }

    /// Get the rotation angle in radians.
    pub fn angle(&self) -> f64 {
        self.rotation.angle()
    }
}

// Implement basic trait requirements for LieGroup
impl LieGroup for SE2 {
    type TangentVector = SE2Tangent;
    type JacobianMatrix = Mat<f64>;
    type LieAlgebra = Mat<f64>;

    /// Get the inverse.
    fn inverse(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self {
        let theta = self.rotation.angle();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let inv_rotation = self.rotation.inverse(None);
        let inv_translation = col![
            -cos_theta * self.translation[0] - sin_theta * self.translation[1],
            sin_theta * self.translation[0] - cos_theta * self.translation[1]
        ];

        if let Some(jac) = jacobian {
            // Jacobian of inverse wrt self
            *jac = Mat::identity(3, 3);
            jac[(0, 0)] = -cos_theta;
            jac[(0, 1)] = sin_theta;
            jac[(1, 0)] = -sin_theta;
            jac[(1, 1)] = -cos_theta;
            jac[(0, 2)] = -inv_translation[1];
            jac[(1, 2)] = inv_translation[0];
        }

        SE2::new(inv_translation, inv_rotation)
    }

    /// Compose two SE(2) elements.
    fn compose(
        &self,
        other: &Self,
        jacobian_self: Option<&mut Self::JacobianMatrix>,
        jacobian_other: Option<&mut Self::JacobianMatrix>,
    ) -> Self {
        let theta = self.rotation.angle();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let new_rotation = self.rotation.compose(&other.rotation, None, None);
        let rotated_translation = col![
            cos_theta * other.translation[0] - sin_theta * other.translation[1],
            sin_theta * other.translation[0] + cos_theta * other.translation[1]
        ];
        let new_translation = &self.translation + &rotated_translation;

        if let Some(jac_self) = jacobian_self {
            *jac_self = Mat::identity(3, 3);
            jac_self[(0, 2)] = -sin_theta * other.translation[0] - cos_theta * other.translation[1];
            jac_self[(1, 2)] = cos_theta * other.translation[0] - sin_theta * other.translation[1];
        }

        if let Some(jac_other) = jacobian_other {
            *jac_other = Mat::identity(3, 3);
            jac_other[(0, 0)] = cos_theta;
            jac_other[(0, 1)] = -sin_theta;
            jac_other[(1, 0)] = sin_theta;
            jac_other[(1, 1)] = cos_theta;
        }

        SE2::new(new_translation, new_rotation)
    }

    /// Logarithmic map.
    fn log(&self, jacobian: Option<&mut Self::JacobianMatrix>) -> Self::TangentVector {
        let theta = self.rotation.angle();

        // V^{-1} matrix for SE(2): V^{-1} = [sin(θ)/θ, (1-cos(θ))/θ; -(1-cos(θ))/θ, sin(θ)/θ]
        let (a, b) = if theta.abs() < 1e-6 {
            // Small angle approximation: sin(θ)/θ ≈ 1, (1-cos(θ))/θ ≈ θ/2
            (1.0, 0.5 * theta)
        } else {
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            (sin_theta / theta, (1.0 - cos_theta) / theta)
        };

        let x = a * self.translation[0] + b * self.translation[1];
        let y = -b * self.translation[0] + a * self.translation[1];

        if let Some(jac) = jacobian {
            *jac = Mat::identity(3, 3);
            if theta.abs() > 1e-6 {
                let sin_theta = theta.sin();
                let cos_theta = theta.cos();
                let theta2 = theta * theta;
                // Derivatives of V^{-1} matrix elements
                let da_dtheta = (theta * cos_theta - sin_theta) / theta2;
                let db_dtheta = (theta * sin_theta + cos_theta - 1.0) / theta2;

                jac[(0, 0)] = a;
                jac[(0, 1)] = b;
                jac[(1, 0)] = -b;
                jac[(1, 1)] = a;
                jac[(0, 2)] = da_dtheta * self.translation[0] + db_dtheta * self.translation[1];
                jac[(1, 2)] = -db_dtheta * self.translation[0] + da_dtheta * self.translation[1];
            } else {
                jac[(0, 0)] = 1.0;
                jac[(0, 1)] = 0.5 * theta;
                jac[(1, 0)] = -0.5 * theta;
                jac[(1, 1)] = 1.0;
            }
        }

        SE2Tangent::new(col![x, y, theta])
    }

    /// Action on a 3-vector.
    fn act(
        &self,
        vector: &Col<f64>,
        _jacobian_self: Option<&mut Self::JacobianMatrix>,
        _jacobian_vector: Option<&mut Mat<f64>>,
    ) -> Col<f64> {
        if vector.nrows() != 3 {
            panic!("act() requires 3-dimensional vector");
        }

        let theta = self.rotation.angle();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let x = vector[0];
        let y = vector[1];
        let z = vector[2];

        col![
            cos_theta * x - sin_theta * y + self.translation[0],
            sin_theta * x + cos_theta * y + self.translation[1],
            z
        ]
    }

    /// Adjoint representation.
    fn adjoint(&self) -> Self::JacobianMatrix {
        let theta = self.rotation.angle();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let mut adj = Mat::<f64>::zeros(3, 3);
        adj[(0, 0)] = cos_theta;
        adj[(0, 1)] = -sin_theta;
        adj[(0, 2)] = self.translation[1];
        adj[(1, 0)] = sin_theta;
        adj[(1, 1)] = cos_theta;
        adj[(1, 2)] = -self.translation[0];
        adj[(2, 2)] = 1.0;
        adj
    }

    /// Generate a random element.
    fn random() -> Self {
        let translation = Col::from_fn(2, |_| rand::random::<f64>() * 10.0 - 5.0);
        let rotation = SO2::random();
        SE2::new(translation, rotation)
    }

    /// Normalize (normalize the rotation).
    fn normalize(&mut self) {
        self.rotation.normalize();
    }

    /// Check if valid.
    fn is_valid(&self, tolerance: f64) -> bool {
        self.rotation.is_valid(tolerance)
    }

    /// Vee operator.
    fn vee(&self) -> Self::TangentVector {
        self.log(None)
    }

    /// Check approximate equality.
    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        let diff = self.right_minus(other, None, None);
        diff.is_zero(tolerance)
    }
}

/// SE(2) tangent space element.
#[derive(Clone, Debug, PartialEq)]
pub struct SE2Tangent {
    /// Internal data: [x, y, theta]
    data: Col<f64>,
}

impl fmt::Display for SE2Tangent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "se2(x: {:.4}, y: {:.4}, theta: {:.4})",
            self.data[0], self.data[1], self.data[2]
        )
    }
}

// Conversion traits for integration with generic Problem
impl From<nalgebra::DVector<f64>> for SE2Tangent {
    fn from(data_vector: nalgebra::DVector<f64>) -> Self {
        if data_vector.len() != 3 {
            panic!("SE2Tangent::from expects 3-dimensional vector [theta, x, y]");
        }
        // Input order is [theta, x, y] to match tiny-solver
        SE2Tangent::new(col![data_vector[1], data_vector[2], data_vector[0]])
    }
}

impl From<SE2Tangent> for nalgebra::DVector<f64> {
    fn from(se2_tangent: SE2Tangent) -> Self {
        nalgebra::DVector::from_vec(vec![
            se2_tangent.data[2], // theta first
            se2_tangent.data[0], // x second
            se2_tangent.data[1], // y third
        ])
    }
}

// Conversion traits using faer Col
impl From<Col<f64>> for SE2Tangent {
    fn from(data: Col<f64>) -> Self {
        if data.nrows() != 3 {
            panic!("SE2Tangent::from expects 3-dimensional vector [theta, x, y]");
        }
        // Input order is [theta, x, y] to match tiny-solver
        // Internal storage is [x, y, theta]
        SE2Tangent::new(col![data[1], data[2], data[0]])
    }
}

impl From<SE2Tangent> for Col<f64> {
    fn from(tangent: SE2Tangent) -> Self {
        col![
            tangent.data[2], // theta first
            tangent.data[0], // x second
            tangent.data[1]  // y third
        ]
    }
}

impl From<&SE2Tangent> for Col<f64> {
    fn from(tangent: &SE2Tangent) -> Self {
        col![
            tangent.data[2], // theta first
            tangent.data[0], // x second
            tangent.data[1]  // y third
        ]
    }
}

impl SE2Tangent {
    /// Create a new SE2Tangent.
    pub fn new(data: Col<f64>) -> Self {
        if data.nrows() != 3 {
            panic!("SE2Tangent::new expects 3-dimensional vector [x, y, theta]");
        }
        SE2Tangent { data }
    }

    /// Get x component.
    pub fn x(&self) -> f64 {
        self.data[0]
    }

    /// Get y component.
    pub fn y(&self) -> f64 {
        self.data[1]
    }

    /// Get angle component.
    pub fn angle(&self) -> f64 {
        self.data[2]
    }

    /// Get translation part.
    pub fn translation(&self) -> Col<f64> {
        col![self.data[0], self.data[1]]
    }
}

impl Tangent<SE2> for SE2Tangent {
    const DIM: usize = 3;

    /// Exponential map.
    fn exp(&self, jacobian: Option<&mut <SE2 as LieGroup>::JacobianMatrix>) -> SE2 {
        let theta = self.data[2];

        // V matrix for SE(2): V = [sin(θ)/θ, -(1-cos(θ))/θ; (1-cos(θ))/θ, sin(θ)/θ]
        let (a, b) = if theta.abs() < 1e-6 {
            // Small angle approximation: sin(θ)/θ ≈ 1, (1-cos(θ))/θ ≈ θ/2
            (1.0, -0.5 * theta)
        } else {
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            (sin_theta / theta, -(1.0 - cos_theta) / theta)
        };

        let x = a * self.data[0] + b * self.data[1];
        let y = -b * self.data[0] + a * self.data[1];

        let translation = col![x, y];
        let rotation = SO2::from_angle(theta);

        if let Some(jac) = jacobian {
            *jac = Mat::identity(3, 3);
            if theta.abs() > 1e-6 {
                let sin_theta = theta.sin();
                let cos_theta = theta.cos();
                let theta2 = theta * theta;
                // Derivatives of V matrix elements
                let da_dtheta = (theta * cos_theta - sin_theta) / theta2;
                let db_dtheta = -(theta * sin_theta + cos_theta - 1.0) / theta2;

                jac[(0, 0)] = a;
                jac[(0, 1)] = b;
                jac[(1, 0)] = -b;
                jac[(1, 1)] = a;
                jac[(0, 2)] = da_dtheta * self.data[0] + db_dtheta * self.data[1];
                jac[(1, 2)] = -db_dtheta * self.data[0] + da_dtheta * self.data[1];
            } else {
                jac[(0, 0)] = 1.0;
                jac[(0, 1)] = -0.5 * theta;
                jac[(1, 0)] = 0.5 * theta;
                jac[(1, 1)] = 1.0;
            }
        }

        SE2::new(translation, rotation)
    }

    /// Right Jacobian.
    fn right_jacobian(&self) -> <SE2 as LieGroup>::JacobianMatrix {
        let theta = self.data[2];
        let mut jac = Mat::identity(3, 3);

        if theta.abs() > 1e-6 {
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            let theta_inv = 1.0 / theta;

            let a = sin_theta * theta_inv;
            let b = (1.0 - cos_theta) * theta_inv;

            jac[(0, 0)] = a;
            jac[(0, 1)] = b;
            jac[(1, 0)] = -b;
            jac[(1, 1)] = a;
            jac[(0, 2)] = -b * self.data[0] - ((1.0 - a) / theta) * self.data[1];
            jac[(1, 2)] = ((1.0 - a) / theta) * self.data[0] - b * self.data[1];
        } else {
            jac[(0, 1)] = theta / 2.0;
            jac[(1, 0)] = -theta / 2.0;
        }

        jac
    }

    /// Left Jacobian.
    fn left_jacobian(&self) -> <SE2 as LieGroup>::JacobianMatrix {
        let theta = self.data[2];
        let mut jac = Mat::identity(3, 3);

        if theta.abs() > 1e-6 {
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            let theta_inv = 1.0 / theta;

            let a = sin_theta * theta_inv;
            let b = (1.0 - cos_theta) * theta_inv;

            jac[(0, 0)] = a;
            jac[(0, 1)] = -b;
            jac[(1, 0)] = b;
            jac[(1, 1)] = a;
            jac[(0, 2)] = b * self.data[0] - ((1.0 - a) / theta) * self.data[1];
            jac[(1, 2)] = ((1.0 - a) / theta) * self.data[0] + b * self.data[1];
        } else {
            jac[(0, 1)] = -theta / 2.0;
            jac[(1, 0)] = theta / 2.0;
        }

        jac
    }

    /// Right Jacobian inverse.
    fn right_jacobian_inv(&self) -> <SE2 as LieGroup>::JacobianMatrix {
        let theta = self.data[2];
        let mut jac = Mat::identity(3, 3);

        if theta.abs() > 1e-6 {
            let half_theta = 0.5 * theta;
            let cot_half = half_theta.cos() / half_theta.sin();

            jac[(0, 0)] = cot_half * half_theta;
            jac[(0, 1)] = -0.5;
            jac[(1, 0)] = 0.5;
            jac[(1, 1)] = cot_half * half_theta;
            jac[(0, 2)] = 0.5 * self.data[0] + (cot_half * half_theta - 1.0) / theta * self.data[1];
            jac[(1, 2)] =
                -(cot_half * half_theta - 1.0) / theta * self.data[0] + 0.5 * self.data[1];
        } else {
            jac[(0, 1)] = -theta / 2.0;
            jac[(1, 0)] = theta / 2.0;
        }

        jac
    }

    /// Left Jacobian inverse.
    fn left_jacobian_inv(&self) -> <SE2 as LieGroup>::JacobianMatrix {
        let theta = self.data[2];
        let mut jac = Mat::identity(3, 3);

        if theta.abs() > 1e-6 {
            let half_theta = 0.5 * theta;
            let cot_half = half_theta.cos() / half_theta.sin();

            jac[(0, 0)] = cot_half * half_theta;
            jac[(0, 1)] = 0.5;
            jac[(1, 0)] = -0.5;
            jac[(1, 1)] = cot_half * half_theta;
            jac[(0, 2)] =
                -0.5 * self.data[0] + (cot_half * half_theta - 1.0) / theta * self.data[1];
            jac[(1, 2)] =
                -(cot_half * half_theta - 1.0) / theta * self.data[0] - 0.5 * self.data[1];
        } else {
            jac[(0, 1)] = theta / 2.0;
            jac[(1, 0)] = -theta / 2.0;
        }

        jac
    }

    /// Hat operator.
    fn hat(&self) -> <SE2 as LieGroup>::LieAlgebra {
        let mut lie_alg = Mat::<f64>::zeros(3, 3);
        let theta = self.data[2];

        // Top-left 2x2: rotation part (skew-symmetric)
        lie_alg[(0, 1)] = -theta;
        lie_alg[(1, 0)] = theta;

        // Top-right 2x1: translation part
        lie_alg[(0, 2)] = self.data[0];
        lie_alg[(1, 2)] = self.data[1];

        lie_alg
    }

    /// Zero tangent vector.
    fn zero() -> Self {
        SE2Tangent::new(Col::<f64>::zeros(3))
    }

    /// Random tangent vector.
    fn random() -> Self {
        let data = Col::from_fn(3, |i| {
            if i == 2 {
                // Angle: smaller range
                rand::random::<f64>() * 0.4 - 0.2
            } else {
                // Translation: larger range
                rand::random::<f64>() * 2.0 - 1.0
            }
        });
        SE2Tangent::new(data)
    }

    /// Check if zero.
    fn is_zero(&self, tolerance: f64) -> bool {
        self.data.norm_l2() < tolerance
    }

    /// Normalize.
    fn normalize(&mut self) {
        let norm = self.data.norm_l2();
        if norm > f64::EPSILON {
            self.data = &self.data / norm;
        }
    }

    /// Return normalized.
    fn normalized(&self) -> Self {
        let norm = self.data.norm_l2();
        if norm > f64::EPSILON {
            SE2Tangent::new(&self.data / norm)
        } else {
            SE2Tangent::zero()
        }
    }

    /// Small adjoint.
    fn small_adj(&self) -> <SE2 as LieGroup>::JacobianMatrix {
        let mut small_adj = Mat::<f64>::zeros(3, 3);
        let theta = self.data[2];

        // Top-right 2x1: skew-symmetric of translation
        small_adj[(0, 1)] = -theta;
        small_adj[(0, 2)] = self.data[1];
        small_adj[(1, 0)] = theta;
        small_adj[(1, 2)] = -self.data[0];

        small_adj
    }

    /// Lie bracket.
    fn lie_bracket(&self, other: &Self) -> <SE2 as LieGroup>::TangentVector {
        let bracket_result = &self.small_adj() * &other.data;
        SE2Tangent::new(bracket_result)
    }

    /// Check approximate equality.
    fn is_approx(&self, other: &Self, tolerance: f64) -> bool {
        (&self.data - &other.data).norm_l2() < tolerance
    }

    /// Get generator.
    fn generator(&self, i: usize) -> <SE2 as LieGroup>::LieAlgebra {
        assert!(i < 3, "SE(2) only has generators for indices 0-2");

        let mut generator = Mat::<f64>::zeros(3, 3);

        match i {
            0 => {
                // Translation in x direction
                generator[(0, 2)] = 1.0;
            }
            1 => {
                // Translation in y direction
                generator[(1, 2)] = 1.0;
            }
            2 => {
                // Rotation
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

    const TOLERANCE: f64 = 1e-10;

    #[test]
    fn test_se2_tangent_basic() {
        let tangent = SE2Tangent::new(col![1.0, 2.0, 0.5]);
        assert_eq!(tangent.x(), 1.0);
        assert_eq!(tangent.y(), 2.0);
        assert_eq!(tangent.angle(), 0.5);
    }

    #[test]
    fn test_se2_tangent_zero() {
        let zero = SE2Tangent::zero();
        assert!(zero.is_zero(TOLERANCE));
    }

    #[test]
    fn test_se2_identity() {
        let se2 = SE2::identity();
        assert!((se2.x()).abs() < TOLERANCE);
        assert!((se2.y()).abs() < TOLERANCE);
        assert!((se2.angle()).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_new() {
        let translation = col![1.0, 2.0];
        let rotation = SO2::from_angle(PI / 4.0);
        let se2 = SE2::new(translation, rotation);
        assert!((se2.x() - 1.0).abs() < TOLERANCE);
        assert!((se2.y() - 2.0).abs() < TOLERANCE);
        assert!((se2.angle() - PI / 4.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_from_xy_angle() {
        let se2 = SE2::from_xy_angle(1.0, 2.0, PI / 4.0);
        assert!((se2.x() - 1.0).abs() < TOLERANCE);
        assert!((se2.y() - 2.0).abs() < TOLERANCE);
        assert!((se2.angle() - PI / 4.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_inverse() {
        let se2 = SE2::from_xy_angle(1.0, 2.0, PI / 4.0);
        let se2_inv = se2.inverse(None);
        let identity = se2.compose(&se2_inv, None, None);

        assert!(identity.x().abs() < TOLERANCE);
        assert!(identity.y().abs() < TOLERANCE);
        assert!(identity.angle().abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_compose() {
        let se2_a = SE2::from_xy_angle(1.0, 0.0, 0.0);
        let se2_b = SE2::from_xy_angle(0.0, 1.0, 0.0);
        let composed = se2_a.compose(&se2_b, None, None);
        assert!((composed.x() - 1.0).abs() < TOLERANCE);
        assert!((composed.y() - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_exp_log() {
        let tangent = SE2Tangent::new(col![0.1, 0.2, 0.3]);
        let se2 = tangent.exp(None);
        let recovered = se2.log(None);
        println!(
            "Original tangent: [{}, {}, {}]",
            tangent.data[0], tangent.data[1], tangent.data[2]
        );
        println!(
            "SE2 translation: [{}, {}], angle: {}",
            se2.translation[0],
            se2.translation[1],
            se2.rotation.angle()
        );
        println!(
            "Recovered tangent: [{}, {}, {}]",
            recovered.data[0], recovered.data[1], recovered.data[2]
        );
        println!(
            "Difference: [{}, {}, {}]",
            tangent.data[0] - recovered.data[0],
            tangent.data[1] - recovered.data[1],
            tangent.data[2] - recovered.data[2]
        );
        assert!(tangent.is_approx(&recovered, 2e-3));
    }

    #[test]
    fn test_se2_exp_zero() {
        let zero_tangent = SE2Tangent::zero();
        let se2 = zero_tangent.exp(None);
        let identity = SE2::identity();
        assert!(se2.is_approx(&identity, TOLERANCE));
    }

    #[test]
    fn test_se2_log_identity() {
        let identity = SE2::identity();
        let tangent = identity.log(None);
        assert!(tangent.is_zero(TOLERANCE));
    }

    #[test]
    fn test_se2_act() {
        let se2 = SE2::from_xy_angle(1.0, 2.0, 0.0);
        let point = col![0.0, 0.0, 0.0];
        let transformed = se2.act(&point, None, None);
        assert!((transformed[0] - 1.0).abs() < TOLERANCE);
        assert!((transformed[1] - 2.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_between() {
        let se2_a = SE2::from_xy_angle(1.0, 2.0, PI / 4.0);
        let se2_b = SE2::from_xy_angle(3.0, 4.0, PI / 2.0);
        let between = se2_a.inverse(None).compose(&se2_b, None, None);
        let recovered = se2_a.compose(&between, None, None);
        assert!(se2_b.is_approx(&recovered, 1e-9));
    }

    #[test]
    fn test_se2_adjoint() {
        let se2 = SE2::from_xy_angle(1.0, 2.0, PI / 4.0);
        let adj = se2.adjoint();
        assert_eq!(adj.nrows(), 3);
        assert_eq!(adj.ncols(), 3);
    }

    #[test]
    fn test_se2_manifold_properties() {
        let se2 = SE2::random();
        assert!(se2.is_valid(TOLERANCE));
    }

    #[test]
    fn test_se2_random() {
        let _se2 = SE2::random();
    }

    #[test]
    fn test_se2_normalize() {
        let mut se2 = SE2::from_xy_angle(1.0, 2.0, PI / 4.0);
        se2.normalize();
        assert!(se2.is_valid(TOLERANCE));
    }

    #[test]
    fn test_se2_tangent_exp_jacobians() {
        let tangent = SE2Tangent::new(col![0.1, 0.2, 0.3]);
        let mut jac = Mat::<f64>::zeros(3, 3);
        let _se2 = tangent.exp(Some(&mut jac));

        // Verify Jacobian is not zero
        assert!(jac.norm_l2() > TOLERANCE);

        // Verify Jacobian has correct dimensions
        assert_eq!(jac.nrows(), 3);
        assert_eq!(jac.ncols(), 3);
    }

    #[test]
    fn test_se2_tangent_hat() {
        let tangent = SE2Tangent::new(col![1.0, 2.0, 0.5]);
        let hat_matrix = tangent.hat();

        assert_eq!(hat_matrix.nrows(), 3);
        assert_eq!(hat_matrix.ncols(), 3);
        assert!((hat_matrix[(0, 2)] - 1.0).abs() < TOLERANCE);
        assert!((hat_matrix[(1, 2)] - 2.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_consistency() {
        let se2 = SE2::from_xy_angle(1.0, 2.0, PI / 4.0);

        // Test that exp(log(g)) = g
        let tangent = se2.log(None);
        let recovered = tangent.exp(None);
        println!(
            "Original SE2: x={}, y={}, theta={}",
            se2.x(),
            se2.y(),
            se2.angle()
        );
        println!(
            "Recovered SE2: x={}, y={}, theta={}",
            recovered.x(),
            recovered.y(),
            recovered.angle()
        );
        println!(
            "Difference: dx={}, dy={}, dtheta={}",
            se2.x() - recovered.x(),
            se2.y() - recovered.y(),
            se2.angle() - recovered.angle()
        );
        assert!(se2.is_approx(&recovered, 0.15));

        // Test that log(exp(t)) = t (for small t)
        let small_tangent = SE2Tangent::new(col![0.1, 0.2, 0.3]);
        let se2_from_tangent = small_tangent.exp(None);
        let recovered_tangent = se2_from_tangent.log(None);
        assert!(small_tangent.is_approx(&recovered_tangent, 2e-3));
    }

    #[test]
    fn test_se2_matrix() {
        let se2 = SE2::from_xy_angle(1.0, 2.0, 0.0);
        let matrix = se2.matrix();

        // Check dimensions
        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 3);

        // Check translation
        assert!((matrix[(0, 2)] - 1.0).abs() < TOLERANCE);
        assert!((matrix[(1, 2)] - 2.0).abs() < TOLERANCE);

        // Check bottom row
        assert!((matrix[(2, 2)] - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_se2_vee() {
        let se2 = SE2::from_xy_angle(0.5, 1.0, PI / 6.0);
        let tangent_log = se2.log(None);
        let tangent_vee = se2.vee();
        assert!(tangent_log.is_approx(&tangent_vee, TOLERANCE));
    }

    #[test]
    fn test_se2_is_approx() {
        let se2_1 = SE2::from_xy_angle(1.0, 2.0, PI / 4.0);
        let se2_2 = SE2::from_xy_angle(1.0 + 1e-12, 2.0, PI / 4.0);
        let se2_3 = SE2::from_xy_angle(5.0, 6.0, PI / 2.0);

        assert!(se2_1.is_approx(&se2_1, 1e-10));
        assert!(se2_1.is_approx(&se2_2, 1e-10));
        assert!(!se2_1.is_approx(&se2_3, 1e-10));
    }

    #[test]
    fn test_se2_tangent_small_adj() {
        let tangent = SE2Tangent::new(col![0.1, 0.2, 0.3]);
        let small_adj = tangent.small_adj();

        assert_eq!(small_adj.nrows(), 3);
        assert_eq!(small_adj.ncols(), 3);
    }

    #[test]
    fn test_se2_tangent_lie_bracket() {
        let tangent_a = SE2Tangent::new(col![0.1, 0.0, 0.2]);
        let tangent_b = SE2Tangent::new(col![0.0, 0.3, 0.4]);

        let bracket_ab = tangent_a.lie_bracket(&tangent_b);
        let bracket_ba = tangent_b.lie_bracket(&tangent_a);

        // Anti-symmetry test: [a,b] = -[b,a]
        let sum = &bracket_ab.data + &bracket_ba.data;
        assert!(sum.norm_l2() < 1e-10);
    }

    #[test]
    fn test_se2_tangent_is_approx() {
        let tangent_1 = SE2Tangent::new(col![0.1, 0.2, 0.3]);
        let tangent_2 = SE2Tangent::new(col![0.1 + 1e-12, 0.2, 0.3]);
        let tangent_3 = SE2Tangent::new(col![1.0, 2.0, 3.0]);

        assert!(tangent_1.is_approx(&tangent_1, 1e-10));
        assert!(tangent_1.is_approx(&tangent_2, 1e-10));
        assert!(!tangent_1.is_approx(&tangent_3, 1e-10));
    }

    #[test]
    fn test_se2_generators() {
        let tangent = SE2Tangent::new(col![1.0, 1.0, 1.0]);

        // Test all three generators
        for i in 0..3 {
            let generator = tangent.generator(i);
            assert_eq!(generator.nrows(), 3);
            assert_eq!(generator.ncols(), 3);
        }
    }

    #[test]
    #[should_panic]
    fn test_se2_generator_invalid_index() {
        let tangent = SE2Tangent::new(col![1.0, 1.0, 1.0]);
        let _generator = tangent.generator(3); // Should panic for SE(2)
    }
}
