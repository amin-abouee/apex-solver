//! Variables for optimization on manifolds.
//!
//! This module provides the `Variable` struct, which represents optimization variables that
//! live on manifolds (Lie groups like SE2, SE3, SO2, SO3, or Euclidean spaces Rn). Variables
//! support manifold operations (plus/minus in tangent space), constraints (fixed indices,
//! bounds), and covariance estimation.
//!
//! # Key Concepts
//!
//! ## Manifolds vs. Vector Spaces
//!
//! Unlike standard optimization that operates in Euclidean space (Rⁿ), many robotics problems
//! involve variables on manifolds:
//! - **Poses**: SE(2) or SE(3) - rigid body transformations
//! - **Rotations**: SO(2) or SO(3) - rotation matrices/quaternions
//! - **Landmarks**: R³ - 3D points in Euclidean space
//!
//! Manifolds require special handling:
//! - Updates happen in the tangent space (local linearization)
//! - Plus operation (⊞): manifold × tangent → manifold
//! - Minus operation (⊟): manifold × manifold → tangent
//!
//! ## Tangent Space Updates
//!
//! During optimization, updates are computed as tangent vectors and applied via the plus
//! operation:
//!
//! ```text
//! x_new = x_old ⊞ δx
//! ```
//!
//! where `δx` is a tangent vector (e.g., 6D for SE(3), 3D for SE(2)).
//!
//! ## Constraints
//!
//! Variables support two types of constraints:
//! - **Fixed indices**: Specific DOF held constant during optimization
//! - **Bounds**: Box constraints (min/max) on tangent space components
//!
//! ## Covariance
//!
//! After optimization, the `Variable` can store a covariance matrix representing uncertainty
//! in the tangent space. For SE(3), this is a 6×6 matrix; for SE(2), a 3×3 matrix.
//!
//! # Example: SE(2) Variable
//!
//! ```
//! use apex_solver::core::variable::Variable;
//! use apex_solver::manifold::se2::{SE2, SE2Tangent};
//! use nalgebra::DVector;
//!
//! // Create a 2D pose variable
//! let initial_pose = SE2::from_xy_angle(1.0, 2.0, 0.5);
//! let mut var = Variable::new(initial_pose);
//!
//! // Apply a tangent space update: [dx, dy, dtheta]
//! let delta = SE2Tangent::from(DVector::from_vec(vec![0.1, 0.2, 0.05]));
//! let updated_pose = var.plus(&delta);
//! var.set_value(updated_pose);
//!
//! println!("Updated pose at ({}, {}, {})",
//!     var.value.translation().x,
//!     var.value.translation().y,
//!     var.value.angle()
//! );
//! ```
//!
//! # Example: Variable with Constraints
//!
//! ```
//! use apex_solver::core::variable::Variable;
//! use apex_solver::manifold::rn::Rn;
//! use nalgebra::DVector;
//!
//! // Create a 3D point variable
//! let mut landmark = Variable::new(Rn::new(DVector::from_vec(vec![0.0, 0.0, 0.0])));
//!
//! // Fix the z-coordinate (index 2)
//! landmark.fixed_indices.insert(2);
//!
//! // Constrain x to [-10, 10]
//! landmark.bounds.insert(0, (-10.0, 10.0));
//!
//! // Apply update (will respect constraints)
//! let update = DVector::from_vec(vec![15.0, 5.0, 100.0]); // Large update
//! landmark.update_variable(update);
//!
//! let result = landmark.to_vector();
//! assert_eq!(result[0], 10.0);   // Clamped to upper bound
//! assert_eq!(result[1], 5.0);    // Unconstrained
//! assert_eq!(result[2], 0.0);    // Fixed at original value
//! ```

use std::collections;

use crate::manifold;
use faer;
use nalgebra;

/// Generic Variable struct that uses static dispatch with any manifold type.
///
/// This struct represents optimization variables that live on manifolds and provides
/// type-safe operations for updating variables with tangent space perturbations.
///
/// # Type Parameters
/// * `M` - The manifold type that implements the LieGroup trait
///
/// # Examples
/// ```
/// use apex_solver::core::variable::Variable;
/// use apex_solver::manifold::se2::SE2;
/// use apex_solver::manifold::rn::Rn;
///
/// // Create a Variable for SE2 manifold
/// let se2_value = SE2::from_xy_angle(1.0, 2.0, 0.5);
/// let se2_var = Variable::new(se2_value);
///
/// // Create a Variable for Euclidean space
/// let rn_value = Rn::from_vec(vec![1.0, 2.0, 3.0]);
/// let rn_var = Variable::new(rn_value);
/// ```
#[derive(Clone, Debug)]
pub struct Variable<M: manifold::LieGroup> {
    /// The manifold value
    pub value: M,
    /// Indices that should remain fixed during optimization
    pub fixed_indices: collections::HashSet<usize>,
    /// Bounds constraints on the tangent space representation
    pub bounds: collections::HashMap<usize, (f64, f64)>,
    /// Covariance matrix in the tangent space (uncertainty estimation)
    ///
    /// This is `None` if covariance has not been computed.
    /// When present, it's a square matrix of size `tangent_dim x tangent_dim`
    /// representing the uncertainty in the optimized variable's tangent space.
    ///
    /// For example, for SE3 this would be a 6×6 matrix representing uncertainty
    /// in [translation_x, translation_y, translation_z, rotation_x, rotation_y, rotation_z].
    pub covariance: Option<faer::Mat<f64>>,
}

impl<M> Variable<M>
where
    M: manifold::LieGroup + Clone + 'static,
    M::TangentVector: manifold::Tangent<M>,
{
    /// Create a new Variable from a manifold value.
    ///
    /// # Arguments
    /// * `value` - The initial manifold value
    ///
    /// # Examples
    /// ```
    /// use apex_solver::core::variable::Variable;
    /// use apex_solver::manifold::se2::SE2;
    ///
    /// let se2_value = SE2::from_xy_angle(1.0, 2.0, 0.5);
    /// let variable = Variable::new(se2_value);
    /// ```
    pub fn new(value: M) -> Self {
        Variable {
            value,
            fixed_indices: collections::HashSet::new(),
            bounds: collections::HashMap::new(),
            covariance: None,
        }
    }

    /// Set the manifold value.
    ///
    /// # Arguments
    /// * `value` - The new manifold value
    pub fn set_value(&mut self, value: M) {
        self.value = value;
    }

    /// Get the degrees of freedom (tangent space dimension) of the variable.
    ///
    /// This returns the dimension of the tangent space, which is the number of
    /// parameters that can be optimized for this manifold type.
    ///
    /// # Returns
    /// The tangent space dimension (degrees of freedom)
    pub fn get_size(&self) -> usize {
        self.value.tangent_dim()
    }

    /// Plus operation: apply tangent space perturbation to the manifold value.
    ///
    /// This method takes a tangent vector and returns a new manifold value by applying
    /// the manifold's plus operation (typically the exponential map).
    ///
    /// # Arguments
    /// * `tangent` - The tangent vector to apply as a perturbation
    ///
    /// # Returns
    /// A new manifold value after applying the tangent perturbation
    ///
    /// # Examples
    /// ```
    /// use apex_solver::core::variable::Variable;
    /// use apex_solver::manifold::se2::{SE2, SE2Tangent};
    /// use nalgebra as na;
    ///
    /// let se2_value = SE2::from_xy_angle(1.0, 2.0, 0.0);
    /// let variable = Variable::new(se2_value);
    ///
    /// // Create a tangent vector: [dx, dy, dtheta]
    /// let tangent = SE2Tangent::from(na::DVector::from(vec![0.1, 0.1, 0.1]));
    /// let new_value = variable.plus(&tangent);
    /// ```
    pub fn plus(&self, tangent: &M::TangentVector) -> M {
        self.value.plus(tangent, None, None)
    }

    /// Minus operation: compute tangent space difference between two manifold values.
    ///
    /// This method computes the tangent vector that would transform this variable's
    /// value to the other variable's value using the manifold's minus operation
    /// (typically the logarithmic map).
    ///
    /// # Arguments
    /// * `other` - The other variable to compute the difference to
    ///
    /// # Returns
    /// A tangent vector representing the difference in tangent space
    ///
    /// # Examples
    /// ```
    /// use apex_solver::core::variable::Variable;
    /// use apex_solver::manifold::se2::SE2;
    ///
    /// let se2_1 = SE2::from_xy_angle(2.0, 3.0, 0.5);
    /// let se2_2 = SE2::from_xy_angle(1.0, 2.0, 0.0);
    /// let var1 = Variable::new(se2_1);
    /// let var2 = Variable::new(se2_2);
    ///
    /// let difference = var1.minus(&var2);
    /// ```
    pub fn minus(&self, other: &Self) -> M::TangentVector {
        self.value.minus(&other.value, None, None)
    }

    /// Get the covariance matrix for this variable (if computed).
    ///
    /// Returns `None` if covariance has not been computed.
    ///
    /// # Returns
    /// Reference to the covariance matrix in tangent space
    pub fn get_covariance(&self) -> Option<&faer::Mat<f64>> {
        self.covariance.as_ref()
    }

    /// Set the covariance matrix for this variable.
    ///
    /// The covariance matrix should be square with dimension equal to
    /// the tangent space dimension of this variable.
    ///
    /// # Arguments
    /// * `cov` - Covariance matrix in tangent space
    pub fn set_covariance(&mut self, cov: faer::Mat<f64>) {
        self.covariance = Some(cov);
    }

    /// Clear the covariance matrix.
    pub fn clear_covariance(&mut self) {
        self.covariance = None;
    }
}

// Extension implementation for Rn manifold (special case since it's Euclidean)
use crate::manifold::rn::Rn;

impl Variable<Rn> {
    /// Convert the Rn variable to a vector representation.
    pub fn to_vector(&self) -> nalgebra::DVector<f64> {
        self.value.data().clone()
    }

    /// Create an Rn variable from a vector representation.
    pub fn from_vector(values: nalgebra::DVector<f64>) -> Self {
        Self::new(Rn::new(values))
    }

    /// Update the Rn variable with bounds and fixed constraints.
    pub fn update_variable(&mut self, mut tangent_delta: nalgebra::DVector<f64>) {
        // bound
        for (&idx, &(lower, upper)) in &self.bounds {
            tangent_delta[idx] = tangent_delta[idx].max(lower).min(upper);
        }

        // fix
        for &index_to_fix in &self.fixed_indices {
            tangent_delta[index_to_fix] = self.value.data()[index_to_fix];
        }

        self.value = Rn::new(tangent_delta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{rn::Rn, se2::SE2, se3::SE3, so2::SO2, so3::SO3};
    use nalgebra;
    use std;

    #[test]
    fn test_variable_creation_rn() {
        let vec_data = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let rn_value = Rn::new(vec_data);
        let variable = Variable::new(rn_value);

        // Use get_size for Rn manifold (returns dynamic size)
        assert_eq!(variable.get_size(), 5);
        assert!(variable.fixed_indices.is_empty());
        assert!(variable.bounds.is_empty());
    }

    #[test]
    fn test_variable_creation_se2() {
        let se2 = SE2::from_xy_angle(1.0, 2.0, 0.5);
        let variable = Variable::new(se2);

        assert_eq!(variable.get_size(), SE2::DOF);
        assert!(variable.fixed_indices.is_empty());
        assert!(variable.bounds.is_empty());
    }

    #[test]
    fn test_variable_creation_se3() {
        let se3 = SE3::from_translation_quaternion(
            nalgebra::Vector3::new(1.0, 2.0, 3.0),
            nalgebra::Quaternion::new(1.0, 0.0, 0.0, 0.0),
        );
        let variable = Variable::new(se3);

        assert_eq!(variable.get_size(), SE3::DOF);
        assert!(variable.fixed_indices.is_empty());
        assert!(variable.bounds.is_empty());
    }

    #[test]
    fn test_variable_creation_so2() {
        let so2 = SO2::from_angle(0.5);
        let variable = Variable::new(so2);

        assert_eq!(variable.get_size(), SO2::DOF);
        assert!(variable.fixed_indices.is_empty());
        assert!(variable.bounds.is_empty());
    }

    #[test]
    fn test_variable_creation_so3() {
        let so3 = SO3::from_euler_angles(0.1, 0.2, 0.3);
        let variable = Variable::new(so3);

        assert_eq!(variable.get_size(), SO3::DOF);
        assert!(variable.fixed_indices.is_empty());
        assert!(variable.bounds.is_empty());
    }

    #[test]
    fn test_variable_set_value() {
        let initial_vec = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let mut variable = Variable::new(Rn::new(initial_vec));

        let new_vec = nalgebra::DVector::from_vec(vec![4.0, 5.0, 6.0, 7.0]);
        variable.set_value(Rn::new(new_vec));
        assert_eq!(variable.get_size(), 4);

        let se2_initial = SE2::from_xy_angle(0.0, 0.0, 0.0);
        let mut se2_variable = Variable::new(se2_initial);

        let se2_new = SE2::from_xy_angle(1.0, 2.0, std::f64::consts::PI / 4.0);
        se2_variable.set_value(se2_new);
        assert_eq!(se2_variable.get_size(), SE2::DOF);
    }

    #[test]
    fn test_variable_plus_minus_operations() {
        // Test SE2 manifold plus/minus operations
        let se2_1 = SE2::from_xy_angle(2.0, 3.0, std::f64::consts::PI / 2.0);
        let se2_2 = SE2::from_xy_angle(1.0, 1.0, std::f64::consts::PI / 4.0);
        let var1 = Variable::new(se2_1);
        let var2 = Variable::new(se2_2);

        let diff_tangent = var1.minus(&var2);
        let var2_updated = var2.plus(&diff_tangent);
        let final_diff = var1.minus(&Variable::new(var2_updated));

        assert!(nalgebra::DVector::from(final_diff).norm() < 1e-10);
    }

    #[test]
    fn test_variable_rn_plus_minus_operations() {
        // Test Rn manifold plus/minus operations
        let rn_1 = Rn::new(nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let rn_2 = Rn::new(nalgebra::DVector::from_vec(vec![4.0, 5.0, 6.0]));
        let var1 = Variable::new(rn_1);
        let var2 = Variable::new(rn_2);

        // Test minus operation
        let diff_tangent = var1.minus(&var2);
        assert_eq!(
            diff_tangent.to_vector(),
            nalgebra::DVector::from_vec(vec![-3.0, -3.0, -3.0])
        );

        // Test plus operation
        let var2_updated = var2.plus(&diff_tangent);
        assert_eq!(
            var2_updated.data(),
            &nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0])
        );

        // Test roundtrip consistency
        let final_diff = var1.minus(&Variable::new(var2_updated));
        assert!(final_diff.to_vector().norm() < 1e-10);
    }

    #[test]
    fn test_variable_update_with_bounds() {
        let vec_data = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut variable = Variable::new(Rn::new(vec_data));

        variable.bounds.insert(0, (-1.0, 1.0));
        variable.bounds.insert(2, (0.0, 5.0));

        let new_values = nalgebra::DVector::from_vec(vec![-5.0, 10.0, -3.0, 20.0, 30.0, 40.0]);
        variable.update_variable(new_values);

        let result_vec = variable.to_vector();
        assert!(result_vec.len() == 6);
    }

    #[test]
    fn test_variable_update_with_fixed_indices() {
        let vec_data = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut variable = Variable::new(Rn::new(vec_data.clone()));

        variable.fixed_indices.insert(1);
        variable.fixed_indices.insert(4);

        let delta_values =
            nalgebra::DVector::from_vec(vec![9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0]);
        variable.update_variable(delta_values);

        let result_vec = variable.to_vector();
        assert_eq!(result_vec[1], 2.0);
        assert_eq!(result_vec[4], 5.0);
        assert!(result_vec.len() == 8);
    }

    #[test]
    fn test_variable_combined_bounds_and_fixed() {
        let vec_data = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let mut variable = Variable::new(Rn::new(vec_data.clone()));

        variable.bounds.insert(0, (-2.0, 2.0));
        variable.bounds.insert(3, (-1.0, 1.0));
        variable.fixed_indices.insert(1);
        variable.fixed_indices.insert(5);

        let delta_values =
            nalgebra::DVector::from_vec(vec![-5.0, 100.0, 30.0, 10.0, 50.0, 600.0, 70.0]);
        variable.update_variable(delta_values);

        let result = variable.to_vector();
        assert_eq!(result[1], 2.0);
        assert_eq!(result[5], 6.0);
        assert!(result.len() == 7);
    }

    #[test]
    fn test_variable_type_safety() {
        let se2_var = Variable::new(SE2::from_xy_angle(1.0, 2.0, 0.5));
        let se3_var = Variable::new(SE3::from_translation_quaternion(
            nalgebra::Vector3::new(1.0, 2.0, 3.0),
            nalgebra::Quaternion::new(1.0, 0.0, 0.0, 0.0),
        ));
        let so2_var = Variable::new(SO2::from_angle(0.5));
        let so3_var = Variable::new(SO3::from_euler_angles(0.1, 0.2, 0.3));
        let rn_var = Variable::new(Rn::new(nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0])));

        assert_eq!(se2_var.get_size(), SE2::DOF);
        assert_eq!(se3_var.get_size(), SE3::DOF);
        assert_eq!(so2_var.get_size(), SO2::DOF);
        assert_eq!(so3_var.get_size(), SO3::DOF);
        assert_eq!(rn_var.get_size(), 3);
    }

    #[test]
    fn test_variable_vector_conversion_roundtrip() {
        let original_data = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let rn_var = Variable::new(Rn::new(original_data.clone()));
        let vec_repr = rn_var.to_vector();
        assert_eq!(vec_repr, original_data);

        let reconstructed_var = Variable::<Rn>::from_vector(vec_repr);
        assert_eq!(reconstructed_var.to_vector(), original_data);
    }

    #[test]
    fn test_variable_manifold_operations_consistency() {
        // Test Rn manifold operations (has vector conversion methods)
        let rn_initial = Rn::new(nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let mut rn_var = Variable::new(rn_initial);
        let rn_new_values = nalgebra::DVector::from_vec(vec![2.0, 3.0, 4.0]);
        rn_var.update_variable(rn_new_values);

        let rn_result = rn_var.to_vector();
        assert_eq!(rn_result, nalgebra::DVector::from_vec(vec![2.0, 3.0, 4.0]));

        // Test SE2 manifold plus/minus operations (core functionality)
        let se2_1 = SE2::from_xy_angle(2.0, 3.0, std::f64::consts::PI / 2.0);
        let se2_2 = SE2::from_xy_angle(1.0, 1.0, std::f64::consts::PI / 4.0);
        let var1 = Variable::new(se2_1);
        let var2 = Variable::new(se2_2);

        let diff_tangent = var1.minus(&var2);
        let var2_updated = var2.plus(&diff_tangent);
        let final_diff = var1.minus(&Variable::new(var2_updated));

        // The final difference should be small (close to identity in tangent space)
        assert!(nalgebra::DVector::from(final_diff).norm() < 1e-10);
    }

    #[test]
    fn test_variable_constraints_interaction() {
        let rn_data = nalgebra::DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut rn_var = Variable::new(Rn::new(rn_data));

        rn_var.bounds.insert(0, (-1.0, 1.0));
        rn_var.bounds.insert(2, (-10.0, 10.0));
        rn_var.fixed_indices.insert(1);
        rn_var.fixed_indices.insert(4);

        let large_delta = nalgebra::DVector::from_vec(vec![5.0, 100.0, 15.0, 20.0, 200.0]);
        rn_var.update_variable(large_delta);

        let result = rn_var.to_vector();

        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 10.0);
        assert_eq!(result[3], 20.0);
        assert_eq!(result[4], 0.0);
    }
}
