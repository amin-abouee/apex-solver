use std::collections::{HashMap, HashSet};

use crate::manifold::{LieGroup, Tangent};
use faer::Col;

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
pub struct Variable<M: LieGroup> {
    /// The manifold value
    pub value: M,
    /// Indices that should remain fixed during optimization
    pub fixed_indices: HashSet<usize>,
    /// Bounds constraints on the tangent space representation
    pub bounds: HashMap<usize, (f64, f64)>,
}

impl<M> Variable<M>
where
    M: LieGroup + Clone + 'static,
    M::TangentVector: Tangent<M>,
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
            fixed_indices: HashSet::new(),
            bounds: HashMap::new(),
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
        // For most manifolds, use the compile-time constant
        if M::TangentVector::DIM > 0 {
            M::TangentVector::DIM
        } else {
            // For dynamically sized manifolds like Rn, we need a different approach
            // This is a bit of a hack, but works for our current needs
            match std::any::type_name::<M>() {
                name if name.contains("Rn") => {
                    // For Rn manifold, get the dynamic size
                    if let Some(rn_var) = (self as &dyn std::any::Any)
                        .downcast_ref::<Variable<crate::manifold::rn::Rn>>()
                    {
                        rn_var.dynamic_size()
                    } else {
                        0
                    }
                }
                _ => M::TangentVector::DIM,
            }
        }
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
    /// let mut variable = Variable::new(se2_value);
    ///
    /// // Create a tangent vector: [dx, dy, dtheta]
    /// let tangent = SE2Tangent::from(na::DVector::from(vec![0.1, 0.1, 0.1]));
    /// variable.plus(&tangent);
    /// ```
    pub fn plus(&mut self, tangent: &M::TangentVector) {
        self.value = self.value.plus(tangent, None, None);
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
}

// Extension implementation for Rn manifold (special case since it's Euclidean)
use crate::manifold::rn::Rn;

impl Variable<Rn> {
    /// Get the dynamic size for Rn manifold.
    pub fn dynamic_size(&self) -> usize {
        self.value.data().nrows()
    }

    /// Convert the Rn variable to a vector representation.
    pub fn to_vector(&self) -> Col<f64> {
        self.value.data().clone()
    }

    /// Create an Rn variable from a vector representation.
    pub fn from_vector(values: Col<f64>) -> Self {
        Self::new(Rn::new(values))
    }

    /// Update the Rn variable with bounds and fixed constraints.
    pub fn update_variable(&mut self, mut tangent_delta: Col<f64>) {
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
    use faer::col;
    use std::f64::consts::PI;

    #[test]
    fn test_variable_creation_rn() {
        let vec_data = col![1.0, 2.0, 3.0, 4.0, 5.0];
        let rn_value = Rn::new(vec_data);
        let variable = Variable::new(rn_value);

        // Use dynamic_size for Rn manifold
        assert_eq!(variable.dynamic_size(), 5);
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
        use crate::manifold::quaternion::Quaternion;

        let se3 = SE3::from_translation_quaternion(
            col![1.0, 2.0, 3.0],
            Quaternion::new(1.0, 0.0, 0.0, 0.0).unwrap(),
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
        let initial_vec = col![1.0, 2.0, 3.0];
        let mut variable = Variable::new(Rn::new(initial_vec));

        let new_vec = col![4.0, 5.0, 6.0, 7.0];
        variable.set_value(Rn::new(new_vec));
        assert_eq!(variable.dynamic_size(), 4);

        let se2_initial = SE2::from_xy_angle(0.0, 0.0, 0.0);
        let mut se2_variable = Variable::new(se2_initial);

        let se2_new = SE2::from_xy_angle(1.0, 2.0, PI / 4.0);
        se2_variable.set_value(se2_new);
        assert_eq!(se2_variable.get_size(), SE2::DOF);
    }

    // #[test]
    // fn test_variable_plus_minus_operations() {
    //     // Test SE2 manifold plus/minus operations
    //     let se2_1 = SE2::from_xy_angle(2.0, 3.0, PI / 2.0);
    //     let se2_2 = SE2::from_xy_angle(1.0, 1.0, PI / 4.0);
    //     let var1 = Variable::new(se2_1);
    //     let var2 = Variable::new(se2_2);

    //     let diff_tangent = var1.minus(&var2);
    //     var2.plus(&diff_tangent);
    //     let final_diff = var1.minus(&Variable::new(var2));

    //     assert!(DVector::from(final_diff).norm() < 1e-10);
    // }

    // #[test]
    // fn test_variable_rn_plus_minus_operations() {
    //     // Test Rn manifold plus/minus operations
    //     let rn_1 = Rn::new(na::DVector::from_vec(vec![1.0, 2.0, 3.0]));
    //     let rn_2 = Rn::new(na::DVector::from_vec(vec![4.0, 5.0, 6.0]));
    //     let var1 = Variable::new(rn_1);
    //     let var2 = Variable::new(rn_2);

    //     // Test minus operation
    //     let diff_tangent = var1.minus(&var2);
    //     assert_eq!(
    //         diff_tangent.to_vector(),
    //         na::DVector::from_vec(vec![-3.0, -3.0, -3.0])
    //     );

    //     // Test plus operation
    //     var2.plus(&diff_tangent);
    //     assert_eq!(var2.data(), &na::DVector::from_vec(vec![1.0, 2.0, 3.0]));

    //     // Test roundtrip consistency
    //     let final_diff = var1.minus(&Variable::new(var2_updated));
    //     assert!(final_diff.to_vector().norm() < 1e-10);
    // }

    #[test]
    fn test_variable_update_with_bounds() {
        let vec_data = col![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut variable = Variable::new(Rn::new(vec_data));

        variable.bounds.insert(0, (-1.0, 1.0));
        variable.bounds.insert(2, (0.0, 5.0));

        let new_values = col![-5.0, 10.0, -3.0, 20.0, 30.0, 40.0];
        variable.update_variable(new_values);

        let result_vec = variable.to_vector();
        assert!(result_vec.nrows() == 6);
    }

    #[test]
    fn test_variable_update_with_fixed_indices() {
        let vec_data = col![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut variable = Variable::new(Rn::new(vec_data.clone()));

        variable.fixed_indices.insert(1);
        variable.fixed_indices.insert(4);

        let delta_values = col![9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 72.0];
        variable.update_variable(delta_values);

        let result_vec = variable.to_vector();
        assert_eq!(result_vec[1], 2.0);
        assert_eq!(result_vec[4], 5.0);
        assert!(result_vec.nrows() == 8);
    }

    #[test]
    fn test_variable_combined_bounds_and_fixed() {
        let vec_data = col![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut variable = Variable::new(Rn::new(vec_data.clone()));

        variable.bounds.insert(0, (-2.0, 2.0));
        variable.bounds.insert(3, (-1.0, 1.0));
        variable.fixed_indices.insert(1);
        variable.fixed_indices.insert(5);

        let delta_values = col![-5.0, 100.0, 30.0, 10.0, 50.0, 600.0, 70.0];
        variable.update_variable(delta_values);

        let result = variable.to_vector();
        assert_eq!(result[1], 2.0);
        assert_eq!(result[5], 6.0);
        assert!(result.nrows() == 7);
    }

    #[test]
    fn test_variable_type_safety() {
        use crate::manifold::quaternion::Quaternion;

        let se2_var = Variable::new(SE2::from_xy_angle(1.0, 2.0, 0.5));
        let se3_var = Variable::new(SE3::from_translation_quaternion(
            col![1.0, 2.0, 3.0],
            Quaternion::new(1.0, 0.0, 0.0, 0.0).unwrap(),
        ));
        let so2_var = Variable::new(SO2::from_angle(0.5));
        let so3_var = Variable::new(SO3::from_euler_angles(0.1, 0.2, 0.3));
        let rn_var = Variable::new(Rn::new(col![1.0, 2.0, 3.0]));

        assert_eq!(se2_var.get_size(), SE2::DOF);
        assert_eq!(se3_var.get_size(), SE3::DOF);
        assert_eq!(so2_var.get_size(), SO2::DOF);
        assert_eq!(so3_var.get_size(), SO3::DOF);
        assert_eq!(rn_var.dynamic_size(), 3);
    }

    #[test]
    fn test_variable_vector_conversion_roundtrip() {
        let original_data = col![1.0, 2.0, 3.0, 4.0, 5.0];
        let rn_var = Variable::new(Rn::new(original_data.clone()));
        let vec_repr = rn_var.to_vector();
        assert_eq!(vec_repr.as_ref(), original_data.as_ref());

        let reconstructed_var = Variable::<Rn>::from_vector(vec_repr);
        assert_eq!(
            reconstructed_var.to_vector().as_ref(),
            original_data.as_ref()
        );
    }

    // #[test]
    // fn test_variable_manifold_operations_consistency() {
    //     // Test Rn manifold operations (has vector conversion methods)
    //     let rn_initial = Rn::new(na::DVector::from_vec(vec![1.0, 2.0, 3.0]));
    //     let mut rn_var = Variable::new(rn_initial);
    //     let rn_new_values = na::DVector::from_vec(vec![2.0, 3.0, 4.0]);
    //     rn_var.update_variable(rn_new_values);

    //     let rn_result = rn_var.to_vector();
    //     assert_eq!(rn_result, na::DVector::from_vec(vec![2.0, 3.0, 4.0]));

    //     // Test SE2 manifold plus/minus operations (core functionality)
    //     let se2_1 = SE2::from_xy_angle(2.0, 3.0, PI / 2.0);
    //     let se2_2 = SE2::from_xy_angle(1.0, 1.0, PI / 4.0);
    //     let var1 = Variable::new(se2_1);
    //     let var2 = Variable::new(se2_2);

    //     let diff_tangent = var1.minus(&var2);
    //     let var2_updated = var2.plus(&diff_tangent);
    //     let final_diff = var1.minus(&Variable::new(var2_updated));

    //     // The final difference should be small (close to identity in tangent space)
    //     assert!(DVector::from(final_diff).norm() < 1e-10);
    // }

    #[test]
    fn test_variable_constraints_interaction() {
        let rn_data = col![0.0, 0.0, 0.0, 0.0, 0.0];
        let mut rn_var = Variable::new(Rn::new(rn_data));

        rn_var.bounds.insert(0, (-1.0, 1.0));
        rn_var.bounds.insert(2, (-10.0, 10.0));
        rn_var.fixed_indices.insert(1);
        rn_var.fixed_indices.insert(4);

        let large_delta = col![5.0, 100.0, 15.0, 20.0, 200.0];
        rn_var.update_variable(large_delta);

        let result = rn_var.to_vector();

        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 10.0);
        assert_eq!(result[3], 20.0);
        assert_eq!(result[4], 0.0);
    }
}
