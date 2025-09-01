use std::collections::{HashMap, HashSet};

use crate::manifold::{LieGroup, se2::SE2, se3::SE3, so2::SO2, so3::SO3};
use nalgebra as na;

/// Represents the different types of values a Variable can hold
#[derive(Clone, Debug)]
pub enum VariableType {
    /// Standard Euclidean vector
    Vector(na::DVector<f64>),
    /// SE(2) manifold - 2D rigid transformations
    SE2(SE2),
    /// SE(3) manifold - 3D rigid transformations  
    SE3(SE3),
    /// SO(2) manifold - 2D rotations
    SO2(SO2),
    /// SO(3) manifold - 3D rotations
    SO3(SO3),
}

impl VariableType {
    /// Get the size of the underlying representation
    pub fn get_size(&self) -> usize {
        match self {
            VariableType::Vector(v) => v.len(),
            VariableType::SE2(_) => SE2::DOF,
            VariableType::SE3(_) => SE3::DOF,
            VariableType::SO2(_) => SO2::DOF,
            VariableType::SO3(_) => SO3::DOF,
        }
    }

    /// Plus operation: apply tangent space perturbation
    pub fn plus(&self, tangent: &na::DVector<f64>) -> VariableType {
        match self {
            VariableType::Vector(v) => VariableType::Vector(v + tangent),
            VariableType::SE2(se2) => {
                let tangent_obj = crate::manifold::se2::SE2Tangent::from_vector(tangent.clone());
                let result = se2.plus(&tangent_obj, None, None);
                VariableType::SE2(result)
            }
            VariableType::SE3(se3) => {
                let tangent_obj = crate::manifold::se3::SE3Tangent::from_vector(tangent.clone());
                let result = se3.plus(&tangent_obj, None, None);
                VariableType::SE3(result)
            }
            VariableType::SO2(so2) => {
                let tangent_obj = crate::manifold::so2::SO2Tangent::from_vector(tangent.clone());
                let result = so2.plus(&tangent_obj, None, None);
                VariableType::SO2(result)
            }
            VariableType::SO3(so3) => {
                let tangent_obj = crate::manifold::so3::SO3Tangent::from_vector(tangent.clone());
                let result = so3.plus(&tangent_obj, None, None);
                VariableType::SO3(result)
            }
        }
    }

    /// Minus operation: compute tangent space difference
    pub fn minus(&self, other: &VariableType) -> na::DVector<f64> {
        match (self, other) {
            (VariableType::Vector(v1), VariableType::Vector(v2)) => v1 - v2,
            (VariableType::SE2(se2_1), VariableType::SE2(se2_2)) => {
                let tangent_result = se2_1.minus(se2_2, None, None);
                tangent_result.to_vector()
            }
            (VariableType::SE3(se3_1), VariableType::SE3(se3_2)) => {
                let tangent_result = se3_1.minus(se3_2, None, None);
                tangent_result.to_vector()
            }
            (VariableType::SO2(so2_1), VariableType::SO2(so2_2)) => {
                let tangent_result = so2_1.minus(so2_2, None, None);
                tangent_result.to_vector()
            }
            (VariableType::SO3(so3_1), VariableType::SO3(so3_2)) => {
                let tangent_result = so3_1.minus(so3_2, None, None);
                tangent_result.to_vector()
            }
            _ => {
                panic!("Cannot compute minus between different variable types");
            }
        }
    }

    /// Convert to a vector representation for optimization algorithms
    pub fn to_vector(&self) -> na::DVector<f64> {
        match self {
            VariableType::Vector(v) => v.clone(),
            VariableType::SE2(se2) => {
                // Store as [x, y, cos(theta), sin(theta)]
                let t = se2.translation();
                let angle = se2.rotation_angle();
                na::DVector::from_vec(vec![t.x, t.y, angle.cos(), angle.sin()])
            }
            VariableType::SE3(se3) => {
                // Store as [x, y, z, qx, qy, qz, qw]
                let t = se3.translation();
                let q = se3.rotation_quaternion();
                na::DVector::from_vec(vec![t.x, t.y, t.z, q.i, q.j, q.k, q.w])
            }
            VariableType::SO2(so2) => {
                // Store as [cos(theta), sin(theta)]
                let angle = so2.angle();
                na::DVector::from_vec(vec![angle.cos(), angle.sin()])
            }
            VariableType::SO3(so3) => {
                // Store as [qx, qy, qz, qw]
                let q = so3.quaternion();
                na::DVector::from_vec(vec![q.i, q.j, q.k, q.w])
            }
        }
    }

    /// Create from vector representation
    pub fn from_vector(values: na::DVector<f64>, manifold_type: Option<&str>) -> VariableType {
        match manifold_type {
            Some("SE2") => {
                let x = values[0];
                let y = values[1];
                let cos_theta = values[2];
                let sin_theta = values[3];
                let theta = sin_theta.atan2(cos_theta);
                VariableType::SE2(SE2::from_xy_angle(x, y, theta))
            }
            Some("SE3") => {
                let translation = na::Vector3::new(values[0], values[1], values[2]);
                let quaternion = na::Quaternion::new(values[6], values[3], values[4], values[5]);
                VariableType::SE3(SE3::from_translation_quaternion(translation, quaternion))
            }
            Some("SO2") => {
                let cos_theta = values[0];
                let sin_theta = values[1];
                let theta = sin_theta.atan2(cos_theta);
                VariableType::SO2(SO2::from_angle(theta))
            }
            Some("SO3") => {
                let quaternion = na::Quaternion::new(values[3], values[0], values[1], values[2]);
                let unit_quat = na::UnitQuaternion::from_quaternion(quaternion);
                VariableType::SO3(SO3::from_quaternion(unit_quat))
            }
            None => VariableType::Vector(values),
            Some(unknown) => panic!("Unknown manifold type: {}", unknown),
        }
    }
}

#[derive(Clone)]
pub struct Variable {
    pub values: VariableType,
    pub fixed_indices: HashSet<usize>,
    pub bounds: HashMap<usize, (f64, f64)>,
}

impl Variable {
    /// Create variable from VariableType
    pub fn new(values: VariableType) -> Self {
        Variable {
            values,
            fixed_indices: HashSet::new(),
            bounds: HashMap::new(),
        }
    }

    /// Set values using VariableType
    pub fn set_values(&mut self, values: VariableType) {
        self.values = values;
    }

    /// Get size of the variable
    pub fn get_size(&self) -> usize {
        self.values.get_size()
    }

    /// Update values with bounds and fixed constraints
    pub fn update_values(&mut self, new_values: na::DVector<f64>) {
        let mut constrained_values = new_values;

        // Apply bounds
        for (&idx, &(lower, upper)) in &self.bounds {
            if idx < constrained_values.len() {
                constrained_values[idx] = constrained_values[idx].max(lower).min(upper);
            }
        }

        // Apply fixed indices
        let current_vector = self.values.to_vector();
        for &index_to_fix in &self.fixed_indices {
            if index_to_fix < constrained_values.len() && index_to_fix < current_vector.len() {
                constrained_values[index_to_fix] = current_vector[index_to_fix];
            }
        }

        // Reconstruct the variable type from the constrained vector
        let manifold_type = match &self.values {
            VariableType::Vector(_) => None,
            VariableType::SE2(_) => Some("SE2"),
            VariableType::SE3(_) => Some("SE3"),
            VariableType::SO2(_) => Some("SO2"),
            VariableType::SO3(_) => Some("SO3"),
        };

        self.values = VariableType::from_vector(constrained_values, manifold_type);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{LieGroup, se2::SE2, se3::SE3, so2::SO2, so3::SO3};
    use nalgebra as na;
    use std::f64::consts::PI;

    // ===== VariableType Tests =====

    #[test]
    fn test_variable_type_vector_operations() {
        // Test different vector sizes
        for size in 3..=10 {
            let vec_data: Vec<f64> = (0..size).map(|i| i as f64 + 1.0).collect();
            let vector = na::DVector::from_vec(vec_data.clone());
            let var_type = VariableType::Vector(vector.clone());

            assert_eq!(var_type.get_size(), size);

            // Test to_vector
            let result_vec = var_type.to_vector();
            assert_eq!(result_vec.len(), size);
            for i in 0..size {
                assert_eq!(result_vec[i], (i + 1) as f64);
            }

            // Test plus operation
            let perturbation = na::DVector::from_element(size, 0.1);
            let result_type = var_type.plus(&perturbation);

            if let VariableType::Vector(result_vec) = result_type {
                assert_eq!(result_vec.len(), size);
                for i in 0..size {
                    assert!((result_vec[i] - ((i + 1) as f64 + 0.1)).abs() < 1e-10);
                }
            } else {
                panic!("Expected Vector type");
            }

            // Test minus operation between vectors
            let vec2_data: Vec<f64> = (0..size).map(|i| i as f64 + 2.0).collect();
            let vector2 = na::DVector::from_vec(vec2_data);
            let var_type2 = VariableType::Vector(vector2);

            let diff = var_type2.minus(&var_type);
            assert_eq!(diff.len(), size);
            for i in 0..size {
                assert!((diff[i] - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_variable_type_se2_operations() {
        // Create SE2 manifold
        let se2 = SE2::from_xy_angle(1.0, 2.0, PI / 4.0);
        let var_type = VariableType::SE2(se2.clone());

        // Test size - SE2 has 3 DOF
        assert_eq!(var_type.get_size(), SE2::DOF);
        assert_eq!(var_type.get_size(), 3);

        // Test to_vector conversion
        let vec = var_type.to_vector();
        assert_eq!(vec.len(), 4); // [x, y, cos(theta), sin(theta)]
        assert!((vec[0] - 1.0).abs() < 1e-10); // x
        assert!((vec[1] - 2.0).abs() < 1e-10); // y
        assert!((vec[2] - (PI / 4.0).cos()).abs() < 1e-10); // cos(theta)
        assert!((vec[3] - (PI / 4.0).sin()).abs() < 1e-10); // sin(theta)

        // Test plus operation with small perturbation
        let tangent = na::DVector::from_vec(vec![0.1, 0.1, 0.1]);
        let result_type = var_type.plus(&tangent);

        if let VariableType::SE2(result_se2) = result_type {
            let original_t = se2.translation();
            let result_t = result_se2.translation();
            // Translation should have changed
            assert!((result_t.x - original_t.x).abs() > 1e-10);
            assert!((result_t.y - original_t.y).abs() > 1e-10);
        } else {
            panic!("Expected SE2 type");
        }

        // Test from_vector reconstruction
        let reconstructed = VariableType::from_vector(vec, Some("SE2"));
        if let VariableType::SE2(reconstructed_se2) = reconstructed {
            let orig_t = se2.translation();
            let recon_t = reconstructed_se2.translation();
            assert!((orig_t.x - recon_t.x).abs() < 1e-10);
            assert!((orig_t.y - recon_t.y).abs() < 1e-10);
            assert!((se2.rotation_angle() - reconstructed_se2.rotation_angle()).abs() < 1e-10);
        } else {
            panic!("Expected SE2 type");
        }
    }

    #[test]
    fn test_variable_type_se3_operations() {
        // Create SE3 manifold
        let translation = na::Vector3::new(1.0, 2.0, 3.0);
        let quaternion = na::Quaternion::new(1.0, 0.1, 0.2, 0.3).normalize();
        let se3 = SE3::from_translation_quaternion(translation, quaternion);
        let var_type = VariableType::SE3(se3.clone());

        // Test size - SE3 has 6 DOF
        assert_eq!(var_type.get_size(), SE3::DOF);
        assert_eq!(var_type.get_size(), 6);

        // Test to_vector conversion
        let vec = var_type.to_vector();
        assert_eq!(vec.len(), 7); // [x, y, z, qx, qy, qz, qw]

        // Test plus operation
        let tangent = na::DVector::from_vec(vec![0.1, 0.1, 0.1, 0.05, 0.05, 0.05]);
        let result_type = var_type.plus(&tangent);

        if let VariableType::SE3(result_se3) = result_type {
            let original_t = se3.translation();
            let result_t = result_se3.translation();
            // Translation should have changed
            assert!((result_t.x - original_t.x).abs() > 1e-10);
            assert!((result_t.y - original_t.y).abs() > 1e-10);
            assert!((result_t.z - original_t.z).abs() > 1e-10);
        } else {
            panic!("Expected SE3 type");
        }

        // Test from_vector reconstruction
        let reconstructed = VariableType::from_vector(vec, Some("SE3"));
        if let VariableType::SE3(_) = reconstructed {
            // Successfully reconstructed SE3
        } else {
            panic!("Expected SE3 type");
        }
    }

    #[test]
    fn test_variable_type_so2_operations() {
        // Create SO2 manifold
        let so2 = SO2::from_angle(PI / 3.0);
        let var_type = VariableType::SO2(so2.clone());

        // Test size - SO2 has 1 DOF
        assert_eq!(var_type.get_size(), SO2::DOF);
        assert_eq!(var_type.get_size(), 1);

        // Test to_vector conversion
        let vec = var_type.to_vector();
        assert_eq!(vec.len(), 2); // [cos(theta), sin(theta)]
        assert!((vec[0] - (PI / 3.0).cos()).abs() < 1e-10);
        assert!((vec[1] - (PI / 3.0).sin()).abs() < 1e-10);

        // Test plus operation
        let tangent = na::DVector::from_vec(vec![0.1]);
        let result_type = var_type.plus(&tangent);

        if let VariableType::SO2(result_so2) = result_type {
            // Angle should have changed
            assert!((result_so2.angle() - so2.angle()).abs() > 1e-10);
        } else {
            panic!("Expected SO2 type");
        }

        // Test from_vector reconstruction
        let reconstructed = VariableType::from_vector(vec, Some("SO2"));
        if let VariableType::SO2(reconstructed_so2) = reconstructed {
            assert!((so2.angle() - reconstructed_so2.angle()).abs() < 1e-10);
        } else {
            panic!("Expected SO2 type");
        }
    }

    #[test]
    fn test_variable_type_so3_operations() {
        // Create SO3 manifold
        let quaternion = na::UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let so3 = SO3::from_quaternion(quaternion);
        let var_type = VariableType::SO3(so3.clone());

        // Test size - SO3 has 3 DOF
        assert_eq!(var_type.get_size(), SO3::DOF);
        assert_eq!(var_type.get_size(), 3);

        // Test to_vector conversion
        let vec = var_type.to_vector();
        assert_eq!(vec.len(), 4); // [qx, qy, qz, qw]

        // Test plus operation
        let tangent = na::DVector::from_vec(vec![0.05, 0.05, 0.05]);
        let result_type = var_type.plus(&tangent);

        if let VariableType::SO3(result_so3) = result_type {
            // Quaternion should have changed
            let orig_q = so3.quaternion();
            let result_q = result_so3.quaternion();
            assert!(
                (orig_q.w - result_q.w).abs() > 1e-10
                    || (orig_q.i - result_q.i).abs() > 1e-10
                    || (orig_q.j - result_q.j).abs() > 1e-10
                    || (orig_q.k - result_q.k).abs() > 1e-10
            );
        } else {
            panic!("Expected SO3 type");
        }

        // Test from_vector reconstruction
        let reconstructed = VariableType::from_vector(vec, Some("SO3"));
        if let VariableType::SO3(_) = reconstructed {
            // Successfully reconstructed SO3
        } else {
            panic!("Expected SO3 type");
        }
    }

    #[test]
    fn test_variable_type_minus_operations() {
        // Test SE2 minus
        let se2_1 = SE2::from_xy_angle(2.0, 3.0, PI / 2.0);
        let se2_2 = SE2::from_xy_angle(1.0, 1.0, PI / 4.0);
        let var1 = VariableType::SE2(se2_1);
        let var2 = VariableType::SE2(se2_2);

        let diff = var1.minus(&var2);
        assert_eq!(diff.len(), 3); // SE2 tangent space dimension

        // Test SO3 minus
        let so3_1 = SO3::from_euler_angles(0.2, 0.3, 0.4);
        let so3_2 = SO3::from_euler_angles(0.1, 0.1, 0.1);
        let var1 = VariableType::SO3(so3_1);
        let var2 = VariableType::SO3(so3_2);

        let diff = var1.minus(&var2);
        assert_eq!(diff.len(), 3); // SO3 tangent space dimension
    }

    #[test]
    #[should_panic(expected = "Cannot compute minus between different variable types")]
    fn test_variable_type_minus_different_types() {
        let vec = VariableType::Vector(na::DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let se2 = VariableType::SE2(SE2::from_xy_angle(1.0, 2.0, 0.5));

        // This should panic
        vec.minus(&se2);
    }

    // ===== Variable Struct Tests =====

    #[test]
    fn test_variable_creation_and_basic_operations() {
        // Test with vector
        let vec_data = na::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let var_type = VariableType::Vector(vec_data);
        let variable = Variable::new(var_type);

        assert_eq!(variable.get_size(), 5);
        assert!(variable.fixed_indices.is_empty());
        assert!(variable.bounds.is_empty());

        // Test with SE3
        let se3 = SE3::from_translation_quaternion(
            na::Vector3::new(1.0, 2.0, 3.0),
            na::Quaternion::new(1.0, 0.0, 0.0, 0.0),
        );
        let var_type = VariableType::SE3(se3);
        let variable = Variable::new(var_type);

        assert_eq!(variable.get_size(), SE3::DOF);
    }

    #[test]
    fn test_variable_set_values() {
        let initial_vec = na::DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let mut variable = Variable::new(VariableType::Vector(initial_vec));

        // Set new vector values
        let new_vec = na::DVector::from_vec(vec![4.0, 5.0, 6.0, 7.0]);
        variable.set_values(VariableType::Vector(new_vec));
        assert_eq!(variable.get_size(), 4);

        // Set manifold values
        let se2 = SE2::from_xy_angle(1.0, 2.0, PI / 4.0);
        variable.set_values(VariableType::SE2(se2));
        assert_eq!(variable.get_size(), SE2::DOF);
    }

    #[test]
    fn test_variable_update_values_with_bounds() {
        // Test with vector of size 6
        let vec_data = na::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut variable = Variable::new(VariableType::Vector(vec_data));

        // Set bounds on some indices
        variable.bounds.insert(0, (-1.0, 1.0)); // Clamp first element
        variable.bounds.insert(2, (0.0, 2.0)); // Clamp third element

        // Try to update with values that violate bounds
        let new_values = na::DVector::from_vec(vec![-5.0, 10.0, -3.0, 20.0, 30.0, 40.0]);
        variable.update_values(new_values);

        let result_vec = variable.values.to_vector();
        assert_eq!(result_vec[0], -1.0); // Clamped to lower bound
        assert_eq!(result_vec[1], 10.0); // Unchanged (no bound)
        assert_eq!(result_vec[2], 0.0); // Clamped to lower bound
        assert_eq!(result_vec[3], 20.0); // Unchanged (no bound)
    }

    #[test]
    fn test_variable_update_values_with_fixed_indices() {
        // Test with vector of size 8
        let vec_data = na::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut variable = Variable::new(VariableType::Vector(vec_data.clone()));

        // Fix some indices
        variable.fixed_indices.insert(1); // Fix second element
        variable.fixed_indices.insert(4); // Fix fifth element

        // Try to update with new values
        let new_values =
            na::DVector::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
        variable.update_values(new_values);

        let result_vec = variable.values.to_vector();
        assert_eq!(result_vec[0], 10.0); // Changed
        assert_eq!(result_vec[1], 2.0); // Fixed to original value
        assert_eq!(result_vec[2], 30.0); // Changed
        assert_eq!(result_vec[3], 40.0); // Changed
        assert_eq!(result_vec[4], 5.0); // Fixed to original value
        assert_eq!(result_vec[5], 60.0); // Changed
    }

    #[test]
    fn test_variable_update_values_manifold() {
        // Test with SE2 manifold
        let se2 = SE2::from_xy_angle(1.0, 2.0, PI / 4.0);
        let mut variable = Variable::new(VariableType::SE2(se2.clone()));

        // Fix rotation component (indices 2, 3 in vector representation)
        variable.fixed_indices.insert(2); // cos(theta)
        variable.fixed_indices.insert(3); // sin(theta)

        // Try to update - should preserve the fixed rotation
        let original_vec = variable.values.to_vector();
        let new_values = na::DVector::from_vec(vec![5.0, 6.0, 0.0, 1.0]); // Try to set different rotation
        variable.update_values(new_values);

        let result_vec = variable.values.to_vector();
        assert_eq!(result_vec[0], 5.0); // Translation x changed
        assert_eq!(result_vec[1], 6.0); // Translation y changed
        assert!((result_vec[2] - original_vec[2]).abs() < 1e-10); // cos(theta) preserved
        assert!((result_vec[3] - original_vec[3]).abs() < 1e-10); // sin(theta) preserved
    }

    #[test]
    fn test_variable_different_vector_sizes() {
        // Test vector sizes from 3 to 10
        for size in 3..=10 {
            let vec_data: Vec<f64> = (0..size).map(|i| i as f64).collect();
            let vector = na::DVector::from_vec(vec_data);
            let variable = Variable::new(VariableType::Vector(vector));

            assert_eq!(variable.get_size(), size);

            // Test update with same size
            let new_data: Vec<f64> = (0..size).map(|i| (i as f64) * 2.0).collect();
            let new_vector = na::DVector::from_vec(new_data.clone());
            let mut variable_mut = variable.clone();
            variable_mut.update_values(new_vector);

            let result = variable_mut.values.to_vector();
            assert_eq!(result.len(), size);
            for i in 0..size {
                assert_eq!(result[i], (i as f64) * 2.0);
            }
        }
    }

    #[test]
    fn test_variable_combined_bounds_and_fixed() {
        // Test with size 7 vector
        let vec_data = na::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let mut variable = Variable::new(VariableType::Vector(vec_data.clone()));

        // Set both bounds and fixed indices
        variable.bounds.insert(0, (0.0, 10.0)); // Bound first element
        variable.bounds.insert(3, (-5.0, 5.0)); // Bound fourth element
        variable.fixed_indices.insert(1); // Fix second element
        variable.fixed_indices.insert(5); // Fix sixth element

        // Update with values that test both constraints
        let new_values = na::DVector::from_vec(vec![-5.0, 100.0, 30.0, 10.0, 50.0, 600.0, 70.0]);
        variable.update_values(new_values);

        let result = variable.values.to_vector();
        assert_eq!(result[0], 0.0); // Clamped to lower bound
        assert_eq!(result[1], 2.0); // Fixed to original value
        assert_eq!(result[2], 30.0); // Changed (no constraint)
        assert_eq!(result[3], 5.0); // Clamped to upper bound
        assert_eq!(result[4], 50.0); // Changed (no constraint)
        assert_eq!(result[5], 6.0); // Fixed to original value
        assert_eq!(result[6], 70.0); // Changed (no constraint)
    }
}
