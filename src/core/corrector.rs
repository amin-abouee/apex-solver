//! Corrector algorithm for applying robust loss functions in optimization.
//!
//! The Corrector implements the algorithm from Ceres Solver for transforming a robust loss
//! problem into an equivalent reweighted least squares problem. Instead of modifying the
//! solver internals, the corrector adjusts the residuals and Jacobians before they are
//! passed to the linear solver.
//!
//! # Algorithm Overview
//!
//! Given a residual vector `r` and a robust loss function ρ(s) where `s = ||r||²`, the
//! corrector computes modified residuals and Jacobians such that:
//!
//! ```text
//! minimize Σ ρ(||r_i||²)  ≡  minimize Σ ||r̃_i||²
//! ```
//!
//! where `r̃` are the corrected residuals and the Jacobian is similarly adjusted.
//!
//! # Mathematical Formulation
//!
//! For a residual `r` with Jacobian `J = ∂r/∂x`, the corrector computes:
//!
//! 1. **Square norm**: `s = ||r||² = r^T r`
//! 2. **Loss evaluation**: `[ρ(s), ρ'(s), ρ''(s)]` from the loss function
//! 3. **Scaling factors**:
//!    ```text
//!    √ρ₁ = √(ρ'(s))           (residual scaling)
//!    α² = ρ''(s) / ρ'(s)      (Jacobian correction factor)
//!    ```
//!
//! 4. **Corrected residuals**: `r̃ = √ρ₁ · r`
//! 5. **Corrected Jacobian**:
//!    ```text
//!    J̃ = √ρ₁ · J + α · (J^T r) · r^T / ||r||
//!    ```
//!
//! This ensures that `||r̃||² ≈ ρ(||r||²)` and the gradient is correct.
//!
//! # Reference
//!
//! Based on Ceres Solver implementation:
//! <https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/corrector.cc>
//!
//! See also:
//! - Triggs et al., "Bundle Adjustment — A Modern Synthesis" (1999)
//! - Agarwal et al., "Ceres Solver" (<http://ceres-solver.org/>)
//!
//! # Example
//!
//! ```
//! use apex_solver::core::corrector::Corrector;
//! use apex_solver::core::loss_functions::{Loss, HuberLoss};
//! use nalgebra::{DVector, DMatrix};
//!
//! // Create a robust loss function
//! let loss = HuberLoss::new(1.0).unwrap();
//!
//! // Original residual and Jacobian
//! let residual = DVector::from_vec(vec![2.0, 3.0, 1.0]); // Large residual (outlier)
//! let jacobian = DMatrix::from_row_slice(3, 2, &[
//!     1.0, 0.0,
//!     0.0, 1.0,
//!     1.0, 1.0,
//! ]);
//!
//! // Compute squared norm
//! let squared_norm = residual.dot(&residual);
//!
//! // Create corrector
//! let corrector = Corrector::new(&loss, squared_norm);
//!
//! // Apply corrections
//! let mut corrected_jacobian = jacobian.clone();
//! let mut corrected_residual = residual.clone();
//!
//! corrector.correct_jacobian(&residual, &mut corrected_jacobian);
//! corrector.correct_residuals(&mut corrected_residual);
//!
//! // The corrected values now account for the robust loss function
//! // Outliers have been downweighted appropriately
//! ```

use crate::core::loss_functions::Loss;
use nalgebra::{DMatrix, DVector};

/// Corrector for applying robust loss functions via residual and Jacobian adjustment.
///
/// This struct holds the precomputed scaling factors needed to transform a robust loss
/// problem into an equivalent reweighted least squares problem. It is instantiated once
/// per residual block during each iteration of the optimizer.
///
/// # Fields
///
/// - `sqrt_rho1`: √(ρ'(s)) - Square root of the first derivative, used for residual scaling
/// - `residual_scaling`: √(ρ'(s)) - Same as sqrt_rho1, stored separately for clarity
/// - `alpha_sq_norm`: α² = ρ''(s) / ρ'(s) - Ratio of second to first derivative,
///   used for Jacobian correction
///
/// where `s = ||r||²` is the squared norm of the residual.
#[derive(Debug, Clone)]
pub struct Corrector {
    sqrt_rho1: f64,
    residual_scaling: f64,
    alpha_sq_norm: f64,
}

impl Corrector {
    /// Create a new Corrector by evaluating the loss function at the given squared norm.
    ///
    /// # Arguments
    ///
    /// * `loss_function` - The robust loss function ρ(s)
    /// * `sq_norm` - The squared norm of the residual: `s = ||r||²`
    ///
    /// # Returns
    ///
    /// A `Corrector` instance with precomputed scaling factors
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::core::corrector::Corrector;
    /// use apex_solver::core::loss_functions::{Loss, HuberLoss};
    /// use nalgebra::DVector;
    ///
    /// let loss = HuberLoss::new(1.0).unwrap();
    /// let residual = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    /// let squared_norm = residual.dot(&residual); // 14.0
    ///
    /// let corrector = Corrector::new(&loss, squared_norm);
    /// // corrector is now ready to apply corrections
    /// ```
    pub fn new(loss_function: &dyn Loss, sq_norm: f64) -> Self {
        // Evaluate loss function: [ρ(s), ρ'(s), ρ''(s)]
        let rho = loss_function.evaluate(sq_norm);

        // Extract derivatives
        let rho_1 = rho[1]; // ρ'(s)
        let rho_2 = rho[2]; // ρ''(s)

        // Compute scaling factors
        let sqrt_rho1 = rho_1.sqrt(); // √(ρ'(s))

        // Handle special cases (common case: rho[2] <= 0)
        // This occurs when the loss function has no curvature correction needed
        if sq_norm == 0.0 || rho_2 <= 0.0 {
            return Self {
                sqrt_rho1,
                residual_scaling: sqrt_rho1,
                alpha_sq_norm: 0.0,
            };
        }

        // Compute alpha by solving the quadratic equation:
        // 0.5·α² - α - (ρ''/ρ')·s = 0
        //
        // This gives: α = 1 - √(1 + 2·s·ρ''/ρ')
        //
        // Reference: Ceres Solver corrector.cc
        // https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/corrector.cc
        let d = 1.0 + 2.0 * sq_norm * rho_2 / rho_1;
        let alpha = 1.0 - d.sqrt();

        Self {
            sqrt_rho1,
            residual_scaling: sqrt_rho1 / (1.0 - alpha),
            alpha_sq_norm: alpha / sq_norm,
        }
    }

    /// Apply correction to the Jacobian matrix.
    ///
    /// Transforms the Jacobian `J` into `J̃` according to the Ceres Solver corrector algorithm:
    ///
    /// ```text
    /// J̃ = √(ρ'(s)) · (J - α²·r·r^T·J)
    /// ```
    ///
    /// where:
    /// - `√(ρ'(s))` scales the Jacobian by the loss function weight
    /// - `α` is computed by solving the quadratic equation: 0.5·α² - α - (ρ''/ρ')·s = 0
    /// - The subtractive term `α²·r·r^T·J` is a rank-1 curvature correction
    ///
    /// # Arguments
    ///
    /// * `residual` - The original residual vector `r`
    /// * `jacobian` - Mutable reference to the Jacobian matrix (modified in-place)
    ///
    /// # Implementation Notes
    ///
    /// The correction is applied in-place for efficiency. The algorithm:
    /// 1. Scales all Jacobian entries by `√(ρ'(s))`
    /// 2. Adds the outer product correction: `α · (J^T r) · r^T / ||r||`
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::core::corrector::Corrector;
    /// use apex_solver::core::loss_functions::{Loss, HuberLoss};
    /// use nalgebra::{DVector, DMatrix};
    ///
    /// let loss = HuberLoss::new(1.0).unwrap();
    /// let residual = DVector::from_vec(vec![2.0, 1.0]);
    /// let squared_norm = residual.dot(&residual);
    ///
    /// let corrector = Corrector::new(&loss, squared_norm);
    ///
    /// let mut jacobian = DMatrix::from_row_slice(2, 3, &[
    ///     1.0, 0.0, 1.0,
    ///     0.0, 1.0, 1.0,
    /// ]);
    ///
    /// corrector.correct_jacobian(&residual, &mut jacobian);
    /// // jacobian is now corrected to account for the robust loss
    /// ```
    pub fn correct_jacobian(&self, residual: &DVector<f64>, jacobian: &mut DMatrix<f64>) {
        // Common case (rho[2] <= 0): only apply first-order correction
        // This is the most common scenario for well-behaved loss functions
        if self.alpha_sq_norm == 0.0 {
            *jacobian *= self.sqrt_rho1;
            return;
        }

        // Full correction with curvature term:
        // J̃ = √ρ₁ · (J - α²·r·r^T·J)
        //
        // This is the correct Ceres Solver algorithm:
        // 1. Compute r·r^T·J (outer product of residual with Jacobian)
        // 2. Subtract α²·r·r^T·J from J
        // 3. Scale result by √ρ₁
        //
        // Reference: Ceres Solver corrector.cc
        // https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/corrector.cc

        let r_rtj = residual * residual.transpose() * jacobian.clone();
        *jacobian = (jacobian.clone() - r_rtj * self.alpha_sq_norm) * self.sqrt_rho1;
    }

    /// Apply correction to the residual vector.
    ///
    /// Transforms the residual `r` into `r̃` by scaling:
    ///
    /// ```text
    /// r̃ = √(ρ'(s)) · r
    /// ```
    ///
    /// This ensures that `||r̃||² ≈ ρ(||r||²)`, i.e., the squared norm of the corrected
    /// residual approximates the robust cost.
    ///
    /// # Arguments
    ///
    /// * `residual` - Mutable reference to the residual vector (modified in-place)
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::core::corrector::Corrector;
    /// use apex_solver::core::loss_functions::{Loss, HuberLoss};
    /// use nalgebra::DVector;
    ///
    /// let loss = HuberLoss::new(1.0).unwrap();
    /// let mut residual = DVector::from_vec(vec![2.0, 3.0, 1.0]);
    /// let squared_norm = residual.dot(&residual);
    ///
    /// let corrector = Corrector::new(&loss, squared_norm);
    ///
    /// println!("Original residual norm: {}", residual.norm());
    /// corrector.correct_residuals(&mut residual);
    /// println!("Corrected residual norm: {}", residual.norm());
    /// // Outlier residuals are scaled down
    /// ```
    pub fn correct_residuals(&self, residual: &mut DVector<f64>) {
        // Simple scaling: r̃ = √(ρ'(s)) · r
        //
        // This downweights outliers (where ρ'(s) < 1) and leaves inliers
        // approximately unchanged (where ρ'(s) ≈ 1)
        *residual *= self.residual_scaling;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::loss_functions::{CauchyLoss, HuberLoss};

    #[test]
    fn test_corrector_huber_inlier() {
        // Test corrector behavior for an inlier (small residual)
        let loss = HuberLoss::new(1.0).unwrap();
        let residual = DVector::from_vec(vec![0.1, 0.2, 0.1]); // Small residual
        let squared_norm = residual.dot(&residual); // 0.06

        let corrector = Corrector::new(&loss, squared_norm);

        // For inliers, ρ'(s) ≈ 1, so scaling should be near 1
        assert!((corrector.sqrt_rho1 - 1.0).abs() < 1e-10);
        assert!((corrector.alpha_sq_norm).abs() < 1e-10); // ρ''(s) ≈ 0 for inliers

        // Corrected residual should be nearly unchanged
        let mut corrected_residual = residual.clone();
        corrector.correct_residuals(&mut corrected_residual);
        assert!((corrected_residual - residual).norm() < 1e-10);
    }

    #[test]
    fn test_corrector_huber_outlier() {
        // Test corrector behavior for an outlier (large residual)
        let loss = HuberLoss::new(1.0).unwrap();
        let residual = DVector::from_vec(vec![5.0, 5.0, 5.0]); // Large residual
        let squared_norm = residual.dot(&residual); // 75.0

        let corrector = Corrector::new(&loss, squared_norm);

        // For outliers, ρ'(s) < 1, so scaling should be < 1
        assert!(corrector.sqrt_rho1 < 1.0);
        assert!(corrector.sqrt_rho1 > 0.0);

        // Corrected residual should be downweighted
        let mut corrected_residual = residual.clone();
        corrector.correct_residuals(&mut corrected_residual);
        assert!(corrected_residual.norm() < residual.norm());
    }

    #[test]
    fn test_corrector_cauchy() {
        // Test corrector with Cauchy loss
        let loss = CauchyLoss::new(1.0).unwrap();
        let residual = DVector::from_vec(vec![2.0, 3.0]);
        let squared_norm = residual.dot(&residual); // 13.0

        let corrector = Corrector::new(&loss, squared_norm);

        // Cauchy loss should heavily downweight large residuals
        assert!(corrector.sqrt_rho1 < 1.0);
        assert!(corrector.sqrt_rho1 > 0.0);

        let mut corrected_residual = residual.clone();
        corrector.correct_residuals(&mut corrected_residual);
        assert!(corrected_residual.norm() < residual.norm());
    }

    #[test]
    fn test_corrector_jacobian() {
        // Test Jacobian correction
        let loss = HuberLoss::new(1.0).unwrap();
        let residual = DVector::from_vec(vec![2.0, 1.0]);
        let squared_norm = residual.dot(&residual);

        let corrector = Corrector::new(&loss, squared_norm);

        let mut jacobian = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);

        let original_jacobian = jacobian.clone();
        corrector.correct_jacobian(&residual, &mut jacobian);

        // Jacobian should be modified
        assert!(jacobian != original_jacobian);

        // Each element should be scaled and corrected
        // (Exact values depend on loss function derivatives)
    }
}
