//! Factor implementations for graph-based optimization problems.
//!
//! Factors (also called constraints or error functions) represent measurements or relationships
//! between variables in a factor graph. Each factor computes a residual (error) vector and its
//! Jacobian with respect to the connected variables.
//!
//! # Factor Graph Formulation
//!
//! In graph-based SLAM and bundle adjustment, the optimization problem is represented as:
//!
//! ```text
//! minimize Σ_i ||r_i(x)||²
//! ```
//!
//! where:
//! - `x` is the set of variables (poses, landmarks, etc.)
//! - `r_i(x)` is the residual function for factor i
//! - Each factor connects one or more variables
//!
//! # Factor Types
//!
//! ## Pose Factors
//! - **Between factors**: Relative pose constraints (SE2, SE3)
//! - **Prior factors**: Unary constraints on single variables
//!
//! ## Camera Projection Factors
//!
//! Use [`ProjectionFactor`](camera::ProjectionFactor) with a specific [`CameraModel`](camera::CameraModel).
//!
//! Supported camera models:
//! - [`PinholeCamera`](camera::PinholeCamera)
//! - [`DoubleSphereCamera`](camera::DoubleSphereCamera)
//! - [`EucmCamera`](camera::EucmCamera)
//! - [`FovCamera`](camera::FovCamera)
//! - [`KannalaBrandtCamera`](camera::KannalaBrandtCamera)
//! - [`RadTanCamera`](camera::RadTanCamera)
//! - [`UcmCamera`](camera::UcmCamera)
//!
//! # Linearization
//!
//! Each factor must provide a `linearize` method that computes:
//! 1. **Residual** `r(x)`: The error at the current variable values
//! 2. **Jacobian** `J = ∂r/∂x`: How the residual changes with each variable
//!
//! This information is used by the optimizer to compute parameter updates via Newton-type methods.

use nalgebra::{DMatrix, DVector};
use thiserror::Error;
use tracing::error;

// Pose factors
pub mod between_factor;
pub mod prior_factor;
pub mod projection_factor;

pub use between_factor::BetweenFactor;
pub use prior_factor::PriorFactor;
pub use projection_factor::ProjectionFactor;

// Camera module alias for backward compatibility
// Re-exports the apex-camera-models crate as `camera` module
pub mod camera {
    pub use apex_camera_models::*;
}

/// Factor-specific error types for apex-solver
#[derive(Debug, Clone, Error)]
pub enum FactorError {
    /// Invalid dimension mismatch between expected and actual
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    /// Invalid projection (point behind camera or outside valid range)
    #[error("Invalid projection: {0}")]
    InvalidProjection(String),

    /// Jacobian computation failed
    #[error("Jacobian computation failed: {0}")]
    JacobianFailed(String),

    /// Invalid parameter values
    #[error("Invalid parameter values: {0}")]
    InvalidParameters(String),

    /// Numerical instability detected
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
}

impl FactorError {
    /// Log the error with tracing::error and return self for chaining
    ///
    /// This method allows for a consistent error logging pattern throughout
    /// the factors module, ensuring all errors are properly recorded.
    ///
    /// # Example
    /// ```
    /// # use apex_solver::factors::FactorError;
    /// # fn operation() -> Result<(), FactorError> { Ok(()) }
    /// # fn example() -> Result<(), FactorError> {
    /// operation()
    ///     .map_err(|e| e.log())?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn log(self) -> Self {
        error!("{}", self);
        self
    }

    /// Log the error with the original source error for debugging context
    ///
    /// This method logs both the FactorError and the underlying error
    /// from external libraries or internal operations, providing full
    /// debugging context when errors occur.
    ///
    /// # Arguments
    /// * `source_error` - The original error (must implement Debug)
    ///
    /// # Example
    /// ```
    /// # use apex_solver::factors::FactorError;
    /// # fn compute_jacobian() -> Result<(), std::io::Error> { Ok(()) }
    /// # fn example() -> Result<(), FactorError> {
    /// compute_jacobian()
    ///     .map_err(|e| {
    ///         FactorError::JacobianFailed("Matrix computation failed".to_string())
    ///             .log_with_source(e)
    ///     })?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn log_with_source<E: std::fmt::Debug>(self, source_error: E) -> Self {
        error!("{} | Source: {:?}", self, source_error);
        self
    }
}

/// Result type for factor operations
pub type FactorResult<T> = Result<T, FactorError>;

/// Trait for factor (constraint) implementations in factor graph optimization.
///
/// A factor represents a measurement or constraint connecting one or more variables.
/// It computes the residual (error) and Jacobian for the current variable values,
/// which are used by the optimizer to minimize the total cost.
///
/// # Implementing Custom Factors
///
/// To create a custom factor:
/// 1. Implement this trait
/// 2. Define the residual function `r(x)` (how to compute error from variable values)
/// 3. Compute the Jacobian `J = ∂r/∂x` (analytically or numerically)
/// 4. Return the residual dimension
///
/// # Thread Safety
///
/// Factors must be `Send + Sync` to enable parallel residual/Jacobian evaluation.
///
/// # Example
///
/// ```
/// use apex_solver::factors::Factor;
/// use nalgebra::{DMatrix, DVector};
///
/// // Simple 1D range measurement factor
/// struct RangeFactor {
///     measurement: f64,  // Measured distance
/// }
///
/// impl Factor for RangeFactor {
///     fn linearize(&self, params: &[DVector<f64>], compute_jacobian: bool) -> (DVector<f64>, Option<DMatrix<f64>>) {
///         // params[0] is a 2D point [x, y]
///         let x = params[0][0];
///         let y = params[0][1];
///
///         // Residual: measured distance - actual distance
///         let predicted_distance = (x * x + y * y).sqrt();
///         let residual = DVector::from_vec(vec![self.measurement - predicted_distance]);
///
///         // Jacobian: ∂(residual)/∂[x, y]
///         let jacobian = if compute_jacobian {
///             Some(DMatrix::from_row_slice(1, 2, &[
///                 -x / predicted_distance,
///                 -y / predicted_distance,
///             ]))
///         } else {
///             None
///         };
///
///         (residual, jacobian)
///     }
///
///     fn get_dimension(&self) -> usize { 1 }
/// }
/// ```
pub trait Factor: Send + Sync {
    /// Compute the residual and Jacobian at the given parameter values.
    ///
    /// # Arguments
    ///
    /// * `params` - Slice of variable values (one `DVector` per connected variable)
    /// * `compute_jacobian` - Whether to compute the Jacobian matrix
    ///
    /// # Returns
    ///
    /// Tuple `(residual, jacobian)` where:
    /// - `residual`: N-dimensional error vector
    /// - `jacobian`: N × M matrix where M is the total DOF of all variables
    ///
    /// # Example
    ///
    /// For a between factor connecting two SE2 poses (3 DOF each):
    /// - Input: `params = [pose1 (3×1), pose2 (3×1)]`
    /// - Output: `(residual (3×1), jacobian (3×6))`
    fn linearize(
        &self,
        params: &[DVector<f64>],
        compute_jacobian: bool,
    ) -> (DVector<f64>, Option<DMatrix<f64>>);

    /// Get the dimension of the residual vector.
    ///
    /// # Returns
    ///
    /// Number of elements in the residual vector (number of constraints)
    ///
    /// # Example
    ///
    /// - SE2 between factor: 3 (dx, dy, dtheta)
    /// - SE3 between factor: 6 (translation + rotation)
    /// - Prior factor: dimension of the variable
    fn get_dimension(&self) -> usize;
}
