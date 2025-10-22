//! Robust loss functions for outlier rejection in nonlinear least squares optimization.
//!
//! Loss functions (also called robust cost functions or M-estimators) reduce the influence of
//! outlier measurements on the optimization result. In standard least squares, the cost is
//! the squared norm of residuals: `cost = Σ ||r_i||²`. With a robust loss function ρ(s), the
//! cost becomes: `cost = Σ ρ(||r_i||²)`.
//!
//! # Mathematical Formulation
//!
//! Each loss function implements the `Loss` trait, which evaluates:
//! - **ρ(s)**: The robust cost value
//! - **ρ'(s)**: First derivative (weight function)
//! - **ρ''(s)**: Second derivative (for corrector algorithm)
//!
//! The input `s = ||r||²` is the squared norm of the residual vector.
//!
//! # Usage in Optimization
//!
//! Loss functions are applied via the `Corrector` algorithm (see `corrector.rs`), which
//! modifies the residuals and Jacobians to account for the robust weighting. The optimization
//! then proceeds as if solving a reweighted least squares problem.
//!
//! # Available Loss Functions
//!
//! - [`HuberLoss`]: Quadratic for inliers, linear for outliers
//! - [`CauchyLoss`]: Heavier suppression of large residuals
//!
//! # Example
//!
//! ```
//! use apex_solver::core::loss_functions::{Loss, HuberLoss};
//!
//! let huber = HuberLoss::new(1.345).unwrap();
//!
//! // Evaluate for an inlier (small residual)
//! let s_inlier = 0.5;
//! let [rho, rho_prime, rho_double_prime] = huber.evaluate(s_inlier);
//! assert_eq!(rho, s_inlier); // Quadratic cost in inlier region
//! assert_eq!(rho_prime, 1.0); // Full weight
//!
//! // Evaluate for an outlier (large residual)
//! let s_outlier = 10.0;
//! let [rho, rho_prime, rho_double_prime] = huber.evaluate(s_outlier);
//! // rho grows linearly instead of quadratically
//! // rho_prime < 1.0, downweighting the outlier
//! ```

/// Trait for robust loss functions used in nonlinear least squares optimization.
///
/// A loss function transforms the squared residual `s = ||r||²` into a robust cost `ρ(s)`
/// that reduces the influence of outliers. The trait provides the cost value and its first
/// two derivatives, which are used by the `Corrector` to modify the optimization problem.
///
/// # Returns
///
/// The `evaluate` method returns a 3-element array: `[ρ(s), ρ'(s), ρ''(s)]`
/// - `ρ(s)`: Robust cost value
/// - `ρ'(s)`: First derivative (weight function)
/// - `ρ''(s)`: Second derivative
///
/// # Implementation Notes
///
/// - Loss functions should be smooth (at least C²) for optimization stability
/// - Typically ρ(0) = 0, ρ'(0) = 1, ρ''(0) = 0 (behaves like standard least squares near zero)
/// - For outliers, ρ'(s) should decrease to downweight large residuals
pub trait Loss: Send + Sync {
    /// Evaluate the loss function and its first two derivatives at squared residual `s`.
    ///
    /// # Arguments
    ///
    /// * `s` - The squared norm of the residual: `s = ||r||²` (always non-negative)
    ///
    /// # Returns
    ///
    /// Array `[ρ(s), ρ'(s), ρ''(s)]` containing the cost, first derivative, and second derivative
    fn evaluate(&self, s: f64) -> [f64; 3];
}

/// Huber loss function for moderate outlier rejection.
///
/// The Huber loss is quadratic for small residuals (inliers) and linear for large residuals
/// (outliers), providing a good balance between robustness and efficiency.
///
/// # Mathematical Definition
///
/// ```text
/// ρ(s) = {  s                           if s ≤ δ²
///        {  2δ√s - δ²                  if s > δ²
///
/// ρ'(s) = {  1                          if s ≤ δ²
///         {  δ / √s                    if s > δ²
///
/// ρ''(s) = {  0                         if s ≤ δ²
///          {  -δ / (2s^(3/2))          if s > δ²
/// ```
///
/// where `δ` is the scale parameter (threshold), and `s = ||r||²` is the squared residual norm.
///
/// # Properties
///
/// - **Inlier region** (s ≤ δ²): Behaves like standard least squares (quadratic cost)
/// - **Outlier region** (s > δ²): Cost grows linearly, limiting outlier influence
/// - **Transition point**: At s = δ², the function switches from quadratic to linear
///
/// # Scale Parameter Selection
///
/// Common choices for the scale parameter `δ`:
/// - **1.345**: Approximately 95% efficiency on Gaussian data (most common)
/// - **0.5-1.0**: More aggressive outlier rejection
/// - **2.0-3.0**: More lenient, closer to standard least squares
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{Loss, HuberLoss};
///
/// // Create Huber loss with scale = 1.345 (standard choice)
/// let huber = HuberLoss::new(1.345).unwrap();
///
/// // Small residual (inlier): ||r||² = 0.5
/// let [rho, rho_prime, rho_double_prime] = huber.evaluate(0.5);
/// assert_eq!(rho, 0.5);           // Quadratic: ρ(s) = s
/// assert_eq!(rho_prime, 1.0);     // Full weight
/// assert_eq!(rho_double_prime, 0.0);
///
/// // Large residual (outlier): ||r||² = 10.0
/// let [rho, rho_prime, rho_double_prime] = huber.evaluate(10.0);
/// // ρ(10) ≈ 6.69, grows linearly not quadratically
/// // ρ'(10) ≈ 0.425, downweighted to ~42.5% of original
/// ```
#[derive(Debug, Clone)]
pub struct HuberLoss {
    /// Scale parameter δ
    scale: f64,
    /// Cached value δ² for efficient computation
    scale2: f64,
}

impl HuberLoss {
    /// Create a new Huber loss function with the given scale parameter.
    ///
    /// # Arguments
    ///
    /// * `scale` - The threshold δ that separates inliers from outliers (must be positive)
    ///
    /// # Returns
    ///
    /// `Ok(HuberLoss)` if scale > 0, otherwise an error
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::core::loss_functions::HuberLoss;
    ///
    /// let huber = HuberLoss::new(1.345).unwrap();
    /// ```
    pub fn new(scale: f64) -> crate::core::ApexResult<Self> {
        if scale <= 0.0 {
            return Err(crate::core::ApexError::InvalidInput(
                "scale needs to be larger than zero".to_string(),
            ));
        }
        Ok(HuberLoss {
            scale,
            scale2: scale * scale,
        })
    }
}

impl Loss for HuberLoss {
    /// Evaluate Huber loss function: ρ(s), ρ'(s), ρ''(s).
    ///
    /// # Arguments
    ///
    /// * `s` - Squared residual norm: s = ||r||²
    ///
    /// # Returns
    ///
    /// `[ρ(s), ρ'(s), ρ''(s)]` - Cost, first derivative, second derivative
    fn evaluate(&self, s: f64) -> [f64; 3] {
        if s > self.scale2 {
            // Outlier region: s > δ²
            // Linear cost: ρ(s) = 2δ√s - δ²
            let r = s.sqrt(); // r = √s = ||r||
            let rho1 = (self.scale / r).max(f64::MIN); // ρ'(s) = δ / √s
            [
                2.0 * self.scale * r - self.scale2, // ρ(s)
                rho1,                               // ρ'(s)
                -rho1 / (2.0 * s),                  // ρ''(s) = -δ / (2s√s)
            ]
        } else {
            // Inlier region: s ≤ δ²
            // Quadratic cost: ρ(s) = s, ρ'(s) = 1, ρ''(s) = 0
            [s, 1.0, 0.0]
        }
    }
}

/// Cauchy loss function for aggressive outlier rejection.
///
/// The Cauchy loss (also called Lorentzian loss) provides stronger suppression of outliers
/// than Huber loss. It never fully rejects outliers but reduces their weight significantly.
///
/// # Mathematical Definition
///
/// ```text
/// ρ(s) = (δ²/2) * log(1 + s/δ²)
///
/// ρ'(s) = 1 / (1 + s/δ²)
///
/// ρ''(s) = -1 / (δ² * (1 + s/δ²)²)
/// ```
///
/// where `δ` is the scale parameter, and `s = ||r||²` is the squared residual norm.
///
/// # Properties
///
/// - **Smooth transition**: No sharp boundary between inliers and outliers
/// - **Logarithmic growth**: Cost grows very slowly for large residuals
/// - **Strong downweighting**: Large outliers receive very small weights
/// - **Non-convex**: Can have multiple local minima (harder to optimize than Huber)
///
/// # Scale Parameter Selection
///
/// Typical values:
/// - **2.3849**: Approximately 95% efficiency on Gaussian data
/// - **1.0-2.0**: More aggressive outlier rejection
/// - **3.0-5.0**: More lenient
///
/// # Comparison to Huber Loss
///
/// - **Cauchy**: Stronger outlier rejection, smoother, but non-convex (may converge to local minimum)
/// - **Huber**: Weaker outlier rejection, convex, more predictable convergence
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{Loss, CauchyLoss};
///
/// // Create Cauchy loss with scale = 2.3849 (standard choice)
/// let cauchy = CauchyLoss::new(2.3849).unwrap();
///
/// // Small residual: ||r||² = 0.5
/// let [rho, rho_prime, _] = cauchy.evaluate(0.5);
/// // ρ ≈ 0.47, slightly less than 0.5 (mild downweighting)
/// // ρ' ≈ 0.92, close to 1.0 (near full weight)
///
/// // Large residual: ||r||² = 100.0
/// let [rho, rho_prime, _] = cauchy.evaluate(100.0);
/// // ρ ≈ 8.0, logarithmic growth (much less than 100)
/// // ρ' ≈ 0.05, heavily downweighted (5% of original)
/// ```
pub struct CauchyLoss {
    /// Cached value δ² (scale squared)
    scale2: f64,
    /// Cached value 1/δ² for efficient computation
    c: f64,
}

impl CauchyLoss {
    /// Create a new Cauchy loss function with the given scale parameter.
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter δ (must be positive)
    ///
    /// # Returns
    ///
    /// `Ok(CauchyLoss)` if scale > 0, otherwise an error
    ///
    /// # Example
    ///
    /// ```
    /// use apex_solver::core::loss_functions::CauchyLoss;
    ///
    /// let cauchy = CauchyLoss::new(2.3849).unwrap();
    /// ```
    pub fn new(scale: f64) -> crate::core::ApexResult<Self> {
        if scale <= 0.0 {
            return Err(crate::core::ApexError::InvalidInput(
                "scale needs to be larger than zero".to_string(),
            ));
        }
        let scale2 = scale * scale;
        Ok(CauchyLoss {
            scale2,
            c: 1.0 / scale2,
        })
    }
}

impl Loss for CauchyLoss {
    /// Evaluate Cauchy loss function: ρ(s), ρ'(s), ρ''(s).
    ///
    /// # Arguments
    ///
    /// * `s` - Squared residual norm: s = ||r||²
    ///
    /// # Returns
    ///
    /// `[ρ(s), ρ'(s), ρ''(s)]` - Cost, first derivative, second derivative
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let sum = 1.0 + s * self.c; // 1 + s/δ²
        let inv = 1.0 / sum; // 1 / (1 + s/δ²)

        // Note: sum and inv are always positive, assuming s ≥ 0
        [
            self.scale2 * sum.log2(), // ρ(s) = (δ²/2) * log(1 + s/δ²)
            inv.max(f64::MIN),        // ρ'(s) = 1 / (1 + s/δ²)
            -self.c * (inv * inv),    // ρ''(s) = -1 / (δ² * (1 + s/δ²)²)
        ]
    }
}
