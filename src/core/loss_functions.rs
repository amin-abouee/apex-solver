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
//! ## Basic Loss Functions
//! - [`L2Loss`]: Standard least squares (no robustness)
//! - [`L1Loss`]: Absolute error (simple robust baseline)
//!
//! ## Moderate Robustness
//! - [`HuberLoss`]: Quadratic for inliers, linear for outliers (recommended for general use)
//! - [`FairLoss`]: Smooth transition with continuous derivatives
//! - [`CauchyLoss`]: Heavier suppression of large residuals
//!
//! ## Strong Robustness
//! - [`GemanMcClureLoss`]: Very strong outlier rejection
//! - [`WelschLoss`]: Exponential downweighting
//! - [`TukeyBiweightLoss`]: Complete outlier suppression (redescending)
//!
//! ## Specialized Functions
//! - [`AndrewsWaveLoss`]: Sine-based redescending M-estimator
//! - [`RamsayEaLoss`]: Exponential decay weighting
//! - [`TrimmedMeanLoss`]: Hard threshold cutoff
//! - [`LpNormLoss`]: Generalized Lp norm (flexible p parameter)
//!
//! ## Modern Adaptive
//! - [`BarronGeneralLoss`]: Unified framework encompassing many loss functions (CVPR 2019)
//!
//! # Loss Function Selection Guide
//!
//! | Use Case | Recommended Loss | Tuning Constant |
//! |----------|-----------------|-----------------|
//! | Clean data, no outliers | `L2Loss` | N/A |
//! | Few outliers (<5%) | `HuberLoss` | c = 1.345 |
//! | Moderate outliers (5-10%) | `FairLoss` or `CauchyLoss` | c = 1.3998 / 2.3849 |
//! | Many outliers (>10%) | `WelschLoss` or `TukeyBiweightLoss` | c = 2.9846 / 4.6851 |
//! | Severe outliers | `GemanMcClureLoss` | c = 1.0-2.0 |
//! | Adaptive/unknown | `BarronGeneralLoss` | α adaptive |
//!
//! # Example
//!
//! ```
//! use apex_solver::core::loss_functions::{LossFunction, HuberLoss};
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

use crate::core::CoreError;
use crate::error::ApexSolverResult;

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
pub trait LossFunction: Send + Sync {
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

/// L2 loss function (standard least squares, no robustness).
///
/// The L2 loss is the standard squared error used in ordinary least squares optimization.
/// It provides no outlier robustness and is optimal only when residuals follow a Gaussian
/// distribution.
///
/// # Mathematical Definition
///
/// ```text
/// ρ(s) = s
/// ρ'(s) = 1
/// ρ''(s) = 0
/// ```
///
/// where `s = ||r||²` is the squared residual norm.
///
/// # Properties
///
/// - **Convex**: Globally optimal solution
/// - **Not robust**: Outliers have full influence (squared!)
/// - **Optimal**: For Gaussian noise without outliers
/// - **Fast**: Simplest to compute
///
/// # Use Cases
///
/// - Clean data with known Gaussian noise
/// - Baseline comparison for robust methods
/// - When outliers are already filtered
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{LossFunction, L2Loss};
///
/// let l2 = L2Loss::new();
///
/// let [rho, rho_prime, rho_double_prime] = l2.evaluate(4.0);
/// assert_eq!(rho, 4.0);  // ρ(s) = s
/// assert_eq!(rho_prime, 1.0);  // Full weight
/// assert_eq!(rho_double_prime, 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct L2Loss;

impl L2Loss {
    /// Create a new L2 loss function (no parameters needed).
    pub fn new() -> Self {
        L2Loss
    }
}

impl Default for L2Loss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFunction for L2Loss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        [s, 1.0, 0.0]
    }
}

/// L1 loss function (absolute error, simple robust baseline).
///
/// The L1 loss uses absolute error instead of squared error, providing basic robustness
/// to outliers. It is optimal for Laplacian noise distributions.
///
/// # Mathematical Definition
///
/// ```text
/// ρ(s) = 2√s
/// ρ'(s) = 1/√s
/// ρ''(s) = -1/(2s^(3/2))
/// ```
///
/// where `s = ||r||²` is the squared residual norm.
///
/// # Properties
///
/// - **Convex**: Globally optimal solution
/// - **Moderately robust**: Linear growth vs quadratic
/// - **Unstable at zero**: Derivative undefined at s=0
/// - **Median estimator**: Minimizes to median instead of mean
///
/// # Use Cases
///
/// - Simple outlier rejection
/// - When median is preferred over mean
/// - Sparse optimization problems
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{LossFunction, L1Loss};
///
/// let l1 = L1Loss::new();
///
/// let [rho, rho_prime, _] = l1.evaluate(4.0);
/// assert!((rho - 4.0).abs() < 1e-10);  // ρ(4) = 2√4 = 4
/// assert!((rho_prime - 0.5).abs() < 1e-10);  // ρ'(4) = 1/√4 = 0.5
/// ```
#[derive(Debug, Clone, Copy)]
pub struct L1Loss;

impl L1Loss {
    /// Create a new L1 loss function (no parameters needed).
    pub fn new() -> Self {
        L1Loss
    }
}

impl Default for L1Loss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFunction for L1Loss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        if s < f64::EPSILON {
            // Near zero: use L2 to avoid singularity
            return [s, 1.0, 0.0];
        }
        let sqrt_s = s.sqrt();
        [
            2.0 * sqrt_s,              // ρ(s) = 2√s
            1.0 / sqrt_s,              // ρ'(s) = 1/√s
            -1.0 / (2.0 * s * sqrt_s), // ρ''(s) = -1/(2s√s)
        ]
    }
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
/// use apex_solver::core::loss_functions::{LossFunction, HuberLoss};
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
    pub fn new(scale: f64) -> ApexSolverResult<Self> {
        if scale <= 0.0 {
            return Err(
                CoreError::InvalidInput("scale needs to be larger than zero".to_string()).into(),
            );
        }
        Ok(HuberLoss {
            scale,
            scale2: scale * scale,
        })
    }
}

impl LossFunction for HuberLoss {
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
/// use apex_solver::core::loss_functions::{LossFunction, CauchyLoss};
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
    pub fn new(scale: f64) -> ApexSolverResult<Self> {
        if scale <= 0.0 {
            return Err(
                CoreError::InvalidInput("scale needs to be larger than zero".to_string()).into(),
            );
        }
        let scale2 = scale * scale;
        Ok(CauchyLoss {
            scale2,
            c: 1.0 / scale2,
        })
    }
}

impl LossFunction for CauchyLoss {
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
            self.scale2 * sum.ln() / 2.0, // ρ(s) = (δ²/2) * ln(1 + s/δ²)
            inv.max(f64::MIN),            // ρ'(s) = 1 / (1 + s/δ²)
            -self.c * (inv * inv),        // ρ''(s) = -1 / (δ² * (1 + s/δ²)²)
        ]
    }
}

/// Fair loss function with continuous smooth derivatives.
///
/// The Fair loss provides a good balance between robustness and stability with everywhere-defined
/// continuous derivatives up to third order. It yields a unique solution and is recommended
/// for general use when you need guaranteed smoothness.
///
/// # Mathematical Definition
///
/// ```text
/// ρ(s) = c² * (|x|/c - ln(1 + |x|/c))
/// ρ'(s) = |x| / (c + |x|)
/// ρ''(s) = sign(x) * c / ((c + |x|)²√s)
/// ```
///
/// where `c` is the scale parameter, `s = ||r||²`, and `x = √s = ||r||`.
///
/// # Properties
///
/// - **Smooth**: Continuous derivatives of first three orders
/// - **Unique solution**: Strictly convex near origin
/// - **Moderate robustness**: Between Huber and Cauchy
/// - **Stable**: No discontinuities in optimization
///
/// # Scale Parameter Selection
///
/// - **1.3998**: Approximately 95% efficiency on Gaussian data (recommended)
/// - **0.8-1.2**: More aggressive outlier rejection
/// - **2.0-3.0**: More lenient
///
/// # Comparison
///
/// - Smoother than Huber (no kink at threshold)
/// - Less aggressive than Cauchy
/// - Better behaved numerically than many redescending M-estimators
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{LossFunction, FairLoss};
///
/// let fair = FairLoss::new(1.3998).unwrap();
///
/// let [rho, rho_prime, _] = fair.evaluate(4.0);
/// // Smooth transition, no sharp corners
/// ```
#[derive(Debug, Clone)]
pub struct FairLoss {
    scale: f64,
}

impl FairLoss {
    /// Create a new Fair loss function with the given scale parameter.
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter c (must be positive)
    ///
    /// # Returns
    ///
    /// `Ok(FairLoss)` if scale > 0, otherwise an error
    pub fn new(scale: f64) -> ApexSolverResult<Self> {
        if scale <= 0.0 {
            return Err(
                CoreError::InvalidInput("scale needs to be larger than zero".to_string()).into(),
            );
        }
        Ok(FairLoss { scale })
    }
}

impl LossFunction for FairLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        if s < f64::EPSILON {
            return [s, 1.0, 0.0];
        }

        let x = s.sqrt(); // ||r||
        let abs_x = x.abs();
        let c_plus_x = self.scale + abs_x;

        // ρ(s) = c² * (|x|/c - ln(1 + |x|/c))
        let rho = self.scale * self.scale * (abs_x / self.scale - (1.0 + abs_x / self.scale).ln());

        // ρ'(s) = |x| / (c + |x|) * (1 / 2|x|) = 1 / (2(c + |x|))
        let rho_prime = 0.5 / c_plus_x;

        // ρ''(s) = -1 / (4s(c + |x|)²)
        let rho_double_prime = -1.0 / (4.0 * s * c_plus_x * c_plus_x);

        [rho, rho_prime, rho_double_prime]
    }
}

/// Geman-McClure loss function for very strong outlier rejection.
///
/// The Geman-McClure loss provides one of the strongest forms of outlier suppression,
/// with weights that decay rapidly for large residuals.
///
/// # Mathematical Definition
///
/// ```text
/// ρ(s) = s / (1 + s/c²)
/// ρ'(s) = 1 / (1 + s/c²)²
/// ρ''(s) = -2 / (c² * (1 + s/c²)³)
/// ```
///
/// where `c` is the scale parameter and `s = ||r||²`.
///
/// # Properties
///
/// - **Very strong rejection**: Weights decay as O(1/s²) for large s
/// - **Non-convex**: Multiple local minima possible
/// - **No unique solution**: Requires good initialization
/// - **Aggressive**: Use when outliers are severe
///
/// # Scale Parameter Selection
///
/// - **1.0-2.0**: Typical range
/// - **0.5-1.0**: Very aggressive (use with care)
/// - **2.0-4.0**: More lenient
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{LossFunction, GemanMcClureLoss};
///
/// let geman = GemanMcClureLoss::new(1.0).unwrap();
///
/// let [rho, rho_prime, _] = geman.evaluate(100.0);
/// // Very small weight for large outliers
/// ```
#[derive(Debug, Clone)]
pub struct GemanMcClureLoss {
    c: f64, // 1/scale²
}

impl GemanMcClureLoss {
    /// Create a new Geman-McClure loss function.
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter c (must be positive)
    pub fn new(scale: f64) -> ApexSolverResult<Self> {
        if scale <= 0.0 {
            return Err(
                CoreError::InvalidInput("scale needs to be larger than zero".to_string()).into(),
            );
        }
        let scale2 = scale * scale;
        Ok(GemanMcClureLoss { c: 1.0 / scale2 })
    }
}

impl LossFunction for GemanMcClureLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let denom = 1.0 + s * self.c; // 1 + s/c²
        let inv = 1.0 / denom;
        let inv2 = inv * inv;

        [
            s * inv,                    // ρ(s) = s / (1 + s/c²)
            inv2,                       // ρ'(s) = 1 / (1 + s/c²)²
            -2.0 * self.c * inv2 * inv, // ρ''(s) = -2 / (c²(1 + s/c²)³)
        ]
    }
}

/// Welsch loss function with exponential downweighting.
///
/// The Welsch loss (also called Leclerc loss) uses exponential decay to strongly
/// suppress outliers while maintaining smoothness. It completely suppresses very
/// large outliers (weight → 0 as s → ∞).
///
/// # Mathematical Definition
///
/// ```text
/// ρ(s) = c²/2 * (1 - exp(-s/c²))
/// ρ'(s) = (1/2) * exp(-s/c²)
/// ρ''(s) = -(1/2c²) * exp(-s/c²)
/// ```
///
/// where `c` is the scale parameter and `s = ||r||²`.
///
/// # Properties
///
/// - **Redescending**: Weights decrease to zero for large residuals
/// - **Smooth**: Infinitely differentiable
/// - **Strong suppression**: Exponential decay
/// - **Non-convex**: Requires good initialization
///
/// # Scale Parameter Selection
///
/// - **2.9846**: Approximately 95% efficiency on Gaussian data
/// - **2.0-2.5**: More aggressive
/// - **3.5-4.5**: More lenient
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{LossFunction, WelschLoss};
///
/// let welsch = WelschLoss::new(2.9846).unwrap();
///
/// let [rho, rho_prime, _] = welsch.evaluate(50.0);
/// // Weight approaches zero for large residuals
/// ```
#[derive(Debug, Clone)]
pub struct WelschLoss {
    scale2: f64,
    inv_scale2: f64,
}

impl WelschLoss {
    /// Create a new Welsch loss function.
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter c (must be positive)
    pub fn new(scale: f64) -> ApexSolverResult<Self> {
        if scale <= 0.0 {
            return Err(
                CoreError::InvalidInput("scale needs to be larger than zero".to_string()).into(),
            );
        }
        let scale2 = scale * scale;
        Ok(WelschLoss {
            scale2,
            inv_scale2: 1.0 / scale2,
        })
    }
}

impl LossFunction for WelschLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let exp_term = (-s * self.inv_scale2).exp();

        [
            (self.scale2 / 2.0) * (1.0 - exp_term), // ρ(s) = c²/2 * (1 - exp(-s/c²))
            0.5 * exp_term,                         // ρ'(s) = (1/2) * exp(-s/c²)
            -0.5 * self.inv_scale2 * exp_term,      // ρ''(s) = -(1/2c²) * exp(-s/c²)
        ]
    }
}

/// Tukey biweight loss function with complete outlier suppression.
///
/// The Tukey biweight (bisquare) loss completely suppresses outliers beyond a threshold,
/// setting their weight to exactly zero. This is a "redescending" M-estimator.
///
/// # Mathematical Definition
///
/// For |x| ≤ c:
/// ```text
/// ρ(s) = c²/6 * (1 - (1 - (x/c)²)³)
/// ρ'(s) = (1/2) * (1 - (x/c)²)²
/// ρ''(s) = -(x/c²) * (1 - (x/c)²)
/// ```
///
/// For |x| > c:
/// ```text
/// ρ(s) = c²/6
/// ρ'(s) = 0
/// ρ''(s) = 0
/// ```
///
/// where `c` is the scale parameter, `x = √s`, and `s = ||r||²`.
///
/// # Properties
///
/// - **Complete suppression**: Outliers have exactly zero weight
/// - **Redescending**: Weight goes to zero beyond threshold
/// - **Non-convex**: Multiple local minima
/// - **Aggressive**: Best for severe outlier contamination
///
/// # Scale Parameter Selection
///
/// - **4.6851**: Approximately 95% efficiency on Gaussian data
/// - **3.5-4.0**: More aggressive
/// - **5.5-6.5**: More lenient
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{LossFunction, TukeyBiweightLoss};
///
/// let tukey = TukeyBiweightLoss::new(4.6851).unwrap();
///
/// let [rho, rho_prime, _] = tukey.evaluate(25.0); // |x| = 5 > 4.6851
/// assert_eq!(rho_prime, 0.0); // Complete suppression
/// ```
#[derive(Debug, Clone)]
pub struct TukeyBiweightLoss {
    scale: f64,
    scale2: f64,
}

impl TukeyBiweightLoss {
    /// Create a new Tukey biweight loss function.
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter c (must be positive)
    pub fn new(scale: f64) -> ApexSolverResult<Self> {
        if scale <= 0.0 {
            return Err(
                CoreError::InvalidInput("scale needs to be larger than zero".to_string()).into(),
            );
        }
        Ok(TukeyBiweightLoss {
            scale,
            scale2: scale * scale,
        })
    }
}

impl LossFunction for TukeyBiweightLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let x = s.sqrt();

        if x > self.scale {
            // Complete outlier suppression
            [self.scale2 / 6.0, 0.0, 0.0]
        } else {
            let ratio = x / self.scale;
            let ratio2 = ratio * ratio;
            let one_minus_ratio2 = 1.0 - ratio2;
            let one_minus_ratio2_sq = one_minus_ratio2 * one_minus_ratio2;

            [
                (self.scale2 / 6.0) * (1.0 - one_minus_ratio2 * one_minus_ratio2_sq), // ρ(s)
                0.5 * one_minus_ratio2_sq,                                            // ρ'(s)
                -(ratio / self.scale2) * one_minus_ratio2,                            // ρ''(s)
            ]
        }
    }
}

/// Andrews sine wave loss function (redescending M-estimator).
///
/// The Andrews sine wave loss uses a periodic sine function to create a redescending
/// M-estimator that completely suppresses outliers beyond π*c.
///
/// # Mathematical Definition
///
/// For |x| ≤ πc:
/// ```text
/// ρ(s) = c² * (1 - cos(x/c))
/// ρ'(s) = (1/2) * sin(x/c)
/// ρ''(s) = (1/4c) * cos(x/c) / √s
/// ```
///
/// For |x| > πc:
/// ```text
/// ρ(s) = 2c²
/// ρ'(s) = 0
/// ρ''(s) = 0
/// ```
///
/// where `c` is the scale parameter, `x = √s`, and `s = ||r||²`.
///
/// # Properties
///
/// - **Periodic structure**: Sine-based weighting
/// - **Complete suppression**: Zero weight beyond πc
/// - **Redescending**: Smooth transition to zero
/// - **Non-convex**: Requires careful initialization
///
/// # Scale Parameter Selection
///
/// - **1.339**: Standard tuning constant
/// - **1.0-1.2**: More aggressive
/// - **1.5-2.0**: More lenient
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{LossFunction, AndrewsWaveLoss};
///
/// let andrews = AndrewsWaveLoss::new(1.339).unwrap();
///
/// let [rho, rho_prime, _] = andrews.evaluate(20.0);
/// // Weight is zero for large outliers
/// ```
#[derive(Debug, Clone)]
pub struct AndrewsWaveLoss {
    scale: f64,
    scale2: f64,
    threshold: f64, // π * scale
}

impl AndrewsWaveLoss {
    /// Create a new Andrews sine wave loss function.
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter c (must be positive)
    pub fn new(scale: f64) -> ApexSolverResult<Self> {
        if scale <= 0.0 {
            return Err(
                CoreError::InvalidInput("scale needs to be larger than zero".to_string()).into(),
            );
        }
        Ok(AndrewsWaveLoss {
            scale,
            scale2: scale * scale,
            threshold: std::f64::consts::PI * scale,
        })
    }
}

impl LossFunction for AndrewsWaveLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let x = s.sqrt();

        if x > self.threshold {
            // Complete suppression beyond π*c
            [2.0 * self.scale2, 0.0, 0.0]
        } else {
            let arg = x / self.scale;
            let sin_val = arg.sin();
            let cos_val = arg.cos();

            [
                self.scale2 * (1.0 - cos_val),                       // ρ(s)
                0.5 * sin_val,                                       // ρ'(s)
                (0.25 / self.scale) * cos_val / x.max(f64::EPSILON), // ρ''(s)
            ]
        }
    }
}

/// Ramsay Ea loss function with exponential decay.
///
/// The Ramsay Ea loss uses exponential weighting to provide smooth, strong
/// downweighting of outliers.
///
/// # Mathematical Definition
///
/// ```text
/// ρ(s) = a⁻² * (1 - exp(-a|x|) * (1 + a|x|))
/// ```
///
/// where `a` is the scale parameter, `x = √s`, and `s = ||r||²`.
///
/// # Properties
///
/// - **Exponential decay**: Smooth weight reduction
/// - **Strongly robust**: Good for heavy outliers
/// - **Smooth**: Continuous derivatives
/// - **Non-convex**: Needs good initialization
///
/// # Scale Parameter Selection
///
/// - **0.3**: Standard tuning constant
/// - **0.2-0.25**: More aggressive
/// - **0.35-0.5**: More lenient
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{LossFunction, RamsayEaLoss};
///
/// let ramsay = RamsayEaLoss::new(0.3).unwrap();
///
/// let [rho, rho_prime, _] = ramsay.evaluate(10.0);
/// // Exponential downweighting
/// ```
#[derive(Debug, Clone)]
pub struct RamsayEaLoss {
    scale: f64,
    inv_scale2: f64,
}

impl RamsayEaLoss {
    /// Create a new Ramsay Ea loss function.
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter a (must be positive)
    pub fn new(scale: f64) -> ApexSolverResult<Self> {
        if scale <= 0.0 {
            return Err(
                CoreError::InvalidInput("scale needs to be larger than zero".to_string()).into(),
            );
        }
        Ok(RamsayEaLoss {
            scale,
            inv_scale2: 1.0 / (scale * scale),
        })
    }
}

impl LossFunction for RamsayEaLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        let x = s.sqrt();
        let ax = self.scale * x;
        let exp_term = (-ax).exp();

        // ρ(s) = a⁻² * (1 - exp(-a|x|) * (1 + a|x|))
        let rho = self.inv_scale2 * (1.0 - exp_term * (1.0 + ax));

        // ρ'(s) = (1/2) * exp(-a|x|)
        let rho_prime = 0.5 * exp_term;

        // ρ''(s) = -(a/4|x|) * exp(-a|x|)
        let rho_double_prime = -(self.scale / (4.0 * x.max(f64::EPSILON))) * exp_term;

        [rho, rho_prime, rho_double_prime]
    }
}

/// Trimmed mean loss function with hard threshold.
///
/// The trimmed mean loss is the simplest redescending estimator, applying a hard
/// cutoff at a threshold. Residuals below the threshold use L2 loss, those above
/// contribute a constant.
///
/// # Mathematical Definition
///
/// For s ≤ c²:
/// ```text
/// ρ(s) = s/2
/// ρ'(s) = 1/2
/// ρ''(s) = 0
/// ```
///
/// For s > c²:
/// ```text
/// ρ(s) = c²/2
/// ρ'(s) = 0
/// ρ''(s) = 0
/// ```
///
/// where `c` is the scale parameter and `s = ||r||²`.
///
/// # Properties
///
/// - **Simple**: Easiest to understand and implement
/// - **Hard cutoff**: Discontinuous weight function
/// - **Robust**: Completely ignores large outliers
/// - **Unstable**: Discontinuity can cause optimization issues
///
/// # Scale Parameter Selection
///
/// - **2.0**: Standard tuning constant
/// - **1.5**: More aggressive
/// - **3.0**: More lenient
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{LossFunction, TrimmedMeanLoss};
///
/// let trimmed = TrimmedMeanLoss::new(2.0).unwrap();
///
/// let [rho, rho_prime, _] = trimmed.evaluate(5.0);
/// assert_eq!(rho_prime, 0.0); // Beyond threshold
/// ```
#[derive(Debug, Clone)]
pub struct TrimmedMeanLoss {
    scale2: f64,
}

impl TrimmedMeanLoss {
    /// Create a new trimmed mean loss function.
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale parameter c (must be positive)
    pub fn new(scale: f64) -> ApexSolverResult<Self> {
        if scale <= 0.0 {
            return Err(
                CoreError::InvalidInput("scale needs to be larger than zero".to_string()).into(),
            );
        }
        Ok(TrimmedMeanLoss {
            scale2: scale * scale,
        })
    }
}

impl LossFunction for TrimmedMeanLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        if s <= self.scale2 {
            [s / 2.0, 0.5, 0.0]
        } else {
            [self.scale2 / 2.0, 0.0, 0.0]
        }
    }
}

/// Generalized Lp norm loss function.
///
/// The Lp norm loss allows flexible control over robustness through the p parameter.
/// It interpolates between L1 (p=1) and L2 (p=2) losses.
///
/// # Mathematical Definition
///
/// ```text
/// ρ(s) = |x|^p = s^(p/2)
/// ρ'(s) = (p/2) * s^(p/2-1)
/// ρ''(s) = (p/2) * (p/2-1) * s^(p/2-2)
/// ```
///
/// where `p` is the norm parameter, `x = √s`, and `s = ||r||²`.
///
/// # Properties
///
/// - **Flexible**: Tune robustness with p parameter
/// - **p=2**: L2 norm (standard least squares)
/// - **p=1**: L1 norm (robust median estimator)
/// - **p<1**: Very robust (non-convex)
/// - **1<p<2**: Compromise between L1 and L2
///
/// # Parameter Selection
///
/// - **p = 2.0**: No robustness (L2 loss)
/// - **p = 1.5**: Moderate robustness
/// - **p = 1.2**: Strong robustness
/// - **p = 1.0**: L1 loss (median)
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{LossFunction, LpNormLoss};
///
/// let lp = LpNormLoss::new(1.5).unwrap();
///
/// let [rho, rho_prime, _] = lp.evaluate(4.0);
/// // Between L1 and L2 behavior
/// ```
#[derive(Debug, Clone)]
pub struct LpNormLoss {
    p: f64,
}

impl LpNormLoss {
    /// Create a new Lp norm loss function.
    ///
    /// # Arguments
    ///
    /// * `p` - The norm parameter (0 < p ≤ 2 for practical use)
    pub fn new(p: f64) -> ApexSolverResult<Self> {
        if p <= 0.0 {
            return Err(CoreError::InvalidInput("p must be positive".to_string()).into());
        }
        Ok(LpNormLoss { p })
    }
}

impl LossFunction for LpNormLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        if s < f64::EPSILON {
            return [s, 1.0, 0.0];
        }

        let exp_rho = self.p / 2.0;
        let exp_rho_prime = exp_rho - 1.0;
        let exp_rho_double_prime = exp_rho_prime - 1.0;

        [
            s.powf(exp_rho),                                        // ρ(s) = s^(p/2)
            exp_rho * s.powf(exp_rho_prime),                        // ρ'(s) = (p/2) * s^(p/2-1)
            exp_rho * exp_rho_prime * s.powf(exp_rho_double_prime), // ρ''(s)
        ]
    }
}

/// Barron's general and adaptive robust loss function (CVPR 2019).
///
/// The Barron loss is a unified framework that encompasses many classic loss functions
/// (L2, Charbonnier, Cauchy, Geman-McClure, Welsch) through a single shape parameter α.
/// It can also adapt α automatically during optimization.
///
/// # Mathematical Definition
///
/// ```text
/// ρ(s, α, c) = (|α|/c²) * (|(x/c)² * |α|/2 + 1|^(α/2) - 1)
/// ```
///
/// where `α` controls robustness, `c` is scale, `x = √s`, and `s = ||r||²`.
///
/// # Special Cases (by α value)
///
/// - **α = 2**: L2 loss (no robustness)
/// - **α = 1**: Charbonnier/Pseudo-Huber loss
/// - **α = 0**: Cauchy loss
/// - **α = -1**: Welsch loss
/// - **α = -2**: Geman-McClure loss
/// - **α → -∞**: L0 "norm" (binary)
///
/// # Properties
///
/// - **Unified**: Single framework for many loss functions
/// - **Adaptive**: Can learn optimal α during training
/// - **Smooth**: Continuously differentiable in α
/// - **Modern**: State-of-the-art from computer vision research
///
/// # Parameter Selection
///
/// **Fixed α (manual tuning):**
/// - α = 2.0: Clean data, no outliers
/// - α = 0.0 to 1.0: Moderate outliers
/// - α = -2.0 to 0.0: Heavy outliers
///
/// **Scale c:**
/// - c = 1.0: Standard choice
/// - Adjust based on expected residual magnitude
///
/// # References
///
/// Barron, J. T. (2019). A general and adaptive robust loss function.
/// IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{LossFunction, BarronGeneralLoss};
///
/// // Cauchy-like behavior
/// let barron = BarronGeneralLoss::new(0.0, 1.0).unwrap();
///
/// let [rho, rho_prime, _] = barron.evaluate(4.0);
/// // Behaves like Cauchy loss
/// ```
#[derive(Debug, Clone)]
pub struct BarronGeneralLoss {
    alpha: f64,
    scale: f64,
    scale2: f64,
}

impl BarronGeneralLoss {
    /// Create a new Barron general robust loss function.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The shape parameter (controls robustness)
    /// * `scale` - The scale parameter c (must be positive)
    pub fn new(alpha: f64, scale: f64) -> ApexSolverResult<Self> {
        if scale <= 0.0 {
            return Err(CoreError::InvalidInput("scale must be positive".to_string()).into());
        }
        Ok(BarronGeneralLoss {
            alpha,
            scale,
            scale2: scale * scale,
        })
    }
}

impl LossFunction for BarronGeneralLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        // Handle special case α ≈ 0 (Cauchy loss)
        if self.alpha.abs() < 1e-6 {
            let denom = 1.0 + s / self.scale2;
            let inv = 1.0 / denom;
            return [
                (self.scale2 / 2.0) * denom.ln(),
                inv.max(f64::MIN),
                -inv * inv / self.scale2,
            ];
        }

        // Handle special case α ≈ 2 (L2 loss)
        if (self.alpha - 2.0).abs() < 1e-6 {
            return [s, 1.0, 0.0];
        }

        // General case
        let x = s.sqrt();
        let normalized = x / self.scale;
        let normalized2 = normalized * normalized;

        let inner = self.alpha.abs() / 2.0 * normalized2 + 1.0;
        let power = inner.powf(self.alpha / 2.0);

        // ρ(s) = (|α|/c²) * (power - 1)
        let rho = (self.alpha.abs() / self.scale2) * (power - 1.0);

        // ρ'(s) = (1/2) * inner^(α/2 - 1)
        let rho_prime = 0.5 * inner.powf(self.alpha / 2.0 - 1.0);

        // ρ''(s) = (α - 2)/(4c²) * inner^(α/2 - 2)
        let rho_double_prime =
            (self.alpha - 2.0) / (4.0 * self.scale2) * inner.powf(self.alpha / 2.0 - 2.0);

        [rho, rho_prime, rho_double_prime]
    }
}

/// Student's t-distribution loss function (robust M-estimator).
///
/// The t-distribution loss is derived from the negative log-likelihood of Student's
/// t-distribution. It provides heavy tails for robustness against outliers, with the
/// degrees of freedom parameter ν controlling the tail heaviness.
///
/// # Mathematical Definition
///
/// ```text
/// ρ(s) = (ν + 1)/2 · log(1 + s/ν)
/// ρ'(s) = (ν + 1)/(2(ν + s))
/// ρ''(s) = -(ν + 1)/(2(ν + s)²)
/// ```
///
/// where `ν` is the degrees of freedom and `s = ||r||²` is the squared residual norm.
///
/// # Properties
///
/// - **Heavy tails**: Provides robustness through heavier tails than Gaussian
/// - **Parameter control**: Small ν → heavy tails (more robust), large ν → Gaussian (less robust)
/// - **Well-founded**: Based on maximum likelihood estimation with t-distribution
/// - **Smooth**: Continuous derivatives for all s > 0
///
/// # Degrees of Freedom Selection
///
/// - **ν = 3-4**: Very robust, heavy outlier suppression
/// - **ν = 5**: Recommended default, good balance
/// - **ν = 10**: Moderate robustness
/// - **ν → ∞**: Converges to Gaussian (L2 loss)
///
/// # Use Cases
///
/// - Robust regression with unknown outlier distribution
/// - SLAM and pose graph optimization with loop closure outliers
/// - Bundle adjustment with incorrect feature matches
/// - Any optimization problem with heavy-tailed noise
///
/// # References
///
/// - Student's t-distribution is widely used in robust statistics
/// - Applied in robust SLAM (e.g., Chebrolu et al. 2021, Agarwal et al.)
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{LossFunction, TDistributionLoss};
///
/// let t_loss = TDistributionLoss::new(5.0).unwrap();
///
/// let [rho, rho_prime, _] = t_loss.evaluate(4.0);
/// // Robust to outliers with heavy tails
/// ```
#[derive(Debug, Clone)]
pub struct TDistributionLoss {
    nu: f64,             // Degrees of freedom
    half_nu_plus_1: f64, // (ν + 1)/2 (cached)
}

impl TDistributionLoss {
    /// Create a new Student's t-distribution loss function.
    ///
    /// # Arguments
    ///
    /// * `nu` - Degrees of freedom (must be positive)
    ///
    /// # Recommended Values
    ///
    /// - ν = 5.0: Default, good balance between robustness and efficiency
    /// - ν = 3.0-4.0: More robust to outliers
    /// - ν = 10.0: Less aggressive, closer to Gaussian
    pub fn new(nu: f64) -> ApexSolverResult<Self> {
        if nu <= 0.0 {
            return Err(
                CoreError::InvalidInput("degrees of freedom must be positive".to_string()).into(),
            );
        }
        Ok(TDistributionLoss {
            nu,
            half_nu_plus_1: (nu + 1.0) / 2.0,
        })
    }
}

impl LossFunction for TDistributionLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        // ρ(s) = (ν + 1)/2 · log(1 + s/ν)
        let inner = 1.0 + s / self.nu;
        let rho = self.half_nu_plus_1 * inner.ln();

        // ρ'(s) = (ν + 1)/(2(ν + s))
        let denom = self.nu + s;
        let rho_prime = self.half_nu_plus_1 / denom;

        // ρ''(s) = -(ν + 1)/(2(ν + s)²)
        let rho_double_prime = -self.half_nu_plus_1 / (denom * denom);

        [rho, rho_prime, rho_double_prime]
    }
}

/// Adaptive Barron loss function (simplified version).
///
/// This is a convenience wrapper around `BarronGeneralLoss` with recommended default
/// parameters for adaptive robust optimization. Based on Chebrolu et al. (2021) RAL paper
/// "Adaptive Robust Kernels for Non-Linear Least Squares Problems".
///
/// # Mathematical Definition
///
/// For α ≠ 0:
/// ```text
/// ρ(s) = |α - 2|/α · ((s/c² + 1)^(α/2) - 1)
/// ```
///
/// For α = 0 (Cauchy-like):
/// ```text
/// ρ(s) = log(s/(2c²) + 1)
/// ```
///
/// where `α` is the shape parameter and `c` is the scale parameter.
///
/// # Properties
///
/// - **Adaptive**: Can approximate many M-estimators (Huber, Cauchy, Geman-McClure, etc.)
/// - **Shape parameter α**: Controls the robustness level
/// - **Scale parameter c**: Controls the transition point
/// - **Unified framework**: Single loss function family
///
/// # Parameter Selection
///
/// **Default (α = 0.0, c = 1.0):**
/// - Cauchy-like behavior
/// - Good general-purpose robust loss
/// - Suitable for moderate to heavy outliers
///
/// **Other values:**
/// - α = 2.0: L2 loss (no robustness)
/// - α = 1.0: Pseudo-Huber/Charbonnier-like
/// - α = -2.0: Geman-McClure-like (very robust)
///
/// # Note on Adaptivity
///
/// This simplified version uses fixed parameters. The full adaptive version from
/// Chebrolu et al. requires iterative estimation of α based on residual distribution,
/// which would require integration into the optimizer's main loop.
///
/// # References
///
/// Chebrolu, N., Läbe, T., Vysotska, O., Behley, J., & Stachniss, C. (2021).
/// Adaptive robust kernels for non-linear least squares problems.
/// IEEE Robotics and Automation Letters, 6(2), 2240-2247.
///
/// Barron, J. T. (2019). A general and adaptive robust loss function.
/// IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
///
/// # Example
///
/// ```
/// use apex_solver::core::loss_functions::{LossFunction, AdaptiveBarronLoss};
///
/// // Default Cauchy-like behavior
/// let adaptive = AdaptiveBarronLoss::new(0.0, 1.0).unwrap();
///
/// let [rho, rho_prime, _] = adaptive.evaluate(4.0);
/// // Adaptive robust behavior
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveBarronLoss {
    inner: BarronGeneralLoss,
}

impl AdaptiveBarronLoss {
    /// Create a new adaptive Barron loss function.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Shape parameter (default: 0.0 for Cauchy-like)
    /// * `scale` - Scale parameter c (must be positive)
    ///
    /// # Recommended Defaults
    ///
    /// - α = 0.0, c = 1.0: General-purpose robust loss
    pub fn new(alpha: f64, scale: f64) -> ApexSolverResult<Self> {
        Ok(AdaptiveBarronLoss {
            inner: BarronGeneralLoss::new(alpha, scale)?,
        })
    }

    /// Create default instance without validation (alpha=0.0, scale=1.0).
    ///
    /// This is safe because the default parameters are mathematically valid.
    const fn new_default() -> Self {
        AdaptiveBarronLoss {
            inner: BarronGeneralLoss {
                alpha: 0.0,
                scale: 1.0,
                scale2: 1.0,
            },
        }
    }
}

impl LossFunction for AdaptiveBarronLoss {
    fn evaluate(&self, s: f64) -> [f64; 3] {
        self.inner.evaluate(s)
    }
}

impl Default for AdaptiveBarronLoss {
    /// Creates default AdaptiveBarronLoss with validated parameters (alpha=0.0, scale=1.0).
    fn default() -> Self {
        Self::new_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    const EPSILON: f64 = 1e-6;

    /// Helper function to test derivatives numerically
    fn numerical_derivative(loss: &dyn LossFunction, s: f64, h: f64) -> (f64, f64) {
        let [rho_plus, _, _] = loss.evaluate(s + h);
        let [rho_minus, _, _] = loss.evaluate(s - h);
        let [rho, _, _] = loss.evaluate(s);

        // First derivative: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        let rho_prime_numerical = (rho_plus - rho_minus) / (2.0 * h);

        // Second derivative: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
        let rho_double_prime_numerical = (rho_plus - 2.0 * rho + rho_minus) / (h * h);

        (rho_prime_numerical, rho_double_prime_numerical)
    }

    #[test]
    fn test_l2_loss() -> TestResult {
        let loss = L2Loss;

        // Test at s = 0
        let [rho, rho_prime, rho_double_prime] = loss.evaluate(0.0);
        assert_eq!(rho, 0.0);
        assert_eq!(rho_prime, 1.0);
        assert_eq!(rho_double_prime, 0.0);

        // Test at s = 4.0
        let [rho, rho_prime, rho_double_prime] = loss.evaluate(4.0);
        assert_eq!(rho, 4.0);
        assert_eq!(rho_prime, 1.0);
        assert_eq!(rho_double_prime, 0.0);

        Ok(())
    }

    #[test]
    fn test_l1_loss() -> TestResult {
        let loss = L1Loss;

        // Test at s = 0 (should handle gracefully)
        let [rho, rho_prime, rho_double_prime] = loss.evaluate(0.0);
        assert_eq!(rho, 0.0);
        assert!(rho_prime.is_finite());
        assert!(rho_double_prime.is_finite());

        // Test at s = 4.0 (√s = 2.0, ρ(s) = 2√s = 4.0)
        let [rho, rho_prime, _] = loss.evaluate(4.0);
        assert!((rho - 4.0).abs() < EPSILON); // ρ(s) = 2√s = 2*2 = 4
        assert!((rho_prime - 0.5).abs() < EPSILON); // ρ'(s) = 1/√s = 1/2 = 0.5

        Ok(())
    }

    #[test]
    fn test_fair_loss() -> TestResult {
        let loss = FairLoss::new(1.3999)?;

        // Test at s = 0 (special case handling)
        let [rho, rho_prime, rho_double_prime] = loss.evaluate(0.0);
        assert_eq!(rho, 0.0);
        assert_eq!(rho_prime, 1.0);
        assert_eq!(rho_double_prime, 0.0);

        // Test inlier region (s = 1.0)
        let [_, rho_prime, _] = loss.evaluate(1.0);
        assert!(rho_prime > 0.2 && rho_prime < 0.25); // ρ'(s) = 1/(2(c+|x|)) where x=1, c≈1.4 → ~0.208

        // Test outlier region (s = 100.0)
        let [_, rho_prime_outlier, _] = loss.evaluate(100.0);
        assert!(rho_prime_outlier < rho_prime); // Weight should decrease

        // Test that derivatives are finite and reasonable
        let [_, rho_prime_4, rho_double_prime_4] = loss.evaluate(4.0);
        assert!(rho_prime_4.is_finite() && rho_prime_4 > 0.0);
        assert!(rho_double_prime_4.is_finite() && rho_double_prime_4 < 0.0); // Convex near origin

        Ok(())
    }

    #[test]
    fn test_geman_mcclure_loss() -> TestResult {
        let loss = GemanMcClureLoss::new(1.0)?;

        // Test at s = 0
        let [rho, rho_prime, _] = loss.evaluate(0.0);
        assert_eq!(rho, 0.0);
        assert!((rho_prime - 1.0).abs() < EPSILON);

        // Test outlier suppression
        let [_, rho_prime_small, _] = loss.evaluate(1.0);
        let [_, rho_prime_large, _] = loss.evaluate(100.0);
        assert!(rho_prime_large < rho_prime_small);
        assert!(rho_prime_large < 0.1); // Strong suppression

        // Verify derivatives
        let s = 2.0;
        let [_, rho_prime, rho_double_prime] = loss.evaluate(s);
        let (rho_prime_num, rho_double_prime_num) = numerical_derivative(&loss, s, 1e-5);
        assert!((rho_prime - rho_prime_num).abs() < 1e-4);
        assert!((rho_double_prime - rho_double_prime_num).abs() < 1e-3);

        Ok(())
    }

    #[test]
    fn test_welsch_loss() -> TestResult {
        let loss = WelschLoss::new(2.9846)?;

        // Test at s = 0: ρ'(0) = 0.5 * exp(0) = 0.5
        let [rho, rho_prime, _] = loss.evaluate(0.0);
        assert_eq!(rho, 0.0);
        assert!((rho_prime - 0.5).abs() < EPSILON); // ρ'(s) = 0.5 * exp(-s/c²), at s=0: 0.5

        // Test redescending behavior
        let [_, rho_prime_10, _] = loss.evaluate(10.0);
        let [_, rho_prime_100, _] = loss.evaluate(100.0);
        assert!(rho_prime_100 < rho_prime_10);
        assert!(rho_prime_100 < 0.01); // Nearly zero for large outliers

        // Verify derivatives
        let s = 5.0;
        let [_, rho_prime, rho_double_prime] = loss.evaluate(s);
        let (rho_prime_num, rho_double_prime_num) = numerical_derivative(&loss, s, 1e-5);
        assert!((rho_prime - rho_prime_num).abs() < 1e-4);
        assert!((rho_double_prime - rho_double_prime_num).abs() < 1e-3);

        Ok(())
    }

    #[test]
    fn test_tukey_biweight_loss() -> TestResult {
        let loss = TukeyBiweightLoss::new(4.6851)?;

        // Test at s = 0: ρ'(0) = 0.5 * (1-0)^2 = 0.5
        let [rho, rho_prime, _] = loss.evaluate(0.0);
        assert_eq!(rho, 0.0);
        assert!((rho_prime - 0.5).abs() < EPSILON); // ρ'(s) = 0.5 * (1 - ratio²)², at s=0: 0.5

        // Test within threshold
        let scale2 = 4.6851 * 4.6851;
        let [_, rho_prime_in, _] = loss.evaluate(scale2 * 0.5);
        assert!(rho_prime_in > 0.05);

        // Test beyond threshold (complete suppression)
        let [_, rho_prime_out, _] = loss.evaluate(scale2 * 1.5);
        assert_eq!(rho_prime_out, 0.0);

        // Test that derivatives are finite and reasonable
        let [_, rho_prime_5, rho_double_prime_5] = loss.evaluate(5.0);
        assert!(rho_prime_5.is_finite() && rho_prime_5 > 0.0);
        assert!(rho_double_prime_5.is_finite() && rho_double_prime_5 < 0.0);

        Ok(())
    }

    #[test]
    fn test_andrews_wave_loss() -> TestResult {
        let loss = AndrewsWaveLoss::new(1.339)?;

        // Test at s = 0: ρ'(0) = 0.5 * sin(0) = 0
        let [rho, rho_prime, _] = loss.evaluate(0.0);
        assert_eq!(rho, 0.0);
        assert!(rho_prime.abs() < EPSILON); // ρ'(s) = 0.5 * sin(x/c), at s=0: 0

        // Test within threshold (small s where sin(x/c) gives moderate weight)
        let [_, rho_prime_in, _] = loss.evaluate(1.0);
        assert!(rho_prime_in > 0.33 && rho_prime_in < 0.35); // ~0.3397

        // Test beyond threshold
        let scale = 1.339;
        let [_, rho_prime_out, _] = loss.evaluate((scale * std::f64::consts::PI + 0.1).powi(2));
        assert!(rho_prime_out.abs() < 0.01);

        // Test that derivatives are finite
        let [_, rho_prime_1, rho_double_prime_1] = loss.evaluate(1.0);
        assert!(rho_prime_1.is_finite() && rho_prime_1 > 0.0);
        assert!(rho_double_prime_1.is_finite());

        Ok(())
    }

    #[test]
    fn test_ramsay_ea_loss() -> TestResult {
        let loss = RamsayEaLoss::new(0.3)?;

        // Test at s = 0 (should handle gracefully)
        let [rho, _, _] = loss.evaluate(0.0);
        assert_eq!(rho, 0.0);

        // Test exponential decay behavior
        let [_, rho_prime_small, _] = loss.evaluate(1.0);
        let [_, rho_prime_large, _] = loss.evaluate(100.0);
        assert!(rho_prime_large < rho_prime_small);

        // Verify derivatives
        let s = 4.0;
        let [_, rho_prime, rho_double_prime] = loss.evaluate(s);
        let (rho_prime_num, rho_double_prime_num) = numerical_derivative(&loss, s, 1e-5);
        assert!((rho_prime - rho_prime_num).abs() < 1e-4);
        assert!((rho_double_prime - rho_double_prime_num).abs() < 1e-3);

        Ok(())
    }

    #[test]
    fn test_trimmed_mean_loss() -> TestResult {
        let loss = TrimmedMeanLoss::new(2.0)?;
        let scale2 = 4.0;

        // Test below threshold (L2 behavior)
        let [rho, rho_prime, rho_double_prime] = loss.evaluate(2.0);
        assert!((rho - 1.0).abs() < EPSILON);
        assert!((rho_prime - 0.5).abs() < EPSILON);
        assert_eq!(rho_double_prime, 0.0);

        // Test above threshold (constant)
        let [rho_out, rho_prime_out, rho_double_prime_out] = loss.evaluate(10.0);
        assert!((rho_out - scale2 / 2.0).abs() < EPSILON);
        assert_eq!(rho_prime_out, 0.0);
        assert_eq!(rho_double_prime_out, 0.0);

        Ok(())
    }

    #[test]
    fn test_lp_norm_loss() -> TestResult {
        // Test L1 (p = 1)
        let l1 = LpNormLoss::new(1.0)?;
        let [rho_l1, _, _] = l1.evaluate(4.0);
        assert!((rho_l1 - 2.0).abs() < EPSILON); // ||r||₁ = 2

        // Test L2 (p = 2)
        let l2 = LpNormLoss::new(2.0)?;
        let [rho_l2, rho_prime_l2, rho_double_prime_l2] = l2.evaluate(4.0);
        assert!((rho_l2 - 4.0).abs() < EPSILON);
        assert!((rho_prime_l2 - 1.0).abs() < EPSILON);
        assert_eq!(rho_double_prime_l2, 0.0);

        // Test fractional p (p = 0.5)
        let l05 = LpNormLoss::new(0.5)?;
        let [_, rho_prime_05, _] = l05.evaluate(4.0);
        assert!(rho_prime_05 < 1.0); // Robust behavior

        // Verify derivatives for p = 1.5
        let loss = LpNormLoss::new(1.5)?;
        let s = 4.0;
        let [_, rho_prime, rho_double_prime] = loss.evaluate(s);
        let (rho_prime_num, rho_double_prime_num) = numerical_derivative(&loss, s, 1e-5);
        assert!((rho_prime - rho_prime_num).abs() < 1e-4);
        assert!((rho_double_prime - rho_double_prime_num).abs() < 1e-3);

        Ok(())
    }

    #[test]
    fn test_barron_general_loss_special_cases() -> TestResult {
        // α = 0 (Cauchy-like)
        let cauchy = BarronGeneralLoss::new(0.0, 1.0)?;
        let [_, rho_prime_small, _] = cauchy.evaluate(1.0);
        let [_, rho_prime_large, _] = cauchy.evaluate(100.0);
        assert!(rho_prime_large < rho_prime_small);

        // α = 2 (L2)
        let l2 = BarronGeneralLoss::new(2.0, 1.0)?;
        let [rho, rho_prime, rho_double_prime] = l2.evaluate(4.0);
        assert!((rho - 4.0).abs() < EPSILON);
        assert!((rho_prime - 1.0).abs() < EPSILON);
        assert!(rho_double_prime.abs() < EPSILON);

        // α = 1 (Charbonnier-like)
        let charbonnier = BarronGeneralLoss::new(1.0, 1.0)?;
        let [_, rho_prime_char, _] = charbonnier.evaluate(4.0);
        assert!(rho_prime_char > 0.0 && rho_prime_char < 1.0);

        // Test α = -2 (Geman-McClure-like) - strong outlier suppression
        let gm = BarronGeneralLoss::new(-2.0, 1.0)?;
        let [_, rho_prime_small, _] = gm.evaluate(1.0);
        let [_, rho_prime_large, _] = gm.evaluate(100.0);
        assert!(rho_prime_large < rho_prime_small); // Redescending behavior
        assert!(rho_prime_large < 0.1); // Strong suppression

        Ok(())
    }

    #[test]
    fn test_constructor_validation() -> TestResult {
        // Test that negative or zero scale parameters are rejected
        assert!(FairLoss::new(0.0).is_err());
        assert!(FairLoss::new(-1.0).is_err());
        assert!(GemanMcClureLoss::new(0.0).is_err());
        assert!(WelschLoss::new(-1.0).is_err());
        assert!(TukeyBiweightLoss::new(0.0).is_err());
        assert!(AndrewsWaveLoss::new(-1.0).is_err());
        assert!(RamsayEaLoss::new(0.0).is_err());
        assert!(TrimmedMeanLoss::new(-1.0).is_err());
        assert!(BarronGeneralLoss::new(0.0, 0.0).is_err());
        assert!(BarronGeneralLoss::new(1.0, -1.0).is_err());

        // Test that p ≤ 0 is rejected for LpNormLoss
        assert!(LpNormLoss::new(0.0).is_err());
        assert!(LpNormLoss::new(-1.0).is_err());

        // Test valid constructors
        assert!(FairLoss::new(1.0).is_ok());
        assert!(LpNormLoss::new(1.5).is_ok());
        assert!(BarronGeneralLoss::new(1.0, 1.0).is_ok());

        Ok(())
    }

    #[test]
    fn test_loss_comparison() -> TestResult {
        // Compare robustness: L2 vs Huber vs Cauchy at outlier
        let s_outlier = 100.0;

        let l2 = L2Loss;
        let huber = HuberLoss::new(1.345)?;
        let cauchy = CauchyLoss::new(2.3849)?;

        let [_, w_l2, _] = l2.evaluate(s_outlier);
        let [_, w_huber, _] = huber.evaluate(s_outlier);
        let [_, w_cauchy, _] = cauchy.evaluate(s_outlier);

        // L2 should give highest weight (no robustness)
        assert!(w_l2 > w_huber);
        assert!(w_huber > w_cauchy);

        // Cauchy should strongly suppress outliers
        assert!(w_cauchy < 0.1);

        Ok(())
    }

    #[test]
    fn test_t_distribution_loss() -> TestResult {
        let loss = TDistributionLoss::new(5.0)?;

        // Test at s = 0 (should be well-defined)
        let [rho, rho_prime, _] = loss.evaluate(0.0);
        assert_eq!(rho, 0.0);
        assert!((rho_prime - 0.6).abs() < 0.01); // (ν+1)/(2ν) = 6/10 = 0.6

        // Test heavy tail behavior (downweighting)
        let [_, rho_prime_small, _] = loss.evaluate(1.0);
        let [_, rho_prime_large, _] = loss.evaluate(100.0);
        assert!(rho_prime_large < rho_prime_small);

        // Test that large outliers are strongly downweighted
        assert!(rho_prime_large < 0.1);

        // Verify derivatives numerically
        let s = 4.0;
        let [_, rho_prime, rho_double_prime] = loss.evaluate(s);
        let (rho_prime_num, rho_double_prime_num) = numerical_derivative(&loss, s, 1e-5);
        assert!((rho_prime - rho_prime_num).abs() < 1e-4);
        assert!((rho_double_prime - rho_double_prime_num).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn test_t_distribution_loss_different_nu() -> TestResult {
        // Test that smaller ν is more robust
        let t3 = TDistributionLoss::new(3.0)?;
        let t10 = TDistributionLoss::new(10.0)?;

        let s_outlier = 100.0;
        let [_, w_t3, _] = t3.evaluate(s_outlier);
        let [_, w_t10, _] = t10.evaluate(s_outlier);

        // Smaller ν should downweight more aggressively
        assert!(w_t3 < w_t10);

        Ok(())
    }

    #[test]
    fn test_adaptive_barron_loss() -> TestResult {
        // Test default (Cauchy-like with α = 0)
        let adaptive = AdaptiveBarronLoss::new(0.0, 1.0)?;

        // Test at s = 0
        let [rho, _, _] = adaptive.evaluate(0.0);
        assert!(rho.abs() < EPSILON);

        // Test robustness
        let [_, rho_prime_small, _] = adaptive.evaluate(1.0);
        let [_, rho_prime_large, _] = adaptive.evaluate(100.0);
        assert!(rho_prime_large < rho_prime_small);

        // AdaptiveBarron wraps BarronGeneral which is already tested,
        // so we just verify the wrapper works correctly
        let barron = BarronGeneralLoss::new(0.0, 1.0)?;
        let [rho_a, rho_prime_a, rho_double_prime_a] = adaptive.evaluate(4.0);
        let [rho_b, rho_prime_b, rho_double_prime_b] = barron.evaluate(4.0);

        // Should match the underlying BarronGeneral exactly
        assert!((rho_a - rho_b).abs() < EPSILON);
        assert!((rho_prime_a - rho_prime_b).abs() < EPSILON);
        assert!((rho_double_prime_a - rho_double_prime_b).abs() < EPSILON);

        Ok(())
    }

    #[test]
    fn test_adaptive_barron_default() -> TestResult {
        // Test default constructor
        let adaptive = AdaptiveBarronLoss::default();

        // Should behave like Cauchy
        let [_, rho_prime, _] = adaptive.evaluate(4.0);
        assert!(rho_prime > 0.0 && rho_prime < 1.0);

        Ok(())
    }

    #[test]
    fn test_new_loss_constructor_validation() -> TestResult {
        // T-distribution: reject non-positive degrees of freedom
        assert!(TDistributionLoss::new(0.0).is_err());
        assert!(TDistributionLoss::new(-1.0).is_err());
        assert!(TDistributionLoss::new(5.0).is_ok());

        // Adaptive Barron: reject non-positive scale
        assert!(AdaptiveBarronLoss::new(0.0, 0.0).is_err());
        assert!(AdaptiveBarronLoss::new(1.0, -1.0).is_err());
        assert!(AdaptiveBarronLoss::new(0.0, 1.0).is_ok());

        Ok(())
    }
}
