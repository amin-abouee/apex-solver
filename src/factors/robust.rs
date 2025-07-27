//! Robust kernel implementations for outlier rejection
//!
//! This module provides robust kernel functions that can be applied to factor
//! residuals to reduce the influence of outliers in optimization. These kernels
//! are essential for real-world applications where measurements may contain
//! significant outliers.
//!
//! # Robust Kernels
//!
//! - `HuberKernel`: Huber robust kernel (quadratic for small errors, linear for large)
//! - `CauchyKernel`: Cauchy robust kernel (heavy-tailed distribution)
//! - `TukeyKernel`: Tukey biweight kernel (completely rejects large outliers)
//! - `GemanMcClureKernel`: Geman-McClure kernel (redescending M-estimator)
//! - `L2Kernel`: Standard L2 norm (no robustification)
//!
//! # Mathematical Background
//!
//! Robust kernels modify the cost function from:
//! ```text
//! cost = ||r||²
//! ```
//! to:
//! ```text
//! cost = ρ(||r||²)
//! ```
//! where `ρ` is the robust kernel function.
//!
//! The influence function `ψ(x) = ρ'(x)` determines how much each residual
//! contributes to the gradient, and the weight function `w(x) = ψ(x)/x`
//! determines the effective weight of each residual.

use std::fmt;

/// Trait for robust kernel functions
///
/// Robust kernels provide a way to reduce the influence of outliers in
/// optimization by modifying the cost function. Each kernel implements
/// the kernel function ρ(x), its derivative ψ(x), and weight function w(x).
pub trait RobustKernel: fmt::Debug + Clone + Send + Sync {
    /// Evaluate the robust kernel function ρ(x)
    ///
    /// # Arguments
    /// * `squared_error` - The squared error ||r||²
    ///
    /// # Returns
    /// The robustified cost ρ(||r||²)
    fn rho(&self, squared_error: f64) -> f64;

    /// Evaluate the influence function ψ(x) = ρ'(x)
    ///
    /// # Arguments
    /// * `squared_error` - The squared error ||r||²
    ///
    /// # Returns
    /// The derivative of the kernel function
    fn psi(&self, squared_error: f64) -> f64;

    /// Evaluate the weight function w(x) = ψ(x)/x
    ///
    /// # Arguments
    /// * `squared_error` - The squared error ||r||²
    ///
    /// # Returns
    /// The weight to apply to the residual
    fn weight(&self, squared_error: f64) -> f64 {
        if squared_error.abs() < f64::EPSILON {
            1.0
        } else {
            self.psi(squared_error) / squared_error
        }
    }

    /// Get the kernel parameter (threshold, scale, etc.)
    fn parameter(&self) -> f64;

    /// Set the kernel parameter
    fn set_parameter(&mut self, parameter: f64);
}

/// L2 kernel (no robustification)
///
/// This kernel provides no robustification and is equivalent to standard
/// least squares optimization. It's included for completeness and as a baseline.
///
/// Mathematical formulation:
/// - ρ(x) = x
/// - ψ(x) = 1
/// - w(x) = 1/x
#[derive(Debug, Clone)]
pub struct L2Kernel;

impl L2Kernel {
    /// Create a new L2 kernel
    pub fn new() -> Self {
        Self
    }
}

impl Default for L2Kernel {
    fn default() -> Self {
        Self::new()
    }
}

impl RobustKernel for L2Kernel {
    fn rho(&self, squared_error: f64) -> f64 {
        squared_error
    }

    fn psi(&self, _squared_error: f64) -> f64 {
        1.0
    }

    fn weight(&self, squared_error: f64) -> f64 {
        if squared_error.abs() < f64::EPSILON {
            1.0
        } else {
            1.0 / squared_error
        }
    }

    fn parameter(&self) -> f64 {
        1.0
    }

    fn set_parameter(&mut self, _parameter: f64) {
        // L2 kernel has no parameters
    }
}

/// Huber robust kernel
///
/// The Huber kernel is quadratic for small errors and linear for large errors,
/// providing a good balance between efficiency and robustness.
///
/// Mathematical formulation:
/// - ρ(x) = x if x ≤ δ², else 2δ√x - δ²
/// - ψ(x) = 1 if x ≤ δ², else δ/√x
/// - w(x) = 1/x if x ≤ δ², else δ/(x√x)
#[derive(Debug, Clone)]
pub struct HuberKernel {
    /// Threshold parameter δ
    delta: f64,
    /// Squared threshold for efficiency
    delta_squared: f64,
}

impl HuberKernel {
    /// Create a new Huber kernel with specified threshold
    ///
    /// # Arguments
    /// * `delta` - Threshold parameter (typically 1.345 for 95% efficiency)
    pub fn new(delta: f64) -> Self {
        Self {
            delta,
            delta_squared: delta * delta,
        }
    }

    /// Create a new Huber kernel with default threshold
    pub fn default_threshold() -> Self {
        Self::new(1.345)
    }
}

impl Default for HuberKernel {
    fn default() -> Self {
        Self::default_threshold()
    }
}

impl RobustKernel for HuberKernel {
    fn rho(&self, squared_error: f64) -> f64 {
        if squared_error <= self.delta_squared {
            squared_error
        } else {
            2.0 * self.delta * squared_error.sqrt() - self.delta_squared
        }
    }

    fn psi(&self, squared_error: f64) -> f64 {
        if squared_error <= self.delta_squared {
            1.0
        } else {
            self.delta / squared_error.sqrt()
        }
    }

    fn weight(&self, squared_error: f64) -> f64 {
        if squared_error.abs() < f64::EPSILON {
            1.0
        } else if squared_error <= self.delta_squared {
            1.0 / squared_error
        } else {
            self.delta / (squared_error * squared_error.sqrt())
        }
    }

    fn parameter(&self) -> f64 {
        self.delta
    }

    fn set_parameter(&mut self, delta: f64) {
        self.delta = delta;
        self.delta_squared = delta * delta;
    }
}

/// Cauchy robust kernel
///
/// The Cauchy kernel is based on the Cauchy distribution and provides
/// strong robustness against outliers with a heavy-tailed influence function.
///
/// Mathematical formulation:
/// - ρ(x) = (σ²/2) * ln(1 + x/σ²)
/// - ψ(x) = 1/(2(1 + x/σ²))
/// - w(x) = 1/(2x(1 + x/σ²))
#[derive(Debug, Clone)]
pub struct CauchyKernel {
    /// Scale parameter σ
    sigma: f64,
    /// Squared scale parameter for efficiency
    sigma_squared: f64,
}

impl CauchyKernel {
    /// Create a new Cauchy kernel with specified scale
    ///
    /// # Arguments
    /// * `sigma` - Scale parameter (typically around 1.0)
    pub fn new(sigma: f64) -> Self {
        Self {
            sigma,
            sigma_squared: sigma * sigma,
        }
    }

    /// Create a new Cauchy kernel with default scale
    pub fn default_scale() -> Self {
        Self::new(1.0)
    }
}

impl Default for CauchyKernel {
    fn default() -> Self {
        Self::default_scale()
    }
}

impl RobustKernel for CauchyKernel {
    fn rho(&self, squared_error: f64) -> f64 {
        (self.sigma_squared / 2.0) * (1.0 + squared_error / self.sigma_squared).ln()
    }

    fn psi(&self, squared_error: f64) -> f64 {
        1.0 / (2.0 * (1.0 + squared_error / self.sigma_squared))
    }

    fn weight(&self, squared_error: f64) -> f64 {
        if squared_error.abs() < f64::EPSILON {
            1.0 / (2.0 * self.sigma_squared)
        } else {
            1.0 / (2.0 * squared_error * (1.0 + squared_error / self.sigma_squared))
        }
    }

    fn parameter(&self) -> f64 {
        self.sigma
    }

    fn set_parameter(&mut self, sigma: f64) {
        self.sigma = sigma;
        self.sigma_squared = sigma * sigma;
    }
}

/// Tukey biweight robust kernel
///
/// The Tukey kernel completely rejects outliers beyond a threshold,
/// providing very strong robustness but potentially losing information.
///
/// Mathematical formulation:
/// - ρ(x) = (c²/6)[1 - (1 - x/c²)³] if x ≤ c², else c²/6
/// - ψ(x) = (1 - x/c²)² if x ≤ c², else 0
/// - w(x) = (1 - x/c²)²/x if x ≤ c², else 0
#[derive(Debug, Clone)]
pub struct TukeyKernel {
    /// Threshold parameter c
    c: f64,
    /// Squared threshold for efficiency
    c_squared: f64,
}

impl TukeyKernel {
    /// Create a new Tukey kernel with specified threshold
    ///
    /// # Arguments
    /// * `c` - Threshold parameter (typically 4.685 for 95% efficiency)
    pub fn new(c: f64) -> Self {
        Self {
            c,
            c_squared: c * c,
        }
    }

    /// Create a new Tukey kernel with default threshold
    pub fn default_threshold() -> Self {
        Self::new(4.685)
    }
}

impl Default for TukeyKernel {
    fn default() -> Self {
        Self::default_threshold()
    }
}

impl RobustKernel for TukeyKernel {
    fn rho(&self, squared_error: f64) -> f64 {
        if squared_error <= self.c_squared {
            let ratio = squared_error / self.c_squared;
            let term = 1.0 - ratio;
            (self.c_squared / 6.0) * (1.0 - term * term * term)
        } else {
            self.c_squared / 6.0
        }
    }

    fn psi(&self, squared_error: f64) -> f64 {
        if squared_error <= self.c_squared {
            let ratio = squared_error / self.c_squared;
            let term = 1.0 - ratio;
            term * term
        } else {
            0.0
        }
    }

    fn weight(&self, squared_error: f64) -> f64 {
        if squared_error.abs() < f64::EPSILON {
            1.0 / self.c_squared
        } else if squared_error <= self.c_squared {
            let ratio = squared_error / self.c_squared;
            let term = 1.0 - ratio;
            term * term / squared_error
        } else {
            0.0
        }
    }

    fn parameter(&self) -> f64 {
        self.c
    }

    fn set_parameter(&mut self, c: f64) {
        self.c = c;
        self.c_squared = c * c;
    }
}

/// Geman-McClure robust kernel
///
/// The Geman-McClure kernel is a redescending M-estimator that provides
/// strong robustness while maintaining some influence for large errors.
///
/// Mathematical formulation:
/// - ρ(x) = x/(1 + x/σ²)
/// - ψ(x) = 1/(1 + x/σ²)²
/// - w(x) = 1/(x(1 + x/σ²)²)
#[derive(Debug, Clone)]
pub struct GemanMcClureKernel {
    /// Scale parameter σ
    sigma: f64,
    /// Squared scale parameter for efficiency
    sigma_squared: f64,
}

impl GemanMcClureKernel {
    /// Create a new Geman-McClure kernel with specified scale
    ///
    /// # Arguments
    /// * `sigma` - Scale parameter (typically around 1.0)
    pub fn new(sigma: f64) -> Self {
        Self {
            sigma,
            sigma_squared: sigma * sigma,
        }
    }

    /// Create a new Geman-McClure kernel with default scale
    pub fn default_scale() -> Self {
        Self::new(1.0)
    }
}

impl Default for GemanMcClureKernel {
    fn default() -> Self {
        Self::default_scale()
    }
}

impl RobustKernel for GemanMcClureKernel {
    fn rho(&self, squared_error: f64) -> f64 {
        squared_error / (1.0 + squared_error / self.sigma_squared)
    }

    fn psi(&self, squared_error: f64) -> f64 {
        let denominator = 1.0 + squared_error / self.sigma_squared;
        1.0 / (denominator * denominator)
    }

    fn weight(&self, squared_error: f64) -> f64 {
        if squared_error.abs() < f64::EPSILON {
            1.0 / self.sigma_squared
        } else {
            let denominator = 1.0 + squared_error / self.sigma_squared;
            1.0 / (squared_error * denominator * denominator)
        }
    }

    fn parameter(&self) -> f64 {
        self.sigma
    }

    fn set_parameter(&mut self, sigma: f64) {
        self.sigma = sigma;
        self.sigma_squared = sigma * sigma;
    }
}
