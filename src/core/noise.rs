//! Measurement uncertainty for residual blocks.
//!
//! A [`NoiseModel`] carries the square-root information matrix `S` of an
//! edge's information matrix `Ω` (`Ω = SᵀS`). Residuals and Jacobians are
//! **whitened** (`r̃ = S·r`, `J̃ = S·J`) upstream of the robust-loss
//! corrector, so the optimized objective becomes
//!
//! ```text
//! F(x) = Σ ½·ρ(‖Sᵢ·rᵢ(x)‖²)
//! ```
//!
//! — the Ω-weighted objective that g2o-style benchmarks report — while the
//! Triggs correction, cost accounting and covariance semantics compose
//! unchanged. [`NoiseModel::Null`] is the zero-cost identity default: with it,
//! no allocation happens and no arithmetic is performed.

use nalgebra::{DMatrix, DVector};

use crate::core::CoreError;
use crate::core::CoreResult;

/// Square-root information model for one residual block.
///
/// `S` upper-factors `Ω = SᵀS`; whitening is `r̃ = S·r`, `J̃ = S·J`.
#[derive(Debug, Clone, PartialEq)]
pub enum NoiseModel {
    /// Identity model — unweighted. Zero-cost: no allocation, no arithmetic.
    Null,
    /// Diagonal square-root information (one entry per residual row). Covers
    /// the block-diagonal Ω that most SLAM graphs use.
    Diagonal(DVector<f64>),
    /// Full square-root information matrix (`residual_dim × residual_dim`).
    Dense(DMatrix<f64>),
}

impl NoiseModel {
    /// Identity model.
    pub fn null() -> Self {
        NoiseModel::Null
    }

    /// Diagonal model from per-component standard deviations
    /// (`sqrt_info_i = 1/σ_i`).
    pub fn from_sigmas(sigmas: &[f64]) -> CoreResult<Self> {
        if sigmas.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err(CoreError::InvalidInput(
                "noise model sigmas must be finite and positive".to_string(),
            ));
        }
        Ok(NoiseModel::Diagonal(DVector::from_iterator(
            sigmas.len(),
            sigmas.iter().map(|s| 1.0 / s),
        )))
    }

    /// Diagonal model from square-root information entries (1/σ per row).
    pub fn from_diagonal_sqrt_info(sqrt_info: &[f64]) -> CoreResult<Self> {
        if sqrt_info.iter().any(|s| !s.is_finite() || *s < 0.0) {
            return Err(CoreError::InvalidInput(
                "noise model sqrt-information entries must be finite and non-negative".to_string(),
            ));
        }
        Ok(NoiseModel::Diagonal(DVector::from_column_slice(sqrt_info)))
    }

    /// Dense model from an information matrix `Ω`.
    ///
    /// `Ω` must be symmetric PSD. Real g2o graphs frequently carry
    /// rank-deficient Ω (unobserved DOFs, e.g. zero roll/pitch information),
    /// so negative eigenvalues are clamped to zero — the corresponding
    /// residual directions carry no information and whiten to zero. The
    /// square root is the symmetric `S = V·√Λ⁺·Vᵀ`, satisfying `SᵀS = Ω`.
    pub fn from_information(info: DMatrix<f64>) -> CoreResult<Self> {
        let n = info.nrows();
        if !info.is_square() || n == 0 {
            return Err(CoreError::InvalidInput(
                "information matrix must be non-empty and square".to_string(),
            ));
        }
        if !info.iter().all(|v| v.is_finite()) {
            return Err(CoreError::InvalidInput(
                "information matrix must be finite".to_string(),
            ));
        }
        let sym = (&info + &info.transpose()) * 0.5;
        let se = nalgebra::linalg::SymmetricEigen::new(sym);
        // Negative eigenvalues are clamped to zero (matching g2o/GTSAM, whose
        // pivoting LDLᵀ tolerates indefinite Ω): fp-noise negatives arise on
        // rank-deficient directions, and some real datasets carry slightly
        // indefinite Ω. A genuinely large negative is surfaced as a warning —
        // it means ill-formed measurement information, not a crash.
        let lam_max = se.eigenvalues.iter().copied().fold(f64::MIN, f64::max);
        let tol = lam_max.abs() * 1e-9 + 1e-12;
        if se.eigenvalues.iter().any(|&lam| lam < -tol) {
            tracing::warn!(
                "information matrix has a negative eigenvalue (min = {:.3e}); \
                 clamping to zero — the edge may be ill-formed",
                se.eigenvalues.iter().copied().fold(f64::MAX, f64::min)
            );
        }
        let mut s = DMatrix::zeros(n, n);
        for i in 0..n {
            let lam = se.eigenvalues[i].max(0.0);
            if lam == 0.0 {
                continue;
            }
            let root = lam.sqrt();
            for r in 0..n {
                for c in 0..n {
                    s[(r, c)] += root * se.eigenvectors[(r, i)] * se.eigenvectors[(c, i)];
                }
            }
        }
        Ok(NoiseModel::Dense(s))
    }

    /// Dense model from an explicit square-root information matrix `S`
    /// (callers guarantee `SᵀS` is the intended information matrix).
    pub fn from_sqrt_info(s: DMatrix<f64>) -> CoreResult<Self> {
        if !s.is_square() || s.nrows() == 0 || !s.iter().all(|v| v.is_finite()) {
            return Err(CoreError::InvalidInput(
                "sqrt-information matrix must be non-empty, square and finite".to_string(),
            ));
        }
        Ok(NoiseModel::Dense(s))
    }

    /// Residual dimension the model applies to.
    pub fn dim(&self) -> usize {
        match self {
            NoiseModel::Null => 0,
            NoiseModel::Diagonal(d) => d.len(),
            NoiseModel::Dense(m) => m.nrows(),
        }
    }

    /// Whiten a residual in place: `r ← S·r`.
    pub fn whiten_residual(&self, residual: &mut [f64]) {
        match self {
            NoiseModel::Null => {}
            NoiseModel::Diagonal(d) => {
                for (r, s) in residual.iter_mut().zip(d.iter()) {
                    *r *= s;
                }
            }
            NoiseModel::Dense(m) => {
                let n = m.nrows();
                debug_assert_eq!(residual.len(), n);
                let mut out = vec![0.0; n];
                for (row, o) in out.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for (c, value) in residual.iter().enumerate() {
                        sum += m[(row, c)] * value;
                    }
                    *o = sum;
                }
                residual.copy_from_slice(&out);
            }
        }
    }

    /// Whiten a column-major `(rows × cols)` Jacobian in place: `J ← S·J`.
    pub fn whiten_jacobian(&self, jacobian: &mut [f64], rows: usize, cols: usize) {
        match self {
            NoiseModel::Null => {}
            NoiseModel::Diagonal(d) => {
                debug_assert_eq!(d.len(), rows);
                // Row scaling: row i of every column is multiplied by s_i.
                for c in 0..cols {
                    for r in 0..rows {
                        jacobian[c * rows + r] *= d[r];
                    }
                }
            }
            NoiseModel::Dense(m) => {
                debug_assert_eq!(m.nrows(), rows);
                for c in 0..cols {
                    let col = jacobian[c * rows..(c + 1) * rows].to_vec();
                    for (row, o) in jacobian[c * rows..(c + 1) * rows].iter_mut().enumerate() {
                        let mut sum = 0.0;
                        for (j, value) in col.iter().enumerate() {
                            sum += m[(row, j)] * value;
                        }
                        *o = sum;
                    }
                }
            }
        }
    }
}
