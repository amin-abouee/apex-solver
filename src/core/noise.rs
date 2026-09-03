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

/// What [`NoiseModel::from_information_reporting`] had to repair to make an
/// information matrix PSD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InformationRepair {
    /// Eigen-directions whose negative eigenvalue was clamped to zero. Each
    /// one is a residual direction the measurement no longer constrains.
    pub clamped_directions: usize,
    /// Smallest eigenvalue of the symmetrized `Ω`, before clamping.
    pub min_eigenvalue: f64,
    /// Largest eigenvalue of the symmetrized `Ω`, the scale `min_eigenvalue`
    /// is judged against.
    pub max_eigenvalue: f64,
}

/// Relative size below which a negative eigenvalue is attributed to
/// floating-point noise in the eigendecomposition rather than to bad data.
const INDEFINITE_TOLERANCE: f64 = 1e-9;

impl InformationRepair {
    /// True when clamping discarded real information rather than
    /// floating-point noise.
    ///
    /// Rank-deficient `Ω` (an exactly-zero eigenvalue, an unobserved DOF) is
    /// legitimate and is *not* material — only a negative eigenvalue that is
    /// large relative to the matrix scale is.
    pub fn is_material(&self) -> bool {
        let tol = self.max_eigenvalue.abs() * INDEFINITE_TOLERANCE + 1e-12;
        self.min_eigenvalue < -tol
    }

    /// Most negative eigenvalue as a fraction of the largest, a scale-free
    /// measure of how indefinite `Ω` was. Zero when `Ω` was PSD.
    pub fn relative_indefiniteness(&self) -> f64 {
        if self.max_eigenvalue <= 0.0 || self.min_eigenvalue >= 0.0 {
            return 0.0;
        }
        -self.min_eigenvalue / self.max_eigenvalue
    }
}

/// Strategy for handling an indefinite per-edge information matrix.
///
/// The repair itself lives in [`NoiseModel::from_information_reporting`]; this
/// enum is the single shared vocabulary for choosing what to do with the
/// report, so binaries, benches and libraries stop re-implementing the
/// `clamp | unit-weight` branch with divergent defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RepairStrategy {
    /// Keep the PSD projection (nearest-PSD clamp). The optimized objective
    /// stays the Ω-weighted χ², minus the clamped directions.
    #[default]
    Clamp,
    /// Substitute an identity weight for any edge whose `Ω` needed a
    /// *material* repair. This trades χ² for unweighted cost on those edges —
    /// callers must label the reported objective accordingly.
    UnitWeight,
}

impl RepairStrategy {
    /// Parse a CLI/env value (`"clamp"` or `"unit-weight"`, case-insensitive).
    pub fn from_name(name: &str) -> CoreResult<Self> {
        if name.eq_ignore_ascii_case("clamp") {
            Ok(RepairStrategy::Clamp)
        } else if name.eq_ignore_ascii_case("unit-weight") {
            Ok(RepairStrategy::UnitWeight)
        } else {
            Err(CoreError::InvalidInput(format!(
                "unknown repair strategy {name:?}: expected \"clamp\" or \"unit-weight\""
            )))
        }
    }

    /// Build a noise model from `Ω` under this strategy, reporting what
    /// happened. Returns the model and its [`InformationRepair`]; under
    /// [`RepairStrategy::UnitWeight`] a materially-repaired edge comes back
    /// as [`NoiseModel::Null`] (identity weight).
    pub fn build(&self, info: DMatrix<f64>) -> CoreResult<(NoiseModel, InformationRepair)> {
        let (model, repair) = NoiseModel::from_information_reporting(info)?;
        if *self == RepairStrategy::UnitWeight && repair.is_material() {
            Ok((NoiseModel::null(), repair))
        } else {
            Ok((model, repair))
        }
    }
}

/// Running tally of information-matrix repairs over a loaded graph.
///
/// The library reports per edge ([`InformationRepair`]); aggregation belongs
/// to the caller, and this is the shared counter so every harness reports the
/// same line. Log one summary per dataset, not one `warn!` per edge.
#[derive(Debug, Clone, Copy, Default)]
pub struct RepairSummary {
    /// Edges examined.
    pub edges: usize,
    /// Edges whose `Ω` needed a material repair (real information clamped).
    pub materially_repaired: usize,
    /// Of those, edges substituted with identity weight (`UnitWeight` only).
    pub unit_weighted: usize,
}

impl RepairSummary {
    /// Record one edge's outcome under `strategy`.
    pub fn record(&mut self, strategy: RepairStrategy, repair: &InformationRepair) {
        self.edges += 1;
        if repair.is_material() {
            self.materially_repaired += 1;
            if strategy == RepairStrategy::UnitWeight {
                self.unit_weighted += 1;
            }
        }
    }

    /// True when every edge was PSD as parsed.
    pub fn is_clean(&self) -> bool {
        self.materially_repaired == 0
    }
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
    ///
    /// Clamping is silent here. Callers that must distinguish floating-point
    /// noise from a genuinely ill-formed Ω — where clamping discards real
    /// constraints rather than empty directions — should use
    /// [`Self::from_information_reporting`] and inspect the
    /// [`InformationRepair`] it returns.
    pub fn from_information(info: DMatrix<f64>) -> CoreResult<Self> {
        let (model, repair) = Self::from_information_reporting(info)?;
        if repair.is_material() {
            tracing::warn!(
                "information matrix has a negative eigenvalue (min = {:.3e}, \
                 max = {:.3e}); clamping to zero — the edge may be ill-formed",
                repair.min_eigenvalue,
                repair.max_eigenvalue
            );
        }
        Ok(model)
    }

    /// Same repair as [`Self::from_information`], but reports what it did.
    ///
    /// Clamping an indefinite `Ω` to PSD is the nearest-PSD projection, and on
    /// a genuinely ill-formed matrix it silently deletes constraints: the
    /// clamped directions stop influencing the solution at all. Returning the
    /// [`InformationRepair`] lets a caller decide — keep the repaired `Ω`,
    /// substitute a unit weight, or reject the measurement — instead of
    /// optimizing a quietly mutilated objective.
    pub fn from_information_reporting(
        info: DMatrix<f64>,
    ) -> CoreResult<(Self, InformationRepair)> {
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
        // indefinite Ω.
        let lam_max = se.eigenvalues.iter().copied().fold(f64::MIN, f64::max);
        let lam_min = se.eigenvalues.iter().copied().fold(f64::MAX, f64::min);
        let mut s = DMatrix::zeros(n, n);
        let mut clamped_directions = 0;
        for i in 0..n {
            let lam = se.eigenvalues[i];
            if lam <= 0.0 {
                if lam < 0.0 {
                    clamped_directions += 1;
                }
                continue;
            }
            let root = lam.sqrt();
            for r in 0..n {
                for c in 0..n {
                    s[(r, c)] += root * se.eigenvectors[(r, i)] * se.eigenvectors[(c, i)];
                }
            }
        }
        let repair = InformationRepair {
            clamped_directions,
            min_eigenvalue: lam_min,
            max_eigenvalue: lam_max,
        };
        Ok((NoiseModel::Dense(s), repair))
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

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn repair_strategy_parses_known_names() -> TestResult {
        assert_eq!(RepairStrategy::from_name("clamp")?, RepairStrategy::Clamp);
        assert_eq!(
            RepairStrategy::from_name("UNIT-WEIGHT")?,
            RepairStrategy::UnitWeight
        );
        assert_eq!(RepairStrategy::default(), RepairStrategy::Clamp);
        assert!(RepairStrategy::from_name("project").is_err());
        Ok(())
    }

    #[test]
    fn clamp_keeps_repaired_model_and_unit_weight_substitutes() -> TestResult {
        // diag(1, -4): one materially indefinite direction.
        let info = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -4.0]);
        let (clamped, repair) = RepairStrategy::Clamp.build(info.clone())?;
        assert!(repair.is_material());
        assert!(matches!(clamped, NoiseModel::Dense(_)));

        let (unit, repair) = RepairStrategy::UnitWeight.build(info)?;
        assert!(repair.is_material());
        assert_eq!(unit, NoiseModel::Null);
        Ok(())
    }

    #[test]
    fn clean_matrix_never_counts_as_repaired() -> TestResult {
        let info = DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 1.0]);
        let mut summary = RepairSummary::default();
        for strategy in [RepairStrategy::Clamp, RepairStrategy::UnitWeight] {
            let (model, repair) = strategy.build(info.clone())?;
            summary.record(strategy, &repair);
            assert!(!matches!(model, NoiseModel::Null));
        }
        assert!(summary.is_clean());
        assert_eq!(summary.edges, 2);
        assert_eq!(summary.materially_repaired, 0);
        assert_eq!(summary.unit_weighted, 0);
        Ok(())
    }

    #[test]
    fn summary_counts_material_and_unit_weighted_edges() -> TestResult {
        let bad = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -4.0]);
        let mut summary = RepairSummary::default();
        let (_, repair) = RepairStrategy::Clamp.build(bad.clone())?;
        summary.record(RepairStrategy::Clamp, &repair);
        let (_, repair) = RepairStrategy::UnitWeight.build(bad)?;
        summary.record(RepairStrategy::UnitWeight, &repair);
        assert_eq!(summary.edges, 2);
        assert_eq!(summary.materially_repaired, 2);
        assert_eq!(summary.unit_weighted, 1);
        assert!(!summary.is_clean());
        Ok(())
    }
}
