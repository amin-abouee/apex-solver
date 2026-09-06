//! Turn a set of variables into a Gaussian prior over the ones they touch.
//!
//! A sliding window cannot simply delete an old state: the measurements that
//! constrained it also constrained its neighbours, and dropping them throws that
//! information away. Marginalization eliminates the state and leaves behind the
//! Gaussian it induced on the states it was connected to, carried by a
//! [`MarginalPriorFactor`].
//!
//! ```text
//! [Λ_kk  Λ_km] [δx_k]   [g_k]        Λ_p = Λ_kk − Λ_km·Λ_mm⁻¹·Λ_mk
//! [Λ_mk  Λ_mm] [δx_m] = [g_m]   ⇒    g_p = g_k − Λ_km·Λ_mm⁻¹·g_m
//! ```
//!
//! # Why this is dense, and why it does not use the Schur module
//!
//! [`crate::linalg::schur`] is structurally a *landmark* eliminator: its
//! `ChunkLayout` rejects two eliminated variables that share a residual row, and
//! `SchurPartition::verify_block_diagonal` demands a block-diagonal `Λ_mm`. A
//! visual-inertial window eliminates `{state_0, bias_0, hosted landmarks}`, and
//! `CombinedImuFactor` puts `state_0` and `bias_0` in the *same fifteen rows*.
//! So `Λ_mm` is not block-diagonal and none of that machinery applies.
//!
//! The Markov blanket of one keyframe is `O(10²–10³)` DOF, where a dense
//! factorization is both correct and fast, so this module builds `Λ` explicitly.
//! It works in `nalgebra` rather than `faer` because it runs once per window
//! step, not in the inner loop, and because [`MarginalPriorFactor`] takes its
//! square-root information as a `DMatrix` — a `faer` round trip would buy
//! nothing and cost a conversion at each end.
//!
//! # The sign
//!
//! `NormalEquations::gradient` is `Jᵀr`, **positive**, and the system solved is
//! `H·δ = −g`. The quadratic model is `C(δ) = c + gᵀδ + ½δᵀΛδ`, minimized at
//! `δ* = −Λ⁻¹g`, and the prior's own cost `½(θ − b)ᵀΛ_p(θ − b)` is minimized at
//! `θ = b`. Therefore `b = −Λ_p⁺·g_p`. With the sign flipped the prior pushes
//! the state the wrong way by exactly `2Λ⁻¹g` and still looks plausible, which
//! is why `offset_sign_reproduces_the_conditional_minimizer` exists.

use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use apex_manifolds::{LieGroup, ManifoldType, Tangent, rn, se2, se3, se23, sgal3, sim3, so2, so3};

use crate::core::variable::ManifoldVariable;
use crate::core::{CoreError, FactorKey, VarKey};
use crate::factors::marginal::{LocalLogFn, MarginalPriorFactor};
use crate::linearizer::LinearizerError;
use crate::linearizer::compute_block_into;

use super::problem::Problem;

/// How `Λ_p` is factored into `S` with `SᵀS = Λ_p`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SqrtInformation {
    /// Self-adjoint eigendecomposition with eigenvalue clamping.
    ///
    /// Handles the gauge null space, which a marginal is *expected* to have —
    /// a monocular visual-inertial window cannot observe global position or
    /// heading. Eigenvalues below `rank_tolerance · λ_max` (and any negative
    /// ones, which are numerical noise on a PSD matrix) become zero, producing
    /// zero **rows** in a square `S`.
    Eigen {
        /// Relative threshold below which an eigenvalue is treated as zero.
        rank_tolerance: f64,
    },
    /// Cholesky, `S = Lᵀ`. Faster, and errors on a rank-deficient `Λ_p`.
    Cholesky,
}

impl Default for SqrtInformation {
    fn default() -> Self {
        Self::Eigen {
            rank_tolerance: 1e-9,
        }
    }
}

/// Eliminates variables from a [`Problem`], leaving a Gaussian prior.
#[derive(Debug, Clone)]
pub struct Marginalizer {
    sqrt_information: SqrtInformation,
    symmetrize: bool,
}

impl Default for Marginalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Marginalizer {
    /// A marginalizer with eigen square roots and symmetrization on.
    pub fn new() -> Self {
        Self {
            sqrt_information: SqrtInformation::default(),
            symmetrize: true,
        }
    }

    /// Choose how `Λ_p` is factored.
    pub fn with_sqrt_information(mut self, sqrt_information: SqrtInformation) -> Self {
        self.sqrt_information = sqrt_information;
        self
    }

    /// Symmetrize `Λ_p ← ½(Λ_p + Λ_pᵀ)` before factoring. On by default: the
    /// Schur complement of a symmetric matrix is symmetric only in exact
    /// arithmetic.
    pub fn with_symmetrize(mut self, symmetrize: bool) -> Self {
        self.symmetrize = symmetrize;
        self
    }

    /// Compute the marginal without mutating `problem`.
    ///
    /// # Errors
    ///
    /// See [`MarginalizationError`].
    pub fn compute(
        &self,
        problem: &Problem,
        marginalized: &[VarKey],
    ) -> MarginalizationResult<Marginal> {
        if marginalized.is_empty() {
            return Err(MarginalizationError::EmptyRequest);
        }
        for key in marginalized {
            if problem.variable(*key).is_none() {
                return Err(MarginalizationError::UnknownVariable(*key));
            }
        }
        let dropped: HashSet<VarKey> = marginalized.iter().copied().collect();

        // ── The absorbed set is the Markov blanket's factors, not the graph ──
        //
        // Marginalizing over every block would bake information from factors
        // that stay in the graph into the prior, counting them twice, and would
        // make `Λ_p` dense over the whole state. Only blocks incident to a
        // dropped variable are absorbed; the previous prior is itself incident,
        // so it is re-absorbed, which is the correct recursive behaviour.
        let absorbed: Vec<FactorKey> = problem
            .residual_blocks()
            .iter()
            .filter(|(_, block)| block.variable_keys.iter().any(|k| dropped.contains(k)))
            .map(|(key, _)| key)
            .collect();

        for key in marginalized {
            let touched = problem
                .residual_blocks()
                .iter()
                .any(|(_, block)| block.variable_keys.contains(key));
            if !touched {
                return Err(MarginalizationError::Unobserved(*key));
            }
        }

        // ── Local column layout: kept first, then eliminated ────────────────
        //
        // Ordering both halves by the problem's own column rank makes the result
        // deterministic, and putting kept first makes `Λ` literally 2x2 block so
        // no permutation is needed anywhere below.
        let rank: HashMap<VarKey, usize> = problem
            .variable_keys()
            .enumerate()
            .map(|(rank, key)| (key, rank))
            .collect();

        let mut blanket: Vec<VarKey> = Vec::new();
        for key in &absorbed {
            let Some(block) = problem.residual_blocks().get(*key) else {
                continue;
            };
            for var in &block.variable_keys {
                if !blanket.contains(var) {
                    blanket.push(*var);
                }
            }
        }
        let by_rank = |a: &VarKey, b: &VarKey| {
            rank.get(a)
                .copied()
                .unwrap_or(usize::MAX)
                .cmp(&rank.get(b).copied().unwrap_or(usize::MAX))
        };
        let mut kept: Vec<VarKey> = blanket
            .iter()
            .copied()
            .filter(|k| !dropped.contains(k))
            .collect();
        let mut eliminated: Vec<VarKey> = blanket
            .iter()
            .copied()
            .filter(|k| dropped.contains(k))
            .collect();
        kept.sort_by(by_rank);
        eliminated.sort_by(by_rank);

        if kept.is_empty() {
            let first = marginalized.first().copied().unwrap_or_default();
            return Err(MarginalizationError::NoKeptVariables(first));
        }

        let dof = |key: VarKey| problem.variable(key).map_or(0, ManifoldVariable::dof);
        let mut offsets: HashMap<VarKey, usize> = HashMap::new();
        let mut column = 0usize;
        let mut dims = Vec::with_capacity(kept.len());
        for key in &kept {
            offsets.insert(*key, column);
            let d = dof(*key);
            dims.push(d);
            column += d;
        }
        let kept_dof = column;
        for key in &eliminated {
            offsets.insert(*key, column);
            column += dof(*key);
        }
        let total_dof = column;

        // ── Dense sub-Jacobian over the absorbed blocks ─────────────────────
        //
        // Built through `compute_block_into`, which reuses the whole evaluation
        // path: noise whitening and the robust-loss corrector included. A
        // Huber-weighted block therefore contributes its *robust-weighted*
        // information, matching Ceres and VINS-Mono — the marginal depends on
        // the outlier state at the moment it is taken, and is frozen thereafter.
        let total_rows: usize = absorbed
            .iter()
            .filter_map(|key| problem.residual_blocks().get(*key))
            .map(|block| block.factor.residual_dim())
            .sum();

        let mut jacobian = DMatrix::<f64>::zeros(total_rows, total_dof);
        let mut residual = DVector::<f64>::zeros(total_rows);
        let mut row = 0usize;
        for key in &absorbed {
            let Some(block) = problem.residual_blocks().get(*key) else {
                continue;
            };
            let (rows, cols) = block.factor.jacobian_shape();
            let mut block_jacobian = vec![0.0f64; rows * cols];
            let mut block_residual = vec![0.0f64; block.factor.residual_dim()];
            let (layout, _) = compute_block_into(
                block,
                problem.variables(),
                &mut block_residual,
                Some(&mut block_jacobian),
            )?;

            for (index, value) in block_residual.iter().enumerate() {
                residual[row + index] = *value;
            }
            for (variable, (local_column, size)) in block
                .variable_keys
                .iter()
                .zip(layout.variable_local_idx_size_list.iter())
            {
                let Some(&global_column) = offsets.get(variable) else {
                    continue;
                };
                for c in 0..*size {
                    for r in 0..rows {
                        // `compute_block_into` writes column-major.
                        jacobian[(row + r, global_column + c)] =
                            block_jacobian[(local_column + c) * rows + r];
                    }
                }
            }
            row += rows;
        }

        let information = jacobian.transpose() * &jacobian;
        let gradient = jacobian.transpose() * &residual;

        // ── Schur complement ────────────────────────────────────────────────
        //
        // Solved, not inverted: `O(n_m²·n_k)` instead of `O(n_m³ + n_m²·n_k)`,
        // and better conditioned.
        let eliminated_dof = total_dof - kept_dof;
        let lambda_kk = information.view((0, 0), (kept_dof, kept_dof)).into_owned();
        let lambda_km = information
            .view((0, kept_dof), (kept_dof, eliminated_dof))
            .into_owned();
        let lambda_mm = information
            .view((kept_dof, kept_dof), (eliminated_dof, eliminated_dof))
            .into_owned();
        let g_k = gradient.rows(0, kept_dof).into_owned();
        let g_m = gradient.rows(kept_dof, eliminated_dof).into_owned();

        let lambda_mm_inv = pseudo_inverse(&lambda_mm, 1e-12).ok_or_else(|| {
            MarginalizationError::SingularEliminatedBlock {
                context: format!("{eliminated_dof}-DOF eliminated block has no usable inverse"),
            }
        })?;
        let mut prior_information = lambda_kk - &lambda_km * &lambda_mm_inv * lambda_km.transpose();
        let prior_gradient = g_k - &lambda_km * &lambda_mm_inv * g_m;

        if self.symmetrize {
            prior_information = 0.5 * (&prior_information + prior_information.transpose());
        }

        let (sqrt_information, pseudo, numeric_rank) =
            self.factor_information(&prior_information)?;
        let offset = -(pseudo * &prior_gradient);

        let linearization_point = kept
            .iter()
            .map(|key| {
                problem
                    .variable_params(*key)
                    .map(DVector::from_column_slice)
                    .unwrap_or_else(|| DVector::zeros(0))
            })
            .collect();
        let manifolds = kept
            .iter()
            .map(|key| {
                problem
                    .manifold_type(*key)
                    .ok_or(MarginalizationError::UnknownVariable(*key))
            })
            .collect::<MarginalizationResult<Vec<_>>>()?;

        Ok(Marginal {
            kept,
            dims,
            information: prior_information,
            gradient: prior_gradient,
            sqrt_information,
            offset,
            linearization_point,
            manifolds,
            rank: numeric_rank,
            absorbed,
        })
    }

    /// Compute the marginal, remove what it absorbed, and register the prior.
    ///
    /// Everything is computed before anything is mutated, so a failure leaves
    /// `problem` untouched.
    ///
    /// # Errors
    ///
    /// See [`MarginalizationError`].
    pub fn apply(
        &self,
        problem: &mut Problem,
        marginalized: &[VarKey],
    ) -> MarginalizationResult<AppliedMarginal> {
        let marginal = self.compute(problem, marginalized)?;
        let rank = marginal.rank;
        let dim = marginal.dim();
        let absorbed = marginal.absorbed.clone();

        let (keys, prior) = marginal.into_factor()?;

        let removed_blocks = problem.remove_residual_blocks(&absorbed);
        for key in marginalized {
            problem.try_remove_variable(*key)?;
        }
        let prior_key = problem.try_add_residual_block(&keys, Box::new(prior), None)?;

        Ok(AppliedMarginal {
            prior: prior_key,
            kept: keys,
            rank,
            dim,
            removed_blocks,
        })
    }

    /// `S` with `SᵀS = Λ_p`, the pseudo-inverse `Λ_p⁺`, and the numeric rank.
    fn factor_information(
        &self,
        information: &DMatrix<f64>,
    ) -> MarginalizationResult<(DMatrix<f64>, DMatrix<f64>, usize)> {
        let dim = information.nrows();
        match self.sqrt_information {
            SqrtInformation::Cholesky => {
                let Some(chol) = information.clone().cholesky() else {
                    return Err(MarginalizationError::RankDeficient { rank: 0, dim });
                };
                let sqrt = chol.l().transpose();
                let Some(pseudo) = pseudo_inverse(information, 1e-12) else {
                    return Err(MarginalizationError::RankDeficient { rank: 0, dim });
                };
                Ok((sqrt, pseudo, dim))
            }
            SqrtInformation::Eigen { rank_tolerance } => {
                let eigen = SymmetricEigen::new(information.clone());
                let max = eigen
                    .eigenvalues
                    .iter()
                    .fold(0.0f64, |acc, v| acc.max(v.abs()));
                let floor = rank_tolerance * max;

                let mut sqrt_diagonal = DVector::zeros(dim);
                let mut inverse_diagonal = DVector::zeros(dim);
                let mut rank = 0usize;
                for (index, value) in eigen.eigenvalues.iter().enumerate() {
                    if *value > floor && *value > 0.0 {
                        sqrt_diagonal[index] = value.sqrt();
                        inverse_diagonal[index] = 1.0 / value;
                        rank += 1;
                    }
                }

                // S = diag(√λ)·Vᵀ, square, with a zero row per clamped
                // direction — `MarginalPriorFactor::new` requires a square
                // `sqrt_info`, so a thin S is not an option.
                let vt = eigen.eigenvectors.transpose();
                let sqrt = DMatrix::from_diagonal(&sqrt_diagonal) * &vt;
                let pseudo = &eigen.eigenvectors * DMatrix::from_diagonal(&inverse_diagonal) * &vt;
                Ok((sqrt, pseudo, rank))
            }
        }
    }
}

/// Moore–Penrose pseudo-inverse of a symmetric matrix, clamping small
/// eigenvalues to zero. `None` when every eigenvalue is clamped.
fn pseudo_inverse(matrix: &DMatrix<f64>, relative_tolerance: f64) -> Option<DMatrix<f64>> {
    if matrix.nrows() == 0 {
        return Some(DMatrix::zeros(0, 0));
    }
    let eigen = SymmetricEigen::new(matrix.clone());
    let max = eigen
        .eigenvalues
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
    if max <= 0.0 {
        return None;
    }
    let floor = relative_tolerance * max;
    let mut inverse = DVector::zeros(eigen.eigenvalues.len());
    let mut rank = 0usize;
    for (index, value) in eigen.eigenvalues.iter().enumerate() {
        if value.abs() > floor {
            inverse[index] = 1.0 / value;
            rank += 1;
        }
    }
    if rank == 0 {
        return None;
    }
    Some(&eigen.eigenvectors * DMatrix::from_diagonal(&inverse) * eigen.eigenvectors.transpose())
}

/// The Gaussian left behind by eliminating a set of variables.
pub struct Marginal {
    /// Kept variables, in the order the prior's blocks must be registered.
    pub kept: Vec<VarKey>,
    /// Tangent dimension of each kept block, parallel to [`Self::kept`].
    pub dims: Vec<usize>,
    /// `Λ_p = Λ_kk − Λ_km·Λ_mm⁻¹·Λ_mk`, symmetric positive semi-definite.
    pub information: DMatrix<f64>,
    /// `g_p = g_k − Λ_km·Λ_mm⁻¹·g_m`, with `g = Jᵀr` — the *positive* gradient.
    pub gradient: DVector<f64>,
    /// `S` with `SᵀS = Λ_p`. Square; unobservable directions are zero rows.
    pub sqrt_information: DMatrix<f64>,
    /// `b = −Λ_p⁺·g_p`, the prior's minimizer in the tangent space of `x₀`.
    pub offset: DVector<f64>,
    /// Ambient parameters of each kept block at the linearization point.
    pub linearization_point: Vec<DVector<f64>>,
    /// Manifold of each kept block, parallel to [`Self::kept`].
    pub manifolds: Vec<ManifoldType>,
    /// Numeric rank of `Λ_p`; equals `dim()` for a fully determined marginal.
    pub rank: usize,
    /// The residual blocks this marginal absorbed.
    pub absorbed: Vec<FactorKey>,
}

impl Marginal {
    /// Total kept tangent dimension.
    pub fn dim(&self) -> usize {
        self.dims.iter().sum()
    }

    /// Dimension of the unobservable subspace.
    pub fn nullity(&self) -> usize {
        self.dim().saturating_sub(self.rank)
    }

    /// Build the prior. The returned keys are [`Self::kept`], in order.
    ///
    /// # Errors
    ///
    /// [`MarginalizationError::Factor`] if the factor rejects the dimensions.
    pub fn into_factor(self) -> MarginalizationResult<(Vec<VarKey>, MarginalPriorFactor)> {
        let logs: Vec<BlockLog> = self
            .manifolds
            .iter()
            .zip(self.linearization_point.iter())
            .map(|(manifold, x0)| block_local_log(*manifold, x0.as_slice()))
            .collect();
        let dims = self.dims.clone();

        let local_log: LocalLogFn = Box::new(move |params, out| {
            let mut offset = 0usize;
            for ((log, dim), param) in logs.iter().zip(dims.iter()).zip(params.iter()) {
                let end = (offset + dim).min(out.len());
                if offset < end {
                    log(param, &mut out[offset..end]);
                }
                offset += dim;
            }
        });

        let factor =
            MarginalPriorFactor::new(self.dims, self.sqrt_information, self.offset, local_log)
                .map_err(MarginalizationError::Factor)?;
        Ok((self.kept, factor))
    }
}

/// What [`Marginalizer::apply`] did.
#[derive(Debug, Clone)]
pub struct AppliedMarginal {
    /// The registered prior.
    pub prior: FactorKey,
    /// The variables it constrains.
    pub kept: Vec<VarKey>,
    /// Numeric rank of the marginal information.
    pub rank: usize,
    /// Total kept tangent dimension.
    pub dim: usize,
    /// How many residual blocks were absorbed and removed.
    pub removed_blocks: usize,
}

/// `θ = x ⊟ x₀` for one block.
type BlockLog = Box<dyn Fn(&[f64], &mut [f64]) + Send + Sync>;

/// Build the local-tangent closure for one manifold at a frozen `x₀`.
///
/// `MarginalPriorFactor` is manifold-agnostic by design, so the `⊟` it needs is
/// supplied from here, where the variable's type is known.
fn block_local_log(manifold: ManifoldType, x0: &[f64]) -> BlockLog {
    macro_rules! log_for {
        ($group:ty) => {{
            let anchor = <$group>::from_param_slice(x0);
            Box::new(move |params: &[f64], out: &mut [f64]| {
                let tangent = <$group>::from_param_slice(params).right_minus(&anchor, None, None);
                let slice = tangent.as_slice();
                let n = slice.len().min(out.len());
                out[..n].copy_from_slice(&slice[..n]);
            })
        }};
    }

    match manifold {
        ManifoldType::RN => log_for!(rn::Rn),
        ManifoldType::SO2 => log_for!(so2::SO2),
        ManifoldType::SO3 => log_for!(so3::SO3),
        ManifoldType::SE2 => log_for!(se2::SE2),
        ManifoldType::SE3 => log_for!(se3::SE3),
        ManifoldType::SE23 => log_for!(se23::SE23),
        ManifoldType::SGal3 => log_for!(sgal3::SGal3),
        ManifoldType::Sim3 => log_for!(sim3::Sim3),
    }
}

/// Why a marginalization could not be computed.
#[derive(Debug, thiserror::Error)]
pub enum MarginalizationError {
    /// No variables were requested.
    #[error("nothing to marginalize")]
    EmptyRequest,

    /// A requested key is not in the problem.
    #[error("unknown variable key {0:?}")]
    UnknownVariable(VarKey),

    /// A requested variable participates in no residual block.
    #[error("variable {0:?} appears in no residual block; drop it instead of marginalizing it")]
    Unobserved(VarKey),

    /// Every absorbed block touches only marginalized variables.
    #[error(
        "marginalizing {0:?} would leave no retained variable; the absorbed factors touch \
         nothing else, so there is nothing to summarize"
    )]
    NoKeptVariables(VarKey),

    /// `Λ_mm` is singular, which is a modelling error rather than a gauge
    /// freedom: a marginalized variable nothing observes, or observations that
    /// are degenerate. Drop the variable and its factors instead.
    #[error(
        "the eliminated block is singular ({context}); a marginalized variable is \
         unobserved or its observations are degenerate"
    )]
    SingularEliminatedBlock {
        /// What was being eliminated when it failed.
        context: String,
    },

    /// `Λ_p` is rank deficient and [`SqrtInformation::Cholesky`] was requested.
    #[error(
        "the marginal information is rank deficient ({rank} of {dim}); use \
         SqrtInformation::Eigen to keep the null space"
    )]
    RankDeficient {
        /// Numeric rank.
        rank: usize,
        /// Total dimension.
        dim: usize,
    },

    /// The prior factor rejected its dimensions.
    #[error("marginal prior rejected: {0}")]
    Factor(String),

    /// A block could not be linearized.
    #[error(transparent)]
    Linearization(#[from] LinearizerError),

    /// The problem rejected a mutation.
    #[error(transparent)]
    Core(#[from] CoreError),
}

/// Result alias for this module.
pub type MarginalizationResult<T> = Result<T, MarginalizationError>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::noise::NoiseModel;
    use crate::factors::pose::{BetweenFactor, EuclideanPriorFactor};
    use crate::linalg::JacobianMode;
    use crate::optimizer::levenberg_marquardt::LevenbergMarquardt;
    use apex_manifolds::rn::Rn;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn rn(values: &[f64]) -> DVector<f64> {
        DVector::from_column_slice(values)
    }

    fn solver() -> LevenbergMarquardt {
        LevenbergMarquardt::with_config(
            crate::optimizer::levenberg_marquardt::LevenbergMarquardtConfig::new()
                .with_max_iterations(200)
                .with_cost_tolerance(1e-16)
                .with_parameter_tolerance(1e-14)
                .with_gradient_tolerance(1e-16),
        )
    }

    /// A three-node chain of `Rn(2)` variables. Everything is linear, so the
    /// marginal is *exact* and equivalence can be asserted to solver tolerance
    /// rather than to a fudge factor.
    fn linear_chain(start: [f64; 3]) -> (Problem, Vec<VarKey>, Vec<FactorKey>) {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let keys: Vec<VarKey> = start
            .iter()
            .map(|x| problem.add_variable(ManifoldType::RN, rn(&[*x, 0.0])))
            .collect();

        let mut blocks = vec![problem.add_residual_block_with_noise(
            &[keys[0]],
            Box::new(EuclideanPriorFactor::new(rn(&[0.0, 0.0]))),
            None,
            NoiseModel::from_sigmas(&[0.1, 0.1]).unwrap_or_else(|e| panic!("{e}")),
        )];
        for pair in keys.windows(2) {
            blocks.push(problem.add_residual_block_with_noise(
                &[pair[0], pair[1]],
                Box::new(BetweenFactor::new(Rn::from_vec(vec![1.0, 0.0]))),
                None,
                NoiseModel::from_sigmas(&[0.05, 0.05]).unwrap_or_else(|e| panic!("{e}")),
            ));
        }
        (problem, keys, blocks)
    }

    fn value(problem: &Problem, key: VarKey) -> Vec<f64> {
        problem.variable_params(key).unwrap_or(&[]).to_vec()
    }

    #[test]
    fn absorbed_blocks_are_exactly_the_incident_ones() -> TestResult {
        let (problem, keys, _blocks) = linear_chain([0.0, 1.0, 2.0]);
        let marginal = Marginalizer::new().compute(&problem, &[keys[0]])?;
        // The prior on x0 and the x0-x1 edge; the x1-x2 edge stays.
        assert_eq!(marginal.absorbed.len(), 2);
        assert_eq!(marginal.kept, vec![keys[1]]);
        assert_eq!(marginal.dims, vec![2]);
        Ok(())
    }

    #[test]
    fn sqrt_information_squares_back_to_lambda() -> TestResult {
        let (problem, keys, _blocks) = linear_chain([0.3, 1.4, 2.1]);
        let marginal = Marginalizer::new().compute(&problem, &[keys[0]])?;
        let reconstructed = marginal.sqrt_information.transpose() * &marginal.sqrt_information;
        let error = (&reconstructed - &marginal.information).norm();
        assert!(error < 1e-9, "SᵀS != Λ_p, error {error:.3e}");
        Ok(())
    }

    /// The sign trap. `g = Jᵀr` is positive and the solved system is
    /// `H·δ = −g`, so `b = −Λ⁺g`. Flipped, the prior's minimizer sits `2b`
    /// away — and `b` is far from zero here precisely so that shows up.
    #[test]
    fn offset_sign_reproduces_the_conditional_minimizer() -> TestResult {
        // Linearized away from the optimum, so b is large.
        let (problem, keys, _blocks) = linear_chain([2.0, 5.0, 9.0]);
        let marginal = Marginalizer::new().compute(&problem, &[keys[0]])?;

        assert!(
            marginal.offset.norm() > 0.1,
            "b is ~zero ({:.3e}); this test would pass with either sign",
            marginal.offset.norm()
        );

        // The prior's minimizer is x0 ⊞ b. For Rn that is x0 + b, and it must
        // equal the value x1 takes when x0 is optimized out.
        let x1 = value(&problem, keys[1]);
        let predicted = x1[0] + marginal.offset[0];

        let mut full = problem;
        let result = solver().optimize(&mut full)?;
        let solved_x1 = result.parameters[keys[1]].as_param_slice()[0];

        assert!(
            (predicted - solved_x1).abs() < 1e-6,
            "prior minimizer {predicted:.6} != conditional optimum {solved_x1:.6}; \
             a flipped sign would land near {:.6}",
            x1[0] - marginal.offset[0]
        );
        Ok(())
    }

    /// The acceptance test: a marginalized problem reproduces the full
    /// solution. Exact, because the model is linear.
    #[test]
    fn marginalization_reproduces_the_full_solution() -> TestResult {
        let (mut full, keys, _blocks) = linear_chain([2.0, 5.0, 9.0]);
        let expected = solver().optimize(&mut full)?;
        let x1_full = expected.parameters[keys[1]].as_param_slice()[0];
        let x2_full = expected.parameters[keys[2]].as_param_slice()[0];

        let (mut reduced, keys, _blocks) = linear_chain([2.0, 5.0, 9.0]);
        let applied = Marginalizer::new().apply(&mut reduced, &[keys[0]])?;
        assert_eq!(applied.removed_blocks, 2);
        assert_eq!(reduced.num_variables(), 2, "x0 should be gone");

        // Start the reduced problem somewhere else entirely, so the answer
        // comes from the prior rather than from where it was left.
        reduced.set_variable_params(keys[1], &[-4.0, 0.0])?;
        reduced.set_variable_params(keys[2], &[7.5, 0.0])?;
        let got = solver().optimize(&mut reduced)?;

        let x1 = got.parameters[keys[1]].as_param_slice()[0];
        let x2 = got.parameters[keys[2]].as_param_slice()[0];
        assert!(
            (x1 - x1_full).abs() < 1e-6 && (x2 - x2_full).abs() < 1e-6,
            "reduced ({x1:.6}, {x2:.6}) != full ({x1_full:.6}, {x2_full:.6})"
        );
        Ok(())
    }

    /// A variable held only by a *relative* edge carries no absolute
    /// information, so eliminating it leaves `Λ_p = 0` — nullity equal to the
    /// full dimension.
    ///
    /// This is the gauge case in its purest form, and it is the one that breaks
    /// a Cholesky square root. The eigen route must record it as zero rows in a
    /// square `S`, not fail and not fabricate information in the free
    /// directions; regularizing `Λ_p` to make Cholesky work would manufacture
    /// exactly the information this is protecting.
    #[test]
    fn a_relative_only_constraint_leaves_no_absolute_information() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let keys: Vec<VarKey> = [0.0, 1.0, 2.0]
            .iter()
            .map(|x| problem.add_variable(ManifoldType::RN, rn(&[*x, 0.0])))
            .collect();
        for pair in keys.windows(2) {
            problem.add_residual_block_with_noise(
                &[pair[0], pair[1]],
                Box::new(BetweenFactor::new(Rn::from_vec(vec![1.0, 0.0]))),
                None,
                NoiseModel::from_sigmas(&[0.05, 0.05]).unwrap_or_else(|e| panic!("{e}")),
            );
        }

        let marginal = Marginalizer::new().compute(&problem, &[keys[0]])?;
        assert_eq!(marginal.dim(), 2);
        assert_eq!(
            marginal.rank, 0,
            "a relative-only edge cannot localize the variable it leaves behind"
        );
        assert_eq!(marginal.nullity(), 2);

        // Square, finite, and identically zero — never NaN from dividing by a
        // clamped eigenvalue.
        assert_eq!(marginal.sqrt_information.nrows(), 2);
        assert_eq!(marginal.sqrt_information.ncols(), 2);
        assert!(
            marginal.sqrt_information.norm() < 1e-12,
            "expected an all-zero S, got norm {:.3e}",
            marginal.sqrt_information.norm()
        );
        assert!(
            marginal.offset.iter().all(|v| v.is_finite()),
            "the offset must not divide by a clamped eigenvalue"
        );
        Ok(())
    }

    #[test]
    fn cholesky_mode_rejects_a_rank_deficient_marginal() {
        let information = DMatrix::<f64>::zeros(3, 3);
        let marginalizer = Marginalizer::new().with_sqrt_information(SqrtInformation::Cholesky);
        let Err(MarginalizationError::RankDeficient { dim, .. }) =
            marginalizer.factor_information(&information)
        else {
            panic!("expected RankDeficient on a zero information matrix");
        };
        assert_eq!(dim, 3);
    }

    #[test]
    fn an_empty_request_is_rejected() {
        let (problem, _keys, _blocks) = linear_chain([0.0, 1.0, 2.0]);
        let Err(MarginalizationError::EmptyRequest) = Marginalizer::new().compute(&problem, &[])
        else {
            panic!("expected EmptyRequest");
        };
    }

    #[test]
    fn an_unobserved_variable_is_named() {
        let (mut problem, _keys, _blocks) = linear_chain([0.0, 1.0, 2.0]);
        let lonely = problem.add_variable(ManifoldType::RN, rn(&[9.0, 9.0]));
        let Err(MarginalizationError::Unobserved(key)) =
            Marginalizer::new().compute(&problem, &[lonely])
        else {
            panic!("expected Unobserved");
        };
        assert_eq!(key, lonely);
    }

    /// Marginalizing everything the absorbed factors touch leaves nothing to
    /// summarize, and that is an error rather than an empty prior.
    #[test]
    fn marginalizing_the_whole_blanket_is_rejected() {
        let (problem, keys, _blocks) = linear_chain([0.0, 1.0, 2.0]);
        let Err(MarginalizationError::NoKeptVariables(_)) =
            Marginalizer::new().compute(&problem, &keys)
        else {
            panic!("expected NoKeptVariables");
        };
    }
}
