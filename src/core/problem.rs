use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{Error, Write},
};

use faer::{Mat, sparse::SparseColMat};
use nalgebra::DVector;
use rayon::prelude::*;
use slotmap::{SecondaryMap, SlotMap};

use crate::{
    core::CoreResult,
    core::{
        CoreError, FactorKey, VarKey,
        loss_functions::LossFunction,
        noise::NoiseModel,
        residual_block::ResidualBlock,
        variable::{ManifoldVariable, Variable},
    },
    factors::Factor,
    linalg::JacobianMode,
};
use apex_manifolds::{LieGroup, ManifoldType, rn, se2, se3, se23, sgal3, sim3, so2, so3};

pub use crate::linearizer::cpu::sparse::SymbolicStructure;

pub struct Problem {
    pub(crate) total_residual_dimension: usize,
    pub(crate) jacobian_mode: JacobianMode,
    pub(crate) variables: SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    residual_blocks: SlotMap<FactorKey, ResidualBlock>,
    pub(crate) fixed_variable_indexes: SecondaryMap<VarKey, HashSet<usize>>,
    pub(crate) variable_bounds: SecondaryMap<VarKey, HashMap<usize, (f64, f64)>>,
    pub(crate) schur_landmark_keys: HashSet<VarKey>,
}

impl Default for Problem {
    fn default() -> Self {
        Self::new(JacobianMode::Sparse)
    }
}

impl Problem {
    pub fn new(jacobian_mode: JacobianMode) -> Self {
        Self {
            total_residual_dimension: 0,
            jacobian_mode,
            variables: SlotMap::with_key(),
            residual_blocks: SlotMap::with_key(),
            fixed_variable_indexes: SecondaryMap::new(),
            variable_bounds: SecondaryMap::new(),
            schur_landmark_keys: HashSet::new(),
        }
    }

    /// Mark a variable to be eliminated by the Schur complement solver.
    ///
    /// Nothing is eliminated by default — not even `Rn(3)` landmarks. A
    /// bundle-adjustment problem only uses the Schur path once every landmark
    /// carries an explicit mark (or the solver is configured with
    /// auto-detection); unmarked variables are retained and form the reduced
    /// system. The
    /// eliminated set is not restricted to 3-DOF landmarks: any DOF works, sizes
    /// may be mixed within one problem, and the eliminated variables need not be
    /// adjacent in the variable ordering. That covers inverse-depth
    /// parameterizations (1 DOF), LiDAR features, and sliding-window
    /// marginalization of whole poses (6 DOF), as well as classic bundle
    /// adjustment.
    ///
    /// # Precondition
    ///
    /// Eliminated variables must be **mutually unconnected** — no factor may
    /// touch two of them. That is what makes `H_ee` block-diagonal and its
    /// inverse cheap. Violating it is reported as an error on the first solve,
    /// naming both variables, rather than silently producing a wrong step.
    pub fn mark_for_elimination(&mut self, key: VarKey) {
        self.schur_landmark_keys.insert(key);
    }

    /// Bundle-adjustment-flavoured alias for [`Self::mark_for_elimination`].
    #[deprecated(
        since = "1.6.0",
        note = "renamed to `mark_for_elimination`: elimination is not restricted to landmarks"
    )]
    pub fn mark_as_schur_landmark(&mut self, key: VarKey) {
        self.mark_for_elimination(key);
    }

    /// Add a variable with a given manifold type and initial parameter vector.
    ///
    /// Returns a stable `VarKey` handle for use in `add_residual_block`, `fix_variable`, etc.
    pub fn add_variable(&mut self, manifold_type: ManifoldType, params: DVector<f64>) -> VarKey {
        let var = Self::create_variable(&manifold_type, &params);
        self.variables.insert(var)
    }

    /// Add a residual block (factor + optional loss) connecting the given variables.
    ///
    /// Returns a `FactorKey` that can be used to remove the block later.
    pub fn add_residual_block(
        &mut self,
        variable_keys: &[VarKey],
        factor: Box<dyn Factor + Send + Sync>,
        loss_func: Option<Box<dyn LossFunction + Send + Sync>>,
    ) -> FactorKey {
        self.add_residual_block_with_noise(variable_keys, factor, loss_func, NoiseModel::null())
    }

    /// Register a residual block with a measurement noise model, keeping the
    /// validation behaviour of [`Self::try_add_residual_block`].
    ///
    /// The model carries the square-root information `S`; residuals and
    /// Jacobians are whitened (`S·r`, `S·J`) before the robust-loss corrector,
    /// so the optimized objective is `Σ ½·ρ(‖S·r‖²)` — the Ω-weighted
    /// objective g2o-style benchmarks report. Its dimension must equal the
    /// factor's `residual_dim()`.
    pub fn add_residual_block_with_noise(
        &mut self,
        variable_keys: &[VarKey],
        factor: Box<dyn Factor + Send + Sync>,
        loss_func: Option<Box<dyn LossFunction + Send + Sync>>,
        noise: NoiseModel,
    ) -> FactorKey {
        self.try_add_residual_block_with_noise(variable_keys, factor, loss_func, noise)
            .unwrap_or_else(|e| panic!("invalid residual block: {e}"))
    }

    /// [`Self::add_residual_block_with_noise`] returning a typed error.
    pub fn try_add_residual_block_with_noise(
        &mut self,
        variable_keys: &[VarKey],
        factor: Box<dyn Factor + Send + Sync>,
        loss_func: Option<Box<dyn LossFunction + Send + Sync>>,
        noise: NoiseModel,
    ) -> CoreResult<FactorKey> {
        if noise.dim() != 0 && noise.dim() != factor.residual_dim() {
            return Err(CoreError::DimensionMismatch(format!(
                "noise model covers {} residual rows but the factor produces {}",
                noise.dim(),
                factor.residual_dim()
            )));
        }
        if factor.whitens_internally() && !matches!(noise, NoiseModel::Null) {
            return Err(CoreError::InvalidInput(
                "this factor applies its own square-root information; register it with \
                 NoiseModel::null(). Supplying a noise model as well would whiten the \
                 residual twice."
                    .into(),
            ));
        }
        self.try_add_residual_block_impl(variable_keys, factor, loss_func, noise)
    }

    /// Register a residual block, validating it against the referenced
    /// variables first.
    ///
    /// Runs the factor's [`Factor::validate_variables`] hook so shape
    /// mismatches (e.g. a landmark variable whose parameter count disagrees
    /// with the factor's observations) surface as a typed registration error
    /// instead of a panic inside the parallel assembly. Unknown variable keys
    /// are rejected as well — they would otherwise be silently skipped during
    /// linearization.
    pub fn try_add_residual_block(
        &mut self,
        variable_keys: &[VarKey],
        factor: Box<dyn Factor + Send + Sync>,
        loss_func: Option<Box<dyn LossFunction + Send + Sync>>,
    ) -> CoreResult<FactorKey> {
        self.try_add_residual_block_impl(variable_keys, factor, loss_func, NoiseModel::Null)
    }

    fn try_add_residual_block_impl(
        &mut self,
        variable_keys: &[VarKey],
        factor: Box<dyn Factor + Send + Sync>,
        loss_func: Option<Box<dyn LossFunction + Send + Sync>>,
        noise: NoiseModel,
    ) -> CoreResult<FactorKey> {
        let mut variables: Vec<&dyn ManifoldVariable> = Vec::with_capacity(variable_keys.len());
        for &key in variable_keys {
            let variable = self
                .variables
                .get(key)
                .ok_or_else(|| {
                    CoreError::Variable(format!(
                        "residual block references unknown variable key {key:?}"
                    ))
                })?
                .as_ref();
            variables.push(variable);
        }
        factor
            .validate_variables(&variables)
            .map_err(CoreError::DimensionMismatch)?;

        let new_residual_dimension = factor.residual_dim();
        let row_start = self.total_residual_dimension;
        let fk = self.residual_blocks.insert_with_key(|fk| {
            ResidualBlock::with_noise(fk, row_start, variable_keys, factor, loss_func, noise)
        });
        self.total_residual_dimension += new_residual_dimension;
        Ok(fk)
    }

    /// Reassign residual row offsets so each eliminated variable's rows form
    /// one contiguous range.
    ///
    /// Chunk-wise Schur elimination sweeps chunks in increasing row order,
    /// which requires the rows of any one eliminated variable to be adjacent.
    /// Bundle-adjustment data is usually camera-major — BAL lists observations
    /// camera by camera — so each landmark's rows arrive scattered across the
    /// matrix and the sweep cannot run.
    ///
    /// Rows are a labelling, not part of the problem: permuting them permutes
    /// `r` and the rows of `J` together, leaving `JᵀJ`, `Jᵀr` and hence the step
    /// unchanged. The cost is a sum of squares, so it is unchanged too.
    ///
    /// Blocks touching no eliminated variable (priors on retained variables,
    /// say) are placed first, ahead of every chunk, so they never fall inside a
    /// chunk's range.
    ///
    /// Idempotent, and a no-op when nothing is marked for elimination.
    /// Returns whether any offset actually moved.
    pub(crate) fn group_rows_for_elimination(&mut self) -> bool {
        if self.schur_landmark_keys.is_empty() {
            return false;
        }

        // Order: unchunked blocks first, then blocks grouped by the eliminated
        // variable they touch. Ties broken by the existing offset so the result
        // is deterministic and idempotent.
        //
        // Groups are ordered by the eliminated variable's *column position*
        // (slotmap iteration order — the same order
        // `build_variable_index_map` lays columns out in), not by `VarKey`
        // sort order. Key ordering is version-major, so after a
        // remove-and-reinsert cycle it can disagree with column order, and
        // `ChunkLayout::build` emits ranges in column order while demanding
        // increasing rows — a disagreement it can only report as a layout
        // failure despite the problem being regroupable.
        let rank: std::collections::HashMap<VarKey, usize> = self
            .variables
            .keys()
            .enumerate()
            .map(|(rank, key)| (key, rank))
            .collect();
        let mut ordered: Vec<(Option<usize>, usize, FactorKey)> = self
            .residual_blocks
            .iter()
            .map(|(key, block)| {
                let eliminated = block
                    .variable_keys
                    .iter()
                    .find(|k| self.schur_landmark_keys.contains(k))
                    .copied()
                    .and_then(|k| rank.get(&k).copied());
                (eliminated, block.residual_row_start_idx, key)
            })
            .collect();
        // `None` sorts before `Some`, putting unchunked rows first.
        ordered.sort_by_key(|(eliminated, offset, _)| (*eliminated, *offset));

        let mut moved = false;
        let mut row = 0usize;
        for (_, _, key) in &ordered {
            if let Some(block) = self.residual_blocks.get_mut(*key) {
                if block.residual_row_start_idx != row {
                    block.residual_row_start_idx = row;
                    moved = true;
                }
                row += block.factor.residual_dim();
            }
        }
        debug_assert_eq!(
            row, self.total_residual_dimension,
            "regrouping must preserve the total residual dimension"
        );
        moved
    }

    pub fn remove_residual_block(&mut self, block_id: FactorKey) -> Option<ResidualBlock> {
        if let Some(block) = self.residual_blocks.remove(block_id) {
            self.total_residual_dimension -= block.factor.residual_dim();
            Some(block)
        } else {
            None
        }
    }

    /// Hold tangent-space component `idx` of `var_key` fixed during solving.
    ///
    /// # Panics
    /// If the key is unknown or `idx` is outside the variable's DOF. Use
    /// [`Self::try_fix_variable`] to handle that as an error.
    pub fn fix_variable(&mut self, var_key: VarKey, idx: usize) {
        self.try_fix_variable(var_key, idx)
            .unwrap_or_else(|e| panic!("invalid variable constraint: {e}"))
    }

    /// [`Self::fix_variable`] returning a typed error.
    ///
    /// An out-of-range `idx` used to be accepted and then silently ignored
    /// forever, because the constraint is only ever applied by index against a
    /// tangent vector that never has that component.
    pub fn try_fix_variable(&mut self, var_key: VarKey, idx: usize) -> CoreResult<()> {
        let dof = self.variable_dof(var_key)?;
        if idx >= dof {
            return Err(CoreError::Variable(format!(
                "cannot fix component {idx} of a {dof}-DOF variable"
            )));
        }
        if let Some(set) = self.fixed_variable_indexes.get_mut(var_key) {
            set.insert(idx);
        } else {
            self.fixed_variable_indexes
                .insert(var_key, HashSet::from([idx]));
        }
        Ok(())
    }

    /// Tangent-space dimension of `var_key`, or an error if the key is unknown.
    fn variable_dof(&self, var_key: VarKey) -> CoreResult<usize> {
        self.variables
            .get(var_key)
            .map(|v| v.dof())
            .ok_or_else(|| CoreError::Variable(format!("unknown variable key {var_key:?}")))
    }

    pub fn unfix_variable(&mut self, var_key: VarKey) {
        self.fixed_variable_indexes.remove(var_key);
    }

    /// Constrain tangent-space component `idx` of `var_key` to `[lower, upper]`.
    ///
    /// # Panics
    /// If the key is unknown, `idx` is outside the variable's DOF, or the range
    /// is inverted. Use [`Self::try_set_variable_bounds`] to handle that as an
    /// error.
    pub fn set_variable_bounds(
        &mut self,
        var_key: VarKey,
        idx: usize,
        lower_bound: f64,
        upper_bound: f64,
    ) {
        self.try_set_variable_bounds(var_key, idx, lower_bound, upper_bound)
            .unwrap_or_else(|e| panic!("invalid variable bounds: {e}"))
    }

    /// [`Self::set_variable_bounds`] returning a typed error.
    ///
    /// An inverted range used to be warned about and then dropped, leaving the
    /// caller unable to distinguish "bound set" from "bound rejected".
    pub fn try_set_variable_bounds(
        &mut self,
        var_key: VarKey,
        idx: usize,
        lower_bound: f64,
        upper_bound: f64,
    ) -> CoreResult<()> {
        let dof = self.variable_dof(var_key)?;
        if idx >= dof {
            return Err(CoreError::Variable(format!(
                "cannot bound component {idx} of a {dof}-DOF variable"
            )));
        }
        if lower_bound > upper_bound {
            return Err(CoreError::Variable(format!(
                "lower bound {lower_bound} exceeds upper bound {upper_bound}"
            )));
        }
        if let Some(map) = self.variable_bounds.get_mut(var_key) {
            map.insert(idx, (lower_bound, upper_bound));
        } else {
            self.variable_bounds
                .insert(var_key, HashMap::from([(idx, (lower_bound, upper_bound))]));
        }
        Ok(())
    }

    pub fn remove_variable_bounds(&mut self, var_key: VarKey) {
        self.variable_bounds.remove(var_key);
    }

    fn create_variable(
        manifold_type: &ManifoldType,
        params: &DVector<f64>,
    ) -> Box<dyn ManifoldVariable> {
        match manifold_type {
            ManifoldType::SO2 => {
                Box::new(Variable::new(so2::SO2::from_param_slice(params.as_slice())))
            }
            ManifoldType::SO3 => {
                Box::new(Variable::new(so3::SO3::from_param_slice(params.as_slice())))
            }
            ManifoldType::SE2 => {
                Box::new(Variable::new(se2::SE2::from_param_slice(params.as_slice())))
            }
            ManifoldType::SE3 => {
                Box::new(Variable::new(se3::SE3::from_param_slice(params.as_slice())))
            }
            ManifoldType::RN => Box::new(Variable::new(rn::Rn::new(params.clone()))),
            ManifoldType::SE23 => Box::new(Variable::new(se23::SE23::from_param_slice(
                params.as_slice(),
            ))),
            ManifoldType::SGal3 => Box::new(Variable::new(sgal3::SGal3::from_param_slice(
                params.as_slice(),
            ))),
            ManifoldType::Sim3 => Box::new(Variable::new(sim3::Sim3::from_param_slice(
                params.as_slice(),
            ))),
        }
    }

    /// Apply fixed indices and bounds to a mutable clone of the variable map.
    ///
    /// Optimizers call this after copying `problem.variables` to initialize working state.
    pub fn apply_constraints_to_variables(
        &self,
        variables: &mut SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    ) {
        for (key, var) in variables.iter_mut() {
            if let Some(indexes) = self.fixed_variable_indexes.get(key) {
                var.set_fixed_indices(indexes.clone());
            }
            if let Some(bounds) = self.variable_bounds.get(key) {
                var.set_bounds(bounds.clone());
            }
        }
    }

    pub fn num_residual_blocks(&self) -> usize {
        self.residual_blocks.len()
    }

    /// Total number of scalar residuals across all residual blocks.
    ///
    /// This is the row count `m` of the assembled Jacobian.
    pub fn total_residual_dimension(&self) -> usize {
        self.total_residual_dimension
    }

    pub(crate) fn residual_blocks(&self) -> &SlotMap<FactorKey, ResidualBlock> {
        &self.residual_blocks
    }

    /// Compute only the residual vector (no Jacobian) for the given variable values.
    pub fn compute_residual_sparse(
        &self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    ) -> CoreResult<Mat<f64>> {
        Ok(self.compute_residual_and_cost_sparse(variables)?.0)
    }

    /// Compute the residual vector together with the objective value.
    ///
    /// The cost is **not** `0.5·‖r‖²` of the returned vector. For blocks carrying a
    /// robust loss the returned residual is Triggs-corrected — it exists to drive
    /// the linear system — while the cost is the true robust cost `0.5·ρ(‖r‖²)`.
    /// Squaring the corrected residual gives a different function, which is what
    /// made every reported cost and every trust-region ratio wrong for robust
    /// problems.
    ///
    /// For blocks with no loss function the two coincide at `0.5·‖r‖²`.
    pub fn compute_residual_and_cost_sparse(
        &self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    ) -> CoreResult<(Mat<f64>, f64)> {
        let mut workspace = crate::linearizer::AssemblyWorkspace::build(self);
        self.compute_residual_and_cost_sparse_with_workspace(variables, &mut workspace)
    }

    /// [`Problem::compute_residual_and_cost_sparse`] reusing a per-solve workspace.
    ///
    /// The block ordering and scratch buffers are static for the lifetime of a
    /// solve, so hot paths (per-iteration step evaluation) pass a workspace built
    /// once by [`crate::optimizer::initialize_optimization_state`].
    pub(crate) fn compute_residual_and_cost_sparse_with_workspace(
        &self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        workspace: &mut crate::linearizer::AssemblyWorkspace,
    ) -> CoreResult<(Mat<f64>, f64)> {
        use crate::linearizer::split_by_row_offsets_mut;

        workspace.residual_buf.fill(0.0);
        let residual_slices =
            split_by_row_offsets_mut(&mut workspace.residual_buf, &workspace.offsets_lens);

        // Accumulate the per-block cost on the existing parallel pass rather than
        // making a second traversal.
        let results: Vec<CoreResult<f64>> = residual_slices
            .into_par_iter()
            .zip(workspace.block_order.par_iter())
            .map(|(slice, key)| {
                let block = &self.residual_blocks[*key];
                self.compute_residual_block(block, variables, slice)
            })
            .collect();
        let cost: f64 = results.into_iter().sum::<CoreResult<f64>>()?;

        let n = self.total_residual_dimension;
        Ok((Mat::from_fn(n, 1, |i, _| workspace.residual_buf[i]), cost))
    }

    /// Compute residuals and sparse Jacobian.
    pub fn compute_residual_and_jacobian_sparse(
        &self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
        symbolic_structure: &SymbolicStructure,
    ) -> CoreResult<(Mat<f64>, SparseColMat<usize, f64>)> {
        let mut workspace = crate::linearizer::AssemblyWorkspace::build(self);
        Ok(crate::linearizer::cpu::sparse::assemble_sparse(
            self,
            variables,
            variable_index_map,
            symbolic_structure,
            &mut workspace,
        )?)
    }

    /// Compute residuals and dense Jacobian.
    pub fn compute_residual_and_jacobian_dense(
        &self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
        total_dof: usize,
    ) -> CoreResult<(Mat<f64>, Mat<f64>)> {
        let mut workspace = crate::linearizer::AssemblyWorkspace::build(self);
        Ok(crate::linearizer::cpu::dense::assemble_dense(
            self,
            variables,
            variable_index_map,
            total_dof,
            &mut workspace,
        )?)
    }

    /// Evaluate one residual block into `residual_slice`, returning its cost.
    ///
    /// With a loss function the slice receives the Triggs-corrected residual while
    /// the returned cost is `0.5·ρ(s)`; without one, the slice receives the raw
    /// residual and the cost is `0.5·‖r‖²`.
    fn compute_residual_block(
        &self,
        residual_block: &ResidualBlock,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        residual_slice: &mut [f64],
    ) -> CoreResult<f64> {
        // Single source of truth for block evaluation — the linearizer's
        // shared `compute_block_into` (Jacobian-less here, cost only).
        let (_, cost) =
            crate::linearizer::compute_block_into(residual_block, variables, residual_slice, None)?;
        Ok(cost)
    }

    pub fn log_residual_to_file(
        &self,
        residual: &nalgebra::DVector<f64>,
        filename: &str,
    ) -> Result<(), Error> {
        let mut file = File::create(filename)?;
        writeln!(file, "# Residual vector - {} elements", residual.len())?;
        for (i, &value) in residual.iter().enumerate() {
            writeln!(file, "{}: {:.12}", i, value)?;
        }
        Ok(())
    }

    pub fn log_sparse_jacobian_to_file(
        &self,
        jacobian: &SparseColMat<usize, f64>,
        filename: &str,
    ) -> Result<(), Error> {
        let mut file = File::create(filename)?;
        writeln!(
            file,
            "# Sparse Jacobian matrix - {} x {} ({} non-zeros)",
            jacobian.nrows(),
            jacobian.ncols(),
            jacobian.compute_nnz()
        )?;
        writeln!(file, "# Matrix saved as dimensions and non-zero count only")?;
        Ok(())
    }

    pub fn log_variables_to_file(
        &self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        filename: &str,
    ) -> Result<(), Error> {
        let mut file = File::create(filename)?;
        writeln!(file, "# Variables - {} total", variables.len())?;
        for (_, var) in variables {
            let vec = var.to_dvector();
            write!(file, "[")?;
            for (i, &v) in vec.iter().enumerate() {
                write!(file, "{:.12}", v)?;
                if i < vec.len() - 1 {
                    write!(file, ", ")?;
                }
            }
            writeln!(file, "]")?;
        }
        Ok(())
    }

    /// Compute per-variable covariances at `variables` and store them on the
    /// variables themselves.
    ///
    /// This re-linearizes the problem at the given point and inverts a clean
    /// `H = JᵀJ` — no Levenberg-Marquardt damping and no Jacobi scaling, both of
    /// which are internal solver details and must not appear in the result. See
    /// [`crate::linalg::covariance`].
    ///
    /// `options` selects the factorization algorithm and whether the result is
    /// multiplied by the estimated noise variance `σ̂²` (scaled) or returned as
    /// the raw `H⁻¹` (unscaled). Optimizers pass their `covariance_options`
    /// here.
    ///
    /// Returns `None` if covariance estimation fails (most commonly a
    /// rank-deficient `H` from unfixed gauge freedom), after logging the reason.
    /// Callers that need to handle the failure should use
    /// [`Covariance::compute`](crate::linalg::covariance::Covariance::compute)
    /// directly, which returns a typed error.
    pub fn compute_and_set_covariances(
        &self,
        variables: &mut SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        options: crate::linalg::covariance::CovarianceOptions,
    ) -> Option<SecondaryMap<VarKey, Mat<f64>>> {
        let covariance =
            match crate::linalg::covariance::Covariance::compute(options, self, variables) {
                Ok(covariance) => covariance,
                Err(e) => {
                    tracing::error!("Covariance estimation failed: {e}");
                    return None;
                }
            };

        let per_var = covariance.per_variable();
        for (key, cov) in &per_var {
            if let Some(var) = variables.get_mut(key) {
                var.set_covariance(cov.clone());
            }
        }
        Some(per_var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::loss_functions::HuberLoss;
    use crate::factors::pose::{BetweenFactor, PriorFactor};
    use apex_manifolds::{ManifoldType, se2::SE2, se3::SE3};
    use nalgebra::{Quaternion, Vector3, dvector};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn create_se2_test_problem() -> Result<(Problem, Vec<VarKey>), Box<dyn std::error::Error>> {
        let mut problem = Problem::new(JacobianMode::Sparse);

        let poses = [
            (0.0_f64, 0.0, 0.0),
            (1.0, 0.0, 0.1),
            (1.5, 1.0, 0.5),
            (1.0, 2.0, 1.0),
            (0.0, 2.5, 1.5),
            (-1.0, 2.0, 2.0),
            (-1.5, 1.0, 2.5),
            (-1.0, 0.0, 3.0),
            (-0.5, -0.5, -2.8),
            (0.5, -0.5, -2.3),
        ];

        let keys: Vec<VarKey> = poses
            .iter()
            .map(|&(x, y, t)| problem.add_variable(ManifoldType::SE2, dvector![x, y, t]))
            .collect();

        for i in 0..9 {
            let (fx, fy, ft) = poses[i];
            let (tx, ty, tt) = poses[i + 1];
            problem.add_residual_block(
                &[keys[i], keys[i + 1]],
                Box::new(BetweenFactor::new(SE2::from_xy_angle(
                    tx - fx,
                    ty - fy,
                    tt - ft,
                ))),
                Some(Box::new(HuberLoss::new(1.0)?)),
            );
        }

        let (fx, fy, ft) = poses[9];
        let (tx, ty, tt) = poses[0];
        problem.add_residual_block(
            &[keys[9], keys[0]],
            Box::new(BetweenFactor::new(SE2::from_xy_angle(
                tx - fx,
                ty - fy,
                tt - ft,
            ))),
            Some(Box::new(HuberLoss::new(1.0)?)),
        );

        problem.add_residual_block(
            &[keys[0]],
            Box::new(PriorFactor::<SE2>::new(SE2::identity())),
            None,
        );

        Ok((problem, keys))
    }

    fn create_se3_test_problem() -> Result<(Problem, Vec<VarKey>), Box<dyn std::error::Error>> {
        let mut problem = Problem::new(JacobianMode::Sparse);

        // (tx, ty, tz, qx, qy, qz, qw)
        let poses = [
            (0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.995),
            (1.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.98),
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.3, 0.955),
            (0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.995),
            (1.0, 0.0, 1.0, 0.1, 0.0, 0.1, 0.99),
            (1.0, 1.0, 1.0, 0.1, 0.0, 0.2, 0.975),
            (0.0, 1.0, 1.0, 0.1, 0.0, 0.3, 0.95),
        ];

        let keys: Vec<VarKey> = poses
            .iter()
            .map(|&(tx, ty, tz, qx, qy, qz, qw)| {
                problem.add_variable(ManifoldType::SE3, dvector![tx, ty, tz, qw, qx, qy, qz])
            })
            .collect();

        let edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];

        for (f, t) in edges {
            let fp = poses[f];
            let tp = poses[t];
            let rel = SE3::from_translation_quaternion(
                Vector3::new(tp.0 - fp.0, tp.1 - fp.1, tp.2 - fp.2),
                Quaternion::new(1.0, 0.0, 0.0, 0.0),
            );
            problem.add_residual_block(
                &[keys[f], keys[t]],
                Box::new(BetweenFactor::new(rel)),
                Some(Box::new(HuberLoss::new(1.0)?)),
            );
        }

        problem.add_residual_block(
            &[keys[0]],
            Box::new(PriorFactor::<SE3>::new(SE3::identity())),
            None,
        );

        Ok((problem, keys))
    }

    #[test]
    fn test_problem_construction_se2() -> TestResult {
        let (problem, keys) = create_se2_test_problem()?;
        assert_eq!(problem.num_residual_blocks(), 11);
        assert_eq!(problem.total_residual_dimension, 33);
        assert_eq!(keys.len(), 10);
        assert_eq!(problem.variables.len(), 10);
        Ok(())
    }

    #[test]
    fn test_problem_construction_se3() -> TestResult {
        let (problem, keys) = create_se3_test_problem()?;
        assert_eq!(problem.num_residual_blocks(), 13);
        assert_eq!(keys.len(), 8);
        assert_eq!(problem.variables.len(), 8);
        Ok(())
    }

    #[test]
    fn test_add_variable_returns_distinct_keys() {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let k0 = problem.add_variable(ManifoldType::SE2, dvector![0.0, 0.0, 0.0]);
        let k1 = problem.add_variable(ManifoldType::SE2, dvector![1.0, 0.0, 0.1]);
        assert_ne!(k0, k1);
        assert_eq!(problem.variables.len(), 2);
        assert_eq!(problem.variables[k0].dof(), 3);
    }

    #[test]
    fn test_variable_all_manifold_types() {
        let mut p = Problem::new(JacobianMode::Sparse);
        let so2 = p.add_variable(ManifoldType::SO2, dvector![0.5]);
        let so3 = p.add_variable(ManifoldType::SO3, dvector![1.0, 0.0, 0.0, 0.0]);
        let se2 = p.add_variable(ManifoldType::SE2, dvector![1.0, 2.0, 0.5]);
        let se3 = p.add_variable(
            ManifoldType::SE3,
            dvector![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        );
        let rn = p.add_variable(ManifoldType::RN, dvector![5.0, 6.0]);

        assert_eq!(p.variables[so2].manifold_type_name(), "SO2");
        assert_eq!(p.variables[so3].manifold_type_name(), "SO3");
        assert_eq!(p.variables[se2].manifold_type_name(), "SE2");
        assert_eq!(p.variables[se3].manifold_type_name(), "SE3");
        assert_eq!(p.variables[rn].manifold_type_name(), "Rn");
    }

    #[test]
    fn test_residual_block_add_remove() -> TestResult {
        let mut p = Problem::new(JacobianMode::Sparse);
        let k0 = p.add_variable(ManifoldType::SE2, dvector![0.0, 0.0, 0.0]);
        let k1 = p.add_variable(ManifoldType::SE2, dvector![1.0, 0.0, 0.1]);

        let fk1 = p.add_residual_block(
            &[k0, k1],
            Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.1))),
            Some(Box::new(HuberLoss::new(1.0)?)),
        );
        let fk2 = p.add_residual_block(
            &[k0],
            Box::new(PriorFactor::<SE2>::new(SE2::identity())),
            None,
        );

        assert_ne!(fk1, fk2);
        assert_eq!(p.num_residual_blocks(), 2);
        assert_eq!(p.total_residual_dimension, 6);

        let removed = p.remove_residual_block(fk1);
        assert!(removed.is_some());
        assert_eq!(p.num_residual_blocks(), 1);
        assert_eq!(p.total_residual_dimension, 3);

        assert!(p.remove_residual_block(fk1).is_none());
        Ok(())
    }

    #[test]
    fn test_fix_unfix_variable() {
        let mut p = Problem::new(JacobianMode::Sparse);
        let k0 = p.add_variable(ManifoldType::SE2, dvector![0.0, 0.0, 0.0]);
        let k1 = p.add_variable(ManifoldType::SE2, dvector![1.0, 0.0, 0.1]);

        p.fix_variable(k0, 0);
        p.fix_variable(k0, 1);
        p.fix_variable(k1, 2);

        assert_eq!(p.fixed_variable_indexes[k0].len(), 2);
        assert_eq!(p.fixed_variable_indexes[k1].len(), 1);

        p.unfix_variable(k0);
        assert!(!p.fixed_variable_indexes.contains_key(k0));
        assert!(p.fixed_variable_indexes.contains_key(k1));
    }

    /// Regrouping rows for elimination must not change the problem.
    ///
    /// Row order is a labelling: permuting it permutes `r` and `J`'s rows
    /// together, so the cost — a sum of squares — is invariant.
    #[test]
    fn test_group_rows_for_elimination_preserves_cost() -> TestResult {
        use crate::linearizer::AssemblyWorkspace;

        let mut p = Problem::new(JacobianMode::Sparse);
        let a = p.add_variable(ManifoldType::SE2, dvector![0.1, 0.2, 0.05]);
        let b = p.add_variable(ManifoldType::SE2, dvector![1.0, 0.1, 0.02]);
        let p0 = p.add_variable(ManifoldType::RN, dvector![0.3, 0.4, 0.5]);
        let p1 = p.add_variable(ManifoldType::RN, dvector![0.6, 0.7, 0.8]);

        // Camera-major insertion: each landmark's rows end up scattered.
        for (x, y) in [(a, p0), (a, p1), (b, p0), (b, p1)] {
            p.add_residual_block(
                &[x, y],
                Box::new(crate::factors::pose::BetweenFactor::new(
                    apex_manifolds::se2::SE2::from_xy_angle(0.5, 0.1, 0.01),
                )),
                None,
            );
        }
        p.mark_for_elimination(p0);
        p.mark_for_elimination(p1);

        let variables = p.variables.clone();
        let mut ws = AssemblyWorkspace::build(&p);
        let (_, cost_before) =
            p.compute_residual_and_cost_sparse_with_workspace(&variables, &mut ws)?;

        assert!(
            p.group_rows_for_elimination(),
            "camera-major rows must move"
        );

        let mut ws = AssemblyWorkspace::build(&p);
        let (_, cost_after) =
            p.compute_residual_and_cost_sparse_with_workspace(&variables, &mut ws)?;
        assert!(
            (cost_before - cost_after).abs() < 1e-12,
            "cost changed: {cost_before} -> {cost_after}"
        );

        // Each eliminated variable's rows are now one contiguous range.
        for landmark in [p0, p1] {
            let mut rows: Vec<usize> = p
                .residual_blocks()
                .values()
                .filter(|b| b.variable_keys.contains(&landmark))
                .flat_map(|b| {
                    b.residual_row_start_idx..b.residual_row_start_idx + b.factor.residual_dim()
                })
                .collect();
            rows.sort_unstable();
            let contiguous = rows.windows(2).all(|w| w[1] == w[0] + 1);
            assert!(
                contiguous,
                "rows for {landmark:?} are not contiguous: {rows:?}"
            );
        }

        // Idempotent.
        assert!(
            !p.group_rows_for_elimination(),
            "a second call must be a no-op"
        );
        Ok(())
    }

    /// Mutating the problem after grouping must not corrupt the row layout.
    ///
    /// A block added post-grouping lands at the end, which can violate the
    /// unchunked-first invariant the chunk sweep depends on. Regrouping must
    /// restore a compact, contiguous layout with the total preserved — this is
    /// what every LM entry relies on when a problem is solved, mutated, and
    /// solved again.
    #[test]
    fn test_group_rows_after_mutation_restores_layout() -> TestResult {
        use crate::linearizer::AssemblyWorkspace;

        let mut p = Problem::new(JacobianMode::Sparse);
        let a = p.add_variable(ManifoldType::SE2, dvector![0.1, 0.2, 0.05]);
        let b = p.add_variable(ManifoldType::SE2, dvector![1.0, 0.1, 0.02]);
        let p0 = p.add_variable(ManifoldType::RN, dvector![0.3, 0.4, 0.5]);
        let p1 = p.add_variable(ManifoldType::RN, dvector![0.6, 0.7, 0.8]);

        // Camera-major insertion: each landmark's rows end up scattered.
        for (x, y) in [(a, p0), (a, p1), (b, p0), (b, p1)] {
            p.add_residual_block(
                &[x, y],
                Box::new(crate::factors::pose::BetweenFactor::new(
                    apex_manifolds::se2::SE2::from_xy_angle(0.5, 0.1, 0.01),
                )),
                None,
            );
        }
        p.mark_for_elimination(p0);
        p.mark_for_elimination(p1);
        assert!(
            p.group_rows_for_elimination(),
            "camera-major rows must move"
        );
        assert!(
            !p.group_rows_for_elimination(),
            "grouped layout must be stable"
        );

        // Mutate after grouping: an unchunked block appended at the end.
        p.add_residual_block(
            &[a, b],
            Box::new(crate::factors::pose::BetweenFactor::new(
                apex_manifolds::se2::SE2::from_xy_angle(0.2, 0.0, 0.0),
            )),
            None,
        );
        assert!(
            p.group_rows_for_elimination(),
            "the appended unchunked block must move ahead of the chunks"
        );

        // Layout is compact: every residual row is covered exactly once.
        let total = p.total_residual_dimension;
        let mut covered = vec![false; total];
        for block in p.residual_blocks().values() {
            let start = block.residual_row_start_idx;
            for (row, slot) in covered
                .iter_mut()
                .enumerate()
                .skip(start)
                .take(block.factor.residual_dim())
            {
                assert!(row < total, "row {row} out of range (total {total})");
                assert!(!*slot, "row {row} covered twice");
                *slot = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "layout has gaps: {covered:?}");

        // Unchunked rows come first.
        let touches_eliminated = |keys: &[VarKey]| keys.contains(&p0) || keys.contains(&p1);
        let first_chunk_row = p
            .residual_blocks()
            .values()
            .filter(|b| touches_eliminated(&b.variable_keys))
            .map(|b| b.residual_row_start_idx)
            .min()
            .ok_or_else(|| CoreError::Variable("eliminated block missing".to_string()))?;
        for block in p.residual_blocks().values() {
            if !touches_eliminated(&block.variable_keys) {
                assert!(
                    block.residual_row_start_idx < first_chunk_row,
                    "unchunked block at {} is inside the chunked region (starts at {first_chunk_row})",
                    block.residual_row_start_idx
                );
            }
        }

        // Cost invariant holds across the regroup.
        let variables = p.variables.clone();
        let mut ws = AssemblyWorkspace::build(&p);
        let (_, cost) = p.compute_residual_and_cost_sparse_with_workspace(&variables, &mut ws)?;
        assert!(cost.is_finite(), "cost must stay finite, got {cost}");
        assert!(!p.group_rows_for_elimination(), "layout must be stable now");
        Ok(())
    }

    #[test]
    fn test_variable_bounds_set_remove() {
        let mut p = Problem::new(JacobianMode::Sparse);
        let k0 = p.add_variable(ManifoldType::SE2, dvector![0.0, 0.0, 0.0]);
        let k1 = p.add_variable(ManifoldType::SE2, dvector![0.0, 0.0, 0.0]);

        p.set_variable_bounds(k0, 0, -1.0, 1.0);
        p.set_variable_bounds(k0, 1, -2.0, 2.0);
        p.set_variable_bounds(k1, 0, 0.0, 5.0);

        assert_eq!(p.variable_bounds[k0].len(), 2);
        assert_eq!(p.variable_bounds[k1].len(), 1);

        p.remove_variable_bounds(k0);
        assert!(!p.variable_bounds.contains_key(k0));
        assert!(p.variable_bounds.contains_key(k1));
    }

    #[test]
    fn test_set_variable_bounds_invalid_order() {
        let mut p = Problem::new(JacobianMode::Sparse);
        let k = p.add_variable(ManifoldType::SE2, dvector![0.0, 0.0, 0.0]);

        // An inverted range is rejected, not warned about and dropped: the
        // caller could not previously tell the two apart.
        let Err(err) = p.try_set_variable_bounds(k, 0, 5.0, 1.0) else {
            panic!("inverted range must be rejected");
        };
        assert!(err.to_string().contains("exceeds upper bound"), "{err}");
        assert!(!p.variable_bounds.contains_key(k));
    }

    /// A component index beyond the variable's DOF can never be applied, so it
    /// must be rejected rather than stored and silently ignored forever.
    #[test]
    fn test_variable_constraints_reject_out_of_range_index() {
        let mut p = Problem::new(JacobianMode::Sparse);
        let k = p.add_variable(ManifoldType::SE2, dvector![0.0, 0.0, 0.0]); // 3 DOF

        let Err(err) = p.try_fix_variable(k, 3) else {
            panic!("index 3 is out of range for a 3-DOF variable");
        };
        assert!(err.to_string().contains("3-DOF"), "{err}");

        let Err(err) = p.try_set_variable_bounds(k, 7, -1.0, 1.0) else {
            panic!("index 7 is out of range for a 3-DOF variable");
        };
        assert!(err.to_string().contains("3-DOF"), "{err}");

        assert!(!p.fixed_variable_indexes.contains_key(k));
        assert!(!p.variable_bounds.contains_key(k));
    }

    #[test]
    fn test_problem_default_equals_new_sparse() {
        let d = Problem::default();
        let n = Problem::new(JacobianMode::Sparse);
        assert_eq!(d.jacobian_mode, n.jacobian_mode);
        assert_eq!(d.num_residual_blocks(), 0);
    }

    #[test]
    fn test_compute_residual_sparse_smoke() -> TestResult {
        let (problem, _) = create_se2_test_problem()?;
        let residual = problem.compute_residual_sparse(&problem.variables)?;
        let norm_sq: f64 = (0..residual.nrows())
            .map(|i| residual[(i, 0)].powi(2))
            .sum();
        assert!(norm_sq >= 0.0);
        assert_eq!(residual.nrows(), problem.total_residual_dimension);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Robust cost
    // -------------------------------------------------------------------------

    /// `r = x`, a single scalar residual — lets a test dial ‖r‖ exactly.
    struct IdentityFactor;

    impl crate::factors::Factor for IdentityFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = params[0][0];
            if let Some(mut jac) = jacobian {
                use faer::prelude::ReborrowMut;
                *jac.rb_mut().get_mut(0, 0) = 1.0;
            }
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 1)
        }
    }

    /// The cost of a robust block must be `0.5·ρ(s)`, not `0.5·‖r̃‖²` computed
    /// from the Triggs-corrected residual.
    ///
    /// Huber with δ = 1 at ‖r‖ = 2 (s = 4): ρ(s) = 2δ√s − δ² = 3, so the cost is
    /// 1.5. The old code reported 1.0 — 33% low.
    #[test]
    fn robust_cost_is_half_rho_not_half_corrected_residual_squared() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let x = problem.add_variable(ManifoldType::RN, dvector![2.0]);
        problem.add_residual_block(
            &[x],
            Box::new(IdentityFactor),
            Some(Box::new(HuberLoss::new(1.0)?)),
        );

        let (residual, cost) = problem.compute_residual_and_cost_sparse(&problem.variables)?;

        assert!(
            (cost - 1.5).abs() < 1e-12,
            "robust cost should be 0.5·ρ(4) = 1.5, got {cost}"
        );

        // And confirm the old formula really does differ, so this test cannot
        // silently start passing for the wrong reason.
        let from_corrected_residual = 0.5 * residual.squared_norm_l2();
        assert!(
            (from_corrected_residual - 1.0).abs() < 1e-12,
            "corrected-residual cost should be 1.0, got {from_corrected_residual}"
        );
        Ok(())
    }

    /// Without a loss function the two definitions coincide, so non-robust
    /// problems must be completely unaffected by the change.
    #[test]
    fn cost_without_loss_is_half_squared_norm() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let x = problem.add_variable(ManifoldType::RN, dvector![3.0]);
        problem.add_residual_block(&[x], Box::new(IdentityFactor), None);

        let (residual, cost) = problem.compute_residual_and_cost_sparse(&problem.variables)?;
        assert!(
            (cost - 4.5).abs() < 1e-12,
            "expected 0.5·3² = 4.5, got {cost}"
        );
        assert!((cost - 0.5 * residual.squared_norm_l2()).abs() < 1e-15);
        Ok(())
    }

    /// The cost must equal `0.5·ρ(s)` read straight off the loss function, for
    /// every loss, not just Huber.
    #[test]
    fn robust_cost_matches_loss_function_rho() -> TestResult {
        use crate::core::loss_functions::{CauchyLoss, LossFunction};

        let value = 2.5;
        let mut problem = Problem::new(JacobianMode::Sparse);
        let x = problem.add_variable(ManifoldType::RN, dvector![value]);
        problem.add_residual_block(
            &[x],
            Box::new(IdentityFactor),
            Some(Box::new(CauchyLoss::new(1.0)?)),
        );

        let (_, cost) = problem.compute_residual_and_cost_sparse(&problem.variables)?;
        let expected = 0.5 * CauchyLoss::new(1.0)?.evaluate(value * value)[0];
        assert!(
            (cost - expected).abs() < 1e-12,
            "expected 0.5·ρ(s) = {expected}, got {cost}"
        );
        Ok(())
    }

    #[test]
    fn test_variable_covariance_lifecycle() -> TestResult {
        use faer::Mat;
        let mut p = Problem::new(JacobianMode::Sparse);
        let k = p.add_variable(ManifoldType::SE2, dvector![0.0, 0.0, 0.0]);

        assert!(p.variables[k].covariance().is_none());
        p.variables[k].set_covariance(Mat::identity(3, 3));
        let cov = p.variables[k].covariance().ok_or("no cov")?;
        assert_eq!(cov.nrows(), 3);
        p.variables[k].clear_covariance();
        assert!(p.variables[k].covariance().is_none());
        Ok(())
    }

    #[test]
    fn test_fixed_indices_stored_correctly() {
        let mut p = Problem::new(JacobianMode::Sparse);
        let k = p.add_variable(ManifoldType::SE2, dvector![0.0, 0.0, 0.0]);
        p.fix_variable(k, 0);
        p.fix_variable(k, 2);
        assert_eq!(p.fixed_variable_indexes[k].len(), 2);
        assert!(p.fixed_variable_indexes[k].contains(&0));
        assert!(p.fixed_variable_indexes[k].contains(&2));
    }

    #[test]
    fn test_log_residual_to_file() -> TestResult {
        let p = Problem::new(JacobianMode::Sparse);
        let res = nalgebra::dvector![1.0, 2.0, 3.0];
        let path = std::env::temp_dir().join("apex_test_residual.txt");
        p.log_residual_to_file(&res, path.to_str().ok_or("bad path")?)?;
        assert!(path.exists());
        Ok(())
    }

    #[test]
    fn test_log_variables_to_file() -> TestResult {
        let mut p = Problem::new(JacobianMode::Sparse);
        p.add_variable(ManifoldType::SE2, dvector![1.0, 2.0, 0.3]);
        let vars = p.variables.clone();
        let path = std::env::temp_dir().join("apex_test_variables.txt");
        p.log_variables_to_file(&vars, path.to_str().ok_or("bad path")?)?;
        assert!(path.exists());
        Ok(())
    }

    #[test]
    fn test_log_sparse_jacobian_to_file() -> TestResult {
        use faer::sparse::SparseColMat;
        let p = Problem::new(JacobianMode::Sparse);
        let triplets = vec![faer::sparse::Triplet::new(0usize, 0usize, 1.0f64)];
        let jac =
            SparseColMat::try_new_from_triplets(1, 1, &triplets).map_err(|e| format!("{e:?}"))?;
        let path = std::env::temp_dir().join("apex_test_jacobian.txt");
        p.log_sparse_jacobian_to_file(&jac, path.to_str().ok_or("bad path")?)?;
        assert!(path.exists());
        Ok(())
    }

    #[test]
    fn test_apply_tangent_step_se2() -> TestResult {
        let mut p = Problem::new(JacobianMode::Sparse);
        let k = p.add_variable(ManifoldType::SE2, dvector![0.0, 0.0, 0.0]);
        p.variables[k].apply_tangent_step(&[1.0, 2.0, 3.0]);
        assert_eq!(p.variables[k].dof(), 3);
        Ok(())
    }

    #[test]
    fn test_variable_rn_values() -> TestResult {
        let mut p = Problem::new(JacobianMode::Sparse);
        let k = p.add_variable(ManifoldType::RN, dvector![5.0, 6.0]);
        let vec = p.variables[k].to_dvector();
        assert!((vec[0] - 5.0).abs() < 1e-10);
        assert!((vec[1] - 6.0).abs() < 1e-10);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Registration-time factor validation
    // -------------------------------------------------------------------------

    mod validation {
        use super::*;
        use crate::factors::visual::projection::ProjectionFactor;
        use crate::factors::{BundleAdjustment, Factor};
        use apex_camera_models::PinholeCamera;
        use nalgebra::Matrix2xX;
        use rn::Rn;

        fn ba_factor(n_landmarks: usize) -> Box<dyn Factor + Send + Sync> {
            let observations = Matrix2xX::from_fn(n_landmarks, |r, c| (r + c) as f64);
            Box::new(ProjectionFactor::<PinholeCamera, BundleAdjustment>::new(
                observations,
                PinholeCamera::from([500.0, 500.0, 320.0, 240.0]),
            ))
        }

        fn pose_and_landmarks(problem: &mut Problem, n_landmark_params: usize) -> (VarKey, VarKey) {
            let pose = problem.add_variable(
                ManifoldType::SE3,
                dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            );
            let landmarks = problem.add_variable(
                ManifoldType::RN,
                DVector::from_element(n_landmark_params, 1.0),
            );
            (pose, landmarks)
        }

        #[test]
        fn try_add_accepts_matching_landmark_variable() -> TestResult {
            let mut problem = Problem::new(JacobianMode::Sparse);
            let (pose, landmarks) = pose_and_landmarks(&mut problem, 9);
            let key = problem.try_add_residual_block(&[pose, landmarks], ba_factor(3), None)?;
            assert_eq!(problem.residual_blocks().len(), 1);
            assert_eq!(problem.total_residual_dimension(), 6);
            let _ = key;
            Ok(())
        }

        #[test]
        fn try_add_rejects_mismatched_landmark_variable() -> TestResult {
            let mut problem = Problem::new(JacobianMode::Sparse);
            let (pose, landmarks) = pose_and_landmarks(&mut problem, 12);
            let err = problem
                .try_add_residual_block(&[pose, landmarks], ba_factor(3), None)
                .err()
                .ok_or("mismatched landmark count must be rejected")?;
            assert!(matches!(err, CoreError::DimensionMismatch(_)), "{err}");
            Ok(())
        }

        #[test]
        fn add_residual_block_panics_on_invalid_registration() {
            let mut problem = Problem::new(JacobianMode::Sparse);
            let (pose, landmarks) = pose_and_landmarks(&mut problem, 12);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                problem.add_residual_block(&[pose, landmarks], ba_factor(3), None);
            }));
            assert!(result.is_err(), "registration panic expected");
        }

        #[test]
        fn try_add_rejects_unknown_variable_key() -> TestResult {
            let mut problem = Problem::new(JacobianMode::Sparse);
            let (pose, _landmarks) = pose_and_landmarks(&mut problem, 9);

            // A key from a different, larger slot map: its slot index is out of
            // bounds for the problem's store, so it can never resolve.
            let mut other_store: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
            let mut foreign_key = None;
            for i in 0..5 {
                foreign_key =
                    Some(other_store.insert(Box::new(Variable::new(Rn::new(dvector![i as f64])))));
            }
            let foreign_key = foreign_key.ok_or("inserted five keys")?;

            let err = problem
                .try_add_residual_block(&[pose, foreign_key], ba_factor(3), None)
                .err()
                .ok_or("unknown variable key must be rejected")?;
            assert!(matches!(err, CoreError::Variable(_)), "{err}");
            Ok(())
        }
    }
}
