use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{Error, Write},
};

use faer::{Mat, sparse::SparseColMat};
use nalgebra::DVector;
use rayon::prelude::*;
use slotmap::{SecondaryMap, SlotMap};
use tracing::warn;

use crate::{
    core::CoreResult,
    core::{
        FactorKey, VarKey,
        corrector::Corrector,
        loss_functions::LossFunction,
        residual_block::ResidualBlock,
        variable::{ManifoldVariable, Variable},
    },
    factors::Factor,
    linalg::{JacobianMode, LinearSolver, SparseMode, extract_variable_covariances},
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

    /// Mark a variable as a Schur complement landmark (eliminated block).
    ///
    /// Call this for every landmark/point variable when using a Schur complement
    /// solver. Variables not marked here are treated as camera-block variables.
    pub fn mark_as_schur_landmark(&mut self, key: VarKey) {
        self.schur_landmark_keys.insert(key);
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
        factor: Box<dyn Factor + Send>,
        loss_func: Option<Box<dyn LossFunction + Send>>,
    ) -> FactorKey {
        let new_residual_dimension = factor.residual_dim();
        let row_start = self.total_residual_dimension;
        let fk = self.residual_blocks.insert_with_key(|fk| {
            ResidualBlock::new(fk, row_start, variable_keys, factor, loss_func)
        });
        self.total_residual_dimension += new_residual_dimension;
        fk
    }

    pub fn remove_residual_block(&mut self, block_id: FactorKey) -> Option<ResidualBlock> {
        if let Some(block) = self.residual_blocks.remove(block_id) {
            self.total_residual_dimension -= block.factor.residual_dim();
            Some(block)
        } else {
            None
        }
    }

    pub fn fix_variable(&mut self, var_key: VarKey, idx: usize) {
        if let Some(set) = self.fixed_variable_indexes.get_mut(var_key) {
            set.insert(idx);
        } else {
            let mut s = HashSet::new();
            s.insert(idx);
            self.fixed_variable_indexes.insert(var_key, s);
        }
    }

    pub fn unfix_variable(&mut self, var_key: VarKey) {
        self.fixed_variable_indexes.remove(var_key);
    }

    pub fn set_variable_bounds(
        &mut self,
        var_key: VarKey,
        idx: usize,
        lower_bound: f64,
        upper_bound: f64,
    ) {
        if lower_bound > upper_bound {
            warn!("lower bound is larger than upper bound");
        } else if let Some(map) = self.variable_bounds.get_mut(var_key) {
            map.insert(idx, (lower_bound, upper_bound));
        } else {
            self.variable_bounds
                .insert(var_key, HashMap::from([(idx, (lower_bound, upper_bound))]));
        }
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

    pub(crate) fn residual_blocks(&self) -> &SlotMap<FactorKey, ResidualBlock> {
        &self.residual_blocks
    }

    /// Compute only the residual vector (no Jacobian) for the given variable values.
    pub fn compute_residual_sparse(
        &self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    ) -> CoreResult<Mat<f64>> {
        use crate::linearizer::split_by_row_offsets_mut;

        let mut blocks: Vec<&crate::core::residual_block::ResidualBlock> =
            self.residual_blocks.values().collect();
        blocks.sort_by_key(|b| b.residual_row_start_idx);

        let mut residual_buf = vec![0.0f64; self.total_residual_dimension];
        let offsets_lens: Vec<(usize, usize)> = blocks
            .iter()
            .map(|b| (b.residual_row_start_idx, b.factor.residual_dim()))
            .collect();
        let residual_slices = split_by_row_offsets_mut(&mut residual_buf, &offsets_lens);

        let results: Vec<CoreResult<()>> = residual_slices
            .into_par_iter()
            .zip(blocks.par_iter())
            .map(|(slice, block)| self.compute_residual_block(block, variables, slice))
            .collect();
        results.into_iter().collect::<CoreResult<Vec<_>>>()?;

        let n = self.total_residual_dimension;
        Ok(Mat::from_fn(n, 1, |i, _| residual_buf[i]))
    }

    /// Compute residuals and sparse Jacobian.
    pub fn compute_residual_and_jacobian_sparse(
        &self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
        symbolic_structure: &SymbolicStructure,
    ) -> CoreResult<(Mat<f64>, SparseColMat<usize, f64>)> {
        Ok(crate::linearizer::cpu::sparse::assemble_sparse(
            self,
            variables,
            variable_index_map,
            symbolic_structure,
        )?)
    }

    /// Compute residuals and dense Jacobian.
    pub fn compute_residual_and_jacobian_dense(
        &self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
        total_dof: usize,
    ) -> CoreResult<(Mat<f64>, Mat<f64>)> {
        Ok(crate::linearizer::cpu::dense::assemble_dense(
            self,
            variables,
            variable_index_map,
            total_dof,
        )?)
    }

    fn compute_residual_block(
        &self,
        residual_block: &ResidualBlock,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        residual_slice: &mut [f64],
    ) -> CoreResult<()> {
        let mut param_slices: smallvec::SmallVec<[&[f64]; 8]> = smallvec::SmallVec::new();
        for &k in &residual_block.variable_keys {
            if let Some(v) = variables.get(k) {
                param_slices.push(v.as_param_slice());
            }
        }

        residual_block
            .factor
            .linearize(&param_slices, residual_slice, None);

        if let Some(loss_func) = &residual_block.loss_func {
            let squared_norm: f64 = residual_slice.iter().map(|x| x * x).sum();
            let corrector = Corrector::new(loss_func.as_ref(), squared_norm);
            corrector.correct_residual_in_place(residual_slice);
        }

        Ok(())
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

    pub fn compute_and_set_covariances(
        &self,
        linear_solver: &mut Box<dyn LinearSolver<SparseMode>>,
        variables: &mut SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
    ) -> Option<SecondaryMap<VarKey, Mat<f64>>> {
        linear_solver.compute_covariance_matrix()?;
        let full_cov = linear_solver.get_covariance_matrix()?.clone();
        let per_var = extract_variable_covariances(&full_cov, variables, variable_index_map);
        for (key, cov) in &per_var {
            if let Some(var) = variables.get_mut(key) {
                var.set_covariance(cov.clone());
            }
        }
        Some(per_var)
    }

    pub fn compute_and_set_covariances_generic<M: crate::linalg::LinearizationMode>(
        &self,
        linear_solver: &mut dyn crate::linalg::LinearSolver<M>,
        variables: &mut SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
    ) -> Option<SecondaryMap<VarKey, Mat<f64>>> {
        linear_solver.compute_covariance_matrix()?;
        let full_cov = linear_solver.get_covariance_matrix()?.clone();
        let per_var = extract_variable_covariances(&full_cov, variables, variable_index_map);
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
    use crate::factors::{BetweenFactor, PriorFactor};
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
            Box::new(PriorFactor {
                data: dvector![0.0, 0.0, 0.0],
            }),
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
            Box::new(PriorFactor {
                data: dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            }),
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
            Box::new(PriorFactor {
                data: dvector![0.0, 0.0, 0.0],
            }),
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
        p.set_variable_bounds(k, 0, 5.0, 1.0);
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
}
