//! # Explicit Dense Schur Complement Solver
//!
//! Forms the reduced camera system `S = H_kk − H_ke·H_ee⁻¹·H_keᵀ` explicitly
//! over a **dense** Hessian and factorizes it with dense Cholesky. Equivalent
//! to Ceres's `DENSE_SCHUR`.
//!
//! Dense storage makes this simpler than the sparse explicit solver: there is
//! no sparsity pattern to preserve, so extracting `H_kk`/`H_ke`, gathering
//! `H_ee`'s diagonal blocks, and forming `S` are all direct index-and-copy
//! operations rather than a symbolic-structure walk. Chunk-wise elimination
//! (sweeping `J`'s rows to avoid ever forming `JᵀJ`) buys nothing here: the
//! whole point of chunking is avoiding sparse fill-in on a Hessian too large
//! to store, and dense mode's target size (< ~500 DOF) means `H` is already
//! cheap to hold in full. Ceres's own `DENSE_SCHUR` builds `E`/`F`/`S` as
//! plain dense BLAS blocks for the same reason.
//!
//! Targets small-to-medium problems (`JacobianMode::Dense`, < ~500 DOF) —
//! see [`ExplicitSparseSchur`](crate::linalg::sparse::schur::ExplicitSparseSchur)
//! for larger sparse problems.
//!
//! ## Usage Example
//!
//! ```no_run
//! # use apex_solver::linalg::ExplicitDenseSchur;
//! # use apex_solver::linalg::StructureAware;
//! # use apex_solver::core::VarKey;
//! # use apex_solver::core::variable::ManifoldVariable;
//! # use slotmap::{SlotMap, SecondaryMap};
//! # use std::collections::HashSet;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
//! # let variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
//! # let landmark_keys: HashSet<VarKey> = HashSet::new();
//! let mut solver = ExplicitDenseSchur::new();
//! solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
//! # Ok(())
//! # }
//! ```

use faer::Mat;

use crate::core::VarKey;
use crate::core::variable::ManifoldVariable;
use crate::linalg::dense::cholesky::solve_spd;
use crate::linalg::dense::schur::gather::{gather_dense, verify_block_diagonal_dense};
use crate::linalg::schur::{
    BlockSpan, EliminatedBlocks, SchurOrdering, SchurPartition, effective_landmark_keys,
};
use crate::linalg::{Damping, DenseMode, LinAlgError, LinAlgResult, LinearSolver, StructureAware};
use slotmap::{SecondaryMap, SlotMap};

/// Explicit Schur Complement Solver for a dense Hessian. Equivalent to
/// Ceres's `DENSE_SCHUR`.
#[derive(Debug, Clone)]
pub struct ExplicitDenseSchur {
    partition: Option<SchurPartition>,
    /// `H_ee⁻¹` per eliminated block, gathered and inverted once per solve.
    eliminated: EliminatedBlocks,
    ordering: SchurOrdering,

    hessian: Option<Mat<f64>>,
    gradient: Option<Mat<f64>>,
}

impl ExplicitDenseSchur {
    pub fn new() -> Self {
        Self {
            partition: None,
            eliminated: EliminatedBlocks::default(),
            ordering: SchurOrdering::default(),
            hessian: None,
            gradient: None,
        }
    }

    /// Set the automatic group-classification ordering (see [`SchurOrdering`]).
    pub fn with_ordering(mut self, ordering: SchurOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    /// The variable partition, once `initialize_structure` has run.
    pub fn partition(&self) -> Option<&SchurPartition> {
        self.partition.as_ref()
    }

    fn require_partition(&self) -> LinAlgResult<&SchurPartition> {
        self.partition.as_ref().ok_or_else(|| {
            LinAlgError::InvalidInput(
                "Block structure not built. Call initialize_structure() first.".to_string(),
            )
        })
    }

    /// Flattened kept/eliminated global-column lists, in the same order as
    /// [`SchurPartition::kept_local`]/[`SchurPartition::eliminated_offset`]
    /// number them — so index `i` here is local index `i` in the reduced
    /// system (kept) or the eliminated-local space (eliminated).
    fn column_orderings(partition: &SchurPartition) -> (Vec<usize>, Vec<usize>) {
        let kept: Vec<usize> = partition
            .kept_blocks()
            .iter()
            .flat_map(|b| (0..b.dof).map(move |o| b.col_start + o))
            .collect();
        let eliminated: Vec<usize> = partition
            .eliminated_blocks()
            .iter()
            .flat_map(|b| (0..b.dof).map(move |o| b.col_start + o))
            .collect();
        (kept, eliminated)
    }

    /// Eliminate, solve the reduced system, and back-substitute.
    fn solve_with_hessian(
        &mut self,
        hessian: &Mat<f64>,
        neg_gradient: &Mat<f64>,
        damping: Option<&Damping>,
    ) -> LinAlgResult<Mat<f64>> {
        let partition = self.require_partition()?;
        verify_block_diagonal_dense(partition, hessian)?;

        let kept_dof = partition.kept_dof();
        let eliminated_dof = partition.eliminated_dof();
        let (kept_global, elim_global) = Self::column_orderings(partition);

        // H_kk, H_ke as plain dense sub-blocks.
        let mut h_kk = Mat::<f64>::zeros(kept_dof, kept_dof);
        for (lr, &gr) in kept_global.iter().enumerate() {
            for (lc, &gc) in kept_global.iter().enumerate() {
                h_kk[(lr, lc)] = hessian[(gr, gc)];
            }
        }
        let mut h_ke = Mat::<f64>::zeros(kept_dof, eliminated_dof);
        for (lr, &gr) in kept_global.iter().enumerate() {
            for (lc, &gc) in elim_global.iter().enumerate() {
                h_ke[(lr, lc)] = hessian[(gr, gc)];
            }
        }

        let mut g_k = Mat::<f64>::zeros(kept_dof, 1);
        for (lr, &gr) in kept_global.iter().enumerate() {
            g_k[(lr, 0)] = neg_gradient[(gr, 0)];
        }
        let mut g_e = Mat::<f64>::zeros(eliminated_dof, 1);
        for (lr, &gr) in elim_global.iter().enumerate() {
            g_e[(lr, 0)] = neg_gradient[(gr, 0)];
        }

        // λ·D is applied to *both* sides before elimination, matching the
        // sparse solver: damping the reduced system afterwards would not be
        // the same problem.
        if let Some(damping) = damping {
            for i in 0..kept_dof {
                h_kk[(i, i)] += damping.diagonal_term(h_kk[(i, i)]);
            }
        }

        // `self.eliminated` is moved out for the duration of the gather+invert
        // so `partition` (borrowed from `self`) and `self.eliminated` are
        // never borrowed simultaneously, mirroring `ExplicitSparseSchur`.
        let mut eliminated = std::mem::take(&mut self.eliminated);
        let prepared = (|| -> LinAlgResult<()> {
            let partition = self.require_partition()?;
            gather_dense(&mut eliminated, partition, hessian);
            if let Some(damping) = damping {
                eliminated.damp(damping);
            }
            eliminated.invert_in_place(partition)
        })();
        self.eliminated = eliminated;
        prepared?;

        let partition = self.require_partition()?;

        // H_ee⁻¹ as one block-diagonal dense matrix, so S and g_reduced fall
        // out of plain dense matmuls.
        let mut h_ee_inv = Mat::<f64>::zeros(eliminated_dof, eliminated_dof);
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let dof = block.dof;
            let base = partition.eliminated_offset(block_idx);
            let inv = self.eliminated.block(block_idx);
            for c in 0..dof {
                for r in 0..dof {
                    h_ee_inv[(base + r, base + c)] = inv[c * dof + r];
                }
            }
        }

        let coupling = &h_ke * &h_ee_inv;
        let s = &h_kk - &coupling * h_ke.transpose();
        let g_reduced = &g_k - &coupling * &g_e;

        let delta_k = solve_spd(
            &s,
            &g_reduced,
            "Dense Schur complement factorization failed",
        )?;

        // δ_e = H_ee⁻¹·(g_e − H_keᵀ·δ_k)
        let hke_t_delta = h_ke.transpose() * &delta_k;
        let mut delta_e = Mat::<f64>::zeros(eliminated_dof, 1);
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let dof = block.dof;
            let base = partition.eliminated_offset(block_idx);
            let inv = self.eliminated.block(block_idx);
            for r in 0..dof {
                let mut acc = 0.0;
                for c in 0..dof {
                    let rhs = g_e[(base + c, 0)] - hke_t_delta[(base + c, 0)];
                    acc += inv[c * dof + r] * rhs;
                }
                delta_e[(base + r, 0)] = acc;
            }
        }

        let mut delta = Mat::<f64>::zeros(partition.total_dof(), 1);
        for (lr, &gr) in kept_global.iter().enumerate() {
            delta[(gr, 0)] = delta_k[(lr, 0)];
        }
        for (lr, &gr) in elim_global.iter().enumerate() {
            delta[(gr, 0)] = delta_e[(lr, 0)];
        }
        Ok(delta)
    }
}

impl Default for ExplicitDenseSchur {
    fn default() -> Self {
        Self::new()
    }
}

impl StructureAware for ExplicitDenseSchur {
    fn initialize_structure(
        &mut self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
        schur_landmark_keys: &std::collections::HashSet<VarKey>,
    ) -> LinAlgResult<()> {
        let effective_keys =
            effective_landmark_keys(variables, schur_landmark_keys, &self.ordering);

        let mut kept = Vec::new();
        let mut eliminated = Vec::new();
        for (key, variable) in variables {
            let col_start = *variable_index_map.get(key).ok_or_else(|| {
                LinAlgError::InvalidInput(format!("VarKey {:?} not found in index map", key))
            })?;
            let span = BlockSpan {
                key,
                col_start,
                // Free columns only — fixed tangent coordinates own no
                // column in the linear solve (ISSUE-0003).
                dof: variable.free_dof(),
            };
            if effective_keys.contains(&key) {
                eliminated.push(span);
            } else {
                kept.push(span);
            }
        }

        let partition = SchurPartition::new(kept, eliminated)?;
        self.eliminated = EliminatedBlocks::new(&partition);
        self.partition = Some(partition);
        Ok(())
    }
}

impl LinearSolver<DenseMode> for ExplicitDenseSchur {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        self.require_partition()?;
        let hessian = jacobian.transpose() * jacobian;
        let gradient = jacobian.transpose() * residuals;
        let mut neg_gradient = Mat::<f64>::zeros(gradient.nrows(), 1);
        for i in 0..gradient.nrows() {
            neg_gradient[(i, 0)] = -gradient[(i, 0)];
        }

        let result = self.solve_with_hessian(&hessian, &neg_gradient, None);
        if result.is_ok() {
            self.hessian = Some(hessian);
            self.gradient = Some(gradient);
        }
        result
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &Mat<f64>,
        damping: &Damping,
    ) -> LinAlgResult<Mat<f64>> {
        self.require_partition()?;
        let hessian = jacobian.transpose() * jacobian;
        let gradient = jacobian.transpose() * residuals;
        let mut neg_gradient = Mat::<f64>::zeros(gradient.nrows(), 1);
        for i in 0..gradient.nrows() {
            neg_gradient[(i, 0)] = -gradient[(i, 0)];
        }

        let result = self.solve_with_hessian(&hessian, &neg_gradient, Some(damping));
        if result.is_ok() {
            self.hessian = Some(hessian);
            self.gradient = Some(gradient);
        }
        result
    }

    fn hessian_vec_product(&self, v: &Mat<f64>) -> Option<Mat<f64>> {
        Some(
            <DenseMode as crate::linearizer::AssemblyBackend>::hessian_vec_product(
                self.hessian.as_ref()?,
                v,
            ),
        )
    }

    fn get_hessian(&self) -> Option<&Mat<f64>> {
        self.hessian.as_ref()
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::VarKey;
    use crate::core::variable::Variable;
    use apex_manifolds::{LieGroup, rn, se3};
    use nalgebra::DVector;
    use slotmap::{SecondaryMap, SlotMap};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    type TestSetup = (
        SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        SecondaryMap<VarKey, usize>,
        Mat<f64>,
        Mat<f64>,
        std::collections::HashSet<VarKey>,
    );

    /// Same 2-camera + 3-landmark shape as the sparse Schur unit tests, built
    /// as a dense Jacobian directly.
    fn create_schur_test_setup() -> Result<TestSetup, Box<dyn std::error::Error>> {
        let se3_id = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let pt_zero = DVector::from_vec(vec![0.0, 0.0, 0.0]);

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let cam0 = variables.insert(Box::new(Variable::new(se3::SE3::from_param_slice(
            se3_id.as_slice(),
        ))));
        let cam1 = variables.insert(Box::new(Variable::new(se3::SE3::from_param_slice(
            se3_id.as_slice(),
        ))));
        let pt0 = variables.insert(Box::new(Variable::new(rn::Rn::new(pt_zero.clone()))));
        let pt1 = variables.insert(Box::new(Variable::new(rn::Rn::new(pt_zero.clone()))));
        let pt2 = variables.insert(Box::new(Variable::new(rn::Rn::new(pt_zero.clone()))));

        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(cam0, 0);
        variable_index_map.insert(cam1, 6);
        variable_index_map.insert(pt0, 12);
        variable_index_map.insert(pt1, 15);
        variable_index_map.insert(pt2, 18);

        let cam_cols = [0usize, 6];
        let lm_cols = [12usize, 15, 18];
        let mut jacobian = Mat::<f64>::zeros(36, 21);
        // Two identical cameras observing three identical landmarks with a
        // uniform coefficient makes the reduced camera system exactly
        // rank-deficient (the two cameras become linearly dependent) — sparse
        // Cholesky silently regularizes past that, but `solve_spd` does not,
        // matching `DenseCholeskySolver`'s own no-retry contract. Varying the
        // coefficient per (camera, landmark, k) breaks the symmetry while
        // keeping H_cc/H_pp diagonal-dominant, so `S` is genuinely PD.
        for (ci, &cam_col) in cam_cols.iter().enumerate() {
            for (li, &lm_col) in lm_cols.iter().enumerate() {
                let row_base = (ci * 3 + li) * 6;
                for k in 0..6 {
                    let cam_val = 1.0 + 0.1 * (ci as f64 + 1.0);
                    let lm_val = 0.3 + 0.05 * ((k + li + 2 * ci) % 5) as f64;
                    jacobian[(row_base + k, cam_col + k)] = cam_val;
                    jacobian[(row_base + k, lm_col + (k % 3))] = lm_val;
                }
            }
        }
        let residuals = Mat::from_fn(36, 1, |i, _| (i % 5) as f64 * 0.1);

        let mut landmark_keys = std::collections::HashSet::new();
        landmark_keys.insert(pt0);
        landmark_keys.insert(pt1);
        landmark_keys.insert(pt2);

        Ok((
            variables,
            variable_index_map,
            jacobian,
            residuals,
            landmark_keys,
        ))
    }

    #[test]
    fn test_dense_schur_initialize_structure() -> TestResult {
        let (variables, variable_index_map, _, _, landmark_keys) = create_schur_test_setup()?;
        let mut solver = ExplicitDenseSchur::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let partition = solver.partition().ok_or("partition is None")?;
        assert_eq!(partition.kept_dof(), 12);
        assert_eq!(partition.eliminated_dof(), 9);
        Ok(())
    }

    #[test]
    fn test_dense_schur_solve_normal_equation() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ExplicitDenseSchur::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_dense_schur_solve_augmented_equation() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ExplicitDenseSchur::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta = LinearSolver::<DenseMode>::solve_augmented_equation(
            &mut solver,
            &residuals,
            &jacobian,
            &Damping::identity(0.1),
        )?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_dense_schur_matches_dense_cholesky_on_full_system() -> TestResult {
        use crate::linalg::dense::cholesky::DenseCholeskySolver;

        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;

        let mut cholesky = DenseCholeskySolver::new();
        let reference =
            LinearSolver::<DenseMode>::solve_normal_equation(&mut cholesky, &residuals, &jacobian)?;

        let mut schur = ExplicitDenseSchur::new();
        schur.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let step =
            LinearSolver::<DenseMode>::solve_normal_equation(&mut schur, &residuals, &jacobian)?;

        for i in 0..21 {
            assert!(
                (reference[(i, 0)] - step[(i, 0)]).abs() < 1e-9,
                "component {i}: cholesky {}, schur {}",
                reference[(i, 0)],
                step[(i, 0)]
            );
        }
        Ok(())
    }

    #[test]
    fn test_dense_schur_get_hessian_gradient() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ExplicitDenseSchur::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        assert!(LinearSolver::<DenseMode>::get_hessian(&solver).is_none());
        assert!(LinearSolver::<DenseMode>::get_gradient(&solver).is_none());

        LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        let h = LinearSolver::<DenseMode>::get_hessian(&solver).ok_or("hessian is None")?;
        let g = LinearSolver::<DenseMode>::get_gradient(&solver).ok_or("gradient is None")?;
        assert_eq!(h.nrows(), 21);
        assert_eq!(g.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_dense_schur_solve_without_init_returns_error() {
        let jacobian = Mat::<f64>::zeros(1, 1);
        let residuals = Mat::<f64>::zeros(1, 1);
        let mut solver = ExplicitDenseSchur::new();
        let result =
            LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian);
        assert!(result.is_err());
    }

    /// Non-contiguous, mixed-DOF partition — the same generality the sparse
    /// solver supports, proven directly against dense Cholesky.
    #[test]
    fn test_dense_schur_supports_non_contiguous_mixed_dof_partition() -> TestResult {
        let a = Variable::new(se3::SE3::from_param_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]));
        let b = Variable::new(se3::SE3::from_param_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]));
        let p = Variable::new(rn::Rn::new(DVector::from_vec(vec![0.0])));
        let q = Variable::new(rn::Rn::new(DVector::from_vec(vec![0.0])));

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let key_a = variables.insert(Box::new(a));
        let key_p = variables.insert(Box::new(p));
        let key_b = variables.insert(Box::new(b));
        let key_q = variables.insert(Box::new(q));

        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(key_a, 0);
        variable_index_map.insert(key_p, 6);
        variable_index_map.insert(key_b, 7);
        variable_index_map.insert(key_q, 13);

        let mut landmark_keys = std::collections::HashSet::new();
        landmark_keys.insert(key_p);
        landmark_keys.insert(key_q);

        let mut jacobian = Mat::<f64>::zeros(4 + 14, 14);
        let mut row = 0usize;
        for &cam_col in &[0usize, 7] {
            for &lm_col in &[6usize, 13] {
                jacobian[(row, cam_col)] = 1.0;
                jacobian[(row, lm_col)] = 1.0;
                row += 1;
            }
        }
        for col in 0..14 {
            jacobian[(row, col)] = 0.5;
            row += 1;
        }
        let residuals = Mat::from_fn(row, 1, |i, _| 0.1 * (i as f64 + 1.0));

        let mut cholesky = crate::linalg::dense::cholesky::DenseCholeskySolver::new();
        let reference =
            LinearSolver::<DenseMode>::solve_normal_equation(&mut cholesky, &residuals, &jacobian)?;

        let mut solver = ExplicitDenseSchur::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let step =
            LinearSolver::<DenseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        for i in 0..14 {
            assert!(
                (reference[(i, 0)] - step[(i, 0)]).abs() < 1e-9,
                "component {i}: cholesky {}, schur {}",
                reference[(i, 0)],
                step[(i, 0)]
            );
        }
        Ok(())
    }
}
