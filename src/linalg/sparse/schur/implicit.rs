//! # Implicit Sparse Schur Complement Solver
//!
//! This module implements the **implicit** (matrix-free) Schur complement
//! method using Preconditioned Conjugate Gradients (PCG) for bundle
//! adjustment. Equivalent to Ceres's `ITERATIVE_SCHUR`.
//!
//! ## Explicit vs Implicit Schur Complement
//!
//! **Implicit Schur:** This formulation never constructs the reduced camera matrix S
//! explicitly. Instead, it solves the linear system using a matrix-free approach where
//! only the matrix-vector product S·x is computed. This is highly memory-efficient for
//! large-scale problems.
//!
//! **Explicit Schur:** The alternative formulation (see [`explicit`](super::explicit))
//! physically constructs S = B - E C⁻¹ Eᵀ in memory and uses sparse Cholesky factorization.
//!
//! ## When to Use Implicit Schur
//!
//! - Very large bundle adjustment problems (> 10,000 cameras)
//! - Memory-constrained environments
//! - When iterative methods converge well (good preconditioning)
//! - When the reduced camera system S is too large to store explicitly
//!
//! ## Automatic group recognition
//!
//! Like every other Schur solver, the eliminated ("group 0") and retained
//! ("group 1") variable sets come from [`SchurPartition`] — the same
//! structure [`ExplicitSparseSchur`](super::explicit::ExplicitSparseSchur) and
//! [`ExplicitDenseSchur`](crate::linalg::dense::schur::ExplicitDenseSchur)
//! use. That means this solver places no restriction on the eliminated
//! variables' DOF or column layout: 3-D points, 1-DOF inverse-depth
//! landmarks, or a mix of both in one problem all work identically, and the
//! retained and eliminated columns may interleave arbitrarily.
//!
//! ## Algorithm
//!
//! 1. Form Schur complement implicitly: S = H_cc - H_cp * H_pp^{-1} * H_cp^T
//! 2. Solve S*δc = g_reduced using PCG (matrix-free)
//! 3. Back-substitute: δp = H_pp^{-1} * (g_p - H_cp^T * δc)
//!
//! ## Usage Example
//!
//! ```no_run
//! # use apex_solver::linalg::{ImplicitSparseSchur, SchurPreconditioner};
//! # use apex_solver::linalg::StructureAware;
//! # use apex_solver::core::VarKey;
//! # use apex_solver::core::variable::ManifoldVariable;
//! # use slotmap::{SlotMap, SecondaryMap};
//! # use std::collections::HashSet;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
//! # let variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
//! # let landmark_keys: HashSet<VarKey> = HashSet::new();
//! use apex_solver::linalg::{ImplicitSparseSchur, SchurPreconditioner};
//! use apex_solver::linalg::StructureAware;
//!
//! let mut solver = ImplicitSparseSchur::with_config(500, 1e-9, SchurPreconditioner::SchurJacobi);
//! solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
//! # Ok(())
//! # }
//! ```

use crate::core::VarKey;
use crate::core::variable::ManifoldVariable;
use crate::error::ErrorLogging;
use crate::linalg::regularization::invert_with_retry_dyn;
use crate::linalg::schur::{
    BlockSpan, EliminatedBlocks, PcgParams, SchurOrdering, SchurPartition, SchurPreconditioner,
    effective_landmark_keys, solve_pcg,
};
use crate::linalg::sparse::normal_eq::{LazyNormalEquations, NormalEquations};
use crate::linalg::sparse::pattern;
use crate::linalg::{Damping, LinAlgError, LinAlgResult, LinearSolver, SparseMode, StructureAware};
use faer::Mat;
use faer::sparse::SparseColMat;
use nalgebra::DMatrix;
use rayon::prelude::*;
use slotmap::{SecondaryMap, SlotMap};
use std::collections::HashMap;

/// Implicit (matrix-free) Schur complement solver using Preconditioned
/// Conjugate Gradients. Equivalent to Ceres's `ITERATIVE_SCHUR`.
#[derive(Debug, Clone)]
pub struct ImplicitSparseSchur {
    partition: Option<SchurPartition>,
    /// `H_ee⁻¹` per eliminated block, gathered and inverted once per solve.
    eliminated: EliminatedBlocks,
    /// Largest eliminated block's DOF, for sizing the block-apply scratch.
    max_eliminated_dof: usize,
    /// Automatic eliminated/retained ("group 0"/"group 1") classification,
    /// combined with manual marks in `initialize_structure` — the same
    /// mechanism [`ExplicitSparseSchur`](super::explicit::ExplicitSparseSchur)
    /// and [`ExplicitDenseSchur`](crate::linalg::dense::schur::ExplicitDenseSchur)
    /// use, so all three solvers recognize the same groups.
    ordering: SchurOrdering,

    // CG parameters
    max_cg_iterations: usize,
    cg_tolerance: f64,

    // Preconditioner type
    preconditioner_type: SchurPreconditioner,

    // Cached symbolic machinery for forming `JᵀJ` and `Jᵀr` in parallel.
    ne_cache: LazyNormalEquations,

    /// The un-damped `JᵀJ`, published through [`LinearSolver::get_hessian`].
    hessian: Option<SparseColMat<usize, f64>>,
    /// `+Jᵀr`, published through [`LinearSolver::get_gradient`].
    gradient: Option<Mat<f64>>,

    // Workspace buffers for the Schur operator (avoid repeated allocations).
    workspace_lm: Vec<f64>,  // eliminated-DOF sized buffer
    workspace_cam: Vec<f64>, // kept-DOF sized buffer
    block_scratch: Vec<f64>, // max-eliminated-block-DOF sized scratch

    // Visibility index: kept-block index -> Vec<eliminated-block index>.
    // This avoids O(kept * eliminated) iteration in preconditioner computation.
    camera_to_landmark_visibility: Vec<Vec<usize>>,
    /// Structural fingerprint the visibility index was built from, so it is
    /// rebuilt whenever the sparsity changes.
    visibility_fingerprint: Option<pattern::PatternFingerprint>,
}

impl ImplicitSparseSchur {
    /// Create a new implicit Schur solver with default parameters.
    /// Default: Schur-Jacobi preconditioner, 500 max iterations, 1e-9 relative tolerance —
    /// tighter settings that match Ceres Solver behavior for accurate step computation.
    pub fn new() -> Self {
        Self::with_config(500, 1e-9, SchurPreconditioner::SchurJacobi)
    }

    /// Create solver with custom CG parameters.
    pub fn with_cg_params(max_iterations: usize, tolerance: f64) -> Self {
        Self::with_config(max_iterations, tolerance, SchurPreconditioner::SchurJacobi)
    }

    /// Create solver with full configuration.
    pub fn with_config(
        max_iterations: usize,
        tolerance: f64,
        preconditioner: SchurPreconditioner,
    ) -> Self {
        Self {
            partition: None,
            eliminated: EliminatedBlocks::default(),
            max_eliminated_dof: 0,
            ordering: SchurOrdering::default(),
            max_cg_iterations: max_iterations,
            cg_tolerance: tolerance,
            preconditioner_type: preconditioner,
            ne_cache: LazyNormalEquations::default(),
            hessian: None,
            gradient: None,
            workspace_lm: Vec::new(),
            workspace_cam: Vec::new(),
            block_scratch: Vec::new(),
            camera_to_landmark_visibility: Vec::new(),
            visibility_fingerprint: None,
        }
    }

    /// Set the automatic group-classification ordering (see [`SchurOrdering`]).
    pub fn with_ordering(mut self, ordering: SchurOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    /// Borrow the partition, or report that `initialize_structure` was skipped.
    fn require_partition(&self) -> LinAlgResult<&SchurPartition> {
        self.partition.as_ref().ok_or_else(|| {
            LinAlgError::InvalidInput(
                "Block structure not built. Call initialize_structure() first.".to_string(),
            )
            .log()
        })
    }

    /// Apply Schur complement operator: `S·x = (H_kk − H_ke·H_ee⁻¹·H_keᵀ)·x`.
    ///
    /// Computes the matrix-vector product without ever forming `S`, walking
    /// [`SchurPartition`] rather than assuming a contiguous, fixed-DOF layout —
    /// this is the generalization over the pre-`SchurPartition` version of
    /// this solver, which only supported 3-DOF landmarks in a contiguous
    /// column range.
    #[allow(clippy::too_many_arguments)]
    fn apply_schur_operator_fast(
        &self,
        partition: &SchurPartition,
        hessian: &SparseColMat<usize, f64>,
        x: &Mat<f64>,
        result: &mut Mat<f64>,
        temp_lm: &mut [f64],
        temp_cam: &mut [f64],
        block_scratch: &mut [f64],
    ) {
        let symbolic = hessian.symbolic();

        temp_lm.iter_mut().for_each(|v| *v = 0.0);
        temp_cam.iter_mut().for_each(|v| *v = 0.0);

        // Fused Step 1+2: result = H_kk * x AND temp_lm = H_ke^T * x.
        // Walking every kept column once extracts both products.
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let Some(local_col) = partition.kept_local(col) else {
                    continue;
                };
                let x_val = x[(local_col, 0)];
                if x_val == 0.0 {
                    continue;
                }
                let rows = symbolic.row_idx_of_col_raw(col);
                let vals = hessian.val_of_col(col);
                for (idx, &row) in rows.iter().enumerate() {
                    let val = vals[idx];
                    if let Some(local_row) = partition.kept_local(row) {
                        result[(local_row, 0)] += val * x_val;
                    } else if let Some((block_idx, offset)) = partition.eliminated_local(row) {
                        let base = partition.eliminated_offset(block_idx);
                        temp_lm[base + offset] += val * x_val;
                    }
                }
            }
        }

        // Step 3: temp_lm = H_ee^{-1} * temp_lm, blockwise, any DOF.
        for (block_idx, _) in partition.eliminated_blocks().iter().enumerate() {
            let dof = self.eliminated.dof(block_idx);
            if dof == 0 {
                continue;
            }
            let base = partition.eliminated_offset(block_idx);
            let inv = self.eliminated.block(block_idx);
            let scratch = &mut block_scratch[..dof];
            for r in 0..dof {
                let mut acc = 0.0;
                for c in 0..dof {
                    acc += inv[c * dof + r] * temp_lm[base + c];
                }
                scratch[r] = acc;
            }
            temp_lm[base..base + dof].copy_from_slice(scratch);
        }

        // Step 4: temp_cam = H_ke * temp_lm (iterate over eliminated columns).
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let lm_val = temp_lm[base + offset];
                if lm_val == 0.0 {
                    continue;
                }
                let rows = symbolic.row_idx_of_col_raw(col);
                let vals = hessian.val_of_col(col);
                for (idx, &row) in rows.iter().enumerate() {
                    if let Some(local_row) = partition.kept_local(row) {
                        temp_cam[local_row] += vals[idx] * lm_val;
                    }
                }
            }
        }

        // Step 5: result = result - temp_cam = H_kk*x - H_ke*H_ee^{-1}*H_ke^T*x
        for i in 0..partition.kept_dof() {
            result[(i, 0)] -= temp_cam[i];
        }
    }

    /// `H_keᵀ · x`: `x` is kept-DOF, the result is eliminated-DOF.
    fn extract_coupling_transpose_mvp(
        &self,
        hessian: &SparseColMat<usize, f64>,
        x: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        let partition = self.require_partition()?;
        let symbolic = hessian.symbolic();
        let mut result = Mat::<f64>::zeros(partition.eliminated_dof(), 1);

        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let Some(local_col) = partition.kept_local(col) else {
                    continue;
                };
                let x_val = x[(local_col, 0)];
                if x_val == 0.0 {
                    continue;
                }
                let rows = symbolic.row_idx_of_col_raw(col);
                let vals = hessian.val_of_col(col);
                for (idx, &row) in rows.iter().enumerate() {
                    if let Some((block_idx, offset)) = partition.eliminated_local(row) {
                        let base = partition.eliminated_offset(block_idx);
                        result[(base + offset, 0)] += vals[idx] * x_val;
                    }
                }
            }
        }
        Ok(result)
    }

    /// `H_ke · x`: `x` is eliminated-DOF, the result is kept-DOF.
    fn extract_coupling_mvp(
        &self,
        hessian: &SparseColMat<usize, f64>,
        x: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        let partition = self.require_partition()?;
        let symbolic = hessian.symbolic();
        let mut result = Mat::<f64>::zeros(partition.kept_dof(), 1);

        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let x_val = x[(base + offset, 0)];
                if x_val == 0.0 {
                    continue;
                }
                let rows = symbolic.row_idx_of_col_raw(col);
                let vals = hessian.val_of_col(col);
                for (idx, &row) in rows.iter().enumerate() {
                    if let Some(local_row) = partition.kept_local(row) {
                        result[(local_row, 0)] += vals[idx] * x_val;
                    }
                }
            }
        }
        Ok(result)
    }

    /// Apply `H_ee⁻¹` using the cached block inverses.
    fn apply_eliminated_inverse(
        &self,
        input: &Mat<f64>,
        output: &mut Mat<f64>,
    ) -> LinAlgResult<()> {
        let partition = self.require_partition()?;
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let dof = block.dof;
            let base = partition.eliminated_offset(block_idx);
            let inv = self.eliminated.block(block_idx);
            for r in 0..dof {
                let mut acc = 0.0;
                for c in 0..dof {
                    acc += inv[c * dof + r] * input[(base + c, 0)];
                }
                output[(base + r, 0)] = acc;
            }
        }
        Ok(())
    }

    /// Compute block-Jacobi preconditioner: inverts the kept diagonal blocks
    /// of `H_kk` only.
    ///
    /// NOTE: This is NOT the true Schur-Jacobi preconditioner — it ignores the
    /// Schur complement's correction entirely. For better convergence use
    /// [`Self::compute_schur_jacobi_preconditioner`].
    fn compute_block_preconditioner(
        &self,
        hessian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Vec<DMatrix<f64>>> {
        let partition = self.require_partition()?;
        let symbolic = hessian.symbolic();

        let precond_blocks = partition
            .kept_blocks()
            .iter()
            .map(|block| {
                let size = block.dof;
                let mut mat = DMatrix::<f64>::zeros(size, size);
                for local_col in 0..size {
                    let global_col = block.col_start + local_col;
                    let rows = symbolic.row_idx_of_col_raw(global_col);
                    let vals = hessian.val_of_col(global_col);
                    for (idx, &global_row) in rows.iter().enumerate() {
                        if global_row >= block.col_start && global_row < block.col_start + size {
                            mat[(global_row - block.col_start, local_col)] = vals[idx];
                        }
                    }
                }
                invert_with_retry_dyn(&mat).unwrap_or_else(|| DMatrix::identity(size, size))
            })
            .collect();

        Ok(precond_blocks)
    }

    /// Apply block-Jacobi preconditioner: `z = M⁻¹ · r`.
    fn apply_block_preconditioner(
        &self,
        partition: &SchurPartition,
        r: &Mat<f64>,
        precond_blocks: &[DMatrix<f64>],
    ) -> Mat<f64> {
        let mut z = Mat::<f64>::zeros(partition.kept_dof(), 1);

        for (block_idx, block) in partition.kept_blocks().iter().enumerate() {
            let size = block.dof;
            let inv = &precond_blocks[block_idx];
            for i in 0..size {
                let Some(local_row) = partition.kept_local(block.col_start + i) else {
                    continue;
                };
                let mut acc = 0.0;
                for j in 0..size {
                    let Some(local_col) = partition.kept_local(block.col_start + j) else {
                        continue;
                    };
                    acc += inv[(i, j)] * r[(local_col, 0)];
                }
                z[(local_row, 0)] = acc;
            }
        }
        z
    }

    /// Compute the TRUE Schur-Jacobi preconditioner: diagonal blocks of the
    /// Schur complement `S` itself. This is what Ceres uses for
    /// `SCHUR_JACOBI`.
    ///
    /// For each kept block `i`: `S[i,i] = H_kk[i,i] − Σⱼ H_ke[i,j]·H_ee[j,j]⁻¹·H_ke[i,j]ᵀ`,
    /// summed over the eliminated blocks `j` visible to `i` — captured by the
    /// visibility index, so this costs `O(observations)` rather than
    /// `O(kept × eliminated)`.
    fn compute_schur_jacobi_preconditioner(
        &self,
        hessian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Vec<DMatrix<f64>>> {
        let partition = self.require_partition()?;
        let symbolic = hessian.symbolic();
        let visibility = &self.camera_to_landmark_visibility;

        let precond_blocks: Vec<DMatrix<f64>> = partition
            .kept_blocks()
            .par_iter()
            .enumerate()
            .map(|(cam_idx, block)| {
                let cam_size = block.dof;
                let mut s_ii = DMatrix::<f64>::zeros(cam_size, cam_size);

                for local_col in 0..cam_size {
                    let global_col = block.col_start + local_col;
                    let rows = symbolic.row_idx_of_col_raw(global_col);
                    let vals = hessian.val_of_col(global_col);
                    for (idx, &global_row) in rows.iter().enumerate() {
                        if global_row >= block.col_start && global_row < block.col_start + cam_size
                        {
                            s_ii[(global_row - block.col_start, local_col)] = vals[idx];
                        }
                    }
                }

                let visible_landmarks = visibility.get(cam_idx).map_or(&[][..], |v| v.as_slice());
                for &lm_block_idx in visible_landmarks {
                    let lm_block = partition.eliminated_blocks()[lm_block_idx];
                    let dof = lm_block.dof;
                    let mut h_cp = DMatrix::<f64>::zeros(cam_size, dof);

                    for col_offset in 0..dof {
                        let global_col = lm_block.col_start + col_offset;
                        let rows = symbolic.row_idx_of_col_raw(global_col);
                        let vals = hessian.val_of_col(global_col);
                        for (idx, &global_row) in rows.iter().enumerate() {
                            if global_row >= block.col_start
                                && global_row < block.col_start + cam_size
                            {
                                h_cp[(global_row - block.col_start, col_offset)] = vals[idx];
                            }
                        }
                    }

                    // H_pp[j,j]^{-1} from the cached inverses, column-major flat.
                    let hpp_inv = self.eliminated.block(lm_block_idx);

                    // temp = H_cp * H_pp^{-1} (cam_size x dof)
                    let mut temp = DMatrix::<f64>::zeros(cam_size, dof);
                    for i in 0..cam_size {
                        for j in 0..dof {
                            let mut sum = 0.0;
                            for k in 0..dof {
                                sum += h_cp[(i, k)] * hpp_inv[j * dof + k];
                            }
                            temp[(i, j)] = sum;
                        }
                    }

                    // contribution = temp * H_cp^T (cam_size x cam_size)
                    for i in 0..cam_size {
                        for j in 0..cam_size {
                            let mut sum = 0.0;
                            for k in 0..dof {
                                sum += temp[(i, k)] * h_cp[(j, k)];
                            }
                            s_ii[(i, j)] -= sum;
                        }
                    }
                }

                invert_with_retry_dyn(&s_ii)
                    .unwrap_or_else(|| DMatrix::identity(cam_size, cam_size))
            })
            .collect();

        Ok(precond_blocks)
    }

    /// Solve `S·x = b` with the shared PCG primitive and the fast matrix-free
    /// operator.
    fn solve_pcg_block(
        &mut self,
        partition: &SchurPartition,
        hessian: &SparseColMat<usize, f64>,
        b: &Mat<f64>,
        precond_blocks: &[DMatrix<f64>],
    ) -> Mat<f64> {
        // Workspace buffers are taken out of `self` for the duration of the
        // solve so the operator closure can borrow `self` immutably alongside
        // them, mirroring `ExplicitSparseSchur`'s `eliminated` handling.
        let mut workspace_lm = std::mem::take(&mut self.workspace_lm);
        let mut workspace_cam = std::mem::take(&mut self.workspace_cam);
        let mut block_scratch = std::mem::take(&mut self.block_scratch);

        let result = solve_pcg(
            b,
            &PcgParams::new(self.max_cg_iterations, self.cg_tolerance),
            |p, ap| {
                self.apply_schur_operator_fast(
                    partition,
                    hessian,
                    p,
                    ap,
                    &mut workspace_lm,
                    &mut workspace_cam,
                    &mut block_scratch,
                );
            },
            |r, z| {
                *z = self.apply_block_preconditioner(partition, r, precond_blocks);
            },
        );

        self.workspace_lm = workspace_lm;
        self.workspace_cam = workspace_cam;
        self.block_scratch = block_scratch;

        result.x
    }

    /// Gather and invert the `H_ee` diagonal blocks (any DOF, mixed sizes),
    /// reusing the same regularized-retry policy every other Schur solver
    /// uses ([`EliminatedBlocks::invert_in_place`]).
    fn invert_eliminated_blocks(&mut self, hessian: &SparseColMat<usize, f64>) -> LinAlgResult<()> {
        let mut eliminated = std::mem::take(&mut self.eliminated);
        let result = (|| -> LinAlgResult<()> {
            let partition = self.require_partition()?;
            eliminated.gather(hessian, partition);
            eliminated.invert_in_place(partition)
        })();
        self.eliminated = eliminated;
        result
    }

    /// Build kept-block → visible-eliminated-block visibility index from the
    /// Hessian's sparsity, enabling `O(observations)` preconditioner
    /// computation instead of `O(kept × eliminated)`.
    fn build_visibility_index(&mut self, hessian: &SparseColMat<usize, f64>) -> LinAlgResult<()> {
        let fingerprint = pattern::PatternFingerprint::of(hessian);
        if self.visibility_fingerprint == Some(fingerprint)
            && !self.camera_to_landmark_visibility.is_empty()
        {
            return Ok(());
        }

        let partition = self.require_partition()?;
        let symbolic = hessian.symbolic();
        let num_kept_blocks = partition.kept_blocks().len();

        let mut row_to_kept_block: HashMap<usize, usize> = HashMap::new();
        for (idx, block) in partition.kept_blocks().iter().enumerate() {
            for offset in 0..block.dof {
                row_to_kept_block.insert(block.col_start + offset, idx);
            }
        }

        let mut visibility: Vec<Vec<usize>> = vec![Vec::new(); num_kept_blocks];
        for (lm_block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let global_col = block.col_start;
            if global_col >= hessian.ncols() {
                continue;
            }
            let rows = symbolic.row_idx_of_col_raw(global_col);
            for &row in rows {
                if let Some(&kept_idx) = row_to_kept_block.get(&row)
                    && visibility[kept_idx].last() != Some(&lm_block_idx)
                {
                    visibility[kept_idx].push(lm_block_idx);
                }
            }
        }

        self.camera_to_landmark_visibility = visibility;
        self.visibility_fingerprint = Some(fingerprint);
        Ok(())
    }

    /// Internal solve against an explicit system.
    ///
    /// `hessian` is the *damped* `JᵀJ + λ·D` (or the plain `JᵀJ` for an
    /// undamped solve) and `gradient` is `−Jᵀr`, the right-hand side of
    /// `H·dx = −Jᵀr`.
    fn solve_with_system(
        &mut self,
        hessian: &SparseColMat<usize, f64>,
        gradient: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        self.invert_eliminated_blocks(hessian)?;
        self.build_visibility_index(hessian)?;

        // Cloned so it no longer borrows `self`: `solve_pcg_block` below needs
        // `&mut self` for its workspace buffers while the operator closures it
        // drives still need the partition. Cheap — a handful of `Vec<BlockSpan>`
        // entries, cloned once per Newton iteration, not per PCG iteration.
        let partition = self.require_partition()?.clone();
        let kept_dof = partition.kept_dof();
        let eliminated_dof = partition.eliminated_dof();

        let mut g_k = Mat::<f64>::zeros(kept_dof, 1);
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                if let Some(local) = partition.kept_local(col) {
                    g_k[(local, 0)] = gradient[(col, 0)];
                }
            }
        }

        let mut g_e = Mat::<f64>::zeros(eliminated_dof, 1);
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                g_e[(base + offset, 0)] = gradient[(block.col_start + offset, 0)];
            }
        }

        let mut temp = Mat::<f64>::zeros(eliminated_dof, 1);
        self.apply_eliminated_inverse(&g_e, &mut temp)?;
        let correction = self.extract_coupling_mvp(hessian, &temp)?;

        let mut g_reduced = Mat::<f64>::zeros(kept_dof, 1);
        for i in 0..kept_dof {
            g_reduced[(i, 0)] = g_k[(i, 0)] - correction[(i, 0)];
        }

        let precond_blocks = match self.preconditioner_type {
            SchurPreconditioner::SchurJacobi => {
                self.compute_schur_jacobi_preconditioner(hessian)?
            }
            SchurPreconditioner::BlockDiagonal => self.compute_block_preconditioner(hessian)?,
            SchurPreconditioner::None => partition
                .kept_blocks()
                .iter()
                .map(|b| DMatrix::identity(b.dof, b.dof))
                .collect(),
        };

        let delta_cam = self.solve_pcg_block(&partition, hessian, &g_reduced, &precond_blocks);

        let hcp_t_delta_cam = self.extract_coupling_transpose_mvp(hessian, &delta_cam)?;
        let mut rhs_e = Mat::<f64>::zeros(eliminated_dof, 1);
        for i in 0..eliminated_dof {
            rhs_e[(i, 0)] = g_e[(i, 0)] - hcp_t_delta_cam[(i, 0)];
        }

        let mut delta_e = Mat::<f64>::zeros(eliminated_dof, 1);
        self.apply_eliminated_inverse(&rhs_e, &mut delta_e)?;

        self.combine_updates(&delta_cam, &delta_e)
    }

    /// Scatter the two solution halves back into one full-length update.
    fn combine_updates(&self, delta_k: &Mat<f64>, delta_e: &Mat<f64>) -> LinAlgResult<Mat<f64>> {
        let partition = self.require_partition()?;
        let mut delta = Mat::zeros(partition.total_dof(), 1);

        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let global = block.col_start + offset;
                if let Some(local) = partition.kept_local(global) {
                    delta[(global, 0)] = delta_k[(local, 0)];
                }
            }
        }
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                delta[(block.col_start + offset, 0)] = delta_e[(base + offset, 0)];
            }
        }

        Ok(delta)
    }
}

impl Default for ImplicitSparseSchur {
    fn default() -> Self {
        Self::new()
    }
}

impl StructureAware for ImplicitSparseSchur {
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
                LinAlgError::InvalidInput(format!("VarKey {:?} not in index map", key))
            })?;
            let span = BlockSpan {
                key,
                col_start,
                dof: variable.dof(),
            };
            if effective_keys.contains(&key) {
                eliminated.push(span);
            } else {
                kept.push(span);
            }
        }

        let partition = SchurPartition::new(kept, eliminated)?;
        self.max_eliminated_dof = partition
            .eliminated_blocks()
            .iter()
            .map(|b| b.dof)
            .max()
            .unwrap_or(0)
            .max(1);
        self.eliminated = EliminatedBlocks::new(&partition);
        self.workspace_lm = vec![0.0; partition.eliminated_dof()];
        self.workspace_cam = vec![0.0; partition.kept_dof()];
        self.block_scratch = vec![0.0; self.max_eliminated_dof];
        self.camera_to_landmark_visibility.clear();
        self.visibility_fingerprint = None;
        self.partition = Some(partition);
        Ok(())
    }
}

impl LinearSolver<SparseMode> for ImplicitSparseSchur {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        let NormalEquations { hessian, gradient } = self.ne_cache.compute(residuals, jacobian)?;
        let mut neg_gradient = Mat::<f64>::zeros(gradient.nrows(), 1);
        for i in 0..gradient.nrows() {
            neg_gradient[(i, 0)] = -gradient[(i, 0)];
        }

        let delta = self.solve_with_system(&hessian, &neg_gradient);
        if delta.is_ok() {
            self.gradient = Some(gradient);
            self.hessian = Some(hessian);
        }
        delta
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
        damping: &Damping,
    ) -> LinAlgResult<Mat<f64>> {
        let NormalEquations { hessian, gradient } = self.ne_cache.compute(residuals, jacobian)?;
        let mut neg_gradient = Mat::<f64>::zeros(gradient.nrows(), 1);
        for i in 0..gradient.nrows() {
            neg_gradient[(i, 0)] = -gradient[(i, 0)];
        }

        let augmented_hessian = self.ne_cache.damped_hessian(damping)?;

        let delta = self.solve_with_system(&augmented_hessian, &neg_gradient);
        if delta.is_ok() {
            self.hessian = Some(hessian);
            self.gradient = Some(gradient);
        }
        delta
    }

    fn hessian_vec_product(&self, v: &Mat<f64>) -> Option<Mat<f64>> {
        Some(
            <SparseMode as crate::linearizer::AssemblyBackend>::hessian_vec_product(
                self.hessian.as_ref()?,
                v,
            ),
        )
    }

    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>> {
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
    use faer::sparse::Triplet;
    use nalgebra::DVector;
    use slotmap::{SecondaryMap, SlotMap};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    type TestSetup = (
        SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        SecondaryMap<VarKey, usize>,
        SparseColMat<usize, f64>,
        faer::Mat<f64>,
        std::collections::HashSet<VarKey>,
    );

    /// Build the same 2-camera + 3-landmark test setup used in explicit tests.
    /// Jacobian: 36 rows × 21 cols  (H_cc = 3·I₁₂, H_pp = 4·I₃ — positive definite)
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
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        for (ci, &cam_col) in cam_cols.iter().enumerate() {
            for (li, &lm_col) in lm_cols.iter().enumerate() {
                let row_base = (ci * 3 + li) * 6;
                for k in 0..6 {
                    triplets.push(Triplet::new(row_base + k, cam_col + k, 1.0));
                    triplets.push(Triplet::new(row_base + k, lm_col + (k % 3), 1.0));
                }
            }
        }

        let jacobian = SparseColMat::try_new_from_triplets(36, 21, &triplets)?;
        let residuals = faer::Mat::from_fn(36, 1, |i, _| (i % 5) as f64 * 0.1);

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
    fn test_implicit_schur_creation() {
        let solver = ImplicitSparseSchur::new();
        assert_eq!(solver.max_cg_iterations, 500);
        assert_eq!(solver.cg_tolerance, 1e-9);
        assert_eq!(solver.preconditioner_type, SchurPreconditioner::SchurJacobi);
    }

    #[test]
    fn test_with_custom_params() {
        let solver = ImplicitSparseSchur::with_cg_params(100, 1e-8);
        assert_eq!(solver.max_cg_iterations, 100);
        assert_eq!(solver.cg_tolerance, 1e-8);
        assert_eq!(solver.preconditioner_type, SchurPreconditioner::SchurJacobi);
    }

    #[test]
    fn test_with_full_config() {
        let solver =
            ImplicitSparseSchur::with_config(200, 1e-10, SchurPreconditioner::BlockDiagonal);
        assert_eq!(solver.max_cg_iterations, 200);
        assert_eq!(solver.cg_tolerance, 1e-10);
        assert_eq!(
            solver.preconditioner_type,
            SchurPreconditioner::BlockDiagonal
        );
    }

    #[test]
    fn test_implicit_schur_default() {
        let solver = ImplicitSparseSchur::default();
        assert_eq!(solver.max_cg_iterations, 500);
        assert_eq!(solver.cg_tolerance, 1e-9);
        assert!(solver.partition.is_none());
        assert!(solver.hessian.is_none());
    }

    #[test]
    fn test_implicit_schur_initialize_structure() -> TestResult {
        let (variables, variable_index_map, _, _, landmark_keys) = create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let partition = solver.partition.as_ref().ok_or("partition is None")?;
        assert_eq!(partition.kept_blocks().len(), 2);
        assert_eq!(partition.eliminated_blocks().len(), 3);
        assert_eq!(partition.kept_dof(), 12);
        assert_eq!(partition.eliminated_dof(), 9);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_solve_normal_equation() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        assert_eq!(delta.ncols(), 1);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_solve_augmented_equation() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver,
            &residuals,
            &jacobian,
            &Damping::identity(0.1),
        )?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_get_hessian_gradient() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        assert!(LinearSolver::<SparseMode>::get_hessian(&solver).is_none());
        assert!(LinearSolver::<SparseMode>::get_gradient(&solver).is_none());

        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;

        let h = LinearSolver::<SparseMode>::get_hessian(&solver);
        let g = LinearSolver::<SparseMode>::get_gradient(&solver);
        assert!(h.is_some());
        assert!(g.is_some());
        let h = h.ok_or("hessian is None")?;
        let g = g.ok_or("gradient is None")?;
        assert_eq!(h.nrows(), 21);
        assert_eq!(g.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_block_diagonal_preconditioner() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver =
            ImplicitSparseSchur::with_config(500, 1e-6, SchurPreconditioner::BlockDiagonal);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_schur_jacobi_preconditioner() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver =
            ImplicitSparseSchur::with_config(500, 1e-6, SchurPreconditioner::SchurJacobi);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_no_preconditioner() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = ImplicitSparseSchur::with_config(500, 1e-6, SchurPreconditioner::None);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        Ok(())
    }

    #[test]
    fn test_implicit_schur_augmented_lambda_effect() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;

        let mut s1 = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        s1.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let d1 = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut s1,
            &residuals,
            &jacobian,
            &Damping::identity(0.001),
        )?;

        let mut s2 = ImplicitSparseSchur::with_cg_params(500, 1e-6);
        s2.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let d2 = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut s2,
            &residuals,
            &jacobian,
            &Damping::identity(100.0),
        )?;

        let norm_diff: f64 = (0..21).map(|i| (d1[(i, 0)] - d2[(i, 0)]).powi(2)).sum();
        assert!(
            norm_diff > 1e-10,
            "Different λ should yield different updates"
        );
        Ok(())
    }

    #[test]
    fn test_implicit_schur_solve_without_init_returns_error() -> TestResult {
        let triplets: Vec<Triplet<usize, usize, f64>> = vec![Triplet::new(0, 0, 1.0)];
        let jacobian =
            SparseColMat::try_new_from_triplets(1, 1, &triplets).map_err(|e| format!("{e:?}"))?;
        let residuals = faer::Mat::from_fn(1, 1, |_, _| 1.0);
        let mut solver = ImplicitSparseSchur::new();

        let result =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_implicit_initialize_structure_no_cameras_returns_error() {
        use crate::core::variable::Variable;
        use apex_manifolds::rn;

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let k = variables.insert(Box::new(Variable::new(rn::Rn::new(DVector::zeros(3)))));
        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(k, 0);
        let mut landmark_keys = std::collections::HashSet::new();
        landmark_keys.insert(k);

        let mut solver = ImplicitSparseSchur::new();
        let result = solver.initialize_structure(&variables, &variable_index_map, &landmark_keys);
        assert!(
            result.is_err(),
            "Expected Err when no retained variables present"
        );
    }

    #[test]
    fn test_implicit_initialize_structure_no_landmarks_returns_error() {
        use crate::core::variable::Variable;
        use apex_manifolds::se3;

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let k = variables.insert(Box::new(Variable::new(se3::SE3::from_param_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]))));
        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(k, 0);
        let landmark_keys = std::collections::HashSet::<VarKey>::new();

        let mut solver = ImplicitSparseSchur::new();
        let result = solver.initialize_structure(&variables, &variable_index_map, &landmark_keys);
        assert!(
            result.is_err(),
            "Expected Err when no eliminated variables present"
        );
    }

    /// A mixed-DOF, non-contiguous partition (1-DOF inverse depth interleaved
    /// with retained poses) must work end to end — the whole point of moving
    /// this solver onto [`SchurPartition`].
    #[test]
    fn test_implicit_schur_supports_non_contiguous_mixed_dof_partition() -> TestResult {
        use crate::linalg::schur::BlockSpan;

        // Layout: kept A [0..6), eliminated P [6..7) (1-DOF), kept B [7..13),
        // eliminated Q [13..14) (1-DOF). Neither side is contiguous.
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

        // Observations: A-P, B-P, A-Q, B-Q, plus a small prior on every
        // column so the system is SPD.
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        let mut row = 0usize;
        for &cam_col in &[0usize, 7] {
            for &lm_col in &[6usize, 13] {
                for k in 0..1 {
                    triplets.push(Triplet::new(row + k, cam_col + k, 1.0));
                    triplets.push(Triplet::new(row + k, lm_col, 1.0));
                }
                row += 1;
            }
        }
        for col in 0..14 {
            triplets.push(Triplet::new(row, col, 0.5));
            row += 1;
        }
        let jacobian = SparseColMat::try_new_from_triplets(row, 14, &triplets)?;
        let residuals = faer::Mat::from_fn(row, 1, |i, _| 0.1 * (i as f64 + 1.0));

        let mut solver = ImplicitSparseSchur::with_cg_params(500, 1e-10);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 14);
        for i in 0..14 {
            assert!(delta[(i, 0)].is_finite());
        }

        // A block spanning a variable never claimed by anything must still
        // round-trip through BlockSpan without panicking (sanity check on the
        // type used to build the partition above).
        let _ = BlockSpan {
            key: key_a,
            col_start: 0,
            dof: 6,
        };
        Ok(())
    }
}
