//! # Explicit Schur Complement Solver
//!
//! This module implements the **Explicit Schur Complement** method for bundle adjustment
//! and structured optimization problems.
//!
//! ## Explicit vs Implicit Schur Complement
//!
//! **Explicit Schur:** This formulation physically constructs the reduced camera matrix
//! (S = B - E C⁻¹ Eᵀ) in memory and solves it using direct sparse Cholesky factorization.
//! It provides the most accurate results with moderate memory usage.
//!
//! **Implicit Schur:** The alternative formulation (see [`implicit_schur`](super::implicit_schur))
//! never constructs S explicitly, instead solving the system using matrix-free PCG.
//! It's more memory-efficient for very large problems.
//!
//! ## When to Use Explicit Schur
//!
//! - Medium-to-large bundle adjustment problems (< 10,000 cameras)
//! - When accuracy is paramount
//! - When you have sufficient memory to store the reduced camera system
//! - When direct factorization is faster than iterative methods
//!
//! ## Usage Example
//!
//! ```no_run
//! # use apex_solver::linalg::{SparseSchurComplementSolver, SchurVariant, SchurPreconditioner};
//! # use apex_solver::linalg::StructureAware;
//! # use apex_solver::core::VarKey;
//! # use apex_solver::core::variable::ManifoldVariable;
//! # use slotmap::{SlotMap, SecondaryMap};
//! # use std::collections::HashSet;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
//! # let variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
//! # let landmark_keys: HashSet<VarKey> = HashSet::new();
//! use apex_solver::linalg::{SparseSchurComplementSolver, SchurVariant, SchurPreconditioner};
//! use apex_solver::linalg::StructureAware;
//!
//! let mut solver = SparseSchurComplementSolver::new()
//!     .with_variant(SchurVariant::Sparse) // Explicit Schur with Cholesky
//!     .with_preconditioner(SchurPreconditioner::None);
//! solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
//! # Ok(())
//! # }
//! ```

use super::schur_partition::{BlockSpan, EliminatedBlocks, SchurPartition};
use crate::core::VarKey;
use crate::core::variable::ManifoldVariable;
use crate::error::ErrorLogging;
use crate::linalg::sparse::normal_eq::{LazyNormalEquations, NormalEquations};
use crate::linalg::{Damping, LinAlgError, LinAlgResult, LinearSolver, SparseMode, StructureAware};
use apex_manifolds::ManifoldType;
use faer::sparse::{SparseColMat, Triplet};
use faer::{
    Accum, Mat, Side,
    linalg::solvers::Solve,
    sparse::linalg::solvers::{Llt, SymbolicLlt},
};
use rayon::prelude::*;
use slotmap::{SecondaryMap, SlotMap};
use tracing::debug;

/// Schur complement solver variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchurVariant {
    /// Form `S` explicitly, then factorize it with sparse Cholesky.
    ///
    /// Most accurate and fastest per iteration, but `S` is accumulated into a
    /// dense `kept_dof²` buffer, so memory grows quadratically in the retained
    /// set. Equivalent to Ceres's `SPARSE_SCHUR`.
    #[default]
    Sparse,
    /// Never form `S`: apply the Schur operator through `H_ke`/`H_ee⁻¹`
    /// products inside PCG.
    ///
    /// Memory is linear in the problem size, which is what makes very large
    /// camera sets tractable. Equivalent to Ceres's `ITERATIVE_SCHUR` in its
    /// default (implicit) mode, and the variant that honours
    /// [`SchurPreconditioner`].
    ///
    /// Requires 3-DOF eliminated blocks in contiguous column ranges; use
    /// [`Self::Sparse`] or [`Self::ExplicitIterative`] otherwise.
    Iterative,
    /// Form `S` explicitly, then solve it with PCG instead of Cholesky.
    ///
    /// Carries the same `kept_dof²` memory cost as [`Self::Sparse`] while
    /// solving less exactly, so it is rarely the right choice — it exists
    /// because it is what `Iterative` used to do, and it supports the general
    /// partitions the matrix-free path does not. Equivalent to Ceres's
    /// `ITERATIVE_SCHUR` with `use_explicit_schur_complement = true`.
    ExplicitIterative,
    /// Form `S` chunk by chunk **directly from `J`**, then factorize with
    /// sparse Cholesky.
    ///
    /// Algebraically identical to [`Self::Sparse`], but never materializes
    /// `JᵀJ`, `Jᵀ`, or the value permutation forming it requires — on the
    /// largest BAL problem those account for ~24 GB of the ~32 GB needed before
    /// elimination can even start. This is Ceres's `SchurEliminator` strategy.
    ///
    /// Requires each eliminated variable's rows to be contiguous;
    /// `Problem::group_rows_for_elimination` arranges that when a Schur solver
    /// is selected.
    ChunkedSparse,
}

/// Preconditioner type for iterative solvers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchurPreconditioner {
    /// No preconditioning
    None,
    /// Block diagonal of H_cc only (fast but less effective)
    BlockDiagonal,
    /// True Schur-Jacobi: Block diagonal of S = H_cc - H_cp * H_pp^{-1} * H_cp^T
    /// This is what Ceres uses and provides much better PCG convergence
    #[default]
    SchurJacobi,
}

/// Configuration for Schur complement variable ordering
///
/// Note the default eliminates **nothing** until variables are marked with
/// [`Problem::mark_for_elimination`](crate::core::problem::Problem::mark_for_elimination):
/// `auto_detect` is off because `Rn(3)` is ambiguous (landmarks vs
/// self-calibration intrinsics), so a default-constructed ordering only
/// *classifies* — the marks still have to exist.
#[derive(Debug, Clone)]
pub struct SchurOrdering {
    pub eliminate_types: Vec<ManifoldType>,
    /// Only eliminate RN variables with this exact size (default: 3 for 3D landmarks)
    /// This prevents intrinsic variables (6 DOF) from being eliminated
    pub eliminate_rn_size: Option<usize>,
    /// Auto-classify *unmarked* variables as landmarks when their type and size
    /// match [`Self::should_eliminate`].
    ///
    /// Off by default: `Rn(3)` is also how self-calibration represents intrinsic
    /// parameters (`[focal, k1, k2]`), and eliminating those as landmarks
    /// silently corrupts the Schur complement. Manual marks via
    /// `Problem::mark_for_elimination` always apply, with or without this flag.
    pub auto_detect: bool,
}

impl Default for SchurOrdering {
    fn default() -> Self {
        Self {
            eliminate_types: vec![ManifoldType::RN],
            eliminate_rn_size: Some(3), // Only eliminate 3D landmarks, not intrinsics
            auto_detect: false,
        }
    }
}

impl SchurOrdering {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable auto-classification of unmarked variables (see [`Self::auto_detect`]).
    pub fn with_auto_detect(mut self, enabled: bool) -> Self {
        self.auto_detect = enabled;
        self
    }

    /// Check if a variable should be eliminated (treated as landmark).
    ///
    /// Classification is based solely on manifold type and DOF size.
    /// By default, RN variables with exactly 3 DOF are treated as landmarks.
    pub fn should_eliminate(&self, manifold_type: &ManifoldType, size: usize) -> bool {
        if !self.eliminate_types.contains(manifold_type) {
            return false;
        }
        if let Some(required_size) = self.eliminate_rn_size
            && size != required_size
        {
            return false;
        }
        true
    }

    /// [`Self::should_eliminate`] for a variable identified by its manifold's
    /// [`LieGroup::NAME`] string (`"Rn"`, `"SE3"`, …), as reported by
    /// [`ManifoldVariable::manifold_type_name`]. Unknown names never eliminate.
    pub fn should_eliminate_by_name(&self, name: &str, size: usize) -> bool {
        match ManifoldType::from_name(name) {
            Some(manifold_type) => self.should_eliminate(&manifold_type, size),
            None => false,
        }
    }
}
/// Superseded by [`SchurPartition`], which supports eliminated blocks of any
/// DOF, mixed sizes within one problem, and non-contiguous partitions.
///
/// This alias keeps existing imports resolving. The replacement exposes
/// accessors (`kept_blocks()`, `eliminated_dof()`, …) rather than public
/// fields, so code that read the old fields must be updated.
#[deprecated(
    since = "1.6.0",
    note = "renamed and generalized to `SchurPartition`; eliminated blocks are no longer \
            restricted to 3 DOF and the partition may be non-contiguous"
)]
pub type SchurBlockStructure = SchurPartition;

/// Sparse Schur Complement Solver for Bundle Adjustment
#[derive(Debug, Clone)]
pub struct SparseSchurComplementSolver {
    partition: Option<SchurPartition>,
    /// Diagonal blocks of `H_ee`; allocated once per structure and reused.
    eliminated: EliminatedBlocks,
    /// Chunk-wise eliminator, built lazily for [`SchurVariant::ChunkedSparse`].
    chunked: Option<super::schur_eliminator::ChunkedSchurEliminator>,
    /// `J` from the last chunked solve.
    ///
    /// The chunked path never forms `JᵀJ`, so the quadratic model is served as
    /// `Jᵀ(J·v)` — which needs `J` after the solve returns. Holding it costs one
    /// extra copy of the Jacobian; still far less than the `Jᵀ`, permutation and
    /// `JᵀJ` that forming the normal equations would require.
    chunked_jacobian: Option<SparseColMat<usize, f64>>,
    /// Structural fingerprint of the Hessian whose block-diagonality was checked.
    ///
    /// The check is structural, so it only has to run when the sparsity
    /// changes — not on every solve. Keyed on the full
    /// [`PatternFingerprint`](super::pattern::PatternFingerprint): the old
    /// `(nrows, ncols, nnz)` triple aliased equal-`nnz` permutations and could
    /// skip the check for a coupled pattern.
    verified_pattern: Option<super::pattern::PatternFingerprint>,
    ordering: SchurOrdering,
    variant: SchurVariant,
    preconditioner: SchurPreconditioner,

    // CG parameters
    cg_max_iterations: usize,
    cg_tolerance: f64,

    // Cached symbolic machinery for forming `JᵀJ` and `Jᵀr` in parallel.
    ne_cache: LazyNormalEquations,

    // Cached matrices
    hessian: Option<SparseColMat<usize, f64>>,
    gradient: Option<Mat<f64>>,
}

impl SparseSchurComplementSolver {
    pub fn new() -> Self {
        Self {
            partition: None,
            eliminated: EliminatedBlocks::default(),
            chunked: None,
            chunked_jacobian: None,
            verified_pattern: None,
            ordering: SchurOrdering::default(),
            variant: SchurVariant::default(),
            preconditioner: SchurPreconditioner::default(),
            cg_max_iterations: 200, // Match Ceres (was 500)
            cg_tolerance: 1e-6,     // Relaxed for speed (was 1e-9)
            ne_cache: LazyNormalEquations::default(),
            hessian: None,
            gradient: None,
        }
    }

    pub fn with_ordering(mut self, ordering: SchurOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    pub fn with_variant(mut self, variant: SchurVariant) -> Self {
        self.variant = variant;
        self
    }

    pub fn with_preconditioner(mut self, preconditioner: SchurPreconditioner) -> Self {
        self.preconditioner = preconditioner;
        self
    }

    pub fn with_cg_params(mut self, max_iter: usize, tol: f64) -> Self {
        self.cg_max_iterations = max_iter;
        self.cg_tolerance = tol;
        self
    }
    /// The variable partition, once `initialize_structure` has run.
    pub fn partition(&self) -> Option<&SchurPartition> {
        self.partition.as_ref()
    }

    /// Verify `H_ee` is block-diagonal, once per sparsity pattern.
    ///
    /// Eliminating mutually connected variables silently yields a wrong step,
    /// so this is checked rather than assumed — but the check is structural,
    /// so repeating it every iteration would be pure overhead.
    fn ensure_block_diagonal(&mut self, hessian: &SparseColMat<usize, f64>) -> LinAlgResult<()> {
        let fingerprint = super::pattern::PatternFingerprint::of(hessian);
        if self.verified_pattern == Some(fingerprint) {
            return Ok(());
        }
        self.require_partition()?.verify_block_diagonal(hessian)?;
        self.verified_pattern = Some(fingerprint);
        Ok(())
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

    /// Union of the manually marked landmark keys and — when
    /// [`SchurOrdering::auto_detect`] is enabled — the variables matching the
    /// configured ordering.
    fn effective_landmark_keys(
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        schur_landmark_keys: &std::collections::HashSet<VarKey>,
        ordering: &SchurOrdering,
    ) -> std::collections::HashSet<VarKey> {
        let mut keys = schur_landmark_keys.clone();
        if ordering.auto_detect {
            for (key, variable) in variables {
                if ordering.should_eliminate_by_name(variable.manifold_type_name(), variable.dof())
                {
                    keys.insert(key);
                }
            }
        }
        keys
    }
    /// Partition the variables into eliminated and retained sets.
    ///
    /// Both sides accept arbitrary DOF and arbitrary column interleaving: the
    /// eliminated set need not be 3-DOF points, and need not occupy a
    /// contiguous column range. See [`SchurPartition`].
    fn build_partition(
        &mut self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
        eliminate_keys: &std::collections::HashSet<VarKey>,
    ) -> LinAlgResult<()> {
        let mut kept = Vec::new();
        let mut eliminated = Vec::new();

        for (key, variable) in variables {
            let col_start = *variable_index_map.get(key).ok_or_else(|| {
                LinAlgError::InvalidInput(format!("VarKey {:?} not found in index map", key)).log()
            })?;
            let span = BlockSpan {
                key,
                col_start,
                dof: variable.dof(),
            };
            if eliminate_keys.contains(&key) {
                eliminated.push(span);
            } else {
                kept.push(span);
            }
        }

        let partition = SchurPartition::new(kept, eliminated)?;
        self.eliminated = EliminatedBlocks::new(&partition);
        self.partition = Some(partition);
        self.verified_pattern = None;
        Ok(())
    }
    /// Extract `H_kk`, the retained-retained block.
    ///
    /// Indices go through the partition rather than a contiguous range, so the
    /// retained variables need not be adjacent in column space.
    fn extract_kept_block(
        &self,
        hessian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<SparseColMat<usize, f64>> {
        let partition = self.require_partition()?;
        let kept_dof = partition.kept_dof();
        let symbolic = hessian.symbolic();

        let mut triplets = Vec::new();
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let global_col = block.col_start + offset;
                let Some(local_col) = partition.kept_local(global_col) else {
                    continue;
                };
                let rows = symbolic.row_idx_of_col_raw(global_col);
                let vals = hessian.val_of_col(global_col);
                for (idx, &global_row) in rows.iter().enumerate() {
                    if let Some(local_row) = partition.kept_local(global_row) {
                        triplets.push(Triplet::new(local_row, local_col, vals[idx]));
                    }
                }
            }
        }

        SparseColMat::try_new_from_triplets(kept_dof, kept_dof, &triplets)
            .map_err(|e| LinAlgError::SparseMatrixCreation(format!("H_kk: {:?}", e)))
    }
    /// Extract `H_ke`, the coupling between retained and eliminated variables.
    ///
    /// Columns follow the eliminated-local ordering, so block `i` occupies
    /// `partition.eliminated_offset(i) .. + dof`.
    fn extract_coupling_block(
        &self,
        hessian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<SparseColMat<usize, f64>> {
        let partition = self.require_partition()?;
        let symbolic = hessian.symbolic();

        let mut triplets = Vec::new();
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let col_base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                let global_col = block.col_start + offset;
                let rows = symbolic.row_idx_of_col_raw(global_col);
                let vals = hessian.val_of_col(global_col);
                for (idx, &global_row) in rows.iter().enumerate() {
                    if let Some(local_row) = partition.kept_local(global_row) {
                        triplets.push(Triplet::new(local_row, col_base + offset, vals[idx]));
                    }
                }
            }
        }

        SparseColMat::try_new_from_triplets(
            partition.kept_dof(),
            partition.eliminated_dof(),
            &triplets,
        )
        .map_err(|e| LinAlgError::SparseMatrixCreation(format!("H_ke: {:?}", e)))
    }
    /// Split the gradient into its retained and eliminated parts.
    fn extract_gradient_blocks(&self, gradient: &Mat<f64>) -> LinAlgResult<(Mat<f64>, Mat<f64>)> {
        let partition = self.require_partition()?;

        let mut g_k = Mat::zeros(partition.kept_dof(), 1);
        let mut g_e = Mat::zeros(partition.eliminated_dof(), 1);

        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let global = block.col_start + offset;
                if let Some(local) = partition.kept_local(global) {
                    g_k[(local, 0)] = gradient[(global, 0)];
                }
            }
        }
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let base = partition.eliminated_offset(block_idx);
            for offset in 0..block.dof {
                g_e[(base + offset, 0)] = gradient[(block.col_start + offset, 0)];
            }
        }

        Ok((g_k, g_e))
    }

    /// Solve S * x = b using Cholesky factorization with automatic regularization
    ///
    /// If the initial factorization fails (matrix not positive definite),
    /// we add small regularization to the diagonal and retry.
    fn solve_with_cholesky(
        &self,
        a: &SparseColMat<usize, f64>,
        b: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        let sym = SymbolicLlt::try_new(a.symbolic(), Side::Lower).map_err(|e| {
            LinAlgError::FactorizationFailed(format!("Symbolic Cholesky failed: {:?}", e))
        })?;

        // First attempt: direct factorization
        match Llt::try_new_with_symbolic(sym.clone(), a.as_ref(), Side::Lower) {
            Ok(cholesky) => return Ok(cholesky.solve(b)),
            Err(e) => {
                debug!(
                    "Cholesky factorization failed: {:?}. Applying regularization.",
                    e
                );
            }
        }

        // Retry with exponentially increasing regularization
        let n = a.nrows();
        let symbolic = a.symbolic();

        // Compute trace and max diagonal for scaling
        let mut trace = 0.0;
        let mut max_diag = 0.0f64;
        for col in 0..n {
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = a.val_of_col(col);
            for (idx, &row) in row_indices.iter().enumerate() {
                if row == col {
                    trace += col_values[idx];
                    max_diag = max_diag.max(col_values[idx].abs());
                }
            }
        }

        // Try multiple regularization levels
        let avg_diag = trace / n as f64;
        let base_reg = avg_diag.max(max_diag).max(1.0);

        for attempt in 0..5 {
            let reg = base_reg * 10.0f64.powi(attempt - 4); // 1e-4, 1e-3, 1e-2, 1e-1, 1.0 times base
            debug!(
                "Cholesky attempt {}: regularization = {:.2e}",
                attempt + 2,
                reg
            );

            let mut triplets = Vec::with_capacity(n * 10);
            for col in 0..n {
                let row_indices = symbolic.row_idx_of_col_raw(col);
                let col_values = a.val_of_col(col);
                for (idx, &row) in row_indices.iter().enumerate() {
                    triplets.push(Triplet::new(row, col, col_values[idx]));
                }
            }

            for i in 0..n {
                triplets.push(Triplet::new(i, i, reg));
            }

            let a_reg = match SparseColMat::try_new_from_triplets(n, n, &triplets) {
                Ok(m) => m,
                Err(e) => {
                    debug!("Failed to create regularized matrix: {:?}", e);
                    continue;
                }
            };

            // Need to create a new symbolic structure for the regularized matrix
            let sym_reg = match SymbolicLlt::try_new(a_reg.symbolic(), Side::Lower) {
                Ok(s) => s,
                Err(e) => {
                    debug!("Symbolic factorization failed: {:?}", e);
                    continue;
                }
            };

            match Llt::try_new_with_symbolic(sym_reg, a_reg.as_ref(), Side::Lower) {
                Ok(cholesky) => {
                    debug!("Cholesky succeeded with regularization {:.2e}", reg);
                    return Ok(cholesky.solve(b));
                }
                Err(e) => {
                    debug!("Cholesky failed with reg {:.2e}: {:?}", reg, e);
                }
            }
        }

        Err(LinAlgError::SingularMatrix(format!(
            "Schur complement singular after 5 regularization attempts (max reg = {:.2e})",
            base_reg
        )))
    }

    /// Solve using Preconditioned Conjugate Gradients (PCG)
    ///
    /// Uses Jacobi (diagonal) preconditioning for simplicity and robustness.
    /// Reductions and vector updates run through faer's SIMD kernels; the
    /// SpMV uses faer's parallel sparse×dense kernel into a hoisted buffer.
    fn solve_with_pcg(&self, a: &SparseColMat<usize, f64>, b: &Mat<f64>) -> LinAlgResult<Mat<f64>> {
        let n = b.nrows();
        let max_iterations = self.cg_max_iterations;
        let tolerance = self.cg_tolerance;

        // Extract diagonal for Jacobi preconditioner
        let symbolic = a.symbolic();
        let mut precond = vec![1.0; n];
        for (col, precond_val) in precond.iter_mut().enumerate().take(n) {
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = a.val_of_col(col);
            for (idx, &row) in row_indices.iter().enumerate() {
                if row == col {
                    let diag = col_values[idx];
                    *precond_val = if diag.abs() > 1e-12 { 1.0 / diag } else { 1.0 };
                    break;
                }
            }
        }
        let precond_m = Mat::from_fn(n, 1, |i, _| precond[i]);

        // Initialize
        let mut x = Mat::<f64>::zeros(n, 1);

        // r = b - A*x (x starts at 0, so r = b)
        let mut r = b.clone();

        // z = M^{-1} * r (Jacobi preconditioning)
        let mut z = Mat::<f64>::zeros(n, 1);
        faer::zip!(&mut z, &precond_m, &r).for_each(|faer::unzip!(z, m, r)| *z = m * r);

        let mut p = z.clone();

        let mut rz_old: f64 = (r.transpose() * &z)[(0, 0)];

        // Compute initial residual norm for relative tolerance
        let abs_tol = tolerance * r.norm_l2().max(1.0);

        // Ap buffer (reused each iteration)
        let mut ap = Mat::<f64>::zeros(n, 1);

        for _iter in 0..max_iterations {
            // Ap = A * p (parallel sparse×dense faer kernel)
            faer::sparse::linalg::matmul::sparse_dense_matmul(
                ap.as_mut(),
                Accum::Replace,
                a.as_ref(),
                p.as_ref(),
                1.0,
                faer::get_global_parallelism(),
            );

            // alpha = (r^T z) / (p^T Ap)
            let p_ap: f64 = (p.transpose() * &ap)[(0, 0)];

            if p_ap.abs() < 1e-30 {
                break;
            }

            let alpha = rz_old / p_ap;

            // x = x + alpha * p
            faer::zip!(&mut x, &p).for_each(|faer::unzip!(x, p)| *x += alpha * p);

            // r = r - alpha * Ap
            faer::zip!(&mut r, &ap).for_each(|faer::unzip!(r, ap)| *r -= alpha * ap);

            // Check convergence
            if r.norm_l2() < abs_tol {
                break;
            }

            // z = M^{-1} * r
            faer::zip!(&mut z, &precond_m, &r).for_each(|faer::unzip!(z, m, r)| *z = m * r);

            // beta = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)
            let rz_new: f64 = (r.transpose() * &z)[(0, 0)];

            if rz_old.abs() < 1e-30 {
                break;
            }

            let beta = rz_new / rz_old;

            // p = z + beta * p
            faer::zip!(&mut p, &z).for_each(|faer::unzip!(p, z)| *p = *z + beta * *p);

            rz_old = rz_new;
        }

        Ok(x)
    }
    /// Form the Schur complement `S = H_kk − H_ke·H_ee⁻¹·H_keᵀ`.
    ///
    /// Exploits the block-diagonal structure of `H_ee`: each eliminated block
    /// contributes independently, touching only the retained rows it couples
    /// to. Blocks may have any DOF and may differ from one another, so a 1-DOF
    /// inverse depth and a 3-DOF point can be eliminated in the same solve.
    ///
    /// `S` is accumulated densely (`kept_dof²`) and then filtered back to
    /// sparse; that buffer is the current scaling limit for very large
    /// retained sets.
    fn compute_schur_complement(
        &self,
        h_kk: &SparseColMat<usize, f64>,
        h_ke: &SparseColMat<usize, f64>,
        h_ee_inv: &EliminatedBlocks,
    ) -> LinAlgResult<SparseColMat<usize, f64>> {
        let partition = self.require_partition()?;
        let kept_dof = h_kk.nrows();
        let h_ke_symbolic = h_ke.symbolic();

        let mut s_dense = vec![0.0f64; kept_dof * kept_dof];

        // S starts as H_kk.
        let h_kk_symbolic = h_kk.symbolic();
        for col in 0..h_kk.ncols() {
            let rows = h_kk_symbolic.row_idx_of_col_raw(col);
            let vals = h_kk.val_of_col(col);
            for (idx, &row) in rows.iter().enumerate() {
                s_dense[row * kept_dof + col] += vals[idx];
            }
        }

        let max_dof = partition
            .eliminated_blocks()
            .iter()
            .map(|b| b.dof)
            .max()
            .unwrap_or(0)
            .max(1);
        // Scratch reused across blocks; `h_ke_rows` and `contrib` are row-major
        // `n_rows × dof` strips.
        let mut kept_rows: Vec<usize> = Vec::with_capacity(32);
        let mut h_ke_rows: Vec<f64> = Vec::with_capacity(32 * max_dof);
        let mut contrib: Vec<f64> = Vec::with_capacity(32 * max_dof);
        let mut cursors: Vec<usize> = Vec::with_capacity(max_dof);
        let mut col_rows: Vec<&[usize]> = Vec::with_capacity(max_dof);
        let mut col_vals: Vec<&[f64]> = Vec::with_capacity(max_dof);

        for block_idx in 0..h_ee_inv.len() {
            let dof = h_ee_inv.dof(block_idx);
            if dof == 0 {
                continue;
            }
            let col_base = partition.eliminated_offset(block_idx);

            kept_rows.clear();
            h_ke_rows.clear();
            cursors.clear();
            cursors.resize(dof, 0);

            // Hoist the column slices: looking them up per row per column costs
            // more than the merge itself.
            col_rows.clear();
            col_vals.clear();
            for local in 0..dof {
                let col = col_base + local;
                col_rows.push(h_ke_symbolic.row_idx_of_col_raw(col));
                col_vals.push(h_ke.val_of_col(col));
            }

            // Merge the block's `dof` columns into dense rows: each retained
            // variable appears once, carrying one value per eliminated column
            // (zero where structurally absent).
            loop {
                let mut min_row = usize::MAX;
                for local in 0..dof {
                    if let Some(&row) = col_rows[local].get(cursors[local]) {
                        min_row = min_row.min(row);
                    }
                }
                if min_row == usize::MAX {
                    break;
                }

                let strip = h_ke_rows.len();
                h_ke_rows.resize(strip + dof, 0.0);
                for local in 0..dof {
                    if col_rows[local].get(cursors[local]) == Some(&min_row) {
                        h_ke_rows[strip + local] = col_vals[local][cursors[local]];
                        cursors[local] += 1;
                    }
                }
                kept_rows.push(min_row);
            }

            if kept_rows.is_empty() {
                continue;
            }

            // contrib = H_ke_rows · H_ee_inv, over the block's flat
            // column-major values borrowed once.
            let inv = h_ee_inv.block(block_idx);
            contrib.clear();
            contrib.resize(kept_rows.len() * dof, 0.0);
            for r in 0..kept_rows.len() {
                let row = &h_ke_rows[r * dof..(r + 1) * dof];
                for c in 0..dof {
                    let inv_col = &inv[c * dof..(c + 1) * dof];
                    let mut acc = 0.0;
                    for k in 0..dof {
                        acc += row[k] * inv_col[k];
                    }
                    contrib[r * dof + c] = acc;
                }
            }

            // S[i,j] -= contrib[i,:] · H_ke_rows[j,:].
            //
            // The rank-`dof` update is the innermost work in the whole solve,
            // so the 3-DOF case — points in classic BA — gets an unrolled path.
            // The general loop below handles every other size, including mixed
            // ones within the same problem.
            if dof == 3 {
                for (i, &row_i) in kept_rows.iter().enumerate() {
                    let (c0, c1, c2) = (contrib[i * 3], contrib[i * 3 + 1], contrib[i * 3 + 2]);
                    let base = row_i * kept_dof;
                    for (j, &row_j) in kept_rows.iter().enumerate() {
                        let dot = c0 * h_ke_rows[j * 3]
                            + c1 * h_ke_rows[j * 3 + 1]
                            + c2 * h_ke_rows[j * 3 + 2];
                        s_dense[base + row_j] -= dot;
                    }
                }
            } else {
                for (i, &row_i) in kept_rows.iter().enumerate() {
                    let contrib_i = &contrib[i * dof..(i + 1) * dof];
                    let base = row_i * kept_dof;
                    for (j, &row_j) in kept_rows.iter().enumerate() {
                        let h_ke_j = &h_ke_rows[j * dof..(j + 1) * dof];
                        let mut dot = 0.0;
                        for k in 0..dof {
                            dot += contrib_i[k] * h_ke_j[k];
                        }
                        s_dense[base + row_j] -= dot;
                    }
                }
            }
        }

        // Force exact symmetry: accumulation over many blocks drifts.
        for i in 0..kept_dof {
            for j in (i + 1)..kept_dof {
                let avg = (s_dense[i * kept_dof + j] + s_dense[j * kept_dof + i]) * 0.5;
                s_dense[i * kept_dof + j] = avg;
                s_dense[j * kept_dof + i] = avg;
            }
        }

        // Back to sparse, filtering numerical noise. Row-major outer loop keeps
        // the read sequential over the row-major buffer.
        let mut s_triplets: Vec<Triplet<usize, usize, f64>> =
            Vec::with_capacity(kept_dof.saturating_mul(8));
        for row in 0..kept_dof {
            let row_base = row * kept_dof;
            for col in 0..kept_dof {
                let val = s_dense[row_base + col];
                if val.abs() > 1e-12 {
                    s_triplets.push(Triplet::new(row, col, val));
                }
            }
        }

        SparseColMat::try_new_from_triplets(kept_dof, kept_dof, &s_triplets)
            .map_err(|e| LinAlgError::SparseMatrixCreation(format!("Schur S: {:?}", e)))
    }
    /// Reduced right-hand side `g_k − H_ke·H_ee⁻¹·g_e`.
    fn compute_reduced_gradient(
        &self,
        g_k: &Mat<f64>,
        g_e: &Mat<f64>,
        h_ke: &SparseColMat<usize, f64>,
        h_ee_inv: &EliminatedBlocks,
    ) -> LinAlgResult<Mat<f64>> {
        let partition = self.require_partition()?;
        let kept_dof = g_k.nrows();

        // H_ee⁻¹·g_e, blockwise.
        let mut hee_inv_ge = Mat::zeros(g_e.nrows(), 1);
        for block_idx in 0..h_ee_inv.len() {
            let dof = h_ee_inv.dof(block_idx);
            let base = partition.eliminated_offset(block_idx);
            let inv = h_ee_inv.block(block_idx);
            for r in 0..dof {
                let mut acc = 0.0;
                for c in 0..dof {
                    acc += inv[c * dof + r] * g_e[(base + c, 0)];
                }
                hee_inv_ge[(base + r, 0)] = acc;
            }
        }

        // H_ke·(H_ee⁻¹·g_e)
        let mut correction = Mat::<f64>::zeros(kept_dof, 1);
        let symbolic = h_ke.symbolic();
        for col in 0..h_ke.ncols() {
            let rows = symbolic.row_idx_of_col_raw(col);
            let vals = h_ke.val_of_col(col);
            for (idx, &row) in rows.iter().enumerate() {
                correction[(row, 0)] += vals[idx] * hee_inv_ge[(col, 0)];
            }
        }

        let mut g_reduced = Mat::zeros(kept_dof, 1);
        for i in 0..kept_dof {
            g_reduced[(i, 0)] = g_k[(i, 0)] - correction[(i, 0)];
        }
        Ok(g_reduced)
    }
    /// Back-substitute: `δ_e = H_ee⁻¹·(g_e − H_keᵀ·δ_k)`.
    fn back_substitute(
        &self,
        delta_k: &Mat<f64>,
        g_e: &Mat<f64>,
        h_ke: &SparseColMat<usize, f64>,
        h_ee_inv: &EliminatedBlocks,
    ) -> LinAlgResult<Mat<f64>> {
        let partition = self.require_partition()?;
        let eliminated_dof = g_e.nrows();

        // H_keᵀ·δ_k
        let mut hke_t_delta = Mat::<f64>::zeros(eliminated_dof, 1);
        let symbolic = h_ke.symbolic();
        for col in 0..h_ke.ncols() {
            let rows = symbolic.row_idx_of_col_raw(col);
            let vals = h_ke.val_of_col(col);
            let mut acc = 0.0;
            for (idx, &row) in rows.iter().enumerate() {
                acc += vals[idx] * delta_k[(row, 0)];
            }
            hke_t_delta[(col, 0)] = acc;
        }

        let mut delta_e = Mat::zeros(eliminated_dof, 1);
        for block_idx in 0..h_ee_inv.len() {
            let dof = h_ee_inv.dof(block_idx);
            let base = partition.eliminated_offset(block_idx);
            let inv = h_ee_inv.block(block_idx);
            for r in 0..dof {
                let mut acc = 0.0;
                for c in 0..dof {
                    let rhs = g_e[(base + c, 0)] - hke_t_delta[(base + c, 0)];
                    acc += inv[c * dof + r] * rhs;
                }
                delta_e[(base + r, 0)] = acc;
            }
        }

        Ok(delta_e)
    }
}

impl Default for SparseSchurComplementSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl StructureAware for SparseSchurComplementSolver {
    fn initialize_structure(
        &mut self,
        variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        variable_index_map: &SecondaryMap<VarKey, usize>,
        schur_landmark_keys: &std::collections::HashSet<VarKey>,
    ) -> LinAlgResult<()> {
        // Effective landmark set: manual marks plus the SchurOrdering
        // auto-classification, so both the outer structure and the delegate
        // solver partition identically.
        let effective_keys =
            Self::effective_landmark_keys(variables, schur_landmark_keys, &self.ordering);

        self.build_partition(variables, variable_index_map, &effective_keys)?;

        // `SchurVariant::Iterative` is handled by `IterativeSchurSolver`, which
        // the optimizer constructs directly — this solver only ever forms S.
        // A delegate used to be built here and then never read.
        Ok(())
    }
}

impl LinearSolver<SparseMode> for SparseSchurComplementSolver {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        self.require_partition()?;
        if self.variant == SchurVariant::ChunkedSparse {
            return self.solve_chunked(residuals, jacobian, None);
        }

        // 1. Build H = JᵀJ and g = Jᵀr (parallel faer kernels, cached symbolic)
        let NormalEquations { hessian, gradient } = self.ne_cache.compute(residuals, jacobian)?;
        let mut neg_gradient = Mat::zeros(gradient.nrows(), 1);
        for i in 0..gradient.nrows() {
            neg_gradient[(i, 0)] = -gradient[(i, 0)];
        }

        // 2. Split the system. `H_ee` must be block-diagonal for the
        //    elimination to be exact; that is checked, not assumed.
        self.ensure_block_diagonal(&hessian)?;
        let (h_kk, h_ke, g_k, g_e) = {
            let h_kk = self.extract_kept_block(&hessian)?;
            let h_ke = self.extract_coupling_block(&hessian)?;
            let (g_k, g_e) = self.extract_gradient_blocks(&neg_gradient)?;
            (h_kk, h_ke, g_k, g_e)
        };

        // 3. Gather and invert the H_ee diagonal blocks (any DOF, mixed sizes).
        let mut eliminated = std::mem::take(&mut self.eliminated);
        let inversion = (|| -> LinAlgResult<()> {
            let partition = self.require_partition()?;
            eliminated.gather(&hessian, partition);
            eliminated.invert_in_place(partition)
        })();
        if let Err(e) = inversion {
            self.eliminated = eliminated;
            return Err(e);
        }

        // Publish H and g only on success, so neither is cloned — and so a
        // failed solve keeps the previous solve's published system instead of
        // a half-published new one (see the freshness contract on
        // `LinearSolver::get_hessian`).
        let result = self.solve_reduced_system(&h_kk, &h_ke, &g_k, &g_e, &eliminated);
        self.eliminated = eliminated;
        if result.is_ok() {
            self.hessian = Some(hessian);
            self.gradient = Some(gradient);
        }
        result
    }
    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
        damping: &Damping,
    ) -> LinAlgResult<Mat<f64>> {
        self.require_partition()?;
        if self.variant == SchurVariant::ChunkedSparse {
            return self.solve_chunked(residuals, jacobian, Some(damping));
        }

        // 1. Build H = JᵀJ and g = Jᵀr (parallel faer kernels, cached symbolic)
        let NormalEquations { hessian, gradient } = self.ne_cache.compute(residuals, jacobian)?;
        let mut neg_gradient = Mat::zeros(gradient.nrows(), 1);
        for i in 0..gradient.nrows() {
            neg_gradient[(i, 0)] = -gradient[(i, 0)];
        }

        // 2. Split the system, verifying the elimination precondition.
        self.ensure_block_diagonal(&hessian)?;
        let partition = self.require_partition()?;
        let kept_dof = partition.kept_dof();
        let h_kk = self.extract_kept_block(&hessian)?;
        let h_ke = self.extract_coupling_block(&hessian)?;
        let (g_k, g_e) = self.extract_gradient_blocks(&neg_gradient)?;

        debug!(
            "Iteration matrices: H {}×{}, H_kk {}×{}, H_ke {}×{}, {} eliminated blocks",
            hessian.nrows(),
            hessian.ncols(),
            h_kk.nrows(),
            h_kk.ncols(),
            h_ke.nrows(),
            h_ke.ncols(),
            partition.eliminated_blocks().len()
        );

        // 3. λ·D is applied to *both* sides before elimination — damping the
        //    reduced system instead would not be the same problem.
        let h_kk_damped = damp_camera_block(&h_kk, kept_dof, damping)?;

        let mut eliminated = std::mem::take(&mut self.eliminated);
        let prepared = (|| -> LinAlgResult<()> {
            let partition = self.require_partition()?;
            eliminated.gather(&hessian, partition);
            eliminated.damp(damping);
            eliminated.invert_in_place(partition)
        })();
        if let Err(e) = prepared {
            self.eliminated = eliminated;
            return Err(e);
        }

        // Publish the *un-damped* H and g, but only on success: the optimizers
        // build the true quadratic model from these, and a failed solve must
        // keep the previous solve's published system (see the freshness
        // contract on `LinearSolver::get_hessian`).
        let result = self.solve_reduced_system(&h_kk_damped, &h_ke, &g_k, &g_e, &eliminated);
        self.eliminated = eliminated;
        if result.is_ok() {
            self.hessian = Some(hessian);
            self.gradient = Some(gradient);
        }
        result
    }

    fn hessian_vec_product(&self, v: &Mat<f64>) -> Option<Mat<f64>> {
        if let Some(h) = self.hessian.as_ref() {
            return Some(
                <SparseMode as crate::linearizer::AssemblyBackend>::hessian_vec_product(h, v),
            );
        }
        // Chunked path: `JᵀJ` was never formed, so evaluate `Jᵀ(J·v)` from `J`.
        let j = self.chunked_jacobian.as_ref()?;
        let symbolic = j.symbolic();

        let mut jv = Mat::<f64>::zeros(j.nrows(), 1);
        for col in 0..j.ncols() {
            let x = v[(col, 0)];
            if x == 0.0 {
                continue;
            }
            let rows = symbolic.row_idx_of_col_raw(col);
            let vals = j.val_of_col(col);
            for (i, &row) in rows.iter().enumerate() {
                jv[(row, 0)] += vals[i] * x;
            }
        }

        let mut out = Mat::<f64>::zeros(j.ncols(), 1);
        for col in 0..j.ncols() {
            let rows = symbolic.row_idx_of_col_raw(col);
            let vals = j.val_of_col(col);
            let mut acc = 0.0;
            for (i, &row) in rows.iter().enumerate() {
                acc += vals[i] * jv[(row, 0)];
            }
            out[(col, 0)] = acc;
        }
        Some(out)
    }

    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.hessian.as_ref()
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }
}

// Helper methods for SparseSchurComplementSolver
impl SparseSchurComplementSolver {
    /// Chunk-wise solve: `J` straight to the reduced system, no `JᵀJ`.
    ///
    /// The gradient published for the optimizers is `Jᵀr` over the *whole*
    /// system, which the eliminator produces as a by-product; the Hessian is
    /// not published at all, because it is never formed —
    /// [`LinearSolver::hessian_vec_product`] serves the quadratic model as
    /// `Jᵀ(J·v)` instead.
    fn solve_chunked(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
        damping: Option<&Damping>,
    ) -> LinAlgResult<Mat<f64>> {
        // Rebuild the chunk layout only when the sparsity changes.
        if self.chunked.as_ref().is_none_or(|c| !c.matches(jacobian)) {
            let partition = self.require_partition()?;
            self.chunked = Some(super::schur_eliminator::ChunkedSchurEliminator::new(
                jacobian, partition,
            )?);
        }

        let mut eliminator = match self.chunked.take() {
            Some(e) => e,
            None => {
                return Err(LinAlgError::InvalidState(
                    "chunk eliminator not initialized".to_string(),
                )
                .log());
            }
        };

        // The optimizers need +Jᵀr and the action of the un-damped Hessian.
        // Both are published only on success, so a failed solve keeps the
        // previous solve's published system (freshness contract on
        // `LinearSolver::get_hessian`).
        let outcome = (|| -> LinAlgResult<Mat<f64>> {
            let partition = self.require_partition()?;
            let reduced = eliminator.eliminate(jacobian, residuals, partition, damping)?;

            // S is accumulated dense; hand Cholesky a sparse view of it.
            let kept_dof = reduced.kept_dof;
            // faer's `Mat` is column-major, so walk columns outermost to keep
            // the read sequential.
            let mut triplets: Vec<Triplet<usize, usize, f64>> =
                Vec::with_capacity(kept_dof.saturating_mul(8));
            for col in 0..kept_dof {
                for row in 0..kept_dof {
                    let v = reduced.s[(row, col)];
                    if v.abs() > 1e-12 {
                        triplets.push(Triplet::new(row, col, v));
                    }
                }
            }
            let s = SparseColMat::try_new_from_triplets(kept_dof, kept_dof, &triplets)
                .map_err(|e| LinAlgError::SparseMatrixCreation(format!("Schur S: {e:?}")))?;

            // The eliminator returns +g; the reduced system solves S·δ = −g_red.
            let mut rhs = Mat::<f64>::zeros(kept_dof, 1);
            for i in 0..kept_dof {
                rhs[(i, 0)] = -reduced.g_reduced[(i, 0)];
            }
            let delta_k = self.solve_with_cholesky(&s, &rhs)?;

            // δ_e = H_ee⁻¹·(−g_e − H_keᵀ·δ_k); the eliminator already holds
            // H_ee⁻¹·g_e, so only the coupling term is left to apply.
            let delta_e = self.back_substitute_chunked(&delta_k, &reduced, jacobian, partition)?;
            self.combine_updates(&delta_k, &delta_e)
        })();

        self.chunked = Some(eliminator);
        if outcome.is_ok() {
            self.gradient = Some(Self::full_gradient(jacobian, residuals));
            self.hessian = None;
            self.chunked_jacobian = Some(jacobian.clone());
        }
        outcome
    }

    /// `Jᵀr` over the whole system, for the optimizers' gradient.
    ///
    /// Parallel over columns; each column's dot product stays serial, so the
    /// result is bit-identical to the sequential version.
    fn full_gradient(jacobian: &SparseColMat<usize, f64>, residuals: &Mat<f64>) -> Mat<f64> {
        let symbolic = jacobian.symbolic();
        let per_col: Vec<f64> = (0..jacobian.ncols())
            .into_par_iter()
            .map(|col| {
                let rows = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                let mut acc = 0.0;
                for (i, &row) in rows.iter().enumerate() {
                    acc += vals[i] * residuals[(row, 0)];
                }
                acc
            })
            .collect();
        Mat::from_fn(jacobian.ncols(), 1, |r, _| per_col[r])
    }

    /// `δ_e = H_ee⁻¹·(−g_e − H_keᵀ·δ_k)`, evaluated from `J` rather than `H_ke`.
    fn back_substitute_chunked(
        &self,
        delta_k: &Mat<f64>,
        reduced: &super::schur_eliminator::ReducedSystem,
        jacobian: &SparseColMat<usize, f64>,
        partition: &SchurPartition,
    ) -> LinAlgResult<Mat<f64>> {
        let symbolic = jacobian.symbolic();

        // J·δ_k over the retained columns only, giving the coupling term's
        // row-space vector without forming H_ke.
        let mut j_delta = Mat::<f64>::zeros(jacobian.nrows(), 1);
        let mut local = 0usize;
        for block in partition.kept_blocks() {
            for offset in 0..block.dof {
                let col = block.col_start + offset;
                let rows = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                let x = delta_k[(local, 0)];
                if x != 0.0 {
                    for (i, &row) in rows.iter().enumerate() {
                        j_delta[(row, 0)] += vals[i] * x;
                    }
                }
                local += 1;
            }
        }

        let mut delta_e = Mat::<f64>::zeros(partition.eliminated_dof(), 1);
        // Scratch for Eᵀ·(J·δ_k), sized for the largest eliminated block. The
        // partition layer allows arbitrary DOF, so there is no cap: truncating
        // here would silently drop coupling terms and corrupt the step.
        let max_dof = partition
            .eliminated_blocks()
            .iter()
            .map(|block| block.dof)
            .max()
            .unwrap_or(0);
        let mut etjd = vec![0.0f64; max_dof];
        for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
            let dof = block.dof;
            let base = partition.eliminated_offset(block_idx);
            let inv = reduced.eliminated_inverse.block(block_idx);

            // Eᵀ·(J·δ_k) for this chunk.
            for (a, slot) in etjd[..dof].iter_mut().enumerate() {
                let col = block.col_start + a;
                let rows = symbolic.row_idx_of_col_raw(col);
                let vals = jacobian.val_of_col(col);
                let mut acc = 0.0;
                for (i, &row) in rows.iter().enumerate() {
                    acc += vals[i] * j_delta[(row, 0)];
                }
                *slot = acc;
            }

            // δ_e = −H_ee⁻¹·g_e − H_ee⁻¹·Eᵀ(J·δ_k)
            for r in 0..dof {
                let mut acc = -reduced.eliminated_rhs[(base + r, 0)];
                for (c, &term) in etjd[..dof].iter().enumerate() {
                    acc -= inv[c * dof + r] * term;
                }
                delta_e[(base + r, 0)] = acc;
            }
        }

        Ok(delta_e)
    }

    /// Eliminate, solve the reduced system, and back-substitute.
    ///
    /// Shared by the damped and undamped paths, which differ only in whether
    /// `h_kk` and the `H_ee` blocks already carry `λ·D`.
    fn solve_reduced_system(
        &self,
        h_kk: &SparseColMat<usize, f64>,
        h_ke: &SparseColMat<usize, f64>,
        g_k: &Mat<f64>,
        g_e: &Mat<f64>,
        h_ee_inv: &EliminatedBlocks,
    ) -> LinAlgResult<Mat<f64>> {
        let s = self.compute_schur_complement(h_kk, h_ke, h_ee_inv)?;
        let g_reduced = self.compute_reduced_gradient(g_k, g_e, h_ke, h_ee_inv)?;

        let delta_k = match self.variant {
            SchurVariant::ExplicitIterative => self.solve_with_pcg(&s, &g_reduced)?,
            // `Iterative` is the matrix-free solver, which the optimizer
            // constructs directly; standing in with Cholesky here would make
            // the variant a lie.
            SchurVariant::Iterative => {
                return Err(LinAlgError::InvalidInput(
                    "SchurVariant::Iterative is the matrix-free solver, dispatched by the \
                     optimizer to IterativeSchurSolver; SparseSchurComplementSolver handles \
                     only Sparse, ChunkedSparse and ExplicitIterative"
                        .to_string(),
                )
                .log());
            }
            SchurVariant::Sparse | SchurVariant::ChunkedSparse => {
                self.solve_with_cholesky(&s, &g_reduced)?
            }
        };

        let delta_e = self.back_substitute(&delta_k, g_e, h_ke, h_ee_inv)?;
        self.combine_updates(&delta_k, &delta_e)
    }

    /// Scatter the two solution halves back into one full-length update.
    ///
    /// Goes through the partition, so it stays correct when the retained and
    /// eliminated variables interleave in column space.
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

        debug!(
            "Update norms: delta_kept={:.6e}, delta_eliminated={:.6e}, combined={:.6e}",
            delta_k.norm_l2(),
            delta_e.norm_l2(),
            delta.norm_l2()
        );

        Ok(delta)
    }
}

/// `H_cc + λ·D` with `D_jj = clamp(H_jj, min_diagonal, max_diagonal)`.
///
/// Damping only ever touches the diagonal, so when every diagonal entry is
/// already present in the pattern this is a value-only edit: clone the CSC
/// value array and add λ·D at the cached diagonal offsets. That is O(nnz),
/// against the O(cam_size × nnz) linear `find` over a freshly built triplet
/// list that this replaces, and it skips the triplet sort entirely.
///
/// A camera observed by no landmark has a structurally empty diagonal, which
/// damping must materialize. That grows the pattern, so it falls back to
/// rebuilding from triplets — still a single O(nnz) pass, with the missing
/// diagonals appended.
///
/// Mirrors [`NormalEquationsCache::damped_hessian`](crate::linalg::sparse::normal_eq)
/// so both damping sites share one rule.
fn damp_camera_block(
    h_cc: &SparseColMat<usize, f64>,
    cam_size: usize,
    damping: &Damping,
) -> LinAlgResult<SparseColMat<usize, f64>> {
    let symbolic = h_cc.symbolic();

    // Absolute offsets of each column's diagonal entry in the flat value array.
    let diag_pos: Vec<Option<usize>> = (0..h_cc.ncols())
        .map(|col| {
            let col_start = symbolic.col_range(col).start;
            symbolic
                .row_idx_of_col_raw(col)
                .iter()
                .position(|&row| row == col)
                .map(|local| col_start + local)
        })
        .collect();

    if diag_pos.iter().all(Option::is_some) {
        let owned = symbolic.to_owned().map_err(|e| {
            LinAlgError::SparseMatrixCreation(format!("Damped H_cc symbolic: {e:?}"))
        })?;
        let mut values = h_cc.as_ref().val().to_vec();
        for pos in diag_pos.iter().flatten() {
            values[*pos] += damping.diagonal_term(values[*pos]);
        }
        return Ok(SparseColMat::new(owned, values));
    }

    // Slow path: at least one diagonal must be created.
    let mut triplets = Vec::with_capacity(h_cc.compute_nnz() + cam_size);
    for col in 0..h_cc.ncols() {
        let row_indices = symbolic.row_idx_of_col_raw(col);
        let col_values = h_cc.val_of_col(col);
        for (idx, &row) in row_indices.iter().enumerate() {
            let val = col_values[idx];
            let val = if row == col {
                val + damping.diagonal_term(val)
            } else {
                val
            };
            triplets.push(Triplet::new(row, col, val));
        }
    }
    for (i, pos) in diag_pos.iter().enumerate().take(cam_size) {
        if pos.is_none() {
            // H_ii is structurally zero, so the clamp floors the damping at
            // λ·min_diagonal.
            triplets.push(Triplet::new(i, i, damping.diagonal_term(0.0)));
        }
    }
    SparseColMat::try_new_from_triplets(cam_size, cam_size, &triplets)
        .map_err(|e| LinAlgError::SparseMatrixCreation(format!("Damped H_cc: {e:?}")))
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

    // Type alias for the test setup tuple
    type TestSetup = (
        SlotMap<VarKey, Box<dyn ManifoldVariable>>,
        SecondaryMap<VarKey, usize>,
        SparseColMat<usize, f64>,
        Mat<f64>,
        std::collections::HashSet<VarKey>,
    );

    /// Build a minimal BA-style test setup:
    /// 2 SE3 cameras + 3 Rn landmarks
    /// Jacobian: 36 rows × 21 cols
    ///
    /// Structure guarantees H_cc = 3·I₁₂ and H_pp = 4·I₃ (positive definite).
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

        // cam0 → 0..5, cam1 → 6..11, pt0 → 12..14, pt1 → 15..17, pt2 → 18..20
        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(cam0, 0);
        variable_index_map.insert(cam1, 6);
        variable_index_map.insert(pt0, 12);
        variable_index_map.insert(pt1, 15);
        variable_index_map.insert(pt2, 18);

        // Jacobian: 2 cameras × 3 landmarks × 6 rows_per_obs = 36 rows, 21 cols
        // For observation (cam_i, pt_j), row_base = (ci * 3 + li) * 6
        //   J[row_base+k, cam_col+k] = 1.0  (k=0..5, camera DOF)
        //   J[row_base+k, lm_col + (k%3)] = 1.0  (landmark DOF repeats to fill all 3)
        let n_rows = 36;
        let n_cols = 21;
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

        let jacobian = SparseColMat::try_new_from_triplets(n_rows, n_cols, &triplets)?;
        let residuals = Mat::from_fn(n_rows, 1, |i, _| (i % 5) as f64 * 0.1);

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

    /// The reduced camera matrix against naive dense algebra.
    ///
    /// Pins the `dof == 3` unrolled path, the general loop, the flat
    /// column-major `EliminatedBlocks` indexing and the exact-symmetrization
    /// step against one independent `S = H_kk − H_ke·H_ee⁻¹·H_keᵀ` computed
    /// with dense nalgebra — so a future edit breaking any single
    /// representation fails here even if every other path still agrees.
    #[test]
    fn schur_complement_matches_naive_dense() -> TestResult {
        let (variables, variable_index_map, jacobian, _residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = SparseSchurComplementSolver::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        // Dense J, then H = JᵀJ.
        let nrows = jacobian.nrows();
        let ncols = jacobian.ncols();
        let mut dense_j = nalgebra::DMatrix::zeros(nrows, ncols);
        for col in 0..ncols {
            let rows = jacobian.symbolic().row_idx_of_col_raw(col);
            let vals = jacobian.val_of_col(col);
            for (i, &row) in rows.iter().enumerate() {
                dense_j[(row, col)] = vals[i];
            }
        }
        let h = dense_j.transpose() * &dense_j;

        // Kept vs eliminated columns from the same index map the solver used.
        let mut elim_cols = Vec::new();
        for key in &landmark_keys {
            let start = variable_index_map.get(*key).ok_or("elim key missing")?;
            let dof = variables.get(*key).ok_or("elim var missing")?.dof();
            elim_cols.extend(*start..*start + dof);
        }
        elim_cols.sort_unstable();
        let kept_cols: Vec<usize> = (0..ncols).filter(|c| !elim_cols.contains(c)).collect();

        let submatrix = |rows: &[usize], cols: &[usize]| -> nalgebra::DMatrix<f64> {
            nalgebra::DMatrix::from_fn(rows.len(), cols.len(), |r, c| h[(rows[r], cols[c])])
        };
        let h_kk = submatrix(&kept_cols, &kept_cols);
        let h_ke = submatrix(&kept_cols, &elim_cols);
        let h_ee = submatrix(&elim_cols, &elim_cols);
        let h_ee_inv = h_ee.try_inverse().ok_or("naive H_ee must invert")?;
        let s_naive = h_kk - &h_ke * h_ee_inv * h_ke.transpose();

        // Solver's S through its own extraction + arena + complement path,
        // sharing one H so the comparison is legible.
        let h_full = {
            let mut cache =
                crate::linalg::sparse::normal_eq::NormalEquationsCache::try_new(&jacobian)?;
            let residuals = Mat::zeros(nrows, 1);
            cache.compute(&residuals, &jacobian)?.hessian
        };
        let h_kk2 = solver.extract_kept_block(&h_full)?;
        let h_ke2 = solver.extract_coupling_block(&h_full)?;
        let mut blocks = EliminatedBlocks::new(solver.partition().ok_or("partition missing")?);
        blocks.gather(&h_full, solver.partition().ok_or("partition missing")?);
        blocks.invert_in_place(solver.partition().ok_or("partition missing")?)?;
        let s = solver.compute_schur_complement(&h_kk2, &h_ke2, &blocks)?;

        // Exact symmetry, then value agreement with naive dense math.
        // Entries the solver filters (|v| <= 1e-12) read back as 0.0, which
        // stays inside the 1e-9 comparison tolerance by construction.
        let s_val = |i: usize, j: usize| -> f64 {
            let rows = s.symbolic().row_idx_of_col_raw(j);
            let vals = s.val_of_col(j);
            rows.iter()
                .position(|&r| r == i)
                .map(|p| vals[p])
                .unwrap_or(0.0)
        };
        let k = kept_cols.len();
        assert_eq!((s.nrows(), s.ncols()), (k, k));
        for i in 0..k {
            for j in 0..k {
                let (a, b) = (s_val(i, j), s_val(j, i));
                assert!(
                    (a - b).abs() < 1e-12,
                    "S not symmetric at ({i},{j}): {a} vs {b}"
                );
                assert!(
                    (a - s_naive[(i, j)]).abs() < 1e-9,
                    "S[{i},{j}] = {a}, naive dense = {}",
                    s_naive[(i, j)]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_schur_ordering_rn3_eliminated() {
        let ordering = SchurOrdering::default();
        // Default: RN(3) is eliminated (landmark)
        assert!(ordering.should_eliminate(&ManifoldType::RN, 3));
    }

    #[test]
    fn test_schur_ordering_se3_not_eliminated() {
        let ordering = SchurOrdering::default();
        // SE3 variables are never eliminated (camera)
        assert!(!ordering.should_eliminate(&ManifoldType::SE3, 6));
    }

    #[test]
    fn test_schur_ordering_wrong_type_not_eliminated() {
        let ordering = SchurOrdering::default();
        // SE3(6) is not in eliminate_types
        assert!(!ordering.should_eliminate(&ManifoldType::SE3, 6));
    }

    #[test]
    fn test_schur_ordering_wrong_size_not_eliminated() {
        let ordering = SchurOrdering::default();
        // RN with size != 3 is not eliminated
        assert!(!ordering.should_eliminate(&ManifoldType::RN, 6));
        assert!(!ordering.should_eliminate(&ManifoldType::RN, 2));
    }

    #[test]
    fn test_schur_ordering_by_name_matches_type() {
        let ordering = SchurOrdering::default();
        assert!(ordering.should_eliminate_by_name("Rn", 3));
        assert!(!ordering.should_eliminate_by_name("SE3", 6));
        assert!(!ordering.should_eliminate_by_name("NotAManifold", 3));
    }

    #[test]
    fn test_auto_elimination_is_opt_in() -> Result<(), Box<dyn std::error::Error>> {
        use crate::core::variable::Variable;
        use apex_manifolds::{rn, se3};
        use nalgebra::DVector;
        use slotmap::SlotMap;

        // Two SE3 cameras and two Rn(3) landmarks, none marked manually.
        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let se3_data = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let cam0 = variables.insert(Box::new(Variable::new(se3::SE3::from_param_slice(
            se3_data.as_slice(),
        ))));
        let cam1 = variables.insert(Box::new(Variable::new(se3::SE3::from_param_slice(
            se3_data.as_slice(),
        ))));
        let pt_data = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let pt0 = variables.insert(Box::new(Variable::new(rn::Rn::new(pt_data.clone()))));
        let pt1 = variables.insert(Box::new(Variable::new(rn::Rn::new(pt_data.clone()))));

        let mut index_map = SecondaryMap::new();
        index_map.insert(cam0, 0);
        index_map.insert(cam1, 6);
        index_map.insert(pt0, 12);
        index_map.insert(pt1, 15);

        let empty_marks = std::collections::HashSet::new();

        // Default ordering: auto-classification stays off — Rn(3) is ambiguous
        // (landmarks vs self-calibration intrinsics), so only pt0 is eliminated
        // and the unmarked pt1 stays a camera.
        let mut marks = std::collections::HashSet::new();
        marks.insert(pt0);
        let mut solver = SparseSchurComplementSolver::new();
        solver.initialize_structure(&variables, &index_map, &marks)?;
        let structure = solver.partition().ok_or("partition missing")?;
        assert_eq!(
            structure.eliminated_blocks().len(),
            1,
            "auto-detection must be off by default"
        );
        assert_eq!(structure.kept_blocks().len(), 3);
        assert!(structure.kept_blocks().iter().any(|b| b.key == pt1));

        // Opt-in: unmarked Rn(3) variables are eliminated as landmarks.
        let ordering = SchurOrdering::default().with_auto_detect(true);
        let mut solver = SparseSchurComplementSolver::new().with_ordering(ordering);
        solver.initialize_structure(&variables, &index_map, &empty_marks)?;
        let structure = solver.partition().ok_or("partition missing")?;
        assert_eq!(
            structure.eliminated_blocks().len(),
            2,
            "Rn(3) variables must auto-eliminate"
        );
        assert_eq!(
            structure.kept_blocks().len(),
            2,
            "SE3 variables must stay cameras"
        );

        // Manual marks still eliminate regardless of type/size.
        let mut marks = std::collections::HashSet::new();
        marks.insert(cam1);
        let mut solver = SparseSchurComplementSolver::new();
        solver.initialize_structure(&variables, &index_map, &marks)?;
        let structure = solver.partition().ok_or("partition missing")?;
        assert!(structure.eliminated_blocks().iter().any(|b| b.key == cam1));

        Ok(())
    }
    /// A partition needs both sides populated; the shape itself is covered in
    /// depth by `schur_partition`'s own tests.
    #[test]
    fn test_partition_requires_both_sides() {
        assert!(SchurPartition::new(Vec::new(), Vec::new()).is_err());
    }

    #[test]
    fn test_solver_creation() {
        let solver = SparseSchurComplementSolver::new();
        assert!(solver.partition().is_none());
    }

    #[test]
    fn test_schur_variants() {
        let solver = SparseSchurComplementSolver::new()
            .with_variant(SchurVariant::Iterative)
            .with_preconditioner(SchurPreconditioner::BlockDiagonal)
            .with_cg_params(100, 1e-8);

        assert_eq!(solver.cg_max_iterations, 100);
        assert!((solver.cg_tolerance - 1e-8).abs() < 1e-12);
    }
    /// Solver with a partition of `kept_dof` retained columns followed by one
    /// eliminated block of `dof`, plus that block's inverse preloaded.
    fn solver_with_block(
        kept_dof: usize,
        dof: usize,
        inverse: &[f64],
    ) -> Result<(SparseSchurComplementSolver, EliminatedBlocks), LinAlgError> {
        let kept = (0..kept_dof)
            .map(|i| BlockSpan {
                key: VarKey::default(),
                col_start: i,
                dof: 1,
            })
            .collect();
        let eliminated = vec![BlockSpan {
            key: VarKey::default(),
            col_start: kept_dof,
            dof,
        }];
        let partition = SchurPartition::new(kept, eliminated)?;
        let mut blocks = EliminatedBlocks::new(&partition);
        blocks.block_mut(0).copy_from_slice(inverse);
        let mut solver = SparseSchurComplementSolver::new();
        solver.partition = Some(partition);
        Ok((solver, blocks))
    }

    /// S = H_kk − H_ke·H_ee⁻¹·H_keᵀ against hand-computed values, unchanged
    /// from before the generalization.
    #[test]
    fn test_compute_schur_complement_known_matrix() -> Result<(), LinAlgError> {
        use faer::sparse::Triplet;

        // H_ee⁻¹ = 0.5·I₃
        let inv = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5];
        let (solver, blocks) = solver_with_block(2, 3, &inv)?;

        let h_kk = SparseColMat::try_new_from_triplets(
            2,
            2,
            &[Triplet::new(0, 0, 4.0), Triplet::new(1, 1, 5.0)],
        )
        .map_err(|e| LinAlgError::SparseMatrixCreation(format!("{e:?}")))?;
        let h_ke = SparseColMat::try_new_from_triplets(
            2,
            3,
            &[Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 2.0)],
        )
        .map_err(|e| LinAlgError::SparseMatrixCreation(format!("{e:?}")))?;

        let s = solver.compute_schur_complement(&h_kk, &h_ke, &blocks)?;

        assert_eq!(s.nrows(), 2);
        assert_eq!(s.ncols(), 2);

        // S(0,0) = 4 − 1·0.5·1 = 3.5 ; S(1,1) = 5 − 2·0.5·2 = 3.0
        let dense = dense_of(&s);
        assert!((dense[0][0] - 3.5).abs() < 1e-12, "got {}", dense[0][0]);
        assert!((dense[1][1] - 3.0).abs() < 1e-12, "got {}", dense[1][1]);
        Ok(())
    }

    #[test]
    fn test_back_substitute() -> Result<(), LinAlgError> {
        use faer::sparse::Triplet;

        // H_ee⁻¹ = I₃
        let inv = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (solver, blocks) = solver_with_block(2, 3, &inv)?;

        let delta_c = Mat::from_fn(2, 1, |i, _| (i + 1) as f64); // [1; 2]
        let g_p = Mat::from_fn(3, 1, |i, _| (i + 1) as f64); // [1; 2; 3]

        let h_cp = SparseColMat::try_new_from_triplets(
            2,
            3,
            &[Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        )
        .map_err(|e| LinAlgError::SparseMatrixCreation(format!("{e:?}")))?;

        // Compute δp = H_pp^{-1} * (g_p - H_cp^T * δc)
        // H_cp^T * δc = [1*1; 1*2; 0] = [1; 2; 0]
        // g_p - result = [1; 2; 3] - [1; 2; 0] = [0; 0; 3]
        // H_pp^{-1} * [0; 0; 3] = [0; 0; 3]
        let delta_p = solver.back_substitute(&delta_c, &g_p, &h_cp, &blocks)?;

        assert_eq!(delta_p.nrows(), 3);
        assert!((delta_p[(0, 0)]).abs() < 1e-10);
        assert!((delta_p[(1, 0)]).abs() < 1e-10);
        assert!((delta_p[(2, 0)] - 3.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_compute_reduced_gradient() -> Result<(), LinAlgError> {
        use faer::sparse::Triplet;

        // H_ee⁻¹ = 2·I₃
        let inv = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        let (solver, blocks) = solver_with_block(2, 3, &inv)?;

        let g_c = Mat::from_fn(2, 1, |i, _| (i + 1) as f64); // [1; 2]
        let g_p = Mat::from_fn(3, 1, |i, _| (i + 1) as f64); // [1; 2; 3]

        let h_cp = SparseColMat::try_new_from_triplets(
            2,
            3,
            &[Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        )
        .map_err(|e| LinAlgError::SparseMatrixCreation(format!("{e:?}")))?;

        // Compute g_reduced = g_c - H_cp * H_pp^{-1} * g_p
        // H_pp^{-1} * g_p = 2*[1; 2; 3] = [2; 4; 6]
        // H_cp * [2; 4; 6] = [1*2; 1*4] = [2; 4]
        // g_reduced = [1; 2] - [2; 4] = [-1; -2]
        let g_reduced = solver.compute_reduced_gradient(&g_c, &g_p, &h_cp, &blocks)?;

        assert_eq!(g_reduced.nrows(), 2);
        assert!((g_reduced[(0, 0)] + 1.0).abs() < 1e-10);
        assert!((g_reduced[(1, 0)] + 2.0).abs() < 1e-10);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // New tests for uncovered code paths
    // -------------------------------------------------------------------------

    /// Test SparseSchurComplementSolver::default() equals new()
    #[test]
    fn test_solver_default() {
        let solver = SparseSchurComplementSolver::default();
        assert!(solver.partition().is_none());
        assert!(solver.hessian.is_none());
        assert!(solver.gradient.is_none());
    }

    /// Test partition() getter after initialize_structure
    #[test]
    fn test_partition_getter() -> TestResult {
        let (variables, variable_index_map, _, _, landmark_keys) = create_schur_test_setup()?;
        let mut solver = SparseSchurComplementSolver::new();

        assert!(solver.partition().is_none());
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        assert!(solver.partition().is_some());
        Ok(())
    }

    /// Test with_ordering() builder stores the custom ordering
    #[test]
    fn test_with_ordering_builder() {
        let ordering = SchurOrdering {
            eliminate_types: vec![ManifoldType::RN],
            eliminate_rn_size: Some(3),
            auto_detect: false,
        };
        let solver = SparseSchurComplementSolver::new().with_ordering(ordering);
        assert_eq!(solver.ordering.eliminate_rn_size, Some(3));
    }
    /// Diagonal 3-DOF block inversion, the classic-BA case, through the
    /// generalized arena. Sizes 1/2/4/6/9 are covered in `schur_partition`.
    #[test]
    fn test_eliminated_block_inversion_3dof() -> TestResult {
        // diag(2, 3, 4) → diag(0.5, 1/3, 0.25)
        let inv_seed = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        let kept = vec![BlockSpan {
            key: VarKey::default(),
            col_start: 0,
            dof: 1,
        }];
        let eliminated = vec![BlockSpan {
            key: VarKey::default(),
            col_start: 1,
            dof: 3,
        }];
        let partition = SchurPartition::new(kept, eliminated)?;
        let mut blocks = EliminatedBlocks::new(&partition);
        blocks.block_mut(0).copy_from_slice(&inv_seed);
        blocks.invert_in_place(&partition)?;

        assert!((blocks.at(0, 0, 0) - 0.5).abs() < 1e-10);
        assert!((blocks.at(0, 1, 1) - 1.0 / 3.0).abs() < 1e-10);
        assert!((blocks.at(0, 2, 2) - 0.25).abs() < 1e-10);
        Ok(())
    }

    /// Test initialize_structure() correctly partitions 2 cameras + 3 landmarks
    #[test]
    fn test_explicit_schur_initialize_structure() -> TestResult {
        let (variables, variable_index_map, _, _, landmark_keys) = create_schur_test_setup()?;
        let mut solver = SparseSchurComplementSolver::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let bs = solver.partition().ok_or("partition is None")?;
        assert_eq!(bs.kept_blocks().len(), 2);
        assert_eq!(bs.eliminated_blocks().len(), 3);
        assert_eq!(bs.kept_dof(), 12); // 2 × 6
        assert_eq!(bs.eliminated_dof(), 9); // 3 × 3
        Ok(())
    }

    /// Test extract_gradient_blocks() splits gradient correctly
    #[test]
    fn test_extract_gradient_blocks() -> TestResult {
        let (variables, variable_index_map, _, _, landmark_keys) = create_schur_test_setup()?;
        let mut solver = SparseSchurComplementSolver::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        // Gradient over full variable space (21 DOF)
        let gradient = Mat::from_fn(21, 1, |i, _| i as f64);
        let (g_c, g_p) = solver.extract_gradient_blocks(&gradient)?;

        assert_eq!(g_c.nrows(), 12); // camera DOF
        assert_eq!(g_p.nrows(), 9); // landmark DOF
        Ok(())
    }

    /// Test full Schur solve pipeline with Sparse (Cholesky) variant
    #[test]
    fn test_explicit_schur_solve_normal_equation() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = SparseSchurComplementSolver::new().with_variant(SchurVariant::Sparse);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        let delta =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        assert_eq!(delta.nrows(), 21);
        assert_eq!(delta.ncols(), 1);
        Ok(())
    }

    /// Test full Schur augmented solve (LM damping) with Sparse variant
    #[test]
    fn test_explicit_schur_solve_augmented_equation() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = SparseSchurComplementSolver::new().with_variant(SchurVariant::Sparse);
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

    /// Test Schur solve with Iterative (PCG) variant exercises solve_with_pcg path
    #[test]
    fn test_explicit_schur_solve_iterative_variant() -> TestResult {
        let (variables, variable_index_map, _jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = SparseSchurComplementSolver::new()
            .with_variant(SchurVariant::Iterative)
            .with_cg_params(200, 1e-6);
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        // `Iterative` is the matrix-free solver; handing it to this solver
        // must be an error, not a silent Cholesky fallback.
        let result =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &_jacobian);
        let Err(err) = result else {
            panic!("Iterative must not silently run Cholesky on the formed S");
        };
        assert!(err.to_string().contains("matrix-free"), "{err}");
        Ok(())
    }

    /// Test get_hessian() and get_gradient() trait methods after solve
    #[test]
    fn test_explicit_schur_get_hessian_gradient() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = SparseSchurComplementSolver::new();
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

    /// Test two solves with different λ produce different updates
    #[test]
    fn test_explicit_schur_augmented_lambda_effect() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;

        let mut solver1 = SparseSchurComplementSolver::new();
        solver1.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let delta1 = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver1,
            &residuals,
            &jacobian,
            &Damping::identity(0.001),
        )?;

        let mut solver2 = SparseSchurComplementSolver::new();
        solver2.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let delta2 = LinearSolver::<SparseMode>::solve_augmented_equation(
            &mut solver2,
            &residuals,
            &jacobian,
            &Damping::identity(100.0),
        )?;

        // Different λ values should produce different updates
        let norm_diff: f64 = (0..21)
            .map(|i| (delta1[(i, 0)] - delta2[(i, 0)]).powi(2))
            .sum();
        assert!(
            norm_diff > 1e-10,
            "Different λ should yield different updates"
        );
        Ok(())
    }

    /// Test combine_updates() merges camera and landmark deltas at correct offsets
    #[test]
    fn test_combine_updates() -> TestResult {
        let (variables, variable_index_map, _, _, landmark_keys) = create_schur_test_setup()?;
        let mut solver = SparseSchurComplementSolver::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        // Camera delta: 12×1, landmark delta: 9×1
        let delta_c = Mat::from_fn(12, 1, |_, _| 1.0);
        let delta_p = Mat::from_fn(9, 1, |_, _| 2.0);

        let combined = solver.combine_updates(&delta_c, &delta_p)?;
        assert_eq!(combined.nrows(), 21);

        // Camera values (cam_start..cam_end = 0..12)
        for i in 0..12 {
            assert!((combined[(i, 0)] - 1.0).abs() < 1e-10);
        }
        // Landmark values (land_start..land_end = 12..21)
        for i in 12..21 {
            assert!((combined[(i, 0)] - 2.0).abs() < 1e-10);
        }
        Ok(())
    }

    /// Test solve without initialize_structure returns error
    #[test]
    fn test_explicit_schur_solve_without_init_returns_error() -> TestResult {
        let triplets: Vec<Triplet<usize, usize, f64>> = vec![Triplet::new(0, 0, 1.0)];
        let jacobian =
            SparseColMat::try_new_from_triplets(1, 1, &triplets).map_err(|e| format!("{e:?}"))?;
        let residuals = Mat::from_fn(1, 1, |_, _| 1.0);
        let mut solver = SparseSchurComplementSolver::new();

        let result =
            LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian);
        assert!(result.is_err());
        Ok(())
    }

    // -------------------------------------------------------------------------
    // New tests for previously uncovered code paths
    // -------------------------------------------------------------------------

    /// Test SchurOrdering::new() produces the same result as default().
    #[test]
    fn test_schur_ordering_new_equals_default() {
        let a = SchurOrdering::new();
        let b = SchurOrdering::default();
        assert_eq!(a.eliminate_rn_size, b.eliminate_rn_size);
        assert_eq!(a.eliminate_types.len(), b.eliminate_types.len());
    }

    /// Test extract_camera_block produces a square matrix of camera DOF.
    #[test]
    fn test_extract_camera_block() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = SparseSchurComplementSolver::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        // Build Hessian H = J^T J
        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        let hessian = solver.hessian.clone().ok_or("hessian is None")?;

        let mut fresh = SparseSchurComplementSolver::new();
        fresh.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let h_cc = fresh.extract_kept_block(&hessian)?;

        // camera DOF = 12 (2 cameras × 6)
        assert_eq!(h_cc.nrows(), 12);
        assert_eq!(h_cc.ncols(), 12);
        Ok(())
    }

    /// Test extract_coupling_block produces a matrix with camera rows × landmark cols.
    #[test]
    fn test_extract_coupling_block() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = SparseSchurComplementSolver::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        let hessian = solver.hessian.clone().ok_or("hessian is None")?;

        let mut fresh = SparseSchurComplementSolver::new();
        fresh.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let h_cp = fresh.extract_coupling_block(&hessian)?;

        // H_cp: camera DOF rows × landmark DOF cols = 12 × 9
        assert_eq!(h_cp.nrows(), 12);
        assert_eq!(h_cp.ncols(), 9);
        Ok(())
    }

    /// Gathering H_ee yields one block per eliminated variable, sized by its DOF.
    #[test]
    fn test_gather_eliminated_blocks() -> TestResult {
        let (variables, variable_index_map, jacobian, residuals, landmark_keys) =
            create_schur_test_setup()?;
        let mut solver = SparseSchurComplementSolver::new();
        solver.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;

        LinearSolver::<SparseMode>::solve_normal_equation(&mut solver, &residuals, &jacobian)?;
        let hessian = solver.hessian.clone().ok_or("hessian is None")?;

        let mut fresh = SparseSchurComplementSolver::new();
        fresh.initialize_structure(&variables, &variable_index_map, &landmark_keys)?;
        let partition = fresh.partition().ok_or("partition is None")?;
        let mut blocks = EliminatedBlocks::new(partition);
        blocks.gather(&hessian, partition);

        // 3 landmarks → 3 blocks, each 3 DOF
        assert_eq!(blocks.len(), 3);
        for i in 0..blocks.len() {
            assert_eq!(blocks.dof(i), 3);
        }
        Ok(())
    }

    /// Test solve_with_cholesky satisfies Ax ≈ b for a known SPD system.
    ///
    /// Regression probe (diagnostic, kept): assemble a real SE2 pose chain
    /// through `Problem` — grouping, workspace, sparse assembly — and require
    /// the Schur step to equal the Cholesky step on that assembled system,
    /// twice in a row with one solver instance. Hand-built Jacobians cannot
    /// catch assembly-ordering interactions; this one can.
    #[test]
    fn schur_matches_cholesky_on_assembled_pose_chain() -> TestResult {
        use crate::core::problem::Problem;
        use crate::factors::{BetweenFactor, PriorFactor};
        use crate::linalg::JacobianMode;
        use crate::linalg::sparse::cholesky::SparseCholeskySolver;
        use apex_manifolds::{ManifoldType, se2::SE2};
        use nalgebra::dvector;

        let mut problem = Problem::new(JacobianMode::Sparse);
        let mut keys = Vec::new();
        for i in 0..5 {
            keys.push(
                problem.add_variable(ManifoldType::SE2, dvector![i as f64 + 0.1, 0.05, 0.02]),
            );
        }
        for w in keys.windows(2) {
            problem.add_residual_block(
                &[w[0], w[1]],
                Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.0))),
                None,
            );
        }
        problem.add_residual_block(
            &[keys[0]],
            Box::new(PriorFactor::new(SE2::from_xy_angle(0.0, 0.0, 0.0))),
            None,
        );
        problem.mark_for_elimination(keys[1]);
        problem.mark_for_elimination(keys[3]);
        problem.group_rows_for_elimination();

        let state = crate::optimizer::initialize_optimization_state(&mut problem)?;
        let symbolic = state.symbolic_structure.as_ref().ok_or("sparse symbolic")?;
        let mut workspace = crate::linearizer::AssemblyWorkspace::build(&problem);
        let (residuals, jacobian) = crate::linearizer::cpu::sparse::assemble_sparse(
            &problem,
            &state.variables,
            &state.variable_index_map,
            symbolic,
            &mut workspace,
        )?;

        let mut eliminate = std::collections::HashSet::new();
        eliminate.insert(keys[1]);
        eliminate.insert(keys[3]);

        let mut cholesky = SparseCholeskySolver::new();
        let reference =
            crate::linalg::LinearSolver::<crate::linalg::SparseMode>::solve_normal_equation(
                &mut cholesky,
                &residuals,
                &jacobian,
            )?;

        let mut schur = SparseSchurComplementSolver::new();
        schur.initialize_structure(&state.variables, &state.variable_index_map, &eliminate)?;
        let first =
            crate::linalg::LinearSolver::<crate::linalg::SparseMode>::solve_normal_equation(
                &mut schur, &residuals, &jacobian,
            )?;
        let second =
            crate::linalg::LinearSolver::<crate::linalg::SparseMode>::solve_normal_equation(
                &mut schur, &residuals, &jacobian,
            )?;

        let scale = reference.norm_l2().max(1.0);
        for (label, step) in [("first", &first), ("second", &second)] {
            assert_eq!(reference.nrows(), step.nrows(), "{label}: length");
            for i in 0..reference.nrows() {
                let diff = (reference[(i, 0)] - step[(i, 0)]).abs();
                assert!(
                    diff / scale < 1e-9,
                    "{label}: component {i} differs — cholesky {}, schur {} (rel {:.3e})",
                    reference[(i, 0)],
                    step[(i, 0)],
                    diff / scale
                );
            }
        }
        Ok(())
    }

    /// Diagnostic multi-iteration probe (kept): replicate three Gauss-Newton
    /// iterations — assemble, solve with both solvers, compare, apply the
    /// Cholesky step — to locate the first iteration where the Schur step
    /// stops matching on evolving (not fixed) linearizations.
    #[test]
    fn schur_tracks_cholesky_across_iterations() -> TestResult {
        use crate::core::problem::Problem;
        use crate::factors::{BetweenFactor, PriorFactor};
        use crate::linalg::JacobianMode;
        use crate::linalg::sparse::cholesky::SparseCholeskySolver;
        use apex_manifolds::{ManifoldType, se2::SE2};
        use nalgebra::dvector;

        let mut problem = Problem::new(JacobianMode::Sparse);
        let mut keys = Vec::new();
        for i in 0..5 {
            keys.push(
                problem.add_variable(ManifoldType::SE2, dvector![i as f64 + 0.1, 0.05, 0.02]),
            );
        }
        for w in keys.windows(2) {
            problem.add_residual_block(
                &[w[0], w[1]],
                Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.0))),
                None,
            );
        }
        problem.add_residual_block(
            &[keys[0]],
            Box::new(PriorFactor::new(SE2::from_xy_angle(0.0, 0.0, 0.0))),
            None,
        );
        problem.mark_for_elimination(keys[1]);
        problem.mark_for_elimination(keys[3]);
        problem.group_rows_for_elimination();

        let mut state = crate::optimizer::initialize_optimization_state(&mut problem)?;
        let mut eliminate = std::collections::HashSet::new();
        eliminate.insert(keys[1]);
        eliminate.insert(keys[3]);

        let mut cholesky = SparseCholeskySolver::new();
        let mut schur = SparseSchurComplementSolver::new();
        schur.initialize_structure(&state.variables, &state.variable_index_map, &eliminate)?;

        for iter in 0..3 {
            let symbolic = state.symbolic_structure.as_ref().ok_or("sparse symbolic")?;
            let mut workspace = crate::linearizer::AssemblyWorkspace::build(&problem);
            let (residuals, jacobian) = crate::linearizer::cpu::sparse::assemble_sparse(
                &problem,
                &state.variables,
                &state.variable_index_map,
                symbolic,
                &mut workspace,
            )?;
            let reference =
                crate::linalg::LinearSolver::<crate::linalg::SparseMode>::solve_normal_equation(
                    &mut cholesky,
                    &residuals,
                    &jacobian,
                )?;
            let step =
                crate::linalg::LinearSolver::<crate::linalg::SparseMode>::solve_normal_equation(
                    &mut schur, &residuals, &jacobian,
                )?;
            let scale = reference.norm_l2().max(1.0);
            let mut max_rel: f64 = 0.0;
            for i in 0..reference.nrows() {
                max_rel = max_rel.max((reference[(i, 0)] - step[(i, 0)]).abs() / scale);
            }
            assert!(
                max_rel < 1e-9,
                "iteration {iter}: Schur step diverges from Cholesky (max rel {max_rel:.3e})"
            );
            crate::optimizer::apply_parameter_step(
                &mut state.variables,
                step.as_ref(),
                &state.sorted_vars,
            );
        }
        Ok(())
    }

    /// Diagnostic multi-iteration probe, exact-loop replica (kept): same as
    /// above but reusing ONE workspace across iterations, exactly like
    /// `iteration_preamble` does — instead of rebuilding it per iteration.
    /// Diverges here but not above ⟹ workspace-reuse assembly bug.
    #[test]
    fn schur_tracks_cholesky_with_shared_workspace() -> TestResult {
        use crate::core::problem::Problem;
        use crate::factors::{BetweenFactor, PriorFactor};
        use crate::linalg::JacobianMode;
        use crate::linalg::sparse::cholesky::SparseCholeskySolver;
        use apex_manifolds::{ManifoldType, se2::SE2};
        use nalgebra::dvector;

        let mut problem = Problem::new(JacobianMode::Sparse);
        let mut keys = Vec::new();
        for i in 0..5 {
            keys.push(
                problem.add_variable(ManifoldType::SE2, dvector![i as f64 + 0.1, 0.05, 0.02]),
            );
        }
        for w in keys.windows(2) {
            problem.add_residual_block(
                &[w[0], w[1]],
                Box::new(BetweenFactor::new(SE2::from_xy_angle(1.0, 0.0, 0.0))),
                None,
            );
        }
        problem.add_residual_block(
            &[keys[0]],
            Box::new(PriorFactor::new(SE2::from_xy_angle(0.0, 0.0, 0.0))),
            None,
        );
        problem.mark_for_elimination(keys[1]);
        problem.mark_for_elimination(keys[3]);
        problem.group_rows_for_elimination();

        let mut state = crate::optimizer::initialize_optimization_state(&mut problem)?;
        let mut eliminate = std::collections::HashSet::new();
        eliminate.insert(keys[1]);
        eliminate.insert(keys[3]);

        // Drive BOTH solvers from one shared assembly each iteration, so any
        // divergence is the solvers', not the assembly's.
        let costs = |variables: &slotmap::SlotMap<
            crate::core::VarKey,
            Box<dyn crate::core::variable::ManifoldVariable>,
        >|
         -> Result<f64, Box<dyn std::error::Error>> {
            Ok(problem.compute_residual_and_cost_sparse(variables)?.1)
        };

        let mut cholesky = SparseCholeskySolver::new();
        let mut schur = SparseSchurComplementSolver::new();
        schur.initialize_structure(&state.variables, &state.variable_index_map, &eliminate)?;

        for iter in 0..3 {
            let symbolic = state.symbolic_structure.as_ref().ok_or("sparse symbolic")?;
            let (residuals, jacobian) = crate::linearizer::cpu::sparse::assemble_sparse(
                &problem,
                &state.variables,
                &state.variable_index_map,
                symbolic,
                &mut state.workspace,
            )?;
            let reference =
                crate::linalg::LinearSolver::<crate::linalg::SparseMode>::solve_normal_equation(
                    &mut cholesky,
                    &residuals,
                    &jacobian,
                )?;
            let step =
                crate::linalg::LinearSolver::<crate::linalg::SparseMode>::solve_normal_equation(
                    &mut schur, &residuals, &jacobian,
                )?;
            let scale = reference.norm_l2().max(1.0);
            let mut max_rel: f64 = 0.0;
            for i in 0..reference.nrows() {
                max_rel = max_rel.max((reference[(i, 0)] - step[(i, 0)]).abs() / scale);
            }
            assert!(
                max_rel < 1e-9,
                "iteration {iter}: Schur step diverges from Cholesky (max rel {max_rel:.3e})"
            );
            crate::optimizer::apply_parameter_step(
                &mut state.variables,
                reference.as_ref(),
                &state.sorted_vars,
            );
            let _ = costs(&state.variables)?;
        }
        Ok(())
    }

    #[test]
    fn test_solve_with_cholesky_small_spd() -> TestResult {
        let solver = SparseSchurComplementSolver::new();

        // 2×2 SPD matrix A = [[4,1],[1,3]]
        let triplets = vec![
            Triplet::new(0usize, 0usize, 4.0f64),
            Triplet::new(1usize, 0usize, 1.0f64),
            Triplet::new(0usize, 1usize, 1.0f64),
            Triplet::new(1usize, 1usize, 3.0f64),
        ];
        let a =
            SparseColMat::try_new_from_triplets(2, 2, &triplets).map_err(|e| format!("{e:?}"))?;
        let b = Mat::from_fn(2, 1, |i, _| (i + 1) as f64); // [1; 2]

        let x = solver.solve_with_cholesky(&a, &b)?;
        assert_eq!(x.nrows(), 2);

        // Verify: A·x ≈ b
        // A·x = [4*x0+1*x1; 1*x0+3*x1]
        let ax0 = 4.0 * x[(0, 0)] + 1.0 * x[(1, 0)];
        let ax1 = 1.0 * x[(0, 0)] + 3.0 * x[(1, 0)];
        assert!((ax0 - 1.0).abs() < 1e-8, "A·x[0] = {ax0}");
        assert!((ax1 - 2.0).abs() < 1e-8, "A·x[1] = {ax1}");
        Ok(())
    }

    /// Test solve_with_pcg converges on a small diagonal (trivial) system.
    #[test]
    fn test_solve_with_pcg_diagonal_system() -> TestResult {
        let solver = SparseSchurComplementSolver::new();

        // Diagonal SPD: [[2,0],[0,3]]
        let triplets = vec![
            Triplet::new(0usize, 0usize, 2.0f64),
            Triplet::new(1usize, 1usize, 3.0f64),
        ];
        let a =
            SparseColMat::try_new_from_triplets(2, 2, &triplets).map_err(|e| format!("{e:?}"))?;
        let b = Mat::from_fn(2, 1, |i, _| (i + 1) as f64); // [1; 2]

        let x = solver.solve_with_pcg(&a, &b)?;
        // Expected: x = [1/2; 2/3]
        assert!((x[(0, 0)] - 0.5).abs() < 1e-6, "x[0] = {}", x[(0, 0)]);
        assert!((x[(1, 0)] - 2.0 / 3.0).abs() < 1e-6, "x[1] = {}", x[(1, 0)]);
        Ok(())
    }

    /// Test initialize_structure returns Err when only landmark variables are present (no cameras).
    #[test]
    fn test_initialize_structure_no_cameras_returns_error() {
        use crate::core::variable::Variable;
        use apex_manifolds::rn;
        use nalgebra::DVector;

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let k = variables.insert(Box::new(Variable::new(rn::Rn::new(DVector::zeros(3)))));
        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(k, 0);
        let mut landmark_keys = std::collections::HashSet::new();
        landmark_keys.insert(k);

        let mut solver = SparseSchurComplementSolver::new();
        let result = solver.initialize_structure(&variables, &variable_index_map, &landmark_keys);
        assert!(
            result.is_err(),
            "Expected Err when no camera variables present"
        );
    }

    /// Test initialize_structure returns Err when only camera variables are present (no landmarks).
    #[test]
    fn test_initialize_structure_no_landmarks_returns_error() {
        use crate::core::variable::Variable;
        use apex_manifolds::se3;

        let mut variables: SlotMap<VarKey, Box<dyn ManifoldVariable>> = SlotMap::with_key();
        let _k = variables.insert(Box::new(Variable::new(se3::SE3::from_param_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]))));
        let mut variable_index_map: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        variable_index_map.insert(_k, 0);
        let landmark_keys = std::collections::HashSet::<VarKey>::new(); // no landmarks

        let mut solver = SparseSchurComplementSolver::new();
        let result = solver.initialize_structure(&variables, &variable_index_map, &landmark_keys);
        assert!(
            result.is_err(),
            "Expected Err when no landmark variables present"
        );
    }

    /// Dense read-back helper for the damping tests.
    fn dense_of(m: &SparseColMat<usize, f64>) -> Vec<Vec<f64>> {
        let mut out = vec![vec![0.0; m.ncols()]; m.nrows()];
        for (col, out_col) in (0..m.ncols()).zip(0..m.ncols()) {
            let rows = m.symbolic().row_idx_of_col_raw(col);
            let vals = m.val_of_col(col);
            for (i, &r) in rows.iter().enumerate() {
                out[r][out_col] = vals[i];
            }
        }
        out
    }

    /// Fully populated diagonal: damping takes the value-only fast path and
    /// must leave every off-diagonal untouched.
    #[test]
    fn test_damp_camera_block_full_diagonal() -> TestResult {
        let triplets = vec![
            Triplet::new(0usize, 0usize, 4.0_f64),
            Triplet::new(1, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 1, 9.0),
        ];
        let h_cc = SparseColMat::try_new_from_triplets(2, 2, &triplets)?;
        let damping = Damping::new(0.5, 1e-6, 1e32)?;

        let damped = dense_of(&damp_camera_block(&h_cc, 2, &damping)?);

        assert!((damped[0][0] - (4.0 + 0.5 * 4.0)).abs() < 1e-12);
        assert!((damped[1][1] - (9.0 + 0.5 * 9.0)).abs() < 1e-12);
        assert!((damped[0][1] - 1.0).abs() < 1e-12, "off-diagonal changed");
        assert!((damped[1][0] - 1.0).abs() < 1e-12, "off-diagonal changed");
        Ok(())
    }

    /// A camera with no landmark observations has a structurally empty
    /// diagonal. Damping must materialize it at λ·min_diagonal rather than
    /// silently skipping the column.
    #[test]
    fn test_damp_camera_block_missing_diagonal() -> TestResult {
        // Column 1 carries only an off-diagonal entry.
        let triplets = vec![
            Triplet::new(0usize, 0usize, 4.0_f64),
            Triplet::new(0, 1, 2.0),
        ];
        let h_cc = SparseColMat::try_new_from_triplets(2, 2, &triplets)?;
        let damping = Damping::new(0.5, 1e-3, 1e32)?;

        let damped = dense_of(&damp_camera_block(&h_cc, 2, &damping)?);

        assert!((damped[0][0] - (4.0 + 0.5 * 4.0)).abs() < 1e-12);
        // clamp(0, 1e-3, 1e32) * 0.5
        assert!((damped[1][1] - 0.5 * 1e-3).abs() < 1e-12);
        assert!((damped[0][1] - 2.0).abs() < 1e-12, "off-diagonal changed");
        Ok(())
    }
}
