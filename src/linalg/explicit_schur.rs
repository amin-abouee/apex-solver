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
//! # use apex_solver::linalg::{SchurSolverAdapter, SchurVariant, SchurPreconditioner};
//! # use std::collections::HashMap;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let variables = HashMap::new();
//! # let variable_index_map = HashMap::new();
//! use apex_solver::linalg::{SchurSolverAdapter, SchurVariant};
//!
//! let mut solver = SchurSolverAdapter::new_with_structure_and_config(
//!     &variables,
//!     &variable_index_map,
//!     SchurVariant::Sparse, // Explicit Schur with Cholesky
//!     SchurPreconditioner::None,
//! )?;
//! # Ok(())
//! # }
//! ```

use crate::core::problem::VariableEnum;
use crate::linalg::{
    LinAlgError, LinAlgResult, StructuredSparseLinearSolver, implicit_schur::IterativeSchurSolver,
};
use apex_manifolds::ManifoldType;
use faer::sparse::{SparseColMat, Triplet};
use faer::{
    Mat, Side,
    linalg::solvers::Solve,
    sparse::linalg::solvers::{Llt, SymbolicLlt},
};
use nalgebra::DMatrix;
use std::collections::HashMap;
use tracing::{debug, info};

/// Schur complement solver variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchurVariant {
    /// Standard: Direct sparse Cholesky factorization of S
    #[default]
    Sparse,
    /// Iterative: Conjugate Gradients on reduced system
    Iterative,
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
#[derive(Debug, Clone)]
pub struct SchurOrdering {
    pub eliminate_types: Vec<ManifoldType>,
    /// Only eliminate RN variables with this exact size.
    /// `None` (default) means any RN size is accepted for variables matching the
    /// name pattern.  Set to `Some(3)` to restrict to R3 landmarks, etc.
    pub eliminate_rn_size: Option<usize>,
}

impl Default for SchurOrdering {
    fn default() -> Self {
        Self {
            eliminate_types: vec![ManifoldType::RN],
            eliminate_rn_size: None, // Accept any RN size (R1 inverse-depth, R3 world-point, …)
        }
    }
}

impl SchurOrdering {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a variable should be eliminated (treated as landmark).
    ///
    /// Uses variable name pattern matching for robust classification:
    /// - Variables starting with "pt_" are landmarks (any RN dimension)
    /// - All other variables are camera parameters (poses, intrinsics)
    ///
    /// This correctly handles shared intrinsics (single RN variable for all cameras)
    /// without misclassifying them as landmarks.
    pub fn should_eliminate(&self, name: &str, manifold_type: &ManifoldType, size: usize) -> bool {
        if name.starts_with("pt_") {
            if !self.eliminate_types.contains(manifold_type) {
                return false;
            }

            if self
                .eliminate_rn_size
                .is_some_and(|required_size| size != required_size)
            {
                return false;
            }
            true
        } else {
            false
        }
    }
}

/// Block structure for Schur complement solver
#[derive(Debug, Clone)]
pub struct SchurBlockStructure {
    pub camera_blocks: Vec<(String, usize, usize)>,
    pub landmark_blocks: Vec<(String, usize, usize)>,
    pub camera_dof: usize,
    pub landmark_dof: usize,
    pub num_landmarks: usize,
    /// Tangent-space dimension of each landmark block (e.g. 1 for inverse-depth, 3 for R3).
    /// All landmarks in a given problem must share the same block size.
    pub landmark_block_size: usize,
}

impl SchurBlockStructure {
    pub fn new() -> Self {
        Self {
            camera_blocks: Vec::new(),
            landmark_blocks: Vec::new(),
            camera_dof: 0,
            landmark_dof: 0,
            num_landmarks: 0,
            landmark_block_size: 0,
        }
    }

    pub fn camera_col_range(&self) -> (usize, usize) {
        if self.camera_blocks.is_empty() {
            (0, 0)
        } else {
            // Safe: we just checked is_empty() is false
            let start = self.camera_blocks.first().map(|b| b.1).unwrap_or(0);
            (start, start + self.camera_dof)
        }
    }

    pub fn landmark_col_range(&self) -> (usize, usize) {
        if self.landmark_blocks.is_empty() {
            (0, 0)
        } else {
            // Safe: we just checked is_empty() is false
            let start = self.landmark_blocks.first().map(|b| b.1).unwrap_or(0);
            (start, start + self.landmark_dof)
        }
    }
}

impl Default for SchurBlockStructure {
    fn default() -> Self {
        Self::new()
    }
}

/// Sparse Schur Complement Solver for Bundle Adjustment
#[derive(Debug, Clone)]
pub struct SparseSchurComplementSolver {
    block_structure: Option<SchurBlockStructure>,
    ordering: SchurOrdering,
    variant: SchurVariant,
    preconditioner: SchurPreconditioner,

    // CG parameters
    cg_max_iterations: usize,
    cg_tolerance: f64,

    // Cached matrices
    hessian: Option<SparseColMat<usize, f64>>,
    gradient: Option<Mat<f64>>,

    // Delegate solver for iterative variant
    iterative_solver: Option<IterativeSchurSolver>,
}

impl SparseSchurComplementSolver {
    pub fn new() -> Self {
        Self {
            block_structure: None,
            ordering: SchurOrdering::default(),
            variant: SchurVariant::default(),
            preconditioner: SchurPreconditioner::default(),
            cg_max_iterations: 200, // Match Ceres (was 500)
            cg_tolerance: 1e-6,     // Relaxed for speed (was 1e-9)
            hessian: None,
            gradient: None,
            iterative_solver: None,
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

    pub fn block_structure(&self) -> Option<&SchurBlockStructure> {
        self.block_structure.as_ref()
    }

    fn build_block_structure(
        &mut self,
        variables: &HashMap<String, VariableEnum>,
        variable_index_map: &HashMap<String, usize>,
    ) -> LinAlgResult<()> {
        let mut structure = SchurBlockStructure::new();

        for (name, variable) in variables {
            let start_col = *variable_index_map.get(name).ok_or_else(|| {
                LinAlgError::InvalidInput(format!("Variable {} not found in index map", name))
            })?;
            let size = variable.get_size();
            let manifold_type = match variable {
                VariableEnum::SE3(_) => ManifoldType::SE3,
                VariableEnum::SE2(_) => ManifoldType::SE2,
                VariableEnum::SO3(_) => ManifoldType::SO3,
                VariableEnum::SO2(_) => ManifoldType::SO2,
                VariableEnum::Rn(_) => ManifoldType::RN,
            };

            // Use name-based classification via SchurOrdering
            if self.ordering.should_eliminate(name, &manifold_type, size) {
                // Validate all landmarks share the same block size
                if structure.num_landmarks == 0 {
                    structure.landmark_block_size = size;
                } else if size != structure.landmark_block_size {
                    return Err(LinAlgError::InvalidInput(format!(
                        "Landmark {} has DOF {} but expected {} (all landmarks must share the same block size)",
                        name, size, structure.landmark_block_size
                    )));
                }
                structure
                    .landmark_blocks
                    .push((name.clone(), start_col, size));
                structure.landmark_dof += size;
                structure.num_landmarks += 1;
            } else {
                structure
                    .camera_blocks
                    .push((name.clone(), start_col, size));
                structure.camera_dof += size;
            }
        }

        structure.camera_blocks.sort_by_key(|(_, col, _)| *col);
        structure.landmark_blocks.sort_by_key(|(_, col, _)| *col);

        if structure.camera_blocks.is_empty() {
            return Err(LinAlgError::InvalidInput(
                "No camera variables found".to_string(),
            ));
        }
        if structure.landmark_blocks.is_empty() {
            return Err(LinAlgError::InvalidInput(
                "No landmark variables found".to_string(),
            ));
        }

        // Log block structure for diagnostics
        info!("Schur complement block structure:");
        info!(
            "  Camera blocks: {} variables, {} total DOF",
            structure.camera_blocks.len(),
            structure.camera_dof
        );
        info!(
            "  Landmark blocks: {} variables, {} total DOF",
            structure.landmark_blocks.len(),
            structure.landmark_dof
        );
        debug!("  Camera column range: {:?}", structure.camera_col_range());
        debug!(
            "  Landmark column range: {:?}",
            structure.landmark_col_range()
        );
        info!(
            "  Schur complement S size: {} × {}",
            structure.camera_dof, structure.camera_dof
        );

        // Validate column ranges are contiguous
        let (_cam_start, cam_end) = structure.camera_col_range();
        let (land_start, _land_end) = structure.landmark_col_range();
        if cam_end != land_start {
            debug!(
                "WARNING: Gap between camera and landmark blocks! cam_end={}, land_start={}",
                cam_end, land_start
            );
        }

        self.block_structure = Some(structure);
        Ok(())
    }

    /// Extract diagonal blocks from H_pp (one per landmark, size = landmark_block_size)
    fn extract_landmark_blocks(
        &self,
        hessian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Vec<DMatrix<f64>>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not built".to_string()))?;

        let bs = structure.landmark_block_size;
        let mut blocks = Vec::with_capacity(structure.num_landmarks);
        let symbolic = hessian.symbolic();

        for (_, start_col, _) in &structure.landmark_blocks {
            let mut block = DMatrix::<f64>::zeros(bs, bs);

            for local_col in 0..bs {
                let global_col = start_col + local_col;
                let row_indices = symbolic.row_idx_of_col_raw(global_col);
                let col_values = hessian.val_of_col(global_col);

                for (idx, &row) in row_indices.iter().enumerate() {
                    if row >= *start_col && row < start_col + bs {
                        let local_row = row - start_col;
                        block[(local_row, local_col)] = col_values[idx];
                    }
                }
            }

            blocks.push(block);
        }

        Ok(blocks)
    }

    /// Invert all landmark diagonal blocks with numerical robustness.
    fn invert_landmark_blocks(blocks: &[DMatrix<f64>]) -> LinAlgResult<Vec<DMatrix<f64>>> {
        Self::invert_landmark_blocks_with_lambda(blocks, 0.0)
    }

    /// Invert all landmark diagonal blocks with numerical robustness and optional damping.
    ///
    /// For severely ill-conditioned blocks, additional regularization is applied
    /// to ensure numerical stability.
    fn invert_landmark_blocks_with_lambda(
        blocks: &[DMatrix<f64>],
        lambda: f64,
    ) -> LinAlgResult<Vec<DMatrix<f64>>> {
        const CONDITION_THRESHOLD: f64 = 1e10;
        const MIN_EIGENVALUE_THRESHOLD: f64 = 1e-12;
        const REGULARIZATION_SCALE: f64 = 1e-6;

        let mut ill_conditioned_count = 0;
        let mut regularized_count = 0;

        let result: LinAlgResult<Vec<DMatrix<f64>>> = blocks
            .iter()
            .enumerate()
            .map(|(i, block)| {
                let n = block.nrows();
                let eigenvalues = block.clone().symmetric_eigenvalues();
                let min_ev = eigenvalues.min();
                let max_ev = eigenvalues.max();

                let id = DMatrix::<f64>::identity(n, n);

                if min_ev < MIN_EIGENVALUE_THRESHOLD {
                    regularized_count += 1;
                    let reg = lambda.max(REGULARIZATION_SCALE) + max_ev * REGULARIZATION_SCALE;
                    let regularized = block + &id * reg;
                    regularized.try_inverse().ok_or_else(|| {
                        LinAlgError::SingularMatrix(format!(
                            "Landmark block {} singular even with regularization (min_ev={:.2e})",
                            i, min_ev
                        ))
                    })
                } else if max_ev / min_ev > CONDITION_THRESHOLD {
                    ill_conditioned_count += 1;
                    let extra_reg = max_ev * REGULARIZATION_SCALE;
                    let regularized = block + &id * extra_reg;
                    regularized.try_inverse().ok_or_else(|| {
                        LinAlgError::SingularMatrix(format!(
                            "Landmark block {} ill-conditioned (cond={:.2e})",
                            i,
                            max_ev / min_ev
                        ))
                    })
                } else {
                    block.clone().try_inverse().ok_or_else(|| {
                        LinAlgError::SingularMatrix(format!("Landmark block {} is singular", i))
                    })
                }
            })
            .collect();

        if ill_conditioned_count > 0 || regularized_count > 0 {
            debug!(
                "Landmark block conditioning: {} ill-conditioned, {} regularized out of {}",
                ill_conditioned_count,
                regularized_count,
                blocks.len()
            );
        }

        result
    }

    /// Extract H_cc (camera-camera block)
    fn extract_camera_block(
        &self,
        hessian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<SparseColMat<usize, f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not built".to_string()))?;

        let (cam_start, cam_end) = structure.camera_col_range();
        let cam_size = structure.camera_dof;
        let symbolic = hessian.symbolic();

        let mut triplets = Vec::new();

        for global_col in cam_start..cam_end {
            let local_col = global_col - cam_start;
            let row_indices = symbolic.row_idx_of_col_raw(global_col);
            let col_values = hessian.val_of_col(global_col);

            for (idx, &global_row) in row_indices.iter().enumerate() {
                if global_row >= cam_start && global_row < cam_end {
                    let local_row = global_row - cam_start;
                    triplets.push(Triplet::new(local_row, local_col, col_values[idx]));
                }
            }
        }

        SparseColMat::try_new_from_triplets(cam_size, cam_size, &triplets)
            .map_err(|e| LinAlgError::SparseMatrixCreation(format!("H_cc: {:?}", e)))
    }

    /// Extract H_cp (camera-point coupling)
    fn extract_coupling_block(
        &self,
        hessian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<SparseColMat<usize, f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not built".to_string()))?;

        let (cam_start, cam_end) = structure.camera_col_range();
        let (land_start, land_end) = structure.landmark_col_range();
        let cam_size = structure.camera_dof;
        let land_size = structure.landmark_dof;
        let symbolic = hessian.symbolic();

        let mut triplets = Vec::new();

        for global_col in land_start..land_end {
            let local_col = global_col - land_start;
            let row_indices = symbolic.row_idx_of_col_raw(global_col);
            let col_values = hessian.val_of_col(global_col);

            for (idx, &global_row) in row_indices.iter().enumerate() {
                if global_row >= cam_start && global_row < cam_end {
                    let local_row = global_row - cam_start;
                    triplets.push(Triplet::new(local_row, local_col, col_values[idx]));
                }
            }
        }

        SparseColMat::try_new_from_triplets(cam_size, land_size, &triplets)
            .map_err(|e| LinAlgError::SparseMatrixCreation(format!("H_cp: {:?}", e)))
    }

    /// Extract gradient blocks
    fn extract_gradient_blocks(&self, gradient: &Mat<f64>) -> LinAlgResult<(Mat<f64>, Mat<f64>)> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not built".to_string()))?;

        let (cam_start, cam_end) = structure.camera_col_range();
        let (land_start, land_end) = structure.landmark_col_range();

        let mut g_c = Mat::zeros(structure.camera_dof, 1);
        for i in 0..(cam_end - cam_start) {
            g_c[(i, 0)] = gradient[(cam_start + i, 0)];
        }

        let mut g_p = Mat::zeros(structure.landmark_dof, 1);
        for i in 0..(land_end - land_start) {
            g_p[(i, 0)] = gradient[(land_start + i, 0)];
        }

        Ok((g_c, g_p))
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

        // Initialize
        let mut x = Mat::<f64>::zeros(n, 1);

        // r = b - A*x (x starts at 0, so r = b)
        let mut r = b.clone();

        // z = M^{-1} * r (Jacobi preconditioning)
        let mut z = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            z[(i, 0)] = precond[i] * r[(i, 0)];
        }

        let mut p = z.clone();

        let mut rz_old = 0.0;
        for i in 0..n {
            rz_old += r[(i, 0)] * z[(i, 0)];
        }

        // Compute initial residual norm for relative tolerance
        let mut r_norm_init = 0.0;
        for i in 0..n {
            r_norm_init += r[(i, 0)] * r[(i, 0)];
        }
        r_norm_init = r_norm_init.sqrt();
        let abs_tol = tolerance * r_norm_init.max(1.0);

        for _iter in 0..max_iterations {
            // Ap = A * p (sparse matrix-vector product)
            let mut ap = Mat::<f64>::zeros(n, 1);
            for col in 0..n {
                let row_indices = symbolic.row_idx_of_col_raw(col);
                let col_values = a.val_of_col(col);
                for (idx, &row) in row_indices.iter().enumerate() {
                    ap[(row, 0)] += col_values[idx] * p[(col, 0)];
                }
            }

            // alpha = (r^T z) / (p^T Ap)
            let mut p_ap = 0.0;
            for i in 0..n {
                p_ap += p[(i, 0)] * ap[(i, 0)];
            }

            if p_ap.abs() < 1e-30 {
                break;
            }

            let alpha = rz_old / p_ap;

            // x = x + alpha * p
            for i in 0..n {
                x[(i, 0)] += alpha * p[(i, 0)];
            }

            // r = r - alpha * Ap
            for i in 0..n {
                r[(i, 0)] -= alpha * ap[(i, 0)];
            }

            // Check convergence
            let mut r_norm = 0.0;
            for i in 0..n {
                r_norm += r[(i, 0)] * r[(i, 0)];
            }
            r_norm = r_norm.sqrt();

            if r_norm < abs_tol {
                break;
            }

            // z = M^{-1} * r
            for i in 0..n {
                z[(i, 0)] = precond[i] * r[(i, 0)];
            }

            // beta = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)
            let mut rz_new = 0.0;
            for i in 0..n {
                rz_new += r[(i, 0)] * z[(i, 0)];
            }

            if rz_old.abs() < 1e-30 {
                break;
            }

            let beta = rz_new / rz_old;

            // p = z + beta * p
            for i in 0..n {
                p[(i, 0)] = z[(i, 0)] + beta * p[(i, 0)];
            }

            rz_old = rz_new;
        }

        Ok(x)
    }

    /// Compute Schur complement: S = H_cc - H_cp * H_pp^{-1} * H_cp^T
    ///
    /// Exploits block-diagonal H_pp, sparsity of H_cp, and dense accumulation
    /// for the small camera-camera matrix S.  Works for any landmark block size.
    fn compute_schur_complement(
        &self,
        h_cc: &SparseColMat<usize, f64>,
        h_cp: &SparseColMat<usize, f64>,
        hpp_inv_blocks: &[DMatrix<f64>],
    ) -> LinAlgResult<SparseColMat<usize, f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not built".into()))?;
        let bs = structure.landmark_block_size;
        let cam_size = h_cc.nrows();
        let h_cp_sym = h_cp.symbolic();

        let mut s_dense = vec![0.0f64; cam_size * cam_size];

        // Add H_cc to S
        let h_cc_sym = h_cc.symbolic();
        for col in 0..h_cc.ncols() {
            let row_indices = h_cc_sym.row_idx_of_col_raw(col);
            let col_values = h_cc.val_of_col(col);
            for (idx, &row) in row_indices.iter().enumerate() {
                s_dense[row * cam_size + col] += col_values[idx];
            }
        }

        // Pre-allocated flat buffers (cam_rows.len() × bs stride)
        let mut cam_rows: Vec<usize> = Vec::with_capacity(64);
        let mut h_cp_flat: Vec<f64> = Vec::with_capacity(64 * bs);
        let mut contrib_flat: Vec<f64> = Vec::with_capacity(64 * bs);
        let mut pairs: Vec<(usize, usize, f64)> = Vec::with_capacity(64 * bs);

        for (li, hpp_inv) in hpp_inv_blocks.iter().enumerate() {
            let col_start = li * bs;
            if col_start + bs > h_cp.ncols() {
                continue;
            }

            // Gather (row, local_col, value) from all bs columns of this landmark
            pairs.clear();
            for lc in 0..bs {
                let gc = col_start + lc;
                let ri = h_cp_sym.row_idx_of_col_raw(gc);
                let cv = h_cp.val_of_col(gc);
                for (idx, &r) in ri.iter().enumerate() {
                    pairs.push((r, lc, cv[idx]));
                }
            }
            pairs.sort_unstable_by_key(|&(r, c, _)| (r, c));

            // Group by row into cam_rows + h_cp_flat (row-major, stride = bs)
            cam_rows.clear();
            h_cp_flat.clear();
            let mut pi = 0;
            while pi < pairs.len() {
                let row = pairs[pi].0;
                cam_rows.push(row);
                let base = h_cp_flat.len();
                h_cp_flat.resize(base + bs, 0.0);
                while pi < pairs.len() && pairs[pi].0 == row {
                    h_cp_flat[base + pairs[pi].1] = pairs[pi].2;
                    pi += 1;
                }
            }

            if cam_rows.is_empty() {
                continue;
            }

            // contrib = h_cp_rows * hpp_inv  (nc × bs, row-major)
            let nc = cam_rows.len();
            contrib_flat.clear();
            contrib_flat.resize(nc * bs, 0.0);
            for i in 0..nc {
                for k in 0..bs {
                    let mut s = 0.0;
                    for j in 0..bs {
                        s += h_cp_flat[i * bs + j] * hpp_inv[(j, k)];
                    }
                    contrib_flat[i * bs + k] = s;
                }
            }

            // S -= contrib * h_cp_rows^T  (rank-bs outer-product update)
            for i in 0..nc {
                let ri = cam_rows[i];
                for j in 0..nc {
                    let rj = cam_rows[j];
                    let mut dot = 0.0;
                    for k in 0..bs {
                        dot += contrib_flat[i * bs + k] * h_cp_flat[j * bs + k];
                    }
                    s_dense[ri * cam_size + rj] -= dot;
                }
            }
        }

        // Symmetrize to counter floating-point accumulation errors
        for i in 0..cam_size {
            for j in (i + 1)..cam_size {
                let avg = (s_dense[i * cam_size + j] + s_dense[j * cam_size + i]) * 0.5;
                s_dense[i * cam_size + j] = avg;
                s_dense[j * cam_size + i] = avg;
            }
        }

        // Convert to sparse (filter near-zeros)
        let mut s_triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        for col in 0..cam_size {
            for row in 0..cam_size {
                let val = s_dense[row * cam_size + col];
                if val.abs() > 1e-12 {
                    s_triplets.push(Triplet::new(row, col, val));
                }
            }
        }

        SparseColMat::try_new_from_triplets(cam_size, cam_size, &s_triplets)
            .map_err(|e| LinAlgError::SparseMatrixCreation(format!("Schur S: {:?}", e)))
    }

    /// Compute reduced gradient: g_reduced = g_c - H_cp * H_pp^{-1} * g_p
    fn compute_reduced_gradient(
        &self,
        g_c: &Mat<f64>,
        g_p: &Mat<f64>,
        h_cp: &SparseColMat<usize, f64>,
        hpp_inv_blocks: &[DMatrix<f64>],
    ) -> LinAlgResult<Mat<f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not built".into()))?;
        let bs = structure.landmark_block_size;
        let land_size = g_p.nrows();
        let cam_size = g_c.nrows();

        // Compute H_pp^{-1} * g_p block-wise
        let mut hpp_inv_gp = Mat::zeros(land_size, 1);
        for (bi, inv_block) in hpp_inv_blocks.iter().enumerate() {
            let rs = bi * bs;
            for r in 0..bs {
                let mut s = 0.0;
                for c in 0..bs {
                    s += inv_block[(r, c)] * g_p[(rs + c, 0)];
                }
                hpp_inv_gp[(rs + r, 0)] = s;
            }
        }

        // Compute H_cp * (H_pp^{-1} * g_p)
        let mut h_cp_hpp_inv_gp = Mat::<f64>::zeros(cam_size, 1);
        let sym = h_cp.symbolic();
        for col in 0..h_cp.ncols() {
            let row_indices = sym.row_idx_of_col_raw(col);
            let col_values = h_cp.val_of_col(col);
            for (idx, &row) in row_indices.iter().enumerate() {
                h_cp_hpp_inv_gp[(row, 0)] += col_values[idx] * hpp_inv_gp[(col, 0)];
            }
        }

        let mut g_reduced = Mat::zeros(cam_size, 1);
        for i in 0..cam_size {
            g_reduced[(i, 0)] = g_c[(i, 0)] - h_cp_hpp_inv_gp[(i, 0)];
        }
        Ok(g_reduced)
    }

    /// Back-substitute: δp = H_pp^{-1} * (g_p - H_cp^T * δc)
    fn back_substitute(
        &self,
        delta_c: &Mat<f64>,
        g_p: &Mat<f64>,
        h_cp: &SparseColMat<usize, f64>,
        hpp_inv_blocks: &[DMatrix<f64>],
    ) -> LinAlgResult<Mat<f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not built".into()))?;
        let bs = structure.landmark_block_size;
        let land_size = g_p.nrows();

        // Compute H_cp^T * δc
        let mut h_cp_t_delta_c = Mat::<f64>::zeros(land_size, 1);
        let sym = h_cp.symbolic();
        for col in 0..h_cp.ncols() {
            let row_indices = sym.row_idx_of_col_raw(col);
            let col_values = h_cp.val_of_col(col);
            for (idx, &row) in row_indices.iter().enumerate() {
                h_cp_t_delta_c[(col, 0)] += col_values[idx] * delta_c[(row, 0)];
            }
        }

        // rhs = g_p - H_cp^T * δc
        let mut rhs = Mat::zeros(land_size, 1);
        for i in 0..land_size {
            rhs[(i, 0)] = g_p[(i, 0)] - h_cp_t_delta_c[(i, 0)];
        }

        // δp = H_pp^{-1} * rhs  (block-wise)
        let mut delta_p = Mat::zeros(land_size, 1);
        for (bi, inv_block) in hpp_inv_blocks.iter().enumerate() {
            let rs = bi * bs;
            for r in 0..bs {
                let mut s = 0.0;
                for c in 0..bs {
                    s += inv_block[(r, c)] * rhs[(rs + c, 0)];
                }
                delta_p[(rs + r, 0)] = s;
            }
        }
        Ok(delta_p)
    }
}

impl Default for SparseSchurComplementSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl StructuredSparseLinearSolver for SparseSchurComplementSolver {
    fn initialize_structure(
        &mut self,
        variables: &HashMap<String, VariableEnum>,
        variable_index_map: &HashMap<String, usize>,
    ) -> LinAlgResult<()> {
        // Build block structure for all variants
        self.build_block_structure(variables, variable_index_map)?;

        // Initialize delegate solver based on variant
        match self.variant {
            SchurVariant::Iterative => {
                let mut solver =
                    IterativeSchurSolver::with_cg_params(self.cg_max_iterations, self.cg_tolerance);
                solver.initialize_structure(variables, variable_index_map)?;
                self.iterative_solver = Some(solver);
            }
            SchurVariant::Sparse => {
                // No delegate solver needed for sparse variant
            }
        }

        Ok(())
    }

    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        use std::ops::Mul;

        if self.block_structure.is_none() {
            return Err(LinAlgError::InvalidInput(
                "Block structure not built. Call initialize_structure() first.".to_string(),
            ));
        }

        // Sparse and Iterative variants use the same Schur complement formation
        // They differ only in how S*δc = g_reduced is solved:
        // - Sparse: Cholesky factorization
        // - Iterative: PCG

        // 1. Build H = J^T * J and g = -J^T * r
        let jt = jacobians
            .transpose()
            .to_col_major()
            .map_err(|e| LinAlgError::MatrixConversion(format!("Transpose failed: {:?}", e)))?;
        let hessian = jt.mul(jacobians);
        let gradient = jacobians.transpose().mul(residuals);
        let mut neg_gradient = Mat::zeros(gradient.nrows(), 1);
        for i in 0..gradient.nrows() {
            neg_gradient[(i, 0)] = -gradient[(i, 0)];
        }

        self.hessian = Some(hessian.clone());
        // Store the positive gradient (J^T * r) for predicted reduction calculation
        // The Schur solver internally uses neg_gradient (-J^T * r) for the solve
        self.gradient = Some(gradient.clone());

        // 2. Extract blocks
        let h_cc = self.extract_camera_block(&hessian)?;
        let h_cp = self.extract_coupling_block(&hessian)?;
        let hpp_blocks = self.extract_landmark_blocks(&hessian)?;
        let (g_c, g_p) = self.extract_gradient_blocks(&neg_gradient)?;

        // 3. Invert H_pp blocks
        let hpp_inv_blocks = Self::invert_landmark_blocks(&hpp_blocks)?;

        // 4. Compute Schur complement S
        let s = self.compute_schur_complement(&h_cc, &h_cp, &hpp_inv_blocks)?;

        // 5. Compute reduced gradient
        let g_reduced = self.compute_reduced_gradient(&g_c, &g_p, &h_cp, &hpp_inv_blocks)?;

        // 6. Solve S * δc = g_reduced (Cholesky for Sparse, PCG for Iterative)
        let delta_c = match self.variant {
            SchurVariant::Iterative => self.solve_with_pcg(&s, &g_reduced)?,
            _ => self.solve_with_cholesky(&s, &g_reduced)?,
        };

        // 7. Back-substitute for δp
        let delta_p = self.back_substitute(&delta_c, &g_p, &h_cp, &hpp_inv_blocks)?;

        // 8. Combine results
        self.combine_updates(&delta_c, &delta_p)
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobians: &SparseColMat<usize, f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>> {
        use std::ops::Mul;

        if self.block_structure.is_none() {
            return Err(LinAlgError::InvalidInput(
                "Block structure not built. Call initialize_structure() first.".to_string(),
            ));
        }

        // Sparse and Iterative variants use the same Schur complement formation with damping
        // 1. Build H = J^T * J and g = -J^T * r
        let jt = jacobians
            .transpose()
            .to_col_major()
            .map_err(|e| LinAlgError::MatrixConversion(format!("Transpose failed: {:?}", e)))?;
        let hessian = jt.mul(jacobians);
        let gradient = jacobians.transpose().mul(residuals);
        let mut neg_gradient = Mat::zeros(gradient.nrows(), 1);
        for i in 0..gradient.nrows() {
            neg_gradient[(i, 0)] = -gradient[(i, 0)];
        }

        self.hessian = Some(hessian.clone());
        // Store the positive gradient (J^T * r) for predicted reduction calculation
        // The Schur solver internally uses neg_gradient (-J^T * r) for the solve
        self.gradient = Some(gradient.clone());

        // 2. Extract blocks
        let h_cc = self.extract_camera_block(&hessian)?;
        let h_cp = self.extract_coupling_block(&hessian)?;
        let mut hpp_blocks = self.extract_landmark_blocks(&hessian)?;
        let (g_c, g_p) = self.extract_gradient_blocks(&neg_gradient)?;

        // Log matrix dimensions for diagnostics
        debug!("Iteration matrices:");
        debug!(
            "  Hessian (J^T*J): {} × {}",
            hessian.nrows(),
            hessian.ncols()
        );
        debug!("  H_cc (camera): {} × {}", h_cc.nrows(), h_cc.ncols());
        debug!("  H_cp (coupling): {} × {}", h_cp.nrows(), h_cp.ncols());
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;
        let bs = structure.landmark_block_size;
        debug!(
            "  H_pp blocks: {} ({}x{} each)",
            hpp_blocks.len(),
            bs,
            bs
        );

        // 3. Add damping to H_cc and H_pp
        let cam_size = structure.camera_dof;

        // Add λI to H_cc
        let mut h_cc_triplets = Vec::new();
        let h_cc_symbolic = h_cc.symbolic();
        for col in 0..h_cc.ncols() {
            let row_indices = h_cc_symbolic.row_idx_of_col_raw(col);
            let col_values = h_cc.val_of_col(col);
            for (idx, &row) in row_indices.iter().enumerate() {
                h_cc_triplets.push(Triplet::new(row, col, col_values[idx]));
            }
        }
        for i in 0..cam_size {
            if let Some(entry) = h_cc_triplets.iter_mut().find(|t| t.row == i && t.col == i) {
                *entry = Triplet::new(i, i, entry.val + lambda);
            } else {
                h_cc_triplets.push(Triplet::new(i, i, lambda));
            }
        }
        let h_cc_damped =
            SparseColMat::try_new_from_triplets(cam_size, cam_size, &h_cc_triplets)
                .map_err(|e| LinAlgError::SparseMatrixCreation(format!("Damped H_cc: {:?}", e)))?;

        // Add λI to H_pp blocks
        for block in &mut hpp_blocks {
            for d in 0..bs {
                block[(d, d)] += lambda;
            }
        }

        // 4. Invert damped H_pp blocks
        let hpp_inv_blocks = Self::invert_landmark_blocks(&hpp_blocks)?;

        // 5. Compute Schur complement with damped matrices
        let s = self.compute_schur_complement(&h_cc_damped, &h_cp, &hpp_inv_blocks)?;

        // 6. Compute reduced gradient
        let g_reduced = self.compute_reduced_gradient(&g_c, &g_p, &h_cp, &hpp_inv_blocks)?;

        // 7. Solve S * δc = g_reduced (Cholesky for Sparse, PCG for Iterative)
        let delta_c = match self.variant {
            SchurVariant::Iterative => self.solve_with_pcg(&s, &g_reduced)?,
            _ => self.solve_with_cholesky(&s, &g_reduced)?,
        };

        // 8. Back-substitute for δp
        let delta_p = self.back_substitute(&delta_c, &g_p, &h_cp, &hpp_inv_blocks)?;

        // 9. Combine results
        self.combine_updates(&delta_c, &delta_p)
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
    /// Combine camera and landmark updates into full update vector
    fn combine_updates(&self, delta_c: &Mat<f64>, delta_p: &Mat<f64>) -> LinAlgResult<Mat<f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not built".to_string()))?;

        let total_dof = structure.camera_dof + structure.landmark_dof;
        let mut delta = Mat::zeros(total_dof, 1);

        let (cam_start, cam_end) = structure.camera_col_range();
        let (land_start, land_end) = structure.landmark_col_range();

        // Copy camera updates
        for i in 0..(cam_end - cam_start) {
            delta[(cam_start + i, 0)] = delta_c[(i, 0)];
        }

        // Copy landmark updates
        for i in 0..(land_end - land_start) {
            delta[(land_start + i, 0)] = delta_p[(i, 0)];
        }

        // Debug: Log update magnitude
        debug!(
            "Update norms: delta_c={:.6e}, delta_p={:.6e}, combined={:.6e}",
            delta_c.norm_l2(),
            delta_p.norm_l2(),
            delta.norm_l2()
        );

        Ok(delta)
    }
}

/// Adapter to use SparseSchurComplementSolver as a standard SparseLinearSolver
///
/// This adapter bridges the gap between StructuredSparseLinearSolver (which requires
/// variable information) and the standard SparseLinearSolver trait used by optimizers.
pub struct SchurSolverAdapter {
    schur_solver: SparseSchurComplementSolver,
    initialized: bool,
}

impl SchurSolverAdapter {
    /// Create a new Schur solver adapter with problem structure
    ///
    /// This initializes the internal Schur solver with variable partitioning information.
    ///
    /// # Arguments
    /// * `variables` - Map of variable names to their typed instances
    /// * `variable_index_map` - Map from variable names to starting column indices in Jacobian
    ///
    /// # Returns
    /// Initialized adapter ready to solve linear systems
    pub fn new_with_structure(
        variables: &std::collections::HashMap<String, crate::core::problem::VariableEnum>,
        variable_index_map: &std::collections::HashMap<String, usize>,
    ) -> LinAlgResult<Self> {
        let mut solver = SparseSchurComplementSolver::new();
        solver.initialize_structure(variables, variable_index_map)?;
        Ok(Self {
            schur_solver: solver,
            initialized: true,
        })
    }

    /// Create with custom configuration
    pub fn new_with_structure_and_config(
        variables: &std::collections::HashMap<String, crate::core::problem::VariableEnum>,
        variable_index_map: &std::collections::HashMap<String, usize>,
        variant: SchurVariant,
        preconditioner: SchurPreconditioner,
    ) -> LinAlgResult<Self> {
        let mut solver = SparseSchurComplementSolver::new()
            .with_variant(variant)
            .with_preconditioner(preconditioner);
        solver.initialize_structure(variables, variable_index_map)?;
        Ok(Self {
            schur_solver: solver,
            initialized: true,
        })
    }
}

impl crate::linalg::SparseLinearSolver for SchurSolverAdapter {
    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        if !self.initialized {
            return Err(LinAlgError::InvalidInput(
                "Schur solver adapter not initialized with structure".to_string(),
            ));
        }
        self.schur_solver.solve_normal_equation(residuals, jacobian)
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>> {
        if !self.initialized {
            return Err(LinAlgError::InvalidInput(
                "Schur solver adapter not initialized with structure".to_string(),
            ));
        }
        self.schur_solver
            .solve_augmented_equation(residuals, jacobian, lambda)
    }

    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.schur_solver.get_hessian()
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.schur_solver.get_gradient()
    }

    fn compute_covariance_matrix(&mut self) -> Option<&Mat<f64>> {
        // TODO: Implement covariance computation for Schur complement solver
        // This requires inverting the Schur complement S, which is non-trivial
        // For now, return None
        None
    }

    fn get_covariance_matrix(&self) -> Option<&Mat<f64>> {
        // Not implemented yet
        None
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_schur_ordering_shared_intrinsics() {
        let ordering = SchurOrdering::default();

        // Landmarks should be eliminated (any RN size)
        assert!(ordering.should_eliminate("pt_00000", &ManifoldType::RN, 3));
        assert!(ordering.should_eliminate("pt_12345", &ManifoldType::RN, 3));
        assert!(ordering.should_eliminate("pt_00000", &ManifoldType::RN, 1)); // R1 inverse-depth

        // Camera poses should NOT be eliminated
        assert!(!ordering.should_eliminate("cam_0000", &ManifoldType::SE3, 6));
        assert!(!ordering.should_eliminate("cam_0042", &ManifoldType::SE3, 6));

        // Shared intrinsics (RN, 3 DOF) should NOT be eliminated - KEY TEST!
        assert!(!ordering.should_eliminate("shared_intrinsics", &ManifoldType::RN, 3));

        // Per-camera intrinsics should NOT be eliminated
        assert!(!ordering.should_eliminate("intr_0000", &ManifoldType::RN, 3));
        assert!(!ordering.should_eliminate("intr_0042", &ManifoldType::RN, 3));
    }

    #[test]
    fn test_schur_ordering_with_size_constraint() {
        let ordering = SchurOrdering {
            eliminate_rn_size: Some(3),
            ..SchurOrdering::default()
        };

        // R3 landmarks accepted
        assert!(ordering.should_eliminate("pt_00000", &ManifoldType::RN, 3));
        // R1 landmarks rejected by constraint
        assert!(!ordering.should_eliminate("pt_00000", &ManifoldType::RN, 1));
    }

    #[test]
    fn test_schur_ordering_multiple_intrinsic_groups() {
        let ordering = SchurOrdering::default();

        assert!(
            !ordering.should_eliminate("intr_group_0000", &ManifoldType::RN, 3),
            "Intrinsic group 0 should be camera parameter (not eliminated)"
        );
        assert!(
            !ordering.should_eliminate("intr_group_0042", &ManifoldType::RN, 3),
            "Intrinsic group 42 should be camera parameter (not eliminated)"
        );
        assert!(
            ordering.should_eliminate("pt_00000", &ManifoldType::RN, 3),
            "Landmarks should still be eliminated"
        );
        assert!(
            !ordering.should_eliminate("cam_0000", &ManifoldType::SE3, 6),
            "Camera poses should not be eliminated"
        );
    }

    #[test]
    fn test_schur_ordering_invalid_landmark_type() {
        let ordering = SchurOrdering::default();
        assert!(
            !ordering.should_eliminate("pt_00000", &ManifoldType::SE3, 6),
            "Landmark with invalid manifold type should not be eliminated"
        );
    }

    #[test]
    fn test_block_structure_creation() {
        let structure = SchurBlockStructure::new();
        assert_eq!(structure.camera_dof, 0);
        assert_eq!(structure.landmark_dof, 0);
        assert_eq!(structure.landmark_block_size, 0);
    }

    #[test]
    fn test_solver_creation() {
        let solver = SparseSchurComplementSolver::new();
        assert!(solver.block_structure.is_none());
    }

    #[test]
    fn test_3x3_block_inversion() {
        let block = DMatrix::from_row_slice(3, 3, &[
            2.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 4.0,
        ]);
        let inv = SparseSchurComplementSolver::invert_landmark_blocks(&[block])
            .expect("Test: Block inversion should succeed");
        assert!((inv[0][(0, 0)] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_1x1_block_inversion() {
        let block = DMatrix::from_element(1, 1, 4.0);
        let inv = SparseSchurComplementSolver::invert_landmark_blocks(&[block])
            .expect("Test: 1x1 block inversion should succeed");
        assert!((inv[0][(0, 0)] - 0.25).abs() < 1e-10);
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

    fn make_solver_with_structure(cam_dof: usize, lm_block_size: usize, num_landmarks: usize) -> SparseSchurComplementSolver {
        let mut solver = SparseSchurComplementSolver::new();
        let mut structure = SchurBlockStructure::new();
        structure.camera_dof = cam_dof;
        structure.landmark_dof = num_landmarks * lm_block_size;
        structure.num_landmarks = num_landmarks;
        structure.landmark_block_size = lm_block_size;
        for i in 0..num_landmarks {
            structure.landmark_blocks.push((format!("pt_{:05}", i), cam_dof + i * lm_block_size, lm_block_size));
        }
        structure.camera_blocks.push(("cam_0".into(), 0, cam_dof));
        solver.block_structure = Some(structure);
        solver
    }

    #[test]
    fn test_compute_schur_complement_r3() {
        use faer::sparse::Triplet;

        let solver = make_solver_with_structure(2, 3, 1);

        let h_cc_triplets = vec![Triplet::new(0, 0, 4.0), Triplet::new(1, 1, 5.0)];
        let h_cc = SparseColMat::try_new_from_triplets(2, 2, &h_cc_triplets).unwrap();

        let h_cp_triplets = vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 2.0)];
        let h_cp = SparseColMat::try_new_from_triplets(2, 3, &h_cp_triplets).unwrap();

        let hpp_inv = vec![DMatrix::from_diagonal_element(3, 3, 0.5)];

        let s = solver
            .compute_schur_complement(&h_cc, &h_cp, &hpp_inv)
            .expect("Schur complement should succeed");

        assert_eq!(s.nrows(), 2);
        assert!((s[(0, 0)] - 3.5).abs() < 1e-10, "S(0,0) = {}", s[(0, 0)]);
        assert!((s[(1, 1)] - 3.0).abs() < 1e-10, "S(1,1) = {}", s[(1, 1)]);
    }

    #[test]
    fn test_compute_schur_complement_r1() {
        use faer::sparse::Triplet;

        let solver = make_solver_with_structure(2, 1, 2);

        let h_cc_triplets = vec![Triplet::new(0, 0, 4.0), Triplet::new(1, 1, 5.0)];
        let h_cc = SparseColMat::try_new_from_triplets(2, 2, &h_cc_triplets).unwrap();

        // 2 R1 landmarks: lm0 seen by cam row 0, lm1 seen by cam row 1
        let h_cp_triplets = vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 3.0)];
        let h_cp = SparseColMat::try_new_from_triplets(2, 2, &h_cp_triplets).unwrap();

        // hpp_inv scalars: 1/h_pp for each landmark
        let hpp_inv = vec![
            DMatrix::from_element(1, 1, 0.5),
            DMatrix::from_element(1, 1, 0.25),
        ];

        // S(0,0) = 4 - 2*0.5*2 = 4 - 2 = 2
        // S(1,1) = 5 - 3*0.25*3 = 5 - 2.25 = 2.75
        let s = solver
            .compute_schur_complement(&h_cc, &h_cp, &hpp_inv)
            .expect("Schur complement should succeed for R1 landmarks");

        assert_eq!(s.nrows(), 2);
        assert!((s[(0, 0)] - 2.0).abs() < 1e-10, "S(0,0) = {}", s[(0, 0)]);
        assert!((s[(1, 1)] - 2.75).abs() < 1e-10, "S(1,1) = {}", s[(1, 1)]);
    }

    #[test]
    fn test_back_substitute() {
        use faer::sparse::Triplet;

        let solver = make_solver_with_structure(2, 3, 1);

        let delta_c = Mat::from_fn(2, 1, |i, _| (i + 1) as f64);
        let g_p = Mat::from_fn(3, 1, |i, _| (i + 1) as f64);

        let h_cp_triplets = vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)];
        let h_cp = SparseColMat::try_new_from_triplets(2, 3, &h_cp_triplets).unwrap();

        let hpp_inv = vec![DMatrix::identity(3, 3)];

        let delta_p = solver
            .back_substitute(&delta_c, &g_p, &h_cp, &hpp_inv)
            .expect("Back substitution should succeed");

        assert_eq!(delta_p.nrows(), 3);
        assert!((delta_p[(0, 0)]).abs() < 1e-10);
        assert!((delta_p[(1, 0)]).abs() < 1e-10);
        assert!((delta_p[(2, 0)] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_reduced_gradient() {
        use faer::sparse::Triplet;

        let solver = make_solver_with_structure(2, 3, 1);

        let g_c = Mat::from_fn(2, 1, |i, _| (i + 1) as f64);
        let g_p = Mat::from_fn(3, 1, |i, _| (i + 1) as f64);

        let h_cp_triplets = vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)];
        let h_cp = SparseColMat::try_new_from_triplets(2, 3, &h_cp_triplets).unwrap();

        let hpp_inv = vec![DMatrix::from_diagonal_element(3, 3, 2.0)];

        let g_reduced = solver
            .compute_reduced_gradient(&g_c, &g_p, &h_cp, &hpp_inv)
            .expect("Reduced gradient computation should succeed");

        assert_eq!(g_reduced.nrows(), 2);
        assert!((g_reduced[(0, 0)] + 1.0).abs() < 1e-10);
        assert!((g_reduced[(1, 0)] + 2.0).abs() < 1e-10);
    }
}
