//! # Implicit Schur Complement Solver
//!
//! This module implements the **Implicit Schur Complement** method using matrix-free
//! Preconditioned Conjugate Gradients (PCG) for bundle adjustment.
//!
//! ## Explicit vs Implicit Schur Complement
//!
//! **Implicit Schur:** This formulation never constructs the reduced camera matrix S
//! explicitly. Instead, it solves the linear system using a matrix-free approach where
//! only the matrix-vector product S·x is computed. This is highly memory-efficient for
//! large-scale problems.
//!
//! **Explicit Schur:** The alternative formulation (see [`explicit_schur`](super::explicit_schur))
//! physically constructs S = B - E C⁻¹ Eᵀ in memory and uses sparse Cholesky factorization.
//!
//! ## When to Use Implicit Schur
//!
//! - Very large bundle adjustment problems (> 10,000 cameras)
//! - Memory-constrained environments
//! - When iterative methods converge well (good preconditioning)
//! - When the reduced camera system S is too large to store explicitly
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
//! # use apex_solver::linalg::{SchurSolverAdapter, SchurVariant, SchurPreconditioner};
//! # use std::collections::HashMap;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let variables = HashMap::new();
//! # let variable_index_map = HashMap::new();
//! use apex_solver::linalg::{SchurSolverAdapter, SchurVariant, SchurPreconditioner};
//!
//! let mut solver = SchurSolverAdapter::new_with_structure_and_config(
//!     &variables,
//!     &variable_index_map,
//!     SchurVariant::Iterative, // Implicit Schur with PCG
//!     SchurPreconditioner::SchurJacobi, // Recommended for PCG
//! )?;
//! # Ok(())
//! # }
//! ```

use super::explicit_schur::{SchurBlockStructure, SchurOrdering, SchurPreconditioner};
use crate::core::problem::VariableEnum;
use crate::linalg::{LinAlgError, LinAlgResult, StructuredSparseLinearSolver};
use faer::Mat;
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::{DMatrix, DVector, Matrix3};
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Mul;

/// Iterative Schur complement solver using Preconditioned Conjugate Gradients
#[derive(Debug, Clone)]
pub struct IterativeSchurSolver {
    block_structure: Option<SchurBlockStructure>,
    ordering: SchurOrdering,

    // CG parameters
    max_cg_iterations: usize,
    cg_tolerance: f64,

    // Preconditioner type
    preconditioner_type: SchurPreconditioner,

    // Cached for matrix-vector products
    landmark_block_inverses: Vec<Matrix3<f64>>,
    hessian: Option<SparseColMat<usize, f64>>,
    gradient: Option<Mat<f64>>,

    // Workspace buffers for Schur operator (avoid repeated allocations)
    workspace_lm: Vec<f64>,  // landmark DOF sized buffer
    workspace_cam: Vec<f64>, // camera DOF sized buffer

    // Visibility index: camera_block_idx -> Vec<landmark_block_idx>
    // This avoids O(cameras * landmarks) iteration in preconditioner computation
    camera_to_landmark_visibility: Vec<Vec<usize>>,
}

impl IterativeSchurSolver {
    /// Create a new iterative Schur solver with default parameters
    /// Default: Schur-Jacobi preconditioner, 500 max iterations, 1e-9 relative tolerance
    /// These tighter settings match Ceres Solver behavior for accurate step computation.
    pub fn new() -> Self {
        Self {
            block_structure: None,
            ordering: SchurOrdering::default(),
            max_cg_iterations: 500, // More iterations for large BA problems
            cg_tolerance: 1e-9,     // Tighter tolerance for accurate steps
            preconditioner_type: SchurPreconditioner::SchurJacobi,
            landmark_block_inverses: Vec::new(),
            hessian: None,
            gradient: None,
            workspace_lm: Vec::new(),
            workspace_cam: Vec::new(),
            camera_to_landmark_visibility: Vec::new(),
        }
    }

    /// Create solver with custom CG parameters
    pub fn with_cg_params(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            block_structure: None,
            ordering: SchurOrdering::default(),
            max_cg_iterations: max_iterations,
            cg_tolerance: tolerance,
            preconditioner_type: SchurPreconditioner::SchurJacobi,
            landmark_block_inverses: Vec::new(),
            hessian: None,
            gradient: None,
            workspace_lm: Vec::new(),
            workspace_cam: Vec::new(),
            camera_to_landmark_visibility: Vec::new(),
        }
    }

    /// Create solver with full configuration
    pub fn with_config(
        max_iterations: usize,
        tolerance: f64,
        preconditioner: SchurPreconditioner,
    ) -> Self {
        Self {
            block_structure: None,
            ordering: SchurOrdering::default(),
            max_cg_iterations: max_iterations,
            cg_tolerance: tolerance,
            preconditioner_type: preconditioner,
            landmark_block_inverses: Vec::new(),
            hessian: None,
            gradient: None,
            workspace_lm: Vec::new(),
            workspace_cam: Vec::new(),
            camera_to_landmark_visibility: Vec::new(),
        }
    }

    /// Initialize workspace buffers based on problem dimensions
    fn init_workspaces(&mut self) {
        if let Some(structure) = &self.block_structure {
            let lm_dof = structure.landmark_dof;
            let cam_dof = structure.camera_dof;

            if self.workspace_lm.len() != lm_dof {
                self.workspace_lm = vec![0.0; lm_dof];
            }
            if self.workspace_cam.len() != cam_dof {
                self.workspace_cam = vec![0.0; cam_dof];
            }
        }
    }

    /// Apply Schur complement operator: S*x = (H_cc - H_cp * H_pp^{-1} * H_cp^T) * x
    ///
    /// This computes the matrix-vector product without explicitly forming S.
    /// Uses workspace buffers to avoid allocations during PCG iterations.
    fn apply_schur_operator_fast(
        &self,
        x: &Mat<f64>,
        result: &mut Mat<f64>,
        temp_lm: &mut [f64],
        temp_cam: &mut [f64],
    ) -> LinAlgResult<()> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;

        let hessian = self
            .hessian
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Hessian not computed".into()))?;

        let symbolic = hessian.symbolic();
        let (cam_start, cam_end) = structure.camera_col_range();
        let (lm_start, lm_end) = structure.landmark_col_range();
        let cam_dof = structure.camera_dof;

        // Clear workspace buffers
        temp_lm.iter_mut().for_each(|v| *v = 0.0);
        temp_cam.iter_mut().for_each(|v| *v = 0.0);

        // Fused Step 1+2: result = H_cc * x AND temp_lm = H_cp^T * x
        // Process camera columns once, extracting both products
        for col in cam_start..cam_end {
            let local_col = col - cam_start;
            let x_val = x[(local_col, 0)];
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = hessian.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                let val = col_values[idx];
                if row >= cam_start && row < cam_end {
                    // H_cc contribution
                    let local_row = row - cam_start;
                    result[(local_row, 0)] += val * x_val;
                } else if row >= lm_start && row < lm_end {
                    // H_cp^T contribution (camera col -> landmark row)
                    let local_row = row - lm_start;
                    temp_lm[local_row] += val * x_val;
                }
            }
        }

        // Step 3: Apply H_pp^{-1} in-place: temp_lm = H_pp^{-1} * temp_lm
        for (block_idx, (_, start_col, _)) in structure.landmark_blocks.iter().enumerate() {
            let inv_block = &self.landmark_block_inverses[block_idx];
            let local_start = start_col - lm_start;

            // Read input values
            let in0 = temp_lm[local_start];
            let in1 = temp_lm[local_start + 1];
            let in2 = temp_lm[local_start + 2];

            // Apply 3x3 inverse block
            temp_lm[local_start] =
                inv_block[(0, 0)] * in0 + inv_block[(0, 1)] * in1 + inv_block[(0, 2)] * in2;
            temp_lm[local_start + 1] =
                inv_block[(1, 0)] * in0 + inv_block[(1, 1)] * in1 + inv_block[(1, 2)] * in2;
            temp_lm[local_start + 2] =
                inv_block[(2, 0)] * in0 + inv_block[(2, 1)] * in1 + inv_block[(2, 2)] * in2;
        }

        // Step 4: temp_cam = H_cp * temp_lm (iterate over landmark columns)
        for col in lm_start..lm_end {
            let local_col = col - lm_start;
            let lm_val = temp_lm[local_col];
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = hessian.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                if row >= cam_start && row < cam_end {
                    let local_row = row - cam_start;
                    temp_cam[local_row] += col_values[idx] * lm_val;
                }
            }
        }

        // Step 5: result = result - temp_cam = H_cc*x - H_cp*H_pp^{-1}*H_cp^T*x
        for i in 0..cam_dof {
            result[(i, 0)] -= temp_cam[i];
        }

        Ok(())
    }

    /// Extract H_cp^T and multiply with vector: (H_cp^T) * x
    fn extract_camera_landmark_transpose_mvp(
        &self,
        hessian: &SparseColMat<usize, f64>,
        x: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;
        let (cam_start, cam_end) = structure.camera_col_range();
        let (lm_start, lm_end) = structure.landmark_col_range();
        let lm_dof = structure.landmark_dof;

        let mut result = Mat::<f64>::zeros(lm_dof, 1);
        let symbolic = hessian.symbolic();

        // H_cp^T * x = iterate over camera columns, accumulate into landmark rows
        for col in cam_start..cam_end {
            let local_col = col - cam_start;
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = hessian.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                if row >= lm_start && row < lm_end {
                    let local_row = row - lm_start;
                    result[(local_row, 0)] += col_values[idx] * x[(local_col, 0)];
                }
            }
        }

        Ok(result)
    }

    /// Extract H_cp and multiply with vector
    fn extract_camera_landmark_mvp(
        &self,
        hessian: &SparseColMat<usize, f64>,
        x: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;
        let (cam_start, cam_end) = structure.camera_col_range();
        let (lm_start, lm_end) = structure.landmark_col_range();
        let cam_dof = structure.camera_dof;

        let mut result = Mat::<f64>::zeros(cam_dof, 1);
        let symbolic = hessian.symbolic();

        // H_cp * x: iterate over landmark columns
        for col in lm_start..lm_end {
            let local_col = col - lm_start;
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = hessian.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                if row >= cam_start && row < cam_end {
                    let local_row = row - cam_start;
                    result[(local_row, 0)] += col_values[idx] * x[(local_col, 0)];
                }
            }
        }

        Ok(result)
    }

    /// Apply H_pp^{-1} using cached block inverses
    fn apply_landmark_inverse(&self, input: &Mat<f64>, output: &mut Mat<f64>) -> LinAlgResult<()> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;

        for (block_idx, (_, start_col, _)) in structure.landmark_blocks.iter().enumerate() {
            let inv_block = &self.landmark_block_inverses[block_idx];
            let local_start = start_col - structure.landmark_col_range().0;

            for i in 0..3 {
                let mut sum = 0.0;
                for j in 0..3 {
                    sum += inv_block[(i, j)] * input[(local_start + j, 0)];
                }
                output[(local_start + i, 0)] = sum;
            }
        }

        Ok(())
    }

    /// Compute block-Jacobi preconditioner: inverts camera diagonal blocks of H_cc only
    ///
    /// Instead of scalar diagonal (1/H_ii), this inverts the full camera blocks.
    /// For cameras with 6 DOF (SE3), this creates 6×6 inverse blocks.
    ///
    /// NOTE: This is NOT the true Schur-Jacobi preconditioner. It only uses
    /// diagonal blocks of H_cc, not the Schur complement S. For better convergence,
    /// use `compute_schur_jacobi_preconditioner()` instead.
    fn compute_block_preconditioner(&self) -> LinAlgResult<Vec<DMatrix<f64>>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;

        let hessian = self
            .hessian
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Hessian not computed".into()))?;

        let symbolic = hessian.symbolic();

        let mut precond_blocks = Vec::with_capacity(structure.camera_blocks.len());

        for (_, start_col, size) in &structure.camera_blocks {
            // Extract the diagonal block for this camera
            let mut block = DMatrix::<f64>::zeros(*size, *size);

            for local_col in 0..*size {
                let global_col = start_col + local_col;
                let row_indices = symbolic.row_idx_of_col_raw(global_col);
                let col_values = hessian.val_of_col(global_col);

                for (idx, &global_row) in row_indices.iter().enumerate() {
                    if global_row >= *start_col && global_row < start_col + size {
                        let local_row = global_row - start_col;
                        block[(local_row, local_col)] = col_values[idx];
                    }
                }
            }

            // Invert with regularization for numerical stability
            let inv_block = match block.clone().try_inverse() {
                Some(inv) => inv,
                None => {
                    // Add regularization and retry
                    let reg = 1e-6 * block.diagonal().iter().sum::<f64>().abs() / *size as f64;
                    let reg = reg.max(1e-8);
                    for i in 0..*size {
                        block[(i, i)] += reg;
                    }
                    block
                        .try_inverse()
                        .unwrap_or_else(|| DMatrix::identity(*size, *size))
                }
            };

            precond_blocks.push(inv_block);
        }

        Ok(precond_blocks)
    }

    /// Apply block-Jacobi preconditioner: z = M^{-1} * r
    ///
    /// For each camera block, multiply by the inverse block matrix.
    fn apply_block_preconditioner(
        &self,
        r: &Mat<f64>,
        precond_blocks: &[DMatrix<f64>],
    ) -> LinAlgResult<Mat<f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;

        let cam_dof = structure.camera_dof;
        let mut z = Mat::<f64>::zeros(cam_dof, 1);
        let cam_start = structure.camera_col_range().0;

        for (block_idx, (_, start_col, size)) in structure.camera_blocks.iter().enumerate() {
            let local_start = start_col - cam_start;
            let inv_block = &precond_blocks[block_idx];

            // Extract r block as DVector
            let mut r_block = DVector::<f64>::zeros(*size);
            for i in 0..*size {
                r_block[i] = r[(local_start + i, 0)];
            }

            // Apply inverse: z_block = M^{-1} * r_block
            let z_block = inv_block * r_block;

            // Write back to z
            for i in 0..*size {
                z[(local_start + i, 0)] = z_block[i];
            }
        }

        Ok(z)
    }

    /// Compute TRUE Schur-Jacobi preconditioner: diagonal blocks of the Schur complement S
    ///
    /// This is what Ceres Solver uses for SCHUR_JACOBI preconditioner.
    ///
    /// For each camera i:
    ///   S[i,i] = H_cc[i,i] - Σ_j H_cp[i,j] * H_pp[j,j]^{-1} * H_cp[i,j]^T
    ///
    /// where the sum is over all landmarks j observed by camera i.
    ///
    /// This preconditioner captures the effect of point elimination on each camera block,
    /// leading to much faster PCG convergence (typically 20-40 iterations vs 100+).
    fn compute_schur_jacobi_preconditioner(&self) -> LinAlgResult<Vec<DMatrix<f64>>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;

        let hessian = self
            .hessian
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Hessian not computed".into()))?;

        let symbolic = hessian.symbolic();

        // Borrow visibility index for use in parallel iterator
        let visibility = &self.camera_to_landmark_visibility;

        // Compute S[i,i] for each camera in parallel
        // S[i,i] = H_cc[i,i] - Σ_j H_cp[i,j] * H_pp[j,j]^{-1} * H_cp[i,j]^T
        // Using visibility index: only iterate over connected landmarks (O(observations) instead of O(cameras * landmarks))
        let precond_blocks: Vec<DMatrix<f64>> = structure
            .camera_blocks
            .par_iter()
            .enumerate()
            .map(|(cam_idx, (_, cam_col_start, cam_size))| {
                // Step 1: Extract H_cc[i,i] diagonal block
                let mut s_ii = DMatrix::<f64>::zeros(*cam_size, *cam_size);

                for local_col in 0..*cam_size {
                    let global_col = cam_col_start + local_col;
                    let row_indices = symbolic.row_idx_of_col_raw(global_col);
                    let col_values = hessian.val_of_col(global_col);

                    for (idx, &global_row) in row_indices.iter().enumerate() {
                        if global_row >= *cam_col_start && global_row < cam_col_start + cam_size {
                            let local_row = global_row - cam_col_start;
                            s_ii[(local_row, local_col)] = col_values[idx];
                        }
                    }
                }

                // Step 2: For each landmark OBSERVED by this camera (visibility-indexed)
                // This is the key optimization: O(avg_landmarks_per_camera) instead of O(all_landmarks)
                let visible_landmarks = if cam_idx < visibility.len() {
                    &visibility[cam_idx]
                } else {
                    &Vec::new() as &Vec<usize>
                };

                for &lm_block_idx in visible_landmarks {
                    if lm_block_idx >= structure.landmark_blocks.len() {
                        continue;
                    }
                    let (_, lm_col_start, _) = &structure.landmark_blocks[lm_block_idx];

                    // Extract H_cp[i,j] block (cam_size x 3)
                    let mut h_cp = DMatrix::<f64>::zeros(*cam_size, 3);

                    for col_offset in 0..3 {
                        let global_col = lm_col_start + col_offset;
                        let row_indices = symbolic.row_idx_of_col_raw(global_col);
                        let col_values = hessian.val_of_col(global_col);

                        for (idx, &global_row) in row_indices.iter().enumerate() {
                            if global_row >= *cam_col_start && global_row < cam_col_start + cam_size
                            {
                                let local_row = global_row - cam_col_start;
                                h_cp[(local_row, col_offset)] = col_values[idx];
                            }
                        }
                    }

                    // Get H_pp[j,j]^{-1} from cached inverses
                    let hpp_inv = &self.landmark_block_inverses[lm_block_idx];

                    // Compute contribution: H_cp * H_pp^{-1} * H_cp^T
                    // First: temp = H_cp * H_pp^{-1} (cam_size x 3)
                    let mut temp = DMatrix::<f64>::zeros(*cam_size, 3);
                    for i in 0..*cam_size {
                        for j in 0..3 {
                            let mut sum = 0.0;
                            for k in 0..3 {
                                sum += h_cp[(i, k)] * hpp_inv[(k, j)];
                            }
                            temp[(i, j)] = sum;
                        }
                    }

                    // Then: contribution = temp * H_cp^T (cam_size x cam_size)
                    for i in 0..*cam_size {
                        for j in 0..*cam_size {
                            let mut sum = 0.0;
                            for k in 0..3 {
                                sum += temp[(i, k)] * h_cp[(j, k)];
                            }
                            s_ii[(i, j)] -= sum;
                        }
                    }
                }

                // Step 3: Invert S[i,i] with regularization if needed
                match s_ii.clone().try_inverse() {
                    Some(inv) => inv,
                    None => {
                        // Add regularization and retry
                        let trace = s_ii.trace();
                        let reg = (1e-6 * trace.abs() / *cam_size as f64).max(1e-8);
                        for i in 0..*cam_size {
                            s_ii[(i, i)] += reg;
                        }
                        s_ii.try_inverse()
                            .unwrap_or_else(|| DMatrix::identity(*cam_size, *cam_size))
                    }
                }
            })
            .collect();

        Ok(precond_blocks)
    }

    /// Solve S*x = b using Preconditioned Conjugate Gradients with block preconditioner
    /// Uses optimized Schur operator with workspace buffers to minimize allocations.
    fn solve_pcg_block(
        &self,
        b: &Mat<f64>,
        precond_blocks: &[DMatrix<f64>],
        workspace_lm: &mut [f64],
        workspace_cam: &mut [f64],
    ) -> LinAlgResult<Mat<f64>> {
        let cam_dof = b.nrows();
        let mut x = Mat::<f64>::zeros(cam_dof, 1);

        // r = b - S*x (x starts at 0, so r = b)
        let mut r = b.clone();

        // z = M^{-1} * r (block preconditioner)
        let mut z = self.apply_block_preconditioner(&r, precond_blocks)?;

        let mut p = z.clone();
        let mut rz_old = 0.0;
        for i in 0..cam_dof {
            rz_old += r[(i, 0)] * z[(i, 0)];
        }

        // Compute initial residual norm for relative convergence
        let b_norm: f64 = (0..cam_dof)
            .map(|i| b[(i, 0)] * b[(i, 0)])
            .sum::<f64>()
            .sqrt();
        let tol = self.cg_tolerance * b_norm.max(1.0);

        // Ap buffer (reused each iteration)
        let mut ap = Mat::<f64>::zeros(cam_dof, 1);

        for iter in 0..self.max_cg_iterations {
            // Ap = S * p (using fast operator with workspace buffers)
            // Reset ap to zeros
            for i in 0..cam_dof {
                ap[(i, 0)] = 0.0;
            }
            self.apply_schur_operator_fast(&p, &mut ap, workspace_lm, workspace_cam)?;

            // alpha = (r^T z) / (p^T Ap)
            let mut p_ap = 0.0;
            for i in 0..cam_dof {
                p_ap += p[(i, 0)] * ap[(i, 0)];
            }

            if p_ap.abs() < 1e-20 {
                tracing::debug!("PCG: p^T*A*p near zero at iteration {}", iter);
                break;
            }

            let alpha = rz_old / p_ap;

            // x = x + alpha * p
            for i in 0..cam_dof {
                x[(i, 0)] += alpha * p[(i, 0)];
            }

            // r = r - alpha * Ap
            for i in 0..cam_dof {
                r[(i, 0)] -= alpha * ap[(i, 0)];
            }

            // Check convergence
            let r_norm: f64 = (0..cam_dof)
                .map(|i| r[(i, 0)] * r[(i, 0)])
                .sum::<f64>()
                .sqrt();

            if r_norm < tol {
                tracing::debug!(
                    "PCG converged in {} iterations (residual={:.2e})",
                    iter + 1,
                    r_norm
                );
                break;
            }

            // z = M^{-1} * r (block preconditioner)
            z = self.apply_block_preconditioner(&r, precond_blocks)?;

            // beta = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)
            let mut rz_new = 0.0;
            for i in 0..cam_dof {
                rz_new += r[(i, 0)] * z[(i, 0)];
            }

            if rz_old.abs() < 1e-30 {
                break;
            }

            let beta = rz_new / rz_old;

            // p = z + beta * p
            for i in 0..cam_dof {
                p[(i, 0)] = z[(i, 0)] + beta * p[(i, 0)];
            }

            rz_old = rz_new;
        }

        Ok(x)
    }

    /// Extract 3x3 diagonal blocks from H_pp and invert them with numerical robustness
    ///
    /// This function uses parallel processing for the block inversions (156K+ blocks).
    /// Each block's condition number is checked and regularization applied as needed.
    fn invert_landmark_blocks(&mut self, hessian: &SparseColMat<usize, f64>) -> LinAlgResult<()> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;

        let symbolic = hessian.symbolic();

        // Step 1: Extract all 3x3 blocks (sequential - requires sparse matrix access)
        let blocks: Vec<(usize, Matrix3<f64>)> = structure
            .landmark_blocks
            .iter()
            .enumerate()
            .map(|(i, (_, start_col, _))| {
                let mut block = Matrix3::<f64>::zeros();

                for local_col in 0..3 {
                    let global_col = start_col + local_col;
                    let row_indices = symbolic.row_idx_of_col_raw(global_col);
                    let col_values = hessian.val_of_col(global_col);

                    for (idx, &row) in row_indices.iter().enumerate() {
                        if row >= *start_col && row < start_col + 3 {
                            let local_row = row - start_col;
                            block[(local_row, local_col)] = col_values[idx];
                        }
                    }
                }

                (i, block)
            })
            .collect();

        // Step 2: Invert all blocks in parallel
        // Thresholds for numerical robustness
        const CONDITION_THRESHOLD: f64 = 1e10;
        const MIN_EIGENVALUE_THRESHOLD: f64 = 1e-12;
        const REGULARIZATION_SCALE: f64 = 1e-6;

        let results: Vec<Result<Matrix3<f64>, (usize, String)>> = blocks
            .par_iter()
            .map(|(i, block)| {
                // Check conditioning and apply regularization if needed
                let eigenvalues = block.symmetric_eigenvalues();
                let min_ev = eigenvalues.min();
                let max_ev = eigenvalues.max();

                if min_ev < MIN_EIGENVALUE_THRESHOLD {
                    // Severely ill-conditioned: add strong regularization
                    let reg = REGULARIZATION_SCALE + max_ev * REGULARIZATION_SCALE;
                    let regularized = block + Matrix3::identity() * reg;
                    regularized.try_inverse().ok_or_else(|| {
                        (
                            *i,
                            format!("singular even with regularization (min_ev={:.2e})", min_ev),
                        )
                    })
                } else if max_ev / min_ev > CONDITION_THRESHOLD {
                    // Ill-conditioned: add moderate regularization
                    let extra_reg = max_ev * REGULARIZATION_SCALE;
                    let regularized = block + Matrix3::identity() * extra_reg;
                    regularized.try_inverse().ok_or_else(|| {
                        (
                            *i,
                            format!("ill-conditioned (cond={:.2e})", max_ev / min_ev),
                        )
                    })
                } else {
                    // Well-conditioned: standard inversion
                    block
                        .try_inverse()
                        .ok_or_else(|| (*i, "singular".to_string()))
                }
            })
            .collect();

        // Step 3: Collect results and check for errors
        self.landmark_block_inverses.clear();
        self.landmark_block_inverses.reserve(results.len());

        for result in results {
            match result {
                Ok(inv) => self.landmark_block_inverses.push(inv),
                Err((i, msg)) => {
                    return Err(LinAlgError::SingularMatrix(format!(
                        "Landmark block {} {}",
                        i, msg
                    )));
                }
            }
        }

        Ok(())
    }

    /// Build camera->landmark visibility index from H_cp structure
    ///
    /// This scans the Hessian to find which landmarks each camera observes,
    /// enabling O(observations) preconditioner computation instead of O(cameras * landmarks).
    fn build_visibility_index(&mut self, hessian: &SparseColMat<usize, f64>) -> LinAlgResult<()> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;

        let symbolic = hessian.symbolic();
        let (cam_start, cam_end) = structure.camera_col_range();
        let num_cameras = structure.camera_blocks.len();

        // Build a map from global camera row -> camera block index
        let mut cam_row_to_block: HashMap<usize, usize> = HashMap::new();
        for (cam_idx, (_, start_col, size)) in structure.camera_blocks.iter().enumerate() {
            for offset in 0..*size {
                cam_row_to_block.insert(start_col + offset, cam_idx);
            }
        }

        // Initialize visibility: one vec per camera
        let mut visibility: Vec<Vec<usize>> = vec![Vec::new(); num_cameras];

        // Scan landmark columns to find camera connections
        for (lm_block_idx, (_, lm_col_start, _)) in structure.landmark_blocks.iter().enumerate() {
            // Check first column of this landmark block (all 3 columns have same row pattern)
            let global_col = *lm_col_start;
            if global_col >= hessian.ncols() {
                continue;
            }

            let row_indices = symbolic.row_idx_of_col_raw(global_col);

            // Find which cameras observe this landmark
            for &row in row_indices {
                if row >= cam_start
                    && row < cam_end
                    && let Some(&cam_idx) = cam_row_to_block.get(&row)
                {
                    // Only add if not already present (avoid duplicates)
                    if visibility[cam_idx].last() != Some(&lm_block_idx) {
                        visibility[cam_idx].push(lm_block_idx);
                    }
                }
            }
        }

        self.camera_to_landmark_visibility = visibility;
        Ok(())
    }

    /// Internal solve using the already-cached Hessian and gradient.
    /// This avoids rebuilding the Hessian which would lose the damping from solve_augmented_equation.
    fn solve_with_cached_hessian(&mut self) -> LinAlgResult<Mat<f64>> {
        let hessian = self
            .hessian
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Hessian not cached".into()))?
            .clone();
        let gradient = self
            .gradient
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Gradient not cached".into()))?
            .clone();

        // Extract structure info
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;
        let cam_dof = structure.camera_dof;
        let lm_dof = structure.landmark_dof;
        let (cam_start, _cam_end) = structure.camera_col_range();
        let (lm_start, _lm_end) = structure.landmark_col_range();

        // Invert landmark blocks
        self.invert_landmark_blocks(&hessian)?;

        // Build visibility index for efficient preconditioner computation
        self.build_visibility_index(&hessian)?;

        // Extract reduced RHS: g_c - H_cp * H_pp^{-1} * g_p
        let mut g_reduced = Mat::<f64>::zeros(cam_dof, 1);
        for i in 0..cam_dof {
            g_reduced[(i, 0)] = gradient[(cam_start + i, 0)];
        }

        let mut g_lm = Mat::<f64>::zeros(lm_dof, 1);
        for i in 0..lm_dof {
            g_lm[(i, 0)] = gradient[(lm_start + i, 0)];
        }

        let mut temp = Mat::<f64>::zeros(lm_dof, 1);
        self.apply_landmark_inverse(&g_lm, &mut temp)?;

        let correction = self.extract_camera_landmark_mvp(&hessian, &temp)?;
        for i in 0..cam_dof {
            g_reduced[(i, 0)] -= correction[(i, 0)];
        }

        // Initialize workspace buffers if needed
        self.init_workspaces();

        // Solve S*δc = g_reduced using PCG with appropriate preconditioner
        let precond_blocks = match self.preconditioner_type {
            SchurPreconditioner::SchurJacobi => {
                // True Schur-Jacobi: diagonal blocks of S (Ceres-style, best convergence)
                self.compute_schur_jacobi_preconditioner()?
            }
            SchurPreconditioner::BlockDiagonal => {
                // Block diagonal of H_cc only (faster to compute, worse convergence)
                self.compute_block_preconditioner()?
            }
            SchurPreconditioner::None => {
                // Identity preconditioner (for debugging)
                let structure = self.block_structure.as_ref().ok_or_else(|| {
                    LinAlgError::InvalidInput("Block structure not initialized".into())
                })?;
                structure
                    .camera_blocks
                    .iter()
                    .map(|(_, _, size)| DMatrix::identity(*size, *size))
                    .collect()
            }
        };

        // Use workspace buffers for PCG iterations
        let mut workspace_lm = std::mem::take(&mut self.workspace_lm);
        let mut workspace_cam = std::mem::take(&mut self.workspace_cam);

        let delta_cam = self.solve_pcg_block(
            &g_reduced,
            &precond_blocks,
            &mut workspace_lm,
            &mut workspace_cam,
        )?;

        // Restore workspace buffers
        self.workspace_lm = workspace_lm;
        self.workspace_cam = workspace_cam;

        // Back-substitute for landmarks
        let hcp_t_delta_cam = self.extract_camera_landmark_transpose_mvp(&hessian, &delta_cam)?;

        let mut rhs_lm = Mat::<f64>::zeros(lm_dof, 1);
        for i in 0..lm_dof {
            rhs_lm[(i, 0)] = g_lm[(i, 0)] - hcp_t_delta_cam[(i, 0)];
        }

        let mut delta_lm = Mat::<f64>::zeros(lm_dof, 1);
        self.apply_landmark_inverse(&rhs_lm, &mut delta_lm)?;

        // Combine camera and landmark updates
        let total_dof = cam_dof + lm_dof;
        let mut delta = Mat::<f64>::zeros(total_dof, 1);

        for i in 0..cam_dof {
            delta[(cam_start + i, 0)] = delta_cam[(i, 0)];
        }
        for i in 0..lm_dof {
            delta[(lm_start + i, 0)] = delta_lm[(i, 0)];
        }

        Ok(delta)
    }
}

impl Default for IterativeSchurSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl StructuredSparseLinearSolver for IterativeSchurSolver {
    fn initialize_structure(
        &mut self,
        variables: &HashMap<String, VariableEnum>,
        variable_index_map: &HashMap<String, usize>,
    ) -> LinAlgResult<()> {
        let mut structure = SchurBlockStructure::new();

        for (name, variable) in variables {
            let manifold_type = variable.manifold_type();
            let start_col = *variable_index_map.get(name).ok_or_else(|| {
                LinAlgError::InvalidInput(format!("Variable {} not in index map", name))
            })?;
            let size = variable.get_size();

            if self.ordering.should_eliminate(name, &manifold_type, size) {
                structure
                    .landmark_blocks
                    .push((name.clone(), start_col, size));
                structure.landmark_dof += size;

                if size != 3 {
                    return Err(LinAlgError::InvalidInput(format!(
                        "Landmark {} has DOF {}, expected 3",
                        name, size
                    )));
                }
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
                "No camera variables found".into(),
            ));
        }
        if structure.landmark_blocks.is_empty() {
            return Err(LinAlgError::InvalidInput(
                "No landmark variables found".into(),
            ));
        }

        self.block_structure = Some(structure);
        Ok(())
    }

    fn solve_normal_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        // Build H = J^T * J, g = -J^T * r
        let jt = jacobian
            .transpose()
            .to_col_major()
            .map_err(|e| LinAlgError::MatrixConversion(format!("Transpose failed: {:?}", e)))?;
        let hessian = jt.mul(jacobian);
        let jtr = jacobian.transpose().mul(residuals);
        let mut gradient = Mat::<f64>::zeros(jtr.nrows(), 1);
        for i in 0..jtr.nrows() {
            gradient[(i, 0)] = -jtr[(i, 0)];
        }

        self.hessian = Some(hessian);
        self.gradient = Some(gradient);

        // Solve using the cached Hessian
        self.solve_with_cached_hessian()
    }

    fn solve_augmented_equation(
        &mut self,
        residuals: &Mat<f64>,
        jacobian: &SparseColMat<usize, f64>,
        lambda: f64,
    ) -> LinAlgResult<Mat<f64>> {
        // Build H = J^T * J + λI
        let jt = jacobian
            .transpose()
            .to_col_major()
            .map_err(|e| LinAlgError::MatrixConversion(format!("Transpose failed: {:?}", e)))?;
        let jtr = jt.mul(residuals);
        let mut hessian = jacobian
            .transpose()
            .to_col_major()
            .map_err(|e| LinAlgError::MatrixConversion(format!("Transpose failed: {:?}", e)))?
            .mul(jacobian);

        // Add damping to diagonal
        let n = hessian.ncols();
        let symbolic = hessian.symbolic();
        let mut triplets = Vec::new();

        for col in 0..n {
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = hessian.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                triplets.push(Triplet::new(row, col, col_values[idx]));
            }

            // Add lambda to diagonal
            triplets.push(Triplet::new(col, col, lambda));
        }

        hessian = SparseColMat::try_new_from_triplets(n, n, &triplets).map_err(|e| {
            LinAlgError::InvalidInput(format!("Failed to build damped Hessian: {:?}", e))
        })?;

        let mut gradient = Mat::<f64>::zeros(jtr.nrows(), 1);
        for i in 0..jtr.nrows() {
            gradient[(i, 0)] = -jtr[(i, 0)];
        }

        self.hessian = Some(hessian);
        self.gradient = Some(gradient.clone());

        // Solve using the cached damped Hessian (don't call solve_normal_equation
        // which would rebuild the Hessian without damping)
        self.solve_with_cached_hessian()
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

    #[test]
    fn test_iterative_schur_creation() {
        let solver = IterativeSchurSolver::new();
        // Default: 500 max iterations, 1e-9 tolerance, Schur-Jacobi preconditioner
        assert_eq!(solver.max_cg_iterations, 500);
        assert_eq!(solver.cg_tolerance, 1e-9);
        assert_eq!(solver.preconditioner_type, SchurPreconditioner::SchurJacobi);
    }

    #[test]
    fn test_with_custom_params() {
        let solver = IterativeSchurSolver::with_cg_params(100, 1e-8);
        assert_eq!(solver.max_cg_iterations, 100);
        assert_eq!(solver.cg_tolerance, 1e-8);
        // Should still use default Schur-Jacobi preconditioner
        assert_eq!(solver.preconditioner_type, SchurPreconditioner::SchurJacobi);
    }

    #[test]
    fn test_with_full_config() {
        let solver =
            IterativeSchurSolver::with_config(200, 1e-10, SchurPreconditioner::BlockDiagonal);
        assert_eq!(solver.max_cg_iterations, 200);
        assert_eq!(solver.cg_tolerance, 1e-10);
        assert_eq!(
            solver.preconditioner_type,
            SchurPreconditioner::BlockDiagonal
        );
    }
}
