//! Iterative Schur Complement Solver using Preconditioned Conjugate Gradients
//!
//! This variant solves the Schur complement system S*δc = g_reduced using PCG
//! instead of direct factorization, which is faster for large-scale problems.
//!
//! # Algorithm
//!
//! 1. Form Schur complement implicitly: S = H_cc - H_cp * H_pp^{-1} * H_cp^T
//! 2. Solve S*δc = g_reduced using PCG
//! 3. Back-substitute: δp = H_pp^{-1} * (g_p - H_cp^T * δc)
//!
//! # Preconditioner
//!
//! Uses block-diagonal (Schur-Jacobi) preconditioner extracted from diagonal
//! blocks of the Schur complement.

use super::schur::{SchurBlockStructure, SchurOrdering};
use crate::core::problem::VariableEnum;
use crate::linalg::{LinAlgError, LinAlgResult, StructuredSparseLinearSolver};
use faer::Mat;
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::Matrix3;
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

    // Cached for matrix-vector products
    landmark_block_inverses: Vec<Matrix3<f64>>,
    hessian: Option<SparseColMat<usize, f64>>,
    gradient: Option<Mat<f64>>,
}

impl IterativeSchurSolver {
    /// Create a new iterative Schur solver with default parameters
    pub fn new() -> Self {
        Self {
            block_structure: None,
            ordering: SchurOrdering::default(),
            max_cg_iterations: 100,
            cg_tolerance: 1e-6,
            landmark_block_inverses: Vec::new(),
            hessian: None,
            gradient: None,
        }
    }

    /// Create solver with custom CG parameters
    pub fn with_cg_params(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            block_structure: None,
            ordering: SchurOrdering::default(),
            max_cg_iterations: max_iterations,
            cg_tolerance: tolerance,
            landmark_block_inverses: Vec::new(),
            hessian: None,
            gradient: None,
        }
    }

    /// Apply Schur complement operator: S*x = (H_cc - H_cp * H_pp^{-1} * H_cp^T) * x
    ///
    /// This computes the matrix-vector product without explicitly forming S.
    fn apply_schur_operator(&self, x: &Mat<f64>) -> LinAlgResult<Mat<f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;

        let hessian = self
            .hessian
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Hessian not computed".into()))?;

        let cam_dof = structure.camera_dof;
        let lm_dof = structure.landmark_dof;

        // Step 1: y = H_cc * x
        let mut result = self.extract_camera_block_mvp(hessian, x)?;

        // Step 2: temp = H_cp^T * x
        let temp = self.extract_camera_landmark_transpose_mvp(hessian, x)?;

        // Step 3: temp2 = H_pp^{-1} * temp
        let mut temp2 = Mat::<f64>::zeros(lm_dof, 1);
        self.apply_landmark_inverse(&temp, &mut temp2)?;

        // Step 4: temp3 = H_cp * temp2
        let temp3 = self.extract_camera_landmark_mvp(hessian, &temp2)?;

        // Step 5: result = y - temp3 = H_cc*x - H_cp*H_pp^{-1}*H_cp^T*x
        for i in 0..cam_dof {
            result[(i, 0)] -= temp3[(i, 0)];
        }

        Ok(result)
    }

    /// Extract H_cc block and multiply with vector
    fn extract_camera_block_mvp(
        &self,
        hessian: &SparseColMat<usize, f64>,
        x: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;
        let (cam_start, cam_end) = structure.camera_col_range();
        let cam_dof = structure.camera_dof;

        let mut result = Mat::<f64>::zeros(cam_dof, 1);
        let symbolic = hessian.symbolic();

        // Multiply H_cc submatrix with x
        for col in cam_start..cam_end {
            let local_col = col - cam_start;
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

    /// Compute block-diagonal preconditioner (diagonal blocks of Schur complement)
    fn compute_preconditioner(&self) -> LinAlgResult<Vec<f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;

        let hessian = self
            .hessian
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Hessian not computed".into()))?;

        let cam_dof = structure.camera_dof;
        let symbolic = hessian.symbolic();

        // Extract diagonal of Schur complement (simplified: just use H_cc diagonal)
        let mut precond = vec![1.0; cam_dof];
        let (cam_start, cam_end) = structure.camera_col_range();

        for col in cam_start..cam_end {
            let local_col = col - cam_start;
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = hessian.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                if row == col {
                    precond[local_col] = col_values[idx];
                    if precond[local_col].abs() < 1e-12 {
                        precond[local_col] = 1.0;
                    } else {
                        precond[local_col] = 1.0 / precond[local_col];
                    }
                    break;
                }
            }
        }

        Ok(precond)
    }

    /// Solve S*x = b using Preconditioned Conjugate Gradients
    fn solve_pcg(&self, b: &Mat<f64>, precond: &[f64]) -> LinAlgResult<Mat<f64>> {
        let cam_dof = b.nrows();
        let mut x = Mat::<f64>::zeros(cam_dof, 1);

        // r = b - S*x (x starts at 0, so r = b)
        let mut r = b.clone();

        // z = M^{-1} * r
        let mut z = Mat::<f64>::zeros(cam_dof, 1);
        for i in 0..cam_dof {
            z[(i, 0)] = precond[i] * r[(i, 0)];
        }

        let mut p = z.clone();
        let mut rz_old = 0.0;
        for i in 0..cam_dof {
            rz_old += r[(i, 0)] * z[(i, 0)];
        }

        for _iter in 0..self.max_cg_iterations {
            // Ap = S * p
            let ap = self.apply_schur_operator(&p)?;

            // alpha = (r^T z) / (p^T Ap)
            let mut p_ap = 0.0;
            for i in 0..cam_dof {
                p_ap += p[(i, 0)] * ap[(i, 0)];
            }

            if p_ap.abs() < 1e-20 {
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
            let mut r_norm = 0.0;
            for i in 0..cam_dof {
                r_norm += r[(i, 0)] * r[(i, 0)];
            }
            r_norm = r_norm.sqrt();

            if r_norm < self.cg_tolerance {
                break;
            }

            // z = M^{-1} * r
            for i in 0..cam_dof {
                z[(i, 0)] = precond[i] * r[(i, 0)];
            }

            // beta = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)
            let mut rz_new = 0.0;
            for i in 0..cam_dof {
                rz_new += r[(i, 0)] * z[(i, 0)];
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

    /// Extract 3x3 diagonal blocks from H_pp and invert them
    fn invert_landmark_blocks(&mut self, hessian: &SparseColMat<usize, f64>) -> LinAlgResult<()> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;

        self.landmark_block_inverses.clear();
        self.landmark_block_inverses
            .reserve(structure.num_landmarks);

        let symbolic = hessian.symbolic();

        for (_, start_col, _) in &structure.landmark_blocks {
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

            let inv_block = block
                .try_inverse()
                .ok_or_else(|| LinAlgError::SingularMatrix("Landmark block singular".into()))?;

            self.landmark_block_inverses.push(inv_block);
        }

        Ok(())
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

            if self.ordering.should_eliminate(&manifold_type) {
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

        self.hessian = Some(hessian.clone());
        self.gradient = Some(gradient.clone());

        // Extract structure info before mutable borrow
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

        let correction = self.extract_camera_landmark_mvp(
            self.hessian
                .as_ref()
                .ok_or_else(|| LinAlgError::InvalidInput("Hessian not initialized".into()))?,
            &temp,
        )?;
        for i in 0..cam_dof {
            g_reduced[(i, 0)] -= correction[(i, 0)];
        }

        // Solve S*δc = g_reduced using PCG
        let precond = self.compute_preconditioner()?;
        let delta_cam = self.solve_pcg(&g_reduced, &precond)?;

        // Back-substitute for landmarks
        let hcp_t_delta_cam = self.extract_camera_landmark_transpose_mvp(
            self.hessian
                .as_ref()
                .ok_or_else(|| LinAlgError::InvalidInput("Hessian not initialized".into()))?,
            &delta_cam,
        )?;

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

        // Continue with normal solve
        self.solve_normal_equation(residuals, jacobian)
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
        assert_eq!(solver.max_cg_iterations, 100);
        assert_eq!(solver.cg_tolerance, 1e-6);
    }

    #[test]
    fn test_with_custom_params() {
        let solver = IterativeSchurSolver::with_cg_params(50, 1e-8);
        assert_eq!(solver.max_cg_iterations, 50);
        assert_eq!(solver.cg_tolerance, 1e-8);
    }
}
