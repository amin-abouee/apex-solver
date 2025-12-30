//! Power Series Schur Complement Solver (PSSC / PoBA)
//!
//! This variant approximates H_pp^{-1} using a power series expansion, avoiding
//! explicit inversion of landmark blocks. This is memory-efficient for problems
//! with many landmarks.
//!
//! # Algorithm
//!
//! Instead of computing H_pp^{-1} explicitly, approximate it as:
//! H_pp^{-1} ≈ D^{-1} * (I + sum_{k=1}^{n} (I - D^{-1}*H_pp)^k)
//!
//! where D is the block diagonal of H_pp.
//!
//! This allows computing H_pp^{-1} * v iteratively without forming the full inverse.
//!
//! # Reference
//!
//! Based on Ceres Solver's POWER_SERIES_SCHUR and PoBA (Power Bundle Adjustment)
//! - Byröd & Åström, "Conjugate Gradient Bundle Adjustment", ECCV 2010

use super::schur::{SchurBlockStructure, SchurOrdering};
use crate::core::problem::VariableEnum;
use crate::linalg::{LinAlgError, LinAlgResult, StructuredSparseLinearSolver};
use faer::Mat;
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::Matrix3;
use std::collections::HashMap;
use std::ops::Mul;

/// Power series Schur complement solver
#[derive(Debug, Clone)]
pub struct PowerSeriesSchurSolver {
    block_structure: Option<SchurBlockStructure>,
    ordering: SchurOrdering,

    // Power series parameters
    max_series_terms: usize,
    series_tolerance: f64,

    // Cached diagonal blocks of H_pp
    landmark_diagonal_blocks: Vec<Matrix3<f64>>,
    landmark_diagonal_inverses: Vec<Matrix3<f64>>,

    // Cached matrices
    hessian: Option<SparseColMat<usize, f64>>,
    gradient: Option<Mat<f64>>,
}

impl PowerSeriesSchurSolver {
    /// Create a new power series Schur solver with default parameters
    pub fn new() -> Self {
        Self {
            block_structure: None,
            ordering: SchurOrdering::default(),
            max_series_terms: 5,
            series_tolerance: 1e-6,
            landmark_diagonal_blocks: Vec::new(),
            landmark_diagonal_inverses: Vec::new(),
            hessian: None,
            gradient: None,
        }
    }

    /// Create solver with custom power series parameters
    ///
    /// # Arguments
    ///
    /// * `max_terms` - Maximum number of terms in power series (typically 3-10)
    /// * `tolerance` - Convergence tolerance for power series
    pub fn with_series_params(max_terms: usize, tolerance: f64) -> Self {
        Self {
            block_structure: None,
            ordering: SchurOrdering::default(),
            max_series_terms: max_terms,
            series_tolerance: tolerance,
            landmark_diagonal_blocks: Vec::new(),
            landmark_diagonal_inverses: Vec::new(),
            hessian: None,
            gradient: None,
        }
    }

    /// Extract diagonal 3x3 blocks from H_pp
    fn extract_landmark_diagonal_blocks(
        &mut self,
        hessian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<()> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;

        self.landmark_diagonal_blocks.clear();
        self.landmark_diagonal_inverses.clear();
        self.landmark_diagonal_blocks
            .reserve(structure.num_landmarks);
        self.landmark_diagonal_inverses
            .reserve(structure.num_landmarks);

        let symbolic = hessian.symbolic();

        for (_, start_col, _) in &structure.landmark_blocks {
            let mut block = Matrix3::<f64>::zeros();

            // Extract diagonal block
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

            // Invert diagonal block
            let inv_block = block.try_inverse().ok_or_else(|| {
                LinAlgError::SingularMatrix("Landmark diagonal block singular".into())
            })?;

            self.landmark_diagonal_blocks.push(block);
            self.landmark_diagonal_inverses.push(inv_block);
        }

        Ok(())
    }

    /// Apply H_pp to a vector (landmark block only)
    fn apply_landmark_hessian(&self, input: &Mat<f64>, output: &mut Mat<f64>) -> LinAlgResult<()> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;
        let hessian = self
            .hessian
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Hessian not initialized".into()))?;
        let (lm_start, lm_end) = structure.landmark_col_range();
        let symbolic = hessian.symbolic();

        // Zero output
        for i in 0..output.nrows() {
            output[(i, 0)] = 0.0;
        }

        // Multiply H_pp submatrix with input
        for col in lm_start..lm_end {
            let local_col = col - lm_start;
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = hessian.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                if row >= lm_start && row < lm_end {
                    let local_row = row - lm_start;
                    output[(local_row, 0)] += col_values[idx] * input[(local_col, 0)];
                }
            }
        }

        Ok(())
    }

    /// Apply diagonal inverse D^{-1} to a vector
    fn apply_diagonal_inverse(&self, input: &Mat<f64>, output: &mut Mat<f64>) -> LinAlgResult<()> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;
        let lm_start = structure.landmark_col_range().0;

        for (block_idx, (_, start_col, _)) in structure.landmark_blocks.iter().enumerate() {
            let inv_block = &self.landmark_diagonal_inverses[block_idx];
            let local_start = start_col - lm_start;

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

    /// Approximate H_pp^{-1} * v using power series expansion
    ///
    /// Computes: v_out = D^{-1} * (I + sum_{k=1}^{n} (I - D^{-1}*H_pp)^k) * v
    ///
    /// Iteratively:
    ///   w_0 = D^{-1} * v
    ///   w_k = D^{-1} * (v - H_pp * w_{k-1})
    ///   result = sum w_k
    fn apply_landmark_inverse_power_series(
        &self,
        input: &Mat<f64>,
        output: &mut Mat<f64>,
    ) -> LinAlgResult<()> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;
        let lm_dof = structure.landmark_dof;

        // w_0 = D^{-1} * v
        let mut w_current = Mat::<f64>::zeros(lm_dof, 1);
        self.apply_diagonal_inverse(input, &mut w_current)?;

        // Initialize result with w_0
        for i in 0..lm_dof {
            output[(i, 0)] = w_current[(i, 0)];
        }

        let mut temp = Mat::<f64>::zeros(lm_dof, 1);

        // Iterate power series terms
        for _k in 1..=self.max_series_terms {
            // temp = H_pp * w_current
            self.apply_landmark_hessian(&w_current, &mut temp)?;

            // temp = v - temp = v - H_pp * w_current
            for i in 0..lm_dof {
                temp[(i, 0)] = input[(i, 0)] - temp[(i, 0)];
            }

            // w_next = D^{-1} * temp
            let mut w_next = Mat::<f64>::zeros(lm_dof, 1);
            self.apply_diagonal_inverse(&temp, &mut w_next)?;

            // Add w_next to result
            let mut w_norm = 0.0;
            for i in 0..lm_dof {
                output[(i, 0)] += w_next[(i, 0)];
                w_norm += w_next[(i, 0)] * w_next[(i, 0)];
            }
            w_norm = w_norm.sqrt();

            // Check convergence
            if w_norm < self.series_tolerance {
                break;
            }

            w_current = w_next;
        }

        Ok(())
    }

    /// Extract H_cc block and multiply with vector
    #[allow(dead_code)]
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

    /// Extract H_cp^T and multiply with vector
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
}

impl Default for PowerSeriesSchurSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl StructuredSparseLinearSolver for PowerSeriesSchurSolver {
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

        // Extract diagonal blocks and their inverses
        self.extract_landmark_diagonal_blocks(&hessian)?;

        // Extract gradient components
        let mut g_cam = Mat::<f64>::zeros(cam_dof, 1);
        for i in 0..cam_dof {
            g_cam[(i, 0)] = gradient[(cam_start + i, 0)];
        }

        let mut g_lm = Mat::<f64>::zeros(lm_dof, 1);
        for i in 0..lm_dof {
            g_lm[(i, 0)] = gradient[(lm_start + i, 0)];
        }

        // Compute H_pp^{-1} * g_lm using power series
        let mut hpp_inv_g = Mat::<f64>::zeros(lm_dof, 1);
        self.apply_landmark_inverse_power_series(&g_lm, &mut hpp_inv_g)?;

        // Compute reduced RHS: g_c - H_cp * H_pp^{-1} * g_p
        let correction = self.extract_camera_landmark_mvp(
            self.hessian
                .as_ref()
                .ok_or_else(|| LinAlgError::InvalidState("Hessian not initialized".into()))?,
            &hpp_inv_g,
        )?;
        let mut g_reduced = g_cam.clone();
        for i in 0..cam_dof {
            g_reduced[(i, 0)] -= correction[(i, 0)];
        }

        // Form and solve Schur complement system using direct method
        // S = H_cc - H_cp * H_pp^{-1} * H_cp^T
        // For power series, we approximate the Schur complement operations

        // For now, use a simple diagonal approximation for S
        // This is a simplified version - full implementation would compute S explicitly
        // or use iterative methods

        use faer::Side;
        use faer::linalg::solvers::Solve;
        use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};

        // Extract H_cc
        let h_cc = self.extract_camera_hessian_block(
            self.hessian
                .as_ref()
                .ok_or_else(|| LinAlgError::InvalidState("Hessian not initialized".into()))?,
        )?;

        // Approximate Schur complement (simplified - should iterate)
        let symbolic_llt = SymbolicLlt::try_new(h_cc.symbolic(), Side::Lower)
            .map_err(|_| LinAlgError::InvalidInput("Symbolic factorization failed".into()))?;

        let llt =
            Llt::try_new_with_symbolic(symbolic_llt, h_cc.as_ref(), Side::Lower).map_err(|_| {
                LinAlgError::FactorizationFailed("Schur complement factorization failed".into())
            })?;

        let delta_cam = llt.solve(&g_reduced);

        // Back-substitute for landmarks
        let hcp_t_delta_cam = self.extract_camera_landmark_transpose_mvp(
            self.hessian
                .as_ref()
                .ok_or_else(|| LinAlgError::InvalidState("Hessian not initialized".into()))?,
            &delta_cam,
        )?;

        let mut rhs_lm = Mat::<f64>::zeros(lm_dof, 1);
        for i in 0..lm_dof {
            rhs_lm[(i, 0)] = g_lm[(i, 0)] - hcp_t_delta_cam[(i, 0)];
        }

        let mut delta_lm = Mat::<f64>::zeros(lm_dof, 1);
        self.apply_landmark_inverse_power_series(&rhs_lm, &mut delta_lm)?;

        // Combine updates
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

        // Add damping
        let n = hessian.ncols();
        let symbolic = hessian.symbolic();
        let mut triplets = Vec::new();

        for col in 0..n {
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = hessian.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                triplets.push(Triplet::new(row, col, col_values[idx]));
            }

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

        self.solve_normal_equation(residuals, jacobian)
    }

    fn get_hessian(&self) -> Option<&SparseColMat<usize, f64>> {
        self.hessian.as_ref()
    }

    fn get_gradient(&self) -> Option<&Mat<f64>> {
        self.gradient.as_ref()
    }
}

impl PowerSeriesSchurSolver {
    /// Extract H_cc as a separate sparse matrix (helper for factorization)
    fn extract_camera_hessian_block(
        &self,
        hessian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<SparseColMat<usize, f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;
        let (cam_start, cam_end) = structure.camera_col_range();
        let cam_dof = structure.camera_dof;
        let symbolic = hessian.symbolic();

        let mut triplets = Vec::new();

        for col in cam_start..cam_end {
            let local_col = col - cam_start;
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = hessian.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                if row >= cam_start && row < cam_end {
                    let local_row = row - cam_start;
                    triplets.push(Triplet::new(local_row, local_col, col_values[idx]));
                }
            }
        }

        SparseColMat::try_new_from_triplets(cam_dof, cam_dof, &triplets)
            .map_err(|e| LinAlgError::InvalidInput(format!("Failed to extract H_cc: {:?}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_series_creation() {
        let solver = PowerSeriesSchurSolver::new();
        assert_eq!(solver.max_series_terms, 5);
        assert_eq!(solver.series_tolerance, 1e-6);
    }

    #[test]
    fn test_with_custom_params() {
        let solver = PowerSeriesSchurSolver::with_series_params(10, 1e-8);
        assert_eq!(solver.max_series_terms, 10);
        assert_eq!(solver.series_tolerance, 1e-8);
    }
}
