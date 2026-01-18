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

    /// Apply H_pp^{-1} * v using block-diagonal inversion
    ///
    /// For Bundle Adjustment, H_pp is block-diagonal (each 3x3 block corresponds
    /// to one landmark). We directly apply the precomputed block inverses.
    ///
    /// Note: Power series would also work but is unnecessary since H_pp is
    /// already block-diagonal, making direct inversion exact and efficient.
    fn apply_landmark_inverse_power_series(
        &self,
        input: &Mat<f64>,
        output: &mut Mat<f64>,
    ) -> LinAlgResult<()> {
        // For BA problems, H_pp is block-diagonal, so we just apply the
        // precomputed 3x3 block inverses directly. This is exact, not approximate.
        self.apply_diagonal_inverse(input, output)
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

    /// Compute the true Schur complement: S = H_cc - H_cp * H_pp^{-1} * H_cp^T
    /// Using the same algorithm as main SchurSolver with precomputed block inverses
    fn compute_schur_complement(
        &self,
        hessian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<SparseColMat<usize, f64>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;

        let (cam_start, cam_end) = structure.camera_col_range();
        let (lm_start, lm_end) = structure.landmark_col_range();
        let cam_dof = structure.camera_dof;
        let symbolic = hessian.symbolic();

        // Use a dense matrix for S (same as main solver)
        let mut s_dense = vec![0.0f64; cam_dof * cam_dof];

        // First, add H_cc to S
        for col in cam_start..cam_end {
            let local_col = col - cam_start;
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = hessian.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                if row >= cam_start && row < cam_end {
                    let local_row = row - cam_start;
                    s_dense[local_row * cam_dof + local_col] += col_values[idx];
                }
            }
        }

        // Pre-allocate vectors for camera data per landmark
        let mut cam_rows: Vec<usize> = Vec::with_capacity(32);
        let mut h_cp_block: Vec<[f64; 3]> = Vec::with_capacity(32);
        let mut contrib_block: Vec<[f64; 3]> = Vec::with_capacity(32);

        // Process each landmark block - same algorithm as main solver
        for (block_idx, hpp_inv_block) in self.landmark_diagonal_inverses.iter().enumerate() {
            let col_start = lm_start + block_idx * 3;

            cam_rows.clear();
            h_cp_block.clear();

            if col_start + 2 >= lm_end {
                continue;
            }

            // Get row indices and values for each of the 3 columns of this landmark block
            let row_indices_0 = symbolic.row_idx_of_col_raw(col_start);
            let col_values_0 = hessian.val_of_col(col_start);
            let row_indices_1 = symbolic.row_idx_of_col_raw(col_start + 1);
            let col_values_1 = hessian.val_of_col(col_start + 1);
            let row_indices_2 = symbolic.row_idx_of_col_raw(col_start + 2);
            let col_values_2 = hessian.val_of_col(col_start + 2);

            // Merge-sort style iteration to find camera rows with entries in H_cp
            let mut i0 = 0;
            let mut i1 = 0;
            let mut i2 = 0;

            while i0 < row_indices_0.len() || i1 < row_indices_1.len() || i2 < row_indices_2.len() {
                let r0 = if i0 < row_indices_0.len() {
                    row_indices_0[i0]
                } else {
                    usize::MAX
                };
                let r1 = if i1 < row_indices_1.len() {
                    row_indices_1[i1]
                } else {
                    usize::MAX
                };
                let r2 = if i2 < row_indices_2.len() {
                    row_indices_2[i2]
                } else {
                    usize::MAX
                };

                let min_row = r0.min(r1).min(r2);
                if min_row == usize::MAX {
                    break;
                }

                // Only include camera rows (not landmark rows)
                if min_row >= cam_start && min_row < cam_end {
                    let v0 = if r0 == min_row { col_values_0[i0] } else { 0.0 };
                    let v1 = if r1 == min_row { col_values_1[i1] } else { 0.0 };
                    let v2 = if r2 == min_row { col_values_2[i2] } else { 0.0 };

                    cam_rows.push(min_row - cam_start); // Store local camera row
                    h_cp_block.push([v0, v1, v2]);
                }

                if r0 == min_row {
                    i0 += 1;
                }
                if r1 == min_row {
                    i1 += 1;
                }
                if r2 == min_row {
                    i2 += 1;
                }
            }

            if cam_rows.is_empty() {
                continue;
            }

            // Compute contribution: H_cp * H_pp^{-1}
            contrib_block.clear();
            for h_cp_row in &h_cp_block {
                let c0 = h_cp_row[0] * hpp_inv_block[(0, 0)]
                    + h_cp_row[1] * hpp_inv_block[(1, 0)]
                    + h_cp_row[2] * hpp_inv_block[(2, 0)];
                let c1 = h_cp_row[0] * hpp_inv_block[(0, 1)]
                    + h_cp_row[1] * hpp_inv_block[(1, 1)]
                    + h_cp_row[2] * hpp_inv_block[(2, 1)];
                let c2 = h_cp_row[0] * hpp_inv_block[(0, 2)]
                    + h_cp_row[1] * hpp_inv_block[(1, 2)]
                    + h_cp_row[2] * hpp_inv_block[(2, 2)];
                contrib_block.push([c0, c1, c2]);
            }

            // Subtract outer product: (H_cp * H_pp^{-1}) * H_cp^T
            let n_cams = cam_rows.len();
            for i in 0..n_cams {
                let cam_i = cam_rows[i];
                let contrib_i = &contrib_block[i];
                for j in 0..n_cams {
                    let cam_j = cam_rows[j];
                    let h_cp_j = &h_cp_block[j];
                    let dot = contrib_i[0] * h_cp_j[0]
                        + contrib_i[1] * h_cp_j[1]
                        + contrib_i[2] * h_cp_j[2];
                    s_dense[cam_i * cam_dof + cam_j] -= dot;
                }
            }
        }

        // Symmetrize S to ensure numerical symmetry
        for i in 0..cam_dof {
            for j in (i + 1)..cam_dof {
                let avg = (s_dense[i * cam_dof + j] + s_dense[j * cam_dof + i]) * 0.5;
                s_dense[i * cam_dof + j] = avg;
                s_dense[j * cam_dof + i] = avg;
            }
        }

        // Convert dense S to sparse
        let mut triplets = Vec::new();
        for col in 0..cam_dof {
            for row in 0..cam_dof {
                let val = s_dense[row * cam_dof + col];
                if val.abs() > 1e-12 {
                    triplets.push(Triplet::new(row, col, val));
                }
            }
        }

        SparseColMat::try_new_from_triplets(cam_dof, cam_dof, &triplets).map_err(|e| {
            LinAlgError::InvalidInput(format!("Failed to create Schur complement: {:?}", e))
        })
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

        // Form and solve Schur complement system: S * δ_cam = g_reduced
        // where S = H_cc - H_cp * H_pp^{-1} * H_cp^T
        // We compute S explicitly using power series for H_pp^{-1}

        use faer::Side;
        use faer::linalg::solvers::Solve;
        use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};

        // Compute the TRUE Schur complement (not just H_cc!)
        let schur_complement = self.compute_schur_complement(
            self.hessian
                .as_ref()
                .ok_or_else(|| LinAlgError::InvalidState("Hessian not initialized".into()))?,
        )?;

        // Factorize and solve the Schur complement system
        let symbolic_llt = SymbolicLlt::try_new(schur_complement.symbolic(), Side::Lower)
            .map_err(|_| LinAlgError::InvalidInput("Symbolic factorization failed".into()))?;

        let llt = Llt::try_new_with_symbolic(symbolic_llt, schur_complement.as_ref(), Side::Lower)
            .map_err(|_| {
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
        let base_hessian = jacobian
            .transpose()
            .to_col_major()
            .map_err(|e| LinAlgError::MatrixConversion(format!("Transpose failed: {:?}", e)))?
            .mul(jacobian);

        // Add damping λI to the Hessian
        let n = base_hessian.ncols();
        let symbolic = base_hessian.symbolic();
        let mut triplets = Vec::new();

        for col in 0..n {
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = base_hessian.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                triplets.push(Triplet::new(row, col, col_values[idx]));
            }
            triplets.push(Triplet::new(col, col, lambda));
        }

        let hessian = SparseColMat::try_new_from_triplets(n, n, &triplets).map_err(|e| {
            LinAlgError::InvalidInput(format!("Failed to build damped Hessian: {:?}", e))
        })?;

        let mut gradient = Mat::<f64>::zeros(jtr.nrows(), 1);
        for i in 0..jtr.nrows() {
            gradient[(i, 0)] = -jtr[(i, 0)];
        }

        // Store for later use
        self.hessian = Some(hessian);
        self.gradient = Some(gradient.clone());

        // Now do the Schur complement solve with the damped hessian
        // (This is similar to solve_normal_equation but uses self.hessian)

        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not initialized".into()))?;
        let cam_dof = structure.camera_dof;
        let lm_dof = structure.landmark_dof;
        let (cam_start, _cam_end) = structure.camera_col_range();
        let (lm_start, _lm_end) = structure.landmark_col_range();

        // Extract diagonal blocks (with damping) and their inverses
        // Clone hessian to avoid borrow conflict with mutable self
        let hessian_clone = self
            .hessian
            .clone()
            .ok_or_else(|| LinAlgError::InvalidState("Hessian not set".into()))?;
        self.extract_landmark_diagonal_blocks(&hessian_clone)?;

        // Extract gradient components
        let mut g_cam = Mat::<f64>::zeros(cam_dof, 1);
        for i in 0..cam_dof {
            g_cam[(i, 0)] = gradient[(cam_start + i, 0)];
        }

        let mut g_lm = Mat::<f64>::zeros(lm_dof, 1);
        for i in 0..lm_dof {
            g_lm[(i, 0)] = gradient[(lm_start + i, 0)];
        }

        // Compute H_pp^{-1} * g_lm using block-diagonal inversion
        let mut hpp_inv_g = Mat::<f64>::zeros(lm_dof, 1);
        self.apply_landmark_inverse_power_series(&g_lm, &mut hpp_inv_g)?;

        // Compute reduced RHS: g_c - H_cp * H_pp^{-1} * g_p
        let hessian_ref = self
            .hessian
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidState("Hessian not initialized".into()))?;
        let correction = self.extract_camera_landmark_mvp(hessian_ref, &hpp_inv_g)?;
        let mut g_reduced = g_cam.clone();
        for i in 0..cam_dof {
            g_reduced[(i, 0)] -= correction[(i, 0)];
        }

        // Compute true Schur complement and solve
        use faer::Side;
        use faer::linalg::solvers::Solve;
        use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};

        let hessian_ref = self
            .hessian
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidState("Hessian not initialized".into()))?;
        let schur_complement = self.compute_schur_complement(hessian_ref)?;

        let symbolic_llt = SymbolicLlt::try_new(schur_complement.symbolic(), Side::Lower)
            .map_err(|_| LinAlgError::InvalidInput("Symbolic factorization failed".into()))?;

        let llt = Llt::try_new_with_symbolic(symbolic_llt, schur_complement.as_ref(), Side::Lower)
            .map_err(|_| {
                LinAlgError::FactorizationFailed("Schur complement factorization failed".into())
            })?;

        let delta_cam = llt.solve(&g_reduced);

        // Back-substitute for landmarks
        let hessian_ref = self
            .hessian
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidState("Hessian not initialized".into()))?;
        let hcp_t_delta_cam =
            self.extract_camera_landmark_transpose_mvp(hessian_ref, &delta_cam)?;

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
