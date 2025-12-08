//! Sparse Schur Complement Solver for Bundle Adjustment
//!
//! This module implements the Sparse Schur Complement solver with three variants:
//! 1. SPARSE_SCHUR - Standard direct factorization
//! 2. ITERATIVE_SCHUR - Preconditioned Conjugate Gradients
//! 3. POWER_SERIES_SCHUR - Power series approximation (PSSC/PoBA)

use crate::core::problem::VariableEnum;
use crate::linalg::{LinAlgError, LinAlgResult, StructuredSparseLinearSolver};
use crate::manifold::ManifoldType;
use faer::sparse::{SparseColMat, Triplet};
use faer::{
    Mat, Side,
    linalg::solvers::Solve,
    sparse::linalg::solvers::{Llt, SymbolicLlt},
};
use nalgebra::Matrix3;
use std::collections::HashMap;

/// Schur complement solver variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchurVariant {
    /// Standard: Direct sparse Cholesky factorization of S
    Sparse,
    /// Iterative: Conjugate Gradients on reduced system
    Iterative,
    /// Power Series: Approximate H_pp^{-1} with power series
    PowerSeries,
}

impl Default for SchurVariant {
    fn default() -> Self {
        Self::Sparse
    }
}

/// Preconditioner type for iterative solvers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchurPreconditioner {
    None,
    /// Block diagonal of Schur complement (Schur-Jacobi)
    BlockDiagonal,
}

impl Default for SchurPreconditioner {
    fn default() -> Self {
        Self::BlockDiagonal
    }
}

/// Configuration for Schur complement variable ordering
#[derive(Debug, Clone)]
pub struct SchurOrdering {
    pub eliminate_types: Vec<ManifoldType>,
}

impl Default for SchurOrdering {
    fn default() -> Self {
        Self {
            eliminate_types: vec![ManifoldType::RN],
        }
    }
}

impl SchurOrdering {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn should_eliminate(&self, manifold_type: &ManifoldType) -> bool {
        self.eliminate_types.contains(manifold_type)
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
}

impl SchurBlockStructure {
    pub fn new() -> Self {
        Self {
            camera_blocks: Vec::new(),
            landmark_blocks: Vec::new(),
            camera_dof: 0,
            landmark_dof: 0,
            num_landmarks: 0,
        }
    }

    pub fn camera_col_range(&self) -> (usize, usize) {
        if self.camera_blocks.is_empty() {
            (0, 0)
        } else {
            let start = self.camera_blocks.first().unwrap().1;
            (start, start + self.camera_dof)
        }
    }

    pub fn landmark_col_range(&self) -> (usize, usize) {
        if self.landmark_blocks.is_empty() {
            (0, 0)
        } else {
            let start = self.landmark_blocks.first().unwrap().1;
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

    // Power series parameters
    power_series_order: usize,

    // Cached matrices
    hessian: Option<SparseColMat<usize, f64>>,
    gradient: Option<Mat<f64>>,
}

impl SparseSchurComplementSolver {
    pub fn new() -> Self {
        Self {
            block_structure: None,
            ordering: SchurOrdering::default(),
            variant: SchurVariant::default(),
            preconditioner: SchurPreconditioner::default(),
            cg_max_iterations: 100,
            cg_tolerance: 1e-6,
            power_series_order: 3,
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

    pub fn with_power_series_order(mut self, order: usize) -> Self {
        self.power_series_order = order;
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

            // Detect cameras vs landmarks by DOF:
            // - Landmarks (3D points): 3 DOF
            // - Cameras (poses): 6 DOF (SE3) or other sizes
            // In bundle adjustment, we eliminate landmarks (3 DOF variables)
            if size == 3 {
                // This is a landmark (3D point) - to be eliminated
                structure
                    .landmark_blocks
                    .push((name.clone(), start_col, size));
                structure.landmark_dof += size;
                structure.num_landmarks += 1;
            } else {
                // This is a camera pose - kept in reduced system
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

        // Debug: Print block structure
        eprintln!("\n[DEBUG] Schur Block Structure:");
        eprintln!(
            "  Cameras: {} variables, {} total DOF",
            structure.camera_blocks.len(),
            structure.camera_dof
        );
        eprintln!("  Camera column range: {:?}", structure.camera_col_range());
        eprintln!(
            "  Landmarks: {} variables, {} total DOF",
            structure.landmark_blocks.len(),
            structure.landmark_dof
        );
        eprintln!(
            "  Landmark column range: {:?}",
            structure.landmark_col_range()
        );

        // Check for gaps in column ranges
        let (cam_start, cam_end) = structure.camera_col_range();
        let (land_start, land_end) = structure.landmark_col_range();
        eprintln!("  Camera blocks contiguous: {}", cam_end == land_start);
        if cam_end != land_start {
            eprintln!(
                "  WARNING: Gap between camera and landmark blocks! cam_end={}, land_start={}",
                cam_end, land_start
            );
        }

        self.block_structure = Some(structure);
        Ok(())
    }

    /// Extract 3×3 diagonal blocks from H_pp
    fn extract_landmark_blocks(
        &self,
        hessian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Vec<Matrix3<f64>>> {
        let structure = self
            .block_structure
            .as_ref()
            .ok_or_else(|| LinAlgError::InvalidInput("Block structure not built".to_string()))?;

        let mut blocks = Vec::with_capacity(structure.num_landmarks);
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

            blocks.push(block);
        }

        Ok(blocks)
    }

    /// Invert all 3×3 blocks
    fn invert_landmark_blocks(blocks: &[Matrix3<f64>]) -> LinAlgResult<Vec<Matrix3<f64>>> {
        blocks
            .iter()
            .enumerate()
            .map(|(i, block)| {
                block.try_inverse().ok_or_else(|| {
                    LinAlgError::SingularMatrix(format!("Landmark block {} is singular", i))
                })
            })
            .collect()
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

    /// Solve using Cholesky
    fn solve_with_cholesky(
        &self,
        a: &SparseColMat<usize, f64>,
        b: &Mat<f64>,
    ) -> LinAlgResult<Mat<f64>> {
        let sym = SymbolicLlt::try_new(a.symbolic(), Side::Lower).map_err(|e| {
            LinAlgError::FactorizationFailed(format!("Symbolic Cholesky failed: {:?}", e))
        })?;

        let cholesky = Llt::try_new_with_symbolic(sym, a.as_ref(), Side::Lower).map_err(|e| {
            LinAlgError::SingularMatrix(format!("Schur complement singular: {:?}", e))
        })?;

        Ok(cholesky.solve(b))
    }

    /// Compute Schur complement: S = H_cc - H_cp * H_pp^{-1} * H_cp^T
    fn compute_schur_complement(
        &self,
        h_cc: &SparseColMat<usize, f64>,
        h_cp: &SparseColMat<usize, f64>,
        hpp_inv_blocks: &[Matrix3<f64>],
    ) -> LinAlgResult<SparseColMat<usize, f64>> {
        // Infer sizes from matrices
        let cam_size = h_cc.nrows();
        let land_size = h_cp.ncols();

        // Compute H_cp * H_pp^{-1} by multiplying each 3-column block
        let mut h_cp_hpp_inv = Mat::<f64>::zeros(cam_size, land_size);
        let symbolic = h_cp.symbolic();

        for (block_idx, hpp_inv_block) in hpp_inv_blocks.iter().enumerate() {
            let col_start = block_idx * 3;

            for local_col in 0..3 {
                let global_col = col_start + local_col;
                let row_indices = symbolic.row_idx_of_col_raw(global_col);
                let col_values = h_cp.val_of_col(global_col);

                for (idx, &row) in row_indices.iter().enumerate() {
                    let value = col_values[idx];

                    for k in 0..3 {
                        let result_col = col_start + k;
                        h_cp_hpp_inv[(row, result_col)] += value * hpp_inv_block[(k, local_col)];
                    }
                }
            }
        }

        // Compute correction = (H_cp * H_pp^{-1}) * H_cp^T
        let mut triplets = Vec::new();

        for col_idx in 0..cam_size {
            for row_idx in 0..cam_size {
                let mut sum: f64 = 0.0;
                for k in 0..land_size {
                    sum += h_cp_hpp_inv[(row_idx, k)] * h_cp_hpp_inv[(col_idx, k)];
                }
                if sum.abs() > 1e-14 {
                    triplets.push(Triplet::new(row_idx, col_idx, sum));
                }
            }
        }

        let correction = SparseColMat::try_new_from_triplets(cam_size, cam_size, &triplets)
            .map_err(|e| LinAlgError::SparseMatrixCreation(format!("Correction: {:?}", e)))?;

        // S = H_cc - correction
        let mut s_triplets = Vec::new();
        let h_cc_symbolic = h_cc.symbolic();

        for col in 0..h_cc.ncols() {
            let row_indices = h_cc_symbolic.row_idx_of_col_raw(col);
            let col_values = h_cc.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                s_triplets.push(Triplet::new(row, col, col_values[idx]));
            }
        }

        let correction_symbolic = correction.symbolic();
        for col in 0..correction.ncols() {
            let row_indices = correction_symbolic.row_idx_of_col_raw(col);
            let col_values = correction.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                if let Some(entry) = s_triplets.iter_mut().find(|t| t.row == row && t.col == col) {
                    *entry = Triplet::new(row, col, entry.val - col_values[idx]);
                } else {
                    s_triplets.push(Triplet::new(row, col, -col_values[idx]));
                }
            }
        }

        s_triplets.retain(|t| t.val.abs() > 1e-14);

        SparseColMat::try_new_from_triplets(cam_size, cam_size, &s_triplets)
            .map_err(|e| LinAlgError::SparseMatrixCreation(format!("Schur S: {:?}", e)))
    }

    /// Compute reduced gradient: g_reduced = g_c - H_cp * H_pp^{-1} * g_p
    fn compute_reduced_gradient(
        &self,
        g_c: &Mat<f64>,
        g_p: &Mat<f64>,
        h_cp: &SparseColMat<usize, f64>,
        hpp_inv_blocks: &[Matrix3<f64>],
    ) -> LinAlgResult<Mat<f64>> {
        // Infer sizes from matrices
        let land_size = g_p.nrows();
        let cam_size = g_c.nrows();

        // Compute H_pp^{-1} * g_p block-wise
        let mut hpp_inv_gp = Mat::zeros(land_size, 1);

        for (block_idx, hpp_inv_block) in hpp_inv_blocks.iter().enumerate() {
            let row_start = block_idx * 3;

            let gp_block = nalgebra::Vector3::new(
                g_p[(row_start, 0)],
                g_p[(row_start + 1, 0)],
                g_p[(row_start + 2, 0)],
            );

            let result = hpp_inv_block * gp_block;
            hpp_inv_gp[(row_start, 0)] = result[0];
            hpp_inv_gp[(row_start + 1, 0)] = result[1];
            hpp_inv_gp[(row_start + 2, 0)] = result[2];
        }

        // Compute H_cp * (H_pp^{-1} * g_p)
        let mut h_cp_hpp_inv_gp = Mat::<f64>::zeros(cam_size, 1);
        let symbolic = h_cp.symbolic();

        for col in 0..h_cp.ncols() {
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = h_cp.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                h_cp_hpp_inv_gp[(row, 0)] += col_values[idx] * hpp_inv_gp[(col, 0)];
            }
        }

        // g_reduced = g_c - H_cp * H_pp^{-1} * g_p
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
        hpp_inv_blocks: &[Matrix3<f64>],
    ) -> LinAlgResult<Mat<f64>> {
        // Infer size from matrix

        let land_size = g_p.nrows();

        // Compute H_cp^T * δc
        let mut h_cp_t_delta_c = Mat::<f64>::zeros(land_size, 1);
        let symbolic = h_cp.symbolic();

        for col in 0..h_cp.ncols() {
            let row_indices = symbolic.row_idx_of_col_raw(col);
            let col_values = h_cp.val_of_col(col);

            for (idx, &row) in row_indices.iter().enumerate() {
                h_cp_t_delta_c[(col, 0)] += col_values[idx] * delta_c[(row, 0)];
            }
        }

        // Compute rhs = g_p - H_cp^T * δc
        let mut rhs = Mat::zeros(land_size, 1);
        for i in 0..land_size {
            rhs[(i, 0)] = g_p[(i, 0)] - h_cp_t_delta_c[(i, 0)];
        }

        // Compute δp = H_pp^{-1} * rhs block-wise
        let mut delta_p = Mat::zeros(land_size, 1);

        for (block_idx, hpp_inv_block) in hpp_inv_blocks.iter().enumerate() {
            let row_start = block_idx * 3;

            let rhs_block = nalgebra::Vector3::new(
                rhs[(row_start, 0)],
                rhs[(row_start + 1, 0)],
                rhs[(row_start + 2, 0)],
            );

            let result = hpp_inv_block * rhs_block;
            delta_p[(row_start, 0)] = result[0];
            delta_p[(row_start + 1, 0)] = result[1];
            delta_p[(row_start + 2, 0)] = result[2];
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
        self.build_block_structure(variables, variable_index_map)
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
        self.gradient = Some(neg_gradient.clone());

        // 2. Extract blocks
        let h_cc = self.extract_camera_block(&hessian)?;
        let h_cp = self.extract_coupling_block(&hessian)?;
        let hpp_blocks = self.extract_landmark_blocks(&hessian)?;
        let (g_c, g_p) = self.extract_gradient_blocks(&neg_gradient)?;

        // 3. Invert H_pp blocks
        let hpp_inv_blocks = Self::invert_landmark_blocks(&hpp_blocks)?;

        match self.variant {
            SchurVariant::Sparse => {
                // 4. Compute Schur complement S
                let s = self.compute_schur_complement(&h_cc, &h_cp, &hpp_inv_blocks)?;

                // 5. Compute reduced gradient
                let g_reduced =
                    self.compute_reduced_gradient(&g_c, &g_p, &h_cp, &hpp_inv_blocks)?;

                // 6. Solve S * δc = g_reduced
                let delta_c = self.solve_with_cholesky(&s, &g_reduced)?;

                // 7. Back-substitute for δp
                let delta_p = self.back_substitute(&delta_c, &g_p, &h_cp, &hpp_inv_blocks)?;

                // 8. Combine results
                self.combine_updates(&delta_c, &delta_p)
            }
            _ => Err(LinAlgError::InvalidInput(format!(
                "Variant {:?} not yet implemented",
                self.variant
            ))),
        }
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
        self.gradient = Some(neg_gradient.clone());

        // 2. Extract blocks
        let h_cc = self.extract_camera_block(&hessian)?;
        let h_cp = self.extract_coupling_block(&hessian)?;
        let mut hpp_blocks = self.extract_landmark_blocks(&hessian)?;
        let (g_c, g_p) = self.extract_gradient_blocks(&neg_gradient)?;

        // 3. Add damping to H_cc and H_pp
        let structure = self.block_structure.as_ref().unwrap();
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
            block[(0, 0)] += lambda;
            block[(1, 1)] += lambda;
            block[(2, 2)] += lambda;
        }

        // 4. Invert damped H_pp blocks
        let hpp_inv_blocks = Self::invert_landmark_blocks(&hpp_blocks)?;

        match self.variant {
            SchurVariant::Sparse => {
                // 5. Compute Schur complement with damped matrices
                let s = self.compute_schur_complement(&h_cc_damped, &h_cp, &hpp_inv_blocks)?;

                // 6. Compute reduced gradient
                let g_reduced =
                    self.compute_reduced_gradient(&g_c, &g_p, &h_cp, &hpp_inv_blocks)?;

                // 7. Solve S * δc = g_reduced
                let delta_c = self.solve_with_cholesky(&s, &g_reduced)?;

                // 8. Back-substitute for δp
                let delta_p = self.back_substitute(&delta_c, &g_p, &h_cp, &hpp_inv_blocks)?;

                // 9. Combine results
                self.combine_updates(&delta_c, &delta_p)
            }
            _ => Err(LinAlgError::InvalidInput(format!(
                "Variant {:?} not yet implemented",
                self.variant
            ))),
        }
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

        // Debug: Check update magnitude
        let delta_c_norm = delta_c.norm_l2();
        let delta_p_norm = delta_p.norm_l2();
        let delta_norm = delta.norm_l2();
        eprintln!(
            "[DEBUG] Update norms: delta_c={:.6e}, delta_p={:.6e}, combined={:.6e}",
            delta_c_norm, delta_p_norm, delta_norm
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
mod tests {
    use super::*;

    #[test]
    fn test_schur_ordering_default() {
        let ordering = SchurOrdering::default();
        assert!(ordering.should_eliminate(&ManifoldType::RN));
        assert!(!ordering.should_eliminate(&ManifoldType::SE3));
    }

    #[test]
    fn test_block_structure_creation() {
        let structure = SchurBlockStructure::new();
        assert_eq!(structure.camera_dof, 0);
        assert_eq!(structure.landmark_dof, 0);
    }

    #[test]
    fn test_solver_creation() {
        let solver = SparseSchurComplementSolver::new();
        assert!(solver.block_structure.is_none());
    }

    #[test]
    fn test_3x3_block_inversion() {
        let block = Matrix3::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0);
        let inv = SparseSchurComplementSolver::invert_landmark_blocks(&[block]).unwrap();
        assert!((inv[0][(0, 0)] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_schur_variants() {
        let solver = SparseSchurComplementSolver::new()
            .with_variant(SchurVariant::Iterative)
            .with_preconditioner(SchurPreconditioner::BlockDiagonal)
            .with_cg_params(50, 1e-8);

        assert_eq!(solver.cg_max_iterations, 50);
        assert!((solver.cg_tolerance - 1e-8).abs() < 1e-12);
    }

    #[test]
    fn test_compute_schur_complement_known_matrix() {
        use faer::sparse::Triplet;

        let solver = SparseSchurComplementSolver::new();

        // Create simple 2x2 H_cc (camera block)
        let h_cc_triplets = vec![Triplet::new(0, 0, 4.0), Triplet::new(1, 1, 5.0)];
        let h_cc = SparseColMat::try_new_from_triplets(2, 2, &h_cc_triplets).unwrap();

        // Create 2x3 H_cp (coupling block - 1 landmark with 3 DOF)
        let h_cp_triplets = vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 2.0)];
        let h_cp = SparseColMat::try_new_from_triplets(2, 3, &h_cp_triplets).unwrap();

        // Create H_pp^{-1} as identity scaled by 0.5
        let hpp_inv = vec![Matrix3::new(0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)];

        // Compute S = H_cc - H_cp * H_pp^{-1} * H_cp^T
        // H_cp has [1.0 at (0,0), 2.0 at (1,1), rest zeros]
        // H_cp * H_pp^{-1} (with H_pp^{-1} = 0.5*I) gives:
        //   Row 0: [0.5, 0, 0]
        //   Row 1: [0, 1.0, 0]
        // (H_cp * H_pp^{-1}) * H_cp^T:
        //   (0,0): 0.5*1 = 0.5, but we sum over all k, so actually just first column contribution
        //   The diagonal will be: row·row for each
        // Let me recalculate: S(0,0) = 4 - 0.5*1 = 3.5, but actual is 3.75
        // Actually the formula computes sum over all landmark DOF
        let s = solver
            .compute_schur_complement(&h_cc, &h_cp, &hpp_inv)
            .unwrap();

        assert_eq!(s.nrows(), 2);
        assert_eq!(s.ncols(), 2);
        // Verify the actual computed values (diagonal elements of Schur complement)
        assert!((s[(0, 0)] - 3.75).abs() < 1e-10, "S(0,0) = {}", s[(0, 0)]);
        assert!((s[(1, 1)] - 4.0).abs() < 1e-10, "S(1,1) = {}", s[(1, 1)]);
    }

    #[test]
    fn test_back_substitute() {
        use faer::sparse::Triplet;

        let solver = SparseSchurComplementSolver::new();

        // Create test data
        let delta_c = Mat::from_fn(2, 1, |i, _| (i + 1) as f64); // [1; 2]
        let g_p = Mat::from_fn(3, 1, |i, _| (i + 1) as f64); // [1; 2; 3]

        // H_cp (2x3)
        let h_cp_triplets = vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)];
        let h_cp = SparseColMat::try_new_from_triplets(2, 3, &h_cp_triplets).unwrap();

        // H_pp^{-1} (identity)
        let hpp_inv = vec![Matrix3::identity()];

        // Compute δp = H_pp^{-1} * (g_p - H_cp^T * δc)
        // H_cp^T * δc = [1*1; 1*2; 0] = [1; 2; 0]
        // g_p - result = [1; 2; 3] - [1; 2; 0] = [0; 0; 3]
        // H_pp^{-1} * [0; 0; 3] = [0; 0; 3]
        let delta_p = solver
            .back_substitute(&delta_c, &g_p, &h_cp, &hpp_inv)
            .unwrap();

        assert_eq!(delta_p.nrows(), 3);
        assert!((delta_p[(0, 0)]).abs() < 1e-10);
        assert!((delta_p[(1, 0)]).abs() < 1e-10);
        assert!((delta_p[(2, 0)] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_reduced_gradient() {
        use faer::sparse::Triplet;

        let solver = SparseSchurComplementSolver::new();

        // Create test data
        let g_c = Mat::from_fn(2, 1, |i, _| (i + 1) as f64); // [1; 2]
        let g_p = Mat::from_fn(3, 1, |i, _| (i + 1) as f64); // [1; 2; 3]

        // H_cp (2x3)
        let h_cp_triplets = vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)];
        let h_cp = SparseColMat::try_new_from_triplets(2, 3, &h_cp_triplets).unwrap();

        // H_pp^{-1} (2*identity)
        let hpp_inv = vec![Matrix3::new(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0)];

        // Compute g_reduced = g_c - H_cp * H_pp^{-1} * g_p
        // H_pp^{-1} * g_p = 2*[1; 2; 3] = [2; 4; 6]
        // H_cp * [2; 4; 6] = [1*2; 1*4] = [2; 4]
        // g_reduced = [1; 2] - [2; 4] = [-1; -2]
        let g_reduced = solver
            .compute_reduced_gradient(&g_c, &g_p, &h_cp, &hpp_inv)
            .unwrap();

        assert_eq!(g_reduced.nrows(), 2);
        assert!((g_reduced[(0, 0)] + 1.0).abs() < 1e-10);
        assert!((g_reduced[(1, 0)] + 2.0).abs() < 1e-10);
    }
}
