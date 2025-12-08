//! Sparse Schur Complement Solver for Bundle Adjustment
//!
//! This module implements the Sparse Schur Complement solver with three variants:
//! 1. SPARSE_SCHUR - Standard direct factorization
//! 2. ITERATIVE_SCHUR - Preconditioned Conjugate Gradients
//! 3. POWER_SERIES_SCHUR - Power series approximation (PSSC/PoBA)

use crate::core::problem::VariableEnum;
use crate::linalg::{LinAlgError, LinAlgResult, StructuredSparseLinearSolver};
use crate::manifold::ManifoldType;
use faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};
use faer::{linalg::solvers::Solve, sparse::linalg::solvers::{Llt, SymbolicLlt}, Mat, Side};
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
            let manifold_type = variable.manifold_type();
            let start_col = *variable_index_map.get(name).ok_or_else(|| {
                LinAlgError::InvalidInput(format!("Variable {} not found in index map", name))
            })?;
            let size = variable.get_size();

            if self.ordering.should_eliminate(&manifold_type) {
                structure.landmark_blocks.push((name.clone(), start_col, size));
                structure.landmark_dof += size;
                
                if size != 3 {
                    return Err(LinAlgError::InvalidInput(format!(
                        "Landmark variable {} has DOF {}, expected 3",
                        name, size
                    )));
                }
                structure.num_landmarks += 1;
            } else {
                structure.camera_blocks.push((name.clone(), start_col, size));
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

        self.block_structure = Some(structure);
        Ok(())
    }

    /// Extract 3×3 diagonal blocks from H_pp
    fn extract_landmark_blocks(
        &self,
        hessian: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Vec<Matrix3<f64>>> {
        let structure = self.block_structure.as_ref().ok_or_else(|| {
            LinAlgError::InvalidInput("Block structure not built".to_string())
        })?;

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
        let structure = self.block_structure.as_ref().ok_or_else(|| {
            LinAlgError::InvalidInput("Block structure not built".to_string())
        })?;

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
        let structure = self.block_structure.as_ref().ok_or_else(|| {
            LinAlgError::InvalidInput("Block structure not built".to_string())
        })?;

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
    fn extract_gradient_blocks(
        &self,
        gradient: &Mat<f64>,
    ) -> LinAlgResult<(Mat<f64>, Mat<f64>)> {
        let structure = self.block_structure.as_ref().ok_or_else(|| {
            LinAlgError::InvalidInput("Block structure not built".to_string())
        })?;

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
        _residuals: &Mat<f64>,
        _jacobians: &SparseColMat<usize, f64>,
    ) -> LinAlgResult<Mat<f64>> {
        Err(LinAlgError::InvalidInput(
            "Phase 1 implementation in progress".to_string(),
        ))
    }

    fn solve_augmented_equation(
        &mut self,
        _residuals: &Mat<f64>,
        _jacobians: &SparseColMat<usize, f64>,
        _lambda: f64,
    ) -> LinAlgResult<Mat<f64>> {
        Err(LinAlgError::InvalidInput(
            "Phase 1 implementation in progress".to_string(),
        ))
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
}
