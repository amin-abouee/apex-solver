//! Linear algebra utilities and optimizations.
//!
//! This module provides efficient linear algebra operations specifically
//! optimized for optimization problems:
//! - Sparse matrix operations
//! - Linear system solvers (direct and iterative)
//! - Matrix decompositions (Cholesky, QR, SVD)
//! - Specialized routines for optimization

use crate::core::ApexResult;
use nalgebra::{DMatrix, DVector};

#[cfg(feature = "sparse")]
use sprs::{CsMat, CsVec};

/// Type alias for dense matrices
pub type DenseMatrix = DMatrix<f64>;

/// Type alias for dense vectors  
pub type Vector = DVector<f64>;

/// Sparse matrix wrapper with optimization-specific operations
#[cfg(feature = "sparse")]
pub struct SparseMatrix {
    /// The underlying sparse matrix
    pub matrix: CsMat<f64>,
}

#[cfg(feature = "sparse")]
impl SparseMatrix {
    /// Create a new sparse matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            matrix: CsMat::zero((rows, cols)),
        }
    }

    /// Create from triplet format (row indices, col indices, values)
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        row_indices: &[usize],
        col_indices: &[usize],
        values: &[f64],
    ) -> ApexResult<Self> {
        let matrix = CsMat::new_from_triplets((rows, cols), row_indices, col_indices, values);
        Ok(Self { matrix })
    }

    /// Convert to dense matrix
    pub fn to_dense(&self) -> DenseMatrix {
        let (rows, cols) = self.matrix.shape();
        let mut dense = DenseMatrix::zeros(rows, cols);

        for (value, (row, col)) in self.matrix.iter() {
            dense[(row, col)] = *value;
        }

        dense
    }

    /// Matrix-vector multiplication
    pub fn multiply_vector(&self, vector: &Vector) -> Vector {
        let result_data: Vec<f64> = &self.matrix * vector.as_slice();
        Vector::from_vec(result_data)
    }

    /// Transpose
    pub fn transpose(&self) -> Self {
        Self {
            matrix: self.matrix.transpose_view().to_owned(),
        }
    }

    /// Get number of non-zeros
    pub fn nnz(&self) -> usize {
        self.matrix.nnz()
    }
}

#[cfg(not(feature = "sparse"))]
/// Fallback sparse matrix implementation using dense matrices
pub struct SparseMatrix {
    /// Dense matrix used as fallback
    pub matrix: DenseMatrix,
}

#[cfg(not(feature = "sparse"))]
impl SparseMatrix {
    /// Create a new sparse matrix (fallback to dense)
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            matrix: DenseMatrix::zeros(rows, cols),
        }
    }

    /// Create from triplet format
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        row_indices: &[usize],
        col_indices: &[usize],
        values: &[f64],
    ) -> ApexResult<Self> {
        let mut matrix = DenseMatrix::zeros(rows, cols);

        for ((&row, &col), &value) in row_indices.iter().zip(col_indices).zip(values) {
            matrix[(row, col)] = value;
        }

        Ok(Self { matrix })
    }

    /// Convert to dense matrix
    pub fn to_dense(&self) -> DenseMatrix {
        self.matrix.clone()
    }

    /// Matrix-vector multiplication
    pub fn multiply_vector(&self, vector: &Vector) -> Vector {
        &self.matrix * vector
    }

    /// Transpose
    pub fn transpose(&self) -> Self {
        Self {
            matrix: self.matrix.transpose(),
        }
    }

    /// Get number of non-zeros (approximate for dense)
    pub fn nnz(&self) -> usize {
        self.matrix.iter().filter(|&&x| x.abs() > 1e-12).count()
    }
}

/// Sparse Cholesky decomposition solver
pub struct SparseCholesky {
    /// Whether the decomposition is computed
    computed: bool,
    /// The original matrix (for recomputation)
    matrix: Option<SparseMatrix>,
}

impl SparseCholesky {
    /// Create a new Cholesky solver
    pub fn new() -> Self {
        Self {
            computed: false,
            matrix: None,
        }
    }

    /// Analyze and factorize the matrix
    pub fn analyze_and_factorize(&mut self, matrix: SparseMatrix) -> ApexResult<()> {
        // For now, store the matrix. In a real implementation,
        // this would perform the actual Cholesky decomposition
        self.matrix = Some(matrix);
        self.computed = true;
        Ok(())
    }

    /// Solve the linear system Ax = b
    pub fn solve(&self, rhs: &Vector) -> ApexResult<Vector> {
        if !self.computed {
            return Err(crate::core::ApexError::LinearAlgebra(
                "Cholesky decomposition not computed".to_string(),
            ));
        }

        // Fallback implementation using dense solver
        if let Some(matrix) = &self.matrix {
            let dense = matrix.to_dense();
            let chol = dense.cholesky().ok_or_else(|| {
                crate::core::ApexError::LinearAlgebra("Cholesky decomposition failed".to_string())
            })?;
            Ok(chol.solve(rhs))
        } else {
            Err(crate::core::ApexError::LinearAlgebra(
                "No matrix to solve".to_string(),
            ))
        }
    }

    /// Check if decomposition is computed
    pub fn is_computed(&self) -> bool {
        self.computed
    }
}

impl Default for SparseCholesky {
    fn default() -> Self {
        Self::new()
    }
}

/// QR decomposition utilities
pub struct QRSolver;

impl QRSolver {
    /// Solve linear system using QR decomposition
    pub fn solve(matrix: &DenseMatrix, rhs: &Vector) -> ApexResult<Vector> {
        let qr = (*matrix).clone().qr();
        match qr.solve(rhs) {
            Some(solution) => Ok(solution),
            None => Err(crate::core::ApexError::LinearAlgebra(
                "QR solve failed - matrix may be singular".to_string(),
            )),
        }
    }

    /// Solve least squares problem using QR decomposition
    pub fn solve_least_squares(matrix: &DenseMatrix, rhs: &Vector) -> ApexResult<Vector> {
        // For overdetermined systems, use normal equations approach
        let ata = matrix.transpose() * matrix;
        let atb = matrix.transpose() * rhs;

        let chol = ata.cholesky().ok_or_else(|| {
            crate::core::ApexError::LinearAlgebra("Normal equations Cholesky failed".to_string())
        })?;

        Ok(chol.solve(&atb))
    }
}

/// SVD-based solver for rank-deficient systems
pub struct SVDSolver;

impl SVDSolver {
    /// Solve using SVD with optional rank truncation
    pub fn solve(matrix: &DenseMatrix, rhs: &Vector, tolerance: Option<f64>) -> ApexResult<Vector> {
        let svd = (*matrix).clone().svd(true, true);
        let u = svd.u.ok_or_else(|| {
            crate::core::ApexError::LinearAlgebra("SVD U matrix not computed".to_string())
        })?;
        let v_t = svd.v_t.ok_or_else(|| {
            crate::core::ApexError::LinearAlgebra("SVD V^T matrix not computed".to_string())
        })?;

        let tol = tolerance.unwrap_or(1e-12);
        let mut inv_s = Vector::zeros(svd.singular_values.len());

        for (i, &s) in svd.singular_values.iter().enumerate() {
            if s > tol {
                inv_s[i] = 1.0 / s;
            }
        }

        // Compute pseudo-inverse solution: x = V * diag(1/s) * U^T * b
        let ut_b = u.transpose() * rhs;
        let sinv_ut_b = inv_s.component_mul(&ut_b);
        let solution = v_t.transpose() * sinv_ut_b;

        Ok(solution)
    }
}
