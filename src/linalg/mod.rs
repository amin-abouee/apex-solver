//! Linear algebra utilities and optimizations.
//!
//! This module provides efficient linear algebra operations specifically
//! optimized for optimization problems:
//! - Sparse matrix operations using faer
//! - Dense matrix operations using nalgebra
//! - Linear system solvers (direct and iterative)
//! - Matrix decompositions (Cholesky, QR, SVD)
//! - Unified trait interface for sparse operations

use crate::core::{ApexError, ApexResult};

// Re-export submodules
pub mod cholesky;
pub mod qr;

// Re-export main types and traits
pub use cholesky::*;
pub use qr::*;

/// Type alias for sparse matrices using faer
pub type SparseMatrix = faer::sparse::SparseColMat<usize, f64>;

/// Type alias for faer matrices (used for vectors)
pub type FaerMatrix = faer::Mat<f64>;

// ================================================================================================
// TRAITS (merged from traits.rs)
// ================================================================================================

/// Common interface for linear system solvers
///
/// This trait provides a unified interface for solving linear systems
/// regardless of whether the underlying implementation uses sparse or dense
/// matrix operations.
pub trait LinearSolver: Send + Sync {
    /// The matrix type this solver operates on
    type Matrix;

    /// The vector type this solver operates on
    type Vector;

    /// Create a new solver instance
    fn new() -> Self;

    /// Analyze the sparsity pattern and symbolic factorization (if applicable)
    /// For dense solvers, this may be a no-op
    fn analyze(&mut self, matrix: &Self::Matrix) -> ApexResult<()>;

    /// Perform numerical factorization of the matrix
    fn factorize(&mut self, matrix: &Self::Matrix) -> ApexResult<()>;

    /// Solve the linear system Ax = b
    fn solve(&mut self, rhs: &Self::Vector) -> ApexResult<Self::Vector>;

    /// Combined analyze and factorize step for convenience
    fn analyze_and_factorize(&mut self, matrix: &Self::Matrix) -> ApexResult<()> {
        self.analyze(matrix)?;
        self.factorize(matrix)
    }

    /// Check if the solver has been factorized and is ready to solve
    fn is_factorized(&self) -> bool;

    /// Get information about the solver's capabilities
    fn solver_info(&self) -> SolverInfo;
}

/// Extended interface for least squares problems
///
/// This trait extends the basic LinearSolver interface to handle
/// overdetermined systems and least squares problems, commonly
/// encountered in optimization.
pub trait LeastSquaresSolver: LinearSolver {
    /// Solve the least squares problem min ||Ax - b||^2
    fn solve_least_squares(
        &mut self,
        matrix: &Self::Matrix,
        rhs: &Self::Vector,
    ) -> ApexResult<Self::Vector>;

    /// Solve using normal equations: A^T A x = A^T b
    fn solve_normal_equations(
        &mut self,
        matrix: &Self::Matrix,
        rhs: &Self::Vector,
    ) -> ApexResult<Self::Vector>;
}

/// Extended interface for problems requiring Jacobian-based solving
///
/// This trait is specifically designed for optimization problems where
/// we need to solve systems of the form J^T W J x = -J^T W r (weighted least squares)
pub trait OptimizationSolver: LinearSolver {
    /// Solve the system J^T J x = -J^T residuals
    /// This is the core operation in Gauss-Newton methods
    fn solve_jtj(
        &mut self,
        jacobian: &Self::Matrix,
        residuals: &Self::Vector,
    ) -> ApexResult<Self::Vector>;

    /// Solve the weighted system J^T W J x = -J^T W residuals
    /// This is the core operation for robust optimization with M-estimators
    fn solve_jtwj(
        &mut self,
        jacobian: &Self::Matrix,
        weights: &Self::Vector,
        residuals: &Self::Vector,
    ) -> ApexResult<Self::Vector>;

    /// Solve with damping for Levenberg-Marquardt: (J^T J + λI) x = -J^T r
    fn solve_jtj_damped(
        &mut self,
        jacobian: &Self::Matrix,
        residuals: &Self::Vector,
        damping: f64,
    ) -> ApexResult<Self::Vector>;

    /// Solve with weighted damping: (J^T W J + λI) x = -J^T W r
    fn solve_jtwj_damped(
        &mut self,
        jacobian: &Self::Matrix,
        weights: &Self::Vector,
        residuals: &Self::Vector,
        damping: f64,
    ) -> ApexResult<Self::Vector>;
}

/// Information about solver capabilities and characteristics
#[derive(Debug, Clone)]
pub struct SolverInfo {
    /// Name of the solver
    pub name: String,
    /// Whether the solver supports sparse matrices
    pub supports_sparse: bool,
    /// Whether the solver supports dense matrices
    pub supports_dense: bool,
    /// Whether the solver can handle rank-deficient systems
    pub handles_rank_deficient: bool,
    /// Whether the solver preserves sparsity structure
    pub preserves_sparsity: bool,
    /// Computational complexity (rough estimate)
    pub complexity: ComplexityClass,
}

/// Computational complexity classes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityClass {
    /// O(n) - linear complexity
    Linear,
    /// O(n^2) - quadratic complexity
    Quadratic,
    /// O(n^3) - cubic complexity
    Cubic,
    /// Depends on sparsity pattern
    SparseDependent,
}

/// Factory trait for creating solvers
pub trait SolverFactory<S: LinearSolver> {
    /// Create a new solver instance with default settings
    fn create() -> S;

    /// Create a new solver with specific configuration
    fn create_with_config(config: SolverConfig) -> S;
}

/// Configuration for solver creation and behavior
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Numerical tolerance for pivoting/rank detection
    pub tolerance: f64,
    /// Whether to enable iterative refinement
    pub iterative_refinement: bool,
    /// Maximum number of refinement iterations
    pub max_refinement_iterations: usize,
    /// Whether to enable parallel execution
    pub parallel: bool,
    /// Memory management hints
    pub memory_hint: MemoryHint,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            iterative_refinement: false,
            max_refinement_iterations: 3,
            parallel: true,
            memory_hint: MemoryHint::Balanced,
        }
    }
}

/// Memory management hints for solvers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryHint {
    /// Optimize for minimal memory usage
    MinimalMemory,
    /// Balance between memory and performance
    Balanced,
    /// Optimize for maximum performance
    MaxPerformance,
}

/// Adapter traits for converting between different matrix/vector types
pub trait MatrixAdapter<From, To> {
    fn convert(from: &From) -> ApexResult<To>;
}

/// Sparse-specific traits for solvers that work with sparse matrices
pub trait SparseSolver: LinearSolver<Matrix = SparseMatrix, Vector = FaerMatrix> {
    /// Get the number of non-zeros in the factorization
    fn factorization_nnz(&self) -> Option<usize>;

    /// Get fill-in ratio (nnz_factor / nnz_original)
    fn fill_in_ratio(&self) -> Option<f64>;
}

// ================================================================================================
// SOLVER TYPE ENUMERATION AND UTILITIES
// ================================================================================================

/// Solver type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverType {
    SparseCholesky,
    SparseQR,
    SparseSpQR,
}

impl Default for SolverType {
    fn default() -> Self {
        SolverType::SparseCholesky
    }
}

/// Matrix type enumeration (only sparse supported)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixType {
    Sparse,
}

/// Solve error type for linear solvers
#[derive(Debug)]
pub enum SolveError {
    /// Factorization failed
    FactorizationFailed(String),
    /// Solve failed
    SolveFailed(String),
}

/// Utility functions for sparse matrix operations
pub mod conversions {
    use super::*;

    /// Create sparse matrix from triplets using faer
    pub fn triplets_to_sparse(
        rows: usize,
        cols: usize,
        row_indices: &[usize],
        col_indices: &[usize],
        values: &[f64],
    ) -> ApexResult<SparseMatrix> {
        // Create triplets in the format faer expects
        let triplets: Vec<_> = row_indices
            .iter()
            .zip(col_indices.iter())
            .zip(values.iter())
            .map(|((&row, &col), &val)| faer::sparse::Triplet::new(row, col, val))
            .collect();

        // Build the sparse matrix
        faer::sparse::SparseColMat::try_new_from_triplets(rows, cols, &triplets).map_err(|e| {
            ApexError::LinearAlgebra(format!("Failed to create sparse matrix: {:?}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_creation() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let vals = vec![1.0, 2.0, 3.0];

        let sparse = conversions::triplets_to_sparse(3, 3, &rows, &cols, &vals);
        assert!(sparse.is_ok());
    }

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert_eq!(config.tolerance, 1e-12);
        assert!(!config.iterative_refinement);
        assert_eq!(config.max_refinement_iterations, 3);
        assert!(config.parallel);
        assert_eq!(config.memory_hint, MemoryHint::Balanced);
    }

    #[test]
    fn test_complexity_class() {
        let complexity = ComplexityClass::Cubic;
        assert_eq!(complexity, ComplexityClass::Cubic);
        assert_ne!(complexity, ComplexityClass::Linear);
    }

    #[test]
    fn test_memory_hint() {
        let hint = MemoryHint::MaxPerformance;
        assert_eq!(hint, MemoryHint::MaxPerformance);
        assert_ne!(hint, MemoryHint::MinimalMemory);
    }
}
