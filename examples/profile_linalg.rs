//! Micro-profiling example for linear algebra operations
//!
//! This example profiles sparse linear algebra operations used in optimization:
//! - Sparse matrix multiplication (J^T * J)
//! - Sparse-dense multiplication (J^T * r)
//! - Cholesky factorization
//! - QR factorization
//! - Linear system solve
//!
//! Usage:
//! ```bash
//! cargo build --profile profiling --example profile_linalg
//! samply record ./target/profiling/examples/profile_linalg
//! ```

use apex_solver::linalg::{SparseCholeskySolver, SparseLinearSolver, SparseQRSolver};
use faer::{Mat, sparse::SparseColMat};
use std::ops::Mul;

const SMALL_SIZE: usize = 100;
const MEDIUM_SIZE: usize = 1000;
const LARGE_SIZE: usize = 10000;
const ITERATIONS_SMALL: usize = 1000;
const ITERATIONS_MEDIUM: usize = 100;
const ITERATIONS_LARGE: usize = 10;

fn main() {
    println!("=== Profiling Linear Algebra Operations ===");
    println!();

    profile_sparse_matrix_multiply();
    profile_sparse_dense_multiply();
    profile_cholesky_solver();
    profile_qr_solver();

    println!("\n=== Profiling Complete ===");
}

fn create_sparse_jacobian(rows: usize, cols: usize, density: f64) -> SparseColMat<usize, f64> {
    // Create a sparse Jacobian-like matrix with given density
    let mut triplets = Vec::new();
    let nnz_per_col = ((rows as f64) * density).max(1.0) as usize;

    for col in 0..cols {
        for i in 0..nnz_per_col.min(rows) {
            let row = (col * nnz_per_col + i) % rows;
            let value = 1.0 + (row as f64 * 0.01) + (col as f64 * 0.001);
            triplets.push(faer::sparse::Triplet::new(row, col, value));
        }
    }

    SparseColMat::try_new_from_triplets(rows, cols, &triplets).unwrap()
}

fn create_residual_vector(size: usize) -> Mat<f64> {
    Mat::from_fn(size, 1, |i, _| (i as f64 * 0.1).sin())
}

fn profile_sparse_matrix_multiply() {
    println!("Profiling Sparse Matrix Multiplication (J^T * J)...");

    // Small problem
    let jacobian_small = create_sparse_jacobian(600, SMALL_SIZE, 0.05);
    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_SMALL {
        let _ = jacobian_small
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(jacobian_small.as_ref());
    }
    let elapsed = start.elapsed() / ITERATIONS_SMALL as u32;
    println!(
        "  Size {}x{}    : {:?} per operation",
        jacobian_small.nrows(),
        jacobian_small.ncols(),
        elapsed
    );

    // Medium problem
    let jacobian_medium = create_sparse_jacobian(6000, MEDIUM_SIZE, 0.01);
    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_MEDIUM {
        let _ = jacobian_medium
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(jacobian_medium.as_ref());
    }
    let elapsed = start.elapsed() / ITERATIONS_MEDIUM as u32;
    println!(
        "  Size {}x{}   : {:?} per operation",
        jacobian_medium.nrows(),
        jacobian_medium.ncols(),
        elapsed
    );

    // Large problem
    let jacobian_large = create_sparse_jacobian(60000, LARGE_SIZE, 0.005);
    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_LARGE {
        let _ = jacobian_large
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(jacobian_large.as_ref());
    }
    let elapsed = start.elapsed() / ITERATIONS_LARGE as u32;
    println!(
        "  Size {}x{}  : {:?} per operation",
        jacobian_large.nrows(),
        jacobian_large.ncols(),
        elapsed
    );
}

fn profile_sparse_dense_multiply() {
    println!("\nProfiling Sparse-Dense Multiplication (J^T * r)...");

    // Small problem
    let jacobian_small = create_sparse_jacobian(600, SMALL_SIZE, 0.05);
    let residuals_small = create_residual_vector(600);
    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_SMALL {
        let _ = jacobian_small.as_ref().transpose().mul(&residuals_small);
    }
    let elapsed = start.elapsed() / ITERATIONS_SMALL as u32;
    println!(
        "  Size {}x{} * {}x1 : {:?} per operation",
        jacobian_small.nrows(),
        jacobian_small.ncols(),
        residuals_small.nrows(),
        elapsed
    );

    // Medium problem
    let jacobian_medium = create_sparse_jacobian(6000, MEDIUM_SIZE, 0.01);
    let residuals_medium = create_residual_vector(6000);
    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_MEDIUM {
        let _ = jacobian_medium.as_ref().transpose().mul(&residuals_medium);
    }
    let elapsed = start.elapsed() / ITERATIONS_MEDIUM as u32;
    println!(
        "  Size {}x{} * {}x1: {:?} per operation",
        jacobian_medium.nrows(),
        jacobian_medium.ncols(),
        residuals_medium.nrows(),
        elapsed
    );

    // Large problem
    let jacobian_large = create_sparse_jacobian(60000, LARGE_SIZE, 0.005);
    let residuals_large = create_residual_vector(60000);
    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_LARGE {
        let _ = jacobian_large.as_ref().transpose().mul(&residuals_large);
    }
    let elapsed = start.elapsed() / ITERATIONS_LARGE as u32;
    println!(
        "  Size {}x{} * {}x1: {:?} per operation",
        jacobian_large.nrows(),
        jacobian_large.ncols(),
        residuals_large.nrows(),
        elapsed
    );
}

fn profile_cholesky_solver() {
    println!("\nProfiling Sparse Cholesky Solver...");

    // Small problem
    let jacobian_small = create_sparse_jacobian(600, SMALL_SIZE, 0.05);
    let residuals_small = create_residual_vector(600);
    let mut solver_small = SparseCholeskySolver::new();

    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_SMALL {
        let _ = solver_small.solve_normal_equation(&residuals_small, &jacobian_small);
    }
    let elapsed = start.elapsed() / ITERATIONS_SMALL as u32;
    println!(
        "  Size {} variables (normal eq)  : {:?} per solve",
        SMALL_SIZE, elapsed
    );

    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_SMALL {
        let _ = solver_small.solve_augmented_equation(&residuals_small, &jacobian_small, 1e-3);
    }
    let elapsed = start.elapsed() / ITERATIONS_SMALL as u32;
    println!(
        "  Size {} variables (augmented eq): {:?} per solve",
        SMALL_SIZE, elapsed
    );

    // Medium problem
    let jacobian_medium = create_sparse_jacobian(6000, MEDIUM_SIZE, 0.01);
    let residuals_medium = create_residual_vector(6000);
    let mut solver_medium = SparseCholeskySolver::new();

    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_MEDIUM {
        let _ = solver_medium.solve_normal_equation(&residuals_medium, &jacobian_medium);
    }
    let elapsed = start.elapsed() / ITERATIONS_MEDIUM as u32;
    println!(
        "  Size {} variables (normal eq) : {:?} per solve",
        MEDIUM_SIZE, elapsed
    );

    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_MEDIUM {
        let _ = solver_medium.solve_augmented_equation(&residuals_medium, &jacobian_medium, 1e-3);
    }
    let elapsed = start.elapsed() / ITERATIONS_MEDIUM as u32;
    println!(
        "  Size {} variables (augmented eq): {:?} per solve",
        MEDIUM_SIZE, elapsed
    );

    // Large problem
    let jacobian_large = create_sparse_jacobian(60000, LARGE_SIZE, 0.005);
    let residuals_large = create_residual_vector(60000);
    let mut solver_large = SparseCholeskySolver::new();

    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_LARGE {
        let _ = solver_large.solve_normal_equation(&residuals_large, &jacobian_large);
    }
    let elapsed = start.elapsed() / ITERATIONS_LARGE as u32;
    println!(
        "  Size {} variables (normal eq) : {:?} per solve",
        LARGE_SIZE, elapsed
    );

    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_LARGE {
        let _ = solver_large.solve_augmented_equation(&residuals_large, &jacobian_large, 1e-3);
    }
    let elapsed = start.elapsed() / ITERATIONS_LARGE as u32;
    println!(
        "  Size {} variables (augmented eq): {:?} per solve",
        LARGE_SIZE, elapsed
    );
}

fn profile_qr_solver() {
    println!("\nProfiling Sparse QR Solver...");

    // Small problem
    let jacobian_small = create_sparse_jacobian(600, SMALL_SIZE, 0.05);
    let residuals_small = create_residual_vector(600);
    let mut solver_small = SparseQRSolver::new();

    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_SMALL {
        let _ = solver_small.solve_normal_equation(&residuals_small, &jacobian_small);
    }
    let elapsed = start.elapsed() / ITERATIONS_SMALL as u32;
    println!(
        "  Size {} variables (normal eq)  : {:?} per solve",
        SMALL_SIZE, elapsed
    );

    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_SMALL {
        let _ = solver_small.solve_augmented_equation(&residuals_small, &jacobian_small, 1e-3);
    }
    let elapsed = start.elapsed() / ITERATIONS_SMALL as u32;
    println!(
        "  Size {} variables (augmented eq): {:?} per solve",
        SMALL_SIZE, elapsed
    );

    // Medium problem
    let jacobian_medium = create_sparse_jacobian(6000, MEDIUM_SIZE, 0.01);
    let residuals_medium = create_residual_vector(6000);
    let mut solver_medium = SparseQRSolver::new();

    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_MEDIUM {
        let _ = solver_medium.solve_normal_equation(&residuals_medium, &jacobian_medium);
    }
    let elapsed = start.elapsed() / ITERATIONS_MEDIUM as u32;
    println!(
        "  Size {} variables (normal eq) : {:?} per solve",
        MEDIUM_SIZE, elapsed
    );

    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS_MEDIUM {
        let _ = solver_medium.solve_augmented_equation(&residuals_medium, &jacobian_medium, 1e-3);
    }
    let elapsed = start.elapsed() / ITERATIONS_MEDIUM as u32;
    println!(
        "  Size {} variables (augmented eq): {:?} per solve",
        MEDIUM_SIZE, elapsed
    );
}
