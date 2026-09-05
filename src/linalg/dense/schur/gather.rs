//! `Mat<f64>`-flavored counterparts to [`SchurPartition`]'s sparse-only
//! methods (`verify_block_diagonal`, `EliminatedBlocks::gather`), for a dense
//! Hessian.
//!
//! Dense storage has no sparsity pattern to walk, so both operations are a
//! direct index-and-copy rather than a symbolic-structure traversal. At dense
//! mode's target size (< ~500 DOF) the extra `O(total_dof)` factor this costs
//! relative to the sparse versions is negligible next to the `O(kept_dof³)`
//! Cholesky factorization that follows.

use faer::Mat;

use crate::error::ErrorLogging;
use crate::linalg::schur::{EliminatedBlocks, SchurPartition};
use crate::linalg::{LinAlgError, LinAlgResult};

/// Verify that no factor couples two eliminated variables, reading a dense
/// Hessian directly rather than a sparse symbolic structure.
///
/// See [`SchurPartition::verify_block_diagonal`] for why this precondition
/// matters — eliminating mutually-coupled variables yields a wrong step with
/// no other symptom.
pub fn verify_block_diagonal_dense(
    partition: &SchurPartition,
    hessian: &Mat<f64>,
) -> LinAlgResult<()> {
    for block in partition.eliminated_blocks() {
        for offset in 0..block.dof {
            let row = block.col_start + offset;
            let Some((this_block, _)) = partition.eliminated_local(row) else {
                continue;
            };
            for col in 0..hessian.ncols() {
                if hessian[(row, col)] == 0.0 {
                    continue;
                }
                if let Some((other_block, _)) = partition.eliminated_local(col)
                    && other_block != this_block
                {
                    let other = partition.eliminated_blocks()[other_block];
                    return Err(LinAlgError::InvalidInput(format!(
                        "variables {:?} and {:?} are both marked for elimination but are \
                         connected by a factor, so H_ee is not block-diagonal and Schur \
                         elimination would give a wrong step; eliminate only mutually \
                         unconnected variables",
                        block.key, other.key
                    ))
                    .log());
                }
            }
        }
    }
    Ok(())
}

/// Gather the diagonal blocks of `H_ee` out of a dense Hessian.
///
/// Mirrors [`EliminatedBlocks::gather`], reading `hessian[(row, col)]`
/// directly instead of walking a sparse column's nonzeros.
pub fn gather_dense(
    eliminated: &mut EliminatedBlocks,
    partition: &SchurPartition,
    hessian: &Mat<f64>,
) {
    eliminated.clear();
    for (block_idx, block) in partition.eliminated_blocks().iter().enumerate() {
        let mut view = eliminated.block_mut_ref(block_idx);
        for local_col in 0..block.dof {
            let global_col = block.col_start + local_col;
            for local_row in 0..block.dof {
                let global_row = block.col_start + local_row;
                view[(local_row, local_col)] = hessian[(global_row, global_col)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::VarKey;
    use crate::linalg::schur::BlockSpan;
    use slotmap::KeyData;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn key(i: u64) -> VarKey {
        VarKey::from(KeyData::from_ffi((1u64 << 32) | i))
    }

    fn span(k: u64, col_start: usize, dof: usize) -> BlockSpan {
        BlockSpan {
            key: key(k),
            col_start,
            dof,
        }
    }

    #[test]
    fn gather_dense_picks_up_diagonal_blocks_only() -> TestResult {
        let p = SchurPartition::new(vec![span(0, 0, 1)], vec![span(1, 1, 1), span(2, 2, 2)])?;
        let mut h = Mat::<f64>::zeros(5, 5);
        h[(0, 0)] = 9.0;
        h[(1, 1)] = 2.0;
        h[(0, 1)] = 7.0; // kept<->eliminated coupling must be ignored here
        h[(1, 0)] = 7.0;
        h[(2, 2)] = 4.0;
        h[(3, 3)] = 5.0;
        h[(4, 4)] = 6.0;
        h[(2, 3)] = 1.0;
        h[(3, 2)] = 1.0;

        let mut blocks = EliminatedBlocks::new(&p);
        gather_dense(&mut blocks, &p, &h);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks.dof(0), 1);
        assert_eq!(blocks.dof(1), 2);
        assert!((blocks.at(0, 0, 0) - 2.0).abs() < 1e-12);
        assert!((blocks.at(1, 0, 0) - 4.0).abs() < 1e-12);
        assert!((blocks.at(1, 1, 1) - 5.0).abs() < 1e-12);
        assert!((blocks.at(1, 0, 1) - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn verify_block_diagonal_dense_rejects_coupled_eliminated_variables() -> TestResult {
        let p = SchurPartition::new(vec![span(0, 0, 1)], vec![span(1, 1, 1), span(2, 2, 1)])?;

        let mut ok = Mat::<f64>::zeros(3, 3);
        ok[(0, 0)] = 1.0;
        ok[(1, 1)] = 1.0;
        ok[(2, 2)] = 1.0;
        ok[(0, 1)] = 0.5;
        ok[(1, 0)] = 0.5;
        verify_block_diagonal_dense(&p, &ok)?;

        let mut bad = ok.clone();
        bad[(1, 2)] = 0.5;
        bad[(2, 1)] = 0.5;
        let Err(err) = verify_block_diagonal_dense(&p, &bad) else {
            panic!("coupled eliminated variables must be rejected");
        };
        assert!(err.to_string().contains("block-diagonal"), "{err}");
        Ok(())
    }
}
