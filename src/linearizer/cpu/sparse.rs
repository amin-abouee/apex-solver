//! Sparse Jacobian assembly using symbolic sparsity patterns.

use faer::{
    Mat,
    sparse::{SparseColMat, SymbolicSparseColMat},
};
use rayon::prelude::*;
use slotmap::{SecondaryMap, SlotMap};

use crate::core::VarKey;
use crate::error::ErrorLogging;
use crate::linearizer::{
    AssemblyWorkspace, LinearizerError, LinearizerResult, compute_block_into,
    split_by_row_offsets_mut,
};

use crate::core::problem::Problem;
use crate::core::variable::ManifoldVariable;

/// Symbolic structure for sparse matrix operations.
pub struct SymbolicStructure {
    pub pattern: SymbolicSparseColMat<usize>,
    /// Scatter plan for the CSC value array: `(csc_slot, arena_index,
    /// accumulate)`. Each plan step either overwrites or adds-to the CSC
    /// slot from one Jacobian-arena position. Duplicated `(row, col)` pairs
    /// (a factor listing one variable twice) produce a first overwrite
    /// followed by accumulate steps; the common no-duplicate case is a pure
    /// parallel gather.
    pub scatter_ops: Vec<ScatterOp>,
    /// True when any plan step accumulates (duplicate pairs exist), which
    /// forces a serial scatter; the common case gathers in parallel.
    pub scatter_has_duplicates: bool,
}

/// One step of the CSC value scatter plan.
#[derive(Debug, Clone, Copy)]
pub struct ScatterOp {
    pub dest: u32,
    pub src: u32,
    pub accumulate: bool,
}

/// Free local columns of `variable`, as `(local_col, free_offset)` pairs —
/// shared by [`build_symbolic_structure`] and [`scatter_sparse_block`] so
/// their column ordering can never drift apart. Fixed tangent coordinates
/// (see ISSUE-0003) own no column and are skipped by both.
fn free_local_columns(
    variable: &dyn ManifoldVariable,
    var_size: usize,
) -> impl Iterator<Item = (usize, usize)> + '_ {
    (0..var_size).filter_map(move |col| variable.local_free_offset(col).map(|free| (col, free)))
}

/// Build the symbolic sparsity structure for the Jacobian matrix.
///
/// Blocks are visited in `residual_row_start_idx` order, matching
/// [`AssemblyWorkspace`](crate::linearizer::AssemblyWorkspace) `block_order`
/// and therefore the order [`assemble_sparse`] pushes Jacobian values in.
/// That shared order is load-bearing: `new_from_argsort` pairs the pushed
/// values with these pairs positionally, so any disagreement silently
/// permutes `J`'s rows against `r`. Slotmap iteration order coincides with
/// row order only until rows are regrouped
/// ([`Problem::group_rows_for_elimination`](crate::core::problem::Problem::group_rows_for_elimination)),
/// which is exactly when the Schur path needs assembly to be right.
pub fn build_symbolic_structure(
    problem: &Problem,
    variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    variable_index_map: &SecondaryMap<VarKey, usize>,
    total_dof: usize,
) -> LinearizerResult<SymbolicStructure> {
    // Triples in push order: `(col, row, arena_index)` — the arena index is
    // where `assemble_sparse`'s parallel factor evaluation left the value,
    // so the CSC values are a pure permutation (plus duplicate sums) of the
    // arena.
    let mut triples = Vec::<(usize, usize, u32)>::new();
    let mut total_jac_len = 0usize;

    let mut blocks: Vec<_> = problem.residual_blocks().iter().collect();
    blocks.sort_by_key(|(_, block)| block.residual_row_start_idx);
    for (_, block) in &blocks {
        let residual_dim = block.factor.residual_dim();
        let (jac_rows, jac_cols) = block.factor.jacobian_shape();
        debug_assert_eq!(jac_rows, residual_dim);
        let jac_base = total_jac_len;
        total_jac_len += jac_rows * jac_cols;

        let mut var_local_sizes = Vec::<(usize, usize)>::new();
        let mut local_offset = 0;

        for &var_key in &block.variable_keys {
            if let Some(variable) = variables.get(var_key) {
                var_local_sizes.push((local_offset, variable.dof()));
                local_offset += variable.dof();
            }
        }

        for (i, &var_key) in block.variable_keys.iter().enumerate() {
            if let Some(variable) = variables.get(var_key) {
                let Some(&global_col) = variable_index_map.get(var_key) else {
                    return Err(LinearizerError::Variable(format!(
                        "VarKey {var_key:?} missing in variable-to-column-index mapping"
                    ))
                    .log());
                };
                let Some(&(local_col, var_size)) = var_local_sizes.get(i) else {
                    continue;
                };
                let free_cols: smallvec::SmallVec<[(usize, usize); 16]> =
                    free_local_columns(variable.as_ref(), var_size).collect();
                triples.reserve(residual_dim * free_cols.len());
                for row in 0..residual_dim {
                    for &(col, free_col) in &free_cols {
                        // Jacobian arena is column-major over the block:
                        // buf[(local_col + col) * residual_dim + row].
                        triples.push((
                            global_col + free_col,
                            block.residual_row_start_idx + row,
                            (jac_base + (local_col + col) * residual_dim + row) as u32,
                        ));
                    }
                }
            }
        }
    }

    let plan = csc_from_triples(triples, total_dof)?;
    let pattern = SymbolicSparseColMat::new_checked(
        problem.total_residual_dimension,
        total_dof,
        plan.col_ptr,
        None,
        plan.row_idx,
    );

    Ok(SymbolicStructure {
        pattern,
        scatter_ops: plan.scatter_ops,
        scatter_has_duplicates: plan.has_duplicates,
    })
}

/// CSC pattern and scatter plan of the Jacobian, from its `(col, row,
/// arena_index)` triples in push order.
struct CscPlan {
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    scatter_ops: Vec<ScatterOp>,
    has_duplicates: bool,
}

fn csc_from_triples(
    triples: Vec<(usize, usize, u32)>,
    total_dof: usize,
) -> LinearizerResult<CscPlan> {
    // CSC order: grouped by column, rows ascending within a column, duplicate
    // pairs in push order so their summation order matches faer's argsort.
    // A stable counting sort by column gives exactly that without a full
    // comparison sort: blocks are visited in ascending row order and push
    // their rows ascending, so each column's entries already arrive
    // row-sorted. The per-column check sorts (stably) only if that ever
    // stops holding.
    let mut bucket = vec![0usize; total_dof + 1];
    for &(c, _, _) in &triples {
        let Some(count) = bucket.get_mut(c + 1) else {
            return Err(LinearizerError::SymbolicStructure(format!(
                "Jacobian column {c} outside the {total_dof} solve columns"
            ))
            .log());
        };
        *count += 1;
    }
    for c in 0..total_dof {
        bucket[c + 1] += bucket[c];
    }
    let mut next = bucket.clone();
    let mut by_column = vec![(0usize, 0u32); triples.len()];
    for &(c, r, src) in &triples {
        by_column[next[c]] = (r, src);
        next[c] += 1;
    }
    drop(triples);

    let mut col_ptr = vec![0usize; total_dof + 1];
    let mut row_idx = Vec::with_capacity(by_column.len());
    let mut scatter_ops = Vec::<ScatterOp>::with_capacity(by_column.len());
    let mut has_duplicates = false;
    let mut last_dest = 0u32;
    for c in 0..total_dof {
        let entries = &mut by_column[bucket[c]..bucket[c + 1]];
        if !entries.is_sorted_by_key(|&(r, _)| r) {
            entries.sort_by_key(|&(r, _)| r);
        }
        let mut prev: Option<usize> = None;
        for &(r, src) in entries.iter() {
            if prev == Some(r) {
                // faer's argsort semantics: a duplicated (row, col) pair sums
                // into the first occurrence's value slot.
                has_duplicates = true;
                scatter_ops.push(ScatterOp {
                    dest: last_dest,
                    src,
                    accumulate: true,
                });
            } else {
                row_idx.push(r);
                last_dest = row_idx.len() as u32 - 1;
                scatter_ops.push(ScatterOp {
                    dest: last_dest,
                    src,
                    accumulate: false,
                });
            }
            prev = Some(r);
        }
        col_ptr[c + 1] = row_idx.len();
    }

    Ok(CscPlan {
        col_ptr,
        row_idx,
        scatter_ops,
        has_duplicates,
    })
}

/// Assemble residuals and sparse Jacobian from the current variable values.
///
/// Reuses the block ordering, slice offsets and scratch buffers cached in
/// `workspace` — nothing static is rebuilt or reallocated per call.
pub fn assemble_sparse(
    problem: &Problem,
    variables: &SlotMap<VarKey, Box<dyn ManifoldVariable>>,
    _variable_index_map: &SecondaryMap<VarKey, usize>,
    symbolic_structure: &SymbolicStructure,
    workspace: &mut AssemblyWorkspace,
) -> LinearizerResult<(Mat<f64>, SparseColMat<usize, f64>)> {
    let total_nnz = symbolic_structure.pattern.compute_nnz();

    // Reset the residual buffer, then split it (and the Jacobian arena) into
    // non-overlapping slices in the pre-computed block order.
    workspace.residual_buf.fill(0.0);
    let residual_slices =
        split_by_row_offsets_mut(&mut workspace.residual_buf, &workspace.offsets_lens);
    let jac_slices = split_by_row_offsets_mut(&mut workspace.jac_arena, &workspace.jac_offsets);
    let residual_blocks = problem.residual_blocks();

    // Parallel evaluation: each task gets a unique residual slice and a unique
    // Jacobian buffer (mutable, non-aliasing) — pure zero-copy through factor.linearize.
    residual_slices
        .into_par_iter()
        .zip(jac_slices)
        .zip(workspace.block_order.par_iter())
        .map(|((res_slice, jac_buf), key)| {
            let block = &residual_blocks[*key];
            jac_buf.fill(0.0);
            compute_block_into(block, variables, res_slice, Some(jac_buf)).map(|_| ())
        })
        .collect::<LinearizerResult<()>>()?;

    // CSC values are a fixed permutation (plus duplicate sums) of the arena,
    // precomputed in the symbolic structure — no serial scatter, no argsort.
    let mut csc_values = std::mem::take(&mut workspace.jacobian_values);
    csc_values.resize(total_nnz, 0.0);
    if symbolic_structure.scatter_has_duplicates {
        for op in &symbolic_structure.scatter_ops {
            let v = workspace.jac_arena[op.src as usize];
            if op.accumulate {
                csc_values[op.dest as usize] += v;
            } else {
                csc_values[op.dest as usize] = v;
            }
        }
    } else {
        csc_values
            .par_iter_mut()
            .zip(symbolic_structure.scatter_ops.par_iter())
            .for_each(|(v, op)| *v = workspace.jac_arena[op.src as usize]);
    }
    let jacobian_sparse = SparseColMat::new(symbolic_structure.pattern.clone(), csc_values);

    // The value buffer was consumed by the matrix; give the workspace a
    // fresh one (same capacity class) for the next call.
    workspace.jacobian_values = Vec::with_capacity(total_nnz);

    // Convert residual buffer to faer Mat
    let n = problem.total_residual_dimension;
    let residual_faer = faer::Mat::from_fn(n, 1, |i, _| workspace.residual_buf[i]);

    Ok((residual_faer, jacobian_sparse))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::problem::Problem, factors, linalg::JacobianMode};
    use apex_manifolds::ManifoldType;
    use faer::prelude::ReborrowMut;
    use nalgebra::dvector;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    struct LinearFactor {
        target: f64,
    }

    impl factors::Factor for LinearFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = params[0][0] - self.target;
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = 1.0;
            }
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 1)
        }
    }

    fn one_var_problem() -> (Problem, VarKey) {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let k = problem.add_variable(ManifoldType::RN, dvector![5.0]);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 0.0 }), None);
        (problem, k)
    }

    /// Unary factor with a configurable Jacobian gain, so a row permutation
    /// of `J` is observable (uniform gains would hide it).
    struct ScaledLinearFactor {
        target: f64,
        gain: f64,
    }

    impl factors::Factor for ScaledLinearFactor {
        fn linearize(
            &self,
            params: &[&[f64]],
            residual: &mut [f64],
            jacobian: Option<faer::mat::MatMut<'_, f64>>,
        ) {
            residual[0] = params[0][0] - self.target;
            if let Some(mut jac) = jacobian {
                *jac.rb_mut().get_mut(0, 0) = self.gain;
            }
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn jacobian_shape(&self) -> (usize, usize) {
            (1, 1)
        }
    }

    fn build_index_map(problem: &Problem) -> (SecondaryMap<VarKey, usize>, usize) {
        let mut map = SecondaryMap::new();
        let mut offset = 0;
        for (k, v) in &problem.variables {
            map.insert(k, offset);
            offset += v.free_dof();
        }
        (map, offset)
    }

    #[test]
    fn test_build_symbolic_structure_nnz() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        assert_eq!(sym.pattern.compute_nnz(), 1);
        Ok(())
    }

    #[test]
    fn test_build_symbolic_structure_dimensions() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        assert_eq!(sym.pattern.nrows(), 1);
        assert_eq!(sym.pattern.ncols(), 1);
        Ok(())
    }

    #[test]
    fn test_build_symbolic_structure_two_factors() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let k = problem.add_variable(ManifoldType::RN, dvector![5.0]);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 0.0 }), None);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 1.0 }), None);
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        assert_eq!(sym.pattern.compute_nnz(), 2);
        Ok(())
    }

    /// Regrouped rows must not desynchronize the sparse Jacobian from the
    /// residual vector.
    ///
    /// Regression: the symbolic index pairs were pushed in slotmap order while
    /// Jacobian values scatter in row-start order, so after
    /// `group_rows_for_elimination` the two orders disagreed and `J`'s rows no
    /// longer matched `r`'s — silently corrupting every Schur-backed solve
    /// while Cholesky (which never regroups) stayed exact. The dense path
    /// indexes absolutely and is immune, so it serves as the oracle here.
    /// Insertion is deliberately C,B,A so slotmap order and grouped row order
    /// differ.
    #[test]
    fn test_grouped_sparse_assembly_matches_dense() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let c = problem.add_variable(ManifoldType::RN, dvector![3.0]);
        let b = problem.add_variable(ManifoldType::RN, dvector![2.0]);
        let a = problem.add_variable(ManifoldType::RN, dvector![1.0]);
        // Distinct gains per block so a row permutation of J is observable.
        for (key, target, gain) in [(c, 30.0, 3.0), (b, 20.0, 2.0), (a, 10.0, 1.0)] {
            problem.add_residual_block(&[key], Box::new(ScaledLinearFactor { target, gain }), None);
        }
        problem.mark_for_elimination(c);
        assert!(
            problem.group_rows_for_elimination(),
            "insertion order must differ from grouped order for this test"
        );

        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let mut workspace = AssemblyWorkspace::build(&problem);
        let (r_sparse, j_sparse) = assemble_sparse(
            &problem,
            &problem.variables,
            &index_map,
            &sym,
            &mut workspace,
        )?;
        let (r_dense, j_dense) = crate::linearizer::cpu::dense::assemble_dense(
            &problem,
            &problem.variables,
            &index_map,
            total_dof,
            &mut workspace,
        )?;

        assert_eq!(r_sparse.nrows(), 3);
        for i in 0..3 {
            assert!(
                (r_sparse[(i, 0)] - r_dense[(i, 0)]).abs() < 1e-12,
                "residual row {i} differs"
            );
        }
        // Each block is unary: exactly one nonzero per row and column, and it
        // must sit in the same (row, col) in both assemblies — with distinct
        // gains per block, any row permutation shows up as a value mismatch.
        assert_eq!(j_sparse.as_ref().compute_nnz(), 3);
        for col in 0..total_dof {
            let rows = j_sparse.symbolic().row_idx_of_col_raw(col);
            let vals = j_sparse.val_of_col(col);
            assert_eq!(rows.len(), 1, "col {col} must hold exactly one entry");
            assert!(
                (vals[0] - j_dense[(rows[0], col)]).abs() < 1e-12,
                "sparse value {} disagrees with dense {} at ({}, {col})",
                vals[0],
                j_dense[(rows[0], col)],
                rows[0]
            );
            for row in 0..3 {
                if row != rows[0] {
                    assert!(
                        j_dense[(row, col)].abs() < 1e-12,
                        "dense has an unexpected entry at ({row}, {col})"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_basic() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let (residual, _) = assemble_sparse(
            &problem,
            &problem.variables,
            &index_map,
            &sym,
            &mut AssemblyWorkspace::build(&problem),
        )?;
        assert!((residual[(0, 0)] - 5.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_jacobian_value() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let (_, jacobian) = assemble_sparse(
            &problem,
            &problem.variables,
            &index_map,
            &sym,
            &mut AssemblyWorkspace::build(&problem),
        )?;
        let val = jacobian.as_ref().val_of_col(0)[0];
        assert!((val - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_zero_residual() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let k = problem.add_variable(ManifoldType::RN, dvector![3.0]);
        problem.add_residual_block(&[k], Box::new(LinearFactor { target: 3.0 }), None);
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let (residual, _) = assemble_sparse(
            &problem,
            &problem.variables,
            &index_map,
            &sym,
            &mut AssemblyWorkspace::build(&problem),
        )?;
        assert!(residual[(0, 0)].abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_dimensions() -> TestResult {
        let (problem, _) = one_var_problem();
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let (residual, jacobian) = assemble_sparse(
            &problem,
            &problem.variables,
            &index_map,
            &sym,
            &mut AssemblyWorkspace::build(&problem),
        )?;
        assert_eq!(residual.nrows(), 1);
        assert_eq!(residual.ncols(), 1);
        assert_eq!(jacobian.nrows(), 1);
        assert_eq!(jacobian.ncols(), 1);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_two_variables() -> TestResult {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let kx = problem.add_variable(ManifoldType::RN, dvector![2.0]);
        let ky = problem.add_variable(ManifoldType::RN, dvector![7.0]);
        problem.add_residual_block(&[kx], Box::new(LinearFactor { target: 0.0 }), None);
        problem.add_residual_block(&[ky], Box::new(LinearFactor { target: 0.0 }), None);
        let (index_map, total_dof) = build_index_map(&problem);
        let sym = build_symbolic_structure(&problem, &problem.variables, &index_map, total_dof)?;
        let (residual, _) = assemble_sparse(
            &problem,
            &problem.variables,
            &index_map,
            &sym,
            &mut AssemblyWorkspace::build(&problem),
        )?;
        assert_eq!(residual.nrows(), 2);
        let rsum = residual[(0, 0)].abs() + residual[(1, 0)].abs();
        assert!((rsum - 9.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_assemble_sparse_missing_variable_key_returns_error() -> TestResult {
        let (problem, _) = one_var_problem();
        let (_, total_dof) = build_index_map(&problem);
        // The variable-to-column mapping is consumed at symbolic-structure
        // build time (it produces the scatter plan), so a missing entry is
        // reported there.
        let empty: SecondaryMap<VarKey, usize> = SecondaryMap::new();
        let result = build_symbolic_structure(&problem, &problem.variables, &empty, total_dof);
        assert!(result.is_err(), "expected Err for missing variable key");
        Ok(())
    }

    /// `(col_ptr, row_idx, (dest, src, accumulate) per op, has_duplicates)`.
    type PlanParts = (Vec<usize>, Vec<usize>, Vec<(u32, u32, bool)>, bool);

    /// The comparison-sort construction `csc_from_triples` replaced, kept as
    /// the reference its output must equal.
    fn csc_by_stable_sort(mut triples: Vec<(usize, usize, u32)>, total_dof: usize) -> PlanParts {
        triples.sort_by_key(|&(c, r, _)| (c, r));
        let mut col_ptr = vec![0usize; total_dof + 1];
        let mut row_idx = Vec::new();
        let mut ops = Vec::new();
        let mut dup = false;
        let mut prev = None;
        let mut last = 0u32;
        for &(c, r, src) in &triples {
            if prev == Some((c, r)) {
                dup = true;
                ops.push((last, src, true));
            } else {
                col_ptr[c + 1] += 1;
                row_idx.push(r);
                last = row_idx.len() as u32 - 1;
                ops.push((last, src, false));
            }
            prev = Some((c, r));
        }
        for c in 0..total_dof {
            col_ptr[c + 1] += col_ptr[c];
        }
        (col_ptr, row_idx, ops, dup)
    }

    fn plan_parts(plan: CscPlan) -> PlanParts {
        let ops = plan
            .scatter_ops
            .iter()
            .map(|op| (op.dest, op.src, op.accumulate))
            .collect();
        (plan.col_ptr, plan.row_idx, ops, plan.has_duplicates)
    }

    /// Row-ascending push order (what `build_symbolic_structure` produces),
    /// with duplicate pairs, and a shuffled order that forces the per-column
    /// fallback sort: both must match the stable comparison sort exactly —
    /// pattern, destinations, and the summation order of duplicates.
    #[test]
    fn counting_sort_plan_matches_stable_comparison_sort() -> TestResult {
        let total_dof = 7;
        let mut ordered = Vec::new();
        let mut src = 0u32;
        for row in 0..40usize {
            for col in [row % 7, (row * 3 + 1) % 7, (row * 5 + 2) % 7] {
                ordered.push((col, row, src));
                src += 1;
                if row % 6 == 0 {
                    // Same variable listed twice by one factor.
                    ordered.push((col, row, src));
                    src += 1;
                }
            }
        }
        let mut shuffled = ordered.clone();
        let n = shuffled.len();
        for i in 0..n {
            shuffled.swap(i, (i * 7919 + 13) % n);
        }
        for triples in [ordered, shuffled] {
            let want = csc_by_stable_sort(triples.clone(), total_dof);
            let got = plan_parts(csc_from_triples(triples, total_dof)?);
            assert_eq!(got, want);
        }
        assert!(csc_from_triples(vec![(7, 0, 0)], total_dof).is_err());
        Ok(())
    }
}
