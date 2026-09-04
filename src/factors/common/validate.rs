//! Shared parameter-block validation for factors.

use crate::core::variable::ManifoldVariable;

/// Check that a factor is being registered with the variables it expects.
///
/// `expected` lists the parameter-vector length of each block, in order.
///
/// This matters more than the `debug_assert!`s factors also carry: the test
/// profile inherits `release` (see `[profile.test]`), so those assertions are
/// compiled out of `cargo test` and a mismatched registration would surface as
/// a panic or silent corruption deep inside the parallel assembly instead of a
/// typed error at registration.
pub(crate) fn expect_block_sizes(
    variables: &[&dyn ManifoldVariable],
    expected: &[usize],
    message: &str,
) -> Result<(), String> {
    if variables.len() != expected.len() {
        return Err(format!(
            "{message} — got {} blocks, expected {}",
            variables.len(),
            expected.len()
        ));
    }
    for (index, (variable, &want)) in variables.iter().zip(expected).enumerate() {
        let got = variable.as_param_slice().len();
        if got != want {
            return Err(format!(
                "{message} — block {index} has {got} parameters, expected {want}"
            ));
        }
    }
    Ok(())
}

/// Like [`expect_block_sizes`], but every block must have the same given size.
pub(crate) fn expect_uniform_blocks(
    variables: &[&dyn ManifoldVariable],
    count: usize,
    size: usize,
    message: &str,
) -> Result<(), String> {
    expect_block_sizes(variables, &vec![size; count], message)
}
