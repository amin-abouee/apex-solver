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

/// Check that a factor is registered with variables of the given manifolds,
/// by [`ManifoldVariable::manifold_type_name`] — for factors generic over the
/// group, where a parameter count alone is ambiguous (`SE3` and a 7-D `Rn`).
pub(crate) fn expect_manifolds(
    variables: &[&dyn ManifoldVariable],
    expected: &[&str],
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
        let got = variable.manifold_type_name();
        if got != want {
            return Err(format!(
                "{message} — block {index} is {got}, expected {want}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::variable::Variable;
    use apex_manifolds::rn::Rn;
    use apex_manifolds::se3::SE3;
    use apex_manifolds::se23::SE23;
    use nalgebra::DVector;

    #[test]
    fn manifolds_are_matched_by_name_not_size() {
        let se3 = Variable::new(SE3::identity());
        let se23 = Variable::new(SE23::identity());
        // Seven parameters, like SE3 — a size check alone would accept it.
        let rn7 = Variable::new(Rn::new(DVector::zeros(7)));
        assert!(expect_manifolds(&[&se23, &se3], &["SE23", "SE3"], "m").is_ok());
        assert!(expect_manifolds(&[&se23, &rn7], &["SE23", "SE3"], "m").is_err());
        assert!(expect_manifolds(&[&se23], &["SE23", "SE3"], "m").is_err());
        assert!(expect_block_sizes(&[&se23, &rn7], &[10, 7], "m").is_ok());
    }
}
