//! Shared parameter-block validation for factors.

use crate::core::variable::ManifoldVariable;
use crate::error::ErrorLogging;
use crate::factors::{FactorError, FactorResult};

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
    factor: &'static str,
) -> FactorResult<()> {
    expect_count(variables, expected.len(), factor)?;
    for (block, (variable, &want)) in variables.iter().zip(expected).enumerate() {
        let actual = variable.as_param_slice().len();
        if actual != want {
            return Err(FactorError::VariableSize {
                factor,
                block,
                expected: want,
                actual,
            }
            .log());
        }
    }
    Ok(())
}

/// Like [`expect_block_sizes`], but every block must have the same given size.
pub(crate) fn expect_uniform_blocks(
    variables: &[&dyn ManifoldVariable],
    count: usize,
    size: usize,
    factor: &'static str,
) -> FactorResult<()> {
    expect_block_sizes(variables, &vec![size; count], factor)
}

/// Check that a factor is registered with variables of the given manifolds,
/// by [`ManifoldVariable::manifold_type_name`] — for factors generic over the
/// group, where a parameter count alone is ambiguous (`SE3` and a 7-D `Rn`).
pub(crate) fn expect_manifolds(
    variables: &[&dyn ManifoldVariable],
    expected: &[&'static str],
    factor: &'static str,
) -> FactorResult<()> {
    expect_count(variables, expected.len(), factor)?;
    for (block, (variable, &want)) in variables.iter().zip(expected).enumerate() {
        expect_manifold(*variable, block, want, factor)?;
    }
    Ok(())
}

/// Check the number of variables a factor is registered with.
pub(crate) fn expect_count(
    variables: &[&dyn ManifoldVariable],
    expected: usize,
    factor: &'static str,
) -> FactorResult<()> {
    if variables.len() != expected {
        return Err(FactorError::VariableCount {
            factor,
            expected,
            actual: variables.len(),
        }
        .log());
    }
    Ok(())
}

/// Check that block `block` lives on the manifold named `expected`.
pub(crate) fn expect_manifold(
    variable: &dyn ManifoldVariable,
    block: usize,
    expected: &'static str,
    factor: &'static str,
) -> FactorResult<()> {
    let actual = variable.manifold_type_name();
    if actual != expected {
        return Err(FactorError::VariableManifold {
            factor,
            block,
            expected,
            actual,
        }
        .log());
    }
    Ok(())
}

/// Check that block `block` has `expected` parameters.
pub(crate) fn expect_size(
    variable: &dyn ManifoldVariable,
    block: usize,
    expected: usize,
    factor: &'static str,
) -> FactorResult<()> {
    let actual = variable.as_param_slice().len();
    if actual != expected {
        return Err(FactorError::VariableSize {
            factor,
            block,
            expected,
            actual,
        }
        .log());
    }
    Ok(())
}

/// Check that block `block` has `expected` degrees of freedom.
pub(crate) fn expect_dof(
    variable: &dyn ManifoldVariable,
    block: usize,
    expected: usize,
    factor: &'static str,
) -> FactorResult<()> {
    let actual = variable.dof();
    if actual != expected {
        return Err(FactorError::VariableDof {
            factor,
            block,
            expected,
            actual,
        }
        .log());
    }
    Ok(())
}

/// A finite, strictly positive scalar, returned unchanged.
pub(crate) fn expect_positive(
    factor: &'static str,
    what: &'static str,
    value: f64,
) -> FactorResult<f64> {
    if value.is_finite() && value > 0.0 {
        return Ok(value);
    }
    Err(FactorError::InvalidValue {
        factor,
        what,
        requirement: "finite and positive",
        value,
    }
    .log())
}

/// Every value finite.
pub(crate) fn expect_finite<'a>(
    factor: &'static str,
    what: &'static str,
    values: impl IntoIterator<Item = &'a f64>,
) -> FactorResult<()> {
    if values.into_iter().all(|value| value.is_finite()) {
        return Ok(());
    }
    Err(FactorError::NonFinite { factor, what }.log())
}

/// At least `min` items.
pub(crate) fn expect_at_least(
    factor: &'static str,
    what: &'static str,
    actual: usize,
    min: usize,
) -> FactorResult<()> {
    if actual >= min {
        return Ok(());
    }
    Err(FactorError::TooFew {
        factor,
        what,
        min,
        actual,
    }
    .log())
}

/// A group whose tangent dimension fits the factor's fixed-size buffers.
pub(crate) fn expect_group_dof(
    factor: &'static str,
    group: &'static str,
    dof: usize,
    max: usize,
) -> FactorResult<()> {
    if dof <= max {
        return Ok(());
    }
    Err(FactorError::UnsupportedGroup {
        factor,
        group,
        dof,
        max,
    }
    .log())
}

/// An image of at least `2 × 2` pixels whose buffer holds `width · height`
/// values, so bilinear interpolation always has a full neighbourhood.
pub(crate) fn expect_image(
    image: &'static str,
    width: usize,
    height: usize,
    values: usize,
) -> FactorResult<()> {
    expect_at_least(image, "width", width, 2)?;
    expect_at_least(image, "height", height, 2)?;
    if values != width * height {
        return Err(FactorError::DimensionMismatch {
            factor: image,
            what: "pixel buffer",
            expected: width * height,
            actual: values,
        }
        .log());
    }
    Ok(())
}

/// A finite point and a finite, positive residual sigma.
pub(crate) fn expect_point_and_sigma(
    factor: &'static str,
    point: &nalgebra::Vector3<f64>,
    sigma: f64,
) -> FactorResult<()> {
    expect_finite(factor, "point", point.iter())?;
    expect_positive(factor, "sigma", sigma)?;
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
