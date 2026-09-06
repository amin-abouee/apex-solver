//! First-estimate Jacobians.
//!
//! FEJ evaluates a factor's Jacobian at a frozen linearization point while its
//! residual uses the current estimate. The point is consistency: a marginal
//! prior summarizes information at the state where it was taken, and letting the
//! Jacobians of the states it touches drift afterwards fabricates information
//! along the directions the measurements never observed.

mod common;
use common::lm_solver;

use apex_solver::JacobianEvaluation;
use apex_solver::apex_manifolds::ManifoldType;
use apex_solver::apex_manifolds::rn::Rn;
use apex_solver::core::noise::NoiseModel;
use apex_solver::core::problem::Problem;
use apex_solver::core::{CoreError, VarKey};
use apex_solver::factors::pose::{BetweenFactor, EuclideanPriorFactor};
use apex_solver::linalg::JacobianMode;
use nalgebra::DVector;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn chain(start: &[f64]) -> Result<(Problem, Vec<VarKey>), Box<dyn std::error::Error>> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let keys: Vec<VarKey> = start
        .iter()
        .map(|x| problem.add_variable(ManifoldType::RN, DVector::from_column_slice(&[*x, 0.0])))
        .collect();
    problem.add_residual_block_with_noise(
        &[keys[0]],
        Box::new(EuclideanPriorFactor::new(DVector::from_column_slice(&[
            0.0, 0.0,
        ]))),
        None,
        NoiseModel::from_sigmas(&[0.1, 0.1])?,
    );
    for pair in keys.windows(2) {
        problem.add_residual_block_with_noise(
            &[pair[0], pair[1]],
            Box::new(BetweenFactor::new(Rn::from_vec(vec![1.0, 0.0]))),
            None,
            NoiseModel::from_sigmas(&[0.05, 0.05])?,
        );
    }
    Ok((problem, keys))
}

/// Total robust cost at the current estimate.
fn cost(problem: &mut Problem) -> Result<f64, Box<dyn std::error::Error>> {
    Ok(apex_solver::optimizer::initialize_optimization_state(problem)?.current_cost)
}

#[test]
fn freezing_and_clearing_round_trips() -> TestResult {
    let (mut problem, keys) = chain(&[0.0, 1.0, 2.0])?;
    assert!(!problem.is_linearization_point_frozen(keys[1]));

    problem.freeze_linearization_point(keys[1])?;
    assert!(problem.is_linearization_point_frozen(keys[1]));

    problem.clear_linearization_point(keys[1])?;
    assert!(!problem.is_linearization_point_frozen(keys[1]));
    Ok(())
}

#[test]
fn a_removed_key_cannot_be_frozen() -> TestResult {
    let (mut problem, _keys) = chain(&[0.0, 1.0, 2.0])?;
    let stray = problem.add_variable(ManifoldType::RN, DVector::from_column_slice(&[0.0]));
    problem.try_remove_variable(stray)?;
    let Err(CoreError::Variable(_)) = problem.freeze_linearization_point(stray) else {
        return Err("expected a Variable error for a removed key".into());
    };
    Ok(())
}

/// With the mode on but nothing frozen, assembly must be bit-identical to the
/// ordinary path. This is what keeps FEJ off the hot path for problems that
/// never marginalize.
#[test]
fn first_estimate_is_a_no_op_without_frozen_points() -> TestResult {
    let (mut plain, keys) = chain(&[0.3, 1.4, 2.2])?;
    let (mut fej, _) = chain(&[0.3, 1.4, 2.2])?;
    fej.set_jacobian_evaluation(JacobianEvaluation::FirstEstimate);

    let a = lm_solver(50).optimize(&mut plain)?;
    let b = lm_solver(50).optimize(&mut fej)?;

    for key in &keys {
        let x = a.parameters[*key].as_param_slice();
        let y = b.parameters[*key].as_param_slice();
        assert_eq!(x, y, "FEJ changed a solve with nothing frozen");
    }
    Ok(())
}

/// The residual must still describe where the estimate *is*; only the Jacobian
/// is stale. If freezing also froze the residual, the solve would be blind.
#[test]
fn the_residual_still_follows_the_estimate() -> TestResult {
    let (mut problem, keys) = chain(&[0.0, 5.0, 9.0])?;
    problem.set_jacobian_evaluation(JacobianEvaluation::FirstEstimate);
    for key in &keys {
        problem.freeze_linearization_point(*key)?;
    }
    let before = cost(&mut problem)?;

    // Move a variable well away; the cost must react even though every
    // Jacobian is pinned.
    problem.set_variable_params(keys[2], &[40.0, 0.0])?;
    let after = cost(&mut problem)?;

    assert!(
        after > before * 2.0,
        "cost {after:.3e} barely moved from {before:.3e}; the residual is frozen too"
    );
    Ok(())
}

/// A frozen Jacobian is a *stale* one, so a solve that freezes everything from
/// a bad start must converge differently from one that does not — otherwise
/// nothing is actually being held.
#[test]
fn freezing_changes_the_step_it_takes() -> TestResult {
    let start = [0.0, 6.0, 11.0];

    let (mut plain, keys) = chain(&start)?;
    let free = lm_solver(1).optimize(&mut plain)?;

    let (mut frozen, _) = chain(&start)?;
    frozen.set_jacobian_evaluation(JacobianEvaluation::FirstEstimate);
    for key in &keys {
        frozen.freeze_linearization_point(*key)?;
    }
    // Move the estimate away from the frozen point, so the two Jacobians
    // genuinely differ.
    frozen.set_variable_params(keys[1], &[-3.0, 0.0])?;
    frozen.set_variable_params(keys[2], &[20.0, 0.0])?;
    let held = lm_solver(1).optimize(&mut frozen)?;

    let a = free.parameters[keys[2]].as_param_slice()[0];
    let b = held.parameters[keys[2]].as_param_slice()[0];
    assert!(
        (a - b).abs() > 1e-6,
        "the frozen solve took the same step ({a:.6} vs {b:.6}); nothing was held"
    );
    Ok(())
}

/// Covariance describes the problem at the current estimate, so it must ignore
/// frozen linearization points even when the problem is in FEJ mode.
#[test]
fn covariance_ignores_frozen_linearization_points() -> TestResult {
    let (mut plain, keys) = chain(&[0.2, 1.1, 2.4])?;
    let (mut fej, _) = chain(&[0.2, 1.1, 2.4])?;
    fej.set_jacobian_evaluation(JacobianEvaluation::FirstEstimate);
    for key in &keys {
        fej.freeze_linearization_point(*key)?;
    }
    // Move both estimates identically, so the only difference is the freeze.
    for problem in [&mut plain, &mut fej] {
        problem.set_variable_params(keys[2], &[7.0, 0.0])?;
    }

    let options = apex_solver::linalg::covariance::CovarianceOptions::default();
    let mut a = apex_solver::optimizer::initialize_optimization_state(&mut plain)?.variables;
    let mut b = apex_solver::optimizer::initialize_optimization_state(&mut fej)?.variables;
    let Some(first) = plain.compute_and_set_covariances(&mut a, options) else {
        return Err("covariance failed on the plain problem".into());
    };
    let Some(second) = fej.compute_and_set_covariances(&mut b, options) else {
        return Err("covariance failed on the FEJ problem".into());
    };

    for key in &keys {
        let (Some(x), Some(y)) = (first.get(*key), second.get(*key)) else {
            continue;
        };
        for row in 0..x.nrows() {
            for col in 0..x.ncols() {
                assert!(
                    (x[(row, col)] - y[(row, col)]).abs() < 1e-12,
                    "covariance differs at ({row},{col}); FEJ leaked into it"
                );
            }
        }
    }
    Ok(())
}
