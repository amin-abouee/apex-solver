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

// ── The marginalize → freeze → solve → write back → re-marginalize loop ──────
//
// Everything above tests freezing in isolation. What a sliding window actually
// runs is the loop, and until now nothing exercised `Marginalizer` together with
// `JacobianEvaluation::FirstEstimate` at all.

/// Writing a solve result back must not clear the freezes the caller just set.
///
/// `update_values_from` is built on `set_variable_params`, which rebuilds the
/// variable. It used to carry over only the fixed indices and the bounds, so a
/// window that wrote its result back after every step silently reverted to
/// current-estimate Jacobians from the second step onwards — invisible in the
/// cost, and the exact inconsistency FEJ exists to prevent.
#[test]
fn writing_values_back_preserves_the_freeze() -> TestResult {
    let (mut problem, keys) = chain(&[0.0, 1.0, 2.0])?;
    problem.freeze_linearization_point(keys[1])?;
    let frozen = problem
        .variable(keys[1])
        .and_then(|v| v.linearization_point())
        .map(<[f64]>::to_vec);

    let mut solver = lm_solver(20);
    let result = solver.optimize(&mut problem)?;
    problem.update_values_from(&result.parameters)?;

    assert!(
        problem.is_linearization_point_frozen(keys[1]),
        "the freeze must survive a write-back"
    );
    let after = problem
        .variable(keys[1])
        .and_then(|v| v.linearization_point())
        .map(<[f64]>::to_vec);
    assert_eq!(
        frozen, after,
        "the frozen point must not follow the estimate"
    );
    Ok(())
}

/// A caller that rebuilds its problem restores `x₀` explicitly.
///
/// `freeze_linearization_point` captures wherever the variable sits, which in a
/// rebuilt problem is the solved value — a moving target. `set_linearization_
/// point` is what makes the frozen point stable across rebuilds.
#[test]
fn an_explicit_linearization_point_survives_a_rebuild() -> TestResult {
    let (mut problem, keys) = chain(&[0.0, 1.0, 2.0])?;
    let x0 = vec![7.5, -3.25];
    problem.set_linearization_point(keys[1], &x0)?;

    assert!(problem.is_linearization_point_frozen(keys[1]));
    assert_eq!(
        problem
            .variable(keys[1])
            .and_then(|v| v.linearization_point())
            .map(<[f64]>::to_vec),
        Some(x0.clone())
    );
    // The estimate itself is untouched.
    assert_eq!(
        problem.variable_params(keys[1]).map(<[f64]>::to_vec),
        Some(vec![1.0, 0.0])
    );

    let Err(CoreError::Variable(message)) = problem.set_linearization_point(keys[1], &[1.0]) else {
        return Err("a wrong-length linearization point must be rejected".into());
    };
    assert!(message.contains("takes 2 parameters"), "{message}");
    Ok(())
}

/// The whole loop: marginalize, freeze what the prior touches, solve, write
/// back, marginalize again. Ten times, asserting the prior stays well formed and
/// the answer stays where the full batch put it.
#[test]
fn a_repeated_marginalize_and_freeze_loop_stays_consistent() -> TestResult {
    use apex_solver::{JacobianEvaluation as Eval, Marginalizer};

    // The full-batch answer, for reference.
    let (mut full, full_keys) = chain(&[0.0, 1.0, 2.0])?;
    let mut solver = lm_solver(50);
    let reference = solver.optimize(&mut full)?;
    let truth: Vec<f64> = full_keys
        .iter()
        .map(|k| reference.parameters[*k].as_param_slice()[0])
        .collect();

    let (mut problem, keys) = chain(&[0.0, 1.0, 2.0])?;
    problem.set_jacobian_evaluation(Eval::FirstEstimate);
    let marginalizer = Marginalizer::new();

    let applied = marginalizer.apply(&mut problem, &[keys[0]])?;
    assert_eq!(applied.kept, vec![keys[1]]);
    for key in &applied.kept {
        problem.freeze_linearization_point(*key)?;
    }

    for round in 0..10 {
        let mut solver = lm_solver(50);
        let result = solver.optimize(&mut problem)?;
        problem.update_values_from(&result.parameters)?;
        assert!(
            problem.is_linearization_point_frozen(keys[1]),
            "round {round}: the freeze was lost"
        );

        // Re-marginalizing the same set is a no-op on the variable set — the
        // point is that the recursive absorb of the previous prior stays sane.
        let marginal = marginalizer.compute(&problem, &[keys[1]])?;
        assert_eq!(marginal.kept, vec![keys[2]]);
        let information = &marginal.information;
        let asymmetry = (information - information.transpose()).abs().max();
        assert!(
            asymmetry < 1e-9,
            "round {round}: Lambda_p is not symmetric ({asymmetry:.3e})"
        );
        assert!(
            information.iter().all(|v| v.is_finite()),
            "round {round}: Lambda_p went non-finite"
        );
        assert!(marginal.rank <= marginal.dim());
    }

    for (index, expected) in truth.iter().enumerate().skip(1) {
        let got = problem
            .variable_params(keys[index])
            .and_then(|p| p.first().copied())
            .ok_or("missing variable")?;
        assert!(
            (got - expected).abs() < 1e-6,
            "variable {index} drifted to {got} from {expected}"
        );
    }
    Ok(())
}
