//! Noise-model semantics: whitening math, robust-loss composition, FD
//! consistency of the whitened Jacobian, and Null bit-identity.

use apex_solver::ManifoldType;
use apex_solver::core::loss_functions::{HuberLoss, LossFunction};
use apex_solver::core::noise::NoiseModel;
use apex_solver::core::problem::Problem;
use apex_solver::factors::{EuclideanPriorFactor, Factor};
use apex_solver::linalg::JacobianMode;
use faer::prelude::ReborrowMut;
use nalgebra::{DMatrix, DVector, dvector};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A 3-row linear factor: r = x − target, identity Jacobian — so whitening
/// and cost are analytically checkable by hand.
struct Linear3 {
    target: DVector<f64>,
}

impl Factor for Linear3 {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        for i in 0..3 {
            residual[i] = params[0][i] - self.target[i];
        }
        if let Some(mut jac) = jacobian {
            for i in 0..3 {
                *jac.rb_mut().get_mut(i, i) = 1.0;
            }
        }
    }
    fn residual_dim(&self) -> usize {
        3
    }
    fn jacobian_shape(&self) -> (usize, usize) {
        (3, 3)
    }
}

/// Build the Linear3 problem with a noise model and return the INITIAL cost —
/// the objective the optimizer starts from, evaluated on the whitened system.
fn initial_cost(
    noise: NoiseModel,
    loss: Option<HuberLoss>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let x = problem.add_variable(ManifoldType::RN, dvector![1.0, 2.0, 3.0]);
    let loss_boxed = loss.map(|l| Box::new(l) as Box<dyn LossFunction + Send>);
    problem.add_residual_block_with_noise(
        &[x],
        Box::new(Linear3 {
            target: DVector::from_vec(vec![0.0, 0.0, 0.0]),
        }),
        loss_boxed,
        noise,
    );
    // One iteration with a tiny step cannot change the initial cost, which the
    // result reports directly — the cleanest public accessor for the objective.
    let config = apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig::new()
        .with_max_iterations(1);
    let mut solver =
        apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&mut problem)?;
    Ok(result.initial_cost)
}

#[test]
fn diagonal_noise_whitens_cost() -> TestResult {
    // r = (1,2,3); sigmas (1, 0.5, 2) → whitened (1, 4, 1.5);
    // cost = 0.5·(1 + 16 + 2.25) = 9.625.
    let cost = initial_cost(NoiseModel::from_sigmas(&[1.0, 0.5, 2.0])?, None)?;
    assert!((cost - 9.625).abs() < 1e-12, "cost = {cost}");
    Ok(())
}

#[test]
fn dense_noise_from_information_whitens_cost() -> TestResult {
    let info = DMatrix::from_column_slice(2, 2, &[4.0, 1.0, 1.0, 9.0]);
    let noise = NoiseModel::from_information(info.clone())?;

    // SᵀS must reproduce Ω.
    let s = match &noise {
        NoiseModel::Dense(m) => m.clone(),
        _ => return Err("expected dense model".into()),
    };
    assert!((s.transpose() * &s - info).norm() < 1e-12, "SᵀS != Ω");

    // 2-row factor: r = (1,2); cost = ½·‖S·r‖².
    let mut problem = Problem::new(JacobianMode::Sparse);
    let x = problem.add_variable(ManifoldType::RN, dvector![1.0, 2.0]);
    problem.add_residual_block_with_noise(
        &[x],
        Box::new(EuclideanPriorFactor::new(DVector::from_vec(vec![0.0, 0.0]))),
        None,
        noise,
    );
    let config = apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig::new()
        .with_max_iterations(1);
    let mut solver =
        apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardt::with_config(config);
    let result = solver.optimize(&mut problem)?;

    let whitened = &s * DVector::from_vec(vec![1.0, 2.0]);
    let expected = 0.5 * whitened.norm_squared();
    assert!(
        (result.initial_cost - expected).abs() < 1e-12,
        "cost = {} vs {expected}",
        result.initial_cost
    );
    Ok(())
}

#[test]
fn null_noise_is_bit_identical_to_unweighted_path() -> TestResult {
    let build = |use_with_noise: bool| {
        let mut problem = Problem::new(JacobianMode::Sparse);
        let x = problem.add_variable(ManifoldType::RN, dvector![1.0, 2.0, 3.0]);
        let factor: Box<dyn Factor + Send> = Box::new(Linear3 {
            target: DVector::from_vec(vec![0.5, -1.0, 2.5]),
        });
        if use_with_noise {
            problem.add_residual_block_with_noise(&[x], factor, None, NoiseModel::Null);
        } else {
            problem.add_residual_block(&[x], factor, None);
        }
        let config = apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardtConfig::new()
            .with_max_iterations(1);
        let mut solver =
            apex_solver::optimizer::levenberg_marquardt::LevenbergMarquardt::with_config(config);
        solver
            .optimize(&mut problem)
            .unwrap_or_else(|e| panic!("solve failed: {e:?}"))
    };
    let with = build(true);
    let without = build(false);
    assert_eq!(with.initial_cost.to_bits(), without.initial_cost.to_bits());
    assert_eq!(with.final_cost.to_bits(), without.final_cost.to_bits());
    assert_eq!(with.iterations, without.iterations);
    Ok(())
}

#[test]
fn noise_and_robust_loss_compose() -> TestResult {
    // Huber(δ=1): whitened s = ‖S·r‖² = 1 + 16 + 2.25 = 19.25 > 1 →
    // ρ(s) = 2√s − 1; cost = ½·ρ.
    let cost = initial_cost(
        NoiseModel::from_sigmas(&[1.0, 0.5, 2.0])?,
        Some(HuberLoss::new(1.0)?),
    )?;
    let s = 19.25_f64;
    let expected = 0.5 * (2.0 * s.sqrt() - 1.0);
    assert!(
        (cost - expected).abs() < 1e-12,
        "cost = {cost} vs {expected}"
    );
    Ok(())
}

#[test]
fn whitened_jacobian_matches_whitened_fd() -> TestResult {
    // FD of the whitened residual of Linear3 wrt x must reproduce S (S·I).
    let noise = NoiseModel::from_sigmas(&[1.0, 0.5, 2.0])?;
    let s_diag = [1.0, 2.0, 0.5]; // 1/σ
    let eps = 1e-6;
    let target = DVector::from_vec(vec![0.0, 0.0, 0.0]);

    let mut problem = Problem::new(JacobianMode::Sparse);
    let x = problem.add_variable(ManifoldType::RN, dvector![1.0, 2.0, 3.0]);
    problem.add_residual_block_with_noise(
        &[x],
        Box::new(Linear3 {
            target: target.clone(),
        }),
        None,
        noise.clone(),
    );
    // Production initialization path (public): gives variables, index map,
    // symbolic structure and the assembly workspace.
    let mut state = apex_solver::optimizer::initialize_optimization_state(&mut problem)?;
    let symbolic = match state.symbolic_structure.as_ref() {
        Some(symbolic) => symbolic,
        None => return Err("sparse mode must produce a symbolic structure".into()),
    };
    let (_r, jacobian) = apex_solver::linearizer::cpu::sparse::assemble_sparse(
        &problem,
        &state.variables,
        &state.variable_index_map,
        symbolic,
        &mut state.workspace,
    )?;

    for k in 0..3 {
        let mut xp = state.variables.clone();
        let mut xm = state.variables.clone();
        let mut step_p = dvector![0.0, 0.0, 0.0];
        step_p[k] = eps;
        let mut step_m = dvector![0.0, 0.0, 0.0];
        step_m[k] = -eps;
        xp[x].apply_tangent_step(step_p.as_slice());
        xm[x].apply_tangent_step(step_m.as_slice());

        let mut rp = vec![0.0; 3];
        let mut rm = vec![0.0; 3];
        let factor = Linear3 {
            target: target.clone(),
        };
        factor.linearize(&[xp[x].as_param_slice()], &mut rp, None);
        factor.linearize(&[xm[x].as_param_slice()], &mut rm, None);
        noise.whiten_residual(&mut rp);
        noise.whiten_residual(&mut rm);

        for i in 0..3 {
            let fd = (rp[i] - rm[i]) / (2.0 * eps);
            let analytic = jacobian.as_ref().val_of_col(k)[i];
            assert!(
                (fd - analytic).abs() < 1e-6,
                "whitened J[{i}][{k}]: FD {fd} vs {analytic}"
            );
            let expected = if i == k { s_diag[i] } else { 0.0 };
            assert!(
                (analytic - expected).abs() < 1e-12,
                "whitened J[{i}][{k}] = {analytic}, expected {expected}"
            );
        }
    }
    Ok(())
}

#[test]
fn dimension_mismatch_is_rejected() -> TestResult {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let x = problem.add_variable(ManifoldType::RN, dvector![1.0, 2.0, 3.0]);
    let err = problem
        .try_add_residual_block_with_noise(
            &[x],
            Box::new(Linear3 {
                target: DVector::from_vec(vec![0.0; 3]),
            }),
            None,
            NoiseModel::from_sigmas(&[1.0, 2.0])?,
        )
        .err()
        .ok_or("dimension mismatch must be rejected")?;
    assert!(
        matches!(err, apex_solver::core::CoreError::DimensionMismatch(_)),
        "{err}"
    );
    Ok(())
}

#[test]
fn non_positive_information_is_rejected() {
    assert!(NoiseModel::from_sigmas(&[1.0, 0.0]).is_err());
    assert!(NoiseModel::from_sigmas(&[-1.0]).is_err());
    let singular = DMatrix::zeros(2, 2);
    assert!(NoiseModel::from_information(singular).is_err());
}
