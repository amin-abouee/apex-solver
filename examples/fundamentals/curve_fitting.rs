//! # 1 · Curve fitting — the smallest complete problem
//!
//! Fit `y = exp(m·x + c)` to 60 noisy samples. This is the whole life cycle of
//! an apex-solver problem in one file:
//!
//! 1. write a [`Factor`]: a residual and its analytic Jacobian,
//! 2. create a [`Problem`] and add a variable (`ℝ²` holding `[m, c]`),
//! 3. add one residual block per sample,
//! 4. run [`LevenbergMarquardt`] and read the [`SolverResult`] back.
//!
//! ```bash
//! cargo run --release --example curve_fitting
//! ```
//!
//! Next: `pose_graph_2d` replaces the hand-written factor with the built-in
//! pose factors and introduces noise models and gauge freedom.
//!
//! [`SolverResult`]: apex_solver::optimizer::SolverResult

use std::io::Write;

use apex_solver::core::problem::Problem;
use apex_solver::factors::Factor;
use apex_solver::{JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig, ManifoldType};
use faer::mat::MatMut;
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The curve the samples are drawn from.
const TRUE_M: f64 = 0.3;
const TRUE_C: f64 = 0.1;
/// Standard deviation of the noise added to every `y`.
const NOISE_SIGMA: f64 = 0.2;
const SAMPLES: usize = 60;

/// One sample `(x, y)` of the curve.
///
/// A factor owns its measurement and knows how to turn the current value of
/// its variables into a residual. Here there is one variable, `[m, c]`, so the
/// Jacobian is `1 × 2`.
struct ExponentialSample {
    x: f64,
    y: f64,
}

impl Factor for ExponentialSample {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<MatMut<'_, f64>>,
    ) {
        // `params[i]` is the parameter slice of the i-th variable this block
        // was registered with — here the single `[m, c]`.
        let (m, c) = (params[0][0], params[0][1]);
        let prediction = (m * self.x + c).exp();
        residual[0] = prediction - self.y;

        // The Jacobian is requested only when the optimizer needs it (a cost
        // evaluation skips it). Its columns follow the variables' tangent
        // coordinates, in registration order.
        if let Some(mut jacobian) = jacobian {
            jacobian[(0, 0)] = self.x * prediction; // ∂r/∂m
            jacobian[(0, 1)] = prediction; // ∂r/∂c
        }
    }

    fn residual_dim(&self) -> usize {
        1
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (1, 2)
    }
}

/// A zero-mean Gaussian sample by the Box–Muller transform.
fn gaussian(rng: &mut StdRng, sigma: f64) -> f64 {
    let u1: f64 = rng.random_range(f64::EPSILON..1.0);
    let u2: f64 = rng.random();
    sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = std::io::stdout().lock();
    // A fixed seed: every run sees the same data and prints the same numbers.
    let mut rng = StdRng::seed_from_u64(7);

    // `Sparse` is the right default for almost every problem; `Dense` is for
    // small, fully coupled ones (see the `linear_solvers` example).
    let mut problem = Problem::new(JacobianMode::Sparse);

    // A variable is a point on a manifold. `RN` is plain Euclidean space; the
    // initial value is deliberately far from the answer.
    let curve = problem.add_variable(ManifoldType::RN, DVector::from_vec(vec![0.0, 0.0]));

    for i in 0..SAMPLES {
        let x = 5.0 * i as f64 / SAMPLES as f64;
        let y = (TRUE_M * x + TRUE_C).exp() + gaussian(&mut rng, NOISE_SIGMA);
        // No loss function: plain least squares. Robust losses are the topic
        // of the `robust_losses` example.
        problem.add_residual_block(&[curve], Box::new(ExponentialSample { x, y }), None)?;
    }

    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(50)
        .with_cost_tolerance(1e-12);
    let mut solver = LevenbergMarquardt::with_config(config);

    // `optimize` works on a copy of the variables and returns it; `problem`
    // itself keeps the initial values (examples 11–13 rely on this).
    let result = solver.optimize(&mut problem)?;

    let fitted = result
        .parameters
        .get(curve)
        .ok_or("the curve variable is missing from the result")?
        .as_param_slice();

    writeln!(out, "status          {}", result.status)?;
    writeln!(out, "iterations      {}", result.iterations)?;
    writeln!(
        out,
        "cost            {:.4} → {:.4}",
        result.initial_cost, result.final_cost
    )?;
    writeln!(out, "m               {:.4} (true {TRUE_M})", fitted[0])?;
    writeln!(out, "c               {:.4} (true {TRUE_C})", fitted[1])?;
    // With σ = 0.2 noise on 60 samples the estimate is within a few hundredths.
    let error = ((fitted[0] - TRUE_M).powi(2) + (fitted[1] - TRUE_C).powi(2)).sqrt();
    writeln!(out, "parameter error {error:.4}")?;
    if error > 0.1 {
        return Err(format!("fit did not converge to the generating curve (error {error})").into());
    }
    Ok(())
}
