//! # 6 · Linear solvers — dense or sparse, Cholesky or QR
//!
//! Every optimizer iteration solves one linear system. Two independent
//! choices decide how:
//!
//! - **Assembly** ([`JacobianMode`], chosen when the [`Problem`] is created):
//!   `Dense` stores the Jacobian as one full matrix; `Sparse` stores only the
//!   non-zero blocks and reuses the sparsity pattern every iteration.
//! - **Factorization** ([`LinearSolverType`], in the optimizer config):
//!   Cholesky of the normal equations `JᵀJ` (fast) or QR of `J` itself
//!   (slower, but tolerant of a rank-deficient system).
//!
//! The example times all four combinations on the square-lap pose graph of
//! example 2 at three sizes. All four reach the same cost: the choice changes
//! the time, never the answer.
//!
//! What decides is **sparsity, not size**. Each pose touches only its
//! neighbours, so the sparse path wins even at 123 unknowns, and dense
//! Cholesky grows cubically (seconds at 3 000 unknowns, where sparse takes
//! milliseconds). Dense assembly pays off only when a problem is small *and*
//! densely coupled — a handful of variables that every residual touches, such
//! as the curve fit of example 1 or a single-camera pose refinement. QR costs
//! a few times Cholesky; reach for it when the normal equations are
//! rank-deficient or badly conditioned, not by default.
//!
//! The Schur-complement solvers (`ExplicitSparseSchur`, `ImplicitSparseSchur`,
//! `ExplicitDenseSchur`) are for problems with many landmarks; they are shown
//! in the bundle-adjustment examples.
//!
//! ```bash
//! cargo run --release --example linear_solvers
//! ```

use std::io::Write;
use std::time::Instant;

use apex_solver::core::VarKey;
use apex_solver::core::noise::NoiseModel;
use apex_solver::core::problem::Problem;
use apex_solver::factors::pose::{BetweenFactor, PriorFactor};
use apex_solver::{
    JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig, LieGroup, LinearSolverType,
    ManifoldType, SE2,
};
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const SIDE: usize = 10;
const ODOMETRY_SIGMAS: [f64; 3] = [0.05, 0.05, 0.02];
const LOOP_SIGMAS: [f64; 3] = [0.02, 0.02, 0.005];

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn gaussian(rng: &mut StdRng, sigma: f64) -> f64 {
    let u1: f64 = rng.random_range(f64::EPSILON..1.0);
    let u2: f64 = rng.random();
    sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn perturb(rng: &mut StdRng, motion: &SE2, sigmas: &[f64; 3]) -> SE2 {
    SE2::from_xy_angle(
        motion.x() + gaussian(rng, sigmas[0]),
        motion.y() + gaussian(rng, sigmas[1]),
        motion.angle() + gaussian(rng, sigmas[2]),
    )
}

/// `laps` laps of the square: the initial guess and every measurement.
struct Graph {
    initial: Vec<SE2>,
    edges: Vec<(usize, usize, SE2, [f64; 3])>,
}

fn simulate(laps: usize, rng: &mut StdRng) -> Graph {
    let steps = 4 * SIDE * laps;
    let mut truth = vec![SE2::identity()];
    for step in 1..=steps {
        let turn = if step % SIDE == 0 {
            std::f64::consts::FRAC_PI_2
        } else {
            0.0
        };
        let previous = truth[step - 1].clone();
        truth.push(previous.compose(&SE2::from_xy_angle(1.0, 0.0, turn), None, None));
    }
    let relative = |a: usize, b: usize| truth[a].between(&truth[b], None, None);
    let mut edges = Vec::new();
    let mut initial = vec![SE2::identity()];
    for i in 0..steps {
        let measured = perturb(rng, &relative(i, i + 1), &ODOMETRY_SIGMAS);
        let last = initial[i].clone();
        initial.push(last.compose(&measured, None, None));
        edges.push((i, i + 1, measured, ODOMETRY_SIGMAS));
    }
    for i in (4 * SIDE..=steps).step_by(3) {
        let measured = perturb(rng, &relative(i - 4 * SIDE, i), &LOOP_SIGMAS);
        edges.push((i - 4 * SIDE, i, measured, LOOP_SIGMAS));
    }
    Graph { initial, edges }
}

fn build(graph: &Graph, mode: JacobianMode) -> Result<Problem> {
    let mut problem = Problem::new(mode);
    let keys: Vec<VarKey> = graph
        .initial
        .iter()
        .map(|pose| {
            problem.add_variable(
                ManifoldType::SE2,
                DVector::from_column_slice(pose.as_param_slice()),
            )
        })
        .collect();
    problem.add_residual_block_with_noise(
        &[keys[0]],
        Box::new(PriorFactor::new(SE2::identity())),
        None,
        NoiseModel::from_sigmas(&[1e-3, 1e-3, 1e-4])?,
    )?;
    for (i, j, measured, sigmas) in &graph.edges {
        problem.add_residual_block_with_noise(
            &[keys[*i], keys[*j]],
            Box::new(BetweenFactor::new(measured.clone())),
            None,
            NoiseModel::from_sigmas(sigmas)?,
        )?;
    }
    Ok(problem)
}

fn main() -> Result<()> {
    let mut out = std::io::stdout().lock();
    let mut rng = StdRng::seed_from_u64(1);

    // The assembly mode must match the factorization family: a dense solver
    // needs a dense Jacobian and vice versa.
    let combinations = [
        (
            "dense  / Cholesky",
            JacobianMode::Dense,
            LinearSolverType::DenseCholesky,
        ),
        (
            "dense  / QR",
            JacobianMode::Dense,
            LinearSolverType::DenseQR,
        ),
        (
            "sparse / Cholesky",
            JacobianMode::Sparse,
            LinearSolverType::SparseCholesky,
        ),
        (
            "sparse / QR",
            JacobianMode::Sparse,
            LinearSolverType::SparseQR,
        ),
    ];

    for laps in [1, 5, 25] {
        let graph = simulate(laps, &mut rng);
        let unknowns = 3 * graph.initial.len();
        writeln!(
            out,
            "\n{laps} lap(s): {} poses, {unknowns} unknowns",
            graph.initial.len()
        )?;
        writeln!(
            out,
            "{:<18} {:>12} {:>6} {:>10}",
            "assembly / solve", "final cost", "iters", "time [ms]"
        )?;
        for (label, mode, solver) in combinations {
            // Dense QR of a 4 000 × 3 000 Jacobian is the slow corner of the
            // table; skip it at the largest size to keep the example quick.
            if laps == 25 && solver == LinearSolverType::DenseQR {
                writeln!(out, "{label:<18} {:>12}", "(skipped)")?;
                continue;
            }
            let mut problem = build(&graph, mode)?;
            let config =
                LevenbergMarquardtConfig::for_2d_pose_graph().with_linear_solver_type(solver);
            let start = Instant::now();
            let result = LevenbergMarquardt::with_config(config).optimize(&mut problem)?;
            writeln!(
                out,
                "{label:<18} {:>12.4} {:>6} {:>10.1}",
                result.final_cost,
                result.iterations,
                1e3 * start.elapsed().as_secs_f64()
            )?;
        }
    }
    Ok(())
}
