//! # 10 · Incremental estimation — a graph that grows while you solve it
//!
//! A robot does not hand you its whole trajectory at once. Poses arrive one by
//! one, and an estimate is wanted *now*, not at the end. This example streams
//! the `M3500` pose graph pose by pose and keeps **one** [`Problem`] alive:
//!
//! 1. a new pose is predicted from the latest estimate of the previous pose
//!    composed with the new odometry measurement;
//! 2. every edge whose two ends now exist is added (odometry, and loop
//!    closures back to earlier poses);
//! 3. every `SOLVE_EVERY` poses the problem is re-optimized, starting from the
//!    current estimate, with a small iteration budget;
//! 4. the result is **written back** with [`Problem::update_values_from`].
//!
//! Step 4 is the one that is easy to forget: `optimize` returns its answer
//! and leaves the problem untouched, so without the write-back every solve
//! would restart from the predictions.
//!
//! At the end the streamed estimate is compared with a batch solve of the
//! complete graph from the dataset's initial guess. Both should reach the
//! same cost; the incremental one has had an estimate available all along.
//! The cost of a solve grows with the graph, which is what the windowed
//! estimators of examples 11 and 12 avoid.
//!
//! ```bash
//! cargo run --release --example incremental_pose_graph
//! ```

use std::collections::HashMap;
use std::io::Write;
use std::time::{Duration, Instant};

use apex_solver::apex_io::{EdgeSE2, Graph};
use apex_solver::core::VarKey;
use apex_solver::core::noise::NoiseModel;
use apex_solver::core::problem::Problem;
use apex_solver::factors::pose::BetweenFactor;
use apex_solver::{
    JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig, LieGroup, ManifoldType, SE2,
};
use nalgebra::{DMatrix, DVector};

/// Re-optimize after this many new poses.
const SOLVE_EVERY: usize = 100;
/// Iterations per incremental solve: a warm start needs few.
const ITERATIONS_PER_SOLVE: usize = 10;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn noise(edge: &EdgeSE2) -> Result<NoiseModel> {
    Ok(NoiseModel::from_information(DMatrix::from_column_slice(
        3,
        3,
        edge.information.as_slice(),
    ))?)
}

fn add_edge(problem: &mut Problem, keys: &HashMap<usize, VarKey>, edge: &EdgeSE2) -> Result<()> {
    problem.add_residual_block_with_noise(
        &[keys[&edge.from], keys[&edge.to]],
        Box::new(BetweenFactor::new(edge.measurement.clone())),
        None,
        noise(edge)?,
    )?;
    Ok(())
}

/// The reference: the whole graph at once, from the dataset's initial guess.
fn batch(graph: &Graph, ids: &[usize]) -> Result<(f64, Duration)> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut keys = HashMap::new();
    for id in ids {
        let pose = &graph.vertices_se2[id].pose;
        keys.insert(
            *id,
            problem.add_variable(
                ManifoldType::SE2,
                DVector::from_column_slice(pose.as_param_slice()),
            ),
        );
    }
    for edge in &graph.edges_se2 {
        add_edge(&mut problem, &keys, edge)?;
    }
    for coordinate in 0..3 {
        problem.fix_variable(keys[&ids[0]], coordinate)?;
    }
    let start = Instant::now();
    let result = LevenbergMarquardt::with_config(LevenbergMarquardtConfig::for_2d_pose_graph())
        .optimize(&mut problem)?;
    Ok((result.final_cost, start.elapsed()))
}

fn main() -> Result<()> {
    let mut out = std::io::stdout().lock();
    let path = apex_solver::apex_io::ensure_odometry_dataset("M3500")?;
    let graph = apex_solver::apex_io::load_graph(&path)?;
    let mut ids: Vec<usize> = graph.vertices_se2.keys().copied().collect();
    ids.sort_unstable();

    // Index the edges by the later of their two vertices: an edge becomes
    // usable the moment its younger end arrives.
    let mut arriving: HashMap<usize, Vec<&EdgeSE2>> = HashMap::new();
    for edge in &graph.edges_se2 {
        arriving
            .entry(edge.from.max(edge.to))
            .or_default()
            .push(edge);
    }

    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut keys: HashMap<usize, VarKey> = HashMap::new();
    let config =
        LevenbergMarquardtConfig::for_2d_pose_graph().with_max_iterations(ITERATIONS_PER_SOLVE);
    let mut solve_time = Duration::ZERO;
    let mut last_cost = f64::NAN;

    writeln!(
        out,
        "{:>6} {:>7} {:>14} {:>11}",
        "poses", "edges", "cost", "solve [ms]"
    )?;
    for (count, id) in ids.iter().enumerate() {
        // 1. Predict the new pose from its odometry edge and the previous estimate.
        let odometry = arriving
            .get(id)
            .into_iter()
            .flatten()
            .find(|edge| edge.to == *id && edge.from + 1 == *id);
        let prediction = match odometry {
            Some(odometry) => {
                let previous = SE2::from_param_slice(
                    problem
                        .variable_params(keys[&odometry.from])
                        .ok_or("the previous pose is missing")?,
                );
                previous.compose(&odometry.measurement, None, None)
            }
            None => graph.vertices_se2[id].pose.clone(),
        };
        let key = problem.add_variable(
            ManifoldType::SE2,
            DVector::from_column_slice(prediction.as_param_slice()),
        );
        keys.insert(*id, key);
        if count == 0 {
            // The first pose is the gauge anchor.
            for coordinate in 0..3 {
                problem.fix_variable(key, coordinate)?;
            }
        }

        // 2. Every edge that has both ends now.
        for edge in arriving.get(id).into_iter().flatten() {
            add_edge(&mut problem, &keys, edge)?;
        }

        // 3–4. Re-optimize from the current estimate and write the answer back.
        let last = count + 1 == ids.len();
        if (count + 1) % SOLVE_EVERY == 0 || last {
            let start = Instant::now();
            let result = LevenbergMarquardt::with_config(config.clone()).optimize(&mut problem)?;
            let elapsed = start.elapsed();
            solve_time += elapsed;
            problem.update_values_from(&result.parameters)?;
            last_cost = result.final_cost;
            if (count + 1) % (5 * SOLVE_EVERY) == 0 || last {
                writeln!(
                    out,
                    "{:>6} {:>7} {:>14.3} {:>11.1}",
                    count + 1,
                    problem.num_residual_blocks(),
                    result.final_cost,
                    1e3 * elapsed.as_secs_f64()
                )?;
            }
        }
    }

    // A last solve to convergence, as a batch system would also do.
    let polish = LevenbergMarquardt::with_config(LevenbergMarquardtConfig::for_2d_pose_graph())
        .optimize(&mut problem)?;
    problem.update_values_from(&polish.parameters)?;

    let (batch_cost, batch_time) = batch(&graph, &ids)?;
    writeln!(out)?;
    writeln!(
        out,
        "incremental: cost {last_cost:.3} after streaming, {:.3} after a final solve",
        polish.final_cost
    )?;
    writeln!(
        out,
        "             {} solves, {:.0} ms in total",
        ids.len().div_ceil(SOLVE_EVERY),
        1e3 * solve_time.as_secs_f64()
    )?;
    writeln!(
        out,
        "batch:       cost {batch_cost:.3}, {:.0} ms (one solve, at the end)",
        1e3 * batch_time.as_secs_f64()
    )?;
    if (polish.final_cost - batch_cost).abs() > 1e-3 * batch_cost.max(1.0) {
        writeln!(out, "note: the two solutions are different local minima")?;
    }
    Ok(())
}
