//! # 5 · Optimizers — Gauss–Newton, Levenberg–Marquardt, Dog Leg
//!
//! All three optimizers solve the same nonlinear least-squares problem by
//! repeatedly linearizing it; they differ in how they **control the step**:
//!
//! | optimizer | step control | use it when |
//! |---|---|---|
//! | Gauss–Newton | none: the full Newton step | the start is close and the problem well conditioned |
//! | Levenberg–Marquardt | damping `λ` blends Newton and gradient descent | the default; robust from a poor start |
//! | Dog Leg | trust region between the Cauchy point and the Newton step | similar to LM, often fewer linear solves |
//!
//! The example loads a real pose graph (`M3500` 2-D by default, downloaded on
//! first use), runs each optimizer behind the common
//! [`Optimizer`] trait, and records the cost of every iteration with a small
//! custom [`OptObserver`]. Observers are how you hook logging, plotting or
//! early diagnostics into a solve without touching the optimizer.
//!
//! ```bash
//! cargo run --release --example optimizers              # M3500 (2-D)
//! cargo run --release --example optimizers sphere2500   # 3-D
//! cargo run --release --example optimizers intel        # a hard start
//! ```
//!
//! On `M3500` and `sphere2500` all five converge to the same cost. `intel` is
//! instructive the other way round: its initial guess is so poor that the
//! damped methods creep towards the optimum for hundreds of iterations, while
//! Gauss–Newton's undamped step lands in five. LM is the safe default, not a
//! universally faster one; the presets (`for_2d_pose_graph`,
//! `for_large_3d_pose_graph`, `for_bundle_adjustment`) are the first thing to
//! try when it is slow.

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

use apex_solver::core::VarKey;
use apex_solver::core::noise::NoiseModel;
use apex_solver::core::problem::Problem;
use apex_solver::core::variable::ManifoldVariable;
use apex_solver::factors::pose::BetweenFactor;
use apex_solver::observers::OptObserver;
use apex_solver::optimizer::dog_leg::DogLegConfig;
use apex_solver::optimizer::gauss_newton::GaussNewtonConfig;
use apex_solver::{
    DampingUpdate, DogLeg, GaussNewton, JacobianMode, LevenbergMarquardt, LevenbergMarquardtConfig,
    LieGroup, ManifoldType, Optimizer,
};
use nalgebra::{DMatrix, DVector};
use slotmap::SlotMap;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Records the cost after every iteration.
///
/// Observers receive `&self`, because the optimizer may call them from inside
/// its loop; state therefore lives behind a `Mutex`, shared through an `Arc`
/// so the caller can read it after the solve.
#[derive(Clone, Default)]
struct CostTrace(Arc<Mutex<Vec<f64>>>);

impl OptObserver for CostTrace {
    fn on_step(&self, _values: &SlotMap<VarKey, Box<dyn ManifoldVariable>>, _iteration: usize) {}

    fn set_iteration_metrics(
        &self,
        cost: f64,
        _gradient_norm: f64,
        _damping: Option<f64>,
        _step_norm: f64,
        _step_quality: Option<f64>,
    ) {
        if let Ok(mut costs) = self.0.lock() {
            costs.push(cost);
        }
    }
}

impl CostTrace {
    fn costs(&self) -> Vec<f64> {
        self.0.lock().map(|costs| costs.clone()).unwrap_or_default()
    }
}

/// Load a g2o pose graph into a [`Problem`]: one variable per vertex, one
/// [`BetweenFactor`] per edge, weighted by the edge's information matrix.
fn load(name: &str) -> Result<Problem> {
    // Looks in `data/odometry/` and downloads the file on first use.
    let path = apex_solver::apex_io::ensure_odometry_dataset(name)?;
    let graph = apex_solver::apex_io::load_graph(&path)?;
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut keys: HashMap<usize, VarKey> = HashMap::new();

    if !graph.vertices_se2.is_empty() {
        for (id, vertex) in &graph.vertices_se2 {
            let params = DVector::from_column_slice(vertex.pose.as_param_slice());
            keys.insert(*id, problem.add_variable(ManifoldType::SE2, params));
        }
        for edge in &graph.edges_se2 {
            let noise = NoiseModel::from_information(DMatrix::from_column_slice(
                3,
                3,
                edge.information.as_slice(),
            ))?;
            let (from, to) = (keys[&edge.from], keys[&edge.to]);
            problem.add_residual_block_with_noise(
                &[from, to],
                Box::new(BetweenFactor::new(edge.measurement.clone())),
                None,
                noise,
            )?;
        }
    } else {
        for (id, vertex) in &graph.vertices_se3 {
            let params = DVector::from_column_slice(vertex.pose.as_param_slice());
            keys.insert(*id, problem.add_variable(ManifoldType::SE3, params));
        }
        for edge in &graph.edges_se3 {
            let noise = NoiseModel::from_information(DMatrix::from_column_slice(
                6,
                6,
                edge.information.as_slice(),
            ))?;
            let (from, to) = (keys[&edge.from], keys[&edge.to]);
            problem.add_residual_block_with_noise(
                &[from, to],
                Box::new(BetweenFactor::new(edge.measurement.clone())),
                None,
                noise,
            )?;
        }
    }

    // Gauge: hold the first vertex fixed, as g2o does.
    let first = keys
        .keys()
        .min()
        .copied()
        .ok_or("the graph has no vertices")?;
    let dof = if graph.vertices_se2.is_empty() { 6 } else { 3 };
    for coordinate in 0..dof {
        problem.fix_variable(keys[&first], coordinate)?;
    }
    Ok(problem)
}

fn main() -> Result<()> {
    let mut out = std::io::stdout().lock();
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "M3500".to_owned());
    writeln!(out, "dataset {name}\n")?;

    // Same budget and stopping rule for everyone, so the table compares the
    // step strategies and nothing else.
    let iterations = 100;
    let tolerance = 1e-8;

    let mut runs: Vec<(&str, Box<dyn Optimizer>, CostTrace)> = Vec::new();

    let trace = CostTrace::default();
    let mut gn = GaussNewton::with_config(
        GaussNewtonConfig::new()
            .with_max_iterations(iterations)
            .with_cost_tolerance(tolerance),
    );
    gn.add_observer(trace.clone());
    runs.push(("Gauss-Newton", Box::new(gn), trace));

    // Levenberg–Marquardt has the most knobs. Three variants:
    // - the generic default: damping scaled by the Hessian diagonal,
    // - the pose-graph preset: uniform `λ·I` damping (diagonal bounds 1..1),
    //   which pose graphs with mixed translation/rotation scales prefer,
    // - the preset with Marquardt's classic three-band λ update instead of
    //   Nielsen's smooth rule.
    let pose_graph = if name == "sphere2500" || name == "parking-garage" {
        LevenbergMarquardtConfig::for_large_3d_pose_graph()
    } else {
        LevenbergMarquardtConfig::for_2d_pose_graph()
    };
    for (label, config) in [
        ("LM (generic default)", LevenbergMarquardtConfig::new()),
        ("LM (pose-graph preset)", pose_graph.clone()),
        (
            "LM (preset, Marquardt)",
            pose_graph.with_damping_update(DampingUpdate::Marquardt),
        ),
    ] {
        let trace = CostTrace::default();
        let mut lm = LevenbergMarquardt::with_config(
            config
                .with_max_iterations(iterations)
                .with_cost_tolerance(tolerance),
        );
        lm.add_observer(trace.clone());
        runs.push((label, Box::new(lm), trace));
    }

    let trace = CostTrace::default();
    let mut dog_leg = DogLeg::with_config(
        DogLegConfig::new()
            .with_max_iterations(iterations)
            .with_cost_tolerance(tolerance),
    );
    dog_leg.add_observer(trace.clone());
    runs.push(("Dog Leg", Box::new(dog_leg), trace));

    writeln!(
        out,
        "{:<22} {:>14} {:>14} {:>6} {:>9}  status",
        "optimizer", "initial cost", "final cost", "iters", "time [s]"
    )?;
    let mut traces = Vec::new();
    for (label, mut optimizer, trace) in runs {
        // A fresh problem per run: every optimizer starts from the same point.
        let mut problem = load(&name)?;
        let result = optimizer.optimize(&mut problem)?;
        writeln!(
            out,
            "{label:<22} {:>14.6e} {:>14.6e} {:>6} {:>9.3}  {}",
            result.initial_cost,
            result.final_cost,
            result.iterations,
            result.elapsed_time.as_secs_f64(),
            result.status
        )?;
        traces.push((label, trace.costs()));
    }

    writeln!(out, "\ncost per iteration (from the observer)")?;
    for (label, costs) in traces {
        let shown: Vec<String> = costs
            .iter()
            .take(8)
            .map(|cost| format!("{cost:.3e}"))
            .collect();
        let more = if costs.len() > 8 { " …" } else { "" };
        writeln!(out, "{label:<22} {}{more}", shown.join("  "))?;
    }
    Ok(())
}
