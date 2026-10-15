//! Criterion micro-benchmarks for the sliding-window estimator's kernels:
//!
//! - `imu/preintegrate`, `imu/combined_linearize`: one interval's
//!   preintegration, and the 15-row factor evaluated with its Jacobian
//! - `lidar/map_plane_linearize`, `lidar/aggregate_normal_equations`: one
//!   point-to-plane row on `SE₂(3)` with an extrinsic, and 512 of them
//!   compressed into their exact normal equations
//! - `window/lm_three_iterations`: a fixed-lag chain of navigation states
//!   and biases linked by IMU factors, three Levenberg-Marquardt steps
//! - `marginalize/oldest_state_and_bias`: the Schur-style elimination of the
//!   oldest state into a prior over the rest of the window
//!
//! ```bash
//! cargo bench --bench window_kernels
//! ```
//!
//! The window is synthetic and dataset-free: constant body rates and a
//! constant specific force, so every run times the same arithmetic.

use std::hint::black_box;
use std::sync::Arc;

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::se23::SE23;
use apex_solver::JacobianMode;
use apex_solver::ManifoldType;
use apex_solver::core::VarKey;
use apex_solver::core::marginalization::Marginalizer;
use apex_solver::core::noise::NoiseModel;
use apex_solver::core::problem::Problem;
use apex_solver::factors::Factor;
use apex_solver::factors::aggregate::{AggregateFactor, AggregatePart};
use apex_solver::factors::imu::{
    ImuMeasurement, ImuParameters, ImuPreintegration, ImuSensorReadings, SpeedAndBias, se23,
};
use apex_solver::factors::lidar::{LidarMapPlaneFactor, Plane};
use apex_solver::factors::pose::{EuclideanPriorFactor, PriorFactor};
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use nalgebra::{DVector, Vector3};

type SetupResult<T> = Result<T, Box<dyn std::error::Error>>;

const IMU_DT: f64 = 0.005;
const INTERVAL: f64 = 0.25;
const WINDOW_STATES: usize = 8;

fn imu_parameters() -> ImuParameters {
    ImuParameters {
        sigma_g_c: 1.6968e-4,
        sigma_a_c: 2.0e-3,
        sigma_gw_c: 1.9393e-5,
        sigma_aw_c: 3.0e-3,
        g: 9.81,
        ..ImuParameters::default()
    }
}

/// One interval's samples at 200 Hz: constant rates, constant specific force.
fn samples(t0: f64) -> Vec<ImuMeasurement> {
    let count = (INTERVAL / IMU_DT).round() as usize;
    (0..=count)
        .map(|i| {
            ImuMeasurement::new(
                t0 + i as f64 * IMU_DT,
                ImuSensorReadings {
                    gyroscopes: Vector3::new(0.12, -0.05, 0.08),
                    accelerometers: Vector3::new(0.4, -0.2, 9.81),
                },
            )
        })
        .collect()
}

fn preintegration(t0: f64) -> ImuPreintegration {
    ImuPreintegration::new(
        Arc::from(samples(t0)),
        imu_parameters(),
        t0,
        t0 + INTERVAL,
        &SpeedAndBias::zeros(),
    )
}

fn sigmas(count: usize, sigma: f64) -> SetupResult<NoiseModel> {
    Ok(NoiseModel::from_sigmas(&vec![sigma; count])?)
}

/// A window of `WINDOW_STATES` navigation states and biases chained by
/// combined IMU factors, anchored at the start and weakly at every state.
fn window() -> SetupResult<(Problem, Vec<VarKey>, Vec<VarKey>)> {
    let mut problem = Problem::new(JacobianMode::Sparse);
    let mut states = Vec::new();
    let mut biases = Vec::new();
    for _ in 0..WINDOW_STATES {
        let state = SE23::identity();
        states.push(problem.add_variable(
            ManifoldType::SE23,
            DVector::from_column_slice(state.as_param_slice()),
        ));
        biases.push(problem.add_variable(ManifoldType::RN, DVector::zeros(6)));
    }
    problem.add_residual_block_with_noise(
        &[states[0]],
        Box::new(PriorFactor::new(SE23::identity())),
        None,
        sigmas(9, 1e-4)?,
    );
    problem.add_residual_block_with_noise(
        &[biases[0]],
        Box::new(EuclideanPriorFactor::new(DVector::zeros(6))),
        None,
        sigmas(6, 0.05)?,
    );
    for k in 0..WINDOW_STATES - 1 {
        let preint = preintegration(k as f64 * INTERVAL);
        problem.add_residual_block(
            &[states[k], biases[k], states[k + 1], biases[k + 1]],
            Box::new(se23::CombinedImuFactor::new(preint)),
            None,
        );
    }
    for state in &states {
        problem.add_residual_block_with_noise(
            &[*state],
            Box::new(PriorFactor::new(SE23::identity())),
            None,
            sigmas(9, 1.0)?,
        );
    }
    Ok((problem, states, biases))
}

fn bench_imu(c: &mut Criterion) -> SetupResult<()> {
    let measurements = samples(0.0);
    let params = imu_parameters();
    let sb = SpeedAndBias::zeros();
    let mut group = c.benchmark_group("imu");
    group.bench_function("preintegrate_50_samples", |b| {
        b.iter(|| {
            ImuPreintegration::new(
                Arc::from(black_box(measurements.clone())),
                params.clone(),
                0.0,
                INTERVAL,
                &sb,
            )
        });
    });

    let factor = se23::CombinedImuFactor::new(preintegration(0.0));
    let state = SE23::identity();
    let bias = [0.0f64; 6];
    let (rows, cols) = factor.jacobian_shape();
    let mut residual = vec![0.0; rows];
    let mut buffer = vec![0.0; rows * cols];
    group.bench_function("combined_linearize", |b| {
        b.iter(|| {
            buffer.fill(0.0);
            let jacobian = faer::mat::MatMut::from_column_major_slice_mut(&mut buffer, rows, cols);
            factor.linearize(
                &[
                    black_box(state.as_param_slice()),
                    &bias,
                    state.as_param_slice(),
                    &bias,
                ],
                &mut residual,
                Some(jacobian),
            );
        });
    });
    group.finish();
    Ok(())
}

fn bench_lidar(c: &mut Criterion) -> SetupResult<()> {
    let plane = Plane::new(Vector3::new(0.0, 0.0, 1.0), -0.5)?;
    let state = SE23::identity();
    let extrinsic = SE3::identity();
    let point = |i: usize| {
        let a = i as f64 * 0.37;
        Vector3::new(a.cos() * 4.0, a.sin() * 4.0, 0.5 + 0.01 * (i % 7) as f64)
    };

    let one = LidarMapPlaneFactor::<SE23>::new(point(0), plane.clone(), 0.05)?;
    let (rows, cols) = one.jacobian_shape();
    let mut residual = vec![0.0; rows];
    let mut buffer = vec![0.0; rows * cols];
    let mut group = c.benchmark_group("lidar");
    group.bench_function("map_plane_linearize", |b| {
        b.iter(|| {
            buffer.fill(0.0);
            let jacobian = faer::mat::MatMut::from_column_major_slice_mut(&mut buffer, rows, cols);
            one.linearize(
                &[
                    black_box(state.as_param_slice()),
                    extrinsic.as_param_slice(),
                ],
                &mut residual,
                Some(jacobian),
            );
        });
    });

    let mut parts = Vec::new();
    for i in 0..512 {
        parts.push(AggregatePart {
            factor: Box::new(LidarMapPlaneFactor::<SE23>::new(
                point(i),
                plane.clone(),
                0.05,
            )?),
            blocks: vec![0, 1],
        });
    }
    let aggregate = AggregateFactor::new(parts, vec![9, 6])?;
    let (rows, cols) = aggregate.jacobian_shape();
    let mut residual = vec![0.0; rows];
    let mut buffer = vec![0.0; rows * cols];
    group.sample_size(30);
    group.bench_function("aggregate_512_points", |b| {
        b.iter(|| {
            buffer.fill(0.0);
            let jacobian = faer::mat::MatMut::from_column_major_slice_mut(&mut buffer, rows, cols);
            aggregate.linearize(
                &[
                    black_box(state.as_param_slice()),
                    extrinsic.as_param_slice(),
                ],
                &mut residual,
                Some(jacobian),
            );
        });
    });
    group.finish();
    Ok(())
}

fn bench_window(c: &mut Criterion) -> SetupResult<()> {
    let mut group = c.benchmark_group("window");
    group.sample_size(20);
    group.bench_function("lm_three_iterations_8_states", |b| {
        b.iter_batched(
            || window().map(|(problem, _, _)| problem),
            |problem| {
                let Ok(mut problem) = problem else {
                    return;
                };
                let config = LevenbergMarquardtConfig::new().with_max_iterations(3);
                let mut solver = LevenbergMarquardt::with_config(config);
                let _ = black_box(solver.optimize(&mut problem));
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();

    let (problem, states, biases) = window()?;
    let mut group = c.benchmark_group("marginalize");
    group.sample_size(20);
    group.bench_function("oldest_state_and_bias", |b| {
        b.iter(|| {
            let marginalizer = Marginalizer::new();
            let _ = black_box(marginalizer.compute(&problem, &[states[0], biases[0]]));
        });
    });
    group.finish();
    Ok(())
}

fn benches(c: &mut Criterion) {
    for result in [bench_imu(c), bench_lidar(c), bench_window(c)] {
        if let Err(error) = result {
            tracing::error!("window_kernels: setup failed: {error}");
        }
    }
}

criterion_group!(window_kernels, benches);
criterion_main!(window_kernels);
