//! Covariance consistency test for the LOAM `LidarEdgeFactor` / `LidarPlaneFactor`.
//!
//! Builds a synthetic "corner room" (3 orthogonal planes + 2 edges) in a fixed
//! reference frame `T_WA`, and repeatedly estimates an unknown scan pose `T_WB`
//! from noisy point-to-edge / point-to-plane correspondences against known
//! ground truth.
//!
//! The expected covariance of `T_WB` is computed directly from the factors'
//! own analytic Jacobians — `Cov = (sum_i JᵢᵀJᵢ)⁻¹` evaluated at the ground
//! truth linearization point, which is the standard asymptotic covariance for
//! correctly-weighted nonlinear least squares — rather than through the
//! solver's `Problem::compute_and_set_covariances`. That API is known to
//! over-state uncertainty whenever a fixed variable shares a residual with a
//! free one (see `cov_issues/07-fixed-variables-and-covariance.md`, an
//! explicitly deferred core-solver limitation, unrelated to and out of scope
//! for these factors), which is exactly the `T_WA` (fixed) / `T_WB` (free)
//! setup a two-pose registration test needs. Computing the reference
//! covariance directly from the Jacobians sidesteps that limitation and is
//! arguably a more targeted test of the new factors' analytic Jacobians
//! specifically.
//!
//! For each trial we compute the squared Mahalanobis distance (NEES —
//! normalized estimation error squared) between the tangent-space estimation
//! error and this directly-computed covariance; averaged over many trials
//! this should land close to the pose's 6 tangent-space degrees of freedom if
//! the covariance is a statistically consistent estimate of the true
//! estimation uncertainty.

use apex_solver::apex_manifolds::se3::{SE3, SE3Tangent};
use apex_solver::apex_manifolds::{LieGroup, Tangent};
use apex_solver::factors::Factor;
use apex_solver::optimizer::OptimizationStatus;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use apex_solver::{
    JacobianMode, ManifoldType,
    core::problem::Problem,
    factors::lidar::{LidarEdgeFactor, lidar_plane_factor_isotropic},
};
use nalgebra::{DMatrix, DVector, Matrix3, UnitQuaternion, Vector3};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// A point on a plane, in frame A: `plane_point + u*basis_u + v*basis_v`.
struct PlaneFeature {
    point: Vector3<f64>,
    normal: Vector3<f64>,
    basis_u: Vector3<f64>,
    basis_v: Vector3<f64>,
}

/// A point on an edge, in frame A: `edge_point + t*direction`.
struct EdgeFeature {
    point: Vector3<f64>,
    direction: Vector3<f64>,
}

fn planes() -> Vec<PlaneFeature> {
    vec![
        PlaneFeature {
            point: Vector3::new(0.0, 0.0, 0.0),
            normal: Vector3::z(),
            basis_u: Vector3::x(),
            basis_v: Vector3::y(),
        },
        PlaneFeature {
            point: Vector3::new(5.0, 0.0, 0.0),
            normal: Vector3::x(),
            basis_u: Vector3::y(),
            basis_v: Vector3::z(),
        },
        PlaneFeature {
            point: Vector3::new(0.0, 5.0, 0.0),
            normal: Vector3::y(),
            basis_u: Vector3::x(),
            basis_v: Vector3::z(),
        },
    ]
}

fn edges() -> Vec<EdgeFeature> {
    vec![
        EdgeFeature {
            point: Vector3::new(2.0, 2.0, 0.0),
            direction: Vector3::z(),
        },
        EdgeFeature {
            point: Vector3::new(0.0, 2.0, 2.0),
            direction: Vector3::x(),
        },
    ]
}

const POINTS_PER_FEATURE: usize = 8;
const HALF_EXTENT: f64 = 2.0;
const SIGMA: f64 = 0.02;

/// One correspondence's frame-A geometry plus its exact (noise-free) `point_b`.
enum Correspondence {
    Plane {
        point_b_exact: Vector3<f64>,
        plane_point: Vector3<f64>,
        plane_normal: Vector3<f64>,
    },
    Edge {
        point_b_exact: Vector3<f64>,
        edge_point: Vector3<f64>,
        edge_direction: Vector3<f64>,
    },
}

/// Sample a fixed set of correspondences (feature points expressed in frame A,
/// with their exact `point_b` counterpart under `gt_pose`). The set of points
/// sampled is fixed per RNG draw and reused (with different injected noise)
/// across the covariance-reference computation and every trial.
fn sample_correspondences(rng: &mut StdRng, gt_pose: &SE3) -> Vec<Correspondence> {
    let r_gt: Matrix3<f64> = gt_pose.rotation_so3().rotation_matrix();
    let t_gt = gt_pose.translation();

    let mut out = Vec::new();
    for plane in planes() {
        for _ in 0..POINTS_PER_FEATURE {
            let u = rng.random_range(-HALF_EXTENT..HALF_EXTENT);
            let v = rng.random_range(-HALF_EXTENT..HALF_EXTENT);
            let point_a = plane.point + u * plane.basis_u + v * plane.basis_v;
            let point_b_exact = r_gt.transpose() * (point_a - t_gt);
            out.push(Correspondence::Plane {
                point_b_exact,
                plane_point: plane.point,
                plane_normal: plane.normal,
            });
        }
    }
    for edge in edges() {
        for _ in 0..POINTS_PER_FEATURE {
            let t = rng.random_range(-HALF_EXTENT..HALF_EXTENT);
            let point_a = edge.point + t * edge.direction;
            let point_b_exact = r_gt.transpose() * (point_a - t_gt);
            out.push(Correspondence::Edge {
                point_b_exact,
                edge_point: edge.point,
                edge_direction: edge.direction,
            });
        }
    }
    out
}

/// Box-Muller standard-normal sample (no extra dependency on `rand_distr`).
fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(1e-12..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn noisy_vector3(rng: &mut StdRng, sigma: f64) -> Vector3<f64> {
    Vector3::new(
        sigma * standard_normal(rng),
        sigma * standard_normal(rng),
        sigma * standard_normal(rng),
    )
}

fn se3_param_vec(translation: Vector3<f64>, rotation: UnitQuaternion<f64>) -> DVector<f64> {
    let q = rotation.quaternion();
    DVector::from_vec(vec![
        translation.x,
        translation.y,
        translation.z,
        q.w,
        q.i,
        q.j,
        q.k,
    ])
}

/// Build a boxed `Factor` for a correspondence given a (possibly noisy) `point_b`.
fn make_factor(corr: &Correspondence, point_b: Vector3<f64>) -> Box<dyn Factor + Send> {
    match corr {
        Correspondence::Plane {
            plane_point,
            plane_normal,
            ..
        } => Box::new(lidar_plane_factor_isotropic(
            point_b,
            *plane_point,
            *plane_normal,
            SIGMA,
        )),
        Correspondence::Edge {
            edge_point,
            edge_direction,
            ..
        } => Box::new(LidarEdgeFactor::new_isotropic(
            point_b,
            *edge_point,
            *edge_direction,
            SIGMA,
        )),
    }
}

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Reference covariance for `T_WB`, computed directly from the factors' own
/// analytic Jacobians at the noise-free ground-truth linearization point:
/// `Cov = (sum_i Jᵢᵀ Jᵢ)⁻¹`, restricted to the `T_WB` column block (6..12).
fn expected_covariance(
    correspondences: &[Correspondence],
    t_wa: &[f64],
    t_wb_gt: &[f64],
) -> Result<DMatrix<f64>, Box<dyn std::error::Error>> {
    let mut information = DMatrix::<f64>::zeros(6, 6);

    for corr in correspondences {
        let point_b_exact = match corr {
            Correspondence::Plane { point_b_exact, .. } => *point_b_exact,
            Correspondence::Edge { point_b_exact, .. } => *point_b_exact,
        };
        let factor = make_factor(corr, point_b_exact);

        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0f64; rows];
        let mut jac_buf = vec![0.0f64; rows * cols];
        let jac_mut = faer::mat::MatMut::from_column_major_slice_mut(&mut jac_buf, rows, cols);
        factor.linearize(&[t_wa, t_wb_gt], &mut residual, Some(jac_mut));
        let jac = DMatrix::from_column_slice(rows, cols, &jac_buf);

        // Columns 6..12 are the T_WB block (columns 0..6 are T_WA).
        let j_b = jac.columns(6, 6);
        information += j_b.transpose() * j_b;
    }

    information
        .try_inverse()
        .ok_or_else(|| "information matrix from a well-observed corner should be invertible".into())
}

/// Run one trial: build a fresh problem with noisy correspondences, solve for
/// `T_WB`, and return the tangent-space estimation error `T_WB_est ⊟ T_WB_gt`.
fn run_trial(
    rng: &mut StdRng,
    correspondences: &[Correspondence],
    gt_pose: &SE3,
    initial_guess: &DVector<f64>,
) -> Result<[f64; 6], Box<dyn std::error::Error>> {
    let mut problem = Problem::new(JacobianMode::Sparse);

    let t_wa_key = problem.add_variable(
        ManifoldType::SE3,
        se3_param_vec(Vector3::zeros(), UnitQuaternion::identity()),
    );
    for dof in 0..6 {
        problem.fix_variable(t_wa_key, dof);
    }

    let t_wb_key = problem.add_variable(ManifoldType::SE3, initial_guess.clone());

    for corr in correspondences {
        let point_b_exact = match corr {
            Correspondence::Plane { point_b_exact, .. } => *point_b_exact,
            Correspondence::Edge { point_b_exact, .. } => *point_b_exact,
        };
        let point_b_noisy = point_b_exact + noisy_vector3(rng, SIGMA);
        let factor = make_factor(corr, point_b_noisy);
        problem.add_residual_block(&[t_wa_key, t_wb_key], factor, None);
    }

    let config = LevenbergMarquardtConfig::new()
        .with_max_iterations(50)
        .with_cost_tolerance(1e-14)
        .with_parameter_tolerance(1e-14);
    let mut solver = LevenbergMarquardt::with_config(config);

    let result = solver.optimize(&mut problem)?;

    let converged = matches!(
        result.status,
        OptimizationStatus::Converged
            | OptimizationStatus::CostToleranceReached
            | OptimizationStatus::ParameterToleranceReached
            | OptimizationStatus::GradientToleranceReached
            | OptimizationStatus::StalledNoProgress
    );
    assert!(converged, "trial did not converge: {:?}", result.status);

    let est_vec = result
        .parameters
        .get(t_wb_key)
        .ok_or("T_WB should be in result")?
        .to_dvector();
    let est_pose = SE3::from_param_slice(est_vec.as_slice());

    let error_tangent = est_pose.minus(gt_pose, None, None);
    let error = error_tangent.as_slice();
    let mut out = [0.0f64; 6];
    out.copy_from_slice(error);
    Ok(out)
}

#[test]
fn lidar_factors_covariance_is_statistically_consistent() -> TestResult {
    const NUM_TRIALS: usize = 40;
    const DOF: f64 = 6.0;

    let mut rng = StdRng::seed_from_u64(42);

    let gt_translation = Vector3::new(0.3, -0.2, 0.15);
    let gt_rotation = UnitQuaternion::from_axis_angle(
        &nalgebra::Unit::new_normalize(Vector3::new(0.2, 0.5, 0.8)),
        0.25,
    );
    let gt_pose = SE3::new(gt_translation, gt_rotation);
    let t_wa_identity = se3_param_vec(Vector3::zeros(), UnitQuaternion::identity());
    let t_wb_gt = DVector::from_column_slice(gt_pose.as_param_slice());

    // Fixed, deterministic initial-guess offset from ground truth (same every
    // trial) so the optimizer has real work to do without affecting the
    // statistics of the noise-driven estimation error we're checking.
    let offset = SE3Tangent::new(
        Vector3::new(0.1, -0.05, 0.07),
        Vector3::new(0.05, -0.03, 0.02),
    );
    let initial_pose = gt_pose.right_plus(&offset, None, None);
    let initial_guess = DVector::from_column_slice(initial_pose.as_param_slice());

    // Correspondence geometry is fixed once; only the injected measurement
    // noise varies per trial, matching a fixed real-world scene being
    // re-scanned with independent sensor noise each time.
    let correspondences = sample_correspondences(&mut rng, &gt_pose);

    let cov = expected_covariance(
        &correspondences,
        t_wa_identity.as_slice(),
        t_wb_gt.as_slice(),
    )?;
    let cov_inv = cov.try_inverse().ok_or(
        "expected covariance should itself be invertible (it's an inverse of an SPD matrix)",
    )?;

    let mut total_nees = 0.0;
    for _ in 0..NUM_TRIALS {
        let error = run_trial(&mut rng, &correspondences, &gt_pose, &initial_guess)?;
        let e = DVector::from_column_slice(&error);
        let nees = (e.transpose() * &cov_inv * &e)[(0, 0)];
        assert!(nees.is_finite(), "NEES must be finite, got {nees}");
        total_nees += nees;
    }
    let avg_nees = total_nees / NUM_TRIALS as f64;

    // Expected value of NEES for a consistent 6-DOF estimator is 6. With 40
    // trials the sampling variance of the average is large enough that a tight
    // band would be flaky, so we use a generous but still meaningful range
    // that would fail on a badly wrong (e.g. off-by-constant-factor or
    // structurally incorrect) covariance propagation.
    assert!(
        (2.0..14.0).contains(&avg_nees),
        "average NEES = {avg_nees:.3}, expected close to DOF = {DOF} \
         (propagated covariance appears statistically inconsistent with the \
         empirical estimation error)"
    );
    Ok(())
}
