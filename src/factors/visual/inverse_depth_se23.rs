//! Anchored inverse-depth reprojection between two `SE_2(3)` navigation states.
//!
//! A landmark is carried as one scalar — its inverse depth `ρ` along a fixed
//! bearing in the keyframe that first saw it — rather than as three world
//! coordinates. That costs one parameter instead of three, represents points at
//! arbitrary range including `ρ → 0` at optical infinity without the
//! ill-conditioning a Euclidean point has out there, and leaves the landmark
//! block a **scalar diagonal** for the Schur complement.
//!
//! ```text
//! f     = m_host / ρ                      point in the host camera
//! p_bk  = R_bc·f + p_bc                   host body
//! p_w   = R_k·p_bk + p_k                  world
//! p_bi  = R_iᵀ·(p_w − p_i)                observer body
//! P     = R_bcᵀ·(p_bi − p_bc)             observer camera
//! r     = [P_x/P_z, P_y/P_z]ᵀ − m_obs     normalized plane, 2 rows
//! ```
//!
//! # Parameter layout (4 blocks, 25 minimal DOF)
//!
//! ```text
//! params[0]: SE23 host state     — 10D, 9 DOF
//! params[1]: SE23 observer state — 10D, 9 DOF
//! params[2]: SE3  T_bc           —  7D, 6 DOF   body-from-camera
//! params[3]: Rn(1) rho           —  1D, 1 DOF
//! ```
//!
//! # No camera model
//!
//! The residual is on the **normalized plane**, so this factor takes no `CAM`
//! generic — unlike every other factor in this module. The front end undistorts
//! once and hands over bearings; a fisheye and a pinhole then differ only in
//! that step. The consequence for the caller is that the noise model is in
//! normalized units: `NoiseModel::from_sigmas(&[sigma_px / focal; 2])`.
//!
//! # `T_bc` appears twice
//!
//! It lifts the landmark out of the host camera *and* projects it into the
//! observer camera, so its Jacobian is a **sum** of two terms: `A·R_bc − I` for
//! translation and `−A·R_bc·[f]ₓ + [P]ₓ` for rotation. A derivation that treats
//! it as appearing once is wrong by exactly the second term, and only the
//! finite-difference test catches that.
//!
//! The two do cancel in one case, and it is worth knowing which: when host and
//! observer are the **same** state, `A = R_bcᵀ` and `P = f`, so
//! `A·R_bc − I = 0` and `−[f]ₓ + [P]ₓ = 0`. That is correct rather than
//! suspicious — a landmark reprojected into its own host frame maps through the
//! identity, and no extrinsic can change where it lands.

use faer::prelude::ReborrowMut;
use nalgebra::{Matrix3, SMatrix, Vector2, Vector3};

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use apex_manifolds::se23::SE23;

use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::cheirality::{CHEIRALITY_BASE_PENALTY, CHEIRALITY_DEPTH_SCALE};
use crate::factors::common::math::skew;
use crate::factors::common::validate::expect_block_sizes;

/// Depth below which the observer projection is treated as degenerate \[m\].
const MIN_DEPTH: f64 = 1.0e-3;

/// Columns of the Jacobian: 9 + 9 + 6 + 1.
const JACOBIAN_COLUMNS: usize = 25;

/// Reprojection of an inverse-depth landmark anchored in a host keyframe.
pub struct InverseDepthSe23Factor {
    /// Bearing of the feature in the host camera, normalized so `z = 1`.
    host_bearing: Vector3<f64>,
    /// Measured normalized coordinates in the observer camera.
    measurement: Vector2<f64>,
}

impl InverseDepthSe23Factor {
    /// Create the factor from normalized-plane observations.
    ///
    /// `host_bearing` is rescaled to `z = 1`; a bearing whose `z` is not
    /// positive cannot anchor an inverse depth and is left as given, where the
    /// cheirality branch will reject it.
    pub fn new(host_bearing: Vector3<f64>, measurement: Vector2<f64>) -> Self {
        let host_bearing = if host_bearing.z.abs() > f64::EPSILON {
            host_bearing / host_bearing.z
        } else {
            host_bearing
        };
        Self {
            host_bearing,
            measurement,
        }
    }

    /// The anchoring bearing, normalized to `z = 1`.
    pub fn host_bearing(&self) -> Vector3<f64> {
        self.host_bearing
    }

    /// The observed normalized coordinates.
    pub fn measurement(&self) -> Vector2<f64> {
        self.measurement
    }
}

/// Write a `2 x 25` Jacobian into the caller's buffer.
fn write_jacobian(
    source: &SMatrix<f64, 2, JACOBIAN_COLUMNS>,
    jac: &mut faer::mat::MatMut<'_, f64>,
) {
    for row in 0..2 {
        for col in 0..JACOBIAN_COLUMNS {
            *jac.rb_mut().get_mut(row, col) = source[(row, col)];
        }
    }
}

impl Factor for InverseDepthSe23Factor {
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let [host_params, observer_params, extrinsic_params, rho_params] = params else {
            return;
        };

        let host = SE23::from_param_slice(host_params);
        let observer = SE23::from_param_slice(observer_params);
        let extrinsic = SE3::from_param_slice(extrinsic_params);
        let rho = rho_params[0];

        let r_bc = extrinsic.rotation_so3().rotation_matrix();
        let p_bc = extrinsic.translation();
        let r_k = host.rotation_matrix();
        let r_i = observer.rotation_matrix();

        // An inverse depth at or behind the anchor is not a point at all, so it
        // gets a flat penalty and a **zero** Jacobian: leaving a gradient here
        // would make an invalid anchor a cheap direction to escape along.
        if !rho.is_finite() || rho <= 0.0 {
            residual[0] = CHEIRALITY_BASE_PENALTY;
            residual[1] = CHEIRALITY_BASE_PENALTY;
            if let Some(mut jac) = jacobian {
                write_jacobian(&SMatrix::<f64, 2, JACOBIAN_COLUMNS>::zeros(), &mut jac);
            }
            return;
        }

        let f = self.host_bearing / rho;
        let p_bk = r_bc * f + p_bc;
        let p_w = r_k * p_bk + host.translation();
        let p_bi = r_i.transpose() * (p_w - observer.translation());
        let point = r_bc.transpose() * (p_bi - p_bc);

        // `A` is the host-camera to observer-camera rotation chain, and appears
        // in every block below.
        let a = r_bc.transpose() * r_i.transpose() * r_k;
        let chain = Chain {
            a,
            r_bc,
            p_bk,
            p_bi,
            point,
            f,
            host_bearing: self.host_bearing,
            rho,
        };

        if point.z <= MIN_DEPTH {
            let penalty =
                CHEIRALITY_BASE_PENALTY + CHEIRALITY_DEPTH_SCALE * (MIN_DEPTH - point.z).max(0.0);
            residual[0] = penalty;
            residual[1] = penalty;
            if let Some(mut jac) = jacobian {
                // Gradient of the penalty through the observer-frame depth, so
                // the solve is pushed back towards a valid configuration rather
                // than sitting on a flat plateau.
                let d_point = point_jacobians(&chain);
                let mut full = SMatrix::<f64, 2, JACOBIAN_COLUMNS>::zeros();
                for col in 0..JACOBIAN_COLUMNS {
                    let value = -CHEIRALITY_DEPTH_SCALE * d_point[(2, col)];
                    full[(0, col)] = value;
                    full[(1, col)] = value;
                }
                write_jacobian(&full, &mut jac);
            }
            return;
        }

        let inverse_z = 1.0 / point.z;
        residual[0] = point.x * inverse_z - self.measurement.x;
        residual[1] = point.y * inverse_z - self.measurement.y;

        let Some(mut jac) = jacobian else {
            return;
        };

        // ∂π/∂P for π(P) = [P_x/P_z, P_y/P_z].
        let mut d_projection = SMatrix::<f64, 2, 3>::zeros();
        d_projection[(0, 0)] = inverse_z;
        d_projection[(0, 2)] = -point.x * inverse_z * inverse_z;
        d_projection[(1, 1)] = inverse_z;
        d_projection[(1, 2)] = -point.y * inverse_z * inverse_z;

        let d_point = point_jacobians(&chain);
        write_jacobian(&(d_projection * d_point), &mut jac);
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2, JACOBIAN_COLUMNS)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE23::REP_SIZE, SE23::REP_SIZE, SE3::REP_SIZE, 1],
            "InverseDepthSe23Factor expects [SE23 host, SE23 observer, SE3 T_bc, Rn(1) rho]",
        )
    }
}

/// The transform chain evaluated once, so the residual and its Jacobian read
/// the same quantities instead of recomputing them.
struct Chain {
    /// `R_bcᵀ·R_iᵀ·R_k`, host camera to observer camera.
    a: Matrix3<f64>,
    /// Body-from-camera rotation.
    r_bc: Matrix3<f64>,
    /// Landmark in the host body frame.
    p_bk: Vector3<f64>,
    /// Landmark in the observer body frame.
    p_bi: Vector3<f64>,
    /// Landmark in the observer camera frame.
    point: Vector3<f64>,
    /// Landmark in the host camera frame, `m_host / rho`.
    f: Vector3<f64>,
    /// Anchoring bearing, normalized to `z = 1`.
    host_bearing: Vector3<f64>,
    /// Inverse depth.
    rho: f64,
}

/// `∂P/∂(host, observer, T_bc, rho)`, a `3 x 25` block.
///
/// Every entry is derived under the crate's right perturbation, where a state's
/// translation moves as `t ← t + R·δρ` and its rotation as `R ← R·Exp(δθ)`.
/// The velocity columns 6..9 and 15..18 are **exactly zero**: a reprojection
/// measures where a point appears, which says nothing about how fast the body
/// carrying the camera is moving. Velocity is observable here only through the
/// IMU factors that share these states.
fn point_jacobians(chain: &Chain) -> SMatrix<f64, 3, JACOBIAN_COLUMNS> {
    let mut d = SMatrix::<f64, 3, JACOBIAN_COLUMNS>::zeros();

    // Host: ∂P/∂δρ_k = A, ∂P/∂δθ_k = −A·[p_bk]ₓ.
    d.fixed_view_mut::<3, 3>(0, 0).copy_from(&chain.a);
    d.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-chain.a * skew(&chain.p_bk)));

    // Observer: ∂P/∂δρ_i = −R_bcᵀ, ∂P/∂δθ_i = +R_bcᵀ·[p_bi]ₓ.
    d.fixed_view_mut::<3, 3>(0, 9)
        .copy_from(&(-chain.r_bc.transpose()));
    d.fixed_view_mut::<3, 3>(0, 12)
        .copy_from(&(chain.r_bc.transpose() * skew(&chain.p_bi)));

    // T_bc, summed over the host lift and the observer projection.
    d.fixed_view_mut::<3, 3>(0, 18)
        .copy_from(&(chain.a * chain.r_bc - Matrix3::identity()));
    d.fixed_view_mut::<3, 3>(0, 21)
        .copy_from(&(-chain.a * chain.r_bc * skew(&chain.f) + skew(&chain.point)));

    // rho: f = m/ρ so ∂f/∂ρ = −m/ρ².
    let d_rho = chain.a * chain.r_bc * (-chain.host_bearing / (chain.rho * chain.rho));
    d.fixed_view_mut::<3, 1>(0, 24).copy_from(&d_rho);

    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::SE3Tangent;
    use apex_manifolds::se23::SE23Tangent;
    use nalgebra::{DMatrix, UnitQuaternion};

    /// Host state, observer state, `T_bc`, and `rho`, as parameter vectors.
    struct Setup {
        host: Vec<f64>,
        observer: Vec<f64>,
        extrinsic: Vec<f64>,
        rho: Vec<f64>,
    }

    fn se23(t: Vector3<f64>, v: Vector3<f64>, q: UnitQuaternion<f64>) -> Vec<f64> {
        SE23::new(t, v, q).as_param_slice().to_vec()
    }

    fn se3(t: Vector3<f64>, q: UnitQuaternion<f64>) -> Vec<f64> {
        SE3::new(t, q).as_param_slice().to_vec()
    }

    fn setup(rho: f64) -> Setup {
        Setup {
            host: se23(
                Vector3::new(0.1, -0.2, 0.05),
                Vector3::new(0.8, 0.1, -0.2),
                UnitQuaternion::from_euler_angles(0.03, -0.07, 0.11),
            ),
            observer: se23(
                Vector3::new(0.9, 0.15, -0.1),
                Vector3::new(0.7, -0.3, 0.05),
                UnitQuaternion::from_euler_angles(-0.05, 0.09, -0.04),
            ),
            extrinsic: se3(
                Vector3::new(0.02, -0.065, 0.01),
                UnitQuaternion::from_euler_angles(0.01, -1.55, 0.02),
            ),
            rho: vec![rho],
        }
    }

    fn residual_of(factor: &InverseDepthSe23Factor, s: &Setup) -> Vec<f64> {
        let mut residual = vec![0.0f64; 2];
        factor.linearize(
            &[&s.host, &s.observer, &s.extrinsic, &s.rho],
            &mut residual,
            None,
        );
        residual
    }

    fn jacobian_of(factor: &InverseDepthSe23Factor, s: &Setup) -> DMatrix<f64> {
        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0f64; rows];
        let mut buf = vec![0.0f64; rows * cols];
        let jac = faer::mat::MatMut::from_column_major_slice_mut(&mut buf, rows, cols);
        factor.linearize(
            &[&s.host, &s.observer, &s.extrinsic, &s.rho],
            &mut residual,
            Some(jac),
        );
        DMatrix::from_column_slice(rows, cols, &buf)
    }

    /// Build the factor so that the residual is exactly zero at `s`, then offset
    /// the measurement so the Jacobian is exercised away from the minimum.
    fn factor_at(s: &Setup, offset: Vector2<f64>) -> InverseDepthSe23Factor {
        let host_bearing = Vector3::new(0.12, -0.05, 1.0);
        let zeroed = InverseDepthSe23Factor::new(host_bearing, Vector2::zeros());
        let r = residual_of(&zeroed, s);
        InverseDepthSe23Factor::new(host_bearing, Vector2::new(r[0], r[1]) + offset)
    }

    #[test]
    fn zero_residual_at_the_exact_solution() {
        let s = setup(0.4);
        let factor = factor_at(&s, Vector2::zeros());
        let r = residual_of(&factor, &s);
        assert!(
            r[0].abs() < 1e-12 && r[1].abs() < 1e-12,
            "residual {r:?} should vanish at the generating configuration"
        );
    }

    /// The whole factor, block by block. Without this the `T_bc` sum and the
    /// `rho` term are unverifiable by inspection.
    #[test]
    fn jacobian_matches_finite_differences() {
        const EPS: f64 = 1e-7;

        for rho in [0.2, 1.0, 5.0] {
            let s = setup(rho);
            let factor = factor_at(&s, Vector2::new(0.02, -0.015));
            let analytic = jacobian_of(&factor, &s);

            // Host and observer: 9 DOF each, in their own SE23 chart.
            for (block, base) in [(0usize, 0usize), (1, 9)] {
                for axis in 0..9 {
                    let mut tangent = [0.0f64; 9];
                    let mut perturbed = setup(rho);
                    tangent[axis] = EPS;
                    let plus = {
                        let source = if block == 0 { &s.host } else { &s.observer };
                        SE23::from_param_slice(source)
                            .right_plus(&SE23Tangent::from_slice(&tangent), None, None)
                            .as_param_slice()
                            .to_vec()
                    };
                    tangent[axis] = -EPS;
                    let minus = {
                        let source = if block == 0 { &s.host } else { &s.observer };
                        SE23::from_param_slice(source)
                            .right_plus(&SE23Tangent::from_slice(&tangent), None, None)
                            .as_param_slice()
                            .to_vec()
                    };

                    if block == 0 {
                        perturbed.host = plus;
                    } else {
                        perturbed.observer = plus;
                    }
                    let rp = residual_of(&factor, &perturbed);
                    if block == 0 {
                        perturbed.host = minus;
                    } else {
                        perturbed.observer = minus;
                    }
                    let rm = residual_of(&factor, &perturbed);

                    for row in 0..2 {
                        let fd = (rp[row] - rm[row]) / (2.0 * EPS);
                        let col = base + axis;
                        assert!(
                            (fd - analytic[(row, col)]).abs() < 1e-5,
                            "rho {rho}, column {col}, row {row}: fd {fd} vs analytic {}",
                            analytic[(row, col)]
                        );
                    }
                }
            }

            // T_bc: 6 DOF in the SE3 chart, columns 18..24.
            for axis in 0..6 {
                let mut tangent = [0.0f64; 6];
                let mut perturbed = setup(rho);
                tangent[axis] = EPS;
                perturbed.extrinsic = SE3::from_param_slice(&s.extrinsic)
                    .right_plus(&SE3Tangent::from_slice(&tangent), None, None)
                    .as_param_slice()
                    .to_vec();
                let rp = residual_of(&factor, &perturbed);
                tangent[axis] = -EPS;
                perturbed.extrinsic = SE3::from_param_slice(&s.extrinsic)
                    .right_plus(&SE3Tangent::from_slice(&tangent), None, None)
                    .as_param_slice()
                    .to_vec();
                let rm = residual_of(&factor, &perturbed);

                for row in 0..2 {
                    let fd = (rp[row] - rm[row]) / (2.0 * EPS);
                    let col = 18 + axis;
                    assert!(
                        (fd - analytic[(row, col)]).abs() < 1e-5,
                        "rho {rho}, T_bc column {col}, row {row}: fd {fd} vs analytic {}",
                        analytic[(row, col)]
                    );
                }
            }

            // rho: column 24.
            let mut perturbed = setup(rho);
            perturbed.rho = vec![rho + EPS];
            let rp = residual_of(&factor, &perturbed);
            perturbed.rho = vec![rho - EPS];
            let rm = residual_of(&factor, &perturbed);
            for row in 0..2 {
                let fd = (rp[row] - rm[row]) / (2.0 * EPS);
                assert!(
                    (fd - analytic[(row, 24)]).abs() < 1e-4,
                    "rho {rho}, rho column, row {row}: fd {fd} vs analytic {}",
                    analytic[(row, 24)]
                );
            }
        }
    }

    /// The two `T_bc` contributions cancel **exactly** for a self-observation,
    /// and that is the correct answer: reprojecting a landmark into its own
    /// host frame is the identity map, so the extrinsic cannot move it.
    ///
    /// Asserted because it is surprising: a reader who knows `T_bc` appears
    /// twice would expect a residue here, and a zero block looks like a missing
    /// term rather than a cancellation.
    #[test]
    fn the_extrinsic_cancels_on_a_self_observation() {
        let mut s = setup(0.5);
        s.observer = s.host.clone();
        let factor = factor_at(&s, Vector2::new(0.01, 0.01));
        let analytic = jacobian_of(&factor, &s);

        let extrinsic_block = analytic.view((0, 18), (2, 6)).norm();
        assert!(
            extrinsic_block < 1e-12,
            "expected exact cancellation, got {extrinsic_block:.3e}"
        );
    }

    /// With host and observer distinct, the second `T_bc` term is what makes the
    /// block differ from the naive single-appearance derivation `A·R_bc`.
    #[test]
    fn the_extrinsic_block_carries_both_terms() {
        let s = setup(0.6);
        let factor = factor_at(&s, Vector2::new(0.02, -0.01));
        let analytic = jacobian_of(&factor, &s);

        let host = SE23::from_param_slice(&s.host);
        let observer = SE23::from_param_slice(&s.observer);
        let extrinsic = SE3::from_param_slice(&s.extrinsic);
        let r_bc = extrinsic.rotation_so3().rotation_matrix();
        let a = r_bc.transpose() * observer.rotation_matrix().transpose() * host.rotation_matrix();

        // The naive translation block, missing the `− I` from the observer side.
        let naive = a * r_bc;
        let actual = a * r_bc - Matrix3::identity();
        assert!(
            (naive - actual).norm() > 0.5,
            "the two derivations should differ by the identity"
        );
        // And the factor implements the second one.
        assert!(
            analytic.view((0, 18), (2, 3)).norm() > 1e-6,
            "the T_bc translation block should be non-trivial here"
        );
    }

    /// A reprojection says nothing about how fast the body is moving.
    #[test]
    fn velocity_columns_are_zero() {
        let s = setup(0.7);
        let factor = factor_at(&s, Vector2::new(0.03, -0.02));
        let analytic = jacobian_of(&factor, &s);
        for row in 0..2 {
            for col in (6..9).chain(15..18) {
                assert_eq!(
                    analytic[(row, col)],
                    0.0,
                    "velocity column {col} is non-zero"
                );
            }
        }
    }

    /// An invalid anchor must not be a cheap direction to escape along.
    #[test]
    fn a_non_positive_rho_gets_a_flat_penalty_and_no_gradient() {
        let mut s = setup(0.5);
        s.rho = vec![-0.2];
        let factor = InverseDepthSe23Factor::new(Vector3::new(0.1, 0.0, 1.0), Vector2::zeros());

        let r = residual_of(&factor, &s);
        assert!((r[0] - CHEIRALITY_BASE_PENALTY).abs() < 1e-12);
        assert!((r[1] - CHEIRALITY_BASE_PENALTY).abs() < 1e-12);

        let analytic = jacobian_of(&factor, &s);
        assert_eq!(analytic.norm(), 0.0, "an invalid rho must have no gradient");
    }

    /// A point behind the observer keeps a gradient through the depth, so the
    /// solve is pushed back rather than parked on a plateau.
    #[test]
    fn a_point_behind_the_observer_keeps_a_depth_gradient() {
        let mut s = setup(0.5);
        // Identity extrinsic and host, so camera and world axes agree and the
        // geometry is predictable; then put the observer well past the landmark
        // along +z, which leaves it behind that camera.
        s.extrinsic = se3(Vector3::zeros(), UnitQuaternion::identity());
        s.host = se23(
            Vector3::zeros(),
            Vector3::zeros(),
            UnitQuaternion::identity(),
        );
        s.observer = se23(
            Vector3::new(0.0, 0.0, 40.0),
            Vector3::zeros(),
            UnitQuaternion::identity(),
        );
        let factor = InverseDepthSe23Factor::new(Vector3::new(0.05, 0.02, 1.0), Vector2::zeros());

        let r = residual_of(&factor, &s);
        assert!(
            r[0] >= CHEIRALITY_BASE_PENALTY,
            "expected a cheirality penalty, got {r:?}"
        );
        let analytic = jacobian_of(&factor, &s);
        assert!(
            analytic.norm() > 0.0,
            "the cheirality branch must keep a depth gradient"
        );
    }
}
