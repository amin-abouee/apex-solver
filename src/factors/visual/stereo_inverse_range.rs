//! Pixel reprojection of an anchored inverse-range landmark into a **second
//! camera** of a rig, with that camera's intrinsics as a variable.
//!
//! [`super::InverseDepthBearingFactor`] measures a landmark in the camera
//! that hosts it: one body-from-camera extrinsic serves both the host lift and
//! the observer projection. A stereo rig breaks that symmetry. A landmark is
//! anchored in camera 0 and seen by camera 1, so the two ends of the chain use
//! **different** extrinsics, and the observation is a pixel of camera 1 whose
//! intrinsics a caller may want to calibrate online.
//!
//! ```text
//! f    = m_host / ρ                       point in the host camera (camera 0)
//! p_bk = R_b0·f + p_b0                    host body
//! p_w  = R_k·p_bk + p_k                   world
//! p_bi = R_iᵀ·(p_w − p_i)                 observer body
//! P    = R_b1ᵀ·(p_bi − p_b1)              observer camera (camera 1)
//! r    = s·(π_K(P) − uv)                  pixels, whitened by s = 1/σ
//! ```
//!
//! Two factors share this chain:
//!
//! - [`InverseRangeProjectionFactor`] — camera 1 at a **different**
//!   keyframe than the host: blocks `[host SE23, observer SE23, T_b0, T_b1, ρ,
//!   K]`.
//! - [`StereoInverseRangeFactor`] — camera 1 at the **same** instant: the body
//!   state cancels exactly (`p_bi = p_bk`), leaving `[T_b0, T_b1, ρ, K]`. This
//!   is the measurement that fixes a landmark's metric range from the rig's
//!   baseline alone, independent of how the body moved.
//!
//! `K` is an `Rn(CAM::INTRINSIC_DIM)` block decoded with `CAM::try_from`; a
//! caller that does not calibrate intrinsics holds that block fixed.
//!
//! Jacobians follow the crate's right perturbation (`t ← t + R·δρ`,
//! `R ← R·Exp(δθ)`). Summing the two extrinsic blocks of the moving-body factor
//! with `T_b0 = T_b1` gives exactly the single-extrinsic block of
//! [`super::InverseDepthPlaneFactor`], which the tests pin.

use std::marker::PhantomData;

use apex_camera_models::CameraModel;
use apex_manifolds::LieGroup;
use apex_manifolds::capability::RigidPose;
use apex_manifolds::se3::SE3;
use faer::prelude::ReborrowMut;
use nalgebra::{DMatrix, Matrix3, Vector2, Vector3};

use crate::core::CoreError;
use crate::core::CoreResult;
use crate::core::variable::ManifoldVariable;
use crate::factors::Factor;
use crate::factors::common::cheirality::{CHEIRALITY_BASE_PENALTY, CHEIRALITY_DEPTH_SCALE};
use crate::factors::common::math::skew;
use crate::factors::common::validate::expect_block_sizes;

/// Geometric columns of [`InverseRangeProjectionFactor`] over pose group `G`:
/// two states, two `SE3` extrinsics (6 each) and `ρ`.
const fn moving_columns<G: RigidPose>() -> usize {
    2 * G::DOF + 13
}

/// Geometric columns of [`StereoInverseRangeFactor`]: two extrinsics and `ρ`.
const STEREO_COLUMNS: usize = 13;

/// What the two factors share: the anchoring bearing, the pixel, and the
/// whitening.
#[derive(Debug, Clone)]
struct Observation {
    host_bearing: Vector3<f64>,
    measurement: Vector2<f64>,
    sqrt_information: f64,
}

impl Observation {
    fn new(
        host_bearing: Vector3<f64>,
        measurement: Vector2<f64>,
        pixel_sigma: f64,
    ) -> CoreResult<Self> {
        let norm = host_bearing.norm();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(CoreError::InvalidInput(format!(
                "host bearing must be finite and non-zero, got {host_bearing:?}"
            )));
        }
        if !measurement.iter().all(|v| v.is_finite()) {
            return Err(CoreError::InvalidInput(format!(
                "measurement must be finite, got {measurement:?}"
            )));
        }
        if !pixel_sigma.is_finite() || pixel_sigma <= 0.0 {
            return Err(CoreError::InvalidInput(format!(
                "pixel sigma must be finite and positive, got {pixel_sigma}"
            )));
        }
        Ok(Self {
            host_bearing: host_bearing / norm,
            measurement,
            sqrt_information: 1.0 / pixel_sigma,
        })
    }

    /// Project `point` (camera 1) through `camera` and write the whitened
    /// residual and the full Jacobian: `d_point` holds `∂P/∂(geometry)` in its
    /// first `geometry` columns; the intrinsic columns follow.
    fn write<CAM: CameraModel>(
        &self,
        camera: &CAM,
        point: &Vector3<f64>,
        d_point: &DMatrix<f64>,
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let geometry = d_point.ncols();
        let columns = geometry + CAM::INTRINSIC_DIM;
        let (deficit, gradient) = camera.projection_deficit(point);
        let projected = camera.project(point);
        let Ok(pixel) = projected
            .as_ref()
            .map(|p| *p)
            .map_err(|_| ())
            .and_then(|p| if deficit > 0.0 { Err(()) } else { Ok(p) })
        else {
            // Outside the model's domain: a penalty that grows with the depth
            // deficit, whose gradient pushes the point back into the domain.
            let penalty = CHEIRALITY_BASE_PENALTY + CHEIRALITY_DEPTH_SCALE * deficit.max(0.0);
            residual[0] = penalty;
            residual[1] = penalty;
            if let Some(mut jac) = jacobian {
                for col in 0..columns {
                    let value = if col < geometry && deficit > 0.0 {
                        CHEIRALITY_DEPTH_SCALE * (gradient.transpose() * d_point.column(col))[0]
                    } else {
                        0.0
                    };
                    *jac.rb_mut().get_mut(0, col) = value;
                    *jac.rb_mut().get_mut(1, col) = value;
                }
            }
            return;
        };

        let s = self.sqrt_information;
        residual[0] = s * (pixel.x - self.measurement.x);
        residual[1] = s * (pixel.y - self.measurement.y);

        let Some(mut jac) = jacobian else {
            return;
        };
        let d_projection = camera.jacobian_point(point);
        for row in 0..2 {
            for col in 0..geometry {
                let mut value = 0.0;
                for k in 0..3 {
                    value += d_projection[(row, k)] * d_point[(k, col)];
                }
                *jac.rb_mut().get_mut(row, col) = s * value;
            }
        }
        let d_intrinsics = camera.jacobian_intrinsics(point);
        for row in 0..2 {
            for col in 0..CAM::INTRINSIC_DIM {
                *jac.rb_mut().get_mut(row, geometry + col) = s * d_intrinsics[(row, col)];
            }
        }
    }

    /// A residual for an invalid `ρ` or undecodable intrinsics: flat, with a
    /// zero Jacobian, so an invalid configuration is never a cheap direction.
    fn write_invalid(residual: &mut [f64], jacobian: Option<faer::mat::MatMut<'_, f64>>) {
        residual[0] = CHEIRALITY_BASE_PENALTY;
        residual[1] = CHEIRALITY_BASE_PENALTY;
        if let Some(mut jac) = jacobian {
            let (rows, cols) = (jac.nrows(), jac.ncols());
            for row in 0..rows {
                for col in 0..cols {
                    *jac.rb_mut().get_mut(row, col) = 0.0;
                }
            }
        }
    }
}

/// Body-from-camera rotation and translation of an `SE3` extrinsic.
fn extrinsic(params: &[f64]) -> (Matrix3<f64>, Vector3<f64>) {
    let t = SE3::from_param_slice(params);
    (t.rotation_so3().rotation_matrix(), t.translation())
}

/// Camera 1 observes a camera-0-anchored landmark at another keyframe.
///
/// # Parameter layout (6 blocks, 31 + `INTRINSIC_DIM` minimal DOF)
///
/// ```text
/// params[0]: SE23 host state       — 9 DOF
/// params[1]: SE23 observer state   — 9 DOF
/// params[2]: SE3  T_b0             — 6 DOF  body-from-camera-0 (host camera)
/// params[3]: SE3  T_b1             — 6 DOF  body-from-camera-1 (observer camera)
/// params[4]: Rn(1) ρ               — 1 DOF  inverse range along the host bearing
/// params[5]: Rn(N) K               — camera 1 intrinsics
/// ```
///
/// Whitens internally; register with a null noise model.
#[derive(Debug, Clone)]
pub struct InverseRangeProjectionFactor<G, CAM> {
    observation: Observation,
    marker: PhantomData<fn() -> (G, CAM)>,
}

impl<G: RigidPose, CAM> InverseRangeProjectionFactor<G, CAM> {
    /// Create from the host's unit bearing, camera 1's pixel and the pixel
    /// noise.
    ///
    /// # Errors
    ///
    /// A message if the bearing is zero or non-finite, the pixel non-finite,
    /// or the sigma not positive.
    pub fn new(
        host_bearing: Vector3<f64>,
        measurement: Vector2<f64>,
        pixel_sigma: f64,
    ) -> CoreResult<Self> {
        Ok(Self {
            observation: Observation::new(host_bearing, measurement, pixel_sigma)?,
            marker: PhantomData,
        })
    }
}

impl<G, CAM> Factor for InverseRangeProjectionFactor<G, CAM>
where
    G: RigidPose + Send + Sync + 'static,
    CAM: CameraModel,
    for<'a> CAM: TryFrom<&'a [f64]>,
{
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let [host, observer, t_b0, t_b1, rho, intrinsics] = params else {
            return;
        };
        let rho = rho[0];
        let Ok(camera) = CAM::try_from(intrinsics) else {
            Observation::write_invalid(residual, jacobian);
            return;
        };
        if !rho.is_finite() || rho <= 0.0 {
            Observation::write_invalid(residual, jacobian);
            return;
        }

        let host = G::from_param_slice(host);
        let observer = G::from_param_slice(observer);
        let (r_b0, p_b0) = extrinsic(t_b0);
        let (r_b1, p_b1) = extrinsic(t_b1);
        let r_k = host.rotation();
        let r_i = observer.rotation();

        let m = self.observation.host_bearing;
        let f = m / rho;
        let p_bk = r_b0 * f + p_b0;
        let p_w = r_k * p_bk + host.position();
        let p_bi = r_i.transpose() * (p_w - observer.position());
        let point = r_b1.transpose() * (p_bi - p_b1);

        // Body of the host to camera 1 of the observer.
        let a = r_b1.transpose() * r_i.transpose() * r_k;
        let (observer_at, b0_at, b1_at) = (G::DOF, 2 * G::DOF, 2 * G::DOF + 6);
        let mut d = DMatrix::<f64>::zeros(3, moving_columns::<G>());
        d.fixed_view_mut::<3, 3>(0, 0).copy_from(&a);
        d.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-a * skew(&p_bk)));
        d.fixed_view_mut::<3, 3>(0, observer_at)
            .copy_from(&(-r_b1.transpose()));
        d.fixed_view_mut::<3, 3>(0, observer_at + 3)
            .copy_from(&(r_b1.transpose() * skew(&p_bi)));
        d.fixed_view_mut::<3, 3>(0, b0_at).copy_from(&(a * r_b0));
        d.fixed_view_mut::<3, 3>(0, b0_at + 3)
            .copy_from(&(-a * r_b0 * skew(&f)));
        d.fixed_view_mut::<3, 3>(0, b1_at)
            .copy_from(&(-Matrix3::identity()));
        d.fixed_view_mut::<3, 3>(0, b1_at + 3)
            .copy_from(&skew(&point));
        d.fixed_view_mut::<3, 1>(0, b1_at + 6)
            .copy_from(&(a * r_b0 * (-m / (rho * rho))));

        self.observation
            .write(&camera, &point, &d, residual, jacobian);
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2, moving_columns::<G>() + CAM::INTRINSIC_DIM)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[
                G::REP_SIZE,
                G::REP_SIZE,
                SE3::REP_SIZE,
                SE3::REP_SIZE,
                1,
                CAM::INTRINSIC_DIM,
            ],
            "InverseRangeProjectionFactor expects [SE23 host, SE23 observer, SE3 T_b0, SE3 T_b1, Rn(1) rho, Rn(N) intrinsics]",
        )
    }
}

/// Camera 1 observes a camera-0-anchored landmark at the **same** instant.
///
/// ```text
/// P = R_b1ᵀ·(R_b0·m/ρ + p_b0 − p_b1)
/// ```
///
/// # Parameter layout (4 blocks, 13 + `INTRINSIC_DIM` minimal DOF)
///
/// ```text
/// params[0]: SE3  T_b0   — 6 DOF
/// params[1]: SE3  T_b1   — 6 DOF
/// params[2]: Rn(1) ρ     — 1 DOF
/// params[3]: Rn(N) K     — camera 1 intrinsics
/// ```
#[derive(Debug, Clone)]
pub struct StereoInverseRangeFactor<CAM> {
    observation: Observation,
    camera: PhantomData<CAM>,
}

impl<CAM> StereoInverseRangeFactor<CAM> {
    /// Create from the host's unit bearing, camera 1's pixel and the pixel
    /// noise.
    ///
    /// # Errors
    ///
    /// As [`InverseRangeProjectionFactor::new`].
    pub fn new(
        host_bearing: Vector3<f64>,
        measurement: Vector2<f64>,
        pixel_sigma: f64,
    ) -> CoreResult<Self> {
        Ok(Self {
            observation: Observation::new(host_bearing, measurement, pixel_sigma)?,
            camera: PhantomData,
        })
    }
}

impl<CAM> Factor for StereoInverseRangeFactor<CAM>
where
    CAM: CameraModel,
    for<'a> CAM: TryFrom<&'a [f64]>,
{
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let [t_b0, t_b1, rho, intrinsics] = params else {
            return;
        };
        let rho = rho[0];
        let Ok(camera) = CAM::try_from(intrinsics) else {
            Observation::write_invalid(residual, jacobian);
            return;
        };
        if !rho.is_finite() || rho <= 0.0 {
            Observation::write_invalid(residual, jacobian);
            return;
        }

        let (r_b0, p_b0) = extrinsic(t_b0);
        let (r_b1, p_b1) = extrinsic(t_b1);
        let m = self.observation.host_bearing;
        let f = m / rho;
        let point = r_b1.transpose() * (r_b0 * f + p_b0 - p_b1);

        let a = r_b1.transpose() * r_b0;
        let mut d = DMatrix::<f64>::zeros(3, STEREO_COLUMNS);
        d.fixed_view_mut::<3, 3>(0, 0).copy_from(&a);
        d.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-a * skew(&f)));
        d.fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&(-Matrix3::identity()));
        d.fixed_view_mut::<3, 3>(0, 9).copy_from(&skew(&point));
        d.fixed_view_mut::<3, 1>(0, 12)
            .copy_from(&(a * (-m / (rho * rho))));

        self.observation
            .write(&camera, &point, &d, residual, jacobian);
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (2, STEREO_COLUMNS + CAM::INTRINSIC_DIM)
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_block_sizes(
            variables,
            &[SE3::REP_SIZE, SE3::REP_SIZE, 1, CAM::INTRINSIC_DIM],
            "StereoInverseRangeFactor expects [SE3 T_b0, SE3 T_b1, Rn(1) rho, Rn(N) intrinsics]",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_camera_models::{DistortionModel, KannalaBrandtCamera, PinholeParams, RadTanCamera};
    use apex_manifolds::Tangent;
    use apex_manifolds::se3::SE3Tangent;
    use apex_manifolds::se23::{SE23, SE23Tangent};
    use nalgebra::{DVector, UnitQuaternion};

    type TestResult = Result<(), Box<dyn std::error::Error>>;
    type Fallible<T> = Result<T, Box<dyn std::error::Error>>;
    /// Unit host bearing, inverse range, and camera-1 pixel of a landmark.
    type Truth = (Vector3<f64>, f64, Vector2<f64>);
    /// Factor blocks, host bearing, and camera-1 pixel.
    type Blocks = (Vec<Vec<f64>>, Vector3<f64>, Vector2<f64>);

    fn se23(t: Vector3<f64>, v: Vector3<f64>, q: UnitQuaternion<f64>) -> Vec<f64> {
        SE23::new(t, v, q).as_param_slice().to_vec()
    }

    fn se3(t: Vector3<f64>, q: UnitQuaternion<f64>) -> Vec<f64> {
        SE3::new(t, q).as_param_slice().to_vec()
    }

    /// TUM VI-like rig: two downward-facing fisheyes about 10 cm apart.
    fn rig() -> (Vec<f64>, Vec<f64>) {
        let q = UnitQuaternion::from_euler_angles(1.55, 0.02, -3.1);
        (
            se3(Vector3::new(0.045, -0.071, -0.045), q),
            se3(
                Vector3::new(-0.056, -0.069, -0.047),
                q * UnitQuaternion::from_euler_angles(0.04, 0.001, 0.0008),
            ),
        )
    }

    fn kb() -> Result<KannalaBrandtCamera, Box<dyn std::error::Error>> {
        Ok(KannalaBrandtCamera::new(
            PinholeParams::new(190.4, 190.4, 252.6, 254.9)?,
            DistortionModel::KannalaBrandt {
                k1: 0.0034,
                k2: 0.0018,
                k3: -0.0027,
                k4: 0.00033,
            },
        )?)
    }

    fn radtan() -> Result<RadTanCamera, Box<dyn std::error::Error>> {
        Ok(RadTanCamera::new(
            PinholeParams::new(457.6, 456.1, 380.0, 255.2)?,
            DistortionModel::BrownConrady {
                k1: -0.2837,
                k2: 0.0745,
                p1: -0.0001,
                p2: -0.00004,
                k3: 0.0,
            },
        )?)
    }

    /// Tangent dimension of each block, in order.
    fn tangent_sizes(blocks: &[Vec<f64>]) -> Vec<usize> {
        blocks
            .iter()
            .map(|b| match b.len() {
                10 => 9,
                7 => 6,
                n => n,
            })
            .collect()
    }

    /// Perturb block `index` by `delta` along tangent coordinate `coord`.
    fn perturb(blocks: &[Vec<f64>], index: usize, coord: usize, delta: f64) -> Vec<Vec<f64>> {
        let mut out = blocks.to_vec();
        let block = &blocks[index];
        out[index] = match block.len() {
            10 => {
                let mut t = vec![0.0; 9];
                t[coord] = delta;
                SE23::from_param_slice(block)
                    .right_plus(&SE23Tangent::from_slice(&t), None, None)
                    .as_param_slice()
                    .to_vec()
            }
            7 => {
                let mut t = vec![0.0; 6];
                t[coord] = delta;
                SE3::from_param_slice(block)
                    .right_plus(&SE3Tangent::from_slice(&t), None, None)
                    .as_param_slice()
                    .to_vec()
            }
            _ => {
                let mut v = block.clone();
                v[coord] += delta;
                v
            }
        };
        out
    }

    fn evaluate(factor: &dyn Factor, blocks: &[Vec<f64>]) -> (Vec<f64>, DMatrix<f64>) {
        let refs: Vec<&[f64]> = blocks.iter().map(Vec::as_slice).collect();
        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0; rows];
        let mut jac = faer::Mat::<f64>::zeros(rows, cols);
        factor.linearize(&refs, &mut residual, Some(jac.as_mut()));
        let dense = DMatrix::from_fn(rows, cols, |r, c| jac[(r, c)]);
        (residual, dense)
    }

    /// Analytic Jacobian against central differences, column by column.
    fn check_jacobian(factor: &dyn Factor, blocks: &[Vec<f64>]) -> TestResult {
        let (_, analytic) = evaluate(factor, blocks);
        let h = 1e-6;
        let mut column = 0usize;
        for (index, size) in tangent_sizes(blocks).into_iter().enumerate() {
            for coord in 0..size {
                let (plus, _) = evaluate(factor, &perturb(blocks, index, coord, h));
                let (minus, _) = evaluate(factor, &perturb(blocks, index, coord, -h));
                for row in 0..2 {
                    let numeric = (plus[row] - minus[row]) / (2.0 * h);
                    let error = (numeric - analytic[(row, column)]).abs();
                    let scale = 1.0_f64.max(numeric.abs());
                    if error > 1e-4 * scale {
                        return Err(format!(
                            "block {index} coord {coord} row {row}: analytic {} numeric {numeric}",
                            analytic[(row, column)]
                        )
                        .into());
                    }
                }
                column += 1;
            }
        }
        Ok(())
    }

    /// A landmark at `world`, anchored in camera 0 of `host`, and its pixel in
    /// camera 1 of `observer`.
    fn truth<CAM: CameraModel>(
        camera: &CAM,
        host: &[f64],
        observer: &[f64],
        t_b0: &[f64],
        t_b1: &[f64],
        world: Vector3<f64>,
    ) -> Fallible<Truth> {
        let host = SE23::from_param_slice(host);
        let observer = SE23::from_param_slice(observer);
        let (r_b0, p_b0) = extrinsic(t_b0);
        let (r_b1, p_b1) = extrinsic(t_b1);
        let in_host = r_b0.transpose()
            * (host.rotation_matrix().transpose() * (world - host.translation()) - p_b0);
        let range = in_host.norm();
        let in_observer = r_b1.transpose()
            * (observer.rotation_matrix().transpose() * (world - observer.translation()) - p_b1);
        let pixel = camera.project(&in_observer)?;
        Ok((in_host / range, 1.0 / range, pixel))
    }

    fn moving_blocks<CAM: CameraModel>(camera: &CAM, world: Vector3<f64>) -> Fallible<Blocks>
    where
        for<'a> &'a CAM: Into<DVector<f64>>,
    {
        let host = se23(
            Vector3::new(0.1, -0.2, 1.2),
            Vector3::new(0.4, 0.1, -0.2),
            UnitQuaternion::from_euler_angles(0.03, -0.07, 0.11),
        );
        let observer = se23(
            Vector3::new(0.5, 0.15, 1.1),
            Vector3::new(0.3, -0.3, 0.05),
            UnitQuaternion::from_euler_angles(-0.05, 0.09, -0.04),
        );
        let (t_b0, t_b1) = rig();
        let (bearing, rho, pixel) = truth(camera, &host, &observer, &t_b0, &t_b1, world)?;
        let intrinsics: DVector<f64> = camera.into();
        Ok((
            vec![
                host,
                observer,
                t_b0,
                t_b1,
                vec![rho],
                intrinsics.as_slice().to_vec(),
            ],
            bearing,
            pixel,
        ))
    }

    #[test]
    fn moving_factor_is_zero_at_truth_and_its_jacobian_matches_kb() -> TestResult {
        let camera = kb()?;
        let (blocks, bearing, pixel) = moving_blocks(&camera, Vector3::new(0.8, 0.3, -1.5))?;
        let factor =
            InverseRangeProjectionFactor::<SE23, KannalaBrandtCamera>::new(bearing, pixel, 1.0)?;
        let (residual, _) = evaluate(&factor, &blocks);
        assert!(residual.iter().all(|r| r.abs() < 1e-8), "{residual:?}");
        // Off the truth, so the Jacobian is not checked at a degenerate point.
        let factor = InverseRangeProjectionFactor::<SE23, KannalaBrandtCamera>::new(
            bearing,
            pixel + Vector2::new(1.5, -2.0),
            0.7,
        )?;
        check_jacobian(&factor, &blocks)
    }

    #[test]
    fn moving_factor_jacobian_matches_radtan() -> TestResult {
        let camera = radtan()?;
        let (blocks, bearing, pixel) = moving_blocks(&camera, Vector3::new(0.3, 0.2, -1.8))?;
        let factor = InverseRangeProjectionFactor::<SE23, RadTanCamera>::new(bearing, pixel, 1.0)?;
        check_jacobian(&factor, &blocks)
    }

    #[test]
    fn stereo_factor_is_zero_at_truth_and_its_jacobian_matches() -> TestResult {
        let camera = kb()?;
        let (t_b0, t_b1) = rig();
        let body = se23(
            Vector3::zeros(),
            Vector3::zeros(),
            UnitQuaternion::identity(),
        );
        // Two metres in front of camera 0; the body sits at the world origin.
        let (r_b0, p_b0) = extrinsic(&t_b0);
        let world = r_b0 * Vector3::new(0.4, -0.3, 2.0) + p_b0;
        let (bearing, rho, pixel) = truth(&camera, &body, &body, &t_b0, &t_b1, world)?;
        let intrinsics: DVector<f64> = (&camera).into();
        let blocks = vec![t_b0, t_b1, vec![rho], intrinsics.as_slice().to_vec()];
        let factor = StereoInverseRangeFactor::<KannalaBrandtCamera>::new(bearing, pixel, 1.0)?;
        let (residual, _) = evaluate(&factor, &blocks);
        assert!(residual.iter().all(|r| r.abs() < 1e-8), "{residual:?}");
        check_jacobian(&factor, &blocks)
    }

    /// With both cameras the same, the two extrinsic blocks sum to the
    /// single-extrinsic block of the monocular factor's chain.
    #[test]
    fn extrinsic_blocks_sum_to_the_shared_extrinsic_chain() -> TestResult {
        let camera = kb()?;
        let (mut blocks, _, pixel) = moving_blocks(&camera, Vector3::new(0.8, 0.3, -1.5))?;
        blocks[3] = blocks[2].clone();
        let bearing = Vector3::new(0.1, -0.2, 1.0);
        let factor =
            InverseRangeProjectionFactor::<SE23, KannalaBrandtCamera>::new(bearing, pixel, 1.0)?;
        let (_, jac) = evaluate(&factor, &blocks);
        let summed = jac.columns(18, 6) + jac.columns(24, 6);
        // Numerically perturb the shared extrinsic, both blocks at once.
        let h = 1e-6;
        for coord in 0..6 {
            let step = |delta: f64| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
                let moved = perturb(&blocks, 2, coord, delta);
                let mut both = moved.clone();
                both[3] = moved[2].clone();
                Ok(evaluate(&factor, &both).0)
            };
            let (plus, minus) = (step(h)?, step(-h)?);
            for row in 0..2 {
                let numeric = (plus[row] - minus[row]) / (2.0 * h);
                assert!(
                    (numeric - summed[(row, coord)]).abs() < 1e-4 * 1.0_f64.max(numeric.abs()),
                    "coord {coord} row {row}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn a_non_positive_range_is_a_flat_penalty() -> TestResult {
        let camera = kb()?;
        let (mut blocks, bearing, pixel) = moving_blocks(&camera, Vector3::new(0.8, 0.3, -1.5))?;
        blocks[4] = vec![-0.1];
        let factor =
            InverseRangeProjectionFactor::<SE23, KannalaBrandtCamera>::new(bearing, pixel, 1.0)?;
        let (residual, jac) = evaluate(&factor, &blocks);
        assert_eq!(residual, vec![CHEIRALITY_BASE_PENALTY; 2]);
        assert!(jac.iter().all(|v| *v == 0.0));
        Ok(())
    }

    #[test]
    fn construction_rejects_bad_inputs() {
        let ok = Vector3::new(0.0, 0.0, 1.0);
        let pixel = Vector2::new(1.0, 2.0);
        assert!(
            StereoInverseRangeFactor::<KannalaBrandtCamera>::new(Vector3::zeros(), pixel, 1.0)
                .is_err()
        );
        assert!(StereoInverseRangeFactor::<KannalaBrandtCamera>::new(ok, pixel, 0.0).is_err());
        assert!(
            StereoInverseRangeFactor::<KannalaBrandtCamera>::new(
                ok,
                Vector2::new(f64::NAN, 0.0),
                1.0
            )
            .is_err()
        );
    }
}
