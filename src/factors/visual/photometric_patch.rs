//! Direct photometric residual on a LiDAR map point — the FAST-LIVO2 visual
//! update, generic over the pose group and the camera model.
//!
//! A map point `P_W` (from the LiDAR map, held constant) carries a reference
//! patch `R`. In the current image `I` its patch must look the same:
//!
//! ```text
//! c   = T_SC⁻¹ ⊙ (X⁻¹ ⊙ P_W)              (world → body → camera)
//! u   = π(c)
//! rᵢ  = ((I(u + δᵢ) − Ī) − (Rᵢ − R̄)) / σ    for each patch offset δᵢ
//! ```
//!
//! Both patches are zero-mean, so a brightness offset between the views
//! cancels. The patch is axis-aligned in the current image: a reference patch
//! seen from far away should be warped to the current view upstream (the
//! affine warp FAST-LIVO2 computes before its update).
//!
//! # Jacobian
//!
//! ```text
//! ∂rᵢ/∂(X, T_SC) = (∇Iᵢ − ∇̄I)ᵀ/σ · ∂π/∂c · ∂c/∂(X, T_SC)
//! ```
//!
//! with `∇I` the exact gradient of the bilinear interpolant and `∂c/∂·` from
//! the groups' `inverse` and `act`. A point behind the camera or a patch
//! leaving the image contributes nothing: zero residuals, zero Jacobian.

use faer::prelude::ReborrowMut;
use nalgebra::{DMatrix, Matrix2, SMatrix, Vector2, Vector3};
use std::marker::PhantomData;
use std::sync::Arc;

use apex_camera_models::CameraModel;
use apex_manifolds::LieGroup;
use apex_manifolds::capability::ActsOnPoints3;
use apex_manifolds::se3::SE3;

use crate::core::CoreError;
use crate::core::CoreResult;
use crate::core::variable::ManifoldVariable;
use crate::factors::common::group_point::into_anchor;
use crate::factors::common::validate::expect_manifolds;
use crate::factors::{Factor, NormalBlocks};

/// A single-channel image, row-major, sampled bilinearly with pixel centres
/// at integer coordinates.
#[derive(Clone, Debug)]
pub struct IntensityImage {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl IntensityImage {
    /// An image from row-major intensities.
    ///
    /// # Errors
    ///
    /// A size smaller than `2 × 2`, or `data` not `width · height` long.
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> CoreResult<Self> {
        if width < 2 || height < 2 {
            return Err(CoreError::InvalidInput(format!(
                "IntensityImage: {width}×{height} is too small to interpolate"
            )));
        }
        if data.len() != width * height {
            return Err(CoreError::InvalidInput(format!(
                "IntensityImage: {} values for {width}×{height} pixels",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// An image from a function of the pixel coordinates `(x, y)`.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn from_fn(
        width: usize,
        height: usize,
        intensity: impl Fn(usize, usize) -> f32,
    ) -> CoreResult<Self> {
        let data = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| intensity(x, y))
            .collect();
        Self::new(width, height, data)
    }

    /// Image width [px].
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height [px].
    pub fn height(&self) -> usize {
        self.height
    }

    fn at(&self, x: usize, y: usize) -> f64 {
        f64::from(self.data[y * self.width + x])
    }

    /// Bilinear intensity and its gradient `[∂I/∂u, ∂I/∂v]` at `(u, v)`, or
    /// `None` outside the area covered by four pixels.
    pub fn sample(&self, u: f64, v: f64) -> Option<(f64, Vector2<f64>)> {
        if !(u.is_finite() && v.is_finite()) || u < 0.0 || v < 0.0 {
            return None;
        }
        let (x0, y0) = (u.floor(), v.floor());
        let (x, y) = (x0 as usize, y0 as usize);
        if x + 1 >= self.width || y + 1 >= self.height {
            return None;
        }
        let (fx, fy) = (u - x0, v - y0);
        let (i00, i10) = (self.at(x, y), self.at(x + 1, y));
        let (i01, i11) = (self.at(x, y + 1), self.at(x + 1, y + 1));
        let top = i00 + fx * (i10 - i00);
        let bottom = i01 + fx * (i11 - i01);
        let value = top + fy * (bottom - top);
        let d_u = (1.0 - fy) * (i10 - i00) + fy * (i11 - i01);
        let d_v = bottom - top;
        Some((value, Vector2::new(d_u, d_v)))
    }
}

/// Photometric patch residual of one map point in one image.
///
/// Blocks `[X ∈ G, T_SC ∈ SE3]`: the image's state and the camera extrinsic
/// (camera → body). Residual dimension = number of patch pixels.
///
/// Whitens internally: register with `NoiseModel::null()`.
pub struct PhotometricPatchFactor<G: ActsOnPoints3, CAM: CameraModel> {
    camera: CAM,
    image: Arc<IntensityImage>,
    point_w: Vector3<f64>,
    offsets: Arc<[Vector2<f64>]>,
    /// The reference patch, made zero-mean at construction.
    reference: Vec<f64>,
    /// Intensity noise standard deviation.
    sigma: f64,
    group: PhantomData<fn() -> G>,
}

impl<G: ActsOnPoints3, CAM: CameraModel> PhotometricPatchFactor<G, CAM> {
    /// A map point's reference patch against `image`.
    ///
    /// `offsets[i]` is the pixel offset of patch sample `i` from the projected
    /// centre and `reference[i]` its reference intensity.
    ///
    /// # Errors
    ///
    /// A group without a 3-D point action, fewer than two samples, offsets and
    /// reference of different lengths, non-finite inputs, or a `sigma` that is
    /// not finite and positive.
    pub fn new(
        camera: CAM,
        image: Arc<IntensityImage>,
        point_w: Vector3<f64>,
        offsets: impl Into<Arc<[Vector2<f64>]>>,
        reference: &[f64],
        sigma: f64,
    ) -> CoreResult<Self> {
        let offsets = offsets.into();
        if offsets.len() < 2 || offsets.len() != reference.len() {
            return Err(CoreError::InvalidInput(format!(
                "PhotometricPatchFactor: {} offsets and {} reference values (need ≥ 2, equal)",
                offsets.len(),
                reference.len()
            )));
        }
        let finite = point_w.iter().all(|v| v.is_finite())
            && offsets.iter().all(|o| o.iter().all(|v| v.is_finite()))
            && reference.iter().all(|v| v.is_finite());
        if !finite {
            return Err(CoreError::InvalidInput(
                "PhotometricPatchFactor: non-finite input".to_string(),
            ));
        }
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(CoreError::InvalidInput(format!(
                "PhotometricPatchFactor: sigma must be finite and positive, got {sigma}"
            )));
        }
        let mean = reference.iter().sum::<f64>() / reference.len() as f64;
        Ok(Self {
            camera,
            image,
            point_w,
            offsets,
            reference: reference.iter().map(|value| value - mean).collect(),
            sigma,
            group: PhantomData,
        })
    }

    /// Offsets of a `size × size` patch centred on the projection, `step`
    /// pixels apart.
    pub fn square_offsets(size: usize, step: f64) -> Vec<Vector2<f64>> {
        let centre = (size as f64 - 1.0) / 2.0;
        (0..size)
            .flat_map(|row| (0..size).map(move |col| (row, col)))
            .map(|(row, col)| {
                Vector2::new((col as f64 - centre) * step, (row as f64 - centre) * step)
            })
            .collect()
    }
}

/// A patch sampled at a state: each sample's intensity and image gradient,
/// their means, and the pixel's derivative `∂c/∂[X, T_SC]` (one column per
/// tangent axis, the state's then the extrinsic's).
struct Patch {
    samples: Vec<(f64, Vector2<f64>)>,
    mean_value: f64,
    mean_gradient: Vector2<f64>,
    d_pixel: Vec<Vector2<f64>>,
}

impl<G: ActsOnPoints3, CAM: CameraModel> PhotometricPatchFactor<G, CAM> {
    /// The patch at `params`; `None` when the point does not project or a
    /// sample leaves the image.
    fn patch(&self, params: &[&[f64]], with_derivative: bool) -> Option<Patch> {
        let state = G::from_param_slice(params[0]);
        let extrinsic = SE3::from_param_slice(params[1]);
        let body = into_anchor(&state, &self.point_w);
        let cam = into_anchor(&extrinsic, &body.point);
        let centre = self.camera.project(&cam.point).ok()?;
        let samples: Vec<(f64, Vector2<f64>)> = self
            .offsets
            .iter()
            .map(|offset| self.image.sample(centre.x + offset.x, centre.y + offset.y))
            .collect::<Option<_>>()?;
        let count = samples.len() as f64;
        let mean_value = samples.iter().map(|(value, _)| value).sum::<f64>() / count;
        let mean_gradient = samples
            .iter()
            .fold(Vector2::zeros(), |sum, (_, gradient)| sum + gradient)
            / count;
        let d_pixel = if with_derivative {
            let d = self.camera.jacobian_point(&cam.point);
            let d = SMatrix::<f64, 2, 3>::from_fn(|row, col| d[(row, col)]);
            // ∂c/∂X = ∂c/∂s · ∂s/∂X, then ∂c/∂T_SC.
            body.d_anchor
                .iter()
                .map(|column| cam.d_world * Vector3::from(*column))
                .chain(cam.d_anchor.iter().map(|column| Vector3::from(*column)))
                .map(|column| d * column)
                .collect()
        } else {
            Vec::new()
        };
        Some(Patch {
            samples,
            mean_value,
            mean_gradient,
            d_pixel,
        })
    }

    /// Sample `k`'s whitened residual.
    fn residual(&self, patch: &Patch, k: usize) -> f64 {
        (patch.samples[k].0 - patch.mean_value - self.reference[k]) / self.sigma
    }
}

/// No usable patch: zero residuals and an all-zero Jacobian.
fn zero_out(residual: &mut [f64], jacobian: Option<faer::mat::MatMut<'_, f64>>) {
    residual.fill(0.0);
    if let Some(mut jac) = jacobian {
        for row in 0..jac.nrows() {
            for col in 0..jac.ncols() {
                *jac.rb_mut().get_mut(row, col) = 0.0;
            }
        }
    }
}

impl<G, CAM> Factor for PhotometricPatchFactor<G, CAM>
where
    G: ActsOnPoints3 + Send + Sync + 'static,
    CAM: CameraModel,
{
    fn linearize(
        &self,
        params: &[&[f64]],
        residual: &mut [f64],
        jacobian: Option<faer::mat::MatMut<'_, f64>>,
    ) {
        let Some(patch) = self.patch(params, jacobian.is_some()) else {
            zero_out(residual, jacobian);
            return;
        };
        for (k, r) in residual.iter_mut().enumerate() {
            *r = self.residual(&patch, k);
        }
        let Some(mut jac) = jacobian else {
            return;
        };
        for (row, (_, gradient)) in patch.samples.iter().enumerate() {
            let weight = (gradient - patch.mean_gradient) / self.sigma;
            for (col, column) in patch.d_pixel.iter().enumerate() {
                *jac.rb_mut().get_mut(row, col) = weight.dot(column);
            }
        }
    }

    /// Every row is `wᵣᵀ·D` with `wᵣ` the sample's centred gradient over σ
    /// and `D = ∂c/∂[X, T_SC]` shared: `JᵀJ = Dᵀ(Σ wᵣwᵣᵀ)D`,
    /// `Jᵀr = Dᵀ(Σ wᵣrᵣ)` — a 2 × 2 sum instead of the patch's rows.
    fn accumulate_normal(
        &self,
        params: &[&[f64]],
        blocks: Option<&mut NormalBlocks<'_>>,
    ) -> Option<f64> {
        let Some(patch) = self.patch(params, blocks.is_some()) else {
            return Some(0.0);
        };
        let mut cost = 0.0;
        let mut spread = Matrix2::zeros();
        let mut pull = Vector2::zeros();
        for (k, (_, gradient)) in patch.samples.iter().enumerate() {
            let r = self.residual(&patch, k);
            cost += r * r;
            let weight = (gradient - patch.mean_gradient) / self.sigma;
            spread += weight * weight.transpose();
            pull += weight * r;
        }
        if let Some(blocks) = blocks {
            let d = DMatrix::from_fn(2, patch.d_pixel.len(), |r, c| patch.d_pixel[c][r]);
            let dof = G::DOF;
            let parts = [d.columns(0, dof), d.columns(dof, d.ncols() - dof)];
            for (a, da) in parts.iter().enumerate() {
                blocks.add_gradient(a, &(da.transpose() * pull));
                for (b, db) in parts.iter().enumerate() {
                    blocks.add_hessian(a, b, &(da.transpose() * spread * db));
                }
            }
        }
        Some(cost)
    }

    fn residual_dim(&self) -> usize {
        self.offsets.len()
    }

    fn jacobian_shape(&self) -> (usize, usize) {
        (self.offsets.len(), G::DOF + 6)
    }

    fn whitens_internally(&self) -> bool {
        true
    }

    fn validate_variables(&self, variables: &[&dyn ManifoldVariable]) -> Result<(), String> {
        expect_manifolds(
            variables,
            &[G::NAME, "SE3"],
            "PhotometricPatchFactor expects [state, SE3 T_SC]",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apex_camera_models::pinhole::PinholeCamera;
    use apex_manifolds::Tangent;
    use apex_manifolds::se23::SE23;
    use apex_manifolds::sim3::Sim3;
    use faer::Mat;
    use nalgebra::UnitQuaternion;

    fn texture(x: f64, y: f64) -> f64 {
        100.0 + 40.0 * (0.15 * x).sin() + 30.0 * (0.11 * y).cos() + 20.0 * (0.05 * (x + y)).sin()
    }

    fn image() -> Result<Arc<IntensityImage>, CoreError> {
        IntensityImage::from_fn(640, 480, |x, y| texture(x as f64, y as f64) as f32).map(Arc::new)
    }

    fn camera() -> PinholeCamera {
        PinholeCamera::from([500.0, 500.0, 320.0, 240.0])
    }

    fn extrinsic() -> SE3 {
        SE3::new(
            Vector3::new(0.05, -0.02, 0.01),
            UnitQuaternion::from_euler_angles(0.01, -0.02, 0.03),
        )
    }

    fn rotation() -> UnitQuaternion<f64> {
        UnitQuaternion::from_euler_angles(0.1, -0.2, 0.4)
    }

    /// A world point seen at camera coordinates `(0.3, −0.2, 4.0)` from `state`.
    fn world_point<G: ActsOnPoints3>(state: &G) -> Vector3<f64> {
        let body = extrinsic().act(&Vector3::new(0.3, -0.2, 4.0), None, None);
        state.act(&body, None, None)
    }

    /// The reference patch: the image itself at the true projection, plus a
    /// brightness offset the residual must ignore.
    fn factor<G: ActsOnPoints3>(
        state: &G,
        offset: f64,
    ) -> Result<PhotometricPatchFactor<G, PinholeCamera>, CoreError> {
        let image = image()?;
        let point_w = world_point(state);
        let c = Vector3::new(0.3, -0.2, 4.0);
        let centre = camera()
            .project(&c)
            .map_err(|e| CoreError::InvalidInput(e.to_string()))?;
        let offsets = PhotometricPatchFactor::<G, PinholeCamera>::square_offsets(4, 1.0);
        let reference: Vec<f64> = offsets
            .iter()
            .map(|o| {
                image
                    .sample(centre.x + o.x, centre.y + o.y)
                    .map(|(v, _)| v + offset)
            })
            .collect::<Option<_>>()
            .ok_or_else(|| CoreError::InvalidInput("reference outside the image".to_string()))?;
        PhotometricPatchFactor::new(camera(), image, point_w, offsets, &reference, 2.0)
    }

    fn evaluate<G: ActsOnPoints3 + Send + Sync + 'static>(
        factor: &PhotometricPatchFactor<G, PinholeCamera>,
        state: &G,
        extrinsic: &SE3,
    ) -> (Vec<f64>, Mat<f64>) {
        let s = state.as_param_slice().to_vec();
        let e = extrinsic.as_param_slice().to_vec();
        let (rows, cols) = factor.jacobian_shape();
        let mut residual = vec![0.0; rows];
        let mut jacobian = Mat::<f64>::zeros(rows, cols);
        factor.linearize(&[&s, &e], &mut residual, Some(jacobian.as_mut()));
        (residual, jacobian)
    }

    fn central_difference<G: ActsOnPoints3>(
        at: &G,
        axis: usize,
        evaluate: impl Fn(&G) -> Vec<f64>,
    ) -> Vec<f64> {
        const EPS: f64 = 1e-6;
        let mut tangent = vec![0.0; G::DOF];
        tangent[axis] = EPS;
        let plus = evaluate(&at.right_plus(&G::TangentVector::from_slice(&tangent), None, None));
        tangent[axis] = -EPS;
        let minus = evaluate(&at.right_plus(&G::TangentVector::from_slice(&tangent), None, None));
        plus.iter()
            .zip(&minus)
            .map(|(p, m)| (p - m) / (2.0 * EPS))
            .collect()
    }

    /// Zero at truth despite a brightness offset; Jacobians equal central
    /// differences, off truth, in group `G`'s own chart.
    fn check_group<G: ActsOnPoints3 + Send + Sync + 'static>(state: G) -> Result<(), CoreError> {
        let dof = G::DOF;
        let factor = factor(&state, 7.0)?;
        let (residual, _) = evaluate(&factor, &state, &extrinsic());
        assert!(
            residual.iter().all(|r| r.abs() < 1e-9),
            "{}: {residual:?}",
            G::NAME
        );

        let nudge: Vec<f64> = (0..dof).map(|i| 1e-3 * (1.0 + i as f64)).collect();
        let off = state.right_plus(&G::TangentVector::from_slice(&nudge), None, None);
        let (residual, analytic) = evaluate(&factor, &off, &extrinsic());
        assert!(
            residual.iter().any(|r| r.abs() > 1e-3),
            "{}: no signal off truth",
            G::NAME
        );
        for axis in 0..dof {
            let fd = central_difference(&off, axis, |x| evaluate(&factor, x, &extrinsic()).0);
            for (row, value) in fd.iter().enumerate() {
                let a = analytic[(row, axis)];
                assert!(
                    (value - a).abs() < 1e-4 * (1.0 + a.abs()),
                    "{} state {axis} row {row}: fd {value} vs {a}",
                    G::NAME
                );
            }
        }
        crate::factors::assert_normal_equations_match(
            &factor,
            &[off.as_param_slice(), extrinsic().as_param_slice()],
            &[dof, 6],
            1e-9,
        );
        for axis in 0..6 {
            let fd = central_difference(&extrinsic(), axis, |e| evaluate(&factor, &off, e).0);
            for (row, value) in fd.iter().enumerate() {
                let a = analytic[(row, dof + axis)];
                assert!(
                    (value - a).abs() < 1e-4 * (1.0 + a.abs()),
                    "{} extrinsic {axis} row {row}: fd {value} vs {a}",
                    G::NAME
                );
            }
        }
        Ok(())
    }

    /// The odometry's stack — the extrinsic held, the image's state offset
    /// from the sweep's — keeps exact normal equations through both
    /// wrappers.
    #[test]
    fn the_wrapped_patch_keeps_its_normal_equations() -> Result<(), CoreError> {
        use crate::factors::held::HeldBlocks;
        use crate::factors::offset::RightOffset;
        let state = SE23::new(
            Vector3::new(1.0, -0.5, 0.3),
            Vector3::new(0.4, 0.1, 0.0),
            rotation(),
        );
        let delta = SE23::new(
            Vector3::new(0.01, -0.004, 0.002),
            Vector3::zeros(),
            UnitQuaternion::from_euler_angles(0.003, -0.002, 0.01),
        );
        let nudge = [2e-3, -1e-3, 1e-3, 0.0, 0.0, 0.0, 3e-3, 1e-3, -2e-3];
        let off = state.right_plus(
            &<SE23 as LieGroup>::TangentVector::from_slice(&nudge),
            None,
            None,
        );
        let stack = HeldBlocks::new(
            RightOffset::new(factor(&state, 7.0)?, 0, 0, delta)?,
            vec![SE23::DOF, 6],
            vec![None, Some(extrinsic().as_param_slice().to_vec())],
        )?;
        crate::factors::assert_normal_equations_match(&stack, &[off.as_param_slice()], &[9], 1e-9);
        Ok(())
    }

    #[test]
    fn se3_poses() -> Result<(), CoreError> {
        check_group(SE3::new(Vector3::new(1.0, -0.5, 0.3), rotation()))
    }

    #[test]
    fn se23_navigation_states() -> Result<(), CoreError> {
        check_group(SE23::new(
            Vector3::new(1.0, -0.5, 0.3),
            Vector3::new(0.4, 0.1, 0.0),
            rotation(),
        ))
    }

    #[test]
    fn sim3_similarities() -> Result<(), CoreError> {
        check_group(Sim3::new(Vector3::new(1.0, -0.5, 0.3), rotation(), 1.1))
    }

    /// A uniform brightness change of the current image leaves the residual.
    #[test]
    fn a_brightness_offset_cancels() -> Result<(), CoreError> {
        let state = SE3::new(Vector3::new(1.0, -0.5, 0.3), rotation());
        let off = state.right_plus(
            &apex_manifolds::se3::SE3Tangent::from_slice(&[0.01, 0.0, 0.0, 0.0, 0.002, 0.0]),
            None,
            None,
        );
        let a = evaluate(&factor(&state, 0.0)?, &off, &extrinsic()).0;
        let b = evaluate(&factor(&state, 25.0)?, &off, &extrinsic()).0;
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-9));
        Ok(())
    }

    #[test]
    fn a_point_behind_the_camera_or_off_the_image_contributes_nothing() -> Result<(), CoreError> {
        let state = SE3::new(Vector3::new(1.0, -0.5, 0.3), rotation());
        let factor = factor(&state, 0.0)?;
        // Turned around: the point is behind the camera.
        let behind = SE3::new(
            state.translation(),
            state.rotation_quaternion()
                * UnitQuaternion::from_euler_angles(std::f64::consts::PI, 0.0, 0.0),
        );
        // Turned sideways: the point projects off the image.
        let aside = SE3::new(
            state.translation(),
            state.rotation_quaternion() * UnitQuaternion::from_euler_angles(0.0, 0.9, 0.0),
        );
        for pose in [behind, aside] {
            let s = pose.as_param_slice().to_vec();
            let e = extrinsic().as_param_slice().to_vec();
            let mut residual = vec![9.0; 16];
            let mut jacobian = Mat::<f64>::from_fn(16, 12, |_, _| 9.0);
            factor.linearize(&[&s, &e], &mut residual, Some(jacobian.as_mut()));
            assert!(residual.iter().all(|r| *r == 0.0));
            assert!((0..16).all(|r| (0..12).all(|c| jacobian[(r, c)] == 0.0)));
        }
        Ok(())
    }

    #[test]
    fn bilinear_sampling_is_exact_on_pixels_and_has_the_right_gradient() -> Result<(), CoreError> {
        let image = image()?;
        let (value, _) = image
            .sample(100.0, 50.0)
            .ok_or_else(|| CoreError::InvalidInput("outside".to_string()))?;
        assert!((value - f64::from(texture(100.0, 50.0) as f32)).abs() < 1e-9);
        let (u, v, eps) = (123.37, 77.61, 1e-6);
        let (_, gradient) = image
            .sample(u, v)
            .ok_or_else(|| CoreError::InvalidInput("outside".to_string()))?;
        let value_at = |u: f64, v: f64| {
            image
                .sample(u, v)
                .map(|(value, _)| value)
                .ok_or_else(|| CoreError::InvalidInput("outside".to_string()))
        };
        let d_u = (value_at(u + eps, v)? - value_at(u - eps, v)?) / (2.0 * eps);
        let d_v = (value_at(u, v + eps)? - value_at(u, v - eps)?) / (2.0 * eps);
        assert!((gradient.x - d_u).abs() < 1e-5 && (gradient.y - d_v).abs() < 1e-5);
        assert!(image.sample(-0.1, 10.0).is_none());
        assert!(image.sample(639.0, 10.0).is_none());
        assert!(image.sample(f64::NAN, 10.0).is_none());
        assert_eq!((image.width(), image.height()), (640, 480));
        Ok(())
    }

    #[test]
    fn bad_inputs_are_rejected() -> Result<(), CoreError> {
        assert!(IntensityImage::new(1, 5, vec![0.0; 5]).is_err());
        assert!(IntensityImage::new(4, 4, vec![0.0; 15]).is_err());
        let image = image()?;
        let offsets = PhotometricPatchFactor::<SE3, PinholeCamera>::square_offsets(2, 1.0);
        let new = |offsets: Vec<Vector2<f64>>, reference: &[f64], sigma: f64| {
            PhotometricPatchFactor::<SE3, PinholeCamera>::new(
                camera(),
                image.clone(),
                Vector3::z(),
                offsets,
                reference,
                sigma,
            )
        };
        assert!(new(offsets.clone(), &[1.0, 2.0, 3.0, 4.0], 1.0).is_ok());
        assert!(new(offsets.clone(), &[1.0, 2.0, 3.0], 1.0).is_err());
        assert!(new(offsets.clone(), &[1.0, 2.0, 3.0, f64::NAN], 1.0).is_err());
        assert!(new(offsets.clone(), &[1.0, 2.0, 3.0, 4.0], 0.0).is_err());
        assert!(new(vec![Vector2::zeros()], &[1.0], 1.0).is_err());
        Ok(())
    }

    #[test]
    fn square_offsets_are_centred() {
        let offsets = PhotometricPatchFactor::<SE3, PinholeCamera>::square_offsets(4, 2.0);
        assert_eq!(offsets.len(), 16);
        let sum = offsets.iter().fold(Vector2::zeros(), |s, o| s + o);
        assert!(sum.norm() < 1e-12);
        assert_eq!(offsets[0], Vector2::new(-3.0, -3.0));
    }
}
