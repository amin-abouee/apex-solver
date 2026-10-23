//! `Fisheye624`: Project Aria's fisheye model with radial, tangential and
//! thin-prism distortion.
//!
//! The lens of every Project Aria SLAM and RGB camera, and so of the LaMAria
//! dataset's `RAD_TAN_THIN_PRISM_FISHEYE`. It has 16 intrinsic parameters
//! here: `fx fy cx cy`, six radial terms `k0..k5`, two tangential terms
//! `p0 p1`, and four thin-prism terms `s0..s3`.
//!
//! # Projection
//!
//! For a camera-frame point `(x, y, z)`, with `ρ = √(x² + y²)` and
//! `θ = atan2(ρ, z)`:
//!
//! ```text
//! R(θ)      = θ + k0 θ³ + k1 θ⁵ + k2 θ⁷ + k3 θ⁹ + k4 θ¹¹ + k5 θ¹³
//! (xr, yr)  = R(θ)/ρ · (x, y)                       the radially distorted point
//! rd²       = xr² + yr²
//! tangential = ( p0 (2 xr² + rd²) + 2 p1 xr yr ,
//!                p1 (2 yr² + rd²) + 2 p0 xr yr )
//! thin prism = ( s0 rd² + s1 rd⁴ ,  s2 rd² + s3 rd⁴ )
//! u = fx (xr + tangential.x + thin_prism.x) + cx
//! v = fy (yr + tangential.y + thin_prism.y) + cy
//! ```
//!
//! [`Fisheye62Camera`] is the same model without the
//! thin-prism terms.
//!
//! # Unprojection
//!
//! Two Newton iterations, as in the reference: first `(xr, yr)` from the
//! normalized pixel by inverting the tangential and thin-prism offsets, then θ
//! from `|(xr, yr)| = R(θ)`. A pixel outside the model's domain (θ at or past
//! 90°) is an error, never a clamped ray.
//!
//! # Focal lengths
//!
//! Project Aria's own parameter vector has one focal length `f` and 15
//! entries (`f cx cy k0..k5 p0 p1 s0..s3`); the LaMAria calibration files
//! repeat it as `fx fy`. This type keeps both, which is the 16-entry layout, so
//! the Aria layout is the special case `fx == fy`: see
//! [`Fisheye624Camera::from_aria_params`] and
//! [`Fisheye624Camera::to_aria_params`].
//!
//! # Source
//!
//! Model definition, parameter order and the Newton inversions follow Meta's
//! Project Aria tools:
//!
//! * Camera intrinsic models:
//!   <https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/camera_intrinsic_models>
//! * `core/calibration/camera_projections/FisheyeRadTanThinPrism.h`
//!   (`Fisheye624 = FisheyeRadTanThinPrism<6, true, true, true>`) in
//!   <https://github.com/facebookresearch/projectaria_tools>
//!
//! This is an independent implementation in Rust with analytic Jacobians; the
//! tests check it against values computed from the reference's formulas.

use crate::fisheye_core::{Core, NUM_K};
use crate::{
    CameraModel, CameraModelError, DistortionModel, Fisheye62Camera, KannalaBrandtCamera,
    PinholeParams,
};
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Number of intrinsic parameters: `fx fy cx cy k0..k5 p0 p1 s0..s3`.
const DIM: usize = 16;

/// Number of entries in Project Aria's single-focal-length vector.
const ARIA_DIM: usize = 15;

/// Project Aria `Fisheye624` camera, with 16 intrinsic parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fisheye624Camera {
    pub pinhole: PinholeParams,
    pub distortion: DistortionModel,
}

impl Fisheye624Camera {
    /// Creates a camera.
    ///
    /// # Errors
    ///
    /// [`CameraModelError::InvalidParams`] if `distortion` is not
    /// [`DistortionModel::Fisheye624`], and the errors of
    /// [`CameraModel::validate_params`].
    ///
    /// # Example
    ///
    /// ```
    /// use apex_camera_models::{CameraModel, DistortionModel, Fisheye624Camera, PinholeParams};
    ///
    /// let pinhole = PinholeParams::new(241.6, 241.6, 322.9, 240.4)?;
    /// let distortion = DistortionModel::Fisheye624 {
    ///     k: [-0.0248, 0.0963, -0.0633, 0.0062, 0.0035, -0.0007],
    ///     p: [0.0009, 0.0004],
    ///     s: [-0.0014, -0.0001, -0.0005, -0.0001],
    /// };
    /// let camera = Fisheye624Camera::new(pinhole, distortion)?;
    /// assert_eq!(camera.get_model_name(), "fisheye624");
    /// # Ok::<(), apex_camera_models::CameraModelError>(())
    /// ```
    pub fn new(
        pinhole: PinholeParams,
        distortion: DistortionModel,
    ) -> Result<Self, CameraModelError> {
        if !matches!(distortion, DistortionModel::Fisheye624 { .. }) {
            return Err(CameraModelError::InvalidParams(
                "Fisheye624Camera requires DistortionModel::Fisheye624".to_string(),
            ));
        }
        let model = Self {
            pinhole,
            distortion,
        };
        model.validate_params()?;
        Ok(model)
    }

    /// The camera from Project Aria's own 15-entry vector
    /// `[f, cx, cy, k0..k5, p0, p1, s0..s3]`, with `fx = fy = f`.
    ///
    /// # Errors
    ///
    /// [`CameraModelError::InvalidParams`] unless there are exactly 15 entries,
    /// and the errors of [`CameraModel::validate_params`].
    pub fn from_aria_params(params: &[f64]) -> Result<Self, CameraModelError> {
        if params.len() != ARIA_DIM {
            return Err(CameraModelError::InvalidParams(format!(
                "Fisheye624 in Project Aria's layout has {ARIA_DIM} parameters, got {}",
                params.len()
            )));
        }
        let mut native = [0.0; DIM];
        native[0] = params[0];
        native[1] = params[0];
        native[2..].copy_from_slice(&params[1..]);
        try_from_params(&native)
    }

    /// The camera as Project Aria's 15-entry vector
    /// `[f, cx, cy, k0..k5, p0, p1, s0..s3]`.
    ///
    /// # Errors
    ///
    /// [`CameraModelError::InvalidParams`] if `fx` and `fy` differ: Aria's
    /// layout cannot hold two focal lengths.
    pub fn to_aria_params(&self) -> Result<[f64; ARIA_DIM], CameraModelError> {
        let (fx, fy) = (self.pinhole.fx, self.pinhole.fy);
        if (fx - fy).abs() > 1e-9 * fx.abs().max(fy.abs()) {
            return Err(CameraModelError::InvalidParams(format!(
                "Project Aria's layout has one focal length, but fx={fx} and fy={fy}"
            )));
        }
        let native = <[f64; DIM]>::from(self);
        let mut aria = [0.0; ARIA_DIM];
        aria[0] = fx;
        aria[1..].copy_from_slice(&native[2..]);
        Ok(aria)
    }

    fn core(&self) -> Core {
        let (k, p, s) = match self.distortion {
            DistortionModel::Fisheye624 { k, p, s } => (k, p, s),
            _ => ([0.0; NUM_K], [0.0; 2], [0.0; 4]),
        };
        Core {
            fx: self.pinhole.fx,
            fy: self.pinhole.fy,
            cx: self.pinhole.cx,
            cy: self.pinhole.cy,
            k,
            p,
            s,
        }
    }
}

/// `[fx, fy, cx, cy, k0..k5, p0, p1, s0..s3]`.
impl From<&Fisheye624Camera> for [f64; DIM] {
    fn from(camera: &Fisheye624Camera) -> Self {
        let core = camera.core();
        let mut out = [0.0; DIM];
        out[..4].copy_from_slice(&[core.fx, core.fy, core.cx, core.cy]);
        out[4..10].copy_from_slice(&core.k);
        out[10..12].copy_from_slice(&core.p);
        out[12..].copy_from_slice(&core.s);
        out
    }
}

/// `[fx, fy, cx, cy, k0..k5, p0, p1, s0..s3]`.
impl From<&Fisheye624Camera> for DVector<f64> {
    fn from(camera: &Fisheye624Camera) -> Self {
        DVector::from_row_slice(&<[f64; DIM]>::from(camera))
    }
}

/// From `[fx, fy, cx, cy, k0..k5, p0, p1, s0..s3]`; at least 16 entries.
impl TryFrom<&[f64]> for Fisheye624Camera {
    type Error = CameraModelError;

    fn try_from(params: &[f64]) -> Result<Self, Self::Error> {
        if params.len() < DIM {
            return Err(CameraModelError::InvalidParams(format!(
                "Fisheye624Camera requires at least {DIM} parameters, got {}",
                params.len()
            )));
        }
        let mut k = [0.0; NUM_K];
        k.copy_from_slice(&params[4..10]);
        let mut p = [0.0; 2];
        p.copy_from_slice(&params[10..12]);
        let mut s = [0.0; 4];
        s.copy_from_slice(&params[12..16]);
        Ok(Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::Fisheye624 { k, p, s },
        })
    }
}

/// From `[fx, fy, cx, cy, k0..k5, p0, p1, s0..s3]`.
impl From<[f64; DIM]> for Fisheye624Camera {
    fn from(params: [f64; DIM]) -> Self {
        let mut k = [0.0; NUM_K];
        k.copy_from_slice(&params[4..10]);
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::Fisheye624 {
                k,
                p: [params[10], params[11]],
                s: [params[12], params[13], params[14], params[15]],
            },
        }
    }
}

/// Exact: a Kannala-Brandt camera is `Fisheye624` with `k0..k3 = k1..k4` and
/// no `k4`, `k5`, tangential or thin-prism terms.
impl From<&KannalaBrandtCamera> for Fisheye624Camera {
    fn from(camera: &KannalaBrandtCamera) -> Self {
        Fisheye624Camera::from(&Fisheye62Camera::from(camera))
    }
}

/// Exact: `Fisheye62` is `Fisheye624` with the thin-prism terms zero.
impl From<&Fisheye62Camera> for Fisheye624Camera {
    fn from(camera: &Fisheye62Camera) -> Self {
        let native = <[f64; 12]>::from(camera);
        let mut out = [0.0; DIM];
        out[..12].copy_from_slice(&native);
        Fisheye624Camera::from(out)
    }
}

/// Builds a camera from a parameter slice and validates it.
///
/// # Errors
///
/// See [`Fisheye624Camera::try_from`] and [`CameraModel::validate_params`].
pub fn try_from_params(params: &[f64]) -> Result<Fisheye624Camera, CameraModelError> {
    let camera = Fisheye624Camera::try_from(params)?;
    camera.validate_params()?;
    Ok(camera)
}

impl CameraModel for Fisheye624Camera {
    const INTRINSIC_DIM: usize = DIM;
    type IntrinsicJacobian = SMatrix<f64, 2, DIM>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a camera-frame point; [`CameraModelError::PointBehindCamera`]
    /// when `z < GEOMETRIC_PRECISION`.
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        self.core().project(p_cam)
    }

    /// Unprojects a pixel to a unit ray; an error outside the model's domain
    /// or when a Newton inversion does not converge.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        self.core().unproject(point_2d)
    }

    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        self.core().jacobian_point(p_cam)
    }

    /// 2×16 Jacobian with respect to `[fx fy cx cy k0..k5 p0 p1 s0..s3]`.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        self.core().jacobian_params(p_cam)
    }

    /// Validation: `fx`, `fy` positive and finite, `cx`, `cy` and every
    /// distortion term finite.
    fn validate_params(&self) -> Result<(), CameraModelError> {
        self.pinhole.validate()?;
        self.distortion.validate()
    }

    fn get_pinhole_params(&self) -> PinholeParams {
        self.pinhole
    }

    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// `"fisheye624"`.
    fn get_model_name(&self) -> &'static str {
        "fisheye624"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// LaMAria `R_06_medium` `cam0`, from `aria_calibrations/R_06_medium.json`.
    pub(crate) const LAMARIA_CAM0: [f64; DIM] = [
        241.604_892_556_728_4,
        241.604_892_556_728_4,
        322.895_555_771_704,
        240.444_559_956_063_3,
        -0.024_842_433_765_014_98,
        0.096_289_387_521_345_35,
        -0.063_325_466_945_624,
        0.006_200_726_341_569_274,
        0.003_488_761_399_963_396,
        -0.000_729_752_005_685_647_9,
        0.000_895_320_322_159_04,
        0.000_361_066_580_790_014,
        -0.001_368_792_443_870_01,
        -0.000_105_408_341_585_353_1,
        -0.000_463_561_129_848_821_3,
        -0.000_104_985_646_126_264_3,
    ];

    fn camera() -> Result<Fisheye624Camera, CameraModelError> {
        try_from_params(&LAMARIA_CAM0)
    }

    /// Computed with a line-by-line transcription of
    /// `FisheyeRadTanThinPrism<6, true, true, true>::project` in
    /// projectaria_tools, on [`LAMARIA_CAM0`]: not with this crate.
    const REFERENCE: [([f64; 3], [f64; 2]); 5] = [
        ([0.0, 0.0, 2.0], [322.895_555_771_704, 240.444_559_956_063]),
        ([0.3, -0.2, 1.0], [392.354_235_539_084, 194.126_113_883_996]),
        ([1.5, 0.9, 1.0], [545.019_614_472_973, 373.756_040_135_945]),
        ([-2.5, 1.0, 0.8], [32.459_742_637_747, 356.400_202_195_756]),
        (
            [0.05, 0.02, 3.0],
            [326.921_942_052_962, 242.055_121_230_192],
        ),
    ];

    #[test]
    fn projection_matches_the_reference_formulas() -> TestResult {
        let camera = camera()?;
        for (point, expected) in REFERENCE {
            let uv = camera.project(&Vector3::from(point))?;
            assert!(
                (uv.x - expected[0]).abs() < 1e-8 && (uv.y - expected[1]).abs() < 1e-8,
                "{point:?}: got {uv:?}, expected {expected:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn the_optical_axis_projects_to_the_principal_point() -> TestResult {
        let uv = camera()?.project(&Vector3::new(0.0, 0.0, 3.0))?;
        assert!((uv.x - LAMARIA_CAM0[2]).abs() < 1e-12);
        assert!((uv.y - LAMARIA_CAM0[3]).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn a_point_behind_the_camera_is_an_error() -> TestResult {
        let camera = camera()?;
        assert!(matches!(
            camera.project(&Vector3::new(0.1, 0.1, -1.0)),
            Err(CameraModelError::PointBehindCamera { .. })
        ));
        assert!(camera.project(&Vector3::new(1.0, 0.0, 0.0)).is_err());
        Ok(())
    }

    #[test]
    fn project_then_unproject_recovers_the_ray() -> TestResult {
        let camera = camera()?;
        let mut worst = 0.0f64;
        for theta_deg in [0.01, 1.0, 10.0, 30.0, 55.0, 70.0, 80.0, 85.0] {
            for phi_deg in (0..360).step_by(30) {
                let (theta, phi) = (f64::to_radians(theta_deg), f64::to_radians(phi_deg as f64));
                let ray = Vector3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                );
                let back = camera.unproject(&camera.project(&(ray * 2.5))?)?;
                worst = worst.max((back - ray).norm());
            }
        }
        assert!(worst < 1e-9, "worst ray error {worst}");
        Ok(())
    }

    #[test]
    fn a_pixel_far_outside_the_domain_is_an_error() -> TestResult {
        assert!(camera()?.unproject(&Vector2::new(5.0e4, 5.0e4)).is_err());
        assert!(camera()?.unproject(&Vector2::new(f64::NAN, 0.0)).is_err());
        Ok(())
    }

    fn points() -> [Vector3<f64>; 6] {
        [
            Vector3::new(0.3, -0.2, 1.0),
            Vector3::new(1.5, 0.9, 1.0),
            Vector3::new(-2.5, 1.0, 0.8),
            Vector3::new(0.05, 0.02, 3.0),
            Vector3::new(-0.7, -0.6, 1.2),
            Vector3::new(0.0, 0.9, 1.0),
        ]
    }

    #[test]
    fn point_jacobian_matches_central_differences() -> TestResult {
        let camera = camera()?;
        let h = 1e-6;
        for point in points() {
            let analytic = camera.jacobian_point(&point);
            for axis in 0..3 {
                let (mut up, mut down) = (point, point);
                up[axis] += h;
                down[axis] -= h;
                let numeric = (camera.project(&up)? - camera.project(&down)?) / (2.0 * h);
                for row in 0..2 {
                    assert!(
                        (analytic[(row, axis)] - numeric[row]).abs()
                            < 1e-5 * (1.0 + numeric[row].abs()),
                        "{point:?} d{row}/d{axis}: {} vs {}",
                        analytic[(row, axis)],
                        numeric[row]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn intrinsic_jacobian_matches_central_differences() -> TestResult {
        let base = <[f64; DIM]>::from(&camera()?);
        let camera = Fisheye624Camera::from(base);
        for point in points() {
            let analytic = camera.jacobian_intrinsics(&point);
            for column in 0..DIM {
                // The entries differ in size by five orders: a step of at least 1e-6 keeps
                // round-off below the smallest sensitivity.
                let h = 1e-6 * base[column].abs().max(1.0);
                let (mut up, mut down) = (base, base);
                up[column] += h;
                down[column] -= h;
                let numeric = (Fisheye624Camera::from(up).project(&point)?
                    - Fisheye624Camera::from(down).project(&point)?)
                    / (2.0 * h);
                for row in 0..2 {
                    assert!(
                        (analytic[(row, column)] - numeric[row]).abs()
                            < 1e-5 * (1.0 + numeric[row].abs()),
                        "{point:?} d{row}/dparam{column}: {} vs {}",
                        analytic[(row, column)],
                        numeric[row]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn near_the_axis_the_jacobians_are_finite_and_tend_to_the_pinhole_limit() -> TestResult {
        let camera = camera()?;
        let point = Vector3::new(1e-9, -2e-9, 1.5);
        let jacobian = camera.jacobian_point(&point);
        assert!(jacobian.iter().all(|v| v.is_finite()));
        assert!((jacobian[(0, 0)] - camera.pinhole.fx / 1.5).abs() < 1e-6);
        assert!((jacobian[(1, 1)] - camera.pinhole.fy / 1.5).abs() < 1e-6);
        let intrinsics = camera.jacobian_intrinsics(&point);
        assert!(intrinsics.iter().all(|v| v.is_finite()));
        assert_eq!((intrinsics[(0, 2)], intrinsics[(1, 3)]), (1.0, 1.0));
        Ok(())
    }

    /// With no `k4`, `k5`, tangential or thin-prism terms the model is
    /// Kannala-Brandt: an independent model of this crate that must agree.
    #[test]
    fn it_reduces_exactly_to_kannala_brandt() -> TestResult {
        let kb = KannalaBrandtCamera::new(
            PinholeParams::new(280.0, 281.0, 320.0, 240.0)?,
            DistortionModel::KannalaBrandt {
                k1: 0.02,
                k2: -0.01,
                k3: 0.003,
                k4: -0.0004,
            },
        )?;
        let fisheye = Fisheye624Camera::from(&kb);
        for point in points() {
            if point.z < 0.9 {
                continue;
            }
            let (a, b) = (kb.project(&point)?, fisheye.project(&point)?);
            assert!((a - b).norm() < 1e-9, "{point:?}: {a:?} vs {b:?}");
        }
        let pixel = Vector2::new(400.0, 300.0);
        assert!((kb.unproject(&pixel)? - fisheye.unproject(&pixel)?).norm() < 1e-6);
        Ok(())
    }

    #[test]
    fn parameter_vectors_round_trip() -> TestResult {
        let camera = camera()?;
        let array = <[f64; DIM]>::from(&camera);
        assert_eq!(array, LAMARIA_CAM0);
        assert_eq!(DVector::from(&camera).as_slice(), &LAMARIA_CAM0[..]);
        assert_eq!(Fisheye624Camera::from(array), camera);
        assert_eq!(Fisheye624Camera::try_from(&LAMARIA_CAM0[..])?, camera);
        assert!(Fisheye624Camera::try_from(&LAMARIA_CAM0[..15]).is_err());
        Ok(())
    }

    #[test]
    fn project_aria_s_single_focal_layout_converts_both_ways() -> TestResult {
        let aria: Vec<f64> = std::iter::once(LAMARIA_CAM0[0])
            .chain(LAMARIA_CAM0[2..].iter().copied())
            .collect();
        assert_eq!(aria.len(), 15);
        let camera = Fisheye624Camera::from_aria_params(&aria)?;
        assert_eq!(camera, self::camera()?);
        assert_eq!(camera.to_aria_params()?.to_vec(), aria);

        let mut two_focals = camera;
        two_focals.pinhole.fy += 1.0;
        assert!(two_focals.to_aria_params().is_err());
        assert!(Fisheye624Camera::from_aria_params(&aria[..14]).is_err());
        Ok(())
    }

    #[test]
    fn validation_rejects_the_wrong_variant_and_bad_numbers() -> TestResult {
        let pinhole = PinholeParams::new(200.0, 200.0, 320.0, 240.0)?;
        assert!(Fisheye624Camera::new(pinhole, DistortionModel::None).is_err());
        let mut nan = LAMARIA_CAM0;
        nan[13] = f64::NAN;
        assert!(try_from_params(&nan).is_err());
        let mut negative_focal = LAMARIA_CAM0;
        negative_focal[0] = -1.0;
        assert!(try_from_params(&negative_focal).is_err());
        Ok(())
    }

    #[test]
    fn batch_projection_matches_single_projection() -> TestResult {
        let camera = camera()?;
        let pts = points();
        let batch = camera.project_batch(&nalgebra::Matrix3xX::from_columns(&pts));
        for (i, point) in pts.iter().enumerate() {
            assert!((batch.column(i) - camera.project(point)?).norm() < 1e-12);
        }
        Ok(())
    }
}
