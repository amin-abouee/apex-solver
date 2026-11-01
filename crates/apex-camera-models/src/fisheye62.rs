//! `Fisheye62`: Project Aria's fisheye model with radial and tangential
//! distortion.
//!
//! [`Fisheye624Camera`] without the four thin-prism
//! terms, so 12 intrinsic parameters here: `fx fy cx cy`, six radial terms
//! `k0..k5`, and two tangential terms `p0 p1`. The projection and unprojection
//! are those of `Fisheye624` with `s0..s3 = 0`; the formulas are in
//! [`crate::fisheye624`], and both types share one implementation.
//!
//! Project Aria's own vector has one focal length and 11 entries
//! (`f cx cy k0..k5 p0 p1`); see [`Fisheye62Camera::from_aria_params`].
//!
//! # Source
//!
//! Meta's Project Aria tools: the
//! [camera intrinsic models](https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/camera_intrinsic_models)
//! page and `core/calibration/camera_projections/FisheyeRadTanThinPrism.h`
//! (`Fisheye62 = FisheyeRadTanThinPrism<6, true, false, true>`) in
//! <https://github.com/facebookresearch/projectaria_tools>.

use crate::error::ErrorLogging;
use crate::fisheye_core::{Core, NUM_K};
use crate::{
    CameraModel, DistortionModel, Fisheye624Camera, KannalaBrandtCamera, ParamError, ParamResult,
    PinholeParams, ProjectionResult, UnprojectionResult, check_param_count,
};
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Number of intrinsic parameters: `fx fy cx cy k0..k5 p0 p1`.
const DIM: usize = 12;

/// Number of entries in Project Aria's single-focal-length vector.
const ARIA_DIM: usize = 11;

/// Project Aria `Fisheye62` camera, with 12 intrinsic parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fisheye62Camera {
    pub pinhole: PinholeParams,
    /// Must be [`DistortionModel::Fisheye62`]; the checked constructors reject any
    /// other variant, and a struct literal with one projects to NaN.
    pub distortion: DistortionModel,
}

impl Fisheye62Camera {
    /// Creates a camera.
    ///
    /// # Errors
    ///
    /// [`ParamError::DistortionVariant`] if `distortion` is not
    /// [`DistortionModel::Fisheye62`], and the other errors of
    /// [`CameraModel::validate_params`]; logged here.
    ///
    /// # Example
    ///
    /// ```
    /// use apex_camera_models::{CameraModel, DistortionModel, Fisheye62Camera, PinholeParams};
    ///
    /// let pinhole = PinholeParams::new(241.6, 241.6, 322.9, 240.4)?;
    /// let distortion = DistortionModel::Fisheye62 {
    ///     k: [-0.0248, 0.0963, -0.0633, 0.0062, 0.0035, -0.0007],
    ///     p: [0.0009, 0.0004],
    /// };
    /// let camera = Fisheye62Camera::new(pinhole, distortion)?;
    /// assert_eq!(camera.get_model_name(), "fisheye62");
    /// # Ok::<(), apex_camera_models::CameraModelError>(())
    /// ```
    pub fn new(pinhole: PinholeParams, distortion: DistortionModel) -> ParamResult<Self> {
        let model = Self {
            pinhole,
            distortion,
        };
        model.validate_params().map_err(ErrorLogging::log)?;
        Ok(model)
    }

    /// The camera from Project Aria's own 11-entry vector
    /// `[f, cx, cy, k0..k5, p0, p1]`, with `fx = fy = f`.
    ///
    /// # Errors
    ///
    /// [`ParamError::ParamCount`] unless there are exactly 11 entries,
    /// and the errors of [`CameraModel::validate_params`].
    pub fn from_aria_params(params: &[f64]) -> ParamResult<Self> {
        check_param_count("fisheye62 (Aria layout)", ARIA_DIM, params)
            .map_err(ErrorLogging::log)?;
        let mut native = [0.0; DIM];
        native[0] = params[0];
        native[1] = params[0];
        native[2..].copy_from_slice(&params[1..]);
        try_from_params(&native)
    }

    /// The camera as Project Aria's 11-entry vector
    /// `[f, cx, cy, k0..k5, p0, p1]`.
    ///
    /// # Errors
    ///
    /// [`ParamError::UnequalFocalLengths`], logged, if `fx` and `fy` differ.
    pub fn to_aria_params(&self) -> ParamResult<[f64; ARIA_DIM]> {
        let (fx, fy) = (self.pinhole.fx, self.pinhole.fy);
        if (fx - fy).abs() > 1e-9 * fx.abs().max(fy.abs()) {
            return Err(ParamError::UnequalFocalLengths {
                model: "fisheye62 (Aria layout)",
                fx,
                fy,
            }
            .log());
        }
        let native = <[f64; DIM]>::from(self);
        let mut aria = [0.0; ARIA_DIM];
        aria[0] = fx;
        aria[1..].copy_from_slice(&native[2..]);
        Ok(aria)
    }

    fn core(&self) -> Core {
        let (k, p) = match self.distortion {
            DistortionModel::Fisheye62 { k, p } => (k, p),
            _ => ([f64::NAN; NUM_K], [f64::NAN; 2]),
        };
        Core {
            fx: self.pinhole.fx,
            fy: self.pinhole.fy,
            cx: self.pinhole.cx,
            cy: self.pinhole.cy,
            k,
            p,
            s: [0.0; 4],
        }
    }
}

/// `[fx, fy, cx, cy, k0..k5, p0, p1]`.
impl From<&Fisheye62Camera> for [f64; DIM] {
    fn from(camera: &Fisheye62Camera) -> Self {
        let core = camera.core();
        let mut out = [0.0; DIM];
        out[..4].copy_from_slice(&[core.fx, core.fy, core.cx, core.cy]);
        out[4..10].copy_from_slice(&core.k);
        out[10..].copy_from_slice(&core.p);
        out
    }
}

/// `[fx, fy, cx, cy, k0..k5, p0, p1]`.
impl From<&Fisheye62Camera> for DVector<f64> {
    fn from(camera: &Fisheye62Camera) -> Self {
        DVector::from_row_slice(&<[f64; DIM]>::from(camera))
    }
}

/// From `[fx, fy, cx, cy, k0..k5, p0, p1]`: exactly 12 entries forming a valid
/// camera. Not logged: the solver decodes intrinsics blocks with it on every
/// linearization.
impl TryFrom<&[f64]> for Fisheye62Camera {
    type Error = ParamError;

    fn try_from(params: &[f64]) -> ParamResult<Self> {
        check_param_count("fisheye62", DIM, params)?;
        let mut k = [0.0; NUM_K];
        k.copy_from_slice(&params[4..10]);
        let camera = Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::Fisheye62 {
                k,
                p: [params[10], params[11]],
            },
        };
        camera.validate_params()?;
        Ok(camera)
    }
}

/// From `[fx, fy, cx, cy, k0..k5, p0, p1]`.
impl From<[f64; DIM]> for Fisheye62Camera {
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
            distortion: DistortionModel::Fisheye62 {
                k,
                p: [params[10], params[11]],
            },
        }
    }
}

/// Exact: a Kannala-Brandt camera is `Fisheye62` with `k0..k3 = k1..k4`, no
/// `k4`, `k5` and no tangential terms.
impl From<&KannalaBrandtCamera> for Fisheye62Camera {
    fn from(camera: &KannalaBrandtCamera) -> Self {
        let native = <[f64; 8]>::from(camera);
        let mut out = [0.0; DIM];
        out[..8].copy_from_slice(&native);
        Fisheye62Camera::from(out)
    }
}

/// Narrowing: allowed only when the thin-prism terms are zero, so that no
/// distortion is dropped silently.
impl TryFrom<&Fisheye624Camera> for Fisheye62Camera {
    type Error = ParamError;

    fn try_from(camera: &Fisheye624Camera) -> ParamResult<Self> {
        let native = <[f64; 16]>::from(camera);
        if native[12..].iter().any(|s| *s != 0.0) {
            return Err(ParamError::ThinPrismNotZero.log());
        }
        let mut out = [0.0; DIM];
        out.copy_from_slice(&native[..12]);
        Ok(Fisheye62Camera::from(out))
    }
}

/// Builds a camera from a parameter slice and validates it.
///
/// # Errors
///
/// See [`Fisheye62Camera::try_from`] and [`CameraModel::validate_params`].
pub fn try_from_params(params: &[f64]) -> ParamResult<Fisheye62Camera> {
    Fisheye62Camera::try_from(params).map_err(ErrorLogging::log)
}

impl CameraModel for Fisheye62Camera {
    const INTRINSIC_DIM: usize = DIM;
    type IntrinsicJacobian = SMatrix<f64, 2, DIM>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a camera-frame point; [`crate::ProjectionError::PointBehindCamera`]
    /// when `z < GEOMETRIC_PRECISION`.
    fn project(&self, p_cam: &Vector3<f64>) -> ProjectionResult<Vector2<f64>> {
        self.core().project(p_cam)
    }

    /// Unprojects a pixel to a unit ray; an error outside the model's domain
    /// or when a Newton inversion does not converge.
    fn unproject(&self, point_2d: &Vector2<f64>) -> UnprojectionResult<Vector3<f64>> {
        self.core().unproject(point_2d)
    }

    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        self.core().jacobian_point(p_cam)
    }

    /// 2×12 Jacobian with respect to `[fx fy cx cy k0..k5 p0 p1]`.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        self.core()
            .jacobian_params(p_cam)
            .fixed_columns::<DIM>(0)
            .into_owned()
    }

    /// Validation: `fx`, `fy` positive and finite, `cx`, `cy` and every
    /// distortion term finite.
    fn validate_params(&self) -> ParamResult<()> {
        self.pinhole.validate()?;
        self.distortion.expect_variant("fisheye62", "Fisheye62")?;
        self.distortion.validate()
    }

    fn get_pinhole_params(&self) -> PinholeParams {
        self.pinhole
    }

    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// `"fisheye62"`.
    fn get_model_name(&self) -> &'static str {
        "fisheye62"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// LaMAria `R_06_medium` `cam0` without its thin-prism terms.
    const PARAMS: [f64; DIM] = [
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
    ];

    fn camera() -> Result<Fisheye62Camera, ParamError> {
        try_from_params(&PARAMS)
    }

    /// Computed with a transcription of
    /// `FisheyeRadTanThinPrism<6, true, false, true>::project` in
    /// projectaria_tools: not with this crate.
    const REFERENCE: [([f64; 3], [f64; 2]); 3] = [
        ([0.3, -0.2, 1.0], [392.394_065_258_811, 194.139_841_208_183]),
        ([1.5, 0.9, 1.0], [545.432_262_412_886, 373.917_795_308_816]),
        ([-2.5, 1.0, 0.8], [33.086_917_234_357, 356.659_812_050_196]),
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
    fn project_then_unproject_recovers_the_ray() -> TestResult {
        let camera = camera()?;
        let mut worst = 0.0f64;
        for theta_deg in [0.01, 5.0, 25.0, 50.0, 70.0, 82.0] {
            for phi_deg in (0..360).step_by(45) {
                let (theta, phi) = (f64::to_radians(theta_deg), f64::to_radians(phi_deg as f64));
                let ray = Vector3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                );
                let back = camera.unproject(&camera.project(&(ray * 1.7))?)?;
                worst = worst.max((back - ray).norm());
            }
        }
        assert!(worst < 1e-9, "worst ray error {worst}");
        Ok(())
    }

    #[test]
    fn a_point_behind_the_camera_is_an_error() -> TestResult {
        assert!(matches!(
            camera()?.project(&Vector3::new(0.0, 0.2, -1.0)),
            Err(crate::ProjectionError::PointBehindCamera { .. })
        ));
        Ok(())
    }

    #[test]
    fn jacobians_match_central_differences() -> TestResult {
        let camera = camera()?;
        let points = [
            Vector3::new(0.3, -0.2, 1.0),
            Vector3::new(1.5, 0.9, 1.0),
            Vector3::new(-0.7, -0.6, 1.2),
        ];
        for point in points {
            let analytic = camera.jacobian_point(&point);
            for axis in 0..3 {
                let h = 1e-6;
                let (mut up, mut down) = (point, point);
                up[axis] += h;
                down[axis] -= h;
                let numeric = (camera.project(&up)? - camera.project(&down)?) / (2.0 * h);
                for row in 0..2 {
                    assert!(
                        (analytic[(row, axis)] - numeric[row]).abs()
                            < 1e-5 * (1.0 + numeric[row].abs())
                    );
                }
            }
            let intrinsics = camera.jacobian_intrinsics(&point);
            for column in 0..DIM {
                let h = 1e-6 * PARAMS[column].abs().max(1.0);
                let (mut up, mut down) = (PARAMS, PARAMS);
                up[column] += h;
                down[column] -= h;
                let numeric = (Fisheye62Camera::from(up).project(&point)?
                    - Fisheye62Camera::from(down).project(&point)?)
                    / (2.0 * h);
                for row in 0..2 {
                    assert!(
                        (intrinsics[(row, column)] - numeric[row]).abs()
                            < 1e-5 * (1.0 + numeric[row].abs()),
                        "{point:?} d{row}/dparam{column}"
                    );
                }
            }
        }
        Ok(())
    }

    /// `Fisheye62` is `Fisheye624` with the thin-prism terms zero.
    #[test]
    fn it_equals_fisheye624_without_thin_prism_terms() -> TestResult {
        let narrow = camera()?;
        let wide = Fisheye624Camera::from(&narrow);
        for point in [Vector3::new(0.3, -0.2, 1.0), Vector3::new(-2.5, 1.0, 0.8)] {
            assert_eq!(narrow.project(&point)?, wide.project(&point)?);
        }
        let pixel = Vector2::new(100.0, 400.0);
        assert_eq!(narrow.unproject(&pixel)?, wide.unproject(&pixel)?);
        assert_eq!(Fisheye62Camera::try_from(&wide)?, narrow);
        Ok(())
    }

    #[test]
    fn narrowing_a_camera_with_thin_prism_terms_is_refused() -> TestResult {
        let mut native = <[f64; 16]>::from(&Fisheye624Camera::from(&camera()?));
        native[13] = 1e-4;
        let wide = Fisheye624Camera::from(native);
        assert!(Fisheye62Camera::try_from(&wide).is_err());
        Ok(())
    }

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
        let fisheye = Fisheye62Camera::from(&kb);
        for point in [Vector3::new(0.3, -0.2, 1.0), Vector3::new(1.5, 0.9, 1.0)] {
            assert!((kb.project(&point)? - fisheye.project(&point)?).norm() < 1e-9);
        }
        Ok(())
    }

    #[test]
    fn parameter_vectors_round_trip() -> TestResult {
        let camera = camera()?;
        assert_eq!(<[f64; DIM]>::from(&camera), PARAMS);
        assert_eq!(DVector::from(&camera).as_slice(), &PARAMS[..]);
        assert_eq!(Fisheye62Camera::from(PARAMS), camera);
        assert!(Fisheye62Camera::try_from(&PARAMS[..11]).is_err());

        let aria: Vec<f64> = std::iter::once(PARAMS[0])
            .chain(PARAMS[2..].iter().copied())
            .collect();
        assert_eq!(aria.len(), 11);
        assert_eq!(Fisheye62Camera::from_aria_params(&aria)?, camera);
        assert_eq!(camera.to_aria_params()?.to_vec(), aria);
        assert!(Fisheye62Camera::from_aria_params(&aria[..10]).is_err());
        let mut two_focals = camera;
        two_focals.pinhole.fx += 2.0;
        assert!(two_focals.to_aria_params().is_err());
        Ok(())
    }

    #[test]
    fn validation_rejects_the_wrong_variant() -> TestResult {
        let pinhole = PinholeParams::new(200.0, 200.0, 320.0, 240.0)?;
        assert!(Fisheye62Camera::new(pinhole, DistortionModel::None).is_err());
        let wrong = DistortionModel::Fisheye624 {
            k: [0.0; 6],
            p: [0.0; 2],
            s: [0.0; 4],
        };
        assert!(Fisheye62Camera::new(pinhole, wrong).is_err());
        Ok(())
    }
}
