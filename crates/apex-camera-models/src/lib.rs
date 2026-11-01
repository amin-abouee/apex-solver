//! Camera projection models for bundle adjustment, SLAM, and Structure-from-Motion.
//!
//! Every model implements the [`CameraModel`] trait, which exposes projection,
//! unprojection, and analytic point / pose / intrinsic Jacobians. The
//! [cookbook](https://github.com/amin-abouee/apex-solver/tree/main/crates/apex-camera-models/doc/cookbook)
//! has the full mathematical formulations.
//!
//! [`Fisheye62Camera`] and [`Fisheye624Camera`] are Project Aria's fisheye models
//! (LaMAria's `RAD_TAN_THIN_PRISM_FISHEYE`), after Meta's
//! [camera intrinsic models](https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/camera_intrinsic_models)
//! and [`projectaria_tools`](https://github.com/facebookresearch/projectaria_tools).
//! [`convert_camera`] fits a camera of any model to another.

use crate::error::ErrorLogging;
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{Matrix3, Matrix3xX, SMatrix, Vector2, Vector3};

/// Threshold for geometric validity checks (e.g. point in front of camera).
pub const GEOMETRIC_PRECISION: f64 = 1e-6;

/// Step size for numerical differentiation of Jacobians.
pub const NUMERICAL_DERIVATIVE_EPS: f64 = 1e-7;

/// Allowed difference between analytic and numerical Jacobians in tests.
pub const JACOBIAN_TEST_TOLERANCE: f64 = 1e-5;

/// Allowed projection error in projection tests.
pub const PROJECTION_TEST_TOLERANCE: f64 = 1e-10;

/// Minimum depth (meters) for a 3D point to be projectable.
pub const MIN_DEPTH: f64 = 1e-6;

/// Convergence threshold for iterative unprojection (e.g. Kannala-Brandt).
pub const CONVERGENCE_THRESHOLD: f64 = 1e-6;

pub mod error;

pub use error::{
    CameraModelError, CameraModelResult, ConversionError, ConversionResult, EstimationError,
    EstimationResult, ParamError, ParamResult, ProjectionError, ProjectionResult,
    UnprojectionError, UnprojectionResult,
};

/// Linear intrinsic parameters shared by every model except F-Theta.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PinholeParams {
    /// Focal length, x (pixels)
    pub fx: f64,
    /// Focal length, y (pixels)
    pub fy: f64,
    /// Principal point, x (pixels)
    pub cx: f64,
    /// Principal point, y (pixels)
    pub cy: f64,
}

impl PinholeParams {
    /// Create new pinhole parameters with validation.
    ///
    /// # Errors
    ///
    /// The [`ParamError`] of [`PinholeParams::validate`], logged here.
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> ParamResult<Self> {
        let params = Self { fx, fy, cx, cy };
        params.validate().map_err(ErrorLogging::log)?;
        Ok(params)
    }

    /// Validate pinhole parameters: positive finite focal lengths and a finite
    /// principal point. Pure check, does not log.
    ///
    /// # Errors
    ///
    /// [`ParamError::FocalLengthNotPositive`], [`ParamError::FocalLengthNotFinite`]
    /// or [`ParamError::PrincipalPointNotFinite`].
    pub fn validate(&self) -> ParamResult<()> {
        if self.fx <= 0.0 || self.fy <= 0.0 {
            return Err(ParamError::FocalLengthNotPositive {
                fx: self.fx,
                fy: self.fy,
            });
        }
        if !self.fx.is_finite() || !self.fy.is_finite() {
            return Err(ParamError::FocalLengthNotFinite {
                fx: self.fx,
                fy: self.fy,
            });
        }
        if !self.cx.is_finite() || !self.cy.is_finite() {
            return Err(ParamError::PrincipalPointNotFinite {
                cx: self.cx,
                cy: self.cy,
            });
        }
        Ok(())
    }
}

/// Lens distortion models. The exact parameter ranges are enforced by
/// [`DistortionModel::validate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistortionModel {
    /// No distortion (vanilla pinhole).
    None,

    /// BAL-style radial distortion (`k1`, `k2`).
    Radial { k1: f64, k2: f64 },

    /// Brown-Conrady / OpenCV radial-tangential distortion.
    BrownConrady {
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        k3: f64,
    },

    /// Kannala-Brandt polynomial fisheye.
    KannalaBrandt { k1: f64, k2: f64, k3: f64, k4: f64 },

    /// Devernay-Faugeras field-of-view.
    FOV { w: f64 },

    /// Geyer-Daniilidis unified camera model.
    UCM { alpha: f64 },

    /// Khomutenko extended UCM.
    EUCM { alpha: f64, beta: f64 },

    /// Usenko double-sphere.
    DoubleSphere { xi: f64, alpha: f64 },

    /// NVIDIA f-theta polynomial fisheye.
    FTheta { k1: f64, k2: f64, k3: f64, k4: f64 },

    /// Project Aria `Fisheye62`: a fisheye polynomial with six radial terms
    /// `k0..k5` in the incidence angle and two tangential terms `p0, p1`.
    Fisheye62 { k: [f64; 6], p: [f64; 2] },

    /// Project Aria `Fisheye624` (`FisheyeRadTanThinPrism`): `Fisheye62` plus
    /// four thin-prism terms `s0..s3`.
    Fisheye624 {
        k: [f64; 6],
        p: [f64; 2],
        s: [f64; 4],
    },
}

const K_NAMES: [&str; 6] = ["k0", "k1", "k2", "k3", "k4", "k5"];
const P_NAMES: [&str; 2] = ["p0", "p1"];
const S_NAMES: [&str; 4] = ["s0", "s1", "s2", "s3"];

fn check_finite(name: &'static str, value: f64) -> ParamResult<()> {
    if !value.is_finite() {
        return Err(ParamError::DistortionNotFinite { name, value });
    }
    Ok(())
}

fn check_all_finite(names: &[&'static str], values: &[f64]) -> ParamResult<()> {
    for (name, value) in names.iter().zip(values) {
        check_finite(name, *value)?;
    }
    Ok(())
}

fn check_range(
    param: &'static str,
    value: f64,
    min: f64,
    max: f64,
    valid: bool,
) -> ParamResult<()> {
    if !value.is_finite() || !valid {
        return Err(ParamError::ParameterOutOfRange {
            param,
            value,
            min,
            max,
        });
    }
    Ok(())
}

impl DistortionModel {
    /// The variant's name, as written after `DistortionModel::`.
    pub fn variant_name(&self) -> &'static str {
        match self {
            DistortionModel::None => "None",
            DistortionModel::Radial { .. } => "Radial",
            DistortionModel::BrownConrady { .. } => "BrownConrady",
            DistortionModel::KannalaBrandt { .. } => "KannalaBrandt",
            DistortionModel::FOV { .. } => "FOV",
            DistortionModel::UCM { .. } => "UCM",
            DistortionModel::EUCM { .. } => "EUCM",
            DistortionModel::DoubleSphere { .. } => "DoubleSphere",
            DistortionModel::FTheta { .. } => "FTheta",
            DistortionModel::Fisheye62 { .. } => "Fisheye62",
            DistortionModel::Fisheye624 { .. } => "Fisheye624",
        }
    }

    /// Check that this is the variant `expected` (a [`variant_name`](Self::variant_name))
    /// that the camera `model` needs.
    ///
    /// # Errors
    ///
    /// [`ParamError::DistortionVariant`] for any other variant.
    pub fn expect_variant(&self, model: &'static str, expected: &'static str) -> ParamResult<()> {
        let actual = self.variant_name();
        if actual != expected {
            return Err(ParamError::DistortionVariant {
                model,
                expected,
                actual,
            });
        }
        Ok(())
    }

    /// Validate distortion parameters for the given model variant. Pure
    /// check, does not log.
    ///
    /// # Errors
    ///
    /// [`ParamError::DistortionNotFinite`] or [`ParamError::ParameterOutOfRange`].
    pub fn validate(&self) -> ParamResult<()> {
        match *self {
            DistortionModel::None => Ok(()),
            DistortionModel::Radial { k1, k2 } => {
                check_finite("k1", k1)?;
                check_finite("k2", k2)
            }
            DistortionModel::BrownConrady { k1, k2, p1, p2, k3 } => {
                check_finite("k1", k1)?;
                check_finite("k2", k2)?;
                check_finite("p1", p1)?;
                check_finite("p2", p2)?;
                check_finite("k3", k3)
            }
            DistortionModel::KannalaBrandt { k1, k2, k3, k4 } => {
                check_finite("k1", k1)?;
                check_finite("k2", k2)?;
                check_finite("k3", k3)?;
                check_finite("k4", k4)
            }
            DistortionModel::FOV { w } => {
                let pi = std::f64::consts::PI;
                check_range("w", w, 0.0, pi, w > 0.0 && w <= pi)
            }
            DistortionModel::UCM { alpha } => {
                check_range("alpha", alpha, 0.0, 1.0, (0.0..=1.0).contains(&alpha))
            }
            DistortionModel::EUCM { alpha, beta } => {
                check_range("alpha", alpha, 0.0, 1.0, (0.0..=1.0).contains(&alpha))?;
                check_range("beta", beta, 0.0, f64::INFINITY, beta > 0.0)
            }
            DistortionModel::DoubleSphere { xi, alpha } => {
                check_range("xi", xi, -1.0, 1.0, (-1.0..=1.0).contains(&xi))?;
                check_range("alpha", alpha, 0.0, 1.0, alpha > 0.0 && alpha <= 1.0)
            }
            DistortionModel::FTheta { k1, k2, k3, k4 } => {
                check_range("k1", k1, 0.0, f64::INFINITY, k1 > 0.0)?;
                check_finite("k2", k2)?;
                check_finite("k3", k3)?;
                check_finite("k4", k4)
            }
            DistortionModel::Fisheye62 { k, p } => {
                check_all_finite(&K_NAMES, &k)?;
                check_all_finite(&P_NAMES, &p)
            }
            DistortionModel::Fisheye624 { k, p, s } => {
                check_all_finite(&K_NAMES, &k)?;
                check_all_finite(&P_NAMES, &p)?;
                check_all_finite(&S_NAMES, &s)
            }
        }
    }
}

/// Check that a parameter slice has exactly `expected` entries for `model`.
pub(crate) fn check_param_count(
    model: &'static str,
    expected: usize,
    params: &[f64],
) -> ParamResult<()> {
    if params.len() != expected {
        return Err(ParamError::ParamCount {
            model,
            expected,
            actual: params.len(),
        });
    }
    Ok(())
}

/// Returns `Ok(())` if `z >= GEOMETRIC_PRECISION` (1e-6) and
/// `Err(PointAtCameraCenter)` otherwise. Used to reject points too close to
/// the optical axis that would cause numerical instability in the
/// perspective division.
///
/// # Errors
///
/// [`ProjectionError::PointAtCameraCenter`]; not logged (hot path).
pub fn validate_point_in_front(z: f64) -> ProjectionResult<()> {
    if z < crate::GEOMETRIC_PRECISION {
        return Err(ProjectionError::PointAtCameraCenter);
    }
    Ok(())
}

// Camera model modules

pub mod bal_pinhole;
pub mod convert;
pub mod double_sphere;
pub mod eucm;
pub mod fisheye62;
pub mod fisheye624;
pub mod fov;
pub mod ftheta;
pub mod kannala_brandt;
pub mod pinhole;
pub mod rad_tan;
pub mod ucm;

mod fisheye_core;

// Re-export camera types
pub use bal_pinhole::BALPinholeCameraStrict;
pub use convert::{ConversionOptions, ConversionReport, convert_camera};
pub use double_sphere::DoubleSphereCamera;
pub use eucm::EucmCamera;
pub use fisheye62::Fisheye62Camera;
pub use fisheye624::Fisheye624Camera;
pub use fov::FovCamera;
pub use ftheta::FThetaCamera;
pub use kannala_brandt::KannalaBrandtCamera;
pub use pinhole::PinholeCamera;
pub use rad_tan::RadTanCamera;
pub use ucm::UcmCamera;

// Camera Model Trait

/// Trait for camera projection models.
///
/// Defines the interface for camera models used in bundle adjustment and SfM.
///
/// # Type Parameters
///
/// - `INTRINSIC_DIM`: Number of intrinsic parameters
/// - `IntrinsicJacobian`: Jacobian type for intrinsics (2 × INTRINSIC_DIM)
/// - `PointJacobian`: Jacobian type for 3D point (2 × 3)
pub trait CameraModel: Send + Sync + Clone + std::fmt::Debug + 'static {
    /// Number of intrinsic parameters (compile-time constant).
    const INTRINSIC_DIM: usize;

    /// Jacobian type for intrinsics: 2 × INTRINSIC_DIM.
    type IntrinsicJacobian: Clone
        + std::fmt::Debug
        + Default
        + std::ops::Index<(usize, usize), Output = f64>;

    /// Jacobian type for 3D point: 2 × 3.
    type PointJacobian: Clone
        + std::fmt::Debug
        + Default
        + std::ops::Mul<SMatrix<f64, 3, 6>, Output = SMatrix<f64, 2, 6>>
        + std::ops::Mul<Matrix3<f64>, Output = SMatrix<f64, 2, 3>>
        + std::ops::Index<(usize, usize), Output = f64>;

    /// Projects a 3D point in camera coordinates to 2D image coordinates.
    /// See the [cookbook introduction](../doc/cookbook/src/introduction.html)
    /// for the projection pipeline, and the per-model page for the formula.
    ///
    /// # Errors
    ///
    /// A [`ProjectionError`] (`PointBehindCamera`, `DenominatorTooSmall`,
    /// `OutOfDomain`, ...) depending on the model. These are expected
    /// outcomes on the hot path: they do not allocate and are not logged.
    fn project(&self, p_cam: &Vector3<f64>) -> ProjectionResult<Vector2<f64>>;

    /// Unprojects a 2D image point to a normalized 3D ray in camera frame.
    /// Some models use Newton-Raphson for undistortion. See the per-model
    /// cookbook page for the algorithm.
    ///
    /// # Errors
    ///
    /// An [`UnprojectionError`] (`PointOutsideImage`, `NotConverged`, ...).
    /// Like projection errors, these are not allocated or logged.
    fn unproject(&self, point_2d: &Vector2<f64>) -> UnprojectionResult<Vector3<f64>>;

    /// ∂(u,v)/∂(x,y,z) — 2×3 Jacobian of projection w.r.t. the 3D point.
    /// Used for structure optimisation, triangulation, and bundle adjustment.
    /// See the per-model cookbook page for the formula.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian;

    /// ∂(u,v)/∂(δξ) — 2×6 Jacobian of projection w.r.t. the camera pose.
    ///
    /// The pose is a world-to-camera transform `T_wc` with right
    /// perturbation `T' = T · Exp(δξ)`. Returns a pair
    /// `(J_uv_pcam, J_pcam_pose)` where the caller multiplies to get
    /// the full 2×6 Jacobian. See the
    /// [cookbook introduction](../doc/cookbook/src/introduction.html#pose-jacobians-se3)
    /// for the SE(3) conventions.
    fn jacobian_pose(
        &self,
        p_world: &Vector3<f64>,
        pose: &SE3,
    ) -> (Self::PointJacobian, SMatrix<f64, 3, 6>) {
        let p_cam = pose.act(p_world, None, None);
        let d_uv_d_pcam = self.jacobian_point(&p_cam);

        let rotation = pose.rotation_so3().rotation_matrix();
        let p_world_skew = skew_symmetric(p_world);

        let d_pcam_d_pose = SMatrix::<f64, 3, 6>::from_fn(|r, c| {
            if c < 3 {
                rotation[(r, c)]
            } else {
                let col = c - 3;
                -(0..3)
                    .map(|k| rotation[(r, k)] * p_world_skew[(k, col)])
                    .sum::<f64>()
            }
        });

        (d_uv_d_pcam, d_pcam_d_pose)
    }

    /// ∂(u,v)/∂(params) — 2×N Jacobian of projection w.r.t. intrinsic
    /// parameters, where `N = INTRINSIC_DIM`. The parameter order is
    /// model-specific; see the per-model cookbook page.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian;

    /// Signed depth deficit of a camera-frame point outside the projection
    /// domain of [`project`](Self::project), plus its gradient.
    ///
    /// Returns `(deficit, ∂deficit/∂p_cam)`. A positive `deficit` means
    /// `p_cam` is outside the domain `project` accepts, and its magnitude is
    /// the distance to that domain measured along the model's optical axis
    /// (in metres), so a caller can scale it into a penalty that grows as the
    /// point moves deeper into the invalid region. The second element is the
    /// gradient of the deficit with respect to the camera-frame point, which
    /// is what lets that penalty have a real derivative — a barrier an
    /// optimizer can be pushed *back* across, rather than one it can only be
    /// charged for.
    ///
    /// The default implementation is the z-forward deficit
    /// `deficit = -z`, gradient `-e_z`, which is correct for every model
    /// whose domain is `z > GEOMETRIC_PRECISION`. Models with a different
    /// domain must override it:
    ///
    /// * **z-backward models** (the BAL convention, valid for
    ///   `z < -MIN_DEPTH`) must return a deficit that grows with `+z`;
    ///   keeping the default there would produce a gradient that drives
    ///   points *further* behind the camera.
    /// * **models with a non-planar domain** — double sphere's
    ///   `z > -w₂·‖p_cam‖` cone, whose boundary moves with the point — must
    ///   return the distance to that boundary rather than to a fixed plane.
    ///
    /// A model that cannot express its failure smoothly should return a
    /// non-positive deficit: callers clamp it, which degrades to a constant
    /// barrier with no gradient — never free, and never wrong-signed.
    ///
    /// # Status: not currently charged by the solver
    ///
    /// `ProjectionFactor` does **not** apply this deficit yet: the attempt to
    /// route every non-cheirality camera error through a barrier built from it
    /// was measured and reverted, because at BAL t257/dubrovnik-135 it charged
    /// more than the entire problem was worth at the initial guess (3.29e12
    /// and 2.20e10 against 6.86e4 / 1.89e5 goldens) and the solve stopped
    /// making progress. The per-model definitions below stay: they are the
    /// correct domain geometry, unit-tested here, ready for a formulation that
    /// survives those benchmarks. See
    /// `benchmarking_results/experiment_007_projection_domain_barrier_REJECTED.md`.
    ///
    /// # Examples
    ///
    /// ```
    /// use apex_camera_models::{CameraModel, PinholeCamera};
    /// use nalgebra::Vector3;
    ///
    /// let camera = PinholeCamera::from([500.0, 500.0, 320.0, 240.0]);
    ///
    /// // Two metres behind the camera ⇒ a two-metre deficit …
    /// let (deficit, grad) =
    ///     camera.projection_deficit(&Vector3::new(0.3, -0.1, -2.0));
    /// assert!((deficit - 2.0).abs() < 1e-12);
    /// // … whose gradient points along −z, i.e. toward validity.
    /// assert_eq!(grad, Vector3::new(0.0, 0.0, -1.0));
    ///
    /// // In front of the camera the deficit is non-positive: the model is
    /// // inside its domain, so any projection failure there is numerical
    /// // and has no smooth direction out of.
    /// let (deficit, _) = camera.projection_deficit(&Vector3::new(0.3, -0.1, 2.0));
    /// assert!(deficit <= 0.0);
    /// ```
    fn projection_deficit(&self, p_cam: &Vector3<f64>) -> (f64, Vector3<f64>) {
        (-p_cam.z, Vector3::new(0.0, 0.0, -1.0))
    }

    /// Projects N 3D points in one call, one result per column. Models may
    /// override with a vectorised implementation.
    fn project_batch(&self, points_cam: &Matrix3xX<f64>) -> Vec<ProjectionResult<Vector2<f64>>> {
        points_cam
            .column_iter()
            .map(|p| self.project(&Vector3::new(p[0], p[1], p[2])))
            .collect()
    }

    /// Validates camera intrinsic and distortion parameters, including that
    /// the distortion is the variant the model needs. The exact rules are
    /// model-specific; see the per-model cookbook page under "Validation
    /// Rules". Pure check, does not log: the checked constructors log.
    ///
    /// # Errors
    ///
    /// The first [`ParamError`] found.
    fn validate_params(&self) -> ParamResult<()>;

    /// Returns the linear intrinsics `(f_x, f_y, c_x, c_y)`.
    fn get_pinhole_params(&self) -> PinholeParams;

    /// Returns the distortion model and its parameters.
    fn get_distortion(&self) -> DistortionModel;

    /// Returns the camera model identifier (`"pinhole"`, `"rad_tan"`, ...).
    fn get_model_name(&self) -> &'static str;
}

pub(crate) use apex_manifolds::util::skew as skew_symmetric;

#[cfg(test)]
mod tests {
    use super::*;
    use apex_manifolds::LieGroup;
    use apex_manifolds::se3::{SE3, SE3Tangent};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Canonical test for the default `jacobian_pose` implementation (right perturbation).
    ///
    /// Since `jacobian_pose` has a single default implementation shared by all models
    /// except BAL, we test it once here using `PinholeCamera` as a representative model.
    #[test]
    fn test_jacobian_pose_numerical() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let camera = PinholeCamera::new(pinhole, DistortionModel::None)?;
        let pose = SE3::from_translation_euler(0.1, -0.2, 0.3, 0.05, -0.1, 0.15);
        let p_world = Vector3::new(1.0, 0.5, 3.0);

        let (d_uv_d_pcam, d_pcam_d_pose) = camera.jacobian_pose(&p_world, &pose);
        let d_uv_d_pose = d_uv_d_pcam * d_pcam_d_pose;

        let eps = NUMERICAL_DERIVATIVE_EPS;

        for i in 0..6 {
            let mut d = [0.0f64; 6];
            d[i] = eps;
            let delta_plus = SE3Tangent::from_components(d[0], d[1], d[2], d[3], d[4], d[5]);
            d[i] = -eps;
            let delta_minus = SE3Tangent::from_components(d[0], d[1], d[2], d[3], d[4], d[5]);

            // Right perturbation on T_wc: pose' = pose · Exp(δ)
            let p_cam_plus = pose.plus(&delta_plus, None, None).act(&p_world, None, None);
            let p_cam_minus = pose
                .plus(&delta_minus, None, None)
                .act(&p_world, None, None);

            let uv_plus = camera.project(&p_cam_plus)?;
            let uv_minus = camera.project(&p_cam_minus)?;

            let num_deriv = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                let analytical = d_uv_d_pose[(r, i)];
                let numerical = num_deriv[r];
                let rel_err = (analytical - numerical).abs() / (1.0 + numerical.abs());
                assert!(
                    rel_err < JACOBIAN_TEST_TOLERANCE,
                    "jacobian_pose mismatch at ({r},{i}): analytical={analytical}, numerical={numerical}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_skew_symmetric() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let skew = skew_symmetric(&v);

        assert_eq!(skew[(0, 0)], 0.0);
        assert_eq!(skew[(1, 1)], 0.0);
        assert_eq!(skew[(2, 2)], 0.0);

        assert_eq!(skew[(0, 1)], -skew[(1, 0)]);
        assert_eq!(skew[(0, 2)], -skew[(2, 0)]);
        assert_eq!(skew[(1, 2)], -skew[(2, 1)]);

        assert_eq!(skew[(0, 1)], -v.z);
        assert_eq!(skew[(0, 2)], v.y);
        assert_eq!(skew[(1, 0)], v.z);
        assert_eq!(skew[(1, 2)], -v.x);
        assert_eq!(skew[(2, 0)], -v.y);
        assert_eq!(skew[(2, 1)], v.x);

        let w = Vector3::new(4.0, 5.0, 6.0);
        let cross_via_skew = skew * w;
        let cross_direct = v.cross(&w);
        assert!((cross_via_skew - cross_direct).norm() < 1e-10);
    }

    #[test]
    fn test_pinhole_validate_negative_focal_length() {
        let result = PinholeParams::new(-1.0, 300.0, 320.0, 240.0);
        assert!(result.is_err(), "negative fx should fail validation");
    }

    #[test]
    fn test_pinhole_validate_zero_focal_length() {
        let result = PinholeParams::new(0.0, 300.0, 320.0, 240.0);
        assert!(result.is_err(), "fx = 0 should fail validation");
    }

    #[test]
    fn test_pinhole_validate_nan_focal_length() {
        let result = PinholeParams::new(f64::NAN, 300.0, 320.0, 240.0);
        assert!(result.is_err(), "NaN fx should fail validation");
    }

    #[test]
    fn test_pinhole_validate_infinite_focal_length() {
        // Inf is > 0 so passes the first check, but fails the is_finite() check
        let result = PinholeParams::new(f64::INFINITY, 300.0, 320.0, 240.0);
        assert!(result.is_err(), "Inf fx should fail validation");
    }

    #[test]
    fn test_pinhole_validate_nan_principal_point() {
        let result = PinholeParams::new(300.0, 300.0, f64::NAN, 240.0);
        assert!(result.is_err(), "NaN cx should fail validation");
    }

    #[test]
    fn test_distortion_none_is_valid() {
        assert!(DistortionModel::None.validate().is_ok());
    }

    #[test]
    fn test_distortion_radial_nan_fails() {
        let d = DistortionModel::Radial {
            k1: f64::NAN,
            k2: 0.0,
        };
        assert!(d.validate().is_err(), "NaN k1 should fail");
    }

    #[test]
    fn test_distortion_brown_conrady_nan_fails() {
        let d = DistortionModel::BrownConrady {
            k1: 0.0,
            k2: f64::NAN,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        };
        assert!(d.validate().is_err(), "NaN k2 should fail");
    }

    #[test]
    fn test_distortion_kannala_brandt_nan_fails() {
        let d = DistortionModel::KannalaBrandt {
            k1: 0.0,
            k2: 0.0,
            k3: f64::NAN,
            k4: 0.0,
        };
        assert!(d.validate().is_err(), "NaN k3 should fail");
    }

    #[test]
    fn test_distortion_fov_invalid_w_zero() {
        let d = DistortionModel::FOV { w: 0.0 };
        assert!(d.validate().is_err(), "w = 0 should fail (must be > 0)");
    }

    #[test]
    fn test_distortion_fov_invalid_w_too_large() {
        let d = DistortionModel::FOV {
            w: std::f64::consts::PI + 0.1,
        };
        assert!(d.validate().is_err(), "w > π should fail");
    }

    #[test]
    fn test_distortion_fov_valid() {
        let d = DistortionModel::FOV { w: 1.0 };
        assert!(d.validate().is_ok(), "w = 1.0 should be valid");
    }

    #[test]
    fn test_distortion_ucm_alpha_out_of_range() {
        let d = DistortionModel::UCM { alpha: 1.5 };
        assert!(d.validate().is_err(), "alpha > 1 should fail for UCM");
    }

    #[test]
    fn test_distortion_ucm_alpha_valid() {
        let d = DistortionModel::UCM { alpha: 0.5 };
        assert!(d.validate().is_ok());
    }

    #[test]
    fn test_distortion_eucm_alpha_out_of_range() {
        let d = DistortionModel::EUCM {
            alpha: 1.5,
            beta: 1.0,
        };
        assert!(d.validate().is_err(), "alpha > 1 should fail for EUCM");
    }

    #[test]
    fn test_distortion_eucm_beta_nonpositive() {
        let d = DistortionModel::EUCM {
            alpha: 0.5,
            beta: -1.0,
        };
        assert!(d.validate().is_err(), "beta <= 0 should fail for EUCM");
    }

    #[test]
    fn test_distortion_double_sphere_xi_out_of_range() {
        let d = DistortionModel::DoubleSphere {
            xi: 2.0,
            alpha: 0.6,
        };
        assert!(d.validate().is_err(), "xi > 1 should fail");
    }

    #[test]
    fn test_distortion_double_sphere_alpha_invalid() {
        let d = DistortionModel::DoubleSphere {
            xi: 0.0,
            alpha: 0.0,
        };
        assert!(d.validate().is_err(), "alpha = 0 should fail");
    }

    #[test]
    fn test_validate_point_in_front_valid_z() {
        assert!(
            validate_point_in_front(1.0).is_ok(),
            "z = 1.0 should be valid"
        );
    }

    #[test]
    fn test_validate_point_in_front_behind_camera() {
        assert!(
            validate_point_in_front(-1.0).is_err(),
            "z = -1.0 should fail"
        );
    }

    #[test]
    fn test_validate_point_in_front_at_center() {
        // z = 0 < GEOMETRIC_PRECISION (1e-6), should fail
        assert!(validate_point_in_front(0.0).is_err(), "z = 0 should fail");
    }

    #[test]
    fn test_project_batch_default_impl() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let camera = PinholeCamera::new(pinhole, DistortionModel::None)?;

        let pts = Matrix3xX::from_columns(&[
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.1, 0.2, 1.0),
            Vector3::new(-0.1, 0.1, 2.0),
            Vector3::new(0.0, 0.0, -1.0),
        ]);

        let result = camera.project_batch(&pts);
        assert_eq!(result.len(), 4);
        for (i, column) in pts.column_iter().take(3).enumerate() {
            let single = camera.project(&Vector3::new(column[0], column[1], column[2]))?;
            assert_eq!(result[i], Ok(single));
        }
        assert_eq!(
            result[3],
            Err(ProjectionError::PointBehindCamera {
                z: -1.0,
                min_z: GEOMETRIC_PRECISION
            })
        );
        Ok(())
    }

    #[test]
    fn test_camera_model_error_display_focal_length_not_positive() {
        let e = ParamError::FocalLengthNotPositive {
            fx: -1.0,
            fy: 300.0,
        };
        let s = format!("{e}");
        assert!(
            s.contains("fx") && s.contains("-1"),
            "Display should include parameter values: {s}"
        );
    }

    #[test]
    fn test_camera_model_error_display_point_behind_camera() {
        let e = ProjectionError::PointBehindCamera {
            z: -0.5,
            min_z: 1e-6,
        };
        let s = format!("{e}");
        assert!(s.contains("z="), "Display should include z: {s}");
    }

    #[test]
    fn test_camera_model_error_display_parameter_out_of_range() {
        let e = ParamError::ParameterOutOfRange {
            param: "alpha",
            value: 1.5,
            min: 0.0,
            max: 1.0,
        };
        let s = format!("{e}");
        assert!(
            s.contains("alpha") && s.contains("1.5"),
            "Display should include param and value: {s}"
        );
    }

    #[test]
    fn distortion_errors_name_the_coefficient() {
        let d = DistortionModel::Fisheye624 {
            k: [0.0; 6],
            p: [0.0, f64::INFINITY],
            s: [0.0; 4],
        };
        assert_eq!(
            d.validate(),
            Err(ParamError::DistortionNotFinite {
                name: "p1",
                value: f64::INFINITY
            })
        );
    }

    #[test]
    fn ftheta_k1_is_a_range_error_not_a_focal_length() {
        let d = DistortionModel::FTheta {
            k1: -1.0,
            k2: 0.0,
            k3: 0.0,
            k4: 0.0,
        };
        assert!(matches!(
            d.validate(),
            Err(ParamError::ParameterOutOfRange { param: "k1", .. })
        ));
    }

    #[test]
    fn expect_variant_reports_both_names() {
        let d = DistortionModel::UCM { alpha: 0.5 };
        assert_eq!(d.expect_variant("ucm", "UCM"), Ok(()));
        assert_eq!(
            d.expect_variant("eucm", "EUCM"),
            Err(ParamError::DistortionVariant {
                model: "eucm",
                expected: "EUCM",
                actual: "UCM"
            })
        );
    }

    #[test]
    fn param_count_must_match_exactly() {
        assert_eq!(check_param_count("pinhole", 4, &[1.0; 4]), Ok(()));
        for len in [3, 5] {
            assert_eq!(
                check_param_count("pinhole", 4, &vec![1.0; len]),
                Err(ParamError::ParamCount {
                    model: "pinhole",
                    expected: 4,
                    actual: len
                })
            );
        }
    }

    /// A slice of the wrong length is rejected whether it is short or long,
    /// and a non-finite entry anywhere is rejected too.
    fn check_slice_rules<C>(model: &'static str, valid: &[f64]) -> TestResult
    where
        C: CameraModel + for<'a> TryFrom<&'a [f64], Error = ParamError>,
    {
        let n = valid.len();
        C::try_from(valid)?;
        for len in [n - 1, n + 1] {
            let mut params = valid.to_vec();
            params.resize(len, 0.0);
            assert_eq!(
                C::try_from(&params).err(),
                Some(ParamError::ParamCount {
                    model,
                    expected: n,
                    actual: len
                }),
                "{model} with {len} values"
            );
        }
        for index in 0..n {
            let mut params = valid.to_vec();
            params[index] = f64::NAN;
            assert!(
                C::try_from(&params).is_err(),
                "{model} accepted NaN at index {index}"
            );
        }
        Ok(())
    }

    #[test]
    fn every_model_checks_its_slice_length_and_values() -> TestResult {
        let pinhole = [500.0, 500.0, 320.0, 240.0];
        check_slice_rules::<PinholeCamera>("pinhole", &pinhole)?;
        check_slice_rules::<BALPinholeCameraStrict>("bal_pinhole", &[500.0, 0.1, 0.01])?;
        check_slice_rules::<RadTanCamera>(
            "rad_tan",
            &[500.0, 500.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.001, 0.0],
        )?;
        check_slice_rules::<KannalaBrandtCamera>(
            "kannala_brandt",
            &[300.0, 300.0, 320.0, 240.0, 0.1, 0.01, 0.0, 0.0],
        )?;
        check_slice_rules::<FovCamera>("fov", &[300.0, 300.0, 320.0, 240.0, 1.0])?;
        check_slice_rules::<UcmCamera>("ucm", &[300.0, 300.0, 320.0, 240.0, 0.5])?;
        check_slice_rules::<EucmCamera>("eucm", &[300.0, 300.0, 320.0, 240.0, 0.5, 1.0])?;
        check_slice_rules::<DoubleSphereCamera>(
            "double_sphere",
            &[300.0, 300.0, 320.0, 240.0, -0.2, 0.6],
        )?;
        check_slice_rules::<FThetaCamera>("ftheta", &[320.0, 240.0, 500.0, 0.0, 0.0, 0.0])?;
        let mut f62 = [0.0; 12];
        f62[..4].copy_from_slice(&pinhole);
        check_slice_rules::<Fisheye62Camera>("fisheye62", &f62)?;
        let mut f624 = [0.0; 16];
        f624[..4].copy_from_slice(&pinhole);
        check_slice_rules::<Fisheye624Camera>("fisheye624", &f624)?;
        Ok(())
    }

    #[test]
    fn try_from_now_validates_the_values() {
        // alpha = 1.5 is out of the UCM range; TryFrom used to accept it.
        assert_eq!(
            UcmCamera::try_from(&[300.0, 300.0, 320.0, 240.0, 1.5][..]).err(),
            Some(ParamError::ParameterOutOfRange {
                param: "alpha",
                value: 1.5,
                min: 0.0,
                max: 1.0
            })
        );
    }

    #[test]
    fn every_model_rejects_a_wrong_distortion_variant() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let none = DistortionModel::None;
        let radial = DistortionModel::Radial { k1: 0.0, k2: 0.0 };
        let checks: [(ParamResult<()>, &str, &str); 11] = [
            (
                PinholeCamera::new(pinhole, radial).map(|_| ()),
                "pinhole",
                "None",
            ),
            (
                BALPinholeCameraStrict::new(PinholeParams::new(300.0, 300.0, 0.0, 0.0)?, none)
                    .map(|_| ()),
                "bal_pinhole",
                "Radial",
            ),
            (
                RadTanCamera::new(pinhole, none).map(|_| ()),
                "rad_tan",
                "BrownConrady",
            ),
            (
                KannalaBrandtCamera::new(pinhole, none).map(|_| ()),
                "kannala_brandt",
                "KannalaBrandt",
            ),
            (FovCamera::new(pinhole, none).map(|_| ()), "fov", "FOV"),
            (UcmCamera::new(pinhole, none).map(|_| ()), "ucm", "UCM"),
            (EucmCamera::new(pinhole, none).map(|_| ()), "eucm", "EUCM"),
            (
                DoubleSphereCamera::new(pinhole, none).map(|_| ()),
                "double_sphere",
                "DoubleSphere",
            ),
            (
                FThetaCamera::new(320.0, 240.0, none).map(|_| ()),
                "ftheta",
                "FTheta",
            ),
            (
                Fisheye62Camera::new(pinhole, none).map(|_| ()),
                "fisheye62",
                "Fisheye62",
            ),
            (
                Fisheye624Camera::new(pinhole, none).map(|_| ()),
                "fisheye624",
                "Fisheye624",
            ),
        ];
        for (result, model, expected) in checks {
            let actual = if model == "pinhole" { "Radial" } else { "None" };
            assert_eq!(
                result,
                Err(ParamError::DistortionVariant {
                    model,
                    expected,
                    actual
                })
            );
        }
        Ok(())
    }

    /// A struct literal can still hold a mismatched variant. It projects to
    /// NaN, never to a finite pixel computed with zero distortion.
    #[test]
    fn a_mismatched_struct_literal_never_projects_silently() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let none = DistortionModel::None;
        let point = Vector3::new(0.1, 0.2, 1.0);
        let finite = |r: ProjectionResult<Vector2<f64>>| r.is_ok_and(|uv| uv.x.is_finite());
        assert!(!finite(
            FovCamera {
                pinhole,
                distortion: none
            }
            .project(&point)
        ));
        assert!(!finite(
            RadTanCamera {
                pinhole,
                distortion: none
            }
            .project(&point)
        ));
        assert!(!finite(
            KannalaBrandtCamera {
                pinhole,
                distortion: none
            }
            .project(&point)
        ));
        assert!(!finite(
            Fisheye624Camera {
                pinhole,
                distortion: none
            }
            .project(&point)
        ));
        let ftheta = FThetaCamera {
            cx: 320.0,
            cy: 240.0,
            distortion: none,
        };
        assert!(!finite(ftheta.project(&point)));
        assert_eq!(
            ftheta.distortion_params(),
            Err(ParamError::DistortionVariant {
                model: "ftheta",
                expected: "FTheta",
                actual: "None"
            })
        );
        Ok(())
    }

    #[test]
    fn hot_path_errors_name_their_cause() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let behind = Vector3::new(0.0, 0.0, -1.0);
        let kb = KannalaBrandtCamera::new(
            pinhole,
            DistortionModel::KannalaBrandt {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                k4: 0.0,
            },
        )?;
        assert_eq!(
            kb.project(&behind),
            Err(ProjectionError::PointBehindCamera {
                z: -1.0,
                min_z: GEOMETRIC_PRECISION
            })
        );
        let far = Vector2::new(320.0 + 300.0 * 2.0, 240.0);
        assert_eq!(
            kb.unproject(&far),
            Err(UnprojectionError::RadiusOutOfDomain {
                radius: 2.0,
                max: std::f64::consts::FRAC_PI_2
            })
        );
        assert_eq!(
            kb.unproject(&Vector2::new(f64::NAN, 0.0)),
            Err(UnprojectionError::NonFinitePixel)
        );

        let fov = FovCamera::new(pinhole, DistortionModel::FOV { w: 1.0 })?;
        assert_eq!(
            fov.project(&behind),
            Err(ProjectionError::OutOfDomain {
                model: "fov",
                z: -1.0
            })
        );

        let ucm = UcmCamera::new(pinhole, DistortionModel::UCM { alpha: 0.8 })?;
        let outside = Vector2::new(320.0 + 300.0 * 10.0, 240.0);
        assert!(matches!(
            ucm.unproject(&outside),
            Err(UnprojectionError::PointOutsideImage { .. })
        ));

        let eucm = EucmCamera::new(
            pinhole,
            DistortionModel::EUCM {
                alpha: 0.8,
                beta: 1.0,
            },
        )?;
        assert!(matches!(
            eucm.unproject(&outside),
            Err(UnprojectionError::PointOutsideImage { .. })
        ));

        let radtan = RadTanCamera::new(
            pinhole,
            DistortionModel::BrownConrady {
                k1: -1.0,
                k2: 0.0,
                p1: 0.0,
                p2: 0.0,
                k3: 0.0,
            },
        )?;
        assert!(matches!(
            radtan.project(&Vector3::new(2.0, 0.0, 1.0)),
            Err(ProjectionError::DistortionFold { .. })
        ));
        Ok(())
    }

    #[test]
    fn a_projection_error_chains_to_the_root() -> TestResult {
        fn project_one(camera: &PinholeCamera) -> CameraModelResult<Vector2<f64>> {
            Ok(camera.project(&Vector3::new(0.0, 0.0, -2.0))?)
        }
        let camera = PinholeCamera::new(
            PinholeParams::new(500.0, 500.0, 320.0, 240.0)?,
            DistortionModel::None,
        )?;
        let error = project_one(&camera).err().ok_or("expected an error")?;
        assert!(matches!(
            error,
            CameraModelError::Projection(ProjectionError::PointBehindCamera { .. })
        ));
        assert_eq!(
            crate::error::ErrorChainExt::chain(&error),
            "point behind camera: z=-2 (must be > 0.000001)"
        );
        Ok(())
    }

    #[test]
    fn linear_estimation_rejects_mismatched_point_counts() -> TestResult {
        let mut ucm = UcmCamera::new(
            PinholeParams::new(300.0, 300.0, 320.0, 240.0)?,
            DistortionModel::UCM { alpha: 0.5 },
        )?;
        let before = ucm;
        let points_3d = Matrix3xX::from_columns(&[Vector3::new(0.1, 0.2, 1.0)]);
        let points_2d = nalgebra::Matrix2xX::from_columns(&[
            Vector2::new(330.0, 250.0),
            Vector2::new(331.0, 251.0),
        ]);
        assert_eq!(
            ucm.linear_estimation(&points_3d, &points_2d),
            Err(EstimationError::PointCountMismatch {
                points_3d: 1,
                points_2d: 2
            })
        );
        assert_eq!(ucm, before);
        let ftheta = FThetaCamera::new(
            320.0,
            240.0,
            DistortionModel::FTheta {
                k1: 300.0,
                k2: 0.0,
                k3: 0.0,
                k4: 0.0,
            },
        )?;
        assert_eq!(
            ftheta.linear_estimation(&points_3d, &points_2d).err(),
            Some(EstimationError::PointCountMismatch {
                points_3d: 1,
                points_2d: 2
            })
        );
        Ok(())
    }
}
