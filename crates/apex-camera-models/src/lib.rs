//! Camera projection models for bundle adjustment, SLAM, and Structure-from-Motion.
//!
//! Every model implements the [`CameraModel`] trait, which exposes projection,
//! unprojection, and analytic point / pose / intrinsic Jacobians. The
//! [cookbook](https://github.com/amin-abouee/apex-solver/tree/main/crates/apex-camera-models/doc/cookbook)
//! has the full mathematical formulations.

use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{Matrix2xX, Matrix3, Matrix3xX, SMatrix, Vector2, Vector3};

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

/// Camera model errors.
#[derive(thiserror::Error, Debug)]
pub enum CameraModelError {
    /// Focal length must be positive: fx={fx}, fy={fy}
    #[error("Focal length must be positive: fx={fx}, fy={fy}")]
    FocalLengthNotPositive { fx: f64, fy: f64 },

    /// Focal length must be finite: fx={fx}, fy={fy}
    #[error("Focal length must be finite: fx={fx}, fy={fy}")]
    FocalLengthNotFinite { fx: f64, fy: f64 },

    /// Principal point must be finite: cx={cx}, cy={cy}
    #[error("Principal point must be finite: cx={cx}, cy={cy}")]
    PrincipalPointNotFinite { cx: f64, cy: f64 },

    /// Distortion coefficient must be finite
    #[error("Distortion coefficient '{name}' must be finite, got {value}")]
    DistortionNotFinite { name: String, value: f64 },

    /// Parameter out of range
    #[error("Parameter '{param}' must be in range [{min}, {max}], got {value}")]
    ParameterOutOfRange {
        param: String,
        value: f64,
        min: f64,
        max: f64,
    },

    /// Point behind camera
    #[error("Point behind camera: z={z} (must be > {min_z})")]
    PointBehindCamera { z: f64, min_z: f64 },

    /// Point at camera center
    #[error("Point at camera center: 3D point too close to optical axis")]
    PointAtCameraCenter,

    /// Projection denominator too small
    #[error("Projection denominator too small: denom={denom} (threshold={threshold})")]
    DenominatorTooSmall { denom: f64, threshold: f64 },

    /// Projection outside valid image region
    #[error("Projection outside valid image region")]
    ProjectionOutOfBounds,

    /// Point outside image bounds
    #[error("Point outside image bounds: ({x}, {y}) not in valid region")]
    PointOutsideImage { x: f64, y: f64 },

    /// Numerical error
    #[error("Numerical error in {operation}: {details}")]
    NumericalError { operation: String, details: String },

    /// Generic invalid parameters
    #[error("Invalid camera parameters: {0}")]
    InvalidParams(String),
}

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
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Result<Self, CameraModelError> {
        let params = Self { fx, fy, cx, cy };
        params.validate()?;
        Ok(params)
    }

    /// Validate pinhole parameters.
    pub fn validate(&self) -> Result<(), CameraModelError> {
        if self.fx <= 0.0 || self.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthNotPositive {
                fx: self.fx,
                fy: self.fy,
            });
        }
        if !self.fx.is_finite() || !self.fy.is_finite() {
            return Err(CameraModelError::FocalLengthNotFinite {
                fx: self.fx,
                fy: self.fy,
            });
        }
        if !self.cx.is_finite() || !self.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointNotFinite {
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
}

fn check_finite(name: &str, value: f64) -> Result<(), CameraModelError> {
    if !value.is_finite() {
        return Err(CameraModelError::DistortionNotFinite {
            name: name.to_string(),
            value,
        });
    }
    Ok(())
}

impl DistortionModel {
    /// Validate distortion parameters for the given model variant.
    pub fn validate(&self) -> Result<(), CameraModelError> {
        match self {
            DistortionModel::None => Ok(()),
            DistortionModel::Radial { k1, k2 } => {
                check_finite("k1", *k1)?;
                check_finite("k2", *k2)
            }
            DistortionModel::BrownConrady { k1, k2, p1, p2, k3 } => {
                check_finite("k1", *k1)?;
                check_finite("k2", *k2)?;
                check_finite("p1", *p1)?;
                check_finite("p2", *p2)?;
                check_finite("k3", *k3)
            }
            DistortionModel::KannalaBrandt { k1, k2, k3, k4 } => {
                check_finite("k1", *k1)?;
                check_finite("k2", *k2)?;
                check_finite("k3", *k3)?;
                check_finite("k4", *k4)
            }
            DistortionModel::FOV { w } => {
                if !w.is_finite() || *w <= 0.0 || *w > std::f64::consts::PI {
                    return Err(CameraModelError::ParameterOutOfRange {
                        param: "w".to_string(),
                        value: *w,
                        min: 0.0,
                        max: std::f64::consts::PI,
                    });
                }
                Ok(())
            }
            DistortionModel::UCM { alpha } => {
                if !alpha.is_finite() || !(0.0..=1.0).contains(alpha) {
                    return Err(CameraModelError::ParameterOutOfRange {
                        param: "alpha".to_string(),
                        value: *alpha,
                        min: 0.0,
                        max: 1.0,
                    });
                }
                Ok(())
            }
            DistortionModel::EUCM { alpha, beta } => {
                if !alpha.is_finite() || !(0.0..=1.0).contains(alpha) {
                    return Err(CameraModelError::ParameterOutOfRange {
                        param: "alpha".to_string(),
                        value: *alpha,
                        min: 0.0,
                        max: 1.0,
                    });
                }
                if !beta.is_finite() || *beta <= 0.0 {
                    return Err(CameraModelError::ParameterOutOfRange {
                        param: "beta".to_string(),
                        value: *beta,
                        min: 0.0,
                        max: f64::INFINITY,
                    });
                }
                Ok(())
            }
            DistortionModel::DoubleSphere { xi, alpha } => {
                if !xi.is_finite() || !(-1.0..=1.0).contains(xi) {
                    return Err(CameraModelError::ParameterOutOfRange {
                        param: "xi".to_string(),
                        value: *xi,
                        min: -1.0,
                        max: 1.0,
                    });
                }
                if !alpha.is_finite() || *alpha <= 0.0 || *alpha > 1.0 {
                    return Err(CameraModelError::ParameterOutOfRange {
                        param: "alpha".to_string(),
                        value: *alpha,
                        min: 0.0,
                        max: 1.0,
                    });
                }
                Ok(())
            }
            DistortionModel::FTheta { k1, k2, k3, k4 } => {
                if !k1.is_finite() || *k1 <= 0.0 {
                    return Err(CameraModelError::FocalLengthNotPositive { fx: *k1, fy: *k1 });
                }
                check_finite("k2", *k2)?;
                check_finite("k3", *k3)?;
                check_finite("k4", *k4)
            }
        }
    }
}

/// Returns `Ok(())` if `z > √ε` (≈ 1.5e-8) and
/// `Err(PointAtCameraCenter)` otherwise. Used to reject points too close to
/// the optical axis that would cause numerical instability in the
/// perspective division.
pub fn validate_point_in_front(z: f64) -> Result<(), CameraModelError> {
    if z < f64::EPSILON.sqrt() {
        return Err(CameraModelError::PointAtCameraCenter);
    }
    Ok(())
}

// Camera model modules

pub mod bal_pinhole;
pub mod double_sphere;
pub mod eucm;
pub mod fov;
pub mod ftheta;
pub mod kannala_brandt;
pub mod pinhole;
pub mod rad_tan;
pub mod ucm;

// Re-export camera types
pub use bal_pinhole::BALPinholeCameraStrict;
pub use double_sphere::DoubleSphereCamera;
pub use eucm::EucmCamera;
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
    /// Returns `PointBehindCamera`, `PointAtCameraCenter`, `DenominatorTooSmall`,
    /// or `ProjectionOutOfBounds` depending on the model.
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError>;

    /// Unprojects a 2D image point to a normalized 3D ray in camera frame.
    /// Some models use Newton-Raphson for undistortion. See the per-model
    /// cookbook page for the algorithm.
    ///
    /// # Errors
    ///
    /// Returns `PointOutsideImage` or `NumericalError` (e.g. when the
    /// iterative solver fails to converge).
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError>;

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

    /// Projects N 3D points in one call. Invalid projections are replaced
    /// by the sentinel `(1e6, 1e6)`. Models may override with a vectorised
    /// implementation.
    fn project_batch(&self, points_cam: &Matrix3xX<f64>) -> Matrix2xX<f64> {
        let n = points_cam.ncols();
        let mut result = Matrix2xX::zeros(n);
        for i in 0..n {
            let p = Vector3::new(points_cam[(0, i)], points_cam[(1, i)], points_cam[(2, i)]);
            match self.project(&p) {
                Ok(uv) => result.set_column(i, &uv),
                Err(_) => result.set_column(i, &Vector2::new(1e6, 1e6)),
            }
        }
        result
    }

    /// Validates camera intrinsic and distortion parameters. The exact
    /// rules are model-specific; see the per-model cookbook page under
    /// "Validation Rules".
    fn validate_params(&self) -> Result<(), CameraModelError>;

    /// Returns the linear intrinsics `(f_x, f_y, c_x, c_y)`.
    fn get_pinhole_params(&self) -> PinholeParams;

    /// Returns the distortion model and its parameters.
    fn get_distortion(&self) -> DistortionModel;

    /// Returns the camera model identifier (`"pinhole"`, `"rad_tan"`, ...).
    fn get_model_name(&self) -> &'static str;
}

/// Skew-symmetric cross-product matrix `[v]×` such that `[v]× w = v × w`.
#[inline]
pub(crate) fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pinhole::PinholeCamera;
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
        // z = 0 < sqrt(EPSILON) ≈ 1.49e-8, should fail
        assert!(validate_point_in_front(0.0).is_err(), "z = 0 should fail");
    }

    #[test]
    fn test_project_batch_default_impl() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 320.0, 240.0)?;
        let camera = PinholeCamera::new(pinhole, DistortionModel::None)?;

        // 3 valid points + 1 invalid (behind camera)
        let pts = Matrix3xX::from_columns(&[
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.1, 0.2, 1.0),
            Vector3::new(-0.1, 0.1, 2.0),
            Vector3::new(0.0, 0.0, -1.0), // behind camera → sentinel (1e6, 1e6)
        ]);

        let result = camera.project_batch(&pts);
        assert_eq!(result.ncols(), 4);
        assert!(result[(0, 0)].is_finite());
        assert!(result[(1, 0)].is_finite());
        assert!(
            (result[(0, 3)] - 1e6).abs() < 1.0,
            "Invalid projection should be sentinel 1e6, got {}",
            result[(0, 3)]
        );
        assert!(
            (result[(1, 3)] - 1e6).abs() < 1.0,
            "Invalid projection should be sentinel 1e6, got {}",
            result[(1, 3)]
        );
        Ok(())
    }

    #[test]
    fn test_camera_model_error_display_focal_length_not_positive() {
        let e = CameraModelError::FocalLengthNotPositive {
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
        let e = CameraModelError::PointBehindCamera {
            z: -0.5,
            min_z: 1e-6,
        };
        let s = format!("{e}");
        assert!(s.contains("z="), "Display should include z: {s}");
    }

    #[test]
    fn test_camera_model_error_display_parameter_out_of_range() {
        let e = CameraModelError::ParameterOutOfRange {
            param: "alpha".to_string(),
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
}
