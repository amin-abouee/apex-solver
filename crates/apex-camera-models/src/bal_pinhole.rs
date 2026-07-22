//! BAL (Bundle Adjustment in the Large) pinhole camera model.
//!
//! Strict 3-parameter pinhole model that follows the BAL dataset / Bundler convention: a
//! single focal length `f`, no principal point (`cx = cy = 0`), two radial coefficients
//! `k1`, `k2`, and the camera looks down the `-Z` axis. Compatible with Ceres Solver and
//! GTSAM bundle adjustment pipelines. See the
//! [BAL pinhole cookbook chapter](../doc/cookbook/src/bal-pinhole.html) for the projection,
//! unprojection, and Jacobian derivations.

use crate::{CameraModel, CameraModelError, DistortionModel, PinholeParams, skew_symmetric};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Strict BAL camera model matching Snavely's Bundler convention.
///
/// 3 intrinsic parameters: focal length `f` (with `fx = fy = f`), and two radial
/// distortion coefficients `k1`, `k2`. The principal point is fixed at the origin
/// (`cx = cy = 0`) and the camera looks down `-Z`. Matches the BAL file format used
/// by Ceres Solver, GTSAM, and the original Bundler software.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BALPinholeCameraStrict {
    /// Single focal length (fx = fy = f)
    pub f: f64,
    pub distortion: DistortionModel,
}

impl BALPinholeCameraStrict {
    /// Creates a new strict BAL pinhole camera with distortion.
    ///
    /// Requires `pinhole.fx == pinhole.fy` and `pinhole.cx == pinhole.cy == 0`. Distortion
    /// must be [`DistortionModel::Radial`].
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::InvalidParams`] if the strict BAL constraints are
    /// violated or if the distortion type is wrong.
    ///
    /// # Example
    ///
    /// ```
    /// use apex_camera_models::{BALPinholeCameraStrict, PinholeParams, DistortionModel};
    ///
    /// let pinhole = PinholeParams::new(500.0, 500.0, 0.0, 0.0)?;
    /// let distortion = DistortionModel::Radial { k1: -0.1, k2: 0.01 };
    /// let camera = BALPinholeCameraStrict::new(pinhole, distortion)?;
    /// # Ok::<(), apex_camera_models::CameraModelError>(())
    /// ```
    pub fn new(
        pinhole: PinholeParams,
        distortion: DistortionModel,
    ) -> Result<Self, CameraModelError> {
        if (pinhole.fx - pinhole.fy).abs() > 1e-10 {
            return Err(CameraModelError::InvalidParams(
                "BALPinholeCameraStrict requires fx = fy (single focal length)".to_string(),
            ));
        }
        if pinhole.cx.abs() > 1e-10 || pinhole.cy.abs() > 1e-10 {
            return Err(CameraModelError::InvalidParams(
                "BALPinholeCameraStrict requires cx = cy = 0 (no principal point offset)"
                    .to_string(),
            ));
        }

        let camera = Self {
            f: pinhole.fx,
            distortion,
        };
        camera.validate_params()?;
        Ok(camera)
    }

    /// Creates a strict BAL pinhole camera with zero distortion.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError`] if `f` is not positive and finite.
    pub fn new_no_distortion(f: f64) -> Result<Self, CameraModelError> {
        let pinhole = PinholeParams::new(f, f, 0.0, 0.0)?;
        let distortion = DistortionModel::Radial { k1: 0.0, k2: 0.0 };
        Self::new(pinhole, distortion)
    }

    /// Returns the radial distortion coefficients `(k1, k2)`.
    fn distortion_params(&self) -> (f64, f64) {
        match self.distortion {
            DistortionModel::Radial { k1, k2 } => (k1, k2),
            _ => (0.0, 0.0),
        }
    }

    /// Returns `true` if `z` is safely in front of the camera (`z < -MIN_DEPTH`).
    fn check_projection_condition(&self, z: f64) -> bool {
        z < -crate::MIN_DEPTH
    }
}

/// Parameter order: `[f, k1, k2]`.
impl From<&BALPinholeCameraStrict> for DVector<f64> {
    fn from(camera: &BALPinholeCameraStrict) -> Self {
        let (k1, k2) = camera.distortion_params();
        DVector::from_vec(vec![camera.f, k1, k2])
    }
}

/// Parameter order: `[f, k1, k2]`.
impl From<&BALPinholeCameraStrict> for [f64; 3] {
    fn from(camera: &BALPinholeCameraStrict) -> Self {
        let (k1, k2) = camera.distortion_params();
        [camera.f, k1, k2]
    }
}

/// Parameter order: `[f, k1, k2]`.
impl TryFrom<&[f64]> for BALPinholeCameraStrict {
    type Error = CameraModelError;

    fn try_from(params: &[f64]) -> Result<Self, Self::Error> {
        if params.len() < 3 {
            return Err(CameraModelError::InvalidParams(format!(
                "BALPinholeCameraStrict requires at least 3 parameters, got {}",
                params.len()
            )));
        }
        Ok(Self {
            f: params[0],
            distortion: DistortionModel::Radial {
                k1: params[1],
                k2: params[2],
            },
        })
    }
}

/// Parameter order: `[f, k1, k2]`.
impl From<[f64; 3]> for BALPinholeCameraStrict {
    fn from(params: [f64; 3]) -> Self {
        Self {
            f: params[0],
            distortion: DistortionModel::Radial {
                k1: params[1],
                k2: params[2],
            },
        }
    }
}

/// Creates a `BALPinholeCameraStrict` from a parameter slice with validation.
///
/// # Errors
///
/// Returns [`CameraModelError::InvalidParams`] if the slice has fewer than 3 elements,
/// or any other [`CameraModelError`] if the resulting parameters are invalid.
pub fn try_from_params(params: &[f64]) -> Result<BALPinholeCameraStrict, CameraModelError> {
    let camera = BALPinholeCameraStrict::try_from(params)?;
    camera.validate_params()?;
    Ok(camera)
}

impl CameraModel for BALPinholeCameraStrict {
    const INTRINSIC_DIM: usize = 3; // f, k1, k2
    type IntrinsicJacobian = SMatrix<f64, 2, 3>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point in camera frame to pixel coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::ProjectionOutOfBounds`] if the point is not in
    /// front of the camera (`z ≥ -MIN_DEPTH`).
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        if !self.check_projection_condition(p_cam.z) {
            return Err(CameraModelError::ProjectionOutOfBounds);
        }
        let inv_neg_z = -1.0 / p_cam.z;
        let x_n = p_cam.x * inv_neg_z;
        let y_n = p_cam.y * inv_neg_z;

        let (k1, k2) = self.distortion_params();
        let r2 = x_n * x_n + y_n * y_n;
        let r4 = r2 * r2;
        let distortion = 1.0 + k1 * r2 + k2 * r4;

        let x_d = x_n * distortion;
        let y_d = y_n * distortion;

        Ok(Vector2::new(self.f * x_d, self.f * y_d))
    }

    /// Returns the 2×3 Jacobian of the projection with respect to the 3D point in
    /// camera frame. See the [BAL pinhole cookbook chapter][chap] for the full
    /// derivation.
    ///
    /// [chap]: ../doc/cookbook/src/bal-pinhole.html
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let inv_neg_z = -1.0 / p_cam.z;
        let x_n = p_cam.x * inv_neg_z;
        let y_n = p_cam.y * inv_neg_z;

        let (k1, k2) = self.distortion_params();
        let r2 = x_n * x_n + y_n * y_n;
        let r4 = r2 * r2;
        let distortion = 1.0 + k1 * r2 + k2 * r4;
        let d_dist_dr2 = k1 + 2.0 * k2 * r2;

        let dxn_dz = x_n * inv_neg_z;
        let dyn_dz = y_n * inv_neg_z;

        let dx_d_dxn = distortion + x_n * d_dist_dr2 * 2.0 * x_n;
        let dx_d_dyn = x_n * d_dist_dr2 * 2.0 * y_n;
        let dy_d_dxn = y_n * d_dist_dr2 * 2.0 * x_n;
        let dy_d_dyn = distortion + y_n * d_dist_dr2 * 2.0 * y_n;

        let du_dx = self.f * (dx_d_dxn * inv_neg_z);
        let du_dy = self.f * (dx_d_dyn * inv_neg_z);
        let du_dz = self.f * (dx_d_dxn * dxn_dz + dx_d_dyn * dyn_dz);

        let dv_dx = self.f * (dy_d_dxn * inv_neg_z);
        let dv_dy = self.f * (dy_d_dyn * inv_neg_z);
        let dv_dz = self.f * (dy_d_dxn * dxn_dz + dy_d_dyn * dyn_dz);

        SMatrix::<f64, 2, 3>::new(du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz)
    }

    /// Returns the pose Jacobian `(∂(u,v)/∂p_cam, ∂p_cam/∂δξ)` for a 3D point in world
    /// frame and a camera-to-world pose. Uses right perturbation on `SE(3)` and the
    /// skew-symmetric cross-product matrix. See the cookbook chapter on
    /// [SE(3) pose Jacobians][pose] for the general formula.
    ///
    /// [pose]: ../doc/cookbook/src/introduction.html#se3-pose-jacobians
    fn jacobian_pose(
        &self,
        p_world: &Vector3<f64>,
        pose: &SE3,
    ) -> (Self::PointJacobian, SMatrix<f64, 3, 6>) {
        let p_cam = pose.act(p_world, None, None);

        let d_uv_d_pcam = self.jacobian_point(&p_cam);

        // Right perturbation on T_wc:
        //   ∂p_cam/∂δρ = R           (cols 0-2)
        //   ∂p_cam/∂δθ = -R·[p_world]×  (cols 3-5)
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

    /// Returns the 2×3 intrinsic Jacobian `∂(u,v)/∂[f, k1, k2]`. See the
    /// [BAL pinhole cookbook chapter][chap] for the derivation.
    ///
    /// [chap]: ../doc/cookbook/src/bal-pinhole.html
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let inv_neg_z = -1.0 / p_cam.z;
        let x_n = p_cam.x * inv_neg_z;
        let y_n = p_cam.y * inv_neg_z;

        let (k1, k2) = self.distortion_params();
        let r2 = x_n * x_n + y_n * y_n;
        let r4 = r2 * r2;
        let distortion = 1.0 + k1 * r2 + k2 * r4;

        let x_d = x_n * distortion;
        let y_d = y_n * distortion;

        SMatrix::<f64, 2, 3>::new(
            x_d,
            self.f * x_n * r2,
            self.f * x_n * r4,
            y_d,
            self.f * y_n * r2,
            self.f * y_n * r4,
        )
    }

    /// Unprojects a 2D pixel to a unit 3D ray in camera frame (BAL convention: ray
    /// has `z = -1/√(1+r²)`). Uses a fixed 5-iteration fixed-point solve to invert
    /// the radial distortion.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let x_d = point_2d.x / self.f;
        let y_d = point_2d.y / self.f;

        let mut x_n = x_d;
        let mut y_n = y_d;

        let (k1, k2) = self.distortion_params();

        for _ in 0..5 {
            let r2 = x_n * x_n + y_n * y_n;
            let distortion = 1.0 + k1 * r2 + k2 * r2 * r2;
            x_n = x_d / distortion;
            y_n = y_d / distortion;
        }

        let norm = (1.0 + x_n * x_n + y_n * y_n).sqrt();
        Ok(Vector3::new(x_n / norm, y_n / norm, -1.0 / norm))
    }

    /// Validates that `f` is positive and finite, and that the distortion
    /// coefficients are finite.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError`] on any violation.
    fn validate_params(&self) -> Result<(), CameraModelError> {
        self.get_pinhole_params().validate()?;
        self.get_distortion().validate()
    }

    /// Returns `fx = fy = f` and `cx = cy = 0`.
    fn get_pinhole_params(&self) -> PinholeParams {
        PinholeParams {
            fx: self.f,
            fy: self.f,
            cx: 0.0,
            cy: 0.0,
        }
    }

    /// Returns the stored distortion model.
    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// Returns the model name `"bal_pinhole"`.
    fn get_model_name(&self) -> &'static str {
        "bal_pinhole"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_bal_strict_camera_creation() -> TestResult {
        let pinhole = PinholeParams::new(500.0, 500.0, 0.0, 0.0)?;
        let distortion = DistortionModel::Radial { k1: 0.4, k2: -0.3 };
        let camera = BALPinholeCameraStrict::new(pinhole, distortion)?;
        let (k1, k2) = camera.distortion_params();

        assert_eq!(camera.f, 500.0);
        assert_eq!(k1, 0.4);
        assert_eq!(k2, -0.3);
        Ok(())
    }

    #[test]
    fn test_bal_strict_rejects_different_focal_lengths() {
        let pinhole = PinholeParams {
            fx: 500.0,
            fy: 505.0, // Different from fx
            cx: 0.0,
            cy: 0.0,
        };
        let distortion = DistortionModel::Radial { k1: 0.0, k2: 0.0 };
        let result = BALPinholeCameraStrict::new(pinhole, distortion);
        assert!(result.is_err());
    }

    #[test]
    fn test_bal_strict_rejects_non_zero_principal_point() {
        let pinhole = PinholeParams {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0, // Non-zero
            cy: 0.0,
        };
        let distortion = DistortionModel::Radial { k1: 0.0, k2: 0.0 };
        let result = BALPinholeCameraStrict::new(pinhole, distortion);
        assert!(result.is_err());
    }

    #[test]
    fn test_bal_strict_projection_at_optical_axis() -> TestResult {
        let camera = BALPinholeCameraStrict::new_no_distortion(500.0)?;
        let p_cam = Vector3::new(0.0, 0.0, -1.0);

        let uv = camera.project(&p_cam)?;

        // Point on optical axis projects to origin (no principal point offset)
        assert!(uv.x.abs() < 1e-10);
        assert!(uv.y.abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_bal_strict_projection_off_axis() -> TestResult {
        let camera = BALPinholeCameraStrict::new_no_distortion(500.0)?;
        let p_cam = Vector3::new(0.1, 0.2, -1.0);

        let uv = camera.project(&p_cam)?;

        // u = 500 * 0.1 = 50 (no principal point offset)
        // v = 500 * 0.2 = 100
        assert!((uv.x - 50.0).abs() < 1e-10);
        assert!((uv.y - 100.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_bal_strict_from_into_traits() -> TestResult {
        let camera = BALPinholeCameraStrict::new_no_distortion(400.0)?;

        // Test conversion to DVector
        let params: DVector<f64> = (&camera).into();
        assert_eq!(params.len(), 3);
        assert_eq!(params[0], 400.0);
        assert_eq!(params[1], 0.0);
        assert_eq!(params[2], 0.0);

        // Test conversion to array
        let arr: [f64; 3] = (&camera).into();
        assert_eq!(arr, [400.0, 0.0, 0.0]);

        // Test conversion from slice
        let params_slice = [450.0, 0.1, 0.01];
        let camera2 = BALPinholeCameraStrict::try_from(&params_slice[..])?;
        let (cam2_k1, cam2_k2) = camera2.distortion_params();
        assert_eq!(camera2.f, 450.0);
        assert_eq!(cam2_k1, 0.1);
        assert_eq!(cam2_k2, 0.01);

        // Test conversion from array
        let camera3 = BALPinholeCameraStrict::from([500.0, 0.2, 0.02]);
        let (cam3_k1, cam3_k2) = camera3.distortion_params();
        assert_eq!(camera3.f, 500.0);
        assert_eq!(cam3_k1, 0.2);
        assert_eq!(cam3_k2, 0.02);

        Ok(())
    }

    #[test]
    fn test_project_unproject_round_trip() -> TestResult {
        let camera = BALPinholeCameraStrict::new_no_distortion(500.0)?;

        // BAL uses -Z convention: points in front of camera have z < 0
        let test_points = [
            Vector3::new(0.1, 0.2, -1.0),
            Vector3::new(-0.3, 0.1, -2.0),
            Vector3::new(0.05, -0.1, -0.5),
        ];

        for p_cam in &test_points {
            let uv = camera.project(p_cam)?;
            let ray = camera.unproject(&uv)?;
            let dot = ray.dot(&p_cam.normalize());
            assert!(
                (dot - 1.0).abs() < 1e-6,
                "Round-trip failed: dot={dot}, expected ~1.0"
            );
        }

        Ok(())
    }

    #[test]
    fn test_jacobian_pose_numerical() -> TestResult {
        use apex_manifolds::LieGroup;
        use apex_manifolds::se3::{SE3, SE3Tangent};

        let camera = BALPinholeCameraStrict::new_no_distortion(500.0)?;

        // BAL uses -Z convention. Use a pose and world point such that
        // pose_inv.act(p_world) has z < 0.
        let pose = SE3::from_translation_euler(0.1, -0.05, 0.2, 0.0, 0.0, 0.0);
        let p_world = Vector3::new(0.1, 0.05, -3.0);

        let (d_uv_d_pcam, d_pcam_d_pose) = camera.jacobian_pose(&p_world, &pose);
        let d_uv_d_pose = d_uv_d_pcam * d_pcam_d_pose;

        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

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
                assert!(
                    analytical.is_finite(),
                    "jacobian_pose[{r},{i}] is not finite"
                );
                let rel_err = (analytical - numerical).abs() / (1.0 + numerical.abs());
                assert!(
                    rel_err < crate::JACOBIAN_TEST_TOLERANCE,
                    "jacobian_pose mismatch at ({r},{i}): analytical={analytical}, numerical={numerical}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_project_returns_error_behind_camera() -> TestResult {
        let camera = BALPinholeCameraStrict::new_no_distortion(500.0)?;
        // BAL: z > 0 is behind camera
        assert!(camera.project(&Vector3::new(0.0, 0.0, 1.0)).is_err());
        Ok(())
    }

    #[test]
    fn test_project_at_min_depth_boundary() -> TestResult {
        let camera = BALPinholeCameraStrict::new_no_distortion(500.0)?;
        // BAL: min depth is in negative-z direction
        let p_min = Vector3::new(0.0, 0.0, -crate::MIN_DEPTH);
        if let Ok(uv) = camera.project(&p_min) {
            assert!(uv.x.is_finite() && uv.y.is_finite());
        }
        Ok(())
    }
}
