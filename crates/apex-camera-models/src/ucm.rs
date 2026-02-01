//! Unified Camera Model (UCM)
//!
//! A generic camera model for catadioptric and fisheye cameras using
//! a single parameter α for projection onto a unit sphere.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! d = √(x² + y² + z²)
//! denom = α·d + (1-α)·z
//! u = fx · (x/denom) + cx
//! v = fy · (y/denom) + cy
//! ```
//!
//! where α is the projection parameter (typically α ∈ [0, 1]).
//!
//! ## Unprojection (2D → 3D)
//!
//! Algebraic solution using the UCM inverse equations.
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy
//! - **Distortion**: α (projection parameter) (5 parameters total)
//!
//! # Use Cases
//!
//! - Catadioptric systems (mirror-based omnidirectional cameras)
//! - Fisheye lenses
//! - Wide-angle cameras
//!
//! # References
//!
//! - Geyer & Daniilidis, "A Unifying Theory for Central Panoramic Systems"

use crate::{
    CameraModel, CameraModelError, DistortionModel, PinholeParams, Resolution, skew_symmetric,
};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Unified Camera Model with 5 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UcmCamera {
    /// Linear pinhole parameters (fx, fy, cx, cy)
    pub pinhole: PinholeParams,
    /// Lens distortion model and parameters
    pub distortion: DistortionModel,
    /// Image resolution
    pub resolution: Resolution,
}

impl UcmCamera {
    /// Create a new Unified Camera Model (UCM) camera.
    ///
    /// # Arguments
    ///
    /// * `pinhole` - Pinhole parameters (fx, fy, cx, cy)
    /// * `distortion` - MUST be DistortionModel::UCM { alpha, beta }
    /// * `resolution` - Image resolution
    ///
    /// # Errors
    ///
    /// Returns `CameraModelError::InvalidParams` if `distortion` is not `DistortionModel::UCM`.
    pub fn new(
        pinhole: PinholeParams,
        distortion: DistortionModel,
        resolution: crate::Resolution,
    ) -> Result<Self, CameraModelError> {
        let camera = Self {
            pinhole,
            distortion,
            resolution,
        };
        camera.validate_params()?;
        Ok(camera)
    }

    /// Helper method to extract distortion parameter, returning Result for consistency.
    fn distortion_params(&self) -> Result<f64, CameraModelError> {
        match self.distortion {
            DistortionModel::UCM { alpha } => Ok(alpha),
            _ => Err(CameraModelError::InvalidParams(format!(
                "UcmCamera requires UCM distortion model, got {:?}",
                self.distortion
            ))),
        }
    }

    /// Checks the geometric condition for a valid projection.
    pub fn check_projection_condition(&self, z: f64, d: f64) -> bool {
        let alpha = self
            .distortion_params()
            .expect("UcmCamera validated at construction");
        let w = if alpha <= 0.5 {
            alpha / (1.0 - alpha)
        } else {
            (1.0 - alpha) / alpha
        };
        z > -w * d
    }

    fn check_unprojection_condition(&self, r_squared: f64) -> bool {
        let alpha = self
            .distortion_params()
            .expect("UcmCamera validated at construction");
        if alpha > 0.5 {
            let gamma = 1.0 - alpha;
            r_squared <= gamma * gamma / (2.0 * alpha - 1.0)
        } else {
            true
        }
    }
}

/// Convert camera to dynamic vector of intrinsic parameters.
///
/// # Layout
///
/// The parameters are ordered as: [fx, fy, cx, cy, alpha]
impl From<&UcmCamera> for DVector<f64> {
    fn from(camera: &UcmCamera) -> Self {
        let alpha = camera
            .distortion_params()
            .expect("UcmCamera validated at construction");
        DVector::from_vec(vec![
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            alpha,
        ])
    }
}

/// Convert camera to fixed-size array of intrinsic parameters.
///
/// # Layout
///
/// The parameters are ordered as: [fx, fy, cx, cy, alpha]
impl From<&UcmCamera> for [f64; 5] {
    fn from(camera: &UcmCamera) -> Self {
        let alpha = camera
            .distortion_params()
            .expect("UcmCamera validated at construction");
        [
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            alpha,
        ]
    }
}

/// Create camera from slice of intrinsic parameters.
///
/// # Layout
///
/// Expected parameter order: [fx, fy, cx, cy, alpha]
///
/// # Panics
///
/// Panics if the slice has fewer than 5 elements.
impl From<&[f64]> for UcmCamera {
    fn from(params: &[f64]) -> Self {
        assert!(
            params.len() >= 5,
            "UcmCamera requires at least 5 parameters, got {}",
            params.len()
        );
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::UCM { alpha: params[4] },
            resolution: Resolution {
                width: 0,
                height: 0,
            },
        }
    }
}

/// Create camera from fixed-size array of intrinsic parameters.
///
/// # Layout
///
/// Expected parameter order: [fx, fy, cx, cy, alpha]
impl From<[f64; 5]> for UcmCamera {
    fn from(params: [f64; 5]) -> Self {
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::UCM { alpha: params[4] },
            resolution: Resolution {
                width: 0,
                height: 0,
            },
        }
    }
}

impl CameraModel for UcmCamera {
    const INTRINSIC_DIM: usize = 5;
    type IntrinsicJacobian = SMatrix<f64, 2, 5>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// d = √(x² + y² + z²)
    /// denom = α·d + (1-α)·z
    /// u = fx · (x/denom) + cx
    /// v = fy · (y/denom) + cy
    /// ```
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame
    ///
    /// # Returns
    ///
    /// - `Some(uv)` - 2D image coordinates if valid
    /// - `None` - If denom < crate::GEOMETRIC_PRECISION or projection condition fails
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let d = (x * x + y * y + z * z).sqrt();
        let alpha = self
            .distortion_params()
            .expect("UcmCamera validated at construction");
        let denom = alpha * d + (1.0 - alpha) * z;

        // Check projection validity
        if !self.check_projection_condition(z, d) {
            // return Err(CameraModelError::InvalidProjection {
            //     message: format!(
            //         "UCM projection condition failed: z={}, d={}, alpha={}",
            //         z, d, alpha
            //     ),
            // });
            return Err(CameraModelError::PointAtCameraCenter);
        }

        if denom < crate::GEOMETRIC_PRECISION {
            // return Err(CameraModelError::InvalidProjection {
            //     message: format!(
            //         "UCM denominator too small: denom={} < {}",
            //         denom,
            //         crate::GEOMETRIC_PRECISION
            //     ),
            // });
            return Err(CameraModelError::PointAtCameraCenter);
        }

        Ok(Vector2::new(
            self.pinhole.fx * x / denom + self.pinhole.cx,
            self.pinhole.fy * y / denom + self.pinhole.cy,
        ))
    }

    /// Unprojects a 2D image point to a 3D ray.
    ///
    /// # Algorithm
    ///
    /// Algebraic solution for UCM inverse projection.
    ///
    /// # Arguments
    ///
    /// * `point_2d` - 2D point in image coordinates
    ///
    /// # Returns
    ///
    /// - `Ok(ray)` - Normalized 3D ray direction
    /// - `Err` - If unprojection condition fails
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;
        let alpha = self.distortion_params()?;
        let gamma = 1.0 - alpha;
        let xi = alpha / gamma;
        let mx = (u - self.pinhole.cx) / self.pinhole.fx * gamma;
        let my = (v - self.pinhole.cy) / self.pinhole.fy * gamma;

        let r_squared = mx * mx + my * my;

        // Check unprojection condition
        if !self.check_unprojection_condition(r_squared) {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let num = xi + (1.0 + (1.0 - xi * xi) * r_squared).sqrt();
        let denom = 1.0 - r_squared;

        if denom < crate::GEOMETRIC_PRECISION {
            return Err(CameraModelError::PointIsOutSideImage);
        }

        let coeff = num / denom;

        let point3d = Vector3::new(coeff * mx, coeff * my, coeff) - Vector3::new(0.0, 0.0, xi);

        Ok(point3d.normalize())
    }

    /// Checks if a 3D point can be validly projected.
    ///
    /// # Validity Conditions
    ///
    /// - denom = α·d + (1-α)·z must be ≥ PRECISION
    /// - Projection condition: z > -w·d where w depends on α
    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let d = (x * x + y * y + z * z).sqrt();
        let alpha = self
            .distortion_params()
            .expect("UcmCamera validated at construction");
        let denom = alpha * d + (1.0 - alpha) * z;

        denom >= crate::GEOMETRIC_PRECISION && self.check_projection_condition(z, d)
    }

    /// Computes the Jacobian of the projection function with respect to the 3D point in camera frame.
    ///
    /// # Mathematical Derivation
    ///
    /// The UCM projection model maps a 3D point p = (x, y, z) to 2D pixel coordinates (u, v).
    ///
    /// ## Projection Steps
    ///
    /// 1. Compute point distance:
    ///    ```text
    ///    ρ = √(x² + y² + z²)
    ///    ```
    ///
    /// 2. Unified projection denominator:
    ///    ```text
    ///    D = α·ρ + (1-α)·z
    ///    ```
    ///
    /// 3. Pixel coordinates:
    ///    ```text
    ///    u = fx · (x/D) + cx
    ///    v = fy · (y/D) + cy
    ///    ```
    ///
    /// ## Jacobian Computation
    ///
    /// Derivatives of D with respect to (x, y, z):
    /// ```text
    /// ∂D/∂x = α · (x/ρ)
    /// ∂D/∂y = α · (y/ρ)
    /// ∂D/∂z = α · (z/ρ) + (1-α)
    /// ```
    ///
    /// Using the quotient rule for u = fx · (x/D):
    /// ```text
    /// ∂u/∂x = fx · (D - x·∂D/∂x) / D²
    /// ∂u/∂y = fx · (-x·∂D/∂y) / D²
    /// ∂u/∂z = fx · (-x·∂D/∂z) / D²
    /// ```
    ///
    /// Similarly for v:
    /// ```text
    /// ∂v/∂x = fy · (-y·∂D/∂x) / D²
    /// ∂v/∂y = fy · (D - y·∂D/∂y) / D²
    /// ∂v/∂z = fy · (-y·∂D/∂z) / D²
    /// ```
    ///
    /// # References
    ///
    /// - Geyer & Daniilidis, "A Unifying Theory for Central Panoramic Systems", ICCV 2000
    /// - Mei & Rives, "Single View Point Omnidirectional Camera Calibration from Planar Grids", ICRA 2007
    ///
    /// # Verification
    ///
    /// This Jacobian is verified against numerical differentiation in tests.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let rho = (x * x + y * y + z * z).sqrt();
        let alpha = self
            .distortion_params()
            .expect("UcmCamera validated at construction");

        // Denominator D = alpha * rho + (1 - alpha) * z
        // Partial derivatives of D:
        // ∂D/∂x = alpha * x / rho
        // ∂D/∂y = alpha * y / rho
        // ∂D/∂z = alpha * z / rho + (1 - alpha)

        let d_denom_dx = alpha * x / rho;
        let d_denom_dy = alpha * y / rho;
        let d_denom_dz = alpha * z / rho + (1.0 - alpha);

        let denom = alpha * rho + (1.0 - alpha) * z;

        // u = fx * x / denom + cx
        // v = fy * y / denom + cy

        // ∂u/∂x = fx * (denom - x * ∂D/∂x) / denom²
        // ∂u/∂y = fx * (-x * ∂D/∂y) / denom²
        // ∂u/∂z = fx * (-x * ∂D/∂z) / denom²

        // ∂v/∂x = fy * (-y * ∂D/∂x) / denom²
        // ∂v/∂y = fy * (denom - y * ∂D/∂y) / denom²
        // ∂v/∂z = fy * (-y * ∂D/∂z) / denom²

        let denom2 = denom * denom;

        let mut jac = SMatrix::<f64, 2, 3>::zeros();

        jac[(0, 0)] = self.pinhole.fx * (denom - x * d_denom_dx) / denom2;
        jac[(0, 1)] = self.pinhole.fx * (-x * d_denom_dy) / denom2;
        jac[(0, 2)] = self.pinhole.fx * (-x * d_denom_dz) / denom2;

        jac[(1, 0)] = self.pinhole.fy * (-y * d_denom_dx) / denom2;
        jac[(1, 1)] = self.pinhole.fy * (denom - y * d_denom_dy) / denom2;
        jac[(1, 2)] = self.pinhole.fy * (-y * d_denom_dz) / denom2;

        jac
    }

    /// Computes the Jacobian of the projection function with respect to the camera pose.
    ///
    /// # Mathematical Derivation
    ///
    /// The pose Jacobian derivation is identical across camera models. We use SE(3) right perturbation
    /// and compute ∂p_cam/∂ξ = [-I | [p_cam]×], then chain with the point Jacobian.
    ///
    /// See BALPinholeCameraStrict::jacobian_pose() or DoubleSphereCamera::jacobian_pose()
    /// for the complete derivation.
    ///
    /// # References
    ///
    /// - Solà et al., "A Micro Lie Theory for State Estimation in Robotics", arXiv:1812.01537, 2018
    ///
    /// # Verification
    ///
    /// This Jacobian is verified against numerical differentiation in tests.
    fn jacobian_pose(
        &self,
        p_world: &Vector3<f64>,
        pose: &SE3,
    ) -> (Self::PointJacobian, SMatrix<f64, 3, 6>) {
        let pose_inv = pose.inverse(None);
        let p_cam = pose_inv.act(p_world, None, None);

        let d_uv_d_pcam = self.jacobian_point(&p_cam);

        let r_transpose = pose_inv.rotation_so3().rotation_matrix();
        let p_cam_skew = skew_symmetric(&p_cam);

        let d_pcam_d_pose = SMatrix::<f64, 3, 6>::from_fn(|r, c| {
            if c < 3 {
                -r_transpose[(r, c)]
            } else {
                p_cam_skew[(r, c - 3)]
            }
        });

        (d_uv_d_pcam, d_pcam_d_pose)
    }

    /// Computes the Jacobian of the projection function with respect to intrinsic parameters.
    ///
    /// # Mathematical Derivation
    ///
    /// The UCM camera has 5 intrinsic parameters: θ = [fx, fy, cx, cy, α]
    ///
    /// ## Projection Model
    ///
    /// ```text
    /// u = fx · (x/D) + cx
    /// v = fy · (y/D) + cy
    /// ```
    ///
    /// Where D = α·ρ + (1-α)·z and ρ = √(x²+y²+z²)
    ///
    /// ## Jacobian Structure
    ///
    /// Linear parameters (fx, fy, cx, cy):
    /// ```text
    /// ∂u/∂fx = x/D,  ∂u/∂fy = 0,    ∂u/∂cx = 1,    ∂u/∂cy = 0
    /// ∂v/∂fx = 0,    ∂v/∂fy = y/D,  ∂v/∂cx = 0,    ∂v/∂cy = 1
    /// ```
    ///
    /// Projection parameter α:
    /// ```text
    /// ∂D/∂α = ρ - z
    /// ∂u/∂α = -fx · (x/D²) · (ρ - z)
    /// ∂v/∂α = -fy · (y/D²) · (ρ - z)
    /// ```
    ///
    /// # References
    ///
    /// - Geyer & Daniilidis, "A Unifying Theory for Central Panoramic Systems", ICCV 2000
    ///
    /// # Verification
    ///
    /// This Jacobian is verified against numerical differentiation in tests.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let rho = (x * x + y * y + z * z).sqrt();
        let alpha = self
            .distortion_params()
            .expect("UcmCamera validated at construction");
        let denom = alpha * rho + (1.0 - alpha) * z;

        let x_norm = x / denom;
        let y_norm = y / denom;

        let u_cx = self.pinhole.fx * x_norm;
        let v_cy = self.pinhole.fy * y_norm;

        let mut jac = SMatrix::<f64, 2, 5>::zeros();

        // ∂u/∂fx = x / denom
        jac[(0, 0)] = x_norm;

        // ∂v/∂fy = y / denom
        jac[(1, 1)] = y_norm;

        // ∂u/∂cx = 1
        jac[(0, 2)] = 1.0;

        // ∂v/∂cy = 1
        jac[(1, 3)] = 1.0;

        // ∂denom/∂alpha = rho - z
        let d_denom_d_alpha = rho - z;

        // ∂u/∂alpha = -fx * x / denom² * (rho - z) = -u_cx * (rho - z) / denom
        jac[(0, 4)] = -u_cx * d_denom_d_alpha / denom;
        jac[(1, 4)] = -v_cy * d_denom_d_alpha / denom;

        jac
    }

    /// Validates camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - fx, fy must be positive (> 0)
    /// - cx, cy must be finite
    /// - α must be finite
    fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.pinhole.fx <= 0.0 || self.pinhole.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthMustBePositive);
        }

        if !self.pinhole.cx.is_finite() || !self.pinhole.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointMustBeFinite);
        }

        let alpha = self.distortion_params()?;
        if !alpha.is_finite() {
            return Err(CameraModelError::InvalidParams(
                "alpha must be finite".to_string(),
            ));
        }

        Ok(())
    }

    fn get_pinhole_params(&self) -> PinholeParams {
        PinholeParams {
            fx: self.pinhole.fx,
            fy: self.pinhole.fy,
            cx: self.pinhole.cx,
            cy: self.pinhole.cy,
        }
    }

    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    fn get_model_name(&self) -> &'static str {
        "ucm"
    }
}

// ============================================================================
// From/Into Trait Implementations for UcmCamera
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_ucm_camera_creation() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.5 };
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = UcmCamera::new(pinhole, distortion, resolution)?;

        assert_eq!(camera.pinhole.fx, 300.0);
        assert_eq!(camera.distortion_params()?, 0.5);
        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.5 };
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = UcmCamera::new(pinhole, distortion, resolution)?;

        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam)?;
        assert!((uv.x - 320.0).abs() < crate::PROJECTION_TEST_TOLERANCE);
        assert!((uv.y - 240.0).abs() < crate::PROJECTION_TEST_TOLERANCE);
        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.6 };
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = UcmCamera::new(pinhole, distortion, resolution)?;

        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_point(&p_cam);
        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

        for i in 0..3 {
            let mut p_plus = p_cam;
            let mut p_minus = p_cam;
            p_plus[i] += eps;
            p_minus[i] -= eps;

            let uv_plus = camera.project(&p_plus)?;
            let uv_minus = camera.project(&p_minus)?;
            let num_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                let diff = (jac_analytical[(r, i)] - num_jac[r]).abs();
                assert!(
                    diff < crate::JACOBIAN_TEST_TOLERANCE,
                    "Mismatch at ({}, {}): {} vs {}",
                    r,
                    i,
                    jac_analytical[(r, i)],
                    num_jac[r]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_jacobian_intrinsics_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.6 };
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = UcmCamera::new(pinhole, distortion, resolution)?;

        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params: DVector<f64> = (&camera).into();
        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

        for i in 0..5 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = UcmCamera::from(params_plus.as_slice());
            let cam_minus = UcmCamera::from(params_minus.as_slice());

            let uv_plus = cam_plus.project(&p_cam)?;
            let uv_minus = cam_minus.project(&p_cam)?;
            let num_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                let diff = (jac_analytical[(r, i)] - num_jac[r]).abs();
                assert!(
                    diff < crate::JACOBIAN_TEST_TOLERANCE,
                    "Mismatch at ({}, {}): {} vs {}",
                    r,
                    i,
                    jac_analytical[(r, i)],
                    num_jac[r]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_ucm_from_into_traits() -> TestResult {
        let pinhole = PinholeParams::new(400.0, 410.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.7 };
        let resolution = crate::Resolution {
            width: 640,
            height: 480,
        };
        let camera = UcmCamera::new(pinhole, distortion, resolution)?;

        // Test conversion to DVector
        let params: DVector<f64> = (&camera).into();
        assert_eq!(params.len(), 5);
        assert_eq!(params[0], 400.0);
        assert_eq!(params[1], 410.0);
        assert_eq!(params[2], 320.0);
        assert_eq!(params[3], 240.0);
        assert_eq!(params[4], 0.7);

        // Test conversion to array
        let arr: [f64; 5] = (&camera).into();
        assert_eq!(arr, [400.0, 410.0, 320.0, 240.0, 0.7]);

        // Test conversion from slice
        let params_slice = [450.0, 460.0, 330.0, 250.0, 0.8];
        let camera2 = UcmCamera::from(&params_slice[..]);
        assert_eq!(camera2.pinhole.fx, 450.0);
        assert_eq!(camera2.pinhole.fy, 460.0);
        assert_eq!(camera2.pinhole.cx, 330.0);
        assert_eq!(camera2.pinhole.cy, 250.0);
        assert_eq!(camera2.distortion_params()?, 0.8);

        // Test conversion from array
        let camera3 = UcmCamera::from([500.0, 510.0, 340.0, 260.0, 0.9]);
        assert_eq!(camera3.pinhole.fx, 500.0);
        assert_eq!(camera3.pinhole.fy, 510.0);
        assert_eq!(camera3.distortion_params()?, 0.9);

        Ok(())
    }
}
