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

use crate::{skew_symmetric, CameraModel, CameraModelError, DistortionModel, PinholeParams};
use apex_manifolds::se3::SE3;
use apex_manifolds::LieGroup;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// Unified Camera Model with 5 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UcmCamera {
    pub pinhole: PinholeParams,
    pub distortion: DistortionModel,
}

impl UcmCamera {
    /// Create a new Unified Camera Model (UCM) camera.
    ///
    /// # Arguments
    ///
    /// * `pinhole` - Pinhole parameters (fx, fy, cx, cy).
    /// * `distortion` - MUST be [`DistortionModel::UCM`] with `alpha`.
    ///
    /// # Returns
    ///
    /// Returns a new `UcmCamera` instance if the distortion model matches.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::InvalidParams`] if `distortion` is not [`DistortionModel::UCM`].
    pub fn new(
        pinhole: PinholeParams,
        distortion: DistortionModel,
    ) -> Result<Self, CameraModelError> {
        let camera = Self {
            pinhole,
            distortion,
        };
        camera.validate_params()?;
        Ok(camera)
    }

    /// Helper method to extract distortion parameter.
    ///
    /// # Returns
    ///
    /// Returns the `alpha` parameter for UCM.
    /// If the distortion model is incorrect (which shouldn't happen for valid instances), returns `0.0`.
    fn distortion_params(&self) -> f64 {
        match self.distortion {
            DistortionModel::UCM { alpha } => alpha,
            _ => 0.0,
        }
    }

    /// Checks the geometric condition for a valid projection.
    ///
    /// # Arguments
    ///
    /// * `z` - The z-coordinate of the point.
    /// * `d` - The Euclidean distance `√(x² + y² + z²)`.
    ///
    /// # Returns
    ///
    /// Returns `true` if `z > -w * d`, where `w` depends on `alpha`.
    fn check_projection_condition(&self, z: f64, d: f64) -> bool {
        let alpha = self.distortion_params();
        let w = if alpha <= 0.5 {
            alpha / (1.0 - alpha)
        } else {
            (1.0 - alpha) / alpha
        };
        z > -w * d
    }

    /// Checks the geometric condition for a valid unprojection.
    ///
    /// # Arguments
    ///
    /// * `r_squared` - The squared radius in normalized image coordinates.
    ///
    /// # Returns
    ///
    /// Returns `true` if the point satisfies the unprojection condition.
    fn check_unprojection_condition(&self, r_squared: f64) -> bool {
        let alpha = self.distortion_params();
        if alpha > 0.5 {
            let gamma = 1.0 - alpha;
            r_squared <= gamma * gamma / (2.0 * alpha - 1.0)
        } else {
            true
        }
    }

    /// Performs linear estimation to initialize the alpha parameter from point correspondences.
    ///
    /// This method estimates the `alpha` parameter using a linear least squares approach
    /// given 3D-2D point correspondences. It assumes the intrinsic parameters (fx, fy, cx, cy)
    /// are already set.
    ///
    /// # Arguments
    ///
    /// * `points_3d`: Matrix3xX<f64> - 3D points in camera coordinates (each column is a point)
    /// * `points_2d`: Matrix2xX<f64> - Corresponding 2D points in image coordinates
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success or a `CameraModelError` if the estimation fails.
    pub fn linear_estimation(
        &mut self,
        points_3d: &nalgebra::Matrix3xX<f64>,
        points_2d: &nalgebra::Matrix2xX<f64>,
    ) -> Result<(), CameraModelError> {
        // Check if the number of 2D and 3D points match
        if points_2d.ncols() != points_3d.ncols() {
            return Err(CameraModelError::InvalidParams(
                "Number of 2D and 3D points must match".to_string(),
            ));
        }

        // Set up the linear system to solve for alpha
        let num_points = points_2d.ncols();
        let mut a = nalgebra::DMatrix::zeros(num_points * 2, 1);
        let mut b = nalgebra::DVector::zeros(num_points * 2);

        for i in 0..num_points {
            let x = points_3d[(0, i)];
            let y = points_3d[(1, i)];
            let z = points_3d[(2, i)];
            let u = points_2d[(0, i)];
            let v = points_2d[(1, i)];

            let d = (x * x + y * y + z * z).sqrt();
            let u_cx = u - self.pinhole.cx;
            let v_cy = v - self.pinhole.cy;

            a[(i * 2, 0)] = u_cx * (d - z);
            a[(i * 2 + 1, 0)] = v_cy * (d - z);

            b[i * 2] = (self.pinhole.fx * x) - (u_cx * z);
            b[i * 2 + 1] = (self.pinhole.fy * y) - (v_cy * z);
        }

        // Solve the linear system using SVD
        let svd = a.svd(true, true);
        let alpha = match svd.solve(&b, 1e-10) {
            Ok(sol) => sol[0],
            Err(err_msg) => {
                return Err(CameraModelError::NumericalError {
                    operation: "svd_solve".to_string(),
                    details: err_msg.to_string(),
                });
            }
        };

        self.distortion = DistortionModel::UCM { alpha };

        self.validate_params()?;

        Ok(())
    }
}

/// Convert camera to dynamic vector of intrinsic parameters.
///
/// # Layout
///
/// The parameters are ordered as: [fx, fy, cx, cy, alpha]
impl From<&UcmCamera> for DVector<f64> {
    fn from(camera: &UcmCamera) -> Self {
        let alpha = camera.distortion_params();
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
        let alpha = camera.distortion_params();
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
    /// * `p_cam` - 3D point in camera coordinate frame.
    ///
    /// # Returns
    ///
    /// - `Ok(uv)` - 2D image coordinates if valid.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::PointAtCameraCenter`] if the projection condition fails or the denominator is too small.
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let d = (x * x + y * y + z * z).sqrt();
        let alpha = self.distortion_params();
        let denom = alpha * d + (1.0 - alpha) * z;

        // Check projection validity
        if !self.check_projection_condition(z, d) {
            return Err(CameraModelError::PointBehindCamera {
                z,
                min_z: crate::GEOMETRIC_PRECISION,
            });
        }

        if denom < crate::GEOMETRIC_PRECISION {
            return Err(CameraModelError::DenominatorTooSmall {
                denom,
                threshold: crate::GEOMETRIC_PRECISION,
            });
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
    /// * `point_2d` - 2D point in image coordinates.
    ///
    /// # Returns
    ///
    /// - `Ok(ray)` - Normalized 3D ray direction.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::PointOutsideImage`] if the unprojection condition fails.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;
        let alpha = self.distortion_params();
        let gamma = 1.0 - alpha;
        let xi = alpha / gamma;
        let mx = (u - self.pinhole.cx) / self.pinhole.fx * gamma;
        let my = (v - self.pinhole.cy) / self.pinhole.fy * gamma;

        let r_squared = mx * mx + my * my;

        // Check unprojection condition
        if !self.check_unprojection_condition(r_squared) {
            return Err(CameraModelError::PointOutsideImage { x: u, y: v });
        }

        let num = xi + (1.0 + (1.0 - xi * xi) * r_squared).sqrt();
        let denom = 1.0 - r_squared;

        if denom < crate::GEOMETRIC_PRECISION {
            return Err(CameraModelError::PointOutsideImage { x: u, y: v });
        }

        let coeff = num / denom;

        let point3d = Vector3::new(coeff * mx, coeff * my, coeff) - Vector3::new(0.0, 0.0, xi);

        Ok(point3d.normalize())
    }

    /// Checks if a 3D point can be validly projected.
    ///
    /// # Validity Conditions
    ///
    /// - `denom = α·d + (1-α)·z` must be ≥ PRECISION.
    /// - Point must satisfy the specific UCM projection condition.
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame.
    ///
    /// # Returns
    ///
    /// Returns `true` if the point projects to a valid image coordinate, `false` otherwise.
    fn is_valid_point(&self, p_cam: &Vector3<f64>) -> bool {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let d = (x * x + y * y + z * z).sqrt();
        let alpha = self.distortion_params();
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
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame.
    ///
    /// # Returns
    ///
    /// Returns the 2x3 Jacobian matrix.
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
        let alpha = self.distortion_params();

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
    /// # Arguments
    ///
    /// * `p_world` - 3D point in world coordinate frame.
    /// * `pose` - The camera pose in SE(3).
    ///
    /// # Returns
    ///
    /// Returns a tuple `(d_uv_d_pcam, d_pcam_d_pose)`:
    /// - `d_uv_d_pcam`: 2×3 Jacobian of projection w.r.t. point in camera frame
    /// - `d_pcam_d_pose`: 3×6 Jacobian of camera point w.r.t. pose perturbation
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
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame.
    ///
    /// # Returns
    ///
    /// Returns the 2x5 Intrinsic Jacobian matrix.
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
        let alpha = self.distortion_params();
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
    /// - `fx`, `fy` must be positive.
    /// - `fx`, `fy` must be finite.
    /// - `cx`, `cy` must be finite.
    /// - `α` must be in [0, 1].
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError`] if any parameter violates validation rules.
    fn validate_params(&self) -> Result<(), CameraModelError> {
        if self.pinhole.fx <= 0.0 || self.pinhole.fy <= 0.0 {
            return Err(CameraModelError::FocalLengthNotPositive {
                fx: self.pinhole.fx,
                fy: self.pinhole.fy,
            });
        }

        if !self.pinhole.fx.is_finite() || !self.pinhole.fy.is_finite() {
            return Err(CameraModelError::FocalLengthNotFinite {
                fx: self.pinhole.fx,
                fy: self.pinhole.fy,
            });
        }

        if !self.pinhole.cx.is_finite() || !self.pinhole.cy.is_finite() {
            return Err(CameraModelError::PrincipalPointNotFinite {
                cx: self.pinhole.cx,
                cy: self.pinhole.cy,
            });
        }

        let alpha = self.distortion_params();
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(CameraModelError::ParameterOutOfRange {
                param: "alpha".to_string(),
                value: alpha,
                min: 0.0,
                max: 1.0,
            });
        }

        Ok(())
    }

    /// Returns the pinhole parameters of the camera.
    ///
    /// # Returns
    ///
    /// A [`PinholeParams`] struct containing the focal lengths (fx, fy) and principal point (cx, cy).
    fn get_pinhole_params(&self) -> PinholeParams {
        PinholeParams {
            fx: self.pinhole.fx,
            fy: self.pinhole.fy,
            cx: self.pinhole.cx,
            cy: self.pinhole.cy,
        }
    }

    /// Returns the distortion model and parameters of the camera.
    ///
    /// # Returns
    ///
    /// The [`DistortionModel`] associated with this camera (typically [`DistortionModel::UCM`]).
    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// Returns the string identifier for the camera model.
    ///
    /// # Returns
    ///
    /// The string `"ucm"`.
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
        let camera = UcmCamera::new(pinhole, distortion)?;

        assert_eq!(camera.pinhole.fx, 300.0);
        assert_eq!(camera.distortion_params(), 0.5);
        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::UCM { alpha: 0.5 };
        let camera = UcmCamera::new(pinhole, distortion)?;

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
        let camera = UcmCamera::new(pinhole, distortion)?;

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
        let camera = UcmCamera::new(pinhole, distortion)?;

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
        let camera = UcmCamera::new(pinhole, distortion)?;

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
        assert_eq!(camera2.distortion_params(), 0.8);

        // Test conversion from array
        let camera3 = UcmCamera::from([500.0, 510.0, 340.0, 260.0, 0.9]);
        assert_eq!(camera3.pinhole.fx, 500.0);
        assert_eq!(camera3.pinhole.fy, 510.0);
        assert_eq!(camera3.distortion_params(), 0.9);

        Ok(())
    }
}
