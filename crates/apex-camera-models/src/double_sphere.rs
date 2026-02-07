//! Double Sphere Camera Model
//!
//! A two-parameter fisheye model that provides improved accuracy over
//! the Unified Camera Model by using two sphere projections.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! d₁ = √(x² + y² + z²)
//! d₂ = √(x² + y² + (ξ·d₁ + z)²)
//! denom = α·d₂ + (1-α)·(ξ·d₁ + z)
//! u = fx · (x/denom) + cx
//! v = fy · (y/denom) + cy
//! ```
//!
//! where:
//! - ξ (xi) is the first distortion parameter
//! - α (alpha) ∈ (0, 1] is the second distortion parameter
//!
//! ## Unprojection (2D → 3D)
//!
//! Algebraic solution using the double sphere inverse equations.
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy
//! - **Distortion**: ξ (xi), α (alpha) (6 parameters total)
//!
//! # Use Cases
//!
//! - High-quality fisheye calibration
//! - Wide field-of-view cameras
//! - More accurate than UCM for extreme wide-angle lenses
//!
//! # References
//!
//! - Usenko et al., "The Double Sphere Camera Model", 3DV 2018

use crate::{CameraModel, CameraModelError, DistortionModel, PinholeParams, skew_symmetric};
use apex_manifolds::LieGroup;
use apex_manifolds::se3::SE3;
use nalgebra::{DVector, SMatrix, Vector2, Vector3};
use std::fmt;

/// Double Sphere camera model with 6 parameters.
#[derive(Clone, Copy, PartialEq)]
pub struct DoubleSphereCamera {
    pub pinhole: PinholeParams,
    pub distortion: DistortionModel,
}

impl DoubleSphereCamera {
    /// Creates a new Double Sphere camera model.
    ///
    /// # Arguments
    ///
    /// * `pinhole` - Pinhole camera parameters (fx, fy, cx, cy).
    /// * `distortion` - Distortion model (must be [`DistortionModel::DoubleSphere`]).
    ///
    /// # Returns
    ///
    /// Returns a new `DoubleSphereCamera` instance if the parameters are valid.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError`] if:
    /// - The distortion model is not `DoubleSphere`.
    /// - Parameters are invalid (e.g., negative focal length, invalid alpha).
    ///
    /// # Example
    ///
    /// ```
    /// use apex_camera_models::{DoubleSphereCamera, PinholeParams, DistortionModel};
    ///
    /// let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
    /// let distortion = DistortionModel::DoubleSphere { xi: -0.2, alpha: 0.6 };
    /// let camera = DoubleSphereCamera::new(pinhole, distortion)?;
    /// # Ok::<(), apex_camera_models::CameraModelError>(())
    /// ```
    pub fn new(
        pinhole: PinholeParams,
        distortion: DistortionModel,
    ) -> Result<Self, CameraModelError> {
        let model = Self {
            pinhole,
            distortion,
        };
        model.validate_params()?;
        Ok(model)
    }

    /// Helper method to extract Double Sphere distortion parameters.
    ///
    /// This method assumes the caller has already verified the camera model.
    ///
    /// # Returns
    ///
    /// Returns a tuple `(xi, alpha)` containing the Double Sphere parameters.
    fn distortion_params(&self) -> (f64, f64) {
        match self.distortion {
            DistortionModel::DoubleSphere { xi, alpha } => (xi, alpha),
            _ => (0.0, 0.0),
        }
    }

    /// Checks the geometric condition for a valid projection.
    ///
    /// # Arguments
    ///
    /// * `z` - The z-coordinate of the point in the camera frame.
    /// * `d1` - The Euclidean distance to the point.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the point satisfies the projection condition, or `Ok(false)` otherwise.
    fn check_projection_condition(&self, z: f64, d1: f64) -> Result<bool, CameraModelError> {
        let (xi, alpha) = self.distortion_params();
        let w1 = if alpha > 0.5 {
            (1.0 - alpha) / alpha
        } else {
            alpha / (1.0 - alpha)
        };
        let w2 = (w1 + xi) / (2.0 * w1 * xi + xi * xi + 1.0).sqrt();
        Ok(z > -w2 * d1)
    }

    /// Checks the geometric condition for a valid unprojection.
    ///
    /// # Arguments
    ///
    /// * `r_squared` - The squared radius of the point in normalized image coordinates.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the point satisfies the unprojection condition, or `Ok(false)` otherwise.
    fn check_unprojection_condition(&self, r_squared: f64) -> Result<bool, CameraModelError> {
        let (alpha, _) = self.distortion_params();
        if alpha > 0.5 && r_squared > 1.0 / (2.0 * alpha - 1.0) {
            return Ok(false);
        }
        Ok(true)
    }

    /// Performs linear estimation to initialize distortion parameters from point correspondences.
    ///
    /// This method estimates the `alpha` parameter using a linear least squares approach
    /// given 3D-2D point correspondences. It assumes the intrinsic parameters (fx, fy, cx, cy)
    /// are already set. The `xi` parameter is initialized to 0.0.
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

        // Update distortion with estimated alpha, set xi to 0.0
        self.distortion = DistortionModel::DoubleSphere { xi: 0.0, alpha };

        // Validate parameters
        self.validate_params()?;

        Ok(())
    }
}

/// Provides a debug string representation for [`DoubleSphereModel`].
impl fmt::Debug for DoubleSphereCamera {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (xi, alpha) = self.distortion_params();
        write!(
            f,
            "DoubleSphere [fx: {} fy: {} cx: {} cy: {} alpha: {} xi: {}]",
            self.pinhole.fx, self.pinhole.fy, self.pinhole.cx, self.pinhole.cy, alpha, xi
        )
    }
}

/// Convert DoubleSphereCamera to parameter vector.
///
/// Returns intrinsic parameters in the order: [fx, fy, cx, cy, xi, alpha]
impl From<&DoubleSphereCamera> for DVector<f64> {
    fn from(camera: &DoubleSphereCamera) -> Self {
        let (xi, alpha) = camera.distortion_params();
        DVector::from_vec(vec![
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            xi,
            alpha,
        ])
    }
}

/// Convert DoubleSphereCamera to fixed-size parameter array.
///
/// Returns intrinsic parameters as [fx, fy, cx, cy, xi, alpha]
impl From<&DoubleSphereCamera> for [f64; 6] {
    fn from(camera: &DoubleSphereCamera) -> Self {
        let (xi, alpha) = camera.distortion_params();
        [
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            xi,
            alpha,
        ]
    }
}

/// Create DoubleSphereCamera from parameter slice.
///
/// # Panics
///
/// Panics if the slice has fewer than 6 elements.
///
/// # Parameter Order
///
/// params = [fx, fy, cx, cy, xi, alpha]
impl From<&[f64]> for DoubleSphereCamera {
    fn from(params: &[f64]) -> Self {
        assert!(
            params.len() >= 6,
            "DoubleSphereCamera requires at least 6 parameters, got {}",
            params.len()
        );
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::DoubleSphere {
                xi: params[4],
                alpha: params[5],
            },
        }
    }
}

/// Create DoubleSphereCamera from fixed-size parameter array.
///
/// # Parameter Order
///
/// params = [fx, fy, cx, cy, xi, alpha]
impl From<[f64; 6]> for DoubleSphereCamera {
    fn from(params: [f64; 6]) -> Self {
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::DoubleSphere {
                xi: params[4],
                alpha: params[5],
            },
        }
    }
}

/// Creates a `DoubleSphereCamera` from a parameter slice with validation.
///
/// Unlike `From<&[f64]>`, this constructor validates all parameters
/// and returns a `Result` instead of panicking on invalid input.
///
/// # Errors
///
/// Returns `CameraModelError::InvalidParams` if fewer than 6 parameters are provided.
/// Returns validation errors if focal lengths are non-positive or xi/alpha are out of range.
pub fn try_from_params(params: &[f64]) -> Result<DoubleSphereCamera, CameraModelError> {
    if params.len() < 6 {
        return Err(CameraModelError::InvalidParams(format!(
            "DoubleSphereCamera requires at least 6 parameters, got {}",
            params.len()
        )));
    }
    let camera = DoubleSphereCamera::from(params);
    camera.validate_params()?;
    Ok(camera)
}

impl CameraModel for DoubleSphereCamera {
    const INTRINSIC_DIM: usize = 6;
    type IntrinsicJacobian = SMatrix<f64, 2, 6>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// d₁ = √(x² + y² + z²)
    /// d₂ = √(x² + y² + (ξ·d₁ + z)²)
    /// denom = α·d₂ + (1-α)·(ξ·d₁ + z)
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
    /// Returns the 2D image coordinates if valid.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError`] if:
    /// - The point fails the geometric projection condition (`ProjectionOutOfBounds`).
    /// - The denominator is too small, indicating the point is at the camera center (`PointAtCameraCenter`).
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let (xi, alpha) = self.distortion_params();
        let r2 = x * x + y * y;
        let d1 = (r2 + z * z).sqrt();

        // Check projection condition using the helper
        if !self.check_projection_condition(z, d1)? {
            return Err(CameraModelError::ProjectionOutOfBounds);
        }

        let xi_d1_z = xi * d1 + z;
        let d2 = (r2 + xi_d1_z * xi_d1_z).sqrt();
        let denom = alpha * d2 + (1.0 - alpha) * xi_d1_z;

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
    /// Algebraic solution for double sphere inverse projection.
    ///
    /// # Arguments
    ///
    /// * `point_2d` - 2D point in image coordinates.
    ///
    /// # Returns
    ///
    /// Returns the normalized 3D ray direction.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::PointOutsideImage`] if the unprojection condition fails.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;

        let (xi, alpha) = self.distortion_params();
        let mx = (u - self.pinhole.cx) / self.pinhole.fx;
        let my = (v - self.pinhole.cy) / self.pinhole.fy;
        let r2 = mx * mx + my * my;

        if !self.check_unprojection_condition(r2)? {
            return Err(CameraModelError::PointOutsideImage { x: u, y: v });
        }

        let mz_num = 1.0 - alpha * alpha * r2;
        let mz_denom = alpha * (1.0 - (2.0 * alpha - 1.0) * r2).sqrt() + (1.0 - alpha);
        let mz = mz_num / mz_denom;

        let mz2 = mz * mz;

        let num_term = mz * xi + (mz2 + (1.0 - xi * xi) * r2).sqrt();
        let denom_term = mz2 + r2;

        if denom_term < crate::GEOMETRIC_PRECISION {
            return Err(CameraModelError::PointOutsideImage { x: u, y: v });
        }

        let k = num_term / denom_term;

        let x = k * mx;
        let y = k * my;
        let z = k * mz - xi;

        // Manual normalization to reuse computed norm
        let norm = (x * x + y * y + z * z).sqrt();
        Ok(Vector3::new(x / norm, y / norm, z / norm))
    }

    /// Checks if a 3D point can be validly projected.
    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    ///
    /// Computes ∂π/∂p where π is the projection function and p = (x, y, z) is the 3D point.
    ///
    /// # Mathematical Derivation
    ///
    /// Given the Double Sphere projection model:
    /// ```text
    /// d₁ = √(x² + y² + z²)              // Distance to origin
    /// w = ξ·d₁ + z                      // Intermediate value
    /// d₂ = √(x² + y² + w²)              // Second sphere distance
    /// denom = α·d₂ + (1-α)·w            // Denominator
    /// u = fx · (x/denom) + cx           // Pixel u-coordinate
    /// v = fy · (y/denom) + cy           // Pixel v-coordinate
    /// ```
    ///
    /// ## Step 1: Derivatives of intermediate quantities
    ///
    /// ```text
    /// ∂d₁/∂x = x/d₁,  ∂d₁/∂y = y/d₁,  ∂d₁/∂z = z/d₁
    ///
    /// ∂w/∂x = ξ·(∂d₁/∂x) = ξx/d₁
    /// ∂w/∂y = ξ·(∂d₁/∂y) = ξy/d₁
    /// ∂w/∂z = ξ·(∂d₁/∂z) + 1 = ξz/d₁ + 1
    /// ```
    ///
    /// ## Step 2: Derivative of d₂
    ///
    /// Since d₂ = √(x² + y² + w²), using chain rule:
    /// ```text
    /// ∂d₂/∂x = (x + w·∂w/∂x) / d₂ = (x + w·ξx/d₁) / d₂
    /// ∂d₂/∂y = (y + w·∂w/∂y) / d₂ = (y + w·ξy/d₁) / d₂
    /// ∂d₂/∂z = (w·∂w/∂z) / d₂ = w·(ξz/d₁ + 1) / d₂
    /// ```
    ///
    /// ## Step 3: Derivative of denominator
    ///
    /// ```text
    /// ∂denom/∂x = α·∂d₂/∂x + (1-α)·∂w/∂x
    /// ∂denom/∂y = α·∂d₂/∂y + (1-α)·∂w/∂y
    /// ∂denom/∂z = α·∂d₂/∂z + (1-α)·∂w/∂z
    /// ```
    ///
    /// ## Step 4: Derivatives of pixel coordinates (quotient rule)
    ///
    /// For u = fx·(x/denom) + cx:
    /// ```text
    /// ∂u/∂x = fx · ∂(x/denom)/∂x
    ///       = fx · (denom·1 - x·∂denom/∂x) / denom²
    ///       = fx · (denom - x·∂denom/∂x) / denom²
    ///
    /// ∂u/∂y = fx · (0 - x·∂denom/∂y) / denom²
    ///       = -fx·x·∂denom/∂y / denom²
    ///
    /// ∂u/∂z = -fx·x·∂denom/∂z / denom²
    /// ```
    ///
    /// Similarly for v = fy·(y/denom) + cy:
    /// ```text
    /// ∂v/∂x = -fy·y·∂denom/∂x / denom²
    /// ∂v/∂y = fy · (denom - y·∂denom/∂y) / denom²
    /// ∂v/∂z = -fy·y·∂denom/∂z / denom²
    /// ```
    ///
    /// ## Final Jacobian Matrix (2×3)
    ///
    /// ```text
    /// J = [ ∂u/∂x  ∂u/∂y  ∂u/∂z ]
    ///     [ ∂v/∂x  ∂v/∂y  ∂v/∂z ]
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
    /// - Usenko et al., "The Double Sphere Camera Model", 3DV 2018 (Supplementary Material)
    /// - Verified against numerical differentiation in tests
    ///
    /// # Implementation Note
    ///
    /// The implementation uses the chain rule systematically through intermediate quantities
    /// d₁, w, d₂, and denom to ensure numerical stability and code clarity.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let (xi, alpha) = self.distortion_params();
        let r2 = x * x + y * y;
        let d1 = (r2 + z * z).sqrt();
        let xi_d1_z = xi * d1 + z;
        let d2 = (r2 + xi_d1_z * xi_d1_z).sqrt();
        let denom = alpha * d2 + (1.0 - alpha) * xi_d1_z;

        // Cache reciprocals to avoid repeated divisions
        let inv_d1 = 1.0 / d1;
        let inv_d2 = 1.0 / d2;

        // ∂d₁/∂x = x/d₁, ∂d₁/∂y = y/d₁, ∂d₁/∂z = z/d₁
        let dd1_dx = x * inv_d1;
        let dd1_dy = y * inv_d1;
        let dd1_dz = z * inv_d1;

        // ∂(ξ·d₁+z)/∂x = ξ·∂d₁/∂x
        let d_xi_d1_z_dx = xi * dd1_dx;
        let d_xi_d1_z_dy = xi * dd1_dy;
        let d_xi_d1_z_dz = xi * dd1_dz + 1.0;

        // ∂d₂/∂x = (x + (ξ·d₁+z)·∂(ξ·d₁+z)/∂x) / d₂
        let dd2_dx = (x + xi_d1_z * d_xi_d1_z_dx) * inv_d2;
        let dd2_dy = (y + xi_d1_z * d_xi_d1_z_dy) * inv_d2;
        let dd2_dz = (xi_d1_z * d_xi_d1_z_dz) * inv_d2;

        // ∂denom/∂x = α·∂d₂/∂x + (1-α)·∂(ξ·d₁+z)/∂x
        let ddenom_dx = alpha * dd2_dx + (1.0 - alpha) * d_xi_d1_z_dx;
        let ddenom_dy = alpha * dd2_dy + (1.0 - alpha) * d_xi_d1_z_dy;
        let ddenom_dz = alpha * dd2_dz + (1.0 - alpha) * d_xi_d1_z_dz;

        let denom2 = denom * denom;

        // ∂(x/denom)/∂x = (denom - x·∂denom/∂x) / denom²
        let du_dx = self.pinhole.fx * (denom - x * ddenom_dx) / denom2;
        let du_dy = self.pinhole.fx * (-x * ddenom_dy) / denom2;
        let du_dz = self.pinhole.fx * (-x * ddenom_dz) / denom2;

        let dv_dx = self.pinhole.fy * (-y * ddenom_dx) / denom2;
        let dv_dy = self.pinhole.fy * (denom - y * ddenom_dy) / denom2;
        let dv_dz = self.pinhole.fy * (-y * ddenom_dz) / denom2;

        SMatrix::<f64, 2, 3>::new(du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz)
    }

    /// Jacobian of projection w.r.t. camera pose (SE3).
    ///
    /// Computes the full chain: ∂π/∂ξ = (∂π/∂p_cam) · (∂p_cam/∂ξ)
    ///
    /// # Mathematical Derivation
    ///
    /// ## Chain Rule Decomposition
    ///
    /// ```text
    /// ∂π/∂ξ = ∂π/∂p_cam · ∂p_cam/∂ξ
    /// ```
    ///
    /// where:
    /// - `π` is the projection function (3D → 2D)
    /// - `p_cam` is the point in camera coordinates
    /// - `ξ ∈ se(3)` is the camera pose (Lie algebra representation)
    ///
    /// ## Part 1: Point Jacobian (∂π/∂p_cam)
    ///
    /// This is the standard point Jacobian (2×3) computed by `jacobian_point()`.
    /// See that method's documentation for details.
    ///
    /// ## Part 2: Pose Transformation Jacobian (∂p_cam/∂ξ)
    ///
    /// The camera frame point is related to the world frame point by:
    /// ```text
    /// p_cam = T⁻¹ · p_world = (R, t)⁻¹ · p_world
    /// p_cam = R^T · (p_world - t)
    /// ```
    ///
    /// The SE(3) pose is parameterized as ξ = (ω, v) where:
    /// - ω ∈ ℝ³ is the rotation (so(3) Lie algebra, axis-angle representation)
    /// - v ∈ ℝ³ is the translation
    ///
    /// ### Translation Part (∂p_cam/∂v):
    ///
    /// ```text
    /// ∂p_cam/∂v = ∂(R^T·(p_world - t))/∂v
    ///           = -R^T · ∂t/∂v
    ///           = -R^T · I
    ///           = -R^T               (3×3 matrix)
    /// ```
    ///
    /// ### Rotation Part (∂p_cam/∂ω):
    ///
    /// Using the Lie group adjoint relationship:
    /// ```text
    /// ∂p_cam/∂ω = [p_cam]×          (3×3 skew-symmetric matrix)
    /// ```
    ///
    /// where `[p_cam]×` is the skew-symmetric cross-product matrix:
    /// ```text
    /// [p_cam]× = [  0    -pz    py ]
    ///            [  pz     0   -px ]
    ///            [ -py    px     0 ]
    /// ```
    ///
    /// This comes from the derivative of the rotation action on a point.
    ///
    /// ### Combined Jacobian (3×6):
    ///
    /// ```text
    /// ∂p_cam/∂ξ = [ -R^T | [p_cam]× ]     (3×6)
    ///              ︸───︸   ︸──────︸
    ///               ∂/∂v     ∂/∂ω
    /// ```
    ///
    /// ## Final Result (2×6)
    ///
    /// ```text
    /// ∂π/∂ξ = (∂π/∂p_cam) · (∂p_cam/∂ξ)
    ///       = (2×3) · (3×6)
    ///       = (2×6)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `p_world` - 3D point in world coordinate frame.
    /// * `pose` - The camera pose in SE(3).
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// 1. Point Jacobian (∂π/∂p_cam): 2×3 matrix.
    /// 2. Pose Transformation Jacobian (∂p_cam/∂ξ): 3×6 matrix.
    ///
    /// The caller can multiply these to get the full pose Jacobian.
    ///
    /// # References
    ///
    /// - Barfoot, "State Estimation for Robotics", Chapter 7 (Lie Groups)
    /// - Sola et al., "A micro Lie theory for state estimation in robotics", 2021
    ///
    /// # Implementation Note
    ///
    /// The rotation Jacobian uses the skew-symmetric matrix `[p_cam]×` which
    /// is provided by the `skew_symmetric()` helper function.
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

    /// Jacobian of projection w.r.t. intrinsic parameters (2×6).
    ///
    /// Computes ∂π/∂K where K = [fx, fy, cx, cy, ξ, α] are the intrinsic parameters.
    ///
    /// # Mathematical Derivation
    ///
    /// The intrinsic parameters consist of:
    /// 1. **Linear parameters**: fx, fy, cx, cy (pinhole projection)
    /// 2. **Distortion parameters**: ξ (xi), α (alpha) (Double Sphere specific)
    ///
    /// ## Projection Model Recap
    ///
    /// ```text
    /// d₁ = √(x² + y² + z²)
    /// w = ξ·d₁ + z
    /// d₂ = √(x² + y² + w²)
    /// denom = α·d₂ + (1-α)·w
    /// u = fx · (x/denom) + cx
    /// v = fy · (y/denom) + cy
    /// ```
    ///
    /// ## Part 1: Linear Parameters (fx, fy, cx, cy)
    ///
    /// These have direct, simple derivatives:
    ///
    /// ### Focal lengths (fx, fy):
    /// ```text
    /// ∂u/∂fx = x/denom    (coefficient of fx in u)
    /// ∂u/∂fy = 0          (fy doesn't affect u)
    /// ∂v/∂fx = 0          (fx doesn't affect v)
    /// ∂v/∂fy = y/denom    (coefficient of fy in v)
    /// ```
    ///
    /// ### Principal point (cx, cy):
    /// ```text
    /// ∂u/∂cx = 1          (additive constant)
    /// ∂u/∂cy = 0          (cy doesn't affect u)
    /// ∂v/∂cx = 0          (cx doesn't affect v)
    /// ∂v/∂cy = 1          (additive constant)
    /// ```
    ///
    /// ## Part 2: Distortion Parameters (ξ, α)
    ///
    /// These affect the projection through the denominator term.
    ///
    /// ### Derivative w.r.t. ξ (xi):
    ///
    /// Since w = ξ·d₁ + z and d₂ = √(x² + y² + w²), we have:
    /// ```text
    /// ∂w/∂ξ = d₁
    ///
    /// ∂d₂/∂ξ = ∂d₂/∂w · ∂w/∂ξ
    ///        = (w/d₂) · d₁
    ///        = w·d₁/d₂
    ///
    /// ∂denom/∂ξ = α·∂d₂/∂ξ + (1-α)·∂w/∂ξ
    ///           = α·(w·d₁/d₂) + (1-α)·d₁
    ///           = d₁·[α·w/d₂ + (1-α)]
    /// ```
    ///
    /// Using the quotient rule on u = fx·(x/denom) + cx:
    /// ```text
    /// ∂u/∂ξ = fx · ∂(x/denom)/∂ξ
    ///       = fx · (-x/denom²) · ∂denom/∂ξ
    ///       = -fx·x·∂denom/∂ξ / denom²
    /// ```
    ///
    /// Similarly:
    /// ```text
    /// ∂v/∂ξ = -fy·y·∂denom/∂ξ / denom²
    /// ```
    ///
    /// ### Derivative w.r.t. α (alpha):
    ///
    /// Since denom = α·d₂ + (1-α)·w:
    /// ```text
    /// ∂denom/∂α = d₂ - w
    ///
    /// ∂u/∂α = -fx·x·(d₂ - w) / denom²
    /// ∂v/∂α = -fy·y·(d₂ - w) / denom²
    /// ```
    ///
    /// ## Final Jacobian Matrix (2×6)
    ///
    /// ```text
    /// J = [ ∂u/∂fx  ∂u/∂y  ∂u/∂cx  ∂u/∂cy  ∂u/∂ξ  ∂u/∂α ]
    ///     [ ∂v/∂x  ∂v/∂y  ∂v/∂cx  ∂v/∂cy  ∂v/∂ξ  ∂v/∂α ]
    ///
    ///   = [ x/denom    0       1       0      -fx·x·∂denom/∂ξ/denom²  -fx·x·(d₂-w)/denom² ]
    ///     [   0     y/denom    0       1      -fy·y·∂denom/∂ξ/denom²  -fy·y·(d₂-w)/denom² ]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `p_cam` - 3D point in camera coordinate frame.
    ///
    /// # Returns
    ///
    /// Returns the 2x6 Intrinsic Jacobian matrix.
    ///
    /// # References
    ///
    /// - Usenko et al., "The Double Sphere Camera Model", 3DV 2018
    /// - Verified against numerical differentiation in tests
    ///
    /// # Implementation Note
    ///
    /// The implementation computes all intermediate values (d₁, w, d₂, denom)
    /// first, then applies the chain rule derivatives systematically.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let (xi, alpha) = self.distortion_params();
        let r2 = x * x + y * y;
        let d1 = (r2 + z * z).sqrt();
        let xi_d1_z = xi * d1 + z;
        let d2 = (r2 + xi_d1_z * xi_d1_z).sqrt();
        let denom = alpha * d2 + (1.0 - alpha) * xi_d1_z;

        // Cache reciprocals to avoid repeated divisions
        let inv_denom = 1.0 / denom;
        let inv_d2 = 1.0 / d2;

        let x_norm = x * inv_denom;
        let y_norm = y * inv_denom;

        // ∂u/∂fx = x/denom, ∂u/∂fy = 0, ∂u/∂cx = 1, ∂u/∂cy = 0
        // ∂v/∂fx = 0, ∂v/∂fy = y/denom, ∂v/∂cx = 0, ∂v/∂cy = 1

        // For ξ and α derivatives
        let d_xi_d1_z_dxi = d1;
        let dd2_dxi = (xi_d1_z * d_xi_d1_z_dxi) * inv_d2;
        let ddenom_dxi = alpha * dd2_dxi + (1.0 - alpha) * d_xi_d1_z_dxi;

        let ddenom_dalpha = d2 - xi_d1_z;

        let inv_denom2 = inv_denom * inv_denom;

        let du_dxi = -self.pinhole.fx * x * ddenom_dxi * inv_denom2;
        let dv_dxi = -self.pinhole.fy * y * ddenom_dxi * inv_denom2;

        let du_dalpha = -self.pinhole.fx * x * ddenom_dalpha * inv_denom2;
        let dv_dalpha = -self.pinhole.fy * y * ddenom_dalpha * inv_denom2;

        SMatrix::<f64, 2, 6>::new(
            x_norm, 0.0, 1.0, 0.0, du_dxi, du_dalpha, 0.0, y_norm, 0.0, 1.0, dv_dxi, dv_dalpha,
        )
    }

    /// Validates camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - fx, fy must be positive (> 0)
    /// - fx, fy must be finite
    /// - cx, cy must be finite
    /// - ξ must be finite
    /// - α must be in (0, 1]
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError`] if any parameter violates the validation rules.
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

        let (xi, alpha) = self.distortion_params();
        if !xi.is_finite() || !(-1.0..=1.0).contains(&xi) {
            return Err(CameraModelError::ParameterOutOfRange {
                param: "xi".to_string(),
                value: xi,
                min: -1.0,
                max: 1.0,
            });
        }

        if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
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
    /// The [`DistortionModel`] associated with this camera (typically [`DistortionModel::DoubleSphere`]).
    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// Returns the string identifier for the camera model.
    ///
    /// # Returns
    ///
    /// The string `"double_sphere"`.
    fn get_model_name(&self) -> &'static str {
        "double_sphere"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2xX, Matrix3xX};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_double_sphere_camera_creation() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        assert_eq!(camera.pinhole.fx, 300.0);
        let (xi, alpha) = camera.distortion_params();
        assert_eq!(alpha, 0.6);
        assert_eq!(xi, -0.2);

        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam)?;

        assert!((uv.x - 320.0).abs() < crate::PROJECTION_TEST_TOLERANCE);
        assert!((uv.y - 240.0).abs() < crate::PROJECTION_TEST_TOLERANCE);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
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
                    "Mismatch at ({}, {})",
                    r,
                    i
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_jacobian_intrinsics_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::DoubleSphere {
            xi: -0.2,
            alpha: 0.6,
        };
        let camera = DoubleSphereCamera::new(pinhole, distortion)?;
        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params: DVector<f64> = (&camera).into();
        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

        for i in 0..6 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = DoubleSphereCamera::from(params_plus.as_slice());
            let cam_minus = DoubleSphereCamera::from(params_minus.as_slice());

            let uv_plus = cam_plus.project(&p_cam)?;
            let uv_minus = cam_minus.project(&p_cam)?;
            let num_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                let diff = (jac_analytical[(r, i)] - num_jac[r]).abs();
                assert!(
                    diff < crate::JACOBIAN_TEST_TOLERANCE,
                    "Mismatch at ({}, {})",
                    r,
                    i
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_linear_estimation() -> TestResult {
        // Ground truth DoubleSphere camera with xi=0.0 (linear_estimation fixes xi=0.0)
        let gt_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let gt_distortion = DistortionModel::DoubleSphere {
            xi: 0.0,
            alpha: 0.6,
        };
        let gt_camera = DoubleSphereCamera::new(gt_pinhole, gt_distortion)?;

        // Generate synthetic 3D points in camera frame
        let n_points = 50;
        let mut pts_3d = Matrix3xX::zeros(n_points);
        let mut pts_2d = Matrix2xX::zeros(n_points);
        let mut valid = 0;

        for i in 0..n_points {
            let angle = i as f64 * 2.0 * std::f64::consts::PI / n_points as f64;
            let r = 0.1 + 0.3 * (i as f64 / n_points as f64);
            let p3d = Vector3::new(r * angle.cos(), r * angle.sin(), 1.0);

            if let Ok(p2d) = gt_camera.project(&p3d) {
                pts_3d.set_column(valid, &p3d);
                pts_2d.set_column(valid, &p2d);
                valid += 1;
            }
        }
        let pts_3d = pts_3d.columns(0, valid).into_owned();
        let pts_2d = pts_2d.columns(0, valid).into_owned();

        // Initial camera with small alpha (alpha must be > 0 for validation)
        let init_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let init_distortion = DistortionModel::DoubleSphere {
            xi: 0.0,
            alpha: 0.1,
        };
        let mut camera = DoubleSphereCamera::new(init_pinhole, init_distortion)?;

        camera.linear_estimation(&pts_3d, &pts_2d)?;

        // Verify reprojection error
        for i in 0..valid {
            let p3d = pts_3d.column(i).into_owned();
            let projected = camera.project(&Vector3::new(p3d.x, p3d.y, p3d.z))?;
            let err = ((projected.x - pts_2d[(0, i)]).powi(2)
                + (projected.y - pts_2d[(1, i)]).powi(2))
            .sqrt();
            assert!(err < 1.0, "Reprojection error too large: {err}");
        }

        Ok(())
    }
}
