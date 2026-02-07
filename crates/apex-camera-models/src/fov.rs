//! Field-of-View (FOV) Camera Model
//!
//! A fisheye camera model using a field-of-view parameter for radial distortion.
//!
//! # Mathematical Model
//!
//! ## Projection (3D → 2D)
//!
//! For a 3D point p = (x, y, z) in camera coordinates:
//!
//! ```text
//! r = √(x² + y²)
//! atan_wrd = atan2(2·tan(w/2)·r, z)
//! rd = atan_wrd / (r·w)    (if r > 0)
//! rd = 2·tan(w/2) / w       (if r ≈ 0)
//!
//! mx = x · rd
//! my = y · rd
//! u = fx · mx + cx
//! v = fy · my + cy
//! ```
//!
//! where w is the field-of-view parameter (0 < w ≤ π).
//!
//! ## Unprojection (2D → 3D)
//!
//! Uses trigonometric inverse with special handling near optical axis.
//!
//! # Parameters
//!
//! - **Intrinsics**: fx, fy, cx, cy
//! - **Distortion**: w (field-of-view parameter) (5 parameters total)
//!
//! # Use Cases
//!
//! - Fisheye cameras in SLAM applications
//! - Wide field-of-view lenses
//!
//! # References
//!
//! - Zhang et al., "Simultaneous Localization and Mapping with Fisheye Cameras"
//!   https://arxiv.org/pdf/1807.08957

use crate::{CameraModel, CameraModelError, DistortionModel, PinholeParams};
use nalgebra::{DVector, SMatrix, Vector2, Vector3};

/// FOV camera model with 5 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FovCamera {
    pub pinhole: PinholeParams,
    pub distortion: DistortionModel,
}

impl FovCamera {
    /// Create a new Field-of-View (FOV) camera.
    ///
    /// # Arguments
    ///
    /// * `pinhole` - Pinhole parameters (fx, fy, cx, cy).
    /// * `distortion` - MUST be [`DistortionModel::FOV`] with parameter `w`.
    ///
    /// # Returns
    ///
    /// Returns a new `FovCamera` instance if the distortion model matches.
    ///
    /// # Errors
    ///
    /// Returns [`CameraModelError::InvalidParams`] if `distortion` is not [`DistortionModel::FOV`].
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
    /// Returns the `w` parameter for FOV.
    /// If the distortion model is incorrect (which shouldn't happen for valid instances), returns `0.0`.
    fn distortion_params(&self) -> f64 {
        match self.distortion {
            DistortionModel::FOV { w } => w,
            _ => 0.0,
        }
    }

    /// Performs linear estimation to initialize the w parameter from point correspondences.
    ///
    /// This method estimates the `w` parameter using a linear least squares approach
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

        let num_points = points_2d.ncols();

        // Need at least 2 points for linear estimation
        if num_points < 2 {
            return Err(CameraModelError::InvalidParams(
                "Need at least 2 point correspondences for linear estimation".to_string(),
            ));
        }

        // Set up the linear system to solve for w
        // We'll use a simplified approach: estimate w that minimizes reprojection error
        // Start with a reasonable initial value
        let mut best_w = 1.0;
        let mut best_error = f64::INFINITY;

        // Grid search over reasonable w values
        for w_test in (10..300).map(|i| i as f64 / 100.0) {
            let mut error_sum = 0.0;
            let mut valid_count = 0;

            for i in 0..num_points {
                let x = points_3d[(0, i)];
                let y = points_3d[(1, i)];
                let z = points_3d[(2, i)];
                let u_observed = points_2d[(0, i)];
                let v_observed = points_2d[(1, i)];

                // Try projection with this w value
                let r2 = x * x + y * y;
                let r = r2.sqrt();

                let tan_w_half = (w_test / 2.0).tan();
                let atan_wrd = (2.0 * tan_w_half * r).atan2(z);

                let eps_sqrt = f64::EPSILON.sqrt();
                let rd = if r2 < eps_sqrt {
                    2.0 * tan_w_half / w_test
                } else {
                    atan_wrd / (r * w_test)
                };

                let mx = x * rd;
                let my = y * rd;

                let u_predicted = self.pinhole.fx * mx + self.pinhole.cx;
                let v_predicted = self.pinhole.fy * my + self.pinhole.cy;

                let error = ((u_predicted - u_observed).powi(2)
                    + (v_predicted - v_observed).powi(2))
                .sqrt();

                if error.is_finite() {
                    error_sum += error;
                    valid_count += 1;
                }
            }

            if valid_count > 0 {
                let avg_error = error_sum / valid_count as f64;
                if avg_error < best_error {
                    best_error = avg_error;
                    best_w = w_test;
                }
            }
        }

        self.distortion = DistortionModel::FOV { w: best_w };

        // Validate parameters
        self.validate_params()?;

        Ok(())
    }
}

/// Convert camera to dynamic vector of intrinsic parameters.
///
/// # Layout
///
/// The parameters are ordered as: [fx, fy, cx, cy, w]
impl From<&FovCamera> for DVector<f64> {
    fn from(camera: &FovCamera) -> Self {
        let w = camera.distortion_params();
        DVector::from_vec(vec![
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            w,
        ])
    }
}

/// Convert camera to fixed-size array of intrinsic parameters.
///
/// # Layout
///
/// The parameters are ordered as: [fx, fy, cx, cy, w]
impl From<&FovCamera> for [f64; 5] {
    fn from(camera: &FovCamera) -> Self {
        let w = camera.distortion_params();
        [
            camera.pinhole.fx,
            camera.pinhole.fy,
            camera.pinhole.cx,
            camera.pinhole.cy,
            w,
        ]
    }
}

/// Create camera from slice of intrinsic parameters.
///
/// # Layout
///
/// Expected parameter order: [fx, fy, cx, cy, w]
///
/// # Panics
///
/// Panics if the slice has fewer than 5 elements.
impl From<&[f64]> for FovCamera {
    fn from(params: &[f64]) -> Self {
        assert!(
            params.len() >= 5,
            "FovCamera requires at least 5 parameters, got {}",
            params.len()
        );
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::FOV { w: params[4] },
        }
    }
}

/// Create camera from fixed-size array of intrinsic parameters.
///
/// # Layout
///
/// Expected parameter order: [fx, fy, cx, cy, w]
impl From<[f64; 5]> for FovCamera {
    fn from(params: [f64; 5]) -> Self {
        Self {
            pinhole: PinholeParams {
                fx: params[0],
                fy: params[1],
                cx: params[2],
                cy: params[3],
            },
            distortion: DistortionModel::FOV { w: params[4] },
        }
    }
}

/// Creates a `FovCamera` from a parameter slice with validation.
///
/// Unlike `From<&[f64]>`, this constructor validates all parameters
/// and returns a `Result` instead of panicking on invalid input.
///
/// # Errors
///
/// Returns `CameraModelError::InvalidParams` if fewer than 5 parameters are provided.
/// Returns validation errors if focal lengths are non-positive or w is out of range.
pub fn try_from_params(params: &[f64]) -> Result<FovCamera, CameraModelError> {
    if params.len() < 5 {
        return Err(CameraModelError::InvalidParams(format!(
            "FovCamera requires at least 5 parameters, got {}",
            params.len()
        )));
    }
    let camera = FovCamera::from(params);
    camera.validate_params()?;
    Ok(camera)
}

impl CameraModel for FovCamera {
    const INTRINSIC_DIM: usize = 5;
    type IntrinsicJacobian = SMatrix<f64, 2, 5>;
    type PointJacobian = SMatrix<f64, 2, 3>;

    /// Projects a 3D point to 2D image coordinates.
    ///
    /// # Mathematical Formula
    ///
    /// Uses atan-based radial distortion with FOV parameter w.
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
    /// Returns [`CameraModelError::ProjectionOutOfBounds`] if `z` is too small.
    fn project(&self, p_cam: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        // Check if z is valid (too close to camera center)
        if z < f64::EPSILON.sqrt() {
            return Err(CameraModelError::ProjectionOutOfBounds);
        }

        let r = (x * x + y * y).sqrt();
        let w = self.distortion_params();
        let tan_w_2 = (w / 2.0).tan();
        let mul2tanwby2 = tan_w_2 * 2.0;

        let rd = if r > crate::GEOMETRIC_PRECISION {
            let atan_wrd = (mul2tanwby2 * r / z).atan();
            atan_wrd / (r * w)
        } else {
            mul2tanwby2 / w
        };

        let mx = x * rd;
        let my = y * rd;

        Ok(Vector2::new(
            self.pinhole.fx * mx + self.pinhole.cx,
            self.pinhole.fy * my + self.pinhole.cy,
        ))
    }

    /// Unprojects a 2D image point to a 3D ray.
    ///
    /// # Algorithm
    ///
    /// Trigonometric inverse using sin/cos relationships.
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
    /// This model does not explicitly fail unprojection unless internal math errors occur, in which case it propagates them.
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let u = point_2d.x;
        let v = point_2d.y;

        let w = self.distortion_params();
        let tan_w_2 = (w / 2.0).tan();
        let mul2tanwby2 = tan_w_2 * 2.0;

        let mx = (u - self.pinhole.cx) / self.pinhole.fx;
        let my = (v - self.pinhole.cy) / self.pinhole.fy;

        let r2 = mx * mx + my * my;
        let rd = r2.sqrt();

        if rd < crate::GEOMETRIC_PRECISION {
            return Ok(Vector3::new(0.0, 0.0, 1.0));
        }

        let ru = (rd * w).tan() / mul2tanwby2;

        let norm_factor = (1.0 + ru * ru).sqrt();
        let x = mx * ru / (rd * norm_factor);
        let y = my * ru / (rd * norm_factor);
        let z = 1.0 / norm_factor;

        Ok(Vector3::new(x, y, z))
    }

    /// Jacobian of projection w.r.t. 3D point coordinates (2×3).
    ///
    /// # Mathematical Derivation
    ///
    /// For the FOV camera model, projection is defined as:
    ///
    /// ```text
    /// r = √(x² + y²)
    /// α = 2·tan(w/2)·r / z
    /// atan_wrd = atan(α)
    /// rd = atan_wrd / (r·w)    (if r > 0)
    /// rd = 2·tan(w/2) / w       (if r ≈ 0)
    ///
    /// mx = x · rd
    /// my = y · rd
    /// u = fx · mx + cx
    /// v = fy · my + cy
    /// ```
    ///
    /// ## Jacobian Structure
    ///
    /// We need to compute ∂u/∂p and ∂v/∂p where p = (x, y, z):
    ///
    /// ```text
    /// J_point = [ ∂u/∂x  ∂u/∂y  ∂u/∂z ]
    ///           [ ∂v/∂x  ∂v/∂y  ∂v/∂z ]
    /// ```
    ///
    /// ## Chain Rule Application
    ///
    /// Starting from the projection equations:
    /// - u = fx · mx + cx = fx · x · rd + cx
    /// - v = fy · my + cy = fy · y · rd + cy
    ///
    /// We apply the chain rule:
    ///
    /// ```text
    /// ∂u/∂x = fx · ∂(x·rd)/∂x = fx · (rd + x · ∂rd/∂x)
    /// ∂u/∂y = fx · ∂(x·rd)/∂y = fx · x · ∂rd/∂y
    /// ∂u/∂z = fx · ∂(x·rd)/∂z = fx · x · ∂rd/∂z
    ///
    /// ∂v/∂x = fy · ∂(y·rd)/∂x = fy · y · ∂rd/∂x
    /// ∂v/∂y = fy · ∂(y·rd)/∂y = fy · (rd + y · ∂rd/∂y)
    /// ∂v/∂z = fy · ∂(y·rd)/∂z = fy · y · ∂rd/∂z
    /// ```
    ///
    /// ## Computing ∂rd/∂x, ∂rd/∂y, ∂rd/∂z
    ///
    /// For r > 0 case (non-optical axis):
    ///
    /// rd = atan(α) / (r·w) where α = 2·tan(w/2)·r / z
    ///
    /// Using chain rule through r and α:
    ///
    /// ```text
    /// ∂rd/∂r = [∂atan/∂α · ∂α/∂r · r·w - atan(α) · w] / (r·w)²
    ///        = [1/(1+α²) · 2·tan(w/2)/z · r·w - atan(α) · w] / (r·w)²
    ///
    /// ∂rd/∂z = ∂atan/∂α · ∂α/∂z / (r·w)
    ///        = 1/(1+α²) · (-2·tan(w/2)·r/z²) / (r·w)
    /// ```
    ///
    /// Then using ∂r/∂x = x/r and ∂r/∂y = y/r:
    ///
    /// ```text
    /// ∂rd/∂x = ∂rd/∂r · ∂r/∂x = ∂rd/∂r · x/r
    /// ∂rd/∂y = ∂rd/∂r · ∂r/∂y = ∂rd/∂r · y/r
    /// ∂rd/∂z = (computed directly above)
    /// ```
    ///
    /// ## Special Case: Near Optical Axis (r ≈ 0)
    ///
    /// When r < ε, we use rd = 2·tan(w/2) / w (constant), so:
    ///
    /// ```text
    /// ∂rd/∂x = 0, ∂rd/∂y = 0, ∂rd/∂z = 0
    /// ```
    ///
    /// Leading to simplified Jacobian:
    ///
    /// ```text
    /// J_point = [ fx·rd    0       0   ]
    ///           [   0    fy·rd     0   ]
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
    /// - Devernay & Faugeras, "Straight lines have to be straight", Machine Vision and Applications 2001
    /// - Zhang et al., "Fisheye Camera Calibration Using Principal Point Constraints", PAMI 2012
    ///
    /// # Numerical Verification
    ///
    /// This analytical Jacobian is verified against numerical differentiation in
    /// `test_jacobian_point_numerical()` with tolerance < 1e-6.
    fn jacobian_point(&self, p_cam: &Vector3<f64>) -> Self::PointJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r = (x * x + y * y).sqrt();
        let w = self.distortion_params();
        let tan_w_2 = (w / 2.0).tan();
        let mul2tanwby2 = tan_w_2 * 2.0;

        if r < crate::GEOMETRIC_PRECISION {
            let rd = mul2tanwby2 / w;
            return SMatrix::<f64, 2, 3>::new(
                self.pinhole.fx * rd,
                0.0,
                0.0,
                0.0,
                self.pinhole.fy * rd,
                0.0,
            );
        }

        let atan_wrd = (mul2tanwby2 * r / z).atan();
        let rd = atan_wrd / (r * w);

        // Derivatives
        let datan_dr = mul2tanwby2 * z / (z * z + mul2tanwby2 * mul2tanwby2 * r * r);
        let datan_dz = -mul2tanwby2 * r / (z * z + mul2tanwby2 * mul2tanwby2 * r * r);

        let drd_dr = (datan_dr * r - atan_wrd) / (r * r * w);
        let drd_dz = datan_dz / (r * w);

        let dr_dx = x / r;
        let dr_dy = y / r;

        let dmx_dx = rd + x * drd_dr * dr_dx;
        let dmx_dy = x * drd_dr * dr_dy;
        let dmx_dz = x * drd_dz;

        let dmy_dx = y * drd_dr * dr_dx;
        let dmy_dy = rd + y * drd_dr * dr_dy;
        let dmy_dz = y * drd_dz;

        SMatrix::<f64, 2, 3>::new(
            self.pinhole.fx * dmx_dx,
            self.pinhole.fx * dmx_dy,
            self.pinhole.fx * dmx_dz,
            self.pinhole.fy * dmy_dx,
            self.pinhole.fy * dmy_dy,
            self.pinhole.fy * dmy_dz,
        )
    }

    /// Jacobian of projection w.r.t. intrinsic parameters (2×5).
    ///
    /// # Mathematical Derivation
    ///
    /// The FOV camera has 5 intrinsic parameters: [fx, fy, cx, cy, w]
    ///
    /// ## Projection Equations
    ///
    /// ```text
    /// u = fx · mx + cx
    /// v = fy · my + cy
    /// ```
    ///
    /// where mx = x · rd and my = y · rd, with:
    ///
    /// ```text
    /// rd = atan(2·tan(w/2)·r/z) / (r·w)  (for r > 0)
    /// rd = 2·tan(w/2) / w                 (for r ≈ 0)
    /// ```
    ///
    /// ## Jacobian Structure
    ///
    /// We need to compute:
    ///
    /// ```text
    /// J_intrinsics = [ ∂u/∂fx  ∂u/∂fy  ∂u/∂cx  ∂u/∂cy  ∂u/∂w ]
    ///                [ ∂v/∂fx  ∂v/∂fy  ∂v/∂cx  ∂v/∂cy  ∂v/∂w ]
    /// ```
    ///
    /// ## Linear Parameters (fx, fy, cx, cy)
    ///
    /// These appear linearly in the projection equations:
    ///
    /// ```text
    /// ∂u/∂fx = mx,     ∂u/∂fy = 0,      ∂u/∂cx = 1,      ∂u/∂cy = 0
    /// ∂v/∂fx = 0,      ∂v/∂fy = my,     ∂v/∂cx = 0,      ∂v/∂cy = 1
    /// ```
    ///
    /// ## Distortion Parameter (w)
    ///
    /// The parameter w affects the distortion factor rd. We need ∂rd/∂w.
    ///
    /// ### Case 1: r > 0 (Non-Optical Axis)
    ///
    /// Starting from:
    ///
    /// ```text
    /// α = 2·tan(w/2)·r / z
    /// rd = atan(α) / (r·w)
    /// ```
    ///
    /// Taking derivatives:
    ///
    /// ```text
    /// ∂α/∂w = 2·sec²(w/2)·(1/2)·r/z = sec²(w/2)·r/z
    /// ```
    ///
    /// where sec²(w/2) = 1 + tan²(w/2).
    ///
    /// Using the quotient rule for rd = atan(α) / (r·w):
    ///
    /// ```text
    /// ∂rd/∂w = [∂atan(α)/∂w · r·w - atan(α) · r] / (r·w)²
    ///        = [1/(1+α²) · ∂α/∂w · r·w - atan(α) · r] / (r·w)²
    ///        = [sec²(w/2)·r²·w/z·(1/(1+α²)) - atan(α)·r] / (r²·w²)
    /// ```
    ///
    /// Simplifying:
    ///
    /// ```text
    /// ∂rd/∂w = [∂atan(α)/∂α · ∂α/∂w · r·w - atan(α)·r] / (r·w)²
    /// ```
    ///
    /// ### Case 2: r ≈ 0 (Near Optical Axis)
    ///
    /// When r ≈ 0, we use rd = 2·tan(w/2) / w.
    ///
    /// Using the quotient rule:
    ///
    /// ```text
    /// ∂rd/∂w = [2·sec²(w/2)·(1/2)·w - 2·tan(w/2)] / w²
    ///        = [sec²(w/2)·w - 2·tan(w/2)] / w²
    /// ```
    ///
    /// ## Final Jacobian w.r.t. w
    ///
    /// Once we have ∂rd/∂w, we compute:
    ///
    /// ```text
    /// ∂u/∂w = fx · ∂(x·rd)/∂w = fx · x · ∂rd/∂w
    /// ∂v/∂w = fy · ∂(y·rd)/∂w = fy · y · ∂rd/∂w
    /// ```
    ///
    /// ## Matrix Form
    ///
    /// The complete Jacobian matrix is:
    ///
    /// ```text
    /// J = [ mx   0    1    0    fx·x·∂rd/∂w ]
    ///     [  0  my    0    1    fy·y·∂rd/∂w ]
    /// ```
    ///
    /// where mx = x·rd and my = y·rd.
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
    /// - Devernay & Faugeras, "Straight lines have to be straight", Machine Vision and Applications 2001
    /// - Hughes et al., "Rolling Shutter Motion Deblurring", CVPR 2010 (uses FOV model)
    ///
    /// # Numerical Verification
    ///
    /// This analytical Jacobian is verified against numerical differentiation in
    /// `test_jacobian_intrinsics_numerical()` with tolerance < 1e-4.
    ///
    /// # Notes
    ///
    /// The FOV parameter w controls the field of view angle. Typical values range from
    /// 0.5 (narrow FOV) to π (hemispheric fisheye). The derivative ∂rd/∂w captures how
    /// changes in the FOV parameter affect the radial distortion mapping.
    fn jacobian_intrinsics(&self, p_cam: &Vector3<f64>) -> Self::IntrinsicJacobian {
        let x = p_cam[0];
        let y = p_cam[1];
        let z = p_cam[2];

        let r = (x * x + y * y).sqrt();
        let w = self.distortion_params();
        let tan_w_2 = (w / 2.0).tan();
        let mul2tanwby2 = tan_w_2 * 2.0;

        let rd = if r > crate::GEOMETRIC_PRECISION {
            let atan_wrd = (mul2tanwby2 * r / z).atan();
            atan_wrd / (r * w)
        } else {
            mul2tanwby2 / w
        };

        let mx = x * rd;
        let my = y * rd;

        // ∂u/∂fx = mx, ∂u/∂fy = 0, ∂u/∂cx = 1, ∂u/∂cy = 0
        // ∂v/∂fx = 0, ∂v/∂fy = my, ∂v/∂cx = 0, ∂v/∂cy = 1

        // For w derivative: ∂rd/∂w
        let drd_dw = if r > crate::GEOMETRIC_PRECISION {
            let tan_w_2 = (w / 2.0).tan();
            let alpha = 2.0 * tan_w_2 * r / z;
            let atan_alpha = alpha.atan();

            // sec²(w/2) = 1 + tan²(w/2)
            let sec2_w_2 = 1.0 + tan_w_2 * tan_w_2;
            let dalpha_dw = sec2_w_2 * r / z;

            // ∂rd/∂w = [1/(1+α²) · ∂α/∂w · r·w - atan(α) · r] / (r·w)²
            let datan_dw = dalpha_dw / (1.0 + alpha * alpha);
            (datan_dw * r * w - atan_alpha * r) / (r * r * w * w)
        } else {
            let tan_w_2 = (w / 2.0).tan();
            let sec2_w_2 = 1.0 + tan_w_2 * tan_w_2;
            // rd = 2·tan(w/2) / w
            // ∂rd/∂w = [2·sec²(w/2)/2 · w - 2·tan(w/2)] / w²
            //        = [sec²(w/2) · w - 2·tan(w/2)] / w²
            (sec2_w_2 * w - 2.0 * tan_w_2) / (w * w)
        };

        let du_dw = self.pinhole.fx * x * drd_dw;
        let dv_dw = self.pinhole.fy * y * drd_dw;

        SMatrix::<f64, 2, 5>::new(mx, 0.0, 1.0, 0.0, du_dw, 0.0, my, 0.0, 1.0, dv_dw)
    }

    /// Validates camera parameters.
    ///
    /// # Validation Rules
    ///
    /// - fx, fy must be positive (> 0)
    /// - fx, fy must be finite
    /// - cx, cy must be finite
    /// - w must be in (0, π]
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

        let w = self.distortion_params();
        if !w.is_finite() || w <= 0.0 || w > std::f64::consts::PI {
            return Err(CameraModelError::ParameterOutOfRange {
                param: "w".to_string(),
                value: w,
                min: 0.0,
                max: std::f64::consts::PI,
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
    /// The [`DistortionModel`] associated with this camera (typically [`DistortionModel::FOV`]).
    fn get_distortion(&self) -> DistortionModel {
        self.distortion
    }

    /// Returns the string identifier for the camera model.
    ///
    /// # Returns
    ///
    /// The string `"fov"`.
    fn get_model_name(&self) -> &'static str {
        "fov"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2xX, Matrix3xX};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_fov_camera_creation() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;

        assert_eq!(camera.pinhole.fx, 300.0);
        assert_eq!(camera.distortion_params(), 1.5);
        Ok(())
    }

    #[test]
    fn test_projection_at_optical_axis() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;

        let p_cam = Vector3::new(0.0, 0.0, 1.0);
        let uv = camera.project(&p_cam)?;

        assert!((uv.x - 320.0).abs() < 1e-4);
        assert!((uv.y - 240.0).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn test_jacobian_point_numerical() -> TestResult {
        let pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;

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
        let distortion = DistortionModel::FOV { w: 1.5 };
        let camera = FovCamera::new(pinhole, distortion)?;

        let p_cam = Vector3::new(0.1, 0.2, 1.0);

        let jac_analytical = camera.jacobian_intrinsics(&p_cam);
        let params: DVector<f64> = (&camera).into();
        let eps = crate::NUMERICAL_DERIVATIVE_EPS;

        for i in 0..5 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let cam_plus = FovCamera::from(params_plus.as_slice());
            let cam_minus = FovCamera::from(params_minus.as_slice());

            let uv_plus = cam_plus.project(&p_cam)?;
            let uv_minus = cam_minus.project(&p_cam)?;
            let num_jac = (uv_plus - uv_minus) / (2.0 * eps);

            for r in 0..2 {
                let diff = (jac_analytical[(r, i)] - num_jac[r]).abs();
                assert!(diff < 1e-4, "Mismatch at ({}, {})", r, i);
            }
        }
        Ok(())
    }

    #[test]
    fn test_fov_from_into_traits() -> TestResult {
        let pinhole = PinholeParams::new(400.0, 410.0, 320.0, 240.0)?;
        let distortion = DistortionModel::FOV { w: 1.8 };
        let camera = FovCamera::new(pinhole, distortion)?;

        // Test conversion to DVector
        let params: DVector<f64> = (&camera).into();
        assert_eq!(params.len(), 5);
        assert_eq!(params[0], 400.0);
        assert_eq!(params[1], 410.0);
        assert_eq!(params[2], 320.0);
        assert_eq!(params[3], 240.0);
        assert_eq!(params[4], 1.8);

        // Test conversion to array
        let arr: [f64; 5] = (&camera).into();
        assert_eq!(arr, [400.0, 410.0, 320.0, 240.0, 1.8]);

        // Test conversion from slice
        let params_slice = [450.0, 460.0, 330.0, 250.0, 2.0];
        let camera2 = FovCamera::from(&params_slice[..]);
        assert_eq!(camera2.pinhole.fx, 450.0);
        assert_eq!(camera2.pinhole.fy, 460.0);
        assert_eq!(camera2.pinhole.cx, 330.0);
        assert_eq!(camera2.pinhole.cy, 250.0);
        assert_eq!(camera2.distortion_params(), 2.0);

        // Test conversion from array
        let camera3 = FovCamera::from([500.0, 510.0, 340.0, 260.0, 2.5]);
        assert_eq!(camera3.pinhole.fx, 500.0);
        assert_eq!(camera3.pinhole.fy, 510.0);
        assert_eq!(camera3.distortion_params(), 2.5);

        Ok(())
    }

    #[test]
    fn test_linear_estimation() -> TestResult {
        // Ground truth FOV camera
        let gt_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let gt_distortion = DistortionModel::FOV { w: 1.0 };
        let gt_camera = FovCamera::new(gt_pinhole, gt_distortion)?;

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

        // Initial camera with default w (grid search will find best)
        let init_pinhole = PinholeParams::new(300.0, 300.0, 320.0, 240.0)?;
        let init_distortion = DistortionModel::FOV { w: 0.5 };
        let mut camera = FovCamera::new(init_pinhole, init_distortion)?;

        camera.linear_estimation(&pts_3d, &pts_2d)?;

        // FOV uses grid search so tolerance is looser
        for i in 0..valid {
            let p3d = pts_3d.column(i).into_owned();
            let projected = camera.project(&Vector3::new(p3d.x, p3d.y, p3d.z))?;
            let err = ((projected.x - pts_2d[(0, i)]).powi(2)
                + (projected.y - pts_2d[(1, i)]).powi(2))
            .sqrt();
            assert!(err < 5.0, "Reprojection error too large: {err}");
        }

        Ok(())
    }
}
